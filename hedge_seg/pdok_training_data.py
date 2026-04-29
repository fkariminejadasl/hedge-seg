import random
import time
from datetime import datetime
from io import BytesIO
from multiprocessing import Pool
from pathlib import Path
from typing import Optional

import geopandas as gpd
import numpy as np
import pandas as pd
import requests
from PIL import Image
from rasterio.transform import from_bounds
from shapely.geometry import box

from hedge_seg.training_data import (
    clip_px_point,
    dedupe_consecutive_points,
    ensure_dirs,
    geom_to_lines,
    lines_in_bbox,
    polyline_length_px,
    save_chip,
    world_to_pixel_in_window,
)

PDOK_WMS = "https://service.pdok.nl/hwh/luchtfotorgb/wms/v1_0"

_WORKER_GDF = None
_WORKER_SINDEX = None


def prepare_hedge_gdf(shp_path, crs):
    gdf = gpd.read_file(shp_path)

    if gdf.crs is None:
        raise ValueError("Shapefile CRS is missing.")

    if str(gdf.crs) != crs:
        gdf = gdf.to_crs(crs)

    gdf = gdf[~gdf.geometry.is_empty & gdf.geometry.notna()].copy()
    gdf = gdf[gdf.geometry.geom_type.isin(["LineString", "MultiLineString"])].copy()
    gdf = gdf.reset_index(drop=True)

    return gdf


def init_worker(gdf):
    global _WORKER_GDF, _WORKER_SINDEX

    _WORKER_GDF = gdf
    _WORKER_SINDEX = gdf.sindex


def read_bbox_csv(csv_path: Optional[Path]):
    """
    CSV format:
    xmin,ymin,xmax,ymax
    180000,390000,210000,420000
    All coordinates must be in EPSG:28992.
    """
    if csv_path is None:
        return None

    df = pd.read_csv(csv_path)
    required = {"xmin", "ymin", "xmax", "ymax"}
    if not required.issubset(df.columns):
        raise ValueError(f"CSV must contain columns: {sorted(required)}")
    if len(df) != 1:
        raise ValueError("BBox CSV must contain exactly one row.")

    row = df.iloc[0]
    return box(float(row.xmin), float(row.ymin), float(row.xmax), float(row.ymax))


def sample_point_on_linestring(line, rng: random.Random):
    if line.length <= 0:
        p = line.centroid
        return float(p.x), float(p.y)
    d = rng.random() * float(line.length)
    p = line.interpolate(d)
    return float(p.x), float(p.y)


def sample_positive_centers(
    gdf, n_pos: int, chip_size_m: float, rng_seed: int, area_geom=None
):
    """
    Sample random chip centers from hedge polylines.
    If area_geom is given, only chips fully inside that bbox are kept.
    """
    rng = random.Random(rng_seed)
    centers = []
    half = chip_size_m / 2.0

    if area_geom is not None:
        gdf = gdf[gdf.intersects(area_geom)].copy()
        if gdf.empty:
            raise ValueError("No hedge lines intersect the bbox area.")

    geoms = list(gdf.geometry)
    max_tries = max(10 * n_pos, 1000)
    tries = 0

    while len(centers) < n_pos and tries < max_tries:
        tries += 1
        geom = geoms[rng.randrange(len(geoms))]
        if geom.geom_type == "MultiLineString":
            parts = list(geom.geoms)
            geom = parts[rng.randrange(len(parts))]

        cx, cy = sample_point_on_linestring(geom, rng)
        chip_geom = box(cx - half, cy - half, cx + half, cy + half)

        if area_geom is not None and not area_geom.contains(chip_geom):
            continue

        centers.append((cx, cy))

    if len(centers) < n_pos:
        raise RuntimeError(
            f"Could only sample {len(centers)} positive centers out of {n_pos}."
        )

    return centers


def fetch_pdok_chip(layer_name, bbox, out_size_px, image_format, timeout):
    params = {
        "service": "WMS",
        "version": "1.3.0",
        "request": "GetMap",
        "layers": layer_name,
        "styles": "",
        "crs": "EPSG:28992",
        "bbox": ",".join(map(str, bbox)),
        "width": out_size_px,
        "height": out_size_px,
        "format": image_format,
        "transparent": "false",
    }

    last_error = None

    for attempt in range(3):
        try:
            r = requests.get(PDOK_WMS, params=params, timeout=timeout)
            r.raise_for_status()

            img = Image.open(BytesIO(r.content)).convert("RGB")
            return np.array(img)

        except requests.exceptions.RequestException as e:
            last_error = e

            if attempt < 2:
                time.sleep(2 * (attempt + 1))

    raise last_error


def make_polylines_for_chip(lines_gdf, bbox_geom, out_size_px, min_len_px=10.0):
    transform = from_bounds(*bbox_geom.bounds, out_size_px, out_size_px)
    polylines = []

    for geom in lines_gdf.geometry:
        inter = geom.intersection(bbox_geom)
        for ls in geom_to_lines(inter):
            xs, ys = ls.xy
            pts = []
            for x, y in zip(xs, ys):
                px, py = world_to_pixel_in_window(transform, x, y)
                pts.append(clip_px_point(px, py, out_size_px))

            pts = dedupe_consecutive_points(pts)
            if len(pts) < 2:
                continue
            if polyline_length_px(pts) < min_len_px:
                continue
            polylines.append(pts)

    return polylines


def build_one_sample(job):
    global _WORKER_GDF, _WORKER_SINDEX

    if _WORKER_GDF is None or _WORKER_SINDEX is None:
        raise RuntimeError(
            "Worker shapefile data is not initialized. "
            "Call init_worker(gdf) before build_one_sample()."
        )

    sample_id, center_x, center_y, cfg = job

    chip_size_m = cfg.chip_size_m
    out_size_px = cfg.out_size_px
    half = chip_size_m / 2.0

    bbox = (center_x - half, center_y - half, center_x + half, center_y + half)
    bbox_geom = box(*bbox)

    hit = lines_in_bbox(_WORKER_GDF, _WORKER_SINDEX, bbox_geom)

    if hit.empty:
        return None

    polylines = make_polylines_for_chip(
        hit,
        bbox_geom,
        out_size_px=out_size_px,
        min_len_px=cfg.min_len_px,
    )

    if not polylines:
        return None

    try:
        img = fetch_pdok_chip(
            layer_name=cfg.layer_name,
            bbox=bbox,
            out_size_px=out_size_px,
            image_format=cfg.image_format,
            timeout=cfg.timeout,
        )
    except requests.exceptions.RequestException as e:
        print(f"Skipping {sample_id}: PDOK request failed: {e}")
        return None

    label_obj = {
        "id": sample_id,
        "type": "positive",
        "layer": cfg.layer_name,
        "chip_size_px": out_size_px,
        "chip_size_m": chip_size_m,
        "pixel_size_m": chip_size_m / out_size_px,
        "bbox_world": list(bbox),
        "crs": cfg.crs,
        "center_world": [center_x, center_y],
        "n_lines": len(polylines),
        "polylines_px": polylines,
    }

    return sample_id, img, label_obj


def build_dataset(cfg):
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    paths = ensure_dirs(out_dir)

    gdf = prepare_hedge_gdf(cfg.shp_path, cfg.crs)

    area_geom = read_bbox_csv(Path(cfg.bbox_csv)) if cfg.bbox_csv else None

    # Sample more candidates than needed, because some may be rejected later due to various reasons (e.g. PDOK request failure, no lines in bbox, etc.)
    target_n = cfg.n_pos
    candidate_n = int(cfg.n_pos * 1.2)
    centers = sample_positive_centers(
        gdf=gdf,
        n_pos=candidate_n,
        chip_size_m=cfg.chip_size_m,
        rng_seed=cfg.seed,
        area_geom=area_geom,
    )
    print(
        f"{datetime.now().replace(microsecond=0)}: Sampled {len(centers)} positive centers"
    )

    jobs = []
    for i, (cx, cy) in enumerate(centers):
        sample_id = f"pos_{i:06d}"
        jobs.append((sample_id, cx, cy, cfg))

    saved = 0

    if cfg.num_workers == 1:
        init_worker(gdf)

        iterator = map(build_one_sample, jobs)

        for result in iterator:
            if result is None:
                continue

            sample_id, img, label_obj = result
            save_chip(paths, sample_id, img, label_obj, mask=None)

            saved += 1
            if saved % 250 == 0:
                print(
                    f"{datetime.now().replace(microsecond=0)}: saved {saved}/{target_n}"
                )
            if saved >= target_n:
                break

    else:
        with Pool(
            processes=cfg.num_workers,
            initializer=init_worker,
            initargs=(gdf,),
        ) as pool:
            for result in pool.imap_unordered(
                build_one_sample,
                jobs,
                chunksize=1,
            ):
                if result is None:
                    continue

                sample_id, img, label_obj = result
                save_chip(paths, sample_id, img, label_obj, mask=None)

                saved += 1
                if saved % 250 == 0:
                    print(
                        f"{datetime.now().replace(microsecond=0)}: saved {saved}/{target_n}"
                    )
                if saved >= target_n:
                    break

    print(f"Done. Saved {saved} positive samples in {out_dir}")


# def _build_one_sample_starmap(job):
#     return build_one_sample(job)
