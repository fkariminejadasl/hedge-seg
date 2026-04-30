from __future__ import annotations

import json
import math
import random
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from multiprocessing import Pool
from pathlib import Path
from typing import Optional

import cv2
import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import requests
from omegaconf import OmegaConf
from rasterio.enums import Resampling
from rasterio.transform import from_bounds, from_origin
from rasterio.warp import reproject
from rasterio.windows import Window
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

WMTS_CAPABILITIES_URL = (
    "https://service.pdok.nl/hwh/luchtfotorgb/wmts/v1_0"
    "?request=GetCapabilities&service=WMTS"
)

NS = {
    "wmts": "http://www.opengis.net/wmts/1.0",
    "ows": "http://www.opengis.net/ows/1.1",
}


@dataclass
class TileMatrix:
    identifier: str
    resolution: float
    top_left_x: float
    top_left_y: float
    tile_width: int
    tile_height: int
    matrix_width: int
    matrix_height: int


@dataclass
class WmtsInfo:
    template: str
    format: str
    tile_matrix_set: str
    matrices: list[TileMatrix]


def read_bbox_csv(csv_path: Optional[Path]):
    """
    CSV format:
    xmin,ymin,xmax,ymax
    180000,390000,210000,420000
    Coordinates must be EPSG:28992.
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


def fetch_wmts_info(layer_name: str, tile_matrix_set: str) -> WmtsInfo:
    r = requests.get(WMTS_CAPABILITIES_URL, timeout=120)
    r.raise_for_status()

    root = ET.fromstring(r.content)

    layer_el = None
    for candidate in root.findall(".//wmts:Layer", NS):
        ident = candidate.find("ows:Identifier", NS)
        if ident is not None and ident.text == layer_name:
            layer_el = candidate
            break
    if layer_el is None:
        raise ValueError(f"Layer not found in WMTS capabilities: {layer_name}")

    template = None
    img_format = None
    for resource in layer_el.findall("wmts:ResourceURL", NS):
        if resource.attrib.get("resourceType") == "tile":
            template = resource.attrib.get("template")
            img_format = resource.attrib.get("format")
            break
    if template is None:
        raise ValueError(f"No tile ResourceURL found for layer: {layer_name}")

    set_link_ok = False
    for set_link in layer_el.findall("wmts:TileMatrixSetLink", NS):
        set_name = set_link.find("wmts:TileMatrixSet", NS)
        if set_name is not None and set_name.text == tile_matrix_set:
            set_link_ok = True
            break
    if not set_link_ok:
        raise ValueError(
            f"Layer {layer_name} does not expose TileMatrixSet {tile_matrix_set}"
        )

    tms_el = None
    for candidate in root.findall(".//wmts:TileMatrixSet", NS):
        ident = candidate.find("ows:Identifier", NS)
        if ident is not None and ident.text == tile_matrix_set:
            tms_el = candidate
            break
    if tms_el is None:
        raise ValueError(f"TileMatrixSet not found: {tile_matrix_set}")

    matrices: list[TileMatrix] = []
    for matrix_el in tms_el.findall("wmts:TileMatrix", NS):
        ident = matrix_el.find("ows:Identifier", NS)
        scale = matrix_el.find("wmts:ScaleDenominator", NS)
        top_left = matrix_el.find("wmts:TopLeftCorner", NS)
        tile_width = matrix_el.find("wmts:TileWidth", NS)
        tile_height = matrix_el.find("wmts:TileHeight", NS)
        matrix_width = matrix_el.find("wmts:MatrixWidth", NS)
        matrix_height = matrix_el.find("wmts:MatrixHeight", NS)

        top_left_x, top_left_y = [float(v) for v in top_left.text.split()]
        resolution = float(scale.text) * 0.00028

        matrices.append(
            TileMatrix(
                identifier=ident.text,
                resolution=resolution,
                top_left_x=top_left_x,
                top_left_y=top_left_y,
                tile_width=int(tile_width.text),
                tile_height=int(tile_height.text),
                matrix_width=int(matrix_width.text),
                matrix_height=int(matrix_height.text),
            )
        )

    if not matrices:
        raise ValueError(f"No tile matrices found for set: {tile_matrix_set}")

    return WmtsInfo(
        template=template,
        format=img_format,
        tile_matrix_set=tile_matrix_set,
        matrices=matrices,
    )


def choose_matrix(matrices: list[TileMatrix], target_resolution_m: float) -> TileMatrix:
    """
    Prefer a matrix that is at least as fine as the target resolution.
    For 25 cm chips, that usually means a matrix slightly finer than 0.25 m/px,
    then we resample back to exactly 0.25 m/px in the local GeoTIFF.
    """
    finer_or_equal = [m for m in matrices if m.resolution <= target_resolution_m]
    if finer_or_equal:
        return min(
            finer_or_equal, key=lambda m: abs(m.resolution - target_resolution_m)
        )
    return min(matrices, key=lambda m: abs(m.resolution - target_resolution_m))


def tile_range_for_bbox(bbox_geom, matrix: TileMatrix):
    minx, miny, maxx, maxy = bbox_geom.bounds
    res = matrix.resolution
    tile_span_x = matrix.tile_width * res
    tile_span_y = matrix.tile_height * res
    eps = 1e-9

    col_min = math.floor((minx - matrix.top_left_x) / tile_span_x)
    col_max = math.floor((maxx - eps - matrix.top_left_x) / tile_span_x)
    row_min = math.floor((matrix.top_left_y - maxy) / tile_span_y)
    row_max = math.floor((matrix.top_left_y - (miny + eps)) / tile_span_y)

    col_min = max(0, col_min)
    row_min = max(0, row_min)
    col_max = min(matrix.matrix_width - 1, col_max)
    row_max = min(matrix.matrix_height - 1, row_max)

    return row_min, row_max, col_min, col_max


def tile_url(
    template: str, tile_matrix_set: str, matrix_id: str, row: int, col: int
) -> str:
    return (
        template.replace("{TileMatrixSet}", tile_matrix_set)
        .replace("{TileMatrix}", str(matrix_id))
        .replace("{TileRow}", str(row))
        .replace("{TileCol}", str(col))
    )


def download_one_tile(url: str, out_path: Path, timeout: int):
    if out_path.exists():
        return out_path

    out_path.parent.mkdir(parents=True, exist_ok=True)
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    out_path.write_bytes(r.content)
    return out_path


def ensure_tiles_cached(
    *,
    wmts_info: WmtsInfo,
    matrix: TileMatrix,
    bbox_geom,
    cache_dir: Path,
    timeout: int,
    num_download_workers: int,
):
    row_min, row_max, col_min, col_max = tile_range_for_bbox(bbox_geom, matrix)

    jobs = []
    for row in range(row_min, row_max + 1):
        for col in range(col_min, col_max + 1):
            url = tile_url(
                wmts_info.template,
                wmts_info.tile_matrix_set,
                matrix.identifier,
                row,
                col,
            )
            out_path = (
                cache_dir
                / "tiles"
                / wmts_info.tile_matrix_set.replace(":", "_")
                / matrix.identifier
                / str(row)
                / f"{col}.jpg"
            )
            jobs.append((url, out_path, timeout))

    if num_download_workers <= 1:
        for job in jobs:
            download_one_tile(*job)
    else:
        with ThreadPoolExecutor(max_workers=num_download_workers) as ex:
            list(ex.map(lambda args: download_one_tile(*args), jobs))

    return row_min, row_max, col_min, col_max


def read_cached_tile(tile_path: Path) -> np.ndarray:
    img = cv2.imread(str(tile_path), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Could not read tile: {tile_path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def build_local_geotiff_from_wmts(cfg, bbox_geom) -> Path:
    cache_dir = Path(cfg.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    wmts_info = fetch_wmts_info(cfg.layer_name, cfg.tile_matrix_set)
    matrix = choose_matrix(wmts_info.matrices, cfg.target_resolution_m)

    row_min, row_max, col_min, col_max = ensure_tiles_cached(
        wmts_info=wmts_info,
        matrix=matrix,
        bbox_geom=bbox_geom,
        cache_dir=cache_dir,
        timeout=cfg.timeout,
        num_download_workers=cfg.num_download_workers,
    )

    n_tile_rows = row_max - row_min + 1
    n_tile_cols = col_max - col_min + 1
    mosaic_h = n_tile_rows * matrix.tile_height
    mosaic_w = n_tile_cols * matrix.tile_width
    mosaic = np.zeros((3, mosaic_h, mosaic_w), dtype=np.uint8)

    tile_root = (
        cache_dir
        / "tiles"
        / wmts_info.tile_matrix_set.replace(":", "_")
        / matrix.identifier
    )

    for row in range(row_min, row_max + 1):
        for col in range(col_min, col_max + 1):
            tile_path = tile_root / str(row) / f"{col}.jpg"
            tile = read_cached_tile(tile_path)
            r0 = (row - row_min) * matrix.tile_height
            c0 = (col - col_min) * matrix.tile_width
            mosaic[:, r0 : r0 + matrix.tile_height, c0 : c0 + matrix.tile_width] = (
                np.transpose(tile, (2, 0, 1))
            )

    src_minx = matrix.top_left_x + col_min * matrix.tile_width * matrix.resolution
    src_maxy = matrix.top_left_y - row_min * matrix.tile_height * matrix.resolution
    src_transform = from_origin(
        src_minx, src_maxy, matrix.resolution, matrix.resolution
    )

    minx, miny, maxx, maxy = bbox_geom.bounds
    dst_width = int(round((maxx - minx) / cfg.target_resolution_m))
    dst_height = int(round((maxy - miny) / cfg.target_resolution_m))
    dst_transform = from_bounds(minx, miny, maxx, maxy, dst_width, dst_height)
    dst = np.zeros((3, dst_height, dst_width), dtype=np.uint8)

    for b in range(3):
        reproject(
            source=mosaic[b],
            destination=dst[b],
            src_transform=src_transform,
            src_crs=cfg.crs,
            dst_transform=dst_transform,
            dst_crs=cfg.crs,
            resampling=Resampling.bilinear,
        )

    tif_path = cache_dir / f"{cfg.layer_name}_bbox_cache.tif"
    with rasterio.open(
        tif_path,
        "w",
        driver="GTiff",
        width=dst_width,
        height=dst_height,
        count=3,
        dtype=np.uint8,
        crs=cfg.crs,
        transform=dst_transform,
        compress="deflate",
        tiled=True,
        blockxsize=256,
        blockysize=256,
    ) as dst_ds:
        dst_ds.write(dst)

    print(
        f"Built local GeoTIFF cache: {tif_path}\n"
        f"matrix={matrix.identifier}, matrix_resolution={matrix.resolution:.6f} m/px, "
        f"local_resolution={cfg.target_resolution_m:.6f} m/px"
    )
    return tif_path


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


def process_job_chunk(job_chunk, cfg_dict):
    cfg = OmegaConf.create(cfg_dict)
    out_dir = Path(cfg.out_dir)
    paths = ensure_dirs(out_dir)

    area_geom = read_bbox_csv(Path(cfg.bbox_csv)) if cfg.bbox_csv else None

    gdf = gpd.read_file(cfg.shp_path)
    if gdf.crs is None:
        raise ValueError("Shapefile CRS is missing.")
    if str(gdf.crs) != cfg.crs:
        gdf = gdf.to_crs(cfg.crs)
    gdf = gdf[~gdf.geometry.is_empty & gdf.geometry.notna()].copy()
    gdf = gdf[gdf.geometry.geom_type.isin(["LineString", "MultiLineString"])].copy()
    gdf = gdf.reset_index(drop=True)

    if area_geom is not None:
        gdf = gdf[gdf.intersects(area_geom)].copy()
        gdf = gdf.reset_index(drop=True)

    sindex = gdf.sindex

    with rasterio.open(cfg.local_tif_path) as src:
        for sample_id, center_x, center_y in job_chunk:
            half = cfg.chip_size_m / 2.0
            bbox = (center_x - half, center_y - half, center_x + half, center_y + half)
            bbox_geom = box(*bbox)

            row0, col0 = src.index(bbox[0], bbox[3])
            row1, col1 = src.index(bbox[2], bbox[1])
            width = col1 - col0
            height = row1 - row0
            if width <= 0 or height <= 0:
                continue

            win = Window(col_off=col0, row_off=row0, width=width, height=height)
            data = src.read(window=win)
            if data.shape[0] != 3:
                raise ValueError(
                    f"Expected 3-band RGB GeoTIFF, got {data.shape[0]} band(s)."
                )

            img = np.transpose(data, (1, 2, 0))
            if img.shape[0] != cfg.out_size_px or img.shape[1] != cfg.out_size_px:
                img = cv2.resize(
                    img,
                    (cfg.out_size_px, cfg.out_size_px),
                    interpolation=cv2.INTER_CUBIC,
                )

            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            hit = lines_in_bbox(gdf, sindex, bbox_geom)
            if hit.empty:
                continue

            polylines = make_polylines_for_chip(
                hit,
                bbox_geom,
                out_size_px=cfg.out_size_px,
                min_len_px=cfg.min_len_px,
            )
            if not polylines:
                continue

            label_obj = {
                "id": sample_id,
                "type": "positive",
                "layer": cfg.layer_name,
                "chip_size_px": cfg.out_size_px,
                "chip_size_m": cfg.chip_size_m,
                "pixel_size_m": cfg.chip_size_m / cfg.out_size_px,
                "bbox_world": list(bbox),
                "crs": cfg.crs,
                "center_world": [center_x, center_y],
                "n_lines": len(polylines),
                "polylines_px": polylines,
            }

            save_chip(paths, sample_id, img_bgr, label_obj, mask=None)

    return len(job_chunk)


def split_jobs(jobs, num_chunks: int):
    chunks = [[] for _ in range(num_chunks)]
    for i, job in enumerate(jobs):
        chunks[i % num_chunks].append(job)
    return [chunk for chunk in chunks if chunk]


def build_dataset(cfg):
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    area_geom = read_bbox_csv(Path(cfg.bbox_csv)) if cfg.bbox_csv else None
    if area_geom is None:
        raise ValueError(
            "This WMTS-cache workflow expects bbox_csv so it knows what area to cache locally."
        )

    local_tif_path = build_local_geotiff_from_wmts(cfg, area_geom)

    gdf = gpd.read_file(cfg.shp_path)
    if gdf.crs is None:
        raise ValueError("Shapefile CRS is missing.")
    if str(gdf.crs) != cfg.crs:
        gdf = gdf.to_crs(cfg.crs)
    gdf = gdf[~gdf.geometry.is_empty & gdf.geometry.notna()].copy()
    gdf = gdf[gdf.geometry.geom_type.isin(["LineString", "MultiLineString"])].copy()
    gdf = gdf.reset_index(drop=True)

    centers = sample_positive_centers(
        gdf=gdf,
        n_pos=cfg.n_pos,
        chip_size_m=cfg.chip_size_m,
        rng_seed=cfg.seed,
        area_geom=area_geom,
    )

    jobs = []
    for i, (cx, cy) in enumerate(centers):
        jobs.append((f"pos_{i:06d}", cx, cy))

    cfg_for_workers = OmegaConf.to_container(cfg, resolve=True)
    cfg_for_workers["local_tif_path"] = str(local_tif_path)

    if cfg.num_workers == 1:
        process_job_chunk(jobs, cfg_for_workers)
    else:
        chunks = split_jobs(jobs, cfg.num_workers)
        with Pool(cfg.num_workers) as pool:
            pool.starmap(
                process_job_chunk,
                [(chunk, cfg_for_workers) for chunk in chunks],
            )

    print(f"Done. Saved dataset in: {out_dir}")


def main():
    cfg = dict(
        shp_path=Path(
            "/home/fatemeh/Downloads/hedge/Topo10NL2023/Hedges_polylines/Top10NL2023_inrichtingselementen_lijn_heg.shp"
        ),
        bbox_csv=Path("/home/fatemeh/Downloads/hedge/area_bbox.csv"),
        out_dir=Path("/home/fatemeh/Downloads/hedge/results/pdok_dataset_wmts_cache"),
        cache_dir=Path(
            "/home/fatemeh/Downloads/hedge/results/pdok_wmts_training_data_cache"
        ),
        n_pos=17000,
        num_workers=8,
        num_download_workers=8,
        seed=123,
        crs="EPSG:28992",
        layer_name="Actueel_ortho25",  # or "2018_ortho25"
        tile_matrix_set="EPSG:28992",
        target_resolution_m=0.25,
        chip_size_m=250.0,
        out_size_px=1000,
        timeout=120,
        min_len_px=10.0,
    )
    cfg = OmegaConf.create(cfg)
    wmts_info = fetch_wmts_info(cfg.layer_name, cfg.tile_matrix_set)
    # build_dataset(cfg)


if __name__ == "__main__":
    main()
