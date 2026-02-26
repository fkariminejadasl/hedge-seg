"""
Training data generation.
  - Create image chips from source rasters
  - Create initial labels for each chip (polylines and optionally masks / maps)
  - Write images and raw label files to disk in a consistent directory structure
"""

import json
import math
import multiprocessing as mp
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import contextily as ctx
import cv2
import geopandas as gpd
import numpy as np
import rasterio
from affine import Affine
from pyproj import Transformer
from rasterio.features import rasterize
from rasterio.warp import Resampling, reproject
from rasterio.windows import Window
from shapely.geometry import box
from shapely.geometry.base import BaseGeometry

# ----------------------------
# Utils
# ----------------------------


def float_to_uint8(
    data: np.ndarray,
    vmin: float | None = None,
    vmax: float | None = None,
    pmin: float = 2.0,
    pmax: float = 98.0,
) -> np.ndarray:
    """
    Convert float raster to uint8 via clipping and normalization.
    Uses per-chip percentiles by default.
    """
    data = np.nan_to_num(data, nan=0.0)

    if vmin is None:
        vmin = float(np.percentile(data, pmin))
    if vmax is None:
        vmax = float(np.percentile(data, pmax))

    if vmax <= vmin:
        return np.zeros_like(data, dtype=np.uint8)

    data_clipped = np.clip(data, vmin, vmax)
    norm = (data_clipped - vmin) / (vmax - vmin + 1e-12)
    return (norm * 255).astype(np.uint8)


def geom_to_lines(geom: BaseGeometry) -> List[BaseGeometry]:
    """
    Extract LineString parts from an intersection result.
    """
    if geom.is_empty:
        return []
    gtype = geom.geom_type
    if gtype == "LineString":
        return [geom]
    if gtype == "MultiLineString":
        return list(geom.geoms)
    if gtype == "GeometryCollection":
        out = []
        for g in geom.geoms:
            out.extend(geom_to_lines(g))
        return out
    return []


def window_from_center(row: int, col: int, size: int) -> Window:
    half = size // 2
    return Window(col_off=col - half, row_off=row - half, width=size, height=size)


def window_fits(src: rasterio.io.DatasetReader, win: Window) -> bool:
    return (
        win.col_off >= 0
        and win.row_off >= 0
        and (win.col_off + win.width) <= src.width
        and (win.row_off + win.height) <= src.height
    )


def window_bounds(
    src: rasterio.io.DatasetReader, win: Window
) -> Tuple[float, float, float, float]:
    # rasterio.windows.bounds expects window and transform
    from rasterio.windows import bounds as win_bounds

    return win_bounds(win, transform=src.transform)


def world_to_pixel_in_window(w_transform, x: float, y: float) -> Tuple[float, float]:
    """
    Convert world x,y to pixel x,y in the window coordinate system.
    Returns (px, py) where px is column-like and py is row-like.
    """
    inv = ~w_transform
    px, py = inv * (x, y)
    return float(px), float(py)


def read_chip(
    src: rasterio.io.DatasetReader,
    win: Window,
    band: int = 1,
    max_nodata_frac: float = 0.98,
) -> Tuple[np.ndarray, rasterio.Affine] | Tuple[None, None]:
    """
    Read a single-band chip window. Returns (uint8 image, window_transform) or (None, None) if rejected.
    """
    data = src.read(band, window=win, masked=True)
    arr = data.filled(np.nan).astype(np.float32)

    nodata_frac = float(np.isnan(arr).mean())
    if nodata_frac > max_nodata_frac:
        return None, None

    img8 = float_to_uint8(arr)
    return img8, src.window_transform(win)


def ensure_dirs(out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "images": out_dir / "images",
        "labels": out_dir / "labels",
        "masks": out_dir / "masks",
        "osm": out_dir / "osm",
    }
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)
    return paths


# ----------------------------
# Label generation
# ----------------------------


def lines_in_bbox(
    gdf: gpd.GeoDataFrame,
    sindex,
    bbox_geom: BaseGeometry,
) -> gpd.GeoDataFrame:
    """
    Fast candidate query via spatial index, then exact intersects.
    """
    cand_idx = list(sindex.intersection(bbox_geom.bounds))
    if not cand_idx:
        return gdf.iloc[0:0]
    cand = gdf.iloc[cand_idx]
    hit = cand[cand.intersects(bbox_geom)]
    return hit


def make_polyline_labels(
    lines_gdf: gpd.GeoDataFrame,
    bbox_geom: BaseGeometry,
    w_transform,
) -> List[List[List[int]]]:
    """
    Returns list of polylines; each polyline is list of [x,y] integer pixel coords in the chip.
    """
    out: List[List[List[int]]] = []
    for geom in lines_gdf.geometry:
        inter = geom.intersection(bbox_geom)
        parts = geom_to_lines(inter)
        for ls in parts:
            xs, ys = ls.xy
            pts = []
            for x, y in zip(xs, ys):
                px, py = world_to_pixel_in_window(w_transform, x, y)
                # keep points that are inside the chip bounds
                pts.append([int(round(px)), int(round(py))])
            if len(pts) >= 2:
                out.append(pts)
    return out


def make_mask_label(
    lines_gdf: gpd.GeoDataFrame,
    bbox_geom: BaseGeometry,
    w_transform,
    out_shape: Tuple[int, int],
    line_width_px: int = 2,
    res_m: Optional[Tuple[float, float]] = None,
) -> np.ndarray:
    """
    Rasterizes buffered lines into a binary mask.
    """
    if lines_gdf.empty:
        return np.zeros(out_shape, dtype=np.uint8)

    if res_m is None:
        # fall back to 1.0 meters if unknown, but in practice we pass src.res
        res_m = (1.0, 1.0)

    # buffer distance in world units (meters) so mask has roughly line_width_px thickness
    # use max(res) so thickness is not too thin
    buf_dist = (line_width_px * max(res_m)) / 2.0

    shapes = []
    for geom in lines_gdf.geometry:
        inter = geom.intersection(bbox_geom)
        parts = geom_to_lines(inter)
        for ls in parts:
            if ls.is_empty:
                continue
            poly = ls.buffer(buf_dist, cap_style=2, join_style=2)
            if not poly.is_empty:
                shapes.append((poly, 1))

    if not shapes:
        return np.zeros(out_shape, dtype=np.uint8)

    mask = rasterize(
        shapes=shapes,
        out_shape=out_shape,
        transform=w_transform,
        fill=0,
        all_touched=True,
        dtype=np.uint8,
    )
    return (mask * 255).astype(np.uint8)


# ----------------------------
# Sampling
# ----------------------------


def random_point_on_line(line) -> Tuple[float, float]:
    """
    Uniform by arc length (approx) using shapely interpolate.
    """
    if line.length <= 0:
        c = line.centroid
        return float(c.x), float(c.y)
    d = random.random() * float(line.length)
    p = line.interpolate(d)
    return float(p.x), float(p.y)


@dataclass
class ChipSpec:
    size_px: int = 256
    band: int = 1
    max_nodata_frac: float = 0.98
    label_mode: str = "polylines"  # "polylines", "mask", "both"
    line_width_px: int = 2


def generate_dataset(
    shp_path: Path,
    tif_path: Path,
    out_dir: Path,
    *,
    n_pos: int = 2000,
    n_neg: int = 0,
    seed: int = 0,
    max_tries_per_sample: int = 200,
    use_osm: bool = False,
    # Chip parameters
    size_px: int = 256,
    label_mode: str = "polylines",  # "polylines", "mask", "both"
    band: int = 1,
    line_width_px: int = 2,
):
    chip = ChipSpec(
        size_px=size_px,
        label_mode=label_mode,
        band=band,
        line_width_px=line_width_px,
    )

    random.seed(seed)
    np.random.seed(seed)

    paths = ensure_dirs(out_dir)

    gdf = gpd.read_file(shp_path)
    if gdf.crs is None:
        raise ValueError("Shapefile CRS is missing.")

    # Keep only line geometries
    gdf = gdf[~gdf.geometry.is_empty & gdf.geometry.notna()].copy()
    gdf = gdf[gdf.geometry.geom_type.isin(["LineString", "MultiLineString"])].copy()
    gdf = gdf.reset_index(drop=True)

    with rasterio.open(tif_path) as src:
        if src.crs is None:
            raise ValueError("GeoTIFF CRS is missing.")

        if gdf.crs != src.crs:
            gdf = gdf.to_crs(src.crs)

        to_3857 = Transformer.from_crs(src.crs, "EPSG:3857", always_xy=True)
        to_wgs84 = Transformer.from_crs(src.crs, "EPSG:4326", always_xy=True)

        # Build spatial index once
        sindex = gdf.sindex

        size = chip.size_px
        half = size // 2

        def save_one(
            sample_id: str,
            img8: np.ndarray,
            label_obj: dict,
            mask: Optional[np.ndarray],
            w_transform,
        ):
            img_path = paths["images"] / f"{sample_id}.png"
            cv2.imwrite(str(img_path), img8)

            lab_path = paths["labels"] / f"{sample_id}.json"
            with open(lab_path, "w", encoding="utf-8") as f:
                json.dump(label_obj, f, ensure_ascii=False)

            if mask is not None:
                mask_path = paths["masks"] / f"{sample_id}.png"
                cv2.imwrite(str(mask_path), mask)

            if use_osm:
                save_osm_chip(
                    sample_id, label_obj["bbox_world"], size, w_transform, src.crs
                )

        def save_osm_chip(
            sample_id: str, bbox_world, size_px: int, dst_transform, dst_crs
        ):
            minx, miny, maxx, maxy = bbox_world

            # Request tiles in EPSG:3857
            minx_m, miny_m = to_3857.transform(minx, miny)
            maxx_m, maxy_m = to_3857.transform(maxx, maxy)

            # Estimate zoom so OSM pixel size roughly matches your chip size
            cx, cy = (minx + maxx) * 0.5, (miny + maxy) * 0.5
            lon, lat = to_wgs84.transform(cx, cy)
            lat_rad = math.radians(lat)

            meters_per_px = (maxx_m - minx_m) / float(size_px)
            meters_per_px = max(meters_per_px, 1e-6)
            z = math.log2((156543.03392804097 * math.cos(lat_rad)) / meters_per_px)
            zoom = int(np.clip(int(round(z)), 0, 19))

            img, extent = ctx.bounds2img(
                minx_m,
                miny_m,
                maxx_m,
                maxy_m,
                zoom=zoom,
                source=ctx.providers.OpenStreetMap.Mapnik,
                ll=False,
            )

            if img.ndim == 3 and img.shape[2] == 4:
                img = img[:, :, :3]

            # Source georeferencing for the returned mosaic
            xmin_e, xmax_e, ymin_e, ymax_e = extent
            H, W = img.shape[0], img.shape[1]
            src_transform = Affine(
                (xmax_e - xmin_e) / W, 0.0, xmin_e, 0.0, -(ymax_e - ymin_e) / H, ymax_e
            )

            # Reproject OSM (EPSG:3857) onto your chip grid (dst_crs, dst_transform)
            dst = np.zeros((size_px, size_px, 3), dtype=np.uint8)
            for b in range(3):
                reproject(
                    source=img[:, :, b],
                    destination=dst[:, :, b],
                    src_transform=src_transform,
                    src_crs="EPSG:3857",
                    dst_transform=dst_transform,
                    dst_crs=dst_crs,
                    resampling=Resampling.nearest,  # use nearest for strict alignment check
                )

            # contextily returns RGB; convert to BGR for cv2
            img_bgr = cv2.cvtColor(dst, cv2.COLOR_RGB2BGR)

            osm_path = paths["osm"] / f"{sample_id}.png"
            cv2.imwrite(str(osm_path), img_bgr)

        # ----------------------------
        # POSITIVES
        # ----------------------------
        pos_done = 0
        pos_tries = 0

        while pos_done < n_pos and pos_tries < n_pos * max_tries_per_sample:
            pos_tries += 1

            # pick random hedge
            idx = random.randrange(len(gdf))
            geom = gdf.geometry.iloc[idx]
            if geom.geom_type == "MultiLineString":
                # pick a part
                geom = random.choice(list(geom.geoms))

            x, y = random_point_on_line(geom)
            row, col = src.index(x, y)
            win = window_from_center(row, col, size)

            if not window_fits(src, win):
                continue

            img8, w_transform = read_chip(
                src, win, band=chip.band, max_nodata_frac=chip.max_nodata_frac
            )
            if img8 is None:
                continue

            minx, miny, maxx, maxy = window_bounds(src, win)
            bbox_geom = box(minx, miny, maxx, maxy)

            hit = lines_in_bbox(gdf, sindex, bbox_geom)
            if hit.empty:
                # very rare if center is on a line, but can happen if geometry is tiny or indexing odd
                continue

            polylines = []
            mask = None

            if chip.label_mode in ("polylines", "both"):
                polylines = make_polyline_labels(hit, bbox_geom, w_transform)

            if chip.label_mode in ("mask", "both"):
                mask = make_mask_label(
                    hit,
                    bbox_geom,
                    w_transform,
                    out_shape=(size, size),
                    line_width_px=chip.line_width_px,
                    res_m=src.res,
                )

            sample_id = f"pos_{pos_done:06d}"
            label_obj = {
                "id": sample_id,
                "type": "positive",
                "chip_size_px": size,
                "raster_band": chip.band,
                "bbox_world": [minx, miny, maxx, maxy],
                "n_lines": len(hit),
                "polylines_px": polylines,  # list of polylines, each polyline is list of [x,y]
            }
            save_one(sample_id, img8, label_obj, mask, w_transform)

            pos_done += 1

        print(f"Saved positives: {pos_done}/{n_pos} (tries={pos_tries})")

        # ----------------------------
        # NEGATIVES
        # ----------------------------
        neg_done = 0
        neg_tries = 0

        # sample centers directly in pixel space, so the window always fits
        while neg_done < n_neg and neg_tries < n_neg * max_tries_per_sample * 5:
            neg_tries += 1

            row = random.randint(half, src.height - half - 1)
            col = random.randint(half, src.width - half - 1)
            win = window_from_center(row, col, size)

            img8, w_transform = read_chip(
                src, win, band=chip.band, max_nodata_frac=chip.max_nodata_frac
            )
            if img8 is None:
                continue

            minx, miny, maxx, maxy = window_bounds(src, win)
            bbox_geom = box(minx, miny, maxx, maxy)

            hit = lines_in_bbox(gdf, sindex, bbox_geom)
            if not hit.empty:
                continue

            polylines = []
            mask = None
            if chip.label_mode in ("mask", "both"):
                mask = np.zeros((size, size), dtype=np.uint8)

            sample_id = f"neg_{neg_done:06d}"
            label_obj = {
                "id": sample_id,
                "type": "negative",
                "chip_size_px": size,
                "raster_band": chip.band,
                "bbox_world": [minx, miny, maxx, maxy],
                "n_lines": 0,
                "polylines_px": polylines,
            }
            save_one(sample_id, img8, label_obj, mask, w_transform)

            neg_done += 1

        print(f"Saved negatives: {neg_done}/{n_neg} (tries={neg_tries})")


def gererate_dataset_worker(
    part_id: int,
    shp_path: Path,
    tif_path: Path,
    out_root: Path,
    n_pos_part: int,
    n_neg_part: int,
    size_px: int,
    base_seed: int,
    label_mode: str,
    max_tries: int,
    use_osm: bool,
):
    # Separate output per process to avoid filename collisions
    out_dir = out_root / f"part_{part_id:02d}"

    # Different seed per process (still reproducible overall)
    seed = base_seed + part_id

    generate_dataset(
        shp_path=shp_path,
        tif_path=tif_path,
        out_dir=out_dir,
        n_pos=n_pos_part,
        n_neg=n_neg_part,
        size_px=size_px,
        seed=seed,
        max_tries_per_sample=max_tries,
        label_mode=label_mode,
        use_osm=use_osm,
    )


def generate_dataset_mp(
    n_pos_total,
    n_neg_total,
    n_proc,
    shp_path,
    tif_path,
    out_dir,
    size_px,
    seed,
    max_tries,
    label_mode,
    use_osm,
):
    """
    Multiprocess wrapper around generate_dataset. Splits the total number of positives/negatives across processes
    """
    # Split totals across processes. This distributes the remainder nicely
    pos_counts = [n_pos_total // n_proc] * n_proc
    for i in range(n_pos_total % n_proc):
        pos_counts[i] += 1

    neg_counts = [n_neg_total // n_proc] * n_proc
    for i in range(n_neg_total % n_proc):
        neg_counts[i] += 1

    tasks = []
    for part_id in range(n_proc):
        if pos_counts[part_id] == 0 and neg_counts[part_id] == 0:
            continue
        tasks.append(
            (
                part_id,
                shp_path,
                tif_path,
                out_dir,
                pos_counts[part_id],
                neg_counts[part_id],
                size_px,
                seed,
                label_mode,
                max_tries,
                use_osm,
            )
        )

    with mp.Pool(processes=n_proc) as pool:
        _ = pool.starmap(gererate_dataset_worker, tasks)


def merge_parts(out_root: Path) -> Path:
    cats = ["images", "labels", "masks", "osm"]
    for c in cats:
        (out_root / c).mkdir(parents=True, exist_ok=True)

    parts = sorted(p for p in out_root.glob("part_*") if p.is_dir())

    pos_i = 0
    neg_i = 0

    for part in parts:
        for lab_path in sorted((part / "labels").glob("*.json")):
            with open(lab_path, "r", encoding="utf-8") as f:
                obj = json.load(f)

            if obj.get("type") == "positive":
                new_id = f"pos_{pos_i:06d}"
                pos_i += 1
            else:
                new_id = f"neg_{neg_i:06d}"
                neg_i += 1

            # move image/mask/osm first (optional ordering)
            stem = lab_path.stem

            src_img = part / "images" / f"{stem}.png"
            if src_img.exists():
                src_img.replace(out_root / "images" / f"{new_id}.png")

            src_mask = part / "masks" / f"{stem}.png"
            if src_mask.exists():
                src_mask.replace(out_root / "masks" / f"{new_id}.png")

            src_osm = part / "osm" / f"{stem}.png"
            if src_osm.exists():
                src_osm.replace(out_root / "osm" / f"{new_id}.png")

            # rewrite + move label json
            obj["id"] = new_id
            with open(lab_path, "w", encoding="utf-8") as f:
                json.dump(obj, f, ensure_ascii=False)
            lab_path.replace(out_root / "labels" / f"{new_id}.json")

    # remove part_* dirs (including any empty subfolders)
    for part in parts:
        shutil.rmtree(part)

    return out_root
