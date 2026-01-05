import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import rasterize
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
    shp_path: str | Path,
    tif_path: str | Path,
    out_dir: str | Path,
    *,
    n_pos: int = 2000,
    n_neg: int = 2000,
    chip: ChipSpec = ChipSpec(),
    seed: int = 0,
    max_tries_per_sample: int = 200,
):
    random.seed(seed)
    np.random.seed(seed)

    shp_path = Path(shp_path)
    tif_path = Path(tif_path)
    out_dir = Path(out_dir)
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

        # Build spatial index once
        sindex = gdf.sindex

        size = chip.size_px
        half = size // 2

        def save_one(
            sample_id: str,
            img8: np.ndarray,
            label_obj: dict,
            mask: Optional[np.ndarray],
        ):
            img_path = paths["images"] / f"{sample_id}.png"
            cv2.imwrite(str(img_path), img8)

            lab_path = paths["labels"] / f"{sample_id}.json"
            with open(lab_path, "w", encoding="utf-8") as f:
                json.dump(label_obj, f, ensure_ascii=False)

            if mask is not None:
                mask_path = paths["masks"] / f"{sample_id}.png"
                cv2.imwrite(str(mask_path), mask)

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
                "polylines_px": polylines,  # list of polylines, each polyline is list of [x,y]
            }
            save_one(sample_id, img8, label_obj, mask)

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
                "polylines_px": polylines,
            }
            save_one(sample_id, img8, label_obj, mask)

            neg_done += 1

        print(f"Saved negatives: {neg_done}/{n_neg} (tries={neg_tries})")


shp_path = "/home/fatemeh/Downloads/hedg/Topo10NL2023/Hedges_polylines/Top10NL2023_inrichtingselementen_lijn_heg.shp"
tif_path = "/home/fatemeh/Downloads/hedg/LiDAR_metrics_AHN4/ahn4_10m_perc_95_normalized_height.tif"
out_dir = "/home/fatemeh/Downloads/hedg/results/test_dataset"

chip = ChipSpec(
    size_px=128,  # start small for testing
    band=1,
    label_mode="both",  # saves images + json + masks
    line_width_px=2,
)

generate_dataset(
    shp_path=shp_path,
    tif_path=tif_path,
    out_dir=out_dir,
    n_pos=3,
    n_neg=3,
    chip=chip,
    seed=123,  # repeatable
    max_tries_per_sample=200,
)
