from pathlib import Path

import cv2
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.plot import show


def load_geotiff(path: str | Path) -> np.ndarray:
    """
    Load a raster file (GeoTIFF) and return its data as a numpy array.
    """
    # Open GeoTIFF
    with rasterio.open(path) as src:
        print(src.name)
        print(src.crs)  # Coordinate reference system
        print(src.bounds)  # Geographic extent
        data = src.read(1)  # Read first band
        # data = src.read(1, masked=True)  # reads as a masked array

    # Display it
    show(data)

    data_filled = np.where(np.isnan(data), 0, data)
    return data_filled


def save_float_geotiff_as_png(
    data: np.ndarray,
    out_path: str | Path,
    vmin: float | None = None,
    vmax: float | None = None,
) -> None:
    """
    Convert a float32 raster to 8 bit and save as PNG using OpenCV.
    vmin and vmax define the value range that maps to 0-255.
    If not given, they are computed from percentiles to avoid outliers.
    """

    # Replace NaNs with 0 or with some other background value
    data = np.nan_to_num(data, nan=0.0)

    # Decide display range
    if vmin is None:
        vmin = np.percentile(data, 2)  # lower 2 percent
        # vmin = float(np.nanmin(data))
    if vmax is None:
        vmax = np.percentile(data, 98)  # upper 98 percent
        # vmax = float(np.nanmax(data))

    if vmax <= vmin:
        # Edge case: constant image
        img_8bit = np.zeros_like(data, dtype=np.uint8)
    else:
        # Clip to [vmin, vmax]
        data_clipped = np.clip(data, vmin, vmax)

        # Normalize to 0-255
        norm = (data_clipped - vmin) / (vmax - vmin)
        img_8bit = (norm * 255).astype(np.uint8)

    out_path = Path(out_path)
    cv2.imwrite(out_path, img_8bit)


def compute_percentiles_approx(path: str | Path, sample_scale: int = 20):
    """Estimate percentiles from a coarse overview instead of full data."""
    with rasterio.open(path) as src:
        h = src.height // sample_scale
        w = src.width // sample_scale
        overview = src.read(
            1,
            out_shape=(h, w),
            resampling=Resampling.bilinear,
        )
    overview = np.nan_to_num(overview, nan=0.0)
    vmin = np.percentile(overview, 2)
    vmax = np.percentile(overview, 98)
    return float(vmin), float(vmax)


def save_fullres_geotiff_as_png_tiled(
    path: str | Path,
    out_path: str | Path,
) -> None:
    path = Path(path)
    out_path = Path(out_path)

    with rasterio.open(path) as src:
        height, width = src.height, src.width

        # 1) approximate vmin/vmax from an overview
        vmin, vmax = compute_percentiles_approx(path)

        # 2) allocate a uint8 array for the final image
        #    (1 byte per pixel instead of 4 bytes for float32)
        img_8bit = np.zeros((height, width), dtype=np.uint8)

        # 3) loop over blocks (GDAL blocks)
        for i, window in src.block_windows(1):
            block = src.read(1, window=window)
            block = np.nan_to_num(block, nan=0.0)

            block = np.clip(block, vmin, vmax)
            norm = (block - vmin) / (vmax - vmin + 1e-9)
            block_8 = (norm * 255).astype(np.uint8)

            r0 = window.row_off
            c0 = window.col_off
            r1 = r0 + window.height
            c1 = c0 + window.width
            img_8bit[r0:r1, c0:c1] = block_8

    cv2.imwrite(str(out_path), img_8bit)


# image = load_geotiff("/home/fatemeh/Downloads/UK_Knepp_10m_veg_TILE_000_BAND_perc_95_normalized_height.tif")
# (30900, 26600)
# save_fullres_geotiff_as_png_tiled("/home/fatemeh/Downloads/ahn4_10m_perc_95_normalized_height.tif", "/home/fatemeh/Downloads/ahn4_fullres.png")
# image = load_geotiff("/home/fatemeh/Downloads/ahn4_10m_perc_95_normalized_height.tif") # (30900, 26600) 1/10
# save_float_geotiff_as_png(image, "/home/fatemeh/Downloads/ahn4.jpg")
