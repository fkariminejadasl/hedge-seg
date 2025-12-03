from pathlib import Path

import cv2
import numpy as np
import rasterio
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


def download_ahn4_sample():
    """
    Download a sample AHN4 10m normalized height GeoTIFF from Zenodo,
    which is stored inside a zip file on a remote server.
    Uses rasterio's vsizip and vsicurl virtual file systems to read
    the TIF directly from the zip without downloading the entire zip first.
    Saves the extracted TIF locally.
    """
    zip_url = (
        "/vsizip/"
        "{/vsicurl/https://zenodo.org/records/15261042/files/4_AHN4.zip?download=1}"
        "/4_AHN4/ahn4_10m_perc_95_normalized_height.tif"
    )

    # Open the remote TIF inside the zip
    with rasterio.open(zip_url) as src:
        profile = src.profile
        data = src.read()  # you could read a window instead of all data

    # Save locally as a normal GeoTIFF
    profile.update(driver="GTiff")

    with rasterio.open(
        "/home/fatemeh/Downloads/ahn4_10m_perc_95_normalized_height.tif", "w", **profile
    ) as dst:
        dst.write(data)


# download_ahn4_sample()
# image = load_geotiff("/home/fatemeh/Downloads/UK_Knepp_10m_veg_TILE_000_BAND_perc_95_normalized_height.tif")
