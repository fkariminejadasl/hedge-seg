from pathlib import Path

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


"""
UK_Knepp_10m_veg_TILE_000_BAND_perc_95_normalized_height.tif

conda install -c conda-forge gdal

gdalinfo "/home/fatemeh/Downloads/UK_Knepp_10m_veg_TILE_000_BAND_perc_95_normalized_height.tif"

# Convert float32 -> UInt16 with automatic scaling based on min/max in the data
gdal_translate -ot UInt16 -scale "/home/fatemeh/Downloads/UK_Knepp_10m_veg_TILE_000_BAND_perc_95_normalized_height.tif" "/home/fatemeh/Downloads/knepp_95_height_u16.tif"

# Or with explicit scaling (example: expected range 0..50 meters -> full 16-bit)
gdal_translate -ot UInt16 -scale 0 50 0 65535 "/home/fatemeh/Downloads/UK_Knepp_10m_veg_TILE_000_BAND_perc_95_normalized_height.tif" "/home/fatemeh/knepp_95_height_u16.tif"
"""
