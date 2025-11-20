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
