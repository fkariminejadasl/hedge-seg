from pathlib import Path

import cv2
import geopandas as gpd
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.features import geometry_mask
from rasterio.plot import show
from rasterio.windows import from_bounds
from shapely.geometry import LineString


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

    img_8bit = float_to_uint8(data, vmin=vmin, vmax=vmax)

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


import geopandas as gpd

path = "/home/fatemeh/Downloads/hedg/Topo10NL2023/Hedges_polylines/Top10NL2023_inrichtingselementen_lijn_heg.shp"
gdf = gpd.read_file(path)

print(gdf.head())
print(gdf.crs)  # coordinate reference system
print(gdf.geometry.head())
a = gdf.geometry[
    0
]  # <LINESTRING (21605.971 368978.384, 21616.27 368976.488, 21630.911 368975.481...>
a.wkt  # 'LINESTRING (21605.971000000834 368978.3839999996, 21616.269999999553 368976.48800000176, 21630.91099999845 368975.48099999875, 21633.00800000131 368982.36300000176, 21637.787000000477 368989.39400000125, 21715.743999999017 369072.08900000155, 21722.780000001192 369087.4549999982, 21727.78200000152 369093.29399999976, 21732.83999999985 369098.31500000134, 21732.072999998927 369099.8500000015, 21692.936599999666 369112.6526999995, 21693.572000000626 369114.4629999995, 21693.76399999857 369115.01099999994, 21700.219799999148 369132.8814000003, 21699.210000000894 369133.40100000054, 21654.107000000775 369149.73000000045, 21629.05999999866 369158.7980000004, 21610.048000000417 369104.6160000004, 21569.69200000167 368986.2490000017)'
a.length  # 508.8196274877047
np.array(a.xy[0])  # x coordinates
a.xy[1]  # y coordinates
a.coords[:]

# image = load_geotiff("/home/fatemeh/Downloads/UK_Knepp_10m_veg_TILE_000_BAND_perc_95_normalized_height.tif")
# (30900, 26600)
# save_fullres_geotiff_as_png_tiled("/home/fatemeh/Downloads/ahn4_10m_perc_95_normalized_height.tif", "/home/fatemeh/Downloads/ahn4_fullres.png")
# image = load_geotiff("/home/fatemeh/Downloads/ahn4_10m_perc_95_normalized_height.tif") # (30900, 26600) 1/10
# save_float_geotiff_as_png(image, "/home/fatemeh/Downloads/ahn4.jpg")


def raster_values_in_buffer(
    shp_path: str | Path,
    tif_path: str | Path,
    *,
    pick: (
        int | None
    ) = 0,  # row index in the shapefile (or use an attribute filter below)
    buffer_dist: float = 50.0,  # in raster CRS units (often meters)
    band: int = 1,
    all_touched: bool = True,
):
    # 1) Read vector
    gdf = gpd.read_file(shp_path)

    # Choose the LineString you want:
    # Option A: pick by row index
    geom = gdf.geometry.iloc[pick]

    # Option B (recommended): pick by attribute, e.g.
    # geom = gdf.loc[gdf["road_id"] == 123, "geometry"].iloc[0]

    # 2) Open raster and align CRS
    with rasterio.open(tif_path) as src:
        raster_crs = src.crs

        if gdf.crs is None:
            raise ValueError(
                "Shapefile has no CRS (gdf.crs is None). Assign or fix it before reprojecting."
            )

        geom = gpd.GeoSeries([geom], crs=gdf.crs).to_crs(raster_crs).iloc[0]

        # 3) Buffer in raster CRS units
        buf = geom.buffer(buffer_dist)

        # 4) Build a small read window from the buffer bounds
        minx, miny, maxx, maxy = buf.bounds
        win = from_bounds(minx, miny, maxx, maxy, transform=src.transform)

        # Make sure window is clipped to raster
        win = win.intersection(rasterio.windows.Window(0, 0, src.width, src.height))

        data = src.read(band, window=win, masked=True)
        transform = src.window_transform(win)

        # 5) Mask pixels outside the buffer polygon (so you only keep buffer area)
        mask = geometry_mask(
            [buf],
            out_shape=data.shape,
            transform=transform,
            invert=True,  # True = inside geometry
            all_touched=all_touched,
        )

        # Combine raster nodata mask with geometry mask
        inside = np.ma.array(data, mask=(~mask) | np.ma.getmaskarray(data))

        # Flatten to 1D values inside buffer, drop masked
        vals = inside.compressed()

        return {
            "values": vals,  # numpy array of pixel values within buffer
            "count": vals.size,
            "mean": float(vals.mean()) if vals.size else None,
            "min": float(vals.min()) if vals.size else None,
            "max": float(vals.max()) if vals.size else None,
            "window": win,
            "buffer_geom": buf,
            "raster_crs": raster_crs,
        }


def float_to_uint8(
    data: np.ndarray,
    vmin: float | None = None,
    vmax: float | None = None,
) -> np.ndarray:
    """
    Convert a float32 raster to 8 bit.
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
        return np.zeros_like(data, dtype=np.uint8)

    data_clipped = np.clip(data, vmin, vmax)
    norm = (data_clipped - vmin) / (vmax - vmin + 1e-12)
    return (norm * 255).astype(np.uint8)


def save_linestring_buffer_raster_png(
    shp_path: str | Path,
    tif_path: str | Path,
    out_png: str | Path,
    *,
    linestring_index: int = 0,
    buffer_dist: float = 50.0,  # meters in EPSG:28992
    band: int = 1,
    draw_buffer_outline: bool = True,
    line_thickness: int = 2,
):
    shp_path = Path(shp_path)
    tif_path = Path(tif_path)
    out_png = Path(out_png)

    gdf = gpd.read_file(shp_path)
    if gdf.crs is None:
        raise ValueError("Shapefile CRS is missing. gdf.crs is None.")
    line = gdf.geometry.iloc[linestring_index]
    if line is None or line.is_empty:
        raise ValueError("Selected geometry is empty.")
    if line.geom_type != "LineString":
        raise ValueError(f"Expected LineString, got {line.geom_type}")

    with rasterio.open(tif_path) as src:
        if src.crs is None:
            raise ValueError("GeoTIFF CRS is missing.")
        if gdf.crs != src.crs:
            # Not needed for you, but safe
            line = gpd.GeoSeries([line], crs=gdf.crs).to_crs(src.crs).iloc[0]

        buf = line.buffer(buffer_dist)

        # Window around buffer
        minx, miny, maxx, maxy = buf.bounds
        win = from_bounds(minx, miny, maxx, maxy, transform=src.transform)
        win = win.round_offsets().round_lengths()
        win = win.intersection(rasterio.windows.Window(0, 0, src.width, src.height))

        data = src.read(band, window=win, masked=True).filled(np.nan)
        w_transform = src.window_transform(win)

    # Convert raster to 8-bit grayscale
    gray8 = float_to_uint8(data)

    # Convert to BGR so we can draw colored overlays
    img = cv2.cvtColor(gray8, cv2.COLOR_GRAY2BGR)

    # World -> pixel coords in this window
    inv = ~w_transform

    def world_to_rc(x, y):
        col, row = inv * (x, y)
        return int(round(row)), int(round(col))  # row, col

    # Draw the LineString
    xs, ys = line.xy
    pts = np.array(
        [[world_to_rc(x, y)[1], world_to_rc(x, y)[0]] for x, y in zip(xs, ys)],
        dtype=np.int32,
    )
    cv2.polylines(
        img, [pts], isClosed=False, color=(0, 0, 255), thickness=line_thickness
    )

    # Optionally draw buffer outline (polygon exterior)
    if draw_buffer_outline:
        bx, by = buf.exterior.xy
        bpts = np.array(
            [[world_to_rc(x, y)[1], world_to_rc(x, y)[0]] for x, y in zip(bx, by)],
            dtype=np.int32,
        )
        cv2.polylines(img, [bpts], isClosed=True, color=(0, 255, 255), thickness=1)

    cv2.imwrite(str(out_png), img)
    return out_png


# Example usage
shp_path = "/home/fatemeh/Downloads/hedg/Topo10NL2023/Hedges_polylines/Top10NL2023_inrichtingselementen_lijn_heg.shp"
tif_path = "/home/fatemeh/Downloads/hedg/LiDAR_metrics_AHN4/ahn4_10m_perc_95_normalized_height.tif"
save_path = "/home/fatemeh/Downloads/hedg/results/buffer_view.png"

save_linestring_buffer_raster_png(
    shp_path=shp_path,
    tif_path=tif_path,
    out_png=save_path,
    linestring_index=200,
    buffer_dist=100,  # meters
)

result = raster_values_in_buffer(
    shp_path,
    tif_path,
    pick=0,
    buffer_dist=100,  # meters if raster is in a meter-based CRS
)

print(result["count"], result["mean"], result["min"], result["max"])
