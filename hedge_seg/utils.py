import json
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.windows import from_bounds
from tqdm import tqdm


def get_n_polylines(json_dir):
    n_lines_dic = dict()
    for json_path in json_dir.glob("*.json"):
        with json_path.open("r") as f:
            data = json.load(f)
        n_lines = data.get("n_lines")  # None if missing
        n_lines_dic[json_path.stem] = n_lines
    max_lines = max(n_lines_dic.values())
    max_filename = [n for n, v in n_lines_dic.items() if v == max_lines]
    print(f"Max n_lines: {max_lines} in file(s): {max_filename}")
    return n_lines_dic


def get_n_points_stats_in_polylines(json_dir):
    min_n_points_dic = dict()
    max_n_points_dic = dict()
    for json_path in json_dir.glob("*.json"):
        with json_path.open("r") as f:
            data = json.load(f)
        polylines = data.get("polylines_px")
        polylines_lengths = []
        for polyline in polylines:
            polylines_lengths.append(len(polyline))
        min_n_points = min(polylines_lengths)  # if polylines_lengths else 0
        min_n_points_dic[json_path.stem] = min_n_points
        max_n_points = max(polylines_lengths)
        max_n_points_dic[json_path.stem] = max_n_points
    min_points = min(min_n_points_dic.values())
    max_points = max(max_n_points_dic.values())
    print(f"Min n_points: {min_points}")
    print(f"Max n_points: {max_points}")
    return min_n_points_dic, max_n_points_dic


def get_min_polyline_length(json_dir):
    min_length_dic = dict()
    for json_path in json_dir.glob("*.json"):
        with json_path.open("r") as f:
            data = json.load(f)
        polylines = data.get("polylines_px", [])

        lengths = []
        for polyline in polylines:
            length = sum(
                ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
                for (x1, y1), (x2, y2) in zip(polyline[:-1], polyline[1:])
            )
            lengths.append(length)

        min_length_dic[json_path.stem] = min(lengths) if lengths else None

    # min_length = min(v for v in min_length_dic.values() if v is not None)
    # min_filename = [n for n, v in min_length_dic.items() if v == min_length]
    return min_length_dic


def get_closed_polylines(json_dir):
    closed_polylines_dic = dict()
    for json_path in json_dir.glob("*.json"):
        with json_path.open("r") as f:
            data = json.load(f)
        polylines = data.get("polylines_px")
        closed_polylines = []
        for polyline in polylines:
            polyline = np.array(polyline)
            if np.array_equal(polyline[0], polyline[-1]):
                closed_polylines.append(polyline[0])
        if closed_polylines:
            closed_polylines_dic[json_path.stem] = closed_polylines
    return closed_polylines_dic


def get_polyline_length(xs, ys):
    pl_len = 0
    for x1, y1, x2, y2 in zip(xs[:-1], ys[:-1], xs[1:], ys[1:]):
        pl_len += ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
    return pl_len


def get_line_segment_lengths(xs, ys):
    lengths = []
    for x1, y1, x2, y2 in zip(xs[:-1], ys[:-1], xs[1:], ys[1:]):
        length = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
        lengths.append(length)
    return lengths


def get_polyline_intensities(shp_path, tif_path):
    gdf = gpd.read_file(shp_path)

    with rasterio.open(tif_path) as src:
        if gdf.crs != src.crs:
            gdf = gdf.to_crs(src.crs)

        gdf = gdf[gdf.geometry.notna()]
        gdf = gdf[~gdf.geometry.is_empty]
        gdf = gdf[gdf.geometry.geom_type.isin(["LineString", "MultiLineString"])]

        all_values = []

        for geom in tqdm(gdf.geometry):
            minx, miny, maxx, maxy = geom.bounds
            win = from_bounds(minx, miny, maxx, maxy, src.transform)
            win = win.round_offsets().round_lengths()

            row_off = max(0, int(win.row_off))
            col_off = max(0, int(win.col_off))
            height = min(src.height - row_off, int(win.height))
            width = min(src.width - col_off, int(win.width))

            if height <= 0 or width <= 0:
                continue

            win = rasterio.windows.Window(col_off, row_off, width, height)
            arr = src.read(1, window=win, masked=True)
            transform = src.window_transform(win)

            mask = rasterize(
                [(geom, 1)],
                out_shape=(height, width),
                transform=transform,
                fill=0,
                all_touched=True,
                dtype=np.uint8,
            ).astype(bool)

            vals = arr[mask]
            if np.ma.isMaskedArray(vals):
                vals = vals.compressed()

            all_values.extend(vals.tolist())

    all_values = np.asarray(all_values, dtype=float)
    return all_values


"""
# Get raster values on hedge polylines and plot histogram and CDF
shp_path = Path(
    "/home/fatemeh/Downloads/hedge/Topo10NL2023/Hedges_polylines/Top10NL2023_inrichtingselementen_lijn_heg.shp"
)
tif_path = Path(
    "/home/fatemeh/Downloads/hedge/LiDAR_metrics_AHN4/ahn4_10m_perc_95_normalized_height.tif"
)
all_values = get_polyline_intensities(shp_path, tif_path)
bins = np.arange(int(round(min(all_values), 0)), int(round(max(all_values), 0)) + 1, 1)
counts, bin_edges = np.histogram(all_values, bins=bins)
cdf = np.cumsum(counts) / np.sum(counts)

print("n_values:", len(all_values))
plt.figure()
plt.hist(all_values, bins=bins)
plt.xlabel("Intensity value")
plt.ylabel("Count")
plt.title("Histogram of raster values on hedge polylines")

plt.figure()
plt.plot(cdf)
plt.xlabel("Intensity value")
plt.ylabel("CDF")
plt.title("CDF of raster values on hedge polylines")
plt.grid(True)
"""

"""
# test_dataset (256), pos_000092, 423 polylines, (128) 180, (64) 81,
from pathlib import Path

folder = "test_mini6"  # "test_256_None"
json_dir = Path(f"/home/fatemeh/Downloads/hedge/results/{folder}/labels")
# json_dir = Path(f"/home/fkarimineja/data/hedge/{folder}/labels_processed")
n_lines_dic = get_n_polylines(json_dir)
closed_polylines_dic = get_closed_polylines(json_dir)

from collections import Counter
max_pl = max(n_lines_dic.values()) # 276 # 128 max polylines
a = Counter(n_lines_dic.values()).most_common()
f = [(p, p*c) for p, c in a]
b = [(p, (max_pl-p)*c) for p, c in a if p<=max_pl]
sum([c for p, c in f]) / sum([c for p, c in b]) # 691006, 4000994 = 0.17 (max 276), 691006 / 1564050=.44 (128) polylines to non-polylines ratio
sum([c for p, c in a if p>max_pl]) / sum([c for p, c in a if p<=max_pl]) # 1599 / 15401=.1 images with more than 128 polylines to less than 128 ratio

closed_counts = {k: len(v) for k, v in closed_polylines_dic.items()}
sum(closed_counts.values()) # 4121
sum(n_lines_dic.values()) # 691006

min_length_dic = get_min_polyline_length(json_dir)
get_n_points_stats_in_polylines(json_dir)
print("Done")
"""

"""
# all LineString, point [2, 184], no empty, no invalid, all simple (no self crossing), closed=ring 318 items, 
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
from pathlib import Path
from collections import Counter

shp_path = Path(
    "/home/fatemeh/Downloads/hedge/Topo10NL2023/Hedges_polylines/Top10NL2023_inrichtingselementen_lijn_heg.shp"
)
gdf = gpd.read_file(shp_path)
a = [len(gdf.geometry.iloc[i].xy[0]) for i in range(len(gdf))] # [min(a),max(a)]=[2, 184]
b = dict(Counter(a).most_common()) # from collections import Counter
plt.bar(list(b.keys()), list(b.values()));plt.xlabel("n_points");plt.ylabel("n_polylines") # or [22:30]
idxs = np.where(np.asarray(a)==max(a))[0].tolist() # 62070, 62087, 62092 # 13307 min 2 pts
gdf.iloc[62070].geometry.bounds
gdf.iloc[idxs].geometry.is_closed # closed, simple, ring (closed+simple), valid, empty
a = [gdf.iloc[i].geometry.is_closed for i in range(len(gdf))]
idxs = np.where(np.asarray(a)==True)[0].tolist()



# Get polyline lengths and plot histogram
# ========
lengths = dict()
for i in range(len(gdf)):
    xs, ys = gdf.geometry.iloc[i].xy
    pl_len = get_polyline_length(xs, ys)
    lengths[i] = pl_len

max_len = int(round(max(lengths.values()), 0))
min_len = int(round(min(lengths.values()), 0))
bins = np.arange(min_len, max_len + 10, 10)
plt.figure()
plt.hist(lengths.values(), bins=bins)
plt.xlabel("length");plt.ylabel("n_polylines")


min_len = min(lengths.values())
[k for k, v in lengths.items() if v == min_len] # 42645 254463, 539087 -> length .1
len([int(round(v,0)) for k, v in lengths.items() if v < 10]) # 110


# Get line segment lengths and plot histogram
# =========
seg_lens = []
for i in range(len(gdf)):
    xs, ys = gdf.geometry.iloc[i].xy
    seg_lens.extend(get_line_segment_lengths(xs, ys))


bins = np.arange(int(min(seg_lens)), int(max(seg_lens)) + 10, 10)
plt.figure()
plt.hist(seg_lens, bins=bins)
plt.xlabel("segment lengths");plt.ylabel("n_polylines")
"""
