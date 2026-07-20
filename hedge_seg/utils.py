import json
import math
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.windows import from_bounds
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree
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


def iter_polylines_from_json_dir(json_dir):
    """Yield (stem, polyline) for every polyline in every pos_*.json label file."""
    for p in sorted(Path(json_dir).glob("pos_*.json")):
        data = json.loads(p.read_text(encoding="utf-8"))
        for line in data.get("polylines_px", []) or []:
            yield p.stem, line


def iter_polylines_from_npz_dir(npz_dir):
    """Yield (stem, polyline) for every polyline in every *.npz polyline file
    (as written by scripts/data/convert_pdok_polylines_to_detr_polyline.py)."""
    for p in sorted(Path(npz_dir).glob("*.npz")):
        d = np.load(p)
        for line in d["polylines"]:
            yield p.stem, line


def closed_ring_ratio(polyline_iter, gap_eps_px=2.0):
    """
    Fraction of polylines that are closed rings: first and last point within
    gap_eps_px. Polylines with fewer than 3 points are not counted (a 2-point
    line cannot meaningfully be a ring).
    """
    n_total = 0
    n_closed = 0
    for _, line in polyline_iter:
        pts = np.asarray(line, dtype=float)
        if pts.shape[0] < 3:
            continue
        n_total += 1
        gap = math.hypot(pts[-1, 0] - pts[0, 0], pts[-1, 1] - pts[0, 1])
        if gap <= gap_eps_px:
            n_closed += 1
    ratio = n_closed / n_total if n_total else 0.0
    return n_closed, n_total, ratio


def short_polyline_ratio(polyline_iter, min_length_px=40.0):
    """Fraction of polylines shorter (by arc length) than min_length_px."""
    n_total = 0
    n_short = 0
    for _, line in polyline_iter:
        pts = np.asarray(line, dtype=float)
        n_total += 1
        length = (
            float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum())
            if pts.shape[0] >= 2
            else 0.0
        )
        if length < min_length_px:
            n_short += 1
    ratio = n_short / n_total if n_total else 0.0
    return n_short, n_total, ratio


def _load_crop_centers(labels_dir):
    json_files = sorted(Path(labels_dir).glob("pos_*.json"))
    centers, chip_m = [], 0.0
    for f in json_files:
        data = json.loads(f.read_text(encoding="utf-8"))
        centers.append(data["center_world"])
        chip_m = max(chip_m, float(data["chip_size_m"]))
    return json_files, np.asarray(centers, dtype=np.float64), chip_m


def geographic_overlap_stats(labels_dir, val_fraction=0.2, seed=42):
    """
    Quantify how much geographic crop overlap exists in a raw label directory
    (center_world in each pos_*.json), and how much of it would leak into a
    naive random train/val split. Two crops overlap iff the Chebyshev distance
    between their centers is below chip_size_m.
    """
    json_files, centers, chip_m = _load_crop_centers(labels_dir)
    n = len(json_files)

    tree = cKDTree(centers)
    pairs = tree.query_pairs(r=chip_m, p=np.inf, output_type="ndarray")
    deg = np.zeros(n, dtype=int)
    for i, j in pairs:
        deg[i] += 1
        deg[j] += 1

    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    val = set(perm[int((1 - val_fraction) * n) :].tolist())
    adj = {}
    for i, j in pairs:
        adj.setdefault(i, []).append(j)
        adj.setdefault(j, []).append(i)
    leak = sum(1 for v in val if any(nb not in val for nb in adj.get(v, [])))

    return {
        "n_crops": n,
        "chip_m": chip_m,
        "n_overlapping_pairs": len(pairs),
        "frac_crops_with_overlap": float((deg > 0).mean()) if n else 0.0,
        "mean_neighbors": float(deg.mean()) if n else 0.0,
        "max_neighbors": int(deg.max()) if n else 0,
        "naive_split_val_leak_frac": leak / len(val) if val else 0.0,
    }


def verify_no_split_overlap(labels_dir, train_npz_dir, val_npz_dir):
    """
    Independently verify a geographic train/val split: recompute crop overlap
    from the raw label directory's center_world (not from the split code
    itself) and check that no val crop overlaps a train crop.
    """
    val_stems = {p.stem for p in Path(val_npz_dir).glob("*.npz")}
    train_stems = {p.stem for p in Path(train_npz_dir).glob("*.npz")}
    assert not (val_stems & train_stems), "stem present in both splits"

    json_files, centers, chip_m = _load_crop_centers(labels_dir)
    is_val = np.array([f.stem in val_stems for f in json_files])
    assert is_val.sum() == len(val_stems)
    assert (~is_val).sum() == len(train_stems)

    tree = cKDTree(centers)
    pairs = tree.query_pairs(r=chip_m, p=np.inf, output_type="ndarray")
    n_cross = (
        int((is_val[pairs[:, 0]] != is_val[pairs[:, 1]]).sum()) if len(pairs) else 0
    )
    return {
        "n_train": int((~is_val).sum()),
        "n_val": int(is_val.sum()),
        "n_overlapping_pairs": len(pairs),
        "n_val_train_overlap_pairs": n_cross,
    }


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
