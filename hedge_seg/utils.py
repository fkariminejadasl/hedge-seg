import json


def get_n_polylines_from_json(json_dir):
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


def get_n_points_stats_in_polylines_from_json(json_dir):
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


"""
# test_dataset (256), pos_000092, 423 polylines, (128) 180, (64) 81,
from pathlib import Path

folder = "test_mini3"  # "test_mini3" #"test_256" #"test_dataset"
json_dir = Path(f"/home/fatemeh/Downloads/hedge/results/{folder}/labels")
n_lines_dic = get_n_polylines_from_json(json_dir)
get_n_points_stats_in_polylines_from_json(json_dir)
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
"""
