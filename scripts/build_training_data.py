from pathlib import Path

from hedge_seg.embeddings_and_pack import (
    pack_embeddings_polylines_npz,
    save_DINOv3_embeddings,
)
from hedge_seg.label_postprocess import process_labels_for_dir
from hedge_seg.training_data import generate_dataset

# ----------------------------
# Training data generation
# ----------------------------
label_mode = "polylines"
n_pos = 10  # 300_000
res = 256
dir_name = "test_mini2"  # f"test_{res}"
shp_path = Path(
    "/home/fatemeh/Downloads/hedge/Topo10NL2023/Hedges_polylines/Top10NL2023_inrichtingselementen_lijn_heg.shp"
)
tif_path = Path(
    "/home/fatemeh/Downloads/hedge/LiDAR_metrics_AHN4/ahn4_10m_perc_95_normalized_height.tif"
)
out_dir = Path(f"/home/fatemeh/Downloads/hedge/results/{dir_name}")
image_dir = out_dir / "images"
embed_dir = out_dir / "embeddings"
embed_dir.mkdir(parents=True, exist_ok=True)


print(f"Generating dataset with {n_pos} positive samples...")
generate_dataset(
    shp_path=shp_path,
    tif_path=tif_path,
    out_dir=out_dir,
    n_pos=n_pos,
    n_neg=0,
    size_px=res,
    label_mode=label_mode,
    seed=123,
    max_tries_per_sample=200,
    use_osm=False,
)


# ----------------------------
# Process labels (resample polylines, get bboxes)
# ----------------------------
main_dir = Path(f"/home/fatemeh/Downloads/hedge/results/{dir_name}")
labels_dir = main_dir / "labels"
out_dir = main_dir / "labels_processed"
out_dir.mkdir(parents=True, exist_ok=True)
n_points = 20

print(
    f"Processing labels: resampling polylines to {n_points} points, getting bboxes..."
)
process_labels_for_dir(labels_dir, out_dir, n_points)


# ----------------------------
# Save DINOv3 embeddings
# ----------------------------
# huggingface-cli login # from ~/.cache/huggingface/token

out_dir = Path(f"/home/fatemeh/Downloads/hedge/results/{dir_name}")
image_dir = out_dir / "images"
embed_dir = out_dir / "embeddings"
embed_dir.mkdir(parents=True, exist_ok=True)

print(f"Saving DINOv3 embeddings for images in: {image_dir}...")
save_DINOv3_embeddings(image_dir, embed_dir, batch_size=512)


# ----------------------------
# Pack embeddings and polylines into npz
# ----------------------------
main_dir = Path(f"/home/fatemeh/Downloads/hedge/results/{dir_name}")

print(
    f"Packing embeddings and polylines into npz files in: {main_dir / 'embs_polylines'}..."
)
pack_embeddings_polylines_npz(
    embeddings_dir=main_dir / "embeddings",
    labels_dir=main_dir / "labels_processed",
    output_dir=main_dir / "embs_polylines",
)

"""
from pathlib import Path
import json

folder = Path("/home/fatemeh/Downloads/hedge/results/test_dataset/labels")
n_lines_all = []
n_lines_dic = dict()
for json_path in folder.glob("*.json"):
    with json_path.open("r") as f:
        data = json.load(f)
    n_lines = data.get("n_lines")  # None if missing
    n_lines_all.append(n_lines)
    n_lines_dic[json_path.stem] = n_lines
"""

"""
# all LineString, point [2, 184], no empty, no invalid, all simple (no self crossing), closed=ring 318 items, 
gdf = gpd.read_file(shp_path)
a = [len(gdf.geometry.iloc[i].xy[0]) for i in range(len(gdf))] # [min(a),max(a)]=[2, 184]
b = dict(Counter(a).most_common()) # from collections import Counter
plt.bar(list(b.keys()), list(b.values()));plt.xlabel("n_points");plt.ylabel("n_polylines") # or [22:30]
idxs = np.where(np.asarray(a)==max(a))[0].tolist() # 62070, 62087, 62092 # 13307 min 2 pts
gdf.iloc[62070].geometry.bounds
gdf.iloc[idxs].geometry.is_closed # closed, simple, ring (closed+simple), valid, empty
a = [gdf.iloc[i].geometry.is_closed for i in range(len(gdf))]
idxs = np.where(np.asarray(a)==True)[0].tolist()
# 51399  388967 # pos_000000
# 143904.612, 529319.225 # max
# 27587.828  369991.498 # min
folder = Path("/home/fatemeh/Downloads/hedge/results/test_dataset/labels")
n_lines_by_file = {}
for p in folder.glob("pos_*.json"):
    with p.open("r", encoding="utf-8") as f:
        data = json.load(f)
    n_lines_by_file[p.name] = data.get("n_lines")
max_file = max(n_lines_by_file, key=n_lines_by_file.get) # pos_000092.json, 423 polylines
max_value = n_lines_by_file[max_file]
"""
