import glob
import json

import numpy as np
from scipy.spatial import cKDTree

files = sorted(
    glob.glob("/home/fatemeh/Downloads/hedge/results/pdok_dataset3/labels/*.json")
)
centers = np.array([json.load(open(f))["center_world"] for f in files])
print(f"{len(centers)} crops")

tree = cKDTree(centers)
# two 250m crops overlap iff Chebyshev distance between centers < 250
pairs = tree.query_pairs(r=250.0, p=np.inf)
deg = np.zeros(len(centers), int)
for i, j in pairs:
    deg[i] += 1
    deg[j] += 1
print(
    f"crops overlapping at least one other crop: {(deg > 0).sum()} ({(deg > 0).mean() * 100:.1f}%)"
)
print(f"mean overlapping neighbors per crop: {deg.mean():.1f}, max: {deg.max()}")

# simulate a random 80/20 split: fraction of val crops overlapping a train crop
rng = np.random.default_rng(42)
perm = rng.permutation(len(centers))
val = set(perm[int(0.8 * len(centers)) :])
leak = 0
adj = {}
for i, j in pairs:
    adj.setdefault(i, []).append(j)
    adj.setdefault(j, []).append(i)
for v in val:
    if any(
        n not in val for n in adj.get(v, [])
    ):  # means the neighbor belongs to training
        leak += 1
print(
    f"random 80/20 split: {leak}/{len(val)} val crops ({leak / len(val) * 100:.1f}%) overlap a training crop"
)


"""
# independently verify zero val-train overlap
import json, glob
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree

labels = Path("/home/fatemeh/Downloads/hedge/results/pdok_dataset3/labels")
out = Path("/home/fatemeh/Downloads/hedge/results/pdok_dataset3_polylines/polylines")
val_stems = {p.stem for p in (out / "val").glob("*.npz")}
train_stems = {p.stem for p in (out / "train").glob("*.npz")}
assert not (val_stems & train_stems), "stem in both splits!"

stems, centers = [], []
for f in sorted(labels.glob("pos_*.json")):
    stems.append(f.stem)
    centers.append(json.load(open(f))["center_world"])
centers = np.asarray(centers)
is_val = np.array([s in val_stems for s in stems])
assert is_val.sum() == len(val_stems) and (~is_val).sum() == len(train_stems)

tree = cKDTree(centers)
pairs = tree.query_pairs(r=250.0, p=np.inf, output_type="ndarray")
cross = (is_val[pairs[:, 0]] != is_val[pairs[:, 1]]).sum()
print(f"independent check: {len(pairs)} overlapping pairs total, {cross} val-train pairs (must be 0)")
"""
