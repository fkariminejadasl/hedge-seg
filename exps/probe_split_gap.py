"""
How far is each val crop from the nearest train crop?

spatial_train_val_split in scripts/data/convert_pdok_polylines_to_detr_polyline.py
hashes 5 km blocks into val or train. But a group of overlapping crops that
reaches into a train block goes to train whole, so train crops also sit inside
val blocks. The split only promises that no val crop overlaps a train crop.
This measures the actual distance.

The gap is edge to edge: the shortest distance between the two 250 m squares.
0 means they touch.

Result (2026-10-07, on the cluster, the split of every cluster run):

    30000 crops: train 26902, val 3098
    val crops overlapping a train crop: 0
    5 km blocks with crops 1494, with val 285, with both val and train 235
    gap to the nearest train crop, edge to edge:
      min 0 m, 5% 49 m, 25% 266 m, median 593 m, 75% 1072 m,
      95% 2120 m, max 4456 m
    val crops with a train crop within    1 m:    3 (0.1%)
    val crops with a train crop within   50 m:  158 (5.1%)
    val crops with a train crop within  250 m:  737 (23.8%)
    val crops with a train crop within 1000 m: 2227 (71.9%)
    val crops with a train crop within 2500 m: 3013 (97.3%)

So val is ground the model has not seen, but mostly nearby ground, not a
distant region. The 5 km blocks set the val fraction, not the distance.

On the laptop, set val_dir to polylines/val_cluster: the laptop conversion has
its own split in polylines/val. The laptop labels also hold pos_029989, which
the cluster download lacks, and it overlaps val crop pos_024080. So the laptop
shows 1 overlapping pair and 4 touching crops instead of 0 and 3.

    PYTHONPATH=. /home/fatemeh/miniconda3/envs/hedge/bin/python exps/probe_split_gap.py
"""

import numpy as np
from scipy.spatial import cKDTree

from hedge_seg.paths import DATA_ROOT
from hedge_seg.utils import _load_crop_centers


def edge_gap(a, b, chip_m):
    """Shortest distance between two axis-aligned squares of side chip_m."""
    dx = np.maximum(0.0, np.abs(a[..., 0] - b[..., 0]) - chip_m)
    dy = np.maximum(0.0, np.abs(a[..., 1] - b[..., 1]) - chip_m)
    return np.hypot(dx, dy)


def main(cfg):
    files, centers, chip_m = _load_crop_centers(cfg["labels_dir"])
    val_stems = {p.stem for p in cfg["val_dir"].glob("*.npz")}
    is_val = np.array([f.stem in val_stems for f in files])
    assert is_val.sum() == len(val_stems), "val stems missing from labels_dir"
    train, val = centers[~is_val], centers[is_val]
    print(f"{len(files)} crops: train {len(train)}, val {len(val)}")

    pairs = cKDTree(centers).query_pairs(r=chip_m, p=np.inf, output_type="ndarray")
    n_cross = int((is_val[pairs[:, 0]] != is_val[pairs[:, 1]]).sum())
    print(f"val crops overlapping a train crop: {n_cross}")

    blocks = [tuple(b) for b in np.floor(centers / cfg["block_m"]).astype(int)]
    val_blocks = {b for b, v in zip(blocks, is_val) if v}
    train_blocks = {b for b, v in zip(blocks, is_val) if not v}
    print(
        f"{cfg['block_m'] / 1000:.0f} km blocks with crops {len(set(blocks))}, "
        f"with val {len(val_blocks)}, "
        f"with both val and train {len(val_blocks & train_blocks)}"
    )

    # The square nearest by center gives an upper bound g0 on the gap. Any square
    # with a smaller gap has its center within g0 + chip_m * sqrt(2), so
    # searching that radius makes the minimum exact.
    tree = cKDTree(train)
    _, nearest = tree.query(val)
    g0 = edge_gap(val, train[nearest], chip_m)
    gap = np.empty(len(val))
    for i, (v, r) in enumerate(zip(val, g0 + chip_m * np.sqrt(2) + 1)):
        gap[i] = edge_gap(v, train[tree.query_ball_point(v, r)], chip_m).min()

    p = np.percentile(gap, [5, 25, 50, 75, 95])
    print("gap to the nearest train crop, edge to edge:")
    print(
        f"  min {gap.min():.0f} m, 5% {p[0]:.0f} m, 25% {p[1]:.0f} m, "
        f"median {p[2]:.0f} m, 75% {p[3]:.0f} m,"
    )
    print(f"  95% {p[4]:.0f} m, max {gap.max():.0f} m")
    for d in [1, 50, 250, 1000, 2500]:
        n = int((gap < d).sum())
        print(
            f"val crops with a train crop within {d:>4} m: "
            f"{n:>4} ({n / len(val) * 100:.1f}%)"
        )


if __name__ == "__main__":
    cfg = dict(
        labels_dir=DATA_ROOT / "pdok_dataset3/labels",
        # The val crops of every cluster run. Train is every other crop.
        val_dir=DATA_ROOT / "pdok_dataset3_polylines/polylines/val",
        # On the laptop, the same 3,098 crops:
        # val_dir=DATA_ROOT / "pdok_dataset3_polylines/polylines/val_cluster",
        block_m=5000.0,  # block_m of the converter
    )
    main(cfg)
