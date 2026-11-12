"""
Does the laptop now compute the same train/val split as the cluster?

Every run trained and was scored on the split made on the cluster. The laptop
used to compute another one, because avoid_label_dirs saw 10 pdok_dataset2
labels there against 5,000 on the cluster. On 2026-11-09 the laptop data
became the cluster versions (see "Laptop copy of the cluster data" in
docs/experiments_log.md). This reruns spatial_train_val_split of
scripts/data/convert_pdok_polylines_to_detr_polyline.py
with the settings of the cluster conversion and compares it with the val stem
list of cluster run 2 and with the stored polylines/train and polylines/val.

Result (2026-11-09, on the laptop):

    Spatial split (5000m blocks, chip 250m): train=26902, val=3098
      (10.3% val, target 20%), 872 mixed components forced to train,
      12155 crops near avoided areas
    val stems equal to the cluster list: True (3098 against 3098)
    train stems equal to polylines/train: True
    val stems equal to polylines/val: True

So the split no longer depends on the machine.

    PYTHONPATH=. /home/fatemeh/miniconda3/envs/hedge/bin/python exps/probe_split_reproduces.py
"""

from pathlib import Path

from hedge_seg.paths import DATA_ROOT
from scripts.data.convert_pdok_polylines_to_detr_polyline import (
    spatial_train_val_split,
)


def stems(path):
    return {Path(p).stem for p in path}


def main(cfg):
    split = spatial_train_val_split(
        cfg["labels_dir"],
        block_m=cfg["block_m"],
        val_fraction=cfg["val_fraction"],
        seed=cfg["seed"],
        avoid_label_dirs=cfg["avoid_label_dirs"],
    )
    val = {s for s, v in split.items() if v == "val"}
    train = {s for s, v in split.items() if v == "train"}

    cluster_val = stems(Path(cfg["cluster_val_stems"]).read_text().split())
    stored_train = stems((cfg["polyline_dir"] / "train").glob("*.npz"))
    stored_val = stems((cfg["polyline_dir"] / "val").glob("*.npz"))

    print(
        f"val stems equal to the cluster list: {val == cluster_val} "
        f"({len(val)} against {len(cluster_val)})"
    )
    print(f"train stems equal to polylines/train: {train == stored_train}")
    print(f"val stems equal to polylines/val: {val == stored_val}")
    for name, a, b in [("val", val, cluster_val), ("train", train, stored_train)]:
        if a != b:
            print(f"  {name} only here: {sorted(a - b)[:10]}")
            print(f"  {name} only there: {sorted(b - a)[:10]}")


if __name__ == "__main__":
    cfg = dict(
        labels_dir=DATA_ROOT / "pdok_dataset3/labels",
        # The settings of the cluster conversion, 2026-07-21
        # (docs/experiments_log.md, Phase A data conversion).
        block_m=5000.0,
        val_fraction=0.2,
        seed=42,
        avoid_label_dirs=[DATA_ROOT / "pdok_dataset2/labels"],
        # ls of the cluster polylines/val, taken for cluster run 2.
        cluster_val_stems="/home/fatemeh/Downloads/hedge/cluster_val_stems.txt",
        polyline_dir=DATA_ROOT / "pdok_dataset3_polylines/polylines",
    )
    main(cfg)
