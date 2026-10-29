"""
Quantify geographic crop overlap in pdok_dataset3 and verify the spatial
train/val split produced by scripts/data/convert_pdok_polylines_to_detr_polyline.py
leaves zero overlap between splits.
"""

from hedge_seg.paths import DATA_ROOT
from hedge_seg.utils import geographic_overlap_stats, verify_no_split_overlap

labels_dir = DATA_ROOT / "pdok_dataset3/labels"
out_root = DATA_ROOT / "pdok_dataset3_polylines"

stats = geographic_overlap_stats(labels_dir, val_fraction=0.2, seed=42)
print(f"{stats['n_crops']} crops, chip size {stats['chip_m']:.0f}m")
print(
    f"crops overlapping at least one other crop: "
    f"{stats['frac_crops_with_overlap'] * 100:.1f}%"
)
print(
    f"mean overlapping neighbors per crop: {stats['mean_neighbors']:.1f}, "
    f"max: {stats['max_neighbors']}"
)
print(
    f"naive random 80/20 split: "
    f"{stats['naive_split_val_leak_frac'] * 100:.1f}% of val crops "
    f"overlap a training crop"
)

# Independently verify the actual spatial split has zero leakage (recomputes
# overlap from labels_dir's center_world, not from the split code itself).
check = verify_no_split_overlap(
    labels_dir, out_root / "polylines" / "train", out_root / "polylines" / "val"
)
print(
    f"spatial split: train={check['n_train']}, val={check['n_val']}, "
    f"{check['n_overlapping_pairs']} overlapping pairs total, "
    f"{check['n_val_train_overlap_pairs']} val-train pairs (must be 0)"
)
assert check["n_val_train_overlap_pairs"] == 0
