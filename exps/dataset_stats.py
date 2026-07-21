"""
Label-cleaning stats for scripts/data/convert_pdok_polylines_to_detr_polyline.py.

Compares dataset_root (raw pos_*.json labels) against out_root (converted,
train+val polylines/*.npz): closed-ring ratio and short-polyline ratio.
out_root should show ~0 for both, confirming ring-opening and length
filtering worked; dataset_root shows what the raw data looks like before
cleaning. Uses hedge_seg.utils.{iter_polylines_from_json_dir,
iter_polylines_from_npz_dir, closed_ring_ratio, short_polyline_ratio}.
"""

from pathlib import Path

from hedge_seg.paths import DATA_ROOT
from hedge_seg.utils import (
    closed_ring_ratio,
    iter_polylines_from_json_dir,
    iter_polylines_from_npz_dir,
    short_polyline_ratio,
)


def report(dataset_root: Path, out_root: Path, min_length_px: float = 40.0):
    labels_dir = dataset_root / "labels"
    train_dir = out_root / "polylines" / "train"
    val_dir = out_root / "polylines" / "val"

    raw = list(iter_polylines_from_json_dir(labels_dir))
    converted = list(iter_polylines_from_npz_dir(train_dir)) + list(
        iter_polylines_from_npz_dir(val_dir)
    )

    n_closed, n_total, closed_frac = closed_ring_ratio(raw)
    print(
        f"[{dataset_root.name}] closed rings (raw): {n_closed}/{n_total} "
        f"({closed_frac * 100:.2f}%)"
    )
    n_closed_c, n_total_c, closed_frac_c = closed_ring_ratio(converted)
    print(
        f"[{out_root.name}] closed rings (converted, should be ~0): "
        f"{n_closed_c}/{n_total_c} ({closed_frac_c * 100:.2f}%)"
    )

    n_short, n_total_s, short_frac = short_polyline_ratio(raw, min_length_px)
    print(
        f"[{dataset_root.name}] short polylines (raw, < {min_length_px:.0f}px): "
        f"{n_short}/{n_total_s} ({short_frac * 100:.2f}%)"
    )
    n_short_c, n_total_sc, short_frac_c = short_polyline_ratio(
        converted, min_length_px
    )
    print(
        f"[{out_root.name}] short polylines (converted, should be 0): "
        f"{n_short_c}/{n_total_sc} ({short_frac_c * 100:.2f}%)"
    )


if __name__ == "__main__":
    report(
        dataset_root=DATA_ROOT / "pdok_dataset3",
        out_root=DATA_ROOT / "pdok_dataset3_polylines",
        min_length_px=40.0,
    )
