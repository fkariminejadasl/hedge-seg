"""
Show an inference run of scripts/train_detr_unet_polyline.py as two grids:
ground truth and prediction, over the same crops.

Takes the run directory that the training script wrote, which already holds
both `polylines/` (predictions) and `gt/` (links to the ground truth of exactly
the crops that were run), so nothing has to be linked or renamed by hand:

    PYTHONPATH=. python scripts/show_polyline_results.py \
        <data_root>/pdok_dataset3_polylines/inference/1_150_val_cluster_t0.95

Both grids use the same crops in the same order, so panels can be compared
directly. Pass two run directories to compare two checkpoints instead:

    PYTHONPATH=. python scripts/show_polyline_results.py <run_a> <run_b>
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from hedge_seg.paths import DATA_ROOT
from hedge_seg.visualization import show_polyline_grid


def _ids(run_dir: Path, n: int, seed: int):
    """Pick the crops to show, the same ones for every grid of this run."""
    import numpy as np

    stems = sorted(p.stem for p in (run_dir / "polylines").glob("*.npz"))
    if not stems:
        raise RuntimeError(f"No predictions in {run_dir / 'polylines'}")
    rng = np.random.default_rng(seed)
    stems = rng.choice(stems, size=min(n, len(stems)), replace=False)
    return [int(s.split("_")[-1]) for s in sorted(stems)]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_dir", type=Path, nargs="+", help="inference run dir(s)")
    ap.add_argument(
        "--image-dir", type=Path, default=DATA_ROOT / "pdok_dataset3/images"
    )
    ap.add_argument("-n", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    # The GT grid is drawn once, from the first run, because every run of the
    # same split shows the same crops.
    ids = _ids(args.run_dir[0], args.n, args.seed)
    show_polyline_grid(
        image_dir=args.image_dir,
        polyline_dir=args.run_dir[0] / "gt",
        ids=ids,
        n=args.n,
        title="GT",
    )
    for run_dir in args.run_dir:
        show_polyline_grid(
            image_dir=args.image_dir,
            polyline_dir=run_dir / "polylines",
            ids=ids,
            n=args.n,
            title=run_dir.name,
        )
    plt.show()


if __name__ == "__main__":
    main()
