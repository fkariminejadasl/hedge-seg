"""
Show an inference run of scripts/train_detr_unet_polyline.py as a grid of
crops: one figure for the ground truth, one for each run being viewed.

An inference run directory already holds both `polylines/` (the predictions)
and `gt/` (links to the ground truth of exactly the crops that were run), so
listing the run directories is enough. Nothing has to be linked or renamed.

All figures show the same crops in the same order, so the panels line up and
two checkpoints can be compared square by square.

Edit the cfg at the bottom and run:

    PYTHONPATH=. python scripts/show_polyline_results.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from omegaconf import OmegaConf

from hedge_seg.paths import DATA_ROOT
from hedge_seg.visualization import show_polyline_grid


def pick_ids(run_dir: Path, n: int, seed: int):
    """Choose which crops to show. The same crops are used for every figure."""
    stems = sorted(p.stem for p in (run_dir / "polylines").glob("*.npz"))
    if not stems:
        raise RuntimeError(f"No predictions in {run_dir / 'polylines'}")
    rng = np.random.default_rng(seed)
    chosen = rng.choice(stems, size=min(n, len(stems)), replace=False)
    return [int(s.split("_")[-1]) for s in sorted(chosen)]


def main(cfg):
    run_dirs = [Path(d) for d in cfg.run_dirs]
    for run_dir in run_dirs:
        if not (run_dir / "polylines").is_dir():
            raise FileNotFoundError(f"Not an inference run directory: {run_dir}")

    ids = pick_ids(run_dirs[0], cfg.n, cfg.seed)
    print(f"Showing {len(ids)} crops: {ids}")

    # The ground truth is drawn once, from the first run, because every run of
    # the same split has the same ground truth for these crops.
    show_polyline_grid(
        image_dir=cfg.image_dir,
        polyline_dir=run_dirs[0] / "gt",
        ids=ids,
        n=cfg.n,
        title="GT",
    )
    for run_dir in run_dirs:
        show_polyline_grid(
            image_dir=cfg.image_dir,
            polyline_dir=run_dir / "polylines",
            ids=ids,
            n=cfg.n,
            title=run_dir.name,
        )
    plt.show()


if __name__ == "__main__":
    inference = DATA_ROOT / "pdok_dataset3_polylines/inference"
    cfg = dict(
        # One directory per run to view. The first one also supplies the ground
        # truth figure. Add or remove lines to compare different checkpoints.
        run_dirs=[
            inference / "best_1_val_cluster_t0.95",
            inference / "1_150_val_cluster_t0.95",
        ],
        image_dir=DATA_ROOT / "pdok_dataset3/images",
        n=16,  # crops per figure, drawn as a 4x4 grid
        seed=42,  # same seed gives the same crops, so figures stay comparable
    )
    main(OmegaConf.create(cfg))
