"""
Show an inference run of scripts/train_detr_unet_polyline.py as a grid of
crops: one figure for the ground truth, one for each run being viewed.

An inference run directory already holds both `polylines/` (the predictions)
and `gt/` (links to the ground truth of exactly the crops that were run), so
listing the run directories is enough. Nothing has to be linked or renamed.

All figures show the same crops in the same order, so the panels line up and
two checkpoints can be compared square by square.

`score_thresh` filters the stored predictions by their score, so a run inferred
at 0.05 can be viewed at 0.90 or 0.95 without running inference again. Infer
once at 0.05 and pick the operating point here. It is not one value for all
runs: the exp 1 optimum is 0.95 and the exp 2 optimum is 0.90.

`save` writes each figure to `save_dir` as
`<model>_<run>_<split>_t<thresh>.png`, for example
`detr_unet_polyline_best_2_val_cluster_t.95.png`, with the ground truth as
`detr_unet_polyline_gt_val_cluster_t.95.png`.

Edit the cfg at the bottom and run:

    /home/fatemeh/miniconda3/envs/hedge/bin/python scripts/show_polyline_results.py
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


def split_run_name(run_dir: Path):
    """
    `best_2_val_cluster_t0.05` -> ("best_2", "val_cluster"). The directory names
    itself after the checkpoint, the split and the inference cutoff, so the
    figure name can be rebuilt from it.
    """
    stem = run_dir.name.rsplit("_t", 1)[0]
    for split in ("val_cluster", "val", "train"):
        if stem.endswith(f"_{split}"):
            return stem[: -len(split) - 1], split
    return stem, "unknown"


def figure_path(cfg, run_dir: Path, what: str):
    """`<model>_<what>_<split>_t<thresh>.png`, threshold without the leading 0."""
    if not cfg.save:
        return None
    _, split = split_run_name(run_dir)
    thresh = f"{cfg.score_thresh:.2f}".lstrip("0")
    return Path(cfg.save_dir) / f"{cfg.model}_{what}_{split}_t{thresh}.png"


def main(cfg):
    run_dirs = [Path(d) for d in cfg.run_dirs]
    for run_dir in run_dirs:
        if not (run_dir / "polylines").is_dir():
            raise FileNotFoundError(f"Not an inference run directory: {run_dir}")

    ids = pick_ids(run_dirs[0], cfg.n, cfg.seed)
    print(f"Showing {len(ids)} crops at score_thresh={cfg.score_thresh}: {ids}")

    # The ground truth is drawn once, from the first run, because every run of
    # the same split has the same ground truth for these crops. It carries no
    # scores, so score_thresh does not apply to it.
    show_polyline_grid(
        image_dir=cfg.image_dir,
        polyline_dir=run_dirs[0] / "gt",
        ids=ids,
        n=cfg.n,
        title="GT",
        save_path=figure_path(cfg, run_dirs[0], "gt"),
    )
    for run_dir in run_dirs:
        name, _ = split_run_name(run_dir)
        show_polyline_grid(
            image_dir=cfg.image_dir,
            polyline_dir=run_dir / "polylines",
            ids=ids,
            n=cfg.n,
            title=f"{name} t{cfg.score_thresh}",
            score_thresh=cfg.score_thresh,
            save_path=figure_path(cfg, run_dir, name),
        )
    plt.show()


if __name__ == "__main__":
    inference = DATA_ROOT / "pdok_dataset3_polylines/inference"
    cfg = dict(
        # One directory per run to view. The first one also supplies the ground
        # truth figure. Add or remove lines to compare different checkpoints.
        # Use the t0.05 runs: they hold every prediction, so score_thresh below
        # picks the operating point without re-running inference.
        run_dirs=[
            inference / "best_2_val_cluster_t0.05",
            inference / "1_150_val_cluster_t0.05",
        ],
        image_dir=DATA_ROOT / "pdok_dataset3/images",
        score_thresh=0.95,  # exp 1 optimum 0.95, exp 2 optimum 0.90
        n=16,  # crops per figure, drawn as a 4x4 grid
        seed=42,  # same seed gives the same crops, so figures stay comparable
        save=False,
        save_dir=Path("/home/fatemeh/Downloads/hedge/screenshots"),
        model="detr_unet_polyline",  # figure name prefix, says which model it is
    )
    main(OmegaConf.create(cfg))
