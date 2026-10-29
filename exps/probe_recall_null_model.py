"""
Is the high recall at a low score threshold real, or is it clutter?

"Recall is 0.84 at t=0.05, so the lines are found and then thrown away by the
score" is the argument for spending runs on the score head. It has a hole: at
t=0.05 the model draws about 13 lines per image against 2.1 ground-truth lines.
With that many lines, some land near a hedge by luck, and buffered recall
cannot tell luck from skill.

The null model here keeps each crop's predictions but scores them against a
DIFFERENT crop's labels, so the count, length and orientation mix of the
predictions is exactly the same and only the correspondence to the image is
broken. The gap between real recall and null recall is the part that is skill.

    /home/fatemeh/miniconda3/envs/hedge/bin/python exps/probe_recall_null_model.py
"""

import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hedge_seg.metrics import densify, meters_to_px  # noqa: E402
from hedge_seg.paths import DATA_ROOT  # noqa: E402


def recall_precision(pred_list, gt_list, radius):
    """Buffered length P and R per image, then averaged over images."""
    recalls, precisions = [], []
    for pred, gt in zip(pred_list, gt_list):
        if len(gt) == 0:
            continue
        gt_pts = densify(gt)
        if len(pred) == 0:
            recalls.append(0.0)
            continue
        pred_pts = densify(pred)
        d_gt, _ = cKDTree(pred_pts).query(gt_pts)
        recalls.append(float((d_gt <= radius).mean()))
        d_pr, _ = cKDTree(gt_pts).query(pred_pts)
        precisions.append(float((d_pr <= radius).mean()))
    return float(np.mean(precisions)), float(np.mean(recalls))


def main(cfg):
    radius = meters_to_px(cfg["buffer_m"])
    rng = np.random.default_rng(cfg["seed"])

    for run_dir in cfg["run_dirs"]:
        run_dir = Path(run_dir)
        stems = sorted(p.stem for p in (run_dir / "polylines").glob("*.npz"))
        print(f"\n=== {run_dir.name} ===")
        print("    t   pred/img   real P   real R   null R   skill (R - null)")

        preds_by_t = {t: [] for t in cfg["thresholds"]}
        gts = []
        for stem in stems:
            npz = np.load(run_dir / "polylines" / f"{stem}.npz")
            pred, score = npz["polylines"].astype(float), npz["scores"].astype(float)
            gts.append(
                np.load(run_dir / "gt" / f"{stem}.npz")["polylines"].astype(float)
            )
            for t in cfg["thresholds"]:
                preds_by_t[t].append(pred[score >= t])

        # One fixed derangement, reused at every threshold so the rows compare.
        order = rng.permutation(len(gts))
        while np.any(order == np.arange(len(gts))):
            order = rng.permutation(len(gts))

        for t in cfg["thresholds"]:
            preds = preds_by_t[t]
            n_per_img = np.mean([len(p) for p in preds])
            p, r = recall_precision(preds, gts, radius)
            _, r_null = recall_precision([preds[i] for i in order], gts, radius)
            print(
                f"  {t:4.2f}   {n_per_img:7.2f}   {p:6.3f}   {r:6.3f}   "
                f"{r_null:6.3f}   {r - r_null:6.3f}"
            )


if __name__ == "__main__":
    root = DATA_ROOT / "pdok_dataset3_polylines"
    cfg = dict(
        run_dirs=[
            root / "inference/best_2_val_cluster_t0.05",  # exp 2, softmax head
            root / "inference/best_3_val_cluster_t0.05",  # exp 3, focal head
        ],
        buffer_m=10,
        thresholds=[0.05, 0.2, 0.4, 0.9, 0.95],
        seed=0,
    )
    main(cfg)
