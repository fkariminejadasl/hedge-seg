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
cfg.classes scores each class against its own labels, as in
exps/probe_polyline_pr.py.

    /home/fatemeh/miniconda3/envs/hedge/bin/python exps/probe_recall_null_model.py

Results at 10 m, 3,098 val crops:

- Exp 2: real recall 0.840 at t=0.05 with 12.6 lines per crop, null 0.379,
  skill 0.461. At t=0.90 0.696, null 0.266, skill 0.431. Skill peaks at 0.471
  (t=0.40), so a perfect score head is worth about 0.04 of recall.
- Exp 10 (hedges and tree rows as one class) against exp 9 (hedges only),
  hedge labels only. At t=0.90 exp 10 recall 0.795 with 5.76 lines per crop,
  null 0.311, skill 0.483; exp 9 0.733 with 3.33, null 0.260, skill 0.473.
  Best skill over thresholds 0.499 against 0.507. So the extra 0.06 of hedge
  recall is luck from drawing more lines, and the woody class neither helps
  nor hurts finding hedges. Against both label layers, exp 10 skill is 0.501
  at t=0.90 against exp 9's 0.374.
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
        preds = [np.load(run_dir / "polylines" / f"{s}.npz") for s in stems]
        gts = [np.load(run_dir / "gt" / f"{s}.npz") for s in stems]

        # One fixed derangement, reused at every threshold and class so the
        # rows compare.
        order = rng.permutation(len(gts))
        while np.any(order == np.arange(len(gts))):
            order = rng.permutation(len(gts))

        for class_id, name in cfg["classes"].items():
            print(f"\n=== {run_dir.name}, {name} ===")
            print("    t   pred/img   real P   real R   null R   skill (R - null)")
            gt_lines = [_lines(g, class_id) for g in gts]
            for t in cfg["thresholds"]:
                pred_lines = [_lines(p, class_id, t) for p in preds]
                n_per_img = np.mean([len(p) for p in pred_lines])
                p, r = recall_precision(pred_lines, gt_lines, radius)
                _, r_null = recall_precision(
                    [pred_lines[i] for i in order], gt_lines, radius
                )
                print(
                    f"  {t:4.2f}   {n_per_img:7.2f}   {p:6.3f}   {r:6.3f}   "
                    f"{r_null:6.3f}   {r - r_null:6.3f}"
                )


def _lines(npz, class_id, score_thresh=None):
    """
    The polylines of one class, above a score. class_id None keeps every
    class. A file without labels is one class, so it is kept whole.
    """
    lines = npz["polylines"].astype(float)
    keep = np.ones(len(lines), dtype=bool)
    if score_thresh is not None:
        keep &= npz["scores"] >= score_thresh
    if class_id is not None and "labels" in npz:
        keep &= npz["labels"] == class_id
    return lines[keep]


if __name__ == "__main__":
    root = DATA_ROOT / "pdok_dataset3_polylines"
    tree = DATA_ROOT / "pdok_dataset3_tree_polylines"
    cfg = dict(
        run_dirs=[
            tree / "inference/9_val_t0.05",  # exp 9 ep 45, hedges only
            tree / "inference/best_10_val_t0.05",  # exp 10 ep 35, one woody class
            tree / "inference/10_val_t0.05",  # exp 10 ep 45, one woody class
            # root / "inference/best_2_val_cluster_t0.05",  # exp 2, softmax head
            # root / "inference/best_3_val_cluster_t0.05",  # exp 3, focal head
        ],
        # Each class against its own labels, as in exps/probe_polyline_pr.py.
        # A one-class model calls every line 0, so for exps 9 and 10, 0 is all
        # their lines against hedge labels. None is every line against every
        # label, the woody line score on the tree row folder. A crop with no
        # label of the class is skipped, not scored 1.0.
        classes={0: "hedge", None: "woody line"},
        buffer_m=10,
        thresholds=[0.05, 0.2, 0.4, 0.9, 0.95],
        seed=0,
    )
    main(cfg)
