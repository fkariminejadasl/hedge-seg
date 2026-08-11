"""
Does the score say a prediction is right, or only that it is straight?

The score head is the suspected limit on recall. This asks what the score
actually ranks by. A prediction counts as correct when at least `min_cov` of
its length lies within `buffer_m` of a label, which is the same buffered-length
idea as hedge_seg/metrics.py, applied per prediction instead of per image.

Result on exp 2 `best_2.pt` (2026-08-10, all 3,098 val crops from the t=0.05
run, 10 m buffer): 7,956 of 39,104 predictions are correct.

    straightness      median score, correct   median score, wrong
    < 0.85 (bent)              0.453                  0.216
    0.85 to 0.95               0.883                  0.690
    0.95 to 0.99               0.951                  0.868
    > 0.99 (straight)          0.974                  0.957

- A correct bent line scores 0.45, a wrong straight one 0.96, so no single
  threshold keeps both. At t=0.95, 9% of correct bent predictions survive
  against 74% of correct straight ones.
- AUC of the score for correct against wrong is 0.767 overall, but only 0.618
  inside the bent group. Most of its apparent skill is straightness as a proxy.
- Predicted straightness at t=0.80 is 0.908 against 0.909 in the labels, so the
  "exp 2 draws straighter lines than the labels" regression was the 0.95 cut,
  not the model.

Result on exp 3 `best_3.pt` (2026-08-11), the focal sigmoid head this probe
motivated: the ranking did not improve. AUC 0.742 overall against exp 2's
0.767, and 0.648 inside the bent group against 0.618, so a third of a point of
the intended effect and a loss everywhere else. The scores collapsed into two
clumps instead of spreading: a mass near 0.2 and a spike above 0.98, where
correct and wrong straight lines sit together at 0.983 and 0.979. Changing the
classification loss does not help, because the class head reads the mean of the
20 point features and never sees how well those points fit the image.

See `exps/probe_recall_null_model.py` for why this matters less than it looked:
most of the recall below the operating threshold is clutter, not lost hedges.

    /home/fatemeh/miniconda3/envs/hedge/bin/python exps/probe_score_quality.py
"""

import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hedge_seg.metrics import densify, meters_to_px, straightness  # noqa: E402
from hedge_seg.paths import DATA_ROOT  # noqa: E402

BUCKETS = [(0.0, 0.85), (0.85, 0.95), (0.95, 0.99), (0.99, 1.01)]


def auc(scores, positive):
    """Probability that a random correct prediction outranks a random wrong one."""
    n1, n0 = int(positive.sum()), int((~positive).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    order = np.argsort(scores)
    ranks = np.empty(len(scores))
    ranks[order] = np.arange(1, len(scores) + 1)
    return float((ranks[positive].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def main(cfg):
    for run_dir in cfg["run_dirs"]:
        print(f"\n=== {Path(run_dir).name} ===")
        one_run(Path(run_dir), cfg)


def one_run(run_dir, cfg):
    radius = meters_to_px(cfg["buffer_m"])

    s_pred, scores, coverage, s_gt = [], [], [], []
    for path in sorted((run_dir / "polylines").glob("*.npz")):
        npz = np.load(path)
        pred = npz["polylines"].astype(float)
        score = npz["scores"].astype(float)
        gt = np.load(run_dir / "gt" / path.name)["polylines"].astype(float)
        if len(gt) == 0 or len(pred) == 0:
            continue
        s_gt.extend(straightness(g) for g in gt)
        gt_tree = cKDTree(densify(gt))
        for polyline, sc in zip(pred, score):
            d, _ = gt_tree.query(densify([polyline]))
            coverage.append(float((d <= radius).mean()))
            s_pred.append(straightness(polyline))
            scores.append(float(sc))

    s_pred = np.asarray(s_pred)
    scores = np.asarray(scores)
    coverage = np.asarray(coverage)
    s_gt = np.asarray(s_gt)
    ok = coverage >= cfg["min_cov"]

    print(
        f"{len(s_pred)} predictions, {int(ok.sum())} correct "
        f"(>= {cfg['min_cov']:.0%} of length within {cfg['buffer_m']} m of a label)"
    )
    print(
        f"labels: mean straightness {s_gt.mean():.3f}, "
        f"strongly bent (<0.90) {np.mean(s_gt < 0.90):.3f}"
    )

    print("\nmedian score by straightness, correct against wrong:")
    print("  straightness        correct            wrong        kept at t=0.95")
    for lo, hi in BUCKETS:
        in_bucket = (s_pred >= lo) & (s_pred < hi)
        c, w = in_bucket & ok, in_bucket & ~ok
        print(
            f"  [{lo:.2f},{hi:.2f})   {np.median(scores[c]):.3f} (n={int(c.sum()):5d})"
            f"   {np.median(scores[w]):.3f} (n={int(w.sum()):5d})"
            f"      {np.mean(scores[c] >= 0.95):.3f}"
        )

    print(
        f"\nAUC of score for correct against wrong: {auc(scores, ok):.3f} "
        "(0.5 = useless)"
    )
    for lo, hi in [(0.0, 0.90), (0.90, 1.01)]:
        m = (s_pred >= lo) & (s_pred < hi)
        print(f"  within straightness [{lo:.2f},{hi:.2f}): {auc(scores[m], ok[m]):.3f}")

    print("\nstraightness of everything that survives each threshold:")
    print("   t     n_pred   mean   strongly bent (<0.90)")
    for t in cfg["thresholds"]:
        m = scores >= t
        print(
            f"  {t:4.2f}  {int(m.sum()):7d}  {s_pred[m].mean():.3f}   "
            f"{np.mean(s_pred[m] < 0.90):.3f}"
        )
    print(f"  labels                {s_gt.mean():.3f}   {np.mean(s_gt < 0.90):.3f}")

    # The four numbers the docs and the talk quote, spelled out, so a reader
    # does not have to work out which cell of which table they came from.
    bent, straight = s_pred < 0.85, s_pred > 0.99
    t = 0.95
    print(f"\nquoted elsewhere, all at t={t}:")
    print(
        f"  median score of a CORRECT BENT line (straightness < 0.85):  "
        f"{np.median(scores[bent & ok]):.3f}"
    )
    print(
        f"  median score of a WRONG STRAIGHT line (straightness > 0.99): "
        f"{np.median(scores[straight & ~ok]):.3f}"
    )
    print(
        f"  share of correct bent lines kept at t={t}:      "
        f"{np.mean(scores[bent & ok] >= t):.3f}  "
        f"(so {1 - np.mean(scores[bent & ok] >= t):.0%} are thrown away)"
    )
    print(
        f"  share of correct straight lines kept at t={t}:  "
        f"{np.mean(scores[straight & ok] >= t):.3f}"
    )


if __name__ == "__main__":
    root = DATA_ROOT / "pdok_dataset3_polylines"
    cfg = dict(
        run_dirs=[
            root / "inference/best_2_val_cluster_t0.05",  # exp 2, softmax head
            root / "inference/best_3_val_cluster_t0.05",  # exp 3, focal head
        ],
        buffer_m=10,
        min_cov=0.8,  # share of a prediction's length that must sit on a label
        thresholds=[0.05, 0.50, 0.80, 0.90, 0.95, 0.98],
    )
    main(cfg)
