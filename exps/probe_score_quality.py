"""
Does the score say a prediction is right, or only that it is straight?

The score head is the suspected limit on recall. This asks what the score
actually ranks by. A prediction counts as correct when at least `min_cov` of
its length lies within `buffer_m` of a label, which is the same buffered-length
idea as hedge_seg/metrics.py, applied per prediction instead of per image.

Result (2026-08-10, exp 2 `best_2.pt`, all 3,098 val crops from the t=0.05 run,
10 m buffer): 7,956 of 39,104 predictions are correct.

    straightness      median score, correct   median score, wrong
    < 0.85 (bent)              0.453                  0.216
    0.85 to 0.95               0.883                  0.690
    0.95 to 0.99               0.951                  0.868
    > 0.99 (straight)          0.974                  0.957

- A correct bent line scores 0.45, a wrong straight one scores 0.96, so no
  single threshold keeps both. At t=0.95, 9% of correct bent predictions
  survive against 74% of correct straight ones.
- AUC of the score for correct against wrong is 0.767 overall, but only 0.618
  inside the bent group. Most of its apparent skill is straightness as a proxy.
- Predicted straightness at t=0.80 is 0.908 against 0.909 in the labels, so the
  "exp 2 draws straighter lines than the labels" regression was the 0.95 cut,
  not the model.

This is the evidence for exp 3 being a focal sigmoid head rather than a change
to eos_coef, which scales the no-object column uniformly and so cannot make the
score track quality.

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
    run_dir = Path(cfg["run_dir"])
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


if __name__ == "__main__":
    cfg = dict(
        run_dir=DATA_ROOT
        / "pdok_dataset3_polylines/inference/best_2_val_cluster_t0.05",
        buffer_m=10,
        min_cov=0.8,  # share of a prediction's length that must sit on a label
        thresholds=[0.05, 0.50, 0.80, 0.90, 0.95, 0.98],
    )
    main(cfg)
