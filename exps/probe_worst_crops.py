"""
List the crops worth looking at by eye: worst false positives, worst false
negatives, and crops where the labels are probably wrong.

Ranking is by *length*, not by line count, because one 300 m line and one 20 m
stub are not the same mistake. Per crop, at a 10 m buffer:

    unmatched predicted length = predicted length * (1 - precision)
    unmatched GT length        = GT length * (1 - recall)

- worst_fp: most predicted length that is nowhere near a labelled hedge.
- worst_fn: most labelled hedge length with nothing predicted near it.
- missing_labels: high recall and low precision at once. The model found what
  was labelled, so it is not failing here, yet it also drew a lot more. Those
  extra lines are usually real hedges the labels do not have. This is the
  "hedges missing from the labels" artifact in docs/lesson_learned.md, which
  caps precision and makes the reported number a lower bound.

Writes one id per line, e.g. `28000`, so the lists can be pasted into the `ids`
field of scripts/show_polyline_results.py and looked at directly.

    /home/fatemeh/miniconda3/envs/hedge/bin/python exps/probe_worst_crops.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hedge_seg.metrics import (  # noqa: E402
    PDOK_PIXEL_SIZE_M,
    buffered_length_pr,
    meters_to_px,
    polyline_length,
)
from hedge_seg.paths import DATA_ROOT  # noqa: E402


def total_length_m(polylines):
    return sum(polyline_length(p) for p in polylines) * PDOK_PIXEL_SIZE_M


def score_crops(run_dir, score_thresh, buffer_m):
    run_dir = Path(run_dir)
    radius = meters_to_px(buffer_m)
    rows = []
    for path in sorted((run_dir / "polylines").glob("*.npz")):
        pred_npz = np.load(path)
        pred = pred_npz["polylines"].astype(np.float64)
        if "scores" in pred_npz:
            pred = pred[pred_npz["scores"] >= score_thresh]
        gt = np.load(run_dir / "gt" / path.name)["polylines"].astype(np.float64)

        precision, recall = buffered_length_pr(pred, gt, radius)
        pred_m, gt_m = total_length_m(pred), total_length_m(gt)
        rows.append(
            {
                "id": int(path.stem.split("_")[-1]),
                "precision": precision,
                "recall": recall,
                "pred_m": pred_m,
                "gt_m": gt_m,
                "fp_m": pred_m * (1.0 - precision),
                "fn_m": gt_m * (1.0 - recall),
            }
        )
    return rows


def write_ids(rows, path, label):
    Path(path).write_text("\n".join(str(r["id"]) for r in rows) + "\n")
    print(f"\n{label} -> {path}")
    print("     id   prec   rec   pred_m    gt_m    fp_m    fn_m")
    for r in rows[:12]:
        print(
            f"  {r['id']:>6}  {r['precision']:.2f}  {r['recall']:.2f}"
            f"  {r['pred_m']:>7.0f} {r['gt_m']:>7.0f}"
            f" {r['fp_m']:>7.0f} {r['fn_m']:>7.0f}"
        )
    print(f"  ids: {', '.join(str(r['id']) for r in rows[:20])}")


def main(cfg):
    rows = score_crops(cfg["run_dir"], cfg["score_thresh"], cfg["buffer_m"])
    out = Path(cfg["out_dir"])
    tag = Path(cfg["run_dir"]).name
    n = cfg["n"]
    print(
        f"{len(rows)} crops from {tag} at t={cfg['score_thresh']}, "
        f"{cfg['buffer_m']} m buffer"
    )

    worst_fp = sorted(rows, key=lambda r: -r["fp_m"])[:n]
    write_ids(worst_fp, out / f"worst_fp_{tag}.txt", "worst false positives")

    worst_fn = sorted(rows, key=lambda r: -r["fn_m"])[:n]
    write_ids(worst_fn, out / f"worst_fn_{tag}.txt", "worst false negatives")

    # High recall means the model did find the labelled hedges, so the extra
    # length it drew is more likely a missing label than a model error.
    missing = [
        r
        for r in rows
        if r["recall"] >= cfg["missing_min_recall"]
        and r["precision"] <= cfg["missing_max_precision"]
        and r["fp_m"] >= cfg["missing_min_fp_m"]
    ]
    missing = sorted(missing, key=lambda r: -r["fp_m"])[:n]
    write_ids(missing, out / f"missing_labels_{tag}.txt", "probably missing labels")

    # The opposite end, for showing what the model does well. Sorted by how much
    # hedge is in the crop, so these are busy crops that were found anyway, not
    # single short lines that are easy by default.
    best = [
        r
        for r in rows
        if r["precision"] >= cfg["best_min_precision"]
        and r["recall"] >= cfg["best_min_recall"]
        and r["gt_m"] >= cfg["best_min_gt_m"]
    ]
    best = sorted(best, key=lambda r: -r["gt_m"])[:n]
    write_ids(best, out / f"best_{tag}.txt", "best crops")
    print(
        f"\n{len(missing)} crops match the missing-label rule "
        f"(recall >= {cfg['missing_min_recall']}, "
        f"precision <= {cfg['missing_max_precision']}, "
        f"unmatched predicted length >= {cfg['missing_min_fp_m']} m)"
    )


if __name__ == "__main__":
    root = DATA_ROOT / "pdok_dataset3_polylines"
    cfg = dict(
        run_dir=root / "inference/best_2_val_cluster_t0.05",
        out_dir=root,
        score_thresh=0.95,  # exp 2 optimum is 0.90, 0.95 matches the reported table
        buffer_m=10,
        n=40,  # ids per list
        missing_min_recall=0.80,
        missing_max_precision=0.50,
        missing_min_fp_m=100.0,
        best_min_precision=0.85,
        best_min_recall=0.85,
        best_min_gt_m=300.0,  # busy crops only, so "good" is not just an easy crop
    )
    main(cfg)
