"""
Score an inference run of scripts/train_detr_unet_polyline.py.

Reports buffered-length precision and recall (hedge_seg/metrics.py) per image,
averaged over images, at 5, 10 and 15 m. Also runs the checks that decided how
the metric had to be built, so the reasons stay reproducible: the rejected
chamfer + Hungarian variant, the score threshold sweep, GT merging, prediction
straightness, and the score with campsite crops excluded.

An inference run directory already holds `polylines/` and `gt/`, so only the
run directories have to be listed. Edit the cfg at the bottom and run:

    /home/fatemeh/miniconda3/envs/hedge/bin/python exps/probe_polyline_pr.py

Results on the cluster val split, 3,098 crops. Full numbers in
docs/experiment_log.md.

- exp 2 (26,902 train crops) against exp 1 (5,000), both at t=0.95, at 10 m:
  0.783/0.608 F1 0.685 against 0.702/0.589 F1 0.640. At 15 m 0.845/0.674
  F1 0.750. The extra data bought precision (+0.08) far more than recall
  (+0.02), and nothing at all on crowded crops.
- exp 2's best threshold is 0.90, not 0.95: 0.701/0.696 F1 0.699 at 10 m. The
  optimum moves with the model, so re-sweep after any change.
- Buffered length at 10 m within exp 1: best_1.pt 0.75/0.52 (F1 0.614),
  1_150.pt 0.70/0.59 (F1 0.640). 1_150.pt wins at every buffer, so the
  checkpoint the eval loss calls overfit is the better detector. Rank by this
  metric, not by eval loss.
- The same predictions under chamfer + Hungarian: F1 0.41. It scores a
  prediction covering one leg of an L-shaped GT hedge as a total miss. The
  figures agree with 0.64, not with 0.41.
- Threshold: 1_150.pt peaks at 0.95, which is what the run already uses. But at
  0.05 recall is 0.75 while precision falls to 0.41. The correct lines are
  already being drawn and then discarded by a score that cannot rank them, so
  recall is limited by the classification head, not by perception.
- Recall falls with crop density: 0.71 on single-hedge crops, 0.51 on 2-3,
  0.42 on 4-6, 0.38 on 7+. Precision is flat near 0.70 in every bucket. The
  gap is recall, and it is a density problem.
- Excluding the 315 campsite crops moves F1 from 0.640 to 0.652. Those crops
  really are worse (0.542), but at 10% of crops they cannot move a per-image
  average much. Not worth changing the dataset for.
- Merging GT lines within 10 m of each other removes 0.8% of GT lines and moves
  F1 by 0.001. The "one hedge split into several GT lines" artifact does not
  matter in aggregate.
- Predicted straightness 0.89 against GT 0.91, and predicted median length
  116 m against GT 120 m. Predictions bend and stretch like the labels do.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hedge_seg.metrics import (  # noqa: E402
    PDOK_PIXEL_SIZE_M,
    buffered_length_pr,
    f1,
    macro_average,
    matched_pr,
    merge_close_polylines,
    meters_to_px,
    polyline_length,
    straightness,
)
from hedge_seg.paths import DATA_ROOT  # noqa: E402

BUFFERS_M = (5, 10, 15)
GT_BUCKETS = ((1, 1), (2, 3), (4, 6), (7, 10_000))
SWEEP = (0.05, 0.2, 0.4, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98)


def load_run(run_dir):
    """Read a whole inference run into memory once, so sweeps stay cheap."""
    run_dir = Path(run_dir)
    if not (run_dir / "polylines").is_dir():
        raise FileNotFoundError(f"Not an inference run directory: {run_dir}")
    items = []
    for path in sorted((run_dir / "polylines").glob("*.npz")):
        pred = np.load(path)
        gt = np.load(run_dir / "gt" / path.name)
        items.append(
            {
                "stem": path.stem,
                "pred": pred["polylines"].astype(np.float64),
                "scores": pred["scores"] if "scores" in pred else None,
                "gt": gt["polylines"].astype(np.float64),
            }
        )
    return items


def score(items, score_thresh=None, exclude=frozenset(), merge_m=None):
    """Per-image rows. Aggregate them with macro_average, never as a total."""
    rows = []
    for item in items:
        if item["stem"] in exclude:
            continue
        pred = item["pred"]
        if score_thresh is not None and item["scores"] is not None:
            pred = pred[item["scores"] >= score_thresh]
        gt = list(item["gt"])
        if merge_m:
            gt = merge_close_polylines(gt, meters_to_px(merge_m))
        row = {"stem": item["stem"], "n_pred": len(pred), "n_gt": len(gt)}
        for buf in BUFFERS_M:
            radius = meters_to_px(buf)
            precision, recall = buffered_length_pr(pred, gt, radius)
            row[f"p{buf}"] = precision
            row[f"r{buf}"] = recall
        rows.append(row)
    return rows


def score_chamfer(items, score_thresh=None, exclude=frozenset()):
    rows = []
    for item in items:
        if item["stem"] in exclude:
            continue
        pred = item["pred"]
        if score_thresh is not None and item["scores"] is not None:
            pred = pred[item["scores"] >= score_thresh]
        row = {"stem": item["stem"]}
        for buf in BUFFERS_M:
            precision, recall, _ = matched_pr(
                list(pred), list(item["gt"]), meters_to_px(buf)
            )
            row[f"p{buf}"] = precision
            row[f"r{buf}"] = recall
        rows.append(row)
    return rows


def report(rows, label, buffers=BUFFERS_M):
    for buf in buffers:
        precision = macro_average(rows, f"p{buf}")
        recall = macro_average(rows, f"r{buf}")
        print(
            f"  {label + f' @{buf}m':<34} n={len(rows):>5}  "
            f"P={precision:.3f} R={recall:.3f} F1={f1(precision, recall):.3f}"
        )


def report_geometry(items, score_thresh=None):
    """Do predictions bend and stretch the way the labels do?"""
    for key, name in (("gt", "GT"), ("pred", "pred")):
        straight, lengths = [], []
        for item in items:
            lines = item[key]
            if (
                key == "pred"
                and score_thresh is not None
                and item["scores"] is not None
            ):
                lines = lines[item["scores"] >= score_thresh]
            for line in lines:
                straight.append(straightness(line))
                lengths.append(polyline_length(line) * PDOK_PIXEL_SIZE_M)
        straight, lengths = np.asarray(straight), np.asarray(lengths)
        print(
            f"  {name:<5} n={len(straight):>6}  straightness mean={straight.mean():.3f}"
            f"  bent<0.85={np.mean(straight < 0.85):.3f}"
            f"  length median={np.median(lengths):.0f} m mean={lengths.mean():.0f} m"
        )


def main(cfg):
    exclude = frozenset()
    if cfg.get("recreation_stems") and Path(cfg["recreation_stems"]).exists():
        exclude = frozenset(Path(cfg["recreation_stems"]).read_text().split())
        print(
            f"{len(exclude)} campsite crops available to exclude "
            f"(exps/probe_recreation_crops.py)\n"
        )

    thresh = cfg["report_thresh"]
    for run_dir in cfg["run_dirs"]:
        run_dir = Path(run_dir)
        items = load_run(run_dir)
        rows = score(items, score_thresh=thresh)
        print(f"=== {run_dir.name}  ({len(items)} crops, reported at t={thresh}) ===")

        print(" buffered length, all crops")
        report(rows, "buffered")

        print(" by GT line count, 10 m")
        for lo, hi in GT_BUCKETS:
            sub = [r for r in rows if lo <= r["n_gt"] <= hi]
            share = sum(r["n_gt"] for r in sub) / sum(r["n_gt"] for r in rows)
            report(
                sub, f"gt {lo}-{hi if hi < 1000 else '+'} ({share:.0%} of lines)", (10,)
            )

        if exclude:
            print(" campsite crops excluded")
            report(score(items, score_thresh=thresh, exclude=exclude), "rural only")
            report([r for r in rows if r["stem"] in exclude], "campsites only")

        if cfg["sweep"] and items[0]["scores"] is not None:
            print(" score threshold sweep, 10 m")
            for swept_thresh in SWEEP:
                swept = score(items, score_thresh=swept_thresh)
                n_pred = np.mean([r["n_pred"] for r in swept])
                report(swept, f"t={swept_thresh:<5} {n_pred:.2f} pred/img", (10,))

        if cfg["chamfer"]:
            print(" rejected chamfer + Hungarian variant, same predictions")
            report(score_chamfer(items, score_thresh=thresh), "chamfer")

        if cfg["merge_gt"]:
            print(" GT polylines within 10 m merged")
            merged = score(items, score_thresh=thresh, merge_m=10)
            dropped = 1 - sum(r["n_gt"] for r in merged) / sum(r["n_gt"] for r in rows)
            print(f"  {dropped:.1%} of GT lines removed by merging")
            report(merged, "merged GT")

        if cfg["geometry"]:
            print(" polyline geometry")
            report_geometry(items, score_thresh=thresh)
        print()


if __name__ == "__main__":
    root = DATA_ROOT / "pdok_dataset3_polylines"
    cfg = dict(
        run_dirs=[
            root / "inference/best_2_val_cluster_t0.05",  # exp 2, softmax head
            root / "inference/best_3_val_cluster_t0.05",  # exp 3, focal head
        ],
        # Written by exps/probe_recreation_crops.py. Missing file just skips
        # the campsite rows.
        recreation_stems=root / "recreation_crops_val_cluster.txt",
        # Headline threshold. The run directories were inferred at 0.05 so the
        # sweep has something to sweep; 0.95 is the operating point run 1 uses.
        report_thresh=0.95,
        sweep=True,  # needs a run inferred at a low threshold
        # chamfer and merge_gt validate the metric itself, not a run. They were
        # settled on exp 1 (F1 0.41 against 0.64, and 0.001) and cost about
        # 3 min per run directory, so leave them off unless the metric is being
        # questioned again.
        chamfer=False,
        merge_gt=False,
        geometry=True,
    )
    main(cfg)
