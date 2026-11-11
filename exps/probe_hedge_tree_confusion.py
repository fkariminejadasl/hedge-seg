"""
Does a two-class model find the hedges but call them tree rows?

Scores exps 2, 4 and 5 at t=0.90 and 10 m against both label layers of the
tree row dataset, whatever class each run predicts. Exp 2 predicts only class
0, so all its lines count as hedge. Tree row recall is averaged only over the
crops that have a tree row, 1,874 of 3,098. exps/probe_polyline_pr.py gives a
crop with no tree row a perfect 1.0, which is why its tree row numbers are too
high (exp 5: 0.564 there).

    /home/fatemeh/miniconda3/envs/hedge/bin/python exps/probe_hedge_tree_confusion.py

Result, t=0.90, 10 m:

- Tree row recall on crops with a tree row: exp 5 0.280, exp 4 0.425.
- Exp 5 draws a line on more hedge length than exp 2 when all its lines
  count, 0.714 against 0.696, but calls some of them tree rows: counting only
  lines it calls hedge, 0.686.
- It still calls tree rows hedges as often as exp 2: 0.061 of its
  hedge-named length lies on a tree row only, against 0.059. So it mixes the
  two up both ways, and the hedge score does not gain.
- Woody lines, every line against both layers: exp 2 F1 0.633, exp 5 0.654,
  exp 4 0.690. Exp 5 matches exps/probe_polyline_pr.py.
"""

import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "exps"))

from probe_polyline_pr import load_run  # noqa: E402

from hedge_seg.metrics import (  # noqa: E402
    buffered_length_pr,
    densify,
    f1,
    macro_average,
    meters_to_px,
)
from hedge_seg.paths import DATA_ROOT  # noqa: E402

R = meters_to_px(10)
T = 0.90


def near(points, ref):
    if len(points) == 0:
        return np.zeros(0, bool)
    if len(ref) == 0:
        return np.zeros(len(points), bool)
    return cKDTree(ref).query(points)[0] <= R


tree_dir = DATA_ROOT / "pdok_dataset3_tree_polylines/inference"
labels_by_stem = {
    it["stem"]: (it["gt"], it["gt_labels"])
    for it in load_run(tree_dir / "best_5_val_t0.05")
}
runs = {
    "exp 2": DATA_ROOT / "pdok_dataset3_polylines/inference/best_2_val_cluster_t0.05",
    "exp 5": tree_dir / "best_5_val_t0.05",
    "exp 4": tree_dir / "best_4_val_t0.05",
}
for name, run in runs.items():
    rows = []
    share = {k: 0 for k in ("h_on_h", "h_on_t_only", "h_on_none", "h_total")}
    for it in load_run(run):
        gt, gl = labels_by_stem[it["stem"]]
        gt_h, gt_t = gt[gl == 0], gt[gl == 1]
        keep = it["scores"] >= T
        lab = it["pred_labels"]
        pred_all = it["pred"][keep]
        pred_h = it["pred"][keep & (lab == 0)]
        pred_t = it["pred"][keep & (lab == 1)]
        row = {
            "r_h_by_h": buffered_length_pr(pred_h, list(gt_h), R)[1],
            "r_h_by_any": buffered_length_pr(pred_all, list(gt_h), R)[1],
        }
        p, r = buffered_length_pr(pred_all, list(gt_h) + list(gt_t), R)
        row["p_woody"], row["r_woody"] = p, r
        if len(gt_t):
            row["r_t_by_t"] = buffered_length_pr(pred_t, list(gt_t), R)[1]
            row["r_t_by_any"] = buffered_length_pr(pred_all, list(gt_t), R)[1]
        rows.append(row)
        pts = densify(pred_h)
        on_h, on_t = near(pts, densify(gt_h)), near(pts, densify(gt_t))
        share["h_on_h"] += on_h.sum()
        share["h_on_t_only"] += (on_t & ~on_h).sum()
        share["h_on_none"] += (~on_t & ~on_h).sum()
        share["h_total"] += len(pts)

    tree_rows = [r for r in rows if "r_t_by_t" in r]
    pw, rw = macro_average(rows, "p_woody"), macro_average(rows, "r_woody")
    n = share["h_total"]
    print(
        f"=== {name}, t={T}, 10 m, {len(rows)} crops ({len(tree_rows)} with a tree row)"
    )
    print(
        f"  hedge recall: by hedge lines {macro_average(rows, 'r_h_by_h'):.3f}, "
        f"by lines of any class {macro_average(rows, 'r_h_by_any'):.3f}"
    )
    if name != "exp 2":
        print(
            f"  tree recall:  by tree lines {macro_average(tree_rows, 'r_t_by_t'):.3f}, "
            f"by lines of any class {macro_average(tree_rows, 'r_t_by_any'):.3f}"
        )
    else:
        print(
            f"  tree recall:  by lines of any class {macro_average(tree_rows, 'r_t_by_any'):.3f}"
        )
    print(
        f"  hedge-labelled length: on a hedge {share['h_on_h'] / n:.3f}, "
        f"on a tree row only {share['h_on_t_only'] / n:.3f}, "
        f"on neither {share['h_on_none'] / n:.3f}"
    )
    print(
        f"  woody line, any class against both: P={pw:.3f} R={rw:.3f} F1={f1(pw, rw):.3f}"
    )
