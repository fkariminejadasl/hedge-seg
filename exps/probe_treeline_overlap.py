"""
How much of what the model gets "wrong" is actually a tree line?

Top10NL splits linear woody features into two layers. Training uses only
`inrichtingselementen_lijn_heg` ("heg, haag"). Tree rows live in
`inrichtingselementen_lijn_bomenrij` and are not in the labels at all. A tree
row looks much like a hedgerow from above, so the model draws it, and the
metric counts every metre of it as a false positive.

This takes the predicted length that is NOT within the buffer of any training
label, and asks how much of it lands on a tree line instead.

Result (2026-07-29, exp 2 `best_2.pt` at t=0.95, 10 m buffer):

- sanity: 100% of the training GT sits on the heg layer, so the crop clipping
  and the pixel-to-world transform are right.
- 300 random val crops: **24.3%** of the unmatched predicted length is within
  10 m of a tree line. Only 1.2% is near another heg, so the labels are not
  simply missing hedges that the layer already has.
- 40 crops picked by the missing-label rule: 21.3%, the same picture.

So about a quarter of the reported false-positive length is the model
correctly finding a woody line that the labels never contained. Precision at
10 m is 0.783 with 0.217 unmatched, and 24.3% of that is 0.053 of length. That
is the size of the prize for adding tree lines as a second class, and it is
also why the reported precision is a lower bound on hedgerow performance.

    /home/fatemeh/miniconda3/envs/hedge/bin/python exps/probe_treeline_overlap.py
"""

import json
import random
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
from scipy.spatial import cKDTree
from shapely.geometry import box

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hedge_seg.metrics import PDOK_PIXEL_SIZE_M, densify, meters_to_px  # noqa: E402
from hedge_seg.paths import DATA_ROOT  # noqa: E402

TOPO = Path("/home/fatemeh/Downloads/hedge/Topo10NL2023")


def world_to_px(line, bbox):
    """RD New metres to crop pixels. Row grows downward, so y is flipped."""
    minx, _, _, maxy = bbox
    xs, ys = np.asarray(line.coords.xy[0]), np.asarray(line.coords.xy[1])
    return np.stack(
        [(xs - minx) / PDOK_PIXEL_SIZE_M, (maxy - ys) / PDOK_PIXEL_SIZE_M], axis=1
    )


def lines_in_crop(gdf, bbox):
    """Every line of a layer that touches this crop, in crop pixel coordinates."""
    hits = gdf.iloc[list(gdf.sindex.query(box(*bbox), predicate="intersects"))]
    out = []
    for geom in hits.geometry:
        parts = [geom] if geom.geom_type == "LineString" else list(geom.geoms)
        out.extend(a for a in (world_to_px(p, bbox) for p in parts) if len(a) >= 2)
    return out


def main(cfg):
    run_dir = Path(cfg["run_dir"])
    radius = meters_to_px(cfg["buffer_m"])

    stems = sorted(p.stem for p in (run_dir / "polylines").glob("*.npz"))
    random.Random(cfg["seed"]).shuffle(stems)
    stems = stems[: cfg["n_crops"]]
    print(f"{len(stems)} crops from {run_dir.name} at t={cfg['score_thresh']}")

    heg = gpd.read_file(TOPO / cfg["hedge_shp"])
    tree = gpd.read_file(TOPO / cfg["treeline_shp"])
    print(f"heg {len(heg)}, bomenrij {len(tree)}")

    n_unmatched = n_on_tree = n_on_heg = 0
    gt_on_layer = []
    for stem in stems:
        bbox = json.loads(
            (DATA_ROOT / f"pdok_dataset3/labels/{stem}.json").read_text()
        )["bbox_world"]
        npz = np.load(run_dir / "polylines" / f"{stem}.npz")
        pred = npz["polylines"][npz["scores"] >= cfg["score_thresh"]].astype(float)
        gt = np.load(run_dir / "gt" / f"{stem}.npz")["polylines"].astype(float)
        if len(pred) == 0 or len(gt) == 0:
            continue

        pred_pts, gt_pts = densify(pred), densify(gt)
        heg_lines, tree_lines = lines_in_crop(heg, bbox), lines_in_crop(tree, bbox)

        # Sanity: the training labels must lie on the heg layer they came from.
        if heg_lines:
            d, _ = cKDTree(densify(heg_lines)).query(gt_pts)
            gt_on_layer.append(float((d <= radius).mean()))

        d, _ = cKDTree(gt_pts).query(pred_pts)
        unmatched = pred_pts[d > radius]
        n_unmatched += len(unmatched)
        if len(unmatched) == 0:
            continue
        for lines, counter in ((tree_lines, "tree"), (heg_lines, "heg")):
            if not lines:
                continue
            d, _ = cKDTree(densify(lines)).query(unmatched)
            hit = int((d <= radius).sum())
            if counter == "tree":
                n_on_tree += hit
            else:
                n_on_heg += hit

    print(f"\nsanity, GT on the heg layer: {np.mean(gt_on_layer):.1%} (want ~100%)")
    print(f"unmatched predicted length: {n_unmatched} sample points at 1 m spacing")
    print(f"  within {cfg['buffer_m']} m of a tree line: {n_on_tree / n_unmatched:.1%}")
    print(f"  within {cfg['buffer_m']} m of another heg: {n_on_heg / n_unmatched:.1%}")


if __name__ == "__main__":
    cfg = dict(
        run_dir=DATA_ROOT
        / "pdok_dataset3_polylines/inference/best_2_val_cluster_t0.05",
        hedge_shp="Hedges_polylines/Top10NL2023_inrichtingselementen_lijn_heg.shp",
        treeline_shp="Treelines_polylines/"
        "Top10NL2023_inrichtingselementen_lijn_bomenrij.shp",
        score_thresh=0.95,
        buffer_m=10,
        n_crops=300,  # random sample; the full 3,098 takes about 10x longer
        seed=0,
    )
    main(cfg)
