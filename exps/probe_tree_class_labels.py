"""
Did the two classes end up on the right layers?

scripts/data/convert_pdok_polylines_to_detr_polyline.py writes hedges as class 0
and Top10NL tree rows (bomenrij) as class 1. A swap would train the model to
call every hedge a tree row, and nothing else in the pipeline would notice: the
shapes stay right, the loss trains, and the figures look plausible because both
classes are woody lines.

This checks each class against the layer it came from. For every crop, sample
the stored polylines of one class and ask how much of that length sits within
10 m of a feature of each Top10NL layer. Class 0 must sit on heg and class 1 on
bomenrij.

The cross rows are the control. They are not zero, because the two layers do run
near each other sometimes, but they must be far below the matching row or the
check cannot fail. They also say how much the two classes blur at each buffer,
which is why the 5 m score is the cleaner one to compare classes at.

Result (2026-09-16, pdok_dataset3_tree_polylines, 300 val crops):

    buffer   class 0 on heg   class 0 on bomenrij   class 1 on bomenrij   class 1 on heg
      5 m         100.0%              1.7%                 100.0%              2.4%
     10 m         100.0%             11.0%                 100.0%             12.2%

So at 10 m about an eighth of each class sits on the other layer as well, and at
5 m only about a fiftieth.

    /home/fatemeh/miniconda3/envs/hedge/bin/python exps/probe_tree_class_labels.py
"""

import json
import random
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from probe_treeline_overlap import TOPO, lines_in_crop  # noqa: E402

from hedge_seg.metrics import densify, meters_to_px  # noqa: E402
from hedge_seg.paths import DATA_ROOT  # noqa: E402


def main(cfg):
    stems = sorted(p.stem for p in Path(cfg["polyline_dir"]).glob("*.npz"))
    random.Random(cfg["seed"]).shuffle(stems)
    stems = stems[: cfg["n_crops"]]

    layers = {
        "heg": gpd.read_file(TOPO / cfg["hedge_shp"]),
        "bomenrij": gpd.read_file(TOPO / cfg["treeline_shp"]),
    }
    print(
        f"{len(stems)} crops, layers: "
        + ", ".join(f"{k} {len(v)}" for k, v in layers.items())
    )

    # covered[buffer][class][layer] = (points within radius, points total)
    covered = {
        b: {c: {name: [0, 0] for name in layers} for c in cfg["classes"]}
        for b in cfg["buffers_m"]
    }
    for stem in stems:
        bbox = json.loads((Path(cfg["label_dir"]) / f"{stem}.json").read_text())[
            "bbox_world"
        ]
        npz = np.load(Path(cfg["polyline_dir"]) / f"{stem}.npz")
        polylines, labels = npz["polylines"].astype(float), npz["labels"]

        layer_points = {
            name: densify(lines) if lines else None
            for name, lines in (
                (name, lines_in_crop(gdf, bbox)) for name, gdf in layers.items()
            )
        }
        for class_id in cfg["classes"]:
            lines = polylines[labels == class_id]
            if len(lines) == 0:
                continue
            points = densify(lines)
            for name, target_points in layer_points.items():
                if target_points is None:
                    continue
                distance, _ = cKDTree(target_points).query(points)
                for buffer_m in cfg["buffers_m"]:
                    hit = int((distance <= meters_to_px(buffer_m)).sum())
                    covered[buffer_m][class_id][name][0] += hit
                    covered[buffer_m][class_id][name][1] += len(points)

    print("\nshare of stored line length within r of a layer")
    for buffer_m in cfg["buffers_m"]:
        for class_id, name in cfg["classes"].items():
            row = " ".join(
                f"{layer}={hit / max(total, 1):6.1%}"
                for layer, (hit, total) in covered[buffer_m][class_id].items()
            )
            print(f"  r={buffer_m:2d} m  class {class_id} ({name:8s}) {row}")
    print("  each class must be near 100% on its own layer")


if __name__ == "__main__":
    cfg = dict(
        polyline_dir=DATA_ROOT / "pdok_dataset3_tree_polylines/polylines/val",
        label_dir=DATA_ROOT / "pdok_dataset3/labels",
        classes={0: "hedge", 1: "tree row"},
        hedge_shp="Hedges_polylines/Top10NL2023_inrichtingselementen_lijn_heg.shp",
        treeline_shp="Treelines_polylines/"
        "Top10NL2023_inrichtingselementen_lijn_bomenrij.shp",
        buffers_m=(5, 10),
        n_crops=300,
        seed=0,
    )
    main(cfg)
