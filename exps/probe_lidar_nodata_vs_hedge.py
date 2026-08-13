"""
What does nodata mean in the AHN4 metrics?

Nearly half the lidar cells have no value, which looks like broken data until
you read how the metrics are made: 24 of the 25 are computed from vegetation
returns only, so a cell with nothing woody in it has nothing to compute from.
This checks that reading by comparing the cells a hedge passes through with the
rest.

Result (2026-08-11, 300 crops, `lidar_patches.npy`):

    nodata share, all six metrics          about 48.4%
    nodata in cells a hedge passes through       7.2%
    nodata in every other cell                  51.8%

So nodata means "nothing growing here", not "measurement missing". Two things
follow, both used in `DetrPolylineImageDataset._lidar_on_padded_grid`:

- filling the metrics with 0 is right rather than a fudge, because no
  vegetation is zero density and ground-level height
- the presence channel is not bookkeeping. It is probably the most useful
  channel of the seven, since "is anything growing here" is most of what a
  10 m grid can say about a 3 m hedge

    /home/fatemeh/miniconda3/envs/hedge/bin/python exps/probe_lidar_nodata_vs_hedge.py
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hedge_seg.paths import DATA_ROOT  # noqa: E402


def hedge_cell_mask(polylines, cells, crop_px):
    """Cells that a hedge passes through, densified so no cell is skipped."""
    mask = np.zeros((cells, cells), dtype=bool)
    scale = crop_px / cells  # crop pixels per lidar cell
    for polyline in polylines:
        p = np.asarray(polyline, dtype=float)
        for a, b in zip(p[:-1], p[1:]):
            pts = a + np.linspace(0, 1, 100)[:, None] * (b - a)
            c = np.clip((pts[:, 0] / scale).astype(int), 0, cells - 1)
            r = np.clip((pts[:, 1] / scale).astype(int), 0, cells - 1)
            mask[r, c] = True
    return mask


def main(cfg):
    patches_path = Path(cfg["patches"])
    patches = np.load(patches_path, mmap_mode="r")
    meta = json.loads(
        patches_path.with_name(patches_path.stem + "_stems.json").read_text()
    )

    print(f"{patches.shape[0]} crops, nodata share per metric:")
    for c, metric in enumerate(meta["metrics"]):
        print(f"  {metric:36s} {np.isnan(patches[:, c]).mean():.1%}")

    cells = patches.shape[-1]
    on, off = [], []
    for i, stem in enumerate(meta["stems"]):
        polyline_path = next(
            (
                p
                for d in cfg["polyline_dirs"]
                if (p := Path(d) / f"{stem}.npz").exists()
            ),
            None,
        )
        if polyline_path is None:
            continue
        polylines = np.load(polyline_path)["polylines"].astype(float)
        if len(polylines) == 0:
            continue
        mask = hedge_cell_mask(polylines, cells, cfg["crop_px"])
        nodata = np.isnan(patches[i, cfg["metric_index"]])
        on.append(nodata[mask].mean())
        off.append(nodata[~mask].mean())

    print(f"\nusing {meta['metrics'][cfg['metric_index']]}, {len(on)} crops")
    print(f"  nodata in cells a hedge passes through: {np.mean(on):.1%}")
    print(f"  nodata in every other cell:             {np.mean(off):.1%}")


if __name__ == "__main__":
    root = DATA_ROOT / "pdok_dataset3_polylines"
    cfg = dict(
        patches=root / "lidar_patches.npy",
        polyline_dirs=[root / "polylines/train", root / "polylines/val"],
        metric_index=0,  # all six behave the same, so one is enough
        crop_px=1000,  # crop side in pixels, 250 m at 0.25 m
    )
    main(cfg)
