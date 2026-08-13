"""
Cut the AHN4 lidar metrics into one small patch per crop.

Each crop is 250 m, each lidar cell is 10 m, so a patch is 25x25 cells per
metric. Everything is written into a single .npy plus a stem index, not one
file per crop: 30,000 crops would otherwise cost 30,000 inodes, and inodes are
the tight resource on the cluster. The array is small enough to memory-map, so
the dataset reads one crop at a time without loading the whole thing.

Alignment of the patches to the crops is checked by
`exps/probe_lidar_crop_alignment.py`, which finds the best whole-cell offset to
be (0, 0). Run that after changing anything here.

Output, next to the polyline directory:

    lidar_patches.npy         (N, C, 25, 25) float32, NaN where the raster has
                              no data, in the metric order of cfg["metrics"]
    lidar_patches_stems.json  {"stems": [...], "metrics": [...], "cell_m": 10}

One file covering every crop, looked up by stem, so it serves the laptop split
and the cluster split alike. About 26 min and 450 MB for all 30,000 crops, so
`cfg.limit` builds a subset when only a check is needed.

    /home/fatemeh/miniconda3/envs/hedge/bin/python scripts/data/build_lidar_patches.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import rasterio
from omegaconf import OmegaConf
from rasterio.windows import from_bounds
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from hedge_seg.paths import DATA_ROOT  # noqa: E402

LIDAR = Path("/home/fatemeh/Downloads/hedge/LiDAR_metrics_AHN4")


def read_patch(src, bbox_world, cells):
    """One metric over one crop, nodata as NaN, always (cells, cells)."""
    window = from_bounds(*bbox_world, transform=src.transform)
    patch = src.read(
        1, window=window, boundless=True, fill_value=np.nan, out_shape=(cells, cells)
    ).astype(np.float32)
    # Each raster carries its own nodata, and the band ratios use +3.4e38 while
    # the height rasters use -3.4e38 or -99999, so one blanket filter is wrong.
    if src.nodata is not None:
        patch[np.isclose(patch, src.nodata, rtol=1e-6)] = np.nan
    patch[np.abs(patch) > 1e30] = np.nan
    return patch


def main(cfg):
    label_dir = Path(cfg.label_dir)
    # Every crop of the dataset, not one file per split. The train/val split
    # differs between the laptop and the cluster (avoid_label_dirs sees a
    # different number of labels), and a stem-keyed file is immune to that.
    stems = sorted(p.stem for p in label_dir.glob("pos_*.json"))
    if not stems:
        raise FileNotFoundError(f"No label JSONs in {label_dir}")
    if cfg.limit:
        stems = stems[: cfg.limit]

    out = np.empty(
        (len(stems), len(cfg.metrics), cfg.cells, cfg.cells), dtype=np.float32
    )
    sources = [rasterio.open(LIDAR / f"ahn4_10m_{m}.tif") for m in cfg.metrics]
    try:
        for i, stem in enumerate(tqdm(stems, desc="crops")):
            bbox = json.loads((label_dir / f"{stem}.json").read_text())["bbox_world"]
            for c, src in enumerate(sources):
                out[i, c] = read_patch(src, bbox, cfg.cells)
    finally:
        for src in sources:
            src.close()

    out_path = Path(cfg.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, out)
    out_path.with_name(out_path.stem + "_stems.json").write_text(
        json.dumps({"stems": stems, "metrics": list(cfg.metrics), "cell_m": cfg.cell_m})
    )
    print(
        f"{out.shape} -> {out_path} "
        f"({out.nbytes / 1e6:.0f} MB, {float(np.isnan(out).mean()):.2%} nodata)"
    )


if __name__ == "__main__":
    root = DATA_ROOT / "pdok_dataset3_polylines"
    cfg = dict(
        label_dir=DATA_ROOT / "pdok_dataset3/labels",
        out_path=root / "lidar_patches.npy",
        # None builds every crop, about 26 min and 450 MB for 30,000 crops.
        # A small number is enough to check the dataset and the preview, which
        # is what this was first used for.
        limit=None,
        # Chosen by exps/probe_lidar_hedge_vs_tree.py, best single-metric
        # separation of hedge from tree row first. Height is kept because it is
        # the only absolute scale here, but it is the weakest of the six.
        metrics=[
            "band_ratio_1_normalized_height_2",  # 0.739
            "band_ratio_2_normalized_height_3",  # 0.701
            "band_ratio_3_normalized_height",  # 0.696
            "band_ratio_normalized_height_5",  # 0.685
            "coeff_var_normalized_height",  # 0.678
            "perc_95_normalized_height",  # 0.635
        ],
        cells=25,  # 250 m crop / 10 m cell
        cell_m=10.0,
    )
    main(OmegaConf.create(cfg))
