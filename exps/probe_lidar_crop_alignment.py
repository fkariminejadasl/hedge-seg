"""
Does the lidar patch land on the same ground as the crop?

Before lidar can be an input, the world-to-pixel maths has to be right, and it
is easy to get wrong: the crop bbox is in RD New metres, the raster is 10 m in
the same CRS but its rows run north to south, and the polylines are in crop
pixels. A one-cell error is 10 m and would quietly poison training.

This writes one PNG per crop per metric: the aerial image, the lidar patch
stretched over it, and the ground-truth hedges on top. A 250 m crop is exactly
25x25 lidar cells, drawn without smoothing so the cell edges show.

What to look for, with `height_p95` or `BR_above_3`:

- the bright ridge should sit **under** the red hedge lines
- mirrored top to bottom means the y flip is wrong
- transposed means row and column are swapped
- shifted by one cell means an off-by-one in the window

It also writes `crop_footprints.geojson` so the same crops can be found in
QGIS (/home/fatemeh/Downloads/hedge/hedge.qgz), where the rasters and Top10NL
layers are already loaded and nothing is cropped.

This deliberately does not touch the training script. It checks the maths only.
The second half of the check is augmentation: the dataset flips and rotates the
image and polylines together before padding, so a lidar array added there must
be flipped and rotated too. That only shows up in `mode="preview"` with
`augment=True`.

    /home/fatemeh/miniconda3/envs/hedge/bin/python exps/probe_lidar_crop_alignment.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import from_bounds

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hedge_seg.paths import DATA_ROOT  # noqa: E402
from hedge_seg.visualization import show_polyline_single  # noqa: E402

LIDAR = Path("/home/fatemeh/Downloads/hedge/LiDAR_metrics_AHN4")


def lidar_patch(tif_path, bbox_world):
    """The raster over exactly this crop, nodata as NaN. A 250 m crop is 25x25."""
    with rasterio.open(tif_path) as src:
        window = from_bounds(*bbox_world, transform=src.transform)
        patch = src.read(1, window=window, boundless=True, fill_value=np.nan).astype(
            float
        )
        nodata = src.nodata
    # Each raster has its own nodata and the band ratios use +3.4e38, so a
    # single "less than -100" filter would keep the positive sentinels.
    if nodata is not None:
        patch[np.isclose(patch, nodata, rtol=1e-6)] = np.nan
    patch[np.abs(patch) > 1e30] = np.nan
    return patch


def shift_test(cfg, stems, metric):
    """
    Prove the alignment instead of eyeballing it.

    Sample the metric in the cells the hedges pass through and in the cells they
    do not, and take the difference. Then repeat with the patch shifted by whole
    cells. If the maths is right, no shift beats a shift of zero, and one cell
    is 10 m, so an off-by-one is unmissable.
    """
    shifts = range(-2, 3)
    scores = {}
    for dr in shifts:
        for dc in shifts:
            on, off = [], []
            for stem in stems:
                meta = json.loads((Path(cfg["label_dir"]) / f"{stem}.json").read_text())
                patch = lidar_patch(
                    LIDAR / f"ahn4_10m_{metric}.tif", meta["bbox_world"]
                )
                polylines = np.load(Path(cfg["polyline_dir"]) / f"{stem}.npz")[
                    "polylines"
                ].astype(float)
                if len(polylines) == 0:
                    continue
                rows, cols = patch.shape
                scale = cfg["crop_px"] / rows  # crop pixels per lidar cell
                mask = np.zeros(patch.shape, dtype=bool)
                for polyline in polylines:
                    # densify so long segments do not skip cells
                    p = np.asarray(polyline, dtype=float)
                    steps = np.linspace(0, 1, 200)
                    for a, b in zip(p[:-1], p[1:]):
                        pts = a + steps[:, None] * (b - a)
                        c = np.clip((pts[:, 0] / scale).astype(int) + dc, 0, cols - 1)
                        r = np.clip((pts[:, 1] / scale).astype(int) + dr, 0, rows - 1)
                        mask[r, c] = True
                valid = np.isfinite(patch)
                on.extend(patch[mask & valid].ravel())
                off.extend(patch[(~mask) & valid].ravel())
            scores[(dr, dc)] = float(np.mean(on) - np.mean(off))

    best = max(scores, key=scores.get)
    print(f"\nshift test on {metric}, {len(stems)} crops")
    print("  mean value under the hedges minus mean elsewhere, by cell shift")
    print("        " + "".join(f"{dc:>8d}" for dc in shifts) + "   (column shift)")
    for dr in shifts:
        row = "".join(f"{scores[(dr, dc)]:8.2f}" for dc in shifts)
        print(f"  {dr:>3d} |{row}")
    print(f"  (row shift)\n  best shift {best}, want (0, 0)")
    return best


def pick_ids(polyline_dir, n, max_lines=None):
    """
    Crops with the most labelled hedge, longest first.

    `max_lines` keeps only crops with at most that many hedges. For looking at a
    picture that is the setting that matters: one long hedge across an empty
    field shows a shift immediately, while a crop packed with hedges has
    something bright almost everywhere and reads as noise either way. For the
    shift test leave it off, since there more hedge is simply more signal.
    """
    lengths = []
    for path in sorted(Path(polyline_dir).glob("*.npz")):
        polylines = np.load(path)["polylines"].astype(float)
        if len(polylines) == 0:
            continue
        if max_lines is not None and len(polylines) > max_lines:
            continue
        total = sum(
            float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum()) for p in polylines
        )
        lengths.append((total, path.stem))
    lengths.sort(reverse=True)
    return [stem for _, stem in lengths[:n]]


def main(cfg):
    out_dir = Path(cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    stems = cfg["stems"] or pick_ids(
        cfg["polyline_dir"], cfg["n_crops"], max_lines=cfg["max_lines"]
    )
    print(f"{len(stems)} crops: {', '.join(stems)}")

    features = []
    for stem in stems:
        meta = json.loads((Path(cfg["label_dir"]) / f"{stem}.json").read_text())
        bbox = meta["bbox_world"]
        image_path = Path(cfg["image_dir"]) / f"{stem}.png"
        polylines = np.load(Path(cfg["polyline_dir"]) / f"{stem}.npz")["polylines"]

        show_polyline_single(
            image_path,
            polylines=polylines,
            title=f"{stem}  image only",
            save_path=out_dir / f"{stem}_image.png",
        )
        for metric in cfg["metrics"]:
            patch = lidar_patch(LIDAR / f"ahn4_10m_{metric}.tif", bbox)
            show_polyline_single(
                image_path,
                polylines=polylines,
                overlay=patch,
                alpha=cfg["alpha"],
                title=f"{stem}  {metric}  ({patch.shape[0]}x{patch.shape[1]} cells)",
                save_path=out_dir / f"{stem}_{metric}.png",
            )

        minx, miny, maxx, maxy = bbox
        features.append(
            {
                "type": "Feature",
                "properties": {"stem": stem},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        [
                            [minx, miny],
                            [maxx, miny],
                            [maxx, maxy],
                            [minx, maxy],
                            [minx, miny],
                        ]
                    ],
                },
            }
        )

    geojson = out_dir / "crop_footprints.geojson"
    geojson.write_text(
        json.dumps(
            {
                "type": "FeatureCollection",
                "crs": {
                    "type": "name",
                    "properties": {"name": "urn:ogc:def:crs:EPSG::28992"},
                },
                "features": features,
            }
        )
    )
    print(f"\nImages: {out_dir}")
    print(f"QGIS footprints: {geojson}")

    if cfg["n_shift_crops"]:
        shift_test(
            cfg,
            pick_ids(cfg["polyline_dir"], cfg["n_shift_crops"]),
            cfg["metrics"][0],
        )


if __name__ == "__main__":
    cfg = dict(
        image_dir=DATA_ROOT / "pdok_dataset3/images",
        label_dir=DATA_ROOT / "pdok_dataset3/labels",
        polyline_dir=DATA_ROOT / "pdok_dataset3_polylines/polylines/val_cluster",
        out_dir=DATA_ROOT / "pdok_dataset3_polylines/lidar_alignment",
        stems=None,  # None picks the crops with the most labelled hedge
        n_crops=8,  # crops to draw
        max_lines=2,  # draw uncluttered crops; a lone hedge shows a shift best
        # perc_95 and band_ratio_3 show the woody line most clearly, so they are
        # the ones to check alignment on. band_ratio_1_..._2 is the hedge/tree
        # discriminator and is noisier, so check it second.
        metrics=[
            "perc_95_normalized_height",
            "band_ratio_3_normalized_height",
            "band_ratio_1_normalized_height_2",
        ],
        alpha=0.5,
        crop_px=1000,  # crop side in pixels, 250 m at 0.25 m
        n_shift_crops=40,  # crops for the shift test; 0 skips it
    )
    main(cfg)
