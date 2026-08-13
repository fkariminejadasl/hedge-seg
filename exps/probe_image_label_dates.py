"""
How far apart are the aerial photos and the labels?

Two questions, both settled here rather than by reading layer names.

1. Which year are the crops? `pdok_dataset3` was built with the PDOK layer
   `Actueel_ortho25`, and "actueel" means whatever the newest flight is at the
   time of download, so the name does not say. This fetches one crop from
   `Actueel_ortho25` and from each yearly layer and compares the pixels.

2. Which year are the labels? Every Top10NL feature carries `bronactual`, the
   currency of the photo it was last drawn from. This reports the spread.

Result (2026-08-11, crop pos_000000, all 62,415 hedge features):

    mean absolute pixel difference from Actueel_ortho25
      2025_ortho25    0.00     <- identical, so the crops are 2025 imagery
      2022_ortho25   29.51
      2016_ortho25   26.15

    bronactual        2014-2015  75.3%   2016-2022  24.3%   2004-2013  0.4%

So the crops are 2025 and Top10NL2023 was revised on 2022 photos
(`BRT_Actualiteitskaart_april_2023.pdf`): a gap of about three years.

`bronactual` sitting in 2014-2015 for three quarters of features is NOT a ten
year gap. It records the source of each feature's last edit, and Top10NL
revises the whole country every year with trigger-based edits, so an unchanged
hedge keeps its old date. What survives is positional: those geometries have
not been redrawn in a decade.

Needs the network for question 1. Set cfg["layers"] to [] to skip it.

    /home/fatemeh/miniconda3/envs/hedge/bin/python exps/probe_image_label_dates.py
"""

import json
import sys
from io import BytesIO
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import requests
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hedge_seg.paths import DATA_ROOT  # noqa: E402

PDOK_WMS = "https://service.pdok.nl/hwh/luchtfotorgb/wms/v1_0"


def fetch(layer, bbox, size, timeout):
    params = {
        "service": "WMS",
        "request": "GetMap",
        "version": "1.3.0",
        "layers": layer,
        "styles": "",
        "crs": "EPSG:28992",
        "bbox": ",".join(str(v) for v in bbox),
        "width": size,
        "height": size,
        "format": "image/jpeg",
    }
    r = requests.get(PDOK_WMS, params=params, timeout=timeout)
    r.raise_for_status()
    return np.asarray(Image.open(BytesIO(r.content)).convert("RGB"), dtype=float)


def main(cfg):
    if cfg["layers"]:
        bbox = json.loads((Path(cfg["label_dir"]) / f"{cfg['stem']}.json").read_text())[
            "bbox_world"
        ]
        reference = fetch(cfg["reference_layer"], bbox, cfg["size"], cfg["timeout"])
        print(
            f"crop {cfg['stem']}, mean absolute pixel difference from "
            f"{cfg['reference_layer']}"
        )
        for layer in cfg["layers"]:
            diff = float(
                np.abs(
                    fetch(layer, bbox, cfg["size"], cfg["timeout"]) - reference
                ).mean()
            )
            note = "  <- identical" if diff == 0 else ""
            print(f"  {layer:18s} {diff:7.2f}{note}")

    years = pd.to_datetime(
        gpd.read_file(cfg["hedge_shp"], columns=["bronactual"], ignore_geometry=True)[
            "bronactual"
        ],
        errors="coerce",
    ).dt.year
    print(
        f"\nbronactual over {len(years)} hedge features, the photo each was "
        "last drawn from"
    )
    for lo, hi in cfg["year_bands"]:
        share = float(((years >= lo) & (years <= hi)).mean())
        print(
            f"  {lo}-{hi}  {int(((years >= lo) & (years <= hi)).sum()):6d}  {share:.1%}"
        )


if __name__ == "__main__":
    cfg = dict(
        label_dir=DATA_ROOT / "pdok_dataset3/labels",
        stem="pos_000000",
        hedge_shp="/home/fatemeh/Downloads/hedge/Topo10NL2023/Hedges_polylines/"
        "Top10NL2023_inrichtingselementen_lijn_heg.shp",
        reference_layer="Actueel_ortho25",  # what pdok_dataset3 was built with
        layers=["2025_ortho25", "2022_ortho25", "2016_ortho25"],
        size=500,
        timeout=120,
        year_bands=[(2004, 2013), (2014, 2015), (2016, 2022)],
    )
    main(cfg)
