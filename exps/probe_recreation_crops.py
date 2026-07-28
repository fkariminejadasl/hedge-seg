"""
Find the crops that sit inside a campsite or holiday park.

Why this exists. Ranking the run 1 val crops by false-positive length put the
same kind of crop at the top every time: rows of identical chalets and static
caravans on small plots, with a clipped hedge around each plot. The model
answers those with dozens of short stubs.

They are not housing estates, which is what they look like at first glance.
Top10NL lists only 6 building footprints in a whole 250 m crop of one of them
and calls the ground grassland, because a chalet is not a registered building.
So a built-up or building-density filter misses them completely. The layer that
does find them is `functioneel_gebied_vlak`, field `typefunctioneelgebied`.

On the 13 crops that had been sorted by eye the separation was clean: 8 of 8
bad crops fall inside a recreation polygon, 0 of 5 good crops do.

The output is a stem list, not a change to the dataset. That is the point: it
lets `exps/probe_polyline_pr.py` report the score with those crops excluded
without retraining or reconverting anything.

Result (2026-07-28): 315 of the 3,098 val crops hit, 10.2%. Excluding them
moves 1_150.pt from F1 0.640 to 0.652 at 10 m. The crops really are harder
(0.542 against 0.652), but at 10% of crops they cannot move a per-image average
much, so filtering them out of the dataset is not worth doing. Run this first
next time before proposing a data change. The low recall on dense crops is a
real problem, but campsites are not most of it.

    /home/fatemeh/miniconda3/envs/hedge/bin/python exps/probe_recreation_crops.py
"""

import json
import sys
import time
import urllib.parse
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import geopandas as gpd
import numpy as np
from shapely.geometry import box, shape

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hedge_seg.paths import DATA_ROOT  # noqa: E402

API = "https://api.pdok.nl/brt/top10nl/ogc/v1/collections/functioneel_gebied_vlak/items"
RD_NEW = "http://www.opengis.net/def/crs/EPSG/0/28992"

# Values of typefunctioneelgebied that mean "plots divided by clipped hedges".
RECREATION_TYPES = {
    "camping, kampeerterrein",
    "vakantiepark",
    "bungalowpark",
    "caravanpark",
    "jachthaven",  # marinas have the same plot-and-hedge layout
}

TILE_M = 20_000  # one request per 20 km tile, 114 tiles for the 3,098 val crops
PAGE = 1000


def crop_boxes(polyline_dir, labels_dir):
    """(stems, Nx4 array of bbox_world) for the crops in a polyline directory."""
    stems = sorted(p.stem for p in Path(polyline_dir).glob("*.npz"))
    boxes = []
    for stem in stems:
        data = json.loads((Path(labels_dir) / f"{stem}.json").read_text())
        boxes.append(data["bbox_world"])
    return stems, np.asarray(boxes, dtype=np.float64)


def fetch_tile(bounds):
    """All features of the collection inside one bbox, following next links."""
    minx, miny, maxx, maxy = bounds
    params = {
        "bbox": f"{minx},{miny},{maxx},{maxy}",
        "bbox-crs": RD_NEW,
        "crs": RD_NEW,
        "limit": PAGE,
        "f": "json",
    }
    url = f"{API}?{urllib.parse.urlencode(params)}"
    features = []
    for _ in range(50):  # page guard
        for attempt in range(3):
            try:
                with urllib.request.urlopen(url, timeout=60) as resp:
                    payload = json.load(resp)
                break
            except Exception:
                if attempt == 2:
                    return features
                time.sleep(2 * (attempt + 1))
        features.extend(payload.get("features", []))
        nxt = [ln for ln in payload.get("links", []) if ln.get("rel") == "next"]
        if not nxt or len(payload.get("features", [])) < PAGE:
            break
        url = nxt[0]["href"]
    return features


def tiles_for(boxes, tile_m=TILE_M):
    keys = set()
    for minx, miny, maxx, maxy in boxes:
        for x, y in ((minx, miny), (maxx, maxy)):
            keys.add((int(x // tile_m), int(y // tile_m)))
    return [
        (i * tile_m, j * tile_m, (i + 1) * tile_m, (j + 1) * tile_m) for i, j in keys
    ]


def main(polyline_dir, labels_dir, out_path):
    stems, boxes = crop_boxes(polyline_dir, labels_dir)
    tiles = tiles_for(boxes)
    print(f"{len(stems)} crops, {len(tiles)} tiles of {TILE_M / 1000:.0f} km")

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(fetch_tile, tiles))
    features = [f for chunk in results for f in chunk]

    types = Counter(f["properties"].get("typefunctioneelgebied") for f in features)
    print(f"{len(features)} functioneel_gebied_vlak polygons fetched")
    for name, count in types.most_common(15):
        mark = "*" if name in RECREATION_TYPES else " "
        print(f"  {mark} {name}: {count}")

    keep = [
        f
        for f in features
        if f["properties"].get("typefunctioneelgebied") in RECREATION_TYPES
    ]
    if not keep:
        raise SystemExit("no recreation polygons found, nothing to write")

    areas = gpd.GeoDataFrame(
        {"type": [f["properties"]["typefunctioneelgebied"] for f in keep]},
        geometry=[shape(f["geometry"]) for f in keep],
        crs="EPSG:28992",
    ).drop_duplicates(subset="geometry")

    crops = gpd.GeoDataFrame(
        {"stem": stems},
        geometry=[box(*b) for b in boxes],
        crs="EPSG:28992",
    )
    hit = gpd.sjoin(crops, areas, how="inner", predicate="intersects")
    hit_stems = sorted(set(hit["stem"]))

    Path(out_path).write_text("\n".join(hit_stems) + "\n")
    print(
        f"\n{len(areas)} recreation polygons, "
        f"{len(hit_stems)} of {len(stems)} crops hit "
        f"({len(hit_stems) / len(stems):.1%})"
    )
    print(f"written to {out_path}")


if __name__ == "__main__":
    root = DATA_ROOT / "pdok_dataset3_polylines"
    main(
        polyline_dir=root / "polylines/val_cluster",
        labels_dir=DATA_ROOT / "pdok_dataset3/labels",
        out_path=root / "recreation_crops_val_cluster.txt",
    )
