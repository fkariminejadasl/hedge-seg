"""
Does this polyline dataset name crops that exist on this machine?

Crops are numbered by their place in the polyline sampling. A failed PDOK WMS
request leaves a gap, and the download runs on past the end to reach its
count. So two downloads of the same dataset can both hold 30,000 crops and
still not hold the same 30,000. A polyline dataset or a stem-keyed
lidar_patches.npy built against one copy then names crops the other lacks.
This walks the stem sets and reports every mismatch.

Result (2026-11-10, laptop, after its data became the cluster versions):

    images  pos_000000 .. pos_030003, 30,000 crops, 4 gaps:
            pos_029989, pos_029998, pos_030000, pos_030001
    lidar   30,000 stems
    both polyline datasets, train and val: no crop without an image or a
    lidar patch. PASS

The cluster images have the same numbering. Its lidar_patches.npy is the
older one, with no patch for pos_030002 or pos_030003, so a lidar run there
needs the laptop file copied over first.

Why it exists: on 2026-09-17 the laptop copy of pdok_dataset3 ran
pos_000000 .. pos_029999 with no gaps, a different 30,000. The tree dataset
built there killed exp 4 job 26798493 three minutes into epoch 1 with "No image
for pos_029989". DetrPolylineImageDataset now runs the same check at
construction, so the failure is immediate and names every crop at once.

Run it on both machines and compare, since each only sees its own copy:

    /home/fatemeh/miniconda3/envs/hedge/bin/python exps/probe_dataset_stem_mismatch.py
    ssh me "cd dev/hedge-seg && PYTHONPATH=. \
        ~/.conda/envs/hedge/bin/python exps/probe_dataset_stem_mismatch.py"
"""

import json
import re
from pathlib import Path

from hedge_seg.paths import DATA_ROOT

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
STEM_RE = re.compile(r"^(?P<prefix>[a-z]+)_(?P<index>\d+)$")


def image_stems(image_dir: Path) -> set:
    """Stems of every readable image, the same rule the training dataset uses."""
    return {
        p.stem for p in Path(image_dir).iterdir() if p.suffix.lower() in IMAGE_SUFFIXES
    }


def describe_numbering(stems: set) -> str:
    """First, last and the indices missing in between, for pos_<n> style stems."""
    parsed = sorted(
        (int(m.group("index")), m.group("prefix"))
        for m in (STEM_RE.match(s) for s in stems)
        if m is not None
    )
    if not parsed:
        return "no pos_<n> style stems"
    lo, hi = parsed[0][0], parsed[-1][0]
    prefix = parsed[0][1]
    present = {i for i, _ in parsed}
    gaps = [i for i in range(lo, hi + 1) if i not in present]
    width = len(str(hi).zfill(6))
    span = f"{prefix}_{lo:0{width}d} .. {prefix}_{hi:0{width}d}, {len(stems):,} crops"
    if not gaps:
        return f"{span}, no gaps"
    shown = ", ".join(f"{prefix}_{i:0{width}d}" for i in gaps[:8])
    more = "" if len(gaps) <= 8 else f", and {len(gaps) - 8} more"
    return f"{span}, {len(gaps)} gaps: {shown}{more}"


def report(name: str, missing: list) -> bool:
    """Print one check. Returns True when it passed."""
    if not missing:
        print(f"  OK    {name}: none")
        return True
    shown = ", ".join(sorted(missing)[:5])
    more = "" if len(missing) <= 5 else f", and {len(missing) - 5} more"
    print(f"  FAIL  {name}: {len(missing)} ({shown}{more})")
    return False


def main(cfg):
    images = image_stems(cfg["image_dir"])
    print(f"images  {cfg['image_dir']}")
    print(f"        {describe_numbering(images)}")

    lidar = None
    if cfg["lidar_stems_json"] is not None:
        lidar = set(json.loads(Path(cfg["lidar_stems_json"]).read_text())["stems"])
        print(f"lidar   {cfg['lidar_stems_json']}")
        print(f"        {len(lidar):,} stems")

    ok = True
    for polyline_dir in cfg["polyline_dirs"]:
        polyline_dir = Path(polyline_dir)
        stems = {p.stem for p in polyline_dir.glob("*.npz")}
        print(f"\n{polyline_dir}  ({len(stems):,} crops)")
        ok &= report("no image", [s for s in stems if s not in images])
        if lidar is not None:
            ok &= report("no lidar patch", [s for s in stems if s not in lidar])

    if lidar is not None:
        print(f"\n{cfg['image_dir']} against the lidar index")
        # Not fatal on its own: a crop with no patch only matters once a
        # polyline dataset names it.
        report("image with no lidar patch", [s for s in images if s not in lidar])

    print("\nPASS" if ok else "\nFAIL: a dataset names crops this machine lacks")


if __name__ == "__main__":
    cfg = dict(
        image_dir=DATA_ROOT / "pdok_dataset3/images",
        polyline_dirs=[
            DATA_ROOT / "pdok_dataset3_tree_polylines/polylines/train",
            DATA_ROOT / "pdok_dataset3_tree_polylines/polylines/val",
            DATA_ROOT / "pdok_dataset3_polylines/polylines/train",
            DATA_ROOT / "pdok_dataset3_polylines/polylines/val",
        ],
        lidar_stems_json=DATA_ROOT / "pdok_dataset3_polylines/lidar_patches_stems.json",
    )
    main(cfg)
