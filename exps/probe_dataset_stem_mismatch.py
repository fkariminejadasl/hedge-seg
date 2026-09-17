"""
Does this polyline dataset name crops that exist on this machine?

pdok_dataset3 was downloaded twice, once on the laptop and once on the cluster.
Both hold exactly 30,000 crops, so every count matches and nothing looks wrong.
They are not the same 30,000: crops are sampled one per polyline, and a failed
PDOK WMS request shifts the numbering onward, so each download skipped a
different set of indices.

A polyline dataset built on one machine therefore names crops the other does not
have, and the same is true of lidar_patches.npy, which is keyed by stem. This
walks the three stem sets and reports every mismatch, which is the check that
has to pass before a locally built dataset is used on the cluster.

Result (2026-09-17, pdok_dataset3):

    laptop   pos_000000 .. pos_029999, 30,000 crops, no gaps
    cluster  pos_000000 .. pos_030003, 30,000 crops, 4 gaps:
             pos_029989, pos_029998, pos_030000, pos_030001

So the laptop has pos_029989 and pos_029998 which the cluster lacks, and the
cluster has pos_030002 and pos_030003 which the laptop lacks.

Consequences measured the same day:

    pdok_dataset3_tree_polylines  built on the laptop, so on the cluster two of
                                  its train crops had no image (pos_029989,
                                  pos_029998). They were deleted there, leaving
                                  26,900 train against 26,902 on the laptop.
                                  Val is 3,098 on both and is exp 2's val set.
    lidar_patches.npy             built on the laptop, so it has no patch for
                                  pos_030002 or pos_030003. The tree dataset
                                  does not contain those stems, but
                                  pdok_dataset3_polylines does, so the lidar
                                  only ablation has to drop them first.

This is what killed exp 4 job 26798493 three minutes into epoch 1 with "No image
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
