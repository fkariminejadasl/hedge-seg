"""
Are there failed downloads among the stored crops?

The cluster copy of pdok_dataset3 pos_001893 is one solid blue colour: the WMS
request returned an empty map, and nothing raised. It sat in the cluster train
split, so every cluster run trained on it, 1 of 26,902 crops. The laptop copy
is a real photo, and the laptop kept it when its data became the cluster
versions on 2026-11-09.

A real 1000 x 1000 crop is a PNG of about 1.4 MB and the blue one is 5.8 KB,
so this first lists every image under max_kb, then prints pixel statistics of
the copies named in cfg["compare"].

Result (2026-11-09, on the laptop, before snellius_backup was deleted):

    pdok_dataset3/images: 30000 images, 0 under 100 KB
    pdok_dataset2/images: 5000 images, 0 under 100 KB
    pos_001893 here: 1000x1000, 1534.1 KB, 20403 colours,
      mean RGB (77, 98, 79), std within a channel 19.4
    pos_001893 cluster copy: 1000x1000, 5.8 KB, 1 colours,
      mean RGB (0, 0, 254), std within a channel 0.0

So pos_001893 on the cluster is the only failed download, and none of the
5,000 pdok_dataset2 images now on the laptop is one.

On the cluster, DATA_ROOT/pdok_dataset3/images/pos_001893.png is still the
blue copy, so running this there reproduces the second line after the laptop
backup is gone.

    PYTHONPATH=. /home/fatemeh/miniconda3/envs/hedge/bin/python exps/probe_failed_downloads.py
"""

import os
from pathlib import Path

import numpy as np
from PIL import Image

from hedge_seg.paths import DATA_ROOT


def describe(path):
    im = np.asarray(Image.open(path).convert("RGB"))
    colours = len(np.unique(im.reshape(-1, 3), axis=0))
    px = im.reshape(-1, 3).astype(np.float64)
    mean = ", ".join(f"{v:.0f}" for v in px.mean(0))
    return (
        f"{im.shape[1]}x{im.shape[0]}, {os.path.getsize(path) / 1e3:.1f} KB, "
        f"{colours} colours, mean RGB ({mean}), std within a channel {px.std(0).mean():.1f}"
    )


def main(cfg):
    for image_dir in cfg["image_dirs"]:
        sizes = {e.name: e.stat().st_size for e in os.scandir(image_dir)}
        small = sorted(n for n, s in sizes.items() if s < cfg["max_kb"] * 1e3)
        print(
            f"{image_dir}: {len(sizes)} images, {len(small)} under {cfg['max_kb']} KB"
        )
        for name in small:
            print(f"  {name}: {describe(Path(image_dir) / name)}")
    for label, path in cfg["compare"]:
        if Path(path).exists():
            print(f"{label}: {describe(path)}")
        else:
            print(f"{label}: not on this machine ({path})")


if __name__ == "__main__":
    cfg = dict(
        image_dirs=[
            DATA_ROOT / "pdok_dataset3/images",
            DATA_ROOT / "pdok_dataset2/images",
        ],
        max_kb=100,
        compare=[
            ("pos_001893 here", DATA_ROOT / "pdok_dataset3/images/pos_001893.png"),
            (
                "pos_001893 cluster copy",
                "/home/fatemeh/Downloads/hedge/snellius_backup/data/"
                "pdok_dataset3_cluster_only/images/pos_001893.png",
            ),
        ],
    )
    main(cfg)
