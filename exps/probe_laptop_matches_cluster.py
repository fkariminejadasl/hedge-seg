"""
Does the laptop hold the same datasets and runs as the cluster?

Hashes every file with md5 on both machines and compares the two lists. The
cluster side runs over ssh and only reads. Datasets are compared by path. Runs
are compared by content, because the laptop keeps each run's slurm script and
log inside the run folder, and the cluster keeps them beside it.

Run it on the laptop after copying from the cluster, as in "Keeping the laptop
in step with the cluster" in CLAUDE.md. It takes about 5 minutes, most of it
the cluster reading 54 GB of images.

Result (2026-11-11):

    datasets: 130,000 files on the cluster, 130,000 here
      identical 129,997
      differ 3:
        pdok_dataset3/images/pos_001893.png
        pdok_dataset3_polylines/lidar_patches.npy
        pdok_dataset3_polylines/lidar_patches_stems.json
      on the cluster only 0
      here only 0
    runs: 549 files on the cluster, 547 here
      cluster files missing here 31, of which epoch snapshots 31
      missing here, not a snapshot 0
      content here only 9:
        detect_ultralytics/3/best.pt
        detect_ultralytics/3/last.pt
        detect_ultralytics/3/pos_000004.jpg
        detect_ultralytics/3/pos_029997.jpg
        detect_ultralytics/3/results.csv
        seg_ultralytics/1/best.pt
        seg_ultralytics/1/results.csv
        seg_ultralytics/test_m_1epoch.txt
        semseg_unet/5/5_23187937.out

So the copy is complete. The 3 dataset files differ on purpose: the cluster
pos_001893 is a blue failed download where the laptop has the real photo
(exps/probe_failed_downloads.py), and the laptop lidar file was rebuilt with
the two crops the cluster file lacks. The cluster keeps the epoch snapshots
<n>_<epoch>.pt. Of the 9 laptop files, 3 have a cluster twin of the same name
with other content, kept next to them as results_cluster.csv and
5_23187937_cluster.out. The other 6 are older files the cluster no longer has.

    PYTHONPATH=. /home/fatemeh/miniconda3/envs/hedge/bin/python exps/probe_laptop_matches_cluster.py
"""

import re
import shlex
import subprocess
from pathlib import Path

from hedge_seg.paths import CLUSTER_EXP_ROOT, CLUSTER_MARKER, DATA_ROOT, on_cluster

SNAPSHOT = re.compile(r"^\d+_\d+\.pt$")  # <n>_<epoch>.pt


def md5_command(root, paths):
    """Shell command that prints 'md5  path' for every file under paths."""
    names = " ".join(shlex.quote(str(p)) for p in paths)
    return (
        f"cd {shlex.quote(str(root))} && find {names} \\( -type f -o -type l \\) "
        "-print0 | xargs -0 -n 500 -P 4 nice -n 19 md5sum"
    )


def start(cmd, host=None):
    args = ["ssh", "-o", "BatchMode=yes", host, cmd] if host else ["bash", "-c", cmd]
    return subprocess.Popen(
        args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )


def collect(proc, label):
    out, err = proc.communicate()
    if err.strip():
        print(f"{label} stderr: {err.strip()}")
    if proc.returncode == 255 or not out:
        raise RuntimeError(f"{label}: no hashes (exit {proc.returncode})")
    hashes = {}
    for line in out.splitlines():
        md5, path = line.split("  ", 1)
        hashes[path.removeprefix("./")] = md5
    return hashes


def show(label, paths, limit=20):
    print(f"{label} {len(paths):,}" + (":" if paths else ""))
    for p in paths[:limit]:
        print(f"    {p}")
    if len(paths) > limit:
        print(f"    ... and {len(paths) - limit} more")


def compare_data(cluster, laptop):
    print(f"datasets: {len(cluster):,} files on the cluster, {len(laptop):,} here")
    shared = cluster.keys() & laptop.keys()
    differ = sorted(p for p in shared if cluster[p] != laptop[p])
    print(f"  identical {len(shared) - len(differ):,}")
    show("  differ", differ)
    show("  on the cluster only", sorted(cluster.keys() - laptop.keys()))
    show("  here only", sorted(laptop.keys() - cluster.keys()))


def compare_runs(cluster, laptop):
    print(f"runs: {len(cluster):,} files on the cluster, {len(laptop):,} here")
    here, there = set(laptop.values()), set(cluster.values())
    missing = sorted(p for p, h in cluster.items() if h not in here)
    snapshots = [p for p in missing if SNAPSHOT.match(Path(p).name)]
    print(
        f"  cluster files missing here {len(missing)}, "
        f"of which epoch snapshots {len(snapshots)}"
    )
    show("  missing here, not a snapshot", [p for p in missing if p not in snapshots])
    show("  content here only", sorted(p for p, h in laptop.items() if h not in there))


def main(cfg):
    assert not on_cluster(), "run this on the laptop; it reaches the cluster by ssh"
    host = cfg["host"]
    jobs = {
        "cluster data": start(md5_command(cfg["cluster_data"], cfg["datasets"]), host),
        "laptop data": start(md5_command(cfg["laptop_data"], cfg["datasets"])),
        "cluster runs": start(md5_command(cfg["cluster_runs"], ["."]), host),
        "laptop runs": start(md5_command(cfg["laptop_runs"], ["."])),
    }
    h = {label: collect(proc, label) for label, proc in jobs.items()}
    compare_data(h["cluster data"], h["laptop data"])
    compare_runs(h["cluster runs"], h["laptop runs"])


if __name__ == "__main__":
    cfg = dict(
        host="me",
        cluster_data=CLUSTER_MARKER / "data/hedge",
        cluster_runs="exps/hedge",  # relative to the home directory on the cluster
        laptop_data=DATA_ROOT,
        laptop_runs=CLUSTER_EXP_ROOT,
        # Every dataset the training script reads. Add a new one here once it
        # has been copied.
        datasets=[
            "pdok_dataset2/images",
            "pdok_dataset2/labels",
            "pdok_dataset3/images",
            "pdok_dataset3/labels",
            "pdok_dataset3_polylines/polylines/train",
            "pdok_dataset3_polylines/polylines/val",
            "pdok_dataset3_polylines/lidar_patches.npy",
            "pdok_dataset3_polylines/lidar_patches_stems.json",
            "pdok_dataset3_tree_polylines/polylines/train",
            "pdok_dataset3_tree_polylines/polylines/val",
        ],
    )
    main(cfg)
