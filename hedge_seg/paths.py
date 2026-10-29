"""
Machine-dependent path roots, so scripts do not need editing when moving
between the local machine and the Snellius cluster.

Only the roots live here. Dataset names, subfolders and hyperparameters stay
in each script's cfg, so the script content copied into the slurm .out log
still documents the full experiment.

Usage:

    from hedge_seg.paths import CLUSTER_EXP_ROOT, DATA_ROOT, EXP_ROOT

    cfg = dict(
        image_dir=DATA_ROOT / "pdok_dataset3/images",
        save_path=EXP_ROOT / "detr_unet_polyline",
        backbone_ckpt=CLUSTER_EXP_ROOT / "semseg_unet/4/best_4.pt",
    )
"""

from pathlib import Path

# Exists only on the cluster, so no environment variable has to be set.
CLUSTER_MARKER = Path("/projects/prjs1025")


def on_cluster() -> bool:
    return CLUSTER_MARKER.exists()


def get_roots() -> dict:
    """
    data_root: generated datasets (images, labels, converted polylines)
    exp_root: training outputs of the current run (checkpoints, tensorboard)
    cluster_exp_root: outputs of cluster runs, used to read pretrained
        checkpoints. On the cluster this is exp_root itself; locally it is the
        directory the cluster results are copied into, which mirrors the same
        layout (e.g. semseg_unet/4/best_4.pt on both).
    """
    if on_cluster():
        exp_root = Path.home() / "exps/hedge"
        return {
            "data_root": CLUSTER_MARKER / "data/hedge",
            "exp_root": exp_root,
            "cluster_exp_root": exp_root,
        }
    return {
        "data_root": Path("/home/fatemeh/Downloads/hedge/results"),
        "exp_root": Path("/home/fatemeh/Downloads/hedge/results/training"),
        "cluster_exp_root": Path("/home/fatemeh/Downloads/hedge/snellius"),
    }


_ROOTS = get_roots()
DATA_ROOT = _ROOTS["data_root"]
EXP_ROOT = _ROOTS["exp_root"]
CLUSTER_EXP_ROOT = _ROOTS["cluster_exp_root"]


def print_roots() -> None:
    """Call at the start of main() so the resolved roots land in the log."""
    print(
        f"Roots: cluster={on_cluster()}, data_root={DATA_ROOT}, "
        f"exp_root={EXP_ROOT}, cluster_exp_root={CLUSTER_EXP_ROOT}"
    )
