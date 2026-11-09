"""
Is a run still improving when it stops? F1 of a run's snapshots, epoch by
epoch. Each run sets class_id as in exps/probe_polyline_pr.py: 0 scores hedge
lines against hedge labels, None every line against every label (woody lines,
exp 10).

The snapshot inference runs below (6_20 to 6_40, 8_40, 8_60, 10_10 to 10_40)
were deleted on 2026-11-09 after scoring, with the snapshots, so the numbers
here are the record and the cfg shows what was read. The question is
answered: a run stops improving when the learning rate runs out, and training
longer adds little (exp 8). To score a new run, keep its snapshots and infer
them first (scripts/train_detr_unet_polyline.py,
mode="infer", infer_ckpt=<n>/<n>_<epoch>.pt, at infer_score_thresh=0.05).

Result (2026-09-29, exp 6 on the 3,098 val crops, t=0.90):

    epoch   F1 5 m   F1 10 m   F1 15 m   recall at t=0.05   lr / max_lr
    20      0.492    0.667     0.752     0.813              0.59
    30      0.516    0.687     0.764     0.831              0.26
    40      0.546    0.703     0.775     0.845              0.04
    45      0.545    0.703     0.774     0.847              0.01

Exp 8, the exp 2 recipe for 90 epochs, t=0.90:

    epoch   F1 5 m   F1 10 m   F1 15 m   recall at t=0.05   lr / max_lr
    40      0.534    0.692     0.763     0.834              0.59
    60      0.551    0.708     0.773     0.846              0.26
    70      0.556    0.705     0.770     0.851              0.13
    90      0.569    0.712     0.775     0.845              0.01

Exp 10, hedges and tree rows as one class, woody lines (every line against
both label layers), t=0.90:

    epoch   F1 5 m   F1 10 m   F1 15 m   recall at t=0.05   lr / max_lr
    10      0.459    0.659     0.751     0.838              0.88
    20      0.503    0.689     0.769     0.859              0.59
    30      0.529    0.707     0.778     0.867              0.26
    35      0.546    0.720     0.786     0.864              0.13
    40      0.548    0.720     0.785     0.862              0.04
    45      0.551    0.722     0.786     0.866              0.01

Every snapshot scores best at t=0.90. Exp 6 rose about 0.02 per 10 epochs at
10 m and stopped at epoch 40, when the cosine schedule had brought the
learning rate down to 4% of its start. So the stop may be the schedule rather
than the model, which is why exp 8 trains exp 2 for 90 epochs. Exp 10 has the
same shape: about 0.03 per 10 epochs, then flat from epoch 35, at 13% of the
learning rate.

    /home/fatemeh/miniconda3/envs/hedge/bin/python exps/probe_f1_by_epoch.py
"""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from probe_polyline_pr import SWEEP, load_run, score  # noqa: E402

from hedge_seg.metrics import f1, macro_average  # noqa: E402
from hedge_seg.paths import DATA_ROOT  # noqa: E402


def f1_at(rows, buf):
    return f1(macro_average(rows, f"p{buf}"), macro_average(rows, f"r{buf}"))


def lr_fraction(epoch, n_epochs, max_lr=1e-4, eta_min=1e-6):
    """The cosine schedule of the training script, as a share of max_lr."""
    lr = eta_min + (max_lr - eta_min) * (1 + math.cos(math.pi * epoch / n_epochs)) / 2
    return lr / max_lr


def main(cfg):
    for name, run in cfg["runs"].items():
        class_id = run.get("class_id", 0)
        print(f"{name} ({run['n_epochs']} epochs)")
        print(
            f"{'epoch':>5} {'F1 5 m':>7} {'10 m':>6} {'15 m':>6} "
            f"{'best t':>6} {'R@0.05':>7} {'lr':>5}"
        )
        for epoch, run_dir in run["snapshots"].items():
            items = load_run(run_dir)
            rows = score(items, score_thresh=cfg["score_thresh"], class_id=class_id)
            best_t = max(
                SWEEP,
                key=lambda t: f1_at(
                    score(items, score_thresh=t, class_id=class_id), 10
                ),
            )
            recall = macro_average(
                score(items, score_thresh=0.05, class_id=class_id), "r10"
            )
            print(
                f"{epoch:>5} {f1_at(rows, 5):>7.3f} {f1_at(rows, 10):>6.3f} "
                f"{f1_at(rows, 15):>6.3f} {best_t:>6} {recall:>7.3f} "
                f"{lr_fraction(epoch, run['n_epochs']):>5.2f}"
            )
        print()


if __name__ == "__main__":
    inference = DATA_ROOT / "pdok_dataset3_polylines/inference"
    tree = DATA_ROOT / "pdok_dataset3_tree_polylines/inference"
    cfg = dict(
        # Per run: its snapshots by epoch, and its length for the lr column.
        runs={
            "exp 6": dict(
                n_epochs=45,
                snapshots={
                    20: inference / "6_20_val_cluster_t0.05",
                    30: inference / "6_30_val_cluster_t0.05",
                    40: inference / "6_40_val_cluster_t0.05",
                    45: inference / "best_6_val_cluster_t0.05",  # epoch 45
                },
            ),
            "exp 8": dict(
                n_epochs=90,
                snapshots={
                    40: inference / "8_40_val_cluster_t0.05",
                    60: inference / "8_60_val_cluster_t0.05",
                    70: inference / "best_8_val_cluster_t0.05",  # epoch 70
                    90: inference / "8_val_cluster_t0.05",  # epoch 90
                },
            ),
            "exp 10, woody lines": dict(
                n_epochs=45,
                class_id=None,  # every line against both label layers
                snapshots={
                    10: tree / "10_10_val_t0.05",
                    20: tree / "10_20_val_t0.05",
                    30: tree / "10_30_val_t0.05",
                    35: tree / "best_10_val_t0.05",  # epoch 35
                    40: tree / "10_40_val_t0.05",
                    45: tree / "10_val_t0.05",  # epoch 45
                },
            ),
        },
        score_thresh=0.90,
    )
    main(cfg)
