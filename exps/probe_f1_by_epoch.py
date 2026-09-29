"""
Is a run still improving when it stops? Hedge F1 of a run's snapshots, epoch by
epoch.

Each snapshot has to be inferred first (scripts/train_detr_unet_polyline.py,
mode="infer", infer_ckpt=<n>/<n>_<epoch>.pt, at infer_score_thresh=0.05).

Result (2026-09-29, exp 6 on the 3,098 val crops, t=0.90):

    epoch   F1 5 m   F1 10 m   F1 15 m   recall at t=0.05   lr / max_lr
    20      0.492    0.667     0.752     0.813              0.59
    30      0.516    0.687     0.764     0.831              0.26
    40      0.546    0.703     0.775     0.845              0.04
    45      0.545    0.703     0.774     0.847              0.01

Every snapshot scores best at t=0.90. F1 at 10 m rose about 0.02 per 10
epochs and stopped at epoch 40, when the cosine schedule had brought the
learning rate down to 4% of its start. So the stop may be the schedule rather
than the model, which is why exp 8 trains exp 2 for 90 epochs.

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
    print(
        f"{'epoch':>5} {'F1 5 m':>7} {'10 m':>6} {'15 m':>6} "
        f"{'best t':>6} {'R@0.05':>7} {'lr':>5}"
    )
    for epoch, run_dir in cfg["snapshots"].items():
        items = load_run(run_dir)
        rows = score(items, score_thresh=cfg["score_thresh"], class_id=0)
        best_t = max(
            SWEEP, key=lambda t: f1_at(score(items, score_thresh=t, class_id=0), 10)
        )
        recall = macro_average(score(items, score_thresh=0.05, class_id=0), "r10")
        print(
            f"{epoch:>5} {f1_at(rows, 5):>7.3f} {f1_at(rows, 10):>6.3f} "
            f"{f1_at(rows, 15):>6.3f} {best_t:>6} {recall:>7.3f} "
            f"{lr_fraction(epoch, cfg['n_epochs']):>5.2f}"
        )


if __name__ == "__main__":
    inference = DATA_ROOT / "pdok_dataset3_polylines/inference"
    cfg = dict(
        snapshots={
            20: inference / "6_20_val_cluster_t0.05",
            30: inference / "6_30_val_cluster_t0.05",
            40: inference / "6_40_val_cluster_t0.05",
            45: inference / "best_6_val_cluster_t0.05",  # best_6.pt is epoch 45
        },
        n_epochs=45,  # of the run, for the learning rate column
        score_thresh=0.90,
    )
    main(cfg)
