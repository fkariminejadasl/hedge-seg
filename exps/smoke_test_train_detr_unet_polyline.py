"""
Smoke test for scripts/train_detr_unet_polyline.py.

Runs a few real training epochs on tiny train/val subsets with eval_every=2, so
the run must survive several eval->train transitions. This is the transition
that hung with Python 3.14's forkserver default (see _mp_context in the training
script); the test fails (times out) if that regresses. It also exercises the
whole pipeline end to end: backbone checkpoint load, image dataset, augmentation,
forward/backward/optimizer, matcher, eval, and checkpoint saving.

Needs a free GPU (it will OOM if a real training run is using it), the backbone
checkpoint, and pdok_dataset3 + its converted polylines. Run it after changing
the training script, the dataset, or the env:

    PYTHONPATH=. python exps/smoke_test_train_detr_unet_polyline.py
"""

import sys
import time
from pathlib import Path

from omegaconf import OmegaConf

# scripts/, not this file's own directory: the test moved to exps/ but the
# training script it imports stayed in scripts/.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import train_detr_unet_polyline as t  # noqa: E402

from hedge_seg.paths import CLUSTER_EXP_ROOT, DATA_ROOT  # noqa: E402

# Tiny and fast. eval_every=2 with n_epochs=6 gives three eval->train
# transitions, the case that hung before.
N_EPOCHS = 6
EVAL_EVERY = 2
N_TRAIN = 64
N_VAL = 64
BATCH_SIZE = 16
NUM_WORKERS = 4
TIMEOUT_S = 600  # a healthy run finishes in well under a minute on any GPU


def main():
    scratch = Path("/tmp") / "smoke_detr_unet_polyline"
    cfg = dict(
        mode="train",
        exp="smoke",
        save_path=scratch,
        image_dir=DATA_ROOT / "pdok_dataset3/images",
        train_polyline_dir=DATA_ROOT / "pdok_dataset3_polylines/polylines/train",
        val_polyline_dir=DATA_ROOT / "pdok_dataset3_polylines/polylines/val",
        pad_to=1024,
        augment=True,
        # Lidar off: the patches only exist for part of the dataset and no
        # model branch reads them yet. exps/probe_lidar_crop_alignment.py is
        # what exercises the lidar path.
        lidar_path=None,
        lidar_stride=16,
        backbone_ckpt=CLUSTER_EXP_ROOT / "semseg_unet/4/best_4.pt",
        feature_stage="up3",
        freeze_backbone=True,
        num_points=20,
        num_polylines=60,
        num_classes=1,
        d_model=256,
        nhead=8,
        num_encoder_layers=1,
        num_decoder_layers=4,
        dim_feedforward=1024,
        dropout=0.01,
        aux_loss=True,
        query_embed_mode="detr",
        with_refine=True,
        # Mirrors the committed training config: softmax head, the exp 1 and 2
        # setting, which exp 3 failed to beat. To exercise the focal head
        # instead set cls_loss="focal" with class_cost=2.0 and loss_ce=2.0;
        # both paths are worth smoke testing after touching the class head.
        cls_loss="ce",
        eos_coef=0.05,
        focal_alpha=0.25,
        focal_gamma=2.0,
        class_cost=1.0,
        loss_ce=1.0,
        poly_cost=5.0,
        bbox_cost=0.0,
        loss_poly=5.0,
        loss_bbox=0.0,
        loss_bbox_giou=0.0,
        loss_smooth=0.0,
        loss_card=0.0,
        loss_len=0.0,
        loss_dir=0.0,
        aux_weight=0.5,
        n_epochs=N_EPOCHS,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        max_lr=1e-4,
        weight_decay=1e-2,
        disable_tqdm=True,
        save_every=100,
        eval_every=EVAL_EVERY,
        seed=42,
        n_train_subset=N_TRAIN,
        n_val_subset=N_VAL,
        resume_ckpt=None,
        preview_out_dir=scratch / "preview",
        preview_n=2,
        infer_ckpt=None,
        infer_polyline_dir=DATA_ROOT / "pdok_dataset3_polylines/polylines/val",
        infer_out_dir=scratch / "inference",
        infer_score_thresh=0.5,
        infer_topk=60,
    )

    t0 = time.time()
    t.main(OmegaConf.create(cfg))
    dt = time.time() - t0
    assert dt < TIMEOUT_S, f"run took {dt:.0f}s (> {TIMEOUT_S}s): likely a hang"
    print(
        f"SMOKE OK: {N_EPOCHS} epochs, {N_EPOCHS // EVAL_EVERY} eval->train "
        f"transitions in {dt:.0f}s, no hang"
    )


if __name__ == "__main__":
    main()
