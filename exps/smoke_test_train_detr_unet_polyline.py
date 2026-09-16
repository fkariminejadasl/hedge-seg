"""
Smoke test for scripts/train_detr_unet_polyline.py.

Runs a few real training epochs on tiny train/val subsets with eval_every=2, so
the run must survive several eval->train transitions. This is the transition
that hung with Python 3.14's forkserver default (see _mp_context in the training
script); the test fails (times out) if that regresses. It also exercises the
whole pipeline end to end: backbone checkpoint load, image dataset, augmentation,
forward/backward/optimizer, matcher, eval, and checkpoint saving.

It runs the two-class tree dataset with the lidar branch on, so it covers the
exp 4 configuration. Before the training run it checks the lidar branch itself
(`check_lidar_branch`), because two of its properties cannot be seen from a loss
curve: the branch is exactly inert at initialisation, and it does receive
gradients. Without the first, a lidar run would not be comparable with a
no-lidar one; without the second, the branch could sit there doing nothing for
eleven hours.

Needs a free GPU (it will OOM if a real training run is using it), the backbone
checkpoint, and pdok_dataset3 + its converted polylines. Run it after changing
the training script, the dataset, or the env:

    PYTHONPATH=. python exps/smoke_test_train_detr_unet_polyline.py
"""

import sys
import time
from pathlib import Path

import torch
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


def check_lidar_branch(n_lidar_channels=7, grid=8, num_polylines=4, num_points=20):
    """
    Three things about the lidar branch that a training log would not show.

    1. The patch the dataset produces has the shape of the token grid, so the
       two can be added at all.
    2. At initialisation the branch outputs exactly zero, so the model is the
       no-lidar model and a lidar run starts from the same place as one without.
    3. A few steps later it is no longer zero, so it is able to learn.

    A small grid is used here because none of this depends on the size.

    The gradient check has to go through the class head. At initialisation every
    reg_branch's last layer is zero, so the geometry path passes no gradient
    back into the tokens at all and a loss made only of pred_polylines leaves
    the lidar branch with a zero gradient. That is true of the image path too
    and it clears up after the first steps, but a test written the obvious way
    fails for a reason that has nothing to do with lidar.
    """
    ds = t.DetrPolylineImageDataset(
        image_dir=DATA_ROOT / "pdok_dataset3/images",
        polyline_dir=DATA_ROOT / "pdok_dataset3_tree_polylines/polylines/val",
        num_points=20,
        pad_to=1024,
        augment=False,
        lidar_path=DATA_ROOT / "pdok_dataset3_polylines/lidar_patches.npy",
        lidar_stride=16,
    )
    _, target = ds[0]
    assert target["lidar"].shape == (7, 64, 64), target["lidar"].shape
    assert ds.n_lidar_channels == 7, ds.n_lidar_channels

    detr = t.DetrPolylineFromEmbeddings(
        in_dim=32,
        num_classes=2,
        num_polylines=num_polylines,
        d_model=32,
        nhead=4,
        num_encoder_layers=1,
        num_decoder_layers=1,
        dim_feedforward=32,
        dropout=0.0,
        grid_size=(grid, grid),
        num_points=num_points,
        n_lidar_channels=n_lidar_channels,
    ).eval()

    tokens = torch.randn(2, grid * grid, 32)
    lidar = torch.randn(2, n_lidar_channels, grid, grid)

    assert detr.lidar_encoder(lidar).abs().max().item() == 0.0, "branch not zero-init"
    with_lidar = detr(tokens, lidar=lidar)["pred_polylines"]
    detr.lidar_encoder, saved = None, detr.lidar_encoder
    without = detr(tokens)["pred_polylines"]
    detr.lidar_encoder = saved
    assert torch.equal(with_lidar, without), "lidar changed the output at init"

    detr.train()
    optimizer = torch.optim.AdamW(detr.parameters(), lr=1e-3)
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        out = detr(tokens, lidar=lidar)
        (out["pred_logits"].sum() + out["pred_polylines"].sum()).backward()
        grad = detr.lidar_encoder.net[-1].weight.grad
        assert grad is not None and grad.abs().max() > 0, "no gradient into the branch"
        optimizer.step()
    detr.eval()
    assert detr.lidar_encoder(lidar).abs().max() > 0, "branch stayed at zero"
    print(
        "LIDAR BRANCH OK: patch on the token grid, inert at init, learns after 3 steps"
    )


def main():
    check_lidar_branch()
    scratch = Path("/tmp") / "smoke_detr_unet_polyline"
    cfg = dict(
        mode="train",
        exp="smoke",
        save_path=scratch,
        image_dir=DATA_ROOT / "pdok_dataset3/images",
        train_polyline_dir=DATA_ROOT / "pdok_dataset3_tree_polylines/polylines/train",
        val_polyline_dir=DATA_ROOT / "pdok_dataset3_tree_polylines/polylines/val",
        pad_to=1024,
        augment=True,
        # Lidar on, the exp 4 setting. Needs the full patch file: a crop
        # without a patch stops the dataset, which is right during training.
        lidar_path=DATA_ROOT / "pdok_dataset3_polylines/lidar_patches.npy",
        lidar_stride=16,
        backbone_ckpt=CLUSTER_EXP_ROOT / "semseg_unet/4/best_4.pt",
        feature_stage="up3",
        freeze_backbone=True,
        num_points=20,
        num_polylines=60,
        num_classes=2,  # 0 hedge, 1 tree row, the exp 4 dataset
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
        infer_polyline_dir=DATA_ROOT / "pdok_dataset3_tree_polylines/polylines/val",
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
