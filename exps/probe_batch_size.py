"""
Find the largest batch size that fits on the current GPU for
scripts/train_detr_unet_polyline.py.

Runs a real training step (forward + loss + backward + optimizer) per batch
size, since backward and the optimizer state dominate peak memory, and reports
peak memory and seconds per image. Run it on the same GPU type as the training
job (e.g. inside a short slurm job on gpu_a100).
"""

import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import train_detr_unet_polyline as t  # noqa: E402

from hedge_seg.paths import CLUSTER_EXP_ROOT, DATA_ROOT, print_roots  # noqa: E402

BATCH_SIZES = [4, 8, 16, 24, 32, 48, 64]
FEATURE_STAGE = "up3"
NUM_POLYLINES = 60
N_STEPS = 3


def build(feature_stage: str, num_polylines: int, device):
    stage = t.FEATURE_STAGES[feature_stage]
    grid = 1024 // stage["stride"]
    backbone = t.ResNet18UNetFeatures(
        ckpt_path=CLUSTER_EXP_ROOT / "semseg_unet/4/best_4.pt",
        feature_stage=feature_stage,
        frozen=True,
    )
    detr = t.DetrPolylineFromEmbeddings(
        in_dim=stage["channels"],
        num_classes=1,
        num_polylines=num_polylines,
        d_model=256,
        nhead=8,
        num_encoder_layers=1,
        num_decoder_layers=4,
        dim_feedforward=1024,
        dropout=0.01,
        grid_size=(grid, grid),
        num_points=20,
        aux_loss=True,
        with_refine=True,
    )
    return t.DetrPolylineFromImage(backbone, detr).to(device)


def main():
    print_roots()
    device = torch.device("cuda")
    print(torch.cuda.get_device_name(0))
    total_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"total GPU memory: {total_gb:.1f} GB\n")

    ds = t.DetrPolylineImageDataset(
        image_dir=DATA_ROOT / "pdok_dataset3/images",
        polyline_dir=DATA_ROOT / "pdok_dataset3_polylines/polylines/train",
        num_points=20,
        pad_to=1024,
        augment=True,
    )
    matcher = t.HungarianMatcherPolyline(t.MatcherCost(1.0, 5.0, 0.0, 0.0))
    criterion = t.DetrPolylineCriterion(
        num_classes=1,
        matcher=matcher,
        eos_coef=0.05,
        loss_poly=5.0,
        loss_bbox=0.0,
        loss_bbox_giou=0.0,
        loss_smooth=0.0,
        loss_card=0.0,
        loss_len=0.0,
        loss_dir=0.0,
        aux_weight=0.5,
    ).to(device)

    print(f"{'batch':>6} {'peak GB':>9} {'s/step':>8} {'s/image':>9}  status")
    for bs in BATCH_SIZES:
        if bs > len(ds):
            continue
        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            model = build(FEATURE_STAGE, NUM_POLYLINES, device)
            model.train()
            opt = torch.optim.AdamW(
                [p for p in model.parameters() if p.requires_grad], lr=1e-4
            )
            batch = [ds[i] for i in range(bs)]
            images, targets = t.detr_polyline_collate_fn(batch)
            images = images.to(device)
            for tg in targets:
                tg["labels"] = tg["labels"].to(device)
                tg["polylines"] = tg["polylines"].to(device)

            for step in range(N_STEPS):
                if step == 1:
                    torch.cuda.synchronize()
                    t0 = time.time()
                opt.zero_grad(set_to_none=True)
                losses = criterion(model(images), targets)
                losses["loss_total"].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            torch.cuda.synchronize()
            dt = (time.time() - t0) / (N_STEPS - 1)
            peak = torch.cuda.max_memory_allocated() / 1e9
            print(f"{bs:>6} {peak:>9.2f} {dt:>8.3f} {dt / bs:>9.4f}  ok")

            del model, opt, images, targets, losses
        except torch.cuda.OutOfMemoryError:
            print(f"{bs:>6} {'-':>9} {'-':>8} {'-':>9}  OOM")
            break
        finally:
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
