"""
How big is the polyline model, and where do its parameters sit?

Builds the exp 9 model of scripts/train_detr_unet_polyline.py and loads
best_9.pt into it with strict=True, so the counts are of the model that ran,
not of the code as it reads. Exps 2, 7 and 8 have the same architecture.

Counts parameters only. A checkpoint also stores the BatchNorm running
statistics, which is why exps/probe_lidar_branch_absent.py, counting every
tensor in a checkpoint, gets 20,169,322.

Result (2026-10-07):

    total                                  20,157,646
    backbone, ResNet18-UNet                14,385,858
      used (stem, enc1-enc4, up3)          13,536,832
      never used (up2, up1, up0, head)        849,026
    polyline head                           5,771,788
      transformer encoder, 1 layer            789,760
      transformer decoder, 4 layers         4,214,272
      the rest                                767,756

The model reads its features at up3, so the UNet layers after it are carried
along but get no gradient (exps/probe_backbone_trained.py). Tree rows add 257
parameters (one class row) and lidar adds 67,869
(exps/probe_lidar_branch_absent.py).

    /home/fatemeh/miniconda3/envs/hedge/bin/python exps/probe_model_size.py
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import train_detr_unet_polyline as t  # noqa: E402

from hedge_seg.paths import CLUSTER_EXP_ROOT  # noqa: E402


def n_params(*modules):
    return sum(p.numel() for m in modules for p in m.parameters())


def main(cfg):
    backbone = t.ResNet18UNetFeatures(feature_stage="up3", frozen=False)
    head = t.DetrPolylineFromEmbeddings(
        in_dim=256,
        num_classes=1,
        num_polylines=60,
        d_model=256,
        nhead=8,
        num_encoder_layers=1,
        num_decoder_layers=4,
        dim_feedforward=1024,
        dropout=0.01,
        grid_size=(64, 64),
        num_points=20,
        aux_loss=True,
        query_embed_mode="detr",
        with_refine=True,
        cls_loss="ce",
        n_lidar_channels=0,
    )
    model = t.DetrPolylineFromImage(backbone, head)
    ckpt = torch.load(cfg["ckpt"], map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"], strict=True)
    print(f"loaded {cfg['ckpt']} with strict=True")

    u = backbone.unet
    used = n_params(u.stem, u.enc1, u.enc2, u.enc3, u.enc4, u.up3)
    enc = n_params(head.transformer.encoder)
    dec = n_params(head.transformer.decoder)
    rows = [
        ("total", n_params(model)),
        ("backbone, ResNet18-UNet", n_params(backbone)),
        ("  used (stem, enc1-enc4, up3)", used),
        ("  never used (up2, up1, up0, head)", n_params(backbone) - used),
        ("polyline head", n_params(head)),
        ("  transformer encoder, 1 layer", enc),
        ("  transformer decoder, 4 layers", dec),
        ("  the rest", n_params(head) - enc - dec),
    ]
    for name, n in rows:
        print(f"{name:<36} {n:>12,}")


if __name__ == "__main__":
    cfg = dict(
        ckpt=CLUSTER_EXP_ROOT / "detr_unet_polyline/9/best_9.pt",
    )
    main(cfg)
