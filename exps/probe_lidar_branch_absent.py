"""
Does lidar_path=None really remove the lidar branch, and change nothing else?

Exp 5 is the ablation of exp 4: tree rows kept, lidar off. That is only a clean
one-variable test if switching the lidar off removes the lidar branch and
leaves every other weight in place. This checks that, so the claim does not
rest on reading the constructor.

Two checks:

- build. Construct DetrPolylineFromEmbeddings twice from the same settings,
  once with n_lidar_channels=0 and once with 7, and compare the state dicts.
- checkpoints. Count lidar keys in real checkpoints, which says what a finished
  run actually saved rather than what the code would build.

Result (2026-09-23):

    build, 0 vs 7 lidar channels
      lidar keys                  0  against  9
      parameters          5,772,045  against  5,839,914
      lidar branch           67,869
      all other keys identical    True

    checkpoints
      detr_unet_polyline/2/best_2.pt   0 lidar keys, class head (2, 256)
      detr_unet_polyline/4/best_4.pt   9 lidar keys, class head (3, 256)

The build numbers count the polyline head only, which is what holds the lidar
branch. A saved checkpoint is about 20.2 M parameters because it also carries
the frozen ResNet18-UNet backbone.

The class head is (num_classes + 1, d_model) because cls_loss="ce" keeps a
no-object column, so 1 class gives 2 rows and 2 classes give 3.

So lidar_path=None builds no branch at all and touches nothing else, and exp 5
differs from exp 4 by those 67,869 parameters and nothing more.

    /home/fatemeh/miniconda3/envs/hedge/bin/python exps/probe_lidar_branch_absent.py
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import train_detr_unet_polyline as t  # noqa: E402

from hedge_seg.paths import CLUSTER_EXP_ROOT  # noqa: E402


def build(n_lidar_channels, num_classes, seed=42):
    """The exp 4 / exp 5 head, with the lidar branch on or off."""
    torch.manual_seed(seed)
    return t.DetrPolylineFromEmbeddings(
        in_dim=256,
        num_classes=num_classes,
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
        n_lidar_channels=n_lidar_channels,
    )


def n_params(state):
    return sum(v.numel() for v in state.values())


def lidar_keys(state):
    return [k for k in state if "lidar" in k]


def class_head(state):
    for k in state:
        if k.endswith("class_embed.weight"):
            return tuple(state[k].shape)
    return None


def main(cfg):
    off = build(0, cfg["num_classes"]).state_dict()
    on = build(cfg["n_lidar_channels"], cfg["num_classes"]).state_dict()
    rest = {k for k in on if "lidar" not in k}

    print(f"build, 0 vs {cfg['n_lidar_channels']} lidar channels")
    print(
        f"  lidar keys         {len(lidar_keys(off)):>10}  against  {len(lidar_keys(on))}"
    )
    print(f"  parameters         {n_params(off):>10,}  against  {n_params(on):,}")
    print(f"  lidar branch       {n_params(on) - n_params(off):>10,}")
    print(f"  all other keys identical    {set(off) == rest}")

    print("\ncheckpoints")
    for path in cfg["checkpoints"]:
        path = Path(path)
        if not path.exists():
            print(f"  {path}  missing, skipped")
            continue
        state = torch.load(path, map_location="cpu")["model"]
        name = f"{path.parent.parent.name}/{path.parent.name}/{path.name}"
        print(
            f"  {name}   {len(lidar_keys(state))} lidar keys, "
            f"class head {class_head(state)}"
        )


if __name__ == "__main__":
    root = CLUSTER_EXP_ROOT / "detr_unet_polyline"
    cfg = dict(
        # 7 is what build_lidar_patches.py writes: 6 AHN4 metrics plus the
        # presence channel. main() reads it from the patch index file.
        n_lidar_channels=7,
        num_classes=2,  # exp 4 and exp 5; exp 2 used 1
        # Add 5/best_5.pt once exp 5 has finished, to confirm the run that
        # actually trained saved no lidar weights.
        checkpoints=[
            root / "2/best_2.pt",
            root / "4/best_4.pt",
            root / "5/best_5.pt",
        ],
    )
    main(cfg)
