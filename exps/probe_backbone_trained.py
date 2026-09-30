"""
Did a run really train its backbone?

`requires_grad` cannot answer this: it is not saved in a checkpoint, so every
tensor loaded from a file says False. Instead, compare the backbone weights in
the checkpoint with the pretrained weights the run started from,
semseg_unet/4/best_4.pt. A frozen backbone is still exactly those weights. A
trained one has moved away from them.

Result (2026-09-30):

    checkpoint    weights changed   BatchNorm statistics changed
    2/best_2.pt    0 of 86           0 of 56
    9/9.pt        66 of 86          44 of 56

In exp 9 the weights moved by up to 0.016, so the backbone did train, and
slowly, as backbone_lr=1e-5 intends. What did not change are the UNet layers
after up3 (up2, up1, up0, head). The model stops at up3, so they get no
gradient. Everything the model uses did train.

    /home/fatemeh/miniconda3/envs/hedge/bin/python exps/probe_backbone_trained.py
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hedge_seg.paths import CLUSTER_EXP_ROOT  # noqa: E402


def load_state(path):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    return ckpt["model"] if "model" in ckpt else ckpt


def main(cfg):
    pretrained = load_state(cfg["pretrained"])
    for path in cfg["checkpoints"]:
        state = load_state(path)
        # Weights are what training changes. BatchNorm running statistics also
        # change whenever the backbone runs in train mode, so count them apart.
        counts = {"weights": [0, 0, 0.0], "BatchNorm statistics": [0, 0, 0.0]}
        unchanged = set()
        for name, weight in pretrained.items():
            if not weight.dtype.is_floating_point:
                continue  # BatchNorm step counters
            kind = "BatchNorm statistics" if "running_" in name else "weights"
            diff = (state["backbone.unet." + name] - weight).abs().max().item()
            counts[kind][1] += 1
            counts[kind][2] = max(counts[kind][2], diff)
            if diff > 0:
                counts[kind][0] += 1
            else:
                unchanged.add(name.split(".")[0])
        print(f"{Path(path).parent.name}/{Path(path).name}")
        for kind, (n_changed, n_total, largest) in counts.items():
            print(
                f"  {kind}: {n_changed} of {n_total} tensors changed, "
                f"largest change {largest:.2g}"
            )
        if any(c[0] for c in counts.values()):
            print(f"  modules with nothing changed: {sorted(unchanged)}")


if __name__ == "__main__":
    root = CLUSTER_EXP_ROOT / "detr_unet_polyline"
    cfg = dict(
        pretrained=CLUSTER_EXP_ROOT / "semseg_unet/4/best_4.pt",
        checkpoints=[
            root / "2/best_2.pt",  # backbone frozen
            root / "9/9.pt",  # backbone trained at lr 1e-5
        ],
    )
    main(cfg)
