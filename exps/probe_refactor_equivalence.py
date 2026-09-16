"""
Did a refactor change the model, or only the code?

Re-running an old checkpoint and diffing the saved predictions does not answer
this. After adding the lidar branch, exp 2 re-inferred with lidar off gave the
same number of predictions on every crop but coordinates up to 0.19 px
different from the run stored in July, which looks like a regression and is not.

This separates the two causes:

- code: build the current model and the one from a git revision, load the same
  weights into both, and compare the outputs on CPU. Bitwise equal means the
  refactor changed nothing.
- device: run the current model twice on the GPU, then once on the CPU. The two
  GPU runs are bitwise equal, so the GPU is deterministic within a session, but
  GPU and CPU differ, because cuDNN TF32 is on. A driver or library change
  between two months is enough to move the last bits the same way.

Result (2026-09-16, best_2.pt, lidar off, against git ed60e3b):

    old code vs new code, CPU        bitwise identical (max diff 0.0)
    GPU run twice                    bitwise identical
    GPU vs CPU                       0.64 px

(The gap is measured on random input here, so it is the size of the effect
rather than a constant. On real crops it was 0.58 px.)

So use this, not a diff of two inference runs, to check that a refactor left a
model alone. Point `revision` at the last commit before the change.

    /home/fatemeh/miniconda3/envs/hedge/bin/python exps/probe_refactor_equivalence.py
"""

import importlib.util
import subprocess
import sys
import tempfile
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import train_detr_unet_polyline as new  # noqa: E402

from hedge_seg.paths import CLUSTER_EXP_ROOT  # noqa: E402


def load_revision(repo: Path, revision: str, script: str):
    """Import a script as it was at a git revision, without touching the tree."""
    text = subprocess.run(
        ["git", "-C", str(repo), "show", f"{revision}:{script}"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    path = Path(tempfile.mkdtemp()) / "train_old.py"
    path.write_text(text)
    spec = importlib.util.spec_from_file_location("train_old", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build(module, ckpt_path, cfg, **extra):
    """The exp 2 model: frozen up3 backbone, softmax head, one class, no lidar."""
    torch.manual_seed(cfg["seed"])
    backbone = module.ResNet18UNetFeatures(
        ckpt_path=None, feature_stage="up3", frozen=True
    )
    detr = module.DetrPolylineFromEmbeddings(
        in_dim=256,
        num_classes=1,
        num_polylines=60,
        d_model=256,
        nhead=8,
        num_encoder_layers=1,
        num_decoder_layers=4,
        dim_feedforward=1024,
        dropout=0.01,
        grid_size=(cfg["grid"], cfg["grid"]),
        num_points=20,
        aux_loss=True,
        query_embed_mode="detr",
        with_refine=True,
        cls_loss="ce",
        **extra,
    )
    model = module.DetrPolylineFromImage(backbone, detr).eval()
    state = torch.load(ckpt_path, map_location="cpu")["model"]
    model.load_state_dict(state, strict=True)
    return model


def main(cfg):
    repo = Path(__file__).resolve().parents[1]
    ckpt = Path(cfg["ckpt"])
    old = load_revision(repo, cfg["revision"], cfg["script"])

    model_old = build(old, ckpt, cfg)
    model_new = build(new, ckpt, cfg, n_lidar_channels=0)
    print(f"{cfg['revision']} and the working tree both load {ckpt.name}, strict=True")

    torch.manual_seed(cfg["seed"] + 1)
    images = torch.randn(cfg["batch"], 3, cfg["grid"] * 16, cfg["grid"] * 16)

    with torch.no_grad():
        a, b = model_old(images), model_new(images)
    print("\nold code against new code, on CPU")
    for key in ("pred_logits", "pred_polylines", "pred_boxes"):
        diff = (a[key] - b[key]).abs().max().item()
        print(
            f"  {key:16s} identical: {torch.equal(a[key], b[key])}, max diff {diff:.1e}"
        )

    if not torch.cuda.is_available():
        print("\nno GPU, skipping the device half")
        return

    print(
        f"\ncuDNN TF32 {torch.backends.cudnn.allow_tf32}, "
        f"matmul TF32 {torch.backends.cuda.matmul.allow_tf32}"
    )
    model_gpu = model_new.cuda()
    images_gpu = images.cuda()
    with torch.no_grad():
        first = model_gpu(images_gpu)["pred_polylines"]
        second = model_gpu(images_gpu)["pred_polylines"]
    print(f"  same GPU run twice, identical: {torch.equal(first, second)}")
    # Points are normalised by pad_to - 1, so multiply to read the gap in pixels.
    gap = (first.cpu() - b["pred_polylines"]).abs().max().item()
    print(f"  GPU against CPU: {gap * (cfg['grid'] * 16 - 1):.2f} px")


if __name__ == "__main__":
    cfg = dict(
        # The last commit before the change being checked.
        revision="ed60e3b",
        script="scripts/train_detr_unet_polyline.py",
        ckpt=CLUSTER_EXP_ROOT / "detr_unet_polyline/2/best_2.pt",
        grid=64,  # up3 on a 1024 px crop
        batch=2,
        seed=0,
    )
    main(cfg)
