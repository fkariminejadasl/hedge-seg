import sys
from pathlib import Path

from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from train_detr_unet_polyline import build_optimizer  # noqa: E402


class _Model(nn.Module):
    """The two top-level names the training model uses: backbone and detr."""

    def __init__(self, freeze_backbone):
        super().__init__()
        self.backbone = nn.Linear(3, 4)
        self.detr = nn.Linear(4, 2)
        for p in self.backbone.parameters():
            p.requires_grad_(not freeze_backbone)


def test_frozen_backbone_gives_one_group_of_head_weights():
    model = _Model(freeze_backbone=True)
    optimizer = build_optimizer(model, 1e-4, 1e-5, 1e-2)
    assert len(optimizer.param_groups) == 1
    group = optimizer.param_groups[0]
    assert group["lr"] == 1e-4
    assert group["params"] == list(model.detr.parameters())


def test_trained_backbone_gets_its_own_learning_rate():
    model = _Model(freeze_backbone=False)
    optimizer = build_optimizer(model, 1e-4, 1e-5, 1e-2)
    head, backbone = optimizer.param_groups
    assert (head["name"], head["lr"]) == ("head", 1e-4)
    assert (backbone["name"], backbone["lr"]) == ("backbone", 1e-5)
    assert backbone["params"] == list(model.backbone.parameters())


def test_backbone_lr_none_means_max_lr():
    optimizer = build_optimizer(_Model(freeze_backbone=False), 1e-4, None, 1e-2)
    assert optimizer.param_groups[1]["lr"] == 1e-4
