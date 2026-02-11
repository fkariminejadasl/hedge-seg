from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf
from scipy.optimize import linear_sum_assignment
from torch.utils import tensorboard
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# -------------------------
# Utilities: boxes, IoU, GIoU
# -------------------------


def box_xyxy_to_cxcywh(x: torch.Tensor) -> torch.Tensor:
    # x: (..., 4) in xyxy
    x0, y0, x1, y1 = x.unbind(-1)
    cx = (x0 + x1) / 2
    cy = (y0 + y1) / 2
    w = (x1 - x0).clamp(min=0)
    h = (y1 - y0).clamp(min=0)
    return torch.stack([cx, cy, w, h], dim=-1)


def box_cxcywh_to_xyxy(x: torch.Tensor) -> torch.Tensor:
    cx, cy, w, h = x.unbind(-1)
    x0 = cx - 0.5 * w
    y0 = cy - 0.5 * h
    x1 = cx + 0.5 * w
    y1 = cy + 0.5 * h
    return torch.stack([x0, y0, x1, y1], dim=-1)


def box_area_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    # boxes: (N,4) xyxy
    return (boxes[:, 2] - boxes[:, 0]).clamp(min=0) * (boxes[:, 3] - boxes[:, 1]).clamp(
        min=0
    )


def box_iou_xyxy(
    boxes1: torch.Tensor, boxes2: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    # returns iou, union
    area1 = box_area_xyxy(boxes1)
    area2 = box_area_xyxy(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[None, :, :2])  # (N,M,2)
    rb = torch.min(boxes1[:, None, 2:], boxes2[None, :, 2:])  # (N,M,2)

    wh = (rb - lt).clamp(min=0)  # (N,M,2)
    inter = wh[:, :, 0] * wh[:, :, 1]  # (N,M)

    union = area1[:, None] + area2[None, :] - inter
    iou = inter / union.clamp(min=1e-6)
    return iou, union


def generalized_box_iou_xyxy(
    boxes1: torch.Tensor, boxes2: torch.Tensor
) -> torch.Tensor:
    # GIoU for xyxy boxes, outputs (N,M)
    iou, union = box_iou_xyxy(boxes1, boxes2)

    lt = torch.min(boxes1[:, None, :2], boxes2[None, :, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[None, :, 2:])

    wh = (rb - lt).clamp(min=0)
    area_c = wh[:, :, 0] * wh[:, :, 1]  # enclosing area

    return iou - (area_c - union) / area_c.clamp(min=1e-6)


# -------------------------
# Positional encoding (2D sine)
# -------------------------


class PositionEmbeddingSine2D(nn.Module):
    """
    Standard DETR sine-cosine 2D positional embedding for (H,W) grid.
    Returns (B, HW, D).
    """

    def __init__(
        self,
        num_pos_feats: int = 128,
        temperature: int = 10000,
        normalize: bool = True,
        scale: Optional[float] = None,
    ):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        self.scale = scale if scale is not None else 2 * torch.pi

    def forward(self, B: int, H: int, W: int, device: torch.device) -> torch.Tensor:
        y_embed = torch.arange(H, device=device).unsqueeze(1).repeat(1, W)  # (H,W)
        x_embed = torch.arange(W, device=device).unsqueeze(0).repeat(H, 1)  # (H,W)

        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (H - 1 + eps) * self.scale
            x_embed = x_embed / (W - 1 + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, device=device, dtype=torch.float32)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[..., None] / dim_t  # (H,W,F)
        pos_y = y_embed[..., None] / dim_t

        pos_x = torch.stack(
            (pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=-1
        ).flatten(-2)
        pos_y = torch.stack(
            (pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()), dim=-1
        ).flatten(-2)

        pos = torch.cat((pos_y, pos_x), dim=-1)  # (H,W,2F)
        pos = pos.view(H * W, -1).unsqueeze(0).repeat(B, 1, 1)  # (B,HW,D)
        return pos


# -------------------------
# DETR model that consumes DINO embeddings
# -------------------------


class DetrFromEmbeddings(nn.Module):
    """
    Input: x (B, 196, 1024) from DINO v3.
    Reshape: 14x14 tokens.
    Output: pred_logits (B, Q, K+1), pred_boxes (B, Q, 4) in normalized cxcywh [0,1]
    """

    def __init__(
        self,
        in_dim: int = 1024,
        num_classes: int = 1,  # excluding "no-object"
        num_queries: int = 100,
        d_model: int = 256,
        nhead: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        grid_size: Tuple[int, int] = (14, 14),
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_queries = num_queries
        self.grid_h, self.grid_w = grid_size

        self.input_proj = nn.Linear(in_dim, d_model)
        self.pos_embed = PositionEmbeddingSine2D(num_pos_feats=d_model // 2)

        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,  # so we use (B, S, D)
        )

        self.query_embed = nn.Embedding(num_queries, d_model)

        # Heads
        self.class_embed = nn.Linear(d_model, num_classes + 1)  # +1 for "no-object"
        self.bbox_embed = MLPBox(d_model, d_model, 4, num_layers=3)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # x: (B, 196, 1024)
        B, L, C = x.shape
        assert (
            L == self.grid_h * self.grid_w
        ), f"Expected {self.grid_h*self.grid_w} tokens, got {L}"

        x = self.input_proj(x)  # (B, L, d_model)

        pos = self.pos_embed(
            B=B, H=self.grid_h, W=self.grid_w, device=x.device
        )  # (B,L,d_model)
        src = x + pos

        # Queries
        query = self.query_embed.weight.unsqueeze(0).repeat(B, 1, 1)  # (B,Q,D)

        # Transformer: encoder(src), decoder(tgt=query, memory=enc)
        memory = self.transformer.encoder(src)  # (B,L,D)
        hs = self.transformer.decoder(tgt=query, memory=memory)  # (B,Q,D)

        logits = self.class_embed(hs)  # (B,Q,K+1)
        boxes = self.bbox_embed(hs).sigmoid()  # normalized cxcywh in [0,1]

        return {"pred_logits": logits, "pred_boxes": boxes}


class MLPBox(nn.Module):
    def __init__(
        self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int
    ):
        super().__init__()
        layers = []
        for i in range(num_layers):
            in_d = input_dim if i == 0 else hidden_dim
            out_d = output_dim if i == num_layers - 1 else hidden_dim
            layers.append(nn.Linear(in_d, out_d))
        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = F.relu(x)
        return x


# -------------------------
# Matcher + Criterion (losses)
# -------------------------


@dataclass
class MatcherCost:
    class_cost: float = 1.0
    bbox_cost: float = 5.0
    giou_cost: float = 2.0


class HungarianMatcher(nn.Module):
    def __init__(self, cost: MatcherCost):
        super().__init__()
        self.cost = cost

    @torch.no_grad()
    def forward(
        self, outputs: Dict[str, torch.Tensor], targets: List[Dict[str, torch.Tensor]]
    ):
        """
        outputs:
          pred_logits: (B,Q,K+1)
          pred_boxes:  (B,Q,4) normalized cxcywh
        targets: list length B with:
          labels: (Ni,)
          boxes:  (Ni,4) normalized cxcywh
        returns:
          list of size B: (idx_pred, idx_tgt)
        """
        bs, num_queries = outputs["pred_logits"].shape[:2]
        out_prob = outputs["pred_logits"].softmax(-1)  # (B,Q,K+1)
        out_bbox = outputs["pred_boxes"]  # (B,Q,4)

        indices = []
        for b in range(bs):
            tgt_ids = targets[b]["labels"]  # (Ni,)
            tgt_bbox = targets[b]["boxes"]  # (Ni,4)

            if tgt_bbox.numel() == 0:
                indices.append(
                    (
                        torch.empty(0, dtype=torch.int64),
                        torch.empty(0, dtype=torch.int64),
                    )
                )
                continue

            # Classification cost: negative prob of tgt class
            cost_class = -out_prob[b][:, tgt_ids]  # (Q,Ni)

            # Bbox L1 cost
            cost_bbox = torch.cdist(out_bbox[b], tgt_bbox, p=1)  # (Q,Ni)

            # GIoU cost (need xyxy)
            out_xyxy = box_cxcywh_to_xyxy(out_bbox[b])
            tgt_xyxy = box_cxcywh_to_xyxy(tgt_bbox)
            cost_giou = -generalized_box_iou_xyxy(out_xyxy, tgt_xyxy)  # (Q,Ni)

            C = (
                self.cost.class_cost * cost_class
                + self.cost.bbox_cost * cost_bbox
                + self.cost.giou_cost * cost_giou
            )
            C = C.cpu()

            row_ind, col_ind = linear_sum_assignment(C)
            indices.append(
                (
                    torch.as_tensor(row_ind, dtype=torch.int64),
                    torch.as_tensor(col_ind, dtype=torch.int64),
                )
            )
        return indices


class DetrCriterion(nn.Module):
    def __init__(
        self,
        num_classes: int,
        matcher: HungarianMatcher,
        eos_coef: float = 0.1,
        loss_bbox: float = 5.0,
        loss_giou: float = 2.0,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.eos_coef = eos_coef
        self.loss_bbox_w = loss_bbox
        self.loss_giou_w = loss_giou

        # weight for "no-object" class
        empty_weight = torch.ones(num_classes + 1)
        empty_weight[-1] = eos_coef
        self.register_buffer("empty_weight", empty_weight)

    def forward(
        self, outputs: Dict[str, torch.Tensor], targets: List[Dict[str, torch.Tensor]]
    ):
        indices = self.matcher(outputs, targets)

        loss_ce = self.loss_labels(outputs, targets, indices)
        loss_bbox, loss_giou = self.loss_boxes(outputs, targets, indices)

        losses = {
            "loss_ce": loss_ce,
            "loss_bbox": loss_bbox,
            "loss_giou": loss_giou,
        }
        total = loss_ce + self.loss_bbox_w * loss_bbox + self.loss_giou_w * loss_giou
        losses["loss_total"] = total
        return losses

    def loss_labels(self, outputs, targets, indices):
        # outputs["pred_logits"]: (B,Q,K+1)
        src_logits = outputs["pred_logits"]
        B, Q, K1 = src_logits.shape

        # Start as "no-object"
        target_classes = torch.full(
            (B, Q), self.num_classes, dtype=torch.int64, device=src_logits.device
        )

        for b, (src_idx, tgt_idx) in enumerate(indices):
            if len(src_idx) == 0:
                continue
            target_classes[b, src_idx] = targets[b]["labels"][tgt_idx]

        loss_ce = F.cross_entropy(
            src_logits.transpose(1, 2), target_classes, weight=self.empty_weight
        )
        return loss_ce

    def loss_boxes(self, outputs, targets, indices):
        src_boxes = outputs["pred_boxes"]  # (B,Q,4)
        loss_bbox = torch.tensor(0.0, device=src_boxes.device)
        loss_giou = torch.tensor(0.0, device=src_boxes.device)

        n_matched = 0
        for b, (src_idx, tgt_idx) in enumerate(indices):
            if len(src_idx) == 0:
                continue
            s = src_boxes[b, src_idx]
            t = targets[b]["boxes"][tgt_idx]

            loss_bbox = loss_bbox + F.l1_loss(s, t, reduction="sum")

            s_xyxy = box_cxcywh_to_xyxy(s)
            t_xyxy = box_cxcywh_to_xyxy(t)
            giou = generalized_box_iou_xyxy(s_xyxy, t_xyxy).diag()
            loss_giou = loss_giou + (1 - giou).sum()

            n_matched += len(src_idx)

        n_matched = max(n_matched, 1)
        loss_bbox = loss_bbox / n_matched
        loss_giou = loss_giou / n_matched
        return loss_bbox, loss_giou


# -------------------------
# Dataset and collate
# -------------------------


class DetrEmbDataset(Dataset):
    """
    Expected per file:
      feat: (196,1024) or (N,196,1024)
      boxes: (num_obj,4) in xyxy pixel coords OR already normalized
      labels: (num_obj,)
      optionally: image_size: (H,W) to normalize pixel boxes
    """

    def __init__(self, embed_dir: Path, normalize_boxes: bool = True):
        self.files = list(embed_dir.glob("*.npz"))
        self.normalize_boxes = normalize_boxes

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx: int):
        d = np.load(self.files[idx])

        feat = d["feat"]
        boxes = d["boxes"]  # (num_obj,4) xyxy
        labels = d["labels"]  # (num_obj,)
        feat = torch.from_numpy(np.ascontiguousarray(feat)).float()  # (196,1024)
        boxes = torch.from_numpy(np.ascontiguousarray(boxes)).float()
        labels = torch.from_numpy(np.ascontiguousarray(labels)).long()

        if self.normalize_boxes:
            if "image_size" in d:
                H, W = d["image_size"].tolist()
            else:
                # If you do not store image_size, put it in the npz.
                raise ValueError(
                    f"{self.files[idx].name} missing image_size for box normalization"
                )

            # xyxy pixel -> xyxy normalized
            boxes[:, 0::2] = boxes[:, 0::2] / float(W)
            boxes[:, 1::2] = boxes[:, 1::2] / float(H)
            boxes = boxes.clamp(0, 1)

        # DETR expects normalized cxcywh
        boxes = box_xyxy_to_cxcywh(boxes)

        target = {"labels": labels, "boxes": boxes}
        return feat, target


def detr_collate_fn(batch):
    feats, targets = zip(*batch)
    feats = torch.stack(feats, dim=0)  # (B,196,1024)
    targets = list(targets)
    return feats, targets


# -------------------------
# Training / eval
# -------------------------


def tb_add_losses(writer, epoch: int, losses: dict, stage: str):
    losses_dict = dict()
    for k, v in losses.items():
        losses_dict[f"{stage}_{k}"] = v
    writer.add_scalars("losses", losses_dict, epoch)


def train_one_epoch(loader, model, criterion, optimizer, device):
    model.train()
    sums = {"loss_ce": 0.0, "loss_bbox": 0.0, "loss_giou": 0.0, "loss_total": 0.0}
    n = 0

    for feats, targets in loader:
        feats = feats.to(device)
        for t in targets:
            t["labels"] = t["labels"].to(device)
            t["boxes"] = t["boxes"].to(device)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(feats)
        loss_dict = criterion(outputs, targets)  # returns dict with loss_total
        loss = loss_dict["loss_total"]
        loss.backward()
        optimizer.step()

        bs = feats.size(0)
        n += bs
        for k in sums:
            sums[k] += float(loss_dict[k].detach().item()) * bs

    for k in sums:
        sums[k] /= max(n, 1)
    return sums


@torch.no_grad()
def eval_one_epoch(loader, model, criterion, device):
    model.eval()
    sums = {"loss_ce": 0.0, "loss_bbox": 0.0, "loss_giou": 0.0, "loss_total": 0.0}
    n = 0

    for feats, targets in loader:
        feats = feats.to(device)
        for t in targets:
            t["labels"] = t["labels"].to(device)
            t["boxes"] = t["boxes"].to(device)

        outputs = model(feats)
        loss_dict = criterion(outputs, targets)

        bs = feats.size(0)
        n += bs
        for k in sums:
            sums[k] += float(loss_dict[k].detach().item()) * bs

    for k in sums:
        sums[k] /= max(n, 1)
    return sums


# -------------------------
# Main
# -------------------------


def main():
    torch.manual_seed(0)
    cfg = dict(
        exp="detr_from_dino_test",
        save_path=Path("/home/fatemeh/Downloads/hedg/results/training"),
        embed_dir=Path(
            "/home/fatemeh/Downloads/hedg/results/test_dataset_with_osm/packed_pos_npz"
        ),  # contains *.npz
        n_epochs=5,
    )
    cfg = OmegaConf.create(cfg)

    dataset = DetrEmbDataset(embed_dir=cfg.embed_dir, normalize_boxes=True)
    n_train = int(0.8 * len(dataset))
    n_val = len(dataset) - n_train
    train_ds, val_ds = torch.utils.data.random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(
        train_ds,
        batch_size=8,
        shuffle=True,
        num_workers=2,
        collate_fn=detr_collate_fn,
    )
    eval_loader = DataLoader(
        val_ds,
        batch_size=8,
        shuffle=False,
        num_workers=2,
        collate_fn=detr_collate_fn,
    )

    num_classes = 1  # for one foreground class;
    model = DetrFromEmbeddings(
        in_dim=1024,
        num_classes=num_classes,
        num_queries=100,
        d_model=256,
        nhead=8,
        num_encoder_layers=4,
        num_decoder_layers=4,
        dim_feedforward=1024,
        dropout=0.1,
        grid_size=(14, 14),
    )

    matcher = HungarianMatcher(
        MatcherCost(class_cost=1.0, bbox_cost=5.0, giou_cost=2.0)
    )
    criterion = DetrCriterion(
        num_classes=num_classes,
        matcher=matcher,
        eos_coef=0.1,
        loss_bbox=5.0,
        loss_giou=2.0,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

    best_val = 1e9
    with tensorboard.SummaryWriter(cfg.save_path / f"tensorboard/{cfg.exp}") as writer:
        for epoch in tqdm(range(1, cfg.n_epochs + 1)):
            s_time = datetime.now().replace(microsecond=0)
            print(f"Epoch {epoch:03d} starting at {s_time}")
            train_losses = train_one_epoch(
                train_loader, model, criterion, optimizer, device
            )
            eval_losses = eval_one_epoch(eval_loader, model, criterion, device)
            e_time = datetime.now().replace(microsecond=0)
            print(
                f"Epoch {epoch:03d} "
                f"train_total={train_losses['loss_total']:.4f} "
                f"(ce={train_losses['loss_ce']:.4f}, l1={train_losses['loss_bbox']:.4f}, giou={train_losses['loss_giou']:.4f})"
                f"eval_total={eval_losses['loss_total']:.4f} "
                f"(ce={eval_losses['loss_ce']:.4f}, l1={eval_losses['loss_bbox']:.4f}, giou={eval_losses['loss_giou']:.4f})"
            )
            print(
                f"Epoch {epoch:03d} finished at {e_time} (duration {e_time - s_time})"
            )
            tb_add_losses(writer, epoch, train_losses, "train")
            tb_add_losses(writer, epoch, eval_losses, "eval")

            if eval_losses["loss_total"] < best_val:
                best_val = eval_losses["loss_total"]
                torch.save(
                    {"model": model.state_dict(), "epoch": epoch},
                    cfg.save_path / "best_detr_from_dino.pt",
                )
                print(f"Saved best: {best_val:.4f} at epoch {epoch}")


if __name__ == "__main__":
    main()
