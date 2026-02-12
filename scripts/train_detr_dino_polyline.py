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
# Utilities: boxes for optional GIoU-from-polyline
# -------------------------


def box_area_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    return (boxes[:, 2] - boxes[:, 0]).clamp(min=0) * (boxes[:, 3] - boxes[:, 1]).clamp(
        min=0
    )


def box_iou_xyxy(
    boxes1: torch.Tensor, boxes2: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    area1 = box_area_xyxy(boxes1)
    area2 = box_area_xyxy(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[None, :, :2])  # (N,M,2)
    rb = torch.min(boxes1[:, None, 2:], boxes2[None, :, 2:])  # (N,M,2)

    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]

    union = area1[:, None] + area2[None, :] - inter
    iou = inter / union.clamp(min=1e-6)
    return iou, union


def generalized_box_iou_xyxy(
    boxes1: torch.Tensor, boxes2: torch.Tensor
) -> torch.Tensor:
    iou, union = box_iou_xyxy(boxes1, boxes2)

    lt = torch.min(boxes1[:, None, :2], boxes2[None, :, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[None, :, 2:])

    wh = (rb - lt).clamp(min=0)
    area_c = wh[:, :, 0] * wh[:, :, 1]
    return iou - (area_c - union) / area_c.clamp(min=1e-6)


def polyline_to_bbox_xyxy(poly: torch.Tensor) -> torch.Tensor:
    """
    poly: (..., K, 2) normalized in [0,1]
    returns: (..., 4) xyxy normalized
    """
    x = poly[..., :, 0]
    y = poly[..., :, 1]
    x0 = x.min(dim=-1).values
    y0 = y.min(dim=-1).values
    x1 = x.max(dim=-1).values
    y1 = y.max(dim=-1).values
    return torch.stack([x0, y0, x1, y1], dim=-1)


# -------------------------
# Positional encoding (2D sine)
# -------------------------


class PositionEmbeddingSine2D(nn.Module):
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
        y_embed = torch.arange(H, device=device).unsqueeze(1).repeat(1, W)
        x_embed = torch.arange(W, device=device).unsqueeze(0).repeat(H, 1)

        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (H - 1 + eps) * self.scale
            x_embed = x_embed / (W - 1 + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, device=device, dtype=torch.float32)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[..., None] / dim_t
        pos_y = y_embed[..., None] / dim_t

        pos_x = torch.stack(
            (pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=-1
        ).flatten(-2)
        pos_y = torch.stack(
            (pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()), dim=-1
        ).flatten(-2)

        pos = torch.cat((pos_y, pos_x), dim=-1)  # (H,W,D)
        pos = pos.view(H * W, -1).unsqueeze(0).repeat(B, 1, 1)  # (B,HW,D)
        return pos


# -------------------------
# DETR model predicting polylines
# -------------------------


class MLP(nn.Module):
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


class DetrPolylineFromEmbeddings(nn.Module):
    """
    Input: x (B, 196, 1024) from DINO v3.
    Output:
      pred_logits: (B, Q, num_classes+1)
      pred_polylines: (B, Q, K, 2) normalized in [0,1]
    """

    def __init__(
        self,
        in_dim: int = 1024,
        num_classes: int = 1,
        num_queries: int = 100,
        d_model: int = 256,
        nhead: int = 8,
        num_encoder_layers: int = 4,
        num_decoder_layers: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        grid_size: Tuple[int, int] = (14, 14),
        num_points: int = 20,  # K
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_queries = num_queries
        self.grid_h, self.grid_w = grid_size
        self.num_points = num_points

        self.input_proj = nn.Linear(in_dim, d_model)
        self.pos_embed = PositionEmbeddingSine2D(num_pos_feats=d_model // 2)

        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )

        self.query_embed = nn.Embedding(num_queries, d_model)

        self.class_embed = nn.Linear(d_model, num_classes + 1)
        self.poly_embed = MLP(d_model, d_model, 2 * num_points, num_layers=3)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        B, L, _ = x.shape
        assert (
            L == self.grid_h * self.grid_w
        ), f"Expected {self.grid_h*self.grid_w} tokens, got {L}"

        x = self.input_proj(x)  # (B,L,D)
        pos = self.pos_embed(
            B=B, H=self.grid_h, W=self.grid_w, device=x.device
        )  # (B,L,D)
        src = x + pos

        memory = self.transformer.encoder(src)  # (B,L,D)

        query = self.query_embed.weight.unsqueeze(0).repeat(B, 1, 1)  # (B,Q,D)
        tgt = torch.zeros_like(query)
        hs = self.transformer.decoder(tgt=tgt + query, memory=memory)  # (B,Q,D)

        logits = self.class_embed(hs)  # (B,Q,K+1)

        poly = self.poly_embed(hs)  # (B,Q,2*num_points)
        poly = poly.view(
            B, self.num_queries, self.num_points, 2
        ).sigmoid()  # normalized [0,1]

        return {"pred_logits": logits, "pred_polylines": poly}


# -------------------------
# Matcher + Criterion for polylines
# -------------------------


@dataclass
class MatcherCost:
    class_cost: float = 1.0
    poly_cost: float = 5.0
    bbox_giou_cost: float = 1.0  # optional stabilizer using bbox(polyline)


class HungarianMatcherPolyline(nn.Module):
    def __init__(self, cost: MatcherCost):
        super().__init__()
        self.cost = cost

    @torch.no_grad()
    def forward(
        self, outputs: Dict[str, torch.Tensor], targets: List[Dict[str, torch.Tensor]]
    ):
        """
        outputs:
          pred_logits: (B,Q,C+1)
          pred_polylines: (B,Q,K,2) normalized
        targets: list length B:
          labels: (Ni,)
          polylines: (Ni,K,2) normalized
        returns: list of (idx_pred, idx_tgt)
        """
        bs, num_queries = outputs["pred_logits"].shape[:2]
        out_prob = outputs["pred_logits"].softmax(-1)  # (B,Q,C+1)
        out_poly = outputs["pred_polylines"]  # (B,Q,K,2)

        indices = []
        for b in range(bs):
            tgt_ids = targets[b]["labels"]  # (Ni,)
            tgt_poly = targets[b]["polylines"]  # (Ni,K,2)

            if tgt_poly.numel() == 0:
                indices.append(
                    (
                        torch.empty(0, dtype=torch.int64),
                        torch.empty(0, dtype=torch.int64),
                    )
                )
                continue

            # Class cost: negative prob of tgt class
            cost_class = -out_prob[b][:, tgt_ids]  # (Q,Ni)

            # Polyline cost: reverse-invariant mean L1 across points
            # Compute L1 distance to the target polyline in both directions (forward and reversed),
            # then take the minimum cost for each (query, target) pair.
            # Implementation: flatten to (Q, 2K) and (Ni, 2K), use cdist(L1) for forward and reversed,
            # then cost_poly = min(cost_fwd, cost_rev).
            # It can be replaced by either the Hausdorff or Chamfer distance.
            Q = out_poly[b].shape[0]
            K = out_poly[b].shape[1]
            out_flat = out_poly[b].reshape(Q, 2 * K)  # (Q,2K)
            tgt_flat = tgt_poly.reshape(tgt_poly.shape[0], 2 * K)  # (Ni,2K)
            tgt_rev = torch.flip(tgt_poly, dims=[1]).reshape(
                tgt_poly.shape[0], 2 * K
            )  # (Ni,2K)
            cost_fwd = torch.cdist(out_flat, tgt_flat, p=1) / float(2 * K)  # (Q,Ni)
            cost_rev = torch.cdist(out_flat, tgt_rev, p=1) / float(2 * K)  # (Q,Ni)
            cost_poly = torch.minimum(cost_fwd, cost_rev)

            # Optional bbox GIoU cost from polylines
            cost_giou = 0.0
            if self.cost.bbox_giou_cost != 0.0:
                out_bbox = polyline_to_bbox_xyxy(out_poly[b])  # (Q,4)
                tgt_bbox = polyline_to_bbox_xyxy(tgt_poly)  # (Ni,4)
                cost_giou = -generalized_box_iou_xyxy(out_bbox, tgt_bbox)  # (Q,Ni)

            C = (
                self.cost.class_cost * cost_class
                + self.cost.poly_cost * cost_poly
                + self.cost.bbox_giou_cost * cost_giou
            ).cpu()

            row_ind, col_ind = linear_sum_assignment(C)
            indices.append(
                (
                    torch.as_tensor(row_ind, dtype=torch.int64),
                    torch.as_tensor(col_ind, dtype=torch.int64),
                )
            )
        return indices


class DetrPolylineCriterion(nn.Module):
    def __init__(
        self,
        num_classes: int,
        matcher: HungarianMatcherPolyline,
        eos_coef: float = 0.1,
        loss_poly: float = 5.0,
        loss_bbox_giou: float = 1.0,
        loss_smooth: float = 0.0,  # optional regularizer
    ):
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.eos_coef = eos_coef

        self.loss_poly_w = loss_poly
        self.loss_bbox_giou_w = loss_bbox_giou
        self.loss_smooth_w = loss_smooth

        empty_weight = torch.ones(num_classes + 1)
        empty_weight[-1] = eos_coef
        self.register_buffer("empty_weight", empty_weight)

    def forward(
        self, outputs: Dict[str, torch.Tensor], targets: List[Dict[str, torch.Tensor]]
    ):
        indices = self.matcher(outputs, targets)

        loss_ce = self.loss_labels(outputs, targets, indices)
        loss_poly = self.loss_polylines(outputs, targets, indices)
        loss_giou = (
            self.loss_bbox_giou(outputs, targets, indices)
            if self.loss_bbox_giou_w != 0.0
            else torch.tensor(0.0, device=outputs["pred_logits"].device)
        )
        loss_smooth = (
            self.loss_smoothness(outputs, indices)
            if self.loss_smooth_w != 0.0
            else torch.tensor(0.0, device=outputs["pred_logits"].device)
        )

        total = (
            loss_ce
            + self.loss_poly_w * loss_poly
            + self.loss_bbox_giou_w * loss_giou
            + self.loss_smooth_w * loss_smooth
        )

        return {
            "loss_ce": loss_ce,
            "loss_poly": loss_poly,
            "loss_bbox_giou": loss_giou,
            "loss_smooth": loss_smooth,
            "loss_total": total,
        }

    def loss_labels(self, outputs, targets, indices):
        src_logits = outputs["pred_logits"]  # (B,Q,C+1)
        B, Q, _ = src_logits.shape

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

    def loss_polylines(self, outputs, targets, indices):
        pred_poly = outputs["pred_polylines"]  # (B,Q,K,2)
        loss = torch.tensor(0.0, device=pred_poly.device)
        n_matched = 0

        for b, (src_idx, tgt_idx) in enumerate(indices):
            if len(src_idx) == 0:
                continue

            s = pred_poly[b, src_idx]  # (M,K,2)
            t = targets[b]["polylines"][tgt_idx]  # (M,K,2)
            t_rev = torch.flip(t, dims=[1])  # (M,K,2)

            # Per-instance forward and reversed L1
            # (sum over K and xy, keep instance dimension)
            l1_fwd = F.l1_loss(s, t, reduction="none").sum(dim=(1, 2))  # (M,)
            l1_rev = F.l1_loss(s, t_rev, reduction="none").sum(dim=(1, 2))  # (M,)
            l1 = torch.minimum(l1_fwd, l1_rev).sum()  # scalar

            loss = loss + l1
            n_matched += s.shape[0]

        n_matched = max(n_matched, 1)
        # average per matched instance and per point coordinate
        K = pred_poly.shape[2]
        loss = loss / (n_matched * K * 2.0)
        return loss

    def loss_bbox_giou(self, outputs, targets, indices):
        pred_poly = outputs["pred_polylines"]
        loss = torch.tensor(0.0, device=pred_poly.device)
        n_matched = 0

        for b, (src_idx, tgt_idx) in enumerate(indices):
            if len(src_idx) == 0:
                continue
            s_poly = pred_poly[b, src_idx]  # (M,K,2)
            t_poly = targets[b]["polylines"][tgt_idx]  # (M,K,2)

            s_bbox = polyline_to_bbox_xyxy(s_poly)  # (M,4)
            t_bbox = polyline_to_bbox_xyxy(t_poly)  # (M,4)

            giou = generalized_box_iou_xyxy(s_bbox, t_bbox).diag()
            loss = loss + (1.0 - giou).sum()
            n_matched += s_bbox.shape[0]

        n_matched = max(n_matched, 1)
        loss = loss / n_matched
        return loss

    def loss_smoothness(self, outputs, indices):
        """
        Optional: encourages consecutive points to be close (prevents wild oscillations).
        This is a simple second-difference penalty.
        """
        pred_poly = outputs["pred_polylines"]  # (B,Q,K,2)
        loss = torch.tensor(0.0, device=pred_poly.device)
        n = 0

        for b, (src_idx, _) in enumerate(indices):
            if len(src_idx) == 0:
                continue
            p = pred_poly[b, src_idx]  # (M,K,2)
            if p.shape[1] < 3:
                continue
            d2 = p[:, 2:] - 2 * p[:, 1:-1] + p[:, :-2]  # (M,K-2,2)
            loss = loss + (d2**2).mean()
            n += 1

        if n == 0:
            return torch.tensor(0.0, device=pred_poly.device)
        return loss / n


# -------------------------
# Dataset and collate
# -------------------------


class DetrPolylineEmbDataset(Dataset):
    """
    Expected per file:
      feat: (196,1024)
      polylines: (num_obj,K,2) in xy pixel coords (not normalized)
      labels: (num_obj,)
      image_size: (H,W)
    """

    def __init__(self, embed_dir: Path, num_points: int = 20, normalize: bool = True):
        self.files = list(embed_dir.glob("*.npz"))
        self.num_points = num_points
        self.normalize = normalize

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx: int):
        d = np.load(self.files[idx])

        feat = torch.from_numpy(np.ascontiguousarray(d["feat"])).float()  # (196,1024)

        polylines = d["polylines"]  # (Ni,K,2) pixel coords
        labels = d["labels"]  # (Ni,)
        image_size = d["image_size"]  # (2,) np.array(H, W, dtype=np.int32)

        polylines = torch.from_numpy(np.ascontiguousarray(polylines)).float()
        labels = torch.from_numpy(np.ascontiguousarray(labels)).long()

        if polylines.numel() > 0:
            if polylines.shape[1] != self.num_points:
                raise ValueError(
                    f"{self.files[idx].name}: expected K={self.num_points}, got {polylines.shape[1]}"
                )

        if self.normalize:
            if "image_size" not in d:
                raise ValueError(f"{self.files[idx].name} missing image_size")
            H, W = d["image_size"].tolist()

            if polylines.numel() > 0:
                polylines[..., 0] = polylines[..., 0] / float(W)
                polylines[..., 1] = polylines[..., 1] / float(H)
                polylines = polylines.clamp(0, 1)

        target = {"labels": labels, "polylines": polylines, "image_size": image_size}
        return feat, target


def detr_polyline_collate_fn(batch):
    feats, targets = zip(*batch)
    feats = torch.stack(feats, dim=0)  # (B,196,1024)
    return feats, list(targets)


# -------------------------
# Training / eval / inference
# -------------------------


def tb_add_losses(writer, epoch: int, losses: dict, stage: str):
    d = {f"{stage}_{k}": float(v) for k, v in losses.items()}
    writer.add_scalars("losses", d, epoch)


def train_one_epoch(loader, model, criterion, optimizer, device):
    model.train()
    sums = {
        "loss_ce": 0.0,
        "loss_poly": 0.0,
        "loss_bbox_giou": 0.0,
        "loss_smooth": 0.0,
        "loss_total": 0.0,
    }
    n = 0

    for feats, targets in loader:
        feats = feats.to(device)
        for t in targets:
            t["labels"] = t["labels"].to(device)
            t["polylines"] = t["polylines"].to(device)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(feats)
        loss_dict = criterion(outputs, targets)
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
    sums = {
        "loss_ce": 0.0,
        "loss_poly": 0.0,
        "loss_bbox_giou": 0.0,
        "loss_smooth": 0.0,
        "loss_total": 0.0,
    }
    n = 0

    for feats, targets in loader:
        feats = feats.to(device)
        for t in targets:
            t["labels"] = t["labels"].to(device)
            t["polylines"] = t["polylines"].to(device)

        outputs = model(feats)
        loss_dict = criterion(outputs, targets)

        bs = feats.size(0)
        n += bs
        for k in sums:
            sums[k] += float(loss_dict[k].detach().item()) * bs

    for k in sums:
        sums[k] /= max(n, 1)
    return sums


@torch.no_grad()
def detr_polyline_inference(
    model,
    feats: torch.Tensor,
    image_sizes: List[List[int, int]],
    score_thresh: float = 0.5,
    topk: int = 100,
    device: torch.device | None = None,
):
    """
    Run inference for the polyline DETR model.

    Args:
        model: DetrPolylineFromEmbeddings
        feats: (B,196,1024) float tensor (DINO features)
        image_sizes: list/tuple length B, each (H,W) in pixels
        score_thresh: keep predictions with score >= thresh
        topk: keep at most topk predictions per image after filtering
        device: optional device; if given, feats are moved there

    Returns:
        list of length B. Each element is a dict:
          {
            "scores": (M,),
            "labels": (M,),
            "polylines_norm": (M,K,2) in [0,1],
            "polylines_px": (M,K,2) in pixel coords,
          }
        where M <= topk
    """
    model.eval()
    if device is not None:
        feats = feats.to(device)
        model = model.to(device)

    outputs = model(feats)
    logits = outputs["pred_logits"]  # (B,Q,C+1)
    polylines = outputs["pred_polylines"]  # (B,Q,K,2) normalized

    prob = F.softmax(logits, dim=-1)  # (B,Q,C+1)
    scores_all, labels_all = prob[..., :-1].max(dim=-1)  # exclude "no-object" -> (B,Q)

    B, Q = scores_all.shape
    results = []

    for b in range(B):
        H, W = image_sizes[b]

        scores = scores_all[b]
        labels = labels_all[b]
        polys = polylines[b]  # (Q,K,2)

        keep = scores >= score_thresh
        if keep.any():
            scores = scores[keep]
            labels = labels[keep]
            polys = polys[keep]
        else:
            # nothing passed threshold
            results.append(
                {
                    "scores": scores.new_zeros((0,)),
                    "labels": labels.new_zeros((0,), dtype=torch.long),
                    "polylines_norm": polys.new_zeros((0, polys.shape[1], 2)),
                    "polylines_px": polys.new_zeros((0, polys.shape[1], 2)),
                }
            )
            continue

        # topk
        if scores.numel() > topk:
            top_idx = torch.topk(scores, k=topk, largest=True).indices
            scores = scores[top_idx]
            labels = labels[top_idx]
            polys = polys[top_idx]

        # convert to pixel coordinates
        polys_px = polys.clone()
        polys_px[..., 0] = polys_px[..., 0] * float(W)
        polys_px[..., 1] = polys_px[..., 1] * float(H)

        results.append(
            {
                "scores": scores.detach().cpu(),
                "labels": labels.detach().cpu(),
                "polylines_norm": polys.detach().cpu(),
                "polylines_px": polys_px.detach().cpu(),
            }
        )

    return results


# -------------------------
# Main
# -------------------------


def main():
    torch.manual_seed(0)
    cfg = dict(
        exp="detr_polyline_from_dino_test",
        save_path=Path("/home/fatemeh/Downloads/hedg/results/training"),
        embed_dir=Path(
            "/home/fatemeh/Downloads/hedg/results/test_dataset_with_osm/embs_polylines"
        ),
        n_epochs=5,
        num_points=20,
        num_classes=1,
    )
    cfg = OmegaConf.create(cfg)

    dataset = DetrPolylineEmbDataset(
        embed_dir=cfg.embed_dir, num_points=cfg.num_points, normalize=True
    )
    n_train = int(0.8 * len(dataset))
    n_val = len(dataset) - n_train
    train_ds, val_ds = torch.utils.data.random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(
        train_ds,
        batch_size=8,
        shuffle=True,
        num_workers=2,
        collate_fn=detr_polyline_collate_fn,
    )
    eval_loader = DataLoader(
        val_ds,
        batch_size=8,
        shuffle=False,
        num_workers=2,
        collate_fn=detr_polyline_collate_fn,
    )

    model = DetrPolylineFromEmbeddings(
        in_dim=1024,
        num_classes=cfg.num_classes,
        num_queries=100,
        d_model=256,
        nhead=8,
        num_encoder_layers=4,
        num_decoder_layers=4,
        dim_feedforward=1024,
        dropout=0.1,
        grid_size=(14, 14),
        num_points=cfg.num_points,
    )

    matcher = HungarianMatcherPolyline(
        MatcherCost(class_cost=1.0, poly_cost=5.0, bbox_giou_cost=1.0)
    )
    criterion = DetrPolylineCriterion(
        num_classes=cfg.num_classes,
        matcher=matcher,
        eos_coef=0.1,
        loss_poly=5.0,
        loss_bbox_giou=1.0,
        loss_smooth=0.0,  # set small value like 0.1 if you want smoother curves
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

    best_val = 1e9
    tb_dir = cfg.save_path / f"tensorboard/{cfg.exp}"
    tb_dir.mkdir(parents=True, exist_ok=True)

    with tensorboard.SummaryWriter(tb_dir) as writer:
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
                f"(ce={train_losses['loss_ce']:.4f}, poly={train_losses['loss_poly']:.4f}, "
                f"bbox_giou={train_losses['loss_bbox_giou']:.4f}, smooth={train_losses['loss_smooth']:.4f}) "
                f"eval_total={eval_losses['loss_total']:.4f} "
                f"(ce={eval_losses['loss_ce']:.4f}, poly={eval_losses['loss_poly']:.4f}, "
                f"bbox_giou={eval_losses['loss_bbox_giou']:.4f}, smooth={eval_losses['loss_smooth']:.4f})"
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
                    cfg.save_path / "best_detr_polyline_from_dino.pt",
                )
                print(f"Saved best: {best_val:.4f} at epoch {epoch}")

    # Example inference on eval set
    feats, targets = next(iter(eval_loader))
    image_sizes = [t["image_size"].tolist() for t in targets]
    preds = detr_polyline_inference(
        model=model,
        feats=feats,
        image_sizes=image_sizes,
        score_thresh=0.5,
        topk=20,
        device=device,
    )

    # preds[0]["polylines_px"] is (M,K,2) in pixel coords
    print(preds[0]["scores"].shape, preds[0]["polylines_px"].shape)


if __name__ == "__main__":
    main()
