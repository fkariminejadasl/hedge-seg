import copy
import math
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

from hedge_seg.training_utils import set_seed

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


# =========================================================
# DETR-style transformer
# =========================================================


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def _with_pos_embed(tensor: torch.Tensor, pos: Optional[torch.Tensor]):
    return tensor if pos is None else tensor + pos


class DETRTransformer(nn.Module):
    def __init__(
        self,
        d_model=256,
        nhead=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
        normalize_before=False,
        return_intermediate_dec=True,
    ):
        super().__init__()

        encoder_layer = DETRTransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout, activation, normalize_before
        )
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = DETRTransformerEncoder(
            encoder_layer, num_encoder_layers, encoder_norm
        )

        decoder_layer = DETRTransformerDecoderLayer(
            d_model, nhead, dim_feedforward, dropout, activation, normalize_before
        )
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = DETRTransformerDecoder(
            decoder_layer,
            num_decoder_layers,
            decoder_norm,
            return_intermediate=return_intermediate_dec,
        )

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self,
        src: torch.Tensor,  # (B, L, D)
        query_embed: torch.Tensor,  # (Q, D)
        pos_embed: Optional[torch.Tensor] = None,  # (B, L, D)
        mask: Optional[torch.Tensor] = None,  # (B, L)
    ):
        B, _, _ = src.shape
        query_embed = query_embed.unsqueeze(0).expand(B, -1, -1)  # (B,Q,D)
        tgt = torch.zeros_like(query_embed)

        memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)
        hs = self.decoder(
            tgt,
            memory,
            memory_key_padding_mask=mask,
            pos=pos_embed,
            query_pos=query_embed,
        )
        # hs: (num_layers, B, Q, D)
        return hs, memory


class DETRTransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.norm = norm

    def forward(
        self,
        src,
        mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        pos: Optional[torch.Tensor] = None,
    ):
        output = src
        for layer in self.layers:
            output = layer(
                output,
                src_mask=mask,
                src_key_padding_mask=src_key_padding_mask,
                pos=pos,
            )
        if self.norm is not None:
            output = self.norm(output)
        return output


class DETRTransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(
        self,
        tgt,
        memory,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
        pos: Optional[torch.Tensor] = None,
        query_pos: Optional[torch.Tensor] = None,
    ):
        output = tgt
        intermediate = []

        for layer in self.layers:
            output = layer(
                output,
                memory,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
                pos=pos,
                query_pos=query_pos,
            )
            if self.return_intermediate:
                intermediate.append(
                    self.norm(output) if self.norm is not None else output
                )

        if self.norm is not None:
            output = self.norm(output)

        if self.return_intermediate:
            if self.norm is not None:
                intermediate[-1] = output
            return torch.stack(intermediate)

        return output.unsqueeze(0)


class DETRTransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
        normalize_before=False,
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = F.relu if activation == "relu" else F.gelu
        self.normalize_before = normalize_before

    def forward_post(
        self,
        src,
        src_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        pos: Optional[torch.Tensor] = None,
    ):
        q = k = _with_pos_embed(src, pos)
        src2 = self.self_attn(
            q,
            k,
            value=src,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask,
            need_weights=False,
        )[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward_pre(
        self,
        src,
        src_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        pos: Optional[torch.Tensor] = None,
    ):
        src2 = self.norm1(src)
        q = k = _with_pos_embed(src2, pos)
        src2 = self.self_attn(
            q,
            k,
            value=src2,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask,
            need_weights=False,
        )[0]
        src = src + self.dropout1(src2)

        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(
        self,
        src,
        src_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        pos: Optional[torch.Tensor] = None,
    ):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)


class DETRTransformerDecoderLayer(nn.Module):
    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
        normalize_before=False,
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.multihead_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = F.relu if activation == "relu" else F.gelu
        self.normalize_before = normalize_before

    def forward_post(
        self,
        tgt,
        memory,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
        pos: Optional[torch.Tensor] = None,
        query_pos: Optional[torch.Tensor] = None,
    ):
        q = k = _with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(
            q,
            k,
            value=tgt,
            attn_mask=tgt_mask,
            key_padding_mask=tgt_key_padding_mask,
            need_weights=False,
        )[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        tgt2 = self.multihead_attn(
            query=_with_pos_embed(tgt, query_pos),
            key=_with_pos_embed(memory, pos),
            value=memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
            need_weights=False,
        )[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward_pre(
        self,
        tgt,
        memory,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
        pos: Optional[torch.Tensor] = None,
        query_pos: Optional[torch.Tensor] = None,
    ):
        tgt2 = self.norm1(tgt)
        q = k = _with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(
            q,
            k,
            value=tgt2,
            attn_mask=tgt_mask,
            key_padding_mask=tgt_key_padding_mask,
            need_weights=False,
        )[0]
        tgt = tgt + self.dropout1(tgt2)

        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(
            query=_with_pos_embed(tgt2, query_pos),
            key=_with_pos_embed(memory, pos),
            value=memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
            need_weights=False,
        )[0]
        tgt = tgt + self.dropout2(tgt2)

        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(
        self,
        tgt,
        memory,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
        pos: Optional[torch.Tensor] = None,
        query_pos: Optional[torch.Tensor] = None,
    ):
        if self.normalize_before:
            return self.forward_pre(
                tgt,
                memory,
                tgt_mask,
                memory_mask,
                tgt_key_padding_mask,
                memory_key_padding_mask,
                pos,
                query_pos,
            )
        return self.forward_post(
            tgt,
            memory,
            tgt_mask,
            memory_mask,
            tgt_key_padding_mask,
            memory_key_padding_mask,
            pos,
            query_pos,
        )


# =========================================================
# Legacy mode transformer wrapper
# =========================================================


class LegacyTransformerWrapper(nn.Module):
    """
    Your original style:
      - add pos before encoder
      - add query to tgt before decoder
      - optionally collect intermediate decoder layers
    """

    def __init__(
        self,
        d_model=256,
        nhead=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        dim_feedforward=2048,
        dropout=0.1,
        return_intermediate_dec=True,
    ):
        super().__init__()

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_encoder_layers)

        dec_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=num_decoder_layers)

        self.return_intermediate_dec = return_intermediate_dec

    def forward(
        self,
        src: torch.Tensor,  # (B,L,D)
        query_embed: torch.Tensor,  # (Q,D)
        pos_embed: Optional[torch.Tensor] = None,  # (B,L,D)
        mask: Optional[torch.Tensor] = None,  # (B,L)
    ):
        B, _, _ = src.shape

        if pos_embed is not None:
            src = src + pos_embed

        memory = self.encoder(src, src_key_padding_mask=mask)

        query = query_embed.unsqueeze(0).expand(B, -1, -1)  # (B,Q,D)
        output = torch.zeros_like(query) + query

        if self.return_intermediate_dec:
            intermediate = []
            for layer in self.decoder.layers:
                output = layer(
                    output,
                    memory,
                    tgt_key_padding_mask=None,
                    memory_key_padding_mask=mask,
                )
                intermediate.append(
                    self.decoder.norm(output)
                    if self.decoder.norm is not None
                    else output
                )
            hs = torch.stack(intermediate)  # (num_layers,B,Q,D)
        else:
            output = self.decoder(
                output,
                memory,
                tgt_key_padding_mask=None,
                memory_key_padding_mask=mask,
            )
            hs = output.unsqueeze(0)

        return hs, memory


# =========================================================
# Polyline model with switchable query/pos style
# =========================================================


class DetrPolylineFromEmbeddings(nn.Module):
    """
    Input: x (B, L, in_dim), with L = grid_h * grid_w
    Output:
      pred_logits: (B, Q, num_classes+1)
      pred_polylines: (B, Q, K, 2) in [0,1]
      aux_outputs: list[dict], optional
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
        grid_size: Tuple[int, int] = (16, 16),
        num_points: int = 20,
        aux_loss: bool = True,
        normalize_before: bool = False,
        query_embed_mode: str = "detr",  # "detr" or "legacy"
    ):
        super().__init__()
        assert query_embed_mode in {"detr", "legacy"}

        self.num_classes = num_classes
        self.num_queries = num_queries
        self.grid_h, self.grid_w = grid_size
        self.num_points = num_points
        self.aux_loss = aux_loss
        self.query_embed_mode = query_embed_mode

        self.input_proj = nn.Linear(in_dim, d_model)
        self.pos_embed = PositionEmbeddingSine2D(num_pos_feats=d_model // 2)
        self.query_embed = nn.Embedding(num_queries, d_model)

        if query_embed_mode == "detr":
            self.transformer = DETRTransformer(
                d_model=d_model,
                nhead=nhead,
                num_encoder_layers=num_encoder_layers,
                num_decoder_layers=num_decoder_layers,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                normalize_before=normalize_before,
                return_intermediate_dec=aux_loss,
            )
        else:
            self.transformer = LegacyTransformerWrapper(
                d_model=d_model,
                nhead=nhead,
                num_encoder_layers=num_encoder_layers,
                num_decoder_layers=num_decoder_layers,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                return_intermediate_dec=aux_loss,
            )

        self.class_embed = nn.Linear(d_model, num_classes + 1)
        self.poly_embed = MLP(d_model, d_model, 2 * num_points, num_layers=3)

    def _decode_polylines(self, hs: torch.Tensor) -> torch.Tensor:
        # hs: (..., Q, D)
        poly = self.poly_embed(hs)
        poly = poly.view(*hs.shape[:-1], self.num_points, 2).sigmoid()
        return poly

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_poly):
        return [
            {"pred_logits": a, "pred_polylines": b}
            for a, b in zip(outputs_class[:-1], outputs_poly[:-1])
        ]

    def forward(
        self, x: torch.Tensor, return_features: bool = False
    ) -> Dict[str, torch.Tensor]:
        B, L, _ = x.shape
        assert (
            L == self.grid_h * self.grid_w
        ), f"Expected {self.grid_h * self.grid_w} tokens, got {L}"

        dino_tokens = x
        src = self.input_proj(x)  # (B,L,D)
        pos = self.pos_embed(
            B=B, H=self.grid_h, W=self.grid_w, device=x.device
        )  # (B,L,D)

        mask = None
        hs, memory = self.transformer(
            src=src,
            query_embed=self.query_embed.weight,
            pos_embed=pos,
            mask=mask,
        )  # hs: (num_layers,B,Q,D)

        outputs_class = self.class_embed(hs)  # (num_layers,B,Q,C+1)
        outputs_poly = self._decode_polylines(hs)  # (num_layers,B,Q,K,2)

        out = {
            "pred_logits": outputs_class[-1],
            "pred_polylines": outputs_poly[-1],
        }

        if return_features:
            out["memory"] = memory
            out["hs_last"] = hs[-1]
            out["dino_tokens"] = dino_tokens

        if self.aux_loss:
            out["aux_outputs"] = self._set_aux_loss(outputs_class, outputs_poly)

        return out


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
        loss_smooth: float = 0.0,
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

    def _compute_losses(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: List[Dict[str, torch.Tensor]],
    ) -> Dict[str, torch.Tensor]:
        indices = self.matcher(outputs, targets)

        loss_ce = self.loss_labels(outputs, targets, indices)
        loss_poly = self.loss_polylines(outputs, targets, indices)

        if self.loss_bbox_giou_w != 0.0:
            loss_giou = self.loss_bbox_giou(outputs, targets, indices)
        else:
            loss_giou = outputs["pred_logits"].new_tensor(0.0)

        if self.loss_smooth_w != 0.0:
            loss_smooth = self.loss_smoothness(outputs, indices)
        else:
            loss_smooth = outputs["pred_logits"].new_tensor(0.0)

        loss_total = (
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
            "loss_total": loss_total,
        }

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: List[Dict[str, torch.Tensor]],
    ) -> Dict[str, torch.Tensor]:
        # final decoder layer only
        outputs_without_aux = {k: v for k, v in outputs.items() if k != "aux_outputs"}

        losses = self._compute_losses(outputs_without_aux, targets)

        # auxiliary decoder layers
        if "aux_outputs" in outputs:
            aux_total = outputs_without_aux["pred_logits"].new_tensor(0.0)

            for i, aux_outputs in enumerate(outputs["aux_outputs"]):
                l_dict = self._compute_losses(aux_outputs, targets)

                for k, v in l_dict.items():
                    losses[f"{k}_{i}"] = v

                aux_total = aux_total + l_dict["loss_total"]

            losses["loss_total"] = losses["loss_total"] + aux_total

        return losses

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
                polylines[..., 0] = polylines[..., 0] / float(W - 1)
                polylines[..., 1] = polylines[..., 1] / float(H - 1)
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
    wanted = {"loss_total", "loss_diff"}
    d = {f"{stage}_{k}": float(v) for k, v in losses.items() if k in wanted}
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
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
        # a = {n:round(p.grad.max().item(),2) for n, p in model.named_parameters()}
        # {k:v for k, v in a.items() if v == max(a.values())}
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
    image_sizes: List[Tuple[int, int]],
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
        polys_px[..., 0] = polys_px[..., 0] * float(W - 1)
        polys_px[..., 1] = polys_px[..., 1] * float(H - 1)
        polys_px[..., 0] = polys_px[..., 0].clamp(0, W - 1)
        polys_px[..., 1] = polys_px[..., 1].clamp(0, H - 1)

        results.append(
            {
                "scores": scores.detach().cpu(),
                "labels": labels.detach().cpu(),
                "polylines_norm": polys.detach().cpu(),
                "polylines_px": polys_px.detach().cpu(),
            }
        )

    return results


# =========================================================
# Diffusion Model Components
# =========================================================

# =========================================================
# Helpers
# =========================================================


def coords_01_to_m11(x: torch.Tensor) -> torch.Tensor:
    return x * 2.0 - 1.0


def coords_m11_to_01(x: torch.Tensor) -> torch.Tensor:
    return ((x + 1.0) * 0.5).clamp(0.0, 1.0)


def extract(a: torch.Tensor, t: torch.Tensor, x_shape: Tuple[int, ...]) -> torch.Tensor:
    b = t.shape[0]
    out = a.gather(0, t)
    return out.view(b, *([1] * (len(x_shape) - 1)))


# =========================================================
# Timestep embedding
# =========================================================


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        device = t.device
        emb = math.log(10000.0) / max(half - 1, 1)
        emb = torch.exp(torch.arange(half, device=device, dtype=torch.float32) * -emb)
        emb = t.float()[:, None] * emb[None, :]
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


# =========================================================
# Transformer blocks for diffusion decoder
# =========================================================


class AdaLNModulation(nn.Module):
    def __init__(self, hidden_dim: int, cond_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, hidden_dim * 2),
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        scale, shift = self.net(cond).chunk(2, dim=-1)
        return x * (1.0 + scale[:, None, :]) + shift[:, None, :]


class DiffusionDecoderLayer(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        nhead: int,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        use_cross_attn: bool = True,
    ):
        super().__init__()
        self.use_cross_attn = use_cross_attn

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.self_attn = nn.MultiheadAttention(
            hidden_dim, nhead, dropout=dropout, batch_first=True
        )
        self.drop1 = nn.Dropout(dropout)

        if use_cross_attn:
            self.norm2 = nn.LayerNorm(hidden_dim)
            self.cross_attn = nn.MultiheadAttention(
                hidden_dim, nhead, dropout=dropout, batch_first=True
            )
            self.drop2 = nn.Dropout(dropout)
        else:
            self.norm2 = None
            self.cross_attn = None
            self.drop2 = None

        self.norm3 = nn.LayerNorm(hidden_dim)
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, hidden_dim),
        )
        self.drop3 = nn.Dropout(dropout)

        self.ada1 = AdaLNModulation(hidden_dim, hidden_dim)
        self.ada2 = AdaLNModulation(hidden_dim, hidden_dim) if use_cross_attn else None
        self.ada3 = AdaLNModulation(hidden_dim, hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
        time_cond: torch.Tensor,
        memory: Optional[torch.Tensor] = None,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        h = self.ada1(self.norm1(x), time_cond)
        h = self.self_attn(h, h, h, need_weights=False)[0]
        x = x + self.drop1(h)

        if self.use_cross_attn and memory is not None:
            h = self.ada2(self.norm2(x), time_cond)
            h = self.cross_attn(
                query=h,
                key=memory,
                value=memory,
                key_padding_mask=memory_key_padding_mask,
                need_weights=False,
            )[0]
            x = x + self.drop2(h)

        h = self.ada3(self.norm3(x), time_cond)
        h = self.ff(h)
        x = x + self.drop3(h)
        return x


# =========================================================
# Polyline diffusion decoder
# =========================================================


class PolylineDiffusionDecoder(nn.Module):
    """
    Slot-wise coordinate denoiser.

    Inputs:
      x_t:            (B, Q, K, 2) coordinates in [-1, 1]
      t:              (B,)
      cond_tokens:    optional (B, L, C_cond) from raw DINO tokens or encoder memory
      query_context:  optional (B, Q, C_query) from decoder queries
      coarse_polys:   optional (B, Q, K, 2) coarse prediction in [-1, 1]

    Output:
      pred_noise:     (B, Q, K, 2)
    """

    def __init__(
        self,
        num_queries: int,
        num_points: int,
        hidden_dim: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        cond_dim: int = 256,
        query_dim: int = 256,
        use_cond_tokens: bool = True,
        use_query_context: bool = True,
        use_coarse_polylines: bool = True,
    ):
        super().__init__()
        self.num_queries = num_queries
        self.num_points = num_points
        self.hidden_dim = hidden_dim
        self.use_cond_tokens = use_cond_tokens
        self.use_query_context = use_query_context
        self.use_coarse_polylines = use_coarse_polylines

        in_dim = num_points * 2
        if use_coarse_polylines:
            in_dim += num_points * 2

        self.input_proj = nn.Linear(in_dim, hidden_dim)
        self.slot_embed = nn.Embedding(num_queries, hidden_dim)
        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.cond_proj = nn.Linear(cond_dim, hidden_dim) if use_cond_tokens else None
        self.query_proj = (
            nn.Linear(query_dim, hidden_dim) if use_query_context else None
        )

        self.layers = nn.ModuleList(
            [
                DiffusionDecoderLayer(
                    hidden_dim=hidden_dim,
                    nhead=nhead,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                    use_cross_attn=use_cond_tokens,
                )
                for _ in range(num_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, num_points * 2)

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        cond_tokens: Optional[torch.Tensor] = None,
        query_context: Optional[torch.Tensor] = None,
        coarse_polys: Optional[torch.Tensor] = None,
        cond_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, Q, K, _ = x_t.shape
        assert Q == self.num_queries
        assert K == self.num_points

        x_flat = x_t.reshape(B, Q, K * 2)
        pieces = [x_flat]

        if self.use_coarse_polylines:
            if coarse_polys is None:
                coarse_polys = torch.zeros_like(x_t)
            pieces.append(coarse_polys.reshape(B, Q, K * 2))

        x = self.input_proj(torch.cat(pieces, dim=-1))

        slot_ids = torch.arange(Q, device=x.device)
        x = x + self.slot_embed(slot_ids)[None, :, :]

        if self.use_query_context:
            if query_context is None:
                query_context = torch.zeros(
                    B, Q, self.hidden_dim, device=x.device, dtype=x.dtype
                )
            else:
                query_context = self.query_proj(query_context)
            x = x + query_context

        time_cond = self.time_embed(t)

        memory = None
        if self.use_cond_tokens and cond_tokens is not None:
            memory = self.cond_proj(cond_tokens)

        for layer in self.layers:
            x = layer(
                x, time_cond=time_cond, memory=memory, memory_key_padding_mask=cond_mask
            )

        x = self.final_norm(x)
        out = self.out_proj(x).view(B, Q, K, 2)
        return out


# =========================================================
# Diffusion scheduler and loss
# =========================================================


class GaussianPolylineDiffusion(nn.Module):
    def __init__(
        self,
        model: PolylineDiffusionDecoder,
        timesteps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 2e-2,
        objective: str = "eps",
    ):
        super().__init__()
        assert objective in {"eps"}
        self.model = model
        self.timesteps = timesteps
        self.objective = objective

        betas = torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float32)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]], dim=0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod)
        )
        self.register_buffer("sqrt_recip_alphas", torch.sqrt(1.0 / alphas))

        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        self.register_buffer("posterior_variance", posterior_variance.clamp(min=1e-20))

    def q_sample(
        self,
        x_start: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if noise is None:
            noise = torch.randn_like(x_start)
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def p_losses(
        self,
        x_start: torch.Tensor,
        t: torch.Tensor,
        cond_tokens: Optional[torch.Tensor] = None,
        query_context: Optional[torch.Tensor] = None,
        coarse_polys: Optional[torch.Tensor] = None,
        valid_mask: Optional[torch.Tensor] = None,
        cond_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        noise = torch.randn_like(x_start)
        x_t = self.q_sample(x_start=x_start, t=t, noise=noise)
        pred = self.model(
            x_t=x_t,
            t=t,
            cond_tokens=cond_tokens,
            query_context=query_context,
            coarse_polys=coarse_polys,
            cond_mask=cond_mask,
        )

        per_elem = (pred - noise) ** 2
        if valid_mask is not None:
            valid_mask = valid_mask[:, :, None, None].float()
            per_elem = per_elem * valid_mask
            denom = (valid_mask.sum() * x_start.shape[2] * x_start.shape[3]).clamp(
                min=1.0
            )
            loss = per_elem.sum() / denom
        else:
            loss = per_elem.mean()

        return loss, {"loss_diff": loss.detach()}

    @torch.no_grad()
    def p_sample(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        t_index: int,
        cond_tokens: Optional[torch.Tensor] = None,
        query_context: Optional[torch.Tensor] = None,
        coarse_polys: Optional[torch.Tensor] = None,
        cond_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        pred_noise = self.model(
            x_t=x,
            t=t,
            cond_tokens=cond_tokens,
            query_context=query_context,
            coarse_polys=coarse_polys,
            cond_mask=cond_mask,
        )

        betas_t = extract(self.betas, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(
            self.sqrt_one_minus_alphas_cumprod, t, x.shape
        )
        sqrt_recip_alphas_t = extract(self.sqrt_recip_alphas, t, x.shape)

        model_mean = sqrt_recip_alphas_t * (
            x - betas_t * pred_noise / sqrt_one_minus_alphas_cumprod_t
        )

        if t_index == 0:
            return model_mean

        posterior_variance_t = extract(self.posterior_variance, t, x.shape)
        noise = torch.randn_like(x)
        return model_mean + posterior_variance_t.sqrt() * noise

    @torch.no_grad()
    def sample(
        self,
        shape: Tuple[int, int, int, int],
        cond_tokens: Optional[torch.Tensor] = None,
        query_context: Optional[torch.Tensor] = None,
        coarse_polys: Optional[torch.Tensor] = None,
        cond_mask: Optional[torch.Tensor] = None,
        clamp: bool = True,
    ) -> torch.Tensor:
        device = next(self.parameters()).device
        x = torch.randn(shape, device=device)
        for i in reversed(range(self.timesteps)):
            t = torch.full((shape[0],), i, device=device, dtype=torch.long)
            x = self.p_sample(
                x=x,
                t=t,
                t_index=i,
                cond_tokens=cond_tokens,
                query_context=query_context,
                coarse_polys=coarse_polys,
                cond_mask=cond_mask,
            )
        return x.clamp(-1.0, 1.0) if clamp else x


# =========================================================
# Integration wrapper for an existing DETR-like polyline model
# =========================================================


@dataclass
class DiffusionTrainConfig:
    lambda_diff: float = 0.1
    timesteps: int = 1000
    train_condition_source: str = "encoder"  # "encoder" or "dino"
    use_query_context: bool = True
    use_coarse_polylines: bool = True


class DetrWithDiffusion(nn.Module):
    """
    Wraps an existing DETR-like polyline model and adds a diffusion branch.

    Expected base model API:
      outputs = base_model(feats, return_features=True)

    outputs should include at least:
      pred_logits:        (B,Q,C+1)
      pred_polylines:     (B,Q,K,2) in [0,1]

    and, if return_features=True:
      memory:             (B,L,D) optional
      hs_last:            (B,Q,D) optional
      dino_tokens:        (B,L,C) optional, or reuse feats directly

    If your current model does not yet return memory / hs_last, add them there.
    """

    def __init__(
        self,
        base_model: nn.Module,
        diffusion: GaussianPolylineDiffusion,
        num_queries: int,
        num_points: int,
        condition_source: str = "encoder",
        use_query_context: bool = True,
        use_coarse_polylines: bool = True,
    ):
        super().__init__()
        assert condition_source in {"encoder", "dino"}
        self.base_model = base_model
        self.diffusion = diffusion
        self.num_queries = num_queries
        self.num_points = num_points
        self.condition_source = condition_source
        self.use_query_context = use_query_context
        self.use_coarse_polylines = use_coarse_polylines

    def _pick_cond_tokens(
        self, feats: torch.Tensor, outputs: Dict[str, torch.Tensor]
    ) -> Optional[torch.Tensor]:
        if self.condition_source == "encoder":
            return outputs.get("memory", None)
        if self.condition_source == "dino":
            return outputs.get("dino_tokens", feats)
        return None

    def _pick_query_context(
        self, outputs: Dict[str, torch.Tensor]
    ) -> Optional[torch.Tensor]:
        if not self.use_query_context:
            return None
        return outputs.get("hs_last", None)

    def forward(
        self, feats: torch.Tensor, return_features: bool = True
    ) -> Dict[str, torch.Tensor]:
        outputs = self.base_model(feats, return_features=return_features)
        if "dino_tokens" not in outputs:
            outputs["dino_tokens"] = feats
        return outputs


# =========================================================
# Matching bridge from DETR outputs to diffusion training targets
# =========================================================


def build_slot_aligned_diffusion_targets(
    outputs: Dict[str, torch.Tensor],
    targets: List[Dict[str, torch.Tensor]],
    indices: List[Tuple[torch.Tensor, torch.Tensor]],
    detach_coarse: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns:
      x0_gt_m11:    (B,Q,K,2) ground truth coords in [-1,1]
      coarse_m11:   (B,Q,K,2) current model coords in [-1,1]
      valid_mask:   (B,Q) True for matched slots
    """
    pred = outputs["pred_polylines"]
    B, Q, K, _ = pred.shape
    device = pred.device

    x0_gt = torch.zeros_like(pred)
    valid = torch.zeros(B, Q, dtype=torch.bool, device=device)

    for b, (src_idx, tgt_idx) in enumerate(indices):
        if len(src_idx) == 0:
            continue
        tgt_poly = targets[b]["polylines"].to(device)
        x0_gt[b, src_idx] = tgt_poly[tgt_idx]
        valid[b, src_idx] = True

    coarse = pred.detach() if detach_coarse else pred
    return coords_01_to_m11(x0_gt), coords_01_to_m11(coarse), valid


# =========================================================
# Loss utilities for stage 2 and stage 3
# =========================================================


@torch.no_grad()
def set_requires_grad(module: nn.Module, requires_grad: bool):
    for p in module.parameters():
        p.requires_grad = requires_grad


def compute_diffusion_loss_from_matches(
    model_with_diffusion: DetrWithDiffusion,
    feats: torch.Tensor,
    targets: List[Dict[str, torch.Tensor]],
    matcher: nn.Module,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    outputs = model_with_diffusion(feats, return_features=True)
    indices = matcher(outputs, targets)

    x0_gt_m11, coarse_m11, valid_mask = build_slot_aligned_diffusion_targets(
        outputs, targets, indices
    )

    B = feats.shape[0]
    device = feats.device
    t = torch.randint(
        0, model_with_diffusion.diffusion.timesteps, (B,), device=device
    ).long()

    cond_tokens = model_with_diffusion._pick_cond_tokens(feats, outputs)
    query_context = model_with_diffusion._pick_query_context(outputs)
    coarse_polys = coarse_m11 if model_with_diffusion.use_coarse_polylines else None

    diff_loss, diff_log = model_with_diffusion.diffusion.p_losses(
        x_start=x0_gt_m11,
        t=t,
        cond_tokens=cond_tokens,
        query_context=query_context,
        coarse_polys=coarse_polys,
        valid_mask=valid_mask,
        cond_mask=None,
    )

    extra = {
        "indices": indices,
        "outputs": outputs,
        "valid_mask": valid_mask,
        "x0_gt_m11": x0_gt_m11,
        "coarse_m11": coarse_m11,
    }
    return {"loss_diff": diff_loss}, extra


# =========================================================
# Inference utilities
# =========================================================


@torch.no_grad()
def refine_polylines_with_diffusion(
    model_with_diffusion: DetrWithDiffusion,
    feats: torch.Tensor,
    pred_score_thresh: float = 0.5,
    refine_only_positive: bool = True,
) -> Dict[str, torch.Tensor]:
    outputs = model_with_diffusion(feats, return_features=True)

    logits = outputs["pred_logits"]
    probs = logits.softmax(dim=-1)
    fg_scores = probs[..., :-1].max(dim=-1).values
    keep = fg_scores >= pred_score_thresh

    coarse_m11 = coords_01_to_m11(outputs["pred_polylines"])
    cond_tokens = model_with_diffusion._pick_cond_tokens(feats, outputs)
    query_context = model_with_diffusion._pick_query_context(outputs)

    refined_m11 = model_with_diffusion.diffusion.sample(
        shape=coarse_m11.shape,
        cond_tokens=cond_tokens,
        query_context=query_context,
        coarse_polys=coarse_m11 if model_with_diffusion.use_coarse_polylines else None,
        cond_mask=None,
        clamp=True,
    )

    if refine_only_positive:
        refined_m11 = torch.where(keep[:, :, None, None], refined_m11, coarse_m11)

    outputs["pred_polylines_refined"] = coords_m11_to_01(refined_m11)
    outputs["pred_keep_mask"] = keep
    outputs["pred_scores_fg"] = fg_scores
    return outputs


# =========================================================
# Stage-specific training helpers
# =========================================================


def train_stage2_diffusion_only(
    train_loader,
    model_with_diffusion: DetrWithDiffusion,
    matcher: nn.Module,
    optimizer,
    device: torch.device,
):
    model_with_diffusion.train()
    losses_sum = {"loss_diff": 0.0}
    n_batches = 0

    for feats, targets in train_loader:
        feats = feats.to(device)
        targets = move_targets_to_device(targets, device)

        optimizer.zero_grad(set_to_none=True)

        loss_dict, _ = compute_diffusion_loss_from_matches(
            model_with_diffusion=model_with_diffusion,
            feats=feats,
            targets=targets,
            matcher=matcher,
        )
        loss = loss_dict["loss_diff"]
        loss.backward()
        optimizer.step()

        losses_sum["loss_diff"] += float(loss.item())
        n_batches += 1

    return {k: v / max(n_batches, 1) for k, v in losses_sum.items()}


@torch.no_grad()
def eval_stage2_diffusion_only(
    eval_loader,
    model_with_diffusion: DetrWithDiffusion,
    matcher: nn.Module,
    device: torch.device,
):
    model_with_diffusion.eval()
    losses_sum = {"loss_diff": 0.0}
    n_batches = 0

    for feats, targets in eval_loader:
        feats = feats.to(device)
        targets = move_targets_to_device(targets, device)

        loss_dict, _ = compute_diffusion_loss_from_matches(
            model_with_diffusion=model_with_diffusion,
            feats=feats,
            targets=targets,
            matcher=matcher,
        )

        losses_sum["loss_diff"] += float(loss_dict["loss_diff"].item())
        n_batches += 1

    return {k: v / max(n_batches, 1) for k, v in losses_sum.items()}


def train_stage3_joint(
    train_loader,
    model_with_diffusion: DetrWithDiffusion,
    criterion: nn.Module,
    matcher: nn.Module,
    optimizer,
    device: torch.device,
    lambda_diff: float = 0.1,
):
    model_with_diffusion.train()
    losses_sum = {
        "loss_ce": 0.0,
        "loss_poly": 0.0,
        "loss_bbox_giou": 0.0,
        "loss_smooth": 0.0,
        "loss_total_detr": 0.0,
        "loss_diff": 0.0,
        "loss_total": 0.0,
    }
    n_batches = 0

    for feats, targets in train_loader:
        feats = feats.to(device)
        targets = move_targets_to_device(targets, device)

        optimizer.zero_grad(set_to_none=True)

        outputs = model_with_diffusion(feats, return_features=True)
        detr_losses = criterion(outputs, targets)
        indices = matcher(outputs, targets)
        x0_gt_m11, coarse_m11, valid_mask = build_slot_aligned_diffusion_targets(
            outputs, targets, indices, detach_coarse=True
        )

        B = feats.shape[0]
        t = torch.randint(
            0, model_with_diffusion.diffusion.timesteps, (B,), device=device
        ).long()

        cond_tokens = model_with_diffusion._pick_cond_tokens(feats, outputs)
        query_context = model_with_diffusion._pick_query_context(outputs)
        coarse_polys = coarse_m11 if model_with_diffusion.use_coarse_polylines else None

        diff_loss, _ = model_with_diffusion.diffusion.p_losses(
            x_start=x0_gt_m11,
            t=t,
            cond_tokens=cond_tokens,
            query_context=query_context,
            coarse_polys=coarse_polys,
            valid_mask=valid_mask,
        )

        loss = detr_losses["loss_total"] + lambda_diff * diff_loss
        loss.backward()
        optimizer.step()

        losses_sum["loss_ce"] += float(detr_losses["loss_ce"].item())
        losses_sum["loss_poly"] += float(detr_losses["loss_poly"].item())
        losses_sum["loss_bbox_giou"] += float(detr_losses["loss_bbox_giou"].item())
        losses_sum["loss_smooth"] += float(detr_losses["loss_smooth"].item())
        losses_sum["loss_total_detr"] += float(detr_losses["loss_total"].item())
        losses_sum["loss_diff"] += float(diff_loss.item())
        losses_sum["loss_total"] += float(loss.item())
        n_batches += 1

    return {k: v / max(n_batches, 1) for k, v in losses_sum.items()}


@torch.no_grad()
def eval_stage3_joint(
    eval_loader,
    model_with_diffusion: DetrWithDiffusion,
    criterion: nn.Module,
    matcher: nn.Module,
    device: torch.device,
    lambda_diff: float = 0.1,
):
    model_with_diffusion.eval()
    losses_sum = {
        "loss_ce": 0.0,
        "loss_poly": 0.0,
        "loss_bbox_giou": 0.0,
        "loss_smooth": 0.0,
        "loss_total_detr": 0.0,
        "loss_diff": 0.0,
        "loss_total": 0.0,
    }
    n_batches = 0

    for feats, targets in eval_loader:
        feats = feats.to(device)
        targets = move_targets_to_device(targets, device)

        outputs = model_with_diffusion(feats, return_features=True)
        detr_losses = criterion(outputs, targets)
        indices = matcher(outputs, targets)
        x0_gt_m11, coarse_m11, valid_mask = build_slot_aligned_diffusion_targets(
            outputs, targets, indices
        )

        B = feats.shape[0]
        t = torch.randint(
            0, model_with_diffusion.diffusion.timesteps, (B,), device=device
        ).long()

        cond_tokens = model_with_diffusion._pick_cond_tokens(feats, outputs)
        query_context = model_with_diffusion._pick_query_context(outputs)
        coarse_polys = coarse_m11 if model_with_diffusion.use_coarse_polylines else None

        diff_loss, _ = model_with_diffusion.diffusion.p_losses(
            x_start=x0_gt_m11,
            t=t,
            cond_tokens=cond_tokens,
            query_context=query_context,
            coarse_polys=coarse_polys,
            valid_mask=valid_mask,
        )

        loss = detr_losses["loss_total"] + lambda_diff * diff_loss

        losses_sum["loss_ce"] += float(detr_losses["loss_ce"].item())
        losses_sum["loss_poly"] += float(detr_losses["loss_poly"].item())
        losses_sum["loss_bbox_giou"] += float(detr_losses["loss_bbox_giou"].item())
        losses_sum["loss_smooth"] += float(detr_losses["loss_smooth"].item())
        losses_sum["loss_total_detr"] += float(detr_losses["loss_total"].item())
        losses_sum["loss_diff"] += float(diff_loss.item())
        losses_sum["loss_total"] += float(loss.item())
        n_batches += 1

    return {k: v / max(n_batches, 1) for k, v in losses_sum.items()}


def move_targets_to_device(targets, device):
    out = []
    for t in targets:
        out.append(
            {
                k: (
                    torch.as_tensor(v, device=device)
                    if not torch.is_tensor(v)
                    else v.to(device)
                )
                for k, v in t.items()
            }
        )
    return out


def load_checkpoint_flexible(
    module: nn.Module, ckpt_path: Path, key_candidates=None, strict: bool = True
):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if key_candidates is None:
        key_candidates = ["model_with_diffusion", "model", "state_dict"]

    state = None
    for k in key_candidates:
        if k in ckpt:
            state = ckpt[k]
            break

    if state is None:
        state = ckpt

    missing, unexpected = module.load_state_dict(state, strict=strict)
    print(f"Loaded checkpoint from {ckpt_path}")
    print(f"  missing keys: {len(missing)}")
    print(f"  unexpected keys: {len(unexpected)}")
    return ckpt


# -------------------------
# Main
# -------------------------


def main():
    cfg = dict(
        stage=2,  # 1: train DETR polyline, 2: train diffusion, 3: train jointly
        exp="detr_polyline_11",
        save_path=Path("/home/fatemeh/Downloads/hedge/results/training"),
        embed_dir=Path(
            "/home/fatemeh/Downloads/hedge/results/test_256_dino256/embs_polylines"
        ),
        # save_path=Path("/home/fkarimineja/exps/hedge"),
        # embed_dir=Path("/home/fkarimineja/data/hedge/test_256_None/embs_polylines"),
        num_points=20,
        num_polylines=100,  # 276
        num_classes=1,
        grid_size=(16, 16),
        # base model
        d_model=256,
        nhead=8,
        num_encoder_layers=4,
        num_decoder_layers=4,
        dim_feedforward=1024,
        dropout=0.01,  # default 0.1
        aux_loss=False,
        query_embed_mode="legacy",  # "detr" or "legacy"
        loss_bbox_giou=1.0,  # 1.0
        eos_coef=0.1,  # .3, .5
        # diffusion
        diff_hidden_dim=256,
        diff_nhead=8,
        diff_num_layers=4,
        diff_dim_feedforward=1024,
        diff_dropout=0.1,
        diff_timesteps=100,
        condition_source="encoder",  # "encoder" or "dino"
        use_query_context=True,
        use_coarse_polylines=True,
        lambda_diff=0.1,
        # optimizer / training
        n_epochs=200,  # 500, 3000
        batch_size=256,  # 4x256=1024 (dino256), 5x256=1280 (dino224)
        num_workers=15,  # 17
        max_lr=1e-4,  # 3e-5, 1e-4, # 3e-4,
        weight_decay=1e-2,
        use_tqdm=True,
        save_every=3000,
        seed=42,
        # checkpoints
        stage1_ckpt=Path(
            "/home/fatemeh/Downloads/hedge/results/training/best_detr_polyline_11_stage1.pt"
        ),
        stage2_ckpt=None,
        resume_ckpt=None,  # optional full resume for current stage
    )
    cfg = OmegaConf.create(cfg)

    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print(f"Using device: {torch.cuda.get_device_properties()}")

    dataset = DetrPolylineEmbDataset(
        embed_dir=cfg.embed_dir, num_points=cfg.num_points, normalize=True
    )
    n_train = int(0.8 * len(dataset))
    n_val = len(dataset) - n_train
    train_ds, val_ds = torch.utils.data.random_split(dataset, [n_train, n_val])
    print(f"Dataset: total={len(dataset)}, train={len(train_ds)}, val={len(val_ds)}")

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        collate_fn=detr_polyline_collate_fn,
    )
    eval_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        collate_fn=detr_polyline_collate_fn,
    )

    matcher = HungarianMatcherPolyline(
        MatcherCost(class_cost=1.0, poly_cost=5.0, bbox_giou_cost=cfg.loss_bbox_giou)
    )
    criterion = DetrPolylineCriterion(
        num_classes=cfg.num_classes,
        matcher=matcher,
        eos_coef=cfg.eos_coef,
        loss_poly=5.0,
        loss_bbox_giou=cfg.loss_bbox_giou,
        loss_smooth=0.0,  # set small value like 0.1 for smoother curves
    )

    criterion.to(device)

    base_model = DetrPolylineFromEmbeddings(
        in_dim=1024,
        num_classes=cfg.num_classes,
        num_queries=cfg.num_polylines,
        d_model=cfg.d_model,
        nhead=cfg.nhead,
        num_encoder_layers=cfg.num_encoder_layers,
        num_decoder_layers=cfg.num_decoder_layers,
        dim_feedforward=cfg.dim_feedforward,
        dropout=cfg.dropout,
        grid_size=cfg.grid_size,
        num_points=cfg.num_points,
        aux_loss=cfg.aux_loss,
        query_embed_mode=cfg.query_embed_mode,
    )

    diffusion_decoder = PolylineDiffusionDecoder(
        num_queries=cfg.num_polylines,
        num_points=cfg.num_points,
        hidden_dim=cfg.diff_hidden_dim,
        nhead=cfg.diff_nhead,
        num_layers=cfg.diff_num_layers,
        dim_feedforward=cfg.diff_dim_feedforward,
        dropout=cfg.diff_dropout,
        cond_dim=cfg.d_model if cfg.condition_source == "encoder" else 1024,
        query_dim=cfg.d_model,
        use_cond_tokens=True,
        use_query_context=cfg.use_query_context,
        use_coarse_polylines=cfg.use_coarse_polylines,
    )

    diffusion = GaussianPolylineDiffusion(
        model=diffusion_decoder,
        timesteps=cfg.diff_timesteps,
        beta_start=1e-4,
        beta_end=2e-2,
        objective="eps",
    )

    model_with_diffusion = DetrWithDiffusion(
        base_model=base_model,
        diffusion=diffusion,
        num_queries=cfg.num_polylines,
        num_points=cfg.num_points,
        condition_source=cfg.condition_source,
        use_query_context=cfg.use_query_context,
        use_coarse_polylines=cfg.use_coarse_polylines,
    ).to(device)

    tb_dir = cfg.save_path / f"tensorboard/{cfg.exp}_stage{cfg.stage}"
    tb_dir.mkdir(parents=True, exist_ok=True)
    writer = tensorboard.SummaryWriter(tb_dir)
    best_val = float("inf")

    if cfg.stage == 1:
        model = model_with_diffusion.base_model
        model.to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=cfg.max_lr, weight_decay=cfg.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg.n_epochs, eta_min=1e-6
        )

        for epoch in tqdm(range(1, cfg.n_epochs + 1), disable=cfg.use_tqdm):
            s_time = datetime.now().replace(microsecond=0)
            print(f"Epoch {epoch:03d}/{cfg.n_epochs} starting at {s_time}")

            train_losses = train_one_epoch(
                train_loader, model, criterion, optimizer, device
            )
            eval_losses = eval_one_epoch(eval_loader, model, criterion, device)

            e_time = datetime.now().replace(microsecond=0)
            print(
                f"Epoch {epoch:03d}/{cfg.n_epochs} "
                f"train_total={train_losses['loss_total']:.4f} "
                f"(ce={train_losses['loss_ce']:.4f}, poly={train_losses['loss_poly']:.4f}, "
                f"bbox_giou={train_losses['loss_bbox_giou']:.4f}, smooth={train_losses['loss_smooth']:.4f}) "
                f"eval_total={eval_losses['loss_total']:.4f} "
                f"(ce={eval_losses['loss_ce']:.4f}, poly={eval_losses['loss_poly']:.4f}, "
                f"bbox_giou={eval_losses['loss_bbox_giou']:.4f}, smooth={eval_losses['loss_smooth']:.4f})"
            )
            print(
                f"Epoch {epoch:03d}/{cfg.n_epochs} finished at {e_time} (duration {e_time - s_time})"
            )

            tb_add_losses(writer, epoch, train_losses, "train")
            tb_add_losses(writer, epoch, eval_losses, "eval")

            ckpt = {
                "epoch": epoch,
                "model": model.state_dict(),
                # "optimizer": optimizer.state_dict(),
                # "train_losses": train_losses,
                # "eval_losses": eval_losses,
                # "cfg": OmegaConf.to_container(cfg, resolve=True),
            }
            if eval_losses["loss_total"] < best_val:
                best_val = eval_losses["loss_total"]
                torch.save(ckpt, cfg.save_path / f"best_{cfg.exp}_stage{cfg.stage}.pt")
                print(f"Saved best: {best_val:.4f} at epoch {epoch}")
            if epoch % cfg.save_every == 0:
                torch.save(
                    ckpt,
                    cfg.save_path / f"{cfg.exp}_{epoch}_stage{cfg.stage}.pt",
                )
            # scheduler.step()
        torch.save(
            ckpt,
            cfg.save_path / f"{cfg.exp}_stage{cfg.stage}.pt",
        )
        print(f"Saved final model: {best_val:.4f} at epoch {epoch}")

    elif cfg.stage == 2:
        if cfg.stage1_ckpt is None:
            raise ValueError(
                "For stage=2, cfg.stage1_ckpt must point to a trained stage-1 checkpoint."
            )

        load_checkpoint_flexible(
            model_with_diffusion.base_model,
            cfg.stage1_ckpt,
            key_candidates=["model"],
            strict=True,
        )

        if cfg.resume_ckpt is not None:
            load_checkpoint_flexible(
                model_with_diffusion,
                cfg.resume_ckpt,
                key_candidates=["model_with_diffusion", "model"],
                strict=False,
            )

        set_requires_grad(model_with_diffusion.base_model, False)
        set_requires_grad(model_with_diffusion.diffusion, True)

        optimizer = torch.optim.AdamW(
            model_with_diffusion.diffusion.parameters(),
            lr=cfg.max_lr,
            weight_decay=cfg.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg.n_epochs, eta_min=1e-6
        )

        for epoch in tqdm(range(1, cfg.n_epochs + 1), disable=cfg.use_tqdm):
            s_time = datetime.now().replace(microsecond=0)
            print(f"Epoch {epoch:03d}/{cfg.n_epochs} starting at {s_time}")

            train_losses = train_stage2_diffusion_only(
                train_loader=train_loader,
                model_with_diffusion=model_with_diffusion,
                matcher=matcher,
                optimizer=optimizer,
                device=device,
            )
            eval_losses = eval_stage2_diffusion_only(
                eval_loader=eval_loader,
                model_with_diffusion=model_with_diffusion,
                matcher=matcher,
                device=device,
            )

            e_time = datetime.now().replace(microsecond=0)
            print(
                f"Epoch {epoch:03d}/{cfg.n_epochs} "
                f"train_diff={train_losses['loss_diff']:.4f} "
                f"eval_diff={eval_losses['loss_diff']:.4f}"
            )
            print(
                f"Epoch {epoch:03d}/{cfg.n_epochs} finished at {e_time} (duration {e_time - s_time})"
            )

            tb_add_losses(writer, epoch, train_losses, "train")
            tb_add_losses(writer, epoch, eval_losses, "eval")

            ckpt = {
                "model_with_diffusion": model_with_diffusion.state_dict(),
                "epoch": epoch,
            }
            if eval_losses["loss_diff"] < best_val:
                best_val = eval_losses["loss_diff"]
                torch.save(ckpt, cfg.save_path / f"best_{cfg.exp}_stage{cfg.stage}.pt")
                print(f"Saved best: {best_val:.4f} at epoch {epoch}")
            if epoch % cfg.save_every == 0:
                torch.save(
                    ckpt,
                    cfg.save_path / f"{cfg.exp}_{epoch}_stage{cfg.stage}.pt",
                )

            scheduler.step()

        torch.save(
            ckpt,
            cfg.save_path / f"{cfg.exp}_stage{cfg.stage}.pt",
        )
        print(f"Saved final model: {best_val:.4f} at epoch {epoch}")

    elif cfg.stage == 3:
        if cfg.stage2_ckpt is not None:
            load_checkpoint_flexible(
                model_with_diffusion,
                cfg.stage2_ckpt,
                key_candidates=["model_with_diffusion", "model"],
                strict=False,
            )
        elif cfg.stage1_ckpt is not None:
            load_checkpoint_flexible(
                model_with_diffusion.base_model,
                cfg.stage1_ckpt,
                key_candidates=["model"],
                strict=True,
            )
        else:
            raise ValueError(
                "For stage=3, provide cfg.stage2_ckpt or at least cfg.stage1_ckpt."
            )

        if cfg.resume_ckpt is not None:
            load_checkpoint_flexible(
                model_with_diffusion,
                cfg.resume_ckpt,
                key_candidates=["model_with_diffusion", "model"],
                strict=False,
            )

        set_requires_grad(model_with_diffusion.base_model, True)
        set_requires_grad(model_with_diffusion.diffusion, True)

        optimizer = torch.optim.AdamW(
            model_with_diffusion.parameters(),
            lr=cfg.max_lr,
            weight_decay=cfg.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg.n_epochs, eta_min=1e-6
        )

        for epoch in tqdm(range(1, cfg.n_epochs + 1), disable=cfg.use_tqdm):
            s_time = datetime.now().replace(microsecond=0)
            print(f"Epoch {epoch:03d}/{cfg.n_epochs} starting at {s_time}")

            train_losses = train_stage3_joint(
                train_loader=train_loader,
                model_with_diffusion=model_with_diffusion,
                criterion=criterion,
                matcher=matcher,
                optimizer=optimizer,
                device=device,
                lambda_diff=cfg.lambda_diff,
            )
            eval_losses = eval_stage3_joint(
                eval_loader=eval_loader,
                model_with_diffusion=model_with_diffusion,
                criterion=criterion,
                matcher=matcher,
                device=device,
                lambda_diff=cfg.lambda_diff,
            )

            e_time = datetime.now().replace(microsecond=0)
            print(
                f"Epoch {epoch:03d}/{cfg.n_epochs} "
                f"train_total={train_losses['loss_total']:.4f} "
                f"(detr={train_losses['loss_total_detr']:.4f}, diff={train_losses['loss_diff']:.4f}) "
                f"eval_total={eval_losses['loss_total']:.4f} "
                f"(detr={eval_losses['loss_total_detr']:.4f}, diff={eval_losses['loss_diff']:.4f})"
            )
            print(
                f"Epoch {epoch:03d}/{cfg.n_epochs} finished at {e_time} (duration {e_time - s_time})"
            )

            tb_add_losses(writer, epoch, train_losses, "train")
            tb_add_losses(writer, epoch, eval_losses, "eval")

            ckpt = {
                "model_with_diffusion": model_with_diffusion.state_dict(),
                "epoch": epoch,
            }
            if eval_losses["loss_total"] < best_val:
                best_val = eval_losses["loss_total"]
                torch.save(ckpt, cfg.save_path / f"best_{cfg.exp}_stage{cfg.stage}.pt")
                print(f"Saved best: {best_val:.4f} at epoch {epoch}")
            if epoch % cfg.save_every == 0:
                torch.save(
                    ckpt,
                    cfg.save_path / f"{cfg.exp}_{epoch}_stage{cfg.stage}.pt",
                )

            scheduler.step()

        torch.save(
            ckpt,
            cfg.save_path / f"{cfg.exp}_stage{cfg.stage}.pt",
        )
        print(f"Saved final model: {best_val:.4f} at epoch {epoch}")


"""
    model.load_state_dict(
        torch.load("/home/fatemeh/Downloads/hedge/snellius/best_detr_polyline_1.pt", map_location=device)["model"]
    )
    # Example inference on eval set
    feats, targets = next(iter(eval_loader))
    image_sizes = [t["image_size"].tolist() for t in targets]
    preds = detr_polyline_inference(
        model=model,
        feats=feats,
        image_sizes=image_sizes,
        score_thresh=0.9,
        topk=20,
        device=device,
    )

    # preds[0]["polylines_px"] is (M,K,2) in pixel coords
    print(preds[0]["scores"].shape, preds[0]["polylines_px"].shape)
    import cv2
    import matplotlib.pyplot as plt
    def visualize_polylines(im, polylines):
        plt.figure()
        plt.imshow(im)
        for poly in polylines:
            plt.plot(poly[:, 0], poly[:, 1], "*")
        plt.show(block=False)
    i = 0
    inp = dataset.files[val_ds.indices[i]]
    im = cv2.imread(str(inp.parent.parent/f"images/{inp.stem}.png"))
    polylines = np.load(inp.parent.parent/f"embs_polylines/{inp.stem}.npz")["polylines"]
    pred_polylines = preds[i]["polylines_px"] # [M,K,2]

    visualize_polylines(im, pred_polylines)
    visualize_polylines(im, polylines)

    a = np.load(inp) # /home/fatemeh/Downloads/hedge/results/test_mini/embs_polylines/pos_000003.npz
    a = torch.tensor(a["feat"], dtype=torch.float32).unsqueeze(0)
    preds = detr_polyline_inference(model=model,feats=a,image_sizes=[image_sizes[0]],score_thresh=0.9,topk=20,device=device)
"""


if __name__ == "__main__":
    main()
