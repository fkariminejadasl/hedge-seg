"""
Similar code as train_detr_dino_polyline_rel.py with these changes:
- Use MapTR-style query layout (instance embedding + point embedding) instead of DINO-style query layout (polyline queries)
- TODO: Use ResNet backbone instead of DINOv3 backbone
- Remove diffusion stage
"""

import copy
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
# Utilities: boxes
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


def box_xyxy_to_cxcywh(boxes: torch.Tensor) -> torch.Tensor:
    x0, y0, x1, y1 = boxes.unbind(-1)
    cx = (x0 + x1) * 0.5
    cy = (y0 + y1) * 0.5
    w = (x1 - x0).clamp(min=1e-6)
    h = (y1 - y0).clamp(min=1e-6)
    return torch.stack([cx, cy, w, h], dim=-1)


def box_cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    cx, cy, w, h = boxes.unbind(-1)
    x0 = cx - 0.5 * w
    y0 = cy - 0.5 * h
    x1 = cx + 0.5 * w
    y1 = cy + 0.5 * h
    return torch.stack([x0, y0, x1, y1], dim=-1)


def inverse_sigmoid(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    x = x.clamp(min=eps, max=1.0 - eps)
    return torch.log(x / (1.0 - x))


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
        query_content: Optional[torch.Tensor] = None,  # (Q, D)
        pos_embed: Optional[torch.Tensor] = None,  # (B, L, D)
        mask: Optional[torch.Tensor] = None,  # (B, L)
    ):
        B, _, _ = src.shape
        query_embed = query_embed.unsqueeze(0).expand(B, -1, -1)  # (B,Q,D)
        if query_content is None:
            tgt = torch.zeros_like(query_embed)
        else:
            tgt = query_content.unsqueeze(0).expand(B, -1, -1)  # (B,Q,D)

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
        query_content: Optional[torch.Tensor] = None,  # (Q,D)
        pos_embed: Optional[torch.Tensor] = None,  # (B,L,D)
        mask: Optional[torch.Tensor] = None,  # (B,L)
    ):
        B, _, _ = src.shape

        if pos_embed is not None:
            src = src + pos_embed

        memory = self.encoder(src, src_key_padding_mask=mask)

        query = query_embed.unsqueeze(0).expand(B, -1, -1)  # (B,Q,D)
        if query_content is None:
            output = torch.zeros_like(query) + query
        else:
            output = query_content.unsqueeze(0).expand(B, -1, -1) + query

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
      pred_boxes: (B, Q, 4) in [0,1], cxcywh
      pred_polylines: (B, Q, K, 2) in [0,1]
      aux_outputs: list[dict], optional

    MapTR-style query layout:
      instance_embedding: (Q, 2D)
      pts_embedding: (K, 2D)
      object_query_embed: (Q*K, 2D), split into query_pos/query_content
      decoder hs: (num_layers, B, Q*K, D) -> (num_layers, B, Q, K, D)
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
        self.instance_embedding = nn.Embedding(num_queries, d_model * 2)
        self.pts_embedding = nn.Embedding(num_points, d_model * 2)

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
        self.reference_points = nn.Linear(d_model, 2)
        self.point_embed = MLP(d_model, d_model, 2, num_layers=3)

    def _build_hierarchical_queries(self) -> Tuple[torch.Tensor, torch.Tensor]:
        pts_embeds = self.pts_embedding.weight.unsqueeze(0)  # (1,K,2D)
        instance_embeds = self.instance_embedding.weight.unsqueeze(1)  # (Q,1,2D)
        object_query_embed = (pts_embeds + instance_embeds).flatten(0, 1)  # (Q*K,2D)
        query_pos, query_content = object_query_embed.chunk(2, dim=-1)  # (Q*K,D)
        return query_pos, query_content

    def _decode_polylines(
        self,
        hs: torch.Tensor,  # (num_layers,B,Q*K,D)
        query_pos: torch.Tensor,  # (Q*K,D)
    ) -> torch.Tensor:
        reference_points = self.reference_points(query_pos).sigmoid()  # (Q*K,2)
        point_logits = self.point_embed(hs)  # (num_layers,B,Q*K,2)
        point_logits = point_logits + inverse_sigmoid(reference_points)[None, None]
        points = point_logits.sigmoid()

        return points.view(
            hs.shape[0],
            hs.shape[1],
            self.num_queries,
            self.num_points,
            2,
        )

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_boxes, outputs_poly):
        return [
            {
                "pred_logits": a,
                "pred_boxes": b,
                "pred_polylines": c,
            }
            for a, b, c in zip(
                outputs_class[:-1], outputs_boxes[:-1], outputs_poly[:-1]
            )
        ]

    def forward(
        self, x: torch.Tensor, return_features: bool = False
    ) -> Dict[str, torch.Tensor]:
        B, L, _ = x.shape
        assert (
            L == self.grid_h * self.grid_w
        ), f"Expected {self.grid_h * self.grid_w} tokens, got {L}"

        dino_tokens = x
        src = self.input_proj(x)
        pos = self.pos_embed(B=B, H=self.grid_h, W=self.grid_w, device=x.device)
        query_pos, query_content = self._build_hierarchical_queries()

        mask = None
        hs, memory = self.transformer(
            src=src,
            query_embed=query_pos,
            query_content=query_content,
            pos_embed=pos,
            mask=mask,
        )  # hs: (num_layers,B,Q*K,D)

        hs_poly = hs.view(
            hs.shape[0],
            B,
            self.num_queries,
            self.num_points,
            hs.shape[-1],
        )  # (num_layers,B,Q,K,D)
        outputs_class = self.class_embed(hs_poly.mean(dim=3))  # (num_layers,B,Q,C+1)
        outputs_poly = self._decode_polylines(hs, query_pos)  # (num_layers,B,Q,K,2)
        outputs_boxes = box_xyxy_to_cxcywh(
            polyline_to_bbox_xyxy(outputs_poly)
        )  # (num_layers,B,Q,4)

        out = {
            "pred_logits": outputs_class[-1],
            "pred_boxes": outputs_boxes[-1],
            "pred_polylines": outputs_poly[-1],
        }

        if return_features:
            out["memory"] = memory
            out["hs_last"] = hs_poly[-1].mean(dim=2)
            out["dino_tokens"] = dino_tokens

        if self.aux_loss:
            out["aux_outputs"] = self._set_aux_loss(
                outputs_class, outputs_boxes, outputs_poly
            )

        return out


# -------------------------
# Matcher + Criterion for polylines
# -------------------------


@dataclass
class MatcherCost:
    class_cost: float = 1.0
    poly_cost: float = 5.0
    bbox_cost: float = 2.0
    bbox_giou_cost: float = 2.0


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
          pred_boxes: (B,Q,4) cxcywh normalized
          pred_polylines: (B,Q,K,2) normalized
        """
        bs, num_queries = outputs["pred_logits"].shape[:2]
        out_prob = outputs["pred_logits"].softmax(-1)
        out_poly = outputs["pred_polylines"]
        out_box = outputs["pred_boxes"]

        indices = []
        for b in range(bs):
            tgt_ids = targets[b]["labels"]
            tgt_poly = targets[b]["polylines"]

            if tgt_poly.numel() == 0:
                indices.append(
                    (
                        torch.empty(0, dtype=torch.int64),
                        torch.empty(0, dtype=torch.int64),
                    )
                )
                continue

            tgt_bbox_xyxy = polyline_to_bbox_xyxy(tgt_poly)
            tgt_bbox = box_xyxy_to_cxcywh(tgt_bbox_xyxy)

            cost_class = -out_prob[b][:, tgt_ids]

            Q = out_poly[b].shape[0]
            K = out_poly[b].shape[1]

            out_flat = out_poly[b].reshape(Q, 2 * K)
            tgt_flat = tgt_poly.reshape(tgt_poly.shape[0], 2 * K)
            tgt_rev = torch.flip(tgt_poly, dims=[1]).reshape(tgt_poly.shape[0], 2 * K)

            cost_fwd = torch.cdist(out_flat, tgt_flat, p=1) / float(2 * K)
            cost_rev = torch.cdist(out_flat, tgt_rev, p=1) / float(2 * K)
            cost_poly = torch.minimum(cost_fwd, cost_rev)

            cost_bbox = torch.cdist(out_box[b], tgt_bbox, p=1) / 4.0

            out_bbox_xyxy = box_cxcywh_to_xyxy(out_box[b]).clamp(0.0, 1.0)
            cost_giou = -generalized_box_iou_xyxy(out_bbox_xyxy, tgt_bbox_xyxy)

            C = (
                self.cost.class_cost * cost_class
                + self.cost.poly_cost * cost_poly
                + self.cost.bbox_cost * cost_bbox
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
        loss_bbox: float = 2.0,
        loss_bbox_giou: float = 2.0,
        loss_smooth: float = 0.05,
        loss_card: float = 0.5,
        loss_len: float = 1.0,
        loss_dir: float = 0.5,
        aux_weight: float = 0.5,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.eos_coef = eos_coef

        self.loss_poly_w = loss_poly
        self.loss_bbox_w = loss_bbox
        self.loss_bbox_giou_w = loss_bbox_giou
        self.loss_smooth_w = loss_smooth
        self.loss_card_w = loss_card
        self.loss_len_w = loss_len
        self.loss_dir_w = loss_dir
        self.aux_weight = aux_weight

        empty_weight = torch.ones(num_classes + 1)
        empty_weight[-1] = eos_coef
        self.register_buffer("empty_weight", empty_weight)

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: List[Dict[str, torch.Tensor]],
    ) -> Dict[str, torch.Tensor]:
        outputs_without_aux = {k: v for k, v in outputs.items() if k != "aux_outputs"}

        losses = self._compute_losses(outputs_without_aux, targets, suffix="")
        total = losses["loss_total"]

        if "aux_outputs" in outputs:
            for i, aux_outputs in enumerate(outputs["aux_outputs"]):
                aux_losses = self._compute_losses(aux_outputs, targets, suffix="")
                total = total + self.aux_weight * aux_losses["loss_total"]

                for k, v in aux_losses.items():
                    losses[f"aux{i}_{k}"] = v

        losses["loss_total"] = total
        return losses

    def _compute_losses(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: List[Dict[str, torch.Tensor]],
        suffix: str = "",
    ) -> Dict[str, torch.Tensor]:
        indices = self.matcher(outputs, targets)

        loss_ce = self.loss_labels(outputs, targets, indices)
        loss_poly = self.loss_polylines(outputs, targets, indices)
        loss_bbox = self.loss_boxes(outputs, targets, indices)
        loss_giou = self.loss_bbox_giou(outputs, targets, indices)
        loss_smooth = self.loss_smoothness(outputs, indices)
        loss_card = self.loss_cardinality(outputs, targets)
        loss_len, loss_dir = self.loss_length_direction(outputs, targets, indices)

        loss_total = (
            loss_ce
            + self.loss_poly_w * loss_poly
            + self.loss_bbox_w * loss_bbox
            + self.loss_bbox_giou_w * loss_giou
            + self.loss_smooth_w * loss_smooth
            + self.loss_card_w * loss_card
            + self.loss_len_w * loss_len
            + self.loss_dir_w * loss_dir
        )

        return {
            f"loss_ce{suffix}": loss_ce,
            f"loss_poly{suffix}": loss_poly,
            f"loss_bbox{suffix}": loss_bbox,
            f"loss_bbox_giou{suffix}": loss_giou,
            f"loss_smooth{suffix}": loss_smooth,
            f"loss_card{suffix}": loss_card,
            f"loss_len{suffix}": loss_len,
            f"loss_dir{suffix}": loss_dir,
            f"loss_total{suffix}": loss_total,
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

        return F.cross_entropy(
            src_logits.transpose(1, 2), target_classes, weight=self.empty_weight
        )

    def loss_polylines(self, outputs, targets, indices):
        pred_poly = outputs["pred_polylines"]  # (B,Q,K,2)
        loss = pred_poly.new_tensor(0.0)
        n_matched = 0

        for b, (src_idx, tgt_idx) in enumerate(indices):
            if len(src_idx) == 0:
                continue

            s = pred_poly[b, src_idx]  # (M,K,2)
            t = targets[b]["polylines"][tgt_idx]  # (M,K,2)
            t_rev = torch.flip(t, dims=[1])  # (M,K,2)

            l1_fwd = F.l1_loss(s, t, reduction="none").sum(dim=(1, 2))  # (M,)
            l1_rev = F.l1_loss(s, t_rev, reduction="none").sum(dim=(1, 2))  # (M,)
            loss = loss + torch.minimum(l1_fwd, l1_rev).sum()
            n_matched += s.shape[0]

        n_matched = max(n_matched, 1)
        K = pred_poly.shape[2]
        return loss / (n_matched * K * 2.0)

    def loss_boxes(self, outputs, targets, indices):
        pred_box = outputs["pred_boxes"]  # cxcywh
        loss = pred_box.new_tensor(0.0)
        n_matched = 0

        for b, (src_idx, tgt_idx) in enumerate(indices):
            if len(src_idx) == 0:
                continue

            s = pred_box[b, src_idx]
            t_poly = targets[b]["polylines"][tgt_idx]
            t_box = box_xyxy_to_cxcywh(polyline_to_bbox_xyxy(t_poly))

            loss = loss + F.l1_loss(s, t_box, reduction="sum")
            n_matched += s.shape[0]

        n_matched = max(n_matched, 1)
        return loss / (n_matched * 4.0)

    def loss_bbox_giou(self, outputs, targets, indices):
        pred_box = outputs["pred_boxes"]
        loss = pred_box.new_tensor(0.0)
        n_matched = 0

        for b, (src_idx, tgt_idx) in enumerate(indices):
            if len(src_idx) == 0:
                continue

            s = box_cxcywh_to_xyxy(pred_box[b, src_idx]).clamp(0.0, 1.0)
            t_poly = targets[b]["polylines"][tgt_idx]
            t = polyline_to_bbox_xyxy(t_poly)

            giou = generalized_box_iou_xyxy(s, t).diag()
            loss = loss + (1.0 - giou).sum()
            n_matched += s.shape[0]

        n_matched = max(n_matched, 1)
        return loss / n_matched

    def loss_smoothness(self, outputs, indices):
        pred_poly = outputs["pred_polylines"]  # (B,Q,K,2)
        loss = pred_poly.new_tensor(0.0)
        n = 0

        for b, (src_idx, _) in enumerate(indices):
            if len(src_idx) == 0:
                continue
            p = pred_poly[b, src_idx]  # (M,K,2)
            if p.shape[1] < 3:
                continue
            d2 = p[:, 2:] - 2.0 * p[:, 1:-1] + p[:, :-2]  # (M,K-2,2)
            loss = loss + (d2**2).mean()
            n += 1

        if n == 0:
            return pred_poly.new_tensor(0.0)
        return loss / n

    def loss_cardinality(self, outputs, targets):
        pred_logits = outputs["pred_logits"]
        pred_classes = pred_logits.argmax(-1)
        pred_counts = (pred_classes != self.num_classes).sum(dim=1).float()

        tgt_counts = torch.as_tensor(
            [len(t["labels"]) for t in targets],
            dtype=torch.float32,
            device=pred_logits.device,
        )
        return F.l1_loss(pred_counts, tgt_counts)

    def loss_length_direction(self, outputs, targets, indices):
        pred_poly = outputs["pred_polylines"]
        loss_len = pred_poly.new_tensor(0.0)
        loss_dir = pred_poly.new_tensor(0.0)
        n = 0

        for b, (src_idx, tgt_idx) in enumerate(indices):
            if len(src_idx) == 0:
                continue

            s = pred_poly[b, src_idx]
            t = targets[b]["polylines"][tgt_idx]
            t_rev = torch.flip(t, dims=[1])

            s_v = s[:, 1:] - s[:, :-1]
            t_v = t[:, 1:] - t[:, :-1]
            tr_v = t_rev[:, 1:] - t_rev[:, :-1]

            s_len = torch.norm(s_v, dim=-1)
            t_len = torch.norm(t_v, dim=-1)
            tr_len = torch.norm(tr_v, dim=-1)

            len_fwd = F.l1_loss(s_len, t_len, reduction="none").mean(dim=1)
            len_rev = F.l1_loss(s_len, tr_len, reduction="none").mean(dim=1)

            s_dir = F.normalize(s_v, dim=-1, eps=1e-6)
            t_dir = F.normalize(t_v, dim=-1, eps=1e-6)
            tr_dir = F.normalize(tr_v, dim=-1, eps=1e-6)

            dir_fwd = (1.0 - (s_dir * t_dir).sum(dim=-1)).mean(dim=1)
            dir_rev = (1.0 - (s_dir * tr_dir).sum(dim=-1)).mean(dim=1)

            use_rev = (len_rev + dir_rev) < (len_fwd + dir_fwd)

            loss_len = loss_len + torch.where(use_rev, len_rev, len_fwd).sum()
            loss_dir = loss_dir + torch.where(use_rev, dir_rev, dir_fwd).sum()
            n += s.shape[0]

        if n == 0:
            z = pred_poly.new_tensor(0.0)
            return z, z

        return loss_len / n, loss_dir / n


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
        self.files = sorted(embed_dir.glob("*.npz"))
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

        target = {
            "labels": labels,
            "polylines": polylines,
            "image_size": image_size,
            "stem": self.files[idx].stem,
        }
        return feat, target


def detr_polyline_collate_fn(batch):
    feats, targets = zip(*batch)
    feats = torch.stack(feats, dim=0)  # (B,196,1024)
    return feats, list(targets)


# -------------------------
# Training / eval / inference
# -------------------------


def tb_add_losses(writer, epoch: int, losses: dict, stage: str):
    wanted = {"loss_total"}
    d = {f"{stage}_{k}": float(v) for k, v in losses.items() if k in wanted}
    writer.add_scalars("losses", d, epoch)


def train_one_epoch(loader, model, criterion, optimizer, device):
    model.train()
    sums = {
        "loss_ce": 0.0,
        "loss_poly": 0.0,
        "loss_bbox": 0.0,
        "loss_bbox_giou": 0.0,
        "loss_smooth": 0.0,
        "loss_card": 0.0,
        "loss_len": 0.0,
        "loss_dir": 0.0,
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
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        # a = {n:round(p.grad.max().item(),2) for n, p in model.named_parameters()}
        # {k:v for k, v in a.items() if v == max(a.values())}
        optimizer.step()

        bs = feats.size(0)
        n += bs
        for k in sums:
            if k in loss_dict:
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
        "loss_bbox": 0.0,
        "loss_bbox_giou": 0.0,
        "loss_smooth": 0.0,
        "loss_card": 0.0,
        "loss_len": 0.0,
        "loss_dir": 0.0,
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
            if k in loss_dict:
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
    model.eval()
    if device is not None:
        feats = feats.to(device)
        model = model.to(device)

    outputs = model(feats)
    logits = outputs["pred_logits"]  # (B,Q,C+1)
    polylines = outputs["pred_polylines"]  # (B,Q,K,2) normalized
    boxes = outputs["pred_boxes"]  # cxcywh normalized

    prob = F.softmax(logits, dim=-1)  # (B,Q,C+1)
    scores_all, labels_all = prob[..., :-1].max(dim=-1)  # exclude "no-object" -> (B,Q)

    B, Q = scores_all.shape
    results = []

    for b in range(B):
        H, W = image_sizes[b]

        scores = scores_all[b]
        labels = labels_all[b]
        polys = polylines[b]  # (Q,K,2)
        bx = boxes[b]

        keep = scores >= score_thresh
        if keep.any():
            scores = scores[keep]
            labels = labels[keep]
            polys = polys[keep]
            bx = bx[keep]
        else:
            results.append(
                {
                    "scores": scores.new_zeros((0,)),
                    "labels": labels.new_zeros((0,), dtype=torch.long),
                    "polylines_norm": polys.new_zeros((0, polys.shape[1], 2)),
                    "polylines_px": polys.new_zeros((0, polys.shape[1], 2)),
                    "boxes_norm": bx.new_zeros((0, 4)),
                    "boxes_px_xyxy": bx.new_zeros((0, 4)),
                }
            )
            continue

        if scores.numel() > topk:
            top_idx = torch.topk(scores, k=topk, largest=True).indices
            scores = scores[top_idx]
            labels = labels[top_idx]
            polys = polys[top_idx]
            bx = bx[top_idx]

        polys_px = polys.clone()
        polys_px[..., 0] = (polys_px[..., 0] * float(W - 1)).clamp(0, W - 1)
        polys_px[..., 1] = (polys_px[..., 1] * float(H - 1)).clamp(0, H - 1)

        bx_xyxy = box_cxcywh_to_xyxy(bx).clone()
        bx_xyxy[..., 0] = (bx_xyxy[..., 0] * float(W - 1)).clamp(0, W - 1)
        bx_xyxy[..., 1] = (bx_xyxy[..., 1] * float(H - 1)).clamp(0, H - 1)
        bx_xyxy[..., 2] = (bx_xyxy[..., 2] * float(W - 1)).clamp(0, W - 1)
        bx_xyxy[..., 3] = (bx_xyxy[..., 3] * float(H - 1)).clamp(0, H - 1)

        results.append(
            {
                "scores": scores.detach().cpu(),
                "labels": labels.detach().cpu(),
                "polylines_norm": polys.detach().cpu(),
                "polylines_px": polys_px.detach().cpu(),
                "boxes_norm": bx.detach().cpu(),
                "boxes_px_xyxy": bx_xyxy.detach().cpu(),
            }
        )

    return results


@torch.no_grad()
def infer_model(loader, model, device, cfg):
    model.eval()

    out_dir = (
        Path(cfg.infer_out_dir)
        if cfg.infer_out_dir is not None
        else cfg.save_path / f"{cfg.exp}_inference"
    )
    pred_dir = out_dir / "polylines"
    pred_dir.mkdir(parents=True, exist_ok=True)

    for feats, targets in tqdm(loader, disable=cfg.disable_tqdm):
        image_sizes = []
        for t in targets:
            size = t["image_size"]
            if torch.is_tensor(size):
                size = size.tolist()
            elif hasattr(size, "tolist"):
                size = size.tolist()
            image_sizes.append(size)

        preds = detr_polyline_inference(
            model=model,
            feats=feats,
            image_sizes=image_sizes,
            score_thresh=cfg.infer_score_thresh,
            topk=cfg.infer_topk,
            device=device,
        )

        for pred, target, image_size in zip(preds, targets, image_sizes):
            np.savez_compressed(
                pred_dir / f"{target['stem']}.npz",
                polylines=pred["polylines_px"].numpy(),
                polylines_norm=pred["polylines_norm"].numpy(),
                scores=pred["scores"].numpy(),
                labels=pred["labels"].numpy(),
                image_size=np.asarray(image_size, dtype=np.int32),
            )


# =========================================================


def load_checkpoint_flexible(
    module: nn.Module, ckpt_path: Path, key_candidates=None, strict: bool = True
):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if key_candidates is None:
        key_candidates = ["model", "state_dict"]

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


def save_train_val_split_files(dataset, train_ds, val_ds, split_dir: Path):
    split_dir.mkdir(parents=True, exist_ok=True)

    split_paths = {
        "train": [dataset.files[i] for i in train_ds.indices],
        "val": [dataset.files[i] for i in val_ds.indices],
    }

    for split, files in split_paths.items():
        with open(split_dir / f"{split}.txt", "w") as f:
            for path in files:
                f.write(str(path) + "\n")

    print(f"Saved train/val split to {split_dir}")


# -------------------------
# Main
# -------------------------


def main(cfg):
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print(f"Using device: {torch.cuda.get_device_properties()}")

    dataset = DetrPolylineEmbDataset(
        embed_dir=cfg.embed_dir, num_points=cfg.num_points, normalize=True
    )
    n_train = int(0.8 * len(dataset))
    n_val = len(dataset) - n_train
    train_ds, val_ds = torch.utils.data.random_split(
        dataset, [n_train, n_val], generator=torch.Generator().manual_seed(cfg.seed)
    )
    print(f"Dataset: total={len(dataset)}, train={len(train_ds)}, val={len(val_ds)}")

    if cfg.save_splits:
        split_dir = (
            Path(cfg.split_dir)
            if cfg.split_dir is not None
            else cfg.save_path / "splits"
        )
        save_train_val_split_files(dataset, train_ds, val_ds, split_dir)

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
        MatcherCost(
            class_cost=cfg.class_cost,
            poly_cost=cfg.poly_cost,
            bbox_cost=cfg.bbox_cost,
            bbox_giou_cost=cfg.loss_bbox_giou,
        )
    )
    criterion = DetrPolylineCriterion(
        num_classes=cfg.num_classes,
        matcher=matcher,
        eos_coef=cfg.eos_coef,
        loss_poly=cfg.loss_poly,
        loss_bbox=cfg.loss_bbox,
        loss_bbox_giou=cfg.loss_bbox_giou,
        loss_smooth=cfg.loss_smooth,
        loss_card=cfg.loss_card,
        loss_len=cfg.loss_len,
        loss_dir=cfg.loss_dir,
        aux_weight=cfg.aux_weight,
    )

    criterion.to(device)

    model = DetrPolylineFromEmbeddings(
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
    ).to(device)

    if cfg.mode == "infer":
        if cfg.infer_ckpt is None:
            raise ValueError("For mode='infer', cfg.infer_ckpt must be set.")
        load_checkpoint_flexible(model, cfg.infer_ckpt, key_candidates=["model"])

        if cfg.infer_embed_dir is not None:
            infer_ds = DetrPolylineEmbDataset(
                embed_dir=cfg.infer_embed_dir, num_points=cfg.num_points, normalize=True
            )
        elif cfg.infer_split == "train":
            infer_ds = train_ds
        elif cfg.infer_split == "val":
            infer_ds = val_ds
        elif cfg.infer_split == "all":
            infer_ds = dataset
        else:
            raise ValueError("cfg.infer_split must be 'train', 'val', or 'all'.")

        infer_loader = DataLoader(
            infer_ds,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            collate_fn=detr_polyline_collate_fn,
        )
        infer_model(infer_loader, model, device, cfg)
        return

    if cfg.mode != "train":
        raise ValueError("cfg.mode must be 'train' or 'infer'.")

    if cfg.resume_ckpt is not None:
        load_checkpoint_flexible(model, cfg.resume_ckpt, key_candidates=["model"])

    tb_dir = cfg.save_path / f"tensorboard/{cfg.exp}"
    tb_dir.mkdir(parents=True, exist_ok=True)
    writer = tensorboard.SummaryWriter(tb_dir)
    best_val = float("inf")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.max_lr, weight_decay=cfg.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.n_epochs, eta_min=1e-6
    )

    for epoch in tqdm(range(1, cfg.n_epochs + 1), disable=cfg.disable_tqdm):
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
            torch.save(ckpt, cfg.save_path / f"best_{cfg.exp}.pt")
            print(f"Saved best: {best_val:.4f} at epoch {epoch}")
        if epoch % cfg.save_every == 0:
            torch.save(ckpt, cfg.save_path / f"{cfg.exp}_{epoch}.pt")
        # scheduler.step()

    torch.save(ckpt, cfg.save_path / f"{cfg.exp}.pt")
    print(f"Saved final model: {best_val:.4f} at epoch {epoch}")


if __name__ == "__main__":
    cfg = dict(
        mode="infer",  # "train" or "infer"
        exp="detr_polyline_12",
        save_path=Path("/home/fatemeh/Downloads/hedge/results/training"),
        embed_dir=Path(
            "/home/fatemeh/Downloads/hedge/results/test_256_dino256/embs_polylines"
        ),
        # save_path=Path("/home/fkarimineja/exps/hedge"),
        # embed_dir=Path("/home/fkarimineja/data/hedge/test_256_None/embs_polylines"),
        num_points=20,
        num_polylines=100,  # decoder point queries = num_polylines * num_points
        num_classes=1,
        grid_size=(16, 16),
        # base model
        d_model=256,
        nhead=8,
        num_encoder_layers=4,
        num_decoder_layers=4,
        dim_feedforward=1024,
        dropout=0.01,  # default 0.1
        aux_loss=True,
        query_embed_mode="detr",  # "detr" or "legacy"
        eos_coef=0.05,
        # criterion and matcher
        class_cost=1.0,
        poly_cost=5.0,
        bbox_cost=2.0,
        loss_poly=5.0,
        loss_bbox=2.0,
        loss_bbox_giou=2.0,
        loss_smooth=0.05,
        loss_card=0.5,
        loss_len=1.0,
        loss_dir=0.5,
        aux_weight=0.5,
        # optimizer / training
        n_epochs=1,  # 300, 500, 3000
        batch_size=256,  # 4x256=1024 (dino256), 5x256=1280 (dino224)
        num_workers=23,  # 17
        max_lr=1e-4,  # 3e-5, 1e-4, # 3e-4,
        weight_decay=1e-2,
        disable_tqdm=True,
        save_every=3000,
        seed=42,
        # data splits (train/val, optional)
        save_splits=True,
        split_dir=Path("/home/fatemeh/Downloads/hedge/results/test_256_dino256"),
        # checkpoints
        resume_ckpt=None,
        # inference
        infer_ckpt=Path(
            "/home/fatemeh/Downloads/hedge/results/training/detr_polyline_12.pt"
        ),
        infer_embed_dir=None,
        infer_split="val",  # "train", "val", or "all"
        infer_out_dir=Path(
            "/home/fatemeh/Downloads/hedge/results/test_256_dino256/inference"
        ),
        infer_score_thresh=0.0,
        infer_topk=20,
    )
    main(OmegaConf.create(cfg))
