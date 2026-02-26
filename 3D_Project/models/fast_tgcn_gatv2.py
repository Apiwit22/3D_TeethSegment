# models/fast_tgcn_gatv2.py
from __future__ import annotations

from typing import Any, Dict, Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_geometric.nn import GATv2Conv
except Exception as e:
    raise ImportError(
        "FastTGCN_GATv2 requires PyTorch Geometric (torch_geometric).\n"
        "Install torch_geometric matching your torch/cuda build.\n"
        f"Original error: {e}"
    )


def _make_norm(norm_type: str, dim: int) -> nn.Module:
    nt = (norm_type or "bn").lower()
    if nt in ("bn", "batch", "batchnorm", "batch_norm"):
        return nn.BatchNorm1d(dim)
    if nt in ("ln", "layer", "layernorm", "layer_norm"):
        return nn.LayerNorm(dim)
    if nt in ("none", "identity", "id"):
        return nn.Identity()
    raise ValueError(f"Unknown norm_type='{norm_type}'. Use 'bn', 'ln', or 'none'.")


class _PointMLP(nn.Module):
    """Pointwise MLP block (Conv1d 1x1 equivalent) implemented as Linear on last dim."""

    def __init__(self, dim: int, dropout: float = 0.0, norm_type: str = "bn"):
        super().__init__()
        self.lin = nn.Linear(dim, dim)
        self.norm = _make_norm(norm_type, dim)
        self.dropout = float(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N,D)
        h = self.lin(x)
        h = self.norm(h)
        h = F.relu(h, inplace=True)
        h = F.dropout(h, p=self.dropout, training=self.training)
        return h


class FastTGCN_GATv2(nn.Module):
    """
    Fast-TGCN-like two-branch network (GATv2 on normal branch).

    Expected batch dict (from your collate_graph):
      - x_c: (B,F,12), x_n: (B,F,12)  [preferred]
      - or x: (B,F,24) fallback split
      - edge_index: list[Tensor(2,E)] length B
      - F_used or F_used_t to slice valid faces (exclude pad)

    Output:
      - logits: (B,F,num_classes)  (pad region remains zeros)
    """

    def __init__(
        self,
        num_classes: int = 17,
        coord_in: int = 12,
        normal_in: int = 12,
        hidden: int = 128,
        coord_layers: int = 2,
        gnn_layers: int = 4,
        dropout: float = 0.2,          # ✅ recommended default (was 0.3)
        # norm
        norm_type: str = "bn",         # "bn" or "ln" (LayerNorm often more stable in GNN)
        # GATv2 params
        heads: int = 4,
        attn_dropout: float = 0.0,     # ✅ recommended default (was 0.1)
        gat_add_self_loops: bool = False,
        # fuse
        fuse_hidden: Optional[int] = None,
    ):
        super().__init__()
        self.num_classes = int(num_classes)
        self.coord_in = int(coord_in)
        self.normal_in = int(normal_in)
        self.hidden = int(hidden)
        self.coord_layers = int(coord_layers)
        self.gnn_layers = int(gnn_layers)
        self.dropout = float(dropout)

        self.norm_type = str(norm_type)
        self.heads = int(heads)
        self.attn_dropout = float(attn_dropout)
        self.gat_add_self_loops = bool(gat_add_self_loops)

        self.fuse_hidden = int(fuse_hidden) if fuse_hidden is not None else self.hidden

        # -------------------------
        # Coord branch (lightweight)
        # -------------------------
        self.coord_in_lin = nn.Linear(self.coord_in, self.hidden)
        self.coord_norm0 = _make_norm(self.norm_type, self.hidden)
        self.coord_blocks = nn.ModuleList(
            [_PointMLP(self.hidden, dropout=self.dropout, norm_type=self.norm_type) for _ in range(self.coord_layers)]
        )

        # -------------------------
        # Normal branch (GATv2)
        # -------------------------
        self.norm_in_lin = nn.Linear(self.normal_in, self.hidden)
        self.norm_norm0 = _make_norm(self.norm_type, self.hidden)

        self.gconvs = nn.ModuleList()
        self.gnorms = nn.ModuleList()
        for _ in range(self.gnn_layers):
            # concat=False => output dim = out_channels (hidden) regardless heads
            self.gconvs.append(
                GATv2Conv(
                    in_channels=self.hidden,
                    out_channels=self.hidden,
                    heads=self.heads,
                    concat=False,
                    dropout=self.attn_dropout,
                    add_self_loops=self.gat_add_self_loops,
                )
            )
            self.gnorms.append(_make_norm(self.norm_type, self.hidden))

        # -------------------------
        # Fuse + head
        # -------------------------
        self.fuse = nn.Sequential(
            nn.Linear(self.hidden * 2, self.fuse_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout),
        )
        self.head = nn.Sequential(
            nn.Linear(self.fuse_hidden, self.fuse_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout),
            nn.Linear(self.fuse_hidden, self.num_classes),
        )

    def _coord_forward(self, x_c: torch.Tensor) -> torch.Tensor:
        # x_c: (N, coord_in)
        h = self.coord_in_lin(x_c)
        h = self.coord_norm0(h)
        h = F.relu(h, inplace=True)
        h = F.dropout(h, p=self.dropout, training=self.training)

        for blk in self.coord_blocks:
            h0 = h
            h = blk(h)
            h = h + h0
        return h  # (N, hidden)

    def _norm_forward(self, x_n: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        # x_n: (N, normal_in)
        h = self.norm_in_lin(x_n)
        h = self.norm_norm0(h)
        h = F.relu(h, inplace=True)
        h = F.dropout(h, p=self.dropout, training=self.training)

        if edge_index.numel() == 0:
            return h

        for conv, norm in zip(self.gconvs, self.gnorms):
            h0 = h
            h = conv(h, edge_index)
            h = norm(h)
            h = F.relu(h, inplace=True)
            h = F.dropout(h, p=self.dropout, training=self.training)
            h = h + h0
        return h  # (N, hidden)

    def forward(self, batch: Dict[str, Any]) -> torch.Tensor:
        if not isinstance(batch, dict):
            raise TypeError(f"FastTGCN_GATv2 expects batch dict, got: {type(batch)}")

        x_c = batch.get("x_c", None)
        x_n = batch.get("x_n", None)
        if x_c is None or x_n is None:
            x = batch["x"]
            if x.dim() != 3 or x.size(-1) != 24:
                raise ValueError("Need batch['x_c'] and batch['x_n'] or batch['x'] with last dim=24")
            x_c = x[..., :12]
            x_n = x[..., 12:]

        if x_c.dim() != 3 or x_n.dim() != 3:
            raise ValueError(f"x_c/x_n must be (B,F,D). Got {tuple(x_c.shape)} / {tuple(x_n.shape)}")
        if x_c.size(-1) != self.coord_in or x_n.size(-1) != self.normal_in:
            raise ValueError(
                f"Channel mismatch: x_c last={x_c.size(-1)} expected={self.coord_in}, "
                f"x_n last={x_n.size(-1)} expected={self.normal_in}"
            )

        edge_index_list = batch.get("edge_index", None)
        if not isinstance(edge_index_list, list):
            raise ValueError("batch['edge_index'] must be list[Tensor(2,E)] length B")

        F_used = batch.get("F_used_t", batch.get("F_used", None))
        if F_used is None:
            raise KeyError("batch must contain 'F_used' or 'F_used_t'")
        if torch.is_tensor(F_used):
            F_used_list = [int(v) for v in F_used.detach().cpu().tolist()]
        else:
            F_used_list = [int(v) for v in F_used]

        B, Fmax, _ = x_c.shape
        if len(edge_index_list) != B:
            raise ValueError(f"edge_index list length {len(edge_index_list)} != batch size {B}")

        out = x_c.new_zeros((B, Fmax, self.num_classes))

        for b in range(B):
            n = int(F_used_list[b])
            if n <= 0:
                continue

            xc = x_c[b, :n, :]
            xn = x_n[b, :n, :]

            ei = edge_index_list[b]
            if not torch.is_tensor(ei) or ei.ndim != 2 or ei.shape[0] != 2:
                raise ValueError(
                    f"edge_index[{b}] must be Tensor shape (2,E), got {type(ei)} shape={getattr(ei,'shape',None)}"
                )
            ei = ei.to(xc.device, non_blocking=True).long()

            hc = self._coord_forward(xc)
            hn = self._norm_forward(xn, ei)

            h = torch.cat([hc, hn], dim=-1)
            h = self.fuse(h)
            logits = self.head(h)

            out[b, :n, :] = logits

        return out