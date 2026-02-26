# models/fast_tgcn.py
from __future__ import annotations
from typing import Dict, Any, List

import torch
import torch.nn as nn
import torch.nn.functional as F


def _ensure_edge_index(edge_index: torch.Tensor, n: int) -> torch.Tensor:
    if edge_index.numel() == 0:
        return edge_index
    src, dst = edge_index[0], edge_index[1]
    m = (src >= 0) & (src < n) & (dst >= 0) & (dst < n)
    if m.all():
        return edge_index
    return edge_index[:, m]


def _mean_aggregate(x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
    """
    Implements a sparse (D^-1 A) aggregation (mean of neighbors).
    x: (N,C)
    edge_index: (2,E) directed src->dst
    """
    N, C = x.shape
    if edge_index.numel() == 0:
        return torch.zeros((N, C), device=x.device, dtype=x.dtype)

    src, dst = edge_index[0], edge_index[1]
    msg = x[src]  # (E,C)

    out = torch.zeros((N, C), device=x.device, dtype=x.dtype)
    out.index_add_(0, dst, msg)

    deg = torch.zeros((N,), device=x.device, dtype=x.dtype)
    ones = torch.ones((dst.numel(),), device=x.device, dtype=x.dtype)
    deg.index_add_(0, dst, ones)

    return out / deg.clamp_min(1.0).unsqueeze(1)


class Conv1DBlock(nn.Module):
    """
    'Conv1D' in the paper can be implemented as per-node Linear (1x1 conv).
    """
    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.0, use_bn: bool = True):
        super().__init__()
        self.fc = nn.Linear(in_ch, out_ch)
        self.bn = nn.BatchNorm1d(out_ch) if use_bn else nn.Identity()
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc(x)
        x = self.bn(x)
        x = F.relu(x, inplace=True)
        x = self.drop(x)
        return x


class GCB(nn.Module):
    """
    Graph Convolution Block (paper-style):
      GCB(Fn) = sigma( A · Conv1D(Fn) · W )
    We implement it sparsely:
      h = Conv1D(Fn)
      h = mean(A @ h)   (degree-normalized for stability)
      h = Linear(h)     (~ multiply by W)
      sigma = ReLU (+ BN, Dropout)
    """
    def __init__(self, dim: int, dropout: float = 0.0, use_bn: bool = True):
        super().__init__()
        self.pre = Conv1DBlock(dim, dim, dropout=0.0, use_bn=use_bn)  # Conv1D(Fn)
        self.lin_w = nn.Linear(dim, dim)                             # ·W
        self.bn = nn.BatchNorm1d(dim) if use_bn else nn.Identity()
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = self.pre(x)                        # (N,dim)
        h = _mean_aggregate(h, edge_index)     # A @ h (normalized)
        h = self.lin_w(h)
        h = self.bn(h)
        h = F.relu(h, inplace=True)
        h = self.drop(h)
        return h


class FastTGCN(nn.Module):
    """
    Paper-aligned Fast-TGCN:
      - Input: 24D per face-cell (x_c 12D, x_n 12D)
      - Normal branch: GCB stacks guided by adjacency (share-vertex)
      - Coord branch: only Conv1D (no graph conv)
      - Union: concat then FC classifier

    Batch expected:
      - x_c: (B,F,12) or x: (B,F,24)
      - x_n: (B,F,12)
      - edge_index: list[(2,E_i)] length B (built from F_used)
      - F_used: list[int] or tensor/int
    Output:
      - logits: (B,F,num_classes)
    """
    def __init__(
        self,
        num_classes: int = 17,
        hidden_dim: int = 128,
        num_blocks: int = 6,
        dropout: float = 0.2,
        use_bn: bool = True,
    ):
        super().__init__()
        self.num_classes = int(num_classes)
        self.hidden_dim = int(hidden_dim)
        self.num_blocks = int(num_blocks)

        # branch dims (keep them equal for concat)
        h = max(16, self.hidden_dim // 2)

        # Coord branch: F'_c = Conv1D(F_c)
        self.coord_conv = Conv1DBlock(12, h, dropout=dropout, use_bn=use_bn)

        # Normal branch:
        # start with Conv1D to h, then GCB stacks on graph
        self.norm_in = Conv1DBlock(12, h, dropout=dropout, use_bn=use_bn)
        self.gcbs = nn.ModuleList([GCB(h, dropout=dropout, use_bn=use_bn) for _ in range(self.num_blocks)])

        # Head on concatenated feature (h_n ⊕ h_c) -> num_classes
        self.head = nn.Sequential(
            nn.Linear(h * 2, self.hidden_dim),
            nn.BatchNorm1d(self.hidden_dim) if use_bn else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, self.num_classes),
        )

    def forward(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        x = batch.get("x", None)
        x_c = batch.get("x_c", None)
        x_n = batch.get("x_n", None)

        if x_c is None or x_n is None:
            if x is None:
                raise KeyError("FastTGCN requires batch['x'] (24D) or batch['x_c'] & batch['x_n'] (12D each).")
            if x.shape[-1] != 24:
                raise ValueError(f"Expected x last-dim=24, got {tuple(x.shape)}")
            x_c = x[..., :12]
            x_n = x[..., 12:]

        edge_list = batch.get("edge_index", None)
        if edge_list is None or not isinstance(edge_list, list):
            raise KeyError("FastTGCN requires batch['edge_index'] as list of tensors per sample.")

        B, F = x_c.shape[0], x_c.shape[1]

        F_used_any = batch.get("F_used", None)
        if isinstance(F_used_any, list):
            F_used_list = [int(v) for v in F_used_any]
        elif torch.is_tensor(F_used_any) and F_used_any.ndim == 1:
            F_used_list = [int(v.item()) for v in F_used_any]
        elif isinstance(F_used_any, int):
            F_used_list = [int(F_used_any)] * B
        else:
            F_used_list = [F] * B

        logits_all: List[torch.Tensor] = []

        for i in range(B):
            n = max(1, min(int(F_used_list[i]), F))

            xi_c = x_c[i, :n]  # (n,12)
            xi_n = x_n[i, :n]  # (n,12)

            ei = edge_list[i]
            if not torch.is_tensor(ei):
                raise ValueError("edge_index entries must be torch.Tensor")
            ei = _ensure_edge_index(ei.long(), n)

            # Coord branch (no graph)
            hc = self.coord_conv(xi_c)  # (n,h)

            # Normal branch (graph)
            hn = self.norm_in(xi_n)     # (n,h)
            for gcb in self.gcbs:
                hn = gcb(hn, ei)        # (n,h)

            # Union + head
            hu = torch.cat([hn, hc], dim=1)     # (n,2h)
            li = self.head(hu)                  # (n,C)

            # pad back to fixed F
            pad = torch.zeros((F, self.num_classes), device=li.device, dtype=li.dtype)
            pad[:n] = li
            logits_all.append(pad)

        logits = torch.stack(logits_all, dim=0)  # (B,F,C)
        return {"logits": logits}
