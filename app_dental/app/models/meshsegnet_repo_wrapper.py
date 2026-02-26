# models/meshsegnet_repo_wrapper.py
# MeshSegNet wrapper (batch dict) + kNN sparse adjacency
# - Builds sparse row-stochastic adjacency using torch-cluster knn_graph
# - AMP-safe: runs adjacency build + sparse ops in fp32 where needed
# - Works with MeshSegNet that supports sparse via _spmm_batch in models/meshsegnet.py

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn as nn

from models.meshsegnet import MeshSegNet

try:
    # torch-cluster must match your torch + CUDA build
    from torch_cluster import knn_graph
except Exception:
    knn_graph = None


# ============================================================
# sparse helpers
# ============================================================
def _row_normalize_sparse(A: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    A: sparse COO (N,N) with nonnegative weights
    return: sparse COO row-stochastic (each row sums to 1)
    """
    A = A.coalesce()
    idx = A.indices()  # (2,E) rows=idx[0], cols=idx[1]
    val = A.values()   # (E,)
    N = int(A.size(0))

    # row-degree
    deg = torch.zeros((N,), device=val.device, dtype=val.dtype)
    deg.index_add_(0, idx[0], val)

    inv = 1.0 / torch.clamp(deg, min=eps)
    val = val * inv[idx[0]]

    return torch.sparse_coo_tensor(idx, val, size=A.size(), device=A.device).coalesce()


def _build_knn_adj_sparse(
    pos_bnc: torch.Tensor,
    k: int,
    *,
    self_loop: bool = True,
    force_fp32: bool = True,
) -> List[torch.Tensor]:
    """
    pos_bnc: (B,N,3) face centers (float)
    return: list of sparse COO, each (N,N), row-stochastic
    """
    if knn_graph is None:
        raise ImportError("Need torch-cluster for kNN graph. Install torch-cluster.")

    if pos_bnc.dim() != 3 or pos_bnc.size(-1) != 3:
        raise ValueError(f"pos must be (B,N,3), got {tuple(pos_bnc.shape)}")

    B, N, _ = pos_bnc.shape
    out: List[torch.Tensor] = []

    # Build in fp32 to avoid any half-related weirdness during graph build
    pos_bnc_ = pos_bnc.float() if force_fp32 else pos_bnc

    for b in range(B):
        pos = pos_bnc_[b]  # (N,3)

        # knn_graph returns edges (2,E). For torch_cluster, edges are typically (row=target, col=source)
        # We'll define adjacency rows = dst (center), cols = src (neighbor), then row-normalize.
        edge = knn_graph(pos, k=int(k), loop=bool(self_loop))  # (2,E)

        dst = edge[0]  # center i
        src = edge[1]  # neighbor j

        idx = torch.stack([dst, src], dim=0)  # rows=dst, cols=src
        val = torch.ones((idx.size(1),), device=pos.device, dtype=torch.float32)

        A = torch.sparse_coo_tensor(idx, val, (N, N), device=pos.device).coalesce()
        A = _row_normalize_sparse(A)

        # keep adjacency in fp32 (recommended) for stable sparse.mm
        out.append(A.float())

    return out


# ============================================================
# Wrapper
# ============================================================
class MeshSegNetBatch(nn.Module):
    """
    forward_type='batch'
      forward(batch_dict) -> logits (B,N,C)

    Requires in batch:
      - x:   (B,N,in_channels)
      - pos: (B,N,3)
    Optional:
      - y:    (B,N)
      - mask: (B,N) bool
    """

    def __init__(
        self,
        num_classes: int = 17,
        in_channels: int = 15,
        knn_s: int = 16,
        knn_l: int = 64,
        with_dropout: bool = True,
        dropout_p: float = 0.5,
        # debug flags (kept but not printing huge sparse stats by default)
        debug: bool = False,
        debug_every: int = 50,
        debug_first: int = 3,
    ):
        super().__init__()
        self.num_classes = int(num_classes)
        self.in_channels = int(in_channels)

        self.knn_s = int(knn_s)
        self.knn_l = int(knn_l)

        self.debug = bool(debug)
        self.debug_every = int(debug_every)
        self.debug_first = int(debug_first)
        self._step = 0

        self.core = MeshSegNet(
            num_classes=self.num_classes,
            num_channels=self.in_channels,
            with_dropout=with_dropout,
            dropout_p=dropout_p,
        )

    @torch.no_grad()
    def _debug_print(self, x: torch.Tensor, pos: torch.Tensor, A_S, A_L, logits: torch.Tensor):
        B, N, Cin = x.shape
        print("\n================ MeshSegNet DEBUG ================")
        print(f"[step] {self._step} | B={B} N={N} inC={Cin} num_classes={self.num_classes}")
        print(f"[x]   {tuple(x.shape)} {x.dtype} {x.device} min={x.min().item():.4f} max={x.max().item():.4f}")
        print(f"[pos] {tuple(pos.shape)} {pos.dtype} {pos.device} min={pos.min().item():.4f} max={pos.max().item():.4f}")
        # sparse info
        if isinstance(A_S, list) and len(A_S) > 0 and A_S[0].is_sparse:
            nnz_s = [int(a._nnz()) for a in A_S]
            nnz_l = [int(a._nnz()) for a in A_L]
            print(f"[A_S] sparse list len={len(A_S)} nnz(min/mean/max)={min(nnz_s)}/{sum(nnz_s)/len(nnz_s):.1f}/{max(nnz_s)}")
            print(f"[A_L] sparse list len={len(A_L)} nnz(min/mean/max)={min(nnz_l)}/{sum(nnz_l)/len(nnz_l):.1f}/{max(nnz_l)}")
        print(f"[logits] {tuple(logits.shape)} {logits.dtype} finite={torch.isfinite(logits).all().item()} "
              f"min={logits.min().item():.4f} max={logits.max().item():.4f}")
        print("==================================================\n")

    def forward(self, batch: dict) -> torch.Tensor:
        self._step += 1

        x = batch["x"]      # (B,N,C)
        pos = batch["pos"]  # (B,N,3)

        if x.dim() != 3:
            raise ValueError(f"batch['x'] must be (B,N,C), got {tuple(x.shape)}")
        if pos.dim() != 3 or pos.size(-1) != 3:
            raise ValueError(f"batch['pos'] must be (B,N,3), got {tuple(pos.shape)}")
        if x.size(-1) != self.in_channels:
            raise ValueError(f"in_channels mismatch: expected {self.in_channels}, got {x.size(-1)}")
        if x.size(1) != pos.size(1):
            raise ValueError(f"N mismatch: x has N={x.size(1)} but pos has N={pos.size(1)}")

        # Build sparse kNN adjacency in fp32
        # NOTE: Even if AMP is ON globally, these graphs + sparse.mm will be forced fp32 in meshsegnet.py
        A_S = _build_knn_adj_sparse(pos, k=self.knn_s, self_loop=True, force_fp32=True)
        A_L = _build_knn_adj_sparse(pos, k=self.knn_l, self_loop=True, force_fp32=True)

        x_in = x.permute(0, 2, 1).contiguous()  # (B,C,N)
        logits = self.core(x_in, A_S, A_L)      # (B,N,C)

        if self.debug:
            if (self._step <= self.debug_first) or (self.debug_every > 0 and self._step % self.debug_every == 0):
                self._debug_print(x, pos, A_S, A_L, logits)

        return logits
