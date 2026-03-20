# models/tsgcnet2.py
from __future__ import annotations

from typing import Dict, Any, Optional, Tuple, List, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Utils
# ============================================================
def sanitize_nbr(nbr: torch.Tensor, num_nodes: int) -> torch.Tensor:
    if nbr is None or nbr.numel() == 0 or num_nodes <= 0:
        return nbr
    if nbr.ndim != 2:
        raise ValueError(f"sanitize_nbr expects (N,K), got {tuple(nbr.shape)}")

    N, K = nbr.shape
    nbr2 = nbr.clone()
    neg = nbr2 < 0
    nbr2 = torch.clamp(nbr2, 0, max(num_nodes - 1, 0))
    if neg.any():
        self_idx = torch.arange(N, device=nbr2.device).unsqueeze(1).expand(N, K)
        nbr2[neg] = self_idx[neg]
    nbr2 = torch.clamp(nbr2, 0, max(num_nodes - 1, 0))
    return nbr2


# ============================================================
# Basic layers
# ============================================================
class LNLinear(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        *,
        act: bool = True,
        negative_slope: float = 0.2,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.fc = nn.Linear(int(in_ch), int(out_ch))
        self.ln = nn.LayerNorm(int(out_ch))
        self.act = nn.LeakyReLU(negative_slope=float(negative_slope), inplace=True) if act else nn.Identity()
        p = float(dropout)
        self.drop = nn.Dropout(p=p) if p > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc(x)
        x = self.ln(x)
        x = self.act(x)
        x = self.drop(x)
        return x


class MLP(nn.Module):
    def __init__(
        self,
        in_ch: int,
        hidden: int,
        out_ch: int,
        *,
        negative_slope: float = 0.2,
        dropout: float = 0.0,
        act_last: bool = True,
    ):
        super().__init__()
        self.fc1 = LNLinear(
            in_ch, hidden,
            act=True,
            negative_slope=negative_slope,
            dropout=dropout,
        )
        self.fc2 = LNLinear(
            hidden, out_ch,
            act=act_last,
            negative_slope=negative_slope,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.fc1(x))


class ResMLP(nn.Module):
    def __init__(
        self,
        ch: int,
        *,
        hidden_ratio: float = 2.0,
        negative_slope: float = 0.2,
        dropout: float = 0.0,
    ):
        super().__init__()
        ch = int(ch)
        hidden = max(ch, int(round(ch * float(hidden_ratio))))
        self.pre = nn.LayerNorm(ch)
        self.fc1 = nn.Linear(ch, hidden)
        self.act = nn.LeakyReLU(negative_slope=float(negative_slope), inplace=True)
        p = float(dropout)
        self.drop = nn.Dropout(p=p) if p > 0 else nn.Identity()
        self.fc2 = nn.Linear(hidden, ch)
        self.post = nn.LayerNorm(ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.pre(x)
        h = self.fc1(h)
        h = self.act(h)
        h = self.drop(h)
        h = self.fc2(h)
        y = x + h
        y = self.post(y)
        return y


# ============================================================
# C-stream: scalar attention
# ============================================================
class CStreamAttnBlock(nn.Module):
    """
    Conceptually follows paper-like C-stream:
      f_hat_ij = MLP_calib([f_i, f_j])
      e_ij     = MLP_att([f_i - f_j, f_j]) -> scalar
      alpha    = softmax(e_ij over neighbors)
      f_out    = sum_j alpha_ij * f_hat_ij
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        *,
        hidden: Optional[int] = None,
        negative_slope: float = 0.2,
        dropout: float = 0.0,
        agg_chunk: int = 1024,
        k_chunk: int = 8,
        residual: bool = True,
    ):
        super().__init__()
        self.in_ch = int(in_ch)
        self.out_ch = int(out_ch)
        self.agg_chunk = int(agg_chunk)
        self.k_chunk = int(k_chunk)
        self.residual = bool(residual) and (self.in_ch == self.out_ch)

        h = int(hidden) if hidden is not None else max(64, self.out_ch)

        self.mlp_calib = MLP(
            2 * self.in_ch, h, self.out_ch,
            negative_slope=negative_slope,
            dropout=dropout,
            act_last=True,
        )
        self.mlp_att = MLP(
            2 * self.in_ch, h, 1,
            negative_slope=negative_slope,
            dropout=0.0,
            act_last=False,
        )

    def forward(self, f: torch.Tensor, nbr: torch.Tensor) -> torch.Tensor:
        if f.ndim != 2:
            raise ValueError(f"CStreamAttnBlock expects (N,C), got {tuple(f.shape)}")
        N, Cin = f.shape
        if Cin != self.in_ch:
            raise ValueError(f"Cin mismatch: got {Cin}, expected {self.in_ch}")
        if nbr.ndim != 2 or nbr.shape[0] != N:
            raise ValueError(f"nbr must be (N,K), got {tuple(nbr.shape)}")

        K = int(nbr.shape[1])
        if K == 0:
            out = torch.zeros((N, self.out_ch), dtype=f.dtype, device=f.device)
            return out + f if self.residual else out

        nbr = sanitize_nbr(nbr, N)
        out = torch.empty((N, self.out_ch), dtype=f.dtype, device=f.device)
        kc = max(1, int(self.k_chunk))

        for s in range(0, N, self.agg_chunk):
            e = min(N, s + self.agg_chunk)
            idx_all = nbr[s:e]   # (Q,K)
            fi = f[s:e]          # (Q,C)
            Q = int(fi.shape[0])

            max_e = torch.full((Q, 1), -float("inf"), dtype=f.dtype, device=f.device)
            sum_exp = torch.zeros((Q, 1), dtype=f.dtype, device=f.device)

            scores_cache: List[torch.Tensor] = []
            idx_cache: List[torch.Tensor] = []

            # pass 1: scores
            for ks in range(0, K, kc):
                ke = min(K, ks + kc)
                idx = idx_all[:, ks:ke]                   # (Q,kc)
                fj = f[idx]                               # (Q,kc,C)
                fi_rep = fi.unsqueeze(1).expand(-1, ke - ks, -1)

                delta = fi_rep - fj
                att_in = torch.cat([delta, fj], dim=-1).reshape(-1, 2 * Cin)
                e_ij = self.mlp_att(att_in).reshape(Q, ke - ks, 1)

                max_e = torch.maximum(max_e, e_ij.max(dim=1).values)
                scores_cache.append(e_ij)
                idx_cache.append(idx)

            # pass 2: stable softmax denominator
            for e_ij in scores_cache:
                exp_ij = torch.exp(e_ij - max_e.unsqueeze(1))
                sum_exp = sum_exp + exp_ij.sum(dim=1)

            denom = sum_exp.clamp_min(1e-12)

            # pass 3: weighted aggregation
            acc = torch.zeros((Q, self.out_ch), dtype=f.dtype, device=f.device)
            for e_ij, idx in zip(scores_cache, idx_cache):
                fj = f[idx]
                fi_rep = fi.unsqueeze(1).expand(-1, idx.shape[1], -1)

                calib_in = torch.cat([fi_rep, fj], dim=-1).reshape(-1, 2 * Cin)
                f_hat = self.mlp_calib(calib_in).reshape(Q, idx.shape[1], self.out_ch)

                alpha = torch.exp(e_ij - max_e.unsqueeze(1)) / denom.unsqueeze(1)
                acc = acc + (alpha * f_hat).sum(dim=1)

            out[s:e] = acc

        if self.residual:
            out = out + f
        return out


# ============================================================
# N-stream: max pooling
# ============================================================
class NStreamMaxBlock(nn.Module):
    """
    Paper-like:
      f_hat_ij = MLP_calib([f_i, f_j])
      f_out    = max_j f_hat_ij
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        *,
        hidden: Optional[int] = None,
        negative_slope: float = 0.2,
        dropout: float = 0.0,
        agg_chunk: int = 1024,
        k_chunk: int = 8,
        residual: bool = True,
    ):
        super().__init__()
        self.in_ch = int(in_ch)
        self.out_ch = int(out_ch)
        self.agg_chunk = int(agg_chunk)
        self.k_chunk = int(k_chunk)
        self.residual = bool(residual) and (self.in_ch == self.out_ch)

        h = int(hidden) if hidden is not None else max(64, self.out_ch)
        self.mlp_calib = MLP(
            2 * self.in_ch, h, self.out_ch,
            negative_slope=negative_slope,
            dropout=dropout,
            act_last=True,
        )

    def forward(self, f: torch.Tensor, nbr: torch.Tensor) -> torch.Tensor:
        if f.ndim != 2:
            raise ValueError(f"NStreamMaxBlock expects (N,C), got {tuple(f.shape)}")
        N, Cin = f.shape
        if Cin != self.in_ch:
            raise ValueError(f"Cin mismatch: got {Cin}, expected {self.in_ch}")
        if nbr.ndim != 2 or nbr.shape[0] != N:
            raise ValueError(f"nbr must be (N,K), got {tuple(nbr.shape)}")

        K = int(nbr.shape[1])
        if K == 0:
            out = torch.zeros((N, self.out_ch), dtype=f.dtype, device=f.device)
            return out + f if self.residual else out

        nbr = sanitize_nbr(nbr, N)
        out = torch.empty((N, self.out_ch), dtype=f.dtype, device=f.device)
        kc = max(1, int(self.k_chunk))

        for s in range(0, N, self.agg_chunk):
            e = min(N, s + self.agg_chunk)
            idx_all = nbr[s:e]
            fi = f[s:e]
            Q = int(fi.shape[0])

            best = torch.full((Q, self.out_ch), -float("inf"), dtype=f.dtype, device=f.device)

            for ks in range(0, K, kc):
                ke = min(K, ks + kc)
                idx = idx_all[:, ks:ke]
                fj = f[idx]
                fi_rep = fi.unsqueeze(1).expand(-1, ke - ks, -1)

                calib_in = torch.cat([fi_rep, fj], dim=-1).reshape(-1, 2 * Cin)
                f_hat = self.mlp_calib(calib_in).reshape(Q, ke - ks, self.out_ch)

                best = torch.maximum(best, f_hat.max(dim=1).values)

            out[s:e] = best

        if self.residual:
            out = out + f
        return out


# ============================================================
# Fusion helpers
# ============================================================
class StreamGatedFusion(nn.Module):
    """
    Learn per-channel fusion between coordinate-stream and normal-stream.
    """

    def __init__(
        self,
        ch: int,
        *,
        negative_slope: float = 0.2,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.ch = int(ch)
        self.mix = nn.Sequential(
            LNLinear(2 * self.ch, self.ch, act=True, negative_slope=negative_slope, dropout=dropout),
            LNLinear(self.ch, self.ch, act=True, negative_slope=negative_slope, dropout=dropout),
        )
        self.gate = nn.Sequential(
            nn.LayerNorm(2 * self.ch),
            nn.Linear(2 * self.ch, self.ch, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, fc: torch.Tensor, fn: torch.Tensor) -> torch.Tensor:
        cat = torch.cat([fc, fn], dim=-1)   # (N,2C)
        m = self.mix(cat)                   # (N,C)
        g = self.gate(cat)                  # (N,C)
        fused = g * fc + (1.0 - g) * fn
        return torch.cat([fc, fn, fused, m], dim=-1)  # (N,4C)


# ============================================================
# TSGCNet2
# ============================================================
class TSGCNet2(nn.Module):
    """
    Improved TSGCNet for your pipeline:
      - input: batch['x_c'], batch['x_n'], batch['nbr']
      - output: (B,F,C)
      - keeps train/test interface unchanged
    """

    def __init__(
        self,
        *,
        num_classes: int = 17,
        k: int = 32,  # compatibility only; real nbr comes from dataloader
        dims: Tuple[int, int, int] = (64, 128, 256),
        fusion_dim: int = 512,
        negative_slope: float = 0.2,
        dropout: float = 0.08,
        agg_chunk: int = 2048,
        k_chunk: int = 8,
        residual: bool = True,
        global_context: bool = True,
        global_context_mode: str = "meanmax",   # "meanmax" or "mean"
        fusion_gate: bool = True,
        fusion_post_residual: bool = True,
    ):
        super().__init__()
        self.num_classes = int(num_classes)
        self.k = int(k)
        self.global_context = bool(global_context)
        self.global_context_mode = str(global_context_mode).lower().strip()
        self.fusion_gate = bool(fusion_gate)

        d1, d2, d3 = (int(d) for d in dims)

        # C-stream
        self.c1 = CStreamAttnBlock(
            12, d1,
            negative_slope=negative_slope,
            dropout=dropout,
            agg_chunk=agg_chunk,
            k_chunk=k_chunk,
            residual=False,
        )
        self.c2 = CStreamAttnBlock(
            d1, d2,
            negative_slope=negative_slope,
            dropout=dropout,
            agg_chunk=agg_chunk,
            k_chunk=k_chunk,
            residual=residual,
        )
        self.c3 = CStreamAttnBlock(
            d2, d3,
            negative_slope=negative_slope,
            dropout=dropout,
            agg_chunk=agg_chunk,
            k_chunk=k_chunk,
            residual=residual,
        )

        # N-stream
        self.n1 = NStreamMaxBlock(
            12, d1,
            negative_slope=negative_slope,
            dropout=dropout,
            agg_chunk=agg_chunk,
            k_chunk=k_chunk,
            residual=False,
        )
        self.n2 = NStreamMaxBlock(
            d1, d2,
            negative_slope=negative_slope,
            dropout=dropout,
            agg_chunk=agg_chunk,
            k_chunk=k_chunk,
            residual=residual,
        )
        self.n3 = NStreamMaxBlock(
            d2, d3,
            negative_slope=negative_slope,
            dropout=dropout,
            agg_chunk=agg_chunk,
            k_chunk=k_chunk,
            residual=residual,
        )

        hier_dim = d1 + d2 + d3

        self.mlp_c = nn.Sequential(
            LNLinear(hier_dim, fusion_dim, act=True, negative_slope=negative_slope, dropout=dropout),
            LNLinear(fusion_dim, fusion_dim, act=True, negative_slope=negative_slope, dropout=dropout),
        )
        self.mlp_n = nn.Sequential(
            LNLinear(hier_dim, fusion_dim, act=True, negative_slope=negative_slope, dropout=dropout),
            LNLinear(fusion_dim, fusion_dim, act=True, negative_slope=negative_slope, dropout=dropout),
        )

        if self.fusion_gate:
            self.fusion = StreamGatedFusion(
                fusion_dim,
                negative_slope=negative_slope,
                dropout=dropout,
            )
            local_dim = 4 * fusion_dim
        else:
            self.fusion = nn.Identity()
            local_dim = 2 * fusion_dim

        self.fusion_post = ResMLP(
            local_dim,
            hidden_ratio=2.0,
            negative_slope=negative_slope,
            dropout=dropout if fusion_post_residual else 0.0,
        ) if fusion_post_residual else nn.Identity()

        if self.global_context:
            if self.global_context_mode in ("meanmax", "mean_max", "mean+max"):
                in_head = local_dim * 3
            else:
                in_head = local_dim * 2
        else:
            in_head = local_dim

        self.head = nn.Sequential(
            LNLinear(in_head, 512, act=True, negative_slope=negative_slope, dropout=dropout),
            LNLinear(512, 256, act=True, negative_slope=negative_slope, dropout=dropout),
            LNLinear(256, 128, act=True, negative_slope=negative_slope, dropout=dropout),
            nn.Linear(128, self.num_classes),
        )

    def _forward_single(self, x_c: torch.Tensor, x_n: torch.Tensor, nbr: torch.Tensor) -> torch.Tensor:
        N = int(x_c.shape[0])
        nbr = sanitize_nbr(nbr, N)

        c1 = self.c1(x_c, nbr)
        n1 = self.n1(x_n, nbr)

        c2 = self.c2(c1, nbr)
        n2 = self.n2(n1, nbr)

        c3 = self.c3(c2, nbr)
        n3 = self.n3(n2, nbr)

        hc = torch.cat([c1, c2, c3], dim=1)   # (N, d1+d2+d3)
        hn = torch.cat([n1, n2, n3], dim=1)

        fc = self.mlp_c(hc)                   # (N,F)
        fn = self.mlp_n(hn)                   # (N,F)

        if self.fusion_gate:
            z_local = self.fusion(fc, fn)     # (N,4F)
        else:
            z_local = torch.cat([fc, fn], dim=1)

        z_local = self.fusion_post(z_local)

        if self.global_context:
            g_mean = z_local.mean(dim=0, keepdim=True).expand_as(z_local)

            if self.global_context_mode in ("meanmax", "mean_max", "mean+max"):
                g_max = z_local.max(dim=0, keepdim=True).values.expand_as(z_local)
                z = torch.cat([z_local, g_mean, g_max], dim=1)
            else:
                z = torch.cat([z_local, g_mean], dim=1)
        else:
            z = z_local

        logits = self.head(z)
        return logits

    def forward(
        self,
        batch: Optional[Dict[str, Any]] = None,
        *,
        x_c: Optional[torch.Tensor] = None,
        x_n: Optional[torch.Tensor] = None,
        nbr: Optional[torch.Tensor] = None,
        F_used: Optional[Union[List[int], torch.Tensor]] = None,
        valid_face: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if batch is not None:
            if x_c is None:
                x_c = batch.get("x_c", None)
            if x_n is None:
                x_n = batch.get("x_n", None)
            if nbr is None:
                nbr = batch.get("nbr", None)
            if F_used is None:
                F_used = batch.get("F_used", None)
            if valid_face is None:
                valid_face = batch.get("valid_face", None)

        if x_c is None or x_n is None:
            raise KeyError("TSGCNet2 ต้องการ x_c และ x_n (face_feature=twostream24).")
        if nbr is None:
            raise KeyError("TSGCNet2 ต้องการ nbr (graph_fortsgcnet / return_nbr).")

        if x_c.ndim != 3 or x_n.ndim != 3:
            raise ValueError(f"x_c/x_n ต้องเป็น (B,F,12), got x_c={tuple(x_c.shape)} x_n={tuple(x_n.shape)}")
        if nbr.ndim != 3:
            raise ValueError(f"nbr ต้องเป็น (B,F,K), got {tuple(nbr.shape)}")

        B, Fmax, Dc = x_c.shape
        if Dc != 12 or x_n.shape[-1] != 12:
            raise ValueError("x_c และ x_n ต้องมีมิติสุดท้าย = 12")
        if nbr.shape[0] != B or nbr.shape[1] != Fmax:
            raise ValueError(f"nbr shape ต้องเป็น (B,F,K), got {tuple(nbr.shape)} vs F={Fmax}")

        if F_used is None:
            f_used_list = [int(Fmax)] * int(B)
        elif isinstance(F_used, list):
            f_used_list = [int(v) for v in F_used]
        elif torch.is_tensor(F_used):
            f_used_list = [int(v) for v in F_used.detach().cpu().tolist()]
        else:
            raise TypeError(f"F_used must be list[int] or Tensor, got {type(F_used)}")

        _ = valid_face  # compatibility

        device = x_c.device
        logits_out = torch.zeros((B, Fmax, self.num_classes), dtype=x_c.dtype, device=device)

        for b in range(int(B)):
            Nu = max(0, min(int(Fmax), int(f_used_list[b])))
            if Nu <= 0:
                continue

            xc = x_c[b, :Nu].contiguous()
            xn = x_n[b, :Nu].contiguous()
            nb = nbr[b, :Nu].contiguous().long()

            logits_out[b, :Nu] = self._forward_single(xc, xn, nb)

        return logits_out

    def compute_loss(
        self,
        logits: torch.Tensor,
        y: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        *,
        ignore_index: int = -1,
    ) -> torch.Tensor:
        B, Fmax, C = logits.shape
        logits2 = logits.reshape(B * Fmax, C)
        y2 = y.reshape(B * Fmax)

        if mask is None:
            return F.cross_entropy(logits2, y2, ignore_index=int(ignore_index))

        m2 = mask.reshape(B * Fmax).bool()
        if m2.sum().item() == 0:
            return logits2.sum() * 0.0

        return F.cross_entropy(logits2[m2], y2[m2], ignore_index=int(ignore_index))