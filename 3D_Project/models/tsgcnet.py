# src/models/tsgcnet.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple, List, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Utility: MLP (Linear + BN + LeakyReLU)
# ============================================================
class MLP(nn.Module):
    """
    MLP แบบง่าย ๆ สำหรับข้อมูลรูป (N, C)
    - ใช้ Linear + BatchNorm1d + LeakyReLU
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        *,
        act: bool = True,
        bn: bool = True,
        negative_slope: float = 0.2,
        dropout: float = 0.0,
    ):
        super().__init__()
        layers: List[nn.Module] = []
        layers.append(nn.Linear(in_ch, out_ch, bias=not bn))
        if bn:
            layers.append(nn.BatchNorm1d(out_ch))
        if act:
            layers.append(nn.LeakyReLU(negative_slope=negative_slope, inplace=True))
        if dropout and float(dropout) > 0:
            layers.append(nn.Dropout(p=float(dropout)))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, C) เท่านั้น
        return self.net(x)


# ============================================================
# KNN builder (chunked exact KNN by squared Euclidean distance)
# ============================================================
@torch.no_grad()
def knn_indices_chunked(
    x: torch.Tensor,
    k: int,
    *,
    chunk_size: int = 1024,
    exclude_self: bool = True,
) -> torch.Tensor:
    """
    หา KNN แบบ exact ด้วยการคำนวณระยะเป็นก้อน ๆ (กันเมมพัง)
    - x: (N, D) float tensor
    - return: idx (N, k) long

    หมายเหตุ:
    - วิธีนี้หนัก (O(N^2)) แต่ปลอด dependency
    - ถ้า N ใหญ่มาก จะช้า แต่ยัง "รันได้แน่" และไม่กิน RAM ระเบิด
    """
    if x.ndim != 2:
        raise ValueError(f"knn_indices_chunked expects x shape (N,D), got {tuple(x.shape)}")

    N, D = x.shape
    N = int(N)
    k = int(k)

    if N <= 1:
        return torch.zeros((N, 0), dtype=torch.long, device=x.device)

    # ถ้า k >= N ให้ลดลง (เพราะห้ามเลือกตัวเองหรือเพื่อนบ้านเกินจำนวน)
    k_eff = min(k, N - (1 if exclude_self else 0))
    if k_eff <= 0:
        return torch.zeros((N, 0), dtype=torch.long, device=x.device)

    # คำนวณ norm^2 ของ x ล่วงหน้า เพื่อใช้สูตร:
    # dist^2(q, x) = ||q||^2 + ||x||^2 - 2 q x^T
    x_f = x.float()  # คุมให้เป็น float32 เพื่อความเสถียร
    x_norm2 = (x_f * x_f).sum(dim=1)  # (N,)

    out = torch.empty((N, k_eff), dtype=torch.long, device=x.device)

    for s in range(0, N, int(chunk_size)):
        e = min(N, s + int(chunk_size))
        q = x_f[s:e]  # (Q,D)
        q_norm2 = (q * q).sum(dim=1, keepdim=True)  # (Q,1)

        # (Q,N)
        dist2 = q_norm2 + x_norm2.unsqueeze(0) - 2.0 * (q @ x_f.t())
        dist2 = torch.clamp(dist2, min=0.0)

        if exclude_self:
            # ตัด self ของแถวที่เกี่ยวข้องออก
            rows = torch.arange(e - s, device=x.device)
            cols = rows + s
            dist2[rows, cols] = float("inf")

        idx = dist2.topk(k_eff, dim=1, largest=False).indices  # (Q,k)
        out[s:e] = idx

    return out


# ============================================================
# Graph Convolution Blocks (ตาม paper)
# ============================================================
class CStreamAttentionBlock(nn.Module):
    """
    C-stream block: ใช้ Graph Attention aggregation
    ตาม paper:
      f_hat_ij = MLP_calib([f_i, f_ij])
      alpha_ij = softmax( MLP_att([f_i - f_ij, f_ij]) ) (softmax over neighbors)
      f_i_out  = sum_j alpha_ij * f_hat_ij
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        *,
        negative_slope: float = 0.2,
        dropout: float = 0.0,
        agg_chunk: int = 2048,
    ):
        super().__init__()
        self.in_ch = int(in_ch)
        self.out_ch = int(out_ch)
        self.agg_chunk = int(agg_chunk)

        self.mlp_calib = MLP(
            in_ch * 2,
            out_ch,
            bn=True,
            act=True,
            negative_slope=negative_slope,
            dropout=dropout,
        )
        self.mlp_att = MLP(
            in_ch * 2,
            out_ch,
            bn=True,
            act=False,  # score ไม่ต้องผ่าน activation
            negative_slope=negative_slope,
            dropout=0.0,
        )

    def forward(self, f: torch.Tensor, knn_idx: torch.Tensor) -> torch.Tensor:
        """
        f: (N, in_ch)
        knn_idx: (N, K) long
        return: (N, out_ch)
        """
        N, Cin = f.shape
        if Cin != self.in_ch:
            raise ValueError(f"CStreamAttentionBlock Cin mismatch: got {Cin}, expect {self.in_ch}")
        if knn_idx.ndim != 2 or knn_idx.shape[0] != N:
            raise ValueError(f"knn_idx must be (N,K), got {tuple(knn_idx.shape)}")

        K = int(knn_idx.shape[1])
        if K == 0:
            return torch.zeros((N, self.out_ch), dtype=f.dtype, device=f.device)

        out = torch.empty((N, self.out_ch), dtype=f.dtype, device=f.device)

        # ทำเป็น chunk เพื่อลดเมม (สำคัญมากเมื่อ N=16000, K=32, C=256)
        for s in range(0, N, self.agg_chunk):
            e = min(N, s + self.agg_chunk)
            idx = knn_idx[s:e]  # (Q,K)
            fi = f[s:e]  # (Q,Cin)
            fj = f[idx]  # (Q,K,Cin)

            fi_rep = fi.unsqueeze(1).expand(-1, K, -1)  # (Q,K,Cin)

            # calib: [fi, fj]
            cat_calib = torch.cat([fi_rep, fj], dim=-1).reshape(-1, 2 * Cin)  # (Q*K, 2Cin)
            f_hat = self.mlp_calib(cat_calib).reshape(e - s, K, self.out_ch)  # (Q,K,Cout)

            # attention score: [fi - fj, fj]
            delta = fi_rep - fj
            cat_att = torch.cat([delta, fj], dim=-1).reshape(-1, 2 * Cin)  # (Q*K, 2Cin)
            score = self.mlp_att(cat_att).reshape(e - s, K, self.out_ch)  # (Q,K,Cout)

            # softmax over neighbors (dim=1) แยกต่อ channel
            alpha = torch.softmax(score, dim=1)  # (Q,K,Cout)

            # weighted sum
            fo = (alpha * f_hat).sum(dim=1)  # (Q,Cout)
            out[s:e] = fo

        return out


class NStreamMaxBlock(nn.Module):
    """
    N-stream block: ใช้ max-pooling aggregation (ตาม paper)
      f_hat_ij = MLP_calib([f_i, f_ij])
      f_i_out  = max_j f_hat_ij  (channel-wise max)
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        *,
        negative_slope: float = 0.2,
        dropout: float = 0.0,
        agg_chunk: int = 2048,
    ):
        super().__init__()
        self.in_ch = int(in_ch)
        self.out_ch = int(out_ch)
        self.agg_chunk = int(agg_chunk)

        self.mlp_calib = MLP(
            in_ch * 2,
            out_ch,
            bn=True,
            act=True,
            negative_slope=negative_slope,
            dropout=dropout,
        )

    def forward(self, f: torch.Tensor, knn_idx: torch.Tensor) -> torch.Tensor:
        """
        f: (N, in_ch)
        knn_idx: (N, K)
        return: (N, out_ch)
        """
        N, Cin = f.shape
        if Cin != self.in_ch:
            raise ValueError(f"NStreamMaxBlock Cin mismatch: got {Cin}, expect {self.in_ch}")
        if knn_idx.ndim != 2 or knn_idx.shape[0] != N:
            raise ValueError(f"knn_idx must be (N,K), got {tuple(knn_idx.shape)}")

        K = int(knn_idx.shape[1])
        if K == 0:
            return torch.zeros((N, self.out_ch), dtype=f.dtype, device=f.device)

        out = torch.empty((N, self.out_ch), dtype=f.dtype, device=f.device)

        for s in range(0, N, self.agg_chunk):
            e = min(N, s + self.agg_chunk)
            idx = knn_idx[s:e]  # (Q,K)
            fi = f[s:e]  # (Q,Cin)
            fj = f[idx]  # (Q,K,Cin)

            fi_rep = fi.unsqueeze(1).expand(-1, K, -1)  # (Q,K,Cin)

            cat = torch.cat([fi_rep, fj], dim=-1).reshape(-1, 2 * Cin)  # (Q*K,2Cin)
            f_hat = self.mlp_calib(cat).reshape(e - s, K, self.out_ch)  # (Q,K,Cout)

            fo = f_hat.max(dim=1).values  # (Q,Cout)
            out[s:e] = fo

        return out


# ============================================================
# Main Model: TSGCNet (H-fusion ตาม paper)
# ============================================================
class TSGCNet(nn.Module):
    """
    TSGCNet (CVPR 2021) — เวอร์ชันที่ "เข้ากับ dataloader ของคุณ"
    อินพุตที่ต้องมีใน batch:
      - x_c: (B, F, 12)  coords 12D (v0,v1,v2,center)
      - x_n: (B, F, 12)  normals 12D (n0,n1,n2,n_face)
    และแนะนำให้มี:
      - F_used: list[int] หรือ tensor (B,)  จำนวน face จริง (ไม่รวม pad)
      - valid_face: (B, F) bool (optional)

    เอาต์พุต:
      - logits: (B, F, num_classes)  (ตำแหน่งที่ pad จะเป็นศูนย์ และควร ignore ด้วย mask ตอนคำนวณ loss)
    """

    def __init__(
        self,
        *,
        num_classes: int = 17,
        k: int = 32,
        dims: Tuple[int, int, int] = (64, 128, 256),
        fusion_dim: int = 512,
        negative_slope: float = 0.2,
        dropout: float = 0.0,
        knn_chunk: int = 1024,
        agg_chunk: int = 2048,
    ):
        super().__init__()
        self.num_classes = int(num_classes)
        self.k = int(k)
        self.knn_chunk = int(knn_chunk)

        d1, d2, d3 = (int(d) for d in dims)
        self.dims = (d1, d2, d3)

        # C-stream (coords) : attention blocks
        self.c1 = CStreamAttentionBlock(12, d1, negative_slope=negative_slope, dropout=dropout, agg_chunk=agg_chunk)
        self.c2 = CStreamAttentionBlock(d1, d2, negative_slope=negative_slope, dropout=dropout, agg_chunk=agg_chunk)
        self.c3 = CStreamAttentionBlock(d2, d3, negative_slope=negative_slope, dropout=dropout, agg_chunk=agg_chunk)

        # N-stream (normals) : max blocks (ใช้ graph เดียวกับ C-stream ในแต่ละ layer)
        self.n1 = NStreamMaxBlock(12, d1, negative_slope=negative_slope, dropout=dropout, agg_chunk=agg_chunk)
        self.n2 = NStreamMaxBlock(d1, d2, negative_slope=negative_slope, dropout=dropout, agg_chunk=agg_chunk)
        self.n3 = NStreamMaxBlock(d2, d3, negative_slope=negative_slope, dropout=dropout, agg_chunk=agg_chunk)

        # Fusion part:
        # concat multi-scale features: (d1+d2+d3) -> fusion_dim
        hier_dim = d1 + d2 + d3
        self.mlp_c = MLP(hier_dim, fusion_dim, bn=True, act=True, negative_slope=negative_slope, dropout=dropout)
        self.mlp_n = MLP(hier_dim, fusion_dim, bn=True, act=True, negative_slope=negative_slope, dropout=dropout)

        # predictor: (2*fusion_dim) -> 512 -> 256 -> 128 -> C
        self.pred1 = MLP(2 * fusion_dim, 512, bn=True, act=True, negative_slope=negative_slope, dropout=dropout)
        self.pred2 = MLP(512, 256, bn=True, act=True, negative_slope=negative_slope, dropout=dropout)
        self.pred3 = MLP(256, 128, bn=True, act=True, negative_slope=negative_slope, dropout=dropout)
        self.pred4 = nn.Linear(128, self.num_classes)

    def _build_knn(self, f_c: torch.Tensor, k: int) -> torch.Tensor:
        """
        สร้าง KNN graph indices จาก feature ของ C-stream ใน layer นั้น ๆ
        f_c: (N, D)
        return: (N, k)
        """
        return knn_indices_chunked(
            f_c,
            k=k,
            chunk_size=self.knn_chunk,
            exclude_self=True,
        )

    def _forward_single(self, x_c: torch.Tensor, x_n: torch.Tensor) -> torch.Tensor:
        """
        ทำ forward สำหรับ sample เดียว (ไม่รวม pad)
        x_c, x_n: (N,12)
        return logits: (N, C)
        """
        # ----- layer 1 -----
        knn1 = self._build_knn(x_c, self.k)  # KNN graph จาก coords
        c1 = self.c1(x_c, knn1)
        n1 = self.n1(x_n, knn1)

        # ----- layer 2 -----
        knn2 = self._build_knn(c1, self.k)
        c2 = self.c2(c1, knn2)
        n2 = self.n2(n1, knn2)

        # ----- layer 3 -----
        knn3 = self._build_knn(c2, self.k)
        c3 = self.c3(c2, knn3)
        n3 = self.n3(n2, knn3)

        # ----- hierarchical concat (skip connections) -----
        hc = torch.cat([c1, c2, c3], dim=1)  # (N, d1+d2+d3)
        hn = torch.cat([n1, n2, n3], dim=1)  # (N, d1+d2+d3)

        Fc = self.mlp_c(hc)  # (N, fusion_dim)
        Fn = self.mlp_n(hn)  # (N, fusion_dim)

        Ffuse = torch.cat([Fc, Fn], dim=1)  # (N, 2*fusion_dim)

        z = self.pred1(Ffuse)
        z = self.pred2(z)
        z = self.pred3(z)
        logits = self.pred4(z)  # (N, C)

        return logits

    def forward(
        self,
        batch: Optional[Dict[str, Any]] = None,
        *,
        x_c: Optional[torch.Tensor] = None,
        x_n: Optional[torch.Tensor] = None,
        F_used: Optional[Union[List[int], torch.Tensor]] = None,
        valid_face: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        รองรับ 2 แบบ:
          1) model(batch_dict)
          2) model(x_c=..., x_n=..., F_used=..., valid_face=...)

        return logits: (B, F, num_classes)
        """
        if batch is not None:
            if x_c is None:
                x_c = batch.get("x_c", None)
            if x_n is None:
                x_n = batch.get("x_n", None)
            if F_used is None:
                F_used = batch.get("F_used", None)
            if valid_face is None:
                valid_face = batch.get("valid_face", None)

        if x_c is None or x_n is None:
            raise KeyError("TSGCNet ต้องการ x_c และ x_n ใน batch (จาก face_feature=twostream24).")

        if x_c.ndim != 3 or x_n.ndim != 3:
            raise ValueError(f"x_c/x_n ต้องเป็น (B,F,12) แต่ได้ x_c={tuple(x_c.shape)}, x_n={tuple(x_n.shape)}")

        B, Fmax, Dc = x_c.shape
        if Dc != 12 or x_n.shape[-1] != 12:
            raise ValueError("x_c และ x_n ต้องมีมิติสุดท้าย = 12 (two-stream 24D split).")

        device = x_c.device
        logits_out = torch.zeros((B, Fmax, self.num_classes), dtype=x_c.dtype, device=device)

        # แปลง F_used ให้เป็น list[int] ง่าย ๆ
        if F_used is None:
            # ถ้าไม่มี ให้ถือว่าใช้ครบทั้ง F
            f_used_list = [int(Fmax)] * int(B)
        elif isinstance(F_used, list):
            f_used_list = [int(v) for v in F_used]
        elif torch.is_tensor(F_used):
            f_used_list = [int(v) for v in F_used.detach().cpu().tolist()]
        else:
            raise TypeError(f"F_used must be list[int] or Tensor, got {type(F_used)}")

        # ทำทีละ sample เพื่อกัน KNN ข้ามเคส และควบคุมเมม
        for b in range(int(B)):
            Nu = max(0, min(int(Fmax), int(f_used_list[b])))
            if Nu <= 0:
                continue

            xc = x_c[b, :Nu].contiguous()
            xn = x_n[b, :Nu].contiguous()

            # (optional) ถ้ามี valid_face ก็เอามาเช็คความสอดคล้อง (ไม่บังคับ)
            if valid_face is not None:
                # valid_face: (B,F) bool
                vb = valid_face[b]
                # ถ้า Nu เป็นจำนวนจริง ๆ ที่ใช้ ควรมี vb[:Nu] เป็น True เป็นส่วนใหญ่
                # (ไม่ทำอะไรต่อ แค่กัน debug ในอนาคต)
                _ = vb

            logits_real = self._forward_single(xc, xn)  # (Nu,C)
            logits_out[b, :Nu] = logits_real

        return logits_out

    def compute_loss(
        self,
        logits: torch.Tensor,
        y: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        *,
        ignore_index: int = -1,
    ) -> torch.Tensor:
        """
        (ทางเลือก) ฟังก์ชันช่วยคำนวณ loss แบบ masked CE
        - logits: (B,F,C)
        - y:      (B,F)
        - mask:   (B,F) bool (ถ้ามี)
        """
        B, Fmax, C = logits.shape
        logits2 = logits.reshape(B * Fmax, C)
        y2 = y.reshape(B * Fmax)

        if mask is None:
            return F.cross_entropy(logits2, y2, ignore_index=int(ignore_index))

        m2 = mask.reshape(B * Fmax).bool()
        if m2.sum().item() == 0:
            # ถ้า mask ว่าง ให้คืน loss=0 (กัน NaN)
            return logits2.sum() * 0.0

        return F.cross_entropy(logits2[m2], y2[m2], ignore_index=int(ignore_index))