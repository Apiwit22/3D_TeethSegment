# models/pointcnn.py
from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# ------------------------------------------------------------
# Optional torch-cluster (fast kNN)
# ------------------------------------------------------------
try:
    from torch_cluster import knn  # type: ignore
except Exception:
    knn = None


# ============================================================
# Utils: FPS (pure torch)  (B,N,3) -> (B,S)
# ============================================================
def farthest_point_sample(xyz: torch.Tensor, npoint: int) -> torch.Tensor:
    """
    xyz: (B,N,3)
    return: (B,npoint) indices
    """
    if xyz.dim() != 3 or xyz.size(-1) != 3:
        raise ValueError(f"FPS expects (B,N,3), got {tuple(xyz.shape)}")
    B, N, _ = xyz.shape
    if N <= 0:
        raise ValueError("FPS: N must be > 0")
    npoint = int(npoint)
    if npoint <= 0:
        raise ValueError("FPS: npoint must be > 0")
    npoint = min(npoint, N)

    device = xyz.device
    centroids = torch.zeros(B, npoint, dtype=torch.long, device=device)
    distance = torch.full((B, N), 1e10, device=device, dtype=xyz.dtype)
    farthest = torch.randint(0, N, (B,), dtype=torch.long, device=device)
    batch_indices = torch.arange(B, dtype=torch.long, device=device)

    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, dim=-1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, dim=1)[1]
    return centroids


def index_points(points: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """
    points: (B,N,C)
    idx: (B,S) or (B,S,K)
    """
    if points.dim() != 3:
        raise ValueError(f"index_points expects (B,N,C), got {tuple(points.shape)}")
    B = points.size(0)
    device = points.device

    if idx.dim() == 2:
        batch = torch.arange(B, device=device).view(B, 1)
        return points[batch, idx, :]
    if idx.dim() == 3:
        batch = torch.arange(B, device=device).view(B, 1, 1)
        return points[batch, idx, :]
    raise ValueError(f"idx must be 2D or 3D, got {idx.dim()}D")


# ============================================================
# kNN: rep (B,S,3) against all (B,N,3) -> idx (B,S,K)
# ============================================================
def knn_indices(rep_xyz: torch.Tensor, all_xyz: torch.Tensor, k: int) -> torch.Tensor:
    """
    return neighbor indices in all_xyz for each rep point.
    rep_xyz: (B,S,3)
    all_xyz: (B,N,3)
    -> idx: (B,S,K) long
    """
    if rep_xyz.dim() != 3 or all_xyz.dim() != 3:
        raise ValueError("knn_indices expects 3D tensors")
    if rep_xyz.size(-1) != 3 or all_xyz.size(-1) != 3:
        raise ValueError("knn_indices expects last dim=3")

    B, S, _ = rep_xyz.shape
    _, N, _ = all_xyz.shape
    k = int(k)
    if k <= 0:
        raise ValueError("k must be > 0")
    k = min(k, N)

    # fast path: torch-cluster knn(x, y, k) where:
    # x: (Nx,3) database, y: (Ny,3) query
    # returns edge_index (2, Ny*k) with [src=database_idx, dst=query_idx] (depends on version)
    if knn is not None:
        idx_out = []
        for b in range(B):
            x = all_xyz[b]  # (N,3)
            y = rep_xyz[b]  # (S,3)
            edge = knn(x, y, k=k)  # (2, S*k)
            src = edge[0]          # indices into x (all)
            dst = edge[1]          # indices into y (rep) 0..S-1 repeated k times

            order = torch.argsort(dst)
            src = src[order].reshape(S, k)
            idx_out.append(src)
        return torch.stack(idx_out, dim=0)

    # fallback: cdist (slow but works)
    # dist: (B,S,N)
    dist = torch.cdist(rep_xyz, all_xyz)  # fp32 recommended
    idx = torch.topk(dist, k=k, dim=-1, largest=False, sorted=False).indices
    return idx


def gather_neighbors(x_bnc: torch.Tensor, idx_bsk: torch.Tensor) -> torch.Tensor:
    """
    x_bnc: (B,N,C)
    idx_bsk: (B,S,K)
    -> (B,S,K,C)
    """
    B, N, C = x_bnc.shape
    B2, S, K = idx_bsk.shape
    if B2 != B:
        raise ValueError("B mismatch in gather_neighbors")
    idx_flat = idx_bsk.reshape(B, S * K).unsqueeze(-1).expand(B, S * K, C)
    g = torch.gather(x_bnc, dim=1, index=idx_flat)
    return g.reshape(B, S, K, C)


# ============================================================
# X-Conv (Paper-style core idea)
# ============================================================
class XConv(nn.Module):
    """
    Paper-style PointCNN XConv:
      - neighbor features: concat(delta_xyz, neighbor_feat)
      - lift to Cmid
      - learn X (KxK) from local coords, row-softmax
      - transform: X @ lifted
      - aggregate with separable conv over K -> Cout at representative points
    """
    def __init__(
        self,
        Cin: int,
        Cout: int,
        K: int,
        Cmid: Optional[int] = None,
        with_bn: bool = True,
        dropout_p: float = 0.0,
    ):
        super().__init__()
        self.Cin = int(Cin)
        self.Cout = int(Cout)
        self.K = int(K)
        self.Cmid = int(Cmid) if Cmid is not None else int(Cout)

        # Lift: (delta_xyz + feat) -> Cmid
        self.lift = nn.Sequential(
            nn.Linear(self.Cin + 3, self.Cmid),
            nn.ReLU(inplace=True),
            nn.Linear(self.Cmid, self.Cmid),
            nn.ReLU(inplace=True),
        )

        # X-transform: from local coords (K*3) -> K*K
        # (Paper uses MLP + reshape; we'll do row-softmax)
        hidden = max(64, self.K * 8)
        self.x_mlp = nn.Sequential(
            nn.Linear(self.K * 3, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, self.K * self.K),
        )

        # Separable conv aggregator over K
        # depthwise conv over K, then pointwise to Cout
        self.depthwise = nn.Conv1d(self.Cmid, self.Cmid, kernel_size=self.K, groups=self.Cmid, bias=False)
        self.pointwise = nn.Conv1d(self.Cmid, self.Cout, kernel_size=1, bias=False)

        self.bn = nn.BatchNorm1d(self.Cout) if with_bn else nn.Identity()
        self.drop = nn.Dropout(p=dropout_p) if dropout_p > 0 else nn.Identity()

    def forward(
        self,
        rep_xyz: torch.Tensor,      # (B,S,3)
        all_xyz: torch.Tensor,      # (B,N,3)
        all_feat: torch.Tensor,     # (B,N,Cin)
    ) -> torch.Tensor:
        B, S, _ = rep_xyz.shape
        _, N, _ = all_xyz.shape
        if all_feat.shape[:2] != (B, N) or all_feat.shape[-1] != self.Cin:
            raise ValueError(f"all_feat must be (B,N,{self.Cin}), got {tuple(all_feat.shape)}")

        # neighbors in all points for each rep
        idx = knn_indices(rep_xyz, all_xyz, k=self.K)            # (B,S,K)
        nb_xyz = gather_neighbors(all_xyz, idx)                  # (B,S,K,3)
        nb_feat = gather_neighbors(all_feat, idx)                # (B,S,K,Cin)

        # local coords
        delta = nb_xyz - rep_xyz.unsqueeze(2)                    # (B,S,K,3)

        # lift
        e = torch.cat([delta, nb_feat], dim=-1)                  # (B,S,K,3+Cin)
        lifted = self.lift(e)                                    # (B,S,K,Cmid)

        # X transform
        delta_flat = delta.reshape(B, S, self.K * 3)             # (B,S,K*3)
        X = self.x_mlp(delta_flat)                               # (B,S,K*K)
        X = X.view(B, S, self.K, self.K)

        # row-softmax for stability (each row sums to 1)
        X = torch.softmax(X, dim=-1)

        # apply: (K,K) @ (K,Cmid) -> (K,Cmid)
        # lifted: (B,S,K,Cmid)
        lifted_t = torch.matmul(X, lifted)                       # (B,S,K,Cmid)

        # aggregate over K with separable conv
        # to conv1d: (B*S, Cmid, K)
        h = lifted_t.permute(0, 1, 3, 2).contiguous().view(B * S, self.Cmid, self.K)
        h = self.depthwise(h)                                    # (B*S, Cmid, 1)
        h = F.relu(h, inplace=True)
        h = self.pointwise(h)                                    # (B*S, Cout, 1)
        h = self.bn(h)
        h = F.relu(h, inplace=True)
        h = self.drop(h)
        h = h.squeeze(-1).view(B, S, self.Cout)                  # (B,S,Cout)
        return h


# ============================================================
# Feature Propagation (3-NN inverse distance)
# ============================================================
def square_distance(src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
    # src (B,N,3), dst (B,S,3) -> (B,N,S)
    dist = -2 * torch.matmul(src, dst.transpose(1, 2))
    dist += torch.sum(src ** 2, dim=-1).unsqueeze(-1)
    dist += torch.sum(dst ** 2, dim=-1).unsqueeze(1)
    return dist


class FP(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout_p: float = 0.0):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.ReLU(inplace=True),
            nn.Linear(out_channels, out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_p) if dropout_p > 0 else nn.Identity(),
        )

    def forward(
        self,
        xyz1: torch.Tensor,   # (B,N,3) target (dense)
        xyz2: torch.Tensor,   # (B,S,3) source (sparse)
        feat1: Optional[torch.Tensor],  # (B,N,C1) skip
        feat2: torch.Tensor,  # (B,S,C2)
    ) -> torch.Tensor:
        B, N, _ = xyz1.shape
        _, S, _ = xyz2.shape

        if S == 1:
            interp = feat2.repeat(1, N, 1)
        else:
            d = square_distance(xyz1, xyz2)  # (B,N,S)
            k = 3 if S >= 3 else S
            d, idx = torch.topk(d, k=k, dim=-1, largest=False, sorted=True)  # (B,N,k)
            d = torch.clamp(d, min=1e-10)
            w = 1.0 / d
            w = w / torch.sum(w, dim=-1, keepdim=True)

            # gather feat2: (B,N,k,C2)
            idx_expand = idx.unsqueeze(-1).expand(-1, -1, -1, feat2.size(-1))
            f2g = torch.gather(feat2.unsqueeze(1).expand(-1, N, -1, -1), dim=2, index=idx_expand)
            interp = torch.sum(f2g * w.unsqueeze(-1), dim=2)  # (B,N,C2)

        if feat1 is not None:
            fused = torch.cat([feat1, interp], dim=-1)
        else:
            fused = interp
        return self.mlp(fused)


# ============================================================
# PointCNN (Paper-style) Segmentation Network
# ============================================================
class PointCNNPaperSeg(nn.Module):
    """
    forward(batch):
      batch["x"]   (B,N,in_channels)  e.g. xyz_n -> 6
      batch["pos"] (B,N,3)            sampled xyz
    """
    def __init__(
        self,
        num_classes: int = 17,
        in_channels: int = 6,
        num_points: int = 4096,
        # hierarchy sizes
        n1: int = 1024,
        n2: int = 256,
        n3: int = 64,
        # neighbors
        k: int = 16,
        # widths
        c1: int = 64,
        c2: int = 128,
        c3: int = 256,
        dropout_p: float = 0.2,
    ):
        super().__init__()
        self.num_classes = int(num_classes)
        self.in_channels = int(in_channels)
        self.num_points = int(num_points)

        self.n1 = int(n1)
        self.n2 = int(n2)
        self.n3 = int(n3)
        self.k = int(k)

        # Encoder XConv
        self.xconv1 = XConv(Cin=self.in_channels, Cout=int(c1), K=self.k, Cmid=int(c1), dropout_p=dropout_p)
        self.xconv2 = XConv(Cin=int(c1),        Cout=int(c2), K=self.k, Cmid=int(c2), dropout_p=dropout_p)
        self.xconv3 = XConv(Cin=int(c2),        Cout=int(c3), K=self.k, Cmid=int(c3), dropout_p=dropout_p)

        # Decoder FP
        self.fp3 = FP(in_channels=int(c3) + int(c2), out_channels=int(c2), dropout_p=dropout_p)  # l3->l2
        self.fp2 = FP(in_channels=int(c2) + int(c1), out_channels=int(c1), dropout_p=dropout_p)  # l2->l1
        self.fp1 = FP(in_channels=int(c1) + self.in_channels, out_channels=int(c1), dropout_p=dropout_p)  # l1->l0

        # head
        self.head = nn.Sequential(
            nn.Linear(int(c1), 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_p),
            nn.Linear(128, self.num_classes),
        )

    def forward(self, batch: dict) -> torch.Tensor:
        x = batch["x"]     # (B,N,C)
        pos = batch["pos"] # (B,N,3)

        if x.dim() != 3 or pos.dim() != 3:
            raise ValueError(f"Expected x,pos as (B,N,*), got x={tuple(x.shape)} pos={tuple(pos.shape)}")
        if x.shape[0] != pos.shape[0] or x.shape[1] != pos.shape[1]:
            raise ValueError(f"B/N mismatch: x={tuple(x.shape)} pos={tuple(pos.shape)}")
        if x.shape[-1] != self.in_channels:
            raise ValueError(f"in_channels mismatch: expected {self.in_channels}, got {x.shape[-1]}")

        B, N, _ = pos.shape
        # (optional) assert N == num_points, but allow flexible
        # if N != self.num_points: pass

        # ----- level 0 -----
        l0_xyz = pos
        l0_feat = x

        # ----- FPS reps -----
        i1 = farthest_point_sample(l0_xyz, self.n1)          # (B,n1)
        l1_xyz = index_points(l0_xyz, i1)                    # (B,n1,3)
        l1_feat = self.xconv1(l1_xyz, l0_xyz, l0_feat)       # (B,n1,c1)

        i2 = farthest_point_sample(l1_xyz, self.n2)
        l2_xyz = index_points(l1_xyz, i2)
        l2_feat = self.xconv2(l2_xyz, l1_xyz, l1_feat)       # (B,n2,c2)

        i3 = farthest_point_sample(l2_xyz, self.n3)
        l3_xyz = index_points(l2_xyz, i3)
        l3_feat = self.xconv3(l3_xyz, l2_xyz, l2_feat)       # (B,n3,c3)

        # ----- Decoder FP -----
        u2 = self.fp3(l2_xyz, l3_xyz, l2_feat, l3_feat)      # (B,n2,c2)
        u1 = self.fp2(l1_xyz, l2_xyz, l1_feat, u2)           # (B,n1,c1)
        u0 = self.fp1(l0_xyz, l1_xyz, l0_feat, u1)           # (B,N,c1)

        logits = self.head(u0)                               # (B,N,num_classes)
        return logits
