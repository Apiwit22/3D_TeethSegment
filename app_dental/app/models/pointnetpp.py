# models/pointnetpp.py
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------
# PointNet++ utility functions
# -----------------------------
def square_distance(src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
    """
    src: (B, N, 3)
    dst: (B, M, 3)
    return: (B, N, M) squared distance
    """
    if src.dim() != 3 or dst.dim() != 3:
        raise ValueError(f"square_distance expects 3D tensors, got src={src.shape}, dst={dst.shape}")
    if src.size(-1) != 3 or dst.size(-1) != 3:
        raise ValueError(f"square_distance expects last dim=3, got src={src.shape}, dst={dst.shape}")

    B, N, _ = src.shape
    _, M, _ = dst.shape

    dist = -2 * torch.matmul(src, dst.transpose(1, 2))  # (B,N,M)
    dist += torch.sum(src ** 2, dim=-1).view(B, N, 1)
    dist += torch.sum(dst ** 2, dim=-1).view(B, 1, M)
    return dist


def index_points(points: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """
    points: (B, N, C)
    idx: (B, S) or (B, S, K)
    return:
      (B, S, C) or (B, S, K, C)
    """
    if points.dim() != 3:
        raise ValueError(f"index_points expects points as (B,N,C), got {points.shape}")
    device = points.device
    B = points.shape[0]

    if idx.dim() == 2:
        batch_indices = torch.arange(B, device=device).view(B, 1)
        return points[batch_indices, idx, :]
    elif idx.dim() == 3:
        batch_indices = torch.arange(B, device=device).view(B, 1, 1)
        return points[batch_indices, idx, :]
    else:
        raise ValueError(f"idx must be 2D or 3D, got {idx.dim()}D")


def farthest_point_sample(xyz: torch.Tensor, npoint: int) -> torch.Tensor:
    """
    xyz: (B, N, 3)
    return: (B, npoint) indices
    SAFE:
      - clamp npoint <= N
      - error if N == 0
    """
    if xyz.dim() != 3 or xyz.size(-1) != 3:
        raise ValueError(f"farthest_point_sample expects xyz as (B,N,3), got {xyz.shape}")

    device = xyz.device
    B, N, _ = xyz.shape
    if N <= 0:
        raise ValueError("farthest_point_sample: N must be > 0")

    npoint = int(npoint)
    if npoint <= 0:
        raise ValueError("farthest_point_sample: npoint must be > 0")

    # clamp
    npoint = min(npoint, N)

    centroids = torch.zeros(B, npoint, dtype=torch.long, device=device)
    distance = torch.full((B, N), 1e10, device=device, dtype=xyz.dtype)
    farthest = torch.randint(0, N, (B,), dtype=torch.long, device=device)

    batch_indices = torch.arange(B, dtype=torch.long, device=device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)  # (B,1,3)
        dist = torch.sum((xyz - centroid) ** 2, dim=-1)           # (B,N)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, dim=1)[1]
    return centroids


def query_ball_point(radius: float, nsample: int, xyz: torch.Tensor, new_xyz: torch.Tensor) -> torch.Tensor:
    """
    xyz: (B, N, 3)
    new_xyz: (B, S, 3)
    return: group_idx (B, S, nsample)

    SAFE:
      - clamp nsample <= N
      - handle case where no points fall within radius for a group
    """
    if xyz.dim() != 3 or new_xyz.dim() != 3:
        raise ValueError(f"query_ball_point expects 3D tensors, got xyz={xyz.shape}, new_xyz={new_xyz.shape}")
    if xyz.size(-1) != 3 or new_xyz.size(-1) != 3:
        raise ValueError(f"query_ball_point expects last dim=3, got xyz={xyz.shape}, new_xyz={new_xyz.shape}")

    B, N, _ = xyz.shape
    _, S, _ = new_xyz.shape
    if N <= 0:
        raise ValueError("query_ball_point: N must be > 0")
    nsample = int(nsample)
    if nsample <= 0:
        raise ValueError("query_ball_point: nsample must be > 0")

    # clamp
    nsample = min(nsample, N)

    sqrdists = square_distance(new_xyz, xyz)  # (B,S,N)

    # indices 0..N-1
    # NOTE: expand+clone instead of repeat (same memory at the end, but avoids repeat semantics)
    group_idx = torch.arange(N, device=xyz.device).view(1, 1, N).expand(B, S, N).clone()

    # mark out-of-radius as N (sentinel)
    group_idx[sqrdists > radius * radius] = N

    # sort so valid indices come first; take first nsample
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]  # (B,S,nsample)

    # If a group has ZERO valid points, then group_idx[:,:,0] == N -> would break indexing.
    # Fix: set "first" to 0 for those groups.
    first = group_idx[:, :, 0]  # (B,S)
    first_safe = first.clone()
    first_safe[first_safe == N] = 0  # fallback to point 0
    first_safe = first_safe.view(B, S, 1).expand(B, S, nsample)

    # replace sentinel N with first valid (or 0 if none valid)
    mask = group_idx == N
    group_idx[mask] = first_safe[mask]
    return group_idx


def sample_and_group(
    npoint: int,
    radius: float,
    nsample: int,
    xyz: torch.Tensor,
    points: torch.Tensor | None
):
    """
    xyz: (B, N, 3)
    points: (B, N, D) or None
    return:
      new_xyz: (B, npoint', 3)
      new_points: (B, npoint', nsample', 3 + D)
    SAFE:
      - clamp npoint <= N
      - clamp nsample <= N
    """
    if xyz.dim() != 3 or xyz.size(-1) != 3:
        raise ValueError(f"sample_and_group expects xyz as (B,N,3), got {xyz.shape}")
    B, N, _ = xyz.shape
    if N <= 0:
        raise ValueError("sample_and_group: N must be > 0")

    npoint = int(npoint)
    nsample = int(nsample)
    if npoint <= 0 or nsample <= 0:
        raise ValueError("sample_and_group: npoint and nsample must be > 0")

    # clamp to prevent out-of-range
    npoint = min(npoint, N)
    nsample = min(nsample, N)

    fps_idx = farthest_point_sample(xyz, npoint)          # (B,npoint)
    new_xyz = index_points(xyz, fps_idx)                  # (B,npoint,3)
    idx = query_ball_point(radius, nsample, xyz, new_xyz) # (B,npoint,nsample)
    grouped_xyz = index_points(xyz, idx)                  # (B,npoint,nsample,3)
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, npoint, 1, 3)

    if points is not None:
        if points.dim() != 3 or points.shape[0] != B or points.shape[1] != N:
            raise ValueError(f"points must be (B,N,D) aligned with xyz, got {points.shape}, xyz={xyz.shape}")
        grouped_points = index_points(points, idx)        # (B,npoint,nsample,D)
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1)  # (B,npoint,nsample,3+D)
    else:
        new_points = grouped_xyz_norm

    return new_xyz, new_points


def sample_and_group_all(xyz: torch.Tensor, points: torch.Tensor | None):
    """
    group all points into one set
    xyz: (B, N, 3)
    points: (B, N, D) or None
    return:
      new_xyz: (B, 1, 3)
      new_points: (B, 1, N, 3 + D)
    """
    if xyz.dim() != 3 or xyz.size(-1) != 3:
        raise ValueError(f"sample_and_group_all expects xyz as (B,N,3), got {xyz.shape}")

    device = xyz.device
    B, N, _ = xyz.shape
    if N <= 0:
        raise ValueError("sample_and_group_all: N must be > 0")

    new_xyz = torch.zeros(B, 1, 3, device=device, dtype=xyz.dtype)
    grouped_xyz = xyz.view(B, 1, N, 3)
    if points is not None:
        if points.dim() != 3 or points.shape[0] != B or points.shape[1] != N:
            raise ValueError(f"points must be (B,N,D) aligned with xyz, got {points.shape}, xyz={xyz.shape}")
        new_points = torch.cat([grouped_xyz, points.view(B, 1, N, -1)], dim=-1)
    else:
        new_points = grouped_xyz
    return new_xyz, new_points


# -----------------------------
# PointNet++ blocks
# -----------------------------
class PointNetSetAbstraction(nn.Module):
    def __init__(
        self,
        npoint: int | None,
        radius: float | None,
        nsample: int | None,
        in_channel: int,
        mlp: list[int],
        group_all: bool
    ):
        super().__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.group_all = group_all

        last_channel = in_channel
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel

    def forward(self, xyz: torch.Tensor, points: torch.Tensor | None):
        """
        xyz: (B, N, 3)
        points: (B, N, D) or None
        return:
          new_xyz: (B, S, 3)
          new_points: (B, S, mlp[-1])
        """
        if self.group_all:
            new_xyz, new_points = sample_and_group_all(xyz, points)   # (B,1,3), (B,1,N,3+D)
        else:
            if self.npoint is None or self.radius is None or self.nsample is None:
                raise ValueError("SetAbstraction: npoint/radius/nsample must not be None when group_all=False")
            new_xyz, new_points = sample_and_group(self.npoint, self.radius, self.nsample, xyz, points)
            # new_points: (B, S, nsample, 3 + D)

        # to (B, C, nsample, S)
        new_points = new_points.permute(0, 3, 2, 1).contiguous()

        for conv, bn in zip(self.mlp_convs, self.mlp_bns):
            new_points = F.relu(bn(conv(new_points)))

        # max pool over nsample -> (B, mlp[-1], S)
        new_points = torch.max(new_points, dim=2)[0]
        # back to (B, S, mlp[-1])
        new_points = new_points.permute(0, 2, 1).contiguous()
        return new_xyz, new_points


class PointNetFeaturePropagation(nn.Module):
    def __init__(self, in_channel: int, mlp: list[int]):
        super().__init__()
        last_channel = in_channel
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel

    def forward(
        self,
        xyz1: torch.Tensor,
        xyz2: torch.Tensor,
        points1: torch.Tensor | None,
        points2: torch.Tensor
    ):
        """
        xyz1: (B, N, 3) target (dense)
        xyz2: (B, S, 3) source (sparse)
        points1: (B, N, D1) or None
        points2: (B, S, D2)
        return:
          new_points: (B, N, mlp[-1])
        """
        if xyz1.dim() != 3 or xyz2.dim() != 3:
            raise ValueError(f"FP expects xyz1/xyz2 3D, got xyz1={xyz1.shape}, xyz2={xyz2.shape}")
        if xyz1.size(-1) != 3 or xyz2.size(-1) != 3:
            raise ValueError(f"FP expects xyz last dim=3, got xyz1={xyz1.shape}, xyz2={xyz2.shape}")

        B, N, _ = xyz1.shape
        _, S, _ = xyz2.shape
        if N <= 0:
            raise ValueError("FP: N must be > 0")
        if S <= 0:
            raise ValueError("FP: S must be > 0")

        if points2.dim() != 3 or points2.shape[0] != B or points2.shape[1] != S:
            raise ValueError(f"FP: points2 must be (B,S,D2), got {points2.shape}")

        if S == 1:
            interpolated = points2.repeat(1, N, 1)  # (B,N,D2)
        else:
            dists = square_distance(xyz1, xyz2)  # (B,N,S)
            k = 3 if S >= 3 else S
            dists, idx = torch.topk(dists, k=k, dim=-1, largest=False, sorted=True)  # (B,N,k)
            dist_recip = 1.0 / (dists + 1e-10)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm  # (B,N,k)

            grouped_points = index_points(points2, idx)  # (B,N,k,D2)
            interpolated = torch.sum(grouped_points * weight.view(B, N, k, 1), dim=2)  # (B,N,D2)

        if points1 is not None:
            if points1.dim() != 3 or points1.shape[0] != B or points1.shape[1] != N:
                raise ValueError(f"FP: points1 must be (B,N,D1), got {points1.shape}")
            new_points = torch.cat([points1, interpolated], dim=-1)  # (B,N,D1+D2)
        else:
            new_points = interpolated  # (B,N,D2)

        # (B, D, N)
        new_points = new_points.permute(0, 2, 1).contiguous()
        for conv, bn in zip(self.mlp_convs, self.mlp_bns):
            new_points = F.relu(bn(conv(new_points)))
        # back to (B, N, D)
        new_points = new_points.permute(0, 2, 1).contiguous()
        return new_points


# -----------------------------
# PointNet++ Segmentation Model
# -----------------------------
class PointNet2Seg(nn.Module):
    """
    Input: x in either (B,N,C) or (B,C,N), where C = in_channels (default 6 = xyz+normal)
    Output: logits (B,N,num_classes)
    """
    def __init__(
        self,
        in_channels: int = 6,
        num_classes: int = 16,
        sa_npoints=(2048, 512, 128),
        sa_radii=(0.05, 0.10, 0.20),
        sa_nsamples=(32, 32, 32),
        dropout: float = 0.5
    ):
        super().__init__()
        assert in_channels >= 3, "in_channels must include xyz (>=3)"
        assert len(sa_npoints) == 3 and len(sa_radii) == 3 and len(sa_nsamples) == 3, "Expect 3 SA stages before global"
        self.in_channels = int(in_channels)
        self.num_classes = int(num_classes)

        # l0 features are everything except xyz
        l0_feat_dim = self.in_channels - 3

        # SA layers (SSG)
        self.sa1 = PointNetSetAbstraction(
            npoint=int(sa_npoints[0]),
            radius=float(sa_radii[0]),
            nsample=int(sa_nsamples[0]),
            in_channel=3 + l0_feat_dim,
            mlp=[64, 64, 128],
            group_all=False
        )
        self.sa2 = PointNetSetAbstraction(
            npoint=int(sa_npoints[1]),
            radius=float(sa_radii[1]),
            nsample=int(sa_nsamples[1]),
            in_channel=3 + 128,
            mlp=[128, 128, 256],
            group_all=False
        )
        self.sa3 = PointNetSetAbstraction(
            npoint=int(sa_npoints[2]),
            radius=float(sa_radii[2]),
            nsample=int(sa_nsamples[2]),
            in_channel=3 + 256,
            mlp=[256, 256, 512],
            group_all=False
        )
        self.sa4 = PointNetSetAbstraction(
            npoint=None,
            radius=None,
            nsample=None,
            in_channel=3 + 512,
            mlp=[512, 1024],
            group_all=True
        )

        # FP layers
        self.fp4 = PointNetFeaturePropagation(in_channel=1024 + 512, mlp=[512, 512])
        self.fp3 = PointNetFeaturePropagation(in_channel=512 + 256, mlp=[256, 256])
        self.fp2 = PointNetFeaturePropagation(in_channel=256 + 128, mlp=[256, 128])
        self.fp1 = PointNetFeaturePropagation(in_channel=128 + l0_feat_dim, mlp=[128, 128, 128])

        # classifier
        self.conv1 = nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(float(dropout))
        self.conv2 = nn.Conv1d(128, self.num_classes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # accept (B,N,C) or (B,C,N)
        if x.dim() != 3:
            raise ValueError(f"Expected 3D tensor input, got {tuple(x.shape)}")

        # normalize to (B,N,C)
        if x.shape[1] == self.in_channels and x.shape[2] != self.in_channels:
            x = x.permute(0, 2, 1).contiguous()

        if x.shape[-1] != self.in_channels:
            raise ValueError(f"in_channels mismatch: expected {self.in_channels}, got {x.shape[-1]}")

        xyz = x[:, :, 0:3].contiguous()  # (B,N,3)
        points = x[:, :, 3:].contiguous() if self.in_channels > 3 else None  # (B,N,D)

        l0_xyz = xyz
        l0_points = points

        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)  # (B,*,3), (B,*,128)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)  # (B,*,3), (B,*,256)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)  # (B,*,3), (B,*,512)
        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)  # (B,1,3), (B,1,1024)

        # FP (coarse -> fine)
        l3_points = self.fp4(l3_xyz, l4_xyz, l3_points, l4_points)
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(l0_xyz, l1_xyz, l0_points, l1_points)  # (B,N,128)

        # classifier: (B,128,N) -> (B,num_classes,N) -> (B,N,num_classes)
        feat = l0_points.permute(0, 2, 1).contiguous()  # (B,128,N)
        feat = F.relu(self.bn1(self.conv1(feat)))
        feat = self.drop1(feat)
        logits = self.conv2(feat)  # (B,num_classes,N)
        logits = logits.permute(0, 2, 1).contiguous()  # (B,N,num_classes)
        return logits
