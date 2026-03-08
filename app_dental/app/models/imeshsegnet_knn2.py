from __future__ import annotations

from typing import Dict, Any, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_cluster import knn_graph
except Exception:
    knn_graph = None


# ============================================================
# Utils
# ============================================================
def _require_knn():
    if knn_graph is None:
        raise ImportError("Need torch-cluster for knn_graph. Please install torch-cluster.")


def _act(name: str) -> nn.Module:
    n = (name or "relu").lower().strip()
    if n == "gelu":
        return nn.GELU()
    if n in ("silu", "swish"):
        return nn.SiLU()
    return nn.ReLU(inplace=True)


class Norm1d(nn.Module):
    """
    Stable normalization for small/medium batch:
      - default: GroupNorm (more stable than BatchNorm for tooth meshes)
    """
    def __init__(self, c: int, norm: str = "gn", gn_groups: int = 8):
        super().__init__()
        norm = (norm or "gn").lower().strip()
        c = int(c)
        if norm in ("bn", "batchnorm", "batch_norm"):
            self.n = nn.BatchNorm1d(c)
        elif norm in ("ln", "layernorm", "layer_norm"):
            self.n = nn.LayerNorm(c)
        else:
            g = max(1, min(int(gn_groups), c))
            while c % g != 0 and g > 1:
                g -= 1
            self.n = nn.GroupNorm(g, c)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # accepts (B,C,N) for BN/GN, or (B,N,C) for LN
        if isinstance(self.n, nn.LayerNorm):
            return self.n(x)
        return self.n(x)


class ConvBNAct(nn.Module):
    def __init__(self, cin: int, cout: int, *, act: str = "relu", norm: str = "gn", p_drop: float = 0.0):
        super().__init__()
        self.conv = nn.Conv1d(int(cin), int(cout), 1, bias=True)
        self.norm = Norm1d(int(cout), norm=norm)
        self.act = _act(act)
        self.drop = nn.Dropout(p=float(p_drop)) if float(p_drop) > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,C,N)
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.drop(x)
        return x


class MLP(nn.Module):
    def __init__(self, cin: int, hidden: int, cout: int, *, act: str = "relu", p_drop: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(int(cin), int(hidden), bias=True),
            _act(act),
            nn.Dropout(p=float(p_drop)) if float(p_drop) > 0 else nn.Identity(),
            nn.Linear(int(hidden), int(cout), bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ============================================================
# EdgeConv (DGCNN-style)
# ============================================================
class EdgeConv(nn.Module):
    """
    EdgeConv:
      m_ij = MLP([x_i, x_j - x_i])
      y_i = max_j m_ij
    """

    def __init__(self, cin: int, cout: int, *, act: str = "relu", p_drop: float = 0.0, hidden: Optional[int] = None):
        super().__init__()
        cin = int(cin)
        cout = int(cout)
        h = int(hidden) if hidden is not None else max(64, cout)
        self.mlp = MLP(cin * 2, h, cout, act=act, p_drop=p_drop)

    def forward(self, x_bnC: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        x_bnC: (BN, Cin)
        edge_index: (2, E) where row=dst, col=src
        return: (BN, Cout)
        """
        dst = edge_index[0]
        src = edge_index[1]
        xi = x_bnC[dst]               # (E,C)
        xj = x_bnC[src]               # (E,C)
        feat = torch.cat([xi, xj - xi], dim=1)  # (E,2C)
        msg = self.mlp(feat)          # (E,Cout)

        BN = int(x_bnC.shape[0])
        Cout = int(msg.shape[1])

        # scatter max over dst
        out = torch.full((BN, Cout), -float("inf"), device=msg.device, dtype=msg.dtype)
        idx = dst.view(-1, 1).expand(-1, Cout)
        out.scatter_reduce_(0, idx, msg, reduce="amax", include_self=True)

        # replace -inf (isolated) with 0
        out = torch.where(torch.isfinite(out), out, torch.zeros_like(out))
        return out


def build_knn_edge_index(pos_bnc: torch.Tensor, k: int, *, loop: bool = True) -> torch.Tensor:
    """
    pos_bnc: (B,N,3)
    return edge_index (2,E) over flattened BN nodes with batch-aware knn
    """
    _require_knn()
    B, N, _ = pos_bnc.shape
    pos = pos_bnc.reshape(B * N, 3)
    batch = torch.arange(B, device=pos.device).repeat_interleave(N)
    # torch_cluster convention: edge_index[0]=dst, edge_index[1]=src
    edge_index = knn_graph(pos, k=int(k), batch=batch, loop=bool(loop))
    return edge_index


# ============================================================
# iMeshSegNet (TS-MDL-like) with EdgeConv multi-scale
# ============================================================
class IMeshSegNetKNNBatch(nn.Module):
    """
    Closer to TS-MDL/iMeshSegNet:
      - input: x (B,N,15), pos (B,N,3)
      - Stem: 15 -> 64 -> 64
      - GLM-1: EdgeConv(k=6) + local fusion -> 64
      - MLP-2: 64 -> 64 -> 128 -> 512
      - GLM-2: EdgeConv(k=6) and EdgeConv(k=12) -> 128 each + local 128 -> fuse -> 512
      - Global max pool -> upsample -> dense fusion -> MLP-3 -> output
    """

    def __init__(
        self,
        *,
        num_classes: int = 17,
        in_channels: int = 15,
        # kNN (paper-like)
        knn_s: int = 6,
        knn_l: int = 12,
        knn_loop: bool = True,
        # regularization/stability
        dropout_p: float = 0.3,
        act: str = "relu",
        norm: str = "gn",          # "gn" recommended for stability
        gn_groups: int = 8,
        # edgeconv widths
        edge_hidden: int = 128,
    ):
        super().__init__()
        self.num_classes = int(num_classes)
        self.in_channels = int(in_channels)

        self.knn_s = int(knn_s)
        self.knn_l = int(knn_l)
        self.knn_loop = bool(knn_loop)

        self.dropout_p = float(dropout_p)
        self.act_name = act
        self.norm = norm
        self.gn_groups = int(gn_groups)
        self.edge_hidden = int(edge_hidden)

        # ---- Stem (15 -> 64 -> 64) ----
        self.stem1 = ConvBNAct(self.in_channels, 64, act=act, norm=norm, p_drop=0.0)
        self.stem2 = ConvBNAct(64, 64, act=act, norm=norm, p_drop=0.0)

        # ---- GLM-1: EdgeConv(k=6) producing 32, fuse -> 64 ----
        self.ec1 = EdgeConv(64, 32, act=act, p_drop=0.0, hidden=self.edge_hidden)
        self.glm1_fuse = nn.Sequential(
            nn.Conv1d(64 + 32, 64, 1, bias=True),
            Norm1d(64, norm=norm, gn_groups=self.gn_groups),
            _act(act),
        )

        # ---- MLP-2: 64 -> 64 -> 128 -> 512 ----
        self.mlp2_1 = ConvBNAct(64, 64, act=act, norm=norm, p_drop=0.0)
        self.mlp2_2 = ConvBNAct(64, 128, act=act, norm=norm, p_drop=0.0)
        self.mlp2_3 = ConvBNAct(128, 512, act=act, norm=norm, p_drop=self.dropout_p)

        # ---- GLM-2: EdgeConv(k=6) + EdgeConv(k=12) each -> 128, plus local 128 -> fuse -> 512 ----
        self.local2 = nn.Sequential(
            nn.Conv1d(512, 128, 1, bias=True),
            Norm1d(128, norm=norm, gn_groups=self.gn_groups),
            _act(act),
        )
        self.ec2_s = EdgeConv(512, 128, act=act, p_drop=0.0, hidden=self.edge_hidden)
        self.ec2_l = EdgeConv(512, 128, act=act, p_drop=0.0, hidden=self.edge_hidden)
        self.glm2_fuse = nn.Sequential(
            nn.Conv1d(128 * 3, 512, 1, bias=True),
            Norm1d(512, norm=norm, gn_groups=self.gn_groups),
            _act(act),
        )

        # ---- Global max pool + upsample ----
        # (B,512,N) -> (B,512,1) -> upsample to N
        # ---- Dense fusion (like MeshSegNet) ----
        self.mlp3_1 = ConvBNAct(64 + 512 + 512 + 512, 256, act=act, norm=norm, p_drop=0.0)
        self.mlp3_2 = ConvBNAct(256, 256, act=act, norm=norm, p_drop=0.0)
        self.mlp3_3 = ConvBNAct(256, 128, act=act, norm=norm, p_drop=self.dropout_p)
        self.mlp3_4 = ConvBNAct(128, 128, act=act, norm=norm, p_drop=0.0)

        self.out = nn.Conv1d(128, self.num_classes, 1, bias=True)

    def forward(self, batch: Dict[str, Any] | torch.Tensor, **kwargs) -> torch.Tensor:
        """
        train.py sends a dict batch in forward_type='batch'.
        Required:
          batch['x']: (B,N,15)
          batch['pos']: (B,N,3)
        Return:
          logits (B,N,C)
        """
        if isinstance(batch, dict):
            x = batch["x"]
            pos = batch["pos"]
        else:
            # allow direct call: forward(x=..., pos=...)
            x = batch
            pos = kwargs.get("pos", None)

        if pos is None:
            raise ValueError("IMeshSegNetKNNBatch requires 'pos' (B,N,3).")

        if x.dim() != 3:
            raise ValueError(f"x must be (B,N,C), got {tuple(x.shape)}")
        if pos.dim() != 3 or pos.size(-1) != 3:
            raise ValueError(f"pos must be (B,N,3), got {tuple(pos.shape)}")
        if x.size(-1) != self.in_channels:
            raise ValueError(f"in_channels mismatch: expected {self.in_channels}, got {x.size(-1)}")
        if x.size(0) != pos.size(0) or x.size(1) != pos.size(1):
            raise ValueError(f"B/N mismatch: x={tuple(x.shape)} pos={tuple(pos.shape)}")

        B, N, _ = x.shape

        # build edges (batch-aware)
        e_s = build_knn_edge_index(pos, k=self.knn_s, loop=self.knn_loop)
        e_l = build_knn_edge_index(pos, k=self.knn_l, loop=self.knn_loop)

        # (B,N,C) -> (B,C,N)
        x0 = x.permute(0, 2, 1).contiguous()

        # Stem
        f = self.stem1(x0)   # (B,64,N)
        f = self.stem2(f)    # (B,64,N)

        # GLM-1 EdgeConv(k=6)
        f_bnC = f.permute(0, 2, 1).reshape(B * N, 64)  # (BN,64)
        ec1 = self.ec1(f_bnC, e_s).reshape(B, N, 32).permute(0, 2, 1).contiguous()  # (B,32,N)
        f_glm1 = self.glm1_fuse(torch.cat([f, ec1], dim=1))  # (B,64,N)

        # MLP-2
        f2 = self.mlp2_1(f_glm1)   # (B,64,N)
        f2 = self.mlp2_2(f2)       # (B,128,N)
        f512 = self.mlp2_3(f2)     # (B,512,N)

        # GLM-2 multi-scale EdgeConv on 512
        f512_bnC = f512.permute(0, 2, 1).reshape(B * N, 512)  # (BN,512)
        ec2s = self.ec2_s(f512_bnC, e_s).reshape(B, N, 128).permute(0, 2, 1).contiguous()
        ec2l = self.ec2_l(f512_bnC, e_l).reshape(B, N, 128).permute(0, 2, 1).contiguous()
        loc2 = self.local2(f512)  # (B,128,N)

        f_glm2 = self.glm2_fuse(torch.cat([loc2, ec2s, ec2l], dim=1))  # (B,512,N)

        # Global max pool and upsample to N
        g = torch.max(f_glm2, dim=2, keepdim=True).values  # (B,512,1)
        g_up = g.expand(-1, -1, N)                         # (B,512,N)

        # Dense fusion like MeshSegNet
        # concat: [g_up, f_glm1(64), f512(512), f_glm2(512)]
        z = torch.cat([g_up, f_glm1, f512, f_glm2], dim=1)  # (B,64+512+512+512,N)

        z = self.mlp3_1(z)
        z = self.mlp3_2(z)
        z = self.mlp3_3(z)
        z = self.mlp3_4(z)

        logits = self.out(z)                      # (B,C,N)
        logits = logits.permute(0, 2, 1).contiguous()  # (B,N,C)
        return logits