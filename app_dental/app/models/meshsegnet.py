# models/meshsegnet.py

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


# ============================================================
# Sparse / Dense SpMM helpers
# ============================================================
def _spmm_one(A: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
    """
    A: sparse COO (N,N)  (recommended values float32)
    X: dense (N,C) can be fp16 under autocast
    return: (N,C) dtype = X.dtype
    """
    # CUDA sparse addmm is not implemented for fp16/half -> force fp32 just here
    A32 = A.coalesce().float()
    X32 = X.float()

    # IMPORTANT: disable autocast inside sparse.mm so it doesn't cast back to fp16
    if X.is_cuda:
        with torch.amp.autocast("cuda", enabled=False):
            Y32 = torch.sparse.mm(A32, X32)
    else:
        Y32 = torch.sparse.mm(A32, X32)

    return Y32.to(dtype=X.dtype)


def _spmm_batch(A, X: torch.Tensor) -> torch.Tensor:
    """
    A:  (B,N,N) dense OR list[torch.sparse_coo_tensor(N,N)] OR sparse_coo (N,N) when B=1
    X:  (B,N,C) dense
    return: (B,N,C)
    """
    # dense path (original behavior)
    if torch.is_tensor(A) and (not A.is_sparse):
        return torch.bmm(A, X)

    B, N, C = X.shape
    out = []

    if torch.is_tensor(A) and A.is_sparse:
        # single sparse (assume B==1)
        out.append(_spmm_one(A, X[0]))
    else:
        # list sparse per batch
        for b in range(B):
            out.append(_spmm_one(A[b], X[b]))

    return torch.stack(out, dim=0)


# ============================================================
# STN modules
# ============================================================
class STN3d(nn.Module):
    def __init__(self, channel: int):
        super().__init__()
        self.conv1 = nn.Conv1d(channel, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B,C,N)
        return: (B,3,3)
        """
        batchsize = x.size(0)

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(
            torch.from_numpy(np.array([1, 0, 0, 0, 1, 0, 0, 0, 1], dtype=np.float32))
        ).view(1, 9).repeat(batchsize, 1)

        if x.is_cuda:
            iden = iden.to(x.get_device())
        x = x + iden
        x = x.view(-1, 3, 3)
        return x


class STNkd(nn.Module):
    def __init__(self, k: int = 64):
        super().__init__()
        self.conv1 = nn.Conv1d(k, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 512, 1)
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, k * k)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(512)
        self.bn4 = nn.BatchNorm1d(256)
        self.bn5 = nn.BatchNorm1d(128)

        self.k = k

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B,K,N)
        return: (B,K,K)
        """
        batchsize = x.size(0)

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 512)

        # BN can be unstable/crash when batchsize==1 (training mode)
        if self.training and batchsize == 1:
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
        else:
            x = F.relu(self.bn4(self.fc1(x)))
            x = F.relu(self.bn5(self.fc2(x)))

        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.eye(self.k, dtype=np.float32).flatten())).view(
            1, self.k * self.k
        ).repeat(batchsize, 1)

        if x.is_cuda:
            iden = iden.to(x.get_device())
        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x


# ============================================================
# MeshSegNet
# ============================================================
class MeshSegNet(nn.Module):
    def __init__(
        self,
        num_classes: int = 15,
        num_channels: int = 15,
        with_dropout: bool = True,
        dropout_p: float = 0.5,
    ):
        super().__init__()
        self.num_classes = int(num_classes)
        self.num_channels = int(num_channels)
        self.with_dropout = bool(with_dropout)
        self.dropout_p = float(dropout_p)

        # MLP-1 [64, 64]
        self.mlp1_conv1 = nn.Conv1d(self.num_channels, 64, 1)
        self.mlp1_conv2 = nn.Conv1d(64, 64, 1)
        self.mlp1_bn1 = nn.BatchNorm1d(64)
        self.mlp1_bn2 = nn.BatchNorm1d(64)

        # FTM (feature-transformer module)
        self.fstn = STNkd(k=64)

        # GLM-1
        self.glm1_conv1_1 = nn.Conv1d(64, 32, 1)
        self.glm1_conv1_2 = nn.Conv1d(64, 32, 1)
        self.glm1_bn1_1 = nn.BatchNorm1d(32)
        self.glm1_bn1_2 = nn.BatchNorm1d(32)
        self.glm1_conv2 = nn.Conv1d(32 + 32, 64, 1)
        self.glm1_bn2 = nn.BatchNorm1d(64)

        # MLP-2
        self.mlp2_conv1 = nn.Conv1d(64, 64, 1)
        self.mlp2_bn1 = nn.BatchNorm1d(64)
        self.mlp2_conv2 = nn.Conv1d(64, 128, 1)
        self.mlp2_bn2 = nn.BatchNorm1d(128)
        self.mlp2_conv3 = nn.Conv1d(128, 512, 1)
        self.mlp2_bn3 = nn.BatchNorm1d(512)

        # GLM-2
        self.glm2_conv1_1 = nn.Conv1d(512, 128, 1)
        self.glm2_conv1_2 = nn.Conv1d(512, 128, 1)
        self.glm2_conv1_3 = nn.Conv1d(512, 128, 1)
        self.glm2_bn1_1 = nn.BatchNorm1d(128)
        self.glm2_bn1_2 = nn.BatchNorm1d(128)
        self.glm2_bn1_3 = nn.BatchNorm1d(128)
        self.glm2_conv2 = nn.Conv1d(128 * 3, 512, 1)
        self.glm2_bn2 = nn.BatchNorm1d(512)

        # MLP-3
        self.mlp3_conv1 = nn.Conv1d(64 + 512 + 512 + 512, 256, 1)
        self.mlp3_conv2 = nn.Conv1d(256, 256, 1)
        self.mlp3_bn1_1 = nn.BatchNorm1d(256)
        self.mlp3_bn1_2 = nn.BatchNorm1d(256)
        self.mlp3_conv3 = nn.Conv1d(256, 128, 1)
        self.mlp3_conv4 = nn.Conv1d(128, 128, 1)
        self.mlp3_bn2_1 = nn.BatchNorm1d(128)
        self.mlp3_bn2_2 = nn.BatchNorm1d(128)

        # output
        self.output_conv = nn.Conv1d(128, self.num_classes, 1)
        if self.with_dropout:
            self.dropout = nn.Dropout(p=self.dropout_p)

    def forward(self, x: torch.Tensor, a_s, a_l) -> torch.Tensor:
        """
        x:   (B,C,N)
        a_s: dense (B,N,N) OR list[sparse(N,N)] OR sparse(N,N) when B=1
        a_l: dense (B,N,N) OR list[sparse(N,N)] OR sparse(N,N) when B=1
        return: logits (B,N,num_classes)  <-- NO SOFTMAX
        """
        n_pts = x.size(2)

        # MLP-1
        x = F.relu(self.mlp1_bn1(self.mlp1_conv1(x)))
        x = F.relu(self.mlp1_bn2(self.mlp1_conv2(x)))

        # FTM
        trans_feat = self.fstn(x)             # (B,64,64)
        x = x.transpose(2, 1)                 # (B,N,64)
        x_ftm = torch.bmm(x, trans_feat)      # (B,N,64)

        # GLM-1: sap = A_s @ x_ftm
        sap = _spmm_batch(a_s, x_ftm)         # (B,N,64)
        sap = sap.transpose(2, 1)             # (B,64,N)
        x_ftm = x_ftm.transpose(2, 1)         # (B,64,N)

        x = F.relu(self.glm1_bn1_1(self.glm1_conv1_1(x_ftm)))
        glm_1_sap = F.relu(self.glm1_bn1_2(self.glm1_conv1_2(sap)))
        x = torch.cat([x, glm_1_sap], dim=1)
        x = F.relu(self.glm1_bn2(self.glm1_conv2(x)))

        # MLP-2
        x = F.relu(self.mlp2_bn1(self.mlp2_conv1(x)))
        x = F.relu(self.mlp2_bn2(self.mlp2_conv2(x)))
        x_mlp2 = F.relu(self.mlp2_bn3(self.mlp2_conv3(x)))  # (B,512,N)
        if self.with_dropout:
            x_mlp2 = self.dropout(x_mlp2)

        # GLM-2: sap_1 = A_s @ x_mlp2, sap_2 = A_l @ x_mlp2
        x_mlp2 = x_mlp2.transpose(2, 1)       # (B,N,512)

        sap_1 = _spmm_batch(a_s, x_mlp2)      # (B,N,512)
        sap_2 = _spmm_batch(a_l, x_mlp2)      # (B,N,512)

        x_mlp2 = x_mlp2.transpose(2, 1)       # (B,512,N)
        sap_1 = sap_1.transpose(2, 1)         # (B,512,N)
        sap_2 = sap_2.transpose(2, 1)         # (B,512,N)

        x = F.relu(self.glm2_bn1_1(self.glm2_conv1_1(x_mlp2)))
        glm_2_sap_1 = F.relu(self.glm2_bn1_2(self.glm2_conv1_2(sap_1)))
        glm_2_sap_2 = F.relu(self.glm2_bn1_3(self.glm2_conv1_3(sap_2)))
        x = torch.cat([x, glm_2_sap_1, glm_2_sap_2], dim=1)
        x_glm2 = F.relu(self.glm2_bn2(self.glm2_conv2(x)))  # (B,512,N)

        # GMP
        x = torch.max(x_glm2, 2, keepdim=True)[0]           # (B,512,1)

        # Upsample back to N
        x = torch.nn.Upsample(n_pts)(x)                     # (B,512,N)

        # Dense fusion
        x = torch.cat([x, x_ftm, x_mlp2, x_glm2], dim=1)    # (B,64+512+512+512,N)

        # MLP-3
        x = F.relu(self.mlp3_bn1_1(self.mlp3_conv1(x)))
        x = F.relu(self.mlp3_bn1_2(self.mlp3_conv2(x)))
        x = F.relu(self.mlp3_bn2_1(self.mlp3_conv3(x)))
        if self.with_dropout:
            x = self.dropout(x)
        x = F.relu(self.mlp3_bn2_2(self.mlp3_conv4(x)))

        # output logits
        x = self.output_conv(x)                  # (B,C,N)
        x = x.transpose(2, 1).contiguous()       # (B,N,C)
        return x
