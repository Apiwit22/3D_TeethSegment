from __future__ import annotations

from typing import Dict, Any, List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


def _to_bool_mask(valid_face: Optional[torch.Tensor], x: torch.Tensor) -> torch.Tensor:
    """
    Return bool mask shape (B, F)
    """
    if valid_face is None:
        return torch.ones(x.shape[:2], dtype=torch.bool, device=x.device)
    return valid_face.bool()


def _split_streams(
    batch: Dict[str, Any],
    coord_channels: int = 12,
    normal_channels: int = 12,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Prefer x_c / x_n if present.
    Fallback to split x[..., :12] and x[..., 12:24].
    """
    x_c = batch.get("x_c", None)
    x_n = batch.get("x_n", None)

    if x_c is not None and x_n is not None:
        return x_c.float(), x_n.float()

    x = batch["x"].float()
    need = coord_channels + normal_channels
    if x.shape[-1] < need:
        raise ValueError(
            f"FastTGCN expects at least {need} channels, got x.shape={tuple(x.shape)}"
        )
    return x[..., :coord_channels], x[..., coord_channels:coord_channels + normal_channels]


def _mean_aggregate_from_edge_index(
    x_i: torch.Tensor,       # (F, C)
    edge_index_i: torch.Tensor,  # (2, E)
    f_used: int,
) -> torch.Tensor:
    """
    Mean neighbor aggregation for one sample.
    dst receives messages from src.
    """
    F_all, C = x_i.shape
    f_used = int(max(0, min(f_used, F_all)))

    if f_used == 0:
        return torch.zeros_like(x_i)

    out = torch.zeros_like(x_i)
    deg = torch.zeros((F_all, 1), dtype=x_i.dtype, device=x_i.device)

    if edge_index_i.numel() == 0:
        return out

    src = edge_index_i[0].long()
    dst = edge_index_i[1].long()

    keep = (src >= 0) & (src < f_used) & (dst >= 0) & (dst < f_used)
    src = src[keep]
    dst = dst[keep]

    if src.numel() == 0:
        return out

    out.index_add_(0, dst, x_i[src])
    deg.index_add_(0, dst, torch.ones((dst.numel(), 1), dtype=x_i.dtype, device=x_i.device))
    out = out / deg.clamp_min_(1.0)
    return out


def _mean_aggregate_from_nbr(
    x_i: torch.Tensor,   # (F, C)
    nbr_i: torch.Tensor, # (F, K)
    f_used: int,
) -> torch.Tensor:
    """
    Fallback aggregation using neighbor indices from face-mode loader.
    """
    F_all, C = x_i.shape
    f_used = int(max(0, min(f_used, F_all)))

    out = torch.zeros_like(x_i)
    if f_used == 0:
        return out

    nbr = nbr_i[:f_used].long().clone()
    nbr = nbr.clamp(0, max(f_used - 1, 0))
    neigh = x_i[nbr]                 # (F_used, K, C)
    out[:f_used] = neigh.mean(dim=1)
    return out


class MLP(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_ch, out_ch),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, F, C)
        B, F, C = x.shape
        y = self.net(x.reshape(B * F, C))
        return y.reshape(B, F, -1)


class GraphConvMean(nn.Module):
    """
    Simple graph conv:
      h' = MLP([h, mean_neigh(h)])
    Works with:
      - edge_index: list[(2,E)] from collate_graph
      - nbr: (B,F,K) from collate_face
    """
    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.0):
        super().__init__()
        self.mlp = MLP(in_ch * 2, out_ch, dropout=dropout)

    def forward(
        self,
        x: torch.Tensor,                       # (B,F,C)
        *,
        edge_index: Optional[List[torch.Tensor]] = None,
        nbr: Optional[torch.Tensor] = None,
        f_used: Optional[Sequence[int]] = None,
    ) -> torch.Tensor:
        B, F, C = x.shape
        if f_used is None:
            f_used = [F] * B

        aggr_list: List[torch.Tensor] = []
        for i in range(B):
            xi = x[i]
            fi = int(f_used[i])

            if edge_index is not None:
                agg_i = _mean_aggregate_from_edge_index(xi, edge_index[i].to(x.device), fi)
            elif nbr is not None:
                agg_i = _mean_aggregate_from_nbr(xi, nbr[i].to(x.device), fi)
            else:
                raise ValueError("FastTGCN needs either edge_index (graph mode) or nbr (face mode).")

            aggr_list.append(agg_i)

        aggr = torch.stack(aggr_list, dim=0)  # (B,F,C)
        return self.mlp(torch.cat([x, aggr], dim=-1))


class FastTGCNBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.0):
        super().__init__()
        self.conv1 = GraphConvMean(in_ch, out_ch, dropout=dropout)
        self.conv2 = GraphConvMean(out_ch, out_ch, dropout=dropout)

        if in_ch != out_ch:
            self.proj = nn.Linear(in_ch, out_ch)
        else:
            self.proj = nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        *,
        edge_index: Optional[List[torch.Tensor]],
        nbr: Optional[torch.Tensor],
        f_used: Sequence[int],
    ) -> torch.Tensor:
        res = self.proj(x)
        x = self.conv1(x, edge_index=edge_index, nbr=nbr, f_used=f_used)
        x = self.conv2(x, edge_index=edge_index, nbr=nbr, f_used=f_used)
        return x + res


class FastTGCN(nn.Module):
    """
    Practical Fast-TGCN reimplementation for your dataloader.

    Expected batch keys:
      - x or (x_c, x_n)
      - valid_face or mask
      - F_used
      - edge_index (graph mode, preferred) OR nbr (face mode fallback)

    Output:
      - logits: (B, F, num_classes)
    """
    def __init__(
        self,
        in_channels: int = 24,
        coord_channels: int = 12,
        normal_channels: int = 12,
        num_classes: int = 17,
        stem_channels: int = 64,
        block_channels: Sequence[int] = (64, 128, 128, 256),
        classifier_hidden: int = 128,
        dropout: float = 0.1,
        use_global_context: bool = True,
    ):
        super().__init__()

        if in_channels < coord_channels + normal_channels:
            raise ValueError(
                f"in_channels={in_channels} is smaller than "
                f"coord+normal={coord_channels + normal_channels}"
            )

        self.in_channels = int(in_channels)
        self.coord_channels = int(coord_channels)
        self.normal_channels = int(normal_channels)
        self.num_classes = int(num_classes)
        self.use_global_context = bool(use_global_context)

        self.coord_stem = MLP(coord_channels, stem_channels, dropout=dropout)
        self.normal_stem = MLP(normal_channels, stem_channels, dropout=dropout)
        self.fuse = MLP(stem_channels * 2, stem_channels, dropout=dropout)

        blocks = []
        prev = stem_channels
        for ch in block_channels:
            blocks.append(FastTGCNBlock(prev, int(ch), dropout=dropout))
            prev = int(ch)
        self.blocks = nn.ModuleList(blocks)

        head_in = prev * 2 if self.use_global_context else prev
        self.head = nn.Sequential(
            MLP(head_in, classifier_hidden, dropout=dropout),
            nn.Linear(classifier_hidden, num_classes),
        )

    def _masked_global_max(self, x: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        # x: (B,F,C), valid: (B,F)
        B, F, C = x.shape
        neg = torch.finfo(x.dtype).min
        x_masked = x.masked_fill(~valid.unsqueeze(-1), neg)
        g = x_masked.max(dim=1).values
        # for safety if a sample is entirely invalid
        bad = ~valid.any(dim=1)
        if bad.any():
            g[bad] = 0
        return g

    def forward(self, batch: Dict[str, Any]) -> torch.Tensor:
        x_c, x_n = _split_streams(
            batch,
            coord_channels=self.coord_channels,
            normal_channels=self.normal_channels,
        )

        valid_face = batch.get("valid_face", None)
        if valid_face is None:
            valid_face = batch.get("mask", None)
        valid = _to_bool_mask(valid_face, x_c)  # (B,F)

        f_used = batch.get("F_used", None)
        if f_used is None:
            f_used = batch.get("F_used_t", None)
        if torch.is_tensor(f_used):
            f_used = [int(v) for v in f_used.detach().cpu().tolist()]
        elif f_used is None:
            f_used = [x_c.shape[1]] * x_c.shape[0]
        else:
            f_used = [int(v) for v in f_used]

        edge_index = batch.get("edge_index", None)
        nbr = batch.get("nbr", None)

        h_c = self.coord_stem(x_c)
        h_n = self.normal_stem(x_n)
        x = self.fuse(torch.cat([h_c, h_n], dim=-1))

        for block in self.blocks:
            x = block(x, edge_index=edge_index, nbr=nbr, f_used=f_used)

        if self.use_global_context:
            g = self._masked_global_max(x, valid)           # (B,C)
            g = g.unsqueeze(1).expand(-1, x.shape[1], -1)   # (B,F,C)
            x = torch.cat([x, g], dim=-1)

        B, F, C = x.shape
        logits = self.head[0](x)
        logits = self.head[1](logits.reshape(B * F, -1)).reshape(B, F, self.num_classes)

        # zero out padded faces for cleaner downstream behavior
        logits = logits.masked_fill(~valid.unsqueeze(-1), 0.0)
        return logits