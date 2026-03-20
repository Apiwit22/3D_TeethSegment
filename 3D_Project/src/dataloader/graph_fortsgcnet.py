# src/dataloader/graph_fortsgcnet.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import hashlib

import numpy as np
import torch

try:
    from scipy.spatial import cKDTree  # type: ignore
except Exception:
    cKDTree = None

from src.dataloader.faces_twostream24 import TwoStream24FaceDataset
from src.dataloader.faces_base import FaceConfig, build_face_neighbors


# ============================================================
# KNN helpers
# ============================================================
def _knn_indices_numpy(feat: np.ndarray, k: int) -> np.ndarray:
    """
    Return nbr indices (F, k), self excluded if possible.
    Fallback pure numpy if scipy is unavailable.
    """
    feat = np.asarray(feat, dtype=np.float32)
    F = int(feat.shape[0])
    k = int(k)

    if F <= 0 or k <= 0:
        return np.zeros((F, 0), dtype=np.int64)

    k_eff = min(k, max(F - 1, 0))
    if k_eff <= 0:
        out = np.arange(F, dtype=np.int64)[:, None]
        return np.repeat(out, k, axis=1)

    # pairwise squared distance
    # (F,1,C) - (1,F,C) -> (F,F,C)
    diff = feat[:, None, :] - feat[None, :, :]
    dist2 = np.sum(diff * diff, axis=-1)

    # exclude self
    np.fill_diagonal(dist2, np.inf)

    # top-k nearest
    idx = np.argpartition(dist2, kth=k_eff - 1, axis=1)[:, :k_eff]

    # sort those k by real distance for determinism
    row = np.arange(F)[:, None]
    idx = idx[row, np.argsort(dist2[row, idx], axis=1)]

    if k_eff < k:
        pad = np.repeat(np.arange(F, dtype=np.int64)[:, None], k - k_eff, axis=1)
        idx = np.concatenate([idx, pad], axis=1)

    return idx.astype(np.int64, copy=False)


def _knn_indices_ckdtree(feat: np.ndarray, k: int) -> np.ndarray:
    feat = np.asarray(feat, dtype=np.float32)
    F = int(feat.shape[0])
    k = int(k)

    if F <= 0 or k <= 0:
        return np.zeros((F, 0), dtype=np.int64)

    k_eff = min(k, max(F - 1, 0))
    if k_eff <= 0:
        out = np.arange(F, dtype=np.int64)[:, None]
        return np.repeat(out, k, axis=1)

    tree = cKDTree(feat)
    # query self + k neighbors
    _, idx = tree.query(feat, k=min(k_eff + 1, F))
    idx = np.asarray(idx)

    if idx.ndim == 1:
        idx = idx[:, None]

    # drop self (first col usually self)
    if idx.shape[1] > 1:
        idx = idx[:, 1:]
    else:
        idx = np.empty((F, 0), dtype=np.int64)

    if idx.shape[1] < k:
        pad = np.repeat(np.arange(F, dtype=np.int64)[:, None], k - idx.shape[1], axis=1)
        idx = np.concatenate([idx, pad], axis=1)

    return idx[:, :k].astype(np.int64, copy=False)


def _build_knn_nbr(feat: np.ndarray, k: int) -> np.ndarray:
    if cKDTree is not None:
        return _knn_indices_ckdtree(feat, k)
    return _knn_indices_numpy(feat, k)


def _sanitize_nbr_np(nbr: np.ndarray, F: int) -> np.ndarray:
    nbr = np.asarray(nbr, dtype=np.int64)
    if nbr.size == 0:
        return nbr
    nbr = np.clip(nbr, 0, max(F - 1, 0))
    return nbr


# ============================================================
# Dataset
# ============================================================
class GraphForTSGCNetDataset(TwoStream24FaceDataset):
    """
    TSGCNet-specific dataset:
      - inherits TwoStream24FaceDataset
      - returns dense nbr (F,K), not edge_index
      - supports:
          * knn_center : KNN on face centers (3D)
          * knn_xc     : KNN on x_c (12D)
          * face_adj   : share-edge adjacency fallback
      - caches nbr on disk
    """

    def __init__(self, files: List[str], cfg: FaceConfig):
        super().__init__(files, cfg)
        self.gcfg = cfg

        self.graph_type = str(getattr(cfg, "graph_type", "knn_xc")).lower().strip()
        self.k_neighbors = int(getattr(cfg, "k_neighbors", 32))
        self.cache_nbr = bool(getattr(cfg, "cache_nbr", True))
        self.cache_dir = Path(str(getattr(cfg, "cache_dir", "cache/tsgcnet_nbr")))
        self.include_self_if_needed = bool(getattr(cfg, "include_self_if_needed", True))

        if self.cache_nbr:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_config(cls, files: List[str], full_cfg: dict, arch: str):
        data = full_cfg["data"]
        num_classes = int(full_cfg["model"]["kwargs"].get("num_classes", 16))

        base = FaceConfig(
            mode="graph",
            arch=arch,
            normalize=bool(data.get("normalize", True)),
            align_pca=bool(data.get("align_pca", False)),
            ignore_index=int(data.get("ignore_index", -1)),
            unknown_policy=str(data.get("unknown_policy", "raise")),
            require_labels=bool(data.get("require_labels", True)),
            num_classes=num_classes,
            label_source=str(data.get("label_source", "auto")),
            face_fallback=str(data.get("face_fallback", "majority")),
            num_faces=int(data.get("num_faces", 16000)),
            feature=str(data.get("face_feature", "twostream24")),
            return_faces=True,   # keep for fallback adjacency
            return_nbr=False,    # super() should not create old nbr
            k_neighbors=int(data.get("k_neighbors", 32)),
        )

        setattr(
            base,
            "twostream_coord_mode",
            str(data.get("twostream_coord_mode", "absolute")).lower().strip(),
        )

        setattr(base, "graph_type", str(data.get("graph_type", "knn_xc")).lower().strip())
        setattr(base, "cache_nbr", bool(data.get("cache_nbr", True)))
        setattr(base, "cache_dir", str(data.get("cache_dir", "cache/tsgcnet_nbr")))
        setattr(base, "include_self_if_needed", bool(data.get("include_self_if_needed", True)))

        return cls(files, base)

    def _cache_path(self, mesh_path: str, F_used: int) -> Path:
        key = (
            f"{mesh_path}|F={F_used}|"
            f"{self.graph_type}|k={self.k_neighbors}|"
            f"coord_mode={getattr(self, 'twostream_coord_mode', 'absolute')}"
        )
        h = hashlib.md5(key.encode("utf-8")).hexdigest()
        stem = Path(mesh_path).stem
        return self.cache_dir / f"{stem}_{h}.npz"

    def _build_nbr_from_sample(self, out: Dict[str, Any], F_used: int) -> np.ndarray:
        if F_used <= 0:
            return np.zeros((0, self.k_neighbors), dtype=np.int64)

        graph_type = self.graph_type

        if graph_type == "knn_center":
            feat = out["pos"][:F_used].detach().cpu().numpy().astype(np.float32, copy=False)   # (F,3)
            nbr = _build_knn_nbr(feat, self.k_neighbors)

        elif graph_type == "knn_xc":
            feat = out["x_c"][:F_used].detach().cpu().numpy().astype(np.float32, copy=False)   # (F,12)
            nbr = _build_knn_nbr(feat, self.k_neighbors)

        elif graph_type == "face_adj":
            faces_t = out.get("faces", None)
            if faces_t is None:
                raise ValueError("graph_type='face_adj' requires faces in sample.")
            faces_np = faces_t[:F_used].detach().cpu().numpy().astype(np.int64, copy=False)
            nbr = build_face_neighbors(faces_np, k=self.k_neighbors).astype(np.int64, copy=False)

        else:
            raise ValueError(
                f"Unknown graph_type='{graph_type}'. "
                f"Use 'knn_xc', 'knn_center', or 'face_adj'."
            )

        nbr = _sanitize_nbr_np(nbr, F_used)

        if self.include_self_if_needed and nbr.shape[1] < self.k_neighbors:
            pad = np.repeat(np.arange(F_used, dtype=np.int64)[:, None], self.k_neighbors - nbr.shape[1], axis=1)
            nbr = np.concatenate([nbr, pad], axis=1)

        return nbr[:, : self.k_neighbors].astype(np.int64, copy=False)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        out = super().__getitem__(idx)

        F_used = int(out.get("F_used", 0))
        target_F = int(self.fcfg.num_faces)

        if F_used <= 0:
            out["nbr"] = torch.zeros((target_F, self.k_neighbors), dtype=torch.long)
            out["meta"]["graph_type"] = self.graph_type
            out["meta"]["k_neighbors_graph"] = int(self.k_neighbors)
            out["meta"]["nbr_source"] = "empty"
            return out

        cache_path = self._cache_path(out["path"], F_used)
        if self.cache_nbr and cache_path.exists():
            data = np.load(str(cache_path))
            nbr_used = data["nbr"].astype(np.int64, copy=False)
            nbr_source = "cache"
        else:
            nbr_used = self._build_nbr_from_sample(out, F_used)
            nbr_source = "built"
            if self.cache_nbr:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(str(cache_path), nbr=nbr_used)

        # pad nbr to target_F so batch collation can stack safely
        nbr_pad = np.zeros((target_F, self.k_neighbors), dtype=np.int64)
        nbr_pad[:F_used] = nbr_used

        # padded rows point to themselves
        for i in range(F_used, target_F):
            nbr_pad[i, :] = i

        out["nbr"] = torch.from_numpy(nbr_pad).long()
        out["meta"]["graph_type"] = self.graph_type
        out["meta"]["k_neighbors_graph"] = int(self.k_neighbors)
        out["meta"]["nbr_source"] = nbr_source
        out["meta"]["nbr_shape"] = [int(target_F), int(self.k_neighbors)]

        return out