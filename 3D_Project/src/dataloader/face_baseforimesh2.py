# src/dataloader/face_baseforimesh2.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, List, Tuple, Optional
import hashlib

import numpy as np
import torch

from src.dataloader.base import BaseDentalDataset, BaseDentalDatasetConfig


# ============================================================
# Geometry helpers
# ============================================================
def face_centers(pos: np.ndarray, faces: np.ndarray) -> np.ndarray:
    v0 = pos[faces[:, 0]]
    v1 = pos[faces[:, 1]]
    v2 = pos[faces[:, 2]]
    return (v0 + v1 + v2) / 3.0


def face_normals(pos: np.ndarray, faces: np.ndarray) -> np.ndarray:
    v0 = pos[faces[:, 0]]
    v1 = pos[faces[:, 1]]
    v2 = pos[faces[:, 2]]
    n = np.cross(v1 - v0, v2 - v0)
    norm = np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
    return (n / norm).astype(np.float32)


def zscore_per_mesh(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Z-score normalization per sample (mesh):
      x_norm = (x - mean) / (std + eps)
    """
    mu = x.mean(axis=0, keepdims=True)
    sd = x.std(axis=0, keepdims=True)
    return ((x - mu) / (sd + eps)).astype(np.float32, copy=False)


# ============================================================
# Config
# ============================================================
@dataclass
class FaceConfigIMesh2(BaseDentalDatasetConfig):
    """
    TS-MDL/iMeshSegNet-style config.

    sample_policy:
      - "hash": deterministic pseudo-random subset per file (stable across epochs) ✅
      - "first": first target_F faces (fully deterministic, may bias)
      - "random": random each __getitem__ (will make test loss jump)

    feature:
      - "tsmdl15": 15D feature close to TS-MDL description:
          [v0(3), v1(3), v2(3), n(3), rel(3)]
        where:
          rel = center - mean(center_all_mesh)  (relative position, 3D)
        and then z-score per mesh.
    """
    num_faces: int = 16000
    feature: str = "tsmdl15"

    return_faces: bool = False
    return_nbr: bool = False
    k_neighbors: int = 3

    sample_policy: str = "hash"
    sample_seed: int = 42

    # feature normalization
    zscore: bool = True


# ============================================================
# Dataset
# ============================================================
class FaceDatasetIMesh2(BaseDentalDataset):
    """
    For imeshsegnet_knn2:
      - stable face sampling (hash)
      - TS-MDL-like 15D features + zscore
      - outputs:
          x:   (F,15)
          pos: (F,3)  centers (for kNN build)
    """

    def __init__(self, files: List[str], cfg: FaceConfigIMesh2):
        super().__init__(files, cfg)
        self.fcfg = cfg

    @classmethod
    def from_config(cls, files: List[str], full_cfg: dict, arch: str):
        data = full_cfg["data"]
        num_classes = int(full_cfg["model"]["kwargs"].get("num_classes", 16))

        base = FaceConfigIMesh2(
            mode=str(data.get("mode", "face")),
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
            feature=str(data.get("face_feature", "tsmdl15")),

            return_faces=bool(data.get("return_faces", False)),
            return_nbr=bool(data.get("return_nbr", False)),
            k_neighbors=int(data.get("k_neighbors", 3)),

            sample_policy=str(data.get("sample_policy", "hash")).lower().strip(),
            sample_seed=int(data.get("sample_seed", full_cfg.get("experiment", {}).get("seed", 42))),
            zscore=bool(data.get("zscore", True)),
        )
        return cls(files, base)

    # ---------- deterministic sampling ----------
    def _hash_seed(self, path: str, F0: int) -> int:
        s = f"{self.fcfg.sample_seed}|{path}|F0={int(F0)}"
        h = hashlib.md5(s.encode("utf-8")).hexdigest()
        return int(h[:8], 16)

    def _select_face_indices(self, path: str, F0: int, target_F: int) -> np.ndarray:
        policy = str(getattr(self.fcfg, "sample_policy", "hash")).lower().strip()
        if policy in ("first", "deterministic", "fixed"):
            return np.arange(target_F, dtype=np.int64)
        if policy in ("random", "rand"):
            return self.sample_indices(F0, target_F).astype(np.int64, copy=False)

        seed = self._hash_seed(path, F0)
        rng = np.random.RandomState(seed)
        return rng.choice(F0, size=target_F, replace=False).astype(np.int64, copy=False)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        path = self.files[idx]
        arch = self.get_arch(path)
        mesh = self.load_mesh(path)

        if mesh.faces is None:
            raise ValueError(f"PLY has no triangular faces but mode=face: {path}")

        pos_v = mesh.pos.astype(np.float32, copy=False)         # (Nv,3)
        faces_all = mesh.faces.astype(np.int64, copy=False)     # (F0,3)
        F0 = int(faces_all.shape[0])
        target_F = int(self.fcfg.num_faces)

        if F0 <= 0:
            raise ValueError(f"Empty faces in: {path}")

        # ---- select faces (stable) ----
        cropped = False
        if F0 >= target_F:
            fidx = self._select_face_indices(path, F0, target_F)
            faces = faces_all[fidx]
            valid = np.ones((target_F,), dtype=np.bool_)
            padded = False
            F_used = target_F
            cropped = (F0 > target_F)
        else:
            fidx = None
            faces = faces_all
            valid = np.zeros((target_F,), dtype=np.bool_)
            valid[:F0] = True
            padded = True
            F_used = F0

        # ---- labels ----
        if not self.cfg.require_labels:
            y_all = np.full((F0,), int(self.cfg.ignore_index), dtype=np.int64)
            gt_source = "disabled(require_labels=false)"
        else:
            if mesh.face_rgb is not None and str(self.cfg.label_source).lower() != "vertex":
                gt_source = "face_rgb"
            elif mesh.rgb is not None:
                gt_source = "vertex_fallback"
            else:
                gt_source = "missing_colors"
            y_all = self.get_labels_for_mode(mesh, arch)  # (F0,)

        y = y_all[fidx] if fidx is not None else y_all
        y = y.astype(np.int64, copy=False)

        # ---- features: TS-MDL-like 15D ----
        feat_name = str(self.fcfg.feature).lower().strip()
        if feat_name not in ("tsmdl15", "ts-mdl15", "imesh15", "paper15"):
            raise ValueError("face_feature must be 'tsmdl15' for FaceDatasetIMesh2")

        v0 = pos_v[faces[:, 0]]  # (F,3)
        v1 = pos_v[faces[:, 1]]
        v2 = pos_v[faces[:, 2]]
        centers = ((v0 + v1 + v2) / 3.0).astype(np.float32, copy=False)  # (F,3)
        fn_face = face_normals(pos_v, faces).astype(np.float32, copy=False)  # (F,3)

        # relative position: center - mean(center_all_mesh_selected)
        rel = (centers - centers.mean(axis=0, keepdims=True)).astype(np.float32, copy=False)

        # 15D: [v0,v1,v2, n, rel]
        x = np.concatenate([v0, v1, v2, fn_face, rel], axis=1).astype(np.float32, copy=False)  # (F,15)

        # z-score per mesh (recommended by TS-MDL description)
        if bool(self.fcfg.zscore):
            x = zscore_per_mesh(x)

        # ---- pad if needed ----
        if padded:
            C = int(x.shape[1])
            x_pad = np.zeros((target_F, C), dtype=np.float32)
            y_pad = np.full((target_F,), int(self.cfg.ignore_index), dtype=np.int64)
            centers_pad = np.zeros((target_F, 3), dtype=np.float32)
            faces_pad = np.zeros((target_F, 3), dtype=np.int64)

            x_pad[:F_used] = x
            y_pad[:F_used] = y
            centers_pad[:F_used] = centers
            faces_pad[:F_used] = faces

            x, y, centers, faces = x_pad, y_pad, centers_pad, faces_pad

        # ---- mask ----
        ign = int(self.cfg.ignore_index)
        mask = valid & (y != ign)

        out: Dict[str, Any] = {
            "x": torch.from_numpy(x),                                   # (F,15)
            "y": torch.from_numpy(y),                                   # (F,)
            "mask": torch.from_numpy(mask.astype(np.bool_)),            # (F,)
            "valid_face": torch.from_numpy(valid.astype(np.bool_)),     # (F,)
            "F_used": int(F_used),
            "arch": arch,
            "path": path,
            "pos": torch.from_numpy(centers),                           # (F,3) centers for kNN graph
            "meta": {
                **mesh.meta,
                "mode": str(self.cfg.mode).lower(),
                "arch": arch,
                "gt_source": gt_source,
                "face_feature": feat_name,
                "sample_policy": str(getattr(self.fcfg, "sample_policy", "hash")),
                "sample_seed": int(getattr(self.fcfg, "sample_seed", 42)),
                "zscore": bool(getattr(self.fcfg, "zscore", True)),
                "F_raw": int(F0),
                "F_used": int(F_used),
                "F_target": int(target_F),
                "padded": bool(padded),
                "cropped": bool(cropped),
                "label_source": str(self.cfg.label_source),
                "face_fallback": str(self.cfg.face_fallback),
            },
        }

        if bool(self.fcfg.return_faces):
            out["faces"] = torch.from_numpy(faces.astype(np.int64))

        return out