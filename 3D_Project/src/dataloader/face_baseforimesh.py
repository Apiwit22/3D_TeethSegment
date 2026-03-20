# src/dataloader/face_baseforimesh.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, List, Tuple, Optional
import hashlib

import numpy as np
import torch

from src.dataloader.base import BaseDentalDataset, BaseDentalDatasetConfig

# ============================================================
# Geometry helpers (NO color features)
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


def face_area(pos: np.ndarray, faces: np.ndarray) -> np.ndarray:
    v0 = pos[faces[:, 0]]
    v1 = pos[faces[:, 1]]
    v2 = pos[faces[:, 2]]
    a = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)
    return a.astype(np.float32)


def face_edge_len_mean(pos: np.ndarray, faces: np.ndarray) -> np.ndarray:
    v0 = pos[faces[:, 0]]
    v1 = pos[faces[:, 1]]
    v2 = pos[faces[:, 2]]
    e01 = np.linalg.norm(v1 - v0, axis=1)
    e12 = np.linalg.norm(v2 - v1, axis=1)
    e20 = np.linalg.norm(v0 - v2, axis=1)
    e = (e01 + e12 + e20) / 3.0
    return e.astype(np.float32)


# ============================================================
# Face-neighbor helpers (share-edge adjacency)
# ============================================================
def build_face_neighbors(faces: np.ndarray, k: int) -> np.ndarray:
    """
    Deterministic share-edge neighbors.
    Returns: (F, k) int64
    """
    faces = np.asarray(faces, dtype=np.int64)
    F = int(faces.shape[0])
    k = int(k)
    if F <= 0:
        return np.zeros((0, k), dtype=np.int64)

    edge_map: Dict[Tuple[int, int], List[int]] = {}
    for fi, (a, b, c) in enumerate(faces):
        a, b, c = int(a), int(b), int(c)
        for u, v in ((a, b), (b, c), (c, a)):
            if u > v:
                u, v = v, u
            edge_map.setdefault((u, v), []).append(fi)

    nbr = np.full((F, k), -1, dtype=np.int64)
    for fi, (a, b, c) in enumerate(faces):
        neigh: set[int] = set()
        for u, v in ((int(a), int(b)), (int(b), int(c)), (int(c), int(a))):
            if u > v:
                u, v = v, u
            for fj in edge_map.get((u, v), []):
                if fj != fi:
                    neigh.add(int(fj))

        if not neigh:
            nbr[fi, :] = fi
            continue

        neigh_sorted = sorted(neigh)
        if len(neigh_sorted) >= k:
            nbr[fi, :] = np.asarray(neigh_sorted[:k], dtype=np.int64)
        else:
            nbr[fi, : len(neigh_sorted)] = np.asarray(neigh_sorted, dtype=np.int64)
            nbr[fi, len(neigh_sorted) :] = fi

    nbr[nbr < 0] = 0
    return nbr


def neighbor_normal_variation(fn: np.ndarray, nbr: np.ndarray) -> np.ndarray:
    neigh = fn[nbr]  # (F,K,3)
    dots = np.sum(neigh * fn[:, None, :], axis=-1)  # (F,K)
    dots = np.clip(dots, -1.0, 1.0)
    return (1.0 - dots.mean(axis=1)).astype(np.float32)


# ============================================================
# Config
# ============================================================
@dataclass
class FaceConfigIMesh(BaseDentalDatasetConfig):
    """
    sample_policy:
      - "hash"  : deterministic pseudo-random subset per file (stable across epochs)  ✅ RECOMMENDED
      - "first" : take first target_F faces (fully deterministic, may bias)
      - "random": random each call (test loss will jump)
    """
    num_faces: int = 16000
    feature: str = "meshsegnet15"
    return_faces: bool = False
    return_nbr: bool = False
    k_neighbors: int = 3

    sample_policy: str = "hash"
    sample_seed: int = 42


# ============================================================
# Dataset
# ============================================================
class FaceDatasetIMesh(BaseDentalDataset):
    """
    FaceDataset optimized for IMeshSegNet / MeshSegNet-style training:
      - Stable face sampling (hash-based) to stop test-loss oscillation.
      - meshsegnet15 feature for x (15ch) + pos=center (3ch).
    """

    def __init__(self, files: List[str], cfg: FaceConfigIMesh):
        super().__init__(files, cfg)
        self.fcfg = cfg

    @classmethod
    def from_config(cls, files: List[str], full_cfg: dict, arch: str):
        data = full_cfg["data"]
        num_classes = int(full_cfg["model"]["kwargs"].get("num_classes", 16))

        base = FaceConfigIMesh(
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
            feature=str(data.get("face_feature", "meshsegnet15")),
            return_faces=bool(data.get("return_faces", False)),
            return_nbr=bool(data.get("return_nbr", False)),
            k_neighbors=int(data.get("k_neighbors", 3)),

            # NEW knobs (safe defaults)
            sample_policy=str(data.get("sample_policy", "hash")).lower().strip(),
            sample_seed=int(data.get("sample_seed", full_cfg.get("experiment", {}).get("seed", 42))),
        )
        return cls(files, base)

    # ---------- deterministic sampling ----------
    def _hash_seed(self, path: str, F0: int) -> int:
        s = f"{self.fcfg.sample_seed}|{path}|F0={int(F0)}"
        h = hashlib.md5(s.encode("utf-8")).hexdigest()
        return int(h[:8], 16)  # 32-bit-ish

    def _select_face_indices(self, path: str, F0: int, target_F: int) -> np.ndarray:
        policy = str(getattr(self.fcfg, "sample_policy", "hash")).lower().strip()
        if policy in ("first", "deterministic", "fixed"):
            return np.arange(target_F, dtype=np.int64)
        if policy in ("random", "rand"):
            # original behavior: stochastic each call
            return self.sample_indices(F0, target_F).astype(np.int64, copy=False)

        # default: "hash" deterministic pseudo-random per file
        seed = self._hash_seed(path, F0)
        rng = np.random.RandomState(seed)
        # choice without replacement -> stable subset
        return rng.choice(F0, size=target_F, replace=False).astype(np.int64, copy=False)

    def _needs_nbr(self, feat_name: str) -> bool:
        feat_name = str(feat_name).lower().strip()
        if feat_name in ("cn_area_edge_nvar", "cn9", "center_normal_9ch"):
            return True
        return bool(self.fcfg.return_nbr)

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

        # ------------------------------------------------------------
        # Fix face count: deterministic for IMesh (reduces test oscillation)
        # ------------------------------------------------------------
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

        # ------------------------------------------------------------
        # labels (BaseDentalDataset policy)
        # ------------------------------------------------------------
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

        # ------------------------------------------------------------
        # features
        # ------------------------------------------------------------
        feat_name = str(self.fcfg.feature).lower().strip()

        centers = face_centers(pos_v, faces).astype(np.float32, copy=False)   # (F,3)
        fn_face = face_normals(pos_v, faces).astype(np.float32, copy=False)  # (F,3)

        need_nbr = self._needs_nbr(feat_name)
        nbr: Optional[np.ndarray]
        if need_nbr:
            nbr = build_face_neighbors(faces, k=int(self.fcfg.k_neighbors)).astype(np.int64, copy=False)
        else:
            nbr = None

        if feat_name in ("meshsegnet15", "msn15", "mesh15"):
            v0 = pos_v[faces[:, 0]]
            v1 = pos_v[faces[:, 1]]
            v2 = pos_v[faces[:, 2]]
            dv0 = (v0 - centers).astype(np.float32, copy=False)
            dv1 = (v1 - centers).astype(np.float32, copy=False)
            dv2 = (v2 - centers).astype(np.float32, copy=False)

            x = np.concatenate([centers, fn_face, dv0, dv1, dv2], axis=1).astype(np.float32, copy=False)  # (F,15)

        elif feat_name in ("center_normal", "cn", "cn6"):
            x = np.concatenate([centers, fn_face], axis=1).astype(np.float32, copy=False)  # (F,6)

        elif feat_name in ("cn_area_edge_nvar", "cn9", "center_normal_9ch"):
            if nbr is None:
                nbr = build_face_neighbors(faces, k=int(self.fcfg.k_neighbors)).astype(np.int64, copy=False)

            area = face_area(pos_v, faces)
            edge = face_edge_len_mean(pos_v, faces)
            area_n = area / (float(area.mean()) + 1e-9)
            edge_n = edge / (float(edge.mean()) + 1e-9)
            nvar = neighbor_normal_variation(fn_face, nbr)

            x = np.concatenate(
                [centers, fn_face, area_n[:, None], edge_n[:, None], nvar[:, None]],
                axis=1,
            ).astype(np.float32, copy=False)  # (F,9)

        else:
            raise ValueError(
                f"Unknown face_feature='{self.fcfg.feature}'. "
                f"Use 'meshsegnet15' (recommended) or 'center_normal' or 'cn9'."
            )

        # ------------------------------------------------------------
        # Pad to target_F
        # ------------------------------------------------------------
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

            if nbr is not None:
                K = int(self.fcfg.k_neighbors)
                nbr_pad = np.zeros((target_F, K), dtype=np.int64)
                nbr_pad[:F_used] = nbr
                for i in range(F_used, target_F):
                    nbr_pad[i, :] = i
                nbr = nbr_pad

        # ------------------------------------------------------------
        # masks
        # ------------------------------------------------------------
        ign = int(self.cfg.ignore_index)
        mask = valid & (y != ign)

        out: Dict[str, Any] = {
            "x": torch.from_numpy(x),                                   # (F,C)
            "y": torch.from_numpy(y),                                   # (F,)
            "mask": torch.from_numpy(mask.astype(np.bool_)),            # (F,)
            "valid_face": torch.from_numpy(valid.astype(np.bool_)),     # (F,)
            "F_used": int(F_used),
            "arch": arch,
            "path": path,
            "pos": torch.from_numpy(centers),                           # (F,3)
            "meta": {
                **mesh.meta,
                "mode": str(self.cfg.mode).lower(),
                "arch": arch,
                "gt_source": gt_source,
                "face_feature": feat_name,
                "sample_policy": str(getattr(self.fcfg, "sample_policy", "hash")),
                "sample_seed": int(getattr(self.fcfg, "sample_seed", 42)),
                "F_raw": int(F0),
                "F_used": int(F_used),
                "F_target": int(target_F),
                "padded": bool(padded),
                "cropped": bool(cropped),
                "label_source": str(self.cfg.label_source),
                "face_fallback": str(self.cfg.face_fallback),
                "uses_vertex_normals": False,
                "has_nbr": bool(nbr is not None),
                "k_neighbors": int(self.fcfg.k_neighbors),
            },
        }

        if bool(self.fcfg.return_faces):
            out["faces"] = torch.from_numpy(faces.astype(np.int64))

        if bool(self.fcfg.return_nbr) and (nbr is not None):
            out["nbr"] = torch.from_numpy(nbr.astype(np.int64))

        return out