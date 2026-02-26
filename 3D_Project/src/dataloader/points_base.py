# src/dataloader/points_base.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple

import numpy as np
import torch

from src.dataloader.base import BaseDentalDataset, BaseDentalDatasetConfig


# -------------------------
# Surface sampling helpers
# -------------------------
def _triangle_areas(pos: np.ndarray, faces: np.ndarray) -> np.ndarray:
    v0 = pos[faces[:, 0]]
    v1 = pos[faces[:, 1]]
    v2 = pos[faces[:, 2]]
    a = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)
    return a.astype(np.float64)


def sample_points_on_mesh(
    pos: np.ndarray,
    faces: np.ndarray,
    num_points: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Uniformly sample points on mesh surface by triangle area.
    Returns:
      P: (N,3) sampled points
      Nf: (N,3) sampled normals (face normal)
      tri_ids: (N,) face index for each sampled point
    """
    pos = np.asarray(pos, dtype=np.float32)
    faces = np.asarray(faces, dtype=np.int64)
    F = int(faces.shape[0])
    if F <= 0:
        raise ValueError("No faces to sample from")

    areas = _triangle_areas(pos, faces) + 1e-12
    prob = areas / areas.sum()

    tri_ids = rng.choice(F, size=int(num_points), replace=True, p=prob)

    v0 = pos[faces[tri_ids, 0]]
    v1 = pos[faces[tri_ids, 1]]
    v2 = pos[faces[tri_ids, 2]]

    # barycentric sampling (uniform in triangle)
    u = rng.random((num_points, 1), dtype=np.float32)
    v = rng.random((num_points, 1), dtype=np.float32)
    m = (u + v) > 1.0
    u[m] = 1.0 - u[m]
    v[m] = 1.0 - v[m]
    P = v0 + u * (v1 - v0) + v * (v2 - v0)

    # face normals
    fn = np.cross(v1 - v0, v2 - v0)
    fn = fn / (np.linalg.norm(fn, axis=1, keepdims=True) + 1e-12)

    return P.astype(np.float32), fn.astype(np.float32), tri_ids.astype(np.int64)


@dataclass
class PointConfig(BaseDentalDatasetConfig):
    num_points: int = 4096
    feature: str = "xyz_n"   # xyz_n (6ch) or xyz (3ch)
    seed: int = 1234         # deterministic sampling per dataset instance


class PointDataset(BaseDentalDataset):
    """
    Point-mode dataset (for PointNet++ / PointCNN).
    IMPORTANT for your project:
      - GT comes from FACE colors (not vertex colors).
      - Points are sampled on surface (triangles) by area.
      - Point label = label of the triangle it was sampled from.
      - Feature contains ONLY geometry (no rgb).
    """

    def __init__(self, files: List[str], cfg: PointConfig):
        super().__init__(files, cfg)
        self.pcfg = cfg
        self._rng = np.random.default_rng(int(cfg.seed))

    @classmethod
    def from_config(cls, files: List[str], full_cfg: dict, arch: str):
        data = full_cfg["data"]
        num_classes = int(full_cfg["model"]["kwargs"].get("num_classes", 16))

        base = PointConfig(
            mode="point",
            arch=arch,
            normalize=bool(data.get("normalize", True)),
            align_pca=bool(data.get("align_pca", False)),
            ignore_index=int(data.get("ignore_index", -1)),
            unknown_policy=str(data.get("unknown_policy", "raise")),
            require_labels=bool(data.get("require_labels", True)),
            num_classes=num_classes,

            # point specifics
            num_points=int(data.get("num_points", 4096)),
            feature=str(data.get("point_feature", "xyz_n")),
            seed=int(data.get("seed", 1234)),

            # label policy (now meaningful!)
            label_source=str(data.get("label_source", "face")),      # <<< default to face for your data
            face_fallback=str(data.get("face_fallback", "majority")),
        )
        return cls(files, base)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        path = self.files[idx]
        arch = self.get_arch(path)
        mesh = self.load_mesh(path)

        pos = mesh.pos.astype(np.float32, copy=False)          # (Nv,3)
        faces = mesh.faces                                     # (F,3) or None
        ignore_index = int(self.cfg.ignore_index)
        feat = str(self.pcfg.feature).lower()

        if faces is None:
            raise ValueError(f"PLY has no triangular faces but mode=point (needs faces for surface sampling): {path}")
        faces = faces.astype(np.int64, copy=False)

        # -------------------------
        # labels from FACE colors (preferred)
        # -------------------------
        gt_source = "none"
        if not self.cfg.require_labels:
            y_face = np.full((faces.shape[0],), ignore_index, dtype=np.int64)
            gt_source = "disabled(require_labels=false)"
        else:
            # This should respect label_source="face"/"auto"/fallback as in your base.py
            y_face = self.get_labels_for_mode(mesh, arch).astype(np.int64, copy=False)  # (F,)
            gt_source = str(getattr(mesh, "meta", {}).get("gt_source", "face_or_policy"))

        # -------------------------
        # sample points on surface
        # -------------------------
        P, Nf, tri_ids = sample_points_on_mesh(
            pos=pos,
            faces=faces,
            num_points=int(self.pcfg.num_points),
            rng=self._rng,
        )
        y = y_face[tri_ids].astype(np.int64, copy=False)

        if feat == "xyz":
            x = P
        elif feat in ("xyz_n", "xyzn", "xyz+normal", "xyz_normals"):
            x = np.concatenate([P, Nf], axis=1).astype(np.float32, copy=False)  # (N,6)
        else:
            raise ValueError(f"Unknown point_feature='{self.pcfg.feature}'. Use 'xyz' or 'xyz_n'.")

        mask = (y != ignore_index)

        out: Dict[str, Any] = {
            "x": torch.from_numpy(x),                          # (N,3 or 6)
            "y": torch.from_numpy(y),                          # (N,)
            "mask": torch.from_numpy(mask.astype(np.bool_)),
            "arch": arch,
            "path": path,
            "meta": {
                **mesh.meta,
                "mode": "point",
                "arch": arch,
                "gt_source": gt_source,
                "point_feature": feat,
                "num_points": int(self.pcfg.num_points),
                "label_source": str(self.cfg.label_source),
                "face_fallback": str(self.cfg.face_fallback),
            },
            "pos": torch.from_numpy(P),                        # sampled positions (N,3)
            "tri_ids": torch.from_numpy(tri_ids),              # (N,) triangle id (debug/optional)
        }
        return out
