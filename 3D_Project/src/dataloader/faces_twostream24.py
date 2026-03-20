# src/dataloader/faces_twostream24.py
from __future__ import annotations

from typing import Dict, Any, List, Optional

import numpy as np
import torch

from src.dataloader.base import BaseDentalDataset
from src.dataloader.faces_base import (
    FaceConfig,
    face_centers,
    face_normals,
    build_face_neighbors,
)


class TwoStream24FaceDataset(BaseDentalDataset):
    """
    Two-stream 24D per-face features:
      - x_c (12): absolute [v0,v1,v2,center] OR relative [center, v0-center, v1-center, v2-center]
      - x_n (12): [n0,n1,n2,n_face]
      - x   (24): concat(x_c,x_n)
    """

    _ALIASES = (
        "twostream24",
        "two_stream24",
        "two-stream24",
        "tsgc24",
        "tsgcnet24",
        "tgcn24",
        "fast_tgcn_24",
        "paper24",
        "twostream24_topo",
    )

    def __init__(self, files: List[str], cfg: FaceConfig):
        super().__init__(files, cfg)
        self.fcfg = cfg
        self.twostream_coord_mode = str(getattr(cfg, "twostream_coord_mode", "relative")).lower().strip()

    @classmethod
    def from_config(cls, files: List[str], full_cfg: dict, arch: str):
        data = full_cfg["data"]
        num_classes = int(full_cfg["model"]["kwargs"].get("num_classes", 16))

        base = FaceConfig(
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
            feature=str(data.get("face_feature", "twostream24")),
            return_faces=bool(data.get("return_faces", True)),
            return_nbr=bool(data.get("return_nbr", False)),
            k_neighbors=int(data.get("k_neighbors", 3)),
        )

        setattr(base, "twostream_coord_mode", str(data.get("twostream_coord_mode", "relative")).lower().strip())
        return cls(files, base)

    def _needs_nbr(self) -> bool:
        return bool(self.fcfg.return_nbr)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        path = self.files[idx]
        arch = self.get_arch(path)
        mesh = self.load_mesh(path)

        if mesh.faces is None:
            raise ValueError(f"PLY has no triangular faces but mode=face/graph: {path}")

        pos_v = mesh.pos.astype(np.float32, copy=False)         # (Nv,3)
        faces_all = mesh.faces.astype(np.int64, copy=False)     # (F0,3)

        F0 = int(faces_all.shape[0])
        target_F = int(self.fcfg.num_faces)
        if F0 <= 0:
            raise ValueError(f"Empty faces in: {path}")

        cropped = False
        if F0 >= target_F:
            mode = str(self.cfg.mode).lower()
            if mode == "graph":
                fidx = np.arange(target_F, dtype=np.int64)  # deterministic for cached topology
                cropped = (F0 > target_F)
            else:
                fidx = self.sample_indices(F0, target_F)    # random for augmentation
                cropped = (F0 > target_F)

            faces = faces_all[fidx]
            valid = np.ones((target_F,), dtype=np.bool_)
            padded = False
            F_used = target_F
        else:
            fidx = None
            faces = faces_all
            valid = np.zeros((target_F,), dtype=np.bool_)
            valid[:F0] = True
            padded = True
            F_used = F0

        # labels
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
            y_all = self.get_labels_for_mode(mesh, arch)

        y = y_all[fidx] if fidx is not None else y_all
        y = y.astype(np.int64, copy=False)

        feat_name = str(self.fcfg.feature).lower().strip()
        if feat_name not in self._ALIASES:
            raise ValueError(
                f"faces_twostream24.py supports {self._ALIASES}, got '{self.fcfg.feature}'."
            )

        centers = face_centers(pos_v, faces).astype(np.float32, copy=False)   # (F,3)
        fn_face = face_normals(pos_v, faces).astype(np.float32, copy=False)  # (F,3)

        v0 = pos_v[faces[:, 0]]
        v1 = pos_v[faces[:, 1]]
        v2 = pos_v[faces[:, 2]]

        coord_mode = str(getattr(self, "twostream_coord_mode", "relative")).lower().strip()
        if coord_mode in ("rel", "relative"):
            dv0 = (v0 - centers).astype(np.float32, copy=False)
            dv1 = (v1 - centers).astype(np.float32, copy=False)
            dv2 = (v2 - centers).astype(np.float32, copy=False)
            x_c = np.concatenate([centers, dv0, dv1, dv2], axis=1).astype(np.float32, copy=False)
        else:
            x_c = np.concatenate([v0, v1, v2, centers], axis=1).astype(np.float32, copy=False)

        if mesh.normals is None or mesh.normals.shape[0] != pos_v.shape[0]:
            raise ValueError(f"Need vertex normals (Nv,3) for two-stream 24D: {path}")
        vn_all = mesh.normals.astype(np.float32, copy=False)
        n0 = vn_all[faces[:, 0]]
        n1 = vn_all[faces[:, 1]]
        n2 = vn_all[faces[:, 2]]
        x_n = np.concatenate([n0, n1, n2, fn_face], axis=1).astype(np.float32, copy=False)

        x = np.concatenate([x_c, x_n], axis=1).astype(np.float32, copy=False)

        if self._needs_nbr():
            nbr: Optional[np.ndarray] = build_face_neighbors(faces, k=int(self.fcfg.k_neighbors)).astype(np.int64, copy=False)
        else:
            nbr = None

        if padded:
            x_pad = np.zeros((target_F, 24), dtype=np.float32)
            y_pad = np.full((target_F,), int(self.cfg.ignore_index), dtype=np.int64)
            centers_pad = np.zeros((target_F, 3), dtype=np.float32)
            faces_pad = np.zeros((target_F, 3), dtype=np.int64)

            x_c_pad = np.zeros((target_F, 12), dtype=np.float32)
            x_n_pad = np.zeros((target_F, 12), dtype=np.float32)

            x_pad[:F_used] = x
            y_pad[:F_used] = y
            centers_pad[:F_used] = centers
            faces_pad[:F_used] = faces
            x_c_pad[:F_used] = x_c
            x_n_pad[:F_used] = x_n

            x, y, centers, faces = x_pad, y_pad, centers_pad, faces_pad
            x_c, x_n = x_c_pad, x_n_pad

            if nbr is not None:
                K = int(self.fcfg.k_neighbors)
                nbr_pad = np.zeros((target_F, K), dtype=np.int64)
                nbr_pad[:F_used] = nbr
                for i in range(F_used, target_F):
                    nbr_pad[i, :] = i
                nbr = nbr_pad

        ign = int(self.cfg.ignore_index)
        mask = valid & (y != ign)

        out: Dict[str, Any] = {
            "x": torch.from_numpy(x),
            "x_c": torch.from_numpy(x_c),
            "x_n": torch.from_numpy(x_n),
            "y": torch.from_numpy(y),
            "mask": torch.from_numpy(mask.astype(np.bool_)),
            "valid_face": torch.from_numpy(valid.astype(np.bool_)),
            "F_used": int(F_used),
            "arch": arch,
            "path": path,
            "pos": torch.from_numpy(centers),
            "meta": {
                **mesh.meta,
                "mode": str(self.cfg.mode).lower(),
                "arch": arch,
                "gt_source": gt_source,
                "face_feature": feat_name,          # ✅ not hardcoded
                "twostream_coord_mode": coord_mode,
                "F_raw": int(F0),
                "F_used": int(F_used),
                "F_target": int(target_F),
                "padded": bool(padded),
                "cropped": bool(cropped),
                "label_source": str(self.cfg.label_source),
                "face_fallback": str(self.cfg.face_fallback),
                "uses_vertex_normals": True,
                "has_nbr": bool(nbr is not None),
                "k_neighbors": int(self.fcfg.k_neighbors),
            },
        }

        if bool(self.fcfg.return_faces):
            out["faces"] = torch.from_numpy(faces.astype(np.int64, copy=False))
        if nbr is not None:
            out["nbr"] = torch.from_numpy(nbr.astype(np.int64, copy=False))

        return out