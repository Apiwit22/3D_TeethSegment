# src/dataloader/ply_dataloader.py
from __future__ import annotations

from typing import List, Dict, Any

from src.dataloader.points_base import PointDataset
from src.dataloader.faces_base import FaceDataset
from src.dataloader.graph_base import GraphDataset


class PlySegDataLoader:
    """
    Backward-compatible wrapper that builds a Dataset instance (point/face/graph)
    from a minimal set of args, while internally using the unified from_config() API.
    """

    def __init__(
        self,
        files: List[str],
        mode: str,
        arch: str,
        num_points: int = 4096,
        num_faces: int = 16000,
        normalize: bool = True,
        require_labels: bool = True,
        unknown_policy: str = "raise",
        ignore_index: int = -1,
        align_pca: bool = False,
        num_classes: int = 17,
        in_channels: int = 6,
        # NEW (explicit to match BaseDentalDatasetConfig)
        label_source: str = "auto",       # auto/vertex/face
        face_fallback: str = "majority",  # majority/first/none
        **kwargs,
    ):
        mode = str(mode).lower().strip()
        arch = str(arch).lower().strip()

        if mode not in ("point", "face", "graph"):
            raise ValueError(f"mode must be one of: point/face/graph. Got: {mode}")
        if arch not in ("upper", "lower", "both"):
            raise ValueError(f"arch must be one of: upper/lower/both. Got: {arch}")

        self.mode = mode
        self.arch = arch

        # Build a cfg dict that matches your Dataset.from_config() expectations
        full_cfg: Dict[str, Any] = {
            "data": {
                "mode": mode,
                "arch": arch,  # kept for completeness; from_config() uses arch arg too
                "num_points": int(num_points),
                "num_faces": int(num_faces),
                "normalize": bool(normalize),
                "align_pca": bool(align_pca),
                "require_labels": bool(require_labels),
                "unknown_policy": str(unknown_policy),
                "ignore_index": int(ignore_index),

                # label policy (important for preventing weird GT source changes)
                "label_source": str(label_source),
                "face_fallback": str(face_fallback),

                # feature selection (NO color in x by default -> avoids color leakage)
                "point_feature": kwargs.get("point_feature", "xyz_n"),
                "face_feature": kwargs.get("face_feature", "center_normal"),

                # mesh/graph extras
                "return_faces": bool(kwargs.get("return_faces", True)),
                "return_nbr": bool(kwargs.get("return_nbr", True)),
                "k_neighbors": int(kwargs.get("k_neighbors", 3)),
                "graph_type": kwargs.get("graph_type", "face_adj"),
                "return_edge_index": bool(kwargs.get("return_edge_index", True)),
            },
            "model": {
                "kwargs": {
                    "num_classes": int(num_classes),
                    "in_channels": int(in_channels),
                }
            },
        }

        if mode == "point":
            self.ds = PointDataset.from_config(files, full_cfg, arch=arch)
        elif mode == "face":
            self.ds = FaceDataset.from_config(files, full_cfg, arch=arch)
        else:  # graph
            self.ds = GraphDataset.from_config(files, full_cfg, arch=arch)

    @property
    def dataset(self):
        return self.ds

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        return self.ds[idx]
