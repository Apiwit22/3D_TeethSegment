# src/dataloader/ply_dataloader.py
from __future__ import annotations

from typing import List, Dict, Any

from src.dataloader import build_dataset


class PlySegDataLoader:
    """
    Backward-compatible wrapper that builds a Dataset instance (point/face/graph)
    from a minimal set of args, while internally using the unified from_config() API.

    ✅ Updated:
      - ใช้ build_dataset() เพื่อรองรับ TwoStream24FaceDataset + preset injection
      - ส่ง graph params ได้ครบขึ้น (adjacency_mode, add_self_loops, nonmanifold_policy, cache, ...)
      - รองรับชื่อกลาง: face_feature="twostream24_topo" (paper-like opt-in)
      - ยังรองรับ face_feature="paper24" (alias เก่า) แบบไม่เปลี่ยน behavior
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

        # infer in_channels if user forgot (safe)
        face_feature = str(kwargs.get("face_feature", "center_normal")).lower().strip()
        point_feature = str(kwargs.get("point_feature", "xyz_n")).lower().strip()
        if mode in ("face", "graph") and face_feature in ("twostream24", "twostream24_topo", "paper24"):
            in_channels = 24
        elif mode in ("point",) and "xyz_n" in point_feature:
            in_channels = 6
        elif mode in ("point",):
            in_channels = 3
        else:
            in_channels = 6

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

                # two-stream extras
                "twostream_coord_mode": kwargs.get("twostream_coord_mode", None),

                # mesh/graph extras
                "return_faces": bool(kwargs.get("return_faces", True)),
                "return_nbr": bool(kwargs.get("return_nbr", True)),
                "k_neighbors": int(kwargs.get("k_neighbors", 3)),

                # graph configs (forward more knobs)
                "graph_type": kwargs.get("graph_type", "face_adj"),
                "adjacency_mode": kwargs.get("adjacency_mode", None),
                "nonmanifold_policy": kwargs.get("nonmanifold_policy", None),
                "add_self_loops": kwargs.get("add_self_loops", None),
                "graph_k": kwargs.get("graph_k", None),
                "knn_undirected": kwargs.get("knn_undirected", None),

                "cache_edge_index": kwargs.get("cache_edge_index", None),
                "cache_dir": kwargs.get("cache_dir", None),
                "max_faces_per_edge": kwargs.get("max_faces_per_edge", None),
                "max_faces_per_vertex": kwargs.get("max_faces_per_vertex", None),

                "return_edge_index": bool(kwargs.get("return_edge_index", True)),
            },
            "model": {
                "kwargs": {
                    "num_classes": int(num_classes),
                    "in_channels": int(in_channels),
                }
            },
        }

        # drop None so dataset defaults remain intact unless caller overrides
        full_cfg["data"] = {k: v for k, v in full_cfg["data"].items() if v is not None}

        # ✅ Use factory (handles TwoStream24FaceDataset + preset injection)
        self.ds = build_dataset(files, full_cfg, arch=arch)

    @property
    def dataset(self):
        return self.ds

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        return self.ds[idx]