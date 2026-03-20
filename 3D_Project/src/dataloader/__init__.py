# src/dataloader/__init__.py
from __future__ import annotations

from typing import Callable, Dict, Any, List

from .arch import parse_arch_from_filename
from .fdi_colors import FDIColorMap

from .base import BaseDentalDatasetConfig
from .points_base import PointDataset
from .faces_base import FaceDataset
from .graph_base import GraphDataset
from .faces_twostream24 import TwoStream24FaceDataset

# old extra datasets
from .face_baseforimesh import FaceDatasetIMesh
from .face_baseforimesh2 import FaceDatasetIMesh2

from .collate import collate_point, collate_face, collate_graph

# NEW: TSGCNet2 graph dataset
from .graph_fortsgcnet import GraphForTSGCNetDataset


def _get_mode(cfg: Dict[str, Any]) -> str:
    if "data" not in cfg or not isinstance(cfg["data"], dict):
        raise KeyError("cfg must contain dict key 'data'")
    if "mode" not in cfg["data"]:
        raise KeyError("cfg['data'] must contain key 'mode' (point/face/graph)")
    return str(cfg["data"]["mode"]).lower().strip()


def _get_face_feature(cfg: Dict[str, Any]) -> str:
    data = cfg.get("data", {})
    if not isinstance(data, dict):
        return "center_normal"
    return str(data.get("face_feature", "center_normal")).lower().strip()


def _get_graph_loader(cfg: Dict[str, Any]) -> str:
    data = cfg.get("data", {})
    if not isinstance(data, dict):
        return ""
    return str(data.get("graph_loader", "")).lower().strip()


_TWO_STREAM_24_ALIASES = {
    "twostream24",
    "two_stream24",
    "two-stream24",
    "tsgc24",
    "tsgcnet24",
    "tgcn24",
    "fast_tgcn_24",
    "paper24",
    "twostream24_topo",
}

_MESHSEGNET15_ALIASES = {
    "meshsegnet15",
    "msn15",
    "mesh15",
}

_TSMDL15_ALIASES = {
    "tsmdl15",
    "ts-mdl15",
    "ts_mdl15",
    "imesh15",
    "paper15",
}


def build_dataset(files: List[str], cfg: Dict[str, Any], arch: str):
    mode = _get_mode(cfg)
    feat = _get_face_feature(cfg)

    if mode == "point":
        return PointDataset.from_config(files, cfg, arch=arch)

    if mode == "face":
        # old two-stream 24D loader stays unchanged
        if feat in _TWO_STREAM_24_ALIASES:
            return TwoStream24FaceDataset.from_config(files, cfg, arch=arch)

        if feat in _TSMDL15_ALIASES:
            return FaceDatasetIMesh2.from_config(files, cfg, arch=arch)

        if feat in _MESHSEGNET15_ALIASES:
            return FaceDatasetIMesh.from_config(files, cfg, arch=arch)

        return FaceDataset.from_config(files, cfg, arch=arch)

    if mode == "graph":
        graph_loader = _get_graph_loader(cfg)

        # NEW SAFE ROUTE:
        # only use new graph loader when explicitly requested
        if graph_loader in {"tsgcnet2", "graph_fortsgcnet"}:
            return GraphForTSGCNetDataset.from_config(files, cfg, arch=arch)

        # backward-compatible old route
        return GraphDataset.from_config(files, cfg, arch=arch)

    raise ValueError(f"Unknown data.mode: {mode} (expected point/face/graph)")


def build_collate_fn(cfg: Dict[str, Any]) -> Callable:
    mode = _get_mode(cfg)
    if mode == "point":
        return collate_point
    if mode == "face":
        return collate_face
    if mode == "graph":
        return collate_graph
    raise ValueError(f"Unknown data.mode: {mode} (expected point/face/graph)")


__all__ = [
    "parse_arch_from_filename",
    "FDIColorMap",
    "BaseDentalDatasetConfig",
    "PointDataset",
    "FaceDataset",
    "FaceDatasetIMesh",
    "FaceDatasetIMesh2",
    "TwoStream24FaceDataset",
    "GraphForTSGCNetDataset",
    "GraphDataset",
    "collate_point",
    "collate_face",
    "collate_graph",
    "build_dataset",
    "build_collate_fn",
]