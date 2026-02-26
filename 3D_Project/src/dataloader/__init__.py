# src/dataloader/__init__.py
from __future__ import annotations

from typing import Callable, Dict, Any, List

from .arch import parse_arch_from_filename
from .fdi_colors import FDIColorMap

from .base import BaseDentalDatasetConfig
from .points_base import PointDataset
from .faces_base import FaceDataset
from .graph_base import GraphDataset

# ✅ เพิ่ม: dataset สำหรับฟีเจอร์ 24D แบบ two-stream (TSGCNet)
from .faces_twostream24 import TwoStream24FaceDataset

from .collate import collate_point, collate_face, collate_graph


def _get_mode(cfg: Dict[str, Any]) -> str:
    if "data" not in cfg or not isinstance(cfg["data"], dict):
        raise KeyError("cfg must contain dict key 'data'")
    if "mode" not in cfg["data"]:
        raise KeyError("cfg['data'] must contain key 'mode' (point/face/graph)")
    return str(cfg["data"]["mode"]).lower().strip()


def _get_face_feature(cfg: Dict[str, Any]) -> str:
    """
    อ่านชื่อ feature สำหรับ face-mode
    - ถ้าไม่มีจะให้ default เป็น 'center_normal' (เหมือนเดิม)
    """
    data = cfg.get("data", {})
    if not isinstance(data, dict):
        return "center_normal"
    return str(data.get("face_feature", "center_normal")).lower().strip()


# ✅ ชุดชื่อที่เราถือว่าเป็นฟีเจอร์แบบ two-stream 24D
# (รวม alias เก่าไว้ เผื่อ config เดิมบางอันยังใช้)
_TWO_STREAM_24_ALIASES = {
    "twostream24",
    "two_stream24",
    "two-stream24",
    "tsgc24",
    "tsgcnet24",
    "tgcn24",
    "fast_tgcn_24",
    "paper24",
}


def build_dataset(files: List[str], cfg: Dict[str, Any], arch: str):
    """
    Factory: build dataset by cfg['data']['mode'] ∈ {'point','face','graph'}.

    arch:
      - 'upper' / 'lower' => force arch for all files
      - 'both'            => infer per-file from filename (parse_arch_from_filename)
    """
    mode = _get_mode(cfg)

    if mode == "point":
        return PointDataset.from_config(files, cfg, arch=arch)

    if mode == "face":
        # ✅ ถ้า face_feature เป็น two-stream 24D -> ใช้ TwoStream24FaceDataset
        feat = _get_face_feature(cfg)
        if feat in _TWO_STREAM_24_ALIASES:
            return TwoStream24FaceDataset.from_config(files, cfg, arch=arch)

        # ไม่ใช่ two-stream -> ใช้ FaceDataset เดิม (ไม่กระทบโมเดลอื่น)
        return FaceDataset.from_config(files, cfg, arch=arch)

    if mode == "graph":
        # graph-mode ใช้ GraphDataset เหมือนเดิม (รองรับ edge_index ฯลฯ)
        return GraphDataset.from_config(files, cfg, arch=arch)

    raise ValueError(f"Unknown data.mode: {mode} (expected point/face/graph)")


def build_collate_fn(cfg: Dict[str, Any]) -> Callable:
    """
    Return the correct collate_fn for DataLoader.
    """
    mode = _get_mode(cfg)

    if mode == "point":
        return collate_point
    if mode == "face":
        # collate_face รองรับ x_c/x_n อยู่แล้ว (ถ้ามีใน sample)
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
    "TwoStream24FaceDataset",  # ✅ เพิ่ม export เผื่ออยาก import ตรง ๆ
    "GraphDataset",
    "collate_point",
    "collate_face",
    "collate_graph",
    "build_dataset",
    "build_collate_fn",
]