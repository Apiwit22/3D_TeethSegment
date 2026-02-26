# dental_seg_app/app/core/types.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional
import numpy as np


@dataclass
class MeshData:
    """
    Unified mesh struct for the app (face-based inference)
    """
    pos: np.ndarray          # (Nv,3) float32
    faces: np.ndarray        # (F,3) int64
    path: str = ""
    arch: str = "lower"      # upper/lower


@dataclass
class PipelineResult:
    mesh_in: MeshData
    mesh_proc: MeshData
    labels_face: np.ndarray  # (F,) int64 (0..15 teeth, 16 gingiva)
    num_classes: int
    meta: Dict[str, Any]