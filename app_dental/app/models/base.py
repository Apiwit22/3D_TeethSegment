# dental_seg_app/app/models/base.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Protocol
import numpy as np

from app.core.types import MeshData


@dataclass
class RunnerOutput:
    labels_face: np.ndarray
    num_classes: int
    meta: Dict[str, Any]


class BaseRunner(Protocol):
    def infer(self, mesh: MeshData, *, fidx: np.ndarray | None) -> RunnerOutput:
        ...