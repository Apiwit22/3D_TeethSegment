from __future__ import annotations

from functools import lru_cache
from typing import Tuple
import numpy as np

from app.data.fdi_colors import FDIColorMap, GINGIVA_LABEL

RGB = Tuple[int, int, int]


@lru_cache(maxsize=4)
def _cmap() -> FDIColorMap:
    return FDIColorMap()


def ensure_arch(arch: str) -> str:
    a = (arch or "").lower()
    if a not in ("upper", "lower"):
        return "lower"
    return a


def label_array_to_rgb_face(
    labels_face: np.ndarray,
    *,
    arch: str,
    num_classes: int = 17,
    unknown_rgb: RGB = (200, 200, 200),
) -> np.ndarray:
    """
    แปลง labels_face (F,) -> rgb (F,3) ตาม palette FDI จริง
    """
    labels = np.asarray(labels_face, dtype=np.int64).reshape(-1)
    out = np.empty((labels.shape[0], 3), dtype=np.uint8)

    cmap = _cmap()
    arch = ensure_arch(arch)

    for i, lb in enumerate(labels.tolist()):
        try:
            rgb = cmap.label_to_rgb(arch, int(lb), num_classes=int(num_classes))
        except Exception:
            rgb = unknown_rgb
        out[i, 0], out[i, 1], out[i, 2] = int(rgb[0]), int(rgb[1]), int(rgb[2])

    return out