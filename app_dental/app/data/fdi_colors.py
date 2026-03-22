# src/dataloader/fdi_colors.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, Optional, Literal

import numpy as np

RGB = Tuple[int, int, int]
UnknownPolicy = Literal["raise", "ignore", "nearest"]

# ============================================================
# FDI palettes (RGB -> FDI)
# ============================================================
# Quadrant 1 – Upper Right (11–18)
UPPER_11_18: Dict[RGB, int] = {
    (255, 0, 0): 11,
    (255, 128, 0): 12,
    (255, 200, 0): 13,
    (255, 255, 0): 14,
    (170, 255, 0): 15,
    (0, 255, 0): 16,
    (0, 255, 170): 17,
    (0, 255, 255): 18,
}

# Quadrant 2 – Upper Left (21–28)
UPPER_21_28: Dict[RGB, int] = {
    (255, 60, 60): 21,
    (255, 150, 60): 22,
    (255, 200, 60): 23,
    (240, 240, 0): 24,
    (170, 255, 60): 25,
    (0, 200, 0): 26,
    (0, 200, 200): 27,
    (60, 240, 240): 28,
}

# Quadrant 3 – Lower Left (31–38)
LOWER_31_38: Dict[RGB, int] = {
    (255, 0, 150): 31,
    (255, 0, 255): 32,
    (200, 0, 255): 33,
    (140, 0, 255): 34,
    (0, 0, 255): 35,
    (0, 120, 255): 36,
    (0, 200, 255): 37,
    (0, 255, 200): 38,
}

# Quadrant 4 – Lower Right (41–48)
LOWER_41_48: Dict[RGB, int] = {
    (255, 80, 170): 41,
    (255, 80, 255): 42,
    (200, 80, 255): 43,
    (140, 80, 255): 44,
    (60, 80, 255): 45,
    (60, 150, 255): 46,
    (60, 220, 255): 47,
    (60, 255, 220): 48,
}

# ------------------------------------------------------------
# unified dict
# ------------------------------------------------------------
RGB_TO_FDI: Dict[RGB, int] = {}
RGB_TO_FDI.update(UPPER_11_18)
RGB_TO_FDI.update(UPPER_21_28)
RGB_TO_FDI.update(LOWER_31_38)
RGB_TO_FDI.update(LOWER_41_48)

FDI_TO_RGB: Dict[int, RGB] = {v: k for k, v in RGB_TO_FDI.items()}

# ============================================================
# 16-class label scheme per arch (0..15 teeth) + gingiva=16
# ------------------------------------------------------------
# Upper: 11..18 + 21..28
# Lower: 31..38 + 41..48
# ============================================================
FDI_LIST_UPPER_16 = [11, 12, 13, 14, 15, 16, 17, 18, 21, 22, 23, 24, 25, 26, 27, 28]
FDI_LIST_LOWER_16 = [31, 32, 33, 34, 35, 36, 37, 38, 41, 42, 43, 44, 45, 46, 47, 48]

LABEL16_TO_FDI_UPPER = {i: fdi for i, fdi in enumerate(FDI_LIST_UPPER_16)}
LABEL16_TO_FDI_LOWER = {i: fdi for i, fdi in enumerate(FDI_LIST_LOWER_16)}

FDI_UPPER_TO_LABEL16 = {fdi: i for i, fdi in enumerate(FDI_LIST_UPPER_16)}
FDI_LOWER_TO_LABEL16 = {fdi: i for i, fdi in enumerate(FDI_LIST_LOWER_16)}

GINGIVA_LABEL = 16


def _rgb_dist2(a: RGB, b: RGB) -> int:
    return int((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2)


def _nearest_rgb(rgb: RGB) -> RGB:
    best = None
    best_d = 10**18
    for k in RGB_TO_FDI.keys():
        d = _rgb_dist2(rgb, k)
        if d < best_d:
            best_d = d
            best = k
    assert best is not None
    return best


@dataclass
class FDIColorMap:
    """
    map between:
      - fdi number (11..48)
      - rgb (R,G,B)
      - label16 per arch (0..15)
      - label17 scheme (0..15 teeth + 16=gingiva)
    """

    unknown_policy: UnknownPolicy = "raise"

    def rgb_to_fdi(self, rgb: RGB) -> Optional[int]:
        if rgb in RGB_TO_FDI:
            return RGB_TO_FDI[rgb]
        if self.unknown_policy == "ignore":
            return None
        if self.unknown_policy == "nearest":
            return RGB_TO_FDI[_nearest_rgb(rgb)]
        raise KeyError(f"Unknown RGB {rgb}")

    def fdi_to_rgb(self, fdi: int) -> RGB:
        if int(fdi) not in FDI_TO_RGB:
            raise KeyError(f"Unknown FDI {fdi}")
        return FDI_TO_RGB[int(fdi)]

    def label16_to_fdi(self, arch: str, label16: int) -> int:
        arch = (arch or "").lower()
        if arch == "upper":
            return LABEL16_TO_FDI_UPPER[int(label16)]
        if arch == "lower":
            return LABEL16_TO_FDI_LOWER[int(label16)]
        raise ValueError(f"arch must be upper/lower, got {arch}")

    def fdi_to_label16(self, arch: str, fdi: int) -> int:
        arch = (arch or "").lower()
        if arch == "upper":
            return FDI_UPPER_TO_LABEL16[int(fdi)]
        if arch == "lower":
            return FDI_LOWER_TO_LABEL16[int(fdi)]
        raise ValueError(f"arch must be upper/lower, got {arch}")

    def label_to_rgb(self, arch: str, label: int, *, num_classes: int = 17) -> RGB:
        """
        label:
          - 0..15 teeth
          - 16 gingiva (when num_classes>=17)
        """
        label = int(label)
        if num_classes >= 17 and label == GINGIVA_LABEL:
            return (200, 200, 200)
        fdi = self.label16_to_fdi(arch, label)
        return self.fdi_to_rgb(fdi)

    def rgb_to_label(self, arch: str, rgb: RGB, *, num_classes: int = 17) -> Optional[int]:
        """
        reverse mapping (only for teeth palette)
        """
        fdi = self.rgb_to_fdi(rgb)
        if fdi is None:
            return None
        return self.fdi_to_label16(arch, fdi)


# convenience aliases for external code
__all__ = [
    "RGB",
    "FDIColorMap",
    "FDI_LIST_UPPER_16",
    "FDI_LIST_LOWER_16",
    "LABEL16_TO_FDI_UPPER",
    "LABEL16_TO_FDI_LOWER",
    "GINGIVA_LABEL",
]