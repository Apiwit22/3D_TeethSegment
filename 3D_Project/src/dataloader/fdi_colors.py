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
    (255, 100, 140): 31,
    (255, 150, 170): 32,
    (255, 190, 170): 33,
    (255, 230, 150): 34,
    (180, 255, 150): 35,
    (120, 255, 120): 36,
    (120, 255, 200): 37,
    (150, 255, 255): 38,
}

# Quadrant 4 – Lower Right (41–48)
LOWER_41_48: Dict[RGB, int] = {
    (255, 120, 160): 41,
    (255, 170, 170): 42,
    (255, 210, 170): 43,
    (255, 240, 170): 44,
    (200, 255, 170): 45,
    (150, 255, 150): 46,
    (150, 255, 220): 47,
    (170, 240, 255): 48,
}

# Gingiva (GT)
GINGIVA_RGB: RGB = (255, 180, 200)

# Label index used when num_classes >= 17 (16 teeth + gingiva)
GINGIVA_LABEL = 16
GINGIVA_LABEL_17 = GINGIVA_LABEL  # backward-compatible alias

# ============================================================
# Position-aligned label16 ordering per arch
# (you already fixed orientation, so this ordering is stable)
# ============================================================
FDI_LIST_UPPER_16 = [11, 12, 13, 14, 15, 16, 17, 18, 21, 22, 23, 24, 25, 26, 27, 28]
FDI_LIST_LOWER_16 = [31, 32, 33, 34, 35, 36, 37, 38, 41, 42, 43, 44, 45, 46, 47, 48]

FDI_TO_LABEL16_UPPER = {fdi: i for i, fdi in enumerate(FDI_LIST_UPPER_16)}
FDI_TO_LABEL16_LOWER = {fdi: i for i, fdi in enumerate(FDI_LIST_LOWER_16)}

LABEL16_TO_FDI_UPPER = {i: fdi for fdi, i in FDI_TO_LABEL16_UPPER.items()}
LABEL16_TO_FDI_LOWER = {i: fdi for fdi, i in FDI_TO_LABEL16_LOWER.items()}

# ============================================================
# Combined RGB->FDI maps
# ============================================================
_RGB_TO_FDI_UPPER: Dict[RGB, int] = {**UPPER_11_18, **UPPER_21_28}
_RGB_TO_FDI_LOWER: Dict[RGB, int] = {**LOWER_31_38, **LOWER_41_48}

# FDI->RGB (invert)
FDI_TO_RGB_UPPER: Dict[int, RGB] = {fdi: rgb for rgb, fdi in _RGB_TO_FDI_UPPER.items()}
FDI_TO_RGB_LOWER: Dict[int, RGB] = {fdi: rgb for rgb, fdi in _RGB_TO_FDI_LOWER.items()}

# ============================================================
# Fast vectorized helpers (RGB->label)
# ============================================================
def _rgb_to_key(rgb: RGB) -> int:
    r, g, b = rgb
    return (int(r) << 16) | (int(g) << 8) | int(b)

# Palette RGB arrays in *label16 order* (for nearest lookup)
_PAL_RGB_UPPER = np.array([FDI_TO_RGB_UPPER[fdi] for fdi in FDI_LIST_UPPER_16], dtype=np.int16)  # (16,3)
_PAL_RGB_LOWER = np.array([FDI_TO_RGB_LOWER[fdi] for fdi in FDI_LIST_LOWER_16], dtype=np.int16)  # (16,3)
_PAL_LBL16 = np.arange(16, dtype=np.int64)  # (16,)

# ---- FIXED: cleaner & explicit construction
_UPPER_KEY_TO_LBL16 = {
    _rgb_to_key(tuple(map(int, rgb))): i
    for i, rgb in enumerate(_PAL_RGB_UPPER.tolist())
}
_LOWER_KEY_TO_LBL16 = {
    _rgb_to_key(tuple(map(int, rgb))): i
    for i, rgb in enumerate(_PAL_RGB_LOWER.tolist())
}

_UPPER_KEYS = np.array(sorted(_UPPER_KEY_TO_LBL16.keys()), dtype=np.int64)
_UPPER_VALS = np.array([_UPPER_KEY_TO_LBL16[k] for k in _UPPER_KEYS], dtype=np.int64)

_LOWER_KEYS = np.array(sorted(_LOWER_KEY_TO_LBL16.keys()), dtype=np.int64)
_LOWER_VALS = np.array([_LOWER_KEY_TO_LBL16[k] for k in _LOWER_KEYS], dtype=np.int64)

_GING_KEY = _rgb_to_key(GINGIVA_RGB)

# ============================================================
# Main API
# ============================================================
@dataclass(frozen=True)
class FDIColorMap:
    """
    Color mapping utilities:
      - is_gingiva_rgb(rgb)
      - rgb_to_fdi(arch, rgb)
      - rgb_to_label(arch, rgb, num_classes=16/17)
      - rgb_array_to_label_array(arch, colors[N,3], num_classes=16/17)
      - label_to_rgb(arch, label, num_classes=16/17)
    """

    # -------------------------
    # Scalar utilities
    # -------------------------
    @staticmethod
    def _coerce_rgb(rgb) -> RGB:
        """
        Accept:
          - tuple/list/np-array of 3 values
          - values in 0..255 or floats in 0..1
        Return: (r,g,b) int in 0..255 (clipped)
        """
        r, g, b = rgb
        r = float(r)
        g = float(g)
        b = float(b)

        if max(r, g, b) <= 1.0:
            r, g, b = r * 255.0, g * 255.0, b * 255.0

        r = int(round(r))
        g = int(round(g))
        b = int(round(b))

        r = 0 if r < 0 else (255 if r > 255 else r)
        g = 0 if g < 0 else (255 if g > 255 else g)
        b = 0 if b < 0 else (255 if b > 255 else b)
        return (r, g, b)

    @staticmethod
    def is_gingiva_rgb(rgb) -> bool:
        rgb = FDIColorMap._coerce_rgb(rgb)
        return rgb == GINGIVA_RGB

    def rgb_to_fdi(self, arch: str, rgb) -> Optional[int]:
        """
        Return FDI tooth id (11..48) for teeth.
        Return None for gingiva or unknown.
        """
        arch = arch.lower()
        rgb = self._coerce_rgb(rgb)

        if rgb == GINGIVA_RGB:
            return None

        if arch == "upper":
            return _RGB_TO_FDI_UPPER.get(rgb, None)

        if arch == "lower":
            return _RGB_TO_FDI_LOWER.get(rgb, None)

        raise ValueError(f"arch must be upper/lower, got: {arch}")

    def rgb_to_label(
        self,
        arch: str,
        rgb,
        *,
        num_classes: int = 16,
        ignore_index: int = -1,
        unknown_policy: UnknownPolicy = "raise",  # raise/ignore/nearest
        tol: int = 20,  # only used if unknown_policy="nearest"
    ) -> int:
        """
        Map a single RGB to:
          - label 0..15 for tooth (position-aligned per arch)
          - label 16 for gingiva if num_classes>=17
          - ignore_index for gingiva if num_classes==16
        Unknown RGB handling:
          - raise: error
          - ignore: ignore_index
          - nearest: map to nearest palette color within L_inf <= tol, else ignore_index
        """
        arch = arch.lower()
        unknown_policy = str(unknown_policy).lower()
        rgb = self._coerce_rgb(rgb)

        # gingiva
        if rgb == GINGIVA_RGB:
            return GINGIVA_LABEL if int(num_classes) >= 17 else ignore_index

        # exact teeth match
        fdi = self.rgb_to_fdi(arch, rgb)
        if fdi is not None:
            if arch == "upper":
                return FDI_TO_LABEL16_UPPER[fdi]
            if arch == "lower":
                return FDI_TO_LABEL16_LOWER[fdi]

        # unknown
        if unknown_policy == "ignore":
            return ignore_index
        if unknown_policy == "nearest":
            c = np.array(rgb, dtype=np.int16)[None, :]  # (1,3)
            pal = _PAL_RGB_UPPER if arch == "upper" else _PAL_RGB_LOWER
            dist = np.max(np.abs(c[:, None, :] - pal[None, :, :]), axis=2)  # (1,16) L_inf
            j = int(np.argmin(dist[0]))
            if int(dist[0, j]) <= int(tol):
                return int(_PAL_LBL16[j])
            return ignore_index
        raise ValueError(f"Unknown RGB for {arch}: {rgb}")

    # -------------------------
    # Vectorized utilities
    # -------------------------
    @staticmethod
    def rgb_array_to_label_array(
        arch: str,
        colors: np.ndarray,  # (N,3) uint8/int/float
        *,
        num_classes: int = 16,
        ignore_index: int = -1,
        unknown_policy: UnknownPolicy = "raise",  # raise/ignore/nearest
        tol: int = 20,  # used if nearest
    ) -> np.ndarray:
        """
        Vectorized RGB->label mapping for (N,3) colors.
        Output dtype: int64, shape (N,)

        - Exact match: O(N log 16) via searchsorted
        - Nearest: O(M*16) for unknowns only (M = number of unknown colors)
        """
        arch = arch.lower()
        unknown_policy = str(unknown_policy).lower()

        c = np.asarray(colors)
        if c.ndim != 2 or c.shape[1] != 3:
            raise ValueError(f"colors must be (N,3), got {c.shape}")

        # coerce to uint8 0..255
        if np.issubdtype(c.dtype, np.floating):
            if float(np.max(c)) <= 1.0:
                c = c * 255.0
            c = np.rint(np.clip(c, 0, 255)).astype(np.uint8)
        else:
            c = np.clip(c, 0, 255).astype(np.uint8)

        r = c[:, 0].astype(np.int64)
        g = c[:, 1].astype(np.int64)
        b = c[:, 2].astype(np.int64)
        keys = (r << 16) | (g << 8) | b

        if arch == "upper":
            pal_keys, pal_vals = _UPPER_KEYS, _UPPER_VALS
            pal_rgb = _PAL_RGB_UPPER
        elif arch == "lower":
            pal_keys, pal_vals = _LOWER_KEYS, _LOWER_VALS
            pal_rgb = _PAL_RGB_LOWER
        else:
            raise ValueError(f"arch must be upper/lower, got: {arch}")

        out = np.full((len(keys),), int(ignore_index), dtype=np.int64)

        # gingiva
        ging_mask = (keys == _GING_KEY)
        if int(num_classes) >= 17:
            out[ging_mask] = int(GINGIVA_LABEL)

        # exact teeth match via searchsorted
        idx = np.searchsorted(pal_keys, keys)
        ok = (idx >= 0) & (idx < len(pal_keys)) & (pal_keys[idx] == keys)
        out[ok] = pal_vals[idx[ok]]

        # unknown colors
        unk = (~ging_mask) & (~ok)
        if np.any(unk):
            if unknown_policy == "ignore":
                return out
            if unknown_policy == "nearest":
                cu = c[unk].astype(np.int16)  # (M,3)
                dist = np.max(np.abs(cu[:, None, :] - pal_rgb[None, :, :]), axis=2)  # (M,16)
                j = np.argmin(dist, axis=1)
                dmin = dist[np.arange(len(j)), j]
                hit = dmin <= int(tol)
                unk_idx = np.where(unk)[0]
                out[unk_idx[hit]] = _PAL_LBL16[j[hit]]
                return out

            bad = tuple(int(x) for x in c[np.where(unk)[0][0]].tolist())
            raise ValueError(f"Unknown RGB for {arch}: {bad}")

        return out

    # -------------------------
    # Label -> RGB
    # -------------------------
    def label_to_rgb(self, arch: str, label: int, *, num_classes: int = 16) -> RGB:
        arch = arch.lower()
        label = int(label)

        if int(num_classes) >= 17 and label == GINGIVA_LABEL:
            return GINGIVA_RGB

        if label < 0 or label > 15:
            raise KeyError(
                f"Invalid tooth label={label} for arch={arch} (expected 0..15 or gingiva=16)"
            )

        if arch == "upper":
            fdi = LABEL16_TO_FDI_UPPER[label]
            rgb = FDI_TO_RGB_UPPER.get(fdi)
            if rgb is None:
                raise KeyError(f"Cannot find RGB for arch={arch}, label={label}, fdi={fdi}")
            return rgb

        if arch == "lower":
            fdi = LABEL16_TO_FDI_LOWER[label]
            rgb = FDI_TO_RGB_LOWER.get(fdi)
            if rgb is None:
                raise KeyError(f"Cannot find RGB for arch={arch}, label={label}, fdi={fdi}")
            return rgb

        raise ValueError(f"arch must be upper/lower, got: {arch}")

    # -------------------------
    # Backward-compatible names
    # -------------------------
    def rgb_to_label16(
        self,
        arch: str,
        rgb,
        *,
        ignore_index: int = -1,
        unknown_policy: UnknownPolicy = "raise",
        tol: int = 20,
    ) -> int:
        return self.rgb_to_label(
            arch,
            rgb,
            num_classes=16,
            ignore_index=ignore_index,
            unknown_policy=unknown_policy,
            tol=tol,
        )

    def label16_to_rgb(self, arch: str, label: int) -> RGB:
        return self.label_to_rgb(arch, label, num_classes=16)
