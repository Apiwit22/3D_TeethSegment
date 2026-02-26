# src/dataloader/base.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal, Dict, Tuple

import numpy as np
from torch.utils.data import Dataset

from src.dataloader.arch import parse_arch_from_filename
from src.dataloader.ply_io import read_ply, PlyMesh

# import palette + label maps
from src.dataloader.fdi_colors import (
    GINGIVA_RGB,
    GINGIVA_LABEL,
    FDI_TO_LABEL16_UPPER,
    FDI_TO_LABEL16_LOWER,
    UPPER_11_18,
    UPPER_21_28,
    LOWER_31_38,
    LOWER_41_48,
)

LabelSource = Literal["auto", "vertex", "face"]
FaceFallback = Literal["majority", "first", "none"]


def _rgb_key(rgb_u8: np.ndarray) -> np.ndarray:
    """
    rgb_u8: (N,3) uint8
    return: (N,) uint32 key = r<<16 | g<<8 | b
    """
    rgb_u8 = np.asarray(rgb_u8, dtype=np.uint32)
    return (rgb_u8[:, 0] << 16) | (rgb_u8[:, 1] << 8) | (rgb_u8[:, 2])


def _coerce_rgb_array(rgb: np.ndarray) -> np.ndarray:
    """
    Accept:
      - uint8 0..255
      - float 0..1 or float 0..255
    Return uint8 0..255 of shape (N,3)
    """
    rgb = np.asarray(rgb)
    if rgb.ndim != 2 or rgb.shape[1] != 3:
        raise ValueError(f"RGB must be (N,3). Got: {rgb.shape}")

    if rgb.dtype == np.uint8:
        return rgb

    x = rgb.astype(np.float32)
    if x.size == 0:
        return x.astype(np.uint8)

    mx = float(np.nanmax(x))
    if mx <= 1.0:
        x = np.clip(np.round(x * 255.0), 0, 255)
    else:
        x = np.clip(np.round(x), 0, 255)

    return x.astype(np.uint8)


def _build_rgbkey_to_label16_maps() -> Tuple[Dict[int, int], Dict[int, int]]:
    """
    Build dict: rgb_key -> label(0..15)
    """
    rgb_to_fdi_upper = {**UPPER_11_18, **UPPER_21_28}  # (r,g,b)->fdi
    rgb_to_fdi_lower = {**LOWER_31_38, **LOWER_41_48}

    upper_map: Dict[int, int] = {}
    lower_map: Dict[int, int] = {}

    for rgb, fdi in rgb_to_fdi_upper.items():
        r, g, b = rgb
        key = (r << 16) | (g << 8) | b
        upper_map[key] = int(FDI_TO_LABEL16_UPPER[fdi])

    for rgb, fdi in rgb_to_fdi_lower.items():
        r, g, b = rgb
        key = (r << 16) | (g << 8) | b
        lower_map[key] = int(FDI_TO_LABEL16_LOWER[fdi])

    return upper_map, lower_map


_RGBKEY_TO_LABEL16_UPPER, _RGBKEY_TO_LABEL16_LOWER = _build_rgbkey_to_label16_maps()
_GINGIVA_KEY = (GINGIVA_RGB[0] << 16) | (GINGIVA_RGB[1] << 8) | GINGIVA_RGB[2]


@dataclass
class BaseDentalDatasetConfig:
    """
    Shared dataset config for point/face/graph modes.

    - arch:
        * "upper" / "lower" => force that arch
        * "both"            => infer from filename

    - num_classes:
        * 16 = teeth-only (gingiva -> ignore_index)
        * 17 = 16 teeth + gingiva (label=16)

    - label_source:
        * "auto"   : for YOUR dataset (face-only GT) => treated as "face"
        * "vertex" : force vertex rgb as GT source; for face/graph it will be derived to face labels
        * "face"   : use face rgb as GT (requires face rgb; else fallback policy)

    - face_fallback:
        used only when face GT is requested but face_rgb missing OR when label_source="vertex".
        * "majority": vote among 3 vertex labels
        * "first"   : pick first non-ignore among 3 vertices
        * "none"    : no fallback -> follow unknown_policy
    """
    mode: str                  # point/face/graph
    arch: str                  # upper/lower/both
    normalize: bool = True
    align_pca: bool = False
    ignore_index: int = -1
    unknown_policy: str = "raise"  # raise/ignore
    require_labels: bool = True
    num_classes: int = 17          # 16 or 17 (default=17 => 16 teeth + gingiva)

    label_source: LabelSource = "auto"
    face_fallback: FaceFallback = "majority"

    def __post_init__(self):
        m = str(self.mode).lower()
        a = str(self.arch).lower()
        up = str(self.unknown_policy).lower()
        ls = str(self.label_source).lower()
        fb = str(self.face_fallback).lower()

        if m not in ("point", "face", "graph"):
            raise ValueError(f"cfg.mode must be point/face/graph, got: {self.mode}")
        if a not in ("upper", "lower", "both"):
            raise ValueError(f"cfg.arch must be upper/lower/both, got: {self.arch}")
        if up not in ("raise", "ignore"):
            raise ValueError(f"cfg.unknown_policy must be raise/ignore, got: {self.unknown_policy}")

        nc = int(self.num_classes)
        if nc not in (16, 17):
            raise ValueError(f"cfg.num_classes must be 16 or 17, got: {self.num_classes}")

        if ls not in ("auto", "vertex", "face"):
            raise ValueError(f"cfg.label_source must be auto/vertex/face, got: {self.label_source}")
        if fb not in ("majority", "first", "none"):
            raise ValueError(f"cfg.face_fallback must be majority/first/none, got: {self.face_fallback}")


class BaseDentalDataset(Dataset):
    """
    Base dataset:
      - loads PLY via read_ply()
      - provides arch inference and sampling helper
      - provides unified label extraction policy (vertex vs face)

    IMPORTANT (for your dataset):
      - GT is stored on FACE colors only.
      - Therefore, label_source="auto" is treated as "face".
      - Point-mode returns FACE labels (F,) so PointDataset can map via tri_ids -> point labels.
    """
    def __init__(self, files: List[str], cfg: BaseDentalDatasetConfig):
        self.files = list(files)
        self.cfg = cfg

    def __len__(self) -> int:
        return len(self.files)

    def get_arch(self, path: str) -> str:
        a = str(self.cfg.arch).lower()
        if a in ("upper", "lower"):
            return a
        return parse_arch_from_filename(path)

    def load_mesh(self, path: str) -> PlyMesh:
        return read_ply(
            path,
            normalize=bool(self.cfg.normalize),
            align_pca_flag=bool(self.cfg.align_pca),
        )

    @staticmethod
    def sample_indices(n: int, k: int) -> np.ndarray:
        n = int(n)
        k = int(k)
        if n <= 0:
            raise ValueError("Cannot sample from empty set.")
        if k <= 0:
            raise ValueError("k must be > 0.")
        replace = n < k
        return np.random.choice(n, size=k, replace=replace).astype(np.int64)

    def resolve_label_source(self) -> str:
        """
        returns: "vertex" or "face"
        NOTE: for your dataset, "auto" => "face"
        """
        ls = str(self.cfg.label_source).lower()
        if ls in ("vertex", "face"):
            return ls
        # auto
        return "face"

    def rgb_to_labels(self, arch: str, rgb: np.ndarray, *, num_classes: int) -> np.ndarray:
        rgb_u8 = _coerce_rgb_array(rgb)
        keys = _rgb_key(rgb_u8)

        if arch == "upper":
            mp = _RGBKEY_TO_LABEL16_UPPER
        elif arch == "lower":
            mp = _RGBKEY_TO_LABEL16_LOWER
        else:
            raise ValueError(f"arch must be upper/lower, got: {arch}")

        ukeys, inv = np.unique(keys, return_inverse=True)
        lab_u = np.full((ukeys.shape[0],), int(self.cfg.ignore_index), dtype=np.int64)

        unknown_keys = []
        for i, k in enumerate(ukeys.tolist()):
            if k == _GINGIVA_KEY:
                lab_u[i] = int(GINGIVA_LABEL) if int(num_classes) >= 17 else int(self.cfg.ignore_index)
            else:
                v = mp.get(int(k), None)
                if v is None:
                    unknown_keys.append(int(k))
                else:
                    lab_u[i] = int(v)

        if unknown_keys and str(self.cfg.unknown_policy).lower() == "raise":
            def key_to_rgb(kk: int) -> Tuple[int, int, int]:
                r = (kk >> 16) & 255
                g = (kk >> 8) & 255
                b = kk & 255
                return (r, g, b)

            sample = [key_to_rgb(k) for k in unknown_keys[:10]]
            raise ValueError(
                f"Unknown RGB colors for arch={arch}: {sample} "
                f"(total_unknown={len(unknown_keys)}). "
                f"Set unknown_policy='ignore' to drop them."
            )

        return lab_u[inv].astype(np.int64, copy=False)

    def vertex_labels(self, mesh: PlyMesh, arch: str) -> np.ndarray:
        Nv = int(mesh.pos.shape[0])
        if not self.cfg.require_labels:
            return np.full((Nv,), int(self.cfg.ignore_index), dtype=np.int64)

        if mesh.rgb is None:
            if str(self.cfg.unknown_policy).lower() == "raise":
                raise ValueError(f"Missing vertex RGB (GT) in: {mesh.meta.get('path', '')}")
            return np.full((Nv,), int(self.cfg.ignore_index), dtype=np.int64)

        return self.rgb_to_labels(arch, mesh.rgb, num_classes=int(self.cfg.num_classes))

    @staticmethod
    def _face_majority_from_vertex(tri: np.ndarray, vlab: np.ndarray, ignore_index: int) -> np.ndarray:
        labs = vlab[tri]  # (F,3)
        y0, y1, y2 = labs[:, 0], labs[:, 1], labs[:, 2]
        out = np.full((tri.shape[0],), int(ignore_index), dtype=np.int64)

        ok0 = (y0 != ignore_index)
        ok1 = (y1 != ignore_index)
        ok2 = (y2 != ignore_index)
        any_ok = ok0 | ok1 | ok2
        if not np.any(any_ok):
            return out

        m0 = ok0 & ((y0 == y1) | (y0 == y2))
        out[m0] = y0[m0]

        m1 = (out == ignore_index) & ok1 & (y1 == y2)
        out[m1] = y1[m1]

        rem = (out == ignore_index) & any_ok
        if np.any(rem):
            pick0 = rem & ok0
            out[pick0] = y0[pick0]
            rem2 = (out == ignore_index) & rem
            pick1 = rem2 & ok1
            out[pick1] = y1[pick1]
            rem3 = (out == ignore_index) & rem2
            pick2 = rem3 & ok2
            out[pick2] = y2[pick2]

        return out

    def face_labels(self, mesh: PlyMesh, arch: str) -> np.ndarray:
        """
        Face label policy (RESPECTS label_source):
        - If label_source != "vertex" and face_rgb exists: map directly
        - Else fallback via vertex rgb depending on face_fallback
        """
        if mesh.faces is None:
            raise ValueError("Face labels requested but mesh.faces is None.")

        tri = mesh.faces.astype(np.int64)
        F = int(tri.shape[0])

        if not self.cfg.require_labels:
            return np.full((F,), int(self.cfg.ignore_index), dtype=np.int64)

        # if label_source="vertex", DO NOT use face_rgb even if it exists
        use_face_rgb = (str(self.cfg.label_source).lower() != "vertex")

        # 1) face rgb exists -> direct mapping (only when allowed)
        if use_face_rgb and (mesh.face_rgb is not None):
            if mesh.face_rgb.shape[0] != F:
                raise ValueError(f"face_rgb length mismatch: face_rgb={mesh.face_rgb.shape[0]} faces={F}")
            return self.rgb_to_labels(arch, mesh.face_rgb, num_classes=int(self.cfg.num_classes))

        # 2) fallback via vertex labels
        fb = str(self.cfg.face_fallback).lower()
        if fb == "none":
            if str(self.cfg.unknown_policy).lower() == "raise":
                raise ValueError("Face fallback disabled (face_fallback=none) and face_rgb not used/available.")
            return np.full((F,), int(self.cfg.ignore_index), dtype=np.int64)

        if mesh.rgb is None:
            if str(self.cfg.unknown_policy).lower() == "raise":
                raise ValueError("Missing vertex rgb -> cannot derive face labels.")
            return np.full((F,), int(self.cfg.ignore_index), dtype=np.int64)

        vlab = self.vertex_labels(mesh, arch)  # (Nv,)

        if fb == "first":
            labs = vlab[tri]  # (F,3)
            ign = int(self.cfg.ignore_index)
            out = np.full((F,), ign, dtype=np.int64)
            for j in range(3):
                m = (out == ign) & (labs[:, j] != ign)
                out[m] = labs[m, j]
            return out

        return self._face_majority_from_vertex(tri, vlab, int(self.cfg.ignore_index))

    def get_labels_for_mode(self, mesh: PlyMesh, arch: str) -> np.ndarray:
        """
        Unified entry point with STRICT shape:
          - point -> (F,)  (face labels; PointDataset will map tri_ids -> point labels)
          - face  -> (F,)
          - graph -> (F,)
        """
        mode = str(self.cfg.mode).lower()
        src = self.resolve_label_source()

        if mesh.faces is None:
            raise ValueError(f"mode={mode} requires faces to build labels (your GT is face_rgb).")

        # For your dataset: always return face-wise labels for all modes.
        # If src=="face": use face_rgb directly (when allowed).
        # If src=="vertex": derive via fallback (requires mesh.rgb).
        if src in ("face", "vertex"):
            return self.face_labels(mesh, arch)

        # should never reach (resolve_label_source only returns face/vertex)
        return self.face_labels(mesh, arch)
