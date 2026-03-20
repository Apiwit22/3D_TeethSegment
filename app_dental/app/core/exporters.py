from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import trimesh
from plyfile import PlyData, PlyElement

from app.data.fdi_colors import (
    FDI_LIST_LOWER_16,
    FDI_LIST_UPPER_16,
    GINGIVA_LABEL,
)

RGB = Tuple[int, int, int]


def _ensure_vertices_faces(v: np.ndarray, f: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    v = np.asarray(v, dtype=np.float32)
    f = np.asarray(f, dtype=np.int64)

    if v.ndim != 2 or v.shape[1] != 3:
        raise ValueError(f"vertices must have shape (N,3), got {v.shape}")
    if f.ndim != 2 or f.shape[1] != 3:
        raise ValueError(f"faces must have shape (F,3), got {f.shape}")
    if len(v) == 0:
        raise ValueError("vertices is empty")
    if len(f) == 0:
        raise ValueError("faces is empty")

    fmin = int(f.min())
    fmax = int(f.max())
    if fmin < 0 or fmax >= len(v):
        raise ValueError(f"faces contain invalid vertex index range [{fmin}, {fmax}] for {len(v)} vertices")

    return v, f


def export_colored_ply_facecolors(
    out_path: str | Path,
    v: np.ndarray,
    f: np.ndarray,
    face_rgb: np.ndarray,
) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    v, f = _ensure_vertices_faces(v, f)
    rgb = np.asarray(face_rgb, dtype=np.uint8)

    if rgb.ndim != 2 or rgb.shape[1] != 3:
        raise ValueError(f"face_rgb must have shape (F,3), got {rgb.shape}")
    if len(rgb) != len(f):
        raise ValueError(f"face_rgb length must match faces length: {len(rgb)} != {len(f)}")

    v_el = np.empty(len(v), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])
    v_el["x"], v_el["y"], v_el["z"] = v[:, 0], v[:, 1], v[:, 2]

    face_idx = np.empty(len(f), dtype=object)
    for i in range(len(f)):
        face_idx[i] = [int(f[i, 0]), int(f[i, 1]), int(f[i, 2])]

    f_el = np.empty(
        len(f),
        dtype=[
            ("vertex_indices", "O"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
        ],
    )
    f_el["vertex_indices"] = face_idx
    f_el["red"], f_el["green"], f_el["blue"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]

    ply = PlyData(
        [
            PlyElement.describe(v_el, "vertex"),
            PlyElement.describe(f_el, "face"),
        ],
        text=False,
    )
    ply.write(str(out_path))
    return out_path


def export_mesh_ply(out_path: str | Path, v: np.ndarray, f: np.ndarray) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    v, f = _ensure_vertices_faces(v, f)
    mesh = trimesh.Trimesh(vertices=v, faces=f, process=False)
    mesh.export(out_path)
    return out_path


def export_split_by_tooth(
    out_dir: str | Path,
    *,
    v: np.ndarray,
    f: np.ndarray,
    labels_face: np.ndarray,
    arch: str,
    num_classes: int = 17,
) -> Dict[str, Path]:
    """
    Export one .ply per tooth (FDI) and gingiva if present.
    Returns:
        dict[name -> output_path]
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    v, f = _ensure_vertices_faces(v, f)
    labels = np.asarray(labels_face, dtype=np.int64).reshape(-1)
    if len(labels) != len(f):
        raise ValueError(f"labels_face length must match faces length: {len(labels)} != {len(f)}")

    arch = (arch or "").lower()
    if arch not in ("upper", "lower"):
        raise ValueError(f"arch must be 'upper' or 'lower', got {arch!r}")

    fdi_list = FDI_LIST_UPPER_16 if arch == "upper" else FDI_LIST_LOWER_16
    mesh = trimesh.Trimesh(vertices=v, faces=f, process=False)

    out_paths: Dict[str, Path] = {}

    for lab in range(16):
        face_idx = np.where(labels == lab)[0]
        if face_idx.size == 0:
            continue

        sub = mesh.submesh([face_idx], append=True, repair=False)
        fdi = int(fdi_list[lab])
        name = f"tooth_{fdi}"
        path = out_dir / f"{name}.ply"
        sub.export(path)
        out_paths[name] = path

    if num_classes >= 17:
        gingiva_idx = np.where(labels == int(GINGIVA_LABEL))[0]
        if gingiva_idx.size > 0:
            sub = mesh.submesh([gingiva_idx], append=True, repair=False)
            path = out_dir / "gingiva.ply"
            sub.export(path)
            out_paths["gingiva"] = path

    return out_paths