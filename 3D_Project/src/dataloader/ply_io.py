# src/dataloader/ply_io.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple, Iterable

import numpy as np

try:
    from plyfile import PlyData  # type: ignore
except Exception as e:
    raise ImportError("Missing dependency: plyfile. Install with: pip install plyfile") from e


@dataclass
class PlyMesh:
    pos: np.ndarray                    # (Nv,3) float32
    normals: Optional[np.ndarray]      # (Nv,3) float32 or None
    rgb: Optional[np.ndarray]          # (Nv,3) uint8 or None (VERTEX COLOR)
    faces: Optional[np.ndarray]        # (F,3) int64 or None
    face_rgb: Optional[np.ndarray]     # (F,3) uint8 or None (FACE COLOR)
    face_alpha: Optional[np.ndarray]   # (F,)  uint8 or None
    meta: Dict[str, Any]


# -------------------------
# coercers
# -------------------------
def _coerce_u8(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    if x.dtype == np.uint8:
        return x
    xf = x.astype(np.float32)
    if xf.size == 0:
        return xf.astype(np.uint8)
    mx = float(np.nanmax(xf))
    if mx <= 1.0:
        xf = np.clip(np.rint(xf * 255.0), 0, 255)
    else:
        xf = np.clip(np.rint(xf), 0, 255)
    return xf.astype(np.uint8)


def _safe_norm(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return np.linalg.norm(v, axis=1, keepdims=True) + eps


def _normalize_normals(n: np.ndarray) -> np.ndarray:
    n = n.astype(np.float32, copy=False)
    return n / _safe_norm(n)


def _compute_vertex_normals(pos: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """
    Area-weighted vertex normals from triangle faces.
    pos: (Nv,3)
    faces: (F,3)
    return: (Nv,3) normalized
    """
    pos = np.asarray(pos, dtype=np.float32)
    faces = np.asarray(faces, dtype=np.int64)

    nv = int(pos.shape[0])
    vn = np.zeros((nv, 3), dtype=np.float32)

    v0 = pos[faces[:, 0]]
    v1 = pos[faces[:, 1]]
    v2 = pos[faces[:, 2]]
    fn = np.cross(v1 - v0, v2 - v0)  # magnitude ~ 2*area

    np.add.at(vn, faces[:, 0], fn)
    np.add.at(vn, faces[:, 1], fn)
    np.add.at(vn, faces[:, 2], fn)

    return _normalize_normals(vn)


# -------------------------
# vertex parsing
# -------------------------
def _pick_first_existing(names: tuple[str, ...], candidates: tuple[str, ...]) -> Optional[str]:
    for c in candidates:
        if c in names:
            return c
    return None


def _get_vertex_props(vtx) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    names = vtx.dtype.names or ()
    if not all(k in names for k in ("x", "y", "z")):
        raise ValueError("PLY missing vertex positions x,y,z")

    pos = np.stack([vtx["x"], vtx["y"], vtx["z"]], axis=1).astype(np.float32)

    normals = None
    if all(k in names for k in ("nx", "ny", "nz")):
        normals = np.stack([vtx["nx"], vtx["ny"], vtx["nz"]], axis=1).astype(np.float32)

    rgb = None
    # common vertex color names
    if all(k in names for k in ("red", "green", "blue")):
        rgb = np.stack([vtx["red"], vtx["green"], vtx["blue"]], axis=1)
    elif all(k in names for k in ("diffuse_red", "diffuse_green", "diffuse_blue")):
        rgb = np.stack([vtx["diffuse_red"], vtx["diffuse_green"], vtx["diffuse_blue"]], axis=1)
    elif all(k in names for k in ("r", "g", "b")):
        rgb = np.stack([vtx["r"], vtx["g"], vtx["b"]], axis=1)

    if rgb is not None:
        rgb = _coerce_u8(rgb)

    return pos, normals, rgb


# -------------------------
# face parsing (robust + fast path)
# -------------------------
def _triangulate_faces(face_indices: Iterable) -> Optional[np.ndarray]:
    """
    Fan triangulation:
      poly [a,b,c,d] -> (a,b,c), (a,c,d)
    Return (F,3) int64 or None
    """
    tris: list[tuple[int, int, int]] = []
    for poly in face_indices:
        if poly is None:
            continue
        arr = np.asarray(poly, dtype=np.int64).reshape(-1)
        if arr.size < 3:
            continue
        if arr.size == 3:
            tris.append((int(arr[0]), int(arr[1]), int(arr[2])))
        else:
            a0 = int(arr[0])
            for i in range(1, int(arr.size) - 1):
                tris.append((a0, int(arr[i]), int(arr[i + 1])))
    if not tris:
        return None
    return np.asarray(tris, dtype=np.int64)


def _get_faces_and_face_props(ply: PlyData) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Returns:
      faces: (F,3) int64 or None
      face_rgb: (F,3) uint8 or None
      face_alpha: (F,) uint8 or None
    """
    if "face" not in ply:
        return None, None, None

    face = ply["face"].data
    names = face.dtype.names or ()

    idx_field = _pick_first_existing(names, ("vertex_indices", "vertex_index", "indices"))
    if idx_field is None:
        return None, None, None

    faces_raw = face[idx_field]

    # ---- FAST PATH: already triangles numeric (F,3)
    faces = None
    if isinstance(faces_raw, np.ndarray) and faces_raw.dtype != object:
        arr = np.asarray(faces_raw)
        if arr.ndim == 2 and arr.shape[1] == 3:
            faces = arr.astype(np.int64, copy=False)

    # ---- fallback: object polygons -> triangulate
    if faces is None:
        faces = _triangulate_faces(faces_raw)

    # ---- face rgb (common names)
    face_rgb = None
    if all(k in names for k in ("red", "green", "blue")):
        face_rgb = np.stack([face["red"], face["green"], face["blue"]], axis=1)
    elif all(k in names for k in ("diffuse_red", "diffuse_green", "diffuse_blue")):
        face_rgb = np.stack([face["diffuse_red"], face["diffuse_green"], face["diffuse_blue"]], axis=1)
    elif all(k in names for k in ("r", "g", "b")):
        face_rgb = np.stack([face["r"], face["g"], face["b"]], axis=1)

    if face_rgb is not None:
        face_rgb = _coerce_u8(face_rgb)

    # ---- alpha (support more names)
    face_alpha = None
    a_field = _pick_first_existing(names, ("alpha", "opacity", "a"))
    if a_field is not None:
        face_alpha = _coerce_u8(face[a_field])

    # If faces were triangulated from polygons, per-face colors from source won't match
    if faces is not None:
        if faces.ndim != 2 or faces.shape[1] != 3:
            raise ValueError(f"Faces must be triangles (F,3). Got shape={faces.shape}")

        if face_rgb is not None and len(face_rgb) != len(faces):
            face_rgb = None
        if face_alpha is not None and len(face_alpha) != len(faces):
            face_alpha = None

    return faces, face_rgb, face_alpha


# -------------------------
# geometry transforms
# -------------------------
def recenter_scale_unit_sphere(pos: np.ndarray) -> np.ndarray:
    c = pos.mean(axis=0, keepdims=True)
    pos0 = pos - c
    r = np.linalg.norm(pos0, axis=1).max()
    if r < 1e-12:
        return pos0.astype(np.float32)
    return (pos0 / r).astype(np.float32)


def pca_align(pos: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns (aligned_pos, R) where aligned_pos = (pos - mean) @ R
    NOTE: rotate normals with R as well.
    """
    x = pos - pos.mean(axis=0, keepdims=True)
    cov = (x.T @ x) / max(len(x), 1)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    R = eigvecs[:, order]  # 3x3

    if np.linalg.det(R) < 0:
        R[:, -1] *= -1

    xr = x @ R

    s = np.sign(xr.mean(axis=0) + 1e-9).astype(np.float32)
    s[s == 0] = 1.0
    xr = xr * s
    R = R * s.reshape(1, 3)

    return xr.astype(np.float32), R.astype(np.float32)


# -------------------------
# main reader
# -------------------------
def read_ply(path: str, *, normalize: bool = True, align_pca_flag: bool = False) -> PlyMesh:
    ply = PlyData.read(path)
    if "vertex" not in ply:
        raise ValueError(f"PLY missing vertex element: {path}")

    vtx = ply["vertex"].data
    pos, normals, rgb = _get_vertex_props(vtx)
    faces, face_rgb, face_alpha = _get_faces_and_face_props(ply)

    R_used = None
    if align_pca_flag:
        pos, R_used = pca_align(pos)

    if normalize:
        pos = recenter_scale_unit_sphere(pos)

    # ---- normals handling (CRITICAL for Fast-TGCN 24D)
    if normals is None:
        if faces is not None and int(faces.shape[0]) > 0:
            normals = _compute_vertex_normals(pos, faces)  # compute in transformed space
        else:
            normals = np.zeros_like(pos, dtype=np.float32)
    else:
        normals = normals.astype(np.float32, copy=False)
        if R_used is not None:
            normals = normals @ R_used
        normals = _normalize_normals(normals)

    meta = {
        "path": path,
        "nv": int(pos.shape[0]),
        "nf": int(faces.shape[0]) if faces is not None else 0,
        "has_vertex_rgb": bool(rgb is not None),
        "has_face_rgb": bool(face_rgb is not None),
        "has_face_alpha": bool(face_alpha is not None),
        "has_normals": True,
        "normalize": bool(normalize),
        "align_pca": bool(align_pca_flag),
    }

    return PlyMesh(
        pos=pos,
        normals=normals,
        rgb=rgb,
        faces=faces,
        face_rgb=face_rgb,
        face_alpha=face_alpha,
        meta=meta,
    )
