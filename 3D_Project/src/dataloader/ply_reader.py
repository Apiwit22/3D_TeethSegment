# src/dataloader/ply_reader.py
from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, Optional, Tuple, Iterable

import numpy as np


def _peek_ply_format(path: str) -> Optional[str]:
    """
    Fast header peek to detect PLY format without fully parsing the file.
    Returns: "ascii" | "binary_little_endian" | "binary_big_endian" | None
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)

    raw = p.read_bytes()[:8192]
    try:
        text = raw.decode("utf-8", errors="ignore")
    except Exception:
        return None

    for line in text.splitlines():
        s = line.strip()
        if s.startswith("format"):
            parts = s.split()
            if len(parts) >= 2:
                return parts[1].lower()
        if s == "end_header":
            break
    return None


# -------------------------
# Helpers
# -------------------------
def _coerce_rgb_array(rgb: np.ndarray) -> np.ndarray:
    """
    Accept rgb in:
      - uint8 0..255
      - float 0..1 or 0..255
    Return uint8 0..255
    """
    if rgb is None:
        return rgb
    rgb = np.asarray(rgb)

    if np.issubdtype(rgb.dtype, np.floating):
        mx = float(np.nanmax(rgb)) if rgb.size else 0.0
        if mx <= 1.0:
            rgb = rgb * 255.0
        rgb = np.clip(np.rint(rgb), 0, 255).astype(np.uint8)
        return rgb

    rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    return rgb


def _triangulate_faces(face_indices: Iterable) -> Optional[np.ndarray]:
    """
    face_indices: iterable of polygons (list/np-array) with length >=3
    Return (F,3) int64 or None
    Fan triangulation: (0,i,i+1)
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


def _pick_first_existing(names: tuple[str, ...], candidates: tuple[str, ...]) -> Optional[str]:
    for c in candidates:
        if c in names:
            return c
    return None


def _load_with_plyfile(path: str) -> Dict[str, Any]:
    """
    Primary reader using plyfile (supports ASCII + binary).

    Returns dict:
      xyz:   (Nv,3) float32
      nrm:   (Nv,3) float32 or None
      rgb_v: (Nv,3) uint8   or None
      faces: (F,3)  int64   or None
      rgb_f: (F,3)  uint8   or None
      alpha: (F,)   uint8   or None
    """
    try:
        from plyfile import PlyData  # type: ignore
    except ImportError as e:
        raise ImportError("Missing dependency: plyfile. Install with: pip install plyfile") from e

    ply = PlyData.read(path)

    if "vertex" not in ply:
        raise ValueError(f"PLY missing 'vertex' element: {path}")

    v = ply["vertex"].data
    vnames = v.dtype.names or ()
    if not all(k in vnames for k in ("x", "y", "z")):
        raise ValueError(f"PLY missing vertex x/y/z fields: {path}")

    xyz = np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float32)

    nrm = None
    if all(k in vnames for k in ("nx", "ny", "nz")):
        nrm = np.stack([v["nx"], v["ny"], v["nz"]], axis=1).astype(np.float32)

    rgb_v = None
    if all(k in vnames for k in ("red", "green", "blue")):
        rgb_v = np.stack([v["red"], v["green"], v["blue"]], axis=1)
    elif all(k in vnames for k in ("diffuse_red", "diffuse_green", "diffuse_blue")):
        rgb_v = np.stack([v["diffuse_red"], v["diffuse_green"], v["diffuse_blue"]], axis=1)

    if rgb_v is not None:
        rgb_v = _coerce_rgb_array(rgb_v)

    faces = None
    rgb_f = None
    alpha = None

    if "face" in ply:
        f = ply["face"].data
        fnames = f.dtype.names or ()

        # ---- face indices
        idx_field = _pick_first_existing(fnames, ("vertex_indices", "vertex_index", "indices"))
        faces_list = f[idx_field] if idx_field is not None else None

        if faces_list is not None:
            # faces_list is commonly dtype=object, each row is a list/np-array
            faces = _triangulate_faces(faces_list)
            # If triangulation returned None -> keep faces None
        # ---- per-face rgb (optional)
        if all(k in fnames for k in ("red", "green", "blue")):
            rgb_f = np.stack([f["red"], f["green"], f["blue"]], axis=1)
        elif all(k in fnames for k in ("diffuse_red", "diffuse_green", "diffuse_blue")):
            rgb_f = np.stack([f["diffuse_red"], f["diffuse_green"], f["diffuse_blue"]], axis=1)

        if rgb_f is not None:
            rgb_f = _coerce_rgb_array(rgb_f)

        # ---- alpha (optional)
        if "alpha" in fnames:
            alpha = np.asarray(f["alpha"])
            alpha = _coerce_rgb_array(alpha)

        # sanity: if faces exists must be (F,3)
        if faces is not None:
            if faces.ndim != 2 or faces.shape[1] != 3:
                raise ValueError(f"Triangulation failed. faces shape={faces.shape} in {path}")

            # if rgb_f/alpha exist but were for original polygons count, they won't match triangulated count
            # In your dataset you already have triangles, so this won't happen.
            # For safety: drop rgb_f/alpha if lengths mismatch.
            if rgb_f is not None and len(rgb_f) != len(faces):
                rgb_f = None
            if alpha is not None and len(alpha) != len(faces):
                alpha = None

    return dict(xyz=xyz, nrm=nrm, rgb_v=rgb_v, faces=faces, rgb_f=rgb_f, alpha=alpha)


def _parse_header_ascii(lines: list[str]) -> Tuple[int, int, list[str], int]:
    """
    Parse ASCII PLY header.
    Returns: (n_verts, n_faces, vert_prop_names, header_end_line_index_exclusive)
    """
    fmt = None
    n_verts = None
    n_faces = 0
    vert_props: list[str] = []

    i = 0
    in_vertex = False
    while i < len(lines):
        s = lines[i].strip()
        if s.startswith("format"):
            parts = s.split()
            fmt = parts[1].lower() if len(parts) >= 2 else None
            if fmt != "ascii":
                raise ValueError(f"ASCII reader only supports ASCII PLY; got format={fmt}")
        elif s.startswith("element vertex"):
            n_verts = int(s.split()[-1])
            in_vertex = True
        elif s.startswith("element face"):
            n_faces = int(s.split()[-1])
            in_vertex = False
        elif s.startswith("property") and in_vertex:
            parts = s.split()
            if len(parts) >= 3:
                vert_props.append(parts[-1])
        elif s == "end_header":
            i += 1
            break
        i += 1

    if fmt is None or n_verts is None:
        raise ValueError("Bad/unsupported PLY header (missing format/vertex count).")

    return n_verts, n_faces, vert_props, i


def _load_ascii_simple(path: str) -> Dict[str, Any]:
    """
    Fallback ASCII reader (minimal but robust for your GT format).
    Supports:
      - vertex: x y z [nx ny nz] [red green blue] [alpha]
      - face:   3 i j k [red green blue] [alpha]
    """
    p = Path(path)
    lines = p.read_text(encoding="utf-8", errors="ignore").splitlines()

    n_verts, n_faces, vert_props, start = _parse_header_ascii(lines)

    vlines = lines[start: start + n_verts]
    fstart = start + n_verts

    def col(name: str) -> Optional[int]:
        return vert_props.index(name) if name in vert_props else None

    ix, iy, iz = col("x"), col("y"), col("z")
    if ix is None or iy is None or iz is None:
        raise ValueError(f"Vertex must include x y z columns: {path}")

    inx, iny, inz = col("nx"), col("ny"), col("nz")
    ir, ig, ib = col("red"), col("green"), col("blue")
    # vertex alpha exists in some files but we ignore it here (not needed for labels)
    # ia = col("alpha")

    v = np.array([[float(x) for x in row.split()] for row in vlines], dtype=np.float32)
    xyz = v[:, [ix, iy, iz]].astype(np.float32)

    nrm = None
    if None not in (inx, iny, inz):
        nrm = v[:, [inx, iny, inz]].astype(np.float32)

    rgb_v = None
    if None not in (ir, ig, ib):
        rgb_v = v[:, [ir, ig, ib]].astype(np.float32)
        rgb_v = _coerce_rgb_array(rgb_v)

    faces = None
    rgb_f = None
    alpha = None

    if n_faces and n_faces > 0:
        flines = lines[fstart: fstart + n_faces]
        polys = []
        rgb_list = []
        a_list = []

        for row in flines:
            parts = row.split()
            if not parts:
                continue
            n = int(parts[0])
            if n < 3:
                continue

            if len(parts) < 1 + n:
                raise ValueError(f"Bad face row (too few columns): {path}")

            idx = [int(x) for x in parts[1:1 + n]]
            polys.append(idx)

            # optional per-face rgb/alpha
            # layout: n i... r g b [a]
            tail = parts[1 + n:]
            if len(tail) >= 3:
                rgb_list.append((int(tail[0]), int(tail[1]), int(tail[2])))
            if len(tail) >= 4:
                a_list.append(int(tail[3]))

        faces = _triangulate_faces(polys)
        if faces is not None:
            # if rgb/alpha were per-polygon, they don't match triangulated faces => drop for safety
            # (your GT is already triangles so this is fine either way)
            if rgb_list and len(rgb_list) == len(polys):
                rgb_f = _coerce_rgb_array(np.array(rgb_list, dtype=np.int64))
                if len(rgb_f) != len(faces):
                    rgb_f = None
            if a_list and len(a_list) == len(polys):
                alpha = _coerce_rgb_array(np.array(a_list, dtype=np.int64))
                if len(alpha) != len(faces):
                    alpha = None

    return dict(xyz=xyz, nrm=nrm, rgb_v=rgb_v, faces=faces, rgb_f=rgb_f, alpha=alpha)


def load_ply(path: str) -> Dict[str, Any]:
    """
    Unified loader:
      1) Try plyfile (ASCII/binary)
      2) If plyfile fails AND format is ASCII -> fallback simple ASCII parser
      3) If format is binary and plyfile fails -> raise (cannot safely fallback)
    """
    fmt = _peek_ply_format(path)

    try:
        return _load_with_plyfile(path)
    except Exception as e:
        if fmt == "ascii":
            return _load_ascii_simple(path)

        raise RuntimeError(
            f"Failed to read PLY with plyfile (format={fmt}). "
            f"Cannot fallback unless ASCII. Path={path}. Original error: {repr(e)}"
        ) from e
