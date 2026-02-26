from __future__ import annotations
from pathlib import Path
from typing import Dict, Tuple
import numpy as np
import trimesh

from plyfile import PlyData, PlyElement

# ✅ ใช้ของแอพเอง
from app.data.fdi_colors import (
    FDIColorMap,
    FDI_LIST_UPPER_16, FDI_LIST_LOWER_16,
    GINGIVA_LABEL,
)


def export_colored_ply_facecolors(out_path: Path, v: np.ndarray, f: np.ndarray, face_rgb: np.ndarray) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    v = np.asarray(v, dtype=np.float32)
    f = np.asarray(f, dtype=np.int64)
    rgb = np.asarray(face_rgb, dtype=np.uint8)
    if len(rgb) != len(f):
        raise ValueError("face_rgb length must match faces length")

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


def export_split_by_tooth(
    out_dir: Path,
    *,
    v: np.ndarray,
    f: np.ndarray,
    labels_face: np.ndarray,
    arch: str,
    num_classes: int = 17,
) -> Dict[str, Path]:
    """
    แยก export เป็นไฟล์ตามซี่ฟัน (FDI) + เหงือก

    Return:
      dict[name -> path]
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    v = np.asarray(v, dtype=np.float32)
    f = np.asarray(f, dtype=np.int64)
    labels = np.asarray(labels_face, dtype=np.int64).reshape(-1)

    if len(labels) != len(f):
        raise ValueError("labels_face length must match faces length")

    arch = (arch or "").lower()
    if arch not in ("upper", "lower"):
        arch = "lower"

    # mapping label->FDI list
    fdi_list = FDI_LIST_UPPER_16 if arch == "upper" else FDI_LIST_LOWER_16

    # ใช้ trimesh เพื่อ export เฉพาะ subset faces
    mesh = trimesh.Trimesh(vertices=v, faces=f, process=False)

    out_paths: Dict[str, Path] = {}

    # teeth 0..15
    for lab in range(16):
        mask = labels == lab
        if not np.any(mask):
            continue
        sub = mesh.submesh([np.where(mask)[0]], append=True, repair=False)
        fdi = fdi_list[lab]
        name = f"tooth_{fdi}"
        path = out_dir / f"{name}.ply"
        sub.export(path)
        out_paths[name] = path

    # gingiva
    if num_classes >= 17:
        mask = labels == GINGIVA_LABEL
        if np.any(mask):
            sub = mesh.submesh([np.where(mask)[0]], append=True, repair=False)
            name = "gingiva"
            path = out_dir / f"{name}.ply"
            sub.export(path)
            out_paths[name] = path

    return out_paths