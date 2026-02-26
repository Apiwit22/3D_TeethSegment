from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np
import trimesh


@dataclass
class Mesh:
    vertices: np.ndarray
    faces: np.ndarray


def load_mesh(path: str | Path) -> Mesh:
    path = Path(path)
    m = trimesh.load(str(path), force="mesh", process=False)

    if isinstance(m, trimesh.Scene):
        m = trimesh.util.concatenate(tuple(m.geometry.values()))

    if not isinstance(m, trimesh.Trimesh):
        raise ValueError(f"Unsupported mesh type: {type(m)}")

    v = np.asarray(m.vertices, dtype=np.float32)
    f = np.asarray(m.faces, dtype=np.int64)

    if v.ndim != 2 or v.shape[1] != 3 or f.ndim != 2 or f.shape[1] != 3:
        raise ValueError("Invalid mesh format (expect vertices (V,3) and faces (F,3)).")

    return Mesh(vertices=v, faces=f)