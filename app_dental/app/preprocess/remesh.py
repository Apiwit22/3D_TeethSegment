from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class RemeshResult:
    vertices: np.ndarray
    faces: np.ndarray
    meta: dict


def remesh_to_faces(vertices: np.ndarray, faces: np.ndarray, target_faces: int = 16000) -> RemeshResult:
    v = np.asarray(vertices, dtype=np.float32)
    f = np.asarray(faces, dtype=np.int64)
    target_faces = int(target_faces)

    if target_faces <= 0:
        return RemeshResult(v, f, {"mode": "none", "reason": "target_faces<=0"})

    if f.ndim != 2 or f.shape[1] != 3:
        raise ValueError(f"faces must be (F,3), got {f.shape}")

    F0 = int(f.shape[0])
    if F0 == 0:
        raise ValueError("mesh has 0 faces")

    try:
        import pymeshlab  # type: ignore

        ms = pymeshlab.MeshSet()
        m = pymeshlab.Mesh(v, f.astype(np.int32, copy=False))
        ms.add_mesh(m, "m")

        if F0 < target_faces:
            it = 0
            while ms.current_mesh().face_number() < target_faces and it < 4:
                ms.apply_filter("meshing_surface_subdivision_loop", iterations=1)
                it += 1

        ms.apply_filter(
            "meshing_decimation_quadric_edge_collapse",
            targetfacenum=target_faces,
            preservenormal=True,
            preservetopology=True,
            qualitythr=1.0,
            autoclean=True,
        )

        mv = ms.current_mesh().vertex_matrix().astype(np.float32, copy=False)
        mf = ms.current_mesh().face_matrix().astype(np.int64, copy=False)

        meta = {"mode": "pymeshlab", "F0": F0, "F1": int(mf.shape[0]), "target": target_faces}
        return RemeshResult(mv, mf, meta)

    except Exception as e:
        meta = {"mode": "fallback", "F0": F0, "warn": str(e)}
        return RemeshResult(v, f, meta)