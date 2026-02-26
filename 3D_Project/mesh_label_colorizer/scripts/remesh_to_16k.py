# scripts/remesh_to_16k.py
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import open3d as o3d

TARGET_F = 16000


def fcount(m: o3d.geometry.TriangleMesh) -> int:
    return int(len(m.triangles))


def clean(m: o3d.geometry.TriangleMesh) -> o3d.geometry.TriangleMesh:
    m.remove_degenerate_triangles()
    m.remove_duplicated_triangles()
    m.remove_duplicated_vertices()
    m.remove_unreferenced_vertices()
    m.compute_vertex_normals()
    return m


def add_one_tiny_triangle(mesh: o3d.geometry.TriangleMesh, eps: float = 1e-8) -> o3d.geometry.TriangleMesh:
    """
    Patch case where simplify returns 15999 faces: add a tiny triangle so faces + 1.
    """
    V = np.asarray(mesh.vertices, dtype=np.float64)
    F = np.asarray(mesh.triangles, dtype=np.int32)
    if len(F) == 0:
        raise RuntimeError("mesh has no faces")

    t0 = F[0]
    v0, v1, v2 = V[t0[0]], V[t0[1]], V[t0[2]]
    d = v2 - v0
    n = np.linalg.norm(d)
    if n < 1e-12:
        d = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        n = 1.0

    v_new = v2 + (d / n) * eps

    V2 = np.vstack([V, v_new])
    idx_new = V2.shape[0] - 1
    F2 = np.vstack([F, np.array([t0[0], t0[1], idx_new], dtype=np.int32)])

    m2 = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(V2),
        o3d.utility.Vector3iVector(F2),
    )
    return clean(m2)


def remesh_to_16k(in_obj: Path, target_faces: int) -> o3d.geometry.TriangleMesh:
    m = o3d.io.read_triangle_mesh(str(in_obj))
    if m.is_empty():
        raise RuntimeError("empty mesh")

    m = clean(m)

    # If mesh has fewer faces than target, upsample once
    if fcount(m) < target_faces:
        m = m.subdivide_midpoint(1)
        m = clean(m)

    # Primary decimation
    m2 = m.simplify_quadric_decimation(target_number_of_triangles=target_faces)
    m2 = clean(m2)

    # Retry if off by 1
    if fcount(m2) == target_faces - 1:
        tmp = m2.subdivide_midpoint(1)
        tmp = clean(tmp)
        m2 = tmp.simplify_quadric_decimation(target_number_of_triangles=target_faces)
        m2 = clean(m2)

    # Final patch if still off by 1
    if fcount(m2) == target_faces - 1:
        m2 = add_one_tiny_triangle(m2)

    if fcount(m2) != target_faces:
        raise RuntimeError(f"got {fcount(m2)} faces (expected {target_faces})")

    return m2


def iter_obj_files(case_dir: Path):
    """
    Windows is case-insensitive; avoid *.obj + *.OBJ duplicates.
    """
    return sorted([p for p in case_dir.iterdir() if p.is_file() and p.suffix.lower() == ".obj"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jaw", required=True, choices=["lower", "upper"])
    ap.add_argument("--target_faces", type=int, default=TARGET_F)
    args = ap.parse_args()

    raw_parent = Path("data/raw")
    out_parent = Path("data/remeshed")

    subsets = [p for p in sorted(raw_parent.iterdir()) if p.is_dir()]
    if not subsets:
        raise SystemExit(f"[ER] no subset under {raw_parent}")

    for subset in subsets:
        jaw_dir = subset / args.jaw
        if not jaw_dir.exists():
            print(f"[SKIP] no jaw folder: {jaw_dir}")
            continue

        # expect case folders under jaw_dir
        for case_dir in sorted([p for p in jaw_dir.iterdir() if p.is_dir()]):
            objs = iter_obj_files(case_dir)
            if not objs:
                continue

            for obj_path in objs:
                out_dir = out_parent / subset.name / args.jaw / case_dir.name
                out_dir.mkdir(parents=True, exist_ok=True)

                out_obj = out_dir / f"{obj_path.stem}_16k.obj"
                if out_obj.exists():
                    # Skip if already done
                    continue

                try:
                    m2 = remesh_to_16k(obj_path, args.target_faces)
                    # NOTE: OBJ cannot store triangle normals in Open3D; it's fine.
                    o3d.io.write_triangle_mesh(str(out_obj), m2, write_ascii=True)
                    print(f"[OK] {subset.name}/{args.jaw}/{case_dir.name}/{obj_path.name} -> {out_obj.name}")
                except Exception as e:
                    print(f"[ER] {subset.name}/{args.jaw}/{case_dir.name}/{obj_path.name}: {e}")


if __name__ == "__main__":
    main()
