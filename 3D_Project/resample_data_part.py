from __future__ import annotations

from pathlib import Path
import numpy as np
import open3d as o3d

# ============================================================
# Config
# ============================================================
TARGET_F = 16000

IN_DIR = Path(r"D:\Project_Gujabaa\3D_Project\data_part")                 # <-- แก้ path
OUT_DIR = Path(r"D:\Project_Gujabaa\3D_Project\data_part_resampled")      # <-- แก้ path
OUT_DIR.mkdir(parents=True, exist_ok=True)

EXTS = {".obj", ".ply", ".stl"}
OUT_EXT = ".ply"  # output เป็น PLY ASCII 1.0 เสมอ


# ============================================================
# Helpers
# ============================================================
def fcount(mesh: o3d.geometry.TriangleMesh) -> int:
    return int(len(mesh.triangles))


def clean_mesh(mesh: o3d.geometry.TriangleMesh) -> o3d.geometry.TriangleMesh:
    if mesh.is_empty():
        return mesh

    # NOTE: keep clean "moderate" to avoid dropping triangles after we hit target
    try:
        mesh.remove_degenerate_triangles()
    except Exception:
        pass
    try:
        mesh.remove_duplicated_triangles()
    except Exception:
        pass
    try:
        mesh.remove_duplicated_vertices()
    except Exception:
        pass
    try:
        mesh.remove_unreferenced_vertices()
    except Exception:
        pass
    # ไม่เรียก remove_non_manifold_edges ตรงนี้เพื่อกันหาย 1 triangle หลัง simplify ในบางเคส
    try:
        mesh.compute_vertex_normals()
    except Exception:
        pass

    return mesh


def ensure_at_least_target(mesh: o3d.geometry.TriangleMesh, target: int) -> o3d.geometry.TriangleMesh:
    if mesh.is_empty():
        return mesh
    safety = 0
    while fcount(mesh) < target and safety < 6:
        mesh = mesh.subdivide_midpoint(number_of_iterations=1)
        mesh = clean_mesh(mesh)
        safety += 1
    return mesh


def simplify_to_target(mesh: o3d.geometry.TriangleMesh, target: int) -> o3d.geometry.TriangleMesh:
    mesh_s = mesh.simplify_quadric_decimation(target_number_of_triangles=target)
    mesh_s = clean_mesh(mesh_s)
    return mesh_s


def add_one_tiny_triangle(mesh: o3d.geometry.TriangleMesh, eps: float = 1e-8) -> o3d.geometry.TriangleMesh:
    """
    เติม 1 triangle แบบ tiny เพื่อให้ face count +1 (ใช้เมื่อขาดอยู่ 1 triangle เช่น 15999)
    ผลต่อ geometry แทบเป็นศูนย์
    """
    V = np.asarray(mesh.vertices)
    F = np.asarray(mesh.triangles)

    if len(F) == 0 or len(V) < 3:
        raise RuntimeError("mesh too small to patch")

    # เลือก triangle แรก
    t0 = F[0]
    v0, v1, v2 = V[t0[0]], V[t0[1]], V[t0[2]]

    # สร้าง vertex ใหม่ ใกล้ v2 มาก ๆ (offset ตามแนว (v2-v0))
    d = v2 - v0
    n = np.linalg.norm(d)
    if n < 1e-12:
        d = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        n = 1.0
    v_new = v2 + (d / n) * eps

    V2 = np.vstack([V, v_new])
    idx_new = len(V2) - 1

    # เพิ่ม triangle ใหม่ (v0, v1, v_new)
    F2 = np.vstack([F, np.array([t0[0], t0[1], idx_new], dtype=np.int32)])

    mesh2 = o3d.geometry.TriangleMesh()
    mesh2.vertices = o3d.utility.Vector3dVector(V2)
    mesh2.triangles = o3d.utility.Vector3iVector(F2)
    mesh2 = clean_mesh(mesh2)
    return mesh2


def force_exact_16000(mesh: o3d.geometry.TriangleMesh, target: int) -> o3d.geometry.TriangleMesh:
    mesh = clean_mesh(mesh)

    if fcount(mesh) < target:
        mesh = ensure_at_least_target(mesh, target)

    mesh_s = simplify_to_target(mesh, target)
    f1 = fcount(mesh_s)

    if f1 == target:
        return mesh_s

    # ถ้าได้ 15999 ให้เติม 1 triangle
    if f1 == target - 1:
        mesh_s = add_one_tiny_triangle(mesh_s)
        if fcount(mesh_s) != target:
            # เผื่อ clean ทำให้หายอีก ลองเติมอีกรอบ (ควรไม่เกิดบ่อย)
            mesh_s = add_one_tiny_triangle(mesh_s, eps=1e-7)
        return mesh_s

    # ถ้าคลาดมากกว่านั้น ให้ subdivide แล้ว simplify ซ้ำ
    if f1 < target:
        tmp = mesh_s.subdivide_midpoint(number_of_iterations=1)
        tmp = clean_mesh(tmp)
        mesh_s = simplify_to_target(tmp, target)

        if fcount(mesh_s) == target - 1:
            mesh_s = add_one_tiny_triangle(mesh_s)

    # ถ้ายังไม่ตรง ลอง simplify ซ้ำครั้งสุดท้าย
    if fcount(mesh_s) != target:
        mesh_s = simplify_to_target(mesh_s, target)
        if fcount(mesh_s) == target - 1:
            mesh_s = add_one_tiny_triangle(mesh_s)

    return mesh_s


def process_file(in_path: Path, out_path: Path) -> tuple[int, int]:
    mesh = o3d.io.read_triangle_mesh(str(in_path))
    if mesh.is_empty():
        raise RuntimeError("empty mesh")

    f0 = fcount(mesh)
    mesh_out = force_exact_16000(mesh, TARGET_F)
    f1 = fcount(mesh_out)

    ok = o3d.io.write_triangle_mesh(str(out_path), mesh_out, write_ascii=True)
    if not ok:
        raise RuntimeError("failed to write mesh")
    return f0, f1


# ============================================================
# Main
# ============================================================
def main():
    files = [p for p in IN_DIR.rglob("*") if p.suffix.lower() in EXTS]
    print(f"[IN ] {IN_DIR}")
    print(f"[OUT] {OUT_DIR}")
    print(f"[N  ] {len(files)} files")

    bad = 0
    for p in files:
        out_path = OUT_DIR / (p.stem + OUT_EXT)
        try:
            f0, f1 = process_file(p, out_path)
            tag = "OK" if f1 == TARGET_F else "NG"
            if tag == "NG":
                bad += 1
            print(f"[{tag}] {p.name:35s} faces {f0:6d} -> {f1:6d}")
        except Exception as e:
            bad += 1
            print(f"[ER] {p.name}: {e}")

    print(f"[DONE] not_exact_or_error = {bad} / {len(files)}")


if __name__ == "__main__":
    main()
