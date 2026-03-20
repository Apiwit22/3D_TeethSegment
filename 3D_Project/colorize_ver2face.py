# vertex2face_to_sample_format.py
from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import open3d as o3d

# =========================
# CONFIG (EDIT HERE)
# =========================
IN_ROOT  = Path(r"D:\Project_Gujabaa\3D_Project\mesh_label_colorizer\data\colored_ready")  # vertex-colored PLY
OUT_ROOT = Path(r"D:\Project_Gujabaa\3D_Project\data_part_colored_face")  # output

SUBSETS = ["data_part_1", "data_part_2", "data_part_3"]

ALPHA = 255
METHOD = "mode"  # "mode" (คม) หรือ "mean" (เนียน)


def load_mesh(p: Path) -> o3d.geometry.TriangleMesh:
    m = o3d.io.read_triangle_mesh(str(p))
    if m.is_empty():
        raise RuntimeError("empty/unreadable mesh")
    if len(m.triangles) == 0:
        raise RuntimeError("mesh has no faces")
    if not m.has_vertex_colors():
        raise RuntimeError("mesh has no vertex colors")
    if not m.has_vertex_normals():
        m.compute_vertex_normals()
    return m


def face_colors_from_vertex(VC_u8: np.ndarray, F: np.ndarray, method: str) -> np.ndarray:
    c0 = VC_u8[F[:, 0]]
    c1 = VC_u8[F[:, 1]]
    c2 = VC_u8[F[:, 2]]

    if method == "mean":
        fc = (c0.astype(np.float32) + c1.astype(np.float32) + c2.astype(np.float32)) / 3.0
        return np.clip(np.rint(fc), 0, 255).astype(np.uint8)

    if method == "mode":
        eq01 = np.all(c0 == c1, axis=1)
        eq02 = np.all(c0 == c2, axis=1)
        eq12 = np.all(c1 == c2, axis=1)

        out = np.empty_like(c0)
        m0 = eq01 | eq02
        out[m0] = c0[m0]
        m1 = (~m0) & eq12
        out[m1] = c1[m1]
        m2 = (~m0) & (~m1)
        out[m2] = c0[m2]
        return out.astype(np.uint8)

    raise ValueError("METHOD must be 'mode' or 'mean'")


def write_ply_like_sample(out_path: Path, V: np.ndarray, VN: np.ndarray, F: np.ndarray, FC: np.ndarray, alpha: int = 255):
    """
    Match sample header:
      format ascii 1.0
      vertex: float x y z nx ny nz
      face: list uchar int vertex_indices + uchar r g b a
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    V = V.astype(np.float32, copy=False)
    VN = VN.astype(np.float32, copy=False)
    F = F.astype(np.int32, copy=False)  # int indices (not uint)
    FC = FC.astype(np.uint8, copy=False)
    a = int(alpha)

    with out_path.open("w", newline="\n") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(V)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property float nx\n")
        f.write("property float ny\n")
        f.write("property float nz\n")
        f.write(f"element face {len(F)}\n")
        f.write("property list uchar int vertex_indices\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("property uchar alpha\n")
        f.write("end_header\n")

        for (x, y, z), (nx, ny, nz) in zip(V, VN):
            f.write(f"{x:.6f} {y:.6f} {z:.6f} {nx:.6f} {ny:.6f} {nz:.6f}\n")

        for (i0, i1, i2), (r, g, b) in zip(F, FC):
            f.write(f"3 {int(i0)} {int(i1)} {int(i2)} {int(r)} {int(g)} {int(b)} {a}\n")


def process_one(src: Path, dst: Path):
    m = load_mesh(src)

    V = np.asarray(m.vertices, dtype=np.float64)
    VN = np.asarray(m.vertex_normals, dtype=np.float64)
    F = np.asarray(m.triangles, dtype=np.int64)

    VC = np.asarray(m.vertex_colors, dtype=np.float64)  # [0..1]
    VC_u8 = np.clip(np.rint(VC * 255.0), 0, 255).astype(np.uint8)

    FC = face_colors_from_vertex(VC_u8, F, METHOD)

    write_ply_like_sample(dst, V, VN, F, FC, alpha=ALPHA)


def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    total = ok = fail = 0
    for subset in SUBSETS:
        in_subset = IN_ROOT / subset
        if not in_subset.exists():
            print(f"[SKIP] missing subset: {in_subset}")
            continue

        files = sorted({p.resolve() for p in in_subset.rglob("*.ply") if p.is_file()})
        print(f"\n[SUBSET] {subset} files={len(files)}")

        for src in files:
            rel = src.relative_to(IN_ROOT)      # keep same structure
            dst = OUT_ROOT / rel                # same filename
            total += 1
            try:
                process_one(src, dst)
                ok += 1
            except Exception as e:
                fail += 1
                print(f"[ER] {rel}: {e}")

    print(f"\n[DONE] total={total} ok={ok} fail={fail}")
    print(f"[OUT] {OUT_ROOT}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
