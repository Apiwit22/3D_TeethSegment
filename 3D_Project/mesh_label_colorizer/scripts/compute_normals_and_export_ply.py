# scripts/compute_normals_and_export_ply.py
from pathlib import Path
import json
import numpy as np
import trimesh


# =========================
# LOAD COLOR CONFIG
# =========================
def load_color_config(cfg_path: Path):
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))

    gingiva_color = tuple(cfg["gingiva"])          # e.g. [255,180,200]
    fdi_colors = {int(k): tuple(v) for k, v in cfg["fdi"].items()}
    bg_label = int(cfg.get("background_label", 0))

    return bg_label, gingiva_color, fdi_colors


# =========================
# LOAD DATA
# =========================
def load_mesh(obj_path: Path) -> trimesh.Trimesh:
    mesh = trimesh.load(obj_path, force="mesh", process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(mesh.dump()))
    return mesh


def load_labels(json_path: Path):
    data = json.loads(json_path.read_text(encoding="utf-8"))
    return np.asarray(data["labels"], dtype=np.int32)


# =========================
# COLORIZE (RGB ONLY)
# =========================
def apply_vertex_colors(mesh, labels, bg_label, gingiva_color, fdi_colors):
    colors = np.zeros((len(mesh.vertices), 3), dtype=np.uint8)

    for i, lb in enumerate(labels):
        if lb == bg_label:
            colors[i] = gingiva_color
        else:
            colors[i] = fdi_colors.get(lb, (200, 200, 200))

    mesh.visual.vertex_colors = colors


# =========================
# COMPUTE NORMALS
# =========================
def compute_vertex_normals(mesh: trimesh.Trimesh):
    # คำนวณ normal จาก geometry หลัง resample
    mesh.rezero()
    mesh.fix_normals()   # ต้องมี networkx
    return mesh.vertex_normals


# =========================
# EXPORT PLY (STRICT FORMAT, NO ALPHA)
# =========================
def export_ply_with_normals(mesh: trimesh.Trimesh, out_path: Path):
    V = mesh.vertices
    N = mesh.vertex_normals
    C = mesh.visual.vertex_colors[:, :3]  # บังคับ RGB
    F = mesh.faces

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(V)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property float nx\n")
        f.write("property float ny\n")
        f.write("property float nz\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write(f"element face {len(F)}\n")
        f.write("property list uchar int vertex_indices\n")
        f.write("end_header\n")

        # write vertices
        for v, n, c in zip(V, N, C):
            f.write(
                f"{v[0]} {v[1]} {v[2]} "
                f"{n[0]} {n[1]} {n[2]} "
                f"{int(c[0])} {int(c[1])} {int(c[2])}\n"
            )

        # write faces
        for face in F:
            f.write(f"3 {face[0]} {face[1]} {face[2]}\n")


# =========================
# MAIN (single case)
# =========================
def run_single(obj_path: Path, json_path: Path, out_path: Path, color_cfg: Path):
    mesh = load_mesh(obj_path)
    labels = load_labels(json_path)

    if len(labels) != len(mesh.vertices):
        raise RuntimeError(
            f"Label count ({len(labels)}) != vertex count ({len(mesh.vertices)})"
        )

    bg_label, gingiva_color, fdi_colors = load_color_config(color_cfg)

    apply_vertex_colors(mesh, labels, bg_label, gingiva_color, fdi_colors)
    compute_vertex_normals(mesh)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    export_ply_with_normals(mesh, out_path)

    print(f"[OK] exported {out_path}")
