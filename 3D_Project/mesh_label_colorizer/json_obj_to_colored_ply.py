# json_obj_to_colored_ply.py
from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, Tuple
import numpy as np
import trimesh


# =========================
# LOAD COLOR CONFIG
# =========================
def load_color_config(path: Path | None):
    DEFAULT_GINGIVA = (255, 180, 200)
    DEFAULT_UNKNOWN = (200, 200, 200)

    if path is None:
        raise RuntimeError("Color config is required")

    cfg = json.loads(path.read_text(encoding="utf-8"))

    background_label = int(cfg.get("background_label", 0))
    gingiva = tuple(cfg.get("gingiva", DEFAULT_GINGIVA))
    unknown = tuple(cfg.get("unknown_label_color", DEFAULT_UNKNOWN))

    fdi_map: Dict[int, Tuple[int, int, int]] = {}
    for k, v in cfg["fdi"].items():
        fdi_map[int(k)] = tuple(int(x) for x in v)

    return background_label, gingiva, unknown, fdi_map


# =========================
# LOAD DATA
# =========================
def load_mesh(obj_path: Path) -> trimesh.Trimesh:
    mesh = trimesh.load(obj_path, force="mesh", process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(mesh.dump()))
    return mesh


def read_labels(json_path: Path):
    data = json.loads(json_path.read_text(encoding="utf-8"))
    if "labels" not in data:
        raise KeyError(f"No 'labels' in {json_path}")
    return np.asarray(data["labels"], dtype=np.int32)


# =========================
# COLORIZE
# =========================
def colorize(mesh, labels, bg_label, gingiva, unknown, fdi_map):
    if len(labels) != len(mesh.vertices):
        raise RuntimeError("Vertex count mismatch")

    colors = np.zeros((len(labels), 4), dtype=np.uint8)
    colors[:, :3] = unknown
    colors[:, 3] = 255

    colors[labels == bg_label, :3] = gingiva

    for fdi, rgb in fdi_map.items():
        colors[labels == fdi, :3] = rgb

    return colors


# =========================
# EXPORT PLY (STRICT FORMAT)
# =========================
def export_ply_vertex_color(mesh, out_path: Path):
    V = mesh.vertices
    F = mesh.faces
    C = mesh.visual.vertex_colors[:, :4]  # RGBA

    with open(out_path, "w") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(V)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("property uchar alpha\n")
        f.write(f"element face {len(F)}\n")
        f.write("property list uchar int vertex_indices\n")
        f.write("end_header\n")

        for v, c in zip(V, C):
            f.write(
                f"{v[0]} {v[1]} {v[2]} "
                f"{int(c[0])} {int(c[1])} {int(c[2])} {int(c[3])}\n"
            )

        for face in F:
            f.write(f"3 {face[0]} {face[1]} {face[2]}\n")


# =========================
# MAIN (single file)
# =========================
def run_single(obj_path, json_path, out_path, color_cfg):
    bg, gingiva, unknown, fdi_map = load_color_config(color_cfg)

    mesh = load_mesh(obj_path)
    labels = read_labels(json_path)

    colors = colorize(mesh, labels, bg, gingiva, unknown, fdi_map)
    mesh.visual.vertex_colors = colors

    out_path.parent.mkdir(parents=True, exist_ok=True)
    export_ply_vertex_color(mesh, out_path)

    print(f"[OK] Exported: {out_path}")
