# scripts/merge_colored_ply.py
from pathlib import Path
import argparse
import trimesh
import numpy as np


def load_colored_ply(path: Path) -> trimesh.Trimesh:
    mesh = trimesh.load(path, force="mesh", process=False)

    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(mesh.dump()))

    if not isinstance(mesh, trimesh.Trimesh):
        raise RuntimeError(f"Invalid mesh: {path}")

    if mesh.visual.vertex_colors is None or len(mesh.visual.vertex_colors) == 0:
        raise RuntimeError(f"No vertex colors in {path}")

    return mesh


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-root", required=True, type=Path,
                    help="folder containing colored ply files")
    ap.add_argument("--out", required=True, type=Path,
                    help="output merged ply")
    args = ap.parse_args()

    meshes = []
    for ply in sorted(args.in_root.rglob("*.ply")):
        print(f"[LOAD] {ply}")
        m = load_colored_ply(ply)
        meshes.append(m)

    if not meshes:
        raise RuntimeError("No PLY files found")

    merged = trimesh.util.concatenate(meshes)

    # --- force vertex-color only (strip normals / others) ---
    colors = merged.visual.vertex_colors[:, :3].astype(np.uint8)
    merged.visual.vertex_colors = colors

    args.out.parent.mkdir(parents=True, exist_ok=True)

    # export as ASCII PLY (เหมือนไฟล์อ้างอิง)
    merged.export(
        args.out,
        file_type="ply",
        encoding="ascii"
    )

    print(f"\n[OK] merged ply saved -> {args.out}")
    print(f"vertices={len(merged.vertices)}, faces={len(merged.faces)}")


if __name__ == "__main__":
    main()
