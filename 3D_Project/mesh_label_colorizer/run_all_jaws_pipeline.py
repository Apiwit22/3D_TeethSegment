# run_all_jaws_pipeline.py
import subprocess
import sys
from pathlib import Path

PY = sys.executable

def run(cmd):
    print("\n>>>", " ".join(map(str, cmd)))
    subprocess.check_call(list(map(str, cmd)))

# =========================
# STEP 1: REMESH
# =========================
for jaw in ["lower", "upper"]:
    run([PY, "scripts/remesh_to_16k.py", "--jaw", jaw])

# =========================
# STEP 2: LABEL REMAP
# =========================
for jaw in ["lower", "upper"]:
    run([PY, "scripts/map_labels_to_remesh.py", "--jaw", jaw])

# =========================
# STEP 3: COLOR + NORMAL + EXPORT
# =========================
LABEL_ROOT = Path("data/labels_remap")
MESH_ROOT  = Path("data/remeshed")
OUT_ROOT   = Path("data/colored_ready")
COLOR_CFG  = Path("configs/fdi_colors_rgb.json")

from scripts.compute_normals_and_export_ply import run_single

# NEW layout: labels_remap/<subset>/<jaw>/<case>/<stem>_16k.json
for jp in LABEL_ROOT.rglob("*.json"):
    rel = jp.relative_to(LABEL_ROOT)
    if len(rel.parts) < 4:
        continue
    subset, jaw, case = rel.parts[0], rel.parts[1], rel.parts[2]

    stem = jp.stem.replace("_16k", "")
    obj_path = MESH_ROOT / subset / jaw / case / f"{stem}_16k.obj"
    if not obj_path.exists():
        continue

    out_path = OUT_ROOT / subset / jaw / case / f"{stem}_16k_colored.ply"
    run_single(obj_path, jp, out_path, COLOR_CFG)

print("\nALL DONE: PLY with normals + colors generated")
