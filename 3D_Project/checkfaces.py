# checkfaces.py
from __future__ import annotations

import sys
from pathlib import Path
import open3d as o3d

# =========================
# CONFIG
# =========================
COLORED_ROOT = Path(r"D:\Project_Gujabaa\3D_Project\mesh_label_colorizer\data\colored_ready")
TARGET_FACES = 16000
SUBSETS = ["data_part_1", "data_part_2", "data_part_3"]

# แสดงรายชื่อไฟล์ที่เจอ
SHOW_FOUND_FILES = True
PRINT_LIMIT = 50            # แสดงแค่ 50 ไฟล์แรกต่อ subset (กันยาวเกิน)
PRINT_RELATIVE = True       # แสดง path แบบ relative ต่อ COLORED_ROOT


def count_faces(ply_path: Path) -> int:
    mesh = o3d.io.read_triangle_mesh(str(ply_path))
    if mesh.is_empty():
        raise RuntimeError("empty or unreadable mesh")
    return int(len(mesh.triangles))


def scan_subset(subset_dir: Path):
    files = sorted([p for p in subset_dir.rglob("*.ply") if p.is_file()] +
                   [p for p in subset_dir.rglob("*.PLY") if p.is_file()])
    files = sorted({p.resolve() for p in files})  # กันซ้ำบน Windows

    print(f"\n[SCAN] {subset_dir}")
    print(f"[INFO] found ply = {len(files)}")

    if SHOW_FOUND_FILES:
        show = files[:PRINT_LIMIT]
        for p in show:
            if PRINT_RELATIVE:
                try:
                    rp = p.relative_to(COLORED_ROOT)
                    print(f"[FILE] {rp}")
                except Exception:
                    print(f"[FILE] {p}")
            else:
                print(f"[FILE] {p}")

        if len(files) > PRINT_LIMIT:
            print(f"[INFO] ... and {len(files) - PRINT_LIMIT} more files (limit={PRINT_LIMIT})")

    ok = ng = er = 0
    for p in files:
        try:
            n = count_faces(p)
            if n == TARGET_FACES:
                ok += 1
            else:
                ng += 1
                print(f"[NG] {p}  faces={n} (expected {TARGET_FACES})")
        except Exception as e:
            er += 1
            print(f"[ER] {p}  {e}")

    print(f"[DONE] ok={ok}  ng={ng}  err={er}  total={ok+ng+er}")


def main():
    if not COLORED_ROOT.exists():
        raise SystemExit(f"[ER] COLORED_ROOT not found: {COLORED_ROOT}")

    any_found = False
    for name in SUBSETS:
        subset_dir = COLORED_ROOT / name
        if not subset_dir.exists():
            print(f"[SKIP] missing: {subset_dir}")
            continue
        any_found = True
        scan_subset(subset_dir)

    if not any_found:
        print(f"[ER] none of SUBSETS exist under: {COLORED_ROOT}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
