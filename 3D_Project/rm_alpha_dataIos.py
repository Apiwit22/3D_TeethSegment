#---- ลบค่า 'alpha' ออกจาก face (และ vertex ถ้ามี) ของ data_IoSSeg_alpha ----#
from __future__ import annotations
from pathlib import Path
import numpy as np

try:
    from plyfile import PlyData, PlyElement
except Exception as e:
    raise ImportError(
        "This script requires plyfile.\n"
        "Install with: pip install plyfile\n"
        f"Original error: {e}"
    )

# ============================================================
# CONFIG (EDIT THESE 2 PATHS)
# ============================================================
# โฟลเดอร์ input (ชุดที่ 2) ที่มีไฟล์ .ply (ค้นหาแบบ recursive)
IN_ROOT  = Path(r"D:\Project_Gujabaa\data_IoSSeg_alpha")

# โฟลเดอร์ output ที่จะเก็บไฟล์ใหม่ (โครงสร้างโฟลเดอร์จะเหมือนเดิม)
OUT_ROOT = Path(r"D:\Project_Gujabaa\data_IoSSeg_alpha_rgb")

# หาไฟล์แบบ recursive (โดยปกติ *.ply ก็พอ)
PATTERN = "*.ply"

# บังคับให้ output เป็น ASCII PLY ไหม?
# - True  => output เป็น ASCII
# - False => output เป็น Binary
# - None  => คงตามไฟล์ต้นฉบับ (แนะนำ: None)
FORCE_ASCII = True
# ============================================================


def drop_field(struct_arr: np.ndarray, field_name: str) -> np.ndarray:
    """Drop one field from a structured array, preserving field order."""
    names = list(struct_arr.dtype.names or [])
    if field_name not in names:
        return struct_arr

    keep = [n for n in names if n != field_name]
    new_dtype = [(n, struct_arr.dtype.fields[n][0]) for n in keep]
    out = np.empty(struct_arr.shape, dtype=new_dtype)
    for n in keep:
        out[n] = struct_arr[n]
    return out


def convert_one(in_path: Path, out_path: Path) -> tuple[bool, str]:
    """
    Remove 'alpha' from face (and vertex if present).
    Returns (changed, message).
    """
    ply = PlyData.read(str(in_path))

    elements_out = []
    changed = False

    for el in ply.elements:
        data = el.data
        if el.name in ("face", "vertex"):
            if data.dtype.names and "alpha" in data.dtype.names:
                data2 = drop_field(data, "alpha")
                changed = True
                elements_out.append(PlyElement.describe(data2, el.name))
            else:
                elements_out.append(PlyElement.describe(data, el.name))
        else:
            elements_out.append(PlyElement.describe(data, el.name))

    # output ascii/binary
    if FORCE_ASCII is None:
        text_mode = ply.text
    else:
        text_mode = bool(FORCE_ASCII)

    out_ply = PlyData(
        elements_out,
        text=text_mode,
        comments=list(getattr(ply, "comments", [])),
        obj_info=list(getattr(ply, "obj_info", [])),
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_ply.write(str(out_path))

    msg = "CHANGED (alpha removed)" if changed else "SAME (no alpha)"
    return changed, msg


def main():
    if not IN_ROOT.exists():
        raise FileNotFoundError(f"IN_ROOT not found: {IN_ROOT}")

    files = sorted(IN_ROOT.rglob(PATTERN))
    if not files:
        print(f"[WARN] no .ply found under: {IN_ROOT}")
        return

    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    ok = 0
    fail = 0
    changed_n = 0

    print("============================================================")
    print("STRIP FACE ALPHA (and vertex alpha if present)")
    print(f"IN   : {IN_ROOT}")
    print(f"OUT  : {OUT_ROOT}")
    print(f"PAT  : {PATTERN}")
    mode = "KEEP_ORIGINAL" if FORCE_ASCII is None else ("ASCII" if FORCE_ASCII else "BINARY")
    print(f"MODE : {mode}")
    print("============================================================")

    for p in files:
        rel = p.relative_to(IN_ROOT)
        out_path = OUT_ROOT / rel
        try:
            changed, msg = convert_one(p, out_path)
            ok += 1
            if changed:
                changed_n += 1
            print(f"[OK] {msg:22s}  {rel}")
        except Exception as e:
            fail += 1
            print(f"[FAIL] {rel}: {e}")

    print("============================================================")
    print(f"[DONE] OK={ok}  FAIL={fail}  CHANGED={changed_n}")
    print("============================================================")


if __name__ == "__main__":
    main()
