from __future__ import annotations
from pathlib import Path
import numpy as np
from plyfile import PlyData, PlyElement

# ---------------------------
# Config
# ---------------------------
WRITE_BINARY = False  # False=ASCII, True=binary_little_endian

def _pick_vertex_dtype(v_dtype: np.dtype):
    """Build output dtype keeping xyz + normals if present; normals missing -> zeros."""
    names = v_dtype.names or ()
    for k in ("x", "y", "z"):
        if k not in names:
            raise ValueError("vertex missing x/y/z")

    has_n = all(k in names for k in ("nx", "ny", "nz"))
    out = [
        ("x", "f4"), ("y", "f4"), ("z", "f4"),
        ("nx", "f4"), ("ny", "f4"), ("nz", "f4"),
    ]
    return np.dtype(out), has_n

def strip_vertex_colors_keep_face_colors(in_ply: str | Path, out_ply: str | Path) -> None:
    in_ply = Path(in_ply)
    out_ply = Path(out_ply)

    ply = PlyData.read(str(in_ply))
    if "vertex" not in ply:
        raise ValueError("No 'vertex' element found")

    v_in = ply["vertex"].data
    v_out_dtype, has_normals = _pick_vertex_dtype(v_in.dtype)

    v_out = np.empty((len(v_in),), dtype=v_out_dtype)
    v_out["x"] = v_in["x"].astype(np.float32)
    v_out["y"] = v_in["y"].astype(np.float32)
    v_out["z"] = v_in["z"].astype(np.float32)

    if has_normals:
        v_out["nx"] = v_in["nx"].astype(np.float32)
        v_out["ny"] = v_in["ny"].astype(np.float32)
        v_out["nz"] = v_in["nz"].astype(np.float32)
    else:
        v_out["nx"] = 0.0
        v_out["ny"] = 0.0
        v_out["nz"] = 0.0

    elems = [PlyElement.describe(v_out, "vertex")]

    # Keep face element exactly as-is (including face colors)
    if "face" in ply:
        f_in = ply["face"].data
        elems.append(PlyElement.describe(f_in, "face"))

    out = PlyData(elems, text=(not WRITE_BINARY))
    out_ply.parent.mkdir(parents=True, exist_ok=True)
    out.write(str(out_ply))

def strip_folder(
    in_dir: str | Path,
    out_dir: str | Path,
    pattern: str = "*.ply",
    recursive: bool = True,
    verbose: bool = True,
):
    in_dir = Path(in_dir)
    out_dir = Path(out_dir)

    if not in_dir.exists():
        raise FileNotFoundError(f"Input folder not found: {in_dir}")

    files = list(in_dir.rglob(pattern)) if recursive else list(in_dir.glob(pattern))
    if not files:
        print(f"[WARN] no files matched {pattern} under {in_dir}")
        return

    ok = 0
    fail = 0

    for p in files:
        rel = p.relative_to(in_dir)
        out_p = out_dir / rel
        try:
            strip_vertex_colors_keep_face_colors(p, out_p)
            ok += 1
            if verbose:
                print(f"[OK] {rel}")
        except Exception as e:
            fail += 1
            print(f"[FAIL] {rel}: {e}")

    print(f"[DONE] processed={ok} failed={fail}")
    print(f"       in : {in_dir}")
    print(f"       out: {out_dir}")

if __name__ == "__main__":
    strip_folder(
        in_dir=r"D:\Project_Gujabaa\data_IoSSeg_alpha",
        out_dir=r"D:\Project_Gujabaa\public_data_recolor",
        pattern="*.ply",
        recursive=True,
        verbose=True,
    )
