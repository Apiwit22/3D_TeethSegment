#---- ใส่สีให้ฟันแต่ละซี่และเหงือก RGB เท่านั้น "เฉพาะ vertex" และ merge รวมกันเป็นกรามเดียว  ----#
from __future__ import annotations
from pathlib import Path
import re
import numpy as np
from plyfile import PlyData, PlyElement

# ============================================================
# CONFIG (EDIT)
# ============================================================
IN_ROOT  = Path(r"D:\Project_Gujabaa\resampled")
OUT_ROOT = Path(r"D:\Project_Gujabaa\colored_merged_vertex")

SAVE_COLORED_PARTS = True
WRITE_BINARY = False

UPPER_PARTS_DIRNAME = "upper_parts"
LOWER_PARTS_DIRNAME = "lower_parts"

# ============================================================
# COLOR MAP (RGB 0-255)  >>> RGB ONLY (NO ALPHA)
# ============================================================
FDI_RGB = {
    # Q1 upper right 11-18
    11:(255,0,0), 12:(255,128,0), 13:(255,200,0), 14:(255,255,0),
    15:(170,255,0), 16:(0,255,0), 17:(0,255,170), 18:(0,255,255),
    # Q2 upper left 21-28
    21:(255,60,60), 22:(255,150,60), 23:(255,200,60), 24:(240,240,0),
    25:(170,255,60), 26:(0,200,0), 27:(0,200,200), 28:(60,240,240),
    # Q3 lower left 31-38
    31:(255,100,140), 32:(255,150,170), 33:(255,190,170), 34:(255,230,150),
    35:(180,255,150), 36:(120,255,120), 37:(120,255,200), 38:(150,255,255),
    # Q4 lower right 41-48
    41:(255,120,160), 42:(255,170,170), 43:(255,210,170), 44:(255,240,170),
    45:(200,255,170), 46:(150,255,150), 47:(150,255,220), 48:(170,240,255),
}
GINGIVA_RGB = (255, 180, 200)

# ============================================================
# Filename parsing
# ============================================================
_RE_INTS = re.compile(r"(\d{2,})")

def parse_fdi_from_name(stem_lower: str) -> int | None:
    matches = _RE_INTS.findall(stem_lower)
    for m in matches:
        try:
            n = int(m)
            if 11 <= n <= 48 and n in FDI_RGB:
                return n
        except:
            pass
    return None

def is_gingiva_name(stem_lower: str) -> bool:
    return any(k in stem_lower for k in ("jaw", "gingiva", "gum", "lowerjaw", "upperjaw", "gingival"))

def infer_color_from_file(p: Path) -> tuple[int,int,int]:
    s = p.stem.lower()
    if is_gingiva_name(s):
        return GINGIVA_RGB
    fdi = parse_fdi_from_name(s)
    if fdi is not None:
        return FDI_RGB[fdi]
    return GINGIVA_RGB

def infer_arch_from_path_or_name(p: Path) -> str | None:
    parts_lower = [x.lower() for x in p.parts]
    if any(UPPER_PARTS_DIRNAME.lower() == x for x in parts_lower) or "upper" in p.as_posix().lower():
        return "upper"
    if any(LOWER_PARTS_DIRNAME.lower() == x for x in parts_lower) or "lower" in p.as_posix().lower():
        return "lower"
    s = p.stem.lower()
    fdi = parse_fdi_from_name(s)
    if fdi is not None:
        if 11 <= fdi <= 28: return "upper"
        if 31 <= fdi <= 48: return "lower"
    return None

# ============================================================
# Build standard vertex arrays (FORCE vertex RGB ONLY)
# Face keeps ONLY vertex_indices (no color fields)
# ============================================================
VERT_DTYPE = np.dtype([
    ("x", "f4"), ("y", "f4"), ("z", "f4"),
    ("nx", "f4"), ("ny", "f4"), ("nz", "f4"),
    ("red", "u1"), ("green", "u1"), ("blue", "u1"),
])

FACE_ONLY_IDX_DTYPE = np.dtype([
    ("vertex_indices", "i4", (3,)),
])

def get_vertex_xyz_normals(ply: PlyData) -> tuple[np.ndarray, np.ndarray]:
    v = ply["vertex"].data
    names = v.dtype.names or ()
    if not all(k in names for k in ("x","y","z")):
        raise ValueError("vertex missing xyz")

    xyz = np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float32)

    if all(k in names for k in ("nx","ny","nz")):
        nrm = np.stack([v["nx"], v["ny"], v["nz"]], axis=1).astype(np.float32)
    else:
        nrm = np.zeros_like(xyz, dtype=np.float32)
    return xyz, nrm

def get_face_indices(ply: PlyData) -> np.ndarray | None:
    if "face" not in ply:
        return None
    f = ply["face"].data
    fnames = f.dtype.names or ()
    idx_field = None
    for cand in ("vertex_indices", "vertex_index", "indices"):
        if cand in fnames:
            idx_field = cand
            break
    if idx_field is None:
        raise ValueError("face element exists but no vertex_indices field")

    idx = f[idx_field]
    tris = []
    if idx.dtype == object:
        for poly in idx:
            poly = np.asarray(poly, dtype=np.int64).tolist()
            if len(poly) < 3:
                continue
            if len(poly) == 3:
                tris.append(poly)
            else:
                for i in range(1, len(poly)-1):
                    tris.append([poly[0], poly[i], poly[i+1]])
    else:
        idx = np.asarray(idx)
        if idx.ndim == 2 and idx.shape[1] >= 3:
            tris = idx[:, :3].astype(np.int64).tolist()
        else:
            raise ValueError("Unsupported face indices shape")

    if not tris:
        return None
    return np.asarray(tris, dtype=np.int64)

def build_colored_elements(in_path: Path) -> tuple[np.ndarray, np.ndarray | None]:
    ply = PlyData.read(str(in_path))
    xyz, nrm = get_vertex_xyz_normals(ply)
    r, g, b = infer_color_from_file(in_path)

    # vertex: set color
    vout = np.empty((xyz.shape[0],), dtype=VERT_DTYPE)
    vout["x"], vout["y"], vout["z"] = xyz[:,0], xyz[:,1], xyz[:,2]
    vout["nx"], vout["ny"], vout["nz"] = nrm[:,0], nrm[:,1], nrm[:,2]
    vout["red"], vout["green"], vout["blue"] = r, g, b

    # face: keep only indices (no color)
    tris = get_face_indices(ply)
    if tris is None:
        return vout, None

    fout = np.empty((tris.shape[0],), dtype=FACE_ONLY_IDX_DTYPE)
    fout["vertex_indices"] = tris.astype(np.int32)
    return vout, fout

def write_ply(path: Path, v: np.ndarray, f: np.ndarray | None):
    elems = [PlyElement.describe(v, "vertex")]
    if f is not None:
        elems.append(PlyElement.describe(f, "face"))
    out = PlyData(elems, text=(not WRITE_BINARY))
    path.parent.mkdir(parents=True, exist_ok=True)
    out.write(str(path))

# ============================================================
# Merge (concat only, no weld)
# ============================================================
def merge_parts(parts: list[tuple[np.ndarray, np.ndarray | None]]) -> tuple[np.ndarray, np.ndarray | None]:
    v_all = []
    f_all = []
    v_offset = 0
    any_face = False

    for v, f in parts:
        v_all.append(v)
        if f is not None:
            any_face = True
            f2 = f.copy()
            f2["vertex_indices"] = (f2["vertex_indices"].astype(np.int64) + v_offset).astype(np.int32)
            f_all.append(f2)
        v_offset += v.shape[0]

    V = np.concatenate(v_all, axis=0) if v_all else np.empty((0,), dtype=VERT_DTYPE)
    if any_face and f_all:
        F = np.concatenate(f_all, axis=0)
    else:
        F = None
    return V, F

# ============================================================
# Walk cases
# ============================================================
def list_case_dirs(root: Path) -> list[Path]:
    if (root / UPPER_PARTS_DIRNAME).exists() or (root / LOWER_PARTS_DIRNAME).exists():
        return [root]
    return [p for p in sorted(root.iterdir()) if p.is_dir()]

def collect_arch_files(case_dir: Path) -> dict[str, list[Path]]:
    files = list(case_dir.rglob("*.ply"))
    upper, lower = [], []
    for p in files:
        arch = infer_arch_from_path_or_name(p)
        if arch == "upper":
            upper.append(p)
        elif arch == "lower":
            lower.append(p)
    return {"upper": sorted(upper), "lower": sorted(lower)}

def main():
    if not IN_ROOT.exists():
        raise FileNotFoundError(f"IN_ROOT not found: {IN_ROOT}")

    case_dirs = list_case_dirs(IN_ROOT)
    if not case_dirs:
        print(f"[WARN] no case dirs under {IN_ROOT}")
        return

    for case_dir in case_dirs:
        case_id = case_dir.name
        groups = collect_arch_files(case_dir)

        for arch in ("upper", "lower"):
            arch_files = groups[arch]
            if not arch_files:
                continue

            parts_colored = []
            for p in arch_files:
                try:
                    v, f = build_colored_elements(p)
                    parts_colored.append((v, f))

                    if SAVE_COLORED_PARTS:
                        rel = p.relative_to(case_dir)
                        out_part = OUT_ROOT / case_id / f"{arch}_colored_parts" / rel.name
                        write_ply(out_part, v, f)

                except Exception as e:
                    print(f"[FAIL] {case_id} {arch} {p.name}: {e}")

            if not parts_colored:
                continue

            V, F = merge_parts(parts_colored)

            suffix = "_U" if arch == "upper" else "_L"
            out_merged = OUT_ROOT / case_id / f"{case_id}{suffix}.ply"
            write_ply(out_merged, V, F)

            nf = 0 if F is None else len(F)
            print(f"[OK] MERGED {case_id} {arch}: parts={len(parts_colored)}  V={len(V)}  F={nf}  -> {out_merged}")

    print(f"[DONE] colored+merged saved to: {OUT_ROOT}")

if __name__ == "__main__":
    main()
