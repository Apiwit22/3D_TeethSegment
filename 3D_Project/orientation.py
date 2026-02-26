#---- Robust orientation: match private data to public ref (with gingiva-direction constraint) ----#
from __future__ import annotations

from pathlib import Path
import numpy as np

try:
    from scipy.spatial import cKDTree
except Exception as e:
    raise ImportError(
        "This script requires SciPy for KDTree.\n"
        "Install with: pip install scipy\n"
        f"Original error: {e}"
    )

try:
    from plyfile import PlyData, PlyElement
except Exception as e:
    raise ImportError(
        "This script requires plyfile.\n"
        "Install with: pip install plyfile\n"
        f"Original error: {e}"
    )

# ============================================================
# CONFIG
# ============================================================
SET2_UPPER_REF = Path(r"D:\Project_Gujabaa\3D_Project\aligned_public_data_recolor\train\007_U.ply")
SET2_LOWER_REF = Path(r"D:\Project_Gujabaa\3D_Project\aligned_public_data_recolor\train\007_L.ply")

IN_ROOT  = Path(r"D:\Project_Gujabaa\3D_Project\data_part_colored_face")
OUT_ROOT = Path(r"D:\Project_Gujabaa\3D_Project\aligned_data_part_colored_face")

SAMPLE_N = 30000
SEED = 1234

ALLOW_REFLECTION = False
SKIP_IF_EXISTS = True

# Gingiva / base color
GINGIVA_RGB = (255, 180, 200)

# Accept filenames: *_U.ply/_L.ply and/or contains upper/lower
ENABLE_UPPER_LOWER_NAME_FALLBACK = True

# --- NEW: disambiguation weight ---
# RMS error is often ~0.01-0.1 (depends), so direction penalty should dominate only when flipped.
W_DIR = 0.05          # soft penalty for not perfectly aligned direction
HARD_FLIP_PENALTY = 1e3  # huge penalty if direction is opposite

# ============================================================
# PLY IO
# ============================================================
def read_ply_any(path: Path) -> PlyData:
    return PlyData.read(str(path))

def get_xyz_from_vertex(vdata) -> np.ndarray:
    names = vdata.dtype.names
    for k in ("x", "y", "z"):
        if k not in names:
            raise ValueError(f"Missing vertex property '{k}' in {names}")
    xyz = np.stack([vdata["x"], vdata["y"], vdata["z"]], axis=1).astype(np.float64)
    return xyz

def set_xyz_to_vertex(vdata, xyz: np.ndarray):
    vdata["x"] = xyz[:, 0].astype(vdata["x"].dtype, copy=False)
    vdata["y"] = xyz[:, 1].astype(vdata["y"].dtype, copy=False)
    vdata["z"] = xyz[:, 2].astype(vdata["z"].dtype, copy=False)

def has_normals(vdata) -> bool:
    names = vdata.dtype.names
    return all(k in names for k in ("nx", "ny", "nz"))

def get_normals(vdata) -> np.ndarray:
    return np.stack([vdata["nx"], vdata["ny"], vdata["nz"]], axis=1).astype(np.float64)

def set_normals(vdata, N: np.ndarray):
    vdata["nx"] = N[:, 0].astype(vdata["nx"].dtype, copy=False)
    vdata["ny"] = N[:, 1].astype(vdata["ny"].dtype, copy=False)
    vdata["nz"] = N[:, 2].astype(vdata["nz"].dtype, copy=False)

def write_ply_like(src_ply: PlyData, out_path: Path, new_vertex_data) -> None:
    elements = []
    for el in src_ply.elements:
        if el.name == "vertex":
            elements.append(PlyElement.describe(new_vertex_data, "vertex"))
        else:
            elements.append(PlyElement.describe(el.data, el.name))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_ply = PlyData(
        elements,
        text=src_ply.text,
        comments=list(getattr(src_ply, "comments", [])),
        obj_info=list(getattr(src_ply, "obj_info", [])),
    )
    out_ply.write(str(out_path))

# ============================================================
# Color helpers
# ============================================================
def _get_rgb_from_struct(arr):
    names = arr.dtype.names
    for keys in [("red", "green", "blue"), ("r", "g", "b")]:
        if all(k in names for k in keys):
            return np.stack([arr[keys[0]], arr[keys[1]], arr[keys[2]]], axis=1).astype(np.int32)
    return None

def get_faces(ply: PlyData):
    if "face" not in ply:
        return None
    f = ply["face"].data
    if "vertex_indices" not in f.dtype.names:
        return None
    return np.vstack(f["vertex_indices"]).astype(np.int64)

def tooth_vertex_mask_from_face_color(ply: PlyData, n_verts: int, gingiva_rgb=GINGIVA_RGB):
    if "face" not in ply:
        return None
    f = ply["face"].data
    faces = get_faces(ply)
    if faces is None:
        return None
    face_rgb = _get_rgb_from_struct(f)
    if face_rgb is None:
        return None

    ging = np.array(gingiva_rgb, dtype=np.int32)
    tooth_face = np.any(face_rgb != ging[None, :], axis=1)
    if int(tooth_face.sum()) < 50:
        return None

    idx = faces[tooth_face].reshape(-1)
    idx = idx[(idx >= 0) & (idx < n_verts)]
    if len(idx) == 0:
        return None

    mask = np.zeros((n_verts,), dtype=bool)
    mask[np.unique(idx)] = True
    return mask

def tooth_vertex_mask_from_vertex_color(ply: PlyData, gingiva_rgb=GINGIVA_RGB):
    v = ply["vertex"].data
    v_rgb = _get_rgb_from_struct(v)
    if v_rgb is None:
        return None
    ging = np.array(gingiva_rgb, dtype=np.int32)
    mask = np.any(v_rgb != ging[None, :], axis=1)
    if int(mask.sum()) < 50:
        return None
    return mask

def extract_tooth_and_dir(ply: PlyData, sample_n: int, rng: np.random.Generator):
    """
    Returns:
      P_tooth (Nx3) for alignment,
      dir_vec (3,) = centroid(tooth) - centroid(gingiva) in normalized coord (may be None)
      c, s used for normalization
    """
    v = ply["vertex"].data
    V = get_xyz_from_vertex(v)
    nV = len(V)

    mask = tooth_vertex_mask_from_face_color(ply, nV, gingiva_rgb=GINGIVA_RGB)
    if mask is None:
        mask = tooth_vertex_mask_from_vertex_color(ply, gingiva_rgb=GINGIVA_RGB)

    if mask is not None and int(mask.sum()) >= 50:
        V_tooth = V[mask]
        V_ging  = V[~mask] if int((~mask).sum()) >= 50 else None
    else:
        V_tooth = V
        V_ging = None

    # sample tooth points for alignment
    if len(V_tooth) > sample_n:
        idx = rng.choice(len(V_tooth), size=sample_n, replace=False)
        P = V_tooth[idx].astype(np.float64, copy=False)
    else:
        P = V_tooth.astype(np.float64, copy=False)

    # normalization based on tooth points
    c = V_tooth.mean(axis=0)
    Q = V_tooth - c
    s = np.linalg.norm(Q.max(axis=0) - Q.min(axis=0)) + 1e-12

    # direction vector if gingiva exists
    dir_vec = None
    if V_ging is not None and len(V_ging) >= 50:
        ct = V_tooth.mean(axis=0)
        cg = V_ging.mean(axis=0)
        ct_n = (ct - c) / s
        cg_n = (cg - c) / s
        d = ct_n - cg_n
        if np.linalg.norm(d) > 1e-9:
            dir_vec = d.astype(np.float64)

    # normalize tooth points
    Pn = (P - c) / s
    return Pn, dir_vec

# ============================================================
# Alignment utils
# ============================================================
def pca_basis(P: np.ndarray) -> np.ndarray:
    C = (P.T @ P) / max(len(P), 1)
    w, V = np.linalg.eigh(C)
    order = np.argsort(w)[::-1]
    U = V[:, order]
    if np.linalg.det(U) < 0:
        U[:, -1] *= -1
    return U

def generate_axis_rotations(allow_reflection: bool = False):
    mats = []
    perms = [
        (0, 1, 2), (0, 2, 1),
        (1, 0, 2), (1, 2, 0),
        (2, 0, 1), (2, 1, 0),
    ]
    signs = [
        (1, 1, 1), (1, 1, -1), (1, -1, 1), (1, -1, -1),
        (-1, 1, 1), (-1, 1, -1), (-1, -1, 1), (-1, -1, -1),
    ]
    for p in perms:
        Pm = np.zeros((3, 3), dtype=np.float64)
        for i, j in enumerate(p):
            Pm[j, i] = 1.0
        for s in signs:
            Sm = np.diag(s)
            M = Pm @ Sm
            det = int(round(np.linalg.det(M)))
            if allow_reflection or det == 1:
                mats.append(M)

    uniq, seen = [], set()
    for M in mats:
        key = tuple(np.round(M.flatten(), 6))
        if key not in seen:
            seen.add(key)
            uniq.append(M)
    return uniq

def score_alignment(P_tgt: np.ndarray, P_ref: np.ndarray) -> float:
    tree = cKDTree(P_ref)
    d, _ = tree.query(P_tgt, k=1, workers=-1)
    return float(np.sqrt((d * d).mean()))

def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    if n < 1e-12:
        return v
    return v / n

def best_rotation_to_reference(P_tgt: np.ndarray, P_ref: np.ndarray,
                               dir_tgt: np.ndarray | None,
                               dir_ref: np.ndarray | None,
                               allow_reflection=False):
    """
    Choose rotation by:
      - geom RMS (KDTree)
      - + direction constraint (tooth above gingiva)
    """
    U_t = pca_basis(P_tgt)
    U_r = pca_basis(P_ref)
    candidates = generate_axis_rotations(allow_reflection=allow_reflection)

    best_score = 1e18
    best_R = None
    best_err = None
    best_dot = None

    uref = _unit(dir_ref) if dir_ref is not None else None
    utgt = _unit(dir_tgt) if dir_tgt is not None else None

    for Q in candidates:
        R = U_r @ Q @ U_t.T
        if not allow_reflection and np.linalg.det(R) < 0:
            continue

        # geom
        P_rot = P_tgt @ R.T
        err = score_alignment(P_rot, P_ref)

        # direction penalty
        penalty = 0.0
        dotv = None
        if (uref is not None) and (utgt is not None):
            d_rot = utgt @ R.T
            d_rot = _unit(d_rot)
            dotv = float(np.dot(d_rot, uref))

            if dotv < 0.0:
                penalty += HARD_FLIP_PENALTY
            else:
                penalty += W_DIR * (1.0 - dotv)  # small preference to align direction

        score = err + penalty
        if score < best_score:
            best_score = score
            best_R = R
            best_err = err
            best_dot = dotv

    if best_R is None:
        raise RuntimeError("No valid rotation found.")
    return best_R, float(best_err), best_dot, float(best_score)

def apply_rotation_to_vertex_data(vdata, R: np.ndarray):
    V = get_xyz_from_vertex(vdata)
    c = V.mean(axis=0)
    Vc = V - c
    Vn = Vc @ R.T + c
    set_xyz_to_vertex(vdata, Vn)

    if has_normals(vdata):
        N = get_normals(vdata)
        Nn = N @ R.T
        Nn /= (np.linalg.norm(Nn, axis=1, keepdims=True) + 1e-12)
        set_normals(vdata, Nn)

# ============================================================
# Reference building
# ============================================================
def build_reference(ref_path: Path, rng: np.random.Generator):
    ply = read_ply_any(ref_path)
    P_ref, dir_ref = extract_tooth_and_dir(ply, min(SAMPLE_N, 50000), rng)
    return P_ref, dir_ref

# ============================================================
# Recursive scanning
# ============================================================
def classify_arch(p: Path) -> str | None:
    nm = p.name.lower()
    if nm.endswith("_u.ply"):
        return "upper"
    if nm.endswith("_l.ply"):
        return "lower"
    if ENABLE_UPPER_LOWER_NAME_FALLBACK:
        if "upper" in nm:
            return "upper"
        if "lower" in nm:
            return "lower"
    return None

def list_candidate_files(root: Path) -> list[Path]:
    out = []
    for p in root.rglob("*.ply"):
        if classify_arch(p) is not None:
            out.append(p)
    return sorted(out)

def out_path_for(p: Path) -> Path:
    rel = p.relative_to(IN_ROOT)
    return OUT_ROOT / rel

# ============================================================
# Processing
# ============================================================
def process_file(p: Path, out_path: Path, pref_pts: np.ndarray, pref_dir: np.ndarray | None, rng: np.random.Generator):
    ply = read_ply_any(p)
    vdata = ply["vertex"].data.copy()

    P_tgt, dir_tgt = extract_tooth_and_dir(ply, min(SAMPLE_N, 50000), rng)

    R, err, dotv, score = best_rotation_to_reference(
        P_tgt, pref_pts, dir_tgt, pref_dir, allow_reflection=ALLOW_REFLECTION
    )
    apply_rotation_to_vertex_data(vdata, R)
    write_ply_like(ply, out_path, vdata)

    if dotv is None:
        print(f"[OK] {p}  rms={err:.6f}  -> {out_path}")
    else:
        print(f"[OK] {p}  rms={err:.6f}  dot={dotv:+.3f}  score={score:.6f}  -> {out_path}")

def main():
    rng = np.random.default_rng(SEED)

    if not SET2_UPPER_REF.exists() or not SET2_LOWER_REF.exists():
        raise FileNotFoundError("Missing reference PLY(s). Check SET2_*_REF paths.")
    if not IN_ROOT.exists():
        raise FileNotFoundError(IN_ROOT)

    pref_upper_pts, pref_upper_dir = build_reference(SET2_UPPER_REF, rng)
    pref_lower_pts, pref_lower_dir = build_reference(SET2_LOWER_REF, rng)

    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    files = list_candidate_files(IN_ROOT)
    print(f"[IN ] {IN_ROOT}")
    print(f"[FOUND] {len(files)} candidate .ply files (recursive)")
    for ex in files[:5]:
        print(f"  - {ex}")

    if not files:
        print("[WARN] no matching files found")
        return

    ok = fail = skip = 0

    for p in files:
        arch = classify_arch(p)
        if arch is None:
            continue

        out_path = out_path_for(p)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        if SKIP_IF_EXISTS and out_path.exists():
            print(f"[SKIP] exists: {out_path}")
            skip += 1
            continue

        try:
            if arch == "upper":
                process_file(p, out_path, pref_upper_pts, pref_upper_dir, rng)
            else:
                process_file(p, out_path, pref_lower_pts, pref_lower_dir, rng)
            ok += 1
        except Exception as e:
            print(f"[FAIL] {p}: {e}")
            fail += 1

    print("============================================================")
    print(f"[DONE] OK={ok}  FAIL={fail}  SKIP={skip}")
    print(f"OUT_ROOT: {OUT_ROOT}")
    print("============================================================")

if __name__ == "__main__":
    main()




# #---- ปรับทิศของ private data (scanfile_ply) ให้เหมือนกับ public data ----#
# from __future__ import annotations
# from pathlib import Path
# import numpy as np

# try:
#     from scipy.spatial import cKDTree
# except Exception as e:
#     raise ImportError(
#         "This script requires SciPy for KDTree.\n"
#         "Install with: pip install scipy\n"
#         f"Original error: {e}"
#     )

# try:
#     from plyfile import PlyData, PlyElement
# except Exception as e:
#     raise ImportError(
#         "This script requires plyfile.\n"
#         "Install with: pip install plyfile\n"
#         f"Original error: {e}"
#     )

# # ============================================================
# # CONFIG (EDIT THESE)
# # ============================================================
# # Reference (Set2)
# SET2_UPPER_REF = Path(r"D:\Project_Gujabaa\3D_Project\public_data\train\007_U.ply")
# SET2_LOWER_REF = Path(r"D:\Project_Gujabaa\public_data\train\007_L.ply")

# # Input root: colored_merged/<case>/<case>_U.ply and <case>_L.ply
# IN_ROOT = Path(r"D:\Project_Gujabaa\3D_Project\data_part")

# # Output root: aligned_colored_merged/<case>/<same_name>.ply
# OUT_ROOT = Path(r"D:\Project_Gujabaa\3D_Project\aligned_data_part")

# # Sampling / scoring
# SAMPLE_N = 30000
# SEED = 1234

# # If you suspect some files are mirrored (left-right swapped), set True.
# ALLOW_REFLECTION = False

# # Skip if output exists (resume-friendly)
# SKIP_IF_EXISTS = True

# # ============================================================
# # PLY IO (robust: supports ASCII/Binary, keeps properties)
# # ============================================================
# def read_ply_any(path: Path) -> PlyData:
#     return PlyData.read(str(path))

# def get_xyz_from_vertex(vdata) -> np.ndarray:
#     names = vdata.dtype.names
#     for k in ("x", "y", "z"):
#         if k not in names:
#             raise ValueError(f"Missing vertex property '{k}' in {names}")
#     xyz = np.stack([vdata["x"], vdata["y"], vdata["z"]], axis=1).astype(np.float64)
#     return xyz

# def set_xyz_to_vertex(vdata, xyz: np.ndarray):
#     vdata["x"] = xyz[:, 0].astype(vdata["x"].dtype, copy=False)
#     vdata["y"] = xyz[:, 1].astype(vdata["y"].dtype, copy=False)
#     vdata["z"] = xyz[:, 2].astype(vdata["z"].dtype, copy=False)

# def has_normals(vdata) -> bool:
#     names = vdata.dtype.names
#     return all(k in names for k in ("nx", "ny", "nz"))

# def get_normals(vdata) -> np.ndarray:
#     N = np.stack([vdata["nx"], vdata["ny"], vdata["nz"]], axis=1).astype(np.float64)
#     return N

# def set_normals(vdata, N: np.ndarray):
#     vdata["nx"] = N[:, 0].astype(vdata["nx"].dtype, copy=False)
#     vdata["ny"] = N[:, 1].astype(vdata["ny"].dtype, copy=False)
#     vdata["nz"] = N[:, 2].astype(vdata["nz"].dtype, copy=False)

# def write_ply_like(src_ply: PlyData, out_path: Path, new_vertex_data) -> None:
#     """
#     Write new PLY keeping:
#       - element order
#       - non-vertex elements (face + any extras)
#       - comments/obj_info
#       - ASCII/Binary mode (same as input)
#     """
#     elements = []
#     for el in src_ply.elements:
#         if el.name == "vertex":
#             elements.append(PlyElement.describe(new_vertex_data, "vertex"))
#         else:
#             elements.append(PlyElement.describe(el.data, el.name))

#     out_path.parent.mkdir(parents=True, exist_ok=True)
#     out_ply = PlyData(
#         elements,
#         text=src_ply.text,                 # keep ascii/binary
#         comments=list(getattr(src_ply, "comments", [])),
#         obj_info=list(getattr(src_ply, "obj_info", [])),
#     )
#     out_ply.write(str(out_path))

# # ============================================================
# # Alignment utilities
# # ============================================================
# def sample_vertices(V: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
#     if len(V) <= n:
#         return V.copy()
#     idx = rng.choice(len(V), size=n, replace=False)
#     return V[idx]

# def center_and_scale(P: np.ndarray):
#     c = P.mean(axis=0)
#     Q = P - c
#     s = np.linalg.norm(Q.max(axis=0) - Q.min(axis=0)) + 1e-12  # bbox diag
#     return Q / s, c, s

# def pca_basis(P: np.ndarray) -> np.ndarray:
#     C = (P.T @ P) / max(len(P), 1)
#     w, V = np.linalg.eigh(C)
#     order = np.argsort(w)[::-1]
#     U = V[:, order]
#     if np.linalg.det(U) < 0:
#         U[:, -1] *= -1
#     return U

# def generate_axis_rotations(allow_reflection: bool = False):
#     mats = []
#     perms = [
#         (0,1,2),(0,2,1),
#         (1,0,2),(1,2,0),
#         (2,0,1),(2,1,0)
#     ]
#     signs = [
#         (1,1,1),(1,1,-1),(1,-1,1),(1,-1,-1),
#         (-1,1,1),(-1,1,-1),(-1,-1,1),(-1,-1,-1),
#     ]
#     for p in perms:
#         Pm = np.zeros((3,3), dtype=np.float64)
#         for i, j in enumerate(p):
#             Pm[j, i] = 1.0
#         for s in signs:
#             Sm = np.diag(s)
#             M = Pm @ Sm
#             det = int(round(np.linalg.det(M)))
#             if allow_reflection or det == 1:
#                 mats.append(M)

#     uniq, seen = [], set()
#     for M in mats:
#         key = tuple(np.round(M.flatten(), 6))
#         if key not in seen:
#             seen.add(key)
#             uniq.append(M)
#     return uniq

# def score_alignment(P_tgt: np.ndarray, P_ref: np.ndarray) -> float:
#     tree = cKDTree(P_ref)
#     d, _ = tree.query(P_tgt, k=1, workers=-1)
#     return float(np.sqrt((d * d).mean()))

# def best_rotation_to_reference(P_tgt: np.ndarray, P_ref: np.ndarray, allow_reflection=False):
#     U_t = pca_basis(P_tgt)
#     U_r = pca_basis(P_ref)
#     candidates = generate_axis_rotations(allow_reflection=allow_reflection)

#     best_err = 1e18
#     best_R = None

#     for Q in candidates:
#         R = U_r @ Q @ U_t.T
#         if not allow_reflection and np.linalg.det(R) < 0:
#             continue

#         P_rot = P_tgt @ R.T
#         err = score_alignment(P_rot, P_ref)

#         if err < best_err:
#             best_err = err
#             best_R = R

#     if best_R is None:
#         raise RuntimeError("No valid rotation found.")
#     return best_R, best_err

# def apply_rotation_to_vertex_data(vdata, R: np.ndarray):
#     """
#     Rotate x,y,z and nx,ny,nz (if exist) around centroid.
#     Keeps all colors/labels intact (we don't touch them).
#     """
#     V = get_xyz_from_vertex(vdata)
#     c = V.mean(axis=0)
#     Vc = V - c
#     Vn = Vc @ R.T + c
#     set_xyz_to_vertex(vdata, Vn)

#     if has_normals(vdata):
#         N = get_normals(vdata)
#         Nn = N @ R.T
#         Nn /= (np.linalg.norm(Nn, axis=1, keepdims=True) + 1e-12)
#         set_normals(vdata, Nn)

# # ============================================================
# # Reference building
# # ============================================================
# def build_reference_points(ref_path: Path, rng: np.random.Generator) -> np.ndarray:
#     ply = read_ply_any(ref_path)
#     v = ply["vertex"].data
#     V = get_xyz_from_vertex(v)

#     V = sample_vertices(V, min(SAMPLE_N, len(V)), rng)
#     Vn, _, _ = center_and_scale(V)
#     return Vn

# # ============================================================
# # Batch scanning: only merged files at case root
# # ============================================================
# def list_case_dirs(root: Path) -> list[Path]:
#     return [p for p in sorted(root.iterdir()) if p.is_dir()]

# def find_merged_ul(case_dir: Path) -> list[Path]:
#     # only files at case root (NOT rglob)
#     out = []
#     for p in case_dir.iterdir():
#         if not p.is_file() or p.suffix.lower() != ".ply":
#             continue
#         nm = p.name.lower()
#         if nm.endswith("_u.ply") or nm.endswith("_l.ply"):
#             out.append(p)
#     return sorted(out)

# def process_file(p: Path, out_path: Path, pref: np.ndarray, rng: np.random.Generator):
#     ply = read_ply_any(p)
#     vdata = ply["vertex"].data.copy()  # copy so we can edit

#     V = get_xyz_from_vertex(vdata)
#     V_s = sample_vertices(V, min(SAMPLE_N, len(V)), rng)
#     Vn, _, _ = center_and_scale(V_s)

#     R, err = best_rotation_to_reference(Vn, pref, allow_reflection=ALLOW_REFLECTION)
#     apply_rotation_to_vertex_data(vdata, R)

#     write_ply_like(ply, out_path, vdata)
#     print(f"[OK] {p}  score={err:.6f}  -> {out_path}")

# def main():
#     rng = np.random.default_rng(SEED)

#     if not SET2_UPPER_REF.exists() or not SET2_LOWER_REF.exists():
#         raise FileNotFoundError("Missing Set2 reference PLY(s). Check SET2_*_REF paths.")

#     if not IN_ROOT.exists():
#         raise FileNotFoundError(IN_ROOT)

#     # Build references
#     pref_upper = build_reference_points(SET2_UPPER_REF, rng)
#     pref_lower = build_reference_points(SET2_LOWER_REF, rng)

#     case_dirs = list_case_dirs(IN_ROOT)
#     if not case_dirs:
#         print(f"[WARN] no case folders under {IN_ROOT}")
#         return

#     OUT_ROOT.mkdir(parents=True, exist_ok=True)

#     ok = 0
#     fail = 0
#     skip = 0

#     for case_dir in case_dirs:
#         merged = find_merged_ul(case_dir)
#         if not merged:
#             print(f"[SKIP] {case_dir.name}: no merged *_U.ply/*_L.ply at case root")
#             continue

#         out_case = OUT_ROOT / case_dir.name
#         out_case.mkdir(parents=True, exist_ok=True)

#         for p in merged:
#             out_path = out_case / p.name
#             if SKIP_IF_EXISTS and out_path.exists():
#                 print(f"[SKIP] exists: {out_path}")
#                 skip += 1
#                 continue

#             try:
#                 if p.name.lower().endswith("_u.ply"):
#                     process_file(p, out_path, pref_upper, rng)
#                 else:
#                     process_file(p, out_path, pref_lower, rng)
#                 ok += 1
#             except Exception as e:
#                 print(f"[FAIL] {p}: {e}")
#                 fail += 1

#     print("============================================================")
#     print(f"[DONE] OK={ok}  FAIL={fail}  SKIP={skip}")
#     print(f"OUT_ROOT: {OUT_ROOT}")
#     print("============================================================")

# if __name__ == "__main__":
#     main()
