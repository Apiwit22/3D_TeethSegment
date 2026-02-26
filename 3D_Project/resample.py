# ---- resample face ให้รวม 16000 เป๊ะๆ ต่อ arch (upper/lower) ---- #
from __future__ import annotations

from pathlib import Path
import re
import time
import numpy as np

import open3d as o3d

# ============================================================
# CONFIG (EDIT)
# ============================================================
IN_ROOT = Path(r"D:\Project_Gujabaa\scanfile_ply")   # root with case folders, or a single case folder
OUT_ROOT = Path(r"D:\Project_Gujabaa\resampled")

TARGET_FACES_PER_ARCH = 16000

MIN_TOOTH_FACES = 350
MIN_JAW_FACES = 2500

IGNORE_NAMES_CONTAINS = {"temp", "junk"}

WRITE_ASCII = True  # True=ascii ply (debug-friendly), False=binary

# speed / robustness knobs
VERBOSE = True
MAX_POSTFIX_ITERS = 10
MAX_EXACT_TRIES = 8
CLEAN_HEAVY_AT_LOAD = True      # heavy clean once when loading
CLEAN_HEAVY_IN_EXACT = False    # heavy clean every try is slow; keep False normally

# ============================================================
# Helpers: parse arch / part type
# ============================================================
_RE_INTS = re.compile(r"(\d{2,})")

def parse_fdi(stem: str) -> int | None:
    """
    Robust: find all 2+ digit numbers; return first that looks like FDI 11-48.
    """
    s = stem.lower()
    for m in _RE_INTS.findall(s):
        try:
            n = int(m)
            if 11 <= n <= 48:
                return n
        except:
            pass
    return None

def infer_arch(p: Path) -> str | None:
    s = p.stem.lower()
    if "upper" in s or "maxilla" in s:
        return "upper"
    if "lower" in s or "mandible" in s:
        return "lower"

    fdi = parse_fdi(s)
    if fdi is not None:
        if 11 <= fdi <= 28:
            return "upper"
        if 31 <= fdi <= 48:
            return "lower"
    return None

def is_jaw_part(p: Path) -> bool:
    s = p.stem.lower()
    if "jaw" in s or "gingiva" in s or "gum" in s:
        return True
    fdi = parse_fdi(s)
    if fdi is None:
        return True
    return not (11 <= fdi <= 28 or 31 <= fdi <= 48)

# ============================================================
# Mesh utilities
# ============================================================
def load_mesh(path: Path) -> o3d.geometry.TriangleMesh:
    mesh = o3d.io.read_triangle_mesh(str(path))
    if mesh.is_empty():
        raise ValueError(f"Empty mesh: {path}")
    if not mesh.has_triangles():
        raise ValueError(f"No triangles: {path}")
    return mesh

def face_count(mesh: o3d.geometry.TriangleMesh) -> int:
    return int(np.asarray(mesh.triangles).shape[0])

def mesh_area(mesh: o3d.geometry.TriangleMesh) -> float:
    return float(mesh.get_surface_area())

def clean_mesh_light(mesh: o3d.geometry.TriangleMesh) -> o3d.geometry.TriangleMesh:
    mesh = mesh.remove_duplicated_vertices()
    mesh = mesh.remove_duplicated_triangles()
    mesh = mesh.remove_degenerate_triangles()
    return mesh

def clean_mesh_heavy(mesh: o3d.geometry.TriangleMesh) -> o3d.geometry.TriangleMesh:
    mesh = clean_mesh_light(mesh)
    mesh = mesh.remove_non_manifold_edges()
    return mesh

def simplify(mesh: o3d.geometry.TriangleMesh, target_faces: int) -> o3d.geometry.TriangleMesh:
    cur = face_count(mesh)
    target_faces = int(max(4, min(target_faces, cur)))
    if target_faces >= cur:
        out = mesh
    else:
        out = mesh.simplify_quadric_decimation(target_number_of_triangles=target_faces)
    out = clean_mesh_light(out)
    out.compute_vertex_normals()
    return out

def decimate_exact(mesh: o3d.geometry.TriangleMesh, target_faces: int, max_tries: int = 8) -> o3d.geometry.TriangleMesh:
    """
    Try to produce exactly target_faces by adjusting request iteratively.
    """
    orig = face_count(mesh)
    target_faces = int(max(4, min(target_faces, orig)))

    req = target_faces
    last = None

    for t in range(max_tries):
        out = mesh.simplify_quadric_decimation(target_number_of_triangles=int(req))
        out = clean_mesh_heavy(out) if CLEAN_HEAVY_IN_EXACT else clean_mesh_light(out)
        out.compute_vertex_normals()

        cur = face_count(out)
        last = out

        if VERBOSE:
            print(f"      exact try {t+1}/{max_tries}: req={req} -> cur={cur} (target={target_faces})")

        if cur == target_faces:
            return out

        # adjust request based on observed delta
        delta = target_faces - cur
        req = int(max(4, min(orig, req + delta)))

        # small nudge for tiny diffs
        if abs(delta) <= 2:
            req = int(max(4, min(orig, req + (1 if delta > 0 else -1))))

    return last if last is not None else mesh

def trim_faces_exact(mesh: o3d.geometry.TriangleMesh, target_faces: int) -> o3d.geometry.TriangleMesh:
    """
    Guarantee: if mesh has > target_faces, drop triangles to exactly target_faces.
    (Use ONLY for 'excess' correction.)
    """
    tris = np.asarray(mesh.triangles)
    cur = tris.shape[0]
    if cur <= target_faces:
        return mesh

    tris2 = tris[:target_faces].copy()
    out = o3d.geometry.TriangleMesh()
    out.vertices = mesh.vertices
    out.triangles = o3d.utility.Vector3iVector(tris2)
    out = clean_mesh_light(out)
    out.compute_vertex_normals()

    # Note: clean_mesh_light could remove a few degenerate faces -> might go below target.
    # But for most clean meshes it stays exact. If it drops, we'll handle in post-fix loop.
    return out

# ============================================================
# Allocation: sum alloc ~= target_total
# ============================================================
def allocate_faces(files, areas, orig_faces, target_total, mins):
    areas = np.array(areas, dtype=np.float64)
    orig_faces = np.array(orig_faces, dtype=np.int64)
    mins = np.minimum(np.array(mins, dtype=np.int64), orig_faces)

    total_area = float(areas.sum())
    if total_area <= 1e-12:
        w = orig_faces.astype(np.float64)
        raw = target_total * (w / (float(w.sum()) + 1e-12))
    else:
        raw = target_total * (areas / total_area)

    alloc = np.rint(raw).astype(np.int64)
    alloc = np.maximum(alloc, mins)
    alloc = np.minimum(alloc, orig_faces)

    # reduce if too high
    s = int(alloc.sum())
    if s > target_total:
        excess = s - target_total
        order = np.argsort(-alloc)
        for idx in order:
            if excess <= 0:
                break
            slack = int(alloc[idx] - mins[idx])
            if slack <= 0:
                continue
            d = min(slack, excess)
            alloc[idx] -= d
            excess -= d

    # increase if too low
    s = int(alloc.sum())
    if s < target_total:
        need = target_total - s
        jaw_mask = np.array([is_jaw_part(f) for f in files], dtype=bool)
        jaw_idxs = np.where(jaw_mask)[0]
        tooth_idxs = np.where(~jaw_mask)[0]

        jaw_order = jaw_idxs[np.argsort(-areas[jaw_idxs])] if len(jaw_idxs) else np.array([], dtype=int)
        tooth_order = tooth_idxs[np.argsort(-areas[tooth_idxs])] if len(tooth_idxs) else np.array([], dtype=int)
        order = np.concatenate([jaw_order, tooth_order])

        for idx in order:
            if need <= 0:
                break
            cap = int(orig_faces[idx] - alloc[idx])
            if cap <= 0:
                continue
            d = min(cap, need)
            alloc[idx] += d
            need -= d

    s = int(alloc.sum())
    if s != target_total:
        print(f"[WARN] Allocation sum != {target_total}: got {s}, orig_sum={int(orig_faces.sum())}")
    return alloc

def choose_absorber_idx(files: list[Path], face_list: list[int]) -> int:
    jaw_idxs = [i for i, f in enumerate(files) if is_jaw_part(f)]
    if jaw_idxs:
        return max(jaw_idxs, key=lambda i: face_list[i])
    return int(np.argmax(np.array(face_list)))

# ============================================================
# Case processing
# ============================================================
def list_case_dirs(root: Path) -> list[Path]:
    if not root.exists():
        raise ValueError(f"IN_ROOT not found: {root}")

    # If root is a single case folder (contains ply files and no subdirs)
    if root.is_dir() and any(root.glob("*.ply")) and not any(p.is_dir() for p in root.iterdir()):
        return [root]

    if root.is_dir():
        return [p for p in sorted(root.iterdir()) if p.is_dir()]

    raise ValueError(f"IN_ROOT is not a folder: {root}")

def process_arch(case_id: str, arch: str, files: list[Path], out_dir: Path):
    if not files:
        return

    meshes, areas, orig_faces, mins = [], [], [], []

    # load meshes & stats
    for f in files:
        m = load_mesh(f)
        m = clean_mesh_heavy(m) if CLEAN_HEAVY_AT_LOAD else clean_mesh_light(m)
        m.compute_vertex_normals()

        meshes.append(m)
        areas.append(mesh_area(m))
        of = face_count(m)
        orig_faces.append(of)
        mins.append(MIN_JAW_FACES if is_jaw_part(f) else MIN_TOOTH_FACES)

    alloc = allocate_faces(files, areas, orig_faces, TARGET_FACES_PER_ARCH, mins)

    print(f"\n[CASE {case_id}] {arch}: parts={len(files)}  target_total={TARGET_FACES_PER_ARCH}")
    for f, a, of, al in sorted(zip(files, areas, orig_faces, alloc), key=lambda x: -x[3]):
        tag = "JAW " if is_jaw_part(f) else "TOOTH"
        print(f"  - {f.name:25s} {tag}  orig={of:7d}  alloc={int(al):6d}  area={a:10.2f}")

    out_dir.mkdir(parents=True, exist_ok=True)

    # first pass simplify per-part
    out_paths, out_meshes, actual_faces = [], [], []
    for f, m, al in zip(files, meshes, alloc):
        if VERBOSE:
            print(f"    -> simplify {f.name} to ~{int(al)} faces")
        t0 = time.time()
        out = simplify(m, int(al))
        dt = time.time() - t0
        cur = face_count(out)
        if VERBOSE:
            print(f"       done {f.name}: {cur} faces ({dt:.1f}s)")

        out_paths.append(out_dir / f.name)
        out_meshes.append(out)
        actual_faces.append(cur)

    # post-fix loop to guarantee total==TARGET
    target = TARGET_FACES_PER_ARCH
    total = int(sum(actual_faces))
    diff = target - total
    if VERBOSE:
        print(f"    [SUM] first pass total={total} diff={diff}")

    idx = choose_absorber_idx(files, actual_faces)

    it = 0
    while diff != 0 and it < MAX_POSTFIX_ITERS:
        it += 1
        absorber_name = files[idx].name

        if diff < 0:
            # total too high -> GUARANTEE: trim |diff| faces from absorber
            need_drop = -diff
            new_faces = max(4, actual_faces[idx] - need_drop)
            if VERBOSE:
                print(f"    [POST-FIX {it}] total high by {need_drop}. TRIM absorber {absorber_name}: {actual_faces[idx]} -> {new_faces}")
            out_meshes[idx] = trim_faces_exact(out_meshes[idx], new_faces)
            actual_faces[idx] = face_count(out_meshes[idx])

        else:
            # total too low -> increase absorber by diff using exact decimation from ORIGINAL mesh
            new_target = min(orig_faces[idx], actual_faces[idx] + diff)
            new_target = max(4, int(new_target))
            if VERBOSE:
                print(f"    [POST-FIX {it}] total low by {diff}. EXACT absorber {absorber_name}: target={new_target}")
            out_meshes[idx] = decimate_exact(meshes[idx], new_target, max_tries=MAX_EXACT_TRIES)
            actual_faces[idx] = face_count(out_meshes[idx])

        total = int(sum(actual_faces))
        diff = target - total
        if VERBOSE:
            print(f"    [SUM] after fix total={total} diff={diff}")

        # If trim caused slight underflow due to clean removing degens, re-run exact to fill.
        if diff > 0 and it < MAX_POSTFIX_ITERS:
            # keep absorber fixed; next loop will fill
            pass

    if diff != 0:
        print(f"    [WARN] Could not reach EXACT {target}. final_total={total} diff={diff}")
    else:
        print(f"    [OK] EXACT total={total}")

    # write outputs
    total_written = 0
    for p, m in zip(out_paths, out_meshes):
        o3d.io.write_triangle_mesh(str(p), m, write_ascii=WRITE_ASCII)
        total_written += face_count(m)

    print(f"[CASE {case_id}] {arch}: saved -> {out_dir}  actual_total_faces={total_written}")

def process_case(case_dir: Path):
    case_id = case_dir.name
    files = [
        p for p in sorted(case_dir.glob("*.ply"))
        if not any(k in p.name.lower() for k in IGNORE_NAMES_CONTAINS)
    ]

    upper = [p for p in files if infer_arch(p) == "upper"]
    lower = [p for p in files if infer_arch(p) == "lower"]

    if not upper and not lower:
        print(f"[SKIP] {case_id}: no upper/lower parts inferred.")
        return

    out_case = OUT_ROOT / case_id
    process_arch(case_id, "upper", upper, out_case / "upper_parts")
    process_arch(case_id, "lower", lower, out_case / "lower_parts")

def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    case_dirs = list_case_dirs(IN_ROOT)
    for cd in case_dirs:
        if any(cd.glob("*.ply")):
            process_case(cd)
    print(f"\n[DONE] resampled parts saved to: {OUT_ROOT}")

if __name__ == "__main__":
    main()
