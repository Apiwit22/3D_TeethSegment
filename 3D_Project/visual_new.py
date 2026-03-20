from __future__ import annotations

import copy
import inspect
import json
import shutil
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

# ============================================================
# CONFIG 
# ============================================================
RUN_DIR = Path(r"D:\Project_Gujabaa\3D_Project\checkpoints\tsgcnet\TSGCNet2_IoSSeg_ALLDATA_200epoch_run1")
SPLIT_ROOT = Path(r"D:\Project_Gujabaa\3D_Project\FinalDataset_Split")
VIS_OUT = Path(r"D:\Project_Gujabaa\3D_Project\vis_final\tsgcnet") / RUN_DIR.name

CKPT_TAG = "best"  # "best" or "last"
ARCHES = ["upper", "lower"]
MODE = "auto"
UNKNOWN_POLICY = "ignore"
ALIGN_MODE = "auto"

# Export GT
EXPORT_GT = True
EXPORT_GT_PALETTE_COPY = True

# ------------------------------------------------------------
# Post-process: confidence handling (Policy C)
# ------------------------------------------------------------
CONF_THRESH = 0.60               
CONF_LOWCONF_POLICY = "neighbor"
CONF_NEIGHBOR_ITERS = 1          
CONF_APPLY_BEFORE_CLEAN = True   
CONF_APPLY_BEFORE_SMOOTH = True  

# ------------------------------------------------------------
# Post-process: component cleanup 
# ------------------------------------------------------------
CLEAN_SMALL_COMPONENTS = True
MIN_COMP_SIZE = 20               
CLEAN_ITERS = 1                  

# ------------------------------------------------------------
# Post-process: smoothing 
# ------------------------------------------------------------
SMOOTH_ITERS = 1                 
SMOOTH_METHOD = "adjacency"      
SMOOTH_KNN_K = 16                

# แยก subfolder ตามเคส
CASE_SUBFOLDERS = True

# Debug
DEBUG_PRINT_FIRST_OUT = True
DEBUG_BATCH_KEYS = False

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ============================================================
# constants
# ============================================================
GINGIVA_RGB_DEFAULT = (255, 180, 200)

# ============================================================
# Optional deps
# ============================================================
try:
    import trimesh
except Exception as e:
    raise ImportError("This script requires trimesh. Install with: pip install trimesh") from e

try:
    from scipy.spatial import cKDTree as _cKDTree
except Exception:
    _cKDTree = None

# ============================================================
# Imports from your project
# ============================================================
try:
    from src.dataloader import build_dataset
    from src.dataloader.collate import collate_face, collate_point, collate_graph
    from src.dataloader.fdi_colors import FDIColorMap, GINGIVA_LABEL
except Exception as e:
    raise RuntimeError(
        "Failed to import from src.dataloader.\n"
        "Fix src/dataloader/__init__.py to avoid hard-importing missing optional modules.\n"
        f"Original error: {repr(e)}"
    ) from e


# ============================================================
# Utilities
# ============================================================
def load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def import_symbol(spec: str):
    """
    spec: "module.sub:Name"
    """
    if ":" not in spec:
        raise ValueError(f"import spec must be like module:Name, got {spec}")
    mod, name = spec.split(":", 1)
    m = __import__(mod, fromlist=[name])
    return getattr(m, name)


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def case_id_from_path(p: Path) -> str:
    return p.stem


def arch_from_name(p: Path) -> Optional[str]:
    nm = p.name.lower()
    if nm.endswith("_u.ply") or "upper" in nm:
        return "upper"
    if nm.endswith("_l.ply") or "lower" in nm:
        return "lower"
    return None


DEFAULT_TEST_FOLD = {
    "dataset1": "fold1",
    "dataset2": "fold2",
    "dataset3": "fold3",
    "dataset4": "fold4",
    "dataset5": "fold5",
}


def get_test_fold_for_dataset(cfg: dict, dataset: str) -> str:
    tf = (cfg.get("data", {}) or {}).get("test_fold_by_dataset", None)
    if isinstance(tf, dict) and dataset in tf:
        return str(tf[dataset])
    return DEFAULT_TEST_FOLD.get(dataset, "fold1")


def list_ply_files(test_dir: Path, recursive: bool = False) -> List[Path]:
    if not test_dir.exists():
        return []
    if recursive:
        return sorted(test_dir.rglob("*.ply"))
    return sorted([p for p in test_dir.iterdir() if p.is_file() and p.suffix.lower() == ".ply"])


def _to_numpy(x: Any) -> Optional[np.ndarray]:
    if x is None:
        return None
    if isinstance(x, np.ndarray):
        return x
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return None


def _standardize(P: np.ndarray) -> np.ndarray:
    P = np.asarray(P, dtype=np.float64)
    c = P.mean(axis=0, keepdims=True)
    Q = P - c
    r = np.linalg.norm(Q, axis=1).max() + 1e-12
    return (Q / r).astype(np.float64)


# ============================================================
# Model builder
# ============================================================
def build_model_from_cfg(cfg: dict) -> Tuple[torch.nn.Module, str]:
    m = cfg.get("model", {}) or {}
    model_path = (
        m.get("import")
        or m.get("name")
        or m.get("target")
        or m.get("impl")
        or m.get("path")
        or m.get("model")
        or "models.meshsegnet_repo_wrapper:MeshSegNetBatch"
    )
    ModelCls = import_symbol(str(model_path))
    kwargs = (m.get("kwargs") or {})
    model = ModelCls(**kwargs)
    return model, str(model_path)


# ============================================================
# Checkpoint loading (best effort)
# ============================================================
def _looks_like_state_dict(obj: Any) -> bool:
    if not isinstance(obj, dict) or len(obj) == 0:
        return False
    ok = 0
    for k, v in obj.items():
        if not isinstance(k, str):
            return False
        if torch.is_tensor(v):
            ok += 1
        else:
            return False
        if ok >= 5:
            break
    return ok > 0


def _iter_state_dict_candidates(ckpt: Any) -> List[Dict[str, Any]]:
    cands: List[Dict[str, Any]] = []
    if _looks_like_state_dict(ckpt):
        return [ckpt]

    if isinstance(ckpt, dict):
        keys = [
            "state_dict",
            "model_state_dict",
            "model_state",
            "model",
            "net",
            "network",
            "weights",
            "params",
        ]
        for k in keys:
            v = ckpt.get(k, None)
            if v is None:
                continue
            if _looks_like_state_dict(v):
                cands.append(v)
                continue
            if isinstance(v, dict):
                v2 = v.get("state_dict", None)
                if _looks_like_state_dict(v2):
                    cands.append(v2)
                    continue

        for _, v in ckpt.items():
            if _looks_like_state_dict(v):
                cands.append(v)
            elif isinstance(v, dict):
                v2 = v.get("state_dict", None)
                if _looks_like_state_dict(v2):
                    cands.append(v2)

    if len(cands) > 1:
        seen = set()
        uniq = []
        for sd in cands:
            sig = (len(sd), next(iter(sd.keys())))
            if sig in seen:
                continue
            seen.add(sig)
            uniq.append(sd)
        cands = uniq
    return cands


def _strip_prefix(sd: Dict[str, Any], prefix: str) -> Dict[str, Any]:
    if not prefix:
        return sd
    out = {}
    for k, v in sd.items():
        if k.startswith(prefix):
            out[k[len(prefix):]] = v
        else:
            out[k] = v
    return out


def _auto_strip_single_token_prefix(sd: Dict[str, Any]) -> List[Tuple[str, Dict[str, Any]]]:
    keys = list(sd.keys())
    if not keys:
        return []
    toks = []
    for k in keys[:200]:
        if "." not in k:
            return []
        toks.append(k.split(".", 1)[0])
    if len(set(toks)) != 1:
        return []
    tok = toks[0]
    pref = tok + "."
    stripped = {}
    for k, v in sd.items():
        if k.startswith(pref):
            stripped[k[len(pref):]] = v
        else:
            stripped[k] = v
    return [(pref, stripped)]


def _match_score(sd: Dict[str, Any], model_keys: set[str]) -> int:
    return sum(1 for k in sd.keys() if k in model_keys)


def load_checkpoint_to_model_best_effort(
    model: torch.nn.Module, ckpt_path: Path
) -> Tuple[List[str], List[str], int, List[str]]:
    try:
        ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=True)
    except TypeError:
        ckpt = torch.load(str(ckpt_path), map_location="cpu")
    except Exception:
        ckpt = torch.load(str(ckpt_path), map_location="cpu")

    cands = _iter_state_dict_candidates(ckpt)
    if not cands:
        if isinstance(ckpt, dict):
            raise RuntimeError(f"Unknown checkpoint format (cannot find state_dict). Top-level keys={list(ckpt.keys())}")
        raise RuntimeError(f"Unknown checkpoint format (type={type(ckpt)})")

    model_keys = set(model.state_dict().keys())

    prefixes = [
        "",
        "module.",
        "model.",
        "net.",
        "network.",
        "core.",
        "module.core.",
        "model.core.",
    ]

    best_sd = None
    best_score = -1
    used_prefixes: List[str] = []

    for sd0 in cands:
        for p in prefixes:
            sd = _strip_prefix(sd0, p)
            score = _match_score(sd, model_keys)
            if score > best_score:
                best_score = score
                best_sd = sd
                used_prefixes = [p] if p else []
        for p, sd in _auto_strip_single_token_prefix(sd0):
            score = _match_score(sd, model_keys)
            if score > best_score:
                best_score = score
                best_sd = sd
                used_prefixes = [p]

    if best_sd is None:
        raise RuntimeError("No usable state_dict candidates")

    ret = model.load_state_dict(best_sd, strict=False)
    missing = list(ret.missing_keys) if hasattr(ret, "missing_keys") else []
    unexpected = list(ret.unexpected_keys) if hasattr(ret, "unexpected_keys") else []
    return missing, unexpected, int(best_score), used_prefixes


# ============================================================
# Flexible call by signature (dataset/collate)
# ============================================================
def call_by_signature(fn, *args, **kwargs):
    sig = inspect.signature(fn)
    filtered = {}
    for k, v in kwargs.items():
        if k in sig.parameters:
            filtered[k] = v
    return fn(*args, **filtered)


def call_build_dataset_flexible(cfg_full: dict, files: List[str], arch: str, mode: str):
    sig = inspect.signature(build_dataset)
    params = list(sig.parameters.values())

    if len(params) >= 2 and params[0].name in ("files", "file_list", "paths"):
        return call_by_signature(build_dataset, files, cfg_full, arch=arch, mode=mode)

    return call_by_signature(build_dataset, cfg_full, files=files, arch=arch, mode=mode)


def call_build_collate_flexible(mode: str):
    mode = str(mode).lower()
    if mode == "point":
        return collate_point
    if mode == "graph":
        return collate_graph
    return collate_face


# ============================================================
# Alignment utilities (labels/conf -> per-mesh-face)
# ============================================================
def align_array_to_mesh(arr: np.ndarray, mesh: trimesh.Trimesh, batch: Any) -> Tuple[np.ndarray, Dict[str, Any]]:
    arr = np.asarray(arr).reshape(-1)
    F = int(mesh.faces.shape[0])
    info = {"method": "raw", "unique_faces": int(len(arr)), "unassigned": 0, "k": 1}

    if ALIGN_MODE == "raw":
        if len(arr) != F:
            out = np.zeros((F,), dtype=arr.dtype)
            n = min(F, len(arr))
            out[:n] = arr[:n]
            info["method"] = "raw_crop_pad"
            return out, info
        return arr, info

    if ALIGN_MODE in ("auto", "centroid"):
        pos = None
        if isinstance(batch, dict) and "pos" in batch:
            pos = _to_numpy(batch["pos"])

        if pos is None:
            if len(arr) != F:
                out = np.zeros((F,), dtype=arr.dtype)
                n = min(F, len(arr))
                out[:n] = arr[:n]
                info["method"] = "centroid_fallback_crop_pad"
                return out, info
            info["method"] = "centroid_fallback_identity"
            return arr, info

        pos = np.asarray(pos)
        if pos.ndim == 3 and pos.shape[0] == 1:
            pos = pos[0]
        if pos.ndim != 2 or pos.shape[1] != 3:
            if len(arr) != F:
                out = np.zeros((F,), dtype=arr.dtype)
                n = min(F, len(arr))
                out[:n] = arr[:n]
                info["method"] = "centroid_badpos_crop_pad"
                return out, info
            info["method"] = "centroid_badpos_identity"
            return arr, info

        P = _standardize(pos)
        C = _standardize(np.asarray(mesh.triangles_center, dtype=np.float64))

        if _cKDTree is None:
            if len(arr) != F:
                out = np.zeros((F,), dtype=arr.dtype)
                n = min(F, len(arr))
                out[:n] = arr[:n]
                info["method"] = "no_kdtree_crop_pad"
                return out, info
            info["method"] = "no_kdtree_identity"
            return arr, info

        tree = _cKDTree(P)
        _, nn = tree.query(C, k=1, workers=-1)
        nn = nn.astype(np.int64)

        if len(arr) == len(P):
            out = arr[nn]
        else:
            out = np.zeros((F,), dtype=arr.dtype)
            n = min(F, len(arr))
            out[:n] = arr[:n]

        info["method"] = "centroid"
        info["unique_faces"] = int(len(out))
        return out, info

    if ALIGN_MODE == "lexsort":
        C = np.asarray(mesh.triangles_center, dtype=np.float64)
        order = np.lexsort((C[:, 2], C[:, 1], C[:, 0]))
        out = np.zeros((F,), dtype=arr.dtype)
        n = min(F, len(arr))
        out[order[:n]] = arr[:n]
        info["method"] = "lexsort"
        info["unique_faces"] = int(n)
        return out, info

    if len(arr) != F:
        out = np.zeros((F,), dtype=arr.dtype)
        n = min(F, len(arr))
        out[:n] = arr[:n]
        info["method"] = "fallback_crop_pad"
        return out, info
    return arr, info


# ============================================================
# Smoothing + Policy C (neighbor fill)
# ============================================================
def _build_face_neighbors(mesh: trimesh.Trimesh) -> List[List[int]]:
    adj = getattr(mesh, "face_adjacency", None)
    F = int(mesh.faces.shape[0])
    neigh: List[List[int]] = [[] for _ in range(F)]
    if adj is None or len(adj) == 0:
        return neigh
    for a, b in adj:
        a = int(a)
        b = int(b)
        neigh[a].append(b)
        neigh[b].append(a)
    return neigh


def smooth_labels_knn(centers: np.ndarray, labels: np.ndarray, k: int = 16, iters: int = 1) -> np.ndarray:
    if iters <= 0:
        return labels
    if _cKDTree is None:
        return labels

    X = _standardize(centers)
    tree = _cKDTree(X)
    lbl = labels.astype(np.int64).copy()
    for _ in range(iters):
        _, idx = tree.query(X, k=int(k), workers=-1)
        new = lbl.copy()
        for i in range(len(lbl)):
            neigh = lbl[idx[i]]
            u, c = np.unique(neigh, return_counts=True)
            new[i] = int(u[np.argmax(c)])
        lbl = new
    return lbl


def smooth_labels_adjacency(mesh: trimesh.Trimesh, labels: np.ndarray, iters: int = 1) -> np.ndarray:
    if iters <= 0:
        return labels

    neigh = _build_face_neighbors(mesh)
    lbl = labels.astype(np.int64).copy()
    F = len(lbl)

    for _ in range(iters):
        new = lbl.copy()
        for i in range(F):
            ns = neigh[i]
            if not ns:
                continue
            vals = lbl[ns]
            u, c = np.unique(vals, return_counts=True)
            new[i] = int(u[np.argmax(c)])
        lbl = new
    return lbl


def fill_lowconf_by_neighbors(
    mesh: trimesh.Trimesh,
    labels: np.ndarray,
    conf: np.ndarray,
    thresh: float,
    iters: int = 1,
) -> np.ndarray:
    if thresh <= 0:
        return labels

    neigh = _build_face_neighbors(mesh)
    labels = labels.astype(np.int64).copy()
    conf = np.asarray(conf, dtype=np.float32).reshape(-1)

    low = conf < float(thresh)
    if not np.any(low):
        return labels

    for _ in range(int(iters)):
        changed = 0
        new = labels.copy()
        for i in np.where(low)[0]:
            ns = neigh[i]
            if not ns:
                continue
            ns_good = [j for j in ns if not low[j]]
            src = ns_good if ns_good else ns
            vals = labels[src]
            u, c = np.unique(vals, return_counts=True)
            new_label = int(u[np.argmax(c)])
            if new_label != labels[i]:
                new[i] = new_label
                changed += 1
        labels = new
        if changed == 0:
            break
    return labels


# ============================================================
# Component cleanup: remove tiny islands
# ============================================================
def cleanup_small_components(
    mesh: trimesh.Trimesh,
    labels: np.ndarray,
    min_size: int = 20,
    iters: int = 1,
) -> np.ndarray:
    """
    Remove small connected components (by face adjacency).
    For each small component, reassign its faces to the majority label
    among its boundary neighbors (neighbors outside the component).
    """
    if min_size <= 0 or iters <= 0:
        return labels

    neigh = _build_face_neighbors(mesh)
    labels = labels.astype(np.int64).copy()
    F = len(labels)

    for _ in range(int(iters)):
        visited = np.zeros((F,), dtype=np.uint8)
        changed = 0

        for start in range(F):
            if visited[start]:
                continue

            lb = int(labels[start])
            q = deque([start])
            visited[start] = 1
            comp = [start]

            while q:
                u = q.popleft()
                for v in neigh[u]:
                    if visited[v]:
                        continue
                    if int(labels[v]) != lb:
                        continue
                    visited[v] = 1
                    q.append(v)
                    comp.append(v)

            if len(comp) >= int(min_size):
                continue

            boundary_labels = []
            comp_set = set(comp)
            for u in comp:
                for v in neigh[u]:
                    if v in comp_set:
                        continue
                    boundary_labels.append(int(labels[v]))

            if not boundary_labels:
                continue

            uvals, cnts = np.unique(np.array(boundary_labels, dtype=np.int64), return_counts=True)
            new_lb = int(uvals[np.argmax(cnts)])

            if new_lb != lb:
                labels[comp] = new_lb
                changed += len(comp)

        if changed == 0:
            break

    return labels


# ============================================================
# Point mode: point pred/conf -> face pred/conf
# ============================================================
def _find_point_xyz_in_batch(batch: Any, n_pred: int) -> Optional[np.ndarray]:
    if not isinstance(batch, dict):
        return None

    for k in ("pos", "xyz", "points", "coords"):
        if k in batch:
            a = _to_numpy(batch[k])
            if a is None:
                continue
            a = np.asarray(a)
            if a.ndim == 3 and a.shape[0] == 1:
                a = a[0]
            if a.ndim == 2 and a.shape[1] == 3 and a.shape[0] == n_pred:
                return a.astype(np.float64)

    if "x" in batch:
        x = _to_numpy(batch["x"])
        if x is None:
            return None
        x = np.asarray(x)
        if x.ndim == 3 and x.shape[0] == 1:
            x = x[0]

        if x.ndim == 2:
            if x.shape[0] == n_pred and x.shape[1] >= 3:
                return x[:, :3].astype(np.float64)
            if x.shape[1] == n_pred and x.shape[0] >= 3:
                return x[:3, :].T.astype(np.float64)

    return None


def point_array_to_face_array(arr_point: np.ndarray, mesh: trimesh.Trimesh, batch: Any) -> Tuple[np.ndarray, Dict[str, Any]]:
    arr_point = np.asarray(arr_point).reshape(-1)
    n_pred = len(arr_point)

    pts = _find_point_xyz_in_batch(batch, n_pred=n_pred)
    if pts is None:
        raise RuntimeError("Point mode: cannot find point xyz in batch to map point->face.")

    fc = np.asarray(mesh.triangles_center, dtype=np.float64)
    P = _standardize(pts)
    C = _standardize(fc)

    if _cKDTree is None:
        raise RuntimeError("Need scipy for KDTree mapping point->face.")

    tree = _cKDTree(P)
    _, nn = tree.query(C, k=1, workers=-1)
    nn = nn.astype(np.int64)

    face_arr = arr_point[nn]
    info = {"method": "point_to_face_nn", "n_points": int(n_pred), "faces": int(len(fc))}
    return face_arr, info


# ============================================================
# Logits -> labels + confidence
# ============================================================
def logits_to_labels_and_conf(out: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
    if out.ndim != 3:
        raise ValueError(f"model output must be 3D, got {tuple(out.shape)}")
    if out.shape[1] <= 64 and out.shape[2] > 64:
        out = out.permute(0, 2, 1).contiguous()

    prob = torch.softmax(out, dim=-1)
    conf, y = torch.max(prob, dim=-1)
    return (
        y[0].detach().cpu().numpy().astype(np.int64),
        conf[0].detach().cpu().numpy().astype(np.float32),
    )


# ============================================================
# Export helpers (PLY with face colors)
# ============================================================
def labels_to_face_rgb(arch: str, labels_face: np.ndarray, num_classes: int = 17) -> np.ndarray:
    cmap = FDIColorMap()
    rgb = np.zeros((len(labels_face), 3), dtype=np.uint8)
    for i, lb in enumerate(labels_face.astype(int).tolist()):
        rgb[i] = np.array(cmap.label_to_rgb(arch, lb, num_classes=num_classes), dtype=np.uint8)
    return rgb


def export_face_color_ply(mesh: trimesh.Trimesh, out_path: Path, face_rgb: np.ndarray):
    m = mesh.copy()
    m.visual.face_colors = np.concatenate(
        [face_rgb, 255 * np.ones((len(face_rgb), 1), dtype=np.uint8)],
        axis=1,
    )
    ensure_dir(out_path.parent)
    m.export(out_path)


def export_json(out_path: Path, obj: dict):
    ensure_dir(out_path.parent)
    out_path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


# ============================================================
# GT export (raw + palette copy)
# ============================================================
def export_gt_files(in_ply: Path, out_dir: Path, arch: str):
    ensure_dir(out_dir)

    gt_raw = out_dir / f"{in_ply.stem}_gt_raw.ply"
    if not gt_raw.exists():
        shutil.copyfile(in_ply, gt_raw)

    if not EXPORT_GT_PALETTE_COPY:
        return

    gt_pal = out_dir / f"{in_ply.stem}_gt_palette.ply"
    if gt_pal.exists():
        return

    mesh = trimesh.load(str(in_ply), force="mesh", process=False)
    if not hasattr(mesh.visual, "face_colors") or mesh.visual.face_colors is None:
        shutil.copyfile(in_ply, gt_pal)
        return

    face_rgb = np.asarray(mesh.visual.face_colors[:, :3], dtype=np.uint8)
    cmap = FDIColorMap()

    uniq = np.unique(face_rgb.reshape(-1, 3), axis=0)
    uniq_list = [tuple(map(int, u)) for u in uniq.tolist()]

    pal = [cmap.label_to_rgb(arch, i, num_classes=17) for i in range(16)] + [
        cmap.label_to_rgb(arch, GINGIVA_LABEL, num_classes=17)
    ]
    pal = np.array(pal, dtype=np.int16)

    unknowns = []
    for col in uniq_list:
        if tuple(col) == tuple(GINGIVA_RGB_DEFAULT):
            continue
        try:
            _ = cmap.rgb_to_label(arch, col, num_classes=17, unknown_policy="raise")
        except Exception:
            unknowns.append(col)

    fixed = face_rgb
    if unknowns:
        c = face_rgb.astype(np.int16)
        dist = np.max(np.abs(c[:, None, :] - pal[None, :, :]), axis=2)
        j = np.argmin(dist, axis=1)
        fixed = pal[j].astype(np.uint8)

    export_face_color_ply(mesh, gt_pal, fixed)


# ============================================================
# Forward helpers (robust for point models: PointNet++ / PointCNN)
# ============================================================
def _move_to_device(v: Any, device: str):
    if torch.is_tensor(v):
        return v.to(device, non_blocking=True)
    return v


def _ensure_bnc(x: torch.Tensor) -> torch.Tensor:
    """
    Ensure tensor is shaped (B, N, C) when possible.

    Heuristic:
      - if already looks like (B,N,C) with small C, keep it
      - if looks like (B,C,N) with small C and large N, transpose
    """
    if not torch.is_tensor(x) or x.ndim != 3:
        return x

    b, a, c = x.shape

    # already (B,N,C)
    if c <= 32:
        return x

    # likely (B,C,N)
    if a <= 32 and c > 32:
        return x.permute(0, 2, 1).contiguous()

    return x


def _build_point_batch_for_model(batch: Any, device: str) -> Dict[str, Any]:
    """
    Build a dict for point-based models robustly.

    Supported common keys:
      - pos / xyz / points / coords : point positions, shape (B,N,3)
      - x                           : full point feature tensor, e.g. (B,N,6) for xyz+normals
      - features / feats            : alternative full feature tensor

    IMPORTANT:
      For PointNet++ / PointCNN in this project, keep x as the FULL feature tensor
      (e.g. xyz+normals => 6 channels), not just the channels after xyz.
    """
    if not isinstance(batch, dict):
        raise TypeError(f"Point mode expects batch as dict, got {type(batch)}")

    batch_dev = {k: _move_to_device(v, device) for k, v in batch.items()}
    out: Dict[str, Any] = {}

    # -------------------------
    # 1) locate pos
    # -------------------------
    pos = None
    for key in ("pos", "xyz", "points", "coords"):
        if key in batch_dev and torch.is_tensor(batch_dev[key]):
            pos = _ensure_bnc(batch_dev[key])
            break

    # -------------------------
    # 2) locate full feature tensor
    # -------------------------
    x_full = None

    if "features" in batch_dev and torch.is_tensor(batch_dev["features"]):
        x_full = _ensure_bnc(batch_dev["features"])
    elif "feats" in batch_dev and torch.is_tensor(batch_dev["feats"]):
        x_full = _ensure_bnc(batch_dev["feats"])
    elif "x" in batch_dev and torch.is_tensor(batch_dev["x"]):
        x_full = _ensure_bnc(batch_dev["x"])

    # If no explicit pos, derive it from first 3 channels of x_full
    if pos is None:
        if x_full is None:
            raise KeyError('Point batch has neither "pos" nor usable full feature tensor ("x"/"features"/"feats").')
        if x_full.ndim != 3 or x_full.shape[-1] < 3:
            raise RuntimeError(
                f'Cannot derive "pos" from full feature tensor with shape {tuple(x_full.shape)}'
            )
        pos = x_full[..., :3]

    out["pos"] = pos

    # Keep full point feature tensor if available
    if x_full is not None:
        out["x"] = x_full

    # Pass-through keys เผื่อบาง model / downstream ใช้
    for k in ("path", "mask", "valid_mask", "y", "label", "labels"):
        if k in batch_dev and k not in out:
            out[k] = batch_dev[k]

    return out


def forward_point_model(model: torch.nn.Module, batch: Any) -> torch.Tensor:
    """
    Robust forward for point-based models such as PointNet++ and PointCNN.

    Priority:
      1) model({"pos": ..., "x": full_features})
      2) model(original_batch_dict)
      3) model(full_features)
      4) model(full_features permuted)
      5) model(pos, full_features)
      6) model(pos) ONLY if full_features do not exist
    """
    last_err = None

    point_batch = _build_point_batch_for_model(batch, DEVICE)

    # Optional debug
    # print("[DEBUG] point_batch keys:", list(point_batch.keys()))
    # print("[DEBUG] pos shape:", tuple(point_batch["pos"].shape))
    # if "x" in point_batch:
    #     print("[DEBUG] x shape:", tuple(point_batch["x"].shape))

    tries = []

    # 1) preferred: dict with pos + x
    tries.append(lambda: model(point_batch))

    # 2) original batch dict
    if isinstance(batch, dict):
        batch_dev = {k: _move_to_device(v, DEVICE) for k, v in batch.items()}
        tries.append(lambda: model(batch_dev))

    # 3) full feature tensor
    if "x" in point_batch and torch.is_tensor(point_batch["x"]):
        x = point_batch["x"]
        tries.append(lambda: model(x))

        if x.ndim == 3:
            tries.append(lambda: model(x.permute(0, 2, 1).contiguous()))

        # 4) positional style
        tries.append(lambda: model(point_batch["pos"], x))

    # 5) pos-only fallback ONLY when there is no full feature tensor
    if "x" not in point_batch:
        tries.append(lambda: model(point_batch["pos"]))

    for fn in tries:
        try:
            out = fn()
            if torch.is_tensor(out):
                return out
            raise TypeError(f"Model forward returned non-tensor type: {type(out)}")
        except Exception as e:
            last_err = e

    raise RuntimeError(
        f"Point model forward failed for all attempts. Last error: {repr(last_err)}"
    ) from last_err


# ============================================================
# Main
# ============================================================
def main():
    run_dir = RUN_DIR
    cfg_path = run_dir / "config.yaml"

    cfg = load_yaml(cfg_path)
    print(f"[RUN_DIR]  {run_dir}")
    print(f"[CFG]      {cfg_path}")
    print(f"[SPLIT]    {SPLIT_ROOT}")
    print(f"[VIS_OUT]  {VIS_OUT}")
    print(f"[DEVICE]   {DEVICE}")

    mode_run = str((cfg.get("data", {}) or {}).get("mode", MODE)).lower()
    if MODE != "auto":
        mode_run = str(MODE).lower()
    print(f"[MODE]     {mode_run}")

    model, model_spec = build_model_from_cfg(cfg)
    print(f"[MODEL]    {model_spec}")
    model = model.to(DEVICE).eval()

    cfg_full = copy.deepcopy(cfg)
    cfg_full.setdefault("data", {})
    cfg_full["data"]["unknown_policy"] = str(UNKNOWN_POLICY)
    cfg_full["data"]["mode"] = mode_run

    ensure_dir(VIS_OUT)

    datasets = (cfg.get("data", {}) or {}).get("datasets", [])
    if not datasets:
        datasets = ["dataset1", "dataset2", "dataset3", "dataset4", "dataset5"]

    for dataset in datasets:
        dataset = str(dataset)
        test_fold = get_test_fold_for_dataset(cfg, dataset)
        print(f"\n==================== {dataset} | TEST={test_fold} ====================")

        for arch in ARCHES:
            test_dir = SPLIT_ROOT / dataset / test_fold
            if not test_dir.exists():
                print(f"[SKIP] test fold dir missing: {test_dir}")
                continue

            files = list_ply_files(test_dir, recursive=bool(cfg_full["data"].get("recursive", False)))
            files = [p for p in files if arch_from_name(p) == arch]
            print(f"[FOUND] {dataset}/{test_fold}/{arch}: {len(files)} files")

            if not files:
                continue

            ckpt_dir = run_dir / dataset / arch
            ckpt_file = "best_model.pth" if CKPT_TAG == "best" else "last_model.pth"
            ckpt_path = ckpt_dir / ckpt_file
            if not ckpt_path.exists():
                print(f"[SKIP] missing ckpt: {ckpt_path}")
                continue

            print(f"        ckpt={ckpt_file}")

            missing, unexpected, score, prefixes = load_checkpoint_to_model_best_effort(model, ckpt_path)
            print(f"[CKPT] best_match_score={score} | stripped_prefixes={prefixes} | candidates=1")
            if missing:
                print(f"[WARN] missing keys: {len(missing)}")
            if unexpected:
                print(f"[WARN] unexpected keys: {len(unexpected)}")

            ds = call_build_dataset_flexible(cfg_full, files=[str(p) for p in files], arch=arch, mode=mode_run)
            collate_fn = call_build_collate_flexible(mode_run)

            dl = DataLoader(
                ds,
                batch_size=1,
                shuffle=False,
                num_workers=0,
                pin_memory=False,
                drop_last=False,
                collate_fn=collate_fn,
            )

            out_root = VIS_OUT / dataset / test_fold / arch / f"{CKPT_TAG}_viz"
            ensure_dir(out_root)

            for i, batch in enumerate(dl):
                if DEBUG_BATCH_KEYS and isinstance(batch, dict):
                    print("[BATCH KEYS]", list(batch.keys()))

                path_str = batch["path"][0] if isinstance(batch, dict) and "path" in batch else None
                in_ply = Path(path_str) if path_str else Path(files[i])

                mesh = trimesh.load(str(in_ply), force="mesh", process=False)

                # forward
                if mode_run in ("face", "graph"):
                    batch_t = {}
                    for k, v in batch.items():
                        if torch.is_tensor(v):
                            batch_t[k] = v.to(DEVICE, non_blocking=True)
                        else:
                            batch_t[k] = v
                    out = model(batch_t)
                else:
                    out = forward_point_model(model, batch)

                if DEBUG_PRINT_FIRST_OUT and i == 0:
                    print(f"[DEBUG] model out tensor: {tuple(out.shape)} {out.dtype}")

                pred, conf = logits_to_labels_and_conf(out)

                # Map/align to per-face arrays
                if mode_run == "point":
                    pred_face, alst = point_array_to_face_array(pred, mesh, batch)
                    conf_face, _ = point_array_to_face_array(conf, mesh, batch)
                else:
                    pred_face, alst = align_array_to_mesh(pred, mesh, batch)
                    conf_face, _ = align_array_to_mesh(conf, mesh, batch)

                pred_face = np.asarray(pred_face, dtype=np.int64).reshape(-1)
                conf_face = np.asarray(conf_face, dtype=np.float32).reshape(-1)

                # 1) Low-confidence policy (C)
                if CONF_THRESH and float(CONF_THRESH) > 0 and CONF_APPLY_BEFORE_CLEAN:
                    pol = str(CONF_LOWCONF_POLICY).lower()
                    if pol == "neighbor":
                        pred_face = fill_lowconf_by_neighbors(
                            mesh, labels=pred_face, conf=conf_face,
                            thresh=float(CONF_THRESH), iters=int(CONF_NEIGHBOR_ITERS)
                        )
                    elif pol == "gingiva":
                        low = conf_face < float(CONF_THRESH)
                        if np.any(low):
                            pred_face = pred_face.copy()
                            pred_face[low] = int(GINGIVA_LABEL)

                # 2) Component cleanup (remove tiny islands)
                if CLEAN_SMALL_COMPONENTS:
                    pred_face = cleanup_small_components(
                        mesh, labels=pred_face, min_size=int(MIN_COMP_SIZE), iters=int(CLEAN_ITERS)
                    )

                # 3) Smoothing (optional)
                if SMOOTH_ITERS and int(SMOOTH_ITERS) > 0:
                    if str(SMOOTH_METHOD).lower() == "knn":
                        centers = np.asarray(mesh.triangles_center, dtype=np.float64)
                        pred_sm = smooth_labels_knn(
                            centers, pred_face, k=int(SMOOTH_KNN_K), iters=int(SMOOTH_ITERS)
                        )
                    else:
                        pred_sm = smooth_labels_adjacency(mesh, pred_face, iters=int(SMOOTH_ITERS))
                else:
                    pred_sm = pred_face

                # optional: apply conf policy after smoothing if configured
                if CONF_THRESH and float(CONF_THRESH) > 0 and (not CONF_APPLY_BEFORE_SMOOTH):
                    pol = str(CONF_LOWCONF_POLICY).lower()
                    if pol == "neighbor":
                        pred_sm = fill_lowconf_by_neighbors(
                            mesh, labels=pred_sm, conf=conf_face,
                            thresh=float(CONF_THRESH), iters=int(CONF_NEIGHBOR_ITERS)
                        )
                    elif pol == "gingiva":
                        low = conf_face < float(CONF_THRESH)
                        if np.any(low):
                            pred_sm = pred_sm.copy()
                            pred_sm[low] = int(GINGIVA_LABEL)

                u, c = np.unique(pred_sm, return_counts=True)
                hist = sorted([(int(a), int(b)) for a, b in zip(u.tolist(), c.tolist())], key=lambda x: -x[1])
                if i == 0:
                    print(
                        f"[DEBUG] align={alst} | hist_top={hist[:5]} | "
                        f"conf={CONF_LOWCONF_POLICY}:{CONF_THRESH} it={CONF_NEIGHBOR_ITERS} "
                        f"clean=min{MIN_COMP_SIZE}x{CLEAN_ITERS} smooth={SMOOTH_METHOD}:{SMOOTH_ITERS}"
                    )

                case_id = case_id_from_path(in_ply)
                case_dir = (out_root / f"{case_id}") if CASE_SUBFOLDERS else out_root
                ensure_dir(case_dir)

                if EXPORT_GT:
                    export_gt_files(in_ply, case_dir, arch)

                num_classes = int((cfg.get("model", {}) or {}).get("kwargs", {}).get("num_classes", 17))
                rgb_pred = labels_to_face_rgb(arch, pred_sm, num_classes=num_classes)
                out_pred_ply = case_dir / f"{in_ply.stem}_pred_{CKPT_TAG}.ply"
                export_face_color_ply(mesh, out_pred_ply, rgb_pred)

                out_json = case_dir / f"{in_ply.stem}_pred_{CKPT_TAG}.json"
                export_json(
                    out_json,
                    {
                        "in_ply": str(in_ply),
                        "dataset": dataset,
                        "fold": test_fold,
                        "arch": arch,
                        "mode": mode_run,
                        "ckpt": str(ckpt_path),
                        "align": alst,
                        "hist": hist[:50],
                        "conf_thresh": float(CONF_THRESH),
                        "conf_policy": str(CONF_LOWCONF_POLICY),
                        "conf_neighbor_iters": int(CONF_NEIGHBOR_ITERS),
                        "clean_small_components": bool(CLEAN_SMALL_COMPONENTS),
                        "min_comp_size": int(MIN_COMP_SIZE),
                        "clean_iters": int(CLEAN_ITERS),
                        "smooth_iters": int(SMOOTH_ITERS),
                        "smooth_method": str(SMOOTH_METHOD),
                        "smooth_knn_k": int(SMOOTH_KNN_K),
                    },
                )

                if (i + 1) % 10 == 0:
                    print(f"  [OK] {arch}: {i+1}/{len(files)}")

            print(f"[DONE] Exported -> {out_root}")

    print("\nALL DONE.")


if __name__ == "__main__":
    main()