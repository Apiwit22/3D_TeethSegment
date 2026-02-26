# visualize.py
# ------------------------------------------------------------
# Visualizer (Windows local) + GT export
# Supports:
#   - Face-mode models (e.g., MeshSegNetBatch): predict per-face
#   - Point-mode models (e.g., PointNet2Seg):  predict per-point then map -> per-face for export
#   - Graph-mode models: treated like face-mode for export
#
# Key fixes:
#   - Read model class from cfg['model']['import'] (PointNet++ uses this)
#   - Use cfg['data']['mode'] automatically when MODE="auto"
#   - call_by_signature supports positional args
#   - build_dataset signature-robust: supports build_dataset(files, cfg, ...) OR build_dataset(cfg, files=...)
#   - Point-mode: map point labels -> face labels (KDTree)
# ------------------------------------------------------------

from __future__ import annotations

import copy
import inspect
import json
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

# ============================================================
# CONFIG (แก้แค่ตรงนี้)
# ============================================================
RUN_DIR = Path(r"D:\Project_Gujabaa\3D_Project\checkpoints\pointnetpp\PointNet2Seg_IoSSeg_ALLDATA_200epoch_run7")
SPLIT_ROOT = Path(r"D:\Project_Gujabaa\3D_Project\All_dataset_split_face")
VIS_OUT = Path(r"D:\Project_Gujabaa\3D_Project\vis_results") / RUN_DIR.name

CKPT_TAG = "best"  # "best" or "last"
ARCHES = ["upper", "lower"]

# "auto" => use cfg['data']['mode'] (recommended)
# or force "face" / "point" / "graph"
MODE = "auto"

# base.py ของคุณรับ unknown_policy แค่ raise/ignore
UNKNOWN_POLICY = "ignore"

# alignment mode: "auto" / "raw" / "centroid" / "lexsort"
ALIGN_MODE = "auto"

# Export GT
EXPORT_GT = True
EXPORT_GT_PALETTE_COPY = True

# pred smoothing หลัง align (0 ปิด)
SMOOTH_ITERS = 1

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
from src.dataloader import build_dataset
from src.dataloader.collate import collate_face, collate_point, collate_graph
from src.dataloader.fdi_colors import FDIColorMap, GINGIVA_LABEL

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
    """
    Prefer cfg['data']['test_fold_by_dataset'] if exists, else fallback mapping.
    """
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
    """
    Standardize points for stable nearest mapping: center and scale by max radius.
    """
    P = np.asarray(P, dtype=np.float64)
    c = P.mean(axis=0, keepdims=True)
    Q = P - c
    r = np.linalg.norm(Q, axis=1).max() + 1e-12
    return (Q / r).astype(np.float64)


# ============================================================
# Model builder (FIX: support cfg['model']['import'])
# ============================================================
def build_model_from_cfg(cfg: dict) -> Tuple[torch.nn.Module, str]:
    m = cfg.get("model", {}) or {}
    model_path = (
        m.get("import")  # ✅ key used by pointnetpp configs
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
            "model_state",  # <- your trainer uses this
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

    # de-dup
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


def load_checkpoint_to_model_best_effort(model: torch.nn.Module, ckpt_path: Path) -> Tuple[List[str], List[str], int, List[str]]:
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
    """
    Call fn with positional args + only keyword args that exist in signature.
    """
    sig = inspect.signature(fn)
    filtered = {}
    for k, v in kwargs.items():
        if k in sig.parameters:
            filtered[k] = v
    return fn(*args, **filtered)


def call_build_dataset_flexible(cfg_full: dict, files: List[str], arch: str, mode: str):
    """
    Robustly call src.dataloader.build_dataset no matter whether signature is:
      - build_dataset(files, cfg, arch=?, mode=?)
      - build_dataset(cfg, files=?, arch=?, mode=?)
    """
    sig = inspect.signature(build_dataset)
    params = list(sig.parameters.values())

    # Case A: first positional param is named 'files' (your project)
    if len(params) >= 2 and params[0].name in ("files", "file_list", "paths"):
        return call_by_signature(build_dataset, files, cfg_full, arch=arch, mode=mode)

    # Case B: first positional param looks like cfg
    return call_by_signature(build_dataset, cfg_full, files=files, arch=arch, mode=mode)


def call_build_collate_flexible(mode: str):
    mode = str(mode).lower()
    if mode == "point":
        return collate_point
    if mode == "graph":
        return collate_graph
    return collate_face


# ============================================================
# Pred alignment/smoothing (face-based)
# ============================================================
def align_pred_to_mesh(pred: np.ndarray, mesh: trimesh.Trimesh, batch: Any) -> Tuple[np.ndarray, Dict[str, Any]]:
    pred = pred.reshape(-1).astype(np.int64)
    F = int(mesh.faces.shape[0])
    info = {"method": "raw", "unique_faces": int(len(pred)), "unassigned": 0, "k": 10}

    if ALIGN_MODE == "raw":
        if len(pred) != F:
            out = np.full((F,), int(GINGIVA_LABEL), dtype=np.int64)
            n = min(F, len(pred))
            out[:n] = pred[:n]
            info["method"] = "raw_crop_pad"
            return out, info
        return pred, info

    if ALIGN_MODE in ("auto", "centroid"):
        pos = None
        if isinstance(batch, dict) and "pos" in batch:
            pos = _to_numpy(batch["pos"])

        if pos is None:
            if len(pred) != F:
                out = np.full((F,), int(GINGIVA_LABEL), dtype=np.int64)
                n = min(F, len(pred))
                out[:n] = pred[:n]
                info["method"] = "centroid_fallback_crop_pad"
                return out, info
            info["method"] = "centroid_fallback_identity"
            return pred, info

        pos = np.asarray(pos)
        if pos.ndim == 3 and pos.shape[0] == 1:
            pos = pos[0]
        if pos.ndim != 2 or pos.shape[1] != 3:
            if len(pred) != F:
                out = np.full((F,), int(GINGIVA_LABEL), dtype=np.int64)
                n = min(F, len(pred))
                out[:n] = pred[:n]
                info["method"] = "centroid_badpos_crop_pad"
                return out, info
            info["method"] = "centroid_badpos_identity"
            return pred, info

        P = _standardize(pos)
        C = _standardize(np.asarray(mesh.triangles_center, dtype=np.float64))

        if _cKDTree is None:
            if len(pred) != F:
                out = np.full((F,), int(GINGIVA_LABEL), dtype=np.int64)
                n = min(F, len(pred))
                out[:n] = pred[:n]
                info["method"] = "no_kdtree_crop_pad"
                return out, info
            info["method"] = "no_kdtree_identity"
            return pred, info

        tree = _cKDTree(P)
        _, nn = tree.query(C, k=1, workers=-1)
        nn = nn.astype(np.int64)

        out = pred[nn] if len(pred) == len(P) else pred[: len(nn)]
        info["method"] = "centroid"
        info["unique_faces"] = int(len(out))
        return out.astype(np.int64), info

    if ALIGN_MODE == "lexsort":
        C = np.asarray(mesh.triangles_center, dtype=np.float64)
        order = np.lexsort((C[:, 2], C[:, 1], C[:, 0]))
        out = np.full((F,), int(GINGIVA_LABEL), dtype=np.int64)
        n = min(F, len(pred))
        out[order[:n]] = pred[:n]
        info["method"] = "lexsort"
        info["unique_faces"] = int(n)
        return out, info

    if len(pred) != F:
        out = np.full((F,), int(GINGIVA_LABEL), dtype=np.int64)
        n = min(F, len(pred))
        out[:n] = pred[:n]
        info["method"] = "fallback_crop_pad"
        return out, info
    return pred, info


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


# ============================================================
# Point mode: point pred -> face pred
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


def point_pred_to_face_labels(pred_point: np.ndarray, mesh: trimesh.Trimesh, batch: Any) -> Tuple[np.ndarray, Dict[str, Any]]:
    pred_point = pred_point.reshape(-1).astype(np.int64)
    n_pred = len(pred_point)

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

    face_labels = pred_point[nn]
    info = {"method": "point_to_face_nn", "n_points": int(n_pred), "faces": int(len(fc))}
    return face_labels.astype(np.int64), info


# ============================================================
# Logits to labels (support (B,N,C) or (B,C,N))
# ============================================================
def logits_to_labels(out: torch.Tensor) -> np.ndarray:
    if out.ndim != 3:
        raise ValueError(f"model output must be 3D, got {tuple(out.shape)}")

    # heuristic: if second dim looks like C (<=64) and third looks like N (>>64)
    if out.shape[1] <= 64 and out.shape[2] > 64:
        out = out.permute(0, 2, 1).contiguous()  # (B,N,C)

    y = torch.argmax(out, dim=-1)  # (B,N)
    return y[0].detach().cpu().numpy().astype(np.int64)


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

    # build palette list (16 teeth + gingiva)
    if arch == "upper":
        pal = [cmap.label_to_rgb("upper", i, num_classes=17) for i in range(16)] + [
            cmap.label_to_rgb("upper", GINGIVA_LABEL, num_classes=17)
        ]
    else:
        pal = [cmap.label_to_rgb("lower", i, num_classes=17) for i in range(16)] + [
            cmap.label_to_rgb("lower", GINGIVA_LABEL, num_classes=17)
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
        dist = np.max(np.abs(c[:, None, :] - pal[None, :, :]), axis=2)  # (F,17)
        j = np.argmin(dist, axis=1)
        fixed = pal[j].astype(np.uint8)

    export_face_color_ply(mesh, gt_pal, fixed)


# ============================================================
# Forward helpers (robust for point models)
# ============================================================
def forward_point_model(model: torch.nn.Module, batch: Any) -> torch.Tensor:
    """
    Try common PointNet++ forward signatures.
    Default: model(x)
    Fallback: model(x.permute(0,2,1))
    """
    x = batch["x"] if isinstance(batch, dict) else batch
    if not torch.is_tensor(x):
        raise TypeError("Point mode expects batch['x'] as torch.Tensor")

    x = x.to(DEVICE, non_blocking=True)

    # Try model(x)
    try:
        return model(x)
    except Exception:
        pass

    # Try swapping (B,N,C) <-> (B,C,N)
    if x.ndim == 3:
        try:
            return model(x.permute(0, 2, 1).contiguous())
        except Exception:
            pass

    # Last resort: if model accepts dict
    try:
        return model({"x": x})
    except Exception as e:
        raise RuntimeError(f"Point model forward failed for all attempts. Last error: {repr(e)}") from e


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
    print(f"[MODE]    {mode_run}")

    model, model_spec = build_model_from_cfg(cfg)
    print(f"[MODEL]    {model_spec}")
    model = model.to(DEVICE).eval()

    cfg_full = copy.deepcopy(cfg)
    cfg_full.setdefault("data", {})
    cfg_full["data"]["unknown_policy"] = str(UNKNOWN_POLICY)
    cfg_full["data"]["mode"] = mode_run  # ✅ do NOT hardcode face

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

                pred = logits_to_labels(out)

                if mode_run == "point":
                    pred_aligned, alst = point_pred_to_face_labels(pred, mesh, batch)
                else:
                    pred_aligned, alst = align_pred_to_mesh(pred, mesh, batch)

                if SMOOTH_ITERS > 0:
                    centers = np.asarray(mesh.triangles_center, dtype=np.float64)
                    pred_sm = smooth_labels_knn(centers, pred_aligned, k=16, iters=int(SMOOTH_ITERS))
                else:
                    pred_sm = pred_aligned

                u, c = np.unique(pred_sm, return_counts=True)
                hist = sorted([(int(a), int(b)) for a, b in zip(u.tolist(), c.tolist())], key=lambda x: -x[1])
                if i == 0:
                    print(f"[DEBUG] align={alst} | hist_top={hist[:5]}")

                case_id = case_id_from_path(in_ply)
                case_dir = (out_root / f"{case_id}") if CASE_SUBFOLDERS else out_root
                ensure_dir(case_dir)

                if EXPORT_GT:
                    export_gt_files(in_ply, case_dir, arch)

                rgb_pred = labels_to_face_rgb(
                    arch,
                    pred_sm,
                    num_classes=int((cfg.get("model", {}) or {}).get("kwargs", {}).get("num_classes", 17)),
                )
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
                        "smooth_iters": int(SMOOTH_ITERS),
                    },
                )

                if (i + 1) % 10 == 0:
                    print(f"  [OK] {arch}: {i+1}/{len(files)}")

            print(f"[DONE] Exported -> {out_root}")

    print("\nALL DONE.")


if __name__ == "__main__":
    main()