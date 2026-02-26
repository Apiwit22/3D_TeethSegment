# eval_sum.py
from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader


# ============================================================
# USER PATHS
# ============================================================
RUN_DIR = r"D:\Project_Gujabaa\3D_Project\checkpoints\meshsegnet\MeshSegNetBatch_IoSSeg_ALLDATA_200epoch_run3"
SPLIT_ROOT = r"D:\Project_Gujabaa\3D_Project\All_dataset_split_face"
RESULT_ROOT = r"D:\Project_Gujabaa\3D_Project\results_new\meshsegnet"

# ============================================================
EPS = 1e-8

# project root importable
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dataloader.arch import parse_arch_from_filename
from src.dataloader import build_dataset, build_collate_fn


# ============================================================
# Class name mapping (label -> readable name)
# merged: upper label i + lower label i => "Txx/Tyy"
# ============================================================
FDI_UPPER_16 = [11, 12, 13, 14, 15, 16, 17, 18, 21, 22, 23, 24, 25, 26, 27, 28]
FDI_LOWER_16 = [31, 32, 33, 34, 35, 36, 37, 38, 41, 42, 43, 44, 45, 46, 47, 48]


def class_name_for_arch(label: int, arch: str, num_classes: int) -> str:
    label = int(label)
    arch = str(arch).lower()

    # if you use 17 classes: last one is gingiva
    if int(num_classes) >= 17 and label == 16:
        return "Gingiva"

    if label < 0 or label > 15:
        return f"Unknown({label})"

    if arch == "upper":
        return f"T{FDI_UPPER_16[label]}"
    if arch == "lower":
        return f"T{FDI_LOWER_16[label]}"
    if arch == "merged_upper_lower":
        return f"T{FDI_UPPER_16[label]}/T{FDI_LOWER_16[label]}"
    return f"Label{label}"


# ============================================================
# helpers
# ============================================================
def import_from_path(path: str):
    if ":" not in path:
        raise ValueError(f"model.import must be 'module:ClassName', got: {path}")
    mod_name, obj_name = path.split(":", 1)
    mod = __import__(mod_name, fromlist=[obj_name])
    return getattr(mod, obj_name)


def list_ply(dir_: Path, recursive: bool = False) -> List[str]:
    if not dir_.exists():
        return []
    pat = "**/*.ply" if recursive else "*.ply"
    return sorted([str(p) for p in dir_.glob(pat)])


def split_by_arch(files: List[str]) -> Tuple[List[str], List[str]]:
    upper, lower = [], []
    for f in files:
        a = parse_arch_from_filename(f)
        (upper if a == "upper" else lower).append(f)
    return upper, lower


def normalize_logits_to_bnc(logits: torch.Tensor, num_classes: int) -> torch.Tensor:
    # (B,N,C) expected. Accept (B,C,N) too.
    if logits.dim() != 3:
        raise ValueError(f"Model output must be 3D, got {tuple(logits.shape)}")
    if logits.shape[-1] == num_classes:
        return logits
    if logits.shape[1] == num_classes:
        return logits.permute(0, 2, 1).contiguous()
    raise ValueError(f"Cannot infer logits layout from shape {tuple(logits.shape)}")


def forward_model(model: torch.nn.Module, batch: Dict[str, Any], device: torch.device, forward_type: str) -> torch.Tensor:
    ft = str(forward_type).lower()

    if ft == "batch":
        b2: Dict[str, Any] = {}
        for k, v in batch.items():
            if torch.is_tensor(v):
                b2[k] = v.to(device, non_blocking=True)
            elif isinstance(v, list):
                b2[k] = [t.to(device, non_blocking=True) if torch.is_tensor(t) else t for t in v]
            else:
                b2[k] = v
        out = model(b2)
    else:
        x = batch["x"].to(device, non_blocking=True)
        out = model(x)

    if isinstance(out, (tuple, list)):
        out = out[0]
    if isinstance(out, dict):
        for key in ("logits", "out", "pred"):
            if key in out and torch.is_tensor(out[key]):
                out = out[key]
                break
    if not torch.is_tensor(out):
        raise ValueError(f"Model forward must return Tensor logits, got: {type(out)}")
    return out


def _safe_mean(arr: np.ndarray, mask: np.ndarray) -> float:
    if mask.sum() == 0:
        return 0.0
    return float(np.mean(arr[mask]))


def compute_metrics_from_counts(
    tp: np.ndarray,
    fp: np.ndarray,
    fn: np.ndarray,
    correct: int,
    total_valid: int,
) -> Dict[str, Any]:
    """
    Compact set (non-redundant):
      - OA
      - mPrecision/mRecall/mF1/mIoU/mDice (present-only)
      - bookkeeping: correct/total_valid + sums
    """
    precision = tp / (tp + fp + EPS)
    recall = tp / (tp + fn + EPS)
    f1 = (2 * precision * recall) / (precision + recall + EPS)

    iou = tp / (tp + fp + fn + EPS)
    dice = (2 * tp) / (2 * tp + fp + fn + EPS)

    present = (tp + fn) > 0
    OA = correct / (total_valid + EPS)

    return {
        "OA": float(OA),

        "mPrecision": _safe_mean(precision, present),
        "mRecall": _safe_mean(recall, present),
        "mF1": _safe_mean(f1, present),
        "mIoU": _safe_mean(iou, present),
        "mDice": _safe_mean(dice, present),
        "n_present_classes": int(present.sum()),

        # minimal bookkeeping (keep to allow merged OA)
        "total_valid": int(total_valid),
        "correct": int(correct),
        "tp_sum": int(tp.sum()),
        "fp_sum": int(fp.sum()),
        "fn_sum": int(fn.sum()),
    }


@torch.no_grad()
def eval_arch(
    cfg: dict,
    arch: str,
    files_test_all: List[str],
    ckpt_path: Path,
    device: torch.device,
) -> Dict[str, Any]:
    data = cfg["data"]
    ignore_index = int(data.get("ignore_index", -1))
    batch_size = int(data.get("batch_size", 2))
    num_workers = int(data.get("num_workers", 0))
    pin_memory = bool(data.get("pin_memory", True)) and device.type == "cuda"

    # filter files by arch
    te_u, te_l = split_by_arch(files_test_all)
    files_test = te_u if arch == "upper" else te_l
    if len(files_test) == 0:
        raise RuntimeError(f"No test files for arch={arch}")

    # build dataset/loader
    ds_test = build_dataset(files_test, cfg, arch=arch)
    collate_fn = build_collate_fn(cfg)
    dl_test = DataLoader(
        ds_test,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
        drop_last=False,
    )

    # model (prefer info from ckpt to avoid config mismatch)
    ckpt = torch.load(str(ckpt_path), map_location=device)
    model_import = ckpt.get("model_import", cfg["model"]["import"])
    model_kwargs = ckpt.get("model_kwargs", cfg["model"].get("kwargs", {}))
    forward_type = ckpt.get("config", {}).get("model", {}).get("forward_type", cfg["model"].get("forward_type", "x"))
    num_classes = int(model_kwargs.get("num_classes", cfg["model"].get("kwargs", {}).get("num_classes", 16)))

    ModelCls = import_from_path(model_import)
    model = ModelCls(**model_kwargs).to(device)

    state = ckpt.get("model_state", None) or ckpt.get("state_dict", None) or ckpt
    model.load_state_dict(state, strict=True)
    model.eval()

    # accumulators (TN not needed for our compact metrics)
    tp = np.zeros((num_classes,), dtype=np.int64)
    fp = np.zeros((num_classes,), dtype=np.int64)
    fn = np.zeros((num_classes,), dtype=np.int64)

    total_valid = 0
    correct = 0

    for batch in dl_test:
        y = batch["y"].to(device, non_blocking=True)  # (B,N)
        logits = forward_model(model, batch, device, forward_type)
        logits = normalize_logits_to_bnc(logits, num_classes)  # (B,N,C)
        pred = torch.argmax(logits, dim=-1)  # (B,N)

        mask = (y != ignore_index)
        yv = y[mask]
        pv = pred[mask]

        total_valid += int(yv.numel())
        correct += int((pv == yv).sum().item())

        for c in range(num_classes):
            yc = (yv == c)
            pc = (pv == c)
            tp[c] += int((yc & pc).sum().item())
            fp[c] += int((~yc & pc).sum().item())
            fn[c] += int((yc & ~pc).sum().item())

    overall = compute_metrics_from_counts(tp, fp, fn, correct, total_valid)

    # per-class (compact)
    per_class: Dict[str, Any] = {}
    for c in range(num_classes):
        present = int((tp[c] + fn[c]) > 0)
        prec = float(tp[c] / (tp[c] + fp[c] + EPS))
        rec = float(tp[c] / (tp[c] + fn[c] + EPS))
        f1 = float((2 * prec * rec) / (prec + rec + EPS))
        iou = float(tp[c] / (tp[c] + fp[c] + fn[c] + EPS))
        dice = float((2 * tp[c]) / (2 * tp[c] + fp[c] + fn[c] + EPS))

        per_class[str(c)] = {
            "class_name": class_name_for_arch(c, arch, num_classes),
            "present_in_gt": present,
            "tp": int(tp[c]),
            "fp": int(fp[c]),
            "fn": int(fn[c]),
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "iou": iou,
            "dice": dice,
        }

    return {
        "arch": arch,
        "model": model_import,
        "ckpt": str(ckpt_path),
        "overall": overall,
        "per_class": per_class,
    }


def merge_upper_lower(upper: Dict[str, Any], lower: Dict[str, Any]) -> Dict[str, Any]:
    per_u = upper["per_class"]
    per_l = lower["per_class"]

    classes = sorted(set(per_u.keys()) | set(per_l.keys()), key=lambda x: int(x))

    tp = np.array([per_u.get(c, {}).get("tp", 0) + per_l.get(c, {}).get("tp", 0) for c in classes], dtype=np.int64)
    fp = np.array([per_u.get(c, {}).get("fp", 0) + per_l.get(c, {}).get("fp", 0) for c in classes], dtype=np.int64)
    fn = np.array([per_u.get(c, {}).get("fn", 0) + per_l.get(c, {}).get("fn", 0) for c in classes], dtype=np.int64)

    # ✅ merged OA is recoverable from correct/total_valid of upper+lower
    correct_u = int(upper["overall"].get("correct", 0))
    correct_l = int(lower["overall"].get("correct", 0))
    total_u = int(upper["overall"].get("total_valid", 0))
    total_l = int(lower["overall"].get("total_valid", 0))

    correct_m = correct_u + correct_l
    total_m = total_u + total_l
    OA_m = float(correct_m / (total_m + EPS)) if total_m > 0 else 0.0

    precision = tp / (tp + fp + EPS)
    recall = tp / (tp + fn + EPS)
    f1 = (2 * precision * recall) / (precision + recall + EPS)
    iou = tp / (tp + fp + fn + EPS)
    dice = (2 * tp) / (2 * tp + fp + fn + EPS)

    present = (tp + fn) > 0

    per_class: Dict[str, Any] = {}
    for i, c in enumerate(classes):
        lbl = int(c)
        per_class[c] = {
            "class_name": class_name_for_arch(lbl, "merged_upper_lower", len(classes)),
            "present_in_gt": int(present[i]),
            "tp": int(tp[i]),
            "fp": int(fp[i]),
            "fn": int(fn[i]),
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "iou": float(iou[i]),
            "dice": float(dice[i]),
        }

    return {
        "arch": "merged_upper_lower",
        "model": upper["model"],
        "ckpt_upper": upper["ckpt"],
        "ckpt_lower": lower["ckpt"],
        "overall": {
            "OA": OA_m,

            "mPrecision": _safe_mean(precision, present),
            "mRecall": _safe_mean(recall, present),
            "mF1": _safe_mean(f1, present),
            "mIoU": _safe_mean(iou, present),
            "mDice": _safe_mean(dice, present),
            "n_present_classes": int(present.sum()),

            # keep minimal bookkeeping
            "total_valid": int(total_m),
            "correct": int(correct_m),
            "tp_sum": int(tp.sum()),
            "fp_sum": int(fp.sum()),
            "fn_sum": int(fn.sum()),
        },
        "per_class": per_class,
    }


def discover_datasets(run_dir: Path) -> List[str]:
    ds = []
    for p in run_dir.iterdir():
        if p.is_dir() and re.match(r"dataset\d+$", p.name):
            ds.append(p.name)
    return sorted(ds, key=lambda x: int(re.findall(r"\d+", x)[0]))


def load_split_meta(dataset_ckpt_dir: Path) -> Dict[str, Any]:
    p = dataset_ckpt_dir / "split_meta.yaml"
    if not p.exists():
        raise FileNotFoundError(f"split_meta.yaml not found: {p}")
    return yaml.safe_load(p.read_text(encoding="utf-8"))


def main():
    run_dir = Path(RUN_DIR)
    if not run_dir.exists():
        raise SystemExit(f"RUN_DIR not found: {run_dir}")

    cfg_path = run_dir / "config.yaml"
    if not cfg_path.exists():
        raise SystemExit(f"config.yaml not found: {cfg_path}")

    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))

    # device
    exp = cfg.get("experiment", {})
    dev_req = str(exp.get("device", "cuda"))
    if dev_req == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device(dev_req)

    out_root = Path(RESULT_ROOT) / run_dir.name
    out_root.mkdir(parents=True, exist_ok=True)

    split_root = Path(SPLIT_ROOT)
    if not split_root.exists():
        raise SystemExit(f"SPLIT_ROOT not found: {split_root}")

    datasets = discover_datasets(run_dir)
    if not datasets:
        raise SystemExit(f"No dataset folders found under {run_dir} (expected dataset1..dataset5)")

    summary_rows: List[Dict[str, Any]] = []

    for ds in datasets:
        ds_ckpt_dir = run_dir / ds
        split_meta = load_split_meta(ds_ckpt_dir)
        test_fold = str(split_meta["test_fold"])

        test_dir = split_root / ds / test_fold
        files_test_all = list_ply(test_dir, recursive=bool(cfg.get("data", {}).get("recursive", False)))
        if not files_test_all:
            raise SystemExit(f"No test .ply found: {test_dir}")

        ckpt_upper = ds_ckpt_dir / "upper" / "best_model.pth"
        ckpt_lower = ds_ckpt_dir / "lower" / "best_model.pth"
        if not ckpt_upper.exists():
            raise SystemExit(f"Missing: {ckpt_upper}")
        if not ckpt_lower.exists():
            raise SystemExit(f"Missing: {ckpt_lower}")

        print(f"\n==================== EVAL {ds} (test={test_fold}) ====================")

        res_u = eval_arch(cfg, "upper", files_test_all, ckpt_upper, device)
        res_l = eval_arch(cfg, "lower", files_test_all, ckpt_lower, device)
        res_m = merge_upper_lower(res_u, res_l)

        ds_out = out_root / ds
        (ds_out / "upper").mkdir(parents=True, exist_ok=True)
        (ds_out / "lower").mkdir(parents=True, exist_ok=True)
        (ds_out / "merged").mkdir(parents=True, exist_ok=True)

        res_u["split_meta"] = split_meta
        res_l["split_meta"] = split_meta
        res_m["split_meta"] = split_meta

        (ds_out / "upper" / "best_eval.json").write_text(json.dumps(res_u, indent=2), encoding="utf-8")
        (ds_out / "lower" / "best_eval.json").write_text(json.dumps(res_l, indent=2), encoding="utf-8")
        (ds_out / "merged" / "best_eval.json").write_text(json.dumps(res_m, indent=2), encoding="utf-8")

        # compact summary row
        summary_rows.append(
            {
                "dataset": ds,
                "test_fold": test_fold,

                "upper_OA": res_u["overall"]["OA"],
                "upper_mIoU": res_u["overall"]["mIoU"],
                "upper_mDice": res_u["overall"]["mDice"],

                "lower_OA": res_l["overall"]["OA"],
                "lower_mIoU": res_l["overall"]["mIoU"],
                "lower_mDice": res_l["overall"]["mDice"],

                "merged_OA": res_m["overall"]["OA"],
                "merged_mIoU": res_m["overall"]["mIoU"],
                "merged_mDice": res_m["overall"]["mDice"],
            }
        )

        print(f"[OK] saved -> {ds_out}")

    def mean_std(vals: List[float]) -> Dict[str, float]:
        a = np.array(vals, dtype=np.float64)
        return {
            "mean": float(a.mean()) if len(a) else 0.0,
            "std": float(a.std(ddof=1)) if len(a) > 1 else 0.0,
        }

    keys = [
        "upper_OA", "upper_mIoU", "upper_mDice",
        "lower_OA", "lower_mIoU", "lower_mDice",
        "merged_OA", "merged_mIoU", "merged_mDice",
    ]
    agg_metrics = {k: mean_std([r[k] for r in summary_rows]) for k in keys}

    agg = {
        "run_dir": str(run_dir),
        "split_root": str(split_root),
        "n_folds": len(summary_rows),
        "metrics": agg_metrics,
        "rows": summary_rows,
        "note": (
            "All means are computed over classes present in GT (tp+fn>0). "
            "Merged OA is computed as (correct_upper+correct_lower)/(total_valid_upper+total_valid_lower). "
            "per_class has readable names e.g., 'T11/T31'."
        ),
    }

    (out_root / "best_summary.json").write_text(json.dumps(agg, indent=2), encoding="utf-8")
    print(f"\n[ALL DONE] results -> {out_root}")
    print(f"[ALL DONE] summary  -> {out_root / 'best_summary.json'}")


if __name__ == "__main__":
    main()
