# train.py
from __future__ import annotations

import re
import sys
import random
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import yaml

try:
    from tqdm import tqdm
except ImportError as e:
    raise SystemExit("Missing tqdm. Install with: python -m pip install tqdm") from e

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dataloader.arch import parse_arch_from_filename
from src.dataloader import build_dataset, build_collate_fn

DEFAULT_CONFIG = "configs/pointcnn15.yaml"


# ============================================================
# AMP helpers
# ============================================================
def make_grad_scaler(use_amp: bool):
    # torch>=2.0 recommended path
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        try:
            return torch.amp.GradScaler("cuda", enabled=use_amp)
        except TypeError:
            return torch.amp.GradScaler(enabled=use_amp)
    # fallback
    return torch.cuda.amp.GradScaler(enabled=use_amp)


def autocast_ctx(use_amp: bool):
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        try:
            return torch.amp.autocast("cuda", enabled=use_amp)
        except TypeError:
            return torch.amp.autocast(enabled=use_amp)
    return torch.cuda.amp.autocast(enabled=use_amp)


# ============================================================
# utils
# ============================================================
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def list_ply(dir_: str, recursive: bool = False) -> List[str]:
    if not dir_:
        return []
    p = Path(dir_)
    if not p.exists():
        return []
    pat = "**/*.ply" if recursive else "*.ply"
    return sorted([str(x) for x in p.glob(pat)])


def split_by_arch(files: List[str]) -> Tuple[List[str], List[str]]:
    """
    STRICT: any file that cannot be inferred -> raise
    (prevents silently pushing unknown -> lower)
    """
    upper, lower = [], []
    for f in files:
        arch = parse_arch_from_filename(f)
        if arch == "upper":
            upper.append(f)
        elif arch == "lower":
            lower.append(f)
        else:
            raise ValueError(f"Cannot infer arch from filename: {f}")
    return upper, lower


def import_from_path(path: str):
    if ":" not in path:
        raise ValueError(f"model.import must be 'module:ClassName', got: {path}")
    mod_name, obj_name = path.split(":", 1)
    mod = __import__(mod_name, fromlist=[obj_name])
    return getattr(mod, obj_name)


def normalize_logits_to_bnc(logits: torch.Tensor, num_classes: int) -> torch.Tensor:
    """
    Return logits in shape (B, N, C)
    Accept:
      - (B, N, C)  => ok
      - (B, C, N)  => permute
    """
    if logits.dim() != 3:
        raise ValueError(f"Model output must be 3D, got {tuple(logits.shape)}")
    if logits.shape[-1] == num_classes:
        return logits
    if logits.shape[1] == num_classes:
        return logits.permute(0, 2, 1).contiguous()
    raise ValueError(f"Cannot infer logits layout from shape {tuple(logits.shape)} (num_classes={num_classes})")


def next_run_id(parent: Path) -> int:
    parent.mkdir(parents=True, exist_ok=True)
    mx = 0
    for d in parent.iterdir():
        if d.is_dir():
            m = re.search(r"_run(\d+)$", d.name)
            if m:
                mx = max(mx, int(m.group(1)))
    return mx + 1


def infer_tags_from_import(model_import: str) -> tuple[str, str]:
    mod, cls = model_import.split(":", 1)
    group = mod.split(".")[-1]
    tag = cls
    return group, tag


def format_num(n: int) -> str:
    if n >= 1_000_000:
        return f"{n/1_000_000:.2f}M"
    if n >= 1_000:
        return f"{n/1_000:.2f}K"
    return str(n)


def describe_device(device: torch.device) -> str:
    if device.type != "cuda":
        return f"{device} (CPU)"
    idx = device.index if device.index is not None else torch.cuda.current_device()
    prop = torch.cuda.get_device_properties(idx)
    total_gb = prop.total_memory / (1024 ** 3)
    cuda_ver = getattr(torch.version, "cuda", None)
    cudnn_ver = torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None
    cap = f"{prop.major}.{prop.minor}"
    return (
        f"cuda:{idx} | GPU={prop.name} | CC={cap} | VRAM={total_gb:.2f} GB | "
        f"torch={torch.__version__} | cuda={cuda_ver} | cudnn={cudnn_ver}"
    )


def count_params(model: nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return int(total), int(trainable)


# ============================================================
# split resolver (train/test) - supports split_root OR train_dir/test_dir
# ============================================================
def resolve_train_test_files(
    cfg: dict, *, dataset_name: Optional[str] = None
) -> tuple[List[str], List[str], Dict[str, Any]]:
    data = cfg["data"]
    recursive = bool(data.get("recursive", False))

    # split_root mode
    if "split_root" in data and data["split_root"]:
        split_root = Path(str(data["split_root"]))
        ds = dataset_name or str(data.get("dataset", "dataset1"))
        ds_dir = split_root / ds
        if not ds_dir.exists():
            raise FileNotFoundError(f"Dataset dir not found: {ds_dir}")

        folds = data.get("folds", None)
        if not folds:
            folds = sorted([d.name for d in ds_dir.iterdir() if d.is_dir()])

        test_map = data.get("test_fold_by_dataset", {}) or {}
        test_fold = str(test_map.get(ds, data.get("test_fold", "fold1")))

        train_files: List[str] = []
        test_files: List[str] = []
        for fold in folds:
            files = list_ply(str(ds_dir / fold), recursive=recursive)
            if fold == test_fold:
                test_files.extend(files)
            else:
                train_files.extend(files)

        meta = {
            "split_mode": "split_root",
            "dataset": ds,
            "test_fold": test_fold,
            "folds": list(folds),
            "split_root": str(split_root),
        }
        return train_files, test_files, meta

    # legacy train_dir/test_dir mode
    train_dir = str(data.get("train_dir", ""))
    test_dir = str(data.get("test_dir", ""))
    train_files = list_ply(train_dir, recursive=recursive)
    test_files = list_ply(test_dir, recursive=recursive) if test_dir else []
    meta = {"split_mode": "train_dir", "train_dir": train_dir, "test_dir": test_dir}
    return train_files, test_files, meta


# ============================================================
# forward router
# ============================================================
def forward_model(
    model: nn.Module,
    batch: Dict[str, Any],
    device: torch.device,
    forward_type: str,
) -> torch.Tensor:
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


# ============================================================
# masked CE (robust for padding/ignore)
# ============================================================
def masked_ce_loss(
    ce_none: nn.Module,
    logits_bnc: torch.Tensor,  # (B,N,C)
    y: torch.Tensor,           # (B,N)
    mask: torch.Tensor,        # (B,N) bool
    num_classes: int,
) -> tuple[torch.Tensor, int]:
    loss_vec = ce_none(logits_bnc.reshape(-1, num_classes), y.reshape(-1))  # (B*N,)
    m = mask.reshape(-1)
    denom = int(m.sum().item())
    if denom <= 0:
        return loss_vec.sum() * 0.0, 0
    loss = (loss_vec * m.float()).sum() / m.float().sum().clamp_min(1.0)
    return loss, denom


@torch.no_grad()
def eval_loss(
    model: nn.Module,
    loader: Optional[DataLoader],
    device: torch.device,
    ce_none: nn.Module,
    num_classes: int,
    forward_type: str,
    ignore_index: int,
) -> Optional[float]:
    if loader is None:
        return None

    model.eval()
    loss_sum = 0.0
    den_sum = 0

    for batch in loader:
        y = batch["y"].to(device, non_blocking=True)

        mask = batch.get("mask", None)
        if mask is None or (not torch.is_tensor(mask)):
            mask = (y != ignore_index)
        else:
            mask = mask.to(device, non_blocking=True).bool()

        logits = forward_model(model, batch, device, forward_type)
        logits_bnc = normalize_logits_to_bnc(logits, num_classes)

        loss, den = masked_ce_loss(ce_none, logits_bnc, y, mask, num_classes)
        if den > 0:
            loss_sum += float(loss.item()) * den
            den_sum += den

    if den_sum <= 0:
        return None
    return loss_sum / float(den_sum)


# ============================================================
# train one arch
# ============================================================
def train_one_arch(
    cfg: dict,
    arch: str,
    *,
    files_train_all: List[str],
    files_test_all: List[str],
    arch_dir: Path,
    run_dir: Path,
    split_meta: Dict[str, Any],
):
    exp = cfg.get("experiment", {})
    seed = int(exp.get("seed", 42))
    set_seed(seed)

    device_req = str(exp.get("device", "cuda"))
    if device_req == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device(device_req)

    use_amp = bool(exp.get("amp", False)) and device.type == "cuda"
    scaler = make_grad_scaler(use_amp)

    data = cfg["data"]
    mode = str(data["mode"]).lower()
    ignore_index = int(data.get("ignore_index", -1))
    forward_type = cfg["model"].get("forward_type", "x")

    epochs = int(cfg["train"]["epochs"])
    batch_size = int(data["batch_size"])
    num_workers = int(data.get("num_workers", 0))
    pin_memory = bool(data.get("pin_memory", True)) and (device.type == "cuda")

    # IMPORTANT for MeshSegNet (BatchNorm): avoid last batch B=1
    drop_last_train = bool(data.get("drop_last_train", True)) and (batch_size > 1)

    tr_u, tr_l = split_by_arch(files_train_all)
    te_u, te_l = split_by_arch(files_test_all) if files_test_all else ([], [])

    files_train = tr_u if arch == "upper" else tr_l
    files_test = te_u if arch == "upper" else te_l

    if len(files_train) == 0:
        raise RuntimeError(f"No training files for arch='{arch}'. Check filenames and split_root.")

    ds_train = build_dataset(files_train, cfg, arch=arch)
    collate_fn = build_collate_fn(cfg)

    dl_train = DataLoader(
        ds_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
        drop_last=drop_last_train,
    )

    dl_test = None
    if len(files_test) > 0:
        ds_test = build_dataset(files_test, cfg, arch=arch)
        dl_test = DataLoader(
            ds_test,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=collate_fn,
            drop_last=False,
        )

    ModelCls = import_from_path(cfg["model"]["import"])
    model_kwargs = cfg["model"].get("kwargs", {})
    num_classes = int(model_kwargs.get("num_classes", 16))
    model = ModelCls(**model_kwargs).to(device)

    opt = str(cfg["optim"]["name"]).lower()
    lr = float(cfg["optim"]["lr"])
    wd = float(cfg["optim"].get("weight_decay", 0.0))
    if opt == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    elif opt == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    elif opt == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=wd)
    else:
        raise ValueError(f"Unknown optimizer: {opt}")

    ce_none = nn.CrossEntropyLoss(ignore_index=ignore_index, reduction="none")

    arch_dir.mkdir(parents=True, exist_ok=True)
    best_path = arch_dir / "best_model.pth"
    last_path = arch_dir / "last_model.pth"
    best_score = float("inf")

    total_p, train_p = count_params(model)
    print(
        "\n"
        "---------------------- RUN INFO -----------------------\n"
        f"[SPLIT]    {split_meta}\n"
        f"[ARCH]     {arch}\n"
        f"[MODE]     {mode}\n"
        f"[DEVICE]   {describe_device(device)}\n"
        f"[AMP]      {use_amp}\n"
        f"[MODEL]    import={cfg['model']['import']}\n"
        f"[MODEL]    forward_type={forward_type}\n"
        f"[MODEL]    kwargs={model_kwargs}\n"
        f"[MODEL]    params_total={format_num(total_p)} params_trainable={format_num(train_p)}\n"
        f"[DATA]     train_files={len(files_train)} test_files={len(files_test)}\n"
        f"[DATA]     batch_size={batch_size} drop_last_train={drop_last_train}\n"
        f"[OUTPUT]   run_dir={run_dir}\n"
        f"[OUTPUT]   arch_dir={arch_dir}\n"
        f"[OUTPUT]   best_ckpt={best_path}\n"
        f"[OUTPUT]   last_ckpt={last_path}\n"
        "--------------------------------------------------------"
    )

    for epoch in range(1, epochs + 1):
        model.train()
        loss_sum = 0.0
        den_sum = 0

        pbar = tqdm(
            total=len(dl_train),
            desc=f"Epoch {epoch}/{epochs} [TRAIN {arch}]",
            unit="it",
            leave=True,
            dynamic_ncols=True,
            file=sys.stdout,
        )

        for batch in dl_train:
            y = batch["y"].to(device, non_blocking=True)

            mask = batch.get("mask", None)
            if mask is None or (not torch.is_tensor(mask)):
                mask = (y != ignore_index)
            else:
                mask = mask.to(device, non_blocking=True).bool()

            optimizer.zero_grad(set_to_none=True)

            with autocast_ctx(use_amp):
                logits = forward_model(model, batch, device, forward_type)
                logits_bnc = normalize_logits_to_bnc(logits, num_classes)
                loss, den = masked_ce_loss(ce_none, logits_bnc, y, mask, num_classes)

            if use_amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            if den > 0:
                loss_sum += float(loss.item()) * den
                den_sum += den

            pbar.update(1)
            pbar.set_postfix(loss=f"{float(loss.item()):.4f}", valid=f"{den}")

        pbar.close()

        train_loss = (loss_sum / float(max(den_sum, 1))) if den_sum > 0 else float("nan")
        test_loss = eval_loss(model, dl_test, device, ce_none, num_classes, forward_type, ignore_index)

        score = test_loss if (test_loss is not None) else train_loss
        if test_loss is not None:
            print(f"[Epoch {epoch:03d}][{arch}] train_loss={train_loss:.6f} test_loss={test_loss:.6f}")
        else:
            print(f"[Epoch {epoch:03d}][{arch}] train_loss={train_loss:.6f}")

        torch.save(
            {
                "epoch": epoch,
                "arch": arch,
                "mode": mode,
                "split_meta": split_meta,
                "model_import": cfg["model"]["import"],
                "model_kwargs": model_kwargs,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scaler_state": scaler.state_dict() if use_amp else None,
                "train_loss": train_loss,
                "test_loss": test_loss,
                "config": cfg,
                "run_dir": str(run_dir),
                "arch_dir": str(arch_dir),
            },
            last_path,
        )

        if (score is not None) and (score < best_score):
            best_score = float(score)
            torch.save(
                {
                    "epoch": epoch,
                    "arch": arch,
                    "mode": mode,
                    "split_meta": split_meta,
                    "model_import": cfg["model"]["import"],
                    "model_kwargs": model_kwargs,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "scaler_state": scaler.state_dict() if use_amp else None,
                    "train_loss": train_loss,
                    "test_loss": test_loss,
                    "config": cfg,
                    "run_dir": str(run_dir),
                    "arch_dir": str(arch_dir),
                },
                best_path,
            )
            tag = "test_loss" if (test_loss is not None) else "train_loss"
            print(f"[{arch}] -> BEST({tag})={best_score:.6f} (saved: {best_path})")

    print(f"[DONE] arch={arch} best_score={best_score:.6f}")


def train_one_dataset(cfg: dict, *, dataset_name: Optional[str], ds_run_dir: Path):
    arch_cfg = str(cfg["data"]["arch"]).lower()
    if arch_cfg == "both":
        arch_list = ["upper", "lower"]
    elif arch_cfg in ("upper", "lower"):
        arch_list = [arch_cfg]
    else:
        raise ValueError("data.arch must be upper/lower/both")

    files_train_all, files_test_all, split_meta = resolve_train_test_files(cfg, dataset_name=dataset_name)

    ds_run_dir.mkdir(parents=True, exist_ok=True)
    (ds_run_dir / "split_meta.yaml").write_text(
        yaml.safe_dump(split_meta, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )

    for arch in arch_list:
        arch_dir = ds_run_dir / arch
        train_one_arch(
            cfg,
            arch=arch,
            files_train_all=files_train_all,
            files_test_all=files_test_all,
            arch_dir=arch_dir,
            run_dir=ds_run_dir,
            split_meta=split_meta,
        )


def main():
    cfg_path = ROOT / DEFAULT_CONFIG
    if not cfg_path.exists():
        raise SystemExit(f"Config not found: {cfg_path}")

    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))

    exp = cfg.get("experiment", {})
    ckpt_root = exp.get("checkpoints_root", "checkpoints")
    dataset_tag = exp.get("dataset_tag", "IoSSeg")
    epochs = int(cfg["train"]["epochs"])

    model_import = cfg["model"]["import"]
    group_auto, tag_auto = infer_tags_from_import(model_import)
    model_group = exp.get("model_group", group_auto)
    model_tag = exp.get("model_tag", tag_auto)

    parent = Path(ckpt_root) / model_group
    run_id = next_run_id(parent)

    run_all = bool(cfg["data"].get("run_all_datasets", False))
    if run_all:
        run_name = f"{model_tag}_{dataset_tag}_ALLDATA_{epochs}epoch_run{run_id}"
    else:
        run_name = f"{model_tag}_{dataset_tag}_{epochs}epoch_run{run_id}"

    run_dir = parent / run_name
    run_dir.mkdir(parents=True, exist_ok=False)

    (run_dir / "config.yaml").write_text(
        yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )

    print(
        "\n"
        "==================== TRAIN SESSION ====================\n"
        f"[CONFIG]   {cfg_path}\n"
        f"[MODEL]    {model_import}\n"
        f"[RUN]      {run_name}\n"
        f"[CKPTROOT] {parent.resolve()}\n"
        f"[OUT]      {run_dir.resolve()}\n"
        "========================================================"
    )

    if run_all:
        datasets = cfg["data"].get("datasets", None)
        if not datasets:
            split_root = Path(str(cfg["data"]["split_root"]))
            datasets = sorted([d.name for d in split_root.iterdir() if d.is_dir()])

        for ds in datasets:
            ds_run_dir = run_dir / str(ds)
            print(f"\n==================== DATASET: {ds} ====================")
            train_one_dataset(cfg, dataset_name=str(ds), ds_run_dir=ds_run_dir)
    else:
        train_one_dataset(cfg, dataset_name=cfg["data"].get("dataset", None), ds_run_dir=run_dir)

    print("\n[ALL DONE]")


if __name__ == "__main__":
    main()
