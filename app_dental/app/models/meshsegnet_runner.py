# dental_seg_app/app/models/meshsegnet_runner.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple, Mapping

import importlib
import numpy as np
import torch
import torch.nn as nn

from app.core.types import MeshData
from app.models.base import BaseRunner, RunnerOutput


def import_symbol(spec: str):
    if ":" not in spec:
        raise ValueError(f"model_import must be like 'pkg.mod:Name', got: {spec}")
    mod, name = spec.split(":", 1)
    m = importlib.import_module(mod)
    return getattr(m, name)


def _resolve_ckpt_paths(app_root: Path | None, ckpt_path: str | List[str]) -> List[str]:
    """
    ✅ deploy-friendly: resolve ckpt path จาก app_root
    - absolute -> ใช้ตรง ๆ
    - relative -> app_root / path
    """
    ckpts: List[str] = [str(p) for p in ckpt_path] if isinstance(ckpt_path, (list, tuple)) else [str(ckpt_path)]
    out: List[str] = []
    for p in ckpts:
        pp = Path(p)
        if not pp.is_absolute():
            pp = (Path(app_root) / pp).resolve() if app_root is not None else pp.resolve()
        out.append(str(pp))
    return out


def _looks_like_state_dict(obj: Any) -> bool:
    if not isinstance(obj, Mapping) or len(obj) == 0:
        return False
    n = 0
    for k, v in obj.items():
        if not isinstance(k, str):
            return False
        if torch.is_tensor(v) or isinstance(v, np.ndarray):
            n += 1
        else:
            return False
        if n >= 5:
            break
    return n > 0


def _iter_state_dict_candidates(ckpt: Any) -> List[Dict[str, Any]]:
    cands: List[Dict[str, Any]] = []

    if _looks_like_state_dict(ckpt):
        cands.append(dict(ckpt))
        return cands

    if isinstance(ckpt, Mapping):
        keys = [
            "state_dict",
            "model_state_dict",
            "model_state",
            "model",
            "net",
            "network",
            "weights",
            "params",
            "ema",
            "student",
            "teacher",
        ]

        for k in keys:
            v = ckpt.get(k, None)
            if v is None:
                continue
            if _looks_like_state_dict(v):
                cands.append(dict(v))
                continue
            if isinstance(v, Mapping):
                v2 = v.get("state_dict", None)
                if _looks_like_state_dict(v2):
                    cands.append(dict(v2))
                    continue

        for _, v in ckpt.items():
            if _looks_like_state_dict(v):
                cands.append(dict(v))
            elif isinstance(v, Mapping):
                v2 = v.get("state_dict", None)
                if _looks_like_state_dict(v2):
                    cands.append(dict(v2))

    if len(cands) > 1:
        seen = set()
        uniq: List[Dict[str, Any]] = []
        for sd in cands:
            sig = (len(sd), next(iter(sd.keys())))
            if sig in seen:
                continue
            seen.add(sig)
            uniq.append(sd)
        cands = uniq

    return cands


def _strip_prefix_in_keys(sd: Dict[str, Any], prefix: str) -> Dict[str, Any]:
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


def _choose_best_state_dict(sd_list: List[Dict[str, Any]], model: nn.Module) -> Tuple[Dict[str, Any], int, List[str]]:
    model_keys = set(model.state_dict().keys())
    best_sd = None
    best_score = -1
    best_strips: List[str] = []

    prefixes = [
        "",
        "module.",
        "model.",
        "net.",
        "network.",
        "core.",
        "student.",
        "teacher.",
        "module.core.",
        "model.core.",
    ]

    for sd0 in sd_list:
        for p in prefixes:
            sd = _strip_prefix_in_keys(sd0, p)
            score = _match_score(sd, model_keys)
            if score > best_score:
                best_sd = sd
                best_score = score
                best_strips = [p] if p else []

        for p, sd in _auto_strip_single_token_prefix(sd0):
            score = _match_score(sd, model_keys)
            if score > best_score:
                best_sd = sd
                best_score = score
                best_strips = [p]

    if best_sd is None:
        raise RuntimeError("No usable state_dict candidates")

    return best_sd, int(best_score), best_strips


def torch_load_best_effort(path: str | Path, map_location="cpu"):
    p = str(path)
    try:
        return torch.load(p, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(p, map_location=map_location)
    except Exception:
        return torch.load(p, map_location=map_location)


def load_checkpoint_to_model_best_effort(model: nn.Module, ckpt_path: str | Path) -> Tuple[List[str], List[str]]:
    ckpt = torch_load_best_effort(ckpt_path, map_location="cpu")
    cands = _iter_state_dict_candidates(ckpt)

    if not cands:
        if isinstance(ckpt, Mapping):
            raise RuntimeError(
                "Unknown checkpoint format (cannot find state_dict). "
                f"Top-level keys={list(ckpt.keys())}"
            )
        raise RuntimeError(f"Unknown checkpoint format (type={type(ckpt)})")

    best_sd, score, strips = _choose_best_state_dict(cands, model)

    ret = model.load_state_dict(best_sd, strict=False)
    missing = list(ret.missing_keys) if hasattr(ret, "missing_keys") else []
    unexpected = list(ret.unexpected_keys) if hasattr(ret, "unexpected_keys") else []

    print(f"[CKPT] loaded: {Path(ckpt_path).name} | match_score={score} | used_prefix_strip={strips}")
    if missing:
        print(f"[CKPT] missing_keys: {len(missing)}")
    if unexpected:
        print(f"[CKPT] unexpected_keys: {len(unexpected)}")

    return missing, unexpected


def recenter_scale_unit_sphere(pos: np.ndarray) -> tuple[np.ndarray, dict]:
    pos = np.asarray(pos, dtype=np.float32)
    c = pos.mean(axis=0, keepdims=True)
    pos0 = pos - c
    r = float(np.linalg.norm(pos0, axis=1).max() + 1e-12)
    posn = (pos0 / r).astype(np.float32)
    meta = {"center": c.reshape(-1).tolist(), "radius": r}
    return posn, meta


@dataclass
class BuiltModel:
    model: nn.Module
    ckpt_path: str


class MeshSegNetRunner(BaseRunner):
    def __init__(self, models: List[BuiltModel], device: str | torch.device):
        if not models:
            raise ValueError("MeshSegNetRunner requires at least 1 model")
        self.models = models
        self.device = torch.device(device) if not isinstance(device, torch.device) else device
        print(f"[Runner] device='{self.device}' | num_models={len(models)}")

    @staticmethod
    def build(
        model_import: str,
        model_kwargs: Dict[str, Any],
        ckpt_path: str | List[str],
        device: str | torch.device = "cuda",
        *,
        app_root: Path | None = None,  # ✅ เพิ่ม
    ) -> "MeshSegNetRunner":
        dev = torch.device(device) if not isinstance(device, torch.device) else device
        ModelCls = import_symbol(model_import)

        # ✅ resolve ckpt จาก app_root
        ckpts: List[str] = _resolve_ckpt_paths(app_root, ckpt_path)

        built: List[BuiltModel] = []
        for p in ckpts:
            if not Path(p).exists():
                raise FileNotFoundError(f"Checkpoint not found: {p}")

            m = ModelCls(**(model_kwargs or {}))
            m.eval()
            m.to(dev)
            load_checkpoint_to_model_best_effort(m, p)
            built.append(BuiltModel(model=m, ckpt_path=p))

        return MeshSegNetRunner(built, dev)

    @torch.no_grad()
    def infer(self, mesh: MeshData, *, fidx: np.ndarray | None) -> RunnerOutput:
        if mesh.faces is None:
            raise ValueError("MeshSegNetRunner requires mesh.faces (triangle faces).")

        v = np.asarray(mesh.pos, dtype=np.float32)
        f_all = np.asarray(mesh.faces, dtype=np.int64)

        if f_all.ndim != 2 or f_all.shape[1] != 3:
            raise ValueError(f"mesh.faces must be (F,3), got {f_all.shape}")
        if v.ndim != 2 or v.shape[1] != 3:
            raise ValueError(f"mesh.pos must be (Nv,3), got {v.shape}")
        if not np.isfinite(v).all():
            raise ValueError("mesh.pos contains NaN/Inf")

        F_total = int(f_all.shape[0])

        if fidx is not None:
            fidx = np.asarray(fidx, dtype=np.int64).reshape(-1)
            if fidx.size == 0:
                raise ValueError("fidx is empty")
            if int(fidx.min()) < 0 or int(fidx.max()) >= F_total:
                raise ValueError(f"fidx out of range: min={int(fidx.min())}, max={int(fidx.max())}, F={F_total}")
            f = f_all[fidx]
        else:
            f = f_all
            fidx = None

        v_norm, nmeta = recenter_scale_unit_sphere(v)

        v0 = v_norm[f[:, 0]]
        v1 = v_norm[f[:, 1]]
        v2 = v_norm[f[:, 2]]

        centers = (v0 + v1 + v2) / 3.0

        n = np.cross(v1 - v0, v2 - v0)
        nnorm = np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
        n = (n / nnorm).astype(np.float32)

        dv0 = (v0 - centers).astype(np.float32)
        dv1 = (v1 - centers).astype(np.float32)
        dv2 = (v2 - centers).astype(np.float32)

        x = np.concatenate([centers, n, dv0, dv1, dv2], axis=1).astype(np.float32)

        x_t = torch.as_tensor(x, dtype=torch.float32, device=self.device).unsqueeze(0)
        pos_t = torch.as_tensor(centers, dtype=torch.float32, device=self.device).unsqueeze(0)
        batch = {"x": x_t, "pos": pos_t}

        probs_sum = None
        num_classes = None

        for bm in self.models:
            logits = bm.model(batch)  # (1,F,C)
            if logits.ndim != 3:
                raise RuntimeError(f"Model output must be (B,F,C). Got {tuple(logits.shape)}")
            if num_classes is None:
                num_classes = int(logits.shape[-1])
            probs = torch.softmax(logits, dim=-1)
            probs_sum = probs if probs_sum is None else (probs_sum + probs)

        probs_avg = (probs_sum / float(len(self.models))).squeeze(0)  # (F,C)
        labels = torch.argmax(probs_avg, dim=-1)                      # (F,)

        labels_np = labels.detach().long().cpu().numpy()
        probs_np = probs_avg.detach().float().cpu().numpy().astype(np.float16, copy=False)

        meta = {
            "runner": "meshsegnet",
            "device": str(self.device),
            "arch": mesh.arch,
            "path": mesh.path,
            "num_models": int(len(self.models)),
            "ckpt_paths": [bm.ckpt_path for bm in self.models],
            "normalize": True,
            "normalize_meta": nmeta,
            "faces_total": F_total,
            "faces_used": int(f.shape[0]),
            "fidx": (fidx.tolist() if fidx is not None else None),
            "centers_used": centers.astype(np.float32, copy=False),
            "probs_used": probs_np,  # ✅ for postprocess
        }

        return RunnerOutput(labels_face=labels_np, num_classes=int(num_classes or 0), meta=meta)