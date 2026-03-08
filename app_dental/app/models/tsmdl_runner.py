# app/models/tsmdl_runner.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn as nn

from app.core.types import MeshData
from app.models.base import BaseRunner, RunnerOutput

# reuse utility funcs from your existing meshsegnet_runner
from app.models.meshsegnet_runner import (
    import_symbol,
    load_checkpoint_to_model_best_effort,
    recenter_scale_unit_sphere,
)


@dataclass
class BuiltModel:
    model: nn.Module
    ckpt_path: str


def _resolve_ckpt_paths(app_root: Path | None, ckpt_path: str | List[str]) -> List[str]:
    ckpts: List[str] = [str(p) for p in ckpt_path] if isinstance(ckpt_path, (list, tuple)) else [str(ckpt_path)]
    out: List[str] = []
    for p in ckpts:
        pp = Path(p)
        if not pp.is_absolute():
            pp = (Path(app_root) / pp).resolve() if app_root is not None else pp.resolve()
        out.append(str(pp))
    return out


def face_normals(pos_v: np.ndarray, faces: np.ndarray) -> np.ndarray:
    v0 = pos_v[faces[:, 0]]
    v1 = pos_v[faces[:, 1]]
    v2 = pos_v[faces[:, 2]]
    n = np.cross(v1 - v0, v2 - v0).astype(np.float32)
    ln = np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
    return (n / ln).astype(np.float32)


def zscore_per_mesh(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Match training: z-score per mesh (per feature dim)
    """
    x = np.asarray(x, dtype=np.float32)
    mu = x.mean(axis=0, keepdims=True)
    sd = x.std(axis=0, keepdims=True)
    sd = np.maximum(sd, eps)
    return ((x - mu) / sd).astype(np.float32)


def build_tsmdl15_features_like_training(pos_v: np.ndarray, faces: np.ndarray, *, do_zscore: bool = True):
    """
    ✅ EXACTLY match FaceDatasetIMesh2 tsmdl15:

      v0 = pos_v[faces[:,0]]  (F,3)
      v1 = pos_v[faces[:,1]]  (F,3)
      v2 = pos_v[faces[:,2]]  (F,3)
      centers = (v0+v1+v2)/3  (F,3)
      fn_face = face_normals(pos_v, faces) (F,3)
      rel = centers - mean(centers_all_selected) (F,3)

      x = [v0,v1,v2, fn_face, rel] => (F,15)
      if zscore: x = zscore_per_mesh(x)

    pos (for knn_graph) = centers
    """
    v0 = pos_v[faces[:, 0]].astype(np.float32, copy=False)
    v1 = pos_v[faces[:, 1]].astype(np.float32, copy=False)
    v2 = pos_v[faces[:, 2]].astype(np.float32, copy=False)

    centers = ((v0 + v1 + v2) / 3.0).astype(np.float32, copy=False)
    fn_face = face_normals(pos_v, faces).astype(np.float32, copy=False)

    rel = (centers - centers.mean(axis=0, keepdims=True)).astype(np.float32, copy=False)

    x = np.concatenate([v0, v1, v2, fn_face, rel], axis=1).astype(np.float32, copy=False)  # (F,15)

    if do_zscore:
        x = zscore_per_mesh(x)

    pos = centers  # (F,3) used for knn edges
    return x, pos


class TSMdlRunner(BaseRunner):
    """
    Runner for IMeshSegNetKNNBatch (TS-MDL-like):
      - build tsmdl15 features exactly like training dataset
      - forward with {'x':(B,N,15), 'pos':(B,N,3)}
      - ensemble avg probs
      - return labels_face + centers_used + probs_used
    """

    def __init__(self, models: List[BuiltModel], device: str | torch.device, *, in_channels: int = 15, zscore: bool = True):
        if not models:
            raise ValueError("TSMdlRunner requires at least 1 model")
        self.models = models
        self.device = torch.device(device) if not isinstance(device, torch.device) else device
        self.in_channels = int(in_channels)
        self.zscore = bool(zscore)

        print(f"[Runner] tsmdl device='{self.device}' | num_models={len(models)} | in_channels={self.in_channels} | zscore={self.zscore}")

    @staticmethod
    def build(
        model_import: str,
        model_kwargs: Dict[str, Any],
        ckpt_path: str | List[str],
        device: str | torch.device = "cuda",
        *,
        app_root: Path | None = None,
    ) -> "TSMdlRunner":
        dev = torch.device(device) if not isinstance(device, torch.device) else device
        ModelCls = import_symbol(model_import)

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

        in_ch = int((model_kwargs or {}).get("in_channels", 15))
        # training dataset sets zscore=True by default
        zscore = bool((model_kwargs or {}).get("zscore", True))
        return TSMdlRunner(built, dev, in_channels=in_ch, zscore=zscore)

    @torch.no_grad()
    def infer(self, mesh: MeshData, *, fidx: np.ndarray | None) -> RunnerOutput:
        if mesh.faces is None:
            raise ValueError("TSMdlRunner requires mesh.faces (triangles).")

        v = np.asarray(mesh.pos, dtype=np.float32)
        f_all = np.asarray(mesh.faces, dtype=np.int64)
        F_total = int(f_all.shape[0])

        if fidx is not None:
            fidx = np.asarray(fidx, dtype=np.int64).reshape(-1)
            f = f_all[fidx]
        else:
            f = f_all
            fidx = None

        # normalize like other runners (consistent with do_normalize)
        v_norm, nmeta = recenter_scale_unit_sphere(v)

        # build EXACT training features
        x_np, pos_np = build_tsmdl15_features_like_training(v_norm, f, do_zscore=self.zscore)

        if x_np.shape[1] != self.in_channels:
            raise ValueError(f"in_channels mismatch: expected {self.in_channels}, got {x_np.shape[1]}")

        x_t = torch.from_numpy(x_np[None, ...]).to(self.device, dtype=torch.float32)     # (1,N,15)
        pos_t = torch.from_numpy(pos_np[None, ...]).to(self.device, dtype=torch.float32) # (1,N,3)

        probs_sum = None
        C = None

        for bm in self.models:
            logits = bm.model({"x": x_t, "pos": pos_t})
            if logits.ndim != 3:
                raise RuntimeError(f"TSMDL model output must be (B,N,C). Got {tuple(logits.shape)}")
            if C is None:
                C = int(logits.shape[-1])
            probs = torch.softmax(logits, dim=-1)
            probs_sum = probs if probs_sum is None else (probs_sum + probs)

        probs_avg = (probs_sum / float(len(self.models))).squeeze(0)  # (N,C)
        labels = torch.argmax(probs_avg, dim=-1)                      # (N,)

        labels_np = labels.detach().long().cpu().numpy()
        probs_np = probs_avg.detach().float().cpu().numpy().astype(np.float16, copy=False)

        meta: Dict[str, Any] = {
            "runner": "tsmdl",
            "device": str(self.device),
            "arch": mesh.arch,
            "path": mesh.path,
            "num_models": int(len(self.models)),
            "ckpt_paths": [bm.ckpt_path for bm in self.models],
            "normalize_meta": nmeta,
            "faces_total": F_total,
            "faces_used": int(f.shape[0]),
            "fidx": (fidx.tolist() if fidx is not None else None),
            "feature": "tsmdl15(v0,v1,v2,n,rel)+zscore",
            "centers_used": pos_np.astype(np.float32, copy=False),
            "probs_used": probs_np,
        }

        return RunnerOutput(labels_face=labels_np, num_classes=int(C or 0), meta=meta)