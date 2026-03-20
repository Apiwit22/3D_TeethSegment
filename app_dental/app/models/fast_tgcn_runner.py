from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn

from app.core.types import MeshData
from app.models.base import BaseRunner, RunnerOutput
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


def _compute_vertex_normals(v: np.ndarray, f: np.ndarray) -> np.ndarray:
    """
    Area-weighted vertex normals, close to read_ply() fallback behavior.
    """
    v = np.asarray(v, dtype=np.float32)
    f = np.asarray(f, dtype=np.int64)

    nv = int(v.shape[0])
    vn = np.zeros((nv, 3), dtype=np.float32)

    v0 = v[f[:, 0]]
    v1 = v[f[:, 1]]
    v2 = v[f[:, 2]]

    fn = np.cross(v1 - v0, v2 - v0).astype(np.float32)

    np.add.at(vn, f[:, 0], fn)
    np.add.at(vn, f[:, 1], fn)
    np.add.at(vn, f[:, 2], fn)

    ln = np.linalg.norm(vn, axis=1, keepdims=True) + 1e-12
    vn = vn / ln
    return vn.astype(np.float32, copy=False)


def _face_normals(v: np.ndarray, f: np.ndarray) -> np.ndarray:
    v0 = v[f[:, 0]]
    v1 = v[f[:, 1]]
    v2 = v[f[:, 2]]
    n = np.cross(v1 - v0, v2 - v0).astype(np.float32)
    ln = np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
    return (n / ln).astype(np.float32, copy=False)


def _face_centers(v: np.ndarray, f: np.ndarray) -> np.ndarray:
    v0 = v[f[:, 0]]
    v1 = v[f[:, 1]]
    v2 = v[f[:, 2]]
    return ((v0 + v1 + v2) / 3.0).astype(np.float32, copy=False)


def _build_twostream24_relative(
    v: np.ndarray, f: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Match train dataloader twostream24, coord_mode='relative':

      x_c = [center, v0-center, v1-center, v2-center]
      x_n = [n0, n1, n2, n_face]
      x   = concat(x_c, x_n)
    """
    v = np.asarray(v, dtype=np.float32)
    f = np.asarray(f, dtype=np.int64)

    v0 = v[f[:, 0]].astype(np.float32, copy=False)
    v1 = v[f[:, 1]].astype(np.float32, copy=False)
    v2 = v[f[:, 2]].astype(np.float32, copy=False)

    centers = ((v0 + v1 + v2) / 3.0).astype(np.float32, copy=False)
    dv0 = (v0 - centers).astype(np.float32, copy=False)
    dv1 = (v1 - centers).astype(np.float32, copy=False)
    dv2 = (v2 - centers).astype(np.float32, copy=False)
    x_c = np.concatenate([centers, dv0, dv1, dv2], axis=1).astype(np.float32, copy=False)

    vn_all = _compute_vertex_normals(v, f)
    fn_face = _face_normals(v, f).astype(np.float32, copy=False)

    n0 = vn_all[f[:, 0]].astype(np.float32, copy=False)
    n1 = vn_all[f[:, 1]].astype(np.float32, copy=False)
    n2 = vn_all[f[:, 2]].astype(np.float32, copy=False)
    x_n = np.concatenate([n0, n1, n2, fn_face], axis=1).astype(np.float32, copy=False)

    x = np.concatenate([x_c, x_n], axis=1).astype(np.float32, copy=False)
    return centers, x_c, x_n, x


def _unique_edge_index(edges: List[Tuple[int, int]]) -> np.ndarray:
    if not edges:
        return np.zeros((2, 0), dtype=np.int64)
    arr = np.asarray(edges, dtype=np.int64)
    arr = np.unique(arr, axis=0)
    return arr.T.astype(np.int64, copy=False)


def _add_self_loops_np(edge_index: np.ndarray, num_nodes: int) -> np.ndarray:
    if num_nodes <= 0:
        return edge_index
    loops = np.arange(num_nodes, dtype=np.int64)
    loops = np.stack([loops, loops], axis=0)  # (2,N)
    if edge_index.size == 0:
        return loops
    out = np.concatenate([edge_index, loops], axis=1)
    out = np.unique(out.T, axis=0).T if out.size else out
    return out.astype(np.int64, copy=False)


def _build_face_adj_vertex_edge_index(faces: np.ndarray, add_self_loops: bool = True) -> np.ndarray:
    """
    Build graph equivalent to:
      graph_type='face_adj'
      adjacency_mode='vertex'
      add_self_loops=true
    Two faces are adjacent if they share at least one vertex.
    """
    faces = np.asarray(faces, dtype=np.int64)
    F = int(faces.shape[0])
    if F <= 0:
        return np.zeros((2, 0), dtype=np.int64)

    v_map: Dict[int, List[int]] = {}
    for fi, (a, b, c) in enumerate(faces):
        v_map.setdefault(int(a), []).append(fi)
        v_map.setdefault(int(b), []).append(fi)
        v_map.setdefault(int(c), []).append(fi)

    edges: List[Tuple[int, int]] = []
    for inc in v_map.values():
        if len(inc) <= 1:
            continue
        for i in inc:
            for j in inc:
                if i != j:
                    edges.append((i, j))

    ei = _unique_edge_index(edges)
    if add_self_loops:
        ei = _add_self_loops_np(ei, F)
    return ei.astype(np.int64, copy=False)


class FastTGCNRunner(BaseRunner):
    """
    Fast-TGCN runner matching training recipe:
      - normalize mesh to unit sphere
      - twostream24 relative coordinates
      - face adjacency by shared vertices
      - self loops enabled
    """

    def __init__(self, models: List[BuiltModel], device: str | torch.device):
        if not models:
            raise ValueError("FastTGCNRunner requires at least 1 model")
        self.models = models
        self.device = torch.device(device) if not isinstance(device, torch.device) else device
        print(f"[Runner] fast_tgcn device='{self.device}' | num_models={len(models)}")

    @staticmethod
    def build(
        model_import: str,
        model_kwargs: Dict[str, Any],
        ckpt_path: str | List[str],
        device: str | torch.device = "cuda",
        *,
        app_root: Path | None = None,
    ) -> "FastTGCNRunner":
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

            # app version of helper returns only (missing, unexpected)
            missing, unexpected = load_checkpoint_to_model_best_effort(m, p)

            print(f"[fast_tgcn] ckpt={p}")
            print(f"[fast_tgcn] missing={len(missing)} unexpected={len(unexpected)}")
            if missing:
                print(f"[fast_tgcn] missing sample={missing[:10]}")
            if unexpected:
                print(f"[fast_tgcn] unexpected sample={unexpected[:10]}")

            built.append(BuiltModel(model=m, ckpt_path=p))

        return FastTGCNRunner(built, dev)

    @torch.no_grad()
    def infer(self, mesh: MeshData, *, fidx: np.ndarray | None) -> RunnerOutput:
        if mesh.faces is None:
            raise ValueError("FastTGCNRunner requires mesh.faces (triangles).")

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

        # Match train normalize=True
        v_norm, nmeta = recenter_scale_unit_sphere(v)

        centers, x_c_np, x_n_np, x_np = _build_twostream24_relative(v_norm, f)
        edge_index_np = _build_face_adj_vertex_edge_index(f, add_self_loops=True)

        x_c_t = torch.from_numpy(x_c_np[None, ...]).to(self.device, dtype=torch.float32)
        x_n_t = torch.from_numpy(x_n_np[None, ...]).to(self.device, dtype=torch.float32)
        x_t = torch.from_numpy(x_np[None, ...]).to(self.device, dtype=torch.float32)
        valid_face_t = torch.ones((1, f.shape[0]), dtype=torch.bool, device=self.device)
        edge_index_list = [torch.from_numpy(edge_index_np).to(self.device, dtype=torch.long)]

        batch = {
            "x_c": x_c_t,
            "x_n": x_n_t,
            "x": x_t,
            "valid_face": valid_face_t,
            "F_used": [int(f.shape[0])],
            "edge_index": edge_index_list,
        }

        print(
            "[fast_tgcn] infer batch:",
            f"x_c={tuple(x_c_t.shape)} x_n={tuple(x_n_t.shape)} x={tuple(x_t.shape)} "
            f"valid_face={tuple(valid_face_t.shape)} edge_index={tuple(edge_index_list[0].shape)} "
            f"F_used={batch['F_used']}"
        )

        probs_sum = None
        num_classes = None

        for bm in self.models:
            logits = bm.model(batch)  # expected (B,F,C)
            if logits.ndim != 3:
                raise RuntimeError(f"FastTGCN output must be (B,F,C). Got {tuple(logits.shape)}")

            if num_classes is None:
                num_classes = int(logits.shape[-1])

            probs = torch.softmax(logits, dim=-1)
            probs_sum = probs if probs_sum is None else (probs_sum + probs)

        probs_avg = (probs_sum / float(len(self.models))).squeeze(0)  # (F,C)
        labels = torch.argmax(probs_avg, dim=-1)                      # (F,)

        labels_np = labels.detach().long().cpu().numpy()
        probs_np = probs_avg.detach().float().cpu().numpy().astype(np.float16, copy=False)

        meta: Dict[str, Any] = {
            "runner": "fast_tgcn",
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
            "feature": "twostream24_relative",
            "graph_type": "face_adj",
            "adjacency_mode": "vertex",
            "add_self_loops": True,
            "centers_used": centers.astype(np.float32, copy=False),
            "probs_used": probs_np,
        }

        return RunnerOutput(
            labels_face=labels_np,
            num_classes=int(num_classes or 0),
            meta=meta,
        )