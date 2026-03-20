from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn as nn
import trimesh
from scipy.spatial import cKDTree

from app.core.types import MeshData
from app.models.base import BaseRunner, RunnerOutput
from app.models.meshsegnet_runner import (
    import_symbol,
    load_checkpoint_to_model_best_effort,
    recenter_scale_unit_sphere,
)


def _resolve_ckpt_paths(app_root: Path | None, ckpt_path: str | List[str]) -> List[str]:
    ckpts: List[str] = [str(p) for p in ckpt_path] if isinstance(ckpt_path, (list, tuple)) else [str(ckpt_path)]
    out: List[str] = []
    for p in ckpts:
        pp = Path(p)
        if not pp.is_absolute():
            pp = (Path(app_root) / pp).resolve() if app_root is not None else pp.resolve()
        out.append(str(pp))
    return out


def _face_centers(v: np.ndarray, f: np.ndarray) -> np.ndarray:
    v0 = v[f[:, 0]]
    v1 = v[f[:, 1]]
    v2 = v[f[:, 2]]
    return ((v0 + v1 + v2) / 3.0).astype(np.float32)


def _sample_surface_deterministic(tm: trimesh.Trimesh, n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    state = np.random.get_state()
    np.random.seed(int(seed) & 0xFFFFFFFF)
    try:
        P, face_ids = trimesh.sample.sample_surface(tm, int(n))
        P = np.asarray(P, dtype=np.float32)
        face_ids = np.asarray(face_ids, dtype=np.int64)
        return P, face_ids
    finally:
        np.random.set_state(state)


def _sample_one_point_per_face(tm: trimesh.Trimesh, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    v = np.asarray(tm.vertices, dtype=np.float32)
    f = np.asarray(tm.faces, dtype=np.int64)
    F = int(f.shape[0])

    rng = np.random.default_rng(int(seed) & 0xFFFFFFFF)

    v0 = v[f[:, 0]]
    v1 = v[f[:, 1]]
    v2 = v[f[:, 2]]

    r1 = rng.random((F, 1), dtype=np.float32)
    r2 = rng.random((F, 1), dtype=np.float32)
    sr1 = np.sqrt(r1)

    a = 1.0 - sr1
    b = sr1 * (1.0 - r2)
    c = sr1 * r2

    P = (a * v0) + (b * v1) + (c * v2)
    face_ids = np.arange(F, dtype=np.int64)
    return P.astype(np.float32, copy=False), face_ids


@dataclass
class BuiltModel:
    model: nn.Module
    ckpt_path: str


class PointCNNRunner(BaseRunner):
    """
    PointCNN runner:
      - sample points on surface
      - build x + pos
      - predict per-point
      - aggregate to per-face by mean prob
    """

    def __init__(
        self,
        models: List[BuiltModel],
        device: str | torch.device,
        num_points: int = 4096,
        point_feature: str = "xyz_n",
        *,
        deterministic_sampling: bool = True,
        sample_seed: int = 1234,
        mc_passes: int = 1,
        cover_faces: bool = True,
        cover_jitter: float = 0.0,
    ):
        if not models:
            raise ValueError("PointCNNRunner requires at least 1 model")
        self.models = models
        self.device = torch.device(device) if not isinstance(device, torch.device) else device
        self.num_points = int(num_points)
        self.point_feature = str(point_feature).lower()

        self.deterministic_sampling = bool(deterministic_sampling)
        self.sample_seed = int(sample_seed)
        self.mc_passes = int(mc_passes)

        self.cover_faces = bool(cover_faces)
        self.cover_jitter = float(cover_jitter)

        print(
            f"[Runner] pointcnn device='{self.device}' | num_models={len(models)} | "
            f"num_points={self.num_points} | feat={self.point_feature} | "
            f"deterministic={self.deterministic_sampling} seed={self.sample_seed} mc_passes={self.mc_passes} | "
            f"cover_faces={self.cover_faces} cover_jitter={self.cover_jitter}"
        )

    @staticmethod
    def build(
        model_import: str,
        model_kwargs: Dict[str, Any],
        ckpt_path: str | List[str],
        device: str | torch.device = "cuda",
        *,
        num_points: int = 4096,
        point_feature: str = "xyz_n",
        app_root: Path | None = None,
        deterministic_sampling: bool = True,
        sample_seed: int = 1234,
        mc_passes: int = 1,
        cover_faces: bool = True,
        cover_jitter: float = 0.0,
    ) -> "PointCNNRunner":
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

        return PointCNNRunner(
            built,
            dev,
            num_points=int(num_points),
            point_feature=str(point_feature),
            deterministic_sampling=bool(deterministic_sampling),
            sample_seed=int(sample_seed),
            mc_passes=int(mc_passes),
            cover_faces=bool(cover_faces),
            cover_jitter=float(cover_jitter),
        )

    @torch.no_grad()
    def infer(self, mesh: MeshData, *, fidx: np.ndarray | None) -> RunnerOutput:
        if mesh.faces is None:
            raise ValueError("PointCNNRunner requires mesh.faces (triangles).")

        v = np.asarray(mesh.pos, dtype=np.float32)
        f_all = np.asarray(mesh.faces, dtype=np.int64)
        F_total = int(f_all.shape[0])

        if fidx is not None:
            fidx = np.asarray(fidx, dtype=np.int64).reshape(-1)
            f = f_all[fidx]
        else:
            f = f_all
            fidx = None

        v_norm, nmeta = recenter_scale_unit_sphere(v)
        tm = trimesh.Trimesh(vertices=v_norm, faces=f, process=False)

        F_used = int(f.shape[0])

        C_final = None
        sum_prob_face = None
        cnt_face = np.zeros((F_used,), dtype=np.int32)

        all_P: List[np.ndarray] = []
        all_probs: List[np.ndarray] = []

        for t in range(max(self.mc_passes, 1)):
            seed = self.sample_seed + t

            if self.cover_faces:
                P_cov, face_cov = _sample_one_point_per_face(tm, seed=seed)
            else:
                P_cov = np.zeros((0, 3), dtype=np.float32)
                face_cov = np.zeros((0,), dtype=np.int64)

            n_extra = max(self.num_points - int(P_cov.shape[0]), 0)
            if n_extra > 0:
                if self.deterministic_sampling:
                    P_extra, face_extra = _sample_surface_deterministic(tm, n_extra, seed=seed + 7919)
                else:
                    P_extra, face_extra = trimesh.sample.sample_surface(tm, n_extra)
                    P_extra = np.asarray(P_extra, dtype=np.float32)
                    face_extra = np.asarray(face_extra, dtype=np.int64)
                P = np.concatenate([P_cov, P_extra], axis=0).astype(np.float32, copy=False)
                face_ids = np.concatenate([face_cov, face_extra], axis=0).astype(np.int64, copy=False)
            else:
                P = P_cov.astype(np.float32, copy=False)
                face_ids = face_cov.astype(np.int64, copy=False)

            if self.cover_jitter > 0:
                rng = np.random.default_rng((seed * 1103515245 + 12345) & 0xFFFFFFFF)
                jitter = rng.normal(0.0, self.cover_jitter, size=P.shape).astype(np.float32)
                P = P + jitter

            # point normals from nearest vertex normal
            vnormals = np.asarray(tm.vertex_normals, dtype=np.float32)
            tree_v = cKDTree(np.asarray(tm.vertices, dtype=np.float32))
            _, nnv = tree_v.query(P, k=1, workers=-1)
            nnv = np.asarray(nnv, dtype=np.int64)
            Np = vnormals[nnv].astype(np.float32, copy=False)

            if self.point_feature == "xyz":
                x_np = P.astype(np.float32, copy=False)
            elif self.point_feature in {"xyz_n", "xyzn", "xyz+normal", "xyz_normal"}:
                x_np = np.concatenate([P, Np], axis=1).astype(np.float32, copy=False)
            else:
                raise ValueError(f"Unsupported point_feature for PointCNNRunner: {self.point_feature!r}")

            pos_t = torch.from_numpy(P[None, ...]).to(self.device, dtype=torch.float32)
            x_t = torch.from_numpy(x_np[None, ...]).to(self.device, dtype=torch.float32)
            batch = {"pos": pos_t, "x": x_t}

            probs_sum_point = None
            C = None

            for bm in self.models:
                logits = bm.model(batch)
                if logits.ndim != 3:
                    raise RuntimeError(f"PointCNN output must be (B,N,C). Got {tuple(logits.shape)}")
                if C is None:
                    C = int(logits.shape[-1])
                probs = torch.softmax(logits, dim=-1)
                probs_sum_point = probs if probs_sum_point is None else (probs_sum_point + probs)

            probs_avg_point = (probs_sum_point / float(len(self.models))).squeeze(0)  # (N,C)
            probs = probs_avg_point.detach().float().cpu().numpy().astype(np.float32, copy=False)

            if C_final is None:
                C_final = int(C)
            if sum_prob_face is None:
                sum_prob_face = np.zeros((F_used, C_final), dtype=np.float32)

            np.add.at(sum_prob_face, face_ids, probs)
            np.add.at(cnt_face, face_ids, 1)

            all_P.append(P.astype(np.float32, copy=False))
            all_probs.append(probs.astype(np.float32, copy=False))

        assert sum_prob_face is not None and C_final is not None

        face_probs = np.zeros((F_used, C_final), dtype=np.float32)
        has = cnt_face > 0
        face_probs[has] = sum_prob_face[has] / cnt_face[has][:, None]

        face_centers = _face_centers(v_norm, f)

        if np.any(~has):
            P_all = np.concatenate(all_P, axis=0).astype(np.float32, copy=False)
            probs_all = np.concatenate(all_probs, axis=0).astype(np.float32, copy=False)
            tree = cKDTree(P_all)
            _, nn_idx = tree.query(face_centers[~has], k=1, workers=-1)
            nn_idx = np.asarray(nn_idx, dtype=np.int64)
            face_probs[~has] = probs_all[nn_idx]

        labels_face = np.argmax(face_probs, axis=1).astype(np.int64)

        meta = {
            "runner": "pointcnn",
            "device": str(self.device),
            "arch": mesh.arch,
            "path": mesh.path,
            "num_models": int(len(self.models)),
            "ckpt_paths": [bm.ckpt_path for bm in self.models],
            "normalize_meta": nmeta,
            "faces_total": F_total,
            "faces_used": int(F_used),
            "fidx": (fidx.tolist() if fidx is not None else None),
            "num_points": int(self.num_points),
            "point_feature": self.point_feature,
            "deterministic_sampling": bool(self.deterministic_sampling),
            "sample_seed": int(self.sample_seed),
            "mc_passes": int(self.mc_passes),
            "cover_faces": bool(self.cover_faces),
            "cover_jitter": float(self.cover_jitter),
            "centers_used": face_centers.astype(np.float32, copy=False),
            "probs_used": face_probs.astype(np.float16, copy=False),
        }

        return RunnerOutput(labels_face=labels_face, num_classes=int(C_final), meta=meta)