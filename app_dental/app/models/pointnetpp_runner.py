# dental_seg_app/app/models/pointnetpp_runner.py
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

# reuse loader/utils you already have in meshsegnet_runner
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
    # save RNG state
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
    """
    ✅ Coverage sampling: guarantee at least 1 sample per face.
    For each triangle face:
      P = a*v0 + b*v1 + c*v2 where (a,b,c) sampled uniformly on triangle.
    Returns:
      P: (F,3)
      face_ids: (F,) = [0..F-1]
    """
    v = np.asarray(tm.vertices, dtype=np.float32)
    f = np.asarray(tm.faces, dtype=np.int64)
    F = int(f.shape[0])

    rng = np.random.default_rng(int(seed) & 0xFFFFFFFF)

    v0 = v[f[:, 0]]
    v1 = v[f[:, 1]]
    v2 = v[f[:, 2]]

    # Uniform sample on triangle using sqrt trick
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


class PointNetPPRunner(BaseRunner):
    """
    PointNet++ runner:
      - sample points on surface
      - predict per-point
      - aggregate to per-face by mean prob per face
    Output:
      - labels_face: (F,)
      - meta contains centers_used + probs_used for postprocess
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
        # ✅ NEW: guarantee coverage of every face
        cover_faces: bool = True,
        cover_jitter: float = 0.0,  # optional small jitter in unit-sphere space
    ):
        if not models:
            raise ValueError("PointNetPPRunner requires at least 1 model")
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
            f"[Runner] pointnetpp device='{self.device}' | num_models={len(models)} | "
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
        # ✅ NEW
        cover_faces: bool = True,
        cover_jitter: float = 0.0,
    ) -> "PointNetPPRunner":
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

        return PointNetPPRunner(
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
            raise ValueError("PointNetPPRunner requires mesh.faces (triangles).")

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

        # accumulate across MC passes
        C_final = None
        sum_prob_face = None
        cnt_face = np.zeros((F_used,), dtype=np.int32)

        # collect sampled points for fallback KDTree (should rarely be needed if cover_faces=True)
        all_P: List[np.ndarray] = []
        all_probs: List[np.ndarray] = []

        for t in range(max(self.mc_passes, 1)):
            seed = self.sample_seed + t

            # ---- (1) coverage samples: one point per face ----
            if self.cover_faces:
                P_cov, face_cov = _sample_one_point_per_face(tm, seed=seed)
                if self.cover_jitter > 0:
                    rng = np.random.default_rng((seed + 99991) & 0xFFFFFFFF)
                    P_cov = P_cov + rng.normal(0.0, self.cover_jitter, size=P_cov.shape).astype(np.float32)

            else:
                P_cov = np.zeros((0, 3), dtype=np.float32)
                face_cov = np.zeros((0,), dtype=np.int64)

            # ---- (2) extra random samples to reach num_points ----
            n_cov = int(P_cov.shape[0])
            n_extra = max(int(self.num_points) - n_cov, 0)

            if n_extra > 0:
                if self.deterministic_sampling:
                    P_ex, face_ex = _sample_surface_deterministic(tm, n_extra, seed=seed)
                else:
                    P_ex, face_ex = trimesh.sample.sample_surface(tm, n_extra)
                    P_ex = np.asarray(P_ex, dtype=np.float32)
                    face_ex = np.asarray(face_ex, dtype=np.int64)
                P = np.concatenate([P_cov, P_ex], axis=0)
                face_ids = np.concatenate([face_cov, face_ex], axis=0)
            else:
                # if num_points < F_used, we still keep coverage points (>=F_used)
                P = P_cov
                face_ids = face_cov

            # normals: use face normals of sampled faces
            Nf = np.asarray(tm.face_normals, dtype=np.float32)[face_ids]

            if self.point_feature in ("xyz",):
                x_np = P
            elif self.point_feature in ("xyz_n", "xyzn", "xyz+normal", "xyz_normals"):
                x_np = np.concatenate([P, Nf], axis=1).astype(np.float32, copy=False)  # (N,6)
            else:
                raise ValueError(f"Unknown point_feature={self.point_feature}")

            x_t = torch.from_numpy(x_np[None, ...]).to(self.device, dtype=torch.float32)  # (1,N,Cin)

            logits_sum = None
            num_classes = None
            for bm in self.models:
                logits = bm.model(x_t)  # expected (1,N,C)
                if logits.ndim != 3:
                    raise RuntimeError(f"PointNet++ output must be (B,N,C). Got {tuple(logits.shape)}")
                if num_classes is None:
                    num_classes = int(logits.shape[-1])
                logits_sum = logits if logits_sum is None else (logits_sum + logits)

            logits_avg = logits_sum / float(len(self.models))
            probs = torch.softmax(logits_avg, dim=-1)[0].float().cpu().numpy()  # (N,C)
            C = int(probs.shape[1])
            C_final = C if C_final is None else C_final

            if sum_prob_face is None:
                sum_prob_face = np.zeros((F_used, C), dtype=np.float32)

            # aggregate to face (mean prob)
            np.add.at(sum_prob_face, face_ids, probs)
            np.add.at(cnt_face, face_ids, 1)

            all_P.append(P.astype(np.float32, copy=False))
            all_probs.append(probs.astype(np.float32, copy=False))

        assert sum_prob_face is not None and C_final is not None

        face_probs = np.zeros((F_used, C_final), dtype=np.float32)
        has = cnt_face > 0
        face_probs[has] = sum_prob_face[has] / cnt_face[has][:, None]

        face_centers = _face_centers(v_norm, f)

        # fallback for any faces with no samples (should be none if cover_faces=True)
        if np.any(~has):
            P_all = np.concatenate(all_P, axis=0).astype(np.float32, copy=False)
            probs_all = np.concatenate(all_probs, axis=0).astype(np.float32, copy=False)
            tree = cKDTree(P_all)
            _, nn_idx = tree.query(face_centers[~has], k=1, workers=-1)
            nn_idx = np.asarray(nn_idx, dtype=np.int64)
            face_probs[~has] = probs_all[nn_idx]

        labels_face = np.argmax(face_probs, axis=1).astype(np.int64)

        meta = {
            "runner": "pointnetpp",
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
            # ✅ new
            "cover_faces": bool(self.cover_faces),
            "cover_jitter": float(self.cover_jitter),
            # for postprocess
            "centers_used": face_centers.astype(np.float32, copy=False),
            "probs_used": face_probs.astype(np.float16, copy=False),
        }

        return RunnerOutput(labels_face=labels_face, num_classes=int(C_final), meta=meta)