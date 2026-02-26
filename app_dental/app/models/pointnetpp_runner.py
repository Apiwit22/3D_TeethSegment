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
    """
    ✅ deploy-friendly: resolve ckpt path จาก app_root
    """
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


@dataclass
class BuiltModel:
    model: nn.Module
    ckpt_path: str


class PointNetPPRunner(BaseRunner):
    """
    PointNet++ runner (your run7 config):
      - num_points = 4096
      - point_feature = xyz_n (xyz + face normal) => 6 channels
      - num_classes = 17
    Output:
      - labels_face: (F,) for app color/export
      - meta contains centers_used + probs_used for postprocess
    """

    def __init__(
        self,
        models: List[BuiltModel],
        device: str | torch.device,
        num_points: int = 4096,
        point_feature: str = "xyz_n",
    ):
        if not models:
            raise ValueError("PointNetPPRunner requires at least 1 model")
        self.models = models
        self.device = torch.device(device) if not isinstance(device, torch.device) else device
        self.num_points = int(num_points)
        self.point_feature = str(point_feature).lower()

        print(
            f"[Runner] pointnetpp device='{self.device}' | num_models={len(models)} | "
            f"num_points={self.num_points} | feat={self.point_feature}"
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
        app_root: Path | None = None,  # ✅ เพิ่ม
    ) -> "PointNetPPRunner":
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

        return PointNetPPRunner(built, dev, num_points=int(num_points), point_feature=str(point_feature))

    @torch.no_grad()
    def infer(self, mesh: MeshData, *, fidx: np.ndarray | None) -> RunnerOutput:
        if mesh.faces is None:
            raise ValueError("PointNetPPRunner requires mesh.faces (triangles).")

        v = np.asarray(mesh.pos, dtype=np.float32)
        f_all = np.asarray(mesh.faces, dtype=np.int64)
        F_total = int(f_all.shape[0])

        # optional subset faces (unused now, keep compatible)
        if fidx is not None:
            fidx = np.asarray(fidx, dtype=np.int64).reshape(-1)
            f = f_all[fidx]
        else:
            f = f_all
            fidx = None

        # normalize to unit sphere (same as other runners)
        v_norm, nmeta = recenter_scale_unit_sphere(v)

        # surface sample points
        tm = trimesh.Trimesh(vertices=v_norm, faces=f, process=False)
        P, face_ids = trimesh.sample.sample_surface(tm, self.num_points)  # (N,3), (N,)
        P = np.asarray(P, dtype=np.float32)
        face_ids = np.asarray(face_ids, dtype=np.int64)

        # normals: use face normals of sampled faces
        Nf = np.asarray(tm.face_normals, dtype=np.float32)[face_ids]  # (N,3)

        if self.point_feature in ("xyz",):
            x_np = P
        elif self.point_feature in ("xyz_n", "xyzn", "xyz+normal", "xyz_normals"):
            x_np = np.concatenate([P, Nf], axis=1).astype(np.float32, copy=False)  # (N,6)
        else:
            raise ValueError(f"Unknown point_feature={self.point_feature}")

        x_t = torch.from_numpy(x_np[None, ...]).to(self.device, dtype=torch.float32)  # (1,N,C)

        logits_sum = None
        num_classes = None
        for bm in self.models:
            logits = bm.model(x_t)  # (1,N,C)
            if logits.ndim != 3:
                raise RuntimeError(f"PointNet++ output must be (B,N,C). Got {tuple(logits.shape)}")
            if num_classes is None:
                num_classes = int(logits.shape[-1])
            logits_sum = logits if logits_sum is None else (logits_sum + logits)

        logits_avg = logits_sum / float(len(self.models))
        probs = torch.softmax(logits_avg, dim=-1)[0].float().cpu().numpy()  # (N,C)
        C = int(probs.shape[1])

        # map point -> face by mean prob per face (best match to your training label_source=face)
        F_used = int(f.shape[0])
        sum_prob = np.zeros((F_used, C), dtype=np.float32)
        cnt = np.zeros((F_used,), dtype=np.int32)
        np.add.at(sum_prob, face_ids, probs)
        np.add.at(cnt, face_ids, 1)

        face_probs = np.zeros((F_used, C), dtype=np.float32)
        has = cnt > 0
        face_probs[has] = sum_prob[has] / cnt[has][:, None]

        face_centers = _face_centers(v_norm, f)

        # fallback: faces with no sampled points -> nearest point to center
        if np.any(~has):
            tree = cKDTree(P)
            _, nn_idx = tree.query(face_centers[~has], k=1, workers=-1)
            nn_idx = np.asarray(nn_idx, dtype=np.int64)
            face_probs[~has] = probs[nn_idx]

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
            "centers_used": face_centers.astype(np.float32, copy=False),
            "probs_used": face_probs.astype(np.float16, copy=False),
        }

        return RunnerOutput(labels_face=labels_face, num_classes=int(num_classes or C), meta=meta)