# dental_seg_app/app/core/pipeline.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

from app.core.types import MeshData, PipelineResult
from app.core.registry import Preset
from app.io.mesh_loader import load_mesh
from app.preprocess.remesh import remesh_to_faces
from app.preprocess.orient import OrientConfig, orient_mesh_to_reference
from app.models.meshsegnet_runner import MeshSegNetRunner
from app.models.pointnetpp_runner import PointNetPPRunner
from app.data.postprocess import postprocess_labels_conservative


def _resolve_from_app_root(app_root: Path, p: str) -> Path:
    """
    resolve path จาก app_root เสมอ (deploy-friendly)
    - absolute -> ใช้ตรง ๆ
    - relative -> app_root / p
    """
    pp = Path(p)
    return pp if pp.is_absolute() else (Path(app_root) / pp).resolve()


def _resolve_many_from_app_root(app_root: Path, paths: List[str]) -> Tuple[List[str], List[str]]:
    """
    รับ list ของ path (string) แล้วคืน:
      - exist_abs: list[str] ของ absolute path ที่มีจริง
      - missing_abs: list[str] ของ absolute path ที่หาไม่เจอ
    """
    exist_abs: List[str] = []
    missing_abs: List[str] = []
    for p in paths:
        pp = _resolve_from_app_root(app_root, str(p))
        if pp.exists():
            exist_abs.append(str(pp))
        else:
            missing_abs.append(str(pp))
    return exist_abs, missing_abs


@dataclass
class Pipeline:
    """
    One-preset pipeline (MVP)
    ✅ เพิ่ม app_root เพื่อ resolve ref_path/ckpts ให้ถูก base เสมอ
    """
    app_root: Path
    preset: Preset
    device: str = "cuda"

    def _pick_device(self) -> str:
        if self.device == "cuda" and torch.cuda.is_available():
            return "cuda"
        return "cpu"

    def run(self, input_path: str | Path) -> PipelineResult:
        p = Path(input_path)
        app_root = Path(self.app_root)

        # 1) load
        m0 = load_mesh(p)
        mesh_in = MeshData(pos=m0.vertices, faces=m0.faces, path=str(p), arch=self.preset.arch)

        meta: Dict[str, Any] = {"input": str(p), "preset": self.preset.key, "runner": self.preset.runner}

        # 2) remesh to target faces (kept for deterministic export + PP)
        rr = remesh_to_faces(mesh_in.pos, mesh_in.faces, target_faces=int(self.preset.target_faces))
        meta["remesh"] = rr.meta
        v = rr.vertices
        f = rr.faces

        # 3) orient to reference (geometry-only)
        if self.preset.do_orient and self.preset.ref_path:
            # ✅ resolve ref_path จาก app_root
            ref_lower = _resolve_from_app_root(app_root, self.preset.ref_path)

            # กรณีใน yaml ชี้ U หรือ L เราสร้างคู่ให้ครบแบบเดิมของเธอ
            # (ยังคง behavior เดิม: replace ชื่อไฟล์)
            ref_upper = Path(str(ref_lower).replace("007_L.ply", "007_U.ply"))
            ref_lower2 = Path(str(ref_lower).replace("007_U.ply", "007_L.ply"))

            if not ref_upper.exists():
                raise FileNotFoundError(f"Missing reference (upper): {ref_upper}")
            if not ref_lower2.exists():
                raise FileNotFoundError(f"Missing reference (lower): {ref_lower2}")

            ocfg = OrientConfig(
                ref_upper=ref_upper,
                ref_lower=ref_lower2,
                sample_n=30000,
                seed=1234,
                allow_reflection=bool(self.preset.allow_reflection),
            )
            v2, o_meta = orient_mesh_to_reference(v, arch=self.preset.arch, cfg=ocfg)
            v = v2
            meta["orient"] = o_meta
        else:
            meta["orient"] = {"enabled": False}

        mesh_proc = MeshData(
            pos=v.astype(np.float32, copy=False),
            faces=f.astype(np.int64, copy=False),
            path=str(p),
            arch=self.preset.arch,
        )

        # 4) checkpoints (✅ resolve ckpts จาก app_root)
        dev = self._pick_device()

        ckpts_raw = [str(x) for x in (self.preset.ckpts or [])]
        ckpts_exist_abs, ckpts_missing_abs = _resolve_many_from_app_root(app_root, ckpts_raw)

        if not ckpts_exist_abs:
            # โชว์ตัวอย่าง 3 อันแรกให้พอเห็นว่า resolve ไปที่ไหน
            raise FileNotFoundError(f"No checkpoint found. Missing (resolved): {ckpts_missing_abs[:3]} ...")

        meta["ckpt_missing"] = ckpts_missing_abs
        meta["ckpt_exist"] = ckpts_exist_abs

        # 5) build runner (ส่ง app_root ไปด้วย เผื่อ runner จะ resolve ซ้ำอีกชั้น)
        if self.preset.runner == "pointnetpp":
            runner = PointNetPPRunner.build(
                model_import=self.preset.model_import,
                model_kwargs=self.preset.model_kwargs,
                ckpt_path=ckpts_exist_abs,
                device=dev,
                num_points=int(self.preset.num_points),
                point_feature=str(self.preset.point_feature),
                app_root=app_root,
            )
        else:
            runner = MeshSegNetRunner.build(
                model_import=self.preset.model_import,
                model_kwargs=self.preset.model_kwargs,
                ckpt_path=ckpts_exist_abs,
                device=dev,
                app_root=app_root,
            )

        # 6) inference
        out = runner.infer(mesh_proc, fidx=None)
        labels = out.labels_face
        num_classes = int(out.num_classes)
        meta["infer"] = out.meta

        # 7) postprocess (optional)
        if self.preset.do_postprocess:
            centers = out.meta.get("centers_used", None)
            probs = out.meta.get("probs_used", None)

            if centers is not None:
                labels = postprocess_labels_conservative(
                    labels=labels,
                    centers=np.asarray(centers, dtype=np.float32),
                    probs=(np.asarray(probs, dtype=np.float32) if probs is not None else None),
                    num_classes=num_classes,
                    k=int(self.preset.pp_knn_k),
                    smooth_iters=int(self.preset.pp_smooth_iters),
                    ungingiva_margin=float(self.preset.pp_ungingiva_margin),
                    ungingiva_min_tooth_p=float(self.preset.pp_ungingiva_min_tooth_p),
                    keep_gingiva_lcc=bool(self.preset.pp_keep_gingiva_lcc),
                )
                meta["postprocess"] = {"enabled": True}
            else:
                meta["postprocess"] = {"enabled": False, "reason": "missing centers_used"}
        else:
            meta["postprocess"] = {"enabled": False}

        return PipelineResult(
            mesh_in=mesh_in,
            mesh_proc=mesh_proc,
            labels_face=np.asarray(labels, dtype=np.int64),
            num_classes=num_classes,
            meta=meta,
        )