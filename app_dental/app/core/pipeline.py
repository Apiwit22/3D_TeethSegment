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

# ✅ NEW: tsmdl runner
from app.models.tsmdl_runner import TSMdlRunner

# Postprocess
from app.data.postprocess import (
    postprocess_labels_conservative,
    postprocess_labels_best_visual,
)


def _resolve_from_app_root(app_root: Path, p: str) -> Path:
    pp = Path(p)
    return pp if pp.is_absolute() else (Path(app_root) / pp).resolve()


def _resolve_many_from_app_root(app_root: Path, paths: List[str]) -> Tuple[List[str], List[str]]:
    exist_abs: List[str] = []
    missing_abs: List[str] = []
    for p in paths:
        pp = _resolve_from_app_root(app_root, str(p))
        if pp.exists():
            exist_abs.append(str(pp))
        else:
            missing_abs.append(str(pp))
    return exist_abs, missing_abs


def _get(preset: Preset, name: str, default):
    return getattr(preset, name, default)


@dataclass
class Pipeline:
    """
    One-preset pipeline (MVP)
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

        # 2) remesh to target faces
        rr = remesh_to_faces(mesh_in.pos, mesh_in.faces, target_faces=int(self.preset.target_faces))
        meta["remesh"] = rr.meta
        v = rr.vertices
        f = rr.faces

        # 3) orient to reference (geometry-only)
        if self.preset.do_orient and self.preset.ref_path:
            ref_lower = _resolve_from_app_root(app_root, self.preset.ref_path)

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

        # 4) checkpoints (resolve from app_root)
        dev = self._pick_device()

        ckpts_raw = [str(x) for x in (self.preset.ckpts or [])]
        ckpts_exist_abs, ckpts_missing_abs = _resolve_many_from_app_root(app_root, ckpts_raw)

        if not ckpts_exist_abs:
            raise FileNotFoundError(f"No checkpoint found. Missing (resolved): {ckpts_missing_abs[:3]} ...")

        meta["ckpt_missing"] = ckpts_missing_abs
        meta["ckpt_exist"] = ckpts_exist_abs

        # 5) build runner
        if self.preset.runner == "tsmdl":
            runner = TSMdlRunner.build(
                model_import=self.preset.model_import,
                model_kwargs=self.preset.model_kwargs,
                ckpt_path=ckpts_exist_abs,
                device=dev,
                app_root=app_root,
            )

        elif self.preset.runner == "pointnetpp":
            pt_det = bool(_get(self.preset, "pt_deterministic_sampling", True))
            pt_seed = int(_get(self.preset, "pt_sample_seed", 1234))
            pt_mc = int(_get(self.preset, "pt_mc_passes", 1))

            # (optional) cover faces parameters if you later wire them in pointnetpp_runner
            pt_cover = bool(_get(self.preset, "pt_cover_faces", True))
            pt_cover_jitter = float(_get(self.preset, "pt_cover_jitter", 0.0))

            runner = PointNetPPRunner.build(
                model_import=self.preset.model_import,
                model_kwargs=self.preset.model_kwargs,
                ckpt_path=ckpts_exist_abs,
                device=dev,
                num_points=int(self.preset.num_points),
                point_feature=str(self.preset.point_feature),
                app_root=app_root,
                deterministic_sampling=pt_det,
                sample_seed=pt_seed,
                mc_passes=pt_mc,
                cover_faces=pt_cover,
                cover_jitter=pt_cover_jitter,
            )
            meta["pointnetpp_cfg"] = {
                "deterministic_sampling": pt_det,
                "sample_seed": pt_seed,
                "mc_passes": pt_mc,
                "cover_faces": pt_cover,
                "cover_jitter": pt_cover_jitter,
                "num_points": int(self.preset.num_points),
                "point_feature": str(self.preset.point_feature),
            }

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

        # 7) postprocess
        if self.preset.do_postprocess:
            centers = out.meta.get("centers_used", None)
            probs = out.meta.get("probs_used", None)

            if centers is not None:
                centers = np.asarray(centers, dtype=np.float32)
                probs_np = (np.asarray(probs, dtype=np.float32) if probs is not None else None)

                k = int(_get(self.preset, "pp_knn_k", 16))
                smooth_iters = int(_get(self.preset, "pp_smooth_iters", 0))

                ungingiva_margin = float(_get(self.preset, "pp_ungingiva_margin", 0.05))
                ungingiva_min_tooth_p = float(_get(self.preset, "pp_ungingiva_min_tooth_p", 0.12))
                keep_gingiva_lcc = bool(_get(self.preset, "pp_keep_gingiva_lcc", True))

                conf_thresh = float(_get(self.preset, "pp_conf_thresh", 0.60))
                conf_neighbor_iters = int(_get(self.preset, "pp_conf_neighbor_iters", 1))

                min_comp_size = int(_get(self.preset, "pp_min_comp_size", 30))
                clean_iters = int(_get(self.preset, "pp_clean_iters", 1))

                pp_mode = str(_get(self.preset, "pp_mode", "best_visual")).lower()

                if pp_mode == "conservative":
                    labels = postprocess_labels_conservative(
                        labels=labels,
                        centers=centers,
                        probs=probs_np,
                        num_classes=num_classes,
                        k=k,
                        smooth_iters=max(smooth_iters, 1),
                        ungingiva_margin=ungingiva_margin,
                        ungingiva_min_tooth_p=ungingiva_min_tooth_p,
                        keep_gingiva_lcc=keep_gingiva_lcc,
                    )
                    meta["postprocess"] = {"enabled": True, "mode": "conservative"}
                else:
                    labels = postprocess_labels_best_visual(
                        labels=labels,
                        centers=centers,
                        probs=probs_np,
                        num_classes=num_classes,
                        gingiva_label=16,
                        k=k,
                        conf_thresh=conf_thresh,
                        conf_neighbor_iters=conf_neighbor_iters,
                        min_comp_size=min_comp_size,
                        clean_iters=clean_iters,
                        smooth_iters=smooth_iters,
                        keep_gingiva_lcc=keep_gingiva_lcc,
                        ungingiva_margin=ungingiva_margin,
                        ungingiva_min_tooth_p=ungingiva_min_tooth_p,
                    )
                    meta["postprocess"] = {
                        "enabled": True,
                        "mode": "best_visual",
                        "k": k,
                        "smooth_iters": smooth_iters,
                        "conf_thresh": conf_thresh,
                        "conf_neighbor_iters": conf_neighbor_iters,
                        "min_comp_size": min_comp_size,
                        "clean_iters": clean_iters,
                    }
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