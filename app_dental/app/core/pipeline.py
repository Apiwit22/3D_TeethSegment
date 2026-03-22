from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch

from app.core.postprocess import run_postprocess
from app.core.registry import Preset
from app.core.types import MeshData, PipelineResult
from app.io.mesh_loader import load_mesh
from app.preprocess.orient import OrientConfig, orient_mesh_to_reference
from app.preprocess.remesh import remesh_to_faces

from app.models.fast_tgcn_runner import FastTGCNRunner
from app.models.meshsegnet_runner import MeshSegNetRunner
from app.models.pointcnn_runner import PointCNNRunner
from app.models.pointnetpp_runner import PointNetPPRunner
from app.models.tsmdl_runner import TSMdlRunner


ProgressCB = Optional[Callable[[int], None]]
LogCB = Optional[Callable[[str], None]]


def _resolve_from_app_root(app_root: Path, p: str | Path) -> Path:
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


def _infer_ref_pair_from_single_path(ref_path: Path) -> Tuple[Path, Path]:
    s = str(ref_path)
    if "007_L.ply" in s:
        return Path(s.replace("007_L.ply", "007_U.ply")), Path(s)
    if "007_U.ply" in s:
        return Path(s), Path(s.replace("007_U.ply", "007_L.ply"))
    raise ValueError(
        "Cannot infer upper/lower reference pair from ref_path. "
        "Use ref_upper_path and ref_lower_path explicitly in registry.yaml."
    )


def _resolve_reference_paths(app_root: Path, preset: Preset) -> Tuple[Path, Path]:
    if preset.ref_upper_path and preset.ref_lower_path:
        ref_upper = _resolve_from_app_root(app_root, preset.ref_upper_path)
        ref_lower = _resolve_from_app_root(app_root, preset.ref_lower_path)
        return ref_upper, ref_lower

    if preset.ref_path:
        ref_path = _resolve_from_app_root(app_root, preset.ref_path)
        return _infer_ref_pair_from_single_path(ref_path)

    raise ValueError(
        f"Preset '{preset.key}' has do_orient=True but no usable reference path configuration"
    )


def _build_runner(app_root: Path, preset: Preset, ckpts_exist_abs: List[str], device: str):
    runner = preset.runner.lower()

    if runner == "tsmdl":
        return TSMdlRunner.build(
            model_import=preset.model_import,
            model_kwargs=preset.model_kwargs,
            ckpt_path=ckpts_exist_abs,
            device=device,
            app_root=app_root,
        )

    if runner == "pointnetpp":
        return PointNetPPRunner.build(
            model_import=preset.model_import,
            model_kwargs=preset.model_kwargs,
            ckpt_path=ckpts_exist_abs,
            device=device,
            num_points=int(preset.num_points),
            point_feature=str(preset.point_feature),
            app_root=app_root,
            deterministic_sampling=bool(preset.pt_deterministic_sampling),
            sample_seed=int(preset.pt_sample_seed),
            mc_passes=int(preset.pt_mc_passes),
            cover_faces=bool(preset.pt_cover_faces),
            cover_jitter=float(preset.pt_cover_jitter),
        )

    if runner == "pointcnn":
        return PointCNNRunner.build(
            model_import=preset.model_import,
            model_kwargs=preset.model_kwargs,
            ckpt_path=ckpts_exist_abs,
            device=device,
            num_points=int(preset.num_points),
            point_feature=str(preset.point_feature),
            app_root=app_root,
            deterministic_sampling=bool(preset.pt_deterministic_sampling),
            sample_seed=int(preset.pt_sample_seed),
            mc_passes=int(preset.pt_mc_passes),
            cover_faces=bool(preset.pt_cover_faces),
            cover_jitter=float(preset.pt_cover_jitter),
        )

    if runner == "meshsegnet":
        return MeshSegNetRunner.build(
            model_import=preset.model_import,
            model_kwargs=preset.model_kwargs,
            ckpt_path=ckpts_exist_abs,
            device=device,
            app_root=app_root,
        )

    if runner == "fast_tgcn":
        return FastTGCNRunner.build(
            model_import=preset.model_import,
            model_kwargs=preset.model_kwargs,
            ckpt_path=ckpts_exist_abs,
            device=device,
            app_root=app_root,
        )

    raise ValueError(f"Unsupported runner: {preset.runner}")


@dataclass
class Pipeline:
    """
    One-preset pipeline.
    """
    app_root: Path
    preset: Preset
    device: str = "cuda"

    def _pick_device(self) -> str:
        if self.device == "cuda" and torch.cuda.is_available():
            return "cuda"
        return "cpu"

    def run(
        self,
        input_path: str | Path,
        progress_cb: ProgressCB = None,
        log_cb: LogCB = None,
    ) -> PipelineResult:
        def set_progress(v: int) -> None:
            if progress_cb is not None:
                progress_cb(int(v))

        def log(msg: str) -> None:
            if log_cb is not None:
                log_cb(str(msg))

        p = Path(input_path)
        app_root = Path(self.app_root)

        if not p.exists():
            raise FileNotFoundError(f"Input mesh not found: {p}")

        set_progress(8)
        log("Loading mesh ...")

        # 1) load
        m0 = load_mesh(p)
        mesh_in = MeshData(
            pos=np.asarray(m0.vertices, dtype=np.float32),
            faces=np.asarray(m0.faces, dtype=np.int64),
            path=str(p),
            arch=self.preset.arch,
        )
        meta: Dict[str, Any] = {
            "input": str(p),
            "preset": self.preset.key,
            "runner": self.preset.runner,
            "device_requested": self.device,
        }

        set_progress(18)
        log(f"Remeshing to {int(self.preset.target_faces)} faces ...")

        # 2) remesh to target faces
        rr = remesh_to_faces(mesh_in.pos, mesh_in.faces, target_faces=int(self.preset.target_faces))
        meta["remesh"] = rr.meta
        v = np.asarray(rr.vertices, dtype=np.float32)
        f = np.asarray(rr.faces, dtype=np.int64)

        # IMPORTANT:
        # keep remeshed vertices BEFORE orient so result view can render
        # in the original imported pose while preserving remeshed topology.
        v_display = v.copy()
        meta["display_vertices"] = v_display.astype(np.float32, copy=False)

        set_progress(32)

        # 3) orient to reference
        if self.preset.do_orient:
            log("Orienting mesh to reference ...")

            ref_upper, ref_lower = _resolve_reference_paths(app_root, self.preset)

            if not ref_upper.exists():
                raise FileNotFoundError(f"Missing reference (upper): {ref_upper}")
            if not ref_lower.exists():
                raise FileNotFoundError(f"Missing reference (lower): {ref_lower}")

            ocfg = OrientConfig(
                ref_upper=ref_upper,
                ref_lower=ref_lower,
                sample_n=int(self.preset.orient_sample_n),
                seed=int(self.preset.orient_seed),
                allow_reflection=bool(self.preset.allow_reflection),
            )
            v2, o_meta = orient_mesh_to_reference(v, arch=self.preset.arch, cfg=ocfg)
            v = np.asarray(v2, dtype=np.float32)
            meta["orient"] = {
                **dict(o_meta or {}),
                "enabled": True,
                "ref_upper": str(ref_upper),
                "ref_lower": str(ref_lower),
            }
        else:
            meta["orient"] = {"enabled": False}

        mesh_proc = MeshData(
            pos=v.astype(np.float32, copy=False),
            faces=f.astype(np.int64, copy=False),
            path=str(p),
            arch=self.preset.arch,
        )

        set_progress(46)
        log("Resolving checkpoints ...")

        # 4) resolve checkpoints
        dev = self._pick_device()
        meta["device_used"] = dev

        ckpts_raw = [str(x) for x in (self.preset.ckpts or [])]
        ckpts_exist_abs, ckpts_missing_abs = _resolve_many_from_app_root(app_root, ckpts_raw)

        if not ckpts_exist_abs:
            raise FileNotFoundError(
                f"No checkpoint found for preset '{self.preset.key}'. "
                f"Missing resolved paths: {ckpts_missing_abs[:5]}"
            )

        meta["ckpt_missing"] = ckpts_missing_abs
        meta["ckpt_exist"] = ckpts_exist_abs

        set_progress(58)
        log(f"Building runner: {self.preset.runner} ...")

        # 5) build runner
        runner = _build_runner(app_root=app_root, preset=self.preset, ckpts_exist_abs=ckpts_exist_abs, device=dev)

        if self.preset.runner in {"pointnetpp", "pointcnn"}:
            meta[f"{self.preset.runner}_cfg"] = {
                "deterministic_sampling": bool(self.preset.pt_deterministic_sampling),
                "sample_seed": int(self.preset.pt_sample_seed),
                "mc_passes": int(self.preset.pt_mc_passes),
                "cover_faces": bool(self.preset.pt_cover_faces),
                "cover_jitter": float(self.preset.pt_cover_jitter),
                "num_points": int(self.preset.num_points),
                "point_feature": str(self.preset.point_feature),
            }

        set_progress(72)
        log("Running inference ...")

        # 6) inference
        out = runner.infer(mesh_proc, fidx=None)
        labels = np.asarray(out.labels_face, dtype=np.int64)
        num_classes = int(out.num_classes)
        out_meta = dict(out.meta or {})
        meta["infer"] = out_meta

        set_progress(88)
        log("Post-processing labels ...")

        # 7) postprocess
        labels, pp_meta = run_postprocess(
            labels=labels,
            faces=mesh_proc.faces,
            num_classes=num_classes,
            centers=out_meta.get("centers_used", None),
            probs=out_meta.get("probs_used", None),
            preset=self.preset,
        )
        meta["postprocess"] = pp_meta

        set_progress(96)
        log("Finalizing result ...")

        return PipelineResult(
            mesh_in=mesh_in,
            mesh_proc=mesh_proc,
            labels_face=np.asarray(labels, dtype=np.int64),
            num_classes=num_classes,
            meta=meta,
        )