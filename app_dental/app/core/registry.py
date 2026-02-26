from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import yaml


@dataclass(frozen=True)
class Preset:
    key: str
    arch: str
    runner: str
    model_import: str
    model_kwargs: Dict[str, Any]
    ckpts: List[str]

    # preprocess
    target_faces: int = 16000
    do_orient: bool = True
    do_normalize: bool = True
    allow_reflection: bool = False
    ref_path: str = ""

    # point-based
    num_points: int = 4096
    point_feature: str = "xyz_n"

    # postprocess
    do_postprocess: bool = True
    pp_knn_k: int = 16
    pp_smooth_iters: int = 1
    pp_ungingiva_margin: float = 0.05
    pp_ungingiva_min_tooth_p: float = 0.12
    pp_keep_gingiva_lcc: bool = True


class Registry:
    """
    โหลด registry.yaml แล้วเก็บ preset ทั้งหมดไว้ใน dict
    """

    def __init__(self, yaml_path: str | Path):
        self.yaml_path = Path(yaml_path)
        self._presets: Dict[str, Preset] = {}

    def load(self) -> "Registry":
        """อ่านไฟล์ yaml แล้วสร้าง Preset objects"""
        if not self.yaml_path.exists():
            self._presets = {}
            return self

        with self.yaml_path.open("r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}

        presets = raw.get("presets") or {}
        out: Dict[str, Preset] = {}

        for key, cfg in presets.items():
            if not isinstance(cfg, dict):
                continue

            arch = str(cfg.get("arch", "lower")).lower()
            if arch not in ("upper", "lower"):
                arch = "lower"

            out[str(key)] = Preset(
                key=str(key),
                arch=arch,
                runner=str(cfg.get("runner", "meshsegnet")),
                model_import=str(cfg.get("model_import", "")),
                model_kwargs=dict(cfg.get("model_kwargs", {}) or {}),
                ckpts=[str(x) for x in (cfg.get("ckpts", []) or [])],

                target_faces=int(cfg.get("target_faces", 16000)),
                do_orient=bool(cfg.get("do_orient", True)),
                do_normalize=bool(cfg.get("do_normalize", True)),
                allow_reflection=bool(cfg.get("allow_reflection", False)),
                ref_path=str(cfg.get("ref_path", "")),

                num_points=int(cfg.get("num_points", 4096)),
                point_feature=str(cfg.get("point_feature", "xyz_n")),

                do_postprocess=bool(cfg.get("do_postprocess", True)),
                pp_knn_k=int(cfg.get("pp_knn_k", 16)),
                pp_smooth_iters=int(cfg.get("pp_smooth_iters", 1)),
                pp_ungingiva_margin=float(cfg.get("pp_ungingiva_margin", 0.05)),
                pp_ungingiva_min_tooth_p=float(cfg.get("pp_ungingiva_min_tooth_p", 0.12)),
                pp_keep_gingiva_lcc=bool(cfg.get("pp_keep_gingiva_lcc", True)),
            )

        self._presets = out
        return self

    def keys(self) -> list[str]:
        return sorted(self._presets.keys())

    def get(self, key: str) -> Preset:
        if key not in self._presets:
            raise KeyError(f"Preset not found: {key}")
        return self._presets[key]