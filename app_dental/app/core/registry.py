from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

import yaml


_ALLOWED_ARCH = {"upper", "lower"}
_ALLOWED_RUNNERS = {"meshsegnet", "pointnetpp", "tsmdl", "fast_tgcn", "pointcnn"}


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

    # backward-compatible single ref
    ref_path: str = ""

    # preferred explicit refs
    ref_upper_path: str = ""
    ref_lower_path: str = ""

    orient_sample_n: int = 30000
    orient_seed: int = 1234

    # point-based
    num_points: int = 4096
    point_feature: str = "xyz_n"
    pt_deterministic_sampling: bool = True
    pt_sample_seed: int = 1234
    pt_mc_passes: int = 1
    pt_cover_faces: bool = True
    pt_cover_jitter: float = 0.0

    # postprocess
    do_postprocess: bool = True
    pp_mode: str = "best_visual"
    pp_knn_k: int = 16
    pp_smooth_iters: int = 1
    pp_ungingiva_margin: float = 0.05
    pp_ungingiva_min_tooth_p: float = 0.12
    pp_keep_gingiva_lcc: bool = True
    pp_conf_thresh: float = 0.60
    pp_conf_neighbor_iters: int = 1
    pp_min_comp_size: int = 30
    pp_clean_iters: int = 1

    # keep any extra yaml fields for debugging / forward compatibility
    extras: Dict[str, Any] = field(default_factory=dict)


def _as_bool(v: Any, *, field_name: str, preset_key: str) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    if isinstance(v, str):
        s = v.strip().lower()
        if s in {"1", "true", "yes", "y", "on"}:
            return True
        if s in {"0", "false", "no", "n", "off"}:
            return False
    raise ValueError(f"Preset '{preset_key}': field '{field_name}' must be bool-like, got {type(v).__name__}")


def _as_int(v: Any, *, field_name: str, preset_key: str, min_value: int | None = None) -> int:
    try:
        out = int(v)
    except Exception as e:
        raise ValueError(
            f"Preset '{preset_key}': field '{field_name}' must be int-compatible, got {v!r}"
        ) from e
    if min_value is not None and out < min_value:
        raise ValueError(
            f"Preset '{preset_key}': field '{field_name}' must be >= {min_value}, got {out}"
        )
    return out


def _as_float(v: Any, *, field_name: str, preset_key: str, min_value: float | None = None) -> float:
    try:
        out = float(v)
    except Exception as e:
        raise ValueError(
            f"Preset '{preset_key}': field '{field_name}' must be float-compatible, got {v!r}"
        ) from e
    if min_value is not None and out < min_value:
        raise ValueError(
            f"Preset '{preset_key}': field '{field_name}' must be >= {min_value}, got {out}"
        )
    return out


def _as_str(v: Any, *, field_name: str, preset_key: str, allow_empty: bool = True) -> str:
    if v is None:
        s = ""
    else:
        s = str(v).strip()
    if not allow_empty and not s:
        raise ValueError(f"Preset '{preset_key}': field '{field_name}' must not be empty")
    return s


def _as_dict(v: Any, *, field_name: str, preset_key: str) -> Dict[str, Any]:
    if v is None:
        return {}
    if not isinstance(v, dict):
        raise ValueError(f"Preset '{preset_key}': field '{field_name}' must be a dict")
    return dict(v)


def _as_list_of_str(v: Any, *, field_name: str, preset_key: str) -> List[str]:
    if v is None:
        return []
    if isinstance(v, (str, Path)):
        return [str(v)]
    if not isinstance(v, list):
        raise ValueError(f"Preset '{preset_key}': field '{field_name}' must be list[str]")
    return [str(x) for x in v]


class Registry:
    """
    Load registry.yaml and keep validated Preset objects.
    """

    def __init__(self, yaml_path: str | Path):
        self.yaml_path = Path(yaml_path)
        self._presets: Dict[str, Preset] = {}

    def _parse_preset(self, key: str, cfg: Dict[str, Any]) -> Preset:
        if not isinstance(cfg, dict):
            raise ValueError(f"Preset '{key}': config must be a mapping")

        arch = _as_str(cfg.get("arch", "lower"), field_name="arch", preset_key=key).lower()
        if arch not in _ALLOWED_ARCH:
            raise ValueError(f"Preset '{key}': arch must be one of {_ALLOWED_ARCH}, got {arch!r}")

        runner = _as_str(cfg.get("runner", "meshsegnet"), field_name="runner", preset_key=key).lower()
        if runner not in _ALLOWED_RUNNERS:
            raise ValueError(f"Preset '{key}': runner must be one of {_ALLOWED_RUNNERS}, got {runner!r}")

        model_import = _as_str(
            cfg.get("model_import", ""),
            field_name="model_import",
            preset_key=key,
            allow_empty=False,
        )

        model_kwargs = _as_dict(cfg.get("model_kwargs", {}), field_name="model_kwargs", preset_key=key)
        ckpts = _as_list_of_str(cfg.get("ckpts", []), field_name="ckpts", preset_key=key)

        ref_path = _as_str(cfg.get("ref_path", ""), field_name="ref_path", preset_key=key)
        ref_upper_path = _as_str(cfg.get("ref_upper_path", ""), field_name="ref_upper_path", preset_key=key)
        ref_lower_path = _as_str(cfg.get("ref_lower_path", ""), field_name="ref_lower_path", preset_key=key)

        preset = Preset(
            key=key,
            arch=arch,
            runner=runner,
            model_import=model_import,
            model_kwargs=model_kwargs,
            ckpts=ckpts,

            target_faces=_as_int(cfg.get("target_faces", 16000), field_name="target_faces", preset_key=key, min_value=1),
            do_orient=_as_bool(cfg.get("do_orient", True), field_name="do_orient", preset_key=key),
            do_normalize=_as_bool(cfg.get("do_normalize", True), field_name="do_normalize", preset_key=key),
            allow_reflection=_as_bool(cfg.get("allow_reflection", False), field_name="allow_reflection", preset_key=key),
            ref_path=ref_path,
            ref_upper_path=ref_upper_path,
            ref_lower_path=ref_lower_path,
            orient_sample_n=_as_int(cfg.get("orient_sample_n", 30000), field_name="orient_sample_n", preset_key=key, min_value=100),
            orient_seed=_as_int(cfg.get("orient_seed", 1234), field_name="orient_seed", preset_key=key),

            num_points=_as_int(cfg.get("num_points", 4096), field_name="num_points", preset_key=key, min_value=1),
            point_feature=_as_str(cfg.get("point_feature", "xyz_n"), field_name="point_feature", preset_key=key, allow_empty=False),
            pt_deterministic_sampling=_as_bool(
                cfg.get("pt_deterministic_sampling", True),
                field_name="pt_deterministic_sampling",
                preset_key=key,
            ),
            pt_sample_seed=_as_int(cfg.get("pt_sample_seed", 1234), field_name="pt_sample_seed", preset_key=key),
            pt_mc_passes=_as_int(cfg.get("pt_mc_passes", 1), field_name="pt_mc_passes", preset_key=key, min_value=1),
            pt_cover_faces=_as_bool(cfg.get("pt_cover_faces", True), field_name="pt_cover_faces", preset_key=key),
            pt_cover_jitter=_as_float(cfg.get("pt_cover_jitter", 0.0), field_name="pt_cover_jitter", preset_key=key, min_value=0.0),

            do_postprocess=_as_bool(cfg.get("do_postprocess", True), field_name="do_postprocess", preset_key=key),
            pp_mode=_as_str(cfg.get("pp_mode", "best_visual"), field_name="pp_mode", preset_key=key, allow_empty=False).lower(),
            pp_knn_k=_as_int(cfg.get("pp_knn_k", 16), field_name="pp_knn_k", preset_key=key, min_value=1),
            pp_smooth_iters=_as_int(cfg.get("pp_smooth_iters", 1), field_name="pp_smooth_iters", preset_key=key, min_value=0),
            pp_ungingiva_margin=_as_float(cfg.get("pp_ungingiva_margin", 0.05), field_name="pp_ungingiva_margin", preset_key=key),
            pp_ungingiva_min_tooth_p=_as_float(
                cfg.get("pp_ungingiva_min_tooth_p", 0.12),
                field_name="pp_ungingiva_min_tooth_p",
                preset_key=key,
            ),
            pp_keep_gingiva_lcc=_as_bool(
                cfg.get("pp_keep_gingiva_lcc", True),
                field_name="pp_keep_gingiva_lcc",
                preset_key=key,
            ),
            pp_conf_thresh=_as_float(cfg.get("pp_conf_thresh", 0.60), field_name="pp_conf_thresh", preset_key=key),
            pp_conf_neighbor_iters=_as_int(
                cfg.get("pp_conf_neighbor_iters", 1),
                field_name="pp_conf_neighbor_iters",
                preset_key=key,
                min_value=0,
            ),
            pp_min_comp_size=_as_int(cfg.get("pp_min_comp_size", 30), field_name="pp_min_comp_size", preset_key=key, min_value=1),
            pp_clean_iters=_as_int(cfg.get("pp_clean_iters", 1), field_name="pp_clean_iters", preset_key=key, min_value=0),

            extras={k: v for k, v in cfg.items() if k not in {
                "arch", "runner", "model_import", "model_kwargs", "ckpts",
                "target_faces", "do_orient", "do_normalize", "allow_reflection",
                "ref_path", "ref_upper_path", "ref_lower_path",
                "orient_sample_n", "orient_seed",
                "num_points", "point_feature",
                "pt_deterministic_sampling", "pt_sample_seed", "pt_mc_passes",
                "pt_cover_faces", "pt_cover_jitter",
                "do_postprocess", "pp_mode", "pp_knn_k", "pp_smooth_iters",
                "pp_ungingiva_margin", "pp_ungingiva_min_tooth_p",
                "pp_keep_gingiva_lcc", "pp_conf_thresh", "pp_conf_neighbor_iters",
                "pp_min_comp_size", "pp_clean_iters",
            }},
        )

        return preset

    def load(self) -> "Registry":
        if not self.yaml_path.exists():
            raise FileNotFoundError(f"Registry yaml not found: {self.yaml_path}")

        data = yaml.safe_load(self.yaml_path.read_text(encoding="utf-8")) or {}
        if not isinstance(data, dict):
            raise ValueError("registry.yaml root must be a mapping")

        presets_raw = data.get("presets", data)
        if not isinstance(presets_raw, dict):
            raise ValueError("registry.yaml presets section must be a mapping")

        parsed: Dict[str, Preset] = {}
        for key, cfg in presets_raw.items():
            k = str(key).strip()
            if not k:
                raise ValueError("registry.yaml contains empty preset key")
            if k in parsed:
                raise ValueError(f"Duplicate preset key: {k}")
            parsed[k] = self._parse_preset(k, cfg)

        self._presets = parsed
        return self

    def keys(self) -> List[str]:
        return list(self._presets.keys())

    def all(self) -> Dict[str, Preset]:
        return dict(self._presets)

    def get(self, key: str) -> Preset:
        if key not in self._presets:
            raise KeyError(f"Unknown preset: {key}")
        return self._presets[key]