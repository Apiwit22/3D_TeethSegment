from __future__ import annotations

import traceback
from pathlib import Path

from PySide6.QtCore import QObject, QThread, Signal

from app.core.registry import Registry
from app.core.pipeline import Pipeline


class WorkerSignals(QObject):
    log = Signal(str)
    progress = Signal(int)
    done = Signal(object)
    failed = Signal(str)


class InferenceWorker(QThread):
    def __init__(self, *, app_root: Path, mesh_path: Path, preset_key: str, device: str = "cuda"):
        super().__init__()
        self.app_root = Path(app_root)
        self.mesh_path = Path(mesh_path)
        self.preset_key = str(preset_key)
        self.device = str(device)
        self.signals = WorkerSignals()

    def _fail(self, message: str) -> None:
        self.signals.failed.emit(message)

    def run(self) -> None:
        try:
            if not self.mesh_path.exists():
                raise FileNotFoundError(f"Input mesh not found: {self.mesh_path}")

            self.signals.log.emit("Loading registry.yaml ...")
            self.signals.progress.emit(3)

            reg = Registry(self.app_root / "registry.yaml").load()
            preset = reg.get(self.preset_key)

            blocked = {"pointnetpp_upper", "pointnetpp_lower"}
            if preset.key in blocked:
                raise RuntimeError(f"Preset '{preset.key}' is disabled")

            self.signals.log.emit(
                f"Start preset={preset.key} | runner={preset.runner} | arch={preset.arch} | device={self.device}"
        )
            self.signals.progress.emit(5)

            pipe = Pipeline(app_root=self.app_root, preset=preset, device=self.device)

            res = pipe.run(
                self.mesh_path,
                progress_cb=self.signals.progress.emit,
                log_cb=self.signals.log.emit,
            )

            self.signals.progress.emit(100)
            self.signals.log.emit("Inference finished")
            self.signals.done.emit(res)

        except Exception as e:
            tb = traceback.format_exc()
            msg = (
                f"InferenceWorker failed\n"
                f"mesh_path={self.mesh_path}\n"
                f"preset_key={self.preset_key}\n"
                f"device={self.device}\n"
                f"error={type(e).__name__}: {e}\n\n"
                f"{tb}"
            )
            self._fail(msg)