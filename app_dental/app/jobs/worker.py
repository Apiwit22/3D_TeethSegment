# dental_seg_app/app/jobs/worker.py
from __future__ import annotations

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

    def run(self) -> None:
        try:
            self.signals.log.emit("โหลด registry.yaml ...")
            reg = Registry(self.app_root / "registry.yaml").load()
            preset = reg.get(self.preset_key)

            self.signals.log.emit(
                f"เริ่มรัน preset: {preset.key} | runner={preset.runner} | arch={preset.arch}"
            )
            self.signals.progress.emit(5)

            # ✅ ส่ง app_root เข้า Pipeline
            pipe = Pipeline(app_root=self.app_root, preset=preset, device=self.device)
            res = pipe.run(self.mesh_path)

            self.signals.progress.emit(100)
            self.signals.done.emit(res)

        except Exception as e:
            self.signals.failed.emit(str(e))