from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Tuple

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFrame, QScrollArea, QPushButton
)


@dataclass
class LayerInfo:
    name: str
    mesh_path: Optional[Path]
    n_verts: int
    n_faces: int
    arch: str  # "upper" | "lower"
    palette: Dict[str, Tuple[int, int, int]]  # "FDI 11" -> (r,g,b)


class LayerCard(QFrame):
    def __init__(self, info: LayerInfo, parent=None):
        super().__init__(parent)
        self.setObjectName("LayerCard")

        lay = QVBoxLayout(self)
        lay.setContentsMargins(12, 10, 12, 10)
        lay.setSpacing(8)

        title = QLabel(info.name)
        title.setObjectName("LayerTitle")
        lay.addWidget(title)

        sub = QLabel(f"File: {info.mesh_path.name if info.mesh_path else '-'}")
        sub.setObjectName("LayerMeta")
        lay.addWidget(sub)

        meta = QLabel(f"Vertices: {info.n_verts}    Faces: {info.n_faces}")
        meta.setObjectName("LayerMeta")
        lay.addWidget(meta)

        arch = QLabel(f"Arch: {info.arch.upper()}")
        arch.setObjectName("LayerMeta")
        lay.addWidget(arch)

        pal_wrap = QFrame()
        pal_lay = QVBoxLayout(pal_wrap)
        pal_lay.setContentsMargins(0, 0, 0, 0)
        pal_lay.setSpacing(4)

        for k, rgb in info.palette.items():
            row = QHBoxLayout()
            row.setContentsMargins(0, 0, 0, 0)
            row.setSpacing(8)

            sw = QFrame()
            sw.setFixedSize(14, 14)
            sw.setStyleSheet(f"background: rgb({rgb[0]},{rgb[1]},{rgb[2]}); border-radius: 3px;")
            row.addWidget(sw)

            lb = QLabel(k)
            lb.setObjectName("LayerMeta")
            row.addWidget(lb)
            row.addStretch(1)

            pal_lay.addLayout(row)

        lay.addWidget(pal_wrap)


class LayerPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("LayerPanel")

        root = QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(8)

        head = QLabel("Layers")
        head.setObjectName("LayerHeader")
        root.addWidget(head)

        self.btn_back = QPushButton("Back")
        self.btn_back.setObjectName("LayerBackBtn")
        root.addWidget(self.btn_back)

        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setFrameShape(QFrame.NoFrame)

        self.inner = QWidget()
        self.v = QVBoxLayout(self.inner)
        self.v.setContentsMargins(0, 0, 0, 0)
        self.v.setSpacing(10)
        self.v.addStretch(1)

        self.scroll.setWidget(self.inner)
        root.addWidget(self.scroll, 1)

        self.setStyleSheet(
            """
            #LayerPanel { background: rgba(255,255,255,0.04); border-left: 1px solid rgba(255,255,255,0.08); }
            #LayerHeader { color: white; font-size: 14px; font-weight: 600; }
            #LayerBackBtn { padding: 6px 10px; border-radius: 10px; background: rgba(255,255,255,0.08); color: white; }
            #LayerBackBtn:hover { background: rgba(255,255,255,0.12); }
            #LayerCard { background: rgba(255,255,255,0.07); border: 1px solid rgba(255,255,255,0.10); border-radius: 12px; }
            #LayerTitle { color: white; font-size: 13px; font-weight: 700; }
            #LayerMeta { color: rgba(255,255,255,0.80); font-size: 11px; }
            """
        )

    def clear_layers(self) -> None:
        while self.v.count() > 1:
            item = self.v.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()

    def set_layers(self, layers: list[LayerInfo]) -> None:
        self.clear_layers()
        for info in layers:
            self.v.insertWidget(self.v.count() - 1, LayerCard(info))