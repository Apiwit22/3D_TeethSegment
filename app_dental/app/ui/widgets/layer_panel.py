from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

RGB = Tuple[int, int, int]


@dataclass
class SegmentInfo:
    key: str
    display_name: str
    color: RGB
    visible: bool = True


@dataclass
class LayerInfo:
    name: str
    mesh_path: Optional[Path]
    n_verts: int
    n_faces: int
    arch: str
    palette: Dict[str, RGB]
    segments: List[SegmentInfo] = field(default_factory=list)


class SegmentRow(QFrame):
    visibilityChanged = Signal(str, bool)   # segment_key, visible
    selected = Signal(str)                  # segment_key

    def __init__(self, seg: SegmentInfo, parent=None):
        super().__init__(parent)
        self.seg = seg
        self._visible = bool(seg.visible)

        self.setObjectName("SegmentRow")

        row = QHBoxLayout(self)
        row.setContentsMargins(8, 6, 8, 6)
        row.setSpacing(8)

        self.eye_btn = QToolButton()
        self.eye_btn.setObjectName("SegmentEyeBtn")
        self.eye_btn.setText("👁" if self._visible else "⛔")
        self.eye_btn.clicked.connect(self._toggle_visible)
        row.addWidget(self.eye_btn, 0)

        sw = QFrame()
        sw.setFixedSize(14, 14)
        sw.setStyleSheet(
            f"background: rgb({seg.color[0]},{seg.color[1]},{seg.color[2]}); border-radius: 3px;"
        )
        row.addWidget(sw, 0)

        self.name_btn = QPushButton(seg.display_name)
        self.name_btn.setObjectName("SegmentNameBtn")
        self.name_btn.setCursor(Qt.PointingHandCursor)
        self.name_btn.clicked.connect(lambda: self.selected.emit(self.seg.key))
        row.addWidget(self.name_btn, 1)

    def _toggle_visible(self) -> None:
        self._visible = not self._visible
        self.eye_btn.setText("👁" if self._visible else "⛔")
        self.visibilityChanged.emit(self.seg.key, self._visible)

    def set_selected(self, selected: bool) -> None:
        self.setProperty("selected", selected)
        self.style().unpolish(self)
        self.style().polish(self)

    def set_visible_state(self, visible: bool) -> None:
        self._visible = bool(visible)
        self.eye_btn.setText("👁" if self._visible else "⛔")


class LayerCard(QFrame):
    segmentVisibilityChanged = Signal(str, str, bool)   # layer_name, segment_key, visible
    segmentSelected = Signal(str, str)                  # layer_name, segment_key

    def __init__(self, info: LayerInfo, parent=None):
        super().__init__(parent)
        self.info = info
        self._segment_rows: Dict[str, SegmentRow] = {}

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

        if info.segments:
            seg_wrap = QFrame()
            seg_wrap.setObjectName("SegmentWrap")
            seg_lay = QVBoxLayout(seg_wrap)
            seg_lay.setContentsMargins(0, 4, 0, 0)
            seg_lay.setSpacing(6)

            seg_head = QLabel("Segments")
            seg_head.setObjectName("LayerMeta")
            seg_lay.addWidget(seg_head)

            for seg in info.segments:
                row = SegmentRow(seg)
                row.visibilityChanged.connect(
                    lambda segment_key, visible, layer_name=info.name:
                    self.segmentVisibilityChanged.emit(layer_name, segment_key, visible)
                )
                row.selected.connect(
                    lambda segment_key, layer_name=info.name:
                    self.segmentSelected.emit(layer_name, segment_key)
                )
                self._segment_rows[seg.key] = row
                seg_lay.addWidget(row)

            lay.addWidget(seg_wrap)
        else:
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
                sw.setStyleSheet(
                    f"background: rgb({rgb[0]},{rgb[1]},{rgb[2]}); border-radius: 3px;"
                )
                row.addWidget(sw)

                lb = QLabel(k)
                lb.setObjectName("LayerMeta")
                row.addWidget(lb)
                row.addStretch(1)

                pal_lay.addLayout(row)

            lay.addWidget(pal_wrap)

    def clear_selection(self) -> None:
        for row in self._segment_rows.values():
            row.set_selected(False)

    def set_selected_segment(self, segment_key: str) -> None:
        for key, row in self._segment_rows.items():
            row.set_selected(key == segment_key)


class LayerPanel(QWidget):
    segmentVisibilityChanged = Signal(str, str, bool)   # layer_name, segment_key, visible
    segmentSelected = Signal(str, str)                  # layer_name, segment_key

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("LayerPanel")
        self._cards: Dict[str, LayerCard] = {}

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
            #LayerPanel {
              background: rgba(255,255,255,0.04);
              border-left: 1px solid rgba(255,255,255,0.08);
            }
            #LayerHeader {
              color: white;
              font-size: 14px;
              font-weight: 600;
            }
            #LayerBackBtn {
              padding: 6px 10px;
              border-radius: 10px;
              background: rgba(255,255,255,0.08);
              color: white;
            }
            #LayerBackBtn:hover {
              background: rgba(255,255,255,0.12);
            }
            #LayerCard {
              background: rgba(255,255,255,0.07);
              border: 1px solid rgba(255,255,255,0.10);
              border-radius: 12px;
            }
            #LayerTitle {
              color: white;
              font-size: 13px;
              font-weight: 700;
            }
            #LayerMeta {
              color: rgba(255,255,255,0.80);
              font-size: 11px;
            }
            #SegmentRow {
              background: rgba(255,255,255,0.04);
              border: 1px solid rgba(255,255,255,0.06);
              border-radius: 8px;
            }
            #SegmentRow[selected="true"] {
              background: rgba(120,160,255,0.18);
              border: 1px solid rgba(120,160,255,0.40);
            }
            #SegmentEyeBtn {
              color: white;
              background: transparent;
              border: none;
              font-size: 12px;
            }
            #SegmentNameBtn {
              color: white;
              background: transparent;
              border: none;
              text-align: left;
              padding: 2px 4px;
            }
            #SegmentNameBtn:hover {
              color: rgb(170, 200, 255);
            }
            """
        )

    def clear_layers(self) -> None:
        self._cards.clear()
        while self.v.count() > 1:
            item = self.v.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()

    def set_layers(self, layers: List[LayerInfo]) -> None:
        self.clear_layers()
        for info in layers:
            card = LayerCard(info)
            card.segmentVisibilityChanged.connect(self.segmentVisibilityChanged)
            card.segmentSelected.connect(self._on_segment_selected)
            self._cards[info.name] = card
            self.v.insertWidget(self.v.count() - 1, card)

    def _on_segment_selected(self, layer_name: str, segment_key: str) -> None:
        for name, card in self._cards.items():
            if name == layer_name:
                card.set_selected_segment(segment_key)
            else:
                card.clear_selection()
        self.segmentSelected.emit(layer_name, segment_key)