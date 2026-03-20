from __future__ import annotations

from pathlib import Path
from typing import Optional, TYPE_CHECKING

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QPushButton,
    QSizePolicy,
    QSplitter,
    QTabBar,
    QVBoxLayout,
    QWidget,
)

from app.ui.widgets.layer_panel import LayerPanel
from app.ui.widgets.viewer_widget import ViewerWidget

if TYPE_CHECKING:
    from app.ui.main_window import MainWindow


def family_from_preset(preset_key: str) -> str:
    """
    Return preset family name without the final arch suffix only.

    Examples:
      tsmdl_upper         -> tsmdl
      tsmdl_final_upper   -> tsmdl_final
      fast_tgcn_lower     -> fast_tgcn
    """
    k = (preset_key or "").strip()
    if k.endswith("_upper"):
        return k[:-6]
    if k.endswith("_lower"):
        return k[:-6]
    return k


def paired_preset_key(preset_key: str) -> str:
    """
    Swap only the final arch suffix.

    Examples:
      tsmdl_upper         -> tsmdl_lower
      tsmdl_final_upper   -> tsmdl_final_lower
      fast_tgcn_lower     -> fast_tgcn_upper
    """
    k = (preset_key or "").strip()
    if k.endswith("_upper"):
        return k[:-6] + "_lower"
    if k.endswith("_lower"):
        return k[:-6] + "_upper"
    return k


class ChromeTabBar(QTabBar):
    """Chrome/Edge-like tab bar: last '+' tab creates a new tab."""

    requestNewTab = Signal()

    def mousePressEvent(self, e):
        idx = self.tabAt(e.pos())
        if idx == self.count() - 1 and self.tabText(idx) == "+":
            self.requestNewTab.emit()
            e.accept()
            return
        super().mousePressEvent(e)


class StartLikeTab(QWidget):
    """
    Input tab:
    - upper/lower preset selectors
    - run button
    - dual viewers (upper / lower)
    """

    def __init__(self, mainwin: "MainWindow"):
        super().__init__()
        self.mainwin = mainwin

        self.mesh_upper: Optional[Path] = None
        self.mesh_lower: Optional[Path] = None

        # สำคัญ: ให้ root widget มีพื้นหลังทึบทันที
        self.setObjectName("StartLikeTabRoot")
        self.setAttribute(Qt.WA_StyledBackground, True)

        root_lay = QVBoxLayout(self)
        root_lay.setContentsMargins(0, 0, 0, 0)
        root_lay.setSpacing(10)

        top = QHBoxLayout()
        top.setSpacing(10)

        self.cb_upper = QComboBox()
        self.cb_lower = QComboBox()
        self.cb_upper.setObjectName("TopCombo")
        self.cb_lower.setObjectName("TopCombo")

        self.btn_run = QPushButton("▶")
        self.btn_run.setFixedWidth(46)
        self.btn_run.setObjectName("RunCircleBtn")

        top.addWidget(self.cb_upper, 1)
        top.addWidget(self.btn_run, 0)
        top.addWidget(self.cb_lower, 1)
        root_lay.addLayout(top)

        row = QHBoxLayout()
        row.setSpacing(12)

        self.viewer_upper = ViewerWidget()
        self.viewer_lower = ViewerWidget()

        self.viewer_upper.rightClicked.connect(
            lambda pos: mainwin._show_context_menu_from(self.viewer_upper, pos)
        )
        self.viewer_lower.rightClicked.connect(
            lambda pos: mainwin._show_context_menu_from(self.viewer_lower, pos)
        )

        row.addWidget(mainwin._wrap(self.viewer_upper, "UPPER"), 1)
        row.addWidget(mainwin._wrap(self.viewer_lower, "LOWER"), 1)
        root_lay.addLayout(row, 1)

        self.setStyleSheet(
            """
            #StartLikeTabRoot {
              background: #0b1220;
            }

            #TopCombo {
              background: rgba(255,255,255,0.07);
              border: 1px solid rgba(255,255,255,0.10);
              color: rgba(255,255,255,0.92);
              padding: 6px 12px;
              border-radius: 12px;
            }

            #RunCircleBtn {
              background: rgba(255,255,255,0.10);
              border: 1px solid rgba(255,255,255,0.12);
              color: white;
              font-size: 16px;
              border-radius: 14px;
              padding: 6px 0px;
            }

            #RunCircleBtn:hover {
              background: rgba(255,255,255,0.16);
            }
            """
        )

        self._populate_model_combos()
        self.cb_upper.currentIndexChanged.connect(self._sync_lower_from_upper)
        self.cb_lower.currentIndexChanged.connect(self._sync_upper_from_lower)
        self.btn_run.clicked.connect(self._run_from_this_tab)

    def _populate_model_combos(self) -> None:
        keys = list(self.mainwin.registry.keys())

        # hide legacy tsmdl presets from UI
        hidden = {"tsmdl_upper", "tsmdl_lower"}

        upper_opts = sorted(
            [k for k in keys if k.endswith("_upper") and k not in hidden]
        )
        lower_opts = sorted(
            [k for k in keys if k.endswith("_lower") and k not in hidden]
        )

        self.cb_upper.clear()
        self.cb_lower.clear()
        self.cb_upper.addItems(upper_opts)
        self.cb_lower.addItems(lower_opts)

        # default to tsmdl_final if available
        idx_up = self.cb_upper.findText("tsmdl_final_upper")
        if idx_up >= 0:
            self.cb_upper.setCurrentIndex(idx_up)

        idx_lo = self.cb_lower.findText("tsmdl_final_lower")
        if idx_lo >= 0:
            self.cb_lower.setCurrentIndex(idx_lo)

        if self.cb_upper.count() > 0:
            self._sync_lower_from_upper()

    def _sync_lower_from_upper(self) -> None:
        up = self.cb_upper.currentText().strip()
        if not up:
            return
        tgt = paired_preset_key(up)
        idx = self.cb_lower.findText(tgt)
        if idx >= 0:
            self.cb_lower.blockSignals(True)
            self.cb_lower.setCurrentIndex(idx)
            self.cb_lower.blockSignals(False)

    def _sync_upper_from_lower(self) -> None:
        lo = self.cb_lower.currentText().strip()
        if not lo:
            return
        tgt = paired_preset_key(lo)
        idx = self.cb_upper.findText(tgt)
        if idx >= 0:
            self.cb_upper.blockSignals(True)
            self.cb_upper.setCurrentIndex(idx)
            self.cb_upper.blockSignals(False)

    def _run_from_this_tab(self) -> None:
        self.mainwin._run_with_presets(
            preset_upper=self.cb_upper.currentText().strip(),
            preset_lower=self.cb_lower.currentText().strip(),
            owner_tab=self,
        )


class PerTabResultWidget(QWidget):
    """
    Result widget for a tab:
    - single large viewer
    - per-tab layer panel
    """

    def __init__(self, mainwin: "MainWindow", input_tab: StartLikeTab):
        super().__init__()
        self.mainwin = mainwin
        self.input_tab = input_tab

        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(12)

        self.viewer = ViewerWidget()
        self.viewer.rightClicked.connect(
            lambda pos: mainwin._show_context_menu_from(self.viewer, pos)
        )

        self.layer_panel = LayerPanel()
        self.layer_panel.setMinimumWidth(320)
        self.layer_panel.setMaximumWidth(560)
        self.layer_panel.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)

        try:
            self.layer_panel.btn_back.setText("Back")
        except Exception:
            pass

        self.layer_panel.btn_back.clicked.connect(self._back_to_input_tab)

        self.splitter = QSplitter(Qt.Horizontal)
        self.splitter.addWidget(self.viewer)
        self.splitter.addWidget(self.layer_panel)
        self.splitter.setCollapsible(0, False)
        self.splitter.setCollapsible(1, True)
        self.splitter.setStretchFactor(0, 1)
        self.splitter.setStretchFactor(1, 0)
        self.splitter.setSizes([1100, 360])

        lay.addWidget(self.splitter, 1)

    def _back_to_input_tab(self) -> None:
        self.mainwin._restore_input_tab(self)