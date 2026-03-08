from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Dict, Tuple

import numpy as np

from PySide6.QtCore import Qt, Signal, QSize, QTimer
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (
    QMainWindow,
    QWidget,
    QStackedWidget,
    QHBoxLayout,
    QVBoxLayout,
    QLabel,
    QComboBox,
    QPushButton,
    QFileDialog,
    QMessageBox,
    QFrame,
    QTabWidget,
    QTabBar,
    QMenu,
    QSplitter,
    QSizePolicy,
    QToolButton,
    QInputDialog,
)

from app.core.registry import Registry
from app.jobs.worker import InferenceWorker
from app.ui.widgets.viewer_widget import ViewerWidget
from app.ui.widgets.layer_panel import LayerPanel, LayerInfo
from app.data.color_map import label_array_to_rgb_face

PROJECT_EXT = "dsegproj.json"


def _family(preset_key: str) -> str:
    return preset_key.split("_")[0].strip()


@dataclass
class _Job:
    tag: str  # "upper" | "lower"
    mesh_path: Path
    preset_key: str


class ChromeTabBar(QTabBar):
    """Chrome/Edge-like: last tab is '+'; click it to create new tab.
    Intercept mousePressEvent to prevent selecting the '+' tab.
    """
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
      - dropdown upper/lower + run
      - viewer 2 ช่อง upper/lower
    per-tab mesh paths stored here.
    """

    def __init__(self, mainwin: "MainWindow"):
        super().__init__()
        self.mainwin = mainwin

        self.mesh_upper: Optional[Path] = None
        self.mesh_lower: Optional[Path] = None

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

        self.viewer_upper.rightClicked.connect(lambda pos: mainwin._show_context_menu_from(self.viewer_upper, pos))
        self.viewer_lower.rightClicked.connect(lambda pos: mainwin._show_context_menu_from(self.viewer_lower, pos))

        row.addWidget(mainwin._wrap(self.viewer_upper, "UPPER"), 1)
        row.addWidget(mainwin._wrap(self.viewer_lower, "LOWER"), 1)
        root_lay.addLayout(row, 1)

        self.setStyleSheet(
            """
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
            #RunCircleBtn:hover { background: rgba(255,255,255,0.16); }
            """
        )

        self._populate_model_combos()
        self.cb_upper.currentIndexChanged.connect(self._sync_lower_from_upper)
        self.cb_lower.currentIndexChanged.connect(self._sync_upper_from_lower)
        self.btn_run.clicked.connect(self._run_from_this_tab)

    def _populate_model_combos(self) -> None:
        keys = list(self.mainwin.registry.keys())
        upper_opts = sorted([k for k in keys if k.endswith("_upper")])
        lower_opts = sorted([k for k in keys if k.endswith("_lower")])

        self.cb_upper.clear()
        self.cb_lower.clear()
        self.cb_upper.addItems(upper_opts)
        self.cb_lower.addItems(lower_opts)

        if self.cb_upper.count() > 0:
            self._sync_lower_from_upper()

    def _sync_lower_from_upper(self) -> None:
        up = self.cb_upper.currentText().strip()
        if not up:
            return
        fam = _family(up)
        tgt = f"{fam}_lower"
        idx = self.cb_lower.findText(tgt)
        if idx >= 0:
            self.cb_lower.blockSignals(True)
            self.cb_lower.setCurrentIndex(idx)
            self.cb_lower.blockSignals(False)

    def _sync_upper_from_lower(self) -> None:
        lo = self.cb_lower.currentText().strip()
        if not lo:
            return
        fam = _family(lo)
        tgt = f"{fam}_upper"
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
    """Result layout for a single tab: big viewer + LayerPanel (like Result tab)."""

    def __init__(self, mainwin: "MainWindow", input_tab: StartLikeTab):
        super().__init__()
        self.mainwin = mainwin
        self.input_tab = input_tab

        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(12)

        self.viewer = ViewerWidget()
        self.viewer.rightClicked.connect(lambda pos: mainwin._show_context_menu_from(self.viewer, pos))

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


class MainWindow(QMainWindow):
    _RE_UPPER = re.compile(r"(^|[_\-\s])(u|upper)([_\-\s]|$)", re.IGNORECASE)
    _RE_LOWER = re.compile(r"(^|[_\-\s])(l|lower)([_\-\s]|$)", re.IGNORECASE)

    def __init__(self, app_root: Path):
        super().__init__()
        self.setWindowTitle("DentalSeg")
        self.resize(1400, 820)

        self.app_root = Path(app_root)
        self.registry = Registry(self.app_root / "registry.yaml").load()

        # global input (dual page)
        self.mesh_upper: Optional[Path] = None
        self.mesh_lower: Optional[Path] = None
        self.project_path: Optional[Path] = None

        # worker
        self.worker: Optional[InferenceWorker] = None
        self.job_queue: List[_Job] = []
        self.current_job: Optional[_Job] = None

        # last results
        self.res_upper = None
        self.res_lower = None

        self._is_closing = False
        self._last_viewer: Optional[ViewerWidget] = None

        # which StartLikeTab triggered this run
        self._run_owner_tab: Optional[StartLikeTab] = None

        self._tab_seq = 1

        # pages
        self.stack = QStackedWidget()
        self.setCentralWidget(self.stack)

        self.page_dual = self._build_page_dual()
        self.page_result = self._build_page_result()

        self.stack.addWidget(self.page_dual)
        self.stack.addWidget(self.page_result)
        self.stack.setCurrentWidget(self.page_dual)

        self._build_actions()
        self._build_menubar()

        self.setStyleSheet(
            """
            QMainWindow { background: #0b0f18; }
            QMenuBar { background: rgba(255,255,255,0.02); color: rgba(255,255,255,0.92); }
            QMenuBar::item { padding: 6px 10px; }
            QMenuBar::item:selected { background: rgba(255,255,255,0.06); border-radius: 8px; }
            QMenu { background: #121826; color: rgba(255,255,255,0.92); border: 1px solid rgba(255,255,255,0.12); }
            QMenu::item:selected { background: rgba(255,255,255,0.10); }
            """
        )
        self.statusBar().showMessage("Ready")

    # =========================================================
    # Close
    # =========================================================
    def closeEvent(self, event) -> None:
        self._is_closing = True
        self._stop_worker(force=True)

        for vw in (getattr(self, "viewer_upper", None), getattr(self, "viewer_lower", None), getattr(self, "viewer_result", None)):
            if vw is not None:
                try:
                    vw.shutdown()
                except Exception:
                    pass

        if hasattr(self, "tabs"):
            for i in range(self.tabs.count()):
                w = self.tabs.widget(i)
                try:
                    if isinstance(w, StartLikeTab):
                        w.viewer_upper.shutdown()
                        w.viewer_lower.shutdown()
                    elif isinstance(w, PerTabResultWidget):
                        w.viewer.shutdown()
                    elif hasattr(w, "shutdown"):
                        w.shutdown()
                except Exception:
                    pass

        event.accept()

    # =========================================================
    # Active viewer (for reset/export/camera)
    # =========================================================
    def _active_viewer(self) -> ViewerWidget:
        if self.stack.currentWidget() == self.page_result and hasattr(self, "tabs"):
            cur = self.tabs.currentWidget()
            if isinstance(cur, ViewerWidget):
                return cur
            if isinstance(cur, PerTabResultWidget):
                return cur.viewer
            if isinstance(cur, StartLikeTab):
                if self._last_viewer is not None:
                    return self._last_viewer
                return cur.viewer_upper
            return self.viewer_result

        if self._last_viewer is not None:
            return self._last_viewer
        return self.viewer_upper

    # =========================================================
    # Worker
    # =========================================================
    def _set_running_ui(self, running: bool, msg: str = "") -> None:
        if hasattr(self, "btn_run"):
            self.btn_run.setEnabled(not running)
            self.btn_run.setText("⏳" if running else "▶")
        if hasattr(self, "act_run"):
            self.act_run.setEnabled(not running)
        if msg:
            self.statusBar().showMessage(msg, 5000)

    def _stop_worker(self, *, force: bool) -> None:
        w = self.worker
        if w is None:
            return
        try:
            if w.isRunning():
                w.requestInterruption()
                w.quit()
                w.wait(800)
        except Exception:
            pass
        if force:
            try:
                if w.isRunning():
                    w.terminate()
                    w.wait(800)
            except Exception:
                pass

    def _start_next_job(self) -> None:
        if not self.job_queue:
            self.current_job = None
            self.worker = None
            self._set_running_ui(False, "Done")
            self._show_result()
            return

        self.current_job = self.job_queue.pop(0)
        job = self.current_job
        self._set_running_ui(True, f"Running {job.tag.upper()}...")

        self.worker = InferenceWorker(
            app_root=self.app_root,
            mesh_path=job.mesh_path,
            preset_key=job.preset_key,
            device="cuda",
        )
        self.worker.signals.log.connect(lambda s: self._status(job.tag.upper(), s))
        self.worker.signals.progress.connect(lambda p: self._status(job.tag.upper(), f"progress {p}%"))
        self.worker.signals.done.connect(self._on_job_done)
        self.worker.signals.failed.connect(self._on_job_failed)
        self.worker.start()

    def _status(self, tag: str, msg: str) -> None:
        self.statusBar().showMessage(f"[{tag}] {msg}", 5000)

    def _on_job_failed(self, msg: str) -> None:
        self.job_queue.clear()
        self.current_job = None
        self._set_running_ui(False, "Failed")
        self._stop_worker(force=True)
        if not self._is_closing:
            QMessageBox.critical(self, "Run failed", msg)

    def _on_job_done(self, res) -> None:
        if self.current_job is not None:
            if self.current_job.tag == "upper":
                self.res_upper = res
            elif self.current_job.tag == "lower":
                self.res_lower = res

        self.worker = None
        self.current_job = None
        self._start_next_job()

    # =========================================================
    # Actions / Menubar
    # =========================================================
    def _build_actions(self) -> None:
        self.act_import = QAction("Import Mesh...", self)
        self.act_import.setShortcut(QKeySequence.Open)
        self.act_import.triggered.connect(self.on_import_mesh)

        self.act_export_shot = QAction("Export Screenshot...", self)
        self.act_export_shot.triggered.connect(self.on_export_screenshot)

        self.act_clear = QAction("Clear", self)
        self.act_clear.triggered.connect(self.on_clear)

        self.act_exit = QAction("Exit", self)
        self.act_exit.setShortcut(QKeySequence.Quit)
        self.act_exit.triggered.connect(self.close)

        self.act_run = QAction("Run Segmentation", self)
        self.act_run.setShortcut(QKeySequence("Ctrl+R"))
        self.act_run.triggered.connect(
            lambda: self._run_with_presets(
                preset_upper=self.cb_upper.currentText().strip(),
                preset_lower=self.cb_lower.currentText().strip(),
                owner_tab=None,
            )
        )

        self.act_save = QAction("Save Project", self)
        self.act_save.setShortcut(QKeySequence.Save)
        self.act_save.triggered.connect(self.on_save_project)

        self.act_save_as = QAction("Save Project As...", self)
        self.act_save_as.setShortcut(QKeySequence.SaveAs)
        self.act_save_as.triggered.connect(self.on_save_as_project)

        self.act_view_reset = QAction("Reset View", self)
        self.act_view_reset.setShortcut(QKeySequence("R"))
        self.act_view_reset.triggered.connect(lambda: self._active_viewer().reset_view())

        self.act_view_side = QAction("Standard Side View", self)
        self.act_view_side.setShortcut(QKeySequence("Ctrl+1"))
        self.act_view_side.triggered.connect(lambda: self._set_result_camera_side())

        self.act_view_iso = QAction("Isometric", self)
        self.act_view_iso.triggered.connect(lambda: self._active_viewer().set_view("iso"))

        self.act_filters_dummy = QAction("Coming soon...", self)
        self.act_filters_dummy.setEnabled(False)

    def _build_menubar(self) -> None:
        mb = self.menuBar()

        m_file = mb.addMenu("File")
        m_file.addAction(self.act_import)
        m_file.addSeparator()
        m_file.addAction(self.act_export_shot)
        m_file.addSeparator()
        m_file.addAction(self.act_clear)
        m_file.addSeparator()
        m_file.addAction(self.act_exit)

        m_edit = mb.addMenu("Edit")
        m_edit.addAction(self.act_save)
        m_edit.addAction(self.act_save_as)

        m_filters = mb.addMenu("Filters")
        m_filters.addAction(self.act_filters_dummy)

        m_view = mb.addMenu("View")
        m_view.addAction(self.act_view_reset)
        m_view.addSeparator()
        m_view.addAction(self.act_view_side)
        m_view.addAction(self.act_view_iso)

    # =========================================================
    # Page 1: Dual UI (global)
    # =========================================================
    def _build_page_dual(self) -> QWidget:
        root = QWidget()
        root_lay = QVBoxLayout(root)
        root_lay.setContentsMargins(12, 10, 12, 12)
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
        self.btn_run.clicked.connect(
            lambda: self._run_with_presets(
                preset_upper=self.cb_upper.currentText().strip(),
                preset_lower=self.cb_lower.currentText().strip(),
                owner_tab=None,
            )
        )

        top.addWidget(self.cb_upper, 1)
        top.addWidget(self.btn_run, 0)
        top.addWidget(self.cb_lower, 1)
        root_lay.addLayout(top)

        row = QHBoxLayout()
        row.setSpacing(12)

        self.viewer_upper = ViewerWidget()
        self.viewer_lower = ViewerWidget()

        self.viewer_upper.rightClicked.connect(lambda pos: self._show_context_menu_from(self.viewer_upper, pos))
        self.viewer_lower.rightClicked.connect(lambda pos: self._show_context_menu_from(self.viewer_lower, pos))

        row.addWidget(self._wrap(self.viewer_upper, "UPPER"), 1)
        row.addWidget(self._wrap(self.viewer_lower, "LOWER"), 1)
        root_lay.addLayout(row, 1)

        root.setStyleSheet(
            """
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
            #RunCircleBtn:hover { background: rgba(255,255,255,0.16); }
            """
        )

        self._populate_model_combos()
        self.cb_upper.currentIndexChanged.connect(self._sync_lower_from_upper)
        self.cb_lower.currentIndexChanged.connect(self._sync_upper_from_lower)
        return root

    def _wrap(self, viewer: ViewerWidget, title: str) -> QWidget:
        frame = QFrame()
        lay = QVBoxLayout(frame)
        lay.setContentsMargins(10, 10, 10, 10)
        lay.setSpacing(8)

        lb = QLabel(title)
        lb.setAlignment(Qt.AlignHCenter)
        lb.setStyleSheet("color: rgba(255,255,255,0.92); font-size: 16px; letter-spacing: 1px;")
        lay.addWidget(lb)
        lay.addWidget(viewer, 1)

        frame.setStyleSheet(
            """
            QFrame {
              background: rgba(255,255,255,0.06);
              border: 1px solid rgba(255,255,255,0.10);
              border-radius: 14px;
            }
            """
        )
        return frame

    def _populate_model_combos(self) -> None:
        keys = list(self.registry.keys())
        upper_opts = sorted([k for k in keys if k.endswith("_upper")])
        lower_opts = sorted([k for k in keys if k.endswith("_lower")])

        self.cb_upper.clear()
        self.cb_lower.clear()
        self.cb_upper.addItems(upper_opts)
        self.cb_lower.addItems(lower_opts)

        if self.cb_upper.count() > 0:
            self._sync_lower_from_upper()

    def _sync_lower_from_upper(self) -> None:
        up = self.cb_upper.currentText().strip()
        if not up:
            return
        fam = _family(up)
        tgt = f"{fam}_lower"
        idx = self.cb_lower.findText(tgt)
        if idx >= 0:
            self.cb_lower.blockSignals(True)
            self.cb_lower.setCurrentIndex(idx)
            self.cb_lower.blockSignals(False)

    def _sync_upper_from_lower(self) -> None:
        lo = self.cb_lower.currentText().strip()
        if not lo:
            return
        fam = _family(lo)
        tgt = f"{fam}_upper"
        idx = self.cb_upper.findText(tgt)
        if idx >= 0:
            self.cb_upper.blockSignals(True)
            self.cb_upper.setCurrentIndex(idx)
            self.cb_upper.blockSignals(False)

    # =========================================================
    # Page 2: Tabs + global LayerPanel (Result tab only)
    # =========================================================
    def _build_page_result(self) -> QWidget:
        root = QWidget()
        lay = QHBoxLayout(root)
        lay.setContentsMargins(12, 12, 12, 12)
        lay.setSpacing(12)

        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)
        self.tabs.setMovable(True)

        bar = ChromeTabBar()
        bar.requestNewTab.connect(self._add_new_tab_like_start)
        self.tabs.setTabBar(bar)

        self.viewer_result = ViewerWidget()
        self.viewer_result.rightClicked.connect(lambda pos: self._show_context_menu_from(self.viewer_result, pos))
        self.tabs.addTab(self.viewer_result, "Result")

        self.tabs.addTab(QWidget(), "+")
        self.tabs.setCurrentIndex(0)

        self.tabs.setStyleSheet(
            """
            QTabWidget::pane {
              border: 1px solid rgba(255,255,255,0.10);
              top: -1px;
              background: rgba(255,255,255,0.02);
              border-radius: 12px;
            }
            QTabBar::tab {
              background: rgba(255,255,255,0.03);
              color: rgba(255,255,255,0.86);
              border: 1px solid rgba(255,255,255,0.10);
              border-bottom: none;
              padding: 3px 9px;
              margin-right: 5px;
              border-top-left-radius: 10px;
              border-top-right-radius: 10px;
              min-height: 20px;
              font-size: 10px;
            }
            QTabBar::tab:selected {
              background: rgba(255,255,255,0.07);
              color: rgba(255,255,255,0.96);
              border-color: rgba(255,255,255,0.14);
            }
            QTabBar::tab:hover { background: rgba(255,255,255,0.05); }
            """
        )

        self.layer_panel = LayerPanel()
        self.layer_panel.btn_back.clicked.connect(lambda: self.stack.setCurrentWidget(self.page_dual))
        self.layer_panel.setMinimumWidth(320)
        self.layer_panel.setMaximumWidth(560)
        self.layer_panel.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)

        self.result_splitter = QSplitter(Qt.Horizontal)
        self.result_splitter.addWidget(self.tabs)
        self.result_splitter.addWidget(self.layer_panel)
        self.result_splitter.setCollapsible(0, False)
        self.result_splitter.setCollapsible(1, True)
        self.result_splitter.setStretchFactor(0, 1)
        self.result_splitter.setStretchFactor(1, 0)
        self.result_splitter.setSizes([1100, 360])
        lay.addWidget(self.result_splitter, 1)

        self.tabs.currentChanged.connect(self._on_result_tab_changed)

        try:
            self.tabs.tabBar().tabMoved.connect(lambda _a, _b: self._rebind_close_buttons())
            self.tabs.tabBar().tabBarDoubleClicked.connect(self._on_tab_double_clicked)
        except Exception:
            pass

        self._rebind_close_buttons()
        self._set_layers_visible(True)
        return root

    def _set_layers_visible(self, visible: bool) -> None:
        try:
            self.layer_panel.setVisible(bool(visible))
            self.result_splitter.setSizes([1100, 360] if visible else [1400, 0])
        except Exception:
            pass

    def _on_result_tab_changed(self, idx: int) -> None:
        title = self.tabs.tabText(idx) if 0 <= idx < self.tabs.count() else ""
        if title == "+":
            self.tabs.setCurrentIndex(0)
            return
        self._set_layers_visible(title == "Result")

    # =========================================================
    # Tab UI: close x + rename
    # =========================================================
    def _make_close_btn(self, tab_index: int) -> QToolButton:
        btn = QToolButton()
        btn.setText("x")
        btn.setCursor(Qt.PointingHandCursor)
        btn.setAutoRaise(True)
        btn.setFixedSize(QSize(12, 12))
        btn.setProperty("tab_index", int(tab_index))
        btn.setStyleSheet(
            """
            QToolButton {
              color: rgba(255,255,255,0.78);
              background: transparent;
              border: none;
              padding: 0px;
              font-size: 11px;
              font-weight: 600;
            }
            QToolButton:hover { color: rgba(255,255,255,0.98); }
            QToolButton:pressed { color: rgba(255,255,255,0.60); }
            """
        )
        btn.clicked.connect(self._on_close_btn_clicked)
        return btn

    def _rebind_close_buttons(self) -> None:
        bar = self.tabs.tabBar()
        for i in range(self.tabs.count()):
            if self.tabs.tabText(i) == "+":
                bar.setTabButton(i, QTabBar.RightSide, None)
                bar.setTabButton(i, QTabBar.LeftSide, None)
                continue
            bar.setTabButton(i, QTabBar.RightSide, self._make_close_btn(i))

    def _on_close_btn_clicked(self) -> None:
        btn = self.sender()
        if btn is None:
            return
        try:
            idx = int(btn.property("tab_index"))
        except Exception:
            idx = -1

        if not (0 <= idx < self.tabs.count()) or self.tabs.tabBar().tabButton(idx, QTabBar.RightSide) is not btn:
            for i in range(self.tabs.count()):
                if self.tabs.tabBar().tabButton(i, QTabBar.RightSide) is btn:
                    idx = i
                    break

        if 0 <= idx < self.tabs.count():
            self._on_close_tab(idx)

    def _on_tab_double_clicked(self, index: int) -> None:
        if index < 0:
            return
        if self.tabs.tabText(index) == "+":
            return
        old = self.tabs.tabText(index)
        new, ok = QInputDialog.getText(self, "Rename tab", "Tab name:", text=old)
        if ok:
            new = (new or "").strip()
            if new:
                self.tabs.setTabText(index, new)
                self._rebind_close_buttons()

    # =========================================================
    # Replace/restore per-tab result <-> input
    # =========================================================
    def _replace_tab_widget(self, index: int, new_widget: QWidget, title: Optional[str] = None) -> None:
        old_title = self.tabs.tabText(index)
        if title is None:
            title = old_title

        old = self.tabs.widget(index)
        self.tabs.removeTab(index)
        self.tabs.insertTab(index, new_widget, title)
        self.tabs.setCurrentIndex(index)
        self._rebind_close_buttons()

        try:
            if old is not None:
                old.deleteLater()
        except Exception:
            pass

    def _restore_input_tab(self, per_tab_result: PerTabResultWidget) -> None:
        idx = self.tabs.indexOf(per_tab_result)
        if idx < 0:
            return

        input_tab = per_tab_result.input_tab
        tab = StartLikeTab(self)
        tab.mesh_upper = input_tab.mesh_upper
        tab.mesh_lower = input_tab.mesh_lower

        try:
            tab.cb_upper.setCurrentText(input_tab.cb_upper.currentText())
            tab.cb_lower.setCurrentText(input_tab.cb_lower.currentText())
        except Exception:
            pass

        try:
            if tab.mesh_upper is not None:
                tab.viewer_upper.load_base_mesh(tab.mesh_upper, key="upper")
            if tab.mesh_lower is not None:
                tab.viewer_lower.load_base_mesh(tab.mesh_lower, key="lower")
        except Exception:
            pass

        self._replace_tab_widget(idx, tab)

    # =========================================================
    # Tab add/remove
    # =========================================================
    def _find_plus_tab_index(self) -> int:
        for i in range(self.tabs.count()):
            if self.tabs.tabText(i) == "+":
                return i
        return self.tabs.count()

    def _add_new_tab_like_start(self) -> None:
        plus_idx = self._find_plus_tab_index()
        tab = StartLikeTab(self)
        title = f"Tab {self._tab_seq}"
        self._tab_seq += 1
        idx = self.tabs.insertTab(plus_idx, tab, title)
        self.tabs.setCurrentIndex(idx)
        self._rebind_close_buttons()
        self._set_layers_visible(False)

    def _on_close_tab(self, index: int) -> None:
        if self.tabs.tabText(index) == "+":
            return

        alive = [i for i in range(self.tabs.count()) if self.tabs.tabText(i) != "+"]
        if len(alive) <= 1:
            QMessageBox.information(self, "Cannot close", "ต้องมีอย่างน้อย 1 tab")
            return

        w = self.tabs.widget(index)
        self.tabs.removeTab(index)

        try:
            if isinstance(w, StartLikeTab):
                w.viewer_upper.shutdown()
                w.viewer_lower.shutdown()
            elif isinstance(w, PerTabResultWidget):
                w.viewer.shutdown()
            elif hasattr(w, "shutdown"):
                w.shutdown()
        except Exception:
            pass

        try:
            if w is not None and w is not self.viewer_result:
                w.deleteLater()
        except Exception:
            pass

        self._rebind_close_buttons()
        cur_title = self.tabs.tabText(self.tabs.currentIndex()) if self.tabs.count() else ""
        self._set_layers_visible(cur_title == "Result")

    # =========================================================
    # Context menu
    # =========================================================
    def _show_context_menu_from(self, viewer: ViewerWidget, pos_global) -> None:
        self._last_viewer = viewer
        menu = QMenu(self)
        menu.addAction(self.act_import)
        menu.addSeparator()
        menu.addAction(self.act_run)
        menu.addSeparator()
        menu.addAction(self.act_export_shot)
        menu.addSeparator()
        menu.addAction(self.act_save)
        menu.addAction(self.act_save_as)
        menu.exec(pos_global)

    # =========================================================
    # Import / Clear
    # =========================================================
    def on_import_mesh(self) -> None:
        p, _ = QFileDialog.getOpenFileName(
            self,
            "Select mesh (ply/stl/obj)",
            str(self.app_root),
            "Mesh Files (*.ply *.stl *.obj);;All Files (*.*)",
        )
        if not p:
            return
        path = Path(p)

        stem = path.stem
        is_upper = bool(self._RE_UPPER.search(stem))
        is_lower = bool(self._RE_LOWER.search(stem))

        def decide() -> str:
            if is_upper and not is_lower:
                return "upper"
            if is_lower and not is_upper:
                return "lower"
            ret = QMessageBox.question(self, "Upper or Lower?", "ไฟล์นี้เป็น UPPER ใช่ไหม?\nYes=Upper, No=Lower")
            return "upper" if ret == QMessageBox.Yes else "lower"

        tag = decide()

        if self.stack.currentWidget() == self.page_result and hasattr(self, "tabs"):
            cur = self.tabs.currentWidget()
            if isinstance(cur, PerTabResultWidget):
                cur = cur.input_tab

            if isinstance(cur, StartLikeTab):
                if tag == "upper":
                    cur.mesh_upper = path
                    cur.viewer_upper.load_base_mesh(path, key="upper")
                    self._last_viewer = cur.viewer_upper
                else:
                    cur.mesh_lower = path
                    cur.viewer_lower.load_base_mesh(path, key="lower")
                    self._last_viewer = cur.viewer_lower

                self.statusBar().showMessage(f"Loaded: {path}", 4000)
                return

        # global import
        if tag == "upper":
            self.mesh_upper = path
            self.viewer_upper.load_base_mesh(path, key="upper")
            self._last_viewer = self.viewer_upper
        else:
            self.mesh_lower = path
            self.viewer_lower.load_base_mesh(path, key="lower")
            self._last_viewer = self.viewer_lower

        self.statusBar().showMessage(f"Loaded: {path}", 4000)

    def on_clear(self) -> None:
        self._stop_worker(force=True)
        self._set_running_ui(False, "Cleared")

        self.mesh_upper = None
        self.mesh_lower = None
        self.res_upper = None
        self.res_lower = None
        self._run_owner_tab = None

        self.job_queue.clear()
        self.current_job = None
        self.worker = None
        self._last_viewer = None

        self.viewer_upper.clear_scene()
        self.viewer_lower.clear_scene()
        self.viewer_result.clear_scene()

        if hasattr(self, "tabs"):
            for i in range(self.tabs.count()):
                if self.tabs.tabText(i) == "+":
                    continue
                w = self.tabs.widget(i)
                try:
                    if isinstance(w, StartLikeTab):
                        w.viewer_upper.clear_scene()
                        w.viewer_lower.clear_scene()
                        w.mesh_upper = None
                        w.mesh_lower = None
                    elif isinstance(w, PerTabResultWidget):
                        w.viewer.clear_scene()
                except Exception:
                    pass

        self.layer_panel.clear_layers()
        self.stack.setCurrentWidget(self.page_dual)

    # =========================================================
    # Run
    # =========================================================
    def _run_with_presets(self, *, preset_upper: str, preset_lower: str, owner_tab: Optional[StartLikeTab] = None) -> None:
        self._run_owner_tab = owner_tab

        mesh_upper = owner_tab.mesh_upper if owner_tab is not None else self.mesh_upper
        mesh_lower = owner_tab.mesh_lower if owner_tab is not None else self.mesh_lower

        if mesh_upper is None and mesh_lower is None:
            QMessageBox.warning(self, "No input", "กรุณา Import mesh ก่อน")
            return

        if self.worker is not None and self.worker.isRunning():
            QMessageBox.information(self, "Running", "กำลังรันอยู่")
            return

        self.res_upper = None
        self.res_lower = None
        self.job_queue.clear()
        self.current_job = None

        if mesh_upper is not None:
            self.job_queue.append(_Job("upper", mesh_upper, preset_upper))
        if mesh_lower is not None:
            self.job_queue.append(_Job("lower", mesh_lower, preset_lower))

        self._start_next_job()

    # =========================================================
    # Helpers (same as before)
    # =========================================================
    def _infer_gingiva_label(self, num_classes: int) -> Optional[int]:
        if num_classes == 17:
            return 16
        if num_classes == 33:
            return 32
        return None

    def _teeth_face_mask(self, labels_face: np.ndarray, num_classes: int) -> Optional[np.ndarray]:
        if labels_face is None:
            return None
        labels_face = np.asarray(labels_face)
        ging = self._infer_gingiva_label(int(num_classes))
        if ging is not None:
            return labels_face != ging
        try:
            ging2 = int(labels_face.max())
            return labels_face != ging2
        except Exception:
            return None

    def _face_centroids(self, pos: np.ndarray, faces: np.ndarray) -> np.ndarray:
        P = np.asarray(pos, dtype=np.float64)
        F = np.asarray(faces, dtype=np.int64)
        tri = P[F]
        return tri.mean(axis=1)

    def _canonicalize_pose(
        self,
        pos: np.ndarray,
        arch: str,
        *,
        faces: Optional[np.ndarray] = None,
        labels_face: Optional[np.ndarray] = None,
        num_classes: int = 0,
    ) -> np.ndarray:
        P = np.asarray(pos, dtype=np.float64)
        c = P.mean(axis=0)
        X = P - c

        C = (X.T @ X) / max(len(X) - 1, 1)
        _, V = np.linalg.eigh(C)
        v0, v1, v2 = V[:, 0], V[:, 1], V[:, 2]

        Zaxis = v0
        Xaxis = v2
        Yaxis = np.cross(Zaxis, Xaxis)
        nY = float(np.linalg.norm(Yaxis))
        if nY < 1e-9:
            Xaxis = v1
            Yaxis = np.cross(Zaxis, Xaxis)
            nY = float(np.linalg.norm(Yaxis))
            if nY < 1e-9:
                return (X + c).astype(np.float32)

        Yaxis /= nY
        Xaxis = np.cross(Yaxis, Zaxis)
        Xaxis /= (np.linalg.norm(Xaxis) + 1e-12)
        Zaxis /= (np.linalg.norm(Zaxis) + 1e-12)

        R = np.stack([Xaxis, Yaxis, Zaxis], axis=0)
        Xr = X @ R.T

        x = Xr[:, 0]
        x_lo = np.percentile(x, 10)
        x_hi = np.percentile(x, 90)
        left_band = Xr[x <= x_lo]
        right_band = Xr[x >= x_hi]

        def y_iqr(Pband: np.ndarray) -> float:
            if Pband.size == 0:
                return 1e9
            y = Pband[:, 1]
            q1, q3 = np.percentile(y, 25), np.percentile(y, 75)
            return float(q3 - q1)

        if y_iqr(right_band) > y_iqr(left_band):
            Xr[:, 0] *= -1

        Zuse = Xr[:, 2]
        if faces is not None and labels_face is not None and int(num_classes) > 0:
            try:
                cents = self._face_centroids(Xr + c, np.asarray(faces, dtype=np.int64)) - c
                m = self._teeth_face_mask(labels_face, num_classes=int(num_classes))
                if m is not None and m.shape[0] == cents.shape[0] and np.any(m):
                    Zuse = cents[m][:, 2]
            except Exception:
                Zuse = Xr[:, 2]

        z_top = float(np.percentile(Zuse, 95))
        z_bot = float(np.percentile(Zuse, 5))

        if arch == "upper":
            if z_top > abs(z_bot):
                Xr[:, 2] *= -1
        else:
            if abs(z_top) < abs(z_bot):
                Xr[:, 2] *= -1

        return (Xr + c).astype(np.float32)

    def _sample_teeth_points(
        self,
        pos: np.ndarray,
        faces: np.ndarray,
        labels_face: Optional[np.ndarray],
        num_classes: int,
        n_samples: int = 3000,
    ) -> np.ndarray:
        cents = self._face_centroids(pos, faces)
        if labels_face is not None:
            m = self._teeth_face_mask(labels_face, num_classes=num_classes)
            if m is not None and m.shape[0] == cents.shape[0] and np.any(m):
                cents = cents[m]
        if cents.shape[0] == 0:
            return np.asarray(pos, dtype=np.float64)
        if cents.shape[0] > n_samples:
            idx = np.random.choice(cents.shape[0], size=n_samples, replace=False)
            cents = cents[idx]
        return np.asarray(cents, dtype=np.float64)

    def _best_fit_transform(self, A: np.ndarray, B: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        assert A.shape == B.shape
        cA = A.mean(axis=0)
        cB = B.mean(axis=0)
        AA = A - cA
        BB = B - cB
        H = AA.T @ BB
        U, _, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        if np.linalg.det(R) < 0:
            Vt[2, :] *= -1
            R = Vt.T @ U.T
        t = cB - (cA @ R.T)
        return R, t

    def _apply_rt(self, P: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
        return (P @ R.T) + t

    def _nearest_neighbor(self, src: np.ndarray, dst: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        try:
            from scipy.spatial import cKDTree  # type: ignore
            tree = cKDTree(dst)
            d, idx = tree.query(src, k=1)
            return dst[idx], d
        except Exception:
            d2 = np.sum((src[:, None, :] - dst[None, :, :]) ** 2, axis=2)
            idx = np.argmin(d2, axis=1)
            d = np.sqrt(d2[np.arange(src.shape[0]), idx])
            return dst[idx], d

    def _icp_rigid(self, src: np.ndarray, dst: np.ndarray, iters: int = 45, trim: float = 0.75) -> Tuple[np.ndarray, np.ndarray]:
        P = np.asarray(src, dtype=np.float64)
        Q = np.asarray(dst, dtype=np.float64)

        R_all = np.eye(3, dtype=np.float64)
        t_all = np.zeros(3, dtype=np.float64)

        prev_err = None
        for _ in range(int(iters)):
            Qm, d = self._nearest_neighbor(P, Q)

            if d.size > 10:
                k = int(max(10, np.floor(d.size * float(trim))))
                keep = np.argsort(d)[:k]
                Pk = P[keep]
                Qk = Qm[keep]
            else:
                Pk, Qk = P, Qm

            R, t = self._best_fit_transform(Pk, Qk)
            P = self._apply_rt(P, R, t)

            t_all = (t_all @ R.T) + t
            R_all = R @ R_all

            err = float(np.mean(d))
            if prev_err is not None and abs(prev_err - err) < 1e-6:
                break
            prev_err = err

        return R_all, t_all

    def _nn_trimmed_error(self, src: np.ndarray, dst: np.ndarray, trim: float = 0.85) -> float:
        _, d = self._nearest_neighbor(src, dst)
        if d.size < 10:
            return float(np.mean(d)) if d.size else 1e9
        k = int(max(10, np.floor(d.size * float(trim))))
        dd = np.sort(d)[:k]
        return float(np.mean(dd))

    def _rot_z_180(self, P: np.ndarray) -> np.ndarray:
        Q = np.asarray(P, dtype=np.float64).copy()
        Q[:, 0] *= -1
        Q[:, 1] *= -1
        return Q.astype(np.float32)

    def _flip_x(self, P: np.ndarray) -> np.ndarray:
        Q = np.asarray(P, dtype=np.float64).copy()
        Q[:, 0] *= -1
        return Q.astype(np.float32)

    def _flip_y(self, P: np.ndarray) -> np.ndarray:
        Q = np.asarray(P, dtype=np.float64).copy()
        Q[:, 1] *= -1
        return Q.astype(np.float32)

    def _align_upper_to_lower_multihyp(
        self,
        posU: np.ndarray,
        posL: np.ndarray,
        facesU: np.ndarray,
        facesL: np.ndarray,
        labelsU: Optional[np.ndarray],
        labelsL: Optional[np.ndarray],
        numcU: int,
        numcL: int,
    ) -> np.ndarray:
        ptsL = self._sample_teeth_points(posL, facesL, labelsL, numcL, n_samples=3000)

        hyps = [
            ("id", lambda P: P),
            ("rz180", self._rot_z_180),
            ("fx", self._flip_x),
            ("fy", self._flip_y),
        ]

        best_posU = posU
        best_err = 1e18

        for _, fn in hyps:
            candU = fn(posU.copy())
            ptsU = self._sample_teeth_points(candU, facesU, labelsU, numcU, n_samples=3000)

            cL = np.median(ptsL, axis=0)
            cU = np.median(ptsU, axis=0)
            dx = float(cL[0] - cU[0])
            dy = float(cL[1] - cU[1])

            candU[:, 0] += dx
            candU[:, 1] += dy

            ptsU2 = ptsU.copy()
            ptsU2[:, 0] += dx
            ptsU2[:, 1] += dy

            R, t = self._icp_rigid(ptsU2, ptsL, iters=45, trim=0.75)
            candU_aligned = self._apply_rt(candU.astype(np.float64), R, t).astype(np.float32)
            ptsU_aligned = self._apply_rt(ptsU2, R, t)

            err = self._nn_trimmed_error(ptsU_aligned, ptsL, trim=0.85)
            if err < best_err:
                best_err = err
                best_posU = candU_aligned

        return best_posU

    def _close_bite_by_z(
        self,
        upper_pos: np.ndarray,
        lower_pos: np.ndarray,
        *,
        upper_teeth_pts: np.ndarray,
        lower_teeth_pts: np.ndarray,
        target_gap: float = 0.5,
        safety: float = 0.15,
    ) -> np.ndarray:
        U = np.asarray(upper_pos, dtype=np.float64).copy()
        L = np.asarray(lower_pos, dtype=np.float64)

        Upts = np.asarray(upper_teeth_pts, dtype=np.float64)
        Lpts = np.asarray(lower_teeth_pts, dtype=np.float64)

        uz = Upts[:, 2]
        lz = Lpts[:, 2]

        upper_bottom = float(np.percentile(uz, 5))
        lower_top = float(np.percentile(lz, 95))
        dz = upper_bottom - (lower_top + float(target_gap))

        l_top_mesh = float(np.percentile(L[:, 2], 98))
        min_allowed_upper_bottom = l_top_mesh + float(safety)

        upper_bottom_after = float(np.percentile(U[:, 2] - dz, 2))
        if upper_bottom_after < min_allowed_upper_bottom:
            dz = float(np.percentile(U[:, 2], 2) - min_allowed_upper_bottom)

        U[:, 2] -= dz
        return U.astype(np.float32)

    def _set_result_camera_side(self, *, center: Optional[np.ndarray] = None) -> None:
        vw = self._active_viewer()
        try:
            plotter = getattr(vw, "plotter", None)
            if plotter is None:
                try:
                    vw.set_view("xz")
                except Exception:
                    vw.reset_view()
                return

            bounds = None
            try:
                bounds = plotter.renderer.ComputeVisiblePropBounds()
            except Exception:
                bounds = None

            if center is None:
                if bounds and len(bounds) == 6:
                    x0, x1, y0, y1, z0, z1 = bounds
                    center = np.array([(x0 + x1) * 0.5, (y0 + y1) * 0.5, (z0 + z1) * 0.5], dtype=np.float64)
                else:
                    center = np.array([0.0, 0.0, 0.0], dtype=np.float64)

            if bounds and len(bounds) == 6:
                x0, x1, y0, y1, z0, z1 = bounds
                diag = float(np.linalg.norm([x1 - x0, y1 - y0, z1 - z0]))
                dist = max(diag * 1.2, 50.0)
            else:
                dist = 200.0

            cam = plotter.camera
            cam.position = (float(center[0]), float(center[1] + dist), float(center[2]))
            cam.focal_point = (float(center[0]), float(center[1]), float(center[2]))
            cam.up = (0.0, 0.0, 1.0)
            try:
                cam.zoom(1.20)
            except Exception:
                pass
            plotter.render()
        except Exception:
            try:
                vw.reset_view()
            except Exception:
                pass

    def _palette_for_arch(self, arch: str) -> Dict[str, Tuple[int, int, int]]:
        try:
            from app.data.fdi_colors import FDI_TO_RGB, FDI_LIST_UPPER_16, FDI_LIST_LOWER_16
            lst = FDI_LIST_UPPER_16 if arch == "upper" else FDI_LIST_LOWER_16
            out: Dict[str, Tuple[int, int, int]] = {}
            for fdi in lst:
                rgb = FDI_TO_RGB[int(fdi)]
                out[f"FDI {int(fdi)}"] = (int(rgb[0]), int(rgb[1]), int(rgb[2]))
            out["GINGIVA"] = (200, 200, 200)
            return out
        except Exception:
            return {"GINGIVA": (200, 200, 200)}

    # =========================================================
    # Result rendering (FIX: delay render for per-tab result)
    # =========================================================
    def _show_result(self) -> None:
        self.stack.setCurrentWidget(self.page_result)
        owner = getattr(self, "_run_owner_tab", None)

        # compute pose / align / close bite
        posL = None
        posU = None

        if self.res_lower is not None:
            res = self.res_lower
            posL = self._canonicalize_pose(
                res.mesh_proc.pos,
                "lower",
                faces=res.mesh_proc.faces,
                labels_face=getattr(res, "labels_face", None),
                num_classes=int(getattr(res, "num_classes", 0) or 0),
            )

        if self.res_upper is not None:
            res = self.res_upper
            posU = self._canonicalize_pose(
                res.mesh_proc.pos,
                "upper",
                faces=res.mesh_proc.faces,
                labels_face=getattr(res, "labels_face", None),
                num_classes=int(getattr(res, "num_classes", 0) or 0),
            )

        if posL is not None and posU is not None:
            resL = self.res_lower
            resU = self.res_upper

            posU = self._align_upper_to_lower_multihyp(
                posU=posU,
                posL=posL,
                facesU=resU.mesh_proc.faces,
                facesL=resL.mesh_proc.faces,
                labelsU=getattr(resU, "labels_face", None),
                labelsL=getattr(resL, "labels_face", None),
                numcU=int(getattr(resU, "num_classes", 0) or 0),
                numcL=int(getattr(resL, "num_classes", 0) or 0),
            )

            ptsL = self._sample_teeth_points(
                posL,
                resL.mesh_proc.faces,
                getattr(resL, "labels_face", None),
                int(getattr(resL, "num_classes", 0) or 0),
                n_samples=3500,
            )
            ptsU = self._sample_teeth_points(
                posU,
                resU.mesh_proc.faces,
                getattr(resU, "labels_face", None),
                int(getattr(resU, "num_classes", 0) or 0),
                n_samples=3500,
            )

            posU = self._close_bite_by_z(
                posU,
                posL,
                upper_teeth_pts=ptsU,
                lower_teeth_pts=ptsL,
                target_gap=0.5,
                safety=0.15,
            )

        # per-tab: replace StartLikeTab -> PerTabResultWidget and delay render
        if isinstance(owner, StartLikeTab):
            idx_owner = self.tabs.indexOf(owner)
            if idx_owner >= 0:
                per_res = PerTabResultWidget(self, input_tab=owner)
                self._replace_tab_widget(idx_owner, per_res)
            else:
                per_res = PerTabResultWidget(self, input_tab=owner)

            # hide global layer panel
            self._set_layers_visible(False)

            # prepare layers first (panel shows even before VTK ready)
            layers: List[LayerInfo] = []
            if self.res_upper is not None:
                res = self.res_upper
                case_id = getattr(res.mesh_proc, "case_id", "case")
                layers.append(
                    LayerInfo(
                        name=f"{case_id}_upper",
                        mesh_path=owner.mesh_upper,
                        n_verts=int(len(res.mesh_proc.pos)),
                        n_faces=int(len(res.mesh_proc.faces)),
                        arch="upper",
                        palette=self._palette_for_arch("upper"),
                    )
                )
            if self.res_lower is not None:
                res = self.res_lower
                case_id = getattr(res.mesh_proc, "case_id", "case")
                layers.append(
                    LayerInfo(
                        name=f"{case_id}_lower",
                        mesh_path=owner.mesh_lower,
                        n_verts=int(len(res.mesh_proc.pos)),
                        n_faces=int(len(res.mesh_proc.faces)),
                        arch="lower",
                        palette=self._palette_for_arch("lower"),
                    )
                )
            per_res.layer_panel.set_layers(layers)

            # ✅ crucial: delay one tick so QtInteractor gets GL context
            def _render_later():
                viewer = per_res.viewer
                viewer.clear_scene()

                if self.res_lower is not None and posL is not None:
                    resL = self.res_lower
                    viewer.set_base_mesh_from_arrays("lower", posL, resL.mesh_proc.faces)
                    rgbL = label_array_to_rgb_face(resL.labels_face, arch=resL.mesh_proc.arch, num_classes=int(resL.num_classes))
                    viewer.set_overlay_face_rgb("lower", rgbL, opacity=0.90)

                if self.res_upper is not None and posU is not None:
                    resU = self.res_upper
                    viewer.set_base_mesh_from_arrays("upper", posU, resU.mesh_proc.faces)
                    rgbU = label_array_to_rgb_face(resU.labels_face, arch=resU.mesh_proc.arch, num_classes=int(resU.num_classes))
                    viewer.set_overlay_face_rgb("upper", rgbU, opacity=0.90)

                viewer.reset_view()

                cam_center = None
                try:
                    centers = []
                    if posL is not None:
                        centers.append(np.mean(posL, axis=0))
                    if posU is not None:
                        centers.append(np.mean(posU, axis=0))
                    if centers:
                        cam_center = np.mean(np.stack(centers), axis=0)
                except Exception:
                    cam_center = None

                self._set_result_camera_side(center=cam_center)

            QTimer.singleShot(0, _render_later)
            return

        # classic Result tab
        self.tabs.setCurrentIndex(0)
        self._set_layers_visible(True)

        viewer = self.viewer_result
        viewer.clear_scene()

        layers: List[LayerInfo] = []

        if self.res_lower is not None and posL is not None:
            res = self.res_lower
            viewer.set_base_mesh_from_arrays("lower", posL, res.mesh_proc.faces)
            face_rgb = label_array_to_rgb_face(res.labels_face, arch=res.mesh_proc.arch, num_classes=int(res.num_classes))
            viewer.set_overlay_face_rgb("lower", face_rgb, opacity=0.90)
            case_id = getattr(res.mesh_proc, "case_id", "case")
            layers.append(
                LayerInfo(
                    name=f"{case_id}_lower",
                    mesh_path=self.mesh_lower,
                    n_verts=int(len(res.mesh_proc.pos)),
                    n_faces=int(len(res.mesh_proc.faces)),
                    arch="lower",
                    palette=self._palette_for_arch("lower"),
                )
            )

        if self.res_upper is not None and posU is not None:
            res = self.res_upper
            viewer.set_base_mesh_from_arrays("upper", posU, res.mesh_proc.faces)
            face_rgb = label_array_to_rgb_face(res.labels_face, arch=res.mesh_proc.arch, num_classes=int(res.num_classes))
            viewer.set_overlay_face_rgb("upper", face_rgb, opacity=0.90)
            case_id = getattr(res.mesh_proc, "case_id", "case")
            layers.insert(
                0,
                LayerInfo(
                    name=f"{case_id}_upper",
                    mesh_path=self.mesh_upper,
                    n_verts=int(len(res.mesh_proc.pos)),
                    n_faces=int(len(res.mesh_proc.faces)),
                    arch="upper",
                    palette=self._palette_for_arch("upper"),
                ),
            )

        viewer.reset_view()
        self.layer_panel.set_layers(layers)

        cam_center = None
        try:
            centers = []
            if posL is not None:
                centers.append(np.mean(posL, axis=0))
            if posU is not None:
                centers.append(np.mean(posU, axis=0))
            if centers:
                cam_center = np.mean(np.stack(centers), axis=0)
        except Exception:
            cam_center = None

        self._set_result_camera_side(center=cam_center)

    # =========================================================
    # Project save
    # =========================================================
    def on_save_project(self) -> None:
        if self.project_path is None:
            self.on_save_as_project()
            return
        self._save_project(self.project_path)

    def on_save_as_project(self) -> None:
        out, _ = QFileDialog.getSaveFileName(
            self,
            "Save project as",
            str(Path.home() / f"dental_seg.{PROJECT_EXT}"),
            f"Project (*.{PROJECT_EXT});;All Files (*.*)",
        )
        if not out:
            return
        p = Path(out)
        if not p.name.endswith(PROJECT_EXT):
            p = p.with_name(p.name + "." + PROJECT_EXT)
        self.project_path = p
        self._save_project(p)

    def _save_project(self, p: Path) -> None:
        data = {
            "mesh_upper": str(self.mesh_upper) if self.mesh_upper else "",
            "mesh_lower": str(self.mesh_lower) if self.mesh_lower else "",
            "preset_upper": self.cb_upper.currentText().strip(),
            "preset_lower": self.cb_lower.currentText().strip(),
        }
        try:
            p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
            self.statusBar().showMessage(f"Saved project: {p}", 4000)
        except Exception as e:
            QMessageBox.critical(self, "Save failed", str(e))

    # =========================================================
    # Export
    # =========================================================
    def on_export_screenshot(self) -> None:
        out, _ = QFileDialog.getSaveFileName(
            self,
            "Save screenshot",
            str(Path.home() / "screenshot.png"),
            "PNG (*.png);;All Files (*.*)",
        )
        if not out:
            return
        try:
            self._active_viewer().export_screenshot(out)
            self.statusBar().showMessage(f"Saved screenshot: {out}", 3000)
        except Exception as e:
            QMessageBox.critical(self, "Export error", str(e))