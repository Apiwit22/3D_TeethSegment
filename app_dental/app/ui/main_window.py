from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np

from PySide6.QtCore import Qt, QLocale
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMenu,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QSplitter,
    QStackedWidget,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from app.core.registry import Registry
from app.jobs.worker import InferenceWorker
from app.ui.result_presenter import ResultPresenter
from app.ui.tab_manager import TabManager
from app.ui.widgets.layer_panel import LayerPanel
from app.ui.widgets.result_tabs import StartLikeTab, paired_preset_key
from app.ui.widgets.viewer_widget import ViewerWidget

PROJECT_EXT = "dsegproj.json"


@dataclass
class _Job:
    tag: str
    mesh_path: Path
    preset_key: str


class MainWindow(QMainWindow):
    _RE_UPPER = re.compile(r"(^|[_\-\s])(u|upper)([_\-\s]|$)", re.IGNORECASE)
    _RE_LOWER = re.compile(r"(^|[_\-\s])(l|lower)([_\-\s]|$)", re.IGNORECASE)

    def __init__(self, app_root: Path):
        super().__init__()
        self.setWindowTitle("DentalSeg")
        self.resize(1400, 820)

        self.app_root = Path(app_root)
        self.registry = Registry(self.app_root / "registry.yaml").load()

        self.mesh_upper: Optional[Path] = None
        self.mesh_lower: Optional[Path] = None
        self.project_path: Optional[Path] = None

        self.worker: Optional[InferenceWorker] = None
        self.job_queue: List[_Job] = []
        self.current_job: Optional[_Job] = None

        self.res_upper = None
        self.res_lower = None

        self._is_closing = False
        self._last_viewer: Optional[ViewerWidget] = None
        self._run_owner_tab: Optional[StartLikeTab] = None

        self.stack = QStackedWidget()
        self.setCentralWidget(self.stack)

        self.page_dual = self._build_page_dual()
        self.page_result = self._build_page_result()

        self.stack.addWidget(self.page_dual)
        self.stack.addWidget(self.page_result)
        self.stack.setCurrentWidget(self.page_dual)

        self.tab_manager = TabManager(
            mainwin=self,
            tabs=self.tabs,
            layer_panel=self.layer_panel,
            result_splitter=self.result_splitter,
        )
        self.tab_manager.install()

        self.result_presenter = ResultPresenter(self)

        self._build_actions()
        self._build_menubar()
        self._build_status_widgets()

        self.setStyleSheet(
            """
            QMainWindow { background: #0b0f18; }
            QMenuBar { background: rgba(255,255,255,0.02); color: rgba(255,255,255,0.92); }
            QMenuBar::item { padding: 6px 10px; }
            QMenuBar::item:selected { background: rgba(255,255,255,0.06); border-radius: 8px; }
            QMenu { background: #121826; color: rgba(255,255,255,0.92); border: 1px solid rgba(255,255,255,0.12); }
            QMenu::item:selected { background: rgba(255,255,255,0.10); }
            QStatusBar { background: rgba(255,255,255,0.03); color: rgba(255,255,255,0.88); }

            #StatusProgressBar {
              min-width: 180px;
              max-width: 220px;
              border: 1px solid rgba(255,255,255,0.12);
              border-radius: 7px;
              text-align: center;
              background: rgba(255,255,255,0.06);
              color: white;
            }
            #StatusProgressBar::chunk {
              background: rgba(120,160,255,0.85);
              border-radius: 6px;
            }
            """
        )
        self._set_progress_ui(0, "")
        self.statusBar().showMessage("")

    # =========================================================
    # Status / progress
    # =========================================================
    def _build_status_widgets(self) -> None:
        self.status_progress = QProgressBar()
        self.status_progress.setObjectName("StatusProgressBar")
        self.status_progress.setRange(0, 100)
        self.status_progress.setValue(0)
        self.status_progress.setTextVisible(True)
        self.status_progress.setFormat("0%")

        try:
            self.status_progress.setLocale(
                QLocale(QLocale.Language.English, QLocale.Country.UnitedStates)
            )
        except Exception:
            pass

        self.statusBar().addPermanentWidget(self.status_progress)

    def _set_progress_ui(self, percent: int, phase: str) -> None:
        percent = max(0, min(100, int(percent)))
        if hasattr(self, "status_progress"):
            self.status_progress.setValue(percent)
            self.status_progress.setFormat(f"{percent}%")

    # =========================================================
    # Qt lifecycle
    # =========================================================
    def closeEvent(self, event) -> None:
        self._is_closing = True
        self._stop_worker(force=True)

        for vw in (
            getattr(self, "viewer_upper", None),
            getattr(self, "viewer_lower", None),
            getattr(self, "viewer_result", None),
        ):
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
                    elif hasattr(w, "viewer"):
                        w.viewer.shutdown()
                    elif hasattr(w, "shutdown"):
                        w.shutdown()
                except Exception:
                    pass

        event.accept()

    # =========================================================
    # Viewer helpers
    # =========================================================
    def _active_viewer(self) -> ViewerWidget:
        if self.stack.currentWidget() == self.page_result and hasattr(self, "tabs"):
            cur = self.tabs.currentWidget()
            if isinstance(cur, ViewerWidget):
                return cur
            if hasattr(cur, "viewer"):
                return cur.viewer
            if isinstance(cur, StartLikeTab):
                if self._last_viewer is not None:
                    return self._last_viewer
                return cur.viewer_upper
            return self.viewer_result

        if self._last_viewer is not None:
            return self._last_viewer
        return self.viewer_upper

    def _restore_input_tab(self, per_tab_result) -> None:
        self.tab_manager.restore_input_tab(per_tab_result)

    # =========================================================
    # Run state
    # =========================================================
    def _set_all_run_buttons_enabled(self, enabled: bool) -> None:
        if hasattr(self, "btn_run"):
            self.btn_run.setEnabled(enabled)
            self.btn_run.setText("▶" if enabled else "⏳")

        if hasattr(self, "tabs"):
            for i in range(self.tabs.count()):
                w = self.tabs.widget(i)
                if isinstance(w, StartLikeTab):
                    try:
                        w.btn_run.setEnabled(enabled)
                        w.btn_run.setText("▶" if enabled else "⏳")
                    except Exception:
                        pass

    def _set_running_ui(self, running: bool, msg: str = "") -> None:
        self._set_all_run_buttons_enabled(not running)
        if hasattr(self, "act_run"):
            self.act_run.setEnabled(not running)

        if not running:
            self._set_progress_ui(0, "")
        elif msg:
            self.statusBar().showMessage(msg, 3000)

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
            self._set_running_ui(False, "")
            self.statusBar().showMessage("Segmentation finished", 2500)
            self.result_presenter.show()
            return

        self.current_job = self.job_queue.pop(0)
        job = self.current_job
        self._set_running_ui(True, f"Running {job.tag.upper()}...")
        self._set_progress_ui(0, "")

        self.worker = InferenceWorker(
            app_root=self.app_root,
            mesh_path=job.mesh_path,
            preset_key=job.preset_key,
            device="cuda",
        )
        self.worker.signals.log.connect(lambda s: self._status(job.tag.upper(), s))
        self.worker.signals.progress.connect(
            lambda p: self._on_worker_progress(job.tag.upper(), p)
        )
        self.worker.signals.done.connect(self._on_job_done)
        self.worker.signals.failed.connect(self._on_job_failed)
        self.worker.start()

    def _on_worker_progress(self, tag: str, percent: int) -> None:
        self._set_progress_ui(percent, "")

    def _status(self, tag: str, msg: str) -> None:
        self.statusBar().showMessage(f"[{tag}] {msg}", 2500)

    def _on_job_failed(self, msg: str) -> None:
        self.job_queue.clear()
        self.current_job = None
        self._set_running_ui(False, "")
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
    # Actions / menus
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
        self.act_view_reset.triggered.connect(
            lambda: self._active_viewer().reset_view()
        )

        self.act_view_side = QAction("Standard Side View", self)
        self.act_view_side.setShortcut(QKeySequence("Ctrl+1"))
        self.act_view_side.triggered.connect(lambda: self._set_result_camera_side())

        self.act_view_iso = QAction("Isometric", self)
        self.act_view_iso.triggered.connect(
            lambda: self._active_viewer().set_view("iso")
        )

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
    # Page 1: input/import
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

        self.viewer_upper.rightClicked.connect(
            lambda pos: self._show_context_menu_from(self.viewer_upper, pos)
        )
        self.viewer_lower.rightClicked.connect(
            lambda pos: self._show_context_menu_from(self.viewer_lower, pos)
        )

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
        lb.setStyleSheet(
            "color: rgba(255,255,255,0.92); font-size: 16px; letter-spacing: 1px;"
        )
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

        hidden = {
            "tsmdl_upper", "tsmdl_lower",
            "pointnetpp_upper", "pointnetpp_lower",
        }

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

    # =========================================================
    # Page 2: result/tabs
    # =========================================================
    def _build_page_result(self) -> QWidget:
        root = QWidget()
        root.setObjectName("ResultPageRoot")
        lay = QHBoxLayout(root)
        lay.setContentsMargins(12, 12, 12, 12)
        lay.setSpacing(12)

        self.tabs = QTabWidget()
        self.tabs.setObjectName("ResultTabs")
        self.tabs.setDocumentMode(True)
        self.tabs.setMovable(True)
        self.tabs.setUpdatesEnabled(True)

        self.viewer_result = ViewerWidget()
        self.viewer_result.rightClicked.connect(
            lambda pos: self._show_context_menu_from(self.viewer_result, pos)
        )
        self.tabs.addTab(self.viewer_result, "Tab 1")
        self.tabs.addTab(QWidget(), "+")
        self.tabs.setCurrentIndex(0)

        root.setStyleSheet(
            """
            #ResultPageRoot {
              background: #0b1220;
            }

            QTabWidget#ResultTabs::pane {
              border: 1px solid rgba(255,255,255,0.10);
              top: -1px;
              background: #0b1220;
              border-radius: 12px;
            }

            QTabWidget#ResultTabs QWidget {
              background: #0b1220;
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
            QTabBar::tab:hover {
              background: rgba(255,255,255,0.05);
            }
            """
        )

        self.layer_panel = LayerPanel()
        self.layer_panel.btn_back.clicked.connect(
            lambda: self.stack.setCurrentWidget(self.page_dual)
        )
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

        return root

    # =========================================================
    # Context menu / import / remove
    # =========================================================
    def _show_context_menu_from(self, viewer: ViewerWidget, pos_global) -> None:
        self._last_viewer = viewer
        menu = QMenu(self)

        menu.setStyleSheet(
            """
            QMenu {
              background: #121826;
              color: rgba(255,255,255,0.92);
              border: 1px solid rgba(255,255,255,0.12);
              padding: 6px 4px;
            }
            QMenu::item {
              padding: 8px 34px 8px 14px;
              min-width: 210px;
            }
            QMenu::item:selected {
              background: rgba(255,255,255,0.10);
            }
            QMenu::separator {
              height: 1px;
              background: rgba(255,255,255,0.08);
              margin: 4px 8px;
            }
            """
        )

        act_import = menu.addAction("Import Mesh...")
        act_import.setShortcut(QKeySequence.Open)

        menu.addSeparator()

        act_remove_upper = menu.addAction("Remove Upper Mesh")
        act_remove_lower = menu.addAction("Remove Lower Mesh")

        menu.addSeparator()

        act_run = menu.addAction("Run Segmentation")
        act_run.setShortcut(QKeySequence("Ctrl+R"))

        menu.addSeparator()

        act_export = menu.addAction("Export Screenshot...")

        act_save = menu.addAction("Save Project")
        act_save.setShortcut(QKeySequence.Save)

        act_save_as = menu.addAction("Save Project As...")
        act_save_as.setShortcut(QKeySequence.SaveAs)

        menu.setMinimumWidth(260)

        chosen = menu.exec(pos_global)
        if chosen is None:
            return

        owner_tab, _default_target = self._focused_target_for_import()

        if chosen == act_import:
            self.on_import_mesh()
            return

        if chosen == act_remove_upper:
            self._clear_mesh_slot("upper", owner_tab=owner_tab)
            return

        if chosen == act_remove_lower:
            self._clear_mesh_slot("lower", owner_tab=owner_tab)
            return

        if chosen == act_run:
            if owner_tab is not None:
                self._run_with_presets(
                    preset_upper=owner_tab.cb_upper.currentText().strip(),
                    preset_lower=owner_tab.cb_lower.currentText().strip(),
                    owner_tab=owner_tab,
                )
            else:
                self._run_with_presets(
                    preset_upper=self.cb_upper.currentText().strip(),
                    preset_lower=self.cb_lower.currentText().strip(),
                    owner_tab=None,
                )
            return

        if chosen == act_export:
            self.on_export_screenshot()
            return

        if chosen == act_save:
            self.on_save_project()
            return

        if chosen == act_save_as:
            self.on_save_as_project()
            return

    def _focused_target_for_import(self):
        if self.stack.currentWidget() == self.page_result and hasattr(self, "tabs"):
            cur = self.tabs.currentWidget()
            if hasattr(cur, "input_tab"):
                cur = cur.input_tab
            if isinstance(cur, StartLikeTab):
                if cur.cb_upper.hasFocus() or cur.viewer_upper.hasFocus():
                    return cur, "upper"
                if cur.cb_lower.hasFocus() or cur.viewer_lower.hasFocus():
                    return cur, "lower"
                return cur, "upper"

        if self.cb_upper.hasFocus() or self.viewer_upper.hasFocus():
            return None, "upper"
        if self.cb_lower.hasFocus() or self.viewer_lower.hasFocus():
            return None, "lower"
        return None, "upper"

    def _clear_mesh_slot(self, tag: str, owner_tab: Optional[StartLikeTab] = None) -> None:
        tag = (tag or "").strip().lower()
        if tag not in {"upper", "lower"}:
            return

        self.res_upper = None
        self.res_lower = None
        self.job_queue.clear()
        self.current_job = None
        self.worker = None
        self._run_owner_tab = None

        if owner_tab is not None:
            if tag == "upper":
                owner_tab.mesh_upper = None
                owner_tab.viewer_upper.clear_scene()
                self._last_viewer = owner_tab.viewer_lower
            else:
                owner_tab.mesh_lower = None
                owner_tab.viewer_lower.clear_scene()
                self._last_viewer = owner_tab.viewer_upper

            self.statusBar().showMessage(f"Removed {tag} mesh from current tab", 2500)
            return

        if tag == "upper":
            self.mesh_upper = None
            self.viewer_upper.clear_scene()
            self._last_viewer = self.viewer_lower
        else:
            self.mesh_lower = None
            self.viewer_lower.clear_scene()
            self._last_viewer = self.viewer_upper

        self.statusBar().showMessage(f"Removed {tag} mesh", 2500)

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

        owner_tab, default_target = self._focused_target_for_import()

        def decide() -> str:
            if is_upper and not is_lower:
                return "upper"
            if is_lower and not is_upper:
                return "lower"

            if default_target == "lower":
                ret = QMessageBox.question(
                    self,
                    "Confirm Jaw Type",
                    "Could not determine the jaw type from the file name.\n\n"
                    "You are importing into the LOWER slot.\n"
                    "Is this mesh LOWER?\n\n"
                    "Yes = Lower, No = Upper",
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.Yes,
                )
                return "lower" if ret == QMessageBox.Yes else "upper"

            ret = QMessageBox.question(
                self,
                "Confirm Jaw Type",
                "Could not determine the jaw type from the file name.\n\n"
                "You are importing into the UPPER slot.\n"
                "Is this mesh UPPER?\n\n"
                "Yes = Upper, No = Lower",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.Yes,
            )
            return "upper" if ret == QMessageBox.Yes else "lower"

        tag = decide()

        if owner_tab is not None:
            existing = owner_tab.mesh_upper if tag == "upper" else owner_tab.mesh_lower
            if existing is not None:
                ret = QMessageBox.question(
                    self,
                    "Replace mesh",
                    f"The {tag.upper()} slot already has a mesh.\n\nReplace it with the new mesh?",
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.Yes,
                )
                if ret != QMessageBox.Yes:
                    return

            if tag == "upper":
                owner_tab.mesh_upper = path
                owner_tab.viewer_upper.load_base_mesh(path, key="upper")
                self._last_viewer = owner_tab.viewer_upper
            else:
                owner_tab.mesh_lower = path
                owner_tab.viewer_lower.load_base_mesh(path, key="lower")
                self._last_viewer = owner_tab.viewer_lower

            self.res_upper = None
            self.res_lower = None
            self.statusBar().showMessage(f"Loaded: {path}", 3000)
            return

        existing = self.mesh_upper if tag == "upper" else self.mesh_lower
        if existing is not None:
            ret = QMessageBox.question(
                self,
                "Replace mesh",
                f"The {tag.upper()} slot already has a mesh.\n\nReplace it with the new mesh?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.Yes,
            )
            if ret != QMessageBox.Yes:
                return

        if tag == "upper":
            self.mesh_upper = path
            self.viewer_upper.load_base_mesh(path, key="upper")
            self._last_viewer = self.viewer_upper
        else:
            self.mesh_lower = path
            self.viewer_lower.load_base_mesh(path, key="lower")
            self._last_viewer = self.viewer_lower

        self.res_upper = None
        self.res_lower = None
        self.statusBar().showMessage(f"Loaded: {path}", 3000)

    # =========================================================
    # Clear / run
    # =========================================================
    def on_clear(self) -> None:
        self._stop_worker(force=True)
        self._set_running_ui(False, "")

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
                    elif hasattr(w, "viewer"):
                        w.viewer.clear_scene()
                except Exception:
                    pass

        self.layer_panel.clear_layers()
        self.stack.setCurrentWidget(self.page_dual)
        self._set_progress_ui(0, "")

    def _run_with_presets(
        self,
        *,
        preset_upper: str,
        preset_lower: str,
        owner_tab: Optional[StartLikeTab] = None,
    ) -> None:
        self._run_owner_tab = owner_tab

        mesh_upper = owner_tab.mesh_upper if owner_tab is not None else self.mesh_upper
        mesh_lower = owner_tab.mesh_lower if owner_tab is not None else self.mesh_lower

        if mesh_upper is None and mesh_lower is None:
            QMessageBox.warning(self, "No input", "Please import a mesh first.")
            return

        if self.worker is not None and self.worker.isRunning():
            QMessageBox.information(self, "Running", "Segmentation is already running.")
            return

        self.res_upper = None
        self.res_lower = None
        self.job_queue.clear()
        self.current_job = None
        self._set_progress_ui(0, "")

        if mesh_upper is not None:
            self.job_queue.append(_Job("upper", mesh_upper, preset_upper))
        if mesh_lower is not None:
            self.job_queue.append(_Job("lower", mesh_lower, preset_lower))

        self._start_next_job()

    # =========================================================
    # Camera
    # =========================================================
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
                    center = np.array(
                        [(x0 + x1) * 0.5, (y0 + y1) * 0.5, (z0 + z1) * 0.5],
                        dtype=np.float64,
                    )
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

    # =========================================================
    # Save / export
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
            self.statusBar().showMessage(f"Saved project: {p}", 3000)
        except Exception as e:
            QMessageBox.critical(self, "Save failed", str(e))

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
            self.statusBar().showMessage(f"Saved screenshot: {out}", 2500)
        except Exception as e:
            QMessageBox.critical(self, "Export error", str(e))