from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, List, Dict, Any

from PySide6.QtCore import Qt, QSettings
from PySide6.QtGui import QAction, QKeySequence, QIcon
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QDockWidget, QVBoxLayout,
    QLabel, QPushButton, QFileDialog, QComboBox, QProgressBar,
    QPlainTextEdit, QCheckBox, QSlider, QMessageBox, QMenu, QToolBar,
    QToolButton
)

from app.core.registry import Registry
from app.jobs.worker import InferenceWorker
from app.ui.widgets.viewer_widget import ViewerWidget
from app.data.color_map import label_array_to_rgb_face
from app.core.exporters import export_colored_ply_facecolors, export_split_by_tooth


PROJECT_EXT = "dsegproj.json"
SETTINGS_ORG = "DentalSeg"
SETTINGS_APP = "DentalSegApp"


class MainWindow(QMainWindow):
    """
    Minimal UI:
    - Toolbar: Run | Export▼ | Reset View
    - Import/Save/SaveAs: อยู่ใน File menu เท่านั้น
    - เปิดแอพ: มีแค่ Viewer (dock ซ้าย/ขวาซ่อน)
    - Import mesh: dock ซ้าย/ขวาโผล่
    - Right-click viewer: context menu
    - Run done: overlay สีขึ้นแน่นอน (ถ้า res.labels_face + mesh_proc พร้อม)
    """

    def __init__(self, app_root: Path):
        super().__init__()
        self.setWindowTitle("Dental Segmentation App")
        self.resize(1400, 800)

        self.app_root = Path(app_root)
        self.registry = Registry(self.app_root / "registry.yaml").load()

        self.settings = QSettings(SETTINGS_ORG, SETTINGS_APP)

        self.current_mesh: Optional[Path] = None
        self.current_project: Optional[Path] = None
        self.worker: Optional[InferenceWorker] = None
        self.last_result = None

        # --- central viewer ---
        self.viewer = ViewerWidget(self)
        self.setCentralWidget(self.viewer)
        self.viewer.rightClicked.connect(self._show_viewer_context_menu)

        # --- docks (hidden initially) ---
        self.dock_left = self._build_left_dock()
        self.dock_right = self._build_right_dock()
        self.addDockWidget(Qt.LeftDockWidgetArea, self.dock_left)
        self.addDockWidget(Qt.RightDockWidgetArea, self.dock_right)
        self.dock_left.setVisible(False)
        self.dock_right.setVisible(False)

        # --- actions / menu / toolbar ---
        self._build_actions()
        self._build_menu()
        self._build_toolbar_minimal()

        # presets + recent
        self._refresh_presets()
        self._rebuild_recent_menu()

        self.statusBar().showMessage("Ready")

    # =========================================================
    # Actions
    # =========================================================
    def _build_actions(self) -> None:
        # File (menu only)
        self.act_import = QAction("Import Mesh...", self)
        self.act_import.setShortcut(QKeySequence.Open)
        self.act_import.triggered.connect(self.on_import_mesh)

        self.act_open_project = QAction("Open Project...", self)
        self.act_open_project.triggered.connect(self.on_open_project)

        self.act_save = QAction("Save Project", self)
        self.act_save.setShortcut(QKeySequence.Save)
        self.act_save.triggered.connect(self.on_save_project)

        self.act_save_as = QAction("Save Project As...", self)
        self.act_save_as.setShortcut(QKeySequence.SaveAs)
        self.act_save_as.triggered.connect(self.on_save_as_project)

        self.act_exit = QAction("Exit", self)
        self.act_exit.setShortcut(QKeySequence.Quit)
        self.act_exit.triggered.connect(self.close)

        # Run
        self.act_run = QAction("Run", self)
        self.act_run.setShortcut(QKeySequence("Ctrl+R"))
        self.act_run.triggered.connect(self.on_run)

        # Export
        self.act_export_shot = QAction("Screenshot...", self)
        self.act_export_shot.triggered.connect(self.on_export_screenshot)

        self.act_export_ply = QAction("Colored PLY (face color)...", self)
        self.act_export_ply.triggered.connect(self.on_export_colored_ply)

        self.act_export_split = QAction("Split by Tooth...", self)
        self.act_export_split.triggered.connect(self.on_export_split)

        # View
        self.act_view_reset = QAction("Reset View", self)
        self.act_view_reset.setShortcut(QKeySequence("R"))
        self.act_view_reset.triggered.connect(lambda: self.viewer.reset_view())

        self.act_view_iso = QAction("Isometric", self)
        self.act_view_iso.triggered.connect(lambda: self.viewer.set_view("iso"))

        self.act_view_top = QAction("Top", self)
        self.act_view_top.triggered.connect(lambda: self.viewer.set_view("top"))

        self.act_view_front = QAction("Front", self)
        self.act_view_front.triggered.connect(lambda: self.viewer.set_view("front"))

        self.act_view_left = QAction("Left", self)
        self.act_view_left.triggered.connect(lambda: self.viewer.set_view("left"))

        self.act_view_right = QAction("Right", self)
        self.act_view_right.triggered.connect(lambda: self.viewer.set_view("right"))

        # Recent
        self.recent_menu = QMenu("Recent", self)

        # Export dropdown menu for toolbar/context
        self.export_menu = QMenu("Export", self)
        self.export_menu.addAction(self.act_export_shot)
        self.export_menu.addAction(self.act_export_ply)
        self.export_menu.addAction(self.act_export_split)

    # =========================================================
    # Menu bar (minimal + standard)
    # =========================================================
    def _build_menu(self) -> None:
        menubar = self.menuBar()

        m_file = menubar.addMenu("File")
        m_file.addAction(self.act_import)
        m_file.addAction(self.act_open_project)
        m_file.addSeparator()
        m_file.addAction(self.act_save)
        m_file.addAction(self.act_save_as)
        m_file.addSeparator()
        m_file.addMenu(self.recent_menu)
        m_file.addSeparator()
        m_file.addAction(self.act_exit)

        m_run = menubar.addMenu("Run")
        m_run.addAction(self.act_run)

        m_export = menubar.addMenu("Export")
        m_export.addAction(self.act_export_shot)
        m_export.addAction(self.act_export_ply)
        m_export.addAction(self.act_export_split)

        m_view = menubar.addMenu("View")
        m_view.addAction(self.act_view_reset)
        m_view.addSeparator()
        m_view.addAction(self.act_view_iso)
        m_view.addAction(self.act_view_top)
        m_view.addAction(self.act_view_front)
        m_view.addAction(self.act_view_left)
        m_view.addAction(self.act_view_right)

    # =========================================================
    # Toolbar (minimal)
    # =========================================================
    def _build_toolbar_minimal(self) -> None:
        """
        Minimal toolbar:
          [Run] [Export▼] [Reset View]
        Import/Save/SaveAs อยู่ใน File menu เท่านั้น
        """
        tb = QToolBar("Main", self)
        tb.setMovable(False)
        tb.setFloatable(False)
        tb.setToolButtonStyle(Qt.ToolButtonTextOnly)
        tb.setContextMenuPolicy(Qt.PreventContextMenu)
        tb.setStyleSheet(
            """
            QToolBar { spacing: 6px; padding: 4px; }
            QToolButton { padding: 4px 10px; border-radius: 8px; }
            QToolButton:hover { background: rgba(255,255,255,0.06); }
            """
        )
        self.addToolBar(Qt.TopToolBarArea, tb)

        tb.addAction(self.act_run)

        act_export = QAction("Export ▾", self)
        tb.addAction(act_export)
        btn = tb.widgetForAction(act_export)
        if btn is not None:
            btn.setMenu(self.export_menu)
            try:
                btn.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
            except Exception:
                btn.setPopupMode(QToolButton.InstantPopup)

        tb.addAction(self.act_view_reset)

    # =========================================================
    # Right-click context menu (viewer)
    # =========================================================
    def _show_viewer_context_menu(self, pos_global) -> None:
        menu = QMenu(self)
        menu.addAction(self.act_import)
        menu.addSeparator()
        menu.addAction(self.act_run)
        menu.addMenu(self.export_menu)
        menu.addSeparator()

        m_view = menu.addMenu("View")
        m_view.addAction(self.act_view_reset)
        m_view.addAction(self.act_view_iso)
        m_view.addAction(self.act_view_top)
        m_view.addAction(self.act_view_front)
        m_view.addAction(self.act_view_left)
        m_view.addAction(self.act_view_right)

        menu.exec(pos_global)

    # =========================================================
    # Docks
    # =========================================================
    def _build_left_dock(self) -> QDockWidget:
        dock = QDockWidget("Case", self)
        dock.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)
        dock.setFeatures(QDockWidget.NoDockWidgetFeatures)

        w = QWidget(dock)
        lay = QVBoxLayout(w)
        lay.setSpacing(10)

        self.lb_file = QLabel("ยังไม่ได้โหลดไฟล์")
        self.lb_file.setWordWrap(True)
        lay.addWidget(self.lb_file)

        self.ck_base = QCheckBox("Show Base")
        self.ck_base.setChecked(True)
        self.ck_base.toggled.connect(self.viewer.toggle_base_visible)
        lay.addWidget(self.ck_base)

        self.ck_pred = QCheckBox("Show Prediction")
        self.ck_pred.setChecked(True)
        self.ck_pred.toggled.connect(self.viewer.toggle_pred_visible)
        lay.addWidget(self.ck_pred)

        lay.addWidget(QLabel("Base Opacity"))
        s1 = QSlider(Qt.Horizontal)
        s1.setRange(0, 100)
        s1.setValue(100)
        s1.valueChanged.connect(lambda x: self.viewer.set_base_opacity(x / 100.0))
        lay.addWidget(s1)

        lay.addWidget(QLabel("Pred Opacity"))
        s2 = QSlider(Qt.Horizontal)
        s2.setRange(0, 100)
        s2.setValue(85)
        s2.valueChanged.connect(lambda x: self.viewer.set_pred_opacity(x / 100.0))
        lay.addWidget(s2)

        lay.addStretch(1)
        dock.setWidget(w)
        return dock

    def _build_right_dock(self) -> QDockWidget:
        dock = QDockWidget("Run", self)
        dock.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)
        dock.setFeatures(QDockWidget.NoDockWidgetFeatures)

        w = QWidget(dock)
        lay = QVBoxLayout(w)
        lay.setSpacing(10)

        lay.addWidget(QLabel("Model Preset"))
        self.cb = QComboBox()
        lay.addWidget(self.cb)

        self.btn_run = QPushButton("Run Segmentation")
        self.btn_run.clicked.connect(self.on_run)
        lay.addWidget(self.btn_run)

        self.pb = QProgressBar()
        self.pb.setRange(0, 100)
        self.pb.setValue(0)
        lay.addWidget(self.pb)

        lay.addWidget(QLabel("Log"))
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(2000)
        lay.addWidget(self.log, 1)

        dock.setWidget(w)
        return dock

    def _refresh_presets(self) -> None:
        self.cb.clear()
        for k in self.registry.keys():
            self.cb.addItem(k)

    def _show_side_panels(self) -> None:
        if not self.dock_left.isVisible():
            self.dock_left.setVisible(True)
        if not self.dock_right.isVisible():
            self.dock_right.setVisible(True)

    # =========================================================
    # Logging + Recent
    # =========================================================
    def _log(self, msg: str) -> None:
        self.log.appendPlainText(msg)
        self.statusBar().showMessage(msg, 4000)

    def _recent_list(self) -> List[str]:
        return self.settings.value("recent_files", [], type=list)

    def _push_recent(self, path: Path) -> None:
        p = str(path)
        items = [x for x in self._recent_list() if x != p]
        items.insert(0, p)
        items = items[:10]
        self.settings.setValue("recent_files", items)
        self._rebuild_recent_menu()

    def _rebuild_recent_menu(self) -> None:
        self.recent_menu.clear()
        items = self._recent_list()
        if not items:
            a = self.recent_menu.addAction("(empty)")
            a.setEnabled(False)
            return
        for p in items:
            act = self.recent_menu.addAction(p)
            act.triggered.connect(lambda _=False, pp=p: self._open_recent(pp))

    def _open_recent(self, p: str) -> None:
        path = Path(p)
        if not path.exists():
            QMessageBox.warning(self, "Recent", f"File not found:\n{path}")
            items = [x for x in self._recent_list() if x != p]
            self.settings.setValue("recent_files", items)
            self._rebuild_recent_menu()
            return

        if path.suffix.lower() in (".ply", ".stl", ".obj"):
            self._import_mesh_path(path)
        else:
            self._open_project_path(path)

    # =========================================================
    # Import
    # =========================================================
    def on_import_mesh(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select mesh file",
            str(self.app_root),
            "Mesh Files (*.ply *.stl *.obj);;All Files (*.*)",
        )
        if not path:
            return
        self._import_mesh_path(Path(path))

    def _import_mesh_path(self, mesh_path: Path) -> None:
        self.current_mesh = mesh_path
        self.current_project = None
        self.lb_file.setText(str(mesh_path))

        try:
            self.viewer.load_base_mesh(mesh_path)
            self._show_side_panels()
            self._log(f"Loaded: {mesh_path}")
            self._push_recent(mesh_path)
        except Exception as e:
            QMessageBox.critical(self, "Load error", str(e))

    # =========================================================
    # Run
    # =========================================================
    def on_run(self) -> None:
        if self.current_mesh is None:
            QMessageBox.warning(self, "No input", "กรุณา Import mesh ก่อน")
            return
        if self.worker is not None and self.worker.isRunning():
            QMessageBox.information(self, "Running", "กำลังรันอยู่")
            return

        preset_key = self.cb.currentText().strip()
        self._log(f"Run preset: {preset_key}")
        self.pb.setValue(0)

        self.worker = InferenceWorker(
            app_root=self.app_root,
            mesh_path=self.current_mesh,
            preset_key=preset_key,
            device="cuda",
        )
        self.worker.signals.log.connect(self._log)
        self.worker.signals.progress.connect(self.pb.setValue)
        self.worker.signals.done.connect(self._on_done)
        self.worker.signals.failed.connect(self._on_failed)
        self.worker.start()

    def _on_done(self, res) -> None:
        """
        ✅ สำคัญ: overlay ต้องทำตรงนี้
        """
        self.last_result = res
        self._log("Done")

        try:
            # แสดง mesh หลัง preprocess (remesh/orient) ถ้ามี
            self.viewer.load_base_mesh_from_arrays(res.mesh_proc.pos, res.mesh_proc.faces)

            # ทำสี overlay จาก label ต่อ face
            face_rgb = label_array_to_rgb_face(
                res.labels_face,
                arch=res.mesh_proc.arch,
                num_classes=int(res.num_classes),
            )
            self.viewer.set_pred_overlay_face_rgb(face_rgb)

            self._log("Overlay shown")

        except Exception as e:
            self._log(f"[ERROR] overlay fail: {e}")
            QMessageBox.warning(self, "Overlay error", str(e))

        self.pb.setValue(100)

    def _on_failed(self, msg: str) -> None:
        self._log(f"[ERROR] {msg}")
        QMessageBox.critical(self, "Run failed", msg)

    # =========================================================
    # Export
    # =========================================================
    def on_export_screenshot(self) -> None:
        out, _ = QFileDialog.getSaveFileName(
            self, "Save screenshot", str(Path.home() / "screenshot.png"), "PNG (*.png)"
        )
        if not out:
            return
        try:
            self.viewer.export_screenshot(out)
            self._log(f"Saved screenshot: {out}")
        except Exception as e:
            QMessageBox.critical(self, "Export error", str(e))

    def on_export_colored_ply(self) -> None:
        if self.last_result is None:
            QMessageBox.warning(self, "No result", "กรุณา Run ก่อน")
            return
        out, _ = QFileDialog.getSaveFileName(
            self, "Save colored PLY", str(Path.home() / "colored.ply"), "PLY (*.ply)"
        )
        if not out:
            return

        try:
            res = self.last_result
            face_rgb = label_array_to_rgb_face(
                res.labels_face,
                arch=res.mesh_proc.arch,
                num_classes=int(res.num_classes),
            )
            export_colored_ply_facecolors(Path(out), res.mesh_proc.pos, res.mesh_proc.faces, face_rgb)
            self._log(f"Export colored PLY: {out}")
        except Exception as e:
            QMessageBox.critical(self, "Export error", str(e))

    def on_export_split(self) -> None:
        if self.last_result is None:
            QMessageBox.warning(self, "No result", "กรุณา Run ก่อน")
            return
        out = QFileDialog.getExistingDirectory(self, "Select output directory", str(Path.home()))
        if not out:
            return
        try:
            res = self.last_result
            paths = export_split_by_tooth(
                Path(out),
                v=res.mesh_proc.pos,
                f=res.mesh_proc.faces,
                labels_face=res.labels_face,
                arch=res.mesh_proc.arch,
                num_classes=int(res.num_classes),
            )
            self._log(f"Export split: {len(paths)} files -> {out}")
        except Exception as e:
            QMessageBox.critical(self, "Export error", str(e))

    # =========================================================
    # Project Save/Load
    # =========================================================
    def on_open_project(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open project",
            str(Path.home()),
            f"Project (*.{PROJECT_EXT});;All Files (*.*)",
        )
        if not path:
            return
        self._open_project_path(Path(path))

    def _open_project_path(self, proj: Path) -> None:
        try:
            obj = json.loads(proj.read_text(encoding="utf-8"))
            mesh_path = Path(obj.get("mesh_path", ""))
            preset = str(obj.get("preset", "")).strip()

            if not mesh_path.exists():
                raise FileNotFoundError(f"Mesh not found: {mesh_path}")

            self._import_mesh_path(mesh_path)

            if preset:
                idx = self.cb.findText(preset)
                if idx >= 0:
                    self.cb.setCurrentIndex(idx)

            self.current_project = proj
            self._push_recent(proj)
            self._log(f"Opened project: {proj}")
        except Exception as e:
            QMessageBox.critical(self, "Open project failed", str(e))

    def on_save_project(self) -> None:
        if self.current_project is None:
            self.on_save_as_project()
            return
        self._save_project_to(self.current_project)

    def on_save_as_project(self) -> None:
        default_name = f"dental_seg.{PROJECT_EXT}"
        out, _ = QFileDialog.getSaveFileName(
            self,
            "Save project as",
            str(Path.home() / default_name),
            f"Project (*.{PROJECT_EXT});;All Files (*.*)",
        )
        if not out:
            return
        p = Path(out)
        if not p.name.endswith(PROJECT_EXT):
            p = p.with_name(p.name + "." + PROJECT_EXT)
        self._save_project_to(p)

    def _save_project_to(self, p: Path) -> None:
        if self.current_mesh is None:
            QMessageBox.warning(self, "Save", "ยังไม่มี mesh ให้บันทึก")
            return
        data: Dict[str, Any] = {
            "mesh_path": str(self.current_mesh),
            "preset": self.cb.currentText().strip(),
        }
        try:
            p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
            self.current_project = p
            self._push_recent(p)
            self._log(f"Saved project: {p}")
        except Exception as e:
            QMessageBox.critical(self, "Save failed", str(e))