from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import pyvista as pv
from pyvistaqt import QtInteractor

from PySide6.QtCore import Signal, QPoint, QTimer
from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import QWidget, QVBoxLayout


class ViewerWidget(QWidget):
    rightClicked = Signal(QPoint)

    def __init__(self, parent=None):
        super().__init__(parent)

        # ✅ กัน "จอขาวแวบ": ให้ Qt widget มีพื้นหลังเข้มตั้งแต่ก่อน VTK render
        self.setAutoFillBackground(True)
        pal = self.palette()
        pal.setColor(QPalette.Window, QColor("#0b1220"))
        self.setPalette(pal)
        self.setStyleSheet("background: #0b1220;")

        # ลด spam VTK ตอน shutdown (โดยปิด warning display)
        try:
            import vtk  # type: ignore
            vtk.vtkObject.GlobalWarningDisplayOff()
        except Exception:
            pass

        self.plotter = QtInteractor(self)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.plotter.interactor)

        self._polys: Dict[str, pv.PolyData] = {}
        self._base_actors: Dict[str, object] = {}
        self._pred_actors: Dict[str, object] = {}

        self.plotter.set_background("#5b61d6", top="#0b1220")
        self.plotter.enable_anti_aliasing()
        self.plotter.add_axes(interactive=True)
        try:
            self.plotter.enable_eye_dome_lighting()
        except Exception:
            pass

        QTimer.singleShot(0, self._init_blank_frame)

    def _init_blank_frame(self) -> None:
        try:
            self.plotter.interactor.setStyleSheet("background: #0b1220;")
        except Exception:
            pass
        try:
            self.clear_scene()
        except Exception:
            pass

    def contextMenuEvent(self, event) -> None:
        self.rightClicked.emit(event.globalPos())
        event.accept()

    def shutdown(self) -> None:
        try:
            self.clear_scene()
        except Exception:
            pass
        try:
            self.plotter.close()
        except Exception:
            pass
        try:
            self.plotter.interactor.close()
        except Exception:
            pass

    def clear_scene(self) -> None:
        self.plotter.clear()
        self.plotter.add_axes(interactive=True)
        self._polys.clear()
        self._base_actors.clear()
        self._pred_actors.clear()
        try:
            self.plotter.render()
        except Exception:
            pass

    def reset_view(self) -> None:
        try:
            self.plotter.reset_camera()
            self.plotter.camera_position = "iso"
            self.plotter.camera.zoom(1.10)
            self.plotter.render()
        except Exception:
            pass

    def set_view(self, name: str) -> None:
        n = (name or "").lower().strip()
        try:
            if n in ("iso", "isometric"):
                self.plotter.camera_position = "iso"
            elif n == "top":
                self.plotter.camera_position = "xy"
            elif n == "front":
                self.plotter.camera_position = "xz"
            elif n == "left":
                self.plotter.camera_position = "yz"
            elif n == "right":
                self.plotter.camera_position = "zy"
            else:
                self.plotter.camera_position = "iso"
            self.plotter.camera.zoom(1.10)
            self.plotter.render()
        except Exception:
            pass

    def load_base_mesh(self, mesh_path: str | Path, *, key: str) -> None:
        poly = pv.read(str(Path(mesh_path)))
        poly = self._prep_poly(poly)
        self.set_base_poly(key, poly)

    def set_base_mesh_from_arrays(self, key: str, pos: np.ndarray, faces: np.ndarray) -> None:
        V = np.asarray(pos, dtype=np.float32)
        F = np.asarray(faces, dtype=np.int64)

        faces_pv = np.empty((F.shape[0], 4), dtype=np.int64)
        faces_pv[:, 0] = 3
        faces_pv[:, 1:] = F

        poly = pv.PolyData(V, faces_pv.ravel())
        poly = self._prep_poly(poly)
        self.set_base_poly(key, poly)

    def set_base_poly(self, key: str, poly: pv.PolyData) -> None:
        p = self._prep_poly(poly)
        self._polys[key] = p

        if key in self._base_actors:
            try:
                self.plotter.remove_actor(self._base_actors[key])
            except Exception:
                pass
            self._base_actors.pop(key, None)

        actor = self.plotter.add_mesh(
            p,
            color="#f8fafc",
            smooth_shading=True,
            lighting=True,
            show_edges=False,
            name=f"base:{key}",
        )
        self._apply_material(actor, opacity=1.0, overlay=False)
        self._base_actors[key] = actor

        self.reset_view()

    def set_overlay_face_rgb(self, key: str, face_rgb: np.ndarray, *, opacity: float = 0.9) -> None:
        if key not in self._polys:
            raise RuntimeError(f"overlay: ไม่มี base poly สำหรับ key='{key}'")

        rgb = np.asarray(face_rgb, dtype=np.uint8)
        poly = self._polys[key].copy(deep=True)

        n_faces = int(poly.n_cells)
        if rgb.ndim != 2 or rgb.shape[1] != 3:
            raise ValueError("face_rgb ต้องเป็น (F,3)")
        if rgb.shape[0] != n_faces:
            raise ValueError(f"face_rgb ({rgb.shape[0]}) != faces ({n_faces})")

        poly.cell_data["rgb"] = rgb
        poly = self._prep_poly(poly)

        if key in self._pred_actors:
            try:
                self.plotter.remove_actor(self._pred_actors[key])
            except Exception:
                pass
            self._pred_actors.pop(key, None)

        actor = self.plotter.add_mesh(
            poly,
            scalars="rgb",
            rgb=True,
            smooth_shading=True,
            lighting=True,
            show_edges=False,
            name=f"pred:{key}",
        )
        self._apply_material(actor, opacity=float(opacity), overlay=True)
        self._pred_actors[key] = actor

        try:
            self.plotter.render()
        except Exception:
            pass

    def export_screenshot(self, out_path: str | Path) -> None:
        self.plotter.screenshot(str(Path(out_path)))

    def _prep_poly(self, poly: pv.PolyData) -> pv.PolyData:
        p = poly
        try:
            p = p.clean(tolerance=1e-6)
        except Exception:
            pass
        try:
            p = p.triangulate()
        except Exception:
            pass
        try:
            p = p.compute_normals(
                cell_normals=False,
                point_normals=True,
                auto_orient_normals=True,
                consistent_normals=True,
                split_vertices=True,
                feature_angle=60.0,
            )
        except Exception:
            pass
        return p

    def _apply_material(self, actor, *, opacity: float, overlay: bool) -> None:
        if actor is None:
            return
        prop = actor.GetProperty()
        prop.SetOpacity(float(opacity))
        try:
            prop.SetInterpolationToPhong()
        except Exception:
            pass

        if not overlay:
            prop.SetAmbient(0.05)
            prop.SetDiffuse(0.95)
            prop.SetSpecular(0.20)
            prop.SetSpecularPower(40)
        else:
            prop.SetAmbient(0.10)
            prop.SetDiffuse(0.90)
            prop.SetSpecular(0.10)
            prop.SetSpecularPower(25)