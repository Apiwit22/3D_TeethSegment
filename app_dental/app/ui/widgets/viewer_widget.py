from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pyvista as pv
from pyvistaqt import QtInteractor

from PySide6.QtCore import Signal, QPoint
from PySide6.QtWidgets import QWidget, QVBoxLayout


class ViewerWidget(QWidget):
    """
    Viewer 3D (PyVistaQt)
    - load mesh (file/arrays)
    - overlay face RGB
    - right-click signal
    - view controls: reset camera / preset views
    """

    rightClicked = Signal(QPoint)  # global pos for QMenu.exec()

    def __init__(self, parent=None):
        super().__init__(parent)

        self.plotter = QtInteractor(self)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.plotter.interactor)

        self._base_poly: Optional[pv.PolyData] = None
        self._base_actor = None
        self._pred_actor = None

        self._base_visible = True
        self._pred_visible = True
        self._base_opacity = 1.0
        self._pred_opacity = 0.85

        self.plotter.set_background("#5b61d6", top="#0b1220")
        self.plotter.enable_anti_aliasing()
        self.plotter.add_axes(interactive=True)

        # MeshLab-like depth shading
        try:
            self.plotter.enable_eye_dome_lighting()
        except Exception:
            pass

        self.setMouseTracking(True)

    # ----------------------------
    # Context menu hook
    # ----------------------------
    def contextMenuEvent(self, event) -> None:
        self.rightClicked.emit(event.globalPos())
        event.accept()

    # ----------------------------
    # View controls
    # ----------------------------
    def reset_view(self) -> None:
        self.plotter.reset_camera()
        self._set_camera_preset()
        self.plotter.render()

    def set_view(self, name: str) -> None:
        """
        name: iso, top, bottom, front, back, left, right
        """
        n = (name or "").lower().strip()
        if n in ("iso", "isometric"):
            try:
                self.plotter.camera_position = "iso"
            except Exception:
                self.plotter.reset_camera()
        elif n == "top":
            try:
                self.plotter.camera_position = "xy"
            except Exception:
                self.plotter.reset_camera()
        elif n == "bottom":
            try:
                self.plotter.camera_position = "yx"
            except Exception:
                self.plotter.reset_camera()
        elif n == "front":
            try:
                self.plotter.camera_position = "xz"
            except Exception:
                self.plotter.reset_camera()
        elif n == "back":
            try:
                self.plotter.camera_position = "zx"
            except Exception:
                self.plotter.reset_camera()
        elif n == "left":
            try:
                self.plotter.camera_position = "yz"
            except Exception:
                self.plotter.reset_camera()
        elif n == "right":
            try:
                self.plotter.camera_position = "zy"
            except Exception:
                self.plotter.reset_camera()
        else:
            self.plotter.reset_camera()

        self.plotter.camera.zoom(1.10)
        self.plotter.render()

    # ----------------------------
    # Load base mesh
    # ----------------------------
    def load_base_mesh(self, mesh_path: str | Path) -> None:
        poly = pv.read(str(Path(mesh_path)))
        poly = self._prep_poly(poly)
        self._set_base_poly(poly)

    def load_base_mesh_from_arrays(self, pos: np.ndarray, faces: np.ndarray) -> None:
        V = np.asarray(pos, dtype=np.float32)
        F = np.asarray(faces, dtype=np.int64)

        if V.ndim != 2 or V.shape[1] != 3:
            raise ValueError(f"pos must be (V,3), got {V.shape}")
        if F.ndim != 2 or F.shape[1] != 3:
            raise ValueError(f"faces must be (F,3), got {F.shape}")

        faces_pv = np.empty((F.shape[0], 4), dtype=np.int64)
        faces_pv[:, 0] = 3
        faces_pv[:, 1:] = F
        poly = pv.PolyData(V, faces_pv.ravel())
        poly = self._prep_poly(poly)
        self._set_base_poly(poly)

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

    def _set_camera_preset(self) -> None:
        try:
            self.plotter.camera_position = "iso"
            self.plotter.camera.zoom(1.10)
        except Exception:
            pass

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

    def _set_base_poly(self, poly: pv.PolyData) -> None:
        self._base_poly = poly

        self.plotter.clear()
        self.plotter.add_axes(interactive=True)

        self._base_actor = self.plotter.add_mesh(
            poly,
            color="#f8fafc",
            smooth_shading=True,
            lighting=True,
            show_edges=False,
            name="base",
        )
        self._apply_material(self._base_actor, opacity=self._base_opacity, overlay=False)

        self._pred_actor = None
        self.plotter.reset_camera()
        self._set_camera_preset()
        self._apply_visibility()
        self.plotter.render()

    # ----------------------------
    # Overlay
    # ----------------------------
    def set_pred_overlay_face_rgb(self, face_rgb: np.ndarray) -> None:
        if self._base_poly is None:
            return

        rgb = np.asarray(face_rgb, dtype=np.uint8)
        if rgb.ndim != 2 or rgb.shape[1] != 3:
            raise ValueError("face_rgb ต้องเป็น (F,3)")

        poly = self._base_poly.copy(deep=True)
        n_faces = int(poly.n_cells)
        if rgb.shape[0] != n_faces:
            raise ValueError(f"จำนวน face_rgb ({rgb.shape[0]}) ไม่เท่ากับ faces ({n_faces})")

        poly.cell_data["rgb"] = rgb
        poly = self._prep_poly(poly)

        if self._pred_actor is not None:
            try:
                self.plotter.remove_actor(self._pred_actor)
            except Exception:
                pass

        self._pred_actor = self.plotter.add_mesh(
            poly,
            scalars="rgb",
            rgb=True,
            smooth_shading=True,
            lighting=True,
            show_edges=False,
            name="pred",
        )
        self._apply_material(self._pred_actor, opacity=self._pred_opacity, overlay=True)

        self._apply_visibility()
        self.plotter.render()

    # ----------------------------
    # Export / Controls
    # ----------------------------
    def export_screenshot(self, out_path: str | Path) -> None:
        self.plotter.screenshot(str(Path(out_path)))

    def set_base_opacity(self, v: float) -> None:
        self._base_opacity = float(np.clip(v, 0.0, 1.0))
        if self._base_actor is not None:
            self._apply_material(self._base_actor, opacity=self._base_opacity, overlay=False)
        self.plotter.render()

    def set_pred_opacity(self, v: float) -> None:
        self._pred_opacity = float(np.clip(v, 0.0, 1.0))
        if self._pred_actor is not None:
            self._apply_material(self._pred_actor, opacity=self._pred_opacity, overlay=True)
        self.plotter.render()

    def toggle_base_visible(self, visible: bool) -> None:
        self._base_visible = bool(visible)
        self._apply_visibility()

    def toggle_pred_visible(self, visible: bool) -> None:
        self._pred_visible = bool(visible)
        self._apply_visibility()

    def _apply_visibility(self) -> None:
        if self._base_actor is not None:
            self._base_actor.SetVisibility(1 if self._base_visible else 0)
        if self._pred_actor is not None:
            self._pred_actor.SetVisibility(1 if self._pred_visible else 0)
        self.plotter.render()