from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pyvista as pv
from pyvistaqt import QtInteractor

from PySide6.QtCore import Signal, QPoint, QTimer
from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import QWidget, QVBoxLayout

from app.data.fdi_colors import FDIColorMap, GINGIVA_LABEL


class ViewerWidget(QWidget):
    rightClicked = Signal(QPoint)

    def __init__(self, parent=None):
        super().__init__(parent)

        self.setAutoFillBackground(True)
        pal = self.palette()
        pal.setColor(QPalette.Window, QColor("#0b1220"))
        self.setPalette(pal)
        self.setStyleSheet("background: #0b1220;")

        try:
            import vtk  # type: ignore
            vtk.vtkObject.GlobalWarningDisplayOff()
        except Exception:
            pass

        self.plotter = QtInteractor(self)

        # สำคัญ: บังคับสีพื้นของ interactor ทันที
        try:
            self.plotter.interactor.setAutoFillBackground(True)
            ipal = self.plotter.interactor.palette()
            ipal.setColor(QPalette.Window, QColor("#0b1220"))
            self.plotter.interactor.setPalette(ipal)
            self.plotter.interactor.setStyleSheet("background: #0b1220;")
        except Exception:
            pass

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.plotter.interactor)

        self._polys: Dict[str, pv.PolyData] = {}
        self._base_actors: Dict[str, object] = {}
        self._pred_actors: Dict[str, object] = {}
        self._segment_actors: Dict[str, Dict[str, object]] = {}
        self._segment_colors: Dict[str, Dict[str, Tuple[int, int, int]]] = {}
        self._segment_visibility: Dict[str, Dict[str, bool]] = {}
        self._current_selected: Tuple[str, str] | None = None

        try:
            self.plotter.set_background("#5b61d6", top="#0b1220")
        except Exception:
            pass

        try:
            self.plotter.enable_anti_aliasing()
        except Exception:
            pass

        try:
            self.plotter.add_axes(interactive=True)
        except Exception:
            pass

        try:
            self.plotter.enable_eye_dome_lighting()
        except Exception:
            pass

        # เรียกทันที 1 รอบ
        self._init_blank_frame()
        # แล้วย้ำอีกทีรอบถัดไป
        QTimer.singleShot(0, self._init_blank_frame)

    def _init_blank_frame(self) -> None:
        try:
            self.setStyleSheet("background: #0b1220;")
        except Exception:
            pass

        try:
            self.plotter.interactor.setAutoFillBackground(True)
            pal = self.plotter.interactor.palette()
            pal.setColor(QPalette.Window, QColor("#0b1220"))
            self.plotter.interactor.setPalette(pal)
            self.plotter.interactor.setStyleSheet("background: #0b1220;")
        except Exception:
            pass

        try:
            self.plotter.set_background("#5b61d6", top="#0b1220")
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

    # =========================================================
    # Scene / camera
    # =========================================================
    def clear_scene(self) -> None:
        self.plotter.clear()

        try:
            self.plotter.set_background("#5b61d6", top="#0b1220")
        except Exception:
            pass

        try:
            self.plotter.add_axes(interactive=True)
        except Exception:
            pass

        self._polys.clear()
        self._base_actors.clear()
        self._pred_actors.clear()
        self._segment_actors.clear()
        self._segment_colors.clear()
        self._segment_visibility.clear()
        self._current_selected = None

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

    # =========================================================
    # Base mesh
    # =========================================================
    def load_base_mesh(self, mesh_path: str | Path, *, key: str) -> None:
        poly = pv.read(str(Path(mesh_path)))
        poly = self._prep_poly(poly)
        self.set_base_poly(key, poly)

    def set_base_mesh_from_arrays(self, key: str, pos: np.ndarray, faces: np.ndarray) -> None:
        V = np.asarray(pos, dtype=np.float32)
        F = np.asarray(faces, dtype=np.int64)

        if V.ndim != 2 or V.shape[1] != 3:
            raise ValueError(f"pos must have shape (N,3), got {V.shape}")
        if F.ndim != 2 or F.shape[1] != 3:
            raise ValueError(f"faces must have shape (F,3), got {F.shape}")

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
            color="#e5e7eb",
            smooth_shading=True,
            lighting=True,
            show_edges=False,
            name=f"base:{key}",
        )
        self._apply_material(actor, opacity=1.0, overlay=False)
        self._base_actors[key] = actor
        self.reset_view()

    def set_base_visible(self, key: str, visible: bool) -> None:
        actor = self._base_actors.get(key)
        if actor is None:
            return
        try:
            actor.SetVisibility(bool(visible))
            self.plotter.render()
        except Exception:
            pass

    # =========================================================
    # Full overlay (legacy)
    # =========================================================
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

    # =========================================================
    # Segment mode
    # =========================================================
    def clear_segment_overlays(self, key: str) -> None:
        if key in self._segment_actors:
            for actor in self._segment_actors[key].values():
                try:
                    self.plotter.remove_actor(actor)
                except Exception:
                    pass

        self._segment_actors.pop(key, None)
        self._segment_colors.pop(key, None)
        self._segment_visibility.pop(key, None)

    def set_overlay_segments_by_label(
        self,
        key: str,
        labels_face: np.ndarray,
        *,
        arch: str,
        num_classes: int = 17,
        opacity: float = 16,
        show_gingiva: bool = False,
        show_base: bool = False,
        min_faces: int = 30,
    ) -> Dict[str, Tuple[int, int, int]]:
        if key not in self._polys:
            raise RuntimeError(f"segments: ไม่มี base poly สำหรับ key='{key}'")

        base_poly = self._polys[key]
        labels = np.asarray(labels_face, dtype=np.int64).reshape(-1)

        if int(base_poly.n_cells) != int(labels.shape[0]):
            raise ValueError(f"labels_face ({labels.shape[0]}) != faces ({base_poly.n_cells})")

        self.clear_segment_overlays(key)

        cmap = FDIColorMap(unknown_policy="ignore")
        seg_actors: Dict[str, object] = {}
        seg_colors: Dict[str, Tuple[int, int, int]] = {}
        seg_visible: Dict[str, bool] = {}

        unique_labels = [int(x) for x in sorted(np.unique(labels).tolist())]

        for lb in unique_labels:
            face_ids = np.where(labels == lb)[0]
            if face_ids.size < int(min_faces):
                continue

            if num_classes >= 17 and lb == int(GINGIVA_LABEL):
                seg_name = "GINGIVA"
                color = (200, 200, 200)
                visible_default = bool(show_gingiva)
            else:
                try:
                    fdi = cmap.label16_to_fdi(arch, lb)
                except Exception:
                    continue
                seg_name = f"FDI {int(fdi)}"
                color = tuple(int(c) for c in cmap.fdi_to_rgb(int(fdi)))
                visible_default = True

            try:
                sub = base_poly.extract_cells(face_ids)
            except Exception:
                continue

            if sub.n_cells == 0:
                continue

            sub = self._prep_poly(sub)

            rgb_arr = np.tile(np.asarray(color, dtype=np.uint8), (int(sub.n_points), 1))
            try:
                sub.point_data["rgb"] = rgb_arr
                use_scalars = "rgb"
                use_rgb = True
            except Exception:
                use_scalars = None
                use_rgb = False

            actor = self.plotter.add_mesh(
                sub,
                scalars=use_scalars,
                rgb=use_rgb,
                color=None if use_rgb else color,
                smooth_shading=True,
                lighting=True,
                show_edges=False,
                name=f"seg:{key}:{seg_name}",
            )
            self._apply_material(actor, opacity=float(opacity), overlay=True)
            try:
                actor.SetVisibility(bool(visible_default))
            except Exception:
                pass

            seg_actors[seg_name] = actor
            seg_colors[seg_name] = color
            seg_visible[seg_name] = bool(visible_default)

        self._segment_actors[key] = seg_actors
        self._segment_colors[key] = seg_colors
        self._segment_visibility[key] = seg_visible

        self.set_base_visible(key, bool(show_base))

        try:
            self.plotter.render()
        except Exception:
            pass

        return seg_colors

    def set_segment_visible(self, group_key: str, segment_key: str, visible: bool) -> None:
        actor = self._segment_actors.get(group_key, {}).get(segment_key)
        if actor is None:
            return

        try:
            actor.SetVisibility(bool(visible))
            self._segment_visibility.setdefault(group_key, {})[segment_key] = bool(visible)
            self.plotter.render()
        except Exception:
            pass

    def highlight_segment(self, group_key: str, segment_key: str, enabled: bool = True) -> None:
        if self._current_selected is not None:
            prev_group, prev_seg = self._current_selected
            prev_actor = self._segment_actors.get(prev_group, {}).get(prev_seg)
            if prev_actor is not None:
                self._apply_material(prev_actor, opacity=0.96, overlay=True)
                try:
                    prev_actor.GetProperty().SetEdgeVisibility(0)
                except Exception:
                    pass

        if not enabled:
            self._current_selected = None
            try:
                self.plotter.render()
            except Exception:
                pass
            return

        actor = self._segment_actors.get(group_key, {}).get(segment_key)
        if actor is None:
            self._current_selected = None
            return

        for seg_name, seg_actor in self._segment_actors.get(group_key, {}).items():
            if seg_actor is None:
                continue
            is_visible = self._segment_visibility.get(group_key, {}).get(seg_name, True)
            if not is_visible:
                continue
            if seg_name == segment_key:
                continue
            try:
                prop = seg_actor.GetProperty()
                prop.SetOpacity(0.18)
                prop.SetEdgeVisibility(0)
            except Exception:
                pass

        try:
            prop = actor.GetProperty()
            prop.SetOpacity(1.0)
            prop.SetAmbient(0.22)
            prop.SetDiffuse(0.95)
            prop.SetSpecular(0.35)
            prop.SetSpecularPower(48)
            prop.SetEdgeVisibility(1)
            prop.SetLineWidth(2.0)
        except Exception:
            pass

        self._current_selected = (group_key, segment_key)

        try:
            self.plotter.render()
        except Exception:
            pass

    def clear_highlight(self) -> None:
        if self._current_selected is None:
            return

        for group_key, group in self._segment_actors.items():
            for seg_name, actor in group.items():
                if actor is None:
                    continue
                try:
                    self._apply_material(actor, opacity=0.96, overlay=True)
                    actor.GetProperty().SetEdgeVisibility(0)
                except Exception:
                    pass

        self._current_selected = None
        try:
            self.plotter.render()
        except Exception:
            pass

    # =========================================================
    # Export
    # =========================================================
    def export_screenshot(self, out_path: str | Path) -> None:
        self.plotter.screenshot(str(Path(out_path)))

    # =========================================================
    # Internal
    # =========================================================
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

        try:
            prop.SetEdgeVisibility(0)
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