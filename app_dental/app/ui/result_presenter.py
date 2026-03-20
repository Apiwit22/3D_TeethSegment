from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
from PySide6.QtCore import QTimer

from app.core.result_alignment import (
    align_upper_to_lower_multihyp,
    canonicalize_pose,
    close_bite_by_z,
    sample_teeth_points,
)
from app.data.color_map import label_array_to_rgb_face
from app.ui.widgets.layer_panel import LayerInfo
from app.ui.widgets.result_tabs import PerTabResultWidget, StartLikeTab


class ResultPresenter:
    def __init__(self, mainwin):
        self.mainwin = mainwin

    def palette_for_arch(self, arch: str) -> Dict[str, Tuple[int, int, int]]:
        try:
            from app.data.fdi_colors import FDI_LIST_LOWER_16, FDI_LIST_UPPER_16, FDI_TO_RGB

            lst = FDI_LIST_UPPER_16 if arch == "upper" else FDI_LIST_LOWER_16
            out: Dict[str, Tuple[int, int, int]] = {}
            for fdi in lst:
                rgb = FDI_TO_RGB[int(fdi)]
                out[f"FDI {int(fdi)}"] = (int(rgb[0]), int(rgb[1]), int(rgb[2]))
            out["GINGIVA"] = (200, 200, 200)
            return out
        except Exception:
            return {"GINGIVA": (200, 200, 200)}

    def show(self) -> None:
        mw = self.mainwin
        mw.stack.setCurrentWidget(mw.page_result)
        owner = getattr(mw, "_run_owner_tab", None)

        posL = None
        posU = None

        if mw.res_lower is not None:
            res = mw.res_lower
            posL = canonicalize_pose(
                res.mesh_proc.pos,
                "lower",
                faces=res.mesh_proc.faces,
                labels_face=getattr(res, "labels_face", None),
                num_classes=int(getattr(res, "num_classes", 0) or 0),
            )

        if mw.res_upper is not None:
            res = mw.res_upper
            posU = canonicalize_pose(
                res.mesh_proc.pos,
                "upper",
                faces=res.mesh_proc.faces,
                labels_face=getattr(res, "labels_face", None),
                num_classes=int(getattr(res, "num_classes", 0) or 0),
            )

        if posL is not None and posU is not None:
            resL = mw.res_lower
            resU = mw.res_upper

            posU = align_upper_to_lower_multihyp(
                posU=posU,
                posL=posL,
                facesU=resU.mesh_proc.faces,
                facesL=resL.mesh_proc.faces,
                labelsU=getattr(resU, "labels_face", None),
                labelsL=getattr(resL, "labels_face", None),
                numcU=int(getattr(resU, "num_classes", 0) or 0),
                numcL=int(getattr(resL, "num_classes", 0) or 0),
            )

            ptsL = sample_teeth_points(
                posL,
                resL.mesh_proc.faces,
                getattr(resL, "labels_face", None),
                int(getattr(resL, "num_classes", 0) or 0),
                n_samples=3500,
            )
            ptsU = sample_teeth_points(
                posU,
                resU.mesh_proc.faces,
                getattr(resU, "labels_face", None),
                int(getattr(resU, "num_classes", 0) or 0),
                n_samples=3500,
            )

            posU = close_bite_by_z(
                posU,
                posL,
                upper_teeth_pts=ptsU,
                lower_teeth_pts=ptsL,
                target_gap=0.5,
                safety=0.15,
            )

        if isinstance(owner, StartLikeTab):
            self._show_per_tab_result(owner, posU, posL)
        else:
            self._show_global_result(posU, posL)

    def _show_per_tab_result(self, owner: StartLikeTab, posU, posL) -> None:
        mw = self.mainwin

        idx_owner = mw.tabs.indexOf(owner)
        if idx_owner >= 0:
            per_res = PerTabResultWidget(mw, input_tab=owner)
            mw.tab_manager.replace_tab_widget(idx_owner, per_res)
        else:
            per_res = PerTabResultWidget(mw, input_tab=owner)

        layers: List[LayerInfo] = []
        if mw.res_upper is not None:
            res = mw.res_upper
            case_id = getattr(res.mesh_proc, "case_id", "case")
            layers.append(
                LayerInfo(
                    name=f"{case_id}_upper",
                    mesh_path=owner.mesh_upper,
                    n_verts=int(len(res.mesh_proc.pos)),
                    n_faces=int(len(res.mesh_proc.faces)),
                    arch="upper",
                    palette=self.palette_for_arch("upper"),
                )
            )
        if mw.res_lower is not None:
            res = mw.res_lower
            case_id = getattr(res.mesh_proc, "case_id", "case")
            layers.append(
                LayerInfo(
                    name=f"{case_id}_lower",
                    mesh_path=owner.mesh_lower,
                    n_verts=int(len(res.mesh_proc.pos)),
                    n_faces=int(len(res.mesh_proc.faces)),
                    arch="lower",
                    palette=self.palette_for_arch("lower"),
                )
            )
        per_res.layer_panel.set_layers(layers)

        def _render_later():
            viewer = per_res.viewer
            viewer.clear_scene()

            if mw.res_lower is not None and posL is not None:
                resL = mw.res_lower
                viewer.set_base_mesh_from_arrays("lower", posL, resL.mesh_proc.faces)
                rgbL = label_array_to_rgb_face(
                    resL.labels_face,
                    arch=resL.mesh_proc.arch,
                    num_classes=int(resL.num_classes),
                )
                viewer.set_overlay_face_rgb("lower", rgbL, opacity=0.90)

            if mw.res_upper is not None and posU is not None:
                resU = mw.res_upper
                viewer.set_base_mesh_from_arrays("upper", posU, resU.mesh_proc.faces)
                rgbU = label_array_to_rgb_face(
                    resU.labels_face,
                    arch=resU.mesh_proc.arch,
                    num_classes=int(resU.num_classes),
                )
                viewer.set_overlay_face_rgb("upper", rgbU, opacity=0.90)

            viewer.reset_view()
            viewer.update()
            try:
                viewer.plotter.render()
            except Exception:
                pass

            cam_center = self._compute_center(posU, posL)
            mw._set_result_camera_side(center=cam_center)

        QTimer.singleShot(0, _render_later)
        QTimer.singleShot(50, _render_later)

    def _show_global_result(self, posU, posL) -> None:
        mw = self.mainwin

        # ensure global result tab exists even if user closed Tab 1 before
        idx_result = mw.tab_manager.ensure_global_result_tab()
        mw.tabs.setCurrentIndex(idx_result)
        mw.tab_manager.set_layers_visible(True)

        viewer = mw.viewer_result

        layers: List[LayerInfo] = []

        if mw.res_lower is not None and posL is not None:
            res = mw.res_lower
            case_id = getattr(res.mesh_proc, "case_id", "case")
            layers.append(
                LayerInfo(
                    name=f"{case_id}_lower",
                    mesh_path=mw.mesh_lower,
                    n_verts=int(len(res.mesh_proc.pos)),
                        n_faces=int(len(res.mesh_proc.faces)),
                    arch="lower",
                    palette=self.palette_for_arch("lower"),
                )
            )

        if mw.res_upper is not None and posU is not None:
            res = mw.res_upper
            case_id = getattr(res.mesh_proc, "case_id", "case")
            layers.insert(
                0,
                LayerInfo(
                    name=f"{case_id}_upper",
                    mesh_path=mw.mesh_upper,
                    n_verts=int(len(res.mesh_proc.pos)),
                    n_faces=int(len(res.mesh_proc.faces)),
                    arch="upper",
                    palette=self.palette_for_arch("upper"),
                )
            )

        mw.layer_panel.set_layers(layers)

        def _render_later():
            idx_result2 = mw.tab_manager.ensure_global_result_tab()
            mw.tabs.setCurrentIndex(idx_result2)
            viewer.clear_scene()

            if mw.res_lower is not None and posL is not None:
                res = mw.res_lower
                viewer.set_base_mesh_from_arrays("lower", posL, res.mesh_proc.faces)
                face_rgb = label_array_to_rgb_face(
                    res.labels_face,
                    arch=res.mesh_proc.arch,
                    num_classes=int(res.num_classes),
                )
                viewer.set_overlay_face_rgb("lower", face_rgb, opacity=0.90)

            if mw.res_upper is not None and posU is not None:
                res = mw.res_upper
                viewer.set_base_mesh_from_arrays("upper", posU, res.mesh_proc.faces)
                face_rgb = label_array_to_rgb_face(
                    res.labels_face,
                    arch=res.mesh_proc.arch,
                    num_classes=int(res.num_classes),
                )
                viewer.set_overlay_face_rgb("upper", face_rgb, opacity=0.90)

            viewer.reset_view()
            viewer.update()
            try:
                viewer.plotter.render()
            except Exception:
                pass

            cam_center = self._compute_center(posU, posL)
            mw._set_result_camera_side(center=cam_center)

        QTimer.singleShot(0, _render_later)
        QTimer.singleShot(50, _render_later)

    def _compute_center(self, posU, posL):
        try:
            centers = []
            if posL is not None:
                centers.append(np.mean(posL, axis=0))
            if posU is not None:
                centers.append(np.mean(posU, axis=0))
            if centers:
                return np.mean(np.stack(centers), axis=0)
        except Exception:
            pass
        return None