from __future__ import annotations

import re
from typing import Optional

from PySide6.QtCore import QSize, Qt, QSignalBlocker
from PySide6.QtWidgets import (
    QInputDialog,
    QTabBar,
    QTabWidget,
    QToolButton,
    QWidget,
)

from app.ui.widgets.result_tabs import PerTabResultWidget, StartLikeTab
from app.ui.widgets.viewer_widget import ViewerWidget


class TabManager:
    """
    กติกา tab:
    - tab ปกติ = ทุก tab ที่ไม่ใช่ "+"
      (รวม Tab 1 ด้วย -> ปิดได้)
    - tab สุดท้าย = "+"
    """

    _RE_TAB_NUM = re.compile(r"^\s*Tab\s+(\d+)\s*$", re.IGNORECASE)

    def __init__(self, mainwin, tabs: QTabWidget, layer_panel, result_splitter):
        self.mainwin = mainwin
        self.tabs = tabs
        self.layer_panel = layer_panel
        self.result_splitter = result_splitter

        self._handling_plus = False
        self._adding_tab = False

    # =========================================================
    # Setup
    # =========================================================
    def install(self) -> None:
        self.tabs.currentChanged.connect(self.on_result_tab_changed)

        try:
            self.tabs.tabBar().tabMoved.connect(lambda _a, _b: self.rebind_close_buttons())
            self.tabs.tabBar().tabBarDoubleClicked.connect(self.on_tab_double_clicked)
        except Exception:
            pass

        self.ensure_plus_tab()
        self.rebind_close_buttons()
        self.set_layers_visible(self.has_global_result_tab())

    # =========================================================
    # Anti-flicker helpers
    # =========================================================
    def _freeze_tab_updates(self, freeze: bool) -> None:
        enabled = not freeze

        try:
            self.tabs.setUpdatesEnabled(enabled)
        except Exception:
            pass

        try:
            self.tabs.tabBar().setUpdatesEnabled(enabled)
        except Exception:
            pass

        try:
            self.result_splitter.setUpdatesEnabled(enabled)
        except Exception:
            pass

        try:
            self.mainwin.page_result.setUpdatesEnabled(enabled)
        except Exception:
            pass

    # =========================================================
    # Helpers
    # =========================================================
    def find_plus_tab_index(self) -> int:
        for i in range(self.tabs.count()):
            if self.tabs.tabText(i) == "+":
                return i
        return self.tabs.count()

    def ensure_plus_tab(self) -> None:
        plus_idx = self.find_plus_tab_index()

        if plus_idx == self.tabs.count():
            self.tabs.addTab(QWidget(), "+")
        elif plus_idx != self.tabs.count() - 1:
            w = self.tabs.widget(plus_idx)
            self.tabs.removeTab(plus_idx)
            self.tabs.addTab(w, "+")

        self.rebind_close_buttons()

    def is_plus_tab(self, index: int) -> bool:
        return 0 <= index < self.tabs.count() and self.tabs.tabText(index) == "+"

    def is_global_result_tab(self, index: int) -> bool:
        return 0 <= index < self.tabs.count() and self.tabs.widget(index) is self.mainwin.viewer_result

    def has_global_result_tab(self) -> bool:
        return self.tabs.indexOf(self.mainwin.viewer_result) >= 0

    def normal_tab_indices(self) -> list[int]:
        return [i for i in range(self.tabs.count()) if not self.is_plus_tab(i)]

    def has_any_normal_tabs(self) -> bool:
        return len(self.normal_tab_indices()) > 0

    def ensure_global_result_tab(self) -> int:
        """
        Make sure viewer_result exists as a normal tab titled 'Tab 1'.
        Return its index.
        """
        idx = self.tabs.indexOf(self.mainwin.viewer_result)
        if idx >= 0:
            if self.tabs.tabText(idx) != "Tab 1":
                self.tabs.setTabText(idx, "Tab 1")
            return idx

        plus_idx = self.find_plus_tab_index()
        insert_idx = plus_idx if plus_idx >= 0 else self.tabs.count()

        self._freeze_tab_updates(True)
        try:
            idx = self.tabs.insertTab(insert_idx, self.mainwin.viewer_result, "Tab 1")
            self.ensure_plus_tab()
            self.rebind_close_buttons()
        finally:
            self._freeze_tab_updates(False)

        return idx

    def set_layers_visible(self, visible: bool) -> None:
        try:
            self.layer_panel.setVisible(bool(visible))
            self.result_splitter.setSizes([1100, 360] if visible else [1400, 0])
        except Exception:
            pass

    def _refresh_widget_view(self, widget: QWidget | None) -> None:
        if widget is None:
            return

        try:
            if isinstance(widget, ViewerWidget):
                widget.plotter.render()
                return
        except Exception:
            pass

        try:
            viewer = getattr(widget, "viewer", None)
            if viewer is not None:
                viewer.plotter.render()
                return
        except Exception:
            pass

        try:
            vu = getattr(widget, "viewer_upper", None)
            vl = getattr(widget, "viewer_lower", None)
            if vu is not None:
                vu.plotter.render()
            if vl is not None:
                vl.plotter.render()
        except Exception:
            pass

    def _next_available_tab_number(self) -> int:
        """
        Reuse the smallest missing number among user tabs.
        Reserve Tab 1 for the global result tab.
        """
        used = set()

        for i in range(self.tabs.count()):
            if self.is_plus_tab(i):
                continue
            title = self.tabs.tabText(i).strip()
            m = self._RE_TAB_NUM.match(title)
            if not m:
                continue
            try:
                used.add(int(m.group(1)))
            except Exception:
                pass

        n = 2
        while n in used:
            n += 1
        return n

    def _go_back_to_start_page_if_no_tabs(self) -> None:
        """
        If no normal tabs remain, return to the first page of the program.
        """
        if self.has_any_normal_tabs():
            return

        try:
            self.layer_panel.clear_layers()
        except Exception:
            pass

        self.set_layers_visible(False)

        try:
            self.mainwin._last_viewer = None
        except Exception:
            pass

        try:
            self.mainwin.stack.setCurrentWidget(self.mainwin.page_dual)
        except Exception:
            pass

    # =========================================================
    # Current tab changed
    # =========================================================
    def on_result_tab_changed(self, idx: int) -> None:
        if self._handling_plus or self._adding_tab:
            return

        if self.is_plus_tab(idx):
            self._handling_plus = True
            try:
                self.add_new_tab_like_start()
            finally:
                self._handling_plus = False
            return

        self.set_layers_visible(self.is_global_result_tab(idx))

        try:
            self._refresh_widget_view(self.tabs.widget(idx))
        except Exception:
            pass

    # =========================================================
    # Close buttons
    # =========================================================
    def make_close_btn(self, tab_index: int) -> QToolButton:
        btn = QToolButton()
        btn.setText("x")
        btn.setCursor(Qt.PointingHandCursor)
        btn.setAutoRaise(True)
        btn.setFixedSize(QSize(12, 12))
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
        btn.clicked.connect(lambda _=False, b=btn: self.on_close_btn_clicked_for_button(b))
        return btn

    def rebind_close_buttons(self) -> None:
        bar = self.tabs.tabBar()
        for i in range(self.tabs.count()):
            if self.is_plus_tab(i):
                bar.setTabButton(i, QTabBar.RightSide, None)
                bar.setTabButton(i, QTabBar.LeftSide, None)
                continue

            bar.setTabButton(i, QTabBar.RightSide, self.make_close_btn(i))

    def on_close_btn_clicked_for_button(self, btn: QToolButton) -> None:
        idx = -1
        for i in range(self.tabs.count()):
            if self.tabs.tabBar().tabButton(i, QTabBar.RightSide) is btn:
                idx = i
                break

        if idx >= 0:
            self.on_close_tab(idx)

    # =========================================================
    # Rename
    # =========================================================
    def on_tab_double_clicked(self, index: int) -> None:
        if index < 0 or self.is_plus_tab(index):
            return

        old = self.tabs.tabText(index)
        new, ok = QInputDialog.getText(self.mainwin, "Rename tab", "Tab name:", text=old)
        if ok:
            new = (new or "").strip()
            if new:
                self.tabs.setTabText(index, new)
                self.rebind_close_buttons()

    # =========================================================
    # Replace / restore
    # =========================================================
    def replace_tab_widget(
        self,
        index: int,
        new_widget: QWidget,
        title: Optional[str] = None,
    ) -> None:
        if self.is_plus_tab(index):
            return

        old_title = self.tabs.tabText(index)
        if title is None:
            title = old_title

        old = self.tabs.widget(index)
        plus_idx = self.find_plus_tab_index()

        self._freeze_tab_updates(True)
        try:
            blocker = QSignalBlocker(self.tabs)
            self.tabs.removeTab(index)

            if plus_idx > index:
                plus_idx -= 1

            insert_idx = min(index, plus_idx)
            insert_idx = max(0, insert_idx)

            self.tabs.insertTab(insert_idx, new_widget, title)
            self.tabs.setCurrentIndex(insert_idx)
            del blocker

            self.ensure_plus_tab()

            try:
                if old is not None and old is not self.mainwin.viewer_result:
                    old.deleteLater()
            except Exception:
                pass

            self.set_layers_visible(self.is_global_result_tab(insert_idx))
        finally:
            self._freeze_tab_updates(False)

        self._refresh_widget_view(new_widget)
        try:
            self.tabs.update()
            self.mainwin.page_result.update()
        except Exception:
            pass

    def restore_input_tab(self, per_tab_result: PerTabResultWidget) -> None:
        idx = self.tabs.indexOf(per_tab_result)
        if idx < 0 or self.is_plus_tab(idx):
            return

        input_tab = per_tab_result.input_tab
        tab = StartLikeTab(self.mainwin)
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

        self.replace_tab_widget(idx, tab)

    # =========================================================
    # Add / close
    # =========================================================
    def add_new_tab_like_start(self) -> None:
        if self._adding_tab:
            return

        self._adding_tab = True
        try:
            self.ensure_plus_tab()

            plus_idx = self.find_plus_tab_index()
            if plus_idx == self.tabs.count():
                self.tabs.addTab(QWidget(), "+")
                plus_idx = self.find_plus_tab_index()

            tab = StartLikeTab(self.mainwin)

            try:
                tab.viewer_upper._init_blank_frame()
                tab.viewer_lower._init_blank_frame()
            except Exception:
                pass

            title = f"Tab {self._next_available_tab_number()}"

            self._freeze_tab_updates(True)
            try:
                blocker = QSignalBlocker(self.tabs)
                idx = self.tabs.insertTab(plus_idx, tab, title)
                self.tabs.setCurrentIndex(idx)
                del blocker

                self.ensure_plus_tab()
                self.set_layers_visible(False)
            finally:
                self._freeze_tab_updates(False)

            self._refresh_widget_view(tab)
            try:
                self.tabs.update()
                self.mainwin.page_result.update()
            except Exception:
                pass
        finally:
            self._adding_tab = False

    def on_close_tab(self, index: int) -> None:
        if index < 0 or index >= self.tabs.count():
            return
        if self.is_plus_tab(index):
            return

        w = self.tabs.widget(index)

        self._freeze_tab_updates(True)
        try:
            blocker = QSignalBlocker(self.tabs)
            self.tabs.removeTab(index)
            del blocker

            try:
                if isinstance(w, StartLikeTab):
                    w.viewer_upper.shutdown()
                    w.viewer_lower.shutdown()
                elif isinstance(w, PerTabResultWidget):
                    w.viewer.shutdown()
                elif hasattr(w, "shutdown") and w is not self.mainwin.viewer_result:
                    w.shutdown()
            except Exception:
                pass

            try:
                if w is not None and w is not self.mainwin.viewer_result:
                    w.deleteLater()
            except Exception:
                pass

            self.ensure_plus_tab()

            if not self.has_any_normal_tabs():
                self._go_back_to_start_page_if_no_tabs()
                return

            cur = self.tabs.currentIndex()
            if cur < 0 or self.is_plus_tab(cur):
                candidate = min(index, self.find_plus_tab_index() - 1)
                if candidate < 0:
                    normal = self.normal_tab_indices()
                    candidate = normal[0] if normal else -1
                if candidate >= 0:
                    self.tabs.setCurrentIndex(candidate)

            self.set_layers_visible(self.is_global_result_tab(self.tabs.currentIndex()))
        finally:
            self._freeze_tab_updates(False)

        try:
            self._refresh_widget_view(self.tabs.widget(self.tabs.currentIndex()))
            self.tabs.update()
            self.mainwin.page_result.update()
        except Exception:
            pass