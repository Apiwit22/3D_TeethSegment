from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QApplication

from app.ui.main_window import MainWindow


def _app_root() -> Path:
    """
    หา root ของแอพ (ไว้ resolve registry.yaml / assets / models)

    - Dev: app_root = โฟลเดอร์โปรเจค (app_dental/)
    - PyInstaller: app_root = sys._MEIPASS (resource ที่ถูก extract)
    """
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        return Path(sys._MEIPASS)  # type: ignore[attr-defined]
    return Path(__file__).resolve().parents[1]


def _icon_path() -> Path:
    """
    ตำแหน่งไอคอนที่ใช้กับ QApplication/MainWindow
    """
    return _app_root() / "assets" / "icons" / "app.ico"


def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv

    app = QApplication(argv)
    app.setApplicationName("Dental Segmentation App")

    # ✅ ตั้งไอคอน (ทั้ง taskbar + titlebar บางส่วน)
    ico = _icon_path()
    if ico.exists():
        app.setWindowIcon(QIcon(str(ico)))
    else:
        # ไม่ให้พังถ้าไอคอนไม่อยู่
        print(f"[WARN] icon not found: {ico}")

    win = MainWindow(app_root=_app_root())

    # ✅ ตั้งไอคอนของหน้าต่างหลักด้วย (ให้ชัวร์)
    if ico.exists():
        win.setWindowIcon(QIcon(str(ico)))

    win.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())