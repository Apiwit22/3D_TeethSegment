from __future__ import annotations

from pathlib import Path
import numpy as np
from plyfile import PlyData, PlyElement

from app.data.color_map import label_array_to_rgb_face, ensure_arch

# ✅ ใช้ของแอพเอง
from app.data.fdi_colors import LABEL16_TO_FDI_UPPER, LABEL16_TO_FDI_LOWER, GINGIVA_LABEL  # :contentReference[oaicite:28]{index=28}

# ----- โค้ดส่วนที่เหลือ ใช้ของเดิมเธอได้เลย -----
# (ตั้งแต่ _get_pos_faces() ลงไป) :contentReference[oaicite:29]{index=29}