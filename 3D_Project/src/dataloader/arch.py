# src/dataloader/arch.py
from __future__ import annotations

import re
from pathlib import Path
from typing import Union, Iterable


# allowed tokens in a filename chunk (case-insensitive)
_UPPER_TOKENS = {
    "upper", "up", "u",
    "upperjaw", "upper_jaw", "upper-jaw",
    "maxilla", "maxillary",
}
_LOWER_TOKENS = {
    "lower", "low", "l",
    "lowerjaw", "lower_jaw", "lower-jaw",
    "mandible", "mandibular",
}


def _iter_name_hints(p: Path, *, include_parents: bool = True, parent_levels: int = 2) -> Iterable[str]:
    """
    Yield strings that may contain arch hints:
      - filename stem
      - optionally parent folder names (up to parent_levels)
    """
    yield p.stem
    if include_parents:
        cur = p.parent
        for _ in range(max(0, int(parent_levels))):
            if cur is None:
                break
            yield cur.name
            cur = cur.parent


def _tokenize(s: str) -> list[str]:
    """
    Tokenize by separators. Keep alnum chunks.
    Works for:
      - 007_U, 007-LowerJaw, case_UpperJaw, etc.
    Note: "UpperJaw" stays as one token -> we handle it via normalized token match.
    """
    # split by any non-alphanumeric character
    toks = re.split(r"[^A-Za-z0-9]+", s)
    return [t for t in toks if t]


def _normalize_token(t: str) -> str:
    """
    Lowercase, remove non-alnum for robust matching like:
      "UpperJaw" -> "upperjaw"
      "upper_jaw" -> "upperjaw"
    """
    t = t.lower()
    t = re.sub(r"[^a-z0-9]+", "", t)
    return t


def parse_arch_from_filename(
    path: Union[str, Path],
    *,
    include_parents: bool = True,
    parent_levels: int = 2,
) -> str:
    """
    Infer arch from filename and (optionally) parent folder names.

    Supported examples:
      - filename_upper.ply / filename_lower.ply
      - filename_up.ply    / filename_low.ply
      - filename_u.ply     / filename_l.ply
      - 007_U.ply / 007_L.ply
      - 007U.ply  / 007L.ply  (trailing single letter, digit before)
      - 62_UpperJaw.ply / 62_LowerJaw.ply
      - folder hints: .../62_UpperJaw/upper.ply

    Returns: "upper" or "lower"
    Raises: ValueError if cannot infer reliably.
    """
    p = Path(path)

    # 1) token-based parsing from filename stem + optional parents
    for hint in _iter_name_hints(p, include_parents=include_parents, parent_levels=parent_levels):
        tokens = _tokenize(hint)
        for t in tokens:
            tl = t.lower()
            if tl in _UPPER_TOKENS:
                return "upper"
            if tl in _LOWER_TOKENS:
                return "lower"

            # handle glued tokens / CamelCase (UpperJaw -> upperjaw)
            tn = _normalize_token(t)
            if tn in _UPPER_TOKENS:
                return "upper"
            if tn in _LOWER_TOKENS:
                return "lower"

        # also allow a strict substring check on the normalized whole hint
        # (safe because we're looking for "upperjaw"/"lowerjaw" specifically)
        hn = _normalize_token(hint)
        if "upperjaw" in hn or hn.endswith("upper") or hn.endswith("maxilla"):
            return "upper"
        if "lowerjaw" in hn or hn.endswith("lower") or hn.endswith("mandible"):
            return "lower"

    # 2) trailing single-letter fallback on filename stem only (strict anti-false-positive)
    stem = p.stem
    if len(stem) >= 1:
        tail = stem[-1].lower()
        if tail in ("u", "l"):
            if len(stem) == 1:
                return "upper" if tail == "u" else "lower"

            prev = stem[-2]
            # accept only if the trailing u/l is preceded by a NON-letter
            # e.g., "007U" ok, "case12L" ok, "menu" not ok
            if not prev.isalpha():
                return "upper" if tail == "u" else "lower"

    raise ValueError(f"Cannot infer arch from filename/folders: {path}")
