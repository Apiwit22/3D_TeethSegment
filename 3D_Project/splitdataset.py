from __future__ import annotations

import json
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import random


# ============================================================
# CONFIG (EDIT PATHS HERE)
# ============================================================

SRC_A_ROOT = Path(r"D:\Project_Gujabaa\3D_Project\aligned_public_data_recolor")
SRC_A_SUBDIRS = ["train", "test"]

SRC_B_ROOT = Path(r"D:\Project_Gujabaa\3D_Project\aligned_colored_finaldataset")

SRC_C_ROOT = Path(r"D:\Project_Gujabaa\3D_Project\aligned_data_part_colored_face")
SRC_C_PARTS = ["data_part_1", "data_part_2", "data_part_3"]

OUT_ROOT = Path(r"D:\Project_Gujabaa\3D_Project\FinalDataset_Split")

K_FOLDS = 5
SEED = 1234

WRITE_MODE = "hardlink"  # "hardlink" | "copy"
CLEAN_OUT = False


# ============================================================
# Parsing / normalization rules
# ============================================================
ARCH_PAT = re.compile(r"(?i)(?:^|[_\-\s])(?P<arch>u|l|upper|lower)(?:$|[_\-\s])")
CASEID_PAT = re.compile(r"(\d+)")


def normalize_case_id(raw: str) -> str:
    try:
        n = int(raw)
    except Exception:
        return raw
    return f"{n:02d}" if n < 100 else str(n)


def infer_case_id_from_filename(p: Path) -> Optional[str]:
    stem = p.stem
    token = re.split(r"[_\-\s]+", stem)[0]
    if token.isdigit():
        return token
    m = CASEID_PAT.search(stem)
    if m:
        return m.group(1)
    return None


def infer_arch_from_filename(p: Path) -> Optional[str]:
    s = p.stem.lower()
    if re.search(r"(?i)(?:^|[_\-\s])u(?:$|[_\-\s])", s) or "upper" in s:
        return "upper"
    if re.search(r"(?i)(?:^|[_\-\s])l(?:$|[_\-\s])", s) or "lower" in s:
        return "lower"
    m = ARCH_PAT.search(s)
    if not m:
        return None
    a = m.group("arch").lower()
    if a in ("u", "upper"):
        return "upper"
    if a in ("l", "lower"):
        return "lower"
    return None


def normalize_case_folder_name(name: str) -> str:
    s = name.strip()
    s = re.sub(r"(?i)([_\- ]+(upper|lower|u|l))$", "", s)
    return s


def score_candidate(p: Path) -> Tuple[int, int]:
    name = p.name.lower()
    bad_tokens = ["_color", "_white", "_tmp", "_vis", "_preview", "_debug"]
    penalty = sum(1 for t in bad_tokens if t in name)
    try:
        size = p.stat().st_size
    except Exception:
        size = 0
    return (penalty, -size)


def sort_case_key(raw_id: str):
    norm = normalize_case_id(raw_id)
    try:
        return (0, int(norm))
    except Exception:
        return (1, str(norm).lower())


# ============================================================
# Data model
# ============================================================
@dataclass
class CaseItem:
    source: str
    raw_case_id: str
    upper_path: Optional[Path]
    lower_path: Optional[Path]
    base_id: str
    final_id: str
    fold: int  # 1..K


# ============================================================
# Filesystem helpers
# ============================================================
def safe_mkdir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def clean_output(out_root: Path):
    if out_root.exists():
        shutil.rmtree(out_root)


def safe_copy_or_link(src: Path, dst: Path, mode: str):
    safe_mkdir(dst.parent)
    if dst.exists():
        return
    if mode == "hardlink":
        try:
            os.link(str(src), str(dst))
            return
        except Exception:
            shutil.copy2(str(src), str(dst))
            return
    shutil.copy2(str(src), str(dst))


# ============================================================
# Collect cases - Source A
# ============================================================
def collect_source_A(root: Path, subdirs: List[str], source_tag: str) -> Dict[str, Dict[str, List[Path]]]:
    cases: Dict[str, Dict[str, List[Path]]] = {}
    for sd in subdirs:
        d = root / sd
        if not d.exists():
            continue
        for p in d.rglob("*.ply"):
            cid = infer_case_id_from_filename(p)
            if cid is None:
                continue
            arch = infer_arch_from_filename(p)
            if arch is None:
                continue
            cases.setdefault(cid, {}).setdefault(arch, []).append(p)
    return cases


# ============================================================
# Collect cases - Source B
# ============================================================
def collect_source_B(root: Path, source_tag: str) -> Dict[str, Dict[str, List[Path]]]:
    cases: Dict[str, Dict[str, List[Path]]] = {}
    if not root.exists():
        return cases

    for case_dir in sorted([p for p in root.iterdir() if p.is_dir()], key=lambda p: sort_case_key(p.name)):
        raw_id = case_dir.name
        found_any = False
        for p in case_dir.rglob("*.ply"):
            cid = infer_case_id_from_filename(p) or raw_id
            arch = infer_arch_from_filename(p)
            if arch is None:
                continue
            cases.setdefault(cid, {}).setdefault(arch, []).append(p)
            found_any = True
        if not found_any:
            continue
    return cases


# ============================================================
# Collect cases - Source C (data_part_1/2/3)
# ============================================================
def collect_source_C_part(part_root: Path) -> Dict[str, Dict[str, List[Path]]]:
    cases: Dict[str, Dict[str, List[Path]]] = {}
    upper_dir = part_root / "upper"
    lower_dir = part_root / "lower"
    if not upper_dir.exists() and not lower_dir.exists():
        return cases

    if upper_dir.exists():
        for case_dir in sorted([p for p in upper_dir.iterdir() if p.is_dir()], key=lambda p: sort_case_key(p.name)):
            cid = normalize_case_folder_name(case_dir.name)
            for p in case_dir.rglob("*.ply"):
                cases.setdefault(cid, {}).setdefault("upper", []).append(p)

    if lower_dir.exists():
        for case_dir in sorted([p for p in lower_dir.iterdir() if p.is_dir()], key=lambda p: sort_case_key(p.name)):
            cid = normalize_case_folder_name(case_dir.name)
            for p in case_dir.rglob("*.ply"):
                cases.setdefault(cid, {}).setdefault("lower", []).append(p)

    return cases


# ============================================================
# Case selection
# Keep:
# - complete cases (upper + lower)
# - upper-only
# - lower-only
# ============================================================
def choose_best_cases(cases: Dict[str, Dict[str, List[Path]]]) -> List[Tuple[str, Optional[Path], Optional[Path], str]]:
    """
    Return:
      [(raw_id, upper_path_or_None, lower_path_or_None, status), ...]
    status in {"complete", "upper_only", "lower_only"}
    """
    out: List[Tuple[str, Optional[Path], Optional[Path], str]] = []

    for raw_id, g in sorted(cases.items(), key=lambda x: sort_case_key(x[0])):
        uppers = g.get("upper", [])
        lowers = g.get("lower", [])

        upper = sorted(uppers, key=score_candidate)[0] if uppers else None
        lower = sorted(lowers, key=score_candidate)[0] if lowers else None

        if upper is not None and lower is not None:
            out.append((raw_id, upper, lower, "complete"))
        elif upper is not None:
            out.append((raw_id, upper, None, "upper_only"))
        elif lower is not None:
            out.append((raw_id, None, lower, "lower_only"))

    return out


# ============================================================
# Fold assignment (case-level)
# ============================================================
def assign_folds(items: List[Tuple[str, str, Optional[Path], Optional[Path]]], k: int, seed: int) -> List[CaseItem]:
    rng = random.Random(seed)
    by_src: Dict[str, List[Tuple[str, str, Optional[Path], Optional[Path]]]] = {}
    for src, cid, up, lo in items:
        by_src.setdefault(src, []).append((src, cid, up, lo))

    for src in by_src:
        rng.shuffle(by_src[src])

    merged: List[Tuple[str, str, Optional[Path], Optional[Path]]] = []
    buckets = [by_src[s] for s in sorted(by_src.keys())]
    maxlen = max((len(b) for b in buckets), default=0)
    for i in range(maxlen):
        for b in buckets:
            if i < len(b):
                merged.append(b[i])

    assigned: List[CaseItem] = []
    for idx, (src, raw_id, up, lo) in enumerate(merged):
        fold = (idx % k) + 1
        base = normalize_case_id(raw_id)
        assigned.append(CaseItem(
            source=src,
            raw_case_id=raw_id,
            upper_path=up,
            lower_path=lo,
            base_id=base,
            final_id=base,
            fold=fold
        ))
    return assigned


def disambiguate_duplicates(cases: List[CaseItem]) -> List[CaseItem]:
    count: Dict[str, int] = {}
    for c in cases:
        count[c.base_id] = count.get(c.base_id, 0) + 1
    for c in cases:
        c.final_id = f"{c.source}_{c.base_id}" if count.get(c.base_id, 0) > 1 else c.base_id
    return cases


def sanity_each_case_one_fold(cases: List[CaseItem]):
    seen: Dict[str, int] = {}
    for c in cases:
        if c.final_id in seen and seen[c.final_id] != c.fold:
            raise RuntimeError(f"Sanity failed: case {c.final_id} assigned to multiple folds: {seen[c.final_id]} and {c.fold}")
        seen[c.final_id] = c.fold
    print("[OK] sanity: each case assigned to exactly one fold")


# ============================================================
# Export
# ============================================================
def export_all(cases: List[CaseItem], out_root: Path, k: int, mode: str):
    fold_map: Dict[int, List[CaseItem]] = {i: [] for i in range(1, k + 1)}
    for c in cases:
        fold_map[c.fold].append(c)

    for ds in range(1, k + 1):
        ds_dir = out_root / f"dataset{ds}"
        for f in range(1, k + 1):
            safe_mkdir(ds_dir / f"fold{f}")

        for f in range(1, k + 1):
            for c in fold_map[f]:
                fold_dir = ds_dir / f"fold{f}"

                if c.upper_path is not None:
                    out_upper = fold_dir / f"{c.final_id}_upper.ply"
                    safe_copy_or_link(c.upper_path, out_upper, mode)

                if c.lower_path is not None:
                    out_lower = fold_dir / f"{c.final_id}_lower.ply"
                    safe_copy_or_link(c.lower_path, out_lower, mode)

        manifest = {
            "k_folds": k,
            "seed": SEED,
            "test_fold": ds,
            "note": "datasetN defines which fold is test; folds content is identical across dataset1..datasetK by design.",
            "folds": {f"fold{i}": [c.final_id for c in fold_map[i]] for i in range(1, k + 1)},
            "cases": [
                {
                    "final_id": c.final_id,
                    "base_id": c.base_id,
                    "raw_case_id": c.raw_case_id,
                    "source": c.source,
                    "fold": c.fold,
                    "has_upper": c.upper_path is not None,
                    "has_lower": c.lower_path is not None,
                    "upper_src": str(c.upper_path) if c.upper_path else None,
                    "lower_src": str(c.lower_path) if c.lower_path else None,
                    "upper_out": str(ds_dir / f"fold{c.fold}" / f"{c.final_id}_upper.ply") if c.upper_path else None,
                    "lower_out": str(ds_dir / f"fold{c.fold}" / f"{c.final_id}_lower.ply") if c.lower_path else None,
                }
                for c in cases
            ],
        }
        (ds_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(f"[OK] wrote manifest: {ds_dir / 'manifest.json'}")

    global_manifest = {
        "k_folds": k,
        "seed": SEED,
        "write_mode": mode,
        "total_cases": len(cases),
        "complete_cases": sum(1 for c in cases if c.upper_path is not None and c.lower_path is not None),
        "upper_only_cases": sum(1 for c in cases if c.upper_path is not None and c.lower_path is None),
        "lower_only_cases": sum(1 for c in cases if c.upper_path is None and c.lower_path is not None),
        "fold_counts": {f"fold{i}": len(fold_map[i]) for i in range(1, k + 1)},
        "duplicates_policy": "if base_id duplicates across sources -> final_id = source_baseid else final_id=base_id",
        "cases": [
            {
                "final_id": c.final_id,
                "base_id": c.base_id,
                "raw_case_id": c.raw_case_id,
                "source": c.source,
                "fold": c.fold,
                "has_upper": c.upper_path is not None,
                "has_lower": c.lower_path is not None,
                "upper_src": str(c.upper_path) if c.upper_path else None,
                "lower_src": str(c.lower_path) if c.lower_path else None,
            }
            for c in cases
        ],
    }
    (out_root / "manifest.json").write_text(json.dumps(global_manifest, indent=2), encoding="utf-8")
    print(f"[OK] wrote global manifest: {out_root / 'manifest.json'}")


# ============================================================
# Main
# ============================================================
def main():
    if CLEAN_OUT:
        print(f"[WARN] CLEAN_OUT=True -> deleting: {OUT_ROOT}")
        clean_output(OUT_ROOT)

    safe_mkdir(OUT_ROOT)

    merged_items: List[Tuple[str, str, Optional[Path], Optional[Path]]] = []

    # --- Source A
    if not SRC_A_ROOT.exists():
        raise FileNotFoundError(f"SRC_A_ROOT not found: {SRC_A_ROOT}")
    casesA_raw = collect_source_A(SRC_A_ROOT, SRC_A_SUBDIRS, source_tag="setA")
    casesA = choose_best_cases(casesA_raw)
    a_complete = sum(1 for _, up, lo, _ in casesA if up is not None and lo is not None)
    a_upper_only = sum(1 for _, up, lo, _ in casesA if up is not None and lo is None)
    a_lower_only = sum(1 for _, up, lo, _ in casesA if up is None and lo is not None)
    print(f"[INFO] Source A cases: total={len(casesA)} complete={a_complete} upper_only={a_upper_only} lower_only={a_lower_only}")
    for raw_id, up, lo, _ in casesA:
        merged_items.append(("setA", raw_id, up, lo))

    # --- Source B
    if not SRC_B_ROOT.exists():
        raise FileNotFoundError(f"SRC_B_ROOT not found: {SRC_B_ROOT}")
    casesB_raw = collect_source_B(SRC_B_ROOT, source_tag="setB")
    casesB = choose_best_cases(casesB_raw)
    b_complete = sum(1 for _, up, lo, _ in casesB if up is not None and lo is not None)
    b_upper_only = sum(1 for _, up, lo, _ in casesB if up is not None and lo is None)
    b_lower_only = sum(1 for _, up, lo, _ in casesB if up is None and lo is not None)
    print(f"[INFO] Source B cases: total={len(casesB)} complete={b_complete} upper_only={b_upper_only} lower_only={b_lower_only}")
    for raw_id, up, lo, _ in casesB:
        merged_items.append(("setB", raw_id, up, lo))

    # --- Source C
    if SRC_C_ROOT.exists():
        total_c_complete = 0
        total_c_upper_only = 0
        total_c_lower_only = 0

        for part in SRC_C_PARTS:
            part_root = SRC_C_ROOT / part
            casesC_raw = collect_source_C_part(part_root)
            casesC = choose_best_cases(casesC_raw)

            c_complete = sum(1 for _, up, lo, _ in casesC if up is not None and lo is not None)
            c_upper_only = sum(1 for _, up, lo, _ in casesC if up is not None and lo is None)
            c_lower_only = sum(1 for _, up, lo, _ in casesC if up is None and lo is not None)

            total_c_complete += c_complete
            total_c_upper_only += c_upper_only
            total_c_lower_only += c_lower_only

            print(f"[INFO] Source C ({part}) cases: total={len(casesC)} complete={c_complete} upper_only={c_upper_only} lower_only={c_lower_only}")

            src_tag = f"setC_{part}"
            for raw_id, up, lo, _ in casesC:
                merged_items.append((src_tag, raw_id, up, lo))

        print(f"[INFO] Source C total: complete={total_c_complete} upper_only={total_c_upper_only} lower_only={total_c_lower_only}")
    else:
        print(f"[WARN] SRC_C_ROOT not found: {SRC_C_ROOT}")

    if not merged_items:
        raise RuntimeError("No valid cases found.")

    # --- Assign folds
    cases = assign_folds(merged_items, k=K_FOLDS, seed=SEED)
    cases = disambiguate_duplicates(cases)

    # --- Sanity
    sanity_each_case_one_fold(cases)

    # --- Summary
    src_count: Dict[str, int] = {}
    for c in cases:
        src_count[c.source] = src_count.get(c.source, 0) + 1

    complete_cases = sum(1 for c in cases if c.upper_path is not None and c.lower_path is not None)
    upper_only_cases = sum(1 for c in cases if c.upper_path is not None and c.lower_path is None)
    lower_only_cases = sum(1 for c in cases if c.upper_path is None and c.lower_path is not None)

    print("============================================================")
    print(f"[SUMMARY] total cases: {len(cases)}  |  K={K_FOLDS}  seed={SEED}")
    print(f"[SUMMARY] write_mode: {WRITE_MODE}")
    print(f"[SUMMARY] complete={complete_cases} upper_only={upper_only_cases} lower_only={lower_only_cases}")
    print("[SUMMARY] cases per source:")
    for k in sorted(src_count.keys()):
        print(f"  - {k}: {src_count[k]}")
    print("============================================================")

    # --- Export
    export_all(cases, OUT_ROOT, k=K_FOLDS, mode=WRITE_MODE)

    print("============================================================")
    print(f"[DONE] output saved to: {OUT_ROOT}")
    print("============================================================")


if __name__ == "__main__":
    main()