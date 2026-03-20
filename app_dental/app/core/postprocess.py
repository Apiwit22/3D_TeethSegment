from __future__ import annotations

from typing import Any, Dict

import numpy as np

from app.core.registry import Preset
from app.data.postprocess import (
    postprocess_labels_best_visual,
    postprocess_labels_conservative,
)


# ============================================================
# Basic graph helpers
# ============================================================

def _build_face_neighbors_by_edge(faces: np.ndarray) -> list[list[int]]:
    """
    Face adjacency by shared EDGE.
    Stricter than shared-vertex adjacency -> reduces bleeding.
    """
    faces = np.asarray(faces, dtype=np.int64)
    F = int(faces.shape[0])

    neigh: list[list[int]] = [[] for _ in range(F)]
    if F == 0:
        return neigh

    edge_map: dict[tuple[int, int], list[int]] = {}

    for fi, (a, b, c) in enumerate(faces):
        e0 = tuple(sorted((int(a), int(b))))
        e1 = tuple(sorted((int(b), int(c))))
        e2 = tuple(sorted((int(c), int(a))))
        edge_map.setdefault(e0, []).append(fi)
        edge_map.setdefault(e1, []).append(fi)
        edge_map.setdefault(e2, []).append(fi)

    for inc in edge_map.values():
        if len(inc) < 2:
            continue
        for i in inc:
            for j in inc:
                if i != j:
                    neigh[i].append(j)

    for i in range(F):
        if neigh[i]:
            neigh[i] = list(sorted(set(neigh[i])))

    return neigh


def _iter_connected_region_same_label(
    start: int,
    labels: np.ndarray,
    neigh: list[list[int]],
    *,
    allowed_mask: np.ndarray | None = None,
    max_region_size: int | None = None,
) -> tuple[list[int], bool]:
    """
    BFS region on same label, optionally restricted by allowed_mask.
    Returns:
      region, hit_limit
    """
    start_label = int(labels[start])
    q = [start]
    region = [start]

    visited_local = {start}
    hit_limit = False

    qi = 0
    while qi < len(q):
        u = q[qi]
        qi += 1

        for v in neigh[u]:
            if v in visited_local:
                continue
            if int(labels[v]) != start_label:
                continue
            if allowed_mask is not None and not bool(allowed_mask[v]):
                continue

            visited_local.add(v)
            q.append(v)
            region.append(v)

            if max_region_size is not None and len(region) > int(max_region_size):
                hit_limit = True
                break

        if hit_limit:
            break

    return region, hit_limit


def _face_boundary_labels(
    region: list[int],
    labels: np.ndarray,
    neigh: list[list[int]],
) -> np.ndarray:
    region_set = set(region)
    b = []
    for u in region:
        for v in neigh[u]:
            if v in region_set:
                continue
            b.append(int(labels[v]))
    if not b:
        return np.empty((0,), dtype=np.int64)
    return np.asarray(b, dtype=np.int64)


# ============================================================
# Existing cleanup 1: close small gingiva gaps
# ============================================================

def close_small_boundary_gaps(
    labels: np.ndarray,
    faces: np.ndarray,
    *,
    num_classes: int,
    max_iters: int = 2,
    min_support: int = 3,
    dominance: float = 0.60,
    max_region_size: int = 12,
) -> np.ndarray:
    """
    Fill small gingiva wedges/notches enclosed by one dominant tooth label.

    Only converts:
      gingiva -> tooth
    """
    labels = np.asarray(labels, dtype=np.int64).copy()
    faces = np.asarray(faces, dtype=np.int64)

    if labels.ndim != 1:
        labels = labels.reshape(-1)
    if faces.ndim != 2 or faces.shape[1] != 3:
        return labels

    F = int(faces.shape[0])
    if F == 0 or len(labels) != F:
        return labels

    gingiva_label = 16 if int(num_classes) >= 17 else max(int(num_classes) - 1, 0)
    neigh = _build_face_neighbors_by_edge(faces)

    max_iters = max(int(max_iters), 0)
    min_support = max(int(min_support), 1)
    dominance = float(dominance)
    max_region_size = max(int(max_region_size), 1)

    for _ in range(max_iters):
        changed = 0
        visited = np.zeros((F,), dtype=np.uint8)
        new_labels = labels.copy()

        for start in range(F):
            if visited[start]:
                continue

            if int(labels[start]) != gingiva_label:
                visited[start] = 1
                continue

            q = [start]
            visited[start] = 1
            region = [start]
            hit_limit = False

            qi = 0
            while qi < len(q):
                u = q[qi]
                qi += 1
                for v in neigh[u]:
                    if visited[v]:
                        continue
                    if int(labels[v]) != gingiva_label:
                        continue
                    visited[v] = 1
                    q.append(v)
                    region.append(v)
                    if len(region) > max_region_size:
                        hit_limit = True
                        break
                if hit_limit:
                    break

            if hit_limit or len(region) > max_region_size:
                continue

            boundary_labels = []
            region_set = set(region)

            for u in region:
                for v in neigh[u]:
                    if v in region_set:
                        continue
                    lv = int(labels[v])
                    if lv != gingiva_label:
                        boundary_labels.append(lv)

            if not boundary_labels:
                continue

            uniq, cnt = np.unique(np.asarray(boundary_labels, dtype=np.int64), return_counts=True)
            k = int(np.argmax(cnt))
            target_label = int(uniq[k])
            support = int(cnt[k])
            dom = support / float(len(boundary_labels))

            if support >= min_support and dom >= dominance:
                for u in region:
                    new_labels[u] = target_label
                changed += len(region)

        labels = new_labels
        if changed == 0:
            break

    return labels


# ============================================================
# Existing cleanup 2: remove tiny tooth islands on gingiva
# ============================================================

def remove_small_tooth_islands_on_gingiva(
    labels: np.ndarray,
    faces: np.ndarray,
    *,
    num_classes: int,
    probs: np.ndarray | None = None,
    max_iters: int = 2,
    max_region_size: int = 18,
    min_gingiva_support: int = 4,
    gingiva_dominance: float = 0.65,
    max_mean_label_prob: float = 0.60,
    max_mean_max_prob: float = 0.72,
) -> np.ndarray:
    """
    Remove small TOOTH islands that sit on gingiva.

    Only converts:
      tooth -> gingiva
    """
    labels = np.asarray(labels, dtype=np.int64).copy()
    faces = np.asarray(faces, dtype=np.int64)

    if labels.ndim != 1:
        labels = labels.reshape(-1)
    if faces.ndim != 2 or faces.shape[1] != 3:
        return labels

    probs_np = None
    if probs is not None:
        probs_np = np.asarray(probs, dtype=np.float32)
        if probs_np.ndim != 2 or probs_np.shape[0] != len(labels):
            probs_np = None

    F = int(faces.shape[0])
    if F == 0 or len(labels) != F:
        return labels

    gingiva_label = 16 if int(num_classes) >= 17 else max(int(num_classes) - 1, 0)
    neigh = _build_face_neighbors_by_edge(faces)

    max_iters = max(int(max_iters), 0)
    max_region_size = max(int(max_region_size), 1)
    min_gingiva_support = max(int(min_gingiva_support), 1)
    gingiva_dominance = float(gingiva_dominance)
    max_mean_label_prob = float(max_mean_label_prob)
    max_mean_max_prob = float(max_mean_max_prob)

    for _ in range(max_iters):
        changed = 0
        visited = np.zeros((F,), dtype=np.uint8)
        new_labels = labels.copy()

        for start in range(F):
            if visited[start]:
                continue

            start_label = int(labels[start])

            if start_label == gingiva_label:
                visited[start] = 1
                continue

            q = [start]
            visited[start] = 1
            region = [start]
            hit_limit = False

            qi = 0
            while qi < len(q):
                u = q[qi]
                qi += 1
                for v in neigh[u]:
                    if visited[v]:
                        continue
                    if int(labels[v]) != start_label:
                        continue
                    visited[v] = 1
                    q.append(v)
                    region.append(v)
                    if len(region) > max_region_size:
                        hit_limit = True
                        break
                if hit_limit:
                    break

            if hit_limit or len(region) > max_region_size:
                continue

            boundary_labels = _face_boundary_labels(region, labels, neigh)
            if boundary_labels.size == 0:
                continue

            gingiva_support = int(np.sum(boundary_labels == gingiva_label))
            dom = gingiva_support / float(len(boundary_labels))

            if gingiva_support < min_gingiva_support or dom < gingiva_dominance:
                continue

            if probs_np is not None:
                region_idx = np.asarray(region, dtype=np.int64)
                mean_label_prob = float(np.mean(probs_np[region_idx, start_label]))
                mean_max_prob = float(np.mean(np.max(probs_np[region_idx], axis=1)))

                if mean_label_prob > max_mean_label_prob:
                    continue
                if mean_max_prob > max_mean_max_prob:
                    continue

            for u in region:
                new_labels[u] = gingiva_label
            changed += len(region)

        labels = new_labels
        if changed == 0:
            break

    return labels


# ============================================================
# Existing cleanup 3: merge small foreign islands inside a tooth
# ============================================================

def merge_small_foreign_islands_inside_tooth(
    labels: np.ndarray,
    faces: np.ndarray,
    *,
    num_classes: int,
    probs: np.ndarray | None = None,
    max_iters: int = 2,
    max_region_size: int = 16,
    min_boundary_support: int = 4,
    dominance: float = 0.72,
    max_mean_label_prob: float = 0.60,
    max_mean_max_prob: float = 0.75,
    allow_to_gingiva: bool = True,
) -> np.ndarray:
    """
    Merge small foreign-label islands embedded inside another dominant region.

    Typical target:
      - a tiny patch of wrong tooth label inside another tooth
      - a small mislabeled patch on a tooth surface
      - optionally, a patch that should actually be gingiva
    """
    labels = np.asarray(labels, dtype=np.int64).copy()
    faces = np.asarray(faces, dtype=np.int64)

    if labels.ndim != 1:
        labels = labels.reshape(-1)
    if faces.ndim != 2 or faces.shape[1] != 3:
        return labels

    probs_np = None
    if probs is not None:
        probs_np = np.asarray(probs, dtype=np.float32)
        if probs_np.ndim != 2 or probs_np.shape[0] != len(labels):
            probs_np = None

    F = int(faces.shape[0])
    if F == 0 or len(labels) != F:
        return labels

    gingiva_label = 16 if int(num_classes) >= 17 else max(int(num_classes) - 1, 0)
    neigh = _build_face_neighbors_by_edge(faces)

    max_iters = max(int(max_iters), 0)
    max_region_size = max(int(max_region_size), 1)
    min_boundary_support = max(int(min_boundary_support), 1)
    dominance = float(dominance)
    max_mean_label_prob = float(max_mean_label_prob)
    max_mean_max_prob = float(max_mean_max_prob)
    allow_to_gingiva = bool(allow_to_gingiva)

    for _ in range(max_iters):
        changed = 0
        visited = np.zeros((F,), dtype=np.uint8)
        new_labels = labels.copy()

        for start in range(F):
            if visited[start]:
                continue

            start_label = int(labels[start])

            q = [start]
            visited[start] = 1
            region = [start]
            hit_limit = False

            qi = 0
            while qi < len(q):
                u = q[qi]
                qi += 1
                for v in neigh[u]:
                    if visited[v]:
                        continue
                    if int(labels[v]) != start_label:
                        continue
                    visited[v] = 1
                    q.append(v)
                    region.append(v)
                    if len(region) > max_region_size:
                        hit_limit = True
                        break
                if hit_limit:
                    break

            if hit_limit or len(region) > max_region_size:
                continue

            boundary_labels = _face_boundary_labels(region, labels, neigh)
            if boundary_labels.size == 0:
                continue

            if not allow_to_gingiva:
                boundary_labels = boundary_labels[boundary_labels != gingiva_label]
                if boundary_labels.size == 0:
                    continue

            uniq, cnt = np.unique(boundary_labels, return_counts=True)
            k = int(np.argmax(cnt))
            target_label = int(uniq[k])
            support = int(cnt[k])
            dom = support / float(len(boundary_labels))

            if target_label == start_label:
                continue
            if (target_label == gingiva_label) and (not allow_to_gingiva):
                continue
            if support < min_boundary_support or dom < dominance:
                continue

            if probs_np is not None:
                region_idx = np.asarray(region, dtype=np.int64)
                mean_label_prob = float(np.mean(probs_np[region_idx, start_label]))
                mean_max_prob = float(np.mean(np.max(probs_np[region_idx], axis=1)))

                if mean_label_prob > max_mean_label_prob:
                    continue
                if mean_max_prob > max_mean_max_prob:
                    continue

            for u in region:
                new_labels[u] = target_label
            changed += len(region)

        labels = new_labels
        if changed == 0:
            break

    return labels


# ============================================================
# NEW cleanup 4: relabel low-confidence boundary bleeding
# ============================================================

def relabel_boundary_bleeding_lowconf(
    labels: np.ndarray,
    faces: np.ndarray,
    *,
    num_classes: int,
    probs: np.ndarray | None = None,
    max_iters: int = 3,
    max_region_size: int = 120,
    min_boundary_support: int = 6,
    dominance: float = 0.62,
    max_mean_label_prob: float = 0.72,
    max_mean_max_prob: float = 0.82,
    min_mean_target_prob: float = 0.18,
    allow_to_gingiva: bool = False,
) -> np.ndarray:
    """
    Relabel larger LOW-CONFIDENCE strips/patches that "bleed" across a tooth boundary.

    This is for cases where:
      - the wrong area is not tiny anymore, so small-island cleanup does nothing
      - one tooth label spills onto the neighboring tooth
      - two neighboring distal teeth look fused by one label near the boundary

    Only low-confidence same-label regions are considered.
    The region is relabeled to the dominant surrounding boundary label.
    """
    labels = np.asarray(labels, dtype=np.int64).copy()
    faces = np.asarray(faces, dtype=np.int64)

    if labels.ndim != 1:
        labels = labels.reshape(-1)
    if faces.ndim != 2 or faces.shape[1] != 3:
        return labels

    probs_np = None
    if probs is not None:
        probs_np = np.asarray(probs, dtype=np.float32)
        if probs_np.ndim != 2 or probs_np.shape[0] != len(labels):
            probs_np = None

    F = int(faces.shape[0])
    if F == 0 or len(labels) != F:
        return labels

    if probs_np is None:
        return labels

    gingiva_label = 16 if int(num_classes) >= 17 else max(int(num_classes) - 1, 0)
    neigh = _build_face_neighbors_by_edge(faces)

    max_iters = max(int(max_iters), 0)
    max_region_size = max(int(max_region_size), 1)
    min_boundary_support = max(int(min_boundary_support), 1)
    dominance = float(dominance)
    max_mean_label_prob = float(max_mean_label_prob)
    max_mean_max_prob = float(max_mean_max_prob)
    min_mean_target_prob = float(min_mean_target_prob)
    allow_to_gingiva = bool(allow_to_gingiva)

    for _ in range(max_iters):
        changed = 0
        visited = np.zeros((F,), dtype=np.uint8)
        new_labels = labels.copy()

        # recompute each round after labels change
        max_prob = np.max(probs_np, axis=1)

        for start in range(F):
            if visited[start]:
                continue

            start_label = int(labels[start])
            if start_label == gingiva_label:
                visited[start] = 1
                continue

            # seed must be low-confidence for its CURRENT assigned label
            if float(probs_np[start, start_label]) > max_mean_label_prob and float(max_prob[start]) > max_mean_max_prob:
                visited[start] = 1
                continue

            # allowed_mask = same current label AND low-conf-ish
            allowed_mask = (
                (labels == start_label)
                & (
                    (probs_np[:, start_label] <= max_mean_label_prob)
                    | (max_prob <= max_mean_max_prob)
                )
            )

            region, hit_limit = _iter_connected_region_same_label(
                start,
                labels,
                neigh,
                allowed_mask=allowed_mask,
                max_region_size=max_region_size,
            )

            for u in region:
                visited[u] = 1

            if hit_limit or len(region) == 0 or len(region) > max_region_size:
                continue

            boundary_labels = _face_boundary_labels(region, labels, neigh)
            if boundary_labels.size == 0:
                continue

            if not allow_to_gingiva:
                boundary_labels = boundary_labels[boundary_labels != gingiva_label]
                if boundary_labels.size == 0:
                    continue

            uniq, cnt = np.unique(boundary_labels, return_counts=True)
            k = int(np.argmax(cnt))
            target_label = int(uniq[k])
            support = int(cnt[k])
            dom = support / float(len(boundary_labels))

            if target_label == start_label:
                continue
            if (target_label == gingiva_label) and (not allow_to_gingiva):
                continue
            if support < min_boundary_support or dom < dominance:
                continue

            region_idx = np.asarray(region, dtype=np.int64)

            mean_label_prob = float(np.mean(probs_np[region_idx, start_label]))
            mean_max_prob = float(np.mean(np.max(probs_np[region_idx], axis=1)))
            mean_target_prob = float(np.mean(probs_np[region_idx, target_label]))

            # current label should not be too strong
            if mean_label_prob > max_mean_label_prob and mean_max_prob > max_mean_max_prob:
                continue

            # target label should have at least some support in model probs
            if mean_target_prob < min_mean_target_prob:
                continue

            for u in region:
                new_labels[u] = target_label
            changed += len(region)

        labels = new_labels
        if changed == 0:
            break

    return labels


# ============================================================
# Unified entrypoint
# ============================================================

def run_postprocess(
    *,
    labels: np.ndarray,
    faces: np.ndarray | None,
    num_classes: int,
    centers: np.ndarray | None,
    probs: np.ndarray | None,
    preset: Preset,
) -> tuple[np.ndarray, Dict[str, Any]]:
    """
    Unified postprocess entrypoint for all runners.
    """
    if not preset.do_postprocess:
        return np.asarray(labels, dtype=np.int64), {"enabled": False}

    if centers is None:
        return np.asarray(labels, dtype=np.int64), {"enabled": False, "reason": "missing centers_used"}

    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    centers = np.asarray(centers, dtype=np.float32)
    probs_np = np.asarray(probs, dtype=np.float32) if probs is not None else None

    pp_mode = str(preset.pp_mode).lower()

    if pp_mode == "conservative":
        labels_out = postprocess_labels_conservative(
            labels=labels,
            centers=centers,
            probs=probs_np,
            num_classes=int(num_classes),
            k=int(preset.pp_knn_k),
            smooth_iters=max(int(preset.pp_smooth_iters), 1),
            ungingiva_margin=float(preset.pp_ungingiva_margin),
            ungingiva_min_tooth_p=float(preset.pp_ungingiva_min_tooth_p),
            keep_gingiva_lcc=bool(preset.pp_keep_gingiva_lcc),
        )
        meta: Dict[str, Any] = {
            "enabled": True,
            "mode": "conservative",
            "k": int(preset.pp_knn_k),
            "smooth_iters": max(int(preset.pp_smooth_iters), 1),
        }
    else:
        labels_out = postprocess_labels_best_visual(
            labels=labels,
            centers=centers,
            probs=probs_np,
            num_classes=int(num_classes),
            gingiva_label=16 if int(num_classes) >= 17 else max(int(num_classes) - 1, 0),
            k=int(preset.pp_knn_k),
            conf_thresh=float(preset.pp_conf_thresh),
            conf_neighbor_iters=int(preset.pp_conf_neighbor_iters),
            min_comp_size=int(preset.pp_min_comp_size),
            clean_iters=int(preset.pp_clean_iters),
            smooth_iters=int(preset.pp_smooth_iters),
            keep_gingiva_lcc=bool(preset.pp_keep_gingiva_lcc),
            ungingiva_margin=float(preset.pp_ungingiva_margin),
            ungingiva_min_tooth_p=float(preset.pp_ungingiva_min_tooth_p),
        )
        meta = {
            "enabled": True,
            "mode": "best_visual",
            "k": int(preset.pp_knn_k),
            "smooth_iters": int(preset.pp_smooth_iters),
            "conf_thresh": float(preset.pp_conf_thresh),
            "conf_neighbor_iters": int(preset.pp_conf_neighbor_iters),
            "min_comp_size": int(preset.pp_min_comp_size),
            "clean_iters": int(preset.pp_clean_iters),
        }

    labels_out = np.asarray(labels_out, dtype=np.int64).reshape(-1)

    # ---------------------------------------
    # Extra cleanup: remove tiny tooth islands on gingiva
    # ---------------------------------------
    remove_tooth_islands_enabled = bool(
        preset.extras.get("pp_remove_tooth_islands_enabled", True)
    )
    remove_tooth_islands_iters = int(
        preset.extras.get("pp_remove_tooth_islands_iters", 2)
    )
    remove_tooth_islands_max_region_size = int(
        preset.extras.get("pp_remove_tooth_islands_max_region_size", 18)
    )
    remove_tooth_islands_min_gingiva_support = int(
        preset.extras.get("pp_remove_tooth_islands_min_gingiva_support", 4)
    )
    remove_tooth_islands_dominance = float(
        preset.extras.get("pp_remove_tooth_islands_dominance", 0.65)
    )
    remove_tooth_islands_max_mean_label_prob = float(
        preset.extras.get("pp_remove_tooth_islands_max_mean_label_prob", 0.60)
    )
    remove_tooth_islands_max_mean_max_prob = float(
        preset.extras.get("pp_remove_tooth_islands_max_mean_max_prob", 0.72)
    )

    if remove_tooth_islands_enabled and faces is not None:
        labels_out = remove_small_tooth_islands_on_gingiva(
            labels_out,
            np.asarray(faces, dtype=np.int64),
            num_classes=int(num_classes),
            probs=probs_np,
            max_iters=remove_tooth_islands_iters,
            max_region_size=remove_tooth_islands_max_region_size,
            min_gingiva_support=remove_tooth_islands_min_gingiva_support,
            gingiva_dominance=remove_tooth_islands_dominance,
            max_mean_label_prob=remove_tooth_islands_max_mean_label_prob,
            max_mean_max_prob=remove_tooth_islands_max_mean_max_prob,
        )
        meta["remove_tooth_islands_enabled"] = True
        meta["remove_tooth_islands_iters"] = remove_tooth_islands_iters
        meta["remove_tooth_islands_max_region_size"] = remove_tooth_islands_max_region_size
        meta["remove_tooth_islands_min_gingiva_support"] = remove_tooth_islands_min_gingiva_support
        meta["remove_tooth_islands_dominance"] = remove_tooth_islands_dominance
        meta["remove_tooth_islands_max_mean_label_prob"] = remove_tooth_islands_max_mean_label_prob
        meta["remove_tooth_islands_max_mean_max_prob"] = remove_tooth_islands_max_mean_max_prob
    else:
        meta["remove_tooth_islands_enabled"] = False

    # ---------------------------------------
    # Extra cleanup: merge small foreign-label islands inside a dominant region
    # ---------------------------------------
    merge_foreign_islands_enabled = bool(
        preset.extras.get("pp_merge_foreign_islands_enabled", True)
    )
    merge_foreign_islands_iters = int(
        preset.extras.get("pp_merge_foreign_islands_iters", 2)
    )
    merge_foreign_islands_max_region_size = int(
        preset.extras.get("pp_merge_foreign_islands_max_region_size", 16)
    )
    merge_foreign_islands_min_boundary_support = int(
        preset.extras.get("pp_merge_foreign_islands_min_boundary_support", 4)
    )
    merge_foreign_islands_dominance = float(
        preset.extras.get("pp_merge_foreign_islands_dominance", 0.72)
    )
    merge_foreign_islands_max_mean_label_prob = float(
        preset.extras.get("pp_merge_foreign_islands_max_mean_label_prob", 0.60)
    )
    merge_foreign_islands_max_mean_max_prob = float(
        preset.extras.get("pp_merge_foreign_islands_max_mean_max_prob", 0.75)
    )
    merge_foreign_islands_allow_to_gingiva = bool(
        preset.extras.get("pp_merge_foreign_islands_allow_to_gingiva", True)
    )

    if merge_foreign_islands_enabled and faces is not None:
        labels_out = merge_small_foreign_islands_inside_tooth(
            labels_out,
            np.asarray(faces, dtype=np.int64),
            num_classes=int(num_classes),
            probs=probs_np,
            max_iters=merge_foreign_islands_iters,
            max_region_size=merge_foreign_islands_max_region_size,
            min_boundary_support=merge_foreign_islands_min_boundary_support,
            dominance=merge_foreign_islands_dominance,
            max_mean_label_prob=merge_foreign_islands_max_mean_label_prob,
            max_mean_max_prob=merge_foreign_islands_max_mean_max_prob,
            allow_to_gingiva=merge_foreign_islands_allow_to_gingiva,
        )
        meta["merge_foreign_islands_enabled"] = True
        meta["merge_foreign_islands_iters"] = merge_foreign_islands_iters
        meta["merge_foreign_islands_max_region_size"] = merge_foreign_islands_max_region_size
        meta["merge_foreign_islands_min_boundary_support"] = merge_foreign_islands_min_boundary_support
        meta["merge_foreign_islands_dominance"] = merge_foreign_islands_dominance
        meta["merge_foreign_islands_max_mean_label_prob"] = merge_foreign_islands_max_mean_label_prob
        meta["merge_foreign_islands_max_mean_max_prob"] = merge_foreign_islands_max_mean_max_prob
        meta["merge_foreign_islands_allow_to_gingiva"] = merge_foreign_islands_allow_to_gingiva
    else:
        meta["merge_foreign_islands_enabled"] = False

    # ---------------------------------------
    # NEW: relabel larger low-confidence boundary bleeding
    # ---------------------------------------
    relabel_boundary_bleeding_enabled = bool(
        preset.extras.get("pp_relabel_boundary_bleeding_enabled", False)
    )
    relabel_boundary_bleeding_iters = int(
        preset.extras.get("pp_relabel_boundary_bleeding_iters", 3)
    )
    relabel_boundary_bleeding_max_region_size = int(
        preset.extras.get("pp_relabel_boundary_bleeding_max_region_size", 120)
    )
    relabel_boundary_bleeding_min_boundary_support = int(
        preset.extras.get("pp_relabel_boundary_bleeding_min_boundary_support", 6)
    )
    relabel_boundary_bleeding_dominance = float(
        preset.extras.get("pp_relabel_boundary_bleeding_dominance", 0.62)
    )
    relabel_boundary_bleeding_max_mean_label_prob = float(
        preset.extras.get("pp_relabel_boundary_bleeding_max_mean_label_prob", 0.72)
    )
    relabel_boundary_bleeding_max_mean_max_prob = float(
        preset.extras.get("pp_relabel_boundary_bleeding_max_mean_max_prob", 0.82)
    )
    relabel_boundary_bleeding_min_mean_target_prob = float(
        preset.extras.get("pp_relabel_boundary_bleeding_min_mean_target_prob", 0.18)
    )
    relabel_boundary_bleeding_allow_to_gingiva = bool(
        preset.extras.get("pp_relabel_boundary_bleeding_allow_to_gingiva", False)
    )

    if relabel_boundary_bleeding_enabled and faces is not None and probs_np is not None:
        labels_out = relabel_boundary_bleeding_lowconf(
            labels_out,
            np.asarray(faces, dtype=np.int64),
            num_classes=int(num_classes),
            probs=probs_np,
            max_iters=relabel_boundary_bleeding_iters,
            max_region_size=relabel_boundary_bleeding_max_region_size,
            min_boundary_support=relabel_boundary_bleeding_min_boundary_support,
            dominance=relabel_boundary_bleeding_dominance,
            max_mean_label_prob=relabel_boundary_bleeding_max_mean_label_prob,
            max_mean_max_prob=relabel_boundary_bleeding_max_mean_max_prob,
            min_mean_target_prob=relabel_boundary_bleeding_min_mean_target_prob,
            allow_to_gingiva=relabel_boundary_bleeding_allow_to_gingiva,
        )
        meta["relabel_boundary_bleeding_enabled"] = True
        meta["relabel_boundary_bleeding_iters"] = relabel_boundary_bleeding_iters
        meta["relabel_boundary_bleeding_max_region_size"] = relabel_boundary_bleeding_max_region_size
        meta["relabel_boundary_bleeding_min_boundary_support"] = relabel_boundary_bleeding_min_boundary_support
        meta["relabel_boundary_bleeding_dominance"] = relabel_boundary_bleeding_dominance
        meta["relabel_boundary_bleeding_max_mean_label_prob"] = relabel_boundary_bleeding_max_mean_label_prob
        meta["relabel_boundary_bleeding_max_mean_max_prob"] = relabel_boundary_bleeding_max_mean_max_prob
        meta["relabel_boundary_bleeding_min_mean_target_prob"] = relabel_boundary_bleeding_min_mean_target_prob
        meta["relabel_boundary_bleeding_allow_to_gingiva"] = relabel_boundary_bleeding_allow_to_gingiva
    else:
        meta["relabel_boundary_bleeding_enabled"] = False

    # ---------------------------------------
    # Extra boundary-gap closing
    # ---------------------------------------
    close_gap_enabled = bool(preset.extras.get("pp_close_gap_enabled", False))
    close_gap_iters = int(preset.extras.get("pp_close_gap_iters", 2))
    close_gap_min_support = int(preset.extras.get("pp_close_gap_min_support", 3))
    close_gap_dominance = float(preset.extras.get("pp_close_gap_dominance", 0.60))
    close_gap_max_region_size = int(preset.extras.get("pp_close_gap_max_region_size", 12))

    if close_gap_enabled and faces is not None:
        labels_out = close_small_boundary_gaps(
            labels_out,
            np.asarray(faces, dtype=np.int64),
            num_classes=int(num_classes),
            max_iters=close_gap_iters,
            min_support=close_gap_min_support,
            dominance=close_gap_dominance,
            max_region_size=close_gap_max_region_size,
        )
        meta["close_gap_enabled"] = True
        meta["close_gap_iters"] = close_gap_iters
        meta["close_gap_min_support"] = close_gap_min_support
        meta["close_gap_dominance"] = close_gap_dominance
        meta["close_gap_max_region_size"] = close_gap_max_region_size
    else:
        meta["close_gap_enabled"] = False

    return labels_out, meta