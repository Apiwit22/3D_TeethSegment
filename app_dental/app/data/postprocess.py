# app/data/postprocess.py
from __future__ import annotations

from collections import deque
import numpy as np


def build_face_knn(centers: np.ndarray, k: int = 12) -> np.ndarray:
    k = int(k)
    if k <= 0:
        raise ValueError("k must be > 0")

    centers = np.asarray(centers, dtype=np.float32)
    if centers.ndim != 2 or centers.shape[1] != 3:
        raise ValueError(f"centers must be (F,3), got {centers.shape}")

    try:
        from scipy.spatial import cKDTree  # type: ignore
        tree = cKDTree(centers)
        _, idx = tree.query(centers, k=min(k + 1, centers.shape[0]))
        idx = np.asarray(idx, dtype=np.int64)
        if idx.ndim == 1:
            idx = idx[:, None]
        if idx.shape[1] > 1 and np.all(idx[:, 0] == np.arange(centers.shape[0])):
            idx = idx[:, 1:]
        if idx.shape[1] < k:
            pad = np.repeat(idx[:, [-1]], k - idx.shape[1], axis=1)
            idx = np.concatenate([idx, pad], axis=1)
        elif idx.shape[1] > k:
            idx = idx[:, :k]
        return idx
    except Exception:
        pass

    try:
        from sklearn.neighbors import NearestNeighbors  # type: ignore
        nn = NearestNeighbors(n_neighbors=min(k + 1, centers.shape[0]))
        nn.fit(centers)
        idx = nn.kneighbors(centers, return_distance=False).astype(np.int64)
        if idx.shape[1] > 1 and np.all(idx[:, 0] == np.arange(centers.shape[0])):
            idx = idx[:, 1:]
        if idx.shape[1] < k:
            pad = np.repeat(idx[:, [-1]], k - idx.shape[1], axis=1)
            idx = np.concatenate([idx, pad], axis=1)
        elif idx.shape[1] > k:
            idx = idx[:, :k]
        return idx
    except Exception as e:
        raise RuntimeError("Need scipy or scikit-learn for KNN. pip install scipy") from e


def majority_smooth(
    labels: np.ndarray,
    nbr: np.ndarray,
    *,
    num_classes: int,
    iters: int = 1,
    ignore_index: int = -1,
) -> np.ndarray:
    y = np.asarray(labels, dtype=np.int64).copy()
    nbr = np.asarray(nbr, dtype=np.int64)

    F = len(y)
    iters = int(iters)
    num_classes = int(num_classes)

    for _ in range(max(iters, 0)):
        y2 = y.copy()
        for i in range(F):
            li = int(y[i])
            if li == ignore_index:
                continue
            nn = nbr[i]
            votes = np.zeros((num_classes,), dtype=np.int32)
            votes[li] += 1
            for j in nn:
                lj = int(y[j])
                if lj != ignore_index:
                    votes[lj] += 1
            y2[i] = int(np.argmax(votes))
        y = y2
    return y


def largest_connected_component(mask: np.ndarray, nbr: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    nbr = np.asarray(nbr, dtype=np.int64)
    F = len(mask)

    visited = np.zeros((F,), dtype=bool)
    best = []
    for i in range(F):
        if not mask[i] or visited[i]:
            continue
        q = deque([i])
        visited[i] = True
        comp = [i]
        while q:
            u = q.popleft()
            for v in nbr[u]:
                if not visited[v] and mask[v]:
                    visited[v] = True
                    q.append(v)
                    comp.append(v)
        if len(comp) > len(best):
            best = comp

    out = np.zeros((F,), dtype=bool)
    if best:
        out[np.asarray(best, dtype=np.int64)] = True
    return out


def ungingiva_by_margin(
    labels: np.ndarray,
    probs: np.ndarray | None,
    *,
    num_classes: int,
    gingiva_label: int,
    min_tooth_p: float,
    margin: float,
) -> np.ndarray:
    y = np.asarray(labels, dtype=np.int64).copy()
    if probs is None:
        return y

    P = np.asarray(probs, dtype=np.float32)
    if P.ndim != 2 or P.shape[1] != num_classes:
        return y

    g = int(gingiva_label)
    min_tooth_p = float(min_tooth_p)
    margin = float(margin)

    p_g = P[:, g]
    tooth_max = np.max(P[:, :g], axis=1) if g > 0 else np.zeros_like(p_g)
    cond = (y == g) & (tooth_max >= min_tooth_p) & ((tooth_max - p_g) >= margin)

    if np.any(cond):
        y[cond] = np.argmax(P[cond, :g], axis=1).astype(np.int64)
    return y


# ============================================================
# ✅ NEW: low-confidence neighbor fill (Policy C)
# ============================================================
def fill_lowconf_by_neighbors(
    labels: np.ndarray,
    probs: np.ndarray | None,
    nbr: np.ndarray,
    *,
    num_classes: int,
    conf_thresh: float = 0.60,
    iters: int = 1,
    ignore_index: int = -1,
) -> np.ndarray:
    """
    Policy C:
    - compute conf = max(probs)
    - for faces with conf < conf_thresh, replace label with neighbor majority
      (prefer neighbors that are not low-conf)
    """
    y = np.asarray(labels, dtype=np.int64).copy()
    if probs is None:
        return y

    P = np.asarray(probs, dtype=np.float32)
    if P.ndim != 2 or P.shape[1] != int(num_classes):
        return y

    nbr = np.asarray(nbr, dtype=np.int64)
    conf = np.max(P, axis=1)
    low = conf < float(conf_thresh)

    if not np.any(low):
        return y

    num_classes = int(num_classes)
    iters = int(iters)

    for _ in range(max(iters, 0)):
        y2 = y.copy()
        for i in np.where(low)[0]:
            if int(y[i]) == ignore_index:
                continue
            nn = nbr[i]

            # prefer neighbors that are not low-conf
            nn_good = [j for j in nn if (not low[j]) and (int(y[j]) != ignore_index)]
            src = nn_good if nn_good else [j for j in nn if int(y[j]) != ignore_index]
            if not src:
                continue

            votes = np.zeros((num_classes,), dtype=np.int32)
            for j in src:
                votes[int(y[j])] += 1
            y2[i] = int(np.argmax(votes))
        y = y2

    return y


# ============================================================
# ✅ NEW: cleanup tiny components for ALL labels
# ============================================================
def cleanup_small_components_all_labels(
    labels: np.ndarray,
    nbr: np.ndarray,
    *,
    num_classes: int,
    min_size: int = 20,
    iters: int = 1,
    ignore_index: int = -1,
) -> np.ndarray:
    """
    Remove small connected components for every label:
      - find connected components under nbr graph for each label
      - if component size < min_size:
          reassign to boundary neighbor majority (labels outside component)
    """
    y = np.asarray(labels, dtype=np.int64).copy()
    nbr = np.asarray(nbr, dtype=np.int64)

    F = len(y)
    num_classes = int(num_classes)
    min_size = int(min_size)
    iters = int(iters)

    if F == 0 or min_size <= 0 or iters <= 0:
        return y

    for _ in range(iters):
        visited = np.zeros((F,), dtype=bool)
        changed = 0

        for start in range(F):
            if visited[start]:
                continue
            lb = int(y[start])
            if lb == ignore_index:
                visited[start] = True
                continue

            # BFS component for this label
            q = deque([start])
            visited[start] = True
            comp = [start]

            while q:
                u = q.popleft()
                for v in nbr[u]:
                    if visited[v]:
                        continue
                    if int(y[v]) != lb:
                        continue
                    visited[v] = True
                    q.append(v)
                    comp.append(v)

            if len(comp) >= min_size:
                continue

            comp_set = set(comp)

            # boundary votes from neighbors outside component
            boundary = []
            for u in comp:
                for v in nbr[u]:
                    if v in comp_set:
                        continue
                    lv = int(y[v])
                    if lv != ignore_index:
                        boundary.append(lv)

            if not boundary:
                continue

            uvals, cnts = np.unique(np.asarray(boundary, dtype=np.int64), return_counts=True)
            new_lb = int(uvals[np.argmax(cnts)])

            if new_lb != lb:
                y[np.asarray(comp, dtype=np.int64)] = new_lb
                changed += len(comp)

        if changed == 0:
            break

    return y


# ============================================================
# Original conservative postprocess (kept for compatibility)
# ============================================================
def postprocess_labels_conservative(
    labels: np.ndarray,
    centers: np.ndarray,
    probs: np.ndarray | None,
    *,
    num_classes: int,
    gingiva_label: int = 16,
    k: int = 16,
    smooth_iters: int = 1,
    ungingiva_margin: float = 0.05,
    ungingiva_min_tooth_p: float = 0.12,
    keep_gingiva_lcc: bool = True,
) -> np.ndarray:
    y = np.asarray(labels, dtype=np.int64).copy()
    F = len(y)
    if F == 0:
        return y

    nbr = build_face_knn(centers, k=k)

    # 1) smooth
    y = majority_smooth(y, nbr, num_classes=num_classes, iters=smooth_iters, ignore_index=-1)

    # 2) ungingiva
    y = ungingiva_by_margin(
        y,
        probs,
        num_classes=num_classes,
        gingiva_label=gingiva_label,
        min_tooth_p=ungingiva_min_tooth_p,
        margin=ungingiva_margin,
    )

    # 3) keep only largest gingiva component
    if keep_gingiva_lcc:
        g = int(gingiva_label)
        mask = (y == g)
        if np.any(mask):
            lcc = largest_connected_component(mask, nbr)
            y[mask & (~lcc)] = -1  # drop tiny gingiva islands

            # fill dropped with neighbor majority
            dropped = (y == -1)
            if np.any(dropped):
                y2 = y.copy()
                for i in np.where(dropped)[0]:
                    nn = nbr[i]
                    votes = np.zeros((num_classes,), dtype=np.int32)
                    for j in nn:
                        lj = int(y[j])
                        if lj >= 0:
                            votes[lj] += 1
                    y2[i] = int(np.argmax(votes))
                y = y2

    return y


# ============================================================
# ✅ NEW: "best visual" postprocess for app (match visualize.py)
# ============================================================
def postprocess_labels_best_visual(
    labels: np.ndarray,
    centers: np.ndarray,
    probs: np.ndarray | None,
    *,
    num_classes: int,
    gingiva_label: int = 16,
    k: int = 16,
    # Policy C
    conf_thresh: float = 0.60,
    conf_neighbor_iters: int = 1,
    # cleanup
    min_comp_size: int = 20,
    clean_iters: int = 1,
    # smoothing (light)
    smooth_iters: int = 1,
    # keep gingiva lcc (optional)
    keep_gingiva_lcc: bool = True,
    # ungingiva (optional)
    ungingiva_margin: float = 0.05,
    ungingiva_min_tooth_p: float = 0.12,
) -> np.ndarray:
    """
    Recommended app visual pipeline:
      1) build KNN graph on centers
      2) (optional) light smooth
      3) low-conf neighbor fill (Policy C)
      4) cleanup small components for ALL labels
      5) ungingiva + gingiva LCC keep (optional)
      6) final light smooth (optional)
    """
    y = np.asarray(labels, dtype=np.int64).copy()
    F = len(y)
    if F == 0:
        return y

    nbr = build_face_knn(centers, k=int(k))

    # A) light smooth first (reduce noise before decisions)
    if int(smooth_iters) > 0:
        y = majority_smooth(y, nbr, num_classes=int(num_classes), iters=int(smooth_iters), ignore_index=-1)

    # B) Policy C: fill low-confidence by neighbors (no gingiva holes)
    if probs is not None and float(conf_thresh) > 0:
        y = fill_lowconf_by_neighbors(
            y,
            probs,
            nbr,
            num_classes=int(num_classes),
            conf_thresh=float(conf_thresh),
            iters=int(conf_neighbor_iters),
            ignore_index=-1,
        )

    # C) Cleanup small islands for ALL labels (fix "mixed colors in one tooth")
    if int(min_comp_size) > 0 and int(clean_iters) > 0:
        y = cleanup_small_components_all_labels(
            y,
            nbr,
            num_classes=int(num_classes),
            min_size=int(min_comp_size),
            iters=int(clean_iters),
            ignore_index=-1,
        )

    # D) ungingiva (optional)
    y = ungingiva_by_margin(
        y,
        probs,
        num_classes=int(num_classes),
        gingiva_label=int(gingiva_label),
        min_tooth_p=float(ungingiva_min_tooth_p),
        margin=float(ungingiva_margin),
    )

    # E) keep gingiva LCC (optional)
    if keep_gingiva_lcc:
        g = int(gingiva_label)
        mask = (y == g)
        if np.any(mask):
            lcc = largest_connected_component(mask, nbr)
            y[mask & (~lcc)] = -1
            dropped = (y == -1)
            if np.any(dropped):
                # fill dropped by neighbor majority
                y2 = y.copy()
                for i in np.where(dropped)[0]:
                    nn = nbr[i]
                    votes = np.zeros((int(num_classes),), dtype=np.int32)
                    for j in nn:
                        lj = int(y[j])
                        if lj >= 0:
                            votes[lj] += 1
                    y2[i] = int(np.argmax(votes))
                y = y2

    return y