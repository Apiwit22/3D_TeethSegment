# dental_seg_app/app/data/postprocess.py
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

            # fill dropped with neighbor majority (teeth)
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