from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


def infer_gingiva_label(num_classes: int) -> Optional[int]:
    if num_classes == 17:
        return 16
    if num_classes == 33:
        return 32
    return None


def teeth_face_mask(labels_face: np.ndarray, num_classes: int) -> Optional[np.ndarray]:
    if labels_face is None:
        return None
    labels_face = np.asarray(labels_face)
    ging = infer_gingiva_label(int(num_classes))
    if ging is not None:
        return labels_face != ging
    try:
        ging2 = int(labels_face.max())
        return labels_face != ging2
    except Exception:
        return None


def face_centroids(pos: np.ndarray, faces: np.ndarray) -> np.ndarray:
    p = np.asarray(pos, dtype=np.float64)
    f = np.asarray(faces, dtype=np.int64)
    tri = p[f]
    return tri.mean(axis=1)


def canonicalize_pose(
    pos: np.ndarray,
    arch: str,
    *,
    faces: Optional[np.ndarray] = None,
    labels_face: Optional[np.ndarray] = None,
    num_classes: int = 0,
) -> np.ndarray:
    p = np.asarray(pos, dtype=np.float64)
    c = p.mean(axis=0)
    x = p - c

    cov = (x.T @ x) / max(len(x) - 1, 1)
    _, v = np.linalg.eigh(cov)
    v0, v1, v2 = v[:, 0], v[:, 1], v[:, 2]

    z_axis = v0
    x_axis = v2
    y_axis = np.cross(z_axis, x_axis)
    ny = float(np.linalg.norm(y_axis))

    if ny < 1e-9:
        x_axis = v1
        y_axis = np.cross(z_axis, x_axis)
        ny = float(np.linalg.norm(y_axis))
        if ny < 1e-9:
            return (x + c).astype(np.float32)

    y_axis /= ny
    x_axis = np.cross(y_axis, z_axis)
    x_axis /= (np.linalg.norm(x_axis) + 1e-12)
    z_axis /= (np.linalg.norm(z_axis) + 1e-12)

    r = np.stack([x_axis, y_axis, z_axis], axis=0)
    xr = x @ r.T

    xx = xr[:, 0]
    x_lo = np.percentile(xx, 10)
    x_hi = np.percentile(xx, 90)
    left_band = xr[xx <= x_lo]
    right_band = xr[xx >= x_hi]

    def y_iqr(pband: np.ndarray) -> float:
        if pband.size == 0:
            return 1e9
        yy = pband[:, 1]
        q1, q3 = np.percentile(yy, 25), np.percentile(yy, 75)
        return float(q3 - q1)

    # คง heuristic เดิมไว้ก่อน
    # ถ้ายังมีเคสกลับซ้าย-ขวาอีก ค่อยมา tighten logic ตรงนี้ต่อ
    if y_iqr(right_band) > y_iqr(left_band):
        xr[:, 0] *= -1

    z_use = xr[:, 2]
    if faces is not None and labels_face is not None and int(num_classes) > 0:
        try:
            cents = face_centroids(xr + c, np.asarray(faces, dtype=np.int64)) - c
            m = teeth_face_mask(labels_face, num_classes=int(num_classes))
            if m is not None and m.shape[0] == cents.shape[0] and np.any(m):
                z_use = cents[m][:, 2]
        except Exception:
            z_use = xr[:, 2]

    z_top = float(np.percentile(z_use, 95))
    z_bot = float(np.percentile(z_use, 5))

    arch = (arch or "").lower()
    if arch == "upper":
        if z_top > abs(z_bot):
            xr[:, 2] *= -1
    else:
        if abs(z_top) < abs(z_bot):
            xr[:, 2] *= -1

    return (xr + c).astype(np.float32)


def _make_rng(seed: Optional[int] = None, rng: Optional[np.random.Generator] = None) -> np.random.Generator:
    if rng is not None:
        return rng
    if seed is None:
        return np.random.default_rng(1234)
    return np.random.default_rng(int(seed))


def sample_teeth_points(
    pos: np.ndarray,
    faces: np.ndarray,
    labels_face: Optional[np.ndarray],
    num_classes: int,
    n_samples: int = 3000,
    *,
    seed: Optional[int] = 1234,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    cents = face_centroids(pos, faces)
    if labels_face is not None:
        m = teeth_face_mask(labels_face, num_classes=num_classes)
        if m is not None and m.shape[0] == cents.shape[0] and np.any(m):
            cents = cents[m]

    if cents.shape[0] == 0:
        return np.asarray(pos, dtype=np.float64)

    if cents.shape[0] > n_samples:
        rr = _make_rng(seed=seed, rng=rng)
        idx = rr.choice(cents.shape[0], size=n_samples, replace=False)
        cents = cents[idx]

    return np.asarray(cents, dtype=np.float64)


def best_fit_transform(A: np.ndarray, B: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if A.shape != B.shape:
        raise ValueError(f"A and B must have the same shape, got {A.shape} vs {B.shape}")

    cA = A.mean(axis=0)
    cB = B.mean(axis=0)
    AA = A - cA
    BB = B - cB
    H = AA.T @ BB
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # กัน reflection จาก SVD
    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = Vt.T @ U.T

    t = cB - (cA @ R.T)
    return R, t


def apply_rt(P: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    return (P @ R.T) + t


def nearest_neighbor(src: np.ndarray, dst: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if src.size == 0 or dst.size == 0:
        raise ValueError("nearest_neighbor received empty source or destination")

    try:
        from scipy.spatial import cKDTree  # type: ignore

        tree = cKDTree(dst)
        d, idx = tree.query(src, k=1)
        return dst[idx], d
    except Exception:
        d2 = np.sum((src[:, None, :] - dst[None, :, :]) ** 2, axis=2)
        idx = np.argmin(d2, axis=1)
        d = np.sqrt(d2[np.arange(src.shape[0]), idx])
        return dst[idx], d


def icp_rigid(
    src: np.ndarray,
    dst: np.ndarray,
    iters: int = 45,
    trim: float = 0.75,
) -> Tuple[np.ndarray, np.ndarray]:
    P = np.asarray(src, dtype=np.float64)
    Q = np.asarray(dst, dtype=np.float64)

    if P.ndim != 2 or P.shape[1] != 3:
        raise ValueError(f"src must have shape (N,3), got {P.shape}")
    if Q.ndim != 2 or Q.shape[1] != 3:
        raise ValueError(f"dst must have shape (M,3), got {Q.shape}")

    R_all = np.eye(3, dtype=np.float64)
    t_all = np.zeros(3, dtype=np.float64)

    prev_err = None
    for _ in range(int(iters)):
        Qm, d = nearest_neighbor(P, Q)

        if d.size > 10:
            k = int(max(10, np.floor(d.size * float(trim))))
            keep = np.argsort(d)[:k]
            Pk = P[keep]
            Qk = Qm[keep]
        else:
            Pk, Qk = P, Qm

        R, t = best_fit_transform(Pk, Qk)
        P = apply_rt(P, R, t)

        t_all = (t_all @ R.T) + t
        R_all = R @ R_all

        err = float(np.mean(d))
        if prev_err is not None and abs(prev_err - err) < 1e-6:
            break
        prev_err = err

    return R_all, t_all


def nn_trimmed_error(src: np.ndarray, dst: np.ndarray, trim: float = 0.85) -> float:
    _, d = nearest_neighbor(src, dst)
    if d.size < 10:
        return float(np.mean(d)) if d.size else 1e9
    k = int(max(10, np.floor(d.size * float(trim))))
    dd = np.sort(d)[:k]
    return float(np.mean(dd))


def rot_z_180(P: np.ndarray) -> np.ndarray:
    Q = np.asarray(P, dtype=np.float64).copy()
    Q[:, 0] *= -1
    Q[:, 1] *= -1
    return Q.astype(np.float32)


def flip_x(P: np.ndarray) -> np.ndarray:
    Q = np.asarray(P, dtype=np.float64).copy()
    Q[:, 0] *= -1
    return Q.astype(np.float32)


def flip_y(P: np.ndarray) -> np.ndarray:
    Q = np.asarray(P, dtype=np.float64).copy()
    Q[:, 1] *= -1
    return Q.astype(np.float32)


def align_upper_to_lower_multihyp(
    posU: np.ndarray,
    posL: np.ndarray,
    facesU: np.ndarray,
    facesL: np.ndarray,
    labelsU: Optional[np.ndarray],
    labelsL: Optional[np.ndarray],
    numcU: int,
    numcL: int,
    *,
    seed: int = 1234,
) -> np.ndarray:
    ptsL = sample_teeth_points(posL, facesL, labelsL, numcL, n_samples=3000, seed=seed)

    # สำคัญ:
    # ใช้เฉพาะ proper rotations
    # ไม่ใช้ flip_x / flip_y เพราะเป็น mirror reflection
    # ซึ่งทำให้ FDI ซ้าย-ขวาสลับตอนแสดงผลร่วมกับ lower
    hyps = [
        ("id", lambda P: P),
        ("rz180", rot_z_180),
    ]

    best_posU = np.asarray(posU, dtype=np.float32)
    best_err = 1e18

    for i, (_, fn) in enumerate(hyps):
        candU = fn(np.asarray(posU, dtype=np.float32).copy())
        ptsU = sample_teeth_points(
            candU,
            facesU,
            labelsU,
            numcU,
            n_samples=3000,
            seed=seed + i + 1,
        )

        cL = np.median(ptsL, axis=0)
        cU = np.median(ptsU, axis=0)
        dx = float(cL[0] - cU[0])
        dy = float(cL[1] - cU[1])

        candU[:, 0] += dx
        candU[:, 1] += dy

        ptsU2 = ptsU.copy()
        ptsU2[:, 0] += dx
        ptsU2[:, 1] += dy

        R, t = icp_rigid(ptsU2, ptsL, iters=45, trim=0.75)
        candU_aligned = apply_rt(candU.astype(np.float64), R, t).astype(np.float32)
        ptsU_aligned = apply_rt(ptsU2, R, t)

        err = nn_trimmed_error(ptsU_aligned, ptsL, trim=0.85)
        if err < best_err:
            best_err = err
            best_posU = candU_aligned

    return best_posU


def close_bite_by_z(
    upper_pos: np.ndarray,
    lower_pos: np.ndarray,
    *,
    upper_teeth_pts: np.ndarray,
    lower_teeth_pts: np.ndarray,
    target_gap: float = 0.5,
    safety: float = 0.15,
) -> np.ndarray:
    U = np.asarray(upper_pos, dtype=np.float64).copy()
    L = np.asarray(lower_pos, dtype=np.float64)

    Upts = np.asarray(upper_teeth_pts, dtype=np.float64)
    Lpts = np.asarray(lower_teeth_pts, dtype=np.float64)

    if Upts.size == 0 or Lpts.size == 0:
        return U.astype(np.float32)

    uz = Upts[:, 2]
    lz = Lpts[:, 2]

    upper_bottom = float(np.percentile(uz, 5))
    lower_top = float(np.percentile(lz, 95))
    dz = upper_bottom - (lower_top + float(target_gap))

    l_top_mesh = float(np.percentile(L[:, 2], 98))
    min_allowed_upper_bottom = l_top_mesh + float(safety)

    upper_bottom_after = float(np.percentile(U[:, 2] - dz, 2))
    if upper_bottom_after < min_allowed_upper_bottom:
        dz = float(np.percentile(U[:, 2], 2) - min_allowed_upper_bottom)

    U[:, 2] -= dz
    return U.astype(np.float32)