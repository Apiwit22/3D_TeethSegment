# dental_seg_app/app/preprocess/orient.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np

from scipy.spatial import cKDTree


@dataclass(frozen=True)
class OrientConfig:
    ref_upper: Path
    ref_lower: Path
    sample_n: int = 30000
    seed: int = 1234
    allow_reflection: bool = False


def _pca_basis(P: np.ndarray) -> np.ndarray:
    C = (P.T @ P) / max(len(P), 1)
    w, V = np.linalg.eigh(C)
    order = np.argsort(w)[::-1]
    U = V[:, order]
    if np.linalg.det(U) < 0:
        U[:, -1] *= -1
    return U


def _generate_axis_rotations(allow_reflection: bool = False):
    mats = []
    perms = [
        (0, 1, 2), (0, 2, 1),
        (1, 0, 2), (1, 2, 0),
        (2, 0, 1), (2, 1, 0),
    ]
    signs = [
        (1, 1, 1), (1, 1, -1), (1, -1, 1), (1, -1, -1),
        (-1, 1, 1), (-1, 1, -1), (-1, -1, 1), (-1, -1, -1),
    ]
    for p in perms:
        Pm = np.zeros((3, 3), dtype=np.float64)
        for i, j in enumerate(p):
            Pm[j, i] = 1.0
        for s in signs:
            Sm = np.diag(s)
            M = Pm @ Sm
            det = int(round(np.linalg.det(M)))
            if allow_reflection or det == 1:
                mats.append(M)

    uniq, seen = [], set()
    for M in mats:
        key = tuple(np.round(M.flatten(), 6))
        if key not in seen:
            seen.add(key)
            uniq.append(M)
    return uniq


def _normalize_points(P: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    P = np.asarray(P, dtype=np.float64)
    c = P.mean(axis=0)
    Q = P - c
    s = float(np.linalg.norm(Q.max(axis=0) - Q.min(axis=0)) + 1e-12)
    return (Q / s), c, s


def _score_alignment(P_tgt: np.ndarray, P_ref: np.ndarray) -> float:
    tree = cKDTree(P_ref)
    d, _ = tree.query(P_tgt, k=1, workers=-1)
    return float(np.sqrt((d * d).mean()))


def _best_rotation(P_tgt: np.ndarray, P_ref: np.ndarray, allow_reflection: bool) -> np.ndarray:
    U_t = _pca_basis(P_tgt)
    U_r = _pca_basis(P_ref)
    candidates = _generate_axis_rotations(allow_reflection=allow_reflection)

    best = (1e18, None)
    for Q in candidates:
        R = U_r @ Q @ U_t.T
        if (not allow_reflection) and np.linalg.det(R) < 0:
            continue
        err = _score_alignment(P_tgt @ R.T, P_ref)
        if err < best[0]:
            best = (err, R)

    if best[1] is None:
        raise RuntimeError("No valid rotation found.")
    return best[1]


def _load_ref_points(path: Path, sample_n: int, rng: np.random.Generator) -> np.ndarray:
    import trimesh
    m = trimesh.load(str(path), force="mesh", process=False)
    if isinstance(m, trimesh.Scene):
        m = trimesh.util.concatenate(tuple(m.geometry.values()))
    V = np.asarray(m.vertices, dtype=np.float64)
    if V.shape[0] > sample_n:
        idx = rng.choice(V.shape[0], size=sample_n, replace=False)
        V = V[idx]
    Pn, _, _ = _normalize_points(V)
    return Pn


def orient_mesh_to_reference(
    vertices: np.ndarray,
    *,
    arch: str,
    cfg: OrientConfig,
) -> tuple[np.ndarray, dict]:
    """
    Rotate vertices so it best matches reference (geometry-only PCA + 24 rotations + NN RMS).
    """
    arch = (arch or "").lower()
    if arch not in ("upper", "lower"):
        raise ValueError("arch must be upper/lower")

    rng = np.random.default_rng(int(cfg.seed))

    ref_path = cfg.ref_upper if arch == "upper" else cfg.ref_lower
    if not ref_path.exists():
        raise FileNotFoundError(f"Missing reference: {ref_path}")

    V = np.asarray(vertices, dtype=np.float64)
    if V.ndim != 2 or V.shape[1] != 3:
        raise ValueError(f"vertices must be (N,3), got {V.shape}")

    # sample target
    Vt = V
    if Vt.shape[0] > int(cfg.sample_n):
        idx = rng.choice(Vt.shape[0], size=int(cfg.sample_n), replace=False)
        Vt = Vt[idx]

    P_tgt, c_t, s_t = _normalize_points(Vt)
    P_ref = _load_ref_points(ref_path, int(cfg.sample_n), rng)

    R = _best_rotation(P_tgt, P_ref, allow_reflection=bool(cfg.allow_reflection))

    # apply to full vertices around centroid
    c = V.mean(axis=0)
    Vc = V - c
    Vn = (Vc @ R.T) + c

    meta = {
        "ref": str(ref_path),
        "allow_reflection": bool(cfg.allow_reflection),
    }
    return Vn.astype(np.float32, copy=False), meta