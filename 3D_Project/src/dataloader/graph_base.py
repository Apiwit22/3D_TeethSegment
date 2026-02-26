# src/dataloader/graph_base.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple
from pathlib import Path
import hashlib

import numpy as np
import torch

from src.dataloader.faces_base import FaceConfig
from src.dataloader.faces_twostream24 import TwoStream24FaceDataset

try:
    from scipy.spatial import cKDTree  # type: ignore
except Exception:
    cKDTree = None


@dataclass
class GraphConfig(FaceConfig):
    """
    Graph config extends FaceConfig:
      - graph_type: "face_adj" or "knn"
      - adjacency_mode: "edge" or "vertex"
      - nonmanifold_policy: "star" or "drop"
      - add_self_loops: bool
      - graph_k: for knn
      - knn_undirected: bool
      - cache_edge_index: bool
      - cache_dir: str
      - max_faces_per_edge/vertex: for non-manifold handling
    """
    graph_type: str = "face_adj"
    adjacency_mode: str = "edge"
    nonmanifold_policy: str = "star"
    add_self_loops: bool = False

    graph_k: int = 8
    knn_undirected: bool = True

    cache_edge_index: bool = True
    cache_dir: str = "cache/edge_index"

    max_faces_per_edge: int = 4
    max_faces_per_vertex: int = 16


def _unique_undirected_edges(edge_index: np.ndarray) -> np.ndarray:
    """
    edge_index: (2,E)
    return unique undirected edges (2,E2)
    """
    if edge_index.size == 0:
        return edge_index
    a = edge_index[0].astype(np.int64, copy=False)
    b = edge_index[1].astype(np.int64, copy=False)
    u = np.minimum(a, b)
    v = np.maximum(a, b)
    uv = np.stack([u, v], axis=0)  # (2,E)
    # unique columns
    uvT = uv.T
    uvU = np.unique(uvT, axis=0).T
    return uvU


def _add_self_loops(edge_index: np.ndarray, n: int) -> np.ndarray:
    if n <= 0:
        return edge_index
    loops = np.arange(n, dtype=np.int64)
    loops = np.stack([loops, loops], axis=0)  # (2,n)
    if edge_index.size == 0:
        return loops
    return np.concatenate([edge_index, loops], axis=1)


def _face_adj_edges_from_mesh(
    faces: np.ndarray,
    adjacency_mode: str = "edge",
    nonmanifold_policy: str = "star",
    max_faces_per_edge: int = 4,
    max_faces_per_vertex: int = 16,
) -> np.ndarray:
    """
    Build face adjacency edges (2,E) based on mesh topology.
    faces: (F,3)

    adjacency_mode:
      - "edge": neighbors share an edge (strongest, sparse)
      - "vertex": neighbors share a vertex (denser)

    nonmanifold_policy:
      - "star": connect all faces in the same incident set (clipped by max_faces_*)
      - "drop": ignore non-manifold edge/vertex that has too many incident faces
    """
    faces = np.asarray(faces, dtype=np.int64)
    F = int(faces.shape[0])
    if F <= 0:
        return np.zeros((2, 0), dtype=np.int64)

    adj_mode = str(adjacency_mode).lower()
    nm_policy = str(nonmanifold_policy).lower()

    edges = []

    if adj_mode == "edge":
        # build map: undirected edge -> incident faces
        edge_map: Dict[Tuple[int, int], List[int]] = {}
        for fi, (a, b, c) in enumerate(faces):
            e01 = (int(min(a, b)), int(max(a, b)))
            e12 = (int(min(b, c)), int(max(b, c)))
            e20 = (int(min(c, a)), int(max(c, a)))
            for e in (e01, e12, e20):
                edge_map.setdefault(e, []).append(fi)

        for _, inc in edge_map.items():
            if len(inc) == 2:
                i, j = inc
                edges.append((i, j))
                edges.append((j, i))
            elif len(inc) > 2:
                # non-manifold
                if nm_policy == "drop":
                    if len(inc) > int(max_faces_per_edge):
                        continue
                # "star": connect all pairs (clipped)
                inc2 = inc[: int(max_faces_per_edge)]
                for i in inc2:
                    for j in inc2:
                        if i != j:
                            edges.append((i, j))

    elif adj_mode == "vertex":
        # build map: vertex -> incident faces
        v_map: Dict[int, List[int]] = {}
        for fi, (a, b, c) in enumerate(faces):
            v_map.setdefault(int(a), []).append(fi)
            v_map.setdefault(int(b), []).append(fi)
            v_map.setdefault(int(c), []).append(fi)

        for _, inc in v_map.items():
            if len(inc) <= 1:
                continue
            if nm_policy == "drop":
                if len(inc) > int(max_faces_per_vertex):
                    continue
            inc2 = inc[: int(max_faces_per_vertex)]
            for i in inc2:
                for j in inc2:
                    if i != j:
                        edges.append((i, j))
    else:
        raise ValueError(f"Unknown adjacency_mode='{adjacency_mode}'. Use 'edge' or 'vertex'.")

    if len(edges) == 0:
        return np.zeros((2, 0), dtype=np.int64)

    ei = np.array(edges, dtype=np.int64).T  # (2,E)
    return ei


def _knn_edges(centers: np.ndarray, k: int, undirected: bool = True) -> np.ndarray:
    """
    Build kNN edges on face centers.
    centers: (F,3)
    return edge_index (2,E)
    """
    if cKDTree is None:
        raise ImportError("scipy is required for graph_type='knn'. Install: pip install scipy")

    centers = np.asarray(centers, dtype=np.float32)
    F = int(centers.shape[0])
    if F <= 0:
        return np.zeros((2, 0), dtype=np.int64)

    k = int(k)
    if k <= 0:
        return np.zeros((2, 0), dtype=np.int64)

    tree = cKDTree(centers)
    _, idx = tree.query(centers, k=min(k + 1, F))  # include self
    idx = idx[:, 1:]  # drop self -> (F,k)

    src = np.repeat(np.arange(F, dtype=np.int64), idx.shape[1])
    dst = idx.reshape(-1).astype(np.int64, copy=False)

    ei = np.stack([src, dst], axis=0)  # (2,E)

    if undirected:
        ei2 = np.stack([dst, src], axis=0)
        ei = np.concatenate([ei, ei2], axis=1)

    return ei


class GraphDataset(TwoStream24FaceDataset):
    """
    Graph dataset:
      - inherits TwoStream24FaceDataset to get x/x_c/x_n/y/mask/F_used/...
      - builds edge_index for each sample (based on topology adjacency or kNN)
      - collate_graph will keep edge_index as list[Tensor(2,E)] length B
    """

    @classmethod
    def from_config(cls, files: List[str], full_cfg: dict, arch: str):
        data = full_cfg["data"]
        num_classes = int(full_cfg["model"]["kwargs"].get("num_classes", 16))

        base = GraphConfig(
            mode=str(data.get("mode", "graph")),
            arch=arch,
            normalize=bool(data.get("normalize", True)),
            align_pca=bool(data.get("align_pca", False)),
            ignore_index=int(data.get("ignore_index", -1)),
            unknown_policy=str(data.get("unknown_policy", "raise")),
            require_labels=bool(data.get("require_labels", True)),
            num_classes=num_classes,

            # label policy
            label_source=str(data.get("label_source", "auto")),
            face_fallback=str(data.get("face_fallback", "majority")),

            # face sizing / feature
            num_faces=int(data.get("num_faces", 16000)),
            feature=str(data.get("face_feature", "twostream24")),
            return_faces=bool(data.get("return_faces", True)),
            return_nbr=bool(data.get("return_nbr", False)),
            k_neighbors=int(data.get("k_neighbors", 3)),

            # graph configs
            graph_type=str(data.get("graph_type", "face_adj")),
            adjacency_mode=str(data.get("adjacency_mode", "edge")),
            nonmanifold_policy=str(data.get("nonmanifold_policy", "star")),
            add_self_loops=bool(data.get("add_self_loops", False)),
            graph_k=int(data.get("graph_k", 8)),
            knn_undirected=bool(data.get("knn_undirected", True)),
            cache_edge_index=bool(data.get("cache_edge_index", True)),
            cache_dir=str(data.get("cache_dir", "cache/edge_index")),
            max_faces_per_edge=int(data.get("max_faces_per_edge", 4)),
            max_faces_per_vertex=int(data.get("max_faces_per_vertex", 16)),
        )
        return cls(files, base)

    def _edge_cache_path(self, ply_path: str, F_used: int) -> Path:
        p = Path(ply_path)
        # NOTE: cache key MUST include full path (and size/mtime) to avoid collisions across datasets/folds
        # where many files may share the same stem (e.g., 0001_U.ply).
        try:
            st = p.stat()
            f_size = int(st.st_size)
            f_mtime = int(st.st_mtime)
        except OSError:
            f_size = -1
            f_mtime = -1
        key = {
            "path": str(p.resolve()),
            "size": f_size,
            "mtime": f_mtime,
            "stem": p.stem,
            "F": int(F_used),
            "gt": str(self.gcfg.graph_type).lower(),
            "am": str(self.gcfg.adjacency_mode).lower(),
            "nm": str(self.gcfg.nonmanifold_policy).lower(),
            "mfe": int(self.gcfg.max_faces_per_edge),
            "mfv": int(self.gcfg.max_faces_per_vertex),
            "sl": bool(self.gcfg.add_self_loops),
            "k": int(self.gcfg.graph_k),
            "und": bool(self.gcfg.knn_undirected),
        }
        s = repr(sorted(key.items())).encode("utf-8")
        h = hashlib.sha1(s).hexdigest()[:12]
        cache_dir = Path(self.gcfg.cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir / f"{p.stem}_{h}.npz"

    @property
    def gcfg(self) -> GraphConfig:
        return self.cfg  # type: ignore[return-value]

    def _build_edge_index(self, faces_used: np.ndarray, centers_used: np.ndarray) -> np.ndarray:
        gt = str(self.gcfg.graph_type).lower()
        if gt == "face_adj":
            ei = _face_adj_edges_from_mesh(
                faces_used,
                adjacency_mode=str(self.gcfg.adjacency_mode),
                nonmanifold_policy=str(self.gcfg.nonmanifold_policy),
                max_faces_per_edge=int(self.gcfg.max_faces_per_edge),
                max_faces_per_vertex=int(self.gcfg.max_faces_per_vertex),
            )
        elif gt == "knn":
            ei = _knn_edges(
                centers_used,
                k=int(self.gcfg.graph_k),
                undirected=bool(self.gcfg.knn_undirected),
            )
        else:
            raise ValueError(f"Unknown graph_type='{self.gcfg.graph_type}'. Use 'face_adj' or 'knn'.")

        # unique undirected (optional; keep directed for message passing)
        if bool(self.gcfg.knn_undirected):
            # if undirected, we can still keep both directions but remove duplicates
            ei = _unique_undirected_edges(ei)
            # expand to directed
            if ei.size > 0:
                a = ei[0]
                b = ei[1]
                ei = np.concatenate([np.stack([a, b], axis=0), np.stack([b, a], axis=0)], axis=1)

        if bool(self.gcfg.add_self_loops):
            ei = _add_self_loops(ei, int(centers_used.shape[0]))

        return ei.astype(np.int64, copy=False)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        out = super().__getitem__(idx)

        # build graph edges only on valid (non-pad) faces
        F_used = int(out["F_used"])
        if "faces" not in out:
            raise KeyError("GraphDataset expects 'faces' in output. Set data.return_faces=true.")
        faces = out["faces"].numpy().astype(np.int64, copy=False)  # (Fmax,3)

        # centers from x_c:
        # x_c layout depends on coord_mode:
        # - relative: first 3 dims = center
        # - absolute: last 3 dims = center
        x_c = out["x_c"].numpy().astype(np.float32, copy=False)
        coord_mode = str(out.get("meta", {}).get("twostream_coord_mode", "relative")).lower()
        if coord_mode in ("rel", "relative"):
            centers = x_c[:, 0:3]
        else:
            centers = x_c[:, 9:12]

        faces_used = faces[:F_used]
        centers_used = centers[:F_used]

        # caching
        if bool(self.gcfg.cache_edge_index):
            cache_path = self._edge_cache_path(out["path"], F_used)
            if cache_path.exists():
                try:
                    npz = np.load(str(cache_path))
                    ei = npz["edge_index"].astype(np.int64, copy=False)
                except Exception:
                    ei = self._build_edge_index(faces_used, centers_used)
                    np.savez_compressed(str(cache_path), edge_index=ei)
            else:
                ei = self._build_edge_index(faces_used, centers_used)
                np.savez_compressed(str(cache_path), edge_index=ei)
        else:
            ei = self._build_edge_index(faces_used, centers_used)

        out["edge_index"] = torch.from_numpy(ei)  # (2,E)
        return out