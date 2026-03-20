# src/dataloader/graph_base.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import hashlib

import numpy as np
import torch

try:
    from scipy.spatial import cKDTree  # type: ignore
except Exception:
    cKDTree = None

from src.dataloader.faces_twostream24 import TwoStream24FaceDataset
from src.dataloader.faces_base import FaceConfig


def _unique_edge_index(edges: List[Tuple[int, int]]) -> np.ndarray:
    """
    Convert python list[(src,dst)] -> unique edge_index (2,E) as int64.
    This enforces **binary adjacency** (paper uses binary A).
    """
    if not edges:
        return np.zeros((2, 0), dtype=np.int64)
    arr = np.array(edges, dtype=np.int64)  # (E,2)
    # unique rows
    arr = np.unique(arr, axis=0)
    return arr.T  # (2,E)


def _add_self_loops_np(edge_index: np.ndarray, num_nodes: int) -> np.ndarray:
    if num_nodes <= 0:
        return edge_index
    loops = np.arange(num_nodes, dtype=np.int64)
    loops = np.stack([loops, loops], axis=0)  # (2,N)
    if edge_index.size == 0:
        return loops
    return np.concatenate([edge_index, loops], axis=1)


def _face_adj_edges_from_mesh(
    faces: np.ndarray,
    *,
    adjacency_mode: str = "vertex",
    nonmanifold_policy: str = "star",
    max_faces_per_edge: int = 0,
    max_faces_per_vertex: int = 0,
    add_self_loops: bool = False,
) -> np.ndarray:
    """
    Build face adjacency edges.

    Paper-like (Fast-TGCN):
      - adjacency_mode='vertex': two faces are adjacent if they share at least 1 vertex.
      - adjacency is **binary**.

    nonmanifold_policy:
      - 'drop': if incident list is too large -> drop that structure
      - 'star': connect all pairs (optionally clipped)

    max_faces_per_*:
      - <=0 means "no clip"
      - >0 means clip incident list to that length
    """
    faces = np.asarray(faces, dtype=np.int64)
    F = int(faces.shape[0])
    if F <= 0:
        return np.zeros((2, 0), dtype=np.int64)

    adj_mode = str(adjacency_mode).lower().strip()
    nm_policy = str(nonmanifold_policy).lower().strip()

    edges: List[Tuple[int, int]] = []

    if adj_mode == "edge":
        # map undirected edge -> incident faces
        e_map: Dict[Tuple[int, int], List[int]] = {}
        for fi, (a, b, c) in enumerate(faces):
            tri = (int(a), int(b), int(c))
            for u, v in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
                if u <= v:
                    key = (u, v)
                else:
                    key = (v, u)
                e_map.setdefault(key, []).append(fi)

        for inc in e_map.values():
            if len(inc) <= 1:
                continue
            if len(inc) == 2:
                i, j = inc[0], inc[1]
                edges.append((i, j))
                edges.append((j, i))
            else:
                # non-manifold edge
                if nm_policy == "drop" and max_faces_per_edge > 0 and len(inc) > int(max_faces_per_edge):
                    continue
                if max_faces_per_edge > 0:
                    inc = inc[: int(max_faces_per_edge)]
                # connect all ordered pairs i!=j
                for i in inc:
                    for j in inc:
                        if i != j:
                            edges.append((i, j))

    elif adj_mode == "vertex":
        # map vertex -> incident faces
        v_map: Dict[int, List[int]] = {}
        for fi, (a, b, c) in enumerate(faces):
            v_map.setdefault(int(a), []).append(fi)
            v_map.setdefault(int(b), []).append(fi)
            v_map.setdefault(int(c), []).append(fi)

        for inc in v_map.values():
            if len(inc) <= 1:
                continue

            # non-manifold vertex handling
            if nm_policy == "drop" and max_faces_per_vertex > 0 and len(inc) > int(max_faces_per_vertex):
                continue

            # clip only if user asks (>0)
            if max_faces_per_vertex > 0:
                inc = inc[: int(max_faces_per_vertex)]

            # connect all ordered pairs i!=j (share-vertex adjacency)
            for i in inc:
                for j in inc:
                    if i != j:
                        edges.append((i, j))
    else:
        raise ValueError(f"Unknown adjacency_mode='{adjacency_mode}'. Use 'edge' or 'vertex'.")

    ei = _unique_edge_index(edges)  # enforce binary adjacency
    if add_self_loops:
        ei = _add_self_loops_np(ei, F)
        # make unique again (in case some loops existed)
        ei = np.unique(ei.T, axis=0).T if ei.size else ei

    return ei.astype(np.int64, copy=False)


def _knn_edges(centers: np.ndarray, k: int, undirected: bool = True) -> np.ndarray:
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

    # unique for safety
    ei = np.unique(ei.T, axis=0).T if ei.size else ei
    return ei.astype(np.int64, copy=False)


class GraphDataset(TwoStream24FaceDataset):
    """
    Graph dataset:
      - inherits TwoStream24FaceDataset (x/x_c/x_n/y/mask/F_used/faces/pos)
      - builds edge_index (2,E) per sample
      - collate_graph keeps edge_index as list[Tensor] length B
    """

    def __init__(self, files: List[str], cfg: FaceConfig):
        super().__init__(files, cfg)
        self.gcfg = cfg

        self.graph_type = str(getattr(cfg, "graph_type", "face_adj")).lower().strip()
        self.adjacency_mode = str(getattr(cfg, "adjacency_mode", "vertex")).lower().strip()
        self.nonmanifold_policy = str(getattr(cfg, "nonmanifold_policy", "star")).lower().strip()

        # IMPORTANT: paper uses binary adjacency matrix A (includes i=j logically);
        # we allow adding self-loops here to match A's diagonal.
        self.add_self_loops = bool(getattr(cfg, "add_self_loops", False))

        self.cache_edge_index = bool(getattr(cfg, "cache_edge_index", True))
        self.cache_dir = Path(str(getattr(cfg, "cache_dir", "cache/edge_index")))
        self.max_faces_per_edge = int(getattr(cfg, "max_faces_per_edge", 0))
        self.max_faces_per_vertex = int(getattr(cfg, "max_faces_per_vertex", 0))

        # for knn only
        self.k_neighbors = int(getattr(cfg, "k_neighbors", 3))

        if self.cache_edge_index:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_config(cls, files: List[str], full_cfg: dict, arch: str):
        # Reuse TwoStream24FaceDataset config creation, but force mode="graph"
        data = full_cfg["data"]
        num_classes = int(full_cfg["model"]["kwargs"].get("num_classes", 16))

        feat = str(data.get("face_feature", "paper24")).lower().strip()
        default_coord = "absolute" if feat == "paper24" else "relative"

        base = FaceConfig(
            mode="graph",
            arch=arch,
            normalize=bool(data.get("normalize", True)),
            align_pca=bool(data.get("align_pca", False)),
            ignore_index=int(data.get("ignore_index", -1)),
            unknown_policy=str(data.get("unknown_policy", "raise")),
            require_labels=bool(data.get("require_labels", True)),
            num_classes=num_classes,
            label_source=str(data.get("label_source", "auto")),
            face_fallback=str(data.get("face_fallback", "majority")),
            num_faces=int(data.get("num_faces", 16000)),
            feature=str(data.get("face_feature", "paper24")),
            return_faces=True,  # must return faces to build topology adjacency
            return_nbr=bool(data.get("return_nbr", False)),
            k_neighbors=int(data.get("k_neighbors", 3)),
        )

        setattr(base, "twostream_coord_mode", str(data.get("twostream_coord_mode", default_coord)).lower().strip())

        # augmentation flags (same as TwoStream24FaceDataset)
        setattr(base, "is_train", bool(data.get("is_train", False)))
        setattr(base, "augment", bool(data.get("augment", False)))
        setattr(base, "aug_translate", bool(data.get("aug_translate", True)))
        setattr(base, "aug_rotate_z", bool(data.get("aug_rotate_z", True)))
        setattr(base, "aug_translate_ranges", data.get("aug_translate_ranges", [[-6, 6], [-8, 8], [-5, 5]]))
        setattr(base, "aug_rotate_z_range", data.get("aug_rotate_z_range", [-float(np.pi) / 10.0, float(np.pi) / 10.0]))

        # graph params
        setattr(base, "graph_type", str(data.get("graph_type", "face_adj")).lower().strip())
        setattr(base, "adjacency_mode", str(data.get("adjacency_mode", "vertex")).lower().strip())
        setattr(base, "nonmanifold_policy", str(data.get("nonmanifold_policy", "star")).lower().strip())
        setattr(base, "add_self_loops", bool(data.get("add_self_loops", True)))

        setattr(base, "cache_edge_index", bool(data.get("cache_edge_index", True)))
        setattr(base, "cache_dir", str(data.get("cache_dir", "cache/edge_index")))
        setattr(base, "max_faces_per_edge", int(data.get("max_faces_per_edge", 0)))
        setattr(base, "max_faces_per_vertex", int(data.get("max_faces_per_vertex", 0)))

        return cls(files, base)

    def _cache_path(self, mesh_path: str, F_used: int) -> Path:
        key = (
            f"{mesh_path}|F={F_used}|"
            f"{self.graph_type}|{self.adjacency_mode}|{self.nonmanifold_policy}|"
            f"loop={int(self.add_self_loops)}|"
            f"mfe={int(self.max_faces_per_edge)}|mfv={int(self.max_faces_per_vertex)}"
        )
        h = hashlib.md5(key.encode("utf-8")).hexdigest()
        stem = Path(mesh_path).stem
        return self.cache_dir / f"{stem}_{h}.npz"

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        out = super().__getitem__(idx)

        faces_t = out.get("faces", None)
        if faces_t is None:
            raise ValueError("GraphDataset requires out['faces'] (set return_faces=True).")

        F_used = int(out.get("F_used", int(faces_t.shape[0])))
        if F_used <= 0:
            out["edge_index"] = torch.zeros((2, 0), dtype=torch.long)
            return out

        # build / load cache
        cache_path = self._cache_path(out["path"], F_used)
        if self.cache_edge_index and cache_path.exists():
            data = np.load(str(cache_path))
            ei_np = data["edge_index"].astype(np.int64, copy=False)
        else:
            faces_np = faces_t[:F_used].detach().cpu().numpy().astype(np.int64, copy=False)

            if self.graph_type == "face_adj":
                ei_np = _face_adj_edges_from_mesh(
                    faces_np,
                    adjacency_mode=self.adjacency_mode,
                    nonmanifold_policy=self.nonmanifold_policy,
                    max_faces_per_edge=self.max_faces_per_edge,
                    max_faces_per_vertex=self.max_faces_per_vertex,
                    add_self_loops=self.add_self_loops,
                )
            elif self.graph_type == "knn":
                centers = out["pos"][:F_used].detach().cpu().numpy().astype(np.float32, copy=False)
                ei_np = _knn_edges(centers, k=self.k_neighbors, undirected=True)
                if self.add_self_loops:
                    ei_np = _add_self_loops_np(ei_np, F_used)
                    ei_np = np.unique(ei_np.T, axis=0).T if ei_np.size else ei_np
            else:
                raise ValueError(f"Unknown graph_type='{self.graph_type}'. Use 'face_adj' or 'knn'.")

            if self.cache_edge_index:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(str(cache_path), edge_index=ei_np)

        out["edge_index"] = torch.from_numpy(ei_np).long()
        out["meta"]["graph_type"] = self.graph_type
        out["meta"]["adjacency_mode"] = self.adjacency_mode
        out["meta"]["add_self_loops_graph"] = bool(self.add_self_loops)
        out["meta"]["edge_count"] = int(out["edge_index"].shape[1])
        return out