# src/dataloader/collate.py
from __future__ import annotations

from typing import Dict, Any, List, Optional
import torch


def _stack_if_present(samples: List[Dict[str, Any]], key: str) -> Optional[torch.Tensor]:
    if not samples:
        return None
    for s in samples:
        if key not in s:
            return None
        v = s[key]
        if (v is None) or (not torch.is_tensor(v)):
            return None
    return torch.stack([s[key] for s in samples], dim=0)


def _ensure_dtype(out: Dict[str, Any]) -> Dict[str, Any]:
    # float tensors
    for k in ("x", "x_c", "x_n", "pos"):
        if k in out and torch.is_tensor(out[k]):
            out[k] = out[k].float()

    # labels
    if "y" in out and torch.is_tensor(out["y"]):
        out["y"] = out["y"].long()

    # bool masks
    for k in ("mask", "valid_face"):
        if k in out and torch.is_tensor(out[k]):
            out[k] = out[k].bool()

    # long index tensors
    for k in ("faces", "nbr"):
        if k in out and torch.is_tensor(out[k]):
            out[k] = out[k].long()

    # edge_index: Tensor(2,E) or list[Tensor(2,E)]
    if "edge_index" in out:
        if torch.is_tensor(out["edge_index"]):
            out["edge_index"] = out["edge_index"].long()
        elif isinstance(out["edge_index"], list):
            fixed: List[torch.Tensor] = []
            for i, ei in enumerate(out["edge_index"]):
                if not torch.is_tensor(ei):
                    raise TypeError(f"edge_index[{i}] must be torch.Tensor, got {type(ei)}")
                if ei.ndim != 2 or ei.shape[0] != 2:
                    raise ValueError(f"edge_index[{i}] must be shape (2,E), got {tuple(ei.shape)}")
                fixed.append(ei.long())
            out["edge_index"] = fixed
        else:
            raise TypeError(f"edge_index must be Tensor or list[Tensor], got {type(out['edge_index'])}")

    # IMPORTANT: keep F_used as list[int]
    if "F_used" in out and isinstance(out["F_used"], list):
        out["F_used"] = [int(x) for x in out["F_used"]]
        out["F_used_t"] = torch.tensor(out["F_used"], dtype=torch.long)

    return out


def collate_point(samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "x": torch.stack([s["x"] for s in samples], dim=0),
        "y": torch.stack([s["y"] for s in samples], dim=0),
        "mask": torch.stack([s["mask"] for s in samples], dim=0),
        "arch": [s["arch"] for s in samples],
        "path": [s["path"] for s in samples],
        "meta": [s.get("meta", {}) for s in samples],
    }
    pos = _stack_if_present(samples, "pos")
    if pos is not None:
        out["pos"] = pos
    return _ensure_dtype(out)


def collate_face(samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "x": torch.stack([s["x"] for s in samples], dim=0),
        "y": torch.stack([s["y"] for s in samples], dim=0),
        "mask": torch.stack([s["mask"] for s in samples], dim=0),
        "arch": [s["arch"] for s in samples],
        "path": [s["path"] for s in samples],
        "meta": [s.get("meta", {}) for s in samples],
        "F_used": [int(s.get("F_used", s["x"].shape[0])) for s in samples],
    }

    pos = _stack_if_present(samples, "pos")
    if pos is not None:
        out["pos"] = pos

    faces = _stack_if_present(samples, "faces")
    if faces is not None:
        out["faces"] = faces

    nbr = _stack_if_present(samples, "nbr")
    if nbr is not None:
        out["nbr"] = nbr

    x_c = _stack_if_present(samples, "x_c")
    if x_c is not None:
        out["x_c"] = x_c

    x_n = _stack_if_present(samples, "x_n")
    if x_n is not None:
        out["x_n"] = x_n

    valid_face = _stack_if_present(samples, "valid_face")
    if valid_face is not None:
        out["valid_face"] = valid_face

    return _ensure_dtype(out)


def _is_edge_index_tensor(v: Any) -> bool:
    return torch.is_tensor(v) and v.ndim == 2 and v.shape[0] == 2


def _is_edge_index_list(v: Any) -> bool:
    if not isinstance(v, list):
        return False
    return all(_is_edge_index_tensor(x) for x in v)


def collate_graph(samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Backward-compatible graph collate:
      - old graph datasets: require edge_index and keep it as list[Tensor]
      - new TSGCNet2 graph dataset: allow nbr without edge_index
    """
    out = collate_face(samples)
    if not samples:
        return out

    has_edge_index = []
    has_nbr = []

    for s in samples:
        ei = s.get("edge_index", None)
        nb = s.get("nbr", None)

        has_edge_index.append(_is_edge_index_tensor(ei) or _is_edge_index_list(ei))
        has_nbr.append(torch.is_tensor(nb))

    any_edge = any(has_edge_index)
    any_nbr = any(has_nbr)

    # case 1: old graph pipeline
    if any_edge:
        bad = [i for i, ok in enumerate(has_edge_index) if not ok]
        if bad:
            raise KeyError(
                f"Mixed graph batch is not allowed: some samples have edge_index, some do not. Bad indices={bad}"
            )
        out["edge_index"] = [s["edge_index"] for s in samples]
        return _ensure_dtype(out)

    # case 2: new TSGCNet2 pipeline
    if any_nbr:
        bad = [i for i, ok in enumerate(has_nbr) if not ok]
        if bad:
            raise KeyError(
                f"Graph collate expected nbr in every sample for TSGCNet2-style batch. Bad indices={bad}"
            )
        return _ensure_dtype(out)

    # case 3: invalid graph batch
    raise KeyError("Graph collate requires either edge_index or nbr in every sample.")