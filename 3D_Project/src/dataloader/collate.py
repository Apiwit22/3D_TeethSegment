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

    # ✅ IMPORTANT: keep F_used as list[int] for FastTGCN
    # (Do NOT convert to tensor here.)
    # If you want tensor too, add extra field:
    if "F_used" in out and isinstance(out["F_used"], list):
        # sanitize to python ints
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
        "F_used": [int(s.get("F_used", s["x"].shape[0])) for s in samples],  # keep list[int]
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

    # Fast-TGCN optional branches
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
    out = collate_face(samples)
    if not samples:
        return out

    # edge_index must exist per-sample for graph mode
    bad = []
    for i, s in enumerate(samples):
        if "edge_index" not in s:
            bad.append(i)
            continue
        ei = s["edge_index"]
        if not (_is_edge_index_tensor(ei) or _is_edge_index_list(ei)):
            bad.append(i)

    if bad:
        raise KeyError(f"Graph collate requires edge_index in every sample. Bad indices={bad}")

    # keep per-sample graph (Fast-TGCN style)
    out["edge_index"] = [s["edge_index"] for s in samples]  # list[(2,E)]
    return _ensure_dtype(out)
