# scripts/map_labels_to_remesh.py
import argparse
import json
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial import cKDTree

def load_labels(json_path: Path):
    data = json.loads(json_path.read_text(encoding="utf-8"))
    if "labels" not in data:
        raise KeyError(f"No 'labels' key in {json_path}")
    return np.asarray(data["labels"], dtype=np.int32)

def map_labels(orig_mesh, orig_labels, remesh):
    tree = cKDTree(orig_mesh.vertices)
    _, idx = tree.query(remesh.vertices, k=1)
    return orig_labels[idx]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jaw", required=True, choices=["lower", "upper"])
    args = ap.parse_args()

    RAW_PARENT = Path("data/raw")
    REMESH_PARENT = Path("data/remeshed")
    OUT_PARENT = Path("data/labels_remap")

    subsets = [p for p in sorted(RAW_PARENT.iterdir()) if p.is_dir()]
    if not subsets:
        raise SystemExit(f"[ER] no subset under {RAW_PARENT}")

    for subset in subsets:
        raw_jaw = subset / args.jaw
        remesh_jaw = REMESH_PARENT / subset.name / args.jaw
        out_jaw = OUT_PARENT / subset.name / args.jaw

        if not remesh_jaw.exists():
            print(f"[SKIP] missing remesh jaw: {remesh_jaw}")
            continue
        if not raw_jaw.exists():
            print(f"[SKIP] missing raw jaw: {raw_jaw}")
            continue

        for case_dir in remesh_jaw.iterdir():
            if not case_dir.is_dir():
                continue

            for remesh_obj in case_dir.glob("*_16k.obj"):
                stem = remesh_obj.stem.replace("_16k", "")

                orig_obj = raw_jaw / case_dir.name / f"{stem}.obj"
                orig_json = raw_jaw / case_dir.name / f"{stem}.json"

                if not orig_obj.exists() or not orig_json.exists():
                    # บางทีนามสกุล OBJ ตัวใหญ่
                    orig_obj2 = raw_jaw / case_dir.name / f"{stem}.OBJ"
                    if orig_obj2.exists():
                        orig_obj = orig_obj2
                    else:
                        print(f"[SKIP] missing raw for {subset.name}/{args.jaw}/{case_dir.name}/{stem}")
                        continue

                orig_mesh = trimesh.load(orig_obj, force="mesh", process=False)
                remesh = trimesh.load(remesh_obj, force="mesh", process=False)
                labels = load_labels(orig_json)

                if len(labels) != len(orig_mesh.vertices):
                    raise RuntimeError(f"Vertex mismatch: {orig_obj} labels={len(labels)} verts={len(orig_mesh.vertices)}")

                new_labels = map_labels(orig_mesh, labels, remesh)

                out_dir = out_jaw / case_dir.name
                out_dir.mkdir(parents=True, exist_ok=True)
                out_json = out_dir / f"{stem}_16k.json"

                json.dump({"labels": new_labels.tolist()}, open(out_json, "w", encoding="utf-8"), indent=2)
                print(f"[OK] labels mapped -> {out_json}")

if __name__ == "__main__":
    main()
