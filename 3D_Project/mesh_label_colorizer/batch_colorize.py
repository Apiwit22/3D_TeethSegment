# batch_colorize.py
from pathlib import Path
import argparse

from json_obj_to_colored_ply import run_single


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels-root", required=True, type=Path)
    ap.add_argument("--meshes-root", required=True, type=Path)
    ap.add_argument("--out-root", required=True, type=Path)
    ap.add_argument("--colors", required=True, type=Path)
    args = ap.parse_args()

    json_files = sorted(args.labels_root.rglob("*.json"))
    ok = 0

    for jp in json_files:
        rel = jp.relative_to(args.labels_root)
        jaw = rel.parts[0]
        case = rel.parts[1]
        stem = jp.stem.replace("_16k", "")

        obj_path = args.meshes_root / jaw / case / f"{stem}_16k.obj"
        if not obj_path.exists():
            continue

        out_path = args.out_root / jaw / case / f"{stem}_16k_colored.ply"

        run_single(obj_path, jp, out_path, args.colors)
        ok += 1

    print(f"\nDone. ok={ok} scanned_json={len(json_files)}")


if __name__ == "__main__":
    main()
