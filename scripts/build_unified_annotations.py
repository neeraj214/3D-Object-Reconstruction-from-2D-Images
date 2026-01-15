import argparse
import json
import random
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.evaluation.pix3d_gt import read_pix3d_annotations, ensure_gt_pointcloud


def build_pix3d_annotations(root: Path, out_path: Path, n_points: int = 4096, limit: int | None = None) -> None:
    anns = read_pix3d_annotations(root)
    items = []
    for a in anns:
        img = a.get("image_path")
        model = a.get("model_path")
        mask = a.get("mask_path")
        if not img or not model:
            continue
        img_p = Path(img)
        if not img_p.exists() or not Path(model).exists():
            continue
        cat = a.get("category", "unknown")
        pts, pcd_path = ensure_gt_pointcloud(root, cat, img_p.stem, Path(model), n_points=n_points)
        cam = {
            "K": a.get("camera_K"),
            "pose": a.get("pose"),
        }
        item = {
            "dataset": "pix3d",
            "image_front": str(img_p),
            "image_side": None,
            "category": cat,
            "point_cloud_gt": str(pcd_path),
            "mesh_gt": str(model),
            "mask": str(mask) if mask else None,
            "camera_params": cam,
        }
        items.append(item)
        if limit is not None and len(items) >= limit:
            break
    random.Random(42).shuffle(items)
    n = len(items)
    n_train = int(0.7 * n)
    n_val = int(0.15 * n)
    for i, it in enumerate(items):
        if i < n_train:
            it["split"] = "train"
        elif i < n_train + n_val:
            it["split"] = "val"
        else:
            it["split"] = "test"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(items, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str, choices=["pix3d"], default="pix3d")
    ap.add_argument("--root", type=str, default="data/pix3d")
    ap.add_argument("--out", type=str, default="data/pix3d/unified_multiview_annotations.json")
    ap.add_argument("--points", type=int, default=4096)
    ap.add_argument("--limit", type=int, default=500)
    args = ap.parse_args()
    root = Path(args.root)
    out = Path(args.out)
    if args.dataset == "pix3d":
        build_pix3d_annotations(root, out, n_points=args.points, limit=args.limit)


if __name__ == "__main__":
    main()
