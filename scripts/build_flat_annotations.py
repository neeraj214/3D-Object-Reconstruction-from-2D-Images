import json
from pathlib import Path
import argparse
import numpy as np
from PIL import Image

def default_camera_params(w, h):
    f = float(max(w, h))
    return {"K": [[f, 0.0, w/2.0], [0.0, f, h/2.0], [0.0, 0.0, 1.0]]}

def build_from_pix3d(pix_root: Path, limit: int):
    meta = pix_root / "pix3d.json"
    if not meta.exists():
        return []
    data = json.loads(meta.read_text())
    items = []
    for it in data:
        if "img" in it and "category" in it:
            img_path = pix_root / it["img"]
            if not img_path.exists():
                continue
            try:
                with Image.open(img_path) as im:
                    w, h = im.size
            except Exception:
                w, h = 640, 480
            entry = {
                "image_front": str(img_path),
                "image_side": str(img_path),
                "category": it["category"],
                "point_cloud_gt": "",
                "mesh_gt": "",
                "mask": str(pix_root / it["mask"]) if it.get("mask") else "",
                "camera_params": default_camera_params(w, h)
            }
            items.append(entry)
            if limit and len(items) >= limit:
                break
    return items

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=200)
    args = ap.parse_args()
    root = Path("data")
    out_dir = root / "unified"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "annotations.json"
    pix_root = root / "pix3d"
    items = build_from_pix3d(pix_root, args.limit)
    out_file.write_text(json.dumps(items, indent=2))
    print(str(out_file))
    print(len(items))

if __name__ == "__main__":
    main()
