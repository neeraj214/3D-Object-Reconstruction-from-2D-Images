import json
import random
from pathlib import Path
import numpy as np
from PIL import Image
import argparse
o3d = None
try:
    import open3d as _o3d
    o3d = _o3d
except Exception:
    o3d = None

def _ensure_dir(p):
    Path(p).mkdir(parents=True, exist_ok=True)

def _sample_points_from_mesh(mesh_path, n):
    if o3d is None:
        return None
    m = o3d.io.read_triangle_mesh(str(mesh_path))
    m.compute_vertex_normals()
    pcd = m.sample_points_poisson_disk(n)
    pts = np.asarray(pcd.points)
    c = pts.mean(axis=0)
    pts = pts - c
    s = np.linalg.norm(pts, axis=1).max()
    pts = pts / max(s, 1e-8)
    return pts

def _default_camera_params(w, h):
    f = float(max(w, h))
    K = [[f, 0.0, w/2.0], [0.0, f, h/2.0], [0.0, 0.0, 1.0]]
    return {"K": K, "extrinsics": [[1,0,0,0],[0,1,0,0],[0,0,1,0]]}

def build_pix3d_entries(pix_root: Path, out_pcd: Path, n_points: int, limit: int = 200):
    meta = pix_root/"pix3d.json"
    if not meta.exists():
        return []
    data = json.loads(meta.read_text())
    entries = []
    for item in data:
        if "img" in item and "model" in item and "category" in item:
            img_path = pix_root/item["img"]
            mesh_path = pix_root/item["model"]
            mask_path = pix_root/item.get("mask", "")
            mask_path = mask_path if isinstance(mask_path, Path) else (pix_root/mask_path if mask_path else None)
            if img_path.exists() and mesh_path.exists():
                pts = _sample_points_from_mesh(mesh_path, n_points)
                base = Path(img_path).stem
                out_file = out_pcd/f"{base}.npy"
                _ensure_dir(out_pcd)
                if pts is not None:
                    np.save(out_file, pts)
                entry = {
                    "image_front": str(img_path),
                    "image_side": str(img_path),
                    "category": item["category"],
                    "point_cloud_gt": str(out_file),
                    "mesh_gt": str(mesh_path),
                    "camera_params": _default_camera_params(640, 480)
                }
                if mask_path and Path(mask_path).exists():
                    entry["mask"] = str(mask_path)
                entries.append(entry)
                if limit and len(entries) >= limit:
                    break
    return entries

def split_entries(entries, train_ratio=0.7, val_ratio=0.15):
    random.shuffle(entries)
    n = len(entries)
    n_train = int(train_ratio*n)
    n_val = int(val_ratio*n)
    train = entries[:n_train]
    val = entries[n_train:n_train+n_val]
    test = entries[n_train+n_val:]
    return {"train": train, "val": val, "test": test}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--synthetic", type=int, default=50)
    args = ap.parse_args()

    root = Path("data")
    out_root = root/"unified"
    _ensure_dir(out_root)
    out_ann = out_root/"annotations.json"
    out_pcd = out_root/"pointclouds"
    _ensure_dir(out_pcd)
    masks_dir = out_root/"masks"
    _ensure_dir(masks_dir)
    meshes_dir = out_root/"meshes"
    _ensure_dir(meshes_dir)
    entries = []
    pix = root/"pix3d"
    if pix.exists():
        entries += build_pix3d_entries(pix, out_pcd, 2048, limit=args.limit)
    if len(entries) == 0:
        img_dir = out_root/"images"
        _ensure_dir(img_dir)
        _ensure_dir(out_pcd)
        for i in range(max(1, args.synthetic)):
            w, h = 256, 256
            arr = np.zeros((h,w,3), dtype=np.uint8)
            for y in range(h):
                arr[y,:,0] = int(255*y/h)
                arr[y,:,1] = int(255*(1 - y/h))
                arr[y,:,2] = 128
            img_f = img_dir/f"front_{i}.jpg"; img_s = img_dir/f"side_{i}.jpg"
            Image.fromarray(arr).save(img_f)
            Image.fromarray(np.roll(arr, shift=32, axis=1)).save(img_s)
            m_arr = np.zeros((h,w), dtype=np.uint8)
            cy, cx, r = h//2, w//2, min(h,w)//3
            yy, xx = np.ogrid[:h, :w]
            mask_circle = (yy-cy)**2 + (xx-cx)**2 <= r*r
            m_arr[mask_circle] = 255
            mask_path = masks_dir/f"synthetic_{i}.png"
            Image.fromarray(m_arr).save(mask_path)
            cube_v = np.array([
                [-0.5,-0.5,-0.5], [0.5,-0.5,-0.5], [0.5,0.5,-0.5], [-0.5,0.5,-0.5],
                [-0.5,-0.5, 0.5], [0.5,-0.5, 0.5], [0.5,0.5, 0.5], [-0.5,0.5, 0.5]
            ], dtype=np.float32)
            cube_f = np.array([
                [0,1,2],[0,2,3], [4,5,6],[4,6,7], [0,1,5],[0,5,4], [2,3,7],[2,7,6], [1,2,6],[1,6,5], [0,3,7],[0,7,4]
            ], dtype=np.int32)
            mesh_path = meshes_dir/f"synthetic_{i}.obj"
            with open(mesh_path, 'w') as fobj:
                for v in cube_v:
                    fobj.write(f"v {v[0]} {v[1]} {v[2]}\n")
                for tri in cube_f:
                    fobj.write(f"f {tri[0]+1} {tri[1]+1} {tri[2]+1}\n")
            # Sample points on cube surface
            rng = np.random.default_rng(42+i)
            tris = cube_v[cube_f]
            v0 = tris[:,0,:]; v1 = tris[:,1,:]; v2 = tris[:,2,:]
            u = rng.random(2048)
            v = rng.random(2048)
            mask = u+v > 1
            u[mask] = 1 - u[mask]
            v[mask] = 1 - v[mask]
            tri_idx = rng.integers(0, len(tris), size=2048)
            p = v0[tri_idx] + u[:,None]*(v1[tri_idx]-v0[tri_idx]) + v[:,None]*(v2[tri_idx]-v0[tri_idx])
            p = p - p.mean(axis=0)
            p = p / max(np.linalg.norm(p, axis=1).max(), 1e-8)
            np.save(out_pcd/f"synthetic_{i}.npy", p.astype(np.float32))
            entries.append({
                "image_front": str(img_f),
                "image_side": str(img_s),
                "category": "synthetic",
                "point_cloud_gt": str(out_pcd/f"synthetic_{i}.npy"),
                "mesh_gt": str(mesh_path),
                "mask": str(mask_path),
                "camera_params": _default_camera_params(w, h)
            })
    splits = split_entries(entries)
    payload = {"counts": {k: len(v) for k,v in splits.items()}, "items": splits}
    out_ann.write_text(json.dumps(payload, indent=2))
    print(str(out_ann))
    print(json.dumps(payload["counts"]))

if __name__ == "__main__":
    main()