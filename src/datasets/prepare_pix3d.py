import os
import json
import argparse
import random
from pathlib import Path
import numpy as np
import open3d as o3d

def _ensure_dir(p):
    Path(p).mkdir(parents=True, exist_ok=True)

def _sample_points_from_mesh(mesh_path, n=8192):
    m = o3d.io.read_triangle_mesh(str(mesh_path))
    m.compute_vertex_normals()
    pcd = m.sample_points_poisson_disk(n)
    pts = np.asarray(pcd.points)
    center = pts.mean(axis=0)
    pts = pts - center
    scale = np.linalg.norm(pts, axis=1).max()
    pts = pts / max(scale, 1e-8)
    return pts

def run(pix3d_root, out_root):
    _ensure_dir(out_root)
    rgb_out = Path(out_root) / 'rgb'
    mask_out = Path(out_root) / 'masks'
    mesh_out = Path(out_root) / 'meshes'
    pcd_out = Path(out_root) / 'pcd'
    for d in [rgb_out, mask_out, mesh_out, pcd_out]:
        _ensure_dir(d)

    meta_path = Path(pix3d_root) / 'pix3d.json'
    if not meta_path.exists():
        return
    with open(meta_path) as f:
        meta = json.load(f)

    entries = []
    for item in meta:
        if 'img' in item and 'mask' in item and 'model' in item:
            rgb_src = Path(pix3d_root) / item['img']
            mask_src = Path(pix3d_root) / item['mask']
            mesh_src = Path(pix3d_root) / item['model']
            if rgb_src.exists() and mask_src.exists() and mesh_src.exists():
                entries.append((rgb_src, mask_src, mesh_src))

    random.shuffle(entries)
    n = len(entries)
    n_train = int(0.7 * n)
    n_val = int(0.2 * n)
    splits = {'train': entries[:n_train], 'val': entries[n_train:n_train+n_val], 'test': entries[n_train+n_val:]}

    split_file = Path(out_root) / 'splits.json'
    with open(split_file, 'w') as f:
        json.dump({'counts': {k: len(v) for k, v in splits.items()}}, f)

    for split, items in splits.items():
        for i, (rgb_src, mask_src, mesh_src) in enumerate(items):
            rgb_dst = rgb_out / f'{split}_{i}.png'
            mask_dst = mask_out / f'{split}_{i}.png'
            mesh_dst = mesh_out / f'{split}_{i}.obj'
            try:
                Path(rgb_dst).parent.mkdir(parents=True, exist_ok=True)
                Path(mask_dst).parent.mkdir(parents=True, exist_ok=True)
                Path(mesh_dst).parent.mkdir(parents=True, exist_ok=True)
                os.replace(str(rgb_src), str(rgb_dst))
                os.replace(str(mask_src), str(mask_dst))
                os.replace(str(mesh_src), str(mesh_dst))
                pts = _sample_points_from_mesh(mesh_dst, 8192)
                np.save(pcd_out / f'{split}_{i}.npy', pts)
            except Exception:
                pass

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pix3d_root', type=str, required=True)
    ap.add_argument('--out', type=str, required=True)
    args = ap.parse_args()
    run(args.pix3d_root, args.out)

if __name__ == '__main__':
    main()