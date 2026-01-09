import json
from pathlib import Path
import numpy as np
import trimesh

def read_pix3d_annotations(root: Path):
    ann = root/'pix3d.json'
    if not ann.exists():
        return []
    with open(ann, 'r') as f:
        data = json.load(f)
    samples = []
    for item in data:
        img = root/item.get('img','')
        model = root/item.get('model','') if 'model' in item else None
        mask = root/item.get('mask','') if 'mask' in item else None
        camera_K = item.get('camera_K')
        pose = item.get('pose')
        cat = item.get('category','unknown')
        samples.append({'image_path': str(img), 'model_path': str(model) if model else None, 'mask_path': str(mask) if mask else None, 'camera_K': camera_K, 'pose': pose, 'category': cat})
    return samples

def sample_mesh_to_pointcloud(mesh_path: Path, n_points: int = 20000):
    m = trimesh.load(str(mesh_path), force='mesh')
    pts = m.sample(n_points)
    return pts

def ensure_gt_pointcloud(root: Path, category: str, image_stem: str, model_path: Path, n_points: int = 20000):
    out_dir = root/'pointclouds'/category
    out_dir.mkdir(parents=True, exist_ok=True)
    out_npy = out_dir/f'{image_stem}.npy'
    if out_npy.exists():
        try:
            arr = np.load(str(out_npy))
            if arr.ndim == 2 and arr.shape[1] == 3:
                return arr, out_npy
        except Exception:
            pass
    pts = sample_mesh_to_pointcloud(model_path, n_points)
    np.save(str(out_npy), pts)
    return pts, out_npy

def render_depth_from_points(points: np.ndarray, camera_K, pose, image_size):
    h, w = image_size
    fx = camera_K[0][0] if camera_K else None
    fy = camera_K[1][1] if camera_K else None
    cx = camera_K[0][2] if camera_K else w/2
    cy = camera_K[1][2] if camera_K else h/2
    if fx is None or fy is None:
        fx = max(w,h); fy = max(w,h)
    R = np.eye(3)
    t = np.zeros((3,))
    if pose and 'R' in pose and 't' in pose:
        R = np.array(pose['R']); t = np.array(pose['t']).reshape(3)
    pc = (R @ points.T).T + t
    z = pc[:,2]
    x = (pc[:,0] * fx / (z + 1e-8)) + cx
    y = (pc[:,1] * fy / (z + 1e-8)) + cy
    x = np.clip(x, 0, w-1).astype(np.int32)
    y = np.clip(y, 0, h-1).astype(np.int32)
    depth = np.full((h,w), np.inf, dtype=np.float32)
    for xi, yi, zi in zip(x, y, z):
        if zi <= 0:
            continue
        if zi < depth[yi, xi]:
            depth[yi, xi] = zi
    depth[~np.isfinite(depth)] = 0.0
    return depth