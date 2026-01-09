import sys
import json
import time
import argparse
from pathlib import Path
import numpy as np
import cv2
import matplotlib.pyplot as plt
import importlib.util
import open3d as o3d

RESULTS_DIR = Path('results')
SINGLE_DIR = RESULTS_DIR/'single_image_test'
TEST_OUT = Path('test_output')
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
SINGLE_DIR.mkdir(parents=True, exist_ok=True)
TEST_OUT.mkdir(parents=True, exist_ok=True)

def _load_module(module_path: str):
    p = Path(module_path)
    spec = importlib.util.spec_from_file_location(p.stem, str(p))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m

_mod_depth = _load_module('src/models/depth_dpt.py')
_mod_pcd = _load_module('src/inference/depth_to_pcd.py')
DPTDepthPredictor = getattr(_mod_depth, 'DPTDepthPredictor')
depth_to_pointcloud = getattr(_mod_pcd, 'depth_to_pointcloud')

def synthesize_camera_K(h, w):
    f = float(max(h, w))
    return np.array([[f, 0, w/2.0], [0, f, h/2.0], [0, 0, 1.0]], dtype=np.float32)

def visualize_points(pts, out_dir):
    fig = plt.figure(figsize=(9,3))
    ax1 = fig.add_subplot(131, projection='3d')
    ax2 = fig.add_subplot(132, projection='3d')
    ax3 = fig.add_subplot(133, projection='3d')
    ax1.scatter(pts[:,0], pts[:,1], pts[:,2], s=1)
    ax2.scatter(pts[:,0], pts[:,1], pts[:,2], s=1)
    ax3.scatter(pts[:,0], pts[:,1], pts[:,2], s=1)
    ax1.view_init(elev=20, azim=35)
    ax2.view_init(elev=0, azim=0)
    ax3.view_init(elev=90, azim=0)
    plt.tight_layout()
    fig.savefig(out_dir/'visualization.png', dpi=200)
    plt.close(fig)

def compute_stats(points: np.ndarray):
    num = int(points.shape[0])
    mn = points.min(axis=0)
    mx = points.max(axis=0)
    bb = {'min': mn.tolist(), 'max': mx.tolist()}
    vol = float(max(mx[0]-mn[0],1e-9)*max(mx[1]-mn[1],1e-9)*max(mx[2]-mn[2],1e-9))
    density = float(num/vol) if vol>0 else float('inf')
    ctr = points.mean(axis=0)
    spread = float(np.sqrt(((points-ctr)**2).sum(axis=1)).mean())
    q = 0.5*np.tanh(density/1000.0) + 0.5*np.exp(-spread)
    return {'num_points': num, 'density': density, 'spread': spread, 'bounding_box': bb, 'quality_score': float(q)}

def run(image_path: str, n_points: int = 20000):
    log_lines = []
    img_p = Path(image_path)
    if not img_p.exists():
        log_lines.append(f'Image not found: {image_path}')
        candidates = []
        for root in ['data/pix3d/img','results/reconstruction_examples']:
            p = Path(root)
            if p.exists():
                for ext in ['*.jpg','*.png','*.jpeg']:
                    candidates += list(p.rglob(ext))
        if candidates:
            img_p = candidates[0]
            log_lines.append(f'Fallback image: {img_p}')
        else:
            raise FileNotFoundError('No images available for test')
    img = cv2.imread(str(img_p))
    h, w = img.shape[:2]
    camK = synthesize_camera_K(h, w)
    predictor = DPTDepthPredictor(model_name='DPT_Hybrid')
    t0 = time.time()
    depth = predictor.predict_depth(img)
    t1 = time.time()
    pts, cols = depth_to_pointcloud(depth, image_bgr=img, camera_K=camK, n_points=int(n_points), mask=None, use_mask=False, denoise=True, gaussian=False, edge_sample=True, upsample_scale=1, smooth=True, sample_name=img_p)
    visualize_points(pts, SINGLE_DIR)
    # Save depth visualization
    try:
        fig = plt.figure(figsize=(4,4))
        ax = fig.add_subplot(111)
        ax.imshow(depth, cmap='magma')
        ax.axis('off')
        fig.tight_layout()
        fig.savefig(str(SINGLE_DIR/'depth.png'), dpi=200)
        plt.close(fig)
    except Exception:
        pass
    # Save refined point cloud
    pcd = o3d.geometry.PointCloud(); pcd.points = o3d.utility.Vector3dVector(pts)
    if cols is not None:
        pcd.colors = o3d.utility.Vector3dVector(cols)
    o3d.io.write_point_cloud(str(SINGLE_DIR/'point_cloud.ply'), pcd)
    # Stats
    stats = compute_stats(pts)
    with open(SINGLE_DIR/'pointcloud_stats.json','w') as f:
        f.write(json.dumps(stats, indent=2))
    np.save(SINGLE_DIR/'depth.npy', depth)
    quality = {
        'image': str(img_p),
        'num_points': int(len(pts)),
        'bbox_min': pts.min(axis=0).tolist(),
        'bbox_max': pts.max(axis=0).tolist(),
        'runtime_depth_s': float(t1-t0),
    }
    with open(SINGLE_DIR/'quality.txt','w') as f:
        f.write(json.dumps(quality, indent=2))
    with open(RESULTS_DIR/'testing_logs.txt','a') as f:
        for ln in log_lines:
            f.write(ln+'\n')
        f.write(json.dumps(quality)+'\n')

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('image', type=str)
    ap.add_argument('--n_points', type=int, default=20000)
    args = ap.parse_args()
    run(args.image, n_points=args.n_points)