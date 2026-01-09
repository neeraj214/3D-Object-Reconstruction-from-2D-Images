import sys
import json
import time
import random
from pathlib import Path
import numpy as np
import cv2
import matplotlib.pyplot as plt
import importlib.util

RESULTS_DIR = Path('results')
OUT_BASE = Path('category_images')
PREVIEW_BASE = Path('data')/'preview'
PUBLIC_DS_BASE = Path('frontend')/'public'/'datasets'
PUBLIC_CAT_BASE = Path('frontend')/'public'/'datasets'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
OUT_BASE.mkdir(parents=True, exist_ok=True)

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

def save_depth_image(depth: np.ndarray, out_path: Path):
    fig = plt.figure(figsize=(4,4))
    ax = fig.add_subplot(111)
    ax.imshow(depth, cmap='magma')
    ax.axis('off')
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=200)
    plt.close(fig)

def save_views(points: np.ndarray, out_dir: Path, stem: str):
    for name, elev, azim in [('front',20,35),('side',0,0),('top',90,0)]:
        fig = plt.figure(figsize=(3,3))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(points[:,0], points[:,1], points[:,2], s=1)
        ax.view_init(elev=elev, azim=azim)
        plt.tight_layout()
        fig.savefig(str(out_dir/f'{stem}_{name}.png'), dpi=200)
        plt.close(fig)

def find_files(root: Path, patterns):
    files = []
    for pat in patterns:
        files += list(root.rglob(pat))
    return files

def pick_samples(category_dir: Path, max_samples: int):
    imgs = find_files(category_dir, ['*.jpg','*.png','*.jpeg'])
    random.shuffle(imgs)
    return imgs[:max_samples]

def generate_for_image(img_path: Path, out_dir: Path, predictor: DPTDepthPredictor, log_lines: list):
    img = cv2.imread(str(img_path))
    if img is None:
        log_lines.append(f'NOT AVAILABLE – missing image: {img_path}')
        return False
    h, w = img.shape[:2]
    camK = synthesize_camera_K(h, w)
    t0 = time.time()
    depth = predictor.predict_depth(img)
    pts, cols = depth_to_pointcloud(depth, image_bgr=img, camera_K=camK, n_points=20000, mask=None, use_mask=False, denoise=True, gaussian=False, edge_sample=True, upsample_scale=1, smooth=True, sample_name=img_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = img_path.stem
    cv2.imwrite(str(out_dir/f'{stem}_original.png'), img)
    np.save(out_dir/f'{stem}_depth.npy', depth)
    save_depth_image(depth, out_dir/f'{stem}_depth.png')
    try:
        import open3d as o3d
        pcd = o3d.geometry.PointCloud(); pcd.points = o3d.utility.Vector3dVector(pts)
        if cols is not None:
            pcd.colors = o3d.utility.Vector3dVector(cols)
        o3d.io.write_point_cloud(str(out_dir/f'{stem}_pointcloud.ply'), pcd)
    except Exception:
        pass
    np.save(out_dir/f'{stem}_pointcloud.npy', pts)
    save_views(pts, out_dir, stem)
    t1 = time.time()
    log_lines.append(json.dumps({'image': str(img_path), 'out_dir': str(out_dir), 'num_points': int(len(pts)), 'runtime_s': float(t1-t0)}))
    # Copy previews to data/preview for frontend static serving
    try:
        dst_dir = PREVIEW_BASE/out_dir.relative_to(OUT_BASE)
        dst_dir.mkdir(parents=True, exist_ok=True)
        for name in [f'{stem}_original.png', f'{stem}_depth.png', f'{stem}_front.png', f'{stem}_side.png', f'{stem}_top.png']:
            src = out_dir/name
            if src.exists():
                import shutil
                shutil.copyfile(src, dst_dir/name)
                # also export to frontend/public/datasets/<dataset>/<category>/
                ds = out_dir.parts[-2]
                cat = out_dir.parts[-1]
                cat_dir = PUBLIC_DS_BASE/ds/cat
                cat_dir.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(src, cat_dir/name)
                # also export to frontend/public/datasets/<category>/
                cat_dir2 = PUBLIC_CAT_BASE/cat
                cat_dir2.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(src, cat_dir2/name)
    except Exception:
        pass
    return True

def list_categories(dataset_root: Path, image_dirs: list, mesh_dirs: list):
    cats = set()
    for d in image_dirs:
        p = dataset_root/d
        if p.exists():
            for sub in p.iterdir():
                if sub.is_dir():
                    cats.add(sub.name)
    for d in mesh_dirs:
        p = dataset_root/d
        if p.exists():
            for sub in p.iterdir():
                if sub.is_dir():
                    cats.add(sub.name)
    return sorted(list(cats))

def main(max_samples_per_category: int = 3, only: list = None):
    data_root = Path('data')
    datasets = {
        'pix3d': {'image_dirs': ['img','images'], 'mesh_dirs': []},
        'shapenet': {'image_dirs': ['images'], 'mesh_dirs': ['meshes']},
        'pascal3d': {'image_dirs': ['Images','images'], 'mesh_dirs': []},
        'objectnet3d': {'image_dirs': ['images'], 'mesh_dirs': []},
        'co3d': {'image_dirs': ['images'], 'mesh_dirs': []},
        'google_scanned_objects': {'image_dirs': ['images'], 'mesh_dirs': ['meshes']}
    }
    predictor = DPTDepthPredictor(model_name='DPT_Hybrid')
    category_map = {}
    index_lines = []
    summary = {'datasets': {}, 'missing': []}
    log_lines = []
    for ds_name, cfg in datasets.items():
        if only and ds_name not in only:
            continue
        ds_root = data_root/ds_name
        if not ds_root.exists():
            summary['missing'].append(f'dataset_missing:{ds_name}')
            continue
        cats = list_categories(ds_root, cfg['image_dirs'], cfg['mesh_dirs'])
        category_map[ds_name] = cats
        index_lines.append(f'# {ds_name}')
        summary['datasets'][ds_name] = {'total_categories': len(cats), 'generated': 0, 'missing_samples': 0}
        for cat in cats:
            out_dir = OUT_BASE/ds_name/cat
            out_dir.mkdir(parents=True, exist_ok=True)
            if max_samples_per_category <= 0:
                index_lines.append(f'- {cat}: {out_dir.as_posix()} (indexed)')
                continue
            found = []
            for img_dir in cfg['image_dirs']:
                p = ds_root/img_dir/cat
                if p.exists():
                    found += pick_samples(p, max_samples_per_category)
            if not found:
                summary['datasets'][ds_name]['missing_samples'] += 1
                log_lines.append(f'NOT AVAILABLE – missing image/mesh/camera_K: {ds_name}/{cat}')
                index_lines.append(f'- {cat}: {out_dir.as_posix()} (missing)')
                continue
            count = 0
            for img_path in found:
                ok = generate_for_image(img_path, out_dir, predictor, log_lines)
                if ok:
                    count += 1
            summary['datasets'][ds_name]['generated'] += count
            index_lines.append(f'- {cat}: {out_dir.as_posix()} ({count} samples)')
    (OUT_BASE/'category_list.json').write_text(json.dumps(category_map, indent=2))
    (OUT_BASE/'category_image_index.md').write_text('\n'.join(index_lines))
    lines = ['# Category Image Summary']
    for ds_name, info in summary['datasets'].items():
        lines.append(f'- {ds_name}: categories={info["total_categories"]}, generated={info["generated"]}, missing_categories={info["missing_samples"]}')
    if summary['missing']:
        lines.append(f'Missing datasets: {", ".join(summary["missing"]) }')
    (OUT_BASE/'category_image_summary.md').write_text('\n'.join(lines))
    # Write preview summary JSON for frontend
    preview_summary = {}
    for ds_name, cats in category_map.items():
        preview_summary[ds_name] = {}
        for cat in cats:
            cat_dir = PREVIEW_BASE/ds_name/cat
            if cat_dir.exists():
                items = []
                for f in sorted(cat_dir.glob('*.png')):
                    rel = Path('preview')/ds_name/cat/f.name
                    items.append({'url': f"/data/{rel.as_posix()}", 'file': f.name})
                preview_summary[ds_name][cat] = items
    (PREVIEW_BASE/'preview_summary.json').write_text(json.dumps(preview_summary, indent=2))
    log_path = RESULTS_DIR/'category_image_log.txt'
    with open(log_path, 'a') as f:
        for ln in log_lines:
            f.write(str(ln)+'\n')
        if not log_lines:
            f.write('')

if __name__ == '__main__':
    n = 3
    only = None
    if len(sys.argv) > 1:
        try:
            n = int(sys.argv[1])
        except Exception:
            n = 3
    if len(sys.argv) > 2:
        only = [s.strip() for s in sys.argv[2].split(',') if s.strip()]
    try:
        main(max_samples_per_category=n, only=only)
    except Exception as e:
        log_path = RESULTS_DIR/'category_image_log.txt'
        with open(log_path, 'a') as f:
            f.write(f'ERROR: {str(e)}\n')