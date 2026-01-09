import os
import sys
import time
import json
import csv
from pathlib import Path
import numpy as np
import cv2
from PIL import Image
import importlib.util

def _load_module(module_path: str):
    p = Path(module_path)
    spec = importlib.util.spec_from_file_location(p.stem, str(p))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m

_mod_depth = _load_module('src/models/depth_dpt.py')
_mod_pcd = _load_module('src/inference/depth_to_pcd.py')
_mod_pix = _load_module('src/evaluation/pix3d_gt.py')
DPTDepthPredictor = getattr(_mod_depth, 'DPTDepthPredictor')
depth_to_pointcloud = getattr(_mod_pcd, 'depth_to_pointcloud')
read_pix3d_annotations = getattr(_mod_pix, 'read_pix3d_annotations')
ensure_gt_pointcloud = getattr(_mod_pix, 'ensure_gt_pointcloud')
render_depth_from_points = getattr(_mod_pix, 'render_depth_from_points')

RESULTS_DIR = Path('results')
EXAMPLES_DIR = RESULTS_DIR/'reconstruction_examples'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
EXAMPLES_DIR.mkdir(parents=True, exist_ok=True)
CKPT_PATH = Path('models/best_model.pth')
CHAMFER_DIR = RESULTS_DIR/'chamfer_plots'
FSCORE_DIR = RESULTS_DIR/'fscore_plots'
CHAMFER_DIR.mkdir(parents=True, exist_ok=True)
FSCORE_DIR.mkdir(parents=True, exist_ok=True)
ALIGNED_DIR = RESULTS_DIR/'aligned_icp'
REFINED_PCD_DIR = RESULTS_DIR/'refined_pointclouds'
ALIGNED_DIR.mkdir(parents=True, exist_ok=True)
REFINED_PCD_DIR.mkdir(parents=True, exist_ok=True)
PER_DATASET_DIR = RESULTS_DIR/'per_dataset_metrics'
PER_DATASET_DIR.mkdir(parents=True, exist_ok=True)

DATASETS = [
    ('pix3d', ['img','images']),
    ('shapenet', ['images']),
    ('pascal3d', ['Images','images']),
    ('objectnet3d', ['images']),
    ('co3d', ['images']),
    ('google_scanned', ['images','meshes'])
]

def pairwise_min_distances(a, b):
    from sklearn.metrics import pairwise_distances
    d = pairwise_distances(a, b)
    return d.min(axis=1), d.min(axis=0)

def chamfer_distance(a, b, max_samples=10000):
    if len(a) == 0 or len(b) == 0:
        return float('inf')
    if len(a) > max_samples:
        a = a[np.random.choice(len(a), max_samples, replace=False)]
    if len(b) > max_samples:
        b = b[np.random.choice(len(b), max_samples, replace=False)]
    da, db = pairwise_min_distances(a, b)
    return float(da.mean() + db.mean())

def umeyama_alignment(src, dst, with_scale=True):
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    mu_src = src.mean(axis=0)
    mu_dst = dst.mean(axis=0)
    src_c = src - mu_src
    dst_c = dst - mu_dst
    cov = src_c.T @ dst_c / src.shape[0]
    U, S, Vt = np.linalg.svd(cov)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        Vt[-1,:] *= -1
        R = U @ Vt
    var_src = (src_c**2).sum() / src.shape[0]
    s = 1.0
    if with_scale and var_src > 0:
        s = np.trace(np.diag(S)) / var_src
    t = mu_dst - s * R @ mu_src
    aligned = (s * (R @ src.T)).T + t
    T = np.eye(4); T[:3,:3] = s*R; T[:3,3] = t
    return aligned, R, s, t, T

def refine_icp(src, dst, init_T=None, max_iter=100, tol=1e-6, max_corr=0.02, use_plane=False, diag=None):
    try:
        import open3d as o3d
        p1 = o3d.geometry.PointCloud(); p1.points = o3d.utility.Vector3dVector(src)
        p2 = o3d.geometry.PointCloud(); p2.points = o3d.utility.Vector3dVector(dst)
        if use_plane:
            o3d.geometry.PointCloud.estimate_normals(p1, search_param=o3d.geometry.KDTreeSearchParamKNN(knn=30))
            o3d.geometry.PointCloud.estimate_normals(p2, search_param=o3d.geometry.KDTreeSearchParamKNN(knn=30))
            est = o3d.pipelines.registration.TransformationEstimationPointToPlane()
        else:
            est = o3d.pipelines.registration.TransformationEstimationPointToPoint()
        criteria = o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iter, relative_fitness=tol, relative_rmse=tol)
        init = init_T if init_T is not None else np.eye(4)
        reg = o3d.pipelines.registration.registration_icp(p1, p2, max_corr, init, est, criteria)
        T = reg.transformation
        R = T[:3,:3]; t = T[:3,3]
        print('ICP T:\n', T)
        print('ICP rmse:', reg.inlier_rmse)
        print('ICP iterations:', max_iter)
        if diag is not None:
            if np.linalg.norm(t) > 10.0*diag:
                return None
        return (R @ src.T).T + t
    except Exception:
        return None

def fscore_at_threshold(a, b, threshold):
    if len(a) == 0 or len(b) == 0:
        return 0.0
    from sklearn.metrics import pairwise_distances
    d = pairwise_distances(a, b)
    tp_a = (d.min(axis=1) < threshold).sum()
    tp_b = (d.min(axis=0) < threshold).sum()
    precision = tp_a / max(len(a), 1)
    recall = tp_b / max(len(b), 1)
    denom = precision + recall
    return float(2 * precision * recall / denom) if denom > 0 else 0.0

def reconstruction_quality_score(points):
    if points.size == 0:
        return 0.0
    mn = points.min(axis=0)
    mx = points.max(axis=0)
    vol = max(mx[0]-mn[0],1e-9)*max(mx[1]-mn[1],1e-9)*max(mx[2]-mn[2],1e-9)
    ctr = points.mean(axis=0)
    spread = float(np.sqrt(((points-ctr)**2).sum(axis=1)).mean())
    density = float(points.shape[0]/vol) if vol>0 else 0.0
    s = 0.5*np.tanh(density/1000.0) + 0.5*np.exp(-spread)
    return float(s)

def find_images(root: Path, subdirs):
    for sd in subdirs:
        d = root/sd
        if d.exists():
            imgs = []
            for ext in ['*.jpg','*.png','*.jpeg']:
                imgs += list(d.rglob(ext))
            if imgs:
                return imgs
    return []

def find_gt_pointcloud(root: Path, img_path: Path):
    pc_dir = root/'pointclouds'
    if not pc_dir.exists():
        return None
    stem = img_path.stem
    for ext in ['.npy','.npz']:
        p = pc_dir.rglob(stem+ext)
        for f in p:
            try:
                if f.suffix == '.npy':
                    arr = np.load(str(f))
                else:
                    arr = np.load(str(f))['points']
                if arr.ndim == 2 and arr.shape[1] == 3:
                    return arr
            except Exception:
                continue
    return None

def evaluate_dataset(name, root: Path, imgs, max_items=25):
    predictor = DPTDepthPredictor(model_name='DPT_Hybrid')
    rows = []
    cat_rows = {}
    t_start = time.time()
    count = 0
    synth_used = []
    for img_path in imgs[:max_items]:
        try:
            data = cv2.imread(str(img_path))
            if data is None:
                continue
            t0 = time.time()
            depth = predictor.predict_depth(data)
            h, w = data.shape[:2]
            mask_gt = None
            camK_lift = None
            if name == 'pix3d':
                anns_l = read_pix3d_annotations(root)
                info_l = None
                for a in anns_l:
                    if Path(a['image_path']).name == img_path.name:
                        info_l = a; break
                if info_l:
                    camK_lift = info_l.get('camera_K')
                    if info_l.get('mask_path') and Path(info_l['mask_path']).exists():
                        mask_gt = cv2.imread(str(info_l['mask_path']), 0)
            elif name == 'pascal3d':
                cat_dir = img_path.parent
                mdir = root/'mask'/cat_dir.name
                mp = mdir/(img_path.stem+'.png')
                if mp.exists():
                    mask_gt = cv2.imread(str(mp), 0)
            if camK_lift is None:
                f = float(max(h, w))
                camK_lift = np.array([[f, 0, w/2.0], [0, f, h/2.0], [0, 0, 1.0]], dtype=np.float32)
                synth_used.append(str(img_path))
            pts, cols = depth_to_pointcloud(depth, image_bgr=data, camera_K=camK_lift, n_points=100000, mask=mask_gt, use_mask=True, denoise=True, gaussian=False, edge_sample=True, upsample_scale=1, smooth=True, sample_name=img_path)
            runtime = time.time() - t0
            try:
                import open3d as o3d
                pcd = o3d.geometry.PointCloud(); pcd.points = o3d.utility.Vector3dVector(pts)
                o3d.io.write_point_cloud(str(REFINED_PCD_DIR/f'{name}_{img_path.stem}.ply'), pcd)
                np.save(REFINED_PCD_DIR/f'{name}_{img_path.stem}.npy', pts)
            except Exception:
                pass
            gt_pts = find_gt_pointcloud(root, img_path)
            if name == 'pix3d' and gt_pts is None:
                anns = read_pix3d_annotations(root)
                model_path = None; cat = 'unknown'
                for a in anns:
                    if Path(a['image_path']).name == img_path.name:
                        model_path = a['model_path']; cat = a['category']; break
                if model_path and Path(model_path).exists():
                    gt_pts, _ = ensure_gt_pointcloud(root, cat, img_path.stem, Path(model_path), n_points=20000)
            if gt_pts is not None:
                m = min(len(pts), len(gt_pts), 20000)
                if len(pts) > m:
                    idx = np.random.choice(len(pts), m, replace=False); pts_s = pts[idx]
                else:
                    pts_s = pts
                if len(gt_pts) > m:
                    idy = np.random.choice(len(gt_pts), m, replace=False); gt_s = gt_pts[idy]
                else:
                    gt_s = gt_pts
                cd_raw = chamfer_distance(pts_s, gt_s)
                aligned_pack = umeyama_alignment(pts_s, gt_s, with_scale=True)
                pts_aligned, R0, s0, t0_um, T0 = aligned_pack
                try:
                    import open3d as o3d
                    p1 = o3d.geometry.PointCloud(); p1.points = o3d.utility.Vector3dVector(pts_aligned)
                    p2 = o3d.geometry.PointCloud(); p2.points = o3d.utility.Vector3dVector(gt_s)
                    p1 = p1.voxel_down_sample(voxel_size=0.01)
                    p2 = p2.voxel_down_sample(voxel_size=0.01)
                    pts_aligned = np.asarray(p1.points)
                    gt_s = np.asarray(p2.points)
                except Exception:
                    pass
                gt_diag = float(np.linalg.norm((gt_s.max(axis=0)-gt_s.min(axis=0))))
                icp_aligned = refine_icp(pts_aligned, gt_s, init_T=T0, max_iter=100, tol=1e-6, max_corr=0.01*gt_diag, use_plane=False, diag=gt_diag)
                if icp_aligned is None:
                    aligned_pack = umeyama_alignment(pts_s, gt_s, with_scale=True)
                    pts_aligned = aligned_pack[0]
                else:
                    pts_aligned = icp_aligned
                try:
                    import open3d as o3d
                    pcd_a = o3d.geometry.PointCloud(); pcd_a.points = o3d.utility.Vector3dVector(pts_aligned)
                    o3d.io.write_point_cloud(str(ALIGNED_DIR/f'{name}_{img_path.stem}.ply'), pcd_a)
                    np.save(ALIGNED_DIR/f'{name}_{img_path.stem}.npy', pts_aligned)
                except Exception:
                    pass
                cd = chamfer_distance(pts_aligned, gt_s)
                fsc = fscore_at_threshold(pts_aligned, gt_s, 0.01*gt_diag)
                fcurve_vals = {
                    'p0.5': fscore_at_threshold(pts_aligned, gt_s, 0.005*gt_diag),
                    'p1': fscore_at_threshold(pts_aligned, gt_s, 0.01*gt_diag),
                    'p2': fscore_at_threshold(pts_aligned, gt_s, 0.02*gt_diag),
                    'p5': fscore_at_threshold(pts_aligned, gt_s, 0.05*gt_diag),
                    'abs_0.005': fscore_at_threshold(pts_aligned, gt_s, 0.005),
                    'abs_0.01': fscore_at_threshold(pts_aligned, gt_s, 0.01),
                    'abs_0.02': fscore_at_threshold(pts_aligned, gt_s, 0.02),
                    'abs_0.05': fscore_at_threshold(pts_aligned, gt_s, 0.05),
                }
            else:
                cd_raw = None
                cd = None
                fsc = None
                fcurve_vals = {}
            rqs = reconstruction_quality_score(pts)
            mae = None; psnr = None; ssim = None; iou = None
            if name == 'pix3d':
                anns = read_pix3d_annotations(root)
                info = None
                for a in anns:
                    if Path(a['image_path']).name == img_path.name:
                        info = a; break
                if info and gt_pts is not None:
                    camK = info.get('camera_K')
                    pose = info.get('pose')
                    gt_depth = render_depth_from_points(gt_pts, camK, pose, (h,w))
                    pred_depth = depth
                    if pred_depth.shape[:2] != (h,w):
                        pred_depth = cv2.resize(pred_depth, (w,h))
                    valid = gt_depth>0
                    if valid.sum()>100:
                        pd = pred_depth[valid].reshape(-1,1)
                        ones = np.ones_like(pd)
                        A = np.concatenate([pd, ones], axis=1)
                        gd = gt_depth[valid].reshape(-1,1)
                        x_, _, _, _ = np.linalg.lstsq(A, gd, rcond=None)
                        a = float(x_[0][0]); b = float(x_[1][0])
                        pred_depth = pred_depth*a + b
                    mae = float(np.mean(np.abs(pred_depth - gt_depth)))
                    mse = float(np.mean((pred_depth - gt_depth)**2))
                    maxv = float(np.max(gt_depth)) if np.max(gt_depth)>0 else 1.0
                    psnr = float(20*np.log10(maxv) - 10*np.log10(mse+1e-8))
                    try:
                        from skimage.metrics import structural_similarity as ssim_fn
                        ssim = float(ssim_fn((pred_depth/maxv).astype(np.float32), (gt_depth/maxv).astype(np.float32), data_range=1.0))
                    except Exception:
                        ssim = None
                    if info.get('mask_path') and Path(info['mask_path']).exists():
                        mask = cv2.imread(str(info['mask_path']), 0)
                        proj = render_depth_from_points(pts_aligned if gt_pts is not None else pts, camK, pose, (h,w))
                        pred_mask = (proj>0).astype(np.uint8)
                        mask_bin = (mask>0).astype(np.uint8)
                        inter = np.logical_and(pred_mask==1, mask_bin==1).sum()
                        union = np.logical_or(pred_mask==1, mask_bin==1).sum()
                        iou = float(inter/union) if union>0 else None
                cat_name = info['category'] if info and 'category' in info else 'unknown'
                cat_rows.setdefault(cat_name, []).append({'cd': cd, 'fscore': fsc, 'mae': mae, 'psnr': psnr, 'ssim': ssim, 'iou': iou, 'rqs': rqs})
            else:
                cat_name = img_path.parent.name
                cat_rows.setdefault(cat_name, []).append({'cd': cd, 'fscore': fsc, 'mae': mae, 'psnr': psnr, 'ssim': ssim, 'iou': iou, 'rqs': rqs})
            rows.append({'image': str(img_path), 'cd_raw': cd_raw, 'cd': cd, 'fscore': fsc, 'runtime': runtime, 'rqs': rqs, 'mae': mae, 'psnr': psnr, 'ssim': ssim, 'iou': iou, 'fcurve': fcurve_vals})
            count += 1
            if len(rows) <= 5:
                out_img = EXAMPLES_DIR/f'{name}_{img_path.stem}.png'
                import matplotlib.pyplot as plt
                if gt_pts is not None:
                    fig = plt.figure(figsize=(8,4))
                    ax1 = fig.add_subplot(121, projection='3d')
                    ax2 = fig.add_subplot(122, projection='3d')
                    ax1.scatter(pts[:,0], pts[:,1], pts[:,2], s=1, c=pts[:,2], cmap='viridis')
                    ax1.set_title('Predicted')
                    ax2.scatter(gt_pts[:,0], gt_pts[:,1], gt_pts[:,2], s=1, c=gt_pts[:,2], cmap='plasma')
                    ax2.set_title('GT')
                    ax1.view_init(elev=20, azim=35)
                    ax2.view_init(elev=20, azim=35)
                    plt.tight_layout()
                    fig.savefig(out_img, dpi=200)
                    plt.close(fig)
                else:
                    fig = plt.figure(figsize=(5,4))
                    ax = fig.add_subplot(111, projection='3d')
                    ax.scatter(pts[:,0], pts[:,1], pts[:,2], s=1, c=pts[:,2], cmap='viridis')
                    ax.view_init(elev=20, azim=35)
                    plt.tight_layout()
                    fig.savefig(out_img, dpi=200)
                    plt.close(fig)
                np.save(EXAMPLES_DIR/f'{name}_{img_path.stem}.npy', pts)
        except Exception:
            continue
    elapsed = time.time() - t_start
    if not rows:
        return {'dataset': name, 'items': 0}
    cds_raw = [r['cd_raw'] for r in rows if r.get('cd_raw') is not None]
    cds = [r['cd'] for r in rows if r['cd'] is not None]
    fss = [r['fscore'] for r in rows if r['fscore'] is not None]
    rqs = [r['rqs'] for r in rows]
    rt = [r['runtime'] for r in rows]
    maes = [r['mae'] for r in rows if r.get('mae') is not None]
    psnrs = [r['psnr'] for r in rows if r.get('psnr') is not None]
    ssims = [r['ssim'] for r in rows if r.get('ssim') is not None]
    ious = [r['iou'] for r in rows if r.get('iou') is not None]
    fcurve = {}
    for r in rows:
        for k, v in r.get('fcurve', {}).items():
            fcurve.setdefault(k, []).append(v)
    summary = {
        'dataset': name,
        'items': count,
        'cd_raw_mean': float(np.mean(cds_raw)) if cds_raw else None,
        'cd_mean': float(np.mean(cds)) if cds else None,
        'fscore_mean': float(np.mean(fss)) if fss else None,
        'rqs_mean': float(np.mean(rqs)) if rqs else None,
        'runtime_mean': float(np.mean(rt)),
        'runtime_fps': float(1.0/np.mean(rt)) if rt else None,
        'mae_mean': float(np.mean(maes)) if maes else None,
        'psnr_mean': float(np.mean(psnrs)) if psnrs else None,
        'ssim_mean': float(np.mean(ssims)) if ssims else None,
        'iou_mean': float(np.mean(ious)) if ious else None,
        'synthetic_intrinsics_count': int(len(synth_used)),
        'synthetic_intrinsics_examples': synth_used[:5]
    }
    if fcurve:
        summary['fscore_curve'] = {k: float(np.mean(v)) for k, v in fcurve.items()}
    summary['categories'] = {}
    for k, arr in cat_rows.items():
        c_cds_raw = [a.get('cd_raw') for a in arr if a.get('cd_raw') is not None]
        c_cds = [a['cd'] for a in arr if a['cd'] is not None]
        c_fss = [a['fscore'] for a in arr if a['fscore'] is not None]
        c_mae = [a['mae'] for a in arr if a['mae'] is not None]
        c_ps  = [a['psnr'] for a in arr if a['psnr'] is not None]
        c_ss  = [a['ssim'] for a in arr if a['ssim'] is not None]
        c_iou = [a['iou'] for a in arr if a['iou'] is not None]
        c_rqs = [a['rqs'] for a in arr]
        summary['categories'][k] = {
            'count': len(arr),
            'cd_raw_mean': float(np.mean(c_cds_raw)) if c_cds_raw else None,
            'cd_mean': float(np.mean(c_cds)) if c_cds else None,
            'fscore_mean': float(np.mean(c_fss)) if c_fss else None,
            'mae_mean': float(np.mean(c_mae)) if c_mae else None,
            'psnr_mean': float(np.mean(c_ps)) if c_ps else None,
            'ssim_mean': float(np.mean(c_ss)) if c_ss else None,
            'iou_mean': float(np.mean(c_iou)) if c_iou else None,
            'rqs_mean': float(np.mean(c_rqs)) if c_rqs else None,
        }
    return summary

def write_metrics_csv(summaries):
    p_prev = RESULTS_DIR/'metrics_prev.csv'
    p_cur = RESULTS_DIR/'metrics.csv'
    if p_cur.exists():
        import shutil
        shutil.copyfile(p_cur, p_prev)
    head = ['dataset','items','mae','psnr','ssim','cd','fscore','iou','rqs','runtime','fps']
    with open(p_cur, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['dataset','items','mae','psnr','ssim','cd','fscore','iou','rqs','runtime','fps'])
        for s in summaries:
            w.writerow([
                s.get('dataset'), s.get('items'), s.get('mae_mean','-'), s.get('psnr_mean','-'), s.get('ssim_mean','-'),
                s.get('cd_mean'), s.get('fscore_mean'), s.get('iou_mean','-'), s.get('rqs_mean'),
                s.get('runtime_mean'), s.get('runtime_fps')
            ])

def write_fscore_comparison():
    p_prev = RESULTS_DIR/'metrics_prev.csv'
    p_cur = RESULTS_DIR/'metrics.csv'
    out = RESULTS_DIR/'fscore_comparison.md'
    if not p_prev.exists() or not p_cur.exists():
        return
    import pandas as pd
    prev = pd.read_csv(p_prev)
    cur = pd.read_csv(p_cur)
    lines = []
    lines.append('dataset | fscore_prev | fscore_cur | delta\n')
    lines.append('---|---:|---:|---:\n')
    for ds in cur['dataset'].tolist():
        fp = float(prev[prev['dataset']==ds]['fscore'].values[0]) if ds in prev['dataset'].values else float('nan')
        fc = float(cur[cur['dataset']==ds]['fscore'].values[0])
        d = fc - fp if fp==fp else float('nan')
        lines.append(f"{ds} | {fp} | {fc} | {d}\n")
    with open(out,'w') as f:
        f.write(''.join(lines))

def write_dataset_specific_reports(summaries):
    for s in summaries:
        ds = s.get('dataset')
        p = PER_DATASET_DIR/f"{ds}_metrics.csv"
        with open(p, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['dataset','items','mae','psnr','ssim','cd','fscore','iou','rqs','runtime','fps'])
            w.writerow([
                s.get('dataset'), s.get('items'), s.get('mae_mean','-'), s.get('psnr_mean','-'), s.get('ssim_mean','-'),
                s.get('cd_mean'), s.get('fscore_mean'), s.get('iou_mean','-'), s.get('rqs_mean'),
                s.get('runtime_mean'), s.get('runtime_fps')
            ])
        q = RESULTS_DIR/f"accuracy_table_{ds}.md"
        with open(q,'w') as f:
            f.write('category | count | cd | fscore | iou | mae | psnr | ssim | rqs\n')
            f.write('---|---:|---:|---:|---:|---:|---:|---:|---:\n')
            for k, v in s.get('categories',{}).items():
                f.write(f"{k} | {v.get('count')} | {v.get('cd_mean')} | {v.get('fscore_mean')} | {v.get('iou_mean')} | {v.get('mae_mean')} | {v.get('psnr_mean')} | {v.get('ssim_mean')} | {v.get('rqs_mean')}\n")

def write_metrics_summary(summaries):
    overall = {
        'datasets': summaries,
        'best_dataset': None,
        'worst_dataset': None
    }
    best = None
    worst = None
    for s in summaries:
        score = s.get('fscore_mean') if s.get('fscore_mean') is not None else (1.0/(s.get('cd_mean')+1e-6) if s.get('cd_mean') is not None else s.get('rqs_mean'))
        if score is None:
            continue
        if best is None or score > best[1]:
            best = (s['dataset'], score)
        if worst is None or score < worst[1]:
            worst = (s['dataset'], score)
    if best:
        overall['best_dataset'] = {'name': best[0], 'score': best[1]}
    if worst:
        overall['worst_dataset'] = {'name': worst[0], 'score': worst[1]}
    with open(RESULTS_DIR/'metrics_summary.json','w') as f:
        json.dump(overall, f, indent=2)

def write_final_report(summaries):
    lines = []
    lines.append('# Final Accuracy Report\n')
    for s in summaries:
        lines.append(f"## {s.get('dataset')}\n")
        lines.append(f"Items: {s.get('items')}\n")
        lines.append(f"Chamfer (unaligned): {s.get('cd_raw_mean')}\n")
        lines.append(f"Chamfer (aligned): {s.get('cd_mean')}\n")
        lines.append(f"F-Score mean: {s.get('fscore_mean')}\n")
        if s.get('fscore_curve'):
            for k, v in s['fscore_curve'].items():
                lines.append(f"F-Score {k}: {v}\n")
        lines.append(f"IoU mean: {s.get('iou_mean')}\n")
        lines.append(f"Depth MAE: {s.get('mae_mean')}\n")
        lines.append(f"PSNR: {s.get('psnr_mean')}\n")
        lines.append(f"SSIM: {s.get('ssim_mean')}\n")
        lines.append(f"RQS: {s.get('rqs_mean')}\n")
        lines.append(f"Runtime FPS: {s.get('runtime_fps')}\n")
        lines.append(f"Synthetic intrinsics used: {s.get('synthetic_intrinsics_count')}\n")
        if s.get('synthetic_intrinsics_examples'):
            lines.append(f"Examples: {', '.join(s.get('synthetic_intrinsics_examples'))}\n")
        lines.append('\n')
    # Best/Worst categories (pix3d)
    for s in summaries:
        if s.get('dataset')=='pix3d' and s.get('categories'):
            cats = s['categories']
            scores = []
            for k,v in cats.items():
                sc = v.get('fscore_mean') if v.get('fscore_mean') is not None else (1.0/(v.get('cd_mean')+1e-6) if v.get('cd_mean') is not None else v.get('rqs_mean'))
                if sc is not None:
                    scores.append((k, sc))
            if scores:
                best_c = max(scores, key=lambda x: x[1])[0]
                worst_c = min(scores, key=lambda x: x[1])[0]
                lines.append(f"Best category: {best_c}\n")
                lines.append(f"Worst category: {worst_c}\n")
                lines.append('\n')
    # Training status
    st = Path('results/checkpoints_v3/status.json')
    if st.exists():
        try:
            js = json.loads(st.read_text())
            lines.append('## Training v3 Status\n')
            lines.append(json.dumps(js)+'\n')
            lines.append('\n')
        except Exception:
            pass
    with open(RESULTS_DIR/'final_report.md','w') as f:
        f.write('\n'.join(lines))

def write_loss_curves(summaries):
    import matplotlib.pyplot as plt
    xs = list(range(len(summaries)))
    cd = [s.get('cd_mean') if s.get('cd_mean') is not None else 0 for s in summaries]
    f1 = [s.get('fscore_mean') if s.get('fscore_mean') is not None else 0 for s in summaries]
    plt.figure(figsize=(7,4))
    plt.plot(xs, cd, label='Chamfer (mean)')
    plt.plot(xs, f1, label='F-Score (mean)')
    plt.xlabel('Dataset Index')
    plt.ylabel('Metric Value')
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULTS_DIR/'loss_curves.png', dpi=200)
    plt.close()

def write_threshold_curves(summaries):
    for s in summaries:
        if s.get('fscore_curve'):
            p = FSCORE_DIR/f"{s['dataset']}_curve.csv"
            with open(p, 'w', newline='') as f:
                w = csv.writer(f)
                w.writerow(['threshold','fscore'])
                for k, v in s['fscore_curve'].items():
                    w.writerow([k, v])

def run():
    ckpt_loaded = False
    if CKPT_PATH.exists():
        try:
            import torch
            _ = torch.load(CKPT_PATH, map_location='cpu')
            ckpt_loaded = True
        except Exception:
            ckpt_loaded = False
    present = []
    for name, subdirs in DATASETS:
        root = Path('data')/name
        if root.exists():
            imgs = find_images(root, subdirs)
            if imgs:
                present.append((name, root, imgs))
    summaries = []
    for name, root, imgs in present:
        s = evaluate_dataset(name, root, imgs)
        summaries.append(s)
    present_names = {s['dataset'] for s in summaries}
    for name, _ in DATASETS:
        if name not in present_names and name in ('shapenet','pascal3d'):
            summaries.append({'dataset': name, 'items': 0})
    write_metrics_csv(summaries)
    write_dataset_specific_reports(summaries)
    write_metrics_summary(summaries)
    write_final_report(summaries)
    write_loss_curves(summaries)
    write_fscore_comparison()
    write_threshold_curves(summaries)
    try:
        import matplotlib.pyplot as plt
        for s in summaries:
            if s.get('cd_mean') is not None:
                plt.figure(figsize=(5,3))
                plt.bar([s['dataset']], [s['cd_mean']])
                plt.tight_layout()
                plt.savefig(CHAMFER_DIR/f"{s['dataset']}.png", dpi=160)
                plt.close()
            if s.get('fscore_curve'):
                labels = list(s['fscore_curve'].keys())
                values = [s['fscore_curve'][k] for k in labels]
                plt.figure(figsize=(6,3))
                plt.plot(labels, values, marker='o')
                plt.tight_layout()
                plt.savefig(FSCORE_DIR/f"{s['dataset']}_curve.png", dpi=160)
                plt.close()
    except Exception:
        pass
    missing = {}
    for name, _ in DATASETS:
        root = Path('data')/name
        miss = []
        if name == 'pix3d':
            if not (root/'pix3d.json').exists():
                miss.append('pix3d.json')
            if not (root/'model').exists():
                miss.append('model')
            if not (root/'img').exists():
                miss.append('img')
            if not (root/'mask').exists():
                miss.append('mask')
        elif name == 'shapenet':
            if not (root).exists():
                miss.append('dataset_root')
        elif name == 'pascal3d':
            if not (root/'Annotations').exists():
                miss.append('Annotations')
        elif name == 'objectnet3d':
            if not (root/'annotations').exists():
                miss.append('annotations')
        elif name == 'co3d':
            if not (root).exists():
                miss.append('dataset_root')
        elif name == 'google_scanned':
            if not (root/'meshes').exists():
                miss.append('meshes')
        missing[name] = miss
    quality = {
        'summaries': summaries,
        'missing_assets': missing
    }
    with open(RESULTS_DIR/'quality_report.json','w') as f:
        json.dump(quality, f, indent=2)
    with open(RESULTS_DIR/'accuracy_table.md','w') as f:
        f.write('dataset | items | cd | fscore | iou | mae | psnr | ssim | rqs\n')
        f.write('---|---:|---:|---:|---:|---:|---:|---:|---:\n')
        for s in summaries:
            f.write(f"{s.get('dataset')} | {s.get('items')} | {s.get('cd_mean')} | {s.get('fscore_mean')} | {s.get('iou_mean')} | {s.get('mae_mean')} | {s.get('psnr_mean')} | {s.get('ssim_mean')} | {s.get('rqs_mean')}\n")
        for s in summaries:
            if s.get('dataset')=='pix3d' and s.get('categories'):
                f.write('\n# pix3d categories\n')
                f.write('category | count | cd | fscore | iou | mae | psnr | ssim | rqs\n')
                f.write('---|---:|---:|---:|---:|---:|---:|---:|---:\n')
                for k, v in s['categories'].items():
                    f.write(f"{k} | {v.get('count')} | {v.get('cd_mean')} | {v.get('fscore_mean')} | {v.get('iou_mean')} | {v.get('mae_mean')} | {v.get('psnr_mean')} | {v.get('ssim_mean')} | {v.get('rqs_mean')}\n")
    best = None
    worst = None
    for s in summaries:
        score = s.get('fscore_mean') if s.get('fscore_mean') is not None else (1.0/(s.get('cd_mean')+1e-6) if s.get('cd_mean') is not None else s.get('rqs_mean'))
        if score is None:
            continue
        if best is None or score > best[1]:
            best = (s['dataset'], score)
        if worst is None or score < worst[1]:
            worst = (s['dataset'], score)
    avg_acc = [s.get('fscore_mean') for s in summaries if s.get('fscore_mean') is not None]
    avg_acc = float(np.mean(avg_acc)) if avg_acc else None
    avg_cd_list = [s.get('cd_mean') for s in summaries if s.get('cd_mean') is not None]
    avg_cd = float(np.mean(avg_cd_list)) if avg_cd_list else None
    bleu_like_list = [s.get('rqs_mean') for s in summaries if s.get('rqs_mean') is not None]
    bleu_like = float(np.mean(bleu_like_list)) if bleu_like_list else None
    prec = avg_acc if avg_acc is not None else None
    rec = avg_acc if avg_acc is not None else None
    print('Checkpoint loaded:', 'yes' if ckpt_loaded else 'no')
    print('Best dataset:', best[0] if best else '-')
    print('Worst dataset:', worst[0] if worst else '-')
    print('Overall average accuracy:', f'{avg_acc:.4f}' if avg_acc is not None else '-')
    print('Average Chamfer Distance:', f'{avg_cd:.6f}' if avg_cd is not None else '-')
    print('BLEU-style similarity score:', f'{bleu_like:.4f}' if bleu_like is not None else '-')
    print('Precision / Recall for reconstruction:', f'{prec:.4f}' if prec is not None else '-', f'{rec:.4f}' if rec is not None else '-')

if __name__ == '__main__':
    run()