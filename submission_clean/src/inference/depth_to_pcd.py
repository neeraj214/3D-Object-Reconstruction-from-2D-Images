import argparse
from pathlib import Path
import numpy as np
import cv2
import open3d as o3d
import logging
try:
    from sklearn.neighbors import KDTree
except Exception:
    KDTree = None

def _mls_smooth_points(pts, radius=0.02, iterations=2):
    if KDTree is None:
        return pts
    kdt = KDTree(pts)
    for _ in range(iterations):
        new_pts = pts.copy()
        for i in range(len(pts)):
            idx = kdt.query_radius(pts[i:i+1], r=radius)[0]
            if len(idx) > 5:
                neighborhood = pts[idx]
                mu = neighborhood.mean(axis=0)
                X = neighborhood - mu
                C = (X.T @ X) / max(len(neighborhood)-1, 1)
                w, v = np.linalg.eigh(C)
                n = v[:,0]
                p = pts[i]
                new_pts[i] = p - (n * (np.dot(p - mu, n)))
        pts = new_pts
    return pts

def _normalize_align(points: np.ndarray, mask: np.ndarray = None, axis: str = 'x z -y') -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    if mask is not None and mask.size == pts.shape[0]:
        m = (mask.astype(np.uint8) > 0)
        sel = pts[m]
        ctr = sel.mean(axis=0, keepdims=True) if sel.size else pts.mean(axis=0, keepdims=True)
    else:
        ctr = pts.mean(axis=0, keepdims=True)
    pts = pts - ctr
    d = np.linalg.norm(pts, axis=1)
    s = float(np.max(d)) if d.size else 1.0
    if s > 0:
        pts = pts / s
    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
    if axis == 'x z -y':
        pts = np.stack([x, z, -y], axis=1)
    elif axis == 'x -y z':
        pts = np.stack([x, -y, z], axis=1)
    else:
        pts = np.stack([x, y, z], axis=1)
    return pts.astype(np.float32)

def depth_to_pointcloud(depth, image_bgr=None, camera_K=None, n_points=100000, mask=None, use_mask=True, denoise=False, gaussian=False, edge_sample=True, upsample_scale=1, smooth=False, sample_name=None):
    if camera_K is None and image_bgr is not None:
        h0, w0 = image_bgr.shape[:2]
        camera_K = np.array(compute_intrinsics(h0, w0, f_scale=1.0), dtype=np.float32)
        _check_intrinsics_consistency("depth_to_pcd", camera_K, h0, w0, f_scale=1.0)
    elif camera_K is None:
        sp = str(sample_name) if sample_name else 'unknown'
        logging.error(f"camera_K missing and image_bgr unavailable for sample: {sp}")
        raise ValueError("camera_K is required or image_bgr must be provided to compute intrinsics")
    d32 = depth.astype(np.float32)
    dproc = cv2.bilateralFilter(d32, 5, 0.1, 2.0)
    if gaussian:
        dproc = cv2.GaussianBlur(dproc, (3,3), 0.5)
    if upsample_scale and upsample_scale > 1:
        h0, w0 = dproc.shape
        dproc = cv2.resize(dproc, (int(w0*upsample_scale), int(h0*upsample_scale)), interpolation=cv2.INTER_LINEAR)
        if image_bgr is not None:
            image_bgr = cv2.resize(image_bgr, (int(w0*upsample_scale), int(h0*upsample_scale)), interpolation=cv2.INTER_LINEAR)
        if mask is not None:
            mask = cv2.resize(mask, (int(w0*upsample_scale), int(h0*upsample_scale)), interpolation=cv2.INTER_NEAREST)
    h, w = dproc.shape
    ys, xs = np.mgrid[0:h, 0:w]
    fx = float(camera_K[0][0]); fy = float(camera_K[1][1])
    cx = float(camera_K[0][2]); cy = float(camera_K[1][2])
    z = dproc
    x = (xs - cx) * z / fx
    y = (ys - cy) * z / fy
    if use_mask and mask is not None:
        m = (mask>0) & (z>0)
        x = x[m]; y = y[m]; z = z[m]
    pts = np.stack([x, y, z], axis=-1).reshape(-1, 3)
    if smooth:
        pcd = o3d.geometry.PointCloud(); pcd.points = o3d.utility.Vector3dVector(pts)
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=30))
        pts = _mls_smooth_points(np.asarray(pcd.points), radius=0.02, iterations=2)
    if denoise:
        pcd = o3d.geometry.PointCloud(); pcd.points = o3d.utility.Vector3dVector(pts)
        bbox = np.array([pts.min(axis=0), pts.max(axis=0)])
        diag = float(np.linalg.norm(bbox[1]-bbox[0]))
        cl, _ = pcd.remove_statistical_outlier(nb_neighbors=30, std_ratio=2.0)
        pcd = cl
        pcd, _ = pcd.remove_radius_outlier(nb_points=10, radius=max(1e-6, 0.01*diag))
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=30))
        pts = np.asarray(pcd.points)
    prob = None
    if edge_sample:
        mag = None
        try:
            gx = cv2.Sobel(dproc, cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(dproc, cv2.CV_32F, 0, 1, ksize=3)
            mag = np.sqrt(gx**2 + gy**2)
        except Exception:
            mag = None
        if mag is None and image_bgr is not None:
            gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
            gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
            mag = np.sqrt(gx**2 + gy**2)
        prob = mag.reshape(-1).astype(np.float64)
        prob = prob - prob.min(); maxp = prob.max(); prob = prob / max(maxp, 1e-6)
        if use_mask and mask is not None:
            border = cv2.morphologyEx((mask>0).astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((3,3), np.uint8))
            prob = prob + 0.5*border.reshape(-1)
        prob += 1e-6
    colors = None
    if image_bgr is not None:
        img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        flat = img_rgb.reshape(-1, 3).astype(np.float32)/255.0
        if use_mask and mask is not None:
            flat = flat[m.reshape(-1)]
        colors = flat
    N = len(pts)
    target = int(n_points)
    if N > target:
        if prob is not None and len(prob)==N:
            prob = prob / prob.sum()
            idx = np.random.choice(N, target, replace=False, p=prob)
        else:
            idx = np.random.choice(N, target, replace=False)
        pts = pts[idx]
        if colors is not None:
            colors = colors[idx]
    elif N < target and N > 0:
        rep = np.random.choice(N, target - N, replace=True)
        jitter = np.random.normal(scale=1e-3, size=(target - N, 3))
        new_pts = pts[rep] + jitter
        pts = np.concatenate([pts, new_pts], axis=0)
        if colors is not None:
            colors = np.concatenate([colors, colors[rep]], axis=0)
    pts = _normalize_align(pts, mask=(mask.reshape(-1) if (use_mask and mask is not None) else None), axis='x z -y')
    if sample_name is not None:
        out_dir = Path('results')/'refined_pointclouds'
        out_dir.mkdir(parents=True, exist_ok=True)
        base = Path(sample_name).stem
        try:
            pcd_out = o3d.geometry.PointCloud(); pcd_out.points = o3d.utility.Vector3dVector(pts)
            if colors is not None:
                pcd_out.colors = o3d.utility.Vector3dVector(colors)
            o3d.io.write_point_cloud(str(out_dir/f'{base}_refined.ply'), pcd_out)
            np.save(out_dir/f'{base}_refined.npy', pts)
        except Exception as e:
            logging.error(f"Failed to save refined point cloud: {e}")
    return pts, colors

def save_outputs(pts, colors, out_dir, name):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    np.save(Path(out_dir) / f'{name}.npy', pts)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_point_cloud(str(Path(out_dir) / f'{name}.ply'), pcd)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--depth', type=str, required=True)
    ap.add_argument('--image', type=str, required=True)
    ap.add_argument('--camera_K', type=str, required=True)
    ap.add_argument('--mask', type=str, default=None)
    ap.add_argument('--out', type=str, default='results/refined_pointclouds')
    ap.add_argument('--name', type=str, default=None)
    ap.add_argument('--n_points', type=int, default=100000)
    ap.add_argument('--denoise', action='store_true')
    ap.add_argument('--gaussian', action='store_true')
    ap.add_argument('--edge_sample', action='store_true')
    ap.add_argument('--use_mask', action='store_true')
    args = ap.parse_args()
    d = np.load(args.depth) if args.depth.endswith('.npy') else cv2.imread(args.depth, cv2.IMREAD_UNCHANGED).astype(np.float32)
    img = cv2.imread(args.image) if args.image else None
    camK = np.load(args.camera_K) if args.camera_K.endswith('.npy') else None
    m = cv2.imread(args.mask, 0) if args.mask else None
    pts, colors = depth_to_pointcloud(d, image_bgr=img, camera_K=camK, n_points=args.n_points, mask=m, use_mask=args.use_mask, denoise=args.denoise, gaussian=args.gaussian, edge_sample=args.edge_sample, sample_name=args.image)
    base = Path(args.image).stem if args.name is None else args.name
    save_outputs(pts, colors, args.out, f'{base}_refined')

if __name__ == '__main__':
    main()
def compute_intrinsics(h, w, f_scale=1.0):
    f = max(h, w) * float(f_scale)
    cx = w / 2.0
    cy = h / 2.0
    return [[f, 0, cx], [0, f, cy], [0, 0, 1.0]]

def _check_intrinsics_consistency(name, K, h, w, f_scale=1.0):
    try:
        K_ref = np.array(compute_intrinsics(h, w, f_scale=f_scale), dtype=np.float32)
        K_in = np.array(K, dtype=np.float32) if K is not None else K_ref
        match = np.allclose(K_in, K_ref, rtol=1e-5, atol=1e-5)
        print(f"[intrinsics-check] {name}: match={match}")
        if not match:
            print(f"[intrinsics-check] {name}: expected={K_ref.tolist()} got={K_in.tolist()}")
    except Exception as e:
        print(f"[intrinsics-check] {name}: failed {e}")
