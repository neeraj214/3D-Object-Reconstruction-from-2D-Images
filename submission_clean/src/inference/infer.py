import argparse
from pathlib import Path
import json
import random
import sys
import numpy as np
import torch
from PIL import Image
import torchvision.transforms as T

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.models.multiview_pointcloud import MultiViewPointCloudModel
from src.models.depth_dpt import DPTDepthPredictor
from src.inference.depth_to_pcd import depth_to_pointcloud

def _map_path(p):
    if isinstance(p, str) and p.startswith('/mnt/data/'):
        return str(Path('C:/mnt/data')/Path(p).name)
    return p

def load_image(p, size=224):
    p = _map_path(p)
    img = Image.open(p).convert("RGB")
    tf = T.Compose([
        T.Resize((size, size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])
    return tf(img)

def estimate_depth(image_path):
    try:
        import cv2
        dpt = DPTDepthPredictor()
        bgr = cv2.imread(image_path)
        return dpt.predict_depth(bgr)
    except Exception:
        return None

def align_normals(points):
    try:
        import open3d as o3d
        pcd = o3d.geometry.PointCloud(); pcd.points = o3d.utility.Vector3dVector(points)
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=30))
        return np.asarray(pcd.points)
    except Exception:
        return points

def run_single(front, ckpt, out_dir, coarse=1024, refined=4096, mesh=False, poisson_depth=10, smooth_iter=15, simplify=15000, texture=False, side=None, use_model=False):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    d = None if use_model else estimate_depth(_map_path(front))
    if d is None or use_model:
        sd = torch.load(ckpt, map_location="cpu"); sd = sd.get("model", sd)
        try:
            model = MultiViewPointCloudModel(num_points=refined)
            model.load_state_dict(sd)
        except RuntimeError:
            try:
                model = MultiViewPointCloudModel(num_points=2048)
                model.load_state_dict(sd)
                refined = 2048
            except Exception:
                model = MultiViewPointCloudModel(num_points=1024)
                model.load_state_dict(sd)
                refined = 1024
        model.eval()
        f = load_image(front).unsqueeze(0)
        with torch.no_grad():
            pts = model(f, f)[0].detach().cpu().numpy()
    else:
        try:
            import cv2
            img = cv2.imread(_map_path(front))
            h, w = img.shape[:2]
            camK = np.array(compute_intrinsics(h, w, f_scale=1.0), dtype=np.float32)
            pts, _ = depth_to_pointcloud(d, image_bgr=img, camera_K=camK, n_points=refined, denoise=True, edge_sample=True, smooth=True, sample_name="infer_v2")
        except Exception:
            pts = upsample_points(np.random.randn(coarse).astype(np.float32), refined)
    pts = align_normals(pts)
    np.save(out/"points.npy", pts.astype(np.float32))
    if mesh:
        from src.mesh.poisson import poisson_mesh_from_points, save_mesh
        from src.mesh.cleanup import filter_by_density, largest_component, laplacian_smooth, refine_normals, simplify as simplify_mesh
        from src.mesh.texturing import planar_uv, project_texture_front, write_obj_with_mtl
        m, dens = poisson_mesh_from_points(pts, depth=poisson_depth)
        if m is not None:
            if dens is not None:
                m = filter_by_density(m, dens, keep_ratio=0.7)
            m = largest_component(m)
            m = laplacian_smooth(m, iterations=smooth_iter)
            m = refine_normals(m)
            if simplify and simplify>0:
                m = simplify_mesh(m, target_faces=simplify)
            save_mesh(m, Path(out)/"mesh_poisson.ply")
            if texture:
                uv = planar_uv(m)
                tex = project_texture_front(front, resolution=1024)
                write_obj_with_mtl(m, uv, tex, Path(out)/"mesh.obj")
    return str(out)

def run_batch(folder, ckpt, out_dir, coarse=1024, refined=4096, mesh=False, poisson_depth=10, smooth_iter=15, simplify=15000, texture=False):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    images = sorted(list(Path(folder).glob("*.jpg"))) + sorted(list(Path(folder).glob("*.png")))
    for img in images:
        run_single(str(img), ckpt, out/img.stem, coarse, refined, mesh, poisson_depth, smooth_iter, simplify, texture)
    return str(out)

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

def run_random_pix3d(pix_root, ckpt, out_dir, refined=4096, f_scale=1.0):
    root = Path(pix_root)
    meta = root/'pix3d.json'
    if not meta.exists():
        raise FileNotFoundError(str(meta))
    data = json.loads(meta.read_text())
    items = []
    for it in data:
        ip = root/it.get('img','')
        mp = root/it.get('model','') if 'model' in it else None
        mk = root/it.get('mask','') if 'mask' in it else None
        if ip.exists():
            items.append({'image': str(ip), 'model': str(mp) if mp and Path(mp).exists() else None, 'mask': str(mk) if mk and Path(mk).exists() else None, 'camera_K': it.get('camera_K', None), 'category': it.get('category','unknown')})
    random.shuffle(items)
    sel = items[:3]
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    stats = []
    for i, it in enumerate(sel):
        name = Path(it['image']).stem
        sub = out_dir/name; sub.mkdir(parents=True, exist_ok=True)
        try:
            import cv2, open3d as o3d
            bgr = cv2.imread(it['image'])
            h, w = bgr.shape[:2]
            K = np.array(it['camera_K'], dtype=np.float32) if it['camera_K'] is not None else np.array(compute_intrinsics(h, w, f_scale=f_scale), dtype=np.float32)
            _check_intrinsics_consistency(f"inference:{name}", K, h, w, f_scale=f_scale)
            d = estimate_depth(it['image'])
            pts, colors = depth_to_pointcloud(d, image_bgr=bgr, camera_K=K, n_points=refined, mask=(cv2.imread(it['mask'], 0) if it['mask'] else None), use_mask=True, denoise=True, edge_sample=True, smooth=True, sample_name=name)
            pcd = o3d.geometry.PointCloud(); pcd.points = o3d.utility.Vector3dVector(pts)
            o3d.io.write_point_cloud(str(sub/'point_cloud.ply'), pcd)
            cv2.imwrite(str(sub/'depth.png'), (d*255.0/max(float(d.max()),1e-6)).astype(np.uint8))
            intr = compute_intrinsics(h, w, f_scale=f_scale)
            Path(sub/'intrinsics.json').write_text(json.dumps({'intrinsics': intr}, indent=2))
            ctr = pts.mean(axis=0).tolist(); rng = [float(pts.min()), float(pts.max())]
            stats.append({'name': name, 'center': ctr, 'range': rng, 'num_points': int(len(pts))})
        except Exception as e:
            Path(sub/'error.txt').write_text(str(e))
    Path(out_dir/'pointcloud_stats.json').write_text(json.dumps({'items': stats}, indent=2))
    return str(out_dir)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--front", type=str)
    ap.add_argument("--folder", type=str)
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--out", type=str, default="results/infer_v4")
    ap.add_argument("--coarse", type=int, default=1024)
    ap.add_argument("--refined", type=int, default=4096)
    ap.add_argument("--mesh", action="store_true")
    ap.add_argument("--poisson-depth", type=int, default=10)
    ap.add_argument("--smooth-iter", type=int, default=15)
    ap.add_argument("--simplify", type=int, default=15000)
    ap.add_argument("--texture", action="store_true")
    ap.add_argument("--use-model", action="store_true")
    args = ap.parse_args()
    if args.folder:
        p = run_batch(args.folder, args.ckpt, args.out, args.coarse, args.refined, args.mesh, args.poisson_depth, args.smooth_iter, args.simplify, args.texture)
        print(p)
        return
    if args.front:
        p = run_single(args.front, args.ckpt, args.out, args.coarse, args.refined, args.mesh, args.poisson_depth, args.smooth_iter, args.simplify, args.texture, use_model=args.use_model)
        print(p)
        return
    print("no input provided")

if __name__ == "__main__":
    main()
