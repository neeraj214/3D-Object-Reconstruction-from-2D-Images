import argparse
from pathlib import Path
import sys
import numpy as np
import torch
from PIL import Image
import torchvision.transforms as T

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.models.multiview_pointcloud import MultiViewPointCloudModel

def load_image(p, size=224):
    img = Image.open(p).convert("RGB")
    tf = T.Compose([
        T.Resize((size, size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])
    return tf(img)

def upsample_points(pts, target):
    n = pts.shape[0]
    if n >= target:
        return pts[:target]
    rep = np.random.choice(n, target-n, replace=True)
    jitter = np.random.normal(0, 0.005, (target-n, 3)).astype(np.float32)
    return np.vstack([pts, pts[rep] + jitter])

def poisson_mesh(points):
    try:
        import open3d as o3d
        pcd = o3d.geometry.PointCloud(); pcd.points = o3d.utility.Vector3dVector(points)
        pcd.estimate_normals()
        mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=8)
        mesh = mesh.filter_smooth_simple(number_of_iterations=3)
        mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=max(1000, len(mesh.triangles)//3))
        return mesh
    except Exception:
        return None

def run(front_path, side_path, ckpt, out_dir, coarse_points=1024, refined_points=4096):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state = torch.load(ckpt, map_location="cpu")
    sd = state["model"] if isinstance(state, dict) and "model" in state else state
    try:
        w = sd.get("decoder.net.4.weight") if isinstance(sd, dict) else None
        if w is not None:
            out_feats = w.shape[0]
            trained_points = int(out_feats//3)
        else:
            trained_points = coarse_points
    except Exception:
        trained_points = coarse_points
    model = MultiViewPointCloudModel(num_points=trained_points).to(device)
    model.load_state_dict(sd)
    model.eval()
    f = load_image(front_path).unsqueeze(0).to(device)
    s = load_image(side_path).unsqueeze(0).to(device)
    with torch.no_grad():
        pts_trained = model(f, s)[0].detach().cpu().numpy()
    # Downsample to coarse, then upsample to refined
    idx = np.arange(pts_trained.shape[0])
    if pts_trained.shape[0] >= coarse_points:
        np.random.shuffle(idx)
        pts = pts_trained[idx[:coarse_points]]
    else:
        pts = upsample_points(pts_trained, coarse_points)
    ref = upsample_points(pts, refined_points)
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    np.save(out/"coarse.npy", pts.astype(np.float32))
    np.save(out/"refined.npy", ref.astype(np.float32))
    mesh = poisson_mesh(ref)
    if mesh is not None:
        try:
            import open3d as o3d
            o3d.io.write_triangle_mesh(str(out/"mesh_poisson.ply"), mesh)
        except Exception:
            pass
    else:
        # Fallback: write a minimal PLY mesh from points
        ply_path = out/"mesh_poisson.ply"
        try:
            pts = ref.astype(np.float32)
            faces = []
            for i in range(0, max(0, pts.shape[0]-2), 3):
                faces.append([i, i+1, i+2])
            with open(ply_path, 'w') as f:
                f.write('ply\nformat ascii 1.0\n')
                f.write(f'element vertex {pts.shape[0]}\n')
                f.write('property float x\nproperty float y\nproperty float z\n')
                f.write(f'element face {len(faces)}\n')
                f.write('property list uchar int vertex_indices\nend_header\n')
                for p in pts:
                    f.write(f'{p[0]} {p[1]} {p[2]}\n')
                for tri in faces:
                    f.write(f'3 {tri[0]} {tri[1]} {tri[2]}\n')
        except Exception:
            pass
    print(str(out))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--front", type=str, required=True)
    ap.add_argument("--side", type=str, required=True)
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--out", type=str, default="results/final_inference")
    ap.add_argument("--coarse", type=int, default=1024)
    ap.add_argument("--refined", type=int, default=4096)
    args = ap.parse_args()
    name = Path(args.front).stem
    od = Path(args.out)/name
    run(args.front, args.side, args.ckpt, od, coarse_points=args.coarse, refined_points=args.refined)

if __name__ == "__main__":
    main()
