import argparse
from pathlib import Path
import numpy as np
import open3d as o3d

def save_image(pcd, out_path):
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False)
    vis.add_geometry(pcd)
    ctr = vis.get_view_control()
    vis.poll_events()
    vis.update_renderer()
    vis.capture_screen_image(out_path)
    vis.destroy_window()

def save_turntable(pcd, out_dir, frames=36):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False)
    vis.add_geometry(pcd)
    ctr = vis.get_view_control()
    for i in range(frames):
        ctr.rotate(10.0, 0.0)
        vis.poll_events()
        vis.update_renderer()
        vis.capture_screen_image(str(Path(out_dir)/f'frame_{i:03d}.png'))
    vis.destroy_window()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pcd', type=str, required=True)
    ap.add_argument('--out', type=str, required=True)
    args = ap.parse_args()
    if args.pcd.endswith('.ply'):
        pcd = o3d.io.read_point_cloud(args.pcd)
    else:
        pts = np.load(args.pcd).astype(np.float32)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
    save_image(pcd, args.out)

if __name__ == '__main__':
    main()