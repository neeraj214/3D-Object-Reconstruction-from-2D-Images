import sys
import os
from PIL import Image
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

# Ensure project root is on path
sys.path.append(os.path.abspath('.'))

from src.utils.pointcloud_utils import generate_point_cloud_from_image

def main():
    if len(sys.argv) < 5:
        print("Usage: python tools/generate_pointcloud_for_image.py <image_path> <out_npy> <out_ply> <out_png>")
        sys.exit(1)

    image_path = sys.argv[1]
    out_npy = sys.argv[2]
    out_ply = sys.argv[3]
    out_png = sys.argv[4]

    # Load image
    img = Image.open(image_path).convert('RGB')

    # Generate point cloud
    pc = generate_point_cloud_from_image(img, num_points=2048, device='cpu')

    # Save npy
    os.makedirs(os.path.dirname(out_npy), exist_ok=True)
    np.save(out_npy, pc.numpy())

    # Save PLY
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc.numpy())
    os.makedirs(os.path.dirname(out_ply), exist_ok=True)
    o3d.io.write_point_cloud(out_ply, pcd)

    # Save PNG visualization
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection='3d')
    pts = pc.numpy()
    sc = ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=pts[:, 2], cmap='viridis', s=8, alpha=0.85, edgecolors='black', linewidth=0.05)
    ax.view_init(elev=20, azim=45)
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    cbar = plt.colorbar(sc, shrink=0.7); cbar.set_label('Z')
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    print("Saved:", out_npy, out_ply, out_png)

if __name__ == '__main__':
    main()