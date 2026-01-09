import numpy as np
from pathlib import Path

def poisson_mesh_from_points(points, depth=10):
    try:
        import open3d as o3d
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
        pcd.estimate_normals()
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=int(depth))
        return mesh, np.asarray(densities)
    except Exception:
        return None, None

def save_mesh(mesh, path):
    try:
        import open3d as o3d
        if path.suffix.lower() == '.ply':
            o3d.io.write_triangle_mesh(str(path), mesh)
        elif path.suffix.lower() == '.obj':
            o3d.io.write_triangle_mesh(str(path), mesh)
    except Exception:
        pass