import numpy as np
from pathlib import Path

def turntable_images(mesh_vertices, mesh_faces, out_dir, frames=60):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    try:
        import open3d as o3d
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(np.asarray(mesh_vertices))
        mesh.triangles = o3d.utility.Vector3iVector(np.asarray(mesh_faces))
        mesh.compute_vertex_normals()
        vis = o3d.visualization.Visualizer()
        vis.create_window(visible=False)
        vis.add_geometry(mesh)
        ctr = vis.get_view_control()
        for i in range(frames):
            ctr.rotate(10.0, 0.0)
            img = vis.capture_screen_float_buffer(False)
            arr = (np.asarray(img)*255).astype(np.uint8)
            from PIL import Image
            Image.fromarray(arr).save(out/f"frame_{i:03d}.png")
        vis.destroy_window()
    except Exception:
        pass
    return str(out)