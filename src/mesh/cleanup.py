import numpy as np

def filter_by_density(mesh, densities, keep_ratio=0.7):
    try:
        import open3d as o3d
        thr = np.quantile(densities, 1.0 - (1.0 - keep_ratio))
        verts_to_keep = densities >= thr
        mesh.remove_vertices_by_mask(~verts_to_keep)
        return mesh
    except Exception:
        return mesh

def largest_component(mesh):
    try:
        import open3d as o3d
        labels = np.array(mesh.cluster_connected_triangles()[0])
        counts = np.bincount(labels)
        if len(counts) == 0:
            return mesh
        largest = counts.argmax()
        tris_to_keep = labels == largest
        mesh.remove_triangles_by_mask(~tris_to_keep)
        mesh.remove_unreferenced_vertices()
        return mesh
    except Exception:
        return mesh

def laplacian_smooth(mesh, iterations=15):
    try:
        import open3d as o3d
        mesh = mesh.filter_smooth_simple(number_of_iterations=int(iterations))
        return mesh
    except Exception:
        return mesh

def refine_normals(mesh):
    try:
        import open3d as o3d
        mesh.compute_vertex_normals()
        mesh.orient_triangles()
        return mesh
    except Exception:
        return mesh

def simplify(mesh, target_faces=15000):
    try:
        import open3d as o3d
        mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=int(target_faces))
        return mesh
    except Exception:
        return mesh