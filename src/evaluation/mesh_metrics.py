import numpy as np

def mesh_point_to_surface_distance(mesh_vertices, mesh_faces, points):
    v = np.asarray(mesh_vertices)
    p = np.asarray(points)
    from sklearn.neighbors import KDTree
    kd = KDTree(v)
    d, _ = kd.query(p, k=1)
    return float(np.mean(d))

def mesh_normal_consistency(mesh_normals, gt_normals):
    a = np.asarray(mesh_normals)
    b = np.asarray(gt_normals)
    if len(a) == 0 or len(b) == 0:
        return 0.0
    n = min(len(a), len(b))
    a = a[:n]; b = b[:n]
    cos = (a*b).sum(axis=1)
    return float(np.mean(np.abs(cos)))

def mesh_edge_smoothness(mesh_vertices, mesh_faces):
    v = np.asarray(mesh_vertices)
    f = np.asarray(mesh_faces)
    if len(f) == 0:
        return 0.0
    e = []
    for tri in f:
        a,b,c = tri
        e.append(np.linalg.norm(v[a]-v[b]))
        e.append(np.linalg.norm(v[b]-v[c]))
        e.append(np.linalg.norm(v[c]-v[a]))
    return float(np.std(e))

def mesh_completeness(points_pred, points_gt, thresh=0.01):
    p = np.asarray(points_pred)
    g = np.asarray(points_gt)
    from sklearn.neighbors import KDTree
    kd = KDTree(p)
    d, _ = kd.query(g, k=1)
    return float(np.mean(d < thresh))