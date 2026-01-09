import argparse
from pathlib import Path
import json
import numpy as np
import sys
sys.path.append(str(Path(__file__).resolve().parents[2]))

def load_mesh(path):
    import open3d as o3d
    m = o3d.io.read_triangle_mesh(str(path))
    m.compute_vertex_normals()
    return m

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pred', type=str, required=True)
    ap.add_argument('--gt', type=str, required=True)
    ap.add_argument('--out', type=str, default='results/eval_mesh.json')
    args = ap.parse_args()
    from src.evaluation.mesh_metrics import mesh_point_to_surface_distance, mesh_edge_smoothness, mesh_completeness, mesh_normal_consistency
    mp = load_mesh(args.pred)
    mg = load_mesh(args.gt)
    vp = np.asarray(mp.vertices)
    fp = np.asarray(mp.triangles)
    vg = np.asarray(mg.vertices)
    fg = np.asarray(mg.triangles)
    p2s = mesh_point_to_surface_distance(vg, fg, vp)
    nc = mesh_normal_consistency(np.asarray(mp.vertex_normals), np.asarray(mg.vertex_normals))
    sm = mesh_edge_smoothness(vp, fp)
    comp = mesh_completeness(vp, vg)
    out = {
        'point_to_surface': p2s,
        'normal_consistency': nc,
        'edge_smoothness_std': sm,
        'completeness': comp
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))

if __name__ == '__main__':
    main()