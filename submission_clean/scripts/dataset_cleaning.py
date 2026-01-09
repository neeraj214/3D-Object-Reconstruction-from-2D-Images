import os
import json
from pathlib import Path
import cv2
import numpy as np
import open3d as o3d

DATASETS = ['pix3d','shapenet','pascal3d','objectnet3d','co3d','google_scanned']

def validate_and_clean(root: Path):
    report = {'root': str(root), 'removed': [], 'corrupted': [], 'missing': []}
    if not root.exists():
        report['missing'].append('root')
        return report
    imgs = []
    for ext in ['*.jpg','*.png','*.jpeg']:
        imgs += list(root.rglob(ext))
    for p in imgs:
        try:
            img = cv2.imread(str(p))
            if img is None or img.size == 0:
                report['corrupted'].append(str(p))
                try:
                    p.unlink()
                    report['removed'].append(str(p))
                except Exception:
                    pass
        except Exception:
            report['corrupted'].append(str(p))
    return report

def convert_meshes_to_pointclouds(root: Path, out_pc: Path, n_points=50000):
    out_pc.mkdir(parents=True, exist_ok=True)
    mesh_paths = []
    for ext in ['*.obj','*.ply','*.stl','*.off']:
        mesh_paths += list(root.rglob(ext))
    converted = []
    for mp in mesh_paths:
        try:
            mesh = o3d.io.read_triangle_mesh(str(mp))
            if mesh.is_empty():
                continue
            mesh.compute_vertex_normals()
            pcd = mesh.sample_points_poisson_disk(n_points)
            name = mp.stem
            out_npy = out_pc/f'{name}.npy'
            out_ply = out_pc/f'{name}.ply'
            np.save(out_npy, np.asarray(pcd.points))
            o3d.io.write_point_cloud(str(out_ply), pcd)
            converted.append(str(out_ply))
        except Exception:
            continue
    return converted

def write_dataset_info(root: Path, info_path: Path, extras: dict):
    info = {'root': str(root)}
    info.update(extras)
    with open(info_path,'w') as f:
        json.dump(info, f, indent=2)

def main():
    base = Path('data')
    base.mkdir(parents=True, exist_ok=True)
    lines = []
    for name in DATASETS:
        root = base/name
        report = validate_and_clean(root)
        extras = {}
        if name == 'shapenet':
            pcs_out = root/'pointclouds'
            converted = convert_meshes_to_pointclouds(root, pcs_out, 50000)
            extras['converted_meshes'] = converted
        info_path = root/'dataset_info.json'
        info_path.parent.mkdir(parents=True, exist_ok=True)
        write_dataset_info(root, info_path, extras)
        lines.append(f"Dataset: {name}\nRoot: {root}\nMissing: {report['missing']}\nCorrupted: {len(report['corrupted'])}\nRemoved: {len(report['removed'])}\nConverted: {len(extras.get('converted_meshes', []))}\n\n")
    with open(base/'dataset_cleaning_report.md','w') as f:
        f.write('\n'.join(lines))

if __name__ == '__main__':
    main()