import json
from pathlib import Path
import cv2
import numpy as np
import open3d as o3d

DATASETS = {
    'pix3d': {'required': ['pix3d.json','img','mask','model']},
    'shapenet': {'required': []},
    'pascal3d': {'required': ['Annotations','Images']},
    'objectnet3d': {'required': ['images','annotations']},
    'co3d': {'required': []},
    'google_scanned': {'required': ['meshes']},
}

def validate_dataset(name, root: Path):
    info = {'name': name, 'root': str(root), 'missing': [], 'counts': {}}
    reqs = DATASETS.get(name, {}).get('required', [])
    for r in reqs:
        if not (root/r).exists():
            info['missing'].append(r)
    imgs = []
    for ext in ['*.jpg','*.png','*.jpeg']:
        imgs += list((root).rglob(ext))
    info['counts']['images'] = len(imgs)
    masks = list((root/'mask').rglob('*.png')) if (root/'mask').exists() else []
    info['counts']['masks'] = len(masks)
    meshes = []
    for ext in ['*.obj','*.ply','*.stl','*.off']:
        meshes += list((root).rglob(ext))
    info['counts']['meshes'] = len(meshes)
    cams = list(root.rglob('camera_K.npy'))
    info['counts']['camera_K'] = len(cams)
    return info

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
            out_ply = out_pc/f'{name}.ply'
            o3d.io.write_point_cloud(str(out_ply), pcd)
            converted.append(str(out_ply))
        except Exception:
            continue
    return converted

def run():
    root = Path('data')
    report = {'datasets': {}, 'converted': {}}
    for name in DATASETS.keys():
        ds_root = root/name
        if not ds_root.exists():
            report['datasets'][name] = {'missing_root': True}
            continue
        info = validate_dataset(name, ds_root)
        report['datasets'][name] = info
        if name in ('shapenet','google_scanned'):
            out_pc = ds_root/'pointclouds'
            conv = convert_meshes_to_pointclouds(ds_root, out_pc, n_points=50000)
            report['converted'][name] = conv
    Path('data').mkdir(parents=True, exist_ok=True)
    with open(root/'dataset_info.json','w') as f:
        json.dump(report, f, indent=2)
    lines = []
    for name, info in report['datasets'].items():
        lines.append(f"## {name}\n")
        lines.append(json.dumps(info, indent=2)+"\n")
    with open(root/'dataset_cleaning_report.md','w') as f:
        f.write('\n'.join(lines))

if __name__ == '__main__':
    run()