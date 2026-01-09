import json
from pathlib import Path

def _ensure_dir(p):
    Path(p).mkdir(parents=True, exist_ok=True)

def write_dataset_info(dataset_root, name, source_url=None):
    info = {
        'name': name,
        'root': str(dataset_root),
        'source_url': source_url or '',
        'format': {
            'images': 'images/',
            'masks': 'masks/',
            'annotations': 'annotations.json',
            'pointclouds': 'pointclouds/'
        }
    }
    with open(Path(dataset_root)/'dataset_info.json','w') as f:
        json.dump(info, f, indent=2)
    for d in ['images','masks','pointclouds']:
        _ensure_dir(Path(dataset_root)/d)
    ann = Path(dataset_root)/'annotations.json'
    if not ann.exists():
        ann.write_text('[]')

DATASETS = ['shapenet','pascal3d','objectnet3d','co3d','google_scanned']

def ensure_structure(root: Path, name: str, url: str):
    _ensure_dir(root)
    write_dataset_info(str(root), name, url)
    ann = root/'annotations.json'
    if not ann.exists():
        ann.write_text('[]')

def main():
    mapping = {
        'shapenet': ('ShapeNet', 'https://www.shapenet.org/'),
        'pascal3d': ('Pascal3D+', 'http://cvgl.stanford.edu/projects/pascal3d/'),
        'objectnet3d': ('ObjectNet3D', 'http://cvgl.stanford.edu/projects/objectnet3d/'),
        'co3d': ('CO3D', 'https://github.com/facebookresearch/co3d'),
        'google_scanned': ('Google Scanned Objects', 'https://research.google/tools/datasets/')
    }
    for d in DATASETS:
        name, url = mapping[d]
        ensure_structure(Path('data')/d, name, url)
    print('prepared dataset structures')

if __name__ == '__main__':
    main()