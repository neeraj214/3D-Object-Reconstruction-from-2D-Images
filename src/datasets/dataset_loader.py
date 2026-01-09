import json
from pathlib import Path
from typing import Dict, Any, List

class StandardDataset:
    def __init__(self, root: str):
        self.root = Path(root)
        self.images_dir = self.root/'images'
        self.masks_dir = self.root/'masks'
        self.pcd_dir = self.root/'pointclouds'
        ann_path = self.root/'annotations.json'
        if ann_path.exists():
            self.ann = json.loads(ann_path.read_text())
        else:
            self.ann = []
        if not self.ann:
            files = []
            for ext in ['*.jpg','*.png']:
                files += list(self.images_dir.glob(ext))
            self.ann = [{'image': f.name} for f in files]

    def __len__(self):
        return len(self.ann)

    def samples(self) -> List[Dict[str, Any]]:
        out = []
        for a in self.ann:
            img = self.images_dir/a['image']
            m = a.get('mask', None)
            mask = str(self.masks_dir/m) if m else None
            pc = a.get('pointcloud', None)
            pcd = str(self.pcd_dir/pc) if pc else None
            out.append({'image': str(img), 'mask': mask, 'pointcloud': pcd})
        return out

def register_datasets() -> Dict[str, StandardDataset]:
    roots = {
        'pix3d': 'data/pix3d',
        'shapenet': 'data/shapenet',
        'pascal3d': 'data/pascal3d',
        'objectnet3d': 'data/objectnet3d',
        'co3d': 'data/co3d',
        'google_scanned': 'data/google_scanned'
    }
    reg = {}
    for name, r in roots.items():
        rp = Path(r)
        if rp.exists():
            reg[name] = StandardDataset(r)
    return reg