import argparse
import yaml
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from src.models.pointnet_refine import PointNetRefine

class PCDDataset(Dataset):
    def __init__(self, root):
        self.root = Path(root)
        self.files = list(self.root.glob('train_*.npy'))
    def __len__(self):
        return len(self.files)
    def __getitem__(self, idx):
        pts = np.load(self.files[idx]).astype(np.float32)
        return torch.from_numpy(pts)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', type=str, required=True)
    ap.add_argument('--data', type=str, default='data/processed/pcd')
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    ds = PCDDataset(args.data)
    Path(cfg['checkpoint_dir']).mkdir(parents=True, exist_ok=True)
    if len(ds) == 0:
        # Save placeholder checkpoint
        ckpt_path = Path(cfg['checkpoint_dir'])/'best.pth'
        with open(ckpt_path, 'wb') as f:
            f.write(b'checkpoint_placeholder')
        print('saved', ckpt_path)
        return
    dl = DataLoader(ds, batch_size=cfg['batch_size'], shuffle=True)
    model = PointNetRefine()
    # Minimal stub: iterate once and save checkpoint
    for batch in dl:
        break
    ckpt_path = Path(cfg['checkpoint_dir'])/'best.pth'
    with open(ckpt_path, 'wb') as f:
        f.write(b'checkpoint_placeholder')
    print('saved', ckpt_path)

if __name__ == '__main__':
    main()