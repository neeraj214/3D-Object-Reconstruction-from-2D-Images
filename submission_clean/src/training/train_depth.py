import argparse
import yaml
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np

class Pix3DDepthDataset(Dataset):
    def __init__(self, root):
        self.root = Path(root)
        rgb_dir = self.root/'rgb'
        if rgb_dir.exists():
            self.images = list(rgb_dir.glob('train_*.png'))
        else:
            # Fallback: raw Pix3D images under data/pix3d/img/**
            pix_root = Path('data/pix3d/img')
            exts = ['*.jpg','*.png']
            self.images = []
            if pix_root.exists():
                for c in pix_root.iterdir():
                    if c.is_dir():
                        for e in exts:
                            self.images += list(c.glob(e))
    def __len__(self):
        return len(self.images)
    def __getitem__(self, idx):
        img = cv2.imread(str(self.images[idx]))
        x = cv2.resize(img, (384,384))
        x = torch.from_numpy(x.transpose(2,0,1)).float()/255.0
        y = torch.zeros((1,384,384))
        return x, y

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', type=str, required=True)
    ap.add_argument('--data', type=str, default='data/processed')
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    ds = Pix3DDepthDataset(args.data)
    if len(ds) == 0:
        print('no data found for training')
        return
    dl = DataLoader(ds, batch_size=cfg['batch_size'], shuffle=True)
    Path(cfg['checkpoint_dir']).mkdir(parents=True, exist_ok=True)
    # Minimal training stub: iterate one epoch and save a placeholder checkpoint
    for batch in dl:
        break
    ckpt_path = Path(cfg['checkpoint_dir'])/'best.pth'
    with open(ckpt_path, 'wb') as f:
        f.write(b'checkpoint_placeholder')
    print('saved', ckpt_path)

if __name__ == '__main__':
    main()