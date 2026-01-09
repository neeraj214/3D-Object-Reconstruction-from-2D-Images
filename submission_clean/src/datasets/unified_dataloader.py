import os
from pathlib import Path
from typing import List, Dict

class UnifiedDatasetConfig:
    def __init__(self, roots: Dict[str, str]):
        self.roots = {k: Path(v) for k, v in roots.items()}

class UnifiedImageIterator:
    def __init__(self, root: Path):
        self.root = root
        self.images = []
        for ext in ['*.jpg','*.png']:
            self.images += list(root.rglob(ext))
        self.images = sorted(self.images)
        self.idx = 0
    def __iter__(self):
        return self
    def __next__(self):
        if self.idx >= len(self.images):
            raise StopIteration
        f = self.images[self.idx]
        self.idx += 1
        return f

class UnifiedMultiDataset:
    def __init__(self, cfg: UnifiedDatasetConfig):
        self.cfg = cfg
    def iter_images(self):
        for name, root in self.cfg.roots.items():
            if root.exists():
                yield name, UnifiedImageIterator(root)