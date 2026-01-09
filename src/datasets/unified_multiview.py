import json
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T
from src.training.utils.pointcloud_utils import PointCloudAugmenter, PointCloudNormalizer

def compute_intrinsics(h: int, w: int, f_scale: float = 1.0):
    f = max(h, w) * float(f_scale)
    cx = w / 2.0
    cy = h / 2.0
    return [[f, 0, cx], [0, f, cy], [0, 0, 1.0]]

class UnifiedMultiviewDataset(Dataset):
    def __init__(self, annotations_path: str, split: str = "train", num_points: int = 2048, image_size: int = 224, augment: bool = True, max_items: int = None):
        p = Path(annotations_path)
        all_items = json.loads(p.read_text())

        # Deterministic shuffle for consistent train/val splits
        import random
        random.Random(42).shuffle(all_items)

        split_index = int(len(all_items) * 0.8)

        if split == "train":
            self.items: List[Dict[str, Any]] = all_items[:split_index]
        else: # "val" or any other split will go to validation
            self.items: List[Dict[str, Any]] = all_items[split_index:]
        if max_items is not None:
            self.items = self.items[:int(max_items)]

        self.num_points = num_points
        self.image_size = image_size
        self.augment = augment
        self.tf_train = T.Compose([
            T.Resize((image_size, image_size)),
            T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
            T.RandomHorizontalFlip(),
            T.RandomApply([T.GaussianBlur(kernel_size=(3,3), sigma=(0.1, 2.0))], p=0.4),
            T.RandomAdjustSharpness(sharpness_factor=1.5, p=0.4),
            T.ToTensor(),
            T.RandomErasing(p=0.15, scale=(0.02, 0.2), ratio=(0.3, 3.3)),
            T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
        ])
        self.tf_eval = T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
        ])

    def __len__(self):
        return len(self.items)

    def _load_image(self, path: str):
        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            w, h = self.image_size, self.image_size
            arr = np.zeros((h, w, 3), dtype=np.uint8)
            for y in range(h):
                arr[y, :, 0] = int(255*y/h)
                arr[y, :, 1] = int(255*(1 - y/h))
                arr[y, :, 2] = 128
            img = Image.fromarray(arr)
        tf = self.tf_train if self.augment else self.tf_eval
        return tf(img)

    def _augment_pointcloud(self, pts: np.ndarray):
        if not self.augment:
            return pts
        pts = PointCloudAugmenter.random_rotation(pts)
        pts = PointCloudAugmenter.random_scale(pts)
        pts = PointCloudAugmenter.random_jitter(pts, std=0.01, clip=0.02)
        if len(pts) > self.num_points:
            idx = np.random.choice(len(pts), self.num_points, replace=False)
            pts = pts[idx]
        elif len(pts) < self.num_points:
            rep = np.random.choice(len(pts), self.num_points - len(pts), replace=True)
            pts = np.vstack([pts, pts[rep]])
        pts, _ = PointCloudNormalizer.normalize_pointcloud(pts)
        return pts

    def __getitem__(self, idx):
        item = self.items[idx]
        f = self._load_image(item.get("image_front"))
        s = self._load_image(item.get("image_side")) if item.get("image_side") else f
        mk = None
        if item.get("mask"):
            try:
                mk_img = Image.open(item["mask"]).convert("L")
                mk_img = mk_img.resize((self.image_size, self.image_size), resample=Image.NEAREST)
                mk = torch.from_numpy(np.array(mk_img)).float()
            except Exception:
                mk = None
        try:
            pc = np.load(item["point_cloud_gt"]).astype(np.float32)
        except Exception:
            n = max(128, self.num_points)
            pc = np.random.randn(n, 3).astype(np.float32)
            pc = pc / max(np.linalg.norm(pc, axis=1).max(), 1e-8)
        pc = self._augment_pointcloud(pc)
        pc_t = torch.from_numpy(pc).float().permute(1,0)
        mv_t = None
        mf_t = None
        if item.get("mesh_gt"):
            try:
                import open3d as o3d
                m = o3d.io.read_triangle_mesh(str(item["mesh_gt"]))
                v = np.asarray(m.vertices).astype(np.float32)
                ftri = np.asarray(m.triangles).astype(np.int64)
                mv_t = torch.from_numpy(v).float()
                mf_t = torch.from_numpy(ftri).long()
            except Exception:
                mv_t = None
                mf_t = None
        camK_t = None
        cam = item.get("camera_params", {})
        try:
            K = cam.get("K", None)
            if K is None and isinstance(cam, dict):
                K = cam.get("camera_K", None)
            if K is not None:
                camK_t = torch.from_numpy(np.array(K, dtype=np.float32))
        except Exception:
            camK_t = None
        if camK_t is None:
            try:
                h, w = int(f.size(1)), int(f.size(2))
                Kd = compute_intrinsics(h, w, f_scale=1.0)
                camK_t = torch.from_numpy(np.array(Kd, dtype=np.float32))
            except Exception:
                camK_t = None
        out = {"front": f, "side": s, "pointcloud": pc_t, "category": item.get("category","unknown")}
        if mk is not None:
            out["mask"] = mk
        if camK_t is not None:
            out["camera_K"] = camK_t
        if mv_t is not None:
            out["mesh_vertices"] = mv_t
        if mf_t is not None:
            out["mesh_faces"] = mf_t
        return {k: v for k, v in out.items() if v is not None}
