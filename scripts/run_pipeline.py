import json
import os
from pathlib import Path
import cv2
import numpy as np

from src.datasets.unified_dataloader import UnifiedDatasetConfig, UnifiedMultiDataset
from src.models.depth_dpt import DPTDepthPredictor
import torch
from src.inference.depth_to_pcd import depth_to_pointcloud, save_outputs

def ensure_dirs():
    Path('outputs/visualizations').mkdir(parents=True, exist_ok=True)
    Path('outputs').mkdir(parents=True, exist_ok=True)
    Path('results/reconstructions').mkdir(parents=True, exist_ok=True)
    Path('checkpoints/dpt').mkdir(parents=True, exist_ok=True)

def _get_offline_transform():
    try:
        tf = torch.hub.load("intel-isl/MiDaS", "transforms").dpt_transform
        print("[INFO] Using MiDaS official transforms.")
        return tf
    except Exception as e:
        from torchvision import transforms
        print("[WARN] MiDaS transforms failed. Using offline fallback. Error:", e)
        return transforms.Compose([
            transforms.Resize((384, 384)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

def run_inference_across_datasets(max_per_dataset=5):
    cfg = UnifiedDatasetConfig({
        'pix3d': 'data/pix3d/img',
        'shapenet': 'data/shapenet',
        'pascal3d': 'data/pascal3d',
        'objectnet3d': 'data/objectnet3d',
        'co3d': 'data/co3d',
        'google_scanned': 'data/google_scanned'
    })
    datasets = UnifiedMultiDataset(cfg)
    predictor = DPTDepthPredictor(model_name='DPT_Hybrid')
    _ = _get_offline_transform()
    results = []
    for name, it in datasets.iter_images():
        count = 0
        for f in it:
            if count >= max_per_dataset:
                break
            img = cv2.imread(str(f))
            if img is None:
                continue
            depth = predictor.predict_depth(img)
            f = float(max(img.shape[0], img.shape[1]))
            camK = np.array([[f, 0.0, img.shape[1]/2.0],[0.0, f, img.shape[0]/2.0],[0.0, 0.0, 1.0]], dtype=np.float32)
            pts, colors = depth_to_pointcloud(depth, image_bgr=img, camera_K=camK, n_points=8192)
            out_name = f'{name}_{f.stem}'
            save_outputs(pts, colors, 'results/reconstructions', out_name)
            results.append({'dataset': name, 'file': str(f), 'points': len(pts)})
            count += 1
    return results

def main():
    ensure_dirs()
    res = run_inference_across_datasets()
    summary = {
        'status': 'pipeline_initialized',
        'inference_samples': len(res),
        'datasets_seen': sorted(list(set(r['dataset'] for r in res))),
        'results_dir': 'results/reconstructions'
    }
    with open('outputs/results.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print('pipeline run complete')

if __name__ == '__main__':
    main()