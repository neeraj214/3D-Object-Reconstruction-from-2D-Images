import os
import json
from pathlib import Path
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Ensure project root is on path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.models.depth_dpt import DPTDepthPredictor
from src.inference.depth_to_pcd import depth_to_pointcloud, compute_intrinsics, save_outputs

OUT_DIR = Path('test_results')

def visualize_pointcloud(pts: np.ndarray, out_path: Path):
    fig, axes = plt.subplots(1, 3, figsize=(9, 3))
    axes[0].scatter(pts[:, 0], pts[:, 2], s=0.5, c='k'); axes[0].set_title('front'); axes[0].set_xticks([]); axes[0].set_yticks([])
    axes[1].scatter(pts[:, 1], pts[:, 2], s=0.5, c='k'); axes[1].set_title('side'); axes[1].set_xticks([]); axes[1].set_yticks([])
    axes[2].scatter(pts[:, 0], pts[:, 1], s=0.5, c='k'); axes[2].set_title('top');  axes[2].set_xticks([]); axes[2].set_yticks([])
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=200)
    plt.close(fig)

def make_examples():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    samples = []
    root = Path('data') / 'pix3d' / 'img' / 'bed'
    if root.exists():
        for f in sorted(list(root.glob('*.jpg')))[:3]:
            samples.append(f)
    else:
        raise FileNotFoundError('Sample images not found at data/pix3d/img/bed')

    predictor = DPTDepthPredictor(model_name='DPT_Hybrid')

    results = []
    for idx, img_path in enumerate(samples):
        name = f'example_{idx+1}'
        out_sub = OUT_DIR / name
        out_sub.mkdir(parents=True, exist_ok=True)
        try:
            img = cv2.imread(str(img_path))
            if img is None:
                raise RuntimeError(f'Failed to load image: {img_path}')

            d = predictor.predict_depth(img)
            h, w = img.shape[:2]
            K = np.array(compute_intrinsics(h, w, f_scale=1.0), dtype=np.float32)
            pts, cols = depth_to_pointcloud(d, image_bgr=img, camera_K=K, n_points=40000, mask=None, use_mask=False, denoise=True, gaussian=True, edge_sample=True, upsample_scale=1, smooth=True, sample_name=name)

            # Save assets
            cv2.imwrite(str(out_sub / 'image.jpg'), img)
            vis_path = out_sub / 'pointcloud_views.png'
            visualize_pointcloud(pts, vis_path)
            save_outputs(pts, cols, out_sub, 'point_cloud')
            try:
                # Save depth visualization
                dv = (d * 255.0 / max(float(d.max()), 1e-6)).astype(np.uint8)
                cv2.imwrite(str(out_sub / 'depth.png'), dv)
            except Exception:
                pass

            stats = {
                'name': name,
                'image_path': str(img_path),
                'num_points': int(len(pts)),
                'bounds': {
                    'min': float(np.min(pts)),
                    'max': float(np.max(pts))
                }
            }
            (out_sub / 'pointcloud_stats.json').write_text(json.dumps(stats, indent=2))
            results.append(stats)
        except Exception as e:
            (out_sub / 'error.txt').write_text(str(e))
            continue

    (OUT_DIR / 'summary.json').write_text(json.dumps({'items': results}, indent=2))
    print(f'Examples saved under {OUT_DIR}')

if __name__ == '__main__':
    make_examples()
