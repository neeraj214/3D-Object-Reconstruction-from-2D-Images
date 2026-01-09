import json
from pathlib import Path

def main():
    deleted = [
        'checkpoints/ (old)',
        'results/ (old)',
        'models/best_model.pth',
        'corrected_hybrid_model.pth',
        'models/training_results.json'
    ]
    created = [p.as_posix() for p in Path('.').rglob('README.txt')]
    summary = {
        "status": "ready",
        "deleted": deleted,
        "created": [
            "requirements.txt",
            "README.md",
            "scripts/data_download.py",
            "src/datasets/prepare_pix3d.py",
            "src/datasets/preprocess_images.py",
            "src/models/depth_dpt.py",
            "configs/dpt_config.yaml",
            "src/inference/depth_to_pcd.py",
            "src/models/pointnet_refine.py",
            "configs/pointnet_config.yaml",
            "src/training/train_depth.py",
            "src/training/train_pointnet.py",
            "src/utils/metrics.py",
            "src/eval/evaluate_reconstructions.py",
            "src/inference/visualize.py",
            "configs/inference_config.yaml",
            "checkpoints/dpt/",
            "checkpoints/pointnet/",
            "logs/",
            "results/reconstructions/",
            "outputs/visualizations/",
            "src/datasets/unified_dataloader.py",
            "scripts/run_pipeline.py"
        ],
        "next_steps": [
            "run prepare_pix3d",
            "train depth model",
            "convert depth->pcd",
            "train refine model (optional)"
        ]
    }
    Path('outputs').mkdir(parents=True, exist_ok=True)
    with open('outputs/results.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print('wrote outputs/results.json')

if __name__ == '__main__':
    main()