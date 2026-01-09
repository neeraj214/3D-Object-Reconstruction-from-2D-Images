# 3D Object Reconstruction from 2D Images

## Note

Due to the 1 GB upload limit, full  datasets (Pix3D, ShapeNet, ObjectNet3D),
training checkpoints, and cache files have been removed from  this submission.

Only the complete source code, environment files, and sample outputs are  included.

Instructions to download required datasets and pretrained weights are  provided
inside README.md for full  reproducibility.

## Installation

- Python 3.12 recommended
- Create a virtual environment and install Python dependencies:

```
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

- Frontend setup:

```
cd frontend
npm install
```

## Dataset Download Links

- Pix3D: https://github.com/xingyuansun/pix3d
- ShapeNet: https://www.shapenet.org/
- ObjectNet3D: https://cvgl.stanford.edu/projects/objectnet3d/
- Pascal3D+: https://cvgl.stanford.edu/projects/pascal3d.html
- CO3D (Common Objects in 3D): https://github.com/facebookresearch/co3d
- Google Scanned Objects: https://github.com/google-research-datasets/ScannedObjects

Place datasets under `data/` with the expected structure, e.g.:

```
data/
  pix3d/
    img/
      bed/
        0001.png ...
```

## Pretrained Weights

- MiDaS DPT Hybrid weights are cached automatically by torch hub.
- Other model checkpoints should be placed under `checkpoints/` (create as needed), for training resume or evaluation.

## Running Backend and Frontend

- Start backend (FastAPI):

```
python -m uvicorn server.api:app --host 0.0.0.0 --port 8000
```

- Start frontend (Vite):

```
cd frontend
npm run dev
```

Open `http://localhost:5173` and upload an image to trigger `/api/reconstruct`.

## Training

- Example training script (multiview pipeline):

```
python src/training/train_multiview.py --annotations path/to/annotations.json --out results/checkpoints_multiview --epochs1 10 --epochs2 20
```

- Enhanced trainer (single-view point cloud):

```
python -c "from src.training.enhanced_trainer import create_training_config, EnhancedTrainer; cfg=create_training_config(); t=EnhancedTrainer(cfg); print('Trainer ready')"
```

## Inference

- Depth → Point Cloud (DPT + lifting):

```
python scripts/generate_ppt_pointcloud_examples.py
```

- Model-based point cloud (multiview):

```
python src/inference/multiview_inference.py --front path/to/front.jpg --side path/to/side.jpg --ckpt path/to/checkpoint.pth --out results/final_inference
```

## Included Samples

- `samples/images/` contains 5 small sample images
- `samples/outputs/point_cloud_sample.ply` contains a tiny demo point cloud

## Zip Directory Structure

The submission includes only source code, configs, scripts, and light samples:

```
submission_clean/
  README.md
  requirements.txt
  Dockerfile.backend
  Dockerfile.frontend
  configs/
  scripts/
  server/
    api.py
    static/ (no outputs)
  src/
    ... (all Python source)
  frontend/
    index.html
    package.json
    package-lock.json
    vite.config.js
    tailwind.config.cjs
    postcss.config.cjs
    src/
      ... (React source)
    public/
      (no datasets, no heavy outputs)
  samples/
    images/
      0042.jpg
      0043.jpg
      0044.jpg
      0045.jpg
      0046.jpg
    outputs/
      point_cloud_sample.ply
      README.txt
```

## Notes

- No full datasets or checkpoints are included to keep the archive under 1 GB.
- To reproduce full experiments, restore datasets under `data/` and checkpoints under `checkpoints/`, then run training/inference as above.
