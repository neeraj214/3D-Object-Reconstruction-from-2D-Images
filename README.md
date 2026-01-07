# 3D Object Reconstruction from 2D Images

![Python](https://img.shields.io/badge/Python-3.12-blue?style=for-the-badge&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c?style=for-the-badge&logo=pytorch)
![React](https://img.shields.io/badge/React-18-61dafb?style=for-the-badge&logo=react)
![FastAPI](https://img.shields.io/badge/FastAPI-0.95+-009688?style=for-the-badge&logo=fastapi)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Active-success?style=for-the-badge)

> **Single-View 3D Reconstruction using Deep Learning and Point Clouds.**

## 1. Project Overview

This project implements a state-of-the-art deep learning pipeline capable of reconstructing 3D geometry (specifically point clouds) from single-view 2D images. By bridging the gap between 2D visual perception and 3D spatial understanding, this system enables applications in Augmented Reality (AR), Virtual Reality (VR), robotics navigation, and 3D content creation.

The solution features a hybrid neural architecture combining Convolutional Neural Networks (CNNs) and Vision Transformers (ViTs) to capture both local textures and global geometric context, delivered through a modern, interactive web interface.

## 2. Key Features

*   **Single-View Reconstruction:** Generates detailed 3D point clouds from a single RGB input image.
*   **Hybrid Encoder-Decoder:** Utilizes ResNet50/ViT backbones with Transformer-based decoders for robust feature extraction.
*   **Multi-Representation Support:** Configurable architecture supporting Point Clouds, NeRF, SDF, and Occupancy networks.
*   **Interactive Web UI:** Drag-and-drop interface with real-time 3D visualization using React and Three.js.
*   **Unified Data Pipeline:** Seamlessly handles diverse datasets like Pix3D, ShapeNet, and Pascal3D+.
*   **Real-Time Inference:** Optimized FastAPI backend for low-latency predictions.

## 3. Technology Stack

### Core AI/ML
*   **Frameworks:** PyTorch, PyTorch Lightning, PyTorch3D
*   **Libraries:** Open3D, NumPy, SciPy, Timm (PyTorch Image Models)
*   **Computer Vision:** OpenCV, Pillow

### Backend
*   **Server:** FastAPI, Uvicorn
*   **Processing:** Python 3.12

### Frontend
*   **Framework:** React 18 (Vite)
*   **Styling:** Tailwind CSS
*   **Visualization:** Three.js, React-Three-Fiber
*   **HTTP Client:** Axios

## 4. Project Architecture

The system follows an end-to-end learning approach:

1.  **Input Processing:** Input image (224x224) is normalized and augmented.
2.  **Feature Encoding:**
    *   **CNN Branch (ResNet50):** Extracts local texture and shape features.
    *   **Transformer Branch (ViT):** Captures global context and long-range dependencies.
3.  **Feature Fusion:** Cross-attention mechanisms align 2D image features with 3D geometry queries.
4.  **Geometry Decoding:** A coordinate-based decoder predicts (x, y, z) positions for N points (default: 2048/4096).
5.  **Refinement:** Optional post-processing for outlier removal and smoothing.
6.  **Visualization:** The point cloud is rendered interactively in the frontend.

## 5. Installation & Setup

### Prerequisites
*   Python 3.12+
*   Node.js 18+
*   CUDA-compatible GPU (recommended for training)

### Backend Setup
1.  Clone the repository:
    ```bash
    git clone https://github.com/yourusername/3d-reconstruction.git
    cd 3d-reconstruction
    ```
2.  Create a virtual environment:
    ```bash
    python -m venv .venv
    # Windows
    .\.venv\Scripts\activate
    # Linux/Mac
    source .venv/bin/activate
    ```
3.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
4.  Start the server:
    ```bash
    python -m uvicorn server.api:app --host 0.0.0.0 --port 8000 --reload
    ```

### Frontend Setup
1.  Navigate to the frontend directory:
    ```bash
    cd frontend
    ```
2.  Install dependencies:
    ```bash
    npm install
    ```
3.  Start the development server:
    ```bash
    npm run dev
    ```
    Access the UI at `http://localhost:5173`.

## 6. Dataset Details

The project utilizes a **Unified Dataset** structure, supporting:
*   **Pix3D:** Real-world images aligned with 3D CAD models.
*   **ShapeNet:** Large-scale synthetic dataset.
*   **Pascal3D+ / ObjectNet3D:** additional benchmarks.

**Data Structure:**
```
data/
  ├── pix3d/
  │   ├── img/
  │   ├── model/
  │   └── pix3d.json
  └── unified/
      ├── annotations.json
      └── images/
```

## 7. Performance Metrics

Model performance is evaluated using standard 3D reconstruction metrics (lower is better for CD, higher for F-Score).

| Metric | Value (Approx) | Description |
| :--- | :--- | :--- |
| **F-Score @ 1%** | 45.2% | Accuracy at strict threshold |
| **F-Score @ 10%** | 77.18% | Accuracy at loose threshold |
| **Chamfer Distance (CD)** | 0.058 | Average distance between nearest points |
| **IoU** | 0.65 | Volumetric intersection over union |

*Note: Metrics based on evaluation on the Pix3D test split (Epoch 4).*

## 8. Project Status

![Status](https://img.shields.io/badge/Status-In%20Progress-yellow)

*   [x] **Phase 1:** Baseline Model & Data Pipeline (Completed)
*   [x] **Phase 2:** Web Interface Integration (Completed)
*   [x] **Phase 3:** Enhanced Training & Mixed Precision (Completed)
*   [ ] **Phase 4:** Mesh Generation & Texture Mapping (Experimental)
*   [ ] **Phase 5:** Real-Time Video Reconstruction (Planned)

## 9. Folder Structure

```
3d-reconstruction/
├── configs/                # Training configurations
├── data/                   # Datasets (Pix3D, ShapeNet, etc.)
├── frontend/               # React Web Application
│   ├── src/
│   └── public/
├── results/                # Training outputs & checkpoints
├── samples/                # Sample images & outputs
├── scripts/                # Utility scripts (generation, eval)
├── server/                 # FastAPI backend
│   └── api.py
├── src/                    # Core Source Code
│   ├── datasets/           # Data loaders
│   ├── models/             # PyTorch model definitions
│   ├── training/           # Training loops & trainers
│   └── inference/          # Inference pipelines
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
```

## 10. Author

**Neeraj**
*   [GitHub Profile](https://github.com/yourusername)

---
*Built with ❤️ using PyTorch and React.*


<div align="center">
  <sub>Built with ❤️ by <a href="https://github.com/neeraj214">Neeraj Negi</a> as part of the MCA Curriculum.</sub>
</div>
