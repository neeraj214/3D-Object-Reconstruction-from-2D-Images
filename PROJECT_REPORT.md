# 3D Object Reconstruction from 2D Images - Project Report

## 1. Project Overview
**Title:** 3D Object Reconstruction from 2D Images  
**Objective:** To develop a deep learning system capable of reconstructing 3D geometry (point clouds/meshes) from single-view 2D images. The system aims to bridge the gap between 2D perception and 3D understanding using advanced neural architectures.

## 2. Technical Architecture

### 2.1 Frontend
The user interface is built as a modern Single Page Application (SPA).
*   **Framework:** React 18 (via Vite)
*   **3D Visualization:** Three.js (raw) for rendering point clouds and meshes.
*   **Styling:** Tailwind CSS for responsive and modern UI components.
*   **Animations:** Framer Motion for smooth transitions.
*   **API Communication:** Axios for handling HTTP requests to the backend.
*   **Key Features:**
    *   Drag-and-drop image upload.
    *   Real-time training status monitoring.
    *   Interactive 3D viewer (Orbit controls, Point size adjustment).

### 2.2 Backend
The backend serves as the inference engine and training controller.
*   **Framework:** FastAPI (High-performance async web framework).
*   **Server:** Uvicorn (ASGI server).
*   **Image Processing:** OpenCV (cv2), Pillow.
*   **3D Processing:** Open3D (Point cloud manipulation), Pyntcloud.
*   **ML Engine:** PyTorch (Deep Learning).

## 3. Machine Learning & Deep Learning Specifications

### 3.1 Model Architecture: Enhanced 3D Reconstruction Model
The core model utilizes a hybrid approach combining 2D feature extraction with 3D geometry prediction.

*   **Encoder:** 
    *   Configurable backbone (CNN or Vision Transformer/ViT).
    *   **Multi-Scale Feature Extractor:** Captures details at various resolutions (Feature Pyramid Network style) to preserve both high-level semantics and low-level textures.
*   **Attention Mechanisms:**
    *   **Multi-Head Attention:** Self-attention within feature maps.
    *   **Cross-Attention:** Links 2D image features with 3D queries to align geometry with visual cues.
*   **Decoders (Modular):**
    *   **NeRF Decoder:** Neural Radiance Fields for view synthesis (Density + Color prediction).
    *   **SDF Decoder:** Signed Distance Function for implicit surface reconstruction.
    *   **Occupancy Decoder:** Predicts probability of occupancy at 3D coordinates.
    *   **Point Cloud Predictor:** Direct regression of 3D point coordinates.

### 3.2 Depth Estimation
*   **DPT (Dense Prediction Transformer):** Utilized for initial depth map estimation from input images, providing a strong geometric prior for the reconstruction task.

### 3.3 Training Pipeline (`EnhancedTrainer`)
*   **Optimization:** 
    *   **Optimizer:** AdamW (Adaptive Moment Estimation with Weight Decay).
    *   **Scheduler:** OneCycleLR / CosineAnnealingWarmRestarts for learning rate scheduling.
*   **Mixed Precision:** Implemented using `torch.cuda.amp` (GradScaler) to reduce memory usage and speed up training on supported hardware.
*   **Loss Functions (`AdvancedLossFunction`):**
    *   **Geometric Losses:** 
        *   **Chamfer Distance:** Measures similarity between predicted and ground truth point clouds.
        *   **EMD (Earth Mover's Distance):** Optimal transport metric for point distributions.
        *   **Normal Consistency Loss:** Aligns surface normals.
    *   **Implicit Losses:** 
        *   **SDF / Occupancy Loss:** Binary Cross Entropy / MSE for implicit fields.
    *   **Visual Losses:**
        *   **NeRF Loss:** Color reconstruction loss + Depth consistency.
        *   **Silhouette Loss:** IoU (Intersection over Union) of 2D projections.
    *   **Regularization:** Laplacian smoothness, Uncertainty regularization.
*   **Data Augmentation:** 3D rotations, scaling, and jittering implemented in `DataAugmentation3D`.

### 3.4 Data Pipeline
*   **Dataset:** Pix3D (Real images aligned with 3D CAD models).
*   **Preprocessing:** 
    *   Image Resizing (224x224).
    *   Normalization (ImageNet stats).
    *   Point Cloud Sampling (Farthest Point Sampling).
*   **Splitting:** 80% Training, 20% Validation (Deterministic shuffle).

## 4. Performance Metrics
*   **Primary Metric:** F-Score@10%
    *   **Current Performance:** ~77.18%
    *   **Definition:** Harmonic mean of Precision and Recall at a distance threshold of 10% of the object's diameter.
*   **Secondary Metrics:**
    *   **Chamfer Distance:** Lower is better (measures geometric error).
    *   **IoU (Intersection over Union):** Volumetric overlap accuracy.

## 5. Visualizations & Review Assets
The project generates professional-grade assets for review and presentation:
*   **Model Architecture Diagram:** Visualizes the flow from Input -> Encoder -> Attention -> Decoder -> 3D Output.
*   **Loss Curves:** Training vs. Validation loss over epochs to demonstrate convergence and check for overfitting.
*   **Model Comparison:** Benchmarking against baselines using Accuracy and F1-Score.
*   **Confusion Matrix:** Analysis of classification performance (if applicable).
*   **Distribution Histograms:** Analysis of dataset properties or error distributions.
*   **3D Examples:** Visual comparisons of Input Image vs. Reconstructed Point Cloud.

## 6. Project Status
*   **Current State:** Training Phase (Epoch 3 ongoing).
*   **Functionality:** 
    *   Full End-to-End Pipeline (Upload -> Process -> Visualize).
    *   Presentation Asset Generation Script (`scripts/build_ml_presentation_visuals.py`).
    *   Metrics Reporting System.
