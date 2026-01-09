"""
Final Inference and Visualization Script
Performs complete 2D-to-3D reconstruction and classification pipeline with visualization
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image
import torchvision.transforms as transforms
from typing import Optional, Tuple
import os
import sys

# Add src to path to enable imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.models.hybrid_classifier import HybridClassifier
from src.utils.pointcloud_utils import generate_point_cloud_from_image
import numpy as np
import open3d as o3d

def load_point_cloud_for_inference(pointcloud_path: str, num_points: int = 1024) -> torch.Tensor:
    """
    Load point cloud for inference from .npy file.
    
    Args:
        pointcloud_path: Path to .npy file containing point cloud
        num_points: Number of points to sample
        
    Returns:
        Point cloud tensor of shape (num_points, 3)
    """
    try:
        # Load numpy array
        pointcloud_array = np.load(pointcloud_path)
        
        # Handle different array shapes
        if len(pointcloud_array.shape) == 1:
            # Flat array, reshape to (N, 3)
            num_total_points = len(pointcloud_array) // 3
            pointcloud_array = pointcloud_array[:num_total_points * 3].reshape(-1, 3)
        elif len(pointcloud_array.shape) == 3:
            # 3D array, take first batch if needed
            if pointcloud_array.shape[0] == 1:
                pointcloud_array = pointcloud_array[0]
            else:
                pointcloud_array = pointcloud_array[0]  # Take first sample
        
        # Ensure we have exactly num_points
        if len(pointcloud_array) > num_points:
            # Randomly sample points
            indices = np.random.choice(len(pointcloud_array), num_points, replace=False)
            pointcloud_array = pointcloud_array[indices]
        elif len(pointcloud_array) < num_points:
            # Duplicate points with small noise
            if len(pointcloud_array) > 0:
                additional_points = pointcloud_array[np.random.choice(len(pointcloud_array), num_points-len(pointcloud_array), replace=True)]
                noise = np.random.normal(0, 0.001, additional_points.shape)
                pointcloud_array = np.vstack([pointcloud_array, additional_points + noise])
        
        # Normalize the point cloud
        # Center at origin
        centroid = np.mean(pointcloud_array, axis=0)
        pointcloud_array = pointcloud_array - centroid
        
        # Scale to unit sphere
        max_distance = np.max(np.linalg.norm(pointcloud_array, axis=1))
        if max_distance > 0:
            pointcloud_array = pointcloud_array / max_distance
        
        # Convert to torch tensor
        pointcloud_tensor = torch.from_numpy(pointcloud_array.astype(np.float32))
        
        return pointcloud_tensor
        
    except Exception as e:
        print(f"Error loading point cloud from {pointcloud_path}: {e}")
        # Return random point cloud as fallback
        return torch.randn(num_points, 3)

def show_final_prediction(image_path: str, gt_model_path: str, class_names: list, trained_model_path: str) -> Tuple[plt.Figure, str, float]:
    """
    Perform complete inference pipeline and create visualization.
    
    Args:
        image_path: Path to input 2D image
        gt_model_path: Path to ground truth 3D model (.npy file)
        class_names: List of class names for classification
        trained_model_path: Path to trained HybridClassifier model
        
    Returns:
        Tuple of (matplotlib figure, predicted class name, confidence score)
    """
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load and preprocess the 2D image
    print(f"Loading image from: {image_path}")
    image = Image.open(image_path).convert('RGB')
    
    # Image preprocessing
    image_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    image_tensor = image_transform(image).unsqueeze(0).to(device)
    print(f"Image tensor shape: {image_tensor.shape}")
    
    # Generate point cloud from 2D image ("Ours")
    print("Generating point cloud from 2D image...")
    generated_pointcloud = generate_point_cloud_from_image(
        image, 
        num_points=1024, 
        device=str(device)
    )
    print(f"Generated point cloud shape: {generated_pointcloud.shape}")
    
    # Load ground truth point cloud
    print(f"Loading ground truth from: {gt_model_path}")
    gt_pointcloud = load_point_cloud_for_inference(gt_model_path, num_points=1024)
    gt_pointcloud = gt_pointcloud.to(device)
    print(f"Ground truth point cloud shape: {gt_pointcloud.shape}")
    
    # Load trained model
    print(f"Loading trained model from: {trained_model_path}")
    model = HybridClassifier(num_classes=len(class_names))
    
    # Load model weights
    if os.path.exists(trained_model_path):
        checkpoint = torch.load(trained_model_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print("Model weights loaded successfully")
    else:
        print(f"Warning: Model file not found at {trained_model_path}. Using random weights.")
    
    model.to(device)
    model.eval()
    
    # Perform classification
    print("Performing classification...")
    with torch.no_grad():
        # Prepare point cloud for model (add batch dimension and ensure correct shape)
        if len(generated_pointcloud.shape) == 2:
            generated_pointcloud = generated_pointcloud.unsqueeze(0)
        
        # Ensure point cloud is on correct device and has correct shape
        generated_pointcloud = generated_pointcloud.to(device)
        
        # Forward pass through model
        logits = model(image_tensor, generated_pointcloud)
        probabilities = torch.softmax(logits, dim=1)
        
        # Get prediction and confidence
        confidence, predicted_idx = torch.max(probabilities, dim=1)
        predicted_class = class_names[predicted_idx.item()]
        confidence_score = confidence.item()
        
        print(f"Predicted class: {predicted_class} (confidence: {confidence_score:.3f})")
        print(f"All class probabilities:")
        for i, (class_name, prob) in enumerate(zip(class_names, probabilities[0].cpu().numpy())):
            print(f"  {class_name}: {prob:.3f}")
    
    # Create visualization
    print("Creating visualization...")
    fig = plt.figure(figsize=(18, 6))
    
    # Subplot 1: Input RGB Image
    ax1 = fig.add_subplot(1, 3, 1)
    ax1.imshow(image)
    ax1.set_title('Input RGB Image', fontsize=14, fontweight='bold')
    ax1.axis('off')
    
    # Subplot 2: Ground Truth Point Cloud
    ax2 = fig.add_subplot(1, 3, 2, projection='3d')
    
    # Convert GT point cloud to numpy for visualization
    gt_np = gt_pointcloud.cpu().numpy()
    if len(gt_np.shape) == 3:
        gt_np = gt_np[0]  # Remove batch dimension if present
    
    # Ensure we have exactly 1024 points
    if len(gt_np) > 1024:
        indices = np.random.choice(len(gt_np), 1024, replace=False)
        gt_np = gt_np[indices]
    
    ax2.scatter(gt_np[:, 0], gt_np[:, 1], gt_np[:, 2], 
                c=gt_np[:, 2], cmap='viridis', s=20, alpha=0.7)
    ax2.set_title('Ground Truth (GT)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    
    # Subplot 3: Generated Point Cloud (Ours)
    ax3 = fig.add_subplot(1, 3, 3, projection='3d')
    
    # Convert generated point cloud to numpy for visualization
    gen_np = generated_pointcloud.cpu().numpy()
    if len(gen_np.shape) == 3:
        gen_np = gen_np[0]  # Remove batch dimension if present
    
    # Ensure we have exactly 1024 points
    if len(gen_np) > 1024:
        indices = np.random.choice(len(gen_np), 1024, replace=False)
        gen_np = gen_np[indices]
    
    ax3.scatter(gen_np[:, 0], gen_np[:, 1], gen_np[:, 2], 
                c=gen_np[:, 2], cmap='plasma', s=20, alpha=0.7)
    ax3.set_title('Reconstructed (Ours)', fontsize=14, fontweight='bold')
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_zlabel('Z')
    
    # Add super title with prediction results
    plt.suptitle(f'Predicted: {predicted_class} | Confidence: {confidence_score:.2f}', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    
    print("Visualization created successfully!")
    
    return fig, predicted_class, confidence_score

def main():
    """Example usage of the visualization function"""
    
    # Define class names (same as in training)
    class_names = ['airplane', 'bathtub', 'bed', 'bench', 'bookshelf', 
                   'bottle', 'chair', 'cone', 'cup', 'curtain']
    
    # Example paths (adjust these based on your actual files)
    example_image_path = "../../data/corrected_dataset/images/train/airplane/airplane_000.jpg"
    example_gt_path = "../../data/corrected_dataset/pointclouds/train/airplane/airplane_000.npy"
    example_model_path = "../../corrected_hybrid_model.pth"  # Using the newly trained corrected model
    
    # Check if example files exist, otherwise use alternatives
    if not os.path.exists(example_image_path):
        print(f"Example image not found: {example_image_path}")
        # Try to find any image in the corrected dataset
        for root, dirs, files in os.walk("corrected_dataset/images"):
            for file in files:
                if file.endswith('.jpg'):
                    example_image_path = os.path.join(root, file)
                    # Find corresponding point cloud
                    rel_path = os.path.relpath(example_image_path, "corrected_dataset/images")
                    example_gt_path = os.path.join("corrected_dataset/pointclouds", 
                                                   rel_path.replace('.jpg', '.npy'))
                    if os.path.exists(example_gt_path):
                        break
            else:
                continue
            break
    
    if not os.path.exists(example_model_path):
        print(f"Model not found: {example_model_path}. Using random weights.")
    
    print(f"Using image: {example_image_path}")
    print(f"Using GT point cloud: {example_gt_path}")
    
    # Run the visualization
    try:
        fig, predicted_class, confidence = show_final_prediction(
            example_image_path, 
            example_gt_path, 
            class_names, 
            example_model_path
        )
        
        # Save the figure
        output_path = "inference_result.png"
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\n✅ Visualization saved to: {output_path}")
        
        # Show the plot
        plt.show()
        
    except Exception as e:
        print(f"Error during inference: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()