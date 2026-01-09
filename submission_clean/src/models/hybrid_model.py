"""
Hybrid 2D-to-3D Feature Extraction Modules

This module contains the feature extraction components for both 2D images and 3D point clouds.
These extractors will feed into the final classification layer for hybrid 2D-3D object recognition.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from typing import Tuple, Optional
import logging


class ImageFeatureExtractor(nn.Module):
    """
    2D Feature Extractor using pre-trained ResNet-50.
    
    This module extracts high-level visual features from 2D images using
    a pre-trained ResNet-50 model with the final classification layer removed.
    """
    
    def __init__(self, pretrained: bool = True, feature_dim: int = 2048):
        """
        Initialize the Image Feature Extractor.
        
        Args:
            pretrained: Whether to use pre-trained weights
            feature_dim: Dimension of output features (default: 2048 for ResNet-50)
        """
        super(ImageFeatureExtractor, self).__init__()
        
        # Load pre-trained ResNet-50
        self.resnet = models.resnet50(pretrained=pretrained)
        
        # Remove the final fully connected layer and replace with Identity
        # This gives us the 2048-dimensional feature vector before classification
        self.resnet.fc = nn.Identity()
        
        self.feature_dim = feature_dim
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Initialized ImageFeatureExtractor with feature_dim={feature_dim}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the feature extractor.
        
        Args:
            x: Input image tensor of shape (B, C, H, W) where C=3 for RGB
            
        Returns:
            Feature vector of shape (B, 2048)
        """
        # Ensure input is in the correct format
        if x.dim() == 3:  # Single image (C, H, W)
            x = x.unsqueeze(0)  # Add batch dimension
        
        # Normalize input if not already normalized
        if x.max() > 1.0:
            x = x / 255.0  # Convert to [0, 1] range if needed
        
        # Extract features using ResNet-50
        features = self.resnet(x)
        
        # Ensure output has correct shape
        if features.dim() == 1:  # Single sample
            features = features.unsqueeze(0)
            
        return features
    
    def get_feature_dim(self) -> int:
        """Get the dimension of output features."""
        return self.feature_dim


class PointNetFeatures(nn.Module):
    """
    3D Feature Extractor using PointNet architecture.
    
    This module extracts geometric features from 3D point clouds using
    1D convolutions followed by global max pooling, similar to the original PointNet.
    """
    
    def __init__(self, input_dim: int = 3, feature_dims: list = None):
        """
        Initialize the PointNet Feature Extractor.
        
        Args:
            input_dim: Dimension of input points (default: 3 for x,y,z coordinates)
            feature_dims: List of feature dimensions for each conv layer
                         Default: [64, 128, 1024] as per standard PointNet
        """
        super(PointNetFeatures, self).__init__()
        
        if feature_dims is None:
            feature_dims = [64, 128, 1024]
        
        self.input_dim = input_dim
        self.feature_dims = feature_dims
        
        # Build the convolutional layers
        self.conv_layers = nn.ModuleList()
        
        # First layer: input_dim -> 64
        self.conv_layers.append(nn.Conv1d(input_dim, feature_dims[0], 1))
        
        # Second layer: 64 -> 128
        self.conv_layers.append(nn.Conv1d(feature_dims[0], feature_dims[1], 1))
        
        # Third layer: 128 -> 1024
        self.conv_layers.append(nn.Conv1d(feature_dims[1], feature_dims[2], 1))
        
        # Batch normalization for each conv layer
        self.bn_layers = nn.ModuleList()
        for dim in feature_dims:
            self.bn_layers.append(nn.BatchNorm1d(dim))
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Initialized PointNetFeatures with input_dim={input_dim}, feature_dims={feature_dims}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the PointNet feature extractor.
        
        Args:
            x: Input point cloud tensor of shape (B, N, 3) where
               B = batch size, N = number of points, 3 = x,y,z coordinates
               
        Returns:
            Global feature vector of shape (B, 1024)
        """
        # Ensure input is in the correct format
        if x.dim() == 2:  # Single point cloud (N, 3)
            x = x.unsqueeze(0)  # Add batch dimension
        
        batch_size, num_points, coords = x.shape
        
        # Transpose to (B, 3, N) for Conv1D processing
        x = x.transpose(2, 1)  # Shape: (B, 3, N)
        
        # Apply convolutional layers with batch norm and ReLU
        for conv, bn in zip(self.conv_layers, self.bn_layers):
            x = conv(x)          # Conv1d
            x = bn(x)            # BatchNorm
            x = F.relu(x)        # ReLU activation
        
        # x now has shape: (B, 1024, N)
        
        # Apply global max pooling across the point dimension (dim=2)
        # This gives us the global feature vector for each point cloud
        global_features = torch.max(x, 2)[0]  # Shape: (B, 1024)
        
        return global_features
    
    def get_output_dim(self) -> int:
        """Get the dimension of output features."""
        return self.feature_dims[-1]


class HybridFeatureExtractor(nn.Module):
    """
    Combined feature extractor that processes both 2D images and 3D point clouds.
    
    This module combines the ImageFeatureExtractor and PointNetFeatures
    to extract features from both modalities simultaneously.
    """
    
    def __init__(self, 
                 image_pretrained: bool = True,
                 pointnet_input_dim: int = 3,
                 pointnet_feature_dims: list = None):
        """
        Initialize the Hybrid Feature Extractor.
        
        Args:
            image_pretrained: Whether to use pre-trained ResNet-50
            pointnet_input_dim: Input dimension for PointNet (default: 3)
            pointnet_feature_dims: Feature dimensions for PointNet conv layers
        """
        super(HybridFeatureExtractor, self).__init__()
        
        # Initialize individual feature extractors
        self.image_extractor = ImageFeatureExtractor(pretrained=image_pretrained)
        self.pointnet_extractor = PointNetFeatures(
            input_dim=pointnet_input_dim,
            feature_dims=pointnet_feature_dims
        )
        
        # Get feature dimensions
        self.image_feature_dim = self.image_extractor.get_feature_dim()
        self.pointnet_feature_dim = self.pointnet_extractor.get_output_dim()
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Initialized HybridFeatureExtractor")
        self.logger.info(f"  Image feature dim: {self.image_feature_dim}")
        self.logger.info(f"  PointNet feature dim: {self.pointnet_feature_dim}")
        self.logger.info(f"  Total feature dim: {self.image_feature_dim + self.pointnet_feature_dim}")
    
    def forward(self, image: torch.Tensor, pointcloud: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through both feature extractors.
        
        Args:
            image: Input image tensor of shape (B, C, H, W)
            pointcloud: Input point cloud tensor of shape (B, N, 3)
            
        Returns:
            Tuple of (image_features, pointcloud_features)
            - image_features: Shape (B, 2048)
            - pointcloud_features: Shape (B, 1024)
        """
        # Extract 2D features from image
        image_features = self.image_extractor(image)
        
        # Extract 3D features from point cloud
        pointcloud_features = self.pointnet_extractor(pointcloud)
        
        return image_features, pointcloud_features
    
    def get_feature_dims(self) -> Tuple[int, int]:
        """Get the dimensions of output features for both modalities."""
        return self.image_feature_dim, self.pointnet_feature_dim


def test_feature_extractors():
    """Test the feature extraction modules with dummy data."""
    print("Testing Feature Extraction Modules...")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Test ImageFeatureExtractor
    print("\n1. Testing ImageFeatureExtractor...")
    image_extractor = ImageFeatureExtractor(pretrained=False)  # Use non-pretrained for testing
    image_extractor.to(device)
    
    # Create dummy image batch
    batch_size = 2
    dummy_images = torch.randn(batch_size, 3, 224, 224).to(device)
    
    with torch.no_grad():
        image_features = image_extractor(dummy_images)
    
    print(f"   Input shape: {dummy_images.shape}")
    print(f"   Output shape: {image_features.shape}")
    print(f"   Expected: ({batch_size}, 2048)")
    assert image_features.shape == (batch_size, 2048), f"Unexpected shape: {image_features.shape}"
    print("   ✓ ImageFeatureExtractor test passed!")
    
    # Test PointNetFeatures
    print("\n2. Testing PointNetFeatures...")
    pointnet_extractor = PointNetFeatures(input_dim=3)
    pointnet_extractor.to(device)
    
    # Create dummy point cloud batch
    num_points = 1024
    dummy_pointclouds = torch.randn(batch_size, num_points, 3).to(device)
    
    with torch.no_grad():
        pointcloud_features = pointnet_extractor(dummy_pointclouds)
    
    print(f"   Input shape: {dummy_pointclouds.shape}")
    print(f"   Output shape: {pointcloud_features.shape}")
    print(f"   Expected: ({batch_size}, 1024)")
    assert pointcloud_features.shape == (batch_size, 1024), f"Unexpected shape: {pointcloud_features.shape}"
    print("   ✓ PointNetFeatures test passed!")
    
    # Test HybridFeatureExtractor
    print("\n3. Testing HybridFeatureExtractor...")
    hybrid_extractor = HybridFeatureExtractor(
        image_pretrained=False,
        pointnet_input_dim=3
    )
    hybrid_extractor.to(device)
    
    with torch.no_grad():
        img_feats, pcd_feats = hybrid_extractor(dummy_images, dummy_pointclouds)
    
    print(f"   Image features shape: {img_feats.shape}")
    print(f"   Point cloud features shape: {pcd_feats.shape}")
    print(f"   Expected: Image ({batch_size}, 2048), PointCloud ({batch_size}, 1024)")
    assert img_feats.shape == (batch_size, 2048), f"Unexpected image features shape: {img_feats.shape}"
    assert pcd_feats.shape == (batch_size, 1024), f"Unexpected point cloud features shape: {pcd_feats.shape}"
    print("   ✓ HybridFeatureExtractor test passed!")
    
    print("\n✅ All feature extraction modules tested successfully!")
    print(f"   Total feature dimensions: {2048 + 1024} = 3072")


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Test the modules
    test_feature_extractors()