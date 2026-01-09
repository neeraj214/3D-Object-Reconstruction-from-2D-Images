"""
Hybrid Classifier Module

This module implements the complete hybrid 2D-to-3D classification system that combines
ImageFeatureExtractor and PointNetFeatures with a fusion layer for object classification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
import logging

# Import the feature extraction modules
from .hybrid_model import ImageFeatureExtractor, PointNetFeatures


class HybridClassifier(nn.Module):
    """
    Complete hybrid classifier that combines 2D image features and 3D point cloud features
    for object classification.
    
    This module integrates:
    1. ImageFeatureExtractor (ResNet-50 based) for 2D visual features
    2. PointNetFeatures for 3D geometric features  
    3. Fusion Layer for combining features and classification
    """
    
    def __init__(self, 
                 num_classes: int = 10,
                 image_pretrained: bool = True,
                 pointnet_input_dim: int = 3,
                 pointnet_feature_dims: list = None,
                 fusion_hidden_dim: int = 512,
                 fusion_dropout: float = 0.5):
        """
        Initialize the Hybrid Classifier.
        
        Args:
            num_classes: Number of object classes for classification
            image_pretrained: Whether to use pre-trained ResNet-50 weights
            pointnet_input_dim: Input dimension for PointNet (default: 3 for x,y,z)
            pointnet_feature_dims: Feature dimensions for PointNet conv layers
            fusion_hidden_dim: Hidden dimension for fusion layer
            fusion_dropout: Dropout probability for fusion layer
        """
        super(HybridClassifier, self).__init__()
        
        self.num_classes = num_classes
        
        # Initialize feature extractors
        self.image_extractor = ImageFeatureExtractor(
            pretrained=image_pretrained,
            feature_dim=2048
        )
        
        self.pointnet_extractor = PointNetFeatures(
            input_dim=pointnet_input_dim,
            feature_dims=pointnet_feature_dims or [64, 128, 1024]
        )
        
        # Get feature dimensions
        self.image_feature_dim = self.image_extractor.get_feature_dim()  # 2048
        self.pointnet_feature_dim = self.pointnet_extractor.get_output_dim()  # 1024
        self.combined_feature_dim = self.image_feature_dim + self.pointnet_feature_dim  # 3072
        
        # Define Fusion Layer (classification head)
        self.fusion_layer = nn.Sequential(
            # First linear layer: 3072 -> 512
            nn.Linear(self.combined_feature_dim, fusion_hidden_dim),
            nn.BatchNorm1d(fusion_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(fusion_dropout),
            
            # Second linear layer: 512 -> num_classes
            nn.Linear(fusion_hidden_dim, num_classes)
        )
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Initialized HybridClassifier")
        self.logger.info(f"  Image feature dim: {self.image_feature_dim}")
        self.logger.info(f"  PointNet feature dim: {self.pointnet_feature_dim}")
        self.logger.info(f"  Combined feature dim: {self.combined_feature_dim}")
        self.logger.info(f"  Fusion hidden dim: {fusion_hidden_dim}")
        self.logger.info(f"  Number of classes: {num_classes}")
        self.logger.info(f"  Dropout: {fusion_dropout}")
    
    def forward(self, image_input: torch.Tensor, pc_input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the hybrid classifier.
        
        Args:
            image_input: Input image tensor of shape (B, C, H, W)
                        where B=batch size, C=channels (3 for RGB), H=height, W=width
            pc_input: Input point cloud tensor of shape (B, N, 3)
                     where B=batch size, N=number of points, 3=x,y,z coordinates
        
        Returns:
            Classification logits of shape (B, num_classes)
            These are unnormalized scores for each class
        """
        # Extract 2D features from image
        features_2d = self.image_extractor(image_input)
        
        # Extract 3D features from point cloud
        features_3d = self.pointnet_extractor(pc_input)
        
        # Concatenate features along the feature dimension (dim=1)
        combined_features = torch.cat([features_2d, features_3d], dim=1)
        
        # Pass through fusion layer to get classification logits
        logits = self.fusion_layer(combined_features)
        
        return logits
    
    def get_feature_dimensions(self) -> Tuple[int, int, int]:
        """
        Get the feature dimensions for each component.
        
        Returns:
            Tuple of (image_feature_dim, pointnet_feature_dim, combined_feature_dim)
        """
        return self.image_feature_dim, self.pointnet_feature_dim, self.combined_feature_dim
    
    def get_classification_weights(self) -> torch.Tensor:
        """
        Get the final classification layer weights for analysis.
        
        Returns:
            Weight tensor of shape (num_classes, fusion_hidden_dim)
        """
        return self.fusion_layer[-1].weight.data


class HybridClassifierWithConfidence(HybridClassifier):
    """
    Extended version of HybridClassifier that also outputs confidence scores.
    
    This version provides both classification logits and confidence scores
    for each prediction, useful for uncertainty estimation and model interpretation.
    """
    
    def __init__(self, *args, **kwargs):
        """Initialize with same parameters as parent class."""
        super(HybridClassifierWithConfidence, self).__init__(*args, **kwargs)
        
        # Add confidence estimation layer
        self.confidence_layer = nn.Sequential(
            nn.Linear(self.combined_feature_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid()  # Output confidence between 0 and 1
        )
    
    def forward(self, image_input: torch.Tensor, pc_input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with confidence estimation.
        
        Args:
            image_input: Input image tensor of shape (B, C, H, W)
            pc_input: Input point cloud tensor of shape (B, N, 3)
        
        Returns:
            Tuple of (logits, confidence_scores)
            - logits: Classification logits of shape (B, num_classes)
            - confidence: Confidence scores of shape (B, 1)
        """
        # Extract 2D features from image
        features_2d = self.image_extractor(image_input)
        
        # Extract 3D features from point cloud
        features_3d = self.pointnet_extractor(pc_input)
        
        # Concatenate features along the feature dimension (dim=1)
        combined_features = torch.cat([features_2d, features_3d], dim=1)
        
        # Pass through fusion layer to get classification logits
        logits = self.fusion_layer(combined_features)
        
        # Estimate confidence
        confidence = self.confidence_layer(combined_features)
        
        return logits, confidence


def test_hybrid_classifier():
    """Test the HybridClassifier with dummy data."""
    print("Testing HybridClassifier...")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize classifier
    num_classes = 10
    classifier = HybridClassifier(
        num_classes=num_classes,
        image_pretrained=False,  # Use non-pretrained for testing
        pointnet_input_dim=3,
        fusion_hidden_dim=512,
        fusion_dropout=0.5
    )
    classifier.to(device)
    
    # Create dummy input data
    batch_size = 4
    dummy_images = torch.randn(batch_size, 3, 224, 224).to(device)
    dummy_pointclouds = torch.randn(batch_size, 1024, 3).to(device)
    
    print(f"Input shapes:")
    print(f"  Images: {dummy_images.shape}")
    print(f"  Point clouds: {dummy_pointclouds.shape}")
    
    # Test forward pass
    classifier.eval()
    with torch.no_grad():
        logits = classifier(dummy_images, dummy_pointclouds)
    
    print(f"Output shape: {logits.shape}")
    print(f"Expected: ({batch_size}, {num_classes})")
    assert logits.shape == (batch_size, num_classes), f"Unexpected output shape: {logits.shape}"
    
    # Test feature dimensions
    img_dim, pc_dim, combined_dim = classifier.get_feature_dimensions()
    print(f"Feature dimensions:")
    print(f"  Image features: {img_dim}")
    print(f"  Point cloud features: {pc_dim}")
    print(f"  Combined features: {combined_dim}")
    
    # Test confidence version
    print("\nTesting HybridClassifierWithConfidence...")
    classifier_conf = HybridClassifierWithConfidence(
        num_classes=num_classes,
        image_pretrained=False,
        pointnet_input_dim=3,
        fusion_hidden_dim=512,
        fusion_dropout=0.5
    )
    classifier_conf.to(device)
    
    with torch.no_grad():
        logits_conf, confidence = classifier_conf(dummy_images, dummy_pointclouds)
    
    print(f"Output shapes:")
    print(f"  Logits: {logits_conf.shape}")
    print(f"  Confidence: {confidence.shape}")
    print(f"Expected confidence: ({batch_size}, 1)")
    assert confidence.shape == (batch_size, 1), f"Unexpected confidence shape: {confidence.shape}"
    
    # Check confidence values are in [0, 1] range
    assert torch.all(confidence >= 0) and torch.all(confidence <= 1), "Confidence values should be in [0, 1] range"
    
    print("\n✅ HybridClassifier tests passed successfully!")
    print(f"   Model is ready for {num_classes}-class classification")
    print(f"   Total trainable parameters: {sum(p.numel() for p in classifier.parameters()):,}")
    print(f"   Fusion layer parameters: {sum(p.numel() for p in classifier.fusion_layer.parameters()):,}")


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Test the classifier
    test_hybrid_classifier()