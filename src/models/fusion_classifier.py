import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import numpy as np


class FusionClassifier(nn.Module):
    """
    Fusion classifier that combines 2D and 3D features for object classification.
    """
    
    def __init__(self,
                 input_dim_2d: int = 2048,
                 input_dim_3d: int = 1024,
                 hidden_dim: int = 512,
                 num_classes: int = 10,
                 dropout: float = 0.3,
                 fusion_method: str = "concat"):
        """
        Initialize the fusion classifier.
        
        Args:
            input_dim_2d: Dimension of 2D features
            input_dim_3d: Dimension of 3D features
            hidden_dim: Dimension of hidden layers
            num_classes: Number of output classes
            dropout: Dropout probability
            fusion_method: Method to fuse features ("concat", "add", "attention")
        """
        super(FusionClassifier, self).__init__()
        
        self.input_dim_2d = input_dim_2d
        self.input_dim_3d = input_dim_3d
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.dropout_prob = dropout
        self.fusion_method = fusion_method
        
        # Feature normalization layers
        self.norm_2d = nn.LayerNorm(input_dim_2d)
        self.norm_3d = nn.LayerNorm(input_dim_3d)
        
        # Feature projection layers (optional, for dimension matching)
        if fusion_method in ["add", "attention"]:
            self.proj_2d = nn.Linear(input_dim_2d, hidden_dim)
            self.proj_3d = nn.Linear(input_dim_3d, hidden_dim)
            fusion_input_dim = hidden_dim
        else:  # concat
            fusion_input_dim = input_dim_2d + input_dim_3d
        
        # Attention mechanism for feature fusion
        if fusion_method == "attention":
            self.attention_2d = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 1)
            )
            self.attention_3d = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 1)
            )
            self.attention_softmax = nn.Softmax(dim=1)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(fusion_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
        # Confidence estimation
        self.confidence_head = nn.Sequential(
            nn.Linear(fusion_input_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, features_2d: torch.Tensor, features_3d: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            features_2d: 2D features of shape (batch_size, input_dim_2d)
            features_3d: 3D features of shape (batch_size, input_dim_3d)
            
        Returns:
            Tuple of (classifications, confidences)
            - classifications: (batch_size, num_classes)
            - confidences: (batch_size, 1)
        """
        # Normalize features
        features_2d_norm = self.norm_2d(features_2d)
        features_3d_norm = self.norm_3d(features_3d)
        
        # Fuse features
        if self.fusion_method == "concat":
            fused_features = torch.cat([features_2d_norm, features_3d_norm], dim=1)
        elif self.fusion_method == "add":
            proj_2d = self.proj_2d(features_2d_norm)
            proj_3d = self.proj_3d(features_3d_norm)
            fused_features = proj_2d + proj_3d
        elif self.fusion_method == "attention":
            proj_2d = self.proj_2d(features_2d_norm)
            proj_3d = self.proj_3d(features_3d_norm)
            
            # Compute attention weights
            attn_2d = self.attention_2d(proj_2d)
            attn_3d = self.attention_3d(proj_3d)
            
            # Concatenate attention scores and apply softmax
            attention_scores = torch.cat([attn_2d, attn_3d], dim=1)
            attention_weights = self.attention_softmax(attention_scores)
            
            # Apply attention weights
            fused_features = (attention_weights[:, 0:1] * proj_2d + 
                            attention_weights[:, 1:2] * proj_3d)
        else:
            raise ValueError(f"Unknown fusion method: {self.fusion_method}")
        
        # Classification
        classifications = self.classifier(fused_features)
        
        # Confidence estimation
        confidences = self.confidence_head(fused_features)
        
        return classifications, confidences
    
    def get_feature_importance(self, features_2d: torch.Tensor, features_3d: torch.Tensor) -> dict:
        """
        Get feature importance scores for interpretability.
        
        Args:
            features_2d: 2D features
            features_3d: 3D features
            
        Returns:
            Dictionary with importance scores
        """
        self.eval()
        
        with torch.no_grad():
            # Normalize features
            features_2d_norm = self.norm_2d(features_2d)
            features_3d_norm = self.norm_3d(features_3d)
            
            if self.fusion_method == "attention":
                proj_2d = self.proj_2d(features_2d_norm)
                proj_3d = self.proj_3d(features_3d_norm)
                
                attn_2d = self.attention_2d(proj_2d)
                attn_3d = self.attention_3d(proj_3d)
                
                attention_scores = torch.cat([attn_2d, attn_3d], dim=1)
                attention_weights = self.attention_softmax(attention_scores)
                
                return {
                    'attention_2d': attention_weights[:, 0].cpu().numpy(),
                    'attention_3d': attention_weights[:, 1].cpu().numpy(),
                    'fusion_method': 'attention'
                }
            else:
                # For other methods, compute cosine similarity
                if hasattr(self, 'proj_2d'):
                    proj_2d = self.proj_2d(features_2d_norm)
                    proj_3d = self.proj_3d(features_3d_norm)
                    
                    # Compute similarity scores
                    similarity = F.cosine_similarity(proj_2d, proj_3d, dim=1)
                    
                    return {
                        'similarity': similarity.cpu().numpy(),
                        'fusion_method': self.fusion_method
                    }
                else:
                    return {
                        'fusion_method': self.fusion_method,
                        'note': 'Feature importance not available for concat fusion'
                    }


class Hybrid2D3DClassifier(nn.Module):
    """
    Complete hybrid 2D-3D classifier combining all components.
    """
    
    def __init__(self,
                 resnet_model_name: str = "resnet50",
                 pointnet_input_dim: int = 3,
                 pointnet_feature_dim: int = 1024,
                 fusion_hidden_dim: int = 512,
                 num_classes: int = 10,
                 dropout: float = 0.3,
                 fusion_method: str = "concat",
                 device: str = "cuda"):
        """
        Initialize the complete hybrid classifier.
        
        Args:
            resnet_model_name: ResNet model name for 2D features
            pointnet_input_dim: Input dimension for PointNet
            pointnet_feature_dim: Feature dimension for PointNet
            fusion_hidden_dim: Hidden dimension for fusion
            num_classes: Number of classes
            dropout: Dropout probability
            fusion_method: Fusion method
            device: Device to run on
        """
        super(Hybrid2D3DClassifier, self).__init__()
        
        self.device = device if torch.cuda.is_available() else "cpu"
        self.num_classes = num_classes
        
        # 2D feature extractor
        from .resnet_2d import ResNet2DFeatureExtractor
        self.feature_extractor_2d = ResNet2DFeatureExtractor(
            model_name=resnet_model_name,
            pretrained=True,
            freeze_backbone=True
        )
        
        # 3D feature extractor
        from .pointnet_3d import PointNet3DFeatureExtractor
        self.feature_extractor_3d = PointNet3DFeatureExtractor(
            input_dim=pointnet_input_dim,
            feature_dim=pointnet_feature_dim,
            num_points=1024
        )
        
        # Fusion classifier
        self.fusion_classifier = FusionClassifier(
            input_dim_2d=2048,  # ResNet-50 output
            input_dim_3d=pointnet_feature_dim,
            hidden_dim=fusion_hidden_dim,
            num_classes=num_classes,
            dropout=dropout,
            fusion_method=fusion_method
        )
        
        self.to(self.device)
    
    def forward(self, image: torch.Tensor, pointcloud: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the complete pipeline.
        
        Args:
            image: Input image tensor (batch_size, 3, H, W)
            pointcloud: Input point cloud tensor (batch_size, 3, num_points)
            
        Returns:
            Tuple of (classifications, confidences)
        """
        # Extract 2D features
        features_2d = self.feature_extractor_2d(image)
        
        # Extract 3D features
        features_3d = self.feature_extractor_3d(pointcloud)
        
        # Fuse and classify
        classifications, confidences = self.fusion_classifier(features_2d, features_3d)
        
        return classifications, confidences
    
    def predict(self, image: torch.Tensor, pointcloud: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Make predictions with class probabilities.
        
        Args:
            image: Input image
            pointcloud: Input point cloud
            
        Returns:
            Tuple of (predicted_classes, probabilities, confidences)
        """
        self.eval()
        
        with torch.no_grad():
            classifications, confidences = self.forward(image, pointcloud)
            probabilities = F.softmax(classifications, dim=1)
            predicted_classes = torch.argmax(probabilities, dim=1)
        
        return predicted_classes, probabilities, confidences


def create_fusion_classifier(config: dict) -> FusionClassifier:
    """
    Factory function to create a fusion classifier from config.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        FusionClassifier instance
    """
    fusion_config = config.get('model', {}).get('fusion', {})
    
    return FusionClassifier(
        input_dim_2d=fusion_config.get('input_dim_2d', 2048),
        input_dim_3d=fusion_config.get('input_dim_3d', 1024),
        hidden_dim=fusion_config.get('hidden_dim', 512),
        num_classes=fusion_config.get('num_classes', 10),
        dropout=fusion_config.get('dropout', 0.3),
        fusion_method=fusion_config.get('fusion_method', 'concat')
    )


if __name__ == "__main__":
    # Test the fusion classifier
    print("Testing Fusion Classifier...")
    
    # Create model
    model = FusionClassifier()
    print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Test with random input
    batch_size = 4
    features_2d = torch.randn(batch_size, 2048)
    features_3d = torch.randn(batch_size, 1024)
    
    classifications, confidences = model(features_2d, features_3d)
    
    print(f"Input shapes: 2D {features_2d.shape}, 3D {features_3d.shape}")
    print(f"Output shapes: classifications {classifications.shape}, confidences {confidences.shape}")
    print(f"Expected classifications shape: ({batch_size}, 10)")
    print(f"Expected confidences shape: ({batch_size}, 1)")
    
    # Test feature importance
    importance = model.get_feature_importance(features_2d[:1], features_3d[:1])
    print(f"Feature importance: {importance}")
    
    print("Fusion Classifier test completed successfully!")