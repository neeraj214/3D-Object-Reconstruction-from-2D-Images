import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from typing import Union, Tuple


class ResNet2DFeatureExtractor(nn.Module):
    """
    2D Feature Extractor using pre-trained ResNet-50.
    Extracts 2048-dimensional visual features from input images.
    """
    
    def __init__(self, 
                 model_name: str = "resnet50",
                 feature_dim: int = 2048,
                 pretrained: bool = True,
                 freeze_backbone: bool = False):
        """
        Initialize the ResNet 2D feature extractor.
        
        Args:
            model_name: Name of the ResNet model
            feature_dim: Dimension of output features
            pretrained: Whether to use pre-trained weights
            freeze_backbone: Whether to freeze the backbone during training
        """
        super(ResNet2DFeatureExtractor, self).__init__()
        
        self.model_name = model_name
        self.feature_dim = feature_dim
        self.pretrained = pretrained
        self.freeze_backbone = freeze_backbone
        
        # Load pre-trained ResNet model
        if model_name == "resnet50":
            self.backbone = models.resnet50(pretrained=pretrained)
        elif model_name == "resnet18":
            self.backbone = models.resnet18(pretrained=pretrained)
        elif model_name == "resnet101":
            self.backbone = models.resnet101(pretrained=pretrained)
        else:
            raise ValueError(f"Unsupported model: {model_name}")
        
        # Remove the final classification layer
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])
        
        # Freeze backbone if specified
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Feature normalization layer
        self.feature_norm = nn.LayerNorm(feature_dim)
        
        # Image preprocessing transforms
        self.transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to extract 2D features.
        
        Args:
            x: Input tensor of shape (batch_size, 3, H, W)
            
        Returns:
            Feature tensor of shape (batch_size, feature_dim)
        """
        # Extract features using backbone
        features = self.backbone(x)
        
        # Flatten the features
        features = features.view(features.size(0), -1)
        
        # Apply feature normalization
        features = self.feature_norm(features)
        
        return features
    
    def extract_features_from_image(self, 
                                   image: Union[str, np.ndarray, Image.Image]) -> torch.Tensor:
        """
        Extract features directly from an image file or array.
        
        Args:
            image: Input image (file path, numpy array, or PIL Image)
            
        Returns:
            Feature tensor of shape (1, feature_dim)
        """
        # Load and preprocess image
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        elif not isinstance(image, Image.Image):
            raise ValueError(f"Unsupported image type: {type(image)}")
        
        # Apply transforms
        image_tensor = self.transforms(image).unsqueeze(0)
        
        # Extract features
        with torch.no_grad():
            features = self.forward(image_tensor)
        
        return features
    
    def extract_batch_features(self, 
                              image_paths: list) -> torch.Tensor:
        """
        Extract features from a batch of images.
        
        Args:
            image_paths: List of image file paths
            
        Returns:
            Feature tensor of shape (batch_size, feature_dim)
        """
        images = []
        for path in image_paths:
            image = Image.open(path).convert('RGB')
            image_tensor = self.transforms(image)
            images.append(image_tensor)
        
        batch_tensor = torch.stack(images)
        
        with torch.no_grad():
            features = self.forward(batch_tensor)
        
        return features


def create_resnet_2d_extractor(config: dict) -> ResNet2DFeatureExtractor:
    """
    Factory function to create a ResNet 2D feature extractor from config.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        ResNet2DFeatureExtractor instance
    """
    model_config = config.get('model', {}).get('resnet', {})
    
    return ResNet2DFeatureExtractor(
        model_name=model_config.get('model_name', 'resnet50'),
        feature_dim=model_config.get('feature_dim', 2048),
        pretrained=model_config.get('pretrained', True),
        freeze_backbone=model_config.get('freeze_backbone', False)
    )


if __name__ == "__main__":
    # Test the 2D feature extractor
    print("Testing ResNet 2D Feature Extractor...")
    
    # Create model
    model = ResNet2DFeatureExtractor()
    print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Test with random input
    test_input = torch.randn(1, 3, 224, 224)
    features = model(test_input)
    print(f"Output feature shape: {features.shape}")
    print(f"Expected feature dimension: 2048")
    
    print("2D Feature Extractor test completed successfully!")