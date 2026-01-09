import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import open3d as o3d
from typing import Union, Optional


class TNet(nn.Module):
    """
    T-Net for learning input-dependent transformations.
    """
    
    def __init__(self, k: int = 3):
        """
        Initialize T-Net.
        
        Args:
            k: Dimension of transformation matrix (3 for 3D points)
        """
        super(TNet, self).__init__()
        self.k = k
        
        # Shared MLP layers
        self.conv1 = nn.Conv1d(k, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        
        # Fully connected layers
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k * k)
        
        # Batch normalization
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)
        
        # Initialize transformation matrix as identity
        self.fc3.weight.data.zero_()
        self.fc3.bias.data.copy_(torch.eye(k).view(-1))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, k, num_points)
            
        Returns:
            Transformation matrix of shape (batch_size, k, k)
        """
        batch_size = x.size(0)
        
        # Shared MLP
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        
        # Global feature
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(batch_size, -1)
        
        # Fully connected layers
        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        
        # Transformation matrix
        transformation = self.fc3(x).view(batch_size, self.k, self.k)
        
        return transformation


class PointNetBackbone(nn.Module):
    """
    PointNet backbone for feature extraction.
    """
    
    def __init__(self, input_dim: int = 3, feature_dim: int = 1024):
        """
        Initialize PointNet backbone.
        
        Args:
            input_dim: Dimension of input points (3 for xyz)
            feature_dim: Dimension of output features
        """
        super(PointNetBackbone, self).__init__()
        
        self.input_dim = input_dim
        self.feature_dim = feature_dim
        
        # Input transformation network
        self.input_transform = TNet(k=input_dim)
        
        # Shared MLP layers
        self.conv1 = nn.Conv1d(input_dim, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 128, 1)
        self.conv4 = nn.Conv1d(128, 512, 1)
        self.conv5 = nn.Conv1d(512, 2048, 1)
        
        # Batch normalization
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(128)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(2048)
        
        # Feature transformation network (expects 64 channels from conv1)
        self.feature_transform = TNet(k=64)
        
        # Final feature projection
        self.fc1 = nn.Linear(2048, 1024)
        self.fc2 = nn.Linear(1024, feature_dim)
        
        self.bn6 = nn.BatchNorm1d(1024)
        self.bn7 = nn.BatchNorm1d(feature_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim, num_points)
            
        Returns:
            Feature tensor of shape (batch_size, feature_dim)
        """
        batch_size = x.size(0)
        num_points = x.size(2)
        
        # Input transformation
        input_transform = self.input_transform(x)
        x = torch.bmm(input_transform, x)
        
        # Shared MLP 1
        x = F.relu(self.bn1(self.conv1(x)))
        
        # Feature transformation (expects 64 channels from conv1)
        feature_transform = self.feature_transform(x)
        x = torch.bmm(feature_transform, x)
        
        # Shared MLP 2
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.bn5(self.conv5(x))
        
        # Global feature
        x = torch.max(x, 2, keepdim=False)[0]
        
        # Final projection
        x = F.relu(self.bn6(self.fc1(x)))
        x = self.bn7(self.fc2(x))
        
        return x


class PointNet3DFeatureExtractor(nn.Module):
    """
    3D Feature Extractor using PointNet.
    Extracts 1024-dimensional geometric features from point clouds.
    """
    
    def __init__(self, 
                 input_dim: int = 3,
                 feature_dim: int = 1024,
                 num_points: int = 1024):
        """
        Initialize the PointNet 3D feature extractor.
        
        Args:
            input_dim: Dimension of input points (3 for xyz)
            feature_dim: Dimension of output features
            num_points: Expected number of points in input
        """
        super(PointNet3DFeatureExtractor, self).__init__()
        
        self.input_dim = input_dim
        self.feature_dim = feature_dim
        self.num_points = num_points
        
        # PointNet backbone
        self.backbone = PointNetBackbone(input_dim=input_dim, feature_dim=feature_dim)
        
        # Feature normalization
        self.feature_norm = nn.LayerNorm(feature_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to extract 3D features.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim, num_points)
            
        Returns:
            Feature tensor of shape (batch_size, feature_dim)
        """
        # Extract features using PointNet
        features = self.backbone(x)
        
        # Apply feature normalization
        features = self.feature_norm(features)
        
        return features
    
    def extract_features_from_pointcloud(self, 
                                        pointcloud: Union[o3d.geometry.PointCloud, np.ndarray]) -> torch.Tensor:
        """
        Extract features directly from a point cloud.
        
        Args:
            pointcloud: Input point cloud (Open3D or numpy array)
            
        Returns:
            Feature tensor of shape (1, feature_dim)
        """
        # Convert to numpy array
        if isinstance(pointcloud, o3d.geometry.PointCloud):
            points = np.asarray(pointcloud.points)
        elif isinstance(pointcloud, np.ndarray):
            points = pointcloud
        else:
            raise ValueError(f"Unsupported point cloud type: {type(pointcloud)}")
        
        # Ensure correct shape
        if points.shape[1] != self.input_dim:
            if points.shape[0] == self.input_dim:
                points = points.T
            else:
                raise ValueError(f"Expected points with {self.input_dim} dimensions, got {points.shape[1]}")
        
        # Ensure correct number of points
        if points.shape[0] != self.num_points:
            if points.shape[0] > self.num_points:
                # Random sampling
                indices = np.random.choice(points.shape[0], self.num_points, replace=False)
                points = points[indices]
            else:
                # Pad with zeros
                padding = np.zeros((self.num_points - points.shape[0], self.input_dim))
                points = np.vstack([points, padding])
        
        # Convert to tensor and add batch dimension
        points_tensor = torch.FloatTensor(points.T).unsqueeze(0)
        
        # Extract features
        with torch.no_grad():
            features = self.forward(points_tensor)
        
        return features
    
    def extract_batch_features(self, 
                              pointclouds: list) -> torch.Tensor:
        """
        Extract features from a batch of point clouds.
        
        Args:
            pointclouds: List of point clouds
            
        Returns:
            Feature tensor of shape (batch_size, feature_dim)
        """
        batch_points = []
        
        for pcd in pointclouds:
            if isinstance(pcd, o3d.geometry.PointCloud):
                points = np.asarray(pcd.points)
            elif isinstance(pcd, np.ndarray):
                points = pcd
            else:
                raise ValueError(f"Unsupported point cloud type: {type(pcd)}")
            
            # Ensure correct shape and number of points
            if points.shape[1] != self.input_dim:
                if points.shape[0] == self.input_dim:
                    points = points.T
                else:
                    raise ValueError(f"Expected points with {self.input_dim} dimensions")
            
            if points.shape[0] != self.num_points:
                if points.shape[0] > self.num_points:
                    indices = np.random.choice(points.shape[0], self.num_points, replace=False)
                    points = points[indices]
                else:
                    padding = np.zeros((self.num_points - points.shape[0], self.input_dim))
                    points = np.vstack([points, padding])
            
            batch_points.append(points.T)
        
        # Convert to tensor
        batch_tensor = torch.FloatTensor(np.array(batch_points))
        
        # Extract features
        with torch.no_grad():
            features = self.forward(batch_tensor)
        
        return features


def create_pointnet_3d_extractor(config: dict) -> PointNet3DFeatureExtractor:
    """
    Factory function to create a PointNet 3D feature extractor from config.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        PointNet3DFeatureExtractor instance
    """
    model_config = config.get('model', {}).get('pointnet', {})
    
    return PointNet3DFeatureExtractor(
        input_dim=model_config.get('input_dim', 3),
        feature_dim=model_config.get('feature_dim', 1024),
        num_points=model_config.get('num_points', 1024)
    )


if __name__ == "__main__":
    # Test the PointNet 3D feature extractor
    print("Testing PointNet 3D Feature Extractor...")
    
    # Create model
    model = PointNet3DFeatureExtractor()
    print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Test with random input
    test_input = torch.randn(2, 3, 1024)  # batch_size=2, 3D points, 1024 points
    features = model(test_input)
    print(f"Output feature shape: {features.shape}")
    print(f"Expected feature dimension: 1024")
    
    # Test with Open3D point cloud
    test_pcd = o3d.geometry.PointCloud()
    test_points = np.random.randn(1024, 3)
    test_pcd.points = o3d.utility.Vector3dVector(test_points)
    
    features_from_pcd = model.extract_features_from_pointcloud(test_pcd)
    print(f"Features from point cloud shape: {features_from_pcd.shape}")
    
    print("PointNet 3D Feature Extractor test completed successfully!")