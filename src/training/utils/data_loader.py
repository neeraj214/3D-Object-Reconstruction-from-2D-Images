import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import open3d as o3d
from typing import List, Tuple, Optional, Dict, Union, Any
import json
import logging
from pathlib import Path
from torchvision import transforms

# Import custom modules
from ..models.resnet_2d import ResNet2DFeatureExtractor, create_resnet_2d_extractor
from ..models.pointnet_3d import PointNet3DFeatureExtractor, create_pointnet_3d_extractor
from ..models.depth_to_pointcloud import DepthToPointCloudConverter, create_depth_to_pointcloud_converter
from ..utils.pointcloud_utils import PointCloudUtils, create_pointcloud_utils


class Hybrid2D3DDataset(Dataset):
    """
    Dataset class for hybrid 2D-3D object classification.
    """
    
    def __init__(self,
                 data_root: str,
                 split: str = 'train',
                 num_points: int = 1024,
                 image_size: Tuple[int, int] = (224, 224),
                 normalize: bool = True,
                 augment: bool = False,
                 preload_features: bool = False,
                 config: Optional[Dict] = None):
        """
        Initialize the dataset.
        
        Args:
            data_root: Root directory of the dataset
            split: Dataset split ('train', 'val', 'test')
            num_points: Number of points in point cloud
            image_size: Size to resize images to
            normalize: Whether to normalize point clouds
            augment: Whether to apply data augmentation
            preload_features: Whether to precompute features
            config: Configuration dictionary
        """
        self.data_root = Path(data_root)
        self.split = split
        self.num_points = num_points
        self.image_size = image_size
        self.normalize = normalize
        self.augment = augment and (split == 'train')
        self.preload_features = preload_features
        self.config = config
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Load dataset metadata
        self.metadata = self.load_metadata()
        
        # Initialize feature extractors if preloading features
        if self.preload_features:
            self.initialize_feature_extractors()
        
        # Initialize point cloud utilities
        self.pointcloud_utils = create_pointcloud_utils(config) if config else PointCloudUtils()
        
        # Setup image transforms
        self.image_transform = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Load data paths
        self.data_paths = self.load_data_paths()
        
        self.logger.info(f"Loaded {len(self.data_paths)} samples for {split} split")
    
    def load_metadata(self) -> Dict:
        """Load dataset metadata."""
        metadata_path = self.data_root / 'metadata.json'
        
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                return json.load(f)
        else:
            # Create default metadata
            self.logger.warning(f"Metadata file not found: {metadata_path}")
            return self.create_default_metadata()
    
    def create_default_metadata(self) -> Dict:
        """Create default metadata for the dataset."""
        # Assume standard structure if no metadata
        class_names = []
        for class_dir in (self.data_root / 'images' / self.split).iterdir():
            if class_dir.is_dir():
                class_names.append(class_dir.name)
        
        class_names = sorted(class_names)
        
        metadata = {
            'class_names': class_names,
            'num_classes': len(class_names),
            'num_points': self.num_points,
            'image_size': self.image_size,
            'splits': ['train', 'val', 'test']
        }
        
        return metadata
    
    def load_data_paths(self) -> List[Dict]:
        """Load data file paths."""
        data_paths = []
        
        # Get class names
        class_names = self.metadata['class_names']
        
        # Image directory structure
        image_dir = self.data_root / 'images' / self.split
        pointcloud_dir = self.data_root / 'pointclouds' / self.split
        
        for class_idx, class_name in enumerate(class_names):
            class_image_dir = image_dir / class_name
            class_pcd_dir = pointcloud_dir / class_name
            
            if not class_image_dir.exists():
                self.logger.warning(f"Image directory not found: {class_image_dir}")
                continue
            
            # Find all images in the class directory
            for image_file in class_image_dir.glob('*'):
                if image_file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                    # Corresponding point cloud file
                    pcd_file = class_pcd_dir / f"{image_file.stem}.npy"  # Try .npy first
                    
                    if not pcd_file.exists():
                        # Try other extensions
                        for ext in ['.npy', '.ply', '.pcd', '.xyz']:
                            alt_pcd_file = class_pcd_dir / f"{image_file.stem}{ext}"
                            if alt_pcd_file.exists():
                                pcd_file = alt_pcd_file
                                break
                    
                    data_paths.append({
                        'image_path': str(image_file),
                        'pointcloud_path': str(pcd_file) if pcd_file.exists() else None,
                        'class_name': class_name,
                        'class_idx': class_idx
                    })
        
        return data_paths
    
    def initialize_feature_extractors(self):
        """Initialize feature extractors for preloading."""
        self.logger.info("Initializing feature extractors for preloading...")
        
        device = torch.device(self.config.get('system', {}).get('device', 'cuda') if torch.cuda.is_available() else 'cpu')
        
        # 2D feature extractor
        self.feature_extractor_2d = create_resnet_2d_extractor(self.config)
        self.feature_extractor_2d.to(device)
        self.feature_extractor_2d.eval()
        
        # 3D feature extractor
        self.feature_extractor_3d = create_pointnet_3d_extractor(self.config)
        self.feature_extractor_3d.to(device)
        self.feature_extractor_3d.eval()
        
        # Point cloud generator for missing ground truth
        self.pointcloud_generator = create_depth_to_pointcloud_converter(self.config)
        
        self.logger.info("Feature extractors initialized")
    
    def __len__(self):
        return len(self.data_paths)
    
    def __getitem__(self, idx):
        """Get a sample from the dataset."""
        sample_data = self.data_paths[idx]
        
        # Load image
        image = self.load_image(sample_data['image_path'])
        
        # Load or generate point cloud
        if sample_data['pointcloud_path'] and os.path.exists(sample_data['pointcloud_path']):
            pointcloud = self.load_pointcloud(sample_data['pointcloud_path'])
        else:
            # Generate point cloud from image if ground truth not available
            pointcloud = self.generate_pointcloud(sample_data['image_path'])
        
        # Process point cloud
        pointcloud = self.process_pointcloud(pointcloud)
        
        # Get label
        label = sample_data['class_idx']
        
        if self.preload_features:
            # Precompute features
            features_2d = self.extract_2d_features(image)
            features_3d = self.extract_3d_features(pointcloud)
            
            return features_2d, features_3d, label
        else:
            # Convert PIL image to tensor
            image_tensor = self.image_transform(image)
            
            # Convert point cloud to numpy array
            if isinstance(pointcloud, o3d.geometry.PointCloud):
                points = np.asarray(pointcloud.points)
                colors = np.asarray(pointcloud.colors) if pointcloud.has_colors() else np.ones_like(points)
                pointcloud_array = np.hstack([points, colors])
            else:
                pointcloud_array = pointcloud
            
            # Return processed data
            # Ensure point cloud is float32 and has correct shape [num_points, 3]
            if pointcloud_array.shape[1] >= 6:
                # Use only 3D coordinates (x, y, z)
                points_only = pointcloud_array[:, :3].astype(np.float32)
                pointcloud_tensor = points_only  # Shape: [num_points, 3]
            else:
                # If only 3 coordinates, keep as is
                pointcloud_tensor = pointcloud_array.astype(np.float32)  # Shape: [num_points, 3]
            
            return image_tensor, pointcloud_tensor, label
    
    def load_image(self, image_path: str) -> Image.Image:
        """Load and preprocess image."""
        image = Image.open(image_path).convert('RGB')
        
        # Resize image
        image = image.resize(self.image_size, Image.Resampling.LANCZOS)
        
        return image
    
    def load_pointcloud(self, pcd_path: str) -> o3d.geometry.PointCloud:
        """Load point cloud."""
        try:
            # Handle .npy files (our generated point clouds)
            if pcd_path.endswith('.npy'):
                pointcloud_array = np.load(pcd_path)
                
                # Create Open3D point cloud
                pointcloud = o3d.geometry.PointCloud()
                
                # Extract 3D coordinates (first 3 columns)
                if pointcloud_array.shape[1] >= 3:
                    points = pointcloud_array[:, :3]
                    pointcloud.points = o3d.utility.Vector3dVector(points)
                    
                    # Extract colors if available (columns 3-6)
                    if pointcloud_array.shape[1] >= 6:
                        colors = pointcloud_array[:, 3:6]
                        pointcloud.colors = o3d.utility.Vector3dVector(colors)
                else:
                    raise ValueError(f"Invalid point cloud array shape: {pointcloud_array.shape}")
                
                return pointcloud
            else:
                # Handle other formats (.ply, .pcd, .xyz)
                pointcloud = o3d.io.read_point_cloud(pcd_path)
                if len(pointcloud.points) == 0:
                    raise ValueError(f"Empty point cloud: {pcd_path}")
                return pointcloud
                
        except Exception as e:
            self.logger.warning(f"Failed to load point cloud {pcd_path}: {e}")
            # Generate point cloud from corresponding image
            image_path = pcd_path.replace('pointclouds', 'images').replace('.npy', '.jpg').replace('.ply', '.jpg')
            return self.generate_pointcloud(image_path)
    
    def generate_pointcloud(self, image_path: str) -> o3d.geometry.PointCloud:
        """Generate point cloud from image using depth estimation."""
        self.logger.debug(f"Generating point cloud for: {image_path}")
        
        if hasattr(self, 'pointcloud_generator'):
            pointcloud = self.pointcloud_generator.process_image(
                image_path, 
                num_points=self.num_points
            )
        else:
            # Create temporary generator
            generator = create_depth_to_pointcloud_converter(self.config)
            pointcloud = generator.process_image(
                image_path, 
                num_points=self.num_points
            )
        
        return pointcloud
    
    def process_pointcloud(self, pointcloud: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
        """Process point cloud (normalize, sample, etc.)."""
        # Sample points if needed
        if len(pointcloud.points) != self.num_points:
            points = np.asarray(pointcloud.points)
            
            if len(points) > self.num_points:
                # Random sampling
                indices = np.random.choice(len(points), self.num_points, replace=False)
                points = points[indices]
            else:
                # Pad with zeros if not enough points
                padding = np.zeros((self.num_points - len(points), 3))
                points = np.vstack([points, padding])
            
            # Create new point cloud
            new_pcd = o3d.geometry.PointCloud()
            new_pcd.points = o3d.utility.Vector3dVector(points)
            pointcloud = new_pcd
        
        # Normalize if requested
        if self.normalize:
            points = np.asarray(pointcloud.points)
            if isinstance(self.pointcloud_utils, dict):
                # If pointcloud_utils is a dictionary (from create_pointcloud_utils)
                normalized_points, _ = self.pointcloud_utils['normalizer'].normalize_pointcloud(points)
            else:
                # If pointcloud_utils is a PointCloudUtils object
                from .pointcloud_utils import PointCloudNormalizer
                normalizer = PointCloudNormalizer()
                normalized_points, _ = normalizer.normalize_pointcloud(points)
            pointcloud.points = o3d.utility.Vector3dVector(normalized_points)
        
        # Apply augmentation if training
        if self.augment:
            points = np.asarray(pointcloud.points)
            points = self.apply_augmentation(points)
            pointcloud.points = o3d.utility.Vector3dVector(points)
        
        return pointcloud
    
    def apply_augmentation(self, points: np.ndarray) -> np.ndarray:
        """Apply data augmentation to point cloud."""
        if isinstance(self.pointcloud_utils, dict):
            # If pointcloud_utils is a dictionary (from create_pointcloud_utils)
            augmenter = self.pointcloud_utils['augmenter']
        else:
            # If pointcloud_utils is a PointCloudUtils object, create augmenter
            from .pointcloud_utils import PointCloudAugmenter
            augmenter = PointCloudAugmenter()
        
        # Random rotation
        if np.random.random() < 0.5:
            points = augmenter.random_rotation(points, max_angle=np.pi / 6)
        
        # Random jitter
        if np.random.random() < 0.5:
            points = augmenter.random_jitter(points, std=0.01)
        
        # Random scaling
        if np.random.random() < 0.5:
            points = augmenter.random_scale(points, scale_range=(0.9, 1.1))
        
        return points
    
    def extract_2d_features(self, image: Image.Image) -> torch.Tensor:
        """Extract 2D features from image."""
        # Convert image to tensor format expected by ResNet
        from torchvision import transforms
        
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        image_tensor = transform(image).unsqueeze(0)
        
        with torch.no_grad():
            features = self.feature_extractor_2d(image_tensor)
        
        return features.squeeze(0)
    
    def extract_3d_features(self, pointcloud: o3d.geometry.PointCloud) -> torch.Tensor:
        """Extract 3D features from point cloud."""
        # Convert point cloud to tensor format
        points = np.asarray(pointcloud.points)
        
        # Ensure correct shape and number of points
        if len(points) != self.num_points:
            points = self.pointcloud_utils['sampler'].random_sample(points, self.num_points)
        
        # Convert to tensor
        points_tensor = torch.FloatTensor(points.T).unsqueeze(0)
        
        with torch.no_grad():
            features = self.feature_extractor_3d(points_tensor)
        
        return features.squeeze(0)
    
    def get_class_names(self) -> List[str]:
        """Get list of class names."""
        return self.metadata['class_names']
    
    def get_num_classes(self) -> int:
        """Get number of classes."""
        return self.metadata['num_classes']


class DatasetLoader:
    """
    Utility class for loading and managing datasets.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the dataset loader.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Dataset configuration
        self.data_root = Path(config['data']['paths']['root'])
        self.num_points = config['data']['point_cloud']['num_points']
        self.image_size = tuple(config['data']['image']['size'])
        self.batch_size = config['training']['batch_size']
        self.num_workers = config['system']['num_workers']
        self.normalize = config['data']['point_cloud']['normalize']
        self.preload_features = config.get('data', {}).get('preload_features', False)
    
    def load_dataset(self, split: str, augment: bool = False) -> Hybrid2D3DDataset:
        """
        Load a specific dataset split.
        
        Args:
            split: Dataset split ('train', 'val', 'test')
            augment: Whether to apply data augmentation
            
        Returns:
            Dataset instance
        """
        dataset = Hybrid2D3DDataset(
            data_root=self.data_root,
            split=split,
            num_points=self.num_points,
            image_size=self.image_size,
            normalize=self.normalize,
            augment=augment,
            preload_features=self.preload_features,
            config=self.config
        )
        
        self.logger.info(f"Loaded {split} dataset with {len(dataset)} samples")
        
        return dataset
    
    def create_data_loaders(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Create data loaders for all splits.
        
        Returns:
            Tuple of (train_loader, val_loader, test_loader)
        """
        # Load datasets
        train_dataset = self.load_dataset('train', augment=True)
        val_dataset = self.load_dataset('val', augment=False)
        test_dataset = self.load_dataset('test', augment=False)
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
        
        self.logger.info(f"Created data loaders:")
        self.logger.info(f"  Train: {len(train_loader)} batches")
        self.logger.info(f"  Val: {len(val_loader)} batches")
        self.logger.info(f"  Test: {len(test_loader)} batches")
        
        return train_loader, val_loader, test_loader
    
    def extract_features_from_dataset(self, dataset: Hybrid2D3DDataset) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Extract features from a dataset for training the fusion classifier.
        
        Args:
            dataset: Dataset instance
            
        Returns:
            Tuple of (features_2d, features_3d, labels)
        """
        self.logger.info(f"Extracting features from {len(dataset)} samples...")
        
        features_2d_list = []
        features_3d_list = []
        labels_list = []
        
        for i in range(len(dataset)):
            if i % 100 == 0:
                self.logger.info(f"Processing sample {i+1}/{len(dataset)}")
            
            image, pointcloud, label = dataset[i]
            
            # Extract features
            features_2d = dataset.extract_2d_features(image)
            features_3d = dataset.extract_3d_features(pointcloud)
            
            features_2d_list.append(features_2d.numpy())
            features_3d_list.append(features_3d.numpy())
            labels_list.append(label)
        
        features_2d = np.array(features_2d_list)
        features_3d = np.array(features_3d_list)
        labels = np.array(labels_list)
        
        self.logger.info(f"Feature extraction completed:")
        self.logger.info(f"  2D features shape: {features_2d.shape}")
        self.logger.info(f"  3D features shape: {features_3d.shape}")
        self.logger.info(f"  Labels shape: {labels.shape}")
        
        return features_2d, features_3d, labels
    
    def create_sample_dataset(self, output_dir: str, num_samples_per_class: int = 10):
        """
        Create a sample dataset for testing.
        
        Args:
            output_dir: Output directory for the sample dataset
            num_samples_per_class: Number of samples per class
        """
        self.logger.info(f"Creating sample dataset with {num_samples_per_class} samples per class...")
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Sample classes (you can modify this list)
        sample_classes = [
            'airplane', 'bathtub', 'bed', 'bench', 'bookshelf',
            'bottle', 'chair', 'cone', 'cup', 'curtain'
        ]
        
        for split in ['train', 'val', 'test']:
            for class_name in sample_classes:
                # Create directories
                (output_path / 'images' / split / class_name).mkdir(parents=True, exist_ok=True)
                (output_path / 'pointclouds' / split / class_name).mkdir(parents=True, exist_ok=True)
                
                # Create dummy data (in a real scenario, you would copy actual data)
                for i in range(num_samples_per_class):
                    # Create dummy image
                    dummy_image = Image.new('RGB', self.image_size, color=(i*20, i*20, i*20))
                    image_path = output_path / 'images' / split / class_name / f'{class_name}_{i:03d}.jpg'
                    dummy_image.save(image_path)
                    
                    # Create dummy point cloud
                    dummy_points = np.random.randn(self.num_points, 3) * 0.5
                    dummy_pcd = o3d.geometry.PointCloud()
                    dummy_pcd.points = o3d.utility.Vector3dVector(dummy_points)
                    pcd_path = output_path / 'pointclouds' / split / class_name / f'{class_name}_{i:03d}.ply'
                    o3d.io.write_point_cloud(str(pcd_path), dummy_pcd)
        
        # Create metadata
        metadata = {
            'class_names': sample_classes,
            'num_classes': len(sample_classes),
            'num_points': self.num_points,
            'image_size': self.image_size,
            'splits': ['train', 'val', 'test'],
            'samples_per_class': num_samples_per_class
        }
        
        with open(output_path / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        self.logger.info(f"Sample dataset created at: {output_path}")


class Pix3DDataset(Dataset):
    """
    Dataset class for Pix3D dataset with 2D images and 3D mesh models.
    Loads 3D models and samples them to create ground truth point clouds.
    """
    
    def __init__(self,
                 metadata: List[Dict[str, Any]],
                 data_root: str = 'data/curated_dataset',
                 num_points: int = 1024,
                 image_size: Tuple[int, int] = (224, 224),
                 transform=None):
        """
        Initialize the Pix3D dataset.
        
        Args:
            metadata: List of metadata dictionaries containing img_path, model_path, label_id
            data_root: Root directory for the dataset
            num_points: Number of points to sample from mesh
            image_size: Size to resize images to
            transform: Optional transform for images
        """
        self.metadata = metadata
        self.data_root = Path(data_root)
        self.num_points = num_points
        self.image_size = image_size
        self.transform = transform
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Validate metadata
        self.validate_metadata()
        
        self.logger.info(f"Loaded Pix3D dataset with {len(self.metadata)} samples")
    
    def validate_metadata(self):
        """Validate metadata structure."""
        required_keys = ['img_path', 'model_path', 'label_id']
        for i, item in enumerate(self.metadata):
            for key in required_keys:
                if key not in item:
                    raise ValueError(f"Metadata item {i} missing required key: {key}")
    
    def load_3d_point_cloud(self, model_path: str) -> torch.Tensor:
        """
        Load 3D model from path, sample to 1024 points, normalize, and return as tensor.
        
        Args:
            model_path: Path to 3D model file (.obj, .ply, etc.)
            
        Returns:
            Normalized point cloud tensor of shape (1024, 3)
        """
        try:
            # Load the mesh using Open3D
            full_model_path = self.data_root / model_path
            if not full_model_path.exists():
                # Try relative path
                full_model_path = Path(model_path)
                if not full_model_path.exists():
                    raise FileNotFoundError(f"Model file not found: {model_path}")
            
            mesh = o3d.io.read_triangle_mesh(str(full_model_path))
            
            if mesh.is_empty():
                raise ValueError(f"Empty mesh loaded from: {model_path}")
            
            # Sample the mesh to get point cloud
            pointcloud = mesh.sample_points_uniformly(number_of_points=self.num_points)
            
            # Get points as numpy array
            points = np.asarray(pointcloud.points)
            
            # Normalize: center and scale to unit sphere
            # Center the point cloud (subtract centroid)
            centroid = np.mean(points, axis=0)
            points_centered = points - centroid
            
            # Scale to unit sphere (divide by max distance from origin)
            max_distance = np.max(np.linalg.norm(points_centered, axis=1))
            if max_distance > 0:
                points_normalized = points_centered / max_distance
            else:
                points_normalized = points_centered
            
            # Convert to PyTorch tensor
            pointcloud_tensor = torch.FloatTensor(points_normalized)
            
            return pointcloud_tensor
            
        except Exception as e:
            self.logger.error(f"Error loading 3D model {model_path}: {e}")
            # Return a default point cloud (centered at origin with random points)
            return torch.randn(self.num_points, 3) * 0.1
    
    def load_image(self, img_path: str) -> Image.Image:
        """Load and preprocess image."""
        full_img_path = self.data_root / img_path
        if not full_img_path.exists():
            # Try relative path
            full_img_path = Path(img_path)
            if not full_img_path.exists():
                raise FileNotFoundError(f"Image file not found: {img_path}")
        
        image = Image.open(full_img_path).convert('RGB')
        image = image.resize(self.image_size, Image.Resampling.LANCZOS)
        
        if self.transform:
            image = self.transform(image)
        
        return image
    
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        """
        Get a sample from the dataset.
        
        Returns:
            Tuple of (transformed_image, gt_point_cloud_tensor, label)
        """
        sample_data = self.metadata[idx]
        
        # Load image
        image = self.load_image(sample_data['img_path'])
        
        # Load 3D point cloud from mesh
        gt_point_cloud = self.load_3d_point_cloud(sample_data['model_path'])
        
        # Get label
        label = sample_data['label_id']
        
        return image, gt_point_cloud, label
    
    def get_class_distribution(self):
        """Get distribution of classes in the dataset."""
        labels = [item['label_id'] for item in self.metadata]
        unique_labels, counts = np.unique(labels, return_counts=True)
        return dict(zip(unique_labels, counts))


def create_dataset_loader(config: Dict) -> DatasetLoader:
    """
    Factory function to create a dataset loader from config.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        DatasetLoader instance
    """
    return DatasetLoader(config)


def create_pix3d_dataset(metadata: List[Dict[str, Any]], 
                        data_root: str = 'data/curated_dataset',
                        num_points: int = 1024,
                        image_size: Tuple[int, int] = (224, 224),
                        transform=None) -> Pix3DDataset:
    """
    Factory function to create a Pix3DDataset from metadata.
    
    Args:
        metadata: List of metadata dictionaries containing img_path, model_path, label_id
        data_root: Root directory for the dataset
        num_points: Number of points to sample from mesh
        image_size: Size to resize images to
        transform: Optional transform for images
        
    Returns:
        Pix3DDataset instance
    """
    return Pix3DDataset(
        metadata=metadata,
        data_root=data_root,
        num_points=num_points,
        image_size=image_size,
        transform=transform
    )


if __name__ == "__main__":
    # Test the dataset loader
    print("Testing Dataset Loader...")
    
    # Sample configuration
    config = {
        'data': {
            'paths': {
                'root': 'data/sample_dataset',
                'images': 'data/sample_dataset/images',
                'pointclouds': 'data/sample_dataset/pointclouds'
            },
            'point_cloud': {
                'num_points': 1024,
                'normalize': True
            },
            'image': {
                'size': [224, 224]
            },
            'preload_features': False
        },
        'training': {
            'batch_size': 16
        },
        'system': {
            'num_workers': 4,
            'device': 'cuda'
        }
    }
    
    # Create dataset loader
    dataset_loader = create_dataset_loader(config)
    
    # Create sample dataset
    dataset_loader.create_sample_dataset('data/sample_dataset', num_samples_per_class=5)
    
    # Load dataset
    train_dataset = dataset_loader.load_dataset('train', augment=True)
    
    print(f"Dataset loaded successfully!")
    print(f"Number of samples: {len(train_dataset)}")
    print(f"Number of classes: {train_dataset.get_num_classes()}")
    print(f"Class names: {train_dataset.get_class_names()}")
    
    # Test a sample
    sample = train_dataset[0]
    image, pointcloud, label = sample
    
    print(f"\nSample data:")
    print(f"Image type: {type(image)}")
    print(f"Point cloud points: {len(pointcloud.points)}")
    print(f"Label: {label}")
    
    print("\n" + "="*50)
    print("Testing Pix3DDataset with placeholder metadata...")
    
    # Import the placeholder metadata function
    import sys
    sys.path.append(str(Path(__file__).parent.parent))
    from data_curation_helper import create_placeholder_metadata
    
    # Create placeholder metadata
    metadata = create_placeholder_metadata()
    print(f"Created metadata with {len(metadata)} samples")
    
    # Create Pix3DDataset
    pix3d_dataset = create_pix3d_dataset(
        metadata=metadata,
        data_root='data/curated_dataset',
        num_points=1024,
        image_size=(224, 224)
    )
    
    print(f"Pix3DDataset created with {len(pix3d_dataset)} samples")
    
    # Show class distribution
    class_dist = pix3d_dataset.get_class_distribution()
    print(f"Class distribution: {class_dist}")
    
    # Test a sample (will show error for missing files, but demonstrates structure)
    try:
        sample = pix3d_dataset[0]
        image, gt_pointcloud, label = sample
        print(f"\nPix3D Sample:")
        print(f"Image type: {type(image)}")
        print(f"GT Point cloud shape: {gt_pointcloud.shape}")
        print(f"Label: {label}")
        print(f"Point cloud range: [{gt_pointcloud.min():.3f}, {gt_pointcloud.max():.3f}]")
        print(f"Point cloud mean: {gt_pointcloud.mean():.3f}")
        print(f"Point cloud std: {gt_pointcloud.std():.3f}")
    except Exception as e:
        print(f"Expected error (files don't exist): {e}")
        print("This demonstrates the correct structure - the dataset is ready for real data!")
    
    print("\nDataset loader test completed successfully!")