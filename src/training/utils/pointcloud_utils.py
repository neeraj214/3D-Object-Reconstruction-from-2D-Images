import numpy as np
import open3d as o3d
from typing import Union, Tuple, Optional
import torch
import cv2
from PIL import Image
import logging


class PointCloudNormalizer:
    """
    Utilities for normalizing and preprocessing point clouds.
    """
    
    @staticmethod
    def center_pointcloud(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Center point cloud at origin.
        
        Args:
            points: Point cloud of shape (N, 3)
            
        Returns:
            Tuple of (centered_points, centroid)
        """
        centroid = np.mean(points, axis=0)
        centered_points = points - centroid
        return centered_points, centroid
    
    @staticmethod
    def scale_to_unit_sphere(points: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Scale point cloud to unit sphere.
        
        Args:
            points: Point cloud of shape (N, 3)
            
        Returns:
            Tuple of (scaled_points, scale_factor)
        """
        distances = np.linalg.norm(points, axis=1)
        max_distance = np.max(distances)
        
        if max_distance > 0:
            scaled_points = points / max_distance
        else:
            scaled_points = points
            
        return scaled_points, max_distance
    
    @staticmethod
    def normalize_pointcloud(points: np.ndarray, 
                           center: bool = True, 
                           scale: bool = True) -> Tuple[np.ndarray, dict]:
        """
        Complete normalization of point cloud.
        
        Args:
            points: Point cloud of shape (N, 3)
            center: Whether to center at origin
            scale: Whether to scale to unit sphere
            
        Returns:
            Tuple of (normalized_points, normalization_params)
        """
        normalization_params = {}
        normalized_points = points.copy()
        
        if center:
            normalized_points, centroid = PointCloudNormalizer.center_pointcloud(normalized_points)
            normalization_params['centroid'] = centroid
        
        if scale:
            normalized_points, scale_factor = PointCloudNormalizer.scale_to_unit_sphere(normalized_points)
            normalization_params['scale_factor'] = scale_factor
        
        return normalized_points, normalization_params
    
    @staticmethod
    def denormalize_pointcloud(normalized_points: np.ndarray, 
                             normalization_params: dict) -> np.ndarray:
        """
        Reverse normalization of point cloud.
        
        Args:
            normalized_points: Normalized point cloud
            normalization_params: Parameters from normalization
            
        Returns:
            Denormalized point cloud
        """
        points = normalized_points.copy()
        
        # Reverse scaling
        if 'scale_factor' in normalization_params:
            points = points * normalization_params['scale_factor']
        
        # Reverse centering
        if 'centroid' in normalization_params:
            points = points + normalization_params['centroid']
        
        return points


class PointCloudSampler:
    """
    Utilities for sampling and resampling point clouds.
    """
    
    @staticmethod
    def random_sample(points: np.ndarray, 
                     num_points: int, 
                     colors: Optional[np.ndarray] = None) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Randomly sample points from point cloud.
        
        Args:
            points: Point cloud of shape (N, 3)
            num_points: Number of points to sample
            colors: Optional colors of shape (N, 3)
            
        Returns:
            Sampled points (and colors if provided)
        """
        n_points = len(points)
        
        if n_points >= num_points:
            # Sample without replacement
            indices = np.random.choice(n_points, num_points, replace=False)
        else:
            # Sample with replacement if not enough points
            indices = np.random.choice(n_points, num_points, replace=True)
        
        sampled_points = points[indices]
        
        if colors is not None:
            sampled_colors = colors[indices]
            return sampled_points, sampled_colors
        
        return sampled_points
    
    @staticmethod
    def farthest_point_sample(points: np.ndarray, num_points: int) -> np.ndarray:
        """
        Farthest point sampling for better coverage.
        
        Args:
            points: Point cloud of shape (N, 3)
            num_points: Number of points to sample
            
        Returns:
            Sampled points
        """
        n_points = len(points)
        
        if n_points <= num_points:
            return points
        
        # Initialize with random point
        sampled_indices = [np.random.randint(n_points)]
        
        # Compute distances to first point
        distances = np.linalg.norm(points - points[sampled_indices[0]], axis=1)
        
        # Iteratively select farthest points
        for _ in range(num_points - 1):
            # Select point with maximum distance
            farthest_idx = np.argmax(distances)
            sampled_indices.append(farthest_idx)
            
            # Update distances
            new_distances = np.linalg.norm(points - points[farthest_idx], axis=1)
            distances = np.minimum(distances, new_distances)
        
        return points[sampled_indices]
    
    @staticmethod
    def uniform_sample(points: np.ndarray, num_points: int) -> np.ndarray:
        """
        Uniform sampling using voxel grid.
        
        Args:
            points: Point cloud of shape (N, 3)
            num_points: Number of points to sample
            
        Returns:
            Sampled points
        """
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        
        # Downsample using voxel grid
        voxel_size = np.max(np.std(points, axis=0)) / (num_points ** (1/3))
        downsampled = pcd.voxel_down_sample(voxel_size=voxel_size)
        
        sampled_points = np.asarray(downsampled.points)
        
        # If still too many points, randomly sample
        if len(sampled_points) > num_points:
            indices = np.random.choice(len(sampled_points), num_points, replace=False)
            sampled_points = sampled_points[indices]
        
        return sampled_points


class PointCloudAugmenter:
    """
    Data augmentation techniques for point clouds.
    """
    
    @staticmethod
    def random_rotation(points: np.ndarray, 
                       max_angle: float = np.pi / 4) -> np.ndarray:
        """
        Apply random rotation to point cloud.
        
        Args:
            points: Point cloud of shape (N, 3)
            max_angle: Maximum rotation angle in radians
            
        Returns:
            Rotated point cloud
        """
        # Random rotation angles
        angles = np.random.uniform(-max_angle, max_angle, 3)
        
        # Rotation matrices
        Rx = np.array([[1, 0, 0],
                      [0, np.cos(angles[0]), -np.sin(angles[0])],
                      [0, np.sin(angles[0]), np.cos(angles[0])]])
        
        Ry = np.array([[np.cos(angles[1]), 0, np.sin(angles[1])],
                      [0, 1, 0],
                      [-np.sin(angles[1]), 0, np.cos(angles[1])]])
        
        Rz = np.array([[np.cos(angles[2]), -np.sin(angles[2]), 0],
                      [np.sin(angles[2]), np.cos(angles[2]), 0],
                      [0, 0, 1]])
        
        # Apply rotations
        rotation_matrix = Rz @ Ry @ Rx
        return points @ rotation_matrix.T
    
    @staticmethod
    def random_jitter(points: np.ndarray, 
                       std: float = 0.01, 
                       clip: float = 0.02) -> np.ndarray:
        """
        Add random jitter to point cloud.
        
        Args:
            points: Point cloud of shape (N, 3)
            std: Standard deviation of noise
            clip: Maximum noise value
            
        Returns:
            Jittered point cloud
        """
        noise = np.random.normal(0, std, points.shape)
        noise = np.clip(noise, -clip, clip)
        return points + noise
    
    @staticmethod
    def random_scale(points: np.ndarray, 
                     scale_range: Tuple[float, float] = (0.8, 1.2)) -> np.ndarray:
        """
        Apply random scaling to point cloud.
        
        Args:
            points: Point cloud of shape (N, 3)
            scale_range: Range of scaling factors
            
        Returns:
            Scaled point cloud
        """
        scale = np.random.uniform(scale_range[0], scale_range[1])
        return points * scale
    
    @staticmethod
    def random_dropout(points: np.ndarray, 
                      dropout_ratio: float = 0.2) -> np.ndarray:
        """
        Randomly drop points from point cloud.
        
        Args:
            points: Point cloud of shape (N, 3)
            dropout_ratio: Ratio of points to drop
            
        Returns:
            Point cloud with dropped points
        """
        n_points = len(points)
        n_keep = int(n_points * (1 - dropout_ratio))
        indices = np.random.choice(n_points, n_keep, replace=False)
        return points[indices]


class PointCloudUtils:
    """
    General utility functions for point cloud processing.
    """
    
    @staticmethod
    def pointcloud_to_tensor(pointcloud: Union[o3d.geometry.PointCloud, np.ndarray],
                           num_points: int = 1024,
                           normalize: bool = True) -> torch.Tensor:
        """
        Convert point cloud to tensor format for deep learning.
        
        Args:
            pointcloud: Input point cloud
            num_points: Number of points to sample
            normalize: Whether to normalize the point cloud
            
        Returns:
            Tensor of shape (3, num_points)
        """
        # Convert to numpy
        if isinstance(pointcloud, o3d.geometry.PointCloud):
            points = np.asarray(pointcloud.points)
        else:
            points = pointcloud
        
        # Sample points
        if len(points) != num_points:
            points = PointCloudSampler.random_sample(points, num_points)
        
        # Normalize if requested
        if normalize:
            points, _ = PointCloudNormalizer.normalize_pointcloud(points)
        
        # Convert to tensor and transpose
        tensor = torch.FloatTensor(points.T)
        
        return tensor
    
    @staticmethod
    def tensor_to_pointcloud(tensor: torch.Tensor) -> o3d.geometry.PointCloud:
        """
        Convert tensor back to point cloud.
        
        Args:
            tensor: Tensor of shape (3, N) or (N, 3)
            
        Returns:
            Open3D point cloud
        """
        # Ensure correct shape
        if tensor.shape[0] == 3:
            points = tensor.T.numpy()
        else:
            points = tensor.numpy()
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        
        return pcd
    
    @staticmethod
    def compute_pointcloud_stats(pointcloud: Union[o3d.geometry.PointCloud, np.ndarray]) -> dict:
        """
        Compute statistics for point cloud.
        
        Args:
            pointcloud: Input point cloud
            
        Returns:
            Dictionary with statistics
        """
        if isinstance(pointcloud, o3d.geometry.PointCloud):
            points = np.asarray(pointcloud.points)
        else:
            points = pointcloud
        
        stats = {
            'num_points': len(points),
            'bounds': {
                'x': [float(points[:, 0].min()), float(points[:, 0].max())],
                'y': [float(points[:, 1].min()), float(points[:, 1].max())],
                'z': [float(points[:, 2].min()), float(points[:, 2].max())]
            },
            'centroid': np.mean(points, axis=0).tolist(),
            'std': np.std(points, axis=0).tolist(),
            'mean_distance_from_origin': float(np.mean(np.linalg.norm(points, axis=1)))
        }
        
        return stats


def create_pointcloud_utils(config: dict) -> dict:
    """
    Factory function to create point cloud utilities from config.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Dictionary with utility instances
    """
    data_config = config.get('data', {}).get('point_cloud', {})
    
    return {
        'normalizer': PointCloudNormalizer(),
        'sampler': PointCloudSampler(),
        'augmenter': PointCloudAugmenter(),
        'utils': PointCloudUtils(),
        'config': data_config
    }


def generate_point_cloud_from_image(image_tensor: Union[torch.Tensor, Image.Image, np.ndarray],
                                  num_points: int = 1024,
                                  camera_intrinsics: Optional[dict] = None,
                                  depth_scale: float = 1000.0,
                                  device: str = 'cuda' if torch.cuda.is_available() else 'cpu') -> torch.Tensor:
    """
    Generate a 3D point cloud from a 2D image using MiDaS depth estimation.
    
    This function serves as the core 2D-to-3D reconstruction step, allowing
    the system to take any 2D image and automatically generate its 3D structure.
    
    Args:
        image_tensor: Input 2D RGB image (torch.Tensor, PIL.Image, or numpy array)
        num_points: Number of points to sample in the final point cloud (default: 1024)
        camera_intrinsics: Camera intrinsic parameters (optional, uses defaults if None)
        depth_scale: Scale factor for depth values (default: 1000.0)
        device: Device to run MiDaS model on ('cuda' or 'cpu')
        
    Returns:
        Normalized point cloud tensor of shape (num_points, 3)
        
    Raises:
        RuntimeError: If MiDaS model fails to load or process image
        ValueError: If input image format is invalid
    """
    logger = logging.getLogger(__name__)
    
    try:
        # Load MiDaS model (using the smaller, faster version)
        logger.info("Loading MiDaS depth estimation model...")
        midas = torch.hub.load('intel-isl/MiDaS', 'MiDaS_small', pretrained=True)
        midas.to(device)
        midas.eval()
        
        # Load corresponding transform
        transform = torch.hub.load('intel-isl/MiDaS', 'transforms')
        transform = transform.small_transform
        
    except Exception as e:
        logger.error(f"Failed to load MiDaS model: {e}")
        raise RuntimeError(f"Failed to load MiDaS model: {e}")
    
    try:
        # Convert input to appropriate format
        if isinstance(image_tensor, torch.Tensor):
            # Convert torch tensor to numpy
            if image_tensor.dim() == 4:  # Batch dimension
                image_tensor = image_tensor.squeeze(0)
            if image_tensor.dim() == 3 and image_tensor.shape[0] in [1, 3]:  # CHW format
                image_tensor = image_tensor.permute(1, 2, 0)
            input_image = image_tensor.cpu().numpy()
        elif isinstance(image_tensor, Image.Image):
            input_image = np.array(image_tensor)
        elif isinstance(image_tensor, np.ndarray):
            input_image = image_tensor.copy()
        else:
            raise ValueError(f"Unsupported image type: {type(image_tensor)}")
        
        # Ensure RGB format
        if input_image.shape[2] == 4:  # RGBA
            input_image = input_image[:, :, :3]
        elif len(input_image.shape) == 2:  # Grayscale
            input_image = np.stack([input_image] * 3, axis=2)
        
        original_height, original_width = input_image.shape[:2]
        logger.info(f"Processing image of size {original_width}x{original_height}")
        
        # Preprocess image for MiDaS
        input_batch = transform(input_image).to(device)
        
        # Generate depth map
        logger.info("Generating depth map...")
        with torch.no_grad():
            prediction = midas(input_batch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=(original_height, original_width),
                mode="bicubic",
                align_corners=False,
            ).squeeze()
        
        # Convert depth map to numpy
        depth_map = prediction.cpu().numpy()
        
        # Normalize depth map to reasonable range
        depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())
        depth_map = depth_map * depth_scale
        
        logger.info(f"Depth map generated with range [{depth_map.min():.2f}, {depth_map.max():.2f}]")
        
    except Exception as e:
        logger.error(f"Failed to generate depth map: {e}")
        raise RuntimeError(f"Failed to generate depth map: {e}")
    
    try:
        # Set up camera intrinsics with realistic placeholder values
        if camera_intrinsics is None:
            # Default camera intrinsics for a typical camera
            focal_length = max(original_width, original_height) * 0.8  # Reasonable focal length
            camera_intrinsics = {
                'width': original_width,
                'height': original_height,
                'fx': focal_length,  # Focal length in x
                'fy': focal_length,  # Focal length in y
                'cx': original_width / 2,  # Principal point x
                'cy': original_height / 2,  # Principal point y
            }
        
        # Create Open3D camera intrinsic object
        intrinsic = o3d.camera.PinholeCameraIntrinsic(
            width=camera_intrinsics['width'],
            height=camera_intrinsics['height'],
            fx=camera_intrinsics['fx'],
            fy=camera_intrinsics['fy'],
            cx=camera_intrinsics['cx'],
            cy=camera_intrinsics['cy']
        )
        
        # Convert depth map to Open3D format
        depth_image = o3d.geometry.Image((depth_map * 1000).astype(np.uint16))  # Scale for uint16
        color_image = o3d.geometry.Image(input_image.astype(np.uint8))
        
        # Create RGBD image
        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color_image, 
            depth_image,
            depth_scale=1000.0,  # Convert mm to meters
            depth_trunc=depth_scale,  # Maximum depth in meters
            convert_rgb_to_intensity=False
        )
        
        # Generate point cloud from RGBD image
        logger.info("Creating point cloud from depth map...")
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intrinsic)
        
        # Get points as numpy array
        points = np.asarray(pcd.points)
        
        if len(points) == 0:
            raise RuntimeError("Generated point cloud is empty")
        
        logger.info(f"Generated point cloud with {len(points)} points")
        
    except Exception as e:
        logger.error(f"Failed to create point cloud: {e}")
        raise RuntimeError(f"Failed to create point cloud: {e}")
    
    try:
        # Apply normalization and sampling (same logic as load_3d_point_cloud)
        logger.info("Normalizing and sampling point cloud...")
        
        # Sample to desired number of points
        if len(points) > num_points:
            indices = np.random.choice(len(points), num_points, replace=False)
            points = points[indices]
        elif len(points) < num_points:
            # Pad with random points if not enough
            padding = np.random.choice(len(points), num_points - len(points), replace=True)
            points = np.vstack([points, points[padding]])
        
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
        
        logger.info(f"Point cloud normalized. Shape: {pointcloud_tensor.shape}")
        logger.info(f"Point cloud range: [{points_normalized.min():.3f}, {points_normalized.max():.3f}]")
        
        return pointcloud_tensor
        
    except Exception as e:
        logger.error(f"Failed to normalize point cloud: {e}")
        raise RuntimeError(f"Failed to normalize point cloud: {e}")