import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
import open3d as o3d
from transformers import DPTImageProcessor, DPTForDepthEstimation
from typing import Union, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class DepthToPointCloudConverter:
    """
    Converts 2D images to 3D point clouds using MiDaS depth estimation.
    """
    
    def __init__(self, 
                 model_name: str = "Intel/dpt-large",
                 depth_scale: float = 1000.0,
                 max_depth: float = 10.0,
                 min_depth: float = 0.1,
                 device: str = "cuda"):
        """
        Initialize the depth estimation and point cloud converter.
        
        Args:
            model_name: Name of the depth estimation model
            depth_scale: Scale factor for depth values
            max_depth: Maximum depth value to clip
            min_depth: Minimum depth value to clip
            device: Device to run the model on
        """
        self.model_name = model_name
        self.depth_scale = depth_scale
        self.max_depth = max_depth
        self.min_depth = min_depth
        self.device = device if torch.cuda.is_available() else "cpu"
        
        # Load pre-trained depth estimation model
        try:
            self.processor = DPTImageProcessor.from_pretrained(model_name)
            self.depth_model = DPTForDepthEstimation.from_pretrained(model_name)
            self.depth_model.to(self.device)
            self.depth_model.eval()
        except Exception as e:
            print(f"Warning: Could not load model {model_name}. Using default.")
            # Fallback to a simpler model
            self.model_name = "Intel/dpt-large"
            self.processor = DPTImageProcessor.from_pretrained(self.model_name)
            self.depth_model = DPTForDepthEstimation.from_pretrained(self.model_name)
            self.depth_model.to(self.device)
            self.depth_model.eval()
    
    def estimate_depth(self, image: Union[str, np.ndarray, Image.Image]) -> np.ndarray:
        """
        Estimate depth map from input image.
        
        Args:
            image: Input image (file path, numpy array, or PIL Image)
            
        Returns:
            Depth map as numpy array
        """
        # Load and preprocess image
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        elif isinstance(image, np.ndarray):
            if image.shape[2] == 4:  # RGBA
                image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
            elif image.shape[2] == 3:  # BGR
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)
        elif not isinstance(image, Image.Image):
            raise ValueError(f"Unsupported image type: {type(image)}")
        
        # Get original dimensions
        original_width, original_height = image.size
        
        # Process image for depth estimation
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Estimate depth
        with torch.no_grad():
            outputs = self.depth_model(**inputs)
            predicted_depth = outputs.predicted_depth
        
        # Interpolate to original size
        prediction = torch.nn.functional.interpolate(
            predicted_depth.unsqueeze(1),
            size=(original_height, original_width),
            mode="bicubic",
            align_corners=False,
        )
        
        # Convert to numpy and process
        depth_map = prediction.squeeze().cpu().numpy()
        
        # Normalize and clip depth values
        depth_map = self._process_depth_map(depth_map)
        
        return depth_map
    
    def _process_depth_map(self, depth_map: np.ndarray) -> np.ndarray:
        """
        Process depth map by normalizing and clipping values.
        
        Args:
            depth_map: Raw depth map
            
        Returns:
            Processed depth map
        """
        # Normalize depth values
        depth_min = depth_map.min()
        depth_max = depth_map.max()
        depth_map = (depth_map - depth_min) / (depth_max - depth_min)
        
        # Scale and clip
        depth_map = depth_map * self.depth_scale
        depth_map = np.clip(depth_map, self.min_depth, self.max_depth)
        
        return depth_map
    
    def depth_to_pointcloud(self, 
                           image: Union[str, np.ndarray, Image.Image],
                           depth_map: Optional[np.ndarray] = None,
                           num_points: int = 1024,
                           normalize: bool = True) -> o3d.geometry.PointCloud:
        """
        Convert depth map to 3D point cloud.
        
        Args:
            image: Input image
            depth_map: Pre-computed depth map (optional)
            num_points: Number of points to sample
            normalize: Whether to normalize the point cloud
            
        Returns:
            Open3D PointCloud object
        """
        # Load image if needed
        if isinstance(image, str):
            color_image = Image.open(image).convert('RGB')
        elif isinstance(image, np.ndarray):
            if image.shape[2] == 4:  # RGBA
                color_image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
            elif image.shape[2] == 3:  # BGR
                color_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            color_image = Image.fromarray(color_image)
        elif isinstance(image, Image.Image):
            color_image = image.convert('RGB')
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")
        
        # Get depth map if not provided
        if depth_map is None:
            depth_map = self.estimate_depth(color_image)
        
        # Convert to numpy arrays
        color_np = np.array(color_image)
        height, width = color_np.shape[:2]
        
        # Create coordinate grids
        u, v = np.meshgrid(np.arange(width), np.arange(height))
        
        # Convert depth to 3D coordinates (simple pinhole camera model)
        # Assuming focal length and principal point at center
        focal_length = max(width, height)
        cx, cy = width / 2, height / 2
        
        # Calculate 3D coordinates
        z = depth_map.flatten()
        x = (u.flatten() - cx) * z / focal_length
        y = (v.flatten() - cy) * z / focal_length
        
        # Stack coordinates
        points = np.stack([x, y, z], axis=1)
        colors = color_np.reshape(-1, 3) / 255.0
        
        # Create Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        
        # Remove invalid points
        pcd = self._remove_invalid_points(pcd)
        
        # Downsample to desired number of points
        if len(pcd.points) > num_points:
            pcd = pcd.random_down_sample(num_points / len(pcd.points))
        elif len(pcd.points) < num_points:
            # Pad with zeros if not enough points
            points = np.asarray(pcd.points)
            colors = np.asarray(pcd.colors)
            
            # Add random points
            n_missing = num_points - len(points)
            if n_missing > 0:
                # Add random points near existing ones
                random_indices = np.random.choice(len(points), n_missing)
                random_points = points[random_indices] + np.random.normal(0, 0.01, (n_missing, 3))
                random_colors = colors[random_indices]
                
                points = np.vstack([points, random_points])
                colors = np.vstack([colors, random_colors])
                
                pcd.points = o3d.utility.Vector3dVector(points)
                pcd.colors = o3d.utility.Vector3dVector(colors)
        
        # Normalize if requested
        if normalize:
            pcd = self._normalize_pointcloud(pcd)
        
        return pcd
    
    def _remove_invalid_points(self, pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
        """
        Remove invalid points (NaN, infinite values).
        
        Args:
            pcd: Input point cloud
            
        Returns:
            Cleaned point cloud
        """
        points = np.asarray(pcd.points)
        colors = np.asarray(pcd.colors)
        
        # Remove NaN and infinite points
        valid_mask = np.isfinite(points).all(axis=1)
        valid_points = points[valid_mask]
        valid_colors = colors[valid_mask]
        
        cleaned_pcd = o3d.geometry.PointCloud()
        cleaned_pcd.points = o3d.utility.Vector3dVector(valid_points)
        cleaned_pcd.colors = o3d.utility.Vector3dVector(valid_colors)
        
        return cleaned_pcd
    
    def _normalize_pointcloud(self, pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
        """
        Normalize point cloud to unit sphere centered at origin.
        
        Args:
            pcd: Input point cloud
            
        Returns:
            Normalized point cloud
        """
        points = np.asarray(pcd.points)
        colors = np.asarray(pcd.colors)
        
        # Center at origin
        centroid = np.mean(points, axis=0)
        points_centered = points - centroid
        
        # Scale to unit sphere
        max_distance = np.max(np.linalg.norm(points_centered, axis=1))
        if max_distance > 0:
            points_normalized = points_centered / max_distance
        else:
            points_normalized = points_centered
        
        normalized_pcd = o3d.geometry.PointCloud()
        normalized_pcd.points = o3d.utility.Vector3dVector(points_normalized)
        normalized_pcd.colors = o3d.utility.Vector3dVector(colors)
        
        return normalized_pcd
    
    def process_image(self, 
                     image: Union[str, np.ndarray, Image.Image],
                     num_points: int = 1024,
                     return_depth: bool = False) -> Union[o3d.geometry.PointCloud, Tuple[o3d.geometry.PointCloud, np.ndarray]]:
        """
        Complete pipeline: image -> depth -> point cloud.
        
        Args:
            image: Input image
            num_points: Number of points in output cloud
            return_depth: Whether to return depth map
            
        Returns:
            Point cloud (and optionally depth map)
        """
        # Estimate depth
        depth_map = self.estimate_depth(image)
        
        # Convert to point cloud
        pcd = self.depth_to_pointcloud(image, depth_map, num_points)
        
        if return_depth:
            return pcd, depth_map
        return pcd


def create_depth_to_pointcloud_converter(config: dict) -> DepthToPointCloudConverter:
    """
    Factory function to create a depth to point cloud converter from config.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        DepthToPointCloudConverter instance
    """
    reconstruction_config = config.get('reconstruction', {})
    system_config = config.get('system', {})
    
    return DepthToPointCloudConverter(
        model_name=reconstruction_config.get('depth_model', 'Intel/dpt-large'),
        depth_scale=reconstruction_config.get('depth_scale', 1000.0),
        max_depth=reconstruction_config.get('max_depth', 10.0),
        min_depth=reconstruction_config.get('min_depth', 0.1),
        device=system_config.get('device', 'cuda')
    )


if __name__ == "__main__":
    # Test the depth to point cloud converter
    print("Testing Depth to Point Cloud Converter...")
    
    # Create converter
    converter = DepthToPointCloudConverter()
    print(f"Depth model loaded: {converter.model_name}")
    
    # Create a simple test image (gradient)
    test_image = np.zeros((256, 256, 3), dtype=np.uint8)
    for i in range(256):
        test_image[i, :] = [i, i, i]
    
    # Convert to point cloud
    pcd = converter.process_image(test_image, num_points=1024)
    
    print(f"Generated point cloud with {len(pcd.points)} points")
    print(f"Point cloud bounds:")
    points = np.asarray(pcd.points)
    print(f"  X: [{points[:, 0].min():.3f}, {points[:, 0].max():.3f}]")
    print(f"  Y: [{points[:, 1].min():.3f}, {points[:, 1].max():.3f}]")
    print(f"  Z: [{points[:, 2].min():.3f}, {points[:, 2].max():.3f}]")
    
    print("Depth to Point Cloud Converter test completed successfully!")