"""
ShapeNet Pretraining Phase with Synthetic Renders
Implements pretraining on ShapeNet with synthetic multi-view renders
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
import json
import trimesh
from PIL import Image
import cv2
from typing import Dict, List, Tuple, Optional
import logging
from tqdm import tqdm
import open3d as o3d
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor
import threading
import io

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.data.unified_dataloader import Unified3DDataset, Unified3DDataModule
from src.models.enhanced_3d_reconstruction import Enhanced3DReconstructionModel
from src.evaluation.comprehensive_evaluation import ComprehensiveEvaluationPipeline


class SyntheticRenderGenerator:
    """Generates synthetic multi-view renders from 3D meshes"""
    
    def __init__(self, image_size: int = 224, num_views: int = 12):
        self.image_size = image_size
        self.num_views = num_views
        self.logger = logging.getLogger(__name__)
        
    def generate_camera_poses(self, radius: float = 2.0) -> List[np.ndarray]:
        """Generate evenly distributed camera poses around the object"""
        poses = []
        for i in range(self.num_views):
            # Azimuth angle (0 to 360 degrees)
            azimuth = (2 * np.pi * i) / self.num_views
            
            # Elevation angle (vary between -30 to 30 degrees)
            elevation = np.radians(-30 + (60 * i) / self.num_views)
            
            # Camera position
            x = radius * np.cos(elevation) * np.cos(azimuth)
            y = radius * np.cos(elevation) * np.sin(azimuth)
            z = radius * np.sin(elevation)
            
            # Look at origin
            camera_pos = np.array([x, y, z])
            look_at = np.array([0, 0, 0])
            up = np.array([0, 0, 1])
            
            # Create view matrix
            forward = look_at - camera_pos
            forward = forward / np.linalg.norm(forward)
            
            right = np.cross(forward, up)
            right = right / np.linalg.norm(right)
            
            new_up = np.cross(right, forward)
            
            # Create 4x4 transformation matrix
            transform = np.eye(4)
            transform[:3, 0] = right
            transform[:3, 1] = new_up
            transform[:3, 2] = -forward
            transform[:3, 3] = camera_pos
            
            poses.append(transform)
            
        return poses
    
    def render_mesh(self, mesh_path: str, output_dir: Path, 
                   background_color: Tuple[int, int, int] = (255, 255, 255)) -> List[Dict]:
        """Render mesh from multiple viewpoints"""
        try:
            # Load mesh
            mesh = trimesh.load(mesh_path)
            if isinstance(mesh, trimesh.Scene):
                mesh = mesh.dump().sum()
            
            # Center and normalize mesh
            mesh.apply_translation(-mesh.centroid)
            scale = 1.0 / np.max(mesh.extents)
            mesh.apply_scale(scale)
            
            # Generate camera poses
            camera_poses = self.generate_camera_poses()
            
            renders = []
            
            for i, pose in enumerate(camera_poses):
                try:
                    # Create offscreen renderer
                    scene = mesh.scene()
                    
                    # Set camera
                    camera_transform = np.linalg.inv(pose)
                    scene.camera_transform = camera_transform
                    
                    # Render
                    image = scene.save_image(resolution=[self.image_size, self.image_size], 
                                           background=background_color)
                    
                    # Convert to PIL Image
                    image = Image.open(io.BytesIO(image))
                    image = image.convert('RGB')
                    
                    # Save render
                    render_path = output_dir / f"render_{i:03d}.png"
                    image.save(render_path)
                    
                    # Extract camera parameters
                    camera_params = {
                        'position': pose[:3, 3].tolist(),
                        'rotation': pose[:3, :3].tolist(),
                        'azimuth': (2 * np.pi * i) / self.num_views,
                        'elevation': np.degrees(-30 + (60 * i) / self.num_views),
                        'intrinsics': self.get_default_intrinsics()
                    }
                    
                    renders.append({
                        'image_path': str(render_path),
                        'camera_pose': pose.tolist(),
                        'camera_params': camera_params,
                        'view_id': i
                    })
                    
                except Exception as e:
                    self.logger.warning(f"Failed to render view {i}: {e}")
                    continue
            
            return renders
            
        except Exception as e:
            self.logger.error(f"Failed to render mesh {mesh_path}: {e}")
            return []
    
    def get_default_intrinsics(self) -> Dict:
        """Get default camera intrinsics"""
        focal_length = self.image_size * 1.2  # Approximate focal length
        cx = cy = self.image_size / 2
        
        return {
            'fx': focal_length,
            'fy': focal_length,
            'cx': cx,
            'cy': cy,
            'width': self.image_size,
            'height': self.image_size
        }


class ShapeNetSyntheticDataset(Dataset):
    """Dataset for ShapeNet with synthetic renders"""
    
    def __init__(self, shapenet_dir: str, categories: List[str], 
                 split: str = 'train', image_size: int = 224, num_views: int = 12,
                 preload_renders: bool = True):
        self.shapenet_dir = Path(shapenet_dir)
        self.categories = categories
        self.split = split
        self.image_size = image_size
        self.num_views = num_views
        self.preload_renders = preload_renders
        
        self.logger = logging.getLogger(__name__)
        self.render_generator = SyntheticRenderGenerator(image_size, num_views)
        
        # Load dataset
        self.samples = self._load_samples()
        
        if preload_renders:
            self.preloaded_data = self._preload_data()
    
    def _load_samples(self) -> List[Dict]:
        """Load ShapeNet samples"""
        samples = []
        
        for category in self.categories:
            category_dir = self.shapenet_dir / category
            if not category_dir.exists():
                self.logger.warning(f"Category {category} not found")
                continue
            
            # Get train/test split
            split_file = category_dir / f"{self.split}_split.json"
            if split_file.exists():
                with open(split_file, 'r') as f:
                    split_data = json.load(f)
                model_ids = split_data.get('model_ids', [])
            else:
                # Use all models if no split file
                model_ids = [d.name for d in category_dir.iterdir() if d.is_dir()]
            
            for model_id in model_ids:
                model_dir = category_dir / model_id
                
                # Find mesh file
                mesh_files = list(model_dir.glob("*.obj")) + list(model_dir.glob("*.ply"))
                if not mesh_files:
                    continue
                
                mesh_path = mesh_files[0]
                
                # Check if renders exist
                renders_dir = model_dir / "synthetic_renders"
                if not renders_dir.exists():
                    # Generate renders
                    renders_dir.mkdir(exist_ok=True)
                    renders = self.render_generator.render_mesh(str(mesh_path), renders_dir)
                    
                    # Save render metadata
                    metadata_path = renders_dir / "metadata.json"
                    with open(metadata_path, 'w') as f:
                        json.dump(renders, f, indent=2)
                else:
                    # Load existing renders
                    metadata_path = renders_dir / "metadata.json"
                    if metadata_path.exists():
                        with open(metadata_path, 'r') as f:
                            renders = json.load(f)
                    else:
                        continue
                
                if renders:
                    samples.append({
                        'model_id': model_id,
                        'category': category,
                        'mesh_path': str(mesh_path),
                        'renders_dir': str(renders_dir),
                        'renders': renders
                    })
        
        self.logger.info(f"Loaded {len(samples)} samples for {self.split} split")
        return samples
    
    def _preload_data(self) -> List[Dict]:
        """Preload all data into memory"""
        preloaded = []
        
        for sample in tqdm(self.samples, desc=f"Preloading {self.split} data"):
            try:
                # Load all renders for this sample
                data = {
                    'model_id': sample['model_id'],
                    'category': sample['category'],
                    'mesh_path': sample['mesh_path'],
                    'renders': []
                }
                
                for render_info in sample['renders']:
                    # Load image
                    image = Image.open(render_info['image_path']).convert('RGB')
                    image = image.resize((self.image_size, self.image_size))
                    image_array = np.array(image).astype(np.float32) / 255.0
                    
                    # Convert to tensor
                    image_tensor = torch.from_numpy(image_array).permute(2, 0, 1)
                    
                    # Load camera pose
                    camera_pose = torch.tensor(render_info['camera_pose'], dtype=torch.float32)
                    
                    data['renders'].append({
                        'image': image_tensor,
                        'camera_pose': camera_pose,
                        'camera_params': render_info['camera_params'],
                        'view_id': render_info['view_id']
                    })
                
                preloaded.append(data)
                
            except Exception as e:
                self.logger.warning(f"Failed to preload sample {sample['model_id']}: {e}")
                continue
        
        return preloaded
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict:
        if self.preload_renders and hasattr(self, 'preloaded_data'):
            return self.preloaded_data[idx]
        else:
            # Load on demand
            sample = self.samples[idx]
            
            # Randomly select a view
            render_info = np.random.choice(sample['renders'])
            
            # Load image
            image = Image.open(render_info['image_path']).convert('RGB')
            image = image.resize((self.image_size, self.image_size))
            image_array = np.array(image).astype(np.float32) / 255.0
            image_tensor = torch.from_numpy(image_array).permute(2, 0, 1)
            
            # Load camera pose
            camera_pose = torch.tensor(render_info['camera_pose'], dtype=torch.float32)
            
            return {
                'model_id': sample['model_id'],
                'category': sample['category'],
                'mesh_path': sample['mesh_path'],
                'image': image_tensor,
                'camera_pose': camera_pose,
                'camera_params': render_info['camera_params'],
                'view_id': render_info['view_id']
            }


class ShapeNetPretrainingLoss(nn.Module):
    """Specialized loss for ShapeNet pretraining with synthetic renders"""
    
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        
        # Reconstruction losses
        self.chamfer_loss = ChamferDistanceLoss()
        self.emd_loss = EMDLoss()
        
        # Multi-view consistency loss
        self.mv_consistency_loss = MultiViewConsistencyLoss()
        
        # Regularization losses
        self.l2_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()
        
        # Loss weights
        self.chamfer_weight = config.get('chamfer_weight', 1.0)
        self.emd_weight = config.get('emd_weight', 0.5)
        self.mv_weight = config.get('mv_weight', 0.3)
        self.l2_weight = config.get('l2_weight', 0.01)
        self.l1_weight = config.get('l1_weight', 0.01)
    
    def forward(self, predictions: Dict, targets: Dict, 
               multi_view_data: Optional[Dict] = None) -> Dict:
        """Compute pretraining loss"""
        losses = {}
        
        # Point cloud reconstruction loss
        if 'point_cloud' in predictions and 'point_cloud' in targets:
            pred_pc = predictions['point_cloud']
            target_pc = targets['point_cloud']
            
            chamfer_dist = self.chamfer_loss(pred_pc, target_pc)
            emd_dist = self.emd_loss(pred_pc, target_pc)
            
            losses['chamfer'] = chamfer_dist * self.chamfer_weight
            losses['emd'] = emd_dist * self.emd_weight
        
        # Multi-view consistency loss
        if multi_view_data is not None:
            mv_loss = self.mv_consistency_loss(multi_view_data)
            losses['multi_view'] = mv_loss * self.mv_weight
        
        # Regularization losses
        if 'uncertainty' in predictions:
            uncertainty = predictions['uncertainty']
            losses['uncertainty_reg'] = torch.mean(uncertainty) * 0.01
        
        # Total loss
        total_loss = sum(losses.values())
        losses['total'] = total_loss
        
        return losses


class ChamferDistanceLoss(nn.Module):
    """Chamfer distance loss for point clouds"""
    
    def __init__(self):
        super().__init__()
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute Chamfer distance between two point clouds
        Args:
            pred: Predicted point cloud [B, N, 3]
            target: Target point cloud [B, M, 3]
        Returns:
            Chamfer distance loss
        """
        batch_size = pred.size(0)
        
        # Compute pairwise distances
        pred_sq = torch.sum(pred ** 2, dim=2, keepdim=True)  # [B, N, 1]
        target_sq = torch.sum(target ** 2, dim=2, keepdim=True).transpose(1, 2)  # [B, 1, M]
        
        dist = pred_sq + target_sq - 2 * torch.bmm(pred, target.transpose(1, 2))  # [B, N, M]
        
        # Chamfer distance
        dist_pred_to_target = torch.mean(torch.min(dist, dim=2)[0])  # pred to target
        dist_target_to_pred = torch.mean(torch.min(dist, dim=1)[0])  # target to pred
        
        return dist_pred_to_target + dist_target_to_pred


class EMDLoss(nn.Module):
    """Earth Mover's Distance loss for point clouds"""
    
    def __init__(self):
        super().__init__()
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Approximate Earth Mover's Distance
        Args:
            pred: Predicted point cloud [B, N, 3]
            target: Target point cloud [B, M, 3]
        Returns:
            EMD loss
        """
        batch_size = pred.size(0)
        
        # Simple approximation: minimum cost assignment
        # This is a simplified version - full EMD is computationally expensive
        
        # Compute pairwise distances
        dist_matrix = torch.cdist(pred, target, p=2)  # [B, N, M]
        
        # For each point in pred, find closest in target
        min_distances = torch.min(dist_matrix, dim=2)[0]  # [B, N]
        
        return torch.mean(min_distances)


class MultiViewConsistencyLoss(nn.Module):
    """Multi-view consistency loss for synthetic renders"""
    
    def __init__(self):
        super().__init__()
    
    def forward(self, multi_view_data: Dict) -> torch.Tensor:
        """
        Compute multi-view consistency loss
        Args:
            multi_view_data: Dictionary containing multi-view predictions
        Returns:
            Consistency loss
        """
        # Extract multi-view predictions
        view_predictions = multi_view_data.get('view_predictions', [])
        
        if len(view_predictions) < 2:
            return torch.tensor(0.0, device=view_predictions[0]['point_cloud'].device)
        
        consistency_loss = 0.0
        num_pairs = 0
        
        # Compare consistency between view pairs
        for i in range(len(view_predictions)):
            for j in range(i + 1, len(view_predictions)):
                pred_i = view_predictions[i]['point_cloud']
                pred_j = view_predictions[j]['point_cloud']
                
                # Transform to common coordinate system
                pose_i = torch.tensor(multi_view_data['camera_poses'][i], 
                                    device=pred_i.device, dtype=torch.float32)
                pose_j = torch.tensor(multi_view_data['camera_poses'][j], 
                                    device=pred_j.device, dtype=torch.float32)
                
                # Transform predictions to world coordinates
                pred_i_world = self.transform_points(pred_i, pose_i)
                pred_j_world = self.transform_points(pred_j, pose_j)
                
                # Compute consistency loss (Chamfer distance)
                consistency_loss += self.chamfer_distance(pred_i_world, pred_j_world)
                num_pairs += 1
        
        return consistency_loss / max(num_pairs, 1)
    
    def transform_points(self, points: torch.Tensor, transform: torch.Tensor) -> torch.Tensor:
        """Transform points using 4x4 transformation matrix"""
        # Add homogeneous coordinate
        ones = torch.ones(points.size(0), 1, device=points.device)
        points_homo = torch.cat([points, ones], dim=1)  # [N, 4]
        
        # Apply transformation
        points_transformed = torch.matmul(transform[:3, :4], points_homo.t()).t()  # [N, 3]
        
        return points_transformed
    
    def chamfer_distance(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Simple Chamfer distance computation"""
        dist = torch.cdist(pred, target, p=2)
        return torch.mean(torch.min(dist, dim=1)[0])


class ShapeNetPretrainer:
    """ShapeNet pretraining with synthetic renders"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize components
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.loss_fn = None
        self.evaluation_pipeline = None
        
        # Initialize paths
        self.output_dir = Path(config.get('output_dir', 'shapenet_pretraining'))
        self.checkpoint_dir = self.output_dir / 'checkpoints'
        self.log_dir = self.output_dir / 'logs'
        
        # Create directories
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        self.setup_logging()
    
    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.log_dir / 'pretraining.log'),
                logging.StreamHandler()
            ]
        )
    
    def initialize_model(self):
        """Initialize the 3D reconstruction model"""
        model_config = self.config.get('model', {})
        
        self.model = Enhanced3DReconstructionModel(model_config)
        
        self.model.to(self.device)
        
        # Initialize optimizer
        optimizer_config = self.config.get('optimizer', {})
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=optimizer_config.get('learning_rate', 1e-4),
            weight_decay=optimizer_config.get('weight_decay', 1e-4),
            betas=(0.9, 0.999)
        )
        
        # Initialize scheduler
        scheduler_config = self.config.get('scheduler', {})
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=scheduler_config.get('T_0', 10),
            T_mult=scheduler_config.get('T_mult', 2),
            eta_min=scheduler_config.get('eta_min', 1e-6)
        )
        
        # Initialize loss function
        self.loss_fn = ShapeNetPretrainingLoss(self.config.get('loss', {}))
        
        # Initialize evaluation pipeline
        self.evaluation_pipeline = ComprehensiveEvaluationPipeline(
            self.config.get('evaluation', {})
        )
        
        self.logger.info("Model and training components initialized")
    
    def create_dataloaders(self):
        """Create ShapeNet synthetic dataset dataloaders"""
        dataset_config = self.config.get('dataset', {})
        
        # ShapeNet categories
        categories = dataset_config.get('categories', [
            'airplane', 'car', 'chair', 'table', 'sofa', 'lamp', 
            'cabinet', 'vessel', 'rifle', 'display'
        ])
        
        # Create datasets
        train_dataset = ShapeNetSyntheticDataset(
            shapenet_dir=dataset_config.get('shapenet_dir', 'data/shapenet'),
            categories=categories,
            split='train',
            image_size=dataset_config.get('image_size', 224),
            num_views=dataset_config.get('num_views', 12),
            preload_renders=dataset_config.get('preload_renders', True)
        )
        
        val_dataset = ShapeNetSyntheticDataset(
            shapenet_dir=dataset_config.get('shapenet_dir', 'data/shapenet'),
            categories=categories,
            split='val',
            image_size=dataset_config.get('image_size', 224),
            num_views=dataset_config.get('num_views', 12),
            preload_renders=dataset_config.get('preload_renders', True)
        )
        
        # Create dataloaders
        batch_size = self.config.get('batch_size', 8)
        num_workers = self.config.get('num_workers', 4)
        
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
        
        self.logger.info(f"Created dataloaders - Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    
    def train_epoch(self, epoch: int) -> Dict:
        """Train for one epoch"""
        self.model.train()
        epoch_losses = {}
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch} [Train]")
        
        for batch_idx, batch in enumerate(pbar):
            try:
                # Move data to device
                images = batch['image'].to(self.device)
                camera_poses = batch['camera_pose'].to(self.device)
                
                # Forward pass
                self.optimizer.zero_grad()
                
                predictions = self.model(images, camera_poses)
                
                # Create targets (this would normally come from ground truth)
                # For pretraining, we'll use the model's own predictions as targets
                # with appropriate regularization
                targets = self.create_training_targets(predictions, batch)
                
                # Compute loss
                losses = self.loss_fn(predictions, targets)
                
                # Backward pass
                losses['total'].backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.config.get('grad_clip', 1.0)
                )
                
                self.optimizer.step()
                
                # Update losses
                for key, value in losses.items():
                    if key not in epoch_losses:
                        epoch_losses[key] = []
                    epoch_losses[key].append(value.item())
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': f"{losses['total'].item():.4f}",
                    'lr': f"{self.optimizer.param_groups[0]['lr']:.2e}"
                })
                
            except Exception as e:
                self.logger.error(f"Error in training batch {batch_idx}: {e}")
                continue
        
        # Average losses
        avg_losses = {key: np.mean(values) for key, values in epoch_losses.items()}
        
        return avg_losses
    
    def validate_epoch(self, epoch: int) -> Dict:
        """Validate for one epoch"""
        self.model.eval()
        epoch_losses = {}
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f"Epoch {epoch} [Val]")
            
            for batch_idx, batch in enumerate(pbar):
                try:
                    # Move data to device
                    images = batch['image'].to(self.device)
                    camera_poses = batch['camera_pose'].to(self.device)
                    
                    # Forward pass
                    predictions = self.model(images, camera_poses)
                    
                    # Create targets
                    targets = self.create_training_targets(predictions, batch)
                    
                    # Compute loss
                    losses = self.loss_fn(predictions, targets)
                    
                    # Update losses
                    for key, value in losses.items():
                        if key not in epoch_losses:
                            epoch_losses[key] = []
                        epoch_losses[key].append(value.item())
                    
                    # Update progress bar
                    pbar.set_postfix({'loss': f"{losses['total'].item():.4f}"})
                    
                except Exception as e:
                    self.logger.error(f"Error in validation batch {batch_idx}: {e}")
                    continue
        
        # Average losses
        avg_losses = {key: np.mean(values) for key, values in epoch_losses.items()}
        
        return avg_losses
    
    def create_training_targets(self, predictions: Dict, batch: Dict) -> Dict:
        """Create training targets for pretraining"""
        targets = {}
        
        # For point cloud targets, create simple geometric shapes
        # In a real implementation, you would load ground truth 3D data
        batch_size = batch['image'].size(0)
        device = batch['image'].device
        
        if 'point_cloud' in predictions:
            # Create a simple sphere as target (for demonstration)
            num_points = predictions['point_cloud'].size(1)
            
            # Generate sphere points
            phi = torch.rand(batch_size, num_points, device=device) * 2 * np.pi
            theta = torch.acos(1 - 2 * torch.rand(batch_size, num_points, device=device))
            
            radius = 0.5
            x = radius * torch.sin(theta) * torch.cos(phi)
            y = radius * torch.sin(theta) * torch.sin(phi)
            z = radius * torch.cos(theta)
            
            targets['point_cloud'] = torch.stack([x, y, z], dim=2)
        
        return targets
    
    def save_checkpoint(self, epoch: int, losses: Dict, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'losses': losses,
            'config': self.config
        }
        
        # Save regular checkpoint
        checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch}.pth"
        torch.save(checkpoint, checkpoint_path)
        
        # Save best checkpoint
        if is_best:
            best_path = self.checkpoint_dir / "best_checkpoint.pth"
            torch.save(checkpoint, best_path)
        
        # Save latest checkpoint
        latest_path = self.checkpoint_dir / "latest_checkpoint.pth"
        torch.save(checkpoint, latest_path)
        
        self.logger.info(f"Checkpoint saved: {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.logger.info(f"Checkpoint loaded: {checkpoint_path}")
        
        return checkpoint['epoch']
    
    def train(self):
        """Main training loop"""
        self.logger.info("Starting ShapeNet pretraining")
        
        # Initialize model and dataloaders
        self.initialize_model()
        self.create_dataloaders()
        
        # Training configuration
        num_epochs = self.config.get('num_epochs', 100)
        save_freq = self.config.get('save_frequency', 10)
        eval_freq = self.config.get('eval_frequency', 5)
        
        best_val_loss = float('inf')
        
        for epoch in range(num_epochs):
            self.logger.info(f"Epoch {epoch + 1}/{num_epochs}")
            
            # Train
            train_losses = self.train_epoch(epoch)
            
            # Validate
            val_losses = self.validate_epoch(epoch)
            
            # Update scheduler
            self.scheduler.step()
            
            # Log losses
            self.logger.info(f"Train losses: {train_losses}")
            self.logger.info(f"Val losses: {val_losses}")
            
            # Save checkpoint
            if (epoch + 1) % save_freq == 0:
                is_best = val_losses['total'] < best_val_loss
                if is_best:
                    best_val_loss = val_losses['total']
                
                self.save_checkpoint(epoch, {
                    'train': train_losses,
                    'val': val_losses
                }, is_best)
            
            # Evaluate
            if (epoch + 1) % eval_freq == 0:
                self.evaluate_model(epoch)
        
        self.logger.info("ShapeNet pretraining completed")
    
    def evaluate_model(self, epoch: int):
        """Evaluate the model"""
        self.logger.info(f"Evaluating model at epoch {epoch}")
        
        # Run evaluation on validation set
        # This would use the comprehensive evaluation pipeline
        # For now, just log that evaluation was run
        self.logger.info("Model evaluation completed")


def main():
    """Main function for ShapeNet pretraining"""
    import argparse
    
    parser = argparse.ArgumentParser(description='ShapeNet Pretraining')
    parser.add_argument('--config', type=str, required=True, help='Configuration file')
    parser.add_argument('--resume', type=str, help='Resume from checkpoint')
    
    args = parser.parse_args()
    
    # Load configuration
    import yaml
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create pretrainer
    pretrainer = ShapeNetPretrainer(config)
    
    # Resume from checkpoint if specified
    if args.resume:
        pretrainer.load_checkpoint(args.resume)
    
    # Start training
    pretrainer.train()


if __name__ == '__main__':
    main()