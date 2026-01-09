
"""
Comprehensive Training Pipeline for Enhanced 3D Reconstruction
Supports mixed precision, advanced augmentation, curriculum learning, and distributed training
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torch.cuda.amp import autocast, GradScaler
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Any
from tqdm import tqdm
import wandb
import os
from pathlib import Path
import json
import time

from src.models.enhanced_3d_reconstruction import Enhanced3DReconstructionModel
from src.optimization.performance_optimizer import MixedPrecisionTrainer, get_optimization_config
from src.training.losses import silhouette_iou_loss
import torch.nn.functional as F

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AdvancedLossFunction(nn.Module):
    """Advanced loss function combining multiple loss terms."""
    
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        
        # Loss weights
        self.sdf_loss_weight = config.get('sdf_loss_weight', 1.0)
        self.normal_loss_weight = config.get('normal_loss_weight', 0.1)
        self.laplacian_loss_weight = config.get('laplacian_loss_weight', 0.01)
        self.chamfer_loss_weight = config.get('chamfer_loss_weight', 1.0)
        self.occupancy_loss_weight = config.get('occupancy_loss_weight', 1.0)
        self.nerf_loss_weight = config.get('nerf_loss_weight', 1.0)
        self.uncertainty_loss_weight = config.get('uncertainty_loss_weight', 0.1)
        self.silhouette_loss_weight = config.get('silhouette_loss_weight', 0.05)
        self.emd_loss_weight = config.get('emd_loss_weight', 0.25)
        self.emd_gp_weight = config.get('emd_gp_weight', 0.05)
        
        # Loss functions
        self.l1_loss = nn.L1Loss()
        self.l2_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()
        self.huber_loss = nn.HuberLoss(delta=0.1)
        
    def compute_sdf_loss(self, pred_sdf: torch.Tensor, target_sdf: torch.Tensor,
                        uncertainty: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute SDF loss with uncertainty weighting."""
        if uncertainty is not None:
            # Weight by inverse uncertainty (more certain = higher weight)
            weights = 1.0 / (uncertainty + 1e-6)
            weights = weights / weights.mean()  # Normalize
            loss = (weights * self.l1_loss(pred_sdf, target_sdf)).mean()
        else:
            loss = self.l1_loss(pred_sdf, target_sdf)
        
        return loss
    
    def compute_occupancy_loss(self, pred_occ: torch.Tensor, target_occ: torch.Tensor) -> torch.Tensor:
        """Compute occupancy loss with focal loss for class imbalance."""
        # Focal loss for handling class imbalance
        gamma = self.config.get('focal_gamma', 2.0)
        alpha = self.config.get('focal_alpha', 0.75)
        
        bce_loss = self.bce_loss(pred_occ, target_occ)
        p_t = torch.exp(-bce_loss)
        focal_loss = alpha * (1 - p_t) ** gamma * bce_loss
        
        return focal_loss.mean()
    
    def compute_normal_loss(self, pred_normals: torch.Tensor, target_normals: torch.Tensor) -> torch.Tensor:
        """Compute normal consistency loss."""
        # Cosine similarity loss
        cos_sim = F.cosine_similarity(pred_normals, target_normals, dim=-1)
        return (1 - cos_sim).mean()
    
    def compute_laplacian_loss(self, vertices: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
        """Compute Laplacian smoothness loss."""
        # This is a simplified version - full implementation would need mesh laplacian
        if vertices.shape[0] < 3:
            return torch.tensor(0.0, device=vertices.device)
        
        # Compute edge lengths
        edges = vertices[faces[:, 1]] - vertices[faces[:, 0]]
        edge_lengths = torch.norm(edges, dim=1)
        
        # Encourage uniform edge lengths
        target_length = edge_lengths.mean()
        return self.l2_loss(edge_lengths, target_length.expand_as(edge_lengths))
    
    def compute_chamfer_loss(self, pred_points: torch.Tensor, target_points: torch.Tensor) -> torch.Tensor:
        """Compute Chamfer distance between point clouds."""
        if pred_points.shape[1] == 0 or target_points.shape[1] == 0:
            return torch.tensor(0.0, device=pred_points.device)
        
        # Compute pairwise distances
        pred_to_target_dist = torch.cdist(pred_points, target_points, p=2)  # [B, N_pred, N_target]
        target_to_pred_dist = torch.cdist(target_points, pred_points, p=2)  # [B, N_target, N_pred]
        
        # Chamfer distance
        pred_to_target = pred_to_target_dist.min(dim=2)[0].mean()  # pred -> target
        target_to_pred = target_to_pred_dist.min(dim=2)[0].mean()  # target -> pred
        
        return pred_to_target + target_to_pred

    def compute_emd_loss(self, pred_points: torch.Tensor, target_points: torch.Tensor, tau: float = 0.02) -> torch.Tensor:
        if pred_points.shape[1] == 0 or target_points.shape[1] == 0:
            return torch.tensor(0.0, device=pred_points.device)
        d = torch.cdist(pred_points, target_points, p=2)
        r = torch.softmax(-d / tau, dim=2)
        c = torch.softmax(-d / tau, dim=1)
        l = (r * d).sum(dim=2).mean(dim=1) + (c * d).sum(dim=1).mean(dim=1)
        return l.mean()

    def compute_emd_grad_penalty(self, pred_points: torch.Tensor, target_points: torch.Tensor, tau: float = 0.02) -> torch.Tensor:
        try:
            emd = self.compute_emd_loss(pred_points, target_points, tau)
            grads = torch.autograd.grad(emd, pred_points, create_graph=True, retain_graph=True, only_inputs=True)[0]
            gp = (grads.norm(dim=2) - 1.0).abs().mean()
            return gp
        except Exception:
            return torch.tensor(0.0, device=pred_points.device)
    
    def compute_nerf_loss(self, pred_colors: torch.Tensor, target_colors: torch.Tensor,
                         pred_depth: Optional[torch.Tensor] = None,
                         target_depth: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute NeRF rendering loss."""
        # Color loss
        color_loss = self.l2_loss(pred_colors, target_colors)
        
        # Depth loss if available
        depth_loss = torch.tensor(0.0, device=color_loss.device)
        if pred_depth is not None and target_depth is not None:
            depth_loss = self.l1_loss(pred_depth, target_depth)
        
        return color_loss + 0.1 * depth_loss
    
    def forward(self, predictions: Dict[str, torch.Tensor], 
                targets: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute total loss and individual loss components."""
        total_loss = torch.tensor(0.0, device=next(iter(predictions.values())).device)
        loss_components = {}
        
        # SDF loss
        if 'sdf_pred' in predictions and 'sdf_target' in targets:
            sdf_loss = self.compute_sdf_loss(
                predictions['sdf_pred'], 
                targets['sdf_target'],
                predictions.get('uncertainty')
            )
            total_loss += self.sdf_loss_weight * sdf_loss
            loss_components['sdf_loss'] = sdf_loss.item()
        
        # Occupancy loss
        if 'occupancy_pred' in predictions and 'occupancy_target' in targets:
            occ_loss = self.compute_occupancy_loss(
                predictions['occupancy_pred'],
                targets['occupancy_target']
            )
            total_loss += self.occupancy_loss_weight * occ_loss
            loss_components['occupancy_loss'] = occ_loss.item()
        
        # NeRF loss
        if 'rendered_colors' in predictions and 'target_colors' in targets:
            nerf_loss = self.compute_nerf_loss(
                predictions['rendered_colors'],
                targets['target_colors'],
                predictions.get('rendered_depth'),
                targets.get('target_depth')
            )
            total_loss += self.nerf_loss_weight * nerf_loss
            loss_components['nerf_loss'] = nerf_loss.item()
        
        # Normal loss
        if 'normals_pred' in predictions and 'normals_target' in targets:
            normal_loss = self.compute_normal_loss(
                predictions['normals_pred'],
                targets['normals_target']
            )
            total_loss += self.normal_loss_weight * normal_loss
            loss_components['normal_loss'] = normal_loss.item()
        
        # Laplacian loss
        if 'vertices' in predictions and 'faces' in predictions:
            laplacian_loss = self.compute_laplacian_loss(
                predictions['vertices'],
                predictions['faces']
            )
            total_loss += self.laplacian_loss_weight * laplacian_loss
            loss_components['laplacian_loss'] = laplacian_loss.item()
        
        # Chamfer loss
        if 'point_cloud_pred' in predictions and 'point_cloud_target' in targets:
            chamfer_loss = self.compute_chamfer_loss(
                predictions['point_cloud_pred'],
                targets['point_cloud_target']
            )
            total_loss += self.chamfer_loss_weight * chamfer_loss
            loss_components['chamfer_loss'] = chamfer_loss.item()

            emd_loss = self.compute_emd_loss(
                predictions['point_cloud_pred'],
                targets['point_cloud_target']
            )
            total_loss += self.emd_loss_weight * emd_loss
            loss_components['emd_loss'] = emd_loss.item()
            if self.emd_gp_weight > 0:
                gp = self.compute_emd_grad_penalty(
                    predictions['point_cloud_pred'],
                    targets['point_cloud_target']
                )
                total_loss += self.emd_gp_weight * gp
                loss_components['emd_gp'] = gp.item()

        if 'point_cloud_pred' in predictions and 'camera_K' in targets and 'mask' in targets:
            try:
                sil = silhouette_iou_loss(
                    predictions['point_cloud_pred'],
                    targets['camera_K'],
                    targets['mask']
                ).mean()
                total_loss += self.silhouette_loss_weight * sil
                loss_components['silhouette_loss'] = sil.item()
            except Exception:
                pass
        
        # Uncertainty regularization
        if 'uncertainty' in predictions and self.uncertainty_loss_weight > 0:
            # Encourage uncertainty to be neither too high nor too low
            uncertainty_reg = torch.mean(predictions['uncertainty'] * (1 - predictions['uncertainty']))
            total_loss += self.uncertainty_loss_weight * uncertainty_reg
            loss_components['uncertainty_reg'] = uncertainty_reg.item()
        
        loss_components['total_loss'] = total_loss.item()
        
        return total_loss, loss_components

class DataAugmentation3D:
    """Advanced 3D data augmentation techniques."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.augmentation_prob = config.get('augmentation_prob', 0.5)
        
    def random_rotation_3d(self, points: torch.Tensor, max_angle: float = 15.0) -> torch.Tensor:
        """Apply random 3D rotation."""
        if torch.rand(1) > self.augmentation_prob:
            return points
        
        # Random angles in degrees
        angles = torch.rand(3) * max_angle * 2 - max_angle
        angles_rad = torch.deg2rad(angles)
        
        # Rotation matrices
        Rx = torch.tensor([
            [1, 0, 0],
            [0, torch.cos(angles_rad[0]), -torch.sin(angles_rad[0])],
            [0, torch.sin(angles_rad[0]), torch.cos(angles_rad[0])]
        ], device=points.device)
        
        Ry = torch.tensor([
            [torch.cos(angles_rad[1]), 0, torch.sin(angles_rad[1])],
            [0, 1, 0],
            [-torch.sin(angles_rad[1]), 0, torch.cos(angles_rad[1])]
        ], device=points.device)
        
        Rz = torch.tensor([
            [torch.cos(angles_rad[2]), -torch.sin(angles_rad[2]), 0],
            [torch.sin(angles_rad[2]), torch.cos(angles_rad[2]), 0],
            [0, 0, 1]
        ], device=points.device)
        
        # Combined rotation
        R = Rz @ Ry @ Rx
        return points @ R.T
    
    def random_scale_3d(self, points: torch.Tensor, scale_range: Tuple[float, float] = (0.8, 1.2)) -> torch.Tensor:
        """Apply random 3D scaling."""
        if torch.rand(1) > self.augmentation_prob:
            return points
        
        scale = torch.rand(1) * (scale_range[1] - scale_range[0]) + scale_range[0]
        return points * scale
    
    def random_jitter_3d(self, points: torch.Tensor, std: float = 0.01) -> torch.Tensor:
        """Add random noise to 3D points."""
        if torch.rand(1) > self.augmentation_prob:
            return points
        
        noise = torch.randn_like(points) * std
        return points + noise
    
    def random_dropout_3d(self, points: torch.Tensor, dropout_ratio: float = 0.1) -> torch.Tensor:
        """Randomly drop 3D points."""
        if torch.rand(1) > self.augmentation_prob:
            return points
        
        n_points = points.shape[0]
        n_keep = int(n_points * (1 - dropout_ratio))
        indices = torch.randperm(n_points)[:n_keep]
        return points[indices]
    
    def apply_augmentation(self, points: torch.Tensor) -> torch.Tensor:
        """Apply all augmentation techniques."""
        points = self.random_rotation_3d(points)
        points = self.random_scale_3d(points)
        points = self.random_jitter_3d(points)
        points = self.random_dropout_3d(points)
        return points

class CurriculumLearning:
    """Curriculum learning with progressive difficulty."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.current_difficulty = 0
        self.max_difficulty = config.get('max_difficulty', 5)
        self.difficulty_threshold = config.get('difficulty_threshold', 0.8)
        self.patience = config.get('patience', 10)
        self.best_metric = float('inf')
        self.patience_counter = 0
        
    def should_increase_difficulty(self, current_metric: float) -> bool:
        """Check if difficulty should be increased."""
        if current_metric < self.best_metric:
            self.best_metric = current_metric
            self.patience_counter = 0
        else:
            self.patience_counter += 1
        
        # Increase difficulty if metric improves or patience runs out
        return (current_metric < self.difficulty_threshold or 
                self.patience_counter >= self.patience)
    
    def increase_difficulty(self) -> bool:
        """Increase curriculum difficulty."""
        if self.current_difficulty < self.max_difficulty:
            self.current_difficulty += 1
            self.patience_counter = 0
            logger.info(f"Increased curriculum difficulty to {self.current_difficulty}")
            return True
        return False
    
    def get_current_difficulty(self) -> int:
        """Get current difficulty level."""
        return self.current_difficulty

class EnhancedTrainer:
    """Enhanced training pipeline with advanced techniques."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
        
        # Initialize components
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.loss_fn = AdvancedLossFunction(config.get('loss_config', {}))
        self.augmentation = DataAugmentation3D(config.get('augmentation_config', {}))
        self.curriculum = CurriculumLearning(config.get('curriculum_config', {}))
        
        # Training state
        self.epoch = 0
        self.num_epochs = self.config.get('epochs', 100)
        self.global_step = 0
        self.best_metric = float('inf')
        self.early_stop_counter = 0
        
        # Mixed precision training
        self.mixed_precision = config.get('mixed_precision', True)
        self.scaler = GradScaler(enabled=self.mixed_precision)
        
        # Distributed training
        self.distributed = config.get('distributed', False)
        if self.distributed:
            self.setup_distributed()
        
        # Logging
        self.use_wandb = config.get('use_wandb', False)
        if self.use_wandb:
            wandb.init(project=config.get('project_name', '3d-reconstruction'))
    
    def setup_distributed(self):
        """Setup distributed training."""
        if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
            rank = int(os.environ['RANK'])
            world_size = int(os.environ['WORLD_SIZE'])
            dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)
            torch.cuda.set_device(rank)
            self.device = torch.device(f'cuda:{rank}')
            logger.info(f"Distributed training setup: rank {rank}/{world_size}")
    
    def create_model(self, model_config: Dict) -> Enhanced3DReconstructionModel:
        """Create the enhanced 3D reconstruction model."""
        model = Enhanced3DReconstructionModel(model_config)
        
        if self.distributed:
            model = model.to(self.device)
            model = DDP(model, device_ids=[self.device.index])
        else:
            model = model.to(self.device)
        
        return model
    
    def create_optimizer(self, model: nn.Module) -> torch.optim.Optimizer:
        """Create optimizer with different learning rates for different components."""
        # Separate parameters for different components
        encoder_params = []
        decoder_params = []
        other_params = []
        
        for name, param in model.named_parameters():
            if 'encoder' in name:
                encoder_params.append(param)
            elif 'decoder' in name:
                decoder_params.append(param)
            else:
                other_params.append(param)
        
        # Different learning rates for different components
        optimizer_config = self.config.get('optimizer_config', {})
        base_lr = optimizer_config.get('lr', 1e-4)
        
        param_groups = [
            {'params': encoder_params, 'lr': base_lr * 0.1},  # Lower LR for encoder
            {'params': decoder_params, 'lr': base_lr},          # Base LR for decoder
            {'params': other_params, 'lr': base_lr * 0.5}     # Medium LR for others
        ]
        
        optimizer_type = optimizer_config.get('type', 'adamw')
        if optimizer_type.lower() == 'adamw':
            optimizer = optim.AdamW(param_groups, weight_decay=optimizer_config.get('weight_decay', 1e-4))
        elif optimizer_type.lower() == 'adam':
            optimizer = optim.Adam(param_groups)
        else:
            optimizer = optim.SGD(param_groups, momentum=0.9)
        
        return optimizer
    
    def create_scheduler(self, optimizer: torch.optim.Optimizer):
        """Create learning rate scheduler with warm-up and cosine annealing."""
        scheduler_config = self.config.get('scheduler_config', {})
        
        # Warm-up scheduler
        warmup_epochs = scheduler_config.get('warmup_epochs', 5)
        total_epochs = self.config.get('epochs', 100)
        
        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                return epoch / warmup_epochs
            else:
                # Cosine annealing
                progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
                return 0.5 * (1 + np.cos(np.pi * progress))
        
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        
        return scheduler

    def _rebalance_losses(self):
        return

    def _move_to_device(self, batch: Any) -> Any:
        if isinstance(batch, dict):
            return {k: (v.to(self.device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
        if isinstance(batch, (list, tuple)):
            t = type(batch)
            return t(self._move_to_device(x) for x in batch)
        if isinstance(batch, torch.Tensor):
            return batch.to(self.device)
        return batch

    def _apply_augmentation(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        pc = batch.get('pointcloud')
        if isinstance(pc, torch.Tensor):
            if pc.dim() == 3:
                bs = pc.size(0)
                target_n = pc.size(-1)
                augmented = []
                for i in range(bs):
                    pts = pc[i].permute(1, 0).contiguous()
                    pts = self.augmentation.apply_augmentation(pts)
                    n = pts.size(0)
                    if n > target_n:
                        idx = torch.randperm(n, device=pts.device)[:target_n]
                        pts = pts[idx]
                    elif n < target_n and n > 0:
                        extra = torch.randint(0, n, (target_n - n,), device=pts.device)
                        pts = torch.cat([pts, pts[extra]], dim=0)
                    augmented.append(pts.permute(1, 0).contiguous())
                batch['pointcloud'] = torch.stack(augmented, dim=0)
            elif pc.dim() == 2:
                pts = pc.permute(1, 0).contiguous()
                pts = self.augmentation.apply_augmentation(pts)
                target_n = pc.size(-1)
                n = pts.size(0)
                if n > target_n:
                    idx = torch.randperm(n, device=pts.device)[:target_n]
                    pts = pts[idx]
                elif n < target_n and n > 0:
                    extra = torch.randint(0, n, (target_n - n,), device=pts.device)
                    pts = torch.cat([pts, pts[extra]], dim=0)
                batch['pointcloud'] = pts.permute(1, 0).contiguous()
        return batch

    def early_stop(self, current_metric: float) -> bool:
        if current_metric < self.best_metric:
            self.best_metric = current_metric
            self.early_stop_counter = 0
            return False
        self.early_stop_counter += 1
        patience = self.config.get('early_stop_patience', 10)
        return self.early_stop_counter >= patience

    def save_checkpoint(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        state = {
            'model': self.model.state_dict() if self.model is not None else {},
            'optimizer': self.optimizer.state_dict() if self.optimizer is not None else {},
            'scheduler': self.scheduler.state_dict() if self.scheduler is not None else {},
            'epoch': self.epoch,
            'global_step': self.global_step,
            'best_metric': self.best_metric,
            'config': self.config,
        }
        torch.save(state, path)

    def load_checkpoint(self, path: str):
        if not os.path.isfile(path):
            return
        state = torch.load(path, map_location=self.device)
        if self.model is not None and 'model' in state:
            self.model.load_state_dict(state['model'])
        if self.optimizer is not None and 'optimizer' in state:
            self.optimizer.load_state_dict(state['optimizer'])
        if self.scheduler is not None and 'scheduler' in state:
            self.scheduler.load_state_dict(state['scheduler'])
        self.epoch = state.get('epoch', 0)
        self.global_step = state.get('global_step', 0)
        self.best_metric = state.get('best_metric', float('inf'))
    
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        
        epoch_losses = []
        epoch_metrics = {}
        
        progress_bar = tqdm(train_loader, desc=f'Epoch {self.epoch + 1}')
        
        for batch_idx, batch in enumerate(progress_bar):
            if batch_idx % 100 == 0:
                self._rebalance_losses()
            
            # Move data to device
            batch = self._move_to_device(batch)
            
            # Apply data augmentation
            batch = self._apply_augmentation(batch)

            images = batch.get('front') if batch.get('front') is not None else batch.get('side')
            targets = {}
            if batch.get('pointcloud') is not None:
                pc_tgt = batch['pointcloud']
                if pc_tgt.dim() == 3:
                    targets['point_cloud_target'] = pc_tgt.permute(0, 2, 1).contiguous()
                elif pc_tgt.dim() == 2:
                    targets['point_cloud_target'] = pc_tgt.permute(1, 0).unsqueeze(0).contiguous()
            if batch.get('camera_K') is not None:
                targets['camera_K'] = batch['camera_K']
            if batch.get('mask') is not None:
                targets['mask'] = batch['mask']
            
            # Forward pass with mixed precision
            with autocast(enabled=self.mixed_precision):
                model_out = self.model(images)
                predictions = {}
                if 'points' in model_out:
                    predictions['point_cloud_pred'] = model_out['points']
                if 'uncertainty' in model_out:
                    predictions['uncertainty'] = model_out['uncertainty']
                loss, loss_components = self.loss_fn(predictions, targets)
            
            # Backward pass
            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            # Update progress bar
            progress_bar.set_postfix(loss=loss.item())
            
            # Log metrics
            if self.use_wandb:
                wandb.log({'train_loss': loss.item(), 'global_step': self.global_step})
            
            epoch_losses.append(loss.item())
            self.global_step += 1
        
        return {'avg_loss': np.mean(epoch_losses)}
    
    def eval_epoch(self, val_loader: DataLoader) -> Dict[str, float]:
        """Evaluate for one epoch."""
        self.model.eval()
        
        epoch_losses = []
        epoch_metrics = {}
        
        progress_bar = tqdm(val_loader, desc=f'Eval Epoch {self.epoch + 1}')
        
        for batch in progress_bar:
            batch = self._move_to_device(batch)
            images = batch.get('front') if batch.get('front') is not None else batch.get('side')
            targets = {}
            if batch.get('pointcloud') is not None:
                pc_tgt = batch['pointcloud']
                if pc_tgt.dim() == 3:
                    targets['point_cloud_target'] = pc_tgt.permute(0, 2, 1).contiguous()
                elif pc_tgt.dim() == 2:
                    targets['point_cloud_target'] = pc_tgt.permute(1, 0).unsqueeze(0).contiguous()
            if batch.get('camera_K') is not None:
                targets['camera_K'] = batch['camera_K']
            if batch.get('mask') is not None:
                targets['mask'] = batch['mask']

            with torch.no_grad():
                with autocast(enabled=self.mixed_precision):
                    model_out = self.model(images)
                    predictions = {}
                    if 'points' in model_out:
                        predictions['point_cloud_pred'] = model_out['points']
                    if 'uncertainty' in model_out:
                        predictions['uncertainty'] = model_out['uncertainty']
                    loss, loss_components = self.loss_fn(predictions, targets)
            
            epoch_losses.append(loss.item())
            
            # Update progress bar
            progress_bar.set_postfix(loss=loss.item())
        
        avg_loss = np.mean(epoch_losses)
        
        if self.use_wandb:
            wandb.log({'val_loss': avg_loss, 'epoch': self.epoch})
        
        return {'avg_loss': avg_loss}
    
    def train(self, train_dataset, val_dataset):
        """Main training loop."""
        # Create data loaders
        train_sampler = DistributedSampler(train_dataset) if self.distributed else None
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.config.get('batch_size', 8), 
            shuffle=(train_sampler is None),
            sampler=train_sampler,
            num_workers=self.config.get('num_workers', 0),
            pin_memory=True
        )
        
        val_sampler = DistributedSampler(val_dataset) if self.distributed else None
        val_loader = DataLoader(
            val_dataset, 
            batch_size=self.config.get('batch_size', 8), 
            shuffle=False,
            sampler=val_sampler,
            num_workers=self.config.get('num_workers', 0),
            pin_memory=True
        )
        
        # Create model, optimizer, and scheduler
        model_config = self.config.get('model_config', {})
        self.model = self.create_model(model_config)
        self.optimizer = self.create_optimizer(self.model)
        self.scheduler = self.create_scheduler(self.optimizer)
        
        # Load checkpoint if provided
        if self.config.get('load_checkpoint'):
            self.load_checkpoint(self.config['load_checkpoint'])
        
        for epoch in range(self.num_epochs):
            self.epoch = epoch
            
            if self.distributed:
                train_sampler.set_epoch(epoch)
            
            # Train for one epoch
            train_metrics = self.train_epoch(train_loader)
            
            # Evaluate for one epoch
            val_metrics = self.eval_epoch(val_loader)
            
            # Update scheduler
            self.scheduler.step()
            
            # Log metrics
            logger.info(f"Epoch {epoch + 1}: Train Loss: {train_metrics['avg_loss']:.4f}, Val Loss: {val_metrics['avg_loss']:.4f}")
            
            # Check for early stopping
            if self.early_stop(val_metrics['avg_loss']):
                logger.info("Early stopping triggered.")
                break
            
            # Save checkpoint periodically
            if (epoch + 1) % self.config.get('checkpoint_interval', 10) == 0:
                checkpoint_path = Path(self.config.get('checkpoint_dir', 'checkpoints')) / f'epoch_{epoch + 1}.pth'
                self.save_checkpoint(str(checkpoint_path))
        
        logger.info("Training completed!")
        
        # Final model save
        final_path = Path(self.config.get('checkpoint_dir', 'checkpoints')) / 'final_model.pth'
        self.save_checkpoint(str(final_path))

def create_training_config() -> Dict:
    """Create default training configuration."""
    return {
        # Model configuration
        'model_config': {
            'encoder_type': 'cnn',  # 'cnn' or 'vit'
            'representation': 'sdf',  # 'sdf', 'occupancy', 'nerf'
            'img_size': 224,
            'base_channels': 64,
            'encoder_depth': 4,
            'decoder_hidden_dim': 512,
            'decoder_depth': 8,
            'uncertainty_quantification': True,
            'positional_encoding': True
        },
        
        # Training configuration
        'epochs': 100,
        'batch_size': 8,
        'mixed_precision': True,
        'distributed': False,
        'device': 'cuda',
        
        # Optimizer configuration
        'optimizer_config': {
            'type': 'adamw',
            'lr': 1e-4,
            'weight_decay': 1e-4
        },
        
        # Scheduler configuration
        'scheduler_config': {
            'warmup_epochs': 5
        },
        
        # Loss configuration
        'loss_config': {
            'sdf_loss_weight': 1.0,
            'occupancy_loss_weight': 1.0,
            'normal_loss_weight': 0.1,
            'laplacian_loss_weight': 0.01,
            'chamfer_loss_weight': 1.0,
            'uncertainty_loss_weight': 0.1,
            'focal_gamma': 2.0,
            'focal_alpha': 0.75
        },
        
        # Data augmentation configuration
        'augmentation_config': {
            'augmentation_prob': 0.5
        },
        
        # Curriculum learning configuration
        'curriculum_config': {
            'max_difficulty': 5,
            'difficulty_threshold': 0.8,
            'patience': 10
        },
        
        # Other configurations
        'early_stop_patience': 10,
        'checkpoint_interval': 10,
        'checkpoint_dir': 'checkpoints',
        'use_wandb': False,
        'project_name': '3d-reconstruction-enhanced'
    }
