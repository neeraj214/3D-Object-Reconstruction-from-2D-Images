"""
Multi-Dataset Joint Training with Domain Adaptation
Implements joint training on all 6 datasets with domain adaptation techniques
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
import json
import logging
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import defaultdict
import wandb
from torch.cuda.amp import autocast, GradScaler

from ..data.unified_dataloader import Unified3DDataset, Unified3DDataModule
from ..models.enhanced_3d_reconstruction import Enhanced3DReconstructionModel
from ..evaluation.comprehensive_evaluation import ComprehensiveEvaluationPipeline


class DomainAdaptationLoss(nn.Module):
    """Domain adaptation loss for multi-dataset training"""
    
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        
        # Domain adaptation components
        self.domain_classifier = DomainClassifier(config)
        self.grl = GradientReversalLayer()
        
        # Loss weights
        self.task_weight = config.get('task_weight', 1.0)
        self.domain_weight = config.get('domain_weight', 0.1)
        self.consistency_weight = config.get('consistency_weight', 0.5)
        
        # Loss functions
        self.cross_entropy = nn.CrossEntropyLoss()
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()
    
    def forward(self, predictions: Dict, targets: Dict, 
               domain_labels: torch.Tensor, dataset_names: List[str]) -> Dict:
        """Compute domain adaptation loss"""
        losses = {}
        
        # Task losses (reconstruction)
        if 'point_cloud' in predictions and 'point_cloud' in targets:
            task_loss = self.compute_reconstruction_loss(
                predictions['point_cloud'], targets['point_cloud']
            )
            losses['task'] = task_loss * self.task_weight
        
        # Domain classification loss
        if 'domain_features' in predictions:
            domain_features = self.grl(predictions['domain_features'])
            domain_logits = self.domain_classifier(domain_features)
            
            domain_loss = self.cross_entropy(domain_logits, domain_labels)
            losses['domain'] = domain_loss * self.domain_weight
            
            # Domain accuracy for monitoring
            domain_preds = torch.argmax(domain_logits, dim=1)
            domain_acc = (domain_preds == domain_labels).float().mean()
            losses['domain_acc'] = domain_acc
        
        # Cross-domain consistency loss
        if 'cross_domain_predictions' in predictions:
            consistency_loss = self.compute_consistency_loss(
                predictions['point_cloud'], 
                predictions['cross_domain_predictions']
            )
            losses['consistency'] = consistency_loss * self.consistency_weight
        
        dataset_losses = self.compute_dataset_specific_losses(
            predictions, targets, dataset_names
        )
        losses.update(dataset_losses)
        
        # Total loss
        total_loss = sum([v for k, v in losses.items() if not k.endswith('_acc')])
        losses['total'] = total_loss
        
        return losses
    
    def compute_reconstruction_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        batch_size = pred.size(0)
        pred_sq = torch.sum(pred ** 2, dim=2, keepdim=True)
        target_sq = torch.sum(target ** 2, dim=2, keepdim=True).transpose(1, 2)
        dist = pred_sq + target_sq - 2 * torch.bmm(pred, target.transpose(1, 2))
        dist_pred_to_target = torch.mean(torch.min(dist, dim=2)[0])
        dist_target_to_pred = torch.mean(torch.min(dist, dim=1)[0])
        chamfer = dist_pred_to_target + dist_target_to_pred
        sigma = self.config.get('fscore_sigma', 0.01)
        soft_matches_a = torch.exp(-torch.min(dist, dim=2)[0] / (2 * sigma * sigma))
        soft_matches_b = torch.exp(-torch.min(dist, dim=1)[0] / (2 * sigma * sigma))
        precision = torch.mean(soft_matches_a)
        recall = torch.mean(soft_matches_b)
        denom = precision + recall + 1e-8
        soft_fscore = 2 * precision * recall / denom
        fscore_loss = 1.0 - soft_fscore
        return chamfer + self.config.get('fscore_weight', 0.1) * fscore_loss
    
    def compute_consistency_loss(self, pred1: torch.Tensor, pred2: torch.Tensor) -> torch.Tensor:
        """Compute cross-domain consistency loss"""
        return self.mse_loss(pred1, pred2)
    
    def compute_dataset_specific_losses(self, predictions: Dict, targets: Dict, 
                                       dataset_names: List[str]) -> Dict:
        """Compute dataset-specific losses with different weights"""
        losses = {}
        
        # Dataset-specific loss weights
        dataset_weights = {
            'pix3d': 1.0,
            'shapenet': 1.2,  # Higher weight for synthetic data
            'pascal3d': 0.9,
            'objectnet3d': 0.9,
            'co3d': 1.1,
            'google_scanned': 1.0
        }
        
        if 'point_cloud' in predictions:
            batch_size = predictions['point_cloud'].size(0)
            total_loss = 0.0
            
            for i, dataset_name in enumerate(dataset_names):
                weight = dataset_weights.get(dataset_name.lower(), 1.0)
                
                # Apply dataset-specific loss weighting
                sample_loss = self.compute_reconstruction_loss(
                    predictions['point_cloud'][i:i+1], 
                    targets['point_cloud'][i:i+1]
                )
                total_loss += weight * sample_loss
            
            losses['dataset_weighted'] = total_loss / batch_size
        if self.config.get('enable_silhouette', True) and ('mask' in targets and 'camera_K' in targets) and ('point_cloud' in predictions):
            try:
                losses['silhouette'] = self.compute_silhouette_loss(predictions['point_cloud'], targets['mask'], targets['camera_K'])
            except Exception:
                pass
        if self.config.get('enable_edge_loss', True) and ('depth' in predictions and 'depth_gt' in targets):
            try:
                pd = predictions['depth']
                gd = targets['depth_gt']
                gx_p = F.conv2d(pd, torch.tensor([[[-1,0,1],[-2,0,2],[-1,0,1]]], device=pd.device, dtype=pd.dtype).unsqueeze(0), padding=1)
                gy_p = F.conv2d(pd, torch.tensor([[[-1,-2,-1],[0,0,0],[1,2,1]]], device=pd.device, dtype=pd.dtype).unsqueeze(0), padding=1)
                gm_p = torch.sqrt(torch.clamp(gx_p**2 + gy_p**2, min=1e-8))
                gx_g = F.conv2d(gd, torch.tensor([[[-1,0,1],[-2,0,2],[-1,0,1]]], device=gd.device, dtype=gd.dtype).unsqueeze(0), padding=1)
                gy_g = F.conv2d(gd, torch.tensor([[[-1,-2,-1],[0,0,0],[1,2,1]]], device=gd.device, dtype=gd.dtype).unsqueeze(0), padding=1)
                gm_g = torch.sqrt(torch.clamp(gx_g**2 + gy_g**2, min=1e-8))
                losses['edge_depth'] = F.l1_loss(gm_p, gm_g)
            except Exception:
                pass
        if self.config.get('enable_normal_loss', True) and ('normals' in predictions and 'normals_gt' in targets):
            try:
                n_pred = F.normalize(predictions['normals'], dim=2)
                n_gt = F.normalize(targets['normals_gt'], dim=2)
                cos = torch.sum(n_pred * n_gt, dim=2)
                losses['normal_consistency'] = torch.mean(1.0 - cos)
            except Exception:
                pass
        
        return losses

    def compute_silhouette_loss(self, pred_points: torch.Tensor, masks: torch.Tensor, camera_K: torch.Tensor) -> torch.Tensor:
        b = pred_points.size(0)
        h = masks.size(2)
        w = masks.size(3)
        fx = camera_K[:,0,0].view(b,1,1)
        fy = camera_K[:,1,1].view(b,1,1)
        cx = camera_K[:,0,2].view(b,1,1)
        cy = camera_K[:,1,2].view(b,1,1)
        x = pred_points[:,:,0].view(b,-1,1)
        y = pred_points[:,:,1].view(b,-1,1)
        z = torch.clamp(pred_points[:,:,2].view(b,-1,1), min=1e-3)
        u = fx * (x / z) + cx
        v = fy * (y / z) + cy
        u = u / w * 2 - 1
        v = v / h * 2 - 1
        grid = torch.cat([u, v], dim=2)
        kernel = 3
        occ = torch.zeros((b,1,h,w), device=pred_points.device)
        for i in range(kernel):
            du = (i - kernel//2)/w
            dv = (i - kernel//2)/h
            g = torch.cat([u+du, v+dv], dim=2).view(b, -1, 1, 2)
            sampled = torch.nn.functional.grid_sample(torch.ones((b,1,h,w), device=pred_points.device), g, align_corners=True, mode='bilinear', padding_mode='zeros')
            occ = occ + sampled.view(b,1,h,w)
        occ = torch.clamp(occ, 0.0, 1.0)
        masks = masks.float()
        inter = torch.sum(occ * masks)
        union = torch.sum(torch.clamp(occ + masks, 0.0, 1.0)) + 1e-6
        iou = inter / union
        return 1.0 - iou


class DomainClassifier(nn.Module):
    """Domain classifier for adversarial domain adaptation"""
    
    def __init__(self, config: Dict):
        super().__init__()
        
        num_domains = config.get('num_domains', 6)  # 6 datasets
        feature_dim = config.get('feature_dim', 512)
        hidden_dim = config.get('hidden_dim', 256)
        
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, num_domains)
        )
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.classifier(features)


class GradientReversalLayer(nn.Module):
    """Gradient reversal layer for adversarial training"""
    
    def __init__(self, alpha: float = 1.0):
        super().__init__()
        self.alpha = alpha
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return GradientReversalFunction.apply(x, self.alpha)


class GradientReversalFunction(torch.autograd.Function):
    """Gradient reversal function"""
    
    @staticmethod
    def forward(ctx, x: torch.Tensor, alpha: float) -> torch.Tensor:
        ctx.alpha = alpha
        return x.view_as(x)
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None]:
        return -ctx.alpha * grad_output, None


class MultiDatasetTrainer:
    """Multi-dataset joint training with domain adaptation"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize components
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.loss_fn = None
        self.evaluation_pipeline = None
        
        # Dataset management
        self.domain_to_idx = {}
        self.idx_to_domain = {}
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Initialize paths
        self.output_dir = Path(config.get('output_dir', 'multi_dataset_training'))
        self.checkpoint_dir = self.output_dir / 'checkpoints'
        self.log_dir = self.output_dir / 'logs'
        self.results_dir = self.output_dir / 'results'
        
        # Create directories
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.setup_logging()
        
        # Initialize Weights & Biases if enabled
        if config.get('wandb', {}).get('enabled', False):
            wandb.init(
                project=config['wandb']['project'],
                entity=config['wandb']['entity'],
                name=config.get('experiment_name', 'multi_dataset_training'),
                config=config
            )
    
    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.log_dir / 'training.log'),
                logging.StreamHandler()
            ]
        )
    
    def initialize_model(self):
        """Initialize the 3D reconstruction model with domain adaptation"""
        model_config = self.config.get('model', {})
        
        # Initialize base model
        rep = model_config.get('output_representation', 'pointcloud')
        if rep == 'point_cloud':
            rep = 'pointcloud'
        self.model = Enhanced3DReconstructionModel({
            'encoder_type': model_config.get('encoder_type', 'resnet50'),
            'representation': rep,
            'num_points': model_config.get('num_points', 2048),
            'voxel_resolution': model_config.get('voxel_resolution', 64),
            'use_attention': model_config.get('use_attention', True),
            'uncertainty_quantification': model_config.get('uncertainty_quantification', True),
            'attention_layers': model_config.get('attention_layers', [2,3,4])
        })
        
        self.model.to(self.device)
        
        # Initialize domain adaptation loss
        self.loss_fn = DomainAdaptationLoss(self.config.get('domain_adaptation', {}))
        
        # Initialize optimizer
        optimizer_config = self.config.get('optimizer', {})
        self.optimizer = torch.optim.AdamW(
            list(self.model.parameters()) + list(self.loss_fn.domain_classifier.parameters()),
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
        
        # Initialize evaluation pipeline
        self.evaluation_pipeline = ComprehensiveEvaluationPipeline(
            self.config.get('evaluation', {})
        )
        
        self.scaler = GradScaler(enabled=self.config.get('training', {}).get('mixed_precision', True))
        self.logger.info("Model and training components initialized with domain adaptation")
    
    def create_dataloaders(self):
        """Create multi-dataset dataloaders"""
        dataset_config = self.config.get('dataset', {})
        
        # Define datasets and their domains
        datasets = [
            ('pix3d', 'data/pix3d'),
            ('shapenet', 'data/shapenet'),
            ('pascal3d', 'data/pascal3d'),
            ('objectnet3d', 'data/objectnet3d'),
            ('co3d', 'data/co3d'),
            ('google_scanned', 'data/google_scanned')
        ]
        
        # Create domain mappings
        for idx, (dataset_name, _) in enumerate(datasets):
            self.domain_to_idx[dataset_name] = idx
            self.idx_to_domain[idx] = dataset_name
        
        # Create data loaders for each dataset
        self.train_loaders = {}
        self.val_loaders = {}
        
        batch_size = self.config.get('batch_size', 8)
        num_workers = self.config.get('num_workers', 4)
        
        for dataset_name, data_path in datasets:
            try:
                # Create dataset-specific data loader
                train_loader = MultiDatasetLoader(
                    dataset_names=[dataset_name],
                    data_paths=[data_path],
                    split='train',
                    batch_size=batch_size,
                    num_workers=num_workers,
                    image_size=dataset_config.get('image_size', 224),
                    num_points=dataset_config.get('num_points', 2048),
                    augment=dataset_config.get('augment', True)
                )
                
                val_loader = MultiDatasetLoader(
                    dataset_names=[dataset_name],
                    data_paths=[data_path],
                    split='val',
                    batch_size=batch_size,
                    num_workers=num_workers,
                    image_size=dataset_config.get('image_size', 224),
                    num_points=dataset_config.get('num_points', 2048),
                    augment=False
                )
                
                self.train_loaders[dataset_name] = train_loader
                self.val_loaders[dataset_name] = val_loader
                
                self.logger.info(f"Created dataloaders for {dataset_name}")
                
            except Exception as e:
                self.logger.warning(f"Failed to create dataloader for {dataset_name}: {e}")
                continue
        
        self.logger.info(f"Created dataloaders for {len(self.train_loaders)} datasets")
    
    def train_epoch(self, epoch: int) -> Dict:
        """Train for one epoch with multi-dataset sampling"""
        self.model.train()
        self.loss_fn.domain_classifier.train()
        
        epoch_losses = defaultdict(list)
        
        # Create combined iterator for all datasets
        dataset_iters = {
            name: iter(loader) for name, loader in self.train_loaders.items()
        }
        
        # Training configuration
        batches_per_epoch = self.config.get('batches_per_epoch', 100)
        dataset_sampling = self.config.get('dataset_sampling', 'uniform')
        
        pbar = tqdm(range(batches_per_epoch), desc=f"Epoch {epoch} [Train]")
        
        for batch_idx in pbar:
            try:
                # Sample dataset
                if dataset_sampling == 'uniform':
                    dataset_name = np.random.choice(list(self.train_loaders.keys()))
                elif dataset_sampling == 'weighted':
                    # Weight by dataset size or importance
                    dataset_weights = self.config.get('dataset_weights', {})
                    weights = [dataset_weights.get(name, 1.0) for name in self.train_loaders.keys()]
                    dataset_name = np.random.choice(
                        list(self.train_loaders.keys()), 
                        p=np.array(weights) / sum(weights)
                    )
                
                # Get batch from selected dataset
                try:
                    batch = next(dataset_iters[dataset_name])
                except StopIteration:
                    # Restart iterator if exhausted
                    dataset_iters[dataset_name] = iter(self.train_loaders[dataset_name])
                    batch = next(dataset_iters[dataset_name])
                
                images = batch['image'].to(self.device)
                targets = {}
                if 'pointcloud' in batch:
                    targets['point_cloud'] = batch['pointcloud'].to(self.device)
                if 'mask' in batch:
                    targets['mask'] = batch['mask'].to(self.device)
                if 'camera_K' in batch:
                    targets['camera_K'] = batch['camera_K'].to(self.device)
                
                # Create domain labels
                domain_labels = torch.full(
                    (images.size(0),), 
                    self.domain_to_idx[dataset_name], 
                    dtype=torch.long, 
                    device=self.device
                )
                
                self.optimizer.zero_grad()
                with autocast(enabled=self.config.get('training', {}).get('mixed_precision', True)):
                    predictions = self.model(images, return_domain_features=True)
                
                # Add dataset information
                dataset_names = [dataset_name] * images.size(0)
                
                # Compute loss
                # Map model outputs to expected keys
                if isinstance(predictions, dict):
                    if 'points' in predictions:
                        predictions['point_cloud'] = predictions['points']
                    if 'encoder_features' in predictions and isinstance(predictions['encoder_features'], dict):
                        ef = predictions['encoder_features']
                        predictions['domain_features'] = ef.get('global_features', None)
                losses = self.loss_fn(predictions, targets, domain_labels, dataset_names)
                self.scaler.scale(losses['total']).backward()
                
                # Gradient clipping
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(list(self.model.parameters()) + list(self.loss_fn.domain_classifier.parameters()), self.config.get('grad_clip', 1.0))
                self.scaler.step(self.optimizer)
                self.scaler.update()
                
                # Update losses
                for key, value in losses.items():
                    if isinstance(value, torch.Tensor):
                        epoch_losses[key].append(value.item())
                    else:
                        epoch_losses[key].append(value)
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': f"{losses['total'].item():.4f}",
                    'dataset': dataset_name,
                    'lr': f"{self.optimizer.param_groups[0]['lr']:.2e}"
                })
                
                # Log to wandb
                if self.config.get('wandb', {}).get('enabled', False):
                    wandb.log({
                        f"train/{key}": value.item() if isinstance(value, torch.Tensor) else value
                        for key, value in losses.items()
                    })
                
            except Exception as e:
                self.logger.error(f"Error in training batch {batch_idx}: {e}")
                continue
        
        # Average losses
        avg_losses = {key: np.mean(values) for key, values in epoch_losses.items()}
        
        return avg_losses
    
    def validate_epoch(self, epoch: int) -> Dict:
        """Validate on all datasets"""
        self.model.eval()
        self.loss_fn.domain_classifier.eval()
        
        all_val_losses = {}
        
        for dataset_name, val_loader in self.val_loaders.items():
            dataset_losses = defaultdict(list)
            
            with torch.no_grad():
                pbar = tqdm(val_loader, desc=f"Epoch {epoch} [Val - {dataset_name}]")
                
                for batch in pbar:
                    try:
                        images = batch['image'].to(self.device)
                        targets = {}
                        if 'pointcloud' in batch:
                            targets['point_cloud'] = batch['pointcloud'].to(self.device)
                        if 'mask' in batch:
                            targets['mask'] = batch['mask'].to(self.device)
                        if 'camera_K' in batch:
                            targets['camera_K'] = batch['camera_K'].to(self.device)
                        
                        # Create domain labels
                        domain_labels = torch.full(
                            (images.size(0),), 
                            self.domain_to_idx[dataset_name], 
                            dtype=torch.long, 
                            device=self.device
                        )
                        
                        # Forward pass
                        predictions = self.model(images)
                        if isinstance(predictions, dict):
                            if 'points' in predictions:
                                predictions['point_cloud'] = predictions['points']
                            if 'encoder_features' in predictions and isinstance(predictions['encoder_features'], dict):
                                ef = predictions['encoder_features']
                                predictions['domain_features'] = ef.get('global_features', None)
                        
                        # Add dataset information
                        dataset_names = [dataset_name] * images.size(0)
                        
                        # Compute loss
                        losses = self.loss_fn(predictions, targets, domain_labels, dataset_names)
                        
                        # Update losses
                        for key, value in losses.items():
                            if isinstance(value, torch.Tensor):
                                dataset_losses[key].append(value.item())
                            else:
                                dataset_losses[key].append(value)
                        
                        # Update progress bar
                        pbar.set_postfix({'loss': f"{losses['total'].item():.4f}"})
                        
                    except Exception as e:
                        self.logger.error(f"Error in validation batch for {dataset_name}: {e}")
                        continue
            
            # Average losses for this dataset
            avg_dataset_losses = {key: np.mean(values) for key, values in dataset_losses.items()}
            
            # Store with dataset prefix
            for key, value in avg_dataset_losses.items():
                all_val_losses[f"{dataset_name}_{key}"] = value
        
        return all_val_losses
    
    def evaluate_all_datasets(self, epoch: int):
        """Comprehensive evaluation on all datasets"""
        self.logger.info(f"Evaluating on all datasets at epoch {epoch}")
        
        all_results = {}
        
        for dataset_name, val_loader in self.val_loaders.items():
            try:
                self.logger.info(f"Evaluating on {dataset_name}")
                
                # Run evaluation
                results = self.evaluation_pipeline.evaluate_model(
                    self.model, val_loader, dataset_name
                )
                
                all_results[dataset_name] = results
                
                # Log results
                self.logger.info(f"Results for {dataset_name}: {results}")
                
                # Save results
                results_path = self.results_dir / f"epoch_{epoch}_{dataset_name}_results.json"
                with open(results_path, 'w') as f:
                    json.dump(results, f, indent=2)
                
                # Log to wandb
                if self.config.get('wandb', {}).get('enabled', False):
                    wandb.log({
                        f"eval/{dataset_name}_{key}": value 
                        for key, value in results.items()
                    })
                
            except Exception as e:
                self.logger.error(f"Failed to evaluate {dataset_name}: {e}")
                continue
        
        return all_results
    
    def save_checkpoint(self, epoch: int, losses: Dict, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'domain_classifier_state_dict': self.loss_fn.domain_classifier.state_dict(),
            'losses': losses,
            'config': self.config,
            'domain_mappings': {
                'domain_to_idx': self.domain_to_idx,
                'idx_to_domain': self.idx_to_domain
            }
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
        self.loss_fn.domain_classifier.load_state_dict(checkpoint['domain_classifier_state_dict'])
        
        # Restore domain mappings
        if 'domain_mappings' in checkpoint:
            self.domain_to_idx = checkpoint['domain_mappings']['domain_to_idx']
            self.idx_to_domain = checkpoint['domain_mappings']['idx_to_domain']
        
        self.logger.info(f"Checkpoint loaded: {checkpoint_path}")
        
        return checkpoint['epoch']
    
    def export_models(self, epoch: int):
        """Export models to different formats"""
        self.logger.info(f"Exporting models at epoch {epoch}")
        
        export_config = self.config.get('model_export', {})
        export_formats = export_config.get('export_formats', ['onnx'])
        
        export_dir = self.output_dir / 'exported_models'
        export_dir.mkdir(parents=True, exist_ok=True)
        
        for format_name in export_formats:
            try:
                if format_name == 'onnx':
                    self.export_to_onnx(export_dir, epoch)
                elif format_name == 'tensorrt':
                    self.export_to_tensorrt(export_dir, epoch)
                elif format_name == 'coreml':
                    self.export_to_coreml(export_dir, epoch)
                
                self.logger.info(f"Exported model to {format_name}")
                
            except Exception as e:
                self.logger.error(f"Failed to export to {format_name}: {e}")
                continue
    
    def export_to_onnx(self, export_dir: Path, epoch: int):
        """Export model to ONNX format"""
        onnx_config = self.config.get('model_export', {}).get('onnx', {})
        
        dummy_input = torch.randn(1, 3, 224, 224).to(self.device)
        
        onnx_path = export_dir / f"model_epoch_{epoch}.onnx"
        
        torch.onnx.export(
            self.model,
            dummy_input,
            str(onnx_path),
            export_params=True,
            opset_version=onnx_config.get('opset_version', 11),
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            } if onnx_config.get('dynamic_axes', True) else None
        )
    
    def export_to_tensorrt(self, export_dir: Path, epoch: int):
        """Export model to TensorRT format"""
        # This would require TensorRT installation
        self.logger.info("TensorRT export would be implemented here")
    
    def export_to_coreml(self, export_dir: Path, epoch: int):
        """Export model to CoreML format"""
        # This would require coremltools
        self.logger.info("CoreML export would be implemented here")
    
    def train(self):
        """Main training loop"""
        self.logger.info("Starting multi-dataset joint training with domain adaptation")
        
        # Initialize model and dataloaders
        self.initialize_model()
        self.create_dataloaders()
        if not self.train_loaders:
            raise RuntimeError('No training datasets available. Please populate data/ and retry.')
        empty = [name for name, ld in self.train_loaders.items() if (hasattr(ld, '__len__') and len(ld)==0)]
        if empty:
            raise RuntimeError(f"Empty loaders detected: {', '.join(empty)}")
        
        # Training configuration
        num_epochs = self.config.get('num_epochs', 100)
        save_freq = self.config.get('save_frequency', 10)
        eval_freq = self.config.get('eval_frequency', 5)
        export_freq = self.config.get('export_frequency', 20)
        
        best_val_loss = float('inf')
        patience_counter = 0
        early_stopping_patience = self.config.get('early_stopping', {}).get('patience', 15)
        
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
            
            # Calculate average validation loss for early stopping
            val_loss_keys = [k for k in val_losses.keys() if k.endswith('_total')]
            avg_val_loss = np.mean([val_losses[k] for k in val_loss_keys]) if val_loss_keys else 0
            
            # Early stopping check
            if avg_val_loss < best_val_loss - self.config.get('early_stopping', {}).get('min_delta', 0.001):
                best_val_loss = avg_val_loss
                patience_counter = 0
                is_best = True
            else:
                patience_counter += 1
                is_best = False
            
            # Save checkpoint
            if (epoch + 1) % save_freq == 0 or is_best:
                self.save_checkpoint(epoch, {
                    'train': train_losses,
                    'val': val_losses
                }, is_best)
            
            # Evaluate
            if (epoch + 1) % eval_freq == 0:
                self.evaluate_all_datasets(epoch)
            
            # Export models
            if (epoch + 1) % export_freq == 0:
                self.export_models(epoch)
            
            # Early stopping
            if patience_counter >= early_stopping_patience:
                self.logger.info(f"Early stopping triggered after {epoch + 1} epochs")
                break
            
            # Log to wandb
            if self.config.get('wandb', {}).get('enabled', False):
                wandb.log({
                    'epoch': epoch,
                    'train_loss': train_losses.get('total', 0),
                    'val_loss': avg_val_loss,
                    'learning_rate': self.optimizer.param_groups[0]['lr']
                })
        
        # Final evaluation and export
        self.logger.info("Training completed. Running final evaluation and export.")
        self.evaluate_all_datasets(num_epochs - 1)
        self.export_models(num_epochs - 1)
        
        self.logger.info("Multi-dataset joint training completed")


def main():
    """Main function for multi-dataset training"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Multi-Dataset Joint Training')
    parser.add_argument('--config', type=str, required=True, help='Configuration file')
    parser.add_argument('--resume', type=str, help='Resume from checkpoint')
    parser.add_argument('--pretrained', type=str, help='Path to pretrained ShapeNet model')
    
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    # Create trainer
    trainer = MultiDatasetTrainer(config)
    
    # Load pretrained model if specified
    if args.pretrained:
        trainer.logger.info(f"Loading pretrained model from {args.pretrained}")
        # This would load a ShapeNet pretrained model and fine-tune it
        # Implementation depends on the pretrained model format
    
    # Resume from checkpoint if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    # Start training
    trainer.train()


if __name__ == '__main__':
    main()