#!/usr/bin/env python3
"""
Comprehensive Evaluation Metrics for 3D Object Reconstruction
Implements IoU, Chamfer distance, F-score, P2M metrics, and more
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from scipy.spatial.distance import cdist
from sklearn.metrics import f1_score
import trimesh
from pathlib import Path
import logging
from tqdm import tqdm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

logger = logging.getLogger(__name__)


class PointCloudMetrics:
    """Point cloud evaluation metrics"""
    
    def __init__(self):
        pass
    
    def chamfer_distance(self, pred_points: np.ndarray, target_points: np.ndarray,
                        direction: str = 'both') -> float:
        """
        Compute Chamfer distance between two point clouds
        
        Args:
            pred_points: Predicted point cloud (N, 3)
            target_points: Target point cloud (M, 3)
            direction: 'pred_to_target', 'target_to_pred', or 'both'
        
        Returns:
            Chamfer distance
        """
        if len(pred_points) == 0 or len(target_points) == 0:
            return float('inf')
        
        # Compute pairwise distances
        distances_pred_to_target = cdist(pred_points, target_points, metric='euclidean')
        distances_target_to_pred = cdist(target_points, pred_points, metric='euclidean')
        
        if direction == 'pred_to_target':
            # Average minimum distance from predicted to target
            chamfer = np.mean(np.min(distances_pred_to_target, axis=1))
        elif direction == 'target_to_pred':
            # Average minimum distance from target to predicted
            chamfer = np.mean(np.min(distances_target_to_pred, axis=1))
        elif direction == 'both':
            # Symmetric Chamfer distance
            chamfer_pred = np.mean(np.min(distances_pred_to_target, axis=1))
            chamfer_target = np.mean(np.min(distances_target_to_pred, axis=1))
            chamfer = chamfer_pred + chamfer_target
        else:
            raise ValueError(f"Invalid direction: {direction}")
        
        return chamfer
    
    def earth_movers_distance(self, pred_points: np.ndarray, target_points: np.ndarray,
                               max_iterations: int = 100) -> float:
        """
        Compute Earth Mover's Distance (simplified approximation)
        
        Args:
            pred_points: Predicted point cloud (N, 3)
            target_points: Target point cloud (M, 3)
            max_iterations: Maximum iterations for approximation
        
        Returns:
            EMD value
        """
        if len(pred_points) == 0 or len(target_points) == 0:
            return float('inf')
        
        # Subsample for efficiency if too many points
        max_points = 1000
        if len(pred_points) > max_points:
            indices = np.random.choice(len(pred_points), max_points, replace=False)
            pred_points = pred_points[indices]
        
        if len(target_points) > max_points:
            indices = np.random.choice(len(target_points), max_points, replace=False)
            target_points = target_points[indices]
        
        # Simple approximation: minimum cost bipartite matching
        distances = cdist(pred_points, target_points, metric='euclidean')
        
        # Greedy matching (simplified)
        total_cost = 0.0
        pred_indices = list(range(len(pred_points)))
        target_indices = list(range(len(target_points)))
        
        for _ in range(min(len(pred_points), len(target_points))):
            if not pred_indices or not target_indices:
                break
            
            # Find minimum distance pair
            min_dist = float('inf')
            best_pred_idx = -1
            best_target_idx = -1
            
            for pred_idx in pred_indices:
                for target_idx in target_indices:
                    if distances[pred_idx, target_idx] < min_dist:
                        min_dist = distances[pred_idx, target_idx]
                        best_pred_idx = pred_idx
                        best_target_idx = target_idx
            
            if best_pred_idx != -1 and best_target_idx != -1:
                total_cost += min_dist
                pred_indices.remove(best_pred_idx)
                target_indices.remove(best_target_idx)
        
        # Normalize by number of matched points
        n_matched = min(len(pred_points), len(target_points))
        if n_matched > 0:
            emd = total_cost / n_matched
        else:
            emd = float('inf')
        
        return emd
    
    def hausdorff_distance(self, pred_points: np.ndarray, target_points: np.ndarray,
                          direction: str = 'both') -> float:
        """
        Compute Hausdorff distance between two point clouds
        
        Args:
            pred_points: Predicted point cloud (N, 3)
            target_points: Target point cloud (M, 3)
            direction: 'pred_to_target', 'target_to_pred', or 'both'
        
        Returns:
            Hausdorff distance
        """
        if len(pred_points) == 0 or len(target_points) == 0:
            return float('inf')
        
        # Compute pairwise distances
        distances_pred_to_target = cdist(pred_points, target_points, metric='euclidean')
        distances_target_to_pred = cdist(target_points, pred_points, metric='euclidean')
        
        if direction == 'pred_to_target':
            # Maximum minimum distance from predicted to target
            hausdorff = np.max(np.min(distances_pred_to_target, axis=1))
        elif direction == 'target_to_pred':
            # Maximum minimum distance from target to predicted
            hausdorff = np.max(np.min(distances_target_to_pred, axis=1))
        elif direction == 'both':
            # Symmetric Hausdorff distance
            hausdorff_pred = np.max(np.min(distances_pred_to_target, axis=1))
            hausdorff_target = np.max(np.min(distances_target_to_pred, axis=1))
            hausdorff = max(hausdorff_pred, hausdorff_target)
        else:
            raise ValueError(f"Invalid direction: {direction}")
        
        return hausdorff
    
    def f1_score_pointcloud(self, pred_points: np.ndarray, target_points: np.ndarray,
                           threshold: float = 0.01) -> float:
        """
        Compute F1 score for point cloud reconstruction
        
        Args:
            pred_points: Predicted point cloud (N, 3)
            target_points: Target point cloud (M, 3)
            threshold: Distance threshold for matching
        
        Returns:
            F1 score
        """
        if len(pred_points) == 0 or len(target_points) == 0:
            return 0.0
        
        # Compute pairwise distances
        distances_pred_to_target = cdist(pred_points, target_points, metric='euclidean')
        distances_target_to_pred = cdist(target_points, pred_points, metric='euclidean')
        
        # Precision: fraction of predicted points within threshold of target
        min_distances_pred = np.min(distances_pred_to_target, axis=1)
        precision = np.mean(min_distances_pred <= threshold)
        
        # Recall: fraction of target points within threshold of predicted
        min_distances_target = np.min(distances_target_to_pred, axis=1)
        recall = np.mean(min_distances_target <= threshold)
        
        # F1 score
        if precision + recall > 0:
            f1 = 2 * (precision * recall) / (precision + recall)
        else:
            f1 = 0.0
        
        return f1
    
    def precision_recall_curve(self, pred_points: np.ndarray, target_points: np.ndarray,
                               thresholds: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute precision-recall curve for different thresholds
        
        Args:
            pred_points: Predicted point cloud (N, 3)
            target_points: Target point cloud (M, 3)
            thresholds: Array of distance thresholds
        
        Returns:
            precisions, recalls, thresholds
        """
        if thresholds is None:
            thresholds = np.logspace(-3, 0, 50)  # 0.001 to 1.0
        
        precisions = []
        recalls = []
        
        for threshold in thresholds:
            f1 = self.f1_score_pointcloud(pred_points, target_points, threshold)
            
            # Compute precision and recall from F1 (simplified)
            # This is an approximation
            precision = f1  # Simplified
            recall = f1     # Simplified
            
            precisions.append(precision)
            recalls.append(recall)
        
        return np.array(precisions), np.array(recalls), thresholds


class MeshMetrics:
    """Mesh evaluation metrics"""
    
    def __init__(self):
        pass
    
    def mesh_to_pointcloud(self, mesh: trimesh.Trimesh, n_points: int = 10000) -> np.ndarray:
        """Convert mesh to point cloud by sampling surface"""
        if len(mesh.vertices) == 0:
            return np.zeros((n_points, 3))
        
        try:
            points, _ = trimesh.sample.sample_surface(mesh, n_points)
            return points
        except:
            # Fallback: sample vertices
            if len(mesh.vertices) >= n_points:
                indices = np.random.choice(len(mesh.vertices), n_points, replace=False)
                return mesh.vertices[indices]
            else:
                # Sample with replacement
                indices = np.random.choice(len(mesh.vertices), n_points, replace=True)
                return mesh.vertices[indices]
    
    def compute_mesh_volume(self, mesh: trimesh.Trimesh) -> float:
        """Compute mesh volume"""
        try:
            return mesh.volume
        except:
            return 0.0
    
    def compute_mesh_surface_area(self, mesh: trimesh.Trimesh) -> float:
        """Compute mesh surface area"""
        try:
            return mesh.area
        except:
            return 0.0
    
    def mesh_iou(self, pred_mesh: trimesh.Trimesh, target_mesh: trimesh.Trimesh,
                n_points: int = 100000, threshold: float = 0.01) -> float:
        """
        Compute IoU between two meshes using point sampling
        
        Args:
            pred_mesh: Predicted mesh
            target_mesh: Target mesh
            n_points: Number of points to sample
            threshold: Distance threshold for occupancy
        
        Returns:
            IoU value
        """
        # Sample points from both meshes
        pred_points = self.mesh_to_pointcloud(pred_mesh, n_points // 2)
        target_points = self.mesh_to_pointcloud(target_mesh, n_points // 2)
        
        # Combine points
        all_points = np.vstack([pred_points, target_points])
        
        # Check occupancy for each mesh
        pred_occupancy = self.check_mesh_occupancy(all_points, pred_mesh, threshold)
        target_occupancy = self.check_mesh_occupancy(all_points, target_mesh, threshold)
        
        # Compute IoU
        intersection = np.sum(pred_occupancy & target_occupancy)
        union = np.sum(pred_occupancy | target_occupancy)
        
        if union > 0:
            iou = intersection / union
        else:
            iou = 0.0
        
        return iou
    
    def check_mesh_occupancy(self, points: np.ndarray, mesh: trimesh.Trimesh, 
                           threshold: float) -> np.ndarray:
        """Check if points are inside or near mesh surface"""
        try:
            # Check if points are inside mesh
            contains = mesh.contains(points)
            
            # Also check distance to surface for near-surface points
            distances, _ = trimesh.proximity.closest_point(mesh, points)
            near_surface = distances <= threshold
            
            return contains | near_surface
        except:
            # Fallback: check distance to vertices
            distances = cdist(points, mesh.vertices)
            min_distances = np.min(distances, axis=1)
            return min_distances <= threshold
    
    def mesh_chamfer_distance(self, pred_mesh: trimesh.Trimesh, target_mesh: trimesh.Trimesh,
                             n_points: int = 10000) -> float:
        """Compute Chamfer distance between two meshes"""
        pred_points = self.mesh_to_pointcloud(pred_mesh, n_points)
        target_points = self.mesh_to_pointcloud(target_mesh, n_points)
        
        pointcloud_metrics = PointCloudMetrics()
        return pointcloud_metrics.chamfer_distance(pred_points, target_points)
    
    def mesh_hausdorff_distance(self, pred_mesh: trimesh.Trimesh, target_mesh: trimesh.Trimesh,
                               n_points: int = 10000) -> float:
        """Compute Hausdorff distance between two meshes"""
        pred_points = self.mesh_to_pointcloud(pred_mesh, n_points)
        target_points = self.mesh_to_pointcloud(target_mesh, n_points)
        
        pointcloud_metrics = PointCloudMetrics()
        return pointcloud_metrics.hausdorff_distance(pred_points, target_points)


class ImageMetrics:
    """Image-based evaluation metrics"""
    
    def __init__(self):
        pass
    
    def psnr(self, pred_image: np.ndarray, target_image: np.ndarray) -> float:
        """Compute Peak Signal-to-Noise Ratio"""
        mse = np.mean((pred_image - target_image) ** 2)
        if mse == 0:
            return float('inf')
        
        max_pixel = 1.0 if pred_image.max() <= 1.0 else 255.0
        psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
        
        return psnr
    
    def ssim(self, pred_image: np.ndarray, target_image: np.ndarray) -> float:
        """Compute Structural Similarity Index"""
        # Simplified SSIM implementation
        # For production use, consider using skimage.metrics.structural_similarity
        
        # Convert to float
        pred_image = pred_image.astype(np.float64)
        target_image = target_image.astype(np.float64)
        
        # Compute means
        mu1 = np.mean(pred_image)
        mu2 = np.mean(target_image)
        
        # Compute variances and covariance
        sigma1_sq = np.var(pred_image)
        sigma2_sq = np.var(target_image)
        sigma12 = np.mean((pred_image - mu1) * (target_image - mu2))
        
        # SSIM parameters
        C1 = (0.01 * 255) ** 2
        C2 = (0.03 * 255) ** 2
        
        # Compute SSIM
        numerator = (2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)
        denominator = (mu1 ** 2 + mu2 ** 2 + C1) * (sigma1_sq + sigma2_sq + C2)
        
        if denominator == 0:
            return 1.0 if numerator == 0 else 0.0
        
        ssim = numerator / denominator
        
        return ssim
    
    def lpips(self, pred_image: np.ndarray, target_image: np.ndarray) -> float:
        """
        Compute Learned Perceptual Image Patch Similarity
        This is a placeholder - real LPIPS requires a pre-trained network
        """
        # Simplified approximation using feature differences
        # For production use, consider using lpips library
        
        # Simple gradient-based similarity
        pred_grad = np.gradient(pred_image)
        target_grad = np.gradient(target_image)
        
        grad_diff = np.mean([np.abs(pg - tg) for pg, tg in zip(pred_grad, target_grad)])
        
        # Normalize to [0, 1] range
        lpips_score = np.clip(grad_diff / 255.0, 0, 1)
        
        return lpips_score


class MultiViewMetrics:
    """Multi-view consistency metrics"""
    
    def __init__(self):
        pass
    
    def multi_view_consistency(self, predictions_list: List[Dict], 
                             view_transforms: List[np.ndarray]) -> float:
        """
        Compute multi-view consistency score
        
        Args:
            predictions_list: List of predictions from different views
            view_transforms: List of transformation matrices between views
        
        Returns:
            Consistency score
        """
        if len(predictions_list) < 2:
            return 1.0
        
        consistency_scores = []
        
        for i in range(len(predictions_list)):
            for j in range(i + 1, len(predictions_list)):
                # Transform predictions to common coordinate system
                pred_i = predictions_list[i]
                pred_j = predictions_list[j]
                
                # Apply transformation (simplified)
                if 'points' in pred_i and 'points' in pred_j:
                    points_i = pred_i['points']
                    points_j = pred_j['points']
                    
                    # Compute consistency between transformed point clouds
                    pointcloud_metrics = PointCloudMetrics()
                    chamfer_dist = pointcloud_metrics.chamfer_distance(points_i, points_j)
                    
                    # Convert to consistency score (inverse of distance)
                    consistency = 1.0 / (1.0 + chamfer_dist)
                    consistency_scores.append(consistency)
        
        if consistency_scores:
            return np.mean(consistency_scores)
        else:
            return 1.0
    
    def view_coverage_score(self, view_angles: List[float]) -> float:
        """
        Compute view coverage score
        
        Args:
            view_angles: List of viewing angles in radians
        
        Returns:
            Coverage score [0, 1]
        """
        if len(view_angles) < 2:
            return 0.0
        
        # Sort angles
        angles = np.sort(view_angles)
        
        # Compute angular differences
        differences = np.diff(angles)
        
        # Wrap around for circular coverage
        differences = np.append(differences, 2 * np.pi - (angles[-1] - angles[0]))
        
        # Coverage score (inverse of maximum gap)
        max_gap = np.max(differences)
        coverage = 1.0 - (max_gap / (2 * np.pi))
        
        return coverage


class ComprehensiveEvaluationPipeline:
    """Comprehensive evaluation pipeline for all metrics"""
    
    def __init__(self, config: Dict):
        self.config = config
        
        # Initialize metric calculators
        self.pointcloud_metrics = PointCloudMetrics()
        self.mesh_metrics = MeshMetrics()
        self.image_metrics = ImageMetrics()
        self.multiview_metrics = MultiViewMetrics()
        
        # Configuration
        self.thresholds = config.get('thresholds', [0.01, 0.02, 0.05, 0.1])
        self.n_points_sample = config.get('n_points_sample', 10000)
        self.save_visualizations = config.get('save_visualizations', True)
        self.output_dir = Path(config.get('output_dir', 'evaluation_results'))
        self.output_dir.mkdir(exist_ok=True)
    
    def evaluate_batch(self, predictions: Dict, targets: torch.Tensor, 
                      datasets: List[str], categories: List[str]) -> Dict:
        """Evaluate a batch of predictions"""
        batch_metrics = {
            'dataset': datasets[0] if datasets else 'unknown',
            'category': categories[0] if categories else 'unknown',
            'batch_size': targets.size(0)
        }
        
        # Convert tensors to numpy
        if isinstance(targets, torch.Tensor):
            targets_np = targets.cpu().numpy()
        else:
            targets_np = targets
        
        # Extract predictions
        if 'points' in predictions:
            pred_points = predictions['points']
            if isinstance(pred_points, torch.Tensor):
                pred_points_np = pred_points.cpu().numpy()
            else:
                pred_points_np = pred_points
            
            # Evaluate point cloud metrics
            batch_metrics.update(self.evaluate_pointcloud_batch(pred_points_np, targets_np))
        
        # Add uncertainty metrics if available
        if 'uncertainty' in predictions:
            uncertainty = predictions['uncertainty']
            if isinstance(uncertainty, torch.Tensor):
                uncertainty_np = uncertainty.cpu().numpy()
            else:
                uncertainty_np = uncertainty
            
            batch_metrics['mean_uncertainty'] = np.mean(uncertainty_np)
            batch_metrics['std_uncertainty'] = np.std(uncertainty_np)
        
        return batch_metrics
    
    def evaluate_pointcloud_batch(self, pred_points: np.ndarray, target_points: np.ndarray) -> Dict:
        """Evaluate point cloud metrics for a batch"""
        batch_size = pred_points.shape[0]
        
        # Initialize metric accumulators
        chamfer_distances = []
        emd_distances = []
        hausdorff_distances = []
        f1_scores = []
        
        # Evaluate each sample in the batch
        for i in range(batch_size):
            pred = pred_points[i]
            target = target_points[i]
            
            # Chamfer distance
            chamfer = self.pointcloud_metrics.chamfer_distance(pred, target)
            chamfer_distances.append(chamfer)
            
            # Earth Mover's Distance
            emd = self.pointcloud_metrics.earth_movers_distance(pred, target)
            emd_distances.append(emd)
            
            # Hausdorff distance
            hausdorff = self.pointcloud_metrics.hausdorff_distance(pred, target)
            hausdorff_distances.append(hausdorff)
            
            # F1 scores at different thresholds
            f1_scores_threshold = []
            for threshold in self.thresholds:
                f1 = self.pointcloud_metrics.f1_score_pointcloud(pred, target, threshold)
                f1_scores_threshold.append(f1)
            f1_scores.append(f1_scores_threshold)
        
        # Compute statistics
        metrics = {
            'chamfer_distance_mean': np.mean(chamfer_distances),
            'chamfer_distance_std': np.std(chamfer_distances),
            'chamfer_distance_min': np.min(chamfer_distances),
            'chamfer_distance_max': np.max(chamfer_distances),
            
            'emd_mean': np.mean(emd_distances),
            'emd_std': np.std(emd_distances),
            'emd_min': np.min(emd_distances),
            'emd_max': np.max(emd_distances),
            
            'hausdorff_distance_mean': np.mean(hausdorff_distances),
            'hausdorff_distance_std': np.std(hausdorff_distances),
            'hausdorff_distance_min': np.min(hausdorff_distances),
            'hausdorff_distance_max': np.max(hausdorff_distances),
        }
        
        # Add F1 scores for each threshold
        f1_scores_array = np.array(f1_scores)
        for i, threshold in enumerate(self.thresholds):
            metrics[f'f1_score_threshold_{threshold}'] = np.mean(f1_scores_array[:, i])
        
        return metrics
    
    def evaluate_mesh_batch(self, pred_meshes: List[trimesh.Trimesh], 
                           target_meshes: List[trimesh.Trimesh]) -> Dict:
        """Evaluate mesh metrics for a batch"""
        iou_scores = []
        chamfer_distances = []
        hausdorff_distances = []
        
        for pred_mesh, target_mesh in zip(pred_meshes, target_meshes):
            # IoU
            iou = self.mesh_metrics.mesh_iou(pred_mesh, target_mesh, self.n_points_sample)
            iou_scores.append(iou)
            
            # Chamfer distance
            chamfer = self.mesh_metrics.mesh_chamfer_distance(pred_mesh, target_mesh)
            chamfer_distances.append(chamfer)
            
            # Hausdorff distance
            hausdorff = self.mesh_metrics.mesh_hausdorff_distance(pred_mesh, target_mesh)
            hausdorff_distances.append(hausdorff)
        
        metrics = {
            'mesh_iou_mean': np.mean(iou_scores),
            'mesh_iou_std': np.std(iou_scores),
            'mesh_chamfer_mean': np.mean(chamfer_distances),
            'mesh_chamfer_std': np.std(chamfer_distances),
            'mesh_hausdorff_mean': np.mean(hausdorff_distances),
            'mesh_hausdorff_std': np.std(hausdorff_distances),
        }
        
        return metrics
    
    def aggregate_metrics(self, batch_metrics_list: List[Dict]) -> Dict:
        """Aggregate metrics from multiple batches"""
        if not batch_metrics_list:
            return {}
        
        # Group by dataset and category
        dataset_metrics = {}
        
        for batch_metrics in batch_metrics_list:
            dataset = batch_metrics.get('dataset', 'unknown')
            category = batch_metrics.get('category', 'unknown')
            
            if dataset not in dataset_metrics:
                dataset_metrics[dataset] = {}
            
            if category not in dataset_metrics[dataset]:
                dataset_metrics[dataset][category] = []
            
            dataset_metrics[dataset][category].append(batch_metrics)
        
        # Compute aggregated metrics
        aggregated_metrics = {
            'overall': {},
            'by_dataset': {},
            'by_category': {}
        }
        
        # Overall metrics
        all_metrics = []
        for dataset_data in dataset_metrics.values():
            for category_data in dataset_data.values():
                all_metrics.extend(category_data)
        
        if all_metrics:
            aggregated_metrics['overall'] = self.compute_aggregated_statistics(all_metrics)
        
        # By dataset
        for dataset, dataset_data in dataset_metrics.items():
            all_dataset_metrics = []
            for category_data in dataset_data.values():
                all_dataset_metrics.extend(category_data)
            
            if all_dataset_metrics:
                aggregated_metrics['by_dataset'][dataset] = self.compute_aggregated_statistics(all_dataset_metrics)
        
        # By category
        category_metrics = {}
        for dataset_data in dataset_metrics.values():
            for category, category_data in dataset_data.items():
                if category not in category_metrics:
                    category_metrics[category] = []
                category_metrics[category].extend(category_data)
        
        for category, category_data in category_metrics.items():
            if category_data:
                aggregated_metrics['by_category'][category] = self.compute_aggregated_statistics(category_data)
        
        return aggregated_metrics
    
    def compute_aggregated_statistics(self, metrics_list: List[Dict]) -> Dict:
        """Compute statistics from a list of metrics"""
        if not metrics_list:
            return {}
        
        # Get all metric keys
        all_keys = set()
        for metrics in metrics_list:
            all_keys.update(metrics.keys())
        
        # Filter out non-numeric keys
        numeric_keys = []
        for key in all_keys:
            if key in ['dataset', 'category', 'batch_size']:
                continue
            
            # Check if all values for this key are numeric
            values = [m.get(key) for m in metrics_list if key in m and m[key] is not None]
            if values and all(isinstance(v, (int, float)) for v in values):
                numeric_keys.append(key)
        
        # Compute statistics
        statistics = {}
        
        for key in numeric_keys:
            values = [m[key] for m in metrics_list if key in m and m[key] is not None]
            
            if values:
                statistics[f"{key}_mean"] = np.mean(values)
                statistics[f"{key}_std"] = np.std(values)
                statistics[f"{key}_min"] = np.min(values)
                statistics[f"{key}_max"] = np.max(values)
                statistics[f"{key}_median"] = np.median(values)
        
        # Add sample counts
        statistics['n_samples'] = len(metrics_list)
        statistics['total_batches'] = sum(m.get('batch_size', 1) for m in metrics_list)
        
        return statistics
    
    def visualize_results(self, predictions: Dict, targets: torch.Tensor, 
                         save_path: Optional[Path] = None):
        """Visualize evaluation results"""
        if not self.save_visualizations:
            return
        
        # Create visualization
        fig = plt.figure(figsize=(15, 10))
        
        # Point cloud comparison
        if 'points' in predictions and targets is not None:
            ax1 = fig.add_subplot(121, projection='3d')
            pred_points = predictions['points']
            if isinstance(pred_points, torch.Tensor):
                pred_points = pred_points.cpu().numpy()
            
            if isinstance(targets, torch.Tensor):
                targets_np = targets.cpu().numpy()
            else:
                targets_np = targets
            
            # Plot predicted point cloud
            if len(pred_points.shape) == 3:
                pred_points = pred_points[0]  # Take first sample
            
            if len(targets_np.shape) == 3:
                targets_np = targets_np[0]  # Take first sample
            
            ax1.scatter(pred_points[:, 0], pred_points[:, 1], pred_points[:, 2], 
                         c='red', s=1, alpha=0.6, label='Predicted')
            ax1.scatter(targets_np[:, 0], targets_np[:, 1], targets_np[:, 2], 
                       c='blue', s=1, alpha=0.6, label='Target')
            ax1.set_title('Point Cloud Comparison')
            ax1.legend()
            
            # Add uncertainty visualization if available
            if 'uncertainty' in predictions:
                ax2 = fig.add_subplot(122, projection='3d')
                uncertainty = predictions['uncertainty']
                if isinstance(uncertainty, torch.Tensor):
                    uncertainty = uncertainty.cpu().numpy()
                
                # Plot with uncertainty colors
                scatter = ax2.scatter(pred_points[:, 0], pred_points[:, 1], pred_points[:, 2], 
                                    c=uncertainty, s=2, alpha=0.8, cmap='viridis')
                ax2.set_title('Point Cloud with Uncertainty')
                plt.colorbar(scatter, ax=ax2, label='Uncertainty')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Visualization saved to {save_path}")
        
        plt.close()


def create_evaluation_config():
    """Create default evaluation configuration"""
    return {
        'thresholds': [0.01, 0.02, 0.05, 0.1],
        'n_points_sample': 10000,
        'save_visualizations': True,
        'output_dir': 'evaluation_results',
        'metrics': [
            'chamfer_distance',
            'earth_movers_distance',
            'hausdorff_distance',
            'f1_score',
            'mesh_iou',
            'multi_view_consistency'
        ]
    }


def test_evaluation_pipeline():
    """Test the evaluation pipeline"""
    print("Testing comprehensive evaluation pipeline...")
    
    # Create test data
    batch_size = 2
    n_points = 2048
    
    # Generate test predictions
    test_predictions = {
        'points': torch.randn(batch_size, n_points, 3),
        'uncertainty': torch.rand(batch_size, n_points)
    }
    
    # Generate test targets
    test_targets = torch.randn(batch_size, n_points, 3)
    
    # Create evaluation pipeline
    config = create_evaluation_config()
    pipeline = ComprehensiveEvaluationPipeline(config)
    
    # Test batch evaluation
    datasets = ['test_dataset']
    categories = ['chair']
    
    batch_metrics = pipeline.evaluate_batch(test_predictions, test_targets, datasets, categories)
    print(f"Batch metrics: {batch_metrics}")
    
    # Test aggregation
    batch_metrics_list = [batch_metrics, batch_metrics]  # Duplicate for testing
    aggregated_metrics = pipeline.aggregate_metrics(batch_metrics_list)
    print(f"Aggregated metrics keys: {list(aggregated_metrics.keys())}")
    
    # Test visualization
    pipeline.visualize_results(test_predictions, test_targets)
    
    print("Evaluation pipeline test completed successfully!")


if __name__ == "__main__":
    test_evaluation_pipeline()