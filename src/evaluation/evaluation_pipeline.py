"""
Evaluation Metrics and Deployment Pipeline for Enhanced 3D Reconstruction
Supports comprehensive 3D metrics, model serving, and edge deployment
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
import time
import json
from pathlib import Path
import trimesh
from sklearn.metrics import precision_recall_curve, average_precision_score
import open3d as o3d

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MeshMetrics:
    """Comprehensive mesh evaluation metrics."""
    
    @staticmethod
    def compute_chamfer_distance(pred_points: np.ndarray, gt_points: np.ndarray,
                                num_samples: int = 10000) -> float:
        """Compute Chamfer distance between point clouds."""
        # Sample points if too many
        if len(pred_points) > num_samples:
            indices = np.random.choice(len(pred_points), num_samples, replace=False)
            pred_points = pred_points[indices]
        
        if len(gt_points) > num_samples:
            indices = np.random.choice(len(gt_points), num_samples, replace=False)
            gt_points = gt_points[indices]
        
        # Compute distances
        from sklearn.metrics import pairwise_distances
        distances_pred_to_gt = pairwise_distances(pred_points, gt_points).min(axis=1)
        distances_gt_to_pred = pairwise_distances(gt_points, pred_points).min(axis=1)
        
        chamfer_dist = np.mean(distances_pred_to_gt) + np.mean(distances_gt_to_pred)
        return chamfer_dist
    
    @staticmethod
    def compute_hausdorff_distance(pred_points: np.ndarray, gt_points: np.ndarray) -> float:
        """Compute Hausdorff distance between point clouds."""
        from scipy.spatial.distance import directed_hausdorff
        
        hausdorff_1, _ = directed_hausdorff(pred_points, gt_points)
        hausdorff_2, _ = directed_hausdorff(gt_points, pred_points)
        
        return max(hausdorff_1, hausdorff_2)
    
    @staticmethod
    def compute_normal_consistency(pred_normals: np.ndarray, gt_normals: np.ndarray,
                                 pred_points: np.ndarray, gt_points: np.ndarray) -> float:
        """Compute normal consistency between predicted and ground truth."""
        # Find nearest neighbors
        from sklearn.neighbors import NearestNeighbors
        
        nbrs = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(gt_points)
        distances, indices = nbrs.kneighbors(pred_points)
        
        # Compute normal consistency
        gt_normals_matched = gt_normals[indices.squeeze()]
        normal_consistency = np.mean(np.abs(np.sum(pred_normals * gt_normals_matched, axis=1)))
        
        return normal_consistency
    
    @staticmethod
    def compute_mesh_accuracy(pred_mesh: trimesh.Trimesh, gt_mesh: trimesh.Trimesh,
                            threshold: float = 0.01) -> Dict[str, float]:
        """Compute mesh accuracy metrics."""
        # Sample points from meshes
        pred_points = pred_mesh.sample(10000)
        gt_points = gt_mesh.sample(10000)
        
        # Compute metrics
        chamfer_dist = MeshMetrics.compute_chamfer_distance(pred_points, gt_points)
        hausdorff_dist = MeshMetrics.compute_hausdorff_distance(pred_points, gt_points)
        
        # Accuracy and completeness
        distances_pred_to_gt = trimesh.proximity.signed_distance(gt_mesh, pred_points)
        distances_gt_to_pred = trimesh.proximity.signed_distance(pred_mesh, gt_points)
        
        accuracy = np.mean(np.abs(distances_pred_to_gt) < threshold)
        completeness = np.mean(np.abs(distances_gt_to_pred) < threshold)
        
        return {
            'chamfer_distance': chamfer_dist,
            'hausdorff_distance': hausdorff_dist,
            'accuracy': accuracy,
            'completeness': completeness
        }

class VoxelMetrics:
    """Voxel-based evaluation metrics."""
    
    @staticmethod
    def compute_voxel_iou(pred_voxels: np.ndarray, gt_voxels: np.ndarray) -> float:
        """Compute voxel IoU."""
        intersection = np.logical_and(pred_voxels, gt_voxels).sum()
        union = np.logical_or(pred_voxels, gt_voxels).sum()
        
        if union == 0:
            return 1.0 if intersection == 0 else 0.0
        
        return intersection / union
    
    @staticmethod
    def compute_voxel_precision_recall(pred_voxels: np.ndarray, gt_voxels: np.ndarray) -> Tuple[float, float]:
        """Compute voxel precision and recall."""
        true_positives = np.logical_and(pred_voxels, gt_voxels).sum()
        predicted_positives = pred_voxels.sum()
        actual_positives = gt_voxels.sum()
        
        precision = true_positives / (predicted_positives + 1e-8)
        recall = true_positives / (actual_positives + 1e-8)
        
        return precision, recall
    
    @staticmethod
    def compute_voxel_f1_score(pred_voxels: np.ndarray, gt_voxels: np.ndarray) -> float:
        """Compute voxel F1 score."""
        precision, recall = VoxelMetrics.compute_voxel_precision_recall(pred_voxels, gt_voxels)
        
        if precision + recall == 0:
            return 0.0
        
        return 2 * (precision * recall) / (precision + recall)

class SDFMetrics:
    """SDF-based evaluation metrics."""
    
    @staticmethod
    def compute_sdf_accuracy(pred_sdf: np.ndarray, gt_sdf: np.ndarray, 
                           threshold: float = 0.01) -> float:
        """Compute SDF accuracy within threshold."""
        abs_diff = np.abs(pred_sdf - gt_sdf)
        return np.mean(abs_diff < threshold)
    
    @staticmethod
    def compute_sdf_l1_error(pred_sdf: np.ndarray, gt_sdf: np.ndarray) -> float:
        """Compute SDF L1 error."""
        return np.mean(np.abs(pred_sdf - gt_sdf))
    
    @staticmethod
    def compute_sdf_l2_error(pred_sdf: np.ndarray, gt_sdf: np.ndarray) -> float:
        """Compute SDF L2 error."""
        return np.sqrt(np.mean((pred_sdf - gt_sdf) ** 2))

class NeRFMetrics:
    """NeRF-specific evaluation metrics."""
    
    @staticmethod
    def compute_psnr(pred_image: np.ndarray, gt_image: np.ndarray) -> float:
        """Compute PSNR between rendered images."""
        mse = np.mean((pred_image - gt_image) ** 2)
        if mse == 0:
            return float('inf')
        
        return 20 * np.log10(1.0) - 10 * np.log10(mse)
    
    @staticmethod
    def compute_ssim(pred_image: np.ndarray, gt_image: np.ndarray) -> float:
        """Compute SSIM between rendered images."""
        from skimage.metrics import structural_similarity
        
        # Convert to grayscale if needed
        if len(pred_image.shape) == 3:
            pred_image = np.mean(pred_image, axis=2)
            gt_image = np.mean(gt_image, axis=2)
        
        return structural_similarity(pred_image, gt_image, data_range=1.0)
    
    @staticmethod
    def compute_lpips(pred_image: np.ndarray, gt_image: np.ndarray) -> float:
        """Compute LPIPS perceptual loss."""
        try:
            import lpips
            loss_fn = lpips.LPIPS(net='alex')
            
            # Convert to torch tensors
            pred_tensor = torch.from_numpy(pred_image).permute(2, 0, 1).unsqueeze(0).float()
            gt_tensor = torch.from_numpy(gt_image).permute(2, 0, 1).unsqueeze(0).float()
            
            # Normalize to [-1, 1]
            pred_tensor = pred_tensor * 2 - 1
            gt_tensor = gt_tensor * 2 - 1
            
            lpips_value = loss_fn(pred_tensor, gt_tensor).item()
            return lpips_value
        except ImportError:
            logger.warning("LPIPS not available. Install lpips package.")
            return 0.0

class ComprehensiveEvaluator:
    """Comprehensive evaluation pipeline for 3D reconstruction."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.representation = config.get('representation', 'sdf')  # sdf, occupancy, nerf
        
    def evaluate_mesh(self, pred_mesh_path: str, gt_mesh_path: str) -> Dict[str, float]:
        """Evaluate mesh reconstruction."""
        try:
            pred_mesh = trimesh.load(pred_mesh_path)
            gt_mesh = trimesh.load(gt_mesh_path)
            
            # Compute mesh metrics
            mesh_metrics = MeshMetrics.compute_mesh_accuracy(pred_mesh, gt_mesh)
            
            # Add mesh-specific metrics
            mesh_metrics['pred_vertices'] = len(pred_mesh.vertices)
            mesh_metrics['pred_faces'] = len(pred_mesh.faces)
            mesh_metrics['gt_vertices'] = len(gt_mesh.vertices)
            mesh_metrics['gt_faces'] = len(gt_mesh.faces)
            
            return mesh_metrics
            
        except Exception as e:
            logger.error(f"Mesh evaluation failed: {e}")
            return {}
    
    def evaluate_voxels(self, pred_voxels: np.ndarray, gt_voxels: np.ndarray) -> Dict[str, float]:
        """Evaluate voxel reconstruction."""
        metrics = {}
        
        # Basic voxel metrics
        metrics['iou'] = VoxelMetrics.compute_voxel_iou(pred_voxels, gt_voxels)
        precision, recall = VoxelMetrics.compute_voxel_precision_recall(pred_voxels, gt_voxels)
        metrics['precision'] = precision
        metrics['recall'] = recall
        metrics['f1_score'] = VoxelMetrics.compute_voxel_f1_score(pred_voxels, gt_voxels)
        
        # Additional metrics
        metrics['pred_occupancy'] = np.mean(pred_voxels)
        metrics['gt_occupancy'] = np.mean(gt_voxels)
        metrics['voxel_count_diff'] = abs(np.sum(pred_voxels) - np.sum(gt_voxels))
        
        return metrics
    
    def evaluate_sdf(self, pred_sdf: np.ndarray, gt_sdf: np.ndarray) -> Dict[str, float]:
        """Evaluate SDF reconstruction."""
        metrics = {}
        
        # SDF-specific metrics
        metrics['accuracy_1mm'] = SDFMetrics.compute_sdf_accuracy(pred_sdf, gt_sdf, threshold=0.001)
        metrics['accuracy_5mm'] = SDFMetrics.compute_sdf_accuracy(pred_sdf, gt_sdf, threshold=0.005)
        metrics['l1_error'] = SDFMetrics.compute_sdf_l1_error(pred_sdf, gt_sdf)
        metrics['l2_error'] = SDFMetrics.compute_sdf_l2_error(pred_sdf, gt_sdf)
        
        # Surface metrics (near zero level set)
        surface_mask = np.abs(gt_sdf) < 0.01
        if surface_mask.sum() > 0:
            metrics['surface_l1_error'] = np.mean(np.abs(pred_sdf[surface_mask] - gt_sdf[surface_mask]))
        
        return metrics
    
    def evaluate_nerf(self, pred_images: List[np.ndarray], gt_images: List[np.ndarray]) -> Dict[str, float]:
        """Evaluate NeRF reconstruction."""
        metrics = {}
        
        psnr_values = []
        ssim_values = []
        lpips_values = []
        
        for pred_img, gt_img in zip(pred_images, gt_images):
            psnr_values.append(NeRFMetrics.compute_psnr(pred_img, gt_img))
            ssim_values.append(NeRFMetrics.compute_ssim(pred_img, gt_img))
            lpips_values.append(NeRFMetrics.compute_lpips(pred_img, gt_img))
        
        metrics['psnr'] = np.mean(psnr_values)
        metrics['ssim'] = np.mean(ssim_values)
        metrics['lpips'] = np.mean(lpips_values)
        
        metrics['psnr_std'] = np.std(psnr_values)
        metrics['ssim_std'] = np.std(ssim_values)
        metrics['lpips_std'] = np.std(lpips_values)
        
        return metrics
    
    def evaluate_uncertainty(self, predictions: np.ndarray, uncertainties: np.ndarray,
                           ground_truth: np.ndarray) -> Dict[str, float]:
        """Evaluate uncertainty estimation quality."""
        metrics = {}
        
        # Compute prediction errors
        errors = np.abs(predictions - ground_truth)
        
        # Correlation between uncertainty and error
        correlation = np.corrcoef(uncertainties.flatten(), errors.flatten())[0, 1]
        metrics['uncertainty_error_correlation'] = correlation
        
        # Calibration metrics
        sorted_indices = np.argsort(uncertainties.flatten())
        n_bins = 10
        bin_size = len(sorted_indices) // n_bins
        
        calibration_errors = []
        for i in range(n_bins):
            start_idx = i * bin_size
            end_idx = (i + 1) * bin_size if i < n_bins - 1 else len(sorted_indices)
            
            bin_indices = sorted_indices[start_idx:end_idx]
            bin_errors = errors.flatten()[bin_indices]
            bin_uncertainties = uncertainties.flatten()[bin_indices]
            
            # Expected vs actual error in bin
            expected_error = np.mean(bin_uncertainties)
            actual_error = np.mean(bin_errors)
            
            calibration_errors.append(abs(expected_error - actual_error))
        
        metrics['calibration_error'] = np.mean(calibration_errors)
        metrics['mean_uncertainty'] = np.mean(uncertainties)
        metrics['uncertainty_std'] = np.std(uncertainties)
        
        return metrics
    
    def comprehensive_evaluation(self, predictions: Dict[str, np.ndarray],
                               ground_truths: Dict[str, np.ndarray],
                               additional_data: Optional[Dict] = None) -> Dict[str, Any]:
        """Run comprehensive evaluation across all metrics."""
        results = {}
        
        # Representation-specific evaluation
        if self.representation == 'sdf' and 'sdf_pred' in predictions and 'sdf_gt' in ground_truths:
            results['sdf_metrics'] = self.evaluate_sdf(predictions['sdf_pred'], ground_truths['sdf_gt'])
            
            # Uncertainty evaluation if available
            if 'uncertainty' in predictions:
                results['uncertainty_metrics'] = self.evaluate_uncertainty(
                    predictions['sdf_pred'], predictions['uncertainty'], ground_truths['sdf_gt']
                )
        
        elif self.representation == 'occupancy' and 'occupancy_pred' in predictions and 'occupancy_gt' in ground_truths:
            results['voxel_metrics'] = self.evaluate_voxels(predictions['occupancy_pred'], ground_truths['occupancy_gt'])
        
        elif self.representation == 'nerf' and 'rendered_images' in predictions and 'gt_images' in ground_truths:
            results['nerf_metrics'] = self.evaluate_nerf(predictions['rendered_images'], ground_truths['gt_images'])
        
        # Mesh evaluation if mesh data is available
        if additional_data and 'pred_mesh_path' in additional_data and 'gt_mesh_path' in additional_data:
            results['mesh_metrics'] = self.evaluate_mesh(
                additional_data['pred_mesh_path'], additional_data['gt_mesh_path']
            )
        
        # Compute overall score
        results['overall_score'] = self._compute_overall_score(results)
        
        return results
    
    def _compute_overall_score(self, metrics: Dict[str, Any]) -> float:
        """Compute overall evaluation score."""
        scores = []
        
        # SDF metrics
        if 'sdf_metrics' in metrics:
            sdf_score = (
                metrics['sdf_metrics'].get('accuracy_5mm', 0) * 0.4 +
                (1 - min(metrics['sdf_metrics'].get('l1_error', 1), 1)) * 0.3 +
                (1 - min(metrics['sdf_metrics'].get('l2_error', 1), 1)) * 0.3
            )
            scores.append(sdf_score)
        
        # Voxel metrics
        if 'voxel_metrics' in metrics:
            voxel_score = (
                metrics['voxel_metrics'].get('iou', 0) * 0.5 +
                metrics['voxel_metrics'].get('f1_score', 0) * 0.5
            )
            scores.append(voxel_score)
        
        # NeRF metrics
        if 'nerf_metrics' in metrics:
            nerf_score = (
                metrics['nerf_metrics'].get('psnr', 0) / 30 * 0.4 +  # Normalize PSNR
                metrics['nerf_metrics'].get('ssim', 0) * 0.3 +
                (1 - metrics['nerf_metrics'].get('lpips', 1)) * 0.3
            )
            scores.append(nerf_score)
        
        # Mesh metrics
        if 'mesh_metrics' in metrics:
            mesh_score = (
                metrics['mesh_metrics'].get('accuracy', 0) * 0.4 +
                metrics['mesh_metrics'].get('completeness', 0) * 0.4 +
                (1 - min(metrics['mesh_metrics'].get('chamfer_distance', 1), 1)) * 0.2
            )
            scores.append(mesh_score)
        
        return np.mean(scores) if scores else 0.0
    
    def save_evaluation_report(self, results: Dict[str, Any], output_path: str):
        """Save comprehensive evaluation report."""
        report = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'representation': self.representation,
            'config': self.config,
            'results': results
        }
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=self._convert_numpy_types)
        
        logger.info(f"Evaluation report saved to {output_path}")
    
    def _convert_numpy_types(self, obj):
        """Convert numpy types to Python types for JSON serialization."""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

class ModelServer:
    """Model serving infrastructure for deployment."""
    
    def __init__(self, model_path: str, config: Dict):
        self.model_path = model_path
        self.config = config
        self.model = None
        self.device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
        
    def load_model(self):
        """Load model for serving."""
        try:
            from src.models.enhanced_3d_reconstruction import Enhanced3DReconstructionModel
            
            # Load model configuration
            checkpoint = torch.load(self.model_path, map_location=self.device)
            model_config = checkpoint.get('config', {}).get('model_config', {})
            
            # Create model
            self.model = Enhanced3DReconstructionModel(model_config)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.to(self.device)
            self.model.eval()
            
            logger.info(f"Model loaded successfully from {self.model_path}")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def predict_single(self, image: np.ndarray, resolution: int = 128) -> Dict[str, Any]:
        """Predict 3D reconstruction for single image."""
        if self.model is None:
            self.load_model()
        
        # Preprocess image
        image_tensor = self._preprocess_image(image)
        
        # Create query coordinates
        coordinates = self._create_query_coordinates(resolution)
        
        # Run inference
        with torch.no_grad():
            outputs = self.model(image_tensor, coordinates)
        
        # Post-process results
        results = self._postprocess_outputs(outputs, resolution)
        
        return results
    
    def _preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """Preprocess input image."""
        # Resize to model input size
        from torchvision import transforms
        
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        image_tensor = transform(image).unsqueeze(0).to(self.device)
        return image_tensor
    
    def _create_query_coordinates(self, resolution: int) -> torch.Tensor:
        """Create 3D query coordinates."""
        coords = np.linspace(-1, 1, resolution)
        x, y, z = np.meshgrid(coords, coords, coords, indexing='ij')
        coordinates = np.stack([x.flatten(), y.flatten(), z.flatten()], axis=1)
        
        return torch.from_numpy(coordinates).float().unsqueeze(0).to(self.device)
    
    def _postprocess_outputs(self, outputs: Dict[str, torch.Tensor], resolution: int) -> Dict[str, Any]:
        """Post-process model outputs."""
        results = {}
        
        # Convert to numpy and reshape
        for key, tensor in outputs.items():
            if tensor is not None:
                np_array = tensor.cpu().numpy()
                if len(np_array.shape) == 3 and np_array.shape[1] == resolution**3:
                    # Reshape volumetric data
                    results[key] = np_array.reshape(-1, resolution, resolution, resolution)
                else:
                    results[key] = np_array
        
        return results

class EdgeDeploymentOptimizer:
    """Optimization for edge device deployment."""
    
    def __init__(self, model: nn.Module, config: Dict):
        self.model = model
        self.config = config
        
    def optimize_for_mobile(self) -> nn.Module:
        """Optimize model for mobile deployment."""
        logger.info("Optimizing model for mobile deployment...")
        
        # Quantization
        if self.config.get('mobile_quantization', True):
            self.model = self._apply_mobile_quantization()
        
        # Pruning (if configured)
        if self.config.get('mobile_pruning', False):
            self.model = self._apply_mobile_pruning()
        
        return self.model
    
    def optimize_for_jetson(self) -> nn.Module:
        """Optimize model for NVIDIA Jetson deployment."""
        logger.info("Optimizing model for Jetson deployment...")
        
        # TensorRT optimization
        if self.config.get('jetson_tensorrt', True):
            self.model = self._apply_tensorrt_optimization()
        
        # Mixed precision
        if self.config.get('jetson_mixed_precision', True):
            self.model = self.model.half()
        
        return self.model
    
    def optimize_for_raspberry_pi(self) -> nn.Module:
        """Optimize model for Raspberry Pi deployment."""
        logger.info("Optimizing model for Raspberry Pi deployment...")
        
        # Aggressive quantization
        if self.config.get('rpi_quantization', True):
            self.model = self._apply_aggressive_quantization()
        
        # Model distillation (if teacher model available)
        if self.config.get('rpi_distillation', False):
            self.model = self._apply_model_distillation()
        
        return self.model
    
    def _apply_mobile_quantization(self) -> nn.Module:
        """Apply mobile-optimized quantization."""
        import torch.quantization as quant
        
        # Dynamic quantization for mobile
        model_quantized = quant.quantize_dynamic(
            self.model, 
            {nn.Linear, nn.Conv2d}, 
            dtype=torch.qint8
        )
        
        return model_quantized
    
    def _apply_aggressive_quantization(self) -> nn.Module:
        """Apply aggressive quantization for resource-constrained devices."""
        import torch.quantization as quant
        
        # Static quantization with calibration
        self.model.eval()
        self.model.qconfig = quant.get_default_qconfig('qnnpack')
        model_prepared = quant.prepare(self.model)
        
        # Calibrate with dummy data (in practice, use real calibration data)
        dummy_input = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            _ = model_prepared(dummy_input)
        
        model_quantized = quant.convert(model_prepared)
        return model_quantized
    
    def _apply_tensorrt_optimization(self) -> nn.Module:
        """Apply TensorRT optimization."""
        # This would require TensorRT conversion
        # Placeholder for actual TensorRT implementation
        logger.info("TensorRT optimization would be applied here")
        return self.model
    
    def _apply_mobile_pruning(self) -> nn.Module:
        """Apply structured pruning for mobile."""
        # Placeholder for pruning implementation
        logger.info("Mobile pruning would be applied here")
        return self.model
    
    def _apply_model_distillation(self) -> nn.Module:
        """Apply knowledge distillation."""
        # Placeholder for distillation implementation
        logger.info("Model distillation would be applied here")
        return self.model
    
    def benchmark_edge_performance(self, test_input: torch.Tensor, device: str = 'cpu') -> Dict[str, float]:
        """Benchmark model performance on edge device."""
        self.model.to(device)
        self.model.eval()
        
        # Warm up
        with torch.no_grad():
            for _ in range(5):
                _ = self.model(test_input.to(device))
        
        # Benchmark inference time
        start_time = time.time()
        with torch.no_grad():
            for _ in range(100):
                _ = self.model(test_input.to(device))
        
        avg_time = (time.time() - start_time) / 100
        
        # Memory usage (approximate)
        model_size = sum(p.numel() for p in self.model.parameters()) * 4  # Assuming float32
        
        return {
            'avg_inference_time_ms': avg_time * 1000,
            'model_size_mb': model_size / (1024 * 1024),
            'device': device,
            'fps': 1.0 / avg_time
        }

def create_evaluation_config() -> Dict:
    """Create default evaluation configuration."""
    return {
        'representation': 'sdf',  # sdf, occupancy, nerf
        'mesh_metrics': True,
        'voxel_metrics': True,
        'sdf_metrics': True,
        'nerf_metrics': True,
        'uncertainty_metrics': True,
        'save_detailed_results': True,
        'output_dir': 'evaluation_results'
    }

def create_deployment_config(target_device: str = 'mobile') -> Dict:
    """Create deployment configuration for target device."""
    configs = {
        'mobile': {
            'mobile_quantization': True,
            'mobile_pruning': False,
            'target_input_size': (224, 224),
            'max_model_size_mb': 50
        },
        'jetson': {
            'jetson_tensorrt': True,
            'jetson_mixed_precision': True,
            'target_input_size': (256, 256),
            'max_model_size_mb': 200
        },
        'raspberry_pi': {
            'rpi_quantization': True,
            'rpi_distillation': False,
            'target_input_size': (192, 192),
            'max_model_size_mb': 20
        }
    }
    
    return configs.get(target_device, configs['mobile'])