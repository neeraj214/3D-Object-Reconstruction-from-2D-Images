import torch
import numpy as np
import argparse
import yaml
import os
import logging
from PIL import Image
import open3d as o3d
from typing import Union, Optional, Tuple, Dict, List
import time
from datetime import datetime

# Import custom modules
from ..models.resnet_2d import ResNet2DFeatureExtractor, create_resnet_2d_extractor
from ..models.pointnet_3d import PointNet3DFeatureExtractor, create_pointnet_3d_extractor
from ..models.depth_to_pointcloud import DepthToPointCloudConverter, create_depth_to_pointcloud_converter
from ..models.fusion_classifier import FusionClassifier, create_fusion_classifier
from ..utils.visualization import PointCloudVisualizer, create_visualizer
from ..utils.pointcloud_utils import PointCloudUtils, create_pointcloud_utils


class Hybrid2D3DPredictor:
    """
    End-to-end predictor for hybrid 2D-to-3D reconstruction and classification.
    """
    
    def __init__(self, config_path: str, model_path: str, device: str = "cuda"):
        """
        Initialize the predictor.
        
        Args:
            config_path: Path to configuration file
            model_path: Path to trained model
            device: Device to run on
        """
        self.config_path = config_path
        self.model_path = model_path
        self.device = device if torch.cuda.is_available() else "cpu"
        
        # Load configuration
        self.config = self.load_config()
        
        # Setup logging
        self.setup_logging()
        
        # Initialize components
        self.initialize_components()
        
        self.logger.info(f"Predictor initialized on device: {self.device}")
    
    def load_config(self) -> dict:
        """Load configuration from YAML file."""
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    
    def setup_logging(self):
        """Setup logging configuration."""
        log_config = self.config.get('logging', {})
        log_level = getattr(logging, log_config.get('level', 'INFO'))
        log_format = log_config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        
        logging.basicConfig(level=log_level, format=log_format)
        self.logger = logging.getLogger(__name__)
    
    def initialize_components(self):
        """Initialize all model components."""
        self.logger.info("Initializing model components...")
        
        # 2D feature extractor
        self.feature_extractor_2d = create_resnet_2d_extractor(self.config)
        self.feature_extractor_2d.to(self.device)
        self.feature_extractor_2d.eval()
        
        # 3D point cloud generator
        self.pointcloud_generator = create_depth_to_pointcloud_converter(self.config)
        
        # 3D feature extractor
        self.feature_extractor_3d = create_pointnet_3d_extractor(self.config)
        self.feature_extractor_3d.to(self.device)
        self.feature_extractor_3d.eval()
        
        # Fusion classifier
        self.fusion_classifier = create_fusion_classifier(self.config)
        self.fusion_classifier.to(self.device)
        
        # Load trained fusion model
        self.load_trained_model()
        self.fusion_classifier.eval()
        
        # Visualizer
        self.visualizer = create_visualizer(self.config)
        
        # Point cloud utilities
        self.pointcloud_utils = create_pointcloud_utils(self.config)
        
        self.logger.info("All components initialized successfully")
    
    def load_trained_model(self):
        """Load the trained fusion model."""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        
        checkpoint = torch.load(self.model_path, map_location=self.device)
        
        # Load model state
        if 'best_model_state_dict' in checkpoint:
            self.fusion_classifier.load_state_dict(checkpoint['best_model_state_dict'])
            self.logger.info("Loaded best model state")
        elif 'model_state_dict' in checkpoint:
            self.fusion_classifier.load_state_dict(checkpoint['model_state_dict'])
            self.logger.info("Loaded model state")
        else:
            self.fusion_classifier.load_state_dict(checkpoint)
            self.logger.info("Loaded model state (direct)")
        
        self.logger.info(f"Model loaded from {self.model_path}")
    
    def process_2d_image(self, image_path: str) -> Tuple[torch.Tensor, Image.Image]:
        """
        Process 2D image and extract features.
        
        Args:
            image_path: Path to input image
            
        Returns:
            Tuple of (2D features, PIL Image)
        """
        self.logger.info(f"Processing 2D image: {image_path}")
        
        # Load image
        image = Image.open(image_path).convert('RGB')
        
        # Extract 2D features
        with torch.no_grad():
            features_2d = self.feature_extractor_2d.extract_features_from_image(image)
            features_2d = features_2d.to(self.device)
        
        self.logger.info(f"Extracted 2D features: {features_2d.shape}")
        
        return features_2d, image
    
    def generate_3d_pointcloud(self, image_path: str) -> o3d.geometry.PointCloud:
        """
        Generate 3D point cloud from 2D image.
        
        Args:
            image_path: Path to input image
            
        Returns:
            Generated point cloud
        """
        self.logger.info(f"Generating 3D point cloud from image: {image_path}")
        
        # Generate point cloud
        pointcloud = self.pointcloud_generator.process_image(
            image_path, 
            num_points=self.config['data']['point_cloud']['num_points']
        )
        
        self.logger.info(f"Generated point cloud with {len(pointcloud.points)} points")
        
        return pointcloud
    
    def extract_3d_features(self, pointcloud: o3d.geometry.PointCloud) -> torch.Tensor:
        """
        Extract 3D features from point cloud.
        
        Args:
            pointcloud: Input point cloud
            
        Returns:
            3D features tensor
        """
        self.logger.info("Extracting 3D features from point cloud")
        
        # Extract 3D features
        with torch.no_grad():
            features_3d = self.feature_extractor_3d.extract_features_from_pointcloud(pointcloud)
            features_3d = features_3d.to(self.device)
        
        self.logger.info(f"Extracted 3D features: {features_3d.shape}")
        
        return features_3d
    
    def classify(self, features_2d: torch.Tensor, features_3d: torch.Tensor) -> Tuple[str, float, torch.Tensor]:
        """
        Classify object using 2D and 3D features.
        
        Args:
            features_2d: 2D features
            features_3d: 3D features
            
        Returns:
            Tuple of (predicted_class, confidence, class_probabilities)
        """
        self.logger.info("Classifying object using fusion classifier")
        
        # Classification
        with torch.no_grad():
            class_logits, confidence_score = self.fusion_classifier(features_2d, features_3d)
            class_probabilities = torch.softmax(class_logits, dim=1)
            
            predicted_class_idx = torch.argmax(class_probabilities, dim=1).item()
            predicted_class = self.get_class_name(predicted_class_idx)
            confidence = confidence_score.item()
        
        self.logger.info(f"Predicted class: {predicted_class}, Confidence: {confidence:.3f}")
        
        return predicted_class, confidence, class_probabilities
    
    def get_class_name(self, class_idx: int) -> str:
        """
        Get class name from index.
        
        Args:
            class_idx: Class index
            
        Returns:
            Class name
        """
        # Define class names (can be loaded from config)
        class_names = [
            'airplane', 'bathtub', 'bed', 'bench', 'bookshelf',
            'bottle', 'chair', 'cone', 'cup', 'curtain'
        ]
        
        if class_idx < len(class_names):
            return class_names[class_idx]
        else:
            return f"class_{class_idx}"
    
    def predict(self, 
               image_path: str, 
               ground_truth_pcd_path: Optional[str] = None,
               visualize: bool = True,
               save_results: bool = True,
               output_dir: str = "results") -> Dict:
        """
        Complete prediction pipeline.
        
        Args:
            image_path: Path to input image
            ground_truth_pcd_path: Optional path to ground truth point cloud
            visualize: Whether to create visualizations
            save_results: Whether to save results
            output_dir: Output directory for results
            
        Returns:
            Dictionary with prediction results
        """
        self.logger.info(f"Starting prediction for: {image_path}")
        start_time = time.time()
        
        # Process 2D image
        features_2d, input_image = self.process_2d_image(image_path)
        
        # Generate 3D point cloud
        generated_pcd = self.generate_3d_pointcloud(image_path)
        
        # Extract 3D features
        features_3d = self.extract_3d_features(generated_pcd)
        
        # Classification
        predicted_class, confidence, class_probabilities = self.classify(features_2d, features_3d)
        
        # Load ground truth if provided
        ground_truth_pcd = None
        if ground_truth_pcd_path and os.path.exists(ground_truth_pcd_path):
            ground_truth_pcd = o3d.io.read_point_cloud(ground_truth_pcd_path)
            self.logger.info(f"Loaded ground truth point cloud: {len(ground_truth_pcd.points)} points")
        
        # Create results dictionary
        results = {
            'input_image_path': image_path,
            'predicted_class': predicted_class,
            'confidence': confidence,
            'class_probabilities': class_probabilities.cpu().numpy().tolist(),
            'processing_time': time.time() - start_time,
            'timestamp': datetime.now().isoformat()
        }
        
        # Add ground truth if available
        if ground_truth_pcd is not None:
            results['ground_truth_pcd_path'] = ground_truth_pcd_path
            results['ground_truth_points'] = len(ground_truth_pcd.points)
        
        results['generated_points'] = len(generated_pcd.points)
        
        self.logger.info(f"Prediction completed in {results['processing_time']:.2f} seconds")
        
        # Visualization
        if visualize:
            self.create_visualization(
                input_image, generated_pcd, ground_truth_pcd,
                predicted_class, confidence, output_dir, save_results
            )
        
        # Save results
        if save_results:
            self.save_results(results, output_dir)
        
        return results
    
    def create_visualization(self,
                           input_image: Image.Image,
                           generated_pcd: o3d.geometry.PointCloud,
                           ground_truth_pcd: Optional[o3d.geometry.PointCloud],
                           predicted_class: str,
                           confidence: float,
                           output_dir: str,
                           save_results: bool):
        """
        Create visualization of results.
        
        Args:
            input_image: Input image
            generated_pcd: Generated point cloud
            ground_truth_pcd: Ground truth point cloud (optional)
            predicted_class: Predicted class name
            confidence: Prediction confidence
            output_dir: Output directory
            save_results: Whether to save results
        """
        self.logger.info("Creating visualization")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Use ground truth if available, otherwise use generated for comparison
        comparison_pcd = ground_truth_pcd if ground_truth_pcd is not None else generated_pcd
        
        # Create comparison visualization
        fig = self.visualizer.create_comparison_visualization(
            input_image, comparison_pcd, generated_pcd,
            predicted_class, confidence, show=False
        )
        
        if save_results:
            viz_path = os.path.join(output_dir, f"result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
            fig.savefig(viz_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Visualization saved to: {viz_path}")
        
        # Also create interactive visualization
        interactive_fig = self.visualizer.create_interactive_comparison(
            input_image, comparison_pcd, generated_pcd,
            predicted_class, confidence
        )
        
        if save_results:
            interactive_path = os.path.join(output_dir, f"result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html")
            interactive_fig.write_html(interactive_path)
            self.logger.info(f"Interactive visualization saved to: {interactive_path}")
    
    def save_results(self, results: Dict, output_dir: str):
        """
        Save prediction results.
        
        Args:
            results: Results dictionary
            output_dir: Output directory
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Save as JSON
        import json
        results_path = os.path.join(output_dir, f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        self.logger.info(f"Results saved to: {results_path}")
    
    def batch_predict(self,
                     image_paths: list,
                     ground_truth_pcd_paths: Optional[list] = None,
                     output_dir: str = "batch_results") -> List[Dict]:
        """
        Batch prediction on multiple images.
        
        Args:
            image_paths: List of image paths
            ground_truth_pcd_paths: Optional list of ground truth point cloud paths
            output_dir: Output directory
            
        Returns:
            List of prediction results
        """
        self.logger.info(f"Starting batch prediction for {len(image_paths)} images")
        
        results = []
        
        for i, image_path in enumerate(image_paths):
            self.logger.info(f"Processing image {i+1}/{len(image_paths)}: {image_path}")
            
            # Get ground truth if available
            gt_pcd_path = None
            if ground_truth_pcd_paths and i < len(ground_truth_pcd_paths):
                gt_pcd_path = ground_truth_pcd_paths[i]
            
            # Predict
            try:
                result = self.predict(
                    image_path, 
                    gt_pcd_path,
                    visualize=False,  # Don't visualize individual results
                    save_results=False,
                    output_dir=output_dir
                )
                results.append(result)
            except Exception as e:
                self.logger.error(f"Error processing {image_path}: {str(e)}")
                results.append({
                    'input_image_path': image_path,
                    'error': str(e)
                })
        
        # Save batch results
        os.makedirs(output_dir, exist_ok=True)
        import json
        batch_results_path = os.path.join(output_dir, f"batch_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        
        with open(batch_results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        self.logger.info(f"Batch prediction completed. Results saved to: {batch_results_path}")
        
        return results


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description="Hybrid 2D-to-3D Point Cloud Reconstruction and Classification")
    parser.add_argument('--config', type=str, required=True, help='Path to configuration file')
    parser.add_argument('--model', type=str, required=True, help='Path to trained model')
    parser.add_argument('--image', type=str, required=True, help='Path to input image')
    parser.add_argument('--ground_truth_pcd', type=str, help='Path to ground truth point cloud (optional)')
    parser.add_argument('--output_dir', type=str, default='results', help='Output directory')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (cuda/cpu)')
    parser.add_argument('--no_visualize', action='store_true', help='Skip visualization')
    parser.add_argument('--no_save', action='store_true', help='Skip saving results')
    
    args = parser.parse_args()
    
    # Create predictor
    predictor = Hybrid2D3DPredictor(
        config_path=args.config,
        model_path=args.model,
        device=args.device
    )
    
    # Run prediction
    results = predictor.predict(
        image_path=args.image,
        ground_truth_pcd_path=args.ground_truth_pcd,
        visualize=not args.no_visualize,
        save_results=not args.no_save,
        output_dir=args.output_dir
    )
    
    # Print results
    print("\n" + "="*50)
    print("PREDICTION RESULTS")
    print("="*50)
    print(f"Input Image: {results['input_image_path']}")
    print(f"Predicted Class: {results['predicted_class']}")
    print(f"Confidence: {results['confidence']:.3f}")
    print(f"Processing Time: {results['processing_time']:.2f} seconds")
    print("="*50)


if __name__ == "__main__":
    main()