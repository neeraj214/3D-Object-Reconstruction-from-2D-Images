import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import open3d as o3d
from PIL import Image
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Union, Optional, Tuple, List
import torch


class PointCloudVisualizer:
    """
    Visualization utilities for point clouds and results.
    """
    
    def __init__(self, 
                 point_size: int = 2,
                 background_color: str = 'black',
                 colormap: str = 'viridis',
                 save_format: str = 'png',
                 dpi: int = 300):
        """
        Initialize the visualizer.
        
        Args:
            point_size: Size of points in visualization
            background_color: Background color
            colormap: Colormap for point coloring
            save_format: Format for saving figures
            dpi: DPI for saved figures
        """
        self.point_size = point_size
        self.background_color = background_color
        self.colormap = colormap
        self.save_format = save_format
        self.dpi = dpi
        
        # Set matplotlib style
        plt.style.use('default')
        if background_color == 'black':
            plt.style.use('dark_background')
    
    def visualize_pointcloud_matplotlib(self, 
                                      pointcloud: Union[o3d.geometry.PointCloud, np.ndarray],
                                      title: str = "Point Cloud",
                                      ax: Optional[plt.Axes] = None,
                                      show: bool = True,
                                      save_path: Optional[str] = None) -> Optional[plt.Figure]:
        """
        Visualize point cloud using matplotlib.
        
        Args:
            pointcloud: Input point cloud
            title: Plot title
            ax: Existing axes to plot on
            show: Whether to show the plot
            save_path: Path to save the figure
            
        Returns:
            Figure object if not showing
        """
        # Convert to numpy
        if isinstance(pointcloud, o3d.geometry.PointCloud):
            points = np.asarray(pointcloud.points)
            if pointcloud.has_colors():
                colors = np.asarray(pointcloud.colors)
            else:
                colors = None
        else:
            points = pointcloud
            colors = None
        
        # Create figure if no axes provided
        if ax is None:
            fig = plt.figure(figsize=(8, 8))
            ax = fig.add_subplot(111, projection='3d')
        else:
            fig = ax.figure
        
        # Plot points
        if colors is not None:
            ax.scatter(points[:, 0], points[:, 1], points[:, 2], 
                      c=colors, s=self.point_size, alpha=0.8)
        else:
            ax.scatter(points[:, 0], points[:, 1], points[:, 2], 
                      c=points[:, 2], cmap=self.colormap, s=self.point_size, alpha=0.8)
        
        # Set labels and title
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(title)
        
        # Set equal aspect ratio
        max_range = np.array([points[:, 0].max() - points[:, 0].min(),
                             points[:, 1].max() - points[:, 1].min(),
                             points[:, 2].max() - points[:, 2].min()]).max() / 2.0
        
        mid_x = (points[:, 0].max() + points[:, 0].min()) * 0.5
        mid_y = (points[:, 1].max() + points[:, 1].min()) * 0.5
        mid_z = (points[:, 2].max() + points[:, 2].min()) * 0.5
        
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
        
        # Save if requested
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        
        if show:
            plt.show()
        else:
            return fig
    
    def visualize_pointcloud_plotly(self, 
                                   pointcloud: Union[o3d.geometry.PointCloud, np.ndarray],
                                   title: str = "Point Cloud",
                                   colors: Optional[np.ndarray] = None) -> go.Figure:
        """
        Visualize point cloud using plotly (interactive).
        
        Args:
            pointcloud: Input point cloud
            title: Plot title
            colors: Optional colors for points
            
        Returns:
            Plotly figure object
        """
        # Convert to numpy
        if isinstance(pointcloud, o3d.geometry.PointCloud):
            points = np.asarray(pointcloud.points)
            if colors is None and pointcloud.has_colors():
                colors = np.asarray(pointcloud.colors)
        else:
            points = pointcloud
        
        # Create colors if not provided
        if colors is None:
            colors = points[:, 2]  # Use z-coordinate for coloring
        
        # Create 3D scatter plot
        fig = go.Figure(data=[go.Scatter3d(
            x=points[:, 0],
            y=points[:, 1],
            z=points[:, 2],
            mode='markers',
            marker=dict(
                size=self.point_size,
                color=colors,
                colorscale=self.colormap,
                opacity=0.8
            )
        )])
        
        # Update layout
        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z',
                aspectmode='cube'
            ),
            width=800,
            height=800
        )
        
        return fig
    
    def create_comparison_visualization(self,
                                        input_image: Union[str, np.ndarray, Image.Image],
                                        ground_truth_pcd: Union[o3d.geometry.PointCloud, np.ndarray],
                                        generated_pcd: Union[o3d.geometry.PointCloud, np.ndarray],
                                        predicted_class: str = "Unknown",
                                        confidence: float = 0.0,
                                        save_path: Optional[str] = None) -> plt.Figure:
        """
        Create the main comparison visualization with three plots.
        
        Args:
            input_image: Input 2D image
            ground_truth_pcd: Ground truth point cloud
            generated_pcd: Generated point cloud
            predicted_class: Predicted object class
            confidence: Prediction confidence
            save_path: Path to save the figure
            
        Returns:
            Matplotlib figure object
        """
        # Load and process input image
        if isinstance(input_image, str):
            img = Image.open(input_image).convert('RGB')
            img_array = np.array(img)
        elif isinstance(input_image, Image.Image):
            img_array = np.array(input_image)
        elif isinstance(input_image, np.ndarray):
            img_array = input_image
        else:
            raise ValueError(f"Unsupported image type: {type(input_image)}")
        
        # Convert point clouds to numpy
        if isinstance(ground_truth_pcd, o3d.geometry.PointCloud):
            gt_points = np.asarray(ground_truth_pcd.points)
            gt_colors = np.asarray(ground_truth_pcd.colors) if ground_truth_pcd.has_colors() else None
        else:
            gt_points = ground_truth_pcd
            gt_colors = None
        
        if isinstance(generated_pcd, o3d.geometry.PointCloud):
            gen_points = np.asarray(generated_pcd.points)
            gen_colors = np.asarray(generated_pcd.colors) if generated_pcd.has_colors() else None
        else:
            gen_points = generated_pcd
            gen_colors = None
        
        # Create figure with three subplots
        fig = plt.figure(figsize=(18, 6))
        
        # Plot 1: Input Image
        ax1 = fig.add_subplot(131)
        ax1.imshow(img_array)
        ax1.set_title('Input 2D Image', fontsize=14, fontweight='bold')
        ax1.axis('off')
        
        # Plot 2: Ground Truth Point Cloud
        ax2 = fig.add_subplot(132, projection='3d')
        if gt_colors is not None:
            ax2.scatter(gt_points[:, 0], gt_points[:, 1], gt_points[:, 2], 
                       c=gt_colors, s=self.point_size, alpha=0.8)
        else:
            ax2.scatter(gt_points[:, 0], gt_points[:, 1], gt_points[:, 2], 
                       c=gt_points[:, 2], cmap=self.colormap, s=self.point_size, alpha=0.8)
        
        ax2.set_title('Ground Truth Point Cloud', fontsize=14, fontweight='bold')
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_zlabel('Z')
        
        # Plot 3: Generated Point Cloud
        ax3 = fig.add_subplot(133, projection='3d')
        if gen_colors is not None:
            ax3.scatter(gen_points[:, 0], gen_points[:, 1], gen_points[:, 2], 
                       c=gen_colors, s=self.point_size, alpha=0.8)
        else:
            ax3.scatter(gen_points[:, 0], gen_points[:, 1], gen_points[:, 2], 
                       c=gen_points[:, 2], cmap=self.colormap, s=self.point_size, alpha=0.8)
        
        ax3.set_title('Generated Point Cloud (Ours)', fontsize=14, fontweight='bold')
        ax3.set_xlabel('X')
        ax3.set_ylabel('Y')
        ax3.set_zlabel('Z')
        
        # Add prediction text
        fig.suptitle(f'Hybrid 2D-to-3D Reconstruction and Classification\n'
                    f'Predicted Class: {predicted_class} | Confidence: {confidence:.3f}', 
                    fontsize=16, fontweight='bold', y=0.95)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save if requested
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        
        return fig
    
    def create_interactive_comparison(self,
                                     input_image: Union[str, np.ndarray, Image.Image],
                                     ground_truth_pcd: Union[o3d.geometry.PointCloud, np.ndarray],
                                     generated_pcd: Union[o3d.geometry.PointCloud, np.ndarray],
                                     predicted_class: str = "Unknown",
                                     confidence: float = 0.0) -> go.Figure:
        """
        Create interactive comparison using plotly.
        
        Args:
            input_image: Input 2D image
            ground_truth_pcd: Ground truth point cloud
            generated_pcd: Generated point cloud
            predicted_class: Predicted object class
            confidence: Prediction confidence
            
        Returns:
            Plotly figure object
        """
        # Load input image
        if isinstance(input_image, str):
            img = Image.open(input_image).convert('RGB')
            img_array = np.array(img)
        elif isinstance(input_image, Image.Image):
            img_array = np.array(input_image)
        elif isinstance(input_image, np.ndarray):
            img_array = input_image
        else:
            raise ValueError(f"Unsupported image type: {type(input_image)}")
        
        # Convert point clouds
        if isinstance(ground_truth_pcd, o3d.geometry.PointCloud):
            gt_points = np.asarray(ground_truth_pcd.points)
            gt_colors = np.asarray(ground_truth_pcd.colors) if ground_truth_pcd.has_colors() else gt_points[:, 2]
        else:
            gt_points = ground_truth_pcd
            gt_colors = gt_points[:, 2]
        
        if isinstance(generated_pcd, o3d.geometry.PointCloud):
            gen_points = np.asarray(generated_pcd.points)
            gen_colors = np.asarray(generated_pcd.colors) if generated_pcd.has_colors() else gen_points[:, 2]
        else:
            gen_points = generated_pcd
            gen_colors = gen_points[:, 2]
        
        # Create subplots
        fig = make_subplots(
            rows=1, cols=3,
            subplot_titles=('Input 2D Image', 'Ground Truth Point Cloud', 'Generated Point Cloud (Ours)'),
            specs=[[{"type": "image"}, {"type": "scatter3d"}, {"type": "scatter3d"}]]
        )
        
        # Add input image
        fig.add_trace(
            go.Image(z=img_array),
            row=1, col=1
        )
        
        # Add ground truth point cloud
        fig.add_trace(
            go.Scatter3d(
                x=gt_points[:, 0],
                y=gt_points[:, 1],
                z=gt_points[:, 2],
                mode='markers',
                marker=dict(
                    size=self.point_size,
                    color=gt_colors,
                    colorscale=self.colormap,
                    opacity=0.8
                ),
                name='Ground Truth'
            ),
            row=1, col=2
        )
        
        # Add generated point cloud
        fig.add_trace(
            go.Scatter3d(
                x=gen_points[:, 0],
                y=gen_points[:, 1],
                z=gen_points[:, 2],
                mode='markers',
                marker=dict(
                    size=self.point_size,
                    color=gen_colors,
                    colorscale=self.colormap,
                    opacity=0.8
                ),
                name='Generated'
            ),
            row=1, col=3
        )
        
        # Update layout
        fig.update_layout(
            title=dict(
                text=f'Hybrid 2D-to-3D Reconstruction and Classification<br>'
                     f'<sub>Predicted Class: {predicted_class} | Confidence: {confidence:.3f}</sub>',
                font=dict(size=16)
            ),
            height=600,
            width=1200
        )
        
        return fig
    
    def visualize_depth_map(self,
                           depth_map: np.ndarray,
                           title: str = "Depth Map",
                           colormap: str = 'plasma') -> plt.Figure:
        """
        Visualize depth map.
        
        Args:
            depth_map: Depth map as numpy array
            title: Plot title
            colormap: Colormap for depth visualization
            
        Returns:
            Matplotlib figure object
        """
        fig, ax = plt.subplots(figsize=(8, 6))
        
        im = ax.imshow(depth_map, cmap=colormap, aspect='auto')
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Depth Value', rotation=270, labelpad=15)
        
        plt.tight_layout()
        return fig
    
    def save_visualization(self, fig: Union[plt.Figure, go.Figure], save_path: str):
        """
        Save visualization to file.
        
        Args:
            fig: Figure object (matplotlib or plotly)
            save_path: Path to save the figure
        """
        if isinstance(fig, plt.Figure):
            fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        elif isinstance(fig, go.Figure):
            fig.write_html(save_path.replace('.png', '.html'))
            fig.write_image(save_path, width=1200, height=600, scale=2)
        else:
            raise ValueError(f"Unsupported figure type: {type(fig)}")


class ResultsVisualizer:
    """
    Specialized visualizer for classification results and analysis.
    """
    
    def __init__(self):
        """Initialize the results visualizer."""
        pass
    
    def plot_confusion_matrix(self, 
                             y_true: np.ndarray, 
                             y_pred: np.ndarray, 
                             class_names: List[str],
                             title: str = "Confusion Matrix") -> plt.Figure:
        """
        Plot confusion matrix.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            class_names: List of class names
            title: Plot title
            
        Returns:
            Matplotlib figure object
        """
        from sklearn.metrics import confusion_matrix
        import seaborn as sns
        
        cm = confusion_matrix(y_true, y_pred)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names, ax=ax)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        
        plt.tight_layout()
        return fig
    
    def plot_classification_report(self,
                                  y_true: np.ndarray,
                                  y_pred: np.ndarray,
                                  class_names: List[str]) -> plt.Figure:
        """
        Plot classification report as heatmap.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            class_names: List of class names
            
        Returns:
            Matplotlib figure object
        """
        from sklearn.metrics import classification_report
        import pandas as pd
        
        report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
        report_df = pd.DataFrame(report).iloc[:-1, :-3].T  # Remove support and averages
        
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.heatmap(report_df, annot=True, fmt='.3f', cmap='YlOrRd', ax=ax)
        ax.set_title('Classification Report', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        return fig
    
    def plot_feature_importance(self,
                               feature_importance: dict,
                               title: str = "Feature Importance") -> plt.Figure:
        """
        Plot feature importance from fusion classifier.
        
        Args:
            feature_importance: Dictionary with importance scores
            title: Plot title
            
        Returns:
            Matplotlib figure object
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        if feature_importance['fusion_method'] == 'attention':
            attention_2d = feature_importance['attention_2d']
            attention_3d = feature_importance['attention_3d']
            
            x = np.arange(len(attention_2d))
            width = 0.35
            
            ax.bar(x - width/2, attention_2d, width, label='2D Features', alpha=0.8)
            ax.bar(x + width/2, attention_3d, width, label='3D Features', alpha=0.8)
            
            ax.set_xlabel('Sample Index')
            ax.set_ylabel('Attention Weight')
            ax.set_title(title)
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        elif feature_importance['fusion_method'] == 'add':
            similarity = feature_importance['similarity']
            ax.hist(similarity, bins=20, alpha=0.7, edgecolor='black')
            ax.set_xlabel('Cosine Similarity')
            ax.set_ylabel('Frequency')
            ax.set_title(f'{title} - Cosine Similarity Distribution')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig


def create_visualizer(config: dict) -> PointCloudVisualizer:
    """
    Factory function to create a visualizer from config.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        PointCloudVisualizer instance
    """
    viz_config = config.get('visualization', {})
    
    return PointCloudVisualizer(
        point_size=viz_config.get('point_size', 2),
        background_color=viz_config.get('background_color', 'black'),
        colormap=viz_config.get('colormap', 'viridis'),
        save_format=viz_config.get('save_format', 'png'),
        dpi=viz_config.get('dpi', 300)
    )


if __name__ == "__main__":
    # Test the visualization utilities
    print("Testing Point Cloud Visualization...")
    
    # Create test data
    test_points = np.random.randn(1000, 3) * 0.5
    test_colors = np.random.rand(1000, 3)
    
    # Create test point cloud
    test_pcd = o3d.geometry.PointCloud()
    test_pcd.points = o3d.utility.Vector3dVector(test_points)
    test_pcd.colors = o3d.utility.Vector3dVector(test_colors)
    
    # Create visualizer
    visualizer = PointCloudVisualizer()
    
    # Test matplotlib visualization
    print("Testing matplotlib visualization...")
    fig = visualizer.visualize_pointcloud_matplotlib(test_pcd, show=False)
    print("Matplotlib visualization test completed!")
    
    # Test plotly visualization
    print("Testing plotly visualization...")
    plotly_fig = visualizer.visualize_pointcloud_plotly(test_pcd)
    print("Plotly visualization test completed!")
    
    # Test comparison visualization
    print("Testing comparison visualization...")
    test_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    comparison_fig = visualizer.create_comparison_visualization(
        test_image, test_pcd, test_pcd, "Test Class", 0.95, show=False
    )
    print("Comparison visualization test completed!")
    
    print("All visualization tests completed successfully!")