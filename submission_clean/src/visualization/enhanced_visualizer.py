#!/usr/bin/env python3
"""
Enhanced 3D Visualization Module for 3D Reconstruction Results
Supports multiple 3D representations and advanced visualization techniques
"""

import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import trimesh
import open3d as o3d
from typing import Dict, List, Optional, Tuple, Any
import torch
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class Enhanced3DVisualizer:
    """Enhanced 3D visualization with support for multiple representations"""
    
    def __init__(self, width: int = 1200, height: int = 800):
        self.width = width
        self.height = height
        self.color_schemes = {
            'depth': 'viridis',
            'normal': 'rgb',
            'uncertainty': 'plasma',
            'confidence': 'RdYlGn',
            'height': 'terrain'
        }
    
    def visualize_pointcloud_with_uncertainty(self, pointcloud: np.ndarray, 
                                            uncertainty: Optional[np.ndarray] = None,
                                            colors: Optional[np.ndarray] = None,
                                            title: str = "3D Point Cloud with Uncertainty") -> go.Figure:
        """Visualize point cloud with uncertainty quantification"""
        
        fig = go.Figure()
        
        # Base point cloud
        if colors is None:
            colors = pointcloud[:, 2] if pointcloud.shape[1] >= 3 else np.zeros(len(pointcloud))
        
        # Add uncertainty visualization if available
        if uncertainty is not None:
            # Normalize uncertainty for color mapping
            uncertainty_norm = (uncertainty - uncertainty.min()) / (uncertainty.max() - uncertainty.min())
            
            fig.add_trace(go.Scatter3d(
                x=pointcloud[:, 0],
                y=pointcloud[:, 1],
                z=pointcloud[:, 2],
                mode='markers',
                marker=dict(
                    size=3,
                    color=uncertainty_norm,
                    colorscale=self.color_schemes['uncertainty'],
                    opacity=0.8,
                    colorbar=dict(title="Uncertainty"),
                    line=dict(width=1, color='black')
                ),
                name='Point Cloud with Uncertainty'
            ))
        else:
            fig.add_trace(go.Scatter3d(
                x=pointcloud[:, 0],
                y=pointcloud[:, 1],
                z=pointcloud[:, 2],
                mode='markers',
                marker=dict(
                    size=3,
                    color=colors,
                    colorscale=self.color_schemes['depth'],
                    opacity=0.8,
                    colorbar=dict(title="Depth"),
                    line=dict(width=1, color='black')
                ),
                name='Point Cloud'
            ))
        
        # Update layout
        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z',
                aspectmode='data',
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.5)
                )
            ),
            width=self.width,
            height=self.height,
            template='plotly_white'
        )
        
        return fig
    
    def visualize_mesh_comparison(self, predicted_mesh: trimesh.Trimesh, 
                                ground_truth_mesh: Optional[trimesh.Trimesh] = None,
                                error_map: Optional[np.ndarray] = None,
                                title: str = "Mesh Comparison") -> go.Figure:
        """Visualize mesh comparison with error mapping"""
        
        if ground_truth_mesh is not None:
            fig = make_subplots(
                rows=1, cols=2,
                specs=[[{'type': 'mesh3d'}, {'type': 'mesh3d'}]],
                subplot_titles=('Predicted Mesh', 'Ground Truth Mesh')
            )
            
            # Predicted mesh
            fig.add_trace(go.Mesh3d(
                x=predicted_mesh.vertices[:, 0],
                y=predicted_mesh.vertices[:, 1],
                z=predicted_mesh.vertices[:, 2],
                i=predicted_mesh.faces[:, 0],
                j=predicted_mesh.faces[:, 1],
                k=predicted_mesh.faces[:, 2],
                color='lightblue',
                opacity=0.8,
                name='Predicted'
            ), row=1, col=1)
            
            # Ground truth mesh
            fig.add_trace(go.Mesh3d(
                x=ground_truth_mesh.vertices[:, 0],
                y=ground_truth_mesh.vertices[:, 1],
                z=ground_truth_mesh.vertices[:, 2],
                i=ground_truth_mesh.faces[:, 0],
                j=ground_truth_mesh.faces[:, 1],
                k=ground_truth_mesh.faces[:, 2],
                color='lightgreen',
                opacity=0.8,
                name='Ground Truth'
            ), row=1, col=2)
            
        else:
            fig = go.Figure()
            
            # Single mesh with error mapping
            if error_map is not None:
                vertex_colors = error_map
            else:
                vertex_colors = predicted_mesh.vertices[:, 2]
            
            fig.add_trace(go.Mesh3d(
                x=predicted_mesh.vertices[:, 0],
                y=predicted_mesh.vertices[:, 1],
                z=predicted_mesh.vertices[:, 2],
                i=predicted_mesh.faces[:, 0],
                j=predicted_mesh.faces[:, 1],
                k=predicted_mesh.faces[:, 2],
                vertexcolor=vertex_colors,
                colorscale='RdYlBu_r',
                opacity=0.8,
                name='Predicted Mesh'
            ))
        
        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z',
                aspectmode='data'
            ),
            width=self.width,
            height=self.height
        )
        
        return fig
    
    def visualize_nerf_rendering(self, rgb_image: np.ndarray, depth_map: np.ndarray,
                               uncertainty_map: Optional[np.ndarray] = None,
                               alpha_map: Optional[np.ndarray] = None,
                               title: str = "NeRF Rendering Results") -> go.Figure:
        """Visualize NeRF rendering results with multi-view support"""
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('RGB Image', 'Depth Map', 'Uncertainty Map', 'Alpha Map'),
            specs=[[{'type': 'image'}, {'type': 'image'}],
                   [{'type': 'image'}, {'type': 'image'}]]
        )
        
        # RGB Image
        fig.add_trace(go.Image(z=rgb_image), row=1, col=1)
        
        # Depth Map
        fig.add_trace(go.Image(z=depth_map), row=1, col=2)
        
        # Uncertainty Map
        if uncertainty_map is not None:
            fig.add_trace(go.Image(z=uncertainty_map), row=2, col=1)
        else:
            fig.add_trace(go.Image(z=np.zeros_like(depth_map)), row=2, col=1)
        
        # Alpha Map
        if alpha_map is not None:
            fig.add_trace(go.Image(z=alpha_map), row=2, col=2)
        else:
            fig.add_trace(go.Image(z=np.ones_like(depth_map)), row=2, col=2)
        
        fig.update_layout(title=title, height=self.height, width=self.width)
        
        return fig
    
    def visualize_sdf_isosurface(self, sdf_volume: np.ndarray, 
                               isolevel: float = 0.0,
                               gradient_magnitude: Optional[np.ndarray] = None,
                               title: str = "SDF Isosurface") -> go.Figure:
        """Visualize SDF isosurface with gradient information"""
        
        try:
            # Extract isosurface using marching cubes
            from skimage import measure
            
            vertices, faces, normals, values = measure.marching_cubes(
                sdf_volume, level=isolevel
            )
            
            fig = go.Figure()
            
            # Base isosurface
            fig.add_trace(go.Mesh3d(
                x=vertices[:, 0],
                y=vertices[:, 1],
                z=vertices[:, 2],
                i=faces[:, 0],
                j=faces[:, 1],
                k=faces[:, 2],
                vertexcolor=values if gradient_magnitude is None else gradient_magnitude,
                colorscale='viridis',
                opacity=0.8,
                name='SDF Isosurface'
            ))
            
            # Add gradient arrows if available
            if gradient_magnitude is not None and len(vertices) > 0:
                # Sample vertices for gradient visualization
                sample_indices = np.random.choice(len(vertices), min(100, len(vertices)), replace=False)
                sample_vertices = vertices[sample_indices]
                
                # Compute gradients at sample points
                gradients = np.gradient(sdf_volume)
                grad_x = np.interp(sample_vertices[:, 0], np.arange(sdf_volume.shape[0]), gradients[0].flatten())
                grad_y = np.interp(sample_vertices[:, 1], np.arange(sdf_volume.shape[1]), gradients[1].flatten())
                grad_z = np.interp(sample_vertices[:, 2], np.arange(sdf_volume.shape[2]), gradients[2].flatten())
                
                fig.add_trace(go.Cone(
                    x=sample_vertices[:, 0],
                    y=sample_vertices[:, 1],
                    z=sample_vertices[:, 2],
                    u=grad_x,
                    v=grad_y,
                    w=grad_z,
                    sizemode="scaled",
                    sizeref=2,
                    showscale=False,
                    name='Gradient Field'
                ))
            
            fig.update_layout(
                title=title,
                scene=dict(
                    xaxis_title='X',
                    yaxis_title='Y',
                    zaxis_title='Z',
                    aspectmode='data'
                ),
                width=self.width,
                height=self.height
            )
            
            return fig
            
        except ImportError:
            logger.warning("scikit-image not available for isosurface extraction")
            return self.create_placeholder_figure("SDF Isosurface - scikit-image required")
    
    def visualize_occupancy_grid(self, occupancy_grid: np.ndarray, 
                               threshold: float = 0.5,
                               confidence: Optional[np.ndarray] = None,
                               title: str = "Occupancy Grid") -> go.Figure:
        """Visualize occupancy grid with confidence mapping"""
        
        # Convert occupancy to binary and extract surface voxels
        occupied_voxels = occupancy_grid > threshold
        
        if not np.any(occupied_voxels):
            return self.create_placeholder_figure("No occupied voxels found")
        
        # Get occupied voxel coordinates
        occupied_coords = np.argwhere(occupied_voxels)
        
        fig = go.Figure()
        
        # Voxel visualization
        if confidence is not None:
            voxel_colors = confidence[occupied_voxels]
        else:
            voxel_colors = occupancy_grid[occupied_voxels]
        
        fig.add_trace(go.Scatter3d(
            x=occupied_coords[:, 0],
            y=occupied_coords[:, 1],
            z=occupied_coords[:, 2],
            mode='markers',
            marker=dict(
                size=4,
                color=voxel_colors,
                colorscale='RdYlBu',
                opacity=0.8,
                colorbar=dict(title="Occupancy/Confidence"),
                line=dict(width=1, color='black')
            ),
            name='Occupied Voxels'
        ))
        
        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z',
                aspectmode='data'
            ),
            width=self.width,
            height=self.height
        )
        
        return fig
    
    def visualize_multi_view_comparison(self, results_dict: Dict[str, Dict[str, np.ndarray]], 
                                      view_angles: List[Tuple[float, float]] = None) -> go.Figure:
        """Create multi-view comparison of different reconstruction methods"""
        
        if view_angles is None:
            view_angles = [(0, 0), (45, 45), (90, 0), (0, 90)]
        
        n_methods = len(results_dict)
        n_views = len(view_angles)
        
        fig = make_subplots(
            rows=n_views, cols=n_methods,
            subplot_titles=list(results_dict.keys()),
            specs=[[{'type': 'scatter3d'} for _ in range(n_methods)] for _ in range(n_views)]
        )
        
        for col, (method_name, data) in enumerate(results_dict.items()):
            for row, (azimuth, elevation) in enumerate(view_angles):
                
                # Extract point cloud data
                if 'pointcloud' in data:
                    pc = data['pointcloud']
                    colors = data.get('colors', pc[:, 2] if pc.shape[1] >= 3 else np.zeros(len(pc)))
                    
                    fig.add_trace(go.Scatter3d(
                        x=pc[:, 0],
                        y=pc[:, 1],
                        z=pc[:, 2],
                        mode='markers',
                        marker=dict(
                            size=3,
                            color=colors,
                            colorscale='viridis',
                            opacity=0.8
                        ),
                        showlegend=(row == 0),
                        name=method_name
                    ), row=row + 1, col=col + 1)
                
                # Extract mesh data
                elif 'mesh_vertices' in data and 'mesh_faces' in data:
                    vertices = data['mesh_vertices']
                    faces = data['mesh_faces']
                    
                    fig.add_trace(go.Mesh3d(
                        x=vertices[:, 0],
                        y=vertices[:, 1],
                        z=vertices[:, 2],
                        i=faces[:, 0],
                        j=faces[:, 1],
                        k=faces[:, 2],
                        color='lightblue',
                        opacity=0.8,
                        showlegend=(row == 0),
                        name=method_name
                    ), row=row + 1, col=col + 1)
        
        # Update camera positions for different views
        for row in range(n_views):
            for col in range(n_methods):
                azimuth, elevation = view_angles[row]
                fig.update_scenes(
                    row=row + 1, col=col + 1,
                    camera=dict(
                        eye=dict(
                            x=np.cos(np.radians(elevation)) * np.cos(np.radians(azimuth)),
                            y=np.cos(np.radians(elevation)) * np.sin(np.radians(azimuth)),
                            z=np.sin(np.radians(elevation))
                        )
                    )
                )
        
        fig.update_layout(
            title="Multi-view Comparison of 3D Reconstruction Methods",
            height=self.height * n_views,
            width=self.width * n_methods
        )
        
        return fig
    
    def visualize_evaluation_metrics(self, metrics_dict: Dict[str, float], 
                                   title: str = "Evaluation Metrics") -> go.Figure:
        """Visualize evaluation metrics as radar chart and bar chart"""
        
        metrics = list(metrics_dict.keys())
        values = list(metrics_dict.values())
        
        # Create subplots
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Radar Chart', 'Bar Chart'),
            specs=[[{'type': 'polar'}, {'type': 'bar'}]]
        )
        
        # Radar chart
        fig.add_trace(go.Scatterpolar(
            r=values + [values[0]],  # Close the polygon
            theta=metrics + [metrics[0]],
            fill='toself',
            name='Metrics',
            line=dict(color='blue', width=2)
        ), row=1, col=1)
        
        # Bar chart
        fig.add_trace(go.Bar(
            x=metrics,
            y=values,
            name='Metrics',
            marker_color='lightblue',
            text=[f'{v:.3f}' for v in values],
            textposition='auto'
        ), row=1, col=2)
        
        fig.update_layout(
            title=title,
            height=self.height // 2,
            width=self.width
        )
        
        return fig
    
    def visualize_training_progress(self, training_history: Dict[str, List[float]], 
                                  title: str = "Training Progress") -> go.Figure:
        """Visualize training progress with loss curves and metrics"""
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Training Loss', 'Validation Loss', 'Learning Rate', 'Metrics'),
            specs=[[{'type': 'scatter'}, {'type': 'scatter'}],
                   [{'type': 'scatter'}, {'type': 'scatter'}]]
        )
        
        epochs = list(range(1, len(training_history.get('train_loss', [])) + 1))
        
        # Training Loss
        if 'train_loss' in training_history:
            fig.add_trace(go.Scatter(
                x=epochs,
                y=training_history['train_loss'],
                mode='lines',
                name='Training Loss',
                line=dict(color='blue')
            ), row=1, col=1)
        
        # Validation Loss
        if 'val_loss' in training_history:
            fig.add_trace(go.Scatter(
                x=epochs,
                y=training_history['val_loss'],
                mode='lines',
                name='Validation Loss',
                line=dict(color='red')
            ), row=1, col=2)
        
        # Learning Rate
        if 'learning_rate' in training_history:
            fig.add_trace(go.Scatter(
                x=epochs,
                y=training_history['learning_rate'],
                mode='lines',
                name='Learning Rate',
                line=dict(color='green')
            ), row=2, col=1)
        
        # Additional Metrics
        metric_colors = ['purple', 'orange', 'brown', 'pink']
        color_idx = 0
        for metric_name, values in training_history.items():
            if metric_name not in ['train_loss', 'val_loss', 'learning_rate'] and len(values) > 0:
                fig.add_trace(go.Scatter(
                    x=epochs[:len(values)],
                    y=values,
                    mode='lines',
                    name=metric_name.replace('_', ' ').title(),
                    line=dict(color=metric_colors[color_idx % len(metric_colors)])
                ), row=2, col=2)
                color_idx += 1
        
        fig.update_layout(
            title=title,
            height=self.height,
            width=self.width,
            showlegend=True
        )
        
        return fig
    
    def create_placeholder_figure(self, title: str) -> go.Figure:
        """Create a placeholder figure when visualization is not possible"""
        fig = go.Figure()
        fig.add_annotation(
            text=f"{title}<br>Visualization not available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=20)
        )
        fig.update_layout(
            title=title,
            height=self.height,
            width=self.width
        )
        return fig
    
    def save_visualization(self, fig: go.Figure, filepath: str, format: str = 'html'):
        """Save visualization to file"""
        try:
            if format.lower() == 'html':
                fig.write_html(filepath)
            elif format.lower() in ['png', 'jpg', 'jpeg', 'webp']:
                fig.write_image(filepath)
            elif format.lower() == 'json':
                fig.write_json(filepath)
            else:
                logger.warning(f"Unsupported format: {format}. Saving as HTML.")
                fig.write_html(filepath)
            
            logger.info(f"Visualization saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to save visualization: {e}")
    
    def create_interactive_dashboard(self, reconstruction_results: Dict[str, Any]) -> str:
        """Create an interactive HTML dashboard with all visualizations"""
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Enhanced 3D Reconstruction Dashboard</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .visualization {{ margin: 20px 0; border: 1px solid #ddd; padding: 10px; }}
                .metrics {{ background-color: #f5f5f5; padding: 15px; margin: 10px 0; }}
                .metric-item {{ display: inline-block; margin: 5px 15px; }}
                .metric-value {{ font-weight: bold; color: #2196F3; }}
            </style>
        </head>
        <body>
            <h1>Enhanced 3D Reconstruction Results Dashboard</h1>
            <div class="metrics">
                <h3>Evaluation Metrics</h3>
        """
        
        # Add metrics if available
        if 'metrics' in reconstruction_results:
            for metric_name, value in reconstruction_results['metrics'].items():
                html_content += f"""
                <div class="metric-item">
                    {metric_name.replace('_', ' ').title()}: 
                    <span class="metric-value">{value:.4f}</span>
                </div>
                """
        
        html_content += """
            </div>
        """
        
        # Add visualizations
        if 'pointcloud' in reconstruction_results:
            fig = self.visualize_pointcloud_with_uncertainty(
                reconstruction_results['pointcloud'],
                reconstruction_results.get('uncertainty')
            )
            html_content += f"""
            <div class="visualization">
                <h3>Point Cloud Visualization</h3>
                <div id="pointcloud-viz"></div>
                <script>
                    var pointcloudData = {fig.to_json()};
                    Plotly.newPlot('pointcloud-viz', pointcloudData.data, pointcloudData.layout);
                </script>
            </div>
            """
        
        html_content += """
        </body>
        </html>
        """
        
        return html_content


class RealTimeVisualizer:
    """Real-time visualization for streaming 3D reconstruction"""
    
    def __init__(self, update_interval: float = 0.1):
        self.update_interval = update_interval
        self.fig = None
        self.is_running = False
    
    def start_realtime_visualization(self, initial_pointcloud: Optional[np.ndarray] = None):
        """Start real-time visualization"""
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        
        plt.ion()  # Enable interactive mode
        self.fig = plt.figure(figsize=(10, 8))
        self.ax = self.fig.add_subplot(111, projection='3d')
        
        if initial_pointcloud is not None:
            self.update_pointcloud(initial_pointcloud)
        
        self.is_running = True
        logger.info("Real-time visualization started")
    
    def update_pointcloud(self, pointcloud: np.ndarray, colors: Optional[np.ndarray] = None):
        """Update the point cloud visualization"""
        if self.ax is None:
            return
        
        # Clear previous data
        self.ax.clear()
        
        # Plot new point cloud
        if colors is not None:
            self.ax.scatter(pointcloud[:, 0], pointcloud[:, 1], pointcloud[:, 2], 
                          c=colors, s=1, alpha=0.8)
        else:
            self.ax.scatter(pointcloud[:, 0], pointcloud[:, 1], pointcloud[:, 2], 
                          s=1, alpha=0.8)
        
        # Set labels and limits
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        
        # Auto-scale
        max_range = np.array([
            pointcloud[:, 0].max() - pointcloud[:, 0].min(),
            pointcloud[:, 1].max() - pointcloud[:, 1].min(),
            pointcloud[:, 2].max() - pointcloud[:, 2].min()
        ]).max() / 2.0
        
        mid_x = (pointcloud[:, 0].max() + pointcloud[:, 0].min()) * 0.5
        mid_y = (pointcloud[:, 1].max() + pointcloud[:, 1].min()) * 0.5
        mid_z = (pointcloud[:, 2].max() + pointcloud[:, 2].min()) * 0.5
        
        self.ax.set_xlim(mid_x - max_range, mid_x + max_range)
        self.ax.set_ylim(mid_y - max_range, mid_y + max_range)
        self.ax.set_zlim(mid_z - max_range, mid_z + max_range)
        
        plt.draw()
        plt.pause(self.update_interval)
    
    def stop_realtime_visualization(self):
        """Stop real-time visualization"""
        self.is_running = False
        if self.fig is not None:
            plt.ioff()  # Disable interactive mode
            plt.close(self.fig)
        logger.info("Real-time visualization stopped")


def create_comprehensive_visualization_report(results: Dict[str, Any], 
                                            output_dir: str = "visualization_results") -> str:
    """Create a comprehensive visualization report with all results"""
    
    visualizer = Enhanced3DVisualizer()
    os.makedirs(output_dir, exist_ok=True)
    
    report_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>3D Reconstruction Visualization Report</title>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            .section {{ margin: 30px 0; }}
            .visualization {{ margin: 20px 0; text-align: center; }}
            h1, h2 {{ color: #2196F3; }}
            .summary {{ background-color: #f5f5f5; padding: 15px; border-radius: 5px; }}
        </style>
    </head>
    <body>
        <h1>Comprehensive 3D Reconstruction Visualization Report</h1>
        
        <div class="section summary">
            <h2>Summary</h2>
            <p>This report contains visualizations of the 3D reconstruction results including point clouds, 
            meshes, uncertainty quantification, and evaluation metrics.</p>
        </div>
    """
    
    # Generate and save visualizations
    visualizations = []
    
    if 'pointcloud' in results:
        fig = visualizer.visualize_pointcloud_with_uncertainty(
            results['pointcloud'], 
            results.get('uncertainty')
        )
        viz_path = os.path.join(output_dir, "pointcloud_visualization.html")
        visualizer.save_visualization(fig, viz_path)
        visualizations.append(("Point Cloud Visualization", viz_path))
    
    if 'mesh_vertices' in results and 'mesh_faces' in results:
        # Create mesh from vertices and faces
        mesh = trimesh.Trimesh(
            vertices=results['mesh_vertices'],
            faces=results['mesh_faces']
        )
        fig = visualizer.visualize_mesh_comparison(mesh)
        viz_path = os.path.join(output_dir, "mesh_visualization.html")
        visualizer.save_visualization(fig, viz_path)
        visualizations.append(("Mesh Visualization", viz_path))
    
    if 'metrics' in results:
        fig = visualizer.visualize_evaluation_metrics(results['metrics'])
        viz_path = os.path.join(output_dir, "metrics_visualization.html")
        visualizer.save_visualization(fig, viz_path)
        visualizations.append(("Evaluation Metrics", viz_path))
    
    # Add visualizations to report
    for viz_name, viz_path in visualizations:
        report_html += f"""
        <div class="section">
            <h2>{viz_name}</h2>
            <div class="visualization">
                <iframe src="{os.path.basename(viz_path)}" width="100%" height="600" frameborder="0"></iframe>
            </div>
        </div>
        """
    
    report_html += """
    </body>
    </html>
    """
    
    # Save main report
    report_path = os.path.join(output_dir, "visualization_report.html")
    with open(report_path, 'w') as f:
        f.write(report_html)
    
    return report_path