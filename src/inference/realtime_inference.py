#!/usr/bin/env python3
"""
Real-time 3D Reconstruction Inference Pipeline
Supports streaming inference with performance optimization and visualization
"""

import time
import threading
import queue
import numpy as np
import torch
import cv2
from typing import Dict, Any, Optional, Callable, List
from pathlib import Path
import logging
from dataclasses import dataclass
from enum import Enum

from src.models.enhanced_3d_reconstruction import Enhanced3DReconstructionModel
from src.visualization.enhanced_visualizer import Enhanced3DVisualizer, RealTimeVisualizer
from src.optimization.performance_optimizer import ModelQuantizer, TensorRTOptimizer

logger = logging.getLogger(__name__)


class InferenceMode(Enum):
    """Inference modes for different use cases"""
    SINGLE_IMAGE = "single_image"
    BATCH_PROCESSING = "batch_processing"
    STREAMING = "streaming"
    REALTIME = "realtime"


@dataclass
class InferenceConfig:
    """Configuration for real-time inference"""
    mode: InferenceMode = InferenceMode.REALTIME
    batch_size: int = 1
    input_size: tuple = (224, 224)
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    enable_quantization: bool = True
    enable_tensorrt: bool = True
    visualization_enabled: bool = True
    save_results: bool = True
    output_dir: str = "realtime_output"
    max_queue_size: int = 10
    inference_timeout: float = 5.0
    visualization_update_interval: float = 0.1


class RealtimeInferencePipeline:
    """Real-time 3D reconstruction inference pipeline"""
    
    def __init__(self, model: Enhanced3DReconstructionModel, config: InferenceConfig):
        self.model = model
        self.config = config
        self.device = torch.device(config.device)
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Optimization
        self.quantized_model = None
        self.tensorrt_model = None
        self._setup_optimization()
        
        # Inference components
        self.input_queue = queue.Queue(maxsize=config.max_queue_size)
        self.output_queue = queue.Queue(maxsize=config.max_queue_size)
        self.inference_thread = None
        self.visualization_thread = None
        self.is_running = False
        
        # Visualization
        self.visualizer = Enhanced3DVisualizer() if config.visualization_enabled else None
        self.realtime_visualizer = RealTimeVisualizer(config.visualization_update_interval) if config.visualization_enabled else None
        
        # Performance tracking
        self.inference_times = []
        self.frame_count = 0
        self.fps_history = []
        
        # Callbacks
        self.preprocessing_callbacks: List[Callable] = []
        self.postprocessing_callbacks: List[Callable] = []
        self.result_callbacks: List[Callable] = []
        
        logger.info(f"Real-time inference pipeline initialized with mode: {config.mode}")
    
    def _setup_optimization(self):
        """Setup model optimization for real-time inference"""
        if self.config.enable_quantization:
            logger.info("Setting up model quantization...")
            quantizer = ModelQuantizer()
            self.quantized_model = quantizer.quantize_model(self.model)
        
        if self.config.enable_tensorrt and self.device.type == "cuda":
            logger.info("Setting up TensorRT optimization...")
            try:
                tensorrt_optimizer = TensorRTOptimizer()
                example_input = torch.randn(1, 3, *self.config.input_size).to(self.device)
                self.tensorrt_model = tensorrt_optimizer.optimize_model(
                    self.model, example_input, "realtime_tensorrt_model.trt"
                )
            except Exception as e:
                logger.warning(f"TensorRT optimization failed: {e}")
                self.tensorrt_model = None
    
    def add_preprocessing_callback(self, callback: Callable):
        """Add preprocessing callback"""
        self.preprocessing_callbacks.append(callback)
    
    def add_postprocessing_callback(self, callback: Callable):
        """Add postprocessing callback"""
        self.postprocessing_callbacks.append(callback)
    
    def add_result_callback(self, callback: Callable):
        """Add result callback"""
        self.result_callbacks.append(callback)
    
    def preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """Preprocess input image for inference"""
        # Resize image
        if image.shape[:2] != self.config.input_size:
            image = cv2.resize(image, self.config.input_size)
        
        # Convert to RGB if needed
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
        elif image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Normalize
        image = image.astype(np.float32) / 255.0
        image = (image - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
        
        # Apply preprocessing callbacks
        for callback in self.preprocessing_callbacks:
            image = callback(image)
        
        # Convert to tensor
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
        
        return image_tensor.to(self.device)
    
    def postprocess_output(self, output: Dict[str, torch.Tensor]) -> Dict[str, np.ndarray]:
        """Postprocess model output"""
        results = {}
        
        # Convert tensors to numpy arrays
        for key, tensor in output.items():
            if isinstance(tensor, torch.Tensor):
                if tensor.device.type == "cuda":
                    tensor = tensor.cpu()
                results[key] = tensor.detach().numpy()
            else:
                results[key] = tensor
        
        # Apply postprocessing callbacks
        for callback in self.postprocessing_callbacks:
            results = callback(results)
        
        return results
    
    def single_image_inference(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """Run inference on a single image"""
        start_time = time.time()
        
        # Preprocess
        input_tensor = self.preprocess_image(image)
        
        # Select model (prefer optimized versions)
        model = self.tensorrt_model or self.quantized_model or self.model
        
        # Inference
        with torch.no_grad():
            if hasattr(model, 'predict'):
                output = model.predict(input_tensor)
            else:
                output = model(input_tensor)
        
        # Postprocess
        results = self.postprocess_output(output)
        
        # Track performance
        inference_time = time.time() - start_time
        self.inference_times.append(inference_time)
        self.frame_count += 1
        
        # Trigger result callbacks
        for callback in self.result_callbacks:
            callback(results, inference_time)
        
        logger.debug(f"Single image inference completed in {inference_time:.3f}s")
        
        return results
    
    def batch_inference(self, images: List[np.ndarray]) -> List[Dict[str, np.ndarray]]:
        """Run inference on a batch of images"""
        results = []
        
        # Process in batches
        for i in range(0, len(images), self.config.batch_size):
            batch_images = images[i:i + self.config.batch_size]
            
            # Preprocess batch
            batch_tensors = []
            for image in batch_images:
                tensor = self.preprocess_image(image)
                batch_tensors.append(tensor)
            
            # Stack tensors
            batch_input = torch.cat(batch_tensors, dim=0)
            
            # Select model
            model = self.tensorrt_model or self.quantized_model or self.model
            
            # Inference
            start_time = time.time()
            with torch.no_grad():
                if hasattr(model, 'predict_batch'):
                    batch_output = model.predict_batch(batch_input)
                else:
                    batch_output = model(batch_input)
            
            # Postprocess batch results
            for j, output in enumerate(batch_output):
                if isinstance(batch_output, dict):
                    # Handle dictionary output
                    single_output = {k: v[j:j+1] for k, v in batch_output.items()}
                else:
                    # Handle tensor output
                    single_output = {'output': batch_output[j:j+1]}
                
                result = self.postprocess_output(single_output)
                results.append(result)
                
                # Track performance
                self.frame_count += 1
            
            batch_time = time.time() - start_time
            logger.debug(f"Batch inference completed in {batch_time:.3f}s")
        
        return results
    
    def start_streaming_inference(self, input_source: str = "0"):
        """Start streaming inference from camera or video file"""
        if self.is_running:
            logger.warning("Inference pipeline is already running")
            return
        
        self.is_running = True
        
        # Start inference thread
        self.inference_thread = threading.Thread(target=self._inference_worker)
        self.inference_thread.start()
        
        # Start visualization thread if enabled
        if self.config.visualization_enabled:
            self.visualization_thread = threading.Thread(target=self._visualization_worker)
            self.visualization_thread.start()
        
        # Start capture thread
        capture_thread = threading.Thread(target=self._capture_worker, args=(input_source,))
        capture_thread.start()
        
        logger.info(f"Streaming inference started from source: {input_source}")
    
    def stop_streaming_inference(self):
        """Stop streaming inference"""
        self.is_running = False
        
        # Wait for threads to complete
        if self.inference_thread:
            self.inference_thread.join(timeout=5)
        
        if self.visualization_thread:
            self.visualization_thread.join(timeout=5)
        
        if self.realtime_visualizer:
            self.realtime_visualizer.stop_realtime_visualization()
        
        logger.info("Streaming inference stopped")
    
    def _capture_worker(self, input_source: str):
        """Worker thread for capturing frames"""
        cap = cv2.VideoCapture(input_source if input_source != "0" else 0)
        
        if not cap.isOpened():
            logger.error(f"Failed to open video source: {input_source}")
            return
        
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        frame_interval = 1.0 / fps
        
        logger.info(f"Capture started with FPS: {fps}")
        
        try:
            while self.is_running:
                ret, frame = cap.read()
                if not ret:
                    logger.warning("Failed to read frame from video source")
                    break
                
                # Add frame to input queue
                try:
                    self.input_queue.put(frame, timeout=0.1)
                except queue.Full:
                    logger.debug("Input queue full, dropping frame")
                    continue
                
                # Control frame rate
                time.sleep(frame_interval)
        
        finally:
            cap.release()
            logger.info("Capture worker stopped")
    
    def _inference_worker(self):
        """Worker thread for inference"""
        logger.info("Inference worker started")
        
        try:
            while self.is_running:
                try:
                    # Get frame from input queue
                    frame = self.input_queue.get(timeout=1.0)
                    
                    # Run inference
                    start_time = time.time()
                    results = self.single_image_inference(frame)
                    inference_time = time.time() - start_time
                    
                    # Add results to output queue
                    result_package = {
                        'frame': frame,
                        'results': results,
                        'inference_time': inference_time,
                        'timestamp': time.time()
                    }
                    
                    try:
                        self.output_queue.put(result_package, timeout=0.1)
                    except queue.Full:
                        logger.debug("Output queue full, dropping result")
                        continue
                
                except queue.Empty:
                    continue
                except Exception as e:
                    logger.error(f"Inference error: {e}")
                    continue
        
        finally:
            logger.info("Inference worker stopped")
    
    def _visualization_worker(self):
        """Worker thread for visualization"""
        logger.info("Visualization worker started")
        
        # Initialize real-time visualizer
        if self.realtime_visualizer:
            self.realtime_visualizer.start_realtime_visualization()
        
        try:
            while self.is_running:
                try:
                    # Get result from output queue
                    result_package = self.output_queue.get(timeout=0.1)
                    
                    frame = result_package['frame']
                    results = result_package['results']
                    
                    # Extract point cloud if available
                    pointcloud = results.get('pointcloud')
                    if pointcloud is not None:
                        # Update real-time visualization
                        if self.realtime_visualizer:
                            self.realtime_visualizer.update_pointcloud(pointcloud)
                        
                        # Save results if enabled
                        if self.config.save_results:
                            self._save_result(frame, results, result_package['timestamp'])
                    
                    # Update FPS tracking
                    self._update_fps()
                
                except queue.Empty:
                    continue
                except Exception as e:
                    logger.error(f"Visualization error: {e}")
                    continue
        
        finally:
            if self.realtime_visualizer:
                self.realtime_visualizer.stop_realtime_visualization()
            logger.info("Visualization worker stopped")
    
    def _save_result(self, frame: np.ndarray, results: Dict[str, np.ndarray], timestamp: float):
        """Save inference results"""
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # Save frame
        frame_path = output_dir / f"frame_{timestamp:.3f}.jpg"
        cv2.imwrite(str(frame_path), frame)
        
        # Save point cloud if available
        if 'pointcloud' in results:
            pc_path = output_dir / f"pointcloud_{timestamp:.3f}.npy"
            np.save(pc_path, results['pointcloud'])
        
        # Save other results
        results_path = output_dir / f"results_{timestamp:.3f}.json"
        # Convert numpy arrays to lists for JSON serialization
        json_results = {}
        for key, value in results.items():
            if isinstance(value, np.ndarray):
                json_results[key] = value.tolist()
            else:
                json_results[key] = value
        
        import json
        with open(results_path, 'w') as f:
            json.dump(json_results, f, indent=2)
    
    def _update_fps(self):
        """Update FPS calculation"""
        current_time = time.time()
        if len(self.fps_history) > 0:
            fps = 1.0 / (current_time - self.fps_history[-1]) if current_time > self.fps_history[-1] else 0
            if len(self.fps_history) > 30:  # Keep last 30 frames
                self.fps_history.pop(0)
        else:
            fps = 0
        
        self.fps_history.append(current_time)
        
        if self.frame_count % 30 == 0:  # Log every 30 frames
            avg_inference_time = np.mean(self.inference_times[-30:]) if self.inference_times else 0
            logger.info(f"FPS: {fps:.1f}, Avg Inference Time: {avg_inference_time:.3f}s")
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        if not self.inference_times:
            return {"status": "no_data"}
        
        stats = {
            "total_frames": self.frame_count,
            "avg_inference_time": np.mean(self.inference_times),
            "min_inference_time": np.min(self.inference_times),
            "max_inference_time": np.max(self.inference_times),
            "std_inference_time": np.std(self.inference_times),
            "current_fps": len(self.fps_history) / (time.time() - self.fps_history[0]) if self.fps_history else 0,
            "queue_status": {
                "input_queue_size": self.input_queue.qsize(),
                "output_queue_size": self.output_queue.qsize()
            }
        }
        
        return stats
    
    def process_video_file(self, video_path: str, output_path: Optional[str] = None):
        """Process a video file and save results"""
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            logger.error(f"Failed to open video file: {video_path}")
            return
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        logger.info(f"Processing video: {video_path} ({total_frames} frames, {fps} FPS)")
        
        # Create output video writer if output path provided
        out = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        all_results = []
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Run inference
                start_time = time.time()
                results = self.single_image_inference(frame)
                inference_time = time.time() - start_time
                
                # Store results
                result_package = {
                    'frame_id': frame_count,
                    'results': results,
                    'inference_time': inference_time,
                    'timestamp': time.time()
                }
                all_results.append(result_package)
                
                # Visualize if enabled
                if self.config.visualization_enabled and 'pointcloud' in results:
                    # Create simple visualization overlay
                    overlay_text = f"Inference: {inference_time:.3f}s"
                    cv2.putText(frame, overlay_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Write output frame
                if out:
                    out.write(frame)
                
                frame_count += 1
                
                # Progress logging
                if frame_count % 30 == 0:
                    progress = (frame_count / total_frames) * 100
                    avg_time = np.mean([r['inference_time'] for r in all_results[-30:]])
                    logger.info(f"Progress: {progress:.1f}%, Avg inference: {avg_time:.3f}s")
        
        finally:
            cap.release()
            if out:
                out.release()
            
            logger.info(f"Video processing completed: {frame_count} frames processed")
            
            # Save results summary
            if self.config.save_results:
                self._save_video_results(all_results, video_path)
        
        return all_results
    
    def _save_video_results(self, results: List[Dict], video_path: str):
        """Save video processing results"""
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # Save summary
        summary = {
            "video_path": video_path,
            "total_frames": len(results),
            "avg_inference_time": np.mean([r['inference_time'] for r in results]),
            "total_processing_time": sum([r['inference_time'] for r in results]),
            "fps": len(results) / sum([r['inference_time'] for r in results]) if results else 0
        }
        
        import json
        summary_path = output_dir / "video_processing_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Video processing summary saved to {summary_path}")


def create_realtime_demo():
    """Create a real-time demo application"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Real-time 3D Reconstruction Demo")
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model')
    parser.add_argument('--config', type=str, default='configs/enhanced_config.json', help='Config file')
    parser.add_argument('--source', type=str, default='0', help='Video source (0 for webcam, or file path)')
    parser.add_argument('--output', type=str, default=None, help='Output video path')
    parser.add_argument('--no_visualization', action='store_true', help='Disable visualization')
    parser.add_argument('--save_results', action='store_true', help='Save inference results')
    
    args = parser.parse_args()
    
    # Load model
    logger.info(f"Loading model from {args.model_path}")
    model = Enhanced3DReconstructionModel.load_from_checkpoint(args.model_path)
    
    # Create inference config
    config = InferenceConfig(
        mode=InferenceMode.REALTIME,
        visualization_enabled=not args.no_visualization,
        save_results=args.save_results
    )
    
    # Create pipeline
    pipeline = RealtimeInferencePipeline(model, config)
    
    # Add result callback for logging
    def log_result(results, inference_time):
        if 'pointcloud' in results:
            logger.info(f"Generated point cloud with {len(results['pointcloud'])} points in {inference_time:.3f}s")
    
    pipeline.add_result_callback(log_result)
    
    if args.source == "0" or args.source.isdigit():
        # Webcam streaming
        logger.info("Starting webcam streaming inference...")
        pipeline.start_streaming_inference(args.source)
        
        try:
            input("Press Enter to stop streaming...")
        except KeyboardInterrupt:
            pass
        finally:
            pipeline.stop_streaming_inference()
    
    else:
        # Video file processing
        logger.info(f"Processing video file: {args.source}")
        results = pipeline.process_video_file(args.source, args.output)
        
        # Print summary
        if results:
            avg_time = np.mean([r['inference_time'] for r in results])
            logger.info(f"Processing completed. Average inference time: {avg_time:.3f}s")


if __name__ == "__main__":
    create_realtime_demo()