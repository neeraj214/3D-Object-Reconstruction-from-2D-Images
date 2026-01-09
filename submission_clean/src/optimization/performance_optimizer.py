"""
Performance Optimization Module for 3D Reconstruction Models
Supports quantization, mixed precision training, TensorRT optimization, and edge deployment
"""

import torch
import torch.nn as nn
import torch.quantization as quant
from torch.cuda.amp import autocast, GradScaler
import logging
from typing import Dict, Optional, Tuple, Any
import numpy as np
import time

try:
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit
    TENSORRT_AVAILABLE = True
except ImportError:
    TENSORRT_AVAILABLE = False
    logging.warning("TensorRT not available. Install tensorrt and pycuda for TensorRT optimization.")

try:
    import onnx
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    logging.warning("ONNX not available. Install onnx and onnxruntime for ONNX optimization.")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MixedPrecisionTrainer:
    """Mixed precision training with automatic scaling and optimization."""
    
    def __init__(self, model: nn.Module, optimizer: torch.optim.Optimizer,
                 loss_scale: float = 2.**16, growth_factor: float = 2.0,
                 backoff_factor: float = 0.5, growth_interval: int = 2000):
        self.model = model
        self.optimizer = optimizer
        self.scaler = GradScaler(
            init_scale=loss_scale,
            growth_factor=growth_factor,
            backoff_factor=backoff_factor,
            growth_interval=growth_interval
        )
        self.enabled = torch.cuda.is_available()
        
        if self.enabled:
            logger.info("Mixed precision training enabled")
        else:
            logger.info("CUDA not available, mixed precision disabled")
    
    def train_step(self, inputs: Dict[str, torch.Tensor], 
                   targets: Dict[str, torch.Tensor],
                   loss_fn: callable) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Single training step with mixed precision."""
        self.optimizer.zero_grad()
        
        # Forward pass with autocast
        with autocast(enabled=self.enabled):
            outputs = self.model(**inputs)
            loss = loss_fn(outputs, targets)
        
        # Backward pass with gradient scaling
        self.scaler.scale(loss).backward()
        
        # Gradient clipping
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        # Optimizer step
        self.scaler.step(self.optimizer)
        self.scaler.update()
        
        # Get metrics
        metrics = {
            'loss': loss.item(),
            'scale': self.scaler.get_scale(),
            'grad_norm': self._compute_grad_norm()
        }
        
        return loss, metrics
    
    def _compute_grad_norm(self) -> float:
        """Compute gradient norm for monitoring."""
        total_norm = 0.0
        for p in self.model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        return total_norm ** 0.5

class ModelQuantization:
    """Model quantization for inference optimization."""
    
    def __init__(self, model: nn.Module, quantization_config: Dict):
        self.model = model
        self.config = quantization_config
        self.quantized_model = None
    
    def prepare_model_for_quantization(self) -> nn.Module:
        """Prepare model for quantization by adding quantization stubs."""
        # Add quantization stubs
        self.model.eval()
        
        # Configure quantization
        if self.config.get('backend', 'fbgemm') == 'fbgemm':
            torch.backends.quantized.engine = 'fbgemm'
        elif self.config.get('backend', 'fbgemm') == 'qnnpack':
            torch.backends.quantized.engine = 'qnnpack'
        
        # Fuse modules for better quantization
        self.model = torch.quantization.fuse_modules(
            self.model,
            self.config.get('fuse_modules', []),
            inplace=True
        )
        
        return self.model
    
    def quantize_model(self, calibration_data: Optional[torch.Tensor] = None) -> nn.Module:
        """Quantize the model using post-training static quantization."""
        logger.info("Starting model quantization...")
        
        # Prepare model
        self.prepare_model_for_quantization()
        
        # Configure quantization
        self.model.qconfig = quant.get_default_qconfig(self.config.get('backend', 'fbgemm'))
        
        # Prepare for static quantization
        self.model = quant.prepare(self.model)
        
        # Calibrate with sample data if provided
        if calibration_data is not None:
            logger.info("Calibrating quantization with sample data...")
            with torch.no_grad():
                for batch in calibration_data:
                    _ = self.model(batch)
        
        # Convert to quantized model
        self.quantized_model = quant.convert(self.model)
        
        logger.info("Model quantization completed!")
        return self.quantized_model
    
    def quantize_dynamic(self) -> nn.Module:
        """Apply dynamic quantization to the model."""
        logger.info("Applying dynamic quantization...")
        
        # Apply dynamic quantization
        self.quantized_model = torch.quantization.quantize_dynamic(
            self.model,
            self.config.get('dynamic_quant_modules', {nn.Linear, nn.Conv2d}),
            dtype=self.config.get('dtype', torch.qint8)
        )
        
        logger.info("Dynamic quantization completed!")
        return self.quantized_model
    
    def benchmark_quantized_model(self, test_input: torch.Tensor, num_runs: int = 100) -> Dict[str, float]:
        """Benchmark quantized vs original model."""
        if self.quantized_model is None:
            logger.warning("No quantized model available. Run quantize_model first.")
            return {}
        
        # Warm up
        with torch.no_grad():
            _ = self.model(test_input)
            _ = self.quantized_model(test_input)
        
        # Benchmark original model
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start_time = time.time()
        with torch.no_grad():
            for _ in range(num_runs):
                _ = self.model(test_input)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        original_time = (time.time() - start_time) / num_runs
        
        # Benchmark quantized model
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start_time = time.time()
        with torch.no_grad():
            for _ in range(num_runs):
                _ = self.quantized_model(test_input)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        quantized_time = (time.time() - start_time) / num_runs
        
        # Calculate model sizes
        original_size = sum(p.numel() for p in self.model.parameters()) * 4  # Assuming float32
        quantized_size = self._estimate_quantized_model_size()
        
        results = {
            'original_time_ms': original_time * 1000,
            'quantized_time_ms': quantized_time * 1000,
            'speedup': original_time / quantized_time,
            'original_size_mb': original_size / (1024 * 1024),
            'quantized_size_mb': quantized_size / (1024 * 1024),
            'compression_ratio': original_size / quantized_size
        }
        
        logger.info(f"Benchmark results:")
        logger.info(f"  Original model: {results['original_time_ms']:.2f}ms, {results['original_size_mb']:.2f}MB")
        logger.info(f"  Quantized model: {results['quantized_time_ms']:.2f}ms, {results['quantized_size_mb']:.2f}MB")
        logger.info(f"  Speedup: {results['speedup']:.2f}x, Compression: {results['compression_ratio']:.2f}x")
        
        return results
    
    def _estimate_quantized_model_size(self) -> float:
        """Estimate quantized model size in bytes."""
        if self.quantized_model is None:
            return 0
        
        total_size = 0
        for module in self.quantized_model.modules():
            if hasattr(module, 'weight'):
                if isinstance(module.weight, torch.Tensor):
                    total_size += module.weight.numel() * 1  # int8
                elif hasattr(module.weight, 'dtype'):
                    if module.weight.dtype == torch.qint8:
                        total_size += module.weight.numel() * 1
                    else:
                        total_size += module.weight.numel() * 4  # float32
            
            if hasattr(module, 'bias') and module.bias is not None:
                total_size += module.bias.numel() * 4  # bias usually kept as float32
        
        return total_size

class TensorRTOptimization:
    """TensorRT optimization for NVIDIA GPUs."""
    
    def __init__(self, model: nn.Module, config: Dict):
        self.model = model
        self.config = config
        self.engine = None
        
        if not TENSORRT_AVAILABLE:
            logger.warning("TensorRT not available. TensorRT optimization disabled.")
    
    def convert_to_onnx(self, dummy_input: torch.Tensor, onnx_path: str) -> bool:
        """Convert PyTorch model to ONNX format."""
        if not ONNX_AVAILABLE:
            logger.error("ONNX not available for conversion.")
            return False
        
        logger.info(f"Converting model to ONNX: {onnx_path}")
        
        try:
            torch.onnx.export(
                self.model,
                dummy_input,
                onnx_path,
                export_params=True,
                opset_version=self.config.get('onnx_opset', 11),
                do_constant_folding=True,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
            )
            
            # Verify ONNX model
            onnx_model = onnx.load(onnx_path)
            onnx.checker.check_model(onnx_model)
            
            logger.info("ONNX conversion successful!")
            return True
            
        except Exception as e:
            logger.error(f"ONNX conversion failed: {e}")
            return False
    
    def build_tensorrt_engine(self, onnx_path: str, engine_path: str,
                            max_batch_size: int = 1, max_workspace_size: int = 1 << 30) -> bool:
        """Build TensorRT engine from ONNX model."""
        if not TENSORRT_AVAILABLE:
            logger.error("TensorRT not available.")
            return False
        
        logger.info(f"Building TensorRT engine: {engine_path}")
        
        try:
            # Create TensorRT logger
            TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
            
            # Create builder and network
            builder = trt.Builder(TRT_LOGGER)
            network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
            parser = trt.OnnxParser(network, TRT_LOGGER)
            
            # Parse ONNX model
            with open(onnx_path, 'rb') as model:
                if not parser.parse(model.read()):
                    logger.error("Failed to parse ONNX model")
                    for error in range(parser.num_errors):
                        logger.error(parser.get_error(error))
                    return False
            
            # Configure builder
            config = builder.create_builder_config()
            config.max_workspace_size = max_workspace_size
            
            # Set precision
            if self.config.get('fp16', True):
                config.set_flag(trt.BuilderFlag.FP16)
            if self.config.get('int8', False):
                config.set_flag(trt.BuilderFlag.INT8)
            
            # Build engine
            engine_bytes = builder.build_serialized_network(network, config)
            
            if engine_bytes is None:
                logger.error("Failed to build TensorRT engine")
                return False
            
            # Save engine
            with open(engine_path, 'wb') as f:
                f.write(engine_bytes)
            
            logger.info("TensorRT engine built successfully!")
            return True
            
        except Exception as e:
            logger.error(f"TensorRT engine build failed: {e}")
            return False
    
    def load_tensorrt_engine(self, engine_path: str) -> bool:
        """Load pre-built TensorRT engine."""
        if not TENSORRT_AVAILABLE:
            logger.error("TensorRT not available.")
            return False
        
        try:
            TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
            runtime = trt.Runtime(TRT_LOGGER)
            
            with open(engine_path, 'rb') as f:
                engine_bytes = f.read()
            
            self.engine = runtime.deserialize_cuda_engine(engine_bytes)
            logger.info("TensorRT engine loaded successfully!")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load TensorRT engine: {e}")
            return False

class InferenceOptimizer:
    """Comprehensive inference optimization combining multiple techniques."""
    
    def __init__(self, model: nn.Module, config: Dict):
        self.model = model
        self.config = config
        self.optimization_techniques = []
        
        # Initialize optimization components
        if config.get('mixed_precision', False):
            self.mixed_precision = True
            self.optimization_techniques.append('mixed_precision')
        else:
            self.mixed_precision = False
        
        if config.get('quantization', False):
            self.quantization = ModelQuantization(model, config.get('quantization_config', {}))
            self.optimization_techniques.append('quantization')
        else:
            self.quantization = None
        
        if config.get('tensorrt', False) and TENSORRT_AVAILABLE:
            self.tensorrt = TensorRTOptimization(model, config.get('tensorrt_config', {}))
            self.optimization_techniques.append('tensorrt')
        else:
            self.tensorrt = None
        
        logger.info(f"Initialized inference optimizer with techniques: {self.optimization_techniques}")
    
    def optimize_model(self, sample_input: torch.Tensor) -> nn.Module:
        """Apply all configured optimization techniques."""
        optimized_model = self.model
        
        if self.quantization:
            logger.info("Applying quantization optimization...")
            optimized_model = self.quantization.quantize_model(sample_input)
        
        if self.tensorrt:
            logger.info("Applying TensorRT optimization...")
            onnx_path = self.config.get('onnx_path', 'optimized_model.onnx')
            engine_path = self.config.get('tensorrt_engine_path', 'optimized_model.trt')
            
            if self.tensorrt.convert_to_onnx(sample_input, onnx_path):
                if self.tensorrt.build_tensorrt_engine(onnx_path, engine_path):
                    self.tensorrt.load_tensorrt_engine(engine_path)
                    logger.info("TensorRT optimization completed!")
                else:
                    logger.warning("TensorRT optimization failed, continuing with other optimizations")
        
        return optimized_model
    
    def benchmark_optimizations(self, test_input: torch.Tensor, num_runs: int = 100) -> Dict[str, Any]:
        """Benchmark different optimization techniques."""
        results = {}
        
        # Benchmark original model
        logger.info("Benchmarking original model...")
        original_time = self._benchmark_model(self.model, test_input, num_runs)
        results['original'] = {'time_ms': original_time * 1000}
        
        # Benchmark quantized model if available
        if self.quantization and self.quantization.quantized_model:
            logger.info("Benchmarking quantized model...")
            quantized_time = self._benchmark_model(self.quantization.quantized_model, test_input, num_runs)
            results['quantized'] = {
                'time_ms': quantized_time * 1000,
                'speedup': original_time / quantized_time
            }
        
        # Benchmark TensorRT if available
        if self.tensorrt and self.tensorrt.engine:
            logger.info("Benchmarking TensorRT model...")
            # TensorRT benchmarking would require additional implementation
            results['tensorrt'] = {'status': 'engine_built'}
        
        return results
    
    def _benchmark_model(self, model: nn.Module, test_input: torch.Tensor, num_runs: int) -> float:
        """Benchmark a single model."""
        model.eval()
        
        # Warm up
        with torch.no_grad():
            for _ in range(10):
                _ = model(test_input)
        
        # Benchmark
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start_time = time.time()
        
        with torch.no_grad():
            for _ in range(num_runs):
                _ = model(test_input)
        
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        total_time = time.time() - start_time
        
        return total_time / num_runs

# Configuration templates
OPTIMIZATION_CONFIGS = {
    'mobile_deployment': {
        'mixed_precision': True,
        'quantization': True,
        'quantization_config': {
            'backend': 'qnnpack',
            'dynamic_quant_modules': {nn.Linear, nn.Conv2d},
            'dtype': torch.qint8
        }
    },
    'server_deployment': {
        'mixed_precision': True,
        'tensorrt': True,
        'tensorrt_config': {
            'fp16': True,
            'int8': False,
            'onnx_opset': 11
        }
    },
    'balanced': {
        'mixed_precision': True,
        'quantization': True,
        'quantization_config': {
            'backend': 'fbgemm',
            'dynamic_quant_modules': {nn.Linear},
            'dtype': torch.qint8
        }
    }
}

def get_optimization_config(deployment_target: str = 'balanced') -> Dict:
    """Get optimization configuration for deployment target."""
    return OPTIMIZATION_CONFIGS.get(deployment_target, OPTIMIZATION_CONFIGS['balanced'])