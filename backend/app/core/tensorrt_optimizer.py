"""
TensorRT Model Optimization Utilities
"""
import logging
from pathlib import Path
from typing import Optional
import torch

logger = logging.getLogger(__name__)


class TensorRTOptimizer:
    """Utilities for converting PyTorch models to TensorRT"""
    
    @staticmethod
    def export_yolo_to_tensorrt(
        model_path: str,
        output_path: Optional[str] = None,
        img_size: int = 640,
        precision: str = "fp16",
        workspace: int = 4,
        batch_size: int = 1,
        device: str = "cuda:0"
    ) -> Path:
        """
        Export YOLOv8 model to TensorRT
        
        Args:
            model_path: Path to .pt model
            output_path: Output .engine path
            img_size: Input image size
            precision: fp32, fp16, or int8
            workspace: Max workspace in GB
            batch_size: Batch size
            device: CUDA device
        
        Returns:
            Path to exported engine
        """
        try:
            from ultralytics import YOLO
            
            model_path = Path(model_path)
            
            if output_path is None:
                output_path = model_path.with_suffix('.engine')
            
            logger.info(f"Exporting {model_path} to TensorRT...")
            logger.info(f"Precision: {precision}, Workspace: {workspace}GB")
            
            # Load model
            model = YOLO(str(model_path))
            
            # Export to TensorRT
            engine_path = model.export(
                format='engine',
                imgsz=img_size,
                half=(precision == 'fp16'),
                int8=(precision == 'int8'),
                workspace=workspace,
                batch=batch_size,
                device=device,
                simplify=True,
                verbose=True
            )
            
            logger.info(f"Export successful: {engine_path}")
            return Path(engine_path)
            
        except Exception as e:
            logger.error(f"TensorRT export failed: {e}")
            raise
    
    @staticmethod
    def verify_tensorrt_engine(engine_path: str) -> bool:
        """
        Verify TensorRT engine is valid
        
        Args:
            engine_path: Path to .engine file
        
        Returns:
            True if valid
        """
        try:
            from ultralytics import YOLO
            
            model = YOLO(engine_path)
            
            # Try a dummy prediction
            import numpy as np
            dummy_img = np.zeros((640, 640, 3), dtype=np.uint8)
            results = model.predict(dummy_img, verbose=False)
            
            logger.info(f"Engine verification passed: {engine_path}")
            return True
            
        except Exception as e:
            logger.error(f"Engine verification failed: {e}")
            return False
    
    @staticmethod
    def compare_model_speed(
        pytorch_path: str,
        tensorrt_path: str,
        num_iterations: int = 100
    ) -> dict:
        """
        Compare PyTorch vs TensorRT inference speed
        
        Args:
            pytorch_path: Path to .pt model
            tensorrt_path: Path to .engine model
            num_iterations: Number of test iterations
        
        Returns:
            Speed comparison dict
        """
        import time
        import numpy as np
        from ultralytics import YOLO
        
        dummy_img = np.zeros((640, 640, 3), dtype=np.uint8)
        
        # Test PyTorch
        logger.info("Testing PyTorch model...")
        pytorch_model = YOLO(pytorch_path)
        pytorch_times = []
        
        for _ in range(num_iterations):
            start = time.time()
            pytorch_model.predict(dummy_img, verbose=False)
            pytorch_times.append(time.time() - start)
        
        # Test TensorRT
        logger.info("Testing TensorRT model...")
        tensorrt_model = YOLO(tensorrt_path)
        tensorrt_times = []
        
        for _ in range(num_iterations):
            start = time.time()
            tensorrt_model.predict(dummy_img, verbose=False)
            tensorrt_times.append(time.time() - start)
        
        pytorch_avg = np.mean(pytorch_times) * 1000  # ms
        tensorrt_avg = np.mean(tensorrt_times) * 1000  # ms
        speedup = pytorch_avg / tensorrt_avg
        
        results = {
            "pytorch_avg_ms": pytorch_avg,
            "tensorrt_avg_ms": tensorrt_avg,
            "speedup": speedup,
            "iterations": num_iterations
        }
        
        logger.info(f"PyTorch: {pytorch_avg:.2f}ms, TensorRT: {tensorrt_avg:.2f}ms, Speedup: {speedup:.2f}x")
        
        return results
    
    @staticmethod
    def get_model_info(model_path: str) -> dict:
        """Get model information"""
        try:
            from ultralytics import YOLO
            
            model = YOLO(model_path)
            
            return {
                "path": model_path,
                "type": "TensorRT" if model_path.endswith('.engine') else "PyTorch",
                "task": model.task,
                "model_name": model.model_name if hasattr(model, 'model_name') else None
            }
        except Exception as e:
            logger.error(f"Failed to get model info: {e}")
            return {}