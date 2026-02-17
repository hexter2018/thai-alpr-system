"""
Vehicle Detector using YOLOv8 with TensorRT Optimization
Detects vehicles (car, truck, bus, van, motorcycle) from video frames
"""
import logging
from typing import List, Dict, Tuple, Optional
import numpy as np
import cv2
from pathlib import Path

logger = logging.getLogger(__name__)


class VehicleDetector:
    """
    YOLOv8-based vehicle detector with TensorRT support
    Optimized for NVIDIA RTX 3060
    """
    
    def __init__(
        self,
        model_path: str,
        use_tensorrt: bool = True,
        confidence_threshold: float = 0.5,
        iou_threshold: float = 0.45,
        device: str = "cuda:0"
    ):
        """
        Initialize vehicle detector
        
        Args:
            model_path: Path to vehicles.pt or vehicles.engine
            use_tensorrt: Use TensorRT optimized model
            confidence_threshold: Minimum confidence for detections
            iou_threshold: IoU threshold for NMS
            device: Device to run on
        """
        self.model_path = Path(model_path)
        self.use_tensorrt = use_tensorrt
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.device = device
        
        # Vehicle classes (COCO format)
        self.vehicle_classes = {
            2: "car",
            3: "motorcycle", 
            5: "bus",
            7: "truck",
            # Add custom classes if model is trained differently
        }
        
        # Load model
        self.model = self._load_model()
        
        logger.info(
            f"Vehicle detector initialized: {self.model_path.name}, "
            f"TensorRT: {self.use_tensorrt}, Device: {self.device}"
        )
    
    def _load_model(self):
        """Load YOLOv8 model (PyTorch or TensorRT)"""
        try:
            from ultralytics import YOLO
            
            model_path_str = str(self.model_path)
            
            # Check if TensorRT engine exists
            if self.use_tensorrt:
                engine_path = self.model_path.with_suffix('.engine')
                
                if engine_path.exists():
                    logger.info(f"Loading TensorRT engine: {engine_path}")
                    model = YOLO(str(engine_path))
                else:
                    logger.warning(
                        f"TensorRT engine not found: {engine_path}. "
                        f"Loading PyTorch model and will export to TensorRT."
                    )
                    model = YOLO(model_path_str)
                    # Export to TensorRT will be done on first inference
            else:
                logger.info(f"Loading PyTorch model: {self.model_path}")
                model = YOLO(model_path_str)
            
            # Move to device
            model.to(self.device)
            
            return model
            
        except ImportError:
            logger.error("ultralytics not installed. Install: pip install ultralytics")
            raise
        except Exception as e:
            logger.error(f"Failed to load vehicle detection model: {e}")
            raise
    
    def export_to_tensorrt(
        self,
        img_size: int = 640,
        precision: str = "fp16",
        workspace: int = 4
    ) -> Path:
        """
        Export PyTorch model to TensorRT
        
        Args:
            img_size: Input image size
            precision: fp32, fp16, or int8
            workspace: Max workspace size in GB
        
        Returns:
            Path to exported engine file
        """
        logger.info(f"Exporting to TensorRT (precision: {precision}, workspace: {workspace}GB)")
        
        try:
            # Export model
            engine_path = self.model.export(
                format="engine",
                imgsz=img_size,
                half=(precision == "fp16"),
                int8=(precision == "int8"),
                workspace=workspace,
                device=self.device
            )
            
            logger.info(f"TensorRT export successful: {engine_path}")
            return Path(engine_path)
            
        except Exception as e:
            logger.error(f"TensorRT export failed: {e}")
            raise
    
    def detect(
        self,
        frame: np.ndarray,
        classes: Optional[List[int]] = None
    ) -> List[Dict]:
        """
        Detect vehicles in frame
        
        Args:
            frame: Input frame (BGR)
            classes: Filter by specific classes
        
        Returns:
            List of detections with bbox, confidence, class
        """
        try:
            # Filter for vehicle classes only
            if classes is None:
                classes = list(self.vehicle_classes.keys())
            
            # Run inference
            results = self.model.predict(
                frame,
                conf=self.confidence_threshold,
                iou=self.iou_threshold,
                classes=classes,
                device=self.device,
                verbose=False,
                stream=False
            )
            
            # Parse results
            detections = self._parse_results(results[0])
            
            return detections
            
        except Exception as e:
            logger.error(f"Vehicle detection failed: {e}")
            return []
    
    def _parse_results(self, result) -> List[Dict]:
        """
        Parse YOLO results into standardized format
        
        Returns:
            List of detections: {
                "bbox": [x1, y1, x2, y2],
                "confidence": float,
                "class_id": int,
                "class_name": str
            }
        """
        detections = []
        
        if result.boxes is None or len(result.boxes) == 0:
            return detections
        
        boxes = result.boxes.xyxy.cpu().numpy()  # [x1, y1, x2, y2]
        confidences = result.boxes.conf.cpu().numpy()
        class_ids = result.boxes.cls.cpu().numpy().astype(int)
        
        for box, conf, cls_id in zip(boxes, confidences, class_ids):
            detection = {
                "bbox": box.tolist(),
                "confidence": float(conf),
                "class_id": int(cls_id),
                "class_name": self.vehicle_classes.get(cls_id, "vehicle"),
                "center": self._calculate_center(box)
            }
            detections.append(detection)
        
        return detections
    
    def _calculate_center(self, bbox: np.ndarray) -> Tuple[int, int]:
        """Calculate bounding box center point"""
        x1, y1, x2, y2 = bbox
        center_x = int((x1 + x2) / 2)
        center_y = int((y1 + y2) / 2)
        return (center_x, center_y)
    
    def crop_vehicle_roi(
        self,
        frame: np.ndarray,
        bbox: List[float],
        padding: float = 0.05
    ) -> np.ndarray:
        """
        Crop vehicle region of interest with padding
        
        Args:
            frame: Original frame
            bbox: Bounding box [x1, y1, x2, y2]
            padding: Padding ratio (0.05 = 5% padding)
        
        Returns:
            Cropped image
        """
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = bbox
        
        # Add padding
        pad_w = (x2 - x1) * padding
        pad_h = (y2 - y1) * padding
        
        x1 = max(0, int(x1 - pad_w))
        y1 = max(0, int(y1 - pad_h))
        x2 = min(w, int(x2 + pad_w))
        y2 = min(h, int(y2 + pad_h))
        
        # Crop
        cropped = frame[y1:y2, x1:x2]
        
        return cropped
    
    def draw_detections(
        self,
        frame: np.ndarray,
        detections: List[Dict],
        color: Tuple[int, int, int] = (0, 255, 0),
        thickness: int = 2
    ) -> np.ndarray:
        """
        Draw bounding boxes on frame
        
        Args:
            frame: Input frame
            detections: List of detections
            color: Box color (BGR)
            thickness: Line thickness
        
        Returns:
            Frame with drawn boxes
        """
        output = frame.copy()
        
        for det in detections:
            bbox = det["bbox"]
            conf = det["confidence"]
            class_name = det["class_name"]
            
            x1, y1, x2, y2 = map(int, bbox)
            
            # Draw box
            cv2.rectangle(output, (x1, y1), (x2, y2), color, thickness)
            
            # Draw label
            label = f"{class_name} {conf:.2f}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            label_y = max(y1, label_size[1] + 10)
            
            cv2.rectangle(
                output,
                (x1, label_y - label_size[1] - 10),
                (x1 + label_size[0], label_y),
                color,
                -1
            )
            cv2.putText(
                output,
                label,
                (x1, label_y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                1
            )
        
        return output
    
    def get_model_info(self) -> Dict:
        """Get model information"""
        return {
            "model_path": str(self.model_path),
            "model_name": self.model_path.name,
            "use_tensorrt": self.use_tensorrt,
            "device": self.device,
            "confidence_threshold": self.confidence_threshold,
            "iou_threshold": self.iou_threshold,
            "supported_classes": self.vehicle_classes
        }


def create_vehicle_detector(
    model_path: str,
    use_tensorrt: bool = True,
    confidence_threshold: float = 0.5
) -> VehicleDetector:
    """
    Factory function to create vehicle detector
    
    Args:
        model_path: Path to model file
        use_tensorrt: Use TensorRT optimization
        confidence_threshold: Detection threshold
    
    Returns:
        VehicleDetector instance
    """
    return VehicleDetector(
        model_path=model_path,
        use_tensorrt=use_tensorrt,
        confidence_threshold=confidence_threshold
    )