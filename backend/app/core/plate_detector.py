"""
License Plate Detector using YOLOv8
Detects and crops license plates from vehicle images
"""
import logging
from typing import List, Dict, Optional, Tuple
import numpy as np
import cv2
from pathlib import Path

logger = logging.getLogger(__name__)


class PlateDetector:
    """
    YOLOv8-based license plate detector with TensorRT support
    Specialized for Thai license plates
    """
    
    def __init__(
        self,
        model_path: str,
        use_tensorrt: bool = True,
        confidence_threshold: float = 0.4,
        iou_threshold: float = 0.45,
        device: str = "cuda:0"
    ):
        """
        Initialize plate detector
        
        Args:
            model_path: Path to best.pt or best.engine
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
        
        # Load model
        self.model = self._load_model()
        
        logger.info(
            f"Plate detector initialized: {self.model_path.name}, "
            f"TensorRT: {self.use_tensorrt}, Device: {self.device}"
        )
    
    def _load_model(self):
        """Load YOLOv8 model (PyTorch/ONNX or TensorRT)."""
        try:
            from ultralytics import YOLO
            
            model_path = self.model_path
            
            # Check if TensorRT engine exists
            if self.use_tensorrt:
                engine_path = model_path.with_suffix('.engine')
                
                if engine_path.exists():
                    logger.info(f"Loading TensorRT engine: {engine_path}")
                    model = YOLO(str(engine_path))
                else:
                    fallback_path = self._resolve_fallback_model_path()
                    logger.warning(
                        f"TensorRT engine not found: {engine_path}. "
                        f"Loading fallback model from {fallback_path}."
                    )
                    model = YOLO(str(fallback_path))
                    model = self._try_export_and_reload_tensorrt(model, engine_path, fallback_path)
            else:
                fallback_path = self._resolve_fallback_model_path()
                logger.info(f"Loading model without TensorRT: {fallback_path}")
                model = YOLO(str(fallback_path))

            model.to(self.device)
            
            return model
            
        except ImportError:
            logger.error("ultralytics not installed. Install: pip install ultralytics")
            raise
        except Exception as e:
            logger.error(f"Failed to load plate detection model: {e}")
            raise

    def _resolve_fallback_model_path(self) -> Path:
        """Resolve an existing source model path for loading/export fallback."""
        if self.model_path.exists() and self.model_path.suffix != '.engine':
            return self.model_path

        candidates = [
            self.model_path.with_suffix('.pt'),
            self.model_path.with_suffix('.onnx'),
            self.model_path.with_name('best.pt'),
            self.model_path.with_name('best.onnx'),
        ]

        for candidate in candidates:
            if candidate.exists():
                return candidate

        raise FileNotFoundError(
            f"TensorRT engine not found at {self.model_path} and no fallback model (.pt/.onnx) was found. "
            f"Checked: {', '.join(str(path) for path in candidates)}"
        )

    def _try_export_and_reload_tensorrt(self, model, engine_path: Path, source_path: Path):
        """Try exporting fallback model to TensorRT and reload if export succeeds."""
        try:
            logger.info(f"Attempting TensorRT export from fallback model: {source_path}")
            exported_path = Path(
                model.export(
                    format='engine',
                    imgsz=640,
                    half=True,
                    workspace=4,
                    device=self.device,
                )
            )

            selected_engine = exported_path if exported_path.exists() else engine_path
            if selected_engine.exists():
                from ultralytics import YOLO
                model = YOLO(str(selected_engine))
                model.to(self.device)
                logger.info(f"TensorRT export successful and reloaded engine: {selected_engine}")
            else:
                logger.warning(
                    f"TensorRT export reported success but engine file was not found. "
                    f"Expected one of: {exported_path}, {engine_path}."
                )
        except Exception as export_error:
            logger.warning(
                f"TensorRT export skipped/failed; continuing with fallback model {source_path}: {export_error}"
            )

        return model
    
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
        vehicle_image: np.ndarray,
        return_best_only: bool = True
    ) -> Optional[Dict]:
        """
        Detect license plate in vehicle image
        
        Args:
            vehicle_image: Cropped vehicle ROI
            return_best_only: Return only highest confidence detection
        
        Returns:
            Detection dict or None if no plate found
        """
        try:
            # Run inference
            results = self.model.predict(
                vehicle_image,
                conf=self.confidence_threshold,
                iou=self.iou_threshold,
                device=self.device,
                verbose=False,
                stream=False
            )
            
            # Parse results
            detections = self._parse_results(results[0])
            
            if not detections:
                return None
            
            if return_best_only:
                # Return highest confidence detection
                best = max(detections, key=lambda x: x["confidence"])
                return best
            
            return detections
            
        except Exception as e:
            logger.error(f"Plate detection failed: {e}")
            return None
    
    def _parse_results(self, result) -> List[Dict]:
        """
        Parse YOLO results into standardized format
        
        Returns:
            List of detections with bbox and confidence
        """
        detections = []
        
        if result.boxes is None or len(result.boxes) == 0:
            return detections
        
        boxes = result.boxes.xyxy.cpu().numpy()  # [x1, y1, x2, y2]
        confidences = result.boxes.conf.cpu().numpy()
        
        for box, conf in zip(boxes, confidences):
            detection = {
                "bbox": box.tolist(),
                "confidence": float(conf),
                "center": self._calculate_center(box),
                "area": self._calculate_area(box)
            }
            detections.append(detection)
        
        return detections
    
    def _calculate_center(self, bbox: np.ndarray) -> Tuple[int, int]:
        """Calculate bounding box center point"""
        x1, y1, x2, y2 = bbox
        center_x = int((x1 + x2) / 2)
        center_y = int((y1 + y2) / 2)
        return (center_x, center_y)
    
    def _calculate_area(self, bbox: np.ndarray) -> int:
        """Calculate bounding box area"""
        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1
        return int(width * height)
    
    def crop_plate(
        self,
        vehicle_image: np.ndarray,
        bbox: List[float],
        padding: float = 0.05,
        min_height: int = 40
    ) -> Optional[np.ndarray]:
        """
        Crop license plate with padding and quality checks
        
        Args:
            vehicle_image: Vehicle ROI image
            bbox: Plate bounding box [x1, y1, x2, y2]
            padding: Padding ratio
            min_height: Minimum plate height (pixels)
        
        Returns:
            Cropped plate image or None if invalid
        """
        h, w = vehicle_image.shape[:2]
        x1, y1, x2, y2 = bbox
        
        # Add padding
        pad_w = (x2 - x1) * padding
        pad_h = (y2 - y1) * padding
        
        x1 = max(0, int(x1 - pad_w))
        y1 = max(0, int(y1 - pad_h))
        x2 = min(w, int(x2 + pad_w))
        y2 = min(h, int(y2 + pad_h))
        
        # Validate dimensions
        plate_height = y2 - y1
        plate_width = x2 - x1
        
        if plate_height < min_height or plate_width < min_height:
            logger.warning(f"Plate too small: {plate_width}x{plate_height}")
            return None
        
        # Crop
        cropped = vehicle_image[y1:y2, x1:x2]
        
        # Additional quality checks
        if cropped.size == 0:
            return None
        
        # Check if too blurry
        if self._is_blurry(cropped):
            logger.warning("Plate image is too blurry")
            # Still return it, but log warning
        
        return cropped
    
    def _is_blurry(self, image: np.ndarray, threshold: float = 100.0) -> bool:
        """
        Check if image is blurry using Laplacian variance
        
        Args:
            image: Input image
            threshold: Variance threshold (lower = more blurry)
        
        Returns:
            True if blurry
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        variance = cv2.Laplacian(gray, cv2.CV_64F).var()
        return variance < threshold
    
    def detect_and_crop(
        self,
        vehicle_image: np.ndarray,
        padding: float = 0.05
    ) -> Tuple[Optional[np.ndarray], Optional[Dict]]:
        """
        Detect and crop plate in one call
        
        Args:
            vehicle_image: Vehicle ROI
            padding: Crop padding
        
        Returns:
            (cropped_plate, detection_info) or (None, None)
        """
        # Detect
        detection = self.detect(vehicle_image, return_best_only=True)
        
        if detection is None:
            return None, None
        
        # Crop
        plate_image = self.crop_plate(
            vehicle_image,
            detection["bbox"],
            padding=padding
        )
        
        if plate_image is None:
            return None, None
        
        return plate_image, detection
    
    def enhance_plate_image(self, plate_image: np.ndarray) -> np.ndarray:
        """
        Enhance plate image for better OCR
        
        Args:
            plate_image: Raw plate crop
        
        Returns:
            Enhanced image
        """
        # Convert to grayscale
        if len(plate_image.shape) == 3:
            gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = plate_image.copy()
        
        # Denoise
        denoised = cv2.fastNlMeansDenoising(gray)
        
        # Increase contrast (CLAHE)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(denoised)
        
        # Sharpen
        kernel = np.array([[-1, -1, -1],
                          [-1,  9, -1],
                          [-1, -1, -1]])
        sharpened = cv2.filter2D(enhanced, -1, kernel)
        
        return sharpened
    
    def draw_detection(
        self,
        vehicle_image: np.ndarray,
        detection: Dict,
        color: Tuple[int, int, int] = (0, 255, 255),
        thickness: int = 2
    ) -> np.ndarray:
        """
        Draw plate detection on vehicle image
        
        Args:
            vehicle_image: Vehicle ROI
            detection: Detection dict
            color: Box color (BGR)
            thickness: Line thickness
        
        Returns:
            Image with drawn box
        """
        output = vehicle_image.copy()
        
        bbox = detection["bbox"]
        conf = detection["confidence"]
        
        x1, y1, x2, y2 = map(int, bbox)
        
        # Draw box
        cv2.rectangle(output, (x1, y1), (x2, y2), color, thickness)
        
        # Draw label
        label = f"Plate {conf:.2f}"
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
            "iou_threshold": self.iou_threshold
        }


def create_plate_detector(
    model_path: str,
    use_tensorrt: bool = True,
    confidence_threshold: float = 0.4
) -> PlateDetector:
    """
    Factory function to create plate detector
    
    Args:
        model_path: Path to model file
        use_tensorrt: Use TensorRT optimization
        confidence_threshold: Detection threshold
    
    Returns:
        PlateDetector instance
    """
    return PlateDetector(
        model_path=model_path,
        use_tensorrt=use_tensorrt,
        confidence_threshold=confidence_threshold
    )