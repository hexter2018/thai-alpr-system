"""
Main AI Core - Thai ALPR Processing Pipeline
Orchestrates vehicle detection, tracking, zone checking, plate detection, and OCR
"""
import logging
import asyncio
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from pathlib import Path
import numpy as np
import cv2

from .vehicle_detector import VehicleDetector
from .plate_detector import PlateDetector
from .ocr_engine import ThaiOCREngine
from .tracker import VehicleTracker
from ..utils.thai_provinces import normalize_province_name, validate_plate_format

logger = logging.getLogger(__name__)


class ThaiALPRCore:
    """
    Main ALPR processing pipeline
    Coordinates all AI components for end-to-end license plate recognition
    """
    
    def __init__(
        self,
        vehicle_model_path: str,
        plate_model_path: str,
        use_tensorrt: bool = True,
        ocr_engine: str = "paddleocr",
        high_confidence_threshold: float = 0.95,
        storage_path: str = "./storage",
        device: str = "cuda:0"
    ):
        """
        Initialize ALPR core with all AI components
        
        Args:
            vehicle_model_path: Path to vehicles.pt or .engine
            plate_model_path: Path to best.pt or .engine
            use_tensorrt: Use TensorRT optimization
            ocr_engine: OCR engine ("paddleocr" or "tesseract")
            high_confidence_threshold: Threshold for auto-approval
            storage_path: Path for image storage
            device: CUDA device
        """
        self.high_confidence_threshold = high_confidence_threshold
        self.storage_path = Path(storage_path)
        self.device = device
        
        # Create storage directories
        self.images_path = self.storage_path / "images"
        self.plates_path = self.storage_path / "plates"
        self.dataset_path = self.storage_path / "dataset" / "train"
        
        for path in [self.images_path, self.plates_path, self.dataset_path]:
            path.mkdir(parents=True, exist_ok=True)
        
        # Initialize AI components
        logger.info("Initializing ALPR Core components...")
        
        self.vehicle_detector = VehicleDetector(
            model_path=vehicle_model_path,
            use_tensorrt=use_tensorrt,
            device=device
        )
        
        self.plate_detector = PlateDetector(
            model_path=plate_model_path,
            use_tensorrt=use_tensorrt,
            device=device
        )
        
        self.ocr_engine = ThaiOCREngine(
            engine=ocr_engine,
            use_gpu=True,
            lang="th"
        )
        
        self.tracker = VehicleTracker(
            track_thresh=0.5,
            track_buffer=30
        )
        
        # Processing statistics
        self.stats = {
            "total_processed": 0,
            "successful_detections": 0,
            "failed_detections": 0,
            "high_confidence": 0,
            "low_confidence": 0
        }
        
        logger.info("ALPR Core initialized successfully")
    
    async def process_frame(
        self,
        frame: np.ndarray,
        frame_id: int,
        camera_id: str,
        polygon_zone: Optional[List[Dict]] = None,
        redis_client = None
    ) -> List[Dict]:
        """
        Process a single frame through the complete ALPR pipeline
        
        Args:
            frame: Input frame (BGR)
            frame_id: Frame number
            camera_id: Camera identifier
            polygon_zone: Optional polygon zone coordinates
            redis_client: Redis client for deduplication
        
        Returns:
            List of detection results
        """
        start_time = datetime.now()
        results = []
        
        try:
            # Step 1: Detect vehicles
            vehicle_detections = self.vehicle_detector.detect(frame)
            
            if not vehicle_detections:
                return results
            
            # Step 2: Track vehicles
            tracked_vehicles = self.tracker.update(vehicle_detections, frame_id)
            
            # Step 3: Process each tracked vehicle
            for vehicle in tracked_vehicles:
                result = await self._process_vehicle(
                    frame=frame,
                    vehicle=vehicle,
                    frame_id=frame_id,
                    camera_id=camera_id,
                    polygon_zone=polygon_zone,
                    redis_client=redis_client
                )
                
                if result:
                    results.append(result)
            
            # Update statistics
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            self.stats["total_processed"] += 1
            
            logger.debug(
                f"Frame {frame_id} processed: {len(results)} plates detected "
                f"in {processing_time:.2f}ms"
            )
            
            return results
            
        except Exception as e:
            logger.error(f"Frame processing failed: {e}", exc_info=True)
            return results
    
    async def _process_vehicle(
        self,
        frame: np.ndarray,
        vehicle: Dict,
        frame_id: int,
        camera_id: str,
        polygon_zone: Optional[List[Dict]],
        redis_client
    ) -> Optional[Dict]:
        """
        Process a single vehicle detection
        
        Pipeline: Zone Check → Dedup → Plate Detect → OCR → Save
        """
        tracking_id = vehicle["tracking_id"]
        vehicle_bbox = vehicle["bbox"]
        vehicle_center = self._calculate_center(vehicle_bbox)
        
        # Step 1: Zone check
        in_zone = True
        if polygon_zone:
            in_zone = self._point_in_polygon(vehicle_center, polygon_zone)
        
        if not in_zone:
            logger.debug(f"Vehicle {tracking_id} outside zone, skipping")
            return None
        
        # Step 2: Deduplication check
        if redis_client:
            is_duplicate = await self._check_deduplication(
                redis_client,
                tracking_id,
                camera_id
            )
            if is_duplicate:
                logger.debug(f"Vehicle {tracking_id} already processed (dedup)")
                return None
        
        # Step 3: Crop vehicle ROI
        vehicle_roi = self.vehicle_detector.crop_vehicle_roi(frame, vehicle_bbox)
        
        if vehicle_roi is None or vehicle_roi.size == 0:
            logger.warning(f"Failed to crop vehicle ROI for {tracking_id}")
            return None
        
        # Step 4: Detect license plate
        plate_image, plate_detection = self.plate_detector.detect_and_crop(vehicle_roi)
        
        if plate_image is None:
            logger.debug(f"No plate detected for vehicle {tracking_id}")
            self.stats["failed_detections"] += 1
            return None
        
        # Step 5: OCR
        ocr_result = self.ocr_engine.recognize_plate(plate_image, preprocess=True)
        
        if not ocr_result.get("success") or not ocr_result.get("text"):
            logger.warning(f"OCR failed for vehicle {tracking_id}")
            self.stats["failed_detections"] += 1
            return None
        
        # Step 6: Post-process and validate
        detected_plate = ocr_result["text"]
        detected_province = ocr_result.get("province")
        confidence = ocr_result["confidence"]
        
        # Normalize province
        if detected_province:
            detected_province = normalize_province_name(detected_province)
        
        # Validate plate format
        is_valid_format = validate_plate_format(detected_plate)
        
        # Determine status
        status = self._determine_status(confidence, is_valid_format)
        
        # Update statistics
        self.stats["successful_detections"] += 1
        if confidence > self.high_confidence_threshold:
            self.stats["high_confidence"] += 1
        else:
            self.stats["low_confidence"] += 1
        
        # Step 7: Save images
        timestamp = datetime.now()
        full_image_path = await self._save_full_image(frame, tracking_id, timestamp)
        plate_crop_path = await self._save_plate_crop(plate_image, tracking_id, timestamp)
        
        # Step 8: Mark as processed in Redis
        if redis_client:
            await self._mark_processed(redis_client, tracking_id, camera_id)
        
        # Step 9: Build result
        result = {
            "tracking_id": str(tracking_id),
            "frame_id": frame_id,
            "camera_id": camera_id,
            "timestamp": timestamp,
            
            # Vehicle info
            "vehicle_bbox": vehicle_bbox,
            "vehicle_center": vehicle_center,
            "vehicle_type": vehicle.get("class_name", "vehicle"),
            "vehicle_confidence": vehicle.get("confidence", 0.0),
            
            # Plate info
            "plate_bbox": plate_detection["bbox"],
            "plate_confidence": plate_detection["confidence"],
            
            # OCR results
            "detected_plate": detected_plate,
            "detected_province": detected_province,
            "ocr_confidence": confidence,
            "is_valid_format": is_valid_format,
            "status": status,
            
            # Paths
            "full_image_path": str(full_image_path),
            "plate_crop_path": str(plate_crop_path),
            
            # Raw OCR output
            "ocr_raw": ocr_result.get("raw_output"),
            
            # Model versions
            "model_versions": {
                "vehicle_model": self.vehicle_detector.model_path.name,
                "plate_model": self.plate_detector.model_path.name,
                "ocr_engine": self.ocr_engine.engine_type
            }
        }
        
        return result
    
    def _calculate_center(self, bbox: List[float]) -> Tuple[int, int]:
        """Calculate bounding box center"""
        x1, y1, x2, y2 = bbox
        center_x = int((x1 + x2) / 2)
        center_y = int((y1 + y2) / 2)
        return (center_x, center_y)
    
    def _point_in_polygon(
        self,
        point: Tuple[int, int],
        polygon: List[Dict]
    ) -> bool:
        """
        Check if point is inside polygon using ray casting algorithm
        
        Args:
            point: (x, y) coordinates
            polygon: List of {"x": int, "y": int} points
        
        Returns:
            True if point is inside polygon
        """
        x, y = point
        n = len(polygon)
        inside = False
        
        p1x, p1y = polygon[0]["x"], polygon[0]["y"]
        
        for i in range(1, n + 1):
            p2x, p2y = polygon[i % n]["x"], polygon[i % n]["y"]
            
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            
            p1x, p1y = p2x, p2y
        
        return inside
    
    async def _check_deduplication(
        self,
        redis_client,
        tracking_id: int,
        camera_id: str,
        ttl: int = 60
    ) -> bool:
        """
        Check if vehicle was recently processed
        
        Args:
            redis_client: Redis client
            tracking_id: Vehicle tracking ID
            camera_id: Camera identifier
            ttl: Time to live in seconds
        
        Returns:
            True if already processed
        """
        key = f"processed:{camera_id}:{tracking_id}"
        
        try:
            exists = await redis_client.exists(key)
            return bool(exists)
        except Exception as e:
            logger.error(f"Redis dedup check failed: {e}")
            return False
    
    async def _mark_processed(
        self,
        redis_client,
        tracking_id: int,
        camera_id: str,
        ttl: int = 60
    ):
        """Mark vehicle as processed in Redis"""
        key = f"processed:{camera_id}:{tracking_id}"
        
        try:
            await redis_client.setex(key, ttl, "1")
        except Exception as e:
            logger.error(f"Failed to mark as processed: {e}")
    
    def _determine_status(self, confidence: float, is_valid_format: bool) -> str:
        """
        Determine processing status based on confidence
        
        Returns:
            "ALPR_AUTO" or "PENDING_VERIFY"
        """
        if confidence > self.high_confidence_threshold and is_valid_format:
            return "ALPR_AUTO"
        return "PENDING_VERIFY"
    
    async def _save_full_image(
        self,
        frame: np.ndarray,
        tracking_id: int,
        timestamp: datetime
    ) -> Path:
        """Save full frame image"""
        filename = f"{timestamp.strftime('%Y%m%d_%H%M%S')}_{tracking_id}.jpg"
        filepath = self.images_path / filename
        
        try:
            # Save asynchronously
            await asyncio.to_thread(cv2.imwrite, str(filepath), frame)
            return filepath
        except Exception as e:
            logger.error(f"Failed to save full image: {e}")
            return None
    
    async def _save_plate_crop(
        self,
        plate_image: np.ndarray,
        tracking_id: int,
        timestamp: datetime
    ) -> Path:
        """Save cropped plate image"""
        filename = f"{timestamp.strftime('%Y%m%d_%H%M%S')}_{tracking_id}_plate.jpg"
        filepath = self.plates_path / filename
        
        try:
            await asyncio.to_thread(cv2.imwrite, str(filepath), plate_image)
            return filepath
        except Exception as e:
            logger.error(f"Failed to save plate crop: {e}")
            return None
    
    async def save_for_training(
        self,
        plate_image_path: str,
        corrected_text: str,
        corrected_province: str
    ) -> bool:
        """
        Save corrected plate to training dataset
        Used for active learning
        
        Args:
            plate_image_path: Path to original plate crop
            corrected_text: Human-corrected plate text
            corrected_province: Human-corrected province
        
        Returns:
            Success status
        """
        try:
            # Copy image to dataset
            src_path = Path(plate_image_path)
            dst_image_path = self.dataset_path / "images" / src_path.name
            dst_image_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Copy image
            import shutil
            shutil.copy2(src_path, dst_image_path)
            
            # Create label file (YOLO format or text format)
            label_filename = src_path.stem + ".txt"
            dst_label_path = self.dataset_path / "labels" / label_filename
            dst_label_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Write label
            with open(dst_label_path, 'w', encoding='utf-8') as f:
                f.write(f"{corrected_text} {corrected_province}\n")
            
            logger.info(f"Saved training sample: {corrected_text} {corrected_province}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save training data: {e}")
            return False
    
    def get_statistics(self) -> Dict:
        """Get processing statistics"""
        total = self.stats["total_processed"]
        
        return {
            **self.stats,
            "success_rate": (
                self.stats["successful_detections"] / total * 100
                if total > 0 else 0
            ),
            "high_confidence_rate": (
                self.stats["high_confidence"] / self.stats["successful_detections"] * 100
                if self.stats["successful_detections"] > 0 else 0
            )
        }
    
    def reset_statistics(self):
        """Reset processing statistics"""
        self.stats = {
            "total_processed": 0,
            "successful_detections": 0,
            "failed_detections": 0,
            "high_confidence": 0,
            "low_confidence": 0
        }
    
    def draw_results(
        self,
        frame: np.ndarray,
        results: List[Dict]
    ) -> np.ndarray:
        """
        Draw detection results on frame for visualization
        
        Args:
            frame: Input frame
            results: Detection results
        
        Returns:
            Frame with drawn annotations
        """
        output = frame.copy()
        
        for result in results:
            # Draw vehicle bbox
            vehicle_bbox = result["vehicle_bbox"]
            x1, y1, x2, y2 = map(int, vehicle_bbox)
            
            # Color based on status
            if result["status"] == "ALPR_AUTO":
                color = (0, 255, 0)  # Green for high confidence
            else:
                color = (0, 165, 255)  # Orange for pending
            
            cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
            
            # Draw plate text
            plate_text = result["detected_plate"]
            if result["detected_province"]:
                plate_text += f" {result['detected_province']}"
            
            label = f"{plate_text} ({result['ocr_confidence']:.2f})"
            
            # Background for text
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(
                output,
                (x1, y1 - label_size[1] - 10),
                (x1 + label_size[0], y1),
                color,
                -1
            )
            
            # Text
            cv2.putText(
                output,
                label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                2
            )
        
        return output


def create_alpr_core(
    vehicle_model_path: str,
    plate_model_path: str,
    **kwargs
) -> ThaiALPRCore:
    """
    Factory function to create ALPR core instance
    
    Args:
        vehicle_model_path: Path to vehicle detection model
        plate_model_path: Path to plate detection model
        **kwargs: Additional configuration
    
    Returns:
        ThaiALPRCore instance
    """
    return ThaiALPRCore(
        vehicle_model_path=vehicle_model_path,
        plate_model_path=plate_model_path,
        **kwargs
    )