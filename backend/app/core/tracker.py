"""
Object Tracker for Vehicle Tracking
Implements ByteTrack/BoT-SORT for persistent vehicle ID assignment
"""
import logging
from typing import List, Dict, Optional
import numpy as np

logger = logging.getLogger(__name__)


class VehicleTracker:
    """
    Multi-object tracker for vehicles using ByteTrack algorithm
    Maintains persistent tracking IDs across frames
    """
    
    def __init__(
        self,
        track_thresh: float = 0.5,
        track_buffer: int = 30,
        match_thresh: float = 0.8,
        min_box_area: float = 100
    ):
        """
        Initialize vehicle tracker
        
        Args:
            track_thresh: Detection confidence threshold for tracking
            track_buffer: Number of frames to keep lost tracks
            track_buffer: Matching threshold for associating detections
            min_box_area: Minimum bounding box area
        """
        self.track_thresh = track_thresh
        self.track_buffer = track_buffer
        self.match_thresh = match_thresh
        self.min_box_area = min_box_area
        
        try:
            # Try to import ByteTrack from supervision or boxmot
            self._init_tracker()
        except ImportError:
            logger.warning("ByteTrack not available, using fallback tracker")
            self.tracker = None
            self.use_fallback = True
            self.tracked_objects = {}
            self.next_id = 1
    
    def _init_tracker(self):
        """Initialize ByteTrack tracker"""
        try:
            # Option 1: Use supervision library (recommended)
            from supervision import ByteTrack
            
            self.tracker = ByteTrack(
                track_activation_threshold=self.track_thresh,
                lost_track_buffer=self.track_buffer,
                minimum_matching_threshold=self.match_thresh,
                minimum_consecutive_frames=1
            )
            self.use_fallback = False
            logger.info("ByteTrack initialized successfully")
            
        except ImportError:
            try:
                # Option 2: Use boxmot library
                from boxmot import BYTETracker
                
                self.tracker = BYTETracker(
                    track_thresh=self.track_thresh,
                    track_buffer=self.track_buffer,
                    match_thresh=self.match_thresh
                )
                self.use_fallback = False
                logger.info("BYTETracker (boxmot) initialized successfully")
                
            except ImportError:
                raise ImportError(
                    "No tracking library found. Install: "
                    "pip install supervision or pip install boxmot"
                )
    
    def update(
        self,
        detections: List[Dict],
        frame_id: int
    ) -> List[Dict]:
        """
        Update tracker with new detections
        
        Args:
            detections: List of detection dicts with bbox and confidence
            frame_id: Current frame number
        
        Returns:
            List of tracked objects with tracking_id
        """
        if self.use_fallback:
            return self._fallback_update(detections, frame_id)
        
        try:
            # Convert detections to format expected by tracker
            if not detections:
                # Update with no detections
                tracks = self.tracker.update_with_detections([])
                return []
            
            # Prepare detection data
            detection_array = self._prepare_detections(detections)
            
            # Update tracker
            if hasattr(self.tracker, 'update'):
                # boxmot style
                tracks = self.tracker.update(detection_array, None)
            else:
                # supervision style - needs Detections object
                from supervision import Detections
                
                det_obj = Detections(
                    xyxy=detection_array[:, :4],
                    confidence=detection_array[:, 4],
                    class_id=np.zeros(len(detection_array), dtype=int)
                )
                tracks = self.tracker.update_with_detections(det_obj)
            
            # Format tracked objects
            tracked = self._format_tracks(tracks, detections)
            
            return tracked
            
        except Exception as e:
            logger.error(f"Tracking update failed: {e}, using fallback")
            return self._fallback_update(detections, frame_id)
    
    def _prepare_detections(self, detections: List[Dict]) -> np.ndarray:
        """
        Convert detection list to numpy array
        Format: [x1, y1, x2, y2, confidence, class_id]
        """
        det_array = []
        
        for det in detections:
            bbox = det["bbox"]
            conf = det["confidence"]
            cls = det.get("class_id", 0)
            
            det_array.append([*bbox, conf, cls])
        
        return np.array(det_array)
    
    def _format_tracks(
        self,
        tracks: np.ndarray,
        original_detections: List[Dict]
    ) -> List[Dict]:
        """
        Format tracked results
        
        Args:
            tracks: Tracking results from tracker
            original_detections: Original detection list
        
        Returns:
            Formatted tracked objects
        """
        tracked_objects = []
        
        if tracks is None or len(tracks) == 0:
            return tracked_objects
        
        # Handle different tracker output formats
        if hasattr(tracks, 'tracker_id'):
            # supervision format
            for i in range(len(tracks)):
                obj = {
                    "tracking_id": int(tracks.tracker_id[i]),
                    "bbox": tracks.xyxy[i].tolist(),
                    "confidence": float(tracks.confidence[i]),
                    "class_id": int(tracks.class_id[i]) if hasattr(tracks, 'class_id') else 0,
                    "class_name": original_detections[i].get("class_name", "vehicle") if i < len(original_detections) else "vehicle"
                }
                tracked_objects.append(obj)
        else:
            # boxmot format: [x1, y1, x2, y2, track_id, conf, class_id, ...]
            for track in tracks:
                obj = {
                    "tracking_id": int(track[4]),
                    "bbox": track[:4].tolist(),
                    "confidence": float(track[5]),
                    "class_id": int(track[6]) if len(track) > 6 else 0,
                    "class_name": "vehicle"
                }
                # Match with original detection to get class name
                for det in original_detections:
                    if self._iou(obj["bbox"], det["bbox"]) > 0.5:
                        obj["class_name"] = det.get("class_name", "vehicle")
                        break
                
                tracked_objects.append(obj)
        
        return tracked_objects
    
    def _fallback_update(
        self,
        detections: List[Dict],
        frame_id: int
    ) -> List[Dict]:
        """
        Simple fallback tracker using IoU matching
        Not as robust as ByteTrack but works without dependencies
        """
        tracked_objects = []
        matched_ids = set()
        
        # Match with existing tracks
        for det in detections:
            det_bbox = det["bbox"]
            best_iou = 0.0
            best_id = None
            
            for track_id, track_info in self.tracked_objects.items():
                iou = self._iou(det_bbox, track_info["bbox"])
                if iou > best_iou and iou > 0.3:
                    best_iou = iou
                    best_id = track_id
            
            if best_id is not None:
                # Update existing track
                tracking_id = best_id
                matched_ids.add(best_id)
            else:
                # New track
                tracking_id = self.next_id
                self.next_id += 1
            
            # Update track info
            self.tracked_objects[tracking_id] = {
                "bbox": det_bbox,
                "last_frame": frame_id,
                "confidence": det["confidence"]
            }
            
            tracked_objects.append({
                "tracking_id": tracking_id,
                "bbox": det_bbox,
                "confidence": det["confidence"],
                "class_id": det.get("class_id", 0),
                "class_name": det.get("class_name", "vehicle")
            })
        
        # Remove lost tracks
        lost_ids = []
        for track_id, track_info in self.tracked_objects.items():
            if track_id not in matched_ids:
                if frame_id - track_info["last_frame"] > self.track_buffer:
                    lost_ids.append(track_id)
        
        for track_id in lost_ids:
            del self.tracked_objects[track_id]
        
        return tracked_objects
    
    def _iou(self, bbox1: List[float], bbox2: List[float]) -> float:
        """Calculate IoU between two bounding boxes"""
        x1_min, y1_min, x1_max, y1_max = bbox1
        x2_min, y2_min, x2_max, y2_max = bbox2
        
        # Intersection
        inter_xmin = max(x1_min, x2_min)
        inter_ymin = max(y1_min, y2_min)
        inter_xmax = min(x1_max, x2_max)
        inter_ymax = min(y1_max, y2_max)
        
        inter_area = max(0, inter_xmax - inter_xmin) * max(0, inter_ymax - inter_ymin)
        
        # Union
        bbox1_area = (x1_max - x1_min) * (y1_max - y1_min)
        bbox2_area = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = bbox1_area + bbox2_area - inter_area
        
        if union_area == 0:
            return 0.0
        
        return inter_area / union_area
    
    def reset(self):
        """Reset tracker state"""
        if hasattr(self, 'tracker') and self.tracker:
            if hasattr(self.tracker, 'reset'):
                self.tracker.reset()
        
        if self.use_fallback:
            self.tracked_objects = {}
            self.next_id = 1
        
        logger.info("Tracker reset")
    
    def get_track_count(self) -> int:
        """Get number of active tracks"""
        if self.use_fallback:
            return len(self.tracked_objects)
        return 0  # Difficult to get from ByteTrack


def create_tracker(
    track_thresh: float = 0.5,
    track_buffer: int = 30
) -> VehicleTracker:
    """
    Factory function to create vehicle tracker
    
    Args:
        track_thresh: Tracking confidence threshold
        track_buffer: Frame buffer for lost tracks
    
    Returns:
        VehicleTracker instance
    """
    return VehicleTracker(
        track_thresh=track_thresh,
        track_buffer=track_buffer
    )