"""
Zone Manager for Polygon Detection Areas
"""
import logging
from typing import List, Dict, Tuple, Optional
import cv2
import numpy as np

logger = logging.getLogger(__name__)


class ZoneManager:
    """Manage polygon detection zones"""
    
    def __init__(self, polygon_points: Optional[List[Dict]] = None):
        """
        Initialize zone manager
        
        Args:
            polygon_points: List of {"x": int, "y": int} points
        """
        self.polygon_points = polygon_points or []
        self.polygon_array = self._convert_to_array()
    
    def _convert_to_array(self) -> Optional[np.ndarray]:
        """Convert polygon points to numpy array"""
        if not self.polygon_points:
            return None
        
        points = [(p["x"], p["y"]) for p in self.polygon_points]
        return np.array(points, dtype=np.int32)
    
    def is_point_in_zone(self, point: Tuple[int, int]) -> bool:
        """
        Check if point is inside polygon zone using ray casting
        
        Args:
            point: (x, y) coordinates
        
        Returns:
            True if point is inside zone
        """
        if not self.polygon_points:
            return True  # No zone defined = all points valid
        
        x, y = point
        n = len(self.polygon_points)
        inside = False
        
        p1x, p1y = self.polygon_points[0]["x"], self.polygon_points[0]["y"]
        
        for i in range(1, n + 1):
            p2x, p2y = self.polygon_points[i % n]["x"], self.polygon_points[i % n]["y"]
            
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            
            p1x, p1y = p2x, p2y
        
        return inside
    
    def is_bbox_in_zone(self, bbox: List[float], check_center: bool = True) -> bool:
        """
        Check if bounding box is in zone
        
        Args:
            bbox: [x1, y1, x2, y2]
            check_center: Check center point only or any corner
        
        Returns:
            True if bbox is in zone
        """
        if not self.polygon_points:
            return True
        
        if check_center:
            # Check center point
            center_x = int((bbox[0] + bbox[2]) / 2)
            center_y = int((bbox[1] + bbox[3]) / 2)
            return self.is_point_in_zone((center_x, center_y))
        else:
            # Check any corner
            corners = [
                (int(bbox[0]), int(bbox[1])),  # Top-left
                (int(bbox[2]), int(bbox[1])),  # Top-right
                (int(bbox[2]), int(bbox[3])),  # Bottom-right
                (int(bbox[0]), int(bbox[3]))   # Bottom-left
            ]
            return any(self.is_point_in_zone(corner) for corner in corners)
    
    def draw_zone(self, frame: np.ndarray, color: Tuple[int, int, int] = (0, 255, 0), 
                  thickness: int = 2, alpha: float = 0.3) -> np.ndarray:
        """
        Draw zone on frame
        
        Args:
            frame: Input frame
            color: Zone color (BGR)
            thickness: Line thickness
            alpha: Fill transparency
        
        Returns:
            Frame with drawn zone
        """
        if not self.polygon_array is not None:
            return frame
        
        output = frame.copy()
        
        # Draw filled polygon
        overlay = output.copy()
        cv2.fillPoly(overlay, [self.polygon_array], color)
        cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)
        
        # Draw border
        cv2.polylines(output, [self.polygon_array], isClosed=True, 
                     color=color, thickness=thickness)
        
        return output
    
    def get_zone_bounds(self) -> Optional[Tuple[int, int, int, int]]:
        """Get zone bounding box (x1, y1, x2, y2)"""
        if not self.polygon_points:
            return None
        
        xs = [p["x"] for p in self.polygon_points]
        ys = [p["y"] for p in self.polygon_points]
        
        return (min(xs), min(ys), max(xs), max(ys))
    
    def update_zone(self, polygon_points: List[Dict]):
        """Update zone polygon"""
        self.polygon_points = polygon_points
        self.polygon_array = self._convert_to_array()