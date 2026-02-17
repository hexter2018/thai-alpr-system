"""
Custom Validators for Thai ALPR System
"""
import re
from typing import Optional
from datetime import datetime


def validate_thai_plate(plate: str) -> bool:
    """
    Validate Thai license plate format
    
    Args:
        plate: License plate text
    
    Returns:
        True if valid format
    """
    if not plate:
        return False
    
    patterns = [
        r'^[ก-ฮ]{2}\d{4}$',           # กก1234
        r'^[ก-ฮ]{2}-?\d{4}$',         # กก-1234
        r'^\d{1}[ก-ฮ]{2}\d{4}$',      # 1กก1234
        r'^\d{2}[ก-ฮ]{2}\d{4}$',      # 12กก1234
        r'^[ก-ฮ]{3}\d{4}$',           # กกก1234
        r'^\d{1,2}-\d{4}$',           # 30-1234 (motorcycle)
    ]
    
    return any(re.match(pattern, plate) for pattern in patterns)


def validate_rtsp_url(url: str) -> bool:
    """
    Validate RTSP URL format
    
    Args:
        url: RTSP URL
    
    Returns:
        True if valid
    """
    pattern = r'^rtsp://[\w\.-]+(:\d+)?(/.*)?$'
    return bool(re.match(pattern, url))


def validate_camera_id(camera_id: str) -> bool:
    """
    Validate camera ID format
    
    Args:
        camera_id: Camera identifier
    
    Returns:
        True if valid
    """
    # Alphanumeric, underscore, hyphen only
    pattern = r'^[a-zA-Z0-9_-]{1,50}$'
    return bool(re.match(pattern, camera_id))


def validate_tracking_id(tracking_id: str) -> bool:
    """
    Validate tracking ID format
    
    Args:
        tracking_id: Tracking ID
    
    Returns:
        True if valid
    """
    # Can be numeric or alphanumeric
    pattern = r'^[a-zA-Z0-9_-]{1,100}$'
    return bool(re.match(pattern, tracking_id))


def validate_confidence(confidence: float) -> bool:
    """
    Validate confidence score
    
    Args:
        confidence: Confidence value
    
    Returns:
        True if valid (0.0 - 1.0)
    """
    return 0.0 <= confidence <= 1.0


def validate_polygon_zone(polygon: list) -> bool:
    """
    Validate polygon zone format
    
    Args:
        polygon: List of {"x": int, "y": int} points
    
    Returns:
        True if valid
    """
    if not isinstance(polygon, list):
        return False
    
    if len(polygon) < 3:  # Minimum 3 points for polygon
        return False
    
    for point in polygon:
        if not isinstance(point, dict):
            return False
        
        if "x" not in point or "y" not in point:
            return False
        
        if not isinstance(point["x"], (int, float)):
            return False
        
        if not isinstance(point["y"], (int, float)):
            return False
    
    return True


def validate_date_range(start_date: Optional[datetime], end_date: Optional[datetime]) -> bool:
    """
    Validate date range
    
    Args:
        start_date: Start date
        end_date: End date
    
    Returns:
        True if valid
    """
    if start_date and end_date:
        return start_date <= end_date
    return True


def validate_image_file(filename: str) -> bool:
    """
    Validate image file extension
    
    Args:
        filename: Filename
    
    Returns:
        True if valid image file
    """
    allowed_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.webp']
    return any(filename.lower().endswith(ext) for ext in allowed_extensions)


def sanitize_filename(filename: str) -> str:
    """
    Sanitize filename for safe storage
    
    Args:
        filename: Original filename
    
    Returns:
        Sanitized filename
    """
    # Remove path separators and special chars
    sanitized = re.sub(r'[^\w\.-]', '_', filename)
    
    # Limit length
    if len(sanitized) > 255:
        name, ext = sanitized.rsplit('.', 1) if '.' in sanitized else (sanitized, '')
        sanitized = name[:250] + ('.' + ext if ext else '')
    
    return sanitized


def validate_pagination(skip: int, limit: int, max_limit: int = 100) -> bool:
    """
    Validate pagination parameters
    
    Args:
        skip: Offset
        limit: Page size
        max_limit: Maximum allowed limit
    
    Returns:
        True if valid
    """
    return skip >= 0 and 0 < limit <= max_limit