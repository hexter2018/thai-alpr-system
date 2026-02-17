"""
SQLAlchemy ORM Models for Thai ALPR System
Handles vehicle master data and access logs with complete audit trail
"""
from datetime import datetime
from typing import Optional
from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean, Text, Enum as SQLEnum, ForeignKey, Index
from sqlalchemy.orm import relationship, declarative_base
from sqlalchemy.dialects.postgresql import JSONB
import enum

Base = declarative_base()


class ProcessStatus(str, enum.Enum):
    """Processing status for license plate detection"""
    ALPR_AUTO = "ALPR_AUTO"           # Auto-detected with high confidence (>0.95)
    PENDING_VERIFY = "PENDING_VERIFY" # Low confidence, needs manual verification
    MLPR = "MLPR"                      # Manually corrected by operator
    REJECTED = "REJECTED"              # Invalid detection, rejected by operator


class VehicleType(str, enum.Enum):
    """Vehicle categories"""
    CAR = "car"
    TRUCK = "truck"
    MOTORCYCLE = "motorcycle"
    BUS = "bus"
    VAN = "van"
    UNKNOWN = "unknown"


class MasterVehicle(Base):
    """
    Master table for verified/registered vehicles
    Stores ground truth for known vehicles
    """
    __tablename__ = "master_vehicles"

    id = Column(Integer, primary_key=True, index=True)
    license_plate = Column(String(20), unique=True, nullable=False, index=True)
    province = Column(String(100), nullable=False)  # Thai province name
    
    # Vehicle details
    vehicle_type = Column(SQLEnum(VehicleType), default=VehicleType.UNKNOWN)
    brand = Column(String(100))
    model = Column(String(100))
    color = Column(String(50))
    year = Column(Integer)
    
    # Owner information
    owner_name = Column(String(255))
    owner_phone = Column(String(20))
    owner_address = Column(Text)
    
    # Access control
    is_authorized = Column(Boolean, default=True)
    is_blacklisted = Column(Boolean, default=False)
    notes = Column(Text)
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    created_by = Column(String(100))
    
    # Relationships
    access_logs = relationship("AccessLog", back_populates="vehicle", cascade="all, delete-orphan")

    # Indexes
    __table_args__ = (
        Index('idx_plate_province', 'license_plate', 'province'),
        Index('idx_authorized', 'is_authorized'),
    )

    def __repr__(self):
        return f"<MasterVehicle(id={self.id}, plate={self.license_plate}, province={self.province})>"


class AccessLog(Base):
    """
    Access logs for every vehicle detection event
    Complete audit trail with AI confidence and manual corrections
    """
    __tablename__ = "access_logs"

    id = Column(Integer, primary_key=True, index=True)
    
    # Detection metadata
    tracking_id = Column(String(50), index=True)  # ByteTrack/BoT-SORT tracking ID
    detection_timestamp = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    camera_id = Column(String(100))  # RTSP stream identifier
    
    # Image paths
    full_image_path = Column(String(500))      # Full frame image
    plate_crop_path = Column(String(500))      # Cropped license plate image
    
    # Detection results
    detected_plate = Column(String(20), index=True)  # Raw OCR output
    detected_province = Column(String(100))
    confidence_score = Column(Float)  # OCR confidence (0.0 - 1.0)
    
    # Vehicle detection info
    vehicle_type = Column(SQLEnum(VehicleType), default=VehicleType.UNKNOWN)
    vehicle_bbox = Column(JSONB)  # {"x1": 100, "y1": 200, "x2": 300, "y2": 400}
    plate_bbox = Column(JSONB)    # License plate bounding box within vehicle ROI
    
    # Processing status
    status = Column(SQLEnum(ProcessStatus), default=ProcessStatus.PENDING_VERIFY, index=True)
    
    # Manual correction (MLPR - Manual License Plate Recognition)
    corrected_plate = Column(String(20))
    corrected_province = Column(String(100))
    verified_at = Column(DateTime)
    verified_by = Column(String(100))  # Username/operator ID
    verification_notes = Column(Text)
    
    # Active learning flag
    added_to_training = Column(Boolean, default=False)  # Flag if added to dataset
    
    # Foreign key to master vehicles (optional, if plate is registered)
    vehicle_id = Column(Integer, ForeignKey("master_vehicles.id"), nullable=True, index=True)
    vehicle = relationship("MasterVehicle", back_populates="access_logs")
    
    # Additional metadata
    processing_time_ms = Column(Float)  # Total processing time
    ocr_raw_output = Column(JSONB)      # Complete OCR response for debugging
    model_versions = Column(JSONB)       # {"vehicle_model": "vehicles.pt", "plate_model": "best.pt"}
    
    # Indexes for common queries
    __table_args__ = (
        Index('idx_status_timestamp', 'status', 'detection_timestamp'),
        Index('idx_tracking_timestamp', 'tracking_id', 'detection_timestamp'),
        Index('idx_camera_timestamp', 'camera_id', 'detection_timestamp'),
    )

    def __repr__(self):
        plate = self.corrected_plate or self.detected_plate
        return f"<AccessLog(id={self.id}, plate={plate}, status={self.status})>"

    @property
    def final_plate(self) -> str:
        """Return the final plate (corrected or detected)"""
        return self.corrected_plate if self.corrected_plate else self.detected_plate

    @property
    def final_province(self) -> str:
        """Return the final province (corrected or detected)"""
        return self.corrected_province if self.corrected_province else self.detected_province

    @property
    def is_verified(self) -> bool:
        """Check if log has been manually verified"""
        return self.status in [ProcessStatus.MLPR, ProcessStatus.ALPR_AUTO, ProcessStatus.REJECTED]


class SystemMetric(Base):
    """
    System performance metrics for dashboard KPIs
    Stores aggregated statistics for quick dashboard loading
    """
    __tablename__ = "system_metrics"

    id = Column(Integer, primary_key=True, index=True)
    metric_date = Column(DateTime, default=datetime.utcnow, index=True)
    
    # Daily statistics
    total_detections = Column(Integer, default=0)
    alpr_auto_count = Column(Integer, default=0)
    mlpr_count = Column(Integer, default=0)
    pending_count = Column(Integer, default=0)
    rejected_count = Column(Integer, default=0)
    
    # Accuracy metrics
    average_confidence = Column(Float)
    high_confidence_percentage = Column(Float)  # % with confidence > 0.95
    
    # Performance metrics
    avg_processing_time_ms = Column(Float)
    unique_vehicles_detected = Column(Integer, default=0)
    
    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        Index('idx_metric_date', 'metric_date'),
    )


class CameraConfig(Base):
    """
    Camera/RTSP stream configuration
    Stores polygon zones and camera settings
    """
    __tablename__ = "camera_configs"

    id = Column(Integer, primary_key=True, index=True)
    camera_id = Column(String(100), unique=True, nullable=False, index=True)
    camera_name = Column(String(255), nullable=False)
    
    # RTSP connection
    rtsp_url = Column(String(500), nullable=False)
    is_active = Column(Boolean, default=True)
    
    # Detection zone (polygon coordinates)
    polygon_zone = Column(JSONB)  # [{"x": 100, "y": 200}, {"x": 300, "y": 400}, ...]
    
    # Processing settings
    frame_skip = Column(Integer, default=2)  # Process every N frames
    min_confidence_vehicle = Column(Float, default=0.5)
    min_confidence_plate = Column(Float, default=0.4)
    
    # Deduplication settings
    dedup_window_seconds = Column(Integer, default=60)  # Redis TTL for tracking IDs
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def __repr__(self):
        return f"<CameraConfig(id={self.camera_id}, name={self.camera_name})>"