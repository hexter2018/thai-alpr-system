"""
Pydantic Schemas for API Request/Response Validation
"""
from datetime import datetime
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, validator
from enum import Enum


# Enums
class ProcessStatusEnum(str, Enum):
    ALPR_AUTO = "ALPR_AUTO"
    PENDING_VERIFY = "PENDING_VERIFY"
    MLPR = "MLPR"
    REJECTED = "REJECTED"


class VehicleTypeEnum(str, Enum):
    CAR = "car"
    TRUCK = "truck"
    MOTORCYCLE = "motorcycle"
    BUS = "bus"
    VAN = "van"
    UNKNOWN = "unknown"


# Vehicle Schemas
class VehicleBase(BaseModel):
    license_plate: str = Field(..., max_length=20)
    province: str = Field(..., max_length=100)
    vehicle_type: VehicleTypeEnum = VehicleTypeEnum.UNKNOWN
    brand: Optional[str] = Field(None, max_length=100)
    model: Optional[str] = Field(None, max_length=100)
    color: Optional[str] = Field(None, max_length=50)
    year: Optional[int] = Field(None, ge=1900, le=2100)
    owner_name: Optional[str] = Field(None, max_length=255)
    owner_phone: Optional[str] = Field(None, max_length=20)
    owner_address: Optional[str] = None
    is_authorized: bool = True
    is_blacklisted: bool = False
    notes: Optional[str] = None


class VehicleCreate(VehicleBase):
    created_by: Optional[str] = Field(None, max_length=100)


class VehicleUpdate(BaseModel):
    license_plate: Optional[str] = Field(None, max_length=20)
    province: Optional[str] = Field(None, max_length=100)
    vehicle_type: Optional[VehicleTypeEnum] = None
    brand: Optional[str] = Field(None, max_length=100)
    model: Optional[str] = Field(None, max_length=100)
    color: Optional[str] = Field(None, max_length=50)
    year: Optional[int] = Field(None, ge=1900, le=2100)
    owner_name: Optional[str] = Field(None, max_length=255)
    owner_phone: Optional[str] = Field(None, max_length=20)
    owner_address: Optional[str] = None
    is_authorized: Optional[bool] = None
    is_blacklisted: Optional[bool] = None
    notes: Optional[str] = None


class VehicleResponse(VehicleBase):
    id: int
    created_at: datetime
    updated_at: datetime
    created_by: Optional[str]

    class Config:
        from_attributes = True


# Access Log Schemas
class AccessLogBase(BaseModel):
    tracking_id: str
    camera_id: Optional[str] = None
    detected_plate: str
    detected_province: Optional[str] = None
    confidence_score: float = Field(..., ge=0.0, le=1.0)
    vehicle_type: VehicleTypeEnum = VehicleTypeEnum.UNKNOWN


class AccessLogCreate(AccessLogBase):
    full_image_path: Optional[str] = None
    plate_crop_path: Optional[str] = None
    vehicle_bbox: Optional[Dict[str, float]] = None
    plate_bbox: Optional[Dict[str, float]] = None
    status: ProcessStatusEnum = ProcessStatusEnum.PENDING_VERIFY
    processing_time_ms: Optional[float] = None
    ocr_raw_output: Optional[Dict[str, Any]] = None
    model_versions: Optional[Dict[str, str]] = None


class AccessLogVerify(BaseModel):
    """Schema for manual verification/correction"""
    corrected_plate: str = Field(..., max_length=20)
    corrected_province: str = Field(..., max_length=100)
    status: ProcessStatusEnum = Field(..., description="MLPR or REJECTED")
    verified_by: str = Field(..., max_length=100)
    verification_notes: Optional[str] = None

    @validator('status')
    def validate_status(cls, v):
        if v not in [ProcessStatusEnum.MLPR, ProcessStatusEnum.REJECTED]:
            raise ValueError('Status must be MLPR or REJECTED for manual verification')
        return v


class AccessLogResponse(AccessLogBase):
    id: int
    detection_timestamp: datetime
    full_image_path: Optional[str]
    plate_crop_path: Optional[str]
    vehicle_bbox: Optional[Dict[str, float]]
    plate_bbox: Optional[Dict[str, float]]
    status: ProcessStatusEnum
    corrected_plate: Optional[str]
    corrected_province: Optional[str]
    verified_at: Optional[datetime]
    verified_by: Optional[str]
    verification_notes: Optional[str]
    vehicle_id: Optional[int]
    processing_time_ms: Optional[float]
    added_to_training: bool
    final_plate: str
    final_province: Optional[str]

    class Config:
        from_attributes = True


# Detection Result Schema (Real-time)
class DetectionResult(BaseModel):
    """Real-time detection result from video stream"""
    tracking_id: str
    frame_number: int
    timestamp: datetime
    vehicle_bbox: Dict[str, float]
    vehicle_type: VehicleTypeEnum
    vehicle_confidence: float
    plate_detected: bool
    plate_bbox: Optional[Dict[str, float]] = None
    plate_text: Optional[str] = None
    plate_confidence: Optional[float] = None
    province: Optional[str] = None
    in_zone: bool
    skipped_dedup: bool


# Camera Config Schemas
class PolygonPoint(BaseModel):
    x: int
    y: int


class CameraConfigBase(BaseModel):
    camera_id: str = Field(..., max_length=100)
    camera_name: str = Field(..., max_length=255)
    rtsp_url: str = Field(..., max_length=500)
    is_active: bool = True
    polygon_zone: List[PolygonPoint]
    frame_skip: int = Field(2, ge=1, le=30)
    min_confidence_vehicle: float = Field(0.5, ge=0.0, le=1.0)
    min_confidence_plate: float = Field(0.4, ge=0.0, le=1.0)
    dedup_window_seconds: int = Field(60, ge=10, le=600)


class CameraConfigCreate(CameraConfigBase):
    pass


class CameraConfigUpdate(BaseModel):
    camera_name: Optional[str] = Field(None, max_length=255)
    rtsp_url: Optional[str] = Field(None, max_length=500)
    is_active: Optional[bool] = None
    polygon_zone: Optional[List[PolygonPoint]] = None
    frame_skip: Optional[int] = Field(None, ge=1, le=30)
    min_confidence_vehicle: Optional[float] = Field(None, ge=0.0, le=1.0)
    min_confidence_plate: Optional[float] = Field(None, ge=0.0, le=1.0)
    dedup_window_seconds: Optional[int] = Field(None, ge=10, le=600)


class CameraConfigResponse(CameraConfigBase):
    id: int
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


# Statistics Schemas
class DashboardKPI(BaseModel):
    """Dashboard KPI metrics"""
    total_detections: int
    alpr_auto_count: int
    mlpr_count: int
    pending_count: int
    rejected_count: int
    accuracy_percentage: float
    avg_confidence: float
    unique_vehicles: int
    avg_processing_time_ms: float


class DateRangeStats(BaseModel):
    """Statistics for a date range"""
    start_date: datetime
    end_date: datetime
    total_detections: int
    daily_breakdown: List[Dict[str, Any]]


# Image Upload Schema
class ImageUploadResponse(BaseModel):
    """Response for batch image upload"""
    filename: str
    detected_plate: Optional[str]
    province: Optional[str]
    confidence: Optional[float]
    success: bool
    error: Optional[str] = None
    image_url: Optional[str] = None
    plate_crop_url: Optional[str] = None


# Pagination
class PaginatedResponse(BaseModel):
    """Generic paginated response"""
    items: List[Any]
    total: int
    page: int
    page_size: int
    total_pages: int

    class Config:
        from_attributes = True


# WebSocket Messages
class WSMessage(BaseModel):
    """WebSocket message format"""
    type: str  # "detection", "status_update", "error"
    data: Dict[str, Any]
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# Health Check
class HealthCheck(BaseModel):
    status: str
    database: bool
    redis: bool
    models_loaded: bool
    gpu_available: bool
    version: str