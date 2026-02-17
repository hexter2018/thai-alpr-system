"""
Configuration Management
Load settings from environment variables
"""
import os
import re
from pathlib import Path
from typing import Optional, List
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

class Settings(BaseSettings):
    """Application settings"""
    
    # Application
    APP_NAME: str = "Thai ALPR System"
    APP_VERSION: str = "1.0.0"
    ENVIRONMENT: str = Field(default="development", env="ENVIRONMENT")
    DEBUG: bool = Field(default=False, env="DEBUG")
    SECRET_KEY: str = Field(..., env="SECRET_KEY")
    
    # Server
    HOST: str = Field(default="0.0.0.0", env="HOST")
    PORT: int = Field(default=8000, env="PORT")
    RELOAD: bool = Field(default=False, env="RELOAD")
    
    # CORS
    CORS_ORIGINS: str = Field(
        default="http://localhost:3000,http://localhost:5173",
        env="CORS_ORIGINS"
    )
    
    @property
    def cors_origins_list(self) -> List[str]:
        """Return CORS origins parsed from comma-separated settings value."""
        if not self.CORS_ORIGINS or self.CORS_ORIGINS.strip() == "":
            return ["http://localhost:3000", "http://localhost:5173"]
        return [origin.strip() for origin in self.CORS_ORIGINS.split(",") if origin.strip()]
    
    # Database
    DATABASE_URL: str = Field(..., env="DATABASE_URL")
    DB_ECHO: bool = Field(default=False, env="DB_ECHO")
    
    # Redis
    REDIS_HOST: str = Field(default="localhost", env="REDIS_HOST")
    REDIS_PORT: int = Field(default=6379, env="REDIS_PORT")
    REDIS_PASSWORD: Optional[str] = Field(default=None, env="REDIS_PASSWORD")
    REDIS_DB: int = Field(default=0, env="REDIS_DB")
    
    # AI Models
    VEHICLE_MODEL_PATH: str = Field(
        default="./models/vehicles.pt",
        env="VEHICLE_MODEL_PATH"
    )
    PLATE_MODEL_PATH: str = Field(
        default="./models/best.pt",
        env="PLATE_MODEL_PATH"
    )
    USE_TENSORRT: bool = Field(default=True, env="USE_TENSORRT")
    TENSORRT_PRECISION: str = Field(default="fp16", env="TENSORRT_PRECISION")
    DEVICE: str = Field(default="cuda:0", env="DEVICE")
    
    # Detection Thresholds
    MIN_VEHICLE_CONFIDENCE: float = Field(default=0.5, env="MIN_VEHICLE_CONFIDENCE")
    MIN_PLATE_CONFIDENCE: float = Field(default=0.4, env="MIN_PLATE_CONFIDENCE")
    HIGH_CONFIDENCE_THRESHOLD: float = Field(default=0.95, env="HIGH_CONFIDENCE_THRESHOLD")
    
    # OCR
    OCR_ENGINE: str = Field(default="paddleocr", env="OCR_ENGINE")
    PADDLEOCR_LANG: str = Field(default="th", env="PADDLEOCR_LANG")
    PADDLEOCR_USE_GPU: bool = Field(default=True, env="PADDLEOCR_USE_GPU")
    
    # Processing
    FRAME_SKIP: int = Field(default=2, env="FRAME_SKIP")
    DEDUP_WINDOW_SECONDS: int = Field(default=60, env="DEDUP_WINDOW_SECONDS")
    
    # Storage
    STORAGE_PATH: str = Field(default="./storage", env="STORAGE_PATH")
    IMAGES_PATH: str = Field(default="./storage/images", env="IMAGES_PATH")
    PLATES_PATH: str = Field(default="./storage/plates", env="PLATES_PATH")
    DATASET_PATH: str = Field(default="./storage/dataset", env="DATASET_PATH")
    
    # Active Learning
    ACTIVE_LEARNING_ENABLED: bool = Field(default=True, env="ACTIVE_LEARNING_ENABLED")
    MIN_SAMPLES_FOR_TRAINING: int = Field(default=100, env="MIN_SAMPLES_FOR_TRAINING")
    
    # Logging
    LOG_LEVEL: str = Field(default="INFO", env="LOG_LEVEL")
    LOG_FORMAT: str = Field(default="json", env="LOG_FORMAT")
    LOG_FILE: Optional[str] = Field(default=None, env="LOG_FILE")
    
    # Security
    API_KEY: Optional[str] = Field(default=None, env="API_KEY")
    JWT_SECRET_KEY: Optional[str] = Field(default=None, env="JWT_SECRET_KEY")
    JWT_ALGORITHM: str = Field(default="HS256", env="JWT_ALGORITHM")
    JWT_EXPIRATION_MINUTES: int = Field(default=60, env="JWT_EXPIRATION_MINUTES")
    
    # Rate Limiting
    RATE_LIMIT_ENABLED: bool = Field(default=True, env="RATE_LIMIT_ENABLED")
    RATE_LIMIT_PER_MINUTE: int = Field(default=60, env="RATE_LIMIT_PER_MINUTE")
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
    )


# Global settings instance
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """Get application settings (singleton)"""
    global _settings
    
    if _settings is None:
        _settings = Settings()
    
    return _settings


def reload_settings():
    """Reload settings from environment"""
    global _settings
    _settings = Settings()

    return _settings

def get_env_camera_configs(env_file: str = ".env") -> List[dict]:
    """
    Load camera configs from environment variables and/or .env file.

    Supported naming format:
    - CAMERA_ID_1, RTSP_URL_1
    - CAMERA_ID_2, RTSP_URL_2
    - ...
    """
    cameras: List[dict] = []
    env_values: dict = {}

    env_path = Path(env_file)
    if env_path.exists():
        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            env_values[key.strip()] = value.strip()

    # Runtime environment variables should override .env values
    env_values.update(os.environ)

    camera_slots = []
    for key, value in env_values.items():
        if not isinstance(value, str):
            continue
        match = re.fullmatch(r"CAMERA_ID_(\d+)", key)
        if not match:
            continue
        camera_index = int(match.group(1))
        camera_id = value.strip()
        if not camera_id:
            continue
        camera_slots.append((camera_index, camera_id))

    try:
        frame_skip_default = int(str(env_values.get("FRAME_SKIP", "2")))
    except ValueError:
        frame_skip_default = 2

    for camera_index, camera_id in sorted(camera_slots, key=lambda x: x[0]):
        rtsp_url = str(env_values.get(f"RTSP_URL_{camera_index}", "")).strip()
        if not rtsp_url:
            continue

        cameras.append(
            {
                "camera_id": camera_id,
                "rtsp_url": rtsp_url,
                "frame_skip": frame_skip_default,
                "polygon_zone": None,
            }
        )

    return cameras