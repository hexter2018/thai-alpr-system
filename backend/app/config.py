"""
Configuration Management
Load settings from environment variables
"""
import os
from typing import Optional, List, Annotated
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict
from pydantic import Field, field_validator


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
    CORS_ORIGINS: Annotated[List[str], NoDecode] = Field(
        default=["http://localhost:3000", "http://localhost:5173"],
        env="CORS_ORIGINS"
    )
    
    @field_validator("CORS_ORIGINS", pre=True)
    @classmethod
    def parse_cors_origins(cls, v):
        if isinstance(v, str):
            if not v or v.strip() == "":
                return ["http://localhost:3000", "http://localhost:5173"]
            return [origin.strip() for origin in v.split(",")]
        return v
    
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