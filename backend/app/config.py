"""
Configuration Management
Load settings from environment variables
"""
import os
import re
from pathlib import Path
from typing import Optional, List

from pydantic_settings import BaseSettings
from pydantic import Field, validator


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

    # CORS - stored as comma-separated string to avoid pydantic-settings JSON parse issues
    CORS_ORIGINS: str = Field(
        default="http://localhost:3000,http://localhost:5173",
        env="CORS_ORIGINS"
    )

    @property
    def cors_origins_list(self) -> List[str]:
        if not self.CORS_ORIGINS or not self.CORS_ORIGINS.strip():
            return ["http://localhost:3000", "http://localhost:5173"]
        return [o.strip() for o in self.CORS_ORIGINS.split(",") if o.strip()]

    # Database
    DATABASE_URL: str = Field(..., env="DATABASE_URL")
    DB_ECHO: bool = Field(default=False, env="DB_ECHO")

    # Redis
    REDIS_HOST: str = Field(default="localhost", env="REDIS_HOST")
    REDIS_PORT: int = Field(default=6379, env="REDIS_PORT")
    REDIS_PASSWORD: Optional[str] = Field(default=None, env="REDIS_PASSWORD")
    REDIS_DB: int = Field(default=0, env="REDIS_DB")

    # AI Models
    VEHICLE_MODEL_PATH: str = Field(default="./models/vehicles.pt", env="VEHICLE_MODEL_PATH")
    PLATE_MODEL_PATH: str = Field(default="./models/best.pt", env="PLATE_MODEL_PATH")
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

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True


# ── Singleton ────────────────────────────────────────────────────────────────

_settings: Optional[Settings] = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def reload_settings() -> Settings:
    global _settings
    _settings = Settings()
    return _settings


# ── Camera config loader ─────────────────────────────────────────────────────

def _strip_inline_comment(value: str) -> str:
    """
    Remove inline shell-style comments from a .env value.
    e.g.  '2  # Process every N frames'  →  '2'
    Also strips surrounding quotes.
    """
    # Remove inline comment (space + # …)
    value = re.sub(r'\s+#.*$', '', value)
    # Strip surrounding single/double quotes
    value = value.strip().strip('"').strip("'")
    return value.strip()


def _load_raw_env(env_file: str = ".env") -> dict:
    """
    Read .env file into a raw dict, stripping inline comments from values.
    Runtime os.environ always overrides file values.
    """
    raw: dict = {}

    path = Path(env_file)
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            raw[key.strip()] = _strip_inline_comment(val)

    # Runtime env overrides file (also strip comments from env vars just in case)
    for key, val in os.environ.items():
        raw[key] = _strip_inline_comment(val)

    return raw


def get_env_camera_configs(env_file: str = ".env") -> List[dict]:
    """
    Load camera configs from .env / environment variables.

    Supported format:
        CAMERA_ID_1=PCN_MM04
        RTSP_URL_1=rtsp://...

        CAMERA_ID_2=PCN_MM05
        RTSP_URL_2=rtsp://...

    Returns list of camera dicts ready for RTSPStreamManager.add_camera().
    """
    env = _load_raw_env(env_file)

    # Determine default frame_skip (safe parse)
    try:
        frame_skip_default = int(env.get("FRAME_SKIP", "2"))
    except (ValueError, TypeError):
        frame_skip_default = 2

    cameras: List[dict] = []

    # Collect all CAMERA_ID_N indices
    for key, camera_id in sorted(env.items()):
        m = re.fullmatch(r"CAMERA_ID_(\d+)", key)
        if not m:
            continue
        idx = m.group(1)
        camera_id = camera_id.strip()
        if not camera_id:
            continue

        rtsp_url = env.get(f"RTSP_URL_{idx}", "").strip()
        if not rtsp_url:
            continue  # skip cameras without an RTSP URL

        cameras.append({
            "camera_id":   camera_id,
            "rtsp_url":    rtsp_url,
            "frame_skip":  frame_skip_default,
            "polygon_zone": None,
        })

    return cameras