#!/usr/bin/env python3
"""
FastAPI Main Application - Thai ALPR System Backend
Optimized startup with TensorRT check disabled & Robust Camera Loading
"""

# IMPORTANT: Set these BEFORE importing any ultralytics modules
import os
os.environ['YOLO_OFFLINE'] = '1'
os.environ['ULTRALYTICS_AUTOINSTALL'] = 'False'

import logging
import sys
import json
from contextlib import asynccontextmanager
from typing import Dict, Any, List
from datetime import datetime

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy import select

# Disable Ultralytics checks after import
try:
    from ultralytics.utils import SETTINGS
    # Newer Ultralytics versions removed some legacy keys (e.g. "checks").
    # Only update settings that exist to avoid startup-time KeyError.
    for setting_key in ("sync", "checks"):
        if setting_key in SETTINGS:
            SETTINGS[setting_key] = False
except ImportError:
    pass

from app.config import get_settings, get_env_camera_configs
from app.database import init_database, get_db_manager
from app.services.redis_service import init_redis, close_redis, get_redis
from app.services.rtsp_service import init_stream_manager, get_stream_manager
from app.services.active_learning import init_active_learning
from app.core.ai_core import create_alpr_core

# Setup Logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

alpr_core = None
stream_manager = None


# ==================== WebSocket Manager (Moved Up for Global Access) ====================
class WebSocketManager:
    def __init__(self):
        self.active_connections: Dict[str, List[WebSocket]] = {}

    async def connect(self, websocket: WebSocket, camera_id: str):
        await websocket.accept()
        if camera_id not in self.active_connections:
            self.active_connections[camera_id] = []
        self.active_connections[camera_id].append(websocket)
        logger.info(f"WS connected: {camera_id} (total={len(self.active_connections[camera_id])})")

    def disconnect(self, websocket: WebSocket, camera_id: str):
        if camera_id in self.active_connections:
            try:
                if websocket in self.active_connections[camera_id]:
                    self.active_connections[camera_id].remove(websocket)
            except ValueError:
                pass
            
            # Clean up empty lists
            if not self.active_connections[camera_id]:
                del self.active_connections[camera_id]
                
        logger.info(f"WS disconnected: {camera_id}")

    async def broadcast(self, camera_id: str, data: Any):
        """Broadcast JSON data to all clients subscribed to camera_id"""
        if camera_id not in self.active_connections:
            return

        connections = self.active_connections[camera_id]
        dead_connections = []
        
        for connection in connections:
            try:
                await connection.send_json(data)
            except Exception as e:
                logger.warning(f"Failed to send to WS {camera_id}: {e}")
                dead_connections.append(connection)
        
        # Cleanup dead connections
        for dead in dead_connections:
            try:
                if dead in self.active_connections[camera_id]:
                    self.active_connections[camera_id].remove(dead)
            except ValueError:
                pass

websocket_manager = WebSocketManager()


# ==================== Detection Handler ====================
async def handle_detection(camera_id: str, results: list):
    """Save detections to DB and broadcast via WebSocket"""
    try:
        from app.models import AccessLog, ProcessStatus

        # 1. Save to DB
        db_manager = get_db_manager()
        try:
            async with db_manager.session_scope() as session:
                for result in results:
                    log = AccessLog(
                        tracking_id=result.get("tracking_id"),
                        detection_timestamp=result.get("timestamp", datetime.now()),
                        camera_id=camera_id,
                        full_image_path=result.get("full_image_path"),
                        plate_crop_path=result.get("plate_crop_path"),
                        detected_plate=result.get("detected_plate", ""),
                        detected_province=result.get("detected_province", ""),
                        confidence_score=result.get("ocr_confidence", 0.0),
                        vehicle_type=result.get("vehicle_type", "unknown"),
                        # Convert list/dict to JSON string for DB storage if needed, or rely on SQLAlchemy JSON type
                        vehicle_bbox=result.get("vehicle_bbox"), 
                        plate_bbox=result.get("plate_bbox"),
                        status=ProcessStatus[result.get("status", "COMPLETED")],
                        ocr_raw_output=result.get("ocr_raw"),
                        model_versions=result.get("model_versions")
                    )
                    session.add(log)
                # Auto-commit happens on exit of session_scope
        except Exception as db_err:
            logger.error(f"DB Save Error: {db_err}")

        # 2. Prepare for WebSocket (Serialize)
        serialized_results = []
        for r in results:
            item = r.copy()
            # Convert non-serializable objects
            for k, v in item.items():
                if isinstance(v, datetime):
                    item[k] = v.isoformat()
            serialized_results.append(item)

        # 3. Broadcast
        await websocket_manager.broadcast(camera_id, {
            "type": "detection",
            "camera_id": camera_id,
            "detections": serialized_results,
            "timestamp": datetime.now().isoformat()
        })

        logger.info(f"Processed {len(results)} detections from camera {camera_id}")

    except Exception as e:
        logger.error(f"Failed to handle detection: {e}", exc_info=True)


# ==================== Lifespan (Startup/Shutdown) ====================
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 Starting Thai ALPR System...")
    settings = get_settings()

    try:
        # 1. Database
        logger.info("📊 Initializing database...")
        db_manager = init_database(database_url=settings.DATABASE_URL, echo=settings.DB_ECHO)
        if not db_manager.test_connection():
            logger.error("❌ Database connection failed initially. Retrying/Ignoring for now...")
            # raise RuntimeError("Database connection failed") # Don't crash yet, try to continue
        else:
            db_manager.initialize_schema()

        # 2. Redis
        logger.info("🔴 Initializing Redis...")
        try:
            redis_service = await init_redis(
                host=settings.REDIS_HOST, port=settings.REDIS_PORT,
                password=settings.REDIS_PASSWORD, db=settings.REDIS_DB
            )
            if not await redis_service.is_connected():
                logger.warning("⚠️ Redis connection failed. Caching features will be disabled.")
        except Exception as redis_err:
            logger.error(f"❌ Redis Init Error: {redis_err}")
            # Mock or minimal service if needed, but for now we proceed

        # 3. AI Core
        logger.info("🤖 Initializing AI Core...")
        global alpr_core
        try:
            alpr_core = create_alpr_core(
                vehicle_model_path=settings.VEHICLE_MODEL_PATH,
                plate_model_path=settings.PLATE_MODEL_PATH,
                use_tensorrt=settings.USE_TENSORRT,
                ocr_engine=settings.OCR_ENGINE,
                high_confidence_threshold=settings.HIGH_CONFIDENCE_THRESHOLD,
                storage_path=settings.STORAGE_PATH,
                device=settings.DEVICE
            )
        except Exception as ai_err:
            logger.error(f"❌ AI Core Init Failed: {ai_err}")
            # If AI fails, we can still serve API/Stream (empty) but it's critical.
            # We'll allow startup to debug.

        # 4. Stream Manager
        logger.info("📹 Initializing stream manager...")
        global stream_manager
        stream_manager = init_stream_manager(
            alpr_core=alpr_core,
            redis_service=redis_service if 'redis_service' in locals() else None,
            on_detection=handle_detection
        )

        # 5. Load cameras from DB (Robust Mode)
        logger.info("📷 Loading active cameras from DB...")
        try:
            from app.models import CameraConfig
            async with db_manager.session_scope() as session:
                result = await session.execute(
                    select(CameraConfig).where(CameraConfig.is_active.is_(True))
                )
                active_cameras = result.scalars().all()
            
            db_started = 0
            for cam in active_cameras:
                try:
                    # Pass kwargs safely
                    added = await stream_manager.add_camera(
                        camera_id=cam.camera_id,
                        rtsp_url=cam.rtsp_url,
                        frame_skip=cam.frame_skip or 2,
                        polygon_zone=cam.polygon_zone,
                    )
                    if added:
                        if await stream_manager.start_camera(cam.camera_id):
                            db_started += 1
                            logger.info(f"   - Started DB camera: {cam.camera_id}")
                        else:
                            logger.warning(f"   - Failed to start DB camera: {cam.camera_id}")
                except Exception as cam_err:
                    logger.error(f"   ⚠️ Error adding DB camera {cam.camera_id}: {cam_err}")

            logger.info(f"📹 Started {db_started}/{len(active_cameras)} cameras from DB")
        except Exception as db_cam_err:
            logger.error(f"❌ Failed to load cameras from DB: {db_cam_err}")

        # 6. Load cameras from .env (Fallback)
        env_cameras = get_env_camera_configs()
        env_started = 0
        for cam in env_cameras:
            try:
                if cam["camera_id"] in stream_manager.processors:
                    continue
                
                added = await stream_manager.add_camera(
                    camera_id=cam["camera_id"],
                    rtsp_url=cam["rtsp_url"],
                    frame_skip=cam["frame_skip"],
                    polygon_zone=cam["polygon_zone"],
                )
                if added:
                    if await stream_manager.start_camera(cam["camera_id"]):
                        env_started += 1
            except Exception as env_err:
                logger.error(f"   ⚠️ Error adding .env camera {cam['camera_id']}: {env_err}")

        if env_cameras:
            logger.info(f"📹 Started {env_started}/{len(env_cameras)} cameras from .env")

        # 7. Active Learning
        try:
            init_active_learning(dataset_path=settings.DATASET_PATH, min_samples=settings.MIN_SAMPLES_FOR_TRAINING)
        except Exception as al_err:
            logger.warning(f"Active Learning Init Warning: {al_err}")

        total = len(stream_manager.get_active_cameras())
        logger.info(f"✅ Startup complete! {total} camera(s) running.")

        yield

        # Shutdown
        logger.info("🛑 Shutting down...")
        if stream_manager:
            await stream_manager.shutdown()
        if 'redis_service' in locals() and redis_service:
            await close_redis()
        if 'db_manager' in locals() and db_manager:
            db_manager.close()
            await db_manager.async_close()
        logger.info("✅ Shutdown complete!")

    except Exception as e:
        logger.error(f"❌ Critical Startup Error: {e}", exc_info=True)
        # We try to yield even if error, so Uvicorn doesn't just die immediately 
        # allowing us to see logs.
        yield


app = FastAPI(
    title="Thai ALPR System API",
    description="Automatic License Plate Recognition System for Thai Vehicles",
    version="1.0.0",
    lifespan=lifespan
)

settings = get_settings()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allow all for debugging connection issues
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

try:
    app.mount("/storage", StaticFiles(directory=settings.STORAGE_PATH), name="storage")
except RuntimeError:
    logger.warning(f"Storage directory not found: {settings.STORAGE_PATH}")
    # Create it if missing to avoid error
    os.makedirs(settings.STORAGE_PATH, exist_ok=True)
    try:
        app.mount("/storage", StaticFiles(directory=settings.STORAGE_PATH), name="storage")
    except:
        pass


# ==================== Health Check ====================
@app.get("/health")
async def health_check():
    try:
        db_ok = get_db_manager().test_connection()
        redis_ok = False
        try:
            redis_service = await get_redis()
            redis_ok = await redis_service.is_connected()
        except:
            pass
            
        import torch
        gpu_ok = torch.cuda.is_available()
        sm = get_stream_manager()
        
        return {
            "status": "healthy" if db_ok else "degraded",
            "timestamp": datetime.now().isoformat(),
            "database": db_ok,
            "redis": redis_ok,
            "gpu_available": gpu_ok,
            "models_loaded": alpr_core is not None,
            "active_cameras": sm.get_active_cameras() if sm else [],
            "version": settings.APP_VERSION
        }
    except Exception as e:
        return JSONResponse(status_code=503, content={"status": "error", "message": str(e)})


@app.get("/")
async def root():
    return {"name": settings.APP_NAME, "version": settings.APP_VERSION, "status": "running"}


# ==================== API Routes ====================
from app.api.routes import alpr, vehicles, logs, stream, stats, verification

app.include_router(alpr.router, prefix="/api/alpr", tags=["ALPR"])
app.include_router(vehicles.router, prefix="/api/vehicles", tags=["Vehicles"])
app.include_router(logs.router, prefix="/api/logs", tags=["Logs"])
app.include_router(stream.router, prefix="/api/stream", tags=["Stream"])
app.include_router(stats.router, prefix="/api/stats", tags=["Statistics"])
app.include_router(verification.router, prefix="/api/verification", tags=["Verification"])


# ==================== WebSocket Endpoint ====================
@app.websocket("/ws/{camera_id}")
async def websocket_endpoint(websocket: WebSocket, camera_id: str):
    await websocket_manager.connect(websocket, camera_id)

    # Send initial status
    try:
        sm = get_stream_manager()
        is_running = sm.is_camera_running(camera_id)
        stats_data = sm.get_camera_stats(camera_id) if is_running else {}
        
        await websocket.send_json({
            "type": "status",
            "camera_id": camera_id,
            "status": "running" if is_running else "stopped",
            "stats": stats_data
        })
    except Exception as e:
        logger.warning(f"WS Init Error: {e}")

    try:
        while True:
            data = await websocket.receive_text()
            if data == "ping":
                await websocket.send_text("pong")
    except WebSocketDisconnect:
        websocket_manager.disconnect(websocket, camera_id)
    except Exception as e:
        logger.error(f"WebSocket Loop Error: {e}")
        websocket_manager.disconnect(websocket, camera_id)


# ==================== Exception Handlers ====================
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error",
                 "message": str(exc) if settings.DEBUG else "An error occurred"}
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host=settings.HOST, port=settings.PORT,
                reload=settings.RELOAD, log_level=settings.LOG_LEVEL.lower())