"""
FastAPI Main Application
Thai ALPR System Backend
"""
import logging
import sys
from contextlib import asynccontextmanager
from typing import Dict, Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy.ext.asyncio import AsyncSession

from .config import get_settings
from .database import init_database, get_db_manager, get_async_db
from .services.redis_service import init_redis, close_redis, get_redis
from .services.rtsp_service import init_stream_manager, get_stream_manager
from .services.active_learning import init_active_learning, get_active_learning
from .core.ai_core import create_alpr_core

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

# Global instances
alpr_core = None
stream_manager = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan events: startup and shutdown
    """
    # Startup
    logger.info("🚀 Starting Thai ALPR System...")
    
    settings = get_settings()
    
    try:
        # 1. Initialize Database
        logger.info("📊 Initializing database...")
        db_manager = init_database(
            database_url=settings.DATABASE_URL,
            echo=settings.DB_ECHO
        )
        
        if not db_manager.test_connection():
            raise RuntimeError("Database connection failed")
        
        # 2. Initialize Redis
        logger.info("🔴 Initializing Redis...")
        redis_service = await init_redis(
            host=settings.REDIS_HOST,
            port=settings.REDIS_PORT,
            password=settings.REDIS_PASSWORD,
            db=settings.REDIS_DB
        )
        
        if not await redis_service.is_connected():
            raise RuntimeError("Redis connection failed")
        
        # 3. Initialize AI Core
        logger.info("🤖 Initializing AI Core...")
        global alpr_core
        alpr_core = create_alpr_core(
            vehicle_model_path=settings.VEHICLE_MODEL_PATH,
            plate_model_path=settings.PLATE_MODEL_PATH,
            use_tensorrt=settings.USE_TENSORRT,
            ocr_engine=settings.OCR_ENGINE,
            high_confidence_threshold=settings.HIGH_CONFIDENCE_THRESHOLD,
            storage_path=settings.STORAGE_PATH,
            device=settings.DEVICE
        )
        
        # 4. Initialize Stream Manager
        logger.info("📹 Initializing stream manager...")
        global stream_manager
        stream_manager = init_stream_manager(
            alpr_core=alpr_core,
            redis_service=redis_service,
            on_detection=handle_detection
        )
        
        # 5. Initialize Active Learning
        logger.info("🎓 Initializing active learning...")
        init_active_learning(
            dataset_path=settings.DATASET_PATH,
            min_samples=settings.MIN_SAMPLES_FOR_TRAINING
        )
        
        logger.info("✅ Startup complete!")
        
        yield
        
        # Shutdown
        logger.info("🛑 Shutting down Thai ALPR System...")
        
        # Stop all streams
        if stream_manager:
            await stream_manager.shutdown()
        
        # Close Redis
        await close_redis()
        
        # Close database
        db_manager.close()
        await db_manager.async_close()
        
        logger.info("✅ Shutdown complete!")
        
    except Exception as e:
        logger.error(f"❌ Startup failed: {e}", exc_info=True)
        raise


# Create FastAPI app
app = FastAPI(
    title="Thai ALPR System API",
    description="Automatic License Plate Recognition System for Thai Vehicles",
    version="1.0.0",
    lifespan=lifespan
)

settings = get_settings()

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
try:
    app.mount("/storage", StaticFiles(directory=settings.STORAGE_PATH), name="storage")
except RuntimeError:
    logger.warning(f"Storage directory not found: {settings.STORAGE_PATH}")


# ==================== Detection Handler ====================

async def handle_detection(camera_id: str, results: list):
    """
    Handle detection results
    Save to database and broadcast via WebSocket
    """
    try:
        from .models import AccessLog, ProcessStatus
        from datetime import datetime
        
        db_manager = get_db_manager()
        
        async with db_manager.session_scope() as session:
            for result in results:
                # Create access log entry
                log = AccessLog(
                    tracking_id=result["tracking_id"],
                    detection_timestamp=result["timestamp"],
                    camera_id=camera_id,
                    
                    # Images
                    full_image_path=result["full_image_path"],
                    plate_crop_path=result["plate_crop_path"],
                    
                    # Detection results
                    detected_plate=result["detected_plate"],
                    detected_province=result.get("detected_province"),
                    confidence_score=result["ocr_confidence"],
                    
                    # Vehicle info
                    vehicle_type=result["vehicle_type"],
                    vehicle_bbox=result["vehicle_bbox"],
                    plate_bbox=result["plate_bbox"],
                    
                    # Status
                    status=ProcessStatus[result["status"]],
                    
                    # Metadata
                    ocr_raw_output=result.get("ocr_raw"),
                    model_versions=result.get("model_versions")
                )
                
                session.add(log)
            
            await session.commit()
        
        # Broadcast via WebSocket
        await websocket_manager.broadcast(camera_id, results)
        
        logger.info(f"Saved {len(results)} detections to database")
        
    except Exception as e:
        logger.error(f"Failed to handle detection: {e}", exc_info=True)


# ==================== WebSocket Manager ====================

class WebSocketManager:
    """Manage WebSocket connections"""
    
    def __init__(self):
        self.active_connections: Dict[str, list] = {}
    
    async def connect(self, websocket: WebSocket, camera_id: str):
        """Connect WebSocket client"""
        await websocket.accept()
        
        if camera_id not in self.active_connections:
            self.active_connections[camera_id] = []
        
        self.active_connections[camera_id].append(websocket)
        logger.info(f"WebSocket connected: {camera_id}")
    
    def disconnect(self, websocket: WebSocket, camera_id: str):
        """Disconnect WebSocket client"""
        if camera_id in self.active_connections:
            self.active_connections[camera_id].remove(websocket)
        logger.info(f"WebSocket disconnected: {camera_id}")
    
    async def broadcast(self, camera_id: str, data: Any):
        """Broadcast data to all connected clients"""
        if camera_id not in self.active_connections:
            return
        
        dead_connections = []
        
        for connection in self.active_connections[camera_id]:
            try:
                await connection.send_json(data)
            except Exception as e:
                logger.error(f"WebSocket send failed: {e}")
                dead_connections.append(connection)
        
        # Remove dead connections
        for conn in dead_connections:
            self.active_connections[camera_id].remove(conn)


websocket_manager = WebSocketManager()


# ==================== Health Check ====================

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Check database
        db_manager = get_db_manager()
        db_ok = db_manager.test_connection()
        
        # Check Redis
        redis_service = await get_redis()
        redis_ok = await redis_service.is_connected()
        
        # Check GPU
        import torch
        gpu_ok = torch.cuda.is_available()
        
        status = "healthy" if (db_ok and redis_ok) else "unhealthy"
        
        return {
            "status": status,
            "timestamp": datetime.now().isoformat(),
            "database": db_ok,
            "redis": redis_ok,
            "gpu_available": gpu_ok,
            "models_loaded": alpr_core is not None,
            "version": settings.APP_VERSION
        }
    except Exception as e:
        return JSONResponse(
            status_code=503,
            content={"status": "error", "message": str(e)}
        )


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "status": "running",
        "docs_url": "/docs"
    }


# ==================== Import API Routes ====================

from .api.routes import alpr, vehicles, logs, stream, stats

# Register routers
app.include_router(alpr.router, prefix="/api/alpr", tags=["ALPR"])
app.include_router(vehicles.router, prefix="/api/vehicles", tags=["Vehicles"])
app.include_router(logs.router, prefix="/api/logs", tags=["Logs"])
app.include_router(stream.router, prefix="/api/stream", tags=["Stream"])
app.include_router(stats.router, prefix="/api/stats", tags=["Statistics"])


# ==================== WebSocket Endpoint ====================

@app.websocket("/ws/{camera_id}")
async def websocket_endpoint(websocket: WebSocket, camera_id: str):
    """WebSocket endpoint for real-time updates"""
    await websocket_manager.connect(websocket, camera_id)
    
    try:
        while True:
            # Keep connection alive
            data = await websocket.receive_text()
            
            # Echo back for heartbeat
            if data == "ping":
                await websocket.send_text("pong")
    
    except WebSocketDisconnect:
        websocket_manager.disconnect(websocket, camera_id)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        websocket_manager.disconnect(websocket, camera_id)


# ==================== Exception Handlers ====================

from fastapi import Request
from fastapi.responses import JSONResponse


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler"""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": str(exc) if settings.DEBUG else "An error occurred"
        }
    )


# ==================== Run Server ====================

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.RELOAD,
        log_level=settings.LOG_LEVEL.lower()
    )