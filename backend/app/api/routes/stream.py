"""RTSP Stream Control API — with zone management AND video streaming"""
import logging
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List, Optional, Dict, Any
from pydantic import BaseModel
import cv2
import asyncio
import numpy as np

from ...database import get_async_db
from ...models import CameraConfig
from ...services.rtsp_service import get_stream_manager
from ...config import get_env_camera_configs

router = APIRouter()
logger = logging.getLogger(__name__)

# --- Models ---
class CameraAdd(BaseModel):
    camera_id: str
    camera_name: str
    rtsp_url: str
    frame_skip: int = 2
    polygon_zone: Optional[List[List[int]]] = None # List of [x, y] points

class ZoneUpdate(BaseModel):
    """Polygon zone update payload. Pass null / empty list to clear the zone."""
    polygon_zone: Optional[List[List[int]]] = None


# --- [ส่วนที่เพิ่มใหม่] Video Generator Helper ---
async def frame_generator(camera_id: str):
    """
    Async generator that yields MJPEG frames for a specific camera.
    Fetches frames from the RTSPStreamManager in a non-blocking way.
    """
    manager = get_stream_manager()
    
    # Simple loop control
    err_count = 0
    
    try:
        while True:
            # 1. Check if camera is actually running in the manager
            if not manager.is_camera_running(camera_id):
                # If stopped, we can either break or yield a "offline" image
                # For now, let's break to close the connection
                logger.info(f"Streaming stopped for {camera_id}: Camera not running.")
                break

            # 2. Fetch latest frame (non-blocking async call)
            frame = await manager.get_camera_frame(camera_id)

            if frame is None:
                # No frame available yet (initializing or lag)
                err_count += 1
                if err_count > 100: # Timeout after ~10 seconds
                     logger.warning(f"No frames from {camera_id} for 10s.")
                     err_count = 0 
                await asyncio.sleep(0.1) 
                continue
            
            err_count = 0

            # 3. Encode to JPEG
            # Optimization: Resize or lower quality if needed for bandwidth
            success, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
            if not success:
                continue
            
            frame_bytes = buffer.tobytes()

            # 4. Yield MJPEG frame
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            
            # 5. FPS Control: Sleep to limit browser load (e.g. 20 FPS)
            await asyncio.sleep(0.05)
            
    except asyncio.CancelledError:
        logger.info(f"Client disconnected from stream: {camera_id}")
    except Exception as e:
        logger.error(f"Stream error for {camera_id}: {e}")


# --- [ส่วนที่เพิ่มใหม่] API สำหรับดูภาพสด ---
@router.get("/video/{camera_id}")
async def video_feed(camera_id: str):
    """
    GET /api/stream/video/{camera_id}
    Returns a multipart/x-mixed-replace stream (MJPEG) for use in <img> tags.
    """
    manager = get_stream_manager()
    
    # Check if active
    if not manager.is_camera_running(camera_id):
        # Optional: Try to auto-start if configured but not running?
        # For now, just 404
        raise HTTPException(status_code=404, detail=f"Camera {camera_id} is not currently active.")

    return StreamingResponse(
        frame_generator(camera_id), 
        media_type="multipart/x-mixed-replace;boundary=frame"
    )


# --- Existing API Routes (Adjusted for Robustness) ---

@router.get("/list")
async def list_cameras(db: AsyncSession = Depends(get_async_db)):
    """
    List ALL cameras: currently running + registered in DB + configured in .env.
    Returns a dict keyed by camera_id.
    """
    manager = get_stream_manager()
    cameras: Dict[str, Any] = {}

    # 1. Running cameras (Source of Truth for Live Status)
    # Note: manager.get_all_stats() returns {cam_id: stats_dict}
    for cam_id, stats in manager.get_all_stats().items():
        # Retrieve config from the processor to get current zone
        processor = manager.processors.get(cam_id)
        current_zone = processor.config.get("polygon_zone") if processor else None
        
        cameras[cam_id] = {
            **stats,
            "camera_id": cam_id,
            "is_running": True,
            "source": "running",
            "polygon_zone": current_zone
        }

    # 2. DB cameras (Merge Config)
    result = await db.execute(select(CameraConfig))
    db_cams = result.scalars().all()

    for cam in db_cams:
        cid = cam.camera_id
        if cid in cameras:
            # Already running -> Enrich with DB metadata (Name, stored Zone)
            cameras[cid]["camera_name"] = cam.camera_name
            cameras[cid]["source"] = "database (active)"
            # Use DB zone if Runtime zone is missing? Or prefer Runtime?
            # Let's prefer Runtime zone if it exists, else DB.
            if not cameras[cid].get("polygon_zone"):
                 cameras[cid]["polygon_zone"] = cam.polygon_zone
        else:
            # Not running -> Add as stopped
            cameras[cid] = {
                "camera_id": cid,
                "camera_name": cam.camera_name,
                "rtsp_url": cam.rtsp_url,
                "is_running": False,
                "status": "stopped",
                "source": "database",
                "polygon_zone": cam.polygon_zone,
            }

    # 3. .env cameras (Fallback)
    for env_cam in get_env_camera_configs():
        cid = env_cam["camera_id"]
        if cid not in cameras:
            cameras[cid] = {
                "camera_id": cid,
                "camera_name": cid,
                "rtsp_url": env_cam["rtsp_url"],
                "is_running": False,
                "status": "stopped",
                "source": "env",
                "polygon_zone": env_cam.get("polygon_zone"),
            }

    active_count = len(manager.get_active_cameras())
    return {"cameras": cameras, "total": len(cameras), "active": active_count}


@router.post("/start/{camera_id}")
async def start_camera(camera_id: str, db: AsyncSession = Depends(get_async_db)):
    """Start a camera stream. Auto-registers from DB or .env if needed."""
    manager = get_stream_manager()

    # If already running, just return success
    if manager.is_camera_running(camera_id):
        return {"message": "Camera already running", "camera_id": camera_id}

    # Not running, need to find config
    # 1. Try DB
    result = await db.execute(
        select(CameraConfig).where(CameraConfig.camera_id == camera_id)
    )
    cam = result.scalar_one_or_none()

    if cam:
        # Found in DB
        await manager.add_camera(
            camera_id=cam.camera_id,
            rtsp_url=cam.rtsp_url,
            frame_skip=cam.frame_skip or 2,
            polygon_zone=cam.polygon_zone,
        )
    else:
        # 2. Try .env
        env_cam = next(
            (c for c in get_env_camera_configs() if c["camera_id"] == camera_id),
            None,
        )
        if env_cam:
            await manager.add_camera(
                camera_id=env_cam["camera_id"],
                rtsp_url=env_cam["rtsp_url"],
                frame_skip=env_cam["frame_skip"],
                polygon_zone=env_cam["polygon_zone"],
            )
        else:
            raise HTTPException(
                status_code=404,
                detail=f"Camera '{camera_id}' not found in DB or .env",
            )

    # Now start it
    # Note: add_camera calls start_camera internally in our new RTSP service, 
    # but we can call explicit start just to be sure if implementation changes.
    if not manager.is_camera_running(camera_id):
        await manager.start_camera(camera_id, rtsp_url="") # URL is cached in add_camera
    
    return {"message": "Camera started", "camera_id": camera_id}


@router.post("/stop/{camera_id}")
async def stop_camera(camera_id: str):
    manager = get_stream_manager()
    await manager.remove_camera(camera_id)
    return {"message": "Camera stopped", "camera_id": camera_id}


@router.post("/add")
async def add_camera(body: CameraAdd, db: AsyncSession = Depends(get_async_db)):
    """Persist a new camera to DB and start streaming immediately."""
    
    # 1. Save/Upsert to DB
    result = await db.execute(select(CameraConfig).where(CameraConfig.camera_id == body.camera_id))
    existing = result.scalar_one_or_none()

    if existing:
        existing.camera_name = body.camera_name
        existing.rtsp_url = body.rtsp_url
        existing.is_active = True
        existing.frame_skip = body.frame_skip
        if body.polygon_zone is not None:
             existing.polygon_zone = body.polygon_zone
    else:
        db_cam = CameraConfig(
            camera_id=body.camera_id,
            camera_name=body.camera_name,
            rtsp_url=body.rtsp_url,
            is_active=True,
            frame_skip=body.frame_skip,
            polygon_zone=body.polygon_zone,
        )
        db.add(db_cam)
    
    await db.commit()

    # 2. Start in Runtime
    manager = get_stream_manager()
    await manager.add_camera(
        camera_id=body.camera_id,
        rtsp_url=body.rtsp_url,
        frame_skip=body.frame_skip,
        polygon_zone=body.polygon_zone,
    )

    return {"message": "Camera added and started", "camera_id": body.camera_id}


@router.delete("/remove/{camera_id}")
async def remove_camera(camera_id: str, db: AsyncSession = Depends(get_async_db)):
    """Stop and deregister a camera."""
    manager = get_stream_manager()
    
    # Stop runtime
    await manager.remove_camera(camera_id)

    # Soft delete from DB
    result = await db.execute(
        select(CameraConfig).where(CameraConfig.camera_id == camera_id)
    )
    cam = result.scalar_one_or_none()
    if cam:
        cam.is_active = False 
        await db.commit()

    return {"message": "Camera removed", "camera_id": camera_id}


@router.put("/zone/{camera_id}")
async def update_zone(
    camera_id: str,
    body: ZoneUpdate,
    db: AsyncSession = Depends(get_async_db),
):
    """
    Update (or clear) the detection polygon zone.
    Applied immediately to running stream AND DB.
    """
    manager = get_stream_manager()

    # 1. Hot-update Runtime (Config)
    if camera_id in manager.processors:
        processor = manager.processors[camera_id]
        # Update config directly. 
        # The AI loop should read this config every frame or so.
        processor.config["polygon_zone"] = body.polygon_zone
        logger_msg = "Zone updated in Runtime"
    else:
        logger_msg = "Camera not running, Zone updated in DB only"

    # 2. Persist to DB
    result = await db.execute(
        select(CameraConfig).where(CameraConfig.camera_id == camera_id)
    )
    cam = result.scalar_one_or_none()

    if cam:
        cam.polygon_zone = body.polygon_zone
        await db.commit()
        return {
            "message": logger_msg,
            "camera_id": camera_id,
            "zone_points": len(body.polygon_zone) if body.polygon_zone else 0,
        }
    
    # Case: Camera exists in Runtime (e.g. from .env) but not in DB
    if camera_id in manager.processors:
         return {
            "message": "Zone updated (Runtime only - Camera not in DB)",
            "camera_id": camera_id,
            "zone_points": len(body.polygon_zone) if body.polygon_zone else 0,
        }

    raise HTTPException(status_code=404, detail=f"Camera '{camera_id}' not found")


@router.get("/zone/{camera_id}")
async def get_zone(camera_id: str, db: AsyncSession = Depends(get_async_db)):
    """Get the current polygon zone for a camera."""
    manager = get_stream_manager()

    # 1. Check live processor first (most up to date)
    processor = manager.processors.get(camera_id)
    if processor:
        zone = processor.config.get("polygon_zone")
        return {
            "camera_id":    camera_id,
            "polygon_zone": zone,
            "zone_points":  len(zone) if zone else 0,
            "source":       "live",
        }

    # 2. Fall back to DB
    result = await db.execute(
        select(CameraConfig).where(CameraConfig.camera_id == camera_id)
    )
    cam = result.scalar_one_or_none()
    if cam:
        return {
            "camera_id":    camera_id,
            "polygon_zone": cam.polygon_zone,
            "zone_points":  len(cam.polygon_zone) if cam.polygon_zone else 0,
            "source":       "database",
        }

    raise HTTPException(status_code=404, detail=f"Camera '{camera_id}' not found")