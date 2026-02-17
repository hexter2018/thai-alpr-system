"""RTSP Stream Control API"""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List, Optional
from pydantic import BaseModel

from ...database import get_async_db
from ...models import CameraConfig
from ...services.rtsp_service import get_stream_manager
from ...config import get_env_camera_configs

router = APIRouter()


class CameraAdd(BaseModel):
    camera_id: str
    camera_name: str
    rtsp_url: str
    frame_skip: int = 2
    polygon_zone: Optional[List] = None


@router.get("/list")
async def list_cameras(db: AsyncSession = Depends(get_async_db)):
    """
    List ALL cameras: currently running + registered in DB + configured in .env.
    Returns a dict keyed by camera_id so the frontend can do Object.entries().
    """
    manager = get_stream_manager()

    # 1. Running cameras (live stats)
    cameras: dict = {}
    for cam_id, stats in manager.get_all_stats().items():
        cameras[cam_id] = {
            **stats,
            "is_running": True,
            "source": "running",
        }

    # 2. DB cameras (may or may not be running)
    result = await db.execute(select(CameraConfig))
    for cam in result.scalars().all():
        if cam.camera_id not in cameras:
            cameras[cam.camera_id] = {
                "camera_id": cam.camera_id,
                "camera_name": cam.camera_name,
                "rtsp_url": cam.rtsp_url,
                "is_active": cam.is_active,
                "is_running": False,
                "source": "database",
            }

    # 3. .env cameras
    for env_cam in get_env_camera_configs():
        cid = env_cam["camera_id"]
        if cid not in cameras:
            cameras[cid] = {
                "camera_id": cid,
                "camera_name": cid,
                "rtsp_url": env_cam["rtsp_url"],
                "is_running": False,
                "source": "env",
            }

    active_count = len(manager.get_active_cameras())
    return {"cameras": cameras, "total": len(cameras), "active": active_count}


@router.post("/start/{camera_id}")
async def start_camera(camera_id: str, db: AsyncSession = Depends(get_async_db)):
    """Start a camera stream. Auto-registers from DB or .env if not yet loaded."""
    manager = get_stream_manager()

    # Register if not already known to the manager
    if camera_id not in manager.processors:
        # Try DB first
        result = await db.execute(
            select(CameraConfig).where(
                CameraConfig.camera_id == camera_id,
                CameraConfig.is_active.is_(True)
            )
        )
        cam = result.scalar_one_or_none()

        if cam:
            added = await manager.add_camera(
                camera_id=cam.camera_id,
                rtsp_url=cam.rtsp_url,
                frame_skip=cam.frame_skip or 2,
                polygon_zone=cam.polygon_zone,
            )
        else:
            # Try .env
            env_cam = next(
                (c for c in get_env_camera_configs() if c["camera_id"] == camera_id),
                None,
            )
            if env_cam:
                added = await manager.add_camera(
                    camera_id=env_cam["camera_id"],
                    rtsp_url=env_cam["rtsp_url"],
                    frame_skip=env_cam["frame_skip"],
                    polygon_zone=env_cam["polygon_zone"],
                )
            else:
                raise HTTPException(
                    status_code=404,
                    detail=f"Camera '{camera_id}' not found in DB or .env"
                )

        if not added:
            raise HTTPException(status_code=400, detail="Failed to register camera")

    if manager.is_camera_running(camera_id):
        return {"message": "Camera already running", "camera_id": camera_id}

    success = await manager.start_camera(camera_id)
    if not success:
        raise HTTPException(status_code=400, detail="Failed to start camera stream")

    return {"message": "Camera started", "camera_id": camera_id}


@router.post("/stop/{camera_id}")
async def stop_camera(camera_id: str):
    manager = get_stream_manager()
    success = await manager.stop_camera(camera_id)
    if not success:
        raise HTTPException(status_code=400, detail="Failed to stop camera")
    return {"message": "Camera stopped", "camera_id": camera_id}


@router.get("/status/{camera_id}")
async def camera_status(camera_id: str):
    manager = get_stream_manager()
    stats = manager.get_camera_stats(camera_id)
    if stats is None:
        raise HTTPException(status_code=404, detail="Camera not found")
    return stats


@router.post("/add")
async def add_camera(body: CameraAdd, db: AsyncSession = Depends(get_async_db)):
    """Persist a new camera to DB and start streaming immediately."""
    # Save to DB
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

    # Start streaming
    manager = get_stream_manager()
    await manager.add_camera(
        camera_id=body.camera_id,
        rtsp_url=body.rtsp_url,
        frame_skip=body.frame_skip,
        polygon_zone=body.polygon_zone,
    )
    await manager.start_camera(body.camera_id)

    return {"message": "Camera added and started", "camera_id": body.camera_id}


@router.delete("/remove/{camera_id}")
async def remove_camera(camera_id: str, db: AsyncSession = Depends(get_async_db)):
    """Stop and deregister a camera."""
    manager = get_stream_manager()
    await manager.remove_camera(camera_id)

    # Mark inactive in DB
    result = await db.execute(
        select(CameraConfig).where(CameraConfig.camera_id == camera_id)
    )
    cam = result.scalar_one_or_none()
    if cam:
        cam.is_active = False
        await db.commit()

    return {"message": "Camera removed", "camera_id": camera_id}