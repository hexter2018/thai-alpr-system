"""RTSP Stream Control API"""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ...database import get_async_db
from ...models import CameraConfig
from ...services.rtsp_service import get_stream_manager
from ...config import get_env_camera_configs

router = APIRouter()

@router.post("/start/{camera_id}")
async def start_camera(camera_id: str, db: AsyncSession = Depends(get_async_db)):
    manager = get_stream_manager()

    if camera_id not in manager.processors:
        query = select(CameraConfig).where(
            CameraConfig.camera_id == camera_id,
            CameraConfig.is_active.is_(True)
        )
        result = await db.execute(query)
        camera_config = result.scalar_one_or_none()

        if camera_config:
            await manager.add_camera(
                camera_id=camera_config.camera_id,
                rtsp_url=camera_config.rtsp_url,
                frame_skip=camera_config.frame_skip,
                polygon_zone=camera_config.polygon_zone
            )

        else:
            env_cameras = get_env_camera_configs()
            env_camera = next((c for c in env_cameras if c["camera_id"] == camera_id), None)
            if env_camera:
                await manager.add_camera(
                    camera_id=env_camera["camera_id"],
                    rtsp_url=env_camera["rtsp_url"],
                    frame_skip=env_camera["frame_skip"],
                    polygon_zone=env_camera["polygon_zone"],
                )

    if manager.is_camera_running(camera_id):
        return {"message": "Camera already running", "camera_id": camera_id}
    
    success = await manager.start_camera(camera_id)
    if not success:
        raise HTTPException(status_code=400, detail="Failed to start camera")
    
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
    if not stats:
        raise HTTPException(status_code=404, detail="Camera not found")
    return stats

@router.get("/list")
async def list_cameras():
    manager = get_stream_manager()
    return {"cameras": manager.get_all_stats()}
