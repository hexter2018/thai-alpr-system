"""RTSP Stream Control API"""
from fastapi import APIRouter, HTTPException
from ...services.rtsp_service import get_stream_manager

router = APIRouter()

@router.post("/start/{camera_id}")
async def start_camera(camera_id: str):
    manager = get_stream_manager()
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
