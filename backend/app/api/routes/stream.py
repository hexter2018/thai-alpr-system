"""RTSP Stream Control API — with zone management"""
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List, Optional
from pydantic import BaseModel
import cv2
import asyncio

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


class ZoneUpdate(BaseModel):
    """Polygon zone update payload. Pass null / empty list to clear the zone."""
    polygon_zone: Optional[List] = None

# --- [ส่วนที่เพิ่มใหม่] Video Generator Helper ---
async def frame_generator(camera_id: str):
    """ฟังก์ชันดึงภาพทีละเฟรมจาก Manager แล้วส่งเป็น MJPEG Stream"""
    manager = get_stream_manager()
    
    while True:
        # 1. ถ้ากล้องหยุดวิ่ง ให้เลิกส่งภาพ
        if not manager.is_camera_running(camera_id):
            break

        # 2. ดึงภาพล่าสุด (Async)
        frame = await manager.get_camera_frame(camera_id)

        if frame is None:
            await asyncio.sleep(0.1) # รอภาพมา
            continue

        # 3. แปลงภาพเป็น JPEG
        success, buffer = cv2.imencode('.jpg', frame)
        if not success:
            continue
        
        frame_bytes = buffer.tobytes()

        # 4. ส่งข้อมูลตามมาตรฐาน MJPEG (Multipart)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        
        # คุม FPS ขาออก (0.04s = 25 FPS) เพื่อไม่ให้ Browser โหลดหนักเกินไป
        await asyncio.sleep(0.04)

# --- [ส่วนที่เพิ่มใหม่] API สำหรับดูภาพสด ---
@router.get("/video/{camera_id}")
async def video_feed(camera_id: str):
    """API สำหรับดูภาพสด (MJPEG)"""
    manager = get_stream_manager()
    
    # เช็คว่ากล้องทำงานอยู่ไหม
    if not manager.is_camera_running(camera_id):
        raise HTTPException(status_code=404, detail=f"Camera {camera_id} is not active")

    return StreamingResponse(
        frame_generator(camera_id), 
        media_type="multipart/x-mixed-replace;boundary=frame"
    )

@router.get("/list")
async def list_cameras(db: AsyncSession = Depends(get_async_db)):
    """
    List ALL cameras: currently running + registered in DB + configured in .env.
    Returns a dict keyed by camera_id.
    """
    manager = get_stream_manager()

    cameras: dict = {}

    # 1. Running cameras (live stats)
    for cam_id, stats in manager.get_all_stats().items():
        # ดึง zone จาก config ของ processor
        processor = manager.processors.get(cam_id)
        current_zone = processor.config.get("polygon_zone") if processor else None

        cameras[cam_id] = {
            **stats,
            "is_running": True,
            "source": "running",
            "polygon_zone": current_zone
        }

    # 2. DB cameras
    result = await db.execute(select(CameraConfig))
    for cam in result.scalars().all():
        if cam.camera_id not in cameras:
            cameras[cam.camera_id] = {
                "camera_id":   cam.camera_id,
                "camera_name": cam.camera_name,
                "rtsp_url":    cam.rtsp_url,
                "is_active":   cam.is_active,
                "is_running":  False,
                "source":      "database",
                "polygon_zone": cam.polygon_zone,
            }
        else:
            # Enrich running entry with DB meta
            cameras[cam.camera_id].setdefault("camera_name", cam.camera_name)
            # ถ้าใน Runtime ไม่มี zone ให้เอาจาก DB ไปโชว์
            if not cameras[cam.camera_id].get("polygon_zone"):
                 cameras[cam.camera_id]["polygon_zone"] = cam.polygon_zone

    # 3. .env cameras
    for env_cam in get_env_camera_configs():
        cid = env_cam["camera_id"]
        if cid not in cameras:
            cameras[cid] = {
                "camera_id":   cid,
                "camera_name": cid,
                "rtsp_url":    env_cam["rtsp_url"],
                "is_running":  False,
                "source":      "env",
                "polygon_zone": env_cam.get("polygon_zone"),
            }

    active_count = len(manager.get_active_cameras())
    return {"cameras": cameras, "total": len(cameras), "active": active_count}


@router.post("/start/{camera_id}")
async def start_camera(camera_id: str, db: AsyncSession = Depends(get_async_db)):
    """Start a camera stream. Auto-registers from DB or .env if needed."""
    manager = get_stream_manager()

    if camera_id not in manager.processors:
        # Try DB first
        result = await db.execute(
            select(CameraConfig).where(
                CameraConfig.camera_id == camera_id,
                CameraConfig.is_active.is_(True),
            )
        )
        cam = result.scalar_one_or_none()

        if cam:
            # ใช้ add_camera alias ที่เราแก้ให้รองรับ kwargs แล้ว
            await manager.add_camera(
                camera_id=cam.camera_id,
                rtsp_url=cam.rtsp_url,
                frame_skip=cam.frame_skip or 2,
                polygon_zone=cam.polygon_zone, # ส่ง zone เข้าไปเก็บใน config
            )
        else:
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

    if manager.is_camera_running(camera_id):
        return {"message": "Camera already running", "camera_id": camera_id}

    # add_camera มันเรียก start_camera ให้แล้ว แต่เรียกซ้ำเพื่อความชัวร์ก็ไม่เสียหาย (เพราะมี check is_running)
    # แต่จริงๆ บรรทัดข้างบนมัน start ให้แล้ว
    return {"message": "Camera started", "camera_id": camera_id}


@router.post("/stop/{camera_id}")
async def stop_camera(camera_id: str):
    manager = get_stream_manager()
    # remove_camera alias เรียก stop_camera ให้
    await manager.remove_camera(camera_id)
    return {"message": "Camera stopped", "camera_id": camera_id}


@router.post("/add")
async def add_camera(body: CameraAdd, db: AsyncSession = Depends(get_async_db)):
    """Persist a new camera to DB and start streaming immediately."""
    
    # 1. Save to DB first
    # เช็คก่อนว่ามีอยู่แล้วไหม (Upsert logic)
    result = await db.execute(select(CameraConfig).where(CameraConfig.camera_id == body.camera_id))
    existing = result.scalar_one_or_none()

    if existing:
        existing.camera_name = body.camera_name
        existing.rtsp_url = body.rtsp_url
        existing.is_active = True
        existing.frame_skip = body.frame_skip
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

    # 2. Start Runtime
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
    await manager.remove_camera(camera_id)

    result = await db.execute(
        select(CameraConfig).where(CameraConfig.camera_id == camera_id)
    )
    cam = result.scalar_one_or_none()
    if cam:
        cam.is_active = False # Soft delete (แค่ปิด active)
        # db.delete(cam) # หรือถ้าจะลบจริงๆ ให้ใช้บรรทัดนี้
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

    # 1. Hot-update Runtime
    if camera_id in manager.processors:
        processor = manager.processors[camera_id]
        # อัปเดตเข้าไปใน config โดยตรง
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
    
    # ถ้าไม่มีใน DB แต่มีใน Runtime (เช่น .env)
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

    # Check live processor first
    processor = manager.processors.get(camera_id)
    if processor:
        zone = processor.config.get("polygon_zone")
        return {
            "camera_id":    camera_id,
            "polygon_zone": zone,
            "zone_points":  len(zone) if zone else 0,
            "source":       "live",
        }

    # Fall back to DB
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