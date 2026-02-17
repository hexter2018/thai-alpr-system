"""Access Logs Query API"""
from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from typing import List, Optional
from datetime import datetime
from ...database import get_async_db
from ...models import AccessLog
from ...schemas import AccessLogResponse

router = APIRouter()

@router.get("/", response_model=List[AccessLogResponse])
async def list_logs(
    skip: int = 0, 
    limit: int = 50,
    camera_id: Optional[str] = None,
    status: Optional[str] = None,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    db: AsyncSession = Depends(get_async_db)
):
    query = select(AccessLog)
    if camera_id:
        query = query.where(AccessLog.camera_id == camera_id)
    if status:
        query = query.where(AccessLog.status == status)
    if start_date:
        query = query.where(AccessLog.detection_timestamp >= start_date)
    if end_date:
        query = query.where(AccessLog.detection_timestamp <= end_date)
    query = query.order_by(AccessLog.detection_timestamp.desc()).offset(skip).limit(limit)
    result = await db.execute(query)
    return result.scalars().all()

@router.get("/search")
async def search_logs(plate: str, db: AsyncSession = Depends(get_async_db)):
    query = select(AccessLog).where(
        (AccessLog.detected_plate.contains(plate)) | 
        (AccessLog.corrected_plate.contains(plate))
    ).order_by(AccessLog.detection_timestamp.desc()).limit(20)
    result = await db.execute(query)
    return result.scalars().all()

@router.get("/camera/{camera_id}", response_model=List[AccessLogResponse])
async def logs_by_camera(camera_id: str, skip: int = 0, limit: int = 50, db: AsyncSession = Depends(get_async_db)):
    query = select(AccessLog).where(AccessLog.camera_id == camera_id).order_by(
        AccessLog.detection_timestamp.desc()
    ).offset(skip).limit(limit)
    result = await db.execute(query)
    return result.scalars().all()
