"""Statistics and Dashboard API"""
from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from datetime import datetime, timedelta
from ...database import get_async_db
from ...models import AccessLog, ProcessStatus

router = APIRouter()

@router.get("/dashboard")
async def dashboard_stats(db: AsyncSession = Depends(get_async_db)):
    # Total detections
    total_query = select(func.count(AccessLog.id))
    total = await db.execute(total_query)
    total_count = total.scalar()
    
    # By status
    alpr_auto = await db.execute(select(func.count(AccessLog.id)).where(AccessLog.status == ProcessStatus.ALPR_AUTO))
    mlpr = await db.execute(select(func.count(AccessLog.id)).where(AccessLog.status == ProcessStatus.MLPR))
    pending = await db.execute(select(func.count(AccessLog.id)).where(AccessLog.status == ProcessStatus.PENDING_VERIFY))
    
    # Average confidence
    avg_conf = await db.execute(select(func.avg(AccessLog.confidence_score)))
    
    return {
        "total_detections": total_count,
        "alpr_auto_count": alpr_auto.scalar(),
        "mlpr_count": mlpr.scalar(),
        "pending_count": pending.scalar(),
        "accuracy_percentage": (alpr_auto.scalar() / total_count * 100) if total_count > 0 else 0,
        "avg_confidence": float(avg_conf.scalar() or 0)
    }

@router.get("/daily")
async def daily_stats(days: int = 7, db: AsyncSession = Depends(get_async_db)):
    start_date = datetime.now() - timedelta(days=days)
    query = select(
        func.date(AccessLog.detection_timestamp).label('date'),
        func.count(AccessLog.id).label('count')
    ).where(AccessLog.detection_timestamp >= start_date).group_by(func.date(AccessLog.detection_timestamp))
    result = await db.execute(query)
    return [{"date": str(row.date), "count": row.count} for row in result]

@router.get("/accuracy")
async def accuracy_stats(db: AsyncSession = Depends(get_async_db)):
    high_conf = await db.execute(select(func.count(AccessLog.id)).where(AccessLog.confidence_score > 0.95))
    total = await db.execute(select(func.count(AccessLog.id)))
    return {
        "high_confidence_count": high_conf.scalar(),
        "total_count": total.scalar(),
        "high_confidence_rate": (high_conf.scalar() / total.scalar() * 100) if total.scalar() > 0 else 0
    }
