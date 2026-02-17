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
    alpr_auto_count = alpr_auto.scalar() or 0
    mlpr_count = mlpr.scalar() or 0
    pending_count = pending.scalar() or 0
    
    # Average confidence
    avg_conf = await db.execute(select(func.avg(AccessLog.confidence_score)))
    
    return {
        "total_detections": total_count,
        "alpr_auto_count": alpr_auto_count,
        "mlpr_count": mlpr_count,
        "pending_count": pending_count,
        "accuracy_percentage": (alpr_auto_count / total_count * 100) if total_count > 0 else 0,
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
    high_conf_count = high_conf.scalar() or 0
    total_count = total.scalar() or 0
    return {
        "high_confidence_count": high_conf_count,
        "total_count": total_count,
        "high_confidence_rate": (high_conf_count / total_count * 100) if total_count > 0 else 0
    }
