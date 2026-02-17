"""
Verification API Routes
Manual verification and correction of detections
"""
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_
from typing import List, Optional
from datetime import datetime

from ...database import get_async_db
from ...models import AccessLog, ProcessStatus
from ...schemas import AccessLogResponse, AccessLogVerify
from ...services.active_learning import get_active_learning
from ...dependencies import get_current_user, CurrentUser
import logging

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/queue", response_model=List[AccessLogResponse])
async def get_verification_queue(
    skip: int = 0,
    limit: int = 20,
    camera_id: Optional[str] = None,
    min_confidence: Optional[float] = None,
    max_confidence: Optional[float] = None,
    db: AsyncSession = Depends(get_async_db)
):
    """
    Get pending verification queue
    
    Query Parameters:
    - skip: Pagination offset
    - limit: Page size (max 100)
    - camera_id: Filter by camera
    - min_confidence: Minimum confidence score
    - max_confidence: Maximum confidence score
    """
    query = select(AccessLog).where(
        AccessLog.status == ProcessStatus.PENDING_VERIFY
    )
    
    # Apply filters
    if camera_id:
        query = query.where(AccessLog.camera_id == camera_id)
    
    if min_confidence is not None:
        query = query.where(AccessLog.confidence_score >= min_confidence)
    
    if max_confidence is not None:
        query = query.where(AccessLog.confidence_score <= max_confidence)
    
    # Order by timestamp (oldest first for verification)
    query = query.order_by(AccessLog.detection_timestamp.asc()).offset(skip).limit(limit)
    
    result = await db.execute(query)
    logs = result.scalars().all()
    
    return logs


@router.get("/queue/count")
async def get_queue_count(
    camera_id: Optional[str] = None,
    db: AsyncSession = Depends(get_async_db)
):
    """Get count of pending verifications"""
    from sqlalchemy import func
    
    query = select(func.count(AccessLog.id)).where(
        AccessLog.status == ProcessStatus.PENDING_VERIFY
    )
    
    if camera_id:
        query = query.where(AccessLog.camera_id == camera_id)
    
    result = await db.execute(query)
    count = result.scalar()
    
    return {"pending_count": count, "camera_id": camera_id}


@router.post("/{log_id}/verify", response_model=AccessLogResponse)
async def verify_detection(
    log_id: int,
    verification: AccessLogVerify,
    db: AsyncSession = Depends(get_async_db),
    current_user: CurrentUser = Depends(get_current_user)
):
    """
    Manually verify and correct detection
    
    Path Parameters:
    - log_id: Access log ID
    
    Body:
    - corrected_plate: Corrected license plate
    - corrected_province: Corrected province
    - status: MLPR (corrected) or REJECTED
    - verified_by: Operator ID
    - verification_notes: Optional notes
    """
    # Get log
    result = await db.execute(select(AccessLog).where(AccessLog.id == log_id))
    log = result.scalar_one_or_none()
    
    if not log:
        raise HTTPException(status_code=404, detail="Detection not found")
    
    # Check if already verified
    if log.status in [ProcessStatus.MLPR, ProcessStatus.REJECTED]:
        logger.warning(f"Log {log_id} already verified with status {log.status}")
    
    # Update with corrections
    log.corrected_plate = verification.corrected_plate
    log.corrected_province = verification.corrected_province
    log.status = verification.status
    log.verified_by = verification.verified_by or current_user.username
    log.verified_at = datetime.utcnow()
    log.verification_notes = verification.verification_notes
    
    # Add to training dataset if MLPR
    if verification.status == ProcessStatus.MLPR and log.plate_crop_path:
        try:
            active_learning = get_active_learning()
            success = await active_learning.add_correction_sample(
                plate_image_path=log.plate_crop_path,
                original_text=log.detected_plate,
                corrected_text=verification.corrected_plate,
                corrected_province=verification.corrected_province,
                confidence=log.confidence_score,
                verified_by=log.verified_by
            )
            
            if success:
                log.added_to_training = True
                logger.info(f"Added log {log_id} to training dataset")
        except Exception as e:
            logger.error(f"Failed to add to training: {e}")
    
    await db.commit()
    await db.refresh(log)
    
    logger.info(f"Detection {log_id} verified by {log.verified_by}: {verification.corrected_plate}")
    
    return log


@router.post("/{log_id}/approve", response_model=AccessLogResponse)
async def approve_detection(
    log_id: int,
    db: AsyncSession = Depends(get_async_db),
    current_user: CurrentUser = Depends(get_current_user)
):
    """
    Approve detection without changes
    Changes status from PENDING_VERIFY to ALPR_AUTO
    """
    result = await db.execute(select(AccessLog).where(AccessLog.id == log_id))
    log = result.scalar_one_or_none()
    
    if not log:
        raise HTTPException(status_code=404, detail="Detection not found")
    
    if log.status != ProcessStatus.PENDING_VERIFY:
        raise HTTPException(
            status_code=400, 
            detail=f"Can only approve PENDING_VERIFY status, current: {log.status}"
        )
    
    # Approve by changing status
    log.status = ProcessStatus.ALPR_AUTO
    log.verified_by = current_user.username
    log.verified_at = datetime.utcnow()
    log.verification_notes = "Approved without changes"
    
    await db.commit()
    await db.refresh(log)
    
    logger.info(f"Detection {log_id} approved by {current_user.username}")
    
    return log


@router.post("/{log_id}/reject", response_model=AccessLogResponse)
async def reject_detection(
    log_id: int,
    reason: Optional[str] = None,
    db: AsyncSession = Depends(get_async_db),
    current_user: CurrentUser = Depends(get_current_user)
):
    """
    Reject detection as invalid
    """
    result = await db.execute(select(AccessLog).where(AccessLog.id == log_id))
    log = result.scalar_one_or_none()
    
    if not log:
        raise HTTPException(status_code=404, detail="Detection not found")
    
    log.status = ProcessStatus.REJECTED
    log.verified_by = current_user.username
    log.verified_at = datetime.utcnow()
    log.verification_notes = reason or "Rejected by operator"
    
    await db.commit()
    await db.refresh(log)
    
    logger.info(f"Detection {log_id} rejected by {current_user.username}: {reason}")
    
    return log


@router.get("/history", response_model=List[AccessLogResponse])
async def get_verification_history(
    skip: int = 0,
    limit: int = 50,
    verified_by: Optional[str] = None,
    status: Optional[ProcessStatus] = None,
    db: AsyncSession = Depends(get_async_db)
):
    """
    Get verification history
    Shows all verified (MLPR/REJECTED) detections
    """
    query = select(AccessLog).where(
        AccessLog.verified_at.isnot(None)
    )
    
    if verified_by:
        query = query.where(AccessLog.verified_by == verified_by)
    
    if status:
        query = query.where(AccessLog.status == status)
    
    query = query.order_by(AccessLog.verified_at.desc()).offset(skip).limit(limit)
    
    result = await db.execute(query)
    logs = result.scalars().all()
    
    return logs


@router.get("/stats")
async def get_verification_stats(
    verified_by: Optional[str] = None,
    db: AsyncSession = Depends(get_async_db)
):
    """Get verification statistics"""
    from sqlalchemy import func
    
    # Total verified
    query = select(func.count(AccessLog.id)).where(
        AccessLog.verified_at.isnot(None)
    )
    
    if verified_by:
        query = query.where(AccessLog.verified_by == verified_by)
    
    total_result = await db.execute(query)
    total_verified = total_result.scalar()
    
    # By status
    mlpr_result = await db.execute(
        select(func.count(AccessLog.id)).where(
            and_(
                AccessLog.status == ProcessStatus.MLPR,
                AccessLog.verified_by == verified_by if verified_by else True
            )
        )
    )
    mlpr_count = mlpr_result.scalar()
    
    rejected_result = await db.execute(
        select(func.count(AccessLog.id)).where(
            and_(
                AccessLog.status == ProcessStatus.REJECTED,
                AccessLog.verified_by == verified_by if verified_by else True
            )
        )
    )
    rejected_count = rejected_result.scalar()
    
    return {
        "total_verified": total_verified,
        "mlpr_count": mlpr_count,
        "rejected_count": rejected_count,
        "approved_count": total_verified - mlpr_count - rejected_count,
        "verified_by": verified_by
    }
