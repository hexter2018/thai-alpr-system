"""
ALPR API Routes - FIXED VERSION
Better error handling for image processing
"""
from fastapi import APIRouter, UploadFile, File, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from typing import List
import cv2
import numpy as np
from datetime import datetime

from ...database import get_async_db
from ...models import AccessLog, ProcessStatus
from ...schemas import AccessLogResponse, AccessLogVerify
from ...services.active_learning import get_active_learning
import logging

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/process-image", response_model=AccessLogResponse)
async def process_image(
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_async_db)
):
    """Process uploaded image for license plate detection"""
    try:
        from ...main import alpr_core
        
        # Check if alpr_core is initialized
        if alpr_core is None:
            logger.error("ALPR Core not initialized")
            raise HTTPException(
                status_code=503, 
                detail="ALPR system not ready. Please wait for system initialization."
            )
        
        # Validate file type
        if not file.content_type or not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Read image
        contents = await file.read()
        if len(contents) == 0:
            raise HTTPException(status_code=400, detail="Empty file")
        
        logger.info(f"Processing image upload: {file.filename}, size: {len(contents)} bytes")
        
        # Decode image
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None or image.size == 0:
            raise HTTPException(status_code=400, detail="Invalid image format or corrupted file")
        
        logger.info(f"Image decoded: shape={image.shape}")
        
        # Process through ALPR
        results = await alpr_core.process_frame(
            frame=image,
            frame_id=0,
            camera_id="upload",
            polygon_zone=None,
            redis_client=None
        )
        
        if not results:
            raise HTTPException(
                status_code=422,
                detail="No license plate detected in image. Try a clearer image with a visible license plate."
            )
        
        result = results[0]
        
        logger.info(f"Detection result: plate={result.get('detected_plate')}, confidence={result.get('ocr_confidence')}")
        
        # Save to database
        log = AccessLog(
            tracking_id=f"upload_{datetime.now().timestamp()}",
            detection_timestamp=datetime.now(),
            camera_id="upload",
            full_image_path=result.get("full_image_path"),
            plate_crop_path=result.get("plate_crop_path"),
            detected_plate=result.get("detected_plate"),
            detected_province=result.get("detected_province"),
            confidence_score=result.get("ocr_confidence", 0.0),
            vehicle_type=result.get("vehicle_type", "unknown"),
            vehicle_bbox=result.get("vehicle_bbox"),
            plate_bbox=result.get("plate_bbox"),
            status=ProcessStatus[result.get("status", "PENDING_VERIFY")],
            ocr_raw_output=result.get("ocr_raw"),
            model_versions=result.get("model_versions")
        )
        
        db.add(log)
        await db.commit()
        await db.refresh(log)
        
        logger.info(f"Saved to database: log_id={log.id}")
        
        return log
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Image processing failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500, 
            detail=f"Image processing error: {str(e)}"
        )


@router.get("/pending", response_model=List[AccessLogResponse])
async def get_pending_verifications(
    skip: int = 0,
    limit: int = 20,
    db: AsyncSession = Depends(get_async_db)
):
    """Get pending verification queue"""
    query = select(AccessLog).where(
        AccessLog.status == ProcessStatus.PENDING_VERIFY
    ).order_by(AccessLog.detection_timestamp.desc()).offset(skip).limit(limit)
    
    result = await db.execute(query)
    logs = result.scalars().all()
    
    return logs


@router.post("/verify/{log_id}", response_model=AccessLogResponse)
async def verify_detection(
    log_id: int,
    verification: AccessLogVerify,
    db: AsyncSession = Depends(get_async_db)
):
    """Manually verify/correct license plate"""
    # Get log
    result = await db.execute(select(AccessLog).where(AccessLog.id == log_id))
    log = result.scalar_one_or_none()
    
    if not log:
        raise HTTPException(status_code=404, detail="Log not found")
    
    # Update with corrections
    log.corrected_plate = verification.corrected_plate
    log.corrected_province = verification.corrected_province
    log.status = verification.status
    log.verified_by = verification.verified_by
    log.verified_at = datetime.now()
    log.verification_notes = verification.verification_notes
    
    # Add to training dataset if MLPR
    if verification.status == ProcessStatus.MLPR:
        active_learning = get_active_learning()
        await active_learning.add_correction_sample(
            plate_image_path=log.plate_crop_path,
            original_text=log.detected_plate,
            corrected_text=verification.corrected_plate,
            corrected_province=verification.corrected_province,
            confidence=log.confidence_score,
            verified_by=verification.verified_by
        )
        log.added_to_training = True
    
    await db.commit()
    await db.refresh(log)
    
    return log


@router.get("/detections/{log_id}", response_model=AccessLogResponse)
async def get_detection(
    log_id: int,
    db: AsyncSession = Depends(get_async_db)
):
    """Get detection by ID"""
    result = await db.execute(select(AccessLog).where(AccessLog.id == log_id))
    log = result.scalar_one_or_none()
    
    if not log:
        raise HTTPException(status_code=404, detail="Detection not found")
    
    return log