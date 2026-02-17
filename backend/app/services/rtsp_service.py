"""
RTSP Service for Video Stream Processing
Handles multiple camera streams with frame processing
"""
import logging
import asyncio
from typing import Optional, Dict, Callable, List
from datetime import datetime
import cv2
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)


class RTSPStreamProcessor:
    """
    RTSP stream processor for a single camera
    Reads frames and processes them through ALPR pipeline
    """
    
    def __init__(
        self,
        camera_id: str,
        rtsp_url: str,
        alpr_core,
        redis_service,
        frame_skip: int = 2,
        polygon_zone: Optional[List[Dict]] = None,
        on_detection: Optional[Callable] = None
    ):
        """
        Initialize RTSP processor
        
        Args:
            camera_id: Camera identifier
            rtsp_url: RTSP stream URL
            alpr_core: ALPR core instance
            redis_service: Redis service instance
            frame_skip: Process every N frames
            polygon_zone: Detection zone
            on_detection: Callback for detections
        """
        self.camera_id = camera_id
        self.rtsp_url = rtsp_url
        self.alpr_core = alpr_core
        self.redis_service = redis_service
        self.frame_skip = frame_skip
        self.polygon_zone = polygon_zone
        self.on_detection = on_detection
        
        self.capture: Optional[cv2.VideoCapture] = None
        self.is_running = False
        self.frame_count = 0
        self.detection_count = 0
        
        # Statistics
        self.stats = {
            "frames_processed": 0,
            "frames_skipped": 0,
            "detections": 0,
            "errors": 0,
            "fps": 0.0,
            "started_at": None
        }
    
    async def start(self):
        """Start processing stream"""
        if self.is_running:
            logger.warning(f"Stream {self.camera_id} already running")
            return
        
        logger.info(f"Starting RTSP stream: {self.camera_id}")
        
        try:
            # Open video capture
            self.capture = cv2.VideoCapture(self.rtsp_url)
            
            if not self.capture.isOpened():
                raise RuntimeError(f"Failed to open RTSP stream: {self.rtsp_url}")
            
            # Set buffer size (reduce latency)
            self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            self.is_running = True
            self.stats["started_at"] = datetime.now()
            
            # Mark camera as active in Redis
            await self.redis_service.set_camera_active(self.camera_id, True)
            
            # Start processing loop
            await self._process_loop()
            
        except Exception as e:
            logger.error(f"Failed to start stream {self.camera_id}: {e}")
            await self.stop()
            raise
    
    async def stop(self):
        """Stop processing stream"""
        logger.info(f"Stopping RTSP stream: {self.camera_id}")
        
        self.is_running = False
        
        if self.capture:
            self.capture.release()
            self.capture = None
        
        # Mark camera as inactive
        await self.redis_service.set_camera_active(self.camera_id, False)
    
    async def _process_loop(self):
        """Main processing loop"""
        logger.info(f"Processing loop started for {self.camera_id}")
        
        last_time = datetime.now()
        fps_counter = 0
        
        while self.is_running:
            try:
                # Read frame
                ret, frame = await asyncio.to_thread(self.capture.read)
                
                if not ret or frame is None:
                    logger.warning(f"Failed to read frame from {self.camera_id}")
                    await asyncio.sleep(0.1)
                    continue
                
                self.frame_count += 1
                
                # Frame skip logic
                if self.frame_count % self.frame_skip != 0:
                    self.stats["frames_skipped"] += 1
                    continue
                
                # Process frame
                await self._process_frame(frame)
                
                self.stats["frames_processed"] += 1
                fps_counter += 1
                
                # Calculate FPS every second
                now = datetime.now()
                elapsed = (now - last_time).total_seconds()
                if elapsed >= 1.0:
                    self.stats["fps"] = fps_counter / elapsed
                    fps_counter = 0
                    last_time = now
                
                # Small delay to prevent CPU overload
                await asyncio.sleep(0.001)
                
            except asyncio.CancelledError:
                logger.info(f"Processing cancelled for {self.camera_id}")
                break
            except Exception as e:
                logger.error(f"Error processing frame {self.camera_id}: {e}")
                self.stats["errors"] += 1
                await asyncio.sleep(0.1)
        
        logger.info(f"Processing loop ended for {self.camera_id}")
    
    async def _process_frame(self, frame: np.ndarray):
        """Process a single frame through ALPR pipeline"""
        try:
            # Process through ALPR core
            results = await self.alpr_core.process_frame(
                frame=frame,
                frame_id=self.frame_count,
                camera_id=self.camera_id,
                polygon_zone=self.polygon_zone,
                redis_client=self.redis_service.client
            )
            
            if results:
                self.detection_count += len(results)
                self.stats["detections"] += len(results)
                
                # Publish to Redis
                for result in results:
                    await self.redis_service.publish_detection(
                        self.camera_id,
                        result
                    )
                
                # Call detection callback
                if self.on_detection:
                    await self.on_detection(self.camera_id, results)
                
                logger.info(
                    f"Camera {self.camera_id}: Detected {len(results)} plates "
                    f"(Frame {self.frame_count})"
                )
        
        except Exception as e:
            logger.error(f"Frame processing error: {e}", exc_info=True)
            self.stats["errors"] += 1
    
    def get_stats(self) -> Dict:
        """Get processing statistics"""
        runtime = None
        if self.stats["started_at"]:
            runtime = (datetime.now() - self.stats["started_at"]).total_seconds()
        
        return {
            **self.stats,
            "camera_id": self.camera_id,
            "is_running": self.is_running,
            "frame_count": self.frame_count,
            "detection_count": self.detection_count,
            "runtime_seconds": runtime
        }
    
    async def get_current_frame(self) -> Optional[np.ndarray]:
        """Get current frame from stream"""
        if not self.capture or not self.is_running:
            return None
        
        try:
            ret, frame = await asyncio.to_thread(self.capture.read)
            if ret:
                return frame
        except Exception as e:
            logger.error(f"Failed to get current frame: {e}")
        
        return None


class RTSPStreamManager:
    """
    Manager for multiple RTSP stream processors
    Handles starting, stopping, and monitoring multiple cameras
    """
    
    def __init__(
        self,
        alpr_core,
        redis_service,
        on_detection: Optional[Callable] = None
    ):
        """
        Initialize stream manager
        
        Args:
            alpr_core: ALPR core instance
            redis_service: Redis service instance
            on_detection: Global detection callback
        """
        self.alpr_core = alpr_core
        self.redis_service = redis_service
        self.on_detection = on_detection
        
        self.processors: Dict[str, RTSPStreamProcessor] = {}
        self.tasks: Dict[str, asyncio.Task] = {}
    
    async def add_camera(
        self,
        camera_id: str,
        rtsp_url: str,
        frame_skip: int = 2,
        polygon_zone: Optional[List[Dict]] = None
    ) -> bool:
        """
        Add a camera stream
        
        Args:
            camera_id: Camera identifier
            rtsp_url: RTSP URL
            frame_skip: Frame skip rate
            polygon_zone: Detection zone
        
        Returns:
            Success status
        """
        if camera_id in self.processors:
            logger.warning(f"Camera {camera_id} already exists")
            return False
        
        try:
            processor = RTSPStreamProcessor(
                camera_id=camera_id,
                rtsp_url=rtsp_url,
                alpr_core=self.alpr_core,
                redis_service=self.redis_service,
                frame_skip=frame_skip,
                polygon_zone=polygon_zone,
                on_detection=self.on_detection
            )
            
            self.processors[camera_id] = processor
            logger.info(f"Camera {camera_id} added")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add camera {camera_id}: {e}")
            return False
    
    async def start_camera(self, camera_id: str) -> bool:
        """Start processing a camera stream"""
        if camera_id not in self.processors:
            logger.error(f"Camera {camera_id} not found")
            return False
        
        if camera_id in self.tasks and not self.tasks[camera_id].done():
            logger.warning(f"Camera {camera_id} already running")
            return False
        
        try:
            processor = self.processors[camera_id]
            task = asyncio.create_task(processor.start())
            self.tasks[camera_id] = task
            
            logger.info(f"Camera {camera_id} started")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start camera {camera_id}: {e}")
            return False
    
    async def stop_camera(self, camera_id: str) -> bool:
        """Stop processing a camera stream"""
        if camera_id not in self.processors:
            logger.error(f"Camera {camera_id} not found")
            return False
        
        try:
            processor = self.processors[camera_id]
            await processor.stop()
            
            # Cancel task if running
            if camera_id in self.tasks:
                task = self.tasks[camera_id]
                if not task.done():
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass
                del self.tasks[camera_id]
            
            logger.info(f"Camera {camera_id} stopped")
            return True
            
        except Exception as e:
            logger.error(f"Failed to stop camera {camera_id}: {e}")
            return False
    
    async def remove_camera(self, camera_id: str) -> bool:
        """Remove a camera"""
        if camera_id not in self.processors:
            logger.error(f"Camera {camera_id} not found")
            return False
        
        try:
            # Stop first
            await self.stop_camera(camera_id)
            
            # Remove processor
            del self.processors[camera_id]
            
            logger.info(f"Camera {camera_id} removed")
            return True
            
        except Exception as e:
            logger.error(f"Failed to remove camera {camera_id}: {e}")
            return False
    
    async def start_all(self) -> int:
        """Start all cameras"""
        started = 0
        for camera_id in self.processors.keys():
            if await self.start_camera(camera_id):
                started += 1
        
        logger.info(f"Started {started}/{len(self.processors)} cameras")
        return started
    
    async def stop_all(self) -> int:
        """Stop all cameras"""
        stopped = 0
        for camera_id in list(self.processors.keys()):
            if await self.stop_camera(camera_id):
                stopped += 1
        
        logger.info(f"Stopped {stopped} cameras")
        return stopped
    
    def get_camera_stats(self, camera_id: str) -> Optional[Dict]:
        """Get statistics for a specific camera"""
        if camera_id not in self.processors:
            return None
        
        return self.processors[camera_id].get_stats()
    
    def get_all_stats(self) -> Dict[str, Dict]:
        """Get statistics for all cameras"""
        return {
            camera_id: processor.get_stats()
            for camera_id, processor in self.processors.items()
        }
    
    def get_active_cameras(self) -> List[str]:
        """Get list of active camera IDs"""
        return [
            camera_id
            for camera_id, processor in self.processors.items()
            if processor.is_running
        ]
    
    def is_camera_running(self, camera_id: str) -> bool:
        """Check if camera is running"""
        if camera_id not in self.processors:
            return False
        return self.processors[camera_id].is_running
    
    async def get_camera_frame(self, camera_id: str) -> Optional[np.ndarray]:
        """Get current frame from camera"""
        if camera_id not in self.processors:
            return None
        
        return await self.processors[camera_id].get_current_frame()
    
    async def shutdown(self):
        """Shutdown all streams"""
        logger.info("Shutting down RTSP stream manager")
        await self.stop_all()


# Global stream manager instance
_stream_manager: Optional[RTSPStreamManager] = None


def get_stream_manager() -> RTSPStreamManager:
    """Get stream manager instance"""
    global _stream_manager
    
    if _stream_manager is None:
        raise RuntimeError("Stream manager not initialized")
    
    return _stream_manager


def init_stream_manager(
    alpr_core,
    redis_service,
    on_detection: Optional[Callable] = None
) -> RTSPStreamManager:
    """Initialize stream manager"""
    global _stream_manager
    
    _stream_manager = RTSPStreamManager(
        alpr_core=alpr_core,
        redis_service=redis_service,
        on_detection=on_detection
    )
    
    return _stream_manager