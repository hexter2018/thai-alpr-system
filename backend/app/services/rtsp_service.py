"""
RTSP Service with ALPR Processing - FIXED VERSION
Now actually processes frames through AI!
"""
import logging
import asyncio
import os
import threading
import time
from typing import Optional, Dict, Callable, List, Any
import cv2
import numpy as np

logger = logging.getLogger(__name__)

MAX_CONSECUTIVE_FAILS = 10
RECONNECT_BASE_DELAY = 2.0
RECONNECT_MAX_DELAY = 30.0


class RTSPStreamProcessor:
    """
    RTSP camera processor WITH AI detection
    """
    def __init__(
        self, 
        camera_id: str, 
        rtsp_url: str, 
        config: Dict[str, Any] = None,
        alpr_core = None,
        redis_service = None,
        on_detection: Optional[Callable] = None
    ):
        self.camera_id = camera_id
        self.rtsp_url = rtsp_url
        self.config = config or {}
        self.alpr_core = alpr_core
        self.redis_service = redis_service
        self.on_detection = on_detection
        
        # Processing config
        self.frame_skip = self.config.get('frame_skip', 2)
        self.polygon_zone = self.config.get('polygon_zone')
        
        # State
        self.is_running = False
        self.latest_frame: Optional[np.ndarray] = None
        self.lock = threading.Lock()
        self.last_read_time = 0.0
        self.consecutive_fails = 0
        self.frame_counter = 0
        
        # Statistics
        self.stats = {
            "fps": 0,
            "total_frames": 0,
            "processed_frames": 0,
            "detections": 0,
            "status": "stopped",
            "last_error": None
        }

        # Threading
        self.thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        
        # Async processing
        self._processing_loop_task = None

    def start(self):
        if self.is_running:
            return
        
        logger.info(f"Starting RTSP processor for camera {self.camera_id}")
        self.is_running = True
        self._stop_event.clear()
        self.stats["status"] = "connecting"
        
        # Start capture thread
        self.thread = threading.Thread(
            target=self._capture_loop, 
            daemon=True, 
            name=f"Cam-{self.camera_id}"
        )
        self.thread.start()
        
        # Start processing loop (async)
        if self.alpr_core:
            asyncio.create_task(self._processing_loop())
            logger.info(f"AI processing enabled for {self.camera_id}")

    def stop(self):
        logger.info(f"Stopping RTSP processor for camera {self.camera_id}")
        self.is_running = False
        self._stop_event.set()
        
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2.0)
        
        self.stats["status"] = "stopped"

    def get_current_frame(self) -> Optional[np.ndarray]:
        with self.lock:
            if self.latest_frame is None:
                return None
            return self.latest_frame.copy()

    def get_stats(self) -> Dict:
        return self.stats

    def _build_capture(self) -> cv2.VideoCapture:
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
        cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap

    def _capture_loop(self):
        """Thread for capturing frames from RTSP"""
        cap = None
        fps_counter = 0
        fps_timer = time.time()

        while not self._stop_event.is_set():
            try:
                # Connect if needed
                if cap is None or not cap.isOpened():
                    self.stats["status"] = "connecting"
                    cap = self._build_capture()
                    
                    if not cap.isOpened():
                        self._handle_error("Failed to open RTSP stream")
                        time.sleep(
                            min(
                                RECONNECT_BASE_DELAY * (2 ** self.consecutive_fails), 
                                RECONNECT_MAX_DELAY
                            )
                        )
                        self.consecutive_fails += 1
                        continue
                    
                    self.consecutive_fails = 0
                    self.stats["status"] = "active"
                    logger.info(f"Camera {self.camera_id} connected")

                # Read frame
                ret, frame = cap.read()

                if ret:
                    self.last_read_time = time.time()
                    
                    # Update latest frame
                    with self.lock:
                        self.latest_frame = frame
                    
                    # Update stats
                    self.frame_counter += 1
                    self.stats["total_frames"] += 1
                    fps_counter += 1
                    
                    if time.time() - fps_timer >= 1.0:
                        self.stats["fps"] = fps_counter
                        fps_counter = 0
                        fps_timer = time.time()
                else:
                    logger.warning(f"Camera {self.camera_id} empty frame")
                    cap.release()
                    cap = None
                    time.sleep(1)

            except Exception as e:
                logger.error(f"Capture error {self.camera_id}: {e}")
                self.stats["last_error"] = str(e)
                if cap:
                    cap.release()
                cap = None
                time.sleep(RECONNECT_BASE_DELAY)

        # Cleanup
        if cap and cap.isOpened():
            cap.release()
        logger.info(f"Capture loop exit: {self.camera_id}")

    async def _processing_loop(self):
        """Async loop for AI processing"""
        logger.info(f"Starting AI processing loop for {self.camera_id}")
        
        while self.is_running:
            try:
                # Skip frames based on frame_skip
                if self.frame_counter % self.frame_skip != 0:
                    await asyncio.sleep(0.01)
                    continue
                
                # Get current frame
                frame = self.get_current_frame()
                
                if frame is None:
                    await asyncio.sleep(0.1)
                    continue
                
                # Process through ALPR
                results = await self.alpr_core.process_frame(
                    frame=frame,
                    frame_id=self.frame_counter,
                    camera_id=self.camera_id,
                    polygon_zone=self.polygon_zone,
                    redis_client=self.redis_service.client if self.redis_service else None
                )
                
                self.stats["processed_frames"] += 1
                
                # Handle detections
                if results:
                    self.stats["detections"] += len(results)
                    logger.info(
                        f"Camera {self.camera_id}: Detected {len(results)} plates "
                        f"(Frame {self.frame_counter})"
                    )
                    
                    # Callback
                    if self.on_detection:
                        await self.on_detection(self.camera_id, results)
                
                # Small delay to prevent overwhelming
                await asyncio.sleep(0.05)
                
            except Exception as e:
                logger.error(f"Processing error {self.camera_id}: {e}", exc_info=True)
                await asyncio.sleep(1)
        
        logger.info(f"Processing loop exit: {self.camera_id}")

    def _handle_error(self, msg: str):
        self.stats["status"] = "error"
        self.stats["last_error"] = msg
        logger.error(f"[{self.camera_id}] {msg}")


class RTSPStreamManager:
    """Manager for multiple cameras"""
    
    def __init__(self, alpr_core=None, redis_service=None, on_detection=None):
        self.alpr_core = alpr_core
        self.redis_service = redis_service
        self.on_detection = on_detection
        self.processors: Dict[str, RTSPStreamProcessor] = {}
        
        logger.info(
            f"Stream manager initialized "
            f"(ALPR: {alpr_core is not None}, Redis: {redis_service is not None})"
        )

    async def start_camera(self, camera_id: str, rtsp_url: str = "", **kwargs):
        """Start camera with AI processing"""
        if camera_id in self.processors:
            if self.processors[camera_id].is_running:
                logger.info(f"Camera {camera_id} already running")
                return True
            else:
                await self.stop_camera(camera_id)
        
        # Use cached URL if not provided
        if not rtsp_url and camera_id in self.processors:
            rtsp_url = self.processors[camera_id].rtsp_url
        
        if not rtsp_url:
            logger.error(f"No RTSP URL for camera {camera_id}")
            return False

        processor = RTSPStreamProcessor(
            camera_id=camera_id,
            rtsp_url=rtsp_url,
            config=kwargs,
            alpr_core=self.alpr_core,
            redis_service=self.redis_service,
            on_detection=self.on_detection
        )
        
        processor.start()
        self.processors[camera_id] = processor
        
        logger.info(f"Camera {camera_id} started with config: {kwargs}")
        return True

    async def add_camera(self, camera_id: str, rtsp_url: str, **kwargs):
        """Alias for start_camera"""
        return await self.start_camera(camera_id, rtsp_url, **kwargs)

    async def stop_camera(self, camera_id: str):
        if camera_id in self.processors:
            self.processors[camera_id].stop()
            del self.processors[camera_id]
            logger.info(f"Camera {camera_id} stopped")

    async def remove_camera(self, camera_id: str):
        """Alias for stop_camera"""
        await self.stop_camera(camera_id)

    async def stop_all(self):
        for cid in list(self.processors.keys()):
            await self.stop_camera(cid)

    def get_all_stats(self) -> Dict[str, Dict]:
        return {cid: p.get_stats() for cid, p in self.processors.items()}
    
    def get_camera_stats(self, camera_id: str) -> Dict:
        p = self.processors.get(camera_id)
        return p.get_stats() if p else {}

    def get_active_cameras(self) -> List[str]:
        return [cid for cid, p in self.processors.items() if p.is_running]

    def is_camera_running(self, camera_id: str) -> bool:
        p = self.processors.get(camera_id)
        return p.is_running if p else False

    async def get_camera_frame(self, camera_id: str) -> Optional[np.ndarray]:
        p = self.processors.get(camera_id)
        return p.get_current_frame() if p else None

    async def shutdown(self):
        logger.info("Shutting down RTSP manager...")
        await self.stop_all()
        logger.info("RTSP manager shutdown complete")


# Singleton
_stream_manager: Optional[RTSPStreamManager] = None

def get_stream_manager() -> RTSPStreamManager:
    global _stream_manager
    if _stream_manager is None:
        raise RuntimeError("Stream manager not initialized")
    return _stream_manager

def init_stream_manager(
    alpr_core=None,
    redis_service=None,
    on_detection: Optional[Callable] = None,
) -> RTSPStreamManager:
    global _stream_manager
    _stream_manager = RTSPStreamManager(alpr_core, redis_service, on_detection)
    return _stream_manager