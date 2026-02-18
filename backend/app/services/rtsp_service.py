"""
RTSP Service for Video Stream Processing
Optimized for Non-blocking I/O and Real-time performance.
"""
import logging
import asyncio
import os
import threading
import time
from typing import Optional, Dict, Callable, List
import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ── Tuning constants ─────────────────────────────────────────────────────────
MAX_CONSECUTIVE_FAILS  = 10
RECONNECT_BASE_DELAY   = 2.0
RECONNECT_MAX_DELAY    = 30.0
FRAME_READ_TIMEOUT_SEC = 5

class RTSPStreamProcessor:
    """
    Manages a single RTSP camera connection using a dedicated thread.
    This prevents blocking the main asyncio loop.
    """
    def __init__(self, camera_id: str, rtsp_url: str, on_detection: Optional[Callable] = None):
        self.camera_id = camera_id
        self.rtsp_url = rtsp_url
        self.on_detection = on_detection
        
        # State
        self.is_running = False
        self.latest_frame: Optional[np.ndarray] = None
        self.lock = threading.Lock()
        self.last_read_time = 0.0
        self.consecutive_fails = 0
        
        # Statistics
        self.stats = {
            "fps": 0,
            "total_frames": 0,
            "status": "stopped",  # stopped, connecting, active, error
            "last_error": None
        }

        # Threading
        self.thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

    def start(self):
        if self.is_running:
            return
        
        logger.info(f"Starting RTSP processor for camera {self.camera_id}")
        self.is_running = True
        self._stop_event.clear()
        self.stats["status"] = "connecting"
        
        # Start the capture loop in a separate daemon thread
        self.thread = threading.Thread(target=self._capture_loop, daemon=True, name=f"Cam-{self.camera_id}")
        self.thread.start()

    def stop(self):
        logger.info(f"Stopping RTSP processor for camera {self.camera_id}")
        self.is_running = False
        self._stop_event.set()
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2.0)
        self.stats["status"] = "stopped"

    def get_current_frame(self) -> Optional[np.ndarray]:
        """Thread-safe way to get the latest frame."""
        with self.lock:
            if self.latest_frame is None:
                return None
            return self.latest_frame.copy()

    def get_stats(self) -> Dict:
        return self.stats

    def _build_capture(self) -> cv2.VideoCapture:
        # Force TCP to avoid UDP packet loss artifacts
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
        cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimal buffer for low latency
        return cap

    def _capture_loop(self):
        """Dedicated thread for reading frames."""
        cap = None
        fps_counter = 0
        fps_timer = time.time()

        while not self._stop_event.is_set():
            try:
                # 1. Connect if not connected
                if cap is None or not cap.isOpened():
                    self.stats["status"] = "connecting"
                    cap = self._build_capture()
                    if not cap.isOpened():
                        self._handle_error("Failed to open RTSP stream")
                        time.sleep(min(RECONNECT_BASE_DELAY * (2 ** self.consecutive_fails), RECONNECT_MAX_DELAY))
                        self.consecutive_fails += 1
                        continue
                    
                    self.consecutive_fails = 0
                    self.stats["status"] = "active"
                    logger.info(f"Camera {self.camera_id} connected.")

                # 2. Read Frame (Blocking call, but safe inside this thread)
                ret, frame = cap.read()

                if ret:
                    self.last_read_time = time.time()
                    
                    # Update latest frame safely
                    with self.lock:
                        self.latest_frame = frame
                    
                    # Update Stats
                    self.stats["total_frames"] += 1
                    fps_counter += 1
                    if time.time() - fps_timer >= 1.0:
                        self.stats["fps"] = fps_counter
                        fps_counter = 0
                        fps_timer = time.time()
                    
                    # (Optional) Call callback if needed, but usually we pull frames from AI service
                    # if self.on_detection: self.on_detection(frame)

                else:
                    logger.warning(f"Camera {self.camera_id} returned empty frame.")
                    cap.release()
                    cap = None
                    time.sleep(1)

            except Exception as e:
                logger.error(f"Error in capture loop for {self.camera_id}: {e}")
                self.stats["last_error"] = str(e)
                if cap:
                    cap.release()
                cap = None
                time.sleep(RECONNECT_BASE_DELAY)

        # Cleanup
        if cap and cap.isOpened():
            cap.release()
        logger.info(f"Capture loop exit for {self.camera_id}")

    def _handle_error(self, msg: str):
        self.stats["status"] = "error"
        self.stats["last_error"] = msg
        logger.error(f"[{self.camera_id}] {msg}")


class RTSPStreamManager:
    """
    Singleton Manager for multiple cameras.
    """
    def __init__(self, alpr_core, redis_service, on_detection=None):
        self.alpr_core = alpr_core
        self.redis_service = redis_service
        self.on_detection = on_detection
        self.processors: Dict[str, RTSPStreamProcessor] = {}

    async def start_camera(self, camera_id: str, rtsp_url: str):
        if camera_id in self.processors:
            if self.processors[camera_id].is_running:
                return
            else:
                await self.stop_camera(camera_id)

        processor = RTSPStreamProcessor(camera_id, rtsp_url, self.on_detection)
        processor.start()
        self.processors[camera_id] = processor
        logger.info(f"Camera {camera_id} registered and started.")

    async def stop_camera(self, camera_id: str):
        if camera_id in self.processors:
            self.processors[camera_id].stop()
            del self.processors[camera_id]

    async def stop_all(self):
        for cid in list(self.processors.keys()):
            await self.stop_camera(cid)

    def get_all_stats(self) -> Dict[str, Dict]:
        return {cid: p.get_stats() for cid, p in self.processors.items()}

    def get_active_cameras(self) -> List[str]:
        return [cid for cid, p in self.processors.items() if p.is_running]

    def is_camera_running(self, camera_id: str) -> bool:
        p = self.processors.get(camera_id)
        return p.is_running if p else False

    # Note: This is an async wrapper, but the underlying get is instant
    async def get_camera_frame(self, camera_id: str) -> Optional[np.ndarray]:
        p = self.processors.get(camera_id)
        return p.get_current_frame() if p else None

    async def shutdown(self):
        logger.info("Shutting down RTSP manager...")
        await self.stop_all()
        logger.info("RTSP manager shutdown complete.")


# ── Singleton ────────────────────────────────────────────────────────────────

_stream_manager: Optional[RTSPStreamManager] = None

def get_stream_manager() -> RTSPStreamManager:
    if _stream_manager is None:
        raise RuntimeError("Stream manager not initialized. Call init_stream_manager() first.")
    return _stream_manager

def init_stream_manager(
    alpr_core,
    redis_service,
    on_detection: Optional[Callable] = None,
) -> RTSPStreamManager:
    global _stream_manager
    _stream_manager = RTSPStreamManager(alpr_core, redis_service, on_detection)
    return _stream_manager