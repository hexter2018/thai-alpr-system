"""
RTSP Service for Video Stream Processing
Supports H.264 and H.265 (HEVC) via FFmpeg backend
Auto-starts cameras on backend startup
"""
import logging
import asyncio
import os
from typing import Optional, Dict, Callable, List
from datetime import datetime
import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ── Tuning constants ─────────────────────────────────────────────────────────
MAX_CONSECUTIVE_FAILS  = 10
RECONNECT_BASE_DELAY   = 2.0
RECONNECT_MAX_DELAY    = 30.0
OPEN_TIMEOUT_SEC       = 15
FRAME_READ_TIMEOUT_SEC = 8


def _build_capture(rtsp_url: str) -> cv2.VideoCapture:
    """
    Open RTSP/H.265 stream using OpenCV FFmpeg backend with TCP transport.
    Falls back to default backend if FFmpeg fails.
    """
    # Force FFmpeg + TCP to handle H.265 (HEVC) streams reliably
    os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")

    cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)

    if cap.isOpened():
        # Low-latency: keep only 1 frame in the decode buffer
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        logger.debug("FFmpeg backend opened: %s", rtsp_url)
        return cap

    # Fallback — let OpenCV choose (GStreamer, etc.)
    cap.release()
    logger.warning("FFmpeg backend failed, trying default backend for %s", rtsp_url)
    cap = cv2.VideoCapture(rtsp_url)
    if cap.isOpened():
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


async def _open_capture_async(
    rtsp_url: str, timeout: float = OPEN_TIMEOUT_SEC
) -> Optional[cv2.VideoCapture]:
    """Open VideoCapture in a thread-pool with a wall-clock timeout."""
    try:
        cap = await asyncio.wait_for(
            asyncio.to_thread(_build_capture, rtsp_url),
            timeout=timeout,
        )
        if cap and cap.isOpened():
            return cap
        if cap:
            cap.release()
        return None
    except asyncio.TimeoutError:
        logger.error("RTSP open timed-out (%ss): %s", timeout, rtsp_url)
        return None
    except Exception as exc:
        logger.error("RTSP open error: %s — %s", rtsp_url, exc)
        return None


async def _read_frame_async(
    cap: cv2.VideoCapture, timeout: float = FRAME_READ_TIMEOUT_SEC
):
    """Read one frame with a timeout to avoid blocking the event loop."""
    try:
        return await asyncio.wait_for(
            asyncio.to_thread(cap.read),
            timeout=timeout,
        )
    except asyncio.TimeoutError:
        return False, None


# ── Per-camera processor ─────────────────────────────────────────────────────

class RTSPStreamProcessor:
    """Single-camera RTSP processor with automatic reconnect and zone support."""

    def __init__(
        self,
        camera_id: str,
        rtsp_url: str,
        alpr_core,
        redis_service,
        frame_skip: int = 2,
        polygon_zone: Optional[List[Dict]] = None,
        on_detection: Optional[Callable] = None,
    ):
        self.camera_id      = camera_id
        self.rtsp_url       = rtsp_url
        self.alpr_core      = alpr_core
        self.redis_service  = redis_service
        self.frame_skip     = max(1, frame_skip)
        self.polygon_zone   = polygon_zone  # List[{"x": int, "y": int}]
        self.on_detection   = on_detection

        self.capture: Optional[cv2.VideoCapture] = None
        self.is_running    = False
        self.frame_count   = 0
        self.detection_count = 0

        self.stats: Dict = {
            "frames_processed": 0,
            "frames_skipped":   0,
            "detections":       0,
            "errors":           0,
            "fps":              0.0,
            "started_at":       None,
            "rtsp_url":         rtsp_url,
        }

    # ── Zone management ───────────────────────────────────────────────────────

    def update_zone(self, polygon_zone: Optional[List[Dict]]):
        """Hot-update the detection polygon zone without restarting the stream."""
        self.polygon_zone = polygon_zone
        logger.info(
            "Zone updated for camera %s: %s points",
            self.camera_id,
            len(polygon_zone) if polygon_zone else 0,
        )

    # ── Public API ────────────────────────────────────────────────────────────

    async def start(self):
        if self.is_running:
            logger.warning("Stream %s already running", self.camera_id)
            return

        logger.info("▶ Starting H.265/RTSP stream: %s  url=%s", self.camera_id, self.rtsp_url)
        self.is_running = True
        self.stats["started_at"] = datetime.now().isoformat()

        try:
            await self.redis_service.set_camera_active(self.camera_id, True)
            await self._process_loop()
        finally:
            self.is_running = False
            self._release()
            try:
                await self.redis_service.set_camera_active(self.camera_id, False)
            except Exception:
                pass
            logger.info("⏹ Stream %s stopped.", self.camera_id)

    async def stop(self):
        logger.info("Stopping stream: %s", self.camera_id)
        self.is_running = False
        self._release()
        try:
            await self.redis_service.set_camera_active(self.camera_id, False)
        except Exception:
            pass

    def get_stats(self) -> Dict:
        started = self.stats.get("started_at")
        runtime = None
        if started:
            try:
                dt = datetime.fromisoformat(started)
                runtime = (datetime.now() - dt).total_seconds()
            except Exception:
                pass
        return {
            **self.stats,
            "camera_id":       self.camera_id,
            "is_running":      self.is_running,
            "frame_count":     self.frame_count,
            "detection_count": self.detection_count,
            "runtime_seconds": runtime,
            "has_zone":        bool(self.polygon_zone),
            "zone_points":     len(self.polygon_zone) if self.polygon_zone else 0,
        }

    async def get_current_frame(self) -> Optional[np.ndarray]:
        if not self.capture or not self.is_running:
            return None
        ret, frame = await _read_frame_async(self.capture)
        return frame if ret else None

    # ── Internals ─────────────────────────────────────────────────────────────

    def _release(self):
        if self.capture:
            try:
                self.capture.release()
            except Exception:
                pass
            self.capture = None

    async def _open_with_backoff(self) -> bool:
        delay   = RECONNECT_BASE_DELAY
        attempt = 0

        while self.is_running:
            attempt += 1
            logger.info("🔌 Connecting to %s (attempt %d)  url=%s", self.camera_id, attempt, self.rtsp_url)
            self._release()

            cap = await _open_capture_async(self.rtsp_url)
            if cap is not None:
                self.capture = cap
                w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = cap.get(cv2.CAP_PROP_FPS) or 0
                logger.info(
                    "✅ Connected %s — %dx%d @ %.1f fps",
                    self.camera_id, w, h, fps,
                )
                return True

            logger.warning("❌ Cannot connect %s — retry in %.0fs", self.camera_id, delay)
            await asyncio.sleep(delay)
            delay = min(delay * 1.5, RECONNECT_MAX_DELAY)

        return False

    async def _process_loop(self):
        logger.info("Processing loop started: %s", self.camera_id)

        consecutive_fails = 0
        last_fps_time     = datetime.now()
        fps_counter       = 0

        if not await self._open_with_backoff():
            return

        while self.is_running:
            try:
                ret, frame = await _read_frame_async(self.capture)

                # ── Bad frame ─────────────────────────────────────────────
                if not ret or frame is None:
                    consecutive_fails += 1
                    self.stats["errors"] += 1

                    if consecutive_fails == 1:
                        logger.warning(
                            "Frame read failed on %s (reconnect after %d fails)",
                            self.camera_id, MAX_CONSECUTIVE_FAILS,
                        )

                    if consecutive_fails < MAX_CONSECUTIVE_FAILS:
                        await asyncio.sleep(min(0.5 * consecutive_fails, 5.0))
                        continue

                    logger.error(
                        "%d consecutive failures on %s — reconnecting…",
                        consecutive_fails, self.camera_id,
                    )
                    consecutive_fails = 0
                    if not await self._open_with_backoff():
                        break
                    continue

                # ── Good frame ────────────────────────────────────────────
                consecutive_fails  = 0
                self.frame_count  += 1

                if self.frame_count % self.frame_skip != 0:
                    self.stats["frames_skipped"] += 1
                    await asyncio.sleep(0)
                    continue

                await self._process_frame(frame)
                self.stats["frames_processed"] += 1
                fps_counter += 1

                now     = datetime.now()
                elapsed = (now - last_fps_time).total_seconds()
                if elapsed >= 1.0:
                    self.stats["fps"] = fps_counter / elapsed
                    fps_counter   = 0
                    last_fps_time = now

                await asyncio.sleep(0)

            except asyncio.CancelledError:
                logger.info("Loop cancelled: %s", self.camera_id)
                break
            except Exception as exc:
                logger.error("Unexpected error in %s: %s", self.camera_id, exc, exc_info=True)
                self.stats["errors"] += 1
                await asyncio.sleep(1.0)

        logger.info("Processing loop ended: %s", self.camera_id)

    async def _process_frame(self, frame: np.ndarray):
        try:
            results = await self.alpr_core.process_frame(
                frame=frame,
                frame_id=self.frame_count,
                camera_id=self.camera_id,
                polygon_zone=self.polygon_zone,
                redis_client=self.redis_service.client,
            )
            if results:
                self.detection_count += len(results)
                self.stats["detections"] += len(results)

                for r in results:
                    await self.redis_service.publish_detection(self.camera_id, r)

                if self.on_detection:
                    await self.on_detection(self.camera_id, results)

                logger.info(
                    "Camera %s: %d plate(s) (frame %d)",
                    self.camera_id, len(results), self.frame_count,
                )
        except Exception as exc:
            logger.error("Frame processing error %s: %s", self.camera_id, exc, exc_info=True)
            self.stats["errors"] += 1


# ── Manager ──────────────────────────────────────────────────────────────────

class RTSPStreamManager:
    """
    Manages multiple RTSP camera processors.
    Auto-starts cameras registered in DB or .env on backend startup.
    """

    def __init__(self, alpr_core, redis_service, on_detection: Optional[Callable] = None):
        self.alpr_core     = alpr_core
        self.redis_service = redis_service
        self.on_detection  = on_detection
        self.processors: Dict[str, RTSPStreamProcessor] = {}
        self.tasks:      Dict[str, asyncio.Task]         = {}

    async def add_camera(
        self,
        camera_id: str,
        rtsp_url: str,
        frame_skip: int = 2,
        polygon_zone: Optional[List[Dict]] = None,
    ) -> bool:
        if camera_id in self.processors:
            logger.warning("Camera %s already registered", camera_id)
            return False
        try:
            self.processors[camera_id] = RTSPStreamProcessor(
                camera_id=camera_id,
                rtsp_url=rtsp_url,
                alpr_core=self.alpr_core,
                redis_service=self.redis_service,
                frame_skip=frame_skip,
                polygon_zone=polygon_zone,
                on_detection=self.on_detection,
            )
            logger.info("Camera %s registered (zone=%s)", camera_id, bool(polygon_zone))
            return True
        except Exception as exc:
            logger.error("Failed to register %s: %s", camera_id, exc)
            return False

    async def start_camera(self, camera_id: str) -> bool:
        if camera_id not in self.processors:
            logger.error("Camera %s not registered", camera_id)
            return False
        task = self.tasks.get(camera_id)
        if task and not task.done():
            logger.warning("Camera %s task already running", camera_id)
            return False
        self.tasks[camera_id] = asyncio.create_task(
            self.processors[camera_id].start(),
            name=f"rtsp-{camera_id}",
        )
        logger.info("Camera %s task started", camera_id)
        return True

    async def stop_camera(self, camera_id: str) -> bool:
        if camera_id not in self.processors:
            return False
        await self.processors[camera_id].stop()
        task = self.tasks.pop(camera_id, None)
        if task and not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        return True

    async def remove_camera(self, camera_id: str) -> bool:
        if camera_id not in self.processors:
            return False
        await self.stop_camera(camera_id)
        del self.processors[camera_id]
        return True

    async def update_camera_zone(
        self,
        camera_id: str,
        polygon_zone: Optional[List[Dict]],
    ) -> bool:
        """Hot-update polygon zone for a running camera."""
        processor = self.processors.get(camera_id)
        if not processor:
            logger.error("Camera %s not registered — cannot update zone", camera_id)
            return False
        processor.update_zone(polygon_zone)
        return True

    async def start_all(self) -> int:
        started = 0
        for cid in list(self.processors.keys()):
            if await self.start_camera(cid):
                started += 1
        return started

    async def stop_all(self) -> int:
        stopped = 0
        for cid in list(self.processors.keys()):
            if await self.stop_camera(cid):
                stopped += 1
        return stopped

    def get_camera_stats(self, camera_id: str) -> Optional[Dict]:
        p = self.processors.get(camera_id)
        return p.get_stats() if p else None

    def get_all_stats(self) -> Dict[str, Dict]:
        return {cid: p.get_stats() for cid, p in self.processors.items()}

    def get_active_cameras(self) -> List[str]:
        return [cid for cid, p in self.processors.items() if p.is_running]

    def is_camera_running(self, camera_id: str) -> bool:
        p = self.processors.get(camera_id)
        return p.is_running if p else False

    async def get_camera_frame(self, camera_id: str) -> Optional[np.ndarray]:
        p = self.processors.get(camera_id)
        return await p.get_current_frame() if p else None

    async def shutdown(self):
        logger.info("Shutting down RTSP manager…")
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
    _stream_manager = RTSPStreamManager(
        alpr_core=alpr_core,
        redis_service=redis_service,
        on_detection=on_detection,
    )
    logger.info("RTSP stream manager initialized.")
    return _stream_manager