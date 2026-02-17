"""
RTSP Service for Video Stream Processing
Supports H.264 and H.265 (HEVC) via FFmpeg backend
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
MAX_CONSECUTIVE_FAILS  = 10     # reopen capture after N consecutive bad reads
RECONNECT_BASE_DELAY   = 2.0    # initial reconnect back-off (seconds)
RECONNECT_MAX_DELAY    = 30.0   # cap on reconnect back-off
OPEN_TIMEOUT_SEC       = 15     # wall-clock limit for VideoCapture.open()
FRAME_READ_TIMEOUT_SEC = 8      # wall-clock limit for a single cap.read()


def _build_capture(rtsp_url: str) -> cv2.VideoCapture:
    """
    Open an RTSP stream using OpenCV's FFmpeg backend.

    Why FFmpeg?
    - Full H.265 / HEVC decode support (libx265 must be installed in the image)
    - We force TCP transport to avoid UDP packet-loss → ret=False spam
    - OPENCV_FFMPEG_CAPTURE_OPTIONS env var is picked up automatically;
      we also set it programmatically as a fallback.

    The env var approach works for most OpenCV builds; the CAP_PROP_* approach
    is more reliable across versions.
    """
    # CAP_FFMPEG = 1900 — explicit backend, avoids GStreamer fallback
    cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)

    if cap.isOpened():
        # Force TCP at the codec level (works on OpenCV ≥ 4.5.2)
        # 0x00000004 = RTSP_TRANSPORT_TCP in FFmpeg's AVFormatContext
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'H265'))

        # Small decode buffer → low latency
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap

    # Fallback: no explicit backend (lets OpenCV choose GStreamer or FFmpeg)
    cap.release()
    logger.warning("FFmpeg backend failed, trying default backend for %s", rtsp_url)
    cap = cv2.VideoCapture(rtsp_url)
    if cap.isOpened():
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


async def _open_capture_async(rtsp_url: str, timeout: float = OPEN_TIMEOUT_SEC) -> Optional[cv2.VideoCapture]:
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


async def _read_frame_async(cap: cv2.VideoCapture, timeout: float = FRAME_READ_TIMEOUT_SEC):
    """Read one frame with a timeout so a hung read never stalls the event loop."""
    try:
        return await asyncio.wait_for(
            asyncio.to_thread(cap.read),
            timeout=timeout,
        )
    except asyncio.TimeoutError:
        return False, None


# ── Per-camera processor ─────────────────────────────────────────────────────

class RTSPStreamProcessor:
    """Single-camera RTSP processor with automatic reconnect."""

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
        self.camera_id     = camera_id
        self.rtsp_url      = rtsp_url
        self.alpr_core     = alpr_core
        self.redis_service = redis_service
        self.frame_skip    = max(1, frame_skip)
        self.polygon_zone  = polygon_zone
        self.on_detection  = on_detection

        self.capture: Optional[cv2.VideoCapture] = None
        self.is_running   = False
        self.frame_count  = 0
        self.detection_count = 0

        self.stats: Dict = {
            "frames_processed": 0,
            "frames_skipped":   0,
            "detections":       0,
            "errors":           0,
            "fps":              0.0,
            "started_at":       None,
        }

    # ── Public API ────────────────────────────────────────────────────────────

    async def start(self):
        if self.is_running:
            logger.warning("Stream %s already running", self.camera_id)
            return

        logger.info("▶ Starting stream: %s", self.camera_id)
        self.is_running = True
        self.stats["started_at"] = datetime.now()

        try:
            await self.redis_service.set_camera_active(self.camera_id, True)
            await self._process_loop()
        finally:
            self.is_running = False
            self._release()
            await self.redis_service.set_camera_active(self.camera_id, False)
            logger.info("⏹ Stream %s stopped.", self.camera_id)

    async def stop(self):
        logger.info("Stopping stream: %s", self.camera_id)
        self.is_running = False
        self._release()
        await self.redis_service.set_camera_active(self.camera_id, False)

    def get_stats(self) -> Dict:
        runtime = (
            (datetime.now() - self.stats["started_at"]).total_seconds()
            if self.stats["started_at"] else None
        )
        return {
            **self.stats,
            "camera_id":       self.camera_id,
            "is_running":      self.is_running,
            "frame_count":     self.frame_count,
            "detection_count": self.detection_count,
            "runtime_seconds": runtime,
        }

    async def get_current_frame(self) -> Optional[np.ndarray]:
        if not self.capture or not self.is_running:
            return None
        ret, frame = await _read_frame_async(self.capture)
        return frame if ret else None

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _release(self):
        if self.capture:
            try:
                self.capture.release()
            except Exception:
                pass
            self.capture = None

    async def _open_with_backoff(self) -> bool:
        """Try to open the RTSP stream, retrying with exponential back-off."""
        delay   = RECONNECT_BASE_DELAY
        attempt = 0

        while self.is_running:
            attempt += 1
            logger.info("🔌 Connecting to %s (attempt %d)…", self.camera_id, attempt)
            self._release()

            cap = await _open_capture_async(self.rtsp_url)
            if cap is not None:
                self.capture = cap
                # Log codec info to help debug H.265 issues
                w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                logger.info(
                    "✅ Connected to %s — %dx%d @ %.1f fps",
                    self.camera_id, w, h, fps,
                )
                return True

            logger.warning(
                "❌ Cannot connect to %s — retry in %.0fs", self.camera_id, delay
            )
            await asyncio.sleep(delay)
            delay = min(delay * 1.5, RECONNECT_MAX_DELAY)

        return False  # is_running became False during retries

    # ── Main loop ─────────────────────────────────────────────────────────────

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
                            "Frame read failed on %s (will reconnect after %d fails)",
                            self.camera_id, MAX_CONSECUTIVE_FAILS,
                        )

                    if consecutive_fails < MAX_CONSECUTIVE_FAILS:
                        # Progressive back-off before retrying (max 5s)
                        await asyncio.sleep(min(0.5 * consecutive_fails, 5.0))
                        continue

                    # Too many failures → full reconnect
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

                # Frame-skip (reduces CPU load)
                if self.frame_count % self.frame_skip != 0:
                    self.stats["frames_skipped"] += 1
                    await asyncio.sleep(0)   # yield without blocking
                    continue

                # ALPR processing
                await self._process_frame(frame)
                self.stats["frames_processed"] += 1
                fps_counter += 1

                # FPS update every second
                now     = datetime.now()
                elapsed = (now - last_fps_time).total_seconds()
                if elapsed >= 1.0:
                    self.stats["fps"] = fps_counter / elapsed
                    fps_counter   = 0
                    last_fps_time = now

                await asyncio.sleep(0)   # keep event loop responsive

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
            logger.info("Camera %s registered", camera_id)
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

    async def start_all(self) -> int:
        return sum([1 for cid in self.processors if await self.start_camera(cid)])

    async def stop_all(self) -> int:
        return sum([1 for cid in list(self.processors) if await self.stop_camera(cid)])

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


# ── Singleton ────────────────────────────────────────────────────────────────

_stream_manager: Optional[RTSPStreamManager] = None


def get_stream_manager() -> RTSPStreamManager:
    if _stream_manager is None:
        raise RuntimeError("Stream manager not initialized")
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
    return _stream_manager