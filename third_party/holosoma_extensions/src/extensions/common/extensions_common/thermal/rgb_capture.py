"""RGB video recorder for test documentation.

Records continuously from a USB camera (OpenCV) for the duration of a test.
"""

from __future__ import annotations

import logging
import threading
import time

logger = logging.getLogger(__name__)


class RGBVideoRecorder:
    """Records video from a USB camera to an MP4/AVI file.

    Parameters
    ----------
    output_path : str
        Output file path (e.g., "results/test_video.avi").
    device : int
        Camera device index (default 0).
    fps : float
        Recording frame rate (default 30).
    resolution : tuple[int, int] | None
        (width, height) to request from camera. None = camera default.
    fourcc : str
        FourCC codec string (default "MJPG" for broad compatibility).
    """

    def __init__(
        self,
        output_path: str,
        device: int = 0,
        fps: float = 30.0,
        resolution: tuple[int, int] | None = None,
        fourcc: str = "MJPG",
    ):
        self._output_path = output_path
        self._device = device
        self._fps = fps
        self._resolution = resolution
        self._fourcc = fourcc
        self._running = False
        self._thread: threading.Thread | None = None
        self._frame_count = 0

    def start(self) -> None:
        """Start background recording thread."""
        if self._running:
            return
        self._running = True
        self._frame_count = 0
        self._thread = threading.Thread(target=self._run_loop, daemon=True, name="rgb-recorder")
        self._thread.start()
        logger.info("RGB recording started → %s (device=%d, fps=%.0f)", self._output_path, self._device, self._fps)

    def stop(self) -> int:
        """Stop recording. Returns the number of frames written."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=10.0)
            self._thread = None
        logger.info("RGB recording stopped: %d frames → %s", self._frame_count, self._output_path)
        return self._frame_count

    @property
    def frame_count(self) -> int:
        return self._frame_count

    def _run_loop(self) -> None:
        import cv2

        cap = cv2.VideoCapture(self._device)
        if not cap.isOpened():
            logger.error("Failed to open camera device %d", self._device)
            return

        if self._resolution is not None:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._resolution[0])
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._resolution[1])

        # Read one frame to get actual resolution
        ret, frame = cap.read()
        if not ret:
            logger.error("Failed to read initial frame from camera %d", self._device)
            cap.release()
            return

        h, w = frame.shape[:2]
        codec = cv2.VideoWriter_fourcc(*self._fourcc)
        writer = cv2.VideoWriter(self._output_path, codec, self._fps, (w, h))
        if not writer.isOpened():
            logger.error("Failed to open video writer: %s", self._output_path)
            cap.release()
            return

        logger.info("Recording %dx%d @ %.0f fps", w, h, self._fps)
        writer.write(frame)
        self._frame_count = 1

        interval = 1.0 / self._fps
        try:
            while self._running:
                ret, frame = cap.read()
                if not ret:
                    logger.warning("Camera read failed at frame %d", self._frame_count)
                    break
                writer.write(frame)
                self._frame_count += 1
                time.sleep(interval)
        finally:
            writer.release()
            cap.release()
