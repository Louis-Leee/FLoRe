"""IR thermal camera capture scheduler.

The actual camera hardware abstraction is injected via a callable, keeping
this module hardware-agnostic.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from typing import Callable

import numpy as np

logger = logging.getLogger(__name__)


class IRCameraCapture:
    """Schedules periodic IR thermal image captures in a background thread.

    Parameters
    ----------
    capture_fn : Callable[[], np.ndarray]
        Function that captures and returns a thermal image (e.g., FLIR SDK
        call, ROS2 topic snapshot, or mock). Should return an ndarray
        (typically H x W float32 in Celsius).
    interval_s : float
        Seconds between captures (default 60).
    """

    def __init__(self, capture_fn: Callable[[], np.ndarray], interval_s: float = 60.0, max_captures: int = 0):
        self._capture_fn = capture_fn
        self._interval_s = interval_s
        self._max_captures = max_captures
        self._running = False
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        maxlen = max_captures if max_captures > 0 else None
        self._captures: deque[tuple[float, np.ndarray]] = deque(maxlen=maxlen)

    def start(self) -> None:
        """Start background capture thread."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True, name="ir-capture")
        self._thread.start()
        logger.info("IR capture started (interval=%.0fs)", self._interval_s)

    def stop(self) -> list[tuple[float, np.ndarray]]:
        """Stop capturing and return all ``(monotonic_timestamp, image)`` pairs."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=10.0)
            self._thread = None
        with self._lock:
            return list(self._captures)

    def get_captures(self) -> list[tuple[float, np.ndarray]]:
        """Return captures collected so far (non-destructive)."""
        with self._lock:
            return list(self._captures)

    @property
    def num_captures(self) -> int:
        with self._lock:
            return len(self._captures)

    def _run_loop(self) -> None:
        # Capture immediately on start, then every interval_s
        while self._running:
            try:
                img = self._capture_fn()
                now = time.monotonic()
                with self._lock:
                    self._captures.append((now, img))
                logger.debug("IR capture #%d acquired", len(self._captures))
            except Exception:
                logger.exception("IR capture failed")
            # Sleep in small increments so stop() is responsive
            deadline = time.monotonic() + self._interval_s
            while self._running and time.monotonic() < deadline:
                time.sleep(0.5)
