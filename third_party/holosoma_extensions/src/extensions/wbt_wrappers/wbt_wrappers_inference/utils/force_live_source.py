"""Live per-hand F/T reading source for the offboard ``run_policy`` deploy path.

The Python analogue of the browser's ``computeHybridFMeas`` extras-bus feed: it
connects to the ``force_bridge`` WebSocket (default ``ws://localhost:8771``) and
keeps the latest per-hand SENSOR-frame contact force so ``HolosomaWBTPolicy`` can
fill its ``force_meas_local`` (hybrid) / ``force_ext_local`` (force-ext) obs
terms — currently zeroed at deploy — with the real Bota reading each step.

Wire frame (text JSON, emitted by ``force_bridge``):

    {"type": "ft", "seq": int, "t": wall_seconds,
     "left":  [Fx, Fy, Fz],     # SENSOR-frame Newtons, compensated + tared (or null)
     "right": [Fx, Fy, Fz]}     # (or null when that hand has no sensor)

The values are the gravity/inertia-COMPENSATED, tared wrench in the Bota SENSOR
frame; the policy rotates sensor→wrist→pelvis itself (``force_deploy_math``), so
the bridge is frame-agnostic and stays a thin sensor tap.

SAFETY — the sign is INVERTED vs ``ViveLiveSource``: a stale/absent hand-command
returns **zeros**, not ``None``. A zero force reading is the physically correct
"no contact" state (the force obs was zero at deploy before this existed), so a
dropped stream degrades to pure position tracking — NOT a held stale force that
would keep the servo pushing. ``latest()`` returns ``(left3, right3)`` with zeros
for any hand that has never reported or is stale; a hand that is present but
explicitly ``null`` in the frame (no sensor on that side) also reads zeros.

Deliberately depends only on ``numpy`` + ``websockets`` (no holosoma / mujoco /
bota at runtime), runs its reader on a daemon thread with its own asyncio loop,
and is entirely opt-in: ``HolosomaWBTPolicy`` constructs it only when
``HOLOSOMA_FORCE_WS_URL`` is set, so the default (force-zero) path is
byte-identical when off.
"""

from __future__ import annotations

import asyncio
import json
import threading
import time

import numpy as np
from loguru import logger


class ForceLiveSource:
    """Background WebSocket client holding the latest per-hand sensor-frame force.

    Args:
        ws_url: bridge WebSocket URL (e.g. ``ws://localhost:8771``).
        stale_timeout: seconds; if the newest message is older than this,
            ``latest()`` returns zeros (safe "no contact"), NOT the last force.
            ``0`` disables the staleness gate (not recommended on hardware).
    """

    def __init__(self, ws_url: str, stale_timeout: float = 0.25):
        self._ws_url = ws_url
        self._stale_timeout = float(stale_timeout)
        self._lock = threading.Lock()
        self._left = np.zeros(3, dtype=np.float32)  # sensor-frame N
        self._right = np.zeros(3, dtype=np.float32)
        self._last_t: float = 0.0
        self._seq: int = -1
        self._have_data = False
        self._connected = False
        self._stop = threading.Event()
        self._tare_pending = threading.Event()  # set by request_tare(), consumed in the reader loop
        self._stale_logged = False
        self._thread = threading.Thread(target=self._run, name="force_live_ws", daemon=True)
        self._thread.start()
        logger.info(f"ForceLiveSource: connecting to {ws_url} (stale_timeout={stale_timeout}s)")

    # --- public API -------------------------------------------------------
    def latest(self) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(left3, right3)`` sensor-frame forces (float32).

        Zeros for any hand not yet reported, and zeros for BOTH when the stream
        is stale / never connected. Never returns ``None`` — a missing force is
        physically "no contact", so the caller collapses to position tracking.
        """
        with self._lock:
            if not self._have_data:
                return np.zeros(3, dtype=np.float32), np.zeros(3, dtype=np.float32)
            if self._stale_timeout > 0 and (time.time() - self._last_t) > self._stale_timeout:
                if not self._stale_logged:
                    logger.warning(
                        f"ForceLiveSource: reading STALE (>{self._stale_timeout * 1000:.0f} ms) "
                        "— treating as ZERO force (no contact)"
                    )
                    self._stale_logged = True
                return np.zeros(3, dtype=np.float32), np.zeros(3, dtype=np.float32)
            if self._stale_logged:
                logger.info("ForceLiveSource: live reading fresh again")
                self._stale_logged = False
            return self._left.copy(), self._right.copy()

    def request_tare(self) -> None:
        """Ask the bridge to re-tare BOTH sensors (fire in FREE SPACE).

        Thread-safe: sets a flag the reader loop consumes on its next cycle,
        sending ``{"cmd":"tare"}`` over the live WS. A no-op until connected (the
        flag persists, so a tare pressed while briefly disconnected fires on
        reconnect). Wired to the joystick tare button in the policy.
        """
        self._tare_pending.set()

    def shutdown(self) -> None:
        self._stop.set()

    # --- background reader ------------------------------------------------
    def _run(self) -> None:
        try:
            asyncio.run(self._reader())
        except Exception as exc:  # noqa: BLE001
            logger.error(f"ForceLiveSource reader crashed: {exc!r}")

    async def _reader(self) -> None:
        # Imported here (not at module top) so the dependency is only needed on
        # the live path. A MISSING dependency is FATAL — do not silently run with
        # zero force while logging "FORCE ENABLED" (operator would think the
        # sensor is feeding the policy but it is not).
        try:
            import websockets
            from websockets.exceptions import InvalidURI
        except ImportError as exc:
            logger.error(
                f"ForceLiveSource: `websockets` not installed ({exc!r}) — live F/T "
                "CANNOT run. Install websockets>=11. Aborting the live source."
            )
            self._stop.set()
            return

        backoff = 0.5
        while not self._stop.is_set():
            try:
                async with websockets.connect(self._ws_url, max_size=None) as ws:
                    if not self._connected:
                        logger.info(f"ForceLiveSource: connected to {self._ws_url}")
                        self._connected = True
                    backoff = 0.5
                    while not self._stop.is_set():
                        # Forward a pending tare request to the bridge (the joystick
                        # tare button sets the flag; consumed here on the WS thread).
                        if self._tare_pending.is_set():
                            self._tare_pending.clear()
                            await ws.send(json.dumps({"cmd": "tare"}))
                            logger.info("ForceLiveSource: sent tare request to the bridge")
                        raw = await asyncio.wait_for(ws.recv(), timeout=1.0)
                        self._ingest(raw)
            except asyncio.TimeoutError:
                continue  # no message in the window; loop and re-check _stop
            except InvalidURI as exc:
                logger.error(f"ForceLiveSource: invalid WS URL {self._ws_url!r} ({exc!r}) — stopping live source.")
                self._stop.set()
                return
            except Exception as exc:  # noqa: BLE001 — reconnect on transient drops
                self._connected = False
                if not self._stop.is_set():
                    logger.warning(f"ForceLiveSource: WS error ({exc!r}); reconnecting in {backoff:.1f}s")
                    await asyncio.sleep(backoff)
                    backoff = min(backoff * 2, 5.0)

    def _ingest(self, raw: str | bytes) -> None:
        try:
            msg = json.loads(raw)
        except (ValueError, TypeError):
            return
        if msg.get("type") != "ft":
            return  # ignore control/ack frames

        def _vec3(v) -> np.ndarray | None:
            if not isinstance(v, list) or len(v) != 3:
                return None
            try:
                arr = np.asarray([float(v[0]), float(v[1]), float(v[2])], dtype=np.float32)
            except (ValueError, TypeError):
                return None
            # Drop NaN/inf frames: JSON round-trips NaN/Inf (json.dumps emits them,
            # json.loads accepts them), and a non-finite force would sail through
            # np.clip in the obs normalizer straight to a motor command. Returning
            # None lets the stream go stale→ZERO (safe "no contact") instead.
            if not np.all(np.isfinite(arr)):
                return None
            return arr

        left = _vec3(msg.get("left"))
        right = _vec3(msg.get("right"))
        if left is None and right is None:
            return  # nothing usable in this frame
        with self._lock:
            # A hand explicitly null (no sensor that side) reads zero; a hand
            # present overwrites its cached value.
            self._left = left if left is not None else np.zeros(3, dtype=np.float32)
            self._right = right if right is not None else np.zeros(3, dtype=np.float32)
            self._last_t = time.time()
            self._seq = int(msg.get("seq", self._seq + 1))
            self._have_data = True
