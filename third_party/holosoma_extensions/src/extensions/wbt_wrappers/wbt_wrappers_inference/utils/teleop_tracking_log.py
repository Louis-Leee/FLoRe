"""Capture + math helpers for quantifying VIVE-teleop tracking error.

This is the recording half of the "how well does the robot's hand follow my
tracker?" tool. ``HolosomaWBTPolicy`` (when ``HOLOSOMA_TELEOP_TRACK_LOG`` is set)
appends one row per inference tick holding two hand poses, BOTH expressed in the
**robot's own (pelvis) frame**:

* ``cmd`` — the ``vr_2point_pos`` / ``vr_2point_orn`` the policy actually saw this
  tick: the tracker-driven hand TARGET after the relative/clutch mapping + the
  per-tick slew limiter. This is "what the tracker asked the hand to do".
* ``exec`` — the SAME ``_compute_vr_fk`` run on the robot's MEASURED joint
  positions: where the hand actually ended up after the policy + low-level PD
  loop. This is "what the controller actually executed".

The apples-to-apples is exact because ``_compute_vr_fk`` is **root-invariant** —
its output is the wrist pose relative to the pelvis, in the pelvis frame, a pure
function of joint angles (the floating-base pose cancels in
``quat_rotate_inverse(anchor, vr_pos_w - anchor_pos)``). So commanding it with
the reference joints (``cmd``) vs the measured joints (``exec``) yields two poses
in the identical frame; their difference is the tracking error the user wants.

``live`` (the raw PRE-clutch absolute bridge target, pelvis frame — what the
bridge sent, BEFORE the policy's relative/clutch + slew mapping) is also stored
for diagnostics but is NOT the headline comparison — ``cmd`` is this same target
AFTER the clutch + slew shaping, so ``cmd`` vs ``exec`` is the operationally
meaningful "what the hand was asked to do vs what it did". ``live`` is recorded
only on armed ticks (it is the live sample the obs actually consumed); disarmed
ticks store an all-NaN ``live`` row.

Deliberately numpy-only (no matplotlib / holosoma at import) so the recording
side stays a thin, dependency-light add-on to the realtime policy; the plotting
lives in ``analyze_teleop_tracking.py`` and is imported only offline.
"""

from __future__ import annotations

import numpy as np
from loguru import logger

# rot6d layout: [L_rot6d(6), R_rot6d(6)] = 12; position: [L_xyz(3), R_xyz(3)] = 6.
HAND_LABELS = ("left", "right")
POS_DIM = 6
ORN_DIM = 12


def rot6d_to_R(r6: np.ndarray) -> np.ndarray:
    """Rebuild a 3x3 rotation matrix from a 6D rep (first two columns of R).

    Gram-Schmidt orthonormalization — the exact inverse of the
    ``matrix_from_quat(q)[..., :2].reshape(-1)`` rep that
    ``HolosomaWBTPolicy._compute_vr_fk`` emits, and bit-for-bit the same
    procedure as ``HolosomaWBTPolicy._rot6d_to_R`` (kept duplicated so this
    module has no holosoma import). The producer flattens the first two COLUMNS
    of R in row-major order, i.e. ``[r00, r01, r10, r11, r20, r21]``; ``reshape(3, 2)``
    then puts column 0 of R in ``m[:, 0]`` and column 1 in ``m[:, 1]`` (which is
    exactly what this function consumes).
    """
    m = np.asarray(r6, dtype=np.float64).reshape(3, 2)
    b0 = m[:, 0]
    n0 = np.linalg.norm(b0)
    c0 = b0 / n0 if n0 > 1e-8 else np.array([1.0, 0.0, 0.0])
    b1 = m[:, 1]
    c1 = b1 - (c0 @ b1) * c0
    n1 = np.linalg.norm(c1)
    c1 = c1 / n1 if n1 > 1e-8 else np.array([0.0, 1.0, 0.0])
    c2 = np.cross(c0, c1)
    return np.stack([c0, c1, c2], axis=1)


def geodesic_angle_deg(r6_a: np.ndarray, r6_b: np.ndarray) -> float:
    """Geodesic (shortest-arc) angle in degrees between two rot6d orientations.

    ``angle = arccos((trace(R_aᵀ R_b) - 1) / 2)``, clamped for numerical safety.
    Frame-agnostic: both inputs must already be in the same frame (they are —
    both pelvis-local rot6d).
    """
    ra = rot6d_to_R(r6_a)
    rb = rot6d_to_R(r6_b)
    cos = (np.trace(ra.T @ rb) - 1.0) * 0.5
    cos = float(np.clip(cos, -1.0, 1.0))
    return float(np.degrees(np.arccos(cos)))


class TeleopTrackingLog:
    """Per-tick accumulator of (commanded, executed, live) hand poses.

    All arrays are pelvis-frame. ``record`` is called once per inference tick;
    ``save`` writes a single NPZ and is idempotent (safe to call from the stop
    handler AND the atexit/​signal path without double-writing).

    Memory: each tick appends ~8 small ndarrays (~1 KB), so the in-RAM buffer
    grows ~3 MB/min (~180 MB/hr) at 50 Hz. This is a bounded diagnostic tool —
    record a teleop maneuver (e.g. a few circles), not an all-day session. Pass
    ``max_ticks`` (env ``HOLOSOMA_TELEOP_TRACK_MAX_TICKS``) to hard-cap the buffer;
    once reached, further ticks are dropped with a one-time warning (the data up
    to the cap still saves).
    """

    def __init__(
        self,
        path: str,
        rl_rate: float = 50.0,
        model_path: str = "",
        motion_path: str = "",
        relative: bool = True,
        orientation: bool = False,
        max_ticks: int = 0,
    ) -> None:
        self._path = path
        self._rl_rate = float(rl_rate)
        self._model_path = model_path
        self._motion_path = motion_path
        self._relative = bool(relative)
        self._orientation = bool(orientation)
        self._max_ticks = int(max_ticks) if max_ticks and max_ticks > 0 else 0
        self._capped_logged = False
        self._saved = False
        # Parallel lists (cheap appends; stacked once on save).
        self._frame: list[int] = []
        self._t: list[float] = []
        self._cmd_pos: list[np.ndarray] = []
        self._cmd_orn: list[np.ndarray] = []
        self._exec_pos: list[np.ndarray] = []
        self._exec_orn: list[np.ndarray] = []
        self._live_pos: list[np.ndarray] = []
        self._live_orn: list[np.ndarray] = []
        self._armed: list[bool] = []
        logger.info(f"TeleopTrackingLog: recording cmd-vs-exec hand poses → {path}")

    def __len__(self) -> int:
        return len(self._frame)

    @staticmethod
    def _as(vec, dim: int) -> np.ndarray:
        """Coerce to a fresh float32 (dim,) array; None → all-NaN (no data)."""
        if vec is None:
            return np.full(dim, np.nan, dtype=np.float32)
        return np.asarray(vec, dtype=np.float32).reshape(-1)[:dim].copy()

    def record(
        self,
        frame: int,
        t: float,
        cmd_pos,
        cmd_orn,
        exec_pos,
        exec_orn,
        live_pos=None,
        live_orn=None,
        armed: bool = False,
    ) -> None:
        """Append one tick. ``cmd``/``exec`` are required; ``live`` may be None."""
        if self._max_ticks and len(self._frame) >= self._max_ticks:
            if not self._capped_logged:
                logger.warning(
                    f"TeleopTrackingLog: reached max_ticks={self._max_ticks} — dropping further ticks "
                    "(data up to the cap is preserved)."
                )
                self._capped_logged = True
            return
        self._frame.append(int(frame))
        self._t.append(float(t))
        self._cmd_pos.append(self._as(cmd_pos, POS_DIM))
        self._cmd_orn.append(self._as(cmd_orn, ORN_DIM))
        self._exec_pos.append(self._as(exec_pos, POS_DIM))
        self._exec_orn.append(self._as(exec_orn, ORN_DIM))
        self._live_pos.append(self._as(live_pos, POS_DIM))
        self._live_orn.append(self._as(live_orn, ORN_DIM))
        self._armed.append(bool(armed))
        # New data invalidates a prior save, so a later save() (e.g. a second
        # START→STOP cycle followed by atexit) re-writes the full set instead of
        # no-op'ing on the stale flag.
        self._saved = False

    def save(self) -> None:
        """Write the NPZ once. No-op on empty or already-saved logs.

        Sets the idempotency guard BEFORE the (non-re-entrant) ``np.savez`` and
        rolls it back only on failure, so a signal arriving mid-write (the
        SIGINT/SIGTERM handler also calls this) no-ops instead of launching a
        second concurrent ``np.savez`` to the same path.
        """
        if self._saved or not self._frame:
            return
        self._saved = True
        try:
            np.savez(
                self._path,
                frame=np.asarray(self._frame, dtype=np.int64),
                t=np.asarray(self._t, dtype=np.float64),
                cmd_pos=np.stack(self._cmd_pos),
                cmd_orn=np.stack(self._cmd_orn),
                exec_pos=np.stack(self._exec_pos),
                exec_orn=np.stack(self._exec_orn),
                live_pos=np.stack(self._live_pos),
                live_orn=np.stack(self._live_orn),
                armed=np.asarray(self._armed, dtype=bool),
                rl_rate=np.asarray(self._rl_rate, dtype=np.float64),
                model_path=np.asarray(self._model_path),
                motion_path=np.asarray(self._motion_path),
                relative=np.asarray(self._relative, dtype=bool),
                orientation=np.asarray(self._orientation, dtype=bool),
            )
        except BaseException:  # noqa: BLE001 — re-raise after clearing the guard so a retry can re-save
            self._saved = False
            raise
        logger.info(f"TeleopTrackingLog: saved {len(self._frame)} ticks → {self._path}")
