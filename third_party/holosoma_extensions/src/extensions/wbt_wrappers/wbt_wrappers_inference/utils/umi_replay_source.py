"""Recorded-take REPLAY source for the offboard ``run_policy`` VIVE-teleop path.

This is the OFFLINE twin of ``ViveLiveSource``: instead of streaming a live
operator's trackers over a WebSocket, it plays back a right-hand VIVE trajectory
recorded by the UMI collection rig (``UMI_collection/run_replay_viz.sh`` reads
the SAME ``take_*`` dirs). It exposes the IDENTICAL ``latest() -> (pos[6],
orn[12])`` interface so ``HolosomaWBTPolicy`` can drop it into ``self._vive_live``
and drive the robot's right hand through the EXACT SAME relative/clutch obs
override the live path uses — no new obs term, no physics injection, byte-
identical when off.

Why the clutch reuse is CORRECT (not a coincidence):

* A UMI ``take`` stores one right-hand tracker pose stream ``tracker.npy`` in a
  robot-yaw-aligned world frame (``recording_format.TRACKER_COLS`` =
  ``t seq x y z qw qx qy qz stale``, quaternion WXYZ). The UMI replay
  (``replay_raw_viz.py`` → ``g1_right_hand_ik.py``) drives the arm by
  ``p_target = p_home + reach·(p_cur − p_ref)`` and ``R_target = ΔR_world·R_home``
  with ``ΔR_world = R_cur·R_ref⁻¹`` — i.e. it maps the tracker DISPLACEMENT SINCE
  FRAME 0 onto the hand.
* Our clutch computes ``target = ref + (live − live0)`` for position and
  ``R_target = (R_live·R_live0ᵀ)·R_ref`` for orientation, where the anchor
  ``(live0)`` is snapshotted at arm time. If we arm with the replay PAUSED at
  frame 0 (``live0 == recorded[0]``) and then advance the recording, the delta
  fed to the clutch is exactly ``recorded[k] − recorded[0]`` — the same
  frame-0-relative displacement UMI replays. The constant puck-to-hand lever,
  the world origin, and the absolute tracker orientation all CANCEL in the delta,
  so no puck-mount / no VR_2POINT lever transform is needed.

So the replay is no more off-distribution than live teleop: it feeds the same
pelvis-frame relative delta stream, just pre-recorded instead of live.

Frame conventions:

* ``pos[6]`` = ``[L_xyz, R_xyz]``. UMI records only the RIGHT hand, so the LEFT
  half is held CONSTANT (zeros) across the whole take — a constant value makes
  ``live_L − live0_L == 0`` every tick, so the left hand holds the reference pose
  and never moves. Only the right half carries the recorded trajectory.
* ``orn[12]`` = ``[L_rot6d, R_rot6d]`` where rot6d = the first two COLUMNS of the
  rotation matrix, row-major (``R[:, :2].reshape(-1)``) — bit-for-bit the rep
  ``HolosomaWBTPolicy._compute_vr_fk`` emits and ``_rot6d_to_R`` inverts. Left =
  identity (constant → ΔR_L = I → left wrist holds ref); right = the recorded
  wrist rotation. Orientation only actually moves the wrist when the policy is
  launched with ``HOLOSOMA_VIVE_ORIENTATION=1`` (else the clutch keeps orn = the
  frozen FK ref); we always supply it so the flag works.

Playback timing: ``latest()`` is CLOCK-driven off ``play()`` (real-time replay),
so it is idempotent within a control tick (the policy polls ``latest()`` a few
times per tick — arm gate, obs, recorder — and must get the SAME frame each
time; a per-call frame counter would race and skip). Before ``play()`` it holds
frame 0; after the take ends it holds the last frame (or loops when enabled).

Deliberately depends only on ``numpy`` (no holosoma / websockets / mujoco), so it
is importable in the deploy env and unit-testable with an injected clock.
"""

from __future__ import annotations

import os
import time

import numpy as np
from loguru import logger

# tracker.npy column layout (UMI recording_format.TRACKER_COLS). Kept as a local
# constant so this module has no UMI_collection import dependency.
_COL_T = 0
_COL_X = 2
_COL_QW = 5
_COL_STALE = 9
_N_TRACKER_COLS = 10

_TRACKER_FILE = "tracker.npy"
_TAKE_META_FILE = "take_meta.json"


def _wxyz_to_R(q: np.ndarray) -> np.ndarray:
    """3x3 rotation matrix from a WXYZ unit quaternion (no scipy dep).

    Only used to derive the rot6d rep; since the clutch consumes the delta
    ``R_live·R_live0ᵀ`` the absolute convention cancels — this only needs to be
    internally consistent and a proper rotation.
    """
    w, x, y, z = q
    n = w * w + x * x + y * y + z * z
    if n < 1e-12:
        return np.eye(3)
    s = 2.0 / n
    wx, wy, wz = s * w * x, s * w * y, s * w * z
    xx, xy, xz = s * x * x, s * x * y, s * x * z
    yy, yz, zz = s * y * y, s * y * z, s * z * z
    return np.array(
        [[1.0 - (yy + zz), xy - wz, xz + wy], [xy + wz, 1.0 - (xx + zz), yz - wx], [xz - wy, yz + wx, 1.0 - (xx + yy)]]
    )


def _R_to_rot6d(R: np.ndarray) -> np.ndarray:
    """First two columns of R, flattened row-major — matches the policy's rep."""
    return R[:, :2].reshape(-1).astype(np.float32)


def _make_strictly_increasing(t: np.ndarray) -> np.ndarray:
    """Boolean-free: return the index mask keeping only strictly-increasing t."""
    keep = np.ones(t.shape[0], dtype=bool)
    last = -np.inf
    for i in range(t.shape[0]):
        if t[i] > last:
            last = t[i]
        else:
            keep[i] = False
    return keep


def _resolve_take_dir(path: str, take: str | None) -> str:
    """Resolve a session dir (+ optional take name) or a take dir directly.

    Mirrors ``replay_raw_viz.find_take``: a dir whose basename starts ``take_``
    and holds ``take_meta.json`` is a take; otherwise treat ``path`` as a session
    dir and pick ``take`` (or the first committed take)."""
    path = os.path.abspath(os.path.expanduser(path))
    base = os.path.basename(path.rstrip("/"))
    if base.startswith("take_") and os.path.exists(os.path.join(path, _TAKE_META_FILE)):
        return path
    if take:
        cand = os.path.join(path, take)
        if os.path.exists(os.path.join(cand, _TAKE_META_FILE)):
            return cand
        raise FileNotFoundError(f"UMI replay: take {take!r} not found under {path}")
    takes = sorted(
        os.path.join(path, x)
        for x in os.listdir(path)
        if os.path.isdir(os.path.join(path, x))
        and x.startswith("take_")
        and not x.endswith(".tmp")
        and os.path.exists(os.path.join(path, x, _TAKE_META_FILE))
    )
    if not takes:
        raise FileNotFoundError(f"UMI replay: no committed take_* dirs under {path}")
    return takes[0]


class UmiReplaySource:
    """Replay one UMI ``take``'s right-hand tracker as a 2pt hand command.

    Args:
        path: a take dir (``.../take_002``) or a session dir (uses ``take`` /
            first committed take).
        take: take name (e.g. ``take_002``) when ``path`` is a session dir.
        control_hz: playback resample rate; use the policy ``rl_rate`` (50 Hz) so
            one recording grid frame maps to one control tick at real-time speed.
        loop: replay repeatedly instead of holding the last frame.
        time_fn: monotonic clock (injected for deterministic tests).
    """

    def __init__(
        self, path: str, take: str | None = None, control_hz: float = 50.0, loop: bool = False, time_fn=time.monotonic
    ):
        self._take_dir = _resolve_take_dir(path, take)
        self._control_hz = float(control_hz)
        self._loop = bool(loop)
        self._time_fn = time_fn

        pos6, orn12, t_grid = self._load_and_resample()
        self._pos6 = pos6  # (K, 6) float32
        self._orn12 = orn12  # (K, 12) float32
        self._t_grid = t_grid  # (K,) seconds from frame 0, on the control grid
        self._K = int(pos6.shape[0])
        self._duration = float(t_grid[-1]) if self._K else 0.0

        self._playing = False
        self._t_play: float | None = None
        self._done_logged = False

        logger.info(
            "UmiReplaySource: loaded {} ({} frames @ {:.0f} Hz = {:.1f}s right-hand replay; loop={})".format(
                os.path.basename(self._take_dir), self._K, self._control_hz, self._duration, self._loop
            )
        )

    # --- public API (mirrors ViveLiveSource) ------------------------------
    def latest(self) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Return ``(pos[6], orn[12])`` for the current replay frame.

        Never returns ``(None, None)`` — a replay has no staleness; before
        ``play()`` it holds frame 0, after the end it holds the last frame (or
        wraps when ``loop``). Idempotent within a tick (clock-driven)."""
        if self._K == 0:
            return None, None
        frame = self._current_frame()
        return self._pos6[frame].copy(), self._orn12[frame].copy()

    def play(self) -> None:
        """(Re)start playback from frame 0 at the current clock instant."""
        self._playing = True
        self._t_play = self._time_fn()
        self._done_logged = False
        logger.info(f"UmiReplaySource: PLAY — replaying {self._duration:.1f}s right-hand trajectory.")

    def reset(self) -> None:
        """Pause and rewind to frame 0 (holds the reference pose)."""
        self._playing = False
        self._t_play = None
        self._done_logged = False

    def shutdown(self) -> None:  # parity with ViveLiveSource; nothing to tear down
        self._playing = False

    # --- introspection ----------------------------------------------------
    @property
    def num_frames(self) -> int:
        return self._K

    @property
    def duration(self) -> float:
        return self._duration

    @property
    def finished(self) -> bool:
        """True once a non-looping replay has reached the last frame."""
        if not self._playing or self._t_play is None or self._loop:
            return False
        return (self._time_fn() - self._t_play) >= self._duration

    # --- internals --------------------------------------------------------
    def _current_frame(self) -> int:
        if not self._playing or self._t_play is None:
            return 0
        elapsed = self._time_fn() - self._t_play
        if self._loop and self._duration > 0:
            elapsed = elapsed % self._duration
        if elapsed >= self._duration:
            if not self._done_logged:
                logger.info("UmiReplaySource: replay COMPLETE — holding final pose (STOP or re-trigger to replay).")
                self._done_logged = True
            return self._K - 1
        # t_grid is uniform (k / control_hz); searchsorted gives the frame at/just
        # before elapsed. Clamp defensively.
        frame = int(np.searchsorted(self._t_grid, elapsed, side="right")) - 1
        return max(0, min(frame, self._K - 1))

    def _load_and_resample(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        tr_path = os.path.join(self._take_dir, _TRACKER_FILE)
        if not os.path.exists(tr_path):
            raise FileNotFoundError(f"UMI replay: no {_TRACKER_FILE} in {self._take_dir}")
        tr = np.asarray(np.load(tr_path), dtype=np.float64)
        if tr.ndim != 2 or tr.shape[0] == 0 or tr.shape[1] < _N_TRACKER_COLS:
            raise ValueError(f"UMI replay: {tr_path} is not a valid tracker array (shape {tr.shape})")
        # Drop stale frames UNCONDITIONALLY, then dedupe to strictly-increasing
        # timestamps. An all-stale take (tracker dead/occluded the whole span) must
        # fall through to the <2-frame guard and RAISE — never silently replay the
        # rows the recorder explicitly flagged unreliable. (A conditional
        # `if live.any()` would keep the full stale array when nothing is live.)
        tr = tr[tr[:, _COL_STALE] < 0.5]
        if tr.shape[0] >= 1:
            tr = tr[_make_strictly_increasing(tr[:, _COL_T])]
        if tr.shape[0] < 2:
            raise ValueError(
                f"UMI replay: fewer than 2 live (stale<0.5) tracker frames in {tr_path} — "
                "the take is empty or was recorded with a dead/occluded tracker."
            )

        t = tr[:, _COL_T]
        t = t - t[0]
        pos = tr[:, _COL_X : _COL_X + 3]  # (N, 3) right-hand world pos
        quat = tr[:, _COL_QW : _COL_QW + 4]  # (N, 4) WXYZ
        quat = quat / np.clip(np.linalg.norm(quat, axis=1, keepdims=True), 1e-9, None)
        # Sign-continuity so the nlerp never takes the long way round.
        for i in range(1, quat.shape[0]):
            if np.dot(quat[i], quat[i - 1]) < 0.0:
                quat[i] = -quat[i]

        duration = float(t[-1])
        k = max(2, int(round(duration * self._control_hz)) + 1)
        t_grid = np.arange(k, dtype=np.float64) / self._control_hz
        t_grid = np.clip(t_grid, 0.0, duration)

        # Position: linear interp per axis.
        pos_g = np.stack([np.interp(t_grid, t, pos[:, a]) for a in range(3)], axis=1)
        # Orientation: nlerp between bracketing samples (adequate 79->50 Hz), then
        # renormalize. searchsorted gives the right bracket index.
        idx = np.clip(np.searchsorted(t, t_grid, side="right"), 1, quat.shape[0] - 1)
        t0 = t[idx - 1]
        t1 = t[idx]
        denom = np.clip(t1 - t0, 1e-9, None)
        a = np.clip((t_grid - t0) / denom, 0.0, 1.0)[:, None]
        q_g = (1.0 - a) * quat[idx - 1] + a * quat[idx]
        q_g = q_g / np.clip(np.linalg.norm(q_g, axis=1, keepdims=True), 1e-9, None)

        eye6 = np.array([1.0, 0.0, 0.0, 1.0, 0.0, 0.0], dtype=np.float32)
        pos6 = np.zeros((k, 6), dtype=np.float32)
        orn12 = np.zeros((k, 12), dtype=np.float32)
        pos6[:, 3:6] = pos_g.astype(np.float32)  # right hand; left held at 0
        orn12[:, 0:6] = eye6  # left = identity (constant -> holds ref)
        for i in range(k):
            orn12[i, 6:12] = _R_to_rot6d(_wxyz_to_R(q_g[i]))
        return pos6, orn12, t_grid
