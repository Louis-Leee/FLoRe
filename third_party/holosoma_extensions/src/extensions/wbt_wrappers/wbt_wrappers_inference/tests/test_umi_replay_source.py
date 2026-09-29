"""Tests for ``UmiReplaySource`` — the recorded-take REPLAY twin of ViveLiveSource.

Correctness claims covered:

* the source loads a synthetic UMI ``take`` and exposes the SAME
  ``latest() -> (pos[6], orn[12])`` rep the policy's clutch consumes;
* the LEFT hand half is held CONSTANT (so ``live_L − live0_L == 0`` ⇒ the left
  hand holds the reference), only the RIGHT half carries the recorded motion;
* orn12 is a valid rot6d (left = identity, right = a proper rotation);
* playback is CLOCK-driven and idempotent within a tick (two ``latest()`` calls
  at the same clock instant return the same frame — required because the policy
  polls ``latest()`` several times per control tick);
* frame 0 == the recording origin before ``play()``, advances after, holds the
  final frame at the end (and wraps when ``loop``);
* the frame-0-relative delta the clutch would feed (``latest − latest@0``)
  reproduces the recorded right-hand displacement, i.e. the replay is a drop-in
  for the live clutch.

No holosoma / mujoco / websockets import — pure numpy + a synthetic take on tmp.

Run: pytest src/extensions/wbt_wrappers/wbt_wrappers_inference/tests/test_umi_replay_source.py
"""

from __future__ import annotations

import json
import os

import numpy as np
import pytest

from wbt_wrappers_inference.utils.umi_replay_source import UmiReplaySource, _R_to_rot6d, _wxyz_to_R


class _Clock:
    """Injectable monotonic clock; advance with ``tick(dt)``."""

    def __init__(self):
        self.t = 0.0

    def __call__(self) -> float:
        return self.t

    def tick(self, dt: float) -> None:
        self.t += dt


def _quat_z(theta: float) -> np.ndarray:
    """WXYZ quaternion for a rotation of ``theta`` about +Z."""
    return np.array([np.cos(theta / 2.0), 0.0, 0.0, np.sin(theta / 2.0)])


def _write_take(tmp_path, n=200, hz=79.0, *, stale_head=0) -> str:
    """Write a synthetic ``take_000`` with a smooth right-hand reach + yaw sweep.

    Returns the take dir. Columns: t seq x y z qw qx qy qz stale.
    """
    take_dir = os.path.join(str(tmp_path), "session_x", "take_000")
    os.makedirs(take_dir, exist_ok=True)
    t = np.arange(n) / hz
    s = t / t[-1]  # 0..1
    x = 0.10 + 0.30 * s  # reach forward 30 cm
    y = 0.05 - 0.10 * s
    z = 0.20 + 0.15 * np.sin(np.pi * s)
    rows = np.zeros((n, 10), dtype=np.float64)
    rows[:, 0] = 1000.0 + t  # absolute wall clock
    rows[:, 1] = np.arange(n)
    rows[:, 2] = x
    rows[:, 3] = y
    rows[:, 4] = z
    for i in range(n):
        rows[i, 5:9] = _quat_z(0.6 * s[i])  # sweep 0..0.6 rad yaw
    rows[:, 9] = 0.0
    for i in range(min(stale_head, n)):
        rows[i, 9] = 1.0  # mark leading frames stale (should be dropped)
    np.save(os.path.join(take_dir, "tracker.npy"), rows)
    with open(os.path.join(take_dir, "take_meta.json"), "w") as f:
        json.dump({"take_dir": "take_000", "quaternion_convention": "wxyz"}, f)
    return take_dir


def test_loads_and_shapes(tmp_path):
    take = _write_take(tmp_path)
    clk = _Clock()
    src = UmiReplaySource(take, control_hz=50.0, time_fn=clk)
    pos, orn = src.latest()
    assert pos.shape == (6,) and orn.shape == (12,)
    assert pos.dtype == np.float32 and orn.dtype == np.float32
    assert src.num_frames >= 2
    # ~200 frames @ 79 Hz ≈ 2.5 s → ~127 frames @ 50 Hz.
    assert abs(src.duration - 199 / 79.0) < 0.05


def test_left_hand_is_constant_zero(tmp_path):
    """Left half of pos is all-zero and orn left half is identity for every frame."""
    take = _write_take(tmp_path)
    clk = _Clock()
    src = UmiReplaySource(take, control_hz=50.0, loop=False, time_fn=clk)
    eye6 = np.array([1.0, 0.0, 0.0, 1.0, 0.0, 0.0], dtype=np.float32)
    src.play()
    for _ in range(src.num_frames + 5):
        pos, orn = src.latest()
        np.testing.assert_array_equal(pos[0:3], np.zeros(3, dtype=np.float32))
        np.testing.assert_allclose(orn[0:6], eye6, atol=1e-6)
        clk.tick(0.02)


def test_right_orn_is_valid_rotation(tmp_path):
    take = _write_take(tmp_path)
    src = UmiReplaySource(take, control_hz=50.0, time_fn=_Clock())
    _pos, orn = src.latest()
    R = _rebuild(orn[6:12])
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-5)
    assert np.isclose(np.linalg.det(R), 1.0, atol=1e-5)


def test_idempotent_within_a_tick(tmp_path):
    """Two latest() calls at the SAME clock instant return the SAME frame."""
    take = _write_take(tmp_path)
    clk = _Clock()
    src = UmiReplaySource(take, control_hz=50.0, time_fn=clk)
    src.play()
    clk.tick(0.5)
    p1, o1 = src.latest()
    p2, o2 = src.latest()  # no clock advance
    np.testing.assert_array_equal(p1, p2)
    np.testing.assert_array_equal(o1, o2)


def test_frame0_before_play_and_advances_after(tmp_path):
    take = _write_take(tmp_path)
    clk = _Clock()
    src = UmiReplaySource(take, control_hz=50.0, time_fn=clk)
    # Before play(): frame 0 regardless of clock.
    p0, _ = src.latest()
    clk.tick(1.0)
    p0b, _ = src.latest()
    np.testing.assert_array_equal(p0, p0b)
    # After play(): advances.
    src.play()
    clk.tick(0.5)
    pmid, _ = src.latest()
    assert not np.array_equal(p0[3:6], pmid[3:6])
    # Right-hand x is monotone increasing in the synthetic reach.
    assert pmid[3] > p0[3]


def test_holds_final_frame_when_done(tmp_path):
    take = _write_take(tmp_path)
    clk = _Clock()
    src = UmiReplaySource(take, control_hz=50.0, loop=False, time_fn=clk)
    src.play()
    clk.tick(src.duration + 5.0)
    assert src.finished
    p_end, _ = src.latest()
    # The very last recorded x is the max reach (~0.40).
    assert p_end[3] > 0.38


def test_loop_wraps(tmp_path):
    take = _write_take(tmp_path)
    clk = _Clock()
    src = UmiReplaySource(take, control_hz=50.0, loop=True, time_fn=clk)
    src.play()
    p_start, _ = src.latest()
    clk.tick(src.duration)  # one full period → back near frame 0
    p_wrapped, _ = src.latest()
    assert not src.finished  # looping never reports finished
    np.testing.assert_allclose(p_wrapped[3:6], p_start[3:6], atol=2e-2)


def test_stale_head_dropped(tmp_path):
    """Leading stale frames are dropped; frame 0 is the first LIVE sample."""
    take = _write_take(tmp_path, stale_head=10)
    src = UmiReplaySource(take, control_hz=50.0, time_fn=_Clock())
    p0, _ = src.latest()
    # frame-0 x should be the first live x (s=10/199 → 0.10 + 0.30*10/199 ≈ 0.115),
    # NOT the stale-frame value (which is identical here, but the point is it loads).
    assert 0.10 <= p0[3] <= 0.13


def test_delta_reproduces_recorded_displacement(tmp_path):
    """The frame-0-relative delta the clutch feeds == recorded right-hand motion.

    This is the load-bearing claim: because the policy clutch commands
    ``ref + (live − live0)`` with ``live0 = latest()@frame0``, the fed delta must
    equal the recording's displacement since frame 0."""
    take = _write_take(tmp_path)
    clk = _Clock()
    src = UmiReplaySource(take, control_hz=50.0, time_fn=clk)
    p0, _ = src.latest()  # frame 0 (== live0 the clutch snapshots on arm)
    src.play()
    clk.tick(src.duration)  # end
    pend, _ = src.latest()
    delta = pend[3:6] - p0[3:6]
    # Synthetic reach: dx≈+0.30, dy≈−0.10, dz≈~0 (sin returns to 0 at s=1).
    assert abs(delta[0] - 0.30) < 0.02
    assert abs(delta[1] + 0.10) < 0.02
    assert abs(delta[2] - 0.0) < 0.02


def test_resolve_session_dir_picks_take(tmp_path):
    """Passing the SESSION dir (not the take dir) resolves to the take."""
    take = _write_take(tmp_path)
    session = os.path.dirname(take)
    src = UmiReplaySource(session, take="take_000", control_hz=50.0, time_fn=_Clock())
    assert src.num_frames >= 2


def test_missing_take_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        UmiReplaySource(os.path.join(str(tmp_path), "does_not_exist"), time_fn=_Clock())


def test_all_stale_take_raises(tmp_path):
    """A take where EVERY frame is stale must raise, not silently replay stale data.

    (Regression: a conditional stale-drop `if live.any()` would keep the full
    stale array and pass the <2-frame guard.)"""
    take = _write_take(tmp_path, n=100, stale_head=100)  # all 100 frames stale
    with pytest.raises(ValueError, match="fewer than 2 live"):
        UmiReplaySource(take, control_hz=50.0, time_fn=_Clock())


# --------------------------------------------------------------------------- #
def _rebuild(r6: np.ndarray) -> np.ndarray:
    """Gram-Schmidt rebuild of a 3x3 rotation from the 6D rep (test-local)."""
    m = r6.reshape(3, 2).astype(np.float64)
    b0 = m[:, 0] / np.linalg.norm(m[:, 0])
    b1 = m[:, 1] - (b0 @ m[:, 1]) * b0
    b1 = b1 / np.linalg.norm(b1)
    b2 = np.cross(b0, b1)
    return np.stack([b0, b1, b2], axis=1)


def test_wxyz_to_R_matches_rot6d_roundtrip():
    """_wxyz_to_R → rot6d → rebuild recovers the same rotation (internal consistency)."""
    q = _quat_z(0.7)
    R = _wxyz_to_R(q)
    R_hat = _rebuild(_R_to_rot6d(R))
    np.testing.assert_allclose(R_hat, R, atol=1e-6)
