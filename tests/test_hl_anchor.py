"""Tests for the chunk-anchor selector (measured vs command / PLAI).

The safety-relevant properties, in order of how badly they hurt if broken:

1. ``measured`` mode cannot alter anything (the un-flagged rig path must be untouched).
2. The leak BOUNDS the carried lead — replayed unclamped over the recorded takes, command
   integration diverges to ~22 cm, so a clamp that only masked the output while letting the
   internal state grow would be worse than useless.
3. Contact decays the carried anchor back toward the arm instead of advancing it.
4. A non-finite input can never become an anchor (holosoma serializes NaN unchecked).
"""

from __future__ import annotations

import numpy as np
import pytest

from flore.s1_policy.serving.hl_anchor import (
    MAX_EPS_M,
    MAX_EPS_RAD,
    AnchorConfig,
    ChunkAnchor,
)
from flore.utils.rot6d import rot6d_to_rotmat, rotmat_to_rot6d, rotmat_to_rotvec, rotvec_to_rotmat

IDENT = np.array([1.0, 0, 0, 0, 1, 0])


def _rot6d(axis, angle):
    return rotmat_to_rot6d(rotvec_to_rotmat(np.asarray(axis, dtype=np.float64) * angle))


def _angle_between(a, b):
    return float(np.linalg.norm(rotmat_to_rotvec(rot6d_to_rotmat(a).T @ rot6d_to_rotmat(b))))


# --------------------------------------------------------------------------- default is inert
def test_measured_mode_is_the_identity_and_reports_disabled():
    a = ChunkAnchor()
    assert a.cfg.mode == "measured"
    assert a.disabled()
    a.note_commanded(np.array([9.0, 9.0, 9.0]), _rot6d([0, 0, 1], 1.0))  # must be ignored
    pos, rot = a.resolve(np.array([0.1, 0.2, 0.3]), IDENT, force_n=0.0)
    np.testing.assert_allclose(pos, [0.1, 0.2, 0.3])
    np.testing.assert_allclose(rot, IDENT)
    assert a.stats()["n_resolved"] == 0  # not even counted: nothing happened


def test_command_mode_bootstraps_from_the_measurement():
    a = ChunkAnchor(AnchorConfig(mode="command"))
    assert not a.disabled()
    pos, rot = a.resolve(np.array([0.1, 0.0, 0.0]), IDENT, force_n=0.0)
    np.testing.assert_allclose(pos, [0.1, 0.0, 0.0])
    assert a.stats()["n_bootstrap"] == 1


def test_command_mode_returns_the_commanded_pose_when_inside_the_leak():
    a = ChunkAnchor(AnchorConfig(mode="command", eps_m=0.03))
    a.resolve(np.zeros(3), IDENT, force_n=0.0)          # bootstrap
    a.note_commanded(np.array([0.010, 0.0, 0.0]), IDENT)
    pos, _ = a.resolve(np.array([0.006, 0.0, 0.0]), IDENT, force_n=0.0)
    # The arm reached 6 mm of the 10 mm asked; the anchor stays at the COMMAND, not the arm.
    np.testing.assert_allclose(pos, [0.010, 0.0, 0.0])
    assert a.stats()["n_clamped_pos"] == 0


# ------------------------------------------------------------------------------ the leak bound
def test_repeated_shortfall_cannot_diverge_past_eps():
    """The load-bearing one. The arm realizes g=0.61 of every increment (measured), so without a
    leak the carried anchor runs away — 224 mm by end of take on the recorded data."""
    eps = 0.03
    a = ChunkAnchor(AnchorConfig(mode="command", eps_m=eps))
    g = 0.61
    meas = np.zeros(3)
    cmd = np.zeros(3)
    a.resolve(meas, IDENT, force_n=0.0)
    for _ in range(500):
        anchor, _ = a.resolve(meas, IDENT, force_n=0.0)
        cmd = anchor + np.array([0.01, 0.0, 0.0])       # the chunk asks for another centimetre
        meas = meas + g * (cmd - meas) * 0.0 + g * 0.01 * np.array([1.0, 0, 0])
        a.note_commanded(cmd, IDENT)
        assert np.linalg.norm(anchor - meas) <= eps + 0.011 + 1e-9
    # and the CARRIED state is bounded too, not just the returned value
    final, _ = a.resolve(meas, IDENT, force_n=0.0)
    assert np.linalg.norm(final - meas) <= eps + 1e-9
    assert a.stats()["n_clamped_pos"] > 0


def test_eps_zero_collapses_to_measured_behaviour():
    a = ChunkAnchor(AnchorConfig(mode="command", eps_m=0.0, eps_rad=0.0))
    a.resolve(np.zeros(3), IDENT, force_n=0.0)
    a.note_commanded(np.array([0.5, 0.0, 0.0]), _rot6d([0, 0, 1], 0.9))
    pos, rot = a.resolve(np.array([0.02, 0.0, 0.0]), IDENT, force_n=0.0)
    np.testing.assert_allclose(pos, [0.02, 0.0, 0.0], atol=1e-12)
    assert _angle_between(rot, IDENT) < 1e-9


def test_rotation_lead_is_bounded_by_eps_rad():
    eps_rad = np.radians(10.0)
    a = ChunkAnchor(AnchorConfig(mode="command", eps_rad=eps_rad))
    a.resolve(np.zeros(3), IDENT, force_n=0.0)
    a.note_commanded(np.zeros(3), _rot6d([0, 0, 1], np.radians(60.0)))
    _, rot = a.resolve(np.zeros(3), IDENT, force_n=0.0)
    assert _angle_between(IDENT, rot) == pytest.approx(eps_rad, abs=1e-6)
    assert a.stats()["n_clamped_rot"] == 1


# ------------------------------------------------------------------------------ contact cut-out
def test_contact_decays_the_carried_anchor_toward_the_arm():
    a = ChunkAnchor(AnchorConfig(mode="command", contact_n=1.5, decay=0.5))
    a.resolve(np.zeros(3), IDENT, force_n=0.0)
    a.note_commanded(np.array([0.02, 0.0, 0.0]), IDENT)
    pos, _ = a.resolve(np.zeros(3), IDENT, force_n=5.0)
    np.testing.assert_allclose(pos, [0.01, 0.0, 0.0])       # halved toward the arm
    pos, _ = a.resolve(np.zeros(3), IDENT, force_n=5.0)
    np.testing.assert_allclose(pos, [0.005, 0.0, 0.0])      # and again — it unwinds
    assert a.stats()["n_contact_pullback"] == 2


def test_contact_decay_never_snaps():
    """Decayed, not zeroed: snapping would put back the 16 mm boundary jump the mode removes."""
    a = ChunkAnchor(AnchorConfig(mode="command", contact_n=1.5, decay=0.9))
    a.resolve(np.zeros(3), IDENT, force_n=0.0)
    a.note_commanded(np.array([0.02, 0.0, 0.0]), IDENT)
    pos, _ = a.resolve(np.zeros(3), IDENT, force_n=2.0)
    assert 0.017 < float(np.linalg.norm(pos)) < 0.02


def test_unknown_force_does_not_cut_out():
    """None means 'no reading', which must not silently disarm a guard the caller thinks is on."""
    a = ChunkAnchor(AnchorConfig(mode="command", contact_n=1.5, decay=0.5))
    a.resolve(np.zeros(3), IDENT, force_n=0.0)
    a.note_commanded(np.array([0.02, 0.0, 0.0]), IDENT)
    pos, _ = a.resolve(np.zeros(3), IDENT, force_n=None)
    np.testing.assert_allclose(pos, [0.02, 0.0, 0.0])
    assert a.stats()["n_contact_pullback"] == 0


def test_contact_is_checked_before_the_clamp_so_a_saturated_hand_still_unwinds():
    a = ChunkAnchor(AnchorConfig(mode="command", eps_m=0.02, contact_n=1.5, decay=0.5))
    a.resolve(np.zeros(3), IDENT, force_n=0.0)
    a.note_commanded(np.array([0.02, 0.0, 0.0]), IDENT)   # exactly at the clamp
    pos, _ = a.resolve(np.zeros(3), IDENT, force_n=9.0)
    assert float(np.linalg.norm(pos)) == pytest.approx(0.01)


# ----------------------------------------------------------------------------------- lifecycle
def test_reset_drops_the_carried_anchor():
    a = ChunkAnchor(AnchorConfig(mode="command"))
    a.resolve(np.zeros(3), IDENT, force_n=0.0)
    a.note_commanded(np.array([0.02, 0.0, 0.0]), IDENT)
    assert a.stats()["carrying"]
    a.reset()
    assert not a.stats()["carrying"]
    pos, _ = a.resolve(np.array([0.004, 0.0, 0.0]), IDENT, force_n=0.0)
    np.testing.assert_allclose(pos, [0.004, 0.0, 0.0])    # bootstrapped from the arm again


def test_non_finite_command_is_never_carried():
    a = ChunkAnchor(AnchorConfig(mode="command"))
    a.resolve(np.zeros(3), IDENT, force_n=0.0)
    a.note_commanded(np.array([np.nan, 0.0, 0.0]), IDENT)
    assert not a.stats()["carrying"]
    pos, _ = a.resolve(np.array([0.004, 0.0, 0.0]), IDENT, force_n=0.0)
    assert np.all(np.isfinite(pos))


def test_non_finite_measurement_falls_back_to_that_measurement_and_does_not_poison_state():
    a = ChunkAnchor(AnchorConfig(mode="command"))
    a.resolve(np.zeros(3), IDENT, force_n=0.0)
    a.note_commanded(np.array([0.01, 0.0, 0.0]), IDENT)
    pos, _ = a.resolve(np.array([np.nan, 0.0, 0.0]), IDENT, force_n=0.0)
    assert not np.all(np.isfinite(pos))         # the caller's own bad input comes straight back
    # …and the next good measurement bootstraps cleanly rather than decoding from garbage.
    pos, _ = a.resolve(np.array([0.004, 0.0, 0.0]), IDENT, force_n=0.0)
    assert np.all(np.isfinite(pos))


# --------------------------------------------------------------------------------- config gates
@pytest.mark.parametrize("kwargs", [
    {"mode": "nonsense"},
    {"mode": "command", "eps_m": MAX_EPS_M + 0.01},
    {"mode": "command", "eps_m": -0.01},
    {"mode": "command", "eps_rad": MAX_EPS_RAD + 0.1},
    {"mode": "command", "decay": 0.0},
    {"mode": "command", "decay": 1.5},
    {"mode": "command", "contact_n": -1.0},
])
def test_bad_config_is_refused_loudly(kwargs):
    with pytest.raises(ValueError):
        ChunkAnchor(AnchorConfig(**kwargs))


def test_eps_ceiling_is_the_same_as_the_reach_clamp_ceiling():
    """Both bound an un-asked-for addition to a physical arm's target, so they must not drift
    apart: an operator who learned one ceiling should not be surprised by the other."""
    from flore.s1_policy.serving import hl_reach

    assert MAX_EPS_M == hl_reach.MAX_CLAMP_M
