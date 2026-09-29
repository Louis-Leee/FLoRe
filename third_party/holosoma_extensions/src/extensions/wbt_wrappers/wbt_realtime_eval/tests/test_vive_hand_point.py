"""`--hand-point lever|tcp`: WHICH point on each hand the VIVE bridge steers.

The two candidates are 6.7 cm apart per hand and the policy obs is the SAME width
(170D/680D) in both geometries — only the term name differs (`vr_3point_pos` vs
`vr_3point_pos_tcp`). So nothing downstream can catch a mismatch, and these tests are
the guard: the TCP arrays must describe the SAME wrist with a longer lever (not a
re-typed pose), the head point must not move, and the historical lever path must stay
byte-identical.

pysurvive is injected via PYTHONPATH by `scripts/_vive_teleop_bridge.sh`, not pip —
so it is stubbed here (the bridge only touches it inside `run_pysurvive`, which these
tests never enter).
"""

from __future__ import annotations

import sys
import types

import numpy as np
import pytest

# Stub the PYTHONPATH-injected USB singleton before the bridge module is imported.
for _name in ("pysurvive",):
    if _name not in sys.modules:
        _m = types.ModuleType(_name)
        _m.SimpleContext = object
        _m.survive_simple_serial_number = lambda *_a, **_k: b"?"
        sys.modules[_name] = _m

from wbt_realtime_eval.vive import load_home_anchor  # noqa: E402
from wbt_realtime_eval.vive import vive_teleop_bridge as vtb  # noqa: E402

# TCP_OFFSET_X, spelled once. The bridge reads it from home_anchor.json; asserting the
# literal here is what makes a silent edit of that file fail a test.
TCP_OFFSET_X = 0.242
LEVER_OFFSET = (0.18, 0.025)   # (+x, |lateral|)
HEAD_OFFSET = (0.0, 0.0, 0.35)  # torso_link -> head point, geometry-independent
EXPECTED_SHIFT_M = 0.0669       # documented 6.7 cm/hand TCP-vs-lever gap


class _Args:
    """Minimal stand-in for the argparse namespace `ViveTeleopBridge` reads."""

    def __init__(self, **kw):
        defaults = dict(
            mode="3pt", left_serial="LHR-L", right_serial="LHR-R", head_serial="LHR-H",
            hand_point="lever", yaw_deg=0.0, reach_scale=1.0, z_offset=0.0,
            orientation="on", stale_timeout=0.25, no_align=False,
            no_yaw_autocal=False, calib_delay=3.0,
        )
        defaults.update(kw)
        self.__dict__.update(defaults)


def _rot(q_wxyz) -> np.ndarray:
    """wxyz quaternion -> 3x3, inlined to match the bridge's own math."""
    w, x, y, z = (float(v) for v in q_wxyz)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


@pytest.fixture(scope="module")
def anchor() -> dict:
    return load_home_anchor()


@pytest.fixture(scope="module")
def lever():
    return vtb.ViveTeleopBridge(_Args(hand_point="lever"))


@pytest.fixture(scope="module")
def tcp():
    return vtb.ViveTeleopBridge(_Args(hand_point="tcp"))


def test_lever_is_the_default_and_matches_the_historical_arrays(anchor):
    """Default (no --hand-point) must be byte-identical to the pre-TCP bridge."""
    b = vtb.ViveTeleopBridge(_Args(mode="2pt"))
    assert b._hand_point == "lever"
    assert np.allclose(b._vr_offsets, anchor["vr_2point_offsets"])
    assert np.allclose(b._home_pos, anchor["home_pos_pelvis"])
    assert b._home_sep == anchor["hand_separation_m"]


def test_lever_geometry_is_the_0p18_vr_lever(lever):
    for k in (0, 1):
        off = lever._vr_offsets[k]
        assert off[0] == pytest.approx(LEVER_OFFSET[0])
        assert abs(off[1]) == pytest.approx(LEVER_OFFSET[1])
        assert off[2] == pytest.approx(0.0)


def test_tcp_geometry_is_the_0p242_fingertip_with_no_lateral_term(tcp):
    """Both hands on (0.242, 0, 0): the +-0.025 lateral term is DROPPED on purpose —
    keeping it would leave position and force 2.5 cm apart, i.e. not unified."""
    for k in (0, 1):
        assert np.allclose(tcp._vr_offsets[k], [TCP_OFFSET_X, 0.0, 0.0])
    assert np.allclose(tcp._vr_offsets[0], tcp._vr_offsets[1]), "hands must be symmetric"


def test_head_point_is_identical_in_both_geometries(lever, tcp, anchor):
    """The third point is the SAME point in both: torso_link + 0.35 z. Only the two
    HAND rows are swapped, which is why a chest/head tracker needs no re-calibration."""
    for b in (lever, tcp):
        assert np.allclose(b._vr_offsets_3pt[2], HEAD_OFFSET)
        assert np.allclose(b._home_pos_3pt[2], anchor["home_pos_pelvis_3pt"][2])
    assert np.allclose(lever._vr_offsets_3pt[2], tcp._vr_offsets_3pt[2])
    assert np.allclose(lever._home_pos_3pt[2], tcp._home_pos_3pt[2])


def test_the_two_geometries_describe_the_same_wrist(lever, tcp):
    """home = wrist_origin + R(home_q) @ offset. Recovering wrist_origin from each
    geometry must give the SAME point — that is what proves the TCP home was DERIVED
    by extending the lever, not typed in as an independent pose."""
    for k in (0, 1):
        rot = _rot(lever._home_wrist_q[k])
        w_lever = lever._home_pos[k] - rot @ lever._vr_offsets[k]
        w_tcp = tcp._home_pos[k] - rot @ tcp._vr_offsets[k]
        assert np.allclose(w_lever, w_tcp, atol=1e-5), (k, w_lever, w_tcp)


def test_per_hand_shift_is_the_documented_6p7_cm(lever, tcp):
    for k in (0, 1):
        shift = float(np.linalg.norm(tcp._home_pos[k] - lever._home_pos[k]))
        assert shift == pytest.approx(EXPECTED_SHIFT_M, abs=5e-4)


def test_orientation_home_is_geometry_independent(lever, tcp, anchor):
    """The TCP shares the wrist's orientation, so the rot6d / quat home must NOT move —
    a changed orn home would rotate every commanded wrist."""
    for b in (lever, tcp):
        assert np.allclose(b._home_wrist_q, anchor["home_wrist_quat_wxyz"])
        assert np.allclose(b._home_orn6d, anchor["home_orn_rot6d"])
        assert np.allclose(b._home_body_q_3pt, anchor["home_body_quat_wxyz_3pt"])


def test_one_geometry_not_two(lever, tcp):
    """The 2pt hand arrays and the 3pt array's hand rows must agree, or `_hand_command`
    (which reads the 2pt arrays) and `_head_command` (which reads the 3pt array) would
    be steering two different geometries in one session."""
    for b in (lever, tcp):
        assert np.allclose(b._vr_offsets_3pt[:2], b._vr_offsets)
        assert np.allclose(b._home_pos_3pt[:2], b._home_pos)


def test_hand_separation_tracks_the_active_geometry(lever, tcp):
    """`_home_sep` is what the CALIBRATED log line prints for the operator to sanity
    check against their own hand spacing — it must be the ACTIVE geometry's."""
    for b in (lever, tcp):
        assert b._home_sep == pytest.approx(float(np.linalg.norm(b._home_pos[0] - b._home_pos[1])), abs=1e-4)
    assert tcp._home_sep > lever._home_sep, "the TCP hands are further apart (outward lever)"


def test_tcp_is_refused_in_2pt():
    """No 2pt checkpoint tracks the 0.242 m TCP, so accepting it would be a silent
    6.7 cm/hand train-deploy mismatch. Refuse loudly instead."""
    with pytest.raises(SystemExit) as ei:
        vtb.ViveTeleopBridge(_Args(mode="2pt", hand_point="tcp"))
    assert ei.value.code == 2


def test_cli_default_is_lever():
    """A bare `wbt-vive-teleop-bridge` must not silently change geometry."""
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--hand-point", choices=["lever", "tcp"], default="lever")
    assert ap.parse_args([]).hand_point == "lever"
    assert ap.parse_args(["--hand-point", "tcp"]).hand_point == "tcp"
