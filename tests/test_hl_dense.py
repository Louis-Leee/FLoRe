"""Contract tests for the DENSE deploy path: motion completion -> 29 joint targets -> :8770.

CPU-only and model-free. What the completion itself produces needs a GPU, the g1_mosyn conda
env and a checkpoint, and is covered by ``scripts/smoke_test_hl_dense.py``; what is pinned here
is everything that decides whether a correct completion reaches the robot correctly:

* **the by-name remap**, fed a REVERSED source order — the only input that can tell a by-name
  implementation apart from an index-aligned one that happens to work today.
* **the joint clutch**, whose whole reason to exist is a measured number: the completion places
  the arm 0.40 rad away from the ready pose in the null space while putting the hand within
  0.84 mm of where it was told, so publishing absolutely either fails the 0.20 rad pre-arm gate
  or snaps the arm 23 deg on the first executed step.
* **the wire**, whose consumer (``ViveLiveSource``) validates only WIDTH and DROPS a dense frame
  with no ``joint_names`` — so the names are not optional and a NaN cannot be caught downstream.
* **the 3-point wire staying byte-identical**, because the same ``S0Driver`` serves both planners
  and the left_only deploy is in use.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from fuji.client.s0_driver import S0Driver
from fuji.s1_policy.serving import hl_dense

_NAMES = tuple(f"joint_{i:02d}" for i in range(hl_dense.G1_DOF))


# --------------------------------------------------------------------------- the remap
def test_the_remap_is_by_name_and_a_reversed_order_proves_it():
    """The one test an index-aligned implementation cannot pass.

    MC's joint order and holosoma's ``dof_names`` are byte-identical TODAY, so
    ``values[...]`` with no permutation returns the right answer on every realistic input and
    scrambles the body the day either order changes. Reversing the source is what separates
    them: a by-name remap returns the values re-sorted, an index-aligned one returns them
    unchanged.
    """
    values = np.arange(hl_dense.G1_DOF, dtype=np.float64)
    reversed_names = list(reversed(_NAMES))
    got = hl_dense.remap_by_name(values, reversed_names, list(_NAMES))
    np.testing.assert_allclose(got, values[::-1])
    assert not np.allclose(got, values), "an index-aligned remap would return values unchanged"
    # Identity order round-trips, which is the case that must NOT regress.
    np.testing.assert_allclose(
        hl_dense.remap_by_name(values, list(_NAMES), list(_NAMES)), values)


def test_the_remap_refuses_a_missing_joint_instead_of_zero_filling():
    """A silently-zeroed joint is a limb going limp on a physical robot."""
    values = np.arange(hl_dense.G1_DOF, dtype=np.float64)
    wanted = list(_NAMES[:-1]) + ["a_joint_that_does_not_exist"]
    with pytest.raises(ValueError, match="absent from the source order"):
        hl_dense.remap_by_name(values, list(_NAMES), wanted)
    with pytest.raises(ValueError, match="duplicates"):
        hl_dense.remap_by_name(values, ["dup"] * hl_dense.G1_DOF, list(_NAMES))
    with pytest.raises(ValueError, match="width"):
        hl_dense.remap_by_name(values[:5], list(_NAMES), list(_NAMES))


def test_the_remap_carries_a_leading_axis_so_a_whole_chunk_goes_through_at_once():
    chunk = np.arange(4 * hl_dense.G1_DOF, dtype=np.float64).reshape(4, hl_dense.G1_DOF)
    got = hl_dense.remap_by_name(chunk, list(reversed(_NAMES)), list(_NAMES))
    np.testing.assert_allclose(got, chunk[:, ::-1])


def test_the_ready_pose_targets_come_back_in_the_wires_joint_order():
    """The idle hold is the SDK's ready pose expressed in MC's order — the wire's order."""
    sdk = np.linspace(-1.0, 1.0, hl_dense.G1_DOF)
    sdk_names = [f"sdk_{i}" for i in range(hl_dense.G1_DOF)]
    mc_names = list(reversed(sdk_names))
    np.testing.assert_allclose(
        hl_dense.ready_pose_joint_targets(sdk, sdk_names, mc_names), sdk[::-1])
    with pytest.raises(ValueError, match="29 SDK joint angles"):
        hl_dense.ready_pose_joint_targets(sdk[:10], sdk_names, mc_names)


# --------------------------------------------------------------------------- the clutch
def _chunk(offset: float = 0.0, travel: float = 0.1) -> np.ndarray:
    """A 5-step chunk that STARTS ``offset`` away from zero and moves ``travel``."""
    base = np.full(hl_dense.G1_DOF, offset)
    return np.stack([base + travel * i / 4.0 for i in range(5)])


def test_the_clutch_makes_step_0_a_no_op_on_every_joint():
    """THE REASON IT EXISTS, as a number.

    Measured on ``ckpt_0831_final`` seeded exactly at the deploy ready pose, the completion's
    arm joints at step 0 are 0.40 rad (23 deg) from the zeroed arms it was conditioned on, and
    its knee 0.54 rad out of the ready crouch — while the commanded gripper lands within
    0.84 mm. That is arm redundancy, not an error, and it must not become a command: the
    joint-space pre-arm gate tolerates 0.20 rad.
    """
    ready = np.zeros(hl_dense.G1_DOF)
    clutch = hl_dense.JointClutch(ready)
    published = clutch.apply(_chunk(offset=0.40, travel=0.1))
    np.testing.assert_allclose(published[0], ready, atol=0.0)
    # ...and the MOTION survives intact: only the absolute offset was removed.
    np.testing.assert_allclose(np.diff(published, axis=0),
                               np.diff(_chunk(offset=0.40, travel=0.1), axis=0))


def test_clutch_none_is_the_identity_and_therefore_publishes_the_offset():
    chunk = _chunk(offset=0.40)
    plain = hl_dense.JointClutch(np.zeros(hl_dense.G1_DOF), mode="none").apply(chunk)
    np.testing.assert_allclose(plain, chunk, atol=0.0)
    assert float(np.abs(plain[0]).max()) == pytest.approx(0.40)
    with pytest.raises(ValueError, match="mode must be one of"):
        hl_dense.JointClutch(np.zeros(hl_dense.G1_DOF), mode="sometimes")


def test_the_base_advances_to_the_last_executed_step_not_the_chunk_end():
    """A chunk boundary must not snap the arm back, and must not credit un-executed steps.

    Executing 3 of 5 steps and re-predicting has to continue from step 2, exactly as the pose
    anchor does. If the base stayed at the ready pose the arm would return there every chunk;
    if it advanced to the chunk END it would jump forward past motion the arm never made.
    """
    clutch = hl_dense.JointClutch(np.zeros(hl_dense.G1_DOF))
    first = clutch.apply(_chunk(offset=0.4, travel=0.4))
    clutch.commit(first[2])
    np.testing.assert_allclose(clutch.base, first[2])
    second = clutch.apply(_chunk(offset=-1.7, travel=0.4))   # a wildly different null space
    np.testing.assert_allclose(second[0], first[2], atol=0.0)
    assert clutch.drift_rad() == pytest.approx(float(np.abs(first[2]).max()))


def test_a_measured_sync_wins_and_a_garbage_one_is_ignored():
    """RECORD=1 re-bases on the robot; a bad readback must hold, never zero the base."""
    clutch = hl_dense.JointClutch(np.full(hl_dense.G1_DOF, 0.25))
    measured = np.full(hl_dense.G1_DOF, 0.5)
    clutch.sync(measured)
    np.testing.assert_allclose(clutch.base, measured)
    for bad in (np.full(hl_dense.G1_DOF, np.nan), np.zeros(3)):
        clutch.sync(bad)
        np.testing.assert_allclose(clutch.base, measured), "a bad readback moved the base"
    clutch.reset()
    np.testing.assert_allclose(clutch.base, np.full(hl_dense.G1_DOF, 0.25))


# --------------------------------------------------------------------------- safety
def test_a_non_finite_joint_target_is_refused_before_the_wire():
    """``json.dumps`` serializes NaN and the low level checks only WIDTH, so this is the gate."""
    good = np.zeros(hl_dense.G1_DOF)
    assert hl_dense.is_safe_dense_command(good) == (True, "")
    for i, value in ((3, np.nan), (11, np.inf)):
        bad = good.copy()
        bad[i] = value
        ok, reason = hl_dense.is_safe_dense_command(bad)
        assert not ok and f"index {i}" in reason
    ok, reason = hl_dense.is_safe_dense_command(good, np.full(hl_dense.G1_DOF, np.nan))
    assert not ok and "joint_vel is non-finite" in reason
    ok, reason = hl_dense.is_safe_dense_command(np.zeros(12))
    assert not ok and "want 29" in reason


def test_an_absurd_joint_target_is_refused_as_a_units_failure():
    """The G1's widest joint range is under ~2.9 rad from zero, so 6.5 is not an aggressive
    pose — it is a decode or a units error, and the per-joint slew downstream would happily
    walk the robot there one tick at a time."""
    bad = np.zeros(hl_dense.G1_DOF)
    bad[0] = 12.0
    ok, reason = hl_dense.is_safe_dense_command(bad)
    assert not ok and "sanity bound" in reason
    assert hl_dense.is_safe_dense_command(np.full(hl_dense.G1_DOF, 2.5))[0]


def test_the_velocity_is_a_causal_backward_difference():
    chunk = np.stack([np.full(hl_dense.G1_DOF, 0.1 * i) for i in range(4)])
    vel = hl_dense.joint_velocity(chunk, 1.0 / 30.0)
    np.testing.assert_allclose(vel[0], 0.0, atol=0.0)     # no earlier frame to difference
    np.testing.assert_allclose(vel[1:], 0.1 * 30.0)
    np.testing.assert_allclose(hl_dense.joint_velocity(chunk[:1], 1 / 30), 0.0)


# --------------------------------------------------------------------------- the wire
def _wire(driver: S0Driver) -> dict:
    return json.loads(driver._latest)


def test_the_dense_frame_is_exactly_what_vive_live_source_reads():
    """Widths, names and the mode tag — the three things the consumer validates.

    ``ViveLiveSource`` reads ``joint_pos``/``joint_vel``/``joint_names`` off a ``mode=="dense"``
    frame and DROPS the frame if the names are absent or not 29 long, so shipping them every
    frame is the contract rather than an optimization.
    """
    driver = S0Driver(port=8999)   # never started: send_dense only fills the latch
    q = np.arange(hl_dense.G1_DOF, dtype=np.float64) * 0.01
    driver.send_dense(q, q * 2.0, _NAMES)
    msg = _wire(driver)
    assert msg["mode"] == "dense"
    assert len(msg["joint_pos"]) == len(msg["joint_vel"]) == len(msg["joint_names"]) == 29
    assert msg["joint_names"] == list(_NAMES)
    assert "f_cmd" not in msg, "a position-only dense frame must carry no f_cmd key"
    np.testing.assert_allclose(msg["joint_pos"], q)
    assert set(msg) == {"mode", "joint_pos", "joint_vel", "joint_names", "seq", "t"}


def test_a_dense_frame_carries_f_cmd_only_when_given_one():
    driver = S0Driver(port=8999)
    q = np.zeros(hl_dense.G1_DOF)
    driver.send_dense(q, q, _NAMES, f_cmd=np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]))
    assert _wire(driver)["f_cmd"] == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]


def test_a_ragged_dense_frame_is_refused_rather_than_published():
    """Three arrays that must agree; a mismatch would put values under the wrong names."""
    driver = S0Driver(port=8999)
    with pytest.raises(ValueError, match="width mismatch"):
        driver.send_dense(np.zeros(29), np.zeros(28), _NAMES)
    with pytest.raises(ValueError, match="width mismatch"):
        driver.send_dense(np.zeros(29), np.zeros(29), _NAMES[:-1])


def test_the_three_point_wire_is_byte_identical_to_before():
    """The same S0Driver serves the left_only planner, which is in use on the rig."""
    driver = S0Driver(port=8999)
    driver.send(np.arange(9, dtype=float), np.arange(12, dtype=float))
    msg = _wire(driver)
    assert set(msg) == {"mode", "pos", "orn", "seq", "t"}
    assert msg["mode"] == "3pt"
    driver.send(np.zeros(9), np.zeros(12), f_cmd=np.zeros(3))
    assert _wire(driver)["f_cmd"] == [0.0, 0.0, 0.0]


def test_the_sequence_number_is_shared_across_both_wire_modes():
    """One monotonic counter, because the consumer dedups on it and a mode switch mid-session
    must not hand it a sequence that goes backwards."""
    driver = S0Driver(port=8999)
    driver.send(np.zeros(9), np.zeros(12))
    first = _wire(driver)["seq"]
    driver.send_dense(np.zeros(29), np.zeros(29), _NAMES)
    assert _wire(driver)["seq"] == first + 1


# --------------------------------------------------------------------------- the schedule
def test_the_window_schedule_matches_the_completions_own_constants():
    """Duplicated numbers, pinned. ``hl_dense`` cannot import ``mc_frames`` (it must load with
    no ``third_party`` on the path), so the three schedule constants are copied and checked
    against the submodule's source here — every latency claim in the module docstring is
    derived from 30/10."""
    from pathlib import Path
    import re

    source = (Path(__file__).resolve().parents[1] / "third_party" / "UMI_collection"
              / "motion_completion" / "mc_frames.py")
    if not source.exists():
        pytest.skip("UMI_collection submodule is not checked out")
    text = source.read_text()
    match = re.search(r"def window_starts\(n_frames: int, window: int, overlap: int\)", text)
    assert match is not None, "mc_frames no longer defines window_starts(n, window, overlap)"
    # The stride is what the docstring's "2 windows for a 50-step chunk" rests on.
    assert hl_dense.MC_STRIDE == hl_dense.MC_WINDOW - hl_dense.MC_OVERLAP == 20
    assert hl_dense.MC_RATE_HZ == 30.0
    # A 50-step chunk really is two windows on this schedule.
    starts = [s for s in range(0, 50, hl_dense.MC_STRIDE) if s + hl_dense.MC_WINDOW <= 50]
    assert starts == [0, 20]


def test_a_completer_with_no_checkout_reports_a_reason_instead_of_raising():
    """A fresh clone must get a REFUSAL WITH A REASON, not a traceback — and the planner turns
    that reason into "serve a 3pt low level instead", which it cannot do from an exception."""
    completer = hl_dense.ChunkCompleter(umi_dir="/definitely/not/here")
    assert not completer.available
    assert "motion_completion" in completer.disabled_reason
    with pytest.raises(RuntimeError, match="unavailable"):
        completer.complete(object())


def test_the_post_process_default_is_foot_ik_off_and_the_override_is_episode_scoped():
    """The measured verdict, pinned. Foot IK is 1376 ms/chunk at its CHEAPEST setting against a
    481 ms re-plan budget (``scripts/ablate_mc_foot_ik.py``), so the default must stay off — and
    the override must land on the EPISODE, because switching a post-process inside a chunk would
    put two different bodies in one motion."""
    assert hl_dense.DEFAULT_FOOT_FIX == "none"
    completer = hl_dense.ChunkCompleter(umi_dir="/definitely/not/here")
    assert completer._foot_fix == "none" and completer._foot_ik_iters is None
    completer.set_post_process(wrist_ik=False, foot_fix="inertialize", foot_ik_iters=40)
    assert (completer._wrist_ik, completer._foot_fix, completer._foot_ik_iters) == (
        False, "inertialize", 40)
    # A bare call clears the iteration override rather than silently keeping the last arm's.
    completer.set_post_process(foot_fix="none")
    assert completer._foot_ik_iters is None and completer._wrist_ik is False
    # The planner's own route to the same knob (--mc-foot-ik / --mc-foot-ik-iters).
    forced = hl_dense.ChunkCompleter(umi_dir="/definitely/not/here",
                                     foot_fix="inertialize", foot_ik_iters=40)
    assert (forced._foot_fix, forced._foot_ik_iters) == ("inertialize", 40)


def test_the_stage_split_sums_the_windows_and_names_the_transport():
    """``stage_ms`` is what the ablation reads, so it has to add up: the backend's own total is
    the sum of its stages, and the gap to the wall-clock ``ms`` is the pipe, never negative."""
    chunk = hl_dense.CompletedChunk(
        joint_pos=np.zeros((4, hl_dense.G1_DOF)), joint_vel=np.zeros((4, hl_dense.G1_DOF)),
        joint_names=tuple(f"j{i}" for i in range(hl_dense.G1_DOF)),
        fk_pos=np.zeros((4, 30, 3)), fk_quat=np.zeros((4, 30, 4)),
        body_names=tuple(f"b{i}" for i in range(30)),
        root_pos=np.zeros((4, 3)), root_quat=np.zeros((4, 4)),
        valid=np.ones(4, dtype=bool), cond=object(), windows=2, ms=100.0,
        timings=({"t_flow_ms": 5.0, "t_decode_ms": 12.0, "t_ik_ms": 20.0, "t_foot_ms": 0.0,
                  "t_total_ms": 37.0},
                 {"t_flow_ms": 5.0, "t_decode_ms": 13.0, "t_ik_ms": 21.0, "t_foot_ms": 0.0,
                  "t_total_ms": 39.0}))
    split = chunk.stage_ms()
    assert split["flow"] == 10.0 and split["ik"] == 41.0 and split["foot"] == 0.0
    assert split["backend"] == 76.0
    assert split["transport"] == pytest.approx(24.0)
    # No timings at all (a bench that never asked) must not go negative or raise.
    bare = hl_dense.CompletedChunk(**{**chunk.__dict__, "timings": ()})
    assert bare.stage_ms()["transport"] == pytest.approx(100.0)


# --------------------------------------------------------------------------- the 5pt wire
def _pose(xyz, rotmat=None) -> tuple:
    return (np.asarray(xyz, dtype=float), np.eye(3) if rotmat is None else np.asarray(rotmat))


def test_the_keypoint_readout_is_the_low_levels_own_relativisation():
    """``p_local = R_pelvis^T (p - p_pelvis)``, ``q_local = q_pelvis^-1 (x) q_body``.

    Checked against a hand-computed case with a NON-identity pelvis, because with an identity
    one a dropped ``R^T`` and a correct implementation return the same answer — and the deploy
    anchor's pelvis IS identity, so the rig would never catch the omission.
    """
    from scipy.spatial.transform import Rotation

    R_pelvis = Rotation.from_euler("z", 90.0, degrees=True).as_matrix()
    p_pelvis = np.array([1.0, 2.0, 0.75])
    body_p = np.array([[1.0, 2.5, 0.75]])                       # +0.5 along world +y
    body_R = np.eye(3)[None, ...]
    pos, orn = hl_dense.relativize_keypoints(body_p[None, ...], body_R[None, ...],
                                             p_pelvis, R_pelvis)
    # World +y is the pelvis's +x after a +90 deg yaw, so the point reads as 0.5 m AHEAD.
    np.testing.assert_allclose(pos[0], [0.5, 0.0, 0.0], atol=1e-9)
    # And the body's identity rotation reads as -90 deg about z in the pelvis frame.
    got = Rotation.from_quat(orn[0]).as_euler("xyz", degrees=True)
    np.testing.assert_allclose(got, [0.0, 0.0, -90.0], atol=1e-6)


def test_every_quaternion_is_canonicalised_onto_the_plus_w_branch():
    """``q`` and ``-q`` are one rotation and two DIFFERENT obs vectors, and the 5pt policy saw
    only ``+w`` (na475b60's normalizer: w mean +0.73 hands / +0.98 head / +0.96 feet), so an
    antipode is ~20 sigma OOD. Nothing upstream pins the branch, and a chest pose near the
    180 deg seam flips every keypoint at once."""
    from scipy.spatial.transform import Rotation

    # 179.9 deg about x: the branch scipy returns for this is the one that matters.
    R = Rotation.from_euler("x", 179.9, degrees=True).as_matrix()
    _, orn = hl_dense.relativize_keypoints(np.zeros((1, 1, 3)), R[None, None, ...],
                                           np.zeros(3))
    assert orn[0, 3] >= 0.0, "the scalar part must be non-negative on the wire"
    # ...and it is still the SAME rotation, i.e. the flip is a re-representation, not a change.
    np.testing.assert_allclose(Rotation.from_quat(orn[0]).as_matrix(), R, atol=1e-9)
    # The flip is on the xyzw SCALAR (index 3). Flipping index 0 instead would leave a
    # negative w on the wire for this input, which is exactly the bug being pinned.
    assert float(orn[0, 3]) == pytest.approx(abs(float(orn[0, 3])))


def test_the_ready_pose_five_point_frame_is_the_gates_reference_and_the_wire_starts_there():
    """THE REASON THE 5pt FAMILY NEEDS NO CLUTCH, as a number.

    The low level's keypoint pre-arm gate compares the broadcast wire against
    ``_compute_vr_fk(motion_command_t[0])`` = FK of the READY POSE. Read out against the
    anchor's own pelvis, the ready pose's hands land at ``(0.4418, +-0.1486, 0.0952)`` — the
    value holosoma's own gate reference is documented at, to 0.1 mm — so the idle wire and the
    gate agree by construction.
    """
    pelvis_z = 0.7567
    anchor = type("A", (), {})()
    anchor.pos = {"left": np.array([0.4418, 0.1486, 0.8519]),
                  "right": np.array([0.4418, -0.1486, 0.8519]),
                  "head": np.array([-0.004, 0.0, 1.1507])}
    anchor.rotmat = {r: np.eye(3) for r in anchor.pos}
    anchor.foot_pos = {"left": np.array([-0.0014, 0.1185, 0.035]),
                       "right": np.array([-0.0014, -0.1185, 0.035])}
    anchor.foot_rotmat = {"left": np.eye(3), "right": np.eye(3)}
    readout = hl_dense.KeypointReadout(pelvis_pos=(0.0, 0.0, pelvis_z))
    pos, orn = hl_dense.ready_pose_five_point(anchor, readout)
    assert (pos.size, orn.size) == (15, 20)
    np.testing.assert_allclose(pos[0:3], [0.4418, 0.1486, 0.0952], atol=1e-4)
    np.testing.assert_allclose(pos[3:6], [0.4418, -0.1486, 0.0952], atol=1e-4)
    # The feet sit BELOW the pelvis, which is what makes them a negative z here.
    assert pos[11] < -0.7 and pos[14] < -0.7
    wire = hl_dense.FivePointWire(readout, pos, orn)
    np.testing.assert_allclose(wire.idle()[0], pos, atol=0.0)
    ok, note = wire.gate_report(wire.idle())
    assert ok and "PASS" in note and "LEFT hand 0 mm" in note


def test_the_five_point_wire_never_chases_the_measured_pose():
    """The dense wire tracks measured while un-armed; this one must NOT, and the difference is
    a fact about the gate rather than a preference: the keypoint gate's reference is the FROZEN
    reference clip, so moving the wire toward the encoders moves it AWAY from what the gate
    wants. Returning None here is what makes the planner's idle re-sync a no-op."""
    readout = hl_dense.KeypointReadout(pelvis_pos=(0.0, 0.0, 0.7567))
    wire = hl_dense.FivePointWire(readout, np.zeros(15), np.tile([0.0, 0, 0, 1.0], 5))
    rframe = type("R", (), {"dof_pos": np.full(hl_dense.G1_DOF, 0.3), "vive_armed": False})()
    assert wire.measured(rframe) is None
    assert "not re-based" in wire.sync(rframe)
    wire.commit(wire.idle())          # a no-op, and it must not raise
    np.testing.assert_allclose(wire.idle()[0], np.zeros(15))


def test_a_bad_keypoint_frame_is_refused_before_the_wire():
    """``ViveLiveSource`` validates the (pos, orn) width PAIR and nothing else — not
    finiteness, not unit norm — so this is the gate. A de-normalised quaternion matters
    because the low level COMPOSES it with the arm clutch's anchor: it scales the commanded
    rotation instead of failing."""
    good_pos = np.tile([0.4, 0.1, 0.1], 5)
    good_orn = np.tile([0.0, 0.0, 0.0, 1.0], 5)
    assert hl_dense.is_safe_keypoint_command(good_pos, good_orn) == (True, "")
    bad = good_pos.copy()
    bad[4] = np.nan
    ok, why = hl_dense.is_safe_keypoint_command(bad, good_orn)
    assert not ok and "index 4" in why
    ok, why = hl_dense.is_safe_keypoint_command(np.tile([9.0, 0, 0], 5), good_orn)
    assert not ok and "sanity bound" in why
    stretched = good_orn.copy()
    stretched[4:8] = [0.0, 0.0, 0.0, 1.5]
    ok, why = hl_dense.is_safe_keypoint_command(good_pos, stretched)
    assert not ok and "RIGHT hand" in why and "1.5" in why
    for pos, orn, want in ((np.zeros(9), good_orn, "want 15"),
                           (good_pos, np.zeros(12), "want 20")):
        ok, why = hl_dense.is_safe_keypoint_command(pos, orn)
        assert not ok and want in why


def test_the_five_point_wire_publishes_exactly_what_vive_live_source_reads():
    driver = S0Driver(port=8999)   # never started: send_5pt only fills the latch
    readout = hl_dense.KeypointReadout(pelvis_pos=(0.0, 0.0, 0.7567))
    pos = np.arange(15, dtype=float) * 0.01
    orn = np.tile([0.0, 0.0, 0.0, 1.0], 5)
    wire = hl_dense.FivePointWire(readout, pos, orn)
    ok, reason = wire.publish(driver, wire.idle(), None)
    assert ok, reason
    msg = _wire(driver)
    assert msg["mode"] == "5pt"
    assert (len(msg["pos"]), len(msg["orn"])) == (15, 20)
    assert "f_cmd" not in msg
    assert set(msg) == {"mode", "pos", "orn", "seq", "t"}
    # Force rides the SAME wire in this family too (ViveLiveSource parses f_cmd on any frame).
    wire.publish(driver, wire.idle(), np.arange(6, dtype=float))
    assert _wire(driver)["f_cmd"] == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    # And a ragged frame is refused by the DRIVER as well, not only by the gate above.
    with pytest.raises(ValueError, match="width mismatch"):
        driver.send_5pt(np.zeros(15), np.zeros(19))
    # The walk skill's legs: an OPTIONAL `ref_joints` block, exactly what ViveLiveSource parses
    # ({names[29], pos[29], vel[29]}), absent unless handed over (byte-identical otherwise).
    names = [f"j{i}_joint" for i in range(29)]
    q = np.linspace(-1.0, 1.0, 29)
    qd = np.linspace(2.0, -2.0, 29)
    ok, reason = wire.publish(driver, wire.idle(), None, ref_joints=(names, q, qd))
    assert ok, reason
    msg = _wire(driver)
    assert set(msg) == {"mode", "pos", "orn", "seq", "t", "ref_joints"}
    assert msg["ref_joints"]["names"] == names
    assert msg["ref_joints"]["pos"] == q.tolist() and msg["ref_joints"]["vel"] == qd.tolist()
    ok, _ = wire.publish(driver, wire.idle(), None)
    assert ok and "ref_joints" not in _wire(driver)
    # A malformed block is refused HERE with a reason, never silently dropped on the robot.
    for bad in ((names[:12], q[:12], qd[:12]), (names, q, np.full(29, np.nan)), (["j"] * 29, q, qd)):
        with pytest.raises(ValueError, match="ref_joints"):
            driver.send_5pt(np.zeros(15), np.tile([0.0, 0.0, 0.0, 1.0], 5), ref_joints=bad)
    # `extra` may not smuggle one in either: the key is the wire's own.
    with pytest.raises(ValueError, match="collide"):
        driver.send_5pt(np.zeros(15), np.tile([0.0, 0.0, 0.0, 1.0], 5), extra={"ref_joints": {}})


def test_the_five_point_bodies_and_gate_count_match_the_submodules_own_contract():
    """Three duplicated facts, pinned: the body list, its ORDER, and how many are gated.

    ``mc_frames.KEYPOINT_MODES`` is holosoma's ``VR_5POINT_BODIES`` quoted in one place (its
    own test re-reads ``robot_config.py`` with ``ast``), and ``hl_dense`` cannot import it —
    this module must load with no ``third_party`` on the path. A drift here puts the head's
    quaternion in a foot's slot with no error anywhere.
    """
    from pathlib import Path
    import re

    source = (Path(__file__).resolve().parents[1] / "third_party" / "UMI_collection"
              / "motion_completion" / "mc_frames.py")
    if not source.exists():
        pytest.skip("UMI_collection submodule is not checked out")
    text = source.read_text()
    block = re.search(r'"5pt":\s*\((.*?)\),\s*\n\s*"29dof"', text, re.S)
    assert block is not None, "mc_frames.KEYPOINT_MODES no longer spells a '5pt' entry"
    assert tuple(re.findall(r'"([a-z_]+)"', block.group(1))) == hl_dense.FIVE_POINT_BODIES
    assert len(hl_dense.FIVE_POINT_LABELS) == len(hl_dense.FIVE_POINT_BODIES)
    # The gated count is holosoma's `_VR_5POINT_OPERATOR_PLACED`: the two FEET are derived.
    assert hl_dense.FIVE_POINT_GATED == len(hl_dense.FIVE_POINT_BODIES) - 2
    assert hl_dense.WIRE_MODES == ("dense", "5pt")


def test_the_stats_summary_survives_having_nothing_to_summarize():
    stats = hl_dense.DenseStats()
    assert "no completion yet" in stats.summary()
    stats.note_error(RuntimeError("backend wedged"))
    assert "backend wedged" in stats.summary() and "1 error" in stats.summary()


# ------------------------------------------------------ the POST-COMPLETION command anchor
def test_post_anchor_off_is_byte_identical_on_both_wires():
    """The default must not move a single published number, in either family.

    ``mode="off"`` is not "a small correction of zero" — it is the previous code path, and this
    is the test that says so: the dense clutch re-bases on measured and the 5pt wire publishes
    the readout untouched, both exactly as before the mode existed.
    """
    off = hl_dense.PostAnchorConfig()
    assert not off.carrying and "previous behaviour" in off.describe()

    base = np.zeros(hl_dense.G1_DOF)
    clutch = hl_dense.JointClutch(base, post=off)
    clutch.commit(np.full(hl_dense.G1_DOF, 0.5))          # pretend a chunk executed
    measured = np.full(hl_dense.G1_DOF, 0.1)
    assert "MEASURED" in clutch.sync(measured)
    np.testing.assert_allclose(clutch.base, measured)     # the carry was DISCARDED

    readout = hl_dense.KeypointReadout(pelvis_pos=(0.0, 0.0, 0.7567))
    ready_pos, ready_orn = np.arange(15) * 0.01, np.tile([0.0, 0, 0, 1.0], 5)
    wire = hl_dense.FivePointWire(readout, ready_pos, ready_orn)
    chunk_pos = np.stack([ready_pos + 0.3 * i for i in range(4)])
    chunk_orn = np.tile(ready_orn, (4, 1))
    got_pos, got_orn = wire.clutch.apply(chunk_pos, chunk_orn)
    np.testing.assert_allclose(got_pos, chunk_pos, atol=0.0)   # the IDENTITY, not a re-base
    np.testing.assert_allclose(got_orn, chunk_orn, atol=0.0)


def test_the_dense_carry_survives_the_chunk_boundary_and_the_leak_bounds_it():
    """``command`` keeps the accumulated command; the leak is what stops it running away.

    Unclamped integration diverges 224 mm on the recorded takes (``hl_anchor``), so the two
    halves are pinned together: the carry happens, AND it is bounded per joint.
    """
    post = hl_dense.PostAnchorConfig(mode="command", eps_joint_rad=0.10)
    clutch = hl_dense.JointClutch(np.zeros(hl_dense.G1_DOF), post=post)
    measured = np.zeros(hl_dense.G1_DOF)

    # FIRST Predict after arm: nothing executed yet, so it bootstraps from measured however the
    # mode is set — a carried command from before an operator intervention is never branched on.
    assert "MEASURED" in clutch.sync(measured)

    clutch.commit(np.full(hl_dense.G1_DOF, 0.04))     # a chunk executed, ending 0.04 rad out
    label = clutch.sync(measured)                     # the robot did NOT move
    assert "CARRIED" in label
    np.testing.assert_allclose(clutch.base, np.full(hl_dense.G1_DOF, 0.04))

    # Now wind past the bound: the clamp is WRITTEN BACK, so the state cannot run away.
    clutch.commit(np.full(hl_dense.G1_DOF, 0.40))
    assert "clamped" in clutch.sync(measured)
    np.testing.assert_allclose(clutch.base, np.full(hl_dense.G1_DOF, 0.10))
    assert clutch.post_stats()["n_clamped"] == 1

    # Step 0 of the next chunk still lands on the base bit-exactly — the no-op the pre-arm gate
    # and the first executed step both depend on.
    published = clutch.apply(np.stack([np.full(hl_dense.G1_DOF, 1.234)] * 3))
    np.testing.assert_allclose(published[0], clutch.base)


def test_the_dense_leak_is_per_joint_not_on_the_norm():
    """One joint may not wind up while the others sit still, which a norm bound would allow."""
    post = hl_dense.PostAnchorConfig(mode="command", eps_joint_rad=0.05)
    clutch = hl_dense.JointClutch(np.zeros(hl_dense.G1_DOF), post=post)
    measured = np.zeros(hl_dense.G1_DOF)
    wound = np.zeros(hl_dense.G1_DOF)
    wound[7] = 1.0
    clutch.commit(wound)
    clutch.sync(measured)
    assert clutch.base[7] == pytest.approx(0.05)
    assert np.all(clutch.base[np.arange(hl_dense.G1_DOF) != 7] == 0.0)


def test_contact_decays_the_dense_carry_instead_of_advancing_it():
    """A command that keeps advancing into a blocked hand is a press, not a reach. Also: an
    UNKNOWN force must not disable the guard, so None is not treated as free space."""
    post = hl_dense.PostAnchorConfig(mode="command", eps_joint_rad=0.30,
                                     contact_n=1.5, decay=0.5)
    clutch = hl_dense.JointClutch(np.zeros(hl_dense.G1_DOF), post=post)
    measured = np.zeros(hl_dense.G1_DOF)
    clutch.commit(np.full(hl_dense.G1_DOF, 0.20))
    clutch.sync(measured, force_n=8.0)                       # in contact
    np.testing.assert_allclose(clutch.base, np.full(hl_dense.G1_DOF, 0.10))
    assert clutch.post_stats()["n_contact_pullback"] == 1
    # None: no reading, so no cut-out is claimed either way — the carry is kept, not decayed.
    clutch.commit(np.full(hl_dense.G1_DOF, 0.20))
    clutch.sync(measured, force_n=None)
    np.testing.assert_allclose(clutch.base, np.full(hl_dense.G1_DOF, 0.20))


def test_the_keypoint_carry_offsets_position_and_rotation_by_a_constant():
    """The 5pt integrator. Step 0 lands on the base, and the ROTATION offset must be a fixed
    pelvis-frame rotation — composing on the wrong side conjugates it by the chunk's own motion,
    so a hand that turns would also drift."""
    from fuji.utils.rot6d import quat_xyzw_to_rotmat, rotmat_to_quat_xyzw, rotvec_to_rotmat

    n = len(hl_dense.FIVE_POINT_BODIES)
    base_pos = np.tile([0.40, 0.10, 0.05], n)
    base_rot = np.stack([rotvec_to_rotmat(np.array([0.0, 0.0, 0.20]))] * n)
    base_orn = rotmat_to_quat_xyzw(base_rot).reshape(-1)
    post = hl_dense.PostAnchorConfig(mode="command")
    clutch = hl_dense.KeypointClutch(base_pos, base_orn, post=post, n_points=n)

    # A chunk that starts somewhere else and TURNS: 3 steps, yaw 0 -> 0.1 -> 0.2 rad.
    chunk_rot = np.stack([np.stack([rotvec_to_rotmat(np.array([0.0, 0.0, 0.1 * h]))] * n)
                          for h in range(3)])
    chunk_pos = np.stack([np.tile([0.0, 0.0, 0.0], n) + 0.05 * h for h in range(3)])
    chunk_orn = rotmat_to_quat_xyzw(chunk_rot).reshape(3, -1)

    k = clutch.n_carried
    got_pos, got_orn = clutch.apply(chunk_pos, chunk_orn)
    # step 0 of the CARRIED points = the base, bit-exactly
    np.testing.assert_allclose(got_pos.reshape(3, n, 3)[0, :k],
                               base_pos.reshape(n, 3)[:k], atol=1e-12)
    got_rot = quat_xyzw_to_rotmat(got_orn.reshape(3, n, 4))
    np.testing.assert_allclose(got_rot[0, :k], base_rot[:k], atol=1e-9)
    # The offset E = R_base @ R_chunk[0]^T is CONSTANT across the chunk.
    offset = base_rot[0] @ chunk_rot[0][0].T
    for h in range(3):
        np.testing.assert_allclose(got_rot[h][0], offset @ chunk_rot[h][0], atol=1e-9)
    # and the position delta is the chunk's own, unchanged.
    np.testing.assert_allclose(got_pos[2] - got_pos[0], chunk_pos[2] - chunk_pos[0], atol=1e-12)
    # every published quaternion stays on the +w branch the 5pt policy was trained on
    assert np.all(got_orn.reshape(3, n, 4)[..., 3] >= 0.0)
    # THE FEET ARE NOT CARRIED: they pass through absolutely, because re-basing the completion's
    # stance onto the ready pose's 38 deg crouch is a LEG command (measured 74.7 mm on the first
    # chunk of a real completion). Same operator-placed / derived split holosoma's gate makes.
    assert k == hl_dense.FIVE_POINT_GATED == n - 2
    np.testing.assert_allclose(got_pos.reshape(3, n, 3)[:, k:],
                               chunk_pos.reshape(3, n, 3)[:, k:], atol=0.0)
    np.testing.assert_allclose(got_rot[:, k:], chunk_rot[:, k:], atol=1e-9)


def test_the_keypoint_carry_freezes_when_there_is_nothing_to_bound_it_against():
    """No measured keypoints ⇒ the anchor is HELD, never integrated blind. This is the whole
    reason ``KeypointClutch`` demands a reference instead of defaulting one: on this wire
    "where the robot is" only exists as 29 encoder readings, and clamping to the ready pose
    instead would cap the task (a real reach moves the hand ~48 cm)."""
    n = len(hl_dense.FIVE_POINT_BODIES)
    ready_pos, ready_orn = np.tile([0.4, 0.1, 0.05], n), np.tile([0.0, 0, 0, 1.0], n)
    post = hl_dense.PostAnchorConfig(mode="command")
    clutch = hl_dense.KeypointClutch(ready_pos, ready_orn, post=post, n_points=n)
    readout = hl_dense.KeypointReadout(pelvis_pos=(0.0, 0.0, 0.7567))
    wire = hl_dense.FivePointWire(readout, ready_pos, ready_orn, clutch=clutch,
                                  measured_frame=None)
    rframe = type("R", (), {"dof_pos": np.zeros(hl_dense.G1_DOF), "vive_armed": True})()

    assert "nothing executed yet" in wire.sync(rframe)
    carried = (ready_pos + 0.20, ready_orn)
    wire.commit(carried)
    label = wire.sync(rframe)
    assert "FROZEN" in label and "held, not advanced" in label
    np.testing.assert_allclose(wire.clutch.base[0], carried[0])   # held, not reset, not advanced
    assert wire.clutch.post_stats()["n_frozen_no_reference"] == 1


def test_the_keypoint_leak_is_bounded_against_the_measured_frame():
    """With a reference the carry IS bounded — per keypoint, on the position norm and on the
    geodesic orientation deviation, and the clamp is written back into the state."""
    n = len(hl_dense.FIVE_POINT_BODIES)
    ready_pos, ready_orn = np.tile([0.0, 0.0, 0.0], n), np.tile([0.0, 0, 0, 1.0], n)
    post = hl_dense.PostAnchorConfig(mode="command", eps_m=0.02, eps_rad=0.05)
    clutch = hl_dense.KeypointClutch(ready_pos, ready_orn, post=post, n_points=n)
    readout = hl_dense.KeypointReadout(pelvis_pos=(0.0, 0.0, 0.7567))
    measured = (ready_pos.copy(), ready_orn.copy())
    wire = hl_dense.FivePointWire(readout, ready_pos, ready_orn, clutch=clutch,
                                  measured_frame=lambda _rf: measured)
    rframe = type("R", (), {"dof_pos": np.zeros(hl_dense.G1_DOF), "vive_armed": True})()

    wire.commit((np.tile([0.5, 0.0, 0.0], n), ready_orn))      # 0.5 m past the robot
    assert "clamped" in wire.sync(rframe)
    k = wire.clutch.n_carried
    gaps = hl_dense.keypoint_gaps_m(wire.clutch.base[0], measured[0])
    assert np.all(gaps[:k] <= 0.02 + 1e-9) and np.all(gaps[:k] > 0.0)
    # The un-carried tail is NOT clamped, because it is never carried — clamping a foot would be
    # clamping the completion's own stance against the ready crouch.
    assert np.all(gaps[k:] > 0.02)
    assert wire.clutch.drift_m() <= 0.02 + 1e-9      # drift excludes the feet for that reason

    # A big carried ROTATION is clamped on the geodesic, not silently accepted.
    from fuji.utils.rot6d import quat_xyzw_to_rotmat, rotmat_to_quat_xyzw, rotmat_to_rotvec, rotvec_to_rotmat

    turned = rotmat_to_quat_xyzw(
        np.stack([rotvec_to_rotmat(np.array([0.0, 0.0, 1.0]))] * n)).reshape(-1)
    wire.commit((wire.clutch.base[0], turned))
    wire.sync(rframe)
    got = quat_xyzw_to_rotmat(wire.clutch.base[1].reshape(n, 4))
    for j in range(k):
        assert float(np.linalg.norm(rotmat_to_rotvec(got[j]))) <= 0.05 + 1e-9
    # and again, the un-carried tail keeps its own 1 rad turn: never carried, never clamped.
    for j in range(k, n):
        assert float(np.linalg.norm(rotmat_to_rotvec(got[j]))) == pytest.approx(1.0, abs=1e-9)


def test_a_reset_drops_the_carried_command_on_both_wires():
    """Reset / re-arm must not branch from a command issued before the operator intervened —
    the same rule ``hl_anchor.ChunkAnchor.reset`` states, and the reason the planner's Reset
    handler calls ``wire.clutch.reset()`` for whichever family is live."""
    post = hl_dense.PostAnchorConfig(mode="command")
    clutch = hl_dense.JointClutch(np.zeros(hl_dense.G1_DOF), post=post)
    clutch.commit(np.full(hl_dense.G1_DOF, 0.3))
    clutch.reset()
    np.testing.assert_allclose(clutch.base, np.zeros(hl_dense.G1_DOF))
    assert "MEASURED" in clutch.sync(np.full(hl_dense.G1_DOF, 0.07))   # bootstraps again

    n = len(hl_dense.FIVE_POINT_BODIES)
    ready_pos, ready_orn = np.tile([0.4, 0.1, 0.05], n), np.tile([0.0, 0, 0, 1.0], n)
    kp = hl_dense.KeypointClutch(ready_pos, ready_orn, post=post, n_points=n)
    kp.commit((ready_pos + 0.1, ready_orn))
    assert kp.drift_m() > 0.0
    kp.reset()
    np.testing.assert_allclose(kp.base[0], ready_pos)
    assert kp.drift_m() == 0.0


def test_the_post_anchor_config_refuses_an_unsafe_leak():
    """The eps bounds are the whole safety story of the mode, so they are validated at
    construction rather than clamped silently — a typo must not ask a robot for a lead it
    would take four ticks to unwind."""
    with pytest.raises(ValueError, match="post-anchor mode"):
        hl_dense.PostAnchorConfig(mode="command_ish")
    with pytest.raises(ValueError, match="MAX_POST_EPS_RAD"):
        hl_dense.PostAnchorConfig(eps_joint_rad=hl_dense.MAX_POST_EPS_RAD + 0.01)
    with pytest.raises(ValueError, match="eps_m must be"):
        hl_dense.PostAnchorConfig(eps_m=1.0)
    with pytest.raises(ValueError, match="decay must be"):
        hl_dense.PostAnchorConfig(decay=0.0)
    # eps 0 collapses to "off" rather than to unbounded integration.
    clutch = hl_dense.JointClutch(
        np.zeros(hl_dense.G1_DOF),
        post=hl_dense.PostAnchorConfig(mode="command", eps_joint_rad=0.0))
    clutch.commit(np.full(hl_dense.G1_DOF, 0.5))
    clutch.sync(np.zeros(hl_dense.G1_DOF))
    np.testing.assert_allclose(clutch.base, np.zeros(hl_dense.G1_DOF))
