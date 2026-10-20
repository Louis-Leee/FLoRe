"""Contract tests for the ``diffusion_umi_wholebody`` DEPLOY + OFFLINE-INFERENCE paths.

The training-side contract is ``tests/test_diffusion_umi_wholebody.py``. This is what happens to
a 43-D chunk AFTER it is sampled: the wire it becomes (``flore/s1_policy/serving/hl_wholebody.py``,
``scripts/hl_serving_wholebody.py``) and the page it is drawn on
(``flore/s1_policy/inference/wholebody.py`` + ``wholebody_viewer.py``,
``scripts/infer_s1_wholebody.py``).

Everything pinned here fails SILENTLY otherwise — again the selection criterion:

* any permutation of 29 joint angles is a valid robot pose, so the policy-order == SDK-order
  assumption that makes ``DenseWire``'s remap an identity cannot be left implicit;
* the absolute joints are in the DATASET's ``cond_mode="home"`` posture space, ~0.5 rad from the
  pose the robot holds — published absolutely that is a real command, and the only thing standing
  between the two is the joint clutch making step 0 a no-op;
* the PROPRIO has the same gap and a clutch cannot fix an input, so a ring that stops advancing
  reads as a real measurement of a different posture. (``push_proprio_joints`` really did fail to
  advance the GRIPPER ring once: both jaw columns read 0.0 m — the CLOSED end of a legal travel —
  for a whole session while ``push_gripper`` updated a latch nothing read. Regression test below.)
* ``mask[..., []].all(-1)`` is ``True`` and a norm over 0 columns is exactly ``0.0``, so the
  keypoint metric arithmetic reports a PERFECT 0.00 cm position error for a layout with no
  position columns;
* ``--legs hold`` drops 12 of 29 predicted columns, which is shape-correct and invisible.

The two SIBLING paths must stay untouched: a keypoint layout is refused by name at every entry
point here, and a wholebody checkpoint is refused by the keypoint ones.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from flore.s1_policy.data.layout import (
    JOINT_DIM,
    WB_JOINT_NAMES,
    get_layout,
)
from flore.s1_policy.data.transforms import GRIPPER_WIDTH_REF_M
from flore.s1_policy.serving import hl_dense, hl_wholebody
from flore.s1_policy.serving.hl_online import replay_only_model
from flore.s1_policy.serving.live_robot import G1_29DOF_SDK_DOF_NAMES

# Reuse the sibling's exporter-backed BUILDERS rather than a second synthetic writer: a drifting
# second schema writer is how a serving test starts passing against data no converter produces.
# The builders are imported and the fixtures re-declared under local names — importing the
# sibling's fixtures directly works at runtime but shadows them (ruff F811), and a `noqa` on a
# test signature is worse than four lines of fixture.
from tests.test_diffusion_umi_wholebody import _augment
from tests.test_hl_dataset import _write_hl_zarr

LAY = get_layout("wholebody")
#: A ready pose stand-in with a distinct value per joint, so a permutation shows up as a value
#: mismatch rather than a shape error.
READY = np.arange(JOINT_DIM, dtype=np.float64) * 0.01 - 0.14


@pytest.fixture()
def source_zarr(tmp_path):
    """The 3-tracker rig with NO ``wb_*`` arrays: 3 pucks, 2 cameras, 2 F/T, 2 grippers."""
    return _write_hl_zarr(
        tmp_path / "dual.zarr.zip",
        roles=("left", "right", "head"),
        force_roles=("left", "right"),
        gripper_roles=("left", "right"),
        n_cameras=2,
        attrs_override={"hands_mode": "both_head_dual"},
    )


@pytest.fixture()
def wb_zarr(tmp_path, source_zarr):
    """``(wb_path, arrays)`` — the same recording after the EXPORTER adds its ``wb_*`` arrays."""
    return _augment(source_zarr, str(tmp_path / "dual_wb.zarr.zip"))


def _chunk(h: int = 8, *, seed: int = 0, force_enabled: bool = False,
           gate_n: float = 0.0) -> tuple[hl_wholebody.WholeBodyChunk, np.ndarray]:
    """``(decoded chunk, the raw (H,43) prediction it came from)`` — both, so a test can assert
    the decode against its own input rather than against a second derivation."""
    rng = np.random.default_rng(seed)
    pred = rng.normal(scale=0.2, size=(h, LAY.action_dim))
    return hl_wholebody.decode_wholebody_chunk(
        pred, LAY, force_enabled=force_enabled, gate_n=gate_n), pred


def _planner(**kw):
    """A model-less :class:`WholeBodyPlanner` — the rings and the state assembly, no torch."""
    loaded = replay_only_model("wholebody", obs_history_len=kw.pop("To", 2))
    return hl_wholebody.WholeBodyPlanner(loaded, obs_hz=30.0, **kw)


# ---------------------------------------------------------------------------
# the joint order: the assumption every reuse in this file rests on
# ---------------------------------------------------------------------------
def test_the_policy_joint_order_is_the_sdk_order():
    """``WholeBodyWire`` inherits ``DenseWire``'s by-name remap, and it is only an IDENTITY
    because these two lists agree. ``hl_wholebody`` refuses to import otherwise; this states the
    fact the refusal guards, so a reordering fails here with a diff rather than at the robot."""
    assert tuple(WB_JOINT_NAMES) == tuple(G1_29DOF_SDK_DOF_NAMES)


def test_the_leg_set_is_exactly_the_twelve_lower_body_joints():
    """``--legs hold`` drops these 12 columns, so which 12 they are is a safety fact. Matched by
    substring in the module (so the set tracks the name list) and enumerated here."""
    names = [WB_JOINT_NAMES[i] for i in hl_wholebody.LEG_JOINT_INDICES]
    assert names == [
        "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint", "left_knee_joint",
        "left_ankle_pitch_joint", "left_ankle_roll_joint",
        "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
        "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    ]
    # And the complement is the 17 the held wire still drives: waist(3) + arms(2x7).
    rest = [n for i, n in enumerate(WB_JOINT_NAMES)
            if i not in set(hl_wholebody.LEG_JOINT_INDICES)]
    assert len(rest) == 17
    assert all(("waist" in n) or ("shoulder" in n) or ("elbow" in n) or ("wrist" in n)
               for n in rest), rest


def test_the_violet_is_the_same_violet_as_the_other_two_planners():
    """The offline page cannot import ``scripts/hl_serving.py`` (a CLI entry point that installs
    sys.path entries at module scope), so the ghost colour is duplicated. Pinned here instead:
    across this repo violet means "predicted", and it must be ONE violet."""
    import re
    from pathlib import Path

    from flore.s1_policy.inference.wholebody_viewer import PREDICTION_RGBA

    src = (Path(__file__).resolve().parents[1] / "scripts" / "hl_serving.py").read_text()
    match = re.search(r"^_GHOST_RGBA = \(([^)]*)\)", src, re.M)
    assert match, "scripts/hl_serving.py no longer defines _GHOST_RGBA at module scope"
    assert tuple(float(x) for x in match.group(1).split(",")) == tuple(PREDICTION_RGBA)


# ---------------------------------------------------------------------------
# decode: absolute, and refused for anything else
# ---------------------------------------------------------------------------
def test_the_decode_is_the_joints_verbatim_with_no_anchor():
    """The layout declares no pose dims, so nothing is un-relativized. An accidental transform
    here is undetectable downstream — a relative 29-vector is a plausible absolute one."""
    chunk, pred = _chunk(h=6, seed=1)
    assert chunk.joints.shape == (6, JOINT_DIM)
    assert np.array_equal(chunk.joints, pred[:, LAY.joints])
    assert chunk.n == 6


def test_the_gripper_comes_back_in_metres_and_force_in_physical_units():
    chunk, pred = _chunk(h=4, seed=2, force_enabled=True, gate_n=0.0)
    for role in ("left", "right"):
        slots = LAY.slots(role)
        assert np.allclose(chunk.gripper_m[role], pred[:, slots.gripper] * GRIPPER_WIDTH_REF_M)
        assert np.array_equal(chunk.wrench_pred[role], pred[:, slots.force])
        # gate_n=0 passes anything with |F| > 0, so the command IS the prediction here.
        assert np.array_equal(chunk.wrench_cmd[role], chunk.wrench_pred[role])


def test_a_below_gate_hand_is_published_as_an_exact_zero():
    """The gate must produce EXACTLY 0, not a small number: that is what collapses the low
    level's force integrator and x_tgt_shift to true position control."""
    chunk, _pred = _chunk(h=4, seed=3, force_enabled=True, gate_n=1e6)
    for role in ("left", "right"):
        assert np.all(chunk.wrench_cmd[role] == 0.0)
    # And the UNGATED prediction is still carried, for the viz/charts.
    assert any(np.any(chunk.wrench_pred[role] != 0.0) for role in ("left", "right"))


def test_the_master_switch_off_zeroes_both_hands():
    chunk, _pred = _chunk(h=4, seed=4, force_enabled=False, gate_n=0.0)
    for role in ("left", "right"):
        assert np.all(chunk.wrench_cmd[role] == 0.0)


@pytest.mark.parametrize("mode", ["left_only", "both_hands_head", "both_hands_head_dual"])
def test_a_keypoint_layout_is_refused_for_a_wholebody_decode_by_name(mode):
    lay = get_layout(mode)
    with pytest.raises(ValueError, match="no joint block"):
        hl_wholebody.decode_wholebody_chunk(
            np.zeros((4, lay.action_dim)), lay, force_enabled=False, gate_n=0.0)


def test_a_wrong_width_chunk_is_refused_rather_than_sliced():
    with pytest.raises(ValueError, match=r"must be \(H, 43\)"):
        hl_wholebody.decode_wholebody_chunk(
            np.zeros((4, 41)), LAY, force_enabled=False, gate_n=0.0)


# ---------------------------------------------------------------------------
# the wire
# ---------------------------------------------------------------------------
def test_step_zero_is_a_bit_exact_no_op_under_the_joint_clutch():
    """This is the ONE property that stands between the model's posture space and the robot's:
    the chunk is ~0.5 rad from the ready pose and the pre-arm gate is 0.20 rad, so step 0 has to
    land exactly on the base rather than approximately."""
    chunk, _ = _chunk(h=10, seed=5)
    clutch = hl_dense.JointClutch(READY, mode="joint")
    plan = hl_wholebody.WholeBodyWire(clutch, READY).plan(chunk)
    assert np.array_equal(plan["published"][0], clutch.base)
    # ...and the MOTION survives: only the offset changed. NOT bit-exact, and deliberately not
    # asserted as such — `(q[i] - q[0] + base) - (q[i-1] - q[0] + base)` differs from
    # `q[i] - q[i-1]` by float round-off. 1e-12 rad is 6e-11 degrees, i.e. eleven orders of
    # magnitude below the low level's 0.05 rad/tick slew, so the bound is tight enough to catch a
    # real rescale while not asserting IEEE associativity.
    assert np.allclose(np.diff(plan["published"], axis=0), np.diff(chunk.joints, axis=0),
                       atol=1e-12, rtol=0)


def test_clutch_none_publishes_the_absolute_posture_unchanged():
    """The escape hatch has to really be one — it is the mode an operator picks to MEASURE the
    posture gap, so it must not quietly still clutch."""
    chunk, _ = _chunk(h=5, seed=6)
    plan = hl_wholebody.WholeBodyWire(
        hl_dense.JointClutch(READY, mode="none"), READY).plan(chunk)
    assert np.array_equal(plan["published"], chunk.joints)


def test_legs_hold_pins_the_legs_and_still_drives_the_arms():
    chunk, _ = _chunk(h=12, seed=7)
    legs = np.asarray(hl_wholebody.LEG_JOINT_INDICES, dtype=int)
    arms = np.array([i for i in range(JOINT_DIM) if i not in set(legs.tolist())], dtype=int)
    plan = hl_wholebody.WholeBodyWire(
        hl_dense.JointClutch(READY, mode="joint"), READY, legs="hold").plan(chunk)
    held = plan["published"][:, legs]
    assert np.array_equal(held, np.broadcast_to(held[0], held.shape))
    # Held AT THE CLUTCH BASE, i.e. where the robot already is — not at the model's leg posture.
    assert np.array_equal(held[0], READY[legs])
    assert np.abs(np.diff(plan["published"][:, arms], axis=0)).max() > 0
    # And the arms carry the model's MOTION — holding the legs must not rescale anything. Same
    # float-round-off bound and same reasoning as the no-op test above.
    assert np.allclose(np.diff(plan["published"][:, arms], axis=0),
                       np.diff(chunk.joints[:, arms], axis=0), atol=1e-12, rtol=0)


def test_the_held_wire_says_so_in_the_gate_note_and_the_status_line():
    """A wire that drops 12 predicted columns must not look like one that does not."""
    chunk, _ = _chunk(h=4, seed=8)
    predict = hl_wholebody.WholeBodyWire(hl_dense.JointClutch(READY, mode="joint"), READY)
    hold = hl_wholebody.WholeBodyWire(hl_dense.JointClutch(READY, mode="joint"), READY,
                                      legs="hold")
    assert "LEGS HELD" in hold.gate_note and "LEGS HELD" not in predict.gate_note
    assert "legs HELD" in hold.status(hold.frame(hold.plan(chunk), 0))
    assert "legs HELD" not in predict.status(predict.frame(predict.plan(chunk), 0))


def test_legs_hold_is_refused_without_the_joint_clutch():
    """With clutch 'none' there is no base to hold AT, so the legs would be pinned to the
    MODEL's absolute leg posture — 0.5 rad from the robot's, and pinned there for the whole
    chunk. Refused by name rather than silently meaning something else."""
    with pytest.raises(ValueError, match="needs the joint clutch"):
        hl_wholebody.WholeBodyWire(hl_dense.JointClutch(READY, mode="none"), READY, legs="hold")


def test_an_unknown_legs_mode_is_refused():
    with pytest.raises(ValueError, match="legs must be one of"):
        hl_wholebody.WholeBodyWire(hl_dense.JointClutch(READY, mode="joint"), READY, legs="maybe")


def test_the_idle_frame_is_the_ready_pose_at_rest():
    """The FIRST heartbeat is what the low level's pre-arm gate closes against, and a leftover
    reference velocity would tell the tracker the robot is moving while it holds."""
    wire = hl_wholebody.WholeBodyWire(hl_dense.JointClutch(READY, mode="joint"), READY)
    pos, vel = wire.idle()
    assert np.array_equal(pos, READY)
    assert np.all(vel == 0.0)


def test_row_zero_velocity_is_zero_and_the_rest_is_a_backward_difference():
    chunk, _ = _chunk(h=6, seed=9)
    plan = hl_wholebody.WholeBodyWire(
        hl_dense.JointClutch(READY, mode="joint"), READY).plan(chunk)
    qvel = plan["qvel"]
    assert np.all(qvel[0] == 0.0)
    assert np.allclose(qvel[1:], np.diff(plan["published"], axis=0) * hl_dense.MC_RATE_HZ)


def test_a_non_finite_frame_never_reaches_the_socket():
    """``json.dumps`` serializes NaN unchecked and the low level validates only WIDTH, so
    ``publish`` is the only place a bad frame can be stopped."""
    wire = hl_wholebody.WholeBodyWire(hl_dense.JointClutch(READY, mode="joint"), READY)
    sent: list = []

    class _S0:
        def send_dense(self, *a, **k):
            sent.append((a, k))

    bad = np.array(READY, dtype=np.float64).copy()
    bad[3] = np.nan
    ok, reason = wire.publish(_S0(), (bad, np.zeros(JOINT_DIM)), None)
    assert not ok and "non-finite" in reason
    assert sent == []
    ok, _ = wire.publish(_S0(), wire.idle(), None)
    assert ok and len(sent) == 1


def test_the_measured_frame_is_remapped_by_name_and_published_at_rest():
    """The un-armed idle re-sync publishes the robot's own joints so the pre-arm gate can close.
    Velocity must be ZERO there: the robot is holding, not moving."""
    wire = hl_wholebody.WholeBodyWire(hl_dense.JointClutch(READY, mode="joint"), READY)
    measured = np.arange(JOINT_DIM, dtype=np.float64) * 0.02

    class _RF:
        dof_pos = measured

    pos, vel = wire.measured(_RF())
    assert np.array_equal(pos, measured)     # identity remap: policy order IS SDK order
    assert np.all(vel == 0.0)
    assert wire.measured(None) is None

    class _Bad:
        dof_pos = None

    assert wire.measured(_Bad()) is None


def test_the_leg_excursion_is_reported_with_the_offending_joint_named():
    """This number is what decides whether ``--legs hold`` is wanted, and 29 columns is enough
    that an index costs a lookup."""
    joints = np.zeros((5, JOINT_DIM))
    knee = WB_JOINT_NAMES.index("right_knee_joint")
    joints[:, knee] = np.array([0.0, 0.1, 0.2, 0.3, 0.4])
    joints[:, WB_JOINT_NAMES.index("left_elbow_joint")] = 5.0   # an ARM, must not be reported
    rad, name = hl_wholebody.leg_excursion_rad(joints)
    assert name == "right_knee_joint"
    assert rad == pytest.approx(0.4)
    # Degenerate input reports nothing rather than raising — it is a log line, not a gate.
    assert hl_wholebody.leg_excursion_rad(np.zeros((0, JOINT_DIM))) == (0.0, "")


# ---------------------------------------------------------------------------
# the proprio seed
# ---------------------------------------------------------------------------
def test_the_seed_is_the_mean_of_the_training_data_episode_starts(wb_zarr):
    wb, arrays = wb_zarr
    seed, note = hl_wholebody.state_seed_from_zarr(wb)
    assert seed.shape == (JOINT_DIM,)
    assert "episode-start" in note and "cond_mode='home'" in note
    # Derived from the file, not from a constant: recompute it independently.
    import zarr

    with zarr.ZipStore(wb, mode="r") as store:
        root = zarr.group(store=store)
        ends = np.asarray(root["meta"]["episode_ends"][:], dtype=np.int64)
    starts = np.concatenate([[0], ends[:-1]]).astype(int)
    rows = arrays["wb_state_joints"][starts]
    ok = arrays["wb_state_valid"][starts].reshape(starts.size, -1)[:, 0] > 0.5
    assert np.allclose(seed, rows[ok].mean(axis=0))


def test_no_zarr_falls_back_to_the_home_anchor_and_says_why():
    seed, note = hl_wholebody.state_seed_from_zarr(None)
    assert np.array_equal(seed, np.zeros(JOINT_DIM))
    assert "cond_mode=home anchor" in note


def test_a_dual_zarr_with_no_wb_arrays_falls_back_by_name(source_zarr):
    seed, note = hl_wholebody.state_seed_from_zarr(source_zarr)
    assert np.array_equal(seed, np.zeros(JOINT_DIM))
    assert "wb_windows" in note


def test_a_seed_read_never_raises(tmp_path):
    """A seed problem must not stop a bring-up — it must be PRINTED. So every failure path
    returns the fallback with a reason instead of propagating."""
    missing = tmp_path / "nope.zarr.zip"
    seed, note = hl_wholebody.state_seed_from_zarr(str(missing))
    assert np.array_equal(seed, np.zeros(JOINT_DIM))
    assert "could not read" in note
    garbage = tmp_path / "junk.zarr.zip"
    garbage.write_bytes(b"not a zip")
    seed, note = hl_wholebody.state_seed_from_zarr(str(garbage))
    assert np.array_equal(seed, np.zeros(JOINT_DIM))
    assert "could not read" in note


# ---------------------------------------------------------------------------
# the planner
# ---------------------------------------------------------------------------
def test_the_planner_refuses_a_keypoint_checkpoint_by_name():
    loaded = replay_only_model("both_hands_head_dual")
    with pytest.raises(ValueError, match="needs a wholebody layout"):
        hl_wholebody.WholeBodyPlanner(loaded)


def test_an_unknown_state_source_is_refused():
    loaded = replay_only_model("wholebody")
    with pytest.raises(ValueError, match="state_source must be one of"):
        hl_wholebody.WholeBodyPlanner(loaded, state_source="guess")


def test_the_state_is_joints_then_the_two_grippers():
    """The layout says ``[joints 29][L grip][R grip]``; the planner must fill exactly that, and
    the base class's ``wrt_start_rot6d`` key must not be reachable (it is absent by design)."""
    planner = _planner()
    obs, _ap, _ar = planner.snapshot_obs()
    assert obs.state.shape == (2, 31)
    assert np.array_equal(obs.state[:, 0:JOINT_DIM],
                          np.zeros((2, JOINT_DIM)))    # the home-anchor seed
    slots = LAY.state_slots()
    assert slots["left"] == {"gripper": 29} and slots["right"] == {"gripper": 30}
    assert "wrt_start_rot6d" not in slots["left"]


def test_the_seed_is_flagged_unmeasured_until_something_is_committed():
    """The seed is a real pose but not a MEASUREMENT of this robot, so the model's own no-data
    embedding must handle it rather than being told the seed is where the body is."""
    planner = _planner(seed_joints=READY)
    obs, _, _ = planner.snapshot_obs()
    assert np.array_equal(obs.state[-1, LAY.joint_state_slot], READY)
    assert not obs.state_valid[:, LAY.joint_state_slot].any()

    planner.set_joint_target(READY + 0.5)
    planner.push_proprio_joints()
    obs, _, _ = planner.snapshot_obs()
    assert np.allclose(obs.state[-1, LAY.joint_state_slot], READY + 0.5)
    assert obs.state_valid[:, LAY.joint_state_slot].all()


def test_push_proprio_joints_also_advances_the_gripper_ring():
    """REGRESSION. It once advanced only the joint ring, so ``_buf_grip`` stayed at its
    ``__init__`` zeros forever: both jaw columns read 0.0 m — the CLOSED end of a legal travel —
    for the whole session while ``push_gripper`` updated a latch nothing ever read. Nothing
    downstream could notice, because a shut jaw is not an error."""
    planner = _planner()
    planner.push_gripper("left", 0.09)
    planner.push_gripper("right", 0.05)
    for _ in range(3):
        planner.push_proprio_joints()
    obs, _, _ = planner.snapshot_obs()
    assert obs.state[-1, 29] == pytest.approx(0.09 / GRIPPER_WIDTH_REF_M)
    assert obs.state[-1, 30] == pytest.approx(0.05 / GRIPPER_WIDTH_REF_M)
    assert obs.state_valid[:, 29].all() and obs.state_valid[:, 30].all()


def test_an_unmeasured_gripper_is_flagged_rather_than_fed_as_a_closed_jaw():
    """0.0 m is a LEGAL width, which is exactly why its absence cannot be inferred from the
    value. Same rule the training converter applies to a dead encoder."""
    planner = _planner()
    planner.push_proprio_joints()
    obs, _, _ = planner.snapshot_obs()
    assert not obs.state_valid[:, 29].any() and not obs.state_valid[:, 30].any()


def test_measured_joints_are_ignored_in_command_mode_and_used_in_measured_mode():
    """The ingest loop calls ``push_measured_joints`` unconditionally so the decision lives in
    exactly one place. That only works if the planner really ignores it."""
    cmd = _planner(state_source="command", seed_joints=READY)
    cmd.push_measured_joints(READY + 1.0)
    cmd.push_proprio_joints()
    obs, _, _ = cmd.snapshot_obs()
    assert np.array_equal(obs.state[-1, LAY.joint_state_slot], READY)

    meas = _planner(state_source="measured", seed_joints=READY)
    meas.push_measured_joints(READY + 1.0)
    meas.push_proprio_joints()
    obs, _, _ = meas.snapshot_obs()
    assert np.allclose(obs.state[-1, LAY.joint_state_slot], READY + 1.0)


def test_a_non_finite_or_wrong_width_commit_is_dropped_not_latched():
    planner = _planner(seed_joints=READY)
    bad = READY.copy()
    bad[2] = np.nan
    planner.set_joint_target(bad)
    planner.set_joint_target(np.zeros(5))
    planner.push_proprio_joints()
    obs, _, _ = planner.snapshot_obs()
    assert np.array_equal(obs.state[-1, LAY.joint_state_slot], READY)
    assert not obs.state_valid[:, LAY.joint_state_slot].any()   # still never written


def test_reset_restores_the_seed_and_re_flags_it_unmeasured():
    planner = _planner(seed_joints=READY)
    planner.set_joint_target(READY + 0.4)
    planner.push_proprio_joints()
    planner.reset_joints()
    obs, _, _ = planner.snapshot_obs()
    assert np.array_equal(obs.state[-1, LAY.joint_state_slot], READY)
    assert not obs.state_valid[:, LAY.joint_state_slot].any()


def test_infer_chunk_refuses_a_planner_with_no_policy():
    planner = _planner()
    with pytest.raises(RuntimeError, match="no policy"):
        planner.infer_chunk(seed=0, force_enabled=False, gate_n=1.5)


def test_the_joint_readout_reports_never_written_as_none():
    planner = _planner(seed_joints=READY)
    q, age = planner.joint_state()
    assert np.array_equal(q, READY) and age is None
    planner.set_joint_target(READY + 0.1)
    q, age = planner.joint_state()
    assert np.allclose(q, READY + 0.1) and age is not None and age >= 0.0


# ---------------------------------------------------------------------------
# the offline result + its metrics
# ---------------------------------------------------------------------------
def _fake_sample(h: int, *, mask=None) -> dict:
    rng = np.random.default_rng(11)
    target = rng.normal(scale=0.3, size=(h, LAY.action_dim))
    return {
        "action_chunk": torch.from_numpy(target),
        "action_mask": torch.from_numpy(
            np.ones((h, LAY.action_dim), bool) if mask is None else mask),
        "state": torch.from_numpy(rng.normal(size=(LAY.state_dim,))),      # To == 1: bare (D,)
        "state_valid": torch.from_numpy(np.ones(LAY.state_dim, bool)),
        "force_history": torch.from_numpy(np.zeros((1, LAY.force_hist_channels))),
        "force_history_valid": torch.tensor([True, True]),
        "image": torch.from_numpy(np.zeros((2, 3, 4, 4), dtype=np.float32)),
    }


class _FakeSource:
    layout = LAY
    image_mean = 0.5
    image_std = 0.5


def _rollout(h: int = 12, *, mask=None):
    from flore.s1_policy.inference.common import SampleRef
    from flore.s1_policy.inference.wholebody import build_wholebody_result

    sample = _fake_sample(h, mask=mask)
    pred = sample["action_chunk"].numpy() + 0.05
    ref = SampleRef(dataset_index=0, file_index=0, source_path="x", global_episode_id=0,
                    file_episode_id=0, file_frame=0, episode_frame=0, split="train")
    return build_wholebody_result(algorithm="diffusion", num_steps=2, seed=0, ref=ref,
                                 sample=sample, predicted_actions=pred,
                                 source=_FakeSource()), sample, pred


def test_the_offline_result_never_reports_a_position_it_cannot_compute():
    """``mask[:, []].all(-1)`` is True everywhere and a norm over 0 columns is exactly 0.0, so
    the keypoint arithmetic logs a PERFECT 0.00 cm error for a layout with no position columns.
    Omitting the key means a consumer that reads it gets a KeyError naming what it asked for."""
    result, _s, _p = _rollout()
    assert set(result.metrics) == {"joint_deg", "gripper_mm", "force_n", "torque_nm"}
    assert "position_cm" not in result.metrics
    assert "rotation_geodesic_deg" not in result.metrics
    assert result.metrics["joint_deg"] > 0.0


def test_the_offline_decode_agrees_with_the_serving_decode_bit_for_bit():
    """TWO decoders of one layout is one place they can disagree. The joints, the grippers and
    the wrenches must come out identical, or an offline page would be validating a different
    chunk from the one the robot is handed."""
    result, _s, pred = _rollout(h=9)
    chunk = hl_wholebody.decode_wholebody_chunk(pred, LAY, force_enabled=True, gate_n=0.0)
    assert np.array_equal(result.predicted_joints, chunk.joints)
    for role in ("left", "right"):
        assert np.array_equal(result.predicted_gripper_m[role], chunk.gripper_m[role])
        assert np.array_equal(result.predicted_wrench[role], chunk.wrench_pred[role])


def test_the_state_is_normalized_to_two_dimensions():
    """At ``To == 1`` HLDataset emits a bare ``(31,)``. Un-normalized, ``state[-1]`` is the last
    COLUMN rather than the newest FRAME — which is every sample on the current v3 grid."""
    result, _s, _p = _rollout()
    assert result.state.shape == (1, LAY.state_dim)
    assert result.state_valid.shape == (1, LAY.state_dim)


def test_masked_steps_are_nan_in_the_error_series_not_zero():
    """0 is a legitimate error, so an unsupervised tail must not draw as a perfect one."""
    h = 10
    mask = np.ones((h, LAY.action_dim), bool)
    mask[6:] = False
    result, _s, _p = _rollout(h=h, mask=mask)
    err = result.joint_error_deg()
    assert np.isfinite(err[:6]).all()
    assert np.isnan(err[6:]).all()
    assert result.joint_valid[:6].all() and not result.joint_valid[6:].any()


def test_the_summary_names_the_worst_joint():
    from flore.s1_policy.inference.wholebody import wholebody_summary

    result, _s, _p = _rollout()
    summary = wholebody_summary(result)
    assert summary["layout"] == "wholebody"
    assert summary["worst_joint"] in WB_JOINT_NAMES
    assert summary["shape"] == [12, LAY.action_dim]
    assert summary["metrics"]["joint_deg"] is not None
    assert "position_cm" not in summary["metrics"]


@pytest.mark.parametrize("mode", ["left_only", "both_hands_head_dual"])
def test_a_keypoint_layout_is_refused_by_the_offline_builder(mode):
    from flore.s1_policy.inference.common import SampleRef
    from flore.s1_policy.inference.wholebody import build_wholebody_result

    lay = get_layout(mode)

    class _Src:
        layout = lay
        image_mean = 0.5
        image_std = 0.5

    ref = SampleRef(dataset_index=0, file_index=0, source_path="x", global_episode_id=0,
                    file_episode_id=0, file_frame=0, episode_frame=0, split="train")
    sample = {
        "action_chunk": torch.zeros((4, lay.action_dim), dtype=torch.float64),
        "action_mask": torch.ones((4, lay.action_dim), dtype=torch.bool),
        "state": torch.zeros(lay.state_dim, dtype=torch.float64),
        "state_valid": torch.ones(lay.state_dim, dtype=torch.bool),
        "force_history": torch.zeros((1, max(lay.force_hist_channels, 1))),
        "force_history_valid": torch.tensor([True]),
        "image": torch.zeros((1, 3, 4, 4)),
    }
    with pytest.raises(ValueError, match="needs a wholebody layout"):
        build_wholebody_result(algorithm="diffusion", num_steps=1, seed=0, ref=ref,
                               sample=sample,
                               predicted_actions=np.zeros((4, lay.action_dim)), source=_Src())


def test_the_keypoint_builder_refuses_a_wholebody_layout_by_name():
    """The other direction, and it is the one that used to CRASH: ``build_rollout_result`` would
    reshape a ``(H, 0)`` rot6d slice to ``(-1, 2, 3)`` and name a rotation array on a layout that
    has no rotations, which points at the wrong thing entirely."""
    from flore.s1_policy.inference.common import SampleRef, build_rollout_result

    ref = SampleRef(dataset_index=0, file_index=0, source_path="x", global_episode_id=0,
                    file_episode_id=0, file_frame=0, episode_frame=0, split="train")
    with pytest.raises(ValueError, match="infer_s1_wholebody"):
        build_rollout_result(algorithm="diffusion", num_steps=1, seed=0, ref=ref,
                             sample=_fake_sample(4),
                             predicted_actions=np.zeros((4, LAY.action_dim)),
                             source=_FakeSource())


# ---------------------------------------------------------------------------
# the viewer's one piece of load-bearing display logic
# ---------------------------------------------------------------------------
def test_an_unsupervised_target_step_holds_instead_of_snapping_to_zeros():
    """An unsupervised step's stored value is the zero-fill — 29 zeros, i.e. the G1 standing bolt
    upright. Drawn, the pale body SNAPS straight at the first masked step and reads as a
    demonstration that did something violent rather than one that ended."""
    from flore.s1_policy.inference.wholebody_viewer import _hold_last_valid

    joints = np.zeros((6, JOINT_DIM))
    joints[:3] = 0.3
    valid = np.ones((6, JOINT_DIM), bool)
    valid[3:] = False
    held = _hold_last_valid(joints, valid)
    assert np.all(held[3:] == 0.3)
    assert np.array_equal(held[:3], joints[:3])
    # All-masked is left exactly as-is rather than guessing a pose.
    allbad = _hold_last_valid(joints, np.zeros((6, JOINT_DIM), bool))
    assert np.array_equal(allbad, joints)


# ---------------------------------------------------------------------------
# wb_joint_target: the one distinction the wire CANNOT see, and the two that matter
# ---------------------------------------------------------------------------
def test_the_joint_clutch_publishes_the_same_wire_for_both_targets():
    """``base + (q[i] - q[0])`` is invariant to a constant offset, so a delta chunk and the
    absolute chunk it came from publish IDENTICAL joint targets.

    This is the whole reason ``wb_joint_target`` is a training A/B rather than a second deploy
    path: nothing between ``decode_wholebody_chunk`` and ``:8770`` has to branch on it. If this
    ever stops holding, the delta preset has silently become a different wire.
    """
    chunk, _pred = _chunk(h=6, seed=3)
    absolute = np.asarray(chunk.joints, dtype=np.float64)
    delta = absolute - absolute[0]
    a_wire = hl_wholebody.WholeBodyWire(hl_dense.JointClutch(READY, mode="joint"), READY)
    d_wire = hl_wholebody.WholeBodyWire(hl_dense.JointClutch(READY, mode="joint"), READY,
                                        joint_target="delta")
    a_plan = a_wire.plan(chunk)
    d_plan = d_wire.plan(hl_wholebody.WholeBodyChunk(
        joints=delta, gripper_m=chunk.gripper_m,
        wrench_pred=chunk.wrench_pred, wrench_cmd=chunk.wrench_cmd))
    # `frame()` is (dof_pos, qvel) -- BOTH must agree, since the velocities are a finite
    # difference of the positions and a constant offset must cancel there too.
    for i in range(chunk.n):
        a_pos, a_vel = a_wire.frame(a_plan, i)
        d_pos, d_vel = d_wire.frame(d_plan, i)
        np.testing.assert_allclose(a_pos, d_pos, rtol=0, atol=1e-12)
        np.testing.assert_allclose(a_vel, d_vel, rtol=0, atol=1e-12)
    # And step 0 lands exactly on the clutch base in BOTH -- the no-op first executed step.
    np.testing.assert_allclose(d_wire.frame(d_plan, 0)[0], READY, rtol=0, atol=1e-12)
    np.testing.assert_allclose(a_wire.frame(a_plan, 0)[0], READY, rtol=0, atol=1e-12)


def test_the_clutchless_wire_refuses_a_delta_checkpoint():
    """``--wb-clutch none`` publishes the chunk verbatim, and a delta chunk's row 0 is ~0 rad --
    i.e. the 29-joints-zero pose, commanded to a standing robot at the instant it arms. Every
    width and finiteness check the wire runs would pass, so it has to be a refusal by name."""
    with pytest.raises(ValueError, match=r"--wb-clutch none cannot publish"):
        hl_wholebody.WholeBodyWire(hl_dense.JointClutch(READY, mode="none"), READY,
                                   joint_target="delta")
    # The same pair with the default target is still allowed (that combination is only loud).
    hl_wholebody.WholeBodyWire(hl_dense.JointClutch(READY, mode="none"), READY)


@pytest.mark.parametrize("bad", ["Delta", "relative", ""])
def test_an_unknown_joint_target_is_refused_by_the_wire_and_the_planner(bad):
    with pytest.raises(ValueError, match=r"joint_target must be one of"):
        hl_wholebody.WholeBodyWire(hl_dense.JointClutch(READY, mode="joint"), READY,
                                   joint_target=bad)
    with pytest.raises(ValueError, match=r"joint_target must be one of"):
        _planner(joint_target=bad)


def test_the_proprio_latch_integrates_a_delta_row_and_replaces_an_absolute_one():
    """``commit_chunk_row`` is the ONE place the target reaches the observation.

    ``absolute``: the row IS the posture, so the latch becomes it -- byte-identical to the shipped
    ``set_joint_target`` behaviour. ``delta``: the row is motion relative to the observation the
    chunk was conditioned on, so the posture it implies is ``latch + row`` and the latch
    integrates. Getting this backwards is invisible in every shape and would feed the model a
    state 0.5 rad wrong from the second chunk onward.
    """
    seed = np.full(JOINT_DIM, 0.3)
    row = np.full(JOINT_DIM, 0.05)

    absolute = _planner(To=1, seed_joints=seed)
    absolute.commit_chunk_row(row)
    absolute.push_proprio_joints()
    obs, _ap, _ar = absolute.snapshot_obs()
    np.testing.assert_allclose(obs.state[-1, :JOINT_DIM], row, rtol=0, atol=1e-12)

    delta = _planner(To=1, seed_joints=seed, joint_target="delta")
    delta.commit_chunk_row(row)
    delta.push_proprio_joints()
    obs_d, _ap, _ar = delta.snapshot_obs()
    np.testing.assert_allclose(obs_d.state[-1, :JOINT_DIM], seed + row, rtol=0, atol=1e-12)
    # It ACCUMULATES: a second chunk advances from the new latch, not from the seed.
    delta.commit_chunk_row(row)
    delta.push_proprio_joints()
    obs_d2, _ap, _ar = delta.snapshot_obs()
    np.testing.assert_allclose(obs_d2.state[-1, :JOINT_DIM], seed + 2 * row, rtol=0, atol=1e-12)


def test_a_measured_readback_is_never_integrated():
    """``push_measured_joints`` goes through ``set_joint_target``, NOT ``commit_chunk_row``:
    encoder angles are an absolute posture whatever the target is, and an integrator there would
    double the robot's own pose on the first frame of --state-source measured."""
    seed = np.full(JOINT_DIM, 0.3)
    q = np.full(JOINT_DIM, -0.2)
    p = _planner(To=1, seed_joints=seed, state_source="measured", joint_target="delta")
    p.push_measured_joints(q)
    p.push_proprio_joints()
    obs, _ap, _ar = p.snapshot_obs()
    np.testing.assert_allclose(obs.state[-1, :JOINT_DIM], q, rtol=0, atol=1e-12)


def test_a_non_finite_or_wrong_width_row_is_dropped_not_latched():
    """Same discipline as ``set_joint_target``: a bad row must leave the latch alone rather than
    poison every subsequent observation (in delta mode a single NaN would be permanent)."""
    seed = np.full(JOINT_DIM, 0.3)
    p = _planner(To=1, seed_joints=seed, joint_target="delta")
    p.commit_chunk_row(np.full(JOINT_DIM, np.nan))
    p.commit_chunk_row(np.zeros(JOINT_DIM - 1))
    p.push_proprio_joints()
    obs, _ap, _ar = p.snapshot_obs()
    np.testing.assert_allclose(obs.state[-1, :JOINT_DIM], seed, rtol=0, atol=1e-12)


def test_the_loaded_checkpoint_defaults_the_target_to_absolute():
    """Every wholebody checkpoint written before this knob existed carries no ``wb_joint_target``,
    and every keypoint one has no joint block at all -- both must read ``absolute``."""
    assert replay_only_model("wholebody").wb_joint_target == "absolute"
    assert replay_only_model("both_hands_head_dual").wb_joint_target == "absolute"
    assert _planner().joint_target == "absolute"


def test_the_offline_page_draws_postures_for_a_delta_checkpoint():
    """``build_wholebody_result`` un-relativizes a ``delta`` checkpoint ONCE, so every field of
    ``WholeBodyRollout`` is an absolute posture -- and every reported metric is bit-identical.

    Both matter and they are separate claims. The DRAWING needs postures: this page's whole content
    is two G1 bodies (a layout with no position columns has no trajectory to fall back on), and a
    raw delta chunk would put both of them at the 29-joints-zero pose plus a few centidegrees. The
    METRICS must not move, because a shared offset cancels in every difference -- if they did, the
    two targets could not be compared and the A/B would be meaningless.
    """
    from flore.s1_policy.inference.common import SampleRef
    from flore.s1_policy.inference.wholebody import build_wholebody_result

    class _DeltaSource(_FakeSource):
        wb_joint_target = "delta"

    h = 10
    sample = _fake_sample(h)
    anchor = np.asarray(sample["state"].numpy(), dtype=np.float64).reshape(
        -1, LAY.state_dim)[-1, LAY.joint_state_slot]
    # A DELTA sample: the joint columns of both the target and the prediction are motion.
    delta_sample = dict(sample)
    tgt = sample["action_chunk"].numpy().astype(np.float64).copy()
    tgt[:, LAY.joints] -= anchor
    delta_sample["action_chunk"] = torch.from_numpy(tgt)
    pred = tgt + 0.05
    ref = SampleRef(dataset_index=0, file_index=0, source_path="x", global_episode_id=0,
                    file_episode_id=0, file_frame=0, episode_frame=0, split="train")
    kw = dict(algorithm="diffusion", num_steps=2, seed=0, ref=ref, predicted_actions=pred)
    delta = build_wholebody_result(sample=delta_sample, source=_DeltaSource(), **kw)
    # The drawn bodies are POSTURES: the target's joints are back where the absolute sample had
    # them, and `predicted_actions` / `target_actions` carry the same un-relativized block.
    np.testing.assert_allclose(delta.target_joints,
                               sample["action_chunk"].numpy()[:, LAY.joints],
                               rtol=0, atol=1e-9)
    np.testing.assert_allclose(delta.target_actions[:, LAY.joints], delta.target_joints,
                               rtol=0, atol=0)
    # ... and identical to what the ABSOLUTE builder reports for the same underlying motion.
    absolute = build_wholebody_result(
        sample=sample, source=_FakeSource(),
        **{**kw, "predicted_actions": sample["action_chunk"].numpy().astype(np.float64) + 0.05})
    assert absolute.metrics.keys() == delta.metrics.keys()
    for key in absolute.metrics:
        np.testing.assert_allclose(delta.metrics[key], absolute.metrics[key], rtol=0, atol=1e-9)
    assert "position_cm" not in delta.metrics
    # A DEFAULT source (no attribute at all = every pre-knob checkpoint) must not be transformed.
    plain = build_wholebody_result(
        sample=sample, source=_FakeSource(),
        **{**kw, "predicted_actions": sample["action_chunk"].numpy().astype(np.float64) + 0.05})
    np.testing.assert_array_equal(plain.target_joints,
                                  sample["action_chunk"].numpy()[:, LAY.joints])
