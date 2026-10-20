"""The ``diffusion_umi_5pt`` DEPLOY path and its display IK, on a CPU with no robot.

Two halves, and the split is deliberate:

* **No URDF needed** — the wire, the clutch, the observation assembly and the refusals. Pure
  numpy over :mod:`flore.s1_policy.serving.hl_5pt`, so every safety claim in that module's
  docstring is checkable in CI.
* **URDF needed** (skipped without the ``UMI_collection`` submodule / ``yourdfpy``) — the anchor,
  the readout's own numbers, and :mod:`flore.s1_policy.inference.keypoint_ik`. The IK's round trip
  through the SAME FK the dataset and the wire are built on is the one test that can catch a lever,
  a chain or a root-placement mistake, so it is worth the skip.

``scripts/smoke_test_hl_5pt.py`` is the other half of the verification: the real checkpoint, the
real chunk, and the questions only a trained model can answer.
"""

from __future__ import annotations

import os
from types import SimpleNamespace

import numpy as np
import pytest

from flore.s1_policy.data.layout import get_layout
from flore.s1_policy.serving import hl_5pt, hl_dense
from flore.s1_policy.serving.hl_online import decode_chunk, replay_only_model
from flore.utils.rot6d import rotmat_to_rot6d

_URDF = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "third_party", "UMI_collection", "robot_assets", "g1_yam", "g1_29dof_yam.urdf")
_ANCHOR_JSON = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "third_party", "holosoma_extensions", "src", "extensions", "wbt_wrappers",
    "wbt_realtime_eval", "wbt_realtime_eval", "vive", "home_anchor.json")

needs_rig = pytest.mark.skipif(
    not (os.path.isfile(_URDF) and os.path.isfile(_ANCHOR_JSON)),
    reason="yam URDF / home_anchor.json not present (submodules not init'd)")

MODE = "fivepoint_geom_xy3_rpy"
LAY = get_layout(MODE)
N = len(hl_dense.FIVE_POINT_BODIES)

#: A ready frame with a DISTINCT value per point and per axis, so a slot permutation shows up as a
#: value mismatch rather than a shape error. Quaternions are identity (w last, xyzw).
READY_POS = (np.arange(3 * N, dtype=np.float64) * 0.01 + 0.1)
READY_ORN = np.tile(np.array([0.0, 0.0, 0.0, 1.0]), N)


def _wire(*, post: str = "command", carry: str = "all", measured_frame=None):
    clutch = hl_5pt.build_clutch(READY_POS, READY_ORN,
                                 post=hl_dense.PostAnchorConfig(mode=post), carry=carry)
    return hl_5pt.FivePointDirectWire(hl_dense.KeypointReadout(np.zeros(3)), READY_POS, READY_ORN,
                                      clutch=clutch, measured_frame=measured_frame)


def _chunk(h: int = 8, *, seed: int = 0, layout=LAY):
    """A decoded five-role chunk from a random prediction, against an identity-ish anchor."""
    rng = np.random.default_rng(seed)
    pred = rng.normal(scale=0.05, size=(h, layout.action_dim))
    n = len(layout.roles)
    anchor_pos = np.arange(3 * n, dtype=np.float64).reshape(n, 3) * 0.1
    anchor_rot6d = np.tile(rotmat_to_rot6d(np.eye(3)), (n, 1))
    return decode_chunk(pred, layout, anchor_pos, anchor_rot6d,
                        force_enabled=False, gate_n=0.0)


def _planner(**kw):
    """A model-less :class:`FivePointPlanner` — the rings and the state assembly, no torch."""
    loaded = replay_only_model(MODE, obs_history_len=kw.pop("To", 2))
    return hl_5pt.FivePointPlanner(loaded, obs_hz=30.0, **kw)


# ---------------------------------------------------------------------------
# the slot order: the assumption every readout in this path rests on
# ---------------------------------------------------------------------------
def test_the_wire_roles_are_the_layout_roles_in_the_wires_own_order():
    """``WIRE_ROLES`` is derived from ``FIVE_POINT_BODIES`` through ``ready_pose.ROLE_BODY``. If it
    ever disagreed with ``layout.roles``, the gate's per-point labels and the published vector
    would describe different points — and both would still be the right SHAPE."""
    assert hl_5pt.WIRE_ROLES == tuple(LAY.roles)
    assert len(hl_5pt.WIRE_ROLES) == N


def test_the_geometry_reads_only_the_three_tracked_bodies():
    """``rel_xy3`` and ``rpy`` read the two hands and the chest and NOTHING else. That is what
    makes a five-point observation buildable at deploy: a foot pose is never needed for it, so
    ``obs_mode=last_target`` on the three tracked points is the whole state."""
    assert set(hl_5pt.GEOM_BODY_ROLE.values()) == {"left", "right", "head"}


def test_all_from_decoded_uses_the_wires_slot_order_not_the_layouts():
    """The readout must fill slot k with the role that owns ``FIVE_POINT_BODIES[k]``. Checked by
    giving each role a distinguishable position and reading the slots back."""
    readout = hl_dense.KeypointReadout(np.zeros(3))
    rows = {role: {"position": np.full((3, 3), float(i + 1)),
                   "rot6d": np.tile(rotmat_to_rot6d(np.eye(3)), (3, 1))}
            for i, role in enumerate(hl_5pt.WIRE_ROLES)}
    pos, orn = readout.all_from_decoded(rows)
    assert pos.shape == (3, 3 * N) and orn.shape == (3, 4 * N)
    for k in range(N):
        assert np.allclose(pos[0, 3 * k:3 * k + 3], float(k + 1))


def test_a_missing_role_is_refused_by_name_not_zero_filled():
    readout = hl_dense.KeypointReadout(np.zeros(3))
    rows = {role: {"position": np.zeros((2, 3)),
                   "rot6d": np.tile(rotmat_to_rot6d(np.eye(3)), (2, 1))}
            for role in ("left", "right", "head")}
    with pytest.raises(ValueError, match="left_foot|right_foot"):
        readout.all_from_decoded(rows)


# ---------------------------------------------------------------------------
# the wire + the clutch
# ---------------------------------------------------------------------------
def test_step_zero_lands_on_the_ready_frame_under_the_clutch():
    """THE headline claim of this planner: with the clutch on, the wire's step 0 IS the low level's
    own pre-arm gate reference. Measured on the shipped checkpoint the model's absolute step 0 sits
    59-63 mm away against a 70 mm tolerance, so this is what closes the gate by construction."""
    wire = _wire(post="command", carry="all")
    wire.clutch.sync(None)                       # nothing executed yet -> base = the READY frame
    plan = wire.plan(_chunk())
    # 1e-12 m = 1 picometre, not `array_equal`: `apply` is `p[i] + (base - p[0])`, so step 0 is
    # `p[0] + (base - p[0])` — algebraically the base and, in float64, the base to a few ULPs.
    # Asserting bit-equality would fail on that and teach nothing; the smoke test reports the same
    # quantity as "worst 0.000 um".
    assert np.abs(plan["pos"][0] - READY_POS).max() < 1e-12
    assert np.abs(plan["orn"][0] - READY_ORN).max() < 1e-12


def test_carry_upper_leaves_the_two_feet_absolute():
    """``--carry upper`` is the completion path's behaviour, available here. The three GATED points
    land on the base; the feet keep whatever the chunk said."""
    chunk = _chunk()
    readout = hl_dense.KeypointReadout(np.zeros(3))
    rows = {rc.role: {"position": rc.position, "rot6d": rc.rot6d} for rc in chunk.roles}
    raw_pos, _ = readout.all_from_decoded(rows)
    wire = _wire(post="command", carry="upper")
    wire.clutch.sync(None)
    plan = wire.plan(chunk)
    k = hl_dense.FIVE_POINT_GATED
    assert np.abs(plan["pos"][0, :3 * k] - READY_POS[:3 * k]).max() < 1e-12
    assert np.array_equal(plan["pos"][:, 3 * k:], raw_pos[:, 3 * k:])   # untouched: a real copy


def test_post_anchor_off_publishes_the_absolute_readout_unchanged():
    """``off`` must be the identity, so "the model's own claim" stays available for a measurement."""
    chunk = _chunk()
    rows = {rc.role: {"position": rc.position, "rot6d": rc.rot6d} for rc in chunk.roles}
    raw_pos, raw_orn = hl_dense.KeypointReadout(np.zeros(3)).all_from_decoded(rows)
    wire = _wire(post="off")
    plan = wire.plan(chunk)
    assert np.allclose(plan["pos"], raw_pos) and np.allclose(plan["orn"], raw_orn)


def test_the_clutch_offset_is_a_constant_over_the_whole_chunk():
    """``p_base + (p[i] - p[0])`` is a RIGID shift, so every per-step difference survives it. If it
    did not, the clutch would be changing the motion and not only where it starts."""
    chunk = _chunk(h=12, seed=3)
    rows = {rc.role: {"position": rc.position, "rot6d": rc.rot6d} for rc in chunk.roles}
    raw_pos, _ = hl_dense.KeypointReadout(np.zeros(3)).all_from_decoded(rows)
    wire = _wire(post="command", carry="all")
    wire.clutch.sync(None)
    got = wire.plan(chunk)["pos"]
    assert np.allclose(np.diff(got, axis=0), np.diff(raw_pos, axis=0))


def test_an_unknown_carry_mode_is_refused():
    with pytest.raises(ValueError, match="carry must be one of"):
        hl_5pt.build_clutch(READY_POS, READY_ORN, post=hl_dense.PostAnchorConfig(), carry="feet")


def test_the_status_line_never_claims_a_completion():
    """The inherited ``FivePointWire.status`` ends in ``upper=... feet=completion``, which is true
    of the parent's sources and a LIE about this wire — there is no completion in the chain."""
    wire = _wire()
    line = wire.status(wire.ready_frame)
    assert "upper=" not in line and "direct (no completion)" in line
    assert f"carrying {N}/{N}" in line


def test_the_gate_note_names_the_policy_not_the_completion():
    assert "POLICY" in _wire().gate_note
    assert "completion" not in _wire().gate_note


def test_a_non_finite_frame_never_reaches_the_socket():
    """The publish gate is inherited, so this states the property rather than re-testing the gate:
    holosoma validates only WIDTH and ``json.dumps`` serializes NaN unchecked."""
    wire = _wire(post="off")
    sent: list = []
    driver = type("S0", (), {"send_5pt": lambda _s, p, o, f_cmd=None: sent.append((p, o))})()
    bad = READY_POS.copy()
    bad[4] = np.nan
    ok, reason = wire.publish(driver, (bad, READY_ORN), None)
    assert not ok and not sent and reason


def test_the_idle_frame_is_the_ready_frame_and_never_chases_measured():
    """A keypoint wire's gate reference is the low level's FROZEN reference clip, not its encoders,
    so tracking measured would move the wire AWAY from what the gate wants. The dense wire tracks;
    this one must not, and the asymmetry is a documented consequence rather than an oversight."""
    wire = _wire()
    assert np.array_equal(wire.idle()[0], READY_POS)
    assert wire.measured(object()) is None


def test_a_carried_command_freezes_with_no_measured_reference():
    """Refusing to integrate blind is the whole reason the clutch asks for a measured frame."""
    wire = _wire(post="command", measured_frame=None)
    wire.clutch.commit((READY_POS + 0.01, READY_ORN))
    note = wire.sync(object(), None)
    assert "FROZEN" in note


def test_sync_hold_slots_keeps_the_held_bases_where_commit_put_them_and_bounds_the_rest():
    """The leg override's feet are on the wire while the planner's own foot target is committed here;
    the MEASURED feet are walking, so bounding the planner's stance against them would drag it along
    the gait. Held slots keep their committed base; every other carried slot is bounded as before."""
    wire = _wire(post="command", measured_frame=lambda rframe: rframe)
    committed_pos = READY_POS.reshape(5, 3) + np.array([[0.01, 0, 0]] * 5)
    wire.clutch.commit((committed_pos.reshape(-1), READY_ORN.reshape(-1)))
    measured = committed_pos.copy()
    measured[:3] += [0.20, 0.0, 0.0]          # the hands + torso lag 20 cm (a lead to bound)
    measured[3:] += [0.30, 0.0, 0.05]         # the feet are mid-stride, 30 cm away
    wire.externally_driven = frozenset({3, 4})
    assert wire.hold_slots == (3, 4)
    note = wire.sync((measured.reshape(-1), READY_ORN.reshape(-1)), None)
    base = wire.clutch.base[0].reshape(5, 3)
    eps = wire.clutch.post.eps_m
    # Bounded slots: pulled to exactly eps from measured. Held slots: untouched.
    assert np.allclose(np.linalg.norm(base[:3] - measured[:3], axis=1), eps)
    assert np.array_equal(base[3:], committed_pos[3:])
    assert "2 slot(s) HELD" in note and "CARRIED" in note
    # Release: the same measured frame now bounds the feet too, and the note no longer says HELD.
    wire.externally_driven = frozenset()
    note2 = wire.sync((measured.reshape(-1), READY_ORN.reshape(-1)), None)
    base2 = wire.clutch.base[0].reshape(5, 3)
    assert np.allclose(np.linalg.norm(base2[3:] - measured[3:], axis=1), eps) and "HELD" not in note2
    # The status line names the externally driven slots by role while they are.
    wire.externally_driven = frozenset({3, 4})
    assert "left_foot,right_foot EXTERNALLY DRIVEN" in wire.status(wire.idle())
    wire.externally_driven = frozenset()
    assert "EXTERNALLY DRIVEN" not in wire.status(wire.idle())


def test_hold_slots_default_empty_is_byte_identical_and_the_completion_wire_holds_nothing():
    a = _wire(post="command", measured_frame=lambda rframe: rframe)
    b = _wire(post="command", measured_frame=lambda rframe: rframe)
    frame = (READY_POS.reshape(-1) + 0.01, READY_ORN.reshape(-1))
    meas = (READY_POS.reshape(-1) + 0.2, READY_ORN.reshape(-1))
    a.clutch.commit(frame)
    b.clutch.commit(frame)
    na = a.sync(meas, None)
    nb = b.clutch.sync(meas, None)               # the clutch called with no hold_slots at all
    assert na == nb and np.array_equal(a.clutch.base[0], b.clutch.base[0])
    completion = hl_dense.FivePointWire(hl_dense.KeypointReadout(np.zeros(3)), READY_POS, READY_ORN)
    assert completion.hold_slots == ()


def test_the_slew_ratio_flags_a_chunk_the_low_level_will_clamp():
    """The one risk on a keypoint wire that NO gate catches: every clamped frame is a perfectly
    valid keypoint command, so an over-budget chunk reads as a sluggish policy. Measured on the
    shipped H=50 checkpoint the worst per-step hand demand is 182-285 mm against 8.33 mm."""
    labels = hl_dense.FIVE_POINT_LABELS
    slow = np.tile(READY_POS, (4, 1)) + np.arange(4)[:, None] * 0.001    # 1 mm/step
    fast = np.tile(READY_POS, (4, 1)) + np.arange(4)[:, None] * 0.100    # 100 mm/step
    assert hl_5pt.chunk_span(slow, labels).slew_ratio < 1.0
    assert "⚠️" not in hl_5pt.chunk_span(slow, labels).describe()
    span = hl_5pt.chunk_span(fast, labels)
    assert span.slew_ratio > 4.0
    assert "CLAMP" in span.describe()


def test_the_slew_budget_is_the_LAUNCHERS_cap_not_the_submodule_default():
    """⚠️ `3_run_policy_vive_force.sh` — the launcher every 5pt runbook T4 invokes — pins
    HOLOSOMA_VIVE_MAX_POS_STEP=0.005 as an INLINE prefix on its single run_policy call, so it beats
    any exported env var and applies to every MODE. Quoting the submodule's 0.02 code default
    under-reported every printed ratio by exactly 4x. A chunk step is also 1.67 TICKS, not one."""
    assert hl_5pt.LOW_LEVEL_SLEW_M_PER_TICK == pytest.approx(0.005)
    assert hl_5pt.SLEW_BUDGET_PER_STEP_M == pytest.approx(0.005 * 50.0 / 30.0)
    assert hl_5pt.SLEW_BUDGET_PER_STEP_M * 1e3 == pytest.approx(8.333, abs=1e-3)


def test_the_ratio_is_scored_on_the_CONCATENATED_step_because_the_cap_is_one_pool():
    """`_slew_vec` takes one L2 over the whole (15,) vector and scales the ENTIRE delta, so five
    points moving 1 mm each demand sqrt(5) mm, not 1 mm. A per-point maximum under-reports the
    demand by up to sqrt(5) on top of the 4x — the two errors compounded to ~8.9x."""
    labels = hl_dense.FIVE_POINT_LABELS
    pos = np.tile(READY_POS, (2, 1))
    pos[1].reshape(N, 3)[:, 0] += 0.001                  # every point moves 1 mm, in +x only
    span = hl_5pt.chunk_span(pos, labels)
    assert span.slew_mm.max() == pytest.approx(1.0, abs=1e-6)          # per point: 1 mm
    assert span.step_mm_total == pytest.approx(np.sqrt(N), rel=1e-6)   # all points: sqrt(5) mm
    assert span.step_mm_total >= span.slew_mm.max()


def test_the_implausible_speed_flag_needs_no_budget_and_no_per_task_tuning():
    """The cap-independent half of the report. A speed transfers between tasks and checkpoints; a
    budget ratio moves whenever the launcher is edited. 285 mm per 30 Hz step is 8.6 m/s, which no
    operator demonstrated — that is a prediction DISCONTINUITY, and raising the cap would execute
    the jump instead of clamping it."""
    labels = hl_dense.FIVE_POINT_LABELS
    demo = np.tile(READY_POS, (2, 1))
    demo[1].reshape(N, 3)[0, 0] += 0.004                     # one hand, 4 mm/step = 0.12 m/s
    jump = np.tile(READY_POS, (2, 1))
    jump[1].reshape(N, 3)[0, 0] += 0.285                     # one hand, 285 mm/step = 8.55 m/s
    assert hl_5pt.chunk_span(demo, labels).peak_speed_ms == pytest.approx(0.12, abs=1e-3)
    assert not hl_5pt.chunk_span(demo, labels).implausible
    span = hl_5pt.chunk_span(jump, labels)
    assert span.peak_speed_ms == pytest.approx(8.55, abs=0.01)
    assert span.implausible and "DISCONTINUITY" in span.describe()
    # A looser cap cannot silence it — that is the point.
    loose = hl_5pt.chunk_span(jump, labels, budget=hl_5pt.SlewBudget(m_per_tick=10.0))
    assert loose.slew_ratio < 1.0 and loose.implausible


def test_the_printed_budget_says_whether_it_was_measured_or_assumed():
    """An assumed budget and a measured one produce identical-looking ratios. Provenance is the only
    thing that distinguishes 'the robot is clamping you' from 'we guessed at the launcher'."""
    labels = hl_dense.FIVE_POINT_LABELS
    pos = np.tile(READY_POS, (2, 1))
    pos[1] += 0.100
    assert "assumed" in hl_5pt.chunk_span(pos, labels).describe()
    measured = hl_5pt.SlewBudget(m_per_tick=0.005, source="MEASURED from RecordTap — 200 ticks")
    assert "MEASURED" in hl_5pt.chunk_span(pos, labels, budget=measured).describe()


def _rframe(vr, seq, armed=True, slewed=None):
    return SimpleNamespace(vr_cmd_pos=np.asarray(vr, dtype=np.float64), seq=seq, vive_armed=armed,
                           vr_cmd_slewed=None if slewed is None else np.asarray(slewed, dtype=bool))


def test_the_monitor_measures_only_the_points_the_low_level_says_it_caps():
    """`HOLOSOMA_VIVE_SLEW_FEET=0` exempts the two feet and the low level says so on every frame
    (`vr_cmd_slewed`). A whole-vector norm would then read a walking foot's 40 mm/tick as the
    ceiling and void the upper points' genuine 5 mm plateau; the norm runs over the covered rows."""
    mon = hl_5pt.SlewMonitor()
    assert mon.feet_slewed is None and mon.slew_mask is None
    p = READY_POS.reshape(N, 3)
    mask = [True, True, True, False, False]
    for i in range(120):
        q = p.copy()
        q[0, 0] += 0.005 * i             # upper: saturated at the cap
        q[3, 0] += 0.040 * i             # feet: a gait, 8x the cap, un-slewed
        q[4, 0] -= 0.040 * i
        mon.observe(_rframe(q, seq=i, slewed=mask))
    b = mon.budget
    assert b.m_per_tick == pytest.approx(0.005, rel=1e-6) and "MEASURED" in b.source
    assert mon.feet_slewed is False and list(mon.slew_mask) == mask
    assert "FEET EXEMPT" in mon.describe()
    # Without the key (an older low level) the SAME frames read as one pool — the feet's 40 mm
    # dominate, so there is no plateau at 5 mm and nothing is measured: byte-identical to before.
    old = hl_5pt.SlewMonitor()
    for i in range(120):
        q = p.copy()
        q[0, 0] += 0.005 * i
        q[3, 0] += 0.040 * i
        q[4, 0] -= 0.040 * i
        old.observe(_rframe(q, seq=i))
    assert old.feet_slewed is None and "assumed ALL points" in old.describe()
    assert old._max_step > 0.05
    # A mask that says NOTHING is capped measures nothing (there is no ceiling), and a mask of the
    # wrong width is ignored rather than mis-applied.
    none = hl_5pt.SlewMonitor()
    for i in range(60):
        q = p.copy()
        q[0, 0] += 0.005 * i
        none.observe(_rframe(q, seq=i, slewed=[False] * 5))
    assert none.budget.n_ticks == 0 and none.feet_slewed is False
    bad = hl_5pt.SlewMonitor()
    bad.observe(_rframe(p, seq=0, slewed=[True, False]))
    assert bad.slew_mask is None
    # The coverage is the robot's statement: remembered from UN-armed frames too.
    idle = hl_5pt.SlewMonitor()
    idle.observe(_rframe(p, seq=0, armed=False, slewed=mask))
    assert idle.feet_slewed is False and idle.budget.n_ticks == 0


def test_chunk_span_scores_the_covered_pool_when_the_feet_are_exempt():
    """A chunk whose FEET are fast is a clamp under a whole-vector cap and NOT under the
    launcher's feet-exempt cap; `covered` (the reported mask) is what tells the two apart. The
    cap-independent numbers (per-point step, peak speed) are unchanged either way."""
    labels = hl_dense.FIVE_POINT_LABELS
    pos = np.tile(READY_POS.reshape(-1), (20, 1))
    pos[:, 9] += np.arange(20) * 0.020          # left foot: 20 mm/step, 2.4x the 8.33 mm/step budget
    pos[:, 0] += np.arange(20) * 0.002          # left hand: 2 mm/step
    whole = hl_5pt.chunk_span(pos, labels)
    feet_free = hl_5pt.chunk_span(pos, labels, covered=[True, True, True, False, False])
    assert whole.slew_ratio > 2.0 and "⚠️" in whole.describe()
    assert feet_free.slew_ratio < 0.5 and "⚠️" not in feet_free.describe()
    assert "exempt" in feet_free.describe() and feet_free.covered == (True, True, True, False, False)
    assert np.allclose(whole.slew_mm, feet_free.slew_mm) and whole.peak_speed_ms == feet_free.peak_speed_ms
    # None / a wrong-width mask: the historical whole-vector reading.
    assert hl_5pt.chunk_span(pos, labels, covered=None).step_mm_total == whole.step_mm_total
    assert hl_5pt.chunk_span(pos, labels, covered=[True, False]).step_mm_total == whole.step_mm_total


def test_the_monitor_MEASURES_the_cap_off_recordtap_when_the_wire_saturates():
    """The whole answer to 'I will forget to retune this per task': the number is read off the
    robot. A clamped low level emits the SAME per-tick step every tick (`_slew_vec` scales the delta
    to exactly max_step), so a plateau in `vr_cmd_pos` differences IS the cap."""
    mon = hl_5pt.SlewMonitor()
    p = np.tile(READY_POS.reshape(N, 3), (1, 1))
    for i in range(120):                     # saturated: exactly 5 mm/tick, all in one point
        q = p.copy()
        q[0, 0] += 0.005 * i
        mon.observe(_rframe(q, seq=i))
    b = mon.budget
    assert b.m_per_tick == pytest.approx(0.005, rel=1e-6)
    assert "MEASURED" in b.source and mon.saturated_frac > 0.9
    assert b.mm_per_step == pytest.approx(8.333, abs=1e-3)


def test_a_skipped_tick_is_normalised_per_tick_rather_than_read_as_a_double_ceiling():
    """RecordTap is ONE frame per control tick, but a reader misses one now and then (a 50 Hz
    publisher against a 50 Hz tap — the walk skill) or reads it only at a chunk boundary. A raw
    difference over k ticks is k times the cap: on the mock rig ONE skipped tick read as a
    '10.0 mm/tick ceiling' and VOIDED a genuine 5 mm plateau over 367 clamped ticks, so the page
    printed 'assumed ... no clamping observed' while every tick was clamped."""
    mon = hl_5pt.SlewMonitor()
    p = READY_POS.reshape(N, 3)
    seq = 0
    for i in range(120):                     # saturated at 5 mm/tick, one tick SKIPPED every 10
        seq += 2 if i % 10 == 0 else 1
        q = p.copy()
        q[0, 0] += 0.005 * seq
        mon.observe(_rframe(q, seq=seq))
    b = mon.budget
    assert b.m_per_tick == pytest.approx(0.005, rel=1e-6)
    assert "MEASURED" in b.source and mon.saturated_frac > 0.9
    # A gap past MAX_GAP_TICKS re-anchors WITHOUT measuring: the target may have turned in between,
    # so the displacement says nothing about a per-tick cap. (This is what a chunk-boundary reader
    # ~27 ticks apart now does, instead of reporting a 27-tick displacement as one step.)
    far = hl_5pt.SlewMonitor()
    far.observe(_rframe(p, seq=0))
    q = p.copy()
    q[0, 0] += 0.005 * 30
    far.observe(_rframe(q, seq=30))
    assert far.budget.n_ticks == 0 and "no armed" in far.describe()
    # ... and a gap of exactly 1 is byte-identical to before (the existing saturation test above).
    assert hl_5pt.SlewMonitor.MAX_GAP_TICKS >= 2


def test_a_slow_take_leaves_the_conservative_estimate_rather_than_tightening_it():
    """⚠️ Without a plateau the observation is only 'the wire never asked for more than X', which
    bounds the cap from BELOW. Reporting that as the budget would invent a clamp that is not
    happening and tighten every ratio on a slow take."""
    mon = hl_5pt.SlewMonitor()
    p = READY_POS.reshape(N, 3)
    rng = np.random.default_rng(0)
    for i in range(200):                     # random small steps: no ceiling is ever hit twice
        q = p + rng.uniform(0.0, 0.0004, size=(N, 3)) * i
        mon.observe(_rframe(q, seq=i))
    b = mon.budget
    assert b.m_per_tick == pytest.approx(hl_5pt.LOW_LEVEL_SLEW_M_PER_TICK)
    assert "assumed" in b.source and "never reached a ceiling" in b.source


def test_the_monitor_ignores_UNARMED_ticks_and_never_bridges_across_an_arm():
    """Un-armed, the low level is not applying the wire — its target is whatever the idle path
    holds, so differencing there measures nothing about the cap, and differencing the first ARMED
    tick against a stale idle target would fabricate a huge step."""
    mon = hl_5pt.SlewMonitor()
    p = READY_POS.reshape(N, 3)
    for i in range(60):
        mon.observe(_rframe(p + 0.5, seq=i, armed=False))     # far-away idle target
    assert mon.budget.n_ticks == 0 and "no armed" in mon.describe()
    q = p.copy()
    q[0, 0] += 0.005
    mon.observe(_rframe(p, seq=100))                          # first ARMED tick
    mon.observe(_rframe(q, seq=101))
    assert mon._max_step == pytest.approx(0.005, rel=1e-6)    # 5 mm, not the 0.5 m idle gap


def test_an_empty_or_single_step_span_is_zero_not_a_crash():
    labels = hl_dense.FIVE_POINT_LABELS
    for pos in (np.zeros((0, 3 * N)), np.zeros((1, 3 * N))):
        span = hl_5pt.chunk_span(pos, labels)
        assert span.slew_mm.shape == (N,) and float(span.slew_mm.max()) == 0.0
        assert span.step_mm_total == 0.0 and not span.implausible


# ---------------------------------------------------------------------------
# the observation
# ---------------------------------------------------------------------------
def test_the_planner_refuses_every_layout_it_cannot_serve_by_name():
    for mode, needle in (("wholebody", "29 JOINT"),
                         ("wholebody_geom_xy_rpy", "29 JOINT"),
                         ("legacy32", "pre-v2")):
        with pytest.raises(SystemExit, match=needle):
            hl_5pt.validate_layout(get_layout(mode))
    assert hl_5pt.validate_layout(LAY) == "fivepoint"
    # A wrt_start keypoint layout is no longer refused: its missing slots are HELD (see the
    # "keypoint SUBSET" section below). It is the OTHER family, and the return value says so.
    assert hl_5pt.validate_layout(get_layout("both_hands_head_dual")) == "keypoint_subset"


@pytest.mark.parametrize("mode", ["wholebody", "both_hands_head_dual"])
def test_the_planner_constructor_refuses_the_same_layouts(mode):
    with pytest.raises(ValueError):
        hl_5pt.FivePointPlanner(replay_only_model(mode, obs_history_len=1), obs_hz=30.0)


def test_the_state_is_the_anchors_own_geometry_then_the_two_grippers():
    """The load-bearing claim: the state is computed from the anchor poses through the SAME
    ``wb_geom.geom_from_bodies`` training used, with no pelvis correction (``rel_xy3`` is
    translation-invariant and the anchor frame's pelvis rotation is identity)."""
    from flore.s1_policy.data import wb_geom

    planner = _planner(To=2)
    pos = np.array([[0.44, 0.15, 0.85], [0.44, -0.15, 0.85], [0.0, 0.0, 1.15],
                    [0.0, 0.12, 0.03], [0.0, -0.12, 0.03]])
    rot6d = np.tile(rotmat_to_rot6d(np.eye(3)), (N, 1))
    planner.seed_anchor(pos, rot6d)
    planner.push_proprio_geom()
    obs, _ap, _ar = planner.snapshot_obs()
    want = wb_geom.geom_from_bodies(
        {body: pos[hl_5pt.WIRE_ROLES.index(role)] for body, role in hl_5pt.GEOM_BODY_ROLE.items()},
        {body: np.eye(3) for body in hl_5pt.GEOM_BODY_ROLE},
        tuple(LAY.state_geom))
    got = np.concatenate([obs.state[-1, LAY.state_geom_slots()[b]] for b in LAY.state_geom])
    assert np.allclose(got, want, atol=0)
    # And the gripper columns are where the layout says, not folded into the geometry.
    for role in LAY.gripper_roles:
        col = LAY.state_slots()[role]["gripper"]
        assert obs.state[-1, col] == pytest.approx(0.0)


def test_the_seeded_state_is_flagged_unmeasured_until_something_is_committed():
    """The ready pose is a real pose but not a target THIS robot was given, so the model's own
    no-data embedding must handle it rather than being told the seed is where the body is."""
    planner = _planner(To=1)
    _geom, written = planner.geom_state()
    assert not written
    obs, _ap, _ar = planner.snapshot_obs()
    for block in LAY.state_geom:
        assert not obs.state_valid[:, LAY.state_geom_slots()[block]].any()
    planner.commit_role("left", np.array([0.4, 0.1, 0.8]), rotmat_to_rot6d(np.eye(3)))
    _geom, written = planner.geom_state()
    assert written
    obs, _ap, _ar = planner.snapshot_obs()
    for block in LAY.state_geom:
        assert obs.state_valid[:, LAY.state_geom_slots()[block]].all()


def test_push_proprio_geom_advances_all_three_rings():
    """⚠️ A REAL BUG IN THE SIBLING: advancing only the pose rings left both gripper columns at
    their ``__init__`` zeros — the CLOSED end of a legal range — for the whole session, which
    nothing downstream can notice because a shut jaw is not an error."""
    planner = _planner(To=2)
    planner.push_gripper("left", 0.05)
    planner.push_gripper("right", 0.09)
    planner.push_proprio_geom()
    obs, _ap, _ar = planner.snapshot_obs()
    left = obs.state[-1, LAY.state_slots()["left"]["gripper"]]
    right = obs.state[-1, LAY.state_slots()["right"]["gripper"]]
    assert left > 0.0 and right > left


def test_a_reset_re_seeds_the_ring_and_re_flags_it():
    planner = _planner(To=2)
    planner.commit_role("left", np.array([0.4, 0.1, 0.8]), rotmat_to_rot6d(np.eye(3)))
    assert planner.geom_state()[1]
    planner.seed_anchor(np.zeros((N, 3)), np.tile(rotmat_to_rot6d(np.eye(3)), (N, 1)))
    assert not planner.geom_state()[1]


def test_infer_chunk_refuses_a_planner_with_no_policy():
    with pytest.raises(RuntimeError, match="no policy"):
        _planner(To=1).infer_chunk(seed=0, force_enabled=False, gate_n=0.0)


# ---------------------------------------------------------------------------
# the rig-backed half: the anchor, the readout, and the display IK
# ---------------------------------------------------------------------------
@needs_rig
def test_the_deploy_anchor_answers_for_all_five_roles():
    from flore.s1_policy.inference import ready_pose as RP

    anchor, why = RP.resolve_deploy_anchor(LAY.roles)
    assert anchor is not None, why
    assert set(anchor.pos) == set(LAY.roles)
    # The foot ROLE and the foot SIDE must be the same numbers reached two ways.
    assert np.allclose(anchor.pos["left_foot"], anchor.foot_pos["left"])
    assert np.allclose(anchor.rotmat["right_foot"], anchor.foot_rotmat["right"])


@needs_rig
def test_the_ready_pose_wire_is_holosomas_own_gate_reference():
    """The readout pelvis must be the ANCHOR's ``(0, 0, pelvis_z)`` and never
    ``mc_frames.PELVIS_STAND_Z`` (0.79186), which belongs to the teleop path's ``cond_mode=home``
    placement and would put every keypoint 35 mm low."""
    from flore.s1_policy.inference import ready_pose as RP

    anchor, why = RP.resolve_deploy_anchor(LAY.roles)
    assert anchor is not None, why
    readout = hl_dense.KeypointReadout(np.array([0.0, 0.0, anchor.pelvis_z]))
    pos, orn = hl_dense.ready_pose_five_point(anchor, readout)
    p = pos.reshape(N, 3)
    # holosoma's own baked 3-point reference, to 0.1 mm (the value hl_dense's docstring quotes).
    assert np.allclose(p[0], [0.4418, 0.1486, 0.0952], atol=1e-4)
    assert np.allclose(p[1], [0.4418, -0.1486, 0.0952], atol=1e-4)
    assert orn.shape == (4 * N,)
    assert np.all(orn.reshape(N, 4)[:, 3] >= 0.0)          # the +w branch the policy was trained on


@needs_rig
def test_the_ik_reproduces_the_ready_pose_exactly():
    """The ONE test that catches a lever, a chain or a root-placement mistake: FK a real
    configuration into five keypoints, solve them back, FK the answer. Any disagreement between
    this IK and the FK the dataset and the wire are built on shows up here as millimetres."""
    pytest.importorskip("yourdfpy")
    from flore.s1_policy.inference import ready_pose as RP
    from flore.s1_policy.inference.keypoint_ik import KeypointIK

    ik, why = KeypointIK.load()
    assert ik is not None, why
    anchor, why2 = RP.resolve_deploy_anchor(LAY.roles)
    assert anchor is not None, why2
    mm, deg = ik.check_round_trip(anchor.dof)
    assert mm < 1e-6, f"{mm} mm"
    assert deg < 1e-6, f"{deg} deg"


@needs_rig
def test_the_ik_round_trips_an_arbitrary_reachable_posture():
    """Not only the ready pose: a perturbed configuration exercises all four chains away from the
    seed, which is where a wrong Jacobian axis or a swapped chain would show."""
    pytest.importorskip("yourdfpy")
    from flore.s1_policy.inference import ready_pose as RP
    from flore.s1_policy.inference.keypoint_ik import KeypointIK

    ik, _ = KeypointIK.load()
    anchor, _ = RP.resolve_deploy_anchor(LAY.roles)
    rng = np.random.default_rng(0)
    worst = 0.0
    for _ in range(3):
        dof = np.asarray(anchor.dof, dtype=np.float64) + rng.normal(scale=0.10, size=29)
        mm, _deg = ik.check_round_trip(dof)
        worst = max(worst, mm)
    assert worst < 1.0, f"worst {worst} mm over 3 perturbed postures"


@needs_rig
def test_the_chest_is_the_root_so_its_residual_is_structurally_zero():
    """"Treat the chest as the pelvis" is a placement, not a solve — which is why the page can say
    the drawn pelvis is a display artefact. If the chest were ever solved instead, its residual
    would become non-zero and the claim in the panel would quietly stop being true."""
    pytest.importorskip("yourdfpy")
    from flore.s1_policy.inference import ready_pose as RP
    from flore.s1_policy.inference.keypoint_ik import IK_ROLES, ROOT_ROLE, KeypointIK

    ik, _ = KeypointIK.load()
    anchor, _ = RP.resolve_deploy_anchor(LAY.roles)
    pos_b, rot_b = ik.fk.bodies(np.asarray(anchor.dof) + 0.05, lift_z=0.0)
    poses = {r: (pos_b[RP.ROLE_BODY[r]][None, :], rot_b[RP.ROLE_BODY[r]][None, ...])
             for r in IK_ROLES}
    out = ik.solve_chunk(poses, seed_dof=anchor.dof)
    assert out.roles[0] == ROOT_ROLE
    assert float(out.residual_mm[0, 0]) == pytest.approx(0.0, abs=1e-9)


@needs_rig
def test_the_ik_refuses_a_layout_it_cannot_draw_a_body_from():
    pytest.importorskip("yourdfpy")
    from flore.s1_policy.inference.keypoint_ik import IK_ROLES, KeypointIK

    ik, _ = KeypointIK.load()
    partial = {r: (np.zeros((2, 3)), np.tile(np.eye(3), (2, 1, 1)))
               for r in IK_ROLES[:3]}
    with pytest.raises(ValueError, match="left_foot|right_foot"):
        ik.solve_chunk(partial)


@needs_rig
def test_the_ik_chains_are_seven_seven_six_six():
    """An arm is redundant (7 against a 6-DoF target) and a leg is determined (6). A URDF rename
    that made the substring match select the wrong set is refused at construction, by name."""
    pytest.importorskip("yourdfpy")
    from flore.s1_policy.inference.keypoint_ik import CHAIN_DOF, CHAIN_SPEC

    assert set(CHAIN_SPEC) == set(CHAIN_DOF)
    assert [CHAIN_DOF[r] for r in ("left", "right", "left_foot", "right_foot")] == [7, 7, 6, 6]


@needs_rig
def test_the_ik_dof_and_cfg_are_the_same_solution():
    """``cfg`` drives the ViserUrdf and ``dof`` is the SDK view of it; a caller that FKs ``dof``
    back must land where the drawn body is. The two must agree under the by-name remap."""
    pytest.importorskip("yourdfpy")
    from flore.s1_policy.inference import ready_pose as RP
    from flore.s1_policy.inference.keypoint_ik import IK_ROLES, KeypointIK

    ik, _ = KeypointIK.load()
    anchor, _ = RP.resolve_deploy_anchor(LAY.roles)
    pos_b, rot_b = ik.fk.bodies(anchor.dof, lift_z=0.0)
    poses = {r: (pos_b[RP.ROLE_BODY[r]][None, :], rot_b[RP.ROLE_BODY[r]][None, ...])
             for r in IK_ROLES}
    out = ik.solve_chunk(poses, seed_dof=anchor.dof)
    assert out.dof.shape == (1, 29)
    assert np.allclose(out.dof[0], anchor.dof, atol=1e-6)


# ---------------------------------------------------------------------------
# the sampler seed: Reset must restore it, and a pin must reach every sample site
# ---------------------------------------------------------------------------
def test_the_seed_increments_per_sample_when_unpinned():
    """The historical numbering, kept: the FIRST sample is seed 1, not 0.

    A log line from before ``--seed`` existed names a chunk by this number, so shifting the origin
    would silently re-point every one of them.
    """
    c = hl_5pt.SeedCounter()
    assert [c.next() for _ in range(4)] == [1, 2, 3, 4]
    assert not c.fixed


def test_reset_returns_the_seed_so_the_next_predict_repeats_the_first():
    """THE REGRESSION this was added for.

    ``Reset to ready pose`` restored the observation exactly — ``seed_anchor`` re-seeds the pose
    rings and clears ``_geom_written`` (see the two tests above) — and the next Predict still
    answered differently, because the seed had moved on. Measured on the ``push_cart`` H=100
    checkpoint with the observation held bit-identical, the seed alone is worth 6.7-18.6 mm at the
    worst step of a 65-88 mm chunk: the same gesture on a visibly different path, with nothing on
    the page to say why.
    """
    c = hl_5pt.SeedCounter()
    first = [c.next() for _ in range(3)]
    c.reset()
    assert [c.next() for _ in range(3)] == first == [1, 2, 3]


def test_predicting_again_without_a_reset_is_still_a_fresh_draw():
    """The escape hatch the reset must not close: a bad chunk is re-sampled by pressing Predict.

    Only ``Reset`` rewinds, so "give me another candidate" and "go back to the start" stay two
    different gestures.
    """
    c = hl_5pt.SeedCounter()
    assert c.next() != c.next()


def test_a_pinned_seed_is_constant_at_every_sample_site():
    """One observation, one chunk — including across the streaming cadence's own sampler thread.

    ``next()`` is the ONLY way a seed is drawn in ``hl_serving_5pt.py`` (``_predict`` and the
    stream's ``_sample_once``), so pinning cannot be honoured at one site and forgotten at another.
    """
    c = hl_5pt.SeedCounter(42)
    assert c.fixed and [c.next() for _ in range(5)] == [42] * 5
    c.reset()
    assert c.next() == 42


def test_seed_zero_is_a_pin_and_not_absent():
    """``--seed 0`` must pin, not read as "unset" — the flag's default is ``None`` for exactly this.

    A truthiness test here would make 0 the one unpinnable seed, and it is the value anyone reaches
    for first.
    """
    c = hl_5pt.SeedCounter(0)
    assert c.fixed and [c.next() for _ in range(3)] == [0, 0, 0]


def test_the_seed_status_names_which_regime_it_is_in():
    """The page shows this string; an operator has to be able to tell a pin from a counter."""
    assert "PINNED" in hl_5pt.SeedCounter(7).describe()
    unpinned = hl_5pt.SeedCounter()
    assert "PINNED" not in unpinned.describe()
    unpinned.next()
    assert "1" in unpinned.describe()


# ---------------------------------------------------------------------------
# the violet body's staging: Reset must retire a solve that is still running
# ---------------------------------------------------------------------------
def test_a_solve_in_flight_cannot_publish_after_a_retire():
    """THE REGRESSION. ``Reset to ready pose`` parked the violet body and dropped the solved
    chunk, but left the generation alone — so an IK solve already running still passed its own
    ``gen`` check and republished the OLD chunk's body a moment later. The purple robot came back
    to where the last take ended and stayed there until the next Predict, and nothing downstream
    could notice, the body being display-only.

    The horizon is why it looked intermittent rather than like a race: a solve is ~0.3 s at H=50
    and seconds at H=200, against a 533 ms drip.
    """
    gate = hl_5pt.PreviewGate()
    gen = gate.submit()                      # a Predict queues a solve
    gate.publish(gen, "body-A", "chunk A")
    assert gate.solved == "body-A"

    gen_inflight = gate.submit()             # a second Predict; this solve is now running
    gate.retire()                            # ... and the operator presses Reset mid-solve
    assert gate.solved is None
    assert gate.publish(gen_inflight, "body-B", "chunk B") is False, \
        "a solve retired by Reset must not be able to put a body back on screen"
    assert gate.solved is None
    assert gate.status() == "no solve yet"


def test_submit_retires_the_previous_solve_so_only_the_newest_body_lands():
    """The other half of the same invariant: a superseded chunk's solve must lose to the newer
    one whatever order the two threads finish in."""
    gate = hl_5pt.PreviewGate()
    stale = gate.submit()
    fresh = gate.submit()
    assert gate.publish(stale, "old", "old") is False
    assert gate.publish(fresh, "new", "new") is True
    assert gate.solved == "new"


def test_a_pending_solve_says_the_body_shown_is_the_previous_chunk():
    """A 200-step solve outlasts the drip, so during a sequential run the page is usually showing
    the PREVIOUS chunk's body. That must not read as the current plan."""
    gate = hl_5pt.PreviewGate()
    assert gate.status() == "no solve yet"
    gate.submit()
    assert gate.status() == "solving …"           # nothing on screen yet
    gate.publish(gate._gen, "body", "residual 0.4 mm")   # noqa: SLF001 - the token IS the test
    assert gate.status() == "residual 0.4 mm"
    gate.submit()                                  # a new Predict, old body still drawn
    assert gate.status() == "solving … (the body shown is the PREVIOUS chunk)"
    assert gate.solved == "body"


def test_a_failed_solve_clears_pending_without_publishing():
    gate = hl_5pt.PreviewGate()
    gen = gate.submit()
    gate.give_up(gen)
    assert gate.solved is None and not gate.pending
    assert gate.status() == "no solve yet"


def test_give_up_from_a_retired_solve_does_not_clear_a_newer_pending():
    """A failure report from a solve Reset already retired must not cancel the flag belonging to
    the solve that replaced it."""
    gate = hl_5pt.PreviewGate()
    stale = gate.submit()
    gate.submit()                                  # the newer solve is now the pending one
    gate.give_up(stale)
    assert gate.pending, "the newer solve is still on its way"


# ---------------------------------------------------------------------------
# no regression on the siblings
# ---------------------------------------------------------------------------
def test_the_violet_is_the_same_violet_everywhere():
    """One vocabulary across the offline pages and the rig: violet = what is about to happen."""
    from flore.s1_policy.inference import keypoint_humanoid as KH
    from flore.s1_policy.inference import wholebody_viewer as WV

    assert KH.PREDICTION_RGBA == WV.PREDICTION_RGBA
    assert KH.TARGET_RGBA == WV.TARGET_RGBA


def test_the_completion_path_wire_is_untouched():
    """``FivePointWire``'s own defaults are the reviewed ones — this preset inverts them in its
    SUBCLASS, not in the base. A change here would silently re-default the rig's 3-tracker path."""
    ready_pos, ready_orn = READY_POS.copy(), READY_ORN.copy()
    wire = hl_dense.FivePointWire(hl_dense.KeypointReadout(np.zeros(3)), ready_pos, ready_orn)
    assert wire.upper_source == "completion" and wire.foot_source == "completion"
    assert wire.clutch.n_carried == hl_dense.FIVE_POINT_GATED      # three, not five
    assert not wire.clutch.post.carrying                          # `off`, so `apply` is identity


def test_the_keypoint_clutch_default_carry_is_still_three():
    """``KeypointClutch``'s own default must stay the completion path's, so ``hl_5pt`` asking for
    five is a visible declaration rather than a shared default nobody can see."""
    clutch = hl_dense.KeypointClutch(READY_POS, READY_ORN)
    assert clutch.n_carried == hl_dense.FIVE_POINT_GATED


# ---------------------------------------------------------------------------
# a keypoint SUBSET on the same wire: left_only (the plain diffusion_umi preset) first of all
# ---------------------------------------------------------------------------
LEFT = get_layout("left_only")


def _held_for(layout):
    """A distinct held pose per wire slot, so a slot permutation shows as a value mismatch."""
    return {r: (np.full(3, 10.0 + k), rotmat_to_rot6d(np.eye(3)))
            for k, r in enumerate(hl_5pt.WIRE_ROLES) if r not in layout.roles}


def _subset_planner(mode: str = "left_only", To: int = 2):
    return hl_5pt.SubsetKeypointPlanner(replay_only_model(mode, obs_history_len=To), obs_hz=30.0)


def test_check_layout_names_the_family_from_the_role_set_not_a_flag():
    assert hl_5pt.check_layout(LAY) == "fivepoint"
    for mode in ("left_only", "right_only", "both_hands", "both_hands_head", "both_hands_head_dual"):
        assert hl_5pt.check_layout(get_layout(mode)) == "keypoint_subset"
    for mode in ("wholebody", "wholebody_geom_xy_rpy", "wholebody_geom_none"):
        with pytest.raises(ValueError, match="29 JOINT"):
            hl_5pt.check_layout(get_layout(mode))
    assert set(hl_5pt.LAYOUT_FAMILIES) == {"fivepoint", "keypoint_subset"}


def test_held_roles_are_the_wire_slots_the_layout_does_not_predict_in_wire_order():
    assert hl_5pt.held_roles(LEFT) == ("right", "head", "left_foot", "right_foot")
    assert hl_5pt.held_roles(get_layout("both_hands")) == ("head", "left_foot", "right_foot")
    assert hl_5pt.held_roles(get_layout("both_hands_head_dual")) == ("left_foot", "right_foot")
    assert hl_5pt.held_roles(LAY) == ()


def test_fill_held_roles_broadcasts_a_constant_row_and_refuses_an_unfilled_slot_by_name():
    chunk = _chunk(h=6, layout=LEFT)
    rows = {rc.role: {"position": rc.position, "rot6d": rc.rot6d} for rc in chunk.roles}
    held = _held_for(LEFT)
    out = hl_5pt.fill_held_roles(rows, held, 6)
    assert set(out) == set(hl_5pt.WIRE_ROLES)
    assert out["left"]["position"] is rows["left"]["position"]        # predicted: untouched
    for role, (pos, rot6d) in held.items():
        assert out[role]["position"].shape == (6, 3) and out[role]["rot6d"].shape == (6, 6)
        assert np.all(out[role]["position"] == pos) and np.all(out[role]["rot6d"] == rot6d)
    # Neither predicted nor held: refused, naming the slot — never zero-filled.
    with pytest.raises(ValueError, match="right_foot"):
        hl_5pt.fill_held_roles(rows, {k: v for k, v in held.items() if k != "right_foot"}, 6)
    with pytest.raises(ValueError, match="at least one row"):
        hl_5pt.fill_held_roles(rows, held, 0)


def test_held_anchor_poses_come_from_the_anchor_and_refuse_a_missing_role():
    anchor = SimpleNamespace(pos={r: np.full(3, float(k)) for k, r in enumerate(hl_5pt.WIRE_ROLES)},
                             rotmat={r: np.eye(3) for r in hl_5pt.WIRE_ROLES})
    held = hl_5pt.held_anchor_poses(anchor, LEFT)
    assert set(held) == {"right", "head", "left_foot", "right_foot"}
    assert held["head"][0].tolist() == [2.0, 2.0, 2.0]
    assert np.allclose(held["head"][1], rotmat_to_rot6d(np.eye(3)))
    assert hl_5pt.held_anchor_poses(anchor, LAY) == {}                 # five-point: nothing held
    partial = SimpleNamespace(pos={"left": np.zeros(3)}, rotmat={"left": np.eye(3)})
    with pytest.raises(ValueError, match="right"):
        hl_5pt.held_anchor_poses(partial, LEFT)


def test_a_left_only_chunk_publishes_five_slots_with_the_held_ones_on_the_ready_frame():
    """THE headline claim for a one-hand checkpoint: the wire is still 15/20 wide, the LEFT slot
    carries the chunk, and every HELD slot is the READY frame on EVERY step — under the clutch
    (base + 0) and under ``off`` (the held pose IS the ready pose) alike — so the pre-arm gate closes
    on them by construction and a Predict never moves them."""
    chunk = _chunk(h=8, layout=LEFT)
    readout = hl_dense.KeypointReadout(np.zeros(3))
    held = {r: (READY_POS[3 * k:3 * k + 3], rotmat_to_rot6d(np.eye(3)))
            for k, r in enumerate(hl_5pt.WIRE_ROLES) if r != "left"}
    left = chunk.role("left").position
    for post in ("command", "off"):
        clutch = hl_5pt.build_clutch(READY_POS, READY_ORN,
                                     post=hl_dense.PostAnchorConfig(mode=post), carry="all")
        wire = hl_5pt.FivePointDirectWire(readout, READY_POS, READY_ORN, clutch=clutch, held=held)
        assert wire.held_roles == ("right", "head", "left_foot", "right_foot")
        plan = wire.plan(chunk)
        assert plan["pos"].shape == (8, 15) and plan["orn"].shape == (8, 20) and plan["n"] == 8
        for k, role in enumerate(hl_5pt.WIRE_ROLES):
            if role == "left":
                continue
            assert np.array_equal(plan["pos"][:, 3 * k:3 * k + 3],
                                  np.tile(READY_POS[3 * k:3 * k + 3], (8, 1))), (post, role)
        if post == "command":
            assert np.allclose(plan["pos"][0, 0:3], READY_POS[0:3])                # step 0 exact
            assert np.allclose(plan["pos"][:, 0:3] - plan["pos"][0, 0:3], left - left[0])
        else:
            assert np.allclose(plan["pos"][:, 0:3], left)     # identity readout, absolute
        ok, _note = wire.gate_report(wire.frame(plan, 0))
        assert "HELD at the ready pose: right,head,left_foot,right_foot" in wire.status(
            wire.frame(plan, 0))
        for i in range(8):
            assert hl_dense.is_safe_keypoint_command(plan["pos"][i], plan["orn"][i], n_points=5)[0]


def test_a_held_slot_survives_the_streamed_two_row_integrator():
    """The CONTINUOUS cadence hands the wire ``mini_chunk``'s two rows of the PREDICTED roles only;
    the wire fills the held slots exactly as it does for a full chunk, so streaming a left_only
    checkpoint is the same clutch arithmetic on the same five-slot frame."""
    from flore.s1_policy.serving import hl_stream

    chunk = _chunk(h=8, layout=LEFT)
    held = {r: (READY_POS[3 * k:3 * k + 3], rotmat_to_rot6d(np.eye(3)))
            for k, r in enumerate(hl_5pt.WIRE_ROLES) if r != "left"}
    clutch = hl_5pt.build_clutch(READY_POS, READY_ORN,
                                 post=hl_dense.PostAnchorConfig(mode="command"), carry="all")
    wire = hl_5pt.FivePointDirectWire(hl_dense.KeypointReadout(np.zeros(3)), READY_POS, READY_ORN,
                                      clutch=clutch, held=held)
    stream = hl_stream.ChunkStream(8, LEFT.roles)
    stream.add(chunk, 0)
    plan = wire.plan(hl_stream.mini_chunk(stream.at(0), stream.at(1), LEFT.roles))
    assert plan["pos"].shape == (2, 15) and plan["orn"].shape == (2, 20)
    frame = wire.frame(plan, 1)
    assert np.array_equal(frame[0][3:], READY_POS[3:])              # held: on the base
    assert hl_dense.is_safe_keypoint_command(frame[0], frame[1], n_points=5)[0]


def test_decoded_positions_are_in_wire_order_and_a_held_slot_is_still():
    chunk = _chunk(h=5, layout=LEFT)
    wire = hl_5pt.FivePointDirectWire(hl_dense.KeypointReadout(np.zeros(3)), READY_POS, READY_ORN,
                                      held=_held_for(LEFT))
    pos = wire.decoded_positions(chunk)
    assert pos.shape == (5, 15)
    assert np.allclose(pos[:, 0:3], chunk.role("left").position)
    span = hl_5pt.chunk_span(pos, hl_dense.FIVE_POINT_LABELS)
    assert span.excursion_mm[1:].max() == 0.0 and span.slew_mm[1:].max() == 0.0
    assert span.excursion_mm[0] > 0.0


def test_the_wire_refuses_a_held_role_that_is_not_a_slot_or_is_malformed():
    readout = hl_dense.KeypointReadout(np.zeros(3))
    with pytest.raises(ValueError, match="not a wire slot"):
        hl_5pt.FivePointDirectWire(readout, READY_POS, READY_ORN,
                                   held={"torso": (np.zeros(3), rotmat_to_rot6d(np.eye(3)))})
    with pytest.raises(ValueError, match="finite"):
        hl_5pt.FivePointDirectWire(readout, READY_POS, READY_ORN,
                                   held={"right": (np.array([np.nan, 0.0, 0.0]),
                                                   rotmat_to_rot6d(np.eye(3)))})
    with pytest.raises(ValueError, match="finite"):
        hl_5pt.FivePointDirectWire(readout, READY_POS, READY_ORN,
                                   held={"right": (np.zeros(2), rotmat_to_rot6d(np.eye(3)))})


def test_a_five_point_wire_with_nothing_held_is_byte_identical_to_before():
    """The five-point path must not change: no held slots, the same plan, no HELD note."""
    chunk = _chunk(h=6)
    wire0, wire1 = _wire(), _wire()
    assert wire1.held == {} and wire1.held_roles == ()
    assert np.array_equal(wire0.plan(chunk)["pos"], wire1.plan(chunk)["pos"])
    assert "HELD" not in wire1.status(wire1.frame(wire1.plan(chunk), 0))
    # A five-point chunk on a wire that DOES carry held poses ignores them: predicted wins.
    wire2 = hl_5pt.FivePointDirectWire(hl_dense.KeypointReadout(np.zeros(3)), READY_POS, READY_ORN,
                                       held=_held_for(LEFT))
    assert np.array_equal(wire2.decoded_positions(chunk)[:, 3:6], chunk.role("right").position)


def test_force_command_is_always_both_hands_wide_with_a_zero_for_a_hand_not_predicted():
    from flore.utils.force_frame import cmd_from_meas_force

    row = np.array([0.0, 0.0, -5.0, 0.1, 0.2, 0.3])
    out = hl_5pt.force_command({"left": row}, gate_n=1.5)              # left_only
    assert out.shape == (6,)
    assert np.allclose(out[0:3], cmd_from_meas_force(row[0:3])) and np.all(out[3:6] == 0.0)
    out = hl_5pt.force_command({"right": row}, gate_n=1.5)             # right_only
    assert np.all(out[0:3] == 0.0) and np.allclose(out[3:6], cmd_from_meas_force(row[0:3]))
    out = hl_5pt.force_command({"left": None, "right": row}, gate_n=1.5)
    assert np.all(out[0:3] == 0.0) and np.allclose(out[3:6], cmd_from_meas_force(row[0:3]))
    # Below the gate, non-finite, absent: a hard zero for that hand; no live hand -> None (no key).
    assert hl_5pt.force_command({"left": np.array([0.0, 0.0, -1.0])}, gate_n=1.5) is None
    assert hl_5pt.force_command({}, gate_n=1.5) is None
    out = hl_5pt.force_command({"left": np.array([np.nan, 0.0, 0.0]), "right": row}, gate_n=1.5)
    assert np.all(out[0:3] == 0.0) and np.isfinite(out).all()
    assert hl_5pt.FORCE_HANDS == ("left", "right")


def test_state_names_follow_state_slots_for_both_families():
    assert hl_5pt.state_names(LEFT) == ("left_r6d0", "left_r6d1", "left_r6d2", "left_r6d3",
                                        "left_r6d4", "left_r6d5", "left_grip")
    assert len(hl_5pt.state_names(LEFT)) == LEFT.state_dim
    names5 = hl_5pt.state_names(LAY)
    assert len(names5) == LAY.state_dim and names5[-2:] == ("left_grip", "right_grip")
    dual = get_layout("both_hands_head_dual")
    names = hl_5pt.state_names(dual)
    assert len(names) == dual.state_dim
    assert names[dual.state_slots()["right"]["gripper"]] == "right_grip"
    assert names[dual.state_slots()["head"]["wrt_start_rot6d"].start] == "head_r6d0"


def test_build_planner_picks_the_family_and_refuses_the_rest():
    assert isinstance(hl_5pt.build_planner(replay_only_model(MODE, obs_history_len=1)),
                      hl_5pt.FivePointPlanner)
    planner = hl_5pt.build_planner(replay_only_model("left_only", obs_history_len=1))
    assert isinstance(planner, hl_5pt.SubsetKeypointPlanner)
    assert planner.roles == ("left",)
    with pytest.raises(ValueError, match="29 JOINT"):
        hl_5pt.build_planner(replay_only_model("wholebody", obs_history_len=1))


@pytest.mark.parametrize("mode", [MODE, "wholebody"])
def test_the_subset_planner_refuses_the_geometric_and_wholebody_layouts(mode):
    with pytest.raises(ValueError):
        hl_5pt.SubsetKeypointPlanner(replay_only_model(mode, obs_history_len=1), obs_hz=30.0)


def test_the_subset_state_is_the_identity_at_the_seed_then_R_start_T_R_now():
    """The observation a left_only checkpoint trained on: per role ``R_start^T R_now`` as rot6d,
    then the jaw. Identity before anything is commanded (every episode's frame 0), the committed
    rotation relative to the SEEDED one after — and ``state_readout`` must equal what
    ``snapshot_obs`` feeds the model."""
    from flore.s1_policy.data.transforms import rot6d_relative
    from flore.utils.rot6d import rot6d_to_rotmat as _r6, rotmat_to_rot6d as _r

    def rz(deg):
        c, s_ = np.cos(np.deg2rad(deg)), np.sin(np.deg2rad(deg))
        return np.array([[c, -s_, 0.0], [s_, c, 0.0], [0.0, 0.0, 1.0]])

    planner = _subset_planner(To=2)
    r0 = _r(rz(30.0))
    planner.seed_anchor(np.array([[0.4, 0.1, 0.9]]), r0[None])
    names, vec, written = planner.state_readout()
    assert names == hl_5pt.state_names(LEFT) and written is True
    assert np.allclose(vec[0:6], _r(np.eye(3)), atol=1e-12)          # identity at the seed
    assert vec[6] == 0.0
    planner.push_gripper("left", 0.05)
    planner.push_proprio()
    r1 = _r(rz(75.0))
    planner.commit_role("left", np.array([0.5, 0.1, 0.9]), r1)
    planner.push_proprio()
    obs, ap, ar = planner.snapshot_obs()
    _n, vec2, _w = planner.state_readout()
    assert np.allclose(vec2, obs.state[-1], atol=1e-12)               # the readout IS the input
    assert np.allclose(vec2[0:6], rot6d_relative(r1, r0), atol=1e-12)
    assert np.allclose(_r6(vec2[0:6]), rz(45.0), atol=1e-9)           # 75 - 30
    assert vec2[6] > 0.0 and obs.state.shape == (2, LEFT.state_dim)
    assert ap.shape == (1, 3) and ar.shape == (1, 6) and np.allclose(ap[0], [0.5, 0.1, 0.9])
    assert obs.state_valid[:, 0:6].all()                              # never flagged UNMEASURED


def test_the_subset_planner_is_exactly_its_layouts_roles_wide():
    planner = _subset_planner(To=1)
    with pytest.raises(ValueError):
        planner.seed_anchor(np.zeros((N, 3)), np.tile(rotmat_to_rot6d(np.eye(3)), (N, 1)))
    planner.seed_anchor(np.zeros((1, 3)), np.tile(rotmat_to_rot6d(np.eye(3)), (1, 1)))
    with pytest.raises(ValueError):
        planner.commit_role("right", np.zeros(3), rotmat_to_rot6d(np.eye(3)))


def test_the_five_point_planner_offers_the_same_two_methods():
    planner = _planner(To=2)
    planner.push_gripper("left", 0.05)
    planner.push_proprio()
    names, vec, written = planner.state_readout()
    assert names == hl_5pt.state_names(LAY) and vec.shape == (LAY.state_dim,) and not written
    assert vec[-2] > 0.0 and vec[-1] == 0.0                           # left grip, right grip
    obs, _ap, _ar = planner.snapshot_obs()
    assert np.allclose(vec, obs.state[-1])


# ---------------------------------------------------------------------------
# the offline overlay: held roles give a one-hand checkpoint a whole body
# ---------------------------------------------------------------------------
def _fake_rollout(h: int = 4):
    role = SimpleNamespace(role="left", predicted_gripper_m=np.full(h, 0.05), target_gripper_m=None)
    result = SimpleNamespace(layout_mode="left_only", algorithm="diffusion", seed=3,
                             sample=SimpleNamespace(dataset_index=12), roles=(role,))
    wxyz = tuple((1.0, 0.0, 0.0, 0.0) for _ in range(h))
    poses = {"left": SimpleNamespace(predicted_position=np.linspace(0, 1, 3 * h).reshape(h, 3),
                                     predicted_wxyz=wxyz,
                                     target_position=np.zeros((h, 3)), target_wxyz=wxyz)}
    return result, poses


def test_the_offline_overlay_holds_the_missing_roles_for_both_bodies():
    from flore.s1_policy.inference import keypoint_humanoid as KH
    from flore.s1_policy.inference.keypoint_ik import IK_ROLES

    held = {r: (np.full(3, 1.0 + k), np.eye(3)) for k, r in enumerate(IK_ROLES) if r != "left"}
    kh = KH.KeypointHumanoid(server=None, enabled=False, held_poses=held)
    assert set(kh.held_poses) == set(held)
    # Pretend the IK loaded (the queue is what is under test, not yourdfpy).
    kh.enabled, kh.ik, kh.disabled_reason = True, object(), ""
    kh._ensure_worker = lambda: None
    result, poses = _fake_rollout(h=4)
    kh.on_rollout(result, poses, None, True)
    payload = kh._q.get_nowait()
    assert set(payload["poses"]) == set(IK_ROLES)
    for role in held:
        for series in KH.SERIES:
            pos, rot = payload["poses"][role][series]
            assert pos.shape == (4, 3) and rot.shape == (4, 3, 3)
            assert np.all(pos == held[role][0]) and np.all(np.diff(pos, axis=0) == 0)
    assert np.allclose(payload["poses"]["left"]["prediction"][0], poses["left"].predicted_position)
    assert kh.available


def test_the_offline_overlay_still_refuses_an_unfilled_role_by_name():
    from flore.s1_policy.inference import keypoint_humanoid as KH

    kh = KH.KeypointHumanoid(server=None, enabled=False)
    kh.enabled, kh.ik, kh.disabled_reason = True, object(), ""
    kh._ensure_worker = lambda: None
    result, poses = _fake_rollout(h=2)
    kh.on_rollout(result, poses, None, True)
    assert kh.ik is None and "right" in kh.disabled_reason and "held_poses" in kh.disabled_reason
    with pytest.raises(ValueError, match="held pose"):
        KH.KeypointHumanoid(server=None, enabled=False, held_poses={"right": (np.zeros(2), np.eye(3))})


# ---------------------------------------------------------------------------
# the page's wiring: the family is read off the layout, never a flag
# ---------------------------------------------------------------------------
_PAGE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "scripts", "hl_serving_5pt.py")


def _page_src() -> str:
    with open(_PAGE) as fh:
        return fh.read()


def test_the_page_builds_the_planner_and_the_held_slots_from_the_layout():
    src = _page_src()
    assert "hl_5pt.build_planner(loaded" in src
    assert "hl_5pt.held_anchor_poses(anchor, loaded.layout)" in src
    assert "RP.resolve_deploy_anchor(hl_5pt.WIRE_ROLES" in src        # all five, whatever the layout
    assert "held=held_poses" in src
    assert "planner.push_proprio()" in src and "push_proprio_geom()" not in src
    assert "planner.state_readout()" in src and "geom_state()" not in src
    assert "hl_5pt.force_command(wrench_rows, gate_n)" in src
    assert "cmd_from_meas_force" not in src                           # one implementation
    assert "wire.all_rows(chunk)" in src                               # the IK sees five
    assert "wire.decoded_positions(chunk)" in src                      # the span sees five
    assert "hl_5pt.FivePointPlanner(loaded" not in src                # never hardcoded


def test_the_page_trims_the_camera_list_to_the_checkpoint_rather_than_refusing():
    src = _page_src()
    assert "if len(cams) > loaded.num_cameras:" in src
    assert "cams = cams[:loaded.num_cameras]" in src
    assert "Pass --camera-right-ws '' for a" not in src


def test_the_page_parser_accepts_a_left_only_checkpoint_in_its_help():
    import importlib.util
    import sys

    spec = importlib.util.spec_from_file_location("_hl5_subset_wiring", _PAGE)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_hl5_subset_wiring"] = mod
    spec.loader.exec_module(mod)
    parser = mod.build_parser()
    help_text = parser.format_help()
    assert "left_only" in help_text and "HELD at the ready pose" in help_text


# ---------------------------------------------------------------------------
# STOP is the operator's interrupt, not the process's shutdown — and Reset is refused while ARMED.
# Both found on the rig 2026-09-07 as "the first run is fine, everything after a STOP is wrong".
# ---------------------------------------------------------------------------
_SCRIPT = os.path.join(os.path.dirname(__file__), "..", "scripts", "hl_serving_5pt.py")


def _script_src() -> str:
    with open(_SCRIPT, encoding="utf-8") as fh:
        return fh.read()


def _nested_def(src: str, header: str) -> str:
    """One nested ``def`` of ``_run_viser``: its header up to the next non-blank, non-comment line
    at the same indent (the next def / decorator / statement)."""
    i = src.index(header)
    indent = len(header) - len(header.lstrip())
    lines = src[i:].splitlines(keepends=True)
    out = [lines[0]]
    for ln in lines[1:]:
        s = ln.lstrip()
        if s and not s.startswith("#") and len(ln) - len(s) <= indent:
            break
        out.append(ln)
    return "".join(out)


def test_stop_is_not_shutdown_so_the_daemons_outlive_a_stop():
    """As ONE Event, STOP ended the heartbeat, the ingest and the IK worker for the rest of the
    process, and every later Predict sampled from the observation frozen at the press (the stream's
    sampler never checks freshness, so silently). The three daemon loops exit on ``shutdown`` only,
    and ``shutdown`` is set exactly once, on the way out — never by a button."""
    src = _script_src()
    assert "shutdown = threading.Event()" in src
    for header in ("    def _heartbeat() -> None:", "    def _ingest_loop() -> None:",
                   "    def _ik_worker() -> None:"):
        body = _nested_def(src, header)
        assert "while not shutdown.is_set():" in body, header
        assert "stop.is_set()" not in body, header
    assert src.count("shutdown.set()") == 1
    assert src.index("shutdown.set()") > src.index("except KeyboardInterrupt:")


def test_stop_sets_only_the_operator_event_and_drops_the_force_command():
    """The STOP handler interrupts (``stop``), retires the run and leaves the surviving heartbeat a
    PURE-POSITION frame to re-send. It never touches ``shutdown``."""
    src = _script_src()
    # The body lives in `_stop_now()`; the button AND the SPACE hotkey (tty_keys) both call it.
    i = src.index("    def _stop_now() -> str:")
    handler = src[i:src.index("\n    @b_auto.on_click", i)]
    assert "stop.set()" in handler
    button = src[src.index("    @b_stop.on_click"):]
    assert "_stop_now()" in button[:120]
    assert "shutdown.set()" not in handler   # the comment may NAME it; nothing may SET it
    assert 'last_wire["fcmd"] = None' in handler


def test_reset_is_refused_whole_while_the_low_level_is_armed():
    """The armed check comes BEFORE anything is reset, and the refusal names the protocol that
    works: `start` (disarm — the hands return to the ready pose) → Reset → `start` → ▶▶, or ▶▶
    alone to continue from the stopped pose."""
    src = _script_src()
    i = src.index('if flags["reset"]:')
    branch = src[i:src.index('if flags["predict"]:', i)]
    k_armed = branch.index("rframe.vive_armed")
    assert k_armed < branch.index("wire.clutch.reset()")
    assert k_armed < branch.index("planner.seed_anchor(")
    assert k_armed < branch.index("_set_wire(wire.idle(), None)")
    assert "RESET_REFUSED_ARMED" in branch
    j = src.index("RESET_REFUSED_ARMED = (")
    msg = src[j:src.index(")\n", j)]
    for word in ("ARMED", "`start`", "DISARM", "Reset", "▶▶", "no Reset needed"):
        assert word in msg, word


def test_a_reset_clutch_pins_step_zero_on_the_ready_frame_whatever_measured_says():
    """The mechanism behind the refusal. After ``reset()`` nothing is carried, ``sync`` answers READY
    without reading ``measured``, and ``apply`` pins step 0 on the ready frame — 316 mm from a hand
    that stopped 316 mm away. Without the reset the same chunk continues from where it stopped."""
    post = hl_dense.PostAnchorConfig(mode="command", eps_m=0.05, eps_rad=np.deg2rad(10.0))
    home_pos = np.array([[0.44, 0.15, 0.10], [0.44, -0.15, 0.10], [0.0, 0.0, 0.40],
                         [0.0, 0.12, -0.72], [0.0, -0.12, -0.72]]).reshape(-1)
    home_orn = np.tile([0.0, 0.0, 0.0, 1.0], 5)
    orn = np.tile(home_orn, (17, 1))
    walk = np.tile(home_pos, (17, 1))          # LEFT walks 30 cm forward, 10 cm right
    walk[:, 0] += np.linspace(0.0, 0.30, 17)
    walk[:, 1] -= np.linspace(0.0, 0.10, 17)
    nxt = np.tile(home_pos, (17, 1))           # the next chunk, decoded against a READY anchor
    nxt[:, 0] += np.linspace(0.0, 0.05, 17)

    def run_then_stop():
        c = hl_dense.KeypointClutch(home_pos, home_orn, post=post, n_points=5, n_carried=5)
        c.sync(None)
        p, q = c.apply(walk, orn)
        c.commit((p[-1], q[-1]))               # STOP after the executed steps
        return c, (p[-1].copy(), q[-1].copy())  # the robot arrived: measured == wire

    c, stopped = run_then_stop()
    c.reset()                                  # what the old armed Reset did
    assert c.sync(stopped).startswith("READY frame")
    p, _ = c.apply(nxt, orn)
    jump = np.linalg.norm(p[0].reshape(5, 3)[0] - stopped[0].reshape(5, 3)[0])
    assert jump == pytest.approx(np.hypot(0.30, 0.10), abs=1e-9)   # 316 mm back toward home

    c2, stopped2 = run_then_stop()             # STOP -> ▶▶ with no Reset
    assert c2.sync(stopped2).startswith("CARRIED")
    p2, _ = c2.apply(nxt, orn)
    assert np.linalg.norm(p2[0].reshape(5, 3)[0] - stopped2[0].reshape(5, 3)[0]) < 1e-9


# ---------------------------------------------------------------------------------------------------
# Reach compensation on the HANDS (--reach-ki): hl_5pt.HandReach over hl_reach.ReachCompensator.
# The arithmetic is tests/test_hl_reach.py's; what is pinned here is the five-point CONTRACT — hands
# only, published never committed, identity while un-armed, a silent Bota is contact — and the page
# wiring, source-level, for the same reason the STOP / Reset tests above are.
# ---------------------------------------------------------------------------------------------------
from flore.s1_policy.serving.hl_reach import ReachConfig  # noqa: E402


def _frame5(pos_by_role=None):
    """A 5pt wire frame ``(pos (15,), orn (20,))`` on a ready-like stance, the named roles moved."""
    pos = np.array([[0.44, 0.15, 0.10], [0.44, -0.15, 0.10], [0.0, 0.0, 0.40],
                    [0.0, 0.12, -0.72], [0.0, -0.12, -0.72]])
    for role, p in (pos_by_role or {}).items():
        pos[hl_5pt.WIRE_ROLES.index(role)] = p
    return pos.reshape(-1), np.tile([0.0, 0.0, 0.0, 1.0], 5)


def _stall(reach, n, step=0.01, armed=True, force=None):
    """``n`` published steps: the LEFT command walks +x by ``step`` per step, the robot never moves."""
    meas = _frame5()
    outs = []
    for k in range(n):
        cmd = _frame5({"left": [0.44 + step * (k + 1), 0.15, 0.10]})
        outs.append(reach.apply(cmd, measured=meas, armed=armed,
                                force_by_hand={"left": 0.0, "right": 0.0} if force is None else force))
    return outs


def test_hand_reach_off_or_paused_or_unarmed_returns_the_same_frame_object():
    """The byte-identical claim: with --reach-ki 0, with the page toggle off, and while the low level
    is un-armed, the publish path hands `_send` the very object it had before the feature."""
    frame = _frame5()
    off = hl_5pt.HandReach(ReachConfig(), ("left", "right"), force_roles=("left", "right"))
    assert not off.enabled
    assert off.apply(frame, measured=frame, armed=True, force_by_hand={}) is frame
    on = hl_5pt.HandReach(ReachConfig(ki=0.05), ("left", "right"), force_roles=("left", "right"))
    assert on.enabled and on.live and on.roles == ("left", "right")
    on.active = False
    assert not on.live
    assert on.apply(frame, measured=frame, armed=True, force_by_hand={}) is frame
    on.active = True
    assert on.apply(frame, measured=frame, armed=False, force_by_hand={}) is frame
    assert on.n_unarmed_resets == 1
    # ki > 0 but no hand predicted: nothing to compensate, and it says so rather than pretending
    none = hl_5pt.HandReach(ReachConfig(ki=0.05), ("head",), force_roles=())
    assert not none.enabled and none.roles == ()


def test_hand_reach_compensates_only_the_predicted_hands_and_nothing_else():
    """A left_only checkpoint: RIGHT is HELD, so it gets no compensator; torso and feet never do."""
    cfg = ReachConfig(ki=0.05, window=3)
    reach = hl_5pt.HandReach(cfg, ("left",), force_roles=("left",))
    assert reach.roles == ("left",) and "right" not in reach.roles
    outs = _stall(reach, 12)
    last_cmd = _frame5({"left": [0.44 + 0.12, 0.15, 0.10]})
    pub_pos = outs[-1][0].reshape(5, 3)
    cmd_pos = last_cmd[0].reshape(5, 3)
    assert reach.residual_mm()["left"] > 1.0                          # it accumulated
    assert pub_pos[0][0] > cmd_pos[0][0]                              # LEFT asks for MORE, along +x
    assert abs(pub_pos[0][1] - cmd_pos[0][1]) < 1e-12 and abs(pub_pos[0][2] - cmd_pos[0][2]) < 1e-12
    assert np.linalg.norm(pub_pos[0] - cmd_pos[0]) <= cfg.clamp_m + 1e-12
    assert np.array_equal(pub_pos[1:], cmd_pos[1:])                  # RIGHT, torso, both feet: identical
    assert np.array_equal(outs[-1][1], last_cmd[1])                   # orientation: untouched...
    f = _frame5({"left": [0.57, 0.15, 0.10]})
    assert reach.apply(f, measured=_frame5(), armed=True, force_by_hand={"left": 0.0})[1] is f[1]  # ...same object


def test_hand_reach_saturates_at_the_clamp_and_the_slots_follow_wire_order():
    cfg = ReachConfig(ki=0.5, window=2, clamp_m=0.02)
    reach = hl_5pt.HandReach(cfg, ("left", "right"), force_roles=("left", "right"))
    assert reach._slots == {"left": 0, "right": 3}
    _stall(reach, 40)
    assert reach.residual_mm()["left"] == pytest.approx(20.0, abs=1e-6)
    assert reach.residual_mm()["right"] == 0.0                       # the RIGHT never moved nor was asked to
    assert reach.comps["left"].saturated and not reach.comps["right"].saturated
    d = reach.describe()
    assert d.startswith("L ") and "SATURATED" in d and " · R 0.0 mm" in d


def test_hand_reach_treats_a_silent_bota_as_contact_and_names_an_unprotected_hand():
    # The rig HAS a left Bota but the bridge did not answer: CONTACT (hold + decay), never growth.
    reach = hl_5pt.HandReach(ReachConfig(ki=0.05, window=2), ("left",), force_roles=("left",))
    _stall(reach, 10, force={})
    assert reach.residual_mm()["left"] == 0.0
    assert reach.comps["left"].stats()["n_contact_blocked"] == 10 and reach.unprotected == ()
    # A real contact reading at the gate blocks the same way.
    _stall(reach, 10, force={"left": 1.5})
    assert reach.residual_mm()["left"] == 0.0
    # A hand with NO Bota on the layout follows the module's None contract (accumulates) and is NAMED.
    free = hl_5pt.HandReach(ReachConfig(ki=0.05, window=2), ("left",), force_roles=())
    assert free.unprotected == ("left",)
    _stall(free, 10, force={})
    assert free.residual_mm()["left"] > 0.0


def test_hand_reach_holds_on_a_stale_readback_and_reset_drops_the_residual():
    reach = hl_5pt.HandReach(ReachConfig(ki=0.05, window=2), ("left",), force_roles=("left",))
    _stall(reach, 10)
    r_mm = reach.residual_mm()["left"]
    assert r_mm > 0.0
    cmd = _frame5({"left": [0.60, 0.15, 0.10]})
    held = reach.apply(cmd, measured=None, armed=None, force_by_hand={})   # no fresh RecordTap: HOLD
    assert reach.residual_mm()["left"] == pytest.approx(r_mm)
    assert np.linalg.norm(held[0][:3] - cmd[0][:3]) == pytest.approx(r_mm * 1e-3)
    reach.reset()
    assert reach.residual_mm() == {"left": 0.0}
    assert reach.apply(cmd, measured=None, armed=None, force_by_hand={})[0][:3].tolist() == cmd[0][:3].tolist()


def test_hand_reach_refuses_a_frame_or_measurement_of_the_wrong_width():
    reach = hl_5pt.HandReach(ReachConfig(ki=0.05), ("left",), force_roles=("left",))
    with pytest.raises(ValueError):
        reach.apply((np.zeros(9), np.zeros(12)), measured=None, armed=True, force_by_hand={})
    with pytest.raises(ValueError):
        reach.apply(_frame5(), measured=(np.zeros(9), np.zeros(12)), armed=True, force_by_hand={})


def test_measured_keypoints_is_the_clutch_measurement_and_never_raises():
    readout = hl_dense.KeypointReadout(np.array([0.0, 0.0, 0.7567]))
    pos, orn = _frame5()

    def boom(_rframe):
        raise RuntimeError("bad readback")

    w = hl_5pt.FivePointDirectWire(readout, pos, orn, measured_frame=boom)
    assert w.measured_keypoints(object()) is None and w.measured_keypoints(None) is None
    w2 = hl_5pt.FivePointDirectWire(readout, pos, orn, measured_frame=lambda _rf: (pos + 1.0, orn))
    assert np.array_equal(w2.measured_keypoints(object())[0], pos + 1.0)
    w3 = hl_5pt.FivePointDirectWire(readout, pos, orn)
    assert w3.measured_keypoints(object()) is None
    assert w3.measured(object()) is None                             # the idle hold is still never chased


def test_reach_is_applied_at_both_publish_seams_and_never_reaches_the_clutch_base():
    src = _script_src()
    drip = _nested_def(src, "    def _drip(out) -> int:")
    assert "pub = _reach_apply(frame)" in drip and "_set_wire(pub, fcmd)" in drip
    assert "_send(pub, _gate_force(rows))" in drip and "_send(frame" not in drip
    assert "wire.commit(wire.frame(plan, last))" in drip and "wire.commit(pub" not in drip
    stream = _nested_def(src, "    def _run_streaming() -> None:")
    assert "pub = _reach_apply(frame)" in stream and "_set_wire(pub, fcmd)" in stream
    assert "sent = _send(pub, _gate_force(rows))" in stream
    assert "wire.commit(frame)" in stream and "wire.commit(pub" not in src
    body = src[src.index("def _run_viser("):]
    assert body.count("= _reach_apply(frame)") == 2                  # the two publish seams, no more
    for header in ("    def _heartbeat() -> None:", "    def _predict() -> str:"):
        assert "_reach_apply" not in _nested_def(src, header), header
    walk = src[src.index("def _run_walk()"):src.index("@b_walk.on_click")]
    assert "_reach_apply" not in walk                                # the walk HOLDS the hands as published


def test_reach_residual_is_dropped_on_every_abandoned_intent():
    src = _script_src()
    i = src.index("def _stop_now() -> str:")   # the STOP body (button + SPACE hotkey)
    assert '_reach_reset("stop")' in src[i:src.index("@b_auto.on_click", i)]
    j = src.index('if flags["reset"]:')
    reset = src[j:src.index("_set_wire(wire.idle(), None)", j)]
    assert '_reach_reset("reset to ready pose")' in reset
    assert reset.index("if armed:") < reset.index('_reach_reset("reset to ready pose")')   # only un-armed
    walk = src[src.index("def _run_walk()"):src.index("@b_walk.on_click")]
    k = walk.index('_reach_reset("walk start")')
    assert walk.index('base = tuple(a.copy() for a in last_wire["frame"])') < k < walk.index("_send(frame")
    t = src.index("@c_reach.on_update")
    toggle = src[t:src.index("@c_ghost.on_update", t)]
    assert "reach.active = bool(c_reach.value)" in toggle
    assert '_reach_reset("compensation toggled off")' in toggle
    # The panel sits ABOVE the walk block (the walk block is pinned to hold ONE folder) and the page
    # loop shows the residual; a bad config is refused at launch by value, not at the first Run.
    assert 'add_folder("Reach compensation (hands)")' in src
    assert (src.index('add_folder("Reach compensation (hands)")')
            < src.index("# ---------------- the WALK SKILL (extension) ----------------"))
    assert "g_reach.value = reach.describe()" in src
    assert 'raise SystemExit(f"--reach-*: {exc}")' in src


def test_reach_flags_default_off_and_mirror_run_hl_serving():
    import importlib.util
    import re
    import sys

    spec = importlib.util.spec_from_file_location("_hl5_reach_wiring", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_hl5_reach_wiring"] = mod
    spec.loader.exec_module(mod)
    a = mod.build_parser().parse_args(["--checkpoint", "x"])
    defaults = {"reach_ki": 0.0, "reach_clamp": 0.03, "reach_stall_frac": 0.5, "reach_window": 9,
                "reach_gate": 0.0, "reach_decay": 0.995, "reach_contact_n": 1.5}
    for name, val in defaults.items():
        assert getattr(a, name) == val, name
    other = open(os.path.join(os.path.dirname(__file__), "..", "scripts", "hl_serving.py"),
                 encoding="utf-8").read()
    for name, val in defaults.items():
        flag = "--" + name.replace("_", "-")
        m = re.search(rf'add_argument\("{re.escape(flag)}", type=(?:float|int), default=([0-9.]+)', other)
        assert m and float(m.group(1)) == float(val), flag                # one contract, two launchers
    a2 = mod.build_parser().parse_args(["--checkpoint", "x", "--reach-ki", "0.05", "--reach-clamp", "0.02"])
    assert a2.reach_ki == 0.05 and a2.reach_clamp == 0.02


# ---------------------------------------------------------------------------
# the observation's RANGE against the checkpoint's own norm stats (task-agnostic, 2026-09-12)
# ---------------------------------------------------------------------------
def _stats(q01, q99):
    from flore.s1_policy.data.norm_stats import NormStats

    return NormStats(q01=np.asarray(q01, dtype=np.float64), q99=np.asarray(q99, dtype=np.float64))


def test_state_range_report_names_the_columns_outside_the_band_and_how_far():
    names = ("lr_dx", "lr_dy", "left_grip")
    rep = hl_5pt.state_range_report(names, [0.15, 0.30, 0.5],
                                    _stats([-0.10, 0.20, 0.0], [0.10, 0.40, 1.0]))
    assert rep.checked and not rep.ok and rep.n_columns == 3
    assert [o[0] for o in rep.outside] == ["lr_dx"]
    name, value, lo, hi, excess = rep.outside[0]
    assert (value, lo, hi) == (0.15, -0.10, 0.10)
    assert excess == pytest.approx(0.05 / 0.20)          # a quarter of the band past q99
    assert "lr_dx" in rep.describe() and "OUTSIDE" in rep.describe()


def test_state_range_report_is_clean_inside_the_band_and_at_its_edges():
    names = ("lr_dx", "lr_dy")
    rep = hl_5pt.state_range_report(names, [-0.10, 0.40], _stats([-0.10, 0.20], [0.10, 0.40]))
    assert rep.ok and rep.outside == () and rep.wrapping == ()
    assert "inside" in rep.describe()


def test_state_range_report_tolerates_a_value_a_hair_past_the_edge():
    # The live LEFT jaw at its open stop read 0.9912 against a training q99 of 0.9910 (the same
    # stop, float precision) and the first rig run logged an "OUTSIDE the training range" warning
    # on every Predict. 1% of the band is inside; 2% is not; edge_tol=0 restores the strict check.
    stats = _stats([0.289], [0.991])
    assert hl_5pt.state_range_report(("left_grip",), [0.9912], stats).ok
    assert hl_5pt.state_range_report(("left_grip",), [0.991 + 0.02 * 0.702], stats).outside[0][0] == "left_grip"
    assert not hl_5pt.state_range_report(("left_grip",), [0.9912], stats, edge_tol=0.0).ok


def test_state_range_report_uses_the_normalizers_floor_on_a_degenerate_band():
    # A column that never moved in training (q01 == q99) is scaled by the 0.02 floor in
    # normalize_quantile, so a 0.01 excursion is half a band, not infinitely many.
    rep = hl_5pt.state_range_report(("left_grip",), [0.51], _stats([0.5], [0.5]))
    assert rep.outside[0][4] == pytest.approx(0.5)


def test_state_range_report_flags_an_euler_column_near_the_wrap_even_without_stats():
    rep = hl_5pt.state_range_report(("left_yaw", "lr_dx"), [np.radians(171.0), 0.0], None)
    assert not rep.checked and rep.outside == ()
    assert rep.wrapping == (("left_yaw", pytest.approx(171.0)),)
    assert "unchecked" in rep.describe()
    rep2 = hl_5pt.state_range_report(("left_yaw",), [np.radians(171.0)], _stats([-3.2], [3.2]))
    assert rep2.checked and not rep2.ok and "wrap" in rep2.describe()


def test_state_range_report_refuses_a_width_mismatch_by_name():
    with pytest.raises(ValueError, match="not this checkpoint"):
        hl_5pt.state_range_report(("a", "b"), [0.0, 0.0], _stats([0.0], [1.0]))
    with pytest.raises(ValueError, match="column names"):
        hl_5pt.state_range_report(("a",), [0.0, 0.0], None)


def test_state_range_report_matches_the_planners_own_readout_shape():
    """The page hands it `planner.state_readout()`'s names + row and `loaded.norm_stats['state']`;
    the widths must agree for a fivepoint layout, which is the contract this pins."""
    names = hl_5pt.state_names(LAY)
    assert len(names) == LAY.state_dim
    rep = hl_5pt.state_range_report(names, np.zeros(LAY.state_dim),
                                    _stats(-np.ones(LAY.state_dim), np.ones(LAY.state_dim)))
    assert rep.ok and rep.n_columns == LAY.state_dim


def test_state_range_report_ignore_leaves_the_startup_jaws_out_of_the_band_check():
    names = ("lr_dx", "left_grip", "right_grip")
    stats = _stats([-0.1, 0.3, 0.3], [0.1, 1.0, 1.0])
    # Before a jaw bridge reports, the grips read 0.0 — outside a training band of open jaws.
    assert [o[0] for o in hl_5pt.state_range_report(names, [0.0, 0.0, 0.0], stats).outside] == [
        "left_grip", "right_grip"]
    rep = hl_5pt.state_range_report(names, [0.0, 0.0, 0.0], stats, ignore=("left_grip", "right_grip"))
    assert rep.ok and rep.n_columns == 1
    # `ignore` never hides a geometry column.
    rep2 = hl_5pt.state_range_report(names, [0.5, 0.0, 0.0], stats, ignore=("left_grip", "right_grip"))
    assert [o[0] for o in rep2.outside] == ["lr_dx"]


# ---------------------------------------------------------------------------
# --ood-stop: ending a ▶▶ run on the OBSERVATION rather than on the wire (2026-09-13)
# ---------------------------------------------------------------------------
def _page_module():
    """Import scripts/hl_serving_5pt.py for its parser and module constants (no viser, no model)."""
    import importlib.util
    import sys

    spec = importlib.util.spec_from_file_location("_hl5_ood_wiring", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_hl5_ood_wiring"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_worst_body_excursion_reports_the_worst_geometry_column():
    names = ("lr_dx", "lh_dx", "left_grip")
    stats = _stats([-0.10, 0.37, 0.0], [0.10, 0.574, 1.0])
    # lh_dx 0.705 is (0.705 - 0.574) / (0.574 - 0.370) = 0.64 band widths past q99 — the runaway
    # the two rig logs show; lr_dx is a smaller excursion in the same row.
    rep = hl_5pt.state_range_report(names, [0.15, 0.705, 0.5], stats)
    assert {o[0] for o in rep.outside} == {"lr_dx", "lh_dx"}
    col, bands = rep.worst_body_excursion()
    assert col == "lh_dx" and 0.6 < bands < 0.7


def test_worst_body_excursion_is_zero_when_only_the_JAWS_are_out_of_band():
    # A jaw resting at its stop can sit outside a narrower training band on a healthy run, and a
    # jaw is not the channel a diverging BODY appears in. It must not end a rollout.
    names = ("lr_dx", "left_grip", "right_grip")
    stats = _stats([-0.1, 0.3, 0.3], [0.1, 1.0, 1.0])
    rep = hl_5pt.state_range_report(names, [0.0, 0.0, 0.0], stats)
    assert not rep.ok and [o[0] for o in rep.outside] == ["left_grip", "right_grip"]
    assert rep.worst_body_excursion() == ("", 0.0)


def test_worst_body_excursion_is_zero_with_no_statistics_and_inside_the_band():
    assert hl_5pt.state_range_report(("lr_dx",), [99.0], None).worst_body_excursion() == ("", 0.0)
    rep = hl_5pt.state_range_report(("lr_dx",), [0.0], _stats([-0.1], [0.1]))
    assert rep.ok and rep.worst_body_excursion() == ("", 0.0)


def test_state_names_and_the_stop_decision_agree_on_the_jaw_suffix():
    # The stop decision reads back the suffix `state_names` writes; a drift would silently make
    # every jaw column gate a rollout (or every geometry column stop gating one).
    names = hl_5pt.state_names(LAY)
    jaws = [n for n in names if n.endswith(hl_5pt.GRIP_COLUMN_SUFFIX)]
    assert jaws == [f"{r}{hl_5pt.GRIP_COLUMN_SUFFIX}" for r in LAY.gripper_roles]
    assert len(jaws) == len(LAY.gripper_roles) >= 1


def test_the_gate_is_a_backstop_and_the_LEASH_is_what_is_on_by_default():
    """The primary answer to a diverging belief is the leash, not ending the run.

    Pinned as a pair because the two defaults only make sense together: `--obs-leak-m` PREVENTS the
    divergence (so it is on), `--ood-stop` ends the run on the symptom (so it is off).
    """
    mod = _page_module()
    d = mod.build_parser().parse_args(["--checkpoint", "x"])
    assert d.ood_stop == 0.0                                     # OFF: a symptom, not the cause
    assert d.obs_leak_m == mod.OBS_LEAK_M_DEFAULT == 0.05        # ON: the cause
    assert d.obs_leak_deg == mod.OBS_LEAK_DEG_DEFAULT == 10.0
    # The leash's defaults are deliberately the clutch's own numbers — the wire had this bound and
    # the observation did not, which is the whole bug.
    assert d.obs_leak_m == d.post_anchor_eps_m
    assert d.obs_leak_deg == d.post_anchor_eps_deg
    # ...but they are separate knobs, so a run can put the belief on a shorter leash than the wire.
    a = mod.build_parser().parse_args(["--checkpoint", "x", "--obs-leak-m", "0.02"])
    assert a.obs_leak_m == 0.02 and a.post_anchor_eps_m == 0.05
    # The backstop still arms, and both off-switches still reach zero.
    assert mod.build_parser().parse_args(["--checkpoint", "x", "--ood-stop", "0.25"]).ood_stop == 0.25
    for argv in (["--ood-stop", "0"], ["--no-ood-stop"]):
        assert mod.build_parser().parse_args(["--checkpoint", "x", *argv]).ood_stop == 0.0
    assert mod.build_parser().parse_args(
        ["--checkpoint", "x", "--ood-stop", "0.5", "--no-ood-stop"]).ood_stop == 0.0
    assert mod.build_parser().parse_args(["--checkpoint", "x", "--obs-leak-m", "0"]).obs_leak_m == 0.0


def test_the_ood_gate_runs_before_the_sample_on_BOTH_cadences_and_ends_the_run():
    """One gate, both cadences, and the refusal has to precede ``infer_chunk``.

    Checking the ORDER is the point: a gate that ran after the sample would already have published
    nothing but would have burned the seed and, on the stream, added the chunk to the blend.
    """
    src = open(_SCRIPT, encoding="utf-8").read()
    assert src.count("def _obs_range_gate()") == 1

    # Sequential: the gate is inside _predict, above the sampler, and its refusal string reaches
    # the caller — which is what makes `staged["out"]` stay None and end the rollout.
    pred = src[src.index("    def _predict()"):src.index("    # ---------------- execute ----------------")]
    assert pred.index("_obs_range_gate()") < pred.index("planner.infer_chunk")
    assert "refuse = _obs_range_gate()" in pred and "return refuse" in pred

    # The rollout's existing "nothing staged -> end the run" branch is what a refusal lands in.
    loop = src[src.index("    def _run_to_completion()"):src.index("    def _run_streaming()")]
    assert "note = _predict()" in loop
    branch = loop[loop.index("if out is None:"):]
    assert "break" in branch[:branch.index("flags[")]      # the branch ENDS the rollout

    # Stream: same gate, in _sample_once, above sample_stamped; and its refusal retires the
    # generation instead of being ridden out like a failed sample.
    once = src[src.index("        def _sample_once()"):src.index("        # WARM UP SYNCHRONOUSLY")]
    assert once.index("_obs_range_gate()") < once.index("hl_stream.sample_stamped")
    assert 'return None, 0, refuse' in once
    sampler = src[src.index("        def _sampler()"):src.index("        threading.Thread(target=_sampler")]
    assert 'why.startswith("REFUSED:")' in sampler
    assert sampler.index('why.startswith("REFUSED:")') < sampler.index("the blend keeps running")
    assert 'auto_ctl["running"] = False' in sampler

    # It never publishes, commits or re-bases on the way out: holding the last executed step IS
    # the behaviour, so the gate's own body may only read and log.
    gate = src[src.index("    def _obs_range_gate()"):
                   src.index("    # ---------------- predict ----------------")]
    for forbidden in ("_send(", "wire.commit", "wire.sync", "wire.publish", "planner.commit_role",
                      "_set_wire", "stop.set("):
        assert forbidden not in gate, forbidden


def _measured_frame_at(pos):
    """A ``(pos, orn)`` wire frame at ``pos`` (n,3) with identity rotations — what RecordTap's FK
    gives the clutch and now the belief leash."""
    from flore.utils.rot6d import rotmat_to_quat_xyzw

    p = np.asarray(pos, dtype=np.float64).reshape(-1, 3)
    q = np.tile(rotmat_to_quat_xyzw(np.eye(3)), (p.shape[0], 1))
    return p.reshape(-1).copy(), q.reshape(-1).copy()


def test_the_belief_leash_is_a_NO_OP_inside_the_bound():
    """A healthy run must be byte-identical: ``action[:, 0] == state`` still holds exactly."""
    pl = _planner(To=1)
    pos = pl._anchor_pos.copy()
    rot = pl._anchor_rot6d.copy()
    before = pl.state_readout()[1].copy()
    # Measured 1 cm away, leash 5 cm -> nothing to correct.
    lead, clamped = pl.leak_anchor_toward(_measured_frame_at(pos + 0.01),
                                          eps_m=0.05, eps_rad=np.radians(10.0))
    assert not clamped and 0.017 < lead < 0.018          # |(0.01,0.01,0.01)| = 0.0173
    assert np.array_equal(pl._anchor_pos, pos) and np.array_equal(pl._anchor_rot6d, rot)
    assert np.array_equal(pl.state_readout()[1], before)


def test_the_belief_leash_pulls_a_RUNAWAY_anchor_back_to_exactly_the_bound():
    """The correction is the clutch's own arithmetic: move along the straight line to measured
    until the lead is ``eps_m``. Not to measured, and not by a fixed step."""
    pl = _planner(To=1)
    n = len(pl.roles)
    measured = np.zeros((n, 3))
    runaway = np.zeros((n, 3))
    runaway[:, 0] = 0.30                                  # 30 cm ahead of the robot on x
    pl.seed_anchor(runaway, np.tile(rotmat_to_rot6d(np.eye(3)), (n, 1)))
    lead, clamped = pl.leak_anchor_toward(_measured_frame_at(measured),
                                          eps_m=0.05, eps_rad=np.radians(10.0))
    assert clamped and abs(lead - 0.30) < 1e-12
    got = np.linalg.norm(pl._anchor_pos.reshape(n, 3) - measured, axis=1)
    assert np.allclose(got, 0.05)                         # every point ON the bound
    # ...and the direction is preserved, so a corrected belief is still the same reach, shortened.
    assert np.allclose(pl._anchor_pos.reshape(n, 3)[:, 0], 0.05)
    assert np.allclose(pl._anchor_pos.reshape(n, 3)[:, 1:], 0.0)


def test_the_belief_leash_reaches_the_OBSERVATION_not_just_the_anchor():
    """The state is read off the RINGS. A correction that only moved the anchor would take effect at
    the next 30 Hz push instead of at this Predict — a one-frame race that reads as "sometimes it
    does nothing"."""
    pl = _planner(To=2)
    n = len(pl.roles)
    idx = {r: i for i, r in enumerate(pl.roles)}
    measured = np.zeros((n, 3))
    # ⚠️ Only the LEFT hand runs away. `rel_xy3` is translation-invariant, so shifting every point
    # by the same amount changes no state column at all — the observation only ever moves through
    # per-point DIFFERENCES, which is exactly the divergence that matters (a hand the arm cannot
    # reach while the chest stays put).
    runaway = measured.copy()
    runaway[idx["left"], 0] = 0.40
    pl.seed_anchor(runaway, np.tile(rotmat_to_rot6d(np.eye(3)), (n, 1)))
    pl.push_proprio_geom()
    before = pl.snapshot_obs()[0].state.copy()
    pl.leak_anchor_toward(_measured_frame_at(measured), eps_m=0.05, eps_rad=np.radians(10.0))
    after = pl.snapshot_obs()[0].state.copy()
    assert not np.allclose(before, after)                  # the OBSERVATION moved, immediately
    # EVERY history frame, not only t0: a To>1 checkpoint would otherwise read a corrected present
    # against an uncorrected past and see motion that never happened.
    assert np.allclose(after[0], after[-1])


def test_a_uniform_shift_of_the_belief_moves_no_state_column():
    """The property the leash's effect on the OBSERVATION rests on, stated as its own claim.

    ``rel_xy3`` is three pairwise differences among the tracked points, so the whole state is
    invariant to a rigid TRANSLATION of the anchor. That is why the leash matters through per-point
    differences and not through absolute position — and why a runaway shows up as ``lh_dx`` (hand
    minus chest) rather than as "the hand is far from the pelvis".
    """
    pl = _planner(To=1)
    n = len(pl.roles)
    base = pl.state_readout()[1].copy()
    shifted = pl._anchor_pos.reshape(n, 3) + np.array([0.7, -0.3, 0.2])
    pl.seed_anchor(shifted, pl._anchor_rot6d.reshape(n, 6))
    assert np.allclose(pl.state_readout()[1], base)


def test_the_belief_leash_HOLDS_rather_than_guessing_with_no_recordtap():
    """``sync``'s own rule: no measurement is "unknown", never a correction toward a guess."""
    pl = _planner(To=1)
    pos = pl._anchor_pos.copy()
    assert pl.leak_anchor_toward(None, eps_m=0.05, eps_rad=np.radians(10.0)) == (0.0, False)
    assert np.array_equal(pl._anchor_pos, pos)
    # A non-finite readback is the same answer.
    bad = _measured_frame_at(np.full((len(pl.roles), 3), np.nan))
    assert pl.leak_anchor_toward(bad, eps_m=0.05, eps_rad=np.radians(10.0)) == (0.0, False)
    assert np.array_equal(pl._anchor_pos, pos)


def test_the_belief_leash_would_have_held_the_measured_runaway_in_band():
    """The rig's own curve, replayed: the run folded because ``lh_dx`` reached 0.876 m while the
    robot's hand was where the 5 cm-bounded WIRE had put it. With the belief on the same leash the
    believed hand cannot be more than 5 cm past the measured one, so ``lh_dx`` cannot run away.
    """
    pl = _planner(To=1)
    n = len(pl.roles)
    idx = {r: i for i, r in enumerate(pl.roles)}
    # The robot's hand at 0.50 m in front of the chest; the belief at the logged 0.876.
    measured = np.zeros((n, 3))
    measured[idx["left"], 0] = 0.50
    believed = measured.copy()
    believed[idx["left"], 0] = 0.876
    pl.seed_anchor(believed, np.tile(rotmat_to_rot6d(np.eye(3)), (n, 1)))
    lead, clamped = pl.leak_anchor_toward(_measured_frame_at(measured),
                                          eps_m=0.05, eps_rad=np.radians(10.0))
    assert clamped and abs(lead - 0.376) < 1e-9
    lh_dx = pl._anchor_pos.reshape(n, 3)[idx["left"], 0]
    assert abs(lh_dx - 0.55) < 1e-9            # 0.50 measured + the 5 cm leash, not 0.876
    # push_gantry_new's own band for lh_dx is [0.370, 0.574] -> 0.55 is INSIDE it, 0.876 is 1.5
    # band widths past. The leash is what keeps the gate from ever having something to fire on.
    assert 0.370 <= lh_dx <= 0.574


def test_the_belief_bound_is_called_before_the_gate_and_the_sample_on_both_cadences():
    src = open(_SCRIPT, encoding="utf-8").read()
    assert src.count("def _belief_bound()") == 1
    pred = src[src.index("    def _predict()"):src.index("    # ---------------- execute ----------------")]
    assert pred.index("_belief_bound()") < pred.index("_obs_range_gate()") < pred.index("planner.infer_chunk")
    once = src[src.index("        def _sample_once()"):src.index("        # WARM UP SYNCHRONOUSLY")]
    assert once.index("_belief_bound()") < once.index("_obs_range_gate()") < once.index("sample_stamped")
    # It corrects the BELIEF and publishes nothing: the wire is the clutch's business.
    body = src[src.index("    def _belief_bound()"):
               src.index("    # ---------------- the observation's own provenance ----------------")]
    for forbidden in ("_send(", "wire.commit", "wire.publish", "_set_wire", "wire.sync"):
        assert forbidden not in body, forbidden
    assert "wire.measured_keypoints" in body       # the SAME readback the clutch bounds against


def test_the_ood_default_is_past_a_recovering_start_and_inside_a_runaway():
    """The threshold has to sit between the two measured curves, or it is not a detector.

    Both rig runs are monotone: 0.16 / 0.28 / 0.33 / 0.47 / 0.65 band widths over five chunks, and
    0.03 -> 1.15 over sixteen. Every shipped checkpoint's ready-pose observation reads 0.00.
    """
    mod = _page_module()
    limit = mod.OOD_STOP_BANDS_DEFAULT
    healthy_start = 0.0
    first_excursions = (0.03, 0.16)          # neither run is stopped on its FIRST warning
    runaway = (0.28, 0.33, 0.47, 0.65, 1.15)
    assert healthy_start <= limit
    assert all(e <= limit for e in first_excursions)
    assert all(e > limit for e in runaway)


# ---------------------------------------------------------------------------
# the belief's leash on the keypoint SUBSET family (2026-09-14), and the manual jaws
# ---------------------------------------------------------------------------
def _wire_frame(pos5, rotmats5):
    """A five-slot ``(pos, orn)`` wire frame — what ``FivePointWire.measured_keypoints`` hands the
    leash — with one rotation per WIRE slot."""
    from flore.utils.rot6d import rotmat_to_quat_xyzw

    p = np.asarray(pos5, dtype=np.float64).reshape(5, 3)
    q = np.stack([rotmat_to_quat_xyzw(np.asarray(r, dtype=np.float64)) for r in rotmats5])
    return p.reshape(-1).copy(), q.reshape(-1).copy()


def test_the_subset_planner_has_the_leash_and_reads_ITS_OWN_wire_slot():
    """`_belief_bound` gates on ``hasattr(planner, "leak_anchor_toward")`` — so a subset planner
    without the method was silently unprotected. It has it now, and it indexes the five-slot frame by
    the role's WIRE slot rather than by position 0."""
    pl = _subset_planner("left_only", To=1)
    assert hasattr(pl, "leak_anchor_toward")
    slot = hl_5pt.WIRE_ROLES.index("left")
    assert slot == 0 and len(hl_5pt.WIRE_ROLES) == 5
    pos5 = np.arange(15.0).reshape(5, 3)                  # a distinct measured point per slot
    pl.seed_anchor(pos5[[slot]] + np.array([[0.30, 0.0, 0.0]]), rotmat_to_rot6d(np.eye(3))[None])
    lead, clamped = pl.leak_anchor_toward(_wire_frame(pos5, [np.eye(3)] * 5),
                                          eps_m=0.05, eps_rad=np.radians(10.0))
    assert clamped and abs(lead - 0.30) < 1e-12
    assert np.allclose(pl._anchor_pos[0], pos5[slot] + np.array([0.05, 0.0, 0.0]))
    # held slots are the wire's: the planner is still exactly one role wide
    assert pl._anchor_pos.shape == (1, 3) and pl._anchor_rot6d.shape == (1, 6)
    # no readback / a non-finite one = HELD, never a guess
    assert pl.leak_anchor_toward(None, eps_m=0.05, eps_rad=0.1) == (0.0, False)
    bad = _wire_frame(np.full((5, 3), np.nan), [np.eye(3)] * 5)
    assert pl.leak_anchor_toward(bad, eps_m=0.05, eps_rad=0.1) == (0.0, False)


def test_the_leash_compares_the_anchor_and_the_readback_in_ONE_frame():
    """Measured on the mock rig with the robot ON the ready pose (``rotate_box_all``, 2026-09-18): the
    FIRST Predict logged ``belief re-based: … 757 mm ahead of the measured keypoints`` — exactly
    ``pelvis_z``. The page seeds the planner with the WORLD anchor (floor z = 0) and
    ``FivePointWire.measured_keypoints`` is the wire's PELVIS frame, so compared raw every role of
    BOTH families is ``pelvis_z`` apart before anything moves, and the leash drags the belief 0.7 m
    down. ``_belief_bound`` now converts the readback with ``hl_5pt.wire_frame_to_anchor``; the
    ready pose's own wire frame must come back as the anchor, and the planners must read ZERO lead
    through it. The raw gap is pinned too, so the bug cannot return silently."""
    from flore.s1_policy.inference import ready_pose as RP

    anchor, why = RP.resolve_deploy_anchor(hl_5pt.WIRE_ROLES)
    if anchor is None:
        pytest.skip(f"no deploy anchor on this box: {why}")
    readout = hl_dense.KeypointReadout(np.array([0.0, 0.0, float(anchor.pelvis_z)]))
    ready_pos, ready_orn = hl_dense.ready_pose_five_point(anchor, readout)
    back_pos, back_orn = hl_5pt.wire_frame_to_anchor((ready_pos, ready_orn), readout)
    for slot, role in enumerate(hl_5pt.WIRE_ROLES):
        assert np.allclose(back_pos.reshape(5, 3)[slot], anchor.pos[role], atol=1e-9), role
    assert np.array_equal(back_orn, ready_orn)          # identity pelvis rotation: untouched
    assert hl_5pt.wire_frame_to_anchor(None, readout) is None

    dual = get_layout("both_hands_head_dual")
    for pl, roles in ((_planner(To=1), LAY.roles), (_subset_planner("both_hands_head_dual", To=1), dual.roles)):
        pl.seed_anchor(np.stack([anchor.pos[r] for r in roles]),
                       np.stack([rotmat_to_rot6d(anchor.rotmat[r]) for r in roles]))
        # through the conversion: the robot IS on the anchor, so nothing to correct
        lead, clamped = pl.leak_anchor_toward((back_pos, back_orn), eps_m=0.05, eps_rad=np.radians(10.0))
        assert not clamped and lead < 1e-6, (type(pl).__name__, lead)
        # raw: the bug — pelvis_z of "lead" on a robot that has not moved (eps 10 m: measure, don't move)
        lead_raw, _ = pl.leak_anchor_toward((ready_pos, ready_orn), eps_m=10.0, eps_rad=np.radians(10.0))
        assert abs(lead_raw - float(anchor.pelvis_z)) < 1e-6, (type(pl).__name__, lead_raw)

    # a rotated readout is refused, never half-converted
    c, s_ = np.cos(0.3), np.sin(0.3)
    turned = hl_dense.KeypointReadout(np.zeros(3), np.array([[c, -s_, 0.0], [s_, c, 0.0], [0.0, 0.0, 1.0]]))
    with pytest.raises(ValueError, match="identity"):
        hl_5pt.wire_frame_to_anchor((ready_pos, ready_orn), turned)


def test_the_page_converts_the_readback_before_the_leash_and_proves_the_frame_at_startup():
    src = open(_SCRIPT, encoding="utf-8").read()
    body = src[src.index("    def _belief_bound()"):
               src.index("    # ---------------- the observation's own provenance ----------------")]
    assert "hl_5pt.wire_frame_to_anchor(wire.measured_keypoints(" in body and "wire.readout" in body
    # and the startup check sits between seeding the anchor and the first state readout
    seed = src.index("    planner.seed_anchor(")
    assert seed < src.index("belief-leash frame check") < src.index("names0, state0, _written = planner.state_readout()")
    assert "raise SystemExit" in src[seed:src.index("names0, state0, _written = planner.state_readout()")]


def test_the_subset_leash_bounds_the_ROTATION_the_wrt_start_state_reads():
    """The rig's own curve (wipe_board_new h50, ⏺ run 2026-09-14_14-16-01): the published LEFT
    rotation reached 67.1° from the start against a measured 47.2° — 20° of belief the robot never
    had — and the state (``R_start^T R_anchor``) walked 1.65 band widths out while the WIRE stayed
    bounded. With the 10° leash the believed rotation is at most 10° past the measured one, and the
    observation follows the correction at THIS Predict."""
    from flore.utils.rot6d import rot6d_to_rotmat, rotmat_to_rotvec, rotvec_to_rotmat

    pl = _subset_planner("left_only", To=1)
    slot = hl_5pt.WIRE_ROLES.index("left")
    start = np.eye(3)
    pl.seed_anchor(np.zeros((1, 3)), rotmat_to_rot6d(start)[None])
    believed = rotvec_to_rotmat(np.array([0.0, 0.0, np.radians(67.1)]))
    measured = rotvec_to_rotmat(np.array([0.0, 0.0, np.radians(47.2)]))
    pl.commit_role("left", np.zeros(3), rotmat_to_rot6d(believed))     # the model's own executed step
    rots = [np.eye(3)] * 5
    rots[slot] = measured
    frame = _wire_frame(np.zeros((5, 3)), rots)
    before = pl.state_readout()[1].copy()
    lead, clamped = pl.leak_anchor_toward(frame, eps_m=0.05, eps_rad=np.radians(10.0))
    assert clamped and lead == 0.0            # the position never moved: this was a ROTATION runaway
    got = rot6d_to_rotmat(pl._anchor_rot6d[0])
    dev = np.degrees(np.linalg.norm(rotmat_to_rotvec(measured.T @ got)))
    assert abs(dev - 10.0) < 1e-6             # ON the bound, along the geodesic
    names, row, written = pl.state_readout()
    assert written and not np.array_equal(row, before)
    r6d = row[LEFT.state_slots()["left"]["wrt_start_rot6d"]]
    assert np.allclose(r6d, rotmat_to_rot6d(start.T @ got))
    assert names[0] == "left_r6d0" and r6d[0] > before[0]     # 57.2° reads closer to 1 than 67.1° did
    # inside the bound nothing moves: a healthy run stays byte-identical
    lead2, clamped2 = pl.leak_anchor_toward(frame, eps_m=0.05, eps_rad=np.radians(10.0))
    assert lead2 == 0.0 and not clamped2
    assert np.array_equal(pl.state_readout()[1], row)


def test_both_families_share_ONE_leash_arithmetic():
    """The two planners had drifted (one had the leash, the other a docstring saying it was immune);
    the arithmetic is one function now, and the five-point method is behaviour-identical through it."""
    pl = _planner(To=1)
    n = len(pl.roles)
    measured = np.zeros((n, 3))
    runaway = np.zeros((n, 3))
    runaway[:, 0] = 0.30
    pl.seed_anchor(runaway, np.tile(rotmat_to_rot6d(np.eye(3)), (n, 1)))
    pos, rot, worst, clamped = hl_5pt._leash_toward(
        runaway, np.tile(np.eye(3), (n, 1, 1)), measured, np.tile(np.eye(3), (n, 1, 1)), 0.05, 0.1)
    lead, clamped_pl = pl.leak_anchor_toward(_measured_frame_at(measured), eps_m=0.05, eps_rad=0.1)
    assert clamped and clamped_pl and worst == lead == 0.30
    assert np.allclose(pl._anchor_pos.reshape(n, 3), pos)


def test_a_manual_jaw_press_holds_that_hand_against_the_policy_until_release():
    o = hl_5pt.ManualJawOverride(("left", "right"))
    assert o.roles == () and not o.blocks("left") and not o.intercept("left")
    assert o.describe("left") is None
    assert o.take("left", 0.0) is True            # a POLICY jaw: the press blocks the policy
    assert o.blocks("left") and not o.blocks("right")
    assert [o.intercept("left") for _ in range(3)] == [True] * 3 and not o.intercept("right")
    assert o.roles == ("left",)
    d = o.describe("left")
    assert "MANUAL OVERRIDE" in d and "0% open" in d and "3 policy cmd" in d and "Reset" in d
    assert o.describe("right") is None
    o.take("left", 0.5)                           # a second press moves the held target, keeps the latch
    assert "50% open" in o.describe("left") and o.blocks("left")
    # the lock is re-entrant: the publish paths hold it around check + send, the press around
    # take + send, and every method takes it again inside
    with o.lock:
        with o.lock:
            assert o.intercept("left")
    assert o.release_all() == ("left",)           # Reset is the ONLY release
    assert o.roles == () and not o.blocks("left") and o.describe("left") is None
    # a hand the policy does not predict is recorded but never blocked (nothing to block)
    one = hl_5pt.ManualJawOverride(("left",))
    assert one.take("right", 0.0) is False and not one.blocks("right") and not one.intercept("right")
    assert one.describe("right") == "manual 0% open" and one.roles == ()
    assert one.release_all() == ()


def test_the_page_subscribes_BOTH_jaws_and_a_manual_press_OVERRIDES_the_policy_until_Reset():
    src = open(_SCRIPT, encoding="utf-8").read()
    main = src[src.index("def main("):src.index("def _run_viser(")]
    assert 'for role in ("left", "right"):' in main and "grip_urls[role]" in main
    # the leash's startup log is gated on the planner's CAPABILITY, like _belief_bound itself
    assert 'family == "fivepoint"' not in src
    assert src.count('hasattr(planner, "leak_anchor_toward")') >= 3
    assert "jaw_override = hl_5pt.ManualJawOverride(grip_roles)" in src
    body = src[src.index("    def _manual_jaw("):src.index("    g_jaw: dict = {}")]
    assert '{"target_frac": frac}' in body                       # the SAME frame _drip sends
    for want in ("sub.connected", "jaw_override.take(role, frac)", "with jaw_override.lock:",
                 "manual_jaw_touched.add(role)", "MANUAL OVERRIDE until Reset"):
        assert want in body, want
    # the press is no longer refused while the policy drives the jaw — it WINS
    assert "the POLICY is commanding" not in body and body.count("refused") == 1
    for guard in ("grip_enabled.is_set()", "dripping.is_set()", "auto_running.is_set()"):
        assert guard not in body, guard
    # BOTH publish paths — the chunked drip and the stream — drop the policy's width for a held
    # hand, under the override's lock and BEFORE the jaw state machine
    needle = "if jaw_override.intercept(role):\n                                continue"
    assert src.count(needle) == 2
    start = 0
    for _ in range(2):
        s = src.index(needle, start)
        assert "with jaw_override.lock:" in src[s - 400:s]
        after = src[s:s + 500]
        assert "grip_cmd[role][0](" in after and '.send({"target_frac": float(frac)})' in after
        start = s + len(needle)
    # Reset is the ONLY release: one call, inside the reset branch, after the jaws are opened
    assert src.count("jaw_override.release_all()") == 1
    reset = src[src.index('if flags["reset"]:'):src.index("_rec_reset()   # the ⏺ button back")]
    assert reset.index('_open_jaws("reset")') < reset.index("jaw_override.release_all()")
    # shutdown releases manual jaws too; Reset opens the policy's jaws (as before) and hands them back
    assert '_open_jaws("shutdown", include_manual=True)' in src
    readout = src[src.index("    def _jaw_readout("):src.index("    def _manual_jaw(")]
    assert "motor_ok" in readout and "MOTOR FAULT" in readout
    assert "jaw_override.describe(role)" in readout


# ---------------------------------------------------------------------------
# 2026-09-20: the "📷 Live cameras" folder — and the channel-order bug it would have shown.
# `_decode_jpeg_b64` converts cv2's BGR to RGB and the converter stores RGB (`jpeg_decode` does the
# same BGR->RGB), so a second flip fed the model BGR on three pages: in a live wipe_board_new frame the
# red apple (0.83% of the pixels) became BLUE, a colour the 181 training episodes never contain
# (0.21% red, 0.00% blue). The 3pt page never flipped.
# ---------------------------------------------------------------------------
_SIBLING_PAGES = tuple(os.path.join(os.path.dirname(_PAGE), name)
                       for name in ("hl_serving_mc.py", "hl_serving_wholebody.py", "hl_serving.py"))


def test_the_model_is_fed_the_bridge_frame_in_its_natural_channel_order_on_every_page():
    for path in (_PAGE,) + _SIBLING_PAGES:
        with open(path, encoding="utf-8") as fh:
            src = fh.read()
        assert "frame[:, :, ::-1]" not in src, path
        assert "img[:, :, ::-1]" not in src, path
    src = _page_src()
    assert "planner.push_image(HL._resize_center_crop_224(frame), camera=i)" in src
    for path in _SIBLING_PAGES[:2]:
        with open(path, encoding="utf-8") as fh:
            assert "planner.push_image(HL._resize_center_crop_224(frame), camera=i)" in fh.read(), path


def test_the_live_camera_folder_shows_every_configured_bridge_and_the_models_own_crop_display_only():
    src = _page_src()
    # the folder, and a DISPLAY-ONLY subscription for a bridge the checkpoint does not read
    assert '"📷 Live cameras (what the D405s see NOW)"' in src
    assert 'subs[f"view{i}"] = HL._WSSubscriber(url, f"view{i}")' in src
    assert "_camera_urls(args)[len(cams):], start=len(cams)" in src
    # the ingest, the self-check and the recorder stay on the model's own views
    assert ('for i in range(loaded.num_cameras):\n'
            '                    msg, _wall, _n = subs[f"camera{i}"].latest()') in src
    assert "_self_check(subs, cams, loaded" in src
    assert 'for i in range(int(loaded.num_cameras)) if f"camera{i}" in subs]' in src
    # the refresh: natural colours, a downscale, the model's NEWEST crop per camera; never the planner,
    # the wire or the recorder
    tick = _nested_def(src, "    def _live_cameras_tick(now: float) -> None:")
    assert 'HL._decode_jpeg_b64(msg.get("jpeg_b64"))' in tick and "::-1" not in tick
    assert "_live_thumb(frame, int(args.live_camera_width))" in tick
    assert "planner.snapshot_display_images()" in tick
    assert "idx = c * to + to - 1" in tick                      # camera-major, oldest-first per camera
    for forbidden in ("push_image", "_send(", "wire.commit", "recorder.", "taps."):
        assert forbidden not in tick, forbidden
    # a failure is logged ONCE and the loop goes on; 0 removes the folder and its subscriptions
    assert 'live_cam_ui["failed"] = True' in src
    assert src.count("if args.live_camera_hz > 0:") == 2


def test_live_thumb_keeps_the_aspect_and_the_channel_order_and_the_flags_default_on():
    import importlib.util
    import sys

    spec = importlib.util.spec_from_file_location("_hl5_live_thumb", _PAGE)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_hl5_live_thumb"] = mod
    spec.loader.exec_module(mod)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    frame[..., 0], frame[..., 1], frame[..., 2] = 200, 100, 30          # R, G, B
    small = mod._live_thumb(frame, 320)
    assert small.shape == (240, 320, 3) and small.dtype == np.uint8
    assert [int(round(float(small[..., c].mean()))) for c in range(3)] == [200, 100, 30]
    assert mod._live_thumb(frame, 0) is frame                            # 0 = no downscale
    assert mod._live_thumb(frame, 1280) is frame                         # never upscaled
    args = mod.build_parser().parse_args(["--checkpoint", "x"])
    assert args.live_camera_hz == 4.0 and args.live_camera_width == 320
    assert mod.build_parser().parse_args(["--checkpoint", "x", "--live-camera-hz", "0"]).live_camera_hz == 0
