"""``flore.s1_policy.serving.hl_stream`` — HATO's asynchronous cadence, on a CPU with no model.

The three claims that make streaming safe on this wire, and nothing that needs a checkpoint:

1.  THE WEIGHTS ARE HATO'S. All four modes, oldest-first, convex, and the two selecting modes
    reproduce their chosen prediction rather than approximating it.
2.  A CHUNK IS ADDRESSED IN ABSOLUTE TIME. Row ``k`` of a chunk stamped ``t_obs`` belongs to step
    ``t_obs + k``: a step before the stamp sees nothing of it, a step past ``t_obs + H`` evicts it,
    and a step in between reads exactly that row. This is the latency skip.
3.  THE INTEGRATOR IS THE REVIEWED CLUTCH. :func:`mini_chunk` + ``KeypointClutch.apply`` must equal
    ``base + (cur - prev)`` with the rotation offset composed on the LEFT, the first frame must land
    bit-exactly on the base, and ``--post-anchor off`` must publish the absolute stream unchanged.

``scripts/smoke_test_hl_5pt.py`` covers what only a trained model can answer.
"""

from __future__ import annotations

import numpy as np
import pytest

from flore.s1_policy.serving import hl_stream
from flore.s1_policy.serving.hl_dense import (
    FIVE_POINT_BODIES,
    FIVE_POINT_GATED,
    KeypointClutch,
    PostAnchorConfig,
)
from flore.s1_policy.serving.hl_online import ChunkResult, RoleChunk
from flore.utils.rot6d import rot6d_to_rotmat, rotmat_to_rot6d

ROLES = ("left", "right", "head", "left_foot", "right_foot")
H = 12


def _rotvec_to_rot6d(v):
    from flore.utils.rot6d import rotvec_to_rotmat

    return rotmat_to_rot6d(rotvec_to_rotmat(np.asarray(v, dtype=np.float64).reshape(1, 3)))[0]


def _chunk(offset: float = 0.0, *, horizon: int = H, roles=ROLES, grip: bool = True,
           wrench: bool = True, spin: float = 0.0) -> ChunkResult:
    """A deterministic chunk: role ``i`` ramps along +x from ``offset``, optionally rotating."""
    out = []
    for i, role in enumerate(roles):
        k = np.arange(horizon, dtype=np.float64)
        pos = np.stack([offset + 0.01 * k + i, np.zeros(horizon), np.full(horizon, 0.5)], axis=1)
        rot = np.stack([_rotvec_to_rot6d([0.0, 0.0, spin * kk]) for kk in k])
        out.append(RoleChunk(
            role=role, position=pos, rot6d=rot,
            gripper_m=(0.05 + 0.001 * k) if grip else None,
            wrench_pred=(np.tile(np.arange(6.0), (horizon, 1)) + offset) if wrench else None,
            wrench_cmd=None))
    return ChunkResult(roles=tuple(out), action_times_s=np.arange(horizon) / 30.0,
                       end_anchor_pos=np.zeros((len(roles), 3)),
                       end_anchor_rot6d=np.zeros((len(roles), 6)))


# ---------------------------------------------------------------------------
# 1. the weights
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("mode", hl_stream.ENSEMBLE_MODES)
@pytest.mark.parametrize("n", [1, 2, 3, 7, 40])
def test_weights_are_convex_for_every_mode(mode, n):
    w = hl_stream.ensemble_weights(n, mode, tau=0.5)
    assert w.shape == (n,)
    assert np.all(w >= 0.0)
    assert pytest.approx(1.0, abs=1e-12) == float(w.sum())


def test_new_and_old_are_one_hot_at_the_right_end():
    """OLDEST FIRST is the ordering the blend assumes; a flip here silently swaps the two modes."""
    assert hl_stream.ensemble_weights(5, "old")[0] == 1.0
    assert hl_stream.ensemble_weights(5, "new")[-1] == 1.0


def test_act_unrolls_hatos_own_recursion():
    """``z = c[0]``; ``z = z*tau + c[i]*(1-tau)`` — hato/inference_node.py:134-138."""
    tau, cands = 0.35, np.array([1.0, 2.0, 5.0, -3.0])
    z = cands[0]
    for c in cands[1:]:
        z = z * tau + c * (1.0 - tau)
    w = hl_stream.ensemble_weights(len(cands), "act", tau)
    assert pytest.approx(float(z), rel=1e-12) == float(w @ cands)


def test_act_at_tau_zero_is_new_and_avg_is_uniform():
    assert hl_stream.ensemble_weights(4, "act", 0.0)[-1] == pytest.approx(1.0)
    assert np.allclose(hl_stream.ensemble_weights(4, "avg"), 0.25)


def test_act_survives_tau_underflow_at_a_thick_overlap():
    """``tau**(n-1)`` underflows for a 200-step horizon; the result must still be convex."""
    w = hl_stream.ensemble_weights(200, "act", 0.5)
    assert pytest.approx(1.0, abs=1e-12) == float(w.sum())
    assert np.all(np.isfinite(w))


def test_unknown_mode_and_bad_tau_are_refused():
    with pytest.raises(ValueError, match="mode must be one of"):
        hl_stream.ensemble_weights(3, "mean")
    with pytest.raises(ValueError, match="tau must be finite"):
        hl_stream.ensemble_weights(3, "act", 1.5)
    with pytest.raises(ValueError, match="n must be"):
        hl_stream.ensemble_weights(0, "avg")


# ---------------------------------------------------------------------------
# 1b. the rotation blend
# ---------------------------------------------------------------------------
def test_blend_rot6d_returns_an_orthonormal_rotation():
    a = _rotvec_to_rot6d([0.0, 0.0, 0.0])
    b = _rotvec_to_rot6d([0.0, 0.0, np.deg2rad(20.0)])
    out = hl_stream.blend_rot6d(np.stack([a, b]), np.array([0.5, 0.5]))
    R = rot6d_to_rotmat(out[None])[0]
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-12)
    assert pytest.approx(1.0, abs=1e-12) == float(np.linalg.det(R))
    # The blend of two rotations 20 deg apart sits between them, near the halfway point.
    ang = np.degrees(np.arccos(np.clip((np.trace(R @ rot6d_to_rotmat(a[None])[0].T) - 1) / 2,
                                      -1, 1)))
    assert 8.0 < float(ang) < 12.0


def test_a_one_hot_blend_reproduces_its_chosen_rotation():
    a = _rotvec_to_rot6d([0.1, -0.2, 0.3])
    b = _rotvec_to_rot6d([0.0, 0.0, 1.2])
    out = hl_stream.blend_rot6d(np.stack([a, b]), np.array([0.0, 1.0]))
    assert np.allclose(rot6d_to_rotmat(out[None])[0], rot6d_to_rotmat(b[None])[0], atol=1e-12)


# ---------------------------------------------------------------------------
# 2. absolute-time indexing
# ---------------------------------------------------------------------------
def test_an_empty_stream_returns_none_rather_than_zeros():
    s = hl_stream.ChunkStream(H, ROLES)
    assert s.at(0) is None
    assert s.live_at(0) == 0


def test_row_k_belongs_to_step_t_obs_plus_k():
    s = hl_stream.ChunkStream(H, ROLES)
    ch = _chunk()
    s.add(ch, t_obs=5)
    assert s.at(4) is None, "a step BEFORE the stamp must see nothing of the chunk"
    for k in (0, 1, H - 1):
        got = s.at(5 + k)
        assert got is not None
        assert np.allclose(got.position["left"], ch.role("left").position[k])
        assert got.n_blended == 1
        assert got.newest_age == k


def test_a_chunk_is_evicted_once_its_horizon_is_past():
    s = hl_stream.ChunkStream(H, ROLES)
    s.add(_chunk(), t_obs=0)
    assert s.at(H - 1) is not None
    assert s.at(H) is None
    assert s.stats()["evicted"] == 1
    assert s.stats()["stored"] == 0


def test_the_latency_skip_never_replays_a_past_row():
    """A chunk stamped in the past contributes its CURRENT row, not its row 0.

    This is HATO's ``while pt < self.t: acts = acts[1:]``. Getting it wrong executes an action late
    and would look like the policy lagging rather than like a bookkeeping error.
    """
    s = hl_stream.ChunkStream(H, ROLES)
    ch = _chunk()
    s.add(ch, t_obs=0)
    got = s.at(4)
    assert np.allclose(got.position["left"], ch.role("left").position[4])
    assert got.oldest_age == 4


def test_overlapping_chunks_blend_at_the_same_absolute_step():
    s = hl_stream.ChunkStream(H, ROLES, mode="avg")
    s.add(_chunk(offset=0.0), t_obs=0)     # older
    s.add(_chunk(offset=1.0), t_obs=2)     # newer, same ramp shifted by 1 m
    got = s.at(4)
    assert got.n_blended == 2 and got.oldest_age == 4 and got.newest_age == 2
    old_row = 0.01 * 4 + 0.0                       # role 0 (left) has i = 0
    new_row = 1.0 + 0.01 * 2
    assert pytest.approx(0.5 * (old_row + new_row), abs=1e-12) == float(got.position["left"][0])


@pytest.mark.parametrize("mode,expect", [("old", 0.04), ("new", 1.02)])
def test_selecting_modes_pick_exactly_one_prediction(mode, expect):
    s = hl_stream.ChunkStream(H, ROLES, mode=mode)
    s.add(_chunk(offset=0.0), t_obs=0)
    s.add(_chunk(offset=1.0), t_obs=2)
    assert pytest.approx(expect, abs=1e-12) == float(s.at(4).position["left"][0])


def test_one_live_chunk_is_bit_identical_to_that_chunks_row():
    """The degenerate case must be the chunked cadence, or the two paths are not comparable."""
    s = hl_stream.ChunkStream(H, ROLES, mode="avg")
    ch = _chunk(spin=0.05)
    s.add(ch, t_obs=0)
    for k in range(H):
        got = s.at(k)
        for role in ROLES:
            assert np.array_equal(got.position[role], ch.role(role).position[k])
            assert np.allclose(rot6d_to_rotmat(got.rot6d[role][None])[0],
                               rot6d_to_rotmat(ch.role(role).rot6d[k][None])[0], atol=1e-12)


def test_gripper_and_wrench_ride_the_same_weights():
    s = hl_stream.ChunkStream(H, ROLES, mode="avg")
    s.add(_chunk(offset=0.0), t_obs=0)
    s.add(_chunk(offset=1.0), t_obs=0)
    got = s.at(3)
    assert pytest.approx(0.05 + 0.001 * 3, abs=1e-12) == got.gripper_m["left"]
    # wrench rows are arange(6) + offset, so the mean adds 0.5
    assert np.allclose(got.wrench["left"], np.arange(6.0) + 0.5)


def test_a_member_that_cannot_contribute_is_dropped_not_counted_as_zero():
    """A NaN width must not drag the blended width toward zero: it is UNKNOWN, not closed."""
    s = hl_stream.ChunkStream(H, ROLES, mode="avg")
    bad = _chunk(offset=0.0)
    widths = bad.role("left").gripper_m.copy()
    widths[3] = np.nan
    rc = [r if r.role != "left" else RoleChunk(
        role="left", position=r.position, rot6d=r.rot6d, gripper_m=widths,
        wrench_pred=r.wrench_pred, wrench_cmd=None) for r in bad.roles]
    s.add(ChunkResult(roles=tuple(rc), action_times_s=bad.action_times_s,
                      end_anchor_pos=bad.end_anchor_pos,
                      end_anchor_rot6d=bad.end_anchor_rot6d), t_obs=0)
    s.add(_chunk(offset=0.0), t_obs=0)
    got = s.at(3)
    assert pytest.approx(0.05 + 0.001 * 3, abs=1e-12) == got.gripper_m["left"]


def test_a_role_without_a_gripper_simply_has_no_entry():
    s = hl_stream.ChunkStream(H, ROLES)
    s.add(_chunk(grip=False, wrench=False), t_obs=0)
    got = s.at(0)
    assert got.gripper_m == {} and got.wrench == {}


def test_a_non_finite_chunk_is_refused_before_it_enters_the_blend():
    """Refusing at ``add`` is the point: a bad sample would otherwise poison every step it covers,
    and the wire gate only sees one frame at a time."""
    s = hl_stream.ChunkStream(H, ROLES)
    ch = _chunk()
    ch.role("left").position[2, 0] = np.inf
    with pytest.raises(FloatingPointError, match="non-finite"):
        s.add(ch, t_obs=0)
    assert s.at(0) is None


def test_a_short_chunk_is_refused_by_name():
    s = hl_stream.ChunkStream(H, ROLES)
    with pytest.raises(ValueError, match="built for"):
        s.add(_chunk(horizon=H - 1), t_obs=0)


def test_a_chunk_missing_a_role_is_refused():
    s = hl_stream.ChunkStream(H, ROLES)
    with pytest.raises(ValueError, match=r"no \['right_foot'\] role"):
        s.add(_chunk(roles=ROLES[:-1]), t_obs=0)


def test_overflow_drops_the_oldest_and_says_so():
    s = hl_stream.ChunkStream(H, ROLES, max_entries=3)
    for t in range(6):
        s.add(_chunk(), t_obs=t)
    st = s.stats()
    assert st["stored"] == 3 and st["overflow_dropped"] == 3


def test_reset_drops_everything():
    s = hl_stream.ChunkStream(H, ROLES)
    s.add(_chunk(), t_obs=0)
    s.reset()
    assert s.at(0) is None


def test_mode_and_tau_are_live_and_validated():
    s = hl_stream.ChunkStream(H, ROLES)
    s.mode = "act"
    s.tau = 0.25
    assert s.mode == "act" and s.tau == 0.25
    with pytest.raises(ValueError):
        s.mode = "nope"
    with pytest.raises(ValueError):
        s.tau = -1.0


def test_concurrent_add_and_at_do_not_raise():
    import threading

    s = hl_stream.ChunkStream(H, ROLES)
    errors: list[BaseException] = []

    def writer():
        try:
            for t in range(300):
                s.add(_chunk(offset=t * 0.001), t_obs=t % 40)
        except BaseException as exc:  # noqa: BLE001 - the test IS the assertion
            errors.append(exc)

    def reader():
        try:
            for t in range(300):
                s.at(t % 40)
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    ts = [threading.Thread(target=writer), threading.Thread(target=reader)]
    for th in ts:
        th.start()
    for th in ts:
        th.join(timeout=30)
    assert not errors, errors


# ---------------------------------------------------------------------------
# 2b. the step clock
# ---------------------------------------------------------------------------
def test_step_clock_is_reentrant_and_marks():
    c = hl_stream.StepClock()
    assert c.value == 0
    with c.lock:
        with c.lock:              # the publisher's critical section nests into planner methods
            c.mark(7)
    assert c.value == 7
    c.reset()
    assert c.value == 0


# ---------------------------------------------------------------------------
# 3. the integrator IS the reviewed clutch
# ---------------------------------------------------------------------------
def _clutch(carry_all: bool = True) -> KeypointClutch:
    n = len(FIVE_POINT_BODIES)
    base_pos = np.arange(3 * n, dtype=np.float64) * 0.01
    base_orn = np.tile(np.array([0.0, 0.0, 0.0, 1.0]), n)
    return KeypointClutch(base_pos, base_orn, post=PostAnchorConfig(mode="command"),
                          n_points=n, n_carried=n if carry_all else FIVE_POINT_GATED)


def _steps(offset_a: float, offset_b: float, k_a: int, k_b: int, spin: float = 0.0):
    a = _chunk(offset=offset_a, spin=spin)
    b = _chunk(offset=offset_b, spin=spin)
    s = hl_stream.ChunkStream(H, ROLES, mode="new")
    s.add(a, t_obs=0)
    prev = s.at(k_a)
    s.reset()
    s.add(b, t_obs=0)
    cur = s.at(k_b)
    return prev, cur


def test_mini_chunk_row_1_is_base_plus_the_delta():
    """``p_pub = p_base + (p_cur - p_prev)``, which is ``apply``'s own formula at H = 2."""
    clutch = _clutch()
    clutch.commit((clutch.base[0], clutch.base[1]))          # arm the carry
    prev, cur = _steps(0.0, 0.0, 2, 5)
    mini = hl_stream.mini_chunk(prev, cur, ROLES)
    from flore.s1_policy.serving.hl_dense import KeypointReadout

    readout = KeypointReadout(np.zeros(3))
    rows = {rc.role: {"position": rc.position, "rot6d": rc.rot6d} for rc in mini.roles}
    pos, orn = readout.all_from_decoded(rows)
    pub_pos, _pub_orn = clutch.apply(pos, orn)
    base = clutch.base[0].reshape(-1, 3)
    delta = (pos[1] - pos[0]).reshape(-1, 3)
    assert np.allclose(pub_pos[1].reshape(-1, 3), base + delta, atol=1e-12)


def test_the_first_published_frame_lands_on_the_base():
    """``prev is cur`` at the start of a run, so the delta is zero and step 0 IS the clutch base —
    the same property the chunked cadence gets from a chunk's own row 0.

    Asserted to float precision rather than bit-exactly, deliberately: ``apply`` computes
    ``a + (base - a)``, which is only exactly ``base`` when the two are close enough for the
    subtraction to be exact. On the rig they ARE (the chunk's step 0 is 0-63 mm from the ready
    frame, which is where ``smoke_test_hl_5pt`` measures 0.0), but this synthetic chunk is a metre
    away and asserting bit-equality here would be asserting a property of the test's own numbers.
    """
    from flore.s1_policy.serving.hl_dense import KeypointReadout

    clutch = _clutch()
    clutch.commit((clutch.base[0], clutch.base[1]))
    _prev, cur = _steps(0.0, 0.0, 3, 3)
    mini = hl_stream.mini_chunk(cur, cur, ROLES)
    readout = KeypointReadout(np.zeros(3))
    rows = {rc.role: {"position": rc.position, "rot6d": rc.rot6d} for rc in mini.roles}
    pub_pos, pub_orn = clutch.apply(*readout.all_from_decoded(rows))
    assert float(np.abs(pub_pos[1] - clutch.base[0]).max()) < 1e-15
    assert np.allclose(pub_orn[1], clutch.base[1], atol=1e-12)


def test_the_rotation_offset_is_composed_on_the_left():
    """``R_pub = (R_base R_prev^T) R_cur``. Right-composing conjugates the offset by the chunk's own
    motion, so a turning hand would also drift — the trap ``KeypointClutch`` documents."""
    from flore.s1_policy.serving.hl_dense import KeypointReadout, _rotmats_from_quat_xyzw

    n = len(FIVE_POINT_BODIES)
    base_pos = np.zeros(3 * n)
    base_orn = np.tile(_rot_quat(np.deg2rad(30.0)), n)
    clutch = KeypointClutch(base_pos, base_orn, post=PostAnchorConfig(mode="command"),
                            n_points=n, n_carried=n)
    clutch.commit((base_pos, base_orn))
    prev, cur = _steps(0.0, 0.0, 1, 4, spin=0.1)
    readout = KeypointReadout(np.zeros(3))
    rows = {rc.role: {"position": rc.position, "rot6d": rc.rot6d}
            for rc in hl_stream.mini_chunk(prev, cur, ROLES).roles}
    pos, orn = readout.all_from_decoded(rows)
    _pp, pub_orn = clutch.apply(pos, orn)
    R_prev = _rotmats_from_quat_xyzw(orn[0], n)
    R_cur = _rotmats_from_quat_xyzw(orn[1], n)
    R_base = _rotmats_from_quat_xyzw(base_orn, n)
    want = np.einsum("nij,njk->nik", np.einsum("nij,nkj->nik", R_base, R_prev), R_cur)
    got = _rotmats_from_quat_xyzw(pub_orn[1], n)
    assert np.allclose(got, want, atol=1e-9)


def _rot_quat(yaw: float) -> np.ndarray:
    return np.array([0.0, 0.0, np.sin(yaw / 2.0), np.cos(yaw / 2.0)])


def test_post_anchor_off_publishes_the_absolute_stream_unchanged():
    from flore.s1_policy.serving.hl_dense import KeypointReadout

    n = len(FIVE_POINT_BODIES)
    clutch = KeypointClutch(np.zeros(3 * n), np.tile(np.array([0.0, 0.0, 0.0, 1.0]), n),
                            post=PostAnchorConfig(mode="off"), n_points=n)
    prev, cur = _steps(0.0, 0.0, 2, 5)
    readout = KeypointReadout(np.zeros(3))
    rows = {rc.role: {"position": rc.position, "rot6d": rc.rot6d}
            for rc in hl_stream.mini_chunk(prev, cur, ROLES).roles}
    pos, orn = readout.all_from_decoded(rows)
    pub_pos, pub_orn = clutch.apply(pos, orn)
    assert np.array_equal(pub_pos[1], pos[1])
    assert np.array_equal(pub_orn[1], orn[1])


def test_integrating_a_stream_telescopes_to_one_constant_offset():
    """The whole safety argument for streaming's clutch: publishing per-step deltas from an advancing
    base is the SAME as one fixed offset held for the run, so ``--post-anchor``/``--carry`` keep
    their meaning and nothing accumulates that a chunked run would not."""
    from flore.s1_policy.serving.hl_dense import KeypointReadout

    clutch = _clutch()
    clutch.commit((clutch.base[0], clutch.base[1]))
    base0 = clutch.base[0].copy()
    readout = KeypointReadout(np.zeros(3))
    s = hl_stream.ChunkStream(H, ROLES, mode="avg")
    s.add(_chunk(offset=0.0), t_obs=0)
    prev = s.at(0)
    first_abs = None
    for k in range(H):
        cur = s.at(k)
        rows = {rc.role: {"position": rc.position, "rot6d": rc.rot6d}
                for rc in hl_stream.mini_chunk(prev, cur, ROLES).roles}
        pos, orn = readout.all_from_decoded(rows)
        if first_abs is None:
            first_abs = pos[0].copy()
        pub_pos, pub_orn = clutch.apply(pos, orn)
        clutch.commit((pub_pos[1], pub_orn[1]))
        # published == base0 + (absolute_now - absolute_first), for every step
        assert np.allclose(pub_pos[1], base0 + (pos[1] - first_abs), atol=1e-12)
        prev = cur


# ---------------------------------------------------------------------------
# 4. the run's own readout
# ---------------------------------------------------------------------------
def test_stream_stats_reports_the_publish_cadence_and_the_overlap():
    st = hl_stream.StreamStats()
    st.note_sample(51.0)
    for p in (33.0, 34.0, 120.0):
        st.note_step(p, blended=17, newest_age=2)
    assert st.steps == 3 and st.worst_period_ms == 120.0
    text = st.describe(1.0 / 30.0)
    assert "overlap 17" in text and "frozen 0" in text and "51 ms" in text


def test_stream_stats_rings_stay_bounded():
    st = hl_stream.StreamStats()
    for _ in range(5000):
        st.note_step(33.0, 1, 0)
        st.note_sample(50.0)
    assert len(st.period_ms) <= 2048 and len(st.sample_ms) <= 256
