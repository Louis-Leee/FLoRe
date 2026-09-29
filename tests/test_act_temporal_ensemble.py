"""Temporal ensembling (ACT's ``--temporal_agg``): the weights, the frame contract, and the
one place this port MUST diverge from upstream.

Why this has its own file: ensembling is the only part of the ACT preset that changes what
reaches the ROBOT rather than what the model computes. Its two failure modes are both silent
— a blend that mixes reference frames still produces smooth-looking motion in the wrong
place, and an ensembler that always has exactly one survivor looks like it is running while
doing nothing.
"""

from __future__ import annotations

import numpy as np
import pytest

from fuji.s1_policy.model.act import TemporalEnsembler
from fuji.utils.rot6d import rot6d_to_rotmat, rotmat_to_rot6d, rotvec_to_rotmat

_IDENT6 = rotmat_to_rot6d(np.eye(3))


def _rot6d_about_z(deg: float) -> np.ndarray:
    return rotmat_to_rot6d(rotvec_to_rotmat(np.array([0.0, 0.0, np.radians(deg)])))


def _chunk(H: int, start: float = 0.0, step: float = 0.1) -> np.ndarray:
    """A straight-line absolute path of H steps along +x."""
    return np.stack([start + step * np.arange(H), np.zeros(H), np.zeros(H)], axis=1)


# ---------------------------------------------------------------------------
# The weights — upstream's formula and, critically, its ORDERING
# ---------------------------------------------------------------------------
def test_the_weights_match_upstreams_formula():
    """Upstream: ``exp_weights = np.exp(-k * np.arange(n)); /= sum`` with k=0.01
    (imitate_episodes.py:255-257). Reproduced independently here rather than by calling the
    implementation, so this is a real assertion and not a tautology."""
    te = TemporalEnsembler(50, k=0.01)
    for n in (1, 2, 7, 50):
        expected = np.exp(-0.01 * np.arange(n))
        expected /= expected.sum()
        np.testing.assert_allclose(te.weights(n), expected, rtol=0, atol=1e-15)
        assert te.weights(n).sum() == pytest.approx(1.0)


def test_the_OLDEST_surviving_prediction_gets_the_MOST_weight():
    """Upstream's ordering, and it is deliberate — not a bug to "fix". `i = 0` indexes the
    OLDEST survivor, so `exp(-k*i)` weights it highest. An older chunk saw a more distant
    observation but has been corroborated by every step since; weighting it up is what
    smooths the executed path instead of chasing the newest, noisiest chunk.

    Getting this backwards would still produce a plausible-looking blend, which is exactly
    why it needs pinning."""
    w = TemporalEnsembler(50, k=0.5).weights(5)
    assert w[0] > w[-1], "the oldest survivor must dominate (upstream's ordering)"
    assert list(w) == sorted(w, reverse=True), "weights must decay monotonically with age"


def test_k_zero_is_a_plain_mean():
    w = TemporalEnsembler(10, k=0.0).weights(4)
    np.testing.assert_allclose(w, np.full(4, 0.25))


# ---------------------------------------------------------------------------
# THE FRAME CONTRACT — the one place this port must diverge from upstream
# ---------------------------------------------------------------------------
def test_blending_absolutely_survives_a_moving_anchor():
    """THE reason this port takes ABSOLUTE poses rather than the raw chunk.

    Upstream's actions are ABSOLUTE joint targets, so two chunks predicted one step apart
    share a frame. FUJI's chunk is ANCHOR-RELATIVE, so blending raw values would mix frames
    — measured up to 82 cm of error at the H=50 window's oldest survivor at 0.5 m/s.

    Here every chunk agrees on the same ABSOLUTE path while its own anchor marches along it.
    A frame-correct ensembler reproduces the path EXACTLY; one that blended relative
    coordinates would smear it backwards by the accumulated anchor drift."""
    H = 10
    te = TemporalEnsembler(H, k=0.01)
    path = _chunk(40, step=0.1)  # 10 cm per step -> a big, obvious drift
    got = []
    for t in range(20):
        # The chunk predicted at wall-step t covers path[t : t+H], decoded against the anchor
        # that was live at t (i.e. path[t]) -- which is what infer_chunk already returns.
        out = te.add(path[t:t + H], np.repeat(_IDENT6[None], H, axis=0))
        got.append(out["position"])
    np.testing.assert_allclose(np.array(got), path[:20], rtol=0, atol=1e-12)


def test_a_relative_space_blend_would_have_been_wrong_by_a_lot():
    """Quantifies the bug the frame contract avoids, so the docstring's 82 cm is testable
    rather than a claim. Simulates what upstream's algorithm does verbatim: blend the RAW
    relative rows. With a marching anchor the result lags the truth badly."""
    H = 10
    k = 0.01
    path = _chunk(40, step=0.1)
    # Relative chunk at t: path[t:t+H] - path[t]. Blending THOSE (upstream, verbatim) and
    # then adding the CURRENT anchor is the frame error.
    wrong = []
    for t in range(H - 1, 20):
        rows = [path[t_src:t_src + H][t - t_src] - path[t_src]
                for t_src in range(t - (H - 1), t + 1)]
        w = np.exp(-k * np.arange(len(rows)))
        w /= w.sum()
        wrong.append((np.stack(rows) * w[:, None]).sum(axis=0) + path[t])
    err = np.abs(np.array(wrong)[:, 0] - path[H - 1:20, 0])
    assert err.max() > 0.4, (
        f"a relative-space blend should be badly wrong with a moving anchor; got "
        f"{err.max():.3f} m — if this shrank, the test's premise changed")


def test_the_ensembler_saturates_at_the_horizon_and_does_not_stall_at_one():
    """The silent-no-op failure: an ensembler that always has ONE survivor looks like it is
    running while doing nothing. Survivor count must grow to H and stay there."""
    H = 6
    te = TemporalEnsembler(H, k=0.01)
    counts = [te.add(_chunk(H), np.repeat(_IDENT6[None], H, axis=0))["n_blended"]
              for _ in range(12)]
    assert counts[:H] == list(range(1, H + 1)), f"survivors must ramp 1..H, got {counts[:H]}"
    assert set(counts[H:]) == {H}, f"then saturate at H={H}, got {counts[H:]}"


def test_an_all_zero_prediction_is_NOT_treated_as_absent():
    """Upstream detects occupancy with ``torch.all(actions != 0, axis=1)`` against a
    zero-initialized buffer, so an exactly-all-zero prediction reads as ABSENT. On ALOHA's
    joint targets that is merely unlikely; here "hold still" is genuinely all-zeros in a
    relative pose block and a masked step's gripper/force columns are hard zeros. Dropping
    such a prediction would silently change the blend."""
    H = 4
    te = TemporalEnsembler(H, k=0.0)
    te.add(np.zeros((H, 3)), np.repeat(_IDENT6[None], H, axis=0))
    out = te.add(np.ones((H, 3)), np.repeat(_IDENT6[None], H, axis=0))
    assert out["n_blended"] == 2, "the all-zero chunk must still count as a prediction"
    # k=0 -> plain mean of 0 and 1.
    np.testing.assert_allclose(out["position"], np.full(3, 0.5))


# ---------------------------------------------------------------------------
# Rotation blending
# ---------------------------------------------------------------------------
def test_the_blended_rotation_is_a_valid_rotation():
    """A weighted mean of rot6d rows is not a rotation. Averaging MATRICES then
    re-orthonormalizing is the standard first-order blend, and the result must land back on
    SO(3) — `inference/common.py::_valid_rot6d` rejects a degenerate rot6d at inference, so
    a sloppy blend would surface as a refusal on the rig."""
    H = 3
    te = TemporalEnsembler(H, k=0.01)
    rots = np.stack([_rot6d_about_z(d) for d in (0.0, 15.0, 30.0)])
    for _ in range(3):
        out = te.add(np.zeros((H, 3)), rots)
    R = rot6d_to_rotmat(out["rot6d"][None])[0]
    np.testing.assert_allclose(R @ R.T, np.eye(3), rtol=0, atol=1e-9)
    assert float(np.linalg.det(R)) == pytest.approx(1.0, abs=1e-9)


def test_identical_rotations_blend_to_themselves_exactly():
    H = 5
    te = TemporalEnsembler(H, k=0.01)
    r = _rot6d_about_z(37.0)
    for _ in range(H + 2):
        out = te.add(np.zeros((H, 3)), np.repeat(r[None], H, axis=0))
    np.testing.assert_allclose(out["rot6d"], r, rtol=0, atol=1e-12)


def test_a_blend_of_two_rotations_lands_between_them():
    H = 2
    te = TemporalEnsembler(H, k=0.0)  # plain mean
    te.add(np.zeros((H, 3)), np.repeat(_rot6d_about_z(0.0)[None], H, axis=0))
    out = te.add(np.zeros((H, 3)), np.repeat(_rot6d_about_z(40.0)[None], H, axis=0))
    R = rot6d_to_rotmat(out["rot6d"][None])[0]
    angle = np.degrees(np.arctan2(R[1, 0], R[0, 0]))
    assert 0.0 < angle < 40.0, f"the blend must lie between the inputs, got {angle:.1f} deg"


# ---------------------------------------------------------------------------
# Gripper / wrench, validity, and the guards
# ---------------------------------------------------------------------------
def test_gripper_and_wrench_are_blended_and_non_finite_entries_are_skipped():
    """A NaN gripper/wrench entry must be EXCLUDED from its blend, not poison it — the
    dataset leaves both unmeasured on some steps, and a NaN reaching the wire would command
    a nonsense jaw target or force."""
    H = 3
    te = TemporalEnsembler(H, k=0.0)
    rots = np.repeat(_IDENT6[None], H, axis=0)
    te.add(np.zeros((H, 3)), rots, np.full(H, 0.04), np.zeros((H, 6)))
    bad_grip = np.full(H, np.nan)
    bad_wrench = np.full((H, 6), np.nan)
    out = te.add(np.zeros((H, 3)), rots, bad_grip, bad_wrench)
    assert out["gripper_m"] == pytest.approx(0.04), "the NaN entry must be skipped, not averaged"
    np.testing.assert_allclose(out["wrench"], np.zeros(6))


def test_absent_gripper_and_wrench_yield_none_rather_than_a_fabricated_zero():
    """A role with no gripper (the head) must report None. Zero is a LEGAL jaw width and a
    legal wrench, so fabricating one would be indistinguishable from a measurement."""
    H = 3
    te = TemporalEnsembler(H, k=0.01)
    out = te.add(np.zeros((H, 3)), np.repeat(_IDENT6[None], H, axis=0))
    assert out["gripper_m"] is None and out["wrench"] is None


def test_reset_clears_every_stored_prediction():
    H = 4
    te = TemporalEnsembler(H, k=0.01)
    for _ in range(H):
        te.add(_chunk(H), np.repeat(_IDENT6[None], H, axis=0))
    assert len(te) == H and te.step == H
    te.reset()
    assert len(te) == 0 and te.step == 0
    assert te.add(_chunk(H), np.repeat(_IDENT6[None], H, axis=0))["n_blended"] == 1


@pytest.mark.parametrize(("kwargs", "message"), [
    (dict(action_horizon=0), "action_horizon"),
    (dict(action_horizon=10, k=-1.0), "k must be"),
    (dict(action_horizon=10, k=float("nan")), "k must be"),
    (dict(action_horizon=10, capacity=5), "capacity"),
])
def test_construction_is_validated(kwargs, message):
    with pytest.raises(ValueError, match=message):
        TemporalEnsembler(**kwargs)


def test_a_wrong_chunk_shape_or_a_nan_pose_is_refused():
    H = 4
    te = TemporalEnsembler(H, k=0.01)
    rots = np.repeat(_IDENT6[None], H, axis=0)
    with pytest.raises(ValueError, match="position must be"):
        te.add(np.zeros((H + 1, 3)), rots)
    with pytest.raises(ValueError, match="rot6d must be"):
        te.add(np.zeros((H, 3)), np.zeros((H, 9)))
    with pytest.raises(FloatingPointError, match="non-finite"):
        te.add(np.full((H, 3), np.nan), rots)


# ---------------------------------------------------------------------------
# The serving seam
# ---------------------------------------------------------------------------
def test_the_config_default_is_ON_for_act_and_absent_for_the_siblings():
    """Ensembling is ACT's own published mechanism and is default-ON because this data is
    measurably bimodal. The siblings must not acquire the field: flow has RTC and diffusion
    has a plain sampler, and a stray attribute would make LivePlanner's inert path live."""
    from fuji.s1_policy.model.dp_act import DPACTConfig
    from fuji.s1_policy.model.dp_diffusion import DPDiffusionConfig
    from fuji.s1_policy.model.dp_flow import DPFlowConfig

    assert DPACTConfig().temporal_ensemble is True
    assert DPACTConfig().temporal_ensemble_k == 0.01
    assert not hasattr(DPFlowConfig(), "temporal_ensemble")
    assert not hasattr(DPDiffusionConfig(), "temporal_ensemble")


def test_a_non_ensembling_checkpoint_refuses_the_ensembled_entry_point():
    """A caller must not silently get single-chunk behavior while believing it is
    ensembling."""
    from fuji.s1_policy.serving.hl_online import LivePlanner, replay_only_model

    planner = LivePlanner(replay_only_model("left_only"), obs_hz=30.0)
    assert planner.temporal_ensemble_enabled is False
    assert planner._ensembler("left") is None
    with pytest.raises(RuntimeError, match="does not use temporal"):
        planner.infer_step_ensembled(seed=0, force_enabled=False, gate_n=6.0)
    planner.reset_temporal_ensemble()  # must be a harmless no-op


def test_the_planner_caches_one_ensembler_per_role_and_reset_reaches_it():
    """Cached (not rebuilt per call), or every step would start from a single survivor —
    the silent-no-op failure again, this time at the serving seam."""
    import types

    from fuji.s1_policy.serving.hl_online import LivePlanner, replay_only_model

    loaded = replay_only_model("left_only")
    # replay_only_model has config=None; stand in the ACT knobs the planner reads.
    object.__setattr__(loaded, "config",
                       types.SimpleNamespace(temporal_ensemble=True, temporal_ensemble_k=0.01,
                                             action_horizon=50))
    planner = LivePlanner(loaded, obs_hz=30.0)
    assert planner.temporal_ensemble_enabled is True
    first = planner._ensembler("left")
    assert first is not None and planner._ensembler("left") is first, "must be cached"
    first.add(_chunk(50), np.repeat(_IDENT6[None], 50, axis=0))
    assert len(first) == 1
    planner.reset_temporal_ensemble()
    assert len(first) == 0, "reset must reach the cached ensembler"


# ===========================================================================
# THE SERVING WIRING — is the feature actually REACHABLE from the rig path?
#
# These exist because of a real bug: every piece below the UI (the ensembler, the planner
# entry point, the config flag) was implemented and unit-tested, while ``hl_serving.py``
# only ever called ``infer_chunk`` — so an ACT checkpoint carrying temporal_ensemble=True
# ran UN-ensembled on the robot, with nothing in the logs saying so. Unit-testing a feature
# does not prove it is CONNECTED; these assert the connection.
# ===========================================================================
def _hl_serving_source() -> str:
    from pathlib import Path
    return (Path(__file__).resolve().parents[1] / "scripts" / "hl_serving.py").read_text()


def test_the_execute_path_ACTUALLY_calls_the_ensembled_entry_point():
    """The regression guard for the shipped bug: infer_step_ensembled must be reachable
    from the executor, not merely defined in the library.

    Checks REACHABILITY, not just presence — the bug being pinned is precisely "the code
    exists but nothing reaches it", so a mere substring test would have passed while the
    feature was dead. The Run path must both call ``_drip_ensembled`` and decide to do so
    from the live toggle rather than from a constant."""
    src = _hl_serving_source()
    assert "infer_step_ensembled" in src, (
        "hl_serving.py never calls infer_step_ensembled — temporal ensembling is dead code "
        "on the rig even though the checkpoint asks for it (this is the bug this test pins)")
    # The executor must be INVOKED, not merely defined + passed around.
    assert "_drip_ensembled(" in src.replace("def _drip_ensembled(", ""), (
        "_drip_ensembled is never called")
    # And the decision must be data-driven: a hardcoded False anywhere in the decision
    # region would disable the feature while leaving every call site intact — exactly how the
    # original bug read. Check the WHOLE region (first assignment through the branch), not
    # just its first line: the value is refined over several statements.
    i = src.index("use_ens = ")
    decision = src[i:src.index("if use_ens:", i)]
    assert "ens_ctl[" in decision, "the cadence decision does not read the live toggle"
    for line in decision.split("\n"):
        stripped = line.split("#")[0].strip()
        assert stripped not in ("use_ens = False", "use_ens = True"), (
            f"the cadence decision is hardcoded ({stripped!r}), not read from the toggle")


def test_the_ui_exposes_a_toggle_and_routes_the_run_button_through_it():
    """A knob the operator cannot reach is the same defect as no knob."""
    src = _hl_serving_source()
    assert "Temporal ensembling" in src, "no viser control for ensembling"
    assert "ens_ctl" in src and "ens_lock" in src
    # The Run path must CHOOSE a cadence, i.e. read the toggle where it executes.
    assert 'ens_ctl["enabled"]' in src


def test_a_recorded_replay_is_never_ensembled():
    """Re-querying blends the POLICY; a recorded chunk has no policy behind it. If replay
    were routed through the ensembler, 'compare the demo and the model on the same rig'
    would silently stop comparing the demo.

    Both execute paths must exclude it: the manual Run (``_run_staged``, which has a staged
    chunk that may have come from the replay source) and the sequential rollout (which refuses
    the replay source outright, since replay has its own Play button)."""
    src = _hl_serving_source()
    staged = src[src.index("def _run_staged"):src.index("# NOTE: continuous mode")]
    i = staged.index("use_ens = ")
    assert 'out.get("replay") is None' in staged[i:i + 400], (
        "the manual Run cadence choice does not exclude a recorded-replay chunk")
    seq = src[src.index("def _run_to_completion"):src.index("def _predict(")]
    assert 'replay_ctl["source"] == "replay"' in seq, (
        "the sequential rollout does not exclude a recorded-replay source")


def test_the_per_step_resync_does_NOT_reset_the_ensembler():
    """The subtle one. Blending is ABSOLUTE, so a moving anchor is fine; resetting on each
    re-anchor would leave exactly one survivor and disable the feature while appearing to
    run it. Assert the per-step loop re-syncs WITHOUT resetting.

    Scoped to the per-step LOOP BODY rather than the whole function, because the execute is
    also where an abandoned-intent request (⏹ Stop, or the low level being stopped) is
    consumed — exactly once, before the first step. A reset per STEP is the defect; a reset per
    EXECUTE is the fix. Both halves are pinned here so neither can drift into the other."""
    src = _hl_serving_source()
    start = src.index("def _drip_ensembled")
    body = src[start:src.index("def _drip(", start)]
    assert "quiet=True" in body, "the 30 Hz loop must not log once per step"
    head, sep, loop = body.partition("for _i in range(int(n_steps))")
    assert sep, "could not locate the per-step loop in _drip_ensembled"
    assert "reset_temporal_ensemble" not in loop, (
        "the per-step loop resets the ensembler — that silently reduces the blend to one "
        "survivor (see TemporalEnsembler.reset's docstring)")
    assert "take_pending_ens_reset" in head, (
        "the execute never consumes an abandoned-intent reset request, so a Stop / a stopped "
        "low level would leave the survivors of the aborted rollout to outvote the next query")
    # ...and that entry reset must be REQUESTED, never unconditional: an unconditional drop here
    # would empty the store on every chunk of a sequential run — the same silent failure as the
    # per-step reset, just moved one level out.
    assert head.count("reset_temporal_ensemble") == 1, "expected exactly one entry-level drop"
    assert "if take_pending_ens_reset" in head, "the entry-level drop is not gated on a request"


def test_reset_start_pose_DOES_drop_the_survivors():
    """The other half of the contract: an ABANDONED intent must clear the buffer, or the arm
    gets dragged back toward a trajectory the operator discarded.

    ``reset_start_pose`` has TWO call sites and only one of them is an abandoned intent, so
    this cannot key on the first textual match:
      1. the STARTUP anchor re-seed, which re-latches the wrt_start reference onto the
         arms-zeroed ready pose (initialization — no chunk has been predicted yet, so there
         are no survivors to drop and a call there would be dead code);
      2. the operator's ↺ Reset start pose button — the abandoned intent this test is about.
    """
    import re

    src = _hl_serving_source()
    sites = [m.start() for m in re.finditer(r"planner\.reset_start_pose\(\)", src)]
    assert len(sites) == 2, (
        f"expected exactly 2 reset_start_pose call sites (startup re-latch + operator button), "
        f"found {len(sites)} — if a third was added, decide explicitly whether it abandons an "
        f"intent and extend this test")
    operator_site = sites[-1]  # the button callback is the later one
    assert "reset_temporal_ensemble" in src[operator_site:operator_site + 700], (
        "Reset start pose does not clear the ensembler")


def test_the_ensembled_loop_reuses_the_shared_wire_and_gate_helpers():
    """The two cadences must not grow independent copies of the frame conversion or the
    force gate — that is how a frame seam or a gate bypass gets into one path only."""
    src = _hl_serving_source()
    start = src.index("def _drip_ensembled")
    body = src[start:src.index("def _drip(", start)]
    assert "_wire_from_step" in body, "the ensembled path builds its wire ad hoc"
    assert "_set_last_wire" in body, "the ensembled path does not update the broadcast wire"
    assert "_commit_left_step" in body, "the ensembled path does not advance the decode anchor"
    # THE FORCE GATE IS THE SAFETY-CRITICAL ONE, so check every value that reaches the wire
    # rather than merely that the helper is mentioned somewhere in the body. Both publish
    # sites (the initial _set_last_wire and the drip_hz re-send inside the hold loop) must
    # route their force through the gate; an ungated wrench here would push the low level
    # with the raw prediction regardless of the master switch.
    assert body.count("_gate_wire_force(") >= 2, (
        "the ensembled path has a force publish that bypasses the LIVE gate — expected the "
        "gate at BOTH the _set_last_wire and the _s0_send re-publish")
    for call in ("_set_last_wire(pos_i, orn_i, f_i)", "_s0_send(pos_i, orn_i, _gate_wire_force("):
        assert call in body, f"expected gated publish {call!r} in the ensembled loop"
    # And nothing may hand a raw wrench slice straight to the wire.
    assert "wrench_i[0:3]" not in body and "wrench_i[:3]" not in body, (
        "a raw (ungated) wrench slice reaches the wire in the ensembled loop")


def test_a_flow_or_diffusion_checkpoint_cannot_enable_ensembling():
    """``supported`` is fixed from the checkpoint, so the toggle cannot promise a cadence the
    model has no entry point for."""
    src = _hl_serving_source()
    assert 'ens_ctl["supported"]' in src
    i = src.index("_ens_supported = ")
    assert "temporal_ensemble_enabled" in src[i:i + 200], (
        "support is not derived from the loaded checkpoint's own flag")


# ===========================================================================
# BEHAVIORAL: drive a real LivePlanner through infer_step_ensembled.
# The source-level guards above prove the wiring EXISTS; these prove it WORKS.
# ===========================================================================
def _ensembling_planner():
    """A LivePlanner on a real (tiny) ACT model with ensembling on, fed a plausible obs."""
    import torch

    from fuji.s1_policy.model.dp_act import DPACTConfig, DPACTModel
    from fuji.s1_policy.serving.hl_online import LivePlanner, replay_only_model

    torch.manual_seed(0)
    cfg = DPACTConfig(
        hidden_dim=32, dim_feedforward=64, nheads=4, enc_layers=1, dec_layers=1,
        vae_enc_layers=1, latent_dim=8, action_dim=16, action_horizon=50, state_dim=7,
        obs_history_len=1, force_hist_len=1, force_hist_channels=6,
        force_enc_width=16, force_enc_out_dim=8, layout_mode="left_only",
        pretrained=False, temporal_ensemble=True, temporal_ensemble_k=0.01,
    )
    model = DPACTModel(cfg).eval()
    # obs_history_len=1 must match the model's To (HL contract v3 is a single obs frame);
    # replay_only_model defaults to 2, which would feed a 14-wide state to a 7-wide model.
    loaded = replay_only_model("left_only", obs_history_len=1)
    object.__setattr__(loaded, "config", cfg)
    object.__setattr__(loaded, "model", model)
    object.__setattr__(loaded, "algorithm", "act")
    object.__setattr__(loaded, "num_steps", 1)
    object.__setattr__(loaded, "device", torch.device("cpu"))
    object.__setattr__(loaded, "norm_stats", {})

    planner = LivePlanner(loaded, obs_hz=30.0)
    planner.seed_anchor(np.array([[0.44, 0.15, 0.10]]), np.array([[1.0, 0, 0, 0, 1.0, 0]]))
    rng = np.random.default_rng(0)
    for _ in range(4):
        planner.push_image((rng.random((224, 224, 3)) * 255).astype(np.uint8))
        planner.push_force(rng.normal(0, 1, (1, 6)))
        planner.push_gripper("left", 0.05)
        planner.push_proprio_from_anchor()
    return planner


def test_end_to_end_the_survivor_count_actually_grows_across_steps():
    """The live failure mode is a blend that always has ONE survivor: it looks like it is
    ensembling and is doing nothing. Drive the real entry point and watch n_blended climb."""
    planner = _ensembling_planner()
    counts = [planner.infer_step_ensembled(
        seed=i, force_enabled=True, gate_n=0.0, role="left")["n_blended"] for i in range(6)]
    assert counts == [1, 2, 3, 4, 5, 6], counts


def test_end_to_end_a_moving_anchor_does_not_reset_the_blend():
    """The per-step loop commits the anchor to each commanded step (that is the real cadence).
    Because the blend is ABSOLUTE this must NOT reduce the survivor count."""
    planner = _ensembling_planner()
    n_last = 0
    for i in range(5):
        out = planner.infer_step_ensembled(seed=i, force_enabled=True, gate_n=0.0, role="left")
        # exactly what _drip_ensembled does between steps
        planner.commit_role("left", out["position"], out["rot6d"])
        n_last = out["n_blended"]
    assert n_last == 5, f"anchor motion collapsed the blend to {n_last} survivors"


def test_end_to_end_the_blend_returns_a_usable_absolute_pose():
    """A blended step must be a finite absolute pose + a valid rotation, or the wire build
    downstream would publish garbage to the low level."""
    planner = _ensembling_planner()
    out = planner.infer_step_ensembled(seed=0, force_enabled=True, gate_n=0.0, role="left")
    assert out["position"].shape == (3,) and np.isfinite(out["position"]).all()
    assert out["rot6d"].shape == (6,)
    R = rot6d_to_rotmat(out["rot6d"][None])[0]
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-9), "blended rotation is not orthonormal"
    assert np.isclose(np.linalg.det(R), 1.0, atol=1e-9)


def test_end_to_end_reset_returns_the_blend_to_a_single_survivor():
    planner = _ensembling_planner()
    for i in range(4):
        planner.infer_step_ensembled(seed=i, force_enabled=True, gate_n=0.0, role="left")
    planner.reset_temporal_ensemble()
    out = planner.infer_step_ensembled(seed=9, force_enabled=True, gate_n=0.0, role="left")
    assert out["n_blended"] == 1, "reset did not clear the stored predictions"


def test_a_single_obs_frame_is_NOT_throttled_by_the_history_stride():
    """``img_stride`` time-anchors the To obs HISTORY frames. With To=1 (HL contract v3)
    there is no history to space, so the stride must be 1.

    The bug this pins: the derivation ``round(obs_history_dt_s * obs_hz)`` gave 8 at
    obs_hz=30, so a new camera frame reached the ring only every 8 ticks (3.75 Hz) while the
    dataset's camera — what the model trained on — runs at 29.988 Hz. In the chunk cadence
    that mostly hid (one Predict per chunk), but temporal ensembling re-queries every step,
    and 8 consecutive queries seeing a BYTE-IDENTICAL image makes the blend average near-
    duplicate predictions: the feature appears to run and cannot do its job."""
    src = _hl_serving_source()
    i = src.index("_derived_stride")
    region = src[i - 900:i + 900]
    assert "obs_history_len) <= 1" in region, (
        "the stride derivation does not special-case To=1, so a single-observation "
        "checkpoint gets its only obs frame throttled below the trained camera rate")


def test_the_ensembled_loop_warns_when_it_cannot_hold_the_action_grid():
    """A re-query that overruns the grid period plays the chunk slower than trained. That
    must be SAID, not silently absorbed as a sluggish-looking policy."""
    src = _hl_serving_source()
    start = src.index("def _drip_ensembled")
    body = src[start:src.index("def _drip(", start)]
    assert "not keeping up" in body, "no warning when the re-query misses the grid period"
    assert "slow_logged" in body, "the slow-step warning is not rate-limited"


# ===========================================================================
# SEQUENTIAL MODEL ROLLOUT — the cadence upstream ACT actually evaluates in
# (imitate_episodes.py:225, `for t in range(max_timesteps)`), where the ensembler's
# buffer persists across the whole episode.
# ===========================================================================
def test_the_sequential_rollout_exists_and_is_bounded():
    """A policy has no end of its own, so an unbounded loop would keep driving the arm.
    It must be stoppable AND time-bounded, or 'Run to completion' is unsafe."""
    src = _hl_serving_source()
    assert "def _run_to_completion" in src, "no sequential model rollout"
    start = src.index("def _run_to_completion")
    body = src[start:src.index("def _predict(", start)]
    assert "stop.is_set()" in body, "the process stop event does not interrupt the rollout"
    assert 'auto_model_ctl["gen"] != my_gen' in body, "Stop cannot retire an in-flight loop"
    assert "busy.acquire" in body and "busy.release" in body, (
        "the rollout does not hold the GPU/IK mutex — a manual Predict could run concurrently")
    # THE TIME CEILING IS THE GUARD THAT STOPS THE ARM IF THE OPERATOR LOOKS AWAY, so assert
    # the COMPARISON and its `break`, not merely that the variable is mentioned. (A first cut
    # of this test checked `"max_s" in body` and survived deleting the whole check.)
    loop = body[body.index("while not stop.is_set()"):]
    assert "elapsed >= max_s" in loop, (
        "the rollout does not compare elapsed time against the ceiling — it is UNBOUNDED and "
        "would keep driving the arm until Stop")
    assert "elapsed = time.time() - t0" in loop, "elapsed is never recomputed inside the loop"

    # EVERY exit condition must actually LEAVE the loop, checked one by one. A condition that
    # is evaluated and then falls through is indistinguishable from no condition at all, and
    # both of the stop paths here are what bring the arm to rest. (An earlier version of this
    # test asserted only that *some* `break` existed after the ceiling, and survived deleting
    # the Stop-path break entirely.)
    for label, cond in (
        ("operator Stop", 'auto_model_ctl["gen"] != my_gen'),
        ("wall-clock ceiling", "elapsed >= max_s"),
    ):
        after = loop[loop.index(cond):]
        # the break must come before the next statement at the loop's own indent level
        assert "break" in after[:400], (
            f"the {label} exit is evaluated but never breaks out of the rollout loop")


def test_the_sequential_rollout_does_NOT_reset_the_blend_between_chunks():
    """THE reason this button matters for ACT. Upstream's eval is ONE continuous loop and the
    ensembler's buffer persists across it; survivors only reach the full horizon after a
    horizon of UNINTERRUPTED steps. Resetting per iteration would cap the blend at the
    exec-steps count and quietly reproduce the chunk-at-a-time behavior."""
    src = _hl_serving_source()
    start = src.index("def _run_to_completion")
    body = src[start:src.index("def _predict(", start)]
    loop = body[body.index("while not stop.is_set()"):]
    assert "reset_temporal_ensemble" not in loop, (
        "the sequential loop resets the ensembler between chunks — that caps the blend at "
        "exec-steps survivors instead of the horizon")
    # It SHOULD reset once, before the loop: a new rollout is a new intent.
    pre = body[:body.index("while not stop.is_set()")]
    assert "reset_temporal_ensemble" in pre, (
        "a fresh rollout must drop survivors from an abandoned earlier attempt")


def test_the_sequential_rollout_refuses_a_recorded_replay_source():
    """Replay has its own Play button with episode-end/masked-frame semantics; routing it
    through the model loop would ignore the cursor entirely."""
    src = _hl_serving_source()
    start = src.index("def _run_to_completion")
    body = src[start:src.index("def _predict(", start)]
    assert 'replay_ctl["source"] == "replay"' in body, (
        "the sequential model loop does not refuse a recorded-replay source")


def test_the_gripper_readonly_default_is_stated_not_silent():
    """The reported symptom: the arm reaches the object and never closes, because
    --command-gripper is off and the predicted width is discarded, while the jaw bridge prints
    'target_frac=freeze'. A silently-disabled control reads as a policy bug."""
    src = _hl_serving_source()
    assert "target_frac=freeze" in src, (
        "nothing connects the read-only default to the symptom the operator actually sees")
    assert "GRIPPER IS READ-ONLY" in src, "no startup warning for the read-only jaw"
    i = src.index('gui.add_folder("Gripper")')
    folder = src[i:i + 1400]
    assert "--command-gripper" in folder, (
        "the greyed-out checkbox does not say what would enable it")


# ===========================================================================
# GRIPPER CLOSING OVERSHOOT — turning a position target into a grip.
#
# Measured motivation (session_2026.08.08_2228_v3, all 32806 valid anchors): the recorded
# gripper action is bimodal and its CLOSED mode is 42-48 mm, never 0, because during
# collection the jaw closed ONTO THE TOOL and the logged width is the tool's thickness.
# Replaying that as a position target stops at the object's surface -> a touch, not a grasp.
# ===========================================================================
def _calib():
    import importlib.util
    import sys
    from pathlib import Path
    p = Path(__file__).resolve().parents[1] / "scripts" / "hl_serving.py"
    spec = importlib.util.spec_from_file_location("_hs_calib", p)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_hs_calib"] = mod
    spec.loader.exec_module(mod)
    return mod.GripperCalib(width_open_m=0.095, width_closed_m=0.0)


def test_zero_overshoot_is_byte_identical_to_the_old_behavior():
    """The escape hatch must be exact: --grip-overshoot-mm 0 has to reproduce the previous
    command stream, or 'turn it off to compare' is not a real control."""
    c = _calib()
    for w_mm in np.linspace(0.0, 95.0, 200):
        w = float(w_mm) / 1000.0
        got, _ = c.to_target_frac_squeezed(w, close_below_m=0.070, overshoot_m=0.0)
        assert got == c.to_target_frac(w)


def test_the_overshoot_only_touches_the_CLOSED_mode():
    """An overshoot applied while OPENING would under-open the jaw — the opposite of the
    intent, and it would silently degrade every release."""
    c = _calib()
    for w_mm in np.linspace(70.0, 95.0, 60):
        w = float(w_mm) / 1000.0
        got, squeezing = c.to_target_frac_squeezed(w, close_below_m=0.070, overshoot_m=0.020)
        assert got == c.to_target_frac(w), f"open-mode width {w_mm} mm was altered"
        assert squeezing is False


def test_a_closed_command_is_driven_past_the_object():
    """The actual fix: at the measured closed width the published target must be NARROWER
    than the prediction, so the jaw stalls on the object and position error becomes force."""
    c = _calib()
    w = 0.0421  # the measured minimum of the closed cluster
    plain = c.to_target_frac(w)
    got, squeezing = c.to_target_frac_squeezed(w, close_below_m=0.070, overshoot_m=0.010)
    assert squeezing is True
    assert got < plain, "a closed command was not driven past the surface"
    # 10 mm of a 95 mm span is 0.1053 of travel.
    assert got == pytest.approx(plain - 0.010 / 0.095, abs=1e-12)


def test_the_overshoot_can_never_command_a_negative_or_over_open_target():
    """A large overshoot must saturate, not wrap or go negative — an out-of-range target_frac
    on the jaw bridge is undefined behavior."""
    c = _calib()
    for over in (0.05, 0.5, 5.0):
        got, _ = c.to_target_frac_squeezed(0.045, close_below_m=0.070, overshoot_m=over)
        assert 0.0 <= got <= 1.0 and got == 0.0
    for w_mm in (0.0, 50.0, 94.9, 95.0):
        got, _ = c.to_target_frac_squeezed(w_mm / 1000.0, close_below_m=0.070, overshoot_m=0.0)
        assert 0.0 <= got <= 1.0


def test_the_observed_width_is_NOT_squeezed():
    """to_width_m feeds the model's PROPRIOCEPTION. Applying the overshoot there would report
    a width the jaw is not at, putting the model out of distribution on its own gripper
    channel — the squeeze is a publish-time concern only."""
    src = _hl_serving_source()
    i = src.index("def to_width_m")
    body = src[i:src.index("def to_target_frac", i)]
    assert "overshoot" not in body, "the overshoot leaked into the OBSERVED width"
    # And the ingest path must use the plain converter.
    j = src.index("planner.push_gripper")
    assert "to_width_m" in src[j - 400:j], "ingest no longer uses the plain width converter"


def test_both_execute_cadences_squeeze_through_ONE_helper():
    """The chunk path and the ensembled path must not compute the jaw command separately, or
    one of them will keep publishing an un-squeezed target."""
    src = _hl_serving_source()
    assert src.count("_grip_target_frac(") >= 2, (
        "expected the shared helper to be used by BOTH execute cadences")
    assert "def build_grip_command" in src, "the jaw state machine is not testable in isolation"
    # Neither publish site may call the plain converter directly any more.
    for marker in ('frac = calib.to_target_frac(float(blend["gripper_m"]))',
                   "frac = calib.to_target_frac(float(grip_m[i]))"):
        assert marker not in src, f"a publish site bypasses the overshoot: {marker}"


# ===========================================================================
# THE GRASP LATCH + HYSTERESIS — why a grip does not loosen mid-hold.
#
# Reported: "gripper grips at first, then loosens again." Measured cause is mostly NOT the
# overshoot: temporal ensembling AVERAGES A BIMODAL signal (closed 42-48 mm, open 84-94 mm,
# nothing between), so whenever the survivors disagree the blend sits at the weighted
# mixture. Simulated on the real cluster means, one brief open flicker inside an established
# grasp walks the commanded target 0.397 -> 0.498 = 10.1% of travel of slow re-opening, of
# which 6.6% is pure loosening. A single threshold also flips mid-ramp, adding a 12% step.
# ===========================================================================
def _blended_widths() -> np.ndarray:
    """The measured failure input: an ensembled bimodal gripper signal. Closed from step 20,
    with a brief open flicker at 78-84 (the model hesitating mid-grasp)."""
    H, k = 50, 0.01
    OPEN, CLOSED = 0.0916, 0.0477  # measured cluster means, metres
    out = []
    for t in range(60, 110):
        vals = [OPEN if (t - age < 20 or 78 <= t - age <= 84) else CLOSED
                for age in range(min(t, H - 1), -1, -1)]
        w = np.exp(-k * np.arange(len(vals)))
        w /= w.sum()
        out.append(float(np.dot(w, vals)))
    return np.asarray(out)


def _blend_scenario(latch: bool, below_mm: float, release_mm: float, over_mm: float = 10.0):
    """Feed the failure input through hl_serving's REAL ``_grip_target_frac`` state machine.

    Deliberately NOT a reimplementation of the latch here: an earlier version of these tests
    replayed the algorithm inline and therefore passed even when the shipped latch was
    disabled, the hysteresis collapsed to one threshold, or the latch froze the first value
    instead of the tightest (all three verified by mutation). So this drives the actual
    closure out of the module, and only the INPUT is synthesized.
    """
    import importlib.util
    import sys
    import threading
    from pathlib import Path

    p = Path(__file__).resolve().parents[1] / "scripts" / "hl_serving.py"
    spec = importlib.util.spec_from_file_location("_hs_grip", p)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_hs_grip"] = mod
    spec.loader.exec_module(mod)

    grip_ctl = {"overshoot_m": over_mm / 1000.0, "close_below_m": below_mm / 1000.0,
                "confirm_steps": 3, "hold_tightest": bool(latch),
                "engaged": False, "run": 0, "run_closing": False, "held_frac": None,
                "last_frac": None, "squeezing": False}
    to_frac, _release = mod.build_grip_command(
        mod.GripperCalib(0.095, 0.0), grip_ctl, threading.Lock())
    return np.asarray([to_frac(float(w))[0] for w in _blended_widths()])


def _max_loosening(seq: np.ndarray) -> float:
    """Largest amount the target ever moves LOOSER than a tighter value already commanded."""
    return float(np.max(np.maximum.accumulate(seq[::-1])[::-1] - seq))


def test_without_the_latch_an_ensembled_grasp_DOES_loosen():
    """Pins the premise. If this stops reproducing, the bimodal-blend mechanism changed and
    the latch's justification needs rechecking rather than silently keeping the code."""
    loose = _max_loosening(_blend_scenario(latch=False, below_mm=70.0, release_mm=70.0))
    assert loose > 0.05, (
        f"expected a bimodal blend to loosen a grasp by >5% of travel; got {loose * 100:.1f}%")


def test_the_latch_eliminates_mid_grasp_loosening():
    """THE fix for the reported symptom."""
    loose = _max_loosening(_blend_scenario(latch=True, below_mm=70.0, release_mm=80.0))
    assert loose == pytest.approx(0.0, abs=1e-12), (
        f"the latch still let the jaw back off by {loose * 100:.1f}% of travel")


def test_the_latch_holds_the_TIGHTEST_target_not_the_first():
    """A grasp that legitimately tightens further must be allowed to; the latch is a floor on
    tightness, not a freeze on the first value."""
    seq = _blend_scenario(latch=True, below_mm=70.0, release_mm=80.0)
    assert np.all(np.diff(seq) <= 1e-12), "the latched target must be monotonically tightening"
    assert seq.min() < seq[0], "the latch froze the first value instead of tracking tighter ones"


def test_a_sustained_OPEN_cancels_the_overshoot():
    """The other half of the rule, and the one that matters for releasing: once the policy
    really means to open, the overshoot must go away and the target follow the prediction
    verbatim — otherwise the jaw would stay biased closed and never let go."""
    import importlib.util
    import sys
    import threading
    from pathlib import Path

    p = Path(__file__).resolve().parents[1] / "scripts" / "hl_serving.py"
    spec = importlib.util.spec_from_file_location("_hs_open", p)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_hs_open"] = mod
    spec.loader.exec_module(mod)

    c = mod.GripperCalib(0.095, 0.0)
    ctl = {"overshoot_m": 0.010, "close_below_m": 0.070, "confirm_steps": 3,
           "hold_tightest": True, "engaged": False, "run": 0, "run_closing": False,
           "held_frac": None, "last_frac": None, "squeezing": False}
    to_frac, _ = mod.build_grip_command(c, ctl, threading.Lock())

    for _ in range(10):          # establish a grasp
        frac, engaged = to_frac(0.0477)
    assert engaged is True and frac < c.to_target_frac(0.0477)
    for _ in range(10):          # then a sustained open
        frac, engaged = to_frac(0.0916)
    assert engaged is False, "the overshoot never cancelled on a sustained open"
    assert frac == pytest.approx(c.to_target_frac(0.0916), abs=1e-12), (
        "after release the target must follow the prediction verbatim, with no residual bias")


def test_a_deliberate_open_clears_the_grasp_state():
    """Otherwise the held target survives into the NEXT grasp and clamps the jaw the instant it
    engages, before the model asked for that width."""
    src = _hl_serving_source()
    assert "def release_latch" in src, "no way to clear the latch"
    i = src.index("def _release_gripper")
    body = src[i:i + 2000]
    assert "_grip_release_latch()" in body, (
        "Reset start pose opens the jaw but leaves the grasp latch engaged")


def test_the_confirmation_run_stops_the_squeeze_chattering():
    """A single flickering prediction must not toggle the overshoot. Turn the width guard OFF
    so the engage/release STATE is what is under test, then dither across the threshold: with
    confirm=1 the squeeze flips on every crossing; a confirmation run rejects the flicker."""
    import importlib.util
    import sys
    import threading
    from pathlib import Path

    p = Path(__file__).resolve().parents[1] / "scripts" / "hl_serving.py"
    spec = importlib.util.spec_from_file_location("_hs_hyst", p)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_hs_hyst"] = mod
    spec.loader.exec_module(mod)

    def transitions(confirm):
        ctl = {"overshoot_m": 0.010, "close_below_m": 0.070, "confirm_steps": confirm,
               "hold_tightest": False,  # isolate the STATE machine from the width guard
               "engaged": False, "run": 0, "run_closing": False, "held_frac": None,
               "last_frac": None, "squeezing": False}
        to_frac, _ = mod.build_grip_command(mod.GripperCalib(0.095, 0.0), ctl, threading.Lock())
        # alternate one step either side of the threshold — the pathological flicker
        widths = [(69.0 if i % 2 == 0 else 71.0) / 1000.0 for i in range(40)]
        states = [to_frac(w)[1] for w in widths]
        return int(np.sum(np.diff(np.asarray(states, dtype=int)) != 0))

    n1, n3 = transitions(1), transitions(3)
    assert n1 > 4, f"premise broken: confirm=1 should chatter on an alternating dither, got {n1}"
    assert n3 == 0, (
        f"a {3}-step confirmation run still let the squeeze toggle {n3} times on a "
        f"single-step flicker (confirm=1 gives {n1})")


# ===========================================================================
# ABANDONED INTENT — dropping the stored predictions when the operator stops.
#
# Live symptom that produced these tests: after stopping mid-rollout (viser Stop, or joystick
# B to stop the low-level controller) and re-arming, the arm sometimes lurched off along the
# motion that had just been aborted. The ensembler is the store that survives a stop: its
# per-entry cursors advance per ADD, not per wall-clock second, so a re-arm minutes later
# still resumes the old chunk at the next index — and because the weights are oldest-first,
# the one fresh query cannot outvote the survivors.
# ===========================================================================
def _hs_reset_protocol():
    import importlib.util
    import sys
    from pathlib import Path
    p = Path(__file__).resolve().parents[1] / "scripts" / "hl_serving.py"
    spec = importlib.util.spec_from_file_location("_hs_ensreset", p)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_hs_ensreset"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_a_stopped_rollout_OUTVOTES_the_fresh_query_until_it_is_dropped():
    """The severity premise, quantified — this is WHY a Stop must drop the store.

    Fill the window with an abandoned plan heading -x, then ask for the only currently-valid
    motion (+1 m in +x). The blend still moves the arm the WRONG WAY, because the fresh query
    is one survivor among 50 and holds the SMALLEST weight (exp(-k*49)/sum = 1.6%)."""
    H = 50
    te = TemporalEnsembler(H, k=0.01)
    rots = np.repeat(_IDENT6[None], H, axis=0)
    abandoned = np.stack([-0.01 * np.arange(H), np.zeros(H), np.zeros(H)], axis=1)
    for _ in range(H - 1):
        te.add(abandoned, rots)
    fresh = np.stack([np.ones(H), np.zeros(H), np.zeros(H)], axis=1)

    out = te.add(fresh, rots)
    assert out["n_blended"] == H, "premise broken: the window should be full of survivors"
    assert out["position"][0] < 0.0, (
        f"the abandoned plan must be shown to dominate for this fix to be needed, but the "
        f"blend gave x={out['position'][0]:.4f} — expected NEGATIVE (following -x) while the "
        f"only valid command was +1.0 m")

    # ...and dropping the store is what makes the very next step obey the fresh query alone.
    te.reset()
    after = te.add(fresh, rots)
    assert after["n_blended"] == 1
    np.testing.assert_allclose(after["position"], [1.0, 0.0, 0.0], atol=1e-12)


def test_the_reset_request_is_consumed_exactly_once():
    """Single-shot is load-bearing. Stop/the watchdog can only REQUEST the drop (they run on
    the viser-callback and status threads; the ensembler belongs to whoever holds `busy`), and
    the drip loop consumes it once per execute. A flag that survived being read would re-empty
    the ensembler on EVERY chunk of a sequential run, leaving one survivor per blend —
    ensembling silently disabled while the page still reported it ON."""
    import threading

    hs = _hs_reset_protocol()
    ens_ctl, ens_lock = {"pending_reset": False}, threading.Lock()

    assert hs.take_pending_ens_reset(ens_ctl, ens_lock) is False, "nothing requested yet"
    hs.request_ens_reset(ens_ctl, ens_lock)
    assert hs.take_pending_ens_reset(ens_ctl, ens_lock) is True
    for _ in range(5):  # a sequential run's subsequent chunks must NOT re-reset
        assert hs.take_pending_ens_reset(ens_ctl, ens_lock) is False


def test_the_reset_protocol_is_inert_without_an_ensembler():
    """flow / diffusion / replay-only sessions have no ensembler and pass None, so neither end
    of the protocol may raise — the Stop button and the low-level watchdog are shared by every
    algorithm and must not need to branch on it."""
    import threading

    hs = _hs_reset_protocol()
    hs.request_ens_reset(None, threading.Lock())          # no-op, no raise
    hs.request_ens_reset({"pending_reset": False}, None)   # no-op, no raise
    assert hs.take_pending_ens_reset(None, threading.Lock()) is False
    assert hs.take_pending_ens_reset({"pending_reset": True}, None) is False


def test_stop_requests_the_drop():
    """⏹ Stop is one of the three documented reset triggers (TemporalEnsembler.reset: "a fresh
    episode, a Reset-start-pose, a Stop"). It was the missing one: ▶▶ Run to completion resets
    at its own entry, so only the manual Stop → Predict → ▶ Run path carried survivors of the
    stopped rollout into the next execute."""
    src = _hl_serving_source()
    i = src.index("@auto_stop_btn.on_click")
    handler = src[i:src.index("@predict_btn.on_click", i)]
    assert "request_ens_reset" in handler, "Stop does not drop the stored predictions"


def test_losing_the_low_level_drops_the_staged_plan_and_the_survivors():
    """Joystick B (stop the controller) is INVISIBLE as a flag: the RecordTap frame is published
    at the end of the rl_inference tick, which only runs while ``use_policy_action`` is True, so
    a STOP shows up as the frame stream simply ceasing — never as ``vive_armed=False``. The
    readback going stale is therefore the only observable the planner has, and it must be
    treated as the intent being abandoned: the low level has dropped to stiff hold and cleared
    its own slew/clutch anchors, and will next be re-armed from a pose nothing was planned from.

    Pinned by source because the watchdog lives inside ``_run_viser``'s status tick (which needs
    a viser server and a live RecordTap to run at all), the same way the other UI-path
    invariants in this file are pinned."""
    src = _hl_serving_source()
    i = src.index('_ll_watch["was_live"] and not r_fresh')
    block = src[i:i + 1600]
    assert "request_ens_reset" in block, (
        "a stopped low level does not drop the survivors — re-arming would resume the "
        "abandoned motion, which is the symptom this watchdog exists for")
    assert 'staged["out"] = None' in block, (
        "the staged chunk survives the stop; it was decoded against the pre-stop anchor, and "
        "while one is staged the idle heartbeat refuses to re-track the measured pose")
    assert 'auto_model_ctl["gen"] += 1' in block, (
        "a sequential run is not retired; it watches only its own generation counter and would "
        "keep dripping into a low level that stopped listening")
    # The edge must be a FALLING edge off a remembered state, not "currently stale" — otherwise
    # a planner started before the low level would fire on every tick until the robot appears.
    assert '_ll_watch["was_live"] = bool(r_fresh)' in src, "the watchdog never updates its edge state"
