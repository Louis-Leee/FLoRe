"""Online HL-v2 planner core: force gate, chunk decode, standalone loader, live obs."""

from __future__ import annotations

import numpy as np
import pytest

from fuji.s1_policy.data.layout import ACTION_HORIZON, FORCE_DIM, get_layout
from fuji.s1_policy.data.transforms import GRIPPER_WIDTH_REF_M, absolute_to_delta_ee, rot6d_relative
from fuji.s1_policy.serving.hl_online import (
    LivePlanner,
    decode_chunk,
    exec_step_choices,
    gate_force_command,
    load_hl_model,
    snap_exec_steps,
)

# Reuse the tiny-checkpoint fixture builder from the offline-inference tests.
from tests.test_inference import _make_bundle

pytest.importorskip("zarr")


_IDENTITY_ROT6D = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float64)


# ---------------------------------------------------------------------------
# The force gate (the load-bearing safety contract) — pure, no model.
# ---------------------------------------------------------------------------
def test_gate_switch_off_zeros_everything_regardless_of_magnitude():
    w = np.array([[100.0, 0, 0, 5, 5, 5], [0.1, 0, 0, 0, 0, 0]])
    out = gate_force_command(w, force_enabled=False, gate_n=6.0)
    assert np.all(out == 0.0)
    # Even a zero gate with the switch off is still all-zeros.
    assert np.all(gate_force_command(w, force_enabled=False, gate_n=0.0) == 0.0)


def test_gate_below_threshold_hard_zeros_above_passes_through():
    # 10 N passes a 6 N gate; 3 N is gated; torque rides with the force decision.
    w = np.array([[10.0, 0, 0, 1, 2, 3], [3.0, 0, 0, 9, 9, 9]])
    out = gate_force_command(w, force_enabled=True, gate_n=6.0)
    np.testing.assert_array_equal(out[0], w[0])  # above-gate: full passthrough incl torque
    np.testing.assert_array_equal(out[1], np.zeros(6))  # below-gate: exact zero


def test_gate_uses_force_magnitude_not_torque():
    # Small force but huge torque must STILL be gated — the deadband is on ||F||.
    w = np.array([[1.0, 0.0, 0.0, 50.0, 50.0, 50.0]])
    out = gate_force_command(w, force_enabled=True, gate_n=6.0)
    np.testing.assert_array_equal(out[0], np.zeros(6))


def test_gate_boundary_is_strictly_greater_than():
    # Exactly gate_n stays gated (strict >), just above passes.
    at = np.array([[6.0, 0.0, 0.0, 0, 0, 0]])
    just_over = np.array([[6.0001, 0.0, 0.0, 0, 0, 0]])
    assert np.all(gate_force_command(at, force_enabled=True, gate_n=6.0) == 0.0)
    assert np.allclose(gate_force_command(just_over, force_enabled=True, gate_n=6.0), just_over)


def test_gate_rejects_bad_threshold_and_width():
    with pytest.raises(ValueError, match="gate_n must be finite"):
        gate_force_command(np.zeros((1, 6)), force_enabled=True, gate_n=-1.0)
    with pytest.raises(ValueError, match="gate_n must be finite"):
        gate_force_command(np.zeros((1, 6)), force_enabled=True, gate_n=float("nan"))
    with pytest.raises(ValueError, match="last dim"):
        gate_force_command(np.zeros((1, 3)), force_enabled=True, gate_n=6.0)


# ---------------------------------------------------------------------------
# decode_chunk: per-role un-relativize round-trip + gate wiring — pure.
# ---------------------------------------------------------------------------
def test_decode_chunk_round_trips_absolute_poses_and_applies_gate():
    layout = get_layout("left_only")
    H = 16

    # Build a known ABSOLUTE left trajectory + wrench, relativize it against a chosen
    # anchor, and confirm decode_chunk recovers the absolute poses exactly.
    anchor_pos = np.array([[0.2, 0.1, -0.1]])
    anchor_rot6d = np.array([_IDENTITY_ROT6D])
    abs_pos = np.array([[0.2 + 0.01 * t, 0.1, -0.1 + 0.002 * t] for t in range(H)])
    abs_rot6d = np.tile(_IDENTITY_ROT6D, (H, 1))
    rel_pos, rel_rot6d = absolute_to_delta_ee(abs_pos, abs_rot6d, anchor_pos[0], anchor_rot6d[0])

    pred = np.zeros((H, layout.action_dim), dtype=np.float64)
    slots = layout.slots("left")
    pred[:, slots.pos] = rel_pos
    pred[:, slots.rot6d] = rel_rot6d
    pred[:, slots.gripper] = 0.5  # -> 0.5 * ref metres
    wrench = np.tile(np.array([8.0, 0, 0, 0.1, 0, 0]), (H, 1))  # 8 N > 6 N gate
    pred[:, slots.force] = wrench

    result = decode_chunk(pred, layout, anchor_pos, anchor_rot6d, force_enabled=True, gate_n=6.0)
    lc = result.role("left")
    np.testing.assert_allclose(lc.position, abs_pos, atol=1e-9)
    np.testing.assert_allclose(lc.gripper_m, np.full(H, 0.5 * GRIPPER_WIDTH_REF_M), atol=1e-12)
    np.testing.assert_allclose(lc.wrench_pred, wrench, atol=1e-12)  # ungated (viz)
    np.testing.assert_allclose(lc.wrench_cmd, wrench, atol=1e-12)  # 8 N passes the gate
    # The chunk-end anchor is the last absolute step (for commit_anchor).
    np.testing.assert_allclose(result.end_anchor_pos[0], abs_pos[-1], atol=1e-9)


def test_decode_chunk_gate_off_zeros_command_but_keeps_pred():
    layout = get_layout("left_only")
    H = 4
    pred = np.zeros((H, layout.action_dim))
    slots = layout.slots("left")
    pred[:, slots.rot6d] = _IDENTITY_ROT6D  # valid rotation
    pred[:, slots.force] = np.array([9.0, 0, 0, 0, 0, 0])
    result = decode_chunk(
        pred, layout, np.zeros((1, 3)), np.array([_IDENTITY_ROT6D]), force_enabled=False, gate_n=6.0
    )
    lc = result.role("left")
    np.testing.assert_allclose(lc.wrench_pred[:, 0], 9.0)  # model output preserved for viz
    assert np.all(lc.wrench_cmd == 0.0)  # switch OFF -> no command


def test_decode_chunk_head_has_no_gripper_or_force():
    layout = get_layout("both_hands_head")
    H = 4
    pred = np.zeros((H, layout.action_dim))
    for role in layout.roles:
        pred[:, layout.slots(role).rot6d] = _IDENTITY_ROT6D
    n = len(layout.roles)
    result = decode_chunk(
        pred, layout, np.zeros((n, 3)), np.tile(_IDENTITY_ROT6D, (n, 1)),
        force_enabled=True, gate_n=6.0,
    )
    assert result.role("head").gripper_m is None
    assert result.role("head").wrench_pred is None
    # right hand reserves force columns but is NOT a force role -> no wrench emitted
    assert result.role("right").wrench_pred is None
    assert result.role("left").wrench_pred is not None


def test_serving_builds_the_force_window_the_CHECKPOINT_wants(tmp_path, monkeypatch):
    """A checkpoint trained under a DIFFERENT force-window contract must still serve.

    A checkpoint is a self-contained artifact and the rig does not change when the repo's
    contract does (v2's 8 bins -> v3's 1 bin is exactly that move). So the live planner sizes
    its window from `loaded.force_hist_len`, and `load_hl_model` exempts that one field from
    the build-contract pin via `grid_from_checkpoint=True`.

    Without both halves, bumping FORCE_HIST_LEN bricks every existing checkpoint: the pin
    refuses it at load, and even past that the planner would hand a 1-bin encoder an 8-bin
    window, which `ForceHistoryEncoder` hard-rejects. Note the weights need no editing here —
    the encoder's state_dict is L-INDEPENDENT (Conv1d + a 2*width readout never see the length),
    which is exactly why the config field, not the tensor shapes, is the only thing that knows."""
    import json

    from fuji.s1_policy.data.layout import ACTION_DT, FORCE_HIST_LEN

    bundle = _make_bundle(tmp_path, monkeypatch, "flow")
    import fuji.s1_policy.serving.hl_online as hl_online
    from tests.test_inference import _StubEncoder

    monkeypatch.setattr(hl_online, "_build_model", lambda algo, block: _flow_pair(block, _StubEncoder))

    # Rewrite the checkpoint's declared window to a contract this build does NOT use.
    other_len = FORCE_HIST_LEN + 7
    cfg_path = bundle.checkpoint / "config.json"
    raw = json.loads(cfg_path.read_text())
    raw["model"]["force_hist_len"] = other_len
    cfg_path.write_text(json.dumps(raw))

    loaded = load_hl_model(bundle.checkpoint, "flow", device="cpu")
    assert loaded.force_hist_len == other_len
    assert loaded.force_hist_span_s == pytest.approx(other_len * ACTION_DT)

    lp = LivePlanner(loaded, obs_hz=30.0)
    for k in range(other_len):
        lp.push_force(np.arange(FORCE_DIM, dtype=np.float64), t=100.0 + k * ACTION_DT)
    batch, _, _ = lp.build_batch()
    assert tuple(batch["force_history"].shape) == (1, other_len, loaded.layout.force_hist_channels)
    assert bool(batch["force_history_valid"].item()) is True
    # And the model actually accepts it — the encoder's own length assert is the backstop.
    lp.infer_chunk(seed=0, force_enabled=False, gate_n=0.0)


def test_a_dataset_backed_load_still_pins_the_force_window_to_this_build(tmp_path, monkeypatch):
    """The exemption is SERVING-only. An offline session feeds the model from a zarr whose
    attrs are pinned to FORCE_HIST_LEN, so a checkpoint that disagrees would be handed windows
    of the wrong length — that one must still be refused."""
    from fuji.s1_policy.data.layout import ACTION_HORIZON, FORCE_HIST_LEN, get_layout
    from fuji.s1_policy.inference.common import validate_model_layout
    from fuji.s1_policy.model.dp_flow import DPFlowConfig

    layout = get_layout("left_only")
    # action_horizon is passed explicitly rather than left at DPFlowConfig's literal default
    # (50, which also serves the frozen legacy pipeline): validate_model_layout pins it
    # UNCONDITIONALLY, so under FUJI_ACTION_HORIZON=30 the default would trip the horizon
    # check first and this test would stop exercising the force_hist_len exemption at all.
    common = dict(action_dim=layout.action_dim, state_dim=layout.state_dim, obs_history_len=1,
                  num_cameras=1, force_hist_channels=layout.force_hist_channels,
                  action_horizon=ACTION_HORIZON, layout_mode="left_only")
    stale = DPFlowConfig(**common, force_hist_len=FORCE_HIST_LEN + 7)
    hl_data = {"obs_history_len": 1}

    with pytest.raises(ValueError, match="force_hist_len"):
        validate_model_layout(stale, hl_data, layout)          # dataset-backed: refused
    validate_model_layout(stale, hl_data, layout, grid_from_checkpoint=True)  # serving: allowed

    # An incoherent (len, channels) pair is refused on BOTH paths: `grid_from_checkpoint`
    # exempts the length from the build pin, not from having to mean something.
    incoherent = DPFlowConfig(**{**common, "force_hist_channels": layout.force_hist_channels},
                              force_hist_len=0)
    with pytest.raises(ValueError, match="disagree about whether force is present"):
        validate_model_layout(incoherent, hl_data, layout, grid_from_checkpoint=True)


def test_a_force_LESS_checkpoint_is_not_pinned_to_this_build_s_bin_count():
    """A model with no force encoder must load on the DATASET-BACKED path too.

    Both force-less ablations make training record ``force_hist_len=0`` (config.py writes
    ``FORCE_HIST_LEN if layout.force_input_channels else 0``), so pinning the build's bin
    count unconditionally refused every offline consumer -- infer_s1_*.py, reinfer_hl_take.py,
    eval_checkpoints_physical.py -- for a checkpoint ``load_hl_model`` serves happily. There is
    no window to be the wrong length when the model reads zero force channels.

    The regression this replaces was silent about its cause: `force_hist_len=0 disagrees with
    1` carries no hint, unlike the channel mismatch beside it.
    """
    from fuji.s1_policy.data import layout as layout_mod
    from fuji.s1_policy.data.layout import ACTION_HORIZON, FORCE_HIST_LEN, get_layout
    from fuji.s1_policy.inference.common import validate_model_layout
    from fuji.s1_policy.model.dp_flow import DPFlowConfig

    # A patch CONTEXT, not the plain fixture: `get_layout` reads the module flag at CALL time,
    # so the force-ful layout below has to be built with the ablation out of scope again. Same
    # mechanism as test_layout.py -- setattr rather than setenv, because the env var resolves at
    # IMPORT (see _resolve_drop_force_input).
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(layout_mod, "DROP_FORCE_INPUT", True)
        ablated = layout_mod.get_layout("left_only")
    forceful = get_layout("left_only")
    assert ablated.force_input_channels == 0
    assert ablated.force_hist_channels != 0, "the DATASET width must be untouched by the ablation"
    assert forceful.force_input_channels == forceful.force_hist_channels != 0

    common = dict(action_dim=ablated.action_dim, state_dim=ablated.state_dim, obs_history_len=1,
                  num_cameras=1, action_horizon=ACTION_HORIZON, layout_mode="left_only")
    hl_data = {"obs_history_len": 1}
    # Exactly what a NOFORCEIN=1 run writes: no encoder, no window.
    validate_model_layout(DPFlowConfig(**common, force_hist_channels=0, force_hist_len=0),
                          hl_data, ablated)

    # Still refused: a checkpoint that DOES consume force, handed to the ablated process. The
    # channel key is compared before the length one, so this keeps its actionable hint.
    with pytest.raises(ValueError, match="force_hist_channels"):
        validate_model_layout(
            DPFlowConfig(**common, force_hist_channels=ablated.force_hist_channels,
                         force_hist_len=FORCE_HIST_LEN),
            hl_data, ablated)

    # And the build pin is intact for a force-FUL process: 0 channels there is the reverse
    # mismatch, and a wrong-LENGTH window is still a refusal.
    with pytest.raises(ValueError, match="force_hist_channels"):
        validate_model_layout(DPFlowConfig(**common, force_hist_channels=0, force_hist_len=0),
                              hl_data, forceful)
    with pytest.raises(ValueError, match="force_hist_len"):
        validate_model_layout(
            DPFlowConfig(**common, force_hist_channels=forceful.force_hist_channels,
                         force_hist_len=FORCE_HIST_LEN + 7),
            hl_data, forceful)


# ---------------------------------------------------------------------------
# Execute-N-of-S: the page's chunk window must FOLLOW the trained horizon.
# ---------------------------------------------------------------------------
# The bug these pin: every whole-body planner page hardcoded ("8","16","30","50"), so an H=100
# or H=200 checkpoint could not be asked to execute more than a QUARTER of the chunk it was
# trained for. Nothing failed — the drip clamps to min(selected, chunk_len) — the option simply
# was not there, which reads as "the model only does 50 steps".
def test_exec_choices_grow_with_the_horizon():
    assert exec_step_choices(50) == (8, 16, 30, 50)
    assert exec_step_choices(100) == (8, 16, 30, 50, 100)
    assert exec_step_choices(200) == (8, 16, 30, 50, 100, 200)


def test_exec_choices_never_offer_more_steps_than_the_chunk_has():
    # The permissive direction is the one that misleads: a page advertising 200 on a 50-step
    # model would silently execute 50 and report 200 as the cadence.
    for horizon in (1, 4, 7, 8, 16, 29, 30, 31, 50, 99, 100, 137, 200, 333):
        choices = exec_step_choices(horizon)
        assert choices, f"H={horizon} produced an EMPTY dropdown, which is an unusable page"
        assert max(choices) == horizon, (
            f"H={horizon}: 'execute the whole chunk' must always be offered, got {choices}")
        assert all(1 <= c <= horizon for c in choices), choices
        assert list(choices) == sorted(set(choices)), f"{choices} is not sorted/deduplicated"


def test_exec_choices_refuse_a_nonsense_horizon():
    for bad in (0, -1, -50):
        with pytest.raises(ValueError, match="must be >= 1 step"):
            exec_step_choices(bad)


def test_left_only_ladder_is_unchanged_by_the_shared_helper():
    """``hl_serving.py`` keeps ``(16, 30)`` — pinned because its rig default depends on it.

    That planner's ``--exec-steps`` defaults to 4, which SNAPS UP to 16 against this ladder.
    Adding the shared ladder's 8 rung would re-snap it to 8, halving how much of every chunk a
    reviewed hardware path executes as a side effect of a UI change."""
    assert exec_step_choices(50, ladder=(16, 30)) == (16, 30, 50)
    assert exec_step_choices(30, ladder=(16, 30)) == (16, 30)
    assert snap_exec_steps(4, exec_step_choices(50, ladder=(16, 30))) == 16


def test_snap_puts_a_free_form_flag_onto_the_dropdown():
    # viser does NOT reject an off-list initial_value: it renders a blank selector whose .value
    # is still that number, so the page shows nothing while the drip executes it.
    choices = exec_step_choices(200)
    assert snap_exec_steps(20, choices) == 16
    assert snap_exec_steps(999, choices) == 200
    assert snap_exec_steps(0, choices) == 8
    for c in choices:                       # already-valid values must be untouched
        assert snap_exec_steps(c, choices) == c


def test_snap_breaks_ties_toward_the_SHORTER_window():
    """12 is equidistant from 8 and 16; the shorter wins because it re-plans sooner."""
    assert snap_exec_steps(12, exec_step_choices(200)) == 8
    assert snap_exec_steps(40, exec_step_choices(200)) == 30
    with pytest.raises(ValueError, match="no execute-N-of-S choices"):
        snap_exec_steps(16, ())


@pytest.mark.parametrize("script", ["hl_serving_5pt", "hl_serving_mc", "hl_serving_wholebody",
                                    "hl_serving"])
def test_no_planner_page_hardcodes_its_exec_options(script):
    """Source-text pin: the dropdown options must be BUILT from ``_EXEC_CHOICES``.

    A source check rather than a behavioural one because building the page needs viser, a
    display and six sensor bridges. What it protects is narrow and exact: the four planners each
    had (or nearly had) their own literal tuple, and a literal is invisible to every horizon test
    above."""
    from pathlib import Path

    text = (Path(__file__).resolve().parents[1] / "scripts" / f"{script}.py").read_text()
    assert '"8", "16", "30", "50"' not in text, (
        f"{script}.py hardcodes its execute-N options again — they must come from "
        "_EXEC_CHOICES so they follow the checkpoint's horizon")
    assert "_EXEC_CHOICES" in text and "exec_step_choices(" in text, (
        f"{script}.py no longer derives its execute-N choices from the horizon")
    assert "snap_exec_steps(" in text, (
        f"{script}.py no longer snaps --exec-steps onto the offered choices, so a free-form "
        "flag can leave the dropdown blank while the drip executes it")


# ---------------------------------------------------------------------------
# Standalone loader + live observation, against a real tiny checkpoint.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("algorithm", ["flow", "diffusion"])
def test_load_hl_model_standalone_no_zarr_and_infers(tmp_path, monkeypatch, algorithm):
    bundle = _make_bundle(tmp_path, monkeypatch, algorithm)
    # The stub encoder patched onto the inference module also backs hl_online's builder.
    import fuji.s1_policy.serving.hl_online as hl_online
    from tests.test_inference import _StubEncoder

    if algorithm == "flow":

        monkeypatch.setattr(
            hl_online, "_build_model",
            lambda algo, block: _flow_pair(block, _StubEncoder),
        )
    else:

        monkeypatch.setattr(
            hl_online, "_build_model",
            lambda algo, block: _diffusion_pair(block, _StubEncoder),
        )

    # Loads WITHOUT the training zarr on disk (standalone norm_stats).
    loaded = load_hl_model(bundle.checkpoint, algorithm, device="cpu")
    assert loaded.algorithm == algorithm
    assert loaded.layout.mode == "left_only"
    assert loaded.n_force_hands == 1
    assert "actions" in loaded.norm_stats

    lp = LivePlanner(loaded)
    lp.seed_anchor(np.array([[0.2, 0.1, -0.1]]), np.array([_IDENTITY_ROT6D]))
    a = lp.infer_chunk(seed=5, force_enabled=False, gate_n=6.0)
    b = lp.infer_chunk(seed=5, force_enabled=False, gate_n=6.0)
    lc = a.role("left")
    assert lc.position.shape == (ACTION_HORIZON, 3)
    assert np.isfinite(lc.position).all()
    assert lc.wrench_pred.shape == (ACTION_HORIZON, 6)
    assert np.all(lc.wrench_cmd == 0.0)  # switch off
    # Deterministic per seed (both algos use a seeded generator + model.reset()).
    np.testing.assert_array_equal(lc.position, b.role("left").position)


def test_live_state_channel_matches_wrt_start_contract(tmp_path, monkeypatch):
    """build_batch's state must be wrt_start_rot6d = R_start^T R_now per keypoint, with
    NO absolute position — the exact channel HLDataset produces."""
    bundle = _make_bundle(tmp_path, monkeypatch, "flow")
    import fuji.s1_policy.serving.hl_online as hl_online
    from tests.test_inference import _StubEncoder

    monkeypatch.setattr(hl_online, "_build_model", lambda algo, block: _flow_pair(block, _StubEncoder))
    loaded = load_hl_model(bundle.checkpoint, "flow", device="cpu")
    # Drop normalization so we can read the raw state channel directly.
    object.__setattr__(loaded, "norm_stats", {})
    lp = LivePlanner(loaded)

    # Latch a start rotation, then rotate the anchor; state must carry R_start^T R_now.
    start_rot6d = _IDENTITY_ROT6D.copy()
    lp.seed_anchor(np.array([[0.3, 0.0, 0.0]]), np.array([start_rot6d]))
    # A 30-degree yaw as the new "now" orientation.
    from fuji.utils.rot6d import rotmat_to_rot6d

    c, s = np.cos(0.5236), np.sin(0.5236)
    now_R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)
    now_rot6d = rotmat_to_rot6d(now_R)
    lp.commit_anchor(np.array([[0.3, 0.0, 0.0]]), np.array([now_rot6d]))
    lp.push_proprio_from_anchor()

    batch, _ap, _ar = lp.build_batch()
    state = batch["state"].cpu().numpy()[0]  # (To, state_dim)
    entry = loaded.layout.state_slots()["left"]
    wrt_sl = entry["wrt_start_rot6d"]
    expected = rot6d_relative(now_rot6d, start_rot6d)
    # The newest (last) frame reflects the committed now-rotation.
    np.testing.assert_allclose(state[-1, wrt_sl], expected, atol=1e-6)
    # State width is exactly the layout's (wrt_start 6 + gripper 1 for left_only).
    assert state.shape[1] == loaded.layout.state_dim


def test_force_history_valid_flag_tracks_whole_window(tmp_path, monkeypatch):
    bundle = _make_bundle(tmp_path, monkeypatch, "flow")
    import fuji.s1_policy.serving.hl_online as hl_online
    from fuji.s1_policy.data.layout import FORCE_HIST_LEN, FORCE_HIST_SPAN_S
    from tests.test_inference import _StubEncoder

    monkeypatch.setattr(hl_online, "_build_model", lambda algo, block: _flow_pair(block, _StubEncoder))
    loaded = load_hl_model(bundle.checkpoint, "flow", device="cpu", )
    lp = LivePlanner(loaded, obs_hz=16.0)
    # Cold start: the ring is EMPTY -> invalid window (learned no-data embed).
    batch, _, _ = lp.build_batch()
    assert bool(batch["force_history_valid"].item()) is False
    # Every bin measured -> valid. Timestamps are explicit so the window is exact rather than
    # dependent on how fast this test's loop happens to run.
    dt = FORCE_HIST_SPAN_S / FORCE_HIST_LEN
    for k in range(FORCE_HIST_LEN):
        lp.push_force(np.arange(FORCE_DIM, dtype=np.float64), t=100.0 + k * dt)
    batch2, _, _ = lp.build_batch()
    assert bool(batch2["force_history_valid"].item()) is True
    assert tuple(batch2["force_history"].shape) == (1, FORCE_HIST_LEN, loaded.layout.force_hist_channels)
    # A dropout INSIDE the span invalidates the window.
    lp.push_force(None, t=100.0 + FORCE_HIST_LEN * dt)
    assert bool(lp.build_batch()[0]["force_history_valid"].item()) is False
    # ...and stops mattering once it ages OUT of the span, which is the converter's
    # per-window semantic (validity is a property of the window, not of the ring's history).
    for k in range(FORCE_HIST_LEN + 1):
        lp.push_force(np.arange(FORCE_DIM, dtype=np.float64),
                      t=100.0 + (FORCE_HIST_LEN + 1 + k) * dt)
    assert bool(lp.build_batch()[0]["force_history_valid"].item()) is True


def test_force_window_is_span_wide_regardless_of_push_rate(tmp_path, monkeypatch):
    """The window spans exactly FORCE_HIST_SPAN_S — samples older than that are EXCLUDED.

    REGRESSION. The ring used to be keyed on COUNT (``ceil(span*obs_hz) + 2`` slots, all
    averaged). At the v2 grid (8 bins / 1 s) the ``+2`` slack was ~12% and harmless; on the v3
    grid (ONE 33.3 ms bin) it became 3 slots at obs_hz=30 — a 100 ms average published as the
    33.3 ms bin the model trained on. Age-keying makes the width independent of the rate, so
    a bridge that pushes at 200 Hz and one that pushes at 30 Hz agree on the mean."""
    bundle = _make_bundle(tmp_path, monkeypatch, "flow")
    import fuji.s1_policy.serving.hl_online as hl_online
    from fuji.s1_policy.data.layout import FORCE_HIST_SPAN_S
    from tests.test_inference import _StubEncoder

    monkeypatch.setattr(hl_online, "_build_model", lambda algo, block: _flow_pair(block, _StubEncoder))
    loaded = load_hl_model(bundle.checkpoint, "flow", device="cpu")
    object.__setattr__(loaded, "norm_stats", {})  # read the raw binned window
    ch = loaded.layout.force_hist_channels

    def window(rate_hz: float, *, value_in: float, value_out: float) -> np.ndarray:
        lp = LivePlanner(loaded, obs_hz=30.0)
        step = 1.0 / rate_hz
        t_end = 1000.0
        # 3 spans' worth of samples: everything older than one span carries `value_out` and
        # must NOT reach the mean; the last span carries `value_in`.
        n = max(2, int(np.ceil(3.0 * FORCE_HIST_SPAN_S / step)))
        for k in range(n, -1, -1):
            t = t_end - k * step
            v = value_in if (t_end - t) < FORCE_HIST_SPAN_S else value_out
            lp.push_force(np.full(ch, v), t=t)
        return lp.build_batch()[0]["force_history"].numpy()[0]

    for rate in (204.0, 30.0):
        w = window(rate, value_in=5.0, value_out=-100.0)
        assert np.allclose(w, 5.0), (
            f"at {rate} Hz the window leaked pre-span samples: {w[:, 0]}")


def test_edge_aligned_pushes_fill_every_bin(tmp_path, monkeypatch):
    """One sample per bin, pushed exactly ON the bin edges, must fill every bin.

    REGRESSION, and the aligned case is the COMMON one: a bridge running at the action rate,
    or a replay source stepping the action grid, puts every sample on an edge. There `age/dt`
    is a difference of accumulated floats and lands just below the integer — measured
    2.9999999999998295 for the 3*dt-old sample at L=8, so `floor` filed it one bin late, the
    bin it should have filled read EMPTY, and the window reported INVALID even though every
    sample was a real measurement. The encoder would then get its learned no-data embedding on
    fully measured force. Pinned at several L so the tolerance cannot be tuned away."""
    from fuji.s1_policy.data.layout import ACTION_DT, get_layout
    from fuji.s1_policy.serving.hl_online import replay_only_model

    loaded = replay_only_model("left_only", obs_history_len=1)
    ch = get_layout("left_only").force_hist_channels
    for L in (1, 3, 6, 8):
        lp = LivePlanner(loaded, obs_hz=30.0)
        object.__setattr__(lp, "_fh_len", L)
        object.__setattr__(lp, "_fh_span_s", L * ACTION_DT)
        # Value k marks bin k, pushed exactly at the bin's right edge.
        for k in range(L):
            lp.push_force(np.full(ch, float(k)), t=100.0 + k * ACTION_DT)
        win, valid = lp._bin_force_history(list(lp._buf_force))
        assert valid, f"L={L}: an all-real edge-aligned window read INVALID; bins {win[:, 0]}"
        np.testing.assert_allclose(
            win[:, 0], np.arange(L, dtype=np.float64), atol=1e-9,
            err_msg=f"L={L}: samples landed in the wrong bins")


def test_force_history_binning_is_causal_mean(tmp_path, monkeypatch):
    """Each bin is the MEAN of the raw samples whose timestamps land in it (anti-aliasing),
    with the last bin ending AT t0 — the converter's ``(lo, hi]`` causal tiling."""
    bundle = _make_bundle(tmp_path, monkeypatch, "flow")
    import fuji.s1_policy.serving.hl_online as hl_online
    from fuji.s1_policy.data.layout import FORCE_HIST_LEN, FORCE_HIST_SPAN_S
    from tests.test_inference import _StubEncoder

    monkeypatch.setattr(hl_online, "_build_model", lambda algo, block: _flow_pair(block, _StubEncoder))
    loaded = load_hl_model(bundle.checkpoint, "flow", device="cpu")
    object.__setattr__(loaded, "norm_stats", {})
    lp = LivePlanner(loaded, obs_hz=16.0)
    ch = loaded.layout.force_hist_channels
    # Two samples per bin, at values k and k+1 -> bin k must read k + 0.5.
    dt = FORCE_HIST_SPAN_S / FORCE_HIST_LEN
    t0 = 500.0
    for b in range(FORCE_HIST_LEN):
        # Both strictly inside bin b's (t0-span+b*dt, t0-span+(b+1)*dt].
        base = t0 - FORCE_HIST_SPAN_S + b * dt
        lp.push_force(np.full(ch, float(b)), t=base + 0.25 * dt)
        lp.push_force(np.full(ch, float(b) + 1.0), t=base + 1.0 * dt)
    fh = lp.build_batch()[0]["force_history"].numpy()[0]  # (FORCE_HIST_LEN, ch)
    assert fh.shape == (FORCE_HIST_LEN, ch)
    expected = np.arange(FORCE_HIST_LEN, dtype=np.float64) + 0.5
    np.testing.assert_allclose(fh[:, 0], expected, atol=1e-9)


def test_snapshot_display_images_roundtrips_pushed_frame(tmp_path, monkeypatch):
    """The viser 'conditioning images' panel must show the EXACT frames the model sees:
    snapshot_display_images denormalizes the stored [-1,1] tensors back to the uint8 RGB
    that was pushed, and returns To frames oldest-first (last == t0)."""
    bundle = _make_bundle(tmp_path, monkeypatch, "flow")
    import fuji.s1_policy.serving.hl_online as hl_online
    from tests.test_inference import _StubEncoder

    monkeypatch.setattr(hl_online, "_build_model", lambda algo, block: _flow_pair(block, _StubEncoder))
    loaded = load_hl_model(bundle.checkpoint, "flow", device="cpu")
    lp = LivePlanner(loaded)
    # A known frame with distinct per-channel values survives the /255→norm→denorm round-trip.
    frame = np.zeros((224, 224, 3), dtype=np.uint8)
    frame[..., 0] = 10
    frame[..., 1] = 128
    frame[..., 2] = 245
    lp.push_image(frame)
    disp = lp.snapshot_display_images()
    assert len(disp) == loaded.obs_history_len
    for im in disp:
        assert im.shape == (224, 224, 3) and im.dtype == np.uint8
    # The newest slot (t0, last) is the frame we just pushed (within rounding).
    np.testing.assert_allclose(disp[-1][0, 0], [10, 128, 245], atol=1)


def test_reanchor_to_measured_ee_is_clutch_base(tmp_path, monkeypatch):
    """The VIVE-clutch contract: after committing the LEFT anchor to a MEASURED EE pose,
    decoding a chunk yields 'measured pose + delta' — step-0 lands ~on the measured pose
    (no fly), and re-anchoring to a DIFFERENT measured pose re-bases the SAME delta there
    (press-start-from-current-pose), never flying to a fixed absolute target."""
    layout = get_layout("left_only")
    slots = layout.slots("left")
    H = 4
    # A pure EE-local +x reach delta, identity rotation (what the model would emit).
    pred = np.zeros((H, layout.action_dim))
    pred[:, slots.rot6d] = _IDENTITY_ROT6D
    pred[:, slots.pos] = np.linspace([0, 0, 0], [0.05, 0, 0], H)

    def decode_at(anchor_pos, anchor_rot6d):
        return decode_chunk(
            pred, layout, np.array([anchor_pos]), np.array([anchor_rot6d]),
            force_enabled=False, gate_n=6.0,
        ).role("left").position

    # Measured pose A: step-0 sits on the measured position (identity delta at step 0).
    posA = np.array([0.24, 0.25, -0.17])
    absA = decode_at(posA, _IDENTITY_ROT6D)
    np.testing.assert_allclose(absA[0], posA, atol=1e-9)  # NO fly: start from here

    # Move the robot to pose B; the SAME delta re-bases on B, not on A.
    posB = np.array([0.21, 0.25, -0.13])
    absB = decode_at(posB, _IDENTITY_ROT6D)
    np.testing.assert_allclose(absB[0], posB, atol=1e-9)  # clutch: from current pose
    assert np.linalg.norm(absA[0] - absB[0]) > 0.01  # different base, same delta

    # And LivePlanner.commit_role is the seam the server uses to set that measured anchor.
    bundle = _make_bundle(tmp_path, monkeypatch, "flow")
    import fuji.s1_policy.serving.hl_online as hl_online
    from tests.test_inference import _StubEncoder

    monkeypatch.setattr(hl_online, "_build_model", lambda algo, block: _flow_pair(block, _StubEncoder))
    loaded = load_hl_model(bundle.checkpoint, "flow", device="cpu")
    lp = LivePlanner(loaded)
    lp.seed_anchor(np.array([[0.0, 0.0, 0.0]]), np.array([_IDENTITY_ROT6D]))
    lp.commit_role("left", posB, _IDENTITY_ROT6D)   # <- the resync call
    _batch, ap, ar = lp.build_batch()
    np.testing.assert_allclose(ap[0], posB, atol=1e-12)
    np.testing.assert_allclose(ar[0], _IDENTITY_ROT6D, atol=1e-12)


def test_wiremapper_holds_right_head_at_measured_not_home():
    """The pre-arm-gate fix: full_wire must hold RIGHT+HEAD at the MEASURED hold pose (set via
    set_hold_wire from live FK), not at the frozen home_anchor. Otherwise a cross-policy home
    (0.31 m off, the observed bug) would (a) fail the low-level pre-arm pose gate and (b) fly
    the un-predicted right arm to the stale target on execute."""
    from scripts.hl_serving import WireMapper

    # A minimal home_anchor with DISTINCT home vs measured R/H so a regression is visible.
    home = {
        "home_pos_pelvis_3pt": [[0.22, 0.22, -0.16], [0.22, -0.22, -0.16], [0.0, 0.0, 0.39]],
        "home_orn_quat_xyzw_3pt": [[0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 1]],
        "vr_3point_offsets": [[0.18, -0.025, 0.0], [0.18, 0.025, 0.0], [0.0, 0.0, 0.35]],
    }
    m = WireMapper(home)
    left_tcp, left_rot6d = m.left_home_tcp_anchor()

    # Before any sync: R/H fall at the home anchor (back-compat / RECORD-off fallback).
    wp0, _wo0 = m.full_wire(left_tcp, left_rot6d)
    np.testing.assert_allclose(wp0[3:6], home["home_pos_pelvis_3pt"][1], atol=1e-9)  # RIGHT=home
    np.testing.assert_allclose(wp0[6:9], home["home_pos_pelvis_3pt"][2], atol=1e-9)  # HEAD=home

    # After syncing a MEASURED wire 0.31 m away: R/H track the measured pose, LEFT still comes
    # from the model TCP (overwritten every call), never from the stale hold slot 0.
    meas_pos9 = np.array([0.99, 0.99, 0.99, 0.53, -0.22, -0.16, 0.05, 0.0, 0.40])
    meas_orn12 = np.array([0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1], dtype=float)
    m.set_hold_wire(meas_pos9, meas_orn12)
    wp1, _wo1 = m.full_wire(left_tcp, left_rot6d)
    np.testing.assert_allclose(wp1[3:6], meas_pos9[3:6], atol=1e-9)   # RIGHT = measured
    np.testing.assert_allclose(wp1[6:9], meas_pos9[6:9], atol=1e-9)   # HEAD  = measured
    np.testing.assert_allclose(wp1[0:3], wp0[0:3], atol=1e-9)         # LEFT unchanged by hold
    assert np.linalg.norm(wp1[3:6] - wp0[3:6]) > 0.1                  # genuinely moved off home


def test_frame_seam_is_derived_from_the_dataset_extrinsic():
    """The seam ``S`` (``R_stored = R_wrist @ S``) must be DERIVED from the served dataset's own
    ``T_tracker_tcp``, never hardcoded.

    History this locks out: the seam used to be a constant ``Rx(180) = diag(1,-1,-1)``. A 180°
    flip really was needed for pre-2026-08-06 data, but about the wrong axis — the true puck→wrist
    relabel M is 180° about (1,-1,0)/sqrt2, 45° from +x, so the two composed to a spurious 90°
    about the wrist z. Measured on clean94 that cost 47.96 deg mean tool-attack-angle error and
    2.58 deg of proprio error. No constant can be right for both conventions, which is the
    property asserted here.
    """
    from scripts.hl_serving import (
        _PUCK_TO_WRIST_RELABEL, _seam_from_extrinsic,
    )

    M = _PUCK_TO_WRIST_RELABEL
    # M is the measured (-y,-x,-z) relabel: a proper rotation, its own inverse, 180 deg about an
    # axis 45 deg away from +x (which is precisely why the old Rx(180) left a 90 deg residue).
    assert np.isclose(np.linalg.det(M), 1.0)
    np.testing.assert_allclose(M @ M, np.eye(3), atol=1e-12)
    np.testing.assert_allclose(M, np.array([[0, -1, 0], [-1, 0, 0], [0, 0, -1]], float), atol=0)

    def _t(rot, trans=(0.0, -0.225, 0.05)):
        out = np.eye(4)
        out[:3, :3] = rot
        out[:3, 3] = trans
        return out

    # CURRENT convention (extrinsic rotation == M): the stored pose IS the wrist-oriented
    # fingertip TCP, so the seam is EXACTLY identity and nothing is applied at the boundaries.
    np.testing.assert_allclose(_seam_from_extrinsic(_t(M)), np.eye(3), atol=1e-12)
    # PRE-FIX convention (identity extrinsic rotation): the stored orientation is the raw puck, so
    # the seam is M itself -- 180 deg, and NOT equal to the old Rx(180) constant.
    seam_old = _seam_from_extrinsic(_t(np.eye(3), (0.225, 0.0, -0.05)))
    np.testing.assert_allclose(seam_old, M, atol=1e-12)
    assert not np.allclose(seam_old, np.diag([1.0, -1.0, -1.0]), atol=1e-6), \
        "the pre-fix seam is NOT Rx(180) — that conflation is the 90 deg bug this test guards"
    # The two conventions therefore need DIFFERENT seams: no constant can serve both.
    assert not np.allclose(_seam_from_extrinsic(_t(M)), seam_old, atol=1e-6)
    # A missing extrinsic applies NO seam (safe default), and a corrupt one is a hard error
    # rather than a guessed orientation on the wire.
    np.testing.assert_allclose(_seam_from_extrinsic(None), np.eye(3), atol=0)
    for bad in (_t(np.full((3, 3), 0.5)), np.eye(3), _t(np.diag([1.0, 1.0, np.nan]))):
        with pytest.raises(ValueError):
            _seam_from_extrinsic(bad)


def test_frame_seam_boundaries_round_trip_and_preserve_the_reach_axis():
    """The two boundary converters must invert each other for a GENERAL seam, and must leave the
    +x reach lever alone.

    The transpose in :func:`_model_rot6d_to_urdf` is load-bearing and easy to lose: the old
    ``Rx(180)`` was self-inverse, so ONE shared implementation happened to work for both
    directions. With a general seam, reusing ``S`` instead of ``S.T`` DOUBLES the rotation instead
    of undoing it, and the round-trip below is what catches that."""
    import scripts.hl_serving as hs
    from fuji.utils.rot6d import rot6d_to_rotmat, rotmat_to_rot6d

    rng = np.random.default_rng(7)
    R_urdf = np.linalg.qr(rng.standard_normal((3, 3)))[0]
    if np.linalg.det(R_urdf) < 0:
        R_urdf[:, 0] *= -1.0
    r6_urdf = rotmat_to_rot6d(R_urdf)

    for label, seam in (("identity (current)", np.eye(3)),
                        ("relabel M (pre-fix)", hs._PUCK_TO_WRIST_RELABEL),
                        ("legacy Rx180", np.diag([1.0, -1.0, -1.0]))):
        prev = hs._MODEL_SEAM.copy()
        try:
            hs._set_frame_seam(seam, f"test:{label}")
            np.testing.assert_allclose(
                hs._model_rot6d_to_urdf(hs._urdf_rot6d_to_model(r6_urdf)), r6_urdf, atol=1e-9,
                err_msg=f"URDF->model->URDF must round-trip for seam {label}")
            np.testing.assert_allclose(
                hs._urdf_rot6d_to_model(hs._model_rot6d_to_urdf(r6_urdf)), r6_urdf, atol=1e-9)
            # The seam right-multiplies, so the MODEL frame's basis is the wrist basis re-mixed by
            # S; the +x reach lever _TCP_OFFSET is preserved because every seam we use fixes +x up
            # to sign flips of the OTHER axes -- this is why positions never showed the axis bug.
            R_model = rot6d_to_rotmat(hs._urdf_rot6d_to_model(r6_urdf))
            np.testing.assert_allclose(R_model, R_urdf @ seam, atol=1e-9)
        finally:
            hs._set_frame_seam(prev, "test:restore")


def test_identity_seam_makes_the_decode_a_pure_passthrough():
    """On the CURRENT convention the whole model/URDF seam must vanish: a decoded chunk goes to
    the wire in the frame it was decoded in, with no hidden rotation.

    This is the property the operator asked for -- the low level is fed the raw-tracker action put
    through ONE extrinsic and nothing else -- so it gets its own assertion rather than being
    implied by the round-trip above."""
    import scripts.hl_serving as hs
    from fuji.utils.rot6d import rotmat_to_rot6d

    rng = np.random.default_rng(11)
    R = np.linalg.qr(rng.standard_normal((3, 3)))[0]
    if np.linalg.det(R) < 0:
        R[:, 0] *= -1.0
    r6 = rotmat_to_rot6d(R)
    prev = hs._MODEL_SEAM.copy()
    try:
        hs._set_frame_seam(np.eye(3), "test:identity")
        np.testing.assert_allclose(hs._urdf_rot6d_to_model(r6), r6, atol=1e-12)
        np.testing.assert_allclose(hs._model_rot6d_to_urdf(r6), r6, atol=1e-12)
    finally:
        hs._set_frame_seam(prev, "test:restore")


@pytest.mark.parametrize("seam_name", ["identity", "relabel_M"])
def test_frame_flip_applied_at_seed_and_decode_wire_boundaries(seam_name):
    """INTEGRATION guard for the seam's CALL SITES (not just the helper math): a regression that
    dropped the conversion at the anchor-commit boundary, or published the model-frame orn straight
    to the URDF wire, must be caught even though the helpers themselves are correct.

      * _seed_planner_anchor must store the LEFT decode anchor in the MODEL frame
        (= home URDF orn @ S), not the raw URDF home orn.
      * a zero-delta chunk decoded against that model-frame anchor, then routed to the wire the way
        _infer does (_model_rot6d_to_urdf -> WireMapper.full_wire), must reproduce the ORIGINAL home
        URDF wire orn — i.e. the two conversions compose to identity through the real anchor + real
        decode, so the LL sees the same URDF pose the pre-arm/measured path publishes.

    Run under BOTH a non-trivial seam and the identity seam the current convention resolves to.
    Under identity the seed legitimately EQUALS the raw URDF orn, so a "seed != raw URDF" assertion
    would be wrong there; the invariant that holds for every seam is "seed == URDF @ S", which is
    what is asserted. The non-trivial case is what keeps that from being vacuous.
    """
    import scripts.hl_serving as hs
    from scripts.hl_serving import (
        WireMapper, _model_rot6d_to_urdf, _seed_planner_anchor, _urdf_rot6d_to_model,
    )
    from fuji.s1_policy.data.transforms import delta_ee_to_absolute
    from fuji.utils.rot6d import rot6d_to_rotmat, rotmat_to_rot6d, quat_xyzw_to_rotmat

    seam = np.eye(3) if seam_name == "identity" else hs._PUCK_TO_WRIST_RELABEL
    _prev_seam = hs._MODEL_SEAM.copy()
    hs._set_frame_seam(seam, f"test:{seam_name}")

    # A non-trivial home LEFT wrist orn (xyzw) so the flip is observable; matches home_anchor style.
    home = {
        "home_pos_pelvis_3pt": [[0.223, 0.218, -0.155], [0.223, -0.218, -0.155], [0.0, 0.0, 0.394]],
        "home_orn_quat_xyzw_3pt": [
            [0.089704, 0.386436, 0.046281, 0.916776],  # LEFT (from the real home_anchor.json)
            [-0.089704, 0.386436, -0.046281, 0.916776],
            [0.0, 0.0, 0.0, 1.0],
        ],
        "vr_3point_offsets": [[0.18, -0.025, 0.0], [0.18, 0.025, 0.0], [0.0, 0.0, 0.35]],
    }
    mapper = WireMapper(home)
    _, left_home_rot6d_urdf = mapper.left_home_tcp_anchor()  # URDF wrist frame

    # --- seed boundary: a tiny stub planner just records what _seed_planner_anchor writes. ---
    class _StubPlanner:
        roles = ("left",)

        def __init__(self):
            self.seeded_pos = self.seeded_rot6d = None

        def seed_anchor(self, ap, ar):
            self.seeded_pos = np.asarray(ap, dtype=np.float64)
            self.seeded_rot6d = np.asarray(ar, dtype=np.float64)

    try:
        planner = _StubPlanner()
        _seed_planner_anchor(planner, mapper)
        # The stored LEFT anchor must be the MODEL frame = URDF @ S (for ANY seam).
        np.testing.assert_allclose(
            planner.seeded_rot6d[0], _urdf_rot6d_to_model(left_home_rot6d_urdf), atol=1e-9)
        np.testing.assert_allclose(
            rot6d_to_rotmat(planner.seeded_rot6d[0]),
            rot6d_to_rotmat(left_home_rot6d_urdf) @ seam, atol=1e-9)
        if seam_name != "identity":
            # With a real seam the seed must NOT be the raw URDF orn — this is the assertion that
            # would catch a dropped conversion at the seed boundary. (Under the identity seam the
            # two are equal BY DESIGN, so asserting inequality there would be wrong.)
            assert not np.allclose(planner.seeded_rot6d[0], left_home_rot6d_urdf, atol=1e-6), \
                "seed stored the RAW URDF orn — the URDF->model conversion was dropped"

        # --- decode+wire boundary: zero-delta chunk against the model-frame anchor, then to the
        # wire exactly as _infer does (model->URDF, then full_wire). Must recover the home URDF wire.
        anchor_pos, anchor_rot6d_model = planner.seeded_pos[0], planner.seeded_rot6d[0]
        I6 = rotmat_to_rot6d(np.eye(3))
        abs_pos, abs_rot6d_model = delta_ee_to_absolute(
            np.zeros(3)[None], I6[None], anchor_pos, anchor_rot6d_model)  # step-0 == anchor (model)
        wire_orn_urdf = _model_rot6d_to_urdf(abs_rot6d_model[0])  # what _infer publishes
        # It must equal the RAW home URDF orn (the frame the LL's _compute_vr_fk gates/servos in).
        np.testing.assert_allclose(
            rot6d_to_rotmat(wire_orn_urdf),
            quat_xyzw_to_rotmat(np.asarray(home["home_orn_quat_xyzw_3pt"][0])), atol=1e-7)
        # And the wire the LL receives (full_wire LEFT slot) matches the measured/home publish path.
        wp, wo = mapper.full_wire(abs_pos[0], wire_orn_urdf)
        hp, ho = mapper.full_wire(*mapper.left_home_tcp_anchor())
        np.testing.assert_allclose(wo[0:4], ho[0:4], atol=1e-7)  # LEFT wire quat == home URDF
    finally:
        hs._set_frame_seam(_prev_seam, "test:restore")


def test_the_rendered_mesh_pose_matches_the_triad_and_anchor_pose():
    """The MESH, the triad cluster and the decode ANCHOR must all be the low level's READY pose.

    THE BUG THIS PINS (reported from a screenshot, caught by no test): the mesh was posed at the
    mink STAND keyframe (``ThreePointIK.home_q()``) while ``/follow``, the three triads and the
    decode anchor all sit at ``home_anchor.default_qpos``. Those are different arm configurations —
    **166.3 mm and 47.08 deg apart** at ``left_wrist_yaw_link`` — so the labelled cluster and the
    green trajectory hung in space beside the rendered arm and read exactly like a frame bug. The
    anchor was never wrong; only the mesh's pose was.

    Asserted here as the INEQUALITY of the two poses (so the two-poses hazard stays visible) plus
    the EQUALITY that must hold: the ready-pose wrist FK == the wrist implied by
    ``home_anchor``'s own vr_3point wire, which is what ``WireMapper`` seeds the anchor from. That
    equality is the real invariant — it ties the rendered pose to the commanded one through two
    independent routes (URDF FK vs the stored wire), so they cannot drift apart silently.

    Skipped when the URDF/yourdfpy/humi_ik render stack is unavailable (this is a headless test of
    a render invariant, not of the render itself)."""
    yourdfpy = pytest.importorskip("yourdfpy")
    import json
    import os

    import scripts.hl_serving as hs
    from fuji.s1_policy.serving.live_robot import N_SDK_DOF, build_sdk_to_urdf_index, remap_dof_to_urdf
    from fuji.utils.rot6d import quat_xyzw_to_rotmat

    urdf_path = os.path.join(
        hs._UMI, "robot_assets", "g1_yam", "g1_29dof_yam.urdf")
    if not os.path.exists(urdf_path) or not os.path.exists(hs._HOME_ANCHOR_PATH):
        pytest.skip("yam URDF / home_anchor.json not checked out")

    home = json.load(open(hs._HOME_ANCHOR_PATH))
    ready = np.asarray(home.get("default_qpos", ()), dtype=float)
    assert ready.shape == (N_SDK_DOF,), "home_anchor.default_qpos must be the 29-DOF SDK vector"

    urdf = yourdfpy.URDF.load(urdf_path, load_meshes=False, build_scene_graph=True)
    remap = build_sdk_to_urdf_index(list(urdf.actuated_joint_names))
    assert not remap.missing_sdf, remap.missing_sdf

    # The pose the mesh is now initialized to (and the pose /follow + the triads are seeded at).
    urdf.update_cfg(remap_dof_to_urdf(ready, remap))
    ready_T = np.asarray(
        urdf.get_transform(frame_to="left_wrist_yaw_link", frame_from="pelvis"), dtype=float)

    # The wrist implied by home_anchor's own vr_3point wire — the route WireMapper seeds through.
    wire_R = quat_xyzw_to_rotmat(np.asarray(home["home_orn_quat_xyzw_3pt"][0], dtype=float))
    wire_wrist = np.asarray(home["home_pos_pelvis_3pt"][0], dtype=float) - wire_R @ hs._WIRE_OFFSET
    np.testing.assert_allclose(ready_T[:3, 3], wire_wrist, atol=1e-5)
    np.testing.assert_allclose(ready_T[:3, :3], wire_R, atol=1e-5)

    # And the mink STAND pose is genuinely FAR from it — the two-poses hazard, kept visible so this
    # test cannot silently pass by both routes collapsing onto the same value.
    humi = pytest.importorskip("g1_three_point_ik", reason="humi_ik not on sys.path")
    ik = humi.ThreePointIK(config_name="proposal", reach_scale=1.0, active_roles=("left",))
    from g1_right_hand_ik import YamConfigMapper, yam_actuated_joint_names

    mapper = YamConfigMapper(ik._humi_joint_names, yam_actuated_joint_names(urdf_path))
    urdf.update_cfg(mapper.to_yam(ik.home_q(), left_jaw_m=0.0))
    stand_T = np.asarray(
        urdf.get_transform(frame_to="left_wrist_yaw_link", frame_from="pelvis"), dtype=float)
    gap_mm = float(np.linalg.norm(stand_T[:3, 3] - ready_T[:3, 3])) * 1000.0
    gap_deg = float(np.degrees(np.arccos(np.clip(
        (np.trace(stand_T[:3, :3] @ ready_T[:3, :3].T) - 1.0) / 2.0, -1.0, 1.0))))
    assert gap_mm > 100.0 and gap_deg > 30.0, (
        f"the mink stand and ready poses are only {gap_mm:.1f} mm / {gap_deg:.1f} deg apart; this "
        "test's premise (that posing the mesh from the wrong one is visibly wrong) no longer holds")


def test_zero_arms_seed_pose_matches_the_low_levels_ready_pose():
    """The seed pose must be the pose the LOW LEVEL actually holds: ``default_dof_angles`` with all
    **14** arm joints at zero.

    NOT cosmetic. ``3_run_policy_vive_force.sh`` runs ``HOLOSOMA_INIT_READY_POSE=1``, so after 'A'
    ``holosoma_wbt::_maybe_apply_init_ready_pose`` pins the INIT target AND the frozen VIVE hold to
    that pose, and ``_apply_hold_dof_override`` writes it into ``_motion_command_t[0,:29]`` — the
    very array the PRE-ARM POSE GATE reads via ``_compute_vr_fk``. Seeding from the raw crouch
    ``default_qpos`` instead puts the broadcast wire ~310 mm/hand from the gate reference against a
    70 mm tolerance, so arming would REFUSE.

    Pinned here:
      (1) exactly the 14 ARM joints change (legs/waist keep the crouch, so the render still stands
          on the grid — an all-29-zero pose would move the feet), and BOTH arms are included;
      (2) every zeroed joint is inside its URDF limit;
      (3) both forearms come LEVEL, and the left TCP is clear of the torso;
      (4) the two candidate poses are far enough apart to matter vs the gate tolerance.
    """
    yourdfpy = pytest.importorskip("yourdfpy")
    import os

    import scripts.hl_serving as hs
    from fuji.s1_policy.serving.live_robot import build_sdk_to_urdf_index, remap_dof_to_urdf

    urdf_path = os.path.join(hs._UMI, "robot_assets", "g1_yam", "g1_29dof_yam.urdf")
    if not os.path.exists(urdf_path) or not os.path.exists(hs._HOME_ANCHOR_PATH):
        pytest.skip("yam URDF / home_anchor.json not checked out")

    ready = hs._display_seed_dof(False)
    flat = hs._display_seed_dof(True)
    assert ready is not None and flat is not None

    # (1) exactly the 14 ARM indices differ, and they are all zero in the seeded pose. The count and
    # the both-arms coverage are asserted because the low level zeroes 14 (not 7) — a LEFT-only
    # version would leave the RIGHT hand 310 mm from the gate reference.
    changed = set(np.flatnonzero(~np.isclose(ready, flat)).tolist())
    assert changed <= set(hs._ARM_SDK_IDX), "a NON-arm joint was modified"
    np.testing.assert_allclose(flat[list(hs._ARM_SDK_IDX)], 0.0, atol=0)
    assert len(hs._ARM_SDK_IDX) == 14, hs._ARM_SDK_IDX
    from fuji.s1_policy.serving.live_robot import G1_29DOF_SDK_DOF_NAMES as _NAMES

    _seeded = {_NAMES[i] for i in hs._ARM_SDK_IDX}
    assert sum("left" in n for n in _seeded) == 7 and sum("right" in n for n in _seeded) == 7, _seeded

    urdf = yourdfpy.URDF.load(urdf_path, load_meshes=False, build_scene_graph=True)
    remap = build_sdk_to_urdf_index(list(urdf.actuated_joint_names))
    if remap.missing_sdf:
        pytest.skip(f"URDF lacks SDK joints {remap.missing_sdf}")

    def _probe(dof):
        urdf.update_cfg(remap_dof_to_urdf(dof, remap))
        out = {}
        for side in ("left", "right"):
            T = np.asarray(urdf.get_transform(
                frame_to=f"{side}_wrist_yaw_link", frame_from="pelvis"), dtype=float)
            # elevation of the wrist +x reach axis: 0 == level, negative == nose-down
            out[side] = (T, float(np.degrees(np.arcsin(np.clip(T[2, 0], -1.0, 1.0)))))
        out["sole"] = min(
            float(np.asarray(urdf.get_transform(frame_to=link, frame_from="pelvis"),
                             dtype=float)[2, 3])
            for link in urdf.link_map if "ankle_roll" in link)
        return out

    crouch = _probe(ready)
    level = _probe(flat)

    # (3) BOTH forearms level in the seeded pose, both drooping in the crouch pose. 0.01 deg rather
    # than 0: with the joints at zero the wrist rotation is identity to 2e-4, leaving a -0.0031 deg
    # residual from the URDF's own fixed shoulder-mount origins (a real, tiny hardware offset).
    # Asserting exact zero would be asserting the URDF is idealised, which it is not.
    for side in ("left", "right"):
        assert crouch[side][1] < -30.0, f"{side} crouch should be nose-down: {crouch[side][1]:.2f}"
        assert abs(level[side][1]) < 0.01, f"{side} should be LEVEL: {level[side][1]:.4f} deg"
    # (1) the stance is untouched: identical sole height, unlike an all-29-joints-zero pose.
    assert np.isclose(crouch["sole"], level["sole"], atol=1e-9), (crouch["sole"], level["sole"])
    # (2) limits, for all 14.
    for joint_name in urdf.actuated_joint_names:
        if not any(k in joint_name for k in ("shoulder", "elbow", "wrist")):
            continue
        limit = urdf._joint_map[joint_name].limit
        if limit is not None and limit.lower is not None:
            assert limit.lower <= 0.0 <= limit.upper, f"{joint_name} cannot be zero: {limit}"
    # (3) left TCP clear of the torso (pelvis x well positive).
    flat_T = level["left"][0]
    flat_tcp = flat_T[:3, 3] + flat_T[:3, :3] @ hs._TCP_OFFSET
    assert flat_tcp[0] > 0.30, flat_tcp
    # (4) the two poses are far apart RELATIVE TO THE GATE TOLERANCE (0.07 m), per hand — this is
    # what makes seeding at the wrong one an arming failure rather than a cosmetic difference. Uses
    # the vr_3point CONTACT points, which is what the gate actually compares.
    for i, side in ((0, "left"), (1, "right")):
        c_contact = crouch[side][0][:3, 3] + crouch[side][0][:3, :3] @ mapper_offsets()[i]
        l_contact = level[side][0][:3, 3] + level[side][0][:3, :3] @ mapper_offsets()[i]
        assert np.linalg.norm(l_contact - c_contact) > 0.07, (
            f"{side} contact gap {np.linalg.norm(l_contact - c_contact):.4f} m is within the "
            "pre-arm gate tolerance, so this test no longer proves the seed pose matters")


def mapper_offsets():
    """``home_anchor.vr_3point_offsets`` — the wrist/torso->contact levers the low level's
    ``_compute_vr_fk`` (and therefore its pre-arm gate) applies."""
    import json

    import scripts.hl_serving as hs

    return np.asarray(json.load(open(hs._HOME_ANCHOR_PATH))["vr_3point_offsets"],
                      dtype=float).reshape(3, 3)


def test_seed_anchor_rejects_wrong_shape(tmp_path, monkeypatch):
    bundle = _make_bundle(tmp_path, monkeypatch, "flow")
    import fuji.s1_policy.serving.hl_online as hl_online
    from tests.test_inference import _StubEncoder

    monkeypatch.setattr(hl_online, "_build_model", lambda algo, block: _flow_pair(block, _StubEncoder))
    loaded = load_hl_model(bundle.checkpoint, "flow", device="cpu")
    lp = LivePlanner(loaded)
    with pytest.raises(ValueError, match="seed_anchor expects"):
        lp.seed_anchor(np.zeros((2, 3)), np.zeros((2, 6)))  # left_only has 1 role


# --- helpers: build the stub-encoder-backed models for the tiny fixtures ------
def _flow_pair(config_block, stub_encoder_cls):
    from fuji.s1_policy.inference.common import dataclass_config
    from fuji.s1_policy.model.dp_flow import DPFlowConfig, DPFlowModel

    config = dataclass_config(DPFlowConfig, config_block)
    return config, DPFlowModel(config, encoder=stub_encoder_cls())


def _diffusion_pair(config_block, stub_encoder_cls):
    from fuji.s1_policy.inference.common import dataclass_config
    from fuji.s1_policy.model.dp_diffusion import DPDiffusionConfig, DPDiffusionModel

    config = dataclass_config(DPDiffusionConfig, config_block)
    return config, DPDiffusionModel(config, encoder=stub_encoder_cls())
