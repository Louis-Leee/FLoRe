"""Recorded-episode REPLAY on the FIVE-POINT page (``scripts/hl_serving_5pt.py --replay-zarr``, 2026-09-19).

What is pinned here is the CONTRACT that makes "the only difference is where the chunk came from"
true on this wire:

* a recorded chunk goes through the SAME ``decode_chunk`` → ``FivePointDirectWire.plan`` → clutch
  path a model chunk does, so the four held slots stay on the ready frame, step 0 lands ON the
  gate reference, and the recorded wrench reaches ``hl_5pt.force_command`` as ``-F_meas``;
* the page's ``_predict`` branches to the recorded source with ONE check, ``_drip`` advances the
  cursor by what it executed, ▶▶ plays it on the sequential cadence and names the run ``replay``
  to the recorder, Reset rewinds, and nothing exists on the page without ``--replay-zarr``;
* the launcher's ``REPLAY`` / ``REPLAY_ZARR`` / ``REPLAY_EPISODE`` / ``REPLAY_MODE`` knobs.
"""

from __future__ import annotations

import os
import re
import subprocess

import numpy as np
import pytest

from fuji.s1_policy.data.layout import get_layout
from fuji.s1_policy.serving import hl_5pt, hl_dense, hl_replay
from fuji.s1_policy.serving.hl_online import decode_chunk
from fuji.utils.rot6d import rotmat_to_rot6d

zarr = pytest.importorskip("zarr")

from tests.test_hl_dataset import _write_hl_zarr  # noqa: E402

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PAGE = os.path.join(_ROOT, "scripts", "hl_serving_5pt.py")
_LAUNCHER = os.path.join(_ROOT, "scripts", "run_hl_serving_5pt.sh")
_RUNBOOK = os.path.join(_ROOT, "docs", "RUNBOOK_hl_serving_5pt.md")

N = len(hl_dense.FIVE_POINT_BODIES)
READY_POS = np.arange(3 * N, dtype=np.float64) * 0.01 + 0.1
READY_ORN = np.tile(np.array([0.0, 0.0, 0.0, 1.0]), N)
LAY = get_layout("left_only")


def _src():
    with open(_PAGE, encoding="utf-8") as fh:
        return fh.read()


def _held_for(layout, anchor_pos_by_role):
    return {r: (anchor_pos_by_role[r], rotmat_to_rot6d(np.eye(3)))
            for r in hl_5pt.WIRE_ROLES if r not in layout.roles}


def _wire(held):
    clutch = hl_5pt.build_clutch(READY_POS, READY_ORN, post=hl_dense.PostAnchorConfig(mode="command"), carry="all")
    return hl_5pt.FivePointDirectWire(hl_dense.KeypointReadout(np.zeros(3)), READY_POS, READY_ORN,
                                      clutch=clutch, held=held)


# ---------------------------------------------------------------------------
# the recorded chunk on the five-slot wire
# ---------------------------------------------------------------------------
@pytest.fixture()
def left_zarr(tmp_path):
    return _write_hl_zarr(tmp_path / "replay_left.zarr.zip")


def test_a_recorded_left_only_chunk_publishes_five_slots_with_step0_on_the_ready_frame(left_zarr):
    """The page's replay path, end to end below the page: recorded chunk → decode against the
    running anchor → validity → the wire with the four held slots → the clutch. Step 0 is the gate
    reference, the held slots never move, and the LEFT slot moves by the RECORDED deltas."""
    src = hl_replay.RecordedEpisodeSource(left_zarr, LAY)
    rc = src.chunk_at(0, 2)
    anchor_pos = np.array([[0.30, 0.20, 0.10]])
    anchor_rot6d = rotmat_to_rot6d(np.eye(3))[None]
    chunk = decode_chunk(rc.actions, LAY, anchor_pos, anchor_rot6d, force_enabled=True, gate_n=0.0)
    chunk, note = hl_replay.apply_step_validity(chunk, rc.force_step_valid, rc.gripper_step_valid)
    assert note == {"force": {}, "gripper": {}, "gripper_dropped": []}
    held_anchor = {r: np.array([0.5, i * 0.1, 0.9]) for i, r in enumerate(hl_5pt.WIRE_ROLES)}
    wire = _wire(_held_for(LAY, held_anchor))
    assert wire.held_roles == ("right", "head", "left_foot", "right_foot")
    plan = wire.plan(chunk)
    assert plan["n"] == rc.n_valid
    pos = plan["pos"].reshape(plan["n"], N, 3)
    ready = READY_POS.reshape(N, 3)
    # step 0 ON the ready frame for every slot (the clutch), the held slots constant for the whole chunk
    np.testing.assert_allclose(pos[0], ready, atol=1e-9)
    for i in range(1, N):
        np.testing.assert_allclose(pos[:, i], np.tile(ready[i], (plan["n"], 1)), atol=1e-9)
    # the LEFT slot moves by exactly the RECORDED body-frame deltas (identity anchor rotation here)
    rec_delta = rc.actions[:, LAY.slots("left").pos]
    np.testing.assert_allclose(pos[:, 0] - ready[0], rec_delta, atol=1e-9)
    assert np.linalg.norm(pos[-1, 0] - pos[0, 0]) > 0.0
    ok, why = wire.gate_report(wire.frame(plan, 0))
    assert ok, why


def test_the_recorded_wrench_becomes_minus_F_meas_on_the_left_and_zero_on_the_right(left_zarr):
    src = hl_replay.RecordedEpisodeSource(left_zarr, LAY)
    rc = src.chunk_at(1, 4)
    chunk = decode_chunk(rc.actions, LAY, np.zeros((1, 3)), rotmat_to_rot6d(np.eye(3))[None],
                         force_enabled=True, gate_n=0.0)
    rec_force = rc.actions[3, LAY.slots("left").force]                 # raw physical, bota_sensor frame
    np.testing.assert_allclose(chunk.role("left").wrench_pred[3], rec_force)
    fcmd = hl_5pt.force_command({"left": chunk.role("left").wrench_pred[3]}, gate_n=1.5)
    assert fcmd is not None and fcmd.shape == (6,)
    np.testing.assert_allclose(fcmd[:3], -rec_force[:3], atol=1e-9)    # -F_meas, the same sign as a policy chunk
    np.testing.assert_allclose(fcmd[3:], 0.0)                          # the held hand reads ZERO force
    # an unmeasured row is zeroed by the validity step, and zero is what the gate turns into "no F^cmd"
    fsv = {"left": np.ones(rc.n_valid, dtype=bool)}
    fsv["left"][3] = False
    blanked, _ = hl_replay.apply_step_validity(chunk, fsv, {})
    assert hl_5pt.force_command({"left": blanked.role("left").wrench_pred[3]}, gate_n=1.5) is None


def test_the_recorded_jaw_reaches_the_chunk_in_metres(left_zarr):
    src = hl_replay.RecordedEpisodeSource(left_zarr, LAY)
    rc = src.chunk_at(0, 0)
    chunk = decode_chunk(rc.actions, LAY, np.zeros((1, 3)), rotmat_to_rot6d(np.eye(3))[None],
                         force_enabled=True, gate_n=0.0)
    np.testing.assert_allclose(chunk.role("left").gripper_m, 0.0475, atol=1e-6)   # the fixture's stored width


# ---------------------------------------------------------------------------
# the page: one check in _predict, the cursor in _drip, ▶▶, Reset, the recorder, the panel
# ---------------------------------------------------------------------------
def _page_module():
    import importlib.util
    import sys

    spec = importlib.util.spec_from_file_location("_hl5_replay_wiring", _PAGE)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_hl5_replay_wiring"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_the_parser_has_the_replay_flags_with_the_reviewed_defaults():
    parser = _page_module().build_parser()
    args = parser.parse_args(["--checkpoint", "x"])
    assert args.replay_zarr is None and args.replay_episode == 0 and args.replay_mode == "sequential"
    args = parser.parse_args(["--checkpoint", "x", "--replay-zarr", "checkpoint", "--replay-episode", "7",
                              "--replay-mode", "openloop"])
    assert args.replay_zarr == "checkpoint" and args.replay_episode == 7 and args.replay_mode == "openloop"
    with pytest.raises(SystemExit):
        parser.parse_args(["--checkpoint", "x", "--replay-mode", "stream"])
    help_text = parser.format_help()
    assert "recorded-episode replay" in help_text and "F^cmd" in help_text


def test_predict_branches_to_the_recorded_source_first_and_decodes_it_like_a_model_chunk():
    src = _src()
    predict = src[src.index("    def _predict() -> str:"):src.index("    # ---------------- execute ----------------")]
    body = predict[predict.index('"""', predict.index('"""') + 3) + 3:]      # after the docstring
    # the source question is answered FIRST; the ⏺ run recorder's `begin` (2026-09-20) sits between the
    # answer and the branch so a recorded-episode Predict opens a recording exactly as a model one does
    assert body.lstrip().startswith("replaying = _replay_active()")
    assert (body.index('taps.begin("replay" if replaying else "sequential")')
            < body.index("if replaying:\n            return _predict_replay()") < body.index("HL._obs_freshness("))
    rp = src[src.index("    def _predict_replay() -> str:"):src.index("    def _predict() -> str:")]
    # the SAME decode, ungated (the live gate is _drip's), then the validity blanking
    assert "decode_chunk(actions, loaded.layout, anchor_pos, anchor_rot6d, force_enabled=True, gate_n=0.0)" in rp
    assert "hl_replay.apply_step_validity(chunk, fsv, gsv)" in rp
    assert "planner.snapshot_anchor()" in rp
    # the same leash, the same re-base, the same plan/span/gate, the same taps and paths
    for want in ("_belief_bound()", "wire.sync(_record_frame(), _measured_force_n())", "plan = wire.plan(chunk)",
                 "hl_5pt.chunk_span(plan[\"pos\"]", "wire.gate_report(wire.frame(plan, 0))",
                 'taps.chunk(seq=stats["n"], cadence="replay"', 'source="replay"', "_draw_paths(chunk)",
                 "_draw_replay_paths(episode, frame)", "_queue_ik(chunk)"):
        assert want in rp, want
    # what a model chunk has and a recorded one must NOT: a sampler seed, the observation gates
    assert "seed_ctl.next()" not in rp and "_obs_range_gate()" not in rp and "HL._obs_freshness(" not in rp
    assert "seed=None" in rp
    # both modes, and the episode end rewinds rather than raising
    assert 'if mode == "openloop":' in rp and "replay.episode_plan(episode, frame)" in rp
    assert "replay.chunk_at(episode, frame)" in rp and "replay.next_usable_frame(episode, frame)" in rp
    assert 'taps.event("replay_end"' in rp and "replay.first_usable_frame(episode)" in rp


def test_drip_executes_the_whole_openloop_plan_and_advances_the_cursor_by_what_ran():
    src = _src()
    drip = src[src.index("    def _drip(out) -> int:"):src.index("    def _release_jaw_latches()")]
    assert 'n_want = out.get("n_exec") or exec_ctl["n"]' in drip
    assert "n_steps = max(1, min(int(n_want), int(plan[\"n\"])))" in drip
    # the cursor moves AFTER both commits, by TIME, onto a playable frame, only if nobody moved it
    assert drip.index("wire.commit(wire.frame(plan, last))") < drip.index('rp = out.get("replay")')
    for want in ('replay.advance_frame(rp["episode"], rp["frame"], last + 1)',
                 'replay.next_usable_frame(rp["episode"], nxt)',
                 'replay_ctl["episode"] == rp["episode"] and replay_ctl["frame"] == rp["frame"]',
                 'if rp.get("mode") == "openloop":'):
        assert want in drip, want
    # a model chunk carries no "replay" key and no "n_exec": the reviewed path, byte for byte
    predict = src[src.index("    def _predict() -> str:"):src.index("    # ---------------- execute ----------------")]
    assert 'staged["out"] = {"chunk": chunk, "plan": plan, "wire_src": wire_src, "seq": stats["n"]}' in predict


def test_the_run_button_plays_a_replay_on_the_sequential_cadence_and_names_it_to_the_recorder():
    src = _src()
    start = src[src.index("    def _auto_start() -> str:"):src.index("    def _auto_retire() -> str:")]
    assert "replaying = _replay_active()" in start
    assert "if replaying and streaming:" in start and "streaming = False" in start
    assert 'cadence = "replay" if replaying else ("stream" if streaming else "sequential")' in start
    assert "taps.wrap_run(target, cadence" in start
    assert 'auto_ctl["max_s"]' in start                      # the ceiling is named when the episode is longer
    # the sequential loop itself is untouched: it ends when _predict stages nothing (the episode end)
    loop = src[src.index("    def _run_to_completion() -> None:"):src.index("    # ---------------- the CONTINUOUS (HATO) rollout")]
    assert "replay" not in loop.lower().replace("the heartbeat replays it", "")


def test_reset_rewinds_the_recorded_episode_and_the_recorder_learns_the_source():
    src = _src()
    reset = src[src.index('if flags["reset"]:'):src.index("_rec_reset()   # the ⏺ button back")]
    assert "replay.first_usable_frame(_ep)" in reset and "_replay_seek(_ep, _first" in reset
    assert reset.index("seed_ctl.reset()") < reset.index("_replay_seek(_ep, _first")
    info = src[src.index("    def _rec_info() -> dict:"):src.index("    def _rec_git_sha() -> str:")]
    assert '"source": source, "replay": rp_info' in info
    assert '"replay_zarr": (None if replay is None else os.path.abspath(replay.path))' in src
    # the recorder's chunk record carries the source (the tap signature grew two OPTIONAL fields)
    from fuji.s1_policy.serving import rollout_record as R
    import inspect

    params = inspect.signature(R.RunRecorder.chunk).parameters
    assert params["source"].default is None and params["replay"].default is None


def test_the_panel_exists_only_with_the_flag_and_refuses_changes_during_a_run():
    src = _src()
    assert src.count("if replay is not None:\n        _rp_lens = replay.episode_lengths()") == 1
    panel = src[src.index("    # ---------------- RECORDED-EPISODE REPLAY: the panel (extension)"):
                src.index('    with server.gui.add_folder("Continuous cadence (HATO)"):')]
    for want in ('add_folder("Replay recorded episode"', '"Chunk SOURCE", ("recorded replay", "model")',
                 'add_dropdown(\n                "Episode"', 'add_number(\n                "Episode #"',
                 '"Start frame"', '"Replay mode"', "⏮ Rewind", 'add_image(', "the FIRST recorded frame",
                 "the frame the next chunk starts at", "hl_replay.episode_options(replay, _rp_first)",
                 "replay.first_usable_frames()"):
        assert want in panel, want
    # every picker asks _rp_busy() and puts the widgets back when a run owns the cursor
    assert panel.count("why = _rp_busy()") >= 5 and "auto_running.is_set()" in panel and "dripping.is_set()" in panel
    assert panel.count("_rp_set_widgets(") >= 4 and 'replay_ui["syncing"]' in panel
    # the first-frame image is per camera the CHECKPOINT reads, the cursor image on camera0
    assert "for _key in _rp_cams:" in panel and "replay.image_at(episode, 0, key)" in src
    assert "replay.image_at(episode, frame, key0)" in src
    # the transport angle is REPORTED, never applied
    assert "hl_replay.transport_rotation_deg(rec_rot6d, anchor_rot6d[ri])" in src
    assert "delta_ee_to_absolute(rel_pos, rel_rot6d, anchor_pos[ri], anchor_rot6d[ri])" in src   # the green line


def test_main_resolves_the_checkpoints_zarr_refuses_the_fivepoint_family_and_closes_the_source():
    src = _src()
    main = src[src.index("def main("):src.index("def _run_viser(")]
    assert 'if replay_path == "checkpoint":' in main and 'get("hl_data", {}).get("zarr_paths")' in main
    assert 'if family != "keypoint_subset":' in main and "raise SystemExit(" in main
    assert "hl_replay.RecordedEpisodeSource(replay_path, loaded.layout)" in main
    assert "hl_replay.speed_report(replay, int(args.replay_episode), float(args.step_hz), _EXEC_CHOICES)" in main
    assert "replay=replay)" in main and "replay.close()" in main
    assert re.search(r"def _run_viser\(.*\n.*replay=None\) -> int:", src)


def test_the_launcher_forwards_the_replay_knobs_and_the_runbook_documents_them():
    with open(_LAUNCHER, encoding="utf-8") as fh:
        sh = fh.read()
    for want in ("REPLAY_ZARR_EFF", 'EXTRA+=(--replay-zarr "$REPLAY_ZARR_EFF")', 'EXTRA+=(--replay-episode "$REPLAY_EPISODE")',
                 'EXTRA+=(--replay-mode "$REPLAY_MODE")', "sequential|openloop) ;;", "RECORDED-EPISODE REPLAY"):
        assert want in sh, want
    assert subprocess.run(["bash", "-n", _LAUNCHER], capture_output=True).returncode == 0
    if not os.path.isfile(_RUNBOOK):  # the runbook ships with docs/, not with every checkout
        return
    with open(_RUNBOOK, encoding="utf-8") as fh:
        doc = fh.read()
    assert "--replay-zarr" in doc and "REPLAY=1" in doc and "apply_step_validity" in doc
