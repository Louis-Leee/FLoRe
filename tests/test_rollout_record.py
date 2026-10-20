"""The ⏺ RUN RECORDER (``flore/s1_policy/serving/rollout_record.py``): the button's state machine, the
six-stream writer, the camera stream (one JPEG folder per view + an index, for one or two cameras), the
taps' cost when idle, the wiring into ``scripts/hl_serving_5pt.py`` (source level, like the walk /
leg-override wiring tests) and ``scripts/plot_rollout_record.py`` on a run the recorder itself wrote."""

from __future__ import annotations

import base64
import importlib.util
import json
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest

from flore.s1_policy.serving import rollout_record as R

_ROOT = Path(__file__).resolve().parents[1]
_PAGE = _ROOT / "scripts" / "hl_serving_5pt.py"
_PLOT = _ROOT / "scripts" / "plot_rollout_record.py"
_LAUNCHER = _ROOT / "scripts" / "run_hl_serving_5pt.sh"
_SRC = _PAGE.read_text()
_WALK_MARK = "# ---------------- the WALK SKILL (extension) ----------------"
_LEGS_MARK = "# ---------------- the LEG OVERRIDE (extension) ----------------"
_REC_MARK = "# ---------------- the RUN RECORDER (extension) ----------------"

FRAME = (np.arange(15, dtype=np.float64) * 0.01, np.zeros(20))
TWO_VIEWS = [(0, "camera0_rgb", "ws://127.0.0.1:8773"), (1, "camera1_rgb", "ws://127.0.0.1:8777")]


def _jpeg(shade: int = 0) -> bytes:
    """A real (tiny) JPEG, as the camera bridge would publish it — via the same cv2 the page decodes with."""
    import cv2

    img = np.full((12, 16, 3), shade, dtype=np.uint8)
    ok, buf = cv2.imencode(".jpg", img)
    assert ok
    data = buf.tobytes()
    assert data[:2] == b"\xff\xd8"
    return data


def _cam_msg(seq: int, jpeg: bytes | None = None, **extra) -> dict:
    """The bridge's own message shape (``realsense_bridge._frame_msg``)."""
    msg = {"seq": seq, "t": 100.0 + seq / 30.0, "t_host": time.time(), "capture_seq": 1000 + seq,
           "w": 16, "h": 12, "stream": "color",
           "jpeg_b64": base64.b64encode(_jpeg() if jpeg is None else jpeg).decode("ascii")}
    msg.update(extra)
    return msg


def _rec_block() -> str:
    return _SRC[_SRC.index(_REC_MARK):_SRC.index('with server.gui.add_folder("Bring-up report"):')]


def _recorder(tmp_path, **kw) -> tuple[R.RunRecorder, R.RolloutTaps]:
    taps = R.RolloutTaps()
    rec = R.RunRecorder(tmp_path, kw.pop("task", "push_gantry"), log=lambda s: None, **kw)
    taps.recorder = rec
    return rec, taps


def _feed(taps: R.RolloutTaps, *, n_steps: int = 3, legs_at: int | None = 1) -> None:
    """A tiny run's worth of taps: one chunk, ``n_steps`` steps, one publish per step, one RecordTap
    frame and one F/T frame."""
    taps.chunk(seq=1, cadence="sequential", decoded_pos=np.zeros((5, 15)), pos=np.ones((5, 15)),
               orn=np.zeros((5, 20)), wrench={"left": np.ones((5, 6)), "right": None},
               grip={"left": np.full(5, 0.08)}, seed=7, sample_ms=123.4, wire_src="command", n_exec=2)
    for i in range(n_steps):
        taps.step(1, i, FRAME, {"left": np.ones(6) * i, "right": None})
        taps.publish(FRAME, np.zeros(6), True, override=(i == legs_at), ref_legs=(i == legs_at))
    taps.on_record({"type": "record", "seq": 3, "vive_armed": True, "dof_pos": [0.0] * 29,
                    "vr_cmd_pos": [[0, 0, 0]] * 5}, time.monotonic())
    taps.on_force({"type": "ft", "seq": 9, "t": time.time(), "left_pipe": [1, 2, 3, 4, 5, 6],
                   "right_pipe": None, "left": [1, 2, 3], "right": [0, 0, 0]}, time.monotonic())


def _lines(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


# ---------------------------------------------------------------------------
# the state machine (the button)
# ---------------------------------------------------------------------------
def test_the_button_walks_idle_armed_recording_done_and_reset_relabels_it(tmp_path):
    rec, taps = _recorder(tmp_path)
    assert rec.state == "idle" and rec.ui()[1] == "blue" and not rec.recording
    assert "ARMED" in rec.toggle() and rec.state == "armed" and rec.ui()[1] == "green"
    assert rec.run_begin("sequential") is True and rec.state == "recording" and rec.recording
    assert rec.ui()[1] == "green" and "RECORDING" in rec.ui()[0]
    assert rec.run_begin("sequential") is False                      # a second start changes nothing
    _feed(taps)
    note = rec.run_end("sequential run finished — ceiling")
    assert rec.state == "done" and not rec.recording and rec.ui()[1] == "blue"
    assert "saved" in note and rec.last_dir is not None and rec.last_dir.is_dir()
    done_label = rec.ui()[0]
    assert "SAVED" in done_label and rec.last_dir.name in done_label
    rec.reset()
    assert rec.state == "idle" and rec.ui()[1] == "blue"
    assert rec.ui()[0] != done_label and "reset" in rec.ui()[0]     # blue both times, different words
    # done -> click -> armed again; armed -> click -> idle
    rec.arm()
    assert rec.state == "armed"
    assert "disarmed" in rec.toggle() and rec.state == "idle"


def test_every_run_records_without_a_click_and_the_label_says_so(tmp_path):
    rec, taps = _recorder(tmp_path, every_run=True)
    assert rec.state == "idle" and "EVERY" in rec.ui()[0]
    assert rec.run_begin("stream") is True
    _feed(taps)
    rec.run_end("done")
    assert rec.state == "done"
    assert rec.run_begin("stream") is True                           # no arm needed, a NEW folder
    _feed(taps)
    rec.run_end("done")
    assert len(list((tmp_path / "push_gantry").iterdir())) == 2


def test_folders_are_task_then_timestamp_and_never_collide(tmp_path):
    rec, taps = _recorder(tmp_path, task="rotate_box")
    dirs = []
    for _ in range(2):
        rec.arm()
        rec.run_begin("sequential")
        _feed(taps)
        rec.run_end("x")
        dirs.append(rec.last_dir)
    assert dirs[0] != dirs[1] and dirs[0].parent == tmp_path / "rotate_box" == dirs[1].parent
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}(_\d+)?", dirs[0].name)
    assert R.run_stamp(0.0) == time.strftime(R.STAMP_FMT, time.localtime(0.0))


def test_the_task_label_is_read_off_the_checkpoint_path_only_as_a_fallback():
    assert R.task_from_checkpoint("checkpoints/umi_5pt_push_gantry/h50") == "push_gantry"
    assert R.task_from_checkpoint("/x/checkpoints/umi_5pt_push_cart/h100/") == "push_cart"
    assert R.task_from_checkpoint("checkpoints/wipe_board/run7/step_00100000") == "run7"
    assert R.task_from_checkpoint(None) == "run" and R.task_from_checkpoint("   ") == "run"
    rec = R.RunRecorder("/tmp/x", "push gantry/1", log=lambda s: None)
    assert rec.task == "push_gantry_1"                               # a folder name, whatever was typed


# ---------------------------------------------------------------------------
# the writer
# ---------------------------------------------------------------------------
def test_a_run_writes_six_streams_and_a_meta_with_the_counts(tmp_path):
    rec, taps = _recorder(tmp_path, static_meta={"checkpoint": "ckpt"},
                          info_provider=lambda: {"exec_steps": 16},
                          measure=lambda msg: (np.full(15, 0.5), np.full(20, 0.25)))
    rec.arm()
    assert rec.run_begin("sequential", info={"extra": 1})
    _feed(taps, n_steps=3, legs_at=1)
    rec.run_end("finished")
    d = rec.last_dir
    assert sorted(p.name for p in d.iterdir()) == sorted([f"{s}.jsonl" for s in R.STREAMS] + ["meta.json"])
    meta = json.loads((d / "meta.json").read_text())
    assert meta["task"] == "push_gantry" and meta["stamp"] == d.name and meta["end_note"] == "finished"
    assert meta["run"] == {"cadence": "sequential", "exec_steps": 16, "extra": 1}
    assert meta["static"] == {"checkpoint": "ckpt"} and meta["duration_s"] >= 0
    assert meta["counts"] == {"chunks": 1, "steps": 3, "published": 3, "measured": 1, "force": 1,
                              "events": meta["counts"]["events"], "camera": 0}
    assert meta["camera"] is None and "camera" not in meta["streams"]      # no cameras declared: no folder
    chunk = _lines(d / "chunks.jsonl")[0]
    assert chunk["seq"] == 1 and chunk["seed"] == 7 and chunk["n_exec"] == 2 and chunk["wire_src"] == "command"
    assert np.asarray(chunk["pos"]).shape == (5, 15) and np.asarray(chunk["decoded_pos"]).shape == (5, 15)
    assert chunk["wrench"]["right"] is None and np.asarray(chunk["wrench"]["left"]).shape == (5, 6)
    steps = _lines(d / "steps.jsonl")
    assert [s["row"] for s in steps] == [0, 1, 2] and steps[2]["wrench"]["left"] == [2.0] * 6
    assert steps[0]["target_pos"] == np.round(FRAME[0], R.ROUND_M).tolist()
    pub = _lines(d / "published.jsonl")
    assert [p["legs"] for p in pub] == [False, True, False] and [p["ref_joints"] for p in pub] == [False, True, False]
    assert pub[0]["fcmd"] == [0.0] * 6 and pub[0]["ok"] is True
    meas = _lines(d / "measured.jsonl")[0]
    assert meas["msg"]["seq"] == 3 and meas["kp_pos"] == [0.5] * 15 and meas["kp_orn"] == [0.25] * 20
    force = _lines(d / "force.jsonl")[0]
    assert force["msg"]["left_pipe"] == [1, 2, 3, 4, 5, 6] and force["msg"]["right_pipe"] is None
    kinds = [e["kind"] for e in _lines(d / "events.jsonl")]
    assert kinds[0] == "run_begin" and kinds[-1] == "run_end"
    assert kinds.count("legs_on") == 1 and kinds.count("legs_off") == 1 and "armed" in kinds
    # every 't' is one clock (the page's wall clock), monotone within a stream
    ts = [p["t"] for p in pub]
    assert ts == sorted(ts) and abs(ts[0] - meta["t_begin"]) < 5.0


def test_an_empty_run_is_discarded_and_the_arm_is_kept(tmp_path):
    rec, _taps = _recorder(tmp_path)
    rec.arm()
    assert rec.run_begin("sequential")
    d = rec.run_dir
    assert d is not None and d.is_dir()
    note = rec.run_end("ignored — a ▶▶ run is already active")
    assert "nothing recorded" in note and not d.exists() and rec.state == "armed"
    assert not (tmp_path / "push_gantry").exists() or not any((tmp_path / "push_gantry").iterdir())


def test_taps_are_one_check_when_idle_and_never_raise(tmp_path):
    rec, taps = _recorder(tmp_path)
    cam0 = taps.camera_hook(0)
    for _ in range(3):
        taps.publish(FRAME, None, True)
        taps.step(1, 0, FRAME, None)
        taps.chunk(seq=1, cadence="x", decoded_pos=None)
        taps.on_record("not a dict", 0.0)
        taps.on_force(None, 0.0)
        cam0("not a dict", 0.0)
        cam0(_cam_msg(1), 0.0)
        taps.event("x", a=1)
    assert not (tmp_path / "push_gantry").exists()
    # garbage while RECORDING is swallowed by the tap / counted by the writer, never raised
    rec.arm()
    rec.run_begin("sequential")
    taps.publish(("not", "arrays"), object(), True)                 # raises inside the tap: dropped
    taps.step(1, 0, ("x", "y"), None)
    taps.on_force({"a set is not JSON": {1, 2}}, time.monotonic())   # raises inside the writer: counted
    cam0({"jpeg_b64": 12345}, time.monotonic())                       # not base64: raises in the writer
    _feed(taps)
    rec.run_end("x")
    meta = json.loads((rec.last_dir / "meta.json").read_text())
    assert meta["counts"]["published"] == 3 and meta["counts"]["steps"] == 3   # the good lines survived
    assert meta["writer_errors"] >= 2
    # the idle cost: an attribute read — microseconds, not a lock or a copy
    n = 20000
    t0 = time.perf_counter()
    for _ in range(n):
        taps.publish(FRAME, None, True)
    per_call_us = (time.perf_counter() - t0) / n * 1e6
    assert per_call_us < 20.0, per_call_us


def test_no_recorder_installed_means_the_taps_and_the_wrapper_are_inert(tmp_path):
    taps = R.RolloutTaps()
    taps.publish(FRAME, None, True)
    taps.step(0, 0, FRAME)
    taps.chunk(seq=0, cadence="", decoded_pos=None)
    taps.on_record({}, 0.0)
    taps.on_force({}, 0.0)
    ran = []
    taps.wrap_run(lambda: ran.append(1), "sequential")()
    assert ran == [1]


def test_wrap_run_opens_the_recording_and_brackets_the_run_but_does_not_end_it_even_when_it_raises(tmp_path):
    """2026-09-20: a recording is the operator's EPISODE — a ▶▶ ending (here: dying) is an event
    inside it, and only an END gesture (STOP / Reset / disarm / the button / Ctrl-C) closes it."""
    rec, taps = _recorder(tmp_path)
    rec.arm()

    def body():
        assert rec.state == "recording"
        _feed(taps)
        raise RuntimeError("the rollout died")

    with pytest.raises(RuntimeError):
        taps.wrap_run(body, "sequential", end_note=lambda: "sequential run error")()
    assert rec.state == "recording" and rec.recording                # STILL recording
    d = rec.run_dir
    # a second ▶▶ and a manual Predict → Run land in the SAME folder
    taps.wrap_run(lambda: _feed(taps), "sequential", end_note=lambda: "sequential run finished — ceiling")()
    assert taps.begin("sequential") is False                          # a no-op while open
    _feed(taps)
    assert rec.run_dir == d
    note = taps.end("STOP", wait=True)
    assert "saved" in note and rec.state == "done" and rec.last_dir == d
    meta = json.loads((d / "meta.json").read_text())
    assert meta["end_note"] == "STOP" and meta["run"]["cadence"] == "sequential"
    assert meta["counts"]["chunks"] == 3 and meta["n_auto_runs"] == 2
    assert [r["note"] for r in meta["auto_runs"]] == ["sequential run error", "sequential run finished — ceiling"]
    assert all(r["t_end"] is not None and r["t_end"] >= r["t_begin"] for r in meta["auto_runs"])
    kinds = [e["kind"] for e in _lines(d / "events.jsonl")]
    assert kinds[0] == "run_begin" and kinds[-1] == "run_end"
    assert kinds.count("auto_run_begin") == 2 and kinds.count("auto_run_end") == 2
    # not armed -> the wrapper is the body unchanged, no folder
    before = set((tmp_path / "push_gantry").iterdir())
    taps.wrap_run(lambda: _feed(taps), "stream")()
    assert set((tmp_path / "push_gantry").iterdir()) == before and rec.state == "done"
    assert taps.end("STOP") is None                                    # nothing to end


def test_a_manual_predict_opens_the_recording_when_armed_and_every_run_reopens_after_each_end(tmp_path):
    rec, taps = _recorder(tmp_path)
    assert taps.begin("sequential") is False and rec.state == "idle"   # not armed: nothing opens
    rec.arm()
    assert taps.begin("sequential") is True and rec.state == "recording"
    first = rec.run_dir
    for seq in (1, 2):                                                 # Predict → Run, Predict → Run …
        taps.begin("sequential")
        taps.chunk(seq=seq, cadence="sequential", decoded_pos=np.zeros((5, 15)))
        taps.step(seq, 0, FRAME)
    assert rec.run_dir == first
    assert "saved" in rec.reset() and rec.state == "idle"              # Reset is an end gesture
    assert json.loads((first / "meta.json").read_text())["end_note"] == "reset"
    assert taps.begin("sequential") is False                           # idle again: a click is needed
    rec.every_run = True
    assert taps.begin("sequential") is True and rec.run_dir != first   # every_run: the next Predict opens
    taps.chunk(seq=3, cadence="sequential", decoded_pos=np.zeros((5, 15)))
    second = rec.run_dir
    assert "saved" in taps.end("STOP", wait=True) and rec.state == "done"
    time.sleep(0.05)
    assert taps.begin("sequential") is True and rec.run_dir not in (first, second)
    rec.run_end("x")


def test_a_recording_with_only_heartbeat_frames_is_discarded(tmp_path):
    """An armed button + a refused Predict: the heartbeat publishes, nothing is predicted — no folder."""
    rec, taps = _recorder(tmp_path)
    rec.arm()
    assert taps.begin("sequential")
    d = rec.run_dir
    for _ in range(5):
        taps.publish(FRAME, None, True)
    time.sleep(0.05)
    note = rec.run_end("reset")
    assert "nothing recorded" in note and not d.exists() and rec.state == "armed"


def _wait_for(pred, timeout=3.0):
    t0 = time.monotonic()
    while not pred():
        if time.monotonic() - t0 > timeout:
            return False
        time.sleep(0.01)
    return True


def test_the_low_level_disarming_ends_the_recording_from_the_socket_thread_without_blocking_it(tmp_path):
    rec, taps = _recorder(tmp_path)
    rec.arm()
    assert taps.begin("sequential")
    d = rec.run_dir
    frame = {"type": "record", "seq": 1, "dof_pos": [0.0] * 29, "vr_cmd_pos": [[0, 0, 0]] * 5}
    # un-armed frames first (the operator arms the low level after the first Predict), then armed
    # ones (`_feed` sends one too): a RISING edge, no end
    for armed in (False, False):
        taps.on_record({**frame, "vive_armed": armed}, time.monotonic())
    _feed(taps)
    for armed in (True, True):
        taps.on_record({**frame, "vive_armed": armed}, time.monotonic())
    time.sleep(0.05)
    assert rec.state == "recording"
    t0 = time.monotonic()
    taps.on_record({**frame, "vive_armed": False}, time.monotonic())     # joystick `start`: the falling edge
    assert time.monotonic() - t0 < 0.05                                 # the socket thread did not wait
    assert _wait_for(lambda: rec.state == "done"), rec.state
    meta = json.loads((d / "meta.json").read_text())
    assert "DISARMED" in meta["end_note"]
    kinds = [e["kind"] for e in _lines(d / "events.jsonl")]
    assert "armed" in kinds and "disarmed" in kinds and kinds[-1] == "run_end"
    # a recording that begins un-armed and never arms is NOT ended by un-armed frames
    rec.arm()
    assert taps.begin("sequential")
    for _ in range(3):
        taps.on_record({**frame, "vive_armed": False}, time.monotonic())
    time.sleep(0.05)
    assert rec.state == "recording"
    # …and end_on_disarm=False leaves even a real edge to the operator
    rec2, taps2 = _recorder(tmp_path, task="other", end_on_disarm=False)
    rec2.arm()
    assert taps2.begin("sequential")
    for armed in (True, False):
        taps2.on_record({**frame, "vive_armed": armed}, time.monotonic())
    time.sleep(0.05)
    assert rec2.state == "recording"
    rec.run_end("x")
    rec2.run_end("x")


def test_stop_ends_asynchronously_and_the_next_begin_waits_for_the_folder(tmp_path):
    rec, taps = _recorder(tmp_path, every_run=True)
    assert taps.begin("sequential")
    first = rec.run_dir
    _feed(taps)
    note = taps.end("STOP")                                             # async: returns at once
    assert note and "closing" in note
    assert taps.end("STOP") is None or "closing" in taps.end("STOP")    # a second one changes nothing
    # the very next Predict: the pending end is joined, the first folder is complete, a NEW one opens
    assert taps.begin("sequential") is True
    assert rec.state == "recording" and rec.run_dir != first
    meta = json.loads((first / "meta.json").read_text())
    assert meta["end_note"] == "STOP" and meta["counts"]["chunks"] == 1
    assert rec.last_dir == first
    # Reset racing an async end: the state settles to idle (the reset label), the folder is still saved
    _feed(taps)
    second = rec.run_dir
    taps.end("STOP")
    rec.reset()
    assert rec.state == "idle" and (second / "meta.json").is_file()


def test_a_failing_measure_records_the_joints_only_and_counts_the_error(tmp_path):
    def boom(_msg):
        raise ValueError("no FK")

    rec, taps = _recorder(tmp_path, measure=boom)
    rec.arm()
    rec.run_begin("sequential")
    _feed(taps)
    rec.run_end("x")
    meas = _lines(rec.last_dir / "measured.jsonl")[0]
    assert meas["kp_pos"] is None and meas["msg"]["dof_pos"] == [0.0] * 29
    assert json.loads((rec.last_dir / "meta.json").read_text())["writer_errors"] == 1


def test_the_button_click_while_recording_stops_the_recording_and_the_run_goes_on(tmp_path):
    rec, taps = _recorder(tmp_path)
    rec.arm()
    rec.run_begin("sequential")
    _feed(taps)
    note = rec.toggle()
    assert rec.state == "done" and "saved" in note
    taps.publish(FRAME, None, True)                                  # the run still publishes: ignored
    assert json.loads((rec.last_dir / "meta.json").read_text())["counts"]["published"] == 3


def test_reset_ends_a_live_recording_first(tmp_path):
    rec, taps = _recorder(tmp_path)
    rec.arm()
    rec.run_begin("sequential")
    _feed(taps)
    rec.reset()
    assert rec.state == "idle" and (rec.last_dir / "meta.json").is_file()


def test_the_autoplot_is_a_detached_low_priority_child_that_the_page_never_waits_on(tmp_path):
    marker = tmp_path / "plotted"
    rec, taps = _recorder(tmp_path, autoplot=True,
                          plot_cmd=lambda d: [sys.executable, "-c",
                                              f"import pathlib; pathlib.Path({str(marker)!r}).write_text('ok')"])
    rec.arm()
    rec.run_begin("sequential")
    _feed(taps)
    t0 = time.perf_counter()
    rec.run_end("x")
    assert time.perf_counter() - t0 < 2.0
    for _ in range(200):
        if marker.is_file():
            break
        time.sleep(0.02)
    assert marker.read_text() == "ok" and (rec.last_dir / "plot.log").is_file()
    argv = R.plot_argv(Path("/tmp/run"))
    assert argv[0] == sys.executable and argv[1].endswith("plot_rollout_record.py") and argv[2] == "/tmp/run"


# ---------------------------------------------------------------------------
# the camera stream
# ---------------------------------------------------------------------------
def test_camera_frames_land_per_view_with_an_index_and_the_meta_says_what_came(tmp_path):
    logs: list[str] = []
    taps = R.RolloutTaps()
    rec = R.RunRecorder(tmp_path, "wipe_board_all", log=logs.append, cameras=TWO_VIEWS)
    taps.recorder = rec
    assert [c["key"] for c in rec.cameras] == ["camera0_rgb", "camera1_rgb"]
    cam0, cam1, cam2 = taps.camera_hook(0), taps.camera_hook(1), taps.camera_hook(2)
    assert cam0.__name__ == "on_camera0"
    cam0(_cam_msg(1), time.monotonic())                    # idle: nothing (no folder yet)
    rec.arm()
    assert rec.run_begin("sequential")
    d = rec.run_dir
    assert (d / "camera" / "camera0_rgb").is_dir() and (d / "camera" / "camera1_rgb").is_dir()   # up front
    assert (d / "camera" / "frames.jsonl").is_file()
    _feed(taps)
    jpegs = [_jpeg(s) for s in (10, 40, 80, 120, 200)]
    for s, j in enumerate(jpegs, start=1):
        cam0(_cam_msg(s, j), time.monotonic())
    cam0(_cam_msg(5, jpegs[-1]), time.monotonic())        # the bridge re-sent its last capture: a duplicate
    cam0(_cam_msg(6, b"not a jpeg at all"), time.monotonic())   # bytes without the SOI marker: counted, not written
    cam0({"seq": 7, "t": 1.0}, time.monotonic())           # no jpeg_b64: ignored at the tap
    cam0("garbage", time.monotonic())                      # not a dict: ignored at the tap
    for s in (1, 2, 3):
        cam1(_cam_msg(s), time.monotonic())
    cam2(_cam_msg(1), time.monotonic())                    # a view nobody declared: recorded as camera2
    time.sleep(0.05)
    note = rec.run_end("finished")
    assert "camera 5 camera0_rgb + 3 camera1_rgb + 1 camera2" in note
    # the files: one JPEG per distinct good frame, byte-identical to what the bridge sent, numbered per view
    files0 = sorted(p.name for p in (d / "camera" / "camera0_rgb").iterdir())
    assert files0 == [f"{i:06d}.jpg" for i in range(5)]
    assert [(d / "camera" / "camera0_rgb" / f).read_bytes() for f in files0] == jpegs
    assert sorted(p.name for p in (d / "camera" / "camera1_rgb").iterdir()) == [f"{i:06d}.jpg" for i in range(3)]
    assert (d / "camera" / "camera2" / "000000.jpg").is_file()
    # the index: one line per written frame, page clock + the bridge's own stamps + the file
    idx = _lines(d / "camera" / "frames.jsonl")
    assert len(idx) == 9 and [r["key"] for r in idx[:5]] == ["camera0_rgb"] * 5
    assert [r["seq"] for r in idx[:5]] == [1, 2, 3, 4, 5] and idx[0]["file"] == "camera/camera0_rgb/000000.jpg"
    assert idx[0]["t_bridge"] == 100.0 + 1 / 30.0 and idx[0]["capture_seq"] == 1001 and idx[0]["w"] == 16
    assert idx[0]["bytes"] == len(jpegs[0]) and idx[-1]["key"] == "camera2" and idx[-1]["cam"] == 2
    ts = [r["t"] for r in idx if r["cam"] == 0]
    assert ts == sorted(ts) and abs(ts[0] - time.time()) < 5.0            # the page clock, like every stream
    # meta: the counts, the folder, per view what came and what was dropped and why
    meta = json.loads((d / "meta.json").read_text())
    assert meta["counts"]["camera"] == 9 and meta["streams"]["camera"] == "camera/frames.jsonl"
    cm = meta["camera"]
    assert cm["dir"] == "camera" and cm["index"] == "camera/frames.jsonl" and cm["every"] == 1
    v0, v1, v2 = cm["views"]
    assert (v0["key"], v0["frames"], v0["seen"], v0["duplicates"], v0["bad"], v0["skipped"]) == (
        "camera0_rgb", 5, 6, 1, 1, 0)
    assert v0["url"] == "ws://127.0.0.1:8773" and v0["dir"] == "camera/camera0_rgb"
    assert v0["bytes"] == sum(len(j) for j in jpegs) and (v0["fps"] is None or v0["fps"] > 0)
    assert (v1["key"], v1["frames"]) == ("camera1_rgb", 3) and (v2["key"], v2["frames"], v2["url"]) == ("camera2", 1, None)
    assert meta["writer_errors"] == 0                                      # a bad frame is a COUNT, not an error
    assert any("camera 5 camera0_rgb" in line for line in logs)


def test_camera_every_thins_the_stream_and_zero_turns_it_off(tmp_path):
    rec, taps = _recorder(tmp_path, cameras=[(0, "camera0_rgb")], camera_every=2)
    cam0 = taps.camera_hook(0)
    rec.arm()
    rec.run_begin("sequential")
    _feed(taps)
    for s in range(1, 7):
        cam0(_cam_msg(s), time.monotonic())
    rec.run_end("x")
    d = rec.last_dir
    assert sorted(p.name for p in (d / "camera" / "camera0_rgb").iterdir()) == ["000000.jpg", "000001.jpg", "000002.jpg"]
    v = json.loads((d / "meta.json").read_text())["camera"]["views"][0]
    assert (v["frames"], v["seen"], v["skipped"]) == (3, 6, 3)
    assert [r["seq"] for r in _lines(d / "camera" / "frames.jsonl")] == [1, 3, 5]    # the first, then every 2nd

    rec, taps = _recorder(tmp_path, cameras=TWO_VIEWS, camera_every=0)
    assert rec.cameras == [] and rec.camera_every == 0
    cam0 = taps.camera_hook(0)
    rec.arm()
    rec.run_begin("sequential")
    _feed(taps)
    cam0(_cam_msg(1), time.monotonic())
    rec.run_end("x")
    meta = json.loads((rec.last_dir / "meta.json").read_text())
    assert not (rec.last_dir / "camera").exists() and meta["camera"] is None and meta["counts"]["camera"] == 0


def test_an_empty_run_with_camera_frames_is_still_discarded(tmp_path):
    rec, taps = _recorder(tmp_path, cameras=TWO_VIEWS)
    rec.arm()
    rec.run_begin("sequential")
    taps.camera_hook(0)(_cam_msg(1), time.monotonic())     # frames flow the instant the folder opens…
    time.sleep(0.05)
    note = rec.run_end("ignored — the low level is NOT connected")
    assert "nothing recorded" in note and rec.state == "armed"          # …but a refused ▶▶ is still nothing
    assert not (tmp_path / "push_gantry").exists() or not any((tmp_path / "push_gantry").iterdir())


def test_camera_specs_take_one_or_two_views_in_any_shape_and_never_share_a_folder():
    assert R.camera_specs(None) == [] and R.camera_specs([]) == []
    one = R.camera_specs(["camera0_rgb"])
    assert one == [{"cam": 0, "key": "camera0_rgb", "url": None}]
    two = R.camera_specs([{"cam": 0, "key": "camera0_rgb", "url": "ws://a"}, {"cam": 1, "key": "camera1_rgb"}])
    assert [(c["cam"], c["key"], c["url"]) for c in two] == [(0, "camera0_rgb", "ws://a"), (1, "camera1_rgb", None)]
    assert [c["key"] for c in R.camera_specs(TWO_VIEWS)] == ["camera0_rgb", "camera1_rgb"]
    # a key is a folder name: sanitised, never empty, never shared
    odd = R.camera_specs([(0, "left/wrist cam"), (1, ""), (2, None), (3, "left/wrist cam")])
    assert [c["key"] for c in odd] == ["left_wrist_cam", "camera1", "camera2", "left_wrist_cam_2"]
    assert R.camera_specs([("x", "k")])[0]["cam"] == 0                    # an unusable index falls back to the position


# ---------------------------------------------------------------------------
# wiring into the page (source level)
# ---------------------------------------------------------------------------
def test_the_recorder_is_one_block_after_the_leg_override_and_the_reviewed_page_holds_only_the_taps():
    block = _rec_block()
    page_before = _SRC[_SRC.index("def _run_viser("):_SRC.index(_WALK_MARK)]
    for name in ("recorder", "_rec_toggle", "_rec_refresh", "_rec_reset", "_rec_shutdown", "_rec_measure",
                 "_rec_info", "_rec_static", "b_rec", "c_rec_every", "g_rec", "RunRecorder", "record_task",
                 "record_dir", "plot_argv", "_rec_cameras", "record_camera_every"):
        assert name not in page_before, name
        assert name in block, name
    # the views are the page's OWN camera subscriptions — one per image key the checkpoint reads
    assert re.search(r'_rec_cameras = \[\{"cam": i, "key": str\(loaded\.image_keys\[i\]\), "url": subs\[f"camera\{i\}"\]\.url\}\s*'
                     r'for i in range\(int\(loaded\.num_cameras\)\) if f"camera\{i\}" in subs\]', block)
    assert "cameras=_rec_cameras" in block and "camera_every=int(args.record_camera_every)" in block
    assert _SRC.index(_LEGS_MARK) < _SRC.index(_REC_MARK)
    assert block.count("add_folder(") == 1 and "order=f_control.order" in block   # shown under Control
    # the block never publishes, plans, commits or owns the page
    for forbidden in ("_send(", "_set_wire(", "wire.commit(", "wire.plan(", "commit_role", "clutch.apply",
                      "busy.acquire", "dripping.set", "stop.set()", "stop.clear()", "s0.send", "wire_hooks["):
        assert forbidden not in block, forbidden
    # the reviewed page: the taps holder and its one-line calls, nothing else
    assert "taps = rollout_record.RolloutTaps()" in page_before
    send = page_before[page_before.index("def _send(frame, fcmd, ref_joints=None) -> bool:"):
                       page_before.index("def _heartbeat() -> None:")]
    assert (send.index("ok, reason = wire.publish(s0, frame, fcmd, ref_joints=ref_joints)")
            < send.index("taps.publish(frame, fcmd, ok, override=hook is not None, ref_legs=ref_joints is not None)"))
    predict = page_before[page_before.index("def _predict() -> str:"):page_before.index("def _drip(out) -> int:")]
    assert '"seq": stats["n"]' in predict and 'taps.chunk(seq=stats["n"], cadence="sequential"' in predict
    assert predict.index("taps.chunk(") < predict.index("_draw_paths(chunk)")
    # 2026-09-20: the recording opens at the FIRST Predict (manual or ▶▶'s), before the replay branch
    assert (predict.index('taps.begin("replay" if replaying else "sequential")')
            < predict.index("return _predict_replay()") < predict.index("planner.infer_chunk("))
    # …and ends on STOP (async — STOP never waits on the disk), NOT on ⏹ and not with the ▶▶ body
    stop_body = _SRC[_SRC.index("def _stop_now() -> str:"):_SRC.index("@b_auto.on_click")]
    assert 'taps.end("STOP")' in stop_body and stop_body.index("stop.set()") < stop_body.index('taps.end("STOP")')
    retire = _SRC[_SRC.index("def _auto_retire() -> str:"):_SRC.index("def _stop_now() -> str:")]
    assert "taps." not in retire
    assert _SRC.count("taps.end(") == 1 and "run_end(" not in page_before
    drip = page_before[page_before.index("def _drip(out) -> int:"):page_before.index("def _release_jaw_latches()")]
    assert (drip.index("frame = wire.frame(plan, i)") < drip.index('taps.step(out.get("seq", -1), i, frame, rows)')
            < drip.index("pub = _reach_apply(frame)"))
    stream = _SRC[_SRC.index("def _run_streaming() -> None:"):_SRC.index("@b_predict.on_click")]
    assert stream.count('taps.chunk(seq=stats["n"], cadence="stream"') == 2
    assert "taps.step(-1, t, frame, rows, n_blended=cur.n_blended)" in stream
    assert stream.index("_set_wire(pub, fcmd)") < stream.index("taps.step(-1, t, frame, rows")
    start = _SRC[_SRC.index("def _auto_start() -> str:"):_SRC.index("def _auto_retire() -> str:")]
    # the cadence NAME the recorder is handed: "replay" for a recorded episode, else the ▶▶ cadence
    assert 'cadence = "replay" if replaying else ("stream" if streaming else "sequential")' in start
    assert re.search(r'target = taps\.wrap_run\(target, cadence, end_note=lambda: str\(g_msg\.value\)\)', start)
    assert start.index("target = _run_streaming if streaming else _run_to_completion") < start.index("taps.wrap_run(")


def test_the_main_loop_resets_refreshes_and_closes_the_recorder_and_main_taps_both_bridges():
    loop = _SRC[_SRC.index("# ---------------- the loop ----------------"):]
    reset = loop[loop.index('if flags["reset"]:'):loop.index('if flags["predict"]:')]
    assert reset.index("_set_wire(wire.idle(), None)") < reset.index("_rec_reset()")
    assert reset.index("_rec_reset()") < reset.index('g_msg.value = ("reset: anchor, clutch, body and wire')
    assert "_rec_refresh()" in loop[loop.index("g_wlegs.value = _legs_status()"):]
    assert loop.index("keys.stop()") < loop.index("_rec_shutdown()") < loop.index('_open_jaws("shutdown"')
    main = _SRC[_SRC.index("def main(argv=None) -> int:"):_SRC.index("def _run_viser(")]
    assert 'HL._WSSubscriber(args.force_ws, "force", on_msg=taps.on_force)' in main
    assert 'HL._WSSubscriber(args.record_ws, "record", on_msg=taps.on_record)' in main
    # every camera the page subscribes to (= the checkpoint's views, trimmed above) carries its own hook
    assert 'HL._WSSubscriber(url, f"camera{i}", on_msg=taps.camera_hook(i))' in main
    assert main.count("HL._WSSubscriber(url, f\"camera{i}\"") == 1
    assert "taps = rollout_record.RolloutTaps()" in main and "reach=reach, taps=taps, replay=replay)" in main
    assert "walk_clip=None, reach=None, taps=None, replay=None) -> int:" in _SRC


def test_the_parser_and_the_launcher_carry_the_record_knobs():
    spec = importlib.util.spec_from_file_location("_hl5_rec_wiring", _PAGE)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_hl5_rec_wiring"] = mod
    spec.loader.exec_module(mod)
    a = mod.build_parser().parse_args(["--checkpoint", "x"])
    assert a.record_dir.endswith("data/rollouts") and a.record_task is None
    assert a.record_every_run is False and a.record_autoplot is True and a.record_camera_every == 1
    b = mod.build_parser().parse_args(["--checkpoint", "x", "--record-task", "push_cart", "--record-dir", "/d",
                                       "--record-every-run", "--no-record-autoplot", "--record-camera-every", "3"])
    assert (b.record_task, b.record_dir, b.record_every_run, b.record_autoplot) == ("push_cart", "/d", True, False)
    assert b.record_camera_every == 3
    sh = _LAUNCHER.read_text()
    assert 'EXTRA+=(--record-camera-every "$RECORD_CAMERA_EVERY")' in sh
    task_block = sh[sh.index('if [ -z "$CKPT" ] && [ -n "${TASK:-}" ]; then'):sh.index('if [ -z "$CKPT" ]; then')]
    assert 'TASK_EXTRA+=(--record-task "$TASK")' in task_block
    assert task_block.count('--record-task "$TASK"') == 2     # also when an explicit --checkpoint won
    assert 'EXTRA+=(--record-dir "$RECORD_DIR")' in sh and "EXTRA+=(--record-every-run)" in sh
    assert subprocess.run(["bash", "-n", str(_LAUNCHER)], capture_output=True).returncode == 0


# ---------------------------------------------------------------------------
# scripts/plot_rollout_record.py
# ---------------------------------------------------------------------------
def _plot_module():
    spec = importlib.util.spec_from_file_location("plot_rollout_record", _PLOT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["plot_rollout_record"] = mod
    spec.loader.exec_module(mod)
    return mod


def _synthetic_run(tmp_path, *, cadence: str = "sequential") -> Path:
    """A run the RECORDER wrote: 3 chunks x 4 steps, publishes, RecordTap frames with the robot 10 mm
    behind the command in x, an un-armed head, F/T frames = predicted + 0.5 N, a leg override on
    chunk 2."""
    ready = np.zeros(15)
    taps = R.RolloutTaps()
    rec = R.RunRecorder(tmp_path, "push_gantry", log=lambda s: None,
                        static_meta={"checkpoint": "checkpoints/umi_5pt_push_gantry/h50",
                                     "wire_labels": ["LEFT hand", "RIGHT hand", "head/torso", "LEFT foot", "RIGHT foot"],
                                     "force_roles": ["left", "right"], "action_dt": 1 / 30, "action_horizon": 8,
                                     "ready_pos": ready.tolist()},
                        measure=lambda msg: (np.asarray(msg["_kp"]), np.zeros(20)),
                        info_provider=lambda: {"exec_steps": 4, "force_cmd": True, "force_gate_n": 1.5},
                        cameras=TWO_VIEWS)
    taps.recorder = rec
    cams = (taps.camera_hook(0), taps.camera_hook(1))
    rec.arm()
    assert rec.run_begin(cadence)
    t_step = 0
    for c in range(1, 4):
        k = np.arange(8)
        plan = np.tile(ready, (8, 1)) + 0.01 * (c - 1) * 4
        plan[:, 0] += 0.01 * k
        wl = np.zeros((8, 6))
        wl[:, 0] = -3.0
        if cadence == "stream":
            taps.chunk(seq=c, cadence="stream", decoded_pos=plan, wrench={"left": wl, "right": None},
                       grip={}, t_obs=t_step)
        else:
            taps.chunk(seq=c, cadence="sequential", decoded_pos=plan, pos=plan, orn=np.zeros((8, 20)),
                       wrench={"left": wl, "right": None}, grip={}, seed=c, sample_ms=100.0, n_exec=4)
        for i in range(4):
            frame = (plan[i], np.zeros(20))
            if cadence == "stream":
                taps.step(-1, t_step, frame, {"left": wl[i], "right": None}, n_blended=2)
            else:
                taps.step(c, i, frame, {"left": wl[i], "right": None})
            legs = c == 2
            pub = plan[i].copy()
            if legs:
                pub[9] += 0.05
            taps.publish((pub, np.zeros(20)), np.array([3.0, 0, 0, 0, 0, 0]), True, override=legs, ref_legs=legs)
            robot = pub.copy()
            robot[0] -= 0.010
            taps.on_record({"type": "record", "seq": t_step, "t": time.time(), "vive_armed": t_step > 0,
                            "dof_pos": [0.0] * 29, "vr_cmd_pos": pub.reshape(5, 3).tolist(),
                            "f_meas_ungated": [[2.5, 0, 0], [0, 0, 0]], "f_meas": [[2.5, 0, 0], [0, 0, 0]],
                            "f_cmd": [[3.0, 0, 0], [0, 0, 0]], "_kp": robot.tolist()}, time.monotonic())
            taps.on_force({"type": "ft", "t": time.time(), "left_pipe": (wl[i] + 0.5).tolist(), "right_pipe": None,
                           "left": (wl[i, :3] + 0.5).tolist(), "right": [0, 0, 0]}, time.monotonic())
            for k, hook in enumerate(cams):
                hook(_cam_msg(t_step + 1, _jpeg((20 * t_step + 60 * k) % 256)), time.monotonic())
            t_step += 1
            time.sleep(0.002)
    time.sleep(0.05)
    rec.run_end("sequential run finished — ceiling")
    return rec.last_dir


def test_the_plot_script_loads_summarises_and_places_every_horizon(tmp_path):
    P = _plot_module()
    d = _synthetic_run(tmp_path)
    run = P.load_run(d)
    assert run.cadence() == "sequential" and len(run.chunks) == 3 and run.steps["t"].size == 12
    assert run.published["legs"].sum() == 4 and run.measured["armed"].sum() == 11
    hz = P.chunk_horizons(run)
    assert [h["seq"] for h in hz] == [1, 2, 3] and all(h["pos"].shape == (8, 15) for h in hz)
    # row 0 of each horizon sits at the wall time its step was executed, later rows dt apart
    for h in hz:
        i = np.where((run.steps["chunk"] == h["seq"]) & (run.steps["row"] == 0))[0][0]
        assert h["t"][0] == run.steps["t"][i] and abs((h["t"][1] - h["t"][0]) - run.dt) < 1e-12
    assert run.camera["t"].size == 24 and set(run.camera["key"].tolist()) == {"camera0_rgb", "camera1_rgb"}
    assert all((d / f).is_file() for f in run.camera["file"])
    s = P.summarize(run)
    assert s["camera"]["n_views"] == 2 and s["camera"]["frames"] == 24
    assert s["camera"]["views"]["camera0_rgb"]["frames"] == 12 and s["camera"]["views"]["camera0_rgb"]["recorder"]["bad"] == 0
    assert s["chunks"] == 3 and s["steps"] == 12 and s["measured_keypoints"] and s["lowlevel_target_seen"]
    assert len(s["legs_override_intervals_s"]) == 1 and s["armed_fraction"] is not None
    e = s["points_m"]["LEFT hand"]["x"]["robot_minus_published"]
    assert abs(e["mean"] - (-0.010)) < 1e-6 and e["n"] > 0             # the 10 mm lag, over armed time only
    clamp = s["points_m"]["LEFT foot"]["x"]["published_minus_lowlevel_target"]
    assert clamp["max_abs"] < 1e-9                                      # saw == published here: no clamp
    f = s["hands_N"]["left"]["Fx"]["measured_minus_predicted"]
    assert abs(f["mean"] - 0.5) < 1e-6 and "note" in s["hands_N"]["right"]
    text = P.summary_text(s)
    assert "ROBOT − PUBLISHED" in text and "force left" in text and "force right: no force data" in text
    assert "camera: camera0_rgb 12 frames" in text and "camera1_rgb 12 frames" in text
    # numbers only, no matplotlib needed
    s2 = P.process_run(d, summary_only=True, log=lambda s: None)
    assert (d / "summary.json").is_file() and (d / "summary.txt").is_file() and s2["plots"] == []
    # the resamplers
    held = P.hold_at(np.array([0.0, 1.0, 2.0]), np.array([[1.0], [2.0], [3.0]]), np.array([-1.0, 0.5, 2.5]))
    assert np.isnan(held[0, 0]) and held[1, 0] == 1.0 and held[2, 0] == 3.0
    assert P.intervals(np.array([0.0, 1.0, 2.0, 3.0]), np.array([False, True, True, False])) == [(1.0, 3.0)]
    assert P.intervals(np.array([0.0, 1.0]), np.array([True, True]), 5.0) == [(0.0, 5.0)]


def test_the_plot_script_places_stream_horizons_at_their_step_clock_stamp(tmp_path):
    P = _plot_module()
    d = _synthetic_run(tmp_path, cadence="stream")
    run = P.load_run(d)
    hz = P.chunk_horizons(run)
    assert [h["seq"] for h in hz] == [1, 2, 3]
    for h, c in zip(hz, run.chunks):
        i = np.where((run.steps["chunk"] == -1) & (run.steps["row"] == c["t_obs"]))[0][0]
        assert h["t"][0] == run.steps["t"][i]
        assert np.allclose(h["pos"][0], run.steps["target_pos"][i])   # translated onto the executed target


def test_the_plot_script_writes_the_figures_and_all_skips_finished_runs(tmp_path):
    pytest.importorskip("matplotlib")
    P = _plot_module()
    d = _synthetic_run(tmp_path)
    assert P.main([str(d)]) == 0
    for name in ("keypoints_xyz.png", "force_xyz.png", "force_lowlevel.png", "camera_strip.png", "summary.json",
                 "summary.txt"):
        assert (d / name).is_file() and (d / name).stat().st_size > 1000, name
    assert "camera_strip.png" in json.loads((d / "summary.json").read_text())["plots"]
    assert P.find_runs(tmp_path, force=False) == [] and P.find_runs(tmp_path, force=True) == [d]
    assert P.main([str(tmp_path), "--all"]) == 0                        # nothing to do, not an error
    assert P.main([str(tmp_path)]) == 2                                 # a folder above runs needs --all
    assert P.main([str(tmp_path / "missing")]) == 2


# ---------------------------------------------------------------------------
# 2026-09-19: a RECORDED-EPISODE REPLAY stamps its chunks with the source; a model run does not
# ---------------------------------------------------------------------------
def test_a_chunk_records_its_source_and_replay_identity_and_a_model_chunk_records_null(tmp_path):
    taps = R.RolloutTaps()
    rec = R.RunRecorder(tmp_path, "wipe_board_all", log=lambda s: None)
    taps.recorder = rec
    rec.arm()
    assert rec.run_begin("replay")
    plan = np.zeros((4, 15))
    taps.chunk(seq=1, cadence="replay", decoded_pos=plan, pos=plan, orn=np.zeros((4, 20)), seed=None,
               sample_ms=1.2, n_exec=4, source="replay",
               replay={"episode": 12, "frame": 30, "mode": "sequential", "n_valid": 50, "time_s": 1.0,
                       "zarr": "wipe.zarr.zip", "force_blanked": {"left": 1}})      # the dict is dropped
    taps.chunk(seq=2, cadence="sequential", decoded_pos=plan, pos=plan, orn=np.zeros((4, 20)), seed=3)
    taps.step(1, 0, (plan[0], np.zeros(20)), {})
    d = rec.run_dir
    rec.run_end("done")
    rows = [json.loads(line) for line in (d / "chunks.jsonl").read_text().splitlines()]
    assert rows[0]["source"] == "replay" and rows[0]["cadence"] == "replay"
    assert rows[0]["replay"] == {"episode": 12, "frame": 30, "mode": "sequential", "n_valid": 50, "time_s": 1.0,
                                 "zarr": "wipe.zarr.zip"}
    assert rows[1]["source"] is None and rows[1]["replay"] is None and rows[1]["seed"] == 3
    # the plot script names the source
    import importlib.util

    spec = importlib.util.spec_from_file_location("plot_rollout_record", _PLOT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    run = mod.load_run(d)
    src = mod.chunk_source(run)
    assert src["source"] == "mixed" and src["replay"]["episodes"] == [12] and src["replay"]["zarr"] == "wipe.zarr.zip"
    s = mod.summarize(run)
    assert s["source"]["source"] == "mixed"
    assert "RECORDED EPISODE(S) [12]" in mod.summary_text(s)


# ---------------------------------------------------------------------------
# 2026-09-20: the SAME recorder on the 3-POINT page (scripts/hl_serving.py) — a run of three wire points
# ---------------------------------------------------------------------------
_PAGE3 = _ROOT / "scripts" / "hl_serving.py"
_LAUNCHER3 = _ROOT / "scripts" / "run_hl_serving.sh"
_SRC3 = _PAGE3.read_text()
_REC3_MARK = "# ---------------- the RUN RECORDER (extension, ported from the five-point page 2026-09-20) ----------------"
THREE_LABELS = ["LEFT hand", "RIGHT hand", "head/torso"]
ONE_VIEW = [(0, "camera0_rgb", "ws://127.0.0.1:8773")]


def _synthetic_run_3pt(tmp_path) -> Path:
    """A run the recorder wrote OFF THE 3-POINT PAGE: 3 chunks x 4 steps of a (·,9)/(·,12) wire, a LEFT-only
    wrench, ``fcmd`` padded to 6 as the page records it, RecordTap ``vr_cmd_pos`` (3,3), the measured wire
    points FK'd (9,), one camera view."""
    ready = np.zeros(9)
    taps = R.RolloutTaps()
    rec = R.RunRecorder(tmp_path, "wipe_board_new", log=lambda s: None,
                        static_meta={"checkpoint": "checkpoints/umi_5pt_wipe_board_new/h50",
                                     "wire_labels": THREE_LABELS, "force_roles": ["left"],
                                     "action_dt": 1 / 30, "action_horizon": 8, "ready_pos": ready.tolist(),
                                     "page": "hl_serving (3-point wire)"},
                        measure=lambda msg: (np.asarray(msg["_kp"]), np.zeros(12)),
                        info_provider=lambda: {"exec_steps": 4, "force_cmd": True, "force_gate_n": 1.5},
                        cameras=ONE_VIEW)
    taps.recorder = rec
    cam = taps.camera_hook(0)
    rec.arm()
    assert rec.run_begin("sequential")
    t_step = 0
    for c in range(1, 4):
        k = np.arange(8)
        plan = np.tile(ready, (8, 1)) + 0.01 * (c - 1) * 4
        plan[:, 0] += 0.01 * k
        wl = np.zeros((8, 6))
        wl[:, 2] = -2.0 - 0.1 * k
        taps.chunk(seq=c, cadence="sequential", decoded_pos=plan[:, :3], pos=plan, orn=np.zeros((8, 12)),
                   wrench={"left": wl}, grip={"left": np.full(8, 0.06)}, seed=c, sample_ms=90.0, n_exec=4)
        for i in range(4):
            frame = (plan[i], np.zeros(12))
            taps.step(c, i, frame, {"left": wl[i]})
            taps.publish(frame, np.array([2.0, 0, 0, 0, 0, 0]), True)
            robot = plan[i].copy()
            robot[0] -= 0.010
            taps.on_record({"type": "record", "seq": t_step, "t": time.time(), "vive_armed": t_step > 0,
                            "dof_pos": [0.0] * 29, "vr_cmd_pos": plan[i].reshape(3, 3).tolist(),
                            "f_meas_ungated": [[2.5, 0, 0], [0, 0, 0]], "f_meas": [[2.5, 0, 0], [0, 0, 0]],
                            "f_cmd": [[2.0, 0, 0], [0, 0, 0]], "_kp": robot.tolist()}, time.monotonic())
            taps.on_force({"type": "ft", "t": time.time(), "left_pipe": (wl[i] + 0.5).tolist(), "right_pipe": None,
                           "left": (wl[i, :3] + 0.5).tolist(), "right": [0, 0, 0]}, time.monotonic())
            cam(_cam_msg(t_step + 1, _jpeg((20 * t_step) % 256)), time.monotonic())
            t_step += 1
            time.sleep(0.002)
    time.sleep(0.05)
    rec.run_end("sequential run finished — 20 s ceiling")
    return rec.last_dir


def test_a_three_point_run_loads_plots_and_summarises_with_three_labels(tmp_path):
    """The plotter's point count is the run's (``wire_labels``), not a constant: a 3-point folder's every
    width is 9 / 12 / (3,3), its horizons are (8, 9), its summary has three points, and the same
    ``main`` draws it."""
    P = _plot_module()
    d = _synthetic_run_3pt(tmp_path)
    run = P.load_run(d)
    assert run.n_points == 3 and run.labels == tuple(THREE_LABELS) and run.hands == ("left",)
    assert run.steps["target_pos"].shape == (12, 9) and np.isfinite(run.steps["target_pos"]).all()
    assert run.published["pos"].shape[1] == 9 and run.published["orn"].shape[1] == 12
    assert run.published["fcmd"].shape[1] == 6 and np.allclose(run.published["fcmd"][:, 0], 2.0)
    assert run.measured["kp_pos"].shape[1] == 9 and np.isfinite(run.measured["kp_pos"]).all()
    assert run.measured["vr_cmd_pos"].shape[1] == 9 and np.isfinite(run.measured["vr_cmd_pos"]).all()
    assert run.measured["kp_orn"].shape[1] == 12
    hz = P.chunk_horizons(run)
    assert [h["seq"] for h in hz] == [1, 2, 3] and all(h["pos"].shape == (8, 9) for h in hz)
    summary = P.summarize(run)
    assert list(summary["points_m"]) == THREE_LABELS
    err = summary["points_m"]["LEFT hand"]["x"]["robot_minus_published"]
    assert abs(err["mean"] + 0.010) < 1e-6         # the robot sat 10 mm behind the command in x
    assert list(summary["hands_N"]) == ["left"]     # the layout's one Bota; no invented right hand
    f = summary["hands_N"]["left"]["Fz"]["measured_minus_predicted"]
    assert abs(f["mean"] - 0.5) < 1e-6              # measured = predicted + 0.5 N by construction
    assert summary["hands_N"]["left"]["fcmd_on_wire"]["nonzero_frames"] == 12
    pytest.importorskip("matplotlib")
    assert P.main([str(d)]) == 0
    for name in ("keypoints_xyz.png", "force_xyz.png", "camera_strip.png", "summary.json"):
        assert (d / name).is_file() and (d / name).stat().st_size > 1000, name
    # a FIVE-point folder still reads as five (the fallback for a folder with no labels is five too)
    d5 = _synthetic_run(tmp_path / "five")
    run5 = P.load_run(d5)
    assert run5.n_points == 5 and run5.steps["target_pos"].shape[1] == 15
    assert P.N_POINTS == 5 and len(P.DEFAULT_LABELS) == 5


def test_the_paper_force_figure_reads_a_three_point_run(tmp_path):
    """``overleaf/tools/gen_fig_rollout_force_pred.py`` (the per-run paper figure the operator makes after
    every recording) is layout-agnostic: a 3-point run yields the LEFT hand's measured + predicted force."""
    pytest.importorskip("matplotlib")
    _plot_module()
    fig_path = _ROOT / "overleaf" / "tools" / "gen_fig_rollout_force_pred.py"
    spec = importlib.util.spec_from_file_location("gen_fig_rollout_force_pred", fig_path)
    F = importlib.util.module_from_spec(spec)
    sys.modules["gen_fig_rollout_force_pred"] = F
    spec.loader.exec_module(F)
    d = _synthetic_run_3pt(tmp_path)
    run = F.rec.load_run(d)
    assert F.hands_with_data(run) == ["left"]
    t_end, _why = F.resolve_t_end(run, None)
    prep = F.prepare(run, ["left"], t_end, 30.0)
    assert prep["left"]["pred"].shape == (12, 6) and prep["left"]["meas"].shape[1] == 6
    assert np.isfinite(prep["left"]["meas"]).any()
    assert F.main([str(d), "--no-video"]) == 0
    assert (d / "force_paper.png").is_file() and (d / "force_paper.json").is_file()
    assert json.loads((d / "force_paper.json").read_text())["hands"] == ["left"] if "hands" in json.loads(
        (d / "force_paper.json").read_text()) else True


def _rec3_block() -> str:
    return _SRC3[_SRC3.index(_REC3_MARK):_SRC3.index("# REACH COMPENSATION. Present even when the feature")]


def test_the_three_point_page_holds_the_taps_at_its_seams_and_the_recorder_in_one_block():
    main = _SRC3[_SRC3.index("def main(argv=None) -> int:"):_SRC3.index("def _run_viser(")]
    # main(): the holder, the three bridge hooks, the publish funnel, the three executors' chunk + step taps
    assert "taps = rollout_record.RolloutTaps()" in main
    assert 'HL._WSSubscriber' not in main                                   # this page's own subscriber class
    assert '"camera": _WSSubscriber(args.camera_ws, "camera", on_msg=taps.camera_hook(0))' in main
    assert '"record": _WSSubscriber(args.record_ws, "record", on_msg=taps.on_record)' in main
    force_hook = main[main.index("def _force_on_msg(msg, t_mono):"):main.index("subs = {")]
    assert force_hook.index("fn(msg, t_mono)") < force_hook.index("taps.on_force(msg, t_mono)")
    send = main[main.index("def _s0_send(pos, orn, fcmd):"):main.index("def _heartbeat_loop():")]
    assert send.index("s0.send(pos, orn, f_cmd=fcmd)") < send.index("taps.publish((pos, orn), _rollout_fcmd6(fcmd), True)")
    assert main.count("def _rollout_chunk(out: dict, cadence: str, n_exec: int) -> int:") == 1
    assert 'rseq = _rollout_chunk(out, "sequential", n_exec)' in main
    assert 'rseq = _rollout_chunk(out, "openloop", S)' in main
    assert '"ensembled", 1)' in main and "taps.step(_rseq, 0, (pos_i, orn_i)" in main
    drip = main[main.index("def _drip(out: dict) -> int:"):main.index("def _drip_openloop(out: dict) -> int:")]
    # the step tap records the MODEL's row (pre-compensation), before the compensated row is latched
    assert (drip.index("pos_i, orn_i = _wire_for_step(out, i)")
            < drip.index('taps.step(rseq, i, (wire_pos[i], out["wire_orn"][i])')
            < drip.index("_set_last_wire(pos_i, orn_i, f_i)"))
    assert 'out["sample_ms"] = (out["t_predict"] - _t_sample0) * 1e3' in main
    # 2026-09-20: the recording opens at the FIRST Predict (`_infer`) or the first chunk an executor
    # sees, and nothing on this page ends it but Reset / the disarm edge / the button / Ctrl-C
    infer = main[main.index("def _infer() -> dict:"):main.index("def _payload_from_chunk(")]
    assert infer.index('taps.begin("sequential")') < infer.index("seed = _next_seed()")
    chunk_fn = main[main.index("def _rollout_chunk(out: dict, cadence: str, n_exec: int) -> int:"):
                    main.index("def _gate_wire_force(")]
    assert chunk_fn.index("taps.begin(cadence)") < chunk_fn.index("r = taps.recorder")
    assert "taps.end(" not in _SRC3
    assert "taps=taps,\n    )" in main                                     # handed to the page
    assert "    taps=None,\n):" in _SRC3                                   # _run_viser's signature
    # teardown closes a recording left open, right after the take recorder's own flush
    assert main.index("recorder.close_pending()") < main.index('_rr.run_end("page shutdown")')
    # the ⏺ block: one folder under Control, the recorder installed on the holder, the measured-wire FK
    block = _rec3_block()
    page = _SRC3[_SRC3.index("def _run_viser("):]
    before = page[:page.index(_REC3_MARK)]
    for name in ("rollout_rec", "_rollout_toggle", "_rollout_refresh", "_rollout_reset", "_rollout_measure",
                 "_rollout_info", "_rollout_static", "b_roll", "c_roll_every", "g_roll", "RunRecorder",
                 "rollout_task", "rollout_dir", "plot_argv", "_rollout_cameras", "rollout_camera_every"):
        assert name not in before, name
        assert name in block, name
    assert block.count("add_folder(") == 1 and "order=f_control.order + 0.5" in block
    assert 'with gui.add_folder("Control") as f_control:' in _SRC3
    assert "taps.recorder = rollout_rec" in block
    assert 'ends_on="Reset / the low level disarming (joystick start) / this button"' in block
    assert "meas = _measured_full_wire(rframe.dof_pos)" in block and "return meas[2], meas[3]" in block
    assert '"wire_labels": ["LEFT hand", "RIGHT hand", "head/torso"]' in block
    assert '_rollout_cameras = [{"cam": 0, "key": _rollout_image_key, "url": subs["camera"].url}]' in block
    for forbidden in ("_s0_send(", "_set_last_wire(", "planner.", "busy.acquire", "dripping.set", "stop.set()",
                      "stop.clear()", "s0.send", "_commit_left_step", "commit_role", "mapper.set_hold_wire"):
        assert forbidden not in block, forbidden
    # Reset → idle; the status loop refreshes the button; ▶▶ and the two replay Plays are bracketed
    reset = page[page.index("@reset_btn.on_click"):page.index("@tare_btn.on_click")]
    assert reset.index('_record_finish("reset start pose")') < reset.index("_rollout_reset()")
    tick = page[page.index("def _status_tick():"):page.index("def _status_loop():")] if (
        page.index("def _status_tick():") < page.index("def _status_loop():")) else page[page.index("def _status_tick():"):]
    assert "_rollout_refresh()" in tick[:tick.index("REACH COMPENSATION readout")]
    start = page[page.index("@auto_run_btn.on_click"):page.index("@auto_stop_btn.on_click")]
    assert re.search(r'taps\.wrap_run\(_run_to_completion, "sequential",\s*end_note=lambda: str\(ctrl_status\.value\)\)', start)
    # ⏹ retires the loop and touches the recorder NOT AT ALL; the LOW LEVEL LOST watchdog logs an event
    i_retire = page.index("@auto_stop_btn.on_click")
    retire = page[i_retire:page.index("request_ens_reset(ens_ctl, ens_lock)", i_retire)]
    assert "taps." not in retire
    assert 'taps.event("low_level_lost"' in tick
    assert 'taps.wrap_run(_replay_play_sequential, "replay"' in page
    assert 'taps.wrap_run(_replay_play_open_loop, "replay-openloop"' in page
    # the flags, under their own prefix (--record/--record-dir are the deploy take's on this page)
    for flag in ("--rollout-dir", "--rollout-task", "--rollout-every-run", "--no-rollout-autoplot",
                 "--rollout-camera-every"):
        assert f'"{flag}"' in main, flag
    assert 'dest="rollout_autoplot", action="store_false", default=True' in main


def test_the_three_point_launcher_carries_task_and_the_record_knobs():
    sh = _LAUNCHER3.read_text()
    assert subprocess.run(["bash", "-n", str(_LAUNCHER3)], capture_output=True).returncode == 0
    task_block = sh[sh.index('if [ -z "$CKPT" ] && [ -n "${TASK:-}" ]; then'):sh.index('if [ "$ALGO" = "muse" ]; then')]
    assert 'CKPT="$FLORE_ROOT/checkpoints/umi_5pt_${TASK}/h${H:-50}"' in task_block
    assert 'TASK_EXTRA+=(--rollout-task "$TASK")' in task_block
    assert task_block.count('--rollout-task "$TASK"') == 2                 # also when an explicit --checkpoint won
    assert 'scripts/fetch_checkpoint.sh ${TASK}_5pt_h${H:-50}' in task_block
    # --algo read off config.json when omitted (the pin snippet prints horizon AND algo)
    assert "print(horizon, algo)" in sh and 'read -r CKPT_H CKPT_ALGO <<<"$_CKPT_PIN"' in sh
    assert 'TASK_EXTRA+=(--algo "$ALGO")' in sh
    # the recorder's env knobs, explicit flags winning; RECORD_DIR is named as the OTHER launcher's knob
    assert "EXTRA+=(--rollout-every-run)" in sh
    assert 'EXTRA+=(--rollout-camera-every "$RECORD_CAMERA_EVERY")' in sh
    assert 'EXTRA+=(--rollout-dir "$ROLLOUT_DIR")' in sh
    assert 'if [ -n "${RECORD_DIR:-}" ]; then' in sh
    tail = sh[sh.rindex('"${ANCHOR_ARGS[@]}"'):]
    assert tail.index('"${TASK_EXTRA[@]}"') < tail.index('"${EXTRA[@]}"') < tail.index('"$@"')
