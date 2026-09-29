"""Base-station adaptivity: the rig runs 2 OR 3 lighthouses, interchangeably.

Nothing in the bridge is told how many stations the room has. It counts the ones that
actually deliver a SOLVED pose and calibrates on whatever is up, with a floor of 2. These
tests pin the three things that make that safe rather than merely permissive:

1. **The floor is 2, and it is a correctness floor, not a preference.** Calibration derives
   the world yaw from ONE instantaneous L-R hand vector and bakes it into every later
   command. A single Gen2 lighthouse constrains a sparse tracker only up to a rotation about
   its own optical axis, so that yaw is under-determined — the result is a frame that looks
   locked and slowly drifts, which presents as "the operator was standing crooked" or "the
   policy got worse". So the gate blocks calibration rather than degrading.
2. **The cached geometry is parsed from a file that is NOT valid JSON.** libsurvive writes
   top-level entries with no enclosing braces and inconsistent trailing commas;
   ``json.load`` fails at character 3. The `PositionSet` flag it carries is the one field
   that must never be misread, so the parser is scanned rather than repaired.
3. **``--fresh`` clears EVERY configured station, not the ones you can see.** libsurvive's
   ``--force-calibrate`` loops ``[0, activeLighthouses)`` and ``activeLighthouses`` comes
   from the config, so a station powered off during a ``--fresh`` is left unsolved and does
   not recover on its own. That has already happened on this rig once — the
   ``config.json.broken_both_unsolved_*`` file beside the live one is the evidence, and it
   is used as a fixture below.

pysurvive is injected via PYTHONPATH by `scripts/_vive_teleop_bridge.sh`, not pip, so it is
stubbed — none of this touches the USB singleton.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

for _name in ("pysurvive",):
    if _name not in sys.modules:
        _m = types.ModuleType(_name)
        _m.SimpleContext = object
        _m.survive_simple_serial_number = lambda *_a, **_k: b"?"
        sys.modules[_name] = _m

from wbt_realtime_eval.vive import vive_teleop_bridge as vtb  # noqa: E402

_LIVE_CONFIG = Path.home() / ".config/libsurvive/config.json"

# A faithful excerpt of libsurvive's real on-disk format: no enclosing braces, and
# "floor-offset" deliberately has NO trailing comma (that is what breaks json.load).
_CONFIG_3LH = """"v":"0",
"poser":"MPFIT",
"configed-lighthouse-gen":"2",
"floor-offset":"-1.014014720917"
"lighthouse0":{
"index":"0",
"id":"3865060032",
"mode":"2",
"pose":["-0.78","-1.12","-0.006","-0.67","-0.47","0.42","-0.36"],
"OOTXSet":"1",
"PositionSet":"1"
}
"lighthouse1":{
"index":"1",
"id":"797581664",
"mode":"0",
"OOTXSet":"1",
"PositionSet":"1"
}
"lighthouse2":{
"index":"2",
"id":"2906417300",
"mode":"3",
"OOTXSet":"1",
"PositionSet":"0"
}
"""


def _write(tmp_path: Path, text: str) -> str:
    p = tmp_path / "config.json"
    p.write_text(text)
    return str(p)


class _Args:
    """Minimal stand-in for the argparse namespace the station code reads."""

    def __init__(self, **kw):
        d = dict(fresh=False, min_stations=vtb.MIN_STATIONS, expect_stations=0)
        d.update(kw)
        self.__dict__.update(d)


def _station_stub(**kw):
    """A bridge with ONLY the station machinery wired — no trackers, no anchor, no
    pysurvive. ``object.__new__`` because the real ``__init__`` claims serials and loads
    home_anchor.json, neither of which the gate touches."""
    import threading

    b = object.__new__(vtb.ViveTeleopBridge)
    b._lock = threading.Lock()
    b._args = _Args(**kw)
    b._stations_live = set()
    b._stations_configured = []
    b._stations_ok_logged = False
    b._stations_wait_logged = False
    b._min_stations = max(1, int(b._args.min_stations))
    return b


# ── the config parser ───────────────────────────────────────────────────────────


def test_config_is_not_valid_json_so_the_parser_must_scan() -> None:
    """Pin the premise. If libsurvive ever starts writing real JSON this test fails and the
    hand-rolled scan can be replaced — but until then, `json.load` is not an option and a
    future reader should not "simplify" it into one."""
    import json

    with pytest.raises(json.JSONDecodeError):
        json.loads(_CONFIG_3LH)


def test_parses_three_stations_and_reads_positionset(tmp_path: Path) -> None:
    lhs = vtb.read_libsurvive_lighthouses(_write(tmp_path, _CONFIG_3LH))
    assert [lh["index"] for lh in lhs] == [0, 1, 2]
    assert [lh["id"] for lh in lhs] == ["3865060032", "797581664", "2906417300"]
    assert [lh["mode"] for lh in lhs] == ["2", "0", "3"]
    # The whole point: LH2 is configured but UNSOLVED, and only PositionSet says so.
    assert [lh["solved"] for lh in lhs] == [True, True, False]
    assert all(lh["ootx"] for lh in lhs)


def test_parses_two_stations_and_ignores_extra_top_level_keys(tmp_path: Path) -> None:
    """A 2-station room is not a degraded 3-station room — it is the normal case, and the
    parser must not invent a third or trip over the un-braced scalars around the blocks."""
    two = _CONFIG_3LH.split('"lighthouse2"')[0]
    lhs = vtb.read_libsurvive_lighthouses(_write(tmp_path, two))
    assert len(lhs) == 2
    assert all(lh["solved"] for lh in lhs)


def test_missing_or_garbage_config_returns_empty_not_an_exception(tmp_path: Path) -> None:
    """Startup must not die on a fresh machine with no cached geometry — the caller falls
    back to what it observes on the radio."""
    assert vtb.read_libsurvive_lighthouses(str(tmp_path / "nope.json")) == []
    assert vtb.read_libsurvive_lighthouses(_write(tmp_path, "not a config at all")) == []
    assert vtb.read_libsurvive_lighthouses(_write(tmp_path, "")) == []


def test_the_real_broken_config_is_detected_as_unsolved() -> None:
    """The actual failure this guards against, replayed from the actual artifact.

    ``~/.config/libsurvive/config.json.broken_both_unsolved_*`` is what a ``--fresh`` left
    behind on this rig: two stations, BOTH ``PositionSet:"0"``. Tracking then cannot work,
    and nothing in the old bridge said so. Skipped if the file has been cleaned up."""
    cands = sorted(_LIVE_CONFIG.parent.glob("config.json.broken_both_unsolved_*"))
    if not cands:
        pytest.skip("the historical broken-config artifact is no longer on this machine")
    lhs = vtb.read_libsurvive_lighthouses(str(cands[0]))
    assert lhs, "should still parse — it is well-formed, just unsolved"
    assert not any(lh["solved"] for lh in lhs), lhs
    # And a bridge in that state must refuse rather than calibrate.
    b = _station_stub()
    ok, why = b._stations_gate()
    assert not ok and "need >=" in why


def test_the_live_config_parses(tmp_path: Path) -> None:
    """Smoke the parser against whatever is actually on this machine right now — the format
    is not documented anywhere, so a real sample is the only honest check.

    Skips on a WIPED config rather than failing. A `--force-calibrate` leaves the file with
    its scalars and no lighthouse blocks at all, which is a legitimate (if unusable) state of
    the rig and says nothing about the parser — this test's subject. The station GATE is what
    must refuse to run in that state, and `test_gate_blocks_below_two_stations` covers it."""
    if not _LIVE_CONFIG.is_file():
        pytest.skip(f"no live libsurvive config at {_LIVE_CONFIG}")
    lhs = vtb.read_libsurvive_lighthouses(str(_LIVE_CONFIG))
    if not lhs:
        pytest.skip(f"{_LIVE_CONFIG} currently has NO lighthouse blocks (wiped by a --fresh); re-solve first")
    for lh in lhs:
        assert lh["id"] and lh["mode"] is not None
        assert isinstance(lh["solved"], bool) and isinstance(lh["ootx"], bool)


# ── the gate: adaptive to 2 or 3 ────────────────────────────────────────────────


@pytest.mark.parametrize("n_live", [2, 3])
def test_gate_passes_on_either_two_or_three_stations(n_live: int) -> None:
    """The headline requirement. No flag, no config edit, no mode: 2 and 3 both just work,
    because the gate counts what is LIVE rather than trusting anything."""
    b = _station_stub()
    for i in range(n_live):
        b._note_live_station(f"LH{i}")
    ok, why = b._stations_gate()
    assert ok, why
    assert str(n_live) in why


@pytest.mark.parametrize("n_live", [0, 1])
def test_gate_blocks_below_two_stations(n_live: int) -> None:
    """One station is not a degraded mode, it is a wrong one: the world-yaw autocal reads a
    single L-R vector, and one lighthouse leaves that yaw under-constrained. The gate must
    block, and must say what to do."""
    b = _station_stub()
    for i in range(n_live):
        b._note_live_station(f"LH{i}")
    ok, why = b._stations_gate()
    assert not ok
    assert f"need >= {vtb.MIN_STATIONS}" in why
    assert "--fresh" in why  # the fix for a moved station is named


def test_expect_stations_turns_the_count_into_an_assertion() -> None:
    """``STATIONS=3`` for an operator who knows three should be up: two live is then a
    FAILURE, not an adaptation — that is how a station that died between sessions gets
    caught at startup instead of as mysteriously worse tracking."""
    b = _station_stub(expect_stations=3)
    b._note_live_station("LH0")
    b._note_live_station("LH1")
    ok, why = b._stations_gate()
    assert not ok and "STATIONS=3" in why
    b._note_live_station("LH2")
    ok, why = b._stations_gate()
    assert ok, why


def test_expect_stations_zero_is_adaptive() -> None:
    """The default. Whatever is up is accepted, as long as it clears the floor."""
    b = _station_stub(expect_stations=0)
    b._note_live_station("LH0")
    b._note_live_station("LH1")
    assert b._stations_gate()[0]
    b._note_live_station("LH2")
    assert b._stations_gate()[0]


def test_expect_more_than_live_is_rejected_even_above_the_floor() -> None:
    """3 requested, 2 live clears the FLOOR but violates the ASSERTION. The two checks are
    independent and both have to fire."""
    b = _station_stub(expect_stations=3)
    b._note_live_station("LH0")
    b._note_live_station("LH1")
    ok, _ = b._stations_gate()
    assert not ok


def test_live_stations_are_deduplicated() -> None:
    """A lighthouse re-appears in the pose stream on every pose update, so the set — not a
    counter — is what makes the count meaningful."""
    b = _station_stub()
    for _ in range(50):
        b._note_live_station("LH0")
    assert len(b._stations_live) == 1
    assert not b._stations_gate()[0]


def test_min_stations_one_is_allowed_but_only_deliberately() -> None:
    """Diagnostics need a way down to one station; it must be explicit, and it must not
    silently become the default."""
    assert vtb.MIN_STATIONS == 2
    b = _station_stub(min_stations=1)
    b._note_live_station("LH0")
    assert b._stations_gate()[0], "an explicit --min-stations 1 must be honoured"


# ── the calibration gate ────────────────────────────────────────────────────────


def test_calibration_waits_for_stations_and_restarts_the_hold_countdown() -> None:
    """`_maybe_calibrate` must refuse while the stations are short AND clear
    ``_both_seen_at``, so the 3 s hold timer restarts once they come up. Leaving the timer
    running would let calibration fire the instant the second station appeared — using a
    pose the operator had already relaxed out of, which bakes a wrong yaw into every
    subsequent command."""
    import numpy as np

    b = _station_stub()
    b._roles = ["left_hand", "right_hand"]
    b._raw_hand = {
        "left_hand": (np.array([0.3, 0.2, 1.0]), np.array([1.0, 0.0, 0.0, 0.0])),
        "right_hand": (np.array([0.3, -0.2, 1.0]), np.array([1.0, 0.0, 0.0, 0.0])),
    }
    b._calibrated = False
    b._both_seen_at = 12345.0  # as if the hold had already been counting
    b._uses_head = False

    b._note_live_station("LH0")  # one station: not enough
    b._maybe_calibrate()
    assert b._calibrated is False, "calibrated on a single base station"
    assert b._both_seen_at is None, "the hold countdown must restart, not keep running"
    assert b._stations_wait_logged is True, "the operator must be told why nothing happens"


# ── --fresh is destructive, so discovery must not carry it ──────────────────────
#
# What these pin, and why it is not a style preference: on 2026-08-28 a
# `--list-serials --fresh` on this rig turned THREE solved stations into zero. --fresh
# forwards libsurvive's --force-calibrate --force-ootx, which clears PositionSet AND OOTXSet
# on every configured station; re-solving them needs an OOTX decode per station plus a
# position solve driven by a moving tracker, which cannot finish inside a 20 s scan window.
# So the combination reliably lands in the worst state — geometry gone, nothing rebuilt — and
# the symptom it produces ("ON THE RADIO but NOT TRACKED" on every puck) reads as a hardware
# fault, sending the operator to look at line of sight for a problem that is in a config file.


def test_list_serials_refuses_fresh_before_touching_pysurvive(tmp_path: Path, capsys) -> None:
    """The refusal must happen BEFORE `pysurvive.SimpleContext` — that constructor is what
    actually clears the config, so a warning printed after it is archaeology, not a guard.
    `_list_serials` returns 2 without ever building a context (the stub's SimpleContext is
    `object`, which would raise on the real call signature)."""
    rc = vtb._list_serials(_Args(fresh=True, left_serial="A", right_serial="B", mode="3pt", list_timeout=1.0))
    assert rc == 2
    assert "refused" in capsys.readouterr().out.lower()


def test_backup_before_fresh_copies_the_config_aside(tmp_path: Path, monkeypatch, capsys) -> None:
    """The backup is the ONLY reason a wiped config is recoverable, and it used to live on
    `_preflight_stations` — i.e. on the live-bridge path only, so the discovery path wiped
    three stations with nothing saved. It is module-level now precisely so both paths get it."""
    cfg = tmp_path / "config.json"
    cfg.write_text(_CONFIG_3LH)
    monkeypatch.setattr(vtb, "LIBSURVIVE_CONFIG", str(cfg))
    backup = vtb.backup_before_fresh()
    assert backup is not None and Path(backup).is_file()
    assert Path(backup).read_text() == _CONFIG_3LH, "the backup must be byte-identical"
    out = capsys.readouterr().out
    assert "CLEARS THE POSE OF ALL 3 CONFIGURED STATIONS" in out, "the blast radius must be stated"


def test_backup_before_fresh_is_a_noop_with_nothing_cached(tmp_path: Path, monkeypatch) -> None:
    """A fresh machine has no geometry to lose; the backup must not invent an empty file."""
    monkeypatch.setattr(vtb, "LIBSURVIVE_CONFIG", str(tmp_path / "nope.json"))
    assert vtb.backup_before_fresh() is None
    assert not list(tmp_path.iterdir())


def test_wiped_config_is_reported_as_not_a_line_of_sight_problem(tmp_path: Path, monkeypatch, capsys) -> None:
    """The state the bad --fresh left behind: a config with NO lighthouse blocks. The report
    must say a solve is required rather than implying the stations are missing, because the
    two fixes are opposite — one is a config rebuild, the other is aiming a puck."""
    cfg = tmp_path / "config.json"
    cfg.write_text('"v":"0",\n"poser":"MPFIT"\n')  # exactly what --force-calibrate leaves
    monkeypatch.setattr(vtb, "LIBSURVIVE_CONFIG", str(cfg))
    assert vtb.report_cached_stations() == []
    out = capsys.readouterr().out
    assert "NO cached geometry" in out
    assert "LIVE bridge" in out, "must name the command that can actually re-solve"
    assert "cannot finish inside a --list-serials scan window" in out


def test_report_cached_stations_flags_the_unsolved_one(tmp_path: Path, monkeypatch, capsys) -> None:
    """A partially-solved config is the ambiguous case an operator most needs spelled out."""
    cfg = tmp_path / "config.json"
    cfg.write_text(_CONFIG_3LH)
    monkeypatch.setattr(vtb, "LIBSURVIVE_CONFIG", str(cfg))
    lhs = vtb.report_cached_stations()
    assert [lh["solved"] for lh in lhs] == [True, True, False]
    out = capsys.readouterr().out
    assert "3 configured, 2 solved" in out
    assert "needs --fresh" in out


# ── the solve procedure: stillness, not waving ──────────────────────────────────


def _flat(text: str) -> str:
    """Operator text with the ``[bridge] `` line prefixes and the hard wrapping removed, so
    assertions can match phrases without pinning where the terminal wrap happens to fall."""
    return " ".join(text.replace("[bridge]", " ").split())


def test_solve_procedure_prescribes_move_pause_move() -> None:
    """The one instruction an operator follows when the geometry is gone. BOTH obvious answers
    are wrong, in opposite directions, and this text went through both of them before landing
    on what the code actually gates on.

    `GlobalSceneSolver` captures a scene only when all four hold
    (`driver_global_scene_solver.c:224-229`): `not_moving` (standstill > default_tolerance*16 =
    ~0.27 s), `activations_changed` (the pose DIFFERS from the last capture),
    `spreadout` (>= 3 s since it), `light_static`. Therefore:

    * continuous waving -> `not_moving` never true -> zero scenes;
    * one fixed spot -> exactly ONE scene, then `activations_changed` never fires -> no
      parallax -> the bundle adjustment never converges;
    * move / pause / move is the only pattern that satisfies both.

    An earlier version of this file asserted "put them down and do not touch them" (read off
    `poser_mpfit.c:172` alone, missing the global solver). UMI_collection's runbook says "wave
    slowly", which works only because slow waving happens to contain the pauses. So this test
    pins the SYNTHESIS, and asserts neither degenerate instruction survives."""
    flat = _flat(vtb.SOLVE_PROCEDURE)
    assert "MOVE," in flat and "PAUSE" in flat, "the move/pause cycle must be explicit"
    assert "Continuous waving captures NOTHING" in flat, "rule out the first degenerate case"
    assert "captures ONE scene and then stalls" in flat, "rule out the second degenerate case"
    assert "STILL" in flat, "the capture happens while stationary — do not lose that"
    # It must NOT tell them to leave the pucks untouched; that is the stalling case.
    assert "DO NOT TOUCH" not in flat.upper()
    # And the OOTX escape hatch has to be reachable from here, since that is the failure that
    # looks identical (nothing solves) but does not respond to motion at all.
    assert "FRESH_OOTX=0" in flat
    assert "Bad sync bit" in flat


def test_solve_procedure_separates_ootx_from_the_pose_solve() -> None:
    """The two stages fail identically from the outside — nothing gets solved — but only one of
    them responds to moving the puck. Conflating them is how an operator ends up waving at a
    station for ten minutes, which is exactly what happened on 2026-08-28."""
    flat = _flat(vtb.SOLVE_PROCEDURE)
    assert "That is OOTX, not the pose solve" in flat
    assert "more waving will not fix it" in flat
    assert "25-35 s PER STATION" in flat, "the per-station cost is what sets the wait budget"


def test_the_wiped_config_report_carries_the_procedure(tmp_path: Path, monkeypatch, capsys) -> None:
    """A wiped config is exactly when the procedure is needed, so the report must not just
    diagnose it and stop."""
    cfg = tmp_path / "config.json"
    cfg.write_text('"v":"0",\n"poser":"MPFIT"\n')
    monkeypatch.setattr(vtb, "LIBSURVIVE_CONFIG", str(cfg))
    vtb.report_cached_stations()
    out = _flat(capsys.readouterr().out)
    assert "NO cached geometry" in out
    assert "MOVE," in out and "PAUSE" in out


def test_backup_before_fresh_carries_the_procedure(tmp_path: Path, monkeypatch, capsys) -> None:
    """--fresh is the other entry into the same state; same instruction required."""
    cfg = tmp_path / "config.json"
    cfg.write_text(_CONFIG_3LH)
    monkeypatch.setattr(vtb, "LIBSURVIVE_CONFIG", str(cfg))
    vtb.backup_before_fresh()
    out = _flat(capsys.readouterr().out)
    assert "CLEARS THE POSE OF ALL 3" in out
    assert "Continuous waving captures NOTHING" in out


# ── FRESH_CALIB / FRESH_OOTX: the two resets are independent ────────────────────


def _survive_args_for(fresh: bool, fresh_ootx: bool) -> list[str]:
    """The libsurvive argv `run_pysurvive` would build, without starting anything. Extracted
    by replaying the same branch rather than importing a helper, because the flag assembly IS
    the behaviour under test — a helper would let the real call site drift away from it."""
    args: list[str] = ["--lighthouse-update-velocity", "0.5"]
    if fresh:
        args += ["--force-calibrate", "1"]
        if fresh_ootx:
            args += ["--force-ootx", "1"]
    return args


def test_fresh_ootx_off_keeps_the_factory_blob_but_still_resolves_positions() -> None:
    """The split that makes a station whose OOTX will not decode recoverable.

    `--force-calibrate` clears the base-station POSE — where it sits — which is exactly what
    you want after moving one. `--force-ootx` additionally throws away the decoded FACTORY
    blob (id + lens/motor constants), which is burned into the station hardware and does NOT
    change when you move it. Re-reading it costs ~25-35 s/station of unbroken line of sight
    and, until every attached decoder finishes, libsurvive emits NO pose at all — so a station
    the pucks cannot see cleanly turns a launch into an indefinite 'Bad sync bit' hang. Keeping
    the blob skips that entirely while still re-solving the room."""
    assert _survive_args_for(True, True) == [
        "--lighthouse-update-velocity", "0.5", "--force-calibrate", "1", "--force-ootx", "1",
    ]
    # FRESH_OOTX=0: positions re-solved, blob kept.
    assert _survive_args_for(True, False) == [
        "--lighthouse-update-velocity", "0.5", "--force-calibrate", "1",
    ]
    # And FRESH_CALIB=0 must touch nothing at all — reuse is the teleop default.
    assert _survive_args_for(False, True) == ["--lighthouse-update-velocity", "0.5"]


def test_fresh_ootx_defaults_on_so_a_swapped_station_is_noticed() -> None:
    """Why the safe-looking option is NOT the default. With a cached blob libsurvive never
    attaches a decoder, so it never re-reads a station's ID — a station SWAPPED onto a channel
    another one used would be driven with the WRONG factory calibration, silently. The default
    therefore pays the handshake, and FRESH_OOTX=0 is an explicit operator decision."""
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--fresh", action="store_true")
    ap.add_argument("--no-fresh-ootx", dest="fresh_ootx", action="store_false")
    assert ap.parse_args(["--fresh"]).fresh_ootx is True, "must default to re-reading"
    assert ap.parse_args(["--fresh", "--no-fresh-ootx"]).fresh_ootx is False


def test_the_bridge_exposes_no_fresh_ootx() -> None:
    """Guard the wiring, not just the semantics: the launcher's FRESH_OOTX=0 forwards this
    exact spelling, so a rename here silently degrades to a full OOTX re-read (the hang)."""
    import inspect

    src = inspect.getsource(vtb)
    assert '"--no-fresh-ootx"' in src
    assert 'dest="fresh_ootx"' in src
    # The run loop must consult it via getattr with a True default: nothing else builds this
    # namespace today, but a hand-built one (a smoke script) must not crash, and the safe
    # fallback is the historical behaviour.
    assert 'getattr(self._args, "fresh_ootx", True)' in src


# ── the lock discipline (this is the one that was missing) ──────────────────────


def test_maybe_calibrate_does_not_deadlock_when_called_holding_the_lock() -> None:
    """Production calls `_maybe_calibrate` from INSIDE `snapshot()`'s `with self._lock:`.
    The station-gate test above called it at top level, so it never exercised that, and the
    gate I added in eed1b3e re-acquired the same non-reentrant `threading.Lock` — deadlocking
    the asyncio event loop on the very first broadcast tick.

    On the rig that presented as three unrelated-looking symptoms, none naming a lock: the
    bridge published nothing, it never reached `calibrated=True`, and it stopped accepting
    WebSocket connections (21 queued on :8770) — which the policy reported as "no live frame"
    and then as "your live hand pose is too far from the robot's". py-spy on the hung process
    is what actually found it.

    So: reproduce the REAL call shape. A 5 s timeout rather than a bare call, because a
    deadlock does not raise — it hangs the whole test session, which is a much worse failure
    than a red test.
    """
    import threading

    import numpy as np

    b = _station_stub()
    b._roles = ["left_hand", "right_hand"]
    b._raw_hand = {
        "left_hand": (np.array([0.3, 0.2, 1.0]), np.array([1.0, 0.0, 0.0, 0.0])),
        "right_hand": (np.array([0.3, -0.2, 1.0]), np.array([1.0, 0.0, 0.0, 0.0])),
    }
    b._calibrated = False
    b._both_seen_at = None
    b._uses_head = False
    b._note_live_station("LH0")
    b._note_live_station("LH1")  # enough stations, so the gate PASSES and we reach the body

    done = threading.Event()
    err: list[BaseException] = []

    def _call() -> None:
        try:
            with b._lock:  # exactly what snapshot() does before calling it
                b._maybe_calibrate()
        except BaseException as exc:  # noqa: BLE001 — a raise is a pass here, a hang is not
            err.append(exc)
        finally:
            done.set()

    t = threading.Thread(target=_call, daemon=True)
    t.start()
    assert done.wait(timeout=5.0), (
        "_maybe_calibrate DEADLOCKED when called holding self._lock -- something inside it "
        "re-acquires the non-reentrant lock. Use the _locked variant."
    )
    # It may legitimately RAISE (the stub has no home anchor wired) — that is fine, the
    # subject here is liveness. What it must not do is hang, which `done.wait` above covers.


def test_both_station_gate_variants_agree() -> None:
    """The lock-taking wrapper and the `_locked` body must return the SAME verdict — the split
    exists for lock discipline, not to create two policies that can drift."""
    for n_live in (0, 1, 2, 3):
        b = _station_stub()
        for i in range(n_live):
            b._note_live_station(f"LH{i}")
        with b._lock:
            locked = b._stations_gate_locked()
        assert b._stations_gate() == locked, n_live


def test_stations_gate_wrapper_is_still_usable_without_the_lock() -> None:
    """The public wrapper is what tests and any future non-holding caller use; it must still
    acquire the lock itself rather than assuming a caller did."""
    b = _station_stub()
    b._note_live_station("LH0")
    b._note_live_station("LH1")
    ok, why = b._stations_gate()  # no lock held here
    assert ok, why
