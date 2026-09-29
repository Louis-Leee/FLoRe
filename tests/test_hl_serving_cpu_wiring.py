"""The CPU throttles in ``scripts/hl_serving.py``, and the lines that must NOT ride them.

The planner is one GIL-bound process, so its ceiling is one core; measured on the 24-core deploy
box it sat at ~70% of that core with nothing predicted, and the recorded takes show the 30 Hz
render loop achieving 15-17 Hz. Two of the three biggest costs were rate-limited (the idle
clutch's FK at 60 Hz, the live-mesh repose at 30 Hz) and the websocket compression was turned
off. Each has a specific way to be silently broken later, which is what these pin:

* throttling the mesh block TOO FAR — sweeping the RecordTap parse, ``live_dof``/``live_armed``/
  ``f_meas`` or the deploy recorder into the same guard — would halve the recorded measured
  streams' rate. That is invisible on screen and only shows up much later as a half-rate
  ``dof_pos.npy`` in an exec-gap analysis.
* rate-limiting the wire REPUBLISH along with the clutch would cut the :8770 refresh from
  ``drip_hz`` to ``IDLE_CLUTCH_HZ``, eating the margin against ``S0Driver.FRESHNESS_TIMEOUT_S``
  and holosoma's 0.25 s stale gate. The whole point of the split is that the cheap republish
  keeps its rate and only the expensive re-derivation slows down.

Source-level, for the same reason ``test_hl_serving_gil_wiring.py`` is: these seams live inside
``main()``/``_run_viser()``'s closures and cannot be imported without standing up a policy, five
sockets and a viser page. Structure is what matters here, and a detached guard is exactly what a
structural test catches.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

_SRC_PATH = Path(__file__).resolve().parents[1] / "scripts" / "hl_serving.py"
_SRC = _SRC_PATH.read_text()
_TREE = ast.parse(_SRC)

_DRIVER_PATH = Path(__file__).resolve().parents[1] / "fuji" / "client" / "s0_driver.py"


def _fn_node(name: str) -> ast.FunctionDef:
    for node in ast.walk(_TREE):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"function {name} not found in hl_serving.py")


def _dotted(func: ast.expr) -> str:
    """``a.b.c`` for an attribute/name call target, else ''."""
    parts: list[str] = []
    while isinstance(func, ast.Attribute):
        parts.append(func.attr)
        func = func.value
    if isinstance(func, ast.Name):
        parts.append(func.id)
    elif isinstance(func, ast.Subscript):  # e.g. wire_sync["fn"](...)
        parts.append("<subscript>")
    else:
        return ""
    return ".".join(reversed(parts))


def _calls(node: ast.AST) -> set[str]:
    return {_dotted(n.func) for n in ast.walk(node) if isinstance(n, ast.Call)}


def _assigned(node: ast.AST) -> set[str]:
    out: set[str] = set()
    for n in ast.walk(node):
        if isinstance(n, ast.Assign):
            for t in n.targets:
                if isinstance(t, ast.Name):
                    out.add(t.id)
    return out


def _render_loop() -> ast.While:
    """The render loop: the ``while`` in ``_run_viser`` that polls the replay path."""
    for node in ast.walk(_fn_node("_run_viser")):
        if isinstance(node, ast.While) and "_poll_replay_path" in _calls(node):
            return node
    raise AssertionError("could not find the render loop in _run_viser")


def _guard_containing(root: ast.AST, marker: str) -> ast.If:
    """The innermost ``if`` whose TEST mentions ``marker``."""
    hits = [n for n in ast.walk(root)
            if isinstance(n, ast.If) and marker in ast.dump(n.test)]
    assert len(hits) == 1, f"expected exactly one guard testing {marker!r}, found {len(hits)}"
    return hits[0]


# --- the live-mesh repose ------------------------------------------------------------------

def test_the_live_mesh_repose_is_throttled():
    """``ViserUrdf.update_cfg`` measured 3.4 ms wall / 3.7 ms CPU — 11% of a core at 30 Hz, all
    GIL-held, on the thread whose latency the operator feels."""
    guard = _guard_containing(_render_loop(), "_mesh_last_draw")
    assert "urdf.update_cfg" in _calls(guard), (
        "the live-mesh repose is no longer inside the _mesh_last_draw guard")


def test_no_unthrottled_live_mesh_repose_in_the_render_loop():
    """A second, ungated ``urdf.update_cfg`` anywhere in the loop would undo the throttle."""
    loop = _render_loop()
    guard = _guard_containing(loop, "_mesh_last_draw")
    guarded = {id(n) for n in ast.walk(guard)}
    stray = [n for n in ast.walk(loop)
             if isinstance(n, ast.Call) and _dotted(n.func) == "urdf.update_cfg"
             and id(n) not in guarded]
    assert not stray, f"{len(stray)} unthrottled urdf.update_cfg call(s) in the render loop"


def test_the_measurements_do_not_ride_the_mesh_throttle():
    """The reason the guard is around ONE call and not the block: everything else there is a
    measurement the deploy recorder and the executed trace consume at full rate."""
    guard = _guard_containing(_render_loop(), "_mesh_last_draw")
    body_calls = _calls(guard)
    for forbidden in ("parse_record_frame", "recorder.on_executed", "trace.add",
                      "_measured_full_wire"):
        assert forbidden not in body_calls, (
            f"{forbidden} is inside the 15 Hz mesh throttle — the recorded measured streams "
            f"would silently drop to the mesh rate (they are already at the render rate, which "
            f"is the bug this throttle exists to relieve)")
    for forbidden in ("live_dof", "live_armed", "live_fmeas", "live_fmeas_ungated", "live_tcp",
                      "live_posed"):
        assert forbidden not in _assigned(guard), (
            f"{forbidden} is assigned inside the mesh throttle, so on a skipped tick it would "
            f"keep a stale value (or None) and the recorder would log a hole")


def test_one_throttle_clock_per_render_tick():
    """The three throttles (mesh 15 Hz, F/T charts 8 Hz, executed trace 4 Hz) share ONE ``_now``,
    so they cannot disagree about when "now" is inside a tick. Other ``monotonic()`` calls in the
    loop are fine and deliberate — ``trace.add`` stamps a SAMPLE and must read the clock at the
    moment it records — so what is pinned is the single assignment, not the single call."""
    loop = _render_loop()
    binds = [n for n in ast.walk(loop)
             if isinstance(n, ast.Assign)
             and any(isinstance(t, ast.Name) and t.id == "_now" for t in n.targets)]
    assert len(binds) == 1, (
        f"the render loop binds _now {len(binds)} times per tick; expected exactly 1")
    for period in ("_mesh_draw_period", "_ft_draw_period", "_trace_draw_period"):
        assert re.search(rf"_now\s*-\s*\w+\[.t.\]\)?\s*>=\s*{period}", _SRC), (
            f"{period} is no longer compared against the shared _now")


def test_the_ghost_is_only_posed_and_solved_while_it_is_visible():
    """The ghost costs a second ``update_cfg`` plus its IK thread (18.4% of a core during a
    sequential run), and unticking it in the page is the operator's zero-code way out of both.
    That only works while the repose AND every solve dispatch stay behind the visibility flag.
    (The solver thread itself is not gated — it blocks on the request event, and the gate is at
    the two places that set it: the staging path and the re-tick handler.)"""
    guard = _guard_containing(_render_loop(), "_ghost_on")
    assert "ghost_urdf.update_cfg" in _calls(guard), (
        "the ghost repose escaped the visibility guard")
    parent = {}
    for node in ast.walk(_TREE):
        for child in ast.iter_child_nodes(node):
            parent[id(child)] = node
    dispatches = [n for n in ast.walk(_TREE)
                  if isinstance(n, ast.Call) and _dotted(n.func) == "ghost_solve_req.set"]
    assert dispatches, "nothing dispatches a ghost solve any more"
    for call in dispatches:
        # The gate takes either shape in this file: an `if show_ghost.value:` around the
        # dispatch (the staging path) or an early `if not (... show_ghost.value): return` at the
        # top of the handler (the re-tick path). Accept both, reject neither being present.
        fn, node = None, call
        while id(node) in parent:
            node = parent[id(node)]
            if isinstance(node, ast.FunctionDef):
                fn = node
                break
        assert fn is not None, f"ghost dispatch at line {call.lineno} is not inside a function"
        inside = {id(n) for n in ast.walk(fn)}
        gated = False
        for cond in (n for n in ast.walk(fn)
                     if isinstance(n, ast.If) and "show_ghost" in ast.dump(n.test)):
            covered = id(call) in {id(n) for n in ast.walk(cond)}
            returns = any(isinstance(n, ast.Return) for n in ast.walk(cond))
            if covered or returns:
                gated = True
                break
        assert id(call) in inside and gated, (
            f"a ghost solve is dispatched at line {call.lineno} without consulting show_ghost — "
            f"unticking the box would stop hiding the ~100 ms/chunk IK")


# --- the idle clutch ----------------------------------------------------------------------

def test_the_idle_clutch_is_rate_limited_but_the_republish_is_not():
    hb = _fn_node("_heartbeat_loop")
    guard = _guard_containing(hb, "next_clutch")
    guarded = {id(n) for n in ast.walk(guard)}
    assert "<subscript>" in _calls(guard) or "sync" in _calls(guard), (
        "the clutch call moved out of the next_clutch guard")
    assert "_s0_send" not in _calls(guard), (
        "_s0_send is inside the clutch guard — the :8770 refresh would drop from drip_hz to "
        "IDLE_CLUTCH_HZ, cutting the margin against S0Driver.FRESHNESS_TIMEOUT_S (0.5 s) and "
        "holosoma's 0.25 s stale gate")
    outside = [n for n in ast.walk(hb)
               if isinstance(n, ast.Call) and _dotted(n.func) == "_s0_send"
               and id(n) not in guarded]
    assert outside, "the heartbeat no longer republishes the wire at all"


def test_the_clutch_deadline_is_not_consumed_when_busy_is_held():
    """``busy`` being held means Predict/Run owns the state, not that the clutch just ran. The
    deadline is therefore advanced INSIDE the guard (i.e. after the acquire), so a contended
    tick retries on the next one instead of waiting out another whole clutch period."""
    hb = _fn_node("_heartbeat_loop")
    guard = _guard_containing(hb, "next_clutch")
    src = ast.get_source_segment(_SRC, guard) or ""
    assert re.search(r"next_clutch\s*=\s*_t\s*\+\s*clutch_period", src), (
        "the clutch deadline is not advanced inside the guard")
    test_src = ast.get_source_segment(_SRC, guard.test) or ""
    assert "busy.acquire" in test_src and test_src.index("next_clutch") < test_src.index(
        "busy.acquire"), (
        "the deadline check must short-circuit BEFORE busy.acquire, so a not-yet-due tick does "
        "not take the mutex at all")


def test_the_clutch_rate_is_slower_than_the_wire_but_far_inside_the_arm_gate():
    m = re.search(r"^\s*IDLE_CLUTCH_HZ\s*=\s*([0-9.]+)", _SRC, re.M)
    assert m, "IDLE_CLUTCH_HZ is gone"
    hz = float(m.group(1))
    # Below the wire rate (that is the whole point) ...
    assert hz < 30.0, f"IDLE_CLUTCH_HZ={hz} is not actually a rate limit"
    # ... but fast enough that the pose the pre-arm gate reads is never stale enough to matter:
    # the arm is not being commanded while this runs, and at the measured median per-tick joint
    # motion of a holding G1 the TCP drift over 1/hz is single-digit mm against a 70 mm gate.
    assert hz >= 5.0, (
        f"IDLE_CLUTCH_HZ={hz} leaves the broadcast wire more than 200 ms behind the measured "
        f"pose; the pre-arm gate tolerance is 70 mm and arming must not jump")


# --- websocket compression ----------------------------------------------------------------

def test_the_bridge_sockets_do_not_negotiate_compression():
    """permessage-deflate on the camera stream cost 1.84 ms per frame in the BRIDGE process
    (5.5% of a core at 30 Hz) and 0.19 ms here, to shrink already-compressed base64 JPEG by
    24% over loopback. Compression is negotiated, so declining it client-side disables it at
    both ends — which is how this one keyword unloads the bridges too."""
    src = ast.get_source_segment(_SRC, _fn_node("run")) or ""
    assert "websockets.connect(" in src, "_WSSubscriber.run no longer opens the socket here"
    assert "compression=None" in src, (
        "_WSSubscriber.run re-enabled permessage-deflate on the sensor sockets")


def test_the_s0_wire_server_does_not_negotiate_compression():
    src = _DRIVER_PATH.read_text()
    m = re.search(r"websockets\.serve\((.*?)\)", src, re.S)
    assert m, "S0Driver no longer calls websockets.serve"
    assert "compression=None" in m.group(1), (
        "the :8770 wire server re-enabled deflate; it re-sends ~500 bytes 200x/s over loopback "
        "and broadcast() cannot share a compression context across clients")
