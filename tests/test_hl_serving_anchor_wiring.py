"""The chunk anchor's WIRING into ``scripts/hl_serving.py`` and ``scripts/run_hl_serving.sh``.

Unit behaviour lives in ``tests/test_hl_anchor.py``. What is pinned here is what makes the
feature shippable: the default leaves the rig untouched, the selector is consulted at the ONE
re-anchor every algorithm goes through (so ``--algo act/diffusion/flow/muse`` are covered by
construction rather than by hope), every drip path records the commanded step, and the launcher
cannot pass a mode the parser will reject.

Source-level, for the same reason ``test_hl_serving_reach_wiring.py`` is: both seams live inside
``main()``/``_run_viser``'s closures and cannot be imported without standing up a policy, five
sockets and a viser page. Structure is exactly what matters, and an orphaned hook — a helper
defined in ``main()`` but referenced from ``_run_viser``, which is a NameError only on the rig —
is precisely what a structural test catches.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_SRC_PATH = _ROOT / "scripts" / "hl_serving.py"
_SRC = _SRC_PATH.read_text()
_SH_PATH = _ROOT / "scripts" / "run_hl_serving.sh"
_SH = _SH_PATH.read_text()


def _func(name: str) -> ast.AST:
    for node in ast.walk(ast.parse(_SRC)):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return node
    raise AssertionError(f"{name} not found in {_SRC_PATH}")


def _calls(node: ast.AST) -> list[str]:
    out = []
    for n in ast.walk(node):
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name):
            out.append(n.func.id)
    return out


# ------------------------------------------------------------------------------- default is off
def test_default_mode_is_measured():
    """The un-flagged rig path must be the previous behaviour."""
    m = re.search(r'add_argument\("--anchor-mode",.*?default="([a-z]+)"', _SRC, re.S)
    assert m, "--anchor-mode flag not found"
    assert m.group(1) == "measured"


def test_default_leak_is_inside_the_module_ceiling():
    from flore.s1_policy.serving import hl_anchor

    m = re.search(r'add_argument\("--anchor-eps",\s*type=float,\s*default=([0-9.]+)', _SRC)
    assert m, "--anchor-eps flag not found"
    assert 0.0 < float(m.group(1)) <= hl_anchor.MAX_EPS_M


def test_default_contact_cutout_matches_the_deploy_force_gate():
    """1.5 N is the same threshold the low level's FORCE_CMD_WIRE_GATE_N and the planner's
    --force-gate-n use. Three guards disagreeing about what 'in contact' means is a bug."""
    m = re.search(r'add_argument\("--anchor-contact-n",\s*type=float,\s*default=([0-9.]+)', _SRC)
    assert m and float(m.group(1)) == 1.5


# ---------------------------------------------------------------- consulted at the ONE re-anchor
def test_the_re_anchor_resolves_through_the_selector():
    """``_resync_anchor_from_robot`` is the single place the LEFT decode anchor is taken from the
    measurement, and it is shared by every algorithm and by recorded replay. If the selector is
    not consulted here, the feature silently does nothing."""
    fn = _func("_resync_anchor_from_robot")
    assert "_anchor_resolve" in _calls(fn), (
        "_resync_anchor_from_robot must resolve the anchor through _anchor_resolve, otherwise "
        "--anchor-mode command is inert")
    # …and it must still be the value handed to commit_role, not a discarded local.
    src = ast.get_source_segment(_SRC, fn) or ""
    assert re.search(r'commit_role\("left",\s*_a_pos,\s*_a_rot6d\)', src), (
        "the resolved anchor must be what commit_role receives")


def test_disarming_drops_the_carried_anchor():
    src = ast.get_source_segment(_SRC, _func("_resync_anchor_from_robot")) or ""
    assert "_anchor_reset" in src, (
        "a disarm must drop the carried command anchor — after the operator takes the arm back, "
        "a command issued before the intervention must not remain the decode base")


def test_every_drip_path_records_the_commanded_step():
    """Three executors publish commands: the chunk cadence, ACT temporal ensembling, and the
    open-loop replay. A path that publishes without recording would hand the next chunk a stale
    base."""
    for name in ("_drip", "_drip_ensembled", "_drip_openloop"):
        assert "_anchor_note" in _calls(_func(name)), f"{name} must call _anchor_note"


def test_the_commanded_step_recorded_is_the_last_EXECUTED_one():
    """Not the chunk end: an interrupted chunk's un-executed tail was never asked for."""
    src = ast.get_source_segment(_SRC, _func("_drip")) or ""
    assert re.search(r"_anchor_note\(step_pos\[last_i\],\s*step_rot6d\[last_i\]\)", src), (
        "_drip must record step last_i (the last EXECUTED step), matching _commit_left_step")


def test_selector_is_threaded_into_the_viser_closure():
    """``_resync_anchor_from_robot`` lives in ``_run_viser`` while the helpers are built in
    ``main()``, so they MUST be parameters — a closure reference across that boundary is a
    NameError that only appears on the rig."""
    params = {a.arg for a in _func("_run_viser").args.args}
    params |= {a.arg for a in _func("_run_viser").args.kwonlyargs}
    for needed in ("chunk_anchor", "_anchor_resolve", "_anchor_reset"):
        assert needed in params, f"_run_viser must take {needed} as a parameter"


def test_anchor_reset_is_reachable_from_every_intent_drop():
    """The carried anchor is abandoned intent at exactly the moments the reach residual is, so it
    is dropped from inside ``_reach_reset`` rather than at five duplicated call sites."""
    assert "_anchor_reset" in _calls(_func("_reach_reset"))


def test_the_selector_never_touches_the_published_wire():
    """Separation of concerns with --reach-*: this changes the DECODE BASE only. If it also
    appeared at the publish seam the two features would compound invisibly."""
    wire_fn = _func("_wire_from_step")
    assert "_anchor_resolve" not in _calls(wire_fn)
    assert "_anchor_note" not in _calls(wire_fn)


# ------------------------------------------------------------------------------------- launcher
def test_launcher_default_is_measured_and_validates_the_value():
    assert 'ANCHOR="${ANCHOR:-measured}"' in _SH
    assert "measured|command)" in _SH, "the launcher must reject a typo'd ANCHOR, not pass it on"


def test_launcher_passes_the_mode_through():
    assert "--anchor-mode" in _SH
    assert '"${ANCHOR_ARGS[@]}"' in _SH, "ANCHOR_ARGS must actually reach the python invocation"


def test_launcher_modes_match_the_parser_choices():
    from flore.s1_policy.serving import hl_anchor

    for mode in hl_anchor.ANCHOR_MODES:
        assert mode in _SH, f"launcher does not mention mode {mode!r}"
