"""The reach compensator's WIRING into scripts/hl_serving.py.

The unit behaviour lives in ``tests/test_hl_reach.py``. What is pinned here is the property that
makes the feature shippable on a live rig: with ``--reach-ki 0`` (the default) the publish path
produces the SAME wire bytes it produced before the feature existed, and the compensator is
consulted only at the publish seam — never on the observation, the decode anchor or the chunk.

These are source-level checks because the two publish seams live inside ``main()``'s closure and
cannot be imported in isolation without standing up the whole rig (a policy, five sockets and a
viser page). Structure is exactly what matters here, and a mis-placed hook is precisely what a
structural test can catch.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import numpy as np
import pytest

from fuji.s1_policy.serving.hl_reach import ReachCompensator, ReachConfig

_SRC_PATH = Path(__file__).resolve().parents[1] / "scripts" / "hl_serving.py"
_SRC = _SRC_PATH.read_text()


def test_default_gain_is_off():
    """The launcher default must leave the rig untouched."""
    m = re.search(r'add_argument\("--reach-ki",\s*type=float,\s*default=([0-9.]+)', _SRC)
    assert m, "--reach-ki flag not found"
    assert float(m.group(1)) == 0.0


def test_disabled_compensator_returns_input_unchanged_for_wire_positions():
    """The OFF path's numerical contract: whatever the chunk decoded to is what gets levered onto
    the wire. Exercised over a real chunk's worth of positions."""
    comp = ReachCompensator(ReachConfig(ki=0.0))
    rng = np.random.default_rng(0)
    for pos in rng.uniform(-0.6, 0.6, size=(50, 3)):
        out = comp.compensate(pos, rng.uniform(-0.6, 0.6, size=3))
        assert out.tolist() == pos.tolist()


def test_publish_seams_are_the_only_call_sites():
    """``_reach_apply`` may be reached ONLY from the two functions that turn a target into a wire
    (``_wire_from_step`` for the per-step/ensembled cadence, ``_wire_for_step`` for the chunk
    cadence). If it ever appears inside a predict/observation/anchor path, the compensator would
    be feeding the model a pose the robot is not at — the one thing this design must not do."""
    tree = ast.parse(_SRC)
    callers: set[str] = set()

    class V(ast.NodeVisitor):
        def __init__(self) -> None:
            self.stack: list[str] = []

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:  # noqa: N802
            self.stack.append(node.name)
            self.generic_visit(node)
            self.stack.pop()

        def visit_Call(self, node: ast.Call) -> None:  # noqa: N802
            f = node.func
            if isinstance(f, ast.Name) and f.id == "_reach_apply" and self.stack:
                callers.add(self.stack[-1])
            self.generic_visit(node)

    V().visit(tree)
    assert callers == {"_wire_from_step", "_wire_for_step"}, (
        f"_reach_apply is called from {sorted(callers)}; it must only be reached from the two "
        "publish seams (never from a predict / observation / anchor path)")


def test_both_executor_cadences_publish_through_a_compensated_seam():
    """_drip (chunk cadence) must publish via ``_wire_for_step`` rather than indexing the
    pre-computed rows: with compensation ON the correction changes every step, so reusing the row
    computed at decode time would apply a correction up to 1.67 s stale."""
    assert "pos_i, orn_i = _wire_for_step(out, i)" in _SRC, \
        "_drip no longer publishes through the compensated seam"
    # ...and the ensembled cadence goes through _wire_from_step, which applies it internally.
    assert "pos_i, orn_i, _urdf = _wire_from_step(blend[\"position\"], blend[\"rot6d\"])" in _SRC


def test_recorded_plan_and_preview_stay_uncompensated():
    """The take's stored plan + the yellow preview must show what the MODEL asked for, so the
    payload's own wire rows are never overwritten in place."""
    m = re.search(r"def _wire_for_step\(.*?\n(.*?)\n    def ", _SRC, re.S)
    assert m, "_wire_for_step not found"
    body = m.group(1)
    assert 'out["wire_pos"][i] =' not in body and 'out["wire_orn"][i] =' not in body, \
        "_wire_for_step must not mutate the payload (that is the recorded plan + the preview)"


def test_residual_is_reset_on_every_abandoned_intent():
    """A residual that survives a Stop / disarm / reset would be applied to the next, unrelated
    motion. Each of these four sites is a documented 'intent abandoned' point in this file."""
    for reason in ("reset start pose", "low level lost", "sequential stop",
                   "new sequential rollout"):
        assert f'_reach_reset("{reason}")' in _SRC, f"missing residual reset for: {reason}"


def test_toggle_off_also_drops_the_residual():
    """Un-ticking must clear, not merely pause: a residual re-applied on a later re-tick would be
    a stale correction for a shortfall that is minutes old."""
    m = re.search(r"@reach_toggle\.on_update\n    def _\(_e\) -> None:\n(.*?)\n\n", _SRC, re.S)
    assert m, "reach toggle callback not found"
    assert '_reach_reset("compensation toggled")' in m.group(1)


def test_argparse_help_strings_escape_percent():
    """argparse %-interpolates help text, so a bare '%' in a help string raises at --help time
    ("%o format: an integer is required, not dict"). These help strings quote measured
    percentages, and an unescaped one broke --help outright — i.e. it broke launch, not just the
    docs. Pinned because the failure is invisible until someone asks for help."""
    reach = re.findall(r'add_argument\("--reach-[a-z-]+".*?help=(".*?")\)\n', _SRC, re.S)
    assert reach, "no --reach-* help strings found"
    for block in reach:
        # Concatenated string literals → the rendered help text.
        text = "".join(re.findall(r'"([^"]*)"', block))
        for m in re.finditer(r"%+", text):
            assert len(m.group(0)) % 2 == 0, (
                "unescaped '%' in a --reach-* help string (argparse raises at --help): "
                f"...{text[max(0, m.start() - 40):m.end() + 20]}...")


def test_tuner_defaults_match_the_launcher():
    """``scripts/tune_reach_comp.py`` exists to predict what ``hl_serving.py`` will do, so its
    clamp/gate/decay defaults must be the SAME numbers. A drift here would have the operator
    tuning against a configuration nobody runs."""
    tuner = (_SRC_PATH.parent / "tune_reach_comp.py").read_text()
    for tuner_flag, serving_flag in (("--clamp", "--reach-clamp"),
                                     ("--stall-frac", "--reach-stall-frac"),
                                     ("--gate", "--reach-gate"),
                                     ("--decay", "--reach-decay"),
                                     ("--contact-n", "--reach-contact-n")):
        t = re.search(rf'add_argument\("{tuner_flag}",\s*type=float,\s*default=([0-9.]+)', tuner)
        s = re.search(rf'add_argument\("{serving_flag}",\s*type=float,\s*default=([0-9.]+)', _SRC)
        assert t and s, f"could not read defaults for {tuner_flag}/{serving_flag}"
        assert float(t.group(1)) == float(s.group(1)), (
            f"{tuner_flag}={t.group(1)} but {serving_flag}={s.group(1)} — the offline audit would "
            "not describe the rig")


def test_clamp_ceiling_is_enforced_at_config_level():
    """The launcher exposes --reach-clamp, so the ceiling has to be enforced where the value is
    consumed rather than in the argparse help text."""
    from fuji.s1_policy.serving.hl_reach import MAX_CLAMP_M

    with pytest.raises(ValueError):
        ReachConfig(ki=0.1, clamp_m=MAX_CLAMP_M + 1e-6)
