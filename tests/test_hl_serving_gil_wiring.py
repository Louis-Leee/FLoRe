"""The yield gate's WIRING into scripts/hl_serving.py.

Unit behaviour is in ``tests/test_gil_yield.py``; the Jacobian is in ``tests/test_ghost_arm_ik.py``.
What is pinned here is that the gate is actually attached at both ends, because either half alone
is useless: a priority section nobody yields to changes nothing, and a yielding viz thread with no
priority section changes nothing.

Source-level checks, for the same reason ``test_hl_serving_reach_wiring.py`` is: the seams live
inside ``main()``/``_run_viser()``'s closures and cannot be imported without standing up a policy,
five sockets and a viser page. Structure is what matters, and a detached hook is precisely what a
structural test catches.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

_SRC_PATH = Path(__file__).resolve().parents[1] / "scripts" / "hl_serving.py"
_SRC = _SRC_PATH.read_text()


def _fn_body(name: str) -> str:
    """Source of the named function (nested defs included), located via AST for exactness."""
    tree = ast.parse(_SRC)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return ast.get_source_segment(_SRC, node) or ""
    raise AssertionError(f"function {name} not found in hl_serving.py")


def test_every_inference_call_site_holds_the_gate():
    """BOTH cadences must run under priority. ``_infer`` is the chunk path (Predict / sequential /
    continuous) and ``infer_step_ensembled`` is the ACT per-step path, which re-queries every step
    and would otherwise be taxed 16/30/50 times per execution."""
    for fn, call in (("_infer", "planner.infer_chunk("),
                     ("_drip_ensembled", "planner.infer_step_ensembled(")):
        body = _fn_body(fn)
        assert call in body, f"{call} no longer lives in {fn} — re-check this wiring test"
        assert "with infer_gate.priority():" in body, (
            f"{fn} calls the model without holding infer_gate — a pure-Python viz thread can "
            f"then stretch the sample (measured 46 ms -> 3543 ms on a diffusion checkpoint)")


def test_no_inference_call_site_is_left_unguarded():
    """A future third cadence must not silently skip the gate: every planner.infer_* call in the
    file has to sit inside a `with infer_gate.priority()` block."""
    tree = ast.parse(_SRC)
    guarded: list[bool] = []

    class V(ast.NodeVisitor):
        def __init__(self) -> None:
            self.depth = 0

        def visit_With(self, node: ast.With) -> None:  # noqa: N802
            opens = any(
                isinstance(it.context_expr, ast.Call)
                and isinstance(it.context_expr.func, ast.Attribute)
                and it.context_expr.func.attr == "priority"
                for it in node.items)
            self.depth += 1 if opens else 0
            self.generic_visit(node)
            self.depth -= 1 if opens else 0

        def visit_Call(self, node: ast.Call) -> None:  # noqa: N802
            f = node.func
            if (isinstance(f, ast.Attribute) and f.attr.startswith("infer_")
                    and isinstance(f.value, ast.Name) and f.value.id == "planner"):
                guarded.append(self.depth > 0)
            self.generic_visit(node)

    V().visit(tree)
    assert guarded, "no planner.infer_* call sites found — the finder is broken, not the code"
    assert all(guarded), f"{guarded.count(False)} planner.infer_* call site(s) outside the gate"


def test_ghost_solver_yields_per_step():
    """The yield must be INSIDE the per-step loop. Checking it once per chunk would make a predict
    wait for a whole ~100 ms solve instead of one ~2 ms step."""
    body = _fn_body("_solve_ghost_cfgs")
    assert "infer_gate.wait()" in body, "the ghost solve no longer yields to inference"
    lines = body.splitlines()
    wait_i = next(i for i, ln in enumerate(lines) if "infer_gate.wait()" in ln)
    loop_i = next(i for i, ln in enumerate(lines) if re.match(r"\s*for i in range\(S\):", ln))
    assert wait_i > loop_i, "infer_gate.wait() must be inside the per-step loop, not before it"


def test_ghost_solve_is_not_dispatched_while_the_ghost_is_hidden():
    """The solve exists only to feed ``ghost_urdf.update_cfg`` behind ``show_ghost``; with the box
    unticked it produced a config nothing renders."""
    body = _fn_body("_solve_and_cache")
    assert 'if bool(show_ghost.value):' in body, \
        "_solve_and_cache dispatches the ghost solve regardless of show_ghost"
    assert "ghost_solve_req.set()" in body


def test_hidden_ghost_still_stages_the_request_so_a_retick_is_immediate():
    """Skipping the dispatch must not also skip STORING the chunk, or re-ticking the box would show
    nothing until the next Predict — which reads as a broken checkbox."""
    body = _fn_body("_solve_and_cache")
    m = re.search(r"with ghost_lock:\n(.*?)\n(\s*)if bool\(show_ghost\.value\):", body, re.S)
    assert m, "expected the pending-request update to sit OUTSIDE the show_ghost branch"
    assert "ghost_pending.update(" in m.group(1)
    # ...and the re-tick has to actually kick the solver.
    assert "@show_ghost.on_update" in _SRC, "no re-tick handler: a hidden->shown flip would stall"


def test_analytic_jacobian_replaced_the_finite_difference_one():
    """The finite-difference inner loop was 88% of the ghost IK's cost (7 extra update_cfg calls
    per iteration). If it comes back, the 646 ms/chunk comes back with it."""
    body = _fn_body("_ghost_dls")
    assert "_ghost_arm_jacobian(" in body, "_ghost_dls no longer uses the analytic Jacobian"
    assert "d = 1e-5" not in body, "the finite-difference Jacobian is back in _ghost_dls"
    j = _fn_body("_ghost_arm_jacobian")
    assert "np.cross(" in j, "the geometric Jacobian's position block is missing"


def test_arm_joints_are_validated_as_revolute():
    """The analytic columns are revolute-only; a prismatic joint would need a different column, so
    the builder must refuse rather than mis-solve."""
    assert 'revolute' in _SRC and "_nonrev" in _SRC, \
        "hl_serving no longer checks that the ghost arm joints are revolute"


def test_gate_is_constructed_once_and_passed_to_the_page():
    """One gate for the process: two gates would each be half-connected."""
    assert len(re.findall(r"infer_gate = YieldGate\(", _SRC)) == 1
    assert re.search(r"def _run_viser\((?:.|\n)*?infer_gate=None", _SRC), \
        "_run_viser does not accept infer_gate"
