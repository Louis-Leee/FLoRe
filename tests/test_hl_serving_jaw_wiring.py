"""The LIVE-MESH jaw's WIRING into ``scripts/hl_serving.py``.

Unit behaviour is elsewhere: ``tests/test_live_robot.py`` round-trips
``jaw_joint_from_width_m`` against the real URDF aperture (so the width→joint polarity is
already pinned), and ``tests/test_hl_obs_freshness.py`` pins the OBSERVATION side's
``GRIP_MAX_AGE_S`` latch. What is pinned HERE is that the mesh render path gates the jaw on
that same age — the one thing neither of those can see.

Source-level checks, for the same reason ``test_hl_serving_gil_wiring.py`` and
``test_hl_serving_reach_wiring.py`` are: the seam lives inside ``_run_viser``'s render-loop
closure and cannot be imported without standing up a policy, five sockets and a viser page.

WHAT THIS CATCHES (a real defect, 2026-08-18). The jaw was the ONLY stream in that loop with
no freshness gate — the RecordTap, the F/T wire and the observation all have one. ``latest()``
latches the newest frame forever, so a gripper bridge that died mid-session left its last
width on screen indefinitely, labelled "· live": a bridge that dropped while holding a grasp
rendered a permanently SHUT hand on a physically open gripper, and there was no way to tell
that from a real measurement.
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


def _render_loop() -> str:
    body = _fn_body("_run_viser")
    assert "urdf.update_cfg(remap_dof_to_urdf(" in body, (
        "the live-mesh update no longer lives in _run_viser — re-check this wiring test")
    return body


def test_the_live_mesh_jaw_is_age_gated():
    """The render path must consume the jaw only while it is FRESH, and must reuse
    ``GRIP_MAX_AGE_S`` rather than a fresh literal — the mesh and the model's own gripper latch
    have to agree on what "fresh" means, or the page can show a jaw the policy already refused."""
    body = _render_loop()
    assert 'subs["gripper"].latest()' in body, "the render loop no longer reads the jaw stream"
    assert "GRIP_MAX_AGE_S" in body, (
        "the live-mesh jaw is not gated on GRIP_MAX_AGE_S — a dead :8784 bridge will keep its "
        "last width latched on screen forever, labelled as a live measurement"
    )
    # The gate must be what decides whether a width is computed at all, not merely mentioned.
    assert re.search(r"_g_fresh\s*=.*GRIP_MAX_AGE_S", body, re.S), (
        "GRIP_MAX_AGE_S appears but does not feed the freshness predicate")
    assert re.search(r"if\s+_g_fresh\s*:", body), (
        "the width/joint conversion is not guarded by the freshness predicate")


def test_a_stale_jaw_renders_open_rather_than_a_fake_grasp():
    """The fallback polarity is load-bearing and is NOT symmetric.

    Drawing an invented GRASP is the worse error — an operator reads a shut hand as "it is
    holding the object". So an absent/stale reading must fall back to ``JAW_JOINT_OPEN_M``,
    which the URDF's closing finger joint renders wide open."""
    body = _render_loop()
    assert re.search(r"jaw_m\s*=\s*JAW_JOINT_OPEN_M", body), (
        "the stale/absent fallback is no longer JAW_JOINT_OPEN_M — check it does not draw a grasp")


def test_a_stale_jaw_is_not_presented_as_a_measurement():
    """The other half of the fix, and the reason the gate alone is not enough: an open-looking
    mesh and an UNKNOWN jaw are indistinguishable on screen, so the readout has to say which it
    is. Before this, ``rd_grip`` was only ever written on the success path, leaving the last
    measured number beside a jaw that was no longer being measured."""
    body = _render_loop()
    writes = re.findall(r"rd_grip\.value\s*=\s*\(?\s*(?:f?\"|f?')?([^\n]{0,40})", body)
    assert writes, "rd_grip is never written in the render loop"
    assert any("NO READING" in w for w in writes), (
        "no rd_grip write announces a missing jaw reading — a stale width stays on screen next "
        "to a mesh that is drawn open by default, which reads as 'measured open'")
    assert "· live" in body, "the fresh path no longer labels the reading as live"
