"""Live VIVE-tracker teleop for the browser 2pt WBT viz.

This subpackage is the *parallel* live-teleop path that sits alongside the
existing motion-replay path (``bake_motions`` + ``main``). It NEVER touches the
replay code: a separate console entry (``wbt-vive-teleop``) builds a bundle for
a NEW preset (``g1-29dof-holosoma-2pt-wbt-live``) whose ``vr_2point_pos`` /
``vr_2point_orn`` obs terms are fed LIVE from two VIVE trackers over a
WebSocket, while every other obs term still comes from the injected
``idle_default`` standing clip + live sim state.

Pieces:
  * ``home_anchor.json`` — the default-pose 2pt VR target (pelvis frame),
    computed once via the bake FK on the trained ONNX. It is BOTH the live
    calibration target AND the pre-first-message fallback the TS obs classes
    return (returning zeros instead would collapse the arms onto the pelvis).
  * ``vive_teleop_bridge.py`` — holds the pysurvive singleton, calibrates the
    operator's hands onto the home anchor, and serves pelvis-frame
    ``{pos[6], orn[12]}`` (orn = rot6d, byte-for-byte the bake convention) over
    a WebSocket the browser connects to.
"""

from __future__ import annotations

import json
from pathlib import Path

_HOME_ANCHOR_PATH = Path(__file__).resolve().parent / "home_anchor.json"


def load_home_anchor() -> dict:
    """Load the default-pose 2pt VR anchor constants (see module docstring)."""
    return json.loads(_HOME_ANCHOR_PATH.read_text())


__all__ = ["load_home_anchor"]
