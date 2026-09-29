"""Build the browser teleop bundle for the MOTION-COMPLETION presets (5pt / dense / wbt).

The rig has THREE VIVE trackers. `exp:g1-wbt-5pt-hybrid-unified-anneal` wants FIVE tracked
points (both hands, the head, and BOTH FEET); `exp:g1-wbt-hybrid-unified-anneal` and the
BASE tracker `exp:g1-wbt` both want all 29 joint targets. So the same 3-tracker stream goes
through Motion_Completion first, and the extra points come out of its 29-DoF FK — see
`vive/mc_teleop_source.py` for the frame chain and the latency floor, and
`vive/mc_teleop_selftest.py` for the acceptance run.

    --mode 5pt    g1-29dof-holosoma-5pt-hybrid-unified-anneal-wbt-live     184D x 4 = 736D
    --mode dense  g1-29dof-holosoma-dense-hybrid-unified-anneal-wbt-live   183D x 4 = 732D
    --mode wbt    g1-29dof-holosoma-wbt-live                               157D x 4 = 628D

`wbt` is the BASE whole-body tracker (run ruo0o6fr) — the only one of the three with NO
force block, and the one whose recorded-clip replay is already verified. It shares the
`dense` bridge wire: both steer the 58D `command` term, so only the PRESET differs.

Two shells:

    MODE=5     bash force_deploy/1_vive_bridge.sh   # or MODE=dense
    bash scripts/_vive_teleop5_build.sh --mode 5pt                   # or --mode dense|wbt

WHAT THIS ENTRY ADDS over `main.setup_builder`, and why each is not optional:

* THE TRACKING-FAMILY GUARD, reused from `main` (`_assert_checkpoint_tracking_family`
  runs inside `setup_builder`). 732D is ALSO the global-vel family's width.
* AN ACTOR-OBS LAYOUT GUARD, reused from `main_vive3`: the checkpoint's own slot names
  must be this preset's, with `_live` stripped. For 5pt that catches a 736D checkpoint
  carrying the 0.18 m-lever term; for dense it catches the same-width global-vel layout
  from the other direction.
* THE HOME FALLBACKS, from `vive/home_anchor.json`. In these modes the bridge does NOT
  anchor on that file (its home is Motion_Completion's 29-joints-ZERO pose), so the
  fallback is deliberately the IDLE_DEFAULT reference pose — the one the browser's
  reference ghost holds — rather than the default_joint_angles pose the 2pt/3pt arrays
  use. See the `_comment_mc_teleop_5pt_dense` note in that file.
* `policy_joint_names` on the dense term. The wire carries the COMPLETION's joint order
  and the policy has its own; both are 29 long and they are NOT the same, so the browser
  remaps BY NAME and needs to be told the target order.

⚠️ THE LAG IS NOT A BUG. ~667 ms, and it belongs to the MC checkpoint: its conditioning is
dense over a 30-frame window with a 20-frame stride, so a window cannot run until its last
cond frame exists, and gap-free playback needs at least stride-1 = 19 frames of lag.
Measured compute is ~46 ms/window (~7% duty) — the host is not the problem. Foot IK stays
OFF: measured 1937 ms/window p50 = 291% duty, which takes the command lag from a constant
700 ms to 6100 ms p50 / 8867 ms max, and no cheaper setting helps.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from loguru import logger

import mjswan  # noqa: F401  (ensures mjswan import side effects match main.py)

from wbt_realtime_eval.main import _read_onnx_metadata, setup_builder
from wbt_realtime_eval.main_vive3 import _validate_actor_obs_matches_preset_named
from wbt_realtime_eval.vive import load_home_anchor

#: mode -> (preset name, {live term key: home_anchor.json field}).
#:
#: The home field for each live term is the pre-first-frame fallback. NEVER zeros: a zero
#: 5pt position commands all five points to the pelvis origin (arms collapsed, feet under
#: the hips), and a zero dense command asks for the 29-joints-zero pose at full authority.
_MODES = {
    "5pt": (
        "g1-29dof-holosoma-5pt-hybrid-unified-anneal-wbt-live",
        {
            "vr_5point_pos_tcp_live": "home_pos_pelvis_5pt_tcp",
            "vr_5point_orn_live": "home_orn_quat_xyzw_5pt",
        },
    ),
    "dense": (
        "g1-29dof-holosoma-dense-hybrid-unified-anneal-wbt-live",
        {"command_live": "home_command_dense"},
    ),
    # BASE whole-body tracker (exp:g1-wbt, run ruo0o6fr). Same 58D joint-space wire as
    # `dense` — the bridge mode is `dense` for this too — but the policy is the 628D base
    # tracker with NO force block. Kept a separate --mode rather than an alias because the
    # PRESET differs, and the actor-obs guard below is keyed on the preset: building the
    # 732D dense-unified checkpoint against this 628D preset (or the reverse) is refused
    # by name, which is the only thing that separates them at build time.
    "wbt": (
        "g1-29dof-holosoma-wbt-live",
        {"command_live": "home_command_dense"},
    ),
}

#: Which bridge `--mode` feeds each build mode. `wbt` and `dense` share the joint-space
#: wire, so the bridge has no `wbt` mode of its own — saying so here keeps the closing
#: hint honest instead of telling the operator to run a mode that does not exist.
_BRIDGE_MODE = {"5pt": "5", "dense": "dense", "wbt": "dense"}


def _flatten(rows) -> list[float]:
    """home_anchor.json stores per-point rows for keypoint terms and a flat list for the
    dense one. Accept both rather than making the caller know which."""
    if rows and isinstance(rows[0], (list, tuple)):
        return [float(v) for row in rows for v in row]
    return [float(v) for v in rows]


def _per_key_params(mode: str, ws_url: str, meta: dict[str, str]) -> dict[str, dict]:
    from wbt_realtime_eval.presets import term_base_dim

    anchor = load_home_anchor()
    preset_name, fields = _MODES[mode]
    # Assert the fallback's LENGTH against the preset's own width table. A short HOME would
    # be zero-padded by the TS side (wbtVrLiveHome returns zeros unless the length matches
    # exactly), which is the one failure mode that looks like "the policy just does that".
    out: dict[str, dict] = {}
    for key, field in fields.items():
        rows = anchor.get(field)
        if not rows:
            raise ValueError(
                f"vive/home_anchor.json is missing {field!r}, the HOME fallback for the live "
                f"term {key!r}. It is DERIVED from bake_motions' idle_default.npz — see the "
                f"_comment_mc_teleop_5pt_dense note in that file and "
                f"tests/test_mc_teleop.py::test_mc_home_fallbacks_are_the_baked_idle_default_pose."
            )
        home = _flatten(rows)
        want = term_base_dim(key)
        if len(home) != want:
            raise ValueError(
                f"{field!r} has {len(home)} values, but the live term {key!r} is {want}-D. "
                f"The browser would silently fall back to ZEROS (wbtVrLiveHome only accepts "
                f"an exact-length array), which commands the pelvis origin."
            )
        out[key] = {"vive_ws_url": ws_url, "home": home}
    if "command_live" in out:
        # The browser must remap the wire's 29 joints onto the POLICY's order by name.
        # Sourced from the checkpoint, not re-typed: it is the order the ONNX was exported
        # with, and it is what `main` already ships to WbtJointPos.
        #
        # Keyed on the TERM, not on `mode == "dense"`: `wbt` carries the same live term and
        # needs the same remap, and an enumeration here would have silently shipped it
        # WITHOUT one — an index-aligned copy across two real, different 29-joint orders,
        # which scrambles the body while every width still checks out.
        names = _policy_joint_names(meta)
        out["command_live"]["policy_joint_names"] = names
        logger.info(f"joint-space live term: policy_joint_names[{len(names)}] from the checkpoint")
    return out


def _policy_joint_names(meta: dict[str, str]) -> list[str]:
    """The policy's 29 joint names, off the ONNX metadata."""
    import json

    for key in ("joint_names", "policy_joint_names", "dof_names"):
        raw = meta.get(key)
        if not raw:
            continue
        try:
            names = json.loads(raw)
        except (TypeError, ValueError):
            names = [s for s in str(raw).replace(",", " ").split() if s]
        if isinstance(names, list) and len(names) == 29:
            return [str(n) for n in names]
    raise ValueError(
        "the checkpoint carries no 29-entry joint-name list (tried joint_names / "
        "policy_joint_names / dof_names). The dense live term CANNOT be wired without it: "
        "the completion's joint order and the policy's are both real orders, both 29 long, "
        "and not the same -- an index-aligned copy would scramble the body silently."
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="wbt-vive-teleop-5pt",
        description=(
            "Live 3-tracker teleop of the 5pt / DENSE unified anneal policies in browser "
            "MuJoCo, via Motion_Completion. Run the tracker bridge separately in the "
            "matching mode (MODE=5 / MODE=dense) -- it holds the pysurvive singleton and "
            "spawns the completion backend."
        ),
    )
    parser.add_argument("--mode", choices=sorted(_MODES), default="5pt")
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument(
        "--motion-dir",
        required=True,
        type=Path,
        help="Reference motions (still required -- supplies the non-tracking obs terms via "
        "the injected idle_default clip, which is also the HOME fallback pose).",
    )
    parser.add_argument("--output-dir", default=None, type=Path)
    parser.add_argument(
        "--vive-ws-url",
        default="ws://localhost:8770",
        help="WebSocket URL of the VIVE bridge. The bridge MUST be in the SAME --mode: its "
        "payload is tagged and the browser ignores a tag it did not ask for, so a mismatch "
        "shows up as triads parked at HOME rather than as wrong motion.",
    )
    parser.add_argument("--scene-xml", default=None, type=Path)
    parser.add_argument("--no-launch", action="store_true")
    args = parser.parse_args()

    preset_name, _ = _MODES[args.mode]
    out_dir = args.output_dir or Path(f"dist/wbt_realtime_eval_vive_{args.mode}")
    model_path = args.model_path.resolve()
    logger.info(
        f"Live motion-completion teleop build: mode={args.mode} preset={preset_name} "
        f"ws={args.vive_ws_url}"
    )
    logger.info(
        "⚠️  command lag ~667 ms by construction (MC window 30 / stride 20, dense cond). "
        "Compute is ~7% duty; foot IK is OFF because it measures 291% (lag 700 ms -> 8867 ms)."
    )
    meta = _read_onnx_metadata(model_path)
    # The checkpoint's own actor obs must BE this preset's layout. For dense this is the
    # only thing separating it from the same-width global-vel family; for 5pt it catches a
    # lever-vs-TCP position term at identical width.
    _validate_actor_obs_matches_preset_named(meta, preset_name)

    builder = setup_builder(
        preset=preset_name,
        model_path=model_path,
        motion_dir=args.motion_dir.resolve(),
        output_dir=out_dir.resolve(),
        scene_xml=args.scene_xml,
        per_key_params=_per_key_params(args.mode, args.vive_ws_url, meta),
    )
    app = builder.build(output_dir=str(out_dir.resolve()))
    logger.info(
        f"Built. Start the tracker bridge in another shell: "
        f"`MODE={_BRIDGE_MODE[args.mode]} bash "
        f"force_deploy/1_vive_bridge.sh`, hold the calibration pose ~3 s, "
        f"then click Recalibrate. Expect ~1 s before the first command (the completion needs "
        f"a full 30-frame window) and ~667 ms of lag thereafter."
    )
    if args.no_launch or os.getenv("MJSWAN_NO_LAUNCH") == "1":
        return
    app.launch()


if __name__ == "__main__":
    main()
