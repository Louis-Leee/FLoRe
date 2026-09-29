#!/usr/bin/env python3
"""Interactive MuJoCo sim2sim for the WBT policy — the live-VIVE validation step.

``run_policy --task.headless-sim-eval`` runs a FIXED-LENGTH eval loop at max
speed and exits, which is fine for an NPZ-replay code-path / metrics check but
cannot be *driven* by a human in realtime. This harness instead opens an
interactive ``mujoco.viewer`` window and ticks the SAME ``HolosomaWBTPolicy``
(``policy.policy_action()``) at the configured rl-rate, so you can:

  * watch the policy track the NPZ in a live window, and
  * (with ``HOLOSOMA_VIVE_WS_URL`` set + the vive_teleop_bridge running) drive
    the robot's hands LIVE with two VIVE trackers — exactly the obs path the
    real-robot ``run_policy`` deploy will use, but with NO motors.

This validates the **policy + obs math + the live-VIVE override** in MuJoCo — it
is the sim gate for those. It is NOT a full motor-path gate: it drives the bare
``MuJoCoInterface`` PD loop (the same one ``run_policy`` offboard eval uses) and
therefore does NOT exercise the on-robot ``DampenedUnitreeInterface`` command
shaping (slew/clip/kp-kd scale) that the driver child applies. It is sim-only
(in-process ``MuJoCoInterface``); it never opens a Unitree SDK participant and
never writes ``rt/lowcmd``. The init ramp here is the base ``get_ready_state``
ramp (motor KP/KD), not the on-robot stiff-KP ``_handle_init_state`` path.

Usage (env hssim):

    # NPZ replay in an interactive window (no VIVE):
    MUJOCO_GL=egl python -m wbt_wrappers_inference.run_sim_interactive \
        inference:g1-29dof-holosoma-2pt-force-wbt \
        --task.model-path logs/WBT-Holosoma/09wvlz16/model_29999.onnx \
        --task.ref-motion-path ~/data/.../A508.npz

    # LIVE VIVE driving the sim (start the bridge first, separate shell):
    HOLOSOMA_VIVE_WS_URL=ws://localhost:8770 \
    MUJOCO_GL=egl python -m wbt_wrappers_inference.run_sim_interactive inference:... \
        --task.model-path ... --task.ref-motion-path ...

This harness auto-selects the policy class from the preset: a MaskedMimic preset
(``inference:g1-29dof-holosoma-maskedmimic-2pt-wbt``) drives
``HolosomaMaskedMimicPolicy``; everything else stays on ``HolosomaWBTPolicy`` (so
the existing exp:g1-wbt-2pt path is unchanged). Live-VIVE 2pt teleop works for
BOTH — the MaskedMimic injection is a world-frame clutch delta on the wrist
target_poses (it has no vr_2point term), but the keyboard flow ([] START+arm, [x]
recenter, [o] STOP) is identical:

    # LIVE VIVE driving the MaskedMimic 2pt policy in sim:
    HOLOSOMA_VIVE_WS_URL=ws://localhost:8770 \
    MUJOCO_GL=egl python -m wbt_wrappers_inference.run_sim_interactive \
        inference:g1-29dof-holosoma-maskedmimic-2pt-wbt \
        --task.model-path logs/WBT-Holosoma/qbdjrrxt/model_13000.onnx \
        --task.ref-motion-path ~/data/sonic_manipulation_test/<clip>.npz

Note: an interactive window needs a real GL context — run on the laptop's
display (``MUJOCO_GL=egl`` works headless for offscreen; for the on-screen
viewer use the default GL or ``glfw``). ``--task.headless-max-frames`` is reused
as a step cap (<=0 → run until the window is closed).
"""

from __future__ import annotations

import sys
import time
from collections import deque

import numpy as np
import tyro
from loguru import logger
from termcolor import colored
from typing_extensions import Annotated

from holosoma_inference.config.config_types.inference import InferenceConfig
from holosoma_inference.config.utils import TYRO_CONFIG

from wbt_wrappers_inference.config_values.inference import DEFAULTS as EXT_DEFAULTS
from wbt_wrappers_inference.policies.holosoma_maskedmimic import HolosomaMaskedMimicPolicy
from wbt_wrappers_inference.policies.holosoma_wbt import HolosomaWBTPolicy
from wbt_wrappers_inference.utils.interface import _create_headless_policy


def _select_interactive_policy_class(config: InferenceConfig):
    """Pick the policy class for the interactive sim from the preset's obs/task.

    MaskedMimic presets (2409/2415-D single-block obs, maskedmimic_mode field) →
    HolosomaMaskedMimicPolicy; everything else stays on HolosomaWBTPolicy so the
    existing exp:g1-wbt-2pt interactive path is untouched.
    """
    if getattr(config.task, "maskedmimic_mode", None) is not None:
        return HolosomaMaskedMimicPolicy
    actor_obs = config.observation.obs_dict.get("actor_obs", [])
    if "reduced_coords" in actor_obs or "target_poses" in actor_obs:
        return HolosomaMaskedMimicPolicy
    return HolosomaWBTPolicy


def _build_config_from_argv() -> InferenceConfig:
    """Same minimal ``inference:<preset> --task.*`` parser as run_policy's fallback."""
    from dataclasses import replace

    args = sys.argv[1:]
    preset_name = None
    remaining = []
    for a in args:
        if a.startswith("inference:"):
            preset_name = a.split(":", 1)[1]
        else:
            remaining.append(a)
    if not preset_name:
        logger.error("Usage: run_sim_interactive inference:<preset> [--task.field value ...]")
        sys.exit(1)
    if preset_name not in EXT_DEFAULTS:
        logger.error(f"Unknown preset: {preset_name}. Available: {list(EXT_DEFAULTS.keys())}")
        sys.exit(1)
    config = EXT_DEFAULTS[preset_name]

    _BOOL_TRUE = {"true", "1", "yes"}
    _BOOL_VALS = _BOOL_TRUE | {"false", "0", "no"}
    i = 0
    task_overrides: dict = {}
    while i < len(remaining):
        key = remaining[i]
        if not key.startswith("--"):
            i += 1
            continue
        parts = key[2:].split(".", 1)
        if len(parts) != 2 or parts[0] != "task":
            i += 1
            continue
        field = parts[1].replace("-", "_")
        field_type = type(getattr(config.task, field, ""))
        if field_type is bool:
            nxt = remaining[i + 1].lower() if i + 1 < len(remaining) else None
            if nxt in _BOOL_VALS:
                task_overrides[field] = nxt in _BOOL_TRUE
                i += 2
            else:
                task_overrides[field] = True
                i += 1
        elif i + 1 < len(remaining):
            val = remaining[i + 1]
            task_overrides[field] = int(val) if field_type is int else float(val) if field_type is float else val
            i += 2
        else:
            i += 1
    if task_overrides:
        config = replace(config, task=replace(config.task, **task_overrides))
        logger.info(f"Config overrides: {task_overrides}")
    logger.info(f"Using preset: {preset_name}")
    return config


# Trajectory-viz colors (RGBA): tracker TARGET = cyan, executed WRIST = orange.
_TRAJ_TARGET_RGBA = np.array([0.10, 0.85, 0.95, 1.0], dtype=np.float32)
_TRAJ_WRIST_RGBA = np.array([1.00, 0.55, 0.10, 1.0], dtype=np.float32)
_TRAJ_MAX_PTS = 600  # ~12 s at 50 Hz per hand per trace
_TRAJ_LINE_W = 4.0  # connector pixel width


class _TrajectoryViz:
    """Accumulates and draws the commanded tracker-target vs executed-wrist world
    polylines in the passive viewer's ``user_scn``, per visible wrist.

    The policy taps ``_vive_last_target_w`` / ``_vive_last_measured_w`` (cond_idx →
    world xyz) each armed tick. We append those to per-hand ring buffers, draw each
    as a line strip (target = cyan, wrist = orange), and print the live tracking
    error. ``reset()`` (on RECENTER / STOP) clears the buffers so each clutch
    session starts a fresh trace. Pure viz — never touches the obs."""

    def __init__(self):
        # cond_idx -> deque[np.ndarray(3)]
        self._target: dict[int, deque] = {}
        self._wrist: dict[int, deque] = {}
        self._last_err: dict[int, float] = {}

    def reset(self) -> None:
        self._target.clear()
        self._wrist.clear()
        self._last_err.clear()

    def update(self, policy) -> None:
        tgt = getattr(policy, "_vive_last_target_w", None) or {}
        meas = getattr(policy, "_vive_last_measured_w", None) or {}
        for bi, p in tgt.items():
            self._target.setdefault(bi, deque(maxlen=_TRAJ_MAX_PTS)).append(np.asarray(p, dtype=np.float64).copy())
        for bi, p in meas.items():
            self._wrist.setdefault(bi, deque(maxlen=_TRAJ_MAX_PTS)).append(np.asarray(p, dtype=np.float64).copy())
            if bi in tgt:
                self._last_err[bi] = float(np.linalg.norm(np.asarray(tgt[bi]) - np.asarray(p)))

    def err_str(self) -> str:
        if not self._last_err:
            return ""
        parts = [f"body{bi}={e * 100:.1f}cm" for bi, e in sorted(self._last_err.items())]
        return "track err " + " ".join(parts)

    def draw(self, user_scn) -> None:
        import mujoco

        user_scn.ngeom = 0

        def _strip(buf, rgba):
            pts = list(buf)
            for i in range(len(pts) - 1):
                if user_scn.ngeom >= user_scn.maxgeom:
                    return
                g = user_scn.geoms[user_scn.ngeom]
                mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_LINE, np.zeros(3), np.zeros(3), np.zeros(9), rgba)
                mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_LINE, _TRAJ_LINE_W, pts[i], pts[i + 1])
                user_scn.ngeom += 1

        for bi, buf in self._target.items():
            _strip(buf, _TRAJ_TARGET_RGBA)
        for bi, buf in self._wrist.items():
            _strip(buf, _TRAJ_WRIST_RGBA)


def main() -> None:
    import mujoco
    import mujoco.viewer

    try:
        from holosoma_inference.config.config_values.inference import get_annotated_inference_config

        annotated = Annotated[InferenceConfig, get_annotated_inference_config()]  # type: ignore[valid-type]
        config = tyro.cli(annotated, config=TYRO_CONFIG)
    except (Exception, SystemExit) as e:
        logger.warning(f"tyro CLI failed ({e}); using minimal argv parser")
        config = _build_config_from_argv()

    # _setup_headless_sim requires headless_output to be set (it only names the
    # would-be video, which the interactive viewer never writes). Default it so
    # the operator doesn't have to pass a meaningless path.
    if not getattr(config.task, "headless_output", ""):
        from dataclasses import replace

        config = replace(config, task=replace(config.task, headless_output="/tmp/run_sim_interactive_unused.mp4"))

    # Build the policy with an in-process MuJoCoInterface (no hardware).
    policy_class = _select_interactive_policy_class(config)
    logger.info(f"Interactive sim policy: {policy_class.__name__}")
    policy = _create_headless_policy(policy_class, config)
    config = policy.config  # _create_headless_policy may rebuild it
    sim = policy._headless_sim
    mj_model = sim["model"]
    mj_data = sim["data"]
    sim_params = sim["sim_params"]
    control_dt = float(sim_params["control_dt"])  # 0.02 s @ 50 Hz

    # Prime the motion buffers at frame 1 (same as the headless eval) so the
    # first policy step has valid ref command, then seed the start pose.
    if hasattr(policy, "preload_motion_frame"):
        policy.preload_motion_frame(1)

    max_steps = int(getattr(config.task, "headless_max_frames", 0) or 0)

    logger.info(
        f"Interactive sim: control_dt={control_dt * 1000:.0f}ms "
        f"(rl_rate={1.0 / control_dt:.0f}Hz), max_steps={max_steps or 'until window closed'}"
    )
    # CRITICAL — the robot must NOT move on its own; the operator arms it.
    # This sim spawns the robot STANDING at the motion's frame-0 pose (the same
    # state the validated headless eval runs from). The trained policy IS the
    # balance controller, so it must run to stand — but we (a) FREEZE the motion
    # clip at frame-0 (motion_clip_progressing=False) so it doesn't play the
    # pick-up motion, and (b) DISARM the VIVE override (_vive_armed=False) so the
    # trackers are ignored. Result: the robot holds its standing pose, dead still,
    # until you press a key.
    #
    # Do NOT use get_ready_state here: that base ramp drives with the low ONNX KP
    # (~40) and the robot COLLAPSES (holosoma_wbt.py:1021-1022) — that was the
    # "rolling on the ground" bug. No ramp is needed: frame-0 IS the standing pose.
    policy.use_policy_action = True  # policy balances (required to stand)
    policy.get_ready_state = False
    policy.motion_clip_progressing = False  # clip FROZEN at frame 0 → no auto motion
    policy._stiff_hold_active = False
    policy._vive_armed = False  # trackers IGNORED until START

    have_vive = policy._vive_live is not None

    # --- keyboard controls in the viewer ---
    #   ] / Enter = START : arm the VIVE override (hands follow your trackers)
    #   o         = STOP  : disarm — freeze hands back at the standing pose
    #   m         = CLIP  : (optional) play the NPZ motion clip instead of VIVE
    pending: dict[str, bool] = {"start": False, "stop": False, "clip": False, "soft": False, "recenter": False}
    has_recenter = hasattr(policy, "_vive_recenter")

    def _key_callback(keycode: int) -> None:
        ch = chr(keycode) if 0 <= keycode < 0x110000 else ""
        if ch in ("]", "\r", "\n") or keycode == 257:  # ] or Enter
            pending["start"] = True
        elif ch in ("o", "O"):
            pending["stop"] = True
        elif ch in ("m", "M"):
            pending["clip"] = True
        elif ch in ("z", "Z"):  # soft-stop (damped descent to limp)
            pending["soft"] = True
        elif ch in ("x", "X"):  # recenter the VIVE clutch (slide hands back to ref)
            pending["recenter"] = True

    banner = (
        "\n============== INTERACTIVE SIM — robot STANDING STILL (not armed) ==============\n"
        "  Click the MuJoCo window, then:\n"
    )
    if have_vive:
        banner += (
            "    []] or Enter   START — stand under the policy, then ARM VIVE (hands follow your trackers)\n"
            "    [x]            RECENTER — slide hands back to the reference pose + re-anchor the clutch (resets traj)\n"
            "    [o]            STOP  — disarm: hands freeze back at the standing pose\n"
            "    (while armed: CYAN line = tracker target, ORANGE line = executed wrist; track err logged ~1 Hz)\n"
        )
    else:
        banner += "    (no VIVE: set HOLOSOMA_VIVE_WS_URL + run the bridge to teleop)\n"
    banner += (
        "    [m]            CLIP  — play the NPZ motion clip (no VIVE needed)\n"
        "    [z]            SOFT-STOP — damped descent: robot SINKS/FOLDS to harness/floor (kp→0, kd held); [i] to recover\n"
        "  Close the window to quit. PURE SIM — no motors.\n"
        "==============================================================================="
    )
    logger.info(colored(banner, "cyan"))

    # Trajectory viz: draw the tracker TARGET (cyan) vs executed WRIST (orange)
    # world polylines while armed, reset each clutch session. Only meaningful for
    # the MM/WBT live-VIVE path that exposes _vive_last_target_w.
    traj = _TrajectoryViz()
    has_traj = hasattr(policy, "_vive_last_target_w")

    step = 0
    with mujoco.viewer.launch_passive(mj_model, mj_data, key_callback=_key_callback) as viewer:
        while viewer.is_running():
            if max_steps and step >= max_steps:
                logger.info(f"Reached max_steps={max_steps}; stopping.")
                break
            t0 = time.time()

            if pending["start"]:
                pending["start"] = False
                # Use the SAME handler the real-robot keyboard ']' hits, so sim and
                # deploy share identical START semantics: it brings the policy up
                # holding the frozen standing pose with VIVE DISARMED. We then arm
                # explicitly here (the operator's ']' = "start AND hand me the
                # trackers" in the sim) via the same _toggle_vive_teleop the
                # real-robot 'start' button hits, so the obs/clutch path is shared.
                policy._handle_start_policy()
                if have_vive:
                    policy._toggle_vive_teleop()  # snapshot clutch + arm
                    traj.reset()  # fresh trajectory trace for this arm session
                    logger.info(colored("START — VIVE armed (clutch). Move your trackers.", "green"))
                else:
                    logger.info(colored("START — NPZ clip playing (no VIVE).", "green"))
            if pending["recenter"]:
                pending["recenter"] = False
                if has_recenter:
                    policy._vive_recenter()
                    traj.reset()  # fresh trace from the re-anchored origin
            if pending["stop"]:
                pending["stop"] = False
                policy._handle_stop_policy()  # disarms VIVE + stiff hold
                traj.reset()
                logger.info(colored("STOP — disarmed, holding the standing pose.", "yellow"))
            if pending["clip"]:
                pending["clip"] = False
                policy._vive_armed = False
                policy.motion_clip_progressing = True  # let the NPZ clip play
                logger.info(colored("CLIP — playing the NPZ motion (VIVE disarmed).", "blue"))
            if pending["soft"]:
                pending["soft"] = False
                if hasattr(policy, "_trigger_soft_stop"):
                    policy._trigger_soft_stop()  # damped descent; robot sinks to harness/floor; [i] recovers
                    logger.info(colored("SOFT-STOP — robot SINKS/FOLDS (kp→0, kd held). Press [i] to recover.", "red"))
                else:
                    logger.info(colored("SOFT-STOP not supported by this policy (use STOP).", "yellow"))

            policy.policy_action()

            # Trajectory viz: append the latest armed tracker-target + executed
            # wrist samples, redraw the polylines, and log the live error (~1 Hz).
            if has_traj:
                traj.update(policy)
                traj.draw(viewer.user_scn)
                if step % 50 == 0:
                    es = traj.err_str()
                    if es:
                        logger.info(colored(f"[traj] {es}  (cyan=tracker target, orange=wrist)", "cyan"))

            viewer.sync()
            step += 1
            dt = control_dt - (time.time() - t0)
            if dt > 0:
                time.sleep(dt)

    if have_vive:
        policy._vive_live.shutdown()
    logger.info(f"Interactive sim finished after {step} steps.")


if __name__ == "__main__":
    main()
