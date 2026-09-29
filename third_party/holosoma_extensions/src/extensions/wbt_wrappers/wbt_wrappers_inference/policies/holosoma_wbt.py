"""Holosoma Whole-Body Tracking Policy.

For ONNX models exported by wbt_training (holosoma format).
Supports all obs modes: dense (157D), global (160D), global_vel (166D), 3pt (144D),
2pt/3pt hybrid (660D/672D), 3pt hybrid global-vel (732D, walk-enabled + 6D orn).
Obs mode is auto-detected from the inference config term list.

- ONNX I/O is auto-detected (two export formats):
    * DEPLOY export: input "obs" + "time_step"; output "actions" (+ aux joint_pos,
      joint_vel, ref_pos_xyz, ref_quat_xyzw — NOT consumed here; the policy sources
      those from the NPZ reference motion).
    * BROWSER/mjswan export: input "actor_obs" only (no time_step); output "action".
  Only the action output is read; the resolved I/O names are set in setup_policy.
- Metadata: action_scale, kp, kd, dof_names, robot_urdf, experiment_config
- Joint ordering: holosoma training order matches robot SDK order (no reordering needed)
- Global modes require mocap for root position tracking (motion_ref_pos_b, base_lin_vel)
"""

from __future__ import annotations

import json
import math
import os
import threading
import time
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import onnxruntime
from loguru import logger
from termcolor import colored

from holosoma_inference.config.config_types.inference import InferenceConfig
from holosoma_inference.policies.base import BasePolicy
from holosoma_inference.policies.wbt import PinocchioRobot, WholeBodyTrackingPolicy
from holosoma_inference.utils.math.quat import (
    matrix_from_quat,
    quat_inverse,
    quat_mul,
    quat_rotate_inverse,
    quat_to_rpy,
    rpy_to_quat,
    subtract_frame_transforms,
    wxyz_to_xyzw,
    xyzw_to_wxyz,
)
from wbt_wrappers_inference.utils.eval_utils import compute_height_scan_mj_ray
from wbt_wrappers_inference.utils.normalizer import (
    clip_normalized_obs as _shared_clip_normalized_obs,
    load_normalizer_from_path,
)
from wbt_wrappers_inference.utils.obs_utils import (
    compute_motion_alignment as _shared_compute_motion_alignment,
    detect_obs_mode,
)
from wbt_wrappers_inference.config_values.robot_config import (
    LOWER_BODY_INDICES as _LOWER_BODY_INDICES,
    VR_2POINT_BODIES as _VR_2POINT_BODIES,
    VR_2POINT_OFFSETS as _VR_2POINT_OFFSETS_LIST,
    VR_3POINT_BODIES as _VR_3POINT_BODIES,
    VR_3POINT_OFFSETS as _VR_3POINT_OFFSETS_LIST,
    VR_3POINT_OFFSETS_TCP as _VR_3POINT_OFFSETS_TCP_LIST,
    VR_5POINT_BODIES as _VR_5POINT_BODIES,
    VR_5POINT_OFFSETS_TCP as _VR_5POINT_OFFSETS_TCP_LIST,
)
from wbt_wrappers_inference.config_values.task import DEFAULT_COMPLIANCE

_VR_3POINT_OFFSETS = np.array(_VR_3POINT_OFFSETS_LIST, dtype=np.float32)
# TCP-unified variant: both hands on the 0.242 m force TCP (user 2026-08-03). Selected
# per-CHECKPOINT by the `vr_3point_pos_tcp` obs term — see `_vr_pos_tcp` in setup_policy().
_VR_3POINT_OFFSETS_TCP = np.array(_VR_3POINT_OFFSETS_TCP_LIST, dtype=np.float32)
_VR_2POINT_OFFSETS = np.array(_VR_2POINT_OFFSETS_LIST, dtype=np.float32)
# 5-POINT (exp:g1-wbt-5pt-hybrid-unified-anneal, run dlcxc59d): the 3pt-TCP triple plus
# both `*_ankle_roll_link` origins. TCP-ONLY — no 0.18 m-lever 5pt checkpoint was ever
# trained, so unlike 3pt there is no per-checkpoint choice to make here.
_VR_5POINT_OFFSETS_TCP = np.array(_VR_5POINT_OFFSETS_TCP_LIST, dtype=np.float32)
# Operator-facing names for the keypoints, in `VR_*POINT_BODIES` order, so a refusal can say
# WHICH point is out instead of printing five bare numbers.
_VR_POINT_LABELS = ("LEFT hand", "RIGHT hand", "head/torso", "LEFT foot", "RIGHT foot")
# How many of the 5pt keypoints the operator can actually PLACE. The rig has THREE pucks; the
# two FOOT points are inferred by Motion_Completion from those same three and cannot be aimed
# — so the pre-arm pose gate must not refuse on them (2026-08-28, measured on the rig: the
# foot gaps bottomed out at 0.101 m / 0.062 m against a 0.070 m tol while the operator closed
# the hand gaps from 0.152 m to 0.084 m, i.e. arming was structurally impossible). The cause
# is a real property of the reference, not a bug: under HOLOSOMA_INIT_READY_POSE (which the
# launcher sets) the held reference is the READY POSE — `default_dof_angles` with the 14 arm
# joints zeroed — and that stance is a 38° CROUCH (`knee = +0.669 rad`), putting both ankles
# only 0.722 m below the pelvis. A completion of a normally-standing operator is nearly
# straight-legged, ~0.80 m: the z term ALONE is ~0.08 m and exceeds the tolerance before y is
# counted. Both reference feet are symmetric at y = ±0.119, x ≈ 0 — the asymmetry is not the
# issue, the knee bend is. Jump safety for these points is unaffected: the ABSOLUTE arm clutch
# captures `held − wire` for ALL FIVE points at arm (see _toggle_vive_teleop), so a derived
# point's offset is re-based to zero exactly like a tracked one's. See _vive_arm_pose_ok.
_VR_5POINT_OPERATOR_PLACED = 3
# A derived (completion-inferred) point this far from the held reference means the completion
# is not producing a STANDING body at all, rather than merely a differently-stanced one. Two
# plausible standing G1s differ by at most ~0.15 m at the ankle; a nonsense completion — pucks
# left lying on a bench — measured 0.215 m / 0.451 m. Warn, do not refuse: the clutch re-bases
# it either way, so the value of noticing is diagnostic.
_VR_DERIVED_GAP_WARN_M = 0.30
# The DENSE twin of _VR_5POINT_OPERATOR_PLACED, by JOINT NAME (never by index — MC's order and
# the policy's are both real orders and they are NOT the same). The three pucks steer the two
# ARMS (shoulder/elbow/wrist, through the completion's wrist IK) and the HEADING (waist_yaw);
# the legs and the torso's non-yaw DoFs are the completion's own inference off those same three
# points, so gating on them is the same lockout the 5pt feet were — and here it is not even
# close: the READY POSE the gate compares against (see _VR_5POINT_OPERATOR_PLACED) holds
# `knee = +0.669 rad`, `hip_pitch = −0.312`, `ankle_pitch = −0.363`, so a straight-legged
# completion misses the knee by 0.67 rad — more than THREE TIMES the 0.20 rad tol, and not
# reachable by an operator holding hand pucks.
_VR_DENSE_DERIVED_JOINT_KEYS = ("hip", "knee", "ankle", "waist_roll", "waist_pitch")
# Per-joint counterpart to _VR_DERIVED_GAP_WARN_M (rad). It MUST clear the reference's own
# crouch: the ready pose holds `knee = +0.669 rad`, so a perfectly good straight-legged
# completion is legitimately 0.669 rad away and anything at or below that fires on every
# normal arm. 1.20 rad ≈ 69° sits above it with margin and still catches a genuinely folded
# completion (the knee's travel runs to ~2.5 rad, i.e. ~1.8 rad from the reference).
# The first value written here was 0.60 — a false alarm on every single arm, caught only
# because the dense test was rewritten to use the REAL gap instead of a token overshoot.
_VR_DENSE_DERIVED_GAP_WARN_RAD = 1.20

# Soft contact-phase (C1 unified presets) defaults — MUST equal HybridContactConfig's
# contact_phase_{theta_on,theta_off,tau_s,warmup_steps} in
# `origin/hybrid_control:src/extensions/wbt_wrappers/wbt_training/config_values/hybrid.py`.
# ⚠️ That training code is NOT vendored on this (sim2real) branch — the wbt_training
# copy here predates the unified work — so these four values are the only copy on this
# BRANCH. Cross-check with `git show origin/hybrid_control:…/hybrid.py` before changing.
# Declared once here so the
# __init__ env parsing and the getattr fallbacks in _update_contact_phase (which
# must tolerate the object.__new__ test stubs) cannot drift apart.
_CONTACT_PHASE_THETA_ON = 2.0  # N — above this = contact
_CONTACT_PHASE_THETA_OFF = 1.5  # N — below this = no contact; hysteresis lower edge
_CONTACT_PHASE_TAU_S = 0.05  # s — first-order LPF time constant
_CONTACT_PHASE_WARMUP_STEPS = 5  # control ticks after START with the latch forced OFF
# Seconds the "live F/T reading is not yet trusted" window extends past the LAST tare
# request. Must cover the bridge actually APPLYING that tare: it averages ~1 s with
# `--ft-algo rnea` (the launcher default) while still republishing the frozen pre-tare
# value, so a smaller value re-opens the un-tared-latch hole. `--ft-algo bota` warms up
# for up to ~3 s — raise HOLOSOMA_FORCE_TARE_SETTLE_SECONDS accordingly.
_FORCE_TARE_SETTLE_S = 1.5


def parse_force_meas_gate_cfg(env) -> tuple[bool, float]:
    """Strictly parse the per-hand F_meas-on-F_cmd gate and its threshold."""
    raw = str(env.get("HOLOSOMA_FORCE_MEAS_GATE_ON_CMD", "0")).strip().lower()
    if raw in ("1", "true", "yes", "on"):
        enabled = True
    elif raw in ("", "0", "false", "no", "off"):
        enabled = False
    else:
        raise ValueError(f"HOLOSOMA_FORCE_MEAS_GATE_ON_CMD must be one of 1/0/true/false/yes/no/on/off, got {raw!r}")

    raw_threshold = str(env.get("HOLOSOMA_FORCE_MEAS_GATE_CMD_N", "1e-6")).strip() or "1e-6"
    threshold = float(raw_threshold)
    if not math.isfinite(threshold) or threshold < 0.0:
        raise ValueError(f"HOLOSOMA_FORCE_MEAS_GATE_CMD_N must be finite and >= 0, got {threshold!r}")
    return enabled, threshold


def parse_force_hands_cfg(env) -> tuple[bool, bool]:
    """Which hands' F/T sensors are USED AT ALL. Returns ``(left_enabled, right_enabled)``.

    This is not a gate and not a threshold — it is "is this sensor part of the system".
    A disabled hand's reading is zeroed at the single point it is read, so it reaches
    NOTHING downstream: not the actor obs, not ``contact_phase``, not the servo, not the
    runaway valve, not the hard stop, and it can carry no ``F^cmd``.

    Why a separate mechanism from ``HOLOSOMA_FORCE_MEAS_GATE_ON_CMD``: that gate keys off
    the *commanded* force, so an unused hand only reads zero as a SIDE EFFECT of never
    being commanded — and it re-opens the moment someone runs with ``GATE_CMD=0`` for a
    fidelity A/B. "This sensor is not in the loop" has to be its own statement.

    Default ``both`` = byte-identical to the two-hand behaviour.
    """
    raw = str(env.get("HOLOSOMA_FORCE_HANDS", "both")).strip().lower() or "both"
    try:
        return {"both": (True, True), "left": (True, False), "right": (False, True), "none": (False, False)}[raw]
    except KeyError:
        raise ValueError(
            f"HOLOSOMA_FORCE_HANDS must be one of both/left/right/none, got {raw!r}"
        ) from None


def parse_force_estop_cfg(env) -> float:
    """Strictly parse the hard F/T stop threshold (N); 0 disables it.

    STRICT rather than ``float(... or 0.0)`` because this is a safety limit and the
    failure mode of a tolerant parse is silent: ``HOLOSOMA_FORCE_ESTOP_N=20N`` or a
    stray space-separated value would fall back to "off" on a run the operator believes
    is protected. A negative threshold is likewise refused rather than clamped — it can
    only mean the caller meant something else (every ‖F‖ ≥ 0 would trip instantly).
    """
    raw = str(env.get("HOLOSOMA_FORCE_ESTOP_N", "0.0")).strip() or "0.0"
    try:
        threshold = float(raw)
    except ValueError as exc:
        raise ValueError(f"HOLOSOMA_FORCE_ESTOP_N must be a number (N), got {raw!r}") from exc
    if not math.isfinite(threshold) or threshold < 0.0:
        raise ValueError(f"HOLOSOMA_FORCE_ESTOP_N must be finite and >= 0 (0 = off), got {threshold!r}")
    return threshold


def resolve_onnx_io_names(input_names, output_names) -> tuple[str, bool, str]:
    """Resolve the two supported Holosoma ONNX export signatures."""
    if "obs" in input_names:
        obs_input = "obs"
    elif "actor_obs" in input_names:
        obs_input = "actor_obs"
    elif input_names:
        obs_input = input_names[0]
    else:
        raise ValueError("ONNX has no inputs")

    if "actions" in output_names:
        action_output = "actions"
    elif "action" in output_names:
        action_output = "action"
    else:
        raise ValueError(
            f"ONNX has no supported action output ('actions' or 'action'); available outputs: {output_names}"
        )
    return obs_input, "time_step" in input_names, action_output


def validate_onnx_action_shape(shape, num_dofs: int) -> None:
    """Reject statically incompatible action tensors; dynamic axes are checked at runtime."""
    if len(shape) not in (1, 2):
        raise ValueError(f"ONNX action output has shape {shape}; expected [{num_dofs}] or [1, {num_dofs}]")
    if len(shape) == 2 and isinstance(shape[0], int) and shape[0] != 1:
        raise ValueError(f"ONNX action output has batch width {shape[0]}; expected 1")
    if isinstance(shape[-1], int) and shape[-1] != num_dofs:
        raise ValueError(f"ONNX action output has width {shape[-1]}; expected {num_dofs}")


def parse_ort_thread_policy(env) -> tuple[int, bool]:
    """Parse ``HOLOSOMA_ORT_INTRA_THREADS`` / ``HOLOSOMA_ORT_SPIN`` — STRICTLY.

    Why this knob exists: ``onnxruntime.InferenceSession(path)`` with NO SessionOptions
    sizes its intra-op pool to the CORE COUNT and lets those workers SPIN-WAIT between
    calls. That is the right trade for a batch job and the wrong one for a 50 Hz
    controller, whose forward pass takes ~0.4 ms of a 20 ms tick — the pool then spends
    ~98% of every tick busy-waiting. MEASURED on the 24-core deploy box with
    wda3h4ll/model_19000:

      rate-limited 50 Hz   default  → 22.7 cores, p99 7.6 ms
                           intra=1  →  0.12 cores, p99 3.1 ms   (190x less CPU, better tail)
      tight batch loop     default  → 3647 it/s @ 22.7 cores
                           intra=1  →  923 it/s @ 1.0 core      (4x SLOWER — batch wants the pool)

    Hence the DEFAULT here is the legacy pool (``0``, spinning ON), so that headless eval
    and every other in-process caller keep their throughput and stay byte-identical. It
    is the DEPLOY LAUNCHER (3_run_policy_vive_force.sh, ``CPU_THREADS``) that opts into
    the low-CPU pair, because only there is the 20 ms tick the binding constraint.

    Actions are BITWISE identical across every intra/spin combination (verified over
    random obs), so this is purely a thread-scheduling knob and can never change what
    the robot does.

    Returns ``(intra_op_num_threads, allow_spinning)``. ``0`` threads means "let
    onnxruntime size the pool from the core count".
    """
    raw_threads = str(env.get("HOLOSOMA_ORT_INTRA_THREADS", "0")).strip() or "0"
    try:
        threads = int(raw_threads)
    except ValueError as exc:
        raise ValueError(f"HOLOSOMA_ORT_INTRA_THREADS must be an integer >= 0, got {raw_threads!r}") from exc
    if threads < 0:
        raise ValueError(f"HOLOSOMA_ORT_INTRA_THREADS must be >= 0, got {threads}")

    raw_spin = str(env.get("HOLOSOMA_ORT_SPIN", "1")).strip().lower() or "1"
    if raw_spin in ("1", "true", "yes", "on"):
        spin = True
    elif raw_spin in ("0", "false", "no", "off"):
        spin = False
    else:
        raise ValueError(f"HOLOSOMA_ORT_SPIN must be one of 1/0/true/false/yes/no/on/off, got {raw_spin!r}")
    return threads, spin


def build_ort_session_options(env):
    """Build the ``SessionOptions`` for :func:`parse_ort_thread_policy`.

    Sets ONLY threading knobs. The graph optimization level is left at onnxruntime's own
    default (``ORT_ENABLE_ALL``) and callers must NOT pass ``providers``, so the loaded
    graph, the executing provider and therefore the numerics are untouched.

    With the defaults (``0``, spinning) the result is equivalent to passing no options at
    all: onnxruntime's own defaults are ``intra=0, inter=0, execution_mode=SEQUENTIAL``.
    Verified by measurement — same throughput, same core count, identical actions — so
    routing every caller through here does not change the legacy path.
    """
    threads, spin = parse_ort_thread_policy(env)
    opts = onnxruntime.SessionOptions()
    opts.intra_op_num_threads = threads
    # Left at onnxruntime's default (0 = auto). Irrelevant in SEQUENTIAL mode, which is
    # also its default: the actor is a plain MLP chain with no branch to overlap.
    opts.inter_op_num_threads = 0
    opts.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
    if not spin:
        # The knobs that actually stop the idle burn — without them a pool of ANY size
        # busy-waits between the 50 Hz calls.
        opts.add_session_config_entry("session.intra_op.allow_spinning", "0")
        opts.add_session_config_entry("session.inter_op.allow_spinning", "0")
    return opts


def parse_action_scale(raw, num_dofs: int) -> np.ndarray:
    """Return a flat finite per-joint action scale."""
    scale = np.asarray(raw, dtype=np.float32).reshape(-1)
    if scale.size == 1:
        scale = np.full(num_dofs, scale.item(), dtype=np.float32)
    elif scale.size != num_dofs:
        raise ValueError(f"action_scale must contain 1 or {num_dofs} values, got {scale.size}")
    if not np.all(np.isfinite(scale)):
        raise ValueError("action_scale must contain only finite values")
    return scale


def parse_unified_integ_cmd_gate(env) -> bool:
    """Parse ``HOLOSOMA_UNIFIED_INTEGRATOR_CMD_GATE`` — STRICTLY, and DEFAULT ON.

    Module-level and pure so ``__init__`` and the unit tests share one implementation.

    Two deliberate departures from the other boolean env knobs in this file:

    * **Default ON.** With the gate off, a sustained free-space F/T offset above the 2 N
      contact gate (a drifted tare, RNEA gravity-model error, a grasped object) winds the
      observe-only integrator to its elementwise anti-windup clip — tens of centimetres of
      commanded Δx that the unified policy chases, with the runaway valve never firing.
    * **Strict.** Every other flag here uses ``value in ("1","true",…)``, whose
      unrecognized-value fallback lands on the SAFE side. Here OFF is the UNSAFE state, so
      a typo (``=y``, ``=2``, ``=enabled``) must raise rather than silently disable a
      mitigation on a real robot.
    """
    raw = str(env.get("HOLOSOMA_UNIFIED_INTEGRATOR_CMD_GATE", "1")).strip().lower()
    if raw in ("1", "true", "yes", "on"):
        return True
    if raw in ("0", "false", "no", "off"):
        return False
    raise ValueError(
        "HOLOSOMA_UNIFIED_INTEGRATOR_CMD_GATE must be one of 1/0/true/false/yes/no/on/off "
        f"(OFF is the UNSAFE setting, so it is never inferred from a typo), got {raw!r}"
    )


def parse_unified_windup_warn_n(env) -> float | None:
    """Parse ``HOLOSOMA_UNIFIED_WINDUP_WARN_N`` — the ‖I‖ alarm threshold (N) read by
    :meth:`HolosomaWBTPolicy._warn_on_integrator_windup`.

    Unset / empty → ``None`` = the reviewed default, half of ``integrator_max`` (20 N at the
    40 N/axis clip). A positive number replaces it; ``0`` switches the alarm OFF. The alarm is
    print-only (it never changes a command), but a silently ignored value would leave the
    operator believing an alarm is armed when it is not — so a typo raises, like the cmd gate.

    Why it exists (2026-09-20, ``3pt-unified-anneal`` driven by the HL planner): a commanded
    hand APPROACHING a surface with an F^cmd of a few newtons and nothing under it yet winds I
    to the 20 N default within seconds — that is the unified policy closing the last
    centimetres, not a drifted tare — and the operator asked for the alarm to stop reporting
    it. ``3_run_policy_vive_force.sh`` therefore forwards ``WINDUP_WARN_N`` (default 200 = 10×,
    unreachable under the per-axis clip since ‖I‖ ≤ 40·√3 ≈ 69 N); ``WINDUP_WARN_N=20`` restores
    the reviewed alarm.
    """
    raw = str(env.get("HOLOSOMA_UNIFIED_WINDUP_WARN_N", "")).strip()
    if not raw:
        return None
    try:
        value = float(raw)
    except ValueError:
        value = math.nan
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(
            "HOLOSOMA_UNIFIED_WINDUP_WARN_N must be a finite non-negative number of newtons "
            f"(0 = alarm off, unset = half of the integrator clip), got {raw!r}"
        )
    return value


def parse_contact_phase_cfg(env) -> tuple[float, float, float, int]:
    """Parse + VALIDATE the four soft-contact-phase knobs from an env mapping.

    Returns ``(theta_on, theta_off, tau_s, warmup_steps)``. Module-level and pure so
    ``HolosomaWBTPolicy.__init__`` and the unit tests exercise the SAME code (a
    hand-copied mirror in a test would let the real guards drift away silently).

    Every check is load-bearing for a signal that is fed straight to the ONNX:

    * **Non-finite is rejected explicitly.** ``nan <= 0.0`` is False, so a bare
      comparison lets NaN through; NaN then rides beta -> phi -> obs -> (np.clip passes
      NaN) -> the ONNX -> the motors, with no guard anywhere downstream. An ``inf`` tau
      is the mirror failure: beta becomes 0 and phi silently freezes at 0 forever.
    * **Both edges must be strictly POSITIVE**, not merely ordered: ``theta_off <= 0``
      makes the release test ``||F|| < theta_off`` unsatisfiable (``||F|| >= 0``), so one
      touch would latch phi=1 for the rest of the run; ``theta_on <= 0`` latches on any
      nonzero reading.
    * **Ordered edges** (a real dead band) and **tau > 0** (the LPF must be a filter).
    * **Non-negative warmup**: a negative value would DISABLE the post-START
      suppression, which is what keeps an un-tared bias from latching phi in free space.
    """
    theta_on = float(env.get("HOLOSOMA_CONTACT_PHASE_THETA_ON", str(_CONTACT_PHASE_THETA_ON)))
    theta_off = float(env.get("HOLOSOMA_CONTACT_PHASE_THETA_OFF", str(_CONTACT_PHASE_THETA_OFF)))
    tau_s = float(env.get("HOLOSOMA_CONTACT_PHASE_TAU_S", str(_CONTACT_PHASE_TAU_S)))
    warmup_steps = int(env.get("HOLOSOMA_CONTACT_PHASE_WARMUP_STEPS", str(_CONTACT_PHASE_WARMUP_STEPS)))
    for name, val in (
        ("HOLOSOMA_CONTACT_PHASE_THETA_ON", theta_on),
        ("HOLOSOMA_CONTACT_PHASE_THETA_OFF", theta_off),
        ("HOLOSOMA_CONTACT_PHASE_TAU_S", tau_s),
    ):
        if not math.isfinite(val):
            raise ValueError(f"{name} must be finite, got {val!r}")
    if not (theta_on > 0.0 and theta_off > 0.0):
        raise ValueError(
            "HOLOSOMA_CONTACT_PHASE_THETA_{ON,OFF} must both be > 0 (a non-positive THETA_OFF makes the "
            f"hysteresis un-releasable), got on={theta_on!r} off={theta_off!r}"
        )
    if theta_off > theta_on:
        raise ValueError(
            f"HOLOSOMA_CONTACT_PHASE_THETA_OFF must be <= THETA_ON (hysteresis lower edge), got {theta_off} > {theta_on}"
        )
    if tau_s <= 0.0:
        raise ValueError(f"HOLOSOMA_CONTACT_PHASE_TAU_S must be > 0, got {tau_s!r}")
    if warmup_steps < 0:
        raise ValueError(
            "HOLOSOMA_CONTACT_PHASE_WARMUP_STEPS must be >= 0 (a negative value would DISABLE the post-START "
            f"suppression), got {warmup_steps!r}"
        )
    return theta_on, theta_off, tau_s, warmup_steps


class HolosomaWBTPolicy(WholeBodyTrackingPolicy):
    """WBT policy for holosoma-trained ONNX models with history stacking.

    Key difference from TML/BM policies: holosoma training uses the same joint
    ordering as the Unitree SDK, so NO joint reordering is needed.
    Training uses all-zeros default joint pos, so observations use raw joint
    positions (not offset by a standing pose).
    """

    # CLASS-LEVEL DEFAULTS for the 5pt / dense-unified teleop state. ``__init__`` assigns
    # every one of these again (see below) — these exist so the obs path cannot raise on an
    # object that never ran ``__init__``: the test suites construct policies with
    # ``object.__new__`` to exercise pure methods without a robot, and un-pickled state from
    # an older build has the same shape. The pre-existing fields solve this with per-site
    # ``getattr(self, "_vr_abs_off_pos", None)``; one declaration here is the same guarantee
    # without a guard to forget at the next call site. Scalars/None only — a mutable default
    # would be shared across instances.
    _is_5pt: bool = False
    _is_dense_unified: bool = False
    _is_dense_base_teleop: bool = False
    _dense_last_q: np.ndarray | None = None
    _dense_last_dq: np.ndarray | None = None
    _dense_clutch_q0: np.ndarray | None = None
    _dense_abs_off_q: np.ndarray | None = None
    _dense_name_perm: np.ndarray | None = None
    _dense_perm_src: tuple[str, ...] | None = None
    _vive_max_joint_step: float = 0.05
    _vive_arm_tol_rad: float = 0.20
    #: Why the last `_vive_arm_pose_ok()` said no: "no_frame" | "too_far". The two need
    #: OPPOSITE operator actions — one is "look at the bridge", the other is "move your
    #: hands" — and the refusal used to print the second unconditionally, which sent an
    #: operator hunting a pose error while the bridge was simply not publishing yet
    #: (2026-08-28: `snapshot()` returns None until the bridge is CALIBRATED, so there was
    #: no frame to be far from). Class-level so a stub built with `object.__new__` reads it.
    _vive_arm_refuse_why: str = ""
    #: Human-readable "which point, how far" for the last refusal, appended to the operator
    #: message. A bare list of five numbers does not say WHICH point to move, and for 5pt two
    #: of them are not the operator's to move at all.
    _vive_arm_refuse_detail: str = ""

    def __init__(self, config: InferenceConfig):
        self._obs_history: dict[str, deque] = {}
        self._history_length = config.observation.history_length_dict.get("actor_obs", 1)
        self._last_action = np.zeros(29, dtype=np.float32)
        self._motion_command_t = np.zeros((1, 58), dtype=np.float32)
        self._motion_command_0 = np.zeros((1, 58), dtype=np.float32)
        self._ref_quat_xyzw_t = np.zeros((1, 4), dtype=np.float32)
        self._ref_quat_xyzw_0 = np.zeros((1, 4), dtype=np.float32)
        self._ref_pos_xyz_t = np.zeros((1, 3), dtype=np.float32)
        self._ref_pos_xyz_0 = np.zeros((1, 3), dtype=np.float32)
        self._ref_vel_xyz_t = np.zeros((1, 3), dtype=np.float32)
        self._ref_vel_xyz_0 = np.zeros((1, 3), dtype=np.float32)
        self._action_scale = np.ones(29, dtype=np.float32)
        self._obs_terms: list[str] = []
        self._obs_dim: int = 0
        self._obs_mode: str = "dense"  # dense, global, global_vel, 5pt, 3pt, 3pt_global, 2pt
        self._is_3pt = False
        self._is_2pt = False
        # 5-POINT keypoint tracking (exp:g1-wbt-5pt-hybrid-unified-anneal): the 3pt triple
        # plus both ankles. Structurally identical to 3pt everywhere — same
        # command_lower_body + keypoint obs shape, same pelvis-frame FK, same xyzw-quat
        # orientation rep — just N=5 instead of N=3, which is why the whole clutch / slew /
        # arm-gate path below is written against the ARRAY WIDTH rather than a hardcoded 3.
        self._is_5pt = False
        # DENSE-UNIFIED tracking (exp:g1-wbt-hybrid-unified-anneal): no keypoints at all —
        # the tracking front-end is the 58D `command` term (29 absolute joint targets ++ 29
        # velocities). Distinguished from the LEGACY dense presets by the unified marker
        # (`contact_phase` in the obs), so nothing about the legacy dense/global deploy path
        # changes. Only this family accepts the bridge's dense joint stream.
        self._is_dense_unified = False
        # BASE tracker (exp:g1-wbt) with a live bridge attached — the same joint-space
        # steering without the force block. Resolved in setup_policy; see the note there
        # for why it also requires a configured live source.
        self._is_dense_base_teleop = False
        # True when the tracked-body orientation obs is the 6D rotation rep: the 2pt
        # family (vr_2point_orn is rot6d) OR a 3pt preset whose obs carries the
        # explicit 6D term `vr_3point_orn_6d` (exp:g1-wbt-3pt-hybrid-global-vel).
        # Detected from the obs term names in setup_policy.
        self._vr_orn_6d = False
        # True ⇒ the two HAND position-tracking points sit on the 0.242 m fingertip TCP
        # (the force anchor) instead of the default 0.18 m VR-tracker lever. Set in
        # setup_policy from the `vr_3point_pos_tcp` obs term — the TCP-unified retrain of
        # exp:g1-wbt-3pt-hybrid-unified[-anneal] (run wda3h4ll onward). Read ONLY through
        # `_vr_offsets()`, never inlined, so the obs, the arm gate and the absolute arm
        # clutch can never end up on different geometry.
        self._vr_pos_tcp = False
        self._height_scan_dim: int = 0
        self._vr_frame_ids: list[int] = []
        self._mj_model = None
        self._mj_data = None

        # Motion alignment for global tracking (NPZ frame → mocap frame)
        self._align_rot_mat: np.ndarray | None = None  # 3x3 yaw rotation
        self._align_pos_offset: np.ndarray | None = None  # 3D translation

        # Lazy yaw-offset capture for pelvis-anchored presets. Initialised here
        # so headless eval (which does not call _reset_motion_state) still has
        # valid attributes on the first rl_inference call.
        self._heading_initialized = False
        self.robot_yaw_offset = 0.0
        self.motion_yaw_offset = 0.0

        # External motion (loaded from ref_motion_path NPZ)
        self._ext_motion = None
        self._ext_motion_length = 0
        self._ext_frame_idx = 0
        self._num_prepend_frames = 0  # number of prepend frames (bypass ONNX during these)
        self._prepend_stiff_active = False  # True = drive prepend from _get_manual_command with stiff KP

        # Init-interpolation state (keeps stiff KP/KD during Y-press init)
        self._init_interpolating = False
        self._init_start_q = None
        self._init_interp_count = 0

        # Soft-stop (damped descent). The Unitree binding has no Damping
        # ControlMode (only PR/AB active modes), so we emulate a soft release by
        # ramping kp→floor (0 by default) while HOLDING kd (joint damping) and
        # freezing q at the measured pose. ⚠️ This is NOT a hover-in-place: with
        # kp→0 there is NO static gravity support, so the load-bearing (crouched)
        # joints SINK/FOLD — damping bounds the velocity (no free-fall slam), but
        # the robot settles into the harness or onto the floor. It is a CONTROLLED
        # DAMPED DESCENT, safe ONLY with a harness or when you intend it to settle.
        # NOT a substitute for the hardware E-stop — it needs the policy loop +
        # DDS link alive; if comms are dead it cannot fire. To keep partial
        # gravity support (a true lowering vs a fold), set a small leg
        # HOLOSOMA_SOFT_STOP_KP_FLOOR. Recover with INIT (ramps kp back up from
        # the sagged pose). Triggered by _trigger_soft_stop().
        self._soft_stop_active = False
        self._soft_stop_count = 0
        self._soft_stop_q = None  # frozen measured pose at trigger
        self._soft_stop_steps = int(os.environ.get("HOLOSOMA_SOFT_STOP_STEPS", "100"))  # ~2 s @ 50 Hz
        self._soft_stop_kp_floor = float(os.environ.get("HOLOSOMA_SOFT_STOP_KP_FLOOR", "0.0"))

        # Action EMA low-pass filter (sim2real anti-chatter). HolosomaWBTPolicy is
        # trained with NO action filter and the offboard run_policy path drives the
        # raw core UnitreeInterface (no Dampener slew/blend) at full training KP, so
        # the unfiltered per-step action can chatter at the pelvis/hips on hardware
        # (the sim is clean — no sensor noise/latency). When alpha<1 we low-pass the
        # action in rl_inference: a_t = alpha*a_raw + (1-alpha)*a_{t-1}, and feed the
        # SMOOTHED action back as the next-step `actions` obs term (so the policy
        # sees what it actually commanded, matching HolosomaMaskedMimicPolicy and
        # avoiding command/feedback windup). alpha=1.0 → pure pass-through →
        # byte-identical to the un-filtered path (off-by-default invariant). The env
        # var wins over the config field for live tuning without re-exporting.
        # Cautious hardware band is 0.7-0.9 (a SMALL alpha adds lag and can
        # destabilize an unfiltered-trained balance policy — see the danger guard).
        _ema_cfg = float(getattr(config.task, "action_ema_alpha", 1.0))
        _ema_env = os.environ.get("HOLOSOMA_ACTION_EMA_ALPHA", "").strip()
        if _ema_env:
            try:
                self._action_ema_alpha = float(_ema_env)
            except ValueError as e:
                raise ValueError(
                    f"HOLOSOMA_ACTION_EMA_ALPHA={_ema_env!r} is not a number; expected a float in (0, 1] "
                    "(1.0 = off; 0.7-0.9 = cautious sim2real smoothing)."
                ) from e
        else:
            self._action_ema_alpha = _ema_cfg
        if not (0.0 < self._action_ema_alpha <= 1.0):
            raise ValueError(
                f"action_ema_alpha must be in (0, 1], got {self._action_ema_alpha} "
                f"(env HOLOSOMA_ACTION_EMA_ALPHA={_ema_env!r}, config={_ema_cfg}). "
                "1.0 = no smoothing; 0.7-0.9 = cautious sim2real smoothing."
            )
        # Danger-band guard. This policy is trained with NO action filter, so a low
        # alpha injects pure phase lag into a 50 Hz balance loop — too aggressive
        # and the policy fights its own lagged command and can DIVERGE/FALL (and
        # there is no software soft-release; the hardware E-stop is the only abort).
        # alpha<0.5 is experimental (gantry/bench only); alpha<0.2 is rejected
        # unless HOLOSOMA_ACTION_EMA_ALLOW_AGGRESSIVE=1 is set as an explicit opt-in.
        _allow_aggressive = os.environ.get("HOLOSOMA_ACTION_EMA_ALLOW_AGGRESSIVE", "") in ("1", "true", "True")
        if self._action_ema_alpha < 0.2 and not _allow_aggressive:
            raise ValueError(
                f"action_ema_alpha={self._action_ema_alpha} is below the 0.2 safety floor "
                "(severe lag → balance-divergence/fall risk on an unfiltered-trained policy). "
                "Start at 0.85 and step down; set HOLOSOMA_ACTION_EMA_ALLOW_AGGRESSIVE=1 to override."
            )
        # Lazily seeded from the first raw action (NOT zeros) so the first command
        # is unfiltered — no ramp-from-zero transient at START. None = not yet seeded.
        self._action_ema_state: np.ndarray | None = None

        # Debug observation logging for sim2real comparison.
        # Activated by HOLOSOMA_DEBUG_LOG env var (set to output NPZ path).
        # Logs obs_raw, obs_clipped, action, dof_pos, timestep, and per-term
        # values at each rl_inference call. Compare against sim2sim debug log
        # to diagnose sim2real discrepancies.
        self._debug_log: dict | None = None
        self._debug_log_path: str = ""

        # Optional eval-time F^cmd override (root/pelvis frame, RAW Newtons):
        # HOLOSOMA_FORCE_CMD_LOCAL="L_fx,L_fy,L_fz,R_fx,R_fy,R_fz". When unset the
        # force_cmd_local obs stays zero (pure position tracking). Training applies
        # ObsTermCfg.scale=0.04 (=1/f_cmd_max=25 N) to this term and the runtime
        # does NOT apply per-term scaling, so we bake the 0.04 in here. F^cmd is
        # obs-only (UniFP convention: never written to PhysX) — it tells the
        # policy what contact force to exert, which it realizes by shifting the
        # Eq.2 hand target ref + F^cmd/K. Right-hand +x 5 N → "0,0,0,5,0,0".
        self._force_cmd_local_scaled: np.ndarray | None = None
        # --F_on_hand_frame ablation parity: a checkpoint trained with
        # --F_on_hand_frame expects force_cmd_local in each HAND (wrist_yaw) frame,
        # NOT the pelvis frame. Set HOLOSOMA_FORCE_CMD_HAND_FRAME=1 when deploying
        # such a checkpoint so the pelvis-frame F^cmd override below is rotated into
        # the hand frame per step (mirrors training's
        # R(q_wrist_yaw)^-1 @ R(q_pelvis) @ F_pelvis; the world root orientation
        # cancels, so only the live wrist-relative-to-pelvis rotation matters —
        # computed by FK on the ACTUAL robot joints). F^cmd=0 ⇒ hand-frame 0, so
        # the default (zero-force) deploy is frame-invariant and needs no flag.
        self._force_cmd_hand_frame: bool = os.environ.get("HOLOSOMA_FORCE_CMD_HAND_FRAME", "") in ("1", "true", "True")
        # --F_on_sensor_frame ablation parity: a checkpoint trained with
        # --F_on_sensor_frame expects force_cmd_local in each {side}_ft_sensor_link
        # frame. That frame is the wrist_yaw frame post-rotated by a CONSTANT
        # R_sensor (URDF ft_sensor_joint rpy="0 -pi/2 pi"), so the deploy obs is
        # F_sensor = R_sensor^T @ (R_rel^-1 @ F_pelvis) — the hand-frame result
        # left-multiplied by the same constant. Uses the constant (not a live
        # ft_sensor URDF frame) for exact train/deploy parity. Mutually exclusive
        # with HOLOSOMA_FORCE_CMD_HAND_FRAME. F^cmd=0 ⇒ 0 (frame-invariant).
        self._force_cmd_sensor_frame: bool = os.environ.get("HOLOSOMA_FORCE_CMD_SENSOR_FRAME", "") in (
            "1",
            "true",
            "True",
        )
        if self._force_cmd_hand_frame and self._force_cmd_sensor_frame:
            raise ValueError(
                "HOLOSOMA_FORCE_CMD_HAND_FRAME and HOLOSOMA_FORCE_CMD_SENSOR_FRAME are mutually "
                "exclusive; set at most one to match the checkpoint's --F_on_{hand,sensor}_frame."
            )
        # Constant wrist→sensor rotation R_sensor from ft_sensor_joint rpy (matches
        # HandForceConfig.sensor_rpy). Symmetric involution; kept as a 3×3 matrix.
        self._force_sensor_rpy: tuple[float, float, float] = (0.0, -1.5707963267948966, 3.141592653589793)
        # Raw pelvis-frame F^cmd (unscaled Newtons, [L_xyz, R_xyz]); kept so the
        # hand-frame rotation is applied fresh each step on the live wrist pose.
        self._force_cmd_pelvis_raw: np.ndarray | None = None
        _FORCE_CMD_OBS_SCALE = 0.04  # = 1 / HandForceConfig.f_cmd_max (25 N)
        _fc_raw = os.environ.get("HOLOSOMA_FORCE_CMD_LOCAL", "")
        if _fc_raw:
            _vals = [float(x) for x in _fc_raw.split(",")]
            if len(_vals) != 6:
                raise ValueError(
                    f"HOLOSOMA_FORCE_CMD_LOCAL must be 6 comma-separated Newtons "
                    f"[L_fx,L_fy,L_fz,R_fx,R_fy,R_fz], got {len(_vals)}: {_fc_raw!r}"
                )
            self._force_cmd_pelvis_raw = np.array(_vals, dtype=np.float32).reshape(2, 3)
            # Pelvis-frame pre-scaled obs (used when NOT hand-frame). For the
            # hand-frame path we rotate per step in _force_cmd_local_obs() instead.
            self._force_cmd_local_scaled = np.array(_vals, dtype=np.float32) * _FORCE_CMD_OBS_SCALE
            _frame = (
                "SENSOR (ft_sensor_link)"
                if self._force_cmd_sensor_frame
                else "HAND (wrist_yaw)"
                if self._force_cmd_hand_frame
                else "pelvis (root)"
            )
            logger.info(
                colored(
                    f"F^cmd override ({_frame} frame, N) = {_vals}"
                    + (
                        ""
                        if (self._force_cmd_hand_frame or self._force_cmd_sensor_frame)
                        else f" → obs {self._force_cmd_local_scaled.tolist()}"
                    ),
                    "magenta",
                )
            )
        self._force_cmd_obs_scale = _FORCE_CMD_OBS_SCALE
        # Resolved lazily in _force_cmd_local_obs(): wrist_yaw Pinocchio frame ids.
        self._force_wrist_frame_ids: list[int] | None = None

        # Optional LIVE keyboard F^cmd control (offboard run_policy → real robot):
        # 6 keys nudge the selected hand's pelvis-frame F^cmd ±x/±y/±z, mirroring
        # the browser sim2sim's F^cmd sliders. Feeds BOTH the force_cmd_local obs
        # AND the Jᵀ servo feed-forward (the same two consumers as the static
        # HOLOSOMA_FORCE_CMD_LOCAL). HOLOSOMA_FORCE_CMD_KEYS=1 opts in; a fresh
        # controller subscribes to the shared keyboard listener in setup_policy
        # (works even under the joystick command channel). Unset → no keyboard
        # capture, byte-identical. Built lazily in setup_policy (needs the obs mode
        # to confirm a force_cmd_local term exists). Step/clamp are tunable.
        self._force_cmd_keys = None
        # The applied F^cmd (_force_cmd_pelvis_raw) SLEWS toward this key-accumulated
        # TARGET at most _force_cmd_key_slew_n N/axis/tick, so a burst of buffered
        # keys (or pre-START taps) can never step the obs/servo feed-forward in one
        # tick — the deploy has no soft-stop, so an OOD F^cmd jump is a real hazard.
        self._force_cmd_target_raw: np.ndarray | None = None
        # All three MUST be > 0. A non-positive slew is a safety hazard: np.clip with
        # a_min > a_max returns a_max for EVERY element, so a negative slew would drive
        # the applied F^cmd AWAY from the target unbounded (the ±clamp bounds only the
        # target, not the applied value) — a runaway obs/servo feed-forward on a robot
        # with no software stop. Floor each at a tiny positive value.
        self._force_cmd_key_step_n = max(float(os.environ.get("HOLOSOMA_FORCE_CMD_KEY_STEP_N", "2.0") or 2.0), 1e-6)
        self._force_cmd_key_clamp_n = max(float(os.environ.get("HOLOSOMA_FORCE_CMD_KEY_CLAMP_N", "25.0") or 25.0), 1e-6)
        # Per-tick slew cap (N/axis). Default == step, so a single keypress applies
        # in ~1 tick (responsive) while a 12-key burst ramps over 12 ticks (~0.24 s).
        self._force_cmd_key_slew_n = max(float(os.environ.get("HOLOSOMA_FORCE_CMD_KEY_SLEW_N", "2.0") or 2.0), 1e-6)
        self._force_cmd_keys_enabled = os.environ.get("HOLOSOMA_FORCE_CMD_KEYS", "").strip().lower() in (
            "1",
            "true",
            "yes",
            "on",
        )

        # Optional LIVE F^cmd from the :8770 WIRE (the HL-v2 online planner streams a
        # per-hand force command in the raw Bota SENSOR frame alongside the pose target;
        # see ViveLiveSource.latest_force + fuji/client/s0_driver.send's f_cmd field).
        # OFF by default → the wire's f_cmd is IGNORED and F^cmd stays 0 (pure position),
        # so the VIVE-teleop / keyboard-F^cmd / position-only-planner paths are all
        # byte-identical. This is the HARD guarantee the operator asked for: force can
        # NEVER flow from the wire into the low level unless this is explicitly enabled.
        # When ON, _poll_force_cmd_wire rotates the sensor-frame command → pelvis with the
        # live wrist FK (the SAME sensor→pelvis chain F_meas uses) and slews it into
        # _force_cmd_pelvis_raw at the existing 2 N/tick cap. Mutually exclusive with the
        # keyboard keys (they would fight over the same buffer/target).
        self._force_cmd_wire_enabled = os.environ.get("HOLOSOMA_FORCE_CMD_WIRE", "").strip().lower() in (
            "1",
            "true",
            "yes",
            "on",
        )
        if self._force_cmd_wire_enabled and self._force_cmd_keys_enabled:
            raise ValueError(
                "HOLOSOMA_FORCE_CMD_WIRE and HOLOSOMA_FORCE_CMD_KEYS are mutually exclusive "
                "(both drive _force_cmd_pelvis_raw/_force_cmd_target_raw); enable only one."
            )
        # A low-level hard-zero deadband on the COMMANDED force (Newtons): a per-hand wire
        # command whose ‖F‖ ≤ this is zeroed BEFORE it reaches the buffer, so below the
        # gate the low level is a pure position controller regardless of what the planner
        # published. Defense-in-depth on top of the planner-side gate (a planner bug can't
        # push below it). Default 0 → no low-level deadband (trust the planner-side gate);
        # set it to mirror the planner's gate for belt-and-suspenders. Only consulted when
        # HOLOSOMA_FORCE_CMD_WIRE is on.
        self._force_cmd_wire_gate_n = max(
            float(os.environ.get("HOLOSOMA_FORCE_CMD_WIRE_GATE_N", "0.0") or 0.0), 0.0
        )

        # Optional LIVE VIVE teleop source (offboard run_policy → real robot,
        # or MuJoCo sim2sim validation). When HOLOSOMA_VIVE_WS_URL is set we
        # connect to the same vive_teleop_bridge WebSocket the browser uses and
        # OVERRIDE the NPZ-FK-derived vr_{2,3}point_pos/orn obs terms with the
        # operator's live pose each step (see _compute_obs_terms). Wire reps:
        # 2pt = pos[6] + orn[12] (2×rot6d, the exact rep _compute_vr_fk produces);
        # 3pt = pos[9] + orn[12] (3×xyzw quats — the QUAT presets consume them
        # as-is; the 6D preset exp:g1-wbt-3pt-hybrid-global-vel converts quat→
        # rot6d at ingest via _quat12_to_rot6d18). Unset → byte-identical
        # NPZ-replay path.
        self._vive_live = None
        # Slew-limiter state: the last vr_{2,3}point target we actually fed to the
        # obs (NPZ-FK or slewed-live). The live override ramps FROM this toward
        # the incoming live pose, capped per tick, so first-connect / reconnect /
        # stale→fresh all RAMP instead of stepping (no arm jerk on hardware).
        self._vr_last_pos: np.ndarray | None = None  # (6,) 2pt / (9,) 3pt / (15,) 5pt
        self._vr_last_orn: np.ndarray | None = None  # (12,) 2pt rot6d / 3pt quat; (18,) 3pt rot6d; (20,) 5pt quat
        # DENSE-UNIFIED live-teleop state, the joint-space analogue of the three fields
        # above. `_dense_last_q` is the last 29-vector of joint TARGETS we fed (so a
        # stale→fresh recovery ramps from where we are), `_dense_clutch_q0` is the
        # completion's joint vector at arm/recenter (the clutch origin; doubles as the
        # armed-anchor sentinel), and `_dense_name_perm` caches the by-NAME remap from the
        # wire's joint order onto this robot's `dof_names` — recomputed whenever the wire's
        # name tuple changes, never index-aligned.
        self._dense_last_q: np.ndarray | None = None
        self._dense_last_dq: np.ndarray | None = None
        self._dense_clutch_q0: np.ndarray | None = None
        # ABSOLUTE-mode arm clutch, joint-space twin of `_vr_abs_off_pos`: the constant
        # `ref - live` captured at arm, so the first armed command equals the held pose.
        self._dense_abs_off_q: np.ndarray | None = None
        self._dense_name_perm: np.ndarray | None = None
        self._dense_perm_src: tuple[str, ...] | None = None
        self._dense_perm_missing: list[str] = []
        # Per-tick cap on the commanded joint-target change, PER JOINT (rad) — not an L2
        # norm over the vector, which would couple the joints (see the note where it is
        # applied). At 50 Hz, 0.05 → 2.5 rad/s per joint: above any teleop speed, below the
        # G1's joint limits, so like the position cap it only bites on transitions (arm,
        # recenter, stale→fresh), not in steady teleop, where the completion's own output is
        # already 30 Hz-smooth.
        self._vive_max_joint_step = float(os.environ.get("HOLOSOMA_VIVE_MAX_JOINT_STEP", "0.05"))
        # Arm gate: even with a live source connected, the override only fires
        # once ARMED. Lets the policy hold its frozen-clip standing pose until the
        # operator explicitly hands control to the trackers (START). Off → the
        # robot ignores tracker motion entirely (stands at the NPZ frame-0 pose).
        self._vive_armed: bool = False
        # Per-hand, per-tick caps. At 50 Hz, 0.02 m/tick = 1.0 m/s max hand speed;
        # 0.05 rot6d-L2/tick ≈ a few deg/tick. Tunable for hardware vs sim.
        self._vive_max_pos_step = float(os.environ.get("HOLOSOMA_VIVE_MAX_POS_STEP", "0.02"))
        self._vive_max_orn_step = float(os.environ.get("HOLOSOMA_VIVE_MAX_ORN_STEP", "0.05"))
        # WHICH keypoints those two caps cover. Default 1 = every point in ONE L2 pool (the
        # byte-identical historical behaviour). 0 = the 5pt FEET (slots 3-4, the two
        # `*_ankle_roll_link` points) pass through UN-slewed while the pool shrinks to the
        # `_VR_5POINT_OPERATOR_PLACED` upper points. The caps exist to turn a far hand/head
        # target into a glide; a GAIT is a far foot target by definition — a walking clip's feet
        # demand 19-46 mm/tick against a 5 mm cap, so slewing them does not slow a walk, it
        # replaces it with a 5 mm/tick crawl toward a target that keeps running away (FUJI's
        # walk skill measured a 287 mm mean / 601 mm max lag on a mock at the launcher's cap).
        # Only the 5pt family has feet; every other preset ignores this. What is GIVEN UP with 0:
        # a stale→fresh bridge recovery and a mid-armed wire discontinuity are no longer ramped
        # on the feet (arming is still zero-jump — the arm clutch re-bases all five points).
        # Reported per tick on RecordTap as `vr_cmd_slewed`, so a consumer measuring the cap off
        # `vr_cmd_pos` knows which points to measure.
        self._vive_slew_feet = os.environ.get("HOLOSOMA_VIVE_SLEW_FEET", "1").strip().lower() not in (
            "0",
            "false",
            "no",
            "off",
        )
        # Does a 5pt keypoint frame's OPTIONAL `ref_joints` field (29 reference joint targets +
        # velocities BY NAME — FUJI's walk skill sends the gait clip's own joints while it plays)
        # override the `command_lower_body` obs term? Default 1 = yes. That term is the reference
        # motion's 12 leg joints + velocities and on the live path it is read from the FROZEN idle
        # clip (the ready stance, zero velocity) — so during a walk the legs were told "stand" by a
        # 24-D channel while the two foot keypoints said "swing", and the policy split the
        # difference as stepping in place. With the clip's legs on the wire the policy sees what a
        # baked clip gives it in mjswan, where it walks. 5pt only, while ARMED, fresh-frame only
        # (a stale frame FREEZES the last fed override exactly as the keypoint target freezes),
        # never slewed (a slewed reference is the crawl the feet were exempted from), and a frame
        # WITHOUT the field is the byte-identical frozen-clip path. Reported per tick on RecordTap
        # as `ref_joints_live`. 0 = ignore the field (A/B on the robot without touching FUJI).
        self._vive_ref_joints = os.environ.get("HOLOSOMA_VIVE_REF_JOINTS", "1").strip().lower() not in (
            "0",
            "false",
            "no",
            "off",
        )
        self._ref_joints_last: np.ndarray | None = None  # (58,) last override fed, policy dof order
        self._ref_joints_perm: np.ndarray | None = None  # dof_names -> wire order, cached per names
        self._ref_joints_perm_src: tuple[str, ...] | None = None
        self._ref_joints_warned = False
        self._ref_joints_applied: bool | None = None  # this tick: None = no 5pt term, else applied?
        # Pre-arm gate tolerance (m, per hand): START refuses to arm VIVE unless
        # the operator's live hand pose is within this of the robot's currently
        # held hand pose, so arming never starts a long uncommanded glide. 0 or
        # negative disables the gate (NOT recommended on hardware).
        #
        # ⚠️ Only consulted in ABSOLUTE mode. The DEFAULT is RELATIVE (clutch)
        # mode (HOLOSOMA_VIVE_RELATIVE=1, on by default), which makes arming a
        # zero-jump operation regardless of where the trackers are, so the pose
        # gate is unnecessary and skipped. Set HOLOSOMA_VIVE_RELATIVE=0 to restore
        # the old absolute mapping (tracker pose IS the hand target) + pose gate.
        self._vive_arm_tol = float(os.environ.get("HOLOSOMA_VIVE_ARM_TOL", "0.07"))
        # DENSE-UNIFIED pre-arm gate, per joint (rad). The dense family has no keypoints to
        # measure a metric gap against, so the ABSOLUTE-mode gate compares the completion's
        # joint solution to the held reference joint-by-joint. 0.20 rad ≈ 11.5°: loose enough
        # that a natural standing operator arms, tight enough to refuse a folded/crouched
        # completion. Same convention as _vive_arm_tol: <= 0 disables the gate.
        #
        # ⚠️ Applied only to the joints the operator can STEER — see
        # `_VR_DENSE_DERIVED_JOINT_KEYS`. The legs and the torso's non-yaw DoFs are the
        # completion's inference off the same three pucks, and the reference clip's frame 0
        # bends both knees 0.29 rad, so gating them was a permanent lockout, not a check.
        # "Refuse a folded/crouched completion" survives for those joints as a WARNING at
        # `_VR_DENSE_DERIVED_GAP_WARN_RAD`.
        self._vive_arm_tol_rad = float(os.environ.get("HOLOSOMA_VIVE_ARM_TOL_RAD", "0.20"))
        # RELATIVE / clutch teleop. When armed (or recentered) we snapshot the
        # tracker origin `_vr_clutch_live0` = the operator's live tracker pos at
        # that instant. Thereafter the commanded hand target is
        #   ref + (live_now - live0)
        # where `ref` is the robot's held hand pose (re-read live each tick as the
        # FK of the frozen clip frame). Only the operator's RELATIVE motion since
        # the anchor drives the hands, so arming never jumps and never glides no
        # matter where the trackers sit. Recenter (SWITCH_MODE key) re-snaps the
        # anchor = hands slide back to the reference pose and the tracker origin is
        # reset. `_vr_clutch_live0 is None` ⇒ not anchored (no override).
        #
        # ORIENTATION clutch (opt-in via HOLOSOMA_VIVE_ORIENTATION=1, requires the
        # bridge launched with --orientation on): when enabled, the wrist target
        # orientation is ALSO relative — per hand
        #   R_target = (R_live · R_live0ᵀ) · R_ref
        # i.e. the wrist's rotation SINCE the anchor applied on top of the held FK
        # reference orientation, so arming is zero-jump in orn too. When OFF
        # (default), the orn target stays the frozen FK reference (position-only
        # teleop) — matching the bridge's position-first default.
        self._vive_relative = os.environ.get("HOLOSOMA_VIVE_RELATIVE", "1").strip().lower() not in (
            "0",
            "false",
            "no",
            "off",
            "",
        )
        self._vive_orientation = os.environ.get("HOLOSOMA_VIVE_ORIENTATION", "0").strip().lower() in (
            "1",
            "true",
            "yes",
            "on",
        )
        self._vr_clutch_live0: np.ndarray | None = None  # (6,)/(9,) tracker pos at anchor
        # Tracker orn at anchor, stored in the OBS rep: (12,) 2pt rot6d / 3pt quat,
        # (18,) 3pt rot6d (the 6D preset converts the quat wire rep at snapshot).
        self._vr_clutch_orn0: np.ndarray | None = None
        # ABSOLUTE-mode ARM CLUTCH (HOLOSOMA_VIVE_ABS_CLUTCH, default ON). Absolute
        # mode feeds the incoming wire as the hand target DIRECTLY, so at the arm
        # instant the obs target steps from the held FK reference to the wire. When
        # a planner (FUJI) runs the low level with HOLOSOMA_INIT_READY_POSE, the
        # policy's steady-state standing pose sits a few cm off that arm-zeroed
        # reference, so wire (== the robot's MEASURED pose, published by FUJI's
        # measured-EE clutch) ≠ FK(reference) and that step is an uncommanded arm
        # lurch (the reported "LEFT hand jerks up-left on start"). This clutch
        # removes it WITHOUT switching to relative mode (which would mis-anchor the
        # planner's absolute left pose): at arm we capture the constant offset
        #   _vr_abs_off_pos = FK(reference)_at_arm − wire_at_arm
        # and thereafter command  target = wire + _vr_abs_off_pos. At the arm tick
        # that is exactly FK(reference) (zero jump); as the wire subsequently moves
        # (Execute), target follows the wire's DELTA from the arm point based at the
        # held pose — algebraically identical to the proven relative clutch
        # (`ref + (wire − wire_at_arm)`, see _apply_clutch_delta), but the base is
        # frozen at arm rather than re-read from FK each tick. Because the whole-body
        # policy re-applies its own steady-state offset δ on top of the commanded
        # target, `hand = target + δ`: with the clutch `target = wire + (ref − wire_arm)`
        # and `ref = wire_arm − δ_arm` (that IS why the arm settled δ_arm off ref), so
        # `hand = wire + (δ_now − δ_arm)`. At arm δ_now == δ_arm ⇒ hand == wire (zero
        # jump); away from the arm pose δ varies, so a RESIDUAL reach bias (δ_now − δ_arm)
        # remains — this REDUCES/re-bases the reach error at the arm pose, it does not
        # eliminate it globally, but it is strictly better than the un-clutched path
        # (which carries the full δ_arm everywhere) and is the SAME property the proven
        # relative clutch already has.
        #
        # ORIENTATION is clutched too, via `_vr_abs_orn0` (the wire orn at arm, stored in
        # the OBS rep). Rotations COMPOSE rather than subtract, so the armed target is
        #   R_target = (R_wire · R_wire0ᵀ) · R_ref
        # computed by the SAME proven helpers the relative clutch uses
        # (`_orn_clutch_target` / `_orn_clutch_target_quat`). At arm wire == wire0 ⇒ delta =
        # I ⇒ target == FK(ref) ⇒ zero orientation jump; afterwards the wrist follows only
        # the wire's orn delta. This matters because the absolute branch has NO frozen-ref
        # orn hold (unlike relative), so without it the wrist target stepped FK-ref → wire
        # at arm — the rotational analogue of the position lurch. Both anchors are cleared
        # on every disarm/stop/soft-stop/switch.
        self._vive_abs_clutch = os.environ.get("HOLOSOMA_VIVE_ABS_CLUTCH", "1").strip().lower() not in (
            "0",
            "false",
            "no",
            "off",
            "",
        )
        self._vr_abs_off_pos: np.ndarray | None = None  # (6,)/(9,) FK(ref)−wire at arm; None ⇒ inactive
        # Wire ORN at arm, in the OBS rep (12D 2pt rot6d / 3pt quat, 18D 3pt-6D preset —
        # the quat→rot6d ingest conversion is applied at capture, like the relative anchor).
        self._vr_abs_orn0: np.ndarray | None = None
        # Warn-once latch for a live/anchor orn width that does not match the obs
        # rep (e.g. bridge left in the wrong MODE) — the obs holds the FK ref, and
        # we log ONCE per (re)anchor instead of at 50 Hz.
        self._orn_mismatch_warned = False
        _vive_ws = os.environ.get("HOLOSOMA_VIVE_WS_URL", "").strip()
        if _vive_ws:
            from wbt_wrappers_inference.utils.vive_live_source import ViveLiveSource

            _stale = float(os.environ.get("HOLOSOMA_VIVE_STALE_TIMEOUT", "0.25"))
            self._vive_live = ViveLiveSource(_vive_ws, stale_timeout=_stale)
            _mode = (
                "RELATIVE/clutch (no pose gate; recenter via 'x'/'X')"
                if self._vive_relative
                else "ABSOLUTE (pose gate)"
            )
            _orn = "POS+ORN (orn relative)" if self._vive_orientation else "POSITION-only (orn=frozen ref)"
            logger.info(
                colored(
                    # The tracking family is not known yet (setup_policy resolves the obs mode
                    # later), so this line stays family-neutral; the per-family banner is the
                    # "{2,3,5}pt FK: body frame IDs …" / "DENSE teleop: …" log from setup_policy.
                    f"LIVE VIVE teleop ENABLED from {_vive_ws} [{_mode}, {_orn}] (overrides the NPZ tracking "
                    f"target; slew cap pos={self._vive_max_pos_step} m/tick, orn={self._vive_max_orn_step}/tick, "
                    f"joint={self._vive_max_joint_step} rad/tick)",
                    "cyan",
                )
            )
            if self._vive_orientation:
                logger.info(
                    colored(
                        "  ⚠️ orientation clutch ON — launch the bridge with --orientation on, else orn stays frozen.",
                        "yellow",
                    )
                )

        # --- Optional LIVE F/T (Bota) source for the hybrid / force-ext presets ---
        # When HOLOSOMA_FORCE_WS_URL is set we connect to the force_bridge WS and
        # fill the force_meas_local (hybrid) / force_ext_local (force-ext) obs terms
        # — otherwise zero — with the real compensated+tared Bota reading each step.
        # The bridge streams per-hand SENSOR-frame Newtons; we rotate sensor→wrist→
        # pelvis here (_force_meas_to_pelvis), the INVERSE of the F^cmd pelvis→sensor
        # chain, so F^cmd and its measured reaction live in one frame. Unset →
        # byte-identical (force obs stays zero ⇒ pure 2pt position tracking). The
        # sign of the physical Bota vs training's "force ON the robot" convention
        # is a hardware unknown resolved on the gantry (HOLOSOMA_FORCE_MEAS_SIGN).
        self._force_live = None
        self._force_meas_sign = float(os.environ.get("HOLOSOMA_FORCE_MEAS_SIGN", "1.0"))
        # Cached per-hand measured force in the PELVIS frame (raw N, [L,R]); the
        # last value computed in _compute_obs_terms, reused by the servo in
        # rl_inference (same tick) so the sensor is read once.
        self._force_meas_pelvis: np.ndarray = np.zeros((2, 3), dtype=np.float64)
        # Optional F_meas low-pass (sim2real anti-chatter on the FORCE path, the exact
        # analogue of HOLOSOMA_ACTION_EMA_ALPHA on the action path). Today the raw Bota
        # reading feeds force_meas_local/force_ext_local + the servo + the runaway check
        # UNFILTERED; on hardware a noisy reading — especially the un-removed inertial
        # reaction under gravity-only compensation — can drive the hybrid loop into an
        # escalating oscillation. HOLOSOMA_FORCE_MEAS_EMA_ALPHA<1 low-passes the
        # PELVIS-frame F_meas at ONE point (so obs, servo, and runaway all see the same
        # smoothed value): F_t = alpha*F_raw + (1-alpha)*F_{t-1}. Default 1.0 = OFF ⇒
        # byte-identical to the unfiltered path. This does NOT depend on a good inertia
        # tensor, so it is the lowest-risk mitigation for the force-path oscillation.
        # Cautious band 0.5-0.85; lower = more lag (F_meas trails true contact, which
        # can hurt closed-loop force tracking on the servo-on checkpoint).
        _fema = os.environ.get("HOLOSOMA_FORCE_MEAS_EMA_ALPHA", "").strip()
        self._force_meas_ema_alpha = float(_fema) if _fema else 1.0
        if not (0.0 < self._force_meas_ema_alpha <= 1.0):
            raise ValueError(
                f"HOLOSOMA_FORCE_MEAS_EMA_ALPHA must be in (0, 1], got {self._force_meas_ema_alpha!r} "
                "(1.0 = off; 0.5-0.85 = cautious force-path smoothing)."
            )
        self._force_meas_ema_state: np.ndarray | None = None  # (2,3) pelvis N; None = not yet seeded
        # Optional F_meas GATE on F^cmd (sim2real anti-jitter, opt-in). When enabled
        # (HOLOSOMA_FORCE_MEAS_GATE_ON_CMD=1), the per-hand measured force is fed to the
        # policy (force_meas_local / force_ext_local obs) AND the Jᵀ servo ONLY while that
        # hand carries a nonzero commanded force (‖F^cmd[hand]‖ > gate_n); otherwise it
        # reads ZERO. So a hand with no F^cmd is a pure POSITION controller — its force
        # obs + servo push are both 0, and it never chases F/T-sensor noise (the SERVO=1
        # free-space jitter). This ties force_meas_local to force_cmd_local PER HAND: the
        # policy only ever sees a measured force paired with a command it issued. The gate
        # reads the APPLIED (slewed) F^cmd (_force_cmd_pelvis_raw), which _poll_force_cmd_keys
        # refreshes each tick BEFORE the obs is built, so pressing u/j/h/k/y/n opens the
        # gate for that hand; c/p zero selected/both targets and l only switches selection.
        # Magnitude is frame-invariant, so the
        # same threshold works for pelvis/hand/sensor-frame checkpoints. DEFAULT OFF ⇒
        # byte-identical (the raw EMA-smoothed reading always flows). ⚠️ TRADE-OFF: with
        # the gate on, a no-command hand will NOT use F_meas to comply with / reject an
        # external force — that is exactly "position control", intended, but different from
        # the trained closed-loop compliance. gate_n is a small positive N so a hand at
        # exactly 0 N is gated while any 2 N key nudge opens it.
        self._force_meas_gate_on_cmd, self._force_meas_gate_cmd_n = parse_force_meas_gate_cfg(os.environ)
        # Auto-tare-on-START: on the START edge (_handle_start_policy, A / ']'), fire a
        # BURST of re-tare requests over a short window so the LAST one zeroes the F/T
        # AFTER the init/ready-pose transient settles. Without it, a nonzero startup
        # reading (a small residual bias, a tare taken mid-motion) feeds force_meas on
        # tick 1 and — with the servo on — nudges the arms before the operator sees it,
        # which is how the first-tick startup instability began. The bridge coalesces
        # repeated {"cmd":"tare"} (a threading.Event), so N spread-out requests give it
        # several chances to catch a clean, still, free-space window. DEFAULT ON when a
        # live F/T source exists (count>0); 0 disables. ⚠️ grippers MUST be in FREE
        # SPACE at START (a contact force at tare bakes in as a constant bias).
        self._force_tare_on_start_n = int(os.environ.get("HOLOSOMA_FORCE_TARE_ON_START", "5"))
        self._force_tare_on_start_s = float(os.environ.get("HOLOSOMA_FORCE_TARE_ON_START_SECONDS", "2.0"))
        # Extra window the burst thread lingers AFTER its final request, so the thread's
        # lifetime covers the bridge actually applying that tare (~1 s of averaging
        # during which it republishes the frozen pre-tare value — see _burst). The C1
        # unified path keys its φ suppression AND its integrator gate on this thread, so
        # this is the difference between "trusted reading" and "the last tare is still
        # in flight". Harmless for every other preset (a daemon thread sleeping).
        self._force_tare_settle_s = float(
            os.environ.get("HOLOSOMA_FORCE_TARE_SETTLE_SECONDS", str(_FORCE_TARE_SETTLE_S))
        )
        # Monotonic deadline until which the live F/T reading is NOT trusted — the window
        # in which the C1 unified path holds contact_phase AND the servo integrator at
        # zero. Armed by the START burst and by every MANUAL re-tare. Keyed on a DEADLINE
        # rather than on the burst thread being alive because (a) a manual tare (R2 / T)
        # has no thread yet needs the same window — it is the documented cure for a
        # latched φ, so it has to actually work — and (b) a deadline does not lengthen
        # `_start_force_tare_burst`'s "don't stack a burst while one is alive" guard.
        #
        # WHY the settle interval exists: `request_tare()` only sets an Event. The bridge
        # then averages for ~1 s (`--ft-algo rnea`; up to ~3 s on the `bota` warm-up path)
        # and keeps republishing the FROZEN pre-tare value with a fresh timestamp — so the
        # reading never looks stale and, because the hysteresis LATCHES, a >θ_on residual
        # would pin φ≈1 in free space until the new zero lands. ⚠️ With FT_ALGO=bota raise
        # HOLOSOMA_FORCE_TARE_SETTLE_SECONDS to ≳3.
        self._force_trust_deadline: float = 0.0
        # These two are read by EVERY preset and were historically unvalidated, so clamp
        # + warn rather than raise (a hard failure would be a behaviour change for the
        # pre-existing servo deploys). Non-finite is the one unrecoverable case.
        for _attr, _name in (
            ("_force_tare_on_start_s", "HOLOSOMA_FORCE_TARE_ON_START_SECONDS"),
            ("_force_tare_settle_s", "HOLOSOMA_FORCE_TARE_SETTLE_SECONDS"),
        ):
            _v = getattr(self, _attr)
            if not math.isfinite(_v):
                raise ValueError(f"{_name} must be finite, got {_v!r}")
            if _v < 0.0:
                logger.warning(colored(f"{_name}={_v} < 0 — clamping to 0.", "yellow"))
                setattr(self, _attr, 0.0)
            elif _v > 30.0:
                logger.warning(
                    colored(f"{_name}={_v}s is very long; the robot ignores contact for that whole window.", "yellow")
                )
        self._force_tare_burst_thread: threading.Thread | None = None
        # Optional Jᵀ force servo (Layer B). DEFAULT OFF — the ablation switch. When
        # on (HOLOSOMA_HYBRID_SERVO=1) the servo injects Δq into the arm PD targets
        # and drives the integrator_state (I) / x_tgt_shift (Δx) obs terms, faithful
        # to the servo-ON checkpoint (31wei7by). Off → I/Δx read zero (obs width is
        # fixed 165D either way) and no Δq is added — Layer-A behavior (6y8314vq).
        self._hybrid_servo = os.environ.get("HOLOSOMA_HYBRID_SERVO", "0").strip().lower() in ("1", "true", "yes", "on")
        # Remember what the operator ASKED for, so setup_policy can warn loudly when a
        # unified (observe-only) preset ignores it. Read at SETUP time only
        # (_resolve_unified_mode); never by the 50 Hz control path.
        self._hybrid_servo_requested = self._hybrid_servo
        # C1 UNIFIED "observe-only" integrator (exp:g1-wbt-3pt-hybrid-unified[-anneal];
        # the -global-vel unified twins share the same mechanism but their inference
        # preset is not on this branch yet). Resolved in setup_policy from the PRESENCE
        # of the `contact_phase` obs term — the one unambiguous marker of the unified
        # family. When True the contact-gated integrator still runs every tick (so
        # integrator_state / x_tgt_shift carry the WOUND I/Δx the actor was trained on)
        # but Δq_arm is NEVER injected into the PD path: the policy is the sole
        # executor, exactly as in training (`servo_observe_only=True`, or an annealed
        # α=0). Keyed off the obs term rather than the checkpoint's baked flag because
        # an R1 (-anneal) ONNX carries servo_observe_only=False and its terminal α
        # lives only in unserialized curriculum state — trusting the checkpoint would
        # hand a second actuator to a policy that never trained alongside one.
        self._servo_observe_only = False
        self._has_contact_phase_term = False  # set in setup_policy; gates the RECORD field
        # Observe-only integrator command gate (see _servo_contact_gate). DEFAULT ON: it
        # is what makes the documented "Run 1 with every F^cmd at 0" bring-up genuinely
        # free of integrator-driven arm motion, given that the deploy contact gate keys
        # on ‖F_meas‖ and therefore cannot tell a drifted tare from a real touch.
        # Set HOLOSOMA_UNIFIED_INTEGRATOR_CMD_GATE=0 for the fidelity A/B.
        # STRICT parse, unlike the other boolean env knobs in this file: for every other
        # flag the unrecognized-value fallback lands on the SAFE side, but here OFF is the
        # unsafe state (it re-opens the phantom-bias windup), so a typo like `=y` or `=2`
        # must fail loudly rather than silently disable a mitigation.
        self._unified_integ_cmd_gate = parse_unified_integ_cmd_gate(os.environ)
        self._integ_windup_warned = False  # one-shot latch for the ‖I‖ alarm
        # Its threshold: None = half of integrator_max (the reviewed default), a number = that
        # many newtons, 0 = alarm OFF. Strict parse — a typo must not silently arm or disarm an
        # alarm the operator counts on either way. The launcher's WINDUP_WARN_N lands here.
        self._integ_windup_warn_n = parse_unified_windup_warn_n(os.environ)
        self._contact_phase_nonfinite_warned = False  # one-shot latch for the φ scrub
        # Soft, DEPLOYABLE per-hand contact-phase label ∈[0,1] (unified actor obs).
        # Hysteresis on ‖F_meas‖ (ON above theta_on, OFF below theta_off, HOLDS in the
        # dead band) → first-order LPF with time constant tau. Defaults mirror
        # HybridContactConfig on `origin/hybrid_control` EXACTLY (theta_on 2.0 N, theta_off 1.5 N,
        # tau 0.05 s, warmup 5 steps) so the deploy label is computed by the SAME
        # decision rule the policy trained on; env-overridable for hardware tuning.
        # ⚠️ NOT the same random variable, though, and knowingly so: training's F_meas
        # includes a simulated inertial term (1.5 kg·a_wrist, which alone can reach
        # θ_on during fast free-space motion), while the deploy bridge's default
        # `--ft-algo rnea` REMOVES m·a at the source. So the deploy label fires on
        # contact and (unlike training) essentially never on motion alone. Since φ is a
        # THRESHOLDED signal this shifts its marginal, not just its noise — the shift is
        # in the favourable direction (fewer phantom contacts) but it is a real
        # sim2real gap, not an equivalence.
        self._contact_phase: np.ndarray = np.zeros(2, dtype=np.float64)  # LPF output, [L,R]
        self._contact_hyst: np.ndarray = np.zeros(2, dtype=np.float64)  # latched binary pre-LPF
        self._contact_phase_ticks = 0  # control ticks since START (post-reset warmup)
        (
            self._contact_phase_theta_on,
            self._contact_phase_theta_off,
            self._contact_phase_tau_s,
            self._contact_phase_warmup_steps,
        ) = parse_contact_phase_cfg(os.environ)
        # Per-hand servo state (pelvis frame): integrator I [N] and setpoint shift Δx [m].
        self._servo_integ: np.ndarray = np.zeros((2, 3), dtype=np.float64)
        self._servo_delta_x: np.ndarray = np.zeros((2, 3), dtype=np.float64)
        self._servo_dq: np.ndarray = np.zeros(29, dtype=np.float64)  # arm PD-target shift, DOF order
        self._servo_arm_cols: list[list[int]] | None = None  # lazy: Pinocchio Jac cols per hand
        self._servo_arm_pd_slice: list[list[int]] = []  # lazy: 29-DOF action indices per hand
        self._force_contact_n = float(os.environ.get("HOLOSOMA_FORCE_CONTACT_N", "2.0"))  # ‖F_meas‖ gate
        self._force_runaway_n = float(os.environ.get("HOLOSOMA_FORCE_RUNAWAY_N", "0.0"))  # 0 = off (warn only)
        # HARD F/T STOP. A measured ‖F‖ above this on EITHER hand means the robot is no
        # longer doing the task — it is colliding, or the arm is being driven into
        # something. Unlike the runaway valve above (which only freezes the optional Jᵀ
        # servo and warns), this STOPS the controller: it runs the exact joystick-'B'
        # handler (`_handle_stop_policy`), so the policy stops driving, stiff hold takes
        # over at the current pose, VIVE/planner arming is dropped and F^cmd is zeroed.
        # 0 = off. See `_check_force_estop` for what is and is not checked, and why.
        self._force_estop_n, self._force_estop_latched = parse_force_estop_cfg(os.environ), False
        # Which hands' F/T sensors are in the loop at all (see parse_force_hands_cfg). A
        # disabled hand is zeroed where the sensor is read, so nothing downstream — obs,
        # contact_phase, servo, runaway valve, hard stop, F^cmd — can see it.
        self._force_hands_enabled = np.array(parse_force_hands_cfg(os.environ), dtype=bool)
        # Servo config (mirrors hybrid.HybridContactConfig defaults; env-overridable).
        self._servo_cfg = None  # built lazily in setup (after force_deploy_math import)
        _force_ws = os.environ.get("HOLOSOMA_FORCE_WS_URL", "").strip()
        if _force_ws:
            from wbt_wrappers_inference.utils.force_deploy_math import ServoConfig
            from wbt_wrappers_inference.utils.force_live_source import ForceLiveSource

            _fstale = float(os.environ.get("HOLOSOMA_FORCE_STALE_TIMEOUT", "0.25"))
            self._force_live = ForceLiveSource(_force_ws, stale_timeout=_fstale)
            self._servo_cfg = ServoConfig(
                servo_gain_k=float(os.environ.get("HOLOSOMA_HYBRID_SERVO_GAIN_K", "0.05")),
                integrator_gain_ki=float(os.environ.get("HOLOSOMA_HYBRID_SERVO_KI", "0.5")),
                integrator_max=float(os.environ.get("HOLOSOMA_HYBRID_SERVO_INTEGRATOR_MAX", "40.0")),
                servo_dq_max=float(os.environ.get("HOLOSOMA_HYBRID_SERVO_DQ_MAX", "0.15")),
                stiffness_K=float(os.environ.get("HOLOSOMA_HYBRID_SERVO_STIFFNESS_K", "200.0")),
                tcp_offset_x=float(os.environ.get("HOLOSOMA_HYBRID_SERVO_TCP_OFFSET_X", "0.242")),
            )
            logger.info(
                colored(
                    f"LIVE F/T ENABLED from {_force_ws} [sign={self._force_meas_sign:+.0f}, "
                    f"servo={'requested ON' if self._hybrid_servo else 'OFF'}, "
                    f"contact_gate={self._force_contact_n:.1f}N, "
                    f"stale={_fstale}s] → fills force_meas_local/force_ext_local",
                    "magenta",
                )
            )
            if self._force_meas_ema_alpha < 1.0:
                logger.info(
                    colored(
                        f"  F_meas low-pass ON: alpha={self._force_meas_ema_alpha} "
                        f"(F_t = {self._force_meas_ema_alpha}*F_raw + {1.0 - self._force_meas_ema_alpha:.3g}*F_prev) "
                        "— sim2real force-path anti-chatter; adds lag, cautious band 0.5-0.85",
                        "magenta",
                    )
                )
            # NOTE the loud "the arms will ACTIVELY push from force error" banner is NOT
            # emitted here. Whether the Jᵀ servo actually gets actuation authority is not
            # known until setup_policy resolves the preset family (a UNIFIED preset forces
            # it off), and printing it in __init__ made a `MODE=3pt-unified SERVO=1` run
            # self-contradict: red "SERVO ON" first, "IGNORED" second. It is emitted from
            # `_emit_servo_authority_banner`, called by `_resolve_unified_mode`.
            if self._force_meas_gate_on_cmd:
                logger.info(
                    colored(
                        f"  F_meas GATE-ON-CMD ON (threshold {self._force_meas_gate_cmd_n:g} N): a hand with no F^cmd "
                        "feeds ZERO measured force to the obs + servo → pure position control; F_meas flows only "
                        "while that hand's applied F^cmd is nonzero (c/p zero selected/both targets; l only switches).",
                        "magenta",
                    )
                )
            _hands_label = {(True, True): "BOTH hands", (True, False): "LEFT hand ONLY",
                            (False, True): "RIGHT hand ONLY", (False, False): "NEITHER hand"}[
                (bool(self._force_hands_enabled[0]), bool(self._force_hands_enabled[1]))]
            if not self._force_hands_enabled.all():
                logger.info(
                    colored(
                        f"  F/T SENSORS IN THE LOOP: {_hands_label}. The excluded hand's reading is zeroed where "
                        "the sensor is read, so it reaches NOTHING: no force_meas_local, no contact_phase, no "
                        "integrator/x_tgt_shift, no runaway valve, no hard stop, and it can carry no F^cmd.",
                        "magenta",
                    )
                )
            if self._force_estop_n > 0.0:
                logger.info(
                    colored(
                        f"  🛑 HARD F/T STOP ARMED at {self._force_estop_n:.0f} N (the bridge's COMPENSATED + tared "
                        f"‖F‖, {_hands_label}): a reading above it STOPS the controller exactly as joystick B does "
                        "(stiff hold, teleop/planner disarmed, F^cmd zeroed). Suppressed during the START tare "
                        "window; one trip per START.",
                        "red",
                    )
                )
            else:
                logger.warning(
                    colored(
                        "  ⚠️ HARD F/T STOP DISABLED (HOLOSOMA_FORCE_ESTOP_N=0): a collision will NOT stop the "
                        "controller. The hardware E-STOP is the only abort.",
                        "yellow",
                    )
                )
            if self._force_tare_on_start_n > 0:
                logger.info(
                    colored(
                        f"  Auto-tare on START: {self._force_tare_on_start_n}× over "
                        f"{self._force_tare_on_start_s:.1f}s (keep grippers in FREE SPACE at START).",
                        "magenta",
                    )
                )

        # Opt-in teleop-data RECORD tap (RECORD=1 / HOLOSOMA_RECORD=1). A broadcast-only
        # localhost WS that ships F^cmd + the pelvis-frame F_meas/servo + the 29-DOF
        # proprioception + a `running` (START) flag each tick to the out-of-process
        # recorder — the ONE signal family that lives only inside this process (F^cmd is
        # broadcast nowhere else; everything else the recorder taps from its own bridge).
        # DEFAULT OFF ⇒ this attribute is None and the per-tick call in rl_inference is a
        # single `is None` early-return, so inference is BYTE-IDENTICAL to a run without
        # it (same guarantee as _debug_log / HOLOSOMA_TELEOP_TRACK_LOG). When on, the
        # publish work is O(copy) + a bounded queue drop-oldest; the socket fan-out runs
        # on a private thread, so a slow/absent recorder can never stall the RL loop.
        self._record_tap = None
        try:
            from wbt_wrappers_inference.utils.record_tap import RecordTap, record_enabled

            if record_enabled():
                _rec_port = int(os.environ.get("HOLOSOMA_RECORD_TAP_PORT", "8779"))
                self._record_tap = RecordTap(port=_rec_port)
        except Exception as exc:  # noqa: BLE001 - the tap must never break the policy
            logger.warning(colored(f"[record-tap] disabled (init failed): {exc!r}", "yellow"))
            self._record_tap = None

        # Register signal handler to save debug log on Ctrl+C / kill
        import signal
        import atexit

        def _emergency_save_debug(signum=None, frame=None):
            self._save_debug_log()
            if signum is not None:
                # Save the debug log, then re-raise to the default handler so the
                # process terminates. ⚠️ This is NOT a soft stop: the base policy
                # does NO motor cleanup on SIGINT/SIGTERM (base.run() catches
                # KeyboardInterrupt and passes; the Unitree interface has no
                # limp/zero-torque path). The G1 firmware LATCHES the last lowcmd
                # at full kp/kd — the robot freezes stiff, it does NOT go limp.
                # The ONLY safe abort is the hardware E-stop.
                signal.signal(signum, signal.SIG_DFL)
                os.kill(os.getpid(), signum)

        signal.signal(signal.SIGINT, _emergency_save_debug)
        signal.signal(signal.SIGTERM, _emergency_save_debug)
        atexit.register(self._save_debug_log)

        super().__init__(config)

    def setup_policy(self, model_path):
        # sess_options ONLY (never `providers`): the provider list stays whatever
        # onnxruntime would have picked, so this is a pure thread-scheduling change.
        # See parse_ort_thread_policy for the 22.7-cores → 0.12-cores measurement.
        _ort_threads, _ort_spin = parse_ort_thread_policy(os.environ)
        self.onnx_policy_session = onnxruntime.InferenceSession(
            model_path, sess_options=build_ort_session_options(os.environ)
        )
        logger.info(
            f"ONNX threads: intra_op={_ort_threads or 'auto(cores)'} spinning={'ON' if _ort_spin else 'off'} "
            f"(HOLOSOMA_ORT_INTRA_THREADS / HOLOSOMA_ORT_SPIN)"
        )
        self.onnx_input_names = [inp.name for inp in self.onnx_policy_session.get_inputs()]
        self.onnx_output_names = [out.name for out in self.onnx_policy_session.get_outputs()]

        # ONNX I/O-name adaptation. Two export formats reach this policy:
        #   * DEPLOY format (e.g. 2pt-hybrid 31wei7by): input "obs" + "time_step",
        #     outputs "actions" (+ aux joint_pos/joint_vel/ref_*). What the legacy
        #     rl_inference hardcoded.
        #   * BROWSER/mjswan format (e.g. 3pt-hybrid n4exfyvp): input "actor_obs"
        #     ONLY (no time_step), output "action" (singular, no aux).
        # Both carry the SAME baked metadata (action_scale/kp/kd/dof_names/urdf/
        # normalizer/experiment_config), so only the I/O NAMES differ. Resolve them
        # once here; rl_inference + the obs-dim readback consume the resolved names.
        # 2pt deploy stays byte-identical (it hits the "obs"/"time_step"/"actions"
        # branch exactly as before).
        self._onnx_obs_input, self._onnx_uses_time_step, self._onnx_action_output = resolve_onnx_io_names(
            self.onnx_input_names, self.onnx_output_names
        )
        action_meta = next(
            out for out in self.onnx_policy_session.get_outputs() if out.name == self._onnx_action_output
        )
        validate_onnx_action_shape(action_meta.shape, self.num_dofs)
        logger.info(
            f"ONNX I/O: obs input={self._onnx_obs_input!r}, time_step={self._onnx_uses_time_step}, "
            f"action output={self._onnx_action_output!r}"
        )

        metadata = dict(self.onnx_policy_session.get_modelmeta().custom_metadata_map)

        # Read body_name_ref early (before _load_external_motion) so the motion
        # loader can locate the training ref body (torso_link vs pelvis) in
        # the NPZ body_names list.
        _exp_config = json.loads(metadata.get("experiment_config", "{}"))
        _motion_cfg = (
            _exp_config.get("command", {})
            .get("setup_terms", {})
            .get("motion_command", {})
            .get("params", {})
            .get("motion_config", {})
        )
        self._training_body_name_ref = _motion_cfg.get("body_name_ref", [None])[0]
        self._ref_body_is_pelvis = self._training_body_name_ref in ("pelvis", None)
        logger.info(
            f"Training body_name_ref: {self._training_body_name_ref} (use_pelvis_shortcut={self._ref_body_is_pelvis})"
        )

        # Parse action_scale (per-joint, JSON list) — same order as SDK
        action_scale_raw = json.loads(metadata["action_scale"])
        self._action_scale = parse_action_scale(action_scale_raw, self.num_dofs)
        # Terrain preset trains with JointPositionTargetActionTerm (implicit PD):
        # q_target = default + action (no scaling). The stored metadata value is
        # a safety cap, not an applied scale. Allow the task config to force-1.0.
        if getattr(self.config.task, "force_unit_action_scale", False):
            logger.info("force_unit_action_scale=True → overriding action_scale to 1.0")
            self._action_scale = np.ones(self.num_dofs, dtype=np.float32)

        # Setup Pinocchio for FK
        robot_urdf = metadata.get("robot_urdf", "")
        if robot_urdf:
            try:
                robot_urdf = json.loads(robot_urdf)
            except json.JSONDecodeError:
                pass
        if not robot_urdf:
            urdf_path = getattr(self.config.task, "robot_urdf_path", "")
            if urdf_path and Path(urdf_path).is_file():
                robot_urdf = Path(urdf_path).read_text()
        if robot_urdf:
            self.pinocchio_robot = PinocchioRobot(self.config.robot, robot_urdf)
            self._robot_urdf_text = robot_urdf

        # Detect obs mode from terms
        self._obs_terms = list(self.config.observation.obs_dict.get("actor_obs", []))
        self._obs_mode = detect_obs_mode(self._obs_terms)
        self._is_3pt = self._obs_mode in ("3pt", "3pt_global")
        self._is_2pt = self._obs_mode == "2pt"
        self._is_5pt = self._obs_mode == "5pt"
        # 6D-orientation VR rep: the whole 2pt family trains rot6d, and a 3pt preset
        # opts in via the explicit term name `vr_3point_orn_6d`
        # (exp:g1-wbt-3pt-hybrid-global-vel: 3 bodies × 6 = 18D). _compute_vr_fk and
        # the live-VIVE orientation path branch on this, NOT on _is_2pt.
        self._vr_orn_6d = self._is_2pt or any("vr_3point_orn_6d" in t for t in self._obs_terms)
        # TCP-UNIFIED hand point (exp:g1-wbt-3pt-hybrid-unified[-anneal] retrained
        # 2026-08-03 onward, run wda3h4ll): both hand tracking points sit on the 0.242 m
        # fingertip TCP — the SAME anchor the force stack uses — instead of the default
        # 0.18 m VR-tracker lever. Keyed on the obs term NAME so the geometry is decided
        # by the CHECKPOINT, never by a preset flag or an env var: a pre-2026-08-03
        # unified checkpoint carries `vr_3point_pos` and keeps 0.18, so the two
        # generations cannot be cross-deployed against the wrong FK. `detect_obs_mode`
        # matches the `vr_3point` substring, so the mode is still "3pt" either way.
        # `vr_5point_pos_tcp` is the only 5pt spelling that exists (no lever twin was ever
        # trained), so 5pt is unconditionally on the TCP.
        self._vr_pos_tcp = any(("vr_3point_pos_tcp" in t or "vr_5point_pos_tcp" in t) for t in self._obs_terms)
        # C1 UNIFIED family detection. `contact_phase` appears in exactly the four
        # unified presets (local 170D + global-vel 185D, R2 and R1) and nowhere else,
        # so its presence IS the marker — no preset name reaches the policy object.
        self._resolve_unified_mode()
        # DENSE-UNIFIED (exp:g1-wbt-hybrid-unified-anneal): the dense obs mode PLUS the
        # unified marker _resolve_unified_mode just set. BOTH halves are required —
        # `contact_phase` alone also matches the keypoint unified presets, and dense alone
        # matches the LEGACY dense/global deploy presets, whose behaviour must not change
        # (they gain neither the ready-pose hold nor the live joint override).
        self._is_dense_unified = (
            self._obs_mode == "dense" and self._has_contact_phase_term and any(t == "command" for t in self._obs_terms)
        )
        # BASE whole-body tracker under LIVE teleop (exp:g1-wbt, preset
        # g1-29dof-holosoma-wbt, W&B run ruo0o6fr). Steered exactly like dense-unified —
        # its entire tracking front-end IS the 58D `command` term, which is what the
        # bridge's `--mode dense` wire carries — but it has NO force block, so
        # `contact_phase` is absent and the check above cannot see it.
        #
        # THE THIRD CONJUNCT IS THE LOAD-BEARING ONE. `_vive_live is not None` means a
        # bridge URL was actually configured (HOLOSOMA_VIVE_WS_URL), i.e. this run IS a
        # teleop run. Without it, widening the family here would ALSO have switched on the
        # init ready-pose hold for every CLIP-REPLAY deploy of this same preset — the
        # verified path — turning "replay the motion" into "hold a standing pose". Gating on
        # the live source keeps replay byte-identical while making teleop possible, and it
        # is why `g1-29dof-bm-wbt` (the other bare-`command` dense preset) is unaffected
        # unless someone deliberately points a bridge at it.
        self._is_dense_base_teleop = (
            self._obs_mode == "dense"
            and not self._has_contact_phase_term
            and any(t == "command" for t in self._obs_terms)
            and self._vive_live is not None
        )
        if self._is_dense_base_teleop:
            logger.info(
                colored(
                    "BASE tracker (exp:g1-wbt) LIVE TELEOP: the 58D `command` term is steered by the "
                    "completion's 29 joint targets — the SAME tracking front-end, wire and code path as "
                    "dense-unified-anneal, whose obs is simply this one plus the 26D force block "
                    "(157+26=183). The only real difference: no force block here, so every F^cmd / servo "
                    "/ contact-phase knob is INERT. The clip still supplies motion_ref_ori_b, so HEADING "
                    "and world position stay pinned to it — trackers steer POSTURE only, and that limit is "
                    "SHARED by 5pt and dense (none of the three carries motion_ref_pos_b).",
                    "cyan",
                )
            )
        self._height_scan_dim = self.config.observation.obs_dims.get("height_scan", 0)
        self._compliance_vec = np.asarray(getattr(self.config.task, "compliance", DEFAULT_COMPLIANCE), dtype=np.float32)
        logger.info(f"Obs mode: {self._obs_mode}")

        # LIVE keyboard F^cmd control (opt-in): only meaningful when the actor obs
        # actually carries a force_cmd_local term (the 2pt-force / 2pt-hybrid
        # presets). Build the controller + seed _force_cmd_pelvis_raw so nudges
        # have a (2,3) buffer to fold into (starts at 0 N unless HOLOSOMA_FORCE_CMD_LOCAL
        # pre-set a launch value). Headless eval never has a TTY, so start() is a
        # no-op there.
        self._setup_force_cmd_keys()
        if self._action_ema_alpha < 1.0:
            _ema_msg = (
                f"Action EMA low-pass ON: alpha={self._action_ema_alpha} "
                f"(a_t = {self._action_ema_alpha}*a_raw + {1.0 - self._action_ema_alpha:.3g}*a_prev) — sim2real anti-chatter"
            )
            if self._action_ema_alpha < 0.5:
                # Danger band: significant phase lag for a balance policy trained
                # with no filter. Warn LOUD so the operator knows they are below the
                # cautious 0.7-0.9 band.
                logger.warning(
                    colored(
                        _ema_msg + " — ⚠️ alpha<0.5 is AGGRESSIVE (lag → balance/fall risk); cautious band is 0.7-0.9",
                        "red",
                    )
                )
            else:
                logger.info(colored(_ema_msg, "magenta"))

        # Setup VR FK body frame IDs
        vr_bodies = (
            _VR_5POINT_BODIES
            if self._is_5pt
            else _VR_3POINT_BODIES
            if self._is_3pt
            else _VR_2POINT_BODIES
            if self._is_2pt
            else None
        )
        if vr_bodies and hasattr(self, "pinocchio_robot"):
            label = "5pt" if self._is_5pt else "3pt" if self._is_3pt else "2pt"
            model = self.pinocchio_robot.robot_model
            for body_name in vr_bodies:
                fid = model.getFrameId(body_name)
                if fid >= model.nframes:
                    logger.warning(f"{label} body '{body_name}' not found in URDF, FK will be inaccurate")
                self._vr_frame_ids.append(fid)
            logger.info(f"{label} FK: body frame IDs {dict(zip(vr_bodies, self._vr_frame_ids))}")
            if self._is_5pt and self._vive_live is not None:
                if getattr(self, "_vive_ref_joints", True):
                    logger.info(
                        colored(
                            "5pt: a wire frame's OPTIONAL `ref_joints` (29 reference joint targets + "
                            "velocities BY NAME) overrides command_lower_body while ARMED — the walk "
                            "skill's gait legs under the gait feet. Absent = the frozen clip, byte-identical. "
                            "Reported on RecordTap as `ref_joints_live`. (HOLOSOMA_VIVE_REF_JOINTS=0 ignores it.)",
                            "cyan",
                        )
                    )
                else:
                    logger.warning(
                        colored(
                            "5pt: HOLOSOMA_VIVE_REF_JOINTS=0 — a wire frame's `ref_joints` is IGNORED; "
                            "command_lower_body stays the frozen clip's legs during a walk skill (the "
                            "in-place-stepping configuration).",
                            "yellow",
                        )
                    )
            if self._vr_pos_tcp:
                logger.info(
                    colored(
                        f"{label} hand tracking point: TCP-UNIFIED — both hands at wrist_yaw +x "
                        f"{_VR_3POINT_OFFSETS_TCP[0][0]:.3f} m (the force TCP), NOT the "
                        f"{_VR_3POINT_OFFSETS[0][0]:.2f} m VR lever. Position and force share one point.",
                        "cyan",
                    )
                )
            if self._is_5pt:
                logger.info(
                    colored(
                        "5pt tracking: the two FOOT points are the bare *_ankle_roll_link origins "
                        "(no lever) — a hand offset is a physical mounting lever and the head offset "
                        "lifts torso_link to HMD height, but no such lever exists for a foot.",
                        "cyan",
                    )
                )

        # Obs dimensions
        obs_dims = self.config.observation.obs_dims
        obs_dim_per_frame = sum(obs_dims.get(t, 0) for t in self._obs_terms)
        obs_dim_with_history = obs_dim_per_frame * self._history_length

        # Verify against ONNX input shape (use the resolved obs input name, so both
        # the deploy-format "obs" and the browser-format "actor_obs" are matched).
        onnx_obs_dim = None
        for inp in self.onnx_policy_session.get_inputs():
            if inp.name == self._onnx_obs_input:
                shape = inp.shape
                if len(shape) >= 2 and isinstance(shape[-1], int):
                    onnx_obs_dim = shape[-1]
        self._obs_dim = onnx_obs_dim or obs_dim_with_history

        if self._obs_dim != obs_dim_with_history:
            raise ValueError(
                f"Obs dim mismatch: ONNX={self._obs_dim}, "
                f"config={obs_dim_per_frame}D x {self._history_length}hist = {obs_dim_with_history}D"
            )

        logger.info(f"HolosomaWBT: {obs_dim_per_frame}D/frame x {self._history_length} hist = {self._obs_dim}D")

        # Validate the deploy config's actor_obs term ORDER against the training obs
        # baked into the ONNX experiment_config. The width check above only pins the
        # TOTAL (obs_dim_with_history) — a term swap that preserves the total (e.g.
        # two equal-width terms transposed, or the un-zero-padded actions-at-slot-2
        # bug) would slip through and silently feed the ONNX a scrambled vector. This
        # asserts the per-term sequence matches training (obsNN_ prefixes stripped,
        # dof<->joint aliased). Skips gracefully when the metadata lacks the structure.
        self._validate_obs_term_order(_exp_config)

        # Init history buffers
        for term in self._obs_terms:
            self._obs_history[term] = deque(maxlen=self._history_length)

        # Load external reference motion (required)
        ref_motion_path = getattr(self.config.task, "ref_motion_path", "")
        if not ref_motion_path:
            raise ValueError("ref_motion_path is required — external NPZ motion must be provided")
        self._load_external_motion(ref_motion_path)

        # Optionally prepend default-pose transition (for models trained with
        # default_pose_transition_strategy='interpolation')
        if getattr(self.config.task, "prepend_default_pose_transition", False):
            duration = getattr(self.config.task, "prepend_duration_s", 2.0)
            self._prepend_default_pose_transition(duration_s=duration)

        # Initial motion reference from external NPZ
        joint_pos_0, joint_vel_0, ref_pos_0, ref_quat_xyzw_0, ref_vel_0 = self._get_ext_motion_at_frame(0)
        joint_pos_0 = joint_pos_0.reshape(1, -1)
        joint_vel_0 = joint_vel_0.reshape(1, -1)
        self._motion_command_t = np.concatenate([joint_pos_0, joint_vel_0], axis=1)
        self._motion_command_0 = self._motion_command_t.copy()
        self._ref_quat_xyzw_t = ref_quat_xyzw_0.reshape(1, 4).astype(np.float32)
        self._ref_quat_xyzw_0 = self._ref_quat_xyzw_t.copy()
        self._ref_pos_xyz_t = ref_pos_0.reshape(1, 3).astype(np.float32)
        self._ref_pos_xyz_0 = self._ref_pos_xyz_t.copy()
        self._ref_vel_xyz_t = ref_vel_0.reshape(1, 3).astype(np.float32)
        self._ref_vel_xyz_0 = self._ref_vel_xyz_t.copy()

        joint_pos_0 = self._motion_command_t[0, :29]

        # Init target — same joint order as SDK, no reordering needed
        self._init_q_target = joint_pos_0.reshape(-1)

        # Reference heading offset
        ref_orn_wxyz = xyzw_to_wxyz(self._ref_quat_xyzw_0)
        _, _, ref_yaw = quat_to_rpy(ref_orn_wxyz.reshape(-1))
        ref_heading = rpy_to_quat((0.0, 0.0, ref_yaw)).reshape(1, 4)
        self._ref_init_heading_inv_wxyz = quat_inverse(ref_heading)

        # Read default_dof_angles from ONNX experiment_config (training standing pose).
        # Previously hardcoded to zeros, but some models train with non-zero defaults
        # (e.g. bent-knee standing pose). This offset is used in both:
        #   - Observation: joint_pos = dof_pos - default_dof_angles
        #   - Action:      q_target = action * scale + default_dof_angles
        # Reading from ONNX metadata ensures sim2real consistency regardless of
        # what default pose the model was trained with.
        exp_config = _exp_config
        default_angles_dict = exp_config.get("robot", {}).get("init_state", {}).get("default_joint_angles", {})
        dof_names = json.loads(metadata.get("dof_names", "[]"))
        if default_angles_dict and dof_names:
            self.default_dof_angles = np.array(
                [default_angles_dict.get(name, 0.0) for name in dof_names], dtype=np.float64
            )
            logger.info(f"default_dof_angles from ONNX (non-zero: {np.count_nonzero(self.default_dof_angles)})")
        else:
            self.default_dof_angles = np.array(self.robot_config.stiff_startup_pos, dtype=np.float64)
            logger.warning("No default_joint_angles in ONNX metadata, using stiff_startup_pos")
        self.default_dof_angles = np.asarray(self.default_dof_angles, dtype=np.float64).reshape(-1)
        if self.default_dof_angles.size != self.num_dofs or not np.all(np.isfinite(self.default_dof_angles)):
            raise ValueError(f"default_dof_angles must contain {self.num_dofs} finite values")

        # INIT / held pose = the forearms-HORIZONTAL ready pose (default_dof_angles
        # with the 14 arm joints = 0), NOT the NPZ clip's frame-0. See
        # _maybe_apply_init_ready_pose. Auto-on for the real-robot path; off for
        # headless eval (which plays the clip from frame-0, so its video stays
        # byte-identical). Leaves the NPZ pristine (select+A plays the real clip).
        self._maybe_apply_init_ready_pose()

        # Load normalizer for observation pre-clipping (matches sim2sim eval).
        self._norm_mean = None
        self._norm_std_eps = None
        self._load_normalizer(model_path)

        # KP/KD from ONNX metadata — must match training gains for correct
        # sim2real transfer.  The shared robot config has TML/BM-tuned values
        # which override ONNX in the base class resolver; force ONNX values here.
        self.onnx_kp = np.array(json.loads(metadata["kp"])) if "kp" in metadata else None
        self.onnx_kd = np.array(json.loads(metadata["kd"])) if "kd" in metadata else None
        if self.onnx_kp is not None and self.onnx_kd is not None:
            from dataclasses import replace as _replace

            self.robot_config = _replace(
                self.robot_config, motor_kp=tuple(self.onnx_kp.tolist()), motor_kd=tuple(self.onnx_kd.tolist())
            )
            self.interface.update_config(self.robot_config)
            logger.info(colored("HolosomaWBT: using ONNX KP/KD (training gains) instead of config override", "green"))

        # Compatibility fields for parent class
        self.motion_command_t = self._motion_command_t.copy()
        self.motion_command_0 = self._motion_command_0.copy()
        self.ref_quat_xyzw_t = self._ref_quat_xyzw_t.copy()
        self.ref_quat_xyzw_0 = self._ref_quat_xyzw_0.copy()
        self.per_joint_policy_action_scale = self._action_scale.reshape(1, -1)

        def policy_act(_input_feed):
            raise RuntimeError("HolosomaWBT uses custom rl_inference.")

        self.policy = policy_act

    # CANONICAL obs term name → a single label BOTH the deploy config and the
    # training config map onto, so a benign naming difference is not flagged:
    #   * dof_{pos,vel} (training manager name) == joint_{pos,vel} (deploy/SDK name)
    #   * motion_command / command / tml_command all denote the motion-command term
    #     (the runtime treats name=="command" and "motion_command" in name alike).
    # Any name not listed canonicalizes to itself.
    _OBS_TERM_CANON = {
        "dof_pos": "joint_pos",
        "dof_vel": "joint_vel",
        "joint_pos": "joint_pos",
        "joint_vel": "joint_vel",
        "motion_command": "command",
        "tml_command": "command",
        "command": "command",
    }

    @classmethod
    def _canon_obs_term(cls, name: str) -> str:
        """Strip a leading zero-padded ``obsNN_`` prefix, then map to the canonical
        term label (``_OBS_TERM_CANON``). Applied to BOTH the deploy term names and
        the training keys so the comparison is label-agnostic. ``obs`` with a
        non-numeric tail (e.g. ``observation_x``) or no ``_`` is left untouched."""
        if name[:3] == "obs" and "_" in name:
            head, _, rest = name.partition("_")
            if head[3:].isdigit():
                name = rest
        return cls._OBS_TERM_CANON.get(name, name)

    def _validate_obs_term_order(self, exp_config: dict) -> None:
        """Cross-check the deploy actor_obs terms against the training obs baked into
        the ONNX ``experiment_config``, raising ``ValueError`` on a genuine mismatch.

        WHY (beyond the width check): the total-width assert only catches a wrong
        *sum*. A term transposition that preserves the total — e.g. two equal-width
        terms swapped, or a missing/extra term offset by another — feeds the ONNX a
        scrambled vector that still passes the width gate and then makes the policy
        "flail and fall". This pins the per-term identity.

        The obs manager concatenates in ``sorted(keys())`` (lexical) order (holosoma
        manager.py). When the training keys are ZERO-PADDED (``obs00_..obs12_``),
        lexical order == the intended sequence, so we assert the full ORDER
        positionally (strongest check — catches a swap). When the keys are bare or
        single-digit (lexical order is ambiguous vs the author's intended sequence,
        e.g. ``obs10_`` sorts before ``obs2_``), we DROP to a SET check (every term
        present exactly once, order NOT asserted) so a legitimate legacy preset is
        never falsely rejected. Both sides are canonicalized (``_canon_obs_term``:
        obsNN_ stripped + dof/joint + command aliases) so a naming difference never
        trips it.

        SKIPS gracefully (info log, no raise) when the metadata lacks the structure
        (older/TML/stripped exports) — the width check remains the backstop. Designed
        to NEVER false-positive a correct preset+checkpoint pairing: it raises only
        when a term is genuinely absent/duplicated (any key scheme) or transposed
        (zero-padded schemes).
        """
        try:
            groups = exp_config.get("observation", {}).get("groups", {})
            actor = groups.get("actor_obs", {})
            train_terms = actor.get("terms", {})
        except AttributeError:
            train_terms = {}
        if not isinstance(train_terms, dict) or not train_terms:
            logger.info("Obs term-order check skipped (no actor_obs terms in ONNX experiment_config).")
            return

        train_keys = list(train_terms.keys())
        deploy_canon = [self._canon_obs_term(t) for t in self._obs_terms]

        # Are the training keys safely zero-padded? (all obsNN_ with a pad width wide
        # enough that lexical sort == numeric order). Only then is asserting ORDER
        # free of false-positives; otherwise (bare or ambiguous single-digit keys)
        # we drop to a SET check so a legitimate legacy preset is never rejected.
        def _prefix_width(k: str) -> int:
            if k[:3] == "obs" and "_" in k:
                head = k.partition("_")[0]
                if head[3:].isdigit():
                    return len(head[3:])
            return -1  # not an obsNN_ key

        widths = {_prefix_width(k) for k in train_keys}
        # len(keys) < 10**min_width ⇒ no index needs more digits than the pad, so
        # lexical key order == numeric order. A -1 (any non-obsNN_ key) disqualifies.
        zero_padded = -1 not in widths and len(train_keys) < 10 ** min(widths)

        train_canon_sorted = [self._canon_obs_term(k) for k in sorted(train_keys)]

        if len(train_canon_sorted) != len(deploy_canon):
            raise ValueError(
                "Obs term-COUNT mismatch vs the ONNX experiment_config: "
                f"deploy has {len(deploy_canon)} terms {deploy_canon}, "
                f"training baked {len(train_canon_sorted)} terms {train_canon_sorted}. "
                "The inference preset does not match this checkpoint."
            )

        if zero_padded:
            mismatches = [
                f"  slot {i}: deploy={self._obs_terms[i]!r} (->{dep!r}) but checkpoint->{tr!r}"
                for i, (dep, tr) in enumerate(zip(deploy_canon, train_canon_sorted))
                if dep != tr
            ]
            if mismatches:
                raise ValueError(
                    "Obs term-ORDER mismatch vs the ONNX experiment_config (the total width "
                    "matched but the per-term sequence does NOT - the policy would receive a "
                    "scrambled observation and fall). Fix the inference preset's actor_obs list "
                    "to match the checkpoint:\n" + "\n".join(mismatches)
                )
            logger.info(
                colored(f"Obs term-order validated against the checkpoint ({len(deploy_canon)} terms match).", "green")
            )
        else:
            # Ambiguous key ordering -> SET check only (presence, not sequence).
            dep_set, tr_set = sorted(deploy_canon), sorted(train_canon_sorted)
            if dep_set != tr_set:
                raise ValueError(
                    "Obs term-SET mismatch vs the ONNX experiment_config: the deploy preset's "
                    f"terms {dep_set} differ from the checkpoint's {tr_set} (a term is missing, "
                    "extra, or renamed). The inference preset does not match this checkpoint."
                )
            logger.info(
                colored(
                    f"Obs term-SET validated against the checkpoint ({len(deploy_canon)} terms; "
                    "order not asserted - training keys not zero-padded).",
                    "green",
                )
            )

    # Arm joints (SDK order) whose held/INIT angle the READY POSE zeroes so the
    # forearms sit HORIZONTAL-forward (the g1_29dof_yam URDF arm-zero pose). The
    # legs + waist keep the trained crouch (default_dof_angles) untouched.
    _READY_POSE_ARM_JOINTS = (
        "left_shoulder_pitch_joint",
        "left_shoulder_roll_joint",
        "left_shoulder_yaw_joint",
        "left_elbow_joint",
        "left_wrist_roll_joint",
        "left_wrist_pitch_joint",
        "left_wrist_yaw_joint",
        "right_shoulder_pitch_joint",
        "right_shoulder_roll_joint",
        "right_shoulder_yaw_joint",
        "right_elbow_joint",
        "right_wrist_roll_joint",
        "right_wrist_pitch_joint",
        "right_wrist_yaw_joint",
    )

    def _maybe_apply_init_ready_pose(self):
        """Make INIT + the frozen 2pt/hybrid HOLD target a forearms-HORIZONTAL
        *ready pose* — the trained ``default_dof_angles`` (legs/waist crouch kept)
        but with all 14 ARM joints forced to 0 — NOT the NPZ clip's frame-0.

        WHY: the deploy path ramped INIT to the NPZ clip's frame-0
        (``_init_q_target = joint_pos_0``) and, because the frozen hold pins
        ``_ext_frame_idx=0``, the held ``vr_2point`` hand target + ``command_lower_body``
        were ALSO frame-0. For a grasp clip (e.g. ``big_heavy_two_hands_hold``)
        frame-0 is a contorted pose far from anything natural for teleop, so on
        hardware the robot INIT'd into an off-distribution pose. The operator's
        chosen INIT/hold stance is instead **forearms horizontal-forward** (upper
        arm hanging, elbow ~straight), which the g1_29dof_yam URDF gives at
        arm-joints-all-zero (FK: forearm elevation ≈ −3°). The ONNX
        ``default_joint_angles`` itself is NOT this pose — its baked arm angles
        (shoulder_pitch 0.2 / roll ±0.2 / elbow 0.6) point the forearms ~47° DOWN.

        Two targets are set (both DOF-only; the NPZ is left PRISTINE):
          * ``_init_q_target`` — the INIT ramp target (``get_init_target``).
          * ``_hold_dof_override`` — substituted for the frozen-hold reference DOF
            (``_motion_command_t[0, :29]``) each tick WHILE ``motion_clip_progressing``
            is False, so the held ``vr_2point`` FK + ``command_lower_body`` read the
            ready pose. Applied in ``rl_inference`` AFTER the per-tick
            ``_get_ext_motion_at_frame`` refresh; the underlying ``_ext_motion`` /
            ``_motion_command_0`` are never mutated, so pressing ``select+A`` plays
            the real clip from its true frame-0 with no 1-tick reference jump
            (unlike the earlier in-place frame-0 overwrite).

        DEFAULT: ON for the real-robot / interactive path, OFF for headless eval
        (which plays the clip from frame-0, so its metrics + video stay
        byte-identical — ``_hold_dof_override`` stays None and the substitution
        branch is never taken). Override with ``HOLOSOMA_INIT_READY_POSE`` =
        ``1/on`` to force on (e.g. an interactive-sim dry run) or ``0/off`` to
        force the legacy frame-0 behavior.
        """
        self._hold_dof_override = None  # sentinel: no frozen-hold DOF substitution
        # 2pt/3pt/5pt + dense-UNIFIED only: the ready pose relies on the frozen-hold
        # keypoint + command_lower_body semantics (the hold reads _motion_command_t[0,:29]
        # every tick). The 3pt-hybrid preset (exp:g1-wbt-3pt-hybrid) uses the SAME
        # command_lower_body + vr_3point structure as the 2pt family (training
        # hybrid_3pt.py: obs00_command_lower_body, no motion_command term), and 5pt widens
        # only the keypoint count — so the ready-pose hold applies identically to all three.
        # (Force/servo terms are zero at the frozen hold either way.)
        #
        # DENSE-UNIFIED is included even though its `command` term is the full 58D DOF
        # pos+vel: `_apply_hold_dof_override` writes BOTH halves (pos = ready pose, vel =
        # 0), so the pos/vel pair it presents is self-consistent. That was not true of the
        # original position-only override, which is what the "dense would be inconsistent"
        # exclusion was written against. LEGACY dense/global presets stay excluded — they
        # are not part of this teleop and their deploy behaviour must not change. The BASE
        # tracker is in `_dense_joint_teleop` only when a live bridge is configured, so its
        # CLIP-REPLAY deploy stays in the excluded set and keeps replaying the motion.
        if not (self._is_2pt or self._is_3pt or self._is_5pt or self._dense_joint_teleop):
            return
        env = os.environ.get("HOLOSOMA_INIT_READY_POSE", "").strip().lower()
        headless = bool(getattr(self.config.task, "headless_sim_eval", False))
        if env in ("0", "false", "no", "off"):
            enabled = False
        elif env in ("1", "true", "yes", "on"):
            enabled = True
        else:
            enabled = not headless  # auto: on for real robot, off for headless eval
        if not enabled:
            return

        # Build the ready pose = default_dof_angles with the 14 arm joints zeroed,
        # resolving arm slots BY NAME (robust to any dof ordering).
        target = self.default_dof_angles.astype(np.float32).reshape(-1).copy()
        dof_names = list(getattr(self.config.robot, "dof_names", []))
        if len(dof_names) < 29:
            logger.warning(
                colored(
                    "HOLOSOMA_INIT_READY_POSE: robot dof_names < 29 — cannot resolve arm "
                    "joints; leaving INIT/hold at default_dof_angles.",
                    "red",
                )
            )
            self._init_q_target = target.copy()
            self._hold_dof_override = target.copy()
            return
        try:
            arm_idx = [dof_names.index(n) for n in self._READY_POSE_ARM_JOINTS]
        except ValueError as exc:
            logger.warning(colored(f"HOLOSOMA_INIT_READY_POSE: arm joint not found in dof_names ({exc}); skip.", "red"))
            return
        target[arm_idx] = 0.0

        self._init_q_target = target.copy()
        self._hold_dof_override = target.copy()
        logger.info(
            colored(
                "INIT + frozen hold pinned to the READY POSE (forearms horizontal: "
                "default_dof_angles with the 14 arm joints = 0) — not the NPZ clip's "
                "frame-0. select+A still plays the real clip. "
                "(HOLOSOMA_INIT_READY_POSE=0 to restore frame-0.)",
                "cyan",
            )
        )

    def _apply_hold_dof_override(self):
        """Substitute the ready-pose DOF for the reference joint_pos WHILE the clip
        is frozen (STAND/VIVE hold, not playing).

        No-op (byte-identical) when ``_hold_dof_override`` is None (headless / env
        disabled) or when the clip is playing (``motion_clip_progressing``). Rewrites
        the 29 DOF POSITION half of ``_motion_command_t`` to the ready pose AND zeros
        its 29 DOF VELOCITY half: a static hold must command zero reference velocity,
        or ``command_lower_body`` would pair the static ready position with the clip
        frame-0's (often nonzero, e.g. ~2 rad/s for a grasp clip) reference velocity —
        an internally inconsistent, off-distribution pos/vel command. Byte-identical
        for the idle clip (frame-0 vel already 0). The root pose and the underlying
        ``_ext_motion`` / ``_motion_command_0`` are left pristine, so a later
        ``select+A`` plays the real clip from its true frame-0.
        """
        if self._hold_dof_override is None or self.motion_clip_progressing:
            return
        self._motion_command_t[0, :29] = self._hold_dof_override
        self._motion_command_t[0, 29:] = 0.0  # static hold → zero reference velocity

    def _load_normalizer(self, model_path: str):
        """Load normalizer mean/std for observation pre-clipping.

        Uses shared :func:`load_normalizer_from_path` (same code as sim2sim).
        """
        self._norm_mean, self._norm_std_eps = load_normalizer_from_path(model_path)
        if self._norm_mean is not None:
            logger.info(f"Loaded normalizer (shape={self._norm_mean.shape})")
        else:
            logger.info("No normalizer found — skipping observation pre-clipping")

    def _save_debug_log(self):
        """Save debug observation log to NPZ for sim2real comparison.

        Called on policy stop. Output NPZ contains arrays keyed by:
        obs_raw, obs_clipped, action, dof_pos, timestep, and term_<name>
        for each obs term. Compare against sim2sim debug log
        to diagnose sim2real discrepancies frame-by-frame.
        """
        if self._debug_log is None or not self._debug_log_path:
            return
        save_dict = {}
        for k, v in self._debug_log.items():
            if v:
                save_dict[k] = np.array(v)
        if save_dict:
            np.savez(self._debug_log_path, **save_dict)
            logger.info(
                colored(
                    f"Debug log saved: {self._debug_log_path} ({len(save_dict.get('obs_raw', []))} frames)", "magenta"
                )
            )
        self._debug_log = None

    def _clip_normalized_obs(self, obs: np.ndarray) -> np.ndarray:
        """Pre-clip obs so the ONNX baked-in normalizer sees values in [-3, 3]."""
        if self._norm_mean is None:
            return obs
        return _shared_clip_normalized_obs(obs, self._norm_mean, self._norm_std_eps)

    def _load_external_motion(self, npz_path: str):
        """Load an external NPZ reference motion.

        NPZ fields: joint_pos [T,36], joint_vel [T,35],
                     body_pos_w [T,N_bodies,3], body_quat_w [T,N_bodies,4] (wxyz),
                     body_names (optional, used to locate the training ref body).
        """
        data = np.load(npz_path)
        self._ext_motion = {
            "joint_pos": data["joint_pos"].astype(np.float32),
            "joint_vel": data["joint_vel"].astype(np.float32),
            "body_pos_w": data["body_pos_w"].astype(np.float32),
            "body_quat_w": data["body_quat_w"].astype(np.float32),
        }
        if "body_lin_vel_w" in data:
            self._ext_motion["body_lin_vel_w"] = data["body_lin_vel_w"].astype(np.float32)
        # Locate the training body_name_ref (torso_link for terrain preset) in
        # the motion NPZ so motion_ref_pos_b / motion_ref_ori_b pull the
        # anchor from the correct body rather than pelvis (body 0).
        self._ext_ref_body_idx = 0
        if "body_names" in data.files:
            motion_body_names = [str(n) for n in data["body_names"].tolist()]
            ref_name = getattr(self, "_training_body_name_ref", None) or "pelvis"
            if ref_name in motion_body_names:
                self._ext_ref_body_idx = motion_body_names.index(ref_name)
                logger.info(f"Motion ref body: {ref_name} at body index {self._ext_ref_body_idx}")
            else:
                logger.warning(
                    f"Training body_name_ref={ref_name!r} not in motion body_names; "
                    f"falling back to body 0 (pelvis). motion_ref_pos_b/ori_b may drift."
                )
        self._ext_motion_length = self._ext_motion["joint_pos"].shape[0]
        self._ext_frame_idx = 0
        logger.info(f"External motion loaded: {self._ext_motion_length} frames from {npz_path}")

    def _prepend_default_pose_transition(self, duration_s: float = 2.0):
        """Prepend interpolated frames from default standing pose to motion start.

        Creates a smooth transition trajectory matching what training produces with
        default_pose_transition_strategy='interpolation'. After prepend, frame 0 of
        the extended motion is the default standing pose, and the original motion
        starts at frame num_prepend_frames.

        Args:
            duration_s: Duration of the transition in seconds. Should match
                default_pose_prepend_duration_s used during training.
        """
        fps = getattr(self.config.task, "rl_rate", 50)
        num_frames = int(duration_s * fps)
        if num_frames <= 0:
            return

        motion = self._ext_motion

        # Frame 0 of the original motion
        jp0 = motion["joint_pos"][0]
        jv0 = motion["joint_vel"][0]
        bp0 = motion["body_pos_w"][0]  # (N_bodies, 3)
        bq0 = motion["body_quat_w"][0]  # (N_bodies, 4) wxyz

        # Default joint positions — matches training's default_dof_angles
        default_jp = np.zeros_like(jp0)
        if jp0.shape[0] > 29:
            # Has root prefix: root_pos(3) + root_quat(4) + joints(29)
            default_jp[7:] = self.default_dof_angles.astype(np.float32)
            default_jp[:3] = bp0[0, :3]  # root pos from frame 0
            default_jp[3:7] = bq0[0]  # root quat from frame 0
        else:
            default_jp[:] = self.default_dof_angles.astype(np.float32)

        alphas = np.linspace(0.0, 1.0, num_frames, endpoint=False, dtype=np.float32)[:, np.newaxis]
        prepend_jp = (1 - alphas) * default_jp + alphas * jp0
        prepend_jv = np.zeros((num_frames, jv0.shape[0]), dtype=np.float32)
        prepend_bp = np.tile(bp0[np.newaxis], (num_frames, 1, 1))
        prepend_bq = np.tile(bq0[np.newaxis], (num_frames, 1, 1))

        motion["joint_pos"] = np.concatenate([prepend_jp, motion["joint_pos"]], axis=0)
        motion["joint_vel"] = np.concatenate([prepend_jv, motion["joint_vel"]], axis=0)
        motion["body_pos_w"] = np.concatenate([prepend_bp, motion["body_pos_w"]], axis=0)
        motion["body_quat_w"] = np.concatenate([prepend_bq, motion["body_quat_w"]], axis=0)
        if "body_lin_vel_w" in motion:
            prepend_bv = np.zeros((num_frames, motion["body_lin_vel_w"].shape[1], 3), dtype=np.float32)
            motion["body_lin_vel_w"] = np.concatenate([prepend_bv, motion["body_lin_vel_w"]], axis=0)

        self._num_prepend_frames = num_frames
        self._ext_motion_length = motion["joint_pos"].shape[0]
        logger.info(
            colored(
                f"Prepended {num_frames} default-pose transition frames ({duration_s}s at {fps}Hz) to motion", "cyan"
            )
        )

    def _get_ext_motion_at_frame(self, frame_idx: int):
        """Get motion reference data from external NPZ at given frame.

        Returns (joint_pos, joint_vel, ref_pos, ref_quat_xyzw, ref_vel).
        Pose/velocity are for the training ref body (``body_name_ref``,
        typically ``torso_link``) so ``motion_ref_pos_b``/``motion_ref_ori_b``
        match training's torso-anchor obs. Falls back to pelvis (body 0) only
        if the motion NPZ has no ``body_names`` or the ref body isn't found.
        """
        idx = min(frame_idx, self._ext_motion_length - 1)
        raw_pos = self._ext_motion["joint_pos"][idx]
        raw_vel = self._ext_motion["joint_vel"][idx]
        joint_pos = raw_pos[7:] if raw_pos.shape[0] > 29 else raw_pos
        joint_vel = raw_vel[6:] if raw_vel.shape[0] > 29 else raw_vel
        body_idx = getattr(self, "_ext_ref_body_idx", 0)
        ref_pos = self._ext_motion["body_pos_w"][idx, body_idx]
        ref_quat_wxyz = self._ext_motion["body_quat_w"][idx, body_idx]
        ref_quat_xyzw = wxyz_to_xyzw(ref_quat_wxyz.reshape(1, 4)).flatten()
        # Ref-body velocity: prefer body_lin_vel_w at the same index, else finite-diff.
        if "body_lin_vel_w" in self._ext_motion:
            ref_vel = self._ext_motion["body_lin_vel_w"][idx, body_idx]
        else:
            dt = 1.0 / getattr(self.config.task, "rl_rate", 50)
            idx_next = min(idx + 1, self._ext_motion_length - 1)
            idx_prev = max(idx - 1, 0)
            if idx_next != idx_prev:
                ref_vel = (
                    self._ext_motion["body_pos_w"][idx_next, body_idx]
                    - self._ext_motion["body_pos_w"][idx_prev, body_idx]
                ) / ((idx_next - idx_prev) * dt)
            else:
                ref_vel = np.zeros(3, dtype=np.float32)
        return joint_pos, joint_vel, ref_pos, ref_quat_xyzw, ref_vel

    def preload_motion_frame(self, frame_idx: int) -> None:
        """Prime the motion-command buffers at ``frame_idx`` for the headless runner.

        Training advances the motion one step before computing obs, so frame 0
        of the policy sees motion frame 1. The headless loop calls this once
        before the first inference to match; the internal cursor is then
        seeded at ``frame_idx + 1`` so ``policy_action`` picks up where the
        preload left off.
        """
        jp, jv, rp, rq, rv = self._get_ext_motion_at_frame(frame_idx)
        self._motion_command_t = np.concatenate([jp.reshape(1, -1), jv.reshape(1, -1)], axis=1)
        self._ref_quat_xyzw_t = rq.reshape(1, 4).astype(np.float32)
        self._ref_pos_xyz_t = rp.reshape(1, 3).astype(np.float32)
        self._ref_vel_xyz_t = rv.reshape(1, 3).astype(np.float32)
        self.motion_command_t = self._motion_command_t.copy()
        self.ref_quat_xyzw_t = self._ref_quat_xyzw_t.copy()
        self._ext_frame_idx = frame_idx + 1

    def _compute_motion_alignment(self, robot_state_data: np.ndarray):
        """Compute transform from NPZ frame to mocap world frame.

        Uses shared :func:`compute_motion_alignment` (same code as maskedmimic).
        """
        robot_pos = robot_state_data[0, 0:3].copy()
        robot_quat_wxyz = robot_state_data[0, 3:7]
        ref_pos_0 = self._ext_motion["body_pos_w"][0, 0]
        ref_quat_wxyz_0 = self._ext_motion["body_quat_w"][0, 0]

        self._align_rot_mat, self._align_pos_offset, yaw_offset = _shared_compute_motion_alignment(
            robot_pos, robot_quat_wxyz, ref_pos_0, ref_quat_wxyz_0
        )

        logger.info(f"Motion alignment: yaw_offset={np.degrees(yaw_offset):.1f}° pos_offset={self._align_pos_offset}")

    def _get_aligned_ref_pos(self) -> np.ndarray:
        """Get reference root position aligned to mocap frame."""
        ref_pos = self._ref_pos_xyz_t.reshape(3)
        if self._align_rot_mat is not None:
            return (self._align_rot_mat @ ref_pos + self._align_pos_offset).astype(np.float32)
        return ref_pos

    def _get_aligned_ref_vel(self) -> np.ndarray:
        """Get reference root velocity rotated to mocap frame."""
        ref_vel = self._ref_vel_xyz_t.reshape(3)
        if self._align_rot_mat is not None:
            return (self._align_rot_mat @ ref_vel).astype(np.float32)
        return ref_vel

    @property
    def _uses_vr_keypoints(self) -> bool:
        """True for every KEYPOINT tracking family (2pt / 3pt / 5pt): the ones whose obs
        carries a ``vr_*point_pos`` + ``vr_*point_orn`` pair built by Pinocchio FK and
        overridable from the live bridge. False for dense/global (no keypoints) — the
        dense-unified teleop overrides the 58D ``command`` term instead.

        Exists so the live-override path is gated on the STRUCTURE it needs rather than on
        an enumeration that has to be widened at every site each time a point count is
        added (which is how 5pt silently fell through to the dense path)."""
        return self._is_2pt or self._is_3pt or self._is_5pt

    @property
    def _dense_joint_teleop(self) -> bool:
        """True for every preset steered in JOINT space by the bridge's dense wire: the C1
        dense-unified policy (`exp:g1-wbt-hybrid-unified-anneal`) and the BASE whole-body
        tracker (`exp:g1-wbt`) once a live bridge is configured.

        The joint-space twin of :attr:`_uses_vr_keypoints`, and it exists for the same
        reason: every teleop site — the ready-pose hold, the arm/disarm toggle, the clutch
        anchor, the pre-arm gate's units, the `command` override — needs "is this steered by
        29 joint targets?", NOT "is this the unified force policy?". Conflating the two is
        what kept the base tracker un-teleoperable while all the machinery it needs was
        already present and tested.

        Note the asymmetry: `_is_dense_unified` is a pure property of the OBS, but
        `_is_dense_base_teleop` also requires a live source, so that clip replay of the base
        preset keeps its exact pre-teleop behaviour. See its assignment in setup_policy."""
        return self._is_dense_unified or self._is_dense_base_teleop

    @property
    def _vr_orn_is_quat(self) -> bool:
        """True when the tracked-body orientation obs is a raw xyzw QUATERNION per body
        (3pt = 3×4, 5pt = 5×4), False when it is the rot6d rep (the whole 2pt family and
        the 6D-orn 3pt preset exp:g1-wbt-3pt-hybrid-global-vel).

        The double-cover guard, the clutch composition and the manifold reprojection all
        branch on the REP, not the point count, so they read this."""
        return self._is_5pt or (self._is_3pt and not self._vr_orn_6d)

    def _vr_offsets(self) -> np.ndarray:
        """The body-local FK offsets for the ACTIVE tracking geometry.

        The single decision point for "where is the tracked point on the hand", so the
        obs target, the pre-arm pose gate and the absolute arm clutch can never disagree.
        Inlining the choice at each call site is what made this worth a helper: the three
        callers must agree by construction, and a 4th caller added later gets it free.

        3pt: the 0.242 m fingertip TCP when the checkpoint's obs carries
        ``vr_3point_pos_tcp`` (the TCP-unified retrain — position and force on ONE point),
        else the historical 0.18 m VR-tracker lever. 2pt: always the 0.18 m lever (no TCP
        variant was trained for that family). 5pt: always the TCP triple + two ZERO foot
        offsets (likewise no lever twin). Callers must already be 2pt, 3pt or 5pt.
        """
        if self._is_5pt:
            return _VR_5POINT_OFFSETS_TCP
        if self._is_3pt:
            return _VR_3POINT_OFFSETS_TCP if self._vr_pos_tcp else _VR_3POINT_OFFSETS
        return _VR_2POINT_OFFSETS

    def _compute_vr_fk(self, ref_joint_pos, offsets):
        """Run Pinocchio FK on reference joint positions to get N-point VR body poses.

        Returns (vr_pos_local [N*3], vr_orn_local) in the anchor body local frame.
        The orientation output is the continuous 6D rotation rep ([N*6], first two
        columns of the rotation matrix) when ``self._vr_orn_6d`` is set — the 2pt
        family (`vr_2point_local_orn_target`) AND the 6D-orn 3pt preset
        exp:g1-wbt-3pt-hybrid-global-vel (`vr_3point_local_orn_target_6d`, term
        `vr_3point_orn_6d`, 3×6=18D) — else the raw xyzw quaternion ([N*4]) for the
        plain 3pt family (`vr_3point_local_orn_target`).
        The 6D rep avoids the quaternion double-cover: rebuilding the wrist quat
        here via Eigen `pin.Quaternion(R).coeffs()` picks an inconsistent sign per
        frame (q ~ -q), an antipode a quaternion-trained policy never saw (OOD).
        """
        import pinocchio as pin

        model = self.pinocchio_robot.robot_model
        data = self.pinocchio_robot.robot_data

        ref_quat_xyzw = self._ref_quat_xyzw_t.reshape(4)
        ref_pos = self._ref_pos_xyz_t.reshape(3)
        joint_pos_pin = ref_joint_pos[self.pinocchio_robot.real2pinocchio_index]
        q = np.concatenate([ref_pos, ref_quat_xyzw, joint_pos_pin])

        pin.framesForwardKinematics(model, data, q)

        anchor_pos = ref_pos.copy()
        anchor_quat_wxyz = xyzw_to_wxyz(ref_quat_xyzw.reshape(1, 4))

        vr_pos_local_list = []
        vr_orn_local_list = []
        for i, fid in enumerate(self._vr_frame_ids):
            frame_pose = data.oMf[fid]
            body_pos_w = frame_pose.translation.astype(np.float32)
            body_quat_xyzw = pin.Quaternion(frame_pose.rotation).coeffs().astype(np.float32)
            body_quat_wxyz = xyzw_to_wxyz(body_quat_xyzw.reshape(1, 4))

            offset = offsets[i].reshape(1, 3)
            offset_world = quat_rotate_inverse(quat_inverse(body_quat_wxyz), offset).reshape(3)
            vr_pos_w = body_pos_w + offset_world

            diff = (vr_pos_w - anchor_pos).reshape(1, 3)
            local_pos = quat_rotate_inverse(anchor_quat_wxyz, diff).reshape(3)
            vr_pos_local_list.append(local_pos)

            local_quat_wxyz = quat_mul(quat_inverse(anchor_quat_wxyz), body_quat_wxyz)
            if self._vr_orn_6d:
                # 6D rotation rep = first two columns of the rotation matrix.
                # matrix_from_quat expects WXYZ — feed local_quat_wxyz (NOT the
                # xyzw form). [..., :2].reshape(-1) -> [r00,r01,r10,r11,r20,r21],
                # bit-for-bit identical to training's
                # quaternion_to_matrix(q, w_last=True)[..., :2].reshape and to
                # the baseline motion_ref_ori_b path (see line ~686). Used by the
                # 2pt family (vr_2point_orn) AND the 3pt 6D preset (vr_3point_orn_6d,
                # exp:g1-wbt-3pt-hybrid-global-vel) → 3 bodies × 6 = 18D there.
                rot6d = matrix_from_quat(local_quat_wxyz)[..., :2].reshape(-1).astype(np.float32)
                vr_orn_local_list.append(rot6d)
            else:
                local_quat_xyzw = wxyz_to_xyzw(local_quat_wxyz).reshape(4)
                # CANONICALIZE THE DOUBLE COVER TO w >= 0 — the same fix the bake applies
                # (bake_motions._fk_into), for the same reason, on the rig side. `q` and
                # `-q` are one rotation but two very different input vectors, and the
                # quaternion-orn presets trained ONLY on the +w branch (na475b60's
                # normalizer: `vr_5point_orn` w mean +0.73 hands / +0.98 head / +0.96 feet,
                # std <= 0.17 — unimodal, never near zero).
                #
                # Neither input here pins the branch: `anchor_quat_wxyz` is the reference
                # clip's stored representative and `body_quat_xyzw` is Eigen's
                # `pin.Quaternion(R).coeffs()`, which the docstring above already notes
                # "picks an inconsistent sign per frame". When the reference pose sits near
                # the 180° seam the product lands on -identity for every body at once, which
                # is a ~20-sigma obs and shows up as the arms snapping to a nonsense pose the
                # instant a clip is selected. Measured offline on the two MC takes: 100% of
                # frames on the wrong branch.
                #
                # No-op whenever the data is already +w, so nothing that worked can change.
                if local_quat_xyzw[3] < 0.0:
                    local_quat_xyzw = -local_quat_xyzw
                vr_orn_local_list.append(local_quat_xyzw)

        vr_pos_local = np.concatenate(vr_pos_local_list).astype(np.float32)
        vr_orn_local = np.concatenate(vr_orn_local_list).astype(np.float32)
        return vr_pos_local, vr_orn_local

    def _force_cmd_local_obs(self, dof_pos: np.ndarray) -> np.ndarray:
        """Build the 6D force_cmd_local obs term (already ×0.04-scaled).

        Zero when no HOLOSOMA_FORCE_CMD_LOCAL override is set. Otherwise returns
        the commanded force in the frame the trained policy expects:

        * pelvis/root frame (default) — the pre-scaled override as-is.
        * each hand's (wrist_yaw) frame, when HOLOSOMA_FORCE_CMD_HAND_FRAME=1
          (a --F_on_hand_frame checkpoint). Training fed the actor
          ``R(q_wrist_yaw_world)^-1 @ R(q_pelvis_world) @ F_pelvis``. The pelvis IS
          the articulation root, so the world root orientation cancels and only
          the live wrist-relative-to-pelvis rotation ``R_rel`` remains. We obtain
          ``R_rel`` from Pinocchio FK on the ACTUAL robot joints with an identity
          freeflyer root (``oMf[wrist].rotation``), then ``F_hand = R_rel^-1 @ F_pelvis``.
          Per-step so the hand-frame components track the live wrist, exactly like
          the time-variant training obs. F^cmd=0 ⇒ 0 (frame-invariant).
        * each {side}_ft_sensor_link frame, when HOLOSOMA_FORCE_CMD_SENSOR_FRAME=1
          (a --F_on_sensor_frame checkpoint). Same FK-derived ``R_rel`` as the hand
          frame, then post-rotated by the CONSTANT wrist→sensor rotation ``R_sensor``
          (ft_sensor_joint rpy="0 -pi/2 pi", == HandForceConfig.sensor_rpy):
          ``F_sensor = R_sensor^T @ (R_rel^-1 @ F_pelvis)``. Using the constant (not a
          live ft_sensor URDF frame) reproduces training bit-for-bit — training also
          composes the constant with the live wrist quat, since ft_sensor is fused in
          sim. F^cmd=0 ⇒ 0 (frame-invariant).
        """
        if self._force_cmd_pelvis_raw is None:
            return np.zeros(6, dtype=np.float32)
        if not (self._force_cmd_hand_frame or self._force_cmd_sensor_frame):
            # Pelvis-frame: the pre-scaled override (unchanged legacy behavior).
            return self._force_cmd_local_scaled.copy()

        # Hand/sensor-frame: rotate the pelvis-frame command into each wrist frame
        # using FK on the actual joints (identity root → R_rel = wrist-rel-to-pelvis).
        _env = "HOLOSOMA_FORCE_CMD_SENSOR_FRAME" if self._force_cmd_sensor_frame else "HOLOSOMA_FORCE_CMD_HAND_FRAME"
        if not hasattr(self, "pinocchio_robot"):
            logger.warning(f"{_env} set but no Pinocchio model; using pelvis frame.")
            return self._force_cmd_local_scaled.copy()
        import pinocchio as pin

        model = self.pinocchio_robot.robot_model
        data = self.pinocchio_robot.robot_data
        if self._force_wrist_frame_ids is None:
            self._force_wrist_frame_ids = []
            for n in _VR_2POINT_BODIES:
                fid = model.getFrameId(n)
                # Pinocchio returns nframes (an out-of-range sentinel) when the
                # name is absent; oMf[that] would silently give a wrong rotation.
                # Warn symmetrically with the VR-FK setup guard above.
                if fid >= model.nframes:
                    logger.warning(f"{_env} wrist body '{n}' not found in URDF; R_rel will be inaccurate")
                self._force_wrist_frame_ids.append(fid)
        # q = [root_pos(3), root_quat_xyzw(4)=identity, joint_pos_pin]. Identity
        # root makes oMf[wrist].rotation the wrist orientation IN the pelvis frame.
        joint_pos_pin = dof_pos[self.pinocchio_robot.real2pinocchio_index]
        q = np.concatenate([np.zeros(3), np.array([0.0, 0.0, 0.0, 1.0]), joint_pos_pin])
        pin.framesForwardKinematics(model, data, q)

        # Constant wrist→sensor rotation (only used in sensor-frame mode).
        r_sensor = self._rot_matrix_from_rpy(self._force_sensor_rpy) if self._force_cmd_sensor_frame else None
        out = np.empty((2, 3), dtype=np.float32)
        for hi, fid in enumerate(self._force_wrist_frame_ids):
            R_rel = np.asarray(data.oMf[fid].rotation)  # wrist rotation in pelvis frame
            f_hand = R_rel.T @ self._force_cmd_pelvis_raw[hi]  # R_rel^-1 @ F_pelvis (world→hand)
            out[hi] = (r_sensor.T @ f_hand) if r_sensor is not None else f_hand  # world→sensor = R_sensor^T @ (…)
        return (out.reshape(-1) * self._force_cmd_obs_scale).astype(np.float32)

    def _resolve_unified_mode(self) -> bool:
        """Detect the C1 UNIFIED preset family and switch the servo to OBSERVE-ONLY.

        The unified presets (`exp:g1-wbt-3pt-hybrid-unified[-anneal]`; the `-global-vel`
        unified twins use the same mechanism but their inference preset has not landed
        on this branch) are the ONLY presets whose actor obs carries a ``contact_phase``
        term, so its presence is an exact, name-free marker — which matters because no
        preset NAME reaches this object. (`wbt_realtime_eval/main.py` exists on this branch
        but carries NO unified rule; the browser's name-based override lives on
        `origin/hybrid_control`, and the in-tree mjswan runtime reads
        `servo_observe_only` straight from the checkpoint — the very source this deploy
        refuses to trust for R1. So do NOT treat the browser as cross-validation of
        this decision; it is a separate implementation with a different key.)

        Two consequences, both load-bearing:

        * **Δq_arm is never injected.** ``HOLOSOMA_HYBRID_SERVO`` is forced OFF (and the
          operator told, loudly, if they asked for it). These checkpoints trained with
          the Jᵀ servo demoted to a pure reference generator — R2 bakes
          ``servo_observe_only=True``; R1 anneals the servo's authority to α=0 but bakes
          ``False`` and never serializes the annealed α — so keying off the CHECKPOINT
          would hand an R1 policy a second actuator it never trained alongside.
        * **The integrator keeps running.** ``integrator_state`` (I) and ``x_tgt_shift``
          (Δx) remain live actor inputs; this is deliberately NOT the ``-layerA``
          servo-off path, which would zero them and feed the policy two 6D channels of
          zeros it never saw in training.

        ⚠️ This removes the SERVO's authority, not the policy's. A force-qualified
        unified checkpoint is itself a force controller: ``x_tgt_shift`` = (F^cmd + I)/K
        is nonzero the moment a nonzero F^cmd exists, even in free space, and the policy
        chases that shifted setpoint. Such a checkpoint still requires two-run hardware
        staging (F^cmd all-zero first, then ramp). The pinned iss0qs71/model_27000 is NOT
        force-qualified, so its launcher disables the keys and keeps F^cmd at zero.

        Returns True when the unified mode was engaged. Fully idempotent: BOTH branches
        assign every flag they own. (`self.config.observation` is fixed for the process, so
        every checkpoint of a multi-checkpoint deploy resolves the same way today; the
        symmetric assignment is defensive, not a live requirement.)
        """
        self._has_contact_phase_term = any("contact_phase" in t for t in self._obs_terms)
        _servo_requested = bool(getattr(self, "_hybrid_servo_requested", getattr(self, "_hybrid_servo", False)))
        if not self._has_contact_phase_term:
            # Explicit (not merely "left alone"): see the idempotence note above.
            self._servo_observe_only = False
            self._hybrid_servo = _servo_requested
            self._emit_servo_authority_banner()
            return False
        self._servo_observe_only = True
        if _servo_requested:
            logger.warning(
                colored(
                    "UNIFIED preset (contact_phase in obs): HOLOSOMA_HYBRID_SERVO=1 is IGNORED. "
                    "These checkpoints trained with the Jᵀ servo as a pure REFERENCE GENERATOR "
                    "(observe-only / annealed to α=0) — the policy is the sole executor, so NO Δq "
                    "is injected. The integrator still winds: integrator_state/x_tgt_shift stay live. "
                    "⚠️ This does NOT make the run safe: the POLICY still pushes from F^cmd.",
                    "yellow",
                )
            )
        self._hybrid_servo = False
        logger.info(
            colored(
                "UNIFIED single-policy mode: observe-only integrator (I/Δx published to the actor, "
                "Δq NEVER injected) + live soft contact_phase "
                f"(θ_on={getattr(self, '_contact_phase_theta_on', _CONTACT_PHASE_THETA_ON):.1f}N / "
                f"θ_off={getattr(self, '_contact_phase_theta_off', _CONTACT_PHASE_THETA_OFF):.1f}N, "
                f"τ={getattr(self, '_contact_phase_tau_s', _CONTACT_PHASE_TAU_S) * 1e3:.0f}ms, "
                f"warmup={getattr(self, '_contact_phase_warmup_steps', _CONTACT_PHASE_WARMUP_STEPS)} ticks "
                "+ the whole auto-tare burst).",
                "magenta",
            )
        )
        # The tare burst is not an optimisation here — it is the window during which the
        # LATCHING contact label is suppressed. With it disabled the label goes live
        # ~100 ms after START, on a reading nobody has zeroed yet, and a residual bias
        # above θ_on sticks φ at ≈1 in free space until the reading falls below θ_off.
        # For a φ-gated single-policy force controller that means "execute the force
        # task" in mid-air, so say so loudly rather than letting TARE_N=0 silently
        # delete the protection.
        _burst_n = int(getattr(self, "_force_tare_on_start_n", 0))
        _burst_window = max(getattr(self, "_force_tare_on_start_s", 0.0), 0.0) + max(
            getattr(self, "_force_tare_settle_s", 0.0), 0.0
        )
        if getattr(self, "_force_live", None) is not None:
            if _burst_n <= 0 or _burst_window <= 0.0:
                logger.warning(
                    colored(
                        "UNIFIED preset with NO usable START auto-tare window "
                        f"(HOLOSOMA_FORCE_TARE_ON_START={_burst_n}, window={_burst_window:.1f}s): contact_phase and "
                        "the integrator go live after only "
                        f"{getattr(self, '_contact_phase_warmup_steps', _CONTACT_PHASE_WARMUP_STEPS)} ticks, on an "
                        "un-tared reading. A residual bias above θ_on will LATCH φ≈1 in free space and wind the "
                        "integrator. Re-enable the burst, or tare manually (R2 / T) in free space before "
                        "commanding anything.",
                        "red",
                    )
                )
            else:
                logger.info(
                    colored(
                        f"  φ + integrator SUPPRESSED for the first {_burst_window:.1f}s after START "
                        f"(tare burst {getattr(self, '_force_tare_on_start_s', 0.0):.1f}s + "
                        f"{getattr(self, '_force_tare_settle_s', 0.0):.1f}s settle) — the robot will not react to "
                        "contact during that window.",
                        "magenta",
                    )
                )
            if getattr(self, "_unified_integ_cmd_gate", True):
                logger.info(
                    colored(
                        "  Integrator COMMAND GATE ON (default): I/Δx only wind on a hand carrying a nonzero "
                        "F^cmd, so a drifted tare cannot ramp Δx in free space. F_meas + φ still reach the "
                        "policy either way. Set HOLOSOMA_UNIFIED_INTEGRATOR_CMD_GATE=0 for the fidelity A/B.",
                        "magenta",
                    )
                )
        return True

    def _emit_servo_authority_banner(self) -> None:
        """Emit the "the arms will ACTIVELY push" banner iff the Jᵀ servo really has
        actuation authority this run.

        Deferred out of ``__init__`` on purpose: `_hybrid_servo` is not final until
        ``_resolve_unified_mode`` has seen the obs terms, and a unified preset forces it
        off. Emitting from ``__init__`` made a `SERVO=1` unified run print a red
        "FORCE SERVO ON — the arms will ACTIVELY push" and then, later, "IGNORED".
        No-op without a live F/T source (the servo cannot run at all).
        """
        if (
            not getattr(self, "_hybrid_servo", False)
            or getattr(self, "_force_live", None) is None
            or getattr(self, "_servo_cfg", None) is None
        ):
            return
        logger.warning(
            colored(
                f"  ⚠️ Jᵀ FORCE SERVO ON (k={self._servo_cfg.servo_gain_k}, "
                f"Ki={self._servo_cfg.integrator_gain_ki}, dq_max={self._servo_cfg.servo_dq_max} rad) — the arms "
                "will ACTIVELY push from force error. Gantry + hardware E-stop mandatory.",
                "yellow",
            )
        )

    def _setup_force_cmd_keys(self) -> None:
        """Build + start the live keyboard F^cmd controller (opt-in).

        No-op unless HOLOSOMA_FORCE_CMD_KEYS is set AND the actor obs carries a
        ``force_cmd_local`` term (2pt-force / 2pt-hybrid). Seeds the applied buffer
        ``_force_cmd_pelvis_raw`` + its slew TARGET ``_force_cmd_target_raw`` to a
        zero (2,3) (or the launch value), so nudges have somewhere to accumulate;
        the pelvis-frame obs ``_force_cmd_local_scaled`` is kept in sync. Starting
        the shared keyboard listener is idempotent, so this coexists with a keyboard
        command channel and also works under the joystick channel (which would not
        otherwise start the listener). Headless eval has no TTY → start() False.

        IDEMPOTENT: on a multi-checkpoint deploy setup_policy runs once per model;
        re-subscribing would leak an orphan listener deque each time (the listener
        appends every keypress to all subscribers). Guard so we subscribe once.
        """
        if not self._force_cmd_keys_enabled:
            return
        if not any("force_cmd_local" in t for t in self._obs_terms):
            logger.warning(
                colored(
                    "HOLOSOMA_FORCE_CMD_KEYS set but the actor obs has no force_cmd_local term "
                    f"(obs mode {self._obs_mode!r}); live F^cmd keys disabled.",
                    "yellow",
                )
            )
            return
        if self._force_cmd_keys is not None:
            return  # already subscribed (idempotent across per-checkpoint setup_policy)
        from wbt_wrappers_inference.utils.force_cmd_keys import HELP, ForceCmdKeyController

        # Seed the applied buffer + slew target + pelvis-frame scaled obs so nudges
        # start from the launch value (or 0 N). _force_cmd_local_obs reads
        # _force_cmd_local_scaled on the (default) pelvis path, so keep it in sync.
        if self._force_cmd_pelvis_raw is None:
            self._force_cmd_pelvis_raw = np.zeros((2, 3), dtype=np.float32)
        self._force_cmd_target_raw = self._force_cmd_pelvis_raw.copy()
        if self._force_cmd_local_scaled is None:
            self._force_cmd_local_scaled = (self._force_cmd_pelvis_raw.reshape(-1) * self._force_cmd_obs_scale).astype(
                np.float32
            )
        self._force_cmd_keys = ForceCmdKeyController(
            step_n=self._force_cmd_key_step_n, clamp_n=self._force_cmd_key_clamp_n
        )
        active = self._force_cmd_keys.start()
        logger.info(
            colored(
                f"LIVE F^cmd keyboard control ENABLED (step={self._force_cmd_key_step_n} N, "
                f"clamp=±{self._force_cmd_key_clamp_n} N, slew={self._force_cmd_key_slew_n} N/tick, "
                f"keyboard {'live' if active else 'NO TTY'}). {HELP}",
                "magenta",
            )
        )

    def _drain_force_cmd_keys(self) -> None:
        """Discard any buffered keypresses + snap the slew target to the current
        applied F^cmd. Called on START (before ``use_policy_action`` flips true) so
        keys pressed while the robot is stopped/INITing — which would otherwise flush
        as one big step on the first live tick — do NOT accumulate into a startup
        transient. START-only is sufficient: ``_poll_force_cmd_keys`` never runs while
        stopped, and the next START drains before the policy resumes."""
        if self._force_cmd_keys is None:
            return
        # Drain-and-discard the listener deque (poll from the CURRENT applied value,
        # ignore the result) so the queue is empty and the target holds steady.
        self._force_cmd_keys.poll(self._force_cmd_pelvis_raw)
        if self._force_cmd_target_raw is not None and self._force_cmd_pelvis_raw is not None:
            self._force_cmd_target_raw = self._force_cmd_pelvis_raw.copy()

    def _poll_force_cmd_keys(self) -> None:
        """Fold pending live F^cmd keypresses into the TARGET, then SLEW the applied
        F^cmd toward it (≤ _force_cmd_key_slew_n N/axis/tick).

        No-op (byte-identical) when the controller is off. Slewing bounds the
        per-tick change so a burst of buffered keys can never step the obs / servo
        feed-forward in one tick (the deploy has no soft-stop). When the applied
        value moves we update ``_force_cmd_pelvis_raw`` (servo feed-forward + hand/
        sensor-frame obs) AND re-derive the pelvis-frame ``_force_cmd_local_scaled``
        (default obs path).
        """
        if self._force_cmd_keys is None:
            return
        target, changed, msgs = self._force_cmd_keys.poll(self._force_cmd_target_raw)
        self._force_cmd_target_raw = target.astype(np.float32)
        for m in msgs:
            self.logger.info(colored(m, "magenta"))
        # Slew the applied command toward the (possibly-updated) target.
        applied = np.asarray(self._force_cmd_pelvis_raw, dtype=np.float32)
        delta = self._force_cmd_target_raw - applied
        if not changed and not np.any(delta):
            return
        step = np.clip(delta, -self._force_cmd_key_slew_n, self._force_cmd_key_slew_n)
        self._force_cmd_pelvis_raw = (applied + step).astype(np.float32)
        self._force_cmd_local_scaled = (self._force_cmd_pelvis_raw.reshape(-1) * self._force_cmd_obs_scale).astype(
            np.float32
        )

    def _check_force_estop(self) -> float | None:
        """Return the tripping ‖F_meas‖ (N) if the hard F/T stop should fire, else None.

        Detection only — the caller stops the controller, so the one place that decides
        "we are stopping" stays in ``rl_inference`` next to the other fault exits.

        WHAT THE THRESHOLD IS COMPARED AGAINST — the bridge's published per-hand force,
        i.e. the PAYLOAD-COMPENSATED (RNEA dynamic by default, gravity-only under
        ``FT_ALGO=bota``), TARED, 12 Hz anti-aliased sensor-frame reading. NOT the raw Bota
        wrench: the tool's own weight is already removed upstream, so ~0 N in free space and
        the threshold is a threshold on real CONTACT. ``force_bridge`` publishes the truly-raw
        wrench under the SEPARATE ``left_raw``/``right_raw`` keys (``FT_EMIT_RAW=1``, recorder only), and
        ``ForceLiveSource`` never reads those. "Raw", where this file uses it of the stop,
        has always meant "before the DEPLOY-side pipeline" — never "uncompensated".

        FOUR deliberate choices, each the reason a naive version would misbehave:

        * **Straight from the sensor stream, per hand, before the deploy-side pipeline.**
          Magnitude is invariant under both the sensor→pelvis rotation and
          ``HOLOSOMA_FORCE_MEAS_SIGN``, so checking here costs nothing in fidelity and gains
          three things: it needs no Pinocchio FK (a missing URDF frame cannot disable the
          stop), it is immune to the F_meas EMA's lag (``HOLOSOMA_FORCE_MEAS_EMA_ALPHA`` < 1
          would otherwise delay a spike by the filter's time constant), and it cannot be
          hidden by the F_meas-on-F^cmd gate, which zeroes an uncommanded hand's reading for
          the obs and the servo. A hand with no F^cmd is exactly the hand a collision hits.
        * **Only while the policy is driving.** ``use_policy_action`` False means stiff
          hold already owns the joints, which IS the post-stop state; tripping there would
          re-run the stop handler every tick and spam its log.
        * **Not while the reading is untrusted.** The pre-tare bias on this rig is the same
          order as the threshold (the gripper's own weight under gravity-only
          compensation), so a trip during the START auto-tare burst would be a false
          positive on EVERY take — and a safety valve that always fires at startup is one
          the operator turns off. The window is the tare-trust DEADLINE
          (``_force_trust_deadline``, extended by the START burst and by a manual re-tare),
          read directly rather than via ``_contact_phase_warmup_active``: that helper also
          gates on ``_contact_phase_ticks``, which only advances once
          ``_update_contact_phase`` runs, so a rig with no Pinocchio frame would pin it at
          0 and suppress this stop FOREVER. A safety limit must not be disableable by an
          unrelated missing URDF frame. With the burst off (``TARE_N=0``, which the runtime
          already warns about) there is no trusted window to wait for and the stop is live
          from the first tick, on the operator's own un-tared reading.
        * **Latched.** One trip per START. Re-arms in ``_handle_start_policy``.

        A stale or absent F/T stream reads as zeros (``ForceLiveSource.latest``), so it
        cannot trip — the freshness of the sensor is a separate concern with its own gate.

        Reads its own state through ``getattr``, the convention the rest of this file uses
        for ``__init__`` attributes touched on the driving path: this runs FIRST in
        ``rl_inference``, so an ``AttributeError`` here would take down the 50 Hz control
        loop — strictly worse than a missed check. A safety valve must not be able to crash
        the thing it protects, and the ``object.__new__`` test stubs have no ``__init__``.
        Every real instance sets these unconditionally in ``__init__``, and the startup
        banner states ARMED or DISABLED in colour, so an accidentally-off stop is visible
        rather than assumed.
        """
        if getattr(self, "_force_estop_n", 0.0) <= 0.0 or getattr(self, "_force_estop_latched", False):
            return None
        if getattr(self, "_force_live", None) is None:
            return None
        if not getattr(self, "use_policy_action", False):
            return None
        if time.monotonic() < float(getattr(self, "_force_trust_deadline", 0.0) or 0.0):
            return None
        # Reads the sensor directly (not the cache), so it must apply the per-hand mask
        # itself. A DISABLED hand cannot stop the robot: on this rig the right-hand RNEA
        # wrench_sign is seeded/unvalidated, so a sign error there inverts gravity
        # compensation and the reading GROWS with orientation instead of staying near zero —
        # plausibly past the threshold with the arm merely held in position control. A sensor
        # nobody trusts enough to put in the control loop must not be able to halt the robot.
        _hands = getattr(self, "_force_hands_enabled", None)
        for hi, hand in enumerate(self._force_live.latest()):  # (3,) sensor-frame N per hand
            if _hands is not None and not bool(_hands[hi]):
                continue
            mag = float(np.linalg.norm(np.asarray(hand, dtype=np.float64)))
            if math.isfinite(mag) and mag > self._force_estop_n:
                return mag
        return None

    def _trip_force_estop(self, magnitude: float) -> None:
        """Latch the hard F/T stop and run the joystick-'B' stop handler.

        ``_handle_stop_policy`` is called rather than a bespoke shutdown so the resulting
        machine state is EXACTLY the one the operator's own B press produces — stiff hold
        at the current pose, VIVE/planner arming dropped, F^cmd zeroed, motion clip reset,
        debug log saved. One state to reason about, and the FUJI planner's
        "low level lost" watchdog (which detects B by the RecordTap stream stopping)
        detects this identically, with no new wire field.

        COLOUR — bold bright WHITE, not red, and this is load-bearing: loguru's CRITICAL
        level colour is ``<RED><bold>``, and in loguru markup an UPPERCASE colour is a
        BACKGROUND. The default format wraps the body in ``<level>{message}</level>``, so a
        termcolor red FOREGROUND on top of it emits ``\\x1b[41m\\x1b[1m\\x1b[31m`` — red on
        red, i.e. the one line the operator must read is the one they cannot. White on that
        red background is legible in any terminal theme and keeps the block unmissable among
        this file's many red/yellow/magenta FOREGROUND messages. Same reasoning applies to
        the two ``logger.critical`` calls in ``_disable_policy_with_stiff_hold``, one of
        which fires on this very path immediately after this one.
        """
        self._force_estop_latched = True
        logger.critical(
            colored(
                f"🛑 F/T HARD STOP: ‖F_meas‖={magnitude:.1f}N > {self._force_estop_n:.0f}N (the bridge's "
                "payload-compensated + tared per-hand force — NOT a raw wrench) — an impact or a runaway, not "
                "the task. STOPPING the controller (same as joystick B): stiff hold at the current pose, "
                "teleop/planner DISARMED, F^cmd zeroed. Clear the obstruction, re-home (Y), then START (A) to "
                "re-arm. The hardware E-STOP is still the only true abort.",
                "white",
                attrs=["bold"],
            )
        )
        self._handle_stop_policy()

    def _poll_force_cmd_wire(self, dof_pos: np.ndarray) -> None:
        """Fold the HL planner's live WIRE F^cmd (raw Bota SENSOR frame, N) into the
        applied F^cmd, rotating sensor→pelvis and slewing at the same 2 N/tick cap the
        keyboard path uses. No-op (byte-identical) unless HOLOSOMA_FORCE_CMD_WIRE is on.

        The planner already gated its output (below its threshold it publishes 0), and
        ViveLiveSource.latest_force returns None when no f_cmd has arrived or it is stale
        — either yields a ZERO target (F^cmd decays to 0 = pure position). An optional
        low-level deadband (_force_cmd_wire_gate_n) hard-zeros a per-hand command below
        threshold BEFORE it enters the buffer, so a planner bug can't push below the gate.

        Frame: the wire carries the command in the SENSOR frame (the model's native force
        frame, same as F_meas). We rotate it to pelvis with the rig's OWN
        sensor_force_to_pelvis + live wrist FK — the planner cannot do this (it has no
        joint readback), which is exactly why the command travels sensor-frame. Torque is
        not carried (the buffer + Jᵀ servo are force-only)."""
        if not self._force_cmd_wire_enabled or self._vive_live is None:
            return
        # Lazily seed the buffers (setup_force_cmd_keys is skipped when keys are disabled).
        # _force_cmd_local_scaled MUST be seeded too: on the default (pelvis) obs path
        # _force_cmd_local_obs reads it every tick, and a first wire tick with zero force
        # returns early below without ever setting it → None.copy() crash otherwise.
        if self._force_cmd_pelvis_raw is None:
            self._force_cmd_pelvis_raw = np.zeros((2, 3), dtype=np.float32)
        if self._force_cmd_target_raw is None:
            self._force_cmd_target_raw = self._force_cmd_pelvis_raw.copy()
        if self._force_cmd_local_scaled is None:
            self._force_cmd_local_scaled = (
                self._force_cmd_pelvis_raw.reshape(-1) * self._force_cmd_obs_scale
            ).astype(np.float32)

        wire_f = self._vive_live.latest_force()  # (3,)=LEFT or (6,)=[L,R] sensor-frame N, or None
        target = np.zeros((2, 3), dtype=np.float32)  # absent/stale ⇒ command 0 (pure position)
        if wire_f is not None:
            f = np.asarray(wire_f, dtype=np.float64).reshape(-1)
            per_hand = [f[0:3]] if f.shape[0] == 3 else [f[0:3], f[3:6]]
            r_rels = self._wrist_R_rel(dof_pos)  # sensor→pelvis needs the live wrist FK
            if r_rels is not None:
                from wbt_wrappers_inference.utils.force_deploy_math import sensor_force_to_pelvis

                _hands = getattr(self, "_force_hands_enabled", None)
                for hi, f_sensor in enumerate(per_hand):
                    if hi >= 2:
                        break
                    # A hand whose sensor is not in the loop takes no force command either:
                    # F^cmd only means anything against the F_meas it is servoed toward, and
                    # that reading is zeroed for a disabled hand. Today the planner publishes
                    # a 3-wide (LEFT-only) f_cmd so `per_hand` has one entry, but this makes
                    # the invariant hold for a future 6-wide publisher too.
                    if _hands is not None and not bool(_hands[hi]):
                        continue
                    # Low-level deadband on the COMMAND magnitude (sensor frame; rotation
                    # preserves magnitude): below the gate ⇒ hard zero ⇒ pure position.
                    if float(np.linalg.norm(f_sensor)) <= self._force_cmd_wire_gate_n:
                        continue
                    target[hi] = sensor_force_to_pelvis(
                        f_sensor, r_rels[hi], self._force_meas_sign
                    ).astype(np.float32)
        # Clamp the target to the same magnitude bound the keyboard path uses, then SLEW
        # the applied F^cmd toward it (never an un-slewed jump — the deploy has no soft stop).
        self._force_cmd_target_raw = np.clip(
            target, -self._force_cmd_key_clamp_n, self._force_cmd_key_clamp_n
        ).astype(np.float32)
        applied = np.asarray(self._force_cmd_pelvis_raw, dtype=np.float32)
        delta = self._force_cmd_target_raw - applied
        if not np.any(delta):
            return
        step = np.clip(delta, -self._force_cmd_key_slew_n, self._force_cmd_key_slew_n)
        self._force_cmd_pelvis_raw = (applied + step).astype(np.float32)
        self._force_cmd_local_scaled = (self._force_cmd_pelvis_raw.reshape(-1) * self._force_cmd_obs_scale).astype(
            np.float32
        )

    def _wrist_R_rel(self, dof_pos: np.ndarray) -> list[np.ndarray] | None:
        """Per-hand wrist_yaw rotation IN the pelvis frame (R_rel), from identity-root
        Pinocchio FK on the MEASURED joints. Returns ``[R_left, R_right]`` (3×3 each)
        or ``None`` if no Pinocchio model. Shared by the F/T obs + the servo Jacobian.

        Identity freeflyer root makes ``oMf[wrist].rotation`` the wrist orientation
        relative to the pelvis (the world root orientation cancels), exactly the
        ``R_rel`` the F^cmd hand-frame path uses — so the measured force rotates into
        the SAME pelvis frame the policy's F^cmd is authored in.
        """
        if not hasattr(self, "pinocchio_robot"):
            return None
        import pinocchio as pin

        model = self.pinocchio_robot.robot_model
        data = self.pinocchio_robot.robot_data
        if self._force_wrist_frame_ids is None:
            self._force_wrist_frame_ids = []
            for n in _VR_2POINT_BODIES:
                fid = model.getFrameId(n)
                if fid >= model.nframes:
                    logger.warning(f"F/T deploy: wrist body '{n}' not found in URDF; R_rel will be inaccurate")
                self._force_wrist_frame_ids.append(fid)
        joint_pos_pin = dof_pos[self.pinocchio_robot.real2pinocchio_index]
        q = np.concatenate([np.zeros(3), np.array([0.0, 0.0, 0.0, 1.0]), joint_pos_pin])
        pin.framesForwardKinematics(model, data, q)
        return [np.asarray(data.oMf[fid].rotation, dtype=np.float64) for fid in self._force_wrist_frame_ids]

    def _update_force_meas_and_servo(self, dof_pos: np.ndarray) -> None:
        """Read the live F/T, rotate sensor→pelvis, and (if servo on) run the Jᵀ
        force servo — caching the pelvis-frame F_meas, integrator I, setpoint shift
        Δx, and per-DOF servo Δq for this tick. No-op (all zero) when no live source.

        Called once per obs build; the obs terms + rl_inference read the caches, so
        the sensor is read and the FK/Jacobian computed exactly once per control tick.
        """
        # Reset the per-tick caches (so a stale/absent source ⇒ zero force obs + no Δq).
        self._force_meas_pelvis[:] = 0.0
        self._servo_dq[:] = 0.0
        if self._force_live is None:
            # No live F/T at all: the soft contact label must read FREE SPACE (its
            # training value at reset), not hold a stale LPF tail. Cheap, and it keeps
            # the headless/NPZ-replay path all-zeros for the unified presets too.
            self._reset_contact_phase_state()
            return
        from wbt_wrappers_inference.utils.force_deploy_math import sensor_force_to_pelvis

        left_s, right_s = self._force_live.latest()  # sensor-frame N (zeros when stale)
        f_sensor = [np.asarray(left_s, dtype=np.float64), np.asarray(right_s, dtype=np.float64)]
        # PER-HAND SENSOR MASK, applied HERE and nowhere else. This is the single point the
        # F/T is read, so zeroing a disabled hand's triple here makes it invisible to
        # everything downstream in one step — the EMA, the F^cmd gate, force_meas_local,
        # contact_phase, the Jᵀ servo, the runaway valve and the hard stop all read from
        # what this loop produces. Scattering the mask into each consumer instead is how one
        # of them keeps a live channel nobody remembers (the F/T-gate review already found
        # exactly that shape of bug in the runaway valve).
        _hands = getattr(self, "_force_hands_enabled", None)
        if _hands is not None:
            for hi in range(2):
                if not bool(_hands[hi]):
                    f_sensor[hi] = np.zeros(3, dtype=np.float64)
        r_rels = self._wrist_R_rel(dof_pos)
        if r_rels is None:
            # No Pinocchio model → cannot rotate; leave the force obs zero. Reset the
            # contact label too, for the same reason as the `_force_live is None` branch
            # above: F_meas is being reported as 0, so a held φ≈1 would be the one
            # pairing (φ high, F_meas identically 0 forever) that never occurs in
            # training. Static condition in practice — but keeping the two "F_meas is
            # zero by construction" exits symmetric costs nothing.
            self._reset_contact_phase_state()
            return
        for hi in range(2):
            self._force_meas_pelvis[hi] = sensor_force_to_pelvis(f_sensor[hi], r_rels[hi], self._force_meas_sign)
        # Belt-and-suspenders: a non-finite reading (NaN/inf) must never reach the
        # obs / servo / motors — np.clip passes NaN through. ForceLiveSource already
        # drops non-finite frames, but zero any that slip through (→ no contact).
        if not np.all(np.isfinite(self._force_meas_pelvis)):
            self._force_meas_pelvis[~np.isfinite(self._force_meas_pelvis)] = 0.0

        # F_meas low-pass (HOLOSOMA_FORCE_MEAS_EMA_ALPHA<1). Applied HERE, before the
        # runaway/obs/servo reads, so all three consume the same smoothed pelvis force.
        # Seeded to the first reading (no ramp-from-zero transient at contact onset).
        # A stale/absent source already zeroed _force_meas_pelvis above; we still filter
        # that zero so the trace decays smoothly to "no contact" instead of snapping.
        # getattr default 1.0 (OFF) matches the use_policy_action idiom below: robust
        # if an object.__new__ stub bypassed __init__ (the force-deploy tests do this).
        _fema_alpha = getattr(self, "_force_meas_ema_alpha", 1.0)
        if _fema_alpha < 1.0:
            _prev = getattr(self, "_force_meas_ema_state", None)
            if _prev is None:
                self._force_meas_ema_state = self._force_meas_pelvis.copy()
            else:
                self._force_meas_ema_state = _fema_alpha * self._force_meas_pelvis + (1.0 - _fema_alpha) * _prev
            self._force_meas_pelvis[:] = self._force_meas_ema_state

        # Snapshot the UNGATED pelvis force BEFORE the gate below zeroes it, so the
        # runaway safety valve always sees the TRUE reading. The gate is meant to
        # remove F_meas from the obs + servo error only — NOT to blind the safety
        # check: a large real/phantom force on an UNCOMMANDED hand (its position layer
        # still driving it into a hard contact, or a bad-tare/wrong-sign spike) must
        # still trip the runaway warning + freeze even though that hand carries no
        # F^cmd. Without this snapshot the gate would zero _force_meas_pelvis[hi]
        # first, so the runaway loop below would read 0 and never fire — silently
        # deleting the operator's "hit the hardware E-STOP" alert whenever the gate is
        # on (found in adversarial review). No-op when the gate is off (same array).
        f_meas_ungated = self._force_meas_pelvis.copy()
        # Persist the ungated reading for the opt-in RECORD tap (read at end of
        # rl_inference). Pure assignment of an already-computed copy — no behavior
        # change whether or not recording is on.
        self._force_meas_ungated_last = f_meas_ungated

        # F_meas GATE ON F^cmd (opt-in): zero the SHARED pelvis-force cache for any hand
        # whose applied F^cmd is ~0, so a no-command hand feeds ZERO measured force to the
        # obs (force_meas_local / force_ext_local) AND the servo below — i.e. it acts as a
        # pure POSITION controller and never chases F/T-sensor noise. Placed AFTER the EMA
        # (so the obs + servo read the SAME gated value) and BEFORE the servo; the runaway
        # valve deliberately reads the UNGATED snapshot above, not this gated cache. Reads
        # the APPLIED (slewed) F^cmd _force_cmd_pelvis_raw, which _poll_force_cmd_keys
        # refreshed this same tick (top of rl_inference) — so the gate opens the instant a
        # key nudge makes ‖F^cmd[hand]‖ exceed the tiny threshold. `c`/`p` zero the
        # target (selected/both), then the gate closes once the applied command slews to zero;
        # `l` only changes the selected hand and does not close anything.
        # Magnitude is frame-invariant (pelvis == the norm of any rotated frame), so the
        # same threshold is correct for the hand/sensor-frame checkpoints. getattr default
        # False keeps the object.__new__ test stubs byte-identical (they bypass __init__).
        # The EMA state above is NOT gated — it keeps tracking the true reading, so the EMA
        # itself needs no re-seed on re-open. ⚠️ But the OBS is a hard gate: at a gate
        # open/close transition the force_meas_local / force_ext_local obs STEPS by
        # 0.04·‖ΔF‖ in one tick (largest when toggling F^cmd while already in contact — the
        # gated hand is at 0 and snaps to the full reading). That step is inherent to a
        # per-hand gate; toggle F^cmd in FREE SPACE (near-zero step) rather than mid-contact.
        if getattr(self, "_force_meas_gate_on_cmd", False):
            f_cmd_raw = self._force_cmd_pelvis_raw
            gate_n = getattr(self, "_force_meas_gate_cmd_n", 1e-6)
            for hi in range(2):
                cmd_mag = float(np.linalg.norm(f_cmd_raw[hi])) if f_cmd_raw is not None else 0.0
                if cmd_mag <= gate_n:
                    self._force_meas_pelvis[hi] = 0.0
                    # Make the position-only invariant immediate for the CURRENT frame.
                    # The normal servo path would clear these via contact_gate=0 below,
                    # but observe-only runaway handling returns early while deliberately
                    # holding prior state. Clear an uncommanded hand here so a noisy
                    # runaway alarm cannot preserve a stale force target behind a gated
                    # F_meas=0 observation.
                    self._servo_integ[hi] = 0.0
                    self._servo_delta_x[hi] = 0.0

        # Soft contact-phase label (C1 unified presets). Driven by the FINAL pelvis
        # F_meas — post-EMA and post-gate — i.e. the exact same numbers the actor's
        # force_meas_local slot carries this tick. That pairing matters: with the
        # gate on, a gated hand reads F_meas=0, and a phase label computed from the
        # UNGATED reading would hand the policy "φ=1 with F_meas=0", a combination
        # that never occurs in training. No-op (all-zero) for non-unified presets,
        # whose obs has no contact_phase slot to read it.
        self._update_contact_phase()

        # Force-runaway safety valve (deploy-only; sim clamped F_ext so this was
        # dormant there). A large F_meas — from a wrong wrench_sign, a bad tare, or a
        # real hard contact — is the failure mode that escalates. There is NO software
        # abort of the WHOLE robot (hardware E-stop is the only true stop), but we DO
        # freeze the Jᵀ SERVO for this tick when tripped, so a phantom-force runaway
        # cannot keep the servo pushing Δq into the arms while the operator reacts.
        # (The F_meas obs itself is still reported — only the Layer-B push is cut.)
        # Checks the UNGATED force so the F_meas-on-cmd gate can never hide a runaway on
        # an uncommanded hand (see the snapshot above).
        _observe_only = getattr(self, "_servo_observe_only", False)
        runaway_tripped = False
        if self._force_runaway_n > 0.0:
            for hi, side in enumerate(("L", "R")):
                mag = float(np.linalg.norm(f_meas_ungated[hi]))
                if mag > self._force_runaway_n:
                    runaway_tripped = True
                    response = (
                        "ALARM ONLY in unified observe-only mode; policy remains active"
                        if _observe_only
                        else "SERVO FROZEN this tick"
                    )
                    logger.warning(
                        colored(
                            f"⚠️ F_meas({side})={mag:.1f}N > runaway {self._force_runaway_n:.0f}N — "
                            f"{response} (NO whole-robot abort; hit the hardware E-STOP).",
                            "red",
                        )
                    )

        if runaway_tripped and _observe_only:
            # OBSERVE-ONLY runaway: there is no Δq to cut, so freezing buys no safety —
            # and ZEROING I/Δx would actively harm, because in this mode I/Δx are the
            # SOLE controller's observation. It would hand the policy
            # {F_meas huge, φ=1, I=0, Δx=0} — a combination training never produces
            # (a large F_meas always comes with a wound I) at the exact instant of a
            # force runaway. So HOLD the last good I/Δx (stop winding on a reading we
            # do not trust, but keep the obs self-consistent) and skip the servo step.
            # Δq is already zero from the per-tick reset at the top. The operator still
            # gets the loud warning above; the hardware E-stop is still the only abort.
            self._servo_dq[:] = 0.0
            return

        if not (self._hybrid_servo or _observe_only) or (self._servo_cfg is None or runaway_tripped):
            # Servo state OFF (Layer-A checkpoint, no observe-only) or runaway-tripped:
            # I/Δx/Δq stay zero; F_meas obs is live, no Δq injected. Zeroing on a
            # runaway stops the servo pushing while the reading is out of range (it
            # resumes automatically once F_meas drops — note `servo_step` MULTIPLIES by
            # the contact gate, so I restarts from 0 rather than from its old value).
            # NOTE the observe-only term: a UNIFIED checkpoint has _hybrid_servo forced
            # False (no Δq ever) but MUST still get the wound I/Δx in its obs, so it
            # takes the live path below (and its runaway case is handled above).
            self._servo_integ[:] = 0.0
            self._servo_delta_x[:] = 0.0
            self._servo_dq[:] = 0.0
            return

        # --- Jᵀ force servo (Layer B): needs F^cmd (pelvis N), the TCP Jacobian, and
        # the contact gate. Only runs while the policy is actually driving motors. ---
        if not getattr(self, "use_policy_action", False):
            self._servo_integ[:] = 0.0
            self._servo_delta_x[:] = 0.0
            return
        f_cmd = self._force_cmd_pelvis_raw if self._force_cmd_pelvis_raw is not None else np.zeros((2, 3))
        dt = 1.0 / float(getattr(self.config.task, "rl_rate", 50.0) or 50.0)
        self._servo_jacobians(dof_pos, r_rels, np.asarray(f_cmd, dtype=np.float64), dt)
        # OBSERVE-ONLY (unified): the integrator/Δx above are exactly the training
        # signals and stay in the obs, but the Jᵀ residual is discarded here so no
        # actuation authority can leak into the PD path. Belt-and-suspenders — the
        # injection site in rl_inference is already gated on _hybrid_servo (forced
        # False for unified) — but zeroing the cache also keeps the RECORD tap and any
        # future consumer honest about what was actually commanded. (Training goes
        # further and skips the Jacobian entirely in observe-only; we keep the shared
        # code path so I/Δx cannot drift from the servo-ON deploy, at the cost of one
        # discarded Jacobian per tick.)
        if _observe_only:
            self._servo_dq[:] = 0.0

    def _reset_contact_phase_state(self) -> None:
        """Zero the soft contact-phase label + its hysteresis latch and warmup counter.

        Lazily creates BOTH buffers, so the ``object.__new__`` policy stubs the
        force-deploy tests use (which bypass ``__init__``) behave like a preset with
        no contact_phase term instead of raising.
        """
        for _attr in ("_contact_phase", "_contact_hyst"):
            if not isinstance(getattr(self, _attr, None), np.ndarray):
                setattr(self, _attr, np.zeros(2, dtype=np.float64))
        self._contact_phase[:] = 0.0
        self._contact_hyst[:] = 0.0
        self._contact_phase_ticks = 0

    def _contact_phase_warmup_active(self) -> bool:
        """True while the soft contact label must be forced OFF after START.

        Two overlapping suppression windows, both needed:

        * **Training parity** — ``contact_phase_warmup_steps`` (5) control ticks, the
          deploy twin of training's post-reset suppression (there the reset teleport's
          inertial spike in F_meas would flip the label ON in free space).
        * **The "reading not yet trusted" deadline** — the whole auto-tare burst window
          plus its settle interval (default 2.0 + 1.5 s ≫ 5 ticks), and the same settle
          interval after every MANUAL re-tare. This one is deploy-specific and is
          the load-bearing half: the burst exists precisely because the pre-tare
          reading is NOT trusted, and the LATCH would make a >θ_on residual bias stick
          at φ≈1 until the reading drops below θ_off — a phantom "in contact" for up to
          the whole burst, on every take. Since the unified rewards are φ-gated, that
          would tell the policy to execute a force task in free space.

        Two consequences, both deliberate:

        * With the burst DISABLED (``HOLOSOMA_FORCE_TARE_ON_START=0``) only the 5-tick
          parity window applies. ``_resolve_unified_mode`` warns loudly about exactly
          this, because it silently removes the protection above.
        * ``_start_force_tare_burst`` does not stack bursts, so a STOP taken mid-burst
          followed by an immediate re-START gets no FRESH tare (it does still inherit the
          old deadline, so it is never armed on an untared reading). Wait for the burst to
          finish before re-STARTing if you want a fresh zero.
        """
        ticks = int(getattr(self, "_contact_phase_ticks", 0))
        warmup = int(getattr(self, "_contact_phase_warmup_steps", _CONTACT_PHASE_WARMUP_STEPS))
        if ticks <= warmup:
            return True
        deadline = float(getattr(self, "_force_trust_deadline", 0.0) or 0.0)
        return time.monotonic() < deadline

    def _update_contact_phase(self) -> None:
        """Advance the soft, deployable per-hand contact label φ ∈[0,1] from ‖F_meas‖.

        Deploy twin of the phase block in ``hybrid._compute_contact_and_sensor`` on the
        TRAINING branch (`origin/hybrid_control`; the wbt_training copy vendored here
        predates it) — the ACTOR-facing ``contact_phase`` obs term of the unified presets:

            hyst ← 1 if ‖F_meas‖ > θ_on ; 0 if ‖F_meas‖ < θ_off ; else HOLD
            hyst ← 0 while the warmup is active
            φ    ← φ + β·(hyst − φ),  β = 1 − exp(−dt/τ)

        Two deliberate deploy choices:

        * **dt = one control tick (1/rl_rate = 20 ms), not training's 5 ms substep.**
          The LPF recursion is separable — training applies β(5 ms) four times per env
          step and exp(−5ms/τ)⁴ = exp(−20ms/τ) — so a single 20 ms update reproduces the
          per-env-step decay the policy observed EXACTLY whenever the latch is constant
          across the env step, and differs by at most a fraction of one tick's lead when
          the latch flips mid-step. It is also the only option available: the Bota
          bridge delivers one sample per control tick.
        * **Warmup counted in control ticks since START**, and additionally held for the
          whole auto-tare burst — see ``_contact_phase_warmup_active``.

        Consumes the FINAL ``_force_meas_pelvis`` (post-EMA, post-gate) so φ and the
        ``force_meas_local`` obs slot always describe the same force in STEADY STATE.
        (They still differ for ~4τ ≈ 200 ms after F_meas drops or the gate closes, while
        the LPF decays — that transient is exactly what training produces on a contact
        release, so it is in-distribution.) Zeroed whenever the policy is not driving
        motors, mirroring training's zeroed reset buffer.
        """
        if not getattr(self, "use_policy_action", False):
            # Pre-START / stopped: free space by definition (and the arms are moving
            # through the ready pose, so any reading is inertial). Also re-arms the
            # warmup for the next START.
            self._reset_contact_phase_state()
            return
        if not isinstance(getattr(self, "_contact_phase", None), np.ndarray) or not isinstance(
            getattr(self, "_contact_hyst", None), np.ndarray
        ):
            self._reset_contact_phase_state()
        # Backfill any MISSING tunable, only for an object.__new__ stub that bypassed
        # __init__ (the force-deploy test idiom). Per-attribute so a partially
        # configured stub keeps the values it did set; on a real policy every name
        # already exists, so nothing here executes.
        for _attr, _default in (
            ("_contact_phase_tau_s", _CONTACT_PHASE_TAU_S),
            ("_contact_phase_warmup_steps", _CONTACT_PHASE_WARMUP_STEPS),
        ):
            if not hasattr(self, _attr):
                setattr(self, _attr, _default)
        # The two hysteresis edges are backfilled as a PAIR: filling one from the default
        # while the other was set by the caller could invert the dead band (theta_off >
        # theta_on makes every reading in between latch ON, because the release `elif`
        # never runs). __init__ validates the ordering for a real policy; this keeps the
        # stub path from silently constructing the invalid combination.
        if not (hasattr(self, "_contact_phase_theta_on") and hasattr(self, "_contact_phase_theta_off")):
            self._contact_phase_theta_on = getattr(self, "_contact_phase_theta_on", _CONTACT_PHASE_THETA_ON)
            self._contact_phase_theta_off = min(
                getattr(self, "_contact_phase_theta_off", _CONTACT_PHASE_THETA_OFF), self._contact_phase_theta_on
            )

        self._contact_phase_ticks = int(getattr(self, "_contact_phase_ticks", 0)) + 1

        mag = np.linalg.norm(np.asarray(self._force_meas_pelvis, dtype=np.float64), axis=-1)  # (2,)
        for hi in range(2):
            if mag[hi] > self._contact_phase_theta_on:
                self._contact_hyst[hi] = 1.0
            elif mag[hi] < self._contact_phase_theta_off:
                self._contact_hyst[hi] = 0.0
            # else: inside the dead band → HOLD the latch (that IS the hysteresis)
        if self._contact_phase_warmup_active():
            self._contact_hyst[:] = 0.0

        dt = 1.0 / float(getattr(self.config.task, "rl_rate", 50.0) or 50.0)
        beta = 1.0 - float(np.exp(-dt / max(self._contact_phase_tau_s, 1e-6)))
        self._contact_phase += beta * (self._contact_hyst - self._contact_phase)
        # Numerical hygiene. The recursion is a convex blend of values in [0,1] with a
        # validated finite tau, so in principle it cannot leave the interval or go
        # non-finite — but this label is fed STRAIGHT to the ONNX and there is NO NaN
        # guard anywhere downstream (np.clip propagates NaN, and BasePolicy adds the
        # scaled action to the PD target unchecked). So scrub non-finite values to 0
        # ("free space", the safe reading) and clamp, exactly as _force_meas_pelvis is
        # scrubbed above. Cost is two vectorized ops on a 2-vector.
        if not np.all(np.isfinite(self._contact_phase)):
            # Warn ONCE: loguru here is synchronous on the 50 Hz control thread, and a
            # non-finite beta (e.g. a corrupted rl_rate) would make this fire every tick.
            if not getattr(self, "_contact_phase_nonfinite_warned", False):
                self._contact_phase_nonfinite_warned = True
                logger.warning(colored("contact_phase went non-finite — forcing 0 (free space). Warned once.", "red"))
            self._contact_phase[~np.isfinite(self._contact_phase)] = 0.0
        np.clip(self._contact_phase, 0.0, 1.0, out=self._contact_phase)

    def _servo_jacobians(self, dof_pos: np.ndarray, r_rels: list[np.ndarray], f_cmd: np.ndarray, dt: float) -> None:
        """Compute per-hand TCP Jacobians, run servo_step, and update the servo
        caches (integrator I, setpoint shift Δx, per-DOF Δq). Servo-on path only.

        The pelvis-frame (identity-root, LOCAL_WORLD_ALIGNED) frame Jacobian at each
        wrist_yaw is sliced to that hand's 7 arm velocity columns, then the linear
        rows are shifted to the TCP (``J_v − [r]× J_ω``, r = R_rel @ (tcp,0,0)) —
        matching hybrid._tcp_jacobian_world. servo_step then maps the task wrench to
        Δq via Jᵀ. Δq is scattered into a 29-vector by joint name (SDK/config order).
        """
        import pinocchio as pin

        from wbt_wrappers_inference.utils.force_deploy_math import servo_step, tcp_jacobian_from_wrist

        model = self.pinocchio_robot.robot_model
        data = self.pinocchio_robot.robot_data
        # q with identity root (pelvis frame); computeJointJacobians needs a fresh FK.
        joint_pos_pin = dof_pos[self.pinocchio_robot.real2pinocchio_index]
        q = np.concatenate([np.zeros(3), np.array([0.0, 0.0, 0.0, 1.0]), joint_pos_pin])
        pin.computeJointJacobians(model, data, q)
        pin.framesForwardKinematics(model, data, q)

        # Resolve, once, each hand's 7 arm-joint velocity columns in the Pinocchio
        # Jacobian and where they map back into the 29-DOF (config/SDK) action vector.
        if getattr(self, "_servo_arm_cols", None) is None:
            self._resolve_servo_arm_columns()

        self._servo_dq[:] = 0.0
        cfg = self._servo_cfg
        for hi, fid in enumerate(self._force_wrist_frame_ids):
            J6 = pin.getFrameJacobian(model, data, fid, pin.LOCAL_WORLD_ALIGNED)  # (6, nv)
            cols = self._servo_arm_cols[hi]  # nv-columns for this hand's 7 arm joints
            j_lin = J6[:3, cols]  # (3, 7)
            j_ang = J6[3:, cols]  # (3, 7)
            r_w = r_rels[hi] @ np.array([cfg.tcp_offset_x, 0.0, 0.0])  # wrist→TCP lever, pelvis frame
            j_tcp_lin = tcp_jacobian_from_wrist(j_lin, j_ang, r_w)  # (3, 7)
            gate = self._servo_contact_gate(hi)
            dq_arm, new_integ, dx = servo_step(
                j_tcp_lin, f_cmd[hi], self._force_meas_pelvis[hi], gate, self._servo_integ[hi], cfg, dt
            )
            self._servo_integ[hi] = new_integ
            self._servo_delta_x[hi] = dx
            for col_i, dof_i in enumerate(self._servo_arm_pd_slice[hi]):
                self._servo_dq[dof_i] += float(dq_arm[col_i])
        self._warn_on_integrator_windup()

    def _servo_contact_gate(self, hi: int) -> float:
        """Per-hand integrator gate ∈{0,1}: the deploy stand-in for training's gate.

        Training gates the integrator on the PRIVILEGED geometric flag ``depth > 0``
        (`hybrid.py`), which is unavailable on hardware, so the servo deploy path has
        always approximated it with ``‖F_meas‖ > HOLOSOMA_FORCE_CONTACT_N``. That
        approximation has a false-POSITIVE mode nobody had to care about while the
        servo was off by default: a *sustained* free-space offset above the threshold —
        a stale/drifted tare, an imperfect RNEA gravity model at an arm pose far from
        where the tare was taken, gripper preload, or simply a grasped 0.3 kg object —
        looks exactly like contact. With ``F^cmd = 0`` the error is then ``e = F_meas``,
        so I ramps to ±``integrator_max`` PER AXIS (the anti-windup clip is elementwise,
        matching training) — up to ``‖I‖ = 40·√3 ≈ 69 N — and ``Δx = (F^cmd+I)/K`` follows:
        ≈22 cm for a single-axis phantom, and up to ≈56 cm with F^cmd also at its 25 N/axis
        clamp. Tens of centimetres of commanded setpoint shift, built up over tens of
        seconds, with the runaway valve (tens of N) never firing because the READING is 3 N.

        For the SERVO presets that only ever drove Δq, and it stays byte-identical here.
        For the UNIFIED presets I/Δx ARE the sole controller's observation, and the
        policy is trained to chase Δx — so an unbounded phantom Δx is unbounded arm
        motion. Two extra observe-only conditions close that:

        * **Command gate** (``HOLOSOMA_UNIFIED_INTEGRATOR_CMD_GATE``, default ON): I only
          winds on a hand that carries a nonzero F^cmd. This makes the documented
          bring-up run (every F^cmd at 0) genuinely equivalent to Layer A — I ≡ Δx ≡ 0,
          no integrator-driven motion at all — while a commanded hand runs the full
          trained loop. Note this is NOT ``GATE_CMD``: F_meas and contact_phase still
          flow into the obs, so the policy still SEES contact; only the integrator is
          held. Set to 0 for a fidelity A/B (training winds I in contact even at
          F^cmd≈0, which at deploy is dominated by the phantom-bias risk above).
        * **Warmup/tare window**: while ``_contact_phase_warmup_active`` holds, the
          reading is by definition untrusted (that is the whole argument for suppressing
          φ) — so do not wind an integrator off it either.
        """
        mag = float(np.linalg.norm(self._force_meas_pelvis[hi]))
        if mag <= self._force_contact_n:
            return 0.0
        if not getattr(self, "_servo_observe_only", False):
            return 1.0  # servo presets: byte-identical to the original threshold gate
        if self._contact_phase_warmup_active():
            return 0.0
        if getattr(self, "_unified_integ_cmd_gate", True):
            f_cmd_raw = getattr(self, "_force_cmd_pelvis_raw", None)
            cmd_mag = float(np.linalg.norm(f_cmd_raw[hi])) if f_cmd_raw is not None else 0.0
            if cmd_mag <= getattr(self, "_force_meas_gate_cmd_n", 1e-6):
                return 0.0
        return 1.0

    def _warn_on_integrator_windup(self) -> None:
        """Throttled operator alarm for a winding integrator (observe-only only).

        With no soft abort, the operator's only defence against the phantom-bias windup
        described in ``_servo_contact_gate`` is SEEING it. Warn once per crossing (with
        a re-arm hysteresis) when ‖I‖ passes half of ``integrator_max`` — and say the
        likely cause, because the cure is almost always "re-tare in free space".

        ``HOLOSOMA_UNIFIED_WINDUP_WARN_N`` (:func:`parse_unified_windup_warn_n`) replaces that
        half-of-the-clip threshold with a fixed number of newtons, ``0`` switching the alarm
        off: under the HL planner an approaching hand with a small F^cmd reaches 20 N in seconds
        by design, and the rig's launcher defaults it to 200 (unreachable, 2026-09-20). Print-only
        either way — nothing here touches a command.
        """
        if not getattr(self, "_servo_observe_only", False) or self._servo_cfg is None:
            return
        override = getattr(self, "_integ_windup_warn_n", None)
        if override is None:
            thresh = 0.5 * float(self._servo_cfg.integrator_max)
        elif float(override) <= 0.0:
            return  # HOLOSOMA_UNIFIED_WINDUP_WARN_N=0: the alarm is switched off explicitly
        else:
            thresh = float(override)
        peak = float(np.max(np.linalg.norm(self._servo_integ, axis=-1)))
        if peak > thresh and not getattr(self, "_integ_windup_warned", False):
            self._integ_windup_warned = True
            f_cmd_raw = getattr(self, "_force_cmd_pelvis_raw", None)
            cmd_peak = float(np.max(np.linalg.norm(f_cmd_raw, axis=-1))) if f_cmd_raw is not None else 0.0
            logger.warning(
                colored(
                    f"⚠️ servo integrator ‖I‖={peak:.1f}N > {thresh:.0f}N (F^cmd peak {cmd_peak:.1f}N). "
                    f"Δx is now ~{peak / float(self._servo_cfg.stiffness_K) * 100:.0f} cm and the UNIFIED policy "
                    "chases it. If you are not pressing something, the F/T tare has drifted — re-tare in FREE "
                    "SPACE (R2 / T) or zero F^cmd (p). Warned once per crossing.",
                    "red",
                )
            )
        elif peak < 0.25 * thresh:
            self._integ_windup_warned = False  # re-arm once it unwinds

    def _resolve_servo_arm_columns(self) -> None:
        """Precompute (once) each hand's 7 arm-joint Pinocchio velocity-Jacobian
        columns and their indices in the 29-DOF (config/SDK) action vector."""
        model = self.pinocchio_robot.robot_model
        dof_names = list(self.config.robot.dof_names)
        arm_joint_names = {
            "left_wrist_yaw_link": [
                "left_shoulder_pitch_joint",
                "left_shoulder_roll_joint",
                "left_shoulder_yaw_joint",
                "left_elbow_joint",
                "left_wrist_roll_joint",
                "left_wrist_pitch_joint",
                "left_wrist_yaw_joint",
            ],
            "right_wrist_yaw_link": [
                "right_shoulder_pitch_joint",
                "right_shoulder_roll_joint",
                "right_shoulder_yaw_joint",
                "right_elbow_joint",
                "right_wrist_roll_joint",
                "right_wrist_pitch_joint",
                "right_wrist_yaw_joint",
            ],
        }
        self._servo_arm_cols = []  # per hand: Pinocchio nv-columns of the 7 arm joints
        self._servo_arm_pd_slice = []  # per hand: indices into the 29-DOF action vector
        for bn in _VR_2POINT_BODIES:
            names = arm_joint_names[bn]
            cols = []
            for nm in names:
                jid = model.getJointId(nm)  # Pinocchio joint id
                cols.append(model.joints[jid].idx_v)  # velocity-Jacobian column (idx_v)
            self._servo_arm_cols.append(cols)
            self._servo_arm_pd_slice.append([dof_names.index(nm) for nm in names])

    @staticmethod
    def _rot_matrix_from_rpy(rpy: tuple[float, float, float]) -> np.ndarray:
        """3×3 rotation for a URDF `rpy` (extrinsic X-Y-Z: R = Rz·Ry·Rx). Numpy twin
        of force_tracking._rot_matrix_from_rpy — same convention, kept local so the
        deploy package has no training-package dependency."""
        roll, pitch, yaw = rpy
        cr, sr = np.cos(roll), np.sin(roll)
        cp, sp = np.cos(pitch), np.sin(pitch)
        cy, sy = np.cos(yaw), np.sin(yaw)
        rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
        ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
        rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
        return (rz @ ry @ rx).astype(np.float32)

    def _trigger_soft_stop(self) -> None:
        """Begin a damped descent to limp: kp→floor, kd held, q frozen.

        Flips OFF the policy and any clip so the manual-command path
        (_get_manual_command) takes over and runs the soft ramp. Idempotent —
        re-triggering does not restart the ramp. This is the graceful software
        stop; the hardware E-stop remains the only link-independent abort.
        """
        if self._soft_stop_active:
            return
        self._soft_stop_active = True
        self._soft_stop_count = 0
        self._soft_stop_q = None  # captured from the measured pose on first tick
        # Hand control to the manual path and stop policy/clip/VIVE immediately.
        self.use_policy_action = False
        self.get_ready_state = False
        self._stiff_hold_active = True
        self._init_interpolating = False
        self._prepend_stiff_active = False
        self.motion_clip_progressing = False
        self._vive_armed = False
        # Drop the VIVE slew + clutch anchors too (parity with STOP/switch/reset),
        # so a later re-arm always re-snapshots cleanly rather than reusing a stale
        # tracker origin from before the descent.
        self._clear_vive_teleop_anchors()
        # Drop the EMA history so any later policy resume re-seeds from raw (the
        # robot pose changes while it sags; a stale filter state would jump).
        self._action_ema_state = None
        self.logger.warning(
            colored("SOFT-STOP triggered — damped descent; robot will sink/fold to harness/floor.", "red")
        )

    def _dispatch_command(self, cmd):
        """Add SOFT-STOP, VIVE-arm, and F/T-TARE commands on top of the WBT dispatch.

        * StateCommand.ZERO_VELOCITY (keyboard 'z', joystick L2) → soft-stop
          (damped descent to limp). Locomotion-only no-op for standing WBT, so
          safe to repurpose; a dedicated 'go limp' key distinct from STOP.
        * StateCommand.STAND_TOGGLE (keyboard '=', joystick 'start') → toggle
          live VIVE teleop on/off. In relative/clutch mode (default) arming snaps
          the anchor at the trackers' current pose (zero jump, no pose gate); in
          absolute mode it runs the pre-arm pose gate. Lets the operator first
          stand under the policy (A=START), then hand the hands to the trackers —
          distinct from select+A which plays the NPZ clip.
        * StateCommand.SWITCH_MODE (keyboard 'x', joystick 'X') → RECENTER the
          clutch: slide the hands back to the reference pose and re-anchor the
          tracker origin (the operator's 'c' = recalibrate-in-place). base treats
          SWITCH_MODE as a no-op, so repurposing it here is safe.
        * StateCommand.STAND (joystick 'R2', keyboard 'T') → re-TARE both F/T
          sensors (zero the reading in free space). STAND is a ROS2-only enum
          never dispatched on the WBT path, so repurposing it is collision-free;
          the joystick 'R2' button is likewise unbound in JOYSTICK_COMMANDS (we
          add the R2→STAND binding on the live provider in _create_input_providers,
          never editing the pinned submodule map).
        """
        from holosoma_inference.inputs.api.commands import StateCommand

        if cmd == StateCommand.ZERO_VELOCITY:
            self._trigger_soft_stop()
        elif cmd == StateCommand.STAND_TOGGLE:
            self._toggle_vive_teleop()
        elif cmd == StateCommand.SWITCH_MODE:
            self._vive_recenter()
        elif cmd == StateCommand.STAND:
            self._trigger_force_tare()
        else:
            super()._dispatch_command(cmd)

    def _trigger_force_tare(self) -> None:
        """Re-tare both F/T sensors via the bridge (joystick 'R2' / keyboard 'T').

        Sends {"cmd":"tare"} over the ForceLiveSource WS so the bridge re-zeroes
        the compensated wrench. No-op (with a warning) when no live F/T source is
        configured. ⚠️ Do this in FREE SPACE — a contact force present at tare is
        baked in as a constant bias. Also clears the local servo integrator so the
        re-zero doesn't leave a stale wound-up I.
        """
        if self._force_live is None:
            self.logger.warning(
                colored("F/T tare requested but no live F/T source (HOLOSOMA_FORCE_WS_URL unset).", "yellow")
            )
            return
        self._force_live.request_tare()
        self._servo_integ[:] = 0.0
        self._servo_delta_x[:] = 0.0
        # A manual re-tare is the documented cure for a LATCHED contact_phase (a drifted
        # bias above θ_on holds φ≈1 in free space). Clearing the latch here makes that
        # cure immediate instead of waiting ~4τ for the LPF to decay once the bridge
        # re-zeroes — and it re-arms the post-tare suppression window, which is exactly
        # what you want after asking for a new zero. No-op for non-unified presets
        # (nothing reads φ) and symmetric with the I/Δx clear above.
        self._reset_contact_phase_state()
        self._integ_windup_warned = False
        # Hold φ + the integrator until this tare has actually been APPLIED, exactly as
        # the START burst does. Without it the latch re-asserts ~120 ms later off the
        # still-frozen pre-tare reading, so the documented cure would not work.
        self._arm_force_trust_window(max(getattr(self, "_force_tare_settle_s", _FORCE_TARE_SETTLE_S), 0.0))
        self.logger.info(
            colored("F/T TARE requested (zeroing both sensors — keep the grippers in FREE SPACE).", "magenta")
        )

    def _start_force_tare_burst(self) -> None:
        """Fire a BURST of F/T re-tares on the START edge so the LAST one zeroes the
        sensor AFTER the init/ready-pose transient settles (no crash from a nonzero
        startup reading).

        Runs on a short daemon thread: ``N`` requests spread over ``_force_tare_on_start_s``
        (default 5 over 2 s). The bridge coalesces repeated ``{"cmd":"tare"}`` (a
        threading.Event) and each tare drains its own free-space window, so several
        spread-out requests give it multiple chances to catch a clean, still window
        while the ready-pose swing finishes. No-op when no live F/T source or the
        count is 0 (byte-identical). A previous burst still in flight is left to
        finish (idempotent — re-START just re-arms after it ends). ⚠️ grippers MUST be
        in FREE SPACE across the whole window (a contact force at tare bakes in as a
        constant bias).

        The burst ONLY sends re-tare requests — it does NOT touch the servo integrator.
        ``_reset_motion_state`` (run by ``_handle_start_motion_clip`` on this same START
        edge) already zeroes I/Δx, and the control loop re-derives them every tick with
        the contact gate forcing I→0 in free space; clearing them from this daemon
        thread would be a redundant, unsynchronized write to state the 50 Hz control
        thread owns. Whenever the count is >1 (or the window is positive) at least ONE
        tare lands at the END of the window, AFTER the ready-pose transient settles —
        the whole point (a lone t≈0 tare would fire mid-transient).
        """
        if self._force_live is None or self._force_tare_on_start_n <= 0:
            return
        if self._force_tare_burst_thread is not None and self._force_tare_burst_thread.is_alive():
            return  # a burst from a prior START is still running; don't stack

        total_s = max(self._force_tare_on_start_s, 0.0)
        # Guarantee a tare AFTER the settle window: with a positive window, fire at
        # least 2 (one early to catch the resting bias, one at the very end once the
        # ready-pose swing has finished). A count of 1 with a window would otherwise
        # fire once at t≈0 — during the transient — defeating the "zero after settle"
        # goal (verified defect, 6-agent review). With total_s=0 the count is honored
        # as-is (all fired back-to-back).
        n = max(self._force_tare_on_start_n, 2) if total_s > 0.0 else self._force_tare_on_start_n
        gap = (total_s / (n - 1)) if n > 1 else 0.0
        source = self._force_live

        def _burst() -> None:
            for i in range(n):
                try:
                    source.request_tare()
                except Exception as exc:  # noqa: BLE001 — never let the burst thread crash startup
                    self.logger.warning(f"Auto-tare burst request failed: {exc!r}")
                    return
                if i < n - 1 and gap > 0.0:
                    time.sleep(gap)

        self.logger.info(
            colored(
                f"Auto-tare BURST on START: {n}× over {total_s:.1f}s "
                f"(+{getattr(self, '_force_tare_settle_s', 1.5):.1f}s settle) — keep the grippers in FREE SPACE.",
                "magenta",
            )
        )
        # Arm the "reading not yet trusted" deadline for the WHOLE burst plus its settle
        # interval, BEFORE starting the thread — so the first control tick of this take
        # already sees the window open (see _contact_phase_warmup_active).
        self._arm_force_trust_window(total_s + max(getattr(self, "_force_tare_settle_s", _FORCE_TARE_SETTLE_S), 0.0))
        self._force_tare_burst_thread = threading.Thread(target=_burst, name="force_tare_burst", daemon=True)
        self._force_tare_burst_thread.start()

    def _arm_force_trust_window(self, seconds: float) -> None:
        """Extend the "live F/T reading is not yet trusted" deadline by ``seconds``.

        Monotonic (immune to wall-clock steps) and never SHORTENS an existing window. The
        C1 unified path holds contact_phase AND the servo integrator at zero while the
        window is open; every other preset ignores it.
        """
        if not math.isfinite(seconds) or seconds <= 0.0:
            return
        self._force_trust_deadline = max(
            float(getattr(self, "_force_trust_deadline", 0.0) or 0.0), time.monotonic() + float(seconds)
        )

    def _create_input_providers(self):
        """Bind the F/T-tare button on the live command provider, then defer to base.

        The Unitree controller produces an 'R2' button label that is UNMAPPED in
        the pinned submodule's JOYSTICK_COMMANDS; we add R2→STAND (a WBT-path no-op
        enum, repurposed as tare) on the provider's per-instance ``_mapping`` copy
        — so the button works without editing the submodule. Keyboard gets 'T'/'t'.
        Off-path safe: the binding is harmless for non-force presets (STAND just
        routes to _trigger_force_tare, which no-ops without a live source).
        """
        super()._create_input_providers()
        try:
            from holosoma_inference.inputs.api.commands import StateCommand

            for prov in {getattr(self, "_command_provider", None), getattr(self, "_velocity_input", None)}:
                mapping = getattr(prov, "_mapping", None)
                if isinstance(mapping, dict):
                    mapping.setdefault("R2", StateCommand.STAND)  # joystick trigger
                    mapping.setdefault("T", StateCommand.STAND)  # keyboard (upper)
                    mapping.setdefault("t", StateCommand.STAND)  # keyboard (lower)
        except Exception as exc:  # noqa: BLE001 — never let a binding tweak break startup
            self.logger.warning(f"Could not bind the F/T tare button: {exc!r}")

    def _clear_vive_teleop_anchors(self) -> None:
        """Drop every live-teleop anchor: the slew state, the relative clutch origin and
        the absolute arm-clutch offset — for BOTH the keypoint and the dense joint path.

        Called from all five places that end or interrupt an armed session (disarm, STOP,
        soft-stop, clip start, policy switch) so a later re-arm always re-snapshots cleanly
        instead of reusing an origin captured before the interruption. Extracted because
        the same six-line block was open-coded at each of those sites, and the dense path
        added four more fields to keep in step — one forgotten line there is a stale
        anchor, i.e. a real arm jump on hardware."""
        self._vr_last_pos = None
        self._vr_last_orn = None
        self._vr_clutch_live0 = None
        self._vr_clutch_orn0 = None
        self._vr_abs_off_pos = None
        self._vr_abs_orn0 = None
        self._dense_last_q = None
        self._dense_last_dq = None
        self._dense_clutch_q0 = None
        self._dense_abs_off_q = None
        self._ref_joints_last = None  # the walk skill's reference legs end with the armed session

    def _snapshot_clutch_anchor(self) -> bool:
        """Snapshot the RELATIVE-teleop anchor = (held hand ref, live tracker pos).

        Returns False (and logs) if there is no live frame. After this, the
        commanded hand target is ``ref + (live_now - live0)`` so the operator's
        trackers can sit anywhere — only motion SINCE the anchor moves the hands,
        giving a zero-jump arm and a clean recenter.
        """
        # DENSE-UNIFIED: the anchor is a JOINT vector, not a keypoint pose. Snapshot the
        # completion's joint solution (already remapped into policy dof order) so the tick
        # path's `ref + (live - live0)` starts at exactly the held reference.
        if self._dense_joint_teleop:
            got = self._dense_live_vectors()
            if got is None:
                self.logger.warning(colored("DENSE clutch: no live joint frame (stale/disconnected).", "red"))
                return False
            self._dense_clutch_q0 = got[0].astype(np.float32).copy()
            self._dense_last_q = None  # first commanded frame == the held reference, then ramps
            self._dense_last_dq = None
            return True
        live_pos, live_orn = self._vive_live.latest()
        if live_pos is None:
            self.logger.warning(colored("VIVE clutch: no live frame (stale/disconnected).", "red"))
            return False
        # The held hand reference (pos AND orn) is re-read live each tick (FK of
        # the frozen clip frame) in the obs override, so we snapshot only the
        # tracker ORIGIN here. `_vr_clutch_live0` doubles as the armed-anchor
        # sentinel (None ⇒ no override).
        self._vr_clutch_live0 = live_pos.astype(np.float32).copy()
        # Orientation origin (only when orn clutch enabled + the bridge sends orn).
        # 6D-orn 3pt preset: the bridge's 3pt orn wire rep is 3×xyzw-quaternion
        # (12D) but the obs rep is 3×rot6d (18D) — store the anchor ALREADY
        # CONVERTED (same ingest conversion as _compute_obs_terms) so the clutch
        # composes rotations in one consistent rep. Guard on a genuine 3pt frame
        # (pos[9]+orn[12]) so a 2pt frame's rot6d bytes are never misread as quats.
        if self._vive_orientation and live_orn is not None:
            if self._is_3pt and self._vr_orn_6d and live_pos.size == 9 and live_orn.size == 12:
                live_orn = self._quat12_to_rot6d18(live_orn)
            self._vr_clutch_orn0 = live_orn.astype(np.float32).copy()
        else:
            self._vr_clutch_orn0 = None
        # Reset the slew anchor so the first commanded frame after (re)anchoring is
        # exactly the held reference (live_now - live0 == 0), then ramps from there.
        self._vr_last_pos = None
        self._vr_last_orn = None
        self._orn_mismatch_warned = False  # fresh anchor → re-arm the warn-once latch
        return True

    def _toggle_vive_teleop(self) -> None:
        """Arm/disarm live VIVE 2pt teleop (the 'start' button).

        Requires the policy to be running (A=START first) and a VIVE source.
        RELATIVE/clutch mode (default): arming snapshots the clutch anchor at the
        trackers' CURRENT pose, so the hands start exactly at the held reference
        and follow only relative motion — no pose gate, no jump, trackers can be
        anywhere. ABSOLUTE mode (HOLOSOMA_VIVE_RELATIVE=0): the tracker pose IS the
        hand target, so arming runs the pre-arm pose gate (refuse unless the live
        hand pose ≈ the robot's held pose). Either way the motion clip is frozen so
        only the hands track while the lower body holds standing. Disarming returns
        the hands to the (frozen) reference pose.
        """
        if self._vive_live is None or not (self._uses_vr_keypoints or self._dense_joint_teleop):
            self.logger.warning(colored("VIVE arm: no live 2pt/3pt/5pt/dense source — nothing to toggle.", "yellow"))
            return
        if not self.use_policy_action:
            self.logger.warning(colored("VIVE arm: press A (START) first so the policy is running.", "yellow"))
            return
        if self._vive_armed:
            self._vive_armed = False
            self._clear_vive_teleop_anchors()
            self._dense_last_q = None  # ...and the joint-space twins of all four
            self._dense_last_dq = None
            self._dense_clutch_q0 = None
            self._dense_abs_off_q = None
            self.logger.info(colored("VIVE teleop DISARMED — the robot holds the standing reference.", "yellow"))
            return
        if self._vive_relative:
            # Clutch: snap the anchor at wherever the trackers are right now.
            if not self._snapshot_clutch_anchor():
                return
            self.motion_clip_progressing = False  # keep the clip frozen; only the tracked slots move
            self._vive_armed = True
            _what = "all 29 joint targets" if self._dense_joint_teleop else "hands"
            self.logger.info(
                colored(
                    f"VIVE teleop ARMED (relative/clutch) — {_what} start at the reference and follow your "
                    "trackers' motion. Press the recenter key (keyboard 'x' / joystick 'X') to slide back "
                    "to the reference and re-anchor.",
                    "green",
                )
            )
            return
        # Absolute mode: pose gate first.
        if not self._vive_arm_pose_ok():
            if self._vive_arm_refuse_why == "no_frame":
                # NOT a pose problem, and telling the operator to stand differently is the
                # one instruction that cannot help: there is no live frame to measure. The
                # usual cause is the bridge withholding its payload — `snapshot()` returns
                # None until it is CALIBRATED, and also whenever any tracked role is unseen
                # or stale — so the fix is on T1, not in the operator's stance.
                self.logger.warning(
                    colored(
                        "VIVE arm REFUSED — NO LIVE FRAME from the bridge (nothing to compare; this is "
                        "NOT a pose error, so moving your hands will not help). On the bridge terminal, "
                        "check its heartbeat: it publishes NOTHING until 'calibrated=True', and also "
                        "while any tracked puck is unseen or STALE. Hold the calibration pose (hands in "
                        "front, separated L/R; head level facing +x) ~3 s until it calibrates, then "
                        "press 'start' again.",
                        "red",
                    )
                )
            else:
                # Units and the override env differ between the keypoint families (metres,
                # HOLOSOMA_VIVE_ARM_TOL) and dense (radians, HOLOSOMA_VIVE_ARM_TOL_RAD), so
                # quoting one pair for both would send the operator at the wrong knob.
                if self._dense_joint_teleop:
                    _which = self._vive_arm_refuse_detail or "one or more steerable joints"
                    _tol = f"{self._vive_arm_tol_rad:.3f} rad"
                    _env = "HOLOSOMA_VIVE_ARM_TOL_RAD"
                    _what = "joints"
                else:
                    _which = self._vive_arm_refuse_detail or "one or more tracked points"
                    _tol = f"{self._vive_arm_tol:.3f} m"
                    _env = "HOLOSOMA_VIVE_ARM_TOL"
                    _what = "points"
                self.logger.warning(
                    colored(
                        f"VIVE arm REFUSED — TOO FAR at: {_which} (tol {_tol}). Move THOSE {_what} "
                        "toward the robot's current pose and press 'start' again. Only what you can "
                        f"actually STEER is gated, so this is always actionable. (Override tol via "
                        f"{_env}, or use relative mode HOLOSOMA_VIVE_RELATIVE=1, which needs no pose "
                        "match at all.)",
                        "red",
                    )
                )
            return
        # ABSOLUTE-mode ARM CLUTCH: capture the constant offset FK(reference) − wire at
        # THIS instant, so the first armed target == the held pose (zero jump) and later
        # motion tracks the wire delta with no reach bias (see __init__). Snapshot the
        # SAME two quantities the pre-arm gate just compared: held = FK(_motion_command_t
        # frozen clip frame), wire = the live source. A missing live frame cannot happen
        # here (the gate above already refused on it), but guard anyway → no clutch (fall
        # back to the raw wire, the pre-fix behavior) rather than a stale/None offset.
        self._vr_abs_off_pos = None
        self._vr_abs_orn0 = None
        self._dense_abs_off_q = None
        if self._dense_joint_teleop:
            # Joint-space twin: off = ref_q − live_q at arm, so the first armed command is
            # the held pose and later motion tracks the completion's joint DELTA. Guard the
            # same way (no live frame ⇒ no clutch ⇒ raw wire) rather than a stale offset.
            if self._vive_abs_clutch:
                _got = self._dense_live_vectors()
                if _got is not None:
                    _ref_q = self._motion_command_t[0, : self.num_dofs].astype(np.float32)
                    self._dense_abs_off_q = (_ref_q - _got[0]).astype(np.float32)
            self._dense_last_q = None
            self._dense_last_dq = None
            self.motion_clip_progressing = False  # keep the clip frozen
            self._vive_armed = True
            _dc = "arm-clutch ON (zero-jump)" if self._dense_abs_off_q is not None else "raw wire (no clutch)"
            self.logger.info(
                colored(
                    f"DENSE teleop ARMED (absolute, {_dc}) — all 29 joint targets follow the "
                    "completion of your trackers (clip frozen).",
                    "green",
                )
            )
            return
        if self._vive_abs_clutch:
            live_pos, live_orn = self._vive_live.latest()
            if live_pos is not None:
                offsets = self._vr_offsets()
                held_pos, _ = self._compute_vr_fk(self._motion_command_t[0, :29], offsets)
                # Per-point over the points the live frame supplies (min in whole
                # 3-vectors); tail points get zero offset so _match_len's ref-held tail
                # is unchanged. Stored length == the obs vr target length (held_pos).
                off = np.zeros_like(held_pos, dtype=np.float32)
                lp = live_pos.reshape(-1, 3)
                hp = held_pos.reshape(-1, 3)
                n = min(lp.shape[0], hp.shape[0])
                off.reshape(-1, 3)[:n] = (hp[:n] - lp[:n]).astype(np.float32)
                self._vr_abs_off_pos = off
                # ORIENTATION anchor: store the wire orn at arm, in the OBS rep, so the
                # armed target composes R_target = (R_wire · R_wire0ᵀ) · R_ref via the SAME
                # proven helpers the relative clutch uses (_orn_clutch_target /
                # _orn_clutch_target_quat). Rotations COMPOSE (they do not subtract), so this
                # is the correct orientation analogue of the position offset above. Apply the
                # 6D-preset ingest conversion here too (quat12 wire → rot6d18 obs rep), on
                # the SAME genuine-3pt guard as the tick path, so the anchor and the live
                # frame are always in one consistent rep.
                if live_orn is not None:
                    _lo = live_orn
                    if self._is_3pt and self._vr_orn_6d and live_pos.size == 9 and live_orn.size == 12:
                        _lo = self._quat12_to_rot6d18(live_orn)
                    self._vr_abs_orn0 = np.asarray(_lo, dtype=np.float32).copy()
        self.motion_clip_progressing = False  # keep the clip frozen; only hands track
        self._vive_armed = True
        _clutch = "arm-clutch ON (zero-jump)" if self._vr_abs_off_pos is not None else "raw wire (no clutch)"
        _oc = "pos+orn" if self._vr_abs_orn0 is not None else "pos only"
        self.logger.info(colored(
            f"VIVE teleop ARMED (absolute, {_clutch}, {_oc}) — hands follow your trackers (clip frozen).", "green"))

    def _vive_recenter(self) -> None:
        """Recenter the clutch (the operator's requested 'c' behavior).

        Re-snaps the clutch anchor at the trackers' current pose, so the hands
        slide back to the (frozen) reference pose and the tracker origin resets —
        a fresh recalibration mid-session without disarming. No-op (with a log)
        when not armed or not in relative mode.
        """
        if not self._vive_armed:
            self.logger.warning(colored("VIVE recenter: not armed — press 'start' to arm first.", "yellow"))
            return
        if not self._vive_relative:
            self.logger.warning(
                colored("VIVE recenter: only available in relative mode (HOLOSOMA_VIVE_RELATIVE=1).", "yellow")
            )
            return
        if self._snapshot_clutch_anchor():
            self.logger.warning(
                colored(
                    "VIVE RECENTERED — hands are now GLIDING back to the reference pose (slew-capped, a real "
                    "motion); tracker origin reset. If pressed by accident, move your trackers to bring the "
                    "hands back.",
                    "cyan",
                )
            )

    def _vive_arm_pose_ok(self) -> bool:
        """True if the live VIVE tracker pose is close enough to arm without a glide.

        Compares the latest live ``vr_{2,3}point_pos`` against the robot's CURRENTLY
        HELD target = FK of the (frozen) clip frame. Returns False (refuse to arm)
        when the source is stale/absent or any point is farther than ``_vive_arm_tol``.
        ``_vive_arm_tol <= 0`` disables the gate. Only reached in ABSOLUTE mode — the
        default RELATIVE/clutch mode (HOLOSOMA_VIVE_RELATIVE=1) skips this gate. For
        3pt the live pose is [L,R,head] (9D); we compare all points present in the
        live frame (a 3-tracker frame → 3 gaps; a 2-tracker frame → 2 gaps).

        ⚠️ 5pt GATES ON THREE OF FIVE. The rig has three pucks; the two FOOT keypoints are
        Motion_Completion's inference off those same three, so an operator cannot aim them
        and refusing on them locks arming out permanently — measured on the rig, see
        ``_VR_5POINT_OPERATOR_PLACED``. They are still COMPUTED and logged (and warned about
        past ``_VR_DERIVED_GAP_WARN_M``, which catches a completion that is not standing),
        because a silently dropped check is indistinguishable from a check that passes.
        Dropping them costs no jump safety: the absolute arm clutch snapshots ``held − wire``
        for all five points, so a derived point's offset is re-based to zero identically.
        """
        if self._vive_live is None:
            self._vive_arm_refuse_why = "no_frame"
            self._vive_arm_refuse_detail = ""
            return False
        # DENSE-UNIFIED: no keypoints exist, so gate in JOINT space — the completion's
        # solution vs the held reference, per joint.
        if self._dense_joint_teleop:
            if self._vive_arm_tol_rad <= 0.0:
                return True  # gate disabled
            got = self._dense_live_vectors()
            if got is None:
                self._vive_arm_refuse_why = "no_frame"
                self._vive_arm_refuse_detail = ""
                self.logger.warning(colored("DENSE arm gate: no live joint frame (stale/disconnected).", "red"))
                return False
            ref_q = self._motion_command_t[0, : self.num_dofs].astype(np.float32)
            gaps = np.abs(got[0] - ref_q)
            _names = list(getattr(self.config.robot, "dof_names", []) or [])
            # GATE ONLY ON THE JOINTS THE OPERATOR CAN STEER — the 5pt fix, by joint name.
            # A joint whose name is unknown here stays GATED: over-gating is a visible
            # refusal, under-gating is a silently skipped check.
            _derived = np.array(
                [
                    any(k in (_names[i] if i < len(_names) else "") for k in _VR_DENSE_DERIVED_JOINT_KEYS)
                    for i in range(int(gaps.shape[0]))
                ],
                dtype=bool,
            )
            _steer = ~_derived
            ok = bool(np.all(gaps[_steer] <= self._vive_arm_tol_rad)) if _steer.any() else True
            _si = np.flatnonzero(_steer)
            _worst = int(_si[np.argmax(gaps[_steer])]) if _si.size else 0
            _wn = _names[_worst] if _worst < len(_names) else f"dof{_worst}"
            self._vive_arm_refuse_why = "" if ok else "too_far"
            self._vive_arm_refuse_detail = (
                ""
                if ok
                else "; ".join(
                    f"{_names[i] if i < len(_names) else f'dof{i}'} {gaps[i]:.3f} rad"
                    for i in _si
                    if gaps[i] > self._vive_arm_tol_rad
                )
            )
            self.logger.info(
                colored(
                    f"DENSE arm gate: max STEERABLE joint gap {float(gaps[_worst]):.3f} rad at {_wn} "
                    f"({int(_steer.sum())} of {int(gaps.shape[0])} joints gated; tol "
                    f"{self._vive_arm_tol_rad:.3f}) → {'PASS' if ok else 'REFUSE'}",
                    "green" if ok else "yellow",
                )
            )
            if _derived.any():
                # Reported, never gated — same contract as the 5pt feet.
                _di = np.flatnonzero(_derived)
                _dworst = int(_di[np.argmax(gaps[_derived])])
                _dwn = _names[_dworst] if _dworst < len(_names) else f"dof{_dworst}"
                _dmax = float(gaps[_dworst])
                _dmsg = (
                    f"DENSE arm gate: derived (completion-inferred, NOT gated) max "
                    f"{_dmax:.3f} rad at {_dwn} over {int(_derived.sum())} joints — the arm clutch "
                    f"re-bases these to zero."
                )
                if _dmax > _VR_DENSE_DERIVED_GAP_WARN_RAD:
                    self.logger.warning(
                        colored(
                            f"{_dmsg} ⚠️  BUT that is > {_VR_DENSE_DERIVED_GAP_WARN_RAD:.2f} rad, which "
                            "means the completion is FOLDED/CROUCHED rather than merely stanced "
                            "differently — check that all three pucks are being WORN and tracked "
                            "before you arm.",
                            "yellow",
                        )
                    )
                else:
                    self.logger.info(colored(_dmsg, "cyan"))
            return ok
        if self._vive_arm_tol <= 0.0:
            return True  # gate disabled
        live_pos, _ = self._vive_live.latest()
        if live_pos is None:
            self._vive_arm_refuse_why = "no_frame"
            self._vive_arm_refuse_detail = ""
            self.logger.warning(colored("VIVE arm gate: no live frame (stale/disconnected).", "red"))
            return False
        ref_joint_pos = self._motion_command_t[0, :29]
        offsets = self._vr_offsets()
        held_pos, _ = self._compute_vr_fk(ref_joint_pos, offsets)
        # Compare only the points the live frame actually supplies (min of the two,
        # in whole 3-vectors), so a 2-tracker live frame against a 3pt hold still
        # gates on the hands and never index-errors.
        n_pts = min(live_pos.reshape(-1, 3).shape[0], held_pos.reshape(-1, 3).shape[0])
        live_p = live_pos.reshape(-1, 3)[:n_pts]
        held_p = held_pos.reshape(-1, 3)[:n_pts]
        gaps = np.linalg.norm(live_p - held_p, axis=1)
        # GATE ONLY ON THE POINTS THE OPERATOR CAN PLACE. For 2pt/3pt that is every point
        # (one puck each). For 5pt the last two are Motion_Completion's inferred FEET —
        # refusing on those is a lockout with no corresponding operator action, and the
        # clutch zeroes them anyway. See _VR_5POINT_OPERATOR_PLACED for the measurement.
        n_gate = min(_VR_5POINT_OPERATOR_PLACED, n_pts) if self._is_5pt else n_pts
        ok = bool(np.all(gaps[:n_gate] <= self._vive_arm_tol))
        self._vive_arm_refuse_why = "" if ok else "too_far"
        _lbl = [
            _VR_POINT_LABELS[i] if i < len(_VR_POINT_LABELS) else f"pt{i}" for i in range(int(n_pts))
        ]
        if ok:
            self._vive_arm_refuse_detail = ""
        else:
            _over = [f"{_lbl[i]} {gaps[i]:.3f} m" for i in range(n_gate) if gaps[i] > self._vive_arm_tol]
            self._vive_arm_refuse_detail = "; ".join(_over)
        self.logger.info(
            colored(
                f"VIVE arm gate: per-point gap {np.round(gaps[:n_gate], 3).tolist()} m "
                f"({', '.join(_lbl[:n_gate])}) (tol {self._vive_arm_tol:.3f}) "
                f"→ {'PASS' if ok else 'REFUSE'}",
                "green" if ok else "yellow",
            )
        )
        if n_gate < n_pts:
            # Reported, never gated. Loud only when the completion is not standing at all.
            _d_gaps = gaps[n_gate:]
            _d_msg = (
                f"VIVE arm gate: derived (completion-inferred, NOT gated) "
                f"{np.round(_d_gaps, 3).tolist()} m ({', '.join(_lbl[n_gate:])}) — the arm clutch "
                f"re-bases these to zero."
            )
            if float(np.max(_d_gaps)) > _VR_DERIVED_GAP_WARN_M:
                self.logger.warning(
                    colored(
                        f"{_d_msg} ⚠️  BUT one is > {_VR_DERIVED_GAP_WARN_M:.2f} m, which means the "
                        "completion is not producing a STANDING body — check that all three pucks "
                        "are actually being WORN and tracked before you arm.",
                        "yellow",
                    )
                )
            else:
                self.logger.info(colored(_d_msg, "cyan"))
        return ok

    @staticmethod
    def _match_len(live: np.ndarray, ref: np.ndarray) -> np.ndarray:
        """Return a target the SAME length as ``ref``, taking leading values from
        ``live`` and holding the tail at ``ref`` when ``live`` is shorter.

        Guards the obs width against a live frame whose point count differs from the
        loaded checkpoint (e.g. a 2-tracker frame reaching a 3pt hold, or a 3-tracker
        frame reaching a 2pt hold): the obs vector MUST stay the trained width, so we
        never grow past ``ref`` and never shrink below it. Equal length → live as-is.
        """
        live = np.asarray(live, dtype=np.float32)
        ref = np.asarray(ref, dtype=np.float32)
        if live.shape == ref.shape:
            return live.copy()
        out = ref.astype(np.float32).copy()
        n = min(live.shape[0], ref.shape[0])
        out[:n] = live[:n]
        return out

    def _apply_clutch_delta(self, ref: np.ndarray, live: np.ndarray, live0: np.ndarray) -> np.ndarray:
        """Relative-clutch position target = ref + (live - live0), length-matched to
        ``ref``.

        The tracker displacement since the anchor drives each point present in the
        live frame; points beyond the live frame's length hold the FK reference (so a
        2-tracker live frame on a 3pt hold moves the hands and freezes the head at
        ref, and a 3-tracker frame on a 2pt hold only ever touches the 2 hand slots).
        Zero-jump preserved: live==live0 ⇒ delta 0 ⇒ target==ref.
        """
        ref = np.asarray(ref, dtype=np.float32)
        live = np.asarray(live, dtype=np.float32)
        live0 = np.asarray(live0, dtype=np.float32)
        out = ref.copy()
        n = min(live.shape[0], live0.shape[0], ref.shape[0])
        out[:n] = ref[:n] + (live[:n] - live0[:n])
        return out.astype(np.float32)

    def _vr_slewed_points(self, n_pts: int) -> int:
        """How many LEADING keypoints the per-tick caps cover (the rest pass through un-slewed).

        ``n_pts`` for every preset and every setting except 5pt with ``HOLOSOMA_VIVE_SLEW_FEET=0``,
        where it is ``_VR_5POINT_OPERATOR_PLACED`` (3): the two feet are exempt. Published on
        RecordTap as the per-point ``vr_cmd_slewed`` mask so the cap can still be MEASURED off
        ``vr_cmd_pos`` — over the covered points only.
        """
        n_pts = int(n_pts)
        if self._is_5pt and not getattr(self, "_vive_slew_feet", True):
            return min(_VR_5POINT_OPERATOR_PLACED, n_pts)
        return n_pts

    def _slew_keypoints(
        self,
        prev_pos: np.ndarray,
        target_pos: np.ndarray,
        prev_orn: np.ndarray,
        target_orn: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """``_slew_vec`` on the keypoint vectors, restricted to the points ``_vr_slewed_points`` covers.

        Byte-identical to two bare ``_slew_vec`` calls whenever every point is covered (the
        default). With the feet exempt, the position pool is the leading ``3*3`` entries and the
        orientation pool the leading ``3*w`` (``w`` = 4 for the quaternion rep, 6 for rot6d); the
        remaining entries are the TARGET, copied verbatim. Falls back to the whole-vector slew if
        the widths do not split cleanly into points (a mismatched live frame), so a guard here can
        never widen the un-slewed set by accident.
        """
        n_pts = target_pos.size // 3
        n_slew = self._vr_slewed_points(n_pts)
        if n_slew >= n_pts or n_pts <= 0 or target_pos.size != 3 * n_pts or target_orn.size % n_pts != 0:
            return (
                self._slew_vec(prev_pos, target_pos, self._vive_max_pos_step),
                self._slew_vec(prev_orn, target_orn, self._vive_max_orn_step),
            )
        w = target_orn.size // n_pts
        pos = target_pos.astype(np.float32).copy()
        orn = target_orn.astype(np.float32).copy()
        pos[: 3 * n_slew] = self._slew_vec(prev_pos[: 3 * n_slew], target_pos[: 3 * n_slew], self._vive_max_pos_step)
        orn[: w * n_slew] = self._slew_vec(prev_orn[: w * n_slew], target_orn[: w * n_slew], self._vive_max_orn_step)
        return pos, orn

    @staticmethod
    def _slew_vec(prev: np.ndarray, target: np.ndarray, max_step: float) -> np.ndarray:
        """Move ``prev`` toward ``target`` by at most ``max_step`` (L2) this tick.

        Caps the per-tick change so a far live target ramps in over several ticks
        instead of stepping. ``max_step <= 0`` disables the cap (hard follow).
        Returns a fresh float32 array (never mutates the inputs).
        """
        delta = target - prev
        dist = float(np.linalg.norm(delta))
        if max_step <= 0.0 or dist <= max_step or dist == 0.0:
            return target.astype(np.float32).copy()
        return (prev + delta * (max_step / dist)).astype(np.float32)

    @staticmethod
    def _rot6d_to_R(r6: np.ndarray) -> np.ndarray:
        """Rebuild a 3x3 rotation matrix from a 6D rep (first two columns).

        Gram-Schmidt orthonormalization, the standard inverse of the
        ``matrix_from_quat(q)[..., :2].reshape(-1)`` rep ``_compute_vr_fk`` emits.
        """
        m = r6.reshape(3, 2).astype(np.float64)
        b0 = m[:, 0]
        n0 = np.linalg.norm(b0)
        c0 = b0 / n0 if n0 > 1e-8 else np.array([1.0, 0.0, 0.0])
        b1 = m[:, 1]
        c1 = b1 - (c0 @ b1) * c0
        n1 = np.linalg.norm(c1)
        c1 = c1 / n1 if n1 > 1e-8 else np.array([0.0, 1.0, 0.0])
        c2 = np.cross(c0, c1)
        return np.stack([c0, c1, c2], axis=1)

    @staticmethod
    def _R_to_rot6d(R: np.ndarray) -> np.ndarray:
        """First two columns of R, flattened — the 6D rep (inverse of _rot6d_to_R)."""
        return R[:, :2].reshape(-1).astype(np.float32)

    def _orn_clutch_target(self, ref6n: np.ndarray, live6n: np.ndarray, live0_6n: np.ndarray) -> np.ndarray:
        """Relative-orientation target: per body, R_target = (R_live · R_live0ᵀ) · R_ref.

        The body's rotation SINCE the anchor (``R_live·R_live0ᵀ``, both in the
        pelvis frame) is applied on top of the held FK reference orientation, so
        arming is zero-jump (live==live0 ⇒ target==ref) and recenter returns to
        ref. All three inputs are the SAME length, N bodies × rot6d: 12D for the
        2pt family ([L,R]), 18D for the 3pt 6D preset ([L,R,head],
        exp:g1-wbt-3pt-hybrid-global-vel). Returns float32 of the same length.
        """
        n_bodies = ref6n.shape[0] // 6
        out = np.zeros(n_bodies * 6, dtype=np.float32)
        for h in range(n_bodies):
            s = slice(h * 6, h * 6 + 6)
            r_ref = self._rot6d_to_R(ref6n[s])
            r_live = self._rot6d_to_R(live6n[s])
            r_live0 = self._rot6d_to_R(live0_6n[s])
            r_target = (r_live @ r_live0.T) @ r_ref
            out[s] = self._R_to_rot6d(r_target)
        return out

    @staticmethod
    def _quat_xyzw_to_R(q: np.ndarray) -> np.ndarray:
        """3×3 rotation from a unit xyzw quaternion (the 3pt vr_3point_orn rep).

        Matches the xyzw convention ``_compute_vr_fk`` emits for the 3pt family
        (``pin.Quaternion(R).coeffs()`` → [x,y,z,w]). Renormalizes defensively.
        """
        q = np.asarray(q, dtype=np.float64)
        n = float(np.linalg.norm(q))
        if n < 1e-8:
            return np.eye(3)
        x, y, z, w = q / n
        return np.array(
            [
                [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
            ]
        )

    @staticmethod
    def _R_to_quat_xyzw(R: np.ndarray) -> np.ndarray:
        """Unit xyzw quaternion from a 3×3 rotation (inverse of _quat_xyzw_to_R).

        Shepperd's method (numerically stable across all trace regimes). Returns
        xyzw to match the 3pt obs rep, CANONICALIZED to ``w >= 0``.

        The sign is not cosmetic. ``q`` and ``-q`` are the same rotation but two very
        different obs vectors, and every quaternion-orn preset trained exclusively on the
        ``+w`` branch (na475b60's normalizer: ``vr_5point_orn`` w mean +0.73 hands / +0.98
        head / +0.96 feet, std <= 0.17). The clutch itself is sign-immune — it goes through
        rotation MATRICES, and ``R(q) == R(-q)`` — so an antipode on the wire cannot reach
        the obs through it; but this function is the last producer on that path, and its
        ``t > 0`` branch is the only one that yields ``w > 0`` for free. Canonicalizing makes
        the invariant structural instead of incidental.
        """
        R = np.asarray(R, dtype=np.float64)
        t = np.trace(R)
        if t > 0.0:
            s = np.sqrt(t + 1.0) * 2.0
            w = 0.25 * s
            x = (R[2, 1] - R[1, 2]) / s
            y = (R[0, 2] - R[2, 0]) / s
            z = (R[1, 0] - R[0, 1]) / s
        elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
            w = (R[2, 1] - R[1, 2]) / s
            x = 0.25 * s
            y = (R[0, 1] + R[1, 0]) / s
            z = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
            w = (R[0, 2] - R[2, 0]) / s
            x = (R[0, 1] + R[1, 0]) / s
            y = 0.25 * s
            z = (R[1, 2] + R[2, 1]) / s
        else:
            s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
            w = (R[1, 0] - R[0, 1]) / s
            x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s
            z = 0.25 * s
        q = np.array([x, y, z, w], dtype=np.float32)
        q = q / (np.linalg.norm(q) + 1e-12)
        # Canonicalize to w >= 0. Shepperd's `t > 0` branch already yields w > 0, but the
        # three trace-negative branches (rotations past ~90 deg) do not, so the guarantee
        # was incidental rather than structural. Making it unconditional here means EVERY
        # producer of a quaternion-orn obs term agrees on the branch the policies trained on
        # (see _compute_vr_fk and bake_motions._fk_into for the measured consequence of
        # getting it wrong: a ~20-sigma obs and the arms snapping to a nonsense pose).
        return -q if q[3] < 0.0 else q

    @classmethod
    def _quat12_to_rot6d18(cls, orn12: np.ndarray) -> np.ndarray:
        """Convert 3 xyzw quaternions (12D, the 3pt VIVE-bridge wire rep) into
        3 rot6d blocks (18D, the `vr_3point_orn_6d` obs rep).

        The 6D-orn 3pt preset (exp:g1-wbt-3pt-hybrid-global-vel) trains on the
        first-two-columns rotation rep, but the vive_teleop_bridge 3pt wire format
        stays 3×quaternion (so the SAME bridge serves the quat and 6D checkpoints).
        We convert AT INGEST — per body q → R (``_quat_xyzw_to_R``) → R[:, :2]
        (``_R_to_rot6d``) — so the whole downstream orn path (clutch anchor, clutch
        target, slew, manifold reprojection) runs purely in the obs rep, exactly
        like the 2pt rot6d family. R(q) == R(−q), so the quaternion double-cover
        dies here and never reaches the obs (the training-side rationale for 6D).
        """
        out = np.zeros(18, dtype=np.float32)
        for b in range(3):
            out[b * 6 : b * 6 + 6] = cls._R_to_rot6d(cls._quat_xyzw_to_R(orn12[b * 4 : b * 4 + 4]))
        return out

    @staticmethod
    def _align_quat_signs(target: np.ndarray, prev: np.ndarray) -> np.ndarray:
        """Per body (N × xyzw), flip ``target``'s sign to the SAME hemisphere as
        ``prev`` (dot ≥ 0) so a linear quaternion slew never crosses the double
        cover (which would lerp through ~0 and renormalize to a wild rotation).

        q and −q are the same rotation, so flipping the sign is free — it only picks
        the shortest-path representative for the interpolation. Returns a fresh
        float32 array. Quaternion reps only (3pt = 3 bodies / 12D, 5pt = 5 bodies /
        20D); the 2pt rot6d rep has no double cover and never reaches here. N is read
        off the array rather than hardcoded, which is what let 5pt inherit this
        unchanged."""
        out = np.asarray(target, dtype=np.float32).copy()
        prev = np.asarray(prev, dtype=np.float32)
        for b in range(out.size // 4):
            s = slice(b * 4, b * 4 + 4)
            if float(np.dot(out[s], prev[s])) < 0.0:
                out[s] = -out[s]
        return out

    def _orn_clutch_target_quat(self, ref: np.ndarray, live: np.ndarray, live0: np.ndarray) -> np.ndarray:
        """Quaternion analogue of ``_orn_clutch_target``: per body,
        R_target = (R_live · R_live0ᵀ) · R_ref, re-expressed as an xyzw quaternion.

        The ``vr_{3,5}point_orn`` rep is a RAW xyzw quaternion (N bodies × 4 — 12D for
        3pt, 20D for 5pt), NOT the 2pt rot6d. We convert each body's quat to a matrix,
        apply the same world-frame relative delta (rotation SINCE the anchor,
        ``R_live·R_live0ᵀ``, both pelvis-frame) on top of the held FK reference, then
        re-extract xyzw. Zero-jump: live==live0 ⇒ delta=I ⇒ target==ref. All three inputs
        must be the same width; the output matches ``ref``.
        """
        out = np.zeros(np.asarray(ref).size, dtype=np.float32)
        for b in range(out.size // 4):
            s = slice(b * 4, b * 4 + 4)
            r_ref = self._quat_xyzw_to_R(ref[s])
            r_live = self._quat_xyzw_to_R(live[s])
            r_live0 = self._quat_xyzw_to_R(live0[s])
            r_target = (r_live @ r_live0.T) @ r_ref
            out[s] = self._R_to_quat_xyzw(r_target)
        return out

    def _dense_joint_perm(self, wire_names: tuple[str, ...]) -> np.ndarray | None:
        """Index map from this robot's ``dof_names`` onto the wire's joint order, BY NAME.

        ``perm[i]`` is the wire index holding policy DOF ``i``, or ``-1`` when the wire
        carries no joint of that name (that slot then HOLDS the reference — see
        :meth:`_dense_live_vectors`). Cached on the wire's name tuple, so the lookup runs
        once per bridge session rather than per tick.

        Returns None (→ the caller holds the reference, no override) when the robot config
        has no usable ``dof_names``: without them there is no honest way to place the
        wire's 29 numbers, and index-aligning two independent real joint orders would
        scramble the body with no error raised anywhere. On the CURRENT data this resolves
        to the identity permutation (the completion's ``g1_fk_params.npz`` order and
        holosoma's ``dof_names`` happen to agree exactly) — which is precisely why the
        remap must stay: nothing would notice if that coincidence ever ended.
        """
        if self._dense_perm_src == wire_names and self._dense_name_perm is not None:
            return self._dense_name_perm
        dof_names = list(getattr(self.config.robot, "dof_names", []) or [])
        if len(dof_names) != self.num_dofs:
            logger.error(
                colored(
                    f"DENSE teleop: robot dof_names has {len(dof_names)} entries, expected "
                    f"{self.num_dofs} — cannot remap the completion's joint targets by name. "
                    "HOLDING the reference (no live override).",
                    "red",
                )
            )
            return None
        wire_idx = {n: k for k, n in enumerate(wire_names)}
        perm = np.full(len(dof_names), -1, dtype=np.int64)
        missing: list[str] = []
        for i, n in enumerate(dof_names):
            k = wire_idx.get(n, -1)
            perm[i] = k
            if k < 0:
                missing.append(n)
        self._dense_name_perm = perm
        self._dense_perm_src = wire_names
        self._dense_perm_missing = missing
        if missing:
            logger.warning(
                colored(
                    f"DENSE teleop: {len(missing)} joint(s) absent from the completion's stream "
                    f"({', '.join(missing[:6])}{' …' if len(missing) > 6 else ''}) — those DOFs HOLD "
                    "the reference pose instead of tracking.",
                    "yellow",
                )
            )
        elif np.array_equal(perm, np.arange(len(dof_names))):
            logger.info(
                colored(
                    "DENSE teleop: the completion's joint order matches dof_names exactly "
                    "(identity remap) — still resolved by NAME, never index-aligned.",
                    "cyan",
                )
            )
        else:
            logger.info(colored(f"DENSE teleop: joint remap resolved by name, permutation {perm.tolist()}", "cyan"))
        return perm

    def _dense_live_vectors(self) -> tuple[np.ndarray, np.ndarray] | None:
        """The wire's latest joint targets/velocities in POLICY dof order, or None when no
        fresh dense frame is available (stale / never connected / unmappable).

        DOFs the wire does not supply are filled with the HELD REFERENCE, not zeros — so
        they contribute a zero clutch delta and simply hold, exactly as the keypoint path's
        ``_match_len`` holds the tail of a short live frame at the FK reference.
        """
        got = self._vive_live.latest_joints()
        if got is None:
            return None
        wire_q, wire_dq, wire_names = got
        perm = self._dense_joint_perm(wire_names)
        if perm is None:
            return None
        ref_q = self._motion_command_t[0, : self.num_dofs].astype(np.float32)
        live_q = ref_q.copy()
        live_dq = np.zeros(self.num_dofs, dtype=np.float32)
        have = perm >= 0
        live_q[have] = wire_q[perm[have]]
        live_dq[have] = wire_dq[perm[have]]
        return live_q, live_dq

    def _ref_joints_perm_for(self, wire_names: tuple[str, ...]) -> np.ndarray | None:
        """Index map from this robot's ``dof_names`` onto a ``ref_joints`` frame's joint order,
        BY NAME (``perm[i]`` = wire index of ``dof_names[i]``, −1 if absent). Cached per names
        tuple. None (warned once) when ``dof_names`` is unusable or any joint the
        ``command_lower_body`` slice reads is missing — a PARTIAL lower-body reference would mix
        the walk's legs with the frozen clip's, so the whole override is refused instead."""
        wire_names = tuple(str(n) for n in wire_names)
        # getattr: a stub built around __init__ (the deploy tests) reaches here too.
        cached = getattr(self, "_ref_joints_perm", None)
        if cached is not None and getattr(self, "_ref_joints_perm_src", None) == wire_names:
            return cached
        dof_names = list(getattr(self.config.robot, "dof_names", []) or [])
        if len(dof_names) != self.num_dofs:
            if not getattr(self, "_ref_joints_warned", False):
                self._ref_joints_warned = True
                logger.warning(
                    colored(
                        f"ref_joints: robot dof_names has {len(dof_names)} entries, want {self.num_dofs} — "
                        "cannot place the wire's reference joints by name; command_lower_body keeps the "
                        "frozen clip. (Warned once.)",
                        "red",
                    )
                )
            return None
        index = {n: i for i, n in enumerate(wire_names)}
        perm = np.array([index.get(n, -1) for n in dof_names], dtype=np.int64)
        missing_lower = [dof_names[i] for i in _LOWER_BODY_INDICES if perm[i] < 0]
        if missing_lower:
            if not getattr(self, "_ref_joints_warned", False):
                self._ref_joints_warned = True
                logger.warning(
                    colored(
                        f"ref_joints: the wire's joint names lack {len(missing_lower)} of the joints "
                        f"command_lower_body reads ({', '.join(missing_lower[:6])}"
                        f"{' …' if len(missing_lower) > 6 else ''}) — REFUSING the override; "
                        "command_lower_body keeps the frozen clip. (Warned once.)",
                        "red",
                    )
                )
            return None
        self._ref_joints_perm = perm
        self._ref_joints_perm_src = wire_names
        return perm

    def _resolve_ref_joints(self, *, live_fresh: bool) -> np.ndarray | None:
        """The ``(58,)`` reference joints (29 targets ++ 29 velocities, policy dof order) that
        ``command_lower_body`` should be sliced from THIS tick, or None for the frozen clip.

        5pt + ``HOLOSOMA_VIVE_REF_JOINTS`` + a live source + ARMED, else None (and nothing held).
        With a FRESH keypoint frame: the frame's ``ref_joints`` remapped by name — or None when
        the frame carries none (the byte-identical path; the walk skill stops sending the field
        when its lead-out ramp has landed on the ready legs, which is also what the frozen clip
        holds). With a STALE frame: the last override fed, so a bridge hiccup mid-walk freezes
        the legs' reference together with the feet instead of snapping it to the ready stance.
        Never slewed: the override is a REFERENCE for the obs, and a 5 mm/tick-style cap on a
        gait's 18 rad/s joint velocities would be the crawl the feet were exempted from. DOFs the
        wire does not name hold the frozen clip's value with zero velocity (none of them is read
        by command_lower_body — a missing lower-body joint refuses the whole override).
        """
        if not (
            self._is_5pt
            and getattr(self, "_vive_ref_joints", True)
            and self._vive_live is not None
            and self._vive_armed
        ):
            self._ref_joints_last = None
            return None
        if not live_fresh:
            return getattr(self, "_ref_joints_last", None)  # stale: freeze (None before any override)
        latest_ref = getattr(self._vive_live, "latest_ref_joints", None)
        got = latest_ref() if callable(latest_ref) else None
        if got is None:
            self._ref_joints_last = None
            return None
        wire_q, wire_dq, wire_names = got
        perm = self._ref_joints_perm_for(wire_names)
        if perm is None or wire_q.size != len(wire_names) or wire_dq.size != len(wire_names):
            self._ref_joints_last = None
            return None
        ref_q = self._motion_command_t[0, : self.num_dofs].astype(np.float32).copy()
        ref_dq = np.zeros(self.num_dofs, dtype=np.float32)
        have = perm >= 0
        ref_q[have] = wire_q[perm[have]]
        ref_dq[have] = wire_dq[perm[have]]
        self._ref_joints_last = np.concatenate([ref_q, ref_dq]).astype(np.float32)
        return self._ref_joints_last

    def _live_dense_command(self) -> np.ndarray | None:
        """The 58D ``command`` term (29 joint targets ++ 29 velocities) under live DENSE
        teleop, or None to fall through to the held reference.

        Mirrors the keypoint override's contract exactly:

        * RELATIVE / clutch (default): ``target = ref + (live - live0)``, so arming is a
          zero-jump operation no matter what posture the operator's trackers imply.
        * ABSOLUTE: the completion's joint solution IS the target, plus the arm-time offset
          ``ref - live_at_arm`` when the arm clutch is on (again zero-jump, no posture bias).
        * SLEW: every joint target moves at most ``_vive_max_joint_step`` rad per tick
          (PER JOINT, not L2 — see the note at the clip), so arm / recenter / stale→fresh
          all RAMP instead of stepping.
        * STALE while armed: FREEZE at the last fed target. Never the raw wire, never zeros
          — 29 zero joint targets is the G1 folded flat.

        VELOCITY: the completion's own smooth derivative in steady state, but ZERO whenever
        the slew is binding or we are frozen — a command that is ramping (or held) must not
        also claim it is moving at the live rate. Same reasoning as the frozen-clip guard on
        ``motion_ref_vel_b`` and the velocity half of ``_apply_hold_dof_override``: an
        internally inconsistent pos/vel pair is off-distribution for the policy.
        """
        ref_q = self._motion_command_t[0, : self.num_dofs].astype(np.float32)
        got = self._dense_live_vectors()
        if got is None:
            if self._dense_last_q is None:
                return None  # first-tick staleness → the held reference, byte-identical
            return np.concatenate([self._dense_last_q, np.zeros(self.num_dofs, dtype=np.float32)]).astype(np.float32)
        live_q, live_dq = got

        if self._vive_relative:
            if self._dense_clutch_q0 is None:
                logger.warning(
                    colored("DENSE teleop: armed without a clutch anchor — holding reference (no override).", "red")
                )
                target_q = ref_q
            else:
                target_q = ref_q + (live_q - self._dense_clutch_q0)
        else:
            target_q = live_q
            _off = self._dense_abs_off_q
            if _off is not None and _off.shape == target_q.shape:
                target_q = target_q + _off

        if self._dense_last_q is None:
            # First fed frame: ramp from the held reference so we glide onto the target.
            self._dense_last_q = ref_q.copy()
        # PER-JOINT rate limit, deliberately NOT `_slew_vec`. That helper caps the L2 norm
        # of the whole vector, which is right for a task-space target (a hand has one speed)
        # but wrong here on both counts: it COUPLES the joints, so one fast wrist throttles
        # the legs, and 29 joints each moving at a normal teleop rate blow an L2 budget that
        # any single joint respects (two arm joints at 2 rad/s already exceed 2.5 rad/s L2).
        # An L2 cap would therefore bind during ordinary motion — rate-limiting the operator
        # and, via the guard below, pinning the reference velocity at zero the whole time.
        # 0.05 rad/tick at 50 Hz = 2.5 rad/s PER JOINT, well above teleop speeds and well
        # below the G1's joint limits, so it bites only on transitions.
        step = max(self._vive_max_joint_step, 0.0)
        delta = target_q - self._dense_last_q
        if step > 0.0:
            binding = bool(np.any(np.abs(delta) > step))
            q = (self._dense_last_q + np.clip(delta, -step, step)).astype(np.float32)
        else:
            binding = False  # cap disabled → hard follow
            q = target_q.astype(np.float32)
        # Zero the reference velocity while the ramp is binding (see the docstring).
        dq = np.zeros(self.num_dofs, dtype=np.float32) if binding else live_dq
        self._dense_last_q = q.copy()
        self._dense_last_dq = dq.copy()
        return np.concatenate([q, dq]).astype(np.float32)

    def _compute_obs_terms(self, robot_state_data) -> dict[str, np.ndarray]:
        """Build observation terms dynamically from config term names.

        Ensures sim2sim and sim2real produce identical observations
        for any obs mode (dense, global, global_vel, 3pt).

        No joint reordering: holosoma training order == SDK order.
        dof_pos is offset by default_dof_angles (standing pose), matching training.
        """
        dof_pos = robot_state_data[0, 7 : 7 + self.num_dofs]
        base_ang_vel = robot_state_data[0, 7 + self.num_dofs + 3 : 7 + self.num_dofs + 6]
        dof_vel = robot_state_data[0, 7 + self.num_dofs + 6 : 7 + self.num_dofs + 6 + self.num_dofs]
        root_pos = robot_state_data[0, 0:3]
        root_quat_wxyz = robot_state_data[0, 3:7].reshape(1, 4)

        # Read the live F/T (Bota) once per obs build → cache the pelvis-frame
        # measured force + (servo-on) the integrator I / setpoint shift Δx / per-DOF
        # servo Δq. No-op (all zero) when no live source is configured, so the
        # force_meas_local / force_ext_local / integrator_state / x_tgt_shift terms
        # below stay zero and the path is byte-identical.
        self._update_force_meas_and_servo(dof_pos)

        # Lazy heading capture (same pattern as TML): on the very first
        # rl_inference call after _reset_motion_state, capture robot and motion
        # yaw so _remove_yaw_offset strips initial heading from both.
        # Training does NOT strip initial yaw (it uses subtract_frame_transforms
        # on instantaneous world-frame quats), so for terrain-style presets
        # (non-pelvis anchor, insertion-ordered obs) skipping the strip avoids a
        # constant step-0 misalignment relative to what the policy saw in
        # training. Pelvis-anchored presets keep the strip for mocap drift.
        _strip_yaw = self._ref_body_is_pelvis
        if _strip_yaw and not self._heading_initialized:
            self._capture_robot_yaw_offset_holosoma()
            motion_wxyz = xyzw_to_wxyz(self._ref_quat_xyzw_t)
            self.motion_yaw_offset = self._quat_yaw(motion_wxyz)
            self._heading_initialized = True

        # Relative orientation (shared by all modes that include motion_ref_ori_b)
        motion_ref_ori = xyzw_to_wxyz(self._ref_quat_xyzw_t)
        if self._ref_body_is_pelvis:
            robot_ref_ori_raw = root_quat_wxyz
            robot_ref_pos = root_pos
        else:
            # motion_ref_pos_b / motion_ref_ori_b anchor on the training
            # body_name_ref (e.g. torso_link), not the floating-root pelvis.
            robot_ref_pos, _robot_ref_ori = self._get_ref_body_pose_in_world(robot_state_data)
            robot_ref_ori_raw = _robot_ref_ori.reshape(1, 4)
        if _strip_yaw:
            motion_ref_ori = self._remove_yaw_offset(motion_ref_ori, self.motion_yaw_offset)
            robot_ref_ori = self._remove_yaw_offset(robot_ref_ori_raw, self.robot_yaw_offset)
        else:
            robot_ref_ori = robot_ref_ori_raw

        # Pre-compute VR FK if any VR terms present (avoids duplicate FK calls)
        vr_pos_local = vr_orn_local = None
        # The walk skill's reference-joint override for command_lower_body (5pt, armed, live —
        # see HOLOSOMA_VIVE_REF_JOINTS). None on every other path: the frozen clip, byte-identical.
        ref_override: np.ndarray | None = None
        self._ref_joints_applied = None
        if self._uses_vr_keypoints:
            ref_joint_pos = self._motion_command_t[0, :29]
            offsets = self._vr_offsets()
            vr_pos_local, vr_orn_local = self._compute_vr_fk(ref_joint_pos, offsets)

        # LIVE VIVE override: drive the VR target from the operator's live trackers.
        # The bridge streams pos/orn in the SAME pelvis frame _compute_vr_fk just
        # produced. latest() returns (None, None) when stale / not yet connected →
        # we FREEZE at the last fed target (first-tick staleness keeps the NPZ-FK
        # target; never command zeros). Works for the 2pt (pos[6], orn[12]=rot6d)
        # and 3pt (pos[9], orn[12]=3×quaternion) families — the QUAT 3pt presets
        # consume the wire orn as-is; the 6D preset (vr_3point_orn_6d,
        # exp:g1-wbt-3pt-hybrid-global-vel) converts quat→rot6d at ingest so the
        # clutch/slew/reprojection run in the obs rep. The position clutch/slew is
        # dimension-agnostic; only the ORIENTATION clutch + manifold reprojection
        # branch on the representation (rot6d vs quaternion).
        #
        # RELATIVE / clutch mode (default): the commanded target is
        #   ref + (live_now - live0)
        # where (ref, live0) is the anchor snapped at arm/recenter. So the target
        # starts exactly at the held reference (delta 0) and follows only the
        # operator's motion SINCE the anchor — trackers can sit anywhere, arming
        # never jumps. Orientation stays the frozen NPZ-FK reference unless the orn
        # clutch is enabled (then relative in the matching rep).
        # ABSOLUTE mode: the tracker pose IS the target (legacy path).
        #
        # We SLEW (cap per-tick change) rather than hard-swap so any first-frame /
        # reconnect / stale→fresh transition ramps instead of stepping (no arm
        # jerk on hardware). `_vr_last_*` tracks the target we actually fed last
        # tick so the ramp starts from where we are, not from a stale anchor.
        if not (self._vive_live is not None and self._uses_vr_keypoints and self._vive_armed):
            # Not driving the keypoints from the wire this tick (un-armed / no source): nothing of
            # the walk skill's reference legs may survive into the next armed session either.
            self._ref_joints_last = None
        if self._vive_live is not None and self._uses_vr_keypoints and self._vive_armed:
            # _uses_vr_keypoints guarantees the FK block above ran, so these are non-None.
            assert vr_pos_local is not None and vr_orn_local is not None
            live_pos, live_orn = self._vive_live.latest()
            # Resolved from the SAME freshness the keypoints use: a fresh frame with the field
            # overrides, a fresh frame without it clears, a stale frame freezes the last fed one.
            ref_override = self._resolve_ref_joints(live_fresh=live_pos is not None)
            if live_pos is not None:
                # latest() sets pos and orn together (None together).
                assert live_orn is not None
                # 6D-orn 3pt preset (exp:g1-wbt-3pt-hybrid-global-vel): the bridge's
                # 3pt wire orn is 3×xyzw-quaternion (12D), but the obs rep is 3×rot6d
                # (18D). Convert AT INGEST (guarded on a genuine 3pt frame,
                # pos[9]+orn[12], so a 2pt frame's 2×rot6d bytes are never
                # misread as quats) — downstream (clutch/slew/reproject) then runs
                # purely in the obs rep, like the 2pt rot6d family. The plain-3pt
                # quat preset and the 2pt family skip this (byte-identical).
                if self._is_3pt and self._vr_orn_6d and live_pos.size == 9 and live_orn.size == 12:
                    live_orn = self._quat12_to_rot6d18(live_orn)
                if self._vive_relative:
                    # Relative/clutch: target = held FK ref + tracker displacement
                    # since the anchor. Re-base on the CURRENT FK frame each tick
                    # (the clip is frozen while armed, so it's constant) and add only
                    # the tracker delta. If the anchor is somehow missing while armed
                    # (should never happen — every relative arm snapshots first), DO
                    # NOT fall through to the absolute tracker pose (that would glide
                    # the target to wherever the trackers physically are). Hold the FK
                    # ref + warn instead. The position delta spans all points present
                    # in the live frame (2pt: 6D; 3pt with a head tracker: 9D); if the
                    # live frame is SHORTER than the FK ref (e.g. a 2-tracker frame on
                    # a 3pt hold), only the leading points are overridden and the rest
                    # hold the FK ref.
                    if self._vr_clutch_live0 is None:
                        logger.warning(
                            colored(
                                "VIVE relative: armed without a clutch anchor — holding reference (no override).", "red"
                            )
                        )
                        target_pos = vr_pos_local
                    else:
                        target_pos = self._apply_clutch_delta(vr_pos_local, live_pos, self._vr_clutch_live0)
                    # Orientation: relative clutch when enabled + an orn origin was
                    # captured + the bridge is sending orn; else keep the FK ref
                    # (position-only teleop, no orn jump). Rep depends on family:
                    #   * quat presets (3pt, 5pt) → _orn_clutch_target_quat (N×xyzw quats)
                    #   * 2pt family / 3pt 6D preset → _orn_clutch_target (N×rot6d;
                    #     live/anchor were quat→rot6d-converted at ingest). Guard on
                    #     matching lengths so a cross-family live frame (e.g. a 2pt
                    #     rot6d frame reaching a 3pt-6D hold) HOLDS the FK ref
                    #     instead of composing mismatched reps.
                    if self._vive_orientation and self._vr_clutch_orn0 is not None and live_orn is not None:
                        # CROSS-FAMILY GUARD: the orn wire rep is only trustworthy
                        # when the live frame's POINT COUNT matches BOTH the hold
                        # AND the anchor frame's point count (positions are
                        # rep-compatible xyz, orientations are NOT: a 3pt frame's
                        # 3×quat[12] byte-collides with the 2pt 2×rot6d[12], and a
                        # 2pt frame's rot6d is not quats — widths alone cannot
                        # disambiguate, so we key on the pos width, including the
                        # ANCHOR's pos width `_vr_clutch_live0.size` to catch a
                        # bridge whose MODE was swapped mid-armed). A bridge in the
                        # wrong MODE therefore holds the FK ref orientation (warn
                        # once per anchor) instead of composing mismatched reps.
                        _orn_ok = (
                            live_pos.size == vr_pos_local.size
                            and self._vr_clutch_live0 is not None
                            and live_pos.size == self._vr_clutch_live0.size
                        )
                        if _orn_ok and self._vr_orn_is_quat:
                            target_orn = self._orn_clutch_target_quat(vr_orn_local, live_orn, self._vr_clutch_orn0)
                        elif (
                            _orn_ok
                            and live_orn.size == vr_orn_local.size
                            and self._vr_clutch_orn0.size == vr_orn_local.size
                        ):
                            target_orn = self._orn_clutch_target(vr_orn_local, live_orn, self._vr_clutch_orn0)
                        else:
                            if not self._orn_mismatch_warned:
                                self._orn_mismatch_warned = True
                                logger.warning(
                                    colored(
                                        "VIVE orn clutch: live frame does not match the obs rep "
                                        f"(live pos {live_pos.size}D/orn {live_orn.size}D, "
                                        f"anchor {self._vr_clutch_orn0.size}D, obs pos {vr_pos_local.size}D/"
                                        f"orn {vr_orn_local.size}D) — holding the FK reference orientation. "
                                        "(Bridge in the wrong MODE? Warned once per anchor.)",
                                        "yellow",
                                    )
                                )
                            target_orn = vr_orn_local
                    else:
                        target_orn = vr_orn_local
                else:
                    # Absolute: the tracker pose is the target directly. Clip to the
                    # FK-ref length so a mismatched live-frame width can't reshape the
                    # obs (short → pad-hold the tail at ref; long → truncate).
                    target_pos = self._match_len(live_pos, vr_pos_local)
                    target_orn = self._match_len(live_orn, vr_orn_local)
                    # ABSOLUTE-mode ARM CLUTCH: add the arm-time offset FK(ref) − wire so
                    # the target holds the arm pose at arm (target == FK(ref) since wire ==
                    # wire_at_arm ⇒ +off ⇒ FK(ref)) and follows the wire DELTA afterwards —
                    # zero jump, no reach bias (see __init__/_toggle_vive_teleop). Shape
                    # guard: the offset is captured at the SAME vr-target length, so a
                    # mid-armed live-frame width change (already handled by _match_len)
                    # never mis-broadcasts; on mismatch skip the add (raw wire, pre-fix).
                    # getattr-guarded: the obs path must never raise on an object that predates
                    # these fields (partially-constructed policies / older pickled state).
                    _abs_off = getattr(self, "_vr_abs_off_pos", None)
                    if _abs_off is not None and _abs_off.shape == target_pos.shape:
                        target_pos = target_pos + _abs_off
                    # ORIENTATION arm clutch: rotations COMPOSE, so instead of adding an
                    # offset we apply the wire's rotation SINCE the arm anchor on top of the
                    # held FK reference — R_target = (R_wire · R_wire0ᵀ) · R_ref — reusing the
                    # SAME helpers the relative clutch uses (rep-specific: quat for plain-3pt,
                    # rot6d for the 2pt family and the 3pt 6D preset). At arm wire == wire0 ⇒
                    # delta = I ⇒ target == FK(ref) ⇒ ZERO orientation jump; afterwards the
                    # wrist follows only the wire's orn DELTA. Widths must match the obs rep on
                    # BOTH the live frame and the anchor (a 3pt 3×quat[12] byte-collides with a
                    # 2pt 2×rot6d[12], so width alone cannot disambiguate — key on the pos
                    # width like the relative path does); on any mismatch we HOLD the raw wire
                    # orn (the pre-fix behavior) rather than compose mismatched reps.
                    _a0 = getattr(self, "_vr_abs_orn0", None)  # see the getattr note above
                    if _a0 is not None and live_orn is not None:
                        _abs_orn_ok = (
                            live_pos.size == vr_pos_local.size
                            and live_orn.size == _a0.size
                        )
                        if _abs_orn_ok and self._vr_orn_is_quat and live_orn.size == vr_orn_local.size:
                            target_orn = self._orn_clutch_target_quat(vr_orn_local, live_orn, _a0)
                        elif _abs_orn_ok and live_orn.size == vr_orn_local.size and _a0.size == vr_orn_local.size:
                            target_orn = self._orn_clutch_target(vr_orn_local, live_orn, _a0)
                if self._vr_last_pos is None:
                    # First fed frame: ramp from the current NPZ-FK target so we
                    # glide onto the (relative or absolute) target, never jump.
                    self._vr_last_pos = vr_pos_local.copy()
                    self._vr_last_orn = vr_orn_local.copy()
                # 3pt DOUBLE-COVER GUARD (quaternion rep ONLY): _slew_vec LINEARLY
                # interpolates the raw 12D orn between _vr_last_orn and target_orn.
                # A quaternion q and its antipode −q are the SAME rotation, but the
                # raw lerp between them passes through ~0 → renormalizing a near-zero
                # vector yields a WILD rotation mid-slew (a real hardware jerk). So,
                # per body, flip target_orn into the SAME hemisphere as _vr_last_orn
                # (dot ≥ 0) BEFORE the slew — the shortest-path convention. The rot6d
                # reps (2pt AND the 3pt 6D preset) have NO double cover, so they are
                # left untouched (byte-identical).
                # The orn target MOVES either via the relative orn clutch (_vr_clutch_orn0) or
                # the ABSOLUTE arm clutch (_vr_abs_orn0). Both the double-cover guard below and
                # the manifold reprojection further down must fire in EITHER case: they were
                # previously keyed on the relative anchor alone, which is always None in
                # absolute mode — safe only while absolute orn never moved between ticks. Now
                # that the absolute arm clutch rotates the target, keying on the relative
                # anchor alone would slew a quaternion through the double cover (wild mid-slew
                # rotation) and feed an off-SO(3) rep to the ONNX.
                _orn_moves = (self._vive_orientation and self._vr_clutch_orn0 is not None) or (
                    getattr(self, "_vr_abs_orn0", None) is not None
                )
                if self._vr_orn_is_quat and _orn_moves:
                    target_orn = self._align_quat_signs(target_orn, self._vr_last_orn)
                # Both caps, over the points they cover (all of them unless HOLOSOMA_VIVE_SLEW_FEET=0
                # exempts the 5pt feet — see _slew_keypoints; byte-identical to two _slew_vec calls
                # otherwise).
                vr_pos_local, vr_orn_local = self._slew_keypoints(
                    self._vr_last_pos, target_pos, self._vr_last_orn, target_orn
                )
                # _slew_vec LINEARLY interpolates the raw rep, which leaves rot6d /
                # quaternion OFF the SO(3) manifold mid-ramp. The ONNX was trained on
                # clean rotations, so re-project each body back onto a valid rotation
                # before feeding the obs. Only needed when orientation actually MOVES
                # (orn clutch on); for the position-only path target_orn==prev==the
                # constant FK ref, so the slew is a no-op and we skip the reprojection
                # to keep that path byte-identical. Rep-specific: quat renorm for the
                # quaternion reps (3pt 3×xyzw, 5pt 5×xyzw); Gram-Schmidt rot6d
                # reprojection for the rot6d reps (2pt = 2×rot6d, 3pt 6D preset =
                # 3×rot6d — loop over however many blocks the obs carries). Fires for the ABSOLUTE
                # arm clutch too (see _orn_moves above) — it rotates the target, so its
                # slew leaves the rep off-manifold exactly like the relative clutch's.
                if _orn_moves:
                    if self._vr_orn_is_quat:
                        for _b in range(vr_orn_local.size // 4):
                            _s = slice(_b * 4, _b * 4 + 4)
                            _n = float(np.linalg.norm(vr_orn_local[_s]))
                            if _n > 1e-8:
                                vr_orn_local[_s] = vr_orn_local[_s] / _n
                    else:
                        for _h in range(vr_orn_local.size // 6):
                            _s = slice(_h * 6, _h * 6 + 6)
                            vr_orn_local[_s] = self._R_to_rot6d(self._rot6d_to_R(vr_orn_local[_s]))
            elif self._vr_last_pos is not None:
                # STALE while armed (bridge hiccup / died): FREEZE at the last fed
                # target — the ViveLiveSource contract ("the caller HOLDS its last
                # good target"). The previous fall-through snapped the target back
                # to the raw FK reference in ONE tick — an un-slewed jump of up to
                # the whole clutch delta (the exact arm-jerk every other transition
                # here is ramped to avoid). Freezing commands ZERO motion; a
                # stale→fresh recovery ramps from this held pose (slew below), and
                # the operator can always disarm/recenter. First-tick staleness
                # (_vr_last_pos is None) keeps the FK ref as before.
                vr_pos_local = self._vr_last_pos.copy()
                vr_orn_local = self._vr_last_orn.copy()
            # Remember what we fed (live-slewed, frozen-stale, OR the NPZ-FK target
            # on first-tick staleness), so a later stale→fresh recovery ramps from
            # the held pose.
            self._vr_last_pos = vr_pos_local.copy()
            self._vr_last_orn = vr_orn_local.copy()

        # LIVE DENSE-UNIFIED override: the joint-space twin of the keypoint block above.
        # exp:g1-wbt-hybrid-unified-anneal has NO keypoints — its tracking front-end is the
        # 58D `command` term (29 absolute joint targets ++ 29 velocities) — so the operator's
        # three trackers reach it as the whole-body joint solution the completion produced,
        # remapped onto this robot's dof order BY NAME. Same contract as the keypoint path
        # in every respect that matters on hardware: armed-only, relative clutch by default
        # (zero jump at arm), slew-capped, and FREEZE (never zero, never the raw wire) when
        # the bridge goes stale.
        dense_cmd = None
        if self._vive_live is not None and self._dense_joint_teleop and self._vive_armed:
            dense_cmd = self._live_dense_command()

        terms = {}
        for name in self._obs_terms:
            # --- Command terms ---
            if "command_lower_body" in name:
                if ref_override is not None:
                    # The wire's reference joints (policy dof order, remapped by name), sliced with
                    # the SAME indices the frozen clip is — so the term is what a baked clip would
                    # have fed (bake_motions' lower_body_jp/jv, pinned equal to this slice).
                    ref_jp = ref_override[:29]
                    ref_jv = ref_override[29:]
                else:
                    ref_jp = self._motion_command_t[0, :29]
                    ref_jv = self._motion_command_t[0, 29:]
                if self._is_5pt:
                    self._ref_joints_applied = ref_override is not None
                terms[name] = np.concatenate([ref_jp[_LOWER_BODY_INDICES], ref_jv[_LOWER_BODY_INDICES]]).astype(
                    np.float32
                )
            elif "vr_5point_pos" in name or "vr_3point_pos" in name or "vr_2point_pos" in name:
                terms[name] = vr_pos_local
            elif "vr_5point_orn" in name or "vr_3point_orn" in name or "vr_2point_orn" in name:
                terms[name] = vr_orn_local
            elif "motion_command" in name or name == "command":
                # `dense_cmd` is None on every non-teleop path (no live source, not armed,
                # not dense-unified, or no dense frame has arrived yet) — then this is the
                # byte-identical NPZ/held-clip reference it always was.
                if dense_cmd is not None:
                    terms[name] = dense_cmd
                else:
                    terms[name] = self._motion_command_t.reshape(-1).astype(np.float32)
            # --- Global tracking terms ---
            elif "motion_ref_pos_b" in name:
                # Anchor on the training ref body (torso for terrain preset,
                # pelvis otherwise). Training computes
                # subtract_frame_transforms(robot_ref_pos_w, robot_ref_quat_w,
                # motion_ref_pos_w, motion_ref_quat_w), i.e. the motion
                # ref-body pose relative to the robot ref-body frame — use
                # the raw (non-yaw-stripped) orientation here.
                ref_pos = self._get_aligned_ref_pos()
                diff = (ref_pos - robot_ref_pos).reshape(1, 3)
                ref_pos_b = quat_rotate_inverse(robot_ref_ori_raw, diff).reshape(-1)
                terms[name] = ref_pos_b.astype(np.float32)
            elif "motion_ref_vel_b" in name:
                # FROZEN-CLIP GUARD: while the clip is NOT progressing (STAND hold
                # after START, VIVE-armed teleop), the reference root is NOT
                # advancing — `_ref_pos_xyz_t` re-reads the SAME frame every tick —
                # so the only self-consistent reference velocity is ZERO. Feeding
                # the frozen frame's STORED velocity instead (nonzero when arming
                # mid-clip, or for a clip trimmed mid-stride) would tell the policy
                # "hold this root position" (motion_ref_pos_b) and "the reference
                # is moving at v" simultaneously — an off-distribution pos/vel
                # contradiction that can keep a walk-enabled policy leaning or
                # stepping while the operator believes it is parked. Exactly the
                # same rationale as _apply_hold_dof_override zeroing the joint-vel
                # half of command_lower_body. While the clip PLAYS, the true
                # aligned reference velocity flows unchanged.
                if not getattr(self, "motion_clip_progressing", False):
                    terms[name] = np.zeros(3, dtype=np.float32)
                else:
                    ref_vel = self._get_aligned_ref_vel()
                    ref_vel_b = quat_rotate_inverse(root_quat_wxyz, ref_vel.reshape(1, 3)).reshape(-1)
                    terms[name] = ref_vel_b.astype(np.float32)
            elif "base_lin_vel" in name:
                base_lin_vel_w = robot_state_data[0, 7 + self.num_dofs : 7 + self.num_dofs + 3]
                base_lin_vel_b = quat_rotate_inverse(root_quat_wxyz, base_lin_vel_w.reshape(1, 3)).reshape(-1)
                terms[name] = base_lin_vel_b.astype(np.float32)
            # --- Standard terms ---
            elif "motion_ref_ori_b" in name:
                motion_ref_ori_b = matrix_from_quat(subtract_frame_transforms(robot_ref_ori, motion_ref_ori))
                terms[name] = motion_ref_ori_b[..., :2].reshape(-1).astype(np.float32)
            elif "projected_gravity" in name:
                gravity_world = np.array([[0.0, 0.0, -1.0]])
                terms[name] = quat_rotate_inverse(root_quat_wxyz, gravity_world).flatten().astype(np.float32)
            elif "base_ang_vel" in name:
                terms[name] = base_ang_vel.astype(np.float32)
            elif "dof_pos" in name or "joint_pos" in name:
                terms[name] = (dof_pos - self.default_dof_angles).astype(np.float32)
            elif "dof_vel" in name or "joint_vel" in name:
                terms[name] = dof_vel.astype(np.float32)
            elif "actions" in name:
                terms[name] = self._last_action.copy()
            elif "force_cmd_local" in name:
                # UniFP F^cmd obs (2 hands × 3-axis = 6D, already ×0.04-scaled).
                # Zero by default → Eq.2 target collapses to the raw motion ref
                # (pure 2pt position tracking). When HOLOSOMA_FORCE_CMD_LOCAL is
                # set, the policy is told to exert that contact force and shifts
                # its hand target by F^cmd/K. Frame is pelvis (root) by default, or
                # each hand's (wrist_yaw) frame when HOLOSOMA_FORCE_CMD_HAND_FRAME=1
                # (matching a --F_on_hand_frame checkpoint).
                terms[name] = self._force_cmd_local_obs(dof_pos)
            elif "force_ext_local" in name:
                # F^ext obs. In the 2pt-force-ext preset the ACTOR sees F^ext (the
                # UniFP-asymmetry revert), and at deploy F^ext ≈ the LIVE compensated
                # F/T reading (gravity/inertia removed) — so we feed the pelvis-frame
                # Bota force here, ×0.04-scaled (=1/25 N), matching training's
                # ObsTermCfg.scale. Zero when no live source (byte-identical). In
                # headless eval there is no disturbance, so it also stays zero.
                terms[name] = (self._force_meas_pelvis.reshape(-1) * 0.04).astype(np.float32)
            elif "force_meas_local" in name:
                # Hybrid controller actor term (exp:g1-wbt-2pt-hybrid): the simulated
                # F/T reading in sim, the LIVE Bota reading at deploy. Pelvis frame,
                # ×0.04-scaled (=1/25 N). Zero when no live source (pure 2pt tracking
                # on the hybrid checkpoint) — byte-identical.
                terms[name] = (self._force_meas_pelvis.reshape(-1) * 0.04).astype(np.float32)
            elif "contact_phase" in name:
                # C1 UNIFIED actor term (exp:g1-wbt-3pt-hybrid-unified[-anneal]): the
                # SOFT per-hand contact label φ ∈[0,1], scale 1.0. Training computes it
                # from the SIMULATED F/T magnitude precisely so the real Bota reproduces
                # it — so at deploy we feed the LIVE label from _update_contact_phase
                # (hysteresis 2/1.5 N + 50 ms LPF on the same pelvis-frame ‖F_meas‖ this
                # tick's force_meas_local slot carries). Zero with no live F/T source
                # (free space, = training's reset buffer) and during the post-START
                # warmup + auto-tare burst. This is the DEPLOYABLE twin of the privileged
                # `contact_gate` (depth>0), which stays critic-only forever.
                terms[name] = np.asarray(getattr(self, "_contact_phase", np.zeros(2)), dtype=np.float32).reshape(-1)
            elif "integrator_state" in name:
                # Jᵀ-servo force-unit integrator I [N], pelvis frame, scale 1.0.
                # Zero when the servo state is off (Layer-A checkpoint) or no live
                # source; LIVE (wound) for the observe-only UNIFIED presets, whose actor
                # consumes it. The obs width is fixed per preset either way — the servo
                # channels are always present, zeroed only when inactive.
                terms[name] = self._servo_integ.reshape(-1).astype(np.float32)
            elif "x_tgt_shift" in name:
                # Jᵀ-servo task-space setpoint shift Δx [m], pelvis frame, scale 1.0.
                # Zero when the servo is off / no live source.
                terms[name] = self._servo_delta_x.reshape(-1).astype(np.float32)
            elif "compliance" in name:
                terms[name] = self._compliance_vec
            elif "height_scan" in name:
                if self._mj_model is not None and self._mj_data is not None:
                    robot_yaw = float(self._quat_yaw(root_quat_wxyz))
                    terms[name] = compute_height_scan_mj_ray(
                        self._mj_model, self._mj_data, root_pos, robot_yaw, n_rays=self._height_scan_dim
                    )
                else:
                    robot_z = root_pos[2] if root_pos is not None else 0.76
                    terms[name] = np.full(self._height_scan_dim, robot_z - 0.5, dtype=np.float32)
            else:
                logger.warning(f"Unknown obs term '{name}', filling with zeros")
                terms[name] = np.zeros(1, dtype=np.float32)

        return terms

    def _build_observation(self, robot_state_data) -> np.ndarray:
        term_values = self._compute_obs_terms(robot_state_data)
        # Cache for debug logging (avoids recomputing in rl_inference)
        self._last_obs_terms = term_values

        flattened_terms = []
        for term in self._obs_terms:
            value = term_values[term].reshape(-1)
            self._obs_history[term].append(value.copy())

            history = list(self._obs_history[term])
            if len(history) < self._history_length:
                missing = self._history_length - len(history)
                history = [np.zeros_like(value)] * missing + history

            stacked = np.concatenate(history[-self._history_length :])
            flattened_terms.append(stacked)

        obs = np.concatenate(flattened_terms).reshape(1, -1).astype(np.float32)
        return obs

    def _disable_policy_with_stiff_hold(self, robot_state_data, reason: str) -> np.ndarray:
        """Disable inference and return the relative action for one finite fixed hold."""
        offsets = np.asarray(self.joint_offsets, dtype=np.float32).reshape(-1)
        candidates = [("measured pose", robot_state_data[:, 7 : 7 + self.num_dofs])]
        if getattr(self, "cmd_q", None) is not None:
            candidates.append(("last command", np.asarray(self.cmd_q).reshape(1, -1) - offsets.reshape(1, -1)))
        if getattr(self, "_stiff_hold_q", None) is not None:
            candidates.append(("previous stiff hold", self._stiff_hold_q))

        hold_q = None
        hold_source = None
        for source, candidate in candidates:
            q = np.asarray(candidate, dtype=np.float32).reshape(1, -1)
            if q.shape[1] == self.num_dofs and offsets.size == self.num_dofs and np.all(np.isfinite(q + offsets)):
                hold_q = q.copy()
                hold_source = source
                break

        self.use_policy_action = False
        self.get_ready_state = False
        self._servo_dq[:] = 0.0
        if hold_q is None:
            # White, not red: loguru paints CRITICAL on a red BACKGROUND, so a red
            # foreground here is red-on-red. See `_trip_force_estop`.
            logger.critical(
                colored(
                    f"{reason} — no finite hold target is available; use hardware E-STOP.",
                    "white",
                    attrs=["bold"],
                )
            )
            raise RuntimeError(f"{reason}; no finite hold target is available")

        # Freeze one captured target. Merely disabling policy would otherwise command
        # each new readback and let a gravity-driven sag become the next target.
        self._stiff_hold_q = hold_q
        self._stiff_hold_active = True
        self._init_interpolating = False
        self._prepend_stiff_active = False
        self.motion_clip_progressing = False
        self._vive_armed = False
        self._action_ema_state = None
        for name in ("_force_cmd_target_raw", "_force_cmd_pelvis_raw", "_force_cmd_local_scaled"):
            value = getattr(self, name, None)
            if value is not None:
                value[:] = 0.0

        logger.critical(colored(f"{reason} — policy disabled; holding the {hold_source}.", "white", attrs=["bold"]))
        return hold_q - np.asarray(self.default_dof_angles, dtype=np.float32).reshape(1, -1)

    def rl_inference(self, robot_state_data):
        # HARD F/T STOP, checked FIRST: before the F^cmd polls, the obs build and the ONNX,
        # so a tripping tick can neither slew a new force command into the buffer nor emit
        # a policy action. `_trip_force_estop` runs the joystick-'B' handler (which flips
        # `use_policy_action` False), and the tick still owes the loop ONE action, so we
        # take the same stiff-hold exit the non-finite-action fault uses.
        _estop_mag = self._check_force_estop()
        if _estop_mag is not None:
            self._trip_force_estop(_estop_mag)
            return self._disable_policy_with_stiff_hold(
                robot_state_data, f"F/T hard stop at {_estop_mag:.1f}N (> {self._force_estop_n:.0f}N)"
            )

        if not self.motion_clip_progressing:
            self.timestep_util.reset(start_timestep=self.config.task.motion_start_timestep)
            self.curr_motion_timestep = self.timestep_util.timestep

        # Live keyboard F^cmd nudges — fold BEFORE the force obs / servo read
        # _force_cmd_pelvis_raw + _force_cmd_local_scaled this tick. No-op when off.
        self._poll_force_cmd_keys()
        # Live WIRE F^cmd from the HL planner (sensor→pelvis, slewed) — same discipline as
        # the keys: folded ONCE here on the DRIVING tick (not inside _compute_obs_terms,
        # which also runs on the non-driving prepend-stiff path). No-op unless
        # HOLOSOMA_FORCE_CMD_WIRE is on. Uses this tick's measured joints for the FK.
        # getattr default keeps a partially-constructed stub (object.__new__ in tests)
        # working — real instances always set it in __init__.
        if getattr(self, "_force_cmd_wire_enabled", False):
            self._poll_force_cmd_wire(robot_state_data[0, 7 : 7 + self.num_dofs])

        # Compute motion alignment on first step for global tracking modes.
        # Requires pelvis mocap (root_pos != 0) — headless sim2sim gets this from
        # MuJoCoInterface, sim2real from OptiTrack/MocapInterfaceWrapper.
        if self._align_rot_mat is None and self._obs_mode in ("global", "global_vel", "2pt", "3pt_global"):
            root_pos = robot_state_data[0, 0:3]
            has_mocap = not np.all(root_pos == 0)
            if has_mocap:
                self._compute_motion_alignment(robot_state_data)

        # Ready-pose hold: substitute the forearms-horizontal DOF into the frozen
        # reference BEFORE the obs is built, so the very first hold tick already
        # commands the ready pose (the end-of-tick refresh re-applies it every tick
        # thereafter). No-op when disabled / clip playing.
        self._apply_hold_dof_override()

        obs = self._build_observation(robot_state_data)
        if self._debug_log is not None:
            obs_raw = obs.flatten().copy()  # snapshot before normalization clipping
        obs = self._clip_normalized_obs(obs)

        # Run ONNX. Feed the obs under the resolved input name ("obs" for the deploy
        # export, "actor_obs" for the browser export) and include "time_step" only
        # when the graph declares it (the browser export drops it). Read the action
        # under the resolved output name ("actions" vs "action"). 2pt deploy is
        # byte-identical: obs input "obs", time_step present, output "actions".
        input_feed = {self._onnx_obs_input: obs}
        if self._onnx_uses_time_step:
            input_feed["time_step"] = np.array([[self.curr_motion_timestep]], dtype=np.float32)
        # The timestep the ONNX actually saw THIS tick, snapshotted BEFORE the end-of-tick
        # frame advance (_set_motion_timestep) so the RECORD tap logs the value aligned with
        # this tick's obs/action — not next tick's. Matches the _debug_log convention.
        record_timestep = float(self.curr_motion_timestep)
        outputs = self.onnx_policy_session.run(None, input_feed)
        out_map = {name: val for name, val in zip(self.onnx_output_names, outputs)}

        action_raw = np.asarray(out_map[self._onnx_action_output])
        action_fault = None
        if action_raw.shape not in ((self.num_dofs,), (1, self.num_dofs)):
            action_fault = (
                f"ONNX produced action shape {action_raw.shape}; expected ({self.num_dofs},) or (1, {self.num_dofs})"
            )
            action = None
        else:
            action = action_raw.reshape(-1).astype(np.float32)
            if not np.all(np.isfinite(action)):
                action_fault = "ONNX produced a non-finite action"
        if action_fault is not None:
            return self._disable_policy_with_stiff_hold(robot_state_data, action_fault)
        action = np.clip(action, -100, 100)

        # Action EMA low-pass (sim2real anti-chatter). alpha=1.0 → pass-through
        # (byte-identical). alpha<1 → a_t = alpha*a_raw + (1-alpha)*a_{t-1}. Seed
        # lazily from the FIRST raw action so the first command is unfiltered (no
        # ramp-from-zero transient). The smoothed action is what reaches the motors
        # AND what feeds back as the next-step `actions` obs term (set below via
        # _last_action), matching HolosomaMaskedMimicPolicy.
        if self._action_ema_alpha < 1.0:
            if self._action_ema_state is None:
                self._action_ema_state = action.copy()
            else:
                self._action_ema_state = (
                    self._action_ema_alpha * action + (1.0 - self._action_ema_alpha) * self._action_ema_state
                ).astype(np.float32)
            action = self._action_ema_state.copy()

        # Debug log BEFORE updating _last_action so that the cached
        # term values (which include the "actions" term = previous action) match
        # what the model actually saw in the observation.
        if self._debug_log is not None:
            dof_pos = robot_state_data[0, 7 : 7 + self.num_dofs]
            self._debug_log["obs_raw"].append(obs_raw)
            self._debug_log["obs_clipped"].append(obs.flatten())
            self._debug_log["action"].append(action.copy())
            self._debug_log["dof_pos"].append(dof_pos.copy())
            self._debug_log["timestep"].append(self.curr_motion_timestep)
            # Use cached terms from _build_observation (avoids recomputing)
            for k in self._obs_terms:
                if k in self._last_obs_terms:
                    self._debug_log.setdefault(f"term_{k}", []).append(self._last_obs_terms[k].flatten())

        self._last_action = action.copy()
        self.last_policy_action = action.reshape(1, -1)

        # Update motion command for next step's observation
        jp, jv, rp, rq, rv = self._get_ext_motion_at_frame(self._ext_frame_idx)
        self._motion_command_t = np.concatenate([jp.reshape(1, -1), jv.reshape(1, -1)], axis=1)
        self._apply_hold_dof_override()  # ready-pose DOF while the clip is frozen
        self._ref_quat_xyzw_t = rq.reshape(1, 4).astype(np.float32)
        self._ref_pos_xyz_t = rp.reshape(1, 3).astype(np.float32)
        self._ref_vel_xyz_t = rv.reshape(1, 3).astype(np.float32)
        self.motion_command_t = self._motion_command_t.copy()
        self.ref_quat_xyzw_t = self._ref_quat_xyzw_t.copy()

        # No joint reordering — holosoma training order matches SDK order.
        # Action scaling same as core WBT: scaled_action = action * per_joint_scale
        # BasePolicy then computes: q_target = scaled_action + default_dof_angles
        self.scaled_policy_action = (action * self._action_scale).reshape(1, -1)

        # Jᵀ force-servo residual (Layer B): add the per-DOF servo Δq [rad] to the
        # scaled action, so BasePolicy's q_target = scaled_action + Δq +
        # default_dof_angles shifts the ARM PD target by Δq rad. This is the deploy
        # analogue of training's `_actions_after_delay += servo_dq/action_scale`
        # (same extra PD torque Kp·Δq). Computed in _update_force_meas_and_servo this
        # same tick; zero when the servo is off / no live source (byte-identical).
        # The `actions` obs term feeds back the RAW policy action (_last_action set
        # above), NOT this servo-augmented target — matching training, where the
        # servo residual is invisible to the policy's action channel.
        #
        # UNIFIED (C1) presets never reach this branch: setup_policy forces
        # `_hybrid_servo = False` for any preset carrying a contact_phase term, and
        # _update_force_meas_and_servo additionally zeroes `_servo_dq` in observe-only
        # mode. Those checkpoints trained with the integrator as a pure reference
        # generator, so a Δq here would be a second actuator the policy never saw.
        if self._hybrid_servo and self._force_live is not None:
            self.scaled_policy_action = self.scaled_policy_action + self._servo_dq.reshape(1, -1)
        if self.scaled_policy_action.shape != (1, self.num_dofs) or not np.all(np.isfinite(self.scaled_policy_action)):
            return self._disable_policy_with_stiff_hold(
                robot_state_data,
                f"policy target became invalid after action scaling/servo: {self.scaled_policy_action.shape}",
            )

        if self.motion_clip_progressing:
            self._set_motion_timestep()
            self._ext_frame_idx = min(self._ext_frame_idx + 1, self._ext_motion_length - 1)

        # Opt-in teleop RECORD tap: ship this tick's F^cmd + pelvis-frame F_meas/servo +
        # 29-DOF proprioception + the running/vive-armed flags to the out-of-process
        # recorder. `_record_tap is None` unless RECORD=1, so this is a single early-return
        # (no work, byte-identical) when off. Placed at the very END of the tick so every
        # quantity is final: F^cmd was slewed at the top (_poll_force_cmd_keys), F_meas +
        # servo were refreshed inside _build_observation→_update_force_meas_and_servo, and
        # dof_pos/vel come straight from this tick's robot_state_data (same slices the obs
        # used). rl_inference only runs while use_policy_action is True, so `running` is
        # True here; its true→false STOP edge shows up to the recorder as the frame stream
        # simply stopping.
        if self._record_tap is not None:
            try:
                nd = self.num_dofs
                # The commanded VR target the policy saw this tick: the vr_2point_pos /
                # vr_3point_pos obs term (tracker-driven hand [+head] TARGET, pelvis frame,
                # metres), cached in _last_obs_terms by _build_observation. Pure O(copy)
                # lookup — reshaped to (N,3) so the recorder can pair it against the
                # executed wrist FK (from dof_pos) for the tracking-error plot. None when
                # neither VR term is present (non-teleop preset) → no tracking data logged.
                # ⚠️ SUBSTRING match, mirroring how `_build_observation` selects the same term
                # (`elif "vr_5point_pos" in name or ...`). An exact `.get()` here published None on
                # EVERY current preset: the live spellings are `vr_3point_pos_tcp` /
                # `vr_5point_pos_tcp` (the TCP-unified retrains — see `_vr_pos_tcp` in
                # setup_policy), so neither bare key was ever present and the tracking-error plot
                # had no data to draw. 5pt was missing outright.
                vr_cmd_pos = None
                vr_cmd_slewed = None
                for _name in self._obs_terms if self._obs_terms else ():
                    if "vr_2point_pos" in _name or "vr_3point_pos" in _name or "vr_5point_pos" in _name:
                        _v = self._last_obs_terms.get(_name) if self._last_obs_terms else None
                        if _v is not None:
                            vr_cmd_pos = np.asarray(_v, dtype=np.float64).reshape(-1, 3)
                            # Which of those rows the slew caps covered this tick (all, unless the
                            # 5pt feet are exempt) — a consumer measuring the cap off vr_cmd_pos
                            # must difference only these rows.
                            _n_slew = self._vr_slewed_points(vr_cmd_pos.shape[0])
                            vr_cmd_slewed = [i < _n_slew for i in range(vr_cmd_pos.shape[0])]
                            break
                self._record_tap.publish(
                    running=bool(getattr(self, "use_policy_action", True)),
                    vive_armed=bool(getattr(self, "_vive_armed", False)),
                    timestep=record_timestep,  # ONNX-fed value this tick (pre-advance)
                    servo_on=bool(self._hybrid_servo),
                    f_cmd=self._force_cmd_pelvis_raw,
                    f_meas=self._force_meas_pelvis,
                    f_meas_ungated=getattr(self, "_force_meas_ungated_last", self._force_meas_pelvis),
                    servo_integ=self._servo_integ,
                    servo_dx=self._servo_delta_x,
                    servo_dq=self._servo_dq,
                    dof_pos=robot_state_data[0, 7 : 7 + nd],
                    dof_vel=robot_state_data[0, 7 + nd + 6 : 7 + nd + 6 + nd],
                    root_quat_wxyz=robot_state_data[0, 3:7],
                    base_ang_vel=robot_state_data[0, 7 + nd + 3 : 7 + nd + 6],
                    vr_cmd_pos=vr_cmd_pos,
                    vr_cmd_slewed=vr_cmd_slewed,
                    # Did the wire's `ref_joints` drive command_lower_body this tick? None on every
                    # preset that is not 5pt (the field is read nowhere else); False = the frozen
                    # clip's legs; True = the walk skill's reference legs. FUJI prints it next to
                    # the walk so "the legs were told to stand" is a fact read off the robot.
                    ref_joints_live=getattr(self, "_ref_joints_applied", None),
                    # The soft contact label the actor consumed this tick — gated on the
                    # obs term actually being PRESENT (not on observe-only), so any
                    # future contact_phase preset records it too. None for every other
                    # preset, so old-take readers are unaffected.
                    contact_phase=(
                        getattr(self, "_contact_phase", None)
                        if getattr(self, "_has_contact_phase_term", False)
                        else None
                    ),
                )
            except Exception as exc:  # noqa: BLE001 - recording must never perturb control
                logger.warning(colored(f"[record-tap] publish failed (dropped): {exc!r}", "yellow"))

        return self.scaled_policy_action

    def get_init_target(self, robot_state_data):
        dof_pos = robot_state_data[:, 7 : 7 + self.num_dofs]
        if self.get_ready_state:
            target = self._init_q_target.reshape(1, -1)
            q_target = dof_pos + (target - dof_pos) * (self.init_count / 500)
            self.init_count += 1
            return q_target
        return dof_pos

    def get_current_obs_buffer_dict(self, robot_state_data):
        # Compatibility stub — rl_inference builds obs directly
        return {
            "motion_command": self._motion_command_t,
            "motion_ref_ori_b": np.zeros((1, 6)),
            "base_ang_vel": robot_state_data[:, 7 + self.num_dofs + 3 : 7 + self.num_dofs + 6],
            "dof_pos": robot_state_data[:, 7 : 7 + self.num_dofs] - self.default_dof_angles,
            "dof_vel": robot_state_data[:, 7 + self.num_dofs + 6 : 7 + self.num_dofs + 6 + self.num_dofs],
            "actions": self.last_policy_action,
        }

    def _handle_init_state(self):
        """Interpolate full body to motion init pose via stiff hold (keeps stiff KP/KD).

        The base class get_ready_state path uses motor KP/KD which are too low
        (training gains ~40) — robot collapses from stiff KP (350). Instead we
        interpolate within _get_manual_command which maintains stiff KP/KD.
        Uses 1000 steps (~5s at 200Hz) for a gentle transition.
        """
        # Recover from a soft-stop: clear it and start the ramp from the CURRENT
        # (sagged) measured pose so kp comes back gently from where the robot
        # actually is, not from a snap to the stiff startup pose.
        if self._soft_stop_active:
            self._soft_stop_active = False
            self._soft_stop_count = 0
            try:
                cur = self.interface.get_low_state()[0, 7 : 7 + self.num_dofs].reshape(1, -1).copy()
                self._stiff_hold_q = cur
            except Exception:  # noqa: BLE001 — fall back to the stiff startup pose
                pass
            self._soft_stop_q = None
        # Re-seed the action EMA on the next policy resume. INIT does NOT route
        # through _reset_motion_state, so without this the EMA would still hold the
        # action from BEFORE this init/recovery — on a soft-stop→INIT resume the
        # robot has sagged far from that stale action, and the first resumed
        # rl_inference would blend `alpha*new + (1-alpha)*STALE`, biasing the first
        # motor command toward the pre-sag pose at the worst moment. Nulling is
        # always safe (lazy seed re-fills from the first raw action, unfiltered).
        self._action_ema_state = None
        self._init_interpolating = True
        self._init_start_q = self._stiff_hold_q.copy()
        self._init_interp_count = 0
        self.logger.info(
            colored("Init: interpolating to motion init pose (stiff KP/KD, ~5s). Wait for 'Init pose reached'.", "cyan")
        )

    def _get_manual_command(self, robot_state_data):
        # Soft-stop takes priority over everything: ramp kp→floor while holding kd
        # and freezing q at the pose measured when soft-stop was triggered, so the
        # robot goes gently limp (damped descent) rather than freezing stiff.
        if self._soft_stop_active:
            if self._soft_stop_q is None:
                self._soft_stop_q = robot_state_data[0, 7 : 7 + self.num_dofs].reshape(1, -1).copy()
            alpha = min(self._soft_stop_count / max(self._soft_stop_steps, 1), 1.0)
            self._soft_stop_count += 1
            # kp: full stiff → floor (0 by default); kd: HELD (damping stays on).
            kp = self._stiff_hold_kp * (1.0 - alpha) + self._soft_stop_kp_floor * alpha
            if self._soft_stop_count == 1:
                self.logger.warning(
                    colored(
                        f"SOFT-STOP: ramping kp→{self._soft_stop_kp_floor} over "
                        f"{self._soft_stop_steps} ticks (kd held). The robot will SINK/FOLD "
                        "into the harness or floor (NOT a hover-in-place). NOT a hardware E-stop.",
                        "red",
                    )
                )
            elif alpha >= 1.0 and self._soft_stop_count == self._soft_stop_steps + 1:
                self.logger.info(
                    colored(
                        "SOFT-STOP: kp floor reached — only kd damping holds; robot sags to harness/floor.", "yellow"
                    )
                )
            return {"q": self._soft_stop_q, "kp": kp, "kd": self._stiff_hold_kd}

        if not self._stiff_hold_active:
            return None

        q_target = self._stiff_hold_q.copy()

        if self._init_interpolating:
            alpha = min(self._init_interp_count / 1000.0, 1.0)
            init_target = self._init_q_target.reshape(1, -1)
            q_target = self._init_start_q + (init_target - self._init_start_q) * alpha
            self._init_interp_count += 1
            if alpha >= 1.0:
                self._init_interpolating = False
                self._stiff_hold_q = q_target.copy()
                self.logger.info(colored("Init pose reached", "green"))
            return {"q": q_target, "kp": self._stiff_hold_kp, "kd": self._stiff_hold_kd}

        if self._prepend_stiff_active:
            # Drive prepend transition with stiff KP — direct position tracking
            # (no ONNX inference), same requirements as init interpolation.
            self._build_observation(robot_state_data)  # keep history populated
            raw_jp = self._ext_motion["joint_pos"][self._ext_frame_idx]
            ref_joints = raw_jp[7:] if raw_jp.shape[0] > 29 else raw_jp
            q_target = ref_joints.reshape(1, -1)
            # Update motion reference for next step's obs
            jp, jv, rp, rq, rv = self._get_ext_motion_at_frame(self._ext_frame_idx)
            self._motion_command_t = np.concatenate([jp.reshape(1, -1), jv.reshape(1, -1)], axis=1)
            self._ref_quat_xyzw_t = rq.reshape(1, 4).astype(np.float32)
            self._ref_pos_xyz_t = rp.reshape(1, 3).astype(np.float32)
            self._ref_vel_xyz_t = rv.reshape(1, 3).astype(np.float32)
            self.motion_command_t = self._motion_command_t.copy()
            self.ref_quat_xyzw_t = self._ref_quat_xyzw_t.copy()
            # Advance frame
            if self.motion_clip_progressing:
                self._set_motion_timestep()
                self._ext_frame_idx = min(self._ext_frame_idx + 1, self._ext_motion_length - 1)
            if self._ext_frame_idx % 25 == 0:
                self.logger.info(
                    f"[Prepend {self._ext_frame_idx}/{self._num_prepend_frames}] "
                    f"stiff KP=[{self._stiff_hold_kp.min():.0f},{self._stiff_hold_kp.max():.0f}]"
                )
            # Prepend done → re-capture heading and enable policy with ONNX KP
            if self._ext_frame_idx >= self._num_prepend_frames:
                self._prepend_stiff_active = False
                # Re-capture heading AFTER prepend — robot orientation changed during
                # joint tracking. Stale heading = aggressive rotation at ONNX start.
                self._capture_robot_yaw_offset_holosoma()
                motion_wxyz = xyzw_to_wxyz(self._ref_quat_xyzw_t)
                self.motion_yaw_offset = self._quat_yaw(motion_wxyz)
                BasePolicy._handle_start_policy(self)
                self._stiff_hold_active = False
                self.logger.info(colored("Prepend complete — policy running with ONNX KP", "green"))
            return {"q": q_target, "kp": self._stiff_hold_kp, "kd": self._stiff_hold_kd}

        return {"q": q_target, "kp": self._stiff_hold_kp, "kd": self._stiff_hold_kd}

    def _handle_start_policy(self):
        """Start policy execution with correct initialization ordering.

        CRITICAL: The ordering of operations here prevents a race condition with
        the policy loop thread (policy_action() in base.py).

        The race condition (before this fix):
          1. BasePolicy._handle_start_policy() sets use_policy_action=True
          2. The policy loop thread (running at ~200Hz) immediately sees
             use_policy_action=True and calls rl_inference()
          3. But motion command state hasn't been reset yet — rl_inference uses
             stale motion_command_t from a previous run (e.g. frame 1015 instead
             of frame 0), causing wrong observations and divergent behavior

        The fix: Set up ALL state that rl_inference depends on BEFORE calling
        BasePolicy._handle_start_policy (which enables use_policy_action).

        Additional subtlety: _stiff_hold_active must stay True during setup.
        If we set it False before enabling the policy, there's a window where
        both stiff hold and policy are disabled — the robot gets no control
        commands and goes limp (discovered during sim2real testing).
        """
        self._init_interpolating = False

        # Discard any F^cmd keypresses buffered while stopped/INITing so they do NOT
        # flush as one large single-tick F^cmd step on the first live inference (the
        # deploy has no soft-stop — an OOD F^cmd jump would be a real transient).
        self._drain_force_cmd_keys()

        # Fire a burst of F/T re-tares so the last one zeroes the sensor AFTER the
        # init/ready-pose transient settles — the policy then starts from ~0 force
        # instead of crashing on a nonzero startup reading. Runs on a daemon thread
        # (spread over ~2 s); no-op without a live F/T source. FREE SPACE at START.
        self._start_force_tare_burst()

        # Re-arm the hard F/T stop. It latches on trip so one impact does not re-run the
        # stop handler every tick; START is the operator saying "I have dealt with it".
        # Ordered AFTER the tare burst so the re-arm inherits that burst's trust window
        # (the burst extends the deadline `_check_force_estop` suppresses inside).
        self._force_estop_latched = False

        # Phase 1: Set up ALL state that rl_inference reads.
        # The policy loop thread cannot call rl_inference yet because
        # use_policy_action is still False at this point.
        self._debug_log_path = os.environ.get("HOLOSOMA_DEBUG_LOG", "")
        if self._debug_log_path:
            self._debug_log = defaultdict(list)
            logger.info(colored(f"Debug logging ON → {self._debug_log_path}", "magenta"))
        # Reset motion command, actions, and obs history to frame 0
        # (prevents stale state from prior run).
        # NOTE: _reset_motion_state zeroes robot_yaw_offset / motion_yaw_offset
        # and sets _heading_initialized=False so heading is lazily re-captured
        # on the first rl_inference call (same pattern as TML).
        self._handle_start_motion_clip()

        # Phase 2: If prepend is active, keep stiff KP during prepend.
        # Direct position tracking (no ONNX) needs stiff KP to prevent collapse.
        if self._num_prepend_frames > 0:
            raise NotImplementedError("prepend mode not yet supported")
            self._prepend_stiff_active = True
            self.logger.info(
                colored(
                    f"Prepend active ({self._num_prepend_frames} frames, "
                    f"~{self._num_prepend_frames / 50.0:.1f}s) — stiff KP held during transition.",
                    "cyan",
                )
            )
        else:
            # No prepend — enable the policy in a NEUTRAL STANDING state: clip
            # FROZEN at frame-0 and VIVE DISARMED. The robot runs the policy and
            # holds the idle/default standing pose, doing nothing else, until the
            # operator explicitly picks a mode:
            #   select+A  → _handle_start_motion_clip : play the NPZ clip
            #   start     → _toggle_vive_teleop        : arm live VIVE tracking
            # This makes START a safe "stand up under the policy and wait" step —
            # the operator confirms the robot is stable on the policy before
            # handing control to a clip or the trackers. (Phase 1's
            # _handle_start_motion_clip already reset to frame-0; we force
            # motion_clip_progressing=False so it does NOT auto-advance.)
            self.motion_clip_progressing = False  # FREEZE — stand, don't play
            self._vive_armed = False  # trackers ignored until 'start' arms them
            BasePolicy._handle_start_policy(self)
            self._stiff_hold_active = False
            self.logger.info(
                colored(
                    "START — policy running, holding the standing pose. "
                    "Press select+A to play the NPZ clip, or 'start' to arm VIVE tracking.",
                    "green",
                )
            )

    def _reset_motion_state(self):
        """Reset motion command, reference frames, action history, and obs history.

        Common reset used by stop_policy, start_motion_clip, restore_state,
        and policy_switched to ensure clean state for the next policy run.
        """
        self._motion_command_t = self._motion_command_0.copy()
        self._ref_quat_xyzw_t = self._ref_quat_xyzw_0.copy()
        self._ref_pos_xyz_t = self._ref_pos_xyz_0.copy()
        self._ref_vel_xyz_t = self._ref_vel_xyz_0.copy()
        self._ext_frame_idx = 0
        self._last_action = np.zeros(29, dtype=np.float32)
        self.last_policy_action = np.zeros((1, 29), dtype=np.float32)
        # Re-seed the EMA on the next START's first action (no stale carry-over /
        # no ramp-from-zero) — mirrors the _last_action reset above.
        self._action_ema_state = None
        self._heading_initialized = False
        self.robot_yaw_offset = 0.0
        self.motion_yaw_offset = 0.0
        self._align_rot_mat = None
        self._align_pos_offset = None
        # Reset the force-servo state so a STOP→re-START never carries a wound-up
        # integrator across arms. The servo_step gate self-clears I on the first
        # free-space tick anyway, but an explicit reset also covers the (rare) case
        # of re-arming with the gripper already loaded (>contact gate) on tick 1.
        self._servo_integ[:] = 0.0
        self._servo_delta_x[:] = 0.0
        self._servo_dq[:] = 0.0
        self._force_meas_pelvis[:] = 0.0
        # Same for the soft contact label: a STOP→re-START must begin in "free space"
        # with the post-START warmup re-armed, never carrying a latched φ≈1 (or a
        # half-decayed LPF tail) into the first tick of the new arm.
        self._reset_contact_phase_state()
        self._integ_windup_warned = False
        # OBSERVE-ONLY ONLY: also zero the applied F^cmd. On the servo presets a stale
        # F^cmd across STOP→re-START merely meant "the servo pushes again once you
        # re-arm", and that behaviour is deliberately preserved. On a unified preset the
        # POLICY is the force controller: a stale 15 N command makes x_tgt_shift = F/K =
        # 7.5 cm on the very FIRST tick of the new arm, un-slewed, and the policy chases
        # it — i.e. STOP would not actually stop the pushing. The operator's mental model
        # ("STOP resets") must be true where it moves the arms.
        if getattr(self, "_servo_observe_only", False) and getattr(self, "_force_cmd_pelvis_raw", None) is not None:
            # The slew TARGET always goes to zero — that is the actual "stop commanding
            # force" instruction.
            if getattr(self, "_force_cmd_target_raw", None) is not None:
                self._force_cmd_target_raw[:] = 0.0
            # The APPLIED value is hard-zeroed only when the policy is NOT driving motors
            # (STOP / restore). This funnel also runs MID-RUN via `select+A` (play clip),
            # where hard-zeroing would step force_cmd_local and Δx in a single tick —
            # bypassing the very 2 N/tick slew that exists to keep the feed-forward from
            # stepping. There the slew ramps it down over the next few ticks instead.
            if not getattr(self, "use_policy_action", False):
                self._force_cmd_pelvis_raw[:] = 0.0
                if getattr(self, "_force_cmd_local_scaled", None) is not None:
                    self._force_cmd_local_scaled[:] = 0.0
        for term in self._obs_history:
            self._obs_history[term].clear()

    def _handle_stop_policy(self):
        """Stop policy and return to stiff hold mode.

        Saves any accumulated debug log to NPZ before resetting state.
        Re-enables stiff hold so the robot maintains position after stop.
        """
        self._save_debug_log()
        self.use_policy_action = False
        self.get_ready_state = False
        self._stiff_hold_active = True
        self._prepend_stiff_active = False
        self._init_interpolating = False
        self.motion_clip_progressing = False
        self._vive_armed = False  # STOP disarms live VIVE teleop
        # Reset the slew anchor + clutch anchor so a later re-START ramps cleanly
        # from the NPZ-FK target rather than from a stale held live pose.
        self._clear_vive_teleop_anchors()
        # Clear soft-stop. If STOP is pressed DURING a soft-stop (robot has
        # sagged), reseed _stiff_hold_q from the CURRENT measured pose BEFORE
        # re-enabling stiff hold — otherwise stiff hold snaps to the stale
        # pre-sag target at full kp = a one-tick jerk toward a pose the robot no
        # longer occupies. (INIT is still the gentler recovery; this just makes
        # the reflexive STOP safe.)
        if self._soft_stop_active:
            try:
                cur = self.interface.get_low_state()[0, 7 : 7 + self.num_dofs].reshape(1, -1).copy()
                self._stiff_hold_q = cur
                self.logger.info(
                    colored("STOP during soft-stop: holding the sagged pose (press INIT to recover).", "yellow")
                )
            except Exception:  # noqa: BLE001 — keep the existing target if readback fails
                pass
        self._soft_stop_active = False
        self._soft_stop_count = 0
        self._soft_stop_q = None
        self.timestep_util.reset(start_timestep=0)
        self.curr_motion_timestep = self.timestep_util.timestep
        self._reset_motion_state()
        if hasattr(self.interface, "no_action"):
            self.interface.no_action = 0
        self.logger.info(colored("HolosomaWBT stopped, entering stiff hold", "yellow"))

    def _handle_start_motion_clip(self):
        """Reset motion clip to frame 0 with clean observation state.

        Resets last_action and obs_history so the first observation after restart
        matches sim behavior (actions term = zeros, history padded with zeros).
        This is a safety net — _handle_start_policy also resets these, but having
        the reset here too ensures correctness if motion clip is restarted
        independently (e.g. via 's' key) during a policy run.
        """
        super()._handle_start_motion_clip()
        self._reset_motion_state()
        # Playing the NPZ clip and live-VIVE teleop are mutually exclusive: clear
        # any VIVE arming so the clip drives vr_2point, not the trackers. (Harmless
        # during START's Phase-1 call — VIVE is not armed yet there.)
        self._vive_armed = False
        self._clear_vive_teleop_anchors()

    def _capture_policy_state(self):
        state = BasePolicy._capture_policy_state(self)
        state.update(
            {
                "motion_command_0": self._motion_command_0.copy(),
                "ref_quat_xyzw_0": self._ref_quat_xyzw_0.copy(),
                "per_joint_policy_action_scale": self.per_joint_policy_action_scale.copy()
                if self.per_joint_policy_action_scale is not None
                else None,
            }
        )
        return state

    def _restore_policy_state(self, state):
        BasePolicy._restore_policy_state(self, state)
        self._motion_command_0 = state["motion_command_0"].copy()
        self._ref_quat_xyzw_0 = state["ref_quat_xyzw_0"].copy()
        saved = state["per_joint_policy_action_scale"]
        self.per_joint_policy_action_scale = saved.copy() if saved is not None else None
        self.motion_clip_progressing = False
        self.timestep_util.reset(start_timestep=0)
        self.curr_motion_timestep = self.timestep_util.timestep
        self._reset_motion_state()

    def _capture_robot_yaw_offset_holosoma(self):
        """Capture robot yaw using the training body_name_ref (not deployment config)."""
        robot_state_data = self.interface.get_low_state()
        if robot_state_data is None:
            self.robot_yaw_offset = 0.0
            self.logger.warning("Unable to capture robot yaw offset - missing robot state.")
            return
        if self._ref_body_is_pelvis:
            robot_ref_ori = robot_state_data[0, 3:7].reshape(1, 4)  # pelvis wxyz from IMU
        else:
            robot_ref_ori = self._get_ref_body_orientation_in_world(robot_state_data)
        yaw = self._quat_yaw(robot_ref_ori)
        self.robot_yaw_offset = yaw
        self.logger.info(
            colored(
                f"Robot yaw offset captured at {np.degrees(yaw):.1f} deg (pelvis={self._ref_body_is_pelvis})", "blue"
            )
        )

    def _on_policy_switched(self, model_path):
        self.motion_clip_progressing = False
        self.timestep_util.reset(start_timestep=0)
        self.curr_motion_timestep = self.timestep_util.timestep
        self._stiff_hold_active = True
        # Disarm live VIVE on a policy switch so a switched-in policy never starts
        # already-armed; the operator must press START again. Clear the slew anchor
        # + clutch anchor too so the re-arm ramps cleanly. (Only _handle_stop_policy
        # disarmed before.)
        self._vive_armed = False
        self._clear_vive_teleop_anchors()
        self._reset_motion_state()
