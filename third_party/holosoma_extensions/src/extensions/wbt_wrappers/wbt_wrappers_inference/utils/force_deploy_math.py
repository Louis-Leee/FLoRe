"""Pure-numpy force-deploy math for the sim2real hybrid / force-ext presets.

Single source of truth for the two load-bearing computations the deploy policy
does with a LIVE F/T sensor reading (the real Bota), kept here as pure functions
so they are headlessly unit-testable (no ONNX / robot / Pinocchio needed):

  1. ``sensor_force_to_pelvis`` — rotate a sensor-frame contact force into the
     robot **pelvis (root)** frame, the frame every force obs term is expressed
     in (``force_meas_local`` / ``force_ext_local`` / ``force_cmd_local``). This
     is the EXACT INVERSE of the F^cmd pelvis→sensor chain in
     ``holosoma_wbt._force_cmd_local_obs`` (same ``R_sensor`` + FK-derived
     ``R_rel``), so a commanded force and its measured reaction live in one frame.

  2. ``servo_step`` — the Layer-B contact-gated Jᵀ force servo, a faithful numpy
     port of ``wbt_training.config_values.hybrid.servo_residual_step`` (and the
     browser ``mjswanRuntime.hybridServoStep``). Given the TCP Jacobian, F^cmd,
     the measured force, and a contact gate, it returns the per-joint arm PD-target
     shift ``Δq`` [rad], the updated force-unit integrator ``I`` [N], and the
     task-space setpoint shift ``Δx`` [m] (the ``x_tgt_shift`` obs term).

FRAME / SIGN conventions (verified vs the URDF + training):

* ``R_SENSOR`` is the constant wrist←sensor rotation from the URDF
  ``{side}_ft_sensor_joint`` ``rpy=(0, -pi/2, pi)``: sensor +z→wrist +x,
  +y→wrist -y, +x→wrist +z (a symmetric involution). ``v_wrist = R_SENSOR @
  v_sensor``. Matches ``hybrid_pf_playground._R_sensor`` and
  ``force_tracking._rot_matrix_from_rpy((0,-pi/2,pi))``.
* ``F_meas`` in training is the external force ON the robot; at the servo fixed
  point it OPPOSES F^cmd (``e = F_cmd + F_meas → 0``). The physical Bota may
  report the opposite sign, so callers pass a ``sign`` (±1) resolved on the
  gantry (``HOLOSOMA_FORCE_MEAS_SIGN``).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Constant wrist←sensor rotation from URDF ft_sensor_joint rpy=(0,-pi/2,pi).
# Symmetric involution (R_SENSOR == R_SENSOR.T): sensor +z→wrist +x, +y→-y, +x→+z.
# f_wrist = R_SENSOR @ f_sensor.
R_SENSOR: np.ndarray = np.array([[0.0, 0.0, 1.0], [0.0, -1.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float64)

# FT-sensor and TCP offsets along wrist_yaw +x (m), from g1_29dof_yam.urdf +
# hybrid.py (FT_OFFSET_X / TCP_OFFSET_X). Their difference is the rigid lever the
# servo uses to shift the wrist Jacobian to the TCP.
FT_OFFSET_X: float = 0.082
TCP_OFFSET_X: float = 0.242


def rot_matrix_from_rpy(rpy: tuple[float, float, float]) -> np.ndarray:
    """3×3 rotation for a URDF ``rpy`` (extrinsic X-Y-Z: R = Rz·Ry·Rx).

    Numpy twin of ``force_tracking._rot_matrix_from_rpy`` /
    ``holosoma_wbt._rot_matrix_from_rpy`` — same convention, kept local so this
    module has no torch / holosoma import.
    """
    r, p, y = rpy
    cr, sr = np.cos(r), np.sin(r)
    cp, sp = np.cos(p), np.sin(p)
    cy, sy = np.cos(y), np.sin(y)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=np.float64)
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=np.float64)
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=np.float64)
    return rz @ ry @ rx


def sensor_force_to_pelvis(f_sensor: np.ndarray, r_rel: np.ndarray, sign: float = 1.0) -> np.ndarray:
    """Rotate a sensor-frame 3-D force into the pelvis (root) frame.

    ``f_pelvis = sign · R_rel @ (R_SENSOR @ f_sensor)`` where:

    * ``R_SENSOR`` (constant) takes sensor→wrist_yaw coords, and
    * ``R_rel`` = the wrist_yaw rotation expressed IN the pelvis frame
      (``oMf[wrist].rotation`` from identity-root Pinocchio FK on the measured
      joints), taking wrist→pelvis coords.

    Exactly the inverse of ``_force_cmd_local_obs``'s pelvis→sensor chain
    ``f_sensor = R_SENSOR^T @ (R_rel^T @ f_pelvis)`` (R_SENSOR is an involution).
    ``sign`` flips the physical Bota convention to training's "external force ON
    the robot" (F_meas opposes F_cmd at the servo fixed point).
    """
    f_sensor = np.asarray(f_sensor, dtype=np.float64).reshape(3)
    f_wrist = R_SENSOR @ f_sensor
    return (float(sign) * (np.asarray(r_rel, dtype=np.float64).reshape(3, 3) @ f_wrist)).astype(np.float64)


def tcp_jacobian_from_wrist(j_wrist_lin: np.ndarray, j_wrist_ang: np.ndarray, r_w: np.ndarray) -> np.ndarray:
    """Shift a wrist-origin world/pelvis-aligned linear Jacobian to the TCP.

    ``J_v_tcp = J_v_wrist - [r_w]× · J_ω`` (rigid-body Jacobian offset), with
    ``r_w`` the wrist→TCP lever expressed in the same (pelvis) frame. Mirrors
    ``hybrid._tcp_jacobian_world`` (``J_v += -[r]× J_ω``). Returns (3, n_arm).
    """
    j_v = np.asarray(j_wrist_lin, dtype=np.float64)  # (3, n_arm)
    j_w = np.asarray(j_wrist_ang, dtype=np.float64)  # (3, n_arm)
    r = np.asarray(r_w, dtype=np.float64).reshape(3)
    skew = np.array([[0.0, -r[2], r[1]], [r[2], 0.0, -r[0]], [-r[1], r[0], 0.0]], dtype=np.float64)
    return j_v - skew @ j_w


@dataclass(frozen=True)
class ServoConfig:
    """Layer-B Jᵀ force-servo knobs. Defaults match ``hybrid.HybridContactConfig``
    (servo_gain_k=0.05, integrator_gain_ki=0.5, integrator_max=40, servo_dq_max=
    0.15, stiffness_K=200) and the browser ``HybridConfig``. All overridable via
    ``HOLOSOMA_HYBRID_SERVO_*`` env vars in the policy."""

    servo_gain_k: float = 0.05  # rad/(N·m) — Δq = k·Jᵀ·task_wrench
    integrator_gain_ki: float = 0.5  # 1/s — force-unit integral gain
    integrator_max: float = 40.0  # N — anti-windup clip on I (force units)
    servo_dq_max: float = 0.15  # rad — per-joint |Δq| clamp
    stiffness_K: float = 200.0  # N/m — task_wrench → Δx setpoint shift
    tcp_offset_x: float = TCP_OFFSET_X  # m — wrist→TCP lever along +x


def servo_step(
    j_tcp_lin: np.ndarray,
    f_cmd_w: np.ndarray,
    f_meas_w: np.ndarray,
    contact_gate: float,
    integ_state: np.ndarray,
    cfg: ServoConfig,
    dt: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One-hand contact-gated Jᵀ force-servo step (numpy port of
    ``hybrid.servo_residual_step``, force channels only, S=(1,1,1)).

    All 3-vectors are in the pelvis (root) frame (world-aligned, identity root):

        e          = F_cmd + F_meas                              [N]  (→0 in contact)
        I          = gate · clip(I + K_I·e·dt, ±I_max)           [N]  (frozen off-contact)
        task_wrench = F_cmd + I                                  [N]
        Δq_arm     = clip(k · Jᵀ_lin · task_wrench, ±dq_max)     [rad]
        Δx         = task_wrench / K                             [m]  (x_tgt_shift obs)

    Args:
        j_tcp_lin: (3, n_arm) linear TCP Jacobian (pelvis-aligned).
        f_cmd_w, f_meas_w: (3,) F^cmd and (signed) measured force, pelvis frame.
        contact_gate: 1.0 in contact else 0.0 (freezes the integrator).
        integ_state: (3,) current force-unit integrator I [N].
        dt: control-tick seconds (1/rl_rate ≈ 0.02; net integral increment per
            tick matches training's 4×physics_dt substeps).

    Returns (dq_arm (n_arm,), new_integ (3,), delta_x (3,)).
    """
    f_cmd = np.asarray(f_cmd_w, dtype=np.float64).reshape(3)
    f_meas = np.asarray(f_meas_w, dtype=np.float64).reshape(3)
    integ = np.asarray(integ_state, dtype=np.float64).reshape(3)
    e = f_cmd + f_meas  # force error [N]; steady-state → 0
    # Gated, anti-windup FORCE-unit integral (frozen in free space when gate=0).
    new_integ = float(contact_gate) * np.clip(
        integ + cfg.integrator_gain_ki * e * float(dt), -cfg.integrator_max, cfg.integrator_max
    )
    task_wrench = f_cmd + new_integ  # feedforward + integral, +F_cmd sign [N]
    dq_arm = cfg.servo_gain_k * (np.asarray(j_tcp_lin, dtype=np.float64).T @ task_wrench)  # Jᵀ·wrench [rad]
    dq_arm = np.clip(dq_arm, -cfg.servo_dq_max, cfg.servo_dq_max)
    delta_x = task_wrench / cfg.stiffness_K  # task-space setpoint shift [m]
    return dq_arm.astype(np.float64), new_integ.astype(np.float64), delta_x.astype(np.float64)
