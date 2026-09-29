"""Pure-numpy math for the offline force-target previewer (design spec
``docs/superpowers/specs/2026-07-09-force-target-preview-design.md``).

This module has NO viser / display dependency so it is unit-testable headless.
It computes, from a retargeted-motion ``.npz`` + a scripted force profile, the
UniFP Eq.2 task-space target shift

    x_target(t) = x_contact(t) + (F_cmd(t) + F_ext(t)) / K          K = 200 N/m

and (via pinocchio, lazily imported) the arm-only inverse kinematics that turns
that task-space shift into a joint configuration for the transparent-orange
"target" ghost mesh.

Frame conventions (verified against the data + URDF, see spec §2/§3):
  * All quaternions are **wxyz** (w-first) — IsaacSim / holosoma native.
  * ``F_cmd`` is authored in the **pelvis (root) frame** and rotated to world by
    the live pelvis quaternion, matching ``force_tracking._advance_force_pulses``.
  * ``x_contact`` is the hand contact point = ``p_wrist + R(q_wrist) @ (off,0,0)``,
    with ``off`` = 0.242 m (TCP, 2026-07-10 was 0.215) or 0.082 m (FT sensor). Both come straight from
    the wrist_yaw body pose stored in the npz (no FK needed for the anchor).
  * The sensor→TCP wrench transform reuses the *validated* 6-D adjoint from the
    playground (``adjoint_sensor_to_tcp``) — single source of truth for the
    geometry constants.

Honesty banner (carried from the playground): ``F_ext = F_meas`` is NOT physically
simulated here — there is no contact object and no policy. Both force profiles are
scripted/illustrative; scenario B's ``F_ext → −F_cmd`` equilibrium is hand-drawn,
NOT loop closure.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Single source of truth for the wrench geometry: the validated playground
# constants + adjoint (spec §4 "reuse, do not duplicate"). This import pulls
# ``mujoco`` transitively, which is present in the ``hssim`` env where both this
# tool and its tests run.
from wbt_wrappers_inference.hybrid_pf_playground import (
    FT_SITE_POS,
    FT_TCP_LEVER_X,
    TCP_SITE_POS,
    _R_sensor,
    adjoint_sensor_to_tcp,
)

# UniFP Eq.2 stiffness (N/m). Mirrors ``HandForceConfig.stiffness_K`` /
# ``eval_utils.FORCE_STIFFNESS_K``. A 20 N command → 20/200 = 0.10 m target shift.
STIFFNESS_K = 200.0

# Default commanded-force peak (N). Deliberately BELOW the 25 N HandForceConfig
# max: an Opus review found that 20 N (a 10 cm shift) pushes ~14% of frames on the
# `avoid_bump_push` clip past the ~0.41 m arm reach, so the IK ghost drops to
# marker-fallback exactly on the extended-arm frames the demo means to show. 12 N
# (a 6 cm shift) keeps the orange ghost visible on the vast majority of frames
# while still being clearly readable. The CLI slider can push it back up to 25 N.
DEFAULT_PEAK_N = 12.0

# Contact-anchor offsets along wrist_yaw +x (m). Both frames share wrist_yaw's
# orientation (every joint in the wrist_yaw→TCP chain has rpy=0), so the anchor
# is a pure translation along wrist +x. See spec §2.1.
TCP_OFFSET_X = TCP_SITE_POS[0]  # 0.242 (2026-07-10, was 0.215)
FT_OFFSET_X = FT_SITE_POS[0]  # 0.082

# npz layout (spec §3, verified).
_NPZ_BASE_POS = slice(0, 3)
_NPZ_BASE_QUAT = slice(3, 7)  # wxyz
_NPZ_DOF = slice(7, 36)  # 29 actuated

# The active-hand wrist_yaw body name → npz/body index is resolved by name at load.
WRIST_YAW_LINK = {"left": "left_wrist_yaw_link", "right": "right_wrist_yaw_link"}

# The 7 arm joints per side that the IK is allowed to vary (spec §4.1). Order is
# proximal→distal; the values are read from / written to the standard G1 DOF order.
ARM_JOINTS = (
    "shoulder_pitch",
    "shoulder_roll",
    "shoulder_yaw",
    "elbow",
    "wrist_roll",
    "wrist_pitch",
    "wrist_yaw",
)


# ---------------------------------------------------------------------------
# Quaternion helpers (wxyz), pure numpy.
# ---------------------------------------------------------------------------
def quat_to_mat_wxyz(q: np.ndarray) -> np.ndarray:
    """Rotation matrix from a wxyz quaternion. Normalizes defensively."""
    q = np.asarray(q, dtype=float)
    w, x, y, z = q / (np.linalg.norm(q) + 1e-12)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def quat_apply_wxyz(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate vector ``v`` by wxyz quaternion ``q``. Matches ``_quat_apply_wxyz``."""
    return quat_to_mat_wxyz(q) @ np.asarray(v, dtype=float)


# ---------------------------------------------------------------------------
# Motion container.
# ---------------------------------------------------------------------------
@dataclass
class Motion:
    """A loaded retargeted-motion clip (spec §3)."""

    fps: int
    joint_pos: np.ndarray  # (T, 36) base pos(3) + base quat wxyz(4) + 29 DOF
    body_pos_w: np.ndarray  # (T, 32, 3)
    body_quat_w: np.ndarray  # (T, 32, 4) wxyz
    body_names: list[str]
    joint_names: list[str]

    @property
    def n_frames(self) -> int:
        return int(self.joint_pos.shape[0])

    @property
    def dt(self) -> float:
        return 1.0 / float(self.fps)

    def _check_t(self, t: int) -> int:
        """Validate a frame index (reject out-of-range / negative-wrap surprises)."""
        if not (0 <= int(t) < self.n_frames):
            raise IndexError(f"frame {t} out of range [0, {self.n_frames})")
        return int(t)

    def base_pos(self, t: int) -> np.ndarray:
        t = self._check_t(t)
        return self.joint_pos[t, _NPZ_BASE_POS].astype(float)

    def base_quat(self, t: int) -> np.ndarray:
        t = self._check_t(t)
        return self.joint_pos[t, _NPZ_BASE_QUAT].astype(float)

    def dof(self, t: int) -> np.ndarray:
        """The 29 actuated joint angles at frame ``t`` (npz joint order)."""
        t = self._check_t(t)
        return self.joint_pos[t, _NPZ_DOF].astype(float)

    def pelvis_quat(self, t: int) -> np.ndarray:
        """Pelvis (root) orientation wxyz — used to rotate F_cmd pelvis→world."""
        t = self._check_t(t)
        return self.body_quat_w[t, self.body_names.index("pelvis"), :].astype(float)

    def wrist_pose(self, t: int, side: str) -> tuple[np.ndarray, np.ndarray]:
        """(pos_w, quat_w wxyz) of the active side's wrist_yaw_link at frame ``t``."""
        t = self._check_t(t)
        if side not in WRIST_YAW_LINK:
            raise ValueError(f"side must be 'left' or 'right', got {side!r}")
        bi = self.body_names.index(WRIST_YAW_LINK[side])
        return self.body_pos_w[t, bi, :].astype(float), self.body_quat_w[t, bi, :].astype(float)


def load_motion(npz_path: str) -> Motion:
    """Load and validate a retargeted-motion ``.npz`` (spec §3 contract)."""
    z = np.load(npz_path, allow_pickle=True)
    required = {"fps", "joint_pos", "body_pos_w", "body_quat_w", "body_names", "joint_names"}
    missing = required - set(z.keys())
    if missing:
        raise ValueError(f"{npz_path}: npz missing keys {sorted(missing)}")
    jp = np.asarray(z["joint_pos"], dtype=float)
    if jp.ndim != 2 or jp.shape[1] != 36:
        raise ValueError(f"{npz_path}: joint_pos must be (T, 36), got {jp.shape}")
    body_names = [str(n) for n in z["body_names"]]
    for needed in ("pelvis", "left_wrist_yaw_link", "right_wrist_yaw_link"):
        if needed not in body_names:
            raise ValueError(f"{npz_path}: body_names missing {needed!r}")
    return Motion(
        fps=int(z["fps"]),
        joint_pos=jp,
        body_pos_w=np.asarray(z["body_pos_w"], dtype=float),
        body_quat_w=np.asarray(z["body_quat_w"], dtype=float),
        body_names=body_names,
        joint_names=[str(n) for n in z["joint_names"]],
    )


# ---------------------------------------------------------------------------
# Contact anchor: x_contact = p_wrist + R(q_wrist) @ (offset_x, 0, 0).
# ---------------------------------------------------------------------------
def contact_point_world(p_wrist: np.ndarray, q_wrist_wxyz: np.ndarray, offset_x: float) -> np.ndarray:
    """World contact point ``offset_x`` metres forward (wrist +x) of wrist_yaw."""
    return np.asarray(p_wrist, dtype=float) + quat_apply_wxyz(q_wrist_wxyz, np.array([offset_x, 0.0, 0.0]))


# ---------------------------------------------------------------------------
# Trapezoidal force pulse (numpy; phase-compatible with force_tracking.py).
# One scalar shape in [0, 1]: 0 → ramp-up → 1 (hold) → ramp-down → 0 (idle).
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class PulseSpec:
    """Trapezoidal-pulse timing in env steps (50 Hz), matching force_tracking."""

    ramp: int = 50
    hold: int = 200
    idle: int = 50

    @property
    def ramp_end(self) -> int:
        return self.ramp

    @property
    def hold_end(self) -> int:
        return self.ramp + self.hold

    @property
    def down_end(self) -> int:
        return self.ramp + self.hold + self.ramp  # symmetric ramps

    @property
    def period(self) -> int:
        return self.ramp + self.hold + self.ramp + self.idle


def pulse_scale(t: int, spec: PulseSpec) -> float:
    """Trapezoid amplitude in [0, 1] at step ``t`` (wraps every ``spec.period``).

    Phases mirror ``force_tracking._step_pulse``: linear ramp-up over ``[0, ramp]``,
    flat hold over ``(ramp, ramp+hold]``, linear ramp-down over the next ``ramp``
    steps, then idle at 0. ``F = 0`` throughout the idle window (F=0 ⇒ pure ref).
    """
    tt = int(t) % spec.period
    if tt <= spec.ramp_end:
        return tt / max(spec.ramp_end, 1)
    if tt <= spec.hold_end:
        return 1.0
    if tt <= spec.down_end:
        return (spec.down_end - tt) / max(spec.down_end - spec.hold_end, 1)
    return 0.0


def pulse_phase(t: int, spec: PulseSpec) -> str:
    """Human-readable phase label for the HUD."""
    tt = int(t) % spec.period
    if tt <= spec.ramp_end:
        return "ramp-up"
    if tt <= spec.hold_end:
        return "hold"
    if tt <= spec.down_end:
        return "ramp-down"
    return "idle"


# ---------------------------------------------------------------------------
# Force profiles.
# ---------------------------------------------------------------------------
def f_cmd_pelvis(t: int, direction_pelvis: np.ndarray, peak_N: float, spec: PulseSpec) -> np.ndarray:
    """Commanded 3-D force in the PELVIS frame at frame ``t`` (trapezoidal)."""
    d = np.asarray(direction_pelvis, dtype=float)
    n = np.linalg.norm(d)
    unit = d / n if n > 1e-9 else np.zeros(3)
    return unit * peak_N * pulse_scale(t, spec)


def f_ext_pelvis(t: int, f_cmd_p: np.ndarray, spec: PulseSpec, scenario: str) -> np.ndarray:
    """External/measured force in the PELVIS frame (spec §2.2 — scripted/illustrative).

    * ``"A"``: F_ext = 0 (pure commanded shift).
    * ``"B"``: contact-gated equilibrium — F_ext ramps to ``−F_cmd`` across the hold
      phase, so the target pushes out then relaxes back toward the reference. This is
      a HAND-DRAWN equilibrium, NOT loop closure (there is no simulated contact here).
    """
    if scenario == "A":
        return np.zeros(3)
    if scenario == "B":
        # Gate 0→1 across the hold window (contact "engaging"), then hold at 1.
        tt = int(t) % spec.period
        if tt <= spec.ramp_end:
            gate = 0.0
        elif tt <= spec.hold_end:
            gate = (tt - spec.ramp_end) / max(spec.hold_end - spec.ramp_end, 1)
        elif tt <= spec.down_end:
            gate = 1.0
        else:
            gate = 0.0
        return -gate * np.asarray(f_cmd_p, dtype=float)
    raise ValueError(f"unknown scenario {scenario!r} (expected 'A' or 'B')")


# ---------------------------------------------------------------------------
# Eq.2 target.
# ---------------------------------------------------------------------------
def eq2_target_world(
    x_contact_w: np.ndarray, f_cmd_w: np.ndarray, f_ext_w: np.ndarray, k: float = STIFFNESS_K
) -> np.ndarray:
    """x_target = x_contact + (F_cmd + F_ext) / K, all world-frame 3-D vectors."""
    return np.asarray(x_contact_w, dtype=float) + (np.asarray(f_cmd_w, dtype=float) + np.asarray(f_ext_w, dtype=float)) / k


def target_shift_world(f_cmd_w: np.ndarray, f_ext_w: np.ndarray, k: float = STIFFNESS_K) -> np.ndarray:
    """The pure shift vector (F_cmd + F_ext)/K — anchor-independent for 3-D force."""
    return (np.asarray(f_cmd_w, dtype=float) + np.asarray(f_ext_w, dtype=float)) / k


# ---------------------------------------------------------------------------
# Per-frame bundle: everything the viser scene needs for one frame.
# ---------------------------------------------------------------------------
@dataclass
class FrameTargets:
    """All rendered quantities for one frame + hand (world frame unless noted)."""

    t: int
    phase: str
    f_cmd_world: np.ndarray  # 3-D commanded force, applied at the TCP
    # 3-D external/measured force, DRAWN at the FT-sensor anchor (that is where a
    # real sensor reads it). NOTE: in v0 this is SCRIPTED in the pelvis frame and
    # rotated to world — it is NOT a sensor-frame reading transformed via the
    # adjoint. The sensor->TCP adjoint (imported, re-exported) stays inert until
    # M1 supplies a real sensor-frame F_meas and the 6-D moment channels (spec §2.1/§5).
    f_ext_world: np.ndarray
    p_tcp: np.ndarray  # TCP contact anchor
    p_ft: np.ndarray  # FT-sensor anchor (the F_meas application point)
    x_target_tcp: np.ndarray  # Eq.2 target anchored at the TCP
    x_target_ft: np.ndarray  # Eq.2 target anchored at the FT sensor
    shift: np.ndarray  # (F_cmd+F_ext)/K
    wrist_pos_w: np.ndarray  # wrist_yaw origin (world), reference
    wrist_quat_w: np.ndarray  # wrist_yaw orientation (world) wxyz, reference


def frame_targets(
    motion: Motion,
    t: int,
    side: str,
    direction_pelvis: np.ndarray,
    peak_N: float,
    spec: PulseSpec,
    scenario: str,
    k: float = STIFFNESS_K,
) -> FrameTargets:
    """Assemble every world-frame quantity the scene renders for frame ``t``.

    F_cmd is authored in the pelvis frame and rotated to world by the live pelvis
    quaternion (matches force_tracking). F_ext is authored in the pelvis frame too,
    then rotated to world. The TCP and FT anchors both use the live wrist pose.
    """
    p_wrist, q_wrist = motion.wrist_pose(t, side)
    R_pelvis = quat_to_mat_wxyz(motion.pelvis_quat(t))

    f_cmd_p = f_cmd_pelvis(t, direction_pelvis, peak_N, spec)
    f_ext_p = f_ext_pelvis(t, f_cmd_p, spec, scenario)
    f_cmd_w = R_pelvis @ f_cmd_p
    f_ext_w = R_pelvis @ f_ext_p

    p_tcp = contact_point_world(p_wrist, q_wrist, TCP_OFFSET_X)
    p_ft = contact_point_world(p_wrist, q_wrist, FT_OFFSET_X)
    shift = target_shift_world(f_cmd_w, f_ext_w, k)
    return FrameTargets(
        t=t,
        phase=pulse_phase(t, spec),
        f_cmd_world=f_cmd_w,
        f_ext_world=f_ext_w,
        p_tcp=p_tcp,
        p_ft=p_ft,
        x_target_tcp=p_tcp + shift,
        x_target_ft=p_ft + shift,
        shift=shift,
        wrist_pos_w=p_wrist,
        wrist_quat_w=q_wrist,
    )


# ---------------------------------------------------------------------------
# Arm-only inverse kinematics for the transparent-orange target ghost.
#
# Solves the active side's 7 arm joints so the wrist_yaw frame reaches the shifted
# pose (position = shifted wrist origin, orientation = reference wrist orientation —
# Eq.2 shifts position, orientation is tracked separately), seeded from the
# reference arm joints. Everything else (legs, waist, other arm, base) stays at
# reference. pinocchio is fixed-base, so its world == the pelvis frame; we express
# the target in the pelvis frame using the reference pelvis pose.
# ---------------------------------------------------------------------------
@dataclass
class IKResult:
    q_arm: dict[str, float]  # solved arm joint angles (name → rad), active side
    residual_m: float  # final position error (m)
    converged: bool
    ori_err_rad: float = 0.0  # wrist orientation error vs reference (soft-follow quality)


class ArmIK:
    """pinocchio damped-least-squares IK on one arm chain. Lazily builds the model."""

    def __init__(
        self,
        urdf_path: str,
        damping: float = 1e-3,
        max_iter: int = 400,
        tol_m: float = 1e-4,
        k_ori: float = 0.5,
        k_post: float = 0.3,
    ):
        import pinocchio as pin  # lazy: frame-math tests don't need pinocchio

        self._pin = pin
        self.model = pin.buildModelFromUrdf(urdf_path)  # fixed base ⇒ world = pelvis
        self.damping = float(damping)
        self.max_iter = int(max_iter)
        self.tol_m = float(tol_m)
        # Hierarchical-IK secondary-task gains (see solve()): orientation-follow and
        # posture-bias, both projected into the position task's nullspace.
        self.k_ori = float(k_ori)
        self.k_post = float(k_post)
        # Map every model joint name → (q index, v index) for building configs.
        self._q_idx: dict[str, int] = {}
        self._v_idx: dict[str, int] = {}
        for jid in range(1, self.model.njoints):  # skip universe (0)
            jname = self.model.names[jid]
            jm = self.model.joints[jid]
            if jm.nq == 1:  # revolute — the only kind we vary
                self._q_idx[jname] = jm.idx_q
                self._v_idx[jname] = jm.idx_v

        # Add a TCP operational frame per side, fixed to wrist_yaw at (TCP_OFFSET_X,0,0).
        # The IK drives THIS frame's position to the shifted TCP target, so the grasp
        # point reaches the target regardless of the (soft) wrist-orientation drift —
        # avoiding the "origin + R_wrist·lever" reconstruction error that a wrist-origin
        # position task would incur once orientation is only softly followed.
        self._tcp_frame_id: dict[str, int] = {}
        for side, link in WRIST_YAW_LINK.items():
            wid = self.model.getFrameId(link)
            wf = self.model.frames[wid]
            placement = wf.placement * pin.SE3(np.eye(3), np.array([TCP_OFFSET_X, 0.0, 0.0]))
            frame = pin.Frame(f"{side}_tcp_ik", wf.parent, wid, placement, pin.FrameType.OP_FRAME)
            self._tcp_frame_id[side] = self.model.addFrame(frame)
        self.data = self.model.createData()  # AFTER frames added

    def _q0_from_reference(self, motion: Motion, dof_ref: np.ndarray) -> np.ndarray:
        """Neutral config with every actuated joint set to its reference value.

        ``dof_ref`` is the 29-vector in npz joint order; gripper jaws (not in the
        npz) stay at the pinocchio neutral (0)."""
        q = self._pin.neutral(self.model)
        name_to_val = dict(zip(motion.joint_names, dof_ref))
        for jname, qi in self._q_idx.items():
            if jname in name_to_val:
                q[qi] = float(name_to_val[jname])
        return q

    def solve(self, motion: Motion, t: int, side: str, x_target_tcp_w: np.ndarray) -> IKResult:
        """Hierarchical arm IK so the TCP reaches ``x_target_tcp_w`` (world grasp target).

        The position task drives the **TCP operational frame** (0.242 m fwd of wrist_yaw)
        directly to the shifted target, so the grasp point lands on the target regardless
        of the soft wrist-orientation drift (no ``origin + R·lever`` reconstruction error).

        Task hierarchy (design §4.1 — Eq.2 shifts POSITION; orientation is tracked by a
        SEPARATE reward, so the ghost must not counter-rotate the wrist just to hold a
        hard orientation lock):

          * PRIMARY  — the 3-DOF TCP position task, solved with a damped pseudo-inverse.
            Keeps the grasp point accurate to ~mm.
          * SECONDARY (projected into the position nullspace, so it never fights the
            primary): (a) an orientation-FOLLOW term pulling the wrist toward its
            REFERENCE orientation with gain ``k_ori``, and (b) a posture-bias term
            pulling the arm toward its reference (seed) joints with gain ``k_post`` to
            resolve the remaining redundancy naturally.

        Why not the old 6-DOF hard pose lock: empirically it forced the wrist to
        counter-rotate (``wrist_pitch`` flipping the "wrong" way, elbow over-swinging)
        to satisfy a mm/deg-exact orientation the task does not actually require — the
        "施加 force 像掰关节而不是关节电机" artifact. With orientation as a soft
        nullspace goal the ghost keeps mm TCP position, follows the reference orientation
        within a few degrees, and the wrist relaxes to its natural sign.

        Returns the solved arm angles + the TCP POSITION residual (the HUD quantity).
        """
        pin = self._pin
        dof_ref = motion.dof(t)
        q = self._q0_from_reference(motion, dof_ref)

        # Express the target in the pelvis frame (pinocchio world), using the
        # reference pelvis pose. Orientation goal = reference wrist orientation.
        p_pelvis = motion.base_pos(t)
        R_pelvis = quat_to_mat_wxyz(motion.pelvis_quat(t))
        _, q_wrist_w = motion.wrist_pose(t, side)
        R_wrist_w = quat_to_mat_wxyz(q_wrist_w)
        p_des_pelvis = R_pelvis.T @ (np.asarray(x_target_tcp_w, dtype=float) - p_pelvis)
        R_des_pelvis = R_pelvis.T @ R_wrist_w

        tcp_id = self._tcp_frame_id[side]  # drive the TCP frame's POSITION
        wrist_id = self.model.getFrameId(WRIST_YAW_LINK[side])  # follow its ORIENTATION
        arm_v = [self._v_idx[f"{side}_{j}_joint"] for j in ARM_JOINTS if f"{side}_{j}_joint" in self._v_idx]
        arm_q = [self._q_idx[f"{side}_{j}_joint"] for j in ARM_JOINTS if f"{side}_{j}_joint" in self._q_idx]
        arm_v_arr = np.array(arm_v, dtype=int)
        arm_q_arr = np.array(arm_q, dtype=int)
        q_seed_arm = q[arm_q_arr].copy()
        na = arm_v_arr.size
        eye_na = np.eye(na)

        for _ in range(self.max_iter):
            pin.forwardKinematics(self.model, self.data, q)
            pin.updateFramePlacements(self.model, self.data)
            # World-frame (LOCAL_WORLD_ALIGNED) Jacobians so the rows commute with the
            # world-frame errors below; slice to arm dofs.
            jac_tcp = pin.computeFrameJacobian(self.model, self.data, q, tcp_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
            jac_wr = pin.computeFrameJacobian(self.model, self.data, q, wrist_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
            jp = jac_tcp[:3, arm_v_arr]  # TCP position rows
            jo = jac_wr[3:, arm_v_arr]  # wrist orientation rows

            e_pos = self.data.oMf[tcp_id].translation - p_des_pelvis  # world-frame TCP error
            # Primary (position) step uses a DAMPED pseudo-inverse (robust near arm
            # singularities). The nullspace PROJECTOR, however, must use the UNDAMPED
            # pseudo-inverse: a damped inverse does not annihilate the primary
            # direction (N@Jp ≠ 0), so the secondary tasks would leak into and creep
            # the position task. The 3×3 Gram matrix jp@jpᵀ is well-conditioned (a
            # 3-DOF task on a 7-DOF arm), so the undamped inverse is numerically safe
            # here; a small floor guards the rare near-singular frame.
            jjt = jp @ jp.T
            jp_pinv_damped = jp.T @ np.linalg.inv(jjt + self.damping * np.eye(3))
            jp_pinv_true = jp.T @ np.linalg.inv(jjt + 1e-9 * np.eye(3))
            dq_primary = -jp_pinv_damped @ e_pos
            nullspace = eye_na - jp_pinv_true @ jp

            # Secondary: orientation-follow (world ang error) + posture bias.
            e_ori = pin.log3(self.data.oMf[wrist_id].rotation @ R_des_pelvis.T)  # world ang error
            dq_secondary = -self.k_ori * (jo.T @ e_ori) + self.k_post * (q_seed_arm - q[arm_q_arr])

            v_arm = dq_primary + nullspace @ dq_secondary
            # Converged once the position is within tol_m AND the step has settled
            # (the nullspace secondary may still be relaxing after position is met).
            if np.linalg.norm(e_pos) < self.tol_m and np.linalg.norm(v_arm) < 1e-6:
                break
            if np.linalg.norm(v_arm) < 1e-7:
                break
            v_full = np.zeros(self.model.nv)
            v_full[arm_v_arr] = v_arm
            q = pin.integrate(self.model, q, v_full)

        # Report the TCP POSITION residual (m) — the HUD quantity — plus the wrist
        # orientation error so callers/tests can see the soft-follow quality.
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        pos_res = float(np.linalg.norm(self.data.oMf[tcp_id].translation - p_des_pelvis))
        ori_err = float(np.linalg.norm(pin.log3(self.data.oMf[wrist_id].rotation @ R_des_pelvis.T)))
        q_arm = {
            f"{side}_{j}_joint": float(q[self._q_idx[f"{side}_{j}_joint"]])
            for j in ARM_JOINTS
            if f"{side}_{j}_joint" in self._q_idx
        }
        return IKResult(q_arm=q_arm, residual_m=pos_res, converged=pos_res < 2e-2, ori_err_rad=ori_err)

__all__ = [
    "STIFFNESS_K",
    "TCP_OFFSET_X",
    "FT_OFFSET_X",
    "FT_TCP_LEVER_X",
    "ARM_JOINTS",
    "WRIST_YAW_LINK",
    "quat_to_mat_wxyz",
    "quat_apply_wxyz",
    "Motion",
    "load_motion",
    "contact_point_world",
    "PulseSpec",
    "pulse_scale",
    "pulse_phase",
    "f_cmd_pelvis",
    "f_ext_pelvis",
    "eq2_target_world",
    "target_shift_world",
    "FrameTargets",
    "frame_targets",
    "IKResult",
    "ArmIK",
    "adjoint_sensor_to_tcp",
    "_R_sensor",
]
