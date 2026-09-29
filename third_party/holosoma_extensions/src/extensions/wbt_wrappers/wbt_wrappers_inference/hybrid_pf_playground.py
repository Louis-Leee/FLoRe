"""Interactive MuJoCo playground for the hybrid position-force controller.

A no-training, no-policy *wrench probe* for the Unitree G1 + YAM gripper. You
push / pull / twist the wrist by hand (or via keyboard-stepped 6D wrench) to feel
the F/T-sensor reading ``F_meas`` (= the external wrench ``F_ext``), while dragging
a commanded 6D wrench ``F_cmd`` on keyboard steppers. The HUD shows the same
wrench in three frames (applied world / raw sensor / TCP-adjoint) plus the joint
torques it induces, so you can *see* the sensor->TCP frame transform (§2 of the
design doc) and feel that the 5 N.m wrists are the binding constraint on moment
commands (no lever amplification -- a 6 N.m moment loads a weak wrist up to its
full 6 N.m only when the command axis aligns with that wrist axis, so whether it
saturates is pose-dependent; empirically 12/22 sampled poses stay under 5 N.m).

Design doc: docs/2026-07-08-hybrid-position-force-controller-design.md §10.
This module is the v0 there, folding in 4 Opus reviews. Key correctness points
(all empirically verified in the hssim env, mujoco 3.8.1):

  * The user wrench is injected via ``data.xfrc_applied`` as a BODY wrench
    (force + moment-arm torque), NOT ``mj_applyFT``/``qfrc_applied`` -- the latter
    is invisible to the force sensor (it never enters cfrc_ext / the RNE force
    transmission the sensor reads).
  * MuJoCo ``force``/``torque`` site sensors report the wrench transmitted through
    the site's body from its PARENT, i.e. the PARENT-on-child reaction. To display
    ``F_meas`` in the design's "external wrench ON the robot" convention we NEGATE
    the raw sensor (see ``sensor_to_fmeas``).
  * The FT site orientation is the quaternion ``(0, 0.707107, 0, 0.707107)`` (wxyz)
    = R_sensor; the euler form is a DIFFERENT rotation (flips the approach axis) --
    do not use euler here.
  * All model edits (sites, force/torque sensors, pelvis weld) are injected at
    LOAD TIME via MjSpec; the vendored thirdparty/ MJCF is never edited in place.

Frontend is native ``mujoco.viewer`` -- run this on a machine WITH a display
(your laptop), not the headless training box.

Run:  python -m wbt_wrappers_inference.hybrid_pf_playground
      (macOS: use ``mjpython -m wbt_wrappers_inference.hybrid_pf_playground`` --
       mujoco.viewer.launch_passive requires mjpython on macOS.)
Env:  hssim conda env (mujoco 3.8.1).
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np

logger = logging.getLogger(__name__)

# --- geometry constants (design doc §1.1, verified vs g1_29dof_yam.urdf) --------
# FT sensor site: wrist_yaw +x 0.082 m, orientation R_sensor (sensor +z -> wrist +x).
FT_SITE_POS = (0.082, 0.0, 0.0)
FT_SITE_QUAT_WXYZ = (0.0, 0.707107, 0.0, 0.707107)  # == R_sensor; NOT the euler form.
# TCP (fingertip grasp midpoint): wrist_yaw +x 0.242 m, orientation == wrist.
# (2026-07-10: hand_palm_joint standoff 0.0295->0.0565 moved TCP 0.215->0.242; FT sensor
# stayed at 0.082. See docs/2026-07-08-hybrid-position-force-controller-design.md §1.1.)
TCP_SITE_POS = (0.242, 0.0, 0.0)
# Lever arm FT->TCP along wrist +x: 0.242 - 0.082 = 0.16 m (was 0.133) — the 6D moment term.
FT_TCP_LEVER_X = TCP_SITE_POS[0] - FT_SITE_POS[0]

WRIST_BODIES = ("left_wrist_yaw_link", "right_wrist_yaw_link")
HANDS = ("left", "right")

# Arm joints whose induced torque we watch against effort limits (design §5.1/D7).
# wrist pitch/yaw are the weak 5 N.m joints; the rest are 25 N.m.
_ARM_JOINTS = ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll", "wrist_pitch", "wrist_yaw")
_WRIST_WEAK = ("wrist_pitch", "wrist_yaw")

# Default scene (floating base + floor + full meshes + 29 actuators).
_DEFAULT_SCENE = (
    Path(__file__).resolve().parents[4]
    / "thirdparty/holosoma/src/holosoma/holosoma/data/robots/g1"
    / "scenes/scene_g1_29dof_wbt_plane_yam.xml"
)

# Demo pose (design §10.9 choice 2). FK-verified: the yam URDF's q=0 ALREADY has the
# upper arm vertical (down) and the forearm horizontal-forward (TCP at x=0.415, chest
# height, clear of the body) -- i.e. the requested "大臂垂直, 小臂水平" is the zero pose.
# So the demo-pose key applies a modest shoulder-pitch reach to give a visibly distinct
# forward-extended pose (better lever arms for the frame/torque lessons) rather than the
# earlier (wrong) elbow=90deg, which points the forearm DOWN.
_DEMO_SHOULDER_PITCH_RAD = -0.6  # lift the upper arm forward/up ~34deg (elbow stays 0).


@dataclass
class PlaygroundConfig:
    """Knobs for the wrench probe."""

    scene_path: Path = _DEFAULT_SCENE
    gravity: bool = False  # design §10.9 choice 3: OFF (cleanest, pure external force).
    # Rigid-instrument hold. A wrench applied at the TCP through a torque-controlled
    # arm is not statically held by a torque PD at 500 Hz -- it goes unstable and the
    # FT sensor then reads the m*a of the thrashing chain, not the applied force
    # (verified: stiffer PD makes |qacc| WORSE). The robust fix for a *probe* is to
    # make the chain quasi-RIGID via large joint armature, so the sensor reads the
    # true static transmitted wrench. At the shipped armature=1e4 the readback is
    # essentially exact (verified: 10 N applied -> 10.00 N read, err ~1e-4 N; the
    # earlier "9.6 N" figure was a compliant-arm / short-settle transient). A light
    # PD then just holds the pose against the tiny residual.
    arm_armature: float = 1.0e4  # quasi-rigid arm/joints (skip the 6 base dof).
    hold_kp: float = 150.0
    hold_kd: float = 30.0
    # Wrench command ranges for the keyboard steppers.
    force_max_n: float = 25.0
    torque_max_nm: float = 5.0  # wrist-limited; feel how small a real moment must be.
    force_step_n: float = 1.0
    torque_step_nm: float = 0.25
    active_hand: str = "right"  # v0: single active hand (design §8.6).


@dataclass
class WrenchState:
    """Mutable 6D wrench steppers (per hand), all in the TCP frame convention.

    F_ext is what the USER applies (becomes F_meas). F_cmd is the commanded wrench
    (INERT in this probe -- no policy -- it only draws an arrow + feeds the error).
    """

    f_ext: np.ndarray = field(default_factory=lambda: np.zeros(6))  # [fx,fy,fz,tx,ty,tz]
    f_cmd: np.ndarray = field(default_factory=lambda: np.zeros(6))


def build_model(cfg: PlaygroundConfig) -> mujoco.MjModel:
    """Load the scene and inject FT/TCP sites, force/torque sensors, and a pelvis
    weld -- all via MjSpec at load time (never editing the vendored MJCF).

    Returns a compiled MjModel with, per hand: sites ``ft_{side}`` / ``tcp_{side}``
    on the wrist_yaw body, and sensors ``ft_{side}_force`` / ``ft_{side}_torque``.
    """
    if not cfg.scene_path.exists():
        raise SystemExit(f"scene not found: {cfg.scene_path}\n(need the yam scene from thirdparty/holosoma)")
    spec = mujoco.MjSpec.from_file(str(cfg.scene_path))

    for side, wrist_body in zip(HANDS, WRIST_BODIES):
        body = spec.body(wrist_body)
        body.add_site(name=f"ft_{side}", pos=list(FT_SITE_POS), quat=list(FT_SITE_QUAT_WXYZ))
        body.add_site(name=f"tcp_{side}", pos=list(TCP_SITE_POS))
        for stype, sname in (
            (mujoco.mjtSensor.mjSENS_FORCE, f"ft_{side}_force"),
            (mujoco.mjtSensor.mjSENS_TORQUE, f"ft_{side}_torque"),
        ):
            sensor = spec.add_sensor()
            sensor.name = sname
            sensor.type = stype
            sensor.objtype = mujoco.mjtObj.mjOBJ_SITE
            sensor.objname = f"ft_{side}"

    # Rigid base: weld pelvis to world. A default weld is a *soft* equality (drifts
    # ~cm under load); we stiffen solref/solimp so the base is effectively rigid
    # (design §10.9 choice 1). With gravity off + quasi-rigid arm the drift is already
    # negligible; this makes it rock-solid regardless.
    weld = spec.add_equality()
    weld.name = "pin_pelvis"
    weld.type = mujoco.mjtEq.mjEQ_WELD
    weld.objtype = mujoco.mjtObj.mjOBJ_BODY
    weld.name1 = "pelvis"
    weld.name2 = ""  # empty == world body
    weld.solref = [0.001, 1.0]  # fast, near-rigid constraint (default is [0.02, 1]).
    weld.solimp = [0.999, 0.9999, 1.0e-4, 0.5, 2.0]

    model = spec.compile()
    if not cfg.gravity:
        model.opt.gravity[:] = 0.0
    # Quasi-rigid chain (skip the 6 base dof). See PlaygroundConfig.arm_armature.
    model.dof_armature[6:] = cfg.arm_armature
    return model


class PlaygroundModel:
    """Resolved ids + per-step physics for the wrench probe (viewer-agnostic).

    Kept separate from the viewer loop so the physics/frame/sensor logic is
    headlessly testable (the native viewer only runs on a machine with a display).
    """

    def __init__(self, model: mujoco.MjModel, cfg: PlaygroundConfig):
        if cfg.active_hand not in HANDS:
            raise ValueError(f"active_hand must be one of {HANDS}, got {cfg.active_hand!r}")
        self.model = model
        self.cfg = cfg
        self.data = mujoco.MjData(model)

        self.wrist_bid = {s: _body_id(model, b) for s, b in zip(HANDS, WRIST_BODIES)}
        self.tcp_sid = {s: _site_id(model, f"tcp_{s}") for s in HANDS}
        self.force_adr = {s: _sensor_adr(model, f"ft_{s}_force") for s in HANDS}
        self.torque_adr = {s: _sensor_adr(model, f"ft_{s}_torque") for s in HANDS}

        # Arm joint dof addresses for the induced-torque readout.
        self.arm_dof = {s: _arm_dof_ids(model, s) for s in HANDS}

        # PD hold target = the reset (all-zero) qpos, optionally the demo pose.
        mujoco.mj_resetData(model, self.data)
        self.hold_qpos = self.data.qpos.copy()
        self.state = WrenchState()

    # ---- pose control ------------------------------------------------------
    def set_demo_pose(self) -> None:
        """A forward-extended reach pose (design §10.9 choice 2).

        q=0 already gives upper-arm-vertical / forearm-horizontal; this lifts the
        shoulder pitch so the arm reaches forward -- a visibly distinct pose with
        cleaner lever arms for the frame/torque lessons. Elbow stays 0.
        """
        for side in HANDS:
            adr = _joint_qposadr(self.model, f"{side}_shoulder_pitch_joint")
            self.hold_qpos[adr] = _DEMO_SHOULDER_PITCH_RAD
        self.reset_pose()

    def reset_pose(self) -> None:
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = self.hold_qpos
        self.state.f_ext[:] = 0.0
        self.state.f_cmd[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

    # ---- per-step physics --------------------------------------------------
    def step(self) -> None:
        """One physics step: PD-hold all joints, inject the user F_ext at the TCP
        as a body wrench via xfrc_applied, integrate."""
        self._apply_pd_hold()
        self._apply_user_wrench()
        mujoco.mj_step(self.model, self.data)

    def _apply_pd_hold(self) -> None:
        m, d = self.model, self.data
        # PD on every actuated joint toward hold_qpos (qvel target 0).
        # actuators are 1:1 with joints in this MJCF (nu=29); use ctrl.
        for j in range(m.nu):
            jnt = m.actuator_trnid[j, 0]
            qadr = m.jnt_qposadr[jnt]
            vadr = m.jnt_dofadr[jnt]
            err = self.hold_qpos[qadr] - d.qpos[qadr]
            d.ctrl[j] = self.cfg.hold_kp * err - self.cfg.hold_kd * d.qvel[vadr]

    def _apply_user_wrench(self) -> None:
        """Write the active hand's F_ext (TCP frame) into xfrc_applied as a world
        body wrench at the wrist COM (force + moment-arm torque about COM).

        We only overwrite the active wrist body's xfrc row each step, leaving other
        bodies' rows (e.g. the viewer's mouse perturbation on a *different* body)
        intact. Note: if the user Ctrl-drags the *active* wrist body, this per-step
        overwrite wins over the mouse force -- they don't sum on that one body.
        """
        d = self.data
        side = self.cfg.active_hand
        bid = self.wrist_bid[side]
        tcp_sid = self.tcp_sid[side]

        # F_ext is authored in the TCP frame; rotate to world using the TCP site.
        R_tcp = d.site_xmat[tcp_sid].reshape(3, 3)
        f_world = R_tcp @ self.state.f_ext[:3]
        m_world = R_tcp @ self.state.f_ext[3:6]

        # Equivalent body wrench at the wrist COM: force plus (r_tcp_from_com x F)
        # so the force acts AT the TCP point (preserves the lever arm the sensor reads).
        com = d.xipos[bid]
        tcp = d.site_xpos[tcp_sid]
        d.xfrc_applied[bid, :3] = f_world
        d.xfrc_applied[bid, 3:6] = m_world + np.cross(tcp - com, f_world)

    # ---- readouts (design §10.4 HUD) --------------------------------------
    def read_fmeas_sensor(self, side: str) -> np.ndarray:
        """Raw F/T sensor reading in the SENSOR (site) frame, 6D [f(3), tau(3)].

        This is the PARENT-on-child reaction (see module docstring). Use
        ``sensor_to_fmeas`` to convert to the design's external-on-robot convention.
        """
        d = self.data
        fa, ta = self.force_adr[side], self.torque_adr[side]
        return np.concatenate([d.sensordata[fa : fa + 3], d.sensordata[ta : ta + 3]])

    def fmeas_external_sensor_frame(self, side: str) -> np.ndarray:
        """F_meas in the external-on-robot convention (§1.2), still SENSOR frame."""
        return sensor_to_fmeas(self.read_fmeas_sensor(side))

    def fmeas_tcp_frame(self, side: str) -> np.ndarray:
        """F_meas mapped SENSOR frame -> TCP frame via the constant 6x6 wrench
        adjoint (design §2.2): rotation R_sensor plus the lever-arm moment term."""
        w_sensor = self.fmeas_external_sensor_frame(side)
        return adjoint_sensor_to_tcp(w_sensor)

    def applied_wrench_world(self, side: str) -> np.ndarray:
        """The user-applied wrench expressed in WORLD frame (HUD row 1)."""
        d = self.data
        R_tcp = d.site_xmat[self.tcp_sid[side]].reshape(3, 3)
        return np.concatenate([R_tcp @ self.state.f_ext[:3], R_tcp @ self.state.f_ext[3:6]])

    def arm_joint_torques(self, side: str) -> dict[str, float]:
        """Joint torques induced by the applied TCP wrench, via J^T (design D7).

        tau = J_tcp^T @ w_world (6D wrench at the TCP site, world frame). Highlights
        the weak wrist pitch/yaw against their 5 N.m limit -- the only way to *feel*
        6D-moment infeasibility in a no-servo probe.
        """
        m, d = self.model, self.data
        jacp = np.zeros((3, m.nv))
        jacr = np.zeros((3, m.nv))
        mujoco.mj_jacSite(m, d, jacp, jacr, self.tcp_sid[side])
        R_tcp = d.site_xmat[self.tcp_sid[side]].reshape(3, 3)
        w_world = np.concatenate([R_tcp @ self.state.f_ext[:3], R_tcp @ self.state.f_ext[3:6]])
        J = np.vstack([jacp, jacr])  # (6, nv)
        tau_full = J.T @ w_world  # (nv,)
        out = {}
        for name, dofadr in self.arm_dof[side].items():
            out[name] = float(tau_full[dofadr])
        return out

    def error_tcp(self, side: str) -> np.ndarray:
        """e = F_cmd + F_meas in the TCP frame (design §1.2/§10.1).

        NOTE (design §10.1): in this probe you can only null e by hand-matching the
        slider -- this is NOT loop closure. It is the residual the trained controller
        will null by acting on the environment.
        """
        return self.state.f_cmd + self.fmeas_tcp_frame(side)


# --- pure wrench/frame helpers (headlessly testable, no MjData needed) ----------
def sensor_to_fmeas(w_sensor_raw: np.ndarray) -> np.ndarray:
    """MuJoCo force/torque sensors report the parent-on-child reaction. Negate to
    get F_meas = the external wrench ON the robot (design §1.2 convention)."""
    return -np.asarray(w_sensor_raw, dtype=float)


def _R_sensor() -> np.ndarray:
    """Sensor->wrist rotation from the URDF ft_sensor_joint rpy=(0,-pi/2,pi).

    Symmetric involution: sensor +z->wrist +x, +x->wrist +z, +y->wrist -y.
    (v_wrist = R_sensor @ v_sensor.)
    """
    return np.array([[0.0, 0.0, 1.0], [0.0, -1.0, 0.0], [1.0, 0.0, 0.0]])


def adjoint_sensor_to_tcp(w_sensor: np.ndarray) -> np.ndarray:
    """Map a 6D wrench from the SENSOR frame to the TCP frame (design §2.2).

    Both frames are rigidly fixed to wrist_yaw, so this is a CONSTANT 6x6 adjoint:
        f_T = R_sensor @ f_S
        m_T = R_sensor @ m_S - (0.16, 0, 0) x f_T
    where R_sensor = R^T_S (== R_sensor, involution) and 0.16 = 0.242 - 0.082 is the
    FT->TCP lever along wrist +x (2026-07-10: was 0.133). The lever term is the 6D-specific moment coupling.
    """
    w = np.asarray(w_sensor, dtype=float)
    R = _R_sensor()
    f_t = R @ w[:3]
    lever = np.array([FT_TCP_LEVER_X, 0.0, 0.0])
    m_t = R @ w[3:6] - np.cross(lever, f_t)
    return np.concatenate([f_t, m_t])


# --- id resolution helpers ------------------------------------------------------
def _body_id(model: mujoco.MjModel, name: str) -> int:
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    if bid < 0:
        raise ValueError(f"body {name!r} not found")
    return bid


def _site_id(model: mujoco.MjModel, name: str) -> int:
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
    if sid < 0:
        raise ValueError(f"site {name!r} not found")
    return sid


def _sensor_adr(model: mujoco.MjModel, name: str) -> int:
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, name)
    if sid < 0:
        raise ValueError(f"sensor {name!r} not found")
    return int(model.sensor_adr[sid])


def _joint_qposadr(model: mujoco.MjModel, name: str) -> int:
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    if jid < 0:
        raise ValueError(f"joint {name!r} not found")
    return int(model.jnt_qposadr[jid])


def _arm_dof_ids(model: mujoco.MjModel, side: str, only=_ARM_JOINTS) -> dict[str, int]:
    """Map arm joint short-name -> dof (velocity) address for the given side."""
    out = {}
    for short in only:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{side}_{short}_joint")
        if jid >= 0:
            out[short] = int(model.jnt_dofadr[jid])
    return out


# --- HUD text -------------------------------------------------------------------
def _fmt6(w: np.ndarray) -> str:
    return "F[{:+.1f} {:+.1f} {:+.1f}]N  M[{:+.2f} {:+.2f} {:+.2f}]Nm".format(*w)


def hud_lines(pm: PlaygroundModel) -> list[str]:
    """The frame + torque readout panel (design §10.4)."""
    side = pm.cfg.active_hand
    lines = [
        f"active hand: {side}   (F_cmd is INERT here: no policy -- arrow/error only)",
        f"applied   (world): {_fmt6(pm.applied_wrench_world(side))}",
        f"F_meas raw(sensor): {_fmt6(pm.fmeas_external_sensor_frame(side))}",
        f"F_meas -> TCP     : {_fmt6(pm.fmeas_tcp_frame(side))}",
        f"F_cmd     (TCP)   : {_fmt6(pm.state.f_cmd)}",
        f"e=Fcmd+Fmeas(TCP) : {_fmt6(pm.error_tcp(side))}  (null by HAND only -- not loop closure)",
    ]
    tau = pm.arm_joint_torques(side)
    weak = "  ".join(
        f"{n}={tau[n]:+.2f}/{5.0 if n in _WRIST_WEAK else 25.0:.0f}Nm"
        + ("!" if abs(tau[n]) > (5.0 if n in _WRIST_WEAK else 25.0) else "")
        for n in tau
    )
    lines.append(f"joint tau (J^T w): {weak}")
    return lines


_AXIS_NAMES = ["fx", "fy", "fz", "tx", "ty", "tz"]


def _run_viewer(pm: PlaygroundModel) -> None:  # pragma: no cover (needs display)
    """Native mujoco.viewer loop. Run on a machine WITH a display (your laptop).

    NOTE: ``launch_passive`` runs the key callback on a SEPARATE thread. GLFW also
    delivers raw physical keycodes with NO modifier info (Shift+1 == keycode for 1),
    so we cannot read Shift. Therefore:
      * the callback only stages lightweight state (wrench element nudges are atomic
        numpy writes; pose changes go through ``pending`` flags applied by the main
        thread) -- never call mj_reset/forward from the callback (data race);
      * we edit F_ext vs F_cmd via a MODE TOGGLE ([c]) instead of Shift+digit.
    """
    import time

    import mujoco.viewer

    cfg = pm.cfg
    ui = {"axis": 0, "edit_cmd": False}  # selected wrench axis; editing F_cmd vs F_ext.
    pending = {"reset": False, "demo": False, "lateral": False, "log": False}

    def _log_hud() -> None:
        logger.info("[editing %s, axis %s]", "F_cmd" if ui["edit_cmd"] else "F_ext", _AXIS_NAMES[ui["axis"]])
        for ln in hud_lines(pm):
            logger.info(ln)

    def key_cb(keycode: int) -> None:
        ch = chr(keycode) if 0 <= keycode < 0x110000 else ""
        st = pm.state
        step = cfg.force_step_n if ui["axis"] < 3 else cfg.torque_step_nm
        target = st.f_cmd if ui["edit_cmd"] else st.f_ext
        if ch in "123456":  # select wrench axis
            ui["axis"] = int(ch) - 1
            logger.info("axis -> %s", _AXIS_NAMES[ui["axis"]])
        elif ch in ("c", "C"):  # toggle which wrench the +/- keys edit
            ui["edit_cmd"] = not ui["edit_cmd"]
            logger.info("editing -> %s", "F_cmd (INERT)" if ui["edit_cmd"] else "F_ext")
        elif ch in ("=", "+"):
            target[ui["axis"]] += step
            _log_hud()
        elif ch in ("-", "_"):
            target[ui["axis"]] -= step
            _log_hud()
        elif ch in ("l", "L"):  # pure lateral force @ TCP demo (design §10.5)
            pending["lateral"] = True
        elif ch in ("r", "R"):
            pending["reset"] = True
        elif ch in ("p", "P"):
            pending["demo"] = True
        elif ch in ("g", "G"):
            pending["log"] = True

    banner = (
        "\n===== HYBRID POSITION-FORCE PLAYGROUND (wrench probe; no policy) =====\n"
        "  Click the window, then:\n"
        "   [1..6]  select wrench axis (fx fy fz tx ty tz, TCP frame)\n"
        "   [c]     toggle editing F_ext <-> F_cmd\n"
        "   [=/-]   nudge the selected axis of the active wrench up/down\n"
        "   [l]     pure lateral force @ TCP  -> watch the phantom sensor moment\n"
        "   [p]     demo pose (upper arm vertical, forearm horizontal)\n"
        "   [r]     reset pose + zero F_ext & F_cmd     [g] log readings to stdout\n"
        "  F_cmd is INERT (no policy): it only draws a purple arrow + feeds e=Fcmd+Fmeas.\n"
        "  You can also Ctrl-drag a NON-active body in the viewer (uncalibrated mouse force).\n"
        "======================================================================"
    )
    logger.info(banner)
    logger.info("axis order: %s", _AXIS_NAMES)

    def _draw_arrows(viewer) -> None:
        # Draw F_ext (orange) and F_cmd (purple) at the active-hand TCP.
        from wbt_wrappers_inference.utils import eval_utils as eu  # local import: optional dep.

        side = cfg.active_hand
        tcp = pm.data.site_xpos[pm.tcp_sid[side]]
        R_tcp = pm.data.site_xmat[pm.tcp_sid[side]].reshape(3, 3)
        eu._add_force_arrow(viewer.user_scn, tcp, R_tcp @ pm.state.f_ext[:3], eu.FORCE_ARROW_RGBA_EXT)
        eu._add_force_arrow(viewer.user_scn, tcp, R_tcp @ pm.state.f_cmd[:3], eu.FORCE_ARROW_RGBA_CMD)

    dt = pm.model.opt.timestep
    with mujoco.viewer.launch_passive(pm.model, pm.data, key_callback=key_cb) as viewer:
        while viewer.is_running():
            t0 = time.time()
            # Apply pending pose/state changes HERE (main thread) -- never in key_cb.
            if pending["reset"]:
                pending["reset"] = False
                pm.reset_pose()
                _log_hud()
            if pending["demo"]:
                pending["demo"] = False
                pm.set_demo_pose()
                _log_hud()
            if pending["lateral"]:
                pending["lateral"] = False
                pm.state.f_ext[:] = 0.0
                pm.state.f_ext[1] = 10.0  # +y TCP -> phantom sensor moment
                _log_hud()
            if pending["log"]:
                pending["log"] = False
                _log_hud()

            pm.step()
            viewer.user_scn.ngeom = 0
            try:
                _draw_arrows(viewer)
            except Exception:  # arrows are cosmetic; never let them kill the loop.
                pass
            viewer.sync()
            sleep = dt - (time.time() - t0)
            if sleep > 0:
                time.sleep(sleep)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scene", type=Path, default=_DEFAULT_SCENE)
    ap.add_argument("--gravity", action="store_true", help="enable gravity (default OFF)")
    ap.add_argument("--hand", choices=HANDS, default="right")
    ap.add_argument("--demo-pose", action="store_true", help="start in the demo pose")
    args = ap.parse_args()

    cfg = PlaygroundConfig(scene_path=args.scene, gravity=args.gravity, active_hand=args.hand)
    model = build_model(cfg)
    pm = PlaygroundModel(model, cfg)
    if args.demo_pose:
        pm.set_demo_pose()
    logger.info(
        "model: nq=%d nv=%d nu=%d nsensor=%d neq=%d  (gravity=%s, hand=%s)",
        model.nq,
        model.nv,
        model.nu,
        model.nsensor,
        model.neq,
        cfg.gravity,
        cfg.active_hand,
    )
    _run_viewer(pm)


if __name__ == "__main__":
    main()
