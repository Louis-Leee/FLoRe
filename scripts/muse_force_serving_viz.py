#!/usr/bin/env python3
"""High-level PLANNER control + visualization page (viser) — the S1 control panel.

Front-end half of the online S0↔S1 force deploy. It SPAWNS the planner backend
(``scripts/muse_force_serving_server.py``, in the ``uvaforce`` env) and talks to it
over the SAME base64-npz line protocol the rollout viz uses. Unlike the OFFLINE
``muse_force_rollout_viz.py`` (which scrubs a recorded episode and draws a blue
"recorded" trajectory + an orange rollout on a click), THIS page is LIVE:

  * there is NO blue recorded line — the robot is running in real time;
  * the ORANGE rollout line is the model's most-recent executed LEFT-hand chunk,
    REDRAWN every time a new chunk streams from the backend (that is the whole
    point of the server — it keeps producing);
  * force (predicted), gripper, and the G1 pose are all visualized live;
  * a right-sidebar status shows the low-level controller GREEN (connected +
    running) or RED (offline) — driven by the backend's ``status`` messages
    (S0Driver.num_clients: the low-level's ViveLiveSource connects to :8770);
  * buttons: **▶ Run 1 chunk** (one plan → drip-fed to the low-level), **⏩ Run
    continuously** (keep planning + executing), **⏹ Stop**, and **↺ Reset start
    pose** (re-latch the episode-start pose the model conditions on).

The G1 is rendered by driving the SAME mink ``ThreePointIK`` the rollout viz uses,
but with ``active_roles=("left",)`` — the model predicts only the LEFT hand, so the
left arm follows the chunk and everything else holds the stand pose ("IK for look").
The backend hands us the LEFT UMI-TCP absolute pose per step; we recover the tracker
pose (inverse 0.21 m extrinsic, exactly like the rollout viz) to drive the IK.

Runs in FLoRe's ``.venv`` (viser + mink + mujoco). Reads only committed files +
the checkpoint; writes nothing; shares NOTHING with the flow-matching rollout_viz
or the offline muse rollout viz (helpers are duplicated on purpose). The launcher
``run_muse_force_serving.sh`` wires the two Pythons + PYTHONPATH.
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_FLORE_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
_UMI = os.path.join(_FLORE_ROOT, "third_party", "UMI_collection")
for _p in (
    os.path.join(_UMI, "third_party", "humi_ik"),
    os.path.join(_UMI, "viz"),
    os.path.join(_UMI, "replay"),
):
    if _p not in sys.path:
        sys.path.insert(0, _p)
if _FLORE_ROOT not in sys.path:
    sys.path.insert(0, _FLORE_ROOT)

import viser  # noqa: E402
from viser import transforms as vtf  # noqa: E402
from viser.extras import ViserUrdf  # noqa: E402

import replay_common as rc  # noqa: E402

from flore.utils.rot6d import (  # noqa: E402
    rot6d_to_rotmat, rotmat_to_quat_xyzw, rotmat_to_rot6d,
)
from flore.s1_policy.data.transforms import gripper_width_to_norm  # noqa: E402

_DEFAULT_YAM = os.path.join(_UMI, "robot_assets", "g1_yam", "g1_29dof_yam.urdf")
_JAW_OPEN_M = 0.0475
_GRASP_THRESH_M = 0.08
_ROLLOUT_ORANGE = (245, 150, 40)
_PRED_FORCE_GREEN = "#31c48d"
_MSG_PREFIX = "MUSEVIZ"

# LEFT-hand TCP extrinsic (pure translation, matches tcp_extrinsic.json 0.21 m +x).
_T_TRACKER_TCP = np.eye(4, dtype=np.float64)
_T_TRACKER_TCP[:3, 3] = [0.21, 0.0, 0.0]
_INV_TCP_EXT = np.linalg.inv(_T_TRACKER_TCP)


# ===========================================================================
# Backend (uvaforce env) subprocess — the planner server, over a pipe.
# ===========================================================================
class ServingBackend:
    """Spawn ``muse_force_serving_server.py`` in the uvaforce python; read its
    npz-line messages on a reader thread; send it JSON commands on stdin.

    Messages the backend emits (tag → payload):
      * ``selfcheck``: ok, report(json)
      * ``ready``: info(json)
      * ``status``: lowlevel_online(0/1), running(0/1), ft_force_n, ft_torque_n,
        ft_age_s, staged(0/1)   [~2 Hz heartbeat — carries the LIVE F/T readout]
      * ``chunk``: tcp_pos(S,3), tcp_rot6d(S,6), force_pred(H,6), grip_m(S,),
        preview(0/1), lowlevel_online
      * ``reset_done`` / ``tare_done`` / ``run_staged_done`` / ``error``
    """

    def __init__(self, backend_python: str, checkpoint: str, device: str,
                 model_size: str, force_fusion: str | None, hand: str,
                 exercise_gripper: bool, extra_args: list[str]):
        server = os.path.join(_HERE, "muse_force_serving_server.py")
        cmd = [backend_python, "-u", server, "--checkpoint", checkpoint,
               "--device", device, "--model-size", model_size, "--hand", hand]
        if force_fusion:
            cmd += ["--force-fusion", force_fusion]
        if exercise_gripper:
            cmd += ["--exercise-gripper"]
        cmd += list(extra_args)
        print("[muse-serve-viz] launching backend:", " ".join(cmd), flush=True)
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=None, text=False, bufsize=0)
        self._send_lock = threading.Lock()
        # Latest-of-each-tag store + callbacks.
        self._lock = threading.Lock()
        self.info: dict | None = None
        self.selfcheck: dict | None = None
        self.died = False
        self.ready_evt = threading.Event()
        self.status = {"lowlevel_online": False, "running": False,
                       "ft_force_n": -1.0, "ft_torque_n": -1.0, "ft_age_s": -1.0,
                       "staged": False}
        self.on_chunk = None    # callable(dict) set by main
        self.on_status = None   # callable(dict) set by main
        self.on_tare = None     # callable(ok: bool) set by main — tare_done ack
        self.on_run_staged = None  # callable(ok: bool) set by main — run_staged_done ack
        self.on_error = None    # callable(msg: str) set by main — backend 'error' backstop
        self._stop = False
        self._reader = threading.Thread(target=self._read_loop, daemon=True, name="be-reader")
        self._reader.start()

    def _read_loop(self) -> None:
        stdout = self.proc.stdout
        assert stdout is not None
        while not self._stop:
            raw = stdout.readline()
            if not raw:
                code = self.proc.poll()
                print(f"[muse-serve-viz] backend exited (code {code}); see its stderr above.", flush=True)
                with self._lock:
                    self.status["lowlevel_online"] = False
                    self.status["running"] = False
                    self.status["staged"] = False   # backend gone: nothing can execute
                self.died = True
                # Unblock wait_ready() immediately so a backend that dies during
                # self-check / model load (bad ckpt, missing bridge/VAE) surfaces in
                # seconds instead of hanging the page for the full handshake timeout.
                self.ready_evt.set()
                if self.on_status:
                    try:
                        self.on_status(dict(self.status))
                    except Exception:  # noqa: BLE001
                        pass
                return
            line = raw.decode("ascii", "replace").rstrip("\n")
            if not line.startswith(_MSG_PREFIX):
                continue
            try:
                tag, b64 = line[len(_MSG_PREFIX):].split(" ", 1)
                data = dict(np.load(io.BytesIO(base64.b64decode(b64)), allow_pickle=False))
            except Exception:  # noqa: BLE001
                continue
            self._dispatch(tag, data)

    def _dispatch(self, tag: str, data: dict) -> None:
        if tag == "ready":
            self.info = json.loads(str(data["info"]))
            self.ready_evt.set()
        elif tag == "selfcheck":
            self.selfcheck = {"ok": int(data.get("ok", 0)), "report": json.loads(str(data.get("report", "{}")))}
        elif tag == "status":
            with self._lock:
                self.status["lowlevel_online"] = bool(int(data.get("lowlevel_online", 0)))
                self.status["running"] = bool(int(data.get("running", 0)))
                self.status["ft_force_n"] = float(data.get("ft_force_n", -1.0))
                self.status["ft_torque_n"] = float(data.get("ft_torque_n", -1.0))
                self.status["ft_age_s"] = float(data.get("ft_age_s", -1.0))
                self.status["staged"] = bool(int(data.get("staged", 0)))
            if self.on_status:
                try:
                    self.on_status(dict(self.status))
                except Exception:  # noqa: BLE001
                    pass
        elif tag == "chunk":
            if self.on_chunk:
                try:
                    self.on_chunk(data)
                except Exception as e:  # noqa: BLE001
                    print(f"[muse-serve-viz] on_chunk error: {e!r}", flush=True)
        elif tag == "tare_done":
            if self.on_tare:
                try:
                    self.on_tare(bool(int(data.get("ok", 0))))
                except Exception:  # noqa: BLE001
                    pass
        elif tag == "run_staged_done":
            if self.on_run_staged:
                try:
                    self.on_run_staged(bool(int(data.get("ok", 0))))
                except Exception:  # noqa: BLE001
                    pass
        elif tag == "error":
            print(f"[muse-serve-viz] backend error: {str(data.get('msg'))}", flush=True)
            if self.on_error:
                try:
                    self.on_error(str(data.get("msg")))
                except Exception:  # noqa: BLE001
                    pass

    def send(self, cmd: str) -> bool:
        """Write one {"cmd":…} line to the backend stdin. Returns True iff the frame
        was actually written (False if the backend already exited or the write failed),
        so callers can gate optimistic UI text on the command really being delivered."""
        if self.proc.poll() is not None:
            return False
        with self._send_lock:
            try:
                self.proc.stdin.write((json.dumps({"cmd": cmd}) + "\n").encode("ascii"))
                self.proc.stdin.flush()
                return True
            except Exception:  # noqa: BLE001
                return False

    def wait_ready(self, timeout: float = 900.0) -> bool:
        return self.ready_evt.wait(timeout)

    def close(self) -> None:
        self._stop = True
        try:
            if self.proc.poll() is None:
                self.send("quit")
                self.proc.wait(timeout=5)
        except Exception:  # noqa: BLE001
            pass
        finally:
            if self.proc.poll() is None:
                self.proc.kill()


# ===========================================================================
# IK helpers (LEFT-only; duplicated from the rollout viz to stay independent).
# ===========================================================================
def _rot6d_to_wxyz(rot6d: np.ndarray) -> np.ndarray:
    q_xyzw = rotmat_to_quat_xyzw(rot6d_to_rotmat(np.asarray(rot6d, dtype=np.float64)))
    return np.asarray([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]], dtype=np.float64)


def _tcp_to_tracker(pos: np.ndarray, rot6d: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Recover the TRACKER pose from a stored UMI-TCP pose (LEFT hand). The IK is
    home-anchored and must be driven from the tracker frame, exactly as the rollout
    viz does: world_T_tracker = world_T_tcp @ inv(T_tracker_tcp). Pure translation,
    so orientation is unchanged (only the 0.21 m lever is removed)."""
    pos = np.asarray(pos, dtype=np.float64)
    rot6d = np.asarray(rot6d, dtype=np.float64)
    K = pos.shape[0]
    world_T_tcp = np.broadcast_to(np.eye(4), (K, 4, 4)).copy()
    world_T_tcp[:, :3, :3] = rot6d_to_rotmat(rot6d)
    world_T_tcp[:, :3, 3] = pos
    world_T_tracker = world_T_tcp @ _INV_TCP_EXT[None]
    return world_T_tracker[:, :3, 3], rotmat_to_rot6d(world_T_tracker[:, :3, :3])


def _make_ik(config_name, reach_scale, yam_urdf):
    """Build a LEFT-only ThreePointIK + yam mapper."""
    from g1_right_hand_ik import YamConfigMapper, yam_actuated_joint_names
    from g1_three_point_ik import ThreePointIK

    ik = ThreePointIK(config_name=config_name, reach_scale=reach_scale, active_roles=("left",))
    mapper = YamConfigMapper(ik._humi_joint_names, yam_actuated_joint_names(yam_urdf))
    return ik, mapper


def _make_force_chart(gui, title, labels):
    from viser import uplot
    ac = rc.AXIS_COLORS
    x0 = np.arange(16, dtype=np.float64)
    zero = np.zeros(16)
    return gui.add_uplot(
        data=(x0, zero, zero, zero),
        series=(uplot.Series(label="step"),
                uplot.Series(label=labels[0], stroke=ac[0], width=1.5),
                uplot.Series(label=labels[1], stroke=ac[1], width=1.5),
                uplot.Series(label=labels[2], stroke=ac[2], width=1.5)),
        scales={"x": uplot.Scale(time=False)},
        title=title, aspect=2.2, height=120,
    )


def _set_force_chart(chart, xyz: np.ndarray) -> None:
    n = xyz.shape[0]
    if n < 2:
        return
    t = np.arange(n, dtype=np.float64)
    chart.data = (t, xyz[:, 0], xyz[:, 1], xyz[:, 2])


# ===========================================================================
def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--backend-python", required=True, help="the uvaforce interpreter")
    p.add_argument("--checkpoint", required=True, help="UVA force policy checkpoint")
    p.add_argument("--device", default="cuda")
    p.add_argument("--model-size", default="mar_base")
    p.add_argument("--force-fusion", default=None, choices=[None, "early", "late", "both"])
    p.add_argument("--hand", default="left", choices=["left"])  # LEFT-canonical (see server)
    p.add_argument("--exercise-gripper", action="store_true")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8087)
    p.add_argument("--yam-urdf", default=_DEFAULT_YAM)
    p.add_argument("--ik-config", default="proposal")
    p.add_argument("--reach-scale", type=float, default=1.0)
    p.add_argument("--cam-offset", type=float, nargs=3, default=(0.10, 0.0, 0.04))
    p.add_argument("--cam-rpy", type=float, nargs=3, default=(4.7124, 1.5708, 0.0))
    # Any trailing args pass through to the backend (e.g. --force-ws, --no-selfcheck).
    args, extra = p.parse_known_args(argv if argv is not None else sys.argv[1:])

    backend = ServingBackend(args.backend_python, args.checkpoint, args.device,
                             args.model_size, args.force_fusion, args.hand,
                             args.exercise_gripper, extra)
    import atexit
    atexit.register(backend.close)
    print("[muse-serve-viz] waiting for backend self-check + model load…", flush=True)
    if not backend.wait_ready(timeout=900.0):
        raise SystemExit("[muse-serve-viz] backend never became ready in 900s (see its stderr).")
    if backend.died or backend.info is None:
        # _read_loop sets ready_evt on backend EOF too, so distinguish a real 'ready'
        # from a died-during-startup (bad ckpt, missing bridge/VAE, self-check fail).
        raise SystemExit("[muse-serve-viz] backend exited during startup/self-check "
                         "(see its stderr above — likely a failed self-check, missing "
                         "checkpoint/VAE, or a bridge not running).")
    info = backend.info or {}
    print(f"[muse-serve-viz] backend ready: {info}", flush=True)

    # --- IK (LEFT only) ---
    ik, mapper = _make_ik(args.ik_config, args.reach_scale, args.yam_urdf)

    # --- scene ---
    server = viser.ViserServer(host=args.host, port=args.port, label="FLoRe MuSe-force PLANNER")
    server.scene.set_up_direction("+z")
    server.scene.add_grid("/grid", plane="xy", shadow_opacity=0.0, infinite_grid=True, cell_size=0.25)
    base_xyz = np.asarray(ik.home_q()[0:3], dtype=float).copy()
    server.scene.add_frame("/robot", show_axes=False, position=tuple(base_xyz))
    urdf = ViserUrdf(server, urdf_or_path=Path(args.yam_urdf), root_node_name="/robot")

    # Drive the URDF to the STAND (home) pose immediately — otherwise ViserUrdf sits at
    # its all-zeros default config (arms/legs collapsed into a garbled mesh) until the
    # first chunk arrives. The IK's home_q() is the stand keyframe; map it to the yam
    # actuated vector (gripper open) so the humanoid renders correctly from t=0. The
    # render loop keeps showing this until a prediction/execution supplies a chunk.
    home_cfg = mapper.to_yam(ik.home_q(), left_jaw_m=_JAW_OPEN_M)
    urdf.update_cfg(home_cfg)
    # Home follow-frame pose (left wrist at the stand config) so the camera + wrench
    # arrow sit on the arm before any chunk. Read straight off the solver's home config.
    _home_conf = ik._solver.configuration
    _home_wrist = _home_conf.get_transform_frame_to_world(
        frame_name="left_wrist_yaw_link", frame_type="body")
    home_wrist_pos = np.asarray(_home_wrist.translation(), dtype=float)
    home_wrist_wxyz = np.asarray(_home_wrist.rotation().wxyz, dtype=float)

    follow = server.scene.add_frame("/follow", show_axes=True, axes_length=0.08,
                                    axes_radius=0.004, origin_radius=0.01,
                                    position=tuple(home_wrist_pos), wxyz=tuple(home_wrist_wxyz))
    rpy = np.asarray(args.cam_rpy, float)
    cam_rot = (vtf.SO3.from_rpy_radians(float(rpy[0]), 0.0, 0.0)
               .multiply(vtf.SO3.from_rpy_radians(0.0, float(rpy[1]), 0.0)))
    server.scene.add_frame("/follow/cam", show_axes=False,
                           position=tuple(np.asarray(args.cam_offset, float)),
                           wxyz=tuple(cam_rot.wxyz))
    wrench = rc.WrenchArrows(server, name="/follow/wrench")

    # --- GUI ---
    gui = server.gui
    gui.configure_theme(control_layout="collapsible", control_width="large")

    with gui.add_folder("Low-level controller"):
        status_md = gui.add_markdown("🔴 **OFFLINE** — waiting for the low-level to connect on :8770")
    with gui.add_folder("Model (UVA force)"):
        gui.add_text("predict_force", initial_value=str(info.get("predict_force")), disabled=True)
        gui.add_text("pose repr", initial_value=str(info.get("action_pose_repr")), disabled=True)
        gui.add_text("H / n_action_steps",
                     initial_value=f"{info.get('obs_history_len')} / {info.get('n_action_steps')}", disabled=True)
        sc = backend.selfcheck or {}
        gui.add_text("self-check", initial_value=("OK" if sc.get("ok") else str(sc.get("report", "?"))[:60]),
                     disabled=True)
    with gui.add_folder("F/T sensor"):
        tare_btn = gui.add_button("⊘ Tare F/T (in free space)")
        tare_status = gui.add_text("tare", initial_value="startup tare", disabled=True)
        # LIVE sensor readout (from the status heartbeat) — confirms the F/T is actually
        # streaming + roughly zeroed BEFORE predicting. Distinct from the PREDICTED-force
        # chart below (a model output that is flat until a chunk runs).
        ft_live = gui.add_text("live |F| / |T|", initial_value="(waiting…)", disabled=True)
    with gui.add_folder("Gripper"):
        # Operator kill-switch for the PHYSICAL LEFT jaw, on top of the backend's
        # low-level-online gate. Seeded from the launch flag (COMMAND_GRIPPER); untick
        # to stop the model from actuating the jaw WITHOUT stopping the arm. When the
        # backend was launched read-only (--command-gripper off) the checkbox is DISABLED
        # — the backend also refuses grip_on in that case, so runtime-enabling a jaw the
        # launch didn't intend (whose shutdown-release path would be skipped) is impossible.
        grip_cmd_on = bool(info.get("command_gripper"))
        grip_toggle = gui.add_checkbox("Command gripper (autonomous)",
                                       initial_value=grip_cmd_on, disabled=not grip_cmd_on)
    with gui.add_folder("Control"):
        # SAFE two-step one-shot: PREDICT (inference + viz ONLY — the robot does NOT move
        # and the model frame is untouched, so you can re-predict freely), then RUN THIS
        # CHUNK to actually execute the previewed plan. "Run this chunk" starts DISABLED
        # and only enables once a prediction is staged (status heartbeat `staged`).
        predict_btn = gui.add_button("🔍 Predict 1 chunk (no motion)")
        run_staged_btn = gui.add_button("▶ Run this chunk", disabled=True)
        run_cont_btn = gui.add_button("⏩ Run continuously")
        stop_btn = gui.add_button("⏹ Stop")
        reset_btn = gui.add_button("↺ Reset start pose")
        ctrl_status = gui.add_text("Status", initial_value="idle", disabled=True)
    with gui.add_folder("Readout"):
        rd_grip = gui.add_text("L gripper", initial_value="-", disabled=True)
        rd_force = gui.add_text("|Fpred[0]| (N)", initial_value="-", disabled=True)
        rd_chunks = gui.add_text("chunks run", initial_value="0", disabled=True)
    with gui.add_folder("View"):
        show_rollout = gui.add_checkbox("Show rollout (orange) path", initial_value=True)
        show_wrench = gui.add_checkbox("Show predicted-force arrow", initial_value=True)
    with gui.add_folder("Charts"):
        grip_chart = rc.make_single_chart(gui, "L gripper width (m)", "width",
                                          color_idx=1, y_range=(0.0, max(_JAW_OPEN_M * 2.2, 0.1)))
        force_chart = _make_force_chart(gui, "Predicted force (N)", ("Fx", "Fy", "Fz"))
        torque_chart = _make_force_chart(gui, "Predicted torque (Nm)", ("Tx", "Ty", "Tz"))

    # --- shared render state (mutated on the backend reader thread; the render loop
    #     picks it up under the lock — GUI callbacks only send commands) ---
    render_lock = threading.Lock()
    latest = {
        "cfgs": None,        # (S, n_yam) URDF configs for the executed chunk
        "wrist_pos": None,   # (S,3) follow-frame position
        "wrist_wxyz": None,  # (S,4)
        "path": None,        # (S,3) orange polyline (left site path)
        "force_pred": None,  # (H,6)
        "grip_m": None,      # (S,)
        "dirty": False,
        "is_preview": False, # last chunk was a Predict preview (no motion), not executed
        "n_chunks": 0,
    }
    roll_handle = {"h": None}
    # The IK is calibrated ONCE (on the first chunk) to a FIXED home reference and then
    # frozen — NOT re-anchored per chunk. The backend advances its anchor across chunks
    # (chunk N+1 starts where N ended, muse_force_serving_server.infer_chunk), so a
    # per-chunk recalibrate would pin every chunk's frame-0 back to home and DISCARD the
    # cumulative motion — the arm would appear to wiggle near home while the real arm
    # drifts across the workspace. Calibrating once (like the offline rollout viz's single
    # calib_ref) keeps absolute pelvis-frame positions meaningful across all chunks.
    calib = {"done": False}

    # NOTE: the mink IK Configuration is mutated ONLY on THIS backend-reader thread
    # (inside _on_chunk). The render loop (main thread) NEVER calls ik.* — it only reads
    # the cached cfgs/wrist/path under render_lock. `ik.home_q()` at scene setup runs
    # before on_chunk is wired, so it cannot overlap. This is the single-owner-thread
    # discipline the rig requires (a different single thread than the offline tool, but
    # still single). Do NOT add an ik.* call to the render loop without a lock.
    def _on_chunk(data: dict) -> None:
        """Backend thread: solve the LEFT-arm IK for the new chunk, cache render data.

        The chunk may be a PREVIEW (``preview=1``, from Predict — no real motion) or an
        EXECUTED chunk (``preview=0``, from Run/continuous). Both are visualized the same
        way (rollout + predicted force + jaw), so the operator sees the identical rollout
        they previewed once they execute it; only the "chunks run" counter distinguishes
        them (previews don't count — nothing moved)."""
        tcp_pos = data["tcp_pos"].astype(np.float64)      # (S,3)
        tcp_rot6d = data["tcp_rot6d"].astype(np.float64)  # (S,6)
        force_pred = data.get("force_pred")
        grip_m = data.get("grip_m")
        is_preview = bool(int(np.asarray(data.get("preview", 0)).reshape(-1)[0])) \
            if data.get("preview") is not None else False
        S = tcp_pos.shape[0]
        # TCP -> tracker (LEFT), then presolve the LEFT-only IK per step.
        tr_pos, tr_rot6d = _tcp_to_tracker(tcp_pos, tcp_rot6d)
        n_yam = len(mapper._yam_names)
        cfgs = np.zeros((S, n_yam))
        wrist_pos = np.zeros((S, 3))
        wrist_wxyz = np.tile([1.0, 0.0, 0.0, 0.0], (S, 1)).astype(float)
        path = np.zeros((S, 3))
        # Calibrate ONCE to the FIRST chunk's frame-0 (the ready-pose neighborhood); keep
        # that anchor for every subsequent chunk so cross-chunk motion renders faithfully.
        if not calib["done"]:
            ik.recalibrate({"left": (tr_pos[0], _rot6d_to_wxyz(tr_rot6d[0]))})
            calib["done"] = True
        prim_wrist_link = "left_wrist_yaw_link"
        for i in range(S):
            q = ik.step({"left": (tr_pos[i], _rot6d_to_wxyz(tr_rot6d[i]))})
            # grip_m[i] is PHYSICAL width (m); normalize by the width reference (0.095 m)
            # to [0,1], THEN scale by the per-finger URDF stroke (_JAW_OPEN_M=0.0475).
            # (Dividing width by _JAW_OPEN_M directly would render the jaw ~2x too open.)
            jaw = float(gripper_width_to_norm(grip_m[i]) * _JAW_OPEN_M) if grip_m is not None else 0.0
            cfgs[i] = mapper.to_yam(q, left_jaw_m=jaw)
            conf = ik._solver.configuration
            wrist = conf.get_transform_frame_to_world(frame_name=prim_wrist_link, frame_type="body")
            wrist_pos[i] = wrist.translation()
            wrist_wxyz[i] = wrist.rotation().wxyz
            path[i] = conf.get_transform_frame_to_world(
                frame_name="left_hand_site", frame_type="site").translation()
        with render_lock:
            # Previews render the rollout but do NOT increment the executed-chunk counter
            # (nothing physically ran); executed chunks do.
            latest.update(cfgs=cfgs, wrist_pos=wrist_pos, wrist_wxyz=wrist_wxyz, path=path,
                          force_pred=(np.asarray(force_pred) if force_pred is not None and np.asarray(force_pred).size else None),
                          grip_m=(np.asarray(grip_m) if grip_m is not None else None),
                          dirty=True, is_preview=is_preview,
                          n_chunks=latest["n_chunks"] + (0 if is_preview else 1))

    # A run_staged click is IN FLIGHT: sent to the backend but not yet consumed+acked.
    # The ~2 Hz status heartbeat carries a STALE staged snapshot (the backend's stdin
    # thread hasn't cleared staged["out"] yet), so without this the heartbeat would
    # RE-ENABLE the button between click and consume — defeating the double-click guard.
    # Set on click, cleared by the run_staged ack (_on_run_staged). Plain bool (GIL-atomic
    # single-word read/write across the click callback thread and the reader thread).
    run_staged_pending = {"v": False}

    def _on_status(st: dict) -> None:
        # Runs on the backend reader thread (same thread + GUI-write pattern as the
        # rollout viz's status handler — viser GUI value writes are the mutation).
        online = bool(st.get("lowlevel_online"))
        running = bool(st.get("running"))
        if online:
            status_md.content = ("🟢 **ONLINE** — low-level connected on :8770"
                                 + (" · running" if running else ""))
        else:
            status_md.content = "🔴 **OFFLINE** — waiting for the low-level to connect on :8770"
        # Live F/T readout: -1 magnitude = no frame yet (bridge down / just starting).
        f_n = float(st.get("ft_force_n", -1.0))
        t_n = float(st.get("ft_torque_n", -1.0))
        age = float(st.get("ft_age_s", -1.0))
        if not (np.isfinite(f_n) and np.isfinite(t_n)) or f_n < 0.0 or age < 0.0:
            # -1 sentinel = no frame; non-finite = a corrupt/NaN wrench (screen it rather
            # than rendering a literal 'nan N', review LOW).
            ft_live.value = "no/invalid F/T frame (bridge down?)"
        elif age > 1.0:
            ft_live.value = f"STALE {age:.1f}s — |F|={f_n:.2f} N |T|={t_n:.3f} Nm"
        else:
            ft_live.value = f"|F|={f_n:.2f} N   |T|={t_n:.3f} Nm"
        # Enable "Run this chunk" only while a prediction is staged AND not mid-run AND no
        # run_staged click is in flight (the stale-heartbeat re-enable guard above).
        run_staged_btn.disabled = not (bool(st.get("staged")) and not running
                                       and not run_staged_pending["v"])

    backend.on_chunk = _on_chunk
    backend.on_status = _on_status

    @predict_btn.on_click
    def _(_e) -> None:
        # SAFE preview: inference + viz only. The robot does NOT move and the model frame
        # is untouched, so re-predicting just refreshes the preview. "Run this chunk"
        # enables once the backend acks the stage (via the status heartbeat's `staged`).
        # REFUSE locally while continuous is running: the backend drops predict_once then
        # (guarded), so without this the status text would lie ("predicting…" forever).
        if backend.status.get("running"):
            ctrl_status.value = "ignored — Stop continuous first"
        elif backend.send("predict_once"):
            ctrl_status.value = "predicting (preview — no motion)…"
        else:
            ctrl_status.value = "NOT sent (backend offline)"

    @run_staged_btn.on_click
    def _(_e) -> None:
        # Execute the previewed chunk. Disable immediately AND latch run_staged_pending so
        # a stale status heartbeat can't re-enable it before the backend consumes the
        # stage (review finding). The backend consumes-before-executing and acks via
        # _on_run_staged, which clears the latch; the button re-arms only on a fresh
        # Predict. This is the ONLY one-shot button that moves the robot, and only after
        # Predict.
        run_staged_pending["v"] = True
        run_staged_btn.disabled = True
        if backend.send("run_staged"):
            ctrl_status.value = "running staged chunk…"
        else:
            run_staged_pending["v"] = False   # nothing sent; let the heartbeat govern again
            ctrl_status.value = "NOT sent (backend offline)"

    @run_cont_btn.on_click
    def _(_e) -> None:
        backend.send("run_continuous")
        ctrl_status.value = "running continuously"

    @stop_btn.on_click
    def _(_e) -> None:
        backend.send("stop")
        ctrl_status.value = "stopped"

    @reset_btn.on_click
    def _(_e) -> None:
        backend.send("reset")
        ctrl_status.value = "reset start pose"

    def _on_run_staged(ok: bool) -> None:
        # Ack from the backend after a run_staged: ok=1 executed; ok=0 refused (nothing
        # staged). Either way the stage is now CONSUMED, so clear the in-flight latch — else
        # the enable gate stays permanently False and "Run this chunk" never re-enables after
        # the first run (review HIGH: the latch was set on click but never cleared here). The
        # button then re-arms only when the next Predict sets staged=1 via the heartbeat.
        run_staged_pending["v"] = False
        ctrl_status.value = "staged chunk done" if ok else "nothing staged — Predict first"
    backend.on_run_staged = _on_run_staged

    def _on_error(_msg: str) -> None:
        # Backstop: a backend 'error' during a staged run (e.g. low-level disconnects
        # mid-drip) must also clear the in-flight latch so "Run this chunk" isn't stuck
        # disabled. The backend also acks run_staged_done ok=0 on that path; this is
        # defense-in-depth in case an error arrives without the ack.
        run_staged_pending["v"] = False
    backend.on_error = _on_error

    def _on_tare(ok: bool) -> None:
        # Definitive ack from the backend: ok=1 → the {"cmd":"tare"} was forwarded to the
        # F/T bridge; ok=0 → REFUSED (a chunk was running/dripping — the arm is loaded, so
        # taring would bake in a contact-force bias). Reflects what actually happened, not
        # an optimistic client guess.
        tare_status.value = "tared (free space)" if ok else "REFUSED — stop + free space first"
    backend.on_tare = _on_tare

    @tare_btn.on_click
    def _(_e) -> None:
        # Re-zero the F/T bridge NOW — do this with the arm raised into FREE SPACE
        # (gripper touching nothing), after the low-level has ramped to the ready pose
        # and BEFORE arming. A contact force present at tare biases every later reading.
        # The backend REFUSES this while a chunk is running/dripping (see server tare guard).
        if not backend.send("tare"):
            tare_status.value = "NOT sent (backend offline)"
        else:
            tare_status.value = "tare requested…"   # replaced by the tare_done ack (_on_tare)

    @grip_toggle.on_update
    def _(_e) -> None:
        # Live enable/disable of autonomous jaw actuation (operator kill-switch). Does
        # NOT stop the arm — the pose wire keeps flowing; only the gripper command stops.
        backend.send("grip_on" if grip_toggle.value else "grip_off")

    print(f"[muse-serve-viz] http://{args.host}:{args.port}  — connect the low-level to :8770, "
          f"then Predict 1 chunk → Run this chunk (or Run continuously)", flush=True)

    # --- render loop (main thread): apply the newest chunk + animate through it ---
    play_i = {"i": 0, "S": 0}
    while True:
        with render_lock:
            dirty = latest["dirty"]
            cfgs = latest["cfgs"]
            wpos = latest["wrist_pos"]
            wwxyz = latest["wrist_wxyz"]
            path = latest["path"]
            force_pred = latest["force_pred"]
            grip_m = latest["grip_m"]
            n_chunks = latest["n_chunks"]
            if dirty:
                latest["dirty"] = False
        if dirty and cfgs is not None:
            play_i["i"] = 0
            play_i["S"] = cfgs.shape[0]
            # orange path — always drop the PREVIOUS chunk's line first, so a short
            # (<2-step) chunk cannot leave the prior chunk's line on screen mismatched
            # with the new animation/charts.
            if roll_handle["h"] is not None:
                roll_handle["h"].remove()
                roll_handle["h"] = None
            if path is not None and path.shape[0] >= 2:
                seg = np.stack([path[:-1], path[1:]], axis=1).astype(np.float32)
                col = np.broadcast_to(np.array(_ROLLOUT_ORANGE, np.uint8), (seg.shape[0], 2, 3)).copy()
                roll_handle["h"] = server.scene.add_line_segments(
                    "/rollout/left", points=seg, colors=col, line_width=3.0)
                roll_handle["h"].visible = bool(show_rollout.value)
            # charts (per-step index axis)
            if grip_m is not None and grip_m.shape[0] >= 2:
                rc.set_full_single(grip_chart, np.arange(grip_m.shape[0], dtype=np.float64), grip_m)
            if force_pred is not None and force_pred.shape[0] >= 2:
                _set_force_chart(force_chart, force_pred[:, 0:3])
                _set_force_chart(torque_chart, force_pred[:, 3:6])
            rd_chunks.value = str(n_chunks)
        # animate through the cached chunk (visual only)
        if cfgs is not None and play_i["S"] > 0:
            i = min(play_i["i"], play_i["S"] - 1)
            urdf.update_cfg(cfgs[i])
            follow.position = tuple(wpos[i])
            follow.wxyz = tuple(wwxyz[i])
            if grip_m is not None:
                gw = float(grip_m[min(i, grip_m.shape[0] - 1)])
                rd_grip.value = f"{gw * 1000:.1f} mm ({'grasp' if gw < _GRASP_THRESH_M else 'open'})"
            wrench.set_visible(bool(show_wrench.value) and force_pred is not None)
            if force_pred is not None and show_wrench.value:
                j = min(i, force_pred.shape[0] - 1)
                wrench.update(wpos[i], wwxyz[i], force_pred[j, 0:3], force_pred[j, 3:6])
                rd_force.value = f"{np.linalg.norm(force_pred[0, 0:3]):.2f}"
            play_i["i"] = i + 1
        time.sleep(1.0 / 30.0)


if __name__ == "__main__":
    raise SystemExit(main())
