"""Interface registration for all deployment modes.

Handles interface setup for:
- Headless sim2sim: MuJoCoInterface (replaces hardware interface)
- FK odometry: FKOdometryWrapper (headless or sim2real)
- Mocap tracking: MocapInterfaceWrapper (sim2real or sim2sim)

Mocap receivers:
- MocapReceiver: ROS2 subscriber for OptiTrack rigid body pose (sim2real)
- SimMocapReceiver: UDP listener for sim bridge root state (sim2sim)
Both feed into MocapInterfaceWrapper which injects position into get_low_state().
"""

from __future__ import annotations

import os
import socket
import struct
import threading
import time

import numpy as np
from loguru import logger

from holosoma_inference.utils.math.quat import quat_apply, quat_inverse, quat_mul

# UDP port for sim mocap (must match holosoma bridge's SIM_MOCAP_UDP_PORT)
_SIM_MOCAP_UDP_PORT = int(os.environ.get("SIM_MOCAP_UDP_PORT", "9870"))


class SimMocapReceiver:
    """UDP receiver for sim2sim root state. Same interface as MocapReceiver.

    In interactive sim2sim, the simulator broadcasts ground-truth root state
    (pos + quat_wxyz + lin_vel) via UDP every step. This receiver listens on
    localhost and provides get_state() compatible with MocapInterfaceWrapper.

    Args:
        port: UDP port to listen on (default: SIM_MOCAP_UDP_PORT env var or 9870).
    """

    def __init__(self, port: int = _SIM_MOCAP_UDP_PORT):
        self._lock = threading.Lock()
        self._position = np.zeros(3, dtype=np.float32)
        self._quat_wxyz = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        self._velocity = np.zeros(3, dtype=np.float32)
        self._has_data = False

        self._port = port
        self._shutdown_event = threading.Event()
        self._thread = threading.Thread(target=self._recv_loop, daemon=True)
        self._thread.start()
        logger.info(f"SimMocapReceiver: listening on UDP 127.0.0.1:{port}")

    def _recv_loop(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind(("127.0.0.1", self._port))
            sock.settimeout(0.1)
            while not self._shutdown_event.is_set():
                try:
                    data, _ = sock.recvfrom(64)
                except socket.timeout:
                    continue
                if len(data) < 40:
                    continue
                vals = struct.unpack("10f", data[:40])
                with self._lock:
                    self._position[:] = vals[0:3]
                    self._quat_wxyz[:] = vals[3:7]
                    self._velocity[:] = vals[7:10]
                    self._has_data = True
        finally:
            sock.close()

    @property
    def has_data(self) -> bool:
        with self._lock:
            return self._has_data

    def get_state(self):
        """Returns (position[3], quat_wxyz[4], velocity[3]) or None if no data."""
        with self._lock:
            if not self._has_data:
                return None
            return (self._position.copy(), self._quat_wxyz.copy(), self._velocity.copy())

    def shutdown(self):
        self._shutdown_event.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)


class MocapReceiver:
    """Thread-safe ROS2 subscriber for OptiTrack rigid body pose.

    Runs a ROS2 node in a background thread, subscribing to
    /<body_name>/pose (geometry_msgs/PoseStamped). Stores the latest
    pose and estimates linear velocity via finite differencing.

    Args:
        body_name: Rigid body name in Motive (becomes the ROS2 topic prefix).
        vel_window: Number of samples for velocity smoothing.
    """

    def __init__(self, body_name: str, vel_window: int = 5):
        self.body_name = body_name
        self._vel_window = vel_window

        self._lock = threading.Lock()
        self._position = np.zeros(3, dtype=np.float64)
        self._quat_wxyz = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        self._timestamp = 0.0
        self._has_data = False
        self._velocity = np.zeros(3, dtype=np.float64)

        # Pre-allocated callback buffers to avoid allocation at ~120Hz
        self._cb_pos = np.zeros(3, dtype=np.float64)
        self._cb_quat = np.zeros(4, dtype=np.float64)

        # Velocity estimation ring buffer
        self._pos_history = np.zeros((vel_window, 3), dtype=np.float64)
        self._time_history = np.zeros(vel_window, dtype=np.float64)
        self._hist_idx = 0

        self._node = None
        self._thread = None
        self._shutdown_event = threading.Event()

        self._start_ros2_thread()

    def _start_ros2_thread(self):
        self._thread = threading.Thread(target=self._ros2_subprocess_bridge, daemon=True)
        self._thread.start()

    def _ros2_subprocess_bridge(self):
        """Subscribe via subprocess to avoid ROS2/Unitree SDK library conflicts."""
        import subprocess
        import struct
        import tempfile

        topic = f"{self.body_name}/pose" if self.body_name.startswith("/") else f"/{self.body_name}/pose"
        script = f"""import sys, struct
import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.qos import QoSProfile, ReliabilityPolicy
rclpy.init()
node = rclpy.create_node("mocap_bridge")
qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
def cb(msg):
    p = msg.pose.position
    q = msg.pose.orientation
    t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
    sys.stdout.buffer.write(struct.pack("!8d", p.x, p.y, p.z, q.x, q.y, q.z, q.w, t))
    sys.stdout.buffer.flush()
node.create_subscription(PoseStamped, "{topic}", cb, qos)
rclpy.spin(node)
"""
        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False)
        tmp.write(script)
        tmp.flush()
        tmp.close()

        env = dict(os.environ)
        if self.body_name.startswith("/"):
            env["RMW_IMPLEMENTATION"] = "rmw_cyclonedds_cpp"
        env.setdefault("ROS_DOMAIN_ID", "0")
        proc = subprocess.Popen(
            ["bash", "-c", f"source /opt/ros/humble/setup.bash && exec python3 {tmp.name}"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
        )
        logger.info(f"MocapReceiver: subprocess bridge for {topic} (PID {proc.pid})")
        buf_size = struct.calcsize("!8d")
        try:
            while not self._shutdown_event.is_set():
                data = proc.stdout.read(buf_size)
                if len(data) < buf_size:
                    proc.kill()
                    try:
                        _, stderr = proc.communicate(timeout=2.0)
                    except Exception:
                        stderr = b""
                    if stderr:
                        logger.error(f"MocapReceiver subprocess stderr: {stderr.decode(errors='replace')}")
                    break
                vals = struct.unpack("!8d", data)
                px, py, pz, qx, qy, qz, qw, t = vals

                cb_pos = self._cb_pos
                cb_pos[0] = px
                cb_pos[1] = py
                cb_pos[2] = pz
                cb_quat = self._cb_quat
                cb_quat[0] = qw
                cb_quat[1] = qx
                cb_quat[2] = qy
                cb_quat[3] = qz
                self._update_state(cb_pos, cb_quat, t)
        finally:
            proc.kill()
            try:
                os.unlink(tmp.name)
            except OSError:
                pass

    def _ros2_spin(self):
        try:
            import rclpy
            from geometry_msgs.msg import PoseStamped
            from rclpy.qos import QoSProfile, ReliabilityPolicy

            if not rclpy.ok():
                rclpy.init()

            self._node = rclpy.create_node(f"mocap_receiver_{self.body_name}")
            topic = f"/{self.body_name}/pose"
            qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
            self._node.create_subscription(PoseStamped, topic, self._pose_callback, qos)
            logger.info(f"MocapReceiver: subscribing to {topic}")

            while not self._shutdown_event.is_set():
                rclpy.spin_once(self._node, timeout_sec=0.01)

            self._node.destroy_node()
        except ImportError:
            logger.error("MocapReceiver requires rclpy. Install ROS2 or run inside a ROS2 environment.")
        except Exception as e:
            logger.error(f"MocapReceiver ROS2 thread error: {e}")

    def _update_state(self, pos, quat_wxyz, t):
        """Thread-safe state update with velocity ring buffer. Called from both
        _pose_callback (in-process ROS2) and _ros2_subprocess_bridge (pipe)."""
        with self._lock:
            self._position[:] = pos
            self._quat_wxyz[:] = quat_wxyz
            self._timestamp = t
            self._has_data = True

            idx = self._hist_idx % self._vel_window
            self._pos_history[idx] = pos
            self._time_history[idx] = t
            self._hist_idx += 1

            count = min(self._hist_idx, self._vel_window)
            if count >= 2:
                oldest_idx = (self._hist_idx - count) % self._vel_window
                newest_idx = (self._hist_idx - 1) % self._vel_window
                dt = self._time_history[newest_idx] - self._time_history[oldest_idx]
                if dt > 1e-6:
                    self._velocity[:] = (self._pos_history[newest_idx] - self._pos_history[oldest_idx]) / dt

    def _pose_callback(self, msg):
        p = msg.pose.position
        q = msg.pose.orientation  # xyzw (ROS2 standard)
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        cb_pos = self._cb_pos
        cb_pos[0] = p.x
        cb_pos[1] = p.y
        cb_pos[2] = p.z
        cb_quat = self._cb_quat
        cb_quat[0] = q.w
        cb_quat[1] = q.x
        cb_quat[2] = q.y
        cb_quat[3] = q.z
        self._update_state(cb_pos, cb_quat, t)

    @property
    def has_data(self) -> bool:
        with self._lock:
            return self._has_data

    def get_state(self) -> tuple[np.ndarray, np.ndarray, np.ndarray | None] | None:
        """Returns (position[3], quat_wxyz[4], velocity[3] or None) or None if no data.

        Single lock acquisition for the 50Hz policy hot path.
        """
        with self._lock:
            if not self._has_data:
                return None
            vel = self._velocity.astype(np.float32) if self._hist_idx >= 2 else None
            return (self._position.astype(np.float32), self._quat_wxyz.astype(np.float32), vel)

    def shutdown(self):
        self._shutdown_event.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)


class MocapInterfaceWrapper:
    """Wraps a robot interface to inject mocap pose into get_low_state().

    Decorates the inner interface's get_low_state() output:
    - base_pos (indices 0:3) ← mocap position
    - base_quat (indices 3:7) ← mocap orientation (if use_mocap_orientation)
    - base_lin_vel (indices 7+N:7+N+3) ← mocap estimated velocity

    When ``head_to_torso=True``, applies the G1 head_link→torso_link offset
    (rotate the body-frame offset by the current mocap orientation) so a head-
    mounted marker can drive the torso-based ``three_point`` tracking mode.

    All other interface methods delegate to the inner interface.
    """

    # G1 URDF: head_joint xyz="0.0039635 0 -0.044" from torso_link to head_link.
    # Inverse (head→torso in head body frame): [-0.0039635, 0, +0.044]
    _HEAD_TO_TORSO_BODY = np.array([-0.004, 0.0, 0.044], dtype=np.float64)

    # Real head marker → torso body-frame offset, calibrated from simultaneous
    # head + pelvis OptiTrack data (FK output vs GT pelvis, 500-frame average).
    _REAL_HEAD_TO_TORSO_BODY = np.array([-0.025, -0.010, -0.526], dtype=np.float64)

    # URDF waist_roll_joint offset (pelvis→torso chain, for dynamic torso→pelvis)
    _WAIST_OFFSET = np.array([-0.0039635, 0.0, 0.044], dtype=np.float64)

    def __init__(
        self,
        inner_interface,
        mocap: MocapReceiver,
        num_joints: int = 29,
        use_mocap_orientation: bool = False,
        head_to_torso: bool = False,
        real_head_to_torso: bool = False,
    ):
        self._inner = inner_interface
        self._mocap = mocap
        self._use_mocap_orientation = use_mocap_orientation
        self._head_offset = None
        self._real_head_mode = False
        self._gt_pelvis_mocap = None
        self._fk_debug_log = []
        if real_head_to_torso:
            self._real_head_mode = True
            self._pelvis_vel_window = 3
            self._pelvis_pos_history = np.zeros((3, 3), dtype=np.float64)
            self._pelvis_time_history = np.zeros(3, dtype=np.float64)
            self._pelvis_hist_idx = 0
            self._fk_calibrated = False
            self._fk_pos_offset = np.zeros(3, dtype=np.float64)
            self._fk_quat_offset = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
            gt_body = os.environ.get("MOCAP_GT_PELVIS", "")
            if gt_body:
                self._gt_pelvis_mocap = MocapReceiver(gt_body)
                logger.info(f"MocapInterfaceWrapper: GT pelvis receiver: {gt_body} — waiting up to 5s...")
                for _ in range(50):
                    if self._gt_pelvis_mocap.has_data:
                        logger.info("GT pelvis mocap ready")
                        break
                    time.sleep(0.1)
                else:
                    logger.warning("GT pelvis mocap not ready after 5s — calibration may miss")
            self._fk_calibrate_trigger = "/tmp/fk_calibrate"
            try:
                os.remove(self._fk_calibrate_trigger)
            except FileNotFoundError:
                pass
            logger.info("MocapInterfaceWrapper: real head→torso→pelvis (full quat + waist FK)")
            logger.info(f">>> To calibrate FK, run: touch {self._fk_calibrate_trigger} <<<")
        elif head_to_torso:
            self._head_offset = self._HEAD_TO_TORSO_BODY
            logger.info("MocapInterfaceWrapper: URDF head→torso offset enabled")
        self._lin_vel_start = 7 + num_joints

    def get_low_state(self) -> np.ndarray:
        state = self._inner.get_low_state()
        mocap_state = self._mocap.get_state()
        if mocap_state is not None:
            pos, quat_wxyz, vel = mocap_state
            if self._real_head_mode:
                waist_joints = state[0, 19:22]
                if self._use_mocap_orientation:
                    orient_quat = quat_wxyz
                    quat_wxyz = self._head_quat_to_pelvis_quat(quat_wxyz, waist_joints)
                else:
                    orient_quat = state[0, 3:7]
                fk_pos = self._apply_real_head_to_pelvis(pos, orient_quat, waist_joints)
                fk_quat = quat_wxyz if self._use_mocap_orientation else orient_quat
                if (
                    not self._fk_calibrated
                    and self._gt_pelvis_mocap is not None
                    and os.path.exists(self._fk_calibrate_trigger)
                ):
                    gt = self._gt_pelvis_mocap.get_state()
                    if gt is not None:
                        self._fk_pos_offset[:] = gt[0].astype(np.float64) - fk_pos.astype(np.float64)
                        gt_quat_f64 = gt[1].astype(np.float64).reshape(1, 4)
                        fk_quat_f64 = fk_quat.astype(np.float64).reshape(1, 4)
                        self._fk_quat_offset[:] = quat_mul(gt_quat_f64, quat_inverse(fk_quat_f64)).flatten()
                        self._fk_calibrated = True
                        try:
                            os.remove(self._fk_calibrate_trigger)
                        except OSError:
                            pass
                        logger.info(
                            f"FK calibrated: pos_offset={self._fk_pos_offset}, "
                            f"quat_offset w={self._fk_quat_offset[0]:.4f} z={self._fk_quat_offset[3]:.4f}"
                        )
                fk_pos = (fk_pos.astype(np.float64) + self._fk_pos_offset).astype(np.float32)
                if self._use_mocap_orientation:
                    quat_wxyz = (
                        quat_mul(self._fk_quat_offset.reshape(1, 4), fk_quat.astype(np.float64).reshape(1, 4))
                        .flatten()
                        .astype(np.float32)
                    )
                vel = self._compute_pelvis_velocity(fk_pos)
                if self._gt_pelvis_mocap is not None:
                    gt = self._gt_pelvis_mocap.get_state()
                    gt_pos = gt[0] if gt else np.zeros(3, dtype=np.float32)
                    gt_quat = gt[1] if gt else np.zeros(4, dtype=np.float32)
                    self._fk_debug_log.append(
                        np.concatenate(
                            [
                                mocap_state[0].flatten(),
                                mocap_state[1].flatten(),
                                fk_pos.flatten(),
                                quat_wxyz.flatten() if self._use_mocap_orientation else fk_quat.flatten(),
                                gt_pos.flatten(),
                                gt_quat.flatten(),
                                waist_joints.flatten(),
                                state[0, 3:7].flatten(),
                            ]
                        )
                    )
                pos = fk_pos
            elif self._head_offset is not None:
                pos = self._apply_head_offset(pos, quat_wxyz)
            state[0, 0:3] = pos
            if self._use_mocap_orientation:
                state[0, 3:7] = quat_wxyz
            if vel is not None:
                state[0, self._lin_vel_start : self._lin_vel_start + 3] = vel
        return state

    def _apply_real_head_to_pelvis(
        self, head_pos: np.ndarray, imu_quat_wxyz: np.ndarray, waist_joints: np.ndarray
    ) -> np.ndarray:
        """Convert real head mocap position to pelvis via torso FK.

        Chain: head → torso (FIXED body-frame offset, full quaternion rotation
        since head+torso are the same rigid body and IMU is in torso)
        → pelvis (dynamic offset via waist roll/pitch joint angles).
        """
        q_roll, q_pitch = float(waist_joints[1]), float(waist_joints[2])
        t2p = self._torso_to_pelvis_body_offset(q_roll, q_pitch)
        body_offset = self._REAL_HEAD_TO_TORSO_BODY + t2p
        world_offset = quat_apply(imu_quat_wxyz.reshape(1, 4), body_offset.reshape(1, 3)).flatten()
        return (head_pos.astype(np.float64) + world_offset).astype(np.float32)

    def _head_quat_to_pelvis_quat(self, head_quat_wxyz: np.ndarray, waist_joints: np.ndarray) -> np.ndarray:
        """Convert head/torso mocap orientation to pelvis orientation via inverse waist FK.

        mid360 is rigidly attached to torso (no mount rotation offset), so q_head = q_torso.
        q_torso = q_pelvis * q_z(yaw) * q_x(roll) * q_y(pitch)
        => q_pelvis = q_torso * q_waist_inv
        """
        q_yaw, q_roll, q_pitch = float(waist_joints[0]), float(waist_joints[1]), float(waist_joints[2])
        cy, sy = np.cos(q_yaw / 2), np.sin(q_yaw / 2)
        cr, sr = np.cos(q_roll / 2), np.sin(q_roll / 2)
        cp, sp = np.cos(q_pitch / 2), np.sin(q_pitch / 2)
        qz = np.array([[cy, 0, 0, sy]], dtype=np.float64)
        qx = np.array([[cr, sr, 0, 0]], dtype=np.float64)
        qy = np.array([[cp, 0, sp, 0]], dtype=np.float64)
        q_waist = quat_mul(quat_mul(qz, qx), qy)
        q_pelvis = quat_mul(head_quat_wxyz.reshape(1, 4).astype(np.float64), quat_inverse(q_waist))
        return q_pelvis.flatten().astype(np.float32)

    def _torso_to_pelvis_body_offset(self, q_roll: float, q_pitch: float) -> np.ndarray:
        """Torso→pelvis offset in torso body frame, accounting for waist joints.

        FK: torso = pelvis + R_pelvis * R_z(yaw) * waist_offset
        Inverse: pelvis = torso - R_torso * R_y(-pitch) * R_x(-roll) * waist_offset
        The yaw cancels (R_z(-yaw)*R_z(yaw)=I), so only roll/pitch matter.
        Returns negated offset to add to torso_pos.
        """
        ox, oy, oz = self._WAIST_OFFSET
        cr, sr = np.cos(q_roll), np.sin(q_roll)
        x1, y1, z1 = ox, cr * oy + sr * oz, -sr * oy + cr * oz
        cp, sp = np.cos(q_pitch), np.sin(q_pitch)
        x2, y2, z2 = cp * x1 - sp * z1, y1, sp * x1 + cp * z1
        return np.array([-x2, -y2, -z2], dtype=np.float64)

    def _compute_pelvis_velocity(self, pelvis_pos: np.ndarray) -> np.ndarray | None:
        t = time.time()
        idx = self._pelvis_hist_idx % self._pelvis_vel_window
        self._pelvis_pos_history[idx] = pelvis_pos
        self._pelvis_time_history[idx] = t
        self._pelvis_hist_idx += 1
        count = min(self._pelvis_hist_idx, self._pelvis_vel_window)
        if count >= 2:
            oldest = (self._pelvis_hist_idx - count) % self._pelvis_vel_window
            newest = (self._pelvis_hist_idx - 1) % self._pelvis_vel_window
            dt = self._pelvis_time_history[newest] - self._pelvis_time_history[oldest]
            if dt > 1e-6:
                return ((self._pelvis_pos_history[newest] - self._pelvis_pos_history[oldest]) / dt).astype(np.float32)
        return None

    def _apply_head_offset(self, pos: np.ndarray, quat_wxyz: np.ndarray) -> np.ndarray:
        """Rotate body-frame offset by orientation and add to mocap position."""
        rotated = quat_apply(quat_wxyz.reshape(1, 4), self._head_offset.reshape(1, 3)).flatten()
        return (pos.astype(np.float64) + rotated).astype(np.float32)

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def shutdown(self):
        if self._fk_debug_log:
            path = os.environ.get("HOLOSOMA_DEBUG_LOG", "")
            if path:
                fk_path = path.replace(".npz", "_fk_compare.npz")
            else:
                fk_path = "/tmp/fk_compare.npz"
            arr = np.array(self._fk_debug_log)
            np.savez(fk_path, fk_compare=arr)
            logger.info(f"FK compare log saved: {fk_path} ({len(arr)} frames)")
        self._mocap.shutdown()
        if self._gt_pelvis_mocap is not None:
            self._gt_pelvis_mocap.shutdown()
        if hasattr(self._inner, "shutdown"):
            self._inner.shutdown()


def register_interface(policy):
    """Register and configure the appropriate interface for the policy.

    For sim2real: pass an already-created policy.
    For headless sim2sim: pass policy_class + config (creates policy with
    MuJoCoInterface injected before __init__, bypassing hardware detection).

    Based on config, this may:
    1. Create policy with MuJoCoInterface (headless sim2sim)
    2. Wrap with FK odometry (headless no-mocap or sim2real fk_odometry)
    3. Wrap with mocap receiver (sim2real with mocap)
    """
    config = policy.config
    task = config.task
    headless = getattr(task, "headless_sim_eval", False)

    # 2. FK odometry wrapping
    _fk_mode = None
    if headless and task.headless_no_mocap and task.headless_fk_mode != "none":
        _fk_mode = task.headless_fk_mode
    elif not headless and getattr(task, "fk_odometry", False):
        _fk_mode = "pinocchio"

    if _fk_mode is not None:
        _register_fk_odometry(policy, _fk_mode)

    # 3. Mocap wrapping (sim2real or sim2sim)
    if not headless:
        mocap_body = getattr(task, "mocap_body_name", "")
        sim_mocap = getattr(task, "sim_mocap", False)
        if mocap_body:
            _register_mocap(policy, mocap_body)
        elif sim_mocap:
            _register_sim_mocap(policy)

    return policy


def _create_headless_policy(policy_class, config):
    """Create policy with MuJoCoInterface, bypassing hardware init.

    Uses object.__new__ + _shared_hardware_source injection so that
    BasePolicy.__init__ skips create_interface() (which needs real hardware).
    """
    from wbt_wrappers_inference.utils.headless_sim_factory import MuJoCoInterface, _HardwareSource, _setup_headless_sim

    mj_sim = _setup_headless_sim(config)
    # _setup_headless_sim may rebuild config (e.g. terrain replay updates ref_motion_path).
    config = mj_sim["config"]
    task = config.task

    sim_params = mj_sim["sim_params"]
    steps_per_control = max(1, int(sim_params["control_dt"] / sim_params["sim_dt"]))

    # Eval-3 external force (UniFP F_ext): real PhysX force injected at the hand.
    # HOLOSOMA_FORCE_EXT_LOCAL="L_fx,L_fy,L_fz,R_fx,R_fy,R_fz" (raw N, pelvis
    # frame), mirroring HOLOSOMA_FORCE_CMD_LOCAL. Unset → no external force
    # (Eval 1/2). The metric side does NOT see this — it reads the realized hand
    # displacement, so a faithful F_cmd=-F_ext canceller shows ~zero drift.
    import numpy as _np

    _fe_raw = os.environ.get("HOLOSOMA_FORCE_EXT_LOCAL", "")
    force_ext_pelvis = None
    if _fe_raw:
        _fe = [float(x) for x in _fe_raw.split(",")]
        if len(_fe) != 6:
            raise ValueError(
                f"HOLOSOMA_FORCE_EXT_LOCAL must be 6 comma-separated Newtons "
                f"[L_fx,L_fy,L_fz,R_fx,R_fy,R_fz], got {len(_fe)}: {_fe_raw!r}"
            )
        force_ext_pelvis = _np.array(_fe, dtype=_np.float64).reshape(2, 3)

    mj_interface = MuJoCoInterface(
        mj_sim["model"],
        mj_sim["data"],
        config.robot,
        kp=sim_params["kp"],
        kd=sim_params["kd"],
        torque_limit=sim_params["torque_limit"],
        steps_per_control=steps_per_control,
        no_mocap=task.headless_no_mocap,
        mocap_yaw_offset=mj_sim["mocap_yaw_rad"],
        mocap_use_orientation=task.mocap_use_orientation,
        force_ext_pelvis=force_ext_pelvis,
    )

    hw_source = _HardwareSource(mj_interface, rl_rate=task.rl_rate)
    # TODO: this is a ugly wordaround for the limitation of the
    # function _init_communication_components in basePolicy
    policy = object.__new__(policy_class)
    policy._shared_hardware_source = hw_source
    policy.__init__(config=config)

    # For headless sim2sim, there's no human operator to press "start".
    # So the policy would sit in stiff-startup mode forever, never running ONNX inference.
    # Setting these two flags bypasses the startup state
    policy.motion_clip_progressing = True
    policy.use_policy_action = True
    policy._headless_sim = mj_sim

    if getattr(config.task, "headless_terrain", "") and hasattr(policy, "_mj_model"):
        policy._mj_model = mj_sim["model"]
        policy._mj_data = mj_sim["data"]

    return policy


def _register_fk_odometry(policy, fk_mode):
    """Wrap policy.interface with FK odometry."""
    config = policy.config
    mj_sim = getattr(policy, "_headless_sim", None)
    control_dt = mj_sim["sim_params"]["control_dt"] if mj_sim else 1.0 / config.task.rl_rate

    if fk_mode == "mujoco":
        from wbt_wrappers_inference.utils.fk_odometry import MuJoCoFKOdometryWrapper

        init_pos = mj_sim["data"].qpos[0:3].copy()
        policy.interface = MuJoCoFKOdometryWrapper(
            policy.interface, mj_sim["model"], mj_sim["data"], control_dt=control_dt, init_pos=init_pos
        )
        logger.info("FK odometry: MuJoCo xpos (sim-only diagnostic)")
    else:
        # Pinocchio FK — URDF from ONNX metadata (headless) or file (sim2real)
        urdf_text = ""
        if mj_sim:
            urdf_text = mj_sim["sim_params"].get("robot_urdf", "")
        else:
            urdf_path = getattr(config.task, "robot_urdf_path", "")
            if urdf_path:
                from pathlib import Path

                p = Path(urdf_path)
                if p.is_file():
                    urdf_text = p.read_text()
            if not urdf_text:
                urdf_text = getattr(policy, "_robot_urdf_text", "")
        if urdf_text:
            init_pos = mj_sim["data"].qpos[0:3].copy() if mj_sim else None
            kwargs = {"control_dt": control_dt}
            if init_pos is not None:
                kwargs["init_pos"] = init_pos
            else:
                kwargs["num_joints"] = policy.num_dofs
            from wbt_wrappers_inference.utils.fk_odometry import FKOdometryWrapper

            policy.interface = FKOdometryWrapper(policy.interface, config.robot, urdf_text, **kwargs)
            logger.info("FK odometry enabled (Pinocchio)")
        else:
            logger.warning("FK odometry requested but no URDF available")


def _register_sim_mocap(policy):
    """Wrap policy.interface with sim mocap (UDP from sim bridge).

    The sim bridge sends ground-truth pelvis state. When head mocap flags are
    set (``mocap_real_head_to_torso``), wraps the receiver with
    SimPelvisToHeadAdapter so MocapInterfaceWrapper receives "fake head mocap"
    and exercises the full head→pelvis conversion path (same as sim2real).
    """
    task = policy.config.task
    mocap_use_ori = getattr(task, "mocap_use_orientation", False)
    real_head_to_torso = getattr(task, "mocap_real_head_to_torso", False)

    sim_mocap = SimMocapReceiver()
    if real_head_to_torso:
        sim_mocap = _SimPelvisToHeadAdapter(sim_mocap)

    policy.interface = MocapInterfaceWrapper(
        policy.interface,
        sim_mocap,
        num_joints=policy.num_dofs,
        use_mocap_orientation=mocap_use_ori,
        real_head_to_torso=real_head_to_torso,
    )
    logger.info(
        f"Sim mocap enabled (UDP from sim bridge), "
        f"use_orientation={mocap_use_ori}, real_head_to_torso={real_head_to_torso} "
        f"— waiting up to 5s for first data..."
    )

    for i in range(50):
        if sim_mocap.has_data:
            pos, quat, vel = sim_mocap.get_state()
            logger.info(f"Sim mocap OK: pos={pos}, vel={vel}")
            break
        time.sleep(0.1)
    else:
        logger.error(
            "NO SIM MOCAP DATA after 5s! Is run_sim.py running with the Unitree bridge? "
            "Check that both processes are on the same machine and SIM_MOCAP_UDP_PORT matches."
        )


class _SimPelvisToHeadAdapter:
    """Converts sim bridge pelvis ground truth to fake head (mid360) mocap.

    Applies the forward URDF offset pelvis→torso→mid360 so that
    MocapInterfaceWrapper's real_head_to_torso path can convert it back,
    exercising the same code path as sim2real head mocap.
    """

    _PELVIS_TO_MID360_BODY = np.array([0.0002835, 0.00003, 0.41618], dtype=np.float64)
    _WAIST_OFFSET = np.array([-0.0039635, 0.0, 0.044], dtype=np.float64)

    def __init__(self, inner: SimMocapReceiver):
        self._inner = inner

    @property
    def has_data(self) -> bool:
        return self._inner.has_data

    def get_state(self):
        state = self._inner.get_state()
        if state is None:
            return None
        pos, quat_wxyz, vel = state
        world_offset = quat_apply(
            quat_wxyz.reshape(1, 4).astype(np.float64), (self._WAIST_OFFSET + self._PELVIS_TO_MID360_BODY).reshape(1, 3)
        ).flatten()
        head_pos = (pos.astype(np.float64) + world_offset).astype(np.float32)
        return (head_pos, quat_wxyz, vel)

    def shutdown(self):
        self._inner.shutdown()


def _register_mocap(policy, mocap_body):
    """Wrap policy.interface with mocap receiver."""
    task = policy.config.task
    mocap_use_ori = getattr(task, "mocap_use_orientation", False)
    head_to_torso = getattr(task, "mocap_head_to_torso", False)
    real_head_to_torso = getattr(task, "mocap_real_head_to_torso", False)
    mocap = MocapReceiver(mocap_body)
    policy.interface = MocapInterfaceWrapper(
        policy.interface,
        mocap,
        num_joints=policy.num_dofs,
        use_mocap_orientation=mocap_use_ori,
        head_to_torso=head_to_torso,
        real_head_to_torso=real_head_to_torso,
    )
    logger.info(
        f"Mocap enabled: body={mocap_body}, use_orientation={mocap_use_ori}, "
        f"head_to_torso={head_to_torso}, real_head_to_torso={real_head_to_torso}"
    )
    topic = mocap_body if mocap_body.startswith("/") else f"/{mocap_body}/pose"
    logger.info(f"Mocap topic: {topic} — waiting up to 5s for first data...")

    for i in range(50):
        if mocap.has_data:
            pos, quat, vel = mocap.get_state()
            logger.info(f"Mocap OK: pos={pos}, quat={quat}")
            break
        time.sleep(0.1)
    else:
        logger.error(
            f"NO MOCAP DATA after 5s! Topic {topic} is not publishing. "
            "Check: (1) mocap source is running, (2) topic/body name matches, "
            "(3) ROS2 DDS discovery works (try: ros2 topic list). "
            "Without mocap, position/velocity will be zeros → policy WILL fail."
        )
