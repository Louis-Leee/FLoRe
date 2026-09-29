#!/usr/bin/env python3
"""High-level PLANNER server — the S1 half of the S0↔S1 force deploy.

This is the online, real-robot sibling of the OFFLINE ``scripts/muse_force_infer_server.py``
(the rollout-viz backend). Where that one replays a recorded zarr window on click, THIS one
assembles a LIVE observation from the running rig every step, runs the MuSe-on-UVA force
policy (``predict_action``), and STREAMS the predicted LEFT-hand pose chunk to the low-level
controller exactly the way the VIVE teleop bridge does — so the trained high-level planner
drives the arm and the low-level RL controller executes it.

It runs INSIDE the ``uvaforce`` conda env (torch 2.7.1+cu128, hydra, diffusers — and NO
viser/mink). The viser control/visualization page (``scripts/muse_force_serving_viz.py``)
runs in FUJI's ``.venv`` and SPAWNS this process, talking to it over the same base64-npz
line protocol the rollout viz uses. That two-process split (mutually exclusive envs) is the
same one ``run_muse_force_rollout_viz.sh`` wires up.

WHAT IT DOES, in order
----------------------
1. **Self-check** (like ``collection_data.sh`` / ``run_collection_viz.sh``): confirm the
   gripper reads (and, optionally, opens/closes), the F/T sensor is streaming, and the
   camera is delivering frames — BEFORE any model inference. Each is a WS probe against
   the bridges the deploy already runs (camera :8773, gripper :8784, F/T :8771).
2. **Live obs-builder**: maintain rolling ring buffers (camera frames, EE-pose targets,
   gripper width, 6-DoF force) and assemble the exact obs dict ``predict_action`` consumes
   at eval time — poses relative to the *current* EE anchor, ``_wrt_start`` relative to a
   *latched* start pose, force in the Bota sensor frame at physical N/N·m, gripper in metres.
3. **RTC-ish chunk loop**: on a Run-1-chunk request, or continuously, run ``predict_action``,
   un-relativize the LEFT-hand chunk to ABSOLUTE poses (against the running anchor), and
   drip-feed the steps to the low-level controller over WS :8770 (``S0Driver``), holding the
   right hand + head at their ready-pose home (UVA-umi is single-robot, LEFT only).
4. **Report to the frontend**: emit the executed chunk + predicted force + gripper + status
   (low-level online?, running?) over the pipe so the viser page can visualize it live.

FRAME CONTRACT (the load-bearing correctness point — see docs/muse_on_uva_force_design.md)
------------------------------------------------------------------------------------------
The model is TRAINED (UMI data) on EE-LOCAL RELATIVE quantities:
  * obs ``robot0_eef_pos``/``robot0_eef_rot_axis_angle`` are relative to the *current* EE
    frame (the last observed pose), ``_wrt_start`` relative to the latched episode-start pose;
  * force is in the Bota SENSOR frame;
  * the action chunk is relative to the *current* EE anchor (``anchor_mat @ action_mat``).
Relative deltas are FRAME-COVARIANT, and the UMI TCP extrinsic is a PURE TRANSLATION
(``T_tracker_tcp`` = [I | 0.21·x], rotation identity — robot_assets/calibration/tcp_extrinsic.json),
so the UMI-TCP frame, the VIVE tracker frame, and ``left_wrist_yaw_link`` share ONE orientation.
Therefore we keep the whole running anchor in the low-level's PELVIS frame (seeded from the
ready-pose home in ``home_anchor.json``), advance it by the model's relative delta IN THAT
FRAME (no axis-convention rotation needed — exactly why the DP-Flow deploy path needs none),
and only apply the fixed TCP→contact-point lever when converting to the ``vr_3point`` wire.
This is the same un-relativize math as ``fuji/s1_policy/serving/server.py::_chunk_to_absolute``
and ``umi.../real_inference_util.get_real_umi_action``.

This process reads NOTHING from ``UMI_collection``/``holosoma_extensions`` at import time; it
only subscribes to their localhost WebSockets. It writes NO files.
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import os
import sys
import threading
import time

import numpy as np

# --- keep fd 1 (stdout) clean: the model/hydra/VAE chatter must go to fd 2 ----
# The stashed real stdout is the pure protocol channel; every stray print() lands
# on stderr. Identical discipline to muse_force_infer_server.py.
_REAL_STDOUT = sys.stdout
sys.stdout = sys.stderr

_HERE = os.path.dirname(os.path.abspath(__file__))
_FUJI_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
_UVA = os.path.join(_FUJI_ROOT, "third_party", "unified_video_action")
_UMI = os.path.join(_FUJI_ROOT, "third_party", "UMI_collection")

# FUJI root on sys.path up front: LivePlanner uses `fuji.utils.rot6d` and main() uses
# `fuji.client.s0_driver`. Backend.__init__ does os.chdir(_UVA), so a bare 'fuji' import
# after that chdir would fail unless the ABSOLUTE FUJI root is already importable — add
# it here (absolute) so it survives the chdir. (uvaforce env: verified `fuji` imports.)
if _FUJI_ROOT not in sys.path:
    sys.path.insert(0, _FUJI_ROOT)

_MSG_PREFIX = "MUSEVIZ"


def _emit(tag: str, **arrays) -> None:
    """Write one protocol line ``MUSEVIZ<tag> <base64(npz)>`` to the real stdout.

    Same wire format as muse_force_infer_server.py so the frontend's reader is reused."""
    buf = io.BytesIO()
    np.savez(buf, **arrays)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    _REAL_STDOUT.write(f"{_MSG_PREFIX}{tag} {b64}\n")
    _REAL_STDOUT.flush()


def _log(msg: str) -> None:
    print(f"[muse-serve] {msg}", file=sys.stderr, flush=True)


# ===========================================================================
# Live sensor subscribers (read-only WS clients; hub-and-spoke, like record_teleop.py)
# ===========================================================================
class _WSSubscriber(threading.Thread):
    """Auto-reconnecting read-only WebSocket client that keeps the latest decoded
    JSON message. It reads continuously; the ONLY upstream traffic is the caller's
    explicit control frames via :meth:`send` — two uses: (1) the F/T tare
    (``{"cmd":"tare"}`` to the F/T bridge) and (2) autonomous LEFT-gripper POSITION
    commands (``{"target_frac": …}`` to the gripper bridge, gated by
    ``--command-gripper``). One thread per bridge; the model loop reads :attr:`latest`
    under the lock. The send queue COALESCES repeated ``target_frac`` frames to the
    newest (a position latch — only the latest matters) so a disconnected bridge can't
    grow the queue without bound, while never dropping a discrete command like tare."""

    def __init__(self, url: str, name: str):
        super().__init__(daemon=True, name=f"ws-{name}")
        self.url = url
        self._name = name
        self._lock = threading.Lock()
        self._latest: dict | None = None
        self._latest_wall = 0.0
        self._n = 0
        self._stop = threading.Event()
        self.connected = False
        self._send_q: list[str] = []

    def latest(self) -> tuple[dict | None, float, int]:
        with self._lock:
            return (self._latest, self._latest_wall, self._n)

    def send(self, obj: dict) -> None:
        """Queue a control frame to send on the next loop tick.

        Two callers: the F/T tare (``{"cmd":"tare"}``) and autonomous gripper position
        commands (``{"target_frac": …}``, gated by ``--command-gripper``). To bound the
        queue when the bridge is disconnected (the drip loop keeps enqueueing at
        drip_hz), a ``target_frac`` frame COALESCES with a pending one — it replaces the
        last queued target_frac instead of appending a stale duplicate, since only the
        newest position matters. Discrete commands (tare, open/close/stop) are NEVER
        coalesced or dropped — they always append and are delivered in order."""
        payload = json.dumps(obj)
        with self._lock:
            if "target_frac" in obj and self._send_q and "target_frac" in self._send_q[-1]:
                self._send_q[-1] = payload      # coalesce: keep only the newest target
            else:
                self._send_q.append(payload)

    def stop(self) -> None:
        self._stop.set()

    def run(self) -> None:
        import asyncio

        async def _loop():
            import websockets

            while not self._stop.is_set():
                try:
                    async with websockets.connect(self.url, max_size=None, ping_interval=None) as ws:
                        self.connected = True
                        _log(f"{self._name}: connected {self.url}")
                        while not self._stop.is_set():
                            # flush any queued control frames
                            with self._lock:
                                pending = self._send_q
                                self._send_q = []
                            for p in pending:
                                await ws.send(p)
                            try:
                                raw = await asyncio.wait_for(ws.recv(), timeout=1.0)
                            except asyncio.TimeoutError:
                                continue
                            try:
                                msg = json.loads(raw)
                            except (ValueError, TypeError):
                                continue
                            with self._lock:
                                self._latest = msg
                                self._latest_wall = time.time()
                                self._n += 1
                except Exception as exc:  # noqa: BLE001 — reconnect on any drop
                    self.connected = False
                    if not self._stop.is_set():
                        _log(f"{self._name}: disconnected ({exc!r}); retrying…")
                        await asyncio.sleep(0.5)
            self.connected = False

        asyncio.new_event_loop().run_until_complete(_loop())


def _decode_jpeg_b64(b64: str) -> np.ndarray | None:
    """Decode a base64 JPEG (camera bridge frame) to an HWC uint8 RGB array.

    The realsense bridge encodes ``rs.format.rgb8`` frames as JPEG, so cv2's BGR
    decode order must be swapped back to RGB (the training converter used
    ``bgr_to_rgb=False`` on an already-RGB source; the bridge's JPEG round-trips
    through cv2 which yields BGR, so we swap once here to land on RGB)."""
    import cv2

    try:
        buf = np.frombuffer(base64.b64decode(b64), dtype=np.uint8)
        bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if bgr is None:
            return None
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    except Exception:  # noqa: BLE001
        return None


def _resize_center_crop_224(img_rgb: np.ndarray) -> np.ndarray:
    """Aspect-preserving resize + CENTER CROP to 224×224 (cover fit), BYTE-IDENTICAL to
    the training converter (umi_convert_core.compute_crop_resize + image_resize).

    The converter uses EXACT INTEGER geometry: ``scale=max(out/in_w,out/in_h)``,
    ``resize=ceil(in*scale)`` (NOT round), clamped ``>= out``, then a floor-div center
    crop, and ALWAYS ``cv2.INTER_AREA`` (not INTER_LINEAR on upscales). A naive
    round()+INTER_LINEAR reimplementation drifts by ~1 px on 640×360/480×270 D405 sizes
    (verified maxdiff 155/160) — an out-of-distribution crop for the model. So we mirror
    the converter's math exactly. Returns HWC uint8 RGB 224×224."""
    import cv2

    in_h, in_w = img_rgb.shape[:2]
    out_w = out_h = 224
    scale = max(out_w / in_w, out_h / in_h)
    resize_w = max(int(np.ceil(in_w * scale)), out_w)
    resize_h = max(int(np.ceil(in_h * scale)), out_h)
    crop_x = (resize_w - out_w) // 2
    crop_y = (resize_h - out_h) // 2
    resized = cv2.resize(img_rgb, (resize_w, resize_h), interpolation=cv2.INTER_AREA)
    return resized[crop_y:crop_y + out_h, crop_x:crop_x + out_w]


class GripperCalib:
    """Encoder-radians → width-in-metres 2-point linear map. PREFERS the live
    per-sample ``open``/``closed`` endpoints from the gripper WS (the self-test's
    discovered stops), falling back to the JSON nominal — matching the converter's
    ``effective_gripper_calib``. The DM4310 re-references per power-cycle, so the
    JSON nominal endpoints usually don't match a live session; using them would
    clamp every width to one rail."""

    def __init__(self, width_open_m: float = 0.095, width_closed_m: float = 0.0):
        self.width_open_m = float(width_open_m)
        self.width_closed_m = float(width_closed_m)

    def to_width_m(self, actual_rad: float, open_rad: float, closed_rad: float) -> float:
        denom = (open_rad - closed_rad)
        if abs(denom) < 1e-6:
            return self.width_closed_m
        frac = (actual_rad - closed_rad) / denom
        frac = float(np.clip(frac, 0.0, 1.0))
        return self.width_closed_m + frac * (self.width_open_m - self.width_closed_m)

    def to_target_frac(self, width_m: float) -> float:
        """Inverse of :meth:`to_width_m` in the fraction sense: a predicted width in
        metres → the open-fraction in [0,1] for the gripper's ``{"target_frac": …}``
        POSITION command (1.0=open, 0.0=closed). Uses the SAME metre endpoints as the
        read path, so a commanded width maps to EXACTLY the open-fraction we would read
        back for it — the ``width_open_m`` constant cancels in the round trip, and the
        gripper bridge re-expands this frac against ITS own live rad endpoints (the
        self-test's discovered stops). Clamped so a bad width can never overtravel."""
        span = self.width_open_m - self.width_closed_m
        if abs(span) < 1e-9:
            return 0.0
        frac = (float(width_m) - self.width_closed_m) / span
        return float(np.clip(frac, 0.0, 1.0))


# ===========================================================================
# Pose math (pelvis-frame, EE-relative). Inlined pytorch3d-free helpers, matching
# muse_force_infer_server._unrelativize_action and get_real_umi_{obs,action}.
# ===========================================================================
def _pose_helpers():
    """Import the UVA pose helpers (available in the uvaforce env, verified)."""
    from umi.common.pose_util import pose_to_mat, mat_to_pose, mat_to_pose10d, pose10d_to_mat
    from unified_video_action.common.pose_repr_util import convert_pose_mat_rep
    return pose_to_mat, mat_to_pose, mat_to_pose10d, pose10d_to_mat, convert_pose_mat_rep


class LivePlanner:
    """Owns the model, the live ring buffers, and the running LEFT-hand anchor.

    Anchor + all internal poses are in the low-level's PELVIS frame (seeded from
    home_anchor.json). The model's obs is built EE-relative from the running
    anchor history; the model's relative action chunk is un-relativized against
    the running anchor (pelvis frame) → absolute pelvis-frame LEFT-hand TCP poses.
    """

    # UMI TCP is +0.21 m along wrist +x (pure translation, rotation identity —
    # tcp_extrinsic.json). The vr_3point wire carries the CONTACT POINT
    # wrist_origin + R(wrist)@offset with offset [0.18,-0.025,0] (home_anchor.json
    # vr_3point_offsets[0]). Both are rigid points on the same wrist; we track the
    # UMI-TCP point internally (the model's native frame) and convert to the wire
    # contact point with this fixed lever difference at publish time.
    _TCP_OFFSET = np.array([0.21, 0.0, 0.0], dtype=np.float64)      # wrist→UMI-TCP
    _WIRE_OFFSET = np.array([0.18, -0.025, 0.0], dtype=np.float64)  # wrist→vr_3point contact (L)

    def __init__(self, backend, home_anchor: dict, obs_hz: float):
        self.be = backend  # the model Backend (below); holds policy + cfg + ft_scale
        self.obs_hz = obs_hz
        self._lock = threading.Lock()

        # --- ready-pose home (pelvis frame), LEFT hand = index 0 ---
        # These are the CONTACT-POINT poses the wire carries (home_anchor.json).
        self._home_pos_wire = np.asarray(home_anchor["home_pos_pelvis_3pt"], dtype=np.float64)     # (3,3)
        self._home_quat_xyzw = np.asarray(home_anchor["home_orn_quat_xyzw_3pt"], dtype=np.float64)  # (3,4)
        # right(1) + head(2) held here for the whole run (UVA predicts LEFT only).

        # LEFT-hand running anchor at the UMI-TCP point (pelvis frame). Seeded from
        # the ready-pose home: convert the wire contact point → UMI-TCP point using
        # the shared wrist orientation (see _wire_to_tcp / _tcp_to_wire).
        self._anchor_pos, self._anchor_rot6d = self._wire_home_to_tcp_anchor()
        # Latched episode-start pose (for _wrt_start), captured on reset().
        self._start_pos = self._anchor_pos.copy()
        self._start_rot6d = self._anchor_rot6d.copy()

        H = int(self.be.obs_history_len)   # 16
        self._img_hist_len = int(self.be.num_img_frames)  # 8 (4 cond + 4 target slots)
        # CRITICAL (consistency review): the action/force head conditions on ONLY the FIRST
        # HALF of the 8 image frames — get_vae_latent does `c, x = torch.chunk(x, 2, dim=2)`
        # and policy_model uses only `c`. The training dataset packs camera0_rgb in ASCENDING
        # include_indices order [-12,-8,-4,0,4,8,12,16], so the conditioning half is the 4
        # RECENT frames [-12,-8,-4,0] ENDING AT THE CURRENT FRAME (the target half [4..16] is
        # the video objective, inert for the action head). So we only need to buffer the 4
        # CONDITIONING frames (newest last, ending at NOW) and duplicate the newest into the
        # target half at stack time. Buffering 8 history frames [oldest..newest] instead put
        # the CURRENT frame in the target half and conditioned the arm on the OLDEST frames.
        self._n_img_cond = max(1, self._img_hist_len // 2)  # 4
        # Ring buffers, newest last. Seeded with the anchor so a cold start is valid.
        self._buf_pos = [self._anchor_pos.copy() for _ in range(H)]
        self._buf_rot6d = [self._anchor_rot6d.copy() for _ in range(H)]
        self._buf_grip = [0.0 for _ in range(H)]
        self._buf_force = [np.zeros(6, dtype=np.float64) for _ in range(H)]
        self._buf_img = [None for _ in range(self._n_img_cond)]
        self._hist_len = H

        # Gripper width in metres (latest), fed as proprio; NOT commanded (manual-only).
        self._grip_m = 0.0

    # --- frame helpers ---------------------------------------------------
    def _wire_home_to_tcp_anchor(self):
        """LEFT ready-pose CONTACT-POINT (wire) → UMI-TCP anchor (pos, rot6d), pelvis."""
        from fuji.utils.rot6d import quat_xyzw_to_rotmat, rotmat_to_rot6d  # type: ignore
        R = quat_xyzw_to_rotmat(self._home_quat_xyzw[0])           # (3,3) wrist orientation
        # contact = wrist + R@WIRE_OFFSET ; tcp = wrist + R@TCP_OFFSET
        wrist = self._home_pos_wire[0] - R @ self._WIRE_OFFSET
        tcp = wrist + R @ self._TCP_OFFSET
        return tcp.astype(np.float64), rotmat_to_rot6d(R).astype(np.float64)

    def _tcp_to_wire(self, tcp_pos: np.ndarray, rot6d: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """UMI-TCP pose (pos, rot6d) → vr_3point LEFT CONTACT-POINT (pos, quat_xyzw).

        Same shared wrist orientation for both points (pure-translation extrinsic):
        wrist = tcp − R@TCP_OFFSET ; contact = wrist + R@WIRE_OFFSET."""
        from fuji.utils.rot6d import rot6d_to_rotmat, rotmat_to_quat_xyzw  # type: ignore
        R = rot6d_to_rotmat(np.asarray(rot6d, dtype=np.float64))
        wrist = np.asarray(tcp_pos, dtype=np.float64) - R @ self._TCP_OFFSET
        contact = wrist + R @ self._WIRE_OFFSET
        return contact.astype(np.float64), rotmat_to_quat_xyzw(R).astype(np.float64)

    def home_wire(self) -> tuple[np.ndarray, np.ndarray]:
        """The full ready-pose vr_3point wire (pos[9], orn[12] xyzw) — LEFT at its
        current running anchor converted to the contact point, RIGHT + HEAD held at
        home. Published as an IDLE HEARTBEAT before/between chunks so the low-level's
        pre-arm pose gate sees a live frame that matches its ready pose (all three
        per-point gaps ~0 → arm PASSES). Before the first chunk the anchor == home, so
        this is exactly the home pose; after a chunk it tracks the last executed step
        (which is where the arm is), so re-arming/holding stays gap-free."""
        with self._lock:
            anchor_pos = self._anchor_pos.copy()
            anchor_rot6d = self._anchor_rot6d.copy()
        lc_pos, lc_quat = self._tcp_to_wire(anchor_pos, anchor_rot6d)
        wire_pos = np.concatenate([lc_pos, self._home_pos_wire[1], self._home_pos_wire[2]])
        wire_orn = np.concatenate([lc_quat, self._home_quat_xyzw[1], self._home_quat_xyzw[2]])
        return wire_pos.astype(np.float64), wire_orn.astype(np.float64)

    # --- live ingest -----------------------------------------------------
    def push_image(self, img_224_rgb: np.ndarray) -> None:
        with self._lock:
            self._buf_img.append(img_224_rgb)
            if len(self._buf_img) > self._n_img_cond:
                self._buf_img.pop(0)

    def push_force(self, wrench6: np.ndarray) -> None:
        with self._lock:
            self._buf_force.append(np.asarray(wrench6, dtype=np.float64))
            if len(self._buf_force) > self._hist_len:
                self._buf_force.pop(0)

    def push_gripper(self, width_m: float) -> None:
        with self._lock:
            self._grip_m = float(width_m)

    def push_proprio_from_anchor(self) -> None:
        """Append the CURRENT running anchor pose + latest gripper to the proprio
        ring buffers. In obs_mode=last_target (the DP-Flow contract) the state the
        model sees at frame t is the last COMMANDED target, so we log the anchor we
        are currently holding/commanding, not a robot readback (which the low-level
        does not expose as an EE pose anyway)."""
        with self._lock:
            self._buf_pos.append(self._anchor_pos.copy())
            self._buf_rot6d.append(self._anchor_rot6d.copy())
            self._buf_grip.append(self._grip_m)
            for b in (self._buf_pos, self._buf_rot6d, self._buf_grip):
                if len(b) > self._hist_len:
                    b.pop(0)

    def reset_start_pose(self) -> None:
        """Latch the episode-start pose = current anchor (for _wrt_start), and
        re-seed the history to the current anchor (fresh episode)."""
        with self._lock:
            self._start_pos = self._anchor_pos.copy()
            self._start_rot6d = self._anchor_rot6d.copy()
            self._buf_pos = [self._anchor_pos.copy() for _ in range(self._hist_len)]
            self._buf_rot6d = [self._anchor_rot6d.copy() for _ in range(self._hist_len)]
            self._buf_grip = [self._grip_m for _ in range(self._hist_len)]

    # --- obs assembly ----------------------------------------------------
    def build_obs(self) -> dict:
        """Assemble the eval-time FLAT obs dict predict_action consumes (keys at top
        level: camera0_rgb, robot0_eef_pos, …; predict_action wraps it in {"obs":…}
        itself and calls resize_image_eval which reads camera0_rgb). All history
        arrays are length-32 (16 real history rows + 16 duplicate 'future' rows the
        eval path never reads for the action head), matching the offline Backend
        window so the model sees identical shapes. Poses are EE-relative to the
        CURRENT anchor; _wrt_start relative to the latched start; force physical
        N/N·m; gripper metres; image (8,3,224,224) float 0-1 CHW."""
        import torch
        (pose_to_mat, _mat_to_pose, mat_to_pose10d, _pose10d_to_mat,
         convert_pose_mat_rep) = _pose_helpers()

        with self._lock:
            pos_hist = np.stack(self._buf_pos)          # (H,3)
            rot6d_hist = np.stack(self._buf_rot6d)      # (H,6)
            grip_hist = np.asarray(self._buf_grip, dtype=np.float64)  # (H,)
            force_hist = np.stack(self._buf_force)      # (H,6)
            imgs = list(self._buf_img)
            anchor_pos = self._anchor_pos.copy()
            anchor_rot6d = self._anchor_rot6d.copy()
            start_pos = self._start_pos.copy()
            start_rot6d = self._start_rot6d.copy()

        # rot6d(6) -> pos6 (pos3 + rotvec3) so pose_to_mat can build SE3. Our buffers
        # store rot6d; convert to axis-angle for the absolute-pose matrices.
        from fuji.utils.rot6d import rot6d_to_rotmat, rotmat_to_rotvec  # type: ignore
        rotvec_hist = np.stack([rotmat_to_rotvec(rot6d_to_rotmat(r)) for r in rot6d_hist])  # (H,3)
        abs_pose6 = np.concatenate([pos_hist, rotvec_hist], axis=1)  # (H,6) pos+rotvec
        pose_mat = pose_to_mat(abs_pose6)                            # (H,4,4)

        # obs poses relative to the CURRENT EE frame. CRITICAL: use the SNAPSHOTTED
        # self._anchor as the base, NOT pose_mat[-1] (the last ingest-buffer row). The
        # anchor is the exact pose infer_chunk un-relativizes the action against, so
        # anchoring the obs to it makes "obs base == action base" a STRUCTURAL guarantee
        # rather than one that depends on the ingest thread having copied the anchor into
        # the buffer before this call. (In the default config pose_mat[-1]==anchor by
        # timing, but at high --drip-hz / low --obs-hz / 1-step replan the buffer can be
        # one chunk stale → a one-chunk position offset injected into the wire.)
        anchor_rotvec = rotmat_to_rotvec(rot6d_to_rotmat(anchor_rot6d))
        anchor_mat = pose_to_mat(np.concatenate([anchor_pos, anchor_rotvec]))
        obs_rel_mat = convert_pose_mat_rep(
            pose_mat, base_pose_mat=anchor_mat, pose_rep=self.be.obs_pose_repr, backward=False)
        obs_pose10d = mat_to_pose10d(obs_rel_mat)                    # (H,9) pos3+rot6d
        eef_pos = obs_pose10d[..., :3]
        eef_rot6d = obs_pose10d[..., 3:9]

        # _wrt_start: relative to the LATCHED start pose (rotation kept). The training
        # reference (real_inference_util.get_real_umi_obs_dict) ALWAYS uses "relative"
        # for the wrt_start base transform regardless of obs_pose_repr, so hardcode it
        # here too (decouples from a future obs_pose_repr!="relative" checkpoint).
        start_rotvec = rotmat_to_rotvec(rot6d_to_rotmat(start_rot6d))
        start_mat = pose_to_mat(np.concatenate([start_pos, start_rotvec]))
        wrt_start_mat = convert_pose_mat_rep(
            pose_mat, base_pose_mat=start_mat, pose_rep="relative", backward=False)
        wrt_start10d = mat_to_pose10d(wrt_start_mat)
        eef_rot6d_wrt_start = wrt_start10d[..., 3:9]                 # (H,6)

        # Duplicate the history to a length-2H window so the eval path's shapes match
        # the training window (predict_action's eval branch reads it whole; the
        # different_history_freq gather indexes with img_indices up to 2H-1, and the
        # force encoder keeps the last H). The duplicate 'future' half is never used
        # by the action head at policy_model.
        def _dup(a):
            return np.concatenate([a, a], axis=0)

        eef_pos32 = _dup(eef_pos)
        eef_rot6d32 = _dup(eef_rot6d)
        eef_rot6d_wrt_start32 = _dup(eef_rot6d_wrt_start)
        grip32 = _dup(grip_hist[:, None])                # (2H,1)
        force32 = _dup(force_hist)                       # (2H,6) physical

        # Image stack (8,3,224,224): the 4 CONDITIONING frames (ending at NOW) in slots
        # 0..3, then the newest frame duplicated into slots 4..7 (the target half, inert for
        # the action head). Mirrors the training layout where torch.chunk's first half =
        # [-12,-8,-4,0] ends at the current frame.
        img_stack = self._image_stack(imgs)

        dev = self.be.device
        obs = {
            "camera0_rgb": torch.as_tensor(img_stack)[None].to(dev),                # (1,8,3,224,224)
            "robot0_eef_pos": torch.as_tensor(eef_pos32)[None].float().to(dev),     # (1,2H,3)
            "robot0_eef_rot_axis_angle": torch.as_tensor(eef_rot6d32)[None].float().to(dev),
            "robot0_eef_rot_axis_angle_wrt_start": torch.as_tensor(eef_rot6d_wrt_start32)[None].float().to(dev),
            "robot0_gripper_width": torch.as_tensor(grip32)[None].float().to(dev),
            "robot0_ft_wrench": torch.as_tensor(force32)[None].float().to(dev),
            "img_indices": torch.as_tensor(self.be.img_indices)[None, :, None].float().to(dev),
        }
        return obs

    def _image_stack(self, imgs) -> np.ndarray:
        """(8,3,224,224) float32 0-1 CHW RGB laid out to MATCH TRAINING's chunk split.

        ``imgs`` is the conditioning ring (len ``_n_img_cond``=4), oldest→newest, ending at
        the CURRENT frame. The model's action head conditions on the FIRST HALF of the
        returned 8-frame tensor (``torch.chunk(x,2,dim=2)`` → first 4), so those 4 slots MUST
        be the recent frames ending at NOW — exactly the training dataset's ascending
        [-12,-8,-4,0] conditioning half. The second half (slots 4..7) is the video target,
        never read by ``task_mode='policy_model'``; we fill it with the newest frame so the
        tensor is well-formed. Cold slots (before 4 real frames arrive) edge-clamp to the
        newest real frame; all-None → mid-grey. This is the CRITICAL fix: the previous
        [oldest..newest] 8-frame buffer put the current frame in the (unused) target half and
        conditioned the arm on the OLDEST frames."""
        n = self._img_hist_len                 # 8
        n_cond = self._n_img_cond              # 4
        real = [im for im in imgs if im is not None]
        newest = real[-1] if real else np.full((224, 224, 3), 128, dtype=np.uint8)
        # Conditioning half: right-align the ring into slots [0..n_cond-1] so the newest lands
        # at slot n_cond-1 (index 3 = "current"); older/missing slots edge-clamp to the
        # oldest real frame (or newest if none), like the dataset's history padding.
        oldest = real[0] if real else newest
        cond = list(imgs)[-n_cond:]
        cond = [oldest] * (n_cond - len(cond)) + cond   # left-pad if the ring isn't full yet
        frames = [(im if im is not None else oldest) for im in cond]
        # Target half: duplicate the newest (inert for the action head).
        frames = frames + [newest] * (n - n_cond)
        out = np.empty((n, 3, 224, 224), dtype=np.float32)
        for i, src in enumerate(frames):
            out[i] = np.transpose(src.astype(np.float32) / 255.0, (2, 0, 1))
        return out

    def commit_anchor(self, pos: np.ndarray, rot6d: np.ndarray) -> None:
        """Advance the running anchor to an EXECUTED chunk step (pelvis-frame UMI-TCP).

        Split out from :meth:`infer_chunk` so the anchor only ever moves to a pose the
        arm was REALLY commanded to: the caller invokes this AFTER a chunk's wire has
        actually been dripped to the low-level. A PREDICT (preview) runs infer_chunk
        but never calls this, so previewing a plan does NOT move the model's frame and
        can be repeated freely before the operator commits to executing it."""
        with self._lock:
            self._anchor_pos = np.asarray(pos, dtype=np.float64).copy()
            self._anchor_rot6d = np.asarray(rot6d, dtype=np.float64).copy()

    # --- one inference + un-relativize (NO anchor advance — see commit_anchor) ---
    def infer_chunk(self, n_action_steps: int | None = None) -> dict:
        """Run ONE model inference and un-relativize the LEFT chunk against the running
        anchor (pelvis frame, UMI-TCP point), returning absolute wire targets +
        predicted force for the frontend. This does NOT advance the anchor (that is an
        explicit :meth:`commit_anchor` the caller makes only after the chunk is really
        executed), so it is safe to call for a PREVIEW that never moves the robot.

        Uses the OFFLINE inference path (``prepare_data_predict_action(eval=False)`` +
        ``sample_tokens(task_mode='policy_model')``) — the SAME path the validated
        rollout backend (muse_force_infer_server.py) uses — because it is proven to
        produce correct token shapes from the length-2H window build_obs() emits. (The
        production ``predict_action`` eval branch forms the image grid differently and
        expects a history-only window; we replicate the offline path to reuse the
        window contract we already match.) The force ÷ft_scale pre-scale is applied
        here (predict_action would have done it internally), and force_pred ×ft_scale.

        Returns a dict of numpy arrays:
          * wire_pos (S,9), wire_orn (S,12): absolute vr_3point targets [L,R,H] to
            drip-feed to the low-level (right+head = ready-pose home each step).
          * tcp_pos (S,3), tcp_rot6d (S,6): the LEFT UMI-TCP absolute chunk (for viz).
          * force_pred (H,6) physical N/N·m, or (0,6) if the model has no force head.
          * grip_m (S,): predicted LEFT gripper width (metres) — reported, NOT commanded.
          * end_pos (3,), end_rot6d (6,): the LAST step's UMI-TCP pose — pass to
            :meth:`commit_anchor` once the chunk is executed so the next chunk branches
            from where this one ended. NOT applied here (a preview must not move it).
        """
        import torch
        (pose_to_mat, mat_to_pose, _mat_to_pose10d, pose10d_to_mat,
         convert_pose_mat_rep) = _pose_helpers()
        from fuji.utils.rot6d import (  # type: ignore
            rot6d_to_rotmat, rotmat_to_rotvec, rotvec_to_rotmat, rotmat_to_rot6d)
        from unified_video_action.utils.data_utils import resize_image, unnormalize_future_action
        from unified_video_action.eval.eval import prepare_data_predict_action

        def rotvec_to_rot6d(rv):
            return rotmat_to_rot6d(rotvec_to_rotmat(np.asarray(rv, dtype=np.float64)))

        obs = self.build_obs()   # flat obs dict, length-2H windows
        cfg = self.be.cfg
        # Force ÷ ft_scale (the dataset/predict_action input-side pre-scale). build_obs
        # emits PHYSICAL wrench; scale here so the encoder sees the training distribution.
        if self.be.use_force and "robot0_ft_wrench" in obs:
            w = obs["robot0_ft_wrench"]
            obs["robot0_ft_wrench"] = w / torch.as_tensor(
                self.be.ft_scale, device=w.device, dtype=w.dtype)

        x = {"obs": obs}
        x = resize_image(cfg, x)                       # camera0_rgb -> image, 224->256
        B, T, C, H, W = x["obs"]["image"].size()
        # action tensor: only shape/T matters for policy_model (diffusion generates it).
        actions = torch.zeros((B, self.be.action_horizon, self.be.action_dim),
                              device=self.be.device, dtype=torch.float32)
        (_x, _real, _lat, c, text_latents, hist_traj, traj, prop) = prepare_data_predict_action(
            cfg, x, actions, self.be.policy, T, self.be.device, language_goal=None)
        amp = cfg.model.policy.autoregressive_model_params
        with torch.no_grad():
            _z, act_out, force_out = self.be.policy.model.sample_tokens(
                bsz=B, cond=c, text_latents=text_latents,
                num_iter=amp.num_iter, cfg=amp.cfg, cfg_schedule=amp.cfg_schedule,
                temperature=amp.temperature, history_nactions=hist_traj,
                nactions=traj, proprioception_input=prop, task_mode="policy_model")
        act_out = unnormalize_future_action(
            normalizer=self.be.policy.normalizer,
            normalizer_type=self.be.policy.normalizer_type, actions=act_out)
        action_full = act_out[0].detach().float().cpu().numpy()  # (H,10) relative
        if force_out is not None:
            force_pred = (force_out[0].detach().float().cpu().numpy()
                          * self.be.ft_scale)  # (H,6) physical
        else:
            force_pred = np.zeros((0, 6), dtype=np.float64)
        # Execute the first n_action_steps of the H-step chunk (deploy replan cadence).
        n_exec = n_action_steps or self.be.n_action_steps
        action = action_full[:max(1, int(n_exec))]  # (S,10) relative

        with self._lock:
            anchor_pos = self._anchor_pos.copy()
            anchor_rot6d = self._anchor_rot6d.copy()

        # Un-relativize each step against the SAME chunk anchor (pelvis-frame UMI-TCP),
        # exactly as get_real_umi_action / _chunk_to_absolute: anchor_mat @ action_mat.
        anchor_rotvec = rotmat_to_rotvec(rot6d_to_rotmat(anchor_rot6d))
        anchor_mat = pose_to_mat(np.concatenate([anchor_pos, anchor_rotvec]))
        action_pose10d = action[..., :9]
        grip_norm_or_m = action[..., 9]  # gripper channel — trained in METRES (identity norm)
        action_mat = convert_pose_mat_rep(
            pose10d_to_mat(action_pose10d), base_pose_mat=anchor_mat,
            pose_rep=self.be.action_pose_repr, backward=True)                 # (S,4,4)
        abs_pose = mat_to_pose(action_mat)                                    # (S,6) pos+rotvec
        tcp_pos = np.asarray(abs_pose[..., :3], dtype=np.float64)             # (S,3)
        tcp_rot6d = rotvec_to_rot6d(np.asarray(abs_pose[..., 3:6], dtype=np.float64))  # (S,6)
        grip_m = np.clip(grip_norm_or_m, 0.0, 0.12).astype(np.float64)       # metres, sane clamp
        S = tcp_pos.shape[0]

        # Build the vr_3point wire per step: LEFT from the chunk (TCP→contact point),
        # RIGHT + HEAD held at ready-pose home.
        wire_pos = np.zeros((S, 9), dtype=np.float64)
        wire_orn = np.zeros((S, 12), dtype=np.float64)
        for i in range(S):
            lc_pos, lc_quat = self._tcp_to_wire(tcp_pos[i], tcp_rot6d[i])
            wire_pos[i, 0:3] = lc_pos
            wire_pos[i, 3:6] = self._home_pos_wire[1]   # right hand held
            wire_pos[i, 6:9] = self._home_pos_wire[2]   # head held
            wire_orn[i, 0:4] = lc_quat
            wire_orn[i, 4:8] = self._home_quat_xyzw[1]
            wire_orn[i, 8:12] = self._home_quat_xyzw[2]

        # NOTE: the running anchor is NOT advanced here. The caller advances it via
        # commit_anchor(end_pos, end_rot6d) ONLY after the chunk's wire has really been
        # dripped to the low-level, so a preview (predict-only) leaves the model frame
        # untouched and can be repeated. Return the chunk-end pose for that commit.
        return dict(
            wire_pos=wire_pos, wire_orn=wire_orn,
            tcp_pos=tcp_pos, tcp_rot6d=tcp_rot6d,
            force_pred=force_pred.astype(np.float64),
            grip_m=grip_m,
            end_pos=tcp_pos[-1].copy(), end_rot6d=tcp_rot6d[-1].copy(),
        )


# ===========================================================================
# Model backend (uvaforce env) — deploy load path. Mirrors the OFFLINE
# muse_force_infer_server.Backend load/guard logic, but builds NO dataset (obs is
# assembled live) and drives predict_action (eval=True) instead of the offline
# prepare_data_predict_action window.
# ===========================================================================
class Backend:
    """Loads the UVA force policy from a (slim) checkpoint and exposes the handful
    of attributes LivePlanner needs (policy, device, ft_scale, obs/action pose repr,
    obs history length, image-frame count, img_indices). Reuses the SAME defensive
    weight-load + VAE + ft_scale checks as the offline rollout backend so a broken
    checkpoint fails loudly rather than serving a partially-loaded model."""

    def __init__(self, checkpoint: str, device: str, model_size: str, force_fusion: str | None):
        if checkpoint:
            checkpoint = os.path.abspath(os.path.expanduser(checkpoint))
        sys.path.insert(0, _UVA)
        os.chdir(_UVA)

        import torch
        from omegaconf import OmegaConf, open_dict

        self.torch = torch
        OmegaConf.register_new_resolver("eval", eval, replace=True)

        if device.startswith("cuda") and not torch.cuda.is_available():
            _log("CUDA unavailable; falling back to cpu.")
            device = "cpu"
        self.device = device
        if not checkpoint:
            raise SystemExit("[muse-serve] --checkpoint is REQUIRED for the live server "
                             "(a random-init model would drive the arm with garbage).")

        from unified_video_action.workspace.train_unified_video_action_workspace import (
            TrainUnifiedVideoActionWorkspace,
        )
        import pathlib

        ckpt = pathlib.Path(checkpoint)
        payload = torch.load(ckpt.open("rb"), map_location="cpu", weights_only=False)
        cfg = payload["cfg"]
        with open_dict(cfg):
            cfg.task.dataset.dataloader_cfg.batch_size = 1
            cfg.task.dataset.dataloader_cfg.num_workers = 0
            if force_fusion:
                cfg.model.policy.force_fusion = force_fusion
        OmegaConf.resolve(cfg)
        self.cfg = cfg
        ws = TrainUnifiedVideoActionWorkspace(cfg)
        ws.load_payload(payload, strict=False)  # strict=False required for the slim ckpt
        policy = ws.ema_model if ws.ema_model is not None else ws.model
        self._assert_weights_loaded(policy, payload)
        self._assert_vae_loaded(cfg)
        policy.to(device).eval()
        self.policy = policy

        # ft_scale: read from cfg.task; the ÷/× contract lives in infer_chunk. Assert the
        # three config nodes (task / model.policy / task.dataset) agree — all are
        # ${task.ft_scale} interpolations, but a stray override of one alone would desync
        # the force units the encoder sees from what the head was scored against.
        # (Consistency-audit LOW: the comment here previously PROMISED this check but the
        # code never ran it; ported from muse_force_infer_server._assert_ft_scale_consistent.)
        self.ft_scale = np.asarray(list(cfg.task.get("ft_scale", [1.0] * 6)), dtype=np.float64)
        self._assert_ft_scale_consistent(cfg)
        self.predict_force = bool(cfg.model.policy.get("predict_force", False))
        self.use_force = bool(cfg.model.policy.get("use_force", False))
        self.obs_pose_repr = str(cfg.task.pose_repr.obs_pose_repr)
        self.action_pose_repr = str(cfg.task.pose_repr.action_pose_repr)
        self.action_dim = int(cfg.task.shape_meta.action.shape[0])  # 10
        # action_horizon=32 (16 history + 16 future); obs history + force history = 16.
        self.action_horizon = int(cfg.task.action_horizon)
        self.obs_history_len = self.action_horizon // 2  # 16
        # n_action_steps: how many chunk steps predict_action returns (deploy replan
        # cadence). Read off the loaded policy (its __init__ param).
        self.n_action_steps = int(getattr(policy, "n_action_steps", 8))

        # down_sample_steps MUST be 1 (the recorded/predicted frame mapping assumes it).
        ds = int(cfg.task.get("obs_down_sample_steps", 1))
        if ds != 1:
            raise SystemExit(f"[muse-serve] obs_down_sample_steps={ds} != 1 unsupported.")

        # image frame count + img_indices = include_indices/ds + (obs_history_len-1),
        # matching data_utils.process_data's umi branch. camera0_rgb offsets are
        # range(-12,17,4) → 8 frames; img_indices = [3,7,11,15,19,23,27,31].
        self.num_img_frames = 8
        base = np.arange(-12, 17, 4, dtype=np.int64)  # [-12,-8,-4,0,4,8,12,16]
        self.img_indices = (base // 1 + (self.obs_history_len - 1)).astype(np.int64)

    @staticmethod
    def _assert_weights_loaded(policy, payload) -> None:
        ckpt_keys = {k.replace("module.", "") for k in payload["state_dicts"]["ema_model"].keys()}
        model_keys = set(policy.state_dict().keys())
        unexpected = sorted(ckpt_keys - model_keys)
        missing = sorted(k for k in (model_keys - ckpt_keys) if not k.startswith("vae_model"))
        if unexpected or missing:
            raise RuntimeError(
                "checkpoint did not load cleanly — "
                f"{len(unexpected)} unexpected {unexpected[:5]}, "
                f"{len(missing)} non-vae MISSING {missing[:5]}. Refusing to serve.")
        _log(f"weight-load OK: {len(ckpt_keys)} ckpt tensors, 0 unexpected, 0 non-vae missing.")

    @staticmethod
    def _assert_ft_scale_consistent(cfg) -> None:
        """Assert the three ft_scale config nodes agree (units == metric scale).

        Backend reads ``cfg.task.ft_scale``; the encoder path (eval) reads
        ``cfg.model.policy.ft_scale``; the dataset divides by ``cfg.task.dataset.ft_scale``.
        All are ``${task.ft_scale}`` interpolations for this config family, but an override
        of one alone would desync the force units the head was scored against from what the
        live encoder sees. Ported verbatim from muse_force_infer_server."""
        def _get(node, key):
            try:
                v = node.get(key, None)
            except Exception:  # noqa: BLE001
                return None
            return None if v is None else [round(float(x), 6) for x in v]
        task = _get(cfg.task, "ft_scale")
        pol = _get(cfg.model.policy, "ft_scale")
        dset = _get(cfg.task.dataset, "ft_scale") if "dataset" in cfg.task else None
        present = [v for v in (task, pol, dset) if v is not None]
        if present and any(v != present[0] for v in present):
            raise ValueError(
                "ft_scale mismatch across config nodes — "
                f"task={task}, model.policy={pol}, task.dataset={dset}. Physical force units "
                "would disagree with the metric's scale. Refusing to serve.")

    def _assert_vae_loaded(self, cfg) -> None:
        try:
            path = cfg.model.policy.vae_model_params.get("autoencoder_path", None)
        except Exception:  # noqa: BLE001
            return
        if not path:
            return
        resolved = path if os.path.isabs(path) else os.path.join(_UVA, path)
        if not os.path.exists(resolved):
            raise FileNotFoundError(
                f"frozen VAE weights not found at {resolved!r}; the slim ckpt relies on it "
                "(else a random VAE → every action is garbage). Run the RUNBOOK download.")


# ===========================================================================
# Self-check (like collection_data.sh): gripper reads (+ optional open/close),
# F/T streams, camera delivers frames — BEFORE any inference.
# ===========================================================================
def _self_check(subs: dict, *, exercise_gripper: bool, allow_3dof: bool = False,
                timeout_s: float = 8.0) -> dict:
    """Confirm each live sensor is delivering. Returns a status dict; a hard failure
    (no F/T, no camera, no gripper) raises. The gripper 'exercise' (open→close) is
    OPTIONAL (default off — manual-only bring-up); when on it commands open then close
    and confirms the encoder moved. Every probe is a passive WS read except that."""
    report = {}

    def _wait_frames(sub, need=2):
        # Per-probe deadline (not a single shared one): a slow-to-start F/T bridge must
        # not eat the whole budget and starve the camera/gripper probes into false fails.
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            _msg, _w, n = sub.latest()
            if n >= need:
                return True
            time.sleep(0.1)
        return False

    # F/T (required)
    ft = subs.get("force")
    ok_ft = ft is not None and _wait_frames(ft)
    msg, _w, _n = ft.latest() if ft else (None, 0, 0)
    report["force"] = {"ok": bool(ok_ft), "has_left": bool(msg and msg.get("left") is not None) if msg else False,
                       "has_left_pipe": bool(msg and msg.get("left_pipe") is not None) if msg else False}
    if not ok_ft:
        raise SystemExit("[muse-serve] SELF-CHECK FAILED: no F/T frames on :8771 "
                         "(is 2_ft_bridge.sh running, ideally with --pipeline?).")
    if not report["force"]["has_left_pipe"]:
        # CONSISTENCY-AUDIT HIGH: training force is ALWAYS 6-DoF gravity-only + tared. Without
        # the pipe key the planner falls back to 3-DoF with torque ZERO-PADDED (_pick_hand_force)
        # — an out-of-distribution wrench the model never saw. HARD-FAIL by default so a live
        # humanoid never runs on OOD force; --allow-3dof-force opts into it for bring-up only.
        if allow_3dof:
            _log("SELF-CHECK WARNING: no left_pipe (6-DoF) key — running with 3-DoF force "
                 "(torque zero-padded) because --allow-3dof-force is set. This is OUT-OF-"
                 "DISTRIBUTION vs training (always 6-DoF); expect degraded force behavior.")
        else:
            raise SystemExit(
                "[muse-serve] SELF-CHECK FAILED: F/T stream has no 'left_pipe' (6-DoF) key — the "
                "model was trained on 6-DoF gravity-only force and the 3-DoF fallback zero-pads "
                "torque (out-of-distribution). Start 2_ft_bridge.sh WITH --pipeline (or "
                "FT_PIPELINE=1). To force a 3-DoF bring-up anyway, pass --allow-3dof-force.")

    # Camera (required)
    cam = subs.get("camera")
    ok_cam = cam is not None and _wait_frames(cam)
    report["camera"] = {"ok": bool(ok_cam)}
    if not ok_cam:
        raise SystemExit("[muse-serve] SELF-CHECK FAILED: no camera frames on :8773 "
                         "(is the realsense bridge running?).")

    # Gripper (required to READ; optional to EXERCISE)
    grip = subs.get("gripper")
    ok_grip = grip is not None and _wait_frames(grip)
    gmsg, _w, _n = grip.latest() if grip else (None, 0, 0)
    report["gripper"] = {"ok": bool(ok_grip),
                         "actual": (gmsg.get("actual") if gmsg else None),
                         "open": (gmsg.get("open") if gmsg else None),
                         "closed": (gmsg.get("closed") if gmsg else None)}
    if not ok_grip:
        raise SystemExit("[muse-serve] SELF-CHECK FAILED: no gripper frames on :8784 "
                         "(is the gripper bridge running?).")
    if exercise_gripper and grip is not None:
        _log("SELF-CHECK: exercising gripper (open→close)…")
        a0 = gmsg.get("actual", 0.0) if gmsg else 0.0
        grip.send({"cmd": "open"})
        time.sleep(1.2)
        m1, _, _ = grip.latest()
        a1 = m1.get("actual", a0) if m1 else a0
        grip.send({"cmd": "close"})
        time.sleep(1.2)
        m2, _, _ = grip.latest()
        a2 = m2.get("actual", a1) if m2 else a1
        moved = abs(a1 - a0) > 0.05 or abs(a2 - a1) > 0.05
        report["gripper"]["exercised"] = bool(moved)
        if not moved:
            _log("SELF-CHECK WARNING: gripper did not visibly move on open/close "
                 "(encoder Δ < 0.05 rad) — check the gripper bridge / self-test.")
    _log(f"SELF-CHECK OK: force={report['force']} camera={report['camera']} gripper={report['gripper']}")
    return report


# ===========================================================================
# Control loop + protocol server.
# ===========================================================================
def _pick_hand_force(msg: dict, hand: str, require_6dof: bool = False) -> np.ndarray | None:
    """Prefer the 6-DoF pipe key. If ``require_6dof`` (the default deploy contract — the
    model was trained on 6-DoF gravity-only force), return None when the 6-DoF pipe is
    ABSENT rather than silently zero-padding torque into an out-of-distribution 3-DoF
    vector (consistency review MEDIUM: the self-check gate was startup-only; a mid-run pipe
    dropout must not silently feed OOD force). When ``require_6dof`` is False (an explicit
    --allow-3dof-force bring-up) the 3-DoF torque-zeroed fallback is allowed."""
    if msg is None:
        return None
    pipe = msg.get(f"{hand}_pipe")
    if pipe is not None and len(pipe) >= 6:
        return np.asarray(pipe[:6], dtype=np.float64)
    if require_6dof:
        return None
    f3 = msg.get(hand)
    if f3 is not None and len(f3) >= 3:
        return np.concatenate([np.asarray(f3[:3], dtype=np.float64), np.zeros(3)])
    return None


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--checkpoint", required=True, help="UVA force policy checkpoint (.slim.pt/.ckpt)")
    p.add_argument("--device", default="cuda")
    p.add_argument("--model-size", default="mar_base")
    p.add_argument("--force-fusion", default=None, choices=[None, "early", "late", "both"])
    p.add_argument("--hand", default="left", choices=["left"],
                   help="which hand the F/T stream + model correspond to. LEFT-only: the "
                        "anchor seed, contact-point lever, and vr_3point LEFT slot are all "
                        "left-canonical, so 'right' is intentionally NOT offered (it would "
                        "drive right-sensor force into the left wire slot).")
    p.add_argument("--force-ws", default="ws://127.0.0.1:8771")
    p.add_argument("--camera-ws", default="ws://127.0.0.1:8773")
    p.add_argument("--gripper-ws", default="ws://127.0.0.1:8784")
    p.add_argument("--s0-ws-host", default="127.0.0.1")
    p.add_argument("--s0-ws-port", type=int, default=8770)
    p.add_argument("--obs-hz", type=float, default=10.0,
                   help="ring-buffer ingest rate (Hz) for proprio/force — should match the "
                        "training camera master-grid rate (the FUJI zarr median is ~10.3 Hz; "
                        "see umi_force.yaml). Proprio/force are stride-1 on this grid.")
    p.add_argument("--step-hz", type=float, default=10.0,
                   help="rate at which the executed chunk ADVANCES through its poses (Hz). MUST "
                        "match the training action grid (~10.3 Hz, one action step == one grid "
                        "frame at action down_sample_steps=1). Distinct from --drip-hz: the wire "
                        "is refreshed at drip-hz for the low-level's freshness gate, but the "
                        "trajectory only steps to the next pose every 1/step-hz. Consistency-audit "
                        "HIGH: dripping one pose per 30 Hz tick replayed the chunk ~3x too fast.")
    p.add_argument("--drip-hz", type=float, default=30.0,
                   help="wire REFRESH rate to S0 (Hz) — re-sends the current chunk pose to keep "
                        "the :8770 stream inside S0Driver's freshness window. Does NOT set the "
                        "trajectory playback speed (that is --step-hz).")
    p.add_argument("--img-stride", type=int, default=4,
                   help="push a camera frame into the 8-slot image ring every N ingest ticks, "
                        "so the buffered frames are N grid-steps apart — matching training "
                        "(camera0_rgb include_indices range(-12,17,4), stride 4). At obs-hz≈grid "
                        "and stride 4 the 8 frames span ~2.8 s of history like training. "
                        "Consistency-audit HIGH: pushing every tick compressed it ~4x.")
    p.add_argument("--allow-3dof-force", action="store_true",
                   help="permit running with only the 3-DoF F/T stream (torque zero-padded) when "
                        "--pipeline is off. Default OFF: the model was trained on 6-DoF gravity-"
                        "only force, so a torque-zeroed 3-DoF vector is out-of-distribution — the "
                        "self-check HARD-FAILS without the 6-DoF pipe unless this is set.")
    p.add_argument("--exercise-gripper", action="store_true",
                   help="open/close the gripper during self-check (default: read-only)")
    p.add_argument("--command-gripper", action="store_true",
                   help="autonomously DRIVE the LEFT gripper from the model's predicted "
                        "per-step width (position command on :8784). Default OFF = read-only "
                        "proprioception (manual-only bring-up); when off the gripper path is "
                        "byte-identical to before.")
    p.add_argument("--no-selfcheck", action="store_true")
    args = p.parse_args(argv if argv is not None else sys.argv[1:])

    be = Backend(args.checkpoint, args.device, args.model_size, args.force_fusion)

    # home_anchor.json from the deploy tree (LEFT-canonical 3pt ready pose).
    ha_path = os.path.join(
        _FUJI_ROOT, "third_party", "holosoma_extensions", "src", "extensions", "wbt_wrappers",
        "wbt_realtime_eval", "wbt_realtime_eval", "vive", "home_anchor.json")
    home_anchor = json.load(open(ha_path))
    planner = LivePlanner(be, home_anchor, args.obs_hz)

    # Live sensor subscribers (read-only).
    subs = {
        "force": _WSSubscriber(args.force_ws, "force"),
        "camera": _WSSubscriber(args.camera_ws, "camera"),
        "gripper": _WSSubscriber(args.gripper_ws, "gripper"),
    }
    for s in subs.values():
        s.start()

    calib = GripperCalib(width_open_m=0.095, width_closed_m=0.0)

    # S0 driver (impersonate the VIVE bridge on :8770) — reuse fuji's exact class.
    sys.path.insert(0, _FUJI_ROOT)
    from fuji.client.s0_driver import S0Driver
    s0 = S0Driver(host=args.s0_ws_host, port=args.s0_ws_port)
    s0.start()

    # Self-check before any inference (report to frontend either way).
    if not args.no_selfcheck:
        try:
            rep = _self_check(subs, exercise_gripper=args.exercise_gripper,
                              allow_3dof=args.allow_3dof_force)
            _emit("selfcheck", ok=np.asarray(1), report=np.asarray(json.dumps(rep)))
        except SystemExit as e:
            _emit("selfcheck", ok=np.asarray(0), report=np.asarray(json.dumps({"error": str(e)})))
            raise

    _emit("ready", info=np.asarray(json.dumps({
        "predict_force": be.predict_force, "action_pose_repr": be.action_pose_repr,
        "obs_pose_repr": be.obs_pose_repr, "obs_history_len": be.obs_history_len,
        "n_action_steps": be.n_action_steps, "ft_scale": be.ft_scale.tolist(),
        "hand": args.hand, "command_gripper": bool(args.command_gripper),
    })))
    _log(f"ready: predict_force={be.predict_force} H={be.obs_history_len} "
         f"n_action_steps={be.n_action_steps} hand={args.hand}")

    # --- background ingest thread: keep the ring buffers fresh at obs_hz ---
    # No lock at this level on purpose: the ring buffers this thread feeds are guarded by
    # the planner's OWN self._lock, i.e. the lock lives with the data it protects, and the
    # cross-thread flags here are Events. A one-shot run is not a flag either -- it goes
    # through the stdin command protocol, whose `staged` dict only that thread touches.
    stop = threading.Event()
    run_continuous = threading.Event()
    last_status = {"lowlevel_online": False}
    # Runtime gripper-actuation enable: an operator kill-switch for the physical jaw,
    # separate from --command-gripper (the launch default). A threading.Event so the
    # drip loop reads it lock-free; seeded from the launch flag. The viser "Command
    # gripper" checkbox toggles it live via grip_on/grip_off (finding #1 asked for an
    # explicit operator enable in the UI, on top of the num_clients liveness gate).
    grip_enabled = threading.Event()
    if args.command_gripper:
        grip_enabled.set()
    # Latches True the first time we ACTUALLY command the jaw, so shutdown-release fires
    # on the effective runtime state, not just the launch flag (review fix: a jaw enabled
    # at runtime must still be released even if launched read-only). {"v":...} for closure.
    jaw_ever_commanded = {"v": False}

    def _ingest_loop():
        # Proprio + force are pushed EVERY tick (stride-1 on the ~10 Hz training grid).
        # Images are pushed every args.img_stride ticks so the 8-slot image ring spans
        # stride-4 grid steps like training (camera0_rgb include_indices range(-12,17,4)),
        # NOT one image per tick. CONSISTENCY-AUDIT HIGH: at obs_hz≈grid and a new camera
        # frame every tick, pushing per-tick made the 8 buffered frames ~100 ms apart
        # (~0.8 s history) vs the trained ~388 ms apart (~2.8 s) → ~4x temporally-compressed
        # visual conditioning and a violated stride-4 img:proprio cadence (img_indices).
        period = 1.0 / max(args.obs_hz, 1.0)
        stride = max(int(args.img_stride), 1)
        require_6dof = not args.allow_3dof_force
        ft_dropout_logged = False
        tick = 0
        while not stop.is_set():
            t0 = time.time()
            fmsg, _fw, _fn = subs["force"].latest()
            fw6 = _pick_hand_force(fmsg, args.hand, require_6dof=require_6dof)
            if fw6 is not None:
                planner.push_force(fw6)
                ft_dropout_logged = False
            elif fmsg is not None and require_6dof:
                # A frame IS arriving but has no 6-DoF pipe key (mid-run --pipeline dropout).
                # Do NOT push an OOD torque-zeroed vector (require_6dof made _pick return None
                # → the model keeps its last-good 6-DoF history). STOP any continuous run so
                # the arm isn't driven on stale force, and surface it once (review MEDIUM).
                if run_continuous.is_set():
                    run_continuous.clear()
                    _emit("error", msg=np.asarray(
                        "F/T 6-DoF pipe dropped mid-run (no left_pipe key) — continuous "
                        "STOPPED; the model would otherwise run on stale/OOD force. Check "
                        "2_ft_bridge.sh --pipeline."))
                if not ft_dropout_logged:
                    _log("INGEST: F/T frame lacks the 6-DoF pipe key — freezing force on the "
                         "last good sample (not zero-padding). Restart the bridge with --pipeline.")
                    ft_dropout_logged = True
            gmsg, _gw, _gn = subs["gripper"].latest()
            if gmsg is not None and gmsg.get("actual") is not None:
                width = calib.to_width_m(
                    float(gmsg["actual"]),
                    float(gmsg.get("open", 4.9)), float(gmsg.get("closed", 0.0)))
                planner.push_gripper(width)
            # Image: sample the LATEST frame at each stride boundary (edge-hold, like the
            # dataset). Camera publishes ~15 Hz > grid/stride (~2.6 Hz), so a fresh frame
            # is always available at the boundary; no new-frame-number dedup needed.
            if tick % stride == 0:
                cmsg, _cw, _cn = subs["camera"].latest()
                if cmsg is not None and cmsg.get("jpeg_b64"):
                    img = _decode_jpeg_b64(cmsg["jpeg_b64"])
                    if img is not None:
                        planner.push_image(_resize_center_crop_224(img))
            # log the current anchor as the last-target proprio sample
            planner.push_proprio_from_anchor()
            tick += 1
            dt = time.time() - t0
            if dt < period:
                time.sleep(period - dt)

    threading.Thread(target=_ingest_loop, daemon=True, name="ingest").start()

    # IDLE HEARTBEAT: continuously re-publish the ready-pose wire (LEFT at the current
    # anchor, right+head at home) at drip_hz whenever a chunk is NOT actively dripping.
    # WHY it is REQUIRED (whole-system review finding #2): S0Driver suppresses broadcast
    # while _latest is None, and ViveLiveSource's pre-arm pose gate refuses to arm unless
    # it sees a LIVE frame matching the robot's held pose. Without this heartbeat the arm
    # could never be armed (nothing published before the first chunk), AND during
    # continuous-mode inference (the diffusion sample can exceed the 0.5 s freshness
    # window) the low-level would stale-gate → hold-glide stutter every chunk. Publishing
    # the anchor pose keeps the wire fresh at exactly where the arm already is (zero-jump)
    # both before the first chunk and between chunks.
    dripping = threading.Event()
    # `_last_wire` is the LAST wire actually sent to S0 (pos[9], orn[12]). The heartbeat
    # republishes exactly this — NOT a recomputed home_wire() — so it can never publish a
    # pose the arm hasn't been commanded to (infer_chunk advances the internal anchor to
    # the chunk end BEFORE the drip feeds the intermediate steps; recomputing from the
    # anchor mid-flight would jump the wire to the endpoint). Seeded to the home pose so
    # the pre-arm pose gate sees the ready pose before the first chunk.
    _wire_lock = threading.Lock()
    _hp0, _ho0 = planner.home_wire()
    last_wire = {"pos": _hp0, "orn": _ho0}

    def _set_last_wire(pos, orn):
        with _wire_lock:
            last_wire["pos"] = np.asarray(pos, dtype=np.float64)
            last_wire["orn"] = np.asarray(orn, dtype=np.float64)

    def _heartbeat_loop():
        period = 1.0 / max(args.drip_hz, 1.0)
        while not stop.is_set():
            if not dripping.is_set():
                try:
                    with _wire_lock:
                        hp = last_wire["pos"]
                        ho = last_wire["orn"]
                    s0.send(hp, ho)
                except Exception as e:  # noqa: BLE001
                    _log(f"heartbeat error: {e!r}")
            time.sleep(period)

    threading.Thread(target=_heartbeat_loop, daemon=True, name="heartbeat").start()

    # A PREDICTED-but-not-yet-executed chunk, staged by `predict_once` for a later
    # `run_staged`. This is the whole point of the predict→preview→execute split
    # (SAFETY: the operator sees the rollout + predicted force BEFORE any real motion):
    #   * predict_once  → infer, emit a PREVIEW chunk, stash it here. NO wire, NO anchor
    #                     advance — the robot does not move and the model frame is intact,
    #                     so predict can be repeated freely.
    #   * run_staged    → drip THIS staged chunk to the low-level, then commit the anchor.
    # Only the stdin thread touches `staged` (predict_once / run_staged / reset / stop are
    # all handled there and are mutually exclusive with a running continuous loop).
    staged = {"out": None}

    def _emit_chunk(out, *, preview: bool) -> None:
        _emit("chunk",
              tcp_pos=out["tcp_pos"], tcp_rot6d=out["tcp_rot6d"],
              force_pred=out["force_pred"], grip_m=out["grip_m"],
              preview=np.asarray(1 if preview else 0),
              lowlevel_online=np.asarray(1 if s0.num_clients > 0 else 0))

    def _predict_chunk():
        """Inference ONLY — emit a preview chunk (rollout + predicted force for the viz),
        stash it for a later run_staged. Does NOT touch the wire or advance the anchor, so
        the robot stays put and the preview is repeatable."""
        out = planner.infer_chunk()
        staged["out"] = out
        _emit_chunk(out, preview=True)
        return out

    def _execute_chunk(out):
        """Drip an ALREADY-PREDICTED chunk to the low-level, then commit the anchor to its
        last step. Used by run_staged (operator-gated) and the continuous loop. Claims the
        wire only for the drip; the heartbeat keeps publishing the LAST-SENT pose (= where
        the arm is) during the diffusion sample so the low-level never stale-gates."""
        wire_pos = out["wire_pos"]
        wire_orn = out["wire_orn"]
        grip_m = out["grip_m"]                          # (S,) predicted LEFT width, metres
        dripping.set()
        try:
            # CADENCE (consistency-audit HIGH): the chunk must ADVANCE at the training action
            # grid rate (--step-hz ≈ 10.3 Hz, one pose per ~97 ms — the rate the model was
            # trained to produce), NOT at the wire refresh rate. We hold each pose for
            # step_dt and RE-SEND it at drip-hz within that hold so S0Driver's 0.5 s freshness
            # gate stays satisfied (the low-level sees a fresh frame every ~33 ms) while the
            # commanded trajectory plays at the correct ~10 Hz speed. Previously we advanced a
            # new pose every 33 ms → the 8-step chunk (~0.8 s of trained motion) executed in
            # ~0.26 s = ~3× EE velocity, an out-of-distribution (and unsafe) speed.
            step_dt = 1.0 / max(args.step_hz, 1.0)
            drip_dt = 1.0 / max(args.drip_hz, 1.0)
            for i in range(wire_pos.shape[0]):
                if stop.is_set():
                    break
                pos_i = wire_pos[i].astype(np.float64)
                orn_i = wire_orn[i].astype(np.float64)
                _set_last_wire(pos_i, orn_i)
                # LEFT gripper: command the model's per-step target WIDTH as a POSITION
                # (``target_frac`` in [0,1]) on the SAME bus the manual keyboard uses, in
                # lockstep with the pose. The gripper bridge LATCHES this target and ramps
                # to it at its own rate, so it also HOLDS between chunks (no re-send needed
                # from the heartbeat) — matching how the arm holds at last_wire. Gated by
                # --command-gripper so the manual-only path stays byte-identical when off.
                # SAFETY (review finding #1, HIGH): only actuate the physical jaw when the
                # low-level is CONNECTED (s0.num_clients>0) — the same connection the arm
                # wire needs (S0Driver only broadcasts to connected clients). NOTE: connected
                # is NOT the same as armed — in the connected-but-not-yet-armed window the arm
                # holds (pose gate not passed) while the jaw could still move, so per the
                # runbook the operator ARMS before Run. This gate still removes the worst case
                # (jaw actuating with NO low-level at all), keeping "Run" inert until the
                # robot is deliberately brought live.
                # SAFETY (finding #3): skip the send on a non-finite prediction rather than
                # letting a bare NaN reach the wire (json emits a bare `NaN` literal).
                if (grip_enabled.is_set() and s0.num_clients > 0
                        and grip_m.shape[0] > i and np.isfinite(grip_m[i])):
                    frac = calib.to_target_frac(float(grip_m[i]))
                    if np.isfinite(frac):
                        subs["gripper"].send({"target_frac": float(frac)})
                        jaw_ever_commanded["v"] = True   # arm the shutdown release
                # Hold this pose for step_dt, re-sending at drip-hz for the freshness gate.
                t_step0 = time.time()
                while True:
                    s0.send(pos_i, orn_i)
                    if stop.is_set():
                        break
                    elapsed = time.time() - t_step0
                    remaining = step_dt - elapsed
                    if remaining <= 0:
                        break
                    time.sleep(min(drip_dt, remaining))
        finally:
            dripping.clear()  # hand the wire back to the heartbeat (now at the last-fed step)
        # Commit the anchor to the LAST executed step so the next chunk branches from where
        # this one ended (only NOW — the chunk was really dripped). infer_chunk no longer
        # self-advances, so a preview never reaches here and never moves the frame.
        planner.commit_anchor(out["end_pos"], out["end_rot6d"])
        _emit_chunk(out, preview=False)
        return out

    def _run_one_chunk():
        """Continuous-mode step: predict + execute + commit, all in one (no operator gate
        between — the operator opted into continuous). Kept as one call so the continuous
        loop reads unchanged."""
        return _execute_chunk(planner.infer_chunk())

    # --- protocol request loop (stdin) + a lightweight status heartbeat ---
    def _status_loop():
        while not stop.is_set():
            online = s0.num_clients > 0
            if online != last_status["lowlevel_online"]:
                last_status["lowlevel_online"] = online
            # LIVE F/T readout: the wrench the planner FEEDS THE MODEL right now. Use the
            # SAME require_6dof contract as the ingest loop so the readout can't show a
            # healthy-looking 3-DoF |F| while the model is actually frozen on a 6-DoF pipe
            # dropout (review LOW). -1 magnitudes signal "no usable F/T frame" (bridge down,
            # or — in the default 6-DoF deploy — the pipe key dropped).
            fmsg, fw, _fn = subs["force"].latest()
            fw6 = _pick_hand_force(fmsg, args.hand, require_6dof=not args.allow_3dof_force)
            ft_age = (time.time() - fw) if fw else -1.0
            if fw6 is not None:
                f_n = float(np.linalg.norm(fw6[0:3]))
                t_n = float(np.linalg.norm(fw6[3:6]))
            else:
                f_n = t_n = -1.0
            _emit("status", lowlevel_online=np.asarray(1 if online else 0),
                  running=np.asarray(1 if run_continuous.is_set() else 0),
                  ft_force_n=np.asarray(f_n), ft_torque_n=np.asarray(t_n),
                  ft_age_s=np.asarray(float(ft_age)),
                  staged=np.asarray(1 if staged["out"] is not None else 0))
            time.sleep(0.5)

    threading.Thread(target=_status_loop, daemon=True, name="status").start()

    def _continuous_loop():
        while not stop.is_set():
            if run_continuous.is_set():
                try:
                    _run_one_chunk()
                except Exception as e:  # noqa: BLE001
                    import traceback
                    _log("continuous chunk error:\n" + traceback.format_exc())
                    _emit("error", msg=np.asarray(f"{e!r}"))
                    run_continuous.clear()
            else:
                time.sleep(0.05)

    threading.Thread(target=_continuous_loop, daemon=True, name="continuous").start()

    _log("serving: awaiting commands on stdin "
         "(predict_once / run_staged / run_once / run_continuous / stop / reset / quit)")
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError:
            continue
        cmd = req.get("cmd")
        if cmd == "quit":
            break
        elif cmd == "reset":
            planner.reset_start_pose()
            staged["out"] = None  # a plan predicted against the old start/history is stale
            # NOTE (consistency-audit LOW): training re-tares the F/T in FREE SPACE at the SAME
            # instant it latches the episode-start pose. We do NOT auto-tare here — reset may be
            # issued between chunks with the arm loaded/in contact, and taring then would bake a
            # contact-force bias into both zeros (the exact hazard the tare guard prevents). The
            # correct per-episode zero is the operator flow: raise the arm into free space →
            # ⊘ Tare F/T → ↺ Reset start pose. Remind, don't auto-fire.
            _log("RESET: episode-start pose re-latched. For a training-faithful per-episode force "
                 "zero, ⊘ Tare F/T in FREE SPACE before/around this reset (not auto-tared — the "
                 "arm may be loaded).")
            _emit("reset_done", ok=np.asarray(1))
        elif cmd == "tare":
            # Manual F/T re-tare from the viser page. Forward {"cmd":"tare"} to the F/T
            # bridge on :8771 (it re-zeroes BOTH the low-level's published force AND the
            # 6-DoF gravity-only planner pipe in lockstep — see force_bridge.py). MUST be
            # done in FREE SPACE (arm up, gripper touching nothing): a contact force present
            # at tare bakes a bias into every subsequent reading. This is the same command
            # the low-level's A=START burst and its `t` hotkey send; issuing it here lets the
            # operator re-tare once the arm is raised into free space, BEFORE arming.
            # GUARD (review finding): REFUSE a tare while a chunk is running/dripping — the
            # arm is then almost certainly loaded/in-contact, and taring on a contact force
            # bakes a permanent bias into BOTH force zeros. Tare only when idle + free.
            if run_continuous.is_set() or dripping.is_set():
                _log("TARE REFUSED: a chunk is running/dripping — stop first, raise the arm "
                     "into free space, then tare.")
                _emit("tare_done", ok=np.asarray(0))
            else:
                subs["force"].send({"cmd": "tare"})
                _log("TARE: forwarded {'cmd':'tare'} to the F/T bridge (:8771) — tare in FREE SPACE.")
                _emit("tare_done", ok=np.asarray(1))
        elif cmd == "predict_once":
            # SAFETY: PREVIEW only. Run inference, emit a preview chunk (rollout +
            # predicted force for the viz), stage it — but do NOT drip to the low-level
            # and do NOT advance the anchor. The robot does not move; repeat freely until
            # the operator is satisfied, then run_staged executes THIS staged plan.
            if run_continuous.is_set() or dripping.is_set():
                _log("predict_once ignored (a chunk is running/dripping)")
            else:
                try:
                    _predict_chunk()
                except Exception as e:  # noqa: BLE001
                    import traceback
                    _log("predict_once error:\n" + traceback.format_exc())
                    _emit("error", msg=np.asarray(f"{e!r}"))
        elif cmd == "run_staged":
            # Execute the LAST predicted chunk (the one the operator previewed) and clear
            # the stage. REFUSE if nothing is staged (must predict first) or if a chunk is
            # already running — so a stray click can't drip an empty/duplicate plan.
            if run_continuous.is_set() or dripping.is_set():
                # ALWAYS ack (ok=0) so the frontend's run_staged_pending latch clears — else
                # "Run this chunk" would stay disabled forever after a refused click.
                _log("run_staged ignored (a chunk is running/dripping)")
                _emit("run_staged_done", ok=np.asarray(0))
            elif staged["out"] is None:
                _log("run_staged REFUSED: nothing staged — Predict 1 chunk first.")
                _emit("run_staged_done", ok=np.asarray(0))
            else:
                out = staged["out"]
                staged["out"] = None   # consume BEFORE executing (no double-run on re-click)
                try:
                    _execute_chunk(out)
                    _emit("run_staged_done", ok=np.asarray(1))
                except Exception as e:  # noqa: BLE001
                    import traceback
                    _log("run_staged error:\n" + traceback.format_exc())
                    _emit("error", msg=np.asarray(f"{e!r}"))
                    # ALWAYS ack even on failure (review MEDIUM): without this the frontend
                    # run_staged_pending latch never clears and "Run this chunk" is stuck
                    # disabled for the rest of the session (e.g. low-level disconnects
                    # mid-drip → s0.send raises). The stage was consumed; arm holds last_wire.
                    _emit("run_staged_done", ok=np.asarray(0))
        elif cmd == "run_once":
            # Legacy one-shot: predict + execute together (kept for compatibility; the UI
            # now uses predict_once → run_staged). Still guarded on continuous/dripping.
            if run_continuous.is_set() or dripping.is_set():
                _log("run_once ignored (a chunk is running/dripping)")
            else:
                # Clear any staged preview FIRST: run_once advances the anchor to this
                # fresh chunk's end, so a plan staged earlier (un-relativized against the
                # OLD anchor) is now stale — leaving it would let a later run_staged drip
                # old-anchor poses from the new arm position (a discontinuous jump) and
                # regress the anchor. Same invalidation reset/run_continuous/run_staged do.
                staged["out"] = None
                try:
                    _run_one_chunk()
                except Exception as e:  # noqa: BLE001
                    import traceback
                    _log("run_once error:\n" + traceback.format_exc())
                    _emit("error", msg=np.asarray(f"{e!r}"))
        elif cmd == "run_continuous":
            staged["out"] = None  # a staged one-shot plan is irrelevant once continuous starts
            run_continuous.set()
        elif cmd == "stop":
            run_continuous.clear()
        elif cmd == "grip_on":
            # REFUSE runtime-enable when launched read-only (--command-gripper OFF): the
            # jaw path was never intended for this session. Without this guard a runtime
            # enable could latch the jaw yet the shutdown-release keyed off the launch flag
            # would skip it (review finding, HIGH). The frontend also disables the checkbox
            # in that case, so this is defense-in-depth.
            if not args.command_gripper:
                _log("grip_on REFUSED: launched read-only (--command-gripper off).")
                _emit("grip_state", enabled=np.asarray(0))
            else:
                grip_enabled.set()
                _log("gripper actuation ENABLED (operator toggle).")
                _emit("grip_state", enabled=np.asarray(1))
        elif cmd == "grip_off":
            grip_enabled.clear()
            _log("gripper actuation DISABLED (operator toggle) — jaw holds last target.")
            _emit("grip_state", enabled=np.asarray(0))

    stop.set()
    run_continuous.clear()
    # SAFETY (review finding #2): on shutdown, RELEASE the LEFT jaw to a known-safe OPEN
    # position — the gripper bridge is a separate long-lived process that otherwise holds
    # the LAST commanded target_frac indefinitely (could be a near-closed grip on a hand/
    # object). Fire the release iff the jaw was EVER commanded this session (runtime state,
    # not just the launch flag — a runtime grip_on with a read-only launch is caught above,
    # but this stays correct either way).
    if jaw_ever_commanded["v"]:
        try:
            # 1) QUIESCE the drip path first. stop.set() is cooperative and the drip loop
            #    can still fire ONE more target_frac after its top-of-step check, which
            #    would (a) be sent after our release, or (b) COALESCE over our release in
            #    the send queue — either way defeating it (review finding, HIGH). Wait for
            #    the current chunk's drip to finish before we touch the queue.
            for _ in range(200):                      # up to ~2 s
                if not dripping.is_set():
                    break
                time.sleep(0.01)
            # 2) Release EXCLUSIVELY: clear any stale queued target_frac under the lock,
            #    then append 1.0 so no trailing drip frame can overwrite it.
            g = subs["gripper"]
            with g._lock:                             # single writer now (drip quiesced)
                g._send_q = [json.dumps({"target_frac": 1.0})]
            # 3) Wait (bounded) for the subscriber to actually FLUSH it while connected,
            #    rather than a blind sleep that a disconnect/reconnect could outlast
            #    (review finding: release dropped if bridge mid-reconnect at quit).
            flushed = False
            for _ in range(200):                      # up to ~2 s
                with g._lock:
                    pending = bool(g._send_q)
                if not pending and g.connected:
                    flushed = True
                    break
                time.sleep(0.01)
            _log(f"shutdown: released LEFT gripper (target_frac=1.0, open); flushed={flushed}"
                 + ("" if flushed else " — bridge disconnected at quit; open the jaw manually"))
        except Exception as e:  # noqa: BLE001
            _log(f"shutdown gripper-release error: {e!r}")
    for s in subs.values():
        s.stop()
    s0.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
