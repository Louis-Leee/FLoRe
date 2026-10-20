#!/usr/bin/env python3
"""MuSe-on-UVA INFERENCE visualization in viser — recorded zarr vs UVA rollout,
with PREDICTED force overlaid on the recorded force.

The UVA-backbone sibling of the flow-matching ``scripts/rollout_viz.py``. Same
idea — load a converted UMI ``.zarr.zip`` episode, stop on ANY frame, and on a
"Rollout" click run the trained model on that frame to produce ONE action +
force chunk, then draw BOTH trajectories in one G1 scene — but the model here is
the **MuSe-on-UVA force policy** (``docs/RUNBOOK_muse_8gpu.md`` /
``docs/muse_on_uva_force_design.md``), which predicts the primary (LEFT) hand
plus a **future force/torque chunk**.

  * RECORDED zarr trajectory (left + right + head) as a BLUE GRADIENT (oldest
    dark -> newest light), driven through the SAME mink ``ThreePointIK`` the
    collection viz uses.
  * UVA ROLLOUT trajectory — the LEFT hand only (UVA-umi is single-robot,
    ``act_dim=10``; the right hand + head are NOT predicted) — in ORANGE,
    branching off the recorded trajectory at the chosen frame. Right hand + head
    are held at their recorded anchor pose so the humanoid still renders whole.
  * PREDICTED future wrench (green) overlaid on the RECORDED wrench (the usual
    x/y/z charts) so you can eyeball the force head against ground truth, in
    physical N / N·m.

WHY THIS IS A TWO-PROCESS TOOL (unlike rollout_viz.py). The UVA model runs only
in its own conda env (``uvaforce``: torch 2.7.1+cu128, hydra, diffusers, and NO
viser/mink), while viser + mink + mujoco live only in FLoRe's ``.venv`` (torch
2.13, and NO torch-2.7/hydra). The two environments cannot coexist in one
interpreter. So this front-end (FLoRe ``.venv``) owns the viser scene + the
ThreePointIK, and spawns ``scripts/muse_force_infer_server.py`` in the UVA env
as a child process; they talk over the child's stdin/stdout with a tiny
base64-npz line protocol (both envs have numpy; FLoRe's .venv has no zmq). The
launcher ``run_muse_force_rollout_viz.sh`` wires the two Pythons together.

ISOLATION: reads only already-committed files + the trained checkpoint. It never
writes zarr/checkpoints, never touches ``~/UMI_collection`` code (the mink IK +
yam URDF are read-only via PYTHONPATH), and — critically — shares NOTHING with
the flow-matching ``rollout_viz.py``: the two tools never import each other, so
they are the "same level, mutually non-interfering" pair the design calls for.
The IK/render plumbing is duplicated on purpose to keep that independence.

Run (run_muse_force_rollout_viz.sh sets both interpreters + PYTHONPATH):
    ./scripts/run_muse_force_rollout_viz.sh \
        --checkpoint third_party/unified_video_action/checkpoints/uva_umi_force/latest.ckpt
    # no checkpoint -> the backend random-inits (pipeline check only).
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
from pathlib import Path

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_FLORE_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
_UMI = os.path.join(_FLORE_ROOT, "third_party", "UMI_collection")
# Same read-only vendored IK + viser helpers rollout_viz.py uses. Added here as a
# fallback so the script runs when launched directly; the launcher sets them too.
for _p in (
    os.path.join(_UMI, "third_party", "humi_ik"),  # humi_data_collection.ikumi
    os.path.join(_UMI, "viz"),                       # g1_three_point_ik / g1_right_hand_ik
    os.path.join(_UMI, "replay"),                     # replay_common (Timeline, WrenchArrows, charts)
):
    if _p not in sys.path:
        sys.path.insert(0, _p)
if _FLORE_ROOT not in sys.path:
    sys.path.insert(0, _FLORE_ROOT)

import viser  # noqa: E402
from viser import transforms as vtf  # noqa: E402
from viser.extras import ViserUrdf  # noqa: E402

import replay_common as rc  # noqa: E402  (UMI_collection/replay: Timeline + helpers)

from flore.s1_policy.data.transforms import (  # noqa: E402
    gripper_norm_to_width,
    gripper_width_to_norm,
    rotvec_to_rot6d,
)
from flore.utils.rot6d import rot6d_to_rotmat, rotmat_to_quat_xyzw, rotmat_to_rot6d  # noqa: E402

_DEFAULT_YAM = os.path.join(_UMI, "robot_assets", "g1_yam", "g1_29dof_yam.urdf")
_JAW_OPEN_M = 0.0475   # URDF jaw prismatic upper limit (matches collection_viz / replay)
_GRASP_THRESH_M = 0.08  # UMI's hard-coded grasp threshold, for the readout label

# Recorded trajectory: BLUE gradient (dark early -> light late). Rollout: ORANGE.
_RECORDED_BLUE_DARK = np.array([25, 40, 110], dtype=np.float64)
_RECORDED_BLUE_LIGHT = np.array([120, 190, 255], dtype=np.float64)
_ROLLOUT_ORANGE = (245, 150, 40)
_PRED_FORCE_GREEN = "#31c48d"   # predicted-force chart series

# The IK roles a recorded 3pt zarr can drive, and the robot frame each role's
# trajectory line is READ FROM (mirrors g1_three_point_ik._ROLE_FRAMES): the
# polyline is drawn from the SOLVED site position so a line always sits on the
# rendered robot part.
_SITE_FRAMES = {
    "left": ("left_hand_site", "site"),
    "right": ("right_hand_site", "site"),
    "head": ("torso_link", "body"),
}
# Raw zarr keys for each recorded keypoint's absolute pose + gripper.
_KP_KEYS = {
    "left": ("robot0_eef_pos", "robot0_eef_rot_axis_angle", "robot0_gripper_width"),
    "right": ("robot1_eef_pos", "robot1_eef_rot_axis_angle", None),
    "head": ("head_pos", "head_rot_axis_angle", None),
}
_MSG_PREFIX = "MUSEVIZ"


# ===========================================================================
# Backend (UVA env) subprocess — the model half, spoken over a pipe.
# ===========================================================================
class InferBackend:
    """Spawn ``muse_force_infer_server.py`` in the UVA python and talk npz-lines.

    ``backend_python`` is the ``uvaforce`` interpreter (the launcher passes it);
    everything else mirrors the backend CLI. The handshake (``ready``) reports
    the resolved zarr path + episode layout so the frontend reads the SAME zarr.
    ``rollout(episode, anchor)`` blocks until the child answers one chunk.
    """

    def __init__(self, backend_python: str, dataset_root: str, checkpoint: str,
                 device: str, model_size: str, force_fusion: str | None):
        server = os.path.join(_HERE, "muse_force_infer_server.py")
        cmd = [backend_python, "-u", server,
               "--dataset-root", dataset_root, "--device", device,
               "--model-size", model_size]
        if checkpoint:
            cmd += ["--checkpoint", checkpoint]
        if force_fusion:
            cmd += ["--force-fusion", force_fusion]
        print("[muse-viz] launching backend:", " ".join(cmd), flush=True)
        # stderr streams straight to our terminal so the model's progress /
        # errors are visible; stdout is the pure protocol channel.
        self.proc = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=None,
            text=False, bufsize=0)
        self._lock = threading.Lock()
        # Generous timeouts: the backend builds a torch model + presolves on the
        # first message, so the handshake gets a long budget; each rollout is one
        # diffusion sample. A wedged (alive-but-stuck) backend then surfaces as a
        # clean error on the loop thread's try/except instead of freezing viser
        # forever (Ctrl-C still works either way, but this keeps the UI alive).
        self._handshake_timeout = 900.0
        self._rollout_timeout = 300.0
        self.ready = self._read_msg(expect="ready", timeout=self._handshake_timeout)
        self.info = json.loads(str(self.ready["info"]))
        self.zarr_path = str(self.ready["zarr_path"])
        self.episode_starts = self.ready["episode_starts"].astype(np.int64)
        self.episode_frame_nums = self.ready["episode_frame_nums"].astype(np.int64)
        self.ft_scale = self.ready["ft_scale"].astype(np.float64)
        self.n_episodes = int(self.episode_frame_nums.shape[0])
        # store-global train/val episode ids (present since the deploy-fidelity
        # fix; tolerate an older backend that omits them).
        self.train_episodes = set(
            int(i) for i in self.ready.get("train_episodes", np.empty(0)).tolist())
        self.val_episodes = set(
            int(i) for i in self.ready.get("val_episodes", np.empty(0)).tolist())

    def _read_msg(self, expect: str | None = None, timeout: float | None = None) -> dict:
        """Read protocol lines from the child until one arrives (skipping any
        stray non-protocol bytes on stdout, which should not happen). Raises if
        the child dies OR (when ``timeout`` is set) if it goes silent for longer
        than ``timeout`` seconds — an alive-but-wedged backend (e.g. stuck in a
        CUDA sync) then surfaces as a clean error rather than freezing the render
        loop forever. Returns the decoded npz as a dict of arrays."""
        import select
        import time

        deadline = None if timeout is None else time.monotonic() + timeout
        fd = self.proc.stdout.fileno()
        while True:
            if deadline is not None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(
                        f"backend produced no '{expect or 'msg'}' within {timeout:.0f}s "
                        "(alive but not responding — see its stderr above).")
                # Wait for data OR timeout. readline() below still blocks until a
                # full line, but only once select says bytes are available, so a
                # truly silent backend can't wedge us past the deadline.
                if not select.select([fd], [], [], min(remaining, 5.0))[0]:
                    if self.proc.poll() is not None:
                        raise RuntimeError(
                            f"backend exited (code {self.proc.returncode}) before a "
                            f"'{expect or 'msg'}' message; see its stderr above.")
                    continue
            raw = self.proc.stdout.readline()
            if not raw:
                code = self.proc.poll()
                raise RuntimeError(
                    f"backend exited (code {code}) before a '{expect or 'msg'}' "
                    "message; see its stderr above.")
            line = raw.decode("ascii", "replace").rstrip("\n")
            if not line.startswith(_MSG_PREFIX):
                continue  # stray line; ignore (stderr carries the real logs)
            tag, b64 = line[len(_MSG_PREFIX):].split(" ", 1)
            data = dict(np.load(io.BytesIO(base64.b64decode(b64)), allow_pickle=False))
            data["_tag"] = tag
            if expect is None or tag == expect or tag == "error":
                return data

    def rollout(self, episode: int, anchor: int) -> dict:
        """Request one chunk; block for the reply (bounded by the rollout timeout).
        Raises on a backend error or timeout."""
        with self._lock:
            req = json.dumps({"episode": int(episode), "anchor": int(anchor)}) + "\n"
            self.proc.stdin.write(req.encode("ascii"))
            self.proc.stdin.flush()
            msg = self._read_msg(timeout=self._rollout_timeout)
        if msg["_tag"] == "error":
            raise RuntimeError(f"backend rollout error: {str(msg['msg'])}")
        return msg

    def close(self) -> None:
        try:
            if self.proc.poll() is None:
                self.proc.stdin.write(b'{"cmd":"quit"}\n')
                self.proc.stdin.flush()
                self.proc.wait(timeout=5)
        except Exception:
            pass
        finally:
            if self.proc.poll() is None:
                self.proc.kill()


# ===========================================================================
# Recorded episode (FLoRe .venv side) — reads the SAME zarr the backend opened.
# ===========================================================================
class RecordedEpisode:
    """One episode of the converted zarr, for the RECORDED (blue) trajectory.

    Reads absolute per-keypoint poses + gripper + F/T + RGB for the chosen
    episode. This is intentionally a thin, independent reader (not shared with
    rollout_viz.Episode) so the two tools stay decoupled. The backend already
    told us which zarr + which store-global episode layout to use, so the blue
    trajectory and the model's obs come from the exact same frames."""

    def __init__(self, zarr_path: str, episode: int, episode_starts: np.ndarray,
                 episode_frame_nums: np.ndarray):
        import zarr
        from imagecodecs.numcodecs import register_codecs

        register_codecs()  # camera0_rgb is JpegXl; decode-on-read needs this
        self.path = zarr_path
        self.g = zarr.open(zarr_path, "r")
        self.attrs = dict(self.g.attrs)
        self.n_ep = int(episode_frame_nums.shape[0])
        self.ep = int(np.clip(episode, 0, self.n_ep - 1))
        self._a = int(episode_starts[self.ep])
        self.T = int(episode_frame_nums[self.ep])
        self._b = self._a + self.T
        self.ep_lengths = [int(n) for n in episode_frame_nums]

        data = self.g["data"]
        self.data_keys = set(data.keys())
        self.primary_role = str(self.attrs.get("robot0_primary_role", "right"))
        self.tracking_mode = str(self.attrs.get("tracking_mode", "?"))
        self.has_ft = "robot0_ft_wrench" in self.data_keys

        # Timestamps are RELATIVE per episode (DECISION-A) — each starts at 0.
        self.ts = self._slice("episode_timestamps").reshape(-1).astype(np.float64)

        # Which recorded keypoints are present -> their absolute pose arrays.
        # {name: (pos[T,3], rot6d[T,6], rotvec[T,3], grip_norm[T]|None)}.
        self.kp: dict[str, tuple] = {}
        for name, (pos_key, rot_key, grip_key) in _KP_KEYS.items():
            if pos_key not in self.data_keys:
                continue
            pos = self._slice(pos_key).astype(np.float64)          # (T,3)
            rotvec = self._slice(rot_key).astype(np.float64)       # (T,3)
            rot6d = rotvec_to_rot6d(rotvec)                        # (T,6)
            grip_norm = None
            if grip_key is not None and grip_key in self.data_keys:
                gw = self._slice(grip_key).reshape(-1).astype(np.float64)  # metres
                grip_norm = gripper_width_to_norm(gw)              # [0,1]
            self.kp[name] = (pos, rot6d, rotvec, grip_norm)

        # The primary keypoint (LEFT-canonical -> 'left') drives the camera + F/T.
        self.primary_kp = "left" if self.primary_role == "left" else "right"
        self.ft = self._slice("robot0_ft_wrench").astype(np.float64) if self.has_ft else None

    def _slice(self, key: str) -> np.ndarray:
        return np.asarray(self.g[f"data/{key}"][self._a:self._b])

    def image_at(self, i: int) -> np.ndarray:
        return np.asarray(self.g["data/camera0_rgb"][self._a + int(i)])


# ===========================================================================
# IK helpers (single-hand-aware) — the same math rollout_viz.py uses, duplicated
# to keep the two tools independent.
# ===========================================================================
def _rot6d_to_wxyz(rot6d: np.ndarray) -> np.ndarray:
    q_xyzw = rotmat_to_quat_xyzw(rot6d_to_rotmat(np.asarray(rot6d, dtype=np.float64)))
    return np.asarray([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]], dtype=np.float64)


def _rotvec_to_wxyz(rotvec: np.ndarray) -> np.ndarray:
    return _rot6d_to_wxyz(rotvec_to_rot6d(np.asarray(rotvec, dtype=np.float64)[None])[0])


def tcp_to_tracker(pos: np.ndarray, rot6d: np.ndarray, inv_ext: np.ndarray):
    """Recover per-frame TRACKER poses from stored TCP poses for a HAND keypoint.

    The converter stored ``world_T_tcp = world_T_tracker @ T_tracker_tcp`` for the
    hands (0.21 m along wrist +x). The home-anchored IK must be driven from the
    TRACKER pose (exactly like replay_raw_viz / rollout_viz):
    ``world_T_tracker = world_T_tcp @ inv(T_tracker_tcp)``. Feeding TCP poses
    straight to the IK injects a spurious lever-arm translation on wrist
    rotations. Head keypoints get NO extrinsic and must not be passed here."""
    pos = np.asarray(pos, dtype=np.float64)
    rot6d = np.asarray(rot6d, dtype=np.float64)
    K = pos.shape[0]
    world_T_tcp = np.broadcast_to(np.eye(4), (K, 4, 4)).copy()
    world_T_tcp[:, :3, :3] = rot6d_to_rotmat(rot6d)
    world_T_tcp[:, :3, 3] = pos
    world_T_tracker = world_T_tcp @ inv_ext[None]
    return world_T_tracker[:, :3, 3], rotmat_to_rot6d(world_T_tracker[:, :3, :3])


def presolve_trajectory(ik, mapper, active_roles, kp_poses, prim_role,
                        prim_grip_norm, other_role, other_grip_norm,
                        inv_ext, hand_roles, calib_ref=None, seed_q=None):
    """Solve the multi-role IK over a trajectory and cache the yam config per frame.

    ``kp_poses`` = {role: (pos[K,3], rot6d[K,6])} for the ACTIVE roles (absolute
    world TCP poses for the hands, raw body pose for the head). HAND poses are
    first recovered to the tracker frame via ``inv_ext``. Calibration is
    home-anchored; ``calib_ref`` overrides the reference (the rollout reuses the
    recorded frame-0 tracker poses so both share ONE calibration and the orange
    line branches off the blue). ``seed_q`` seeds the solver from the recorded
    arm's solved config at the anchor so the branch is tight from step 0.

    Returns (cfgs[K,n_yam], prim_wrist_pos[K,3], prim_wrist_wxyz[K,4],
    site_paths{role:[K,3]}, q_seq[K,nq]). This mirrors rollout_viz.presolve."""
    roles = list(active_roles)
    K = len(next(iter(kp_poses.values()))[0])
    n_yam = len(mapper._yam_names)
    cfgs = np.zeros((K, n_yam))
    prim_wrist_pos = np.zeros((K, 3))
    prim_wrist_wxyz = np.tile([1.0, 0.0, 0.0, 0.0], (K, 1)).astype(float)
    site_paths = {role: np.zeros((K, 3)) for role in roles}
    q_seq = np.zeros((K, ik._solver.configuration.q.shape[0]))
    prim_wrist_link = f"{prim_role}_wrist_yaw_link"

    ik_poses = {}
    for role in roles:
        pos, rot6d = kp_poses[role]
        if role in hand_roles:
            pos, rot6d = tcp_to_tracker(pos, rot6d, inv_ext)
        wxyz = np.stack([_rot6d_to_wxyz(rot6d[i]) for i in range(K)], axis=0)
        ik_poses[role] = (np.asarray(pos, dtype=float), wxyz)

    def pose_at(role, i):
        pos, wxyz = ik_poses[role]
        return pos[i], wxyz[i]

    ref = calib_ref if calib_ref is not None else {role: pose_at(role, 0) for role in roles}
    ik.recalibrate(ref)
    if seed_q is not None:
        ik._solver.configuration.update(q=np.asarray(seed_q, dtype=float))
    for i in range(K):
        q = ik.step({role: pose_at(role, i) for role in roles})
        jaw_kw = {}
        prim_jaw = (float(prim_grip_norm[i]) * _JAW_OPEN_M) if prim_grip_norm is not None else 0.0
        jaw_kw["left_jaw_m" if prim_role == "left" else "right_jaw_m"] = prim_jaw
        if other_role is not None and other_grip_norm is not None:
            other_jaw = float(other_grip_norm[i]) * _JAW_OPEN_M
            jaw_kw["left_jaw_m" if other_role == "left" else "right_jaw_m"] = other_jaw
        cfgs[i] = mapper.to_yam(q, **jaw_kw)
        q_seq[i] = q
        conf = ik._solver.configuration
        wrist = conf.get_transform_frame_to_world(frame_name=prim_wrist_link, frame_type="body")
        prim_wrist_pos[i] = wrist.translation()
        prim_wrist_wxyz[i] = wrist.rotation().wxyz
        for role in roles:
            fn, ft = _SITE_FRAMES[role]
            site_paths[role][i] = conf.get_transform_frame_to_world(
                frame_name=fn, frame_type=ft).translation()
    return cfgs, prim_wrist_pos, prim_wrist_wxyz, site_paths, q_seq


def _tracker_poses_at(ep, active_roles, hand_roles, inv_ext, frame_idx):
    """Recorded TRACKER poses {role:(pos[3],wxyz[4])} at one frame — the shared
    home-anchor reference for the rollout IK (so it continues from the recorded arm)."""
    ref = {}
    for role in active_roles:
        pos = ep.kp[role][0][frame_idx:frame_idx + 1]     # (1,3) TCP/world
        rot6d = ep.kp[role][1][frame_idx:frame_idx + 1]   # (1,6)
        if role in hand_roles:
            pos, rot6d = tcp_to_tracker(pos, rot6d, inv_ext)
        ref[role] = (np.asarray(pos[0], dtype=float), _rot6d_to_wxyz(rot6d[0]))
    return ref


# ---------------------------------------------------------------------------
# Line + extrinsic helpers (duplicated from rollout_viz for independence)
# ---------------------------------------------------------------------------
def _blue_gradient_colors(n: int) -> np.ndarray:
    if n <= 1:
        return _RECORDED_BLUE_LIGHT[None].astype(np.uint8)
    t = np.linspace(0.0, 1.0, n)[:, None]
    ramp = (_RECORDED_BLUE_DARK[None] * (1 - t) + _RECORDED_BLUE_LIGHT[None] * t)
    return np.clip(ramp, 0, 255).astype(np.uint8)


def draw_gradient_path(server, name: str, pts: np.ndarray, line_width: float = 3.0):
    pts = np.asarray(pts, dtype=np.float32)
    if pts.shape[0] < 2:
        return server.scene.add_line_segments(name, points=np.zeros((1, 2, 3), np.float32),
                                              colors=np.zeros((1, 2, 3), np.uint8), visible=False)
    seg_pts = np.stack([pts[:-1], pts[1:]], axis=1)
    vcols = _blue_gradient_colors(pts.shape[0])
    seg_cols = np.stack([vcols[:-1], vcols[1:]], axis=1)
    return server.scene.add_line_segments(name, points=seg_pts, colors=seg_cols, line_width=line_width)


def draw_solid_path(server, name: str, pts: np.ndarray, color, line_width: float = 3.0):
    pts = np.asarray(pts, dtype=np.float32)
    if pts.shape[0] < 2:
        return server.scene.add_line_segments(name, points=np.zeros((1, 2, 3), np.float32),
                                              colors=np.zeros((1, 2, 3), np.uint8), visible=False)
    seg_pts = np.stack([pts[:-1], pts[1:]], axis=1)
    col = np.array(color, dtype=np.uint8)
    seg_cols = np.broadcast_to(col, (seg_pts.shape[0], 2, 3)).copy()
    return server.scene.add_line_segments(name, points=seg_pts, colors=seg_cols, line_width=line_width)


def _load_inv_tcp_extrinsic(ep: "RecordedEpisode", override: str) -> np.ndarray:
    """inv(T_tracker_tcp) as a 4x4, to recover tracker poses from stored TCP poses.

    Resolution: (1) an explicit --tcp-extrinsic JSON, (2) the repo-default
    ``.../calibration/tcp_extrinsic.json``. Identity when the zarr wasn't
    TCP-calibrated (then TCP == tracker, no-op). WARN + identity if calibration is
    claimed but no file resolves."""
    if not bool(ep.attrs.get("tcp_calibrated", False)):
        return np.eye(4)
    path = override or os.path.join(_UMI, "robot_assets", "calibration", "tcp_extrinsic.json")
    if not os.path.exists(path):
        print(f"[muse-viz] WARNING: tcp_calibrated=True but no extrinsic at {path}; "
              "driving IK from raw TCP poses (wrist rotations show a ~0.2 m lever-arm "
              "offset). Pass --tcp-extrinsic to fix.", flush=True)
        return np.eye(4)
    d = json.load(open(path))
    T = np.asarray(d.get("T_tracker_tcp", np.eye(4).tolist()), dtype=np.float64)
    print(f"[muse-viz] tracker->TCP extrinsic from {os.path.relpath(path, _FLORE_ROOT)} "
          f"(translation {np.round(T[:3, 3], 3)} m).", flush=True)
    return np.linalg.inv(T)


def _make_ik(active_roles, config_name, reach_scale, yam_urdf):
    """Build the multi-role IK + yam mapper (LEFT-canonical prefix of left/right/head)."""
    from g1_right_hand_ik import YamConfigMapper, yam_actuated_joint_names
    from g1_three_point_ik import ThreePointIK

    ik = ThreePointIK(config_name=config_name, reach_scale=reach_scale,
                      active_roles=tuple(active_roles))
    mapper = YamConfigMapper(ik._humi_joint_names, yam_actuated_joint_names(yam_urdf))
    return ik, mapper


# ===========================================================================
def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--backend-python", required=True,
                   help="the uvaforce interpreter that runs muse_force_infer_server.py")
    p.add_argument("--dataset-root", required=True,
                   help="dir containing the muse_force zarr (as the UVA config expects)")
    p.add_argument("--checkpoint", default="",
                   help="UVA workspace checkpoint (.ckpt); empty = backend random-init")
    p.add_argument("--episode", type=int, default=0, help="episode index to load")
    p.add_argument("--device", default="cuda", help="backend torch device")
    p.add_argument("--model-size", default="mar_base")
    p.add_argument("--force-fusion", default=None, choices=[None, "early", "late", "both"])
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8087)
    p.add_argument("--yam-urdf", default=_DEFAULT_YAM)
    p.add_argument("--tcp-extrinsic", default="")
    p.add_argument("--ik-config", default="proposal")
    p.add_argument("--reach-scale", type=float, default=1.0)
    p.add_argument("--cam-offset", type=float, nargs=3, default=(0.10, 0.0, 0.04))
    p.add_argument("--cam-rpy", type=float, nargs=3, default=(4.7124, 1.5708, 0.0))
    p.add_argument("--cam-fov-deg", type=float, default=58.0)
    p.add_argument("--selftest-anchor", type=int, default=None,
                   help="headless CI smoke: spawn the backend, run ONE rollout at "
                        "this frame end-to-end (recorded IK + backend + un-relativize "
                        "+ rollout IK), print a summary, and exit WITHOUT serving viser.")
    args = p.parse_args(argv if argv is not None else sys.argv[1:])

    if args.selftest_anchor is not None:
        return _selftest(args)

    # --- spawn the model backend (UVA env) + get the handshake ---
    backend = InferBackend(args.backend_python, args.dataset_root, args.checkpoint,
                           args.device, args.model_size, args.force_fusion)
    # Guarantee the child is reaped no matter how main() exits — a SystemExit
    # (e.g. the non-LEFT-canonical guard below), an exception during IK setup, or
    # a normal shutdown. close() is idempotent, so the explicit call in the
    # timeline.run() finally is harmless if this fires first. (Without this the
    # idle child would still get stdin-EOF when our pipes close on interpreter
    # exit, but an explicit graceful quit is cleaner and frees the GPU sooner.)
    import atexit
    atexit.register(backend.close)
    info = backend.info
    print(f"[muse-viz] backend ready: zarr={backend.zarr_path} "
          f"episodes={backend.n_episodes} H={info['action_horizon']} "
          f"predict_force={info['predict_force']} random_init={info['random_init']}",
          flush=True)

    # --- load the recorded episode (same zarr the backend opened) ---
    ep = RecordedEpisode(backend.zarr_path, args.episode,
                         backend.episode_starts, backend.episode_frame_nums)
    print(f"[muse-viz] {os.path.basename(backend.zarr_path)}: episode {ep.ep}/{ep.n_ep - 1}, "
          f"{ep.T} frames, mode={ep.tracking_mode}, primary={ep.primary_role}, "
          f"keypoints={list(ep.kp)}", flush=True)
    if ep.primary_role != "left":
        # backend.close() runs via the atexit hook registered above.
        raise SystemExit(
            "[muse-viz] this tool drives the humanoid via ThreePointIK (LEFT-canonical, "
            f"'left' primary), but the zarr is primary_role={ep.primary_role!r}. "
            "Re-record/convert LEFT-canonical.")

    # Active IK roles = recorded keypoints, as a prefix of (left, right, head).
    present = set(ep.kp)
    active_roles = tuple(r for r in ("left", "right", "head") if r in present)
    prim_role = ep.primary_kp  # 'left'
    other_hand_role = "right" if "right" in present else None
    hand_roles = {r for r in active_roles if r in ("left", "right")}

    inv_ext = _load_inv_tcp_extrinsic(ep, args.tcp_extrinsic)
    print(f"[muse-viz] building IK ({args.ik_config}, roles={active_roles})…", flush=True)
    ik, mapper = _make_ik(active_roles, args.ik_config, args.reach_scale, args.yam_urdf)

    # --- pre-solve the RECORDED trajectory IK once (whole episode) ---
    def kp_poses_recorded(frames: np.ndarray):
        return {r: (ep.kp[r][0][frames], ep.kp[r][1][frames]) for r in active_roles}

    prim_grip_rec = ep.kp[prim_role][3]
    other_grip_rec = ep.kp[other_hand_role][3] if other_hand_role else None
    all_frames = np.arange(ep.T)
    print(f"[muse-viz] pre-solving RECORDED IK over {ep.T} frames…", flush=True)
    rec_cfgs, rec_wrist_pos, rec_wrist_wxyz, rec_paths, rec_q_seq = presolve_trajectory(
        ik, mapper, active_roles, kp_poses_recorded(all_frames),
        prim_role, prim_grip_rec, other_hand_role, other_grip_rec, inv_ext, hand_roles)
    print("[muse-viz] recorded IK done.", flush=True)

    # --- scene ---
    server = viser.ViserServer(host=args.host, port=args.port, label="FLoRe MuSe-force rollout viz")
    server.scene.set_up_direction("+z")
    server.scene.add_grid("/grid", plane="xy", shadow_opacity=0.0, infinite_grid=True, cell_size=0.25)

    base_xyz = np.asarray(ik.home_q()[0:3], dtype=float).copy()
    server.scene.add_frame("/robot", show_axes=False, position=tuple(base_xyz))
    urdf = ViserUrdf(server, urdf_or_path=Path(args.yam_urdf), root_node_name="/robot")

    follow = server.scene.add_frame("/follow", show_axes=True, axes_length=0.08,
                                    axes_radius=0.004, origin_radius=0.01)
    rpy = np.asarray(args.cam_rpy, float)
    cam_rot = (vtf.SO3.from_rpy_radians(float(rpy[0]), 0.0, 0.0)
               .multiply(vtf.SO3.from_rpy_radians(0.0, float(rpy[1]), 0.0)))
    server.scene.add_frame("/follow/cam", show_axes=False,
                           position=tuple(np.asarray(args.cam_offset, float)),
                           wxyz=tuple(cam_rot.wxyz))
    frustum = server.scene.add_camera_frustum(
        "/follow/cam/view", fov=np.deg2rad(args.cam_fov_deg), aspect=1.0, scale=0.12,
        color=(40, 200, 255))
    wrench = rc.WrenchArrows(server, name="/follow/wrench") if ep.has_ft else None

    rec_path_handles = {r: draw_gradient_path(server, f"/recorded/{r}", rec_paths[r])
                        for r in active_roles if rec_paths[r].shape[0] >= 2}
    roll_path_handles: dict = {}

    # --- rollout state (mutable; cross-thread requests set by GUI callbacks) ---
    roll: dict = {
        "cfgs": None, "wrist_pos": None, "wrist_wxyz": None, "ts": None,
        "paths": None, "prim_grip": None, "force_pred": None, "gt_force": None,
        "req_anchor": 0, "shown_anchor": 0, "pending": False,
        "source": "recorded", "pending_source": None,
    }

    # --- GUI ---
    gui = server.gui
    gui.configure_theme(control_layout="collapsible", control_width="large")
    with gui.add_folder("Dataset"):
        gui.add_text("File", initial_value=os.path.basename(backend.zarr_path), disabled=True)
        gui.add_text("Episode", initial_value=f"{ep.ep} / {ep.n_ep - 1}  ({ep.T} fr)", disabled=True)
        gui.add_text("Mode", initial_value=f"{ep.tracking_mode} (primary={ep.primary_role})", disabled=True)
        # train/val provenance: a tight overlay on a TRAIN episode is in-distribution
        # fit, not generalization — the monitored val_action_l2 metric is the val
        # split only. Warn so a train-episode fit isn't over-read.
        if backend.train_episodes or backend.val_episodes:
            _split = ("val (held-out — matches the monitored metric)"
                      if ep.ep in backend.val_episodes else
                      "TRAIN (in-distribution — expect an optimistically tight fit)"
                      if ep.ep in backend.train_episodes else "unknown")
            gui.add_text("Split", initial_value=_split, disabled=True)
    with gui.add_folder("Model (UVA)"):
        gui.add_text("Checkpoint",
                     initial_value=("RANDOM-INIT" if info["random_init"]
                                    else os.path.basename(args.checkpoint.rstrip("/"))),
                     disabled=True)
        gui.add_text("predict_force", initial_value=str(info["predict_force"]), disabled=True)
        gui.add_text("horizon (future)", initial_value=f"H={info['action_horizon']}", disabled=True)
        gui.add_text("pose repr", initial_value=info["action_pose_repr"], disabled=True)
    with gui.add_folder("Rollout"):
        start_slider = gui.add_slider("Start frame", min=0, max=max(ep.T - 1, 0), step=1, initial_value=0)
        rollout_btn = gui.add_button("▶ Rollout from start frame")
        show_dd = gui.add_dropdown("Showing", ("recorded", "rollout"), initial_value="recorded")
        roll_status = gui.add_text("Status", initial_value="(no rollout yet)", disabled=True)
    with gui.add_folder("Frame readout"):
        rd_t = gui.add_text("t (s, rel)", initial_value="0.000", disabled=True)
        rd_src = gui.add_text("trajectory", initial_value="recorded", disabled=True)
        rd_prim_pos = gui.add_text(f"{prim_role} xyz", initial_value="-", disabled=True)
        rd_prim_grip = gui.add_text(f"{prim_role} gripper", initial_value="-", disabled=True)
        rd_force = gui.add_text("|F| recorded (N)", initial_value="-", disabled=True)
        rd_force_pred = gui.add_text("|F| predicted (N)", initial_value="-", disabled=True)
        rd_torque = gui.add_text("|T| recorded (Nm)", initial_value="-", disabled=True)
    with gui.add_folder("View"):
        show_cam = gui.add_checkbox("Show camera (recorded obs)", initial_value=True)
        show_wrench = gui.add_checkbox("Show F/T arrows", initial_value=bool(ep.has_ft))
        show_recorded = gui.add_checkbox("Show recorded (blue) paths", initial_value=True)
        show_rollout = gui.add_checkbox("Show rollout (orange) path", initial_value=True)
    with gui.add_folder("Charts"):
        grip_chart = rc.make_single_chart(gui, f"{prim_role} gripper width (m)", "width",
                                          color_idx=1, y_range=(0.0, max(_JAW_OPEN_M * 2.2, 0.1)))
        prim_grip_m_rec = (gripper_norm_to_width(prim_grip_rec) if prim_grip_rec is not None
                           else np.zeros(ep.T))
        rc.set_full_single(grip_chart, ep.ts, prim_grip_m_rec)
        # Force charts: recorded x/y/z + a PREDICTED overlay line per axis. The
        # recorded series is the whole episode; the predicted series is drawn only
        # while a rollout is showing (empty otherwise).
        force_chart = torque_chart = None
        if ep.has_ft:
            force_chart = _make_force_chart(gui, "Force (N) — recorded + PRED", ("Fx", "Fy", "Fz"))
            torque_chart = _make_force_chart(gui, "Torque (Nm) — recorded + PRED", ("Tx", "Ty", "Tz"))
            _set_force_recorded(force_chart, ep.ts, ep.ft[:, 0:3])
            _set_force_recorded(torque_chart, ep.ts, ep.ft[:, 3:6])

    # ------------------------------------------------------------------
    # Rollout click -> flag only (GUI thread). The loop thread does the (slow)
    # backend round-trip + IK so the scene is never mutated from a callback.
    # ------------------------------------------------------------------
    @rollout_btn.on_click
    def _(_e) -> None:
        req = int(start_slider.value)
        roll["req_anchor"] = req
        roll["pending"] = True
        roll_status.value = f"rolling out from frame {req}…"

    _dd_suppress = {"on": False}

    @show_dd.on_update
    def _(_e) -> None:
        if _dd_suppress["on"]:
            return
        roll["pending_source"] = str(show_dd.value)

    def _set_dropdown(value: str) -> None:
        _dd_suppress["on"] = True
        try:
            show_dd.value = value
        finally:
            _dd_suppress["on"] = False

    def _retarget_timeline(source: str) -> None:
        want = "rollout" if (source == "rollout" and roll["cfgs"] is not None) else "recorded"
        if want == roll["source"]:
            _set_dropdown(want)
            return
        roll["source"] = want
        timeline.set_series(roll["ts"] if want == "rollout" else ep.ts)
        _set_dropdown(want)
        _refresh_force_pred_overlay()

    def _refresh_force_pred_overlay() -> None:
        """Draw / clear the PREDICTED-force chart overlay for the current source.

        The predicted chunk covers recorded frames anchor+1..anchor+H, so it is
        drawn at those RECORDED episode indices on the (whole-episode) recorded
        force axis — directly over the ground-truth force it is predicting — NOT
        at the front of the axis. ``base_idx`` is that starting recorded index."""
        if not ep.has_ft:
            return
        showing_roll = roll["source"] == "rollout" and roll["force_pred"] is not None
        if showing_roll:
            fp = roll["force_pred"]                 # (H,6) physical
            base_idx = int(roll["shown_anchor"]) + 1  # future frame anchor+1
            _set_force_pred(force_chart, base_idx, fp[:, 0:3])
            _set_force_pred(torque_chart, base_idx, fp[:, 3:6])
        else:
            _clear_force_pred(force_chart)
            _clear_force_pred(torque_chart)

    def compute_rollout(anchor: int) -> None:
        """Slow path (loop thread): ask the backend for one chunk, un-relativized
        to ABSOLUTE poses already, then pre-solve the rollout IK for the LEFT hand
        and cache everything (+ predicted/recorded force) for the timeline."""
        msg = backend.rollout(ep.ep, anchor)
        # Absolute predicted LEFT-hand chunk (world TCP poses) + gripper + force.
        act_pos = msg["action_pos"].astype(np.float64)         # (H,3)
        act_rotvec = msg["action_rotvec"].astype(np.float64)   # (H,3)
        act_grip_m = msg["action_grip"].astype(np.float64)     # (H,) metres
        force_pred = msg["force_pred"].astype(np.float64)      # (H,6) or (0,6)
        gt_force = msg["gt_force_future"].astype(np.float64)   # (H,6)
        H = act_pos.shape[0]
        act_rot6d = rotvec_to_rot6d(act_rotvec)                # (H,6)

        # The model predicts ONLY the LEFT hand. Hold the right hand + head at
        # their recorded ANCHOR pose for the whole rollout so the humanoid still
        # renders and only the predicted arm moves (honest: unpredicted parts frozen).
        kp_poses = {"left": (act_pos, act_rot6d)}
        for role in active_roles:
            if role == "left":
                continue
            pr = ep.kp[role][0][anchor:anchor + 1]
            rr = ep.kp[role][1][anchor:anchor + 1]
            kp_poses[role] = (np.repeat(pr, H, axis=0), np.repeat(rr, H, axis=0))

        prim_grip = np.clip(gripper_width_to_norm(act_grip_m), 0.0, 1.0)  # (H,) [0,1]
        other_grip = None
        if other_hand_role is not None and other_grip_rec is not None:
            og = float(other_grip_rec[anchor])
            other_grip = np.full(H, og, dtype=np.float64)

        # Share the recorded run's EXACT calibration (recorded frame-0 tracker
        # poses as the home anchor) + seed from the recorded solved config at the
        # anchor, so orange branches tightly off blue at the anchor.
        calib_ref = _tracker_poses_at(ep, active_roles, hand_roles, inv_ext, 0)
        cfgs, wpos, wwxyz, paths, _q = presolve_trajectory(
            ik, mapper, active_roles, {r: kp_poses[r] for r in active_roles},
            prim_role, prim_grip, other_hand_role, other_grip, inv_ext, hand_roles,
            calib_ref=calib_ref, seed_q=rec_q_seq[anchor])

        # Rollout scrubber time axis = the REAL recorded timestamps of the future
        # frames anchor+1..anchor+H, rebased to start at 0. Using the actual grid
        # (not arange(H)*median_dt) keeps playback honest on the non-uniform camera
        # timeline (dt swings ~0.06–0.10 s, so a synthetic axis drifts up to a full
        # frame over H steps). Falls back to a uniform grid only past the episode end.
        fut_ts = ep.ts[anchor + 1: anchor + 1 + H]
        if fut_ts.shape[0] == H and H > 0:
            roll_ts = fut_ts - fut_ts[0]
        else:
            dt = float(np.median(np.diff(ep.ts))) if ep.T > 1 else 0.1
            roll_ts = np.arange(H) * dt
        roll["source"] = "recorded"
        roll.update(cfgs=cfgs, wrist_pos=wpos, wrist_wxyz=wwxyz, paths=paths,
                    prim_grip=prim_grip, force_pred=(force_pred if force_pred.size else None),
                    gt_force=gt_force, shown_anchor=anchor, ts=roll_ts)
        # Orange rollout path — LEFT hand only (the model's prediction); other
        # roles are static holds, so no orange line for them.
        for name in ("left",):
            if paths[name].shape[0] < 2:
                draw_solid_path(server, f"/rollout/{name}", paths[name], _ROLLOUT_ORANGE)
                continue
            roll_path_handles[name] = draw_solid_path(
                server, f"/rollout/{name}", paths[name], _ROLLOUT_ORANGE)
            roll_path_handles[name].visible = show_rollout.value
        _retarget_timeline("rollout")
        note = ""
        if info["random_init"]:
            note = "  [RANDOM-INIT — not meaningful]"
        if force_pred.size:
            note += (f"  |Fpred[0]|={np.linalg.norm(force_pred[0, :3]):.2f}N "
                     f"vs |Fgt[0]|={np.linalg.norm(gt_force[0, :3]):.2f}N")
        roll_status.value = f"rollout @ frame {anchor}: H={H} steps{note}"

    # ------------------------------------------------------------------
    # Render: index either the recorded episode or the cached rollout chunk.
    # ------------------------------------------------------------------
    _last = {"cam_i": -1}

    def render_at(i: int) -> None:
        src = roll["source"]
        have_roll = roll["cfgs"] is not None
        use_roll = (src == "rollout") and have_roll

        if use_roll:
            cfgs, wpos, wwxyz, ts = roll["cfgs"], roll["wrist_pos"], roll["wrist_wxyz"], roll["ts"]
        else:
            cfgs, wpos, wwxyz, ts = rec_cfgs, rec_wrist_pos, rec_wrist_wxyz, ep.ts
        i = int(np.clip(i, 0, cfgs.shape[0] - 1))
        urdf.update_cfg(cfgs[i])
        follow.position = tuple(wpos[i])
        follow.wxyz = tuple(wwxyz[i])

        # Recorded ground-truth frame for this step. The UVA future chunk is frames
        # anchor+1..anchor+H (get_trajectory takes the SECOND half of the length-32
        # window), so rollout step i corresponds to recorded frame anchor+1+i — NOT
        # anchor+i. Using that index keeps the camera backdrop, the recorded-force
        # readout, and the model's predicted step i all describing the SAME instant
        # (the honest ground-truth partner the backend returns as gt_force_future[i]).
        rec_i = i if not use_roll else int(np.clip(roll["shown_anchor"] + 1 + i, 0, ep.T - 1))
        frustum.visible = show_cam.value
        if show_cam.value and rec_i != _last["cam_i"]:
            _last["cam_i"] = rec_i
            img = ep.image_at(rec_i)
            h, w = img.shape[:2]
            frustum.aspect = w / max(h, 1)
            frustum.image = np.asarray(img)

        # F/T arrows: RECORDED wrench on the primary wrist, recorded view only.
        if wrench is not None:
            ft_arrows_on = show_wrench.value and ep.has_ft and not use_roll
            wrench.set_visible(ft_arrows_on)
            if ft_arrows_on and ep.ft is not None:
                wrench.update(wpos[i], wwxyz[i], ep.ft[rec_i, 0:3], ep.ft[rec_i, 3:6])

        # Readouts.
        rd_t.value = f"{(ts[i] - ts[0]) if len(ts) else 0.0:.3f}"
        rd_src.value = "rollout (orange)" if use_roll else "recorded (blue)"
        if use_roll:
            prim_xyz = roll["paths"]["left"][i]
            prim_g = roll["prim_grip"][i] if roll["prim_grip"] is not None else None
        else:
            prim_xyz = rec_paths[prim_role][i]
            prim_g = prim_grip_rec[i] if prim_grip_rec is not None else None
        rd_prim_pos.value = np.array2string(np.asarray(prim_xyz), precision=3, suppress_small=True)
        if prim_g is not None:
            gw = float(gripper_norm_to_width(prim_g))
            rd_prim_grip.value = f"{gw * 1000:.1f} mm ({'grasp' if gw < _GRASP_THRESH_M else 'open'})"
        # Force readouts. In the rollout view the recorded reference for step i is
        # the backend's gt_force_future[i] (frame anchor+1+i) — the exact partner of
        # the predicted force_pred[i], so the two readouts describe the SAME instant.
        # In the recorded view it is simply ep.ft at the current frame.
        if ep.ft is not None:
            if use_roll and roll["gt_force"] is not None:
                j = int(np.clip(i, 0, roll["gt_force"].shape[0] - 1))
                rd_force.value = f"{np.linalg.norm(roll['gt_force'][j, 0:3]):.2f}"
                rd_torque.value = f"{np.linalg.norm(roll['gt_force'][j, 3:6]):.3f}"
            else:
                rd_force.value = f"{np.linalg.norm(ep.ft[rec_i, 0:3]):.2f}"
                rd_torque.value = f"{np.linalg.norm(ep.ft[rec_i, 3:6]):.3f}"
        if use_roll and roll["force_pred"] is not None:
            j = int(np.clip(i, 0, roll["force_pred"].shape[0] - 1))
            rd_force_pred.value = f"{np.linalg.norm(roll['force_pred'][j, 0:3]):.2f}"
        else:
            rd_force_pred.value = "-"

    timeline = rc.Timeline(server, ep.ts, render_at, folder_label="Playback")

    _view_last = {"cam": bool(show_cam.value), "wrench": bool(show_wrench.value)}

    def before_render() -> None:
        for h in rec_path_handles.values():
            h.visible = bool(show_recorded.value)
        for h in roll_path_handles.values():
            h.visible = bool(show_rollout.value)
        if bool(show_cam.value) != _view_last["cam"] or bool(show_wrench.value) != _view_last["wrench"]:
            _view_last["cam"] = bool(show_cam.value)
            _view_last["wrench"] = bool(show_wrench.value)
            timeline.request_render()
        if roll["pending"]:
            anchor = int(roll["req_anchor"])
            roll["pending"] = False
            try:
                compute_rollout(anchor)
            except Exception as e:  # noqa: BLE001 — surface, keep the loop alive
                roll_status.value = f"rollout failed: {e!r}"
                print(f"[muse-viz] rollout error: {e!r}", flush=True)
        elif roll["pending_source"] is not None:
            src = roll["pending_source"]
            roll["pending_source"] = None
            _retarget_timeline(src)

    timeline.before_render = before_render
    print(f"[muse-viz] http://{args.host}:{args.port}  — pick a Start frame, click Rollout, "
          f"toggle 'Showing' recorded/rollout", flush=True)
    try:
        timeline.run()
    finally:
        backend.close()
    return 0


def _selftest(args) -> int:
    """Headless end-to-end smoke: spawn the backend, run ONE rollout at
    ``--selftest-anchor``, exercise the recorded + rollout IK and the
    un-relativized absolute chunk, and assert shapes/finiteness — no viser.

    This is the CI-runnable proof that the two-process pipeline cooperates
    (frontend .venv ⇄ backend uvaforce) without needing a display."""
    backend = InferBackend(args.backend_python, args.dataset_root, args.checkpoint,
                           args.device, args.model_size, args.force_fusion)
    try:
        info = backend.info
        ep = RecordedEpisode(backend.zarr_path, args.episode,
                             backend.episode_starts, backend.episode_frame_nums)
        if ep.primary_role != "left":
            raise SystemExit(f"[selftest] non-LEFT-canonical zarr ({ep.primary_role})")
        present = set(ep.kp)
        active_roles = tuple(r for r in ("left", "right", "head") if r in present)
        hand_roles = {r for r in active_roles if r in ("left", "right")}
        inv_ext = _load_inv_tcp_extrinsic(ep, args.tcp_extrinsic)
        ik, mapper = _make_ik(active_roles, args.ik_config, args.reach_scale, args.yam_urdf)

        # Recorded IK over a short prefix around the anchor (fast, still exercises
        # the exact presolve path + the tracker recovery + site extraction).
        anchor = int(np.clip(args.selftest_anchor, 0, ep.T - 1))
        rec_frames = np.arange(0, min(ep.T, anchor + 1))
        rec_poses = {r: (ep.kp[r][0][rec_frames], ep.kp[r][1][rec_frames])
                     for r in active_roles}
        prim_grip_rec = ep.kp["left"][3]
        _c, _wp, _ww, _paths, rec_q_seq = presolve_trajectory(
            ik, mapper, active_roles, rec_poses, "left",
            prim_grip_rec[rec_frames] if prim_grip_rec is not None else None,
            None, None, inv_ext, hand_roles)

        # Backend rollout + un-relativize + rollout IK for the LEFT hand.
        msg = backend.rollout(ep.ep, anchor)
        act_pos = msg["action_pos"].astype(np.float64)
        act_rot6d = rotvec_to_rot6d(msg["action_rotvec"].astype(np.float64))
        force_pred = msg["force_pred"].astype(np.float64)
        gt_force = msg["gt_force_future"].astype(np.float64)
        H = act_pos.shape[0]
        act_grip = np.clip(gripper_width_to_norm(msg["action_grip"].astype(np.float64)), 0, 1)

        kp_poses = {"left": (act_pos, act_rot6d)}
        for role in active_roles:
            if role == "left":
                continue
            pr = ep.kp[role][0][anchor:anchor + 1]
            rr = ep.kp[role][1][anchor:anchor + 1]
            kp_poses[role] = (np.repeat(pr, H, axis=0), np.repeat(rr, H, axis=0))
        calib_ref = _tracker_poses_at(ep, active_roles, hand_roles, inv_ext, 0)
        cfgs, _wp2, _ww2, paths, _q2 = presolve_trajectory(
            ik, mapper, active_roles, {r: kp_poses[r] for r in active_roles},
            "left", act_grip, None, None, inv_ext, hand_roles,
            calib_ref=calib_ref, seed_q=rec_q_seq[anchor])

        # Assertions: everything finite, shapes consistent, force in physical range.
        assert cfgs.shape[0] == H and np.isfinite(cfgs).all(), "rollout cfgs bad"
        assert paths["left"].shape == (H, 3) and np.isfinite(paths["left"]).all()
        if force_pred.size:
            assert force_pred.shape == (H, 6) and np.isfinite(force_pred).all()
            assert gt_force.shape == (H, 6) and np.isfinite(gt_force).all()
        print("---------------------------------------------------------------")
        print(f"[selftest] PASS  episode={ep.ep} anchor={anchor} H={H} "
              f"random_init={info['random_init']}")
        print(f"[selftest]   recorded roles={active_roles}  left-rollout path {paths['left'].shape}")
        print(f"[selftest]   anchor_pos={np.round(msg['anchor_pos'], 4).tolist()}  "
              f"rollout[0]={np.round(act_pos[0], 4).tolist()}")
        if force_pred.size:
            print(f"[selftest]   |Fpred[0]|={np.linalg.norm(force_pred[0, :3]):.3f} N  "
                  f"|Fgt[0]|={np.linalg.norm(gt_force[0, :3]):.3f} N  (ft_scale={backend.ft_scale.tolist()})")
        else:
            print("[selftest]   force head absent (predict_force off)")
        print("---------------------------------------------------------------")
        return 0
    finally:
        backend.close()


# ---------------------------------------------------------------------------
# Force chart with a recorded triple + a predicted triple (6 series total).
# ---------------------------------------------------------------------------
def _force_series(labels, show_pred: bool):
    """The 7-series spec (t + 3 recorded + 3 predicted). The predicted series are
    hidden (``show=False``) until a rollout populates them.

    WHY the hide matters: the predicted columns are all-NaN before any rollout.
    A *shown* all-NaN series makes uPlot auto-range that series to
    ``[Infinity, -Infinity]`` on the SHARED y-scale, which blanks EVERY series on
    that scale — the recorded lines included (observed: recorded curves invisible
    until a rollout). ``show=False`` keeps the NaN series out of both drawing and
    auto-scaling, so the recorded x/y/z always render; we flip it True only when
    real predicted values exist."""
    from viser import uplot
    ac = rc.AXIS_COLORS
    return (
        uplot.Series(label="t"),
        uplot.Series(label=labels[0], stroke=ac[0], width=1.5),
        uplot.Series(label=labels[1], stroke=ac[1], width=1.5),
        uplot.Series(label=labels[2], stroke=ac[2], width=1.5),
        uplot.Series(label=f"{labels[0]}~pred", stroke=_PRED_FORCE_GREEN, width=1.5,
                     dash=(4.0, 3.0), spanGaps=False, show=show_pred),
        uplot.Series(label=f"{labels[1]}~pred", stroke=_PRED_FORCE_GREEN, width=1.5,
                     dash=(4.0, 3.0), spanGaps=False, show=show_pred),
        uplot.Series(label=f"{labels[2]}~pred", stroke=_PRED_FORCE_GREEN, width=1.5,
                     dash=(4.0, 3.0), spanGaps=False, show=show_pred),
    )


def _make_force_chart(gui, title: str, labels):
    """A force chart: 3 SOLID recorded lines (x/y/z) + 3 DASHED predicted lines.

    Built on the same uplot handle rc.make_xyz_chart uses, but with 6 data series
    so the predicted overlay shares the axes. The recorded series use the standard
    axis colors; the predicted series are a distinct green + dashed so they read
    as "prediction" over the recorded ground truth — and start hidden (see
    ``_force_series``) so their pre-rollout all-NaN columns can't blank the
    recorded lines."""
    from viser import uplot
    x0 = np.array([0.0, 1.0])
    zero = np.zeros(2)
    chart = gui.add_uplot(
        data=(x0, zero, zero, zero, zero, zero, zero),
        series=_force_series(labels, show_pred=False),
        scales={"x": uplot.Scale(time=False)},
        title=title, aspect=2.2, height=120,
    )
    chart._muse_labels = tuple(labels)
    chart._muse_pred_shown = False
    return chart


def _chart_state(chart):
    if not hasattr(chart, "_muse_state"):
        chart._muse_state = {"rec_t": np.array([0.0, 1.0]),
                             "rec_xyz": np.zeros((2, 3)),
                             "pred_base": None, "pred_xyz": None}
    return chart._muse_state


def _push(chart) -> None:
    """Rebuild the (t,x,y,z,xp,yp,zp) tuple from cached recorded + predicted parts.

    The x-axis is ALWAYS the whole-episode recorded time axis (``rec_t`` = ep.ts).
    The predicted overlay is placed at the RECORDED frame indices it predicts —
    ``pred_base .. pred_base+H`` (= anchor+1 .. anchor+H) — so a predicted sample
    is drawn at the same x (and thus directly over the recorded ground-truth
    curve) as the frame it forecasts. Everywhere else the predicted series is NaN
    so the dashed line simply doesn't draw (spanGaps=False)."""
    st = _chart_state(chart)
    t = np.asarray(st["rec_t"], dtype=np.float64).reshape(-1)
    rec = np.asarray(st["rec_xyz"], dtype=np.float64)
    n = t.shape[0]
    pred_cols = [np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)]
    if st["pred_base"] is not None and st["pred_xyz"] is not None:
        px = np.asarray(st["pred_xyz"], dtype=np.float64)
        b = int(st["pred_base"])
        lo = max(b, 0)
        hi = min(b + px.shape[0], n)          # exclusive; clamp at episode end
        if hi > lo:
            for k in range(3):
                pred_cols[k][lo:hi] = px[lo - b: hi - b, k]
    if n < 2:
        return
    chart.data = (t, rec[:, 0], rec[:, 1], rec[:, 2],
                  pred_cols[0], pred_cols[1], pred_cols[2])


def _set_force_recorded(chart, t, xyz) -> None:
    st = _chart_state(chart)
    st["rec_t"] = np.asarray(t, dtype=np.float64).reshape(-1)
    st["rec_xyz"] = np.asarray(xyz, dtype=np.float64)
    _push(chart)


def _show_pred_series(chart, show: bool) -> None:
    """Flip the 3 predicted series' visibility (see ``_force_series``). Reassigns
    the whole series tuple — the only way to change ``show`` on the handle — and
    only when it actually changes, to avoid redundant wire traffic."""
    if getattr(chart, "_muse_pred_shown", None) == show:
        return
    labels = getattr(chart, "_muse_labels", ("x", "y", "z"))
    chart.series = _force_series(labels, show_pred=show)
    chart._muse_pred_shown = show


def _set_force_pred(chart, base_idx, xyz) -> None:
    """Overlay the predicted force starting at recorded frame index ``base_idx``."""
    st = _chart_state(chart)
    st["pred_base"] = int(base_idx)
    st["pred_xyz"] = np.asarray(xyz, dtype=np.float64)
    _show_pred_series(chart, True)   # reveal the (now finite) predicted dashes
    _push(chart)


def _clear_force_pred(chart) -> None:
    st = _chart_state(chart)
    st["pred_base"] = None
    st["pred_xyz"] = None
    _show_pred_series(chart, False)  # hide again so all-NaN cols can't blank recorded
    _push(chart)


if __name__ == "__main__":
    raise SystemExit(main())
