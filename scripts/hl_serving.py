#!/usr/bin/env python3
"""Online HL-v2 (flow | diffusion) PLANNER — single-process viser control panel that
drives the low-level G1 controller on :8770, mirroring the muse_force_serving UX
(predict → preview → execute) but for the HL-v2 flow/diffusion checkpoints.

WHY single-process (unlike muse_force_serving's two-process backend+frontend): the
MuSe/UVA planner is forced into a hard conda split (uvaforce torch vs FLoRe-.venv
viser). The HL-v2 flow/diffusion models load in the SAME FLoRe ``.venv`` that already
holds viser + mink + mujoco (verified: torch cu130 + diffusers + viser coexist), so
the model, the viser page, the S0Driver, and the sensor threads all live in ONE
process. The base64-npz IPC, stdout-poisoning discipline, and spawn/handshake are
gone; the predict/execute state machine is direct method calls under a lock.

Reused, near-verbatim, from ``scripts/muse_force_serving_server.py`` +
``muse_force_serving_viz.py`` (the concepts are model-agnostic):
  * ``_WSSubscriber`` sensor clients (:8771 F/T, :8773 camera, :8784 gripper),
    ``_decode_jpeg_b64`` + ``_resize_center_crop_224`` (byte-identical training crop),
    ``GripperCalib`` (rad↔width↔target_frac);
  * the ``S0Driver`` :8770 publish + ``num_clients`` online gate + the IDLE HEARTBEAT
    republishing the last wire so the low-level's pre-arm pose gate can pass;
  * the 3-cadence engine (obs_hz ingest / step_hz playback / drip_hz refresh) with the
    hold-and-resend loop, and the ``img_stride`` image ring;
  * the predict→preview→execute state machine (staged plan, predict_once/run_staged/
    run_continuous/stop/reset/tare, commit-anchor-only-after-drip);
  * the LEFT-only ThreePointIK render (TCP→tracker inverse extrinsic, calibrate-once),
    force/torque charts, ONLINE/OFFLINE status, gripper drip.

NEW here (the point of this planner): the HL-v2 model outputs a per-step 6-DoF force
(physical N/N·m, Bota sensor frame) alongside pose+gripper. A **master switch** and a
**settable gate (N, default 1.5)** decide whether that force is published to the low
level as a force command (F^cmd) on the extended :8770 wire:
  * switch OFF (default) → NO F^cmd on the wire → pure position controller.
  * switch ON + ‖F‖ ≤ gate → the planner HARD-ZEROS F^cmd → still pure position.
  * switch ON + ‖F‖ > gate → F^cmd (raw sensor frame, **NEGATED**) published → the low
    level admits F^cmd AND F_meas (force tracking). The low level rotates sensor→pelvis.

⚠️ **THE MODEL PREDICTS F_meas; THE WIRE CARRIES F^cmd = −F_meas.** Everything trained
on or replayed from a demonstration is the wrench the Bota READ. The low level consumes a
COMMAND and closes ``e = F^cmd + F_meas → 0``, so the command that reproduces a demonstrated
press is the NEGATIVE of the wrench that demo measured — a press reads F_meas pointing OUT of
the table, and the command that presses into it points IN. That flip happens in exactly one
place on the publish path (``_gate_wire_force``) and in exactly one place per viewer, and
every force surface on the page (arrow, charts, readout) shows F^cmd so what is displayed is
the direction the arm will push. The one exception, labelled as such, is the "F/T sensor
(F_meas)" panel, which is a raw measurement. Derivation + witnesses:
:mod:`flore.utils.force_frame`. See :mod:`flore.s1_policy.serving.hl_online` for the gate +
decode contract, and the holosoma submodule edits (vive_live_source / holosoma_wbt) for the
F^cmd wire ingest.

Force reaching the robot needs BOTH ends to opt in, and that is deliberate: the switch
here only fills a wire field, which the low level ignores entirely unless it was launched
with ``FORCE_CMD_WIRE=1``. Neither side alone can put force on the robot. Below the gate
the low level is pure position on BOTH counts — it hard-zeros a sub-gate command
(``FORCE_CMD_WIRE_GATE_N``, defaulted to this gate) *and*, with ``GATE_CMD=1``, keeps the
measured force out of the actor obs and the servo (``GATE_CMD_N``, likewise defaulted to
this gate). So "no force commanded" means "no force loop at all", however hard the Bota is
actually being pressed. Above 40 N of measured force the low level STOPS itself
(``FORCE_ESTOP_N``, the joystick-B path); the planner sees that as its RecordTap going
stale and drops the abandoned plan (see the LOW-LEVEL-LOST watchdog in ``_status_tick``).

Runs in FLoRe's ``.venv``; reads only committed files + the checkpoint; writes nothing.
The launcher ``run_hl_serving.sh`` wires PYTHONPATH (UMI ik/viz) + ports + --algo.
"""
from __future__ import annotations

import argparse
import base64
import collections
import json
import os
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

from flore.s1_policy.data.transforms import (  # noqa: E402
    delta_ee_to_absolute as _delta_ee_to_absolute,
)
from flore.s1_policy.inference.display import (  # noqa: E402
    TRIAD_DRAW_SIZES,
    triad_offsets_from_tcp,
)
from flore.s1_policy.data.layout import ACTION_HZ  # noqa: E402
from flore.s1_policy.serving.hl_online import (  # noqa: E402
    ACTION_HORIZON,
    GRIP_MAX_AGE_S,
    OBS_MAX_AGE_S,
    LivePlanner,
    exec_step_choices,
    load_hl_model,
    replay_only_model,
    resize_center_crop_224 as _resize_center_crop_224,
    snap_exec_steps,
)
from flore.s1_policy.serving import hl_anchor, hl_reach, rollout_record  # noqa: E402
from flore.s1_policy.serving.gil_yield import YieldGate  # noqa: E402
from flore.s1_policy.serving.hl_muse import (  # noqa: E402
    MUSE_ALGO,
    MuseLivePlanner,
    load_muse_model,
    resolve_backend_python,
)
from flore.s1_policy.serving.hl_anchor import AnchorConfig, ChunkAnchor  # noqa: E402
from flore.s1_policy.serving.hl_reach import ReachCompensator, ReachConfig  # noqa: E402
from flore.s1_policy.serving.hl_record import RecordedPrediction, TakeRecorder  # noqa: E402
from flore.s1_policy.serving.hl_trace import (  # noqa: E402
    ExecutionTrace,
    ReferencePath,
    plot_trace_vs_reference,
    resample_reference_to_trace,
    trace_stats,
)
from flore.s1_policy.serving.live_robot import (  # noqa: E402
    G1_29DOF_SDK_DOF_NAMES,
    JAW_JOINT_OPEN_M,
    N_SDK_DOF,
    build_sdk_to_urdf_index,
    jaw_joint_from_width_m,
    parse_record_frame,
    remap_dof_to_urdf,
)
from flore.utils.dataset_paths import resolve_dataset_path  # noqa: E402
from flore.utils.force_frame import (  # noqa: E402
    CMD_FROM_MEAS_NOTE,
    TCP_FROM_SENSOR_NOTE,
    cmd_from_meas_force,
    cmd_from_meas_wrench,
)
from flore.utils.rot6d import (  # noqa: E402
    quat_xyzw_to_rotmat,
    rot6d_to_rotmat,
    rotmat_to_quat_xyzw,
    rotmat_to_rot6d,
)

_JAW_OPEN_M = 0.0475
_GRASP_THRESH_M = 0.08
_ROLLOUT_YELLOW = (255, 232, 20)  # predicted HL rollout path (offline viz uses orange;
                                  # the online planner uses YELLOW per the deploy spec)
_REPLAY_GREEN = (90, 220, 120)    # RECORDED episode path — the same green replay_zarr_viz.py
                                  # draws its ground-truth trajectory in, so the two viewers
                                  # read alike when comparing a demo against the policy.
_TRACE_RED = (235, 60, 60)        # EXECUTED path — where the arm ACTUALLY went (measured
                                  # RecordTap FK), drawn against the green commanded path.
                                  # Distinct from both the yellow plan and the green demo so
                                  # the three lines are never confused in a screenshot.

# --- The GHOST robot: a translucent G1 that PLAYS the predicted chunk -----------------------
# The yellow path shows WHERE the plan goes; the ghost shows the whole ROBOT moving through it —
# a semi-transparent second mesh animated along the chunk in a repeating loop, the way the
# recorded-replay tools walk a demo. It replaces the earlier "force arrows standing on the path"
# view: an operator asked to see the ACTION (the arm running the trajectory), not a static ribbon
# of force vectors. The single wrench arrow still rides the ghost's hand during the playback, so
# the predicted force is read AT the pose it acts on, exactly like a demo's F/T in replay.
#
# VIOLET, translucent (RGBA, 0-1). Deliberately none of the scene's existing hues — yellow is the
# predicted PATH, green (90,220,120) the RECORDED demo, red the EXECUTED trace — so the ghost is
# never confused with a trajectory line. The alpha reads as "a preview, not the real robot": on a
# RECORD=1 rig the SOLID mesh is the live robot and this ghost hovers over the plan it will run.
_GHOST_RGBA = (0.62, 0.36, 0.95, 0.45)

# LEFT-hand wrist→TCP reach lever (pure translation, 0.242 m along the URDF wrist +x).
# This is the URDF-side lever. It is a DIFFERENT LEG from the converter's puck→TCP
# calibration in ``robot_assets/calibration/tcp_extrinsic.json`` (whose translation is
# (0, -0.225, 0.05) in the PUCK frame); the two must NOT be conflated or set equal. Both the
# relabel M and the legacy Rx(180) preserve +x, so this lever is invariant across the model/URDF
# seam below — which is why the seam's axis error never showed up in the decoded POSITIONS.
#
# WHY 0.242 AND NOT 0.21 (this lever moves the ARM — it is not a display constant).
# 0.242 m is the measured wrist_yaw_link→fingertip-midpoint distance, and it is what the
# stored dataset point actually IS: the puck sits at wrist + (0.017, 0, 0.05) (derived from
# the 0.242 TCP and the measured puck→TCP), so
#     stored eef = puck + R_puck @ (0.225, 0, -0.05)  ⇒  wrist + 0.242·x̂,
# i.e. the trained target already lands on the 0.242 fingertip TCP. The URDF agrees
# independently (FK: fingertip meshes span x ∈ [0.1624, 0.2562], so 0.242 is inside the pad
# while 0.21 sits ~45 mm behind it), and so does the LOW LEVEL this planner publishes to —
# ``wbt_training/config_values/hybrid.py::TCP_OFFSET_X = 0.242`` (pinned by its own test),
# ``wbt_wrappers_inference/force_target_math.py``, ``wbt_realtime_eval/main.py`` default.
#
# The error does NOT cancel in the anchor-relative round trip. The planner uses this lever
# twice — measured wrist → anchor TCP (``_measured_full_wire``) and decoded TCP → published
# wrist (``WireMapper.tcp_to_wire_contact``) — and those cancel only while the orientation is
# unchanged. With a rotating chunk the residual is |(R_out − R_anchor) @ (0.032, 0, 0)|:
# measured mean 11.9 mm / max 42.6 mm over random rotations, ~17 mm worst case per chunk on
# real clean94. Purely serving-side: no dataset re-conversion and no re-training.
# Execute-N-of-S choices offered in the page. v3 grid: 50 steps at 30 Hz, so 16/30/50 are
# ~0.53 s / 1.0 s / 1.67 s of demonstration per chunk (the v2 4/8/16 were 0.25/0.5/1.0 s at
# 16 Hz). ACTION_HORIZON is included so "execute the whole chunk" is always offered, and the
# set is filtered so a re-pinned (smaller) horizon cannot offer a choice larger than the
# chunk. MODULE scope because both main() (the speed report + the exec_ctl snap) and
# _run_viser (the dropdown) read it — as a main() local, the dropdown raised NameError
# inside _run_viser's broad except, which surfaced only as a failed page load.
#
# THIS page keeps the historical (16, 30) ladder rather than the shared default, and that is a
# decision about a REVIEWED RIG DEFAULT, not taste: `--exec-steps` here defaults to 4 (see
# main()), which snaps UP to 16 against this ladder. Adding the shared ladder's 8 rung would
# silently re-snap that default from 16 to 8 — halving how much of every chunk this planner
# executes, on the left_only hardware path, as a side effect of a UI change. The whole-body
# siblings default to 16 explicitly and so are free to offer the full ladder.
_EXEC_CHOICES = exec_step_choices(ACTION_HORIZON, ladder=(16, 30))

_TCP_OFFSET = np.array([0.242, 0.0, 0.0], dtype=np.float64)      # wrist→UMI-TCP
_WIRE_OFFSET = np.array([0.18, -0.025, 0.0], dtype=np.float64)  # wrist→vr_3point contact (LEFT)

# MODEL/URDF EE-frame convention — the seam, DERIVED FROM THE DATASET, never hardcoded.
# ---------------------------------------------------------------------------
# The model predicts the LEFT EE orientation in whatever frame the CONVERTER stored, while this
# planner MEASURES the wrist (yourdfpy FK) and PUBLISHES the 3-point wire (what the low level's
# ``_compute_vr_fk`` gates + servos against) in the yam-URDF ``left_wrist_yaw_link`` frame. The
# action decode is RELATIVE (``p_abs = R_ref @ p_rel + p_ref``), so the stored frame enters ONLY
# through the per-role anchor ``R_ref`` and the ``wrt_start`` proprio channel. Keeping the
# planner's INTERNAL rotation state in the MODEL frame (so ``delta_ee_to_absolute`` and the obs
# match training) and converting URDF↔model at exactly the two hardware boundaries —
# measured-EE IN (:func:`_urdf_rot6d_to_model`) and decoded-pose OUT
# (:func:`_model_rot6d_to_urdf`) — is correct and unchanged. What the seam IS, is not.
#
# THE SEAM IS A PROPERTY OF THE DATASET, because the converter composes
# ``world_T_stored = world_T_puck @ T_tracker_tcp`` and stamps that extrinsic into the zarr:
#
#     R_stored = R_wrist @ S ,      S = _PUCK_TO_WRIST_RELABEL @ T_tracker_tcp[:3, :3]
#
# where ``_PUCK_TO_WRIST_RELABEL`` is the measured (−y, −x, −z) puck→wrist body-axis relabel
# (see the UMI submodule's docs/REPLAY_AND_FIDELITY.md §"authoritative pose chain"). On the
# CURRENT convention (extrinsic ``[M | (0, −0.225, 0.05)]``) this composes to **IDENTITY** — the
# stored pose IS the wrist-oriented fingertip TCP, so no seam is applied at all, which is the
# whole point of the 2026-08-06 frame overhaul: the low level is fed the raw-tracker action put
# through ONE extrinsic and nothing else.
#
# ⚠️ HISTORY — WHY THIS IS DERIVED AND NOT A CONSTANT. This used to be a hardcoded
# ``_RX180 = diag(1, −1, −1)`` (180° about +x). A 180° flip really was needed for the
# pre-2026-08-06 datasets, but about the WRONG AXIS: the true relabel ``M`` is 180° about
# (1, −1, 0)/√2, whose axis is 45° from +x, so the two 180° rotations composed to a spurious
# **90° about the wrist z**. That flip fixed the large y/z symptom and left a 90° error, which
# measured on clean94 as: stroke transport 107.1° where the irreducible physical
# ready-pose-vs-grip angle is 55.3°; tool attack angle (is the tool pointing where it is moving —
# for a wiping task, the task itself) off by **47.96° mean / 69.5% of samples beyond 30°**; and
# the ``wrt_start`` proprio fed to the model wrong by 2.58° mean / 26° max — right magnitude,
# wrong axis. With the derived seam all three go to ~0 (attack angle 0.000°, proprio 2e-8°,
# transport exactly the physical 55.3°). A constant cannot be right for both conventions: the
# correct seam is 180° for a pre-fix dataset and IDENTITY for a current one.
#
# The reach lever ``_TCP_OFFSET`` is +x and both M and Rx(180) preserve +x, so positions were
# never affected by the axis error; only orientation, the y-bearing ``_WIRE_OFFSET`` contact
# lever, and the proprio channel are. Only the LEFT role is decoded (``left_only`` checkpoints);
# RIGHT/HEAD are held at their MEASURED URDF wire and are never converted (converting them would
# re-break the pre-arm gate).
#
#: The measured puck→wrist body-axis relabel (new x = old −y, new y = old −x, new z = old −z).
#: A proper rotation (det=+1) and its own inverse. Mirrors the UMI submodule's
#: ``viz/collection_viz.py::_RAW_RELABEL_M`` and the rotation block of
#: ``robot_assets/calibration/tcp_extrinsic.json``; pinned by ``tests/test_hl_online.py``.
_PUCK_TO_WRIST_RELABEL = np.array([[0.0, -1.0, 0.0],
                                   [-1.0, 0.0, 0.0],
                                   [0.0, 0.0, -1.0]], dtype=np.float64)

# The live seam ``S`` (stored/model frame = wrist frame @ S). Set ONCE at startup by
# :func:`_resolve_frame_seam` from the replay dataset's / checkpoint's own ``T_tracker_tcp``, so
# it can never silently disagree with the data being served. Module-level (not threaded through
# every call) because both converters are called from three threads and the value is immutable
# after startup. Defaults to identity = the current convention, which is also the right answer
# when no extrinsic is available: apply no seam rather than a guessed one.
_MODEL_SEAM = np.eye(3, dtype=np.float64)
_MODEL_SEAM_SRC = "identity (default — no extrinsic seen)"


def _seam_from_extrinsic(t_tracker_tcp) -> np.ndarray:
    """The URDF↔model seam ``S`` implied by a dataset's ``T_tracker_tcp`` attr.

    ``S = _PUCK_TO_WRIST_RELABEL @ T_tracker_tcp[:3, :3]``, i.e. puck→wrist composed with the
    converter's own puck→stored rotation, so ``R_stored = R_wrist @ S``. Returns identity for a
    missing/garbled attr — applying NO seam is the safe default (it is the correct answer for
    every current-convention dataset, and a guessed one is what caused the 90° error).

    Raises ``ValueError`` if the attr is present but not a valid rotation, so a corrupt
    extrinsic is an operator-visible error rather than a silently wrong wire.
    """
    if t_tracker_tcp is None:
        return np.eye(3, dtype=np.float64)
    matrix = np.asarray(t_tracker_tcp, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError(
            f"T_tracker_tcp is not a finite 4x4 matrix (shape {matrix.shape}) — refusing to "
            "guess the URDF<->model seam, which decides the commanded ORIENTATION.")
    rot = matrix[:3, :3]
    if not (np.allclose(rot @ rot.T, np.eye(3), atol=1e-6) and np.linalg.det(rot) > 0):
        raise ValueError(
            f"T_tracker_tcp rotation block is not a proper rotation:\n{np.round(rot, 6)}\n"
            "The URDF<->model seam is derived from it, so this would command a skewed "
            "orientation. Re-convert the dataset.")
    return (_PUCK_TO_WRIST_RELABEL @ rot).astype(np.float64)


def _set_frame_seam(seam: np.ndarray, source: str) -> None:
    """Install the process-wide URDF↔model seam. Called once, before any decode."""
    global _MODEL_SEAM, _MODEL_SEAM_SRC
    _MODEL_SEAM = np.asarray(seam, dtype=np.float64).reshape(3, 3).copy()
    _MODEL_SEAM_SRC = str(source)


def _urdf_rot6d_to_model(rot6d: np.ndarray) -> np.ndarray:
    """URDF wrist-frame rot6d → MODEL/stored-frame rot6d (right-multiply the seam ``S``).
    Applied to a MEASURED wrist orientation before it becomes the model's decode anchor /
    ``wrt_start`` base."""
    return rotmat_to_rot6d(
        rot6d_to_rotmat(np.asarray(rot6d, dtype=np.float64)) @ _MODEL_SEAM).astype(np.float64)


def _model_rot6d_to_urdf(rot6d: np.ndarray) -> np.ndarray:
    """MODEL/stored-frame rot6d → URDF wrist-frame rot6d (right-multiply ``S``ᵀ). Applied to a
    DECODED LEFT orientation before it is published on the URDF-frame wire.

    NOTE the transpose: the two directions are NOT the same expression. The old ``_RX180`` was
    self-inverse so a single shared implementation happened to work; a general seam is not, and
    using ``S`` here would double the rotation instead of undoing it."""
    return rotmat_to_rot6d(
        rot6d_to_rotmat(np.asarray(rot6d, dtype=np.float64)) @ _MODEL_SEAM.T).astype(np.float64)

_HOME_ANCHOR_PATH = os.path.join(
    _FLORE_ROOT, "third_party", "holosoma_extensions", "src", "extensions", "wbt_wrappers",
    "wbt_realtime_eval", "wbt_realtime_eval", "vive", "home_anchor.json")


def _log(msg: str) -> None:
    print(f"[hl-serving] {msg}", file=sys.stderr, flush=True)


# ===========================================================================
# Sensor subscriber + decode helpers (copied verbatim from muse_force_serving_server)
# ===========================================================================
class _WSSubscriber(threading.Thread):
    """Auto-reconnecting read-only WebSocket client keeping the latest decoded JSON.

    ``send`` queues control frames (F/T tare, gripper target_frac); a ``target_frac``
    coalesces with a pending one (position latch) so a disconnected bridge can't grow
    the queue unbounded, while discrete commands (tare) are never dropped.

    ``on_msg`` (optional) is invoked with ``(msg, t_monotonic)`` for EVERY arriving frame,
    on this thread, before the frame is latched. ``latest()`` only ever exposes the newest
    frame, which is right for the camera/gripper/record taps (the loops want "current
    state") but WRONG for the F/T: the model's force bin is an anti-aliased mean over every
    raw sample in the window, so sampling ``latest()`` once per 30 Hz tick decimates a
    204 Hz signal and hands the encoder a differently-filtered input than it trained on.
    The callback is how the force ring sees all of them. Exceptions are swallowed — a
    consumer bug must not kill the socket that every loop depends on."""

    def __init__(self, url: str, name: str, on_msg=None):
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
        self._on_msg = on_msg

    def latest(self) -> tuple[dict | None, float, int]:
        with self._lock:
            return (self._latest, self._latest_wall, self._n)

    def send(self, obj: dict) -> None:
        payload = json.dumps(obj)
        with self._lock:
            if "target_frac" in obj and self._send_q and "target_frac" in self._send_q[-1]:
                self._send_q[-1] = payload
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
                    # compression=None: every bridge on this box publishes either tiny JSON
                    # (F/T, jaw, RecordTap) or base64'd JPEG — already-compressed bytes. On the
                    # camera stream permessage-deflate bought 24% of size (69 kB -> 52 kB) for
                    # 1.84 ms of CPU per frame on the BRIDGE and 0.19 ms here, i.e. 5.5% of a
                    # core in the camera bridge process at 30 Hz, to move data over loopback.
                    # Compression is NEGOTIATED, so a client that does not offer it disables it
                    # for this connection at both ends — that is why turning it off here is what
                    # unloads the bridges too, without touching them.
                    async with websockets.connect(self.url, max_size=None, ping_interval=None,
                                                  compression=None) as ws:
                        self.connected = True
                        _log(f"{self._name}: connected {self.url}")
                        while not self._stop.is_set():
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
                            if self._on_msg is not None:
                                # Stamped HERE, at arrival, not at consumption: this is the
                                # closest we get to the sample's measurement time, and the
                                # force bin edges are differences of these stamps.
                                try:
                                    self._on_msg(msg, time.monotonic())
                                except Exception as exc:  # noqa: BLE001 — never kill the socket
                                    _log(f"{self._name}: on_msg raised {exc!r} (frame dropped)")
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


def _limit_cpu_thread_pools(device: str) -> None:
    """Stop OpenCV and torch from fanning this planner's small CPU ops across every core.

    Both default to one thread per core (measured 24 here, nothing in the repo set them).
    That is the right default for batch work and the wrong one for a 30 Hz control loop made
    of many SMALL operations: the per-op parallel launch + join costs more than it saves, and
    the idle pool threads spin.

    Measured on the live camera path (``_decode_jpeg_b64`` + ``_resize_center_crop_224`` at
    1280x720, the per-tick cost at ``img_stride=1``):

        cv2 threads=24  ->  wall 6.66 ms, CPU 13.06 ms   (x1.96 = ~2 cores)
        cv2 threads=1   ->  wall 6.64 ms, CPU  6.64 ms   (x1.00)

    i.e. the fan-out DOUBLED the CPU cost for a 0.3% latency change — at 30 Hz that is 39% of
    a core spent on nothing. Pinned to 1.

    torch's pool is left alone when the model is on the GPU (an ACT forward there measured
    x1.00 wall:CPU, so its threads are already idle) but must NOT be pinned for a CPU device,
    where the matmuls genuinely want the cores."""
    try:
        import cv2

        cv2.setNumThreads(1)
    except Exception as exc:  # noqa: BLE001 — an unthreaded build is fine, just say so
        _log(f"could not pin OpenCV threads ({exc!r}); camera decode may burn extra cores")
    if str(device).startswith("cuda"):
        import torch

        # 4, not 1: the obs assembly's small torch ops are cheap either way, and leaving a
        # little parallelism costs nothing while keeping a --device cpu typo from crawling.
        torch.set_num_threads(min(4, torch.get_num_threads()))


def _decode_jpeg_b64(b64: str) -> np.ndarray | None:
    import cv2

    try:
        buf = np.frombuffer(base64.b64decode(b64), dtype=np.uint8)
        bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if bgr is None:
            return None
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    except Exception:  # noqa: BLE001
        return None


class GripperCalib:
    """Encoder-rad ↔ width-m ↔ target_frac 2-point map (see muse_force_serving_server)."""

    def __init__(self, width_open_m: float = 0.095, width_closed_m: float = 0.0):
        self.width_open_m = float(width_open_m)
        self.width_closed_m = float(width_closed_m)

    def to_width_m(self, actual_rad: float, open_rad: float, closed_rad: float) -> float:
        denom = open_rad - closed_rad
        if abs(denom) < 1e-6:
            return self.width_closed_m
        frac = float(np.clip((actual_rad - closed_rad) / denom, 0.0, 1.0))
        return self.width_closed_m + frac * (self.width_open_m - self.width_closed_m)

    def to_target_frac(self, width_m: float) -> float:
        span = self.width_open_m - self.width_closed_m
        if abs(span) < 1e-9:
            return 0.0
        return float(np.clip((float(width_m) - self.width_closed_m) / span, 0.0, 1.0))

    def to_target_frac_squeezed(
        self, width_m: float, *, close_below_m: float, overshoot_m: float
    ) -> tuple[float, bool]:
        """``to_target_frac`` with a CLOSING OVERSHOOT, so a grasp actually grips.

        Returns ``(frac, squeezing)``.

        WHY THIS IS NEEDED — measured on ``session_2026.08.08_2228_v3``, not guessed. The
        recorded gripper action is bimodal, and its "closed" mode is **42-48 mm, never 0**:

            closed cluster (<70 mm):  35% of anchors, mean 47.7 mm, min 42.1 mm
            open   cluster (>=70 mm): 65% of anchors, mean 91.6 mm

        That is the physically correct recording — during collection the jaw closed ONTO THE
        TOOL, so the logged width IS the tool's grip thickness. But replaying it as a POSITION
        target means commanding the jaw to stop exactly at the object's surface: ``frac =
        0.0421/0.095 = 0.44``, i.e. 44% of travel, arriving with no residual squeeze. On a
        position-controlled jaw that is a touch, not a grasp — the reported "夹不紧".

        So when the commanded width is in the closed mode (``width_m < close_below_m``) we
        subtract ``overshoot_m`` from the TARGET only. The jaw drives past the object's real
        surface, stalls against it, and the position error becomes grip force. This is the
        standard way to get force out of a position-only jaw.

        THREE THINGS THIS DELIBERATELY DOES NOT DO:
          * It never touches the OPEN mode — an overshoot while opening would just under-open.
          * It never touches :meth:`to_width_m`, so the PROPRIOCEPTION the model observes stays
            the true measured width. Feeding a squeezed value back as observation would put the
            model out of distribution on its own gripper channel.
          * It is applied at the publish boundary only, so the yellow preview path, the charts
            and the recorded-replay comparison keep showing the model's ACTUAL prediction.
        """
        span = self.width_open_m - self.width_closed_m
        if abs(span) < 1e-9:
            return 0.0, False
        w = float(width_m)
        squeezing = w < float(close_below_m)
        if squeezing:
            w -= float(overshoot_m)
        return float(np.clip((w - self.width_closed_m) / span, 0.0, 1.0)), squeezing


def new_grip_ctl(*, overshoot_mm: float, close_below_mm: float, confirm_steps: int,
                 hold_tightest: bool = True) -> dict:
    """ONE jaw's control+state dict, in the units and under the key names that
    :func:`build_grip_command` reads.

    A CONSTRUCTOR rather than a dict literal at each call site, because the state machine reads
    its dict with ``[...]`` and the units are not the CLI's: the flags are MILLIMETRES and the
    keys here are ``overshoot_m`` / ``close_below_m`` in METRES. The 3-tracker planner spelled
    them ``overshoot_mm`` / ``close_below_mm`` and left out the run-length state entirely, which
    is invisible until the first jaw command of the first Run raises ``KeyError('overshoot_m')``
    — after step 0 is already on the wire and before the chunk's anchor commits. One
    constructor next to the consumer makes that class of drift impossible.

    ⚠️ **ONE DICT PER HAND.** The run-length state (``engaged`` / ``run`` / ``run_closing`` /
    ``held_frac``) lives in this dict, NOT in the closure, so two hands sharing one dict share
    one confirm counter and one tightest-width latch: the left jaw's sustained close would
    engage the right jaw's overshoot. Call this once per gripper role.
    """
    return {
        "overshoot_m": float(overshoot_mm) / 1000.0,
        "close_below_m": float(close_below_mm) / 1000.0,
        "confirm_steps": int(confirm_steps),
        "hold_tightest": bool(hold_tightest),
        # Run-length state: `engaged` = the overshoot is currently applied; `run` = how many
        # consecutive steps have agreed on `run_closing`.
        "engaged": False, "run": 0, "run_closing": False, "held_frac": None,
        "last_frac": None, "squeezing": False,
    }


def build_grip_command(calib: GripperCalib, grip_ctl: dict, grip_lock) -> tuple:
    """Build the jaw-command state machine → ``(to_frac, release_latch)``.

    MODULE LEVEL rather than a closure inside :func:`main`, specifically so tests can drive the
    REAL state machine. That is not tidiness: the first version of these tests replayed the
    latch logic inline and therefore PASSED while the shipped latch was disabled, the
    hysteresis was collapsed to one threshold, and the latch froze the first value instead of
    the tightest — three mutations, all missed. Now the test calls this.

    ⚠ WHY IT IS A STATE MACHINE AND NOT A PER-STEP FUNCTION — the reported "grips, then
    loosens". The cause is mostly NOT the overshoot; it is temporal ensembling AVERAGING A
    BIMODAL SIGNAL. The gripper action has two modes (closed 42-48 mm, open 84-94 mm, nothing
    between), so a blend over up to 50 survivors sits at the weighted MIXTURE whenever they
    disagree. Measured on the real cluster means: one brief open flicker inside an established
    grasp walks the commanded target 0.397 → 0.498, i.e. **10.1% of travel of slow re-opening**,
    6.6% of it pure loosening — while every individual prediction still looks reasonable.

    THE RULE (deliberately simple): the overshoot follows the policy's *sustained intent*.

    * ``confirm_steps`` consecutive CLOSE predictions → engage: publish the predicted width
      minus ``overshoot_m``, so the jaw drives past the object and stalls into a grip.
    * ``confirm_steps`` consecutive OPEN predictions → release: cancel the overshoot and
      follow the prediction verbatim, so opening is never held back.
    * Anything shorter than that — a single flickering step either way — does not change state.

    The run-length requirement is what rejects the blend's transient disagreement, and because
    the overshoot is a constant offset (not a frozen target), a grasp that genuinely wants to
    adjust its width still can. ``confirm_steps = 1`` reduces this to per-step behavior.
    """

    def to_frac(width_m: float) -> tuple[float, bool]:
        with grip_lock:
            over = float(grip_ctl["overshoot_m"])
            below = float(grip_ctl["close_below_m"])
            need = max(1, int(grip_ctl["confirm_steps"]))
            hold_tightest = bool(grip_ctl.get("hold_tightest", True))
            engaged = bool(grip_ctl["engaged"])
            run = int(grip_ctl["run"])
            run_closing = bool(grip_ctl["run_closing"])
            held = grip_ctl.get("held_frac")

        w = float(width_m)
        closing = w < below  # this step's INTENT, before any confirmation

        # Count how many consecutive steps have agreed on the same intent.
        if closing == run_closing:
            run += 1
        else:
            run, run_closing = 1, closing
        # Flip only once the intent has been sustained long enough.
        if run >= need and closing != engaged:
            engaged = closing

        # ±inf makes the engage decision OURS (the confirmed state above) rather than letting
        # the calibration re-derive it per step from a bare threshold.
        frac, _ = calib.to_target_frac_squeezed(
            w, close_below_m=(float("inf") if engaged else -float("inf")), overshoot_m=over)

        # ⚠ SECOND, INDEPENDENT CAUSE OF LOOSENING — measured, and the confirmation run above
        # cannot fix it. During the flicker the blended width goes 47.7 -> 53.9 mm but NEVER
        # crosses the 70 mm threshold, so the intent stays "closing" the whole time: there is no
        # flip to reject. The command loosens because the WIDTH ITSELF drifts, the blend having
        # averaged in the open-mode survivors. With a 10 mm overshoot that walks the target from
        # 37.7 mm to 43.9 mm — back onto the object's own 42-48 mm surface, i.e. grip force
        # decaying to ~0. So while the grasp is CONFIRMED CLOSED we also refuse to loosen the
        # target (6.6% of travel -> 0.0%). Set grip_ctl['hold_tightest'] False for the plain
        # "overshoot follows the sustained intent" rule with no width guard.
        if engaged and hold_tightest:
            frac = frac if held is None else min(frac, float(held))
            held = frac
        elif not engaged:
            held = None

        with grip_lock:
            grip_ctl["engaged"] = engaged
            grip_ctl["run"] = run
            grip_ctl["run_closing"] = run_closing
            grip_ctl["held_frac"] = held
            grip_ctl["last_frac"] = frac
            grip_ctl["squeezing"] = engaged
        return frac, engaged

    def release_latch() -> None:
        """Drop the grasp state. Called where the jaw is deliberately opened (Reset start pose /
        shutdown) so the next grasp starts from OPEN and has to earn its confirmation run again,
        rather than inheriting an engaged squeeze from the previous episode."""
        with grip_lock:
            grip_ctl["engaged"] = False
            grip_ctl["run"] = 0
            grip_ctl["run_closing"] = False
            grip_ctl["held_frac"] = None
            grip_ctl["squeezing"] = False

    return to_frac, release_latch


# ===========================================================================
# Abandoned-intent handoff for the ACT temporal ensembler.
#
# The ensembler stores up to H=50 chunk predictions and blends them with exp(-k*i) weights
# ordered OLDEST-FIRST, so survivors of an abandoned rollout OUTVOTE a fresh query (49
# survivors + 1 new ⇒ the new one is ~1.5% of the blend). Anything that abandons the intent —
# ⏹ Stop, ↺ Reset start pose, or the low level being stopped from the joystick — must therefore
# drop them, or the next Run resumes the trajectory the operator just aborted.
#
# The complication is ownership: the ensembler belongs to whichever thread holds ``busy`` (the
# drip loop calls it once per step), while Stop runs on a viser callback thread and the
# low-level watchdog on the status thread. Those two can only leave a REQUEST, which the drip
# loop consumes at its own entry. Module level, not closures in :func:`main`, for the same
# reason :func:`build_grip_command` is: so the tests drive the shipped protocol.
# ===========================================================================
def request_ens_reset(ens_ctl: dict | None, ens_lock) -> None:
    """Ask the drip loop to drop every stored ACT prediction before its next execute.

    Inert when there is no ensembler (flow/diffusion/replay-only sessions pass ``None``), so
    callers never have to branch on the algorithm.
    """
    if ens_ctl is None or ens_lock is None:
        return
    with ens_lock:
        ens_ctl["pending_reset"] = True


def take_pending_ens_reset(ens_ctl: dict | None, ens_lock) -> bool:
    """Consume a pending reset request — True at most ONCE per :func:`request_ens_reset`.

    Single-shot is load-bearing, not hygiene. A sequential run calls the drip loop once per
    chunk; if the flag survived being read, every chunk after a single Stop would re-empty the
    ensembler, leaving one survivor per blend — temporal ensembling silently disabled while the
    page still reported it ON.
    """
    if ens_ctl is None or ens_lock is None:
        return False
    with ens_lock:
        pending = bool(ens_ctl.get("pending_reset"))
        ens_ctl["pending_reset"] = False
    return pending


# ===========================================================================
# Wire mapper: HL per-role absolute poses → vr_3point wire (LEFT-canonical, 3pt).
# Deploy-specific (the TCP↔contact lever + home seed) — kept OUT of the model core.
# ===========================================================================
class WireMapper:
    """Seeds the planner's LEFT anchor from home_anchor.json and converts the planner's
    absolute per-role LEFT UMI-TCP pose ↔ the vr_3point contact-point wire.

    LEFT-canonical, single-hand: the HL ``left_only`` checkpoint predicts the LEFT hand
    only; RIGHT + HEAD are held at the ready-pose home for the whole run (their wire
    slots never move), exactly as the muse planner does. The planner tracks the LEFT
    UMI-TCP point (the model's native frame); the fixed wrist→contact lever difference
    is applied only at wire publish, since the extrinsic is a pure translation and the
    wrist orientation is shared by both points."""

    def __init__(self, home_anchor: dict):
        self.home_pos_wire = np.asarray(home_anchor["home_pos_pelvis_3pt"], dtype=np.float64)      # (3,3)
        self.home_quat_xyzw = np.asarray(home_anchor["home_orn_quat_xyzw_3pt"], dtype=np.float64)  # (3,4)
        # Per-body wrist→contact offsets [L, R, HEAD] (pelvis frame), the SAME levers the low
        # level's _compute_vr_fk applies (home_anchor.vr_3point_offsets == the low-level
        # VR_3POINT_OFFSETS). Used to reconstruct the robot's MEASURED 3-point wire from live
        # joint FK so the idle wire tracks the real robot instead of this file's (possibly
        # stale, cross-policy) home pose.
        self.vr_offsets = np.asarray(
            home_anchor.get("vr_3point_offsets", [[0.18, -0.025, 0.0], [0.18, 0.025, 0.0], [0.0, 0.0, 0.35]]),
            dtype=np.float64).reshape(3, 3)
        # The LEFT wrist→contact lever is ALSO hardcoded as the module constant _WIRE_OFFSET
        # (used by tcp_to_wire_contact / left_home_tcp_anchor to build the broadcast LEFT
        # contact). It MUST equal vr_offsets[0] (the low level's own VR_3POINT_OFFSETS[0], which
        # the pre-arm gate compares against) — otherwise the LEFT broadcast point drifts off the
        # gate's LEFT point and re-introduces the exact "per-point gap → REFUSE" bug this class
        # fixes, silently and LEFT-only. Fail loudly if a future home_anchor.json / rig retune
        # diverges them.
        if not np.allclose(_WIRE_OFFSET, self.vr_offsets[0], atol=1e-6):
            raise SystemExit(
                f"WireMapper: _WIRE_OFFSET {_WIRE_OFFSET.tolist()} != home_anchor "
                f"vr_3point_offsets[0] {self.vr_offsets[0].tolist()} — the LEFT broadcast contact "
                "would not match the low-level pre-arm gate (LEFT arm would refuse to arm). "
                "Reconcile the LEFT wrist→contact lever across _WIRE_OFFSET, home_anchor.json, "
                "and the low-level VR_3POINT_OFFSETS.")
        # MUTABLE RIGHT+HEAD hold pose for the EXECUTED chunk (LEFT is always overwritten from
        # the model per step). Defaults to the home anchor; re-synced to the robot's MEASURED
        # R/H at each Predict so the held (un-predicted) hands never fly to a stale home target.
        self.hold_pos_wire = self.home_pos_wire.copy()      # (3,3)
        self.hold_quat_xyzw = self.home_quat_xyzw.copy()    # (3,4)

    def set_hold_wire(self, wire_pos9: np.ndarray, wire_orn12: np.ndarray) -> None:
        """Set the RIGHT+HEAD hold pose from a full measured 3-point wire (pos[9], orn[12]).
        Only slots 1 (right) and 2 (head) are consulted by :meth:`full_wire`; slot 0 (left) is
        stored for completeness but always overwritten by the model's LEFT prediction."""
        self.hold_pos_wire = np.asarray(wire_pos9, dtype=np.float64).reshape(3, 3).copy()
        self.hold_quat_xyzw = np.asarray(wire_orn12, dtype=np.float64).reshape(3, 4).copy()

    def left_home_tcp_anchor(self) -> tuple[np.ndarray, np.ndarray]:
        """LEFT ready-pose contact point (wire) → LEFT UMI-TCP anchor (pos, rot6d)."""
        R = quat_xyzw_to_rotmat(self.home_quat_xyzw[0])
        wrist = self.home_pos_wire[0] - R @ _WIRE_OFFSET
        tcp = wrist + R @ _TCP_OFFSET
        return tcp.astype(np.float64), rotmat_to_rot6d(R).astype(np.float64)

    def tcp_to_wire_contact(self, tcp_pos: np.ndarray, rot6d: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """LEFT UMI-TCP pose → LEFT vr_3point contact point (pos, quat_xyzw)."""
        R = rot6d_to_rotmat(np.asarray(rot6d, dtype=np.float64))
        wrist = np.asarray(tcp_pos, dtype=np.float64) - R @ _TCP_OFFSET
        contact = wrist + R @ _WIRE_OFFSET
        return contact.astype(np.float64), rotmat_to_quat_xyzw(R).astype(np.float64)

    def full_wire(self, left_tcp_pos: np.ndarray, left_rot6d: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """A full vr_3point wire: LEFT from the given TCP pose, RIGHT + HEAD held at the last
        synced hold pose (the robot's MEASURED R/H when RECORD=1; else the home anchor). Holding
        R/H at the measured pose is what keeps the un-predicted arm from flying to a stale
        cross-policy home target when the operator arms + executes."""
        lc_pos, lc_quat = self.tcp_to_wire_contact(left_tcp_pos, left_rot6d)
        wire_pos = np.concatenate([lc_pos, self.hold_pos_wire[1], self.hold_pos_wire[2]])
        wire_orn = np.concatenate([lc_quat, self.hold_quat_xyzw[1], self.hold_quat_xyzw[2]])
        return wire_pos.astype(np.float64), wire_orn.astype(np.float64)


def _commit_left_step(planner: LivePlanner, tcp_pos: np.ndarray, tcp_rot6d: np.ndarray) -> None:
    """Commit ONLY the LEFT role's anchor to an executed step (RIGHT/HEAD stay home).
    LEFT-canonical: the model predicts the LEFT hand; the others are held."""
    planner.commit_role("left", tcp_pos, tcp_rot6d)


def _seed_planner_anchor(planner: LivePlanner, mapper: WireMapper) -> None:
    """Seed every role's anchor: LEFT at its home UMI-TCP pose; RIGHT/HEAD at their home
    contact pose (they are held there and never predicted, but the planner needs a valid
    per-role anchor to decode the reserved columns)."""
    n = len(planner.roles)
    ap = np.zeros((n, 3), dtype=np.float64)
    ar = np.zeros((n, 6), dtype=np.float64)
    left_pos, left_rot6d = mapper.left_home_tcp_anchor()
    for ri, role in enumerate(planner.roles):
        if role == "left":
            ap[ri] = left_pos
            # The decode anchor is the model's DECODE FRAME (puck), not the URDF wrist the home
            # anchor is stored in — convert so delta_ee_to_absolute + wrt_start match training.
            ar[ri] = _urdf_rot6d_to_model(left_rot6d)
        else:
            # right(1)/head(2) home contact point + orientation (held for the whole run).
            slot = {"right": 1, "head": 2}[role]
            ap[ri] = mapper.home_pos_wire[slot]
            ar[ri] = rotmat_to_rot6d(quat_xyzw_to_rotmat(mapper.home_quat_xyzw[slot]))
    planner.seed_anchor(ap, ar)


# ===========================================================================
# F/T frame helper: extract the per-force-hand raw sensor wrench for force_history +
# the published F^cmd, mirroring the training force_history source (raw bota_sensor).
# ===========================================================================
def _pick_force_channels(fmsg: dict | None, force_roles: tuple[str, ...], require_6dof: bool) -> np.ndarray | None:
    """Assemble the (n_force_hands*6,) raw sensor-frame wrench bin for force_history from
    a F/T bridge message. Prefers the 6-DoF gravity-only ``*_pipe`` key (the training
    distribution); returns None if a required 6-DoF pipe is absent (so the caller freezes
    on the last good sample rather than feeding an OOD torque-zeroed vector).

    The bridge message (force_bridge.py) publishes ``{"left":[Fx,Fy,Fz], "right":[...],
    "left_pipe":[6], "right_pipe":[6]}``; ``*_pipe`` is the 6-DoF gravity-only tared
    wrench (present with ``--pipeline``). This mirrors muse ``_pick_hand_force``."""
    if fmsg is None:
        return None
    out = []
    for role in force_roles:
        pipe = fmsg.get(f"{role}_pipe")
        if pipe is not None and len(pipe) == 6:
            w = np.asarray(pipe, dtype=np.float64)
        elif require_6dof:
            return None  # 6-DoF required but pipe missing → freeze (no OOD zero-pad)
        else:
            three = fmsg.get(role)
            if three is None or len(three) != 3:
                return None
            w = np.concatenate([np.asarray(three, dtype=np.float64), np.zeros(3)])
        if not np.all(np.isfinite(w)):
            return None
        out.append(w)
    return np.concatenate(out) if out else None


#: Scrolling window (s) and emit rate (Hz) of the LIVE F_meas charts. 10 s covers a whole press
#: at the 30 Hz action grid (50 steps = 1.67 s) with the approach and release around it; 30 Hz is
#: the render loop's own rate, so the trace is as smooth as the page can show it and no smoother.
FT_CHART_WINDOW_S = 10.0
FT_CHART_HZ = 30.0


class _FtDisplayTrace:
    """Bounded, anti-aliased scrolling history of the live :8771 wrench, for the F_meas charts.

    Fed on the F/T SOCKET THREAD (``_force_on_msg``), which sees every frame at the sensor's own
    ~200 Hz, and read on the RENDER thread. That split is the whole point: sampling the
    latest-value register at the 30 Hz render rate — which is what the collection viz does — would
    ALIAS a 200 Hz signal, so a contact spike could show up at the wrong amplitude or not at all.
    Instead each emitted point is the MEAN of every raw sample in its interval, the same
    anti-aliased interval-mean the converter uses to put F/T on the camera grid. Peak ‖F‖ per
    interval is kept alongside, so a spike that a mean would flatten is still reportable.

    Deliberately independent of what the MODEL consumes: it uses the display roles, tolerates a
    3-DoF bridge (torque zero-filled), and keeps running for a pure-position checkpoint whose
    ``force_roles`` is empty. The model's own ring is a separate consumer of the same callback.
    """

    def __init__(self, window_s: float = FT_CHART_WINDOW_S, hz: float = FT_CHART_HZ) -> None:
        self._period = 1.0 / float(hz)
        self._buf: collections.deque = collections.deque(maxlen=int(round(window_s * hz)) + 1)
        self._lock = threading.Lock()
        self._sum = np.zeros(6, dtype=np.float64)
        self._peak = 0.0
        self._n = 0
        self._t0: float | None = None
        self._next = 0.0

    def push(self, wrench6: np.ndarray | None, t_mono: float) -> None:
        """Accumulate one raw sample (socket thread). ``None`` is ignored, not zero-filled — an
        unreadable frame means "unknown", and charting it as 0 N would read as "no contact"."""
        if wrench6 is None:
            return
        w = np.asarray(wrench6, dtype=np.float64).reshape(-1)[0:6]
        if w.shape[0] != 6 or not np.all(np.isfinite(w)):
            return
        with self._lock:
            if self._t0 is None:
                self._t0 = float(t_mono)
                self._next = float(t_mono) + self._period
            self._sum += w
            self._peak = max(self._peak, float(np.linalg.norm(w[0:3])))
            self._n += 1
            if t_mono >= self._next:
                self._buf.append((float(t_mono) - self._t0, self._sum / self._n, self._peak))
                self._sum = np.zeros(6, dtype=np.float64)
                self._peak = 0.0
                self._n = 0
                # Re-base rather than accumulate the period, so a stalled bridge resumes on the
                # next frame instead of dumping a burst of catch-up points.
                self._next = float(t_mono) + self._period
        return

    def series(self) -> tuple[np.ndarray, np.ndarray, float] | None:
        """``(t_rel (K,), wrench (K,6), peak_force_n)`` over the window, or None below 2 points."""
        with self._lock:
            if len(self._buf) < 2:
                return None
            rows = list(self._buf)
        times = np.asarray([r[0] for r in rows], dtype=np.float64)
        wrench = np.stack([r[1] for r in rows]).astype(np.float64)
        return times, wrench, float(max(r[2] for r in rows))


#: The puck→TCP extrinsic translation of the CURRENT (post-2026-08-06) convention, in the puck
#: body frame. Together with :data:`_PUCK_TO_WRIST_RELABEL` this is the whole contract this
#: planner serves; a dataset carrying anything else is refused by :func:`_install_frame_seam`.
_CURRENT_EXTRINSIC_T = np.array([0.0, -0.225, 0.05], dtype=np.float64)


def _checkpoint_zarr(checkpoint: str) -> str | None:
    """The first zarr a checkpoint declares it was trained on, resolved on THIS machine.
    None if absent.

    Since training moved to a remote workstation the recorded path is typically that box's
    (``/data/flore/...``), so it is relocated by basename into the local dataset dir —
    otherwise the seam derivation would exit on a file that is right here under another
    path. The relocation cannot smuggle in a wrong frame: the seam comes from whatever file
    is opened, and a pre-2026-08-06 extrinsic is still hard-refused below.
    """
    try:
        cfg = json.load(open(os.path.join(checkpoint, "config.json")))
    except (OSError, ValueError):
        return None
    paths = cfg.get("hl_data", {}).get("zarr_paths") or []
    if not paths:
        return None
    return resolve_dataset_path(str(paths[0]), repo_root=_FLORE_ROOT, log=_log)


def _install_frame_seam(args, replay_only: bool) -> None:
    """Derive + install the URDF↔model seam from the dataset actually being served, and REFUSE
    a pre-2026-08-06-convention dataset.

    The seam decides the commanded ORIENTATION, so it is resolved from the data rather than
    assumed (see the :data:`_MODEL_SEAM` block for the 90°-error history that motivates this).
    Source preference: the ``--replay-zarr`` file when replaying (that is the motion being
    executed), else the checkpoint's own training zarr.

    HARD REFUSAL of the old convention is deliberate, not conservatism. A pre-fix dataset's
    ``robot0_eef_pos`` is 33 cm off in position and 180° off in rotation (the wrist-frame lever
    applied in the puck frame — UMI ``docs/REPLAY_AND_FIDELITY.md`` §"Step ④ was wrong"), so the
    stored pose is not the point the low level's ``TCP_OFFSET_X`` lever assumes. A "correct" seam
    would fix the orientation and still command a trajectory anchored to the wrong physical
    point, which is worse than refusing: it looks right. Re-convert instead.
    """
    src_path = None
    if args.replay_zarr and args.replay_zarr != "checkpoint":
        src_path = args.replay_zarr
    elif args.checkpoint:
        src_path = _checkpoint_zarr(args.checkpoint)
    if src_path is None:
        if replay_only:
            raise SystemExit(
                "cannot resolve the URDF<->model frame seam: no dataset to derive it from. "
                "Pass --replay-zarr with an explicit .zarr.zip path.")
        _log("WARNING: no training zarr declared by the checkpoint — assuming the CURRENT "
             "convention (no seam). Verify the commanded orientation before arming.")
        return
    try:
        import zarr as _zarr

        attrs = dict(_zarr.open(str(src_path), "r").attrs)
    except Exception as exc:  # noqa: BLE001 — a bad path is an ordinary operator mistake
        raise SystemExit(f"cannot read {src_path} to derive the frame seam: {exc}") from exc
    t_ext = attrs.get("T_tracker_tcp")
    if t_ext is None:
        raise SystemExit(
            f"{src_path}: no T_tracker_tcp attr, so the URDF<->model seam cannot be derived and "
            "the commanded orientation would be a guess. Re-convert with a calibrated extrinsic.")
    matrix = np.asarray(t_ext, dtype=np.float64)
    if matrix.shape == (4, 4) and np.allclose(matrix[:3, :3], np.eye(3), atol=1e-6):
        raise SystemExit(
            f"{src_path}: PRE-2026-08-06 dataset (T_tracker_tcp has an IDENTITY rotation, "
            f"translation {np.round(matrix[:3, 3], 4).tolist()}). Its robot0_eef_pos is ~33 cm "
            "off in position and 180 deg off in rotation, because a WRIST-frame lever was "
            "applied in the PUCK frame. This planner serves the corrected convention only: "
            f"rotation = the (-y,-x,-z) relabel, translation {_CURRENT_EXTRINSIC_T.tolist()}. "
            "RE-CONVERT the session (processing/convert_to_umi.py --hands left --hl-windows) and "
            "re-train, then replay the new file. See the UMI submodule's "
            "docs/REPLAY_AND_FIDELITY.md 'authoritative pose chain'.")
    try:
        seam = _seam_from_extrinsic(matrix)
    except ValueError as exc:
        raise SystemExit(f"{src_path}: {exc}") from exc
    if not np.allclose(matrix[:3, 3], _CURRENT_EXTRINSIC_T, atol=1e-6):
        _log(f"WARNING: {os.path.basename(str(src_path))} puck->TCP translation "
             f"{np.round(matrix[:3, 3], 4).tolist()} != the current convention's "
             f"{_CURRENT_EXTRINSIC_T.tolist()}; the stored pose is a DIFFERENT physical point on "
             "the tool, so the commanded motion is anchored a few mm/cm off.")
    _set_frame_seam(seam, f"{os.path.basename(str(src_path))}::T_tracker_tcp")
    if np.allclose(seam, np.eye(3), atol=1e-9):
        _log("frame seam: IDENTITY (current convention — the stored pose IS the wrist-oriented "
             f"fingertip TCP, so NO seam is applied). source: {_MODEL_SEAM_SRC}")
    else:
        angle = float(np.degrees(np.arccos(np.clip((np.trace(seam) - 1.0) / 2.0, -1.0, 1.0))))
        _log(f"frame seam: NON-identity, {angle:.2f} deg (derived from {_MODEL_SEAM_SRC}) — the "
             "stored orientation is not the wrist frame; the wire converts at both boundaries.")


#: SDK dof indices of the 14 ARM joints (both arms: shoulder pitch/roll/yaw, elbow, wrist
#: roll/pitch/yaw) in the 29-DOF ``G1_29DOF_SDK_DOF_NAMES`` order. Derived from the names rather
#: than hardcoded, so a future SDK re-ordering cannot silently zero the wrong joints.
#:
#: These are exactly the joints the LOW LEVEL zeroes for its ready pose — see
#: ``holosoma_wbt.py::_READY_POSE_ARM_JOINTS`` (14 names, same set) and
#: ``_maybe_apply_init_ready_pose``, whose own log line reads "INIT + frozen hold pinned to the
#: READY POSE (forearms horizontal: default_dof_angles with the 14 arm joints = 0)".
_ARM_SDK_IDX = tuple(
    i for i, name in enumerate(G1_29DOF_SDK_DOF_NAMES)
    if any(k in name for k in ("shoulder", "elbow", "wrist"))
)


def _display_seed_dof(zero_arms: bool) -> np.ndarray | None:
    """The 29-DOF SDK vector the render + the decode anchor are seeded at. None if unavailable.

    Starts from ``home_anchor.default_qpos`` (the trained ``default_dof_angles``: legs/waist crouch)
    and, when ``zero_arms``, ZEROES THE 14 ARM JOINTS — which is what the low level itself holds at.

    THIS IS NOT A COSMETIC POSE. ``3_run_policy_vive_force.sh`` launches with
    ``HOLOSOMA_INIT_READY_POSE=1``, so after the operator presses **A** the low level pins BOTH its
    INIT target and its frozen VIVE hold to ``default_dof_angles`` with all 14 arm joints at 0
    (``holosoma_wbt::_maybe_apply_init_ready_pose`` → ``_apply_hold_dof_override``, which rewrites
    ``_motion_command_t[0, :29]``). The PRE-ARM POSE GATE then compares the broadcast wire against
    that same ``_motion_command_t[0, :29]`` (``_check_vive_arm_pose_gate`` → ``_compute_vr_fk``).
    Measured consequence of using ``default_qpos`` here instead:

        vr_3point contact point, per hand:  default_qpos  vs  arms-zeroed  =  310.2 mm apart
        pre-arm gate tolerance (HOLOSOMA_VIVE_ARM_TOL) =  70 mm   ->  the gate would REFUSE

    (On a rig with ``RECORD=1`` the idle heartbeat re-syncs the wire to the robot's MEASURED pose
    before arming, which is what has been masking this. Seeding correctly means the very first
    broadcast already matches, instead of relying on that recovery.)

    Zeroing only the ARMS keeps the legs/waist crouch, so the render stands on the grid at the same
    sole height (+0.0383 m either way; an all-29-joints-zero pose would move it to +0.0031 m). All
    14 zeros are inside the URDF limits, the forearms come LEVEL (wrist +x elevation -0.00 deg vs
    the crouch pose's -44.45 deg nose-down droop), and the left TCP lands at pelvis x = +0.442 m,
    clear of the torso.

    The whole TCP trajectory follows this pose because the chunk is anchor-relative: the anchor is
    derived from it, so the green path and the triads move together and step 0 stays exactly on the
    tcp triad. Once armed, the anchor re-syncs to the robot's MEASURED pose at every Predict, so
    this seed only decides the pre-arm view — but making it the pose the robot is ACTUALLY in is
    what lets a demo be replayed straight after pressing start.
    """
    try:
        ready = np.asarray(
            json.load(open(_HOME_ANCHOR_PATH)).get("default_qpos", ()), dtype=float)
    except (OSError, ValueError) as exc:
        _log(f"display seed: cannot read home_anchor.json ({exc!r})")
        return None
    if ready.shape != (N_SDK_DOF,) or not np.all(np.isfinite(ready)):
        _log(f"display seed: default_qpos has shape {ready.shape}, expected ({N_SDK_DOF},)")
        return None
    if zero_arms:
        ready = ready.copy()
        ready[list(_ARM_SDK_IDX)] = 0.0
    return ready


def _obs_freshness(planner, *, max_age_s: float, grip_max_age_s: float,
                   consumes_force: bool = True) -> tuple[list[str], str]:
    """``(blocking_reasons, one_line_readout)`` for the LIVE observation rings.

    Module-level (not a ``_run_viser`` closure) because the same verdict is needed by the
    Predict refusal, the sequential loop and the page readout, and because a staleness rule
    that lives in three places is a staleness rule that will disagree with itself.

    Only the CAMERA and the F/T block a Predict: they are the model's two real senses (measured
    sensitivity: swapping the image moves the predicted path by ~38-53% of its own travel,
    zeroing the force ~28%). The gripper is reported but never blocking — 0.4% sensitivity, and
    :meth:`LivePlanner.snapshot_obs` already downgrades it to ``state_valid=False``, which is
    the same thing the training converter does for an unmeasured jaw. That is a deliberate
    asymmetry: refuse when the model would be guessing, degrade when it can absorb it.

    ``consumes_force=False`` (a PURE-POSITION checkpoint — ``FLORE_DROP_FORCE``) takes the F/T
    OUT of the blocking set. This is the same relaxation :func:`_self_check` makes at startup,
    but here it is not merely vacuous — it is fatal without it: an empty ``force_roles`` makes
    ``_pick_force_channels`` return None for every frame *and* leaves ``require_6dof`` False,
    so NEITHER ``push_force`` branch ever runs and the ring is never written at all. The age
    therefore stays ``None`` for the whole session and this gate refused EVERY Predict on a
    healthy bridge (observed on ``ckpt_noforce``: self-check said ``force: ok (6dof)`` and the
    next line said ``no frame ever arrived``). The readout says ``n/a`` rather than ``none`` so
    a sensor the model does not read cannot be misread as a dead one."""
    ages = planner.obs_ages()
    reasons: list[str] = []
    blocking = [("camera :8773", "image")]
    if consumes_force:
        blocking.append(("F/T :8771", "force"))
    for name, key in blocking:
        age = ages.get(key)
        if age is None:
            reasons.append(f"{name}: no frame ever arrived")
        elif age > max_age_s:
            reasons.append(f"{name}: {age:.1f}s stale (limit {max_age_s:.2f}s)")

    def _fmt(key: str, limit: float) -> str:
        age = ages.get(key)
        if age is None:
            return "none"
        return f"{age * 1e3:.0f}ms" + ("!" if age > limit else "")

    readout = (f"img {_fmt('image', max_age_s)} · "
               f"F/T {_fmt('force', max_age_s) if consumes_force else 'n/a'} · "
               f"grip {_fmt('gripper', grip_max_age_s)}")
    return reasons, readout


def _self_check(subs: dict, *, require_6dof: bool, replay_only: bool = False,
                force_advisory: bool = False) -> dict:
    """Probe the sensor bridges before serving. F/T (:8771) + camera (:8773) required;
    gripper (:8784) required to read. Raises SystemExit on a hard failure.

    ``replay_only`` (a MODEL-FREE ``--replay-zarr`` session) DOWNGRADES the F/T and camera
    checks from hard failures to advisory notes, because in that mode nothing consumes an
    observation: the action chunk is read from the zarr, and the only thing the planner does
    with the live sensors is DISPLAY them. Concretely, ``force_history``'s 6-DoF requirement
    exists so the model isn't fed an out-of-distribution torque-zeroed vector — with no model
    there is no distribution to be out of, so refusing to start would block a pure position
    playback on a sensor the playback never reads. The gripper check is advisory in both modes.

    ``force_advisory`` downgrades the SAME two F/T checks for a PURE-POSITION checkpoint
    (``FLORE_DROP_FORCE``), for the same reason and with one difference: the camera stays
    REQUIRED. A force-free model still consumes images — it is only the wrench it never reads,
    so demanding a 6-DoF ``--pipeline`` bridge before it may start is a prerequisite with
    nothing behind it.
    """
    report: dict = {}
    # Either reason makes the F/T frame something we DISPLAY rather than feed.
    ft_optional = replay_only or force_advisory

    def _wait(sub, need_key=None, tries=40):
        for _ in range(tries):
            msg, _wall, n = sub.latest()
            if msg is not None and (need_key is None or msg.get(need_key) is not None):
                return msg
            time.sleep(0.1)
        return None

    fmsg = _wait(subs["force"])
    if fmsg is None:
        if not ft_optional:
            raise SystemExit("self-check: no F/T frame on :8771 (is 2_ft_bridge.sh running?)")
        report["force"] = ("ABSENT (:8771) — F/T panel will be empty; "
                           + ("replay" if replay_only else "a pure-position model")
                           + " does not need it")
    elif require_6dof and fmsg.get("left_pipe") is None and not ft_optional:
        raise SystemExit(
            "self-check: F/T frame has no 6-DoF 'left_pipe' key — the HL model's "
            "force_history was trained on 6-DoF gravity-only force. Start the bridge with "
            "--pipeline, or pass --allow-3dof-force for a bring-up (OOD force).")
    else:
        report["force"] = "ok (6dof)" if fmsg.get("left_pipe") is not None else (
            "3dof only (no 'left_pipe') — display only" if ft_optional else "ok (3dof)")
    cmsg = _wait(subs["camera"], need_key="jpeg_b64")
    if cmsg is None:
        if not replay_only:
            raise SystemExit("self-check: no camera frame on :8773 (is the realsense bridge running?)")
        report["camera"] = "ABSENT (:8773) — not needed: replay shows the RECORDED frame"
    else:
        report["camera"] = "ok"
    gmsg = _wait(subs["gripper"], need_key="actual")
    report["gripper"] = "ok" if gmsg is not None else "no gripper frame (:8784) — proprio will read 0"
    return report


# ===========================================================================
# The single-process planner: sensors + S0 + ingest/heartbeat + predict/execute +
# viser. Everything the muse server + viz did, but in one process, direct-call.
# ===========================================================================
def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    # --algo/--checkpoint are REQUIRED for policy serving but OPTIONAL for a pure
    # --replay-zarr session: replaying recorded actions needs no policy at all (the chunk
    # comes from the zarr), and loading one would pin a GPU + several GB of VRAM for nothing.
    # Enforced jointly below, after parsing, so the error names the actual missing piece.
    p.add_argument("--algo", choices=["flow", "diffusion", "act", MUSE_ALGO],
                   help="policy family to load: flow matching, DDIM diffusion, ACT (a "
                        "single-pass CVAE — --num-steps must be 1 or omitted), or muse (the "
                        "MuSe-on-UVA force policy, which runs in a uvaforce BACKEND process — "
                        "see --backend-python). OMIT (together with --checkpoint) for a "
                        "MODEL-FREE recorded-episode replay — see --replay-zarr.")
    p.add_argument("--checkpoint", help="HL-v2 checkpoint dir (config.json + model.safetensors), "
                                       "or for --algo muse the UVA .ckpt/.slim.pt FILE. "
                                       "Omit for a model-free replay (see --algo).")
    # --- muse only (inert for every other algorithm) ---
    p.add_argument("--backend-python", default=None,
                   help="[--algo muse] the uvaforce interpreter that can import "
                        "unified_video_action. Defaults to $BACKEND_PY then "
                        "~/miniconda3/envs/uvaforce/bin/python.")
    p.add_argument("--muse-config-name", default="uva_hl_force",
                   help="[--algo muse] hydra config used ONLY for a random-init backend (a real "
                        "checkpoint carries its own cfg).")
    p.add_argument("--muse-force-fusion", default=None, choices=["early", "late", "both"],
                   help="[--algo muse] override the checkpoint's force-fusion mode (ablation "
                        "only; the default is whatever it was trained with).")
    # Constrained to left_only: this planner is LEFT-canonical (it applies the URDF<->model
    # Rx(180) flip and builds the vr_3point LEFT slot for the LEFT hand ONLY), and enforces
    # that a few lines below. Offering the other layouts as free-form text would let an
    # operator reach that guard — or worse, get an unknown-mode ValueError traceback — instead
    # of a clear "not supported". Kept as a flag rather than dropped so the constraint is
    # visible in --help and lifting it later is a one-word change.
    p.add_argument("--layout-mode", default="left_only", choices=["left_only"],
                   help="action layout for a MODEL-FREE replay (only left_only: this planner "
                        "is LEFT-canonical). Ignored when --checkpoint is given — the "
                        "checkpoint's own layout wins.")
    p.add_argument("--device", default="cuda")
    p.add_argument("--num-steps", type=int, default=None, help="sampler steps (default from checkpoint)")
    # (No --hand flag: this planner is LEFT-canonical by construction — the anchor seed, the
    # TCP lever and the vr_3point LEFT slot are all left-only, and --layout-mode already
    # enforces it. A single-valued flag that nothing reads only advertised a right-hand mode
    # that does not exist.)
    p.add_argument("--force-ws", default="ws://127.0.0.1:8771")
    p.add_argument("--camera-ws", default="ws://127.0.0.1:8773")
    p.add_argument("--gripper-ws", default="ws://127.0.0.1:8784")
    p.add_argument("--record-ws", default="ws://127.0.0.1:8779",
                   help="low-level RecordTap broadcast (joint readback + pelvis-frame F_meas). "
                        "Requires the low level launched with RECORD=1. When connected, the viser "
                        "mesh reflects the REAL robot joints and the F/T panel shows the exact "
                        "F_meas the servo consumes; when absent, the mesh falls back to the "
                        "predicted-chunk IK render and the F/T panel shows the :8771 sensor wire.")
    p.add_argument("--s0-ws-host", default="127.0.0.1")
    p.add_argument("--s0-ws-port", type=int, default=8770)
    # Both rates DERIVE from ACTION_HZ rather than repeating a literal: on the v3 grid
    # that is 30 Hz, and a stale 16.0 here would silently play every chunk at 0.53x.
    p.add_argument("--obs-hz", type=float, default=ACTION_HZ,
                   help=f"ring ingest rate (Hz). The action grid is {ACTION_HZ:g} Hz "
                        f"(ACTION_HZ); proprio/force are logged every tick.")
    p.add_argument("--step-hz", type=float, default=ACTION_HZ,
                   help=f"chunk playback rate (Hz) — MUST match the training action grid "
                        f"(ACTION_HZ={ACTION_HZ:g}, one action step per grid frame). "
                        f"Distinct from --drip-hz.")
    p.add_argument("--drip-hz", type=float, default=max(30.0, 2.0 * ACTION_HZ),
                   help="wire refresh rate to S0 (Hz) — re-sends the current pose inside "
                        "S0Driver's 0.5 s freshness window. Not the playback speed. Kept "
                        "at >= 2x the step rate so every step is re-sent at least once.")
    p.add_argument("--img-stride", type=int, default=None,
                   help="push a camera frame into the image ring every N ingest ticks so the "
                        "To obs frames are ~obs_history_dt_s apart. DEFAULT: derived from the "
                        "checkpoint's obs_history_dt_s * obs_hz (an explicit value warns on mismatch).")
    p.add_argument("--allow-3dof-force", action="store_true",
                   help="permit a 3-DoF-only F/T stream (torque zero-padded, OOD). Default "
                        "OFF: self-check hard-fails without the 6-DoF pipe.")
    p.add_argument("--obs-max-age", type=float, default=None, metavar="S",
                   help="refuse to Predict when the camera or F/T ring is older than S seconds "
                        f"(default {OBS_MAX_AGE_S}). Every ingest source degrades by "
                        "REPEATING its last value — a dead camera bridge, a dead Bota reader "
                        "behind a live publisher, a crashed ingest thread — so without an age "
                        "check the model is fed a confident frozen observation and the page "
                        "still reads ONLINE. 0 disables the refusal (the readout stays).")
    p.add_argument("--command-gripper", action="store_true",
                   help="autonomously drive the LEFT jaw from the model's predicted width "
                        "(position command on :8784). Default OFF = read-only proprioception. "
                        "WITHOUT THIS FLAG THE GRIPPER NEVER CLOSES — the model predicts a "
                        "width every step and the planner discards it, so the bridge sits at "
                        "'target_frac=freeze'. Needed for any task involving a grasp.")
    p.add_argument("--grip-overshoot-mm", type=float, default=10.0,
                   help="CLOSING OVERSHOOT (mm): when the model commands a CLOSED jaw, drive "
                        "the target this much narrower than predicted so the jaw stalls against "
                        "the object and the position error becomes grip force. Needed because "
                        "the recorded 'closed' width is the TOOL'S THICKNESS (42-48 mm on the "
                        "0808_2228 session, never 0), so replaying it verbatim stops exactly at "
                        "the object's surface and only touches it. 0 disables. Live-adjustable "
                        "in the viser page; only ever applied to the CLOSED mode, and never to "
                        "the width the model OBSERVES.")
    p.add_argument("--grip-close-below-mm", type=float, default=70.0,
                   help="width (mm) under which a commanded jaw ENGAGES the squeeze. Default "
                        "70 mm is the measured trough between the two modes of the 0808_2228 "
                        "gripper histogram (closed 42-48, open 84-94).")
    p.add_argument("--grip-confirm-steps", type=int, default=3,
                   help="how many CONSECUTIVE close (or open) predictions are required before "
                        "the overshoot engages (or cancels). Default 3. This is what rejects a "
                        "single flickering step: temporal ensembling blends a BIMODAL gripper "
                        "signal, so one transient open inside a grasp otherwise walks the "
                        # `%%`, not `%`: argparse %-formats every help string, so a bare `%` here
                        # made `--help` itself die with "%o format: an integer is required".
                        "commanded target 10.1%% of travel back open — the measured 'grips then "
                        "loosens'. 1 = react to every step (no confirmation).")
    p.add_argument("--grip-hold-tightest", type=int, choices=(0, 1), default=1,
                   help="1 (default): while a grasp is CONFIRMED closed, never loosen the "
                        "published target. Needed because the blended width itself drifts "
                        "(measured 47.7 -> 53.9 mm inside one grasp) WITHOUT ever crossing the "
                        "close threshold, so the confirmation run cannot see it — with a 10 mm "
                        "overshoot that returns the target to the object's own surface and grip "
                        "force decays to ~0. 0: follow the width verbatim once engaged.")
    p.add_argument("--auto-max-seconds", type=float, default=20.0,
                   help="wall-clock ceiling for the SEQUENTIAL model rollout ('▶▶ Run to "
                        "completion'). A policy has no end of its own, so the loop is bounded "
                        "by time; Stop always interrupts sooner. Default 20 s.")
    p.add_argument("--force-gate-n", type=float, default=1.5,
                   help="initial force gate (N): with the master switch ON, a hand whose "
                        "predicted ‖F‖ ≤ this is hard-zeroed (pure position); above it, F^cmd "
                        "is published. Adjustable live in the viser page. The low level MIRRORS "
                        "this number twice (FORCE_CMD_WIRE_GATE_N hard-zeros a sub-gate command "
                        "before it reaches the buffer, and GATE_CMD_N keeps F_meas out of the "
                        "obs+servo below it), so below the gate the whole stack is pure position "
                        "no matter what the Bota reads — change all three together.")
    p.add_argument("--enable-force-cmd", action="store_true",
                   help="start with the force master switch ON (publish gated F^cmd). Default "
                        "OFF = pure position controller (wire carries no force command).")
    p.add_argument("--exec-steps", type=int, default=16,
                   help="how many of the predicted action steps to EXECUTE on Run (the rollout "
                        "always predicts all; this only limits how many are dripped to the robot). "
                        f"Default 16 (of {ACTION_HORIZON}) ≈ 0.53 s at {ACTION_HZ:g} Hz; "
                        f"adjustable live in the viser page (16/30/all).")
    # --- RECORDED-EPISODE REPLAY (drive the rig with a demo instead of the policy) ---
    # Not a separate script: the whole value is that replay reuses the IDENTICAL
    # predict→preview→execute→commit pipeline (same wire, same execute-N-of-S, same gate,
    # same anchor clutch), so "what the demo did" and "what the policy predicts" are
    # comparable on the same rig under the same conditions. The chunk SOURCE is the only
    # difference; see flore/s1_policy/serving/hl_replay.py for why the recorded chunk is fed
    # in the model's own anchor-RELATIVE form (the recorded world frame cancels exactly).
    p.add_argument("--replay-zarr", default=None, metavar="PATH",
                   help="enable RECORDED-EPISODE REPLAY from a converted .zarr.zip (e.g. the "
                        "clean94 session the checkpoint was trained on). The page gains a Replay "
                        "panel: pick an episode + start frame, then Predict/Run execute the "
                        "RECORDED actions instead of the model's. Pass 'checkpoint' to use the "
                        "checkpoint's own training zarr (hl_data.zarr_paths[0]). The model is "
                        "still loaded, and the Source toggle switches between them live.")
    p.add_argument("--replay-episode", type=int, default=0,
                   help="initial replay episode index (default 0); selectable live in the page.")
    # --- DEPLOY RECORDING (turn a real-robot run into a re-runnable take) ---
    # A BUTTON, not an always-on tap: the page's ⏺ toggle decides whether the next execution is
    # recorded, so an operator can bring the rig up, home, arm and rehearse without filling the
    # disk with takes nobody wants. --record only decides the toggle's INITIAL state (and the
    # destination), so a session that never presses it writes nothing at all.
    p.add_argument("--record", action="store_true",
                   help="start with deploy RECORDING armed: every EXECUTION (Run / Play / "
                        "sequential) writes a take under --record-dir holding the executed "
                        "prediction's own observation + raw chunk, the measured 29-DOF joint "
                        "readback + FK'd TCP, the published wire, the raw F/T, and the camera "
                        "JPEGs. Re-run it offline with scripts/reinfer_hl_take.py. Toggleable "
                        "live in the page; a Predict that is never Run is NOT recorded.")
    p.add_argument("--record-dir", default=None, metavar="DIR",
                   help="destination session directory for recorded takes (default "
                        "<repo>/data/deploy_takes/<UTC timestamp>). Takes land in it as "
                        "take_000/, take_001/, … and a session_meta.json names the checkpoint.")
    # --- the ⏺ RUN RECORDER (ported from the five-point page, 2026-09-20) ---
    # A SECOND recorder, not a mode of the take recorder above, because the two answer different
    # questions. A TAKE (--record) is a re-runnable dataset entry: the executed prediction's own
    # observation + raw chunk, so scripts/reinfer_hl_take.py can reproduce the chunk offline. A
    # RUN RECORD (this) is the run's TIME SERIES for the paper figures: every predicted wrench at
    # the wall time it executed, every F/T frame, every RecordTap frame with the robot's three wire
    # points FK'd, every published frame and every camera frame — one folder per ▶▶ run under
    # data/rollouts/<task>/<YYYY-MM-DD_HH-MM-SS>/, the SAME layout scripts/hl_serving_5pt.py
    # writes, so scripts/plot_rollout_record.py and overleaf/tools/gen_fig_rollout_force_pred.py
    # read a 3-point run exactly as they read a 5-point one (the point count is read off
    # meta.static.wire_labels). `rollout-` rather than `record-` because `--record`/`--record-dir`
    # already name the take recorder on this page; the launcher's env knobs are the 5pt launcher's
    # (TASK=, RECORD_EVERY=1, RECORD_CAMERA_EVERY=N) plus ROLLOUT_DIR= for the root.
    ro = p.add_argument_group("run recorder (the ⏺ button: one recording per episode — from the first Predict / "
                              "▶▶ to Reset / the low level disarming — to a time-named folder)")
    ro.add_argument("--rollout-dir", default=os.path.join(_FLORE_ROOT, "data", "rollouts"), metavar="DIR",
                    help="root of the ⏺ run recordings: DIR/<task>/<YYYY-MM-DD_HH-MM-SS>/, one folder per "
                         "recording (chunks / steps / published / measured / force / events .jsonl + camera/ + "
                         "meta.json — the five-point page's format). Plot one with "
                         "scripts/plot_rollout_record.py <folder>. Default data/rollouts. NOT the deploy "
                         "take's --record-dir.")
    ro.add_argument("--rollout-task", default=None, metavar="NAME",
                    help="the <task> folder name (wipe_board_new, wipe_board_all, …). The launcher "
                         "forwards TASK= here; default = read off the --checkpoint path.")
    ro.add_argument("--rollout-every-run", action="store_true",
                    help="start with 'Record EVERY run' ticked: the first Predict / ▶▶ after each end opens "
                         "the next folder instead of asking for a click (launcher: RECORD_EVERY=1).")
    ro.add_argument("--no-rollout-autoplot", dest="rollout_autoplot", action="store_false", default=True,
                    help="do not spawn scripts/plot_rollout_record.py in the background (nice -n 10) when a "
                         "recording is saved. Default: plot automatically; the page never waits on it.")
    ro.add_argument("--rollout-camera-every", type=int, default=1, metavar="N",
                    help="record every N-th frame of the camera this checkpoint reads (:8773, the page's "
                         "own subscription) into <run>/camera/<image_key>/NNNNNN.jpg + camera/frames.jsonl "
                         "— the bridge's own full-size JPEG, not the 224 crop. 1 (default) = every frame "
                         "(~30 fps, ~2 MB/s); 0 = no camera frames (launcher: RECORD_CAMERA_EVERY=N).")
    # REACH COMPENSATION (default OFF ⇒ byte-identical rig behaviour). Measured on the recorded
    # deploy takes: the low level tracks how the wire MOVES (displacement gain 0.80/0.94/0.80,
    # corr +0.37/+0.41/+0.26) but a STANDING position error produces no corrective motion
    # (corr(lead, next tick) = -0.18/-0.04/-0.01), so the hand parks ~2 cm short and the
    # measured-pose re-anchor forgives that residual on every Predict. Nothing in the loop keeps
    # score, which is why a 3-4 cm fine adjustment never lands. --reach-ki supplies the missing
    # integral term on the PUBLISHED WIRE ONLY (never the obs, never the decode anchor, never the
    # chunk). See flore/s1_policy/serving/hl_reach.py for the full measurement writeup.
    p.add_argument("--reach-ki", type=float, default=0.0, metavar="KI",
                   help="ENABLE reach compensation with this integral gain (default 0 = OFF, "
                        "byte-identical). Accumulates the part of the commanded motion the arm "
                        "did not execute and adds it to the published target, so a fine "
                        "adjustment keeps insisting instead of being forgiven at the next "
                        "re-anchor. START AT 0.05 (measured on the recorded takes: with the "
                        "default 1.5 cm dead zone it adds ~0.7 cm of extra reach and saturates "
                        "on only 5%% of steps); 0.1+ spends much longer pinned at the clamp. "
                        "Live-toggleable in the page's 'Reach compensation' folder, and tunable "
                        "offline against your own takes with scripts/tune_reach_comp.py.")
    p.add_argument("--reach-clamp", type=float, default=0.03, metavar="M",
                   help="hard bound (m) on the accumulated residual — the published target can "
                        "NEVER exceed the model's own target by more than this (default 0.03). "
                        "This is the safety limit: saturated, the arm asks for 3 cm of extra "
                        f"reach and no more. Ceiling {hl_reach.MAX_CLAMP_M} m.")
    p.add_argument("--reach-stall-frac", type=float, default=0.5, metavar="F",
                   help="THE GATE (default 0.5): accumulate ONLY while the arm performed less "
                        "than this fraction of the motion the wire asked for, measured over "
                        "--reach-window steps as (dMeasured.dCommanded)/|dCommanded|^2. This is "
                        "the 'commanded 6 cm, moved 3 cm -> leave it alone; commanded 5 cm, "
                        "didn't move -> accumulate' rule. Measured on the 12 takes of "
                        "20260814_235118 it catches 100%% of genuinely stalled steps with 0%% "
                        "false positives, where gating on the shortfall DISTANCE managed only "
                        "18%%/24%% (the standing lead is ~2.5 cm whether the arm is tracking or "
                        "frozen, so distance carries no information about motion).")
    p.add_argument("--reach-window", type=int, default=9, metavar="N",
                   help="how many published steps the realized fraction is measured over "
                        "(default 9 = 0.3 s at 30 Hz). One tick carries ~0.7 mm of intended "
                        "motion and mostly readback noise, so the window is what makes the "
                        "'is it moving' judgement meaningful.")
    p.add_argument("--reach-gate", type=float, default=0.0, metavar="M",
                   help="OPTIONAL secondary distance ceiling (m, default 0 = off): also stop "
                        "growing the residual once the shortfall exceeds this, bounding "
                        "compensation to fine adjustments even if the stall gate opens during a "
                        "large motion. The primary gate is --reach-stall-frac.")
    p.add_argument("--reach-decay", type=float, default=0.995, metavar="D",
                   help="per-step leak on the stored residual (default 0.995, must be <= 1). "
                        "Keeps a stale residual from outliving the situation that produced it.")
    p.add_argument("--reach-contact-n", type=float, default=1.5, metavar="N",
                   help="CONTACT CUT-OUT (N, default 1.5): above this measured force the residual "
                        "stops growing and decays. A hand pressing on something immovable shows "
                        "the SAME small shortfall as one that merely fell short, and integrating "
                        "that commands it to press HARDER — straight toward the 40 N "
                        "FORCE_ESTOP_N stop. Measured on the recorded takes, 5-45%% of the steps "
                        "that would accumulate (per take) had >1.5 N of live contact, so this is "
                        "the common case. 0 disables it; do not on hardware.")
    # --- chunk anchor: what the NEXT chunk decodes against -------------------------------------
    # Orthogonal to --reach-*: that adds to a published TARGET, this changes the DECODE BASE, so
    # they can be used together or separately. Applies to EVERY algorithm (act / diffusion / flow
    # / muse) and to recorded replay, because all of them go through the one re-anchor.
    p.add_argument("--anchor-mode", choices=list(hl_anchor.ANCHOR_MODES), default="measured",
                   help="what the next chunk's decode anchor is. 'measured' (default) re-anchors "
                        "on the arm's measured TCP every Predict — byte-identical to previous "
                        "behaviour. 'command' anchors on the LAST EXECUTED COMMANDED step "
                        "instead (PLAI / command integration), which removes both costs of the "
                        "re-anchor measured on the recorded takes: the 84%% [81,86] of the "
                        "accumulated shortfall it wipes at every chunk boundary, and the 16.0 mm "
                        "[13.8,18.3] BACKWARD jump it puts in the published target there (13.6x a "
                        "normal 1.18 mm step). Re-measure with scripts/measure_exec_gap.py.")
    p.add_argument("--anchor-eps", type=float, default=0.03, metavar="M",
                   help="[--anchor-mode command] LEAK BOUND (m, default 0.03): how far the "
                        "carried command anchor may lead the measured pose before it is pulled "
                        "back onto that ball. NOT a tuning knob — replayed over the recorded "
                        "takes, UNCLAMPED command integration diverges to 224 mm by end of take "
                        "(p95 peak 574 mm); 5 cm saturates at 49 mm, 3 cm at 30 mm. Ceiling "
                        f"{hl_anchor.MAX_EPS_M} m. 0 collapses the mode back to 'measured'.")
    p.add_argument("--anchor-eps-deg", type=float, default=10.0, metavar="DEG",
                   help="[--anchor-mode command] the same bound on the carried ORIENTATION lead "
                        f"(default 10 deg, ceiling {np.degrees(hl_anchor.MAX_EPS_RAD):.0f}).")
    p.add_argument("--anchor-contact-n", type=float, default=1.5, metavar="N",
                   help="[--anchor-mode command] CONTACT CUT-OUT (N, default 1.5): at or above "
                        "this measured force the carried anchor DECAYS back toward the arm "
                        "instead of advancing. Contact multiplies the shortfall 1.8x on the "
                        "recorded takes, so an anchor that keeps advancing against a blocked hand "
                        "is a press, not a reach. 0 disables it; do not on hardware.")
    p.add_argument("--anchor-decay", type=float, default=0.9, metavar="D",
                   help="[--anchor-mode command] pull-back factor per resolve while in contact "
                        "(default 0.9, must be in (0,1]). Decayed rather than snapped so the "
                        "cut-out itself does not put back the jump the mode removes.")
    # Where the "Save comparison plot" button writes. NOTHING is written unless that button is
    # pressed — the trace is in-memory only — so this just decides the destination, and the page
    # reports the resolved absolute path after each save.
    p.add_argument("--trace-dir", default=os.path.join(_FLORE_ROOT, "data", "traces"),
                   help="directory for saved executed-vs-recorded comparison PNGs (default "
                        "<repo>/data/traces). Created on first save; nothing is written unless "
                        "you press 'Save comparison plot' in the page.")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8088)
    p.add_argument("--yam-urdf", default=os.path.join(_UMI, "robot_assets", "g1_yam", "g1_29dof_yam.urdf"))
    p.add_argument("--ik-config", default="proposal")
    p.add_argument("--reach-scale", type=float, default=1.0)
    p.add_argument("--cam-offset", type=float, nargs=3, default=(0.10, 0.0, 0.04))
    p.add_argument("--cam-rpy", type=float, nargs=3, default=(4.7124, 1.5708, 0.0))
    p.add_argument("--no-selfcheck", action="store_true")
    # OFFLINE: look at the trajectory in viser with NO low level and NO bridges running.
    # Deliberately a FLAG rather than auto-detection. "Nothing is connected" and "the low level
    # died / the wire is being refused" are the same observable, and auto-suppressing the
    # reconnect + arming warnings in the second case would hide a real fault on a live rig. With
    # the flag the operator has stated the intent, so the warnings are noise and the page can say
    # OFFLINE instead of pretending it is waiting for a robot.
    p.add_argument("--offline", action="store_true",
                   help="VIZ-ONLY: no low level, no sensor bridges, nothing published. Skips the "
                        "bridge self-check, does not open the sensor sockets at all (so no "
                        "reconnect spam), and suppresses the RecordTap/arming warnings. The page "
                        "renders the URDF at its ready pose, the triads, and the recorded green "
                        "path so you can inspect the FRAME. Predict/Load still builds and previews "
                        "a chunk; Execute is DISABLED (there is nothing to execute to).")
    # DEFAULT ON, because it is what the ROBOT actually holds: 3_run_policy_vive_force.sh runs with
    # HOLOSOMA_INIT_READY_POSE=1, so after 'A' the low level pins its INIT + frozen hold (and hence
    # the pre-arm gate's reference) to default_dof_angles with all 14 arm joints at 0. Seeding from
    # the raw crouch default_qpos instead puts the broadcast wire 310 mm per hand from the gate's
    # reference, against a 70 mm tolerance. --no-zero-arms restores the raw default_qpos for
    # debugging a low level launched WITHOUT the ready-pose override.
    p.add_argument("--zero-arms", action=argparse.BooleanOptionalAction, default=True,
                   help="seed the render AND the decode anchor at the low level's real ready pose: "
                        "home_anchor default_dof_angles with the 14 ARM joints ZEROED (forearms "
                        "horizontal, both hands), matching HOLOSOMA_INIT_READY_POSE=1 in "
                        "3_run_policy_vive_force.sh. Default ON. --no-zero-arms uses the raw crouch "
                        "default_qpos, which is 310 mm/hand from the low level's pre-arm gate "
                        "reference (tol 70 mm) — only for a low level started with "
                        "HOLOSOMA_INIT_READY_POSE=0.")
    args = p.parse_args(argv if argv is not None else sys.argv[1:])

    # --- resolve the two mutually-exclusive launch shapes ---
    # (a) POLICY: --algo + --checkpoint -> load the model; replay is an optional extra source.
    # (b) MODEL-FREE REPLAY: --replay-zarr with neither -> no torch model, no GPU.
    if bool(args.algo) != bool(args.checkpoint):
        raise SystemExit("--algo and --checkpoint must be given TOGETHER (or both omitted "
                         "for a model-free --replay-zarr replay).")
    replay_only = not args.checkpoint
    if replay_only and not args.replay_zarr:
        raise SystemExit(
            "nothing to serve: pass --algo/--checkpoint to run a policy, or --replay-zarr "
            "to replay a recorded episode with no model loaded.")
    if not args.zero_arms:
        _log("WARNING: --no-zero-arms — seeding at the raw crouch default_qpos. If the low level was "
             "started with HOLOSOMA_INIT_READY_POSE=1 (the default in 3_run_policy_vive_force.sh) "
             "its hold + pre-arm gate reference is the ARMS-ZEROED pose, 310 mm/hand away, and "
             "arming will REFUSE until the idle heartbeat re-syncs the wire to measured.")

    _limit_cpu_thread_pools(args.device)

    # --- model + planner core ---
    # `muse` swaps BOTH of these; every other algorithm leaves them at the defaults, so the
    # flow/diffusion/act path is byte-identical to before.
    muse_backend = None
    planner_class = LivePlanner
    if replay_only:
        # No checkpoint => no policy, no GPU, no norm stats. The layout is all the rest of
        # the planner needs (it slices the action vector and seeds the anchor); the recorded
        # chunk is decoded by the SAME decode_chunk the model path uses, so the wire, the
        # force gate, the drip, and the clutch re-anchor are unchanged.
        _log(f"MODEL-FREE recorded replay (layout {args.layout_mode}) — no checkpoint, no GPU.")
        loaded = replay_only_model(args.layout_mode)
    elif args.algo == MUSE_ALGO:
        # MuSe-on-UVA: the model cannot load in THIS interpreter (see hl_muse's docstring), so
        # it comes up as a child process in the uvaforce env and `loaded.model` is a shim over
        # the pipe. Everything after this line — the wire, the force gate, the drip, the
        # recorder, reach compensation, the ghost — is the same code the local algorithms run.
        _log(f"loading muse (UVA) checkpoint {args.checkpoint} via a {args.device} backend …")
        loaded, muse_backend = load_muse_model(
            args.checkpoint,
            backend_python=resolve_backend_python(args.backend_python),
            device=args.device,
            layout_mode=args.layout_mode,
            force_fusion=args.muse_force_fusion,
            config_name=args.muse_config_name,
        )
        planner_class = MuseLivePlanner
    else:
        _log(f"loading {args.algo} checkpoint {args.checkpoint} on {args.device} …")
        loaded = load_hl_model(args.checkpoint, args.algo, device=args.device, num_steps=args.num_steps)
    # LEFT-canonical means "there is a LEFT keypoint to DECODE motion for", which is a claim
    # about roles, not about sensors. This used to test force_roles and so refused a
    # pure-position checkpoint (FLORE_DROP_FORCE leaves force_roles empty) even though such a
    # model decodes the left hand exactly the same way — it just never reads a wrench.
    if "left" not in loaded.layout.roles:
        raise SystemExit(f"checkpoint layout {loaded.layout.mode!r} has no LEFT role; "
                         "this planner is LEFT-canonical.")
    #: Does this checkpoint CONSUME force? False for a pure-position model, which makes every
    #: F/T prerequisite below vacuous rather than merely satisfiable.
    consumes_force = bool(loaded.layout.force_roles)
    if not consumes_force:
        _log("PURE-POSITION checkpoint (no F/T in the layout): force_history is not consumed, "
             "the F/T bridge is advisory, and the force-command switch stays inert.")
    # LEFT-ONLY FRAME ASSUMPTION (load-bearing). Only the LEFT role is DECODED for motion here (its
    # wire is built from the model output; RIGHT/HEAD, when present, are HELD at their measured
    # pose). The URDF↔model (Rx(180), z-down/z-up) frame conversion is applied ONLY to the LEFT
    # anchor + the LEFT decoded wire (see _seed_planner_anchor / _resync_anchor_from_robot /
    # _infer). A layout that also DRIVES the RIGHT hand (both_hands*) would silently publish it in
    # the WRONG frame (its anchor is seeded in raw URDF and its decoded pose is never flipped).
    # Today's deploy checkpoints are left_only; refuse anything else LOUDLY rather than move the
    # right arm 180° off — extend the flip to RIGHT before lifting this guard.
    if loaded.layout.roles != ("left",):
        raise SystemExit(
            f"checkpoint layout {loaded.layout.mode!r} has roles {loaded.layout.roles} — this planner "
            "only applies the URDF↔model (z-down/z-up) frame conversion to the LEFT hand, so any layout "
            "that decodes RIGHT would drive that arm 180° off. Extend the flip in _seed_planner_anchor/"
            "_infer/_resync_anchor_from_robot to every decoded role before deploying a non-left_only ckpt.")
    # MuseLivePlanner is LivePlanner plus the one observation stream HL-v2 does not carry (an
    # ABSOLUTE EE-position history, which UVA conditions on anchor-relatively). Subclassed
    # rather than folded in, so the flow/diffusion/act paths run byte-identical code.
    planner = planner_class(loaded, obs_hz=args.obs_hz)
    # ONE knob moves all three consumers of staleness (page readout, Predict refusal, gripper
    # validity flag), so they can never disagree about what "current" means. `0` turns the
    # REFUSAL off (see `_refuse_stale_obs`) while leaving the thresholds — and therefore the
    # page readout and the gripper flag — at their defaults, because "let me run with a flaky
    # bridge" must not also mean "stop telling me the bridge is flaky".
    planner.obs_max_age_s = (OBS_MAX_AGE_S if not args.obs_max_age or float(args.obs_max_age) <= 0
                             else float(args.obs_max_age))
    planner.grip_max_age_s = max(planner.obs_max_age_s, GRIP_MAX_AGE_S)
    home_anchor = json.load(open(_HOME_ANCHOR_PATH))
    mapper = WireMapper(home_anchor)
    # RESOLVE THE URDF↔MODEL SEAM BEFORE SEEDING THE ANCHOR. _seed_planner_anchor already
    # converts the home wrist orientation into the model frame, so the seam must be installed
    # first or the anchor is seeded with the default and every later decode branches from the
    # wrong frame. See the _MODEL_SEAM block for why this is derived rather than hardcoded.
    _install_frame_seam(args, replay_only)
    _seed_planner_anchor(planner, mapper)

    # Couple the image/proprio stride to the checkpoint's trained obs spacing so the To
    # history frames are obs_history_dt_s apart (time-anchored, like the dataset), instead
    # of trusting a fixed CLI default that only matches at obs_hz=16. If the user passed an
    # explicit --img-stride we honor it (with a warning on a mismatch); otherwise derive it.
    # ⚠ To=1 (HL contract v3) HAS NO HISTORY TO SPACE, so the stride must be 1 there.
    # obs_history_dt_s is the gap BETWEEN history frames; with a single observation frame there
    # is no gap to honor, and a stride > 1 stops being "time-anchor the history" and becomes
    # "throttle the only observation the model gets". At obs_hz=30 the v3 checkpoints derived
    # stride=8, i.e. a new frame every 8 ticks = 3.75 Hz, while the dataset's camera runs at
    # 29.988 Hz — so the model was trained on ~30 Hz observations and served 3.75 Hz ones, and
    # 8 consecutive re-queries saw a BYTE-IDENTICAL image. Harmless-looking in the chunk cadence
    # (one Predict per chunk, ~2 s apart, so the frame was usually fresh anyway) but it silently
    # defeats temporal ensembling, whose entire premise is a FRESH observation each step.
    if int(loaded.obs_history_len) <= 1:
        _derived_stride = 1
    else:
        _derived_stride = max(1, round(loaded.obs_history_dt_s * args.obs_hz))
    if args.img_stride is None:
        args.img_stride = _derived_stride
    elif replay_only:
        pass  # no model consumes the obs history; the stride only paces the viz ring
    elif args.img_stride != _derived_stride:
        if int(loaded.obs_history_len) <= 1:
            _log(f"WARNING: --img-stride={args.img_stride} but this checkpoint has To=1 "
                 f"(no obs history), so the derived stride is 1. A stride > 1 only makes the "
                 f"single observation STALER (a new frame every {args.img_stride} ticks = "
                 f"{args.obs_hz / max(args.img_stride, 1):.2f} Hz vs the ~30 Hz it trained on) "
                 f"— and it defeats temporal ensembling. Pass --img-stride 1 unless you mean it.")
        else:
            _log(f"WARNING: --img-stride={args.img_stride} != derived {_derived_stride} "
                 f"(obs_history_dt_s={loaded.obs_history_dt_s} * obs_hz={args.obs_hz}); the To obs "
                 f"frames will NOT be {loaded.obs_history_dt_s}s apart — OOD unless you meant it.")
    if replay_only:
        _log(f"ready: RECORDED REPLAY ONLY mode={loaded.layout.mode} "
             f"force_hands={loaded.n_force_hands} (no policy loaded)")
    else:
        _log(f"ready: algo={loaded.algorithm} mode={loaded.layout.mode} To={loaded.obs_history_len} "
             f"force_hands={loaded.n_force_hands} steps={loaded.num_steps} img_stride={args.img_stride}")

    # --- optional recorded-episode replay source ---
    replay = None
    if args.replay_zarr:
        from flore.s1_policy.serving.hl_replay import RecordedEpisodeSource

        replay_path = args.replay_zarr
        if replay_path == "checkpoint":
            if replay_only:
                raise SystemExit(
                    "--replay-zarr checkpoint needs a --checkpoint to read the trained zarr "
                    "path from. For a model-free replay, pass the .zarr.zip path explicitly.")
            # The checkpoint records the zarr it was TRAINED on; resolving it here means the
            # replayed demos are guaranteed to be the same distribution as the policy's
            # targets, so the two Source modes are an apples-to-apples comparison.
            ckpt_cfg = json.load(open(os.path.join(args.checkpoint, "config.json")))
            trained_paths = ckpt_cfg.get("hl_data", {}).get("zarr_paths") or []
            if not trained_paths:
                raise SystemExit("--replay-zarr checkpoint: the checkpoint declares no "
                                 "hl_data.zarr_paths; pass an explicit .zarr.zip path.")
            replay_path = trained_paths[0]
            if not os.path.isabs(replay_path):
                replay_path = os.path.join(_FLORE_ROOT, replay_path)
            _log(f"replay: resolved checkpoint training zarr -> {replay_path}")
        # Report a bad dataset as a one-line operator error, not a traceback: --replay-zarr
        # takes a free-form path typed by hand (or ZARR= in the launcher), so "wrong path" and
        # "this file isn't a replayable HL-v2 session" are ordinary operator mistakes, and
        # RecordedEpisodeSource's contract gates raise ValueError with an actionable message.
        try:
            replay = RecordedEpisodeSource(replay_path, loaded.layout)
        except (FileNotFoundError, ValueError, KeyError) as exc:
            raise SystemExit(f"--replay-zarr {replay_path}: {exc}") from exc
        lens = replay.episode_lengths()
        _log(f"replay: {os.path.basename(replay.path)} — {replay.n_episodes} episodes, "
             f"{replay.n_frames} frames, lengths {min(lens)}..{max(lens)}")
        if not 0 <= args.replay_episode < replay.n_episodes:
            raise SystemExit(f"--replay-episode {args.replay_episode} outside "
                             f"[0, {replay.n_episodes - 1}]")
        # A replayed demo whose puck→TCP extrinsic differs from the deploy rig's means the
        # stored pose refers to a DIFFERENT physical point, so the same recorded hand motion
        # implies a different EE-local delta. Warn (don't refuse: the deltas are still a real
        # human motion, just anchored to another point on the tool).
        # A replayed demo whose puck→TCP extrinsic differs from the one the CHECKPOINT was
        # trained against means the stored pose refers to a different physical point on the
        # tool, so the same hand motion implies a different EE-local delta. Compare, don't
        # just log: the datasets directory holds sessions that look interchangeable but are
        # not (clean99 carries the pre-fix (0.21,0,0)), and --replay-zarr takes a free-form
        # path. Warn rather than refuse — the deltas are still a real human motion, just
        # anchored to another point, and the operator may deliberately want the comparison.
        t_ext = replay.attrs.get("T_tracker_tcp")
        if t_ext is not None:
            t_vec = np.asarray(t_ext, dtype=np.float64)[:3, 3]
            _log(f"replay: dataset puck→TCP extrinsic = {np.round(t_vec, 4).tolist()}")
            # Only meaningful against a POLICY: the comparison asks "is this session the one
            # the checkpoint learned?". A model-free replay has no checkpoint to differ from.
            trained_zarr = None if replay_only else (
                json.load(open(os.path.join(args.checkpoint, "config.json")))
                .get("hl_data", {}).get("zarr_paths") or [None])[0]
            if trained_zarr and os.path.basename(str(trained_zarr)) != os.path.basename(replay.path):
                _tp = str(trained_zarr)
                if not os.path.isabs(_tp):
                    _tp = os.path.join(_FLORE_ROOT, _tp)
                try:
                    import zarr as _zarr

                    _t_train = dict(_zarr.open(_tp, "r").attrs).get("T_tracker_tcp")
                except Exception:  # noqa: BLE001 — advisory only
                    _t_train = None
                if _t_train is not None:
                    _tv = np.asarray(_t_train, dtype=np.float64)[:3, 3]
                    if not np.allclose(_tv, t_vec, atol=1e-6):
                        _log("replay: WARNING — this session's puck→TCP extrinsic "
                             f"{np.round(t_vec, 4).tolist()} differs from the one the "
                             f"checkpoint trained on {np.round(_tv, 4).tolist()}. The stored "
                             "pose means a DIFFERENT physical point, so the recorded motion is "
                             "anchored to another point on the tool (a few mm of extra error "
                             "per chunk) and is NOT the policy's own training distribution.")
        # SPEED NOTE (inherent, not a bug — surfaced so nobody reads a speed delta as drift):
        # the demonstration was captured on the CAMERA grid (~15 Hz on this rig) but is executed
        # on the ACTION grid (16 Hz, ACTION_DT), because that is the grid the converter baked the
        # chunks onto and the grid the policy is trained to emit. Matching the policy's timebase
        # is what makes a replay-vs-policy comparison fair, but it does mean replay is NOT a
        # wall-clock-faithful reproduction of the take.
        #
        # The per-chunk delta is NOT simply cam_hz/step_hz: advance_frame lands on the camera
        # frame NEAREST the executed time, so the rounding direction — and hence the cumulative
        # speed — depends on how many steps each chunk executes. Measured on clean94: +6% at
        # exec=4 (0.250 s rounds UP to a 0.267 s gap) but −7% at exec=8 (0.500 s rounds DOWN to
        # 0.467 s) and −1% at exec=16. So report it PER exec-steps choice rather than printing
        # one ratio that is wrong for two of the three settings.
        _ep0_times = replay.episode_times(0)
        if _ep0_times.size > 2:
            _cam_dt = float(np.median(np.diff(_ep0_times)))
            if _cam_dt > 1e-6:
                _step_dt = 1.0 / max(args.step_hz, 1.0)
                _parts = []
                _n_ep0 = replay.episode_lengths()[0]
                _start0 = replay.first_usable_frame(0)
                for _n in _EXEC_CHOICES:
                    # Walk episode 0 the way the sequential loop actually does — including
                    # SKIPPING unplayable frames, since a skip advances the cursor without
                    # consuming execution time and would otherwise be scored as "slow".
                    _cur, _steps, _guard = (_start0 or 0), 0, 0
                    _t_from = float(_ep0_times[_cur])
                    while _cur < _n_ep0 - 1 and _guard < 100000:
                        _guard += 1
                        _u = replay.next_usable_frame(0, _cur)
                        if _u is None:
                            break
                        if _u != _cur:
                            _cur = _u  # skipped frames cost no execution time
                            continue
                        # _drip clamps n_exec to the chunk's own valid length, which shrinks
                        # near the episode end — crediting the full _n there would score the
                        # replay as slower than it is.
                        _n_done = min(_n, replay.chunk_at(0, _cur).n_valid)
                        _cur = replay.advance_frame(0, _cur, _n_done)
                        _steps += _n_done
                    _exec_s = _steps * _step_dt
                    _rec_s = float(_ep0_times[min(_cur, _n_ep0 - 1)]) - _t_from
                    if _exec_s > 1e-6 and _rec_s > 1e-6:
                        _parts.append(f"exec={_n}: {(_rec_s / _exec_s - 1.0) * 100:+.0f}%")
                _log(f"replay: recorded camera grid ~{1.0 / _cam_dt:.2f} Hz vs execution "
                     f"{args.step_hz:.2f} Hz → wall-clock speed vs the original human motion "
                     f"({', '.join(_parts)}; + = faster). This is SPEED only — the executed "
                     "PATH is faithful (~93-101% of the recorded arc length). Each chunk "
                     "re-commands its step 0 (the pose the arm already holds), so a k-step "
                     "chunk spends k ticks to advance the demo by k-1; the smaller the "
                     "chunk, the larger that overhead. Not tracking drift. (This describes the "
                     "SEQUENTIAL '⏩ Play episode' path, which re-anchors every chunk. '⏩ Play "
                     "(open-loop, precomputed)' instead decodes the whole episode once and "
                     "reports a true open-loop tracking error at the end — no exec-N overhead, "
                     "no per-chunk re-anchor.)")

    # --- DEPLOY RECORDER (a take per EXECUTION; nothing written until one happens) ---
    # Built unconditionally so every on_* call site in the hot loops is unconditional too (they
    # no-op while no take is open). OFFLINE still gets one: it records nothing (no wire, no
    # readback, and Execute is refused), so there is no take to write, and keeping the object
    # avoids an `if recorder is not None` at a dozen sites.
    #
    # The session directory is derived from a WALL-CLOCK stamp taken once here rather than per
    # take, so all of one bring-up's takes land together and are ordered on disk the way the
    # operator ran them.
    record_dir = args.record_dir or os.path.join(
        _FLORE_ROOT, "data", "deploy_takes", time.strftime("%Y%m%d_%H%M%S", time.localtime()))
    recorder = TakeRecorder(
        record_dir,
        {
            "algo": loaded.algorithm,
            "checkpoint": os.path.abspath(args.checkpoint) if args.checkpoint else None,
            "layout_mode": loaded.layout.mode,
            "num_steps": int(loaded.num_steps),
            "obs_history_len": int(loaded.obs_history_len),
            "force_hist_len": int(loaded.force_hist_len),
            "replay_zarr": (os.path.abspath(replay.path) if replay is not None else None),
            "frame_seam": _MODEL_SEAM_SRC,
            "step_hz": float(args.step_hz),
            "offline": bool(args.offline),
        },
        layout_mode=loaded.layout.mode,
        image_mean=float(loaded.image_mean),
        image_std=float(loaded.image_std),
    )
    record_enabled = threading.Event()
    if args.record:
        record_enabled.set()
        _log(f"deploy RECORDING armed → {record_dir} (a take per execution; "
             "toggle it live in the page)")

    # --- live sensors + S0 driver ---
    # The RecordTap (:8779) is the low level's OWN broadcast of its live joint state +
    # the pelvis-frame F_meas the servo consumes (opt-in, RECORD=1). It is read-only and
    # entirely optional: if it's absent (RECORD not set) the planner degrades gracefully —
    # the mesh falls back to the predicted-chunk IK render, and the F/T panel shows the
    # :8771 sensor wire instead of the exact servo F_meas. Never gates predict/execute.
    # The F/T tap feeds the model's force ring on ARRIVAL (see `_force_on_msg` below), at the
    # sensor's own ~200 Hz, so the window the encoder sees is the anti-aliased mean the
    # converter built. The other three are polled by the loops that want current state.
    _force_ring_sink: dict = {"fn": None}  # set once `planner` exists; frames before that drop
    # The page's live F_meas charts. Filled HERE, on the socket thread, so the trace is the
    # sensor's own ~200 Hz anti-aliased down to the chart rate — see _FtDisplayTrace. Created
    # before the subscriber (unlike the model's ring, which needs `planner`) so no frame is lost,
    # and it is deliberately fed even when the checkpoint reads no force at all: the 40 N hard
    # stop is driven by this sensor, so the operator wants the trace regardless.
    ft_trace = _FtDisplayTrace()
    # The ⏺ RUN RECORDER's taps (rollout_record.RolloutTaps): one holder whose methods are called at
    # the publish funnel (_s0_send), the three executors and the three bridge subscribers below.
    # `taps.recorder` is None until _run_viser's RECORD block installs one, so until then — and in
    # every test that builds this page without it — each call is one `is None` check.
    taps = rollout_record.RolloutTaps()

    def _force_on_msg(msg, t_mono):
        # Display first, and in its own try: the model's ring is the load-bearing consumer and a
        # charting bug must not cost it a frame. (_WSSubscriber already guards the callback as a
        # whole, but that would drop the ring write too.)
        try:
            ft_trace.push(_pick_force_channels(msg, ("left",), require_6dof=False), t_mono)
        except Exception:  # noqa: BLE001 — viz-only
            pass
        fn = _force_ring_sink["fn"]
        if fn is not None:
            fn(msg, t_mono)
        # The run record sees EVERY F/T frame on arrival (~250 Hz), never a `latest()` sample.
        taps.on_force(msg, t_mono)

    subs = {
        "force": _WSSubscriber(args.force_ws, "force", on_msg=_force_on_msg),
        # The run record's camera + RecordTap taps ride the same per-frame hook: the camera frame
        # lands as the bridge's own JPEG, the RecordTap frame verbatim (+ the FK'd wire points).
        "camera": _WSSubscriber(args.camera_ws, "camera", on_msg=taps.camera_hook(0)),
        "gripper": _WSSubscriber(args.gripper_ws, "gripper"),
        "record": _WSSubscriber(args.record_ws, "record", on_msg=taps.on_record),
    }
    # OFFLINE leaves every socket CLOSED (no thread started, no S0 publish). The subscriber
    # objects still exist so every `subs[...].latest()` call site keeps working and simply reports
    # "no frame" — the same degraded path a dead bridge already takes — rather than needing an
    # `if args.offline` at each of them.
    if not args.offline:
        for s in subs.values():
            s.start()
    calib = GripperCalib(width_open_m=0.095, width_closed_m=0.0)

    from flore.client.s0_driver import S0Driver

    s0 = S0Driver(host=args.s0_ws_host, port=args.s0_ws_port)
    if not args.offline:
        s0.start()

    if args.offline:
        _log("OFFLINE: no bridges opened, nothing published to :8770. Execute is DISABLED; "
             "this page is for inspecting the FRAME (URDF ready pose + triads + recorded path).")
    elif not args.no_selfcheck:
        rep = _self_check(subs, require_6dof=not args.allow_3dof_force, replay_only=replay_only,
                          force_advisory=not consumes_force)
        _log(f"self-check: {rep}")

    # --- shared control state (live-adjustable from the viser page) ---
    stop = threading.Event()
    run_continuous = threading.Event()
    dripping = threading.Event()
    grip_enabled = threading.Event()
    if args.command_gripper:
        grip_enabled.set()
    jaw_ever_commanded = {"v": False}
    # The force master switch + gate + a seed counter, guarded by force_lock so a read
    # (switch/gate at publish time, or seed issuance at infer time) is a consistent,
    # atomic snapshot across the control/heartbeat/drip threads.
    force_lock = threading.Lock()
    force_ctl = {"enabled": bool(args.enable_force_cmd), "gate_n": float(args.force_gate_n), "seed": 0}
    # execute-N-of-S selector: how many of the S predicted steps to actually drip to the
    # robot on Run (the rollout still PREDICTS all S; this only limits execution). Default 4.
    # Live-adjustable in the viser page. SNAP the initial value to the offered dropdown
    # choices {4, 8, S} so the dropdown label and the executed count always AGREE (an
    # arbitrary --exec-steps like 12 would otherwise show "4" while executing 12).
    _req_exec = int(args.exec_steps) if args.exec_steps else 4
    _snapped_exec = snap_exec_steps(_req_exec, _EXEC_CHOICES)
    if _snapped_exec != _req_exec:
        _log(f"--exec-steps={_req_exec} not in {list(_EXEC_CHOICES)}; snapped to {_snapped_exec}.")
    exec_ctl = {"n": _snapped_exec}
    # Say the jaw is read-only AT STARTUP, not only in the page. The failure it prevents is
    # "the arm goes to the object and never closes", which looks like a policy fault; the gripper
    # bridge meanwhile prints `target_frac=freeze` because it was never sent a target at all.
    if args.command_gripper and not replay_only:
        _log(f"gripper: overshoot {args.grip_overshoot_mm:g} mm applied when the policy asks "
             f"for < {args.grip_close_below_mm:g} mm for {args.grip_confirm_steps} consecutive "
             f"steps, and cancelled after the same run of OPEN predictions. The recorded "
             f"'closed' width is the TOOL's thickness (42-48 mm measured), so the overshoot is "
             f"what turns a position target into grip force; the confirmation run is what stops "
             f"an ensembled BIMODAL blend from loosening the grasp mid-hold.")
    if not args.command_gripper and not replay_only:
        _log("GRIPPER IS READ-ONLY (no --command-gripper): the model's predicted jaw width is "
             "read for proprioception and then DISCARDED — the jaw will never close and the "
             "gripper bridge will sit at 'target_frac=freeze'. Relaunch with --command-gripper "
             "to let the planner drive the LEFT jaw.")
    # TEMPORAL ENSEMBLING (ACT only) — upstream's --temporal_agg. Guarded by ens_lock, read
    # as one snapshot at the top of an execute so a mid-drip toggle cannot switch cadence
    # underneath a running chunk (it takes effect on the NEXT Run, like the exec-steps
    # selector). ``supported`` is fixed at load time from the checkpoint's own config; a
    # flow/diffusion or model-free session can never enable it, so the UI offers it disabled
    # with the reason spelled out rather than silently doing nothing.
    #
    # DEFAULT = the checkpoint's own ``temporal_ensemble`` flag, so an ACT checkpoint trained
    # for ensembling runs ensembled unless the operator turns it OFF. That matters: the flag
    # is baked in at train time precisely so serving cannot silently disagree with it.
    ens_lock = threading.Lock()
    _ens_supported = bool(getattr(planner, "temporal_ensemble_enabled", False)) and not replay_only
    # ``pending_reset`` is an ABANDONED-INTENT request: Stop and the low-level-lost watchdog run
    # on the viser-callback / status threads and must not touch the ensembler directly (a blend
    # may be in flight on the drip thread, which owns it under `busy`), so they set this flag and
    # _drip_ensembled applies it before its next add.
    ens_ctl = {"enabled": _ens_supported, "supported": _ens_supported, "n_blended": 0,
               "pending_reset": False}
    # SEQUENTIAL MODEL ROLLOUT state ("Run to completion"): the policy analogue of the replay
    # panel's Play button. ``gen`` is bumped by Stop so an in-flight loop retires itself
    # (identical to replay_ctl["auto_gen"]); ``max_s`` bounds the rollout in WALL-CLOCK seconds
    # because a policy — unlike a recorded episode — has no end of its own to reach.
    auto_model_lock = threading.Lock()
    auto_model_ctl = {"running": False, "gen": 0, "max_s": float(args.auto_max_seconds)}
    # GRIPPER CLOSING OVERSHOOT (see GripperCalib.to_target_frac_squeezed). Read as one atomic
    # snapshot at each publish so a live slider change acts on the very next commanded step.
    grip_lock = threading.Lock()
    # ONE hand on this planner (left_only), so ONE dict; the 3-tracker planner builds one per
    # gripper role from the same constructor.
    grip_ctl = new_grip_ctl(overshoot_mm=args.grip_overshoot_mm,
                            close_below_mm=args.grip_close_below_mm,
                            confirm_steps=args.grip_confirm_steps,
                            hold_tightest=bool(args.grip_hold_tightest))

    _grip_target_frac, _grip_release_latch = build_grip_command(calib, grip_ctl, grip_lock)
    if _ens_supported:
        _log(f"temporal ensembling AVAILABLE (algo={loaded.algorithm}, k="
             f"{float(getattr(loaded.config, 'temporal_ensemble_k', 0.01)):g}) and ON by default: "
             f"Run re-queries the policy EVERY step and executes the exponentially-weighted "
             f"blend of the live predictions. Toggle it in the page's Control folder.")
    # RECORDED-EPISODE REPLAY state, guarded by replay_lock. ``source`` is the live
    # predict/execute chunk SOURCE ("model" | "replay"); ``episode``/``frame`` is the replay
    # cursor (frame is LOCAL to the episode). ``auto`` drives the sequential
    # play-the-whole-episode mode. All of them are read as one atomic snapshot at the top of
    # a Predict so a mid-inference dropdown change can never splice two episodes' frames into
    # one executed chunk.
    replay_lock = threading.Lock()
    replay_ctl = {
        "source": "replay" if replay is not None else "model",
        "episode": int(args.replay_episode) if replay is not None else 0,
        "frame": 0,
        "auto": False,      # sequential auto-advance requested (set by the Play button)
        "auto_gen": 0,      # bumped by Stop/seek so an in-flight auto loop retires itself
    }
    auto_running = threading.Event()  # a sequential replay loop is currently executing
    # A single non-reentrant busy guard so only ONE control action (predict / run_staged /
    # a continuous step) touches the GPU model + the shared IK solver at a time — a
    # double-click Predict or Predict-during-continuous would otherwise run two concurrent
    # model.sample_actions + corrupt the single ThreePointIK solver.
    busy = threading.Lock()
    # GIL PRIORITY for the inference call. The ghost-robot arm IK runs on its own thread but is
    # pure Python, and a sample_actions is ~6100 tiny CUDA launches driven from Python (cpu/wall
    # = 1.00) — so the two do not overlap, they interleave, and each stretches the other.
    # Measured with ckpt_0814_diff_umi (ViT-B + UNet, 16 DDIM steps): a predict costs 46 ms with
    # the ghost idle, 3543 ms with it solving (77x), 59 ms with it yielding through this gate.
    # GPU utilization during the fight is ~1.3% — the GPU is idle because the thread that feeds
    # it cannot get the interpreter. See flore/s1_policy/serving/gil_yield.py.
    infer_gate = YieldGate()
    staged = {"out": None}
    force_roles = loaded.layout.force_roles
    # A pure-position model reads no wrench, so "the 6-DoF pipe is required" has nothing
    # behind it; leaving it True would make _pick_force_channels freeze on every sample.
    require_6dof = (not args.allow_3dof_force) and consumes_force

    # --- F/T -> force ring, on ARRIVAL at the sensor's own rate ---
    # NOT in the obs_hz ingest loop below, where it used to live. The model's force input is
    # the anti-aliased MEAN of every raw sample in a FORCE_HIST_SPAN_S window (measured ~6.8
    # samples at the 204 Hz F/T rate); pushing one `latest()` per 30 Hz tick fed it a
    # decimated sample instead, and the ring's count-based sizing then averaged 3 of those
    # over ~100 ms — a 3x-too-wide window of an aliased signal. Pushing per arrival plus the
    # planner's time-keyed ring reproduces the training filter at any bridge rate.
    _ft_dropout_logged = {"v": False}

    def _push_force_from_msg(msg, t_mono):
        fw = _pick_force_channels(msg, force_roles, require_6dof)
        if fw is not None:
            planner.push_force(fw, t=t_mono)
            # RECORD the raw sample at the sensor's own rate, on the SAME clock the model's
            # window is binned from — so an offline re-inference can rebuild that window
            # rather than approximate it from a decimated copy.
            #
            # `elapsed(t_mono)`, NOT `t_mono`: every stored `t` is TAKE-RELATIVE (see
            # TakeRecorder.elapsed's contract). Passing raw monotonic here put the force stream
            # on a different epoch from executed/commanded — so nothing could be read at a
            # common instant, and a reader asking "what did the sensor feel at t" always landed
            # on sample 0. `elapsed` takes the caller's own `now`, so the sensor's timestamp is
            # preserved exactly; only the zero changes.
            recorder.on_force(recorder.elapsed(t_mono), fw)
            _ft_dropout_logged["v"] = False
        elif require_6dof:
            if not _ft_dropout_logged["v"]:
                _log("INGEST: F/T frame lacks 6-DoF pipe — window marked unmeasured.")
                _ft_dropout_logged["v"] = True
            planner.push_force(None, t=t_mono)  # invalid window -> learned no-data embed

    _force_ring_sink["fn"] = _push_force_from_msg

    # --- ingest loop: keep the rings fresh at obs_hz ---
    # EVERY iteration is wrapped: this thread is the ONLY writer of the image and proprio
    # rings, so an escaping exception froze the entire observation permanently while Predict
    # kept returning confident chunks off the last good frame — no page indication, no refusal,
    # only a traceback in a terminal nobody is watching mid-run. It was not hypothetical: a
    # bridge publishing `"open": null` makes `float(gmsg.get("open", 4.9))` raise (a `.get`
    # default only applies to a MISSING key, not a null value), and neither `base64.b64decode`
    # nor `push_image` is guarded either. Now one bad frame costs one tick, is logged
    # rate-limited, and shows up as staleness in the page if it keeps happening.
    _ingest_err = {"n": 0, "last": 0.0}

    def _ingest_once(stride: int, tick: int) -> None:
        gmsg, _gw, _gn = subs["gripper"].latest()
        if gmsg is not None and gmsg.get("actual") is not None:
            width = calib.to_width_m(
                float(gmsg["actual"]), float(gmsg.get("open", 4.9)), float(gmsg.get("closed", 0.0)))
            planner.push_gripper("left", width)
        # Image AND proprio are pushed on the SAME stride so the To obs frames (image +
        # state) are all obs_history_dt_s apart — the time-anchored history the model was
        # trained on. (Pushing proprio every tick made the To state frames ~1 tick apart
        # while the images were stride apart → a state/image temporal mismatch.) Force is
        # NOT here: it rides the F/T socket's own arrival callback (see above).
        if tick % stride == 0:
            # MODEL-FREE replay consumes no observation, so skip the JPEG decode +
            # resize entirely (it is the most expensive thing in this loop and nothing
            # would ever read the ring). Proprio is likewise model-only.
            cmsg, _cw, _cn = subs["camera"].latest()
            if cmsg is not None and cmsg.get("jpeg_b64"):
                # RECORD the bridge's ORIGINAL jpeg (not the 224 crop): it is what the
                # camera actually saw, so a re-inference can re-crop it exactly as the live
                # ingest does, and a human can see the whole scene. Deduped by the bridge's
                # own seq inside the recorder, so this poll cannot inflate the stream past
                # the camera's real rate. Kept OUTSIDE the `not replay_only` guard below
                # because storing bytes needs no JPEG decode — a model-free replay should
                # still record what the camera saw.
                # `elapsed()`, not raw monotonic — take-relative like every other stream (same
                # bug/fix as on_force above; on the wrong epoch the replay viewer's camera
                # panel froze on the take's first frame for the whole scrub).
                recorder.on_camera(recorder.elapsed(), base64.b64decode(cmsg["jpeg_b64"]),
                                   int(cmsg.get("seq", -1)))
                if not replay_only:
                    img = _decode_jpeg_b64(cmsg["jpeg_b64"])
                    if img is not None:
                        planner.push_image(_resize_center_crop_224(img))
            if not replay_only:
                planner.push_proprio_from_anchor()

    def _ingest_loop():
        period = 1.0 / max(args.obs_hz, 1.0)
        stride = max(int(args.img_stride), 1)
        tick = 0
        while not stop.is_set():
            t0 = time.time()
            try:
                _ingest_once(stride, tick)
            except Exception:  # noqa: BLE001 — a bad frame must cost ONE tick, not the thread
                _ingest_err["n"] += 1
                now = time.monotonic()
                if now - _ingest_err["last"] > 5.0:
                    _ingest_err["last"] = now
                    import traceback
                    _log(f"INGEST error #{_ingest_err['n']} (rings not updated this tick):\n"
                         + traceback.format_exc())
            tick += 1
            dt = time.time() - t0
            if dt < period:
                time.sleep(period - dt)

    threading.Thread(target=_ingest_loop, daemon=True, name="ingest").start()

    # --- idle heartbeat: republish the LAST wire so the pre-arm pose gate can pass ---
    _wire_lock = threading.Lock()
    _hp0, _ho0 = mapper.full_wire(*mapper.left_home_tcp_anchor())
    last_wire = {"pos": _hp0, "orn": _ho0, "fcmd": None}
    # IDLE CLUTCH TRACKING: while idle (not dripping), the heartbeat continuously refreshes
    # the wire to the robot's MEASURED pose (RecordTap FK) via this hook, so the wire the low
    # level gates against == the robot. This is what lets the operator ARM at any time with ~0
    # gap and NO jump — the whole reason the fixed home-anchor wire was refused (cross-policy
    # home was 0.31 m off). Set by _run_viser once the measured-EE FK is built; None ⇒ no live
    # source (RECORD off / FK unavailable) ⇒ the heartbeat holds the last wire (home seed).
    #
    # ...but NOT at `drip_hz`. The wire must be RE-PUBLISHED at drip_hz (that is what keeps it
    # inside S0Driver's 0.5 s freshness window); re-DERIVING the measured pose behind it is a
    # separate, far more expensive job — `_idle_sync_wire` runs a full yourdfpy FK and rebuilds
    # the 3-point wire — and it does not need the same rate. Measured on this box, the heartbeat
    # thread cost 16.1% of a core at 60 Hz and 0.4% with the clutch skipped (which is what
    # happens once a chunk is staged), i.e. the clutch alone was ~2.6 ms of GIL-held Python
    # every 16.7 ms, against a render loop that was already missing its 30 Hz deadline.
    #
    # 10 Hz is safe because of WHAT the clutch is for: passing the low level's pre-arm pose
    # gate, whose tolerance is 70 mm. The arm is NOT being commanded while this runs (it holds
    # the ready pose under its own controller), and at the measured per-tick joint motion of a
    # holding G1 (p50 0.0026 rad over a 53 ms tick, across 34061 recorded render ticks) 100 ms
    # of staleness is single-digit millimetres of TCP drift — 3 orders inside the gate. The
    # freeze-once-armed rule is unaffected: `_idle_sync_wire` re-reads `vive_armed` itself and
    # returns early, and at the instant of arming the wire already EQUALS the measured pose
    # (that is the invariant the clutch maintains), so a sync landing up to 100 ms after arming
    # re-publishes the pose that is already there.
    IDLE_CLUTCH_HZ = 10.0
    wire_sync = {"fn": None}

    def _set_last_wire(pos, orn, fcmd):
        with _wire_lock:
            last_wire["pos"] = np.asarray(pos, dtype=np.float64)
            last_wire["orn"] = np.asarray(orn, dtype=np.float64)
            last_wire["fcmd"] = None if fcmd is None else np.asarray(fcmd, dtype=np.float64)

    def _last_wire_snapshot():
        """Atomic (pos, orn, fcmd) copy of the current broadcast wire. Lets callers that only
        want to drop the latched force re-publish the SAME pose without racing the writer."""
        with _wire_lock:
            hf = last_wire["fcmd"]
            return (last_wire["pos"].copy(), last_wire["orn"].copy(),
                    None if hf is None else hf.copy())

    def _rollout_fcmd6(fcmd):
        """The run record's ``fcmd`` is always ``[L xyz, R xyz]`` (6) or None — the five-point page's
        convention, which the plots read. This wire carries the LEFT triple alone, and the low level
        zeroes the RIGHT (FORCE_HANDS=left), so the right half is recorded as the 0 it is."""
        if fcmd is None:
            return None
        f = np.asarray(fcmd, dtype=np.float64).reshape(-1)
        return f if f.shape[0] == 6 else np.concatenate([f[:3], np.zeros(3)])

    def _s0_send(pos, orn, fcmd):
        # S0Driver.send accepts an optional f_cmd (the phase-1 wire extension). When the
        # master switch is off / below-gate, fcmd is None and the wire is pose-only
        # (byte-identical to the position-only deploy).
        if fcmd is None:
            s0.send(pos, orn)
        else:
            s0.send(pos, orn, f_cmd=fcmd)
        # The run record gets the frame AS PUBLISHED — the drip's, the heartbeat's re-sends, the
        # reach residual applied — after the send. No gate lives here (the low level validates
        # width; finiteness is the planner's), so `ok` is the fact that it went out.
        taps.publish((pos, orn), _rollout_fcmd6(fcmd), True)

    def _heartbeat_loop():
        period = 1.0 / max(args.drip_hz, 1.0)
        clutch_period = 1.0 / IDLE_CLUTCH_HZ
        next_clutch = 0.0  # monotonic deadline; 0.0 ⇒ the first idle tick syncs immediately
        while not stop.is_set():
            if not dripping.is_set():
                try:
                    # IDLE CLUTCH: while the low level is NOT yet armed, refresh the wire to
                    # the robot's MEASURED pose so the pre-arm gate passes with ~0 gap
                    # (best-effort; a stale/absent frame holds the last wire; _idle_sync_
                    # wire itself freezes once vive_armed). Rate-limited to IDLE_CLUTCH_HZ —
                    # see the note beside that constant; the wire below is still re-published
                    # every tick, so the freshness window is untouched. We take `busy`
                    # (non-blocking) across BOTH the staged check AND sync() so the whole
                    # decision is atomic w.r.t. Predict/Run/Reset — they all hold `busy` and
                    # own staged['out'] + mapper.hold_* + the decode anchor, so under `busy` a
                    # staged plan can't be overwritten and mapper.hold_* can't be read torn.
                    # If `busy` is held (a chunk is predicting/running) we simply skip this
                    # tick and hold the wire — and do NOT consume the clutch deadline, so the
                    # next tick retries rather than waiting out another clutch period.
                    _t = time.monotonic()
                    sync = wire_sync["fn"]
                    if sync is not None and _t >= next_clutch and busy.acquire(blocking=False):
                        next_clutch = _t + clutch_period
                        try:
                            if staged["out"] is None:  # never chase measured while a plan is staged
                                sync()  # updates last_wire in place iff a fresh, un-armed frame exists
                        finally:
                            busy.release()
                    with _wire_lock:
                        hp = last_wire["pos"]
                        ho = last_wire["orn"]
                        hf = last_wire["fcmd"]
                    # SAFETY (master switch is authoritative on EVERY tick): when the switch
                    # is OFF the idle wire is pose-only regardless of any latched f_cmd, so
                    # toggling OFF / Stop / Reset makes the low level a pure position
                    # controller immediately — the heartbeat can never republish a stale
                    # force. (When ON, hf may still be a step's gated force held between
                    # chunks, matching the arm holding its last commanded pose.)
                    with force_lock:
                        switch_on = force_ctl["enabled"]
                    _s0_send(hp, ho, hf if switch_on else None)
                except Exception as e:  # noqa: BLE001
                    _log(f"heartbeat error: {e!r}")
            time.sleep(period)

    threading.Thread(target=_heartbeat_loop, daemon=True, name="heartbeat").start()

    # --- the model seam: one inference → wire targets + PER-STEP absolute poses + the
    #     UN-gated predicted wrench (the gate is applied LIVE at publish time in _drip,
    #     so toggling the switch/gate acts as an interrupt on an already-staged chunk). ---
    def _next_seed() -> int:
        """Issue the next inference seed atomically.

        Shared by the chunk cadence (:func:`_infer`) and the per-step ensembled cadence, so
        the two can never hand the model the same seed for different observations."""
        with force_lock:  # atomic seed issuance (the switch/gate are read LIVE at publish)
            seed = int(force_ctl["seed"])
            force_ctl["seed"] = seed + 1  # vary the noise per inference so re-predict differs
        return seed

    def _infer() -> dict:
        """Run one inference (LEFT chunk) and build the drip payload. Does NOT advance
        the anchor (commit happens only after a real drip) — safe for preview. The gate is
        NOT baked in here: we return the ungated per-step wrench + the per-step decoded
        anchor pose, and _drip re-evaluates the gate each step against the LIVE force_ctl."""
        # The ⏺ RUN RECORDER opens its folder at the FIRST Predict — this manual one or ▶▶'s — once
        # the button is armed (or on every_run) and stays open across chunks until Reset / the low
        # level disarming / the button. A no-op while one is open; one `is None` check without it.
        taps.begin("sequential")
        seed = _next_seed()
        # Infer with force ENABLED + gate 0 so the model's raw predicted wrench is returned
        # un-gated; the LIVE gate is applied per-step in _drip. (Pose/gripper are unaffected
        # by the gate, so this does not change them.)
        #
        # ``capture`` collects the EXACT observation + raw chunk of THIS inference for the
        # recorder. It is filled by the same call that feeds the model, so the recorded obs is
        # the fed obs — not a second ring read a tick later, which at 30 Hz would routinely be
        # a different image. Carried on the payload and only turned into a take entry if this
        # plan is actually EXECUTED (the operator's "按run的时候为标准").
        capture: dict = {}
        # infer_gate: keep the pure-Python ghost IK off the interpreter for the duration of the
        # sample (77x on a diffusion checkpoint — see the gate's definition in main()).
        _t_sample0 = time.monotonic()
        with infer_gate.priority():
            result = planner.infer_chunk(seed=seed, force_enabled=True, gate_n=0.0,
                                         capture=capture)
        out = _payload_from_chunk(result.role("left"))
        out["capture"] = capture
        out["t_predict"] = time.monotonic()
        out["sample_ms"] = (out["t_predict"] - _t_sample0) * 1e3  # for the run record only
        return out

    def _infer_replay() -> dict:
        """Build the SAME drip payload from a RECORDED episode instead of the model.

        The recorded chunk arrives as anchor-RELATIVE deltas (exactly the model's own output
        form — see :mod:`flore.s1_policy.serving.hl_replay`) and is decoded by the SAME
        ``decode_chunk`` against the SAME live anchor, so every downstream stage (URDF flip,
        wire build, force gate, drip, anchor commit) is byte-identical to the model path.
        The recorded WORLD frame never enters: only its deltas do, so replay reproduces the
        demonstrated motion from wherever the arm actually is (the clutch semantic).

        ⚠ THE DEMO IS REPLAYED REORIENTED, BY ~107°, AND THAT IS CORRECT. The body-frame
        transport rotates the stroke by ``R_wrist_robot @ R_wrist_handheld^T`` — the physical
        angle between how the human held the tool and how the G1 holds it at its ready pose
        (44° nose-down). It is NOT a frame bug: it is what preserves the tool's attack angle to
        its own path (to 2e-14°, vs 50° mean error for a world-frame variant), keeps the
        lighthouse world cancelling exactly, keeps every episode reachable, and keeps the chunk
        equal to the policy's training target. See
        :func:`flore.s1_policy.serving.hl_replay._body_vs_world_transport` for the measurements.

        Returns the payload plus the replay identity/cursor bookkeeping ``_drip`` needs to
        advance the episode after a real execution."""
        from flore.s1_policy.serving.hl_online import decode_chunk

        with replay_lock:  # one atomic cursor snapshot — never splice two episodes' frames
            episode = int(replay_ctl["episode"])
            frame = int(replay_ctl["frame"])
        rc = replay.chunk_at(episode, frame)
        anchor_pos, anchor_rot6d = planner.snapshot_anchor()
        result = decode_chunk(
            rc.actions, loaded.layout, anchor_pos, anchor_rot6d,
            force_enabled=True, gate_n=0.0,  # ungated; the LIVE gate is applied in _drip
        )
        out = _payload_from_chunk(result.role("left"))
        out["replay"] = dict(
            episode=episode, frame=frame, n_valid=rc.n_valid, time_s=rc.time_s,
            gripper_valid=rc.gripper_valid, force_valid=rc.force_valid,
        )
        # The RECORDED action array + the anchor it was decoded against, so a recorded take of a
        # REPLAY looks like a take of a model run: same "here is the chunk that ran, here is the
        # anchor it branched from" shape, with source="replay" saying where it came from. There
        # is no observation — no model was queried — and the reader says so rather than
        # fabricating one.
        out["replay_actions"] = np.asarray(rc.actions, dtype=np.float64)
        out["capture"] = {"anchor_pos": anchor_pos, "anchor_rot6d": anchor_rot6d}
        out["t_predict"] = time.monotonic()
        return out

    def _infer_replay_openloop(episode: int, start_frame: int) -> dict:
        """Build ONE open-loop payload for the WHOLE episode from ``start_frame``.

        The open-loop sibling of :func:`_infer_replay`. It reads the entire episode as a single
        anchor-relative :class:`EpisodePlan` and decodes it ONCE against ONE snapshot of the
        anchor (the pose the arm is at right now, seeded/re-synced before this call), so the
        commanded trajectory is a single rigid placement of the demonstration — no per-chunk
        re-anchor, and therefore no accumulation of the servo's per-chunk lag into the frame the
        next chunk decodes from. Everything else (URDF flip, wire build, force gate, drip) is the
        SAME ``decode_chunk`` + ``_payload_from_chunk`` the per-chunk paths use, so only the
        LENGTH of the chunk and its single-anchor decode differ.

        Returns the payload plus a ``replay_openloop`` marker (the plan's identity + whether it
        was truncated at a glitch hole) so ``_drip`` and the sequential runner can tell an
        open-loop play from a per-chunk one and skip the cursor bookkeeping (there is no cursor:
        the whole episode is one chunk)."""
        from flore.s1_policy.serving.hl_online import decode_chunk

        plan = replay.episode_plan(episode, start_frame)
        anchor_pos, anchor_rot6d = planner.snapshot_anchor()
        result = decode_chunk(
            plan.actions, loaded.layout, anchor_pos, anchor_rot6d,
            force_enabled=True, gate_n=0.0,  # ungated; the LIVE gate is applied in _drip
        )
        out = _payload_from_chunk(result.role("left"))
        out["replay"] = dict(
            episode=episode, frame=start_frame, n_valid=plan.n_frames,
            time_s=float(plan.times_s[0]),
            gripper_valid=plan.gripper_valid, force_valid=plan.force_valid,
        )
        out["replay_openloop"] = dict(
            episode=episode, start_frame=start_frame, n_frames=plan.n_frames,
            truncated=plan.truncated,
        )
        out["replay_actions"] = np.asarray(plan.actions, dtype=np.float64)
        out["capture"] = {"anchor_pos": anchor_pos, "anchor_rot6d": anchor_rot6d}
        out["t_predict"] = time.monotonic()
        return out

    def _payload_from_chunk(lc) -> dict:
        """Turn ONE decoded LEFT :class:`RoleChunk` into the drip payload.

        Shared by the model and recorded-replay sources so they cannot diverge in the
        frame conversion, the wire lever, or the commit-anchor bookkeeping — the only
        difference between the two modes is where ``lc`` came from."""
        S = lc.position.shape[0]
        # ``lc.rot6d`` is the decoded absolute LEFT orientation in the MODEL/stored frame (the
        # anchor is kept in that frame). Convert to the URDF wrist frame for everything that faces
        # HARDWARE or the RENDER (the wire the low level servos to, and the yellow path/mesh that
        # must register on the live URDF robot). Positions are frame-agnostic absolute pelvis-frame
        # points (and the +x TCP reach lever is seam-invariant), so only orientation converts.
        # The MODEL-frame ``lc.rot6d`` is kept ONLY for ``step_rot6d`` → the next decode anchor.
        urdf_rot6d = np.stack([_model_rot6d_to_urdf(r) for r in lc.rot6d])  # (S,6) URDF wrist frame
        wire_pos = np.zeros((S, 9), dtype=np.float64)
        wire_orn = np.zeros((S, 12), dtype=np.float64)
        for i in range(S):
            wp, wo = mapper.full_wire(lc.position[i], urdf_rot6d[i])
            wire_pos[i] = wp
            wire_orn[i] = wo
        pred_wrench = lc.wrench_pred if lc.wrench_pred is not None else np.zeros((S, 6))
        return dict(
            tcp_pos=lc.position, tcp_rot6d=urdf_rot6d,  # URDF frame → viz lands on the live mesh
            wire_pos=wire_pos, wire_orn=wire_orn,
            pred_wrench=np.asarray(pred_wrench, dtype=np.float64),  # (S,6) UNGATED sensor-frame
            # Per-step decoded anchor pose (UMI-TCP) so commit_anchor can land on the LAST
            # EXECUTED step, not the chunk end (the execute-N-of-S selector). step_rot6d stays in
            # the MODEL frame — it becomes the next decode anchor (commit_role), which must match
            # the frame delta_ee_to_absolute + wrt_start are computed in.
            step_pos=lc.position, step_rot6d=lc.rot6d,
            force_pred=pred_wrench,  # for the viz (ungated — the chunk's own wrench)
            grip_m=(lc.gripper_m if lc.gripper_m is not None else np.zeros(S)),
        )

    def _infer_active() -> dict:
        """One chunk from the ACTIVE source (live Source toggle): model or recorded replay."""
        with replay_lock:
            use_replay = replay_ctl["source"] == "replay"
        if use_replay and replay is not None:
            return _infer_replay()
        if replay_only:
            # Defence in depth: the page never OFFERS the model choice in this mode (the
            # Source dropdown is replay-only), so reaching here means a wiring bug rather
            # than an operator action. Say so instead of tripping over a None model.
            raise RuntimeError(
                "no policy is loaded (model-free --replay-zarr session) — the chunk source "
                "cannot be 'model'. Relaunch with --algo/--checkpoint to run the policy.")
        return _infer()

    # --- RECORDING seam: one place that turns an executed payload into a take entry -------
    # The three executors (_drip, _drip_openloop, _drip_ensembled) all call this, so the take's
    # notion of "an executed prediction" cannot diverge per cadence. `chunk_seq` numbers the
    # executed chunks WITHIN a take so the measured/commanded rows can be attributed to the plan
    # they belong to; it is reset when a take is armed.
    rec_chunk = {"i": -1}
    # WHERE THE ARM IS IN THE PLAN, published for the RENDER thread. The measured stream is
    # sampled by the render loop (the only thread already FKing the live wrist), which knows
    # nothing about which chunk/step is being dripped — so the executors post it here and the
    # render loop stamps each measured row with it. Plain ints under the GIL: a torn read would
    # only mislabel one row's provenance, never corrupt a pose, so a lock on the 30 Hz render
    # path is not worth it. -1 = not dripping / an ensembled step (which has no chunk).
    rec_step = {"chunk": -1, "i": -1}

    def _record_finish(reason: str) -> str | None:
        """COMMIT the open take to disk (no-op when none is open). Returns its path.

        A take is committed at the end of an execution UNIT, not of every chunk: Play /
        open-loop / sequential each close their own, while consecutive manual Runs accumulate
        into one (they walk a single trajectory — the same reason the red trace is not reset
        per press) and are committed by whatever ends that attempt: Reset, Clear, a new take
        arming, the recording toggle going off, or shutdown."""
        try:
            path = recorder.close()
        except Exception as e:  # noqa: BLE001 — recording must never break the control path
            _log(f"record close error (ignored): {e!r}")
            return None
        if path:
            _log(f"recorded take ({reason}) → {path}")
        return path

    def _record_arm(label: str) -> None:
        """Open a take for one execution unit, if recording is enabled.

        Commits any take still open first, so starting a Play right after a few manual Runs
        writes those Runs out rather than silently splicing two different attempts into one
        file."""
        _record_finish("superseded")
        if not record_enabled.is_set():
            return
        rec_chunk["i"] = -1
        recorder.arm(label)

    def _record_prediction(out: dict, n_exec: int, t_exec_start: float) -> int:
        """Register the plan that is about to be dripped → its chunk index in the take.

        Returns -1 when nothing is being recorded, which the executors pass straight through to
        the per-step calls (they no-op on a closed take anyway)."""
        if not recorder.recording:
            return -1
        cap = out.get("capture") or {}
        rp = out.get("replay")
        rec_chunk["i"] += 1
        try:
            recorder.on_prediction(RecordedPrediction(
                obs=cap.get("obs"),
                anchor_pos=cap.get("anchor_pos", np.zeros((len(loaded.layout.roles), 3))),
                anchor_rot6d=cap.get("anchor_rot6d", np.zeros((len(loaded.layout.roles), 6))),
                # A REPLAYED chunk has no sampler output; store the RECORDED action array that
                # was decoded instead, so the take is uniform ("here is the chunk that ran")
                # and the `source` field says where it came from.
                prediction=cap.get("prediction", out.get("replay_actions", np.zeros((0, 0)))),
                seed=int(cap.get("seed", -1)),
                num_steps=int(cap.get("num_steps", 0)),
                t_predict=recorder.elapsed(out.get("t_predict", t_exec_start)),
                t_exec_start=recorder.elapsed(t_exec_start),
                n_exec=int(n_exec),
                source="replay" if rp is not None else "model",
                replay_episode=int(rp["episode"]) if rp is not None else -1,
                replay_frame=int(rp["frame"]) if rp is not None else -1,
            ))
        except Exception as e:  # noqa: BLE001 — recording must never break an execution
            _log(f"record prediction error (ignored): {e!r}")
        return rec_chunk["i"]

    def _record_commanded(chunk_i: int, step_i: int, pos, orn, grip_m, fcmd) -> None:
        """Log ONE published wire step. No-ops on a closed take."""
        if not recorder.recording:
            return
        recorder.on_commanded(
            recorder.elapsed(), wire_pos9=pos, wire_orn12=orn,
            grip_cmd_m=grip_m, f_cmd=fcmd, chunk_index=chunk_i, step_index=step_i)

    # --- the ⏺ RUN RECORDER's chunk seam: the SAME place as _record_prediction (an EXECUTED plan),
    # one call per executor, so the run record's notion of "a chunk" cannot diverge per cadence
    # either. `rollout_seq` numbers the executed chunks for the whole page (the record's `chunk` /
    # `seq` join key; a recording spans one ▶▶ run, so it only ever sees a monotone slice of it).
    rollout_seq = {"n": 0}
    _rollout_grip_roles = tuple(loaded.layout.gripper_roles)

    def _rollout_chunk(out: dict, cadence: str, n_exec: int) -> int:
        """Record the plan about to be dripped (the run record's ``chunks.jsonl``) → its seq."""
        rollout_seq["n"] += 1
        seq = rollout_seq["n"]
        # A chunk that did not come through `_infer` (a loaded recorded chunk, a replay row) opens the
        # recording too, if armed — the same rule at the executors' seam.
        taps.begin(cadence)
        r = taps.recorder
        if r is None or not r.recording:
            return seq
        cap = out.get("capture") or {}
        rp = out.get("replay")
        taps.chunk(seq=seq, cadence=cadence, decoded_pos=out["tcp_pos"],
                   pos=out["wire_pos"], orn=out["wire_orn"],
                   wrench=({"left": out["pred_wrench"]} if force_roles else {}),
                   grip=({"left": out["grip_m"]} if "left" in _rollout_grip_roles else {}),
                   seed=cap.get("seed"), sample_ms=out.get("sample_ms"),
                   wire_src=str(args.anchor_mode), n_exec=int(n_exec),
                   source="replay" if rp is not None else "model", replay=rp)
        return seq

    def _gate_wire_force(wrench_row: np.ndarray) -> np.ndarray | None:
        """Apply the LIVE master switch + gate to one predicted wrench row → the (3,) wire
        F^cmd (sensor frame), or None (pose-only, pure position). Read force_ctl atomically.

        THE SIGN IS FLIPPED HERE, and this is the only place it happens on the publish path.
        The model predicts ``F_meas`` — the wrench the Bota READ while a human did the task —
        but the low level consumes a COMMAND and closes ``e = F^cmd + F_meas → 0``
        (holosoma ``force_deploy_math.servo_step``), so the command that reproduces the
        demonstrated contact is ``-F_meas``. Publishing the prediction verbatim would ask the
        arm to pull AWAY from the surface the demo pressed on, and leave the servo error at
        ``2·F_meas`` — it winds the integrator up instead of settling. See
        ``flore.utils.force_frame`` for the derivation and the three witnesses.

        A scalar commutes with every rotation, so flipping here does NOT trespass on the
        low level's job of rotating sensor→pelvis (which needs joint readback this process does
        not have). The wire still carries an UNROTATED sensor-frame triple.

        The gate is evaluated on the MAGNITUDE, which the flip leaves alone — so the switch and
        the threshold mean exactly what they did before, and the low level's own mirror deadband
        (``FORCE_CMD_WIRE_GATE_N``) still agrees with this one.
        """
        with force_lock:
            enabled = force_ctl["enabled"]
            gate_n = force_ctl["gate_n"]
        if not enabled:
            return None
        force3 = np.asarray(wrench_row[0:3], dtype=np.float64)
        if float(np.linalg.norm(force3)) <= gate_n:  # <= gate ⇒ hard zero ⇒ position
            return None
        return cmd_from_meas_force(force3)  # F^cmd = -F_meas; returns a fresh array

    def _wire_from_step(tcp_pos: np.ndarray, model_rot6d: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """ONE absolute LEFT step (model frame) → (wire_pos9, wire_orn12, urdf_rot6d).

        The per-step half of :func:`_payload_from_chunk`, factored out so the ensembled
        cadence builds its wire through the IDENTICAL frame conversion + lever as the chunk
        cadence. Keeping one implementation is the point: a second inline copy is exactly how
        the two paths would drift a frame seam apart.

        REACH COMPENSATION is applied HERE (and in :func:`_drip`'s per-step publish), i.e. at the
        last moment before a target becomes a wire — never upstream. Position only: the reported
        failure is positional, and a wrist insisting on an un-executed rotation is a far worse
        thing to get wrong on a physical arm. Off by default ⇒ this is the old expression."""
        urdf_rot6d = _model_rot6d_to_urdf(model_rot6d)
        wp, wo = mapper.full_wire(_reach_apply(tcp_pos), urdf_rot6d)
        return wp, wo, urdf_rot6d

    # Set by _run_viser to its _resync_anchor_from_robot once the measured-EE FK exists — the
    # SAME hook pattern as ``wire_sync``, and for the same reason: the clutch re-sync needs the
    # URDF/FK objects that only _run_viser builds, but the ensembled executor lives here beside
    # the other wire writers. None ⇒ no live source (offline / RECORD off) ⇒ open-loop decode
    # from the last committed anchor, exactly as _predict already tolerates.
    anchor_sync = {"fn": None}
    # MEASURED LEFT TCP provider (model frame), same hook pattern and same reason: the FK lives in
    # _run_viser. Returns None when there is no fresh RecordTap frame, which the compensator reads
    # as "the shortfall is unknowable this step" (hold, do not integrate).
    measured_tcp = {"fn": None, "force_fn": None}

    # --- REACH COMPENSATION -------------------------------------------------------------------
    # Default OFF. The state lives here, beside the other wire writers, because this is the only
    # thing that may alter a published target — and it alters ONLY that: not the observation, not
    # the decode anchor, not the chunk. See flore/s1_policy/serving/hl_reach.py for the measurement
    # that motivates it (the low level follows wire VELOCITY and ignores a standing offset, so the
    # re-anchor forgives the residual and a fine adjustment never lands).
    reach_lock = threading.Lock()
    reach_comp = ReachCompensator(ReachConfig(
        ki=max(0.0, float(args.reach_ki)), clamp_m=float(args.reach_clamp),
        stall_frac=float(args.reach_stall_frac), window=int(args.reach_window),
        gate_m=float(args.reach_gate), decay=float(args.reach_decay),
        contact_n=float(args.reach_contact_n)))
    reach_ctl = {"enabled": reach_comp.enabled}

    # CHUNK ANCHOR — what the next chunk DECODES against (orthogonal to reach compensation,
    # which only adds to an already-decoded target). Default "measured" makes every call below a
    # no-op returning its input, so the un-flagged path is unchanged. See hl_anchor.py for the
    # measurement: the re-anchor wipes 84% of the accumulated shortfall and puts a 16 mm backward
    # step in the published target at every chunk boundary.
    anchor_lock = threading.Lock()
    chunk_anchor = ChunkAnchor(AnchorConfig(
        mode=str(args.anchor_mode), eps_m=float(args.anchor_eps),
        eps_rad=float(np.radians(args.anchor_eps_deg)),
        contact_n=float(args.anchor_contact_n), decay=float(args.anchor_decay)))

    if not chunk_anchor.disabled():
        # Loud, once, because this changes what the arm is asked for on EVERY chunk and it is not
        # the default. Names the two guards so the operator can see them in the log they will read
        # after a rig session.
        _log(f"⚠️  chunk anchor = COMMAND (PLAI): the next chunk decodes from the last EXECUTED "
             f"commanded step, leaked onto a {100 * chunk_anchor.cfg.eps_m:.1f} cm / "
             f"{np.degrees(chunk_anchor.cfg.eps_rad):.0f} deg ball around the measured pose, and "
             f"decayed (x{chunk_anchor.cfg.decay:g}) back toward the arm at "
             f">={chunk_anchor.cfg.contact_n:g} N of contact. Un-clamped this diverges ~22 cm on "
             f"the recorded takes — the leak IS the safety, not a tuning knob.")
        if chunk_anchor.cfg.contact_n <= 0.0:
            _log("⚠️  chunk anchor: CONTACT CUT-OUT DISABLED (--anchor-contact-n 0). A command "
                 "anchor that keeps advancing into a blocked hand is a press, not a reach.")

    def _anchor_reset(reason: str = "") -> None:
        """Drop the carried command anchor. Called wherever ``_reach_reset`` is: an abandoned
        intent must not become the base the next chunk decodes from."""
        if chunk_anchor.disabled():
            return
        with anchor_lock:
            carrying = chunk_anchor.stats()["carrying"]
            chunk_anchor.reset()
        if carrying and reason:
            _log(f"chunk anchor: carried command anchor dropped — {reason}.")

    def _anchor_note(tcp_pos: np.ndarray, tcp_rot6d_model: np.ndarray) -> None:
        """Record the LAST EXECUTED commanded step (model frame) as the next chunk's base."""
        if chunk_anchor.disabled():
            return
        with anchor_lock:
            chunk_anchor.note_commanded(tcp_pos, tcp_rot6d_model)

    def _anchor_resolve(tcp_pos: np.ndarray, tcp_rot6d_model: np.ndarray,
                        ) -> tuple[np.ndarray, np.ndarray]:
        """The pose the LEFT decode anchor should be committed to, given a fresh measurement.

        Identity on its input in ``measured`` mode. The contact reading comes from the same
        source ``_reach_apply`` uses, so both guards see one number.
        """
        if chunk_anchor.disabled():
            return tcp_pos, tcp_rot6d_model
        ffn = measured_tcp["force_fn"]
        f_n = ffn() if ffn is not None else None
        with anchor_lock:
            return chunk_anchor.resolve(tcp_pos, tcp_rot6d_model, f_n)

    def _reach_reset(reason: str = "") -> None:
        """Drop the accumulated residual — an abandoned intent must not be re-applied to the next
        motion. Called on Stop, Reset-start-pose, the low-level-lost watchdog and teardown."""
        with reach_lock:
            had = reach_comp.residual_norm
            reach_comp.reset()
        if had > 1e-4 and reason:
            _log(f"reach compensation: residual ({had * 100:.1f} cm) dropped — {reason}.")
        # The carried command anchor is abandoned intent for exactly the same reasons and at
        # exactly the same moments (Stop, Reset-start-pose, new rollout, low-level-lost), so it
        # is dropped from here rather than duplicating five call sites. The one over-reach is the
        # reach-compensation checkbox, which also lands here; that costs a single bootstrap chunk
        # decoded from the measurement, i.e. the default behaviour, and no hazard.
        _anchor_reset(reason)

    def _reach_active() -> bool:
        with reach_lock:
            return bool(reach_comp.enabled and reach_ctl["enabled"])

    def _reach_apply(tcp_pos: np.ndarray) -> np.ndarray:
        """Compensate ONE absolute LEFT TCP target (model frame) just before it is turned into a
        wire. A no-op returning its input when the feature is off or the operator un-ticked it."""
        if not _reach_active():
            return np.asarray(tcp_pos, dtype=np.float64)
        fn = measured_tcp["fn"]
        meas = fn() if fn is not None else None
        ffn = measured_tcp["force_fn"]
        f_n = ffn() if ffn is not None else None
        with reach_lock:
            return reach_comp.compensate(np.asarray(tcp_pos, dtype=np.float64), meas, f_n)

    def _wire_for_step(out: dict, i: int) -> tuple[np.ndarray, np.ndarray]:
        """The wire row to publish for step ``i`` of a chunk payload.

        With compensation OFF this returns the row ``_payload_from_chunk`` already computed —
        the same object as before, so the chunk cadence is byte-identical. With it ON the row is
        REBUILT from that step's TCP target against a FRESH measured pose, because the shortfall
        (and therefore the correction) changes every step; reusing the row computed when the
        chunk was decoded would apply a stale correction for up to 1.67 s.

        The payload's own ``wire_pos``/``wire_orn`` stay uncompensated on purpose: they feed the
        yellow preview path and the take's recorded plan, which must show what the MODEL asked
        for, not what the servo had to be told to get there."""
        if not _reach_active():
            return out["wire_pos"][i], out["wire_orn"][i]
        return mapper.full_wire(_reach_apply(out["tcp_pos"][i]), out["tcp_rot6d"][i])

    def _drip_ensembled(n_steps: int) -> int:
        """ACT temporal ensembling: RE-QUERY the policy every step and execute the blend.

        This is upstream ACT's ``--temporal_agg`` cadence. ``imitate_episodes.py:180-183``
        sets ``query_frequency = 1`` when ``temporal_agg`` is on, so every wall-step gets a
        fresh chunk from the CURRENT observation and the value actually executed is the
        ``exp(-k*i)``-weighted mean of every still-live prediction for that instant
        (``imitate_episodes.py:248-259``). A chunk boundary becomes a smooth handover instead
        of a jump, and one badly-sampled chunk is outvoted by its neighbours.

        WHY THIS IS A SEPARATE LOOP FROM :func:`_drip` (and not a flag inside it): ``_drip``
        replays an ALREADY-predicted array, so its per-step work is a dict lookup. Here each
        step runs a full inference (measured **6.14 ms** against the 33.3 ms budget at 30 Hz —
        5.4x headroom, so the arm still meets its deadline), the anchor must be re-synced to
        the robot BEFORE each query (the observation and the decode frame have to describe the
        same instant), and there is no staged chunk to preview. Forcing one function to do
        both would mean a branch in every line of the hot loop.

        FRAME CONTRACT — the reason this works at all: the blend happens in ABSOLUTE space,
        after ``decode_chunk`` has un-relativized each chunk against the anchor that was live
        when it was predicted. FLoRe's chunks are ANCHOR-RELATIVE, so blending raw values would
        mix frames (up to 82 cm at the H=50 window's oldest survivor at 0.5 m/s). That is also
        why the per-step anchor re-sync below does NOT reset the ensembler: an older prediction
        is still expressed in the same world frame and stays valid, and blending predictions
        made from earlier observations is precisely what temporal ensembling IS.

        Returns how many steps were actually executed (fewer if ``stop`` fired mid-loop)."""
        step_dt = 1.0 / max(args.step_hz, 1.0)
        drip_dt = 1.0 / max(args.drip_hz, 1.0)
        # ABANDONED-INTENT RESET, consumed HERE because this function owns the ensembler under
        # `busy`. A Stop press or the low-level-lost watchdog can only REQUEST the drop (they run
        # on other threads, possibly while a blend is mid-flight). Applied BEFORE the first add,
        # so the stretch starting now can never average in a survivor of the stretch the operator
        # abandoned — which, with exp(-k*i) oldest-first weights, would otherwise OUTVOTE it.
        if take_pending_ens_reset(ens_ctl, ens_lock):
            planner.reset_temporal_ensemble()
        n_done = 0
        slow_logged = False
        dripping.set()
        try:
            for _i in range(int(n_steps)):
                if stop.is_set():
                    break
                t_step0 = time.time()
                # Re-sync the decode anchor to the robot's MEASURED pose before querying, so
                # the chunk is decoded from where the arm ACTUALLY is at THIS instant — the
                # same clutch semantic Predict applies, now per step. Best-effort: offline /
                # RECORD-off keeps the last committed anchor (open loop), exactly as _predict
                # already tolerates.
                _sync = anchor_sync["fn"]
                if _sync is not None:
                    _sync(quiet=True)
                # Each ensembled step IS its own inference, so each gets its own recorded
                # prediction: the observation genuinely differs per step (that is the whole
                # premise of re-querying), and collapsing them into one entry would misreport
                # what the policy saw. The stored chunk is the raw per-step chunk; the EXECUTED
                # value is the blend, which the commanded stream records.
                ens_cap: dict = {}
                # Same GIL priority as _infer: this cadence re-queries EVERY step, so an
                # un-yielded viz thread would tax all 16/30/50 of them.
                with infer_gate.priority():
                    blend = planner.infer_step_ensembled(
                        seed=_next_seed(), force_enabled=True, gate_n=0.0, role="left",
                        capture=ens_cap)
                with ens_lock:
                    ens_ctl["n_blended"] = int(blend["n_blended"])
                pos_i, orn_i, _urdf = _wire_from_step(blend["position"], blend["rot6d"])
                wrench_i = (blend["wrench"] if blend["wrench"] is not None
                            else np.zeros(6, dtype=np.float64))
                f_i = _gate_wire_force(wrench_i)  # LIVE gate — switch OFF/low ⇒ pure position
                _set_last_wire(pos_i, orn_i, f_i)
                rec_i = _record_prediction(
                    {"capture": ens_cap, "t_predict": time.monotonic()}, 1, time.monotonic())
                # The run record: each ensembled step IS its own one-row chunk (the blend), so
                # the plots draw the executed value with its survivor count.
                _rseq = _rollout_chunk(
                    {"capture": ens_cap, "tcp_pos": np.asarray(blend["position"], dtype=np.float64)[None],
                     "wire_pos": pos_i[None], "wire_orn": orn_i[None], "pred_wrench": wrench_i[None],
                     "grip_m": np.asarray([blend["gripper_m"] if blend["gripper_m"] is not None else np.nan])},
                    "ensembled", 1)
                taps.step(_rseq, 0, (pos_i, orn_i), ({"left": wrench_i} if force_roles else None),
                          n_blended=int(blend["n_blended"]))
                rec_step["chunk"] = rec_i
                rec_step["i"] = 0
                grip_cmd_i = None
                if (grip_enabled.is_set() and s0.num_clients > 0
                        and blend["gripper_m"] is not None and np.isfinite(blend["gripper_m"])):
                    frac, _sq = _grip_target_frac(float(blend["gripper_m"]))
                    if np.isfinite(frac):
                        subs["gripper"].send({"target_frac": float(frac)})
                        jaw_ever_commanded["v"] = True
                        grip_cmd_i = calib.width_closed_m + float(frac) * (
                            calib.width_open_m - calib.width_closed_m)
                _record_commanded(rec_i, 0, pos_i, orn_i, grip_cmd_i, f_i)
                # Commit the anchor to the step we just COMMANDED, so the next query's
                # relative decode branches from here (per-step equivalent of _drip's
                # end-of-chunk commit). blend["rot6d"] is the MODEL frame, which is what the
                # decode anchor must be in.
                _commit_left_step(planner, blend["position"], blend["rot6d"])
                # The blend IS this cadence's published command, so it is what --anchor-mode
                # command carries into the next step's re-sync. Note ACT+ensembling already
                # smooths the chunk boundary (that is what the blend is for), so here the mode
                # buys only the un-wiped shortfall, not the jump.
                _anchor_note(blend["position"], blend["rot6d"])
                n_done += 1
                # Hold this step for its full grid period, refreshing at drip_hz for the low
                # level's freshness gate — identical to _drip's inner hold.
                while True:
                    _s0_send(pos_i, orn_i, _gate_wire_force(wrench_i))
                    if stop.is_set():
                        break
                    elapsed = time.time() - t_step0
                    remaining = step_dt - elapsed
                    if remaining <= 0:
                        # The inference did not fit inside the action-grid period, so the arm
                        # is being commanded slower than the 30 Hz the chunk was trained on.
                        # Say so ONCE rather than let it read as a sluggish policy.
                        if elapsed > step_dt * 1.05 and not slow_logged:
                            _log(f"ENSEMBLE: step took {elapsed * 1e3:.1f} ms > the "
                                 f"{step_dt * 1e3:.1f} ms action-grid period — the re-query is "
                                 f"not keeping up, so playback is slower than trained. Turn "
                                 f"temporal ensembling OFF for full-rate chunk playback.")
                            slow_logged = True
                        break
                    time.sleep(min(drip_dt, remaining))
        finally:
            dripping.clear()
            rec_step["chunk"] = rec_step["i"] = -1
        return n_done

    def _drip(out: dict) -> int:
        """Drip the first ``exec_ctl['n']`` steps of an already-predicted chunk to the low
        level at step_hz (re-sent at drip_hz for the freshness gate), applying the gate LIVE
        per step, then commit the anchor to the LAST EXECUTED step (not the chunk end).

        Returns how many steps were ACTUALLY executed (fewer than requested if ``stop`` fired
        mid-chunk). A recorded replay advances its episode cursor by exactly this count, so an
        interrupted chunk cannot skip demonstration frames the arm never actually tracked."""
        # The per-step wire rows are read through _wire_for_step (which honours reach
        # compensation); only the step COUNT is needed from the pre-computed array here.
        wire_pos = out["wire_pos"]
        pred_wrench = out["pred_wrench"]
        grip_m = out["grip_m"]
        step_pos = out["step_pos"]
        step_rot6d = out["step_rot6d"]
        S = wire_pos.shape[0]
        n_exec = max(1, min(int(exec_ctl["n"]), S))  # execute-N-of-S selector (live)
        step_dt = 1.0 / max(args.step_hz, 1.0)
        drip_dt = 1.0 / max(args.drip_hz, 1.0)
        last_i = 0
        # Register the plan BEFORE the first step is published, so the take's prediction entry
        # exists by the time the measured/commanded rows start referring to its index.
        rec_i = _record_prediction(out, n_exec, time.monotonic())
        rseq = _rollout_chunk(out, "sequential", n_exec)
        rec_step["i"] = 0 if rec_i >= 0 else -1
        rec_step["chunk"] = rec_i
        dripping.set()
        try:
            for i in range(n_exec):
                if stop.is_set():
                    break
                # Compensation (default OFF) rebuilds this row against a FRESH measured pose;
                # OFF ⇒ the pre-computed row, unchanged.
                pos_i, orn_i = _wire_for_step(out, i)
                f_i = _gate_wire_force(pred_wrench[i])  # LIVE gate — switch OFF/low ⇒ None
                # The run record's executed step: the MODEL's row (pre-compensation) + the wrench
                # the gate saw; the compensated row it publishes is recorded by _s0_send.
                taps.step(rseq, i, (wire_pos[i], out["wire_orn"][i]),
                          ({"left": pred_wrench[i]} if force_roles else None))
                _set_last_wire(pos_i, orn_i, f_i)
                last_i = i
                rec_step["i"] = i
                grip_cmd_i = None
                if (grip_enabled.is_set() and s0.num_clients > 0
                        and grip_m.shape[0] > i and np.isfinite(grip_m[i])):
                    frac, _sq = _grip_target_frac(float(grip_m[i]))
                    if np.isfinite(frac):
                        subs["gripper"].send({"target_frac": float(frac)})
                        jaw_ever_commanded["v"] = True
                        grip_cmd_i = calib.width_closed_m + float(frac) * (
                            calib.width_open_m - calib.width_closed_m)
                _record_commanded(rec_i, i, pos_i, orn_i, grip_cmd_i, f_i)
                t_step0 = time.time()
                while True:
                    _s0_send(pos_i, orn_i, _gate_wire_force(pred_wrench[i]))
                    if stop.is_set():
                        break
                    elapsed = time.time() - t_step0
                    remaining = step_dt - elapsed
                    if remaining <= 0:
                        break
                    time.sleep(min(drip_dt, remaining))
        finally:
            dripping.clear()
            rec_step["chunk"] = rec_step["i"] = -1  # measured rows after this belong to no step
        # Commit the anchor to the LAST EXECUTED step so the next chunk branches from where
        # the arm actually STOPPED (step last_i), not the un-executed chunk end. Only LEFT
        # moves; RIGHT/HEAD stay at their last-synced measured hold pose (set on Predict/idle).
        _commit_left_step(planner, step_pos[last_i], step_rot6d[last_i])
        # …and remember it as the COMMANDED pose, which is what --anchor-mode command hands the
        # next chunk in place of the measurement the next Predict is about to take. Same step for
        # both, so the two modes differ only in which one survives the re-anchor.
        _anchor_note(step_pos[last_i], step_rot6d[last_i])
        n_done = last_i + 1
        # RECORDED REPLAY: walk the episode cursor forward by the time those steps consumed
        # (see RecordedEpisodeSource.advance_frame — TIME-based, because the 16 Hz action grid
        # and the ~15 Hz camera grid are different clocks). Only when the executed chunk really
        # came from replay AND the cursor has not been moved underneath us (a live episode
        # switch / seek while dripping), so a stale advance can never yank another episode's
        # cursor forward.
        rp = out.get("replay")
        if rp is not None and replay is not None:
            nxt = replay.advance_frame(rp["episode"], rp["frame"], n_done)
            # Land on a PLAYABLE frame so the manual Predict→Run→Predict path can't stop on a
            # masked/glitch frame and greet the next Predict with a raw error (the sequential
            # loop skips these itself, but a manual operator would just see the exception).
            usable = replay.next_usable_frame(rp["episode"], nxt)
            if usable is not None:
                nxt = usable
            with replay_lock:
                if (replay_ctl["episode"] == rp["episode"]
                        and replay_ctl["frame"] == rp["frame"]):
                    replay_ctl["frame"] = nxt
        return n_done

    def _drip_openloop(out: dict) -> int:
        """Drip the ENTIRE pre-computed episode chunk straight through — no re-anchor.

        The open-loop executor for :func:`_infer_replay_openloop`. Unlike :func:`_drip` it
        commands ALL ``S`` steps of the plan (there is no execute-N-of-S selector: the whole
        episode is one chunk), and it does NO per-chunk cursor advance or anchor re-sync — the
        demonstration was placed against the arm ONCE when the plan was decoded, and the rest is
        commanded without correction. That is the entire point: the executed-vs-recorded error
        this produces is a true open-loop tracking number, not one confounded by a reference that
        is re-drawn from the arm's own lagging pose every chunk (see
        :class:`flore.s1_policy.serving.hl_replay.EpisodePlan`).

        Still fully interruptible: ``stop`` (Stop button / teardown) breaks the loop between
        steps and mid-resend, exactly as in :func:`_drip`, so an open-loop play is no less
        abortable than a per-chunk one. Commits the anchor to the LAST executed step at the end
        so a subsequent manual Predict/Reset branches from where the arm actually stopped.

        Returns how many steps were actually executed (``S`` unless ``stop`` fired)."""
        wire_pos = out["wire_pos"]
        wire_orn = out["wire_orn"]
        pred_wrench = out["pred_wrench"]
        grip_m = out["grip_m"]
        step_pos = out["step_pos"]
        step_rot6d = out["step_rot6d"]
        S = wire_pos.shape[0]
        step_dt = 1.0 / max(args.step_hz, 1.0)
        drip_dt = 1.0 / max(args.drip_hz, 1.0)
        last_i = 0
        rec_i = _record_prediction(out, S, time.monotonic())
        rseq = _rollout_chunk(out, "openloop", S)
        rec_step["chunk"] = rec_i
        rec_step["i"] = 0 if rec_i >= 0 else -1
        dripping.set()
        try:
            for i in range(S):
                if stop.is_set():
                    break
                pos_i = wire_pos[i]
                orn_i = wire_orn[i]
                f_i = _gate_wire_force(pred_wrench[i])  # LIVE gate — switch OFF/low ⇒ None
                taps.step(rseq, i, (pos_i, orn_i), ({"left": pred_wrench[i]} if force_roles else None))
                _set_last_wire(pos_i, orn_i, f_i)
                last_i = i
                rec_step["i"] = i
                grip_cmd_i = None
                if (grip_enabled.is_set() and s0.num_clients > 0
                        and grip_m.shape[0] > i and np.isfinite(grip_m[i])):
                    frac, _sq = _grip_target_frac(float(grip_m[i]))
                    if np.isfinite(frac):
                        subs["gripper"].send({"target_frac": float(frac)})
                        jaw_ever_commanded["v"] = True
                        grip_cmd_i = calib.width_closed_m + float(frac) * (
                            calib.width_open_m - calib.width_closed_m)
                _record_commanded(rec_i, i, pos_i, orn_i, grip_cmd_i, f_i)
                t_step0 = time.time()
                while True:
                    _s0_send(pos_i, orn_i, _gate_wire_force(pred_wrench[i]))
                    if stop.is_set():
                        break
                    elapsed = time.time() - t_step0
                    remaining = step_dt - elapsed
                    if remaining <= 0:
                        break
                    time.sleep(min(drip_dt, remaining))
        finally:
            dripping.clear()
            rec_step["chunk"] = rec_step["i"] = -1
        # Anchor lands on the last COMMANDED step (open loop: the plan's own step, not a
        # measured re-sync) so a later manual action starts from where the demo left the arm.
        _commit_left_step(planner, step_pos[last_i], step_rot6d[last_i])
        _anchor_note(step_pos[last_i], step_rot6d[last_i])
        return last_i + 1

    # --- viser page ---
    _run_viser(
        args, loaded, planner, mapper, subs, s0, calib,
        stop, run_continuous, dripping, grip_enabled, jaw_ever_commanded,
        force_ctl, force_lock, exec_ctl, busy, staged, force_roles, require_6dof,
        _infer_active, _drip, _set_last_wire, _last_wire_snapshot, wire_sync,
        replay, replay_ctl, replay_lock, auto_running, replay_only,
        ens_ctl, ens_lock, _drip_ensembled, anchor_sync,
        auto_model_ctl, auto_model_lock, grip_ctl, grip_lock, _grip_release_latch,
        _infer_replay_openloop, _drip_openloop,
        recorder, record_enabled, _record_arm, _record_finish, rec_step,
        reach_comp, reach_ctl, reach_lock, _reach_reset, _reach_active, measured_tcp,
        infer_gate, ft_trace,
        chunk_anchor, _anchor_resolve, _anchor_reset,
        taps=taps,
    )

    # --- teardown ---
    stop.set()
    run_continuous.clear()
    if not chunk_anchor.disabled():
        # The tuning diagnostic, printed once where the operator will actually read it. A high
        # clamp rate means the ask is being CAPPED — the low level genuinely cannot reach, and the
        # answer is a shorter chunk (the shortfall is linear in path since the anchor), NOT a
        # bigger --anchor-eps.
        st = chunk_anchor.stats()
        n = max(1, int(st["n_resolved"]))
        _log(f"chunk anchor summary: mode={st['mode']} resolved={st['n_resolved']} "
             f"bootstrap={st['n_bootstrap']} "
             f"pos-clamped={st['n_clamped_pos']} ({100 * st['n_clamped_pos'] / n:.0f}%) "
             f"rot-clamped={st['n_clamped_rot']} ({100 * st['n_clamped_rot'] / n:.0f}%) "
             f"contact-pullback={st['n_contact_pullback']} "
             f"({100 * st['n_contact_pullback'] / n:.0f}%) "
             f"max_carried_lead={100 * st['max_lead_m']:.2f} cm (eps {100 * st['eps_m']:.1f} cm)")
        if st["n_clamped_pos"] > 0.5 * n:
            _log("chunk anchor: the leak bound was hit on >50% of chunks — the arm is being asked "
                 "for the cap, not for what it needs. Shorten the chunk (--exec-steps) rather "
                 "than raising --anchor-eps; the shortfall is LINEAR in path since the anchor.")
    # Flush a take that was still open when the page closed (Ctrl-C mid-run) BEFORE the sockets
    # go down, and BLOCK on the write: a daemon flush thread would be killed with the
    # interpreter and lose a just-finished execution.
    try:
        recorder.close_pending()
    except Exception as e:  # noqa: BLE001 — teardown must not mask the real exit path
        _log(f"shutdown recorder-flush error: {e!r}")
    # A ⏺ run recording still open (Ctrl-C mid-run): close it so the folder gets its meta.json.
    try:
        _rr = taps.recorder
        if _rr is not None and _rr.state == "recording":
            _log("RECORD: " + _rr.run_end("page shutdown"))
    except Exception as e:  # noqa: BLE001 — teardown must not mask the real exit path
        _log(f"shutdown run-recorder error: {e!r}")
    if jaw_ever_commanded["v"] and args.command_gripper:
        try:
            for _ in range(200):
                if not dripping.is_set():
                    break
                time.sleep(0.01)
            g = subs["gripper"]
            with g._lock:
                g._send_q = [json.dumps({"target_frac": 1.0})]
            for _ in range(200):
                with g._lock:
                    pending = bool(g._send_q)
                if not pending and g.connected:
                    break
                time.sleep(0.01)
            _log("shutdown: released LEFT gripper (open)")
        except Exception as e:  # noqa: BLE001
            _log(f"shutdown gripper-release error: {e!r}")
    for s in subs.values():
        s.stop()
    s0.stop()
    if replay is not None:
        try:
            replay.close()  # release the replay zarr's ZipStore handle
        except Exception as e:  # noqa: BLE001 — teardown must not mask the real exit path
            _log(f"shutdown replay-close error: {e!r}")
    if muse_backend is not None:
        # Reap the uvaforce child. It is NOT a daemon thread we can abandon: it holds a CUDA
        # context and several GB of VRAM, so a leaked one makes the next launch OOM.
        try:
            muse_backend.close()
            _log("shutdown: muse backend closed")
        except Exception as e:  # noqa: BLE001 — teardown must not mask the real exit path
            _log(f"shutdown muse-backend-close error: {e!r}")
    return 0


def _run_viser(
    args, loaded, planner, mapper, subs, s0, calib,
    stop, run_continuous, dripping, grip_enabled, jaw_ever_commanded,
    force_ctl, force_lock, exec_ctl, busy, staged, force_roles, require_6dof,
    _infer, _drip, _set_last_wire, _last_wire_snapshot, wire_sync,
    replay=None, replay_ctl=None, replay_lock=None, auto_running=None, replay_only=False,
    ens_ctl=None, ens_lock=None, _drip_ensembled=None, anchor_sync=None,
    auto_model_ctl=None, auto_model_lock=None, grip_ctl=None, grip_lock=None,
    _grip_release_latch=None,
    _infer_replay_openloop=None, _drip_openloop=None,
    recorder=None, record_enabled=None, _record_arm=None, _record_finish=None, rec_step=None,
    reach_comp=None, reach_ctl=None, reach_lock=None, _reach_reset=None, _reach_active=None,
    measured_tcp=None, infer_gate=None, ft_trace=None,
    chunk_anchor=None, _anchor_resolve=None, _anchor_reset=None,
    taps=None,
):
    """Build the viser page + control callbacks + render loop. Split out so main() reads
    top-to-bottom; all closures capture main()'s control state.

    ``_infer`` is the ACTIVE chunk source (``_infer_active``), so this function is agnostic
    to whether a chunk came from the policy or from a recorded episode. ``replay`` is the
    optional :class:`RecordedEpisodeSource` (None when --replay-zarr was not passed, in which
    case the Replay panel is not built and the page is byte-identical to the model-only UI).

    ``replay_only`` means NO policy was loaded (``--replay-zarr`` with no ``--checkpoint``):
    the page then hides the model-facing panels (the checkpoint readout, the conditioning-image
    strip, the PREDICTED force charts) and the Source dropdown offers only recorded replay,
    since none of those have a meaning without a model. Everything that touches the ROBOT —
    the wire, the force gate, the drip, the clutch re-anchor, the triads — is identical."""
    import viser
    from viser import transforms as vtf
    from viser import uplot
    from viser.extras import ViserUrdf

    import replay_common as rc
    from g1_right_hand_ik import YamConfigMapper, yam_actuated_joint_names
    from g1_three_point_ik import ThreePointIK

    #: Does the CHECKPOINT read a wrench? False for a pure-position model (``FLORE_DROP_FORCE``),
    #: which is what takes the F/T out of the Predict freshness gate (see :func:`_obs_freshness`).
    #: Same definition main() uses (``bool(loaded.layout.force_roles)``), read off what was passed.
    _consumes_force = bool(force_roles)
    #: Roles the F/T PANEL displays, which is NOT the same question as what the model reads. A
    #: pure-position checkpoint leaves ``force_roles`` empty, so reusing it here made
    #: ``_pick_force_channels`` return None and the panel read "no/invalid F/T frame (bridge
    #: down?)" on a perfectly healthy bridge. That reading still matters to the operator — the
    #: low level's 40 N hard stop is driven by this very sensor — so the panel falls back to the
    #: LEFT hand this planner is canonical for. Display only; nothing here reaches the model.
    _display_force_roles = force_roles or ("left",)

    # LEFT-only IK + yam mapper.
    ik = ThreePointIK(config_name=args.ik_config, reach_scale=args.reach_scale, active_roles=("left",))
    yam_mapper = YamConfigMapper(ik._humi_joint_names, yam_actuated_joint_names(args.yam_urdf))

    server = viser.ViserServer(host=args.host, port=args.port, label="FLoRe HL-v2 PLANNER")
    server.scene.set_up_direction("+z")
    server.scene.add_grid("/grid", plane="xy", shadow_opacity=0.0, infinite_grid=True, cell_size=0.25)
    base_xyz = np.asarray(ik.home_q()[0:3], dtype=float).copy()
    server.scene.add_frame("/robot", show_axes=False, position=tuple(base_xyz))
    urdf = ViserUrdf(server, urdf_or_path=Path(args.yam_urdf), root_node_name="/robot")
    # Name-based remap from the low-level's 29-DOF SDK dof_pos (RecordTap :8779) into this
    # URDF's actuated-joint config vector, so the mesh can render the REAL robot joints when
    # RECORD=1 is on. Matching is by joint NAME (immune to the pitch/roll/yaw ordering trap);
    # a missing/unmatched joint is surfaced loudly rather than silently mis-posing a limb.
    dof_remap = build_sdk_to_urdf_index(list(urdf.get_actuated_joint_names()))
    # INITIAL MESH POSE = the low level's REAL READY POSE (home_anchor.default_qpos), not the mink
    # stand keyframe (`ik.home_q()`), and built with the SAME remap_dof_to_urdf the live RecordTap
    # path uses — so the mesh and the triads/anchor agree BY CONSTRUCTION rather than by two
    # independently-maintained poses.
    #
    # WHY (the bug this fixes, reported from a screenshot): the mesh used to be posed at the mink
    # STAND keyframe while /follow + the three triads + the decode anchor are all at the READY pose.
    # Those are different arm configurations — measured **166.3 mm and 47.08 deg apart** at
    # left_wrist_yaw_link — so the drawn cluster and the green trajectory hung in space beside the
    # rendered arm, which reads exactly like a frame bug and is not one. The anchor was never wrong:
    # it matches ready-pose FK to 0.0004 mm / 0.0001 deg. Only the MESH was in the wrong pose.
    # Leg joints are identical in both poses (both feet land at the same z), so this does not float
    # or sink the render. Falls back to the mink pose only if default_qpos is unusable, loudly.
    # ONE seed vector for BOTH the mesh and the /follow triad cluster (see the FK below), which is
    # what makes "the mesh agrees with the triads" structural rather than a coincidence of two
    # separately-maintained poses.
    _seed_dof = _display_seed_dof(bool(args.zero_arms)) if not dof_remap.missing_sdf else None
    _init_cfg = None
    if _seed_dof is not None:
        try:
            # BOTH jaws open. The right hand has no gripper stream in this planner (LEFT-only,
            # :8784), so its finger joint is whatever the seed puts there for the whole session —
            # it must be the OPEN end, exactly like the left one's pre-reading default.
            _init_cfg = remap_dof_to_urdf(_seed_dof, dof_remap,
                                          left_jaw_m=JAW_JOINT_OPEN_M,
                                          right_jaw_m=JAW_JOINT_OPEN_M)
        except Exception as e:  # noqa: BLE001 — fall back to the mink pose, loudly
            _log(f"initial mesh pose: ready-pose remap failed ({e!r}); using the mink stand pose.")
    if _init_cfg is None:
        _log("initial mesh pose: falling back to the mink stand pose — the mesh will sit ~166 mm / "
             "47 deg off the triads until a RECORD=1 frame arrives.")
    # ``to_yam``'s own jaw defaults are 0.0 (the CLOSED end of the travel) and it lives in the
    # UMI_collection submodule, so both hands are passed explicitly here rather than relying on
    # a default this repo does not own.
    urdf.update_cfg(_init_cfg if _init_cfg is not None
                    else yam_mapper.to_yam(ik.home_q(), left_jaw_m=JAW_JOINT_OPEN_M,
                                           right_jaw_m=JAW_JOINT_OPEN_M))
    if dof_remap.missing_sdf:
        _log(f"WARNING: yam URDF lacks SDK joints {dof_remap.missing_sdf} — live mesh would "
             "mis-pose; falling back to predicted-IK mesh only.")
    if dof_remap.unmatched_urdf:
        _log(f"note: URDF actuated joints not driven by dof_pos (held 0): {dof_remap.unmatched_urdf}")
    live_mesh_ok = not dof_remap.missing_sdf

    # GHOST ROBOT — a second, translucent-violet ViserUrdf that PLAYS the predicted chunk (its
    # joints come from the yam-URDF arm IK set up below, NOT the mink IK). Rooted at its own
    # /robot_ghost frame at the same pelvis origin as /robot, so a ghost cfg renders in the SAME
    # world as the yellow path. Its whole subtree is hidden via the frame handle until a chunk's
    # ghost cfgs are ready (viser visibility propagates to children). A failure here is viz-only:
    # the yellow path + charts still work, the plan just isn't "acted out".
    ghost_root = ghost_urdf = None
    try:
        ghost_root = server.scene.add_frame("/robot_ghost", show_axes=False,
                                            position=tuple(base_xyz), visible=False)
        ghost_urdf = ViserUrdf(server, urdf_or_path=Path(args.yam_urdf),
                               root_node_name="/robot_ghost", mesh_color_override=_GHOST_RGBA)
        ghost_urdf.update_cfg(_init_cfg if _init_cfg is not None
                              else yam_mapper.to_yam(ik.home_q(), left_jaw_m=JAW_JOINT_OPEN_M,
                                                     right_jaw_m=JAW_JOINT_OPEN_M))
    except Exception as e:  # noqa: BLE001 — ghost is a preview; never take down the scene
        _log(f"ghost robot unavailable ({e!r}); the plan will show as the yellow path only.")
        ghost_root = ghost_urdf = None

    # Standalone headless FK of the yam URDF for the MEASURED current LEFT EE. We load a
    # SECOND yourdfpy.URDF (no meshes, its own config) so FK never touches the ViserUrdf's
    # config on the render thread. This URDF has ALL 31 joints INCLUDING left_wrist_yaw —
    # the mink ThreePointIK model is MISSING wrist_yaw/waist_pitch/roll, so mink FK would
    # silently drop a real wrist-yaw rotation (the 0.18 m contact lever would not swing);
    # the full URDF FK reproduces the low-level Pinocchio bake bit-for-bit (verified: at
    # default_qpos it hits home_anchor's LEFT contact + wrist quat to machine precision).
    _fk_urdf = None
    _fk_lock = threading.Lock()  # serializes _fk_urdf.update_cfg across Predict / Reset threads
    if live_mesh_ok:
        try:
            import yourdfpy  # noqa: PLC0415

            _fk_urdf = yourdfpy.URDF.load(
                args.yam_urdf, load_meshes=False, build_scene_graph=True)
            _fk_remap = build_sdk_to_urdf_index(list(_fk_urdf.actuated_joint_names))
        except Exception as e:  # noqa: BLE001
            _log(f"measured-EE FK unavailable ({e!r}); re-anchor disabled (open-loop fallback).")
            _fk_urdf = None

    # Body frames + their pelvis-frame wrist→contact offsets, in the vr_3point wire order
    # [LEFT, RIGHT, HEAD] — the SAME bodies + levers the low level's _compute_vr_fk uses to
    # build the pose it gates against, so a wire reconstructed here matches the robot's held
    # pose to ~0 (arming passes, no jump).
    _WIRE_BODIES = ("left_wrist_yaw_link", "right_wrist_yaw_link", "torso_link")

    def _measured_full_wire(dof_pos_29: np.ndarray):
        """FK a live 29-DOF SDK dof_pos to the robot's MEASURED vr_3point wire, in the SAME
        [L, R, HEAD] contact-point convention the low level gates against. Returns
        ``(left_tcp_pos, left_rot6d, wire_pos9, wire_orn12)`` or None.

        ``wire_pos9``/``wire_orn12`` are the full 3-point wire (contact points + body quats,
        pelvis frame) so the idle heartbeat can broadcast the robot's REAL pose and the
        EXECUTED chunk can hold RIGHT+HEAD there. ``left_tcp_pos``/``left_rot6d`` is the LEFT
        UMI-TCP the model's EE-local delta decodes against (the clutch base).

        Serialized under _fk_lock. FOUR callers on THREE threads: _predict and Reset (control
        threads, both also under `busy`), the heartbeat's idle sync, and — since the triad
        cluster follows the live robot — the RENDER thread once per tick. `_fk_lock` is the
        finer-grained guard on the stateful `_fk_urdf.update_cfg` + `get_transform` pair, which
        must not interleave; `busy` does NOT cover the render thread, so this lock (not `busy`)
        is what actually makes the render-thread call safe.

        The lock cannot deadlock: it is acquired at exactly ONE place, never nested, never held
        across a call that re-enters, and the critical section is pure FK (no I/O, no other
        lock). Contention is negligible — measured ~0.41 ms per call against the render loop's
        33 ms period, so the render thread holds it ~1% of the time.

        It is a SEPARATE URDF object from the ViserUrdf mesh (its own config), so it never
        races the render thread's `urdf.update_cfg`. This function is READ-ONLY with respect to
        planner state: it mutates only `_fk_urdf`'s own config and returns values, so a render
        tick calling it can never disturb the anchor, the hold pose, or a staged chunk."""
        if _fk_urdf is None:
            return None
        dof = np.asarray(dof_pos_29, dtype=np.float64).reshape(-1)
        if dof.shape[0] != 29 or not np.all(np.isfinite(dof)):
            return None
        try:
            # Jaws left at the function's default (open). This config is FK-only — nothing here
            # is rendered — but an explicit 0.0 would now read as "closed" and invite a copy-paste
            # into a render path.
            cfg = remap_dof_to_urdf(dof, _fk_remap)
            contacts = np.zeros((3, 3), dtype=np.float64)
            quats = np.zeros((3, 4), dtype=np.float64)
            left_R = None
            left_wrist = None
            with _fk_lock:
                _fk_urdf.update_cfg(cfg)
                for i, body in enumerate(_WIRE_BODIES):
                    T = np.asarray(_fk_urdf.get_transform(frame_to=body, frame_from="pelvis"))
                    R = T[:3, :3]
                    p = T[:3, 3]
                    contacts[i] = p + R @ mapper.vr_offsets[i]  # wrist/torso → contact point
                    quats[i] = rotmat_to_quat_xyzw(R)
                    if i == 0:
                        left_R = R
                        left_wrist = p
            tcp_pos = left_wrist + left_R @ _TCP_OFFSET
            wire_pos9 = contacts.reshape(-1)
            wire_orn12 = quats.reshape(-1)
            if not (np.all(np.isfinite(tcp_pos)) and np.all(np.isfinite(left_R))
                    and np.all(np.isfinite(wire_pos9)) and np.all(np.isfinite(wire_orn12))):
                return None
            return (tcp_pos.astype(np.float64), rotmat_to_rot6d(left_R).astype(np.float64),
                    wire_pos9.astype(np.float64), wire_orn12.astype(np.float64))
        except Exception:  # noqa: BLE001 — FK is best-effort; caller falls back to home seed
            return None

    def _resync_anchor_from_robot(*, quiet: bool = False) -> bool:
        """Re-sync the planner's LEFT decode anchor AND the RIGHT+HEAD hold pose AND the idle
        wire to the robot's MEASURED pose from a fresh RecordTap frame. Returns True on success
        (clutch base = actual pose; idle wire == robot, so the pre-arm gate passes with no
        jump), False if RECORD is off / stale / FK failed (caller keeps the open-loop home).

        ``quiet`` suppresses the armed-re-sync confirmation line. That log is a once-per-Predict
        rig check, but the temporal-ensembling cadence calls this EVERY step (30 Hz), where it
        would both flood the log and put a string format + ``np.round`` on the arm's critical
        path. The re-sync itself is identical either way."""
        if _fk_urdf is None:
            return False
        rmsg, rw, _rn = subs["record"].latest()
        rframe = parse_record_frame(rmsg)
        if rframe is None or not rw or (time.time() - rw) > 1.0:
            return False
        meas = _measured_full_wire(rframe.dof_pos)
        if meas is None:
            return False
        tcp_pos, tcp_rot6d, wire_pos9, wire_orn12 = meas  # tcp_rot6d is the MEASURED URDF wrist
        # The decode anchor + wrt_start live in the MODEL/puck frame (training convention); the
        # wire publish below stays in URDF. Convert ONLY at the commit boundary — do NOT reuse the
        # converted value for full_wire (that is the "same variable, two frames" LEFT-only trap).
        # LEFT decode anchor + wrt_start are ALWAYS re-synced to the measured EE — this is the
        # LEFT clutch (the model decodes its delta from where the arm ACTUALLY is), and it must
        # run whether or not we are armed. Do NOT guard these two lines (guarding them would
        # freeze the LEFT decode anchor while armed → a genuine LEFT drift/jump).
        # --anchor-mode command replaces the measured pose with the LAST EXECUTED COMMANDED one
        # (leaked onto an eps ball around the measurement, and decayed back while in contact).
        # Default "measured" ⇒ identity, so the next two lines are the old ones exactly.
        # NOTE this deliberately also moves the PROPRIO base, because push_proprio_from_anchor
        # derives it from the anchor and "obs base == action base" is structural in LivePlanner.
        # That is the correct direction, not a side effect: training's obs_mode=last_target feeds
        # the last COMMANDED target, so anchoring on the command makes deploy agree with training
        # rather than disagree with it.
        _m_rot6d = _urdf_rot6d_to_model(tcp_rot6d)
        if not rframe.vive_armed:
            # DISARMED means the operator has taken the arm back, so any carried command anchor
            # is stale intent — drop it and re-bootstrap from the measurement. Without this the
            # LEFT decode anchor would sit at a command issued before the intervention (bounded
            # by --anchor-eps, but still not what the operator is looking at).
            if _anchor_reset is not None:
                _anchor_reset("disarmed")
        _a_pos, _a_rot6d = (_anchor_resolve(tcp_pos, _m_rot6d) if _anchor_resolve is not None
                            else (tcp_pos, _m_rot6d))
        planner.commit_role("left", _a_pos, _a_rot6d)  # anchor = measured EE, or the carried command
        planner.push_proprio_from_anchor()               # wrt_start base = the same pose
        # RIGHT+HEAD hold AND the broadcast wire are PRE-ARM ONLY. In ABSOLUTE mode the low level
        # applies the broadcast wire as the hand TARGET every tick (holosoma_wbt.py:3341, gated
        # only by _vive_armed). While ARMED, re-reading the MEASURED pose into the R/H hold (and/or
        # re-publishing it) replaces the servo's held target with the robot's actual (settled)
        # pose, which differs from the last commanded target by the FK/servo steady-state error.
        # The next Predict then re-measures the moved pose and re-commands it → the un-predicted
        # arm ratchets along that error (observed: RIGHT hand creeps outward, one press = one step,
        # the "opening embrace"). The LEFT clutch above is immune (the model overwrites slot 0
        # every step), but R/H are an ABSOLUTE held target with no such correction, so the hold
        # update MUST be gated exactly like the re-publish and _idle_sync_wire (which returns on
        # vive_armed). So: sync the R/H hold + broadcast wire to measured ONLY while NOT yet armed
        # (pre-arm gate sees ~0 gap, arming causes no jump); once armed, R/H stay frozen at the
        # arm-time measured pose and the wire moves solely by a deliberate execute (_drip).
        if not rframe.vive_armed:
            mapper.set_hold_wire(wire_pos9, wire_orn12)  # PRE-ARM: R/H hold tracks measured
            _set_last_wire(*mapper.full_wire(tcp_pos, tcp_rot6d), None)
        elif not quiet:
            # CONFIRM-ON-RIG: prove the ratchet is gone. With the fix the RIGHT hold must be
            # byte-CONSTANT across every armed Predict; before the fix it crept one step per press.
            _log(f"armed re-sync: LEFT anchor updated; RIGHT/HEAD hold FROZEN "
                 f"(right contact held at {np.round(mapper.hold_pos_wire[1], 4).tolist()} m).")
        return True

    def _measured_left_contact_n() -> float | None:
        """Measured LEFT-hand contact force magnitude (N) from a fresh RecordTap frame, or None.

        Uses ``f_meas_ungated`` — the TRUE reading — NOT the gated ``f_meas``. This deploy runs
        with ``GATE_CMD=1``, which hard-zeroes the gated value whenever the planner is commanding
        no force (i.e. all of pure-position mode), so reading the gated buffer would report 0 N
        while the hand is genuinely pressing and the contact cut-out would never fire.
        """
        if not _reach_active():
            return None
        rmsg, rw, _rn = subs["record"].latest()
        rframe = parse_record_frame(rmsg)
        if rframe is None or not rw or (time.time() - rw) > 1.0:
            return None
        f = rframe.f_meas_ungated if rframe.f_meas_ungated is not None else rframe.f_meas
        if f is None:
            return None
        arr = np.asarray(f, dtype=np.float64)
        if arr.shape != (2, 3) or not np.isfinite(arr[0]).all():
            return None
        return float(np.linalg.norm(arr[0]))  # row 0 = LEFT (the only hand the HL commands)

    def _measured_left_tcp() -> np.ndarray | None:
        """The robot's MEASURED LEFT UMI-TCP position (pelvis frame) from a fresh RecordTap frame,
        or None when there is none (RECORD off / stale / offline / FK unavailable).

        Fed to the reach compensator as the "where the arm actually is" term. Position only, and
        deliberately NOT converted between the URDF and model rotation frames: both series are
        absolute pelvis-frame POINTS and the +x TCP lever is seam-invariant (the same reasoning
        ``_payload_from_chunk`` relies on when it converts only orientation), so the difference
        against a decoded chunk position is already the physical shortfall.

        Returns None rather than a stale pose on purpose: the compensator treats None as
        "unknowable this step" and holds its residual instead of integrating against an old
        reading, which is what keeps a dead RecordTap from winding the accumulator."""
        if _fk_urdf is None:
            return None
        rmsg, rw, _rn = subs["record"].latest()
        rframe = parse_record_frame(rmsg)
        if rframe is None or not rw or (time.time() - rw) > 1.0:
            return None
        meas = _measured_full_wire(rframe.dof_pos)
        if meas is None:
            return None
        tcp_pos = np.asarray(meas[0], dtype=np.float64)
        return tcp_pos if np.isfinite(tcp_pos).all() else None

    _idle_warn = {"stale_logged_at": 0.0}  # rate-limit the RECORD-off / stale-frame warning

    def _idle_sync_wire() -> None:
        """PRE-ARM-only: while the low level is NOT yet armed, refresh the broadcast wire +
        the RIGHT/HEAD hold pose to the robot's MEASURED pose, so the pre-arm gate passes with
        ~0 gap and arming causes no jump. Does NOT touch the LEFT decode anchor (that is
        re-synced on Predict/Reset and must persist for a staged chunk); this only mirrors the
        WIRE to the live robot.

        CRITICAL — freezes ONCE ARMED. In ABSOLUTE mode the low level applies the wire pose as
        the hand TARGET directly. If we kept chasing the measured pose while armed, the target
        would be reset to the robot's actual pose every tick, ERASING the position error the
        controller needs to hold against gravity → the arm would droop / go limp. So the
        instant the RecordTap reports vive_armed, we STOP tracking and freeze the wire at
        whatever pose armed it (which, having just tracked measured, == the robot). Predict/Run
        then drive the wire deliberately; the idle heartbeat only holds that last commanded
        pose (a real holding error), never re-chases measured.

        Best-effort: a stale/absent RecordTap frame holds the last wire (the home seed until the
        first live frame) and rate-limit-warns so the operator knows the arm gate will refuse
        (RECORD=1 off / low level not broadcasting). Called by the heartbeat under its own idle
        guard (not dripping, not busy, nothing staged)."""
        if _fk_urdf is None:
            return
        rmsg, rw, _rn = subs["record"].latest()
        rframe = parse_record_frame(rmsg)
        if rframe is None or not rw or (time.time() - rw) > 1.0:
            # FAIL-SAFE: the robot readback is stale, so we can no longer confirm the arm is
            # still tracking a commanded force. Drop any force latched by the last drip while
            # holding the last pose (re-publish the same pose with fcmd=None) so a held F^cmd is
            # not republished forever on a dead readback. The master switch still governs force
            # at publish anyway; this only zeroes the LATCHED value so ON+stale can't push force.
            hp, ho, hf = _last_wire_snapshot()
            if hf is not None:
                _set_last_wire(hp, ho, None)
            now = time.time()
            # In --offline the absent RecordTap is the STATED intent, not a fault, so the warning
            # is noise (and its advice — relaunch with RECORD=1 — is wrong for this mode).
            if not args.offline and now - _idle_warn["stale_logged_at"] > 10.0:
                _idle_warn["stale_logged_at"] = now
                _log("RecordTap stale/absent (RECORD=1 on the low level?) — the broadcast wire "
                     "is the home seed; ABSOLUTE-mode arming will REFUSE until the robot's live "
                     "pose is seen. Home the arm to the ready pose and relaunch with RECORD=1.")
            return
        # ONCE ARMED, FREEZE: never chase the measured pose while the low level is applying the
        # wire as its hand target (would erase the servo's gravity-holding error → droop).
        if rframe.vive_armed:
            return
        meas = _measured_full_wire(rframe.dof_pos)
        if meas is None:
            return
        tcp_pos, tcp_rot6d, wire_pos9, wire_orn12 = meas
        mapper.set_hold_wire(wire_pos9, wire_orn12)
        _set_last_wire(*mapper.full_wire(tcp_pos, tcp_rot6d), None)

    # Register the idle tracker with the heartbeat (only if the measured-EE FK is available;
    # None ⇒ FK unavailable — the heartbeat holds the home seed and _idle_sync_wire's own
    # stale-frame warning never fires, so warn once here). This is what makes ABSOLUTE-mode
    # arming pass without a jump.
    if _fk_urdf is not None:
        wire_sync["fn"] = _idle_sync_wire
    else:
        wire_sync["fn"] = None
        _log("measured-EE FK unavailable — the broadcast wire stays at the home-anchor seed; "
             "ABSOLUTE-mode arming may REFUSE if the running policy's ready pose differs from "
             "the home anchor. (This is the RECORD-off / no-yourdfpy fallback.)")
    # Same registration for the ENSEMBLED executor's per-step clutch re-sync (it lives in
    # main(), beside the other wire writers, but the FK it needs is built here). Left None when
    # the FK is unavailable, which makes the per-step decode open-loop rather than crashing.
    if anchor_sync is not None:
        anchor_sync["fn"] = _resync_anchor_from_robot if _fk_urdf is not None else None
    # Same registration for the reach compensator's measured-pose term. Left None when the FK is
    # unavailable, which makes the compensator hold its residual rather than integrate blind.
    if measured_tcp is not None:
        measured_tcp["fn"] = _measured_left_tcp if _fk_urdf is not None else None
        # The contact cut-out does NOT need the FK (it reads the RecordTap's own pelvis-frame
        # wrench), so register it unconditionally: a rig with no FK still must not integrate a
        # shortfall into a harder press.
        measured_tcp["force_fn"] = _measured_left_contact_n
        if _fk_urdf is None and _reach_active():
            _log("reach compensation is ON but the measured-EE FK is unavailable (RECORD=1 / "
                 "yourdfpy?) — with no readback it cannot see the shortfall, so it will hold a "
                 "zero residual and change nothing. Fix the RecordTap or turn it off.")

    # /follow's SEED pose — where the triad cluster sits before the first RecordTap frame.
    #
    # It MUST be the low level's real ready pose (``home_anchor.json`` default_qpos), NOT the
    # mink stand keyframe. The mink stand pose holds the left wrist LEVEL at (0.200, 0.206,
    # 0.866); the deploy ready pose holds it 44° NOSE-DOWN at (0.097, 0.215, 0.736) — 166 mm
    # and 47° apart. Seeding from mink therefore drew the whole labelled cluster (tcp /
    # tracker / wrist_yaw_link) floating ~17 cm off the rendered arm and pointing the wrong
    # way, which reads exactly like a URDF or frame bug and is not one: the URDF is
    # byte-identical to holosoma's and its FK is correct. Prefer the yam-URDF FK at
    # default_qpos (identical to the low level's Pinocchio bake to machine precision) and fall
    # back to the mink pose only when that FK is unavailable.
    #
    # ``home_anchor`` is re-read from disk here rather than taken from main()'s local: it is NOT a
    # parameter of this function, so the earlier `home_anchor.get(...)` raised NameError inside the
    # broad `except` below and silently degraded EVERY run to the mink fallback — invisible with a
    # live robot (the first RecordTap frame moves /follow onto the measured wrist a moment later)
    # but permanent OFFLINE, where the seed pose is the only pose the triads ever get.
    # SAME ``_seed_dof`` the MESH above was posed with — so the cluster lands on the rendered arm by
    # construction, whichever seed pose is in force. Using a second, independently-derived pose here
    # is exactly how the 166 mm / 47 deg mismatch arose.
    home_wrist_pos = home_wrist_wxyz = None
    seed_anchor_dof = None
    if _fk_urdf is not None and _seed_dof is not None:
        try:
            with _fk_lock:
                _fk_urdf.update_cfg(remap_dof_to_urdf(_seed_dof, _fk_remap))
                _Tr = np.asarray(_fk_urdf.get_transform("left_wrist_yaw_link"), dtype=float)
            home_wrist_pos = base_xyz + _Tr[:3, 3]
            home_wrist_wxyz = _rot6d_to_wxyz(rotmat_to_rot6d(_Tr[:3, :3]))
            seed_anchor_dof = _seed_dof
            _log("follow-frame seed: yam-URDF FK at "
                 + ("default_dof_angles with the 14 ARM joints ZEROED — the low level's real ready "
                    "pose (forearms horizontal, HOLOSOMA_INIT_READY_POSE=1)" if args.zero_arms else
                    "the raw crouch default_qpos (--no-zero-arms)"))
        except Exception as e:  # noqa: BLE001 — fall back to the mink pose, loudly
            _log(f"follow-frame seed: ready-pose FK failed ({e!r}); using the mink stand pose")
    if home_wrist_pos is None:
        _home_conf = ik._solver.configuration
        _home_wrist = _home_conf.get_transform_frame_to_world(
            frame_name="left_wrist_yaw_link", frame_type="body")
        home_wrist_pos = np.asarray(_home_wrist.translation(), dtype=float)
        home_wrist_wxyz = np.asarray(_home_wrist.rotation().wxyz, dtype=float)
        _log("follow-frame seed: mink STAND pose — the triads will sit ~166 mm / 47° off the "
             "robot's ready pose until the first RECORD=1 frame arrives")
    # RE-SEED THE DECODE ANCHOR *AND* THE IDLE WIRE ONTO THE SEED POSE.
    #
    # main() seeded both from home_anchor's stored WIRE, i.e. the raw crouch default_qpos. With
    # --zero-arms (the default) the pose the LOW LEVEL actually holds after 'A' is the arms-zeroed
    # one, so leaving the seed at the crouch pose is wrong in two ways at once:
    #   * the drawn cluster/green path would decode against a TCP 199 mm / 44 deg from the rendered
    #     arm — re-opening the "trajectory doesn't start at the gripper" gap;
    #   * the BROADCAST WIRE would sit 310 mm/hand from the low level's pre-arm gate reference
    #     (``_motion_command_t[0,:29]``, rewritten to the same arms-zeroed pose), against a 70 mm
    #     tolerance — so arming would REFUSE until the idle heartbeat happened to re-sync it.
    # Re-seeding here fixes both, and applies on the LIVE path too (not just --offline) precisely
    # because the wire consequence is a live-rig problem. RIGHT/HEAD are included via set_hold_wire
    # so the held hands start at the pose the robot holds them at, not the crouch pose.
    #
    # Once armed, _resync_anchor_from_robot keeps taking over from the MEASURED pose every Predict;
    # this only makes the PRE-ARM state correct.
    if seed_anchor_dof is not None and _fk_urdf is not None:
        try:
            with _fk_lock:
                _fk_urdf.update_cfg(remap_dof_to_urdf(seed_anchor_dof, _fk_remap))
                _seed_contacts = np.zeros((3, 3))
                _seed_quats = np.zeros((3, 4))
                _Ta = None
                for _i, _body in enumerate(_WIRE_BODIES):
                    _T = np.asarray(_fk_urdf.get_transform(
                        frame_to=_body, frame_from="pelvis"), dtype=float)
                    _seed_contacts[_i] = _T[:3, 3] + _T[:3, :3] @ mapper.vr_offsets[_i]
                    _seed_quats[_i] = rotmat_to_quat_xyzw(_T[:3, :3])
                    if _i == 0:
                        _Ta = _T
            _a_rot6d = rotmat_to_rot6d(_Ta[:3, :3])
            _a_tcp = _Ta[:3, 3] + _Ta[:3, :3] @ _TCP_OFFSET
            planner.commit_role("left", _a_tcp, _urdf_rot6d_to_model(_a_rot6d))
            # RE-LATCH THE wrt_start REFERENCE TOO, not just the anchor. `commit_role` moves the
            # anchor and deliberately never touches `_start_rot6d`, so before this line the
            # state channel was measured against main()'s seed — home_anchor's stored wire, the
            # raw CROUCH pose, whose wrist +x sits 44.45 deg nose-down from the arms-zeroed
            # ready pose the robot actually holds (FK-verified; the two rotations are 47.08 deg
            # apart). That constant pre-rotation then composed into EVERY state of the run and
            # pushed part of the channel outside its training quantiles. Measured cost on
            # ckpt_diff_umi: 0.22 cm of predicted path, 3.6% of a chunk's own travel — small,
            # because the state channel carries ~2-4% of this model's sensitivity, but free to
            # remove. `↺ Reset start pose` always fixed it; nothing MADE the operator press it,
            # and the runbook listing it in the bring-up is not a guarantee. Order matters: the
            # anchor is already the measured ready pose above, so wrt_start is exactly identity
            # here — the same state the button produces.
            planner.reset_start_pose()
            planner.push_proprio_from_anchor()
            mapper.set_hold_wire(_seed_contacts.reshape(-1), _seed_quats.reshape(-1))
            if not args.offline:
                _set_last_wire(*mapper.full_wire(_a_tcp, _a_rot6d), None)
            _log(f"anchor + idle wire seeded at the "
                 f"{'ARMS-ZEROED ready pose (matches HOLOSOMA_INIT_READY_POSE=1)' if args.zero_arms else 'raw crouch default_qpos'}"
                 f": LEFT TCP {np.round(_a_tcp, 4).tolist()}, LEFT contact "
                 f"{np.round(_seed_contacts[0], 4).tolist()} (pelvis).")
        except Exception as e:  # noqa: BLE001 — keep main()'s home-anchor seed
            _log(f"anchor/wire re-seed failed ({e!r}); both stay at home_anchor's stored wire. The "
                 "green path may start off the drawn gripper, and ABSOLUTE-mode arming may REFUSE "
                 "until the idle heartbeat re-syncs to the measured pose.")
    # /follow rides the LEFT WRIST-YAW link (see the render loop: it is driven to
    # ``wrist_pos = tcp_pos - R @ _TCP_OFFSET``). Its own axes are turned OFF and the three
    # named triads below are hung under it as children at their fixed wrist-relative offsets
    # — the SAME cluster the collection viz and replay_zarr_viz draw.
    #
    # WHY THIS MATTERS (the bug this fixes): the frame previously drew its own triad AT THE
    # WRIST, while the yellow rollout path and the green recorded path are both drawn at the
    # **TCP**. The TCP sits ``_TCP_OFFSET`` = 0.242 m along wrist +x, so the only triad in the
    # scene stood 21 cm behind the start of both paths and the trajectory read as if it began
    # off in space. The paths were never wrong (verified: the green path's reference frame
    # coincides with the decoded chunk's step 0 to 0.0 m) — the DATUM was simply unlabelled.
    # Drawing tcp / tracker / wrist_yaw_link makes the datum explicit and matches the zarr
    # replay the operator already trusts, so the two views can be compared triad-for-triad.
    follow = server.scene.add_frame("/follow", show_axes=False, origin_radius=0.0,
                                    position=tuple(home_wrist_pos), wxyz=tuple(home_wrist_wxyz))
    # The "tracker" triad must show the PHYSICAL PUCK's axes, which are the wrist axes
    # right-multiplied by the measured (−y,−x,−z) relabel — i.e. 180° away from the wrist. Passing
    # a position only would make it INHERIT /follow's wrist orientation and render identical to
    # the tcp/wrist triads, which is impossible (the TCP is DERIVED from the raw puck through that
    # relabel) and is exactly the bug the operator caught by eye in the UMI zarr replay. tcp and
    # wrist_yaw_link genuinely DO share the wrist axes (the TCP is a pure +x translation), so they
    # get identity — the difference between the triads is the point of drawing three of them.
    _TRIAD_LOCAL_ROT = {"tracker": _PUCK_TO_WRIST_RELABEL}
    for _name, _off in triad_offsets_from_tcp().items():
        # ``triad_offsets_from_tcp`` measures from the TCP; /follow sits at the WRIST, so add
        # the wrist->TCP lever back to land each triad at its wrist-relative place. Going
        # through the TCP-relative helper (rather than TRIAD_OFFSETS_WRIST directly) keeps the
        # *drawn* geometry tied to the same function the green/yellow path datum is checked
        # against, so a future change to the lever cannot move the paths without moving these.
        _len, _rad, _orig = TRIAD_DRAW_SIZES[_name]
        _rot = _TRIAD_LOCAL_ROT.get(_name, np.eye(3))
        server.scene.add_frame(f"/follow/{_name}", show_axes=True, axes_length=_len,
                               axes_radius=_rad, origin_radius=_orig,
                               position=tuple(_TCP_OFFSET + _off),
                               wxyz=tuple(_rot6d_to_wxyz(rotmat_to_rot6d(_rot))))
        server.scene.add_label(f"/follow/{_name}/label", _name)
    rpy = np.asarray(args.cam_rpy, float)
    cam_rot = (vtf.SO3.from_rpy_radians(float(rpy[0]), 0.0, 0.0)
               .multiply(vtf.SO3.from_rpy_radians(0.0, float(rpy[1]), 0.0)))
    server.scene.add_frame("/follow/cam", show_axes=False,
                           position=tuple(np.asarray(args.cam_offset, float)), wxyz=tuple(cam_rot.wxyz))
    # TWO wrench arrows, both at a TCP, both at the SAME m/N scale so their lengths compare:
    #
    #   /wrench      F^cmd  — the model's PREDICTED wrench, negated (`as_command`), riding the
    #                         VIOLET GHOST's fingertip TCP as the ghost plays the chunk. This is
    #                         "the force this plan will ask for, at the pose it acts on".
    #   /wrench_meas F_meas — the MEASURED wrench, un-negated, at the REAL robot's fingertip TCP
    #                         (FK'd from the RecordTap joints). This is "what the rig actually
    #                         feels right now".
    #
    # WHEN THE LOOP IS CLOSED AND TRACKING, THE TWO POINT OPPOSITE WAYS. That is not a bug and it
    # is the single most useful thing on the page: the low level drives `F^cmd + F_meas -> 0`, so
    # anti-parallel arrows of equal length ARE the force loop at its fixed point. Same-direction
    # arrows mean a sign error somewhere; a long cmd with a short meas means the arm is not
    # reaching the surface.
    #
    # Colours are taken from the offline inference viewer so one vocabulary covers both pages:
    # ORANGE is the prediction there and the commanded wrench here. Measured gets CYAN, a hue no
    # trajectory line uses (yellow plan / green demo / red executed / violet ghost).
    #
    # Neither may hang under /follow: they are fed WORLD-frame origins, and viser parents by path
    # prefix, so a child of a frame at world position p renders its own coordinates offset BY p —
    # drawing the arrow at ~2x the wrist position. Both are top-level, like /rollout and /replay.
    _WRENCH_CMD_RGB = (245, 145, 35)    # == inference/viewer._PREDICTION_ORANGE
    _WRENCH_MEAS_RGB = (0, 200, 220)
    wrench = rc.WrenchArrows(server, name="/wrench", show_torque=False,
                             force_color=_WRENCH_CMD_RGB, torque_color=(250, 205, 130))
    # as_command=False: a measurement is drawn as measured. The relabel still applies when the
    # source is the raw sensor wire; the pelvis-frame source bypasses both (see the render loop).
    wrench_meas = rc.WrenchArrows(server, name="/wrench_meas", show_torque=False, as_command=False,
                                  force_color=_WRENCH_MEAS_RGB, torque_color=(140, 215, 235))

    # --- GHOST ARM IK: a damped-least-squares solve on the LEFT-arm joints, IN THE YAM URDF ---
    #
    # WHY NOT the mink ``ik``: the mink ThreePointIK solves a DIFFERENT kinematic model (the HuMI
    # MJCF, missing wrist_yaw/waist joints) and its joints are name-remapped into the yam URDF for
    # rendering — so a mink solution's RENDERED yam wrist lands up to ~170 mm off the target
    # (measured), and its soft-QP posture task leaves tens of mm of residual on absolute targets.
    # That is fine for teleop (home-anchored, relative) but not for a ghost that must sit ON the
    # yellow path. Solving the 7 left-arm joints directly in the yam URDF — the SAME model the
    # ghost renders in — is frame-faithful to the render by construction (verified ~mm on
    # reachable targets). Legs/torso/right-arm/head stay at the ready pose (this deploy is
    # LEFT-only), so only the left arm + jaw animate.
    #
    # It runs on a background thread (see below), so its ~0.6-0.8 s/chunk cost never touches the
    # drip/arm path — the very stutter the mink fallback was disabled to avoid. yourdfpy FK is
    # ~0.4 ms, and this uses a THIRD yourdfpy URDF (its own config) so it never races the render
    # thread's ghost_urdf.update_cfg or the _fk_urdf measured-EE FK.
    _ghost_ready = ghost_urdf is not None and _init_cfg is not None and _fk_urdf is not None
    _ghost_fk = None
    _larm_urdf_idx = _larm_lo = _larm_hi = None
    _larm_axes = _larm_childs = None
    if _ghost_ready:
        try:
            import yourdfpy  # noqa: PLC0415

            _ghost_fk = yourdfpy.URDF.load(args.yam_urdf, load_meshes=False, build_scene_graph=True)
            _ghost_names = list(_ghost_fk.actuated_joint_names)
            _larm_names = [n for n in _ghost_names
                           if n.startswith("left_") and any(k in n for k in ("shoulder", "elbow", "wrist"))]
            _larm_urdf_idx = np.array([_ghost_names.index(n) for n in _larm_names], dtype=np.intp)
            _jmap = _ghost_fk.joint_map
            _larm_lo = np.array([float(_jmap[n].limit.lower) for n in _larm_names])
            _larm_hi = np.array([float(_jmap[n].limit.upper) for n in _larm_names])
            # For the ANALYTIC Jacobian below: each arm joint's rotation axis (in its own child
            # link frame) and that child frame's name. All 7 are revolute on this URDF; a
            # non-revolute one would need a prismatic column, so refuse rather than mis-solve.
            _nonrev = [n for n in _larm_names if str(_jmap[n].type) != "revolute"]
            if _nonrev:
                raise ValueError(f"ghost arm IK expects revolute arm joints; got {_nonrev}")
            _larm_axes = [np.asarray(_jmap[n].axis, dtype=np.float64) for n in _larm_names]
            _larm_childs = [_jmap[n].child for n in _larm_names]
        except Exception as e:  # noqa: BLE001
            _log(f"ghost arm IK unavailable ({e!r}); the plan will show as the yellow path only.")
            _ghost_ready = False

    def _ghost_wrist_fk(cfg_full: np.ndarray):
        """FK the yam URDF at ``cfg_full`` (actuated-joint order) → left wrist (pelvis frame)."""
        _ghost_fk.update_cfg(cfg_full)
        T = np.asarray(_ghost_fk.get_transform(frame_to="left_wrist_yaw_link", frame_from="pelvis"))
        return T[:3, 3], T[:3, :3]

    def _ghost_arm_jacobian(p_wrist: np.ndarray) -> np.ndarray:
        """(6,7) geometric Jacobian of the left wrist w.r.t. the arm joints, in the pelvis frame.

        ASSUMES ``_ghost_fk`` is already at the configuration of interest (the caller has just
        FK'd it), so this costs 7 cached ``get_transform`` lookups and no ``update_cfg``.

        Replaces a FINITE-DIFFERENCE Jacobian that re-ran ``update_cfg`` 7 extra times per
        iteration. That made 88% of the ghost IK's cost the derivative rather than the solve:
        ``update_cfg`` is 0.382 ms while ``get_transform`` is ~0.4 us, so the old inner loop was
        3.05 ms/iteration versus 0.38 ms here (measured 7.6x; 762 ms -> 100 ms per 50-step
        chunk). Agreement with the finite differences it replaces was verified to 1.7e-6 over
        random poses (see tests/test_ghost_arm_ik.py) — same solve, same trajectory, less work.

        For a revolute joint with world axis ``z`` through origin ``p_i``, the columns are the
        textbook ``[z x (p_wrist - p_i); z]``.
        """
        J = np.empty((6, len(_larm_childs)))
        for i, (child, axis) in enumerate(zip(_larm_childs, _larm_axes)):
            T_i = np.asarray(_ghost_fk.get_transform(frame_to=child, frame_from="pelvis"))
            z = T_i[:3, :3] @ axis
            J[:3, i] = np.cross(z, p_wrist - T_i[:3, 3])
            J[3:, i] = z
        return J

    def _ghost_dls(cfg_full, q, p_tgt, R_tgt, *, iters):
        """Damped least-squares step the 7 arm joints (in ``cfg_full``) toward a wrist target
        (pelvis frame). Warm-started at ``q``; returns the clamped arm-joint vector. Never raises."""
        for _ in range(iters):
            cfg_full[_larm_urdf_idx] = q
            p, R = _ghost_wrist_fk(cfg_full)   # leaves _ghost_fk AT this cfg (the Jacobian needs it)
            e_p = p_tgt - p
            e_r = R @ _so3_log(R.T @ R_tgt)
            if np.linalg.norm(e_p) < 1e-4 and np.linalg.norm(e_r) < 1e-3:
                break
            J = _ghost_arm_jacobian(p)
            e = np.concatenate([e_p, e_r])
            dq = J.T @ np.linalg.solve(J @ J.T + 0.05 * np.eye(6), e)
            q = np.clip(q + dq, _larm_lo, _larm_hi)
        return q

    def _solve_ghost_cfgs(tcp_pos: np.ndarray, tcp_rot6d: np.ndarray, grip_m):
        """Solve one full ghost config per chunk step (actuated-joint order, ready for
        ``ghost_urdf.update_cfg``). The wrist target is the SAME leg the yellow path uses:
        pelvis-frame TCP minus the wrist→TCP lever. Non-left joints stay at the ready pose."""
        tcp_pos = np.asarray(tcp_pos, dtype=np.float64)
        R = rot6d_to_rotmat(np.asarray(tcp_rot6d, dtype=np.float64))          # (S,3,3)
        wrist_p = tcp_pos - np.einsum("sij,j->si", R, _TCP_OFFSET)            # (S,3) pelvis frame
        S = tcp_pos.shape[0]
        cfgs = np.zeros((S, _init_cfg.shape[0]))
        q = _init_cfg[_larm_urdf_idx].copy()                                 # ready-pose arm seed
        for i in range(S):
            # Stand back while an inference is in flight (GIL). Checked PER STEP so a predict
            # never waits on more than one step's solve, and so a long chunk yields repeatedly.
            if infer_gate is not None:
                infer_gate.wait()
            cfg = _init_cfg.copy()
            q = _ghost_dls(cfg, q, wrist_p[i], R[i], iters=(14 if i == 0 else 4))
            cfg[_larm_urdf_idx] = q
            if grip_m is not None and grip_m.shape[0] > i and dof_remap.left_finger >= 0:
                cfg[dof_remap.left_finger] = jaw_joint_from_width_m(grip_m[i])
            cfgs[i] = cfg
        return cfgs

    # Self-check: solve the ghost at the READY pose (the anchor every offline chunk decodes from)
    # and confirm the rendered wrist lands where the yellow path's frame 0 will — a loud residual
    # here means the ghost would float off the plan (the exact failure the mink fallback had).
    if _ghost_ready and _fk_urdf is not None and seed_anchor_dof is not None:
        try:
            _chk = _solve_ghost_cfgs(_a_tcp[None, :], _a_rot6d[None, :], None)
            _wp, _ = _ghost_wrist_fk(_chk[0])
            _res_mm = float(np.linalg.norm((base_xyz + _wp) - home_wrist_pos) * 1000.0)
            _log(f"ghost arm IK: frame-0 residual {_res_mm:.1f} mm vs the ready-pose wrist "
                 f"({'OK' if _res_mm < 10.0 else 'HIGH — ghost may sit off the plan'}).")
        except Exception as e:  # noqa: BLE001
            _log(f"ghost arm IK self-check skipped ({e!r}).")

    gui = server.gui
    gui.configure_theme(control_layout="collapsible", control_width="large")
    with gui.add_folder("Low-level controller"):
        if args.offline:
            # Say VIZ-ONLY, not "waiting for the low-level": in this mode nothing is coming, and
            # the arming instructions below would be actively misleading.
            status_md = gui.add_markdown(
                "⚫ **VIZ-ONLY (--offline)** — no low level, nothing published. Execute disabled.")
            robot_src = gui.add_text("robot mesh", initial_value="URDF ready pose (offline)",
                                     disabled=True)
            gui.add_markdown(
                "The mesh sits at the low level's real **ready pose** (`home_anchor.default_qpos`) "
                "and the three triads (**tcp / tracker / wrist_yaw_link**) hang off its LEFT wrist. "
                "The green recorded path starts exactly at the **tcp** triad — that is the datum.")
        else:
            status_md = gui.add_markdown("🔴 **OFFLINE** — waiting for the low-level on :8770")
            robot_src = gui.add_text("robot mesh", initial_value="predicted-IK (RECORD=1 for live)", disabled=True)
            # ABSOLUTE mode ignores this wire until the operator ARMS it (holosoma_wbt
            # _toggle_vive_teleop, joystick 'start' = StateCommand.STAND_TOGGLE). Execute does
            # NOTHING before that — surfaced here so it isn't mistaken for a dead wire. A=START
            # only stands the policy; the SEPARATE 'start' button hands the hands to this wire.
            gui.add_markdown("Joystick to arm: **Y** ready pose → **A** START → **start** button = ARM")
        # The seam that decides the commanded ORIENTATION, surfaced so a frame check does not
        # require reading stderr. Identity == the current convention (raw tracker -> one extrinsic).
        gui.add_text("frame seam", initial_value=(
            "identity (no seam)" if np.allclose(_MODEL_SEAM, np.eye(3), atol=1e-9)
            else f"{np.degrees(np.arccos(np.clip((np.trace(_MODEL_SEAM) - 1) / 2, -1, 1))):.1f}deg"
        ), disabled=True)
    if replay_only:
        with gui.add_folder("Mode"):
            gui.add_text("source", initial_value="RECORDED REPLAY (no policy)", disabled=True)
            gui.add_text("layout", initial_value=loaded.layout.mode, disabled=True)
    else:
        with gui.add_folder(f"Model (HL-v2 {loaded.algorithm})"):
            gui.add_text("mode", initial_value=loaded.layout.mode, disabled=True)
            gui.add_text("To / steps", initial_value=f"{loaded.obs_history_len} / {loaded.num_steps}", disabled=True)
            gui.add_text("force hands", initial_value=str(loaded.n_force_hands), disabled=True)
    # RGB observation, fixed in the right sidebar (mirrors the offline inference viewer's
    # "Conditioning images" folder: one gui.add_image per obs-history frame, newest = t0).
    # Skipped in model-free replay: there is no model being conditioned, and the Replay panel
    # already shows the RECORDED frame, which is the image that actually explains the motion.
    img_handles = []
    if not replay_only:
        with gui.add_folder("Observation (RGB)"):
            _placeholder = np.zeros((16, 16, 3), dtype=np.uint8)
            for _t in range(loaded.obs_history_len):
                _suffix = " (t0)" if _t == loaded.obs_history_len - 1 else ""
                img_handles.append(gui.add_image(_placeholder, label=f"obs {_t}{_suffix}", format="jpeg"))
    with gui.add_folder("Force command (S0 F^cmd)"):
        # The user's two-level gate: master switch + settable threshold. OFF => pure
        # position controller (no F^cmd on the wire). ON => publish F^cmd only above gate.
        force_switch = gui.add_checkbox("Enable force command (else PURE POSITION)",
                                        initial_value=force_ctl["enabled"])
        gate_slider = gui.add_slider("Gate (N) — below = position, above = force",
                                     min=0.0, max=25.0, step=0.5, initial_value=force_ctl["gate_n"])
        force_status = gui.add_text("F^cmd", initial_value="OFF (pure position)", disabled=True)
        # State the conversion the wire performs, next to the switch that arms it. The model
        # predicts a MEASUREMENT; what is published is its NEGATION, because the low level drives
        # F^cmd + F_meas to zero (see _gate_wire_force).
        gui.add_text("sign", initial_value=CMD_FROM_MEAS_NOTE, disabled=True)
    with gui.add_folder("F/T sensor (F_meas — measured, NOT negated)"):
        tare_btn = gui.add_button("⊘ Tare F/T (in free space)")
        tare_status = gui.add_text("tare", initial_value="(idle)", disabled=True)
        # Source toggle: ticked = the EXACT pelvis-frame F_meas the low-level servo consumes
        # (RecordTap :8779, the _force_meas_pelvis buffer); unticked = the raw sensor-frame
        # wire on :8771 (what the bridge publishes before the low level rotates/gates it).
        ft_show_meas = gui.add_checkbox("Show low-level F_meas (else raw :8771 sensor)",
                                        initial_value=True)
        ft_live = gui.add_text("|F| (N)", initial_value="(waiting…)", disabled=True)
        ft_detail = gui.add_text("per-hand", initial_value="-", disabled=True)
        # OBSERVATION FRESHNESS — the age of the rings the MODEL reads, not of the sockets.
        # Deliberately measured at the planner: that is the only place a dead ingest THREAD is
        # visible (the subscriber would still look healthy), and it is the same number the
        # Predict refusal uses. A trailing "!" marks over-limit.
        obs_fresh = gui.add_text("obs age", initial_value="(waiting…)", disabled=True)
    with gui.add_folder("Gripper"):
        # WHY THE LABEL CARRIES THE REASON: without --command-gripper this checkbox is greyed
        # out, the model's predicted width is discarded every step, and the jaw bridge sits at
        # "target_frac=freeze" — i.e. the arm reaches the object and never closes, with nothing
        # on the page explaining why. A disabled control with no stated cause reads as a bug in
        # the policy rather than a launch flag, so state it here AND in the readout below.
        grip_toggle = gui.add_checkbox(
            "Command gripper (autonomous)" if args.command_gripper
            else "Command gripper — READ-ONLY: relaunch with --command-gripper",
            initial_value=bool(args.command_gripper), disabled=not args.command_gripper)
        gui.add_text(
            "mode",
            initial_value=("COMMANDING the LEFT jaw from the model's predicted width"
                           if args.command_gripper else
                           "READ-ONLY (proprioception only) — the jaw will NEVER close; "
                           "the predicted width is discarded. Relaunch with --command-gripper."),
            disabled=True)
        # CLOSING OVERSHOOT — live, because the right value depends on the object and on how
        # much grip force the operator wants, and neither is knowable from the data. The
        # recorded "closed" width is the TOOL'S thickness (42-48 mm measured), so replaying it
        # verbatim stops at the surface and only touches; this drives past it so the position
        # error becomes grip force. Applied to the CLOSED mode only, and never to the width the
        # model observes.
        grip_over_slider = gui.add_slider(
            "Close overshoot (mm)", min=0.0, max=40.0, step=0.5,
            initial_value=float(args.grip_overshoot_mm), disabled=not args.command_gripper)
        grip_below_slider = gui.add_slider(
            "engage below (mm)", min=10.0, max=95.0, step=1.0,
            initial_value=float(args.grip_close_below_mm), disabled=not args.command_gripper)
        # CONFIRMATION RUN: the fix for "it grips then loosens". Temporal ensembling blends a
        # BIMODAL gripper signal, so a single transient open inside a grasp walks the commanded
        # target 10.1% of travel back open (measured). Requiring N consecutive agreeing
        # predictions before the overshoot engages or cancels rejects exactly that flicker,
        # while still following the policy the moment it really means to open.
        grip_confirm_slider = gui.add_slider(
            "Confirm steps (close/open)", min=1, max=15, step=1,
            initial_value=int(args.grip_confirm_steps), disabled=not args.command_gripper)
        grip_hold_toggle = gui.add_checkbox(
            "Never loosen a confirmed grasp",
            initial_value=bool(args.grip_hold_tightest), disabled=not args.command_gripper)
        grip_sq_status = gui.add_text("squeeze", initial_value="(nothing commanded yet)",
                                      disabled=True)
    with gui.add_folder("Control") as f_control:
        # Same two buttons drive both sources; only the WORDING changes, because "Predict" is
        # the wrong verb for reading a recorded chunk off disk.
        predict_btn = gui.add_button("🔍 Load 1 recorded chunk (no motion)" if replay_only
                                     else "🔍 Predict 1 chunk (no motion)")
        # Offline: permanently disabled AND relabelled. Greying alone would read as "predict
        # first"; the label states WHY it can never enable. _run_staged refuses independently.
        run_staged_btn = gui.add_button(
            "▶ Run this chunk (disabled — offline)" if args.offline else "▶ Run this chunk",
            disabled=True)
        # SEQUENTIAL model rollout: predict→execute repeatedly until Stop / the time ceiling,
        # arming ONCE. This is the cadence upstream ACT actually evaluates in, and the only one
        # where temporal ensembling reaches its full survivor count (see _run_to_completion).
        # Offline / model-free replay have nothing to run, so it is disabled with the reason.
        _auto_ok = (not args.offline) and (not replay_only)
        auto_run_btn = gui.add_button(
            "▶▶ Run to completion (sequential)" if _auto_ok
            else ("▶▶ Run to completion (disabled — offline)" if args.offline
                  else "▶▶ Run to completion (no policy loaded)"),
            disabled=not _auto_ok)
        auto_stop_btn = gui.add_button("⏹ Stop sequential run", disabled=True)
        reset_btn = gui.add_button("↺ Reset start pose")
        # execute-N-of-S: the rollout always PREDICTS all S steps (preview shows the full
        # chunk); this dropdown limits how many are actually dripped to the robot on Run.
        # The anchor commits to the LAST EXECUTED step, so the next chunk branches there.
        # Default 4 (one predict→preview→execute at a time; there is NO continuous mode).
        # exec_ctl["n"] was snapped to these choices in main(), so it is always present.
        _exec_choices = [str(n) for n in _EXEC_CHOICES]
        exec_dropdown = gui.add_dropdown("Execute how many steps",
                                         options=tuple(_exec_choices), initial_value=str(exec_ctl["n"]))
        # TEMPORAL ENSEMBLING (ACT only). Offered DISABLED with the reason in the label when
        # the loaded checkpoint cannot do it, rather than hidden — an operator comparing ACT
        # against flow should be able to see that the knob exists and why it is unavailable.
        _ens_supported = bool(ens_ctl and ens_ctl.get("supported"))
        ens_toggle = gui.add_checkbox(
            "Temporal ensembling (re-query every step)" if _ens_supported
            else "Temporal ensembling (ACT checkpoints only)",
            initial_value=bool(ens_ctl and ens_ctl.get("enabled")),
            disabled=not _ens_supported)
        ens_status = gui.add_text(
            "ensemble",
            initial_value=("ON — Run re-queries every step and blends" if _ens_supported
                           else f"unavailable (algo={loaded.algorithm})"),
            disabled=True)
        # Wall-clock ceiling for the sequential rollout. A policy has no end of its own, so
        # this is the bound that guarantees the arm stops even if the operator looks away.
        auto_secs_slider = gui.add_slider(
            "Sequential run seconds", min=1.0, max=120.0, step=1.0,
            initial_value=float(auto_model_ctl["max_s"] if auto_model_ctl else 20.0))
        ctrl_status = gui.add_text("Status", initial_value="idle", disabled=True)
    # --- RECORDED-EPISODE REPLAY panel (only when --replay-zarr was passed) ---
    # Same Predict/Run buttons above drive it; this panel only selects WHICH chunk they get.
    # "Play episode" is the sequential mode: it repeats predict→execute at the exec-steps
    # cadence until the episode ends or Stop is pressed (the operator arms ONCE).
    replay_ep_dd = replay_frame_slider = replay_src_toggle = None
    replay_play_btn = replay_play_ol_btn = replay_stop_btn = replay_rewind_btn = None
    replay_status = replay_image = replay_show_path = None
    if replay is not None:
        _ep_lens = replay.episode_lengths()
        # The converter masks step 0 of some frames, so an episode's frame 0 is not always
        # playable (on clean94, 8 of 94 episodes start masked). Resolve each episode's FIRST
        # PLAYABLE frame up front and put it in the dropdown label, so the operator never
        # picks an episode that dies on Predict and never has to guess a start frame.
        _ep_first = [replay.first_usable_frame(i) for i in range(replay.n_episodes)]
        _ep_options = []
        for i, n in enumerate(_ep_lens):
            first = _ep_first[i]
            if first is None:
                _ep_options.append(f"ep {i}  ({n} fr)  ⚠ no playable frame")
            elif first != 0:
                _ep_options.append(f"ep {i}  ({n} fr)  ⚠ starts at {first}")
            else:
                _ep_options.append(f"ep {i}  ({n} fr)")
        _n_shifted = sum(1 for f in _ep_first if f not in (0, None))
        _n_dead = sum(1 for f in _ep_first if f is None)
        if _n_shifted or _n_dead:
            _log(f"replay: {_n_shifted} episode(s) have a masked first frame (auto-seeked to "
                 f"their first playable frame); {_n_dead} have none at all.")
        with gui.add_folder("Replay recorded episode"):
            gui.add_text("file", initial_value=os.path.basename(replay.path), disabled=True)
            gui.add_text("episodes", initial_value=str(replay.n_episodes), disabled=True)
            # THE source switch: which chunk Predict/Run consume. Model stays loaded either
            # way, so an operator can A/B the demo against the policy without a relaunch.
            # In a MODEL-FREE session there is no second option to offer — the dropdown is
            # built with the single choice so the control layout stays identical (and the
            # callback below can never select a source that has no chunk behind it).
            replay_src_toggle = gui.add_dropdown(
                "Chunk SOURCE",
                options=("recorded replay",) if replay_only else ("recorded replay", "model"),
                initial_value="recorded replay" if replay_ctl["source"] == "replay" else "model")
            replay_ep_dd = gui.add_dropdown("Episode", options=tuple(_ep_options),
                                            initial_value=_ep_options[replay_ctl["episode"]])
            # Start at the initial episode's first PLAYABLE frame, and keep replay_ctl in
            # sync with what the slider shows (main() seeded frame=0, which may be masked).
            _init_first = _ep_first[replay_ctl["episode"]] or 0
            with replay_lock:
                replay_ctl["frame"] = int(_init_first)
            replay_frame_slider = gui.add_slider(
                "Start frame", min=0, max=max(_ep_lens[replay_ctl["episode"]] - 1, 1),
                step=1, initial_value=int(_init_first))
            replay_play_btn = gui.add_button("⏩ Play episode (sequential)")
            # OPEN-LOOP alternative: decode the WHOLE episode once against the arm's current
            # pose and drip it straight through (no per-chunk re-anchor). The mean tracking
            # error it reports is a true open-loop number; the sequential button re-anchors every
            # chunk and folds the servo lag forward. Both play the same demo — pick the model.
            replay_play_ol_btn = gui.add_button("⏩ Play (open-loop, precomputed)")
            replay_stop_btn = gui.add_button("⏹ Stop", disabled=True)
            replay_rewind_btn = gui.add_button("⏮ Rewind to frame 0")
            replay_status = gui.add_text("replay", initial_value="idle", disabled=True)
            replay_show_path = gui.add_checkbox("Show recorded episode path (green)",
                                                initial_value=True)
            replay_image = gui.add_image(np.zeros((16, 16, 3), dtype=np.uint8),
                                         label="recorded frame", format="jpeg")
    with gui.add_folder("Readout"):
        rd_grip = gui.add_text("L gripper", initial_value="-", disabled=True)
        # ‖F^cmd‖ == ‖F_meas‖ (the meas→cmd conversion is a sign), so this number is the same
        # either way — but it is labelled F^cmd to match the arrow, the charts and the wire.
        rd_force = gui.add_text("|F^cmd[0]| (N)", initial_value="-", disabled=True)
        rd_chunks = gui.add_text("chunks run", initial_value="0", disabled=True)
    with gui.add_folder("View"):
        show_rollout = gui.add_checkbox("Show rollout (yellow) path", initial_value=True)
        # THE GHOST ROBOT — a translucent-violet G1 that plays the plan in a repeating loop, so
        # the operator sees the whole ARM run the chunk (the action), not just the yellow line.
        # On a RECORD=1 rig the SOLID mesh is the live robot; this ghost previews the plan it will
        # run. Disabled if the ghost mesh/IK failed to come up (falls back to the path only).
        # NAME THE JAW'S SOURCE, like the two force checkboxes below do for their arrows. The
        # ghost's jaw is the model's PREDICTED width (_solve_ghost_cfgs feeds it
        # jaw_joint_from_width_m(grip_m[i])) while the solid mesh's is the MEASURED one from
        # :8784 — two jaws in the same place, distinguished only by translucency. The "L gripper"
        # readout is the measured one; untick this to see the solid jaw alone.
        #
        # ⚠ HISTORY, so the next report is not mis-triaged the same way: "a ghost predicting the
        # reach phase drew a wide-OPEN violet jaw over a physically closed hand" was blamed on
        # exactly that overlap. It was not the overlap — jaw_joint_from_width_m was INVERTED
        # (fc624a6), so BOTH jaws rendered backwards and the two just disagreed about *when*.
        # Fixed at the source; if a jaw looks wrong again, check the map before the overlap.
        show_ghost = gui.add_checkbox(
            "Show ghost robot (plays the plan — its JAW is PREDICTED, not measured)",
            initial_value=True, disabled=not _ghost_ready)
        show_wrench = gui.add_checkbox(
            "Show F^cmd arrow — ORANGE, on the ghost's TCP (from recorded F_meas)" if replay_only
            else "Show F^cmd arrow — ORANGE, on the ghost's TCP (from predicted F_meas)",
            initial_value=True,
        )
        # THE MEASURED COUNTERPART. Same scale, same anchor kind (a fingertip TCP), different
        # robot: the SOLID live mesh instead of the violet ghost. Needs the RecordTap for the
        # anchor, so it is only meaningful with the low level up and RECORD=1 — the readout below
        # says so rather than leaving an empty scene to be misread as "no contact".
        show_wrench_meas = gui.add_checkbox(
            "Show F_meas arrow — CYAN, on the REAL robot's TCP", initial_value=True)
        # OFF by default, for parity with the offline inference viewer (which draws force arrows
        # only). The torque arrow is also the weaker quantity: it is re-expressed in TCP axes but
        # never re-referenced to the TCP origin (no r x f lever), so it is "the sensor's torque in
        # TCP axes", which reads better on the charts than as a 3D arrow.
        show_torque_arrows = gui.add_checkbox("…also torque arrows", initial_value=False)
        wrench_status = gui.add_text("arrows", initial_value="-", disabled=True)
        # Both arrows ride a TCP, so the rotation and the sign both need stating: the rotation says
        # which axis, the sign says which WAY, and "is it pushing the right way?" is unanswerable
        # from the page without the pair.
        gui.add_text("Arrow frame", initial_value=TCP_FROM_SENSOR_NOTE, disabled=True)
        gui.add_text("Arrow sign", initial_value=CMD_FROM_MEAS_NOTE, disabled=True)
        gui.add_text("Reading the pair",
                     initial_value="tracking ⇒ ORANGE and CYAN are OPPOSITE and equal "
                                   "(F^cmd + F_meas → 0). Same direction = a sign fault.",
                     disabled=True)
        show_trace = gui.add_checkbox("Show EXECUTED (red) path", initial_value=True)
    # --- EXECUTED-vs-RECORDED comparison sheet ---
    # The red line answers "where did the arm go" in the scene; this writes the same
    # comparison as NUMBERS + time series (xyz, rpy, error, jaw) so a replay can be judged
    # rather than eyeballed. Nothing is written unless the button is pressed — the trace
    # itself is in-memory only, so a session that never saves leaves no files behind.
    save_plot_btn = save_plot_status = trace_clear_btn = None
    with gui.add_folder("Executed vs recorded"):
        trace_readout = gui.add_text("trace", initial_value="(no samples yet)", disabled=True)
        save_plot_btn = gui.add_button("💾 Save comparison plot (PNG)")
        trace_clear_btn = gui.add_button("🧹 Clear executed path")
        save_plot_status = gui.add_text("saved to", initial_value="(nothing saved)", disabled=True)
    # --- DEPLOY RECORDING: turn this run into a re-runnable take ---
    # A TOGGLE, not a start/stop pair: the take boundaries are the EXECUTION boundaries the page
    # already has (Run / Play / sequential), so a second set of manual controls would only let
    # the two disagree. Armed => the next execution is recorded, end to end.
    rec_toggle = rec_status = None
    with gui.add_folder("Deploy recording"):
        rec_toggle = gui.add_checkbox(
            "⏺ Record executions (obs + plan + measured joints)",
            initial_value=bool(record_enabled is not None and record_enabled.is_set()),
            disabled=(recorder is None or args.offline))
        gui.add_text(
            "what",
            initial_value=("VIZ-ONLY — nothing executes offline, so there is nothing to record"
                           if args.offline else
                           "each Run/Play/sequential writes one take: the executed prediction's "
                           "own observation + raw chunk, the measured 29-DOF joints + TCP, the "
                           "published wire, the raw F/T, and the camera frames"),
            disabled=True)
        rec_status = gui.add_text("state", initial_value="idle (not recording)", disabled=True)
        # The RESOLVED absolute destination, on the page: takes land in a timestamped directory
        # by default, and an operator who has to go find them over SSH should not have to
        # reconstruct the path from stderr.
        gui.add_text("session dir",
                     initial_value=(str(recorder.session_dir) if recorder is not None else "-"),
                     disabled=True)

    # ---------------- the RUN RECORDER (extension, ported from the five-point page 2026-09-20) ----------------
    # The ⏺ button: record ONE ▶▶ run — every chunk an executor dripped (the model's wire rows, the
    # predicted wrench + jaw per row), every executed step, every frame that reached :8770 AS
    # PUBLISHED (reach residual applied; the heartbeat's re-sends too), every RecordTap frame with
    # the robot's three wire points FK'd from its 29 joints, every F/T frame, every camera frame —
    # into <--rollout-dir>/<task>/<YYYY-MM-DD_HH-MM-SS>/, the SAME layout scripts/hl_serving_5pt.py
    # writes, for scripts/plot_rollout_record.py and overleaf/tools/gen_fig_rollout_force_pred.py.
    # It is a TAP and nothing else (flore/s1_policy/serving/rollout_record.py): main() holds `taps`
    # and its one-line calls at _s0_send, the three executors and the three bridge subscribers;
    # this block owns the button, the folder and the measured-wire FK. Blue = will not record,
    # green = will / is recording. ⏹, the seconds ceiling and the low level leaving :8779 end the
    # recording WITH the run (they retire the sequential loop, whose wrapper closes the folder);
    # Reset returns the button to blue/idle and the next armed ▶▶ opens a new folder. It never
    # publishes, commits, plans, or reads the wire back, so a rollout with the button green is the
    # same rollout. It is a SIBLING of the deploy take recorder above, not a mode of it: a take is
    # a re-runnable dataset entry (observation + raw chunk, for reinfer_hl_take.py); this is the
    # run's time series (predicted vs measured force at wall time, camera frames) for the figures.
    if taps is None:  # a caller that built the page without main()'s holder: the block still runs
        taps = rollout_record.RolloutTaps()

    def _rollout_measure(msg):
        """RecordTap dict -> the robot's MEASURED 3-point wire ``(pos9, orn12)`` — the contact points
        in the low level's own [L, R, HEAD] convention, i.e. the SAME quantity the published wire
        and RecordTap's ``vr_cmd_pos`` are in. Runs on the recorder's own thread through
        ``_measured_full_wire``, which serialises the shared FK under ``_fk_lock``."""
        rframe = parse_record_frame(msg)
        if rframe is None or rframe.dof_pos is None:
            return None
        meas = _measured_full_wire(rframe.dof_pos)
        if meas is None:
            return None
        return meas[2], meas[3]

    def _rollout_scalars(d) -> dict:
        """A stats dict as plain JSON scalars (numpy floats -> float; anything else -> str)."""
        out = {}
        for k, v in (d or {}).items():
            if isinstance(v, (bool, int, float, np.integer, np.floating)):
                out[str(k)] = float(v) if isinstance(v, (float, np.floating)) else int(v)
            elif v is None:
                out[str(k)] = None
            else:
                out[str(k)] = str(v)
        return out

    def _rollout_info() -> dict:
        """The run's LIVE settings, read once when its folder opens (the static ones are in `meta.static`)."""
        with force_lock:
            f_on, f_gate = force_ctl["enabled"], force_ctl["gate_n"]
        ens_on = False
        if ens_ctl is not None and ens_lock is not None:
            with ens_lock:
                ens_on = bool(ens_ctl["enabled"] and ens_ctl["supported"])
        max_s = None
        if auto_model_ctl is not None and auto_model_lock is not None:
            with auto_model_lock:
                max_s = float(auto_model_ctl["max_s"])
        reach_on = None
        if reach_ctl is not None and reach_lock is not None:
            with reach_lock:
                reach_on = bool(reach_ctl["enabled"])
        source, rp_info = "model", None
        if replay is not None and replay_ctl is not None and replay_lock is not None:
            with replay_lock:
                source = "replay" if replay_ctl["source"] == "replay" else "model"
                rp_info = {"zarr": os.path.basename(str(replay.path)), "episode": int(replay_ctl["episode"]),
                           "frame": int(replay_ctl["frame"])}
        info = {"exec_steps": int(exec_ctl["n"]), "auto_max_s": max_s, "source": source, "replay": rp_info,
                "force_cmd": bool(f_on), "force_gate_n": float(f_gate),
                "grip_cmd": bool(grip_enabled.is_set()), "ensemble": ens_on,
                "reach_enabled": reach_on, "anchor_mode": str(args.anchor_mode)}
        try:
            info["reach"] = _rollout_scalars(reach_comp.stats()) if reach_comp is not None else None
        except Exception as exc:  # noqa: BLE001 - a readout, never a stop
            info["reach"] = repr(exc)
        try:
            info["anchor"] = _rollout_scalars(chunk_anchor.stats()) if chunk_anchor is not None else None
        except Exception as exc:  # noqa: BLE001
            info["anchor"] = repr(exc)
        return info

    def _rollout_git_sha() -> str:
        try:
            import subprocess

            out = subprocess.run(["git", "-C", _FLORE_ROOT, "rev-parse", "--short", "HEAD"],
                                 capture_output=True, text=True, timeout=2.0)
            return out.stdout.strip() if out.returncode == 0 else "unknown"
        except Exception:  # noqa: BLE001 - provenance, never a stop
            return "unknown"

    _rollout_image_key = str((getattr(loaded, "image_keys", None) or ("camera0_rgb",))[0])
    _rollout_static = {
        "page": "hl_serving (3-point wire; RIGHT + HEAD held at measured)",
        "checkpoint": (os.path.abspath(args.checkpoint) if args.checkpoint else None),
        "algo": str(loaded.algorithm), "layout": str(loaded.layout.mode),
        "action_dim": int(getattr(loaded.layout, "action_dim", 0)), "state_dim": int(loaded.layout.state_dim),
        "roles": list(loaded.layout.roles), "force_roles": list(force_roles),
        "grip_roles": list(loaded.layout.gripper_roles),
        "held_roles": ["right", "head"], "wire_roles": ["left", "right", "head"],
        "wire_labels": ["LEFT hand", "RIGHT hand", "head/torso"], "wire_bodies": list(_WIRE_BODIES),
        "wire_offsets": np.asarray(mapper.vr_offsets, dtype=np.float64).round(6).tolist(),
        "ready_pos": np.asarray(mapper.home_pos_wire, dtype=np.float64).reshape(-1).round(6).tolist(),
        "action_dt": 1.0 / float(ACTION_HZ), "action_horizon": int(ACTION_HORIZON),
        "frame_seam": str(_MODEL_SEAM_SRC), "urdf": str(args.yam_urdf),
        "anchor_mode": str(args.anchor_mode), "anchor_eps": float(args.anchor_eps),
        "reach_ki": float(args.reach_ki), "low_level_mode": "3pt-unified-anneal",
        "fcmd_layout": "[left xyz, right xyz] N, sensor frame; this wire carries the LEFT triple only "
                       "(the low level zeroes the RIGHT, FORCE_HANDS=left), so the right half is 0",
        "image_key": _rollout_image_key,
        "deploy_take_dir": (str(recorder.session_dir) if recorder is not None else None),
        "args": {k: (v if isinstance(v, (int, float, str, bool)) or v is None else str(v))
                 for k, v in sorted(vars(args).items())},
        "git": _rollout_git_sha(), "host": os.uname().nodename, "pid": os.getpid(),
        "python": sys.version.split()[0], "measured_fk": _fk_urdf is not None,
    }
    _rollout_task = (str(args.rollout_task).strip() if args.rollout_task
                     else rollout_record.task_from_checkpoint(args.checkpoint))
    # ONE camera view: this page subscribes to :8773 alone (a left_only checkpoint reads one image),
    # named by the checkpoint's own image key so a run folder reads like the training data does.
    _rollout_cameras = [{"cam": 0, "key": _rollout_image_key, "url": subs["camera"].url}]
    rollout_rec = rollout_record.RunRecorder(
        args.rollout_dir, _rollout_task, static_meta=_rollout_static, measure=_rollout_measure,
        info_provider=_rollout_info, log=_log, autoplot=bool(args.rollout_autoplot),
        plot_cmd=rollout_record.plot_argv, every_run=bool(args.rollout_every_run),
        cameras=_rollout_cameras, camera_every=int(args.rollout_camera_every),
        # this page has no STOP interrupt (⏹ retires at a chunk boundary and does NOT end a recording)
        ends_on="Reset / the low level disarming (joystick start) / this button")
    taps.recorder = rollout_rec
    _log(f"RECORD: ⏺ run folders under {rollout_rec.task_dir} — arm the button (blue → green), then Predict or "
         f"▶▶; the recording runs across chunks and ▶▶ runs until ↺ Reset / the low level disarming (joystick "
         f"start) / the button saves it (⏹ and the ceiling do not end it). Plots "
         f"{'run in the background when a recording is saved' if args.rollout_autoplot else 'by hand (--no-rollout-autoplot)'}: "
         f"{rollout_rec.plot_hint(rollout_rec.task_dir / '<stamp>')}")
    if rollout_rec.cameras:
        _log(f"RECORD: camera frames → <run>/camera/{_rollout_image_key}/NNNNNN.jpg + camera/frames.jsonl — "
             f"the bridge's own JPEG" + (f", every {rollout_rec.camera_every}-th frame"
                                         if rollout_rec.camera_every > 1 else ", every frame"))
    else:
        _log("RECORD: camera frames OFF (--rollout-camera-every 0)")
    if _fk_urdf is None:
        _log("RECORD: measured wire points unavailable (no FK URDF); measured.jsonl keeps the joints only")

    # Placed right under Control (viser orders siblings by `order`), where ▶▶ / ⏹ / Reset are.
    with gui.add_folder("⏺ Record a run", order=f_control.order + 0.5):
        _roll_label0, _roll_color0 = rollout_rec.ui()
        b_roll = gui.add_button(
            _roll_label0, color=_roll_color0,
            hint="BLUE = not recording; click → GREEN = the next 🔍 Predict / ▶▶ opens a recording. It stays "
                 "open across chunks — Predict → Run one chunk at a time, ▶▶ runs, ⏹, the seconds ceiling all "
                 "continue it — and ENDS on ↺ Reset, the low level DISARMING (joystick start) or a click on "
                 "this button → blue again with the folder's name. Reset → blue/idle; every recording gets its "
                 f"own time-named folder under {rollout_rec.task_dir}. Recorded: every executed chunk (the 3-point wire rows + "
                 "the predicted wrench + jaw), every executed step, every published frame, RecordTap "
                 "(joints → measured wire points), the F/T stream, and every frame of the camera the "
                 "model reads (camera/<image_key>/NNNNNN.jpg, full-size JPEG). Plot with "
                 "scripts/plot_rollout_record.py <folder>; paper figure: "
                 "overleaf/tools/gen_fig_rollout_force_pred.py --latest data/rollouts/<task>.")
        c_roll_every = gui.add_checkbox(
            "Record EVERY run (stay armed)", initial_value=bool(args.rollout_every_run),
            hint="ON: no click needed — the first Predict / ▶▶ after each end opens the next folder. The "
                 "button still shows recording / saved.")
        g_roll = gui.add_text("recording", initial_value=rollout_rec.describe()[:200], disabled=True)
    roll_ui = {"label": _roll_label0, "color": _roll_color0, "text": rollout_rec.describe()[:200]}

    def _rollout_refresh() -> None:
        """Button label + colour and the status line FROM the recorder's state (the status loop, 2 Hz,
        and every click). An update is sent only when the text changed, so an idle button costs
        nothing and a live count redraws at most 2 Hz."""
        label, color = rollout_rec.ui()
        if label != roll_ui["label"]:
            roll_ui["label"] = label
            b_roll.label = label
        if color != roll_ui["color"]:
            roll_ui["color"] = color
            b_roll.color = color
        text = rollout_rec.describe()[:200]
        if text != roll_ui["text"]:
            roll_ui["text"] = text
            g_roll.value = text

    def _rollout_toggle() -> str:
        """The ⏺ button's body: idle/done → armed, armed → idle, recording → end this recording now."""
        note = rollout_rec.toggle()
        _rollout_refresh()
        _log(f"RECORD: {note}")
        return note

    def _rollout_reset() -> None:
        """Reset's hook: a live recording is ENDED (Reset is an end gesture), then back to blue/idle (a
        different label than 'saved')."""
        note = rollout_rec.reset()
        _rollout_refresh()
        _log(f"RECORD: {note}")

    @b_roll.on_click
    def _(_e) -> None:
        _rollout_toggle()

    @c_roll_every.on_update
    def _(_e) -> None:
        rollout_rec.every_run = bool(c_roll_every.value)
        _rollout_refresh()

    # REACH COMPENSATION. Present even when the feature was not enabled at launch, so the folder
    # documents the knob — but the checkbox is DISABLED unless --reach-ki > 0, because the gain,
    # the clamp and the dead zone are safety-relevant numbers that belong on the launch command
    # (where they end up in the shell history and the take metadata) rather than being dialled
    # into a live rig mid-motion. Once launched with a gain, the tick box turns it on and off.
    with gui.add_folder("Reach compensation"):
        reach_toggle = gui.add_checkbox(
            "↗ Insist on un-executed motion",
            initial_value=bool(reach_comp.enabled),
            disabled=(not reach_comp.enabled) or args.offline)
        gui.add_text(
            "what",
            initial_value=(
                "OFF — relaunch with --reach-ki 0.05 to enable" if not reach_comp.enabled else
                f"adds the un-executed motion back onto the published target, ONLY while the arm "
                f"performed <{reach_comp.config.stall_frac:g} of what was asked over "
                f"{int(reach_comp.config.window)} steps (ki={reach_comp.config.ki:g}, clamp "
                f"{reach_comp.config.clamp_m * 100:.0f} cm, decay {reach_comp.config.decay:g}, "
                f"contact cut-out {reach_comp.config.contact_n:g} N). Wire only — the model's obs "
                f"and decode anchor are untouched."),
            disabled=True)
        reach_status = gui.add_text("residual", initial_value="0.0 mm", disabled=True)
    with gui.add_folder("Charts"):
        grip_chart = rc.make_single_chart(gui, "L gripper width (m)", "width",
                                          color_idx=1, y_range=(0.0, max(_JAW_OPEN_M * 2.2, 0.1)))
        # In replay the source is the chunk's RECORDED force/torque (what the Bota measured
        # during the demonstration) rather than a prediction — but either way what is PLOTTED is
        # the F^cmd derived from it (negated, see _gate_wire_force), so the charts, the arrow and
        # the wire all say the same thing. Label both facts: whose wrench, and which sign.
        _fw = "recorded" if replay_only else "predicted"
        force_chart = _make_force_chart(gui, uplot, rc, f"F^cmd = -{_fw} F_meas (N)", ("Fx", "Fy", "Fz"))
        torque_chart = _make_force_chart(gui, uplot, rc, f"T^cmd = -{_fw} T_meas (Nm)", ("Tx", "Ty", "Tz"))
        # --- LIVE MEASURED WRENCH, scrolling in wall time -----------------------------------
        # A different X AXIS from the two charts above, deliberately: those are per-CHUNK-STEP
        # (index 0..S-1 of one plan), this is the last FT_CHART_WINDOW_S seconds of real time,
        # exactly like the collection viz's in-scene F/T charts — which is the tool this mirrors,
        # down to `rc.make_xyz_chart` (itself a copy of `collection_viz._make_xyz_chart`, so the
        # colours and shape match what the operator already reads during recording).
        #
        # SENSOR frame and MEASURED sign: no relabel, no negation. These charts are the honest
        # readout of what the Bota publishes on :8771 — the same split the zarr/raw replay viewers
        # use (arrow re-expressed for deploy, charts left as the stored signal).
        ft_force_chart = rc.make_xyz_chart(gui, "F_meas force (N) — live, sensor frame", ("Fx", "Fy", "Fz"))
        ft_torque_chart = rc.make_xyz_chart(gui, "F_meas torque (Nm) — live, sensor frame", ("Tx", "Ty", "Tz"))
        gui.add_text("F_meas charts",
                     initial_value=f"last {FT_CHART_WINDOW_S:g} s, {FT_CHART_HZ:g} Hz interval-mean "
                                   f"(raw :8771 · NOT negated · NOT relabelled)",
                     disabled=True)
        # PEAK, not just the trace: the charts show interval MEANS, so a sub-interval spike is
        # flattened there by construction. This is the un-flattened worst case over the window —
        # the number to compare against the 40 N hard stop.
        ft_peak = gui.add_text("peak |F_meas|", initial_value="(waiting…)", disabled=True)

    # Torque arrows: a flag flip on both instances. Callback-thread safe — WrenchArrows only
    # re-adds scene nodes by name (viser's own idiom for a replace), and the render loop reads
    # `show_torque` on its next tick, so there is no shared mutable geometry to race.
    @show_torque_arrows.on_update
    def _(_e) -> None:
        want = bool(show_torque_arrows.value)
        wrench.set_torque_visible(want)
        wrench_meas.set_torque_visible(want)

    # Force-control callbacks: write the shared dict under force_lock (a consistent
    # snapshot is read at infer + at every publish tick + in the heartbeat).
    @force_switch.on_update
    def _(_e) -> None:
        with force_lock:
            force_ctl["enabled"] = bool(force_switch.value)
            on = force_ctl["enabled"]
            gate = force_ctl["gate_n"]
        force_status.value = f"ON (gate {gate:.1f} N)" if on else "OFF (pure position)"

    @gate_slider.on_update
    def _(_e) -> None:
        with force_lock:
            force_ctl["gate_n"] = float(gate_slider.value)
            on = force_ctl["enabled"]
            gate = force_ctl["gate_n"]
        if on:
            force_status.value = f"ON (gate {gate:.1f} N)"

    @exec_dropdown.on_update
    def _(_e) -> None:
        exec_ctl["n"] = int(exec_dropdown.value)
        ctrl_status.value = f"will execute {exec_ctl['n']} of {ACTION_HORIZON} steps"

    @ens_toggle.on_update
    def _(_e) -> None:
        # Takes effect on the NEXT Run (the cadence is chosen once, at the top of an execute),
        # so say so explicitly instead of implying a live switch mid-motion.
        want = bool(ens_toggle.value)
        with ens_lock:
            ens_ctl["enabled"] = want
        # Flipping the cadence changes what a stored prediction MEANS for the executed path,
        # so drop the survivors: re-enabling later must not blend against chunks from before
        # a stretch of un-ensembled motion the ensembler never saw.
        planner.reset_temporal_ensemble()
        _in_motion = dripping.is_set() or (auto_running is not None and auto_running.is_set())
        ens_status.value = (
            ("ON — Run re-queries every step and blends" if want
             else "OFF — Run drips one chunk (flow/diffusion cadence)")
            + (" · applies on the NEXT Run" if _in_motion else ""))

    # --- render state (mutated on the control-callback thread; render loop reads it) ---
    render_lock = threading.Lock()
    # ``ghost_cfgs`` is the translucent ghost robot's per-step yam config, or None until the
    # BACKGROUND arm-IK solve finishes a chunk (a fraction of a second after Predict). ``n_steps``
    # carries the chunk length for the animation cursor + the yellow path, so those keep working
    # while the ghost is still being solved. ``seq`` counts every cached chunk (preview included,
    # unlike ``n_chunks``) so a background solve can tell it has been superseded.
    latest = {"ghost_cfgs": None, "wrist_pos": None, "wrist_wxyz": None, "path": None,
              "force_pred": None, "grip_m": None, "dirty": False,
              "n_chunks": 0, "n_steps": 0, "seq": 0}
    roll_handle = {"h": None}
    # GREEN recorded-episode path (replay mode only). The HANDLE is owned by the render
    # thread; GUI callbacks only raise the REQUEST flag (see _request_replay_path).
    replay_path_handle = {"h": None}
    replay_path_req = {"dirty": False, "episode": 0, "frame": 0}

    # GHOST solve request — COALESCED to the latest staged chunk. _solve_and_cache stores the
    # pose arrays here + sets the event; _ghost_solver_loop (a background thread) does the
    # ~0.6-0.8 s arm IK OFF the drip/render threads and publishes latest["ghost_cfgs"] iff the
    # chunk is still current. Only the newest request matters (ensembling can re-stage every
    # step): an in-flight solve for a superseded chunk is dropped by the seq check at publish.
    ghost_lock = threading.Lock()
    ghost_pending = {"tcp_pos": None, "tcp_rot6d": None, "grip_m": None, "seq": -1}
    ghost_solve_req = threading.Event()

    def _solve_and_cache(out: dict, *, is_preview: bool) -> None:
        """Publish a chunk's render data (callback thread) and QUEUE the ghost arm IK.

        The YELLOW path + the follow frame (camera) + the wrench anchor are the model's ABSOLUTE
        pelvis-frame trajectory placed directly in the render world (pelvis at ``base_xyz``,
        identity orientation, so world = base_xyz + pelvis point) — the TRUE trajectory the low
        level servos to. All cheap (no IK), so it runs inline.

        The translucent GHOST that PLAYS the chunk is solved on a BACKGROUND thread
        (``_ghost_solver_loop``): its yam-URDF arm IK costs ~0.6-0.8 s per chunk, which must never
        sit on the drip/render path. That solve-on-the-hot-path was the "一顿一顿" stutter the old
        mink fallback caused (6.14 ms x 50 x 2 = 46% arm-idle duty), and it is why the ghost is
        queued, not solved here. ``ghost_cfgs`` is reset to None so the render loop hides the
        PREVIOUS chunk's ghost until the new one lands — frame 0 of every chunk is the anchor, so
        there is no misleading pose to freeze at meanwhile."""
        tcp_pos = out["tcp_pos"].astype(np.float64)      # (S,3) absolute pelvis-frame UMI-TCP
        tcp_rot6d = out["tcp_rot6d"].astype(np.float64)  # (S,6) absolute pelvis-frame rot
        force_pred = out.get("force_pred")
        grip_m = out.get("grip_m")
        S = tcp_pos.shape[0]

        # Absolute-frame path/anchors (base_xyz + pelvis point; identity pelvis rotation).
        R_tcp = rot6d_to_rotmat(tcp_rot6d)                                  # (S,3,3)
        path = base_xyz[None, :] + tcp_pos                                 # (S,3) yellow TCP path
        wrist_pos = base_xyz[None, :] + tcp_pos - np.einsum("sij,j->si", R_tcp, _TCP_OFFSET)
        wrist_wxyz = np.stack([_rot6d_to_wxyz(r) for r in tcp_rot6d]).astype(float)  # (S,4)

        grip_arr = np.asarray(grip_m) if grip_m is not None else None
        with render_lock:
            new_seq = latest["seq"] + 1
            latest.update(ghost_cfgs=None, wrist_pos=wrist_pos, wrist_wxyz=wrist_wxyz, path=path,
                          force_pred=(np.asarray(force_pred) if force_pred is not None and np.asarray(force_pred).size else None),
                          grip_m=grip_arr,
                          dirty=True, n_chunks=latest["n_chunks"] + (0 if is_preview else 1),
                          n_steps=int(S), seq=new_seq)
        # Stage the ghost solve (coalesced): the background thread picks up the LATEST request.
        # The request is always STORED but only DISPATCHED while the operator wants a ghost — the
        # solve exists solely to feed ``ghost_urdf.update_cfg`` behind ``show_ghost``, so with the
        # box unticked every one of these ~100 ms produced a config nothing renders (and, before
        # the yield gate, spent it fighting the sampler for the GIL). Storing it regardless is what
        # lets a re-tick show the CURRENT plan immediately instead of waiting for the next Predict
        # (``_ghost_solver_loop``'s seq check still drops it if a newer chunk has landed).
        if _ghost_ready:
            with ghost_lock:
                ghost_pending.update(tcp_pos=tcp_pos, tcp_rot6d=tcp_rot6d, grip_m=grip_arr, seq=new_seq)
            if bool(show_ghost.value):
                ghost_solve_req.set()

    def _ghost_solver_loop() -> None:
        """BACKGROUND thread: solve the ghost arm IK for the latest staged chunk, off the hot
        path. Coalesces (only the newest pending request is solved) and publishes
        ``latest["ghost_cfgs"]`` only if the chunk is still current — a solve superseded
        mid-flight is discarded. Viz-only: any error is logged and the ghost stays hidden."""
        while not stop.is_set():
            if not ghost_solve_req.wait(timeout=0.2):
                continue
            ghost_solve_req.clear()
            with ghost_lock:
                tcp_pos = ghost_pending["tcp_pos"]
                tcp_rot6d = ghost_pending["tcp_rot6d"]
                grip_m = ghost_pending["grip_m"]
                seq = ghost_pending["seq"]
            if tcp_pos is None:
                continue
            try:
                cfgs = _solve_ghost_cfgs(tcp_pos, tcp_rot6d, grip_m)
            except Exception as e:  # noqa: BLE001 — viz only, never crash the thread
                _log(f"ghost arm IK failed (ignored): {e!r}")
                continue
            with render_lock:
                if latest["seq"] == seq:   # still the current chunk (else superseded → drop)
                    latest["ghost_cfgs"] = cfgs

    @show_ghost.on_update
    def _(_e) -> None:
        """Re-ticking solves the CURRENTLY staged chunk instead of waiting for the next Predict.
        Without this, skipping the dispatch while unticked would leave the ghost invisible until
        something new was predicted, which reads as "the checkbox is broken"."""
        if not (_ghost_ready and bool(show_ghost.value)):
            return
        with ghost_lock:
            have = ghost_pending["tcp_pos"] is not None
        with render_lock:
            need = latest["ghost_cfgs"] is None
        if have and need:
            ghost_solve_req.set()

    def _release_gripper() -> str:
        """Command the LEFT jaw FULLY OPEN. Returns a suffix for the status line.

        Called from Reset start pose: a replay that ended mid-grasp leaves the jaw wherever
        the last dripped step put it, so "reset" would restore the ARM's datum while the hand
        stayed clamped — and the next play then starts from a gripper state the demonstration
        never began in. ``target_frac`` 1.0 is the same fully-open command the teardown path
        sends, so there is one meaning of "released" in this program.

        Refuses honestly rather than lying: offline there is no bridge, and with
        --command-gripper absent this planner is read-only on the jaw by contract, so silently
        writing to it would break that promise (and, on a rig where the operator is holding the
        object, would open the hand unasked)."""
        if args.offline:
            return "  · gripper untouched (offline)"
        if not args.command_gripper:
            return "  · gripper NOT reset (planner is read-only on the jaw; relaunch with --command-gripper)"
        # Drop the grasp LATCH first: this is a deliberate open, so the next grasp must start
        # clean rather than inheriting this grasp's tightest held target (which would clamp the
        # jaw the instant the next close engages, before the model asked for it).
        if _grip_release_latch is not None:
            _grip_release_latch()
        try:
            subs["gripper"].send({"target_frac": 1.0})
            jaw_ever_commanded["v"] = True
        except Exception as e:  # noqa: BLE001 — a dead bridge must not fail the pose reset
            _log(f"gripper release error (ignored): {e!r}")
            return "  · gripper release FAILED (bridge down?)"
        return "  · gripper released (open)"

    # --- EXECUTION TRACE: what the arm actually did, for the red line + the QC sheet ---
    # Written by the render loop (the only thread already FKing the live wrist every tick) and
    # read by GUI callbacks. See flore/s1_policy/serving/hl_trace.py for why the reference it is
    # compared against is the TRANSPORTED (commanded) demo rather than the raw recorded world
    # pose: the ~55 deg ready-pose reorientation is physical, and differencing against the raw
    # recording would report it as a huge tracking error while hiding the real one.
    trace = ExecutionTrace()
    trace_handle = {"h": None}
    # The commanded reference captured at ARM time, so the saved sheet compares against the
    # curve that was actually commanded for THIS play (the anchor moves between plays).
    trace_ref = {"ref": None}
    trace_req = {"dirty": False}

    def _capture_trace_reference(episode: int, ref_frame: int) -> None:
        """Snapshot the commanded demonstration path + its clock for the current anchor.

        Same transport as ``_poll_replay_path`` (and therefore as the executed chunk), so the
        red executed line, the green commanded line and the saved sheet all describe one
        curve. Best-effort: without it the trace still records and still draws, and the save
        button reports that there is no reference to compare against."""
        if replay is None:
            trace_ref["ref"] = None
            return
        try:
            rel_pos, rel_rot6d = replay.episode_path_relative(episode, ref_frame, "left")
            anchor_pos, anchor_rot6d = planner.snapshot_anchor()
            ri = loaded.layout.roles.index("left")
            abs_pos, abs_rot6d = _delta_ee_to_absolute(
                rel_pos, rel_rot6d, anchor_pos[ri], anchor_rot6d[ri])
            ts = replay.episode_times(episode)
            # The commanded rotation is published in the URDF wrist frame (the same conversion
            # _payload_from_chunk applies), and the trace records MEASURED URDF-frame wrists —
            # so convert here or the two series would sit a frame-seam apart.
            urdf_rot6d = np.stack([_model_rot6d_to_urdf(r) for r in abs_rot6d])
            trace_ref["ref"] = ReferencePath(
                t=np.asarray(ts, dtype=np.float64),
                pos=base_xyz[None, :] + abs_pos,
                rot6d=urdf_rot6d,
                grip_m=replay.episode_gripper_m(episode, "left"),
                # The trace's t=0 is the moment execution armed, which is THIS frame's time in
                # the demonstration — not the episode start (a play can begin mid-episode).
                t_offset=float(ts[int(ref_frame)]),
            )
        except Exception as e:  # noqa: BLE001 — viz only
            _log(f"trace reference capture error (ignored): {e!r}")
            trace_ref["ref"] = None

    def _arm_trace(episode: int | None, frame: int | None, label: str) -> None:
        """Start a fresh trace for one play (and capture its commanded reference)."""
        if episode is not None and frame is not None:
            _capture_trace_reference(int(episode), int(frame))
        trace.arm(time.monotonic(), episode=episode, label=label)
        trace_req["dirty"] = True

    def _request_trace_redraw() -> None:
        trace_req["dirty"] = True

    def _poll_trace_path() -> None:
        """RENDER-THREAD half of the RED executed path: redraw it from the live trace.

        Same threading contract as the green path — the scene graph belongs to this loop. The
        line is rebuilt (remove + add) rather than appended to because viser line segments are
        immutable; at 30 Hz over a 20 s episode that is a ~600-segment rebuild, so it is done
        on a ~4 Hz cadence instead of every tick (measured: a 600-segment rebuild is ~1 ms, but
        the point of this whole change was to keep the render loop off the arm's critical
        path)."""
        if not trace_req["dirty"]:
            return
        trace_req["dirty"] = False
        try:
            if trace_handle["h"] is not None:
                trace_handle["h"].remove()
                trace_handle["h"] = None
            if show_trace is not None and not bool(show_trace.value):
                return
            pts = trace.positions()  # positions only — no need to stack rot/grip for a line
            if pts.shape[0] >= 2:
                seg = np.stack([pts[:-1], pts[1:]], axis=1).astype(np.float32)
                col = np.broadcast_to(np.array(_TRACE_RED, np.uint8), (seg.shape[0], 2, 3)).copy()
                trace_handle["h"] = server.scene.add_line_segments(
                    "/trace/left", points=seg, colors=col, line_width=3.0)
        except Exception as e:  # noqa: BLE001 — viz only; never take down the render loop
            _log(f"trace path draw error (ignored): {e!r}")

    # --- control-plane callbacks (direct calls; no cross-process protocol) ---
    # ALL three (predict / run_staged / a continuous step) take the single `busy` mutex
    # non-blocking, so only ONE ever touches the GPU model + the shared ThreePointIK solver
    # at a time. A double-click Predict (or Predict during continuous) is dropped, not run
    # concurrently — two concurrent model.sample_actions / ik.step would corrupt the solver.
    # --- recorded-replay helpers (no-ops when --replay-zarr was not passed) ---
    def _replay_episode_len(episode: int) -> int:
        a, b = replay.episode_bounds(episode)
        return b - a

    def _note_replay_chunk(out: dict) -> None:
        """Refresh the Replay panel (cursor readout + recorded RGB) after a chunk is built.

        Viz-only and fully guarded: a stale image decode must never break predict/execute."""
        if replay is None or replay_status is None:
            return
        rp = out.get("replay")
        if rp is None:
            replay_status.value = "source = model (recorded replay idle)"
            return
        try:
            n = _replay_episode_len(rp["episode"])
            # NOT a fault, and deliberately not worded like one: these are the converter's
            # own per-channel masks for THIS frame's step 0. Every clean94 episode reports
            # force unmeasured at its first frames (the auto-tare settles ~0.5 s after the
            # operator pressed record), and ~1/3 report the gripper likewise. The pose is
            # still fully valid, and an unmeasured force is not published anyway (the switch
            # defaults OFF and these frames' |F| is far below the gate).
            flags = []
            if not rp["gripper_valid"]:
                flags.append("gripper unmeasured")
            if not rp["force_valid"]:
                flags.append("force unmeasured (pre-tare)")
            suffix = ("  [step 0: " + ", ".join(flags) + "]") if flags else ""
            replay_status.value = (
                f"ep {rp['episode']} frame {rp['frame']}/{n - 1} "
                f"(t={rp['time_s']:.2f}s, {rp['n_valid']} valid steps){suffix}")
            if replay_image is not None:
                replay_image.image = replay.image_at(rp["episode"], rp["frame"])
            # Re-transport the green path against the CURRENT anchor (Predict re-syncs it to
            # the measured EE, and each execute commits it forward), so the drawn
            # demonstration keeps registering on the arm through a long playthrough.
            _request_replay_path(rp["episode"], rp["frame"])
        except Exception as e:  # noqa: BLE001 — viz only
            _log(f"replay panel refresh error (ignored): {e!r}")

    def _request_replay_path(episode: int, ref_frame: int) -> None:
        """REQUEST a redraw of the green recorded-episode path (flag only, no scene work).

        Honours this file's threading contract — "buttons raise flags; the render loop
        mutates". Every caller is a viser GUI callback thread, and the viser scene graph is
        owned by the render loop (the yellow rollout path is likewise only ever added/removed
        there). Adding/removing a scene node from a callback thread would race the render
        loop's own add/remove of sibling nodes. The render loop consumes this via
        :func:`_poll_replay_path`."""
        if replay is None:
            return
        with replay_lock:
            replay_path_req["episode"] = int(episode)
            replay_path_req["frame"] = int(ref_frame)
            replay_path_req["dirty"] = True

    def _poll_replay_path() -> None:
        """RENDER-THREAD half of the green path: consume a pending request and mutate the
        scene. Draws the WHOLE recorded episode, transported into the robot's world by the
        SAME ``delta_ee_to_absolute`` + anchor the executed chunk decodes with, so the drawn
        demonstration and the commanded motion agree about the FRAME. Green matches the GT
        line colour ``replay_zarr_viz.py`` uses, so the two viewers read alike.

        The transport uses the anchor AS OF THE DRAW. Since the anchor is re-synced to the
        measured EE at each Predict and committed after each execute, the path is redrawn on
        every executed chunk (``_note_replay_chunk``) rather than left pinned to the anchor it
        was first drawn against — otherwise it would progressively misregister from the arm
        during a long sequential playthrough."""
        if replay is None:
            return
        with replay_lock:
            if not replay_path_req["dirty"]:
                return
            replay_path_req["dirty"] = False
            episode = int(replay_path_req["episode"])
            ref_frame = int(replay_path_req["frame"])
        try:
            if replay_path_handle["h"] is not None:
                replay_path_handle["h"].remove()
                replay_path_handle["h"] = None
            if replay_show_path is not None and not bool(replay_show_path.value):
                return
            rel_pos, rel_rot6d = replay.episode_path_relative(episode, ref_frame, "left")
            anchor_pos, anchor_rot6d = planner.snapshot_anchor()
            ri = loaded.layout.roles.index("left")
            # The SAME body-frame transport ``_infer_replay`` commands with, so the drawn line
            # shows where the arm actually goes — including the ~107° ready-pose reorientation
            # (``hl_replay._body_vs_world_transport``). Drawing a world-oriented path over
            # body-oriented motion would look right and be a lie.
            abs_pos, _ = _delta_ee_to_absolute(rel_pos, rel_rot6d, anchor_pos[ri], anchor_rot6d[ri])
            pts = base_xyz[None, :] + abs_pos
            if pts.shape[0] >= 2:
                seg = np.stack([pts[:-1], pts[1:]], axis=1).astype(np.float32)
                col = np.broadcast_to(np.array(_REPLAY_GREEN, np.uint8), (seg.shape[0], 2, 3)).copy()
                replay_path_handle["h"] = server.scene.add_line_segments(
                    "/replay/left", points=seg, colors=col, line_width=2.0)
        except Exception as e:  # noqa: BLE001 — viz only; never take down the render loop
            _log(f"replay path draw error (ignored): {e!r}")

    def _replay_seek(episode: int, frame: int) -> None:
        """Move the replay cursor (and retire any in-flight auto loop that owned it)."""
        with replay_lock:
            replay_ctl["episode"] = int(episode)
            replay_ctl["frame"] = int(frame)
            replay_ctl["auto"] = False
            replay_ctl["auto_gen"] += 1  # invalidate a running sequential loop

    def _replay_play_sequential():
        """Play the selected episode straight through: predict → execute, repeatedly.

        The operator arms ONCE; this loop then walks the whole demonstration at the
        exec-steps cadence. Every iteration goes through the SAME ``_predict``/``_drip`` path
        as a manual press (same clutch re-anchor, same live force gate, same commit), so
        nothing about the safety envelope changes — it only removes the need to click Run
        once per chunk. Stops on: Stop pressed (auto_gen bump), episode end, ``stop`` set,
        or any error."""
        if replay is None:
            return
        with replay_lock:
            replay_ctl["auto"] = True
            my_gen = replay_ctl["auto_gen"]
            episode = int(replay_ctl["episode"])
        auto_running.set()
        if replay_stop_btn is not None:
            replay_stop_btn.disabled = False
        n_chunks = 0
        try:
            n_frames = _replay_episode_len(episode)
            # ONE trace per play, armed BEFORE the first chunk so t=0 is the moment the arm
            # starts. The reference is captured against the anchor as it is now; _predict then
            # re-syncs the anchor to the measured EE per chunk, which moves it by the servo's
            # steady-state error only (sub-mm on this rig) — small compared to what the sheet
            # measures, and re-capturing per chunk would splice N differently-anchored curves.
            with replay_lock:
                _start_frame = int(replay_ctl["frame"])
            _arm_trace(episode, _start_frame, f"play ep{episode} from frame {_start_frame}")
            _record_arm(f"play ep{episode} from frame {_start_frame}")
            while not stop.is_set():
                with replay_lock:
                    if replay_ctl["auto_gen"] != my_gen or not replay_ctl["auto"]:
                        ctrl_status.value = "replay stopped"
                        break
                    if replay_ctl["episode"] != episode:
                        ctrl_status.value = "replay stopped (episode changed)"
                        break
                    cursor = int(replay_ctl["frame"])
                if cursor >= n_frames - 1:
                    ctrl_status.value = f"replay finished — episode {episode} end ({n_chunks} chunks)"
                    break
                # SKIP the converter's masked frames instead of dying on them. Some episodes
                # have a masked frame mid-demonstration (clean94: eps 1/27/28/30/31); without
                # this the sequential loop would abort there and abandon the rest of the
                # playthrough after the operator already armed (ep 1 died 13 frames into 301).
                usable = replay.next_usable_frame(episode, cursor)
                if usable is None:
                    ctrl_status.value = (
                        f"replay finished — no playable frame left in episode {episode} "
                        f"({n_chunks} chunks)")
                    break
                if usable != cursor:
                    _log(f"replay: ep {episode} frames {cursor}..{usable - 1} are masked in the "
                         "recording — skipping them.")
                    with replay_lock:
                        if replay_ctl["episode"] == episode and replay_ctl["frame"] == cursor:
                            replay_ctl["frame"] = usable
                    continue  # re-read the cursor under the lock next iteration
                if not busy.acquire(blocking=False):
                    time.sleep(0.02)  # a manual action holds the mutex; wait it out
                    continue
                try:
                    synced = _resync_anchor_from_robot()
                    out = _infer()
                    if out.get("replay") is None:
                        ctrl_status.value = "replay stopped (source switched to model)"
                        break
                    _solve_and_cache(out, is_preview=True)
                    _note_replay_chunk(out)
                    _drip(out)  # advances the replay cursor by the steps ACTUALLY executed
                    _solve_and_cache(out, is_preview=False)
                    n_chunks += 1
                    ctrl_status.value = (
                        f"replaying ep {episode}: {n_chunks} chunks"
                        + ("" if synced else " (OPEN-LOOP)"))
                finally:
                    busy.release()
        except Exception as e:  # noqa: BLE001
            import traceback
            _log("replay sequential error:\n" + traceback.format_exc())
            ctrl_status.value = f"replay error: {e!r}"
        finally:
            with replay_lock:
                if replay_ctl["auto_gen"] == my_gen:
                    replay_ctl["auto"] = False
            auto_running.clear()
            # Stop recording but KEEP the samples: the operator saves the sheet AFTER the play.
            trace.disarm()
            _request_trace_redraw()
            # A playthrough is one execution unit and it has just ended, so commit its take
            # here rather than leaving it open for the next unrelated action to inherit.
            _record_finish(f"play ep{episode}")
            if replay_stop_btn is not None:
                replay_stop_btn.disabled = True

    def _replay_play_open_loop():
        """Play the selected episode OPEN-LOOP: decode the whole demo ONCE, drip it straight.

        The one-shot alternative to :func:`_replay_play_sequential`. It seeds the decode anchor
        to the arm ONCE (a single clutch), decodes the ENTIRE episode as one anchor-relative
        :class:`flore.s1_policy.serving.hl_replay.EpisodePlan`, and drips every step without ever
        re-anchoring. The operator's ask: "compute the whole trajectory in the robot frame at the
        start, then replay it in one go, and read the mean error" — that error is then a true
        open-loop tracking number, not one measured against a reference redrawn from the arm's own
        lagging pose every chunk.

        Same safety envelope as the per-chunk play: one arm, ``busy`` held for the whole drip,
        ``stop``/Stop interrupts between steps, and the SAME ``decode_chunk``/wire/gate path.
        Only the execution MODEL differs (one placement vs a receding clutch). At the end it
        auto-fills the comparison sheet's numbers (mean/RMS/max error) so the operator sees the
        open-loop result without pressing Save; the PNG is still only written on demand."""
        if replay is None:
            return
        with replay_lock:
            replay_ctl["auto"] = True
            my_gen = replay_ctl["auto_gen"]
            episode = int(replay_ctl["episode"])
            start_frame = int(replay_ctl["frame"])
        auto_running.set()
        if replay_stop_btn is not None:
            replay_stop_btn.disabled = False
        try:
            # Seed the demo against the arm ONCE: re-sync the anchor to the measured EE (RECORD=1)
            # or keep the open-loop home, THEN capture the commanded reference against that exact
            # anchor, THEN decode the whole plan against it — one anchor for all three, so the
            # green line, the command, and the saved sheet are the same single-placement curve.
            if not busy.acquire(blocking=False):
                ctrl_status.value = "busy — a chunk is predicting/running"
                return
            try:
                synced = _resync_anchor_from_robot()
                usable = replay.next_usable_frame(episode, start_frame)
                if usable is None:
                    ctrl_status.value = f"ep {episode}: no playable frame to open-loop from"
                    return
                if usable != start_frame:
                    with replay_lock:
                        if replay_ctl["episode"] == episode and replay_ctl["frame"] == start_frame:
                            replay_ctl["frame"] = usable
                    start_frame = usable
                _arm_trace(episode, start_frame,
                           f"OPEN-LOOP ep{episode} from frame {start_frame}")
                _record_arm(f"OPEN-LOOP ep{episode} from frame {start_frame}")
                out = _infer_replay_openloop(episode, start_frame)
                _note_replay_chunk(out)
                _solve_and_cache(out, is_preview=True)
                info = out.get("replay_openloop", {})
                n_cmd = int(info.get("n_frames", out["wire_pos"].shape[0]))
                trunc = bool(info.get("truncated", False))
                _log(f"OPEN-LOOP replay: ep {episode} from frame {start_frame} — {n_cmd} steps "
                     f"({n_cmd / max(args.step_hz, 1.0):.1f} s)"
                     + (" [TRUNCATED at a glitch hole]" if trunc else "")
                     + ("" if synced else " (OPEN-LOOP anchor — RECORD=1 off)"))
                ctrl_status.value = (
                    f"OPEN-LOOP ep {episode}: dripping {n_cmd} steps"
                    + (" [truncated]" if trunc else ""))
            finally:
                busy.release()
            # Drip OUTSIDE `busy`: the whole episode is one long chunk (up to ~20 s), and holding
            # the mutex that long would block the idle heartbeat's measured-pose refresh and every
            # GUI action. `stop`/Stop still interrupts, and no other chunk can start because
            # `auto_running`/`dripping` are set (Predict/Run/Play all refuse while they are).
            n_done = _drip_openloop(out)
            _solve_and_cache(out, is_preview=False)
            done_lbl = ("stopped" if stop.is_set() or n_done < n_cmd else "finished")
            ctrl_status.value = f"OPEN-LOOP ep {episode} {done_lbl} — {n_done}/{n_cmd} steps"
            _summarize_trace_error(prefix=f"OPEN-LOOP ep {episode}")
        except Exception as e:  # noqa: BLE001
            import traceback
            _log("open-loop replay error:\n" + traceback.format_exc())
            ctrl_status.value = f"open-loop error: {e!r}"
        finally:
            with replay_lock:
                if replay_ctl["auto_gen"] == my_gen:
                    replay_ctl["auto"] = False
            auto_running.clear()
            trace.disarm()  # keep the samples so the operator can still save the PNG
            _request_trace_redraw()
            _record_finish(f"OPEN-LOOP ep{episode}")
            if replay_stop_btn is not None:
                replay_stop_btn.disabled = True

    def _run_to_completion():
        """SEQUENTIAL MODEL ROLLOUT — the policy analogue of the replay Play button.

        WHY THIS IS THE RIGHT SHAPE FOR ACT, not merely a convenience: upstream ACT's
        evaluation IS one continuous loop (``imitate_episodes.py:225`` —
        ``for t in range(max_timesteps)``), and with ``--temporal_agg`` the ensembler's buffer
        must PERSIST across that whole loop. Chunk-at-a-time clicking cannot express that: each
        press re-enters :func:`_drip_ensembled`, and the blend only reaches its full 50
        survivors after a horizon of *uninterrupted* steps. So the manual Run path structurally
        under-uses ensembling, while this loop is the cadence the feature was designed for.

        WHAT IT DOES NOT CHANGE — the safety envelope. Every iteration goes through the SAME
        path as a manual press: the same clutch re-anchor, the same LIVE force gate re-read per
        step, the same anchor commit, the same ``busy`` mutex, the same ``stop`` checks. It only
        removes the click. It is bounded THREE ways, because a policy has no end of its own to
        reach the way a recorded episode does:
          1. ``Stop`` (bumps the generation — an in-flight chunk finishes, then the loop exits),
          2. a wall-clock ceiling (``--auto-max-seconds``, default 20 s),
          3. the process-wide ``stop`` event.

        With ensembling ON this runs ONE continuous ensembled stretch (the buffer is never
        reset mid-rollout, so survivors ramp to the horizon and stay there — exactly upstream's
        behavior). With it OFF it repeats predict→drip per chunk, which is the flow/diffusion
        cadence and directly comparable to them."""
        if args.offline:
            ctrl_status.value = "OFFLINE — execute disabled (no low level). Viz only."
            return
        if replay_only:
            ctrl_status.value = "no policy loaded — use the replay panel's Play instead"
            return
        if auto_running is not None and auto_running.is_set():
            ctrl_status.value = "ignored — a sequential run is already active; press Stop"
            return
        if dripping.is_set():
            ctrl_status.value = "ignored — a chunk is executing"
            return
        # A recorded-replay source has its own Play button with episode-end semantics; running
        # it through here would silently ignore the cursor/end-of-episode handling.
        if replay is not None:
            with replay_lock:
                if replay_ctl["source"] == "replay":
                    ctrl_status.value = ("chunk SOURCE is 'recorded replay' — use ⏩ Play "
                                         "episode, or switch SOURCE to 'model'")
                    return
        with auto_model_lock:
            auto_model_ctl["running"] = True
            my_gen = int(auto_model_ctl["gen"])
            max_s = float(auto_model_ctl["max_s"])
        auto_running.set()
        if auto_stop_btn is not None:
            auto_stop_btn.disabled = False
        with ens_lock:
            use_ens = bool(ens_ctl["enabled"] and ens_ctl["supported"])
        # ONE trace for the whole rollout, armed before the first motion so t=0 is the moment
        # the arm starts. No replay episode/frame, so no commanded reference to compare against.
        _arm_trace(None, None, "sequential model rollout")
        # A sequential rollout is ONE take from arm to stop, so "直接按 sequential run" needs no
        # separate recording gesture: every chunk it predicts and every step it executes lands
        # in the same take, committed when the loop ends.
        _record_arm("sequential model rollout")
        # A fresh rollout is a fresh INTENT: drop any survivors left from earlier presses so the
        # blend does not open by averaging predictions made for a trajectory we have abandoned.
        planner.reset_temporal_ensemble()
        _reach_reset("new sequential rollout")
        n_chunks = n_steps_total = 0
        _seq_openloop_warn = {"t": 0.0}  # rate-limit the OPEN-LOOP line; see the else-branch
        t0 = time.time()
        try:
            while not stop.is_set():
                with auto_model_lock:
                    if auto_model_ctl["gen"] != my_gen or not auto_model_ctl["running"]:
                        ctrl_status.value = f"sequential run stopped ({n_steps_total} steps)"
                        break
                elapsed = time.time() - t0
                if elapsed >= max_s:
                    ctrl_status.value = (f"sequential run finished — {max_s:g} s ceiling "
                                         f"({n_chunks} chunks, {n_steps_total} steps)")
                    break
                if not busy.acquire(blocking=False):
                    time.sleep(0.02)  # a manual action holds the mutex; wait it out
                    continue
                try:
                    n_exec = max(1, min(int(exec_ctl["n"]), ACTION_HORIZON))
                    # A stale camera/F/T is worse here than in a single Predict: nobody is
                    # watching between chunks, so the loop would keep planning off a frozen
                    # scene until the ceiling. Checked for BOTH branches (the ensembled one
                    # re-queries every step, which makes a frozen frame worse, not better).
                    # `break` inside the try runs the `finally: busy.release()` below — do NOT
                    # release here as well.
                    _stale_seq, _ = _obs_freshness(
                        planner, max_age_s=planner.obs_max_age_s,
                        grip_max_age_s=planner.grip_max_age_s,
                        consumes_force=_consumes_force)
                    if _stale_seq and _refuse_stale_obs:
                        _msg = "sequential STOPPED — stale observation: " + "; ".join(_stale_seq)
                        _log(_msg)
                        ctrl_status.value = _msg
                        break
                    if use_ens:
                        # Do NOT reset the ensembler between iterations: this is one continuous
                        # ensembled stretch, which is the whole point (see the docstring).
                        n_done = _drip_ensembled(n_exec)
                        with render_lock:
                            latest["n_chunks"] += 1
                            latest["dirty"] = True
                    else:
                        # The bool WAS discarded here while _predict logs it loudly. Same
                        # failure, same severity: without a fresh RecordTap the chunk decodes
                        # off the last COMMANDED anchor, so errors compound chunk over chunk —
                        # and the LOW-LEVEL-LOST watchdog cannot catch it, because it only
                        # fires on a falling edge and a session that never had RECORD=1 was
                        # never live. Rate-limited so a whole open-loop run does not spam.
                        # (offline/replay_only already returned at the top of this function)
                        if not _resync_anchor_from_robot():
                            _now = time.monotonic()
                            if _now - _seq_openloop_warn["t"] > 5.0:
                                _seq_openloop_warn["t"] = _now
                                _log("SEQUENTIAL: no fresh RecordTap EE (RECORD=1?) — chunks "
                                     "decode OPEN-LOOP from the last commanded anchor.")
                        out = _infer()
                        _solve_and_cache(out, is_preview=True)
                        n_done = _drip(out)
                        _solve_and_cache(out, is_preview=False)
                    n_chunks += 1
                    n_steps_total += int(n_done)
                    with ens_lock:
                        n_bl = int(ens_ctl["n_blended"])
                    ctrl_status.value = (
                        f"sequential {'ENSEMBLED ' if use_ens else ''}run: {n_chunks} chunks, "
                        f"{n_steps_total} steps, {elapsed:.1f}/{max_s:g} s"
                        + (f" · {n_bl} blended" if use_ens else ""))
                    if n_done == 0:  # stop fired inside the drip — do not spin
                        break
                finally:
                    busy.release()
        except Exception as e:  # noqa: BLE001
            import traceback
            _log("sequential model run error:\n" + traceback.format_exc())
            ctrl_status.value = f"sequential run error: {e!r}"
        finally:
            with auto_model_lock:
                if auto_model_ctl["gen"] == my_gen:
                    auto_model_ctl["running"] = False
            auto_running.clear()
            trace.disarm()  # keep the samples; the operator saves the sheet after the run
            _request_trace_redraw()
            _record_finish("sequential model rollout")
            if auto_stop_btn is not None:
                auto_stop_btn.disabled = True
            _log(f"sequential rollout ended: {n_chunks} chunks, {n_steps_total} steps, "
                 f"{time.time() - t0:.1f} s (ensembled={use_ens})")

    # `--obs-max-age 0` = "I know a bridge is flaky, let me drive anyway": the refusal is off,
    # the page readout and the gripper validity flag stay on (they cost nothing and are the
    # only way to notice). Derived from args here rather than threaded through _run_viser's
    # already-long signature.
    _refuse_stale_obs = not args.obs_max_age or float(args.obs_max_age) > 0

    def _replay_source_active() -> bool:
        """True when the page's Chunk SOURCE is the RECORDED episode, which reads no
        observation at all — so the freshness refusal below must not apply to it."""
        if replay is None or replay_ctl is None or replay_lock is None:
            return False
        with replay_lock:
            return replay_ctl["source"] == "replay"

    def _predict():
        if run_continuous.is_set() or dripping.is_set():
            ctrl_status.value = "ignored — Stop continuous first"
            return
        if auto_running is not None and auto_running.is_set():
            ctrl_status.value = ("ignored — a sequential run is active; press Stop first")
            return
        if not busy.acquire(blocking=False):
            ctrl_status.value = "busy — a chunk is predicting/running"
            return
        # STALE-OBSERVATION REFUSAL. Only for a MODEL chunk on a LIVE rig: --offline opens no
        # sockets by construction, a recorded-replay chunk consumes no observation at all, and
        # a model-free session has no model to feed — refusing in those would break three
        # working paths to guard an input none of them reads.
        if (_refuse_stale_obs and not args.offline and not replay_only
                and not _replay_source_active()):
            _stale, _ = _obs_freshness(planner, max_age_s=planner.obs_max_age_s,
                                       grip_max_age_s=planner.grip_max_age_s,
                                       consumes_force=_consumes_force)
            if _stale:
                busy.release()
                msg = "REFUSED — stale observation: " + "; ".join(_stale)
                ctrl_status.value = msg
                _log(msg + " (fix the bridge; --obs-max-age 0 disables this refusal)")
                return
        try:
            # CLUTCH SEMANTIC: re-sync the decode anchor to the robot's ACTUAL current EE
            # (measured via RecordTap dof_pos → FK) at each Predict, so the model's EE-local
            # delta is applied from where the arm ACTUALLY is — press-start-from-here, never
            # fly to a fixed home, no cross-chunk drift. Falls back to the open-loop home
            # anchor (with a loud warning) only when RECORD=1 is off / the frame is stale.
            synced = _resync_anchor_from_robot()
            # Offline there is no RecordTap by construction, so the OPEN-LOOP warning is expected
            # rather than a fault — and its advice (home the robot) does not apply. The anchor is
            # then the home/ready-pose seed, which is exactly the frame being inspected.
            if not synced and not args.offline:
                _log("OPEN-LOOP: no fresh RecordTap EE (RECORD=1?) — decoding from the "
                     "home/last-committed anchor; HOME THE ROBOT to the ready pose before arming.")
            out = _infer()
            # A Predict that REPLACES an already-staged plan is a retry the operator abandoned.
            # It is counted (not stored) in the open take: "how many times did I re-predict
            # before running this one" is worth knowing, while keeping every discarded chunk
            # would multiply the take by the retry count for plans that never moved the robot.
            if staged["out"] is not None and recorder is not None:
                recorder.note_discarded_predict()
            staged["out"] = out
            _solve_and_cache(out, is_preview=True)
            if not args.offline:
                run_staged_btn.disabled = False  # offline: stays disabled, nothing to execute to
            _note_replay_chunk(out)  # replay panel readout + recorded RGB
            _kind = "recorded chunk" if out.get("replay") is not None else "predicted"
            if args.offline:
                ctrl_status.value = f"{_kind} loaded — preview only (offline, anchor = ready pose)"
            else:
                ctrl_status.value = (f"{_kind} (clutch: from measured EE)" if synced
                                     else f"{_kind} (OPEN-LOOP — RECORD=1 off; home the robot)")
        except Exception as e:  # noqa: BLE001
            import traceback
            _log("predict error:\n" + traceback.format_exc())
            ctrl_status.value = f"predict error: {e!r}"
        finally:
            busy.release()

    def _run_staged():
        # OFFLINE has no wire and no robot. Refuse at the ENTRY of the execute path (not just by
        # greying the button) so the sequential Play loop — which calls this directly — is covered
        # by the same guard, and no code path can drip to a driver that was never started.
        if args.offline:
            ctrl_status.value = "OFFLINE — execute disabled (no low level). Viz only."
            return
        if auto_running is not None and auto_running.is_set():
            ctrl_status.value = ("ignored — a sequential run is active; press Stop first")
            return
        if run_continuous.is_set() or dripping.is_set() or staged["out"] is None:
            ctrl_status.value = "nothing staged — Predict first"
            run_staged_btn.disabled = True
            return
        if not busy.acquire(blocking=False):
            ctrl_status.value = "busy — a chunk is predicting/running"
            return
        out = staged["out"]
        staged["out"] = None
        run_staged_btn.disabled = True
        try:
            # A MANUAL Run records too, but only ARMS a fresh trace when one is not already
            # running — so pressing Run repeatedly builds ONE continuous executed path across
            # the chunks (which is what the operator is comparing), instead of resetting the red
            # line and the reference on every press.
            if not trace.recording:
                rp0 = out.get("replay")
                _arm_trace(rp0["episode"] if rp0 else None, rp0["frame"] if rp0 else None,
                           "manual Run")
                # ONE take across consecutive manual Runs, matching the red trace: pressing Run
                # repeatedly walks one trajectory, and splitting it into a take per press would
                # cut the executed path into pieces that cannot be compared against the plan
                # that produced them. A new take starts when the trace does (Reset / Clear /
                # Play / sequential).
                _record_arm("manual Run")
            # CADENCE CHOICE, made once here (never mid-drip). Temporal ensembling re-queries
            # the POLICY every step, so it applies only to a model chunk: a RECORDED replay has
            # no policy behind it and must keep dripping its stored frames, or the "compare the
            # demo against the model on the same rig" guarantee breaks.
            with ens_lock:
                use_ens = bool(ens_ctl["enabled"] and ens_ctl["supported"])
            use_ens = use_ens and out.get("replay") is None and _drip_ensembled is not None
            if use_ens:
                n_done = _drip_ensembled(max(1, min(int(exec_ctl["n"]), ACTION_HORIZON)))
                with ens_lock:
                    n_bl = int(ens_ctl["n_blended"])
                # No _solve_and_cache: there is no single staged chunk to re-render — each step
                # came from its own inference. The RED executed trace (render thread, live
                # wrist FK) is the faithful record of what actually ran, and the chunk counter
                # is bumped so the readout still reflects that an execution happened.
                with render_lock:
                    latest["n_chunks"] += 1
                    latest["dirty"] = True
                ctrl_status.value = (f"executed {n_done} step(s) ENSEMBLED "
                                     f"({n_bl} predictions blended at the last step)")
            else:
                _drip(out)
                _solve_and_cache(out, is_preview=False)  # re-render as executed (counter++)
                if out.get("replay") is not None and replay is not None:
                    # The cursor advanced inside _drip; show where the NEXT chunk will start.
                    with replay_lock:
                        nxt = int(replay_ctl["frame"])
                        epi = int(replay_ctl["episode"])
                    if replay_frame_slider is not None:
                        replay_frame_slider.value = min(nxt, int(replay_frame_slider.max))
                    if replay_status is not None:
                        replay_status.value = (
                            f"executed — next: ep {epi} frame {nxt}/{_replay_episode_len(epi) - 1}")
                ctrl_status.value = "staged chunk executed"
        except Exception as e:  # noqa: BLE001
            import traceback
            _log("run_staged error:\n" + traceback.format_exc())
            ctrl_status.value = f"run error: {e!r}"
        finally:
            busy.release()

    # NOTE: continuous mode is intentionally NOT offered. Per the deploy spec the planner
    # runs one predict → preview → execute chunk at a time (safer on the gantry); there is
    # no "run continuously" button and no autonomous re-plan loop. ``run_continuous`` remains
    # a permanently-clear Event so main()'s ingest/teardown guards stay valid.

    @grip_over_slider.on_update
    def _(_e) -> None:
        with grip_lock:
            grip_ctl["overshoot_m"] = float(grip_over_slider.value) / 1000.0
        _mm = float(grip_over_slider.value)
        grip_sq_status.value = (
            f"overshoot {_mm:.1f} mm on close" if _mm > 0 else
            "overshoot OFF — the jaw stops at the predicted width (touch, not grip)")

    @grip_below_slider.on_update
    def _(_e) -> None:
        with grip_lock:
            grip_ctl["close_below_m"] = float(grip_below_slider.value) / 1000.0


    @grip_confirm_slider.on_update
    def _(_e) -> None:
        with grip_lock:
            grip_ctl["confirm_steps"] = int(grip_confirm_slider.value)
        n = int(grip_confirm_slider.value)
        grip_sq_status.value = (
            f"needs {n} consecutive close/open step(s) to switch the overshoot" if n > 1
            else "reacting to EVERY step (no confirmation — a flicker can loosen the grasp)")

    @grip_hold_toggle.on_update
    def _(_e) -> None:
        want = bool(grip_hold_toggle.value)
        with grip_lock:
            grip_ctl["hold_tightest"] = want
            if not want:
                grip_ctl["held_frac"] = None
        grip_sq_status.value = (
            "holding the tightest target of the grasp" if want else
            "following the width verbatim — an ensembled blend may loosen the grip")

    @auto_secs_slider.on_update
    def _(_e) -> None:
        # Read by the loop each iteration, so shortening it mid-run takes effect immediately
        # (it can only ever stop the arm SOONER, never extend a running rollout past Stop).
        with auto_model_lock:
            auto_model_ctl["max_s"] = float(auto_secs_slider.value)

    @auto_run_btn.on_click
    def _(_e) -> None:
        # The RUN RECORDER brackets the run as an auto-run INSIDE the recording (its folder opens if
        # the ⏺ button is armed and none is open; the run's end — ⏹, the ceiling, the low level lost,
        # an error — is an EVENT, not the recording's end, which is Reset / the low level disarming /
        # the button); with no recorder installed this is the body unchanged.
        threading.Thread(target=taps.wrap_run(_run_to_completion, "sequential",
                                              end_note=lambda: str(ctrl_status.value)),
                         daemon=True).start()

    @auto_stop_btn.on_click
    def _(_e) -> None:
        # Retire the loop by bumping the generation; the in-flight chunk finishes (bounded by
        # exec-steps) and the loop then exits without re-arming. Same contract as replay Stop.
        with auto_model_lock:
            auto_model_ctl["running"] = False
            auto_model_ctl["gen"] += 1
        # A Stop ABANDONS THE INTENT — one of the three documented reset triggers for the
        # temporal ensembler ("a fresh episode, a Reset-start-pose, a Stop", see
        # TemporalEnsembler.reset). Without this, Predict → ▶ Run staged after a Stop opens by
        # blending the ~49 surviving predictions of the STOPPED rollout against ONE fresh query;
        # the weights are exp(-k*i) OLDEST-first, so the fresh query carries ~1.5% of the vote
        # and the arm resumes the trajectory the operator just stopped. (▶▶ Run to completion
        # already resets at its own entry, which is why only the manual path showed this.)
        request_ens_reset(ens_ctl, ens_lock)
        _reach_reset("sequential stop")
        ctrl_status.value = "sequential stop requested"

    @predict_btn.on_click
    def _(_e) -> None:
        threading.Thread(target=_predict, daemon=True).start()

    @run_staged_btn.on_click
    def _(_e) -> None:
        threading.Thread(target=_run_staged, daemon=True).start()

    # --- Replay panel callbacks (flag-only; the worker threads do the work) ---
    if replay is not None:
        @replay_src_toggle.on_update
        def _(_e) -> None:
            want = "replay" if replay_src_toggle.value == "recorded replay" else "model"
            with replay_lock:
                replay_ctl["source"] = want
                if want == "model":  # leaving replay retires any sequential loop
                    replay_ctl["auto"] = False
                    replay_ctl["auto_gen"] += 1
            staged["out"] = None  # a staged chunk from the OTHER source must not be run
            run_staged_btn.disabled = True
            ctrl_status.value = f"chunk source = {want} (Predict again)"
            if replay_status is not None and want == "model":
                replay_status.value = "source = model (recorded replay idle)"

        @replay_ep_dd.on_update
        def _(_e) -> None:
            epi = _ep_options.index(replay_ep_dd.value)
            n = _replay_episode_len(epi)
            first = _ep_first[epi]
            # Retarget the frame slider to the new episode BEFORE seeking, so a stale value
            # past the new episode's end can never be read as a cursor.
            replay_frame_slider.max = max(n - 1, 1)
            replay_frame_slider.value = int(first or 0)  # fires the seek callback below
            _replay_seek(epi, int(first or 0))
            staged["out"] = None
            run_staged_btn.disabled = True
            if replay_status is not None:
                if first is None:
                    replay_status.value = (
                        f"ep {epi}: NO playable frame (every frame's step 0 is masked in the "
                        "recording) — pick another episode")
                elif first != 0:
                    replay_status.value = (
                        f"ep {epi} selected ({n} frames) — frames 0..{first - 1} are masked in "
                        f"the recording, so the cursor starts at {first}; Predict to preview")
                else:
                    replay_status.value = f"ep {epi} selected ({n} frames) — Predict to preview"
            _request_replay_path(epi, int(first or 0))

        @replay_frame_slider.on_update
        def _(_e) -> None:
            with replay_lock:
                epi = int(replay_ctl["episode"])
            fr = int(np.clip(int(replay_frame_slider.value), 0, _replay_episode_len(epi) - 1))
            # A manual scrub can land on a frame the converter masked; snap FORWARD to the
            # next playable one so Predict cannot fail on an operator's slider drag.
            usable = replay.next_usable_frame(epi, fr)
            if usable is not None and usable != fr and replay_status is not None:
                replay_status.value = (
                    f"frame {fr} is masked in the recording — snapped to {usable}")
            fr = fr if usable is None else usable
            _replay_seek(epi, fr)
            staged["out"] = None
            run_staged_btn.disabled = True
            _request_replay_path(epi, fr)

        @replay_play_btn.on_click
        def _(_e) -> None:
            if args.offline:
                ctrl_status.value = "OFFLINE — nothing to play to. Use 'Load 1 recorded chunk'."
                return
            if auto_running.is_set():
                ctrl_status.value = "already replaying — press Stop first"
                return
            if dripping.is_set():
                ctrl_status.value = "ignored — a chunk is executing"
                return
            with replay_lock:
                if replay_ctl["source"] != "replay":
                    ctrl_status.value = "switch Chunk SOURCE to 'recorded replay' first"
                    return
            threading.Thread(target=taps.wrap_run(_replay_play_sequential, "replay",
                                                  end_note=lambda: str(ctrl_status.value)),
                             daemon=True).start()

        @replay_play_ol_btn.on_click
        def _(_e) -> None:
            if args.offline:
                ctrl_status.value = "OFFLINE — nothing to play to. Use 'Load 1 recorded chunk'."
                return
            if auto_running.is_set():
                ctrl_status.value = "already replaying — press Stop first"
                return
            if dripping.is_set():
                ctrl_status.value = "ignored — a chunk is executing"
                return
            with replay_lock:
                if replay_ctl["source"] != "replay":
                    ctrl_status.value = "switch Chunk SOURCE to 'recorded replay' first"
                    return
            threading.Thread(target=taps.wrap_run(_replay_play_open_loop, "replay-openloop",
                                                  end_note=lambda: str(ctrl_status.value)),
                             daemon=True).start()

        @replay_stop_btn.on_click
        def _(_e) -> None:
            # Retire the loop by bumping the generation; the in-flight chunk finishes its
            # current drip (bounded by exec-steps) and the loop then exits without re-arming.
            with replay_lock:
                replay_ctl["auto"] = False
                replay_ctl["auto_gen"] += 1
            ctrl_status.value = "replay stop requested"

        @replay_rewind_btn.on_click
        def _(_e) -> None:
            if auto_running.is_set():
                ctrl_status.value = "ignored — press Stop before rewinding"
                return
            with replay_lock:
                epi = int(replay_ctl["episode"])
            # Rewind to the episode's first PLAYABLE frame, not a hardcoded 0 — otherwise
            # rewinding one of the masked-start episodes re-arms the frame that fails Predict.
            first = int(_ep_first[epi] or 0)
            replay_frame_slider.value = first
            _replay_seek(epi, first)
            staged["out"] = None
            run_staged_btn.disabled = True
            if replay_status is not None:
                replay_status.value = f"ep {epi} rewound to frame {first}"
            # Draw the path against the frame we actually seeked to, NOT a hardcoded 0: the
            # green path is transported relative to its reference frame, so referencing 0
            # while the cursor sits at ``first`` renders the demo displaced from the arm by
            # |p[0] − p[first]| — up to 158 mm on the 12 clean94 episodes whose first
            # playable frame is masked (ep1 156 mm, ep27 158 mm, ep31 140 mm). That is
            # exactly the datum check this view exists to support, so it must not lie.
            _request_replay_path(epi, first)

        @replay_show_path.on_update
        def _(_e) -> None:
            with replay_lock:
                epi = int(replay_ctl["episode"])
                fr = int(replay_ctl["frame"])
            _request_replay_path(epi, fr)

    @reset_btn.on_click
    def _(_e) -> None:
        if run_continuous.is_set() or dripping.is_set():
            ctrl_status.value = "ignored — Stop first"
            return
        # Take `busy` (non-blocking) so Reset is atomic w.r.t. an in-flight Predict/Run: those
        # also hold `busy` and write staged['out'] + the decode anchor + mapper.hold_*, so
        # without this a Predict finishing AFTER Reset would re-stage a chunk decoded against
        # the pre-Reset anchor and re-enable Run — driving the arm along a plan the operator
        # asked to discard. If a chunk is predicting/running, refuse (mirrors _predict).
        if not busy.acquire(blocking=False):
            ctrl_status.value = "busy — a chunk is predicting/running"
            return
        try:
            # Sync the anchor to the robot's ACTUAL current EE first (if RECORD=1), THEN latch
            # the episode-start rotation there — so wrt_start is measured from the real pose.
            synced = _resync_anchor_from_robot()
            planner.reset_start_pose()
            # THE INTENT IS ABANDONED, so drop every stored prediction: Reset re-latches the
            # episode-start pose, and blending the new motion against chunks predicted for the
            # old one would drag the arm back toward a trajectory the operator just discarded.
            # (This is the documented reset trigger — NOT the ordinary per-step re-anchor,
            # which the absolute-space blend tolerates by design.)
            planner.reset_temporal_ensemble()
            # ...and the reach residual, for the same reason: it is un-executed motion belonging
            # to the trajectory just discarded, so applying it to the next attempt would offset
            # that attempt's very first target.
            _reach_reset("reset start pose")
            staged["out"] = None
            run_staged_btn.disabled = True
            jaw = _release_gripper()
            # Reset ENDS the attempt (it re-latches the episode-start pose), so commit any take
            # the manual Runs left open — otherwise the next attempt's rows would land in the
            # same file and the take would describe two different trajectories.
            _rec_path = _record_finish("reset start pose")
            _rollout_reset()  # the ⏺ run recorder: blue/idle, the next armed ▶▶ opens a NEW folder
            ctrl_status.value = (("reset start pose @ measured EE (tare in free space first!)" if synced
                                  else "reset start pose @ home anchor (RECORD=1 off; home the robot)")
                                 + jaw
                                 + (f"  · take {os.path.basename(_rec_path)}" if _rec_path else ""))
        finally:
            busy.release()

    @tare_btn.on_click
    def _(_e) -> None:
        if run_continuous.is_set() or dripping.is_set():
            tare_status.value = "REFUSED — stop + free space first"
            return
        subs["force"].send({"cmd": "tare"})
        tare_status.value = "tared (free space)"

    @grip_toggle.on_update
    def _(_e) -> None:
        if grip_toggle.value:
            grip_enabled.set()
        else:
            grip_enabled.clear()

    @show_trace.on_update
    def _(_e) -> None:
        _request_trace_redraw()  # the render thread adds/removes the line

    @trace_clear_btn.on_click
    def _(_e) -> None:
        # Re-arm at the CURRENT cursor rather than just emptying the log, so the reference the
        # next save compares against matches what will now be recorded. Refused mid-motion:
        # dropping the reference under a running drip would compare the rest of the play
        # against the wrong curve.
        if dripping.is_set() or (auto_running is not None and auto_running.is_set()):
            trace_readout.value = "refused — a chunk is executing (Stop first)"
            return
        epi = fr = None
        if replay is not None:
            with replay_lock:
                epi = int(replay_ctl["episode"])
                fr = int(replay_ctl["frame"])
        _arm_trace(epi, fr, "cleared")
        trace.disarm()  # armed only to reset the clock/reference; recording resumes on Run
        # Clearing the executed path ends the attempt it belonged to, so commit its take too —
        # the trace and the take describe the same motion and must start/stop together.
        _rec_path = _record_finish("executed path cleared")
        trace_readout.value = ("cleared — the next Run/Play records from here"
                               + (f"  · take {os.path.basename(_rec_path)}" if _rec_path else ""))
        save_plot_status.value = "(nothing saved)"

    def _save_trace_plot() -> None:
        """Render the comparison sheet. Runs on a worker thread: matplotlib's import + savefig
        is ~1 s, and blocking a viser callback thread that long makes the whole page feel hung."""
        try:
            snap = trace.snapshot()
            ref = trace_ref["ref"]
            if len(snap) < 2:
                save_plot_status.value = ("nothing recorded — press ▶ Run or ⏩ Play first "
                                          "(the trace records while the arm moves)")
                return
            if ref is None:
                save_plot_status.value = ("no recorded reference for this trace — the sheet "
                                          "compares against the demo, so replay an episode")
                return
            epi = -1 if trace.episode is None else int(trace.episode)
            # Wall-clock stamp so two saves of the SAME episode never overwrite each other. Used
            # for the FILENAME only — the trace's own clock stays monotonic (a wall-clock jump
            # mid-play would bend the time axis).
            stamp = time.strftime("%Y%m%d_%H%M%S")
            base = os.path.splitext(os.path.basename(replay.path))[0] if replay is not None else "policy"
            base = base.replace(".zarr", "")
            name = f"trace_{base}_ep{epi:03d}_{stamp}.png"
            save_plot_status.value = "rendering…"
            path, st = plot_trace_vs_reference(
                snap, ref, os.path.join(args.trace_dir, name),
                title=(f"{base}  ep {epi}  —  solid = recorded data, dashed = executed"
                       + (f"  [{trace.label}]" if trace.label else "")))
            dropped = trace.dropped
            save_plot_status.value = path + (f"   ⚠ {dropped} early samples dropped (trace full)"
                                             if dropped else "")
            _log(f"saved comparison plot: {path}  ({st.one_line()})")
        except Exception as e:  # noqa: BLE001 — a plotting failure must never disturb serving
            import traceback
            _log("save comparison plot error:\n" + traceback.format_exc())
            save_plot_status.value = f"save failed: {e!r}"

    def _summarize_trace_error(*, prefix: str = "") -> None:
        """Compute executed-vs-recorded error from the current trace + reference and SHOW it.

        The numeric half of the comparison sheet, without writing a PNG: an open-loop play ends
        by calling this so the operator reads the mean/RMS/max tracking error immediately (the
        whole point of the one-shot replay). Uses the SAME ``resample_reference_to_trace`` +
        ``trace_stats`` the saved sheet uses, so the readout and a later Save agree. Best-effort
        and viz-only — a missing reference or an empty trace just leaves a short note."""
        try:
            snap = trace.snapshot()
            ref = trace_ref["ref"]
            if len(snap) < 2 or ref is None:
                trace_readout.value = (f"{prefix}: no error (need a recorded reference + motion)"
                                       if prefix else "(no samples yet)")
                return
            ref_pos, ref_rot = resample_reference_to_trace(ref, snap.t)
            st = trace_stats(snap, ref_pos, ref_rot)
            trace_readout.value = (f"{prefix}  " if prefix else "") + st.one_line()
            _log(f"{prefix + ' ' if prefix else ''}executed-vs-recorded: {st.one_line()}")
        except Exception as e:  # noqa: BLE001 — viz only; never disturb serving/teardown
            _log(f"trace error summary failed (ignored): {e!r}")

    @save_plot_btn.on_click
    def _(_e) -> None:
        threading.Thread(target=_save_trace_plot, daemon=True).start()

    @rec_toggle.on_update
    def _(_e) -> None:
        if record_enabled is None:
            return
        want = bool(rec_toggle.value)
        if want:
            record_enabled.set()
            rec_status.value = "armed — the next execution will be recorded"
            _log(f"deploy RECORDING armed → {recorder.session_dir}")
        else:
            record_enabled.clear()
            # Turning it OFF commits whatever is open rather than discarding it: the operator
            # has already run the arm, and throwing that away because they un-ticked a box is
            # the one irreversible thing this control could do.
            path = _record_finish("recording disarmed")
            rec_status.value = ("disarmed — " + (f"wrote {os.path.basename(path)}" if path
                                                 else "nothing was open"))

    @reach_toggle.on_update
    def _(_e) -> None:
        want = bool(reach_toggle.value)
        with reach_lock:
            reach_ctl["enabled"] = want
        # Turning it OFF must also DROP the residual, not merely stop applying it: a stored
        # residual re-applied on a later re-tick would be a stale correction for a shortfall
        # that is minutes old. Turning it ON starts from zero for the same reason.
        _reach_reset("compensation toggled")
        if want:
            _log(f"reach compensation ON (ki={reach_comp.config.ki:g}, clamp "
                 f"{reach_comp.config.clamp_m * 100:.0f} cm, accumulate only while the arm did "
                 f"<{reach_comp.config.stall_frac:g} of the commanded motion over "
                 f"{int(reach_comp.config.window)} steps, contact cut-out "
                 f"{reach_comp.config.contact_n:g} N) — the published target now carries the "
                 f"un-executed motion. The model's obs and decode anchor are unchanged.")
        else:
            _log("reach compensation OFF — the published target is the model's own again.")

    # --- status thread: ONLINE/OFFLINE + live F/T readout + RGB obs + robot-mesh source ---
    # LOW-LEVEL-LOST WATCHDOG state. ``was_live`` is the previous tick's RecordTap freshness, so
    # the falling edge can be detected; it starts False so a planner launched before the low level
    # does not fire on its first tick.
    _ll_watch = {"was_live": False}

    def _status_loop():
        while not stop.is_set():
            try:
                _status_tick()
            except Exception as e:  # noqa: BLE001 — viz-only; log + retry, never kill the thread
                _log(f"status-loop tick error (ignored): {e!r}")
            time.sleep(0.5)

    def _status_tick():
        online = s0.num_clients > 0
        # RecordTap presence drives both the mesh source label and the F_meas readout.
        rmsg, rw, _rn = subs["record"].latest()
        rframe = parse_record_frame(rmsg)
        r_age = (time.time() - rw) if rw else -1.0
        r_fresh = rframe is not None and 0 <= r_age <= 1.0

        # --- LOW LEVEL LOST ⇒ DROP THE ABANDONED PLAN -------------------------------------
        # The RecordTap frame is published at the END of the rl_inference tick, which only runs
        # while `use_policy_action` is True (holosoma_wbt.py says so where it publishes: the
        # true→false STOP edge "shows up to the recorder as the frame stream simply stopping").
        # So joystick B never reports `vive_armed=False` — the stream just STOPS, and the ONLY
        # observable a planner has for "the operator stopped the controller" is this readback
        # going stale. Treat that falling edge as the intent being abandoned, because it is: the
        # low level dropped to stiff hold and cleared its own slew + clutch anchors
        # (_handle_stop_policy), and will next be re-armed from a pose we never planned from.
        #
        # Three things must be dropped or the arm resumes the abandoned motion on the next Run:
        #   * the TEMPORAL ENSEMBLER — up to H=50 stored predictions whose per-entry cursors
        #     advance per add, NOT per wall-clock second. With exp(-k*i) OLDEST-first weights one
        #     fresh query blended against 49 survivors is ~1.5% of the vote, so the first step
        #     after a re-arm is ~98% the trajectory that was just aborted. This is the "it keeps
        #     following the old motion" symptom. Requested, not applied here (see
        #     request_ens_reset / take_pending_ens_reset).
        #   * the STAGED chunk — decoded against the pre-stop anchor; and while one is staged the
        #     idle heartbeat deliberately refuses to re-track the measured pose, so the wire the
        #     pre-arm gate is compared against stays frozen at a mid-chunk target.
        #   * any SEQUENTIAL run — it watches only its own generation counter, so it would go on
        #     inferring and dripping into a low level that stopped listening.
        # Best-effort for `staged` (the status thread does not hold `busy`, matching the other
        # cross-thread clear sites): an inference that lands in the ~ms after the clear can
        # re-stage once, and that Predict already prints the loud OPEN-LOOP warning because the
        # RecordTap it re-anchors from is stale. A >1 s transport hiccup trips this too; the cost
        # is re-pressing Predict, which is the safe direction.
        if _ll_watch["was_live"] and not r_fresh and not args.offline:
            if auto_model_lock is not None and auto_model_ctl is not None:
                with auto_model_lock:
                    auto_model_ctl["running"] = False
                    auto_model_ctl["gen"] += 1
            if replay is not None:
                with replay_lock:
                    replay_ctl["auto"] = False
                    replay_ctl["auto_gen"] += 1
            staged["out"] = None
            run_staged_btn.disabled = True
            request_ens_reset(ens_ctl, ens_lock)
            # The arm is in stiff hold and disarmed; whatever it failed to execute is now void.
            # Keeping the residual would apply it to the first target after a re-arm.
            _reach_reset("low level lost")
            # The ⏺ run recorder does NOT end here (its ends are the operator's: Reset / the button /
            # Ctrl-C — and `start`, which this edge is not); the loss is an event in its folder.
            taps.event("low_level_lost", note="RecordTap :8779 went stale — joystick B / STOP, or the controller died")
            _log("LOW LEVEL LOST (RecordTap :8779 went stale — joystick B / STOP, or the "
                 "controller died): retired any sequential run, dropped the staged chunk and "
                 "requested a temporal-ensemble reset, so re-arming cannot resume the abandoned "
                 "plan. Re-home (Y), START (A), arm (start), then Predict again.")
            ctrl_status.value = "low level stopped — staged plan + ensemble cleared; Predict again"
        _ll_watch["was_live"] = bool(r_fresh)

        if args.offline:
            # Don't churn the banner every tick with a "waiting" message that will never resolve;
            # the folder header already states the mode.
            status_md.content = ("⚫ **VIZ-ONLY (--offline)** — no low level, nothing published. "
                                 "Execute disabled.")
            robot_src.value = "URDF ready pose (offline)"
        else:
            status_md.content = ("🟢 **ONLINE** — low-level connected on :8770"
                                 + (" · RECORD" if r_fresh else "")) if online else \
                "🔴 **OFFLINE** — waiting for the low-level on :8770"
            if not live_mesh_ok:
                robot_src.value = "predicted-IK (URDF/SDK joint mismatch)"
            elif r_fresh:
                robot_src.value = "LIVE joints (RecordTap :8779)"
            else:
                robot_src.value = "predicted-IK (start low level with RECORD=1)"

        # F/T panel: ticked = the EXACT pelvis-frame F_meas the servo consumes (RecordTap
        # f_meas / f_meas_ungated); unticked = the raw sensor-frame wire on :8771.
        if bool(ft_show_meas.value):
            if r_fresh and rframe.f_meas is not None:
                meas = rframe.f_meas  # (2,3) pelvis N (post-gate = servo input)
                ung = rframe.f_meas_ungated if rframe.f_meas_ungated is not None else meas
                lN = float(np.linalg.norm(meas[0]))
                rN = float(np.linalg.norm(meas[1]))
                lNu = float(np.linalg.norm(ung[0]))
                rNu = float(np.linalg.norm(ung[1]))
                ft_live.value = f"F_meas |L|={lN:.2f}  |R|={rN:.2f} N (pelvis, servo input)"
                ft_detail.value = f"ungated |L|={lNu:.2f} |R|={rNu:.2f} N"
            elif online:
                ft_live.value = "F_meas unavailable — start low level with RECORD=1"
                ft_detail.value = "-"
            else:
                ft_live.value = "(waiting for RecordTap :8779…)"
                ft_detail.value = "-"
        else:
            fmsg, fw, _fn = subs["force"].latest()
            f6 = _pick_force_channels(fmsg, _display_force_roles, require_6dof)
            age = (time.time() - fw) if fw else -1.0
            if f6 is not None and age >= 0:
                fN = float(np.linalg.norm(f6[0:3]))
                tN = float(np.linalg.norm(f6[3:6]))
                ft_live.value = (f"STALE {age:.1f}s — |F|={fN:.2f} |T|={tN:.3f}" if age > 1.0
                                 else f"raw sensor |F|={fN:.2f} N  |T|={tN:.3f} Nm (:8771)")
                ft_detail.value = "raw sensor frame (pre low-level rotate/gate)"
            else:
                ft_live.value = "no/invalid F/T frame (bridge down?)"
                ft_detail.value = "-"

        # Ring ages, independent of which F/T view is selected above — the old page showed the
        # `:8771` age ONLY when "Show low-level F_meas" was unticked, and it defaults TICKED, so
        # by default nothing on the page could reveal a frozen sensor or a frozen camera.
        _stale_now, _fresh_txt = _obs_freshness(
            planner, max_age_s=planner.obs_max_age_s, grip_max_age_s=planner.grip_max_age_s,
            consumes_force=_consumes_force)
        obs_fresh.value = (("⚠ STALE — Predict refused · " + _fresh_txt) if _stale_now
                           else _fresh_txt)

        # DEPLOY RECORDING readout: live row counts while a take is open, and the last take
        # written when idle — so an operator can see that recording is actually capturing
        # something (and how much) without tailing stderr.
        if rec_status is not None and recorder is not None:
            if recorder.recording:
                c = recorder.counts()
                rec_status.value = (
                    f"● RECORDING take_{recorder.take_index:03d} — {c['predictions']} plan(s), "
                    f"{c['executed']} measured, {c['commanded']} commanded, {c['camera']} frames")
            elif recorder.last_error:
                rec_status.value = f"⚠ last write FAILED: {recorder.last_error}"
            elif recorder.last_saved:
                rec_status.value = (
                    f"idle — last take {os.path.basename(recorder.last_saved)}"
                    + (" · armed" if record_enabled.is_set() else " · disarmed"))
            else:
                rec_status.value = ("armed — nothing recorded yet" if record_enabled.is_set()
                                    else "idle (not recording)")
        _rollout_refresh()  # the ⏺ run recorder's button + live counts (2 Hz)

        # REACH COMPENSATION readout: the live residual, and SATURATED when it is pinned at the
        # clamp — which is the operator-visible signal that the arm still is not getting there
        # (the compensator is asking for all the extra reach it is allowed and the gap remains).
        if reach_status is not None:
            with reach_lock:
                _on = reach_comp.enabled and reach_ctl["enabled"]
                _res = reach_comp.residual_norm
                _sat = reach_comp.saturated
                _st = reach_comp.stats()
            if not reach_comp.enabled:
                reach_status.value = "OFF (not enabled at launch)"
            elif not _on:
                reach_status.value = "OFF (un-ticked)"
            else:
                _rz = _st["realized"]
                reach_status.value = (
                    f"{_res * 1000:.1f} mm{' ⚠ SATURATED at the clamp' if _sat else ''}"
                    f" · arm did {'—' if _rz is None else f'{_rz * 100:.0f}%'} of the ask"
                    f" · {_st['n_applied']} accumulated / {_st['n_gated']} tracking fine"
                    f" / {_st['n_contact_blocked']} held off by contact")

        # RGB observation: refresh the sidebar panel with the exact frames the model sees.
        # ``img_handles`` is empty in model-free replay (no panel built), so this is a no-op.
        if img_handles:
            try:
                disp = planner.snapshot_display_images()
                for h, im in zip(img_handles, disp):
                    h.image = im
            except Exception:  # noqa: BLE001 — viz-only; never disrupt serving
                pass

    threading.Thread(target=_status_loop, daemon=True, name="status").start()
    if _ghost_ready:
        threading.Thread(target=_ghost_solver_loop, daemon=True, name="ghost-ik").start()

    # Report the port viser ACTUALLY bound, not the one requested: viser silently falls forward
    # when the requested port is taken, and printing the request sends the operator to a stale page
    # (or to another tool) with no hint that anything moved.
    _bound = getattr(server, "get_port", lambda: args.port)()
    if int(_bound) != int(args.port):
        _log(f"NOTE: port {args.port} was busy — viser bound {_bound} instead. "
             f"Use http://{args.host}:{_bound} (something else is on {args.port}).")
    if args.offline:
        print(f"[hl-serving] http://{args.host}:{_bound}  — OFFLINE viz: "
              "Load 1 recorded chunk to preview the demo. Execute is disabled.", flush=True)
    else:
        print(f"[hl-serving] http://{args.host}:{_bound}  — connect the low-level to :8770, then "
              + ("Load 1 recorded chunk → Run this chunk (or ⏩ Play episode)" if replay_only
                 else "Predict 1 chunk → Run this chunk"), flush=True)

    # --- render loop (main thread) ---
    # The MESH prefers the REAL robot joints from RecordTap (:8779, RECORD=1): if a fresh frame
    # is present, the URDF is posed from the low level's live dof_pos (name-remapped) + the live
    # :8784 jaw width. Otherwise it falls back to playing back the predicted chunk's IK (the
    # previous behavior). The YELLOW rollout path, the force charts, the wrench arrow, and the
    # camera-follow frame always come from the predicted chunk.
    play_i = {"i": 0, "S": 0}
    # LIVE-MESH REPOSE cadence — the same argument as the two throttles below, and the most
    # expensive one of the three. `ViserUrdf.update_cfg` is NOT cheap: it runs a full yourdfpy
    # FK and then, for each of the 47 joint frames, a `get_transform` + `SO3.from_matrix` + two
    # scene-graph property writes. Measured on this box: 3.4 ms wall / 3.7 ms CPU per call, so
    # the main mesh alone was 11% of a core at 30 Hz, all of it GIL-held Python — on the very
    # thread whose latency the operator feels. (The recorded takes show this loop achieving
    # 15-17 Hz against its 33.3 ms sleep, i.e. ~20 ms of work per tick.)
    #
    # 15 Hz is a DISPLAY rate for a robot arm and looks the same to a human — and it is already
    # faster than what the loop was actually managing. Nothing else in the mesh block is
    # throttled: the RecordTap parse, `live_tcp`/`live_dof`/`live_armed`/`f_meas`, the jaw
    # readout, the executed trace and the DEPLOY RECORDER all still run every tick, because
    # those are measurements, not pixels. Only the repose is skipped, and skipping it holds the
    # last drawn pose for one extra tick.
    #
    # A memo ("skip when the joints did not move") was measured and REJECTED: across 34061
    # recorded render ticks the per-tick max |Δdof| is 0.0026 rad at the median and under
    # 1e-4 rad for only 5% of ticks — a balancing whole-body controller is never still, so an
    # unchanged-config guard would essentially never fire on the real robot.
    _mesh_draw_period = 1.0 / 15.0
    _mesh_last_draw = {"t": 0.0}
    # Executed-path redraw cadence. The red line is REBUILT (viser line segments are immutable),
    # so doing it every tick would put an O(n) scene rebuild on the same 30 Hz loop whose latency
    # was the whole stutter problem. 4 Hz is smooth to the eye for a growing line.
    _trace_draw_period = 0.25
    _trace_last_draw = {"t": 0.0}
    # Live F_meas chart redraw cadence — same argument, different number: a uPlot .data assignment
    # re-encodes and re-sends the whole window, so it must not ride every tick. 8 Hz looks
    # continuous for a scrolling trace, and the underlying samples are still the socket thread's
    # full-rate interval-mean, so throttling the DRAW loses no signal.
    _ft_draw_period = 0.125
    _ft_last_draw = {"t": 0.0}
    # Draw the initial green path once (replay mode): the operator sees the selected episode
    # before pressing anything. Requested here, applied by _poll_replay_path on this thread.
    if replay is not None:
        with replay_lock:
            _init_ep = int(replay_ctl["episode"])
            _init_fr = int(replay_ctl["frame"])
        _request_replay_path(_init_ep, _init_fr)
    while not stop.is_set():
        # ONE clock per tick, shared by every throttle in this loop (mesh 15 Hz, F/T charts
        # 8 Hz, executed trace 4 Hz) so they cannot disagree about when "now" is inside a tick.
        _now = time.monotonic()
        # Consume any pending green-path request on the RENDER thread (scene-graph owner).
        _poll_replay_path()
        with render_lock:
            dirty = latest["dirty"]
            ghost_cfgs = latest["ghost_cfgs"]
            wpos = latest["wrist_pos"]
            wwxyz = latest["wrist_wxyz"]
            path = latest["path"]
            force_pred = latest["force_pred"]
            grip_m = latest["grip_m"]
            n_chunks = latest["n_chunks"]
            n_steps = latest["n_steps"]
            if dirty:
                latest["dirty"] = False
        # Gated on n_steps, NOT on ghost_cfgs: the ghost is solved on a background thread and is
        # None for the first fraction of a second, but the yellow path / charts / animation cursor
        # must come up the instant a chunk is staged.
        if dirty and n_steps > 0:
            play_i["i"] = 0
            play_i["S"] = n_steps
            if roll_handle["h"] is not None:
                roll_handle["h"].remove()
                roll_handle["h"] = None
            if path is not None and path.shape[0] >= 2:
                seg = np.stack([path[:-1], path[1:]], axis=1).astype(np.float32)
                col = np.broadcast_to(np.array(_ROLLOUT_YELLOW, np.uint8), (seg.shape[0], 2, 3)).copy()
                roll_handle["h"] = server.scene.add_line_segments("/rollout/left", points=seg, colors=col, line_width=3.0)
                roll_handle["h"].visible = bool(show_rollout.value)
            if grip_m is not None and grip_m.shape[0] >= 2:
                rc.set_full_single(grip_chart, np.arange(grip_m.shape[0], dtype=np.float64), grip_m)
            if force_pred is not None and force_pred.shape[0] >= 2:
                # F^cmd, matching the arrow and the wire: ONE flip of the whole wrench, so the
                # force and torque charts cannot end up on opposite conventions.
                _fcmd = cmd_from_meas_wrench(force_pred[:, 0:6])
                _set_force_chart(force_chart, _fcmd[:, 0:3])
                _set_force_chart(torque_chart, _fcmd[:, 3:6])
            rd_chunks.value = str(n_chunks)

        # Live robot mesh from the low-level readback (RECORD=1). The WHOLE block is guarded: any
        # parse/remap/update failure is viz-only and must never take down the render loop. When
        # there is NO fresh live frame (offline, or RECORD off) the main mesh simply HOLDS its
        # last pose — the ready pose it was seeded at, or the last live joints — rather than
        # animating a home-anchored preview (that preview was 166 mm/47° off and is gone). The
        # translucent GHOST, solved in the render URDF, is the animated preview now.
        live_posed = False
        live_follow = None  # (wrist_pos_world, wrist_wxyz) measured, for the triad cluster
        live_tcp = None     # (tcp_pos_world, tcp_rot6d) measured, for the EXECUTED trace
        live_jaw_m = None   # measured jaw width (m) or None, for the trace's gripper panel
        # Raw readback fields the DEPLOY RECORDER stores alongside the FK'd pose. Kept as
        # locals here (rather than re-parsed in the recording block below) so the take logs the
        # very frame this tick rendered, not a newer one that arrived microseconds later.
        live_dof = None      # (29,) SDK dof_pos — the operator's "机器人的各个joint"
        live_armed = False   # was the low level applying the wire at this tick?
        live_fmeas = None    # (2,3) pelvis-frame F_meas the servo consumed (POST-gate)
        # ...and the PRE-gate reading. Both are kept because they answer different questions: the
        # gated one is what the controller acted on (what the take must record), the ungated one is
        # what the sensor actually felt (what the arrow must draw — with GATE_CMD=1 the gated row is
        # identically zero for the whole of pure-position mode).
        live_fmeas_ungated = None
        if live_mesh_ok:
            try:
                rmsg, rw, _rn = subs["record"].latest()
                rframe = parse_record_frame(rmsg)
                if rframe is not None and rw and (time.time() - rw) <= 1.0:
                    gmsg, _gw, _gn = subs["gripper"].latest()
                    # FRESHNESS GATE on the jaw, like every other stream in this loop (record
                    # <=1 s above, F/T <=1 s below, obs via OBS_MAX_AGE_S). It was the ONLY
                    # ungated one, and the failure that costs is silent: a jaw bridge that dies
                    # mid-session leaves its LAST width latched in `latest()` forever, so the
                    # mesh kept drawing that jaw — labelled "· live" — no matter where the real
                    # one was. A bridge that dropped while holding a grasp therefore rendered a
                    # permanently SHUT hand on an open gripper. GRIP_MAX_AGE_S is reused rather
                    # than a fresh literal so the mesh and the model's own gripper latch agree
                    # on what "fresh" means; its docstring already asks for exactly this ("it
                    # only needs to stop being presented as a measurement").
                    _g_age = (time.time() - _gw) if _gw else float("inf")
                    _g_fresh = (gmsg is not None and gmsg.get("actual") is not None
                                and _g_age <= GRIP_MAX_AGE_S)
                    # JAW_JOINT_OPEN_M is the "no reading" default — the TOP of the finger
                    # joint's travel, because the joint OPENS the jaw. The mesh must not sit in
                    # a fake grasp. Kept as the stale fallback too (an invented grasp is the
                    # worse error), but the readout below now SAYS it is not a measurement,
                    # because an open-looking jaw and an unknown one are otherwise
                    # indistinguishable on screen.
                    jaw_m = JAW_JOINT_OPEN_M
                    gwidth = None
                    if _g_fresh:
                        gwidth = calib.to_width_m(
                            float(gmsg["actual"]), float(gmsg.get("open", 4.9)), float(gmsg.get("closed", 0.0)))
                        # An APERTURE -> the joint's per-finger displacement (half the width).
                        # NOT the width itself: that maps 94 mm past the 47.5 mm travel limit and
                        # saturates the whole 44-94 mm working range at wide open.
                        jaw_m = jaw_joint_from_width_m(gwidth)
                    # RIGHT jaw pinned OPEN: there is no right-hand gripper stream here, so it
                    # would otherwise inherit whatever this call's default happens to be.
                    #
                    # Throttled to _mesh_draw_period (see its note): the REPOSE is pixels, and
                    # everything below this line is measurement and stays at full rate. So
                    # `live_posed` is set either way — it means "the low level gave us a fresh
                    # frame", which is what the readouts and the follow frame key off, not
                    # "we redrew the mesh this tick".
                    if (_now - _mesh_last_draw["t"]) >= _mesh_draw_period:
                        _mesh_last_draw["t"] = _now
                        urdf.update_cfg(remap_dof_to_urdf(rframe.dof_pos, dof_remap,
                                                          left_jaw_m=jaw_m,
                                                          right_jaw_m=JAW_JOINT_OPEN_M))
                    live_posed = True
                    live_jaw_m = gwidth
                    live_dof = rframe.dof_pos
                    live_armed = bool(rframe.vive_armed)
                    live_fmeas = rframe.f_meas
                    live_fmeas_ungated = rframe.f_meas_ungated
                    if gwidth is not None:
                        rd_grip.value = f"{gwidth * 1000:.1f} mm ({'grasp' if gwidth < _GRASP_THRESH_M else 'open'}) · live"
                        # SQUEEZE READOUT: the MEASURED width next to the COMMANDED target, so
                        # a stalled jaw is visible as a persistent gap. That gap IS the grip
                        # force (a position-only jaw converts position error into force), so
                        # "measured stays above commanded" is success here, not a tracking fault.
                        if grip_ctl is not None and args.command_gripper:
                            with grip_lock:
                                _lf = grip_ctl["last_frac"]
                                _sq = bool(grip_ctl["squeezing"])
                            if _lf is not None:
                                _cmd_mm = (calib.width_closed_m + _lf * (
                                    calib.width_open_m - calib.width_closed_m)) * 1000.0
                                grip_sq_status.value = (
                                    f"cmd {_cmd_mm:.1f} mm vs meas {gwidth * 1000:.1f} mm"
                                    + (f" · SQUEEZING (Δ{gwidth * 1000 - _cmd_mm:+.1f})" if _sq
                                       else " · no overshoot (open mode)"))
                    else:
                        # NO fresh jaw reading. Say so instead of leaving the last measured value
                        # on screen: the mesh is now drawn OPEN as a safe default, and a stale
                        # number beside an open-looking hand reads as "measured open".
                        rd_grip.value = (
                            f"⚠ NO READING ({_g_age:.1f}s stale) — mesh drawn OPEN as a default, "
                            f"NOT measured; check the :8784 gripper bridge"
                            if gmsg is not None else
                            "⚠ NO READING (gripper bridge :8784 never published) — mesh drawn "
                            "OPEN as a default, NOT measured")
                    # TRIADS FOLLOW THE REAL ROBOT. The tcp/tracker/wrist_yaw_link cluster
                    # hangs off /follow, so /follow must ride the MEASURED wrist whenever the
                    # live joints are available — otherwise the triads sit wherever the last
                    # executed chunk left them (and, before the first chunk, at the mink home
                    # pose, which is a different arm configuration than the real robot). Then
                    # the drawn TCP triad coincides with the live fingertips and the green
                    # recorded path visibly starts AT it, which is the whole point of drawing
                    # them. ~0.4 ms measured — affordable on this 30 Hz loop.
                    meas = _measured_full_wire(rframe.dof_pos)
                    if meas is not None:
                        _mtcp, _mrot6d, _wp9, _wo12 = meas
                        _mR = rot6d_to_rotmat(_mrot6d)
                        live_follow = (base_xyz + _mtcp - _mR @ _TCP_OFFSET,
                                       _rot6d_to_wxyz(_mrot6d))
                        # The EXECUTED trace samples the same TCP the green commanded path is
                        # drawn at (base_xyz + pelvis point), so the red and green lines are
                        # directly comparable in the scene AND in the saved sheet. This is the
                        # FK that already ran for the triads — recording costs a list append.
                        live_tcp = (base_xyz + _mtcp, _mrot6d)
            except Exception:  # noqa: BLE001 — viz-only; fall back to predicted playback
                live_posed = False

        # /follow (and therefore the three triads + the camera frustum) prefers the MEASURED
        # wrist; it falls back to the chunk's per-step wrist only when there is no live frame,
        # which is the same RECORD-off degradation the mesh already has.
        if live_follow is not None:
            follow.position = tuple(live_follow[0])
            follow.wxyz = tuple(live_follow[1])

        # EXECUTED TRACE: record the measured TCP while a chunk is actually being dripped.
        # Gated on `dripping` rather than on `trace.recording` alone so the log holds MOTION and
        # not the seconds of idle hold between chunks — those would flat-line the sheet's time
        # axis and (at exec=16, where a boundary dwell is 33 ms against 533 ms of motion) bury
        # the interesting part. The trace is armed per play; this decides which ticks land in it.
        if live_tcp is not None and dripping.is_set():
            trace.add(time.monotonic(), live_tcp[0], live_tcp[1],
                      float("nan") if live_jaw_m is None else float(live_jaw_m))
            # THE MEASURED ROBOT, into the take: the FK'd TCP (same point + frame as the red
            # line, so the recorded and live comparisons agree) AND the raw 29-DOF joint
            # readback, which is what the operator asked for ("机器人的各个joint"). Recorded on
            # the same `dripping` gate as the trace so a take holds MOTION, not the minutes of
            # idle hold between chunks. Costs a list append — the FK already ran above.
            if recorder is not None and recorder.recording:
                # PELVIS-frame TCP (subtract the render's pelvis origin): the take is a robot
                # record, and base_xyz is a viser placement that would otherwise leak into it.
                recorder.on_executed(
                    recorder.elapsed(),
                    tcp_pos=live_tcp[0] - base_xyz, tcp_rot6d=live_tcp[1],
                    dof_pos=live_dof, jaw_m=live_jaw_m,
                    vive_armed=bool(live_armed), f_meas=live_fmeas,
                    chunk_index=int(rec_step["chunk"]), step_index=int(rec_step["i"]),
                )
        if _now - _trace_last_draw["t"] >= _trace_draw_period:
            _trace_last_draw["t"] = _now
            _n_tr = len(trace)  # cheap: no array copy (see ExecutionTrace.__len__)
            # Redraw only while the line is actually GROWING. Keying this on `trace.recording`
            # instead would rebuild the whole polyline 4x/s forever after a manual Run (which
            # leaves the trace armed so consecutive presses form one path), for no visible change.
            if _n_tr != _trace_last_draw.get("n"):
                _trace_last_draw["n"] = _n_tr
                _request_trace_redraw()
            trace_readout.value = (
                f"{_n_tr} samples · {'recording' if trace.recording else 'idle'}"
                if _n_tr else "(no samples yet)")
        _poll_trace_path()

        # THE GHOST PLAYS THE CHUNK, ON A LOOP. Gated on the chunk LENGTH (not on ghost_cfgs — the
        # readout/wrench animate from the moment a chunk is staged, even before the background solve
        # lands). The cursor WRAPS so the plan replays over and over, the way the recorded-replay
        # viewers walk a demo, until the next chunk is staged. On the rig this is the preview of
        # what Run will execute; the SOLID live mesh (above) simultaneously shows the real motion.
        _cmd_drawn = None   # (‖F^cmd‖, step) actually drawn this tick, for the arrows readout
        if play_i["S"] > 0:
            S_play = play_i["S"]
            i = play_i["i"] % S_play
            # GHOST mesh: pose the translucent robot at this step of the plan (its own /robot_ghost
            # frame), and show/hide its whole subtree via the frame handle. Hidden until the
            # background arm IK has produced this chunk's cfgs (ghost_cfgs None) or the operator
            # turns it off. update_cfg is cheap (a config set + FK); the IK cost is not here.
            if ghost_root is not None:
                _ghost_on = bool(show_ghost.value) and ghost_cfgs is not None
                if _ghost_on:
                    try:
                        ghost_urdf.update_cfg(ghost_cfgs[min(i, ghost_cfgs.shape[0] - 1)])
                    except Exception:  # noqa: BLE001 — viz only
                        _ghost_on = False
                ghost_root.visible = _ghost_on
            # OFFLINE keeps the triad cluster ON the rendered ready-pose gripper (its seed) — the
            # ghost carries the motion, so /follow stays put and the "tcp" triad marks where the
            # plan STARTS (the frame check). Live/RECORD-off it rides the chunk's per-step wrist.
            if live_follow is None and not args.offline:
                follow.position = tuple(wpos[i])
                follow.wxyz = tuple(wwxyz[i])
            if grip_m is not None and not live_posed:
                gw = float(grip_m[min(i, grip_m.shape[0] - 1)])
                rd_grip.value = f"{gw * 1000:.1f} mm ({'grasp' if gw < _GRASP_THRESH_M else 'open'}) · predicted"
            wrench.set_visible(bool(show_wrench.value) and force_pred is not None)
            if force_pred is not None and show_wrench.value:
                j = min(i, force_pred.shape[0] - 1)
                # The F^cmd arrow rides the GHOST'S FINGERTIP TCP as it plays — the commanded
                # force read AT the point it acts on, like a demo's F/T in replay. Anchored to the
                # per-step TCP (`path`, i.e. base_xyz + the model's absolute pelvis-frame TCP), NOT
                # to the wrist it used to sit on: the wrist origin is 24.2 cm back along +x, so the
                # arrow stood off the contact point AND off the "tcp" triad, and the offline
                # inference viewer (which parents its arrows on the TCP-driven eef frame) drew the
                # same quantity somewhere else. One anchor convention across both pages.
                _ap, _aq = (path[i], wwxyz[i])
                # TCP axes == wrist axes (every joint from wrist_yaw to the fingertips has rpy=0),
                # so the ORIENTATION is unchanged by moving the anchor — only the origin moved.
                #
                # The wrench handed over is the RAW MEASURED sensor triple; WrenchArrows applies
                # the sensor->TCP relabel (its SENSOR_TO_TCP, pinned against
                # flore.utils.force_frame.TCP_FROM_SENSOR) AND the meas->cmd negation itself. Doing
                # either here would double-apply it, and both are involutions, so the second
                # application is invisible — that is how the arrow once pointed INTO the wiped
                # surface (169° wrong).
                wrench.update(_ap, _aq, force_pred[j, 0:3], force_pred[j, 3:6])
                # TWO numbers, deliberately, because the arrow ANIMATES: `rd_force` is pinned to
                # step 0 (its label says so — it is the chunk's own headline figure, stable enough
                # to read), while the arrows line below reports the step being DRAWN right now, so
                # it can be compared against the cyan measured arrow beside it at the same instant.
                _cmd_drawn = (float(np.linalg.norm(force_pred[j, 0:3])), int(j))
                rd_force.value = f"{np.linalg.norm(force_pred[0, 0:3]):.2f}"
            play_i["i"] = (i + 1) % S_play

        # --- THE MEASURED WRENCH, on the REAL robot's TCP ------------------------------------
        # The counterpart of the ghost's F^cmd arrow, and the pair is the force loop: the low level
        # drives F^cmd + F_meas -> 0, so when it tracks these two are OPPOSITE and equal.
        #
        # Anchor: `live_tcp[0]` — the same FK'd fingertip TCP the red executed trace samples and
        # the same point the "tcp" triad is drawn at, so all three coincide on the live mesh. NO
        # anchor ⇒ NO arrow: without a fresh RecordTap frame there is no measured TCP to stand on,
        # and drawing at a stale or predicted pose would attach a live reading to a pose that is
        # not the robot's.
        #
        # FRAME — two sources, two DIFFERENT chains, which is why this is not one call:
        #   * low-level F_meas (RecordTap): already in the PELVIS frame, and this scene's world IS
        #     the pelvis frame translated by base_xyz with IDENTITY rotation — so it is drawn
        #     verbatim via `update_world`. Applying the sensor->TCP relabel here would rotate an
        #     already-rotated vector. This is bit-for-bit the vector the servo consumes.
        #   * raw :8771: still in the SENSOR frame, so it goes through `update`, whose
        #     `R_world_tcp @ SENSOR_TO_TCP` is the SAME composition as the low level's own
        #     `sensor_force_to_pelvis(f, R_rel)` — `wwxyz`/`live_tcp[1]` is the URDF wrist rotation
        #     in the pelvis frame, i.e. exactly holosoma's `R_rel`. So the two sources agree, and
        #     flipping the toggle is a live cross-check of the whole F/T chain: they may differ
        #     only by tare / RNEA compensation / the deploy-side EMA.
        # Neither is negated (`as_command=False` on this instance): a measurement is shown as
        # measured, which is what makes the anti-parallel reading meaningful.
        _meas_arrow_n = None
        _meas_note = ""
        if bool(show_wrench_meas.value) and live_tcp is not None:
            try:
                if bool(ft_show_meas.value):
                    # `f_meas_ungated`, not `f_meas`: with GATE_CMD=1 the gated buffer is
                    # identically zero whenever the planner commands no force — i.e. all of
                    # pure-position mode — so the arrow would never appear on exactly the runs
                    # where the operator is checking the sensor. The ungated row is the true
                    # reading. (Row 0 = LEFT, the only hand this planner commands.)
                    _fm = live_fmeas_ungated if live_fmeas_ungated is not None else live_fmeas
                    if _fm is not None:
                        _f3 = np.asarray(_fm, dtype=np.float64)
                        if _f3.shape == (2, 3) and np.isfinite(_f3[0]).all():
                            wrench_meas.update_world(live_tcp[0], _f3[0], None)
                            _meas_arrow_n = float(np.linalg.norm(_f3[0]))
                            _meas_note = "pelvis (servo input, ungated)"
                else:
                    _fmsg, _fw6, _ = subs["force"].latest()
                    _f6 = _pick_force_channels(_fmsg, _display_force_roles, require_6dof=False)
                    if _f6 is not None and _fw6 and (time.time() - _fw6) <= 1.0:
                        wrench_meas.update(live_tcp[0], _rot6d_to_wxyz(live_tcp[1]),
                                           _f6[0:3], _f6[3:6])
                        _meas_arrow_n = float(np.linalg.norm(_f6[0:3]))
                        _meas_note = "raw sensor (:8771)"
            except Exception:  # noqa: BLE001 — viz only; never take down the render loop
                _meas_arrow_n = None
        wrench_meas.set_visible(_meas_arrow_n is not None)
        _cmd_txt = (f"F^cmd {_cmd_drawn[0]:.2f} N (step {_cmd_drawn[1]})" if _cmd_drawn is not None
                    else ("F^cmd off" if not bool(show_wrench.value) else "F^cmd: no chunk yet"))
        if not bool(show_wrench_meas.value):
            _meas_txt = "F_meas off"
        elif _meas_arrow_n is not None:
            _meas_txt = f"F_meas {_meas_arrow_n:.2f} N [{_meas_note}]"
        elif live_tcp is None:
            _meas_txt = ("F_meas: OFFLINE — no robot to anchor on" if args.offline else
                         "F_meas: no measured TCP (start the low level with RECORD=1)")
        else:
            _meas_txt = "F_meas: no/stale F/T frame"
        wrench_status.value = f"{_cmd_txt} · {_meas_txt}"

        # --- LIVE F_meas CHARTS ---------------------------------------------------------------
        # Throttled for the same reason as the executed-path redraw: a uPlot `.data` assignment is
        # a msgpack encode + websocket send of the whole window, and this loop's latency is what
        # the arm feels between chunks. 8 Hz is smooth for a scrolling trace; the DATA behind it is
        # still the full ~200 Hz interval-mean (the socket thread fills it — see _FtDisplayTrace).
        if ft_trace is not None and (_now - _ft_last_draw["t"]) >= _ft_draw_period:
            _ft_last_draw["t"] = _now
            _ser = ft_trace.series()
            if _ser is not None:
                _ts, _w6, _pk = _ser
                rc.set_full_xyz(ft_force_chart, _ts, _w6[:, 0:3])
                rc.set_full_xyz(ft_torque_chart, _ts, _w6[:, 3:6])
                ft_peak.value = f"{_pk:.2f} N in the last {FT_CHART_WINDOW_S:g} s"
        time.sleep(1.0 / 30.0)


# --- LEFT-only IK helpers (duplicated from the muse viz to stay independent) ---
def _so3_log(R: np.ndarray) -> np.ndarray:
    """Rotation-matrix → axis-angle vector (the SO(3) log). Used to build the orientation half
    of the ghost arm IK's 6-D error/Jacobian. Handles the small-angle case without dividing by
    ~0; the near-π branch is not needed here (per-step orientation errors stay small)."""
    R = np.asarray(R, dtype=np.float64)
    cos_t = np.clip((np.trace(R) - 1.0) * 0.5, -1.0, 1.0)
    theta = np.arccos(cos_t)
    if theta < 1e-8:
        return np.zeros(3)
    axis = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    return theta / (2.0 * np.sin(theta)) * axis


def _rot6d_to_wxyz(rot6d: np.ndarray) -> np.ndarray:
    q_xyzw = rotmat_to_quat_xyzw(rot6d_to_rotmat(np.asarray(rot6d, dtype=np.float64)))
    return np.asarray([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]], dtype=np.float64)


def _make_force_chart(gui, uplot, rc, title, labels):
    ac = rc.AXIS_COLORS
    x0 = np.arange(16, dtype=np.float64)
    zero = np.zeros(16)
    return gui.add_uplot(
        data=(x0, zero, zero, zero),
        series=(uplot.Series(label="step"),
                uplot.Series(label=labels[0], stroke=ac[0], width=1.5),
                uplot.Series(label=labels[1], stroke=ac[1], width=1.5),
                uplot.Series(label=labels[2], stroke=ac[2], width=1.5)),
        scales={"x": uplot.Scale(time=False)}, title=title, aspect=2.2, height=120)


def _set_force_chart(chart, xyz: np.ndarray) -> None:
    n = xyz.shape[0]
    if n < 2:
        return
    t = np.arange(n, dtype=np.float64)
    chart.data = (t, xyz[:, 0], xyz[:, 1], xyz[:, 2])


if __name__ == "__main__":
    raise SystemExit(main())
