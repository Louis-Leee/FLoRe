"""Broadcast-only WebSocket tap for the OPT-IN teleop-data recorder (``RECORD=1``).

This is the one policy-process-internal signal source the out-of-process recorder
(``force_deploy/record_teleop.py``) cannot get any other way:
the commanded force ``F^cmd`` (the u/j/h/k/y/n keys), the pelvis-frame measured
force / servo state as the policy actually uses them, the full 29-DOF proprioception
the ONNX saw this tick, and — crucially — the ``running`` (START) edge that begins a
take. Everything else (raw F/T, gripper, VIVE, cameras) is already on its own bridge
WS the recorder subscribes to directly; this tap fills only the in-process gap.

Design mirrors the ``teleop_tracking_log`` / ``_debug_log`` off-by-default idiom so
the realtime policy pays NOTHING when recording is off:

* ``HolosomaWBTPolicy`` constructs a :class:`RecordTap` ONLY when ``RECORD`` is set to
  a truthy value; otherwise ``self._record_tap is None`` and the per-tick call site is
  a single ``is None`` early-return — **byte-identical** to a build without this file.
* When on, :meth:`RecordTap.publish` does O(copy) work on the RL loop thread (snapshot
  a handful of small arrays into a dict, drop it on a bounded queue) and NOTHING else:
  the JSON encode + socket fan-out run on a private asyncio thread, so a slow/absent
  recorder client can never stall — or even perturb the timing of — inference. The
  queue is bounded and drops the OLDEST frame on overflow (a stalled consumer loses
  data, never blocks the producer). This preserves the load-bearing invariant that the
  policy loop's cadence is independent of any observer.

The publisher is a SERVER (the recorder is the client), so the policy never blocks on
a connect and the recorder can attach/detach freely. Wire frame (one text JSON per
tick), all arrays plain nested lists, SI units, pelvis frame unless noted::

    {"type": "record", "seq": int, "t": wall_seconds,
     "running": bool,            # use_policy_action — the START/active gate (take edge)
     "vive_armed": bool,         # trackers actually driving the hands (vs standing pose)
     "timestep": float,          # curr_motion_timestep
     "servo_on": bool,           # HOLOSOMA_HYBRID_SERVO active this session
     "f_cmd":   [[Lx,Ly,Lz],[Rx,Ry,Rz]] | null,   # applied (slewed) F^cmd, pelvis N
     "f_meas":  [[..],[..]],     # _force_meas_pelvis AS THE POLICY USES IT (post EMA+gate), pelvis N
     "f_meas_ungated": [[..],[..]],  # before the gate-on-cmd zeroing (the true reading), pelvis N
     "servo_integ": [[..],[..]], # Jᵀ servo integrator I, pelvis N
     "servo_dx":    [[..],[..]], # servo setpoint shift Δx, pelvis m
     "servo_dq":    [..29..],    # per-DOF arm PD-target shift, rad, SDK order
     "dof_pos":     [..29..],    # ABSOLUTE joint positions, rad, SDK order (NOT obs-relative)
     "dof_vel":     [..29..],    # joint velocities, rad/s, SDK order
     "root_quat_wxyz": [w,x,y,z],# IMU body orientation
     "base_ang_vel":   [wx,wy,wz],
     "vr_cmd_pos": [[Lx,Ly,Lz],[Rx,Ry,Rz](,[Hx,Hy,Hz](,[LFx..],[RFx..]))] | null}
                                 # the vr_2/3/5point_pos obs the policy saw this tick = the
                                 # live-driven hand (+head(+feet)) TARGET, pelvis frame, metres.
                                 # 2/3/5 rows follow the preset. Pairs with dof_pos (→ executed
                                 # wrist FK offline) for the tracking-error plot.
                                 # ⚠️ POST-SLEW: this is the value AFTER _slew_vec applied
                                 # HOLOSOMA_VIVE_MAX_POS_STEP, so successive frames differ by at
                                 # most that cap and a consumer can MEASURE the clamp rather than
                                 # having to be told the launcher's value.
     "vr_cmd_slewed": [true, ...] | null  # per ROW of vr_cmd_pos: did the slew caps cover that
                                 # point this tick? All true unless HOLOSOMA_VIVE_SLEW_FEET=0 exempts
                                 # the 5pt feet ([t,t,t,f,f]). A consumer MEASURING the cap off
                                 # vr_cmd_pos must difference the covered rows only; absent on takes
                                 # recorded before the key existed (treat as all covered).
     "ref_joints_live": true | false | null  # 5pt ONLY, else null: did the wire frame's optional
                                 # `ref_joints` (the walk skill's gait legs, by name) drive the
                                 # command_lower_body obs this tick? false = the frozen idle clip's
                                 # legs (the historical path). Absent on takes recorded before the key.
     "contact_phase": [phiL, phiR] | null   # C1 UNIFIED presets only: the soft per-hand
                                 # contact label the actor consumed this tick (hysteresis on
                                 # ‖F_meas‖ + 50 ms LPF, ∈[0,1]). Emitted as null (not omitted)
                                 # for every preset whose obs has no contact_phase term; takes
                                 # recorded before this key existed simply lack it.

TAKE BOUNDARIES — note ``rl_inference`` (and thus ``publish``) runs ONLY while the
policy is active (``use_policy_action`` True, base.py Stage 3), so ``running`` is
constant-True in every emitted frame — there is no false→true edge WITHIN the stream.
The recorder therefore keys the take on FRAME ARRIVAL: the first frame it sees opens a
take, and the CESSATION of frames (a STOP / limp gap longer than its idle timeout)
closes + flushes it. So recording begins on exactly the same A=START press that starts
the policy (and arms VIVE control), with no extra operator action. ``running`` is kept
in the frame as an explicit marker (and to carry a rare explicit running=false frame).

Deliberately dependency-light: stdlib + numpy + websockets (present in the policy env).
No holosoma / no torch import, so importing this file is cheap and side-effect-free.
"""

from __future__ import annotations

import asyncio
import json
import os
import queue
import threading
import time

import numpy as np
from loguru import logger

# Default recorder tap port. The recorder subscribes here; keep it clear of the
# other bridge ports (VIVE 8770, F/T 8771, cameras 8773/8777, gripper 8784/8785).
DEFAULT_RECORD_TAP_PORT = 8779


def record_enabled() -> bool:
    """True iff the ``RECORD`` (or legacy ``HOLOSOMA_RECORD``) env flag is truthy.

    Matches the truthy-string convention used by the other policy env gates
    (GATE_CMD / HYBRID_SERVO): ``1/true/yes/on`` (case-insensitive)."""
    for var in ("RECORD", "HOLOSOMA_RECORD"):
        val = os.environ.get(var, "").strip().lower()
        if val in ("1", "true", "yes", "on"):
            return True
        if val in ("0", "false", "no", "off", ""):
            continue
    return False


def _as_list(a):
    """None-safe ndarray → nested python list (JSON-ready). None stays None."""
    if a is None:
        return None
    return np.asarray(a, dtype=np.float64).tolist()


class RecordTap:
    """A broadcast-only localhost WebSocket server that ships one JSON frame per
    control tick to the (out-of-process) teleop recorder.

    Construct ONLY when recording is enabled. The RL loop calls :meth:`publish`
    each tick with already-snapshotted arrays; a private asyncio thread owns the
    socket + JSON encode + fan-out, so :meth:`publish` never does socket I/O.
    """

    def __init__(self, host: str = "127.0.0.1", port: int = DEFAULT_RECORD_TAP_PORT, max_queue: int = 256):
        self.host = host
        self.port = port
        self._seq = 0
        # Bounded: a stalled/absent recorder must never back-pressure the RL loop.
        # We drop the OLDEST frame on overflow (recording loses a sample, inference
        # never blocks) — the same "capture is best-effort, control is sacred" rule
        # the force/camera bridges follow with websockets.broadcast fan-out.
        self._q: queue.Queue[dict] = queue.Queue(maxsize=max_queue)
        self._dropped = 0
        self._encode_errors = 0
        self._reported_dropped = 0  # last drop tally the drain thread has logged
        self._stop = threading.Event()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._clients: set = set()
        self._thread = threading.Thread(target=self._run, name="record-tap", daemon=True)
        self._thread.start()
        logger.info(
            f"[record-tap] teleop RECORD tap broadcasting at ws://{host}:{port} "
            "(the recorder subscribes here; inference is unaffected if nothing connects)."
        )

    # ---- producer side (RL loop thread) -----------------------------------
    def publish(
        self,
        *,
        running: bool,
        vive_armed: bool,
        timestep: float,
        servo_on: bool,
        f_cmd,
        f_meas,
        f_meas_ungated,
        servo_integ,
        servo_dx,
        servo_dq,
        dof_pos,
        dof_vel,
        root_quat_wxyz,
        base_ang_vel,
        vr_cmd_pos=None,
        vr_cmd_slewed=None,
        contact_phase=None,
        ref_joints_live=None,
    ) -> None:
        """Snapshot this tick into a plain-list dict + enqueue it. STRICTLY O(copy) on
        the RL loop thread: ``_as_list`` deep-copies each ndarray to a python list (so the
        dict aliases nothing the loop later mutates), and the dict is enqueued as-is. The
        JSON encode AND all logging happen on the private asyncio thread — NOTHING on this
        sacred 50 Hz thread does socket I/O, json.dumps, or a (potentially blocking) log
        write. Never raises into the caller (a bad frame is silently dropped + counted)."""
        try:
            frame = {
                "type": "record",
                "seq": self._seq,
                "t": time.time(),
                "running": bool(running),
                "vive_armed": bool(vive_armed),
                "timestep": float(timestep),
                "servo_on": bool(servo_on),
                "f_cmd": _as_list(f_cmd),
                "f_meas": _as_list(f_meas),
                "f_meas_ungated": _as_list(f_meas_ungated),
                "servo_integ": _as_list(servo_integ),
                "servo_dx": _as_list(servo_dx),
                "servo_dq": _as_list(servo_dq),
                "dof_pos": _as_list(dof_pos),
                "dof_vel": _as_list(dof_vel),
                "root_quat_wxyz": _as_list(root_quat_wxyz),
                "base_ang_vel": _as_list(base_ang_vel),
                "vr_cmd_pos": _as_list(vr_cmd_pos),
                "vr_cmd_slewed": None if vr_cmd_slewed is None else [bool(b) for b in vr_cmd_slewed],
                "contact_phase": _as_list(contact_phase),
                "ref_joints_live": None if ref_joints_live is None else bool(ref_joints_live),
            }
        except Exception:  # noqa: BLE001 - a bad frame must never touch control
            # No log here: logging is synchronous (loguru, non-enqueued) and this runs on
            # the RL loop thread. Count it; the drain thread surfaces the tally.
            self._encode_errors += 1
            return
        # seq advances iff a frame is actually enqueued below (assigned above, bumped here)
        self._seq += 1
        try:
            self._q.put_nowait(frame)
        except queue.Full:
            # Drop the oldest, enqueue the newest — keep the freshest data flowing without
            # ever blocking the producer. No log here (see above); the drain thread reports
            # the drop count periodically off the RL thread.
            try:
                self._q.get_nowait()
                self._q.put_nowait(frame)
            except queue.Empty:
                pass
            self._dropped += 1

    def close(self) -> None:
        self._stop.set()
        loop = self._loop
        if loop is not None:
            try:
                loop.call_soon_threadsafe(lambda: None)  # wake the loop to observe _stop
            except Exception:  # noqa: BLE001
                pass

    # ---- consumer side (private asyncio thread) ---------------------------
    def _run(self) -> None:
        try:
            asyncio.run(self._serve())
        except Exception as exc:  # noqa: BLE001 - the tap failing must not crash the policy
            logger.warning(f"[record-tap] server thread exited: {exc!r} (recording disabled this run).")

    async def _serve(self) -> None:
        import websockets

        self._loop = asyncio.get_running_loop()

        async def handler(ws):
            self._clients.add(ws)
            peer = getattr(ws, "remote_address", "?")
            logger.info(f"[record-tap] recorder connected: {peer} (total {len(self._clients)})")
            try:
                async for _ in ws:  # the recorder never sends; just hold the conn open
                    pass
            except Exception:  # noqa: BLE001
                pass
            finally:
                self._clients.discard(ws)
                logger.info(f"[record-tap] recorder gone: {peer} (total {len(self._clients)})")

        async with websockets.serve(handler, self.host, self.port, max_size=None):
            # Drain the producer queue, JSON-encode HERE (off the RL thread), and fan out.
            # A short timeout keeps the loop responsive to _stop even when the queue is idle,
            # and lets us surface the producer's drop/encode-error tallies periodically —
            # all logging stays on THIS thread, never the RL loop thread.
            while not self._stop.is_set():
                frame = await self._loop.run_in_executor(None, self._drain_one)
                if frame is None:
                    self._report_drops()
                    continue
                try:
                    payload = json.dumps(frame)
                except Exception as exc:  # noqa: BLE001 - a bad frame must not kill the drain loop
                    self._encode_errors += 1
                    logger.warning(f"[record-tap] frame encode failed on drain thread (dropped): {exc!r}")
                    continue
                if self._clients:
                    import websockets as _ws

                    _ws.broadcast(self._clients, payload)

    def _report_drops(self) -> None:
        """Log the accumulated drop/encode-error tally from the DRAIN thread (never the
        RL loop thread). Called on the idle path so it can't add latency to a busy loop."""
        if self._dropped != self._reported_dropped:
            self._reported_dropped = self._dropped
            logger.warning(
                f"[record-tap] recorder not draining; dropped {self._dropped} frame(s), "
                f"{self._encode_errors} encode-error(s) (inference UNAFFECTED — capture is best-effort)."
            )

    def _drain_one(self, timeout: float = 0.2) -> "dict | None":
        try:
            return self._q.get(timeout=timeout)
        except queue.Empty:
            return None
