"""WebSocket SERVER on 127.0.0.1:8770 broadcasting a keypoint/joint target frame to holosoma.

holosoma's ``ViveLiveSource`` (in the ``run_policy`` process) is the WS CLIENT
— it does ``websockets.connect(HOLOSOMA_VIVE_WS_URL)`` with auto-reconnect and
validates ``pos`` length (6/9) + ``orn`` length (12). No handshake beyond the
standard WS upgrade.

This module impersonates the VIVE bridge: it runs a WebSocket server on the
expected port and broadcasts the JSON payload at the control loop rate.
holosoma connects as a client the instant the server binds.

THREE WIRE MODES, one per low-level FAMILY — the same three the bridge itself publishes:

* :meth:`S0Driver.send` → ``mode="3pt"``: 3 keypoint poses, for the ``3pt*`` presets whose
  observation carries a ``vr_3point`` tracking front-end.
* :meth:`S0Driver.send_5pt` → ``mode="5pt"``: 5 keypoint poses, for the ``5pt*`` family —
  the 3pt triple plus both feet, which no tracker supplies.
* :meth:`S0Driver.send_dense` → ``mode="dense"``: 29 joint targets + names, for the
  ``dense`` / ``wbt`` whole-body family, which has no keypoint term at all.

The last two both come from motion completion (``fuji/s1_policy/serving/hl_dense.py``),
because a 3-tracker planner predicts three poses and those families want five points / all
29 joints.

Which one a deploy needs is a fact about the ONNX being served, not a preference — the low
level's width guard drops a frame of the wrong shape, so a mismatch shows up as a robot
holding its ready pose forever.

TWO OPTIONAL EXTENSIONS, both defaulted OFF so every existing deploy is byte-identical
=====================================================================================
* ``on_control`` — a callback for messages the CLIENT sends back up the same socket. holosoma
  never sends any (its ``ViveLiveSource`` is send-nothing), so with the default ``None`` the
  read half stays the drain loop it has always been. The mjswan browser sim, on the other
  hand, has had a ``sendCmd`` vocabulary since its Recalibrate button, which is what lets the
  PAGE be the control surface for a planner rather than a second viser window
  (:mod:`fuji.s1_policy.serving.hl_offline`).
* ``extra`` on each ``send*`` — extra top-level keys on the frame, for telemetry a low level
  ignores and a page reads. Reserved keys are REFUSED rather than overwritten: the wire's
  ``mode``/``pos``/``orn``/``joint_*`` fields are what the width guard routes on, so letting a
  caller shadow one would turn a UI field into a silent frame corruption.
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
from typing import Callable, Mapping

import numpy as np

logger = logging.getLogger(__name__)

_DEFAULT_HOST = "127.0.0.1"
_DEFAULT_PORT = 8770

#: Wire keys :meth:`S0Driver.send` / ``send_5pt`` / ``send_dense`` own. ``extra`` may not carry
#: any of them — see the module docstring.
_RESERVED_WIRE_KEYS = frozenset({
    "mode", "pos", "orn", "joint_pos", "joint_vel", "joint_names", "seq", "t", "f_cmd",
    "ref_joints",
})

#: Width of a ``ref_joints`` block (:meth:`S0Driver.send_5pt`): the G1's 29 actuated joints.
N_REF_JOINTS = 29


def _ref_joints_block(ref_joints) -> dict:
    """``(names, pos, vel)`` -> the wire's ``ref_joints`` dict, validated the way the low level
    validates it (29 unique names, two finite 29-vectors) so a malformed block is refused HERE
    with a reason instead of being silently ignored on the robot."""
    try:
        names, pos, vel = ref_joints
    except (TypeError, ValueError) as exc:
        raise ValueError("ref_joints must be a (names, pos, vel) triple") from exc
    names = [str(n) for n in names]
    p = np.asarray(pos, dtype=float).reshape(-1)
    v = np.asarray(vel, dtype=float).reshape(-1)
    if not (len(names) == p.size == v.size == N_REF_JOINTS):
        raise ValueError(
            f"ref_joints width mismatch: names {len(names)}, pos {p.size}, vel {v.size} — all three "
            f"must be {N_REF_JOINTS} (the low level ignores anything else)")
    if len(set(names)) != N_REF_JOINTS:
        raise ValueError("ref_joints names must be 29 DISTINCT joint names (they are placed by name)")
    if not (np.all(np.isfinite(p)) and np.all(np.isfinite(v))):
        raise ValueError("ref_joints must be finite")
    return {"names": names, "pos": p.tolist(), "vel": v.tolist()}


def _merge_extra(msg: dict, extra: Mapping | None) -> dict:
    """``msg`` plus ``extra``'s keys, refusing to shadow anything the wire owns."""
    if not extra:
        return msg
    clash = sorted(_RESERVED_WIRE_KEYS.intersection(extra))
    if clash:
        raise ValueError(
            f"extra wire keys {clash} collide with the frame's own fields. The low level routes "
            "on `mode` and validates the pos/orn/joint_* widths, so shadowing one of these would "
            "corrupt the command rather than add telemetry.")
    msg.update(dict(extra))
    return msg


class S0Driver:
    """WebSocket server broadcasting vr_3point targets to holosoma's S0 controller.

    Thread-safe: the control loop calls :meth:`send` from its own thread; the WS
    event loop runs on a background thread and broadcasts the latest payload to
    all connected clients (usually just one — holosoma's ViveLiveSource).
    """

    #: Stop re-broadcasting when send() hasn't been called for this long, so
    #: holosoma's ViveLiveSource 0.25 s stale gate can actually fire when the
    #: control loop crashes/hangs. Without this, the background broadcaster
    #: would re-send the last target forever, "looking alive" while frozen.
    FRESHNESS_TIMEOUT_S = 0.5

    def __init__(self, host: str = _DEFAULT_HOST, port: int = _DEFAULT_PORT,
                 on_control: Callable[[dict], None] | None = None):
        if host not in ("127.0.0.1", "localhost"):
            logger.warning("S0Driver binding to non-localhost %s — ensure this is intentional", host)
        self._host = host
        self._port = port
        #: Called on the WS event loop for each JSON OBJECT a client sends. ``None`` (the
        #: default, and what every rig deploy uses) keeps the read half a pure drain. It must
        #: not block: the same loop runs the broadcaster, so a slow callback stalls the wire —
        #: implementations QUEUE and let their own thread act (see ``hl_offline``).
        self._on_control = on_control
        self._lock = threading.Lock()
        self._latest: str | None = None
        self._latest_t = 0.0  # perf_counter() of the last send()
        self._seq = 0
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._running = False
        self._clients: set = set()
        self._stall_logged = False

    def start(self) -> None:
        """Start the WS server on a background thread."""
        self._running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True, name="s0_driver_ws")
        self._thread.start()
        time.sleep(0.1)
        logger.info("S0Driver WS server started on ws://%s:%d", self._host, self._port)

    def _run_loop(self) -> None:
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._loop.run_until_complete(self._serve())

    async def _serve(self) -> None:
        import websockets

        async def handler(ws):
            self._clients.add(ws)
            peer = getattr(ws, "remote_address", "?")
            logger.info("S0Driver: holosoma connected from %s (total %d)", peer, len(self._clients))
            try:
                async for raw in ws:
                    # DRAIN unless a control callback was installed. holosoma sends nothing at
                    # all, so this loop existed only to notice the socket closing; the mjswan
                    # page's `sendCmd` is the one producer, and a planner that did not install a
                    # callback must keep swallowing it rather than grow behaviour by accident.
                    if self._on_control is None:
                        continue
                    try:
                        msg = json.loads(raw)
                    except (TypeError, ValueError):
                        logger.warning("S0Driver: unparseable control message %r", raw)
                        continue
                    if not isinstance(msg, dict):
                        logger.warning("S0Driver: ignoring non-object control message %r", msg)
                        continue
                    try:
                        self._on_control(msg)
                    except Exception:  # noqa: BLE001 - a bad control must not kill the wire
                        logger.exception("S0Driver: control callback raised on %r", msg)
            except Exception:
                pass
            finally:
                self._clients.discard(ws)
                logger.info("S0Driver: holosoma disconnected (%d remaining)", len(self._clients))

        import websockets.server

        # compression=None: the payload is ~500 bytes of JSON floats re-sent 200x/s over
        # loopback. Deflating it costs both this process and the low level per message and
        # saves nothing worth having at that size; `websockets.broadcast` also cannot share a
        # compression context across clients, so the cost would scale with client count.
        async with websockets.serve(handler, self._host, self._port, compression=None):
            while self._running:
                with self._lock:
                    payload = self._latest
                    age = time.perf_counter() - self._latest_t
                # STOP re-broadcasting once the control loop has gone silent for
                # longer than FRESHNESS_TIMEOUT_S. ViveLiveSource stamps freshness
                # on RECEIPT, so an unconditional re-send would defeat its 0.25 s
                # stale gate — a crashed/hung loop would freeze S0 on the last
                # target while looking alive. When we stop sending, S0's own gate
                # fires and it holds-then-safes per its policy.
                if payload is not None and self._clients and age <= self.FRESHNESS_TIMEOUT_S:
                    import websockets

                    websockets.broadcast(self._clients, payload)
                    self._stall_logged = False
                elif payload is not None and age > self.FRESHNESS_TIMEOUT_S and not self._stall_logged:
                    logger.warning(
                        "S0Driver: no send() for %.2fs — stopped re-broadcasting so "
                        "S0's stale gate can fire (control loop stalled?)",
                        age,
                    )
                    self._stall_logged = True
                await asyncio.sleep(0.005)

    def send(self, pos: np.ndarray, orn: np.ndarray, f_cmd: np.ndarray | None = None,
             extra: Mapping | None = None) -> None:
        """Set the latest 3pt target to broadcast.

        Parameters
        ----------
        pos : (9,) float — pelvis-frame absolute positions [L, R, H]
        orn : (12,) float — absolute orientations [L, R, H] as xyzw quats
        f_cmd : optional per-hand force command, RAW Bota SENSOR frame, Newtons. Either
            (3,) = LEFT force, or (6,) = [L, R] forces. OMITTED from the wire when None,
            so the pose-only deploy (VIVE bridge, position-only planner) is byte-identical
            — the ``f_cmd`` key simply never appears. When present, holosoma's
            ``ViveLiveSource`` reads it and the low level rotates it sensor→pelvis itself
            (it has the live joint FK the planner lacks) and folds it through the existing
            2 N/tick F^cmd slew. See the HL-v2 online serving contract.
        extra : optional extra top-level wire keys (telemetry a low level ignores; the page
            reads them). Reserved keys are refused — see the module docstring.
        """
        self._seq += 1
        msg = {
            "mode": "3pt",
            "pos": pos.tolist(),
            "orn": orn.tolist(),
            "seq": self._seq,
            "t": time.time(),
        }
        if f_cmd is not None:
            msg["f_cmd"] = np.asarray(f_cmd, dtype=float).reshape(-1).tolist()
        _merge_extra(msg, extra)
        with self._lock:
            self._latest = json.dumps(msg)
            self._latest_t = time.perf_counter()

    def send_5pt(self, pos: np.ndarray, orn: np.ndarray,
                 f_cmd: np.ndarray | None = None, extra: Mapping | None = None,
                 ref_joints=None) -> None:
        """Set the latest 5pt keypoint target to broadcast.

        The wire the ``5pt-*`` family is steered by: the 3pt triple (both hands on the 0.242 m
        force TCP, the head at ``torso_link`` + 0.35 z) plus **both feet** at the bare
        ``*_ankle_roll_link`` origins. The rig has three trackers and no foot pucks, so those
        last two points always come from motion completion — on the rig out of the VIVE
        bridge's completion pass, here out of the planner's
        (``fuji/s1_policy/serving/hl_dense.FivePointWire``).

        A DIFFERENT ``mode`` rather than a wider ``pos`` on :meth:`send`, because
        ``ViveLiveSource`` validates the ``(len(pos), len(orn))`` PAIR against a per-mode
        table and cross-checks it against the ``mode`` tag: a 2pt frame's 2 x rot6d and a 3pt
        frame's 3 x quat are both 12 wide, so the tag is what tells two reps apart.

        Parameters
        ----------
        pos : (15,) float — PELVIS-frame positions ``[L, R, head, L_foot, R_foot]`` xyz.
        orn : (20,) float — 5 xyzw quaternions of the BODY rotations, in the same order and
            the same pelvis frame. Must be unit-norm; the low level composes them with its own
            arm-clutch anchor, which scales rather than rejects a de-normalised quat.
        f_cmd : as :meth:`send` — omitted from the wire when None, so a position-only 5pt
            deploy is byte-identical to the bridge's own frame.
        ref_joints : optional ``(names, pos, vel)`` — the REFERENCE motion's 29 joint targets
            (rad) and velocities (rad/s) for this tick, BY NAME. OMITTED when None (every frame
            but the walk skill's, byte-identical). Written as ``{"names", "pos", "vel"}``; the
            low level (``ViveLiveSource.latest_ref_joints``) substitutes it for the frozen idle
            clip in its ``command_lower_body`` obs term, 5pt only and while armed, so the legs
            and the two foot keypoints describe ONE gait. Never slewed there. Refused here with
            a reason if not 29 distinct names + two finite 29-vectors.
        """
        p = np.asarray(pos, dtype=float).reshape(-1)
        q = np.asarray(orn, dtype=float).reshape(-1)
        if (p.size, q.size) != (15, 20):
            raise ValueError(
                f"5pt frame width mismatch: pos {p.size} (want 15), orn {q.size} (want 20). "
                "ViveLiveSource validates the PAIR and DROPS anything else, so a wrong-width "
                "frame would look like a robot that never moves.")
        block = None if ref_joints is None else _ref_joints_block(ref_joints)
        self._seq += 1
        msg = {
            "mode": "5pt",
            "pos": p.tolist(),
            "orn": q.tolist(),
            "seq": self._seq,
            "t": time.time(),
        }
        if f_cmd is not None:
            msg["f_cmd"] = np.asarray(f_cmd, dtype=float).reshape(-1).tolist()
        if block is not None:
            msg["ref_joints"] = block
        _merge_extra(msg, extra)
        with self._lock:
            self._latest = json.dumps(msg)
            self._latest_t = time.perf_counter()

    def send_dense(
        self,
        joint_pos: np.ndarray,
        joint_vel: np.ndarray,
        joint_names,
        f_cmd: np.ndarray | None = None,
        extra: Mapping | None = None,
    ) -> None:
        """Set the latest DENSE (29 joint target) frame to broadcast.

        The wire the whole-body ``dense`` / ``wbt`` policy family is steered by, and a
        different ``mode`` rather than a wider ``pos``: those policies have no keypoint
        front-end at all, they take joint targets. Consumed by ``ViveLiveSource``'s
        ``latest_joints()`` (see its module docstring) and produced on the rig by
        ``vive/mc_teleop_source.wire_payload(mode="dense")`` — this method exists so a FUJI
        planner can be the producer instead of the VIVE bridge, with the identical frame.

        Parameters
        ----------
        joint_pos, joint_vel : (29,) float — absolute joint targets (rad) and their
            reference velocity (rad/s).
        joint_names : 29 names, IN THE SAME ORDER as the values. Mandatory, and shipped every
            frame: MC's joint order and holosoma's ``dof_names`` are both real orders and are
            NOT guaranteed equal, so the consumer must remap by name — ``ViveLiveSource``
            DROPS a dense frame whose ``joint_names`` is missing or not 29 long rather than
            risk index-aligning a scrambled body.
        f_cmd : as :meth:`send` — omitted from the wire when None, so a position-only dense
            deploy is byte-identical to the bridge's own frame.
        """
        names = [str(n) for n in joint_names]
        pos = np.asarray(joint_pos, dtype=float).reshape(-1)
        vel = np.asarray(joint_vel, dtype=float).reshape(-1)
        if not (pos.size == vel.size == len(names)):
            raise ValueError(
                f"dense frame width mismatch: joint_pos {pos.size}, joint_vel {vel.size}, "
                f"joint_names {len(names)} — all three must agree")
        self._seq += 1
        msg = {
            "mode": "dense",
            "joint_pos": pos.tolist(),
            "joint_vel": vel.tolist(),
            "joint_names": names,
            "seq": self._seq,
            "t": time.time(),
        }
        if f_cmd is not None:
            msg["f_cmd"] = np.asarray(f_cmd, dtype=float).reshape(-1).tolist()
        _merge_extra(msg, extra)
        with self._lock:
            self._latest = json.dumps(msg)
            self._latest_t = time.perf_counter()

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    @property
    def num_clients(self) -> int:
        return len(self._clients)
