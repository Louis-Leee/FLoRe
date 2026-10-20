"""Live VIVE hand/head-target source for the offboard ``run_policy`` path.

This is the Python analogue of the browser's ``WbtVr2PointPosLive`` /
``WbtVr2PointOrnLive`` obs classes: it connects to the SAME
``vive_teleop_bridge`` WebSocket (default ``ws://localhost:8770``) and keeps the
latest ``{pos, orn}`` VR command so ``HolosomaWBTPolicy`` can override its
NPZ-FK-derived ``vr_{2,3}point_pos`` / ``vr_{2,3}point_orn`` obs terms each step
with the operator's live pose.

The bridge emits the command in the EXACT representation the policy consumes
(see ``vive_teleop_bridge.snapshot()``), tagged with ``"mode"``:

* 2pt (``--mode 2pt``, default): ``pos`` (6) = ``[L_xyz, R_xyz]`` and ``orn``
  (12) = ``[L_rot6d, R_rot6d]`` (rot6d = first two COLUMNS of the
  wrist-relative-to-pelvis rotation matrix) — identical to
  ``HolosomaWBTPolicy._compute_vr_fk``'s 2pt ``vr_pos_local`` / ``vr_orn_local``.
* 3pt (``--mode 3pt``): ``pos`` (9) = ``[L_xyz, R_xyz, head_xyz]`` and ``orn``
  (12) = ``[L_quat, R_quat, head_quat]`` as xyzw QUATERNIONS (3 bodies × 4) —
  identical to the policy's 3pt ``_compute_vr_fk`` output. The 3rd point is
  torso_link + offset (0,0,0.35) in the pelvis frame.
* 5pt (``--mode 5pt``): ``pos`` (15) = ``[L, R, head, L_foot, R_foot]_xyz`` and
  ``orn`` (20) = 5 xyzw quaternions. The rig has only THREE trackers, so the extra
  two points come from a Motion_Completion pass in the bridge, read out of ITS
  29-DoF FK — but the payload is still the policy's own
  ``vr_5point_pos_tcp`` / ``vr_5point_orn`` rep, at the 0.242 m TCP on both hands.
* dense (``--mode dense``): NO keypoints. ``joint_pos`` (29) + ``joint_vel`` (29)
  absolute joint targets, plus ``joint_names`` (29) so the consumer remaps BY NAME
  (see :meth:`latest_joints`). Also from the same completion pass.

* OPTIONAL on a 5pt frame — ``ref_joints``: ``{"names": [29], "pos": [29], "vel": [29]}``,
  the REFERENCE MOTION's joint targets + velocities for this tick, BY NAME. FLoRe's walk
  skill sends it while a gait clip plays so the policy's ``command_lower_body`` term (12
  reference leg joints + velocities, otherwise read from the FROZEN idle clip) describes
  the SAME gait the two foot keypoints do — the input a baked clip gives the policy in
  mjswan, where it walks. Absent on every other frame (byte-identical). Read with
  :meth:`latest_ref_joints`; the policy applies it for 5pt only, while armed, and never
  slews it (a slewed reference is exactly the crawl the feet were exempted from).

So there is **no transform** here: the bytes drop straight into the obs, and the
policy slices them per its detected obs mode (2pt / 3pt / 5pt / dense).

Safety: when no message has ever arrived, or the last message is older than
``stale_timeout``, ``latest()`` returns ``(None, None)`` so the caller HOLDS its
last good / default target rather than commanding a zero hand pose (a zero
``vr_2point`` on a real robot is a sudden hand-target jump). Mid-session the
policy FREEZES at the last target it actually fed (zero commanded motion; a
stale→fresh recovery slew-ramps from there); before any live frame it keeps the
NPZ-FK reference. This mirrors the bridge's own ``snapshot()`` returning
``None`` → browser holds HOME.

Deliberately depends only on ``numpy`` + ``websockets`` (no holosoma / mujoco at
runtime), runs its reader on a daemon thread with its own asyncio loop, and is
entirely opt-in: ``HolosomaWBTPolicy`` constructs it only when
``HOLOSOMA_VIVE_WS_URL`` is set, so the default NPZ-replay path is byte-identical
when off.
"""

from __future__ import annotations

import asyncio
import json
import threading
import time

import numpy as np
from loguru import logger

#: G1 actuated joints — the width of a dense frame's ``joint_pos`` / ``joint_vel``.
_NUM_DOFS = 29

#: Legal ``(len(pos), len(orn))`` pairs, per tracking family. The PAIR matters, not each
#: width alone: 2pt's 2×rot6d and 3pt's 3×xyzw-quat are both 12 wide, so orn width cannot
#: identify the rotation rep on its own and a lone `len(orn) == 12` check would let a 2pt
#: frame's rot6d bytes be read as quaternions.
_MODE_WIDTHS: dict[str, tuple[int, int]] = {
    "2pt": (6, 12),  # [L,R] xyz                      + 2 × rot6d
    "3pt": (9, 12),  # [L,R,head] xyz                 + 3 × quat xyzw
    "5pt": (15, 20),  # [L,R,head,Lfoot,Rfoot] xyz     + 5 × quat xyzw
}
_POSE_WIDTHS = frozenset(_MODE_WIDTHS.values())


class ViveLiveSource:
    """Background WebSocket client that holds the latest live 2pt hand command.

    Args:
        ws_url: bridge WebSocket URL (e.g. ``ws://localhost:8770``).
        stale_timeout: seconds; if the newest message is older than this,
            ``latest()`` returns ``(None, None)`` so the policy holds its last
            good target. ``0`` disables the staleness gate (not recommended on
            hardware). Default 0.25 s, matching the bridge's own stale default.
    """

    def __init__(self, ws_url: str, stale_timeout: float = 0.25):
        self._ws_url = ws_url
        self._stale_timeout = float(stale_timeout)
        self._lock = threading.Lock()
        self._pos: np.ndarray | None = None  # (6,) 2pt / (9,) 3pt / (15,) 5pt float32, pelvis frame
        self._orn: np.ndarray | None = None  # 2×rot6d (2pt) | 3×quat-xyzw (3pt) | 5×quat-xyzw (5pt)
        # DENSE mode only (``--mode dense``): 29 absolute joint targets + velocities in the
        # COMPLETION's joint order, and that order's names. Kept SEPARATE from the pose
        # fields — a dense frame carries no keypoints at all, so a consumer that wants
        # pos/orn must keep getting None rather than a reinterpretation of joint angles.
        self._joint_pos: np.ndarray | None = None
        self._joint_vel: np.ndarray | None = None
        self._joint_names: tuple[str, ...] | None = None
        # OPTIONAL per-hand force command (raw Bota SENSOR frame, N), streamed by the
        # HL-v2 online planner's S0Driver on the SAME wire. (3,)=LEFT or (6,)=[L,R], or
        # None when the frame carries no ``f_cmd`` key (the VIVE bridge + position-only
        # planner never send it, so that path is byte-identical). Stamped with the pose
        # timestamp; see latest_force().
        self._f_cmd: np.ndarray | None = None
        # OPTIONAL reference-joint command riding a KEYPOINT frame (see the module docstring):
        # 29 targets + 29 velocities in the SENDER's joint order plus that order's names, or
        # None when the frame carried no valid ``ref_joints`` — re-evaluated on EVERY keypoint
        # frame, so a frame without the key drops the override (the policy falls back to its
        # frozen clip) rather than holding a stale one against fresh keypoints.
        self._ref_joint_pos: np.ndarray | None = None
        self._ref_joint_vel: np.ndarray | None = None
        self._ref_joint_names: tuple[str, ...] | None = None
        self._last_t: float = 0.0
        self._seq: int = -1
        self._connected = False
        self._stop = threading.Event()
        self._stale_logged = False
        self._joint_names_warned = False
        self._ref_joints_warned = False
        self._thread = threading.Thread(target=self._run, name="vive_live_ws", daemon=True)
        self._thread.start()
        logger.info(f"ViveLiveSource: connecting to {ws_url} (stale_timeout={stale_timeout}s)")

    # --- public API -------------------------------------------------------
    def latest(self) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Return ``(pos[6], orn[12])`` or ``(None, None)`` if stale / no data.

        The caller (``_compute_obs_terms``) should HOLD its last fed target when
        this returns ``None`` (or the NPZ-FK reference before any live frame) —
        never substitute zeros.
        """
        with self._lock:
            if self._pos is None:
                return None, None
            if self._stale_timeout > 0 and (time.time() - self._last_t) > self._stale_timeout:
                if not self._stale_logged:
                    logger.warning(
                        f"ViveLiveSource: command STALE (>{self._stale_timeout * 1000:.0f} ms) "
                        "— policy freezes at its last fed hand target"
                    )
                    self._stale_logged = True
                return None, None
            if self._stale_logged:
                logger.info("ViveLiveSource: live command fresh again")
                self._stale_logged = False
            # _pos and _orn are always set together in _ingest(); the `_pos is
            # None` guard above implies _orn is non-None too.
            assert self._orn is not None
            return self._pos.copy(), self._orn.copy()

    def latest_joints(self) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]] | None:
        """Return ``(joint_pos[29], joint_vel[29], joint_names)`` for a DENSE frame, or
        ``None`` when no dense frame has arrived / the last one is stale.

        Same hold-don't-zero contract as :meth:`latest`: None means "keep commanding the
        held reference", never "command zeros" (29 zero joint targets is the G1 folded
        flat). ``joint_names`` travels WITH the values so the caller must remap by name —
        the completion's order and the policy's ``dof_names`` are two independent real
        orders, and index-aligning them would scramble the body with no error anywhere.
        """
        with self._lock:
            if self._joint_pos is None:
                return None
            if self._stale_timeout > 0 and (time.time() - self._last_t) > self._stale_timeout:
                return None
            assert self._joint_vel is not None and self._joint_names is not None
            return self._joint_pos.copy(), self._joint_vel.copy(), self._joint_names

    def latest_force(self) -> np.ndarray | None:
        """Return the latest live per-hand force command (raw Bota SENSOR frame, N) —
        (3,)=LEFT or (6,)=[L,R] — or ``None`` if no ``f_cmd`` has arrived or the last
        frame is stale (same ``stale_timeout`` gate as :meth:`latest`).

        Returns None (NOT zeros) when absent/stale so the caller holds F^cmd at 0 (a pure
        position controller) rather than commanding a stale force — mirroring how
        :meth:`latest` returns None so the policy freezes its pose target. The VIVE bridge
        and the position-only planner never send ``f_cmd``, so this always returns None
        there (byte-identical position deploy)."""
        with self._lock:
            if self._f_cmd is None:
                return None
            if self._stale_timeout > 0 and (time.time() - self._last_t) > self._stale_timeout:
                return None
            return self._f_cmd.copy()

    def latest_ref_joints(self) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]] | None:
        """Return ``(pos[29], vel[29], names)`` of the ``ref_joints`` the latest KEYPOINT frame
        carried, or ``None`` when that frame carried none / the frame is stale / no frame yet.

        Same stale gate as :meth:`latest`, and the same hold-don't-guess contract: None means "the
        sender is not overriding the reference this tick" — the policy then reads its frozen clip
        (when the pose frame is fresh) or freezes the last override it fed (when the pose frame is
        stale, exactly as it freezes the keypoint target). The names travel WITH the values so the
        consumer remaps by name, never by index (see :meth:`latest_joints`).
        """
        with self._lock:
            if self._ref_joint_pos is None:
                return None
            if self._stale_timeout > 0 and (time.time() - self._last_t) > self._stale_timeout:
                return None
            assert self._ref_joint_vel is not None and self._ref_joint_names is not None
            return self._ref_joint_pos.copy(), self._ref_joint_vel.copy(), self._ref_joint_names

    def shutdown(self) -> None:
        self._stop.set()

    # --- background reader ------------------------------------------------
    def _run(self) -> None:
        try:
            asyncio.run(self._reader())
        except Exception as exc:  # noqa: BLE001
            logger.error(f"ViveLiveSource reader crashed: {exc!r}")

    async def _reader(self) -> None:
        # Imported here (not at module top) so the dependency is only needed on
        # the live path. A MISSING dependency is FATAL — do not silently fall
        # back to NPZ replay while logging "VIVE ENABLED" (operator would think
        # they are teleoperating but the robot replays the clip).
        try:
            import websockets
            from websockets.exceptions import InvalidURI
        except ImportError as exc:
            logger.error(
                f"ViveLiveSource: `websockets` not installed ({exc!r}) — live VIVE teleop "
                "CANNOT run. Install websockets>=11. Aborting the live source."
            )
            self._stop.set()
            return

        backoff = 0.5
        while not self._stop.is_set():
            try:
                async with websockets.connect(self._ws_url, max_size=None) as ws:
                    if not self._connected:
                        logger.info(f"ViveLiveSource: connected to {self._ws_url}")
                        self._connected = True
                    backoff = 0.5
                    while not self._stop.is_set():
                        raw = await asyncio.wait_for(ws.recv(), timeout=1.0)
                        self._ingest(raw)
            except asyncio.TimeoutError:
                continue  # no message in the window; loop and re-check _stop
            except InvalidURI as exc:
                # A malformed URL never becomes valid by retrying — stop loudly
                # so the operator sees the misconfiguration instead of a silent
                # degrade to NPZ replay.
                logger.error(f"ViveLiveSource: invalid WS URL {self._ws_url!r} ({exc!r}) — stopping live source.")
                self._stop.set()
                return
            except Exception as exc:  # noqa: BLE001 — reconnect on transient drops
                self._connected = False
                if not self._stop.is_set():
                    logger.warning(f"ViveLiveSource: WS error ({exc!r}); reconnecting in {backoff:.1f}s")
                    await asyncio.sleep(backoff)
                    backoff = min(backoff * 2, 5.0)

    def _ingest(self, raw: str | bytes) -> None:
        try:
            msg = json.loads(raw)
        except (ValueError, TypeError):
            return
        mode = msg.get("mode")
        # OPTIONAL force command: accept only a finite (3,) or (6,) vector; anything else
        # (absent key, wrong length, NaN/inf) leaves the force feed at None so the policy
        # holds F^cmd=0 (pure position). The pose/joint fields are unaffected — a malformed
        # f_cmd never blocks a valid command frame.
        f_cmd_arr = self._parse_f_cmd(msg.get("f_cmd"))

        # ── DENSE frame: joint targets, no keypoints ─────────────────────────────
        # Handled BEFORE the pose validation, which would otherwise drop it on the
        # missing "pos" key. A dense frame leaves _pos/_orn untouched (None), so a
        # keypoint consumer keeps holding its reference instead of reinterpreting
        # joint angles as positions.
        if mode == "dense":
            jp = self._parse_joint_vec(msg.get("joint_pos"))
            jv = self._parse_joint_vec(msg.get("joint_vel"))
            names = msg.get("joint_names")
            if jp is None or jv is None:
                return
            # The names are MANDATORY, not a nicety: without them the consumer has no
            # honest way to place 29 numbers, and index-aligning two different real
            # joint orders scrambles the body with no error anywhere. Drop the frame.
            if not isinstance(names, list) or len(names) != jp.size or not all(isinstance(n, str) for n in names):
                if not self._joint_names_warned:
                    self._joint_names_warned = True
                    logger.error(
                        "ViveLiveSource: dense frame without a valid 29-name `joint_names` — DROPPING. "
                        "Index-aligning the completion's joint order onto the policy's would silently "
                        "scramble the body. (Warned once.)"
                    )
                return
            with self._lock:
                self._joint_pos = jp
                self._joint_vel = jv
                self._joint_names = tuple(names)
                self._f_cmd = f_cmd_arr
                self._last_t = time.time()
                self._seq = int(msg.get("seq", self._seq + 1))
            return

        # ── KEYPOINT frame (2pt / 3pt / 5pt) ────────────────────────────────────
        pos = msg.get("pos")
        orn = msg.get("orn")
        # Accept the 2pt frame (pos[6] = [L,R]xyz, orn[12] = 2×rot6d), the 3pt frame
        # (pos[9] = [L,R,head]xyz, orn[12] = 3×xyzw-quaternion) and the 5pt frame
        # (pos[15] = [L,R,head,Lfoot,Rfoot]xyz, orn[20] = 5×xyzw-quaternion). The bridge
        # tags each frame with "mode"; we validate the (pos, orn) width PAIR and pass the
        # raw vectors straight to the policy, which slices them per its obs mode. Anything
        # else (control/ack messages, malformed frames) is ignored.
        if not isinstance(pos, list) or not isinstance(orn, list):
            return
        if (len(pos), len(orn)) not in _POSE_WIDTHS:
            return
        # Defense-in-depth: if the bridge tagged a "mode", its widths must agree. A
        # mismatch means a corrupt/spoofed frame — drop it rather than hand a wrong-width
        # vector to the policy. (The policy still slices per its OWN obs mode; this only
        # rejects internally-inconsistent frames.) This is load-bearing for orientation:
        # a 3pt 3×quat and a 2pt 2×rot6d are BOTH 12 wide, so the pos width is the only
        # thing that tells the two reps apart.
        want = _MODE_WIDTHS.get(str(mode))
        if want is not None and (len(pos), len(orn)) != want:
            return
        pos_arr = np.asarray(pos, dtype=np.float32)
        orn_arr = np.asarray(orn, dtype=np.float32)
        # OPTIONAL reference joints (5pt walk skill). Parsed like f_cmd: a malformed value never
        # blocks the pose frame, it just leaves the override at None (warned once, because unlike
        # a missing f_cmd a MALFORMED ref_joints is a sender bug the operator should hear about).
        ref = self._parse_ref_joints(msg.get("ref_joints"))
        if ref is None and msg.get("ref_joints") is not None and not self._ref_joints_warned:
            self._ref_joints_warned = True
            logger.warning(
                "ViveLiveSource: a keypoint frame carried a malformed `ref_joints` (want "
                f"{{names[{_NUM_DOFS}], pos[{_NUM_DOFS}], vel[{_NUM_DOFS}]}}, finite) — IGNORING it; the pose "
                "is applied and command_lower_body keeps the frozen clip. (Warned once.)"
            )
        with self._lock:
            self._pos = pos_arr
            self._orn = orn_arr
            self._f_cmd = f_cmd_arr
            if ref is None:
                self._ref_joint_pos = self._ref_joint_vel = self._ref_joint_names = None
            else:
                self._ref_joint_pos, self._ref_joint_vel, self._ref_joint_names = ref
            self._last_t = time.time()
            self._seq = int(msg.get("seq", self._seq + 1))

    @staticmethod
    def _parse_f_cmd(f_cmd: object) -> np.ndarray | None:
        """A finite flat (3,) or (6,) force command, else None (→ policy holds F^cmd=0)."""
        if not isinstance(f_cmd, list) or len(f_cmd) not in (3, 6):
            return None
        try:
            cand = np.asarray(f_cmd, dtype=np.float32)
        except (ValueError, TypeError):
            return None  # non-numeric / nested / ragged → ignore f_cmd, keep the pose
        # A nested/ragged list can coerce to a >1-D or object array; require a flat
        # numeric vector. A bad f_cmd must NEVER throw here (it would abort the reader
        # loop and drop the valid COMMAND frame → a WS reconnect); leave the force at None.
        if cand.ndim != 1 or not np.all(np.isfinite(cand)):
            return None
        return cand

    @staticmethod
    def _parse_joint_vec(v: object) -> np.ndarray | None:
        """A finite flat 29-vector of joint values, else None. Same never-throw contract
        as :meth:`_parse_f_cmd`: a malformed dense frame is dropped, not raised."""
        if not isinstance(v, list) or len(v) != _NUM_DOFS:
            return None
        try:
            arr = np.asarray(v, dtype=np.float32)
        except (ValueError, TypeError):
            return None
        if arr.ndim != 1 or not np.all(np.isfinite(arr)):
            return None
        return arr

    @classmethod
    def _parse_ref_joints(cls, v: object) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]] | None:
        """``{"names": [29 str], "pos": [29], "vel": [29]}`` -> ``(pos, vel, names)``, else None.

        Never-throw, like :meth:`_parse_f_cmd`. The names are mandatory for the same reason the
        dense frame's are: 29 numbers without names cannot honestly be placed on a robot whose
        ``dof_names`` order is a different real order."""
        if not isinstance(v, dict):
            return None
        pos = cls._parse_joint_vec(v.get("pos"))
        vel = cls._parse_joint_vec(v.get("vel"))
        names = v.get("names")
        if pos is None or vel is None:
            return None
        if (not isinstance(names, list) or len(names) != _NUM_DOFS
                or not all(isinstance(n, str) for n in names) or len(set(names)) != _NUM_DOFS):
            return None
        return pos, vel, tuple(names)
