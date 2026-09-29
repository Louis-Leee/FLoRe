#!/usr/bin/env python3
"""Live Bota F/T → WebSocket bridge for the sim2real hybrid / force-ext deploy.

Stands between the physical Bota force/torque sensor(s) and the offboard
``run_policy`` process: reads one or two Bota sensors, runs the SAME
gravity/inertia payload compensation the validated collection/monitor stack uses,
applies a tare (startup auto-tare + on-demand ``t``-key re-tare), and publishes the
per-hand SENSOR-frame contact force over a localhost WebSocket that
``ForceLiveSource`` (in the policy process) consumes.

Why a separate process/bridge (user decision 2026-07-13): the Bota gen-A toolkit
(``bota_driver`` / ``bota_payload_utils`` / ``bota_control`` / ``bota_tf_utils``)
lives in a dedicated ``bota`` env, and ``read_frame_blocking`` holds the GIL
~1 ms/call, so it must not run inside the RL loop's process. This bridge is
entirely inside ``holosoma_extensions`` and reads NOTHING from ``UMI_collection``
— it re-implements the tiny gen-A wrench-assembly shim (below) so it is
self-contained. Run it in the ``bota`` env; run the policy in ``hsinference``.

Wire frame (text JSON, consumed by ForceLiveSource):

    {"type": "ft", "seq": int, "t": wall_seconds,
     "left":  [Fx, Fy, Fz] | null,    # SENSOR-frame Newtons, compensated + tared
     "right": [Fx, Fy, Fz] | null}    # null for a hand with no sensor configured

The published force is the compensated (tool gravity + inertia removed), tared,
anti-alias-filtered wrench in the Bota SENSOR frame; the policy rotates
sensor→wrist→pelvis itself. Torque is dropped (the deploy force obs is 3-D; moment
channels are dormant, D8).

OPT-IN EXTRA CURVES (``--emit-raw`` / ``FT_EMIT_RAW=1``, for the teleop recorder):
when set, the frame ALSO carries the two OTHER force curves so the recorder can log
all three and the replay viz can toggle raw ↔ gravity-only ↔ RNEA:

    "left_raw":  [Fx,Fy,Fz,Tx,Ty,Tz] | null,  # TRULY-RAW sensor wrench (no comp, no tare)
    "right_raw": [...] | null,
    "left_grav":  [Fx,Fy,Fz] | null,  # gravity-only Bota force (null if unavailable / bota-mode)
    "right_grav": [...] | null

These are SEPARATE keys — ``ForceLiveSource`` reads only ``left``/``right`` and ignores
unknown keys, so ``--emit-raw`` is fully back-compatible and byte-identical to the policy.
DEFAULT OFF ⇒ neither the frame nor the read-loop timing changes on a normal deploy.

PIPELINE 6-DoF FOR THE HIGH-LEVEL PLANNER (``--pipeline`` / ``FT_PIPELINE=1``): the
S1 MuSe-on-UVA force policy (the high-level planner, ``scripts/muse_force_serving_*``)
was TRAINED on the 6-DoF **gravity-only (bota), tared** wrench in the SENSOR frame
(the converted zarr's ``robot0_ft_wrench``; |F| p50≈0.5 N confirms compensated+tared).
The low-level ``left``/``right`` stream above is only 3-DoF force and may be RNEA — so it
is neither 6-DoF nor necessarily in-distribution for the planner. ``--pipeline`` adds a
SEPARATE 6-DoF key per hand that always matches the planner's training distribution,
REGARDLESS of the low-level ``--ft-algo``:

    "left_pipe":  [Fx,Fy,Fz,Tx,Ty,Tz] | null,  # gravity-only (bota), TARED, anti-aliased,
    "right_pipe": [...] | null                   #   SENSOR frame — for the HL planner obs

The WS server ALREADY broadcasts to every connected client (one shared ``clients`` set),
so the low-level controller and the high-level planner can BOTH connect to this one port
simultaneously — the serial is still opened exactly once. ``--pipeline`` only ADDS keys;
the policy-facing ``left``/``right`` and the read-loop timing stay byte-identical (the
low-level ignores the extra keys). When the active algo is already ``bota``, ``left_pipe``
is the same gravity-only engine as the published force, just carried at full 6-DoF; when
the active algo is ``rnea``, a SECOND, independent gravity-only engine (its own tare +
anti-alias) produces ``*_pipe`` without perturbing the RNEA stream the low-level reads.
DEFAULT OFF ⇒ byte-identical. (Composes with ``--emit-raw``; the two are independent.)

COMPENSATION ALGORITHM (``--ft-algo``): ``bota`` = the official gravity-only
compensator (this rig's Bota config); ``rnea`` = an independent single-body
Newton-Euler DYNAMIC compensator (vendored numpy port of the UMI ``bota_demo_py``
reference in ``utils/rnea_compensator.py``) that ALSO removes the tool's inertial
reaction ``m·a`` — the accel-proportional phantom gravity-only leaves in
``force_meas`` when the arm moves, which drove the escalating startup oscillation.

ANTI-ALIAS OUTPUT STAGE (``--antialias-hz``, default 12): a dt-aware first-order
low-pass on the compensated force at the READ rate, BEFORE the 50 Hz policy samples
it. A slow controller reading the raw ~kHz compensated wrench would alias its
high-frequency residual into the control band (phantom contact / shake); 12 Hz is
below the 25 Hz policy Nyquist and its ~13 ms group delay is under one 20 ms tick.

FREQUENCY PIPELINE (why the two downsamples below are safe): sensor ~1 kHz →
[``--read-hz`` throttle, default 250] → [12 Hz anti-alias LP] → [WS publish
``--rate-hz`` 100] → [policy ``ForceLiveSource`` samples at the 50 Hz control tick].
The ONE stage that decimates WITHOUT a preceding digital anti-alias filter is the
1 kHz→250 Hz read throttle (it drops driver-buffer frames): this relies on the
Bota's INTERNAL analog/app-submode filter band-limiting the wrench below the 125 Hz
read-Nyquist — the same assumption the UMI reference demo makes when it reads at its
loop rate. Every downsample AFTER the 12 Hz LP (250→100→50) is far above 2·12 = 24 Hz,
so no aliasing there. If a future rig streams meaningful content above ~125 Hz, run
the reader unthrottled (``--read-hz 0``, 1 kHz) so the 12 Hz LP is the sole
band-limiter — but note two unthrottled 2-sensor read loops can starve the WS
(see ``--read-hz``).

Usage (in the bota env):

    # Bota gravity-only (default), single sensor:
    python -m wbt_wrappers_inference.force_bridge \\
      --right-serial <BOTA_SERIAL> \\
      --right-compensator /path/to/payload_compensator_right.json \\
      --driver-config /path/to/bota_binary.json --port 8771

    # RNEA dynamic compensation, bimanual (needs a per-side rnea_compensator_*.json):
    python -m wbt_wrappers_inference.force_bridge --ft-algo rnea \\
      --left-serial  <L> --left-compensator  .../payload_compensator_left.json \\
      --left-rnea-config  .../rnea_compensator_left.json \\
      --right-serial <R> --right-compensator .../payload_compensator_right.json \\
      --right-rnea-config .../rnea_compensator_right.json \\
      --driver-config /path/to/bota_binary.json --port 8771

    # single sensor with no serial pinning: --right-compensator only (auto-port).

Re-tare BOTH sensors (do it in FREE SPACE — a contact force present at tare bakes
in as a constant bias) three ways: (1) the joystick ``R2`` button (or keyboard
``T``) during a ``run_policy`` session, which sends ``{"cmd":"tare"}`` over the WS;
(2) any WS client sending ``{"cmd":"tare"}``; (3) ``t``+Enter on this bridge's
stdin. Ctrl-C to stop.

⚠️ TARE + FRAME SIGN are hardware unknowns until the gantry check: after launch,
push the gripper along wrist +x and confirm the policy-side startup log shows
F_pelvis +x rising (flip ``HOLOSOMA_FORCE_MEAS_SIGN`` on the policy if inverted).
"""

from __future__ import annotations

import argparse
import asyncio
import glob
import json
import os
import sys
import tempfile
import threading
import time

import numpy as np

_BOTA_BY_ID_GLOB = "/dev/serial/by-id/*Bota_Systems*Force_Torque_Sensor*-if00"
# wrench_frame → imu_frame offset (m) for the M8-series (matches the Bota demo +
# the validated rig's bota_compat.IMU_OFFSET). Leave as-is unless Bota specifies
# a different value for your sensor.
_IMU_OFFSET = (0.0, 0.0, -0.0257)
# A transient bad-dt makes update() return FatalError for a sample or two, then
# recovers on the next clean timestamp. Tolerate this many CONSECUTIVE fatals
# (~0.2 s at 1 kHz) before declaring the stream broken (mirrors bota_compat).
_FATAL_TOLERANCE = 200
# Anti-alias output-stage cutoff (Hz), applied to the compensated force at the read
# rate BEFORE the policy samples it at 50 Hz (see _HandReader / rnea_compensator).
# 12 Hz < 50/2 Nyquist, group delay ~13 ms < one 20 ms control period. 0 = disabled.
_DEFAULT_ANTIALIAS_HZ = 12.0


def _import_rnea_compensator():
    """Import the numpy-only ``utils/rnea_compensator`` module WITHOUT triggering the
    ``wbt_wrappers_inference`` package ``__init__`` (which pulls in holosoma/termcolor
    that are absent from the bota env this bridge runs in). We add the ``utils`` dir to
    ``sys.path`` and import the file as a TOP-LEVEL module, so neither the package nor
    ``utils/__init__`` runs. Kept lazy so the Bota-only path never needs it.
    """
    utils_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "utils")
    if utils_dir not in sys.path:
        sys.path.insert(0, utils_dir)
    import rnea_compensator  # noqa: PLC0415 — standalone numpy-only module

    return rnea_compensator


# --------------------------------------------------------------------------- #
# gen-A (bota_payload_utils >= 1.0) wrench-assembly shim.
#
# Self-contained mirror of UMI_collection/force_torque/bota_compat.py (imported
# from NOTHING outside this repo, per the "no UMI edits/deps" constraint). The
# 1.0.2 toolkit dropped the create_ft_sensor_hwi wrapper: the caller now drives
# the BotaDriver lifecycle and assembles the 12-vector [wrench(6), imu_in_sensor(6)]
# itself, transforming the IMU into the sensor frame via bota_tf_utils.BotaTfImu.
# `bota_control`/`bota_tf_utils` import only where the gen-A toolkit is installed
# (the bota env), so these live behind the lazy imports in FtChain.__init__.
# --------------------------------------------------------------------------- #
def _rc_is_fatal(rc, _rc_enum) -> bool:
    return rc == _rc_enum.FatalError


def _rc_is_ok(rc, _rc_enum) -> bool:
    return rc == _rc_enum.OK


class FtChain:
    """IMU-frame→sensor-frame transform + reusable buffers for the 12-vector.

    Owns the ``BotaTfImu`` block (keeps internal filter state, so make ONE per
    session). ``assemble(frame)`` returns ``[wrench(6), imu_in_sensor(6)]`` to feed
    ``compensator.update``. Buffers are reused; consume before the next call.
    """

    def __init__(self, imu_offset=_IMU_OFFSET):
        import bota_control as bc
        import bota_tf_utils as tf

        self._rc = bc.BotaControlReturnCode
        cfg = tf.BotaTfConfig()
        cfg.name = "imuF_in_imuF_to_sensor_in_sensor"
        cfg.policy = bc.BotaControlBlockPolicy.StrictRt
        block = tf.BotaTfImu(cfg)
        block.set_input_reference_point("fixed_frame", np.asarray(imu_offset, dtype=np.float64))
        block.set_input_expression_axes("fixed_frame", np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64))
        block.set_output_reference_point("fixed_frame", np.array([0.0, 0.0, 0.0], dtype=np.float64))
        block.set_output_expression_axes("fixed_frame", np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64))
        self._tf = block
        self._imu_in = np.zeros(6, dtype=np.float64)
        self._imu_out = np.zeros(6, dtype=np.float64)
        self._raw12 = np.zeros(12, dtype=np.float64)
        self.comp12 = np.zeros(12, dtype=np.float64)
        self.last_tf_rc = self._rc.OK

    def assemble(self, frame) -> np.ndarray:
        self._imu_in[:3] = frame.acceleration
        self._imu_in[3:] = frame.angular_rate
        self.last_tf_rc = self._tf.update(self._imu_in, self._imu_out)
        self._raw12[:3] = frame.force
        self._raw12[3:6] = frame.torque
        self._raw12[6:] = self._imu_out
        return self._raw12

    def tf_is_fatal(self) -> bool:
        return self.last_tf_rc == self._rc.FatalError

    def rc_is_fatal(self, rc) -> bool:
        return _rc_is_fatal(rc, self._rc)

    def rc_is_ok(self, rc) -> bool:
        return _rc_is_ok(rc, self._rc)


def _find_bota_ports() -> list[str]:
    return sorted(glob.glob(_BOTA_BY_ID_GLOB))


def _resolve_driver_config(template_path: str, port: str | None) -> str:
    """Return a driver-config path whose ``com_port`` is ``port`` (temp copy;
    committed template untouched). ``port=None`` → template unchanged."""
    if port is None:
        return template_path
    with open(template_path) as f:
        cfg = json.load(f)
    cfg["driver_config"]["communication_interface_params"]["com_port"] = port
    tmp = tempfile.NamedTemporaryFile("w", suffix=".json", prefix="bota_cfg_", delete=False)
    json.dump(cfg, tmp)
    tmp.close()
    return tmp.name


# --------------------------------------------------------------------------- #
# Compensation-algorithm adapter (Bota official gravity-only vs independent RNEA).
#
# Self-contained mirror of bota_ft_viz._AlgoCompensator, minus the live GUI switch
# (the deploy picks ONE algorithm at launch via --ft-algo / per-hand override, with
# a Bota fallback on any RNEA-config failure). Both engines consume the SAME 12-vector
# FtChain.assemble builds (raw wrench + IMU transported to the SENSOR ORIGIN) and write
# the compensated wrench into the SAME chain.comp12 buffer, so the reader loop is
# algorithm-agnostic. RNEA's string status ("OK"/"DEGRADED") is normalized to the
# reader's BotaControlReturnCode so rc_is_fatal/rc_is_ok are untouched.
#
# CRITICAL INVARIANT (see rnea_compensator + RNEA_INTEGRATION_DESIGN.md): the RNEA
# engine is built with imu_offset=[0,0,0] (build_rnea_from_json default) because the
# 12-vector IMU is ALREADY at the sensor origin — re-applying the lever arm would
# double-count it.
# --------------------------------------------------------------------------- #
class _AlgoCompensator:
    """Holds the Bota engine (always) + (when algo='rnea' and a per-side RNEA config
    is given) the RNEA engine, exposing the reader loop ONE algorithm-agnostic
    interface. Any RNEA build failure falls back to Bota with a status note."""

    def __init__(self, side: str, bota_comp, rnea_config_path: str | None, algo: str, chain: "FtChain"):
        self.side = side
        self._bota = bota_comp
        self._rnea = None
        self._chain = chain  # for BotaControlReturnCode + fatal/ok normalization
        self.status_note = ""
        self.active = "bota"
        if str(algo).lower() == "rnea" and rnea_config_path:
            self._build_rnea(rnea_config_path)

    def _build_rnea(self, path: str) -> None:
        try:
            mod = _import_rnea_compensator()
            self._rnea = mod.build_rnea_from_json(path, zero_imu_offset=True)
            self.active = "rnea"
            if mod.read_wrench_sign_verified(path) is False:
                self.status_note = (
                    "RNEA wrench_sign UNVERIFIED for this hand — validate with a static tilt "
                    "(compensated |F| should stay ~0 as you reorient; if it grows, flip the sign)."
                )
            print(
                f"[force-bridge:{self.side}] RNEA dynamic compensation ACTIVE (from {os.path.basename(path)})"
                + (f"  ⚠ {self.status_note}" if self.status_note else ""),
                flush=True,
            )
        except Exception as exc:  # noqa: BLE001 — never let a bad RNEA config kill the reader
            self._rnea = None
            self.active = "bota"
            self.status_note = f"RNEA unavailable ({type(exc).__name__}); using Bota gravity-only."
            print(f"[force-bridge:{self.side}] WARNING: {self.status_note} ({exc!r})", flush=True)

    def update(self, timestamp: float, raw12: np.ndarray, comp12: np.ndarray):
        """Compensate in place into ``comp12``; return a BotaControlReturnCode."""
        if self.active == "rnea" and self._rnea is not None:
            status = self._rnea.update(timestamp, raw12, comp12)
            rc = self._chain._rc  # bota_control.BotaControlReturnCode
            return rc.OK if status == "OK" else rc.Degraded
        return self._bota.update(timestamp, raw12, comp12)


# --------------------------------------------------------------------------- #
# Per-hand reader thread.
# --------------------------------------------------------------------------- #
class _HandReader(threading.Thread):
    """Reads ONE Bota sensor, runs the payload compensator (Bota gravity-only OR the
    independent RNEA dynamic compensator), applies the anti-alias output stage, and
    holds the latest compensated + tared SENSOR-frame force. Mirrors bota_ft_viz's
    compensated loop (tare + engine adapter), minus the viz/orientation bits.

    The published force = LP(comp_force) − force_offset, where comp_force is the
    gravity(+inertia)-compensated wrench and LP is the ~12 Hz anti-alias filter applied
    at the read rate so the 50 Hz policy sample carries no aliased high-frequency
    residual. ``latest_force()`` returns a (3,) copy or None if no OK sample yet.
    """

    def __init__(
        self,
        side: str,
        driver_config: str,
        compensator_config: str,
        port: str | None,
        read_hz: float,
        *,
        algo: str = "bota",
        rnea_config: str | None = None,
        antialias_hz: float = _DEFAULT_ANTIALIAS_HZ,
        emit_raw: bool = False,
        pipeline: bool = False,
    ):
        super().__init__(daemon=True, name=f"bota-{side}")
        self.side = side
        self._driver_config = driver_config
        self._compensator_config = compensator_config
        self._port = port
        self._read_period = (1.0 / read_hz) if read_hz > 0 else 0.0
        self._algo = str(algo).lower()
        self._rnea_config = rnea_config
        self._antialias_hz = float(antialias_hz)
        # OPT-IN (recorder only): ALSO expose the truly-RAW sensor wrench and the
        # gravity-only Bota-compensated force alongside the published (RNEA, tared,
        # anti-aliased) `self._force`, so the recorder can log all three curves and the
        # replay viz can toggle raw ↔ gravity-only ↔ RNEA. DEFAULT OFF ⇒ the extra
        # per-read compensator work never runs on the normal deploy, so the read-loop
        # timing + the published `left`/`right` are byte-identical to before.
        self._emit_raw = bool(emit_raw)
        # OPT-IN (high-level planner): ALSO expose a 6-DoF gravity-only (bota), TARED,
        # anti-aliased wrench (self._pipe) in the SENSOR frame — the exact distribution
        # the S1 MuSe-on-UVA force policy trained on — REGARDLESS of the active algo.
        # DEFAULT OFF ⇒ never built, so the read loop + published left/right are
        # byte-identical to before. See the module docstring's PIPELINE section.
        self._pipeline = bool(pipeline)
        self._lock = threading.Lock()
        self._force: np.ndarray | None = None  # LP(compensated) − offset, sensor frame
        # Extra faithful curves (only maintained when _emit_raw). Sensor frame, same
        # sign; NOT anti-aliased (the true per-read value) and NOT live-tared.
        self._raw_wrench: np.ndarray | None = None  # (6,) raw [Fx,Fy,Fz,Tx,Ty,Tz]
        self._grav_force: np.ndarray | None = None  # (3,) gravity-only Bota force
        self._grav_ok = True  # False after a gravity-only compute error (best-effort)
        # HL-planner 6-DoF pipeline curve (only maintained when _pipeline): gravity-only,
        # TARED, anti-aliased. Sensor frame [Fx,Fy,Fz,Tx,Ty,Tz]. It rides a DEDICATED
        # gravity-only bota engine (uniform for bota/rnea active modes) plus a SOFTWARE
        # 6-DoF offset captured over the same free-space window as the active tare — so
        # the pipe curve is zeroed at START and on every {"cmd":"tare"} in lockstep with
        # the low-level's force, matching the tared gravity-only training distribution.
        self._force_offset = np.zeros(3, dtype=np.float64)  # (published force is self-tared; vestigial 0)
        self._pipe: np.ndarray | None = None  # (6,) gravity-only, tared, anti-aliased
        self._pipe_ok = True  # False after a pipeline compute error (best-effort)
        self._pipe_offset = np.zeros(6, dtype=np.float64)  # software 6-DoF tare for the pipe
        # Software-tare capture: after each tare (startup + {"cmd":"tare"}) we accumulate
        # the first N gravity-only samples (in free space, per the tare contract) and set
        # `_pipe_offset` to their mean, so the pipe curve is zeroed in lockstep with the
        # low-level force. During capture the pipe publishes null (no un-tared leak).
        self._pipe_capture_remaining = 0
        self._pipe_capture_acc = np.zeros(6, dtype=np.float64)
        self._pipe_capture_n = 0
        self._pipe_capture_len = 0  # set in run() from read_hz
        self._aa = None  # AntiAliasFilter (built in run(), reset on each tare)
        self._aa_pipe = None  # 6-DoF AntiAliasFilter for the pipe curve (built in run())
        self._tare_request = threading.Event()
        self._stop = threading.Event()
        self.connected = False
        self.n_frames = 0
        self.active_algo = self._algo
        self.last_error = ""

    def latest_force(self) -> np.ndarray | None:
        with self._lock:
            return None if self._force is None else self._force.copy()

    def latest_raw_wrench(self) -> np.ndarray | None:
        """(6,) truly-RAW sensor wrench [F(3), T(3)] or None. Only populated when
        ``emit_raw`` is set (recorder session); None otherwise."""
        with self._lock:
            return None if self._raw_wrench is None else self._raw_wrench.copy()

    def latest_grav_force(self) -> np.ndarray | None:
        """(3,) gravity-only Bota-compensated force or None. Only populated when
        ``emit_raw`` is set AND the gravity-only compute is available."""
        with self._lock:
            return None if self._grav_force is None else self._grav_force.copy()

    def latest_pipe(self) -> np.ndarray | None:
        """(6,) gravity-only, TARED, anti-aliased sensor-frame wrench [F(3),T(3)] for
        the high-level planner, or None. Only populated when ``pipeline`` is set AND the
        gravity-only compute is available (the S1 force-policy training distribution)."""
        with self._lock:
            return None if self._pipe is None else self._pipe.copy()

    def latest_all(self) -> "tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]":
        """Return (force, raw_wrench, grav_force) as a COHERENT single-read snapshot —
        all three under ONE lock acquisition, so the RNEA/raw/gravity curves published in
        one WS frame come from the SAME physical sensor read (not up to three different
        reads if fetched via the three separate accessors). Used by the emit-raw publish."""
        with self._lock:
            f = None if self._force is None else self._force.copy()
            rw = None if self._raw_wrench is None else self._raw_wrench.copy()
            gf = None if self._grav_force is None else self._grav_force.copy()
            return f, rw, gf

    def request_tare(self) -> None:
        self._tare_request.set()

    def _arm_pipe_capture(self) -> None:
        """(Re)start the pipe software-tare capture window: reset the pipe AA and mark
        the next ``_pipe_capture_len`` gravity-only samples for the offset mean. During
        the window ``latest_pipe()`` returns None (we hold the previous offset's output
        only after capture completes). No-op unless --pipeline."""
        if not self._pipeline:
            return
        with self._lock:
            self._pipe_capture_remaining = self._pipe_capture_len
            self._pipe_capture_acc[:] = 0.0
            self._pipe_capture_n = 0
            self._pipe = None
        if self._aa_pipe is not None:
            self._aa_pipe.reset()

    def stop(self) -> None:
        self._stop.set()

    def run(self) -> None:
        try:
            import bota_driver
            import bota_payload_utils as bpu
        except ImportError as exc:  # pragma: no cover - env-specific
            self.last_error = f"bota toolkit not importable: {exc!r}"
            print(f"[force-bridge:{self.side}] FATAL: {self.last_error}", flush=True)
            return

        # Build the anti-alias output-stage filter (numpy-only; safe in the bota env).
        # A build failure must NOT kill the stream — degrade to pass-through.
        try:
            self._aa = _import_rnea_compensator().AntiAliasFilter(self._antialias_hz, size=3)
        except Exception as exc:  # noqa: BLE001
            self._aa = None
            print(
                f"[force-bridge:{self.side}] WARNING: anti-alias filter unavailable ({exc!r}); pass-through.",
                flush=True,
            )

        drv_cfg = _resolve_driver_config(self._driver_config, self._port)
        driver = bota_driver.BotaDriver(drv_cfg)
        bota_comp = bpu.BotaPayloadCompensator(self._compensator_config)
        try:
            chain = FtChain(_IMU_OFFSET)
        except Exception as exc:  # noqa: BLE001 - gen-A toolkit missing/broken
            self.last_error = f"FtChain init failed: {exc!r}"
            print(f"[force-bridge:{self.side}] FATAL: {self.last_error}", flush=True)
            return
        comp = _AlgoCompensator(self.side, bota_comp, self._rnea_config, self._algo, chain)
        self.active_algo = comp.active
        # OPT-IN extra-curve capture (recorder only): a SEPARATE gravity-only Bota
        # engine so we can log the gravity-only curve alongside the active (RNEA)
        # one WITHOUT disturbing the active compensator's buffer/state. Only built
        # when the active algo is RNEA (in 'bota' mode the published `left`/`right`
        # IS already the gravity-only curve, so a second engine would be redundant).
        # Its own 12-vector scratch buffer; best-effort — any failure flips
        # self._grav_ok False and the gravity curve simply reports null thereafter,
        # never touching the published RNEA stream the policy consumes.
        # A SEPARATE gravity-only bota engine is needed ONLY in RNEA-active mode (there
        # the active comp12 is NOT gravity-only). It is SHARED by the recorder's
        # --emit-raw gravity curve (needs [:3]) and the HL --pipeline 6-DoF (needs [:6]):
        # run ONCE per read into grav12 (a second .update() would double-advance the
        # compensator's internal state). In BOTA-active mode the active comp12 IS already
        # gravity-only, so no second engine is built and both curves read comp12 directly.
        # Best effort: any failure disables the aux curves and never touches the RNEA
        # stream the low-level policy reads.
        _want_grav_engine = (self._emit_raw or self._pipeline) and comp.active == "rnea"
        grav_comp = None
        grav12 = None
        if _want_grav_engine:
            try:
                grav_comp = bpu.BotaPayloadCompensator(self._compensator_config)
                grav12 = np.zeros(12, dtype=np.float64)
            except Exception as exc:  # noqa: BLE001
                grav_comp = None
                self._grav_ok = False
                self._pipe_ok = False
                print(
                    f"[force-bridge:{self.side}] WARNING: gravity-only engine unavailable ({exc!r}); "
                    "aux curves (raw-grav / pipe) disabled.",
                    flush=True,
                )
        # HL pipeline 6-DoF anti-alias filter (numpy-only; degrade to pass-through).
        if self._pipeline:
            try:
                self._aa_pipe = _import_rnea_compensator().AntiAliasFilter(self._antialias_hz, size=6)
            except Exception as exc:  # noqa: BLE001
                self._aa_pipe = None
                print(
                    f"[force-bridge:{self.side}] WARNING: pipe anti-alias unavailable ({exc!r}); pass-through.",
                    flush=True,
                )
            # ~0.3 s of samples for the software-tare mean (bounded ≥1). At the 250 Hz
            # default that is ~75 samples; short enough not to stall START, long enough
            # to average out sensor noise in the free-space tare window.
            rh = (1.0 / self._read_period) if self._read_period else 250.0
            self._pipe_capture_len = max(1, int(round(0.3 * rh)))
        if self._emit_raw:
            print(
                f"[force-bridge:{self.side}] extra-curve capture ON (raw"
                + ("+gravity-only alongside RNEA" if comp.active == "rnea" else " only; active algo already gravity-only")
                + ").",
                flush=True,
            )
        if self._pipeline:
            print(
                f"[force-bridge:{self.side}] HL pipeline 6-DoF ON (gravity-only, tared, anti-aliased; "
                f"active={comp.active}) → left_pipe/right_pipe.",
                flush=True,
            )
        try:
            if not driver.configure():
                raise RuntimeError("driver.configure() failed")
            if not driver.activate():
                raise RuntimeError("driver.activate() failed")
            self._tare_active(driver, comp, chain)  # Bota async warm-up OR RNEA sync average
            self._arm_pipe_capture()  # capture the pipe software-tare over the next window
            self.connected = True
            print(
                f"[force-bridge:{self.side}] streaming (port={self._port or 'auto'}, "
                f"algo={comp.active}, anti-alias={'off' if not (self._aa and self._aa.enabled) else f'{self._antialias_hz:.0f}Hz'})",
                flush=True,
            )

            consec_fatal = 0
            last_t = time.perf_counter()
            while not self._stop.is_set():
                if self._tare_request.is_set():
                    self._tare_request.clear()
                    self._tare_active(driver, comp, chain)
                    with self._lock:
                        self._force_offset[:] = 0.0  # re-tare zeroes the signal itself
                    if self._aa is not None:
                        self._aa.reset()  # re-seed the LP from the fresh zero (no pre-tare blend)
                    self._arm_pipe_capture()  # re-capture the pipe software-tare in lockstep
                    last_t = time.perf_counter()
                    print(f"[force-bridge:{self.side}] re-tared", flush=True)
                    continue

                frame = driver.read_frame_blocking()
                raw12 = chain.assemble(frame)
                rc = comp.update(time.perf_counter(), raw12, chain.comp12)
                if chain.rc_is_fatal(rc) or chain.tf_is_fatal():
                    consec_fatal += 1
                    if consec_fatal > _FATAL_TOLERANCE:
                        raise RuntimeError(f"compensator FatalError x{consec_fatal} (stream broken)")
                    last_t = time.perf_counter()  # advance the LP clock even on a skip (see below)
                    continue
                consec_fatal = 0
                if not chain.rc_is_ok(rc):
                    # Stale/Degraded → comp12 is zeros/reduced; skip (don't publish
                    # a bogus force). Fresh-again resumes automatically. Advance last_t
                    # here too so the NEXT good sample's dt is ~one read period, not the
                    # accumulated gap: a stale-large dt makes the anti-alias a=dt/(tau+dt)
                    # →1 (pass-through), letting one unfiltered spike through on recovery.
                    last_t = time.perf_counter()
                    continue
                cmp_f = np.array(chain.comp12[:3], dtype=np.float64)
                now = time.perf_counter()
                dt = now - last_t
                last_t = now
                # Anti-alias low-pass at the read rate BEFORE the policy samples at
                # 50 Hz: high-frequency residual (IMU/α noise, contact chatter) above
                # the control Nyquist would otherwise alias into the band and shake.
                if self._aa is not None:
                    cmp_f = self._aa.update(cmp_f, dt)
                # OPT-IN aux curves (recorder --emit-raw AND/OR HL --pipeline). Best-effort
                # and fully isolated from the published `left`/`right` (=cmp_f below): any
                # error degrades that curve to null and never affects the policy stream.
                #
                # The gravity-only 6-DoF wrench `grav_w6` is the SHARED source for BOTH the
                # recorder's gravity curve (needs [:3]) and the HL pipe (needs [:6]):
                #   * rnea-active: run the separate gravity engine ONCE into grav12[:6].
                #   * bota-active: the active comp12 IS gravity-only → reuse chain.comp12[:6].
                # Running the engine once avoids double-advancing its internal state.
                raw_wrench = grav_force = None
                grav_w6 = None  # (6,) gravity-only sensor-frame wrench, pre-tare, pre-AA
                if self._emit_raw or self._pipeline:
                    if comp.active == "rnea":
                        if grav_comp is not None and grav12 is not None and (self._grav_ok or self._pipe_ok):
                            try:
                                grc = grav_comp.update(now, raw12, grav12)
                                if chain.rc_is_ok(grc):
                                    grav_w6 = np.array(grav12[:6], dtype=np.float64)
                            except Exception as exc:  # noqa: BLE001 - isolate; never touch the RNEA stream
                                self._grav_ok = False
                                self._pipe_ok = False
                                print(
                                    f"[force-bridge:{self.side}] gravity-only engine error ({exc!r}); "
                                    "disabling aux curves.",
                                    flush=True,
                                )
                    elif self._grav_ok or self._pipe_ok:
                        # bota-active: the (already-tared, pre-AA) active comp IS gravity-only.
                        # Isolated read: never let it propagate to the outer except (would
                        # kill the policy-facing left/right stream).
                        try:
                            grav_w6 = np.array(chain.comp12[:6], dtype=np.float64)
                        except Exception as exc:  # noqa: BLE001
                            self._grav_ok = False
                            self._pipe_ok = False
                            print(
                                f"[force-bridge:{self.side}] gravity-only read error ({exc!r}); "
                                "disabling aux curves.",
                                flush=True,
                            )

                if self._emit_raw:
                    raw_wrench = np.array(raw12[:6], dtype=np.float64)  # [F(3), T(3)] sensor frame
                    grav_force = None if grav_w6 is None else grav_w6[:3].copy()

                # HL pipe: software-tare (mean over the free-space capture window) then AA.
                # Fully isolated (best-effort): ANY error here degrades the pipe to null and
                # NEVER propagates to the reader's outer except — the published left/right the
                # low-level policy reads must not be affected by a pipe fault (invariant #4).
                pipe6 = None
                if self._pipeline and grav_w6 is not None and self._pipe_ok:
                    try:
                        if self._pipe_capture_remaining > 0:
                            # Still capturing the tare mean — accumulate, publish nothing yet.
                            self._pipe_capture_acc += grav_w6
                            self._pipe_capture_n += 1
                            self._pipe_capture_remaining -= 1
                            if self._pipe_capture_remaining == 0 and self._pipe_capture_n > 0:
                                self._pipe_offset = self._pipe_capture_acc / self._pipe_capture_n
                                if self._aa_pipe is not None:
                                    self._aa_pipe.reset()  # seed AA from the freshly-zeroed signal
                        else:
                            tared = grav_w6 - self._pipe_offset
                            pipe6 = self._aa_pipe.update(tared, dt) if self._aa_pipe is not None else tared
                    except Exception as exc:  # noqa: BLE001 - isolate; never touch the policy stream
                        self._pipe_ok = False
                        pipe6 = None
                        print(
                            f"[force-bridge:{self.side}] pipe curve error ({exc!r}); disabling it.",
                            flush=True,
                        )
                with self._lock:
                    self._force = cmp_f - self._force_offset
                    if self._emit_raw:
                        self._raw_wrench = raw_wrench
                        self._grav_force = grav_force
                    if self._pipeline:
                        self._pipe = None if pipe6 is None else np.asarray(pipe6, dtype=np.float64)
                    self.n_frames += 1
                if self._read_period:
                    time.sleep(self._read_period)
        except Exception as exc:  # noqa: BLE001
            self.last_error = repr(exc)
            print(f"[force-bridge:{self.side}] stream error: {exc!r}", flush=True)
        finally:
            self.connected = False
            try:
                driver.deactivate()
                driver.shutdown()
            except Exception:  # noqa: BLE001
                pass

    def _tare_active(self, driver, comp: "_AlgoCompensator", chain) -> None:
        """Re-zero the ACTIVE engine in free space.

        RNEA: synchronous — average predict-vs-raw over a ~1 s static window and set
        the bias via ``tare_from_prediction`` (valid immediately, no warm-up). Bota:
        async ``tare_compensated_wrench()`` + drain until it settles (``_warm_up_tare``)."""
        if comp.active == "rnea" and comp._rnea is not None:
            n = 0
            acc_raw = np.zeros(6, dtype=np.float64)
            acc_pred = np.zeros(6, dtype=np.float64)
            deadline = time.perf_counter() + 1.0
            while time.perf_counter() < deadline and not self._stop.is_set():
                frame = driver.read_frame_blocking()
                raw12 = chain.assemble(frame)
                pred = comp._rnea.predict_payload_wrench(time.perf_counter(), raw12[6:9], raw12[9:12])
                acc_raw += raw12[:6]
                acc_pred += pred
                n += 1
                if self._read_period:
                    time.sleep(self._read_period)
            if n > 0:
                comp._rnea.tare_from_prediction(acc_raw / n, acc_pred / n)
            return
        # Bota
        comp._bota.tare_compensated_wrench()
        self._warm_up_tare(driver, comp, chain)

    def _warm_up_tare(self, driver, comp, chain, timeout_s: float = 3.0) -> bool:
        """Drain frames until the async Bota compensated tare settles (update()==OK).

        Throttled by ``self._read_period`` for the same reason the steady-state loop
        is: an unthrottled ``read_frame_blocking`` burst here holds the GIL solid at
        startup, which starves the sibling reader's compensator (irregular dt →
        'Failed to process IMU data') and the asyncio WS server (never binds)."""
        deadline = time.perf_counter() + timeout_s
        while time.perf_counter() < deadline and not self._stop.is_set():
            frame = driver.read_frame_blocking()
            rc = comp.update(time.perf_counter(), chain.assemble(frame), chain.comp12)
            if chain.rc_is_ok(rc):
                return True
            if chain.rc_is_fatal(rc):
                return False
            if self._read_period:
                time.sleep(self._read_period)
        return False


# --------------------------------------------------------------------------- #
# WebSocket publisher.
# --------------------------------------------------------------------------- #
async def _serve(
    host: str, port: int, readers: dict[str, _HandReader], rate_hz: float, stop: threading.Event,
    emit_raw: bool = False, pipeline: bool = False,
):
    import websockets

    clients: set = set()

    def _fire_tare(source: str) -> None:
        for r in readers.values():
            r.request_tare()
        print(f"[force-bridge] re-tare requested via {source} (do this in FREE SPACE)", flush=True)

    async def handler(ws):
        clients.add(ws)
        try:
            # Bidirectional: a client (the policy's ForceLiveSource, driven by the
            # joystick tare button) can send {"cmd":"tare"} to re-tare BOTH sensors
            # remotely — no stdin access needed on a gantry run.
            async for raw in ws:
                try:
                    msg = json.loads(raw)
                except (ValueError, TypeError):
                    continue
                if isinstance(msg, dict) and msg.get("cmd") in ("tare", "retare"):
                    _fire_tare("WS client")
        except Exception:  # noqa: BLE001
            pass
        finally:
            clients.discard(ws)

    period = 1.0 / rate_hz if rate_hz > 0 else 0.02
    seq = 0
    async with websockets.serve(handler, host, port, max_size=None):
        print(f"[force-bridge] F/T WS at ws://{host}:{port} (hands={sorted(readers)})", flush=True)
        # Start the GIL-heavy readers ONLY AFTER the socket is bound (rig ordering,
        # bota_ft_viz.py: "build the server BEFORE the reader — a busy reader holds
        # the GIL and will deadlock the server bootstrap"). The reader threads then
        # do their configure/activate/warm-up-tare concurrently while the event loop
        # is already accepting connections, so a slow ~0.5 s tare can't block the WS.
        for r in readers.values():
            if not r.is_alive():
                r.start()
        while not stop.is_set():
            if clients:

                def _side(name: str):
                    r = readers.get(name)
                    if r is None:
                        return None
                    f = r.latest_force()
                    return None if f is None else [float(f[0]), float(f[1]), float(f[2])]

                frame = {"type": "ft", "seq": seq, "t": time.time(), "left": _side("left"), "right": _side("right")}

                # OPT-IN extra curves (recorder): add the truly-RAW wrench and the
                # gravity-only force as SEPARATE keys. Existing consumers (the policy's
                # ForceLiveSource) read only `left`/`right` and ignore unknown keys, so
                # this is fully back-compatible and byte-identical to them. Only added
                # when emit_raw. raw = [Fx,Fy,Fz,Tx,Ty,Tz] (6), grav = [Fx,Fy,Fz] (3).
                # Uses latest_all() so a hand's force/raw/grav in ONE frame are a coherent
                # single-read snapshot (not three separate reads). NOTE: this REPLACES the
                # `left`/`right` set above with the same-snapshot force, so the published
                # RNEA value is identical — just guaranteed consistent with its raw/grav.
                if emit_raw:
                    for name in ("left", "right"):
                        r = readers.get(name)
                        if r is None:
                            frame[name] = None
                            frame[f"{name}_raw"] = None
                            frame[f"{name}_grav"] = None
                            continue
                        f, rw, gf = r.latest_all()
                        frame[name] = None if f is None else [float(f[0]), float(f[1]), float(f[2])]
                        frame[f"{name}_raw"] = None if rw is None else [float(x) for x in rw]
                        frame[f"{name}_grav"] = None if gf is None else [float(gf[0]), float(gf[1]), float(gf[2])]

                # OPT-IN 6-DoF pipe curve for the HL planner (gravity-only, tared,
                # anti-aliased). SEPARATE keys; the low-level ignores them, so this is
                # byte-identical to the policy stream. Independent of --emit-raw.
                if pipeline:
                    for name in ("left", "right"):
                        r = readers.get(name)
                        p = None if r is None else r.latest_pipe()
                        frame[f"{name}_pipe"] = None if p is None else [float(x) for x in p]

                websockets.broadcast(clients, json.dumps(frame))
                seq += 1
            await asyncio.sleep(period)


def _stdin_tare_watcher(readers: dict[str, _HandReader], stop: threading.Event) -> None:
    """Fire a re-tare on all readers when the operator types ``t``+Enter."""
    for line in iter(sys.stdin.readline, ""):
        if stop.is_set():
            break
        if line.strip().lower() in ("t", "tare"):
            for r in readers.values():
                r.request_tare()
            print("[force-bridge] re-tare requested (do this in FREE SPACE)", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    _repo_bota = os.environ.get("HOLOSOMA_BOTA_CONFIG_DIR", "")
    ap.add_argument(
        "--driver-config",
        default=(os.path.join(_repo_bota, "bota_driver", "bota_binary.json") if _repo_bota else None),
        help="Bota driver config template (bota_binary.json).",
    )
    ap.add_argument("--left-serial", default=None, help="by-id serial substring for the LEFT sensor (else auto/omit).")
    ap.add_argument("--right-serial", default=None, help="by-id serial substring for the RIGHT sensor.")
    ap.add_argument("--left-compensator", default=None, help="payload_compensator JSON for the LEFT hand.")
    ap.add_argument("--right-compensator", default=None, help="payload_compensator JSON for the RIGHT hand.")
    ap.add_argument(
        "--ft-algo",
        choices=("bota", "rnea"),
        default=(os.environ.get("FT_ALGO", "bota") or "bota").lower(),
        help="F/T payload-compensation algorithm for BOTH hands: 'bota' (official, "
        "gravity-only as this rig runs it) or 'rnea' (independent single-body "
        "Newton-Euler DYNAMIC compensation — removes the tool's inertial reaction m·a "
        "that gravity-only leaves as an accel-proportional phantom, damping the "
        "startup shake). Per-hand override via --left-algo/--right-algo. Default bota.",
    )
    ap.add_argument(
        "--left-algo", choices=("bota", "rnea"), default=None, help="per-hand override of --ft-algo (LEFT)."
    )
    ap.add_argument(
        "--right-algo", choices=("bota", "rnea"), default=None, help="per-hand override of --ft-algo (RIGHT)."
    )
    ap.add_argument(
        "--left-rnea-config", default=None, help="LEFT-hand rnea_compensator_left.json (required for --ft-algo rnea)."
    )
    ap.add_argument(
        "--right-rnea-config",
        default=None,
        help="RIGHT-hand rnea_compensator_right.json (required for --ft-algo rnea).",
    )
    ap.add_argument(
        "--antialias-hz",
        type=float,
        default=_DEFAULT_ANTIALIAS_HZ,
        help="anti-alias low-pass cutoff (Hz) on the compensated force at the read rate "
        "BEFORE the 50 Hz policy samples it. Prevents high-frequency residual aliasing "
        "into the control band (phantom contact / shake). Must be < policy_rate/2 = 25 Hz; "
        f"default {_DEFAULT_ANTIALIAS_HZ:.0f} (delay ~13 ms < one 20 ms tick). 0 = disabled.",
    )
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8771)
    ap.add_argument("--rate-hz", type=float, default=100.0, help="WS publish rate (Hz).")
    ap.add_argument(
        "--emit-raw",
        action="store_true",
        default=os.environ.get("FT_EMIT_RAW", "").strip().lower() in ("1", "true", "yes", "on"),
        help="ALSO publish the truly-RAW sensor wrench (left_raw/right_raw, 6-vec) and the "
        "gravity-only Bota force (left_grav/right_grav, 3-vec) as EXTRA WS keys, for the "
        "teleop recorder's raw↔gravity↔RNEA toggle. DEFAULT OFF ⇒ the published left/right "
        "(RNEA, tared, anti-aliased) and the read-loop timing are byte-identical to a normal "
        "deploy; the policy's ForceLiveSource ignores the extra keys either way. Set FT_EMIT_RAW=1 "
        "or pass --emit-raw only when recording.",
    )
    ap.add_argument(
        "--pipeline",
        action="store_true",
        default=os.environ.get("FT_PIPELINE", "").strip().lower() in ("1", "true", "yes", "on"),
        help="ALSO publish a 6-DoF gravity-only (bota), TARED, anti-aliased sensor-frame "
        "wrench (left_pipe/right_pipe, 6-vec) as EXTRA WS keys for the S1 MuSe-on-UVA force "
        "policy (the high-level planner), which trained on exactly that curve. Works for both "
        "FT_ALGO=bota and =rnea (a dedicated gravity engine runs alongside RNEA). DEFAULT OFF ⇒ "
        "the published left/right and the read-loop timing are byte-identical; the low-level "
        "ignores the extra keys. The WS already fans out to all clients, so the low-level "
        "controller and the HL planner can both connect to this one port. Set FT_PIPELINE=1 "
        "or pass --pipeline when running the HL planner alongside the deploy.",
    )
    ap.add_argument(
        "--read-hz",
        type=float,
        default=250.0,
        help="per-sensor read throttle in Hz (0 = full rate). MUST stay throttled: "
        "read_frame_blocking is a C call that holds the GIL ~1 ms/read, so two "
        "unthrottled readers starve the compensator's dt filter (→ 'Failed to "
        "process IMU data'), overflow the driver buffer (→ 'frames skipped, reading "
        "too slow'), AND starve the asyncio server so the WS never binds. 250 Hz "
        "matches the validated 2-sensor rig (real_world_viz --ft-read-hz).",
    )
    args = ap.parse_args()

    if not args.driver_config or not os.path.isfile(args.driver_config):
        print(
            f"ERROR: --driver-config not found: {args.driver_config!r} "
            "(pass --driver-config or set HOLOSOMA_BOTA_CONFIG_DIR).",
            file=sys.stderr,
        )
        return 1
    if not args.left_compensator and not args.right_compensator:
        print("ERROR: pass at least one of --left-compensator / --right-compensator.", file=sys.stderr)
        return 1

    # Anti-alias cutoff must stay below the 50 Hz policy Nyquist (25 Hz) or the
    # high-frequency residual it is meant to remove aliases straight into the band.
    if args.antialias_hz > 0 and args.antialias_hz >= 25.0:
        print(
            f"[force-bridge] WARNING: --antialias-hz {args.antialias_hz} >= 25 Hz (50 Hz-policy Nyquist); "
            "high-frequency residual will alias into the control band. Use < 25 (default 12).",
            flush=True,
        )

    # Resolve each configured hand to a device port (pin by serial substring, else
    # first discovered — warn on ambiguity so a wrong sensor is obvious).
    discovered = _find_bota_ports()
    print(f"[force-bridge] Bota by-id ports: {discovered or '<none>'}", flush=True)

    def _resolve_port(serial: str | None) -> str | None:
        if serial:
            hits = [p for p in discovered if serial in p]
            if not hits:
                print(
                    f"[force-bridge] WARNING: no by-id port matches serial {serial!r}; the "
                    "reader will retry via the driver config.",
                    flush=True,
                )
                return None
            return hits[0]
        return discovered[0] if discovered else None  # unpinned: first sensor

    readers: dict[str, _HandReader] = {}
    resolved_ports: dict[str, str | None] = {}
    for side, serial, comp, per_algo, rnea_cfg in (
        ("left", args.left_serial, args.left_compensator, args.left_algo, args.left_rnea_config),
        ("right", args.right_serial, args.right_compensator, args.right_algo, args.right_rnea_config),
    ):
        if not comp:
            continue
        if not os.path.isfile(comp):
            print(f"ERROR: {side} compensator config not found: {comp!r}", file=sys.stderr)
            return 1
        algo = (per_algo or args.ft_algo or "bota").lower()
        if algo == "rnea":
            if not rnea_cfg:
                print(
                    f"ERROR: --ft-algo/--{side}-algo=rnea for the {side} hand but no --{side}-rnea-config given.",
                    file=sys.stderr,
                )
                return 1
            if not os.path.isfile(rnea_cfg):
                print(f"ERROR: {side} RNEA config not found: {rnea_cfg!r}", file=sys.stderr)
                return 1
        elif rnea_cfg:
            # A --*-rnea-config was supplied but the resolved algo is 'bota' (the user
            # likely forgot --ft-algo rnea). Silently running gravity-only would leave
            # the inertial phantom the RNEA config exists to remove — warn loudly.
            print(
                f"[force-bridge] WARNING: --{side}-rnea-config given but the {side} hand's algo is 'bota' "
                f"(pass --ft-algo rnea or --{side}-algo rnea to use it); the RNEA config is IGNORED.",
                flush=True,
            )
        resolved_ports[side] = _resolve_port(serial)
        readers[side] = _HandReader(
            side,
            args.driver_config,
            comp,
            resolved_ports[side],
            args.read_hz,
            algo=algo,
            rnea_config=rnea_cfg if algo == "rnea" else None,
            antialias_hz=args.antialias_hz,
            emit_raw=args.emit_raw,
            pipeline=args.pipeline,
        )

    # Bimanual collision guard: two hands must resolve to DISTINCT devices. This
    # catches BOTH the explicit-identical-serial case AND the unpinned case where
    # neither serial is given so both hands fall back to discovered[0] (the same tty)
    # — opening one Bota from two BotaDriver instances fails or interleaves frames.
    if len(readers) == 2:
        lp, rp = resolved_ports.get("left"), resolved_ports.get("right")
        if lp is not None and lp == rp:
            print(
                f"ERROR: both hands resolved to the SAME Bota device ({lp!r}). "
                "Pin distinct sensors with --left-serial / --right-serial (a Bota port opens in ONE process only).",
                file=sys.stderr,
            )
            return 1
        if lp is None and rp is None:
            print(
                "ERROR: bimanual bridge but neither --left-serial nor --right-serial pinned — both hands would "
                "open the first discovered Bota. Pass --left-serial and --right-serial.",
                file=sys.stderr,
            )
            return 1

    stop = threading.Event()
    # NOTE: the readers are started INSIDE _serve, only after the WS socket binds
    # (see the ordering note there) — do NOT start them here or they'll hog the GIL
    # during the server bootstrap.
    threading.Thread(target=_stdin_tare_watcher, args=(readers, stop), daemon=True).start()

    if args.emit_raw:
        print(
            "[force-bridge] --emit-raw ON: publishing left_raw/right_raw + left_grav/right_grav (recorder curves).",
            flush=True,
        )
    if args.pipeline:
        print(
            "[force-bridge] --pipeline ON: publishing left_pipe/right_pipe (6-DoF gravity-only, tared, "
            "anti-aliased) for the HL planner. The low-level ignores these keys.",
            flush=True,
        )
    print("[force-bridge] type 't'+Enter to re-tare (FREE SPACE); Ctrl-C to stop.", flush=True)
    try:
        asyncio.run(_serve(args.host, args.port, readers, args.rate_hz, stop,
                           emit_raw=args.emit_raw, pipeline=args.pipeline))
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        for r in readers.values():
            r.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
