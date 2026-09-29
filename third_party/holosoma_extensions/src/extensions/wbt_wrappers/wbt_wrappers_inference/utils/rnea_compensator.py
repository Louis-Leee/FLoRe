"""Pure-numpy RNEA dynamic (inertia) F/T compensation + anti-alias output stage.

Self-contained port of the validated UMI ``bota_demo_py`` reference so the sim2real
``force_bridge`` can feed the policy an **RNEA-compensated** contact force (tool
gravity **and** inertial reaction removed) instead of Bota's gravity-only wrench.
Gravity-only leaves an acceleration-proportional phantom (``m·a``) in ``force_meas``
that, under a moving arm, drives the escalating startup oscillation the deploy
kept hitting ([[bota-dynamic-compensation]]); RNEA subtracts it.

Two pieces, both dependency-free (numpy only) so they import cleanly in the Bota
env (where ``force_bridge`` runs by path, and holosoma is absent) AND are headlessly
unit-testable:

  1. ``RneaDynamicCompensator`` — single-body Newton-Euler compensator. A faithful
     re-implementation of ``rnea_core.RneaDynamicCompensator`` (same math, same JSON
     schema, same ``tare_from_prediction`` convention), vendored here so the deploy
     does NOT import from ``UMI_collection`` (the no-UMI-dep constraint of the whole
     force_deploy stack). The physics — see the module docstring of the reference —
     is: transport the IMU specific force to the payload COM, ``F = m·f_com``,
     ``M = I·α + ω×(I·ω) + c×F``, then ``comp = raw − bias − wrench_sign·[F,M]``.

  2. ``AntiAliasFilter`` — the demo's **output stage**: a dt-aware first-order
     low-pass at ``ANTIALIAS_HZ`` (default 12 Hz) applied to the compensated wrench
     at the sensor read rate BEFORE the 50 Hz policy samples it. A 50 Hz controller
     must not read the raw ~1 kHz compensated wrench: its high-frequency residual
     (mostly IMU/α noise, well above 25 Hz) would alias down into the control band
     and cause phantom contact / shake. 12 Hz < 50/2 Nyquist, and its group delay
     ~1/(2π·12) ≈ 13 ms is under one 20 ms control period, so it does not lag the
     50 Hz servo. (The policy-side ``HOLOSOMA_FORCE_MEAS_EMA_ALPHA`` filter runs
     AFTER the 50 Hz sample and so cannot prevent aliasing — this pre-decimation LP
     is the principled fix; with it on, the policy EMA can relax toward 1.0/off.)

CRITICAL INVARIANT (the single most important correctness point, see the reference
``docs/RNEA_INTEGRATION_DESIGN.md``): the bridge's ``FtChain.assemble`` already runs
``BotaTfImu`` to transport the IMU from the imuF point to the **sensor origin**
before packing ``raw12[6:12]``. ``rnea_core`` otherwise expects RAW IMU at the IMU
point and transports IMU→COM itself via ``imu_offset`` (``r_ic = com − imu_offset``).
So when RNEA runs on the pre-transported 12-vector it MUST be built with
``imu_offset = [0, 0, 0]`` — else the lever arm is double-counted. ``com`` is already
sensor-origin-referenced, so ``r_ic = com − 0 = com`` is correct. ``build_from_json``
zeroes ``imu_offset`` for exactly this reason (pass ``zero_imu_offset=False`` only
for the standalone raw-IMU path). ``wrench_sign`` (a global sign) is unaffected.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json

import numpy as np


def _skew(v: np.ndarray) -> np.ndarray:
    """3×3 matrix M with M @ x == np.cross(v, x)."""
    x, y, z = float(v[0]), float(v[1]), float(v[2])
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=np.float64)


def _lp_alpha(cutoff_hz: float, dt: float) -> float:
    """First-order low-pass smoothing factor in [0, 1] for a step ``dt`` (seconds).

    ``<= 0`` cutoff → 1.0 (pass-through). ``a = dt / (tau + dt)``, ``tau = 1/(2π·fc)``.
    Identical to ``rnea_core.AlphaEstimator._lp_alpha`` and the demo's output-stage LP.
    """
    if cutoff_hz <= 0.0:
        return 1.0
    tau = 1.0 / (2.0 * np.pi * cutoff_hz)
    return dt / (tau + dt)


@dataclass
class AlphaEstimatorConfig:
    """ω→α estimator config (matches ``rnea_core.AlphaEstimatorConfig``)."""

    omega_lowpass_cutoff_hz: float = 20.0
    alpha_lowpass_cutoff_hz: float = 20.0
    min_dt: float = 1.0e-4  # 10 kHz ceiling on the differencing dt
    max_dt: float = 1.0e-2  # 100 Hz floor


class AlphaEstimator:
    """Estimate angular acceleration α = dω/dt from a (timestamp, ω) stream.

    Pipeline: ω → [optional LP] → backward finite difference → [optional LP] → α.
    First call returns α = 0 (no history). Finite differencing amplifies gyro noise
    by ~1/dt, so both LP stages default to 20 Hz — this is the single largest error
    source in single-body dynamic compensation (see the reference docs/04).
    """

    def __init__(self, cfg: AlphaEstimatorConfig | None = None) -> None:
        self.cfg = cfg or AlphaEstimatorConfig()
        self._prev_t: float | None = None
        self._omega_f = np.zeros(3, dtype=np.float64)
        self._alpha_f = np.zeros(3, dtype=np.float64)

    def reset(self) -> None:
        self._prev_t = None
        self._omega_f[:] = 0.0
        self._alpha_f[:] = 0.0

    def update(self, timestamp: float, omega: np.ndarray) -> np.ndarray:
        """Return the current α estimate (rad/s²) in the same axes as ω."""
        omega = np.asarray(omega, dtype=np.float64)
        if self._prev_t is None:
            self._omega_f[:] = omega  # seed; no derivative on the first sample
            self._prev_t = timestamp
            return self._alpha_f.copy()

        dt = float(np.clip(timestamp - self._prev_t, self.cfg.min_dt, self.cfg.max_dt))
        self._prev_t = timestamp

        a_w = _lp_alpha(self.cfg.omega_lowpass_cutoff_hz, dt)
        omega_prev = self._omega_f.copy()
        self._omega_f += a_w * (omega - self._omega_f)  # 1) low-pass ω

        alpha_raw = (self._omega_f - omega_prev) / dt  # 2) backward difference

        a_a = _lp_alpha(self.cfg.alpha_lowpass_cutoff_hz, dt)
        self._alpha_f += a_a * (alpha_raw - self._alpha_f)  # 3) low-pass α
        return self._alpha_f.copy()


@dataclass
class RneaCompensatorConfig:
    """Single rigid-body payload + conventions (matches ``rnea_core`` JSON schema)."""

    mass: float = 0.0
    com: np.ndarray = field(default_factory=lambda: np.zeros(3))  # sensor origin → COM [m]
    inertia: np.ndarray = field(default_factory=lambda: np.zeros((3, 3)))  # about COM [kg m²]
    imu_offset: np.ndarray = field(default_factory=lambda: np.zeros(3))  # sensor origin → IMU [m]
    accel_is_specific_force: bool = True  # Bota IMU reports specific force (incl. gravity)
    wrench_sign: float = 1.0  # global sign matching the reading convention (+1/−1)
    enable_linear_inertia: bool = True
    enable_rotational_inertia: bool = True
    enable_com_coupling: bool = True
    bias: np.ndarray = field(default_factory=lambda: np.zeros(6))  # set by tare
    alpha: AlphaEstimatorConfig = field(default_factory=AlphaEstimatorConfig)

    @staticmethod
    def from_json(path: str) -> "RneaCompensatorConfig":
        with open(path) as fh:
            raw = json.load(fh)
        c = raw["rnea_compensator_config"]
        pp = c["payload_properties"]
        com = pp["com"]
        inertia_d = pp["inertia"]
        inertia = np.array(
            [
                [inertia_d["Ixx"], inertia_d["Ixy"], inertia_d["Ixz"]],
                [inertia_d["Ixy"], inertia_d["Iyy"], inertia_d["Iyz"]],
                [inertia_d["Ixz"], inertia_d["Iyz"], inertia_d["Izz"]],
            ],
            dtype=np.float64,
        )
        geo = c.get("geometry", {})
        conv = c.get("conventions", {})
        tog = c.get("toggles", {})
        af = c.get("alpha_estimator", {})
        return RneaCompensatorConfig(
            mass=float(pp["mass"]),
            com=np.array([com["x"], com["y"], com["z"]], dtype=np.float64),
            inertia=inertia,
            imu_offset=np.array(geo.get("imu_offset", [0.0, 0.0, 0.0]), dtype=np.float64),
            accel_is_specific_force=bool(conv.get("accel_is_specific_force", True)),
            wrench_sign=float(conv.get("wrench_sign", 1.0)),
            enable_linear_inertia=bool(tog.get("enable_linear_inertia", True)),
            enable_rotational_inertia=bool(tog.get("enable_rotational_inertia", True)),
            enable_com_coupling=bool(tog.get("enable_com_coupling", True)),
            alpha=AlphaEstimatorConfig(
                omega_lowpass_cutoff_hz=float(af.get("omega_lowpass_cutoff_hz", 20.0)),
                alpha_lowpass_cutoff_hz=float(af.get("alpha_lowpass_cutoff_hz", 20.0)),
                min_dt=float(af.get("min_dt", 1.0e-4)),
                max_dt=float(af.get("max_dt", 1.0e-2)),
            ),
        )


class RneaDynamicCompensator:
    """Single-body Newton-Euler dynamic wrench compensator (numpy port of
    ``rnea_core.RneaDynamicCompensator``).

    Interface mirrors ``BotaPayloadCompensator``/``rnea_core`` so it drops into the
    bridge reader loop: ``update(ts, raw12, comp12)`` reads
    ``[f(3), t(3), a(3), ω(3)]`` (SENSOR frame) from ``raw12``, writes the compensated
    contact wrench into ``comp12[0:6]`` and passes the IMU channels through, returning
    ``"OK"`` (or ``"DEGRADED:<reason>"``). Compensated = ``raw − bias − wrench_sign·predicted``.
    """

    def __init__(self, cfg: RneaCompensatorConfig) -> None:
        if not cfg.accel_is_specific_force:
            # The gravity-removed path needs a sensor-frame gravity vector (i.e.
            # orientation) threaded through update(); this IMU-only single-body path
            # has none. Fail loudly rather than silently dropping m·g.
            raise NotImplementedError(
                "accel_is_specific_force=False (gravity-removed accelerometer) is not "
                "supported: this path has no sensor-frame gravity vector."
            )
        self.cfg = cfg
        self.alpha_est = AlphaEstimator(cfg.alpha)
        self._last_predicted = np.zeros(6, dtype=np.float64)

    # ---- calibration ----
    def set_bias(self, bias6: np.ndarray) -> None:
        self.cfg.bias = np.asarray(bias6, dtype=np.float64).copy()

    def tare_from_prediction(self, wrench_raw6: np.ndarray, predicted6: np.ndarray) -> None:
        """Zero the compensated output at the current (static, free) pose:
        ``bias := raw − wrench_sign·predicted`` so ``update`` returns ~0 now. Call
        ONLY while stationary and free of contact."""
        self.cfg.bias = np.asarray(wrench_raw6, dtype=np.float64) - self.cfg.wrench_sign * np.asarray(
            predicted6, dtype=np.float64
        )

    # ---- core physics ----
    def predict_payload_wrench(self, timestamp: float, accel: np.ndarray, omega: np.ndarray) -> np.ndarray:
        """Predict the payload gravity+inertial wrench at the sensor origin, sensor
        frame, in the reading convention (before ``wrench_sign``). Returns ``[F(3), M(3)]``."""
        cfg = self.cfg
        m = cfg.mass
        c = cfg.com
        inertia = cfg.inertia

        accel = np.asarray(accel, dtype=np.float64)
        omega = np.asarray(omega, dtype=np.float64)
        alpha = self.alpha_est.update(timestamp, omega)

        f_imu = accel  # specific force at the IMU point (gravity already folded in)
        r_ic = c - cfg.imu_offset  # IMU→COM lever (imu_offset zeroed for the bridge path)
        if cfg.enable_com_coupling:
            f_com = f_imu + np.cross(alpha, r_ic) + np.cross(omega, np.cross(omega, r_ic))
        else:
            f_com = f_imu

        force = m * f_com if cfg.enable_linear_inertia else np.zeros(3, dtype=np.float64)

        moment = np.zeros(3, dtype=np.float64)
        if cfg.enable_rotational_inertia:
            moment = moment + inertia @ alpha + np.cross(omega, inertia @ omega)
        if cfg.enable_com_coupling:
            moment = moment + np.cross(c, force)

        out = np.empty(6, dtype=np.float64)
        out[:3] = force
        out[3:] = moment
        self._last_predicted = out
        return out

    # ---- main entry point ----
    def update(self, timestamp: float, wrench_imu_raw: np.ndarray, wrench_imu_compensated: np.ndarray) -> str:
        """Write ``comp = raw − bias − wrench_sign·predicted`` into
        ``wrench_imu_compensated[0:6]`` (IMU channels passed through). In place."""
        raw = np.asarray(wrench_imu_raw, dtype=np.float64)
        if raw.shape[0] != 12:
            raise ValueError(f"wrench_imu_raw must be length 12, got {raw.shape[0]}")

        predicted = self.predict_payload_wrench(timestamp, raw[6:9], raw[9:12])
        comp6 = raw[:6] - self.cfg.bias - self.cfg.wrench_sign * predicted

        wrench_imu_compensated[:6] = comp6
        wrench_imu_compensated[6:] = raw[6:]
        return "OK" if np.all(np.isfinite(comp6)) else "DEGRADED:non_finite_output"

    @property
    def last_predicted_wrench(self) -> np.ndarray:
        return self._last_predicted.copy()


def build_rnea_from_json(path: str, zero_imu_offset: bool = True) -> RneaDynamicCompensator:
    """Build an ``RneaDynamicCompensator`` from a per-side ``rnea_compensator_*.json``.

    ``zero_imu_offset=True`` (the bridge default) forces ``imu_offset=[0,0,0]`` per
    the CRITICAL INVARIANT — the 12-vector IMU is already at the sensor origin, so
    re-applying the lever arm would double-count it. Pass ``False`` only when feeding
    RAW IMU at the IMU point (the standalone reference path).
    """
    cfg = RneaCompensatorConfig.from_json(path)
    if zero_imu_offset:
        cfg.imu_offset = np.zeros(3, dtype=np.float64)
    return RneaDynamicCompensator(cfg)


def read_wrench_sign_verified(path: str) -> bool | None:
    """Return ``conventions.wrench_sign_verified`` from an RNEA config (UMI-only
    marker rnea_core ignores), or ``None`` if unreadable/absent. The bridge surfaces
    a loud warning when a hand's global sign is UNVERIFIED (seeded from the other
    hand): a wrong sign makes RNEA ADD the payload wrench instead of subtracting it.
    """
    try:
        with open(path) as fh:
            conv = json.load(fh)["rnea_compensator_config"].get("conventions", {})
        val = conv.get("wrench_sign_verified")
        return bool(val) if isinstance(val, bool) else None
    except (OSError, KeyError, ValueError, TypeError):
        return None


class AntiAliasFilter:
    """dt-aware first-order low-pass output stage (the demo's ``ANTIALIAS_HZ`` stage).

    Apply to the compensated wrench at the sensor READ rate, BEFORE the 50 Hz policy
    samples it, so high-frequency residual can't alias into the control band. Seeds to
    the first good sample (no ramp-from-zero). A NaN/Inf sample MUST NOT enter the
    state (it would latch the output to NaN forever) — ``update`` holds the last good
    value on a non-finite input. ``cutoff_hz <= 0`` disables filtering (pass-through =
    byte-identical to the pre-anti-alias behavior).
    """

    def __init__(self, cutoff_hz: float, size: int = 3, max_dt: float = 0.1):
        self.cutoff_hz = float(cutoff_hz)
        # Clamp each step's dt to <= max_dt. A stale-large dt (e.g. the gap accumulated
        # across a run of skipped/degraded samples) would make a = dt/(tau+dt) → 1,
        # briefly turning the LP into a pass-through that lets ONE unfiltered spike
        # through on recovery. Clamping keeps the smoothing factor bounded regardless
        # of caller clock hygiene (defense-in-depth; the bridge also advances its clock
        # on skip paths). 0.1 s = one 10 Hz period, far above any real read period.
        self._max_dt = float(max_dt)
        self._state = np.zeros(int(size), dtype=np.float64)
        self._seeded = False

    @property
    def enabled(self) -> bool:
        return self.cutoff_hz > 0.0

    def reset(self) -> None:
        """Drop the filter state (re-seed on the next sample) — used on re-tare so a
        fresh zero is not blended with pre-tare history."""
        self._state[:] = 0.0
        self._seeded = False

    def update(self, x: np.ndarray, dt: float) -> np.ndarray:
        """Return the filtered vector for input ``x`` over step ``dt`` (seconds).

        Pass-through when disabled. On a non-finite ``x`` (or before the first good
        sample when disabled), the value flows through unchanged when filtering is off;
        when filtering is on, a non-finite ``x`` holds the last good state.
        """
        x = np.asarray(x, dtype=np.float64)
        if not self.enabled:
            return x.copy()
        if not np.all(np.isfinite(x)):
            return self._state.copy()  # hold last good; never latch NaN into the state
        if not self._seeded:
            self._state[:] = x
            self._seeded = True
        else:
            dt_c = min(max(dt, 0.0), self._max_dt) if self._max_dt > 0.0 else max(dt, 0.0)
            a = _lp_alpha(self.cutoff_hz, dt_c)
            self._state += a * (x - self._state)
        return self._state.copy()
