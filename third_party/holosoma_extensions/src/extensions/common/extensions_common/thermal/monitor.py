"""ThermalMonitor: background temperature logging with cooldown tracking.

Runs a daemon thread that periodically reads joint temperatures, logs them,
detects thermal equilibrium, monitors shutoff thresholds, and tracks cooldown
decay curves when triggered.
"""

from __future__ import annotations

import bisect
import dataclasses
import logging
import threading
import time
from collections import deque
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from extensions_common.thermal.readers import ThermalReader
    from extensions_common.thermal.report import ThermalReport

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class ThermalConfig:
    """Configuration for thermal monitoring."""

    log_interval_s: float = 1.0
    """How often to sample joint temperatures (seconds)."""
    shutoff_temp_c: float = 105.0
    """Emergency cutoff threshold (Celsius). Any joint hitting this triggers shutoff."""
    warning_temp_c: float = 85.0
    """Warning threshold (Celsius). Logged but does not trigger shutoff."""
    equilibrium_threshold_c: float = 1.0
    """Short window: temperature must vary less than this to be considered stable."""
    equilibrium_window_s: float = 300.0
    """Short window: temperature must be stable for this long (seconds)."""
    equilibrium_threshold_long_c: float = 2.0
    """Long window: catches slow drift that passes the short window."""
    equilibrium_window_long_s: float = 600.0
    """Long window duration (seconds). Both windows must pass simultaneously."""
    early_exit_threshold_s: float = 300.0
    """If shutoff occurs before this many seconds, flag skip_heavier on the report."""
    cooldown_target_c: float = 35.0
    """Default cooldown target: stop tracking when joint falls below this."""
    cooldown_torque_warn_nm: float = 0.5
    """Warn if torque exceeds this during cooldown (joint should be passive)."""
    joints_to_watch: list[str] | None = None
    """Subset of joints to monitor. None = all joints."""
    max_history_samples: int = 0
    """Maximum number of history samples to retain in memory. 0 = unlimited.
    For a 10-hour test at 1Hz, 36000 is sufficient. Older samples are discarded."""


@dataclasses.dataclass
class CooldownRecord:
    """Captured decay curve for one joint's cooldown event."""

    joint: str
    trigger_time: float
    """Monotonic timestamp when cooldown tracking began."""
    peak_temp_c: float
    """Temperature at the moment of trigger."""
    target_temp_c: float
    """Temperature threshold for cooldown completion."""
    timestamps: list[float] = dataclasses.field(default_factory=list)
    """Monotonic timestamps of each sample during cooldown."""
    temperatures: list[float] = dataclasses.field(default_factory=list)
    """Temperature at each sample."""
    torque_warnings: list[tuple[float, float]] = dataclasses.field(default_factory=list)
    """(timestamp, torque_nm) for any torque detected during cooldown."""
    completed: bool = False
    """True if temp fell below target_temp_c."""
    time_to_cool_s: float | None = None
    """Seconds from trigger to reaching target (None if not yet complete)."""
    decay_tau_s: float | None = None
    """Fitted exponential decay time constant (None if insufficient data)."""


class ThermalMonitor:
    """Background thermal monitoring with cooldown tracking.

    Usage::

        monitor = ThermalMonitor(reader, config)
        monitor.start()
        # ... run test ...
        if monitor.is_shutoff_triggered():
            emergency_stop()
        report = monitor.stop()
    """

    def __init__(self, reader: ThermalReader, config: ThermalConfig | None = None):
        self._reader = reader
        self._config = config or ThermalConfig()
        self._dof_names = reader.get_dof_names()
        self._watched = (
            set(self._config.joints_to_watch) if self._config.joints_to_watch is not None else set(self._dof_names)
        )

        # State
        self._lock = threading.Lock()
        self._running = False
        self._thread: threading.Thread | None = None

        # Time-series log: deque of (monotonic_time, {joint: temp})
        maxlen = self._config.max_history_samples or None
        self._history: deque[tuple[float, dict[str, float]]] = deque(maxlen=maxlen)
        self._shutoff_triggered = False
        self._shutoff_joint: str | None = None
        self._start_time: float | None = None
        self._shutoff_time: float | None = None

        # Cooldown tracking: joint_name -> active CooldownRecord
        self._cooldowns: dict[str, CooldownRecord] = {}
        self._completed_cooldowns: list[CooldownRecord] = []

        # Callbacks (lock-protected)
        self._on_shutoff: list = []
        self._on_warning: list = []

    # -- Public API: lifecycle --

    def start(self) -> None:
        """Start background temperature logging thread."""
        if self._running:
            return
        self._running = True
        self._start_time = time.monotonic()
        self._thread = threading.Thread(target=self._run_loop, daemon=True, name="thermal-monitor")
        self._thread.start()
        logger.info(
            "ThermalMonitor started (interval=%.1fs, shutoff=%.0f°C)",
            self._config.log_interval_s,
            self._config.shutoff_temp_c,
        )

    def stop(self) -> ThermalReport:
        """Stop monitoring and return a ThermalReport."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None

        from extensions_common.thermal.report import ThermalReport

        with self._lock:
            all_cooldowns = self._completed_cooldowns + list(self._cooldowns.values())
            # Early exit: if shutoff happened before early_exit_threshold_s
            time_to_shutoff = None
            if self._shutoff_time is not None and self._start_time is not None:
                time_to_shutoff = self._shutoff_time - self._start_time
            return ThermalReport(
                dof_names=self._dof_names,
                watched_joints=sorted(self._watched),
                history=list(self._history),
                cooldowns=all_cooldowns,
                config=self._config,
                time_to_shutoff_s=time_to_shutoff,
                shutoff_joint=self._shutoff_joint,
            )

    # -- Public API: queries --

    def get_current_temps(self) -> dict[str, float]:
        """Latest temperature reading snapshot."""
        with self._lock:
            if self._history:
                return dict(self._history[-1][1])
        return {}

    def is_shutoff_triggered(self) -> bool:
        """True if any watched joint reached the shutoff threshold."""
        with self._lock:
            return self._shutoff_triggered

    def get_shutoff_joint(self) -> str | None:
        """Return the joint name that triggered shutoff, or None."""
        with self._lock:
            return self._shutoff_joint

    def is_equilibrium(self, joint: str) -> bool:
        """True if *joint* temperature has been stable within threshold for the equilibrium window."""
        with self._lock:
            return self._check_equilibrium(joint)

    # -- Public API: cooldown tracking --

    def enter_cooldown_tracking(self, joint: str, target_temp_c: float | None = None) -> None:
        """Manually trigger cooldown tracking for a joint.

        Can be called at any temperature — does not require hitting shutoff.
        The joint should be in passive/zero-torque mode; the monitor will warn
        if torque is detected during cooldown.

        Parameters
        ----------
        joint : str
            Joint name to track.
        target_temp_c : float | None
            Stop tracking when temp falls below this. Defaults to config value.
        """
        target = target_temp_c if target_temp_c is not None else self._config.cooldown_target_c
        with self._lock:
            current_temps = self._history[-1][1] if self._history else {}
            peak = current_temps.get(joint, 0.0)
            now = time.monotonic()
            record = CooldownRecord(joint=joint, trigger_time=now, peak_temp_c=peak, target_temp_c=target)
            self._cooldowns[joint] = record
            logger.info("Cooldown tracking started: %s at %.1f°C → target %.1f°C", joint, peak, target)

    def get_active_cooldowns(self) -> dict[str, CooldownRecord]:
        """Return currently active cooldown records."""
        with self._lock:
            return dict(self._cooldowns)

    def get_completed_cooldowns(self) -> list[CooldownRecord]:
        """Return all completed cooldown records."""
        with self._lock:
            return list(self._completed_cooldowns)

    # -- Public API: callbacks --

    def on_shutoff(self, callback) -> None:
        """Register a callback for shutoff events: ``callback(joint, temp)``."""
        with self._lock:
            self._on_shutoff.append(callback)

    def on_warning(self, callback) -> None:
        """Register a callback for warning events: ``callback(joint, temp)``."""
        with self._lock:
            self._on_warning.append(callback)

    # -- Internal --

    def _run_loop(self) -> None:
        while self._running:
            try:
                self._tick()
            except Exception:
                logger.exception("ThermalMonitor tick failed")
            time.sleep(self._config.log_interval_s)

    def _tick(self) -> None:
        now = time.monotonic()
        temps, torques = self._reader.read_state()

        with self._lock:
            self._history.append((now, temps))

            for joint in self._watched:
                temp = temps.get(joint)
                if temp is None:
                    continue

                # Shutoff check
                if not self._shutoff_triggered and temp >= self._config.shutoff_temp_c:
                    self._shutoff_triggered = True
                    self._shutoff_joint = joint
                    self._shutoff_time = now
                    logger.critical("THERMAL SHUTOFF: %s reached %.1f°C", joint, temp)
                    # Auto-enter cooldown tracking on shutoff
                    if joint not in self._cooldowns:
                        self._cooldowns[joint] = CooldownRecord(
                            joint=joint,
                            trigger_time=now,
                            peak_temp_c=temp,
                            target_temp_c=self._config.cooldown_target_c,
                        )
                    for cb in self._on_shutoff:
                        try:
                            cb(joint, temp)
                        except Exception:
                            logger.exception("Shutoff callback failed")

                # Warning check
                if temp >= self._config.warning_temp_c:
                    for cb in self._on_warning:
                        try:
                            cb(joint, temp)
                        except Exception:
                            logger.exception("Warning callback failed")

            # Update active cooldown records
            completed_joints = []
            for joint, record in self._cooldowns.items():
                temp = temps.get(joint)
                if temp is None:
                    continue

                record.timestamps.append(now)
                record.temperatures.append(temp)

                # Check for torque during cooldown (throttle warnings to once per 30s)
                if torques is not None:
                    torque = torques.get(joint, 0.0)
                    if abs(torque) > self._config.cooldown_torque_warn_nm:
                        record.torque_warnings.append((now, torque))
                        if len(record.torque_warnings) <= 1 or (now - record.torque_warnings[-2][0]) >= 30.0:
                            logger.warning(
                                "Torque detected during cooldown: %s = %.2f Nm (expected passive)", joint, torque
                            )

                # Check if cooldown target reached
                if temp <= record.target_temp_c:
                    record.completed = True
                    record.time_to_cool_s = now - record.trigger_time
                    record.decay_tau_s = self._fit_decay_tau(record)
                    completed_joints.append(joint)
                    logger.info(
                        "Cooldown complete: %s cooled from %.1f°C to %.1f°C in %.0fs (tau=%.1fs)",
                        joint,
                        record.peak_temp_c,
                        temp,
                        record.time_to_cool_s,
                        record.decay_tau_s or 0.0,
                    )

            for joint in completed_joints:
                self._completed_cooldowns.append(self._cooldowns.pop(joint))

    def _check_window_stable(self, joint: str, window_s: float, threshold_c: float) -> bool:
        """Check if joint temp has been stable within threshold for a given window."""
        if not self._history:
            return False

        now = self._history[-1][0]
        window_start = now - window_s

        # Binary search for the start of the window (history is time-sorted)
        times = [e[0] for e in self._history]
        start_idx = bisect.bisect_left(times, window_start)

        temps_in_window = []
        earliest_in_window = None
        for i in range(start_idx, len(self._history)):
            entry = self._history[i]
            if joint in entry[1]:
                temps_in_window.append(entry[1][joint])
                if earliest_in_window is None:
                    earliest_in_window = entry[0]

        if not temps_in_window:
            return False

        # Need samples spanning most of the window
        if earliest_in_window is None or (now - earliest_in_window) < window_s * 0.8:
            return False

        return (max(temps_in_window) - min(temps_in_window)) <= threshold_c

    def _check_equilibrium(self, joint: str) -> bool:
        """Check dual-window equilibrium: both short and long windows must pass."""
        short_ok = self._check_window_stable(
            joint, self._config.equilibrium_window_s, self._config.equilibrium_threshold_c
        )
        long_ok = self._check_window_stable(
            joint, self._config.equilibrium_window_long_s, self._config.equilibrium_threshold_long_c
        )
        return short_ok and long_ok

    @staticmethod
    def _fit_decay_tau(record: CooldownRecord) -> float | None:
        """Fit exponential decay T(t) = T_ambient + (T_peak - T_ambient) * exp(-t/tau).

        Uses least-squares on log-transformed data. Returns tau in seconds,
        or None if insufficient data.
        """
        if len(record.timestamps) < 3:
            return None

        t0 = record.timestamps[0]
        t_arr = np.array([t - t0 for t in record.timestamps])
        temp_arr = np.array(record.temperatures)

        t_ambient = record.target_temp_c
        delta = temp_arr - t_ambient

        # Filter to positive deltas only (above ambient)
        mask = delta > 0.1
        if mask.sum() < 3:
            return None

        t_valid = t_arr[mask]
        delta_valid = delta[mask]

        # Linear regression on ln(delta) = ln(T_peak - T_ambient) - t/tau
        log_delta = np.log(delta_valid)
        # Fit: log_delta = a + b*t  =>  tau = -1/b
        n = len(t_valid)
        sum_t = t_valid.sum()
        sum_log = log_delta.sum()
        sum_t2 = (t_valid**2).sum()
        sum_t_log = (t_valid * log_delta).sum()

        denom = n * sum_t2 - sum_t**2
        if abs(denom) < 1e-12:
            return None

        b = (n * sum_t_log - sum_t * sum_log) / denom
        if b >= 0:
            # Positive slope means temperature is rising, not decaying
            return None

        tau = -1.0 / b
        return tau
