#!/usr/bin/env python3
"""JSONL telemetry viewer, plotter, and cross-trial comparison tool.

Reads JSONL telemetry files produced by the test runner and provides:
- Interactive or saved matplotlib plots (default: temperature vs time)
- Named plot types: power, current, joint-current, joint-torque
- Text summary (--summary), head/tail sample viewing
- Cross-trial comparison from multiple report.json files (--compare)

Usage:
    python telemetry_plot.py test.jsonl                                    # plot temp vs time
    python telemetry_plot.py test.jsonl --fields temp,tau,q                # multi-panel plot
    python telemetry_plot.py test.jsonl --joints right_elbow               # filter joints
    python telemetry_plot.py test.jsonl --imu                              # plot IMU data
    python telemetry_plot.py test.jsonl --plot power                       # bus power vs time
    python telemetry_plot.py test.jsonl --plot joint-torque                # per-joint torque
    python telemetry_plot.py test.jsonl --save plot.png                    # save to file
    python telemetry_plot.py test.jsonl --summary                          # text summary
    python telemetry_plot.py test.jsonl --tail 20                          # last 20 samples
    python telemetry_plot.py --compare results/                            # cross-trial comparison
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


# ── JSONL streaming ──


def _iter_jsonl(path: str):
    """Yield parsed JSON objects from a JSONL file, one per line."""
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _count_lines(path: str) -> int:
    count = 0
    with open(path) as f:
        for _ in f:
            count += 1
    return count


def _load_series(path: str, fields: list[str], joints: list[str] | None, max_points: int = 50000):
    """Stream JSONL and extract time series for motor fields."""
    total = _count_lines(path)
    decimation = max(1, total // max_points)

    times = []
    motor_series: dict[str, dict[str, list[float]]] = {f: {} for f in fields}

    for i, record in enumerate(_iter_jsonl(path)):
        if i % decimation != 0:
            continue
        times.append(record["t"])
        for jname, jdata in record.get("motors", {}).items():
            if joints and jname not in joints:
                continue
            for field in fields:
                val = jdata.get(field)
                if val is not None:
                    motor_series[field].setdefault(jname, []).append(val)

    return times, motor_series, decimation


def _load_derived_series(path: str, joints: list[str] | None, max_points: int = 50000):
    """Stream JSONL and extract derived + bus fields (power, current) time series.

    When external analyzer bus data is present, it is returned alongside estimates.
    """
    total = _count_lines(path)
    decimation = max(1, total // max_points)

    times = []
    bus_power_est = []
    bus_power_mech = []
    bus_power_measured = []
    bus_current_measured = []
    joint_current: dict[str, list[float]] = {}
    joint_mech_power: dict[str, list[float]] = {}
    has_bus = False

    for i, record in enumerate(_iter_jsonl(path)):
        if i % decimation != 0:
            continue
        derived = record.get("derived")
        if derived is None:
            continue
        times.append(record["t"])
        bus_power_est.append(derived.get("bus_power_est", 0.0))
        bus_power_mech.append(derived.get("bus_power_mechanical", 0.0))
        for jname, jd in derived.get("joints", {}).items():
            if joints and jname not in joints:
                continue
            joint_current.setdefault(jname, []).append(jd.get("current_est", 0.0))
            joint_mech_power.setdefault(jname, []).append(jd.get("mech_power", 0.0))

        bus = record.get("bus")
        if bus:
            has_bus = True
            bus_power_measured.append(bus.get("power", 0.0))
            bus_current_measured.append(bus.get("current", 0.0))
        else:
            bus_power_measured.append(None)
            bus_current_measured.append(None)

    return {
        "times": times,
        "bus_power_est": bus_power_est,
        "bus_power_mech": bus_power_mech,
        "bus_power_measured": bus_power_measured if has_bus else None,
        "bus_current_measured": bus_current_measured if has_bus else None,
        "joint_current": joint_current,
        "joint_mech_power": joint_mech_power,
        "decimation": decimation,
        "has_bus": has_bus,
    }


def _load_imu_series(path: str, max_points: int = 50000):
    """Stream JSONL and extract IMU time series."""
    total = _count_lines(path)
    decimation = max(1, total // max_points)

    times = []
    quat = {k: [] for k in ["w", "x", "y", "z"]}
    omega = {k: [] for k in ["x", "y", "z"]}
    accel = {k: [] for k in ["x", "y", "z"]}

    for i, record in enumerate(_iter_jsonl(path)):
        if i % decimation != 0:
            continue
        imu = record.get("imu")
        if imu is None:
            continue
        times.append(record["t"])
        q = imu.get("quat", [0, 0, 0, 0])
        for j, k in enumerate(["w", "x", "y", "z"]):
            quat[k].append(q[j] if j < len(q) else 0.0)
        o = imu.get("omega", [0, 0, 0])
        for j, k in enumerate(["x", "y", "z"]):
            omega[k].append(o[j] if j < len(o) else 0.0)
        a = imu.get("accel", [0, 0, 0])
        for j, k in enumerate(["x", "y", "z"]):
            accel[k].append(a[j] if j < len(a) else 0.0)

    return times, quat, omega, accel


# ── Plotting helpers ──

FIELD_LABELS = {
    "temp": "Temperature (C)",
    "tau": "Torque (Nm)",
    "q": "Position (rad)",
    "cmd_q": "Cmd Position (rad)",
    "dq": "Velocity (rad/s)",
    "ddq": "Acceleration (rad/s2)",
    "volt": "Voltage (V)",
}


def _save_fig(fig, save_path: str | None):
    import matplotlib.pyplot as plt

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved plot to {save_path}")
    else:
        plt.show()
    plt.close(fig)


def _to_minutes(times):
    return [t / 60.0 for t in times]


# ── Per-trial plot functions ──


def _plot_motors(path: str, fields: list[str], joints: list[str] | None, save_path: str | None):
    """Multi-panel motor data plot."""
    import matplotlib.pyplot as plt

    times, motor_series, dec = _load_series(path, fields, joints)
    if not times:
        print("No data to plot.")
        return

    t_min = _to_minutes(times)
    n_panels = len(fields)
    fig, axes = plt.subplots(n_panels, 1, figsize=(14, 4 * n_panels), sharex=True, squeeze=False)

    for row, field in enumerate(fields):
        ax = axes[row, 0]
        for jname in sorted(motor_series[field].keys()):
            vals = motor_series[field][jname]
            ax.plot(t_min[: len(vals)], vals, label=jname.replace("_joint", ""), linewidth=1.0)
        ax.set_ylabel(FIELD_LABELS.get(field, field))
        ax.legend(fontsize=7, loc="upper left")
        ax.grid(True, alpha=0.3)

    axes[-1, 0].set_xlabel("Time (minutes)")
    title = Path(path).stem
    if dec > 1:
        title += f" (decimated {dec}x)"
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    _save_fig(fig, save_path)


def _plot_power(path: str, joints: list[str] | None, save_path: str | None):
    """Bus power (W) vs time with average line. Prefers measured bus data when available."""
    import matplotlib.pyplot as plt

    d = _load_derived_series(path, joints)
    times = d["times"]
    if not times:
        print("No derived data found.")
        return

    t_min = _to_minutes(times)
    fig, ax = plt.subplots(figsize=(14, 5))

    if d["has_bus"] and d["bus_power_measured"]:
        # Plot measured bus power as primary
        vals = [v for v in d["bus_power_measured"] if v is not None]
        t_meas = [t_min[i] for i, v in enumerate(d["bus_power_measured"]) if v is not None]
        avg_meas = sum(vals) / len(vals) if vals else 0
        ax.plot(t_meas, vals, label=f"Bus power measured (avg={avg_meas:.1f}W)", linewidth=1.2, color="C0")
        ax.axhline(avg_meas, color="C0", linestyle="--", alpha=0.5)
        # Show estimate as secondary
        avg_est = sum(d["bus_power_est"]) / len(d["bus_power_est"])
        ax.plot(
            t_min,
            d["bus_power_est"],
            label=f"Bus power est/Kt (avg={avg_est:.1f}W)",
            linewidth=0.8,
            alpha=0.5,
            color="C2",
        )
    else:
        avg_est = sum(d["bus_power_est"]) / len(d["bus_power_est"]) if d["bus_power_est"] else 0
        ax.plot(t_min, d["bus_power_est"], label=f"Bus power est/Kt (avg={avg_est:.1f}W)", linewidth=1.0)
        ax.axhline(avg_est, color="C0", linestyle="--", alpha=0.5)

    avg_mech = sum(d["bus_power_mech"]) / len(d["bus_power_mech"]) if d["bus_power_mech"] else 0
    ax.plot(
        t_min,
        d["bus_power_mech"],
        label=f"Mechanical power (avg={avg_mech:.1f}W)",
        linewidth=1.0,
        alpha=0.7,
        color="C1",
    )
    ax.axhline(avg_mech, color="C1", linestyle="--", alpha=0.5)

    ax.set_xlabel("Time (minutes)")
    ax.set_ylabel("Power (W)")
    title = f"{Path(path).stem} — Bus Power"
    if d["has_bus"]:
        title += " (measured)"
    else:
        title += " (estimated, Kt approx)"
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save_fig(fig, save_path)


def _plot_current(path: str, joints: list[str] | None, save_path: str | None):
    """Total bus current (A) vs time with average line. Prefers measured when available."""
    import matplotlib.pyplot as plt

    d = _load_derived_series(path, joints)
    times = d["times"]
    if not times:
        print("No derived data found.")
        return

    # Sum absolute estimated currents per timestep
    n_steps = len(times)
    total_est = [0.0] * n_steps
    for jname, vals in d["joint_current"].items():
        for i, v in enumerate(vals):
            if i < n_steps:
                total_est[i] += abs(v)

    t_min = _to_minutes(times)
    fig, ax = plt.subplots(figsize=(14, 5))

    if d["has_bus"] and d["bus_current_measured"]:
        vals = [v for v in d["bus_current_measured"] if v is not None]
        t_meas = [t_min[i] for i, v in enumerate(d["bus_current_measured"]) if v is not None]
        avg_meas = sum(abs(v) for v in vals) / len(vals) if vals else 0
        ax.plot(
            t_meas,
            [abs(v) for v in vals],
            label=f"Bus current measured (avg={avg_meas:.2f}A)",
            linewidth=1.2,
            color="C0",
        )
        ax.axhline(avg_meas, color="C0", linestyle="--", alpha=0.5)
        avg_est = sum(total_est) / len(total_est)
        ax.plot(
            t_min, total_est, label=f"Bus current est/Kt (avg={avg_est:.2f}A)", linewidth=0.8, alpha=0.5, color="C2"
        )
    else:
        avg_est = sum(total_est) / len(total_est) if total_est else 0
        ax.plot(t_min, total_est, label=f"Bus current est/Kt (avg={avg_est:.2f}A)", linewidth=1.0)
        ax.axhline(avg_est, color="C0", linestyle="--", alpha=0.5)

    ax.set_xlabel("Time (minutes)")
    ax.set_ylabel("Current (A)")
    title = f"{Path(path).stem} — Bus Current"
    if d["has_bus"]:
        title += " (measured)"
    else:
        title += " (estimated, Kt approx)"
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save_fig(fig, save_path)


def _plot_joint_current(path: str, joints: list[str] | None, save_path: str | None):
    """Per-joint estimated current vs time."""
    import matplotlib.pyplot as plt

    d = _load_derived_series(path, joints)
    times = d["times"]
    if not times:
        print("No derived data found.")
        return

    t_min = _to_minutes(times)
    fig, ax = plt.subplots(figsize=(14, 5))
    for jname in sorted(d["joint_current"].keys()):
        vals = d["joint_current"][jname]
        ax.plot(t_min[: len(vals)], vals, label=jname.replace("_joint", ""), linewidth=1.0)
    ax.set_xlabel("Time (minutes)")
    ax.set_ylabel("Current (A)")
    ax.set_title(f"{Path(path).stem} — Per-Joint Current (estimated, Kt approx)")
    ax.legend(fontsize=7, loc="upper left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save_fig(fig, save_path)


def _plot_joint_torque(path: str, joints: list[str] | None, save_path: str | None):
    """Per-joint torque vs time."""
    import matplotlib.pyplot as plt

    times, motor_series, dec = _load_series(path, ["tau"], joints)
    if not times:
        print("No data to plot.")
        return

    t_min = _to_minutes(times)
    fig, ax = plt.subplots(figsize=(14, 5))
    for jname in sorted(motor_series["tau"].keys()):
        vals = motor_series["tau"][jname]
        ax.plot(t_min[: len(vals)], vals, label=jname.replace("_joint", ""), linewidth=1.0)
    ax.set_xlabel("Time (minutes)")
    ax.set_ylabel("Torque (Nm)")
    ax.set_title(f"{Path(path).stem} — Per-Joint Torque")
    ax.legend(fontsize=7, loc="upper left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save_fig(fig, save_path)


def _plot_imu(path: str, save_path: str | None):
    """3-panel IMU plot: orientation, angular velocity, acceleration."""
    import matplotlib.pyplot as plt

    times, quat, omega, accel = _load_imu_series(path)
    if not times:
        print("No IMU data found.")
        return

    t_min = _to_minutes(times)
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    for k in ["w", "x", "y", "z"]:
        axes[0].plot(t_min, quat[k], label=f"q_{k}", linewidth=1.0)
    axes[0].set_ylabel("Quaternion")
    axes[0].legend(fontsize=7)
    axes[0].grid(True, alpha=0.3)

    for k in ["x", "y", "z"]:
        axes[1].plot(t_min, omega[k], label=f"omega_{k}", linewidth=1.0)
    axes[1].set_ylabel("Angular vel (rad/s)")
    axes[1].legend(fontsize=7)
    axes[1].grid(True, alpha=0.3)

    for k in ["x", "y", "z"]:
        axes[2].plot(t_min, accel[k], label=f"accel_{k}", linewidth=1.0)
    axes[2].set_ylabel("Acceleration (m/s2)")
    axes[2].legend(fontsize=7)
    axes[2].grid(True, alpha=0.3)

    axes[2].set_xlabel("Time (minutes)")
    fig.suptitle(f"{Path(path).stem} — IMU", fontsize=11)
    fig.tight_layout()
    _save_fig(fig, save_path)


# ── Text modes ──


def _print_summary(path: str):
    """Print a concise text summary of the JSONL telemetry file."""
    count = 0
    t_first = None
    t_last = None
    joint_stats: dict[str, dict[str, list]] = {}
    imu_max_omega = 0.0
    has_imu = False
    pwr_sum = 0.0
    pwr_max = 0.0
    cur_sum = 0.0
    cur_max = 0.0
    pwr_count = 0
    bus_pwr_sum = 0.0
    bus_pwr_max = 0.0
    bus_cur_sum = 0.0
    bus_cur_max = 0.0
    bus_count = 0

    for record in _iter_jsonl(path):
        count += 1
        t = record["t"]
        if t_first is None:
            t_first = t
        t_last = t

        for jname, jdata in record.get("motors", {}).items():
            if jname not in joint_stats:
                joint_stats[jname] = {}
            for field, val in jdata.items():
                if not isinstance(val, (int, float)):
                    continue
                if field not in joint_stats[jname]:
                    joint_stats[jname][field] = [val, val, val, 1]
                else:
                    s = joint_stats[jname][field]
                    s[0] = min(s[0], val)
                    s[1] = max(s[1], val)
                    s[2] += val
                    s[3] += 1

        derived = record.get("derived")
        if derived:
            bp = derived.get("bus_power_est", 0.0)
            pwr_sum += bp
            if bp > pwr_max:
                pwr_max = bp
            jd = derived.get("joints", {})
            tc = sum(abs(d.get("current_est", 0.0)) for d in jd.values())
            cur_sum += tc
            if tc > cur_max:
                cur_max = tc
            pwr_count += 1

        bus = record.get("bus")
        if bus:
            bp = bus.get("power", 0.0)
            bc = abs(bus.get("current", 0.0))
            bus_pwr_sum += bp
            if bp > bus_pwr_max:
                bus_pwr_max = bp
            bus_cur_sum += bc
            if bc > bus_cur_max:
                bus_cur_max = bc
            bus_count += 1

        imu = record.get("imu")
        if imu:
            has_imu = True
            omega = imu.get("omega", [0, 0, 0])
            mag = sum(x * x for x in omega) ** 0.5
            if mag > imu_max_omega:
                imu_max_omega = mag

    if count == 0:
        print("Empty file.")
        return

    duration = (t_last - t_first) if t_first is not None and t_last is not None else 0
    rate = count / duration if duration > 0 else 0

    print(f"File:       {path}")
    print(f"Samples:    {count}")
    print(f"Duration:   {duration:.1f}s ({duration / 60:.1f} min)")
    print(f"Rate:       {rate:.1f} Hz")

    if bus_count > 0:
        print("\nBus power (MEASURED — external analyzer):")
        print(f"  Avg bus power:    {bus_pwr_sum / bus_count:.1f} W")
        print(f"  Peak bus power:   {bus_pwr_max:.1f} W")
        print(f"  Avg bus current:  {bus_cur_sum / bus_count:.2f} A")
        print(f"  Peak bus current: {bus_cur_max:.2f} A")
        print(f"  Total energy:     {bus_pwr_sum * 0.5:.1f} J")

    if pwr_count > 0:
        label = "estimated, Kt approx" if bus_count == 0 else "estimated for comparison"
        print(f"\nBus power ({label}):")
        print(f"  Avg bus power:    {pwr_sum / pwr_count:.1f} W")
        print(f"  Peak bus power:   {pwr_max:.1f} W")
        print(f"  Avg bus current:  {cur_sum / pwr_count:.2f} A")
        print(f"  Peak bus current: {cur_max:.2f} A")
        print(f"  Total energy:     {pwr_sum * 0.5:.1f} J")
        if bus_count > 0 and pwr_count > 0:
            cal = (bus_cur_sum / bus_count) / (cur_sum / pwr_count) if (cur_sum / pwr_count) > 0 else 0
            print(f"  Kt correction factor: {cal:.3f}  (measured/estimated avg current)")

    print()

    # Per-joint table
    joints = sorted(joint_stats.keys())
    key_fields = ["temp", "tau", "q", "dq", "ddq", "volt"]

    hdr = f"{'Joint':<30}"
    for f in key_fields:
        if any(f in joint_stats[j] for j in joints):
            hdr += f"  {f:>8} min  {f:>8} max  {f:>8} mean"
    print(hdr)
    print("-" * len(hdr))

    for jname in joints:
        row = f"{jname:<30}"
        for f in key_fields:
            if not any(f in joint_stats[j] for j in joints):
                continue
            s = joint_stats[jname].get(f)
            if s:
                mean = s[2] / s[3]
                row += f"  {s[0]:>12.2f}  {s[1]:>12.2f}  {mean:>12.2f}"
            else:
                row += f"  {'':>12}  {'':>12}  {'':>12}"
        print(row)

    if has_imu:
        print(f"\nIMU: max |omega| = {imu_max_omega:.3f} rad/s")


def _print_samples(path: str, n: int, mode: str):
    """Print first or last N samples as a formatted table."""
    if mode == "tail":
        buf = []
        for record in _iter_jsonl(path):
            buf.append(record)
            if len(buf) > n:
                buf.pop(0)
        samples = buf
    else:
        samples = []
        for record in _iter_jsonl(path):
            samples.append(record)
            if len(samples) >= n:
                break

    if not samples:
        print("No samples.")
        return

    all_joints = set()
    for s in samples:
        all_joints.update(s.get("motors", {}).keys())
    joints = sorted(all_joints)

    hdr = f"{'t':>8}"
    for j in joints:
        short = j.replace("_joint", "")
        hdr += f"  {short + '/temp':>14}  {short + '/tau':>12}  {short + '/q':>10}"
    print(hdr)
    print("-" * len(hdr))

    for s in samples:
        row = f"{s['t']:>8.2f}"
        motors = s.get("motors", {})
        for j in joints:
            jd = motors.get(j, {})
            row += f"  {_fmt(jd.get('temp'), 14)}  {_fmt(jd.get('tau'), 12)}  {_fmt(jd.get('q'), 10)}"
        print(row)


def _fmt(val, width: int) -> str:
    if isinstance(val, (int, float)):
        return f"{val:>{width}.3f}"
    return f"{'':>{width}}"


# ── Cross-trial comparison ──


def _load_reports(results_dir: str) -> list[dict]:
    """Scan results directory for report.json files and load them."""
    reports = []
    for p in sorted(Path(results_dir).rglob("report.json")):
        try:
            with open(p) as f:
                data = json.load(f)
            # Infer trial name from parent directory
            data["_trial"] = p.parent.name
            data["_path"] = str(p)
            reports.append(data)
        except Exception as e:
            print(f"Warning: skipping {p}: {e}", file=sys.stderr)
    return reports


def _parse_trial_label(trial_name: str) -> tuple[str, str]:
    """Parse trial dir name like 'g1_bicep_curl_right_peak_4kg' into (task, condition)."""
    # Strip robot prefix (g1_, x1_)
    name = trial_name
    for prefix in ("g1_", "x1_", "t1_"):
        if name.startswith(prefix):
            name = name[len(prefix) :]
            break

    # Check for peak payload condition
    if "_peak_" in name:
        idx = name.index("_peak_")
        task = name[:idx]
        condition = name[idx + 1 :]  # e.g. "peak_4kg"
    else:
        task = name
        condition = "continuous"

    return task, condition


def _compare_plots(results_dir: str, save_dir: str | None):
    """Generate cross-trial comparison plots from report.json files."""
    import matplotlib.pyplot as plt
    import numpy as np

    reports = _load_reports(results_dir)
    if not reports:
        print(f"No report.json files found in {results_dir}")
        return

    # Extract data per trial
    trials = []
    for r in reports:
        task, condition = _parse_trial_label(r["_trial"])
        power = r.get("power", {})
        # Prefer measured bus values over estimates
        measured = power.get("bus_power_source") == "measured"
        avg_pwr = power.get("measured_avg_bus_power_w") if measured else None
        avg_pwr = avg_pwr if avg_pwr is not None else power.get("avg_bus_power_w", 0)
        peak_pwr = power.get("measured_peak_bus_power_w") if measured else None
        peak_pwr = peak_pwr if peak_pwr is not None else power.get("peak_bus_power_w", 0)
        avg_cur = power.get("measured_avg_bus_current_a") if measured else None
        avg_cur = avg_cur if avg_cur is not None else power.get("avg_bus_current_a", 0)
        peak_cur = power.get("measured_peak_bus_current_a") if measured else None
        peak_cur = peak_cur if peak_cur is not None else power.get("peak_bus_current_a", 0)
        total_e = power.get("measured_total_energy_j") if measured else None
        total_e = total_e if total_e is not None else power.get("total_energy_j", 0)
        trials.append(
            {
                "trial": r["_trial"],
                "task": task,
                "condition": condition,
                "measured": measured,
                "duration_s": r.get("duration_s", 0),
                "avg_power": avg_pwr,
                "peak_power": peak_pwr,
                "avg_current": avg_cur,
                "peak_current": peak_cur,
                "total_energy": total_e,
                "peak_torques": power.get("per_joint_peak_torque", {}),
                "peak_currents": power.get("per_joint_peak_current", {}),
            }
        )

    if not trials:
        print("No trials with power data found.")
        return

    labels = [t["trial"] for t in trials]
    x = np.arange(len(labels))

    def _bar_chart(values, ylabel, title, filename):
        fig, ax = plt.subplots(figsize=(max(10, len(labels) * 1.2), 5))
        bars = ax.bar(x, values, color="steelblue", edgecolor="white")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.3, axis="y")
        for bar, val in zip(bars, values):
            if val > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height(),
                    f"{val:.1f}",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                )
        fig.tight_layout()
        if save_dir:
            path = Path(save_dir) / filename
            path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(str(path), dpi=150, bbox_inches="tight")
            print(f"Saved {path}")
        else:
            plt.show()
        plt.close(fig)

    # 1. Average bus power
    _bar_chart([t["avg_power"] for t in trials], "Power (W)", "Average Bus Power by Trial", "avg_bus_power.png")

    # 2. Peak bus power
    _bar_chart([t["peak_power"] for t in trials], "Power (W)", "Peak Bus Power by Trial", "peak_bus_power.png")

    # 3. Average bus current
    _bar_chart([t["avg_current"] for t in trials], "Current (A)", "Average Bus Current by Trial", "avg_bus_current.png")

    # 4. Peak bus current
    _bar_chart([t["peak_current"] for t in trials], "Current (A)", "Peak Bus Current by Trial", "peak_bus_current.png")

    # 5. Total energy (normalized by duration)
    energy_per_min = []
    for t in trials:
        dur = t["duration_s"]
        e = t["total_energy"] / (dur / 60.0) if dur > 0 else 0
        energy_per_min.append(e)
    _bar_chart(energy_per_min, "Energy (J/min)", "Energy Rate by Trial", "energy_rate.png")

    # 6. Per-joint peak torque heatmap
    all_joints = set()
    for t in trials:
        all_joints.update(t["peak_torques"].keys())
    if all_joints:
        joints_sorted = sorted(all_joints)
        heatmap = np.zeros((len(joints_sorted), len(trials)))
        for j, trial in enumerate(trials):
            for i, jname in enumerate(joints_sorted):
                heatmap[i, j] = trial["peak_torques"].get(jname, 0)

        fig, ax = plt.subplots(figsize=(max(10, len(trials) * 1.2), max(4, len(joints_sorted) * 0.6)))
        im = ax.imshow(heatmap, aspect="auto", cmap="YlOrRd")
        ax.set_xticks(range(len(trials)))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(len(joints_sorted)))
        ax.set_yticklabels([j.replace("_joint", "") for j in joints_sorted], fontsize=8)
        ax.set_title("Per-Joint Peak Torque (Nm)")
        # Annotate cells
        for i in range(len(joints_sorted)):
            for j in range(len(trials)):
                val = heatmap[i, j]
                if val > 0:
                    ax.text(
                        j,
                        i,
                        f"{val:.1f}",
                        ha="center",
                        va="center",
                        fontsize=7,
                        color="white" if val > heatmap.max() * 0.6 else "black",
                    )
        fig.colorbar(im, ax=ax, label="Torque (Nm)")
        fig.tight_layout()
        if save_dir:
            path = Path(save_dir) / "peak_torque_heatmap.png"
            fig.savefig(str(path), dpi=150, bbox_inches="tight")
            print(f"Saved {path}")
        else:
            plt.show()
        plt.close(fig)

    print(f"\nComparison complete: {len(trials)} trials analyzed")


# ── CLI ──

NAMED_PLOTS = ["power", "current", "joint-current", "joint-torque"]


def main():
    parser = argparse.ArgumentParser(description="JSONL telemetry viewer/plotter")
    parser.add_argument("file", nargs="?", help="Path to telemetry .jsonl file")
    parser.add_argument("--fields", default="temp", help="Comma-separated fields: temp,tau,q,cmd_q,dq,ddq,volt")
    parser.add_argument("--joints", default=None, help="Comma-separated joint names to filter")
    parser.add_argument("--imu", action="store_true", help="Plot IMU data")
    parser.add_argument(
        "--plot", default=None, choices=NAMED_PLOTS, help="Named plot type: power, current, joint-current, joint-torque"
    )
    parser.add_argument("--save", default=None, help="Save plot to file")
    parser.add_argument("--summary", action="store_true", help="Print text summary")
    parser.add_argument("--head", type=int, default=None, metavar="N", help="Print first N samples")
    parser.add_argument("--tail", type=int, default=None, metavar="N", help="Print last N samples")
    parser.add_argument("--compare", default=None, metavar="DIR", help="Cross-trial comparison from results directory")
    parser.add_argument(
        "--compare-save", default=None, metavar="DIR", help="Save comparison plots to directory (default: show)"
    )

    args = parser.parse_args()

    # Cross-trial comparison mode
    if args.compare:
        _compare_plots(args.compare, args.compare_save)
        return

    # All other modes need a file
    if not args.file:
        parser.error("file is required (unless using --compare)")
    if not Path(args.file).exists():
        print(f"File not found: {args.file}", file=sys.stderr)
        sys.exit(1)

    # Text modes
    if args.summary:
        _print_summary(args.file)
        return
    if args.head is not None:
        _print_samples(args.file, args.head, "head")
        return
    if args.tail is not None:
        _print_samples(args.file, args.tail, "tail")
        return

    joints = args.joints.split(",") if args.joints else None

    # Named plot modes
    if args.plot == "power":
        _plot_power(args.file, joints, args.save)
    elif args.plot == "current":
        _plot_current(args.file, joints, args.save)
    elif args.plot == "joint-current":
        _plot_joint_current(args.file, joints, args.save)
    elif args.plot == "joint-torque":
        _plot_joint_torque(args.file, joints, args.save)
    elif args.imu:
        _plot_imu(args.file, args.save)
    else:
        fields = [f.strip() for f in args.fields.split(",")]
        _plot_motors(args.file, fields, joints, args.save)


if __name__ == "__main__":
    main()
