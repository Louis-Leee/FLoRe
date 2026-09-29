"""Convert binary telemetry (.npy) to JSONL format for telemetry_plot.py.

Reads telemetry.npy + telemetry_header.json, writes telemetry.jsonl in the
exact format expected by telemetry_plot.py (motors, derived, imu dicts).

Usage:
    # From recording directory (default: decimate 100x → 1kHz to 10Hz)
    python -m extensions_common.thermal.bin2jsonl results/g1_walking/

    # Custom decimation
    python -m extensions_common.thermal.bin2jsonl results/g1_walking/ --decimate 50

    # Full rate (no decimation)
    python -m extensions_common.thermal.bin2jsonl results/g1_walking/ --decimate 1
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from extensions_common.thermal.telemetry_recorder import ColumnLayout

# Estimated torque constants Kt (Nm/A) per joint group.
# Duplicated from runner.py to keep bin2jsonl standalone.
_DEFAULT_KT = 0.35
_KT_BY_GROUP = {"shoulder": 0.35, "elbow": 0.35, "wrist": 0.35, "hip": 0.50, "knee": 0.50, "ankle": 0.40, "waist": 0.50}


def _get_kt(joint_name: str) -> float:
    """Return estimated Kt for a joint based on its name group."""
    base = joint_name.replace("_joint", "")
    for part in base.split("_"):
        if part in _KT_BY_GROUP:
            return _KT_BY_GROUP[part]
    return _DEFAULT_KT


def convert(input_dir: Path, output_path: Path | None = None, decimate: int = 100) -> Path:
    """Convert telemetry.npy + header to JSONL.

    Args:
        input_dir: Directory containing telemetry.npy and telemetry_header.json.
        output_path: Output JSONL path (default: input_dir/telemetry.jsonl).
        decimate: Keep every Nth sample (default 100 = 1kHz → 10Hz).

    Returns:
        Path to written JSONL file.
    """
    input_dir = Path(input_dir)
    header_path = input_dir / "telemetry_header.json"
    npy_path = input_dir / "telemetry.npy"

    with open(header_path) as f:
        header = json.load(f)

    layout = ColumnLayout.from_header(header)
    data = np.load(npy_path)

    if output_path is None:
        output_path = input_dir / "telemetry.jsonl"

    if decimate > 1:
        data = data[::decimate]

    joint_names = layout.joint_names
    motor_fields = layout.MOTOR_FIELDS

    # Pre-compute Kt per joint
    kt_values = {jname: _get_kt(jname) for jname in joint_names}

    n_rows = len(data)
    with open(output_path, "w") as f:
        for row in data:
            t = float(row[0])

            # Build motors dict
            motors = {}
            for ji, jname in enumerate(joint_names):
                entry = {}
                for fi, field in enumerate(motor_fields):
                    col = 1 + ji * layout.FIELDS_PER_JOINT + fi
                    val = float(row[col])
                    if val != 0.0 or field in ("q", "dq", "tau", "temp"):
                        entry[field] = round(val, 6)
                motors[jname] = entry

            # Compute derived fields (current, power)
            derived_joints = {}
            bus_power_est = 0.0
            bus_power_mech = 0.0
            for jname, jdata in motors.items():
                kt = kt_values[jname]
                tau = jdata.get("tau", 0.0)
                dq = jdata.get("dq", 0.0)
                volt = jdata.get("volt", 0.0)
                current_est = tau / kt if kt > 0 else 0.0
                mech_power = tau * dq
                bus_power_est += volt * abs(current_est)
                bus_power_mech += mech_power
                derived_joints[jname] = {"current_est": round(current_est, 4), "mech_power": round(mech_power, 3)}
            derived = {
                "bus_power_est": round(bus_power_est, 3),
                "bus_power_mechanical": round(bus_power_mech, 3),
                "joints": derived_joints,
            }

            record: dict = {"t": round(t, 3), "motors": motors, "derived": derived}

            # IMU fields
            imu_data = {}
            for field, size in layout.IMU_FIELDS.items():
                s, e = layout.imu_slice(field)
                vals = row[s:e].tolist()
                if any(v != 0.0 for v in vals):
                    imu_data[field] = [round(v, 6) for v in vals]
            if imu_data:
                record["imu"] = imu_data

            f.write(json.dumps(record, separators=(",", ":")) + "\n")

    print(f"Converted {n_rows} rows → {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Convert binary telemetry to JSONL")
    parser.add_argument("input_dir", type=Path, help="Directory with telemetry.npy")
    parser.add_argument("--decimate", type=int, default=100, help="Keep every Nth sample (default: 100 = 1kHz→10Hz)")
    parser.add_argument("--output", type=Path, default=None, help="Output JSONL path")
    args = parser.parse_args()

    convert(args.input_dir, output_path=args.output, decimate=args.decimate)


if __name__ == "__main__":
    main()
