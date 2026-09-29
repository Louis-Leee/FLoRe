#!/usr/bin/env python3
"""Merge JSONL telemetry with external power analyzer CSV.

Takes the test runner's JSONL telemetry file and a CSV log from an external
power analyzer, merging them on timestamp into a combined JSONL with a "bus"
key containing measured voltage, current, and power.

The analyzer CSV format is configurable via column mapping flags.

Usage:
    # Basic merge (default CSV column names: time, voltage, current, power)
    python telemetry_merge.py test.jsonl analyzer.csv --output merged.jsonl

    # Custom column names
    python telemetry_merge.py test.jsonl analyzer.csv --output merged.jsonl \
        --time-col timestamp_s --voltage-col v_bus --current-col i_bus --power-col p_bus

    # With time offset (analyzer started 2.5s after robot)
    python telemetry_merge.py test.jsonl analyzer.csv --output merged.jsonl \
        --time-offset 2.5

    # Update report.json with measured bus stats
    python telemetry_merge.py test.jsonl analyzer.csv --output merged.jsonl \
        --update-report results/g1_bicep_curl_right/report.json
"""

from __future__ import annotations

import argparse
import bisect
import csv
import json
import sys
from pathlib import Path


def _load_analyzer_csv(
    path: str, time_col: str, voltage_col: str, current_col: str, power_col: str, time_offset: float
) -> list[dict]:
    """Load analyzer CSV into list of {t, voltage, current, power} dicts."""
    records = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        available = set(reader.fieldnames or [])

        missing = []
        for col_name, col_val in [
            ("time", time_col),
            ("voltage", voltage_col),
            ("current", current_col),
            ("power", power_col),
        ]:
            if col_val and col_val not in available:
                missing.append(f"{col_name}={col_val!r}")
        if missing:
            print(f"Warning: columns not found in CSV: {', '.join(missing)}", file=sys.stderr)
            print(f"Available columns: {sorted(available)}", file=sys.stderr)

        for row in reader:
            try:
                t = float(row[time_col]) + time_offset
            except (KeyError, ValueError):
                continue

            entry = {"t": t}
            if voltage_col and voltage_col in row:
                try:
                    entry["voltage"] = float(row[voltage_col])
                except ValueError:
                    pass
            if current_col and current_col in row:
                try:
                    entry["current"] = float(row[current_col])
                except ValueError:
                    pass
            if power_col and power_col in row:
                try:
                    entry["power"] = float(row[power_col])
                except ValueError:
                    pass
            records.append(entry)

    return records


def _find_nearest(analyzer_times: list[float], target_t: float, max_gap: float = 1.0) -> int | None:
    """Find the nearest analyzer sample to target_t. Returns index or None if too far."""
    if not analyzer_times:
        return None
    idx = bisect.bisect_left(analyzer_times, target_t)
    candidates = []
    if idx > 0:
        candidates.append(idx - 1)
    if idx < len(analyzer_times):
        candidates.append(idx)
    best = min(candidates, key=lambda i: abs(analyzer_times[i] - target_t))
    if abs(analyzer_times[best] - target_t) > max_gap:
        return None
    return best


def merge(
    jsonl_path: str,
    csv_path: str,
    output_path: str,
    time_col: str = "time",
    voltage_col: str = "voltage",
    current_col: str = "current",
    power_col: str = "power",
    time_offset: float = 0.0,
    max_gap: float = 1.0,
) -> dict:
    """Merge JSONL telemetry with analyzer CSV.

    Returns summary stats dict for the merged bus data.
    """
    analyzer = _load_analyzer_csv(csv_path, time_col, voltage_col, current_col, power_col, time_offset)
    if not analyzer:
        print("Warning: no valid analyzer records loaded", file=sys.stderr)

    analyzer_times = [r["t"] for r in analyzer]

    merged_count = 0
    total_count = 0
    bus_power_sum = 0.0
    bus_power_max = 0.0
    bus_current_sum = 0.0
    bus_current_max = 0.0
    bus_samples = 0

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(jsonl_path) as fin, open(output_path, "w") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            total_count += 1
            record = json.loads(line)
            t = record["t"]

            idx = _find_nearest(analyzer_times, t, max_gap)
            if idx is not None:
                a = analyzer[idx]
                bus = {}
                if "voltage" in a:
                    bus["voltage"] = round(a["voltage"], 3)
                if "current" in a:
                    bus["current"] = round(a["current"], 4)
                if "power" in a:
                    bus["power"] = round(a["power"], 3)
                if bus:
                    record["bus"] = bus
                    merged_count += 1

                    p = bus.get("power", 0.0)
                    c = abs(bus.get("current", 0.0))
                    bus_power_sum += p
                    if p > bus_power_max:
                        bus_power_max = p
                    bus_current_sum += c
                    if c > bus_current_max:
                        bus_current_max = c
                    bus_samples += 1

            fout.write(json.dumps(record, separators=(",", ":")) + "\n")

    print(f"Merged {merged_count}/{total_count} records with analyzer data")
    print(f"Output: {output_path}")

    stats = {}
    if bus_samples > 0:
        stats = {
            "bus_power_source": "measured",
            "avg_bus_power_w": round(bus_power_sum / bus_samples, 2),
            "peak_bus_power_w": round(bus_power_max, 2),
            "avg_bus_current_a": round(bus_current_sum / bus_samples, 3),
            "peak_bus_current_a": round(bus_current_max, 3),
            "total_energy_j": round(bus_power_sum * 0.5, 2),  # assuming 2Hz sample rate
            "merged_samples": bus_samples,
        }
        # Calibration factor: actual / estimated
        print("\nMeasured bus stats:")
        print(f"  Avg power:  {stats['avg_bus_power_w']:.1f} W")
        print(f"  Peak power: {stats['peak_bus_power_w']:.1f} W")
        print(f"  Avg current:  {stats['avg_bus_current_a']:.2f} A")
        print(f"  Peak current: {stats['peak_bus_current_a']:.2f} A")

    return stats


def _update_report(report_path: str, bus_stats: dict):
    """Update an existing report.json with measured bus stats."""
    with open(report_path) as f:
        report = json.load(f)

    power = report.get("power", {})
    power["bus_power_source"] = "measured"
    power["measured_avg_bus_power_w"] = bus_stats.get("avg_bus_power_w")
    power["measured_peak_bus_power_w"] = bus_stats.get("peak_bus_power_w")
    power["measured_avg_bus_current_a"] = bus_stats.get("avg_bus_current_a")
    power["measured_peak_bus_current_a"] = bus_stats.get("peak_bus_current_a")
    power["measured_total_energy_j"] = bus_stats.get("total_energy_j")
    # Keep estimated values for comparison
    power["joint_current_source"] = "estimated_kt"
    power["note"] = "Bus-level values from external analyzer. Per-joint current still estimated via Kt."
    report["power"] = power

    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Updated {report_path} with measured bus stats")


def main():
    parser = argparse.ArgumentParser(description="Merge JSONL telemetry with external power analyzer CSV")
    parser.add_argument("jsonl", help="Path to telemetry .jsonl file")
    parser.add_argument("csv", help="Path to analyzer CSV file")
    parser.add_argument("--output", "-o", required=True, help="Output merged .jsonl path")
    parser.add_argument("--time-col", default="time", help="CSV column name for timestamp (default: time)")
    parser.add_argument("--voltage-col", default="voltage", help="CSV column for bus voltage (default: voltage)")
    parser.add_argument("--current-col", default="current", help="CSV column for bus current (default: current)")
    parser.add_argument("--power-col", default="power", help="CSV column for bus power (default: power)")
    parser.add_argument(
        "--time-offset",
        type=float,
        default=0.0,
        help="Offset added to CSV timestamps (seconds). Positive = analyzer started later.",
    )
    parser.add_argument("--max-gap", type=float, default=1.0, help="Max time gap (s) for matching (default: 1.0)")
    parser.add_argument("--update-report", default=None, help="Path to report.json to update with measured stats")

    args = parser.parse_args()

    for path in [args.jsonl, args.csv]:
        if not Path(path).exists():
            print(f"File not found: {path}", file=sys.stderr)
            sys.exit(1)

    stats = merge(
        args.jsonl,
        args.csv,
        args.output,
        time_col=args.time_col,
        voltage_col=args.voltage_col,
        current_col=args.current_col,
        power_col=args.power_col,
        time_offset=args.time_offset,
        max_gap=args.max_gap,
    )

    if args.update_report and stats:
        _update_report(args.update_report, stats)


if __name__ == "__main__":
    main()
