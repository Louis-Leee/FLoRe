"""1kHz binary telemetry recorder with pre-allocated numpy buffers.

Records raw motor + IMU state at 1kHz into numpy arrays, flushing chunks
to disk from a background thread. Convert to JSONL post-test with bin2jsonl.py.

Data layout (float32):
  Col 0: timestamp (seconds since recording start)
  Cols 1..N*6: N joints × 6 fields (cmd_q, q, dq, tau, temp, volt)
  Remaining: IMU (quat:4, omega:3, accel:3, rpy:4)

At 1kHz with 29 joints: 189 cols × 4 bytes = 756 bytes/tick = 756 KB/s.
"""

from __future__ import annotations

import json
import queue
import threading
from pathlib import Path

import numpy as np


class ColumnLayout:
    """Maps joint names + IMU fields to flat array columns."""

    MOTOR_FIELDS = ("cmd_q", "q", "dq", "tau", "temp", "volt")
    FIELDS_PER_JOINT = len(MOTOR_FIELDS)

    IMU_FIELDS = {"quat": 4, "omega": 3, "accel": 3, "rpy": 4}
    IMU_TOTAL = sum(IMU_FIELDS.values())  # 14

    def __init__(self, joint_names: tuple[str, ...]):
        self.joint_names = joint_names
        self.n_joints = len(joint_names)
        self.n_motor_cols = self.n_joints * self.FIELDS_PER_JOINT
        self.n_cols = 1 + self.n_motor_cols + self.IMU_TOTAL

        # Pre-compute motor column indices per field (vectorized assignment)
        self._motor_offsets: dict[str, np.ndarray] = {}
        for fi, field in enumerate(self.MOTOR_FIELDS):
            self._motor_offsets[field] = np.arange(self.n_joints, dtype=np.intp) * self.FIELDS_PER_JOINT + 1 + fi

        # IMU column ranges
        imu_start = 1 + self.n_motor_cols
        self._imu_offsets: dict[str, tuple[int, int]] = {}
        offset = imu_start
        for field, size in self.IMU_FIELDS.items():
            self._imu_offsets[field] = (offset, offset + size)
            offset += size

    def motor_cols(self, field: str) -> np.ndarray:
        """Column indices for a motor field across all joints."""
        return self._motor_offsets[field]

    def imu_slice(self, field: str) -> tuple[int, int]:
        """(start, end) column indices for an IMU field."""
        return self._imu_offsets[field]

    def to_header(self) -> dict:
        """Serialize layout for telemetry_header.json."""
        return {
            "joint_names": list(self.joint_names),
            "motor_fields": list(self.MOTOR_FIELDS),
            "imu_fields": {k: v for k, v in self.IMU_FIELDS.items()},
            "n_cols": self.n_cols,
            "dtype": "float32",
        }

    @classmethod
    def from_header(cls, header: dict) -> ColumnLayout:
        """Reconstruct from telemetry_header.json."""
        return cls(tuple(header["joint_names"]))


class TelemetryRecorder:
    """Pre-allocated numpy buffer recorder for 1kHz telemetry.

    Usage:
        layout = ColumnLayout(dof_names)
        rec = TelemetryRecorder(layout, output_dir)
        # In hot loop:
        rec.record_tick_raw(t, cmd_q, q, dq, tau, temp, volt, ...)
        # On exit:
        npy_path = rec.stop()
    """

    CHUNK_ROWS = 5000  # 5 seconds at 1kHz

    def __init__(self, layout: ColumnLayout, output_dir: Path):
        self.layout = layout
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self._buf = np.zeros((self.CHUNK_ROWS, layout.n_cols), dtype=np.float32)
        self._row = 0
        self._chunk_idx = 0
        self._chunk_paths: list[Path] = []
        self._total_rows = 0

        self._queue: queue.Queue[tuple[np.ndarray, Path] | None] = queue.Queue(maxsize=4)
        self._writer = threading.Thread(target=self._writer_loop, daemon=True)
        self._writer.start()
        self._stopped = False

    def record_tick_raw(
        self,
        timestamp: float,
        cmd_q: np.ndarray,
        q: np.ndarray,
        dq: np.ndarray,
        tau: np.ndarray,
        temp: np.ndarray,
        volt: np.ndarray,
        imu_quat: np.ndarray | None = None,
        imu_omega: np.ndarray | None = None,
        imu_accel: np.ndarray | None = None,
        imu_rpy: np.ndarray | None = None,
    ) -> None:
        """Record one tick. All motor arrays must have n_joints elements."""
        row = self._buf[self._row]
        row[0] = timestamp

        layout = self.layout
        row[layout.motor_cols("cmd_q")] = cmd_q
        row[layout.motor_cols("q")] = q
        row[layout.motor_cols("dq")] = dq
        row[layout.motor_cols("tau")] = tau
        row[layout.motor_cols("temp")] = temp
        row[layout.motor_cols("volt")] = volt

        if imu_quat is not None:
            s, e = layout.imu_slice("quat")
            row[s:e] = imu_quat[:4]
        if imu_omega is not None:
            s, e = layout.imu_slice("omega")
            row[s:e] = imu_omega[:3]
        if imu_accel is not None:
            s, e = layout.imu_slice("accel")
            row[s:e] = imu_accel[:3]
        if imu_rpy is not None:
            s, e = layout.imu_slice("rpy")
            n = min(len(imu_rpy), e - s)
            row[s : s + n] = imu_rpy[:n]

        self._row += 1
        if self._row >= self.CHUNK_ROWS:
            self._flush_chunk()

    def _flush_chunk(self) -> None:
        """Swap buffer and enqueue current chunk for background write."""
        chunk_path = self.output_dir / f"chunk_{self._chunk_idx:04d}.npy"
        filled = self._buf[: self._row].copy()
        self._total_rows += self._row
        self._queue.put((filled, chunk_path))
        self._chunk_paths.append(chunk_path)
        self._chunk_idx += 1
        self._buf[:] = 0
        self._row = 0

    def _writer_loop(self) -> None:
        """Background thread: write numpy chunks to disk."""
        while True:
            item = self._queue.get()
            if item is None:
                break
            data, path = item
            np.save(path, data)

    def stop(self) -> Path:
        """Flush remaining data, concatenate chunks, write header.

        Returns path to telemetry.npy.
        """
        if self._stopped:
            return self.output_dir / "telemetry.npy"
        self._stopped = True

        if self._row > 0:
            self._flush_chunk()

        self._queue.put(None)
        self._writer.join(timeout=30)

        final_path = self.output_dir / "telemetry.npy"
        if self._chunk_paths:
            chunks = [np.load(p) for p in self._chunk_paths]
            combined = np.concatenate(chunks, axis=0)
            np.save(final_path, combined)
            for p in self._chunk_paths:
                p.unlink(missing_ok=True)
        else:
            np.save(final_path, np.zeros((0, self.layout.n_cols), dtype=np.float32))

        header = self.layout.to_header()
        header["total_rows"] = int(self._total_rows)
        with open(self.output_dir / "telemetry_header.json", "w") as f:
            json.dump(header, f, indent=2)

        return final_path

    @property
    def total_rows(self) -> int:
        """Total rows recorded (including current buffer)."""
        return self._total_rows + self._row
