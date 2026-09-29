"""Rosbag recorder — local or remote (SSH).

Starts/stops ``ros2 bag record`` either locally or on a remote robot via SSH,
and optionally pulls the resulting bag back to the operator machine.

Supports both ROS2 Foxy (explicit topic list) and Humble+ (``--regex``).
"""

from __future__ import annotations

import os
import signal
import subprocess
import time
from pathlib import Path


class RosbagRecorder:
    """Manage a ``ros2 bag record`` process, locally or over SSH.

    Parameters
    ----------
    ros_setup_cmd
        Shell command to source the ROS environment, e.g.
        ``"source /opt/ros/humble/setup.bash && export ROS_DOMAIN_ID=33"``.
        Can be empty if ROS is already on PATH (e.g. sourced in .bashrc).
    topics
        Explicit list of topic names (for Foxy which lacks ``--regex``).
        Mutually exclusive with *regex*.
    regex
        Topic regex pattern, e.g. ``"/telemetry/.*"`` (Humble+ only).
        Mutually exclusive with *topics*.
    ssh_target
        SSH target, e.g. ``"unitree@192.168.123.164"``.
        If empty/None, recording runs locally.
    bag_dir
        Directory where bags are stored (default ``~/rosbags``).
    """

    def __init__(
        self,
        ros_setup_cmd: str = "",
        *,
        topics: list[str] | None = None,
        regex: str | None = None,
        ssh_target: str = "",
        bag_dir: str = "~/rosbags",
    ):
        if topics and regex:
            raise ValueError("Specify either topics or regex, not both")
        if not topics and not regex:
            raise ValueError("Must specify either topics or regex")

        self._ssh = ssh_target or ""
        self._ros_setup = ros_setup_cmd
        self._topics = topics
        self._regex = regex
        self._bag_dir = bag_dir
        self._is_local = not self._ssh
        self._proc: subprocess.Popen | None = None
        self._bag_path: str | None = None

    def _build_record_cmd(self) -> str:
        if self._regex:
            return f"ros2 bag record --regex '{self._regex}' -o {self._bag_path}"
        else:
            topic_str = " ".join(self._topics)
            return f"ros2 bag record {topic_str} -o {self._bag_path}"

    def start(self, bag_name: str) -> str:
        """Start recording. Returns the bag path.

        Parameters
        ----------
        bag_name
            Name for the bag directory (e.g. ``"telemetry_20260407_1430"``).
        """
        if self._proc is not None:
            raise RuntimeError("Recording already in progress")

        self._bag_path = f"{self._bag_dir}/{bag_name}"
        record_cmd = self._build_record_cmd()

        if self._is_local:
            # Expand ~ for local paths
            bag_dir_expanded = os.path.expanduser(self._bag_dir)
            os.makedirs(bag_dir_expanded, exist_ok=True)
            self._bag_path = f"{bag_dir_expanded}/{bag_name}"
            record_cmd = self._build_record_cmd()

            if self._ros_setup:
                shell_cmd = f"{self._ros_setup} && {record_cmd}"
            else:
                shell_cmd = record_cmd

            self._proc = subprocess.Popen(
                ["bash", "-c", shell_cmd],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                # Use process group so we can signal the whole group
                preexec_fn=os.setsid,
            )
        else:
            full_cmd = f"{self._ros_setup} && mkdir -p {self._bag_dir} && {record_cmd}"
            self._proc = subprocess.Popen(
                ["ssh", "-tt", self._ssh, full_cmd],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )

        # Give ros2 bag record a moment to start
        time.sleep(2.0)

        if self._proc.poll() is not None:
            output = self._proc.stdout.read().decode(errors="replace") if self._proc.stdout else ""
            raise RuntimeError(f"ros2 bag record exited immediately:\n{output}")

        return self._bag_path

    def stop(self, timeout: float = 10.0) -> None:
        """Stop the recording (sends SIGINT, waits for clean exit)."""
        if self._proc is None:
            return

        try:
            if self._is_local:
                # Send SIGINT to the process group for clean ros2 bag shutdown
                os.killpg(os.getpgid(self._proc.pid), signal.SIGINT)
            else:
                self._proc.terminate()
        except OSError:
            pass

        try:
            self._proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            self._proc.kill()
            self._proc.wait(timeout=5.0)

        self._proc = None

    def pull(self, local_dir: str | Path) -> Path:
        """Copy the recorded bag to a local directory.

        For local recordings, this is a no-op that returns the existing path.
        For SSH recordings, copies via scp.

        Returns the local path to the bag directory.
        """
        if self._bag_path is None:
            raise RuntimeError("No recording to pull (call start() first)")

        if self._is_local:
            return Path(self._bag_path)

        local_dir = Path(local_dir)
        local_dir.mkdir(parents=True, exist_ok=True)

        result = subprocess.run(
            ["scp", "-r", f"{self._ssh}:{self._bag_path}", str(local_dir)], capture_output=True, text=True, timeout=120
        )
        if result.returncode != 0:
            raise RuntimeError(f"scp failed: {result.stderr}")

        bag_name = self._bag_path.rsplit("/", 1)[-1]
        return local_dir / bag_name

    @property
    def is_recording(self) -> bool:
        """True if a recording subprocess is currently running."""
        return self._proc is not None and self._proc.poll() is None

    @property
    def bag_path(self) -> str | None:
        """Path to the bag (None if not started)."""
        return self._bag_path
