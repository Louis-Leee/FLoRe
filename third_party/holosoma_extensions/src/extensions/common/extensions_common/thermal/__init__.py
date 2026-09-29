"""Shared heat dissipation measurement module for humanoid robot testing.

Provides robot-agnostic thermal monitoring, cooldown tracking, equilibrium
detection, and reporting across G1 (Unitree), T1 (Booster), and X1 (XDOF/ROS2).
"""

from extensions_common.thermal.readers import (
    ThermalReader,
    UnitreeThermalReader,
    XdofThermalReader,
    create_thermal_reader,
)
from extensions_common.thermal.monitor import ThermalConfig, ThermalMonitor
from extensions_common.thermal.report import ThermalReport
from extensions_common.thermal.ir_capture import IRCameraCapture
from extensions_common.thermal.rgb_capture import RGBVideoRecorder
from extensions_common.thermal.telemetry_recorder import TelemetryRecorder

__all__ = [
    "ThermalReader",
    "UnitreeThermalReader",
    "XdofThermalReader",
    "create_thermal_reader",
    "ThermalConfig",
    "ThermalMonitor",
    "ThermalReport",
    "IRCameraCapture",
    "RGBVideoRecorder",
    "TelemetryRecorder",
]
