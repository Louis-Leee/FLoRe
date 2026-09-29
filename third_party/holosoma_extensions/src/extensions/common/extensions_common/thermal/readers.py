"""Robot-specific temperature acquisition backends.

Each reader returns a dict mapping joint names to temperature in Celsius.
Use ``create_thermal_reader(robot_type, interface)`` as the factory entry point.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class ThermalReader(ABC):
    """Robot-agnostic interface for reading joint temperatures."""

    @abstractmethod
    def get_dof_names(self) -> tuple[str, ...]:
        """Return the ordered joint names for this robot."""

    @abstractmethod
    def read_temperatures(self) -> dict[str, float]:
        """Return ``{joint_name: temp_celsius}`` for all joints."""

    @abstractmethod
    def read_torques(self) -> dict[str, float] | None:
        """Return ``{joint_name: torque_nm}`` or None if unavailable.

        Used by cooldown tracking to verify passive/zero-torque state.
        """

    def read_state(self) -> tuple[dict[str, float], dict[str, float] | None]:
        """Return (temperatures, torques) from a single hardware read.

        Default implementation calls read_temperatures() and read_torques()
        separately. Subclasses should override for efficiency when both
        come from the same underlying call.
        """
        temps = self.read_temperatures()
        torques = self.read_torques()
        return temps, torques


class UnitreeThermalReader(ThermalReader):
    """G1 (and future T1) via Unitree raw C++ binding.

    Reads ``motor.temperature`` and ``motor.tau_est`` from the low-level state.
    """

    def __init__(self, interface, dof_names: tuple[str, ...]):
        self._raw = interface.unitree_interface
        self._dof_names = dof_names
        self._n = len(dof_names)

    def get_dof_names(self) -> tuple[str, ...]:
        return self._dof_names

    def read_temperatures(self) -> dict[str, float]:
        m = self._raw.read_low_state().motor
        return {name: float(m.temperature[i]) for i, name in enumerate(self._dof_names)}

    def read_torques(self) -> dict[str, float] | None:
        m = self._raw.read_low_state().motor
        return {name: float(m.tau_est[i]) for i, name in enumerate(self._dof_names)}

    def read_state(self) -> tuple[dict[str, float], dict[str, float] | None]:
        m = self._raw.read_low_state().motor
        temps = {name: float(m.temperature[i]) for i, name in enumerate(self._dof_names)}
        torques = {name: float(m.tau_est[i]) for i, name in enumerate(self._dof_names)}
        return temps, torques


class XdofThermalReader(ThermalReader):
    """X1 via XdofInterface (ROS2 ``/motors/state_temperature``)."""

    def __init__(self, interface):
        self._iface = interface
        self._dof_names = tuple(interface.dof_names)

    def get_dof_names(self) -> tuple[str, ...]:
        return self._dof_names

    def read_temperatures(self) -> dict[str, float]:
        temps = self._iface.get_temperatures()
        return {name: float(temps[i]) for i, name in enumerate(self._dof_names)}

    def read_torques(self) -> dict[str, float] | None:
        torques = self._iface.get_torques()
        return {name: float(torques[i]) for i, name in enumerate(self._dof_names)}

    def read_state(self) -> tuple[dict[str, float], dict[str, float] | None]:
        temps = self._iface.get_temperatures()
        torques = self._iface.get_torques()
        return (
            {name: float(temps[i]) for i, name in enumerate(self._dof_names)},
            {name: float(torques[i]) for i, name in enumerate(self._dof_names)},
        )


class BxiThermalReader(ThermalReader):
    """BXI ELF3 via BxiElf3Interface (ROS2 ``hardware/actuator_states``)."""

    def __init__(self, interface, dof_names: tuple[str, ...]):
        self._iface = interface
        self._dof_names = dof_names

    def get_dof_names(self) -> tuple[str, ...]:
        return self._dof_names

    def read_temperatures(self) -> dict[str, float]:
        temps = self._iface.get_temperatures()
        return {name: float(temps[i]) for i, name in enumerate(self._dof_names)}

    def read_torques(self) -> dict[str, float] | None:
        torques = self._iface.get_torques()
        return {name: float(torques[i]) for i, name in enumerate(self._dof_names)}

    def read_state(self) -> tuple[dict[str, float], dict[str, float] | None]:
        temps = self._iface.get_temperatures()
        torques = self._iface.get_torques()
        return (
            {name: float(temps[i]) for i, name in enumerate(self._dof_names)},
            {name: float(torques[i]) for i, name in enumerate(self._dof_names)},
        )


class BoosterThermalReader(ThermalReader):
    """T1 via Booster SDK — stub for future implementation."""

    def __init__(self, interface, dof_names: tuple[str, ...]):
        self._dof_names = dof_names
        self._interface = interface

    def get_dof_names(self) -> tuple[str, ...]:
        return self._dof_names

    def read_temperatures(self) -> dict[str, float]:
        raise NotImplementedError("Booster SDK temperature reading not yet implemented")

    def read_torques(self) -> dict[str, float] | None:
        raise NotImplementedError("Booster SDK torque reading not yet implemented")


def create_thermal_reader(robot_type: str, interface, dof_names: tuple[str, ...] | None = None) -> ThermalReader:
    """Factory: create the correct ThermalReader for a robot type.

    Parameters
    ----------
    robot_type : str
        One of ``"g1"``, ``"t1"``, ``"x1"`` (case-insensitive).
    interface
        The holosoma_inference interface object for this robot.
    dof_names : tuple[str, ...] | None
        Joint names. Required for ``"g1"`` and ``"t1"``; inferred from
        interface for ``"x1"``.
    """
    rt = robot_type.lower()
    if rt == "g1":
        if dof_names is None:
            raise ValueError("dof_names required for G1 UnitreeThermalReader")
        return UnitreeThermalReader(interface, dof_names)
    elif rt == "t1":
        if dof_names is None:
            raise ValueError("dof_names required for T1 BoosterThermalReader")
        return BoosterThermalReader(interface, dof_names)
    elif rt == "x1":
        return XdofThermalReader(interface)
    elif rt == "bxi":
        if dof_names is None:
            raise ValueError("dof_names required for BXI BxiThermalReader")
        return BxiThermalReader(interface, dof_names)
    else:
        raise ValueError(f"Unknown robot type: {robot_type!r}. Expected one of: g1, t1, x1, bxi")
