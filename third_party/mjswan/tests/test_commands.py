"""Tests for mjswan.command.

Layer: L1 (pure Python, no MuJoCo/ONNX required).
"""

import mjswan
from mjswan.command import (
    ButtonConfig,
    CommandTermConfig,
    CommandTermSpec,
    SliderConfig,
    _custom_registry,
    register_command_term,
    ui_command,
    velocity_command,
)


class TestSliderConfig:
    def test_min_max_derived_from_range(self):
        s = SliderConfig(name="x", label="X", range=(-2.0, 3.0))
        assert s.min == -2.0
        assert s.max == 3.0

    def test_to_dict_includes_all_fields(self):
        s = SliderConfig(
            name="lin_vel_x",
            label="Forward Velocity",
            range=(-1.0, 1.0),
            default=0.5,
            step=0.05,
        )
        d = s.to_dict()
        assert d["type"] == "slider"
        assert d["name"] == "lin_vel_x"
        assert d["label"] == "Forward Velocity"
        assert d["min"] == -1.0
        assert d["max"] == 1.0
        assert d["default"] == 0.5
        assert d["step"] == 0.05

    def test_slider_is_alias_for_slider_config(self):
        assert mjswan.Slider is SliderConfig


class TestButtonConfig:
    def test_to_dict_includes_name_and_label(self):
        b = ButtonConfig(name="reset", label="Reset Simulation")
        assert b.to_dict() == {
            "type": "button",
            "name": "reset",
            "label": "Reset Simulation",
        }

    def test_button_is_alias_for_button_config(self):
        assert mjswan.Button is ButtonConfig


class TestUiCommand:
    def test_ui_command_serializes_as_ui_term(self):
        command = ui_command(
            [
                SliderConfig(name="x", label="X", range=(-1.0, 1.0)),
                ButtonConfig(name="reset", label="Reset"),
            ]
        )
        assert command.to_dict() == {
            "name": "UiCommand",
            "ui": {
                "inputs": [
                    {
                        "type": "slider",
                        "name": "x",
                        "label": "X",
                        "min": -1.0,
                        "max": 1.0,
                        "step": 0.01,
                        "default": 0.0,
                    },
                    {
                        "type": "button",
                        "name": "reset",
                        "label": "Reset",
                    },
                ]
            },
        }


class TestVelocityCommand:
    def test_velocity_command_is_ui_command(self):
        cmd = velocity_command()
        assert isinstance(cmd, CommandTermConfig)
        assert cmd.term_name == "UiCommand"

    def test_velocity_command_has_exactly_three_sliders(self):
        cmd = velocity_command()
        inputs = cmd.ui.inputs if cmd.ui is not None else []
        assert len(inputs) == 3
        assert all(isinstance(inp, SliderConfig) for inp in inputs)

    def test_slider_names_are_canonical(self):
        cmd = velocity_command()
        inputs = cmd.ui.inputs if cmd.ui is not None else []
        assert [inp.name for inp in inputs] == ["lin_vel_x", "lin_vel_y", "ang_vel_z"]

    def test_velocity_command_is_accessible_from_mjswan(self):
        assert mjswan.velocity_command is velocity_command


class TestCommandRegistry:
    def test_register_command_term_is_accessible_from_mjswan(self):
        assert mjswan.register_command_term is register_command_term

    def test_custom_term_spec_can_be_registered(self):
        register_command_term(
            "DummyCommandCfg",
            CommandTermSpec(
                ts_name="DummyCommand",
                serializer=lambda cfg: {"value": cfg.value},
            ),
        )

        class DummyCfg:
            value = 3

        spec = _custom_registry["DummyCommandCfg"]
        assert spec.ts_name == "DummyCommand"
        assert spec.serializer(DummyCfg()) == {"value": 3}
