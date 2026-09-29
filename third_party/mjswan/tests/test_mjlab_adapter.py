"""Tests for mjswan.adapters.mjlab_adapter — mjlab type conversion.

Layer: L1 (pure Python, no MuJoCo/ONNX/mjlab required).

These tests simulate mjlab types by creating lightweight mock classes
with the same attributes that the adapter inspects, placed in a fake
``mjlab.*`` module path.
"""

from __future__ import annotations

from typing import Any

import pytest

from mjswan.adapters.mjlab_adapter import (
    adapt_actions,
    adapt_commands,
    adapt_observations,
    adapt_terminations,
)
from mjswan.envs.mdp.observations import ObsFunc
from mjswan.envs.mdp.terminations import TermFunc
from mjswan.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjswan.managers.termination_manager import TerminationTermCfg

# ---------------------------------------------------------------------------
# Fake mjlab types — classes whose __module__ starts with "mjlab"
# ---------------------------------------------------------------------------


def _make_mjlab_class(class_name: str, **defaults: Any) -> type:
    """Create a simple dataclass-like class that appears to come from mjlab."""

    class Cls:
        def __init__(self, **kwargs: Any):
            for k, v in {**defaults, **kwargs}.items():
                setattr(self, k, v)

    Cls.__name__ = class_name
    Cls.__qualname__ = class_name
    Cls.__module__ = "mjlab.fake"
    return Cls


# Fake mjlab observation functions (callables with __name__ and __module__)
def _make_mjlab_obs_func(name: str):
    def fn():
        pass

    fn.__name__ = name
    fn.__module__ = "mjlab.envs.mdp.observations"
    return fn


def _make_mjlab_term_func(name: str):
    def fn():
        pass

    fn.__name__ = name
    fn.__module__ = "mjlab.envs.mdp.terminations"
    return fn


# Fake mjlab config classes
FakeMjlabObsTermCfg = _make_mjlab_class(
    "ObservationTermCfg",
    func=None,
    params={},
    scale=None,
    clip=None,
    history_length=0,
    noise=None,
)

FakeMjlabObsGroupCfg = _make_mjlab_class(
    "ObservationGroupCfg",
    terms={},
    concatenate_terms=True,
    enable_corruption=False,
    history_length=None,
)

FakeMjlabTermTermCfg = _make_mjlab_class(
    "TerminationTermCfg",
    func=None,
    params={},
    time_out=False,
)

FakeMjlabJointPositionActionCfg = _make_mjlab_class(
    "JointPositionActionCfg",
    entity_name="robot",
    clip=None,
    actuator_names=(".*",),
    scale=1.0,
    offset=0.0,
    use_default_offset=True,
    stiffness=None,
    damping=None,
)

FakeMjlabJointEffortActionCfg = _make_mjlab_class(
    "JointEffortActionCfg",
    entity_name="robot",
    clip=None,
    actuator_names=(".*",),
    scale=1.0,
    offset=0.0,
    stiffness=None,
    damping=None,
)

FakeMjlabSceneEntityCfg = _make_mjlab_class(
    "SceneEntityCfg",
    **{
        "name": "robot",
        "joint_names": None,
        "site_names": None,
    },
)

FakeMjlabMotionCommandCfg = _make_mjlab_class(
    "MotionCommandCfg",
    anchor_body_name="torso_link",
    body_names=("pelvis", "torso_link"),
    entity_name="robot",
)


# ===================================================================
# Tests: Observations
# ===================================================================


class TestAdaptObservations:
    def test_none_passthrough(self):
        assert adapt_observations(None) is None

    def test_mjswan_types_unchanged(self):
        obs_func = ObsFunc("BaseLinearVelocity")
        group = ObservationGroupCfg(terms={"vel": ObservationTermCfg(func=obs_func)})
        result = adapt_observations({"policy": group})
        assert result is not None
        assert result["policy"] is group

    def test_mjlab_obs_term_converted(self):
        mjlab_func = _make_mjlab_obs_func("base_lin_vel")
        mjlab_term = FakeMjlabObsTermCfg(func=mjlab_func, params={"world_frame": True})
        mjlab_group = FakeMjlabObsGroupCfg(
            terms={"base_vel": mjlab_term},
            enable_corruption=True,
        )

        result = adapt_observations({"policy": mjlab_group})
        assert result is not None
        group = result["policy"]
        assert isinstance(group, ObservationGroupCfg)
        assert "base_vel" in group.terms
        term = group.terms["base_vel"]
        assert isinstance(term, ObservationTermCfg)
        assert term.func.ts_name == "BaseLinearVelocity"
        assert term.params == {"world_frame": True}

    def test_mjlab_obs_scale_and_history(self):
        mjlab_func = _make_mjlab_obs_func("joint_pos_rel")
        mjlab_term = FakeMjlabObsTermCfg(func=mjlab_func, scale=0.5, history_length=3)
        mjlab_group = FakeMjlabObsGroupCfg(terms={"jp": mjlab_term})

        result = adapt_observations({"policy": mjlab_group})
        assert result is not None
        term = result["policy"].terms["jp"]
        assert term.scale == 0.5
        assert term.history_length == 3

    def test_mjlab_asset_cfg_joint_scope_preserved(self):
        mjlab_func = _make_mjlab_obs_func("joint_pos_rel")
        asset_cfg = FakeMjlabSceneEntityCfg(
            name="robot", joint_names=("joint1", "joint2")
        )
        mjlab_term = FakeMjlabObsTermCfg(
            func=mjlab_func, params={"asset_cfg": asset_cfg}
        )
        mjlab_group = FakeMjlabObsGroupCfg(terms={"jp": mjlab_term})

        result = adapt_observations({"policy": mjlab_group})
        assert result is not None
        term = result["policy"].terms["jp"]
        assert term.params["entity_name"] == "robot"
        assert term.params["joint_names"] == ["joint1", "joint2"]

    def test_unknown_mjlab_obs_func_raises(self):
        mjlab_func = _make_mjlab_obs_func("nonexistent_function")
        mjlab_term = FakeMjlabObsTermCfg(func=mjlab_func)
        mjlab_group = FakeMjlabObsGroupCfg(terms={"x": mjlab_term})

        with pytest.raises(ValueError, match="No mjswan mapping"):
            adapt_observations({"policy": mjlab_group})

    def test_multiple_groups(self):
        f1 = _make_mjlab_obs_func("base_ang_vel")
        f2 = _make_mjlab_obs_func("projected_gravity")
        g1 = FakeMjlabObsGroupCfg(terms={"ang": FakeMjlabObsTermCfg(func=f1)})
        g2 = FakeMjlabObsGroupCfg(terms={"grav": FakeMjlabObsTermCfg(func=f2)})

        result = adapt_observations({"policy": g1, "critic": g2})
        assert result is not None
        assert result["policy"].terms["ang"].func.ts_name == "BaseAngularVelocity"
        assert result["critic"].terms["grav"].func.ts_name == "ProjectedGravityB"

    def test_tracking_observation_functions_are_mapped(self):
        motion_anchor = _make_mjlab_obs_func("motion_anchor_pos_b")
        body_pos = _make_mjlab_obs_func("robot_body_pos_b")
        result = adapt_observations(
            {
                "policy": FakeMjlabObsGroupCfg(
                    terms={
                        "anchor": FakeMjlabObsTermCfg(func=motion_anchor),
                        "body": FakeMjlabObsTermCfg(func=body_pos),
                    }
                )
            }
        )
        assert result is not None
        assert result["policy"].terms["anchor"].func.ts_name == "MotionAnchorPosB"
        assert result["policy"].terms["body"].func.ts_name == "RobotBodyPosB"


# ===================================================================
# Tests: Terminations
# ===================================================================


class TestAdaptTerminations:
    def test_none_passthrough(self):
        assert adapt_terminations(None) is None

    def test_mjswan_types_unchanged(self):
        term_func = TermFunc("TimeOut")
        cfg = TerminationTermCfg(func=term_func, time_out=True)
        result = adapt_terminations({"time_out": cfg})
        assert result is not None
        assert result["time_out"] is cfg

    def test_mjlab_term_converted(self):
        mjlab_func = _make_mjlab_term_func("bad_orientation")
        mjlab_cfg = FakeMjlabTermTermCfg(
            func=mjlab_func,
            params={"limit_angle": 1.0},
            time_out=False,
        )

        result = adapt_terminations({"fallen": mjlab_cfg})
        assert result is not None
        term = result["fallen"]
        assert isinstance(term, TerminationTermCfg)
        assert term.func.ts_name == "BadOrientation"
        assert term.params == {"limit_angle": 1.0}
        assert term.time_out is False

    def test_mjlab_time_out_flag(self):
        mjlab_func = _make_mjlab_term_func("time_out")
        mjlab_cfg = FakeMjlabTermTermCfg(func=mjlab_func, time_out=True)

        result = adapt_terminations({"timeout": mjlab_cfg})
        assert result is not None
        assert result["timeout"].time_out is True
        assert result["timeout"].func.ts_name == "TimeOut"

    def test_mjlab_term_strips_asset_cfg_from_params(self):
        mjlab_func = _make_mjlab_term_func("bad_orientation")
        asset_cfg = FakeMjlabSceneEntityCfg(name="robot", body_names=("torso_link",))
        mjlab_cfg = FakeMjlabTermTermCfg(
            func=mjlab_func,
            params={"limit_angle": 1.0, "asset_cfg": asset_cfg},
            time_out=False,
        )

        result = adapt_terminations({"fallen": mjlab_cfg})
        assert result is not None
        term = result["fallen"]
        assert term.params == {
            "limit_angle": 1.0,
            "entity_name": "robot",
            "body_names": ["torso_link"],
        }

    def test_unknown_mjlab_term_func_raises(self):
        mjlab_func = _make_mjlab_term_func("nonexistent_term")
        mjlab_cfg = FakeMjlabTermTermCfg(func=mjlab_func)

        with pytest.raises(ValueError, match="No mjswan mapping"):
            adapt_terminations({"x": mjlab_cfg})


# ===================================================================
# Tests: Actions
# ===================================================================


class TestAdaptActions:
    def test_none_passthrough(self):
        assert adapt_actions(None) is None


class TestAdaptCommands:
    def test_none_passthrough(self):
        assert adapt_commands(None) is None

    def test_motion_command_cfg_converts_to_tracking_command(self):
        result = adapt_commands({"motion": FakeMjlabMotionCommandCfg()})
        assert result is not None
        command = result["motion"]
        assert command.term_name == "TrackingCommand"
        assert command.params["anchor_body_name"] == "torso_link"
        assert command.params["body_names"] == ["pelvis", "torso_link"]

    def test_mjswan_types_unchanged(self):
        from mjswan.envs.mdp.actions import JointPositionActionCfg

        cfg = JointPositionActionCfg(scale=0.5)
        result = adapt_actions({"joint_pos": cfg})
        assert result is not None
        assert result["joint_pos"] is cfg

    def test_mjlab_joint_position_converted(self):
        mjlab_cfg = FakeMjlabJointPositionActionCfg(
            scale=0.25,
            offset=0.1,
            use_default_offset=True,
            stiffness=40.0,
            damping=2.5,
        )

        result = adapt_actions({"jp": mjlab_cfg})
        assert result is not None
        from mjswan.envs.mdp.actions import JointPositionActionCfg

        action = result["jp"]
        assert isinstance(action, JointPositionActionCfg)
        assert action.scale == 0.25
        assert action.offset == 0.1
        assert action.use_default_offset is True
        assert action.stiffness == 40.0
        assert action.damping == 2.5

    def test_mjlab_joint_effort_converted(self):
        mjlab_cfg = FakeMjlabJointEffortActionCfg(scale=2.0)

        result = adapt_actions({"torque": mjlab_cfg})
        assert result is not None
        from mjswan.envs.mdp.actions import JointEffortActionCfg

        action = result["torque"]
        assert isinstance(action, JointEffortActionCfg)
        assert action.scale == 2.0

    def test_mjlab_unknown_action_warns(self):
        FakeUnknown = _make_mjlab_class(
            "SomeWeirdActionCfg",
            entity_name="robot",
            clip=None,
            actuator_names=(".*",),
            scale=1.0,
            offset=0.0,
        )
        mjlab_cfg = FakeUnknown()

        with pytest.warns(RuntimeWarning, match="no mjswan equivalent"):
            result = adapt_actions({"weird": mjlab_cfg})

        assert result is not None
        assert "weird" not in result

    def test_mixed_mjswan_and_mjlab(self):
        """Both mjswan-native and mjlab types in the same dict."""
        from mjswan.envs.mdp.actions import JointPositionActionCfg

        mjswan_cfg = JointPositionActionCfg(scale=0.5)
        mjlab_cfg = FakeMjlabJointEffortActionCfg(scale=3.0)

        result = adapt_actions({"jp": mjswan_cfg, "torque": mjlab_cfg})
        assert result is not None
        assert result["jp"] is mjswan_cfg
        from mjswan.envs.mdp.actions import JointEffortActionCfg

        assert isinstance(result["torque"], JointEffortActionCfg)


# ===================================================================
# Tests: End-to-end serialization after adaptation
# ===================================================================


class TestAdaptedSerialization:
    """Ensure adapted objects serialize correctly via to_dict() / to_list()."""

    def test_adapted_obs_serializes(self):
        mjlab_func = _make_mjlab_obs_func("last_action")
        mjlab_term = FakeMjlabObsTermCfg(func=mjlab_func)
        mjlab_group = FakeMjlabObsGroupCfg(terms={"la": mjlab_term})

        result = adapt_observations({"policy": mjlab_group})
        assert result is not None
        entries = result["policy"].to_list()
        assert len(entries) == 1
        assert entries[0]["name"] == "PrevActions"
        assert entries[0]["history_steps"] == 1  # from ObsFunc defaults

    def test_adapted_term_serializes(self):
        mjlab_func = _make_mjlab_term_func("root_height_below_minimum")
        mjlab_cfg = FakeMjlabTermTermCfg(
            func=mjlab_func,
            params={"minimum_height": 0.2},
        )

        result = adapt_terminations({"fallen": mjlab_cfg})
        assert result is not None
        d = result["fallen"].to_dict()
        assert d["name"] == "RootHeightBelowMinimum"
        assert d["params"]["minimum_height"] == 0.2

    def test_adapted_action_serializes(self):
        mjlab_cfg = FakeMjlabJointPositionActionCfg(
            scale={"hip": 0.5, "knee": 0.3},
            stiffness=40.0,
            damping=2.5,
        )

        result = adapt_actions({"jp": mjlab_cfg})
        assert result is not None
        d = result["jp"].to_dict()
        assert d["type"] == "joint_position"
        assert d["scale"] == {"hip": 0.5, "knee": 0.3}
        assert d["stiffness"] == 40.0
        assert d["damping"] == 2.5
