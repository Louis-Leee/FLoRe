"""mjswan: Browser-based MuJoCo Playground

Interactive MuJoCo simulations with ONNX policies running entirely in the browser.
"""

__version__ = "0.6.0"

from .app import mjswanApp
from .builder import Builder
from .command import (
    Button,
    ButtonConfig,
    Checkbox,
    CheckboxConfig,
    CommandInput,
    CommandTermConfig,
    CommandTermSpec,
    CommandUiConfig,
    Slider,
    SliderConfig,
    register_command_term,
    ui_command,
    velocity_command,
)
from .envs.mdp.actions import (
    ActionTermCfg,
    JointEffortActionCfg,
    JointPositionActionCfg,
)
from .envs.mdp.events import EventFunc, register_event_func
from .envs.mdp.observations import ObsFunc, register_obs_func
from .envs.mdp.terminations import TermFunc, register_termination_func
from .managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from .managers.termination_manager import TerminationTermCfg
from .motion import MotionConfig, MotionHandle
from .policy import PolicyConfig, PolicyHandle
from .project import ProjectConfig, ProjectHandle
from .scene import SceneConfig, SceneHandle
from .splat import SplatConfig, SplatHandle
from .viewer_config import ViewerConfig

__all__ = [
    # Builder and App
    "Builder",
    "mjswanApp",
    # Handles
    "ProjectHandle",
    "SceneHandle",
    "SplatHandle",
    "PolicyHandle",
    "MotionHandle",
    # Configs
    "ProjectConfig",
    "SceneConfig",
    "ViewerConfig",
    "SplatConfig",
    "PolicyConfig",
    "MotionConfig",
    # Custom observation registry
    "ObsFunc",
    "register_obs_func",
    "EventFunc",
    "register_event_func",
    "TermFunc",
    "register_termination_func",
    # MDP config (mjlab-compatible)
    "ObservationGroupCfg",
    "ObservationTermCfg",
    "ActionTermCfg",
    "JointPositionActionCfg",
    "JointEffortActionCfg",
    "TerminationTermCfg",
    # Commands
    "Slider",
    "SliderConfig",
    "Button",
    "ButtonConfig",
    "Checkbox",
    "CheckboxConfig",
    "CommandInput",
    "CommandTermConfig",
    "CommandTermSpec",
    "CommandUiConfig",
    "register_command_term",
    "ui_command",
    "velocity_command",
]
