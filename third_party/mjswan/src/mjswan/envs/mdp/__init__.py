"""MDP components for mjswan.

Mirrors ``mjlab.envs.mdp``.  Re-exports modules so that the following
mjlab import patterns translate directly::

    # mjlab
    from mjlab.envs.mdp import observations as obs_fns
    from mjlab.envs.mdp import terminations as term_fns
    from mjlab.envs.mdp.actions import JointPositionActionCfg

    # mjswan (identical)
    from mjswan.envs.mdp import observations as obs_fns
    from mjswan.envs.mdp import terminations as term_fns
    from mjswan.envs.mdp.actions import JointPositionActionCfg
"""

from . import actions, observations, terminations

__all__ = ["actions", "observations", "terminations"]
