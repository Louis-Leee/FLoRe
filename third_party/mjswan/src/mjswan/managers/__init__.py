"""Manager modules for mjswan.

Mirrors the ``mjlab.managers`` package layout so that mjlab import paths
translate directly::

    # mjlab
    from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
    from mjlab.managers.termination_manager import TerminationTermCfg

    # mjswan (identical API)
    from mjswan.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
    from mjswan.managers.termination_manager import TerminationTermCfg
"""

from .observation_manager import ObservationGroupCfg, ObservationTermCfg
from .termination_manager import TerminationTermCfg

__all__ = ["ObservationGroupCfg", "ObservationTermCfg", "TerminationTermCfg"]
