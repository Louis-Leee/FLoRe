#!/usr/bin/env python3
"""Sim-to-sim launcher for G1 WBT inference.

Wraps the core ``holosoma.run_sim`` with a G1-specific robot preset whose
spawn height places the feet on (or very near) the ground, and a gantry rest
length that balances gravity at that height so the robot doesn't float or
collapse on spawn.

Usage:
    python -m wbt_wrappers_inference.run_sim              # MuJoCo default
    python -m wbt_wrappers_inference.run_sim simulator:mujoco
"""

from __future__ import annotations

import dataclasses
import sys
import traceback

import tyro
from loguru import logger

import holosoma.config_values.robot as robot
import holosoma.config_values.run_sim as core_run_sim
from holosoma.config_types.run_sim import RunSimConfig
from holosoma.run_sim import run_simulation
from holosoma.utils.eval_utils import init_eval_logging
from holosoma.utils.tyro_utils import TYRO_CONIFG

# --- G1 WBT spawn tuning ------------------------------------------------
_G1_WBT_SPAWN_Z = 0.68

_g1_wbt_robot = dataclasses.replace(
    robot.g1_29dof, init_state=dataclasses.replace(robot.g1_29dof.init_state, pos=[0.0, 0.0, _G1_WBT_SPAWN_Z])
)

# Gantry rest length so the elastic band force ≈ 0 at spawn height.
# anchor_z (3.0) − spawn_z (0.68) = 2.32; subtract a small margin so there
# is a gentle upward force that prevents the robot from collapsing before
# the stiff controller engages.
_GANTRY_LENGTH = 0.5


def _patch_gantry(sim_cfg):
    """Return *sim_cfg* with the gantry rest length overridden."""
    return dataclasses.replace(
        sim_cfg,
        config=dataclasses.replace(
            sim_cfg.config, virtual_gantry=dataclasses.replace(sim_cfg.config.virtual_gantry, length=_GANTRY_LENGTH)
        ),
    )


SIMULATOR_DEFAULTS = {name: _patch_gantry(cfg) for name, cfg in core_run_sim.DEFAULTS.items()}
ROBOT_DEFAULTS = {"g1_29dof": _g1_wbt_robot}


def main() -> None:
    init_eval_logging()
    logger.info("WBT Wrappers – Sim-to-Sim Runner (G1 spawn tuned)")

    config = tyro.cli(
        RunSimConfig,
        default=RunSimConfig(simulator=SIMULATOR_DEFAULTS["mujoco"], robot=_g1_wbt_robot),
        description=(
            "Run sim-to-sim with G1 WBT spawn defaults.\n\n"
            "The robot spawns with feet near the ground so you can go straight\n"
            "to stiff mode + gantry release without manual lowering."
        ),
        config=TYRO_CONIFG,
    )
    try:
        run_simulation(config)
    except Exception as e:
        logger.error(f"Error during simulation: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
