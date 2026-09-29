"""Register every WBT obs func + the metrics tap + WbtTrackingCommand with mjswan.

Called from :func:`main.setup_builder` once per Python process, BEFORE
``add_policy(...)``. Registration is process-global state; ``setup_builder``
calls this exactly once per process. Duplicate ``ts_name``\ s are rejected by
mjswan's node build (the TS concat / ``tsc``), not necessarily by
``register()`` at runtime — so new obs classes MUST use unique ``ts_name``\ s.

The registry is a superset: it contains every obs term that ANY supported
preset uses. The active preset (see :mod:`presets`) decides which subset of
these gets wired into the policy_config.components list.
"""

from __future__ import annotations

from pathlib import Path

import mjswan

_HERE = Path(__file__).resolve().parent
_TS_DIR = _HERE / "template"

_COMMAND_TS = _TS_DIR / "WbtTrackingCommand.ts"
_OBS_TS = _TS_DIR / "WbtObservations.ts"
_METRICS_TS = _TS_DIR / "WbtMetrics.ts"

# Every TS obs class we ship. The (key, ts_name, ts_src) tuples here are
# matched against the active preset's obs_keys list in main.py to build the
# policy_config.components list. Extra registry entries beyond the active
# preset are harmless: mjswan just ignores them when assembling the bundle.
_OBS_FUNC_NAMES: list[tuple[str, str, Path]] = [
    ("command", "WbtCommand", _OBS_TS),
    ("command_lower_body", "WbtCommandLowerBody", _OBS_TS),
    ("vr_2point_pos", "WbtVr2PointPos", _OBS_TS),
    ("vr_2point_orn", "WbtVr2PointOrn", _OBS_TS),
    ("vr_3point_pos", "WbtVr3PointPos", _OBS_TS),
    # TCP-unified 3pt hand point (exp:g1-wbt-3pt-hybrid-unified[-anneal], user
    # 2026-08-03): identical 9D layout, but the two HAND points are baked at the
    # 0.242 m force TCP instead of the 0.18 m VR lever (bake table vr3_pos_tcp_local).
    ("vr_3point_pos_tcp", "WbtVr3PointPosTcp", _OBS_TS),
    ("vr_3point_orn", "WbtVr3PointOrn", _OBS_TS),
    # 6D rot rep of the 3pt orientation (18D) for exp:g1-wbt-3pt-hybrid-global-vel.
    ("vr_3point_orn_6d", "WbtVr3PointOrn6d", _OBS_TS),
    # 5-POINT tracking (exp:g1-wbt-5pt-hybrid-unified-anneal, user 2026-08-27): the
    # 3pt-TCP set + both FEET at the bare *_ankle_roll_link origin. 15D pos / 20D quat,
    # read from the bake tables vr5_pos_tcp_local / vr5_orn_local.
    ("vr_5point_pos_tcp", "WbtVr5PointPosTcp", _OBS_TS),
    ("vr_5point_orn", "WbtVr5PointOrn", _OBS_TS),
    ("motion_ref_pos_b", "WbtMotionRefPosB", _OBS_TS),
    ("motion_ref_ori_b", "WbtMotionRefOriB", _OBS_TS),
    ("motion_ref_vel_b", "WbtMotionRefVelB", _OBS_TS),
    ("projected_gravity", "WbtProjectedGravity", _OBS_TS),
    ("base_ang_vel", "WbtBaseAngVel", _OBS_TS),
    ("base_lin_vel", "WbtBaseLinVel", _OBS_TS),
    ("joint_pos", "WbtJointPos", _OBS_TS),
    ("joint_vel", "WbtJointVel", _OBS_TS),
    ("actions", "WbtPrevActions", _OBS_TS),
    ("force_cmd_local", "WbtForceCmdLocal", _OBS_TS),
    # UniFP F^ext-in-actor term (exp:g1-wbt-2pt-force-ext). Reads the SAME
    # wbt_force_ext_local__* sliders as the orange F^ext arrow, at scale 0.04.
    ("force_ext_local", "WbtForceExtLocal", _OBS_TS),
    # Hybrid causal-contact actor terms (exp:g1-wbt-2pt-hybrid, servo + Layer-A).
    # The mjswan-fork runtime publishes world-frame F^meas / I / Δx to the extras
    # bus each substep; these obs classes rotate world->pelvis + scale.
    ("force_meas_local", "WbtForceMeasLocal", _OBS_TS),
    ("integrator_state", "WbtIntegratorState", _OBS_TS),
    ("x_tgt_shift", "WbtXTgtShift", _OBS_TS),
    ("contact_phase", "WbtContactPhase", _OBS_TS),
    # Live VIVE-tracker 2pt terms (parallel teleop path; same TS file). NEW
    # ts_names so they never collide with the replay WbtVr2PointPos/Orn —
    # mjswan's node build (not register()) would fail on a duplicate ts_name.
    ("vr_2point_pos_live", "WbtVr2PointPosLive", _OBS_TS),
    ("vr_2point_orn_live", "WbtVr2PointOrnLive", _OBS_TS),
    # Live VIVE-tracker 3pt terms (the 3pt hybrid UNIFIED anneal teleop path).
    # Again NEW ts_names: the 3pt orn is 3x xyzw quaternion where the 2pt live orn
    # is 2x rot6d, so sharing a class would silently mis-read one rep as the other
    # at the same 12D width. `contact_phase` is NOT repeated here — the replay
    # entry above already registers WbtContactPhase and the live preset reuses it
    # (a duplicate ts_name fails mjswan's node build, not register()).
    ("vr_3point_pos_tcp_live", "WbtVr3PointPosTcpLive", _OBS_TS),
    ("vr_3point_orn_live", "WbtVr3PointOrnLive", _OBS_TS),
    # MOTION-COMPLETION live terms (bridge --mode 5pt / dense). Same three trackers,
    # routed through Motion_Completion so the extra points come out of its 29-DoF FK.
    # `command_live` replaces WbtCommand (which reads the baked clip) and remaps
    # the wire's 29 joints onto the policy's order BY NAME.
    ("vr_5point_pos_tcp_live", "WbtVr5PointPosTcpLive", _OBS_TS),
    ("vr_5point_orn_live", "WbtVr5PointOrnLive", _OBS_TS),
    ("command_live", "WbtCommandLive", _OBS_TS),
    ("_metrics_tap", "WbtMetricsTap", _METRICS_TS),
]


def register() -> dict[str, mjswan.ObsFunc]:
    """Register all obs funcs + the WbtTrackingCommand. Returns key->ObsFunc map."""
    mjswan.register_command_term(
        "WbtTrackingCommandCfg",
        mjswan.CommandTermSpec(
            ts_name="WbtTrackingCommand",
            serializer=lambda cfg: dict(cfg.params),  # passthrough
            ts_src=str(_COMMAND_TS),
        ),
    )

    funcs: dict[str, mjswan.ObsFunc] = {}
    for key, ts_name, ts_src in _OBS_FUNC_NAMES:
        f = mjswan.ObsFunc(ts_name=ts_name, ts_src=str(ts_src))
        # Use a module-prefixed mjlab_name so we don't collide with any built-in
        # registry entry (e.g. mjswan ships a "joint_pos" mjlab name).
        mjswan.register_obs_func(f"wbt_realtime_eval.{key}", f)
        funcs[key] = f
    return funcs
