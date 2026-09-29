"""Unit tests for the C1 unified single-policy presets (observe-only integrator).

Covers: integrator_observe_step pure math + servo_residual_step delegation parity;
the servo_observe_only injector path (I/Δx updated, actions NEVER touched, no
Jacobian calls); the soft contact_phase label; the unified rewards; the
ServoAuthorityCurriculum (R1 anneal); the 185D obs layout; the experiment presets;
and the inference/realtime-eval config agreement.
All MagicMock — no IsaacSim runtime (mirrors test_hybrid.py).
"""

import types
from unittest.mock import MagicMock

import torch

# ---------------------------------------------------------------------------
# Harness helpers copied VERBATIM from test_hybrid.py (do not import across test
# modules — pytest collects them independently).
# ---------------------------------------------------------------------------
_G1_29DOF_NAMES = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)


def _hybrid_env(num_envs=4, num_hands=2):
    """MagicMock env exposing the tensors the injector reads (no IsaacSim runtime)."""
    env = MagicMock()
    env.num_envs = num_envs
    env.device = torch.device("cpu")
    env.episode_length_buf = torch.zeros(num_envs, dtype=torch.long)
    q = torch.zeros(num_envs, 4)
    q[:, 0] = 1.0  # identity root quat wxyz
    bpos = torch.zeros(num_envs, 2, 3)
    bpos[:, :, 0] = 0.30  # wrist ~30 cm forward
    bquat = torch.zeros(num_envs, 2, 4)
    bquat[:, :, 0] = 1.0  # identity wrist quat wxyz
    bacc = torch.zeros(num_envs, 2, 3)
    env.simulator._robot.data = types.SimpleNamespace(
        root_quat_w=q, body_pos_w=bpos, body_quat_w=bquat, body_lin_acc_w=bacc
    )
    env.simulator.find_rigid_body_indice = lambda name: {"left_wrist_yaw_link": 0, "right_wrist_yaw_link": 1}[name]
    env.simulator.body_ids = [0, 1]
    env.simulator.dof_names = list(_G1_29DOF_NAMES)  # 29-DOF PD/action vector order
    env.simulator._robot.set_external_force_and_torque = MagicMock()

    def _apply():  # base substep hook body (no-op)
        return None

    env._apply_force_in_physics_step = _apply
    return env


def _make_injector(env, cfg):
    from wbt_training.config_values.hybrid import HybridContactInjector

    term_cfg = MagicMock()
    term_cfg.params = {"hybrid_config": cfg}
    inj = HybridContactInjector(term_cfg, env)
    inj.setup()
    # allocate alpha (ForceAlphaCurriculum would in production; here set to 1).
    env._wbt_force_alpha = torch.tensor(1.0)
    return inj


def _pin_plane(env, cfg, inj):
    """Anchor a deterministic +x plane at the reset TCP (outward normal +x); pressing
    the wrist to -x penetrates. Returns nothing; mutates env plane buffers + marks seeded."""
    inj.reset(torch.arange(env.num_envs))  # seeds plane (first real reset)
    tcp0 = env.simulator._robot.data.body_pos_w.clone()
    tcp0[..., 0] += cfg.tcp_offset_x  # reset TCP world x
    env._wbt_plane_normal[:] = torch.tensor([1.0, 0.0, 0.0])
    env._wbt_plane_point[:] = tcp0
    env._wbt_k_env[:] = 200.0  # 200 N/m: 2 cm press → 4 N, 4 cm → 8 N (below 30 N clamp)


def _servo_env():
    """Servo-enabled env: adds root_physx_view.get_jacobians + find_bodies/find_joints stubs +
    a `joint_control` action term (the real PD-torque input the servo injects into) + action_scales,
    so setup() can call _resolve_jacobian_indices() + cache the joint term, and the servo can run."""
    env = _hybrid_env()
    robot = env.simulator._robot
    _body_map = {"left_wrist_yaw_link": 3, "right_wrist_yaw_link": 9}
    robot.find_bodies = lambda name: ([_body_map[name]], [name])
    _joint_map = {nm: 100 + i for i, nm in enumerate(_G1_29DOF_NAMES)}
    robot.find_joints = lambda nm, preserve_order=True: ([_joint_map[nm]], [nm])
    joint_term = types.SimpleNamespace(_actions_after_delay=torch.zeros(env.num_envs, 29), _substep_idx=0)
    env.action_manager = MagicMock()
    env.action_manager.get_term = lambda name: {"joint_control": joint_term}[name]
    env.action_scales = torch.ones(29)  # unit scale → servo_dq maps 1:1 into action units
    return env, _body_map, _joint_map


def _servo_env_wired(cfg):
    """Servo env with a NONZERO Jacobian (joint0 → TCP +x for both hands) + resolved indices,
    ready to drive the real hook. Returns (env, injector)."""
    env, _, _ = _servo_env()
    inj = _make_injector(env, cfg)  # setup() resolves indices (servo on)
    n_dof = max(c for cols in inj._jac_arm_cols for c in cols) + 1
    n_bodies = max(inj._jac_wrist_body_ids) + 1

    def _jac():
        j = torch.zeros(4, n_bodies, 6, n_dof)
        j[:, inj._jac_wrist_body_ids[0], 0, inj._jac_arm_cols[0][0]] = 1.0
        j[:, inj._jac_wrist_body_ids[1], 0, inj._jac_arm_cols[1][0]] = 1.0
        return j

    env.simulator._robot.root_physx_view.get_jacobians = _jac
    return env, inj


# ---------------------------------------------------------------------------
# Task 1: integrator_observe_step + servo_observe_only short-circuit
# ---------------------------------------------------------------------------
def test_integrator_observe_step_matches_servo_residual_step():
    """The extracted pure function and the full servo step must produce IDENTICAL
    I/Δx for the same inputs (the refactor is behavior-preserving)."""
    from wbt_training.config_values.hybrid import HybridContactConfig, integrator_observe_step, servo_residual_step

    cfg = HybridContactConfig(servo_enabled=True)
    n, h, na = 3, 2, 7
    torch.manual_seed(0)
    f_cmd = torch.randn(n, h, 3) * 10
    f_meas = torch.randn(n, h, 3) * 10
    gate = torch.tensor([[1.0, 0.0], [1.0, 1.0], [0.0, 0.0]])
    integ0 = torch.randn(n, h, 3)
    j = torch.randn(n, h, 6, na)
    dt = 0.005
    new_i_obs, dx_obs, tw_obs = integrator_observe_step(f_cmd, f_meas, gate, integ0, cfg, dt)
    dq, new_i_srv, dx_srv = servo_residual_step(j, f_cmd, f_meas, gate, integ0, cfg, dt)
    assert torch.allclose(new_i_obs, new_i_srv)
    assert torch.allclose(dx_obs, dx_srv)
    # task_wrench sanity: s ⊙ (F_cmd + I_new), s = (1,1,1) force axes
    assert torch.allclose(tw_obs, f_cmd + new_i_obs)
    # gate=0 hands freeze/zero the integral
    assert torch.allclose(new_i_obs[0, 1], torch.zeros(3))
    assert torch.allclose(new_i_obs[2], torch.zeros(2, 3))


def test_observe_only_updates_integrator_but_never_touches_actions():
    """servo_observe_only=True: the substep hook updates I/Δx from the causal contact,
    but action_manager is NEVER consulted and no Jacobian view is required."""
    from wbt_training.config_values.hybrid import HybridContactConfig

    env = _hybrid_env()
    # observe-only env has NO action_manager — if the injector touches it,
    # AttributeError fails the test loudly. The Jacobian probe must be explicit:
    # a bare MagicMock's get_jacobians() would silently succeed, so make it raise.
    env.action_manager = None
    env.simulator._robot.root_physx_view.get_jacobians = MagicMock(
        side_effect=AssertionError("observe-only path must never fetch Jacobians")
    )
    cfg = HybridContactConfig(servo_enabled=True, servo_observe_only=True)
    inj = _make_injector(env, cfg)
    _pin_plane(env, cfg, inj)
    # press 4 cm into the +x wall → depth>0 → gate on → integral accumulates
    env.simulator._robot.data.body_pos_w[..., 0] -= 0.04
    env._wbt_force_cmd_w[:] = torch.tensor([-5.0, 0.0, 0.0])  # push along −normal
    env.episode_length_buf += 1
    env._apply_force_in_physics_step()
    assert env._wbt_contact_gate.max() > 0
    assert env._wbt_integ_state.abs().max() > 0  # integrator ran
    assert env._wbt_delta_x.abs().max() > 0


def test_observe_only_skips_jacobian_resolution_at_setup():
    from wbt_training.config_values.hybrid import HybridContactConfig

    env = _hybrid_env()
    env.action_manager = None
    cfg = HybridContactConfig(servo_enabled=True, servo_observe_only=True)
    inj = _make_injector(env, cfg)
    assert inj._jac_arm_cols is None  # never resolved
    assert inj._joint_term is None  # never cached


def test_observe_only_default_false_keeps_existing_presets_unchanged():
    from wbt_training.config_values.hybrid import _DEFAULT_HYBRID_CFG, HybridContactConfig

    assert HybridContactConfig().servo_observe_only is False
    assert _DEFAULT_HYBRID_CFG.servo_observe_only is False


# ---------------------------------------------------------------------------
# Task 2: soft contact_phase gate + force-rate prev buffer
# ---------------------------------------------------------------------------
def test_contact_phase_rises_in_contact_and_decays_in_free_space():
    from wbt_training.config_values.hybrid import HybridContactConfig

    env = _hybrid_env()
    env.action_manager = None
    cfg = HybridContactConfig(
        servo_enabled=True, servo_observe_only=True, sensor_noise_std=0.0, sensor_bias_std=0.0, include_inertial=False
    )
    inj = _make_injector(env, cfg)
    _pin_plane(env, cfg, inj)
    assert torch.allclose(env._wbt_contact_phase, torch.zeros(4, 2))
    # 4 cm press @ K=200 → ‖F_ext‖ = 8 N > θ_on=2 → hysteresis ON, LPF ramps toward 1
    env.simulator._robot.data.body_pos_w[..., 0] -= 0.04
    for _step in range(1, 60):
        env.episode_length_buf += 1
        env._apply_force_in_physics_step()
    assert env._wbt_contact_phase.min() > 0.9  # converged near 1
    # release → force 0 < θ_off → hysteresis OFF, phase decays
    env.simulator._robot.data.body_pos_w[..., 0] += 0.2
    for _step in range(60, 120):
        env.episode_length_buf += 1
        env._apply_force_in_physics_step()
    assert env._wbt_contact_phase.max() < 0.1


def test_contact_phase_hysteresis_band_holds_state():
    """1.8 N (between θ_off=1.5 and θ_on=2) must NOT flip the hysteresis state either way."""
    from wbt_training.config_values.hybrid import HybridContactConfig

    env = _hybrid_env()
    env.action_manager = None
    cfg = HybridContactConfig(
        servo_enabled=True, servo_observe_only=True, sensor_noise_std=0.0, sensor_bias_std=0.0, include_inertial=False
    )
    inj = _make_injector(env, cfg)
    _pin_plane(env, cfg, inj)
    # 0.9 cm press @ K=200 → 1.8 N: inside the dead band, starting from OFF → stays OFF
    env.simulator._robot.data.body_pos_w[..., 0] -= 0.009
    for _step in range(1, 40):
        env.episode_length_buf += 1
        env._apply_force_in_physics_step()
    assert env._wbt_contact_phase.max() < 0.05
    # force hysteresis ON manually, same 1.8 N → stays ON (dead band holds)
    env._wbt_contact_hyst[:] = 1.0
    for _step in range(40, 80):
        env.episode_length_buf += 1
        env._apply_force_in_physics_step()
    assert env._wbt_contact_phase.min() > 0.5


def test_contact_phase_obs_reader_and_reset_zeroing():
    from wbt_training.config_values.hybrid import HybridContactConfig, contact_phase

    env = _hybrid_env()
    env.action_manager = None
    cfg = HybridContactConfig(servo_enabled=True, servo_observe_only=True)
    inj = _make_injector(env, cfg)
    _pin_plane(env, cfg, inj)
    env._wbt_contact_phase[:] = 0.7
    assert contact_phase(env).shape == (4, 2)
    assert torch.allclose(contact_phase(env), env._wbt_contact_phase)  # pure reader
    env._wbt_contact_hyst[:] = 1.0
    env._wbt_force_ext_prev_w[:] = 5.0
    inj.reset(torch.tensor([0, 2]))
    assert torch.allclose(env._wbt_contact_phase[[0, 2]], torch.zeros(2, 2))
    assert torch.allclose(env._wbt_contact_phase[[1, 3]], torch.full((2, 2), 0.7))
    # subset reset must also zero the hysteresis + prev-force buffers (an un-reset
    # hyst=1 in the dead band would silently re-inflate the phase post-reset)
    assert torch.allclose(env._wbt_contact_hyst[[0, 2]], torch.zeros(2, 2))
    assert torch.allclose(env._wbt_contact_hyst[[1, 3]], torch.ones(2, 2))
    assert torch.allclose(env._wbt_force_ext_prev_w[[0, 2]], torch.zeros(2, 2, 3))
    assert torch.allclose(env._wbt_force_ext_prev_w[[1, 3]], torch.full((2, 2, 3), 5.0))


def test_force_ext_prev_buffer_captures_last_step():
    from wbt_training.config_values.hybrid import HybridContactConfig

    env = _hybrid_env()
    env.action_manager = None
    cfg = HybridContactConfig(
        servo_enabled=True, servo_observe_only=True, sensor_noise_std=0.0, sensor_bias_std=0.0, include_inertial=False
    )
    inj = _make_injector(env, cfg)
    _pin_plane(env, cfg, inj)
    env.simulator._robot.data.body_pos_w[..., 0] -= 0.04  # 8 N
    env.episode_length_buf += 1
    env._apply_force_in_physics_step()
    f_step1 = env._wbt_force_ext_w[..., :3].clone()
    assert torch.allclose(env._wbt_force_ext_prev_w, torch.zeros(4, 2, 3))  # step-1 prev = 0
    env.episode_length_buf += 1
    env._apply_force_in_physics_step()
    assert torch.allclose(env._wbt_force_ext_prev_w, f_step1)  # step-2 prev = step-1 value


# ---------------------------------------------------------------------------
# Task 3: unified reward terms (hybrid_unified.py)
# ---------------------------------------------------------------------------
def _reward_env(num_envs=4, num_hands=2):
    """Minimal env exposing exactly the buffers the unified rewards read."""
    env = MagicMock()
    env.num_envs = num_envs
    env.device = torch.device("cpu")
    n, h = num_envs, num_hands
    env._wbt_force_cmd_w = torch.zeros(n, h, 3)
    env._wbt_force_ext_w = torch.zeros(n, h, 6)
    env._wbt_force_meas_w = torch.zeros(n, h, 6)
    env._wbt_force_ext_prev_w = torch.zeros(n, h, 3)
    env._wbt_contact_phase = torch.zeros(n, h)
    env._wbt_plane_normal = torch.zeros(n, h, 3)
    env._wbt_plane_normal[..., 0] = 1.0
    env._wbt_plane_point = torch.zeros(n, h, 3)
    env._wbt_plane_seeded = torch.ones(n, dtype=torch.bool)
    return env


def test_force_error_direct_gated_and_maximal_at_cancellation():
    from wbt_training.config_values.hybrid import HybridContactConfig
    from wbt_training.config_values.hybrid_unified import force_error_direct

    cfg = HybridContactConfig()
    env = _reward_env()
    env._wbt_contact_phase[:] = 1.0
    env._wbt_force_cmd_w[..., 0] = -10.0
    env._wbt_force_ext_w[..., 0] = 10.0  # perfect cancellation e=0
    r_perfect = force_error_direct(env, cfg=cfg)
    assert torch.allclose(r_perfect, torch.ones(4))
    env._wbt_force_ext_w[..., 0] = 5.0  # 5 N error per hand
    r_err = force_error_direct(env, cfg=cfg)
    assert (r_err < r_perfect).all() and (r_err > 0).all()
    env._wbt_contact_phase[:] = 0.0  # free space → reward fully gated off
    assert torch.allclose(force_error_direct(env, cfg=cfg), torch.zeros(4))
    # idle-phase farming guard (2026-07-24 review): in contact but F_cmd=0 → no pay
    env._wbt_contact_phase[:] = 1.0
    env._wbt_force_cmd_w[:] = 0.0
    env._wbt_force_ext_w[..., 0] = 2.0  # light uncommanded brush
    assert torch.allclose(force_error_direct(env, cfg=cfg), torch.zeros(4))


def test_contact_init_bonus_pays_soft_contact_below_ceiling():
    from wbt_training.config_values.hybrid import HybridContactConfig
    from wbt_training.config_values.hybrid_unified import contact_init_bonus

    cfg = HybridContactConfig()
    env = _reward_env()
    env._wbt_contact_phase[:] = 1.0
    env._wbt_force_cmd_w[..., 0] = -10.0  # command ON (bonus is cmd-gated)
    env._wbt_force_meas_w[..., 0] = 10.0  # measured force below max_force_n=35
    assert torch.allclose(contact_init_bonus(env, cfg=cfg), torch.ones(4))
    env._wbt_force_meas_w[..., 0] = 50.0  # above ceiling → no bonus
    assert torch.allclose(contact_init_bonus(env, cfg=cfg), torch.zeros(4))
    # idle-phase farming guard (2026-07-24 review): contact without a command → no pay
    env._wbt_force_meas_w[..., 0] = 10.0
    env._wbt_force_cmd_w[:] = 0.0
    assert torch.allclose(contact_init_bonus(env, cfg=cfg), torch.zeros(4))


def test_approach_guidance_active_only_precontact_with_command():
    from wbt_training.config_values.hybrid import HybridContactConfig
    from wbt_training.config_values.hybrid_unified import approach_guidance

    cfg = HybridContactConfig()
    env = _reward_env()
    # TCP FK inputs (reward uses _force_cur_pos_w → body pos/quat + offsets)
    bpos = torch.zeros(4, 2, 3)
    bpos[..., 0] = 0.10  # wrist 10 cm from plane@origin
    bquat = torch.zeros(4, 2, 4)
    bquat[..., 0] = 1.0
    env.simulator._robot.data = types.SimpleNamespace(body_pos_w=bpos, body_quat_w=bquat)
    env.simulator.find_rigid_body_indice = lambda name: {"left_wrist_yaw_link": 0, "right_wrist_yaw_link": 1}[name]
    env.simulator.body_ids = [0, 1]
    env._wbt_force_offsets_b = torch.tensor([(cfg.tcp_offset_x, 0.0, 0.0)] * 2)
    env._wbt_force_isaac_indices = torch.tensor([0, 1], dtype=torch.long)  # the _ensure_* cache attr
    env._wbt_force_cmd_w[..., 0] = -10.0  # command ON
    r_far = approach_guidance(env, cfg=cfg)
    assert (r_far > 0).all()
    bpos[..., 0] = 0.01 - cfg.tcp_offset_x  # TCP 1 cm from plane → closer → larger reward
    r_near = approach_guidance(env, cfg=cfg)
    assert (r_near > r_far).all()
    env._wbt_contact_phase[:] = 1.0  # in contact → guidance off
    assert torch.allclose(approach_guidance(env, cfg=cfg), torch.zeros(4))
    env._wbt_contact_phase[:] = 0.0
    env._wbt_force_cmd_w[:] = 0.0  # no command → no guidance
    assert torch.allclose(approach_guidance(env, cfg=cfg), torch.zeros(4))


def test_force_rate_penalty_is_normalized_delta():
    from wbt_training.config_values.hybrid import HybridContactConfig
    from wbt_training.config_values.hybrid_unified import force_rate_penalty

    cfg = HybridContactConfig()
    env = _reward_env()
    env._wbt_force_ext_w[..., 0] = 10.0
    env._wbt_force_ext_prev_w[..., 0] = 4.0  # ΔF = 6 N × 2 hands = 12 → /25
    r = force_rate_penalty(env, cfg=cfg)
    assert torch.allclose(r, torch.full((4,), 12.0 / cfg.f_cmd_max))


def test_unified_reward_dict_swaps_preintegrator_for_direct():
    from wbt_training.config_values.hybrid_unified import g1_29dof_wbt_unified_reward as rw

    assert "force_error_preintegrator" not in rw.terms
    assert rw.terms["force_error_direct"].weight == 2.0
    assert rw.terms["contact_init_bonus"].weight == 0.3
    assert rw.terms["approach_guidance"].weight == 0.2
    assert rw.terms["force_rate_penalty"].weight == -0.05
    # inherited load-bearing terms intact
    assert rw.terms["force_track_pos_eq2"].weight == 3.0
    assert "motion_relative_body_position_error_exp_no_wrist" in rw.terms
    assert rw.terms["feet_slip"].weight == -2.0


# ---------------------------------------------------------------------------
# Task 4: 185D actor observation
# ---------------------------------------------------------------------------
_EXPECTED_UNIFIED_ACTOR = {
    "obs00_command_lower_body": 24,
    "obs01_vr_3point_pos": 9,
    "obs02_vr_3point_orn_6d": 18,
    "obs03_motion_ref_pos_b": 3,
    "obs04_motion_ref_vel_b": 3,
    "obs05_force_cmd_local": 6,
    "obs06_force_meas_local": 6,
    "obs07_contact_phase": 2,
    "obs08_integrator_state": 6,
    "obs09_x_tgt_shift": 6,
    "obs10_motion_ref_ori_b": 6,
    "obs11_projected_gravity": 3,
    "obs12_base_ang_vel": 3,
    "obs13_base_lin_vel": 3,
    "obs14_dof_pos": 29,
    "obs15_dof_vel": 29,
    "obs16_actions": 29,
}  # = 185


def test_unified_actor_obs_layout_185d():
    from wbt_training.config_values.hybrid_unified import g1_29dof_wbt_observation_3pt_unified_hybrid_global_vel as obs

    actor = obs.groups["actor_obs"]
    assert list(actor.terms.keys()) == list(_EXPECTED_UNIFIED_ACTOR.keys())
    assert list(actor.terms.keys()) == sorted(actor.terms.keys())  # sorted-concat contract
    assert sum(_EXPECTED_UNIFIED_ACTOR.values()) == 185
    assert actor.history_length == 4
    cp = actor.terms["obs07_contact_phase"]
    assert cp.func == "wbt_training.config_values.hybrid:contact_phase"
    assert cp.scale == 1.0 and cp.noise == 0.0


def test_unified_reuses_global_vel_terms_verbatim():
    from wbt_training.config_values.hybrid_3pt_global_vel import _actor_obs_3pt_hybrid_global_vel_terms as base
    from wbt_training.config_values.hybrid_unified import g1_29dof_wbt_observation_3pt_unified_hybrid_global_vel as obs

    actor = obs.groups["actor_obs"].terms
    # every non-contact_phase term is the SAME ObsTermCfg object as the 183D layout
    remap = {  # unified key → 183D key (slots shift by +1 after obs06)
        "obs00_command_lower_body": "obs00_command_lower_body",
        "obs01_vr_3point_pos": "obs01_vr_3point_pos",
        "obs02_vr_3point_orn_6d": "obs02_vr_3point_orn_6d",
        "obs03_motion_ref_pos_b": "obs03_motion_ref_pos_b",
        "obs04_motion_ref_vel_b": "obs04_motion_ref_vel_b",
        "obs05_force_cmd_local": "obs05_force_cmd_local",
        "obs06_force_meas_local": "obs06_force_meas_local",
        "obs08_integrator_state": "obs07_integrator_state",
        "obs09_x_tgt_shift": "obs08_x_tgt_shift",
        "obs10_motion_ref_ori_b": "obs09_motion_ref_ori_b",
        "obs11_projected_gravity": "obs10_projected_gravity",
        "obs12_base_ang_vel": "obs11_base_ang_vel",
        "obs13_base_lin_vel": "obs12_base_lin_vel",
        "obs14_dof_pos": "obs13_dof_pos",
        "obs15_dof_vel": "obs14_dof_vel",
        "obs16_actions": "obs15_actions",
    }
    for uk, bk in remap.items():
        assert actor[uk] is base[bk], (uk, bk)


def test_unified_critic_is_hybrid_critic_verbatim():
    from wbt_training.config_values.hybrid import g1_29dof_wbt_observation_2pt_hybrid
    from wbt_training.config_values.hybrid_unified import g1_29dof_wbt_observation_3pt_unified_hybrid_global_vel as obs

    # ObsGroupCfg copies the dict at construction; verbatim reuse = per-term object
    # identity (same assertion style as test_hybrid_3pt_global_vel).
    c2 = g1_29dof_wbt_observation_2pt_hybrid.groups["critic_obs"].terms
    cu = obs.groups["critic_obs"].terms
    assert set(cu.keys()) == set(c2.keys())
    assert all(cu[k] is c2[k] for k in c2), "every critic term must be the SAME object (verbatim reuse)"
    assert obs.groups["critic_obs"].history_length == 10


# ---------------------------------------------------------------------------
# Task 5: ServoAuthorityCurriculum (R1 anneal) + alpha-scaled servo injection
# ---------------------------------------------------------------------------
def _authority_curr(env, threshold=2.0, step=0.02):
    from wbt_training.config_values.hybrid import _DEFAULT_HYBRID_CFG
    from wbt_training.config_values.hybrid_unified import ServoAuthorityCurriculum

    term_cfg = MagicMock()
    term_cfg.params = {
        "hybrid_config": _DEFAULT_HYBRID_CFG,
        "force_err_threshold_n": threshold,
        "alpha_step": step,
        "ema_beta": 1.0,  # EMA = latest sample (test determinism)
        "hold_resets": 3,  # small hold so tests exercise the consecutive-hold gate fast
    }
    cur = ServoAuthorityCurriculum(term_cfg, env)
    cur.setup()
    return cur


def test_servo_authority_anneals_only_when_error_below_threshold():
    env = _reward_env()
    env.device = torch.device("cpu")
    env._wbt_servo_alpha = None  # force fresh allocation (MagicMock attr would be truthy)
    cur = _authority_curr(env)
    assert float(env._wbt_servo_alpha) == 1.0
    # in-contact, big error (10 N) → NO anneal ever
    env._wbt_contact_phase[:] = 1.0
    env._wbt_force_cmd_w[..., 0] = -10.0
    env._wbt_force_ext_w[..., 0] = 0.0
    for _ in range(5):
        cur.reset(torch.arange(4))
    assert float(env._wbt_servo_alpha) == 1.0
    # near-perfect tracking (error 1 N < 2 N): must HOLD below threshold for
    # hold_resets=3 consecutive resets before ONE −0.02 drop (not drop-per-reset).
    env._wbt_force_ext_w[..., 0] = 9.0
    cur.reset(torch.arange(4))
    cur.reset(torch.arange(4))
    assert float(env._wbt_servo_alpha) == 1.0  # 2 < hold_resets → no drop yet
    cur.reset(torch.arange(4))
    assert abs(float(env._wbt_servo_alpha) - 0.98) < 1e-6  # 3rd consecutive → one drop
    # a violation resets the hold counter
    env._wbt_force_ext_w[..., 0] = 0.0  # error back to 10 N
    cur.reset(torch.arange(4))
    env._wbt_force_ext_w[..., 0] = 9.0
    cur.reset(torch.arange(4))
    cur.reset(torch.arange(4))
    assert abs(float(env._wbt_servo_alpha) - 0.98) < 1e-6  # only 2 held since violation
    # keeps going down monotonically, clamped at 0
    for _ in range(400):
        cur.reset(torch.arange(4))
    assert float(env._wbt_servo_alpha) == 0.0


def test_servo_authority_holds_until_force_alpha_full():
    """While the force-alpha curriculum is still ramping (α_g<1), F_cmd is scaled
    down and small errors mean small FORCES, not good tracking — the anneal must
    hold until the force task is at full strength (review F1)."""
    env = _reward_env()
    env.device = torch.device("cpu")
    env._wbt_servo_alpha = None
    cur = _authority_curr(env)
    env._wbt_contact_phase[:] = 1.0
    env._wbt_force_cmd_w[..., 0] = -1.0  # tiny ramped command
    env._wbt_force_ext_w[..., 0] = 1.0  # 'perfect' tracking of a tiny force
    env._wbt_force_alpha = torch.tensor(0.3)  # ramp in progress
    for _ in range(10):
        cur.reset(torch.arange(4))
    assert float(env._wbt_servo_alpha) == 1.0  # held during the ramp
    env._wbt_force_alpha = torch.tensor(1.0)  # ramp complete → gate opens
    for _ in range(3):  # hold_resets=3 in the test harness
        cur.reset(torch.arange(4))
    assert abs(float(env._wbt_servo_alpha) - 0.98) < 1e-6


def test_servo_authority_ignores_uncommanded_contact():
    """Incidental contact with F_cmd≈0 (force-alpha ramp start) must NOT feed the
    gate — ‖F_ext‖ alone is not a tracking error."""
    env = _reward_env()
    env.device = torch.device("cpu")
    env._wbt_servo_alpha = None
    cur = _authority_curr(env)
    env._wbt_contact_phase[:] = 1.0  # touching
    env._wbt_force_cmd_w[:] = 0.0  # no command
    env._wbt_force_ext_w[..., 0] = 1.0  # small brush, ‖e‖=1 < threshold
    for _ in range(10):
        cur.reset(torch.arange(4))
    assert float(env._wbt_servo_alpha) == 1.0  # held — no commanded sample


def test_servo_authority_no_contact_no_anneal():
    env = _reward_env()
    env.device = torch.device("cpu")
    env._wbt_servo_alpha = None
    cur = _authority_curr(env)
    env._wbt_contact_phase[:] = 0.0  # free space: no sample → hold α
    cur.reset(torch.arange(4))
    assert float(env._wbt_servo_alpha) == 1.0


def test_servo_step_scales_injection_by_alpha():
    """With _wbt_servo_alpha=0 the FULL servo path must inject exactly the policy
    actions (zero net shift) — terminal R1 state ≡ observe-only actuation-wise."""
    from wbt_training.config_values.hybrid import HybridContactConfig

    cfg = HybridContactConfig(servo_enabled=True)
    env, inj = _servo_env_wired(cfg)
    _pin_plane(env, cfg, inj)
    env.simulator._robot.data.body_pos_w[..., 0] -= 0.04
    env._wbt_force_cmd_w[:] = torch.tensor([-5.0, 0.0, 0.0])
    joint_term = env.action_manager.get_term("joint_control")
    policy_actions = joint_term._actions_after_delay.clone()
    env._wbt_servo_alpha = torch.tensor(0.0)
    env.episode_length_buf += 1
    env._apply_force_in_physics_step()
    assert torch.allclose(joint_term._actions_after_delay, policy_actions)  # α=0 → no shift
    env._wbt_servo_alpha = torch.tensor(1.0)
    env.episode_length_buf += 1
    env._apply_force_in_physics_step()
    assert not torch.allclose(joint_term._actions_after_delay, policy_actions)  # α=1 → servo acts


# ---------------------------------------------------------------------------
# Task 6: experiment presets (R2 unified + R1 unified-anneal)
# ---------------------------------------------------------------------------
def test_unified_presets_registered_and_wired():
    from wbt_training.config_values.experiment import DEFAULTS
    from wbt_training.config_values.hybrid_unified import (
        _UNIFIED_ANNEAL_CFG,
        _UNIFIED_CFG,
        g1_29dof_wbt_observation_3pt_unified_hybrid_global_vel,
    )

    p = DEFAULTS["g1_wbt_3pt_hybrid_global_vel_unified"]
    a = DEFAULTS["g1_wbt_3pt_hybrid_global_vel_unified_anneal"]
    assert p.observation is g1_29dof_wbt_observation_3pt_unified_hybrid_global_vel
    assert a.observation is g1_29dof_wbt_observation_3pt_unified_hybrid_global_vel  # same 185D both
    # R2 injector carries observe-only; R1 carries the full-servo cfg
    p_inj = p.command.setup_terms["hybrid_contact_injector"].params["hybrid_config"]
    a_inj = a.command.setup_terms["hybrid_contact_injector"].params["hybrid_config"]
    assert p_inj is _UNIFIED_CFG and p_inj.servo_observe_only is True and p_inj.servo_enabled is True
    assert a_inj is _UNIFIED_ANNEAL_CFG and a_inj.servo_observe_only is False and a_inj.servo_enabled is True
    # rewards: unified terms present, preintegrator gone
    for exp in (p, a):
        assert "force_error_direct" in exp.reward.terms
        assert "force_error_preintegrator" not in exp.reward.terms
    # curriculum: force-alpha ramp ENABLED on both; servo-authority ONLY on the anneal twin
    p_fa = p.curriculum.setup_terms["force_alpha"].params["hand_force_config"]
    a_fa = a.curriculum.setup_terms["force_alpha"].params["hand_force_config"]
    for fa in (p_fa, a_fa):
        assert fa.force_alpha_init == 0.0 and fa.alpha_step == 0.01
    assert "servo_authority" not in p.curriculum.setup_terms
    assert "servo_authority" in a.curriculum.setup_terms
    assert "servo_authority" in a.curriculum.reset_terms
    # termination: force_runaway estop wired with the matching cfg
    assert p.termination.terms["force_runaway"].params["cfg"] is _UNIFIED_CFG
    assert a.termination.terms["force_runaway"].params["cfg"] is _UNIFIED_ANNEAL_CFG


def test_unified_preset_inherits_3pt_global_vel_base():
    from wbt_training.config_values.experiment import DEFAULTS

    base = DEFAULTS["g1_wbt_3pt_hybrid_global_vel"]
    p = DEFAULTS["g1_wbt_3pt_hybrid_global_vel_unified"]
    a = DEFAULTS["g1_wbt_3pt_hybrid_global_vel_unified_anneal"]
    assert p.robot is base.robot  # yam URDF etc. inherited
    assert p.algo is base.algo
    assert a.robot is base.robot


def test_unified_presets_do_not_disturb_base_preset():
    """The base exp:g1-wbt-3pt-hybrid-global-vel (train + eval) must be untouched:
    same 183D obs object, same servo-ON command/reward/termination as before the
    unified fork (no shared-object mutation)."""
    from wbt_training.config_values.experiment import DEFAULTS
    from wbt_training.config_values.hybrid import (
        g1_29dof_wbt_2pt_hybrid_reward,
        g1_29dof_wbt_hybrid_command,
        g1_29dof_wbt_hybrid_termination,
    )
    from wbt_training.config_values.hybrid_3pt_global_vel import g1_29dof_wbt_observation_3pt_hybrid_global_vel

    base = DEFAULTS["g1_wbt_3pt_hybrid_global_vel"]
    assert base.observation is g1_29dof_wbt_observation_3pt_hybrid_global_vel  # 183D untouched
    assert base.command is g1_29dof_wbt_hybrid_command
    assert base.reward is g1_29dof_wbt_2pt_hybrid_reward
    assert base.termination is g1_29dof_wbt_hybrid_termination
    hc = base.command.setup_terms["hybrid_contact_injector"].params["hybrid_config"]
    assert hc.servo_enabled is True and hc.servo_observe_only is False
    # base reward still carries the preintegrator term (only the unified fork drops it)
    assert "force_error_preintegrator" in base.reward.terms
    # base curriculum's force-alpha stays DISABLED (init 1.0, step 0.0)
    fa = base.curriculum.setup_terms["force_alpha"].params["hand_force_config"]
    assert fa.force_alpha_init == 1.0 and fa.alpha_step == 0.0


# ---------------------------------------------------------------------------
# Task 7: inference configs (185D, training↔inference agreement)
# ---------------------------------------------------------------------------
def test_inference_config_matches_training_185d():
    from wbt_wrappers_inference.config_values.observation import holosoma_3pt_hybrid_global_vel_unified_wbt as iobs

    rename = {"dof_pos": "joint_pos", "dof_vel": "joint_vel"}
    total = 0
    for k, dim in _EXPECTED_UNIFIED_ACTOR.items():
        bare = rename.get(k.split("_", 1)[1], k.split("_", 1)[1])
        assert iobs.obs_dims[bare] == dim, (k, bare)
        total += iobs.obs_dims[bare]
    assert total == 185
    assert list(iobs.obs_dict["actor_obs"]) == [
        rename.get(k.split("_", 1)[1], k.split("_", 1)[1]) for k in _EXPECTED_UNIFIED_ACTOR
    ]
    assert iobs.obs_scales["contact_phase"] == 1.0
    assert iobs.obs_scales["force_cmd_local"] == 0.04
    assert iobs.history_length_dict["actor_obs"] == 4


def test_inference_presets_registered():
    from wbt_wrappers_inference.config_values.inference import DEFAULTS as INF_DEFAULTS

    a = INF_DEFAULTS["g1-29dof-holosoma-3pt-hybrid-global-vel-unified-wbt"]
    b = INF_DEFAULTS["g1-29dof-holosoma-3pt-hybrid-global-vel-unified-anneal-wbt"]
    assert a is b  # same 185D obs serves both training twins
    # base 183D inference preset untouched
    base = INF_DEFAULTS["g1-29dof-holosoma-3pt-hybrid-global-vel-wbt"]
    assert base is not a
    assert "contact_phase" not in base.observation.obs_dims


# ---------------------------------------------------------------------------
# Review-round hardening (2026-07-24): real-dt LPF pin, idempotence, clamp, warmup.
# ---------------------------------------------------------------------------
def test_substep_dt_reads_sim_dt_and_falls_back():
    from wbt_training.config_values.hybrid import HybridContactConfig

    env = _hybrid_env()
    env.action_manager = None
    cfg = HybridContactConfig(servo_enabled=True, servo_observe_only=True)
    inj = _make_injector(env, cfg)
    # MagicMock child attrs are NOT numbers → fallback 0.005 (never float(MagicMock)=1.0)
    assert inj._substep_dt() == 0.005
    env.simulator.sim_dt = 1.0 / 200.0  # the real holosoma attr (= 1/fps)
    assert inj._substep_dt() == 0.005
    env.simulator.sim_dt = 0.01
    assert inj._substep_dt() == 0.01


def test_contact_phase_lpf_exact_first_substep():
    """Sharp pin on the LPF constant: one substep at dt=0.005, τ=0.05 →
    phase == 1−exp(−0.1) ≈ 0.09516 exactly (kills wrong-beta mutants)."""
    import math

    from wbt_training.config_values.hybrid import HybridContactConfig

    env = _hybrid_env()
    env.action_manager = None
    cfg = HybridContactConfig(
        servo_enabled=True,
        servo_observe_only=True,
        sensor_noise_std=0.0,
        sensor_bias_std=0.0,
        include_inertial=False,
        contact_phase_warmup_steps=0,
    )
    inj = _make_injector(env, cfg)
    _pin_plane(env, cfg, inj)
    env.simulator.sim_dt = 0.005
    env.simulator._robot.data.body_pos_w[..., 0] -= 0.04  # 8 N > θ_on
    env.episode_length_buf += 1
    env._apply_force_in_physics_step()
    expected = 1.0 - math.exp(-0.005 / 0.05)
    assert torch.allclose(env._wbt_contact_phase, torch.full((4, 2), expected), atol=1e-5)


def test_force_ext_prev_capture_once_per_env_step():
    """Idempotence: two hook calls at the SAME episode_length_buf must NOT re-capture
    (else the force-rate delta collapses to the same step's substep value)."""
    from wbt_training.config_values.hybrid import HybridContactConfig

    env = _hybrid_env()
    env.action_manager = None
    cfg = HybridContactConfig(
        servo_enabled=True, servo_observe_only=True, sensor_noise_std=0.0, sensor_bias_std=0.0, include_inertial=False
    )
    inj = _make_injector(env, cfg)
    _pin_plane(env, cfg, inj)
    env.simulator._robot.data.body_pos_w[..., 0] -= 0.04  # 8 N
    env.episode_length_buf += 1
    env._apply_force_in_physics_step()  # substep 0: capture fires (prev ← 0)
    assert torch.allclose(env._wbt_force_ext_prev_w, torch.zeros(4, 2, 3))
    env._apply_force_in_physics_step()  # substep 1, SAME step: must NOT re-capture
    assert torch.allclose(env._wbt_force_ext_prev_w, torch.zeros(4, 2, 3))
    env.episode_length_buf += 1
    env._apply_force_in_physics_step()  # next step: prev ← step-1 value (8 N)
    assert env._wbt_force_ext_prev_w.abs().max() > 0


def test_integrator_observe_step_antiwindup_clamp():
    from wbt_training.config_values.hybrid import HybridContactConfig, integrator_observe_step

    cfg = HybridContactConfig(servo_enabled=True)
    n, h = 2, 2
    f_cmd = torch.full((n, h, 3), 100.0)
    f_meas = torch.zeros(n, h, 3)
    integ0 = torch.full((n, h, 3), 100.0)  # far beyond ±integrator_max=40
    gate = torch.ones(n, h)
    new_i, _, _ = integrator_observe_step(f_cmd, f_meas, gate, integ0, cfg, dt=0.005)
    assert new_i.abs().max() <= cfg.integrator_max + 1e-6


def test_contact_phase_warmup_suppresses_reset_transient():
    """Post-reset warmup: even with ‖F_meas‖ ≫ θ_on (the reset-teleport inertial
    spike), the hysteresis stays OFF for the first warmup_steps env steps."""
    from wbt_training.config_values.hybrid import HybridContactConfig

    env = _hybrid_env()
    env.action_manager = None
    cfg = HybridContactConfig(
        servo_enabled=True, servo_observe_only=True, sensor_noise_std=0.0, sensor_bias_std=0.0, include_inertial=False
    )
    inj = _make_injector(env, cfg)
    _pin_plane(env, cfg, inj)
    env.simulator._robot.data.body_pos_w[..., 0] -= 0.04  # 8 N ≫ θ_on from step 1
    for step in range(1, cfg.contact_phase_warmup_steps):  # steps 1..4 < warmup=5
        env.episode_length_buf = torch.full((4,), step, dtype=torch.long)
        env._apply_force_in_physics_step()
    assert env._wbt_contact_phase.max() < 0.05  # suppressed during warmup
    for step in range(cfg.contact_phase_warmup_steps, cfg.contact_phase_warmup_steps + 60):
        env.episode_length_buf = torch.full((4,), step, dtype=torch.long)
        env._apply_force_in_physics_step()
    assert env._wbt_contact_phase.min() > 0.9  # engages normally after warmup
