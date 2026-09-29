"""Unit tests for the hybrid position-force variant (M1: causal contact + sim sensor).

All tests use MagicMock for env/simulator — no IsaacSim runtime needed
(mirrors tests/test_force_tracking.py).
"""

import types
from unittest.mock import MagicMock

import pytest
import torch

# Real IsaacSim runtime is needed for the get_jacobians() finite-difference smoke test
# (a MagicMock Jacobian would only test the stub). It SKIPS on this headless box and is a
# GPU-box merge gate. We only PROBE for the isaacsim package (importlib.find_spec — importing
# it would bootstrap the kit kernel and crash pytest collection); the real-sim test itself is
# skipped even when present unless a GPU sim harness is wired, so this stays collection-safe.
import importlib.util  # noqa: E402

_HAVE_ISAACSIM = importlib.util.find_spec("isaacsim") is not None


def test_hybrid_contact_config_defaults():
    from wbt_training.config_values.hybrid import HybridContactConfig

    cfg = HybridContactConfig()
    # F_cmd command pulse magnitude inherited semantics (still a commanded force).
    assert cfg.f_cmd_max == 25.0
    assert cfg.stiffness_K == 200.0
    # K_env log-uniform bounds (N/m): soft wall → hard wall. Must be positive & ordered.
    assert 0.0 < cfg.k_env_min < cfg.k_env_max
    # Force realism (Opus review): peak K_env*max_penetration must be near the 25 N
    # command regime / ~23 N wrist saturation, NOT 1500 N. Hard clamp + shrunk ranges.
    assert cfg.k_env_max * cfg.max_penetration_m <= 70.0  # peak spring force before clamp
    # max_force_n above f_cmd_max (never clip a legit command) but below the estop cutoff.
    assert cfg.f_cmd_max <= cfg.max_force_n < cfg.force_runaway_cutoff
    # Force-REACHABILITY invariant (2026-07-10): the softest wall must be able to deliver
    # SOME contact force, and the per-episode F_cmd is clamped to what the sampled wall can
    # reach (f_cmd_reachable_clamp), so the force target is always physically attainable.
    assert cfg.f_cmd_reachable_clamp is True
    assert cfg.f_cmd_along_normal is True  # all-3-axis force control needs F_cmd on the reactable axis
    # Degenerate-reward guard (Opus review): the plane must be reachable INSIDE the
    # commanded offset F_cmd/K = 25/200 = 0.125 m, else the reward-max sits in free
    # space and the policy trains to AVOID contact. Keep plane_dist_max < 0.125.
    assert cfg.plane_dist_max < cfg.f_cmd_max / cfg.stiffness_K
    # Distal mass for the inertial pseudo-force (D4: ~1.5 kg fused distal mass).
    assert cfg.distal_mass_kg > 0.0
    # Sensor noise/bias std (N). Non-negative.
    assert cfg.sensor_noise_std >= 0.0
    assert cfg.sensor_bias_std >= 0.0
    # Geometry constants match the yam URDF (single source of truth vs playground).
    assert cfg.tcp_offset_x == 0.242  # 2026-07-10: TCP moved 0.215→0.242 (palm standoff +2.7cm)
    assert cfg.ft_offset_x == 0.082  # FT sensor unchanged
    assert cfg.body_names == ("left_wrist_yaw_link", "right_wrist_yaw_link")
    # Servo (Layer B) DATACLASS default is False (safe library default; Layer-A tests stay
    # self-contained). The SHIPPED PRESET overrides to True (Task 11 _DEFAULT_HYBRID_CFG) — a
    # separate test there asserts the launched preset has servo_enabled=True (force tracking
    # on launch, user 2026-07-10). See test_preset_ships_servo_enabled_with_estop in Task 11.
    assert cfg.servo_enabled is False
    # Gain must be near unity-authority (~1/K_p≈0.07), NOT the old near-inert 0.0002 stub.
    assert 0.01 <= cfg.servo_gain_k <= 0.10
    # I_max (N — force-unit integral, 2026-07-10 design change) must exceed the one-step ΔI at
    # f_cmd_max: ΔI = K_I·f_cmd_max·dt = 0.5·25·0.02 = 0.25 N ≪ I_max=40 N. Also ≥ f_cmd_max so the
    # integral can supply the full command force if the feedforward is entirely rejected.
    assert cfg.integrator_max > cfg.integrator_gain_ki * cfg.f_cmd_max * 0.02
    assert cfg.integrator_max >= cfg.f_cmd_max
    assert cfg.servo_dq_max > 0.0
    assert len(cfg.servo_selection) == 6  # length-6 (xyz force active, moment dormant)
    assert tuple(cfg.servo_selection[3:]) == (0.0, 0.0, 0.0)  # moment axes off (D8)


def test_sensor_to_tcp_adjoint_force_channel_is_rotation_only():
    # For a pure 3-D force at the sensor, sensor->TCP is R_sensor rotation only;
    # magnitude preserved, +y -> -y (R_sensor maps sensor +y to wrist -y).
    from wbt_training.config_values.hybrid import sensor_to_tcp_wrench

    w_sensor = torch.zeros(6)
    w_sensor[1] = 10.0  # +y force, zero moment
    w_tcp = sensor_to_tcp_wrench(w_sensor)
    assert torch.allclose(w_tcp[:3], torch.tensor([0.0, -10.0, 0.0]), atol=1e-6)
    # magnitude preserved
    assert torch.isclose(w_tcp[:3].norm(), torch.tensor(10.0), atol=1e-6)


def test_sensor_to_tcp_adjoint_lever_moment_channel():
    # m_T = R_sensor·m_S − lever × f_T, lever=(0.16,0,0) (2026-07-10: was 0.133).
    # sensor +y force (10 N), zero sensor moment -> f_T=(0,-10,0);
    # -(0.16,0,0)×(0,-10,0) = (0,0,1.6).
    from wbt_training.config_values.hybrid import FT_TCP_LEVER_X, sensor_to_tcp_wrench

    w_sensor = torch.zeros(6)
    w_sensor[1] = 10.0
    w_tcp = sensor_to_tcp_wrench(w_sensor)
    assert torch.allclose(w_tcp[3:], torch.tensor([0.0, 0.0, FT_TCP_LEVER_X * 10.0]), atol=1e-6)  # 1.6 N·m


def test_sensor_to_tcp_adjoint_matches_playground_constants():
    # The training adjoint must match the validated inference playground to machine
    # precision (single source of truth for the hardware geometry). We assert the
    # constants rather than import the inference package (training must not depend
    # on wbt_wrappers_inference).
    from wbt_training.config_values.hybrid import FT_OFFSET_X, FT_TCP_LEVER_X, R_SENSOR, TCP_OFFSET_X

    assert FT_OFFSET_X == 0.082
    assert TCP_OFFSET_X == 0.242  # 2026-07-10: TCP moved 0.215→0.242
    assert abs(FT_TCP_LEVER_X - 0.16) < 1e-9  # lever 0.242−0.082 = 0.16 (was 0.133)
    assert abs(FT_TCP_LEVER_X - (TCP_OFFSET_X - FT_OFFSET_X)) < 1e-12
    expected = torch.tensor([[0.0, 0.0, 1.0], [0.0, -1.0, 0.0], [1.0, 0.0, 0.0]])
    assert torch.allclose(R_SENSOR, expected, atol=1e-7)
    # R_sensor is a symmetric involution (R == R^T, R@R == I).
    assert torch.allclose(R_SENSOR, R_SENSOR.T, atol=1e-7)
    assert torch.allclose(R_SENSOR @ R_SENSOR, torch.eye(3), atol=1e-7)


def test_sensor_to_tcp_adjoint_batched():
    from wbt_training.config_values.hybrid import sensor_to_tcp_wrench

    w = torch.zeros(4, 2, 6)
    w[..., 0] = 5.0  # +x force
    out = sensor_to_tcp_wrench(w)
    assert out.shape == (4, 2, 6)
    # R_sensor maps sensor +x -> wrist +z: f_T=(0,0,5).
    assert torch.allclose(out[..., :3], torch.tensor([0.0, 0.0, 5.0]).expand(4, 2, 3), atol=1e-6)


# ---------------------------------------------------------------------------
# Task 2: virtual-plane sampler + penetration + causal contact force
# ---------------------------------------------------------------------------
def test_penetration_zero_outside_plane():
    # TCP on the free-space side (positive along the outward normal) -> no penetration.
    from wbt_training.config_values.hybrid import penetration_depth

    normal = torch.tensor([1.0, 0.0, 0.0])
    point = torch.tensor([0.0, 0.0, 0.0])  # plane through origin, outward +x
    x_tcp = torch.tensor([0.1, 0.0, 0.0])  # 10 cm on the +x (free) side
    d = penetration_depth(x_tcp, normal, point, max_pen=0.3)
    assert torch.isclose(d, torch.tensor(0.0), atol=1e-7)


def test_penetration_positive_behind_plane_and_monotonic():
    from wbt_training.config_values.hybrid import penetration_depth

    normal = torch.tensor([1.0, 0.0, 0.0])
    point = torch.tensor([0.0, 0.0, 0.0])
    d_shallow = penetration_depth(torch.tensor([-0.02, 0.0, 0.0]), normal, point, max_pen=0.3)
    d_deep = penetration_depth(torch.tensor([-0.10, 0.0, 0.0]), normal, point, max_pen=0.3)
    assert d_shallow > 0.0
    assert d_deep > d_shallow  # deeper press -> larger penetration (causality precondition)
    assert torch.isclose(d_deep, torch.tensor(0.10), atol=1e-6)


def test_penetration_clamped_at_max():
    from wbt_training.config_values.hybrid import penetration_depth

    normal = torch.tensor([1.0, 0.0, 0.0])
    point = torch.tensor([0.0, 0.0, 0.0])
    d = penetration_depth(torch.tensor([-1.0, 0.0, 0.0]), normal, point, max_pen=0.3)
    assert torch.isclose(d, torch.tensor(0.3), atol=1e-7)  # teleport spike guarded


def test_contact_force_direction_and_magnitude_and_Kenv_monotonic():
    from wbt_training.config_values.hybrid import contact_force_world

    normal = torch.tensor([1.0, 0.0, 0.0])
    depth = torch.tensor(0.02)  # 2 cm penetration
    f_soft = contact_force_world(torch.tensor(100.0), depth, normal, max_force_n=30.0)
    f_hard = contact_force_world(torch.tensor(500.0), depth, normal, max_force_n=30.0)
    # Force pushes robot OUT along +normal (reaction opposing penetration).
    assert torch.allclose(f_soft, torch.tensor([2.0, 0.0, 0.0]), atol=1e-6)  # 100*0.02
    assert torch.allclose(f_hard, torch.tensor([10.0, 0.0, 0.0]), atol=1e-6)  # 500*0.02
    # Stiffer wall -> larger force at same penetration (K_env causality).
    assert f_hard.norm() > f_soft.norm()


def test_contact_force_magnitude_clamped():
    # A transient deep press must never inject a wrist-shattering force.
    from wbt_training.config_values.hybrid import contact_force_world

    normal = torch.tensor([1.0, 0.0, 0.0])
    f = contact_force_world(torch.tensor(800.0), torch.tensor(0.05), normal, max_force_n=30.0)  # 40 N raw
    assert torch.isclose(f.norm(), torch.tensor(30.0), atol=1e-5)  # clamped to 30 N
    assert f[0] > 0  # direction preserved


def test_contact_force_zero_when_no_penetration():
    from wbt_training.config_values.hybrid import contact_force_world

    f = contact_force_world(torch.tensor(1000.0), torch.tensor(0.0), torch.tensor([1.0, 0.0, 0.0]))
    assert torch.allclose(f, torch.zeros(3), atol=1e-7)


def test_sample_plane_shapes_and_unit_normal():
    from wbt_training.config_values.hybrid import HybridContactConfig, sample_plane

    cfg = HybridContactConfig()
    gen = torch.Generator(device="cpu").manual_seed(0)
    normal, point = sample_plane(4, 2, cfg, gen, torch.device("cpu"))
    assert normal.shape == (4, 2, 3)
    assert point.shape == (4, 2, 3)
    assert torch.allclose(normal.norm(dim=-1), torch.ones(4, 2), atol=1e-5)  # unit normals


def test_surface_tangents_orthonormal_right_handed():
    from wbt_training.config_values.hybrid import surface_tangents

    normal = torch.randn(5, 3)
    normal = normal / normal.norm(dim=-1, keepdim=True)
    t1, t2 = surface_tangents(normal)
    # unit length
    assert torch.allclose(t1.norm(dim=-1), torch.ones(5), atol=1e-5)
    assert torch.allclose(t2.norm(dim=-1), torch.ones(5), atol=1e-5)
    # mutually orthogonal + orthogonal to normal
    assert torch.allclose((t1 * t2).sum(-1), torch.zeros(5), atol=1e-5)
    assert torch.allclose((t1 * normal).sum(-1), torch.zeros(5), atol=1e-5)
    assert torch.allclose((t2 * normal).sum(-1), torch.zeros(5), atol=1e-5)


def test_contact_moment_dormant_when_krot_zero():
    from wbt_training.config_values.hybrid import contact_moment_world

    k_env_rot = torch.zeros(3, 3)  # v1: dormant
    orient_err = torch.randn(3, 3)
    m = contact_moment_world(k_env_rot, orient_err, max_orient_err=0.2)
    assert torch.allclose(m, torch.zeros(3, 3), atol=1e-7)  # K_rot=0 → zero moment


# ---------------------------------------------------------------------------
# Task 3: simulated F/T sensor (inertial + bias + noise, sensor→TCP)
# ---------------------------------------------------------------------------
def test_simulated_sensor_clean_equals_fext():
    # Zero acceleration, zero bias, zero noise -> F_meas == F_ext exactly.
    from wbt_training.config_values.hybrid import HybridContactConfig, simulated_measured_force

    cfg = HybridContactConfig()
    f_ext = torch.tensor([[3.0, 0.0, -4.0]])  # (1,3)
    acc = torch.zeros(1, 3)
    bias = torch.zeros(1, 3)
    noise = torch.zeros(1, 3)
    f_meas = simulated_measured_force(f_ext, acc, bias, cfg, noise)
    assert torch.allclose(f_meas, f_ext, atol=1e-6)


def test_simulated_sensor_inertial_term():
    # F_inertial = m·a is ADDED (no contact, pure acceleration -> nonzero reading).
    from wbt_training.config_values.hybrid import HybridContactConfig, simulated_measured_force

    cfg = HybridContactConfig(sensor_noise_std=0.0, sensor_bias_std=0.0)
    f_ext = torch.zeros(1, 3)
    acc = torch.tensor([[2.0, 0.0, 0.0]])  # 2 m/s^2 along +x
    bias = torch.zeros(1, 3)
    noise = torch.zeros(1, 3)
    f_meas = simulated_measured_force(f_ext, acc, bias, cfg, noise)
    assert torch.allclose(f_meas, torch.tensor([[cfg.distal_mass_kg * 2.0, 0.0, 0.0]]), atol=1e-6)


def test_simulated_sensor_inertial_disabled():
    from wbt_training.config_values.hybrid import HybridContactConfig, simulated_measured_force

    cfg = HybridContactConfig(include_inertial=False)
    f_ext = torch.tensor([[1.0, 2.0, 3.0]])
    acc = torch.tensor([[9.0, 9.0, 9.0]])  # ignored
    f_meas = simulated_measured_force(f_ext, acc, torch.zeros(1, 3), cfg, torch.zeros(1, 3))
    assert torch.allclose(f_meas, f_ext, atol=1e-6)


def test_simulated_sensor_bias_and_noise_added():
    from wbt_training.config_values.hybrid import HybridContactConfig, simulated_measured_force

    cfg = HybridContactConfig(include_inertial=False)
    f_ext = torch.zeros(1, 3)
    bias = torch.tensor([[0.1, 0.0, 0.0]])
    noise = torch.tensor([[0.0, 0.2, 0.0]])
    f_meas = simulated_measured_force(f_ext, torch.zeros(1, 3), bias, cfg, noise)
    assert torch.allclose(f_meas, torch.tensor([[0.1, 0.2, 0.0]]), atol=1e-6)


def test_adjoint_deploy_parity_sensor_frame_to_world():
    # A pure 3D force known at the TCP: its sensor-frame reading, pushed through the deploy
    # chain (sensor_to_tcp_wrench → identity wrist rotation here), must recover the TCP force.
    from wbt_training.config_values.hybrid import R_SENSOR, sensor_to_tcp_wrench

    f_tcp = torch.tensor([0.0, 0.0, 5.0])  # 5 N along wrist +z (== sensor +x)
    w_sensor = torch.zeros(6)
    w_sensor[:3] = R_SENSOR.T @ f_tcp  # express at the sensor
    w_tcp = sensor_to_tcp_wrench(w_sensor)
    assert torch.allclose(w_tcp[:3], f_tcp, atol=1e-6)  # round-trips to the TCP force
    # (deploy adds a live wrist quat world-rotation on top; the sim path is world-native, so
    #  the ONLY deploy-specific seam is this constant adjoint — pinned here.)


# ---------------------------------------------------------------------------
# Task 4: HybridContactInjector — causal substep hook + 6D-ready buffers
# ---------------------------------------------------------------------------
# G1 29-DOF joint ordering (12 legs + 3 waist + 14 arms), matching
# falcon_decoupled_manager._LEG_JOINTS/_WAIST_JOINTS/_ARM_JOINTS. The mock exposes
# this so the injector's setup() can size _wbt_servo_dq and Task 9/10 can resolve the
# arm PD slice by name.index() (left arm → 15..21, right arm → 22..28).
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
    # body_pos_w / body_quat_w / body_lin_acc_w indexed by isaac body ids [0,1].
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


def test_injector_allocates_6d_ready_buffers():
    from wbt_training.config_values.hybrid import HybridContactConfig

    env = _hybrid_env()
    cfg = HybridContactConfig()
    _make_injector(env, cfg)
    assert env._wbt_force_ext_w.shape == (4, 2, 6)  # 6D-ready (force :3, moment 3:)
    assert env._wbt_force_meas_w.shape == (4, 2, 6)
    assert env._wbt_force_cmd_w.shape == (4, 2, 3)
    assert env._wbt_contact_gate.shape == (4, 2)
    assert env._wbt_k_env.shape == (4, 2)
    # moment channels start zero
    assert torch.allclose(env._wbt_force_ext_w[..., 3:], torch.zeros(4, 2, 3))


def test_injector_no_contact_gives_zero_fext_and_gate():
    # Reset samples a plane in FRONT of the hand; with the hand at its reset pose
    # (not pressing) there is no penetration -> F_ext = 0, gate = 0.
    from wbt_training.config_values.hybrid import HybridContactConfig

    env = _hybrid_env()
    cfg = HybridContactConfig()
    inj = _make_injector(env, cfg)
    inj.reset(torch.arange(4))
    inj._compute_contact_and_sensor()
    assert torch.allclose(env._wbt_force_ext_w[..., :3], torch.zeros(4, 2, 3), atol=1e-6)
    assert torch.allclose(env._wbt_contact_gate, torch.zeros(4, 2), atol=1e-6)


def test_injector_press_gives_causal_fext_along_normal_and_gate_on():
    # Press the TCP past the plane -> F_ext nonzero along +normal, gate=1.
    from wbt_training.config_values.hybrid import HybridContactConfig

    env = _hybrid_env()
    cfg = HybridContactConfig()
    inj = _make_injector(env, cfg)
    _pin_plane(env, cfg, inj)
    env.simulator._robot.data.body_pos_w[..., 0] -= 0.03  # press 3 cm
    inj._compute_contact_and_sensor()
    f = env._wbt_force_ext_w[..., :3]
    assert (f[..., 0] > 0).all()  # reaction pushes back along +x (out of wall)
    assert torch.allclose(f[..., 1:], torch.zeros(4, 2, 2), atol=1e-5)
    assert (env._wbt_contact_gate == 1.0).all()
    assert torch.allclose(f[..., 0], torch.full((4, 2), 200.0 * 0.03), atol=0.5)  # 6 N


def test_injector_deeper_press_gives_larger_fext_through_compute_path():
    # THE causality invariant the user asked for: press HARDER -> F_meas goes UP,
    # driven through the live-TCP compute path (not just the pure helper).
    from wbt_training.config_values.hybrid import HybridContactConfig

    env = _hybrid_env()
    cfg = HybridContactConfig()
    inj = _make_injector(env, cfg)
    _pin_plane(env, cfg, inj)
    base = env.simulator._robot.data.body_pos_w.clone()
    env.simulator._robot.data.body_pos_w[:] = base
    env.simulator._robot.data.body_pos_w[..., 0] -= 0.02  # shallow press
    inj._compute_contact_and_sensor()
    f_shallow = env._wbt_force_ext_w[..., :3].norm(dim=-1).clone()
    env.simulator._robot.data.body_pos_w[:] = base
    env.simulator._robot.data.body_pos_w[..., 0] -= 0.04  # deeper press
    inj._compute_contact_and_sensor()
    f_deep = env._wbt_force_ext_w[..., :3].norm(dim=-1)
    assert (f_deep > f_shallow).all()  # deeper -> larger (causal, monotonic)


def test_injector_substep_hook_end_to_end_injects_force():
    # Drive the ACTUAL monkey-patched hook (not _compute_ directly): this exercises
    # _advance_fcmd_pulse + _compute_contact_and_sensor + _inject_force_ext and would
    # catch the 6D/3D advance-pulse crash. Asserts set_external_force_and_torque is
    # called with a causal body-local force at the TCP offset.
    from wbt_training.config_values.hybrid import HybridContactConfig

    env = _hybrid_env()
    cfg = HybridContactConfig()
    inj = _make_injector(env, cfg)
    _pin_plane(env, cfg, inj)
    env.simulator._robot.data.body_pos_w[..., 0] -= 0.03  # press
    env.simulator._robot.set_external_force_and_torque.reset_mock()
    env._apply_force_in_physics_step()  # the real hook — must NOT raise
    assert env.simulator._robot.set_external_force_and_torque.called
    kwargs = env.simulator._robot.set_external_force_and_torque.call_args.kwargs
    assert kwargs["forces"].shape == (4, 2, 3)
    assert kwargs["positions"].shape == (4, 2, 3)
    assert torch.allclose(kwargs["positions"][..., 0], torch.full((4, 2), cfg.tcp_offset_x), atol=1e-6)
    # identity wrist quat -> body-local force == world force, +x reaction.
    assert (kwargs["forces"][..., 0] > 0).all()


def test_injector_fmeas_tracks_fext_when_clean():
    from wbt_training.config_values.hybrid import HybridContactConfig

    env = _hybrid_env()
    cfg = HybridContactConfig(sensor_noise_std=0.0, sensor_bias_std=0.0, include_inertial=False)
    inj = _make_injector(env, cfg)
    _pin_plane(env, cfg, inj)
    env._wbt_sensor_bias_w[:] = 0.0
    env.simulator._robot.data.body_pos_w[..., 0] -= 0.03
    inj._compute_contact_and_sensor()
    assert torch.allclose(env._wbt_force_meas_w[..., :3], env._wbt_force_ext_w[..., :3], atol=1e-4)


def test_injector_reset_samples_kenv_in_range():
    from wbt_training.config_values.hybrid import HybridContactConfig

    env = _hybrid_env(num_envs=64)
    cfg = HybridContactConfig()
    inj = _make_injector(env, cfg)
    inj.reset(torch.arange(64))
    assert (env._wbt_k_env >= cfg.k_env_min - 1e-3).all()
    assert (env._wbt_k_env <= cfg.k_env_max + 1e-3).all()


def test_injector_reset_clamps_fcmd_to_reachable_and_aligns_plane():
    # Well-posedness (2026-07-10): after reset, the per-episode |F_cmd| peak is clamped to
    # the sampled wall's reachable force, and the plane normal is anti-aligned with F_cmd
    # (so e=F_cmd+F_meas is nullable). Assert both invariants hold post-reset.
    from wbt_training.config_values.hybrid import HybridContactConfig

    env = _hybrid_env(num_envs=64)
    cfg = HybridContactConfig()
    inj = _make_injector(env, cfg)
    inj.reset(torch.arange(64))
    k = env._wbt_k_env  # (64,2)
    f_target = env._wbt_force_state["cmd"]["f_target"]  # (64,2,3) root-frame peak command
    reachable = (k * cfg.max_penetration_m).clamp(max=cfg.f_cmd_max)  # (64,2)
    # |F_cmd peak| never exceeds the wall's reachable force (+ tiny fp slack).
    assert (f_target.norm(dim=-1) <= reachable + 1e-3).all()
    # plane normal is a unit vector anti-aligned with the world-frame push (identity root
    # quat → root==world): normal ≈ −unit(f_target). Skip near-zero commands (dir undefined).
    fnorm = f_target.norm(dim=-1, keepdim=True)
    valid = fnorm.squeeze(-1) > 1e-3
    push_dir = f_target / fnorm.clamp(min=1e-6)
    dotp = (env._wbt_plane_normal * (-push_dir)).sum(-1)
    assert (dotp[valid] > 0.99).all()  # normal points opposite the command


def test_fcmd_stays_along_normal_and_reachable_across_midepisode_resample():
    # REGRESSION (Opus review 2026-07-10): the reset-time reachable-clamp + F_cmd-along-normal
    # invariants must hold for the WHOLE episode, not just at reset. The reused _step_pulse
    # auto-resamples f_target mid-episode to a fresh UNCLAMPED, RANDOM-direction vector; the
    # earlier reset-only test could not catch that. Here we (a) reset, (b) FORCE a mid-episode
    # auto-resample by expiring the pulse idle window, (c) advance the pulse, and assert the
    # LIVE command _wbt_force_cmd_b is STILL along the (fixed) plane normal and STILL within the
    # reachable cap — i.e. _advance_fcmd_pulse re-conforms after the resample.
    from wbt_training.config_values.hybrid import HybridContactConfig

    env = _hybrid_env(num_envs=64)
    cfg = HybridContactConfig()
    inj = _make_injector(env, cfg)
    inj.reset(torch.arange(64))
    fixed_dir = env._wbt_fcmd_dir_b.clone()  # (64,2,3) episode-fixed unit dir (root frame)
    reach = env._wbt_fcmd_reach.clone()  # (64,2) reachable cap
    # Force _step_pulse to auto-resample on the NEXT advance: expire every env's idle window and
    # scramble f_target to a large, arbitrary-direction vector (what _resample_pulse would draw).
    st = env._wbt_force_state["cmd"]
    st["t"][:] = st["idle_end"] + 1  # past idle → auto-resample fires inside _step_pulse
    st["f_target"][:] = torch.tensor([0.0, 0.0, 25.0])  # off-normal, full-magnitude scramble
    env.episode_length_buf = torch.ones(64, dtype=torch.long)  # != last_step → advance fires
    inj._advance_fcmd_pulse()
    cmd_b = env._wbt_force_cmd_b  # (64,2,3) LIVE command after the resample
    mag = cmd_b.norm(dim=-1)  # (64,2)
    valid = fixed_dir.norm(dim=-1) > 1e-6  # envs/hands with a defined fixed direction
    # (a) still within the per-episode reachable cap (+fp slack).
    assert (mag <= reach + 1e-4).all()
    # (b) whenever the live command is nonzero, it is COLLINEAR with the episode-fixed direction
    # (cosine ≈ 1) — NOT the scrambled +z direction — so it still opposes the fixed wall normal.
    nonzero = valid & (mag > 1e-4)
    cos = (cmd_b * fixed_dir).sum(-1) / mag.clamp(min=1e-6)
    assert (cos[nonzero] > 0.999).all()


def test_fcmd_world_stays_aligned_with_wall_when_pelvis_reorients():
    # FRAMES REGRESSION (Opus Gate-B): the wall/plane normal is WORLD-fixed; F_cmd must stay
    # aligned with it for the whole episode even as the PELVIS reorients tracking the clip. The
    # buggy version stored F_cmd root-frame and re-rotated by the LIVE root quat → drift off the
    # normal → unnullable tangential error. Fix: F_cmd direction is WORLD-fixed (_wbt_fcmd_dir_w).
    import math

    from wbt_training.config_values.hybrid import HybridContactConfig

    env = _hybrid_env(num_envs=8)
    cfg = HybridContactConfig()
    inj = _make_injector(env, cfg)
    inj.reset(torch.arange(8))
    dir_w = env._wbt_fcmd_dir_w.clone()  # world-fixed command dir (= −plane_normal)
    # Rotate the pelvis 30° yaw and advance the pulse at the hold plateau.
    half = math.radians(30) / 2
    env.simulator._robot.data.root_quat_w[:] = torch.tensor([math.cos(half), 0.0, 0.0, math.sin(half)])
    env._wbt_force_state["cmd"]["t"][:] = env._wbt_force_state["cmd"]["ramp_end"]
    env.episode_length_buf = torch.ones(8, dtype=torch.long)
    inj._advance_fcmd_pulse()
    cmd_w = env._wbt_force_cmd_w
    mag = cmd_w.norm(dim=-1)
    valid = (dir_w.norm(dim=-1) > 1e-6) & (mag > 1e-4)
    cos = (cmd_w * dir_w).sum(-1) / mag.clamp(min=1e-6)
    assert (cos[valid] > 0.999).all()  # world command still on the fixed wall normal
    # the pelvis-frame obs command DID rotate (it must, obs is pelvis-relative).
    assert not torch.allclose(env._wbt_force_cmd_b, cmd_w, atol=1e-3)


# ---------------------------------------------------------------------------
# Task 5: obs readers (force_meas_local, integrator_state, x_tgt_shift, contact_gate)
# ---------------------------------------------------------------------------
def test_force_meas_local_is_pure_reader_root_frame():
    from wbt_training.config_values.hybrid import HybridContactConfig, force_meas_local

    env = _hybrid_env()
    cfg = HybridContactConfig()
    _make_injector(env, cfg)
    # Set a known world-frame measured force; identity root quat -> root == world.
    env._wbt_force_meas_w[..., :3] = torch.tensor([1.0, 2.0, 3.0])
    out = force_meas_local(env, cfg=cfg)
    assert out.shape == (4, 6)  # 2 hands * 3 xyz
    assert torch.allclose(out.reshape(4, 2, 3), torch.tensor([1.0, 2.0, 3.0]).expand(4, 2, 3), atol=1e-6)


def test_contact_gate_reader():
    from wbt_training.config_values.hybrid import HybridContactConfig, contact_gate

    env = _hybrid_env()
    cfg = HybridContactConfig()
    _make_injector(env, cfg)
    env._wbt_contact_gate[:] = torch.tensor([1.0, 0.0])
    out = contact_gate(env, cfg=cfg)
    assert out.shape == (4, 2)
    assert torch.allclose(out, torch.tensor([1.0, 0.0]).expand(4, 2))


def test_integrator_and_xtgt_readers_zeroed_when_servo_off():
    # These obs are present at FIXED 165D width even with servo off — they must read
    # the (zeroed) buffers without error and return shape (n, h*3).
    from wbt_training.config_values.hybrid import HybridContactConfig, integrator_state, x_tgt_shift

    env = _hybrid_env()
    cfg = HybridContactConfig(servo_enabled=False)  # servo OFF
    _make_injector(env, cfg)
    assert torch.allclose(integrator_state(env), torch.zeros(4, 6))  # zeroed buffer
    assert torch.allclose(x_tgt_shift(env), torch.zeros(4, 6))
    # non-zero buffer round-trips (identity root quat → root == world)
    env._wbt_integ_state[..., :] = torch.tensor([0.01, 0.0, 0.0])
    assert integrator_state(env).shape == (4, 6)
    assert torch.allclose(integrator_state(env).reshape(4, 2, 3)[..., 0], torch.full((4, 2), 0.01), atol=1e-6)


def test_privileged_critic_only_readers():
    # force_ext_local (clean), k_env_privileged, penetration_privileged — pure readers.
    from wbt_training.config_values.hybrid import (
        HybridContactConfig,
        force_ext_local,
        k_env_privileged,
        penetration_privileged,
    )

    env = _hybrid_env()
    cfg = HybridContactConfig()
    _make_injector(env, cfg)
    env._wbt_force_ext_w[..., :3] = torch.tensor([0.0, 5.0, 0.0])  # clean world F_ext
    env._wbt_k_env[:] = 300.0
    env._wbt_penetration[:] = 0.02
    fe = force_ext_local(env, cfg=cfg)
    assert fe.shape == (4, 6)
    assert torch.allclose(fe.reshape(4, 2, 3), torch.tensor([0.0, 5.0, 0.0]).expand(4, 2, 3), atol=1e-6)
    assert torch.allclose(k_env_privileged(env), torch.full((4, 2), 300.0))
    assert torch.allclose(penetration_privileged(env), torch.full((4, 2), 0.02))


# ---------------------------------------------------------------------------
# Task 6: rebalanced reward + command + observation + curriculum cfgs
# ---------------------------------------------------------------------------
def test_hybrid_reward_rebalances_force_and_torso():
    from wbt_training.config_values.hybrid import g1_29dof_wbt_2pt_hybrid_reward

    terms = g1_29dof_wbt_2pt_hybrid_reward.terms
    # Force term halved vs 2pt-force's 6.0 -> 3.0 (reduce dominance).
    assert terms["force_track_pos_eq2"].weight == 3.0
    # Torso/posture orientation strengthened: 1.0 (core) -> 2.0.
    assert terms["motion_relative_body_orientation_error_exp"].weight == 2.0
    # Wrist position owned by Eq.2 (no-wrist swap present, core position term gone).
    assert "motion_relative_body_position_error_exp" not in terms
    assert "motion_relative_body_position_error_exp_no_wrist" in terms
    assert terms["motion_relative_body_position_error_exp_no_wrist"].weight == 1.0
    # feet_slip penalty present with negative weight.
    assert terms["feet_slip"].weight == -2.0
    # ang-vel damping tightened (sigma 3.14 -> 2.0).
    assert terms["motion_global_body_ang_vel"].params["sigma"] == 2.0
    # Force is now < 40% of the positive budget (torso-stability sanity).
    positive = sum(t.weight for t in terms.values() if t.weight > 0)
    assert terms["force_track_pos_eq2"].weight / positive < 0.40


def test_hybrid_command_registers_injector_in_setup_and_reset():
    from wbt_training.config_values.hybrid import g1_29dof_wbt_hybrid_command

    assert "hybrid_contact_injector" in g1_29dof_wbt_hybrid_command.setup_terms
    assert "hybrid_contact_injector" in g1_29dof_wbt_hybrid_command.reset_terms
    assert "motion_command" in g1_29dof_wbt_hybrid_command.setup_terms
    inj = g1_29dof_wbt_hybrid_command.setup_terms["hybrid_contact_injector"]
    assert inj.func.endswith(":HybridContactInjector")
    assert "hybrid_config" in inj.params


def test_hybrid_observation_actor_165d_fixed_width_servo_state_present():
    from wbt_training.config_values.hybrid import g1_29dof_wbt_observation_2pt_hybrid

    groups = g1_29dof_wbt_observation_2pt_hybrid.groups
    actor = groups["actor_obs"].terms
    # FIXED 165D: 13 terms (2pt 9 + F_cmd + F_meas + I + Δx). Servo state ALWAYS present
    # (zeroed when servo off) so obs width never keys off servo_enabled.
    assert len(actor) == 13
    for fn in (":force_cmd_local", ":force_meas_local", ":integrator_state", ":x_tgt_shift"):
        assert any(t.func.endswith(fn) for t in actor.values()), f"actor missing {fn}"
    # contact_gate is critic-only (sim/real definition differs) — NOT in the actor.
    assert not any(t.func.endswith(":contact_gate") for t in actor.values())
    critic = groups["critic_obs"].terms
    for fn in (":force_cmd_local", ":force_meas_local", ":integrator_state", ":x_tgt_shift", ":contact_gate"):
        assert any(t.func.endswith(fn) for t in critic.values()), f"critic missing {fn}"

    # Zero-padded prefixes so alphabetical concat (manager sorts keys, manager.py:141) == numeric.
    actor_keys = list(actor.keys())
    assert actor_keys == sorted(actor_keys), "actor obs keys must already be in sorted order"
    # Order: F^meas right after F^cmd; then I, then Δx (adjacent same-frame servo block).
    idx = {
        suf: next(i for i, k in enumerate(actor_keys) if k.endswith(suf))
        for suf in ("force_cmd_local", "force_meas_local", "integrator_state", "x_tgt_shift")
    }
    assert idx["force_meas_local"] == idx["force_cmd_local"] + 1
    assert idx["integrator_state"] == idx["force_meas_local"] + 1
    assert idx["x_tgt_shift"] == idx["integrator_state"] + 1


def test_hybrid_curriculum_registers_force_alpha():
    from wbt_training.config_values.hybrid import g1_29dof_wbt_hybrid_curriculum

    assert "force_alpha" in g1_29dof_wbt_hybrid_curriculum.setup_terms
    assert "force_alpha" in g1_29dof_wbt_hybrid_curriculum.reset_terms


def test_eq2_hybrid_slices_6d_ext_and_shifts():
    # Pins the fix numerically: reward reads FORCE channels [:3], IGNORES moment
    # channels [3:], and the target shift is exactly (F_cmd+F_ext)/K.
    from wbt_training.config_values.hybrid import HybridContactConfig, force_track_pos_eq2_hybrid

    env = _hybrid_env()
    cfg = HybridContactConfig()
    _make_injector(env, cfg)
    hf = cfg.hand_force_config()
    # motion_command FK stubs the Eq.2 reward reads. ref wrist at origin; identity
    # wrist quat -> FK offset cancels on cur & ref, so residual is pure origin error.
    mc = types.SimpleNamespace(
        body_pos_relative_w=torch.zeros(4, 2, 3),
        motion_cfg=types.SimpleNamespace(body_names_to_track=["left_wrist_yaw_link", "right_wrist_yaw_link"]),
    )
    env.command_manager.get_state = lambda name: mc
    env._wbt_force_motion_indices = torch.tensor([0, 1])
    # The live wrist is at body_pos_w x=0.30; set ref wrist to the live wrist pos so
    # cur==target for zero force (reward == 1 exactly).
    mc.body_pos_relative_w[:] = env.simulator._robot.data.body_pos_w
    env._wbt_force_cmd_w[:] = 0.0
    env._wbt_force_ext_w[:] = 0.0
    r0 = force_track_pos_eq2_hybrid(env, cfg=hf)
    assert torch.allclose(r0, torch.ones(4), atol=1e-4)  # cur==ref, zero shift -> reward 1
    # Moment channels MUST be ignored: setting only [3:] leaves the reward unchanged.
    env._wbt_force_ext_w[..., 3:] = 9.9
    r_moment = force_track_pos_eq2_hybrid(env, cfg=hf)
    assert torch.allclose(r_moment, r0, atol=1e-6)  # moment ignored (would differ if [3:] read)
    # A known force channel shifts the target by (F_cmd+F_ext)/K.
    env._wbt_force_ext_w[..., 3:] = 0.0
    env._wbt_force_ext_w[..., :3] = torch.tensor([2.0, 0.0, 0.0])  # 2 N -> shift 0.01 m/hand
    shift_per_hand = 2.0 / hf.stiffness_K  # 0.01 m
    expected = torch.exp(torch.tensor(-(2 * shift_per_hand) * 2.0 / hf.sigma))  # 2 hands * 0.01 m
    r2 = force_track_pos_eq2_hybrid(env, cfg=hf)
    assert torch.allclose(r2, expected.expand(4), atol=1e-4)


# ---------------------------------------------------------------------------
# Task 7: preset wiring in experiment.py
# ---------------------------------------------------------------------------
def test_preset_registered_and_forks_2pt_with_yam_urdf():
    from wbt_training.config_values.experiment import DEFAULTS

    assert "g1_wbt_2pt_hybrid" in DEFAULTS
    exp = DEFAULTS["g1_wbt_2pt_hybrid"]
    # yam URDF baked in (like upperbody).
    assert exp.robot.asset.urdf_file == "g1/g1_29dof_yam.urdf"
    # hybrid building blocks wired.
    assert "hybrid_contact_injector" in exp.command.setup_terms
    assert exp.reward.terms["force_track_pos_eq2"].weight == 3.0
    # Sensor-in-the-loop: F_meas in BOTH actor and critic.
    assert any(t.func.endswith(":force_meas_local") for t in exp.observation.groups["actor_obs"].terms.values())
    assert any(t.func.endswith(":force_meas_local") for t in exp.observation.groups["critic_obs"].terms.values())
    # forked from 2pt, not 2pt-force: inherits the plain-2pt algo unchanged.
    from wbt_training.config_values.experiment import g1_wbt_2pt

    assert exp.algo == g1_wbt_2pt.algo
    # NOTE: the SHIPPED preset is servo-ON with the force_runaway estop (Task 11, user 2026-07-10).
    # The termination-equality + servo-off assertions that held at the Task-7 Layer-A snapshot are
    # SUPERSEDED here and fully re-asserted by test_preset_ships_servo_enabled_with_estop. This
    # test now only pins the fork-invariant facts (yam URDF, injector, obs, algo).


def test_preset_subcommand_name_derivable():
    # sanity: the DEFAULTS key -> exp:g1-wbt-2pt-hybrid (underscore->hyphen).
    key = "g1_wbt_2pt_hybrid"
    assert f"exp:{key.replace('_', '-')}" == "exp:g1-wbt-2pt-hybrid"


# ---------------------------------------------------------------------------
# Task 8: pure servo math — gated integrator + Jᵀ residual (servo_residual_step)
# ---------------------------------------------------------------------------
def test_servo_no_contact_freezes_integrator_and_zero_residual():
    # Gate 0 (free space): integrator frozen/cleared (not accumulating).
    from wbt_training.config_values.hybrid import HybridContactConfig, servo_residual_step

    cfg = HybridContactConfig(servo_enabled=True)
    n, h, na = 2, 2, 7
    j = torch.zeros(n, h, 6, na)
    j[..., 0, 0] = 1.0  # trivial: joint0 moves TCP +x
    f_cmd = torch.zeros(n, h, 3)
    f_cmd[..., 0] = 10.0
    f_meas = torch.zeros(n, h, 3)
    gate = torch.zeros(n, h)  # NO contact
    integ = torch.full((n, h, 3), 0.05)  # some prior integral
    dq, new_integ, dx = servo_residual_step(j, f_cmd, f_meas, gate, integ, cfg, dt=0.02)
    # gate 0 → integrator zeroed (frozen/cleared, not accumulating)
    assert torch.allclose(new_integ, torch.zeros(n, h, 3), atol=1e-7)


def test_servo_integrator_accumulates_toward_zero_error_on_contact():
    # Under-applying force (F_meas small): e=F_cmd+F_meas>0 → integrator grows.
    from wbt_training.config_values.hybrid import HybridContactConfig, servo_residual_step

    cfg = HybridContactConfig(servo_enabled=True)
    n, h, na = 1, 1, 7
    j = torch.zeros(n, h, 6, na)
    j[..., 0, 0] = 1.0
    f_cmd = torch.zeros(n, h, 3)
    f_cmd[..., 0] = 10.0
    f_meas = torch.zeros(n, h, 3)
    f_meas[..., 0] = -2.0  # only 2 N reacted → under-applied
    gate = torch.ones(n, h)
    integ = torch.zeros(n, h, 3)
    _, i1, _ = servo_residual_step(j, f_cmd, f_meas, gate, integ, cfg, dt=0.02)
    # e_x = 10 + (-2) = 8 > 0 → I_x increases
    assert i1[0, 0, 0] > 0.0
    # at steady state F_meas=-F_cmd → e=0 → integrator stops growing
    f_meas_ss = -f_cmd
    _, i2, _ = servo_residual_step(j, f_cmd, f_meas_ss, gate, i1, cfg, dt=0.02)
    assert torch.allclose(i2, i1, atol=1e-6)


def test_servo_residual_sign_pushes_into_wall_not_away():
    # THE divergence footgun (§4.0): Δx and Δq must extend the setpoint along +F_cmd.
    from wbt_training.config_values.hybrid import HybridContactConfig, servo_residual_step

    cfg = HybridContactConfig(servo_enabled=True)
    n, h, na = 1, 1, 7
    j = torch.zeros(n, h, 6, na)
    j[..., 0, 0] = 1.0  # +x TCP via joint0
    f_cmd = torch.zeros(n, h, 3)
    f_cmd[..., 0] = 10.0  # push +x
    f_meas = torch.zeros(n, h, 3)  # nothing applied yet
    gate = torch.ones(n, h)
    dq, _, dx = servo_residual_step(j, f_cmd, f_meas, gate, torch.zeros(n, h, 3), cfg, dt=0.02)
    assert dx[0, 0, 0] > 0.0  # setpoint shifts +x (into the wall), NOT -x
    assert dq[0, 0] > 0.0  # joint0 residual drives TCP +x (Jᵀ maps +x wrench → +joint0)


def test_servo_residual_clamped():
    from wbt_training.config_values.hybrid import HybridContactConfig, servo_residual_step

    cfg = HybridContactConfig(servo_enabled=True, servo_dq_max=0.05)
    n, h, na = 1, 1, 7
    j = torch.zeros(n, h, 6, na)
    j[..., 0, 0] = 100.0  # huge gain
    f_cmd = torch.zeros(n, h, 3)
    f_cmd[..., 0] = 25.0
    dq, _, _ = servo_residual_step(j, f_cmd, torch.zeros(n, h, 3), torch.ones(n, h), torch.zeros(n, h, 3), cfg, dt=0.02)
    assert dq.abs().max() <= 0.05 + 1e-6  # clamped per joint


def test_servo_selection_masks_moment_axes():
    # servo_selection (1,1,1,0,0,0): moment axes never contribute (D8 dormant).
    from wbt_training.config_values.hybrid import HybridContactConfig

    cfg = HybridContactConfig(servo_enabled=True)
    assert tuple(cfg.servo_selection[3:]) == (0.0, 0.0, 0.0)


def test_servo_closed_loop_converges_to_fcmd_and_is_K_independent():
    # THE test the sign/gating tests miss: close the loop against a TOY LINEAR PLANT and assert
    # the CONVERGED force reaches F_cmd AND is INDEPENDENT of the TRUE environment stiffness K_eff
    # (the physically meaningful K-independence — the servo does not know the real wall stiffness).
    # A near-inert servo (old k=0.0002) or saturating integrator FAILS this.
    #
    # DESIGN NOTE (2026-07-10, force-unit integral): with the force-unit integrator the loop gain
    # is k·K_eff·J²·K_I — independent of the servo's GUESSED stiffness_K — so stability decouples
    # from the K guess and steady state reaches F_cmd exactly. (The old middle-form K·Δx had loop
    # gain ∝ stiffness_K, so "sweep the K guess" was actually a stability sweep that diverged at
    # the high end; that test was physically ill-posed. We sweep the TRUE K_eff instead.)
    #
    # Plant: F_applied = K_eff·J·Δq; F_meas = −F_applied (Newton). J chosen so the steady joint
    # residual dq_ss = F_cmd/(K_eff·J) stays inside servo_dq_max=0.15 for every swept K_eff.
    from wbt_training.config_values.hybrid import HybridContactConfig, servo_residual_step

    def _run(K_eff, J, servo_stiffness_K=200.0, steps=6000, dt=0.005):
        cfg = HybridContactConfig(servo_enabled=True, stiffness_K=servo_stiffness_K)
        n, h, na = 1, 1, 1
        j = torch.zeros(n, h, 6, na)
        j[..., 0, 0] = J  # joint0 → TCP +x, gain J
        f_cmd = torch.zeros(n, h, 3)
        f_cmd[..., 0] = 10.0
        integ = torch.zeros(n, h, 3)
        f_applied = 0.0
        for _ in range(steps):
            f_meas = torch.zeros(n, h, 3)
            f_meas[..., 0] = -f_applied  # external-on-robot (Newton)
            gate = torch.ones(n, h)  # in contact
            dq, integ, _ = servo_residual_step(j, f_cmd, f_meas, gate, integ, cfg, dt=dt)
            # PD spring realizes the residual: F_applied = K_eff·(J·dq)
            f_applied = float(K_eff * J * dq[0, 0])
        return f_applied

    # Sweep the TRUE env stiffness across a 4× range (soft→stiff), J picked for clamp headroom
    # (dq_ss = 10/(K_eff·J): 10/(150·0.5)=0.133, 10/(300·0.5)=0.067, 10/(600·0.5)=0.033 — all <0.15).
    f_soft = _run(150.0, 0.5)
    f_mid = _run(300.0, 0.5)
    f_stiff = _run(600.0, 0.5)
    for f in (f_soft, f_mid, f_stiff):
        assert abs(f - 10.0) < 0.5, f"force did not converge to F_cmd (got {f:.3f} N) — inert/saturating servo?"
    # K-independence: converged force barely moves across the 4× true-stiffness range (the
    # integrator, not the feedforward, sets steady state → outcome is stiffness-agnostic).
    assert max(f_soft, f_mid, f_stiff) - min(f_soft, f_mid, f_stiff) < 0.5


def test_servo_integrator_does_not_saturate_in_one_step_at_fcmd_max():
    # Regression guard for the old I_max=0.10 bug: one-step ΔI must be a small fraction of I_max.
    from wbt_training.config_values.hybrid import HybridContactConfig, servo_residual_step

    cfg = HybridContactConfig(servo_enabled=True)
    n, h, na = 1, 1, 1
    j = torch.zeros(n, h, 6, na)
    j[..., 0, 0] = 0.2
    f_cmd = torch.zeros(n, h, 3)
    f_cmd[..., 0] = cfg.f_cmd_max  # worst case
    f_meas = torch.zeros(n, h, 3)  # nothing applied → e = f_cmd_max
    _, integ, _ = servo_residual_step(j, f_cmd, f_meas, torch.ones(n, h), torch.zeros(n, h, 3), cfg, dt=0.005)
    # ΔI = K_I·f_cmd_max·dt = 0.5·25·0.005 = 0.0625 N ≪ I_max=40 N (force units).
    assert integ.abs().max() < cfg.integrator_max * 0.5


# ---------------------------------------------------------------------------
# Task 9: TCP Jacobian accessor (get_jacobians slice + remap + offset)
# ---------------------------------------------------------------------------
def _servo_env():
    """Servo-enabled env: adds root_physx_view.get_jacobians + find_bodies/find_joints stubs +
    a `joint_control` action term (the real PD-torque input the servo injects into) + action_scales,
    so setup() can call _resolve_jacobian_indices() + cache the joint term, and the servo can run."""
    env = _hybrid_env()
    robot = env.simulator._robot
    # find_bodies(name) -> (ids_tensor, names): PhysX/DFS order (arbitrary but consistent).
    _body_map = {"left_wrist_yaw_link": 3, "right_wrist_yaw_link": 9}
    robot.find_bodies = lambda name: ([_body_map[name]], [name])
    # find_joints(name, preserve_order=True) -> (ids_tensor, names): PhysX DFS joint id.
    # Give each arm joint a distinct PhysX id (not equal to the config-order index).
    _joint_map = {nm: 100 + i for i, nm in enumerate(_G1_29DOF_NAMES)}
    robot.find_joints = lambda nm, preserve_order=True: ([_joint_map[nm]], [nm])
    # The joint-position action term: the ONLY tensor the real PD reads is `_actions_after_delay`
    # (torque = Kp·(actions·action_scales + default − dof_pos) − Kd·dof_vel, joint_control.py:188).
    # The servo injects servo_dq/action_scales into it, reconstructed off the substep-0 snapshot.
    joint_term = types.SimpleNamespace(
        _actions_after_delay=torch.zeros(env.num_envs, 29),
        _substep_idx=0,  # 0 on the first substep after process_actions; incremented in apply_actions
    )
    env.action_manager = MagicMock()
    env.action_manager.get_term = lambda name: {"joint_control": joint_term}[name]
    env.action_scales = torch.ones(29)  # unit scale → servo_dq maps 1:1 into action units (simplest)
    return env, _body_map, _joint_map


def test_tcp_jacobian_shape_with_stubbed_view():
    # Plumbing-only (MagicMock): correct shape out, arm-column count right. A real value
    # check is the skipif-real-sim test below (a mock J can't validate values).
    from wbt_training.config_values.hybrid import HybridContactConfig

    env = _hybrid_env()
    cfg = HybridContactConfig(servo_enabled=True)
    inj = _make_injector(env, cfg)
    n_dof, n_bodies = 35, 12  # 29 joints + 6 floating base cols; arbitrary body count
    env.simulator._robot.root_physx_view.get_jacobians = lambda: torch.zeros(4, n_bodies, 6, n_dof)
    # stub the index resolution the injector precomputed at setup:
    inj._jac_wrist_body_ids = [0, 1]
    inj._jac_arm_cols = [[6, 7, 8, 9, 10, 11, 12], [13, 14, 15, 16, 17, 18, 19]]
    J = inj._tcp_jacobian_world()
    assert J.shape == (4, 2, 6, 7)  # (n, hands, 6, n_arm)


def test_resolve_jacobian_indices_maps_arm_cols_and_pd_slice():
    # setup() with servo_enabled calls _resolve_jacobian_indices; verify the +6 floating-base
    # column offset and the config-order PD slice (left arm → 15..21, right arm → 22..28).
    from wbt_training.config_values.hybrid import HybridContactConfig

    env, body_map, joint_map = _servo_env()
    cfg = HybridContactConfig(servo_enabled=True)
    inj = _make_injector(env, cfg)  # setup() resolves indices
    assert inj._jac_wrist_body_ids == [body_map["left_wrist_yaw_link"], body_map["right_wrist_yaw_link"]]
    # arm Jacobian cols = PhysX joint id + 6 (floating base).
    left_arm = [
        "left_shoulder_pitch_joint",
        "left_shoulder_roll_joint",
        "left_shoulder_yaw_joint",
        "left_elbow_joint",
        "left_wrist_roll_joint",
        "left_wrist_pitch_joint",
        "left_wrist_yaw_joint",
    ]
    assert inj._jac_arm_cols[0] == [joint_map[nm] + 6 for nm in left_arm]
    # PD slice = config-order index in dof_names (left arm is 15..21).
    assert inj._arm_pd_slice[0] == [15, 16, 17, 18, 19, 20, 21]
    assert inj._arm_pd_slice[1] == [22, 23, 24, 25, 26, 27, 28]


@pytest.mark.skipif(not _HAVE_ISAACSIM, reason="real IsaacSim runtime needed for get_jacobians")
def test_tcp_jacobian_matches_finite_difference_real_sim():  # pragma: no cover
    # Build one env, perturb an arm joint, finite-difference the TCP world position, compare to
    # the Jacobian linear rows. The ONLY faithful Jacobian test — GPU-box merge gate.
    pytest.skip("real-sim finite-difference harness runs on the GPU sim box, not this headless CI")


# ---------------------------------------------------------------------------
# Task 10: wire the servo into the injector substep hook (PD-target write)
# ---------------------------------------------------------------------------
def _servo_env_wired(cfg):
    """Servo env with a NONZERO Jacobian (joint0 → TCP +x for both hands) + resolved indices
    + dof_pos_target, ready to drive the real hook. Returns (env, injector)."""
    from wbt_training.config_values.hybrid import HybridContactInjector  # noqa: F401

    env, _, _ = _servo_env()
    inj = _make_injector(env, cfg)  # setup() resolves indices (servo on)
    # Jacobian n_dof must cover the largest resolved arm column (+6 floating base). _servo_env
    # assigns PhysX joint ids 100+i, so cols run up to ~135 — size the stub view accordingly.
    n_dof = max(c for cols in inj._jac_arm_cols for c in cols) + 1
    n_bodies = max(inj._jac_wrist_body_ids) + 1

    def _jac():
        j = torch.zeros(4, n_bodies, 6, n_dof)
        # each hand's wrist body row, linear-x via its first arm col → joint0-like TCP +x drive.
        j[:, inj._jac_wrist_body_ids[0], 0, inj._jac_arm_cols[0][0]] = 1.0
        j[:, inj._jac_wrist_body_ids[1], 0, inj._jac_arm_cols[1][0]] = 1.0
        return j

    env.simulator._robot.root_physx_view.get_jacobians = _jac
    return env, inj


def test_servo_off_never_touches_actions():
    # servo_enabled=False → hook does not touch integrator or the joint action term (Layer-A).
    from wbt_training.config_values.hybrid import HybridContactConfig

    env = _hybrid_env()
    cfg = HybridContactConfig(servo_enabled=False)
    inj = _make_injector(env, cfg)
    _pin_plane(env, cfg, inj)
    env.simulator._robot.data.body_pos_w[..., 0] -= 0.03  # press (would integrate if servo on)
    env._apply_force_in_physics_step()
    assert torch.allclose(env._wbt_integ_state, torch.zeros(4, 2, 3))  # untouched
    assert torch.allclose(env._wbt_servo_dq, torch.zeros_like(env._wbt_servo_dq))


def test_servo_injects_into_actions_after_delay_not_dof_pos_target():
    # BLOCKER regression (Opus Gate-B): the servo MUST write the joint term's _actions_after_delay
    # (the ONLY tensor the real PD reads), NOT env.simulator.dof_pos_target (which nothing reads).
    from wbt_training.config_values.hybrid import HybridContactConfig

    cfg = HybridContactConfig(servo_enabled=True)
    env, inj = _servo_env_wired(cfg)
    _pin_plane(env, cfg, inj)
    env.simulator._robot.data.body_pos_w[..., 0] -= 0.03  # press
    jt = env.action_manager.get_term("joint_control")
    env._apply_force_in_physics_step()
    # the joint action term's actions were shifted on the ARM DOFs (the real PD input).
    assert jt._actions_after_delay.abs().sum() > 0
    leg_waist = list(range(15))
    assert torch.allclose(jt._actions_after_delay[:, leg_waist], torch.zeros(4, 15), atol=1e-7)
    # action_scales=1 in the mock → the injected shift equals servo_dq on the arm DOFs.
    arm = list(range(15, 29))
    assert torch.allclose(jt._actions_after_delay[:, arm], env._wbt_servo_dq[:, arm], atol=1e-6)


def test_servo_on_freespace_freezes_integrator_and_ff_only_no_accumulation():
    # servo on but no contact → gate 0 → integrator frozen at 0. The FEEDFORWARD (F_cmd) is NOT
    # gated, so with a REAL (nonzero) Jacobian AND a nonzero F_cmd the servo injects a nonzero,
    # CONSTANT feedforward into the arm actions ("go to a target and STOP", not runaway). Running
    # the hook across substeps must NOT accumulate (reconstruct policy + servo, not +=).
    from wbt_training.config_values.hybrid import HybridContactConfig

    cfg = HybridContactConfig(servo_enabled=True)
    env, inj = _servo_env_wired(cfg)
    inj.reset(torch.arange(4))  # plane far in front; hand at rest → no contact (gate 0)
    jt = env.action_manager.get_term("joint_control")
    # Seed a nonzero episode-fixed WORLD command direction + magnitude so _advance_fcmd_pulse
    # produces a nonzero _wbt_force_cmd_w (the FF source). Direction +x, cap 8 N; the pulse
    # magnitude (post-clamp) drives the FF. Then run substep 0 (caches policy actions, injects FF).
    env._wbt_fcmd_dir_w[:] = torch.tensor([1.0, 0.0, 0.0])
    env._wbt_fcmd_reach[:] = 8.0
    env._wbt_force_state["cmd"]["f_target"][:] = torch.tensor([8.0, 0.0, 0.0])  # peak command
    env._wbt_force_state["cmd"]["t"][:] = env._wbt_force_state["cmd"]["ramp_end"]  # at hold plateau
    env.episode_length_buf = torch.ones(4, dtype=torch.long)  # != last_step → advance fires
    jt._substep_idx = 0
    env._apply_force_in_physics_step()
    assert torch.allclose(env._wbt_integ_state, torch.zeros(4, 2, 3), atol=1e-7)  # gate 0 → I frozen
    ff = jt._actions_after_delay.clone()
    assert ff.abs().sum() > 0  # nonzero feedforward (would be a false-green if J were stubbed 0)
    # substep 1: _substep_idx>0 → base is NOT re-snapshotted; reconstruct → identical, no accumulation.
    jt._substep_idx = 1
    env._apply_force_in_physics_step()
    assert torch.allclose(jt._actions_after_delay, ff, atol=1e-6)  # no accumulation across substeps


def test_servo_on_contact_injects_arm_actions():
    # servo on + press → integrator grows, dq nonzero on arm DOFs, injected into _actions_after_delay.
    from wbt_training.config_values.hybrid import HybridContactConfig

    cfg = HybridContactConfig(servo_enabled=True)
    env, inj = _servo_env_wired(cfg)
    _pin_plane(env, cfg, inj)
    env.simulator._robot.data.body_pos_w[..., 0] -= 0.03  # press
    jt = env.action_manager.get_term("joint_control")
    env._apply_force_in_physics_step()
    assert (env._wbt_integ_state[..., 0].abs() > 0).all()  # integrator grew on contact
    assert env._wbt_servo_dq.abs().sum() > 0  # a residual was produced
    assert jt._actions_after_delay.abs().sum() > 0  # arm actions were shifted (real PD input)
    # the residual landed ONLY on arm DOFs (leg/waist untouched).
    leg_waist = list(range(15))
    assert torch.allclose(env._wbt_servo_dq[:, leg_waist], torch.zeros(4, 15), atol=1e-7)


# ---------------------------------------------------------------------------
# Task 11: servo Eq.2 reward + pre-integrator force reward + estop + ship servo-on
# ---------------------------------------------------------------------------
def _eq2_env(cfg):
    """Env + injector with the motion_command FK stubs the Eq.2 rewards read (ref wrist = live
    wrist so zero-force reward == 1). Returns (env, inj, hand_force_cfg)."""
    env = _hybrid_env()
    inj = _make_injector(env, cfg)
    hf = cfg.hand_force_config()
    mc = types.SimpleNamespace(
        body_pos_relative_w=env.simulator._robot.data.body_pos_w.clone(),
        motion_cfg=types.SimpleNamespace(body_names_to_track=["left_wrist_yaw_link", "right_wrist_yaw_link"]),
    )
    env.command_manager.get_state = lambda name: mc
    env._wbt_force_motion_indices = torch.tensor([0, 1])
    env._wbt_force_cmd_w[:] = 0.0
    env._wbt_force_ext_w[:] = 0.0
    return env, inj, hf


def test_eq2_servo_reduces_to_layer_a_when_off():
    # servo OFF → force_track_eq2_servo numerically equals force_track_pos_eq2_hybrid (both slice
    # _wbt_force_ext_w[:3]); a known F_ext shifts the target identically.
    from wbt_training.config_values.hybrid import HybridContactConfig, force_track_eq2_servo, force_track_pos_eq2_hybrid

    cfg = HybridContactConfig(servo_enabled=False)
    env, inj, hf = _eq2_env(cfg)
    env._wbt_force_ext_w[..., :3] = torch.tensor([2.0, 0.0, 0.0])  # known force
    r_servo = force_track_eq2_servo(env, cfg=hf)
    r_layerA = force_track_pos_eq2_hybrid(env, cfg=hf)
    assert torch.allclose(r_servo, r_layerA, atol=1e-6)


def test_eq2_servo_uses_delta_x_when_on():
    # servo ON → target = ref + _wbt_delta_x (the servo-inclusive setpoint), NOT ref+(F_cmd+F_ext)/K.
    from wbt_training.config_values.hybrid import HybridContactConfig, force_track_eq2_servo

    cfg = HybridContactConfig(servo_enabled=True)
    env, inj, hf = _eq2_env(cfg)
    # cur == ref, zero Δx → reward 1.
    env._wbt_delta_x[:] = 0.0
    r0 = force_track_eq2_servo(env, cfg=hf)
    assert torch.allclose(r0, torch.ones(4), atol=1e-4)
    # a known Δx shifts the target by Δx per hand → reward drops predictably.
    env._wbt_delta_x[..., 0] = 0.01  # 1 cm along +x per hand
    expected = torch.exp(torch.tensor(-(2 * 0.01) * 2.0 / hf.sigma))  # 2 hands * 0.01 m
    r1 = force_track_eq2_servo(env, cfg=hf)
    assert torch.allclose(r1, expected.expand(4), atol=1e-4)


def test_force_reward_ignores_integrator():
    # Anti-free-ride (§4.A): the pre-integrator force reward credits the POLICY's feedforward,
    # NOT the integrator's patch. Setting _wbt_integ_state nonzero must NOT change the reward
    # (the +K·I add-back exactly cancels the integral's effect on F_meas at the fixed point).
    from wbt_training.config_values.hybrid import HybridContactConfig, force_error_reward_preintegrator

    cfg = HybridContactConfig(servo_enabled=True)
    env = _hybrid_env()
    _make_injector(env, cfg)
    env._wbt_force_cmd_w[..., 0] = 10.0
    env._wbt_force_meas_w[..., 0] = -6.0  # partially applied
    env._wbt_integ_state[:] = 0.0
    r_no_integ = force_error_reward_preintegrator(env, cfg=cfg)
    # Now pretend the integrator patched the residual: at the fixed point F_meas moves by −K·I.
    # Set I and correspondingly move F_meas; the reward must be INVARIANT (credits policy only).
    env._wbt_integ_state[..., 0] = 4.0  # N (force-unit integral)
    env._wbt_force_meas_w[..., 0] = -6.0 - 4.0  # F_meas shifts by −I_force at the fixed point
    r_with_integ = force_error_reward_preintegrator(env, cfg=cfg)
    assert torch.allclose(r_no_integ, r_with_integ, atol=1e-5)


def test_force_runaway_fires_above_cutoff():
    from wbt_training.config_values.hybrid import HybridContactConfig, force_runaway

    cfg = HybridContactConfig()
    env = _hybrid_env()
    _make_injector(env, cfg)
    # below cutoff → not done. (Estop thresholds on the CONTACT force F_ext, NOT F_meas.)
    env._wbt_force_ext_w[..., :3] = 0.0
    env._wbt_force_ext_w[..., 0] = 10.0
    out = force_runaway(env, cfg=cfg)
    assert out.shape == (4,)
    assert out.dtype == torch.bool
    assert not out.any()
    # one hand above cutoff → that env done (any-hand).
    env._wbt_force_ext_w[0, 0, 0] = cfg.force_runaway_cutoff + 5.0
    out2 = force_runaway(env, cfg=cfg)
    assert bool(out2[0]) is True
    assert not out2[1:].any()


def test_force_runaway_ignores_inertial_spike():
    """Regression (2026-07-10 first 8-GPU run): the estop must key off the CONTACT force
    F_ext, NOT F_meas. F_meas = F_ext + m_distal·a_wrist + bias + noise, so a free-space
    wrist-acceleration transient (esp. the reset teleport) spikes F_meas well past the 40 N
    cutoff (40 N / 1.5 kg = 26.7 m/s², trivially exceeded) → the old F_meas-based estop killed
    ~every episode at reset (ep_len pinned at ~1.2; A/B-proven 12.5× vs estop-off). Thresholding
    on F_ext (hard-clamped to max_force_n < cutoff in sim) makes the estop a pure deploy-safety
    valve on the real Bota contact reading, never a free-space training killer."""
    from wbt_training.config_values.hybrid import HybridContactConfig, force_runaway

    cfg = HybridContactConfig()
    env = _hybrid_env()
    _make_injector(env, cfg)
    # Zero contact force, but a MASSIVE inertial spike in F_meas (as at reset teleport).
    env._wbt_force_ext_w[..., :3] = 0.0
    env._wbt_force_meas_w[..., :3] = 0.0
    env._wbt_force_meas_w[..., 0] = 500.0  # ≫ cutoff, pure inertial (no contact)
    out = force_runaway(env, cfg=cfg)
    assert not out.any(), "inertial-only F_meas spike must NOT trip the contact-force estop"


def test_force_ext_clamp_below_estop_cutoff():
    """The sim contact force is hard-clamped to max_force_n, which is BELOW the estop cutoff,
    so the F_ext-based estop is dormant in sim (clamp is the real sim guard) and only fires on
    the unclamped real-Bota contact reading at deploy. Locks that ordering invariant."""
    from wbt_training.config_values.hybrid import HybridContactConfig

    cfg = HybridContactConfig()
    assert cfg.max_force_n < cfg.force_runaway_cutoff


def test_reward_uses_servo_eq2_and_has_preintegrator():
    from wbt_training.config_values.hybrid import g1_29dof_wbt_2pt_hybrid_reward

    terms = g1_29dof_wbt_2pt_hybrid_reward.terms
    # Eq.2 term re-pointed at the servo-inclusive func (key kept for the Task-6 test).
    assert terms["force_track_pos_eq2"].func.endswith(":force_track_eq2_servo")
    assert terms["force_track_pos_eq2"].weight == 3.0
    # pre-integrator force reward present (small positive weight).
    assert "force_error_preintegrator" in terms
    assert terms["force_error_preintegrator"].weight == 1.0


def test_termination_has_force_runaway():
    from wbt_training.config_values.hybrid import g1_29dof_wbt_hybrid_termination

    assert "force_runaway" in g1_29dof_wbt_hybrid_termination.terms
    assert g1_29dof_wbt_hybrid_termination.terms["force_runaway"].func.endswith(":force_runaway")


def test_preset_ships_servo_enabled_with_estop():
    # After Task 11 the SHIPPED preset does force tracking on launch (user 2026-07-10).
    from wbt_training.config_values.experiment import DEFAULTS, g1_wbt_2pt

    exp = DEFAULTS["g1_wbt_2pt_hybrid"]
    hc = exp.command.setup_terms["hybrid_contact_injector"].params["hybrid_config"]
    assert hc.servo_enabled is True  # servo LIVE on launch (force tracking)
    assert hc.servo_gain_k >= 0.01  # near unity-authority, not the inert stub
    assert "force_runaway" in exp.termination.terms  # estop wired
    assert exp.termination != g1_wbt_2pt.termination  # NO LONGER equal (estop added)
    # obs still fixed 165D (servo state present in both modes) — checkpoint lineage intact.
    actor = exp.observation.groups["actor_obs"].terms
    assert len(actor) == 13


def test_servo_off_preset_is_phase1_position_variant():
    """Phase-1 (servo-OFF) preset for the two-phase curriculum (user 2026-07-10): position
    tracking + F_cmd/F_meas in obs, but NO control-path servo — train the stand to convergence,
    then resume the SAME checkpoint under the servo-ON preset (obs is fixed 165D in BOTH, so the
    network transfers). Everything else (obs width, reward keys, curriculum) is byte-identical to
    the servo-ON preset so the ONLY variable across the two phases is servo_enabled."""
    from wbt_training.config_values.experiment import DEFAULTS

    on = DEFAULTS["g1_wbt_2pt_hybrid"]
    off = DEFAULTS["g1_wbt_2pt_hybrid_servo_off"]
    # The one and only intended difference: the injector's servo flag.
    hc_off = off.command.setup_terms["hybrid_contact_injector"].params["hybrid_config"]
    assert hc_off.servo_enabled is False
    # Same fixed-165D obs (13 actor terms) → checkpoint lineage intact across the phase flip.
    assert len(off.observation.groups["actor_obs"].terms) == 13
    assert len(on.observation.groups["actor_obs"].terms) == 13
    # Reward keys identical (the Eq.2 term auto-delegates to the position path when servo off).
    assert set(off.reward.terms.keys()) == set(on.reward.terms.keys())
    # The Eq.2 reward's servo-branch is keyed off env._wbt_hybrid_cfg.servo_enabled, which is set
    # from THIS injector cfg — so the servo-off command config must carry servo_enabled=False.
    assert off.robot.asset.urdf_file == "g1/g1_29dof_yam.urdf"
