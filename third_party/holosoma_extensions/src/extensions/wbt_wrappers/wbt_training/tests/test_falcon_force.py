"""Unit tests for the FALCON-decoupled recipe building blocks.

Step 2: TCP-frame precision reward (tcp_track_pos_exp / tcp_track_orn_exp).
All tests use MagicMock for env/simulator/command — no IsaacSim runtime needed.
"""

import types
from unittest.mock import MagicMock

import torch


# 14-body tracking list (order matches robot_config.G1_29DOF_WBT_BODY_NAMES); only
# the wrist_yaw indices matter for the force/TCP terms.
_WBT_BODY_NAMES = [
    "pelvis", "left_hip_roll_link", "left_knee_link", "left_ankle_roll_link",
    "right_hip_roll_link", "right_knee_link", "right_ankle_roll_link",
    "torso_link", "left_shoulder_roll_link", "left_elbow_link",
    "left_wrist_yaw_link", "right_shoulder_roll_link", "right_elbow_link",
    "right_wrist_yaw_link",
]
_L_WRIST_TRACK_IDX = _WBT_BODY_NAMES.index("left_wrist_yaw_link")   # 10
_R_WRIST_TRACK_IDX = _WBT_BODY_NAMES.index("right_wrist_yaw_link")  # 13
# Sim body ids for the wrist_yaw links (arbitrary but consistent with fixture).
_L_WRIST_SIM_ID = 17
_R_WRIST_SIM_ID = 24


def _eye_quat_wxyz(*shape):
    q = torch.zeros(*shape, 4)
    q[..., 0] = 1.0
    return q


def _eye_quat_xyzw(*shape):
    q = torch.zeros(*shape, 4)
    q[..., 3] = 1.0
    return q


def _make_tcp_env(num_envs=2):
    """MagicMock env with a simulator (live wrist pose, WXYZ) and a motion_command
    state (motion ref pose, XYZW), sufficient for the TCP reward functions."""
    from wbt_training.config_values.force_tracking import HandForceConfig

    cfg = HandForceConfig()
    env = MagicMock()
    env.num_envs = num_envs
    env.device = torch.device("cpu")

    # --- simulator: live wrist pos/quat (WXYZ) ---
    sim = MagicMock()
    sim.find_rigid_body_indice = MagicMock(
        side_effect=lambda name: {"left_wrist_yaw_link": _L_WRIST_SIM_ID, "right_wrist_yaw_link": _R_WRIST_SIM_ID}[name]
    )
    sim.body_ids = list(range(50))
    sim._robot = MagicMock()
    sim._robot.data = types.SimpleNamespace(
        body_pos_w=torch.zeros(num_envs, 50, 3),
        body_quat_w=_eye_quat_wxyz(num_envs, 50),  # WXYZ identity
    )
    env.simulator = sim

    # --- world-frame force buffers (default zero: F=0 invariant) ---
    h = len(cfg.body_names)
    env._wbt_force_cmd_w = torch.zeros(num_envs, h, 3)
    env._wbt_force_ext_w = torch.zeros(num_envs, h, 3)

    # --- motion_command state: global motion ref pose (pos + XYZW quat) + live robot quat ---
    mc = types.SimpleNamespace()
    mc.motion_cfg = types.SimpleNamespace(body_names_to_track=list(_WBT_BODY_NAMES))
    mc.body_pos_w = torch.zeros(num_envs, len(_WBT_BODY_NAMES), 3)
    mc.body_quat_w = _eye_quat_xyzw(num_envs, len(_WBT_BODY_NAMES))  # XYZW
    mc.robot_body_quat_w = _eye_quat_xyzw(num_envs, len(_WBT_BODY_NAMES))  # XYZW
    env.command_manager = MagicMock()
    env.command_manager.get_state = MagicMock(return_value=mc)
    return env, cfg, mc


def test_tcp_offsets_from_urdf_fingertip():
    """TCP offset is the ~0.215 m +x fingertip midpoint (NOT the 0.0815 gripper base)."""
    from wbt_training.config_values.falcon_force import _TCP_OFFSETS_B

    assert _TCP_OFFSETS_B == ((0.215, 0.0, 0.0), (0.215, 0.0, 0.0))


def test_xyzw_to_wxyz_roundtrip():
    from wbt_training.config_values.falcon_force import _xyzw_to_wxyz

    q_xyzw = torch.tensor([[0.1, 0.2, 0.3, 0.9]])
    q_wxyz = _xyzw_to_wxyz(q_xyzw)
    assert torch.allclose(q_wxyz, torch.tensor([[0.9, 0.1, 0.2, 0.3]]))


def test_tcp_pos_reward_is_one_when_perfectly_tracked_and_zero_force():
    """cur == ref, F=0 ⇒ err=0 ⇒ r=exp(0)=1 (F=0 fingertip-tracking invariant)."""
    from wbt_training.config_values.falcon_force import tcp_track_pos_exp

    env, cfg, mc = _make_tcp_env(num_envs=2)
    # Live wrist and motion wrist both at identity orientation, same origin → TCP coincides.
    r = tcp_track_pos_exp(env, cfg=cfg)
    assert r.shape == (2,)
    assert torch.allclose(r, torch.ones(2), atol=1e-6)


def test_tcp_pos_reward_penalizes_wrist_orientation_error_via_lever():
    """A wrist-ORIENTATION error (live ≠ motion) surfaces as a fingertip-position
    error through the 0.215 m lever, LOWERING the reward — the whole point of the
    TCP term vs Eq.2 (which cancels the offset)."""
    from wbt_training.config_values.falcon_force import tcp_track_pos_exp

    env, cfg, mc = _make_tcp_env(num_envs=1)
    # Rotate the LIVE left wrist 90° about z (WXYZ), motion stays identity.
    s2 = (2**0.5) / 2.0
    env.simulator._robot.data.body_quat_w[:, _L_WRIST_SIM_ID, 0] = s2
    env.simulator._robot.data.body_quat_w[:, _L_WRIST_SIM_ID, 3] = s2
    r = tcp_track_pos_exp(env, cfg=cfg)
    # Live TCP: R(z90)@(0.215,0,0) = (0,0.215,0); ref TCP: (0.215,0,0). L1 err on left
    # hand = |0-0.215| + |0.215-0| + 0 = 0.43 m. r = exp(-0.43*2/0.15) < 0.01.
    assert r.item() < 0.01, r.item()


def test_tcp_pos_reward_eq2_shift_agrees_with_origin():
    """The TCP target is Eq.2-shifted by (F^cmd+F^ext)/K, so with perfectly-tracked
    orientation the TCP residual == the pure Eq.2 origin residual (F/K), i.e. a
    commanded force the policy has NOT yet realized lowers the reward by exactly F/K."""
    from wbt_training.config_values.falcon_force import tcp_track_pos_exp

    env, cfg, mc = _make_tcp_env(num_envs=1)
    # Command +x 20 N on the left hand (world frame). K=200 ⇒ target shifts +0.1 m.
    env._wbt_force_cmd_w[:, 0, 0] = 20.0
    r = tcp_track_pos_exp(env, cfg=cfg)
    # cur_tcp = ref_tcp (orientation matched, origin matched); target = ref + 0.1x.
    # err_l1 = 0.1 m (left) + 0 (right) = 0.1. r = exp(-0.1*2/σ_pos), σ_pos=0.15.
    expected = torch.exp(torch.tensor(-0.1 * 2.0 / 0.15))
    assert torch.allclose(r, expected.reshape(1), atol=1e-5), (r.item(), expected.item())


def test_tcp_orn_reward_one_when_matched_and_decays_with_error():
    from wbt_training.config_values.falcon_force import tcp_track_orn_exp

    env, cfg, mc = _make_tcp_env(num_envs=1)
    r0 = tcp_track_orn_exp(env, cfg=cfg)
    assert torch.allclose(r0, torch.ones(1), atol=1e-6)
    # Rotate the LIVE robot left wrist 90° about z (XYZW), motion stays identity.
    s2 = (2**0.5) / 2.0
    mc.robot_body_quat_w[:, _L_WRIST_TRACK_IDX, 2] = s2  # z
    mc.robot_body_quat_w[:, _L_WRIST_TRACK_IDX, 3] = s2  # w
    r1 = tcp_track_orn_exp(env, cfg=cfg)
    assert r1.item() < r0.item(), (r1.item(), r0.item())


def test_tcp_reward_functions_are_pure_readers():
    """Neither TCP reward may mutate the force buffers or motion/sim tensors."""
    from wbt_training.config_values.falcon_force import tcp_track_orn_exp, tcp_track_pos_exp

    env, cfg, mc = _make_tcp_env(num_envs=2)
    env._wbt_force_cmd_w.uniform_(-5, 5)
    snap_cmd = env._wbt_force_cmd_w.clone()
    snap_q = env.simulator._robot.data.body_quat_w.clone()
    snap_mq = mc.body_quat_w.clone()
    _ = tcp_track_pos_exp(env, cfg=cfg)
    _ = tcp_track_orn_exp(env, cfg=cfg)
    assert torch.equal(env._wbt_force_cmd_w, snap_cmd)
    assert torch.equal(env.simulator._robot.data.body_quat_w, snap_q)
    assert torch.equal(mc.body_quat_w, snap_mq)


def test_tcp_pos_reward_zero_force_independent_of_wrist_orientation_on_ref_side():
    """Sanity: with F=0 and live==motion orientation (both rotated identically),
    the TCP still coincides ⇒ r=1. Confirms the ref side uses the MOTION quat, so a
    matched non-identity pose is not penalized."""
    from wbt_training.config_values.falcon_force import tcp_track_pos_exp

    env, cfg, mc = _make_tcp_env(num_envs=1)
    s2 = (2**0.5) / 2.0
    # Rotate BOTH live (WXYZ) and motion (XYZW) left wrist 90° about z.
    env.simulator._robot.data.body_quat_w[:, _L_WRIST_SIM_ID, 0] = s2
    env.simulator._robot.data.body_quat_w[:, _L_WRIST_SIM_ID, 3] = s2
    mc.body_quat_w[:, _L_WRIST_TRACK_IDX, 2] = s2  # z (xyzw)
    mc.body_quat_w[:, _L_WRIST_TRACK_IDX, 3] = s2  # w (xyzw)
    r = tcp_track_pos_exp(env, cfg=cfg)
    assert torch.allclose(r, torch.ones(1), atol=1e-6), r.item()


# ===========================================================================
# Step 3: DualGroupRewardManager — per-group (lower/upper) reward routing.
# We build the class against a stub RewardManager base so it is unit-testable
# without the omni runtime.
# ===========================================================================


def _make_reward_cfg(terms: dict, only_positive: bool = False):
    """Minimal RewardManagerCfg-like object: .terms dict + .only_positive_rewards."""
    return types.SimpleNamespace(terms=terms, only_positive_rewards=only_positive)


def _term(func_name, weight, tags=None, params=None):
    return types.SimpleNamespace(func=func_name, weight=weight, tags=tags or [], params=params or {})


class _StubRewardManager:
    """Stand-in for holosoma RewardManager: same term-init/compute contract the
    DualGroupRewardManager subclass depends on (skips weight==0, parallel name/cfg
    lists, per-term funcs, scalar _reward_buf, episode sums)."""

    def __init__(self, cfg, env, device):
        self.cfg = cfg
        self.env = env
        self.device = device
        self._term_funcs = {}
        self._term_instances = {}
        self._term_names = []
        self._term_cfgs = []
        for name, tc in cfg.terms.items():
            if tc.weight == 0.0:
                continue
            self._term_funcs[name] = tc.func  # func is a plain callable in tests
            self._term_names.append(name)
            self._term_cfgs.append(tc)
        self._reward_buf = torch.zeros(env.num_envs)
        self._episode_sums = {n: torch.zeros(env.num_envs) for n in self._term_names}
        self._episode_sums_raw = {n: torch.zeros(env.num_envs) for n in self._term_names}


def _build_dual_manager_class(monkeypatch_import):
    """Import make_dual_group_reward_manager_class with the core base patched to the stub."""
    import sys

    # Inject a fake holosoma.managers.reward.manager exposing RewardManager=_StubRewardManager.
    fake_mod = types.ModuleType("holosoma.managers.reward.manager")
    fake_mod.RewardManager = _StubRewardManager
    saved = sys.modules.get("holosoma.managers.reward.manager")
    sys.modules["holosoma.managers.reward.manager"] = fake_mod
    try:
        from wbt_training.config_values.falcon_force import make_dual_group_reward_manager_class

        return make_dual_group_reward_manager_class()
    finally:
        if saved is not None:
            sys.modules["holosoma.managers.reward.manager"] = saved
        else:
            del sys.modules["holosoma.managers.reward.manager"]


def test_resolve_reward_group_by_tag():
    from wbt_training.config_values.falcon_force import UPPER_BODY_TAG, _resolve_reward_group

    assert _resolve_reward_group(_term("m:f", 1.0, tags=[UPPER_BODY_TAG])) == "upper"
    assert _resolve_reward_group(_term("m:f", 1.0, tags=["penalty"])) == "lower"
    assert _resolve_reward_group(_term("m:f", 1.0, tags=[])) == "lower"


def test_dual_group_reward_routes_and_sums_to_scalar():
    """Per-group buffers must route by tag AND sum exactly to the scalar rew_buf."""
    n = 3
    env = types.SimpleNamespace(num_envs=n)
    # Two upper terms, two lower terms; constant per-env rewards for determinism.
    terms = {
        "arm_track": _term(lambda e: torch.full((n,), 2.0), weight=1.0, tags=["upper_body"]),
        "force_eq2": _term(lambda e: torch.full((n,), 1.0), weight=4.0, tags=["upper_body"]),
        "vel_track": _term(lambda e: torch.full((n,), 3.0), weight=1.0, tags=["lower_body"]),
        "feet_slip": _term(lambda e: torch.full((n,), 1.0), weight=-2.0, tags=[]),  # no tag → lower
    }
    dual_cls = _build_dual_manager_class(None)
    mgr = dual_cls(_make_reward_cfg(terms), env, "cpu")
    dt = 0.02
    scalar = mgr.compute(dt)
    up = mgr.group_reward_buf["upper"]
    lo = mgr.group_reward_buf["lower"]
    # upper = (2*1 + 1*4)*dt = 6dt ; lower = (3*1 + 1*-2)*dt = 1dt ; scalar = 7dt
    assert torch.allclose(up, torch.full((n,), 6.0 * dt))
    assert torch.allclose(lo, torch.full((n,), 1.0 * dt))
    assert torch.allclose(scalar, torch.full((n,), 7.0 * dt))
    # Invariant: lower + upper == scalar, exactly.
    assert torch.allclose(up + lo, scalar, atol=1e-6)
    # Routing map introspection.
    assert mgr.reward_group_of == {
        "arm_track": "upper", "force_eq2": "upper", "vel_track": "lower", "feet_slip": "lower"
    }


def test_dual_group_reward_skips_zero_weight_terms():
    n = 2
    env = types.SimpleNamespace(num_envs=n)
    terms = {
        "active": _term(lambda e: torch.ones(n), weight=1.0, tags=["upper_body"]),
        "disabled": _term(lambda e: torch.full((n,), 99.0), weight=0.0, tags=["lower_body"]),
    }
    dual_cls = _build_dual_manager_class(None)
    mgr = dual_cls(_make_reward_cfg(terms), env, "cpu")
    mgr.compute(0.02)
    assert "disabled" not in mgr.reward_group_of, "weight==0 term must be skipped"
    assert torch.allclose(mgr.group_reward_buf["lower"], torch.zeros(n))


def test_dual_group_reward_only_positive_clips_each_group():
    """only_positive_rewards must clip BOTH group buffers (not just the scalar)."""
    n = 2
    env = types.SimpleNamespace(num_envs=n)
    terms = {
        "up_pos": _term(lambda e: torch.ones(n), weight=1.0, tags=["upper_body"]),
        "lo_neg": _term(lambda e: torch.ones(n), weight=-5.0, tags=["lower_body"]),  # drives lower negative
    }
    dual_cls = _build_dual_manager_class(None)
    mgr = dual_cls(_make_reward_cfg(terms, only_positive=True), env, "cpu")
    scalar = mgr.compute(0.02)
    assert torch.all(mgr.group_reward_buf["lower"] >= 0.0), "lower group must be clipped to >=0"
    assert torch.all(mgr.group_reward_buf["upper"] >= 0.0)
    assert torch.all(scalar >= 0.0)


def test_falcon_config_survives_dict_reload():
    """REGRESSION (comprehensive review, HIGH): PPODecoupledConfig must survive being
    reloaded from a serialized DICT (the eval/deploy path does
    ExperimentConfig(**checkpoint['experiment_config'])). Without the reload shim,
    pydantic's un-discriminated Union[PPOAlgoConfig, FastSACAlgoConfig] downcasts the
    subclass to a plain PPOConfig, DROPPING lower/upper → PPODecoupled._init_config
    crashes. The _rebuild_falcon_algo_if_needed shim must restore the subclass."""
    import dataclasses

    from wbt_training.config_values.experiment import g1_wbt_2pt_force_falcon
    from wbt_training.utils.config_reload import _rebuild_falcon_algo_if_needed

    # Serialize the whole ExperimentConfig to a plain dict (asdict = what the
    # checkpoint stores) and confirm lower/upper are present in the raw dict.
    exp_dict = dataclasses.asdict(g1_wbt_2pt_force_falcon)
    assert exp_dict["algo"]["_target_"].endswith("PPODecoupled")
    assert "lower" in exp_dict["algo"]["config"] and "upper" in exp_dict["algo"]["config"]

    # The shim rebuilds the algo INSTANCE from the dict; feeding that to
    # ExperimentConfig preserves lower/upper (instance path).
    rebuilt = _rebuild_falcon_algo_if_needed(exp_dict)
    algo = rebuilt["algo"]
    assert type(algo).__name__ == "PPODecoupledAlgoConfig"
    assert type(algo.config).__name__ == "PPODecoupledConfig"
    assert algo.config.lower is not None and algo.config.upper is not None

    # Non-FALCON algo dict → shim is a no-op (returns the input unchanged).
    plain = {"algo": {"_target_": "holosoma.agents.ppo.ppo.PPO", "config": {}}, "x": 1}
    assert _rebuild_falcon_algo_if_needed(plain) is plain


def test_falcon_config_reload_patch_is_idempotent():
    """install_falcon_config_reload_patch must be safe to call repeatedly and must
    patch the two eval_utils loaders exactly once."""
    from wbt_training.utils.config_reload import install_falcon_config_reload_patch

    try:
        from holosoma.utils import eval_utils
    except ImportError:
        import pytest

        pytest.skip("holosoma not installed")
    install_falcon_config_reload_patch()
    f1 = eval_utils._load_config_from_checkpoint
    install_falcon_config_reload_patch()  # second call: no-op
    assert eval_utils._load_config_from_checkpoint is f1
    assert getattr(eval_utils, "_wbt_falcon_config_reload_patched", False) is True


def test_dual_group_reward_compute_is_idempotent_across_calls():
    """Group buffers must be zeroed at the start of each compute (no accumulation)."""
    n = 2
    env = types.SimpleNamespace(num_envs=n)
    terms = {"up": _term(lambda e: torch.ones(n), weight=1.0, tags=["upper_body"])}
    dual_cls = _build_dual_manager_class(None)
    mgr = dual_cls(_make_reward_cfg(terms), env, "cpu")
    r1 = mgr.compute(0.02).clone()
    u1 = mgr.group_reward_buf["upper"].clone()
    r2 = mgr.compute(0.02)
    u2 = mgr.group_reward_buf["upper"]
    assert torch.allclose(r1, r2), "scalar must not accumulate across compute() calls"
    assert torch.allclose(u1, u2), "group buffer must not accumulate across compute() calls"
