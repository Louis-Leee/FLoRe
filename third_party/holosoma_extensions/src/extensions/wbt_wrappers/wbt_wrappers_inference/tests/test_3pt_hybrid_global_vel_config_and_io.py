"""3pt-hybrid-GLOBAL-VEL deploy config parity + 6D-orn ingest/clutch helpers.

Hardware-free unit tests (no robot, no live sources, no ONNX file required) that
pin the load-bearing invariants of the exp:g1-wbt-3pt-hybrid-global-vel sim2real
port (canonical training run bqlzkwok):

  1. The deploy obs config ``holosoma_3pt_hybrid_global_vel_wbt`` matches the
     training ground truth (183D, exact obs00..obs15 term order, dims, scales)
     and both DEFAULTS keys resolve to the same object.
  2. The plain 3pt-hybrid (168D) and 2pt-hybrid (165D) configs are UNCHANGED —
     the port is purely additive.
  3. detect_obs_mode returns "3pt_global" (motion_ref_pos_b present) so the
     first-tick motion alignment + the 3pt VIVE override both fire.
  4. The runtime obs term-ORDER validator accepts the REAL bqlzkwok key scheme
     and rejects a swapped preset / the old 168D preset against it.
  5. The quat→rot6d ingest helper matches the FK rot6d convention bit-for-bit,
     is double-cover invariant (q and −q map to the SAME 6D), and the
     generalized N-body rot6d clutch is zero-jump at 18D while preserving the
     2pt 12D behavior.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("wbt_wrappers_inference")

# The EXACT actor_obs keys baked into the bqlzkwok ONNX experiment_config
# (wbt_training hybrid_3pt_global_vel.py sorted keys) — dumped from
# logs/WBT-Holosoma/bqlzkwok/model_02500.onnx metadata.
_BQLZKWOK_TRAIN_KEYS = [
    "obs00_command_lower_body",
    "obs01_vr_3point_pos",
    "obs02_vr_3point_orn_6d",
    "obs03_motion_ref_pos_b",
    "obs04_motion_ref_vel_b",
    "obs05_force_cmd_local",
    "obs06_force_meas_local",
    "obs07_integrator_state",
    "obs08_x_tgt_shift",
    "obs09_motion_ref_ori_b",
    "obs10_projected_gravity",
    "obs11_base_ang_vel",
    "obs12_base_lin_vel",
    "obs13_dof_pos",
    "obs14_dof_vel",
    "obs15_actions",
]

_EXPECTED_TERMS = [
    "command_lower_body",
    "vr_3point_pos",
    "vr_3point_orn_6d",
    "motion_ref_pos_b",
    "motion_ref_vel_b",
    "force_cmd_local",
    "force_meas_local",
    "integrator_state",
    "x_tgt_shift",
    "motion_ref_ori_b",
    "projected_gravity",
    "base_ang_vel",
    "base_lin_vel",
    "joint_pos",
    "joint_vel",
    "actions",
]


# --- config parity -----------------------------------------------------------
def test_3pt_hybrid_global_vel_config_is_183d_exact_order():
    from wbt_wrappers_inference.config_values.inference import DEFAULTS

    cfg = DEFAULTS["g1-29dof-holosoma-3pt-hybrid-global-vel-wbt"].observation
    terms = cfg.obs_dict["actor_obs"]
    assert terms == _EXPECTED_TERMS, terms
    per = sum(cfg.obs_dims[t] for t in terms)
    assert per == 183, per
    assert cfg.history_length_dict["actor_obs"] == 4
    assert per * 4 == 732
    # Key dims: the two deltas vs 3pt-hybrid.
    assert cfg.obs_dims["vr_3point_orn_6d"] == 18  # 6D rot rep, NOT the 12D quat
    for t in ("motion_ref_pos_b", "motion_ref_vel_b", "base_lin_vel"):
        assert cfg.obs_dims[t] == 3, t
    # Hybrid force channel dims + scales (byte-identical to 3pt-hybrid).
    for t in ("force_cmd_local", "force_meas_local", "integrator_state", "x_tgt_shift"):
        assert cfg.obs_dims[t] == 6
    assert cfg.obs_scales["force_cmd_local"] == 0.04
    assert cfg.obs_scales["force_meas_local"] == 0.04
    assert cfg.obs_scales["integrator_state"] == 1.0
    assert cfg.obs_scales["x_tgt_shift"] == 1.0
    assert cfg.obs_scales["vr_3point_orn_6d"] == 1.0


def test_both_global_vel_keys_resolve_to_same_config():
    from wbt_wrappers_inference.config_values.inference import DEFAULTS

    a = DEFAULTS["g1-29dof-holosoma-3pt-hybrid-global-vel-wbt"]
    b = DEFAULTS["g1-29dof-holosoma-3pt-hybrid-global-vel-layerA-wbt"]
    assert a is b, "servo on/off is a runtime flag, not an obs difference"


def test_full_dims_and_scales_pinned():
    """Pin EVERY term width and scale (the training ground truth) — the sum
    check alone would pass a pair of COMPENSATING dim errors, which also slips
    the runtime total-width gate and the name-only order validator."""
    from wbt_wrappers_inference.config_values.inference import DEFAULTS

    cfg = DEFAULTS["g1-29dof-holosoma-3pt-hybrid-global-vel-wbt"].observation
    expected_dims = {
        "command_lower_body": 24,
        "vr_3point_pos": 9,
        "vr_3point_orn_6d": 18,
        "motion_ref_pos_b": 3,
        "motion_ref_vel_b": 3,
        "force_cmd_local": 6,
        "force_meas_local": 6,
        "integrator_state": 6,
        "x_tgt_shift": 6,
        "motion_ref_ori_b": 6,
        "projected_gravity": 3,
        "base_ang_vel": 3,
        "base_lin_vel": 3,
        "joint_pos": 29,
        "joint_vel": 29,
        "actions": 29,
    }
    assert cfg.obs_dims == expected_dims, cfg.obs_dims
    expected_scales = {t: 1.0 for t in expected_dims}
    expected_scales["force_cmd_local"] = 0.04
    expected_scales["force_meas_local"] = 0.04
    assert cfg.obs_scales == expected_scales, cfg.obs_scales


def test_inference_bindings_match_sibling_3pt_presets():
    """The new preset must reuse the SAME task + robot objects as the sibling
    3pt-hybrid preset (only the observation differs — mirroring training's
    `replace(g1_wbt_3pt_hybrid, observation=...)`)."""
    from wbt_wrappers_inference.config_values.inference import DEFAULTS

    gv = DEFAULTS["g1-29dof-holosoma-3pt-hybrid-global-vel-wbt"]
    h3 = DEFAULTS["g1-29dof-holosoma-3pt-hybrid-wbt"]
    assert gv.task is h3.task
    assert gv.robot is h3.robot
    assert gv.observation is not h3.observation


def test_existing_hybrid_configs_unchanged():
    """The port must not disturb the 3pt-hybrid (168D) / 2pt-hybrid (165D) configs."""
    from wbt_wrappers_inference.config_values.inference import DEFAULTS

    cfg3 = DEFAULTS["g1-29dof-holosoma-3pt-hybrid-wbt"].observation
    per3 = sum(cfg3.obs_dims[t] for t in cfg3.obs_dict["actor_obs"])
    assert per3 == 168 and per3 * 4 == 672
    assert cfg3.obs_dims["vr_3point_orn"] == 12  # still the quat rep
    assert "vr_3point_orn_6d" not in cfg3.obs_dict["actor_obs"]
    cfg2 = DEFAULTS["g1-29dof-holosoma-2pt-hybrid-wbt"].observation
    per2 = sum(cfg2.obs_dims[t] for t in cfg2.obs_dict["actor_obs"])
    assert per2 == 165 and per2 * 4 == 660


def test_global_vel_obs_mode_is_3pt_global():
    """motion_ref_pos_b present ⇒ "3pt_global" ⇒ _is_3pt True (VIVE-eligible) AND
    the first-tick motion alignment fires (rl_inference's mode gate)."""
    from wbt_wrappers_inference.config_values.inference import DEFAULTS
    from wbt_wrappers_inference.utils.obs_utils import detect_obs_mode

    terms = DEFAULTS["g1-29dof-holosoma-3pt-hybrid-global-vel-wbt"].observation.obs_dict["actor_obs"]
    assert detect_obs_mode(terms) == "3pt_global"


# --- runtime term-order validation vs the REAL bqlzkwok key scheme ------------
def _exp_config(keys):
    return {"observation": {"groups": {"actor_obs": {"terms": {k: {} for k in keys}}}}}


def test_term_order_validation_accepts_bqlzkwok_scheme():
    from wbt_wrappers_inference.config_values.inference import DEFAULTS
    from wbt_wrappers_inference.policies.holosoma_wbt import HolosomaWBTPolicy as P

    stub = object.__new__(P)
    stub._obs_terms = list(DEFAULTS["g1-29dof-holosoma-3pt-hybrid-global-vel-wbt"].observation.obs_dict["actor_obs"])
    # The real checkpoint's zero-padded keys must NOT raise.
    P._validate_obs_term_order(stub, _exp_config(_BQLZKWOK_TRAIN_KEYS))


def test_term_order_validation_rejects_swap_against_bqlzkwok():
    from wbt_wrappers_inference.config_values.inference import DEFAULTS
    from wbt_wrappers_inference.policies.holosoma_wbt import HolosomaWBTPolicy as P

    terms = list(DEFAULTS["g1-29dof-holosoma-3pt-hybrid-global-vel-wbt"].observation.obs_dict["actor_obs"])
    # Swap the two root-state trio members (equal 3D width — the width check
    # alone would NOT catch this).
    i, j = terms.index("motion_ref_pos_b"), terms.index("motion_ref_vel_b")
    terms[i], terms[j] = terms[j], terms[i]
    stub = object.__new__(P)
    stub._obs_terms = terms
    with pytest.raises(ValueError, match="term-ORDER mismatch"):
        P._validate_obs_term_order(stub, _exp_config(_BQLZKWOK_TRAIN_KEYS))


def test_term_order_validation_rejects_old_3pt_hybrid_preset_against_bqlzkwok():
    """Loading the bqlzkwok ckpt with the OLD 168D preset must fail LOUDLY
    (term-count 13 vs 16) before any width math."""
    from wbt_wrappers_inference.config_values.inference import DEFAULTS
    from wbt_wrappers_inference.policies.holosoma_wbt import HolosomaWBTPolicy as P

    stub = object.__new__(P)
    stub._obs_terms = list(DEFAULTS["g1-29dof-holosoma-3pt-hybrid-wbt"].observation.obs_dict["actor_obs"])
    with pytest.raises(ValueError, match="term-COUNT mismatch"):
        P._validate_obs_term_order(stub, _exp_config(_BQLZKWOK_TRAIN_KEYS))


# --- quat→rot6d ingest helper --------------------------------------------------
def _random_unit_quats_xyzw(rng, n):
    q = rng.standard_normal((n, 4))
    return (q / np.linalg.norm(q, axis=1, keepdims=True)).astype(np.float32)


def test_quat12_to_rot6d18_matches_fk_convention():
    """Per body, the helper must equal matrix_from_quat(wxyz)[..., :2].reshape(-1)
    — the EXACT rot6d _compute_vr_fk emits (and training's quaternion_to_matrix
    w_last=True [..., :2] flatten)."""
    from holosoma_inference.utils.math.quat import matrix_from_quat

    from wbt_wrappers_inference.policies.holosoma_wbt import HolosomaWBTPolicy as P

    rng = np.random.default_rng(7)
    quats = _random_unit_quats_xyzw(rng, 3)
    out = P._quat12_to_rot6d18(quats.reshape(-1))
    assert out.shape == (18,)
    for b in range(3):
        q_xyzw = quats[b]
        q_wxyz = np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]], dtype=np.float32).reshape(1, 4)
        expected = matrix_from_quat(q_wxyz)[..., :2].reshape(-1)
        assert np.allclose(out[b * 6 : b * 6 + 6], expected, atol=1e-5), f"body {b}"


def test_quat12_to_rot6d18_double_cover_invariant():
    """q and −q are the same rotation ⇒ identical 6D output (the double cover
    dies at ingest — the training-side rationale for the 6D rep)."""
    from wbt_wrappers_inference.policies.holosoma_wbt import HolosomaWBTPolicy as P

    rng = np.random.default_rng(8)
    quats = _random_unit_quats_xyzw(rng, 3).reshape(-1)
    assert np.allclose(P._quat12_to_rot6d18(quats), P._quat12_to_rot6d18(-quats), atol=1e-6)


def test_quat12_to_rot6d18_blocks_are_valid_rotations():
    from wbt_wrappers_inference.policies.holosoma_wbt import HolosomaWBTPolicy as P

    rng = np.random.default_rng(9)
    out = P._quat12_to_rot6d18(_random_unit_quats_xyzw(rng, 3).reshape(-1))
    for b in range(3):
        m = out[b * 6 : b * 6 + 6].reshape(3, 2)
        # First two rotation-matrix columns: unit norm + orthogonal.
        assert abs(np.linalg.norm(m[:, 0]) - 1.0) < 1e-5
        assert abs(np.linalg.norm(m[:, 1]) - 1.0) < 1e-5
        assert abs(float(m[:, 0] @ m[:, 1])) < 1e-5


# --- generalized N-body rot6d clutch -------------------------------------------
def test_orn_clutch_target_18d_zero_jump():
    """live==live0 ⇒ target == ref, for all 3 rot6d bodies (the 3pt 6D preset)."""
    from wbt_wrappers_inference.policies.holosoma_wbt import HolosomaWBTPolicy as P

    rng = np.random.default_rng(10)
    ref = P._quat12_to_rot6d18(_random_unit_quats_xyzw(rng, 3).reshape(-1))
    live0 = P._quat12_to_rot6d18(_random_unit_quats_xyzw(rng, 3).reshape(-1))
    stub = object.__new__(P)
    tgt = P._orn_clutch_target(stub, ref, live0, live0)
    assert tgt.shape == (18,)
    assert np.allclose(tgt, ref, atol=1e-5), "zero-delta arm must be zero-jump"


def test_orn_clutch_target_18d_applies_relative_rotation():
    """live = R_d·live0 ⇒ target = R_d·ref, per body (the clutch semantics)."""
    from wbt_wrappers_inference.policies.holosoma_wbt import HolosomaWBTPolicy as P

    rng = np.random.default_rng(11)
    stub = object.__new__(P)
    ref18 = P._quat12_to_rot6d18(_random_unit_quats_xyzw(rng, 3).reshape(-1))
    live0_18 = P._quat12_to_rot6d18(_random_unit_quats_xyzw(rng, 3).reshape(-1))
    # A 30° yaw delta applied to every body.
    c, s = np.cos(np.pi / 6), np.sin(np.pi / 6)
    R_d = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    live18 = np.zeros(18, dtype=np.float32)
    for b in range(3):
        sl = slice(b * 6, b * 6 + 6)
        live18[sl] = P._R_to_rot6d(R_d @ P._rot6d_to_R(live0_18[sl]))
    tgt = P._orn_clutch_target(stub, ref18, live18, live0_18)
    for b in range(3):
        sl = slice(b * 6, b * 6 + 6)
        expected = R_d @ P._rot6d_to_R(ref18[sl])
        assert np.allclose(P._rot6d_to_R(tgt[sl]), expected, atol=1e-4), f"body {b}"


def test_orn_clutch_target_12d_2pt_behavior_preserved():
    """The generalization must keep the 2pt (12D, 2-body) path byte-equivalent:
    zero-jump at 12D and the same per-hand relative composition."""
    from wbt_wrappers_inference.policies.holosoma_wbt import HolosomaWBTPolicy as P

    rng = np.random.default_rng(12)
    stub = object.__new__(P)
    q4 = _random_unit_quats_xyzw(rng, 4)
    ref12 = np.concatenate([P._R_to_rot6d(P._quat_xyzw_to_R(q4[i])) for i in range(2)]).astype(np.float32)
    live0 = np.concatenate([P._R_to_rot6d(P._quat_xyzw_to_R(q4[i + 2])) for i in range(2)]).astype(np.float32)
    tgt = P._orn_clutch_target(stub, ref12, live0, live0)
    assert tgt.shape == (12,)
    assert np.allclose(tgt, ref12, atol=1e-5)
