"""3pt-hybrid deploy config parity + ONNX I/O-adaptivity + 3pt orn/clutch helpers.

Hardware-free unit tests (no robot, no live sources, no ONNX file required) that
pin the load-bearing invariants of the 3pt-hybrid sim2real port:

  1. The deploy obs config ``holosoma_3pt_hybrid_wbt`` matches the training ground
     truth (168D, exact term order, dims, scales) and both DEFAULTS keys resolve.
  2. The 2pt-hybrid config is UNCHANGED (165D) — the port is purely additive.
  3. The new quaternion helpers round-trip and the 3pt orn clutch is zero-jump.
  4. The mixed-length clutch/match-len helpers guard the obs width.

The ONNX I/O-name resolution logic is exercised as a pure function (the same
priority the policy's ``setup_policy`` uses) without needing an ONNX file.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("wbt_wrappers_inference")


# --- config parity -----------------------------------------------------------
def test_3pt_hybrid_config_is_168d_exact_order():
    from wbt_wrappers_inference.config_values.inference import DEFAULTS

    cfg = DEFAULTS["g1-29dof-holosoma-3pt-hybrid-wbt"].observation
    terms = cfg.obs_dict["actor_obs"]
    # Exact training order (hybrid_3pt.py sorted keys obs00..obs12).
    assert terms == [
        "command_lower_body",
        "vr_3point_pos",
        "vr_3point_orn",
        "force_cmd_local",
        "force_meas_local",
        "integrator_state",
        "x_tgt_shift",
        "motion_ref_ori_b",
        "projected_gravity",
        "base_ang_vel",
        "joint_pos",
        "joint_vel",
        "actions",
    ], terms
    per = sum(cfg.obs_dims[t] for t in terms)
    assert per == 168, per
    assert cfg.history_length_dict["actor_obs"] == 4
    assert per * 4 == 672
    # Key dims + scales.
    assert cfg.obs_dims["vr_3point_pos"] == 9
    assert cfg.obs_dims["vr_3point_orn"] == 12
    for t in ("force_cmd_local", "force_meas_local", "integrator_state", "x_tgt_shift"):
        assert cfg.obs_dims[t] == 6
    assert cfg.obs_scales["force_cmd_local"] == 0.04
    assert cfg.obs_scales["force_meas_local"] == 0.04
    assert cfg.obs_scales["integrator_state"] == 1.0
    assert cfg.obs_scales["x_tgt_shift"] == 1.0


def test_both_3pt_hybrid_keys_resolve_to_same_config():
    from wbt_wrappers_inference.config_values.inference import DEFAULTS

    a = DEFAULTS["g1-29dof-holosoma-3pt-hybrid-wbt"]
    b = DEFAULTS["g1-29dof-holosoma-3pt-hybrid-layerA-wbt"]
    assert a is b, "servo on/off is a runtime flag, not an obs difference"


def test_2pt_hybrid_config_unchanged_165d():
    """The port must not disturb the 2pt-hybrid config (byte-identical deploy)."""
    from wbt_wrappers_inference.config_values.inference import DEFAULTS

    cfg = DEFAULTS["g1-29dof-holosoma-2pt-hybrid-wbt"].observation
    terms = cfg.obs_dict["actor_obs"]
    per = sum(cfg.obs_dims[t] for t in terms)
    assert per == 165 and per * 4 == 660
    assert cfg.obs_dims["vr_2point_pos"] == 6  # 2pt, NOT 9


def test_3pt_obs_mode_detection():
    from wbt_wrappers_inference.utils.obs_utils import detect_obs_mode

    from wbt_wrappers_inference.config_values.inference import DEFAULTS

    terms = DEFAULTS["g1-29dof-holosoma-3pt-hybrid-wbt"].observation.obs_dict["actor_obs"]
    assert detect_obs_mode(terms) == "3pt"


# --- ONNX I/O + action metadata validation (production helpers) ---------------
def _resolve_io(input_names, output_names):
    from wbt_wrappers_inference.policies.holosoma_wbt import resolve_onnx_io_names

    return resolve_onnx_io_names(input_names, output_names)


def test_io_resolution_deploy_format():
    """Deploy export (2pt lineage): obs + time_step → actions."""
    assert _resolve_io(["obs", "time_step"], ["actions", "joint_pos", "ref_quat_xyzw"]) == ("obs", True, "actions")


def test_io_resolution_browser_format():
    """Browser/mjswan export (3pt n4exfyvp): actor_obs only → action."""
    assert _resolve_io(["actor_obs"], ["action"]) == ("actor_obs", False, "action")


def test_io_resolution_rejects_aux_only_outputs():
    with pytest.raises(ValueError, match="no supported action output"):
        _resolve_io(["obs", "time_step"], ["joint_pos", "joint_vel"])


@pytest.mark.parametrize("shape", [(29,), (1, 29), ("batch", 29)])
def test_action_shape_validation_accepts_supported_shapes(shape):
    from wbt_wrappers_inference.policies.holosoma_wbt import validate_onnx_action_shape

    validate_onnx_action_shape(shape, 29)


@pytest.mark.parametrize("shape", [(), (1,), (29, 1), (2, 29), (1, 29, 1)])
def test_action_shape_validation_rejects_incompatible_shapes(shape):
    from wbt_wrappers_inference.policies.holosoma_wbt import validate_onnx_action_shape

    with pytest.raises(ValueError, match="ONNX action output"):
        validate_onnx_action_shape(shape, 29)


def test_action_scale_parser_flattens_and_requires_finite_values():
    from wbt_wrappers_inference.policies.holosoma_wbt import parse_action_scale

    scale = parse_action_scale(np.ones((29, 1)), 29)
    assert scale.shape == (29,)
    with pytest.raises(ValueError, match="finite"):
        parse_action_scale([np.nan] * 29, 29)


# --- obs term-order runtime validation (vs ONNX experiment_config) -----------
def _exp_config_for(terms):
    """A minimal experiment_config with zero-padded actor_obs keys in ``terms``
    order (dof<->joint aliased back to the training dof_ name)."""
    alias = {"joint_pos": "dof_pos", "joint_vel": "dof_vel"}
    keys = {f"obs{i:02d}_{alias.get(t, t)}": {} for i, t in enumerate(terms)}
    return {"observation": {"groups": {"actor_obs": {"terms": keys}}}}


def test_term_order_validation_accepts_matching_order():
    from wbt_wrappers_inference.config_values.inference import DEFAULTS
    from wbt_wrappers_inference.policies.holosoma_wbt import HolosomaWBTPolicy as P

    for preset in ("g1-29dof-holosoma-3pt-hybrid-wbt", "g1-29dof-holosoma-2pt-hybrid-wbt"):
        terms = list(DEFAULTS[preset].observation.obs_dict["actor_obs"])
        stub = object.__new__(P)
        stub._obs_terms = terms
        # matching order (dof<->joint aliased) must NOT raise
        P._validate_obs_term_order(stub, _exp_config_for(terms))


def test_term_order_validation_rejects_swap():
    import pytest

    from wbt_wrappers_inference.config_values.inference import DEFAULTS
    from wbt_wrappers_inference.policies.holosoma_wbt import HolosomaWBTPolicy as P

    terms = list(DEFAULTS["g1-29dof-holosoma-3pt-hybrid-wbt"].observation.obs_dict["actor_obs"])
    good = _exp_config_for(terms)  # checkpoint's true order
    scrambled = terms.copy()
    scrambled[1], scrambled[2] = scrambled[2], scrambled[1]  # deploy has vr_pos/orn swapped
    stub = object.__new__(P)
    stub._obs_terms = scrambled
    with pytest.raises(ValueError, match="term-ORDER mismatch"):
        P._validate_obs_term_order(stub, good)


def test_term_order_validation_skips_without_metadata():
    from wbt_wrappers_inference.config_values.inference import DEFAULTS
    from wbt_wrappers_inference.policies.holosoma_wbt import HolosomaWBTPolicy as P

    stub = object.__new__(P)
    stub._obs_terms = list(DEFAULTS["g1-29dof-holosoma-3pt-hybrid-wbt"].observation.obs_dict["actor_obs"])
    # empty / structureless metadata → graceful skip (no raise)
    P._validate_obs_term_order(stub, {})
    P._validate_obs_term_order(stub, {"observation": {}})


def test_term_order_no_false_positive_on_legacy_presets():
    """The validator runs for EVERY HolosomaWBT preset. It must NOT raise on a
    legitimate legacy preset+checkpoint pairing whose NAMES differ from training:
      * chip: deploy terms KEEP the obsNN_ prefix (training strips to bare names)
      * global/dense: deploy 'command' vs training 'motion_command'
      * terrain: BARE training keys (lexical sort ≠ semantic order → SET check)
    Each is fed a synthetic experiment_config built from the TRAINING-side names in
    the checkpoint's own key scheme; a correct pairing must pass (no raise)."""
    from wbt_wrappers_inference.config_values.inference import DEFAULTS
    from wbt_wrappers_inference.policies.holosoma_wbt import HolosomaWBTPolicy as P

    # deploy-term → training-side term name (the inverse of _canon collisions).
    to_train = {"joint_pos": "dof_pos", "joint_vel": "dof_vel", "command": "motion_command"}

    def _zeropad_cfg(terms):
        keys = {f"obs{i:02d}_{to_train.get(P._canon_obs_term(t), t)}": {} for i, t in enumerate(terms)}
        return {"observation": {"groups": {"actor_obs": {"terms": keys}}}}

    def _bare_cfg(terms):  # terrain-style: no obsNN_ prefix → SET check path
        keys = {to_train.get(P._canon_obs_term(t), t): {} for t in terms}
        return {"observation": {"groups": {"actor_obs": {"terms": keys}}}}

    # chip: deploy terms already carry obsNN_ prefixes; training keys are the SAME
    # zero-padded scheme but stripped to bare names — both canonicalize equal.
    for preset in ("g1-29dof-holosoma-chip-local-wbt", "g1-29dof-holosoma-chip-global-wbt"):
        terms = list(DEFAULTS[preset].observation.obs_dict["actor_obs"])
        # training stores the SAME obsNN_ keys (chip trains with zero-padded keys)
        cfg = {"observation": {"groups": {"actor_obs": {"terms": {t: {} for t in terms}}}}}
        stub = object.__new__(P)
        stub._obs_terms = terms
        P._validate_obs_term_order(stub, cfg)  # must NOT raise

    # global / dense: 'command' (deploy) vs 'motion_command' (training), zero-padded.
    for preset in ("g1-29dof-holosoma-global-wbt", "g1-29dof-holosoma-wbt"):
        terms = list(DEFAULTS[preset].observation.obs_dict["actor_obs"])
        stub = object.__new__(P)
        stub._obs_terms = terms
        P._validate_obs_term_order(stub, _zeropad_cfg(terms))  # must NOT raise

    # terrain: BARE training keys → SET check (order not asserted); must NOT raise.
    terms = list(DEFAULTS["g1-29dof-holosoma-terrain-wbt"].observation.obs_dict["actor_obs"])
    stub = object.__new__(P)
    stub._obs_terms = terms
    P._validate_obs_term_order(stub, _bare_cfg(terms))  # must NOT raise


def test_term_set_check_rejects_missing_term_on_bare_keys():
    """Even on the bare-key SET-check path (order not asserted), a genuinely MISSING
    or renamed term must still raise (a real preset/checkpoint mismatch)."""
    import pytest

    from wbt_wrappers_inference.config_values.inference import DEFAULTS
    from wbt_wrappers_inference.policies.holosoma_wbt import HolosomaWBTPolicy as P

    terms = list(DEFAULTS["g1-29dof-holosoma-terrain-wbt"].observation.obs_dict["actor_obs"])
    to_train = {"joint_pos": "dof_pos", "joint_vel": "dof_vel", "command": "motion_command"}
    keys = {to_train.get(P._canon_obs_term(t), t): {} for t in terms}
    # drop one training term → count/set mismatch
    keys.pop(next(iter(keys)))
    stub = object.__new__(P)
    stub._obs_terms = terms
    with pytest.raises(ValueError):
        P._validate_obs_term_order(stub, {"observation": {"groups": {"actor_obs": {"terms": keys}}}})


# --- quaternion + clutch helpers ---------------------------------------------
def test_quat_matrix_roundtrip():
    from wbt_wrappers_inference.policies.holosoma_wbt import HolosomaWBTPolicy as P

    rng = np.random.default_rng(1)
    for _ in range(100):
        q = rng.standard_normal(4)
        q /= np.linalg.norm(q)
        R = P._quat_xyzw_to_R(q)
        assert np.allclose(R @ R.T, np.eye(3), atol=1e-6)
        assert abs(np.linalg.det(R) - 1.0) < 1e-6
        R2 = P._quat_xyzw_to_R(P._R_to_quat_xyzw(R))
        assert np.allclose(R, R2, atol=1e-5)


@pytest.mark.parametrize("n_bodies", [3, 5])
def test_quat_orn_clutch_zero_jump(n_bodies: int) -> None:
    """live==live0 ⇒ target == ref (per-body, sign-agnostic).

    Parameterized over the point count because the SAME helper serves 3pt (3×xyzw = 12D)
    and 5pt (5×xyzw = 20D): N is read off the array, so 5pt inherits the zero-jump property
    rather than reimplementing it. n=5 would pass vacuously on a hardcoded-3 version — it
    would silently leave the two foot quaternions at zero — so the output width is asserted
    too."""
    from wbt_wrappers_inference.policies.holosoma_wbt import HolosomaWBTPolicy as P

    rng = np.random.default_rng(2)

    def _quats(k: int) -> np.ndarray:
        return np.concatenate(
            [P._R_to_quat_xyzw(P._quat_xyzw_to_R(rng.standard_normal(4))) for _ in range(k)]
        ).astype(np.float32)

    ref = _quats(n_bodies)
    live0 = _quats(n_bodies)
    stub = object.__new__(P)
    tgt = P._orn_clutch_target_quat(stub, ref, live0, live0)
    assert tgt.shape == (4 * n_bodies,), tgt.shape
    for b in range(n_bodies):
        Ra = P._quat_xyzw_to_R(tgt[b * 4 : b * 4 + 4])
        Rr = P._quat_xyzw_to_R(ref[b * 4 : b * 4 + 4])
        assert np.allclose(Ra, Rr, atol=1e-5), f"zero-jump failed body {b}"


def test_apply_clutch_delta_lengths():
    from wbt_wrappers_inference.policies.holosoma_wbt import HolosomaWBTPolicy as P

    stub = object.__new__(P)
    ref9 = np.arange(9.0, dtype=np.float32)
    # zero-jump (live==live0)
    assert np.allclose(P._apply_clutch_delta(stub, ref9, ref9 + 10, ref9 + 10), ref9)
    # short live (2-tracker 6D) on 3pt ref: only the leading 6 move, head holds ref
    out = P._apply_clutch_delta(stub, ref9, ref9[:6] + 5, ref9[:6])
    assert np.allclose(out[:6], ref9[:6] + 5) and np.allclose(out[6:], ref9[6:])


def test_match_len_guards_obs_width():
    from wbt_wrappers_inference.policies.holosoma_wbt import HolosomaWBTPolicy as P

    # equal length → live as-is
    assert np.allclose(P._match_len(np.arange(9.0), np.zeros(9)), np.arange(9.0))
    # short live → pad-hold tail at ref
    mm = P._match_len(np.arange(6.0), np.full(9, -1.0))
    assert np.allclose(mm[:6], np.arange(6.0)) and np.allclose(mm[6:], -1.0)
    # long live → truncate to ref width (never grow the obs)
    assert P._match_len(np.arange(12.0), np.zeros(9)).shape == (9,)


@pytest.mark.parametrize("n_bodies", [3, 5])
def test_quat_slew_double_cover_guard(n_bodies: int) -> None:
    """A quaternion and its antipode −q are the same rotation, but a LINEAR slew
    between them passes through ~0 (renormalizing → a wild rotation mid-ramp — a
    real hardware jerk). ``_align_quat_signs`` flips the target into the prev's
    hemisphere so the slew never crosses the double cover.

    Parameterized over 3pt (12D) and 5pt (20D): a hardcoded-3 version leaves bodies 4-5
    unflipped, which is exactly the silent jerk this guard exists to prevent — and it would
    show up on the FEET, the two points an operator is least likely to be watching."""
    from wbt_wrappers_inference.policies.holosoma_wbt import HolosomaWBTPolicy as P

    rng = np.random.default_rng(3)

    def _quats(k: int) -> np.ndarray:
        return np.concatenate(
            [P._R_to_quat_xyzw(P._quat_xyzw_to_R(rng.standard_normal(4))) for _ in range(k)]
        ).astype(np.float32)

    prev = _quats(n_bodies)
    # Antipodal target (same rotation, opposite sign) must flip back to prev.
    aligned = P._align_quat_signs(-prev.copy(), prev)
    assert np.allclose(aligned, prev, atol=1e-6), "antipode must flip to prev's hemisphere"
    # Post-align every body's dot with prev is non-negative (shortest path).
    for b in range(n_bodies):
        s = slice(b * 4, b * 4 + 4)
        assert float(np.dot(aligned[s], prev[s])) >= 0.0
    # A same-hemisphere target is left UNCHANGED (no spurious flip).
    t2 = _quats(n_bodies)
    t2_same = P._align_quat_signs(t2, prev)
    for b in range(n_bodies):
        s = slice(b * 4, b * 4 + 4)
        assert float(np.dot(t2_same[s], prev[s])) >= 0.0
    # Guarded slew stays on the unit sphere; raw antipodal slew shrinks toward 0.
    guarded = P._slew_vec(prev, aligned, 0.05)
    raw = P._slew_vec(prev, -prev.copy(), 0.05)
    for b in range(n_bodies):
        s = slice(b * 4, b * 4 + 4)
        assert abs(float(np.linalg.norm(guarded[s])) - 1.0) < 0.02
        assert float(np.linalg.norm(raw[s])) < float(np.linalg.norm(guarded[s]))


# ---------------------------------------------------------------------------
# Quaternion double-cover canonicalization (2026-08-30).
#
# `vr_3point_orn` / `vr_5point_orn` are RAW xyzw quaternions, and every one of these
# presets trained exclusively on the `w >= 0` branch — na475b60's own normalizer puts
# `vr_5point_orn`'s w at mean +0.73 (hands) / +0.98 (head) / +0.96 (feet) with std <= 0.17,
# i.e. unimodal and nowhere near zero. `q` and `-q` are the same rotation but a ~20-sigma
# difference in obs space, which is why the branch has to be pinned at every producer.
# ---------------------------------------------------------------------------


def _Rz(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


@pytest.mark.parametrize("angle_deg", [0.0, 45.0, 90.0, 120.0, 179.0, 180.0, 181.0, 270.0, 359.0])
def test_R_to_quat_xyzw_always_returns_the_positive_w_branch(angle_deg):
    """Shepperd's ``t > 0`` arm yields ``w > 0`` for free; the three trace-negative arms
    (rotations past ~90 deg) do not, so the guarantee used to be incidental.

    This is the LAST producer on the live-teleop path — the relative orientation clutch
    composes rotation matrices (sign-immune, ``R(q) == R(-q)``) and then re-extracts a
    quaternion here — so whatever branch this picks is what reaches the policy.
    """
    from wbt_wrappers_inference.policies.holosoma_wbt import HolosomaWBTPolicy as P

    R = _Rz(np.deg2rad(angle_deg))
    q = P._R_to_quat_xyzw(R)
    assert q[3] >= 0.0, f"{angle_deg} deg -> w={q[3]:+.4f}, the branch no preset trained on"
    # And it must still be the SAME rotation, i.e. canonicalized by negating the WHOLE
    # quaternion rather than by flipping w (which would mirror the rotation while passing
    # the assertion above).
    np.testing.assert_allclose(P._quat_xyzw_to_R(q), R, atol=1e-5)


def test_R_to_quat_xyzw_round_trips_every_trace_regime():
    """Cover all four Shepperd branches, not just the z-axis family above: pick rotations
    whose dominant diagonal element is x, y, z and whose trace is positive."""
    from wbt_wrappers_inference.policies.holosoma_wbt import HolosomaWBTPolicy as P

    def axis_angle(axis, deg):
        a = np.asarray(axis, dtype=np.float64)
        a = a / np.linalg.norm(a)
        th = np.deg2rad(deg)
        K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
        return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * (K @ K)

    for axis in ((1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (1, 1, 1), (2, -1, 0.5)):
        for deg in (10.0, 100.0, 179.0, 200.0, 300.0):
            R = axis_angle(axis, deg)
            q = P._R_to_quat_xyzw(R)
            assert q[3] >= 0.0, f"axis={axis} {deg} deg -> w={q[3]:+.4f}"
            np.testing.assert_allclose(np.linalg.norm(q), 1.0, atol=1e-5)
            np.testing.assert_allclose(P._quat_xyzw_to_R(q), R, atol=1e-5)


def test_orn_clutch_quat_output_is_canonical_even_from_negated_inputs():
    """The clutch must be immune to the wire's double-cover sign AND emit ``+w``.

    ``_orn_clutch_target_quat`` composes ``R_live * R_live0^T * R_ref`` through matrices, so
    negating any input quaternion cannot change the target rotation. Assert both halves: the
    output is unchanged by input negation, and it lands on the trained branch.
    """
    from wbt_wrappers_inference.policies.holosoma_wbt import HolosomaWBTPolicy as P

    def q_of(R):
        return P._R_to_quat_xyzw(R)

    ref = np.concatenate([q_of(_Rz(np.deg2rad(d))) for d in (170.0, -175.0, 179.0)])
    live = np.concatenate([q_of(_Rz(np.deg2rad(d))) for d in (150.0, -160.0, 178.0)])
    live0 = np.concatenate([q_of(_Rz(np.deg2rad(d))) for d in (155.0, -165.0, 177.0)])

    base = P._orn_clutch_target_quat(P, ref, live, live0)
    for b in range(3):
        assert base[4 * b + 3] >= 0.0, f"body {b}: clutch emitted w={base[4 * b + 3]:+.4f}"

    # Negate each body's live quat in turn — same rotation, so the target must not move.
    for b in range(3):
        flipped = live.copy()
        flipped[4 * b : 4 * b + 4] *= -1.0
        np.testing.assert_allclose(
            P._orn_clutch_target_quat(P, ref, flipped, live0), base, atol=1e-5,
            err_msg=f"body {b}: clutch output depends on the wire's double-cover sign",
        )


def _import_bridge():
    """Import the VIVE bridge with ``pysurvive`` stubbed.

    The real module is injected on PYTHONPATH by the launcher and claims a USB device, so it
    is absent in a plain test env — same stub pattern test_mc_gate_reference_and_viz uses,
    rather than reshaping production code for a test's convenience.
    """
    import sys
    import types

    if "pysurvive" not in sys.modules:
        stub = types.ModuleType("pysurvive")
        stub.SimpleContext = object
        sys.modules["pysurvive"] = stub
    from wbt_realtime_eval.vive import vive_teleop_bridge

    return vive_teleop_bridge


def test_bridge_applies_canonicalization_to_all_three_3pt_wire_quats():
    """Wiring check: the 3pt snapshot must canonicalize BOTH hands AND the head.

    Semantics are covered below; this asserts the bridge actually calls it on every quat it
    puts on the wire. Missing one body is the realistic regression — the head comes from a
    different helper (`_head_command`) and is easy to forget.
    """
    import re
    from pathlib import Path

    src = Path(_import_bridge().__file__).read_text()
    # Anchored on the 3pt PAYLOAD itself (`"mode": "3pt"`), not on the first `if self._is_3pt:`
    # in `snapshot()`. That earlier one only computes `Hp, Ho`, and a non-greedy match from it
    # ends at whichever payload comes first — so adding a SIBLING branch (e.g. `5pt-direct`,
    # 2026-09-10) silently sliced the 3pt block out of the region being searched, and this test
    # failed while the code it guards was untouched. Slicing outward from the 3pt payload keeps
    # the region pinned to the thing under test however many other wires the bridge grows.
    i = src.index('"mode": "3pt",')
    body = src[src.rindex("if self._is_3pt:", 0, i) : src.index('"orientation": orientation_on,', i)]
    assert "_head_command" not in body, "the slice must be the 3pt PAYLOAD, not the compute block"
    for name in ("Lq", "Rq", "Ho"):
        assert re.search(rf"\b{name}\s*=\s*_canon_xyzw\(", body), (
            f"the 3pt wire quat {name} is not canonicalized — it can reach a "
            f"quaternion-orn policy on the -w branch"
        )


def test_bridge_canon_xyzw_is_idempotent_and_a_noop_on_positive_w():
    """The live 3pt wire's canonicalizer: flips only when needed, and never twice."""
    _canon_xyzw = _import_bridge()._canon_xyzw

    pos = np.array([0.1, -0.2, 0.3, 0.9])
    np.testing.assert_array_equal(_canon_xyzw(pos), pos)          # no-op
    np.testing.assert_array_equal(_canon_xyzw(-pos), pos)         # corrects
    np.testing.assert_array_equal(_canon_xyzw(_canon_xyzw(-pos)), pos)  # idempotent
    # w == 0 is the seam; either representative is legal, so it must be left alone rather
    # than oscillating between them.
    seam = np.array([1.0, 0.0, 0.0, 0.0])
    np.testing.assert_array_equal(_canon_xyzw(seam), seam)


def test_live_5pt_teleop_source_canonicalizes_its_wire_quats():
    """The THIRD producer: the motion-completion live wire (`--mode 5pt`).

    `mc_teleop_source._keypoints_5pt`-style code builds `conj(q_pelvis) * q_body` from the
    completion's FK and the readout frame, so its branch is just as unpinned as the bake's
    and the rig FK's. A chest pose near the 180 deg seam sends all five keypoints to
    -identity at once. Source-level, because driving the real source needs a completion
    checkpoint and a tracker stream; the semantics are covered by the bake tests.
    """
    import re
    from pathlib import Path

    from wbt_realtime_eval.vive import mc_teleop_source

    src = Path(mc_teleop_source.__file__).read_text()
    # The emit site: q_local is built, normalized, then swapped wxyz->xyzw onto the wire.
    hits = re.findall(r"orn\.extend\(quat_wxyz_to_xyzw\(q_local\)", src)
    assert hits, "the 5pt live wire's orn emit site moved — re-point this test"
    for m in re.finditer(r"q_local = q_local / max\(.*?\n(.*?)orn\.extend", src, re.S):
        between = m.group(1)
        assert "q_local[0] < 0.0" in between and "q_local = -q_local" in between, (
            "the 5pt live wire emits a quaternion without canonicalizing the double cover; "
            "wxyz means the w component is index 0, not 3"
        )
