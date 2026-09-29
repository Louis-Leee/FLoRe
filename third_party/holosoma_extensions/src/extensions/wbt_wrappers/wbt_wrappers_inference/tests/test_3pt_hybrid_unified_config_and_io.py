"""LOCAL C1 UNIFIED deploy config parity + observe-only / contact_phase wiring.

Hardware-free unit tests (no robot, no live sources, no ONNX file required) that
pin the load-bearing invariants of the ``exp:g1-wbt-3pt-hybrid-unified`` and
``exp:g1-wbt-3pt-hybrid-unified-anneal`` sim2real port (canonical training runs
m3s0pw9i / iss0qs71):

  1. The deploy obs config ``holosoma_3pt_hybrid_unified_wbt`` matches the
     training ground truth (170D, exact obs00..obs13 term order, dims, scales),
     and BOTH preset name keys (R2 + R1 anneal) resolve to the SAME object.
  2. There is deliberately NO ``-layerA-`` alias: routing a unified checkpoint
     through the servo-OFF path would zero the I/Δx channels the actor needs.
  3. The delta vs the 168D 3pt-hybrid layout is EXACTLY ``contact_phase`` (2D)
     inserted right after ``force_meas_local`` — nothing else moves, and the
     pre-existing 168D / 183D / 165D hybrid configs are untouched (purely
     additive port).
  4. ``detect_obs_mode`` returns plain ``"3pt"`` (no root trio ⇒ root-invariant,
     does NOT walk) so the 3pt VIVE override fires and the quaternion orn rep
     (``_vr_orn_6d`` False) is selected.
  5. The runtime obs term-ORDER validator accepts the REAL unified key scheme and
     rejects an equal-width swap / the old 168D preset against it.
  6. ``_resolve_unified_mode`` engages observe-only off the contact_phase term
     alone, forces ``HOLOSOMA_HYBRID_SERVO`` OFF even when requested, and is a
     no-op for every non-unified preset.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("wbt_wrappers_inference")

_UNIFIED = "g1-29dof-holosoma-3pt-hybrid-unified-wbt"
_UNIFIED_ANNEAL = "g1-29dof-holosoma-3pt-hybrid-unified-anneal-wbt"

# The EXACT actor_obs keys the unified training preset emits (wbt_training
# hybrid_unified.py `_actor_obs_local_unified_terms`, zero-padded sorted keys
# obs00..obs13 = the 168D 3pt-hybrid layout with contact_phase inserted at obs05).
#
# obs01 is `vr_3point_pos_tcp` since the 2026-08-03 TCP unification: both hand tracking
# points moved from the 0.18 m VR lever to the 0.242 m fingertip TCP so position and force
# supervise ONE point. Read off run wda3h4ll's baked experiment_config, not inferred. Width
# is unchanged (9D/170D/680D), so this NAME is the only thing separating the two
# generations — pinning it here is what keeps a retired 0.18 m checkpoint from loading.
_UNIFIED_TRAIN_KEYS = [
    "obs00_command_lower_body",
    "obs01_vr_3point_pos_tcp",
    "obs02_vr_3point_orn",
    "obs03_force_cmd_local",
    "obs04_force_meas_local",
    "obs05_contact_phase",
    "obs06_integrator_state",
    "obs07_x_tgt_shift",
    "obs08_motion_ref_ori_b",
    "obs09_projected_gravity",
    "obs10_base_ang_vel",
    "obs11_dof_pos",
    "obs12_dof_vel",
    "obs13_actions",
]

_EXPECTED_TERMS = [
    "command_lower_body",
    "vr_3point_pos_tcp",
    "vr_3point_orn",
    "force_cmd_local",
    "force_meas_local",
    "contact_phase",
    "integrator_state",
    "x_tgt_shift",
    "motion_ref_ori_b",
    "projected_gravity",
    "base_ang_vel",
    "joint_pos",
    "joint_vel",
    "actions",
]


# --- config parity -----------------------------------------------------------
def test_unified_config_is_170d_exact_order():
    from wbt_wrappers_inference.config_values.inference import DEFAULTS

    cfg = DEFAULTS[_UNIFIED].observation
    terms = cfg.obs_dict["actor_obs"]
    assert terms == _EXPECTED_TERMS, terms
    per = sum(cfg.obs_dims[t] for t in terms)
    assert per == 170, per
    assert cfg.history_length_dict["actor_obs"] == 4
    assert per * 4 == 680


def test_full_dims_and_scales_pinned():
    """Pin EVERY term width and scale (the training ground truth) — the sum check
    alone would pass a pair of COMPENSATING dim errors, which also slips the
    runtime total-width gate and the name-only order validator."""
    from wbt_wrappers_inference.config_values.inference import DEFAULTS

    cfg = DEFAULTS[_UNIFIED].observation
    expected_dims = {
        "command_lower_body": 24,
        "vr_3point_pos_tcp": 9,
        "vr_3point_orn": 12,
        "force_cmd_local": 6,
        "force_meas_local": 6,
        "contact_phase": 2,
        "integrator_state": 6,
        "x_tgt_shift": 6,
        "motion_ref_ori_b": 6,
        "projected_gravity": 3,
        "base_ang_vel": 3,
        "joint_pos": 29,
        "joint_vel": 29,
        "actions": 29,
    }
    assert cfg.obs_dims == expected_dims, cfg.obs_dims
    expected_scales = {t: 1.0 for t in expected_dims}
    expected_scales["force_cmd_local"] = 0.04
    expected_scales["force_meas_local"] = 0.04
    # contact_phase is already ∈[0,1] in training (ObsTermCfg scale=1.0) — a scale
    # here would silently squash the label the unified reward gates on.
    assert cfg.obs_scales == expected_scales, cfg.obs_scales


def test_both_unified_keys_resolve_to_same_config():
    """R2 (observe-only) and R1 (anneal) differ ONLY in the training curriculum —
    identical obs, identical deploy path, so one config object serves both."""
    from wbt_wrappers_inference.config_values.inference import DEFAULTS

    assert DEFAULTS[_UNIFIED] is DEFAULTS[_UNIFIED_ANNEAL]


def test_unified_registry_surface_is_exactly_the_four_families():
    """Pin the whole unified registry surface, and pin WHICH keys share a config object.

    There are four unified obs layouts, not one, and the sharing pattern is the point:

    * LOCAL 3pt 170D — TWO keys (R2 + R1 anneal) → ONE config. The R2/R1 split is a
      training curriculum, not an obs difference.
    * GLOBAL-VEL 3pt 185D — likewise two keys → one config.
    * 5pt 184D and DENSE 183D — ONE key each, `-anneal-` only. Both runs train from
      scratch (a 184D/183D actor cannot warm-start off a 170D checkpoint), so there is no
      R2 twin to name and a non-anneal alias would point at a checkpoint that does not
      exist.

    And no `-layerA-` variant anywhere: that omission is a NAMING decision — observe-only
    is keyed on the contact_phase obs term, not the key — so the functional guarantee is
    asserted separately below.
    """
    from wbt_wrappers_inference.config_values.inference import DEFAULTS

    _GV = "g1-29dof-holosoma-3pt-hybrid-global-vel-unified-wbt"
    _GV_ANNEAL = "g1-29dof-holosoma-3pt-hybrid-global-vel-unified-anneal-wbt"
    _FIVE = "g1-29dof-holosoma-5pt-hybrid-unified-anneal-wbt"
    _DENSE = "g1-29dof-holosoma-dense-hybrid-unified-anneal-wbt"

    keys = sorted(k for k in DEFAULTS if "unified" in k)
    assert keys == sorted([_UNIFIED, _UNIFIED_ANNEAL, _GV, _GV_ANNEAL, _FIVE, _DENSE]), keys
    assert not any("layerA" in k for k in keys), keys
    # Four distinct obs layouts behind six keys.
    assert len({id(DEFAULTS[k]) for k in keys}) == 4
    # The twin pairs really are the SAME object (not merely equal configs).
    assert DEFAULTS[_UNIFIED] is DEFAULTS[_UNIFIED_ANNEAL]
    assert DEFAULTS[_GV] is DEFAULTS[_GV_ANNEAL]
    # 5pt and dense are the two motion-completion teleop families and must not alias
    # anything else. That matters concretely: dense is 732D and so is the global-vel twin,
    # so a mistaken alias here would be invisible to every dimension check.
    for solo in (_FIVE, _DENSE):
        assert all(DEFAULTS[solo] is not DEFAULTS[k] for k in keys if k != solo), solo


def test_observe_only_holds_for_any_alias_of_the_unified_obs():
    """The FUNCTIONAL guarantee behind the missing `-layerA-` key: observe-only is
    resolved from the obs TERM, so even a hypothetically-named alias of the unified
    obs config lands in the observe-only path with I/Δx live and Δq suppressed. This
    is what makes the naming decision safe."""
    from wbt_wrappers_inference.config_values.inference import DEFAULTS

    terms = DEFAULTS[_UNIFIED].observation.obs_dict["actor_obs"]
    for alias_servo_request in (False, True):
        p = _unified_stub(obs_terms=terms, servo_requested=alias_servo_request)
        assert p._resolve_unified_mode() is True
        assert p._servo_observe_only is True
        assert p._hybrid_servo is False


def test_inference_bindings_match_sibling_3pt_presets():
    """Reuse the SAME task + robot objects as the sibling 3pt-hybrid preset — only
    the observation differs (mirroring training's `replace(..., observation=...)`)."""
    from wbt_wrappers_inference.config_values.inference import DEFAULTS

    uni = DEFAULTS[_UNIFIED]
    h3 = DEFAULTS["g1-29dof-holosoma-3pt-hybrid-wbt"]
    assert uni.task is h3.task
    assert uni.robot is h3.robot
    assert uni.observation is not h3.observation


def test_unified_delta_vs_168d_is_contact_phase_plus_tcp_hand_point():
    """Structural diff vs the 168D 3pt-hybrid base: EXACTLY two changes, no more.

      1. contact_phase(2) inserted after force_meas_local (168 → 170).
      2. the hand-position slot renamed vr_3point_pos → vr_3point_pos_tcp (same 9D),
         the 2026-08-03 move from the 0.18 m VR lever to the 0.242 m force TCP.

    Asserted as a closed set so a THIRD, unintended divergence fails here rather than
    surfacing as a scrambled obs on hardware.
    """
    from wbt_wrappers_inference.config_values.inference import DEFAULTS

    uni = DEFAULTS[_UNIFIED].observation
    base = DEFAULTS["g1-29dof-holosoma-3pt-hybrid-wbt"].observation
    stripped = [t for t in uni.obs_dict["actor_obs"] if t != "contact_phase"]
    # Undo delta 2 to recover the base list, so delta 1 is checked in isolation.
    assert [("vr_3point_pos" if t == "vr_3point_pos_tcp" else t) for t in stripped] == list(base.obs_dict["actor_obs"])
    # And delta 2 really is only a RENAME: the base's 0.18 m slot is gone, the TCP slot
    # took its index, and the width did not move with it.
    assert "vr_3point_pos" not in uni.obs_dict["actor_obs"]
    assert stripped.index("vr_3point_pos_tcp") == list(base.obs_dict["actor_obs"]).index("vr_3point_pos")
    assert uni.obs_dims["vr_3point_pos_tcp"] == base.obs_dims["vr_3point_pos"] == 9
    assert uni.obs_scales["vr_3point_pos_tcp"] == base.obs_scales["vr_3point_pos"] == 1.0
    # Insertion point: immediately after force_meas_local (training obs05 slot).
    terms = uni.obs_dict["actor_obs"]
    assert terms[terms.index("contact_phase") - 1] == "force_meas_local"
    assert terms[terms.index("contact_phase") + 1] == "integrator_state"
    for t in stripped:
        if t == "vr_3point_pos_tcp":
            continue  # covered above (the base has no such key to compare against)
        assert uni.obs_dims[t] == base.obs_dims[t], t
        assert uni.obs_scales[t] == base.obs_scales[t], t
    assert uni.history_length_dict == base.history_length_dict


def test_tcp_hand_offsets_match_the_force_anchor():
    """The whole point of the TCP unification: the two HAND tracking offsets must equal
    the force stack's TCP anchor exactly, and the head point must be untouched.

    Sourced from force_deploy_math.TCP_OFFSET_X rather than a literal, so a future edit to
    either constant fails here instead of quietly re-splitting position from force. Also
    pins the blast radius: the shared VR_3POINT_OFFSETS every other preset uses is
    unchanged, and the ±0.025 lateral term is genuinely DROPPED (keeping it would leave
    the two anchors 2.5 cm apart, i.e. not unified)."""
    from wbt_wrappers_inference.config_values.robot_config import VR_3POINT_OFFSETS, VR_3POINT_OFFSETS_TCP
    from wbt_wrappers_inference.utils.force_deploy_math import TCP_OFFSET_X

    assert VR_3POINT_OFFSETS_TCP[0] == [TCP_OFFSET_X, 0.0, 0.0]
    assert VR_3POINT_OFFSETS_TCP[1] == [TCP_OFFSET_X, 0.0, 0.0]
    assert VR_3POINT_OFFSETS_TCP[0] == VR_3POINT_OFFSETS_TCP[1], "hands must be laterally symmetric"
    assert VR_3POINT_OFFSETS_TCP[2] == VR_3POINT_OFFSETS[2] == [0.0, 0.0, 0.35], "head point must not move"
    # Blast radius: the default lever table every other 3pt/2pt preset shares is intact.
    assert VR_3POINT_OFFSETS[0] == [0.18, -0.025, 0.0]
    assert VR_3POINT_OFFSETS[1] == [0.18, 0.025, 0.0]
    # The move is ~6.7 cm per hand — large enough that a silent mismatch is a real defect,
    # which is why the term name (not the width) is the discriminator.
    shift = np.linalg.norm(np.array(VR_3POINT_OFFSETS_TCP[0]) - np.array(VR_3POINT_OFFSETS[0]))
    assert 0.06 < shift < 0.07, shift


def test_vr_offsets_dispatch_follows_the_checkpoint_term():
    """``_vr_offsets()`` must pick the geometry from the obs TERM, not a preset flag —
    that is what makes the two checkpoint generations mutually exclusive at load time.

    Also covers the reason it is a helper at all: three call sites (obs assembly, the
    pre-arm pose gate, the absolute arm clutch) must agree by construction. If they ever
    diverge, the clutch anchors on one lever while the obs commands the other and arming
    steps the hands ~6.7 cm.
    """
    from wbt_wrappers_inference.config_values.robot_config import VR_3POINT_OFFSETS, VR_3POINT_OFFSETS_TCP
    from wbt_wrappers_inference.policies.holosoma_wbt import HolosomaWBTPolicy

    def _stub(is_3pt, terms):
        p = object.__new__(HolosomaWBTPolicy)
        p._is_3pt = is_3pt
        p._is_2pt = not is_3pt
        p._vr_pos_tcp = any("vr_3point_pos_tcp" in t for t in terms)
        return p

    assert np.allclose(_stub(True, _EXPECTED_TERMS)._vr_offsets(), VR_3POINT_OFFSETS_TCP)
    assert np.allclose(_stub(True, ["vr_3point_pos", "vr_3point_orn"])._vr_offsets(), VR_3POINT_OFFSETS)
    # obsNN_-prefixed training keys must resolve identically (substring match).
    assert np.allclose(_stub(True, _UNIFIED_TRAIN_KEYS)._vr_offsets(), VR_3POINT_OFFSETS_TCP)


def test_existing_hybrid_configs_unchanged():
    """The port must be purely additive: 168D / 183D / 165D all intact, and none of
    them may pick up a contact_phase slot."""
    from wbt_wrappers_inference.config_values.inference import DEFAULTS

    for key, width in (
        ("g1-29dof-holosoma-3pt-hybrid-wbt", 168),
        ("g1-29dof-holosoma-3pt-hybrid-global-vel-wbt", 183),
        ("g1-29dof-holosoma-2pt-hybrid-wbt", 165),
    ):
        cfg = DEFAULTS[key].observation
        per = sum(cfg.obs_dims[t] for t in cfg.obs_dict["actor_obs"])
        assert per == width, (key, per)
        assert per * 4 == width * 4
        assert "contact_phase" not in cfg.obs_dict["actor_obs"], key


def test_unified_obs_mode_is_plain_3pt():
    """No motion_ref_pos_b ⇒ plain "3pt": root-invariant (does NOT walk), the 3pt
    VIVE override fires, and the quaternion orn rep is selected (not rot6d)."""
    from wbt_wrappers_inference.config_values.inference import DEFAULTS
    from wbt_wrappers_inference.utils.obs_utils import detect_obs_mode

    terms = DEFAULTS[_UNIFIED].observation.obs_dict["actor_obs"]
    assert detect_obs_mode(terms) == "3pt"
    assert not any("vr_3point_orn_6d" in t for t in terms)
    assert not any("motion_ref_pos_b" in t or "base_lin_vel" in t for t in terms)


# --- runtime term-order validation vs the REAL unified key scheme --------------
def _exp_config(keys):
    return {"observation": {"groups": {"actor_obs": {"terms": {k: {} for k in keys}}}}}


def test_term_order_validation_accepts_unified_scheme():
    from wbt_wrappers_inference.config_values.inference import DEFAULTS
    from wbt_wrappers_inference.policies.holosoma_wbt import HolosomaWBTPolicy as P

    stub = object.__new__(P)
    stub._obs_terms = list(DEFAULTS[_UNIFIED].observation.obs_dict["actor_obs"])
    P._validate_obs_term_order(stub, _exp_config(_UNIFIED_TRAIN_KEYS))


def test_term_order_validation_rejects_swap_against_unified():
    """Swap the two equal-width (6D) servo terms — the total-width gate cannot see
    this, only the term-ORDER validator can."""
    from wbt_wrappers_inference.config_values.inference import DEFAULTS
    from wbt_wrappers_inference.policies.holosoma_wbt import HolosomaWBTPolicy as P

    terms = list(DEFAULTS[_UNIFIED].observation.obs_dict["actor_obs"])
    i, j = terms.index("integrator_state"), terms.index("x_tgt_shift")
    terms[i], terms[j] = terms[j], terms[i]
    stub = object.__new__(P)
    stub._obs_terms = terms
    with pytest.raises(ValueError, match="term-ORDER mismatch"):
        P._validate_obs_term_order(stub, _exp_config(_UNIFIED_TRAIN_KEYS))


def test_term_order_validation_rejects_168d_preset_against_unified():
    """Loading a unified ckpt with the OLD 168D preset must fail LOUDLY (13 vs 14
    terms) — the widths differ too (168 vs 170), but this trips first."""
    from wbt_wrappers_inference.config_values.inference import DEFAULTS
    from wbt_wrappers_inference.policies.holosoma_wbt import HolosomaWBTPolicy as P

    stub = object.__new__(P)
    stub._obs_terms = list(DEFAULTS["g1-29dof-holosoma-3pt-hybrid-wbt"].observation.obs_dict["actor_obs"])
    with pytest.raises(ValueError, match="term-COUNT mismatch"):
        P._validate_obs_term_order(stub, _exp_config(_UNIFIED_TRAIN_KEYS))


def test_term_order_validation_rejects_unified_preset_against_168d_ckpt():
    """And the reverse: a unified preset against a plain 3pt-hybrid checkpoint."""
    from wbt_wrappers_inference.config_values.inference import DEFAULTS
    from wbt_wrappers_inference.policies.holosoma_wbt import HolosomaWBTPolicy as P

    base_keys = [k for k in _UNIFIED_TRAIN_KEYS if "contact_phase" not in k]
    stub = object.__new__(P)
    stub._obs_terms = list(DEFAULTS[_UNIFIED].observation.obs_dict["actor_obs"])
    with pytest.raises(ValueError, match="term-COUNT mismatch"):
        P._validate_obs_term_order(stub, _exp_config(base_keys))


# --- observe-only resolution ---------------------------------------------------
def _unified_stub(*, obs_terms, servo_requested):
    from loguru import logger as _loguru

    from wbt_wrappers_inference.policies.holosoma_wbt import HolosomaWBTPolicy as P

    p = object.__new__(P)
    p.logger = _loguru
    p._obs_terms = list(obs_terms)
    p._hybrid_servo = servo_requested
    p._servo_observe_only = False
    p._contact_phase_theta_on = 2.0
    p._contact_phase_theta_off = 1.5
    p._contact_phase_tau_s = 0.05
    p._contact_phase_warmup_steps = 5
    return p


def test_resolve_unified_mode_engages_observe_only():
    from wbt_wrappers_inference.config_values.inference import DEFAULTS

    terms = DEFAULTS[_UNIFIED].observation.obs_dict["actor_obs"]
    p = _unified_stub(obs_terms=terms, servo_requested=False)
    assert p._resolve_unified_mode() is True
    assert p._servo_observe_only is True
    assert p._hybrid_servo is False


def test_resolve_unified_mode_overrides_requested_servo():
    """HOLOSOMA_HYBRID_SERVO=1 on a unified preset must be IGNORED — an R1
    checkpoint's baked servo_observe_only=False cannot re-enable Δq either."""
    from wbt_wrappers_inference.config_values.inference import DEFAULTS

    terms = DEFAULTS[_UNIFIED].observation.obs_dict["actor_obs"]
    p = _unified_stub(obs_terms=terms, servo_requested=True)
    assert p._resolve_unified_mode() is True
    assert p._hybrid_servo is False, "unified must never inject Δq, even when SERVO=1"
    assert p._servo_observe_only is True


@pytest.mark.parametrize(
    "preset",
    [
        "g1-29dof-holosoma-3pt-hybrid-wbt",
        "g1-29dof-holosoma-3pt-hybrid-global-vel-wbt",
        "g1-29dof-holosoma-2pt-hybrid-wbt",
        "g1-29dof-holosoma-wbt",
    ],
)
def test_resolve_unified_mode_is_noop_for_non_unified(preset):
    """Every pre-existing preset keeps its requested servo setting untouched — the
    regression guard that this port cannot disturb the servo-ON deploys."""
    from wbt_wrappers_inference.config_values.inference import DEFAULTS

    terms = DEFAULTS[preset].observation.obs_dict["actor_obs"]
    p = _unified_stub(obs_terms=terms, servo_requested=True)
    assert p._resolve_unified_mode() is False
    assert p._hybrid_servo is True, "non-unified presets must honor HOLOSOMA_HYBRID_SERVO"
    assert p._servo_observe_only is False


def test_resolve_unified_mode_is_idempotent():
    from wbt_wrappers_inference.config_values.inference import DEFAULTS

    terms = DEFAULTS[_UNIFIED].observation.obs_dict["actor_obs"]
    p = _unified_stub(obs_terms=terms, servo_requested=True)
    for _ in range(3):
        assert p._resolve_unified_mode() is True
    assert p._hybrid_servo is False and p._servo_observe_only is True


def test_contact_phase_thresholds_match_training_defaults():
    """The deploy defaults MUST equal wbt_training HybridContactConfig's
    (theta_on 2.0 N, theta_off 1.5 N, tau 0.05 s, warmup 5) — φ is a TRAINED input,
    so a threshold drift is a silent distribution shift, and the label is what the
    unified reward gates on. Asserted on the module constants, which are the single
    source of truth for both the __init__ env defaults and the getattr fallbacks."""
    import wbt_wrappers_inference.policies.holosoma_wbt as hw

    assert hw._CONTACT_PHASE_THETA_ON == 2.0
    assert hw._CONTACT_PHASE_THETA_OFF == 1.5
    assert hw._CONTACT_PHASE_TAU_S == 0.05
    assert hw._CONTACT_PHASE_WARMUP_STEPS == 5
    # Hysteresis must be a real dead band, and θ_on must sit clear of the trained
    # sensor noise floor (sensor_noise_std=0.5 N ⇒ θ_on = 4σ).
    assert hw._CONTACT_PHASE_THETA_OFF < hw._CONTACT_PHASE_THETA_ON
    # τ must be > the control period, else β ≥ 1 and the "LPF" is a hard switch.
    assert hw._CONTACT_PHASE_TAU_S > 1.0 / 50.0


def test_lpf_uses_the_control_tick_dt_not_the_substep():
    """The deploy LPF must run ONCE per 20 ms control tick, which reproduces
    training's four 5 ms substep applications because the recursion is separable:
    (1-beta_5ms)^4 == (1-beta_20ms). Asserting the identity alone would be vacuous
    (it holds for ANY tau), so drive the REAL _update_contact_phase and check its
    effective beta is the 20 ms one and NOT the 5 ms one — the mutation that matters.
    """
    from types import SimpleNamespace

    from wbt_wrappers_inference.policies.holosoma_wbt import HolosomaWBTPolicy as P

    beta_tick = 1.0 - np.exp(-0.020 / 0.05)
    beta_sub = 1.0 - np.exp(-0.005 / 0.05)
    assert abs((1.0 - beta_sub) ** 4 - (1.0 - beta_tick)) < 1e-12  # the separability claim
    assert abs(beta_tick - beta_sub) > 0.2  # ... and the two are far apart per-application

    p = object.__new__(P)
    p.config = SimpleNamespace(task=SimpleNamespace(rl_rate=50.0))
    p.use_policy_action = True
    p._force_meas_pelvis = np.array([[0.0, 0.0, 6.0], [0.0, 0.0, 0.0]])  # latch ON
    p._contact_phase_warmup_steps = 0  # skip the parity window; no tare burst attribute
    p._update_contact_phase()
    got = float(p._contact_phase[0])
    assert abs(got - beta_tick) < 1e-12, (got, beta_tick)
    assert abs(got - beta_sub) > 0.2, "the implementation is using the substep dt"


# --- wiring guards: the two call sites the behavioural tests cannot reach --------
#
# Both of these are SOURCE-level assertions, deliberately. The stub-based tests above
# exercise `_resolve_unified_mode` / `_update_force_meas_and_servo` directly, so they
# stay green even if the call site that invokes them in production is deleted — and
# both call sites are exactly the two-layer defence that keeps a unified checkpoint
# from being handed a second actuator. Building a real `setup_policy` / `rl_inference`
# fixture needs an ONNX file + a robot interface (none exist in this suite), so pin the
# wiring textually and let the behavioural tests own the semantics.
def _code_only(text: str) -> str:
    """Strip comments and docstring-ish prose so a source guard cannot be satisfied by
    a comment mentioning the symbol (or by a commented-out call)."""
    import io
    import tokenize

    out = []
    try:
        for tok in tokenize.generate_tokens(io.StringIO(text).readline):
            if tok.type in (tokenize.COMMENT, tokenize.STRING):
                continue
            out.append(tok.string)
    except (tokenize.TokenError, IndentationError):
        return "\n".join(ln.split("#", 1)[0] for ln in text.splitlines())
    return " ".join(out)


def _src(fn):
    import inspect
    import textwrap

    return _code_only(textwrap.dedent(inspect.getsource(fn)))


def _calls_attr(fn, attr: str) -> bool:
    """True iff `fn`'s AST contains a real Call to `self.<attr>(...)`."""
    import ast
    import inspect
    import textwrap

    tree = ast.parse(textwrap.dedent(inspect.getsource(fn)))
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == attr:
            return True
    return False


def test_setup_policy_invokes_the_unified_resolution():
    """`setup_policy` MUST call `_resolve_unified_mode()`. Without it a unified
    checkpoint keeps whatever HOLOSOMA_HYBRID_SERVO the operator set, so
    `MODE=3pt-unified-anneal SERVO=1` would inject Δq into a policy that never
    trained alongside a second actuator — the exact failure this port prevents."""
    from wbt_wrappers_inference.policies.holosoma_wbt import HolosomaWBTPolicy as P

    assert _calls_attr(P.setup_policy, "_resolve_unified_mode")


def test_dq_injection_stays_gated_on_hybrid_servo():
    """The OUTER defence layer, checked STRUCTURALLY (reformat-proof): the statement in
    `rl_inference` that adds `_servo_dq` to `scaled_policy_action` must sit inside an
    `if` whose condition references `_hybrid_servo` — which `_resolve_unified_mode`
    forces False for every unified preset. The INNER layer (zeroing `_servo_dq` in
    observe-only) is pinned behaviourally in test_vive_hybrid_3pt_unified_teleop.py, and
    both must hold independently.

    Tolerates a reformat, a renamed local, or an ADDITIONAL tightening term in the
    condition; fails if the guard is removed, inverted, or the injection is hoisted out.
    """
    import ast
    import inspect
    import textwrap

    from wbt_wrappers_inference.policies.holosoma_wbt import HolosomaWBTPolicy as P

    tree = ast.parse(textwrap.dedent(inspect.getsource(P.rl_inference)))

    def _mentions_servo_dq(node) -> bool:
        return any(isinstance(n, ast.Attribute) and n.attr == "_servo_dq" for n in ast.walk(node))

    injections = [
        n
        for n in ast.walk(tree)
        if isinstance(n, (ast.Assign, ast.AugAssign))
        and _mentions_servo_dq(n)
        and any(isinstance(t, ast.Attribute) and t.attr == "scaled_policy_action" for t in ast.walk(n))
    ]
    assert injections, "no `scaled_policy_action (+)= ... _servo_dq ...` statement found in rl_inference"

    guarded = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        cond = {n.attr for n in ast.walk(node.test) if isinstance(n, ast.Attribute)}
        if "_hybrid_servo" not in cond:
            continue
        body_nodes = {id(n) for stmt in node.body for n in ast.walk(stmt)}
        guarded.extend(inj for inj in injections if id(inj) in body_nodes)
    assert len(guarded) == len(injections), (
        "every Δq injection must be inside an `if` testing self._hybrid_servo "
        f"({len(guarded)}/{len(injections)} guarded)"
    )


def test_wiring_guards_ignore_comments():
    """The source guards below match CODE, not prose: strip comments first so a
    commented-out call cannot satisfy them."""
    assert "_resolve_unified_mode" not in _code_only("# self._resolve_unified_mode()\n")
    assert "_resolve_unified_mode" in _code_only("self._resolve_unified_mode()\n")


def test_reset_motion_state_resets_the_contact_phase():
    """STOP → re-START must clear a latched φ. `_reset_motion_state` is the single
    reset funnel (STOP, clip start, policy switch, restore), and it cannot be driven
    from a stub (it touches interface/timestep state), so pin the wiring here; the
    reset semantics themselves are asserted in the teleop suite."""
    from wbt_wrappers_inference.policies.holosoma_wbt import HolosomaWBTPolicy as P

    assert _calls_attr(P._reset_motion_state, "_reset_contact_phase_state")


# --- __init__ env parsing + startup validation ----------------------------------
#
# `__init__` needs a full hardware config, so mirror its exact parse+validate block
# (the convention test_force_deploy.py::test_force_meas_ema_alpha_rejects_out_of_range
# already establishes) and assert the guards reject every value that would otherwise
# reach the ONNX / the motors.
def _parse_contact_phase_env(env):
    """Call the REAL parser/validator used by __init__ (no hand-copied mirror — a
    mirror would stay green while the production guards drifted away)."""
    from wbt_wrappers_inference.policies.holosoma_wbt import parse_contact_phase_cfg

    return parse_contact_phase_cfg(env)


def test_init_uses_the_shared_contact_phase_parser():
    """__init__ must delegate to `parse_contact_phase_cfg`, so the validation the tests
    below exercise is the validation a real deploy runs."""
    import inspect

    from wbt_wrappers_inference.policies.holosoma_wbt import HolosomaWBTPolicy as P

    assert "parse_contact_phase_cfg(os.environ)" in inspect.getsource(P.__init__)


def test_contact_phase_env_defaults_come_from_the_module_constants():
    """The empty environment must yield exactly the training values — this is the
    "single source of truth" half that a constants-only assertion cannot cover."""
    assert _parse_contact_phase_env({}) == (2.0, 1.5, 0.05, 5)


@pytest.mark.parametrize(
    "env",
    [
        # NaN slips past every bare comparison (`nan <= 0` is False) and then rides
        # beta → φ → obs → ONNX → motors with no downstream guard.
        {"HOLOSOMA_CONTACT_PHASE_TAU_S": "nan"},
        {"HOLOSOMA_CONTACT_PHASE_THETA_ON": "nan"},
        {"HOLOSOMA_CONTACT_PHASE_THETA_OFF": "nan"},
        # inf tau ⇒ beta 0 ⇒ φ silently frozen at 0 forever.
        {"HOLOSOMA_CONTACT_PHASE_TAU_S": "inf"},
        # A non-positive THETA_OFF makes `‖F‖ < θ_off` unsatisfiable ⇒ one touch
        # latches φ=1 for the rest of the run.
        {"HOLOSOMA_CONTACT_PHASE_THETA_OFF": "0"},
        {"HOLOSOMA_CONTACT_PHASE_THETA_OFF": "-1"},
        {"HOLOSOMA_CONTACT_PHASE_THETA_ON": "0", "HOLOSOMA_CONTACT_PHASE_THETA_OFF": "0"},
        # Inverted hysteresis.
        {"HOLOSOMA_CONTACT_PHASE_THETA_ON": "1.0"},
        # tau <= 0 ⇒ division blowup / hard switch.
        {"HOLOSOMA_CONTACT_PHASE_TAU_S": "0"},
        {"HOLOSOMA_CONTACT_PHASE_TAU_S": "-0.05"},
        # A negative warmup DISABLES the post-START suppression.
        {"HOLOSOMA_CONTACT_PHASE_WARMUP_STEPS": "-1"},
    ],
)
def test_contact_phase_env_rejects_unsafe_values(env):
    with pytest.raises(ValueError):
        _parse_contact_phase_env(env)


@pytest.mark.parametrize(
    "env,expected",
    [
        ({"HOLOSOMA_CONTACT_PHASE_THETA_ON": "3.0"}, (3.0, 1.5, 0.05, 5)),
        ({"HOLOSOMA_CONTACT_PHASE_TAU_S": "0.1"}, (2.0, 1.5, 0.1, 5)),
        ({"HOLOSOMA_CONTACT_PHASE_WARMUP_STEPS": "100"}, (2.0, 1.5, 0.05, 100)),
        # Equal edges = a degenerate but legal (zero-width) dead band.
        ({"HOLOSOMA_CONTACT_PHASE_THETA_OFF": "2.0"}, (2.0, 2.0, 0.05, 5)),
        ({"HOLOSOMA_CONTACT_PHASE_WARMUP_STEPS": "0"}, (2.0, 1.5, 0.05, 0)),
    ],
)
def test_contact_phase_env_accepts_valid_overrides(env, expected):
    assert _parse_contact_phase_env(env) == expected


# --- the observe-only integrator command gate (blocker-1's mitigation) -----------
def test_integ_cmd_gate_defaults_ON():
    """DEFAULT ON is the whole point: with the gate off a sustained free-space F/T
    offset above the 2 N contact gate winds the observe-only integrator to its
    elementwise clip — tens of centimetres of commanded Δx the unified policy chases,
    with the runaway valve never firing. A flipped default is a silent hardware hazard."""
    from wbt_wrappers_inference.policies.holosoma_wbt import parse_unified_integ_cmd_gate

    assert parse_unified_integ_cmd_gate({}) is True


@pytest.mark.parametrize("raw", ["1", "true", "TRUE", "yes", "on", " On "])
def test_integ_cmd_gate_truthy(raw):
    from wbt_wrappers_inference.policies.holosoma_wbt import parse_unified_integ_cmd_gate

    assert parse_unified_integ_cmd_gate({"HOLOSOMA_UNIFIED_INTEGRATOR_CMD_GATE": raw}) is True


@pytest.mark.parametrize("raw", ["0", "false", "no", "off", " OFF "])
def test_integ_cmd_gate_explicit_off(raw):
    """Disabling it must still be POSSIBLE (the fidelity A/B) — just never accidental."""
    from wbt_wrappers_inference.policies.holosoma_wbt import parse_unified_integ_cmd_gate

    assert parse_unified_integ_cmd_gate({"HOLOSOMA_UNIFIED_INTEGRATOR_CMD_GATE": raw}) is False


def test_windup_warn_n_defaults_to_none_and_parses_numbers():
    """``HOLOSOMA_UNIFIED_WINDUP_WARN_N``: unset/empty → None (half of the clip at runtime),
    a number → that many newtons, 0 → alarm off. The launcher forwards WINDUP_WARN_N here."""
    from wbt_wrappers_inference.policies.holosoma_wbt import parse_unified_windup_warn_n

    assert parse_unified_windup_warn_n({}) is None
    assert parse_unified_windup_warn_n({"HOLOSOMA_UNIFIED_WINDUP_WARN_N": ""}) is None
    assert parse_unified_windup_warn_n({"HOLOSOMA_UNIFIED_WINDUP_WARN_N": " 200 "}) == 200.0
    assert parse_unified_windup_warn_n({"HOLOSOMA_UNIFIED_WINDUP_WARN_N": "0"}) == 0.0
    assert parse_unified_windup_warn_n({"HOLOSOMA_UNIFIED_WINDUP_WARN_N": "2.5e1"}) == 25.0


@pytest.mark.parametrize("raw", ["20N", "-1", "nan", "inf", "off", "twenty"])
def test_windup_warn_n_rejects_a_non_number(raw):
    """A typo must not silently arm or disarm the alarm."""
    from wbt_wrappers_inference.policies.holosoma_wbt import parse_unified_windup_warn_n

    with pytest.raises(ValueError, match="HOLOSOMA_UNIFIED_WINDUP_WARN_N"):
        parse_unified_windup_warn_n({"HOLOSOMA_UNIFIED_WINDUP_WARN_N": raw})


@pytest.mark.parametrize("raw", ["y", "n", "2", "-1", "enabled", "", "tru", "off ff"])
def test_integ_cmd_gate_rejects_typos_instead_of_falling_to_off(raw):
    """STRICT parse, unlike the other boolean knobs in this file: their unrecognized
    fallback lands on the safe side, this one's would land on the UNSAFE side."""
    from wbt_wrappers_inference.policies.holosoma_wbt import parse_unified_integ_cmd_gate

    with pytest.raises(ValueError):
        parse_unified_integ_cmd_gate({"HOLOSOMA_UNIFIED_INTEGRATOR_CMD_GATE": raw})


def test_init_uses_the_shared_cmd_gate_parser():
    import inspect

    from wbt_wrappers_inference.policies.holosoma_wbt import HolosomaWBTPolicy as P

    assert "parse_unified_integ_cmd_gate(os.environ)" in inspect.getsource(P.__init__)


# --- ONNX thread policy (CPU burn, NOT numerics) ------------------------------
# `InferenceSession(path)` sizes its intra-op pool to the core count and lets it SPIN
# between calls. Right for a batch job (3647 it/s vs 923 at intra=1), wrong for the 50 Hz
# rig (22.7 cores + p99 7.6 ms, versus 0.12 cores + p99 3.1 ms at intra=1). So the LIBRARY
# default stays legacy — headless eval must keep its throughput — and the DEPLOY LAUNCHER
# is what opts into the low-CPU pair. These tests pin that split and the strict parse.
def test_ort_thread_policy_defaults_to_the_legacy_pool_so_batch_eval_is_unchanged():
    from wbt_wrappers_inference.policies.holosoma_wbt import parse_ort_thread_policy

    # 0 = onnxruntime sizes the pool from the core count; spinning on. Both are its own
    # defaults, i.e. identical to the pre-2026-08-11 bare InferenceSession(path).
    assert parse_ort_thread_policy({}) == (0, True)


def test_ort_thread_policy_low_cpu_pair_is_what_the_deploy_launcher_sets():
    from wbt_wrappers_inference.policies.holosoma_wbt import parse_ort_thread_policy

    env = {"HOLOSOMA_ORT_INTRA_THREADS": "1", "HOLOSOMA_ORT_SPIN": "0"}
    assert parse_ort_thread_policy(env) == (1, False)


@pytest.mark.parametrize("raw", ["2", " 4 ", "24"])
def test_ort_thread_policy_accepts_a_larger_pool(raw):
    from wbt_wrappers_inference.policies.holosoma_wbt import parse_ort_thread_policy

    assert parse_ort_thread_policy({"HOLOSOMA_ORT_INTRA_THREADS": raw, "HOLOSOMA_ORT_SPIN": "0"}) == (int(raw), False)


@pytest.mark.parametrize("raw", ["-1", "1.5", "all", "auto", "1 2"])
def test_ort_thread_policy_rejects_a_bad_thread_count(raw):
    from wbt_wrappers_inference.policies.holosoma_wbt import parse_ort_thread_policy

    with pytest.raises(ValueError):
        parse_ort_thread_policy({"HOLOSOMA_ORT_INTRA_THREADS": raw})


@pytest.mark.parametrize("raw", ["y", "2", "spin", "tru"])
def test_ort_thread_policy_rejects_a_bad_spin_flag(raw):
    from wbt_wrappers_inference.policies.holosoma_wbt import parse_ort_thread_policy

    with pytest.raises(ValueError):
        parse_ort_thread_policy({"HOLOSOMA_ORT_SPIN": raw})


@pytest.mark.parametrize("raw", ["", "   "])
def test_ort_thread_policy_treats_an_empty_value_as_unset(raw):
    """Same convention as the force-gate threshold above: blank means "not set"."""
    from wbt_wrappers_inference.policies.holosoma_wbt import parse_ort_thread_policy

    assert parse_ort_thread_policy({"HOLOSOMA_ORT_INTRA_THREADS": raw, "HOLOSOMA_ORT_SPIN": raw}) == (0, True)


def test_ort_session_options_default_mirror_onnxruntimes_own_defaults():
    """The legacy path must be a no-op: same values a bare SessionOptions() carries, so
    routing every caller through build_ort_session_options changes nothing for them."""
    onnxruntime = pytest.importorskip("onnxruntime")
    from wbt_wrappers_inference.policies.holosoma_wbt import build_ort_session_options

    bare, opts = onnxruntime.SessionOptions(), build_ort_session_options({})
    assert opts.intra_op_num_threads == bare.intra_op_num_threads == 0
    assert opts.inter_op_num_threads == bare.inter_op_num_threads == 0
    assert opts.execution_mode == bare.execution_mode == onnxruntime.ExecutionMode.ORT_SEQUENTIAL
    # Graph optimization is deliberately NOT set — it must stay at onnxruntime's default
    # so the executed graph, and therefore the numerics, is untouched.
    assert opts.graph_optimization_level == bare.graph_optimization_level


def test_ort_session_options_low_cpu_pair_is_single_threaded_and_sequential():
    onnxruntime = pytest.importorskip("onnxruntime")
    from wbt_wrappers_inference.policies.holosoma_wbt import build_ort_session_options

    opts = build_ort_session_options({"HOLOSOMA_ORT_INTRA_THREADS": "1", "HOLOSOMA_ORT_SPIN": "0"})
    assert opts.intra_op_num_threads == 1
    assert opts.execution_mode == onnxruntime.ExecutionMode.ORT_SEQUENTIAL
    assert opts.graph_optimization_level == onnxruntime.SessionOptions().graph_optimization_level


def test_setup_policy_actually_passes_the_session_options():
    """Guard the wiring: a bare InferenceSession(path) here silently ignores the knob."""
    import inspect

    from wbt_wrappers_inference.policies.holosoma_wbt import HolosomaWBTPolicy as P

    src = inspect.getsource(P.setup_policy)
    assert "sess_options=build_ort_session_options(os.environ)" in src
    # providers must stay unset so the provider choice is byte-identical to before.
    assert "providers=" not in src
