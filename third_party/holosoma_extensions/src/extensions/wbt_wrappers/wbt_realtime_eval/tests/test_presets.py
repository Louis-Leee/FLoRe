"""Verify the preset registry covers every holosoma-*-wbt config and that
each preset's per-frame + total dim matches
``wbt_wrappers_inference.config_values.observation``.

If this test fails, either:

* a new training preset has been added upstream (mirror it in :mod:`presets`),
* an obs term width changed upstream (update :data:`presets._TERMS`), or
* a preset's term order drifted (re-sync the obs_keys tuple).

In all three cases the expected fix is in :mod:`wbt_realtime_eval.presets`.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from wbt_realtime_eval.presets import PRESETS, get_preset

# Map preset name -> the source-of-truth ObservationConfig name.
_PRESET_TO_OBS_ATTR = {
    "g1-29dof-holosoma-wbt": "holosoma_wbt",
    "g1-29dof-holosoma-2pt-wbt": "holosoma_2pt_wbt",
    "g1-29dof-holosoma-2pt-force-wbt": "holosoma_2pt_force_wbt",
    "g1-29dof-holosoma-2pt-force-ext-wbt": "holosoma_2pt_force_ext_wbt",
    # Both 2pt-hybrid presets share the SAME 165D obs config (servo on/off is a
    # runtime flag, not an obs difference).
    "g1-29dof-holosoma-2pt-hybrid-wbt": "holosoma_2pt_hybrid_wbt",
    "g1-29dof-holosoma-2pt-hybrid-layerA-wbt": "holosoma_2pt_hybrid_wbt",
    # Both 3pt-hybrid presets share the SAME 168D obs config (2pt-hybrid layout with
    # vr_2point → vr_3point). servo on/off is a runtime flag, not an obs difference.
    "g1-29dof-holosoma-3pt-hybrid-wbt": "holosoma_3pt_hybrid_wbt",
    "g1-29dof-holosoma-3pt-hybrid-layerA-wbt": "holosoma_3pt_hybrid_wbt",
    # Both 3pt-hybrid-global-vel presets share the SAME 183D obs config (3pt-hybrid +
    # root pos/vel + vr_3point_orn quat→6D). servo on/off is a runtime flag, not obs.
    "g1-29dof-holosoma-3pt-hybrid-global-vel-wbt": "holosoma_3pt_hybrid_global_vel_wbt",
    "g1-29dof-holosoma-3pt-hybrid-global-vel-layerA-wbt": "holosoma_3pt_hybrid_global_vel_wbt",
    # Both C1 unified twins (R2 observe-only / R1 anneal) share the SAME 185D obs config
    # (183D + soft contact_phase). Training route is a curriculum difference, not obs.
    "g1-29dof-holosoma-3pt-hybrid-global-vel-unified-wbt": "holosoma_3pt_hybrid_global_vel_unified_wbt",
    "g1-29dof-holosoma-3pt-hybrid-global-vel-unified-anneal-wbt": "holosoma_3pt_hybrid_global_vel_unified_wbt",
    # Both LOCAL C1 unified twins share the SAME 170D obs config (pelvis-local 168D
    # 3pt-hybrid + soft contact_phase; no root trio, quaternion vr_3point_orn).
    "g1-29dof-holosoma-3pt-hybrid-unified-wbt": "holosoma_3pt_hybrid_unified_wbt",
    "g1-29dof-holosoma-3pt-hybrid-unified-anneal-wbt": "holosoma_3pt_hybrid_unified_wbt",
    # The 5pt / DENSE unified anneal twins (2026-08-27). Only the R1 anneal route was
    # trained for these two (runs dlcxc59d / 7r54008a), so unlike the families above there
    # is no observe-only sibling sharing the config — one preset each.
    "g1-29dof-holosoma-5pt-hybrid-unified-anneal-wbt": "holosoma_5pt_hybrid_unified_wbt",
    "g1-29dof-holosoma-dense-hybrid-unified-anneal-wbt": "holosoma_dense_hybrid_unified_wbt",
    "g1-29dof-holosoma-3pt-wbt": "holosoma_3pt_wbt",
    "g1-29dof-holosoma-2pt-global-wbt": "holosoma_2pt_global_wbt",
    "g1-29dof-holosoma-3pt-global-wbt": "holosoma_3pt_global_wbt",
    "g1-29dof-holosoma-global-wbt": "holosoma_global_wbt",
    "g1-29dof-holosoma-global-vel-wbt": "holosoma_global_vel_wbt",
}

# Live VIVE-teleop presets have NO 1:1 upstream ObservationConfig — they reuse a
# baked preset's layout but source vr_2point_pos/orn LIVE from the bridge (so the
# term KEYS are the _live variants, with identical widths). They are excluded
# from the upstream-config coverage map and validated separately below
# (test_live_presets_mirror_baked) against the baked preset they derive from.
_LIVE_PRESETS = {
    "g1-29dof-holosoma-2pt-wbt-live": "g1-29dof-holosoma-2pt-wbt",
    "g1-29dof-holosoma-2pt-force-wbt-live": "g1-29dof-holosoma-2pt-force-wbt",
    # The BASE tracker (exp:g1-wbt). Unlike the three MC live-only presets below, this one
    # DOES have a baked twin in the registry — the 628D replay preset that is already
    # verified on recorded takes — so it belongs here, where it is mirrored against that
    # twin term-by-term. That is the check worth having: it pins live teleop and the
    # verified replay path to the same layout, so only ONE term's SOURCE differs.
    "g1-29dof-holosoma-wbt-live": "g1-29dof-holosoma-wbt",
}

# Live presets with NO baked twin in the registry, so there is nothing to mirror
# against — validated directly against the upstream ObservationConfig with the
# `_live` suffix stripped (test_live_3pt_unified_preset_matches_upstream_actor_obs).
#
# The 3pt hybrid UNIFIED family is live-only on purpose: `bake_motions` stores only
# the 0.18 m-lever `vr3_pos_local` table, so a REPLAY preset for a checkpoint whose
# position term is the 0.242 m `vr_3point_pos_tcp` would silently feed a target
# 6.7 cm off per hand at the same 9D width. Register a baked twin only together
# with a TCP bake.
_LIVE_ONLY_PRESET_TO_OBS_ATTR = {
    "g1-29dof-holosoma-3pt-hybrid-unified-anneal-wbt-live": "holosoma_3pt_hybrid_unified_wbt",
    # 2026-08-27: the MOTION-COMPLETION teleop twins. Live-only for the same reason the 3pt
    # one is -- there is no baked table to substitute from -- and additionally because their
    # tracking front-end does not exist on the rig at all: three trackers become five
    # keypoints (or 29 joint targets) only by going through Motion_Completion.
    "g1-29dof-holosoma-5pt-hybrid-unified-anneal-wbt-live": "holosoma_5pt_hybrid_unified_wbt",
    "g1-29dof-holosoma-dense-hybrid-unified-anneal-wbt-live": "holosoma_dense_hybrid_unified_wbt",
}


def test_preset_registry_covers_all_holosoma_wbt() -> None:
    # The live presets are validated against their baked source (or, for the
    # live-only ones, against the upstream ObservationConfig), not through
    # _PRESET_TO_OBS_ATTR, so subtract them before the coverage equality.
    assert set(PRESETS) - set(_LIVE_PRESETS) - set(_LIVE_ONLY_PRESET_TO_OBS_ATTR) == set(_PRESET_TO_OBS_ATTR), (
        "Preset coverage drifted from the holosoma-*-wbt family. Update PRESETS, "
        "_PRESET_TO_OBS_ATTR, _LIVE_PRESETS, or _LIVE_ONLY_PRESET_TO_OBS_ATTR."
    )
    # Every live preset must be registered and name a real baked source preset.
    for live, baked in _LIVE_PRESETS.items():
        assert live in PRESETS, f"live preset {live} missing from PRESETS"
        assert baked in PRESETS, f"baked source {baked} for {live} missing from PRESETS"
    for live in _LIVE_ONLY_PRESET_TO_OBS_ATTR:
        assert live in PRESETS, f"live-only preset {live} missing from PRESETS"
        # A live-only preset must NOT also be claiming a baked twin.
        assert live not in _LIVE_PRESETS, f"{live} is listed as both live-only and mirrored"


def test_live_presets_mirror_baked() -> None:
    """Each live preset's obs_keys == its baked source's obs_keys with ONLY the TRACKING
    slot(s) swapped for their _live variants (same positions, same widths, same total dim).
    This is the load-bearing 'live swap preserves the obs contract' invariant — a reorder or
    width drift here would feed the wrong slice to the trained policy.

    The table below is the complete set of legal substitutions, so it stays STRICT: making
    any other term live (say `joint_pos`) still fails here rather than being waved through.
    """
    from wbt_realtime_eval.presets import term_base_dim

    _LIVE_SUB = {
        "vr_2point_pos": "vr_2point_pos_live",
        "vr_2point_orn": "vr_2point_orn_live",
        # Joint-space tracking front-end (the BASE tracker exp:g1-wbt, and the dense-unified
        # policy in _LIVE_ONLY_PRESET_TO_OBS_ATTR): the 58D `command` term IS the command, so
        # it is the slot the completion drives.
        "command": "command_live",
    }
    for live_name, baked_name in _LIVE_PRESETS.items():
        live = get_preset(live_name)
        baked = get_preset(baked_name)
        expected = tuple(_LIVE_SUB.get(k, k) for k in baked.obs_keys)
        assert live.obs_keys == expected, f"{live_name}: obs_keys {live.obs_keys} != baked-with-live-swap {expected}"
        # Swapped pairs keep identical widths -> dims preserved exactly.
        for lk, bk in zip(live.obs_keys, baked.obs_keys, strict=True):
            assert term_base_dim(lk) == term_base_dim(bk), (
                f"{live_name}: term {lk!r} width {term_base_dim(lk)} != baked {bk!r} {term_base_dim(bk)}"
            )
        assert live.per_frame_dim == baked.per_frame_dim
        assert live.total_dim == baked.total_dim
        assert live.history_length == baked.history_length
        assert live.vr_family == baked.vr_family


@pytest.mark.parametrize("preset_name", sorted(_LIVE_ONLY_PRESET_TO_OBS_ATTR))
def test_live_3pt_unified_preset_matches_upstream_actor_obs(preset_name: str) -> None:
    """A live-only preset's obs_keys, with the `_live` suffix stripped, must equal
    the upstream ObservationConfig's actor_obs order EXACTLY, and every term width
    must match upstream.

    This replaces the mirror-against-a-baked-twin check for presets that have no
    twin. It runs over the 680-D 3pt teleop preset and the two MOTION-COMPLETION
    ones (736-D 5pt / 732-D dense), and it is the load-bearing invariant for all
    three:

    * `vr_3point_pos_tcp` (0.242 m fingertip TCP) vs `vr_3point_pos` (0.18 m VR
      lever) are the SAME 9D width over the SAME three bodies, so only the term
      NAME distinguishes them — a slip here feeds the policy a target 6.7 cm off
      per hand with nothing downstream able to notice.
    * `contact_phase` (2D) sits immediately after `force_meas_local`, shifting
      integrator_state / x_tgt_shift / everything after it by +1 relative to the
      168D 2pt-hybrid layout. An off-by-one there mis-slices every later term.
    """
    obs_module = pytest.importorskip("wbt_wrappers_inference.config_values.observation")
    obs_cfg = getattr(obs_module, _LIVE_ONLY_PRESET_TO_OBS_ATTR[preset_name])
    from wbt_realtime_eval.presets import term_base_dim

    preset = get_preset(preset_name)
    stripped = tuple(k.removesuffix("_live") for k in preset.obs_keys)
    expected_keys = tuple(obs_cfg.obs_dict["actor_obs"])
    assert stripped == expected_keys, f"{preset_name}: expected actor_obs={expected_keys}, got {stripped}"
    # Only the TRACKING terms may carry the _live suffix; sourcing anything else live
    # would mean the bundle no longer matches the trained obs. Which tracking terms those
    # are is per-family, and the set is spelled out rather than pattern-matched so a new
    # family has to declare itself here.
    _ALLOWED_LIVE = {
        "g1-29dof-holosoma-3pt-hybrid-unified-anneal-wbt-live":
            {"vr_3point_pos_tcp_live", "vr_3point_orn_live"},
        # Motion completion: the SAME three trackers, but the tracking slot(s) are a slice
        # of a completed 29-DoF body rather than the pucks themselves.
        "g1-29dof-holosoma-5pt-hybrid-unified-anneal-wbt-live":
            {"vr_5point_pos_tcp_live", "vr_5point_orn_live"},
        # DENSE has no keypoint slot at all: the whole 58D joint-target block is live.
        "g1-29dof-holosoma-dense-hybrid-unified-anneal-wbt-live": {"command_live"},
    }
    live_swapped = {k for k in preset.obs_keys if k.endswith("_live")}
    assert live_swapped == _ALLOWED_LIVE[preset_name], (
        f"{preset_name}: only the tracking terms may be live, got {sorted(live_swapped)}"
    )
    assert preset.history_length == obs_cfg.history_length_dict["actor_obs"]
    for live_key, up_key in zip(preset.obs_keys, expected_keys, strict=True):
        assert term_base_dim(live_key) == obs_cfg.obs_dims[up_key], (
            f"{preset_name}: term {live_key!r} dim {term_base_dim(live_key)} != upstream "
            f"{up_key!r} {obs_cfg.obs_dims[up_key]}"
        )
    expected_per_frame = sum(obs_cfg.obs_dims[k] for k in expected_keys)
    _WIDTHS = {
        "g1-29dof-holosoma-3pt-hybrid-unified-anneal-wbt-live": 170,
        "g1-29dof-holosoma-5pt-hybrid-unified-anneal-wbt-live": 184,
        "g1-29dof-holosoma-dense-hybrid-unified-anneal-wbt-live": 183,
    }
    assert preset.per_frame_dim == expected_per_frame == _WIDTHS[preset_name]
    assert preset.total_dim == expected_per_frame * preset.history_length
    # contact_phase directly after force_meas_local — the +1 shift that separates every
    # unified layout from its non-unified ancestor. True for all three families.
    assert preset.obs_keys.index("contact_phase") == preset.obs_keys.index("force_meas_local") + 1
    # The live tracking slot sits at index 1 for the keypoint families (index 0 is
    # command_lower_body) and at index 0 for dense (which has no lower-body slice), and it
    # is never the 0.18 m-LEVER term.
    if "command_live" in preset.obs_keys:
        assert preset.obs_keys[0] == "command_live"
        assert not any(k.startswith("vr_") for k in preset.obs_keys)
    else:
        assert preset.obs_keys[1].endswith("_pos_tcp_live"), preset.obs_keys[1]
    assert not any(k in preset.obs_keys for k in ("vr_3point_pos_live", "vr_5point_pos_live"))
    # Baked 3pt FK is still needed for the non-tracking terms + the idle force metric, and
    # for the MC families the rig streams three trackers regardless of the obs width.
    assert preset.vr_family == "3pt"


def test_live_3pt_preset_does_not_perturb_other_presets() -> None:
    """Non-invasion: registering the live 3pt unified preset leaves every other
    preset's layout + dims byte-identical, and adds no LIVE term to them.

    `contact_phase` is deliberately NOT asserted absent here: it is shared with the
    whole C1 unified family (the 170D/185D/184D/183D twins all carry it), so the only
    terms that must be exclusive to this preset are the two `*_live` ones. The widths
    of the two presets the live tuple is DERIVED from are pinned by value instead,
    which is the check that actually has teeth — the derivation is a dict-substitution
    over the 170D tuple, so a typo in the mapping keys would silently leave the source
    tuple mutated or the live tuple unswapped."""
    assert get_preset("g1-29dof-holosoma-3pt-wbt").per_frame_dim == 144
    assert get_preset("g1-29dof-holosoma-2pt-hybrid-wbt").per_frame_dim == 165
    assert get_preset("g1-29dof-holosoma-2pt-wbt-live").per_frame_dim == 141
    assert get_preset("g1-29dof-holosoma-3pt-hybrid-unified-wbt").total_dim == 680
    assert get_preset("g1-29dof-holosoma-3pt-hybrid-unified-anneal-wbt").total_dim == 680
    assert get_preset("g1-29dof-holosoma-5pt-hybrid-unified-anneal-wbt").total_dim == 736
    assert get_preset("g1-29dof-holosoma-dense-hybrid-unified-anneal-wbt").total_dim == 732
    for name, preset in PRESETS.items():
        if name in _LIVE_ONLY_PRESET_TO_OBS_ATTR:
            continue
        assert "vr_3point_pos_tcp_live" not in preset.obs_keys, name
        assert "vr_3point_orn_live" not in preset.obs_keys, name


@pytest.mark.parametrize("preset_name", sorted(_PRESET_TO_OBS_ATTR))
def test_preset_dims_match_observation_config(preset_name: str) -> None:
    """Every preset's obs_keys order + per-term dim must match the upstream
    ObservationConfig.  This is the load-bearing invariant the rest of the
    package relies on."""
    obs_module = pytest.importorskip("wbt_wrappers_inference.config_values.observation")
    obs_cfg = getattr(obs_module, _PRESET_TO_OBS_ATTR[preset_name])

    preset = get_preset(preset_name)
    expected_keys = tuple(obs_cfg.obs_dict["actor_obs"])
    assert preset.obs_keys == expected_keys, f"{preset_name}: expected obs_keys={expected_keys}, got {preset.obs_keys}"

    expected_history = obs_cfg.history_length_dict["actor_obs"]
    assert preset.history_length == expected_history

    from wbt_realtime_eval.presets import term_base_dim

    for k in preset.obs_keys:
        assert term_base_dim(k) == obs_cfg.obs_dims[k], (
            f"{preset_name}: term {k!r} dim mismatch — registry={term_base_dim(k)}, upstream={obs_cfg.obs_dims[k]}"
        )

    expected_per_frame = sum(obs_cfg.obs_dims[k] for k in preset.obs_keys)
    assert preset.per_frame_dim == expected_per_frame
    assert preset.total_dim == expected_per_frame * expected_history


def test_unsupported_preset_raises() -> None:
    with pytest.raises(ValueError, match="Unsupported preset"):
        get_preset("g1-29dof-holosoma-terrain-wbt")


# ── exp:g1-wbt-2pt-force-ext (F^ext in the actor, run fmotwon3) ──────────────


def test_force_ext_preset_is_153d_in_sorted_training_order() -> None:
    """The force-ext preset MUST be in the alphabetically-SORTED key order the
    trained fmotwon3 ONNX expects, NOT sequential insertion order.

    The holosoma core ObservationManager concatenates a group's terms in
    sorted(term_keys) order (managers/observation/manager.py:139), and the
    force-ext training keys are un-zero-padded obs0..obs10
    (force_tracking.py:1447), so "obs10_actions" string-sorts to slot 2 (right
    after obs0_command_lower_body, before obs1_vr_2point_pos). Verified against
    model_27000.onnx: `actions` is at obs slot [24]. If `actions` is placed last
    (sequential order) the robot flails and falls on load. 153D/frame x 4 = 612D."""
    ext = get_preset("g1-29dof-holosoma-2pt-force-ext-wbt")
    assert ext.obs_keys == (
        "command_lower_body",  # obs0
        "actions",  # obs10 → sorts to slot 2
        "vr_2point_pos",  # obs1
        "vr_2point_orn",  # obs2
        "force_cmd_local",  # obs3
        "force_ext_local",  # obs4
        "motion_ref_ori_b",  # obs5
        "projected_gravity",  # obs6
        "base_ang_vel",  # obs7
        "joint_pos",  # obs8
        "joint_vel",  # obs9
    )
    # force_ext_local sits immediately after force_cmd_local (obs3 → obs4).
    assert ext.obs_keys.index("force_ext_local") == ext.obs_keys.index("force_cmd_local") + 1
    # `actions` must be the SECOND term (slot 2), not last — this is the fall-over guard.
    assert ext.obs_keys.index("actions") == 1
    assert ext.per_frame_dim == 153
    assert ext.total_dim == 612
    assert ext.history_length == 4
    assert ext.vr_family == "2pt"


def test_force_ext_matches_sorted_training_layout() -> None:
    """force-ext key order == sorted(training obs0..obs10 keys) mapped to obs names.
    This is the authoritative guard: reproduce the exact string sort the core
    ObservationManager applies, and assert the preset tuple equals it. Guards
    against a reorder that would silently re-break the fall-over bug."""
    # Training keys (force_tracking.py:_actor_obs_2pt_force_ext_terms), un-padded.
    train_keys = [
        "obs0_command_lower_body",
        "obs1_vr_2point_pos",
        "obs2_vr_2point_orn",
        "obs3_force_cmd_local",
        "obs4_force_ext_local",
        "obs5_motion_ref_ori_b",
        "obs6_projected_gravity",
        "obs7_base_ang_vel",
        "obs8_dof_pos",
        "obs9_dof_vel",
        "obs10_actions",
    ]
    alias = {"dof_pos": "joint_pos", "dof_vel": "joint_vel"}
    expected = tuple(alias.get(k.split("_", 1)[1], k.split("_", 1)[1]) for k in sorted(train_keys))
    ext = get_preset("g1-29dof-holosoma-2pt-force-ext-wbt")
    assert ext.obs_keys == expected


def test_force_ext_is_force_preset_plus_one_term() -> None:
    """force-ext == the shipped 2pt-force actor terms with EXACTLY one extra term
    (force_ext_local). Guards against a term being dropped/added. NOTE: only the
    SET is compared here — the ORDER differs (force-ext is sorted, so `actions`
    moves to slot 2); order is asserted by test_force_ext_matches_sorted_training_layout."""
    force = get_preset("g1-29dof-holosoma-2pt-force-wbt")
    ext = get_preset("g1-29dof-holosoma-2pt-force-ext-wbt")
    assert set(ext.obs_keys) - set(force.obs_keys) == {"force_ext_local"}
    assert set(force.obs_keys) - set(ext.obs_keys) == set()
    assert ext.per_frame_dim == force.per_frame_dim + 6  # one 6D force term


def test_force_ext_does_not_perturb_shipped_force_preset() -> None:
    """Non-invasion: registering force-ext leaves the shipped 2pt-force preset's
    layout + dims byte-identical."""
    force = get_preset("g1-29dof-holosoma-2pt-force-wbt")
    assert force.per_frame_dim == 147
    assert force.total_dim == 588
    assert "force_ext_local" not in force.obs_keys  # F^ext stays critic-only there


# ── exp:g1-wbt-2pt-hybrid (servo-ON run 31wei7by + Layer-A run 6y8314vq) ─────


def test_hybrid_preset_is_165d_sorted_order() -> None:
    """The hybrid actor obs is 165D in hybrid.py's sorted obs00..obs12 order:
    the 2pt base + force_cmd_local, force_meas_local, integrator_state, x_tgt_shift
    (in THAT order), then motion_ref_ori_b/gravity/ang_vel/dof/actions."""
    hy = get_preset("g1-29dof-holosoma-2pt-hybrid-wbt")
    assert hy.obs_keys == (
        "command_lower_body",
        "vr_2point_pos",
        "vr_2point_orn",
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
    )
    assert hy.per_frame_dim == 165
    assert hy.total_dim == 660
    assert hy.history_length == 4
    assert hy.vr_family == "2pt"
    # The four force/servo channels sit contiguously at slots 3..6.
    assert hy.obs_keys.index("force_cmd_local") == 3
    assert hy.obs_keys.index("force_meas_local") == 4
    assert hy.obs_keys.index("integrator_state") == 5
    assert hy.obs_keys.index("x_tgt_shift") == 6
    # F^ext is critic-only in hybrid — NOT in the actor obs.
    assert "force_ext_local" not in hy.obs_keys


def test_hybrid_servo_and_layerA_share_identical_obs() -> None:
    """servo-ON and Layer-A presets share the EXACT same 165D obs (servo on/off is
    a runtime flag, not an obs difference — the network transfers between phases)."""
    servo = get_preset("g1-29dof-holosoma-2pt-hybrid-wbt")
    layer_a = get_preset("g1-29dof-holosoma-2pt-hybrid-layerA-wbt")
    assert servo.obs_keys == layer_a.obs_keys
    assert servo.per_frame_dim == layer_a.per_frame_dim == 165
    assert servo.total_dim == layer_a.total_dim == 660


def test_hybrid_does_not_perturb_shipped_2pt_and_force_presets() -> None:
    """Non-invasion: registering the hybrid presets leaves the plain 2pt and the
    shipped force preset byte-identical."""
    assert get_preset("g1-29dof-holosoma-2pt-wbt").per_frame_dim == 141
    assert get_preset("g1-29dof-holosoma-2pt-force-wbt").per_frame_dim == 147


def test_unified_hybrid_global_vel_preset_terms() -> None:
    """C1 unified presets (185D = 183D + contact_phase after force_meas_local)."""
    from wbt_realtime_eval.presets import _HOLOSOMA_3PT_HYBRID_GLOBAL_VEL_UNIFIED_WBT_TERMS as t
    from wbt_realtime_eval.presets import _TERMS

    assert t.index("contact_phase") == t.index("force_meas_local") + 1
    assert len(t) == 17
    assert _TERMS["contact_phase"].base_dim == 2
    assert _TERMS["contact_phase"].ts_name == "WbtContactPhase"
    p = PRESETS["g1-29dof-holosoma-3pt-hybrid-global-vel-unified-wbt"]
    assert p.obs_keys == t
    assert p.per_frame_dim == 185
    assert p.total_dim == 740  # ×history 4
    assert p.vr_family == "3pt"
    a = PRESETS["g1-29dof-holosoma-3pt-hybrid-global-vel-unified-anneal-wbt"]
    assert a.obs_keys == t


def test_local_unified_hybrid_preset_terms() -> None:
    """LOCAL C1 unified presets (170D = 168D 3pt-hybrid + contact_phase after
    force_meas_local; quaternion vr_3point_orn, no root trio)."""
    from wbt_realtime_eval.presets import _HOLOSOMA_3PT_HYBRID_UNIFIED_WBT_TERMS as t

    assert t.index("contact_phase") == t.index("force_meas_local") + 1
    assert len(t) == 14
    assert "vr_3point_orn" in t and "vr_3point_orn_6d" not in t
    assert not any(k in t for k in ("motion_ref_pos_b", "motion_ref_vel_b", "base_lin_vel"))
    p = PRESETS["g1-29dof-holosoma-3pt-hybrid-unified-wbt"]
    assert p.obs_keys == t
    assert p.per_frame_dim == 170
    assert p.total_dim == 680  # ×history 4
    assert p.vr_family == "3pt"
    a = PRESETS["g1-29dof-holosoma-3pt-hybrid-unified-anneal-wbt"]
    assert a.obs_keys == t
    # the base 168D and the 185D global-vel unified presets are untouched
    assert PRESETS["g1-29dof-holosoma-3pt-hybrid-wbt"].per_frame_dim == 168
    assert PRESETS["g1-29dof-holosoma-3pt-hybrid-global-vel-unified-wbt"].per_frame_dim == 185


def test_unified_preset_names_trigger_observe_only_dispatch() -> None:
    """REGRESSION (local-unified review 2026-07-24): the load-bearing contract
    'neither unified twin injects a servo at deploy' rests on main.py's name-keyed
    dispatch (`"unified" in name.lower()` → observe-only override; `"layera"` →
    servo-off override). Reproduce that exact string logic against every registered
    unified preset name so a rename/tightening of the dispatch (e.g. keying on
    "-global-vel-unified") that misses a family breaks HERE, not on the robot.
    The R1 anneal checkpoints bake the FULL-servo cfg (terminal α=0 is curriculum
    runtime state, never serialized) — the name override is their only guard."""
    unified_names = [n for n in PRESETS if "unified" in n]
    assert sorted(unified_names) == [
        "g1-29dof-holosoma-3pt-hybrid-global-vel-unified-anneal-wbt",
        "g1-29dof-holosoma-3pt-hybrid-global-vel-unified-wbt",
        "g1-29dof-holosoma-3pt-hybrid-unified-anneal-wbt",
        # The LIVE VIVE-teleop twin of the 170D preset. It MUST also match the
        # observe-only name dispatch: it is served from the very same wda3h4ll R1
        # checkpoint, which bakes servo_observe_only=False. (Sorts here, not at the
        # end: "-anneal-wbt-live" < "-unified-wbt" because 'a' < 'w'.)
        "g1-29dof-holosoma-3pt-hybrid-unified-anneal-wbt-live",
        "g1-29dof-holosoma-3pt-hybrid-unified-wbt",
        # 2026-08-27: the 5pt + DENSE anneal twins join the family, each with its LIVE
        # motion-completion teleop variant. All carry "unified" in the name, which is
        # exactly what must stay true — their R1 checkpoints bake
        # servo_observe_only=False, so the name override is their only servo guard too.
        "g1-29dof-holosoma-5pt-hybrid-unified-anneal-wbt",
        "g1-29dof-holosoma-5pt-hybrid-unified-anneal-wbt-live",
        "g1-29dof-holosoma-dense-hybrid-unified-anneal-wbt",
        "g1-29dof-holosoma-dense-hybrid-unified-anneal-wbt-live",
    ]
    for name in unified_names:
        preset = get_preset(name)
        # the hybrid dispatch itself keys off force_meas_local in obs_keys
        assert "force_meas_local" in preset.obs_keys, name
        # main.py:~1156 — observe-only fires, layerA servo-off does NOT
        assert ("unified" in preset.name.lower()) is True, name
        assert ("layera" in preset.name.lower()) is False, name


def test_hybrid_extras_carry_observe_only_and_contact_phase() -> None:
    """REGRESSION (unified-hybrid review): extras.hybrid must carry the C1
    servo_observe_only flag + the contact_phase constants — without them the
    browser runs the FULL Jᵀ ctrl injection on a single-policy checkpoint
    (double actuation the policy never trained with)."""
    from wbt_realtime_eval.main import _hybrid_preset_extras

    # R2 checkpoint bakes the flag → passthrough.
    cfg_r2 = {"servo_enabled": True, "servo_observe_only": True, "contact_phase_theta_on": 2.0}
    _s, extras, _v, _e = _hybrid_preset_extras(cfg_r2)
    assert extras["servo_enabled"] is True
    assert extras["servo_observe_only"] is True
    assert extras["contact_phase_theta_on"] == 2.0
    assert extras["contact_phase_theta_off"] == 1.5
    assert extras["contact_phase_tau_s"] == 0.05
    # R1 anneal checkpoint bakes the FULL-servo cfg (terminal α=0 is curriculum
    # runtime state, not serialized) → the -unified name override forces observe-only
    # while KEEPING servo_enabled=True (the policy needs the wound I/Δx published).
    cfg_r1 = {"servo_enabled": True, "servo_observe_only": False}
    _s, extras_r1, _v, _e = _hybrid_preset_extras(cfg_r1, servo_observe_only_override=True)
    assert extras_r1["servo_enabled"] is True
    assert extras_r1["servo_observe_only"] is True
    # Old hybrid checkpoints: flag absent → False (byte-identical behavior).
    _s, extras_old, _v, _e = _hybrid_preset_extras({"servo_enabled": True})
    assert extras_old["servo_observe_only"] is False


@pytest.mark.skipif(
    not (Path(__file__).resolve().parents[2] / "wbt_training/training_runs/run_hybrid_3pt_unified_realtime_viz.sh").is_file(),
    reason="the 3pt viz launchers are not part of this checkout",
)
def test_local_unified_launchers_exist_and_are_wired() -> None:
    """The LOCAL C1 unified twins ship BOTH mjswan launchers, each pinned to its own
    preset + a free port + the canonical W&B run.

    A missing/mis-pointed launcher is the failure mode that actually bites: the preset
    + inference config + runtime can all be correct while there is no way to LAUNCH the
    viz (or, worse, a launcher whose PRESET string names the servo-ON 168D preset, which
    would build a 672D bundle against a 680D checkpoint and be refused only at the
    normalizer dim check). Ports must also be unique across every launcher, since
    serve_bundle cannot bind over an active LISTEN socket."""
    import re

    # tests/ -> wbt_realtime_eval/ -> wbt_wrappers/ -> (sibling) wbt_training/
    runs = Path(__file__).resolve().parents[2] / "wbt_training" / "training_runs"
    expected = {
        # R2: no run pinned — the TCP-unified retrain has not been launched, and the old
        # pre-TCP run (m3s0pw9i, 0.18 m) must NOT be the default since it is also 170D.
        "run_hybrid_3pt_unified_realtime_viz.sh": ("g1-29dof-holosoma-3pt-hybrid-unified-wbt", None),
        # R1: pfl96opy = WBT-Hybrid3ptUniAnneal-tcp-bones-64gpu-gloo, the TCP-unified
        # retrain (obs01 = vr_3point_pos_tcp, verified on its model_00000.onnx).
        "run_hybrid_3pt_unified_anneal_realtime_viz.sh": (
            "g1-29dof-holosoma-3pt-hybrid-unified-anneal-wbt",
            "pfl96opy",
        ),
        "run_hybrid_3pt_global_vel_unified_realtime_viz.sh": (
            "g1-29dof-holosoma-3pt-hybrid-global-vel-unified-wbt",
            None,
        ),
        "run_hybrid_3pt_global_vel_unified_anneal_realtime_viz.sh": (
            "g1-29dof-holosoma-3pt-hybrid-global-vel-unified-anneal-wbt",
            None,
        ),
        # 5pt (dlcxc59d) and DENSE (7r54008a), 2026-08-27. Both pin a canonical run: unlike
        # the TCP-unified 3pt pair there is no pre-TCP predecessor to be confused with —
        # 736D is unique, and for the 732D dense preset the same-width global-vel family is
        # blocked by main.py's tracking-family guard rather than by leaving the pin empty.
        "run_hybrid_5pt_unified_anneal_realtime_viz.sh": (
            "g1-29dof-holosoma-5pt-hybrid-unified-anneal-wbt",
            "dlcxc59d",
        ),
        "run_hybrid_dense_unified_anneal_realtime_viz.sh": (
            "g1-29dof-holosoma-dense-hybrid-unified-anneal-wbt",
            "7r54008a",
        ),
    }
    for fname, (preset_name, run_id) in expected.items():
        path = runs / fname
        assert path.is_file(), f"missing mjswan launcher {fname}"
        assert os.access(path, os.X_OK), f"{fname} must be executable"
        text = path.read_text()
        # The launcher's PRESET must be a REGISTERED preset (not a typo / stale name).
        assert f'PRESET="${{PRESET:-{preset_name}}}"' in text, f"{fname} must default PRESET to {preset_name}"
        assert preset_name in PRESETS, preset_name
        # Assert on the ASSIGNMENT, not a bare substring: `run_id in text` also matches a
        # mention in a comment, so a launcher that merely *documents* the run while pinning
        # a different one (or none) would pass. And run_id=None must ASSERT emptiness rather
        # than skip — a silently re-pinned stale run is exactly what this guard is for.
        m = re.search(r'^DEFAULT_WANDB_RUN="([^"]*)"', text, re.MULTILINE)
        assert m is not None, f"{fname} must assign DEFAULT_WANDB_RUN (empty string if no canonical run yet)"
        pinned = m.group(1)
        if run_id is None:
            assert pinned == "", (
                f'{fname} must pin NO canonical run (DEFAULT_WANDB_RUN=""), got {pinned!r}. '
                "For the TCP-unified twins this is load-bearing: the pre-TCP runs are also "
                "170D, so a stale pin would build a geometrically wrong bundle."
            )
        else:
            assert pinned.endswith("/" + run_id), (
                f"{fname} must pin the canonical W&B run {run_id} in DEFAULT_WANDB_RUN, got {pinned!r}"
            )
        assert "wbt_realtime_viz_main" in text, f"{fname} must call the shared launcher"

    # Every launcher on disk must claim a DISTINCT default port.
    ports: dict[str, str] = {}
    for sh in sorted(runs.glob("run_*realtime_viz.sh")):
        m = re.search(r'^PORT="\$\{PORT:-(\d+)\}"', sh.read_text(), re.MULTILINE)
        if m is None:
            continue
        port = m.group(1)
        # run_upperbody_realtime_viz.sh predates the shared launcher and shares 8000
        # with run_force_ext_realtime_viz.sh; they are never run together.
        if sh.name in ("run_upperbody_realtime_viz.sh", "run_force_ext_realtime_viz.sh"):
            continue
        assert port not in ports, f"{sh.name} and {ports[port]} both default to port {port}"
        ports[port] = sh.name


def test_default_wandb_run_is_actually_honored() -> None:
    """REGRESSION (2026-08-04): the shared launcher must actually READ DEFAULT_WANDB_RUN.

    Every wrapper sets `DEFAULT_WANDB_RUN=...` and both READMEs promise a bare
    `bash run_<x>_realtime_viz.sh` uses that canonical run. But until 2026-08-04
    `wbt_realtime_viz_main` never read the variable — its ONNX resolution was
    MODEL > WANDB_RUN > RUN_DIR, so a bare invocation silently fell through to "newest
    model_*.onnx anywhere under logs/WBT-Holosoma". That dir is SHARED by every preset,
    so the launcher could grab a checkpoint from an unrelated experiment; with the
    TCP-unified twins it could even grab a pre-TCP 170D unified ONNX, which has the right
    obs width and the WRONG hand geometry. Pin the wiring so it cannot silently regress."""
    common = (
        Path(__file__).resolve().parents[2] / "wbt_training" / "training_runs" / "_realtime_viz_common.sh"
    ).read_text()
    # The resolved run must be able to COME FROM DEFAULT_WANDB_RUN...
    assert '_wandb_run="${DEFAULT_WANDB_RUN:-}"' in common, (
        "wbt_realtime_viz_main must fall back to DEFAULT_WANDB_RUN when the caller named no "
        "checkpoint; otherwise every launcher's canonical run is dead documentation."
    )
    # ...and the W&B download branch must test that RESOLVED value, not WANDB_RUN alone
    # (testing WANDB_RUN would skip the DEFAULT_WANDB_RUN fallback entirely).
    assert 'elif [ -n "$_wandb_run" ]; then' in common, (
        "the W&B download branch must test the RESOLVED run (_wandb_run), not WANDB_RUN."
    )
    # `${DEFAULT_WANDB_RUN:-}` (not `$DEFAULT_WANDB_RUN`) matters: the launchers run under
    # `set -u` and run_upperbody_realtime_viz.sh never defines the variable at all.
    assert "DEFAULT_WANDB_RUN:-" in common, "must use :- expansion so set -u tolerates an unset DEFAULT_WANDB_RUN"
    # An EXPLICIT RUN_DIR must still outrank DEFAULT_WANDB_RUN — `RUN_DIR=/abs/run` is a
    # documented recipe in every launcher header, so honoring the canonical run must not
    # make it unreachable. That requires capturing explicitness BEFORE RUN_DIR is defaulted.
    assert "_RUN_DIR_EXPLICIT" in common, (
        "an explicit RUN_DIR must suppress the DEFAULT_WANDB_RUN fallback; the common lib "
        "defaults RUN_DIR at source time, so explicitness has to be captured beforehand."
    )
    assert 'elif [ "$_RUN_DIR_EXPLICIT" != "1" ]; then' in common, (
        "the DEFAULT_WANDB_RUN fallback must be gated on RUN_DIR not being explicitly set."
    )
    # WANDB_RUN stays ABOVE an explicit RUN_DIR (both are caller-supplied, but WANDB_RUN is
    # the more specific request), so its branch must be checked first.
    assert common.index('if [ -n "${WANDB_RUN:-}" ]; then') < common.index('elif [ "$_RUN_DIR_EXPLICIT" != "1" ]'), (
        "WANDB_RUN must take precedence over an explicit RUN_DIR."
    )


def test_unified_hybrid_viz_block_marks_observe_only() -> None:
    """REGRESSION: the hybrid_viz block handed to WbtTrackingCommand must carry
    servo_observe_only, because the browser UI keys three user-visible behaviors off it:
    the badge text ("UNIFIED POLICY" vs "FORCE SERVO"/"servo off"), suppressing the dead
    Layer-B checkbox (the runtime IGNORES that toggle in observe-only mode), and the
    "Layer B = reference only" legend line. Without the flag the UI would tell the
    operator a Jᵀ servo is actuating when none is — the same class of lie as running
    the injection itself, just in the other direction."""
    from wbt_realtime_eval.main import _hybrid_preset_extras

    _s, extras, _v, _e = _hybrid_preset_extras({"servo_enabled": True}, servo_observe_only_override=True)
    # main.py builds hybrid_viz from these keys; assert they are all present so the
    # dict construction in setup_builder cannot KeyError or silently drop the flag.
    for key in ("servo_enabled", "servo_observe_only", "tcp_offset_x", "max_penetration_m", "plane_dist", "hands"):
        assert key in extras, key
    assert extras["servo_observe_only"] is True
    # servo_enabled MUST stay True: the runtime needs it to keep publishing the wound
    # I/Δx that the 170D/185D actor obs consume (this is NOT the layerA servo-off path).
    assert extras["servo_enabled"] is True


def test_observe_only_preserves_integrator_publishing_contract() -> None:
    """The observe-only override must NEVER be implemented as servo_enabled=False.

    Both are "no Δq reaches ctrl", but they differ in what the OBS sees:
    servo_enabled=False routes the runtime to zeroHybridServoState, which publishes
    I=Δx=0 — feeding the actor zeros for two 6D channels that were live in training.
    Observe-only keeps servo_enabled=True so hybridServoStep still winds I and publishes
    I/Δx, and only skips the ctrl injection. Assert the two overrides stay independent."""
    from wbt_realtime_eval.main import _hybrid_preset_extras

    # layerA path: servo OFF, observe-only untouched.
    _s, layer_a, _v, _e = _hybrid_preset_extras({"servo_enabled": True}, servo_enabled_override=False)
    assert layer_a["servo_enabled"] is False
    assert layer_a["servo_observe_only"] is False
    # unified path: servo stays ON (I/Δx keep publishing), observe-only forced.
    _s, unified, _v, _e = _hybrid_preset_extras({"servo_enabled": True}, servo_observe_only_override=True)
    assert unified["servo_enabled"] is True
    assert unified["servo_observe_only"] is True


def test_observe_only_repairs_missing_servo_enabled() -> None:
    """REGRESSION (review 2026-07-25): observe-only is a CONJUNCTION in the runtime
    (`servoOn = observeOnly ? servoEnabled : <toggle>`), so observe_only=True paired with
    servo_enabled=False routes to zeroHybridServoState and pins I/Δx to 0 — silently
    killing 12 of the unified actor's 170 obs dims while contact_phase keeps working, so
    nothing LOOKS wrong.

    That pairing is reachable WITHOUT any override bug: if `_parse_hybrid_cfg` cannot find
    the HybridContactConfig (absent / unparseable / reshaped experiment_config) it returns
    None and main.py proceeds, at which point servo_enabled falls back to False while the
    preset NAME still forces observe_only=True. Assert the repair holds for every such
    degraded input, since these are exactly the cases the earlier
    test_observe_only_preserves_integrator_publishing_contract did NOT cover (it always
    passed servo_enabled=True)."""
    from wbt_realtime_eval.main import _hybrid_preset_extras

    for degraded in (None, {}, {"stiffness_K": 200.0}, {"servo_enabled": False}):
        _s, extras, _v, _e = _hybrid_preset_extras(degraded, servo_observe_only_override=True)
        assert extras["servo_observe_only"] is True, degraded
        assert extras["servo_enabled"] is True, (
            f"cfg={degraded!r} yielded servo_enabled=False under observe-only — the runtime "
            f"would publish I=Δx=0 into the unified actor's obs."
        )

    # The contradictory combination (explicit layerA servo-off AND observe-only) is not
    # reachable from any registered preset (the name families are disjoint), so it must
    # fail LOUDLY rather than silently resolving one way.
    with pytest.raises(ValueError, match="Contradictory hybrid overrides"):
        _hybrid_preset_extras({"servo_enabled": True}, servo_enabled_override=False, servo_observe_only_override=True)

    # And the repair must NOT leak into non-unified presets: a layerA/servo-off preset
    # (observe_only False) keeps servo_enabled=False.
    _s, layer_a, _v, _e = _hybrid_preset_extras({"servo_enabled": True}, servo_enabled_override=False)
    assert layer_a["servo_enabled"] is False
    assert layer_a["servo_observe_only"] is False


def test_dense_unified_is_exactly_the_base_wbt_obs_plus_the_force_block() -> None:
    """`g1-29dof-holosoma-wbt` and `…-dense-hybrid-unified-anneal-wbt` are the SAME tracking
    family, and the relationship is an exact superset — not a rough analogy.

    Pinning this matters because the two are constantly reasoned about as if they were
    different capabilities ("the base tracker can only do posture, dense can do more"), and
    they are not: BOTH are steered by the identical 58D `command` term, so BOTH require
    Motion_Completion to be teleoperated from three trackers, BOTH ride the bridge's
    `--mode dense` wire, and NEITHER carries a root term. The only difference is force.

    If this ever fails, one of the two tuples drifted and the deploy story for the pair has
    silently forked — which is exactly the confusion this test exists to prevent."""
    base = PRESETS["g1-29dof-holosoma-wbt"].obs_keys
    dense = PRESETS["g1-29dof-holosoma-dense-hybrid-unified-anneal-wbt"].obs_keys

    force_block = ("force_cmd_local", "force_meas_local", "contact_phase", "integrator_state", "x_tgt_shift")

    # 1. Superset, and the ONLY additions are the force block.
    assert set(base) <= set(dense)
    assert set(dense) - set(base) == set(force_block)

    # 2. The base's term ORDER survives inside dense untouched (so the shared slots line up
    #    conceptually — a reordering here would mean the two stopped being a family even
    #    though the set difference still looked right).
    assert tuple(k for k in dense if k in set(base)) == base

    # 3. The force block is contiguous and sits immediately after the shared front-end.
    assert dense[0] == base[0] == "command"
    assert dense[1 : 1 + len(force_block)] == force_block

    # 4. Widths follow: 157 + 26 = 183.
    assert PRESETS["g1-29dof-holosoma-wbt"].per_frame_dim == 157
    assert PRESETS["g1-29dof-holosoma-dense-hybrid-unified-anneal-wbt"].per_frame_dim == 183


def test_no_motion_completion_preset_has_a_root_channel() -> None:
    """The "posture only, will not walk" limit belongs to the whole motion-completion family,
    not to the base tracker alone.

    None of 5pt / dense / wbt carries `motion_ref_pos_b`, so none of them is ever told where
    the reference ROOT is. `x_tgt_shift` is NOT a counter-example: it is the force servo's
    per-hand Δx off the mjswan spring wall (`wbt_xtgt`), which is why it appears in the two
    force presets and not in the base one. Documenting the limit as wbt-specific was wrong,
    and this pins the correction."""
    mc_presets = (
        "g1-29dof-holosoma-wbt",
        "g1-29dof-holosoma-dense-hybrid-unified-anneal-wbt",
        "g1-29dof-holosoma-5pt-hybrid-unified-anneal-wbt",
    )
    for name in mc_presets:
        keys = PRESETS[name].obs_keys
        assert "motion_ref_pos_b" not in keys, f"{name} gained a root position term"
        assert "motion_ref_vel_b" not in keys, f"{name} gained a root velocity term"
        # motion_ref_ori_b IS present in all three — it is what pins heading to the clip.
        assert "motion_ref_ori_b" in keys, name

    # The live twins must inherit the same property (a live preset that quietly gained a root
    # term would change the teleop contract without changing any width).
    for name in mc_presets:
        live = f"{name}-live" if name.endswith("-wbt") else None
        if live and live in PRESETS:
            assert "motion_ref_pos_b" not in PRESETS[live].obs_keys, live


def test_the_three_motion_completion_presets_need_the_completion_not_raw_trackers() -> None:
    """Each MC preset's tracking front-end is something three VIVE pucks cannot supply
    directly, which is *why* Motion_Completion is mandatory for all three rather than an
    optional nicety for some of them.

    - wbt / dense: `command` = 29 joint targets ++ 29 velocities. A puck is not a joint.
    - 5pt: `command_lower_body` (24D of leg joint targets) + FIVE keypoints, two of which are
      the ankles — untracked bodies.

    Contrast the 2pt/3pt presets, whose whole front-end IS the tracked hand/head keypoints;
    those are the only ones a bare bridge can drive."""
    joint_space = ("g1-29dof-holosoma-wbt", "g1-29dof-holosoma-dense-hybrid-unified-anneal-wbt")
    for name in joint_space:
        keys = PRESETS[name].obs_keys
        assert "command" in keys
        assert not any(k.startswith("vr_") for k in keys), f"{name} unexpectedly has a keypoint term"

    five = PRESETS["g1-29dof-holosoma-5pt-hybrid-unified-anneal-wbt"].obs_keys
    assert "command_lower_body" in five
    assert "vr_5point_pos_tcp" in five and "vr_5point_orn" in five

    # The genuinely tracker-only family, for contrast: 3 pucks == its 3 keypoints.
    three = PRESETS["g1-29dof-holosoma-3pt-hybrid-unified-anneal-wbt"].obs_keys
    assert any(k.startswith("vr_3point") for k in three)
    assert "command" not in three
