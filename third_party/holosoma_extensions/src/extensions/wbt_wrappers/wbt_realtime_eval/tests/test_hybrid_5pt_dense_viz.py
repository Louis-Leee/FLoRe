"""mjswan browser viz for the 5-POINT and DENSE unified anneal twins (user 2026-08-27).

Covers the two presets added for the W&B runs

  * ``dlcxc59d`` (WBT-Hybrid5ptUniAnneal)  = exp:g1-wbt-5pt-hybrid-unified-anneal, 184D
  * ``7r54008a`` (WBT-HybridDenseUniAnneal) = exp:g1-wbt-hybrid-unified-anneal,     183D

and their blast radius. Three things here are load-bearing rather than cosmetic:

1. **The 732D collision.** The dense preset and ``g1-29dof-holosoma-3pt-hybrid-global-vel-wbt``
   are BOTH ``[1, 732]`` with completely disjoint layouts, so the obs-dim check cannot tell
   them apart. ``main._assert_checkpoint_tracking_family`` is what does; if it regresses, a
   wrong-family bundle builds cleanly and the robot falls.
2. **The 3pt prefix.** Slots ``[0:9]`` / ``[0:12]`` of the 5pt tables must be byte-identically
   the 3pt-TCP tables, because both the training obs and the bake construct 5pt by appending
   the feet. A reorder would silently shift 14 of 184 dims.
3. **"不影响其他 preset".** Every other preset's obs layout, marker gate and force gate must be
   untouched — asserted positively here, not just left to the pre-existing exhaustive-set
   tests in test_presets.py / test_main_builder.py.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from wbt_realtime_eval.presets import PRESETS, get_preset, term_base_dim

_5PT = "g1-29dof-holosoma-5pt-hybrid-unified-anneal-wbt"
_DENSE = "g1-29dof-holosoma-dense-hybrid-unified-anneal-wbt"
_3PT = "g1-29dof-holosoma-3pt-hybrid-unified-anneal-wbt"
_GLOBAL_VEL = "g1-29dof-holosoma-3pt-hybrid-global-vel-wbt"

_HERE = Path(__file__).resolve().parent
# tests/ -> wbt_realtime_eval/ -> wbt_wrappers/
_PKG = _HERE.parents[0]
_MOTIONS_DIR = _HERE.parents[1] / "wbt_training" / "motions"
_FIRST_MOTION = _MOTIONS_DIR / "idle_stand_1.npz"
_TEMPLATE = _PKG / "wbt_realtime_eval" / "template"

# The actor obs slot names recorded in each run's own W&B config (read from
# observation.groups.actor_obs.terms, i.e. exactly what lands in the ONNX experiment_config).
# Pinned here so the layout tests run without the 437 MB checkpoints present.
_DLCXC59D_TERMS = (
    "obs00_command_lower_body",
    "obs01_vr_5point_pos_tcp",
    "obs02_vr_5point_orn",
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
)
_7R54008A_TERMS = (
    "obs00_motion_command",
    "obs01_force_cmd_local",
    "obs02_force_meas_local",
    "obs03_contact_phase",
    "obs04_integrator_state",
    "obs05_x_tgt_shift",
    "obs06_motion_ref_ori_b",
    "obs07_projected_gravity",
    "obs08_base_ang_vel",
    "obs09_dof_pos",
    "obs10_dof_vel",
    "obs11_actions",
)
# Checkpoint slot spelling -> browser preset key. holosoma names the terms after its own
# obs funcs; presets.py uses the browser's spelling.
_SLOT_RENAME = {"dof_pos": "joint_pos", "dof_vel": "joint_vel", "motion_command": "command"}


def _stripped(slots: tuple[str, ...]) -> list[str]:
    """``obs07_x_tgt_shift`` -> ``x_tgt_shift``, with the browser rename applied."""
    bare = [s.split("_", 1)[1] for s in slots]
    return [_SLOT_RENAME.get(b, b) for b in bare]


def _meta_with_actor_terms(*term_keys: str) -> dict[str, str]:
    """Minimal ONNX metadata carrying just the actor_obs term names, shaped like the real
    serialized experiment_config (mirrors the helper in test_main_builder.py)."""
    return {
        "experiment_config": json.dumps(
            {"observation": {"groups": {"actor_obs": {"terms": {k: {"func": "x"} for k in term_keys}}}}}
        )
    }


# ── preset layout ────────────────────────────────────────────────────────────


def test_5pt_preset_is_184d_and_the_3pt_layout_with_five_points() -> None:
    p = get_preset(_5PT)
    ref = get_preset(_3PT)
    assert p.per_frame_dim == 184
    assert p.total_dim == 736  # x history 4 -> ONNX [1, 736]
    assert p.history_length == 4
    assert p.vr_family == "5pt"
    # Exactly the 170D layout with the two tracking slots widened — same length, same
    # positions for every other term. This is what keeps force_cmd_local in slot 3.
    assert len(p.obs_keys) == len(ref.obs_keys) == 14
    swapped = {"vr_3point_pos_tcp": "vr_5point_pos_tcp", "vr_3point_orn": "vr_5point_orn"}
    assert p.obs_keys == tuple(swapped.get(k, k) for k in ref.obs_keys)
    assert p.per_frame_dim - ref.per_frame_dim == (15 - 9) + (20 - 12) == 14
    # widths: 5 bodies x 3 (pos) and 5 x 4 (xyzw quat, the pelvis-local family's rep)
    assert term_base_dim("vr_5point_pos_tcp") == 15
    assert term_base_dim("vr_5point_orn") == 20
    # no 6D rot rep here (that lives only in the -global-vel family), no root trio
    assert "vr_5point_orn_6d" not in p.obs_keys
    assert not any(k in p.obs_keys for k in ("motion_ref_pos_b", "motion_ref_vel_b", "base_lin_vel"))


def test_dense_preset_is_183d_joint_targets_plus_the_force_block() -> None:
    p = get_preset(_DENSE)
    assert p.per_frame_dim == 183
    assert p.total_dim == 732  # x history 4 -> ONNX [1, 732]
    assert p.history_length == 4
    # DENSE = no keypoints at all, and no lower-body slice either (the 58D command
    # subsumes it).
    assert p.obs_keys[0] == "command"
    assert term_base_dim("command") == 58
    assert not any(k.startswith("vr_") for k in p.obs_keys)
    assert "command_lower_body" not in p.obs_keys
    # The force block sits immediately after the tracking term, in the same ORDER as every
    # other unified preset (that ordering is the sorted-concat contract).
    force_block = ("force_cmd_local", "force_meas_local", "contact_phase", "integrator_state", "x_tgt_shift")
    assert p.obs_keys[1 : 1 + len(force_block)] == force_block
    assert sum(term_base_dim(k) for k in force_block) == 26
    # ...and it is the SAME block, in the same relative position, as the 3pt twin's.
    ref = get_preset(_3PT)
    i = ref.obs_keys.index("force_cmd_local")
    assert ref.obs_keys[i : i + len(force_block)] == force_block
    # 183 = 58 (dense) + 26 (force) + the shared tail
    tail = ("motion_ref_ori_b", "projected_gravity", "base_ang_vel", "joint_pos", "joint_vel", "actions")
    assert p.obs_keys[1 + len(force_block) :] == tail
    assert p.per_frame_dim == 58 + 26 + sum(term_base_dim(k) for k in tail)


def test_both_new_presets_are_hybrid_and_unified_for_the_name_keyed_dispatch() -> None:
    """main.py gates the browser's force/servo behaviour on TWO things: ``force_meas_local``
    in obs_keys (hybrid wall + F/T sensor + 12 sliders + 3-arrow viz) and ``"unified"`` in
    the preset NAME (forces servo_observe_only, because an R1 checkpoint bakes
    servo_observe_only=False and its terminal alpha is never serialized). Both runs are R1
    anneal, so both need both."""
    for name in (_5PT, _DENSE):
        p = get_preset(name)
        assert "force_meas_local" in p.obs_keys, name
        assert "force_cmd_local" in p.obs_keys, name
        assert "contact_phase" in p.obs_keys, name
        assert "unified" in p.name.lower(), name
        # NOT the layerA pattern — servo_enabled must stay True so the runtime keeps
        # publishing the wound I/Δx these actors observe.
        assert "layera" not in p.name.lower(), name


def test_the_two_new_widths_are_distinguishable_or_explicitly_guarded() -> None:
    """736D collides only with its own LIVE twin. 732D collides with a DIFFERENT FAMILY —
    the two 3pt-hybrid-global-vel presets — so the dense preset's safety rests on the
    tracking-family guard, not on the width. Pin both facts.

    A live/baked pair sharing a width is not the hazard this test is about: they are the
    SAME layout and the same tracking family, differing only in whether the tracking slot
    is sourced from the bridge or from a baked table, so a mix-up feeds the right vector
    from the wrong place (and shows up as targets parked at HOME). The dangerous collision
    is two DISJOINT layouts at one width, which is what the 732D assertion below pins.
    """
    by_dim: dict[int, set[str]] = {}
    for name, p in PRESETS.items():
        by_dim.setdefault(p.total_dim, set()).add(name)
    assert by_dim[736] == {_5PT, _5PT + "-live"}, by_dim[736]
    assert by_dim[732] == {
        _DENSE,
        _DENSE + "-live",
        "g1-29dof-holosoma-3pt-hybrid-global-vel-wbt",
        "g1-29dof-holosoma-3pt-hybrid-global-vel-layerA-wbt",
    }, (
        "732D collision set changed. The dense preset can only be told apart from the "
        "global-vel family by term NAMES (main._assert_checkpoint_tracking_family); if this "
        f"set changed, re-check that guard. Got: {sorted(by_dim[732])}"
    )
    # ...and the collision is genuinely a disjoint layout, not a shared one.
    assert set(get_preset(_DENSE).obs_keys).isdisjoint(
        {"vr_3point_pos", "vr_3point_orn_6d", "motion_ref_pos_b", "motion_ref_vel_b", "base_lin_vel"}
    )
    assert get_preset(_DENSE).obs_keys[0] != get_preset(_GLOBAL_VEL).obs_keys[0]


# ── the five points are maskedmimic's five_point ─────────────────────────────


def test_the_five_points_are_the_3pt_set_plus_both_feet() -> None:
    """VR_5POINT_BODIES must be VR_3POINT_BODIES + the two ankle_roll links, in that order —
    the prefix property the bake and the obs layout both rely on."""
    from wbt_wrappers_inference.config_values.robot_config import (
        VR_3POINT_BODIES,
        VR_5POINT_BODIES,
        VR_5POINT_FEET_BODIES,
    )

    assert VR_5POINT_BODIES[: len(VR_3POINT_BODIES)] == list(VR_3POINT_BODIES)
    assert VR_5POINT_BODIES[len(VR_3POINT_BODIES) :] == list(VR_5POINT_FEET_BODIES)
    assert list(VR_5POINT_FEET_BODIES) == ["left_ankle_roll_link", "right_ankle_roll_link"]
    assert len(VR_5POINT_BODIES) == 5
    assert len(set(VR_5POINT_BODIES)) == 5  # no duplicate tracked body


def test_five_points_match_maskedmimics_five_point_after_alias_resolution() -> None:
    """The user pointed at exp:g1-wbt-maskedmimic's 5pt to define WHERE the five points are.
    Resolve that catalogue's aliases and assert the same body set — so this preset cannot
    quietly drift to a different five (e.g. knees instead of ankles)."""
    mm = pytest.importorskip("wbt_training.config_values.maskedmimic_observations")
    from wbt_wrappers_inference.config_values.robot_config import VR_5POINT_BODIES

    modes = mm.DEFAULT_HOVER_MASK_MODES
    five = modes["five_point"]["keypoints"]
    three = modes["three_point"]["keypoints"]
    # maskedmimic's five_point is its three_point plus two more, same as ours.
    assert tuple(five[: len(three)]) == tuple(three)

    alias = {"torso": "torso_link", "left_hand": "left_wrist_yaw_link", "right_hand": "right_wrist_yaw_link"}
    resolved = [alias.get(k, k) for k in five]
    # Same SET of bodies (maskedmimic orders torso first; the VR presets order hands first).
    assert set(resolved) == set(VR_5POINT_BODIES), f"maskedmimic five_point {resolved} != {VR_5POINT_BODIES}"


def test_5pt_hands_are_on_the_force_tcp_and_feet_carry_no_offset() -> None:
    """The unified family's whole point is that position and force share ONE anchor. Both
    hands must sit at the 0.242 m fingertip TCP, the head keeps its +0.35 z lift, and the
    feet must be BARE origins (no invented sole lever)."""
    from wbt_wrappers_inference.config_values.robot_config import (
        TCP_UNIFIED_HAND_OFFSET_X,
        VR_3POINT_OFFSETS_TCP,
        VR_5POINT_OFFSETS_TCP,
    )

    from wbt_realtime_eval.main import _TCP_OFFSET_X

    assert len(VR_5POINT_OFFSETS_TCP) == 5
    # 3pt prefix, byte-identical (built by unpacking, not retyping)
    assert VR_5POINT_OFFSETS_TCP[:3] == [list(o) for o in VR_3POINT_OFFSETS_TCP]
    # hands on the TCP the force stack acts on — and the SAME number main.py anchors the
    # wall / arrows / white marker at, so the spheres are co-located in the browser.
    assert VR_5POINT_OFFSETS_TCP[0] == [TCP_UNIFIED_HAND_OFFSET_X, 0.0, 0.0]
    assert VR_5POINT_OFFSETS_TCP[1] == [TCP_UNIFIED_HAND_OFFSET_X, 0.0, 0.0]
    assert TCP_UNIFIED_HAND_OFFSET_X == _TCP_OFFSET_X == 0.242
    # head unchanged
    assert VR_5POINT_OFFSETS_TCP[2] == [0.0, 0.0, 0.35]
    # feet bare
    assert VR_5POINT_OFFSETS_TCP[3] == [0.0, 0.0, 0.0]
    assert VR_5POINT_OFFSETS_TCP[4] == [0.0, 0.0, 0.0]


# ── the bake ─────────────────────────────────────────────────────────────────


@pytest.mark.skipif(not _FIRST_MOTION.exists(), reason="motion fixtures unavailable")
def test_bake_emits_the_5pt_tables_with_the_3pt_tables_as_an_exact_prefix(tmp_path) -> None:
    """vr5_pos_tcp_local / vr5_orn_local are (T, 15) / (T, 20), and their first 9 / 12
    columns are BIT-IDENTICAL to vr3_pos_tcp_local / vr3_orn_local.

    Bit-identical, not merely close: the bake builds them by hstack from those very arrays,
    so any drift means someone re-ran FK separately and the 5pt policy is being fed a
    different 3pt prefix than its twin."""
    pytest.importorskip("pinocchio")
    pytest.importorskip("holosoma_inference")

    from wbt_realtime_eval.bake_motions import bake_motion

    out = tmp_path / "baked.npz"
    bake_motion(_FIRST_MOTION, out)
    with np.load(out) as d:
        assert {"vr5_pos_tcp_local", "vr5_orn_local"} <= set(d.files)
        p5, p3 = d["vr5_pos_tcp_local"], d["vr3_pos_tcp_local"]
        o5, o3 = d["vr5_orn_local"], d["vr3_orn_local"]
        T = p3.shape[0]
        assert p5.shape == (T, 15)
        assert o5.shape == (T, 20)
        assert p5.dtype == o5.dtype == np.float32
        np.testing.assert_array_equal(p5[:, :9], p3)
        np.testing.assert_array_equal(o5[:, :12], o3)
        # The two foot blocks must be REAL data, not the zero-fill of a forgotten FK pass.
        assert np.abs(p5[:, 9:15]).max() > 0.1, "foot position columns look zero-filled"
        # ...and unit quaternions per foot.
        for k in (3, 4):
            q = o5[:, 4 * k : 4 * k + 4]
            np.testing.assert_allclose(np.linalg.norm(q, axis=1), 1.0, atol=1e-4)
        # Left and right feet are distinct points (an aliasing bug would make them equal).
        assert not np.allclose(p5[:, 9:12], p5[:, 12:15], atol=1e-3)


@pytest.mark.skipif(not _FIRST_MOTION.exists(), reason="motion fixtures unavailable")
def test_baked_foot_columns_equal_the_motions_own_recorded_ankle_poses(tmp_path) -> None:
    """Independent numeric check of the foot FK: the baked pelvis-local foot positions must
    equal the SOURCE NPZ's own recorded world ankle poses, rotated into the pelvis frame.

    This validates two things a shape test cannot: that the right BODIES were resolved
    (ankle_roll, not ankle_pitch or the knee) and that the offset really is zero — a nonzero
    one would show up as a constant lever between the two."""
    pytest.importorskip("pinocchio")
    quat = pytest.importorskip("holosoma_inference.utils.math.quat")

    from wbt_realtime_eval.bake_motions import bake_motion

    out = tmp_path / "baked.npz"
    bake_motion(_FIRST_MOTION, out)
    with np.load(out) as d:
        p5 = d["vr5_pos_tcp_local"]
    with np.load(_FIRST_MOTION) as s:
        names = [str(n) for n in s["body_names"].tolist()]
        pelvis = names.index("pelvis")
        q_pelvis = np.ascontiguousarray(s["body_quat_w"][:, pelvis], dtype=np.float32)
        for name, col in (("left_ankle_roll_link", 9), ("right_ankle_roll_link", 12)):
            bi = names.index(name)
            diff = np.ascontiguousarray(s["body_pos_w"][:, bi] - s["body_pos_w"][:, pelvis], dtype=np.float32)
            expected = quat.quat_rotate_inverse(q_pelvis, diff)
            np.testing.assert_allclose(p5[:, col : col + 3], expected, atol=1e-5, err_msg=name)
        # Contrast: the HAND slot must NOT equal the bare wrist origin — it carries the
        # 0.242 m TCP lever. If it did, the hands would have been baked without their offset.
        wi = names.index("left_wrist_yaw_link")
        wdiff = np.ascontiguousarray(s["body_pos_w"][:, wi] - s["body_pos_w"][:, pelvis], dtype=np.float32)
        wrist_origin = quat.quat_rotate_inverse(q_pelvis, wdiff)
        assert not np.allclose(p5[:, 0:3], wrist_origin, atol=1e-3)


@pytest.mark.skipif(not _FIRST_MOTION.exists(), reason="motion fixtures unavailable")
def test_a_bake_missing_the_5pt_tables_is_stale_and_gets_rebaked(tmp_path) -> None:
    """A pre-2026-08-27 bake is mtime-newer than its source, so ``bake_motion_dir`` would
    happily reuse it — and the 5pt accessors ZERO-FILL on a missing table, driving all five
    tracked points to the pelvis origin with nothing in the UI saying why. Both new keys must
    therefore be in ``_is_current_browser_bake``'s required set. Mirrors the
    vr3_pos_tcp_local guard, pinning each key independently."""
    pytest.importorskip("pinocchio")
    pytest.importorskip("holosoma_inference")

    from wbt_realtime_eval.bake_motions import _is_current_browser_bake, bake_motion, bake_motion_dir

    src_dir = tmp_path / "src"
    out_dir = tmp_path / "out"
    src_dir.mkdir()
    (src_dir / "clip.npz").write_bytes(_FIRST_MOTION.read_bytes())

    for key in ("vr5_pos_tcp_local", "vr5_orn_local"):
        baked = out_dir / "clip.npz"
        bake_motion(src_dir / "clip.npz", baked)
        assert _is_current_browser_bake(baked), "a freshly baked NPZ must count as current"
        with np.load(baked) as d:
            assert key in d.files
            pruned = {k: d[k] for k in d.files if k != key}
        np.savez(baked, **pruned)
        assert not _is_current_browser_bake(baked), (
            f"a bake lacking {key} must NOT count as current — otherwise the mtime shortcut "
            "reuses it and the 5pt obs reads zeros."
        )
        # ...and the dir-level bake must actually restore it.
        outs = bake_motion_dir(src_dir, out_dir)
        assert outs == [baked]
        with np.load(baked) as d:
            assert key in d.files, f"re-bake must restore {key}"


# ── main.py: markers + the tracking-family guard ─────────────────────────────


def test_5pt_marker_gate_is_obs_key_driven() -> None:
    """The five tracking spheres are emitted for EXACTLY the presets carrying
    vr_5point_pos_tcp, and the 3pt gate is untouched — the two branches are mutually
    exclusive because a preset carries one position slot or the other, never both."""
    have_5pt = {n for n, p in PRESETS.items() if "vr_5point_pos_tcp" in p.obs_keys}
    assert have_5pt == {_5PT}, sorted(have_5pt)
    have_3pt = {n for n, p in PRESETS.items() if "vr_3point_pos" in p.obs_keys or "vr_3point_pos_tcp" in p.obs_keys}
    assert have_5pt.isdisjoint(have_3pt)
    # The dense preset draws NO tracking spheres — it has no keypoint to mark. That absence
    # is deliberate (judge it against the reference ghost's whole pose instead).
    assert _DENSE not in have_5pt and _DENSE not in have_3pt


def test_5pt_marker_extras_shape_and_colors() -> None:
    from wbt_wrappers_inference.config_values.robot_config import VR_5POINT_BODIES, VR_5POINT_OFFSETS_TCP

    from wbt_realtime_eval.main import (
        _TCP_OFFSET_X,
        _VR_3POINT_MARKER_RGB,
        _vr_5point_tracking_point_markers_extras,
        _vr_tracking_point_markers_extras,
    )

    markers = _vr_5point_tracking_point_markers_extras()
    assert len(markers) == 5
    assert [m["body_name"] for m in markers] == list(VR_5POINT_BODIES)
    for m, off in zip(markers, VR_5POINT_OFFSETS_TCP, strict=True):
        assert m["offset_b"] == [float(o) for o in off]
        # real vs reference sphere of a point share the hue (light-vs-solid reads as
        # target-vs-actual), and every channel is a valid 0-1 RGB triple.
        assert m["real_color"] == m["ref_color"]
        assert len(m["real_color"]) == 3 and all(0.0 <= c <= 1.0 for c in m["real_color"])
        assert m["radius"] > 0
    # Hands co-located with the white force marker; feet bare.
    assert markers[0]["offset_b"] == markers[1]["offset_b"] == [_TCP_OFFSET_X, 0.0, 0.0]
    assert markers[2]["body_name"] == "torso_link" and markers[2]["offset_b"] == [0.0, 0.0, 0.35]
    assert markers[3]["offset_b"] == markers[4]["offset_b"] == [0.0, 0.0, 0.0]
    # Hands/head keep the 3pt hues, so a 5pt bundle reads like a 3pt one plus two feet...
    tcp3 = _vr_tracking_point_markers_extras(tcp_unified=True)
    for a, b in zip(tcp3, markers[:3], strict=True):
        assert a["real_color"] == b["real_color"] and a["radius"] == b["radius"]
        assert a["offset_b"] == b["offset_b"] and a["body_name"] == b["body_name"]
    # ...and the two feet are visually DISTINCT from every hand/head hue and from each other,
    # so no sphere can be mistaken for another.
    hues = [tuple(m["real_color"]) for m in markers]
    assert len(set(hues)) == 5
    assert set(hues[3:]).isdisjoint({tuple(c) for c in _VR_3POINT_MARKER_RGB})
    # The 3pt (0.18 m) marker path is completely undisturbed by any of this.
    assert _vr_tracking_point_markers_extras()[0]["offset_b"] == [0.18, -0.025, 0.0]


def test_tracking_family_guard_rejects_the_732d_collision_both_ways() -> None:
    """The load-bearing guard: a 3pt-hybrid-global-vel checkpoint in the DENSE preset (and
    vice versa) must be REFUSED. Both are 732D hybrid presets, so nothing else stops them."""
    from wbt_realtime_eval.main import _assert_checkpoint_tracking_family

    dense_meta = _meta_with_actor_terms(*_7R54008A_TERMS)
    keypoint_meta = _meta_with_actor_terms(
        "obs00_command_lower_body", "obs01_vr_3point_pos", "obs02_vr_3point_orn_6d", "obs05_force_cmd_local"
    )
    with pytest.raises(ValueError, match="TRACKING-FAMILY mismatch"):
        _assert_checkpoint_tracking_family(keypoint_meta, get_preset(_DENSE))
    with pytest.raises(ValueError, match="TRACKING-FAMILY mismatch"):
        _assert_checkpoint_tracking_family(dense_meta, get_preset(_GLOBAL_VEL))
    # A 5pt checkpoint in the dense preset is the same class of error.
    with pytest.raises(ValueError, match="TRACKING-FAMILY mismatch"):
        _assert_checkpoint_tracking_family(_meta_with_actor_terms(*_DLCXC59D_TERMS), get_preset(_DENSE))


def test_tracking_family_guard_accepts_matches_and_degrades_gracefully() -> None:
    """Matching families pass, and unreadable metadata NEVER blocks a build (an absent /
    reshaped experiment_config is not evidence of a mismatch — a pre-metadata checkpoint
    would otherwise become unviewable)."""
    from wbt_realtime_eval.main import _assert_checkpoint_tracking_family

    dense_meta = _meta_with_actor_terms(*_7R54008A_TERMS)
    kp5_meta = _meta_with_actor_terms(*_DLCXC59D_TERMS)
    _assert_checkpoint_tracking_family(dense_meta, get_preset(_DENSE))
    _assert_checkpoint_tracking_family(kp5_meta, get_preset(_5PT))
    _assert_checkpoint_tracking_family(kp5_meta, get_preset(_3PT))  # family agrees; geometry
    # is the other guard's job

    for bad in ({}, {"experiment_config": "not json{"}, {"experiment_config": json.dumps({"observation": {}})}):
        _assert_checkpoint_tracking_family(bad, get_preset(_DENSE))
        _assert_checkpoint_tracking_family(bad, get_preset(_5PT))

    # Every pre-existing preset must still accept a checkpoint of its own family — the guard
    # must not have become a blanket refusal.
    _assert_checkpoint_tracking_family(
        _meta_with_actor_terms("obs0_motion_command", "obs1_motion_ref_ori_b"), get_preset("g1-29dof-holosoma-wbt")
    )
    _assert_checkpoint_tracking_family(
        _meta_with_actor_terms("obs0_command_lower_body", "obs1_vr_2point_pos"), get_preset("g1-29dof-holosoma-2pt-wbt")
    )


def test_vr_geometry_guard_still_separates_the_5pt_and_3pt_spellings() -> None:
    """`_pos_slot` learned the 5pt spelling. A 5pt checkpoint must therefore be recognised as
    5pt (not silently read as a 3pt one), and the 3pt/TCP behaviour must be unchanged."""
    from wbt_realtime_eval.main import _assert_checkpoint_vr_geometry

    kp5_meta = _meta_with_actor_terms(*_DLCXC59D_TERMS)
    _assert_checkpoint_vr_geometry(kp5_meta, get_preset(_5PT))  # match
    with pytest.raises(ValueError, match="hand-geometry MISMATCH"):
        _assert_checkpoint_vr_geometry(kp5_meta, get_preset(_3PT))
    # ...and the pre-existing pre-TCP rejection is untouched.
    pre_tcp = _meta_with_actor_terms("obs00_command_lower_body", "obs01_vr_3point_pos", "obs02_vr_3point_orn")
    with pytest.raises(ValueError, match="hand-geometry MISMATCH"):
        _assert_checkpoint_vr_geometry(pre_tcp, get_preset(_3PT))
    # The dense preset has no position slot at all -> never gated by THIS guard.
    _assert_checkpoint_vr_geometry(_meta_with_actor_terms(*_7R54008A_TERMS), get_preset(_DENSE))


# ── the real checkpoints ─────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("preset_name", "slots"),
    [(_5PT, _DLCXC59D_TERMS), (_DENSE, _7R54008A_TERMS)],
    ids=["dlcxc59d-5pt", "7r54008a-dense"],
)
def test_preset_reproduces_the_trained_checkpoints_slot_order(preset_name: str, slots: tuple[str, ...]) -> None:
    """Guard-the-guard against the REAL runs: each preset's obs_keys must be the checkpoint's
    own obs00.. slot order with the prefixes stripped.

    The obs manager concatenates in ALPHABETICALLY-SORTED key order, so this is the actual
    on-the-wire layout, and a mismatch anywhere past slot 0 misaligns every later term (the
    2pt-force-ext preset was bitten by exactly this). Uses the slot names recorded in each
    run's W&B config so the test needs no checkpoint on disk."""
    preset = get_preset(preset_name)
    assert _stripped(slots) == list(preset.obs_keys)
    # Zero-padded two-digit keys, so string sort == numeric sort (an un-padded obs10 would
    # sort between obs1 and obs2 and silently reorder the vector).
    assert list(slots) == sorted(slots)
    assert all(s[:5] == f"obs{i:02d}" for i, s in enumerate(slots))
    # And both guards accept it.
    from wbt_realtime_eval.main import _assert_checkpoint_tracking_family, _assert_checkpoint_vr_geometry

    meta = _meta_with_actor_terms(*slots)
    _assert_checkpoint_vr_geometry(meta, preset)
    _assert_checkpoint_tracking_family(meta, preset)


# ── wiring: TS classes, registry, launchers ──────────────────────────────────


def test_new_obs_terms_are_registered_with_unique_ts_names() -> None:
    """Every preset obs key must resolve to a registered TS class, and ts_names must be
    globally unique — mjswan's node build (tsc), not register(), is what fails on a
    duplicate, i.e. late and confusingly."""
    from wbt_realtime_eval import obs_registry

    names = [ts for _key, ts, _src in obs_registry._OBS_FUNC_NAMES]
    assert len(names) == len(set(names)), "duplicate ts_name in the obs registry"
    keys = {key for key, _ts, _src in obs_registry._OBS_FUNC_NAMES}
    assert {"vr_5point_pos_tcp", "vr_5point_orn"} <= keys
    # No preset may reference an unregistered term (this is what would strand a build).
    for name, p in PRESETS.items():
        missing = [k for k in p.obs_keys if k not in keys]
        assert not missing, f"{name} references unregistered obs term(s) {missing}"


def test_ts_obs_classes_and_getters_exist_with_the_right_widths() -> None:
    """The TS side must declare the two obs classes AND the two WbtTrackingLike getters.

    An undeclared getter is a hard `tsc` error (TS2551: WbtTrackingLike intersects the strict
    CommandTerm interface, which has no index signature), and it only surfaces at bundle-build
    time — minutes into a build, after the ONNX download and the 20-clip bake. Checked as text
    because there is no TS toolchain in the Python test env."""
    obs_ts = (_TEMPLATE / "WbtObservations.ts").read_text()
    cmd_ts = (_TEMPLATE / "WbtTrackingCommand.ts").read_text()

    for cls, width, getter in (
        ("WbtVr5PointPosTcp", 15, "getVr5PosTcpLocal"),
        ("WbtVr5PointOrn", 20, "getVr5OrnLocal"),
    ):
        assert f"export class {cls} extends WbtHistoryBase" in obs_ts, cls
        assert f"super(runner, config, {width});" in obs_ts, f"{cls} width {width}"
        assert f"  {getter}(): Float32Array;" in obs_ts, f"{getter} must be declared on WbtTrackingLike"
        assert f"tr.{getter}()" in obs_ts, f"{cls} must read {getter}"
        assert f"{getter}(): Float32Array {{" in cmd_ts, f"{getter} must be implemented on WbtTrackingCommand"

    # The command term must parse + width-validate both baked tables, and slice them at the
    # right width (a wrong width reads across frame boundaries).
    for key, width in (("vr5_pos_tcp_local", 15), ("vr5_orn_local", 20)):
        assert f"npz['{key}']" in cmd_ts, key
        assert f"!== {width}" in cmd_ts, f"{key} width validation"
    assert "vr5PosTcpLocal ?? null, 15" in cmd_ts
    assert "vr5OrnLocal ?? null, 20" in cmd_ts
    # NO fallback to the 9D 3pt table: slicing a 9-wide buffer at width 15 would read
    # garbage across frames, which is worse than the visible zero-fill.
    assert "vr5PosTcpLocal ?? this.activeBaked?.vr3PosTcpLocal" not in cmd_ts


def test_both_launchers_exist_and_pin_their_run_and_a_free_port() -> None:
    """Each new twin ships its own launcher, pinned to its own preset / canonical W&B run /
    unique port. (test_presets.py::test_local_unified_launchers_exist_and_are_wired asserts
    the same wiring inside the shared exhaustive sweep; this keeps the two runs' ids next to
    the rest of the 5pt/dense evidence.)"""
    import re

    runs = _HERE.parents[1] / "wbt_training" / "training_runs"
    expected = {
        "run_hybrid_5pt_unified_anneal_realtime_viz.sh": (_5PT, "dlcxc59d", "8012"),
        "run_hybrid_dense_unified_anneal_realtime_viz.sh": (_DENSE, "7r54008a", "8013"),
    }
    used_elsewhere = set()
    for sh in sorted(runs.glob("run_*realtime_viz.sh")):
        if sh.name in expected:
            continue
        m = re.search(r'^PORT="\$\{PORT:-(\d+)\}"', sh.read_text(), re.MULTILINE)
        if m:
            used_elsewhere.add(m.group(1))

    for fname, (preset_name, run_id, port) in expected.items():
        path = runs / fname
        assert path.is_file(), f"missing launcher {fname}"
        assert path.stat().st_mode & 0o111, f"{fname} must be executable"
        text = path.read_text()
        assert f'PRESET="${{PRESET:-{preset_name}}}"' in text
        assert preset_name in PRESETS
        assert 'DEFAULT_WANDB_RUN=""' in text, f"{fname} must pin no W&B run (was {run_id})"
        assert f'PORT="${{PORT:-{port}}}"' in text
        assert port not in used_elsewhere, f"{fname} port {port} collides with another launcher"
        assert "wbt_realtime_viz_main" in text
        assert 'source "$SCRIPT_DIR/_realtime_viz_common.sh"' in text


# ── blast radius ─────────────────────────────────────────────────────────────


def test_no_other_preset_changed() -> None:
    """Positive "不影响其他 preset" guard: every pre-2026-08-27 preset keeps its exact term
    tuple and width. Pinned by VALUE so an accidental edit to a shared tuple (the 5pt layout
    is derived from the 3pt one by substitution, so a typo there could mutate both) fails
    here rather than on someone's next viz run."""
    expected_dims = {
        "g1-29dof-holosoma-wbt": 628,
        "g1-29dof-holosoma-2pt-wbt": 564,
        "g1-29dof-holosoma-2pt-wbt-live": 564,
        "g1-29dof-holosoma-2pt-force-wbt": 588,
        "g1-29dof-holosoma-2pt-force-wbt-live": 588,
        "g1-29dof-holosoma-2pt-force-ext-wbt": 612,
        "g1-29dof-holosoma-2pt-hybrid-wbt": 660,
        "g1-29dof-holosoma-2pt-hybrid-layerA-wbt": 660,
        "g1-29dof-holosoma-3pt-hybrid-wbt": 672,
        "g1-29dof-holosoma-3pt-hybrid-layerA-wbt": 672,
        "g1-29dof-holosoma-3pt-hybrid-global-vel-wbt": 732,
        "g1-29dof-holosoma-3pt-hybrid-global-vel-layerA-wbt": 732,
        "g1-29dof-holosoma-3pt-hybrid-global-vel-unified-wbt": 740,
        "g1-29dof-holosoma-3pt-hybrid-global-vel-unified-anneal-wbt": 740,
        "g1-29dof-holosoma-3pt-hybrid-unified-wbt": 680,
        "g1-29dof-holosoma-3pt-hybrid-unified-anneal-wbt": 680,
        # The LIVE VIVE-teleop twin of the 170D preset (2026-08-26). Same width, and
        # its tuple is derived from the 170D one by the same substitution trick the 5pt
        # tuple uses — so it belongs in exactly this pin.
        "g1-29dof-holosoma-3pt-hybrid-unified-anneal-wbt-live": 680,
        "g1-29dof-holosoma-5pt-hybrid-unified-anneal-wbt-live": 736,
        "g1-29dof-holosoma-dense-hybrid-unified-anneal-wbt-live": 732,
        # LIVE twin of the BASE tracker exp:g1-wbt (2026-08-29). Same 628D as its replay
        # twin at the top of this table — the tuple is that one with `command` ->
        # `command_live`, so both belong in this pin and the equality of their widths is
        # itself the guard.
        "g1-29dof-holosoma-wbt-live": 628,
        "g1-29dof-holosoma-3pt-wbt": 576,
        "g1-29dof-holosoma-2pt-global-wbt": 576,
        "g1-29dof-holosoma-3pt-global-wbt": 588,
        "g1-29dof-holosoma-global-wbt": 640,
        "g1-29dof-holosoma-global-vel-wbt": 664,
    }
    for name, dim in expected_dims.items():
        assert get_preset(name).total_dim == dim, name
    # No pre-existing preset may have picked up a 5pt term.
    for name in expected_dims:
        if name.startswith(_5PT):
            continue  # the 5pt pair is SUPPOSED to carry them
        assert not any(k.startswith("vr_5point") for k in get_preset(name).obs_keys), name
    # The registry is exactly the old set plus the two new ones.
    assert set(PRESETS) == set(expected_dims) | {_5PT, _DENSE}


def test_the_3pt_unified_twin_is_byte_identical() -> None:
    """The 5pt layout is DERIVED from the 3pt one (tuple comprehension over its keys), so a
    mistake there could mutate the 170D twin that the live pfl96opy lineage uses."""
    from wbt_realtime_eval.presets import _HOLOSOMA_3PT_HYBRID_UNIFIED_WBT_TERMS as t3

    assert t3 == (
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
    )
    for n in (_3PT, "g1-29dof-holosoma-3pt-hybrid-unified-wbt"):
        assert get_preset(n).obs_keys == t3
        assert get_preset(n).per_frame_dim == 170


def test_existing_bake_tables_keep_their_widths() -> None:
    """The 5pt tables are hstacked FROM the 3pt ones, so the bake now reads those arrays in a
    new place. Pin every pre-existing table's declared width so a refactor there cannot
    quietly resize one (the browser would then throw at motion-select, or worse, mis-slice)."""
    cmd_ts = (_TEMPLATE / "WbtTrackingCommand.ts").read_text()
    for key, width in (
        ("vr_pos_local", 6),
        ("vr_orn_local", 12),
        ("vr3_pos_local", 9),
        ("vr3_pos_tcp_local", 9),
        ("vr3_orn_local", 12),
        ("vr3_orn6d_local", 18),
        ("force_hand_ref_local", 6),
    ):
        assert f"{key} width ${{" in cmd_ts and f"!= {width}`" in cmd_ts, f"{key} != {width}"


@pytest.mark.skipif(not _FIRST_MOTION.exists(), reason="motion fixtures unavailable")
def test_added_bake_keys_are_c_contiguous_in_the_npy_header(tmp_path) -> None:
    """mjswan's browser loadNpz silently DROPS fortran-ordered arrays (surfacing much later as
    'missing baked key'), so both new tables must land with fortran_order=False. hstack of
    C-contiguous inputs is C-contiguous, but the bake pins it with ascontiguousarray and this
    asserts the pin actually reached the file."""
    pytest.importorskip("pinocchio")
    pytest.importorskip("holosoma_inference")
    import zipfile

    from wbt_realtime_eval.bake_motions import bake_motion

    out = tmp_path / "baked.npz"
    bake_motion(_FIRST_MOTION, out)
    with zipfile.ZipFile(out) as z:
        for key in ("vr5_pos_tcp_local", "vr5_orn_local"):
            with z.open(f"{key}.npy") as f:
                raw = f.read(256)
            major = raw[6]
            hstart, hlen = (10, raw[8] | (raw[9] << 8)) if major == 1 else (12, int.from_bytes(raw[8:12], "little"))
            header = raw[hstart : hstart + hlen].decode()
            assert "'fortran_order': False" in header, f"{key}: {header!r}"
