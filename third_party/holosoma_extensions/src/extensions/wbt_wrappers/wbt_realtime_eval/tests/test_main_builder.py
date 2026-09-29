"""End-to-end smoke test for setup_builder.

Verifies that the builder construction pipeline runs without error against a
local 2pt-WBT checkpoint and that bake_motions produced one merged NPZ per
input motion. Skipped on a fresh CI clone where local checkpoints are absent.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[5]
_MOTIONS = _REPO / "src/extensions/wbt_wrappers/wbt_training/motions"
_LOG_ROOT = _REPO / "logs/WBT-Holosoma"

# 2pt-WBT obs dim — the same constant Task 10's test uses.
# 141 per frame (vr_2point_orn is the 12D 6D-rot rep) x 4 history.
_TOTAL_OBS_DIM = 564
# Force preset obs dim: 147 per frame (2pt + force_cmd_local(6)) x 4 history.
_FORCE_OBS_DIM = 588


def _find_checkpoint_with_obs_dim(obs_dim: int) -> Path | None:
    """Return the first locally available model_*.onnx whose 'obs' input has
    last dim == obs_dim, or None."""
    if not _LOG_ROOT.is_dir():
        return None
    onnxruntime = pytest.importorskip("onnxruntime")
    for onnx_path in sorted(_LOG_ROOT.glob("*/model_*.onnx")):
        try:
            sess = onnxruntime.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        except Exception:
            continue
        obs_input = next((i for i in sess.get_inputs() if i.name == "obs"), None)
        if obs_input is None:
            continue
        last_dim = obs_input.shape[-1]
        if isinstance(last_dim, int) and last_dim == obs_dim:
            return onnx_path
    return None


def _find_2pt_checkpoint() -> Path | None:
    """Return the first locally available 2pt-WBT model_*.onnx, or None."""
    return _find_checkpoint_with_obs_dim(_TOTAL_OBS_DIM)


@pytest.mark.skipif(not _MOTIONS.is_dir(), reason="local motions dir required")
def test_setup_builder_runs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("mjswan")
    pytest.importorskip("pinocchio")
    pytest.importorskip("holosoma_inference")

    onnx_path = _find_2pt_checkpoint()
    if onnx_path is None:
        pytest.skip("no 2pt-WBT ONNX checkpoint found locally")
    assert onnx_path is not None  # narrow for mypy after the skip

    # Belt-and-suspenders: don't accidentally launch the dev server in case
    # something hooks main() instead of setup_builder().
    monkeypatch.setenv("MJSWAN_NO_LAUNCH", "1")

    from wbt_realtime_eval.main import setup_builder

    builder = setup_builder(model_path=onnx_path, motion_dir=_MOTIONS, output_dir=tmp_path)

    # Builder structure: 1 project, 1 scene, 1 policy.
    projects = builder.get_projects()
    assert len(projects) == 1, f"expected 1 project, got {len(projects)}"
    project = projects[0]
    assert len(project.scenes) == 1, f"expected 1 scene, got {len(project.scenes)}"
    scene = project.scenes[0]
    assert len(scene.policies) == 1, f"expected 1 policy, got {len(scene.policies)}"
    policy = scene.policies[0]

    # Policy registered the synthetic idle_default + all source NPZs.
    expected_motions = len(list(_MOTIONS.glob("*.npz"))) + 1  # +1 for idle_default
    assert len(policy.motions) == expected_motions, (
        f"expected one synthetic idle_default + one bundle motion per source NPZ "
        f"({expected_motions}), got {len(policy.motions)}"
    )
    assert policy.motions[0].name == "idle_default", (
        f"synthetic default motion must be first; got {policy.motions[0].name}"
    )

    # Default motion is idle_stand_1 when present in the source dir
    # (existing 2pt fixture has it). Synthetic idle_default takes the
    # default slot only when no user-supplied idle_stand_1 exists.
    defaults = [m for m in policy.motions if m.default]
    assert len(defaults) == 1, f"expected exactly 1 default motion, got {len(defaults)}"
    assert defaults[0].name == "idle_stand_1"

    assert policy.config_path is not None
    policy_config = json.loads(Path(policy.config_path).read_text())
    # Obs group name must match the primary ONNX input name so that
    # mjswan's runtime correctly assigns input[group] = obs[group].
    # Different 2pt checkpoints use either "policy" or "obs".
    obs_groups = set(policy_config["observations"])
    assert obs_groups in ({"policy"}, {"obs"}), obs_groups
    obs_group = next(iter(obs_groups))
    policy_obs = policy_config["observations"][obs_group]
    assert obs_group == policy_config["onnx"]["meta"]["in_keys"][0]
    # Each obs class owns its own per-term 4-frame history; group-level stack=1.
    assert policy_obs["history_steps"] == 1
    # 9 policy obs terms + 1 metrics tap (size=0)
    assert len(policy_obs["components"]) == 10
    # Per-term stacked dim = base_dim * 4 (history within each obs class).
    base_dims = {
        "WbtCommandLowerBody": 24,
        "WbtVr2PointPos": 6,
        "WbtVr2PointOrn": 12,
        "WbtMotionRefOriB": 6,
        "WbtProjectedGravity": 3,
        "WbtBaseAngVel": 3,
        "WbtJointPos": 29,
        "WbtJointVel": 29,
        "WbtPrevActions": 29,
        "WbtMetricsTap": 0,
    }
    assert sum(base_dims[entry["name"]] for entry in policy_obs["components"]) * 4 == _TOTAL_OBS_DIM
    # ONNX I/O is auto-detected from the session.  Different 2pt-WBT
    # checkpoints in the wild use either "policy" or "obs" as the obs input
    # name, and either "action" or "actions" (plus extra diagnostic outputs)
    # for the output.  Whichever name the actual file uses must round-trip
    # into the policy config.  ``_strip_policy_onnx`` reduces a full checkpoint
    # to the obs->action subgraph, DROPPING the unused ``time_step`` input, so
    # the served config's in_keys is the single obs/policy key (an
    # already-minimal model is a no-op and is likewise single-input).
    in_keys = policy_config["onnx"]["meta"]["in_keys"]
    primary = [k for k in in_keys if k != "time_step"]
    assert len(primary) == 1, in_keys
    assert primary[0] in {"policy", "obs"}
    out_keys = policy_config["onnx"]["meta"]["out_keys"]
    assert any(k in {"action", "actions"} for k in out_keys), out_keys
    # WbtJointPos must receive the ONNX-baked default_joint_angles for correct
    # joint_pos offset (matches HolosomaWBTPolicy._compute_obs_terms).
    jp_entry = next(c for c in policy_obs["components"] if c["name"] == "WbtJointPos")
    assert "default_joint_angles" in jp_entry, "WbtJointPos missing default_joint_angles param"
    assert len(jp_entry["default_joint_angles"]) == 29
    # Extras must surface normalizer mean/std and default angles. They're
    # stored at the PolicyConfig level (set via add_policy(extras=...)) and
    # only get inlined into the JSON during builder.build(); check the
    # in-memory PolicyConfig directly.
    extras = policy.extras or {}
    assert "normalizer_mean" in extras and len(extras["normalizer_mean"]) == _TOTAL_OBS_DIM
    assert "normalizer_std" in extras and len(extras["normalizer_std"]) == _TOTAL_OBS_DIM
    assert extras.get("normalizer_clip") == 3.0
    assert "default_joint_angles" in extras and len(extras["default_joint_angles"]) == 29

    # bake_motions writes to a SIBLING of output_dir (output_dir is wiped by
    # builder.build before assets are copied). The naming convention is
    # `<output_dir.name>_baked` next to output_dir.
    baked_dir = tmp_path.parent / f"{tmp_path.name}_baked"
    assert baked_dir.is_dir(), f"baked dir missing: {baked_dir}"
    # baked_dir holds one merged NPZ per source motion, PLUS the injected
    # idle_default.npz and its 1-frame source _idle_default_source.npz.
    source_baked = sorted(p for p in baked_dir.glob("*.npz") if p.stem not in ("idle_default", "_idle_default_source"))
    assert len(source_baked) == len(list(_MOTIONS.glob("*.npz"))), (
        f"expected {len(list(_MOTIONS.glob('*.npz')))} source-derived baked NPZs, got {len(source_baked)}"
    )
    idle_baked = baked_dir / "idle_default.npz"
    assert idle_baked.is_file(), "injected idle_default.npz must be baked into baked_dir"
    # idle_default must carry REAL FK tables (not the old zero-fill) so it HOLDS
    # the hand-raised default pose: its VR target must be away from the pelvis.
    import numpy as np

    with np.load(idle_baked) as d:
        assert "force_hand_ref_local" in d.files
        vr = d["vr_pos_local"][0]  # [L_xyz, R_xyz]
        assert np.linalg.norm(vr[:3]) > 0.2 and np.linalg.norm(vr[3:]) > 0.2, (
            f"idle_default VR target collapsed to the pelvis ({vr}); the default pose hands must sit ~0.35 m out"
        )
        # Left hand +y, right hand -y (per-hand sign sanity).
        assert vr[1] > 0 and vr[4] < 0, f"idle_default L/R y-sign wrong: {vr}"

    # idle_default is now a real baked source file (NOT data=bytes); every
    # motion under the policy must reference a baked NPZ.
    baked_path_set = {str(p.resolve()) for p in baked_dir.glob("*.npz")}
    idle_count = 0
    for motion in policy.motions:
        if motion.name == "idle_default":
            idle_count += 1
        assert motion.source is not None, f"motion {motion.name} has no source path"
        assert str(Path(motion.source).resolve()) in baked_path_set, (
            f"motion {motion.name} source {motion.source!r} not in baked dir"
        )
    assert idle_count == 1, f"expected exactly one idle_default motion, got {idle_count}"


@pytest.mark.skipif(not _MOTIONS.is_dir(), reason="local motions dir required")
def test_setup_builder_force_preset(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Force preset (g1-29dof-holosoma-2pt-force-wbt) wiring contract.

    Guards the entire force feature: 588-dim obs with force_cmd_local as the
    4th term, the F^ext-not-in-obs invariant, 12 live sliders (6 F^cmd + 6
    F^ext) with disjoint key prefixes, force_ext_bodies injection config, and
    force_viz_bodies dual-arrow config. Skipped when no 588-dim checkpoint is
    available locally.
    """
    pytest.importorskip("mjswan")
    pytest.importorskip("pinocchio")
    pytest.importorskip("holosoma_inference")

    onnx_path = _find_checkpoint_with_obs_dim(_FORCE_OBS_DIM)
    if onnx_path is None:
        pytest.skip(f"no {_FORCE_OBS_DIM}-dim force checkpoint found locally")
    assert onnx_path is not None  # narrow for mypy after the skip

    monkeypatch.setenv("MJSWAN_NO_LAUNCH", "1")
    from wbt_realtime_eval.main import setup_builder

    builder = setup_builder(
        model_path=onnx_path, motion_dir=_MOTIONS, output_dir=tmp_path, preset="g1-29dof-holosoma-2pt-force-wbt"
    )
    policy = builder.get_projects()[0].scenes[0].policies[0]
    policy_config = json.loads(Path(policy.config_path).read_text())
    obs_group = next(iter(policy_config["observations"]))
    policy_obs = policy_config["observations"][obs_group]

    # 10 force obs terms + 1 metrics tap. force_cmd_local is the 4th obs term.
    comp_names = [c["name"] for c in policy_obs["components"]]
    assert len(comp_names) == 11, f"expected 11 components (10 obs + tap), got {comp_names}"
    assert comp_names[3] == "WbtForceCmdLocal", f"force_cmd_local must be the 4th obs term; got {comp_names}"
    # F^ext is NEVER an obs class — WbtForceCmdLocal is the ONLY force obs term.
    assert comp_names.count("WbtForceCmdLocal") == 1
    assert not any("Ext" in n or "ext" in n for n in comp_names), f"no F^ext obs class may exist; got {comp_names}"
    base_dims = {
        "WbtCommandLowerBody": 24,
        "WbtVr2PointPos": 6,
        "WbtVr2PointOrn": 12,
        "WbtForceCmdLocal": 6,
        "WbtMotionRefOriB": 6,
        "WbtProjectedGravity": 3,
        "WbtBaseAngVel": 3,
        "WbtJointPos": 29,
        "WbtJointVel": 29,
        "WbtPrevActions": 29,
        "WbtMetricsTap": 0,
    }
    assert sum(base_dims[n] for n in comp_names) * 4 == _FORCE_OBS_DIM

    extras = policy.extras or {}
    # Normalizer length must match the 588-D force obs.
    assert len(extras["normalizer_mean"]) == _FORCE_OBS_DIM
    # 12 sliders: 6 F^cmd + 6 F^ext, all keys unique.
    sliders = extras["sliders"]
    assert len(sliders) == 12, f"expected 12 sliders, got {len(sliders)}"
    keys = [s["key"] for s in sliders]
    assert len(set(keys)) == 12, f"slider keys not unique: {keys}"
    cmd_keys = [k for k in keys if k.startswith("wbt_force_cmd_local__")]
    ext_keys = [k for k in keys if k.startswith("wbt_force_ext_local__")]
    assert len(cmd_keys) == 6 and len(ext_keys) == 6, (cmd_keys, ext_keys)
    for s in sliders:
        assert (s["min"], s["max"]) == (-25.0, 25.0), s

    # force_ext_bodies: the physics-injection config. Both wrists, ext-prefixed
    # slider keys ONLY (never the cmd keys), correct per-hand offsets.
    feb = extras["force_ext_bodies"]
    assert [b["body_name"] for b in feb] == ["left_wrist_yaw_link", "right_wrist_yaw_link"]
    assert feb[0]["offset_b"] == [0.0815, 0.003, 0.0]
    assert feb[1]["offset_b"] == [0.0815, -0.003, 0.0]
    for b in feb:
        assert all(k.startswith("wbt_force_ext_local__") for k in b["slider_keys"]), b

    # force_viz_bodies: the dual-arrow config in the motion command params.
    motion_cmd = policy.commands["motion"]
    fvb = motion_cmd.params["force_viz_bodies"]
    assert len(fvb) == 2
    for b in fvb:
        assert all(k.startswith("wbt_force_cmd_local__") for k in b["slider_keys"]), b
        assert all(k.startswith("wbt_force_ext_local__") for k in b["ext_slider_keys"]), b
        assert b["arrow_color_ext"] == [1.0, 0.45, 0.0]  # orange
        assert b["arrow_color_cmd"] == [0.62, 0.20, 0.92]  # purple


# --------------------------------------------------------------------------- #
# The force-extras gate keys on the OBS LAYOUT (force_cmd_local present), not the
# preset NAME, so BOTH the replay force preset and its live-VIVE variant get the
# sliders / F^ext injection / arrows. These two tests guard (a) that the
# generalization didn't silently widen to a non-force preset, and (b) that the
# live force preset combines live VR + hand-frame F^cmd + F^ext.
# --------------------------------------------------------------------------- #


def test_force_extras_gate_is_obs_key_driven() -> None:
    """Every preset that carries force_cmd_local inherits the F^cmd/F^ext slider +
    arrow extras from the obs-key gate in main.setup_builder; no OTHER preset does.
    Checkpoint-free (reads the preset registry only) so it always runs in CI.

    The force family that carries force_cmd_local is: the plain force preset + its
    live variant, the F^ext-in-actor preset (g1-29dof-holosoma-2pt-force-ext-wbt),
    the two 2pt-hybrid presets, the two 3pt-hybrid presets, the two
    3pt-hybrid-global-vel presets (obs*_force_cmd_local in hybrid.py / hybrid_3pt.py /
    hybrid_3pt_global_vel.py), the six C1 unified twins (the two global-vel 185D, the
    two pelvis-local 170D, and the 5pt 184D / dense 183D, hybrid_unified.py +
    hybrid_unified_5pt_dense.py), and the LIVE VIVE-teleop 170D twin — all get the
    shared slider/F^ext/arrow wiring. This set is updated deliberately whenever a new
    force/hybrid preset is added."""
    from wbt_realtime_eval.presets import PRESETS

    have_force_term = {name for name, p in PRESETS.items() if "force_cmd_local" in p.obs_keys}
    assert have_force_term == {
        "g1-29dof-holosoma-2pt-force-wbt",
        "g1-29dof-holosoma-2pt-force-wbt-live",
        "g1-29dof-holosoma-2pt-force-ext-wbt",
        "g1-29dof-holosoma-2pt-hybrid-wbt",
        "g1-29dof-holosoma-2pt-hybrid-layerA-wbt",
        "g1-29dof-holosoma-3pt-hybrid-wbt",
        "g1-29dof-holosoma-3pt-hybrid-layerA-wbt",
        "g1-29dof-holosoma-3pt-hybrid-global-vel-wbt",
        "g1-29dof-holosoma-3pt-hybrid-global-vel-layerA-wbt",
        "g1-29dof-holosoma-3pt-hybrid-global-vel-unified-wbt",
        "g1-29dof-holosoma-3pt-hybrid-global-vel-unified-anneal-wbt",
        "g1-29dof-holosoma-3pt-hybrid-unified-wbt",
        "g1-29dof-holosoma-3pt-hybrid-unified-anneal-wbt",
        # 2026-08-27: the 5pt and DENSE unified anneal twins. Both carry the SAME force
        # block as the 170D twin (that is the whole design claim), so both inherit the
        # slider / F^ext / 3-arrow wiring — intended, not incidental.
        "g1-29dof-holosoma-5pt-hybrid-unified-anneal-wbt",
        "g1-29dof-holosoma-dense-hybrid-unified-anneal-wbt",
        # The LIVE VIVE-teleop twin of the 170D preset: same obs but for the two VR
        # slots, so it takes the same force wiring.
        "g1-29dof-holosoma-3pt-hybrid-unified-anneal-wbt-live",
        # 2026-08-27: the LIVE motion-completion twins of the 5pt / dense presets.
        # 3 trackers -> Motion_Completion -> 5 keypoints / 29 joint targets; the force
        # block is the SAME object, so they inherit the same wiring.
        "g1-29dof-holosoma-5pt-hybrid-unified-anneal-wbt-live",
        "g1-29dof-holosoma-dense-hybrid-unified-anneal-wbt-live",
    }, (
        "The force-extras gate in main.setup_builder keys on "
        "'force_cmd_local' in obs_keys; every force/hybrid preset must match. "
        f"Got: {sorted(have_force_term)}. A new preset that adds force_cmd_local "
        "would inherit the slider/F^ext wiring — update this guard intentionally."
    )


def test_vr_tracking_point_markers_gate_is_obs_key_driven() -> None:
    """The vr_3point tracking-point markers (left hand / right hand / head, on both
    the live robot and the reference ghost) are emitted by main.setup_builder for
    EXACTLY the presets whose actor obs carries a vr_3point POSITION slot — either
    spelling: `vr_3point_pos` (0.18 m VR lever) or `vr_3point_pos_tcp` (0.242 m force
    TCP, the TCP-unified local twins). No other preset gets them. Checkpoint-free
    (reads the preset registry only) so it always runs in CI.

    This is the load-bearing "不相互影响" guard: only the 3pt family draws the
    markers, so every 2pt / force / global / maskmimic preset stays byte-identical."""
    from wbt_realtime_eval.presets import PRESETS

    have_3pt = {
        name for name, p in PRESETS.items() if "vr_3point_pos" in p.obs_keys or "vr_3point_pos_tcp" in p.obs_keys
    }
    assert have_3pt == {
        "g1-29dof-holosoma-3pt-wbt",
        "g1-29dof-holosoma-3pt-global-wbt",
        "g1-29dof-holosoma-3pt-hybrid-wbt",
        "g1-29dof-holosoma-3pt-hybrid-layerA-wbt",
        "g1-29dof-holosoma-3pt-hybrid-global-vel-wbt",
        "g1-29dof-holosoma-3pt-hybrid-global-vel-layerA-wbt",
        # C1 unified twins: all four draw the 3 tracking-point markers. The global-vel
        # pair keeps `vr_3point_pos` (0.18 m; it swaps only the ORIENTATION rep to 6D);
        # the pelvis-local pair carries `vr_3point_pos_tcp` (0.242 m TCP-unified hands,
        # user 2026-08-03) and its markers move onto the TCP accordingly.
        "g1-29dof-holosoma-3pt-hybrid-global-vel-unified-wbt",
        "g1-29dof-holosoma-3pt-hybrid-global-vel-unified-anneal-wbt",
        "g1-29dof-holosoma-3pt-hybrid-unified-wbt",
        "g1-29dof-holosoma-3pt-hybrid-unified-anneal-wbt",
    }, (
        "The vr_3point tracking-point marker gate in main.setup_builder keys on "
        "'vr_3point_pos'/'vr_3point_pos_tcp' in obs_keys; every 3pt preset must match "
        f"and no 2pt/other preset may. Got: {sorted(have_3pt)}. A new 3pt preset would "
        "inherit the markers — update this guard intentionally."
    )
    # Exactly the two local unified twins are on the TCP-unified geometry.
    tcp_presets = {name for name, p in PRESETS.items() if "vr_3point_pos_tcp" in p.obs_keys}
    assert tcp_presets == {
        "g1-29dof-holosoma-3pt-hybrid-unified-wbt",
        "g1-29dof-holosoma-3pt-hybrid-unified-anneal-wbt",
    }, f"TCP-unified geometry must be scoped to the two local unified twins. Got: {sorted(tcp_presets)}"


def test_vr_tracking_point_markers_extras_shape() -> None:
    """The emitted marker config is the THREE vr_3point tracked points in
    VR_3POINT_BODIES order (left hand, right hand, head), each with a body + offset
    sourced from the single-source-of-truth VR_3POINT_OFFSETS and a real/ref color
    pair that shares the same hue (target-vs-actual reads as light-vs-solid)."""
    from wbt_realtime_eval.main import _vr_tracking_point_markers_extras
    from wbt_wrappers_inference.config_values.robot_config import VR_3POINT_BODIES, VR_3POINT_OFFSETS

    markers = _vr_tracking_point_markers_extras()
    assert [m["body_name"] for m in markers] == list(VR_3POINT_BODIES)
    for m, off in zip(markers, VR_3POINT_OFFSETS, strict=True):
        assert m["offset_b"] == [float(o) for o in off]
        # real and reference spheres of a point share the hue (light vs solid, not
        # different colors), and every color channel is a valid 0-1 RGB triple.
        assert m["real_color"] == m["ref_color"]
        assert len(m["real_color"]) == 3 and all(0.0 <= c <= 1.0 for c in m["real_color"])
        assert m["radius"] > 0
    # The head point is torso_link + 0.35 m up (≈ HMD), NOT a wrist lever.
    assert markers[-1]["body_name"] == "torso_link"
    assert markers[-1]["offset_b"] == [0.0, 0.0, 0.35]


def test_vr_tracking_point_markers_extras_tcp_unified() -> None:
    """`tcp_unified=True` (exp:g1-wbt-3pt-hybrid-unified[-anneal]) puts the two HAND
    markers on the 0.242 m force TCP — co-located with the white force sphere — while the
    head marker and all colors/radii are unchanged. This is what makes the browser show
    position and force at ONE point instead of two anchors 6.2 cm apart."""
    from wbt_realtime_eval.main import _TCP_OFFSET_X, _vr_tracking_point_markers_extras
    from wbt_wrappers_inference.config_values.robot_config import VR_3POINT_BODIES

    markers = _vr_tracking_point_markers_extras(tcp_unified=True)
    assert [m["body_name"] for m in markers] == list(VR_3POINT_BODIES)
    # both hands exactly at the force anchor (tcp_offset_x, 0, 0) — no lateral term
    assert markers[0]["offset_b"] == [_TCP_OFFSET_X, 0.0, 0.0]
    assert markers[1]["offset_b"] == [_TCP_OFFSET_X, 0.0, 0.0]
    # head unchanged
    assert markers[2]["body_name"] == "torso_link"
    assert markers[2]["offset_b"] == [0.0, 0.0, 0.35]
    # default (0.18 m) path is NOT disturbed by the new kwarg
    default = _vr_tracking_point_markers_extras()
    assert default[0]["offset_b"] == [0.18, -0.025, 0.0]
    assert default[1]["offset_b"] == [0.18, 0.025, 0.0]
    # colors/radii identical between the two variants (geometry-only change)
    for a, b in zip(default, markers, strict=True):
        assert a["real_color"] == b["real_color"] and a["radius"] == b["radius"]


def _meta_with_actor_terms(*term_keys: str) -> dict[str, str]:
    """A minimal ONNX-metadata dict carrying just the actor_obs term names, shaped like
    the real serialized experiment_config."""
    return {
        "experiment_config": json.dumps(
            {"observation": {"groups": {"actor_obs": {"terms": {k: {"func": "x"} for k in term_keys}}}}}
        )
    }


def test_checkpoint_vr_geometry_guard_rejects_pre_tcp_checkpoint() -> None:
    """A PRE-TCP (0.18 m) checkpoint loaded into a TCP-unified preset must be REFUSED.

    This is the hole the obs-dim check cannot cover: both geometries are 170D → [1,680],
    so m3s0pw9i / 4o9zbs7i would build cleanly against the 0.242 m bake and then command
    the hands 6.2 cm from where they were trained, with nothing in the UI saying why."""
    from wbt_realtime_eval.main import _assert_checkpoint_vr_geometry
    from wbt_realtime_eval.presets import get_preset

    tcp_preset = get_preset("g1-29dof-holosoma-3pt-hybrid-unified-anneal-wbt")
    pre_tcp_meta = _meta_with_actor_terms("obs00_command_lower_body", "obs01_vr_3point_pos", "obs02_vr_3point_orn")
    with pytest.raises(ValueError, match="hand-geometry MISMATCH"):
        _assert_checkpoint_vr_geometry(pre_tcp_meta, tcp_preset)

    # ... and the converse: a TCP checkpoint in a 0.18 m preset is equally wrong.
    old_preset = get_preset("g1-29dof-holosoma-3pt-hybrid-wbt")
    tcp_meta = _meta_with_actor_terms("obs00_command_lower_body", "obs01_vr_3point_pos_tcp", "obs02_vr_3point_orn")
    with pytest.raises(ValueError, match="hand-geometry MISMATCH"):
        _assert_checkpoint_vr_geometry(tcp_meta, old_preset)


def test_checkpoint_vr_geometry_guard_accepts_matching_and_degrades_gracefully() -> None:
    """The guard passes matching geometry, and NEVER blocks a build merely because the
    metadata is unreadable — an absent/reshaped experiment_config is not evidence of a
    mismatch (a pre-metadata checkpoint would otherwise become unviewable)."""
    from wbt_realtime_eval.main import _assert_checkpoint_vr_geometry
    from wbt_realtime_eval.presets import get_preset

    tcp_preset = get_preset("g1-29dof-holosoma-3pt-hybrid-unified-anneal-wbt")
    old_preset = get_preset("g1-29dof-holosoma-3pt-hybrid-wbt")
    tcp_meta = _meta_with_actor_terms("obs00_command_lower_body", "obs01_vr_3point_pos_tcp")
    old_meta = _meta_with_actor_terms("obs00_command_lower_body", "obs01_vr_3point_pos")
    _assert_checkpoint_vr_geometry(tcp_meta, tcp_preset)  # matching TCP
    _assert_checkpoint_vr_geometry(old_meta, old_preset)  # matching 0.18

    # Unreadable / absent metadata → warn-and-continue, not raise.
    _assert_checkpoint_vr_geometry({}, tcp_preset)
    _assert_checkpoint_vr_geometry({"experiment_config": "not json{"}, tcp_preset)
    _assert_checkpoint_vr_geometry({"experiment_config": json.dumps({"observation": {}})}, tcp_preset)

    # A preset with NO 3pt position slot (2pt / force / global) is never gated.
    _assert_checkpoint_vr_geometry(tcp_meta, get_preset("g1-29dof-holosoma-2pt-force-wbt"))


def test_checkpoint_vr_geometry_guard_matches_the_live_tcp_checkpoint() -> None:
    """Guard-the-guard against the REAL run: pfl96opy (the TCP-unified R1 retrain, the
    launcher's pinned default) must be accepted by the anneal preset. Uses the recorded
    obs00..obs13 slot names from its model_00000.onnx so this runs without the 234 MB
    file present; the launcher-wiring test pins the run id itself."""
    from wbt_realtime_eval.main import _assert_checkpoint_vr_geometry
    from wbt_realtime_eval.presets import get_preset

    pfl96opy_terms = (
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
    )
    preset = get_preset("g1-29dof-holosoma-3pt-hybrid-unified-anneal-wbt")
    _assert_checkpoint_vr_geometry(_meta_with_actor_terms(*pfl96opy_terms), preset)
    # The checkpoint's slot ORDER must also be the preset's term order (the sorted-concat
    # contract): strip the obsNN_ prefixes and compare, with dof_pos/dof_vel renamed to
    # the browser's joint_pos/joint_vel spelling.
    rename = {"dof_pos": "joint_pos", "dof_vel": "joint_vel"}
    stripped = [rename.get(t.split("_", 1)[1], t.split("_", 1)[1]) for t in pfl96opy_terms]
    assert stripped == list(preset.obs_keys), f"pfl96opy layout {stripped} != preset {list(preset.obs_keys)}"


@pytest.mark.skipif(not _MOTIONS.is_dir(), reason="local motions dir required")
def test_setup_builder_live_force_preset(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Live force preset (g1-29dof-holosoma-2pt-force-wbt-live): the 588-D force
    obs with the two VR terms fed LIVE (WbtVr2PointPosLive/OrnLive) + hand-frame
    F^cmd + the full force extras (12 sliders, F^ext injection, dual arrows).
    Mirrors how main_vive.py builds it under --force. Skipped without a 588-D
    checkpoint locally."""
    pytest.importorskip("mjswan")
    pytest.importorskip("pinocchio")
    pytest.importorskip("holosoma_inference")

    onnx_path = _find_checkpoint_with_obs_dim(_FORCE_OBS_DIM)
    if onnx_path is None:
        pytest.skip(f"no {_FORCE_OBS_DIM}-dim force checkpoint found locally")

    assert onnx_path is not None  # narrow for mypy after the skip
    monkeypatch.setenv("MJSWAN_NO_LAUNCH", "1")
    from wbt_realtime_eval.main import setup_builder
    from wbt_realtime_eval.main_vive import _live_per_key_params

    builder = setup_builder(
        model_path=onnx_path,
        motion_dir=_MOTIONS,
        output_dir=tmp_path,
        preset="g1-29dof-holosoma-2pt-force-wbt-live",
        per_key_params=_live_per_key_params("ws://localhost:8770"),
        force_cmd_hand_frame=True,
        show_control_point=True,
    )
    policy = builder.get_projects()[0].scenes[0].policies[0]
    assert policy.config_path is not None
    policy_config = json.loads(Path(policy.config_path).read_text())
    obs_group = next(iter(policy_config["observations"]))
    comps = policy_config["observations"][obs_group]["components"]
    comp_names = [c["name"] for c in comps]

    # The two VR terms are the LIVE variants, not the baked ones; force_cmd_local
    # still 4th; same 11-component layout as the replay force preset.
    assert "WbtVr2PointPosLive" in comp_names and "WbtVr2PointOrnLive" in comp_names
    assert "WbtVr2PointPos" not in comp_names and "WbtVr2PointOrn" not in comp_names
    assert comp_names[3] == "WbtForceCmdLocal", comp_names
    assert comp_names.count("WbtForceCmdLocal") == 1
    assert not any("Ext" in n or "ext" in n for n in comp_names), comp_names

    # Hand-frame is wired onto force_cmd_local ONLY.
    fc = next(c for c in comps if c["name"] == "WbtForceCmdLocal")
    assert fc.get("hand_frame") is True
    assert fc.get("wrist_body_names") == ["left_wrist_yaw_link", "right_wrist_yaw_link"]

    # Live VR terms carry the WS url + HOME fallback.
    for live_name in ("WbtVr2PointPosLive", "WbtVr2PointOrnLive"):
        lc = next(c for c in comps if c["name"] == live_name)
        assert lc.get("vive_ws_url") == "ws://localhost:8770", lc
        assert "home" in lc, lc

    # Force extras present and 588-D, exactly like the replay force preset (the
    # obs-key gate fired on the live preset too).
    extras = policy.extras or {}
    assert len(extras["normalizer_mean"]) == _FORCE_OBS_DIM
    assert len(extras["sliders"]) == 12
    feb = extras["force_ext_bodies"]
    assert [b["body_name"] for b in feb] == ["left_wrist_yaw_link", "right_wrist_yaw_link"]
    for b in feb:
        assert all(k.startswith("wbt_force_ext_local__") for k in b["slider_keys"]), b
    # Dual-arrow viz config is also present on the live preset.
    fvb = policy.commands["motion"].params["force_viz_bodies"]
    assert len(fvb) == 2

    # show_control_point=True wired the orange control-point marker at the
    # vr_2point contact point (separate from the force/rubber-hand point).
    cpb = policy.commands["motion"].params["vr_control_point_bodies"]
    assert [b["body_name"] for b in cpb] == ["left_wrist_yaw_link", "right_wrist_yaw_link"]
    assert [b["offset_b"] for b in cpb] == [[0.18, -0.025, 0.0], [0.18, 0.025, 0.0]]


def test_control_point_extras_match_vr_2point() -> None:
    """The control-point marker config mirrors the VR_2POINT single source of
    truth (bodies + the 0.18 m lever offset), so the orange sphere lands exactly
    on the policy's tracked vr_2point point. Checkpoint-free (pure function), so
    it always runs in CI. Note the offset DIFFERS from the force/rubber-hand
    offset (0.0815, ±0.003) — including a flipped y sign — so this guards against
    accidentally reusing the force offset for the marker."""
    pytest.importorskip("holosoma_inference")
    from wbt_realtime_eval.main import _CONTROL_POINT_RGB, _vr_control_point_extras

    cp = _vr_control_point_extras()
    assert [b["body_name"] for b in cp] == ["left_wrist_yaw_link", "right_wrist_yaw_link"]
    assert [b["offset_b"] for b in cp] == [[0.18, -0.025, 0.0], [0.18, 0.025, 0.0]]
    for b in cp:
        assert b["color"] == _CONTROL_POINT_RGB
        assert b["radius"] > 0
    # Cross-check against the single source of truth used by the bake + bridge.
    from wbt_wrappers_inference.config_values.robot_config import VR_2POINT_BODIES, VR_2POINT_OFFSETS

    assert [b["body_name"] for b in cp] == list(VR_2POINT_BODIES)
    assert [b["offset_b"] for b in cp] == [list(o) for o in VR_2POINT_OFFSETS]


def test_control_point_off_by_default(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Without show_control_point the marker config is absent — byte-identical to
    before. Checkpoint-gated like the other build tests."""
    pytest.importorskip("mjswan")
    pytest.importorskip("pinocchio")
    pytest.importorskip("holosoma_inference")
    onnx_path = _find_2pt_checkpoint()
    if onnx_path is None:
        pytest.skip("no 2pt-WBT ONNX checkpoint found locally")
    assert onnx_path is not None  # narrow for mypy after the skip
    monkeypatch.setenv("MJSWAN_NO_LAUNCH", "1")
    from wbt_realtime_eval.main import setup_builder

    builder = setup_builder(model_path=onnx_path, motion_dir=_MOTIONS, output_dir=tmp_path)
    policy = builder.get_projects()[0].scenes[0].policies[0]
    assert "vr_control_point_bodies" not in policy.commands["motion"].params


# --------------------------------------------------------------------------- #
# Render-scene resolution (yam gripper vs old rubber hand). These don't build
# a bundle — they pin the scene-selection contract so a yam checkpoint renders
# the yam hand while old checkpoints keep the old scene (backward compat).
# --------------------------------------------------------------------------- #


def _resolver():
    pytest.importorskip("mjswan")
    pytest.importorskip("holosoma_inference")
    # NB: `from wbt_realtime_eval import main` would yield the main() FUNCTION,
    # because the package __init__ rebinds the `main` attribute via
    # `from .main import main`. Import the submodule explicitly.
    import importlib

    return importlib.import_module("wbt_realtime_eval.main")


def test_resolve_scene_xml_default_for_old_checkpoint() -> None:
    """No --scene-xml + old (g1_29dof) urdf metadata → the default old scene.
    This is the byte-identical-to-before path for every non-yam checkpoint."""
    m = _resolver()
    # robot_urdf_path absent entirely (older exports) → default.
    assert m._resolve_scene_xml(None, {}) == m._HOLOSOMA_XML
    # robot_urdf_path = old urdf → old scene (== default).
    got = m._resolve_scene_xml(None, {"robot_urdf_path": "/any/where/g1_29dof.urdf"})
    assert got == m._HOLOSOMA_XML
    assert got.name == "scene_g1_29dof_wbt_plane.xml"


def test_resolve_scene_xml_autoderives_yam() -> None:
    """A yam-trained checkpoint (robot_urdf_path stem g1_29dof_yam,
    optionally quote-wrapped as the ONNX metadata stores it) auto-selects the
    yam scene with NO flag."""
    m = _resolver()
    for raw in (
        '"/data/.../robots/g1/g1_29dof_yam.urdf"',  # quote-wrapped (real metadata form)
        "/data/.../robots/g1/g1_29dof_yam.urdf",  # bare
    ):
        got = m._resolve_scene_xml(None, {"robot_urdf_path": raw})
        assert got.name == "scene_g1_29dof_wbt_plane_yam.xml", raw
        assert got.is_file(), got


def test_resolve_scene_xml_unknown_stem_falls_back() -> None:
    """An unmapped urdf stem must NOT crash — it falls back to the default
    scene (so a future hand variant degrades gracefully to the old mesh)."""
    m = _resolver()
    got = m._resolve_scene_xml(None, {"robot_urdf_path": "/x/g1_29dof_some_future_hand.urdf"})
    assert got == m._HOLOSOMA_XML


def test_resolve_scene_xml_mapped_but_missing_scene_hard_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    """A MAPPED hand stem whose scene file is missing must RAISE — never
    silently fall back to the old rubber-hand mesh on a known-hand policy.
    Simulates a fresh checkout where the yam scene isn't present yet."""
    m = _resolver()
    monkeypatch.setitem(m._URDF_STEM_TO_SCENE, "g1_29dof_ghost_hand", "scene_does_not_exist.xml")
    with pytest.raises(FileNotFoundError, match="ghost_hand"):
        m._resolve_scene_xml(None, {"robot_urdf_path": "/x/g1_29dof_ghost_hand.urdf"})


def test_resolve_scene_xml_json_encoded_metadata() -> None:
    """robot_urdf_path is stored json.dumps-encoded (quote-wrapped); the decode
    must unwrap it to the bare stem. Also tolerates a bare (un-encoded) value."""
    m = _resolver()
    quoted = m._resolve_scene_xml(None, {"robot_urdf_path": '"/a/b/g1_29dof_yam.urdf"'})
    bare = m._resolve_scene_xml(None, {"robot_urdf_path": "/a/b/g1_29dof_yam.urdf"})
    assert quoted.name == bare.name == "scene_g1_29dof_wbt_plane_yam.xml"


def test_resolve_scene_xml_explicit_overrides(tmp_path: Path) -> None:
    """An explicit --scene-xml wins over metadata; a repo-relative path resolves
    against the repo root; a missing path raises."""
    m = _resolver()
    # Explicit absolute wins even when metadata says old urdf.
    yam_abs = m._HOLOSOMA_SCENES_DIR / "scene_g1_29dof_wbt_plane_yam.xml"
    got = m._resolve_scene_xml(yam_abs, {"robot_urdf_path": "/x/g1_29dof.urdf"})
    assert got == yam_abs.resolve()
    # Missing explicit path raises (not a silent fallback).
    with pytest.raises(FileNotFoundError):
        m._resolve_scene_xml(tmp_path / "nope.xml", {})


def test_idle_default_root_z_rests_feet_on_floor() -> None:
    """The injected idle_default pelvis height must put the default standing
    pose's feet ON the floor, not floating. _ground_contact_root_z FK-resolves
    it per render scene; a fixed 0.78 floated the (crouched) stance ~2.4 cm.
    We re-do the FK check independently here to pin the contract."""
    m = _resolver()
    mujoco = pytest.importorskip("mujoco")
    import numpy as np

    # A representative crouched stance (knees bent), like the ONNX default angles.
    dof_names = list(_g1_dof_names())
    default_qpos = [0.0] * len(dof_names)
    for i, n in enumerate(dof_names):
        if "knee" in n:
            default_qpos[i] = 0.669
        elif "hip_pitch" in n:
            default_qpos[i] = -0.35
        elif "ankle_pitch" in n:
            default_qpos[i] = -0.32

    for scene in (m._HOLOSOMA_XML, m._HOLOSOMA_SCENES_DIR / "scene_g1_29dof_wbt_plane_yam.xml"):
        if not scene.is_file():
            continue
        root_z = m._ground_contact_root_z(scene, default_qpos, dof_names)
        # Re-compile and verify the lowest robot geom is ~on the floor (|z|<5mm).
        model = mujoco.MjModel.from_xml_path(str(scene))
        data = mujoco.MjData(model)
        floor_gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        qpos = np.zeros(model.nq)
        qpos[2] = root_z
        qpos[3] = 1.0
        for n, a in zip(dof_names, default_qpos):
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)
            if jid >= 0:
                qpos[model.jnt_qposadr[jid]] = a
        data.qpos[:] = qpos
        mujoco.mj_forward(model, data)
        zmin = np.inf
        for gi in range(model.ngeom):
            if gi == floor_gid:
                continue
            sz = model.geom_size[gi]
            gtype = model.geom_type[gi]
            sub = (
                (float(sz[2]) if sz[2] > 0 else float(sz[0]))
                if gtype in (mujoco.mjtGeom.mjGEOM_SPHERE, mujoco.mjtGeom.mjGEOM_CAPSULE, mujoco.mjtGeom.mjGEOM_BOX)
                else 0.0
            )
            zmin = min(zmin, float(data.geom_xpos[gi][2]) - sub)
        assert abs(zmin) < 5e-3, f"{scene.name}: lowest geom z={zmin:.4f} not on floor (root_z={root_z:.4f})"


def _g1_dof_names() -> tuple[str, ...]:
    """The 29 g1 dof names in training order (from g1_29dof config)."""
    pytest.importorskip("holosoma_inference")
    from holosoma_inference.config.config_values.robot import g1_29dof

    return tuple(g1_29dof.dof_names)
