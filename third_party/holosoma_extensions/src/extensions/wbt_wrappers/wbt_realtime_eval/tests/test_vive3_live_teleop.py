"""Live VIVE 3pt teleop (``main_vive3``) — the sim twin of MODE=3pt-unified-anneal.

Covers the four things that can be wrong without any dimension check noticing,
because the 3pt unified obs is 680-D either way:

* the HOME anchor's pelvis-frame arrays vs the RENDER SCENE's actual TCP poses
  (the frame contract the on-screen target triads implement),
* the checkpoint's actor obs being the 0.18 m-lever layout instead of the 0.242 m
  TCP one,
* a term reorder around ``contact_phase``,
* the ``dof_pos``/``joint_pos`` training-vs-inference naming difference falsely
  tripping the layout guard (it did, on the first run).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[5]
_MOTIONS_LOCAL = Path.home() / "data/sonic_manipulation_test"
_MOTIONS_REPO = _REPO / "src/extensions/wbt_wrappers/wbt_training/motions"
_LOG_ROOT = _REPO / "logs/WBT-Holosoma"
_YAM_SCENE = _REPO / "thirdparty/holosoma/src/holosoma/holosoma/data/robots/g1/scenes/scene_g1_29dof_wbt_plane_yam.xml"
_PRESET = "g1-29dof-holosoma-3pt-hybrid-unified-anneal-wbt-live"
_UNIFIED_OBS_DIM = 680

# The checkpoint's actor obs, canonicalized (dof_* -> joint_*) and in policy order.
# This is what wda3h4ll (exp:g1-wbt-3pt-hybrid-unified-anneal) actually bakes.
_CKPT_TERMS = (
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
    "dof_pos",  # training spelling; the preset says joint_pos
    "dof_vel",
    "actions",
)


def _meta_with_actor_terms(names: tuple[str, ...], **extra_exp: object) -> dict[str, str]:
    """Fake ONNX metadata carrying an actor_obs group with these term names,
    zero-padded exactly the way holosoma's exporter does (so the guard trusts
    their order)."""
    terms = {f"obs{i:02d}_{n}": {"func": f"x:{n}", "params": {}} for i, n in enumerate(names)}
    exp: dict[str, object] = {"observation": {"groups": {"actor_obs": {"terms": terms}}}}
    exp.update(extra_exp)
    return {"experiment_config": json.dumps(exp)}


# ── the frame contract the on-screen target triads implement ─────────────────


def test_home_anchor_3pt_tcp_unprojects_onto_the_render_scene() -> None:
    """``pelvis_pos_w + R(pelvis_quat_w) @ home_pos_pelvis_3pt_tcp`` must land on
    the render scene's ACTUAL tracked points at the default pose, and likewise
    ``q_pelvis (x) home_orn_quat_xyzw_3pt`` on their actual orientations.

    That composition is exactly what ``WbtTrackingCommand.updateVrLiveTriads``
    does with the pelvis-frame pose the live obs classes publish, so this pins the
    frame convention of the commanded-target triads numerically — including the
    two things a browser screenshot cannot tell you apart from bad tracking: a
    swapped axis and the lever-vs-TCP offset.

    It also fixes the visual baseline the operator reads: at the default pose with
    the bridge still at HOME, each coloured target ball sits ON its white
    achieved-TCP sphere. A visible gap there is a real error, not a convention.

    The anchor was computed by the bake's Pinocchio FK; this checks it against
    MuJoCo's FK on the render scene converted from the same URDF, so agreement
    also proves the two FK paths see the same wrist frames.
    """
    mujoco = pytest.importorskip("mujoco")
    np = pytest.importorskip("numpy")
    if not _YAM_SCENE.is_file():
        pytest.skip(f"render scene not checked out: {_YAM_SCENE}")
    from wbt_realtime_eval.vive import load_home_anchor

    anchor = load_home_anchor()
    model = mujoco.MjModel.from_xml_path(str(_YAM_SCENE))
    data = mujoco.MjData(model)
    qpos = np.zeros(model.nq)
    qpos[3] = 1.0  # identity root quat (wxyz)
    qpos[2] = 0.78
    for name, angle in zip(anchor["dof_names"], anchor["default_qpos"], strict=True):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid >= 0:
            qpos[model.jnt_qposadr[jid]] = angle
    data.qpos[:] = qpos
    mujoco.mj_forward(model, data)

    def body_id(name: str) -> int:
        for b in range(model.nbody):
            bn = model.body(b).name
            if bn == name or bn.endswith(f"/{name}"):
                return b
        raise AssertionError(f"body {name!r} not in {_YAM_SCENE.name}")

    def quat_mat(q) -> object:
        out = np.zeros(9)
        mujoco.mju_quat2Mat(out, np.asarray(q, dtype=float))
        return out.reshape(3, 3)

    pelvis = body_id("pelvis")
    p_pelvis = data.xpos[pelvis].copy()
    q_pelvis = data.xquat[pelvis].copy()
    rot_pelvis = quat_mat(q_pelvis)

    want_pos = np.asarray(anchor["home_pos_pelvis_3pt_tcp"], dtype=float)
    want_quat = np.asarray(anchor["home_orn_quat_xyzw_3pt"], dtype=float)  # xyzw
    bodies = anchor["vr_3point_bodies"]
    offsets = anchor["vr_3point_offsets_tcp"]
    assert len(bodies) == len(offsets) == 3
    # The TCP variant must be the 0.242 m fingertip offset on BOTH hands (and the
    # torso->head lever on the third point), not the 0.18 m VR lever.
    assert [round(o[0], 4) for o in offsets[:2]] == [0.242, 0.242]

    for i, (body_name, offset) in enumerate(zip(bodies, offsets, strict=True)):
        b = body_id(body_name)
        point_w = data.xpos[b] + quat_mat(data.xquat[b]) @ np.asarray(offset, dtype=float)
        local = rot_pelvis.T @ (point_w - p_pelvis)
        assert np.allclose(local, want_pos[i], atol=5e-4), (
            f"{body_name}: anchor {want_pos[i]} vs scene FK {local} "
            f"(|d| = {float(np.linalg.norm(local - want_pos[i])) * 1000:.3f} mm)"
        )
        q_local = np.zeros(4)
        q_pelvis_inv = np.zeros(4)
        mujoco.mju_negQuat(q_pelvis_inv, q_pelvis)
        mujoco.mju_mulQuat(q_local, q_pelvis_inv, data.xquat[b])
        got = np.array([q_local[1], q_local[2], q_local[3], q_local[0]])  # wxyz -> xyzw
        # Quaternion double cover: q and -q are the same rotation.
        dot = abs(float(np.dot(got / np.linalg.norm(got), want_quat[i] / np.linalg.norm(want_quat[i]))))
        assert dot > 1.0 - 1e-6, f"{body_name}: orientation off by {np.degrees(2 * np.arccos(min(1.0, dot))):.4f} deg"


# ── HOME fallback plumbing ───────────────────────────────────────────────────


def test_live_3pt_per_key_params_flatten_the_tcp_anchor() -> None:
    """Both live terms get the WS url + a HOME fallback of the right width, taken
    from the SAME anchor arrays the bridge calibrates onto. A zero/short HOME would
    command all three points to the pelvis origin and collapse the arms."""
    from wbt_realtime_eval.main_vive3 import _live_3pt_per_key_params
    from wbt_realtime_eval.vive import load_home_anchor

    anchor = load_home_anchor()
    params = _live_3pt_per_key_params("ws://example:9999")
    assert set(params) == {"vr_3point_pos_tcp_live", "vr_3point_orn_live"}
    pos = params["vr_3point_pos_tcp_live"]
    orn = params["vr_3point_orn_live"]
    assert pos["vive_ws_url"] == orn["vive_ws_url"] == "ws://example:9999"
    assert len(pos["home"]) == 9
    assert len(orn["home"]) == 12
    assert pos["home"] == [float(v) for row in anchor["home_pos_pelvis_3pt_tcp"] for v in row]
    assert orn["home"] == [float(v) for row in anchor["home_orn_quat_xyzw_3pt"] for v in row]
    # Explicitly NOT the 0.18 m-lever anchor: same 9D width, hands 6.7 cm closer,
    # and only the field NAME distinguishes the two arrays in home_anchor.json.
    lever_home = [float(v) for row in anchor["home_pos_pelvis_3pt"] for v in row]
    assert pos["home"] != lever_home
    # The TCP hands reach further out (+x) than the lever ones.
    assert pos["home"][0] > lever_home[0]


# ── the actor-obs layout guard ───────────────────────────────────────────────


def test_actor_obs_guard_accepts_the_checkpoint_layout() -> None:
    """The real wda3h4ll layout passes — including the dof_pos/dof_vel training
    spelling, which must NOT trip the guard (it did on the first run: the preset
    mirrors the INFERENCE ObservationConfig, which says joint_pos/joint_vel)."""
    from wbt_realtime_eval.main_vive3 import _validate_actor_obs_matches_preset

    _validate_actor_obs_matches_preset(_meta_with_actor_terms(_CKPT_TERMS))


def test_actor_obs_guard_rejects_the_lever_position_term() -> None:
    """The 0.18 m-lever checkpoint (iss0qs71 / m3s0pw9i) is also 680-D with the
    same 9D position slot, so ONLY the term name distinguishes it — and the target
    would be 6.7 cm off per hand. Must refuse, and must say why."""
    from wbt_realtime_eval.main_vive3 import _validate_actor_obs_matches_preset

    lever = tuple("vr_3point_pos" if t == "vr_3point_pos_tcp" else t for t in _CKPT_TERMS)
    with pytest.raises(ValueError, match="0.18 m VR LEVER"):
        _validate_actor_obs_matches_preset(_meta_with_actor_terms(lever))


def test_actor_obs_guard_rejects_a_reorder() -> None:
    """Same terms, wrong sequence: the 680-D width still matches, so nothing else
    catches it, and the policy would be fed a scrambled vector. contact_phase
    landing BEFORE force_meas_local is the realistic slip (it is what separates
    this 170D layout from the 168D non-unified one)."""
    from wbt_realtime_eval.main_vive3 import _validate_actor_obs_matches_preset

    swapped = list(_CKPT_TERMS)
    i, j = swapped.index("force_meas_local"), swapped.index("contact_phase")
    swapped[i], swapped[j] = swapped[j], swapped[i]
    with pytest.raises(ValueError, match="ORDER does not match"):
        _validate_actor_obs_matches_preset(_meta_with_actor_terms(tuple(swapped)))


def test_actor_obs_guard_rejects_a_missing_term() -> None:
    from wbt_realtime_eval.main_vive3 import _validate_actor_obs_matches_preset

    with pytest.raises(ValueError, match="same terms"):
        _validate_actor_obs_matches_preset(_meta_with_actor_terms(_CKPT_TERMS[:-1]))


def test_actor_obs_guard_warns_instead_of_failing_without_metadata() -> None:
    """A stripped / older export has no actor_obs group. The 680-D normalizer check
    is still the backstop, so warn rather than block the build."""
    from wbt_realtime_eval.main_vive3 import _validate_actor_obs_matches_preset

    _validate_actor_obs_matches_preset({})
    _validate_actor_obs_matches_preset({"experiment_config": "not json{"})


def test_actor_obs_guard_falls_back_to_a_set_check_on_unpadded_keys() -> None:
    """Bare (non-obsNN_) keys make lexical order ambiguous, so order is NOT
    asserted — a legitimate legacy export must not be falsely rejected. The term
    SET is still enforced."""
    from wbt_realtime_eval.main_vive3 import _validate_actor_obs_matches_preset

    bare = {n: {} for n in reversed(_CKPT_TERMS)}
    meta = {"experiment_config": json.dumps({"observation": {"groups": {"actor_obs": {"terms": bare}}}})}
    _validate_actor_obs_matches_preset(meta)  # order ignored, set matches


# ── contact_phase constants + the target-triad config ────────────────────────


def test_contact_phase_thresholds_reach_the_runtime_from_the_checkpoint() -> None:
    """The soft contact label must self-configure off the CHECKPOINT, so a retrain that
    moves a hysteresis edge moves the viz with it instead of silently keeping the defaults.

    WHERE they land changed at the 2026-08-27 merge and that is the point of this test.
    An earlier version of the live-3pt work shipped a TS class that recomputed phi from the
    F^meas bus, so the constants went on the OBS TERM's per-key config. hybrid_control's
    `WbtContactPhase` — the version that survived — reads the runtime's own
    `wbt_cphase__{l,r}`, which the spring-wall sim recomputes every 5 ms SUBSTEP (with a
    post-reset warmup self-armed off prevTcpW) rather than once per 20 ms control step. So
    the four constants now travel in the HYBRID extras, to that sim. Assert the surviving
    path; a regression to per-term config would leave the sim on its fallbacks.
    """
    from wbt_realtime_eval.main import _hybrid_preset_extras, _parse_hybrid_cfg

    cfg = {
        "stiffness_K": 200.0,
        "servo_enabled": True,
        "k_env_min": 100.0,
        "contact_phase_theta_on": 3.25,
        "contact_phase_theta_off": 2.5,
        "contact_phase_tau_s": 0.08,
        "contact_phase_warmup_steps": 7,
    }
    meta = {"experiment_config": json.dumps({"reward": {"terms": {"f": {"params": {"cfg": cfg}}}}})}
    _sliders, hybrid, _fviz, _fext = _hybrid_preset_extras(_parse_hybrid_cfg(meta))
    assert hybrid["contact_phase_theta_on"] == 3.25
    assert hybrid["contact_phase_theta_off"] == 2.5
    assert hybrid["contact_phase_tau_s"] == 0.08
    assert hybrid["contact_phase_warmup_steps"] == 7
    # No hybrid block (a non-hybrid checkpoint): fall back to the shipped defaults rather
    # than inventing numbers -- and the defaults must be training's own 2.0 / 1.5 N.
    _s2, hybrid2, _f2, _e2 = _hybrid_preset_extras(_parse_hybrid_cfg({}))
    assert hybrid2["contact_phase_theta_on"] == 2.0
    assert hybrid2["contact_phase_theta_off"] == 1.5


def test_target_triad_extras_cover_the_three_points_in_obs_order() -> None:
    """One triad per tracked point, tagged l/r/h in the SAME order the 9D position
    term is laid out ([L, R, head]) — the TS side indexes points positionally
    against that term, so a reorder here would draw the head's triad at the left
    hand's target."""
    from wbt_realtime_eval.main import _vr3_live_target_triad_extras

    cfg = _vr3_live_target_triad_extras()
    assert [p["tag"] for p in cfg["points"]] == ["l", "r", "h"]
    assert cfg["extras_prefix"] == "wbt_vr3_tgt"
    assert cfg["ok_key"] == "wbt_vr3_live__ok"
    assert cfg["axis_len"] > 0 and cfg["ball_radius"] > 0
    # Distinct ball colours, and none of them white — white marks the ACHIEVED TCP,
    # and the whole point is telling commanded from achieved apart.
    colours = [tuple(p["color"]) for p in cfg["points"]]
    assert len(set(colours)) == 3
    assert (1.0, 1.0, 1.0) not in colours


# ── end-to-end builder ──────────────────────────────────────────────────────


def _find_unified_3pt_checkpoint() -> Path | None:
    """First local model_*.onnx that is 680-D AND carries the TCP position term."""
    if not _LOG_ROOT.is_dir():
        return None
    onnxruntime = pytest.importorskip("onnxruntime")
    from wbt_realtime_eval.main_vive3 import _checkpoint_actor_obs_terms

    for onnx_path in sorted(_LOG_ROOT.glob("*/model_*.onnx")):
        try:
            sess = onnxruntime.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        except Exception:
            continue
        obs_input = next((i for i in sess.get_inputs() if i.name == "obs"), None)
        if obs_input is None or obs_input.shape[-1] != _UNIFIED_OBS_DIM:
            continue
        found = _checkpoint_actor_obs_terms(dict(sess.get_modelmeta().custom_metadata_map))
        if found is not None and "vr_3point_pos_tcp" in found[0]:
            return onnx_path
    return None


def test_setup_builder_live_3pt_unified_preset(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Full builder on the live 3pt unified preset: the two VR terms live, the
    hybrid extras inherited from the obs-key gate, contact_phase configured from
    the checkpoint, and the commanded-target triads wired. Skipped without a local
    TCP-unified 680-D checkpoint."""
    pytest.importorskip("mjswan")
    pytest.importorskip("pinocchio")
    pytest.importorskip("holosoma_inference")
    motions = _MOTIONS_LOCAL if _MOTIONS_LOCAL.is_dir() else _MOTIONS_REPO
    if not motions.is_dir():
        pytest.skip("no motions dir available")
    onnx_path = _find_unified_3pt_checkpoint()
    if onnx_path is None:
        pytest.skip(f"no {_UNIFIED_OBS_DIM}-D vr_3point_pos_tcp checkpoint found locally")
    assert onnx_path is not None  # narrow for mypy after the skip

    monkeypatch.setenv("MJSWAN_NO_LAUNCH", "1")
    from wbt_realtime_eval.main import setup_builder
    from wbt_realtime_eval.main_vive3 import _live_3pt_per_key_params

    builder = setup_builder(
        model_path=onnx_path,
        motion_dir=motions,
        output_dir=tmp_path,
        preset=_PRESET,
        per_key_params=_live_3pt_per_key_params("ws://localhost:8770"),
    )
    policy = builder.get_projects()[0].scenes[0].policies[0]

    # The components list is written to a sibling *_policy_config.json.
    cfg = json.loads((tmp_path.parent / f"{tmp_path.name}_policy_config.json").read_text())
    names = [c["name"] for c in next(iter(cfg["observations"].values()))["components"]]
    assert names[:6] == [
        "WbtCommandLowerBody",
        "WbtVr3PointPosTcpLive",
        "WbtVr3PointOrnLive",
        "WbtForceCmdLocal",
        "WbtForceMeasLocal",
        "WbtContactPhase",
    ]
    by_name = {c["name"]: c for c in next(iter(cfg["observations"].values()))["components"]}
    assert len(by_name["WbtVr3PointPosTcpLive"]["home"]) == 9
    assert len(by_name["WbtVr3PointOrnLive"]["home"]) == 12
    # WbtContactPhase takes NO per-term config: it reads the runtime's own
    # wbt_cphase__{l,r}, which the spring-wall sim recomputes every 5 ms substep. The
    # thresholds travel in the HYBRID extras to that sim (asserted below), not here.
    assert set(by_name["WbtContactPhase"]) == {"name"}, by_name["WbtContactPhase"]

    # Hybrid extras arrive via the obs-key gate (force_meas_local present).
    assert len(policy.extras["normalizer_mean"]) == _UNIFIED_OBS_DIM
    assert len(policy.extras["sliders"]) == 12
    assert policy.extras["hybrid"]["tcp_offset_x"] == pytest.approx(0.242)
    assert policy.extras["hybrid"]["contact_phase_theta_on"] == 2.0
    assert policy.extras["hybrid"]["contact_phase_theta_off"] == 1.5

    params = policy.commands["motion"].params
    triads = params["vr_live_target_triads"]
    assert [p["tag"] for p in triads["points"]] == ["l", "r", "h"]
    # The white ACHIEVED-TCP spheres sit at the same 0.242 m offset the commanded
    # triads are compared against — that pairing IS the correspondence check.
    assert [c["offset_b"][0] for c in params["vr_control_point_bodies"]] == [
        pytest.approx(0.242),
        pytest.approx(0.242),
    ]
