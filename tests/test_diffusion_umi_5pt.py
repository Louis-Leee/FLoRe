"""``diffusion_umi_5pt`` — FIVE keypoint poses out, a pelvis-frame GEOMETRY in.

What these tests are actually protecting, in the order the bugs would happen:

1. **The layout's two questions.** ``is_wholebody`` used to double as "the state is geometric",
   which was true while ``wholebody*`` were the only modes with a ``state_geom``. This family is
   the counterexample: keypoint POSES as the action, a geometric reduction as the observation.
   If those two collapse back into one branch, a 5-point layout writes five ``wrt_start_rot6d``
   blocks into a 14-wide state vector — a shape error if you are lucky and silent corruption of
   the gripper columns if you are not.
2. **The FOOT provenance.** Nothing on this rig sees a foot, so the two foot roles' labels cannot
   be validated against a recording — only against the derivation they claim
   (``KeypointFK(wb_action_joints)``, PELVIS frame, no root pose). A world-frame foot has the
   same shape, the same units and the same magnitude class, and would train to a plausible loss.
   So the frame is checked by VALUE and the row-0 identity is checked numerically.
3. **The no-regression direction.** ``fivepoint_*`` is additive: new layout modes over the SAME
   ``--hands`` value, eight new arrays on top of a wholebody file. The other two presets must
   read the same bytes they did before.

The synthetic-zarr builder and the wholebody exporter wrapper are IMPORTED from
``test_hl_dataset`` / ``test_diffusion_umi_wholebody`` rather than re-implemented — a drifting
second schema writer is how a loader test starts passing against data no tool produces.
"""
from __future__ import annotations

import os

import numpy as np
import pytest

from flore.s1_policy.data import wb_geom
from flore.s1_policy.data.layout import (
    FOOT_ACTION_KEYS,
    FOOT_ROLES,
    POSE_ONLY_ROLES,
    WB_STATE_BLOCK_DIM,
    _5PT_MODES,
    _FOOT_ZARR_KEYS,
    _WB_MODES,
    get_layout,
    hands_flag,
    tracker_roles_required,
    zarr_keys_for,
)

# `dual_zarr` is re-exported into this module's namespace so pytest resolves it as a FIXTURE
# for the tests below; ruff sees the later `def ...(dual_zarr)` parameters as redefinitions,
# which is the standard pytest/linter friction and not a shadowing bug.
from tests.test_diffusion_umi_wholebody import (  # noqa: F401
    CAMS,
    _augment,
    _patch_zarr,
    _read_array,
    dual_zarr,  # noqa: F401 - a pytest fixture, used by name
)
from tests.test_hl_dataset import _ds

MODE = "fivepoint_geom_xy3_rpy"
DUAL_MODE = "both_hands_head_dual"

foot_tool = pytest.importorskip(
    "scripts.add_foot_targets",
    reason="scripts/ is not importable as a package from here")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
def _add_feet(src: str, out: str, **kw) -> str:
    """Run the REAL foot exporter over ``src``. Skips if the FK is unavailable.

    The FK needs ``yourdfpy`` plus the yam URDF from a submodule; a fresh clone has neither, and
    the whole point of ``load_geom_fk``'s ``(None, reason)`` contract is that its absence is
    reported rather than zero-filled. A test that silently exercised a zero geometry would be
    worse than one that skips.
    """
    fk, why = wb_geom.load_geom_fk()
    if fk is None:
        pytest.skip(f"keypoint FK unavailable: {why}")
    kw.setdefault("jobs", 2)
    kw.setdefault("rows_per_task", 8)
    kw.setdefault("urdf_path", None)
    kw.setdefault("anchor_path", None)
    kw.setdefault("overwrite", True)
    rc = foot_tool.build(src, out, **kw)
    assert rc == 0, f"add_foot_targets returned {rc}"
    return out


@pytest.fixture()
def fivept_zarr(tmp_path, dual_zarr):  # noqa: F811
    """A dual zarr + the wholebody joints + the eight foot arrays: what this preset trains on."""
    wb, _ = _augment(dual_zarr, str(tmp_path / "dual_wb.zarr.zip"))
    return _add_feet(wb, str(tmp_path / "dual_wb_5pt.zarr.zip"))


def _5pt_ds(path, **kw):
    kw.setdefault("mode", MODE)
    kw.setdefault("image_keys", CAMS)
    return _ds(path, **kw)


# ---------------------------------------------------------------------------
# 1. The layout: the two questions, kept separate
# ---------------------------------------------------------------------------
def test_action_is_a_prefix_extension_of_the_dual_mode():
    """Slots [0:41] must be BYTE-IDENTICAL in meaning to both_hands_head_dual's.

    Same structural promise holosoma's VR_5POINT_BODIES makes on the wire: slots [0:3] of a
    5-point vector are the 3-point vector. It is what lets one wire builder serve both and one
    reader compare them — the modes are still not cross-loadable (layout_mode is the pin).
    """
    dual, five = get_layout(DUAL_MODE), get_layout(MODE)
    assert dual.action_dim == 41
    assert five.action_dim == 59      # 41 + 2 * 9
    for role in dual.roles:
        assert five.slots(role) == dual.slots(role), role
    assert five.roles[:3] == dual.roles
    assert five.roles[3:] == FOOT_ROLES
    # ...and the FEET are appended, never interleaved.
    assert min(five.slots(r).pos.start for r in FOOT_ROLES) >= dual.action_dim


def test_feet_are_pose_only_and_reserve_nothing():
    """A foot has no jaw and no F/T, and unlike a HAND there is no plausible future hardware
    that would want those columns reserved — holosoma's VR_5POINT_FEET_BODIES carries not even a
    lever ("a nonzero sole offset would price a point nothing else knows")."""
    lay = get_layout(MODE)
    for role in FOOT_ROLES:
        assert role in POSE_ONLY_ROLES
        slots = lay.slots(role)
        assert slots.gripper is None and slots.force is None, role
        assert slots.block.stop - slots.block.start == 9, role
        assert role not in lay.gripper_roles and role not in lay.force_roles
    # The reserved sets are the two HANDS' and nothing more.
    assert len(lay.force_dims()) == 12 and len(lay.gripper_dims()) == 2


@pytest.mark.parametrize(("mode", "state_dim"), [
    ("fivepoint_geom_xy3_rpy", 6 + 6 + 2),
    ("fivepoint_geom_xy3", 6 + 2),
    ("fivepoint_geom_rpy", 6 + 2),
    ("fivepoint_geom_none", 2),
])
def test_state_dim_follows_the_geometric_composition(mode, state_dim):
    lay = get_layout(mode)
    assert lay.state_dim == state_dim
    assert lay.action_dim == 59, "the ACTION must not move with the state composition"


def test_is_wholebody_and_state_is_geometric_are_two_questions():
    """The split this family exists to force. A `fivepoint_*` layout answers them DIFFERENTLY.

    While they were one boolean, "the state is geometric" implied "the action is a joint
    configuration" — so a keypoint action with a geometric state was unrepresentable, and the
    nearest thing that would have compiled writes five rot6d blocks into a 14-wide vector.
    """
    five = get_layout(MODE)
    assert five.state_is_geometric is True
    assert five.is_wholebody is False
    assert five.joints is None and five.joint_state_slot is None
    assert five.joint_dims() == ()
    # ...and no state slot carries wrt_start_rot6d, for any role.
    for role, entry in five.state_slots().items():
        assert "wrt_start_rot6d" not in entry, role
    # Every whole-body mode still answers BOTH yes, including the zero-block one.
    for mode in _WB_MODES:
        lay = get_layout(mode)
        assert lay.is_wholebody and lay.state_is_geometric, mode
    # Every keypoint mode answers both no.
    for mode in (DUAL_MODE, "left_only", "both_hands_head"):
        lay = get_layout(mode)
        assert not lay.is_wholebody and not lay.state_is_geometric, mode


def test_zero_geom_blocks_is_not_the_same_as_no_geometric_state():
    """`fivepoint_geom_none` selects ZERO blocks; `both_hands_head_dual` has no geometric state
    at all. The two readings of an empty `state_geom` differ by 30 columns, which is why
    `state_geometric` is an explicit field rather than `bool(state_geom)`."""
    none5 = get_layout("fivepoint_geom_none")
    dual = get_layout(DUAL_MODE)
    assert none5.state_geom == () and dual.state_geom == ()
    assert none5.state_dim == 2          # the two jaws
    assert dual.state_dim == 20          # 3 x rot6d + 2 jaws
    assert none5.state_is_geometric and not dual.state_is_geometric


def test_a_layout_cannot_claim_geometric_blocks_without_the_geometric_shape():
    """__post_init__ makes the two fields impossible to leave disagreeing — a builder that set
    one and forgot the other would produce a 30-column-too-wide state that still trains."""
    import dataclasses

    five = get_layout(MODE)
    with pytest.raises(ValueError, match="state_geometric=False"):
        dataclasses.replace(five, state_geometric=False)
    wb = get_layout("wholebody")
    with pytest.raises(ValueError, match="state_geometric=False"):
        dataclasses.replace(wb, state_geometric=False, state_geom=())


def test_pose_dims_covers_all_five_roles():
    """Unlike `wholebody` (whose pose_dims() is EMPTY and where `mask[..., []].all(-1)` reported
    a perfect 0.00 cm position error), every role here has position columns — so a metric must
    report position AND must not be fooled into thinking there are none."""
    lay = get_layout(MODE)
    dims = set(lay.pose_dims())
    assert len(dims) == 5 * 9
    for role in lay.roles:
        slots = lay.slots(role)
        assert set(range(slots.pos.start, slots.pos.stop)) <= dims, role
        assert set(range(slots.rot6d.start, slots.rot6d.stop)) <= dims, role


def test_trackers_are_three_while_roles_are_five():
    """The opposite disagreement from `wholebody`'s (roles 2 < trackers 3), for the same reason:
    nobody may read `layout.roles` to answer "whose glitch corrupts this sample". A chest glitch
    DOES corrupt the feet — they were FK'd from joints the completion produced off those three
    points — while no tracker glitch can be attributed to a foot."""
    for mode in _5PT_MODES:
        assert tracker_roles_required(mode) == ("left", "right", "head"), mode
        assert get_layout(mode).roles == ("left", "right", "head", *FOOT_ROLES), mode
        # ONE conversion feeds all three presets.
        assert hands_flag(mode) == "both_head_dual", mode


def test_rel_xy3_is_translation_invariant_and_rel_xy_is_not():
    """The whole reason `rel_xy3` is a sibling of `rel_xy` rather than an extension of it.

    Shift every body by a constant: `rel_xy3` (three pairwise differences) must not move, while
    `rel_xy`'s `lp_*` (the left hand's ABSOLUTE pelvis-frame xy) must move by exactly the shift.
    That absolute term is the channel the two export placements move by 61-140 mm.
    """
    rng = np.random.default_rng(0)
    bodies = (wb_geom.LEFT_BODY, wb_geom.RIGHT_BODY, wb_geom.CHEST_BODY)
    pos = {b: rng.normal(size=3) for b in bodies}
    rot = {b: np.eye(3) for b in bodies}
    shift = np.array([0.31, -0.17, 0.05])
    moved = {b: p + shift for b, p in pos.items()}

    a = wb_geom.geom_from_bodies(pos, rot, ("rel_xy3",))
    b = wb_geom.geom_from_bodies(moved, rot, ("rel_xy3",))
    assert np.allclose(a, b, atol=1e-12), "rel_xy3 must be translation-invariant"

    a4 = wb_geom.geom_from_bodies(pos, rot, ("rel_xy",))
    b4 = wb_geom.geom_from_bodies(moved, rot, ("rel_xy",))
    assert np.allclose(b4[:2], a4[:2], atol=1e-12), "rel_xy's lr_* is a difference too"
    assert np.allclose(b4[2:] - a4[2:], shift[:2], atol=1e-12), "rel_xy's lp_* IS absolute"


def test_rel_xy3_column_order_and_width_agree_with_the_layout():
    """`wb_geom` owns the ORDER, `layout` owns the WIDTH; a disagreement writes one block's
    values into another's columns with no shape error."""
    names = wb_geom.GEOM_BLOCK_NAMES["rel_xy3"]
    assert names == ("lr_dx", "lr_dy", "lh_dx", "lh_dy", "rh_dx", "rh_dy")
    assert WB_STATE_BLOCK_DIM["rel_xy3"] == len(names)
    assert "rel_xy3" in wb_geom.FK_BLOCKS
    pos = {wb_geom.LEFT_BODY: np.array([1.0, 2.0, 9.0]),
           wb_geom.RIGHT_BODY: np.array([0.5, 0.5, 9.0]),
           wb_geom.CHEST_BODY: np.array([0.1, 0.0, 9.0])}
    rot = {b: np.eye(3) for b in pos}
    got = wb_geom.geom_from_bodies(pos, rot, ("rel_xy3",))
    assert np.allclose(got, [0.5, 1.5, 0.9, 2.0, 0.4, 0.5])


def test_noise_sigma_covers_every_block_it_is_handed():
    """A block with no unit entry would be silently UN-jittered: the wrapper still builds (other
    blocks are nonzero), the loss looks healthy, and the augmentation just does not cover the
    channel someone asked it to."""
    lay = get_layout(MODE)
    sigma = wb_geom.noise_sigma(lay, xy_m=0.01, rpy_rad=0.03)
    slots = lay.state_geom_slots()
    assert np.allclose(sigma[slots["rel_xy3"]], 0.01)
    assert np.allclose(sigma[slots["rpy"]], 0.03)
    # The two jaw columns are deliberately zero: a direct measurement with no posture gap.
    assert np.allclose(sigma[lay.state_dim - 2:], 0.0)
    # A block the LAYOUT knows a width for but `noise_sigma` has no UNIT for is refused rather
    # than skipped. It has to be registered in WB_STATE_BLOCK_DIM to get that far — an entirely
    # unknown name already fails in `state_dim`, which is a fine but different error.
    import dataclasses

    monkey = dict(WB_STATE_BLOCK_DIM)
    WB_STATE_BLOCK_DIM["unitless_block"] = 3
    try:
        with pytest.raises(ValueError, match="no unit for state block"):
            wb_geom.noise_sigma(
                dataclasses.replace(lay, state_geom=("rel_xy3", "unitless_block")),
                xy_m=0.01, rpy_rad=0.03)
    finally:
        WB_STATE_BLOCK_DIM.clear()
        WB_STATE_BLOCK_DIM.update(monkey)


# ---------------------------------------------------------------------------
# 2. The FOOT provenance
# ---------------------------------------------------------------------------
def test_foot_arrays_are_wb_not_hl():
    """In a UMI zarr the `hl_` prefix means "the converter's HL windowing wrote this", and
    nothing about a foot came out of the converter. The honest name costs one branch in
    `_resolve_keys`; the convenient one would be a file that lies about its labels."""
    for role in FOOT_ROLES:
        pose_key, valid_key = FOOT_ACTION_KEYS[role]
        anchor_pos, anchor_rot, gripper = _FOOT_ZARR_KEYS[role]
        for key in (pose_key, valid_key, anchor_pos, anchor_rot):
            assert key.startswith("wb_"), key
            assert not key.startswith("hl_"), key
        assert gripper is None
    # ...and they are reachable for EITHER primary_role, since a mode that declares the foot
    # roles needs them regardless of which hand is robot0.
    for primary in ("left", "right"):
        keys = zarr_keys_for(primary)
        for role in FOOT_ROLES:
            assert keys[role] == _FOOT_ZARR_KEYS[role]


def test_exported_foot_pose_is_the_fk_of_the_stored_joint_target(fivept_zarr):
    """The definitional check, and the one a shape cannot make.

    The foot target must be `KeypointFK(wb_action_joints[t,k])`'s ankle pose — NOT an
    interpolation of the per-frame feet. `wb_action_joints[t,k]` is np.interp'd in TIME onto
    `t0 + k/hz` and the camera grid has dropped frames, so step k is not frame t+k; and
    `FK(interp(q)) != interp(FK(q))` regardless.
    """
    from scipy.spatial.transform import Rotation

    fk, why = wb_geom.load_geom_fk()
    if fk is None:
        pytest.skip(f"keypoint FK unavailable: {why}")
    joints = _read_array(fivept_zarr, "wb_action_joints")
    valid = _read_array(fivept_zarr, "wb_action_valid") > 0.5
    checked = 0
    for role in FOOT_ROLES:
        pose_key, _ = FOOT_ACTION_KEYS[role]
        pose = _read_array(fivept_zarr, pose_key)
        body = foot_tool.FOOT_BODY[role]
        for t, k in zip(*np.nonzero(valid)):
            if checked >= 12:
                break
            pos, rot = fk.bodies(joints[t, k].astype(np.float64))
            assert np.allclose(pose[t, k, :3], pos[body], atol=1e-5), (role, t, k)
            want = Rotation.from_matrix(rot[body]).as_rotvec()
            assert np.allclose(pose[t, k, 3:], want, atol=1e-5), (role, t, k)
            checked += 1
    assert checked, "no valid (t,k) pair to check"


def test_foot_chunk_row0_is_its_own_anchor(fivept_zarr):
    """The one relationship that decides whether every foot chunk is relativized against the
    frame it actually starts at. The hands get it from the converter; the feet get it from the
    exporter's own `wb_action_joints[t,0] == wb_state_joints[t]` property, so it is re-measured
    here (add_foot_targets.py also refuses at write time if it stops holding)."""
    valid = _read_array(fivept_zarr, "wb_action_valid") > 0.5
    rows = np.flatnonzero(valid[:, 0])
    assert rows.size, "no anchor with a valid step 0"
    for role in FOOT_ROLES:
        pose_key, _ = FOOT_ACTION_KEYS[role]
        pos_key, rot_key, _ = _FOOT_ZARR_KEYS[role]
        chunk = _read_array(fivept_zarr, pose_key)[rows, 0]
        anchor = np.concatenate([_read_array(fivept_zarr, pos_key)[rows],
                                 _read_array(fivept_zarr, rot_key)[rows]], axis=1)
        assert np.abs(chunk - anchor).max() < 1e-6, role


def test_foot_validity_is_intersected_with_the_anchors_coverage(fivept_zarr):
    """A step whose ANCHOR frame the completion never covered would be relativized against a
    fabricated frame — and a zero rotvec is the IDENTITY rotation, i.e. a perfectly legal wrong
    answer. So foot validity is `wb_action_valid AND wb_state_valid`, never just the former."""
    act = _read_array(fivept_zarr, "wb_action_valid") > 0.5
    st = _read_array(fivept_zarr, "wb_state_valid").reshape(-1) > 0.5
    want = act & st[:, None]
    for role in FOOT_ROLES:
        _, valid_key = FOOT_ACTION_KEYS[role]
        got = _read_array(fivept_zarr, valid_key) > 0.5
        assert np.array_equal(got, want), role


def test_loader_refuses_a_file_with_no_foot_targets(dual_zarr, tmp_path):  # noqa: F811
    """A wholebody file is a strict SUBSET of a 5pt one, so the missing arrays must be named
    rather than discovered as a KeyError deep in __getitem__."""
    wb, _ = _augment(dual_zarr, str(tmp_path / "dual_wb.zarr.zip"))
    with pytest.raises(ValueError, match="wb_foot_targets"):
        _5pt_ds(wb)


def test_loader_refuses_a_world_frame_foot(fivept_zarr):
    """The frame attr is checked by VALUE, not for presence. A foot pose written in the
    completion's world frame is the same shape, the same units and the same magnitude class as a
    pelvis-frame one, and would train to a plausible-looking loss while answering "where in the
    room" for a channel the wire reads as "where under the body"."""
    _patch_zarr(fivept_zarr, attrs={"wb_foot_frame": "world"})
    with pytest.raises(ValueError, match="wb_foot_frame"):
        _5pt_ds(fivept_zarr)


def test_foot_frame_attr_names_the_derivation(fivept_zarr):
    import zarr

    from flore.s1_policy.data.hl_dataset import _register_image_codecs

    _register_image_codecs()
    with zarr.ZipStore(fivept_zarr, mode="r") as store:
        attrs = dict(zarr.group(store=store).attrs)
    assert attrs["wb_foot_targets"] is True
    assert attrs["wb_foot_frame"] == "pelvis_stance_no_root"
    assert attrs["wb_foot_source"] == "KeypointFK(wb_action_joints)"
    assert attrs["wb_foot_bodies"] == ["left_ankle_roll_link", "right_ankle_roll_link"]


# ---------------------------------------------------------------------------
# 3. The loaded sample
# ---------------------------------------------------------------------------
def test_sample_shapes_and_the_identity_first_row(fivept_zarr):
    ds = _5pt_ds(fivept_zarr)
    try:
        lay = get_layout(MODE)
        sample = ds[0]
        assert tuple(sample["action_chunk"].shape)[1] == 59
        assert tuple(sample["state"].shape) == (14,)
        assert tuple(sample["force_history"].shape) == (1, 12)
        action = sample["action_chunk"].numpy()
        # Row 0 of EVERY keypoint is the identity delta: the chunk is anchor-relative and step 0
        # IS the anchor. For the feet this is the exporter's row0 property propagating through
        # absolute_to_delta_ee; for the hands it is the converter's.
        eye6 = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
        for role in lay.roles:
            slots = lay.slots(role)
            assert np.abs(action[0, slots.pos]).max() < 1e-9, role
            assert np.allclose(action[0, slots.rot6d], eye6, atol=1e-6), role
    finally:
        ds.close()


def test_state_carries_the_geometry_and_no_wrt_start(fivept_zarr):
    """The state must be [rel_xy3 6][rpy 6][L grip][R grip] — and in particular must NOT contain
    an orientation channel per keypoint, which is what the old single `is_wholebody` branch
    would have written."""
    ds = _5pt_ds(fivept_zarr)
    try:
        lay = get_layout(MODE)
        state = ds[0]["state"].numpy()
        slots = lay.state_geom_slots()
        assert slots == {"rel_xy3": slice(0, 6), "rpy": slice(6, 12)}
        assert state.shape == (14,)
        # The jaws sit AFTER the geometric blocks, one per gripper role, and nowhere else.
        sslots = lay.state_slots()
        assert sslots["left"]["gripper"] == 12 and sslots["right"]["gripper"] == 13
        for role in ("head", *FOOT_ROLES):
            assert sslots[role] == {}, role
    finally:
        ds.close()


def test_uncovered_anchor_frames_flag_the_geometry_invalid(fivept_zarr, tmp_path):
    """Zero in the geometric block is "both hands at the pelvis" — plausible and fabricated. A
    frame the completion never reached must be flagged, not merely zeroed."""
    st = _read_array(fivept_zarr, "wb_state_valid")
    st[:] = 0.0
    _patch_zarr(fivept_zarr, arrays={"wb_state_valid": st})
    ds = _5pt_ds(fivept_zarr, drop_glitch_windows=False)
    try:
        sample = ds[0]
        valid = sample["state_valid"].numpy()
        assert not valid[:12].any(), "the geometric blocks must be flagged invalid"
    finally:
        ds.close()


def test_dataset_knobs_pin_the_mode_so_norm_stats_cannot_cross(fivept_zarr):
    """`mode` is in the exact-equality dataset_knobs, which is what refuses a wholebody or dual
    stats file here: 59 action columns and a state that is pelvis-frame metres+radians rather
    than rot6d or joint radians."""
    ds = _5pt_ds(fivept_zarr)
    try:
        knobs = ds.split_provenance(val_last_episodes=1)["dataset_knobs"]
        assert knobs["mode"] == MODE
        assert knobs["image_keys"] == list(CAMS)
        # The completion anchor rides along for the same reason it does for wholebody: the
        # geometry is FK'd from joints whose values only mean anything relative to it.
        assert knobs["wb_cond_mode"] == "home"
    finally:
        ds.close()


# ---------------------------------------------------------------------------
# 4. No regression on the other two presets
# ---------------------------------------------------------------------------
def test_the_other_presets_are_unchanged_by_the_foot_arrays(fivept_zarr):
    """A 5pt file is a wholebody file plus eight arrays, which is a wholebody file plus five
    arrays over a dual one. All three presets must load the SAME file and each read its own
    columns — that is what makes one recording session feed all three."""
    for mode, action_dim, state_dim in ((DUAL_MODE, 41, 20),
                                        ("wholebody", 43, 31),
                                        (MODE, 59, 14)):
        ds = _ds(fivept_zarr, mode=mode, image_keys=CAMS)
        try:
            sample = ds[0]
            assert tuple(sample["action_chunk"].shape)[1] == action_dim, mode
            assert tuple(sample["state"].shape) == (state_dim,), mode
        finally:
            ds.close()


def test_no_other_mode_gained_a_foot_role_or_a_geometric_state():
    """The additive claim, asserted rather than assumed."""
    for mode in ("left_only", "right_only", "both_hands", "both_hands_head", DUAL_MODE):
        lay = get_layout(mode)
        assert not any(r in FOOT_ACTION_KEYS for r in lay.roles), mode
        assert not lay.state_is_geometric, mode
    for mode in _WB_MODES:
        lay = get_layout(mode)
        assert not any(r in FOOT_ACTION_KEYS for r in lay.roles), mode
        assert lay.state_geom and lay.state_geom[0] in WB_STATE_BLOCK_DIM or mode.endswith(
            "none"), mode
        assert "rel_xy3" not in lay.state_geom, f"{mode} must keep its own rel_xy"


def test_launcher_plan_is_resolvable_without_a_gpu():
    """`--print-plan` must resolve mode/dataset/stats/output and exit 0 before touching a GPU;
    the grid queue ASKS the launcher for those four strings rather than re-deriving the naming
    rules, so a broken plan is a broken queue."""
    import subprocess

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for state, horizon, want_mode in (("xy3_rpy", "50", "fivepoint_geom_xy3_rpy"),
                                      ("xy3", "100", "fivepoint_geom_xy3"),
                                      ("none", "200", "fivepoint_geom_none")):
        out = subprocess.run(
            ["bash", "scripts/run_train_diffusion_umi_5pt.sh", "--print-plan"],
            cwd=root, capture_output=True, text=True, check=True,
            env={**os.environ, "WB_STATE": state, "H": horizon,
                 "SESSION": "0902_merged3_h200"})
        mode, ds_path, ns, outdir = out.stdout.strip().split("\t")
        assert mode == want_mode
        assert ds_path.endswith("_5pt.zarr.zip"), ds_path
        assert "norm_stats_hl_5pt_" in ns, ns
        if horizon != "50":
            assert f"_h{horizon}" in ns, ns
        assert "diffusion_umi_5pt_" in outdir, outdir


def test_launcher_refuses_a_horizon_the_session_does_not_store():
    """H > STORED_H is a re-CONVERT, never a re-index: re-indexing a stored 50-step window is
    wrong by up to 0.52 rad on 23% of joint steps (measured)."""
    import subprocess

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out = subprocess.run(
        ["bash", "scripts/run_train_diffusion_umi_5pt.sh", "--print-plan"],
        cwd=root, capture_output=True, text=True,
        env={**os.environ, "WB_STATE": "xy3_rpy", "H": "200",
             "SESSION": "0902_merged3"})
    assert out.returncode != 0
    assert "stores only 50" in out.stderr, out.stderr


def test_launcher_refuses_a_joints_state():
    """There is no `joints` cell here: dropping the 29 raw joint angles from the observation is
    the point of the preset, and the message must say where to go instead."""
    import subprocess

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out = subprocess.run(
        ["bash", "scripts/run_train_diffusion_umi_5pt.sh", "--print-plan"],
        cwd=root, capture_output=True, text=True,
        env={**os.environ, "WB_STATE": "joints"})
    assert out.returncode != 0
    assert "run_train_diffusion_umi_wholebody.sh" in out.stderr, out.stderr


# ---------------------------------------------------------------------------
# 5. The deploy-side wire flags
# ---------------------------------------------------------------------------
def test_five_point_wire_source_flags_are_validated():
    from flore.s1_policy.serving import hl_dense

    readout = hl_dense.KeypointReadout(pelvis_pos=(0.0, 0.0, 0.7567))
    pos = np.zeros(15)
    orn = np.tile([0.0, 0.0, 0.0, 1.0], 5).astype(float)
    for bad in ({"upper_source": "nope"}, {"foot_source": "nope"}):
        with pytest.raises(ValueError, match="must be one of"):
            hl_dense.FivePointWire(readout, pos, orn, **bad)
    wire = hl_dense.FivePointWire(readout, pos, orn,
                                  upper_source="policy", foot_source="hold")
    assert wire.upper_source == "policy" and wire.foot_source == "hold"
    # Both appear on the status line: they are the one thing no number reveals, since `policy`
    # and `completion` agree to 0.84 mm and `hold` looks like a completion standing still.
    line = wire.status(wire.ready_frame)
    assert "upper=policy" in line and "feet=hold" in line


def test_upper_from_decoded_matches_a_hand_relativized_by_hand():
    """The substitution must be the SAME relativization the completion path gets — one frozen
    pelvis, positions translated+rotated, rotations left-composed, scalar-positive quats."""
    from flore.s1_policy.serving import hl_dense
    from flore.utils.rot6d import rotmat_to_rot6d

    pelvis = np.array([0.0, 0.0, 0.7567])
    readout = hl_dense.KeypointReadout(pelvis_pos=pelvis)
    rng = np.random.default_rng(3)
    horizon = 4
    decoded, want_pos = {}, []
    for role in hl_dense._UPPER_ROLES:
        pos = rng.normal(scale=0.2, size=(horizon, 3)) + np.array([0.4, 0.0, 0.9])
        rot = np.tile(np.eye(3), (horizon, 1, 1))
        decoded[role] = {"position": pos, "rot6d": rotmat_to_rot6d(rot)}
        want_pos.append(pos - pelvis)
    got_pos, got_orn = readout.upper_from_decoded(decoded)
    assert got_pos.shape == (horizon, 9) and got_orn.shape == (horizon, 12)
    assert np.allclose(got_pos, np.concatenate(want_pos, axis=1))
    # Identity rotations -> identity quats with a NON-NEGATIVE scalar (the +w branch the
    # vr_5point_orn presets were trained on; -q is a ~20-sigma OOD input).
    assert np.allclose(got_orn.reshape(horizon, 3, 4)[..., 3], 1.0)
    with pytest.raises(ValueError, match="head"):
        readout.upper_from_decoded({"left": decoded["left"], "right": decoded["right"]})
