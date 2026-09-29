"""Tests for the URDF→MJCF inertial alignment (browser mass parity).

The load-bearing claim: after alignment the browser's simulated payload equals
the TRAINING URDF's, while nothing kinematic changes. Both halves are pinned
here — a regression that "fixes" mass by perturbing geometry must fail.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import mujoco
import numpy as np
import pytest

from wbt_realtime_eval.urdf_inertia import align_spec_inertials_to_urdf, fused_inertials_from_urdf

_REPO_ROOT = Path(__file__).resolve().parents[5]
_G1_DIR = _REPO_ROOT / "thirdparty/holosoma/src/holosoma/holosoma/data/robots/g1"
_YAM_URDF = _G1_DIR / "g1_29dof_yam.urdf"
_YAM_SCENE = _G1_DIR / "scenes/scene_g1_29dof_wbt_plane_yam.xml"

_needs_assets = pytest.mark.skipif(
    not (_YAM_URDF.is_file() and _YAM_SCENE.is_file()),
    reason="holosoma g1 assets not checked out (run scripts/bootstrap-holosoma.sh)",
)

# The stale MJCF value and the URDF truth, per wrist. Pinned so that if either
# asset changes, this test names exactly what moved instead of silently passing.
_STALE_WRIST_MASS = 0.879804
_URDF_WRIST_MASS = 1.584577


def _two_link_urdf(child_joint_type: str = "fixed") -> str:
    """Minimal parent+child URDF: unit point masses 1 m apart on +x."""
    return textwrap.dedent(f"""
        <robot name="t">
          <link name="base">
            <inertial>
              <origin xyz="0 0 0" rpy="0 0 0"/>
              <mass value="1.0"/>
              <inertia ixx="0.1" iyy="0.1" izz="0.1" ixy="0" ixz="0" iyz="0"/>
            </inertial>
          </link>
          <link name="tip">
            <inertial>
              <origin xyz="0 0 0" rpy="0 0 0"/>
              <mass value="1.0"/>
              <inertia ixx="0.1" iyy="0.1" izz="0.1" ixy="0" ixz="0" iyz="0"/>
            </inertial>
          </link>
          <joint name="j" type="{child_joint_type}">
            <parent link="base"/>
            <child link="tip"/>
            <origin xyz="1 0 0" rpy="0 0 0"/>
          </joint>
        </robot>
    """).strip()


def test_fused_two_link_matches_hand_computation():
    """Analytic check of the parallel-axis sum on a case done by hand.

    Two 1 kg point-ish masses at x=0 and x=1 → M=2, com=0.5. Each contributes
    its own 0.1 plus m·d² = 1·0.25 about the transverse axes; the x-axis gets no
    parallel-axis term (offset is along x).
    """
    fused = fused_inertials_from_urdf(_two_link_urdf(), {"base"})["base"]
    assert fused.mass == pytest.approx(2.0)
    assert fused.com == pytest.approx([0.5, 0.0, 0.0])
    assert fused.inertia[0, 0] == pytest.approx(0.2)  # 0.1 + 0.1, no offset on x
    assert fused.inertia[1, 1] == pytest.approx(0.2 + 2 * 0.25)
    assert fused.inertia[2, 2] == pytest.approx(0.2 + 2 * 0.25)
    assert fused.links == ("base", "tip")


def test_closure_stops_at_a_link_that_is_its_own_body():
    """A child that is ALSO an MJCF body must not be lumped into its parent.

    This is what keeps ``*_foot_contact_point`` from being counted twice.
    """
    both = fused_inertials_from_urdf(_two_link_urdf(), {"base", "tip"})
    assert both["base"].mass == pytest.approx(1.0)
    assert both["base"].links == ("base",)
    assert both["tip"].mass == pytest.approx(1.0)


def test_revolute_child_is_not_lumped():
    """Only fixed/prismatic subtrees fuse; a revolute child is its own body."""
    fused = fused_inertials_from_urdf(_two_link_urdf("revolute"), {"base"})["base"]
    assert fused.mass == pytest.approx(1.0)
    assert fused.links == ("base",)


@_needs_assets
def test_yam_wrist_mass_is_corrected_to_the_urdf_value():
    urdf = _YAM_URDF.read_text()
    spec = mujoco.MjSpec.from_file(str(_YAM_SCENE))

    before = {b.name: float(b.mass) for b in spec.bodies if b.name}
    assert before["left_wrist_yaw_link"] == pytest.approx(_STALE_WRIST_MASS, abs=1e-6), (
        "the yam MJCF's stale wrist mass changed; re-check this fix against the asset"
    )

    corrections = align_spec_inertials_to_urdf(spec, urdf)
    fixed = {c.body: c for c in corrections}

    for side in ("left", "right"):
        body = f"{side}_wrist_yaw_link"
        assert body in fixed, f"{body} should have been corrected"
        assert fixed[body].mass_after == pytest.approx(_URDF_WRIST_MASS, abs=1e-6)
        assert fixed[body].mass_after - fixed[body].mass_before == pytest.approx(0.7048, abs=1e-3)
        # The gripper chain must be what got lumped in.
        assert f"{side}_rubber_hand" in fixed[body].links
        assert f"{side}_ft_sensor_link" in fixed[body].links
        assert f"{side}_gripper_finger1_link" in fixed[body].links

    # Only the wrists are stale — the ±1 g pelvis/foot/torso MJCF rounding must
    # stay below the tolerance, or we'd be churning the whole model.
    assert set(fixed) == {"left_wrist_yaw_link", "right_wrist_yaw_link"}


@_needs_assets
def test_aligned_spec_compiles_and_round_trips_through_xml():
    """The correction must survive spec.compile() AND spec.to_xml().

    mjswan serializes the scene via ``to_xml`` into scene.mjz, so a value that
    only exists on the in-memory spec would never reach the browser.
    """
    urdf = _YAM_URDF.read_text()
    spec = mujoco.MjSpec.from_file(str(_YAM_SCENE))
    align_spec_inertials_to_urdf(spec, urdf)

    model = spec.compile()
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "left_wrist_yaw_link")
    assert float(model.body_mass[bid]) == pytest.approx(_URDF_WRIST_MASS, abs=1e-6)

    # Rebuilding the tensor from MuJoCo's principal-axis form must return the
    # full inertia we asked for (this is where a stale iquat would show up).
    rot = np.zeros(9)
    mujoco.mju_quat2Mat(rot, model.body_iquat[bid])
    rot = rot.reshape(3, 3)
    got = rot @ np.diag(model.body_inertia[bid]) @ rot.T
    want_fi = fused_inertials_from_urdf(urdf, {"left_wrist_yaw_link"})["left_wrist_yaw_link"]
    assert np.allclose(got, want_fi.inertia, atol=1e-10)

    # to_xml keeps 6 significant figures, hence the loose-but-meaningful tol.
    assert 'mass="1.58458"' in spec.to_xml()


@_needs_assets
def test_alignment_does_not_change_kinematics():
    """Mass may change; body frames, geoms and joint layout may not.

    The tracked points (wrist_yaw origins, the 0.242 m TCP, torso head point) are
    pure FK, so a correct mass fix must leave them bit-identical.
    """
    urdf = _YAM_URDF.read_text()
    base = mujoco.MjSpec.from_file(str(_YAM_SCENE)).compile()
    aligned_spec = mujoco.MjSpec.from_file(str(_YAM_SCENE))
    align_spec_inertials_to_urdf(aligned_spec, urdf)
    aligned = aligned_spec.compile()

    assert (base.nq, base.nv, base.nbody, base.ngeom, base.nu) == (
        aligned.nq,
        aligned.nv,
        aligned.nbody,
        aligned.ngeom,
        aligned.nu,
    )
    assert np.array_equal(base.body_pos, aligned.body_pos)
    assert np.array_equal(base.body_quat, aligned.body_quat)
    assert np.array_equal(base.geom_pos, aligned.geom_pos)
    assert np.array_equal(base.jnt_pos, aligned.jnt_pos)
    assert np.array_equal(base.qpos0, aligned.qpos0)

    # FK the tracked points at a non-trivial pose and require exact equality.
    rng = np.random.default_rng(0)
    qpos = np.array(base.qpos0, dtype=np.float64)
    qpos[7:] += rng.uniform(-0.3, 0.3, size=base.nq - 7)
    pts = []
    for model in (base, aligned):
        data = mujoco.MjData(model)
        data.qpos[:] = qpos
        mujoco.mj_forward(model, data)
        row = []
        for body in ("left_wrist_yaw_link", "right_wrist_yaw_link", "torso_link"):
            bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body)
            rot = data.xmat[bid].reshape(3, 3)
            row.append(data.xpos[bid] + rot @ np.array([0.242, 0.0, 0.0]))
        pts.append(np.concatenate(row))
    assert np.array_equal(pts[0], pts[1])


@_needs_assets
def test_inertia_tensors_are_physically_valid():
    """Every corrected tensor must be positive-definite and satisfy the
    triangle inequality — MuJoCo rejects (or mis-simulates) anything else."""
    urdf = _YAM_URDF.read_text()
    spec = mujoco.MjSpec.from_file(str(_YAM_SCENE))
    body_names = {b.name for b in spec.bodies if b.name and b.name != "world"}
    for name, fused in fused_inertials_from_urdf(urdf, body_names).items():
        eig = np.linalg.eigvalsh(fused.inertia)
        assert eig[0] > 0.0, f"{name} inertia not positive-definite: {eig}"
        assert eig[0] + eig[1] >= eig[2] * (1.0 - 1e-9), f"{name} violates triangle inequality: {eig}"


@_needs_assets
def test_lumped_sets_partition_the_urdf_mass():
    """Total lumped mass must equal the URDF's total inertial mass.

    Catches both double-counting (a child lumped into two bodies) and dropped
    links (a subtree nobody claims).
    """
    import xml.etree.ElementTree as ET

    urdf = _YAM_URDF.read_text()
    per_link = {
        link.get("name"): float(link.find("inertial").find("mass").get("value"))
        for link in ET.fromstring(urdf).findall("link")
        if link.find("inertial") is not None
    }
    spec = mujoco.MjSpec.from_file(str(_YAM_SCENE))
    body_names = {b.name for b in spec.bodies if b.name and b.name != "world"}
    fused = fused_inertials_from_urdf(urdf, body_names)

    claimed: dict[str, int] = {}
    for f in fused.values():
        for link in f.links:
            claimed[link] = claimed.get(link, 0) + 1
    assert not [k for k, v in claimed.items() if v > 1], f"double-counted: {claimed}"

    total_fused = sum(f.mass for f in fused.values())
    total_claimed = sum(per_link[k] for k in claimed)
    assert total_fused == pytest.approx(total_claimed, abs=1e-9)
    # Every link with mass is claimed by exactly one body.
    assert set(claimed) == set(per_link), f"unclaimed links: {sorted(set(per_link) - set(claimed))}"


def test_already_correct_scene_is_untouched():
    """A scene whose inertials already match must produce zero corrections."""
    urdf = _two_link_urdf()
    xml = textwrap.dedent("""
        <mujoco model="t">
          <worldbody>
            <body name="base">
              <inertial pos="0.5 0 0" mass="2.0"
                        fullinertia="0.2 0.7 0.7 0 0 0"/>
              <geom type="sphere" size="0.1"/>
            </body>
          </worldbody>
        </mujoco>
    """).strip()
    spec = mujoco.MjSpec.from_string(xml)
    assert align_spec_inertials_to_urdf(spec, urdf) == []


def test_missing_urdf_match_is_a_noop():
    """An unrelated URDF must not silently zero the scene's masses."""
    xml = textwrap.dedent("""
        <mujoco model="t">
          <worldbody>
            <body name="unrelated">
              <inertial pos="0 0 0" mass="5.0" diaginertia="1 1 1"/>
              <geom type="sphere" size="0.1"/>
            </body>
          </worldbody>
        </mujoco>
    """).strip()
    spec = mujoco.MjSpec.from_string(xml)
    assert align_spec_inertials_to_urdf(spec, _two_link_urdf()) == []
    assert float(spec.body("unrelated").mass) == pytest.approx(5.0)
