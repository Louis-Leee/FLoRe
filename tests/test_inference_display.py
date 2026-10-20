"""Display-frame geometry for the offline S1 inference viewer.

Three families of guarantee live here:

1. **Parity with the collection viz / zarr replay.** The triad offsets, the home poses, the
   wrist-offset derivation and the home-anchor rotation math are all COPIES of UMI
   Collection code (the submodule imports viser, so they cannot simply be imported). These
   tests read the submodule's own source and assert the copies still agree, so a change
   there is caught here instead of silently splitting the two views. They skip when the
   submodule is absent.
2. **Error preservation.** The home-anchor rebase must not change any quantity the viewer
   is used to judge, because the Metrics panel reports those same quantities from the RAW
   values. A rebase that quietly shrank the visible error would be the worst possible bug
   in a model-comparison tool.
3. **Dataset-derived placement.** The cluster must follow whatever extrinsic the dataset
   was built with, since the corrected clean94 (``0.225, 0, -0.05``) and the pre-fix
   clean99 (``0.21, 0, 0``) datasets store different physical points.
"""

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from flore.s1_policy.data.layout import get_layout
from flore.s1_policy.inference.display import (
    DISPLAY_HOME_POS,
    TCP_ALONG_WRIST_X_M,
    TRIAD_DRAW_SIZES,
    TRIAD_OFFSETS_WRIST,
    home_anchor_positions,
    home_anchor_rotmats,
    read_stored_extrinsic,
    resolve_role_geometry,
    tcp_triad_offset_from_stored,
    tracker_triad_local_rotmat,
    triad_offsets_from_tcp,
    wrist_offset_from_stored,
)
from flore.utils.rot6d import rotmat_to_quat_xyzw

_REPO = Path(__file__).resolve().parents[1]
_UMI_VIZ = _REPO / "third_party" / "UMI_collection" / "viz"
_UMI_REPLAY = _REPO / "third_party" / "UMI_collection" / "replay"

# The CURRENT (post-2026-08-06) puck->TCP extrinsic: the measured (-y,-x,-z) relabel in the
# rotation block, and the puck->TCP lever expressed in that same PUCK frame. Applying a
# wrist-frame lever in the puck frame is what made every pre-fix stored eef 33 cm / 180 deg wrong.
_M_RELABEL = np.array([[0.0, -1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])
_CORRECTED_T = np.eye(4)
_CORRECTED_T[:3, :3] = _M_RELABEL
_CORRECTED_T[:3, 3] = (0.0, -0.225, 0.05)
# Two PRE-FIX shapes, both pure translations with an identity rotation: the (0.225,0,-0.05)
# clean94 form (right lever, wrong frame) and the older (0.21,0,0) clean99 form.
_PREFIX_T = np.array([[1, 0, 0, 0.225], [0, 1, 0, 0.0], [0, 0, 1, -0.05], [0, 0, 0, 1]], dtype=float)
_PREFIX_OLD_T = np.array([[1, 0, 0, 0.21], [0, 1, 0, 0.0], [0, 0, 1, 0.0], [0, 0, 0, 1]], dtype=float)


def _eval_const(node: ast.AST, namespace: dict):
    """Evaluate a constant expression over ``namespace`` — no ``eval``, no imports.

    Handles exactly the shapes the submodule's display constants use: numbers, tuples,
    references to earlier constants, and +/- arithmetic between them (the offsets are
    written as ``_TCP_ALONG_WRIST_X_M - 0.225``, so plain ``ast.literal_eval`` cannot
    read them). Anything else raises, which is what keeps this parser honest instead of
    quietly returning a wrong number."""
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Tuple):
        return tuple(_eval_const(element, namespace) for element in node.elts)
    if isinstance(node, ast.Name):
        if node.id not in namespace:
            raise KeyError(node.id)
        return namespace[node.id]
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        value = _eval_const(node.operand, namespace)
        return value if isinstance(node.op, ast.UAdd) else -value
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub)):
        left = _eval_const(node.left, namespace)
        right = _eval_const(node.right, namespace)
        return left + right if isinstance(node.op, ast.Add) else left - right
    raise TypeError(f"unsupported constant expression: {ast.dump(node)}")


def _literal_from_source(path: Path, name: str):
    """Read a module-level constant out of a source file without importing the module.

    The UMI viz modules import viser/yourdfpy/mink at module scope, which this test process
    must not require, so the constants are evaluated from the AST instead. Assignments are
    walked in file order so a constant defined in terms of an earlier one resolves."""
    tree = ast.parse(path.read_text())
    namespace: dict = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        try:
            value = _eval_const(node.value, namespace)
        except (KeyError, TypeError, ValueError):
            continue  # not a constant expression we can (or need to) resolve
        for target in node.targets:
            if isinstance(target, ast.Name):
                namespace[target.id] = value
    if name not in namespace:
        raise AssertionError(f"{path.name} has no resolvable module-level constant {name!r}")
    return namespace[name]


def _random_rotmats(count: int, seed: int) -> np.ndarray:
    """``count`` proper rotation matrices via QR (det == +1)."""
    rng = np.random.default_rng(seed)
    out = np.empty((count, 3, 3))
    for index in range(count):
        q, r = np.linalg.qr(rng.normal(size=(3, 3)))
        q = q @ np.diag(np.sign(np.diag(r)))
        if np.linalg.det(q) < 0:
            q[:, 0] *= -1.0
        out[index] = q
    return out


def _geodesic_deg(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    relative = np.einsum("...ij,...kj->...ik", a, b)
    cosine = np.clip((np.trace(relative, axis1=-2, axis2=-1) - 1.0) / 2.0, -1.0, 1.0)
    return np.degrees(np.arccos(cosine))


# ---------------------------------------------------------------------------
# 1. Parity with UMI Collection
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not _UMI_VIZ.is_dir(), reason="UMI_collection submodule not checked out")
def test_triad_offsets_match_collection_viz():
    """The three triad offsets are byte-equal to collection_viz's display constants.

    These are what make the inference viewer draw the SAME cluster the operator sees while
    collecting; a drift here means the two views disagree about where the TCP is."""
    source = _UMI_VIZ / "collection_viz.py"
    assert _literal_from_source(source, "_TCP_ALONG_WRIST_X_M") == TCP_ALONG_WRIST_X_M
    np.testing.assert_allclose(
        TRIAD_OFFSETS_WRIST["tcp"], (_literal_from_source(source, "_TCP_ALONG_WRIST_X_M"), 0.0, 0.0)
    )
    np.testing.assert_allclose(
        TRIAD_OFFSETS_WRIST["tracker"], _literal_from_source(source, "_TRACKER_TRUE_XYZ_M"), atol=1e-12
    )
    np.testing.assert_allclose(
        TRIAD_OFFSETS_WRIST["wrist_yaw_link"], _literal_from_source(source, "_WRIST_YAW_XYZ_M"), atol=1e-12
    )
    # The set of drawn triads is exactly the three the operator asked for — no more.
    assert tuple(TRIAD_OFFSETS_WRIST) == ("tcp", "tracker", "wrist_yaw_link")


def test_tcp_relative_triad_offsets_put_the_tcp_triad_exactly_on_the_datum():
    """``triad_offsets_from_tcp()["tcp"]`` is EXACTLY zero — the load-bearing property.

    The online planner (``scripts/hl_serving.py``) drives a frame at the wrist and hangs these
    triads under it, while BOTH the yellow rollout path and the green recorded-episode path are
    drawn at the **TCP**. The TCP triad is therefore the visible DATUM for those paths: if this
    offset were even slightly nonzero the paths would appear to start beside their own datum,
    which is exactly the confusing symptom this rebase exists to fix. Zero must be exact (not
    merely close) so no rounding can reintroduce a visible gap."""
    offsets = triad_offsets_from_tcp()
    assert tuple(offsets) == tuple(TRIAD_OFFSETS_WRIST)  # same triads, same draw order
    np.testing.assert_array_equal(offsets["tcp"], np.zeros(3))


def test_tcp_relative_offsets_round_trip_to_the_wrist_frame_exactly():
    """Adding the wrist->TCP lever back recovers :data:`TRIAD_OFFSETS_WRIST` bit-for-bit.

    The planner draws each triad at ``_TCP_OFFSET + triad_offsets_from_tcp()[name]``, so this
    is what guarantees the ONLINE cluster is the same cluster the offline viewer, the
    collection viz and the zarr replay draw — the rebase is a pure change of reference point,
    not a second, drifting set of numbers."""
    lever = np.asarray(TRIAD_OFFSETS_WRIST["tcp"], dtype=np.float64)
    for name, tcp_relative in triad_offsets_from_tcp().items():
        np.testing.assert_array_equal(lever + tcp_relative, TRIAD_OFFSETS_WRIST[name])


def test_every_triad_has_a_draw_size():
    """Draw sizes cover exactly the drawn triads, and the TCP one is the largest.

    A missing entry would raise ``KeyError`` mid-scene-build in the planner (after the viser
    server is already up), and equal sizes would make the three triads visually
    indistinguishable — the reason the collection viz draws the TCP bigger."""
    assert set(TRIAD_DRAW_SIZES) == set(TRIAD_OFFSETS_WRIST)
    for name, (length, radius, origin) in TRIAD_DRAW_SIZES.items():
        assert length > 0 and radius > 0 and origin > 0, name
    assert TRIAD_DRAW_SIZES["tcp"][0] > TRIAD_DRAW_SIZES["tracker"][0]


def _import_gripper_only_view():
    """Import the submodule's ``gripper_only_view``, or skip.

    Unlike ``collection_viz``/``replay_zarr_viz`` this module imports only os/dataclasses/
    numpy plus a numpy-only sibling — no viser, no mink — so it can be imported directly and
    its real values compared, instead of parsing its source. That makes the parity checks
    below genuine equality assertions rather than text matching."""
    import importlib
    import sys

    if not _UMI_VIZ.is_dir():
        pytest.skip("UMI_collection submodule not checked out")
    sys.path.insert(0, str(_UMI_VIZ))
    try:
        return importlib.import_module("gripper_only_view")
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"gripper_only_view not importable: {exc!r}")
    finally:
        if str(_UMI_VIZ) in sys.path:
            sys.path.remove(str(_UMI_VIZ))


def test_display_home_matches_gripper_only_view():
    """Left/right display homes equal ``gripper_only_view.HOME_WRIST_POSE``'s real values."""
    gov = _import_gripper_only_view()
    for role in ("left", "right"):
        np.testing.assert_allclose(DISPLAY_HOME_POS[role], gov.HOME_WRIST_POSE[role][0], atol=1e-12)
        # The submodule's home ROTATION is identity, which is what makes the rebase read +Z up.
        np.testing.assert_allclose(gov.HOME_WRIST_POSE[role][1], [1.0, 0.0, 0.0, 0.0], atol=1e-12)


@pytest.mark.skipif(not _UMI_REPLAY.is_dir(), reason="UMI_collection submodule not checked out")
def test_wrist_offset_matches_zarr_replay_derivation():
    """``wrist_offset_from_stored`` equals the zarr replay's OWN function, value for value.

    The zarr replay is the fidelity checker the operator compares against, so the inference
    viewer must hang its cluster at the identical offset. This CALLS the replay's function
    rather than grepping for its source text or re-deriving its formula locally: a grep
    survives a sign flip of the whole expression (and matches a mere comment), and a local
    re-derivation passes even when the submodule's real formula is wrong — either way the
    test would not detect the drift it exists to detect.

    ``replay_zarr_viz`` imports viser at module scope, so the module is loaded with viser
    (and its own heavyweight siblings) stubbed out; only the pure geometry function is used.
    """
    import sys
    import types

    # replay_zarr_viz imports viser + viser.transforms + viser.uplot (via replay_common) at
    # module scope. Stub the whole family: only its pure geometry function is under test, and
    # requiring a real viser here would make this parity check skip on any headless machine.
    stub_names = ("viser", "viser.transforms", "viser.uplot", "replay_common")
    stubs = {}
    for name in stub_names:
        if name in sys.modules:
            continue
        module = types.ModuleType(name)
        stubs[name] = module
    if "viser" in stubs:
        stubs["viser"].transforms = stubs.get("viser.transforms", types.ModuleType("viser.transforms"))
        stubs["viser"].uplot = stubs.get("viser.uplot", types.ModuleType("viser.uplot"))
    sys.path.insert(0, str(_UMI_REPLAY))
    sys.path.insert(0, str(_UMI_VIZ))
    sys.modules.update(stubs)
    try:
        import importlib

        theirs = importlib.import_module("replay_zarr_viz").wrist_offset_from_stored
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"replay_zarr_viz not importable in isolation: {exc!r}")
    finally:
        for name in stubs:
            sys.modules.pop(name, None)
        for path in (str(_UMI_VIZ), str(_UMI_REPLAY)):
            if path in sys.path:
                sys.path.remove(path)

    rotated = _CORRECTED_T.copy()
    rotated[:3, :3] = _random_rotmats(1, seed=5)[0]
    for extrinsic in (_CORRECTED_T, _PREFIX_T, np.eye(4), rotated):
        np.testing.assert_allclose(
            wrist_offset_from_stored(extrinsic), theirs(extrinsic), atol=1e-15
        )
    # Sanity: the shared derivation really is -(tracker + t), so a sign flip on EITHER side
    # would break the comparison above rather than cancelling out.
    tracker = np.asarray(TRIAD_OFFSETS_WRIST["tracker"], dtype=float)
    np.testing.assert_allclose(
        wrist_offset_from_stored(_CORRECTED_T), -(tracker + _CORRECTED_T[:3, 3]), atol=1e-15
    )


@pytest.mark.skipif(not _UMI_REPLAY.is_dir(), reason="UMI_collection submodule not checked out")
def test_tracker_triad_rotation_matches_zarr_replay_derivation():
    """The tracker-triad local rotation equals the zarr replay's ``puck_triad_local_quat``.

    Both draw the tracker triad as a CHILD of the wrist frame, so they must apply the SAME
    local rotation or the two views disagree about the physical puck's orientation — the exact
    bug this fix addresses (TCP + trajectory were right, only the tracker axes were ~180 deg
    off). This CALLS the replay's function (viser stubbed, same recipe as the wrist-offset
    parity test) rather than re-deriving it, so a change there is caught here. Compared as
    quaternions up to the free sign, over identity / corrected / pre-fix / a random rotation.
    """
    import importlib
    import sys
    import types

    stub_names = ("viser", "viser.transforms", "viser.uplot", "replay_common")
    stubs = {name: types.ModuleType(name) for name in stub_names if name not in sys.modules}
    if "viser" in stubs:
        stubs["viser"].transforms = stubs.get("viser.transforms", types.ModuleType("viser.transforms"))
        stubs["viser"].uplot = stubs.get("viser.uplot", types.ModuleType("viser.uplot"))
    sys.path.insert(0, str(_UMI_REPLAY))
    sys.path.insert(0, str(_UMI_VIZ))
    sys.modules.update(stubs)
    try:
        theirs = importlib.import_module("replay_zarr_viz").puck_triad_local_quat
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"replay_zarr_viz not importable in isolation: {exc!r}")
    finally:
        for name in stubs:
            sys.modules.pop(name, None)
        for path in (str(_UMI_VIZ), str(_UMI_REPLAY)):
            if path in sys.path:
                sys.path.remove(path)

    rotated = _CORRECTED_T.copy()
    rotated[:3, :3] = _random_rotmats(1, seed=8)[0]
    for extrinsic in (np.eye(4), _CORRECTED_T, _PREFIX_T, rotated):
        ours_xyzw = rotmat_to_quat_xyzw(tracker_triad_local_rotmat(extrinsic))
        ours_wxyz = np.array([ours_xyzw[3], ours_xyzw[0], ours_xyzw[1], ours_xyzw[2]])
        theirs_wxyz = np.asarray(theirs(extrinsic), dtype=float)
        assert min(
            np.linalg.norm(ours_wxyz - theirs_wxyz), np.linalg.norm(ours_wxyz + theirs_wxyz)
        ) < 1e-9, extrinsic[:3, :3].tolist()


def test_rebase_matches_gripper_only_view_exactly():
    """The rebase MUST reproduce ``gripper_only_view.home_anchor`` (collection viz / zarr
    replay) to machine precision — that parity is the whole requirement.

    The offline-inference ground-truth trajectory has to be drawn IDENTICALLY to the zarr
    replay the operator trusts: same math, same convention. The submodule maps a pose onto a
    home by ``R = (cur·ref⁻¹)·home`` (RIGHT-multiplied world delta) and ``p = home + (cur −
    ref)`` (pure translation). This checks BOTH halves against the submodule over a batch of
    random poses, for a NON-identity reference (the only case that can distinguish a
    left/right swap or a rotate-vs-translate positions bug)."""
    gov = _import_gripper_only_view()
    home_position = np.asarray(DISPLAY_HOME_POS["left"], dtype=float)
    home_quat = np.array([1.0, 0.0, 0.0, 0.0])

    def as_wxyz(matrix: np.ndarray) -> np.ndarray:
        xyzw = rotmat_to_quat_xyzw(matrix)
        return np.array([xyzw[3], xyzw[0], xyzw[1], xyzw[2]])

    reference = _random_rotmats(1, seed=12)[0]
    rotations = _random_rotmats(5, seed=11)
    ref_pos = np.array([0.4, -0.2, 1.1])
    positions = ref_pos + np.array(
        [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.0, 0.1, 0.0], [0.0, 0.0, 0.1], [-0.05, 0.07, -0.03]]
    )

    ours_pos = home_anchor_positions(positions, positions[0], home_position, reference)
    ours_rot = home_anchor_rotmats(rotations, reference)
    for index in range(len(rotations)):
        theirs_pos, theirs_quat = gov.home_anchor(
            home_position, home_quat, positions[0], as_wxyz(reference), positions[index], as_wxyz(rotations[index])
        )
        np.testing.assert_allclose(ours_pos[index], theirs_pos, atol=1e-12)
        # Compare orientations as rotation matrices (quaternion sign is free).
        ours_quat = rotmat_to_quat_xyzw(ours_rot[index])
        ours_quat_wxyz = np.array([ours_quat[3], ours_quat[0], ours_quat[1], ours_quat[2]])
        assert min(
            np.linalg.norm(ours_quat_wxyz - theirs_quat), np.linalg.norm(ours_quat_wxyz + theirs_quat)
        ) < 1e-9, index


def test_home_anchor_preserves_world_directions():
    """Positions are TRANSLATED, not rotated — a +z move stays a +z move.

    This is the property whose absence was the coordinate-system bug: an earlier "rigid"
    rebase rotated positions by ``R_ref^T``, and because the recorded puck's reference points
    +z at the floor, that tilted the whole path (and every world-vertical motion) into the
    puck frame, so the scene read in the wrong coordinate system. The correct rebase keeps
    world directions, so the grid still means "the floor" like the collection viz."""
    reference_rot = _random_rotmats(1, seed=73)[0]  # a puck-like, decidedly non-identity ref
    home = np.asarray(DISPLAY_HOME_POS["left"], dtype=float)
    positions = np.array([[0.4, -0.2, 1.1], [0.4, -0.2, 1.3], [0.5, -0.2, 1.3]])  # +z, then +x
    moved = home_anchor_positions(positions, positions[0], home, reference_rot)
    # Every step VECTOR is identical to the raw step vector (translation preserves differences).
    np.testing.assert_allclose(np.diff(moved, axis=0), np.diff(positions, axis=0), atol=1e-12)
    # Concretely: the first step is a pure +z world move and stays one after the rebase.
    np.testing.assert_allclose(moved[1] - moved[0], [0.0, 0.0, 0.2], atol=1e-12)


def test_triad_style_covers_every_drawn_triad():
    """The viewer's per-triad style table must key-match the offsets it draws.

    ``_ensure_cluster_nodes`` indexes ``_TRIAD_STYLE[name]`` for every name in
    ``TRIAD_OFFSETS_WRIST``, so adding an offset without a style is a KeyError at scene-build
    time — inside a viser render loop, where it surfaces as a dead viewer rather than a
    clear error."""
    from flore.s1_policy.inference.viewer import _TRIAD_STYLE

    assert set(_TRIAD_STYLE) == set(TRIAD_OFFSETS_WRIST)
    for name, style in _TRIAD_STYLE.items():
        assert set(style) == {"axes_length", "axes_radius", "origin_radius"}, name
        assert all(value > 0.0 for value in style.values()), name


# ---------------------------------------------------------------------------
# 2. Error preservation — the load-bearing property of a comparison tool
# ---------------------------------------------------------------------------
def test_home_anchor_pins_the_rotation_multiplication_side():
    """The reference-inverse must be on the RIGHT (world delta ``R_cur @ R_ref^T``).

    This is the one assertion that catches a swap, and it deliberately does NOT need the
    submodule — CI checks out with ``submodules: false``, so the parity test skips there and
    cannot be the only guard. The LEFT-multiplied variant ``R_ref^T @ R_cur`` ALSO lands
    identity at the reference and ALSO preserves the geodesic, so neither property can
    distinguish it; but it is the "rotate positions by R_ref^T" convention that tilted the
    whole scene into the puck's floor-ward frame (the coordinate-system bug), so the side is
    load-bearing."""
    rotation = _random_rotmats(1, seed=71)[0]
    reference = _random_rotmats(1, seed=72)[0]
    right = rotation @ reference.T         # what the implementation must compute (world delta)
    left = reference.T @ rotation          # the plausible-but-wrong variant (tilts positions)
    # The two genuinely differ for this pair, so the assertion below has real content.
    assert np.abs(left - right).max() > 1e-3
    np.testing.assert_allclose(home_anchor_rotmats(rotation[None], reference)[0], right, atol=1e-12)


def test_home_anchor_preserves_both_error_metrics_the_panel_reports():
    """The rebase must preserve the two errors the Metrics panel reports: position distance
    and rotation geodesic.

    The collection-viz convention does not preserve the path-vs-gripper-axis angle (and does
    not claim to — neither does the zarr replay), but it MUST leave the pred-vs-target
    position distance and rotation geodesic exactly as in the raw frame, or the 3D scene would
    disagree with the numbers. Position: translation is rigid on differences. Rotation: a
    shared-reference rebase conjugates the relative rotation, preserving its trace/angle.
    Verified with a NON-identity reference (the map is trivial when it is identity)."""
    reference_rot = _random_rotmats(1, seed=73)[0]
    home = np.asarray(DISPLAY_HOME_POS["left"], dtype=float)
    rng = np.random.default_rng(74)
    target_pos = rng.normal(0.0, 0.1, (5, 3)) + np.array([0.4, -0.2, 1.1])
    pred_pos = target_pos + rng.normal(0.0, 0.02, (5, 3))
    target_rot = _random_rotmats(5, seed=75)
    pred_rot = _random_rotmats(5, seed=76)

    # Position error is unchanged (translation preserves differences).
    moved_t = home_anchor_positions(target_pos, target_pos[0], home, reference_rot)
    moved_p = home_anchor_positions(pred_pos, target_pos[0], home, reference_rot)
    np.testing.assert_allclose(
        np.linalg.norm(moved_p - moved_t, axis=-1),
        np.linalg.norm(pred_pos - target_pos, axis=-1),
        atol=1e-12,
    )
    # Rotation geodesic is unchanged (conjugation preserves the trace).
    reb_t = home_anchor_rotmats(target_rot, reference_rot)
    reb_p = home_anchor_rotmats(pred_rot, reference_rot)
    np.testing.assert_allclose(_geodesic_deg(reb_p, reb_t), _geodesic_deg(pred_rot, target_rot), atol=1e-9)

    # And the whole POSITION map really is rigid on differences: every pairwise distance is unchanged.
    moved = moved_t
    positions = target_pos
    for i in range(len(positions)):
        for j in range(len(positions)):
            np.testing.assert_allclose(
                np.linalg.norm(moved[i] - moved[j]),
                np.linalg.norm(positions[i] - positions[j]),
                atol=1e-12,
            )


def test_home_anchor_positions_without_a_reference_rotation_is_translation_only():
    """The rotation argument is optional but its ABSENCE must be a pure translation.

    Guards the degraded call shape: a caller that rebases rotations while omitting the
    reference here would silently produce the non-rigid map the test above forbids."""
    positions = np.array([[0.0, 0.0, 0.0], [0.1, 0.2, 0.3]])
    home = np.array([1.0, 2.0, 3.0])
    np.testing.assert_allclose(
        home_anchor_positions(positions, positions[0], home),
        positions - positions[0] + home,
        atol=1e-15,
    )


def test_home_anchor_is_identity_at_the_reference():
    """At the reference pose the rebase yields exactly the home pose, identity-rotated.

    This is what makes the scene read +Z up: the anchor frame renders axis-aligned instead
    of the puck's physically-correct-but-unreadable +z-into-the-floor."""
    reference_pos = np.array([1.5, -2.0, 0.25])
    reference_rot = _random_rotmats(1, seed=3)[0]
    home = np.asarray(DISPLAY_HOME_POS["left"], dtype=float)
    np.testing.assert_allclose(
        home_anchor_positions(reference_pos[None], reference_pos, home)[0], home, atol=1e-15
    )
    np.testing.assert_allclose(
        home_anchor_rotmats(reference_rot[None], reference_rot)[0], np.eye(3), atol=1e-12
    )


def test_home_anchor_preserves_position_and_rotation_errors():
    """A shared reference makes the rebase error-preserving for pred-vs-target.

    Position: a common translation cannot change a difference. Rotation: the reference-inverse
    is on the RIGHT, so ``(R_p R_ref^T)(R_t R_ref^T)^T == R_p R_ref^T R_ref R_t^T == R_p R_t^T``
    — the relative rotation is preserved EXACTLY (not merely up to angle) when both share the
    reference. If this ever failed, the scene would show a different error than the Metrics
    panel reports."""
    predicted_pos = _random_rotmats(1, seed=21)[0] * 0.4 + np.array([0.3, 0.2, 0.9])
    target_pos = _random_rotmats(1, seed=22)[0] * 0.4 + np.array([0.31, 0.18, 0.88])
    reference_pos = target_pos[0]
    reference_rot = _random_rotmats(1, seed=23)[0]
    home = np.asarray(DISPLAY_HOME_POS["left"], dtype=float)

    moved_pred = home_anchor_positions(predicted_pos, reference_pos, home)
    moved_target = home_anchor_positions(target_pos, reference_pos, home)
    np.testing.assert_allclose(moved_pred - moved_target, predicted_pos - target_pos, atol=1e-12)
    np.testing.assert_allclose(
        np.linalg.norm(moved_pred - moved_target, axis=-1),
        np.linalg.norm(predicted_pos - target_pos, axis=-1),
        atol=1e-12,
    )

    predicted_rot = _random_rotmats(8, seed=24)
    target_rot = _random_rotmats(8, seed=25)
    np.testing.assert_allclose(
        _geodesic_deg(home_anchor_rotmats(predicted_rot, reference_rot),
                      home_anchor_rotmats(target_rot, reference_rot)),
        _geodesic_deg(predicted_rot, target_rot),
        atol=1e-9,
    )


def test_home_anchor_keeps_the_episode_tail_registered_and_preserves_shape():
    """The green tail stays joined to the target line, and its own shape is unchanged.

    The tail's first row equals the target chunk's step 0 in the raw frame; rebasing all
    three against that same reference must preserve the coincidence, and must not deform
    the tail (it is a rigid translation)."""
    target_pos = np.array([[0.5, 0.5, 1.0], [0.52, 0.5, 1.0], [0.54, 0.51, 1.01]])
    tail = np.array([[0.5, 0.5, 1.0], [0.6, 0.55, 1.05], [0.7, 0.6, 1.10], [0.9, 0.7, 1.2]])
    reference_pos = target_pos[0]
    home = np.asarray(DISPLAY_HOME_POS["left"], dtype=float)
    moved_target = home_anchor_positions(target_pos, reference_pos, home)
    moved_tail = home_anchor_positions(tail, reference_pos, home)
    np.testing.assert_allclose(moved_tail[0], moved_target[0], atol=1e-15)
    np.testing.assert_allclose(np.diff(moved_tail, axis=0), np.diff(tail, axis=0), atol=1e-15)
    # And the rebase actually MOVED the scene (otherwise this test would pass vacuously).
    assert np.linalg.norm(moved_tail[0] - tail[0]) > 0.1


def test_home_anchor_rotation_output_stays_a_proper_rotation():
    """Rebased rotations remain orthonormal with det +1, so viser gets valid quaternions."""
    rebased = home_anchor_rotmats(_random_rotmats(12, seed=31), _random_rotmats(1, seed=32)[0])
    for matrix in rebased:
        np.testing.assert_allclose(matrix @ matrix.T, np.eye(3), atol=1e-12)
        assert float(np.linalg.det(matrix)) == pytest.approx(1.0, abs=1e-12)


def test_home_anchor_rejects_malformed_shapes():
    """Bad shapes raise instead of broadcasting into a silently wrong scene."""
    with pytest.raises(ValueError, match="3-vectors"):
        home_anchor_positions(np.zeros((4, 3)), np.zeros(2), np.zeros(2))
    with pytest.raises(ValueError, match=r"trailing shape"):
        home_anchor_rotmats(np.zeros((4, 2, 2)), np.eye(3))
    with pytest.raises(ValueError, match=r"ref_rotmat must be"):
        home_anchor_rotmats(np.zeros((4, 3, 3)), np.eye(4))
    with pytest.raises(ValueError, match=r"home_rotmat must be"):
        home_anchor_rotmats(np.zeros((4, 3, 3)), np.eye(3), np.eye(4))


# ---------------------------------------------------------------------------
# 3. Dataset-derived cluster placement
# ---------------------------------------------------------------------------
def test_corrected_extrinsic_puts_the_tcp_triad_on_the_trajectory():
    """With the corrected puck->TCP extrinsic the stored point IS the fingertip TCP.

    This is the whole point of the operator's request that both trajectories be drawn from
    the TCP axis origin: for a clean94-style dataset the TCP triad must land exactly on the
    drawn line, with zero residual."""
    np.testing.assert_allclose(tcp_triad_offset_from_stored(_CORRECTED_T), np.zeros(3), atol=1e-15)


@pytest.mark.parametrize(
    # Measured residuals: 102.8 mm for the (0.225,0,-0.05) clean94 form, 61.7 mm for the older
    # (0.21,0,0) one. Both are far above TCP_RESIDUAL_TOL_MM, which is the property that matters.
    ("extrinsic", "least_mm"), [(_PREFIX_T, 100.0), (_PREFIX_OLD_T, 60.0)])
def test_prefix_extrinsic_reports_an_honest_nonzero_residual(extrinsic, least_mm):
    """A pre-fix dataset must NOT be silently drawn as if it stored the TCP.

    Both pre-fix shapes have an IDENTITY rotation, so their stored point is the raw puck rather
    than the fingertip TCP. The viewer must show that gap rather than hiding it, so an old
    dataset cannot be mistaken for a corrected one."""
    residual = tcp_triad_offset_from_stored(extrinsic)
    assert float(np.linalg.norm(residual)) * 1000.0 > least_mm, residual
    # And it is reported, not just computed.
    _, notes = resolve_role_geometry(
        get_layout("left_only"), [{"T_tracker_tcp": np.asarray(extrinsic).tolist()}])
    assert any("is NOT the fingertip TCP" in note for note in notes)
    assert any("PRE-2026-08-06" in note for note in notes)


@pytest.mark.parametrize(
    ("attrs", "expected_translation", "note_match"),
    [
        ({"T_tracker_tcp": _CORRECTED_T.tolist()}, [0.0, -0.225, 0.05], None),
        ({}, [0.0, 0.0, 0.0], "no T_tracker_tcp"),
        ({"T_tracker_tcp": "nonsense"}, [0.0, 0.0, 0.0], "not numeric"),
        ({"T_tracker_tcp": [[1, 0], [0, 1]]}, [0.0, 0.0, 0.0], "finite 4x4"),
        ({"T_tracker_tcp": [[1, 0, 0, float("nan")], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]},
         [0.0, 0.0, 0.0], "finite 4x4"),
    ],
)
def test_read_stored_extrinsic_degrades_loudly(attrs, expected_translation, note_match):
    """A missing or garbled extrinsic falls back to identity but ALWAYS explains itself.

    Silently substituting identity would mis-place the entire cluster by centimetres with no
    visible symptom — exactly the class of bug this viewer exists to catch."""
    matrix, note = read_stored_extrinsic(attrs)
    np.testing.assert_allclose(matrix[:3, 3], expected_translation, atol=1e-12)
    if note_match is None:
        assert note is None
    else:
        assert note is not None and note_match in note


def test_tcp_residual_tolerance_survives_an_f32_round_trip():
    """The corrected extrinsic must read as "on the TCP" even after an f32 round-trip.

    A stored extrinsic could pass through float32 somewhere upstream; that perturbs the
    corrected `(0.225, 0, -0.05)` by ~6e-6 mm. The single shared tolerance
    ``TCP_RESIDUAL_TOL_MM`` (used by BOTH the provenance note and the GUI datum badge) must
    sit above that and below any real miscalibration, so the note and the badge can never
    contradict each other — the bug a pair of hand-picked thresholds (1e-6 vs 0.05) caused."""
    from flore.s1_policy.inference.display import TCP_RESIDUAL_TOL_MM

    perturbed = _CORRECTED_T.astype(np.float32).astype(np.float64)
    residual_mm = float(np.linalg.norm(tcp_triad_offset_from_stored(perturbed))) * 1000.0
    assert 0.0 < residual_mm < TCP_RESIDUAL_TOL_MM, residual_mm
    # And a genuinely stale extrinsic (clean99) is well above the tolerance.
    prefix_mm = float(np.linalg.norm(tcp_triad_offset_from_stored(_PREFIX_T))) * 1000.0
    assert prefix_mm > TCP_RESIDUAL_TOL_MM * 100
    # The note and the tolerance agree: perturbed reads TCP, prefix does not.
    _, note_ok = resolve_role_geometry(get_layout("left_only"), [{"T_tracker_tcp": perturbed.tolist()}])
    assert any("IS the fingertip TCP" in note for note in note_ok)
    _, note_bad = resolve_role_geometry(get_layout("left_only"), [{"T_tracker_tcp": _PREFIX_T.tolist()}])
    assert any("is NOT the fingertip TCP" in note for note in note_bad)


def test_identity_rotation_extrinsic_is_flagged_as_pre_fix_but_still_used():
    """An IDENTITY extrinsic rotation is the ANOMALY now, and must be reported as pre-fix.

    This is the inverse of what the old build asserted. The corrected extrinsic CARRIES the
    (-y,-x,-z) relabel, so a pure-translation one means a pre-2026-08-06 conversion whose stored
    trajectory is ~33 cm / 180 deg wrong. It is still passed through (the viewer draws what is
    stored, honestly) but never silently."""
    matrix, note = read_stored_extrinsic({"T_tracker_tcp": _PREFIX_T.tolist()})
    np.testing.assert_allclose(matrix, _PREFIX_T, atol=1e-15)
    assert note is not None and "IDENTITY" in note and "PRE-2026-08-06" in note
    # The CURRENT convention is the quiet path — no note at all.
    _, note_ok = read_stored_extrinsic({"T_tracker_tcp": _CORRECTED_T.tolist()})
    assert note_ok is None


def test_wrist_pose_from_stored_is_the_full_4x4_inverse():
    """The wrist must be recovered by inverting the stored composition, rotation included.

    The old ``-(tracker_offset + t)`` added a PUCK-frame translation to a WRIST-frame vector and
    dropped the rotation — valid only if the two frames' axes coincided, which they never do
    (measured 102.8 mm / 333.5 mm off). This pins the inverse property directly: mapping the
    recovered wrist origin back through the stored composition must land on the stored point,
    and the recovered rotation must be a proper rotation that is IDENTITY exactly when the
    stored frame IS the wrist frame."""
    from flore.s1_policy.inference.display import PUCK_TO_WRIST_RELABEL, wrist_pose_from_stored

    tracker = np.asarray(TRIAD_OFFSETS_WRIST["tracker"], dtype=float)
    for label, extrinsic, want_identity in (
        ("current", _CORRECTED_T, True),
        ("pre-fix", _PREFIX_T, False),
        ("pre-fix old", _PREFIX_OLD_T, False),
    ):
        offset, rot = wrist_pose_from_stored(extrinsic)
        # A proper rotation, always.
        np.testing.assert_allclose(rot @ rot.T, np.eye(3), atol=1e-12, err_msg=label)
        assert np.isclose(np.linalg.det(rot), 1.0), label
        # INVERSE PROPERTY: stored_T_wrist must undo puck->stored applied to the wrist origin.
        # The wrist origin sits at ``tracker`` in the wrist frame, i.e. at 0 in the puck frame.
        r_ext, t_ext = np.asarray(extrinsic)[:3, :3], np.asarray(extrinsic)[:3, 3]
        stored_from_wrist = r_ext.T @ (-(PUCK_TO_WRIST_RELABEL @ tracker) - t_ext)
        np.testing.assert_allclose(offset, stored_from_wrist, atol=1e-12, err_msg=label)
        # On the current convention the stored frame IS the wrist frame: identity rotation and
        # the wrist exactly 0.242 m back along -x. On a pre-fix one it is 180 deg away.
        if want_identity:
            np.testing.assert_allclose(rot, np.eye(3), atol=1e-12, err_msg=label)
            np.testing.assert_allclose(offset, [-TCP_ALONG_WRIST_X_M, 0.0, 0.0], atol=1e-12)
        else:
            assert not np.allclose(rot, np.eye(3), atol=1e-6), label
            angle = np.degrees(np.arccos(np.clip((np.trace(rot) - 1) / 2, -1, 1)))
            assert np.isclose(angle, 180.0, atol=1e-9), (label, angle)
        # The translation-only helper must agree with the full one (one source of truth).
        np.testing.assert_allclose(wrist_offset_from_stored(extrinsic), offset, atol=0)


def test_resolve_role_geometry_covers_every_layout_role():
    """Every role in a bimanual+head layout gets geometry, with only hands as gripper roles."""
    layout = get_layout("both_hands_head")
    geometry, notes = resolve_role_geometry(layout, [{"T_tracker_tcp": _CORRECTED_T.tolist()}])
    assert set(geometry) == set(layout.roles) == {"left", "right", "head"}
    assert geometry["left"].is_gripper_role and geometry["right"].is_gripper_role
    assert not geometry["head"].is_gripper_role
    # Distinct homes, so the three clusters do not stack on top of each other.
    homes = [tuple(geometry[role].home_pos) for role in layout.roles]
    assert len(set(homes)) == 3
    assert any("IS the fingertip TCP" in note for note in notes)


def test_resolve_role_geometry_reports_disagreeing_source_files():
    """Two source files with different extrinsics is a rig error the operator must see."""
    layout = get_layout("left_only")
    geometry, notes = resolve_role_geometry(
        layout,
        [{"T_tracker_tcp": _CORRECTED_T.tolist()}, {"T_tracker_tcp": _PREFIX_T.tolist()}],
    )
    assert any("DIFFERENT T_tracker_tcp" in note for note in notes)
    # File 0 wins, deterministically.
    np.testing.assert_allclose(
        geometry["left"].wrist_offset, wrist_offset_from_stored(_CORRECTED_T), atol=1e-15
    )


def test_resolve_role_geometry_without_attrs_still_returns_usable_geometry():
    """No attrs at all (a fixture dataset) still yields drawable geometry plus a note."""
    layout = get_layout("left_only")
    geometry, notes = resolve_role_geometry(layout, None)
    assert set(geometry) == {"left"}
    assert np.isfinite(geometry["left"].wrist_offset).all()
    assert any("no T_tracker_tcp" in note for note in notes)


def test_wrist_offset_rejects_a_malformed_extrinsic():
    with pytest.raises(ValueError, match="must be 4x4"):
        wrist_offset_from_stored(np.eye(3))


def test_tracker_triad_local_rotmat_is_the_extrinsic_rotation_transpose():
    """The tracker triad's wrist-relative rotation is ``R_ext^T`` — a proper rotation.

    The triad is a child of the wrist frame, so this is what renders it at the physical puck's
    orientation instead of the inherited TCP axes. On the corrected extrinsic that rotation is
    the ``(-y, -x, -z)`` relabel (its own transpose, a 180 deg involution); on an
    identity-rotation extrinsic (pre-2026-08-06 or head puck) it is identity, so such a triad
    is honestly left at the stored orientation rather than being relabeled."""
    np.testing.assert_array_equal(tracker_triad_local_rotmat(np.eye(4)), np.eye(3))
    rot = tracker_triad_local_rotmat(_CORRECTED_T)
    np.testing.assert_allclose(rot, _M_RELABEL.T, atol=1e-15)
    np.testing.assert_allclose(rot, _M_RELABEL, atol=1e-15)  # symmetric: R_ext == R_ext^T here
    np.testing.assert_allclose(rot @ rot.T, np.eye(3), atol=1e-12)
    assert np.isclose(np.linalg.det(rot), 1.0)
    # A pre-fix (identity-rotation) extrinsic leaves the triad unrotated.
    np.testing.assert_array_equal(tracker_triad_local_rotmat(_PREFIX_T), np.eye(3))
    with pytest.raises(ValueError, match="must be 4x4"):
        tracker_triad_local_rotmat(np.eye(3))


def test_resolve_role_geometry_sets_the_tracker_triad_rotation():
    """Every role's geometry carries the dataset-derived tracker rotation (identity when absent).

    The viewer reads ``geometry.tracker_local_rotmat`` when it builds the tracker triad, so a
    missing/wrong value here is the ~180 deg puck-axis error on screen."""
    geometry, _ = resolve_role_geometry(
        get_layout("left_only"), [{"T_tracker_tcp": _CORRECTED_T.tolist()}])
    np.testing.assert_allclose(
        geometry["left"].tracker_local_rotmat, tracker_triad_local_rotmat(_CORRECTED_T), atol=1e-15)
    # No extrinsic at all: still drawable, tracker triad simply unrotated (identity default).
    geometry_none, _ = resolve_role_geometry(get_layout("left_only"), None)
    np.testing.assert_array_equal(geometry_none["left"].tracker_local_rotmat, np.eye(3))


# ---------------------------------------------------------------------------
# 4. The viewer's own frame-selection logic
# ---------------------------------------------------------------------------
# ``_build_display_poses`` holds the reference choice and the degenerate-reference
# fallback — the two places a wrong frame could reach the screen. It lives in viewer.py,
# which imports viser lazily (only inside ``run_viewer``), so it IS importable headlessly.
def _role_result(target_step0_rot6d=None, step0_valid: bool = True):
    """A minimal RoleTrajectory-shaped stand-in with valid, distinct poses."""
    from flore.utils.rot6d import rotmat_to_rot6d

    horizon = 16
    target_rot6d = rotmat_to_rot6d(_random_rotmats(horizon, seed=61))
    if target_step0_rot6d is not None:
        target_rot6d = target_rot6d.copy()
        target_rot6d[0] = target_step0_rot6d
    rng = np.random.default_rng(60)
    base = np.array([-0.27, -0.37, -0.09])
    positions = np.cumsum(rng.normal(0.0, 0.01, (horizon, 3)), axis=0) + base
    positions[0] = base
    pose_valid = np.ones(horizon, dtype=bool)
    pose_valid[0] = step0_valid
    # Duck-typed stand-in: _build_display_poses reads exactly these six fields.
    return SimpleNamespace(
        role="left",
        predicted_position=positions + rng.normal(0.0, 0.002, (horizon, 3)),
        predicted_rot6d=rotmat_to_rot6d(_random_rotmats(horizon, seed=62)),
        target_position=positions,
        target_rot6d=target_rot6d,
        pose_valid=pose_valid,
    )


def _left_geometry():
    from flore.s1_policy.inference.display import RoleGeometry

    return RoleGeometry(
        role="left",
        wrist_offset=wrist_offset_from_stored(_CORRECTED_T),
        tcp_triad_offset=tcp_triad_offset_from_stored(_CORRECTED_T),
        home_pos=np.asarray(DISPLAY_HOME_POS["left"], dtype=float),
        is_gripper_role=True,
    )


def test_build_display_poses_anchors_the_sample_to_an_upright_home():
    """Home-anchored mode puts step 0 exactly at the home pose, identity-rotated.

    This is the end-to-end statement of the operator's "+Z up" requirement: the drawn
    anchor frame is axis-aligned, so nothing points at the floor."""
    from flore.s1_policy.inference.viewer import _build_display_poses

    role_result = _role_result()
    tail = np.vstack([role_result.target_position[0], [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]])
    poses = _build_display_poses(role_result, tail, _left_geometry(), True)
    np.testing.assert_allclose(poses.target_position[0], DISPLAY_HOME_POS["left"], atol=1e-12)
    np.testing.assert_allclose(poses.target_wxyz[0], (1.0, 0.0, 0.0, 0.0), atol=1e-9)
    # The recorded tail stays joined to the target line.
    np.testing.assert_allclose(poses.episode_position[0], poses.target_position[0], atol=1e-12)


def test_build_display_poses_falls_back_to_raw_on_a_degenerate_reference():
    """A masked/degenerate step-0 target must NOT produce a garbage frame.

    An all-zero rot6d decodes to an all-zero matrix (det 0). Rebasing against that would
    collapse every drawn pose; instead the role degrades to the raw frame and stays
    finite, so the operator sees real data rather than a silently destroyed scene."""
    from flore.s1_policy.inference.viewer import _build_display_poses

    role_result = _role_result(target_step0_rot6d=np.zeros(6))
    poses = _build_display_poses(role_result, None, _left_geometry(), True)
    # Raw frame: positions untouched, i.e. NOT moved to the display home.
    np.testing.assert_allclose(poses.target_position, role_result.target_position, atol=1e-15)
    assert not np.allclose(poses.target_position[0], DISPLAY_HOME_POS["left"])
    assert np.isfinite(poses.predicted_position).all()
    assert np.isfinite(np.asarray(poses.predicted_wxyz)).all()
    # The degenerate rotation itself renders as identity rather than an invalid quaternion.
    np.testing.assert_allclose(poses.target_wxyz[0], (1.0, 0.0, 0.0, 0.0), atol=1e-12)


def test_build_display_poses_refuses_a_masked_step0_reference():
    """A step-0 target that is MASKED INVALID must not become the rebase reference.

    ``action_mask`` can mark step 0 invalid (20 such frames exist in clean94), and those poses
    are physically plausible — a proper rotation with a sensible position — so the
    determinant/finiteness guard does NOT fire on them. Without an explicit validity check the
    whole role would silently re-base onto a pose that is not a real measurement, and because
    every line and the cluster move together nothing would look wrong. Unreachable today
    (``drop_glitch_windows=True`` removes those anchors) but cheap insurance against a
    keep-glitch-windows run or a future filter change."""
    from flore.s1_policy.inference.viewer import _build_display_poses

    role_result = _role_result(step0_valid=False)
    poses = _build_display_poses(role_result, None, _left_geometry(), True)
    assert poses.anchored is False
    np.testing.assert_allclose(poses.target_position, role_result.target_position, atol=1e-15)
    # And the same sample WITH a valid step 0 does get anchored, so this isn't vacuous.
    assert _build_display_poses(_role_result(), None, _left_geometry(), True).anchored is True


def test_build_display_poses_raw_mode_is_a_passthrough():
    """With the toggle off the viewer stays a strict fidelity check."""
    from flore.s1_policy.inference.viewer import _build_display_poses

    role_result = _role_result()
    tail = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
    poses = _build_display_poses(role_result, tail, _left_geometry(), False)
    np.testing.assert_allclose(poses.predicted_position, role_result.predicted_position, atol=1e-15)
    np.testing.assert_allclose(poses.target_position, role_result.target_position, atol=1e-15)
    np.testing.assert_allclose(poses.episode_position, tail, atol=1e-15)


def test_build_display_poses_handles_a_missing_episode_tail():
    """A failed tail read yields None rather than crashing the rollout."""
    from flore.s1_policy.inference.viewer import _build_display_poses

    assert _build_display_poses(_role_result(), None, _left_geometry(), True).episode_position is None


def test_build_display_poses_preserves_errors_end_to_end():
    """The full viewer path preserves both error metrics — the property that matters most.

    Guards the whole chain (reference choice + rebase + quaternion conversion), not just the
    geometry primitives, so a future refactor of the viewer cannot start flattering the
    model without failing here."""
    from flore.s1_policy.inference.viewer import _build_display_poses

    role_result = _role_result()
    geometry = _left_geometry()
    raw = _build_display_poses(role_result, None, geometry, False)
    anchored = _build_display_poses(role_result, None, geometry, True)
    np.testing.assert_allclose(
        np.linalg.norm(anchored.predicted_position - anchored.target_position, axis=-1),
        np.linalg.norm(raw.predicted_position - raw.target_position, axis=-1),
        atol=1e-12,
    )

    def rotmat(wxyz):
        w, x, y, z = wxyz
        return np.array(
            [
                [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
            ]
        )

    for kind in ("raw", "anchored"):
        poses = raw if kind == "raw" else anchored
        errors = _geodesic_deg(
            np.array([rotmat(q) for q in poses.predicted_wxyz]),
            np.array([rotmat(q) for q in poses.target_wxyz]),
        )
        if kind == "raw":
            raw_errors = errors
        else:
            np.testing.assert_allclose(errors, raw_errors, atol=1e-7)


# ``_build_playback_data`` prepares the recorded demonstration tail for the Play button.
# The load-bearing property is that it rebases against the SAME reference the static action
# chunk uses (target step 0), so the swept gripper/TCP/tracker/wrist cluster rides exactly on
# the green recorded-path line and shows the identical frame relationship as the trained
# chunk. These pin that, plus the raw-mode passthrough and the gripper-fraction mapping.
def _forward_poses(tail_len: int = 8, seed: int = 71, with_gripper: bool = True):
    """A ``(times_s, per_role)`` stand-in shaped like ``episode_forward_poses`` output.

    Row 0 is deliberately set equal to a caller-supplied anchor so the test can assert the
    rebase lands it on the display home."""
    from flore.utils.rot6d import rotmat_to_rot6d

    rng = np.random.default_rng(seed)
    times = np.arange(tail_len, dtype=np.float64) / 15.0  # 15 Hz camera grid
    positions = np.cumsum(rng.normal(0.0, 0.01, (tail_len, 3)), axis=0) + np.array([-0.27, -0.37, -0.09])
    rot6d = rotmat_to_rot6d(_random_rotmats(tail_len, seed=seed + 1))
    entry = {
        "position": positions,
        "rot6d": rot6d,
        "gripper_m": (rng.uniform(0.0, 0.095, tail_len) if with_gripper else None),
    }
    return times, {"left": entry}


def _playback_result(forward, step0_valid: bool = True):
    """A minimal result whose target step 0 equals the tail's row 0 (the real invariant)."""
    from flore.utils.rot6d import rotmat_to_rot6d

    _times, per = forward
    tail0_pos = per["left"]["position"][0]
    tail0_rot6d = per["left"]["rot6d"][0]
    horizon = 16
    target_pos = np.repeat(tail0_pos[None], horizon, axis=0)
    target_rot6d = np.repeat(tail0_rot6d[None], horizon, axis=0)
    pose_valid = np.ones(horizon, dtype=bool)
    pose_valid[0] = step0_valid
    role_result = SimpleNamespace(
        role="left",
        predicted_position=target_pos,
        predicted_rot6d=rotmat_to_rot6d(_random_rotmats(horizon, seed=99)),
        target_position=target_pos,
        target_rot6d=target_rot6d,
        pose_valid=pose_valid,
    )
    return SimpleNamespace(roles=[role_result])


def test_build_playback_data_anchors_row0_to_home_like_the_static_chunk():
    """The swept recorded cluster starts EXACTLY at the display home, identity-rotated.

    Row 0 of the recorded tail is the sample's own anchor (== the action chunk's step-0
    target), so rebasing against that same reference must land it on the home pose — this is
    what guarantees the played-back GT cluster rides on the green recorded-path line and
    shares the trained chunk's gripper/TCP/tracker/wrist frame relationship."""
    from flore.s1_policy.inference.viewer import _build_playback_data

    forward = _forward_poses()
    result = _playback_result(forward)
    geometry = {"left": _left_geometry()}
    data = _build_playback_data(result, forward, geometry, home_anchored=True)
    assert data.per_role["left"]["anchored"] is True
    assert data.tail_len == 8
    np.testing.assert_allclose(data.per_role["left"]["position"][0], DISPLAY_HOME_POS["left"], atol=1e-12)
    np.testing.assert_allclose(data.per_role["left"]["wxyz"][0], (1.0, 0.0, 0.0, 0.0), atol=1e-9)
    # times pass through, 0-based and monotonic.
    assert data.times_s[0] == 0.0 and np.all(np.diff(data.times_s) > 0)


def test_build_playback_data_shares_the_static_scenes_frame_exactly():
    """Playback row k and the static chunk must land at the same point for a shared pose.

    Feed the tail's row 0 through BOTH ``_build_display_poses`` (as the target chunk) and
    ``_build_playback_data`` (as the recorded tail); the two must agree to machine precision,
    or the recorded demonstration would sit in a different frame than the GT chunk drawn from
    the same anchor — exactly the drift this shared-reference design exists to prevent."""
    from flore.s1_policy.inference.viewer import _build_display_poses, _build_playback_data

    forward = _forward_poses()
    result = _playback_result(forward)
    geometry = {"left": _left_geometry()}
    playback = _build_playback_data(result, forward, geometry, home_anchored=True)
    display = _build_display_poses(result.roles[0], forward[1]["left"]["position"], geometry["left"], True)
    # The static tail (episode_position) and the playback positions are the SAME recorded
    # points rebased the SAME way, so they must coincide row-for-row.
    np.testing.assert_allclose(playback.per_role["left"]["position"], display.episode_position, atol=1e-12)


def test_build_playback_data_raw_mode_is_a_passthrough():
    """With the toggle off the swept poses are the raw recorded ones (strict fidelity)."""
    from flore.s1_policy.inference.viewer import _build_playback_data

    forward = _forward_poses()
    result = _playback_result(forward)
    geometry = {"left": _left_geometry()}
    data = _build_playback_data(result, forward, geometry, home_anchored=False)
    assert data.per_role["left"]["anchored"] is False
    np.testing.assert_allclose(data.per_role["left"]["position"], forward[1]["left"]["position"], atol=1e-15)


def test_build_playback_data_maps_recorded_gripper_width_to_a_clamped_fraction():
    """The mesh jaw is driven by the RECORDED width during playback (no prediction exists).

    Widths must map to [0, 1] via GRIPPER_WIDTH_REF_M and clamp, since a recording can exceed
    the reference open width."""
    from flore.s1_policy.data.transforms import GRIPPER_WIDTH_REF_M
    from flore.s1_policy.inference.viewer import _build_playback_data

    times, per = _forward_poses()
    # Force a width above the reference to exercise the clamp.
    per["left"]["gripper_m"] = np.array([0.0, GRIPPER_WIDTH_REF_M / 2, GRIPPER_WIDTH_REF_M, 0.2] + [0.05] * 4)
    result = _playback_result((times, per))
    data = _build_playback_data(result, (times, per), {"left": _left_geometry()}, home_anchored=True)
    frac = data.per_role["left"]["gripper_frac"]
    assert frac is not None
    np.testing.assert_allclose(frac[:3], (0.0, 0.5, 1.0), atol=1e-9)
    assert frac[3] == 1.0  # 0.2 m clamps to fully open
    assert np.all((frac >= 0.0) & (frac <= 1.0))


def test_build_playback_data_falls_back_to_raw_on_a_degenerate_reference():
    """A masked/degenerate step-0 anchor leaves playback in the raw frame, mirroring the
    static path — never a silently collapsed sweep."""
    from flore.s1_policy.inference.viewer import _build_playback_data

    forward = _forward_poses()
    result = _playback_result(forward, step0_valid=False)
    data = _build_playback_data(result, forward, {"left": _left_geometry()}, home_anchored=True)
    assert data.per_role["left"]["anchored"] is False
    np.testing.assert_allclose(data.per_role["left"]["position"], forward[1]["left"]["position"], atol=1e-15)


def test_viewer_module_imports_without_viser():
    """viewer.py must stay importable with no viser present (it imports it lazily).

    Otherwise the headless ``--self-test`` path and these tests would depend on a GUI stack.
    """
    import subprocess
    import sys

    code = (
        "import sys; "
        "sys.modules['viser'] = None; "  # poison it: any real import would fail loudly
        "import flore.s1_policy.inference.viewer as v; "
        "assert hasattr(v, '_build_display_poses')"
    )
    subprocess.run([sys.executable, "-c", code], cwd=_REPO, check=True)


def test_display_module_imports_no_viser_or_third_party():
    """The geometry module must stay importable in a headless training/serving process."""
    import subprocess
    import sys

    code = (
        "import sys; import flore.s1_policy.inference.display; "
        "assert 'viser' not in sys.modules; assert 'yourdfpy' not in sys.modules; "
        "bad = [n for n, m in sys.modules.items() "
        "if '/third_party/' in str(getattr(m, '__file__', '')).replace('\\\\', '/')]; "
        "assert not bad, bad"
    )
    subprocess.run([sys.executable, "-c", code], cwd=_REPO, check=True)


# --------------------------------------------------------------------------- heading fallback
def _rz(deg):
    c, s = np.cos(np.radians(deg)), np.sin(np.radians(deg))
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _fake_role(role, rotmat, valid=True):
    from flore.utils.rot6d import rotmat_to_rot6d

    rot6d = np.repeat(rotmat_to_rot6d(np.asarray(rotmat, dtype=np.float64))[None], 3, axis=0)
    return SimpleNamespace(role=role, target_position=np.zeros((3, 3)), target_rot6d=rot6d,
                           pose_valid=np.array([valid, True, True]))


def test_heading_alignment_prefers_the_chest_and_falls_back_to_the_grip_yaw():
    """The one-hand page was MIRRORED: no chest -> no turn, while the hand's orientation was still
    re-based onto the home. wipe_board's LEFT reads -118.8 deg; the fallback turns by +118.8 and
    NAMES its source, the chest still wins whenever the layout has one, and raw mode / opt-out /
    an unreadable hand leave the scene unturned exactly as before."""
    from flore.s1_policy.inference.viewer import _heading_alignment

    left_only = SimpleNamespace(roles=[_fake_role("left", _rz(-118.8))])
    world, heading, source = _heading_alignment(left_only, True, True)
    assert heading == pytest.approx(-118.8)
    np.testing.assert_allclose(world, _rz(118.8), atol=1e-12)
    assert source == "LEFT grip yaw (no chest anchor)"

    with_chest = SimpleNamespace(roles=[_fake_role("left", _rz(-118.8)), _fake_role("head", _rz(-150.0))])
    world, heading, source = _heading_alignment(with_chest, True, True)
    assert source == "chest" and heading == pytest.approx(-150.0)
    np.testing.assert_allclose(world, _rz(150.0), atol=1e-12)

    # home-relative: a geometry whose home faces 30 deg shifts the hand's heading by 30
    geometry = {"left": SimpleNamespace(home_rotmat=_rz(30.0))}
    _, heading, _ = _heading_alignment(left_only, True, True, geometry)
    assert heading == pytest.approx(-148.8)

    assert _heading_alignment(left_only, False, True) == (None, 0.0, None)   # raw frame
    assert _heading_alignment(left_only, True, False) == (None, 0.0, None)   # display homes
    down = _rz(-118.8) @ np.array([[0.0, 0.0, -1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])  # x straight down
    assert _heading_alignment(SimpleNamespace(roles=[_fake_role("left", down)]), True, True) == (None, 0.0, None)
    invalid = SimpleNamespace(roles=[_fake_role("left", _rz(-118.8), valid=False)])
    assert _heading_alignment(invalid, True, True) == (None, 0.0, None)


def test_the_playback_cluster_carries_the_tracker_triad_rotation_like_the_static_one():
    """During ▶ Play the tracker triad sat ON the TCP's axes: the playback builder never passed
    ``tracker_local_rotmat``. Pin that both cluster builders do, so the puck's (-y,-x,-z) relabel
    is drawn on the swept cluster too."""
    import inspect

    from flore.s1_policy.inference import viewer

    src = inspect.getsource(viewer)
    playback = src[src.index("def _ensure_playback_nodes"):src.index("def _hide_playback_cluster")]
    static = src[src.index("def _ensure_cluster_nodes"):src.index("def _ensure_playback_nodes")]
    for segment in (playback, static):
        assert "_wxyz_from_rotmat(geometry.tracker_local_rotmat)" in segment
        assert 'if name == "tracker"' in segment
