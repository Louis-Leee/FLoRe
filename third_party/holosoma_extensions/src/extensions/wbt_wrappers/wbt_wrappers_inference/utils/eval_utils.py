"""Visualization, metrics, and CLI utilities for MuJoCo sim2sim evaluation."""

from __future__ import annotations

import os
import re
import subprocess
import sys

import mujoco
import numpy as np

from wbt_wrappers_inference.utils.obs_utils import detect_obs_mode as _shared_detect_obs_mode
from wbt_wrappers_inference.config_values.robot_config import (
    CONDITIONABLE_BODIES,
    ISAACLAB_TO_MUJOCO_DOF,
    VR_1POINT_BODIES,
    VR_1POINT_OFFSETS as _VR_1POINT_OFFSETS_LIST,
    VR_2POINT_BODIES,
    VR_2POINT_OFFSETS as _VR_2POINT_OFFSETS_LIST,
    VR_3POINT_BODIES,
    VR_3POINT_OFFSETS as _VR_3POINT_OFFSETS_LIST,
    VR_3POINT_OFFSETS_TCP as _VR_3POINT_OFFSETS_TCP_LIST,
)

# ---------------------------------------------------------------------------
# Joint ordering (Isaac Lab <-> MuJoCo)
# ---------------------------------------------------------------------------
ISAAC_TO_MUJOCO = ISAACLAB_TO_MUJOCO_DOF

# ---------------------------------------------------------------------------
# VR tracking bodies and offsets (from wbt_shared, cast to np.float64 for MuJoCo)
# ---------------------------------------------------------------------------
VR_3POINT_OFFSETS = np.array(_VR_3POINT_OFFSETS_LIST, dtype=np.float64)
# TCP-unified 3pt offsets: both HAND points at the 0.242 m force TCP (the TCP-unified
# retrain of exp:g1-wbt-3pt-hybrid-unified[-anneal]). Selected by obs term name in
# get_ref_marker_config, so every other 3pt checkpoint keeps the 0.18 m offsets.
VR_3POINT_OFFSETS_TCP = np.array(_VR_3POINT_OFFSETS_TCP_LIST, dtype=np.float64)
VR_2POINT_OFFSETS = np.array(_VR_2POINT_OFFSETS_LIST, dtype=np.float64)
VR_1POINT_OFFSETS = np.array(_VR_1POINT_OFFSETS_LIST, dtype=np.float64)

VR_3POINT_VIZ_OFFSETS = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.35]], dtype=np.float64)
VR_2POINT_VIZ_OFFSETS = np.zeros((2, 3), dtype=np.float64)
VR_1POINT_VIZ_OFFSETS = np.zeros((1, 3), dtype=np.float64)

_MM_BODIES = list(CONDITIONABLE_BODIES)

BASE_AZIMUTH = 135.0
WIDTH, HEIGHT = 640, 360

# ---------------------------------------------------------------------------
# UniFP force-tracking eval constants (g1-wbt-2pt-force benchmark).
# Must match wbt_training.config_values.force_tracking.HandForceConfig:
#   K=200 N/m, body_offsets_b=(0.0815,±0.003,0), bodies=(left,right)_wrist_yaw_link.
# ---------------------------------------------------------------------------
FORCE_STIFFNESS_K = 200.0  # N/m (HandForceConfig.stiffness_K)
FORCE_HANDS = ("left_wrist_yaw_link", "right_wrist_yaw_link")
# Force-deployment contact offset in each wrist_yaw_link LOCAL frame (left,
# right). This is a TRAINING-CONFIG constant (HandForceConfig.body_offsets_b)
# and MUST stay byte-identical to it. For the YAM gripper the wrist→hand chain
# is wrist_yaw -(ft_sensor 0.072)-> ft_sensor -(palm 0.0095,±0.003)-> rubber_hand,
# so the net wrist_yaw-local offset is 0.0815 (the retired Pika gripper used a
# direct 0.0415 mount). All chain joints have rpy=0, so the force is still applied
# at the rubber_hand point with no force-arm error.
FORCE_HAND_OFFSETS_B = np.array([[0.0815, +0.003, 0.0], [0.0815, -0.003, 0.0]], dtype=np.float64)
# Drop the first N frames before aggregating force metrics: F_cmd is injected as
# a constant STEP at eval (not the trained trapezoidal ramp), so the arm needs
# ~1 s (50 frames @50 Hz) to settle on the displaced target.
FORCE_SETTLE_FRAMES = 50

# --- Force-arrow video overlay (g1-wbt-2pt-force) -------------------------
# Draw the force on each rubber_hand as a colored 3-D arrow in the eval video.
#   ORANGE  = F_ext : the REAL physical disturbance force applied to PhysX
#                     (mj_applyFT at the hand point, Eval-3 only).
#   PURPLE  = F_cmd : the COMMANDED intent force the policy OBSERVES (obs-only;
#                     never written to physics — it only shifts the Eq.2 target).
# Both are PELVIS-frame Newtons rotated to WORLD by the LIVE pelvis quat — the
# identical transform the interface uses to inject F_ext and the obs builder
# uses to feed F_cmd, so the arrow direction is physically faithful.
FORCE_ARROW_RGBA_EXT = np.array([1.0, 0.45, 0.0, 0.95])  # orange = F_ext (real)
FORCE_ARROW_RGBA_CMD = np.array([0.62, 0.20, 0.92, 0.95])  # purple = F_cmd (intent)
FORCE_ARROW_M_PER_N = 0.018  # arrow length per Newton (10 N -> 18 cm, 25 N -> 45 cm)
FORCE_ARROW_MIN_LEN = 0.08  # floor so small forces stay visible (m)
FORCE_ARROW_WIDTH = 0.018  # arrow shaft radius (m)


def _quat_to_mat_wxyz(q):
    """3x3 world rotation matrix from a wxyz quaternion (numpy, MuJoCo-native)."""
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------
def _yaw(qpos):
    w, x, y, z = qpos[3:7]
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y**2 + z**2))


_HEIGHT_SCAN_RAY_DIR = np.array([0.0, 0.0, -1.0], dtype=np.float64)
_HEIGHT_SCAN_GRID: dict[int, np.ndarray] = {}
# Restrict rays to terrain geoms only (group 0 = obstacle meshes, group 2 =
# floor plane). Robot collision geoms live in group 3; without this filter
# mj_ray hits the robot's own head/shoulders/hands and the scan returns
# spurious "obstacle-top" readings even on flat floor — policy-killing noise
# for height-scan terrain models.
_HEIGHT_SCAN_GEOMGROUP = np.array([1, 0, 1, 0, 0, 0], dtype=np.uint8)


def _get_height_scan_grid(n_per_side: int, grid_size: float) -> np.ndarray:
    """Get or create cached grid offsets (n_per_side^2, 2) for height scan.

    Must match IsaacLab's GridPatternCfg(ordering="xy"): torch.meshgrid(x, y,
    indexing="xy") → shape (len(y), len(x)) = y-major flattening. Using "ij"
    here silently transposes the scan and the policy sees obstacles mirrored
    across the grid diagonal — policy-destroying noise even if ray values are
    numerically correct.
    """
    if n_per_side not in _HEIGHT_SCAN_GRID:
        coords = np.linspace(-grid_size / 2, grid_size / 2, n_per_side)
        xx, yy = np.meshgrid(coords, coords, indexing="xy")
        _HEIGHT_SCAN_GRID[n_per_side] = np.column_stack([xx.ravel(), yy.ravel()])
    return _HEIGHT_SCAN_GRID[n_per_side]


def compute_height_scan_mj_ray(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    robot_pos: np.ndarray,
    robot_yaw: float,
    n_rays: int = 289,
    grid_resolution: float = 0.1,
    grid_size: float = 1.6,
    ray_height: float = 20.0,
    offset: float = 0.5,
) -> np.ndarray:
    """Cast rays downward in a yaw-aligned grid to compute height scan.

    Matches IsaacSim RayCasterCfg: GridPatternCfg(resolution=0.1, size=[1.6, 1.6]),
    attach_yaw_only=True, offset=(0, 0, 20).
    """
    n_per_side = int(grid_size / grid_resolution) + 1  # 17
    assert n_per_side * n_per_side == n_rays, f"Grid {n_per_side}x{n_per_side}={n_per_side**2} != n_rays={n_rays}"

    grid = _get_height_scan_grid(n_per_side, grid_size)
    cos_yaw, sin_yaw = np.cos(robot_yaw), np.sin(robot_yaw)
    wz = robot_pos[2] + ray_height
    heights = np.empty(n_rays, dtype=np.float32)
    pnt = np.zeros(3, dtype=np.float64)
    geomid = np.zeros(1, dtype=np.int32)

    for idx in range(n_rays):
        xi, yi = grid[idx]
        pnt[0] = robot_pos[0] + cos_yaw * xi - sin_yaw * yi
        pnt[1] = robot_pos[1] + sin_yaw * xi + cos_yaw * yi
        pnt[2] = wz

        dist = mujoco.mj_ray(model, data, pnt, _HEIGHT_SCAN_RAY_DIR, _HEIGHT_SCAN_GEOMGROUP, 1, -1, geomid)
        if dist >= 0:
            heights[idx] = robot_pos[2] - (wz - dist) - offset
        else:
            heights[idx] = robot_pos[2] - offset

    return heights


# Large flat rectangle under obstacles written by
# wbt_training.utils.obstacle_helpers.save_obstacles_to_file. It z-fights with
# the scene's plane floor, so we drop any connected component matching both
# thresholds (large XY footprint, near-zero z-span above origin).
_GROUND_PLATE_MIN_XY_AREA_M2 = 25.0
_GROUND_PLATE_MAX_Z_M = 0.02


def _split_terrain_components(obj_path: str):
    """Load `obj_path`, split into connected components, drop the ground plate."""
    import trimesh

    combined = trimesh.load(obj_path, force="mesh")
    parts = combined.split(only_watertight=False) if hasattr(combined, "split") else [combined]
    if len(parts) == 0:
        parts = [combined]
    filtered = []
    for p in parts:
        xy_area = float((p.bounds[1, 0] - p.bounds[0, 0]) * (p.bounds[1, 1] - p.bounds[0, 1]))
        z_max = float(p.bounds[1, 2])
        if xy_area > _GROUND_PLATE_MIN_XY_AREA_M2 and z_max <= _GROUND_PLATE_MAX_Z_M:
            continue
        filtered.append(p)
    return filtered if filtered else parts


# Temp dir for per-component OBJs written by inject_terrain_mesh. MuJoCo needs
# the files on disk for the lifetime of the model, so cleanup waits for exit.
_TERRAIN_PARTS_DIR: str | None = None


def _terrain_parts_dir() -> str:
    """One shared scratch dir per process, auto-cleaned on interpreter exit."""
    global _TERRAIN_PARTS_DIR
    if _TERRAIN_PARTS_DIR is None:
        import atexit
        import shutil
        import tempfile

        _TERRAIN_PARTS_DIR = tempfile.mkdtemp(prefix="terrain_parts_")
        atexit.register(shutil.rmtree, _TERRAIN_PARTS_DIR, ignore_errors=True)
    return _TERRAIN_PARTS_DIR


def inject_terrain_mesh(xml_str: str, obj_path: str) -> str:
    """Inject a terrain OBJ into MuJoCo scene XML as per-component mesh geoms.

    MuJoCo wraps each ``<geom type="mesh">`` in a single convex hull, so
    emitting the combined OBJ as one mesh geom creates spurious contacts
    across the whole footprint. Split into connected components (each of
    which is already convex for box-style obstacles) and emit one geom per
    component. The existing ``<geom name="floor">`` plane is kept.
    """
    parts = _split_terrain_components(obj_path)
    tmpdir = _terrain_parts_dir()
    asset_lines = ['    <material name="obstacle_mat" rgba="0.55 0.42 0.30 1"/>']
    geom_lines = []
    for i, part in enumerate(parts):
        part_obj = os.path.join(tmpdir, f"obs_{i:03d}.obj")
        part.export(part_obj)
        asset_lines.append(f'    <mesh name="obs_{i:03d}" file="{part_obj}"/>')
        geom_lines.append(
            f'    <geom name="obstacle_{i:03d}" type="mesh" mesh="obs_{i:03d}" '
            f'material="obstacle_mat" contype="1" conaffinity="1"/>'
        )
    asset_block = "\n".join(asset_lines) + "\n"
    if "<asset>" in xml_str:
        xml_str = xml_str.replace("<asset>", "<asset>\n" + asset_block, 1)
    else:
        xml_str = xml_str.replace("<worldbody>", "<asset>\n" + asset_block + "  </asset>\n\n  <worldbody>", 1)
    geom_block = "\n".join(geom_lines) + "\n"
    xml_str, n = re.subn(r'(<geom\s+name="floor"[^/]*/>)', r"\1\n" + geom_block, xml_str, count=1)
    if n == 0:
        raise ValueError("Could not find <geom name='floor'.../> in XML to inject terrain mesh")
    return xml_str


def inject_terrain_boxes(xml_str: str, npy_path: str, variant: int = 0) -> str:
    """Inject terrain NPY obstacles as individual ``<geom type="box">`` primitives.

    NPY format (from ``terrain_holosoma/*_terrain.npy``): shape
    ``(N_types, N_variants, 10)`` with each row =
    ``[px, py, pz, qw, qx, qy, qz, sx, sy, sz]``.
    ``(px, py, pz)`` is box center, ``(qw, qx, qy, qz)`` is wxyz quaternion,
    ``(sx, sy, sz)`` is the FULL extent (trimesh convention); MuJoCo ``size``
    is HALF-extent so we halve. Floor plane is kept unchanged.
    """
    arr = np.load(npy_path, allow_pickle=True)
    assert arr.ndim == 3 and arr.shape[-1] == 10, f"Expected (N_types, N_variants, 10), got {arr.shape}"
    n_types, n_variants, _ = arr.shape
    v = variant % n_variants
    lines = []
    for ti in range(n_types):
        row = arr[ti, v]
        size_full = row[7:10]
        if np.any(size_full <= 0):
            continue
        px, py, pz = row[0:3]
        qw, qx, qy, qz = row[3:7]
        sx, sy, sz = size_full / 2.0
        lines.append(
            f'    <geom name="obstacle_{ti:02d}" type="box" '
            f'pos="{px:.6f} {py:.6f} {pz:.6f}" '
            f'quat="{qw:.6f} {qx:.6f} {qy:.6f} {qz:.6f}" '
            f'size="{sx:.6f} {sy:.6f} {sz:.6f}" '
            f'contype="1" conaffinity="1" rgba="0.5 0.35 0.2 1.0"/>'
        )
    if not lines:
        return xml_str
    inject_block = "\n".join(lines) + "\n"
    new_xml, n = re.subn(r"(<worldbody[^>]*>)", r"\1\n" + inject_block, xml_str, count=1)
    if n == 0:
        raise ValueError("Could not find <worldbody> tag in XML")
    return new_xml


def _load_model(xml_path: str, terrain_obj: str | None = None) -> mujoco.MjModel:
    abs_xml = os.path.abspath(xml_path)
    xml_dir = os.path.dirname(abs_xml)
    search = xml_dir
    found_assets = False
    for _ in range(5):
        if os.path.isdir(os.path.join(search, "assets")):
            found_assets = True
            break
        parent = os.path.dirname(search)
        if parent == search:
            break
        search = parent
    if not found_assets:
        search = xml_dir
    orig = os.getcwd()
    os.chdir(search)
    try:
        with open(abs_xml) as f:
            xml_str = f.read()
        if 'geom2="floor"' in xml_str and '<geom name="floor"' not in xml_str:
            scene_visual = """
    <visual>
        <headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3" specular="0 0 0"/>
        <rgba haze="0.15 0.25 0.35 1"/>
    </visual>
    <asset>
        <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="3072"/>
        <texture type="2d" name="groundplane" builtin="checker" mark="edge" rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3"
                markrgb="0.8 0.8 0.8" width="300" height="300"/>
        <material name="groundplane" texture="groundplane" texuniform="true" texrepeat="5 5" reflectance="0.2"/>
    </asset>"""
            floor_geom = '<light pos="0 0 1.5" dir="0 0 -1" directional="true"/>\n    <geom name="floor" size="0 0 0.01" type="plane" material="groundplane" group="2"/>'
            xml_str = xml_str.replace("<worldbody>", f"{scene_visual}\n<worldbody>\n    {floor_geom}", 1)
        if terrain_obj is not None:
            abs_path = os.path.abspath(terrain_obj)
            if abs_path.endswith(".npy"):
                xml_str = inject_terrain_boxes(xml_str, abs_path)
            else:
                xml_str = inject_terrain_mesh(xml_str, abs_path)
        return mujoco.MjModel.from_xml_string(xml_str)
    finally:
        os.chdir(orig)


def _make_renderer(model, maxgeom=10000, fixed_camera=False, cam_distance=0.0):
    model.vis.global_.offwidth = max(model.vis.global_.offwidth, WIDTH)
    model.vis.global_.offheight = max(model.vis.global_.offheight, HEIGHT)
    renderer = mujoco.Renderer(model, height=HEIGHT, width=WIDTH)
    renderer._scene = mujoco.MjvScene(model, maxgeom=maxgeom)
    cam = mujoco.MjvCamera()
    if fixed_camera:
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.lookat[:] = [0.0, 0.0, 0.8]
        cam.distance = cam_distance if cam_distance > 0 else 4.0
        cam.elevation = -15.0
        cam.azimuth = BASE_AZIMUTH
    else:
        cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        cam.trackbodyid = 1
        cam.distance = cam_distance if cam_distance > 0 else 3.0
        cam.elevation = -20.0
        cam.azimuth = BASE_AZIMUTH
    return renderer, cam


# ---------------------------------------------------------------------------
# Scene markers (VR axes, MaskedMimic body spheres)
# ---------------------------------------------------------------------------
_MM_BODY_COLORS = {
    "pelvis": [1.0, 0.2, 0.2, 0.85],
    "torso_link": [0.2, 1.0, 0.2, 0.85],
    "right_ankle_roll_link": [0.2, 0.2, 1.0, 0.85],
    "left_ankle_roll_link": [1.0, 1.0, 0.2, 0.85],
    "left_wrist_yaw_link": [1.0, 0.2, 1.0, 0.85],
    "right_wrist_yaw_link": [0.2, 1.0, 1.0, 0.85],
}
_MM_BODY_DEFAULT_COLOR = [1.0, 0.6, 0.0, 0.85]


def _add_mm_body_markers(scene, model, data, body_names, radius=0.04):
    for name in body_names:
        if scene.ngeom >= scene.maxgeom:
            return
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if bid < 0:
            continue
        geom = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(
            geom,
            mujoco.mjtGeom.mjGEOM_SPHERE,
            np.array([radius, 0, 0]),
            data.xpos[bid].copy().astype(np.float64),
            np.eye(3).flatten().astype(np.float64),
            np.array(_MM_BODY_COLORS.get(name, _MM_BODY_DEFAULT_COLOR)),
        )
        scene.ngeom += 1


def _add_vr_axis_markers(scene, positions, rot_matrices, axis_length=0.15, axis_radius=0.008):
    axis_rgba = [[1.0, 0.15, 0.15, 0.95], [0.15, 1.0, 0.15, 0.95], [0.15, 0.15, 1.0, 0.95]]
    for pos, R in zip(positions, rot_matrices):
        if scene.ngeom < scene.maxgeom:
            geom = scene.geoms[scene.ngeom]
            mujoco.mjv_initGeom(
                geom,
                mujoco.mjtGeom.mjGEOM_SPHERE,
                np.array([0.02, 0, 0]),
                pos.astype(np.float64),
                np.eye(3).flatten().astype(np.float64),
                np.array([1.0, 1.0, 1.0, 0.8]),
            )
            scene.ngeom += 1
        for ax in range(3):
            if scene.ngeom >= scene.maxgeom:
                return
            geom = scene.geoms[scene.ngeom]
            axis_dir = R[:, ax]
            end = pos + axis_dir * axis_length
            mid = ((pos + end) / 2).astype(np.float64)
            z = axis_dir / (np.linalg.norm(axis_dir) + 1e-8)
            perp = np.array([0, 1, 0]) if abs(z[0]) < 0.9 else np.array([1, 0, 0])
            x = np.cross(z, perp)
            x /= np.linalg.norm(x) + 1e-8
            y = np.cross(z, x)
            mat = np.column_stack([x, y, z]).flatten().astype(np.float64)
            mujoco.mjv_initGeom(
                geom,
                mujoco.mjtGeom.mjGEOM_CAPSULE,
                np.array([axis_radius, axis_length / 2, 0.0]),
                mid,
                mat,
                np.array(axis_rgba[ax]),
            )
            scene.ngeom += 1


def _get_vr_positions_and_rotations_from_mujoco(model, data, vr_body_names, vr_offsets):
    positions, rotations = [], []
    for i, name in enumerate(vr_body_names):
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        bpos = data.xpos[bid].copy()
        R = data.xmat[bid].reshape(3, 3).copy()
        positions.append(bpos + R @ vr_offsets[i])
        rotations.append(R)
    return positions, rotations


def parse_force_local_env(var_name):
    """Parse a HOLOSOMA_FORCE_*_LOCAL env var to (2,3) pelvis-frame N, or None.

    Returns None when the var is unset/empty OR all-zero, so callers can cheaply
    decide whether to draw an arrow. Format: "L_fx,L_fy,L_fz,R_fx,R_fy,R_fz".
    Mirrors MetricsLogger._parse_force_cmd_env (which returns zeros for the F=0
    baseline metrics) but is None-valued for the rendering path."""
    raw = os.environ.get(var_name, "")
    if not raw:
        return None
    vals = [float(x) for x in raw.split(",")]
    if len(vals) != 6:
        raise ValueError(
            f"{var_name} must be 6 comma-separated Newtons [L_fx,L_fy,L_fz,R_fx,R_fy,R_fz], got {len(vals)}: {raw!r}"
        )
    arr = np.array(vals, dtype=np.float64).reshape(2, 3)
    return arr if np.any(arr != 0.0) else None


def _pelvis_world_R(qpos):
    """Live pelvis (root) world rotation from the free-joint quat (wxyz)."""
    return _quat_to_mat_wxyz(qpos[3:7])


def _add_force_arrow(scene, origin, force_world, rgba, scale=FORCE_ARROW_M_PER_N):
    """Append one MuJoCo arrow geom for a world-frame force vector.

    The arrow starts at ``origin`` (the rubber_hand point) and points along
    ``force_world`` with length proportional to the magnitude. Returns True if a
    geom was added. Near-zero forces are skipped (no glyph)."""
    mag = float(np.linalg.norm(force_world))
    if mag < 1e-6 or scene.ngeom >= scene.maxgeom:
        return False
    length = max(mag * scale, FORCE_ARROW_MIN_LEN)
    frm = np.asarray(origin, dtype=np.float64)
    to = frm + (np.asarray(force_world, dtype=np.float64) / mag) * length
    geom = scene.geoms[scene.ngeom]
    # mjv_initGeom seeds rgba/type; mjv_connector then orients the arrow from->to.
    mujoco.mjv_initGeom(
        geom,
        mujoco.mjtGeom.mjGEOM_ARROW,
        np.zeros(3),
        np.zeros(3),
        np.eye(3).flatten(),
        np.asarray(rgba, dtype=np.float64),
    )
    mujoco.mjv_connector(geom, int(mujoco.mjtGeom.mjGEOM_ARROW), FORCE_ARROW_WIDTH, frm, to)
    scene.ngeom += 1
    return True


def _add_hand_force_arrows(scene, model, data, force_cmd_pelvis=None, force_ext_pelvis=None):
    """Draw F_cmd (purple) and F_ext (orange) arrows on both rubber_hands.

    *force_cmd_pelvis* / *force_ext_pelvis* are (2,3) PELVIS-frame Newton arrays
    ([left, right]) or None. Each non-zero hand vector is rotated to the world
    by the LIVE pelvis quat (identical to the physics injection + obs feed) and
    anchored at the rubber_hand point ``xpos[wrist] + R_wrist @ offset_b`` — the
    exact point mj_applyFT pushes on. Drawn LAST so the arrows sit on top."""
    if force_cmd_pelvis is None and force_ext_pelvis is None:
        return
    R_root = _pelvis_world_R(data.qpos)
    for hi, hname in enumerate(FORCE_HANDS):
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, hname)
        if bid < 0:
            continue
        hand_pt = data.xpos[bid] + data.xmat[bid].reshape(3, 3) @ FORCE_HAND_OFFSETS_B[hi]
        # F_ext first (real physics), then F_cmd on top (intent), so when they
        # oppose in the Eval-3 cancel case both stay legible.
        if force_ext_pelvis is not None:
            _add_force_arrow(scene, hand_pt, R_root @ np.asarray(force_ext_pelvis)[hi], FORCE_ARROW_RGBA_EXT)
        if force_cmd_pelvis is not None:
            _add_force_arrow(scene, hand_pt, R_root @ np.asarray(force_cmd_pelvis)[hi], FORCE_ARROW_RGBA_CMD)


# ---------------------------------------------------------------------------
# Obs mode helpers
# ---------------------------------------------------------------------------
def detect_eval_obs_mode(obs_terms) -> str:
    names = list(obs_terms.keys()) if isinstance(obs_terms, dict) else list(obs_terms)
    mode = _shared_detect_obs_mode(names)
    return "tml" if mode == "dense" else mode


def detect_obs_mode_from_config(exp_config: dict | None) -> str:
    if exp_config is None:
        return "tml"
    terms = exp_config.get("observation", {}).get("groups", {}).get("actor_obs", {}).get("terms", {})
    return detect_eval_obs_mode(terms)


def get_ref_marker_config(obs_mode: str, obs_terms=None):
    """Return (vr_body_names, vr_offsets, vr_viz_offsets, mm_body_names) for the given obs mode.

    ``obs_terms`` (the checkpoint's actor-obs term names) selects the 3pt HAND geometry: a
    ``vr_3point_pos_tcp`` slot means the TCP-unified retrain of
    ``exp:g1-wbt-3pt-hybrid-unified[-anneal]``, whose hands are tracked at the 0.242 m
    force TCP rather than the 0.18 m VR lever. ``obs_mode`` alone cannot tell them apart
    (both detect as ``"3pt"``), so callers should pass the terms; omitting them keeps the
    historical 0.18 m behaviour.

    NOTE on the viz offsets: for the TCP-unified path the VIZ offsets are the REAL offsets.
    The default 3pt viz offsets deliberately zero the hand levers, which draws the hand
    target at the wrist ORIGIN — ~18 cm off the tracked point. Since the whole point of the
    TCP-unified geometry is that position and force share ONE visible point, zeroing here
    would hide exactly what we want to see, so the TCP path returns the true offsets.
    """
    if obs_mode in ("3pt", "3pt_global"):
        names = list(obs_terms.keys()) if isinstance(obs_terms, dict) else list(obs_terms or [])
        if any("vr_3point_pos_tcp" in n for n in names):
            return VR_3POINT_BODIES, VR_3POINT_OFFSETS_TCP, VR_3POINT_OFFSETS_TCP, None
        return VR_3POINT_BODIES, VR_3POINT_OFFSETS, VR_3POINT_VIZ_OFFSETS, None
    elif obs_mode == "2pt":
        return VR_2POINT_BODIES, VR_2POINT_OFFSETS, VR_2POINT_VIZ_OFFSETS, None
    elif obs_mode == "1pt":
        return VR_1POINT_BODIES, VR_1POINT_OFFSETS, VR_1POINT_VIZ_OFFSETS, None
    elif obs_mode == "maskedmimic":
        return None, None, None, _MM_BODIES
    return None, None, None, None


# ---------------------------------------------------------------------------
# Video composition
# ---------------------------------------------------------------------------
_PIL_AVAILABLE = None
_FONT_CACHE: dict[int, object] = {}


def _get_font(fontsize):
    if fontsize in _FONT_CACHE:
        return _FONT_CACHE[fontsize]
    from PIL import ImageFont

    font = None
    for path in ["/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/System/Library/Fonts/Helvetica.ttc"]:
        try:
            font = ImageFont.truetype(path, fontsize)
            break
        except Exception:
            continue
    if font is None:
        font = ImageFont.load_default()
    _FONT_CACHE[fontsize] = font
    return font


def _try_draw_text(img_array, text_items, bar_height=40):
    global _PIL_AVAILABLE
    if _PIL_AVAILABLE is None:
        try:
            from PIL import Image, ImageDraw

            _PIL_AVAILABLE = True
        except ImportError:
            _PIL_AVAILABLE = False

    h, w = img_array.shape[:2]
    new_h = h + bar_height
    canvas = np.zeros((new_h, w, 3), dtype=np.uint8)
    canvas[:h] = img_array
    canvas[h:] = 30
    if _PIL_AVAILABLE:
        from PIL import Image, ImageDraw

        img = Image.fromarray(canvas)
        draw = ImageDraw.Draw(img)
        for text, x, y, color, fontsize in text_items:
            draw.text((x, y), text, fill=color, font=_get_font(fontsize))
        return np.array(img)
    return canvas


def _error_color(val, low=2.0, high=15.0):
    t = np.clip((val - low) / (high - low + 1e-8), 0, 1)
    if t < 0.5:
        r, g = int(255 * t * 2), 255
    else:
        r, g = 255, int(255 * (1 - t) * 2)
    return (r, g, 50)


def compose_and_save(
    ref_frames, policy_frames, output_path, fps=50, model_label="", motion_label="", metrics=None, prepend_frames=0
):
    n = min(len(ref_frames), len(policy_frames))
    if n == 0:
        print("No frames to compose", file=sys.stderr)
        return False

    has_metrics = metrics is not None and len(metrics.get("joint_angle_error_mean_deg", [])) > 0
    summary = metrics.get("summary", {}) if has_metrics else {}

    combined = []
    for i in range(n):
        ref = ref_frames[i].copy()
        pol = policy_frames[i].copy()
        ref[:20, :, :] = 40
        pol[:20, :, :] = 40
        frame = np.concatenate([ref, pol], axis=1)

        if has_metrics and i < len(metrics["joint_angle_error_mean_deg"]):
            je = metrics["joint_angle_error_mean_deg"][i]
            bp = metrics["body_pos_error_mean_cm"][i] if i < len(metrics.get("body_pos_error_mean_cm", [])) else 0
            rp = metrics["root_pos_error_cm"][i] if i < len(metrics.get("root_pos_error_cm", [])) else 0
            HALF = frame.shape[1] // 2
            text_items = [
                (f"Frame {i}/{n}", 10, frame.shape[0] + 5, (200, 200, 200), 13),
                (f"Joint err: {je:.1f}\u00b0", 10, frame.shape[0] + 22, _error_color(je, 2.0, 20.0), 13),
                (f"Body err: {bp:.1f}cm", HALF // 2, frame.shape[0] + 22, _error_color(bp, 3.0, 30.0), 13),
                (f"Root err: {rp:.1f}cm", HALF, frame.shape[0] + 22, _error_color(rp, 5.0, 50.0), 13),
            ]
            if summary:
                text_items.append(
                    (
                        f"Avg joint: {summary.get('mean_joint_angle_error_deg', 0):.1f}\u00b0 | "
                        f"Avg body: {summary.get('mean_body_pos_error_cm', 0):.1f}cm | "
                        f"Survival: {summary.get('survival_ratio', 1.0):.0%}",
                        HALF + HALF // 2,
                        frame.shape[0] + 5,
                        (180, 180, 180),
                        11,
                    )
                )
            frame = _try_draw_text(frame, text_items, bar_height=40)
        if prepend_frames > 0 and i < prepend_frames:
            _pw = frame.shape[1]
            frame = _try_draw_text(
                frame,
                [
                    (
                        "[ Default Pose -> Init Pose Transition ]",
                        _pw // 2 - 180,
                        frame.shape[0] // 2 - 12,
                        (255, 200, 50),
                        18,
                    )
                ],
                bar_height=0,
            )
        combined.append(frame)

    h, w = combined[0].shape[:2]
    HALF = w // 2
    filters = ["drawtext=text='reference motion':fontsize=20:fontcolor=white:x=10:y=5"]
    if motion_label:
        escaped = motion_label.replace("'", "\\'").replace(":", "\\:")
        filters.append(f"drawtext=text='{escaped}':fontsize=20:fontcolor=white:x={HALF + 10}:y=5")
    else:
        filters.append(f"drawtext=text='policy rollout':fontsize=20:fontcolor=white:x={HALF + 10}:y=5")
    if model_label:
        escaped = model_label.replace("'", "\\'").replace(":", "\\:")
        filters.append(f"drawtext=text='{escaped}':fontsize=14:fontcolor=yellow:x={HALF + 10}:y=h-25")

    # Force-arrow legend on the policy panel: only when a force condition is
    # active (same env vars the renderer/obs/physics read — single source of
    # truth). Orange = F_ext (real physics), purple = F_cmd (obs-only intent).
    # NB: commas separate FILTERS in ffmpeg's -vf chain, so the legend text uses
    # spaces (e.g. "R=[10 0 0]N") — a stray comma would void the whole chain and
    # silently drop ALL overlay text via the base_cmd fallback.
    def _fvec(v):
        return f"x{v[0]:.0f} y{v[1]:.0f} z{v[2]:.0f} N"

    def _hands_label(arr):
        """Label whichever hand(s) carry a non-zero force (L and/or R), so a
        left/bimanual force is described too — not just the right hand."""
        parts = []
        if np.any(arr[0] != 0.0):
            parts.append(f"L= {_fvec(arr[0])}")
        if np.any(arr[1] != 0.0):
            parts.append(f"R= {_fvec(arr[1])}")
        return "  ".join(parts)

    _fc = parse_force_local_env("HOLOSOMA_FORCE_CMD_LOCAL")
    _fe = parse_force_local_env("HOLOSOMA_FORCE_EXT_LOCAL")
    _ly = 30
    if _fe is not None:
        _lbl = f"F_ext (real force)  {_hands_label(_fe)}"
        filters.append(
            f"drawtext=text='{_lbl}':fontsize=15:fontcolor=orange:box=1:boxcolor=black@0.5:x={HALF + 10}:y={_ly}"
        )
        _ly += 22
    if _fc is not None:
        _lbl = f"F_cmd (intent)  {_hands_label(_fc)}"
        filters.append(
            f"drawtext=text='{_lbl}':fontsize=15:fontcolor=mediumpurple:box=1:boxcolor=black@0.5:x={HALF + 10}:y={_ly}"
        )

    def _write_frames(cmd, frames):
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        try:
            for f in frames:
                proc.stdin.write(f.tobytes())
            proc.stdin.close()
        except BrokenPipeError:
            pass
        proc.wait()
        return proc.returncode

    base_cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "rawvideo",
        "-pixel_format",
        "rgb24",
        "-video_size",
        f"{w}x{h}",
        "-framerate",
        str(fps),
        "-i",
        "pipe:0",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-crf",
        "23",
        output_path,
    ]
    cmd_with_text = base_cmd[:-1] + ["-vf", ",".join(filters)] + base_cmd[-1:]
    if _write_frames(cmd_with_text, combined) != 0:
        # drawtext chain failed (e.g. an unescaped special char) — fall back to a
        # plain encode, but make it LOUD: a silent fallback once dropped every
        # overlay (labels + force legend) without a trace.
        print(
            f"WARN: ffmpeg drawtext overlay failed; re-encoding WITHOUT text overlay. filters={filters!r}",
            file=sys.stderr,
        )
        _write_frames(base_cmd, combined)

    return os.path.exists(output_path)


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------
# Map a trained-checkpoint URDF stem (from the ONNX ``robot_urdf_path`` metadata)
# to the MuJoCo render scene whose hand geometry matches it. Mirrors the browser
# viz's ``_URDF_STEM_TO_SCENE`` (wbt_realtime_eval/main.py): a checkpoint trained
# with ``g1/g1_29dof_yam.urdf`` must render the yam gripper, not the old rubber
# hand. Pinocchio FK is hand-geometry-independent, so the render scene must be
# chosen separately from the obs-side URDF.
_URDF_STEM_TO_SCENE = {
    "g1_29dof": "scene_g1_29dof_wbt_plane.xml",
    "g1_29dof_yam": "scene_g1_29dof_wbt_plane_yam.xml",
    "g1_29dof_pika_gripper": "scene_g1_29dof_wbt_plane_pika_gripper.xml",
}


def _decode_meta_str(meta: dict | None, key: str) -> str:
    """Read a string ONNX-metadata value, JSON-decoding it if it was stored
    JSON-encoded (holosoma writes metadata via ``json.dumps``). Robust to a bare
    (un-encoded) value too."""
    import json

    raw = (meta or {}).get(key, "")
    if not raw:
        return ""
    try:
        decoded = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return raw.strip().strip('"')
    return decoded if isinstance(decoded, str) else raw


def resolve_xml(xml_arg=None, meta: dict | None = None):
    """Resolve the MuJoCo render/sim scene XML.

    Priority:
      1. An explicit ``xml_arg`` (``--task.headless-xml``) — used verbatim.
      2. Auto-derive from the ONNX ``robot_urdf_path`` stem in *meta* (e.g.
         ``g1_29dof_yam`` → the yam-gripper scene) so a yam checkpoint renders the
         yam gripper with no flag. A mapped-but-missing scene HARD-FAILS rather
         than silently rendering the wrong hand.
      3. Fall back to the default ``scene_g1_29dof_wbt_plane.xml`` for an
         unmapped / absent stem (backward-compatible for any old checkpoint).
    """
    if xml_arg is not None:
        return xml_arg
    # Resolve relative to wbt_wrappers_inference package root (one level up from utils/)
    _pkg_dir = os.path.dirname(os.path.dirname(__file__))
    _scenes_dir = os.path.join(
        _pkg_dir, "../../../../thirdparty/holosoma/src/holosoma/holosoma/data", "robots/g1/scenes"
    )

    urdf_path = _decode_meta_str(meta, "robot_urdf_path")
    if urdf_path:
        stem = os.path.splitext(os.path.basename(urdf_path))[0]
        scene_name = _URDF_STEM_TO_SCENE.get(stem)
        if scene_name is not None:
            candidate = os.path.join(_scenes_dir, scene_name)
            if not os.path.exists(candidate):
                print(
                    f"ONNX robot_urdf_path stem '{stem}' maps to render scene '{scene_name}', "
                    f"but it is missing at {candidate}. Pass --task.headless-xml to override; "
                    f"refusing to render the wrong hand mesh on a '{stem}' policy.",
                    file=sys.stderr,
                )
                sys.exit(1)
            print(f"Render scene auto-derived from ONNX robot_urdf_path (stem '{stem}'): {scene_name}")
            return candidate
        print(
            f"No render scene mapped for ONNX URDF stem '{stem}'; using the default "
            f"rubber-hand scene. Pass --task.headless-xml to override.",
            file=sys.stderr,
        )

    xml_path = os.path.join(_scenes_dir, "scene_g1_29dof_wbt_plane.xml")
    if not os.path.exists(xml_path):
        print(f"Cannot find holosoma XML at {xml_path}", file=sys.stderr)
        sys.exit(1)
    return xml_path


def extract_model_label(onnx_path: str, exp_config: dict | None, meta_map: dict | None = None) -> str:
    ckpt_name = os.path.splitext(os.path.basename(onnx_path))[0]
    step_str = ckpt_name.replace("model_", "step ") if ckpt_name.startswith("model_") else ckpt_name
    if exp_config is not None:
        run_name = exp_config.get("logger", {}).get("name", "")
    else:
        run_name = (meta_map or {}).get("run_path", "")
    return f"{run_name} | {step_str}" if run_name else step_str


def build_metrics_output(metrics, motion_label, model_label):
    out = {
        "motion_name": motion_label,
        "model_label": model_label or "",
        "ref_motion_length": metrics.get("ref_motion_length", 0),
        "survival_frames": metrics.get("survival_frames", 0),
        "survived": metrics.get("survived", True),
        "summary": metrics.get("summary", {}),
        "per_frame_joint_angle_error_mean_deg": metrics.get("joint_angle_error_mean_deg", []),
        "per_frame_body_pos_error_mean_cm": metrics.get("body_pos_error_mean_cm", []),
        "per_frame_mpjpe_torso_mean_cm": metrics.get("mpjpe_torso_mean_cm", []),
        "per_frame_hand_err_torso_mean_cm": metrics.get("hand_err_torso_mean_cm", []),
        "per_frame_root_pos_error_cm": metrics.get("root_pos_error_cm", []),
        # Previously DROPPED here — the per-body arrays the MetricsLogger already
        # computes. Needed for per-hand (wrist_yaw) tracking error.
        "per_frame_body_pos_error_per_body_cm": metrics.get("body_pos_error_per_body_cm", []),
        "body_names_matched": metrics.get("body_names_matched", []),
    }
    # UniFP force-tracking block (present only when body_quat_w was available).
    if "force_summary" in metrics:
        out["force_summary"] = metrics["force_summary"]
        out["force_cmd_local_N"] = metrics.get("force_cmd_local_N", [])
        out["stiffness_K"] = metrics.get("stiffness_K", FORCE_STIFFNESS_K)
        out["force_settle_frames"] = metrics.get("force_settle_frames", FORCE_SETTLE_FRAMES)
        out["force_hand_names"] = metrics.get("force_hand_names", list(FORCE_HANDS))
        out["per_frame_force_hand_drift_cm"] = metrics.get("per_frame_force_hand_drift_cm", [])
        out["per_frame_force_target_err_cm"] = metrics.get("per_frame_force_target_err_cm", [])
        out["per_frame_force_realized_pelvis_N"] = metrics.get("per_frame_force_realized_pelvis_N", [])
    return out


# ---------------------------------------------------------------------------
# Render reference motion (left side of eval video)
# ---------------------------------------------------------------------------
def render_reference(
    npz_path: str,
    xml_path: str,
    max_frames: int,
    vr_body_names=None,
    vr_offsets=None,
    vr_viz_offsets=None,
    mm_body_names=None,
    fixed_camera=False,
    cam_distance=0.0,
    terrain_obj: str | None = None,
) -> list[np.ndarray]:
    with np.load(npz_path, allow_pickle=True) as d:
        joint_pos = d["joint_pos"]
        # For terrain-replayed 36-col NPZs, body_pos_w / body_quat_w carry the
        # obstacle-offset root pose while joint_pos[:, :3] / [:, 3:7] are the
        # pre-offset values — use body_pos_w when present (mirrors the policy-side
        # init in _setup_headless_sim so reference and policy spawn on the same
        # terrain slot). For 30-col NPZs (legacy TML format) body_pos_w is required.
        bp = d["body_pos_w"] if "body_pos_w" in d.files else None
        bq = d["body_quat_w"] if "body_quat_w" in d.files else None
    T = joint_pos.shape[0] if max_frames <= 0 else min(joint_pos.shape[0], max_frames)
    ncols = joint_pos.shape[1]

    model = _load_model(xml_path, terrain_obj=terrain_obj)
    data = mujoco.MjData(model)
    renderer, cam = _make_renderer(model, fixed_camera=fixed_camera, cam_distance=cam_distance)
    use_body = bp is not None and bq is not None

    if use_body:
        assert bp is not None and bq is not None  # narrowed by use_body; for the type checker
        w0, x0, y0, z0 = bq[0, 0]
    elif ncols == 36:
        w0, x0, y0, z0 = joint_pos[0, 3:7]
    else:
        raise ValueError("NPZ must contain body_pos_w/body_quat_w or 36-col joint_pos for root pose")
    init_yaw = np.arctan2(2 * (w0 * z0 + x0 * y0), 1 - 2 * (y0**2 + z0**2))
    draw_vr = vr_body_names is not None

    frames = []
    for t in range(T):
        if use_body:
            assert bp is not None and bq is not None  # narrowed by use_body; for the type checker
            data.qpos[0:3] = bp[t, 0]
            data.qpos[3:7] = bq[t, 0]
            data.qpos[7:36] = joint_pos[t, 7:36] if ncols == 36 else joint_pos[t][ISAAC_TO_MUJOCO]
        else:
            data.qpos[0:3] = joint_pos[t, :3]
            data.qpos[3:7] = joint_pos[t, 3:7]
            data.qpos[7:36] = joint_pos[t, 7:36]
        mujoco.mj_forward(model, data)
        if not fixed_camera:
            cam.azimuth = BASE_AZIMUTH + np.degrees(_yaw(data.qpos) - init_yaw)
        renderer.update_scene(data, camera=cam)
        if draw_vr:
            marker_offsets = vr_viz_offsets if vr_viz_offsets is not None else vr_offsets
            positions, rotations = _get_vr_positions_and_rotations_from_mujoco(
                model, data, vr_body_names, marker_offsets
            )
            _add_vr_axis_markers(renderer._scene, positions, rotations)
        if mm_body_names:
            _add_mm_body_markers(renderer._scene, model, data, mm_body_names)
        frames.append(renderer.render().copy())

    renderer.close()
    return frames


# ---------------------------------------------------------------------------
# MetricsLogger — reusable per-frame metrics collection + summary
# ---------------------------------------------------------------------------
class MetricsLogger:
    """Collects per-frame tracking metrics and computes summary statistics.

    Usage::

        ml = MetricsLogger(ref_npz, mj_model)
        for t in range(n_frames):
            mid = min(t, motion_length - 1)
            ml.log_frame(mid, mj_data)
            if mj_data.qpos[2] < 0.3:
                ml.mark_fallen()
                break
        metrics = ml.finalize(n_logged_frames)
    """

    def __init__(self, ref_npz, mj_model):
        ncols = ref_npz["joint_pos"].shape[1]
        if ncols == 36:
            self._ref_dof_pos = ref_npz["joint_pos"][:, 7:]
            self._ref_root_pos = ref_npz["joint_pos"][:, :3]
        else:
            self._ref_dof_pos = ref_npz["joint_pos"][:, ISAAC_TO_MUJOCO]
            self._ref_root_pos = ref_npz["body_pos_w"][:, 0]

        self._motion_length = ref_npz["joint_pos"].shape[0]

        # Body mapping: NPZ body index → MuJoCo body id
        self._body_map = []
        self._ref_body_pos_w = ref_npz.get("body_pos_w", None)
        _ref_body_names_raw = ref_npz.get("body_names", None)
        if self._ref_body_pos_w is not None and _ref_body_names_raw is not None:
            bnames = [b.decode() if isinstance(b, bytes) else str(b) for b in _ref_body_names_raw.tolist()]
            for bi, bn in enumerate(bnames):
                mjid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, bn)
                if mjid >= 0:
                    self._body_map.append((bi, mjid, bn))

        # --- Torso-anchored tracking metrics (SMPL/AMASS-style, root-drift-free) -
        # Anchor body for the torso-frame metrics: positions are expressed as the
        # per-body vector RELATIVE to torso_link (world-axes translation subtract),
        # so global root TRANSLATION drift cancels and only relative limb tracking
        # remains. This is the literal Δp = (p_body - p_torso)_ref - (p_body -
        # p_torso)_policy of the spec; the torso row is 0 by construction. Contrast
        # with the existing world-frame body_pos_error (NO subtraction), whose
        # surplus over the torso metric IS the global drift (the "does the motion
        # in place" symptom shows up as world≫torso). N/A (empty) if torso_link is
        # absent from either the NPZ body set or the MuJoCo model.
        self._torso_npz_idx = -1
        self._torso_mj_id = -1
        if self._ref_body_pos_w is not None and _ref_body_names_raw is not None:
            _bn = [b.decode() if isinstance(b, bytes) else str(b) for b in _ref_body_names_raw.tolist()]
            if "torso_link" in _bn:
                self._torso_npz_idx = _bn.index("torso_link")
                self._torso_mj_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "torso_link")

        # Per-frame accumulators
        self._joint_err_per_joint = []
        self._joint_err_mean = []
        self._body_err_per_body = []
        self._body_err_mean = []
        self._root_err = []
        self._mpjpe_torso_per_frame = []  # per-frame mean over bodies, torso-anchored, cm
        self._hand_err_torso_per_frame = []  # per-frame 0.5*(L+R) hand-point torso-anchored, cm
        self._survived = True

        # --- UniFP force-tracking state (g1-wbt-2pt-force benchmark) ----------
        # Resolve once: per-hand NPZ body index + MuJoCo body id for both wrists,
        # plus the reference body quats (for FK of the rubber_hand offset).
        # F_cmd is read from the SAME env var the policy obs uses
        # (HOLOSOMA_FORCE_CMD_LOCAL = raw N, pelvis frame, [L_xyz, R_xyz]); we do
        # NOT reuse policy._force_cmd_local_scaled (that one is pre-×0.04).
        self._ref_body_quat_w = ref_npz.get("body_quat_w", None)
        self._force_hand_ids = []  # list of (npz_idx, mj_id, offset_b) for L,R
        if self._ref_body_pos_w is not None and self._ref_body_quat_w is not None:
            _names = (
                [b.decode() if isinstance(b, bytes) else str(b) for b in _ref_body_names_raw.tolist()]
                if _ref_body_names_raw is not None
                else []
            )
            for hi, hname in enumerate(FORCE_HANDS):
                npz_idx = _names.index(hname) if hname in _names else -1
                mj_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, hname)
                if npz_idx >= 0 and mj_id >= 0:
                    self._force_hand_ids.append((npz_idx, mj_id, FORCE_HAND_OFFSETS_B[hi]))
                else:
                    self._force_hand_ids.append(None)
        self._force_enabled = self._ref_body_quat_w is not None and any(h is not None for h in self._force_hand_ids)

        # F_cmd in PELVIS frame, raw N, shape (2,3) = [left, right]. Constant
        # over the rollout (the eval injects a step via HOLOSOMA_FORCE_CMD_LOCAL).
        self._force_cmd_pelvis = self._parse_force_cmd_env()

        # Per-frame force accumulators (per hand). Displacement & residual are
        # measured in the ROOT(PELVIS) frame so root yaw/xy drift cancels and
        # F_cmd (pelvis frame) compares directly — matches the training reward,
        # which scores the yaw-relocated relative position (body_pos_relative_w).
        self._force_drift_cm = []  # [(L,R)] raw ||hand-ref|| in root frame, cm
        self._force_target_err_cm = []  # [(L,R)] ||hand-(ref+Fcmd/K)|| root frame, cm
        self._force_realized_pelvis = []  # [(L_xyz,R_xyz)] F_hat=K*(hand-ref) pelvis frame, N

    @staticmethod
    def _parse_force_cmd_env():
        """Parse HOLOSOMA_FORCE_CMD_LOCAL into a (2,3) raw-N pelvis-frame array.

        Format: "L_fx,L_fy,L_fz,R_fx,R_fy,R_fz". Empty/unset → zeros (the F=0
        baseline). This is the SAME env var the policy reads, so there is a
        single source of truth for the force condition.
        """
        raw = os.environ.get("HOLOSOMA_FORCE_CMD_LOCAL", "")
        if not raw:
            return np.zeros((2, 3), dtype=np.float64)
        vals = [float(x) for x in raw.split(",")]
        if len(vals) != 6:
            raise ValueError(
                f"HOLOSOMA_FORCE_CMD_LOCAL must be 6 comma-separated Newtons "
                f"[L_fx,L_fy,L_fz,R_fx,R_fy,R_fz], got {len(vals)}: {raw!r}"
            )
        return np.array(vals, dtype=np.float64).reshape(2, 3)

    @property
    def motion_length(self):
        return self._motion_length

    def log_frame(self, mid, mj_data):
        """Record metrics for one frame. Call BEFORE policy.policy_action()."""
        q_mujoco = mj_data.qpos[7:36]
        je_deg = np.degrees(np.abs(q_mujoco - self._ref_dof_pos[mid]))
        self._joint_err_per_joint.append(je_deg.tolist())
        self._joint_err_mean.append(float(je_deg.mean()))
        self._root_err.append(float(np.linalg.norm(mj_data.qpos[:3] - self._ref_root_pos[mid]) * 100.0))
        if self._body_map and self._ref_body_pos_w is not None:
            errs = [
                float(np.linalg.norm(self._ref_body_pos_w[mid, ni] - mj_data.xpos[mi]) * 100.0)
                for ni, mi, _ in self._body_map
            ]
            self._body_err_per_body.append(errs)
            self._body_err_mean.append(float(np.mean(errs)))

            # Torso-anchored MPJPE (metric #3) + torso-anchored hand-point error
            # (metric #2). Both subtract the torso_link world offset from ref and
            # policy alike → Δp = (p_body-p_torso)_ref - (p_body-p_torso)_policy,
            # i.e. world-frame error vector of the body MINUS that of torso, so
            # bulk root translation cancels (the anchor/torso row is 0).
            if self._torso_npz_idx >= 0 and self._torso_mj_id >= 0:
                torso_err_vec = self._ref_body_pos_w[mid, self._torso_npz_idx] - mj_data.xpos[self._torso_mj_id]
                terrs = [
                    float(np.linalg.norm((self._ref_body_pos_w[mid, ni] - mj_data.xpos[mi]) - torso_err_vec) * 100.0)
                    for ni, mi, _ in self._body_map
                ]
                self._mpjpe_torso_per_frame.append(float(np.mean(terrs)))

                # Hand metric: the FORCE-DEPLOYMENT contact point (FK on wrist_yaw
                # with the rubber_hand/yam offset_b), NOT the wrist_yaw_link origin
                # — same point the force metrics use. Mean of L/R torso-anchored L2.
                if self._ref_body_quat_w is not None and self._force_hand_ids:
                    hand_errs = []
                    for hand in self._force_hand_ids:
                        if hand is None:
                            continue
                        npz_idx, mj_id, offset_b = hand
                        pol_hand_w = mj_data.xpos[mj_id] + mj_data.xmat[mj_id].reshape(3, 3) @ offset_b
                        ref_hand_w = (
                            self._ref_body_pos_w[mid, npz_idx]
                            + _quat_to_mat_wxyz(self._ref_body_quat_w[mid, npz_idx]) @ offset_b
                        )
                        dp = (ref_hand_w - pol_hand_w) - torso_err_vec
                        hand_errs.append(float(np.linalg.norm(dp) * 100.0))
                    if hand_errs:
                        self._hand_err_torso_per_frame.append(float(np.mean(hand_errs)))

        if self._force_enabled:
            self._log_force_frame(mid, mj_data)

    def _log_force_frame(self, mid, mj_data):
        """Per-frame UniFP force-realization metrics, in the ROOT(PELVIS) frame.

        For each hand h the rubber_hand world position is reconstructed by FK on
        wrist_yaw: ``p_w = wrist_pos_w + R(wrist_quat_w) @ offset_b``. Both the
        policy (live mj_data) and the reference (NPZ body_pos_w/body_quat_w) use
        the SAME offset, so the offset cancels in the residual. Positions are
        then expressed RELATIVE to each side's own root (pelvis) so root yaw/xy
        drift cancels and the residual matches the training reward's
        yaw-relocated relative-position term.

        Realized force (Hookean back-out, pelvis frame): F_hat = K * (hand_pol -
        hand_ref). Force-target residual: ||hand_pol - (hand_ref + F_cmd/K)||,
        which is exactly the Eq.2 position residual (×K = ‖F_cmd − F_hat‖).
        """
        # Policy root (pelvis) world pose from the free joint.
        pol_root_p = mj_data.qpos[0:3]
        pol_root_R = _quat_to_mat_wxyz(mj_data.qpos[3:7])
        # Reference root (pelvis) world pose from the NPZ (body 0 = pelvis).
        ref_root_p = self._ref_body_pos_w[mid, 0]
        ref_root_R = _quat_to_mat_wxyz(self._ref_body_quat_w[mid, 0])

        drift_pair, target_err_pair, realized_pair = [], [], []
        for hi, hand in enumerate(self._force_hand_ids):
            if hand is None:
                drift_pair.append(float("nan"))
                target_err_pair.append(float("nan"))
                realized_pair.append([float("nan")] * 3)
                continue
            npz_idx, mj_id, offset_b = hand
            # Policy rubber_hand world pos (FK on live wrist_yaw).
            pol_hand_w = mj_data.xpos[mj_id] + mj_data.xmat[mj_id].reshape(3, 3) @ offset_b
            # Reference rubber_hand world pos (FK on NPZ wrist_yaw).
            ref_hand_w = (
                self._ref_body_pos_w[mid, npz_idx] + _quat_to_mat_wxyz(self._ref_body_quat_w[mid, npz_idx]) @ offset_b
            )
            # Express each in its own root(pelvis) frame → drift-immune, pelvis-frame.
            pol_hand_rel = pol_root_R.T @ (pol_hand_w - pol_root_p)
            ref_hand_rel = ref_root_R.T @ (ref_hand_w - ref_root_p)
            disp = pol_hand_rel - ref_hand_rel  # realized displacement, pelvis frame (m)
            f_hat = FORCE_STIFFNESS_K * disp  # realized force, pelvis frame (N)
            f_cmd = self._force_cmd_pelvis[hi]  # commanded force, pelvis frame (N)
            target_shift = f_cmd / FORCE_STIFFNESS_K  # intended displacement (m)
            drift_pair.append(float(np.linalg.norm(disp) * 100.0))
            target_err_pair.append(float(np.linalg.norm(disp - target_shift) * 100.0))
            realized_pair.append([float(v) for v in f_hat])
        self._force_drift_cm.append(drift_pair)
        self._force_target_err_cm.append(target_err_pair)
        self._force_realized_pelvis.append(realized_pair)

    def mark_fallen(self):
        self._survived = False

    def finalize(self, n_frames):
        """Compute summary and return the full metrics dict."""
        metrics = {
            "joint_angle_error_per_joint_deg": self._joint_err_per_joint,
            "joint_angle_error_mean_deg": self._joint_err_mean,
            "body_pos_error_per_body_cm": self._body_err_per_body,
            "body_pos_error_mean_cm": self._body_err_mean,
            "mpjpe_torso_mean_cm": self._mpjpe_torso_per_frame,
            "hand_err_torso_mean_cm": self._hand_err_torso_per_frame,
            "root_pos_error_cm": self._root_err,
            "body_names_matched": [bm[2] for bm in self._body_map],
            "ref_motion_length": self._motion_length,
            "survived": self._survived,
            "survival_frames": n_frames,
        }
        if self._joint_err_mean:
            bpe = self._body_err_mean
            mpt = self._mpjpe_torso_per_frame
            het = self._hand_err_torso_per_frame
            metrics["summary"] = {
                "mean_joint_angle_error_deg": float(np.mean(self._joint_err_mean)),
                "max_joint_angle_error_deg": float(np.max(self._joint_err_mean)),
                "mean_body_pos_error_cm": float(np.mean(bpe)) if bpe else 0.0,
                "max_body_pos_error_cm": float(np.max(bpe)) if bpe else 0.0,
                # Torso-anchored (root-drift-free) tracking metrics. 0.0 when
                # torso_link was unavailable → the dashboard treats 0.0 as N/A.
                "mean_mpjpe_torso_cm": float(np.mean(mpt)) if mpt else 0.0,
                "mean_hand_err_torso_cm": float(np.mean(het)) if het else 0.0,
                "mean_root_pos_error_cm": float(np.mean(self._root_err)),
                "survival_ratio": n_frames / self._motion_length if self._motion_length > 0 else 1.0,
                "survival_frames": n_frames,
                "ref_motion_length": self._motion_length,
            }

        if self._force_enabled and self._force_drift_cm:
            metrics.update(self._finalize_force())
        return metrics

    def _finalize_force(self):
        """Force-tracking per-frame arrays + settled-window summary (per hand).

        Aggregation drops the first FORCE_SETTLE_FRAMES (constant-step onset
        transient) then takes the per-hand MEDIAN over the remaining frames.
        e_F (N) = K × the position residual (cm→m); reported alongside r_pos.
        """
        drift = np.asarray(self._force_drift_cm, dtype=np.float64)  # (T,2)
        terr = np.asarray(self._force_target_err_cm, dtype=np.float64)  # (T,2)
        realized = np.asarray(self._force_realized_pelvis, dtype=np.float64)  # (T,2,3)
        out = {
            "force_cmd_local_N": self._force_cmd_pelvis.reshape(-1).tolist(),  # [L_xyz,R_xyz]
            "stiffness_K": FORCE_STIFFNESS_K,
            "force_settle_frames": FORCE_SETTLE_FRAMES,
            "force_hand_names": list(FORCE_HANDS),
            "per_frame_force_hand_drift_cm": drift.tolist(),  # raw |hand-ref|, pelvis frame
            "per_frame_force_target_err_cm": terr.tolist(),  # |hand-(ref+Fcmd/K)|
            "per_frame_force_realized_pelvis_N": realized.tolist(),  # F_hat per hand per axis
        }
        s = max(0, min(FORCE_SETTLE_FRAMES, drift.shape[0] - 1))
        ds, ts, rs = drift[s:], terr[s:], realized[s:]
        summ = {}
        for hi, hname in enumerate(FORCE_HANDS):
            tag = "left" if hi == 0 else "right"
            d_h, t_h, r_h = ds[:, hi], ts[:, hi], rs[:, hi, :]
            if np.all(np.isnan(d_h)):
                continue
            f_cmd = self._force_cmd_pelvis[hi]
            # e_F (N) = ||F_cmd - F_hat||, per frame, then median over settled window.
            e_F = np.linalg.norm(f_cmd[None, :] - r_h, axis=-1)
            summ[f"force_drift_median_cm_{tag}"] = float(np.nanmedian(d_h))
            summ[f"force_target_err_median_cm_{tag}"] = float(np.nanmedian(t_h))
            summ[f"force_realization_err_median_N_{tag}"] = float(np.nanmedian(e_F))
            summ[f"force_realized_median_pelvis_N_{tag}"] = [float(v) for v in np.nanmedian(r_h, axis=0)]
            summ[f"force_cmd_pelvis_N_{tag}"] = [float(v) for v in f_cmd]
            summ[f"force_cmd_mag_N_{tag}"] = float(np.linalg.norm(f_cmd))
            summ[f"expected_drift_cm_{tag}"] = float(np.linalg.norm(f_cmd) / FORCE_STIFFNESS_K * 100.0)
        out["force_summary"] = summ
        return out
