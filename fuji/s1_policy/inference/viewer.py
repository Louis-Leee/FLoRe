"""Model-agnostic viser frontend for native-layout HL-v2 rollouts.

The viewer renders Cartesian trajectories directly and does not import MuSe/UVA,
mink, or MuJoCo. To make a rollout read like the MODE=1pt collection viz (see
``scripts/collect_data.sh``) and the zarr replay, it borrows the three yam gripper
meshes from UMI Collection's ``gripper_only_view`` and draws the SAME three axis
triads — TCP, tracker (the physical VIVE puck), and wrist_yaw_link — around the
trajectory. The mesh overlay is best-effort: if that import or the URDF is
unavailable the viewer degrades to trajectory lines + triads and prints why. The
headless ``--self-test`` path never touches any of this.

Two display decisions, both owned by :mod:`fuji.s1_policy.inference.display`:

* **+Z up.** The recorded pose is the physical puck held in the hand, whose +z points
  at the FLOOR, so a faithful render hangs the gripper upside down. Every drawn pose is
  therefore re-based onto an upright display home, sharing ONE reference across the target
  chunk, the prediction and the recorded tail. Positions are **TRANSLATED, never rotated**
  (``p_home + (p − p_ref)``) and orientations get the right-multiplied world delta
  (``R_home @ (R_cur @ R_ref^T)``) — the ``gripper_only_view.home_anchor`` convention the
  collection viz and the zarr replay use, so world DIRECTIONS are preserved and the grid still
  means "the floor". Because the translation is rigid on DIFFERENCES, the position/rotation
  errors the eye sees are exactly the ones the Metrics panel reports. What it does NOT preserve
  is each path's angle to its own gripper axis (the earlier "rigid" variant that rotated
  positions by ``R_ref^T`` did, but it tilted the whole scene into the puck's floor-ward frame —
  the wrong-coordinate-system bug this convention exists to avoid), nor ABSOLUTE world position
  and orientation. The "3D frame" readout always states which frame is live; toggle the rebase
  off for a strict raw-frame check.
* **The trajectory datum is the TCP.** The cluster is placed from the DATASET's own
  ``T_tracker_tcp`` attr rather than a hardcoded lever, because that extrinsic differs
  between datasets (corrected clean94 stores the fingertip TCP; older ones store a
  different point). The exact residual is reported in the GUI instead of being hidden.

Force is drawn as an ARROW at the TCP, and each arrow **inherits the colour of the trajectory
it belongs to** — BLUE for the masked GT chunk, ORANGE for the prediction, GREEN for the
recorded frame during playback — so the scene's existing blue/orange/green vocabulary answers
"whose force is this?" without a second legend. The arrow nodes are CHILDREN of the same frames
the trajectory point drives, so their segments are TCP-LOCAL and the sensor->TCP relabel
(:mod:`fuji.utils.force_frame`) is the only rotation involved; an arrow therefore cannot drift
from the gripper it belongs to, in either display frame.

Everything force-shaped on this page — arrows AND charts — is the **COMMAND** ``F^cmd``, i.e.
the model's predicted ``F_meas`` NEGATED (:data:`fuji.utils.force_frame.CMD_FROM_MEAS_SIGN`),
because that is the direction the arm would push if this chunk were executed with the force
wire on. The GUI states both conventions (relabel + sign) rather than leaving them to be
inferred, and the flip is applied in exactly one place per surface
(:func:`~fuji.s1_policy.inference.display.force_arrow_local_segment` for the arrows,
``_cmd_chart`` below for the charts) so it cannot land twice. It is a sign shared by target and
prediction, so it does not move any error metric.
"""

from __future__ import annotations

import html
import math
import os
import queue
import socket
import sys
import threading
import time
from dataclasses import dataclass
from numbers import Integral, Real

import numpy as np

from fuji.s1_policy.data.layout import ACTION_HORIZON, tracker_roles_required
from fuji.s1_policy.data.transforms import GRIPPER_WIDTH_REF_M
from fuji.s1_policy.inference.common import RolloutResult
from fuji.s1_policy.inference.display import (
    FORCE_ARROW_SCALE_M_PER_N,
    GRIPPER_ROLES,
    TCP_RESIDUAL_TOL_MM,
    TRIAD_OFFSETS_WRIST,
    force_arrow_local_segment,
    home_anchor_positions,
    home_anchor_rotmats,
    resolve_role_geometry,
    world_rotmat_by_role,
)
from fuji.s1_policy.inference import ready_pose
from fuji.utils.force_frame import CMD_FROM_MEAS_NOTE, TCP_FROM_SENSOR_NOTE, cmd_from_meas_wrench
from fuji.utils.rot6d import rot6d_to_rotmat, rotmat_to_quat_xyzw

_TARGET_BLUE = np.array([55, 135, 245], dtype=np.uint8)
_PREDICTION_ORANGE = np.array([245, 145, 35], dtype=np.uint8)
# The recorded rest-of-episode EE path from the sampled frame onward (same display
# frame as the blue target line; its first point coincides with the blue step 0).
_EPISODE_GREEN = np.array([46, 204, 113], dtype=np.uint8)
_AXIS_COLORS = ("#c0392b", "#1f9d55", "#2f6fd0")

# The three collection-viz triads, drawn at the selected action step on BOTH the
# prediction and the target. Sizes follow UMI Collection's CONVENTION (TCP slightly larger,
# so it reads as the datum the other two are measured from) but are scaled down here, and
# not in the same ratio, because this scene also carries metre-scale trajectory lines the
# upstream 0.08/0.07 m axes would swamp.
_TRIAD_STYLE = {
    "tcp": {"axes_length": 0.050, "axes_radius": 0.0022, "origin_radius": 0.006},
    "tracker": {"axes_length": 0.042, "axes_radius": 0.0018, "origin_radius": 0.005},
    "wrist_yaw_link": {"axes_length": 0.042, "axes_radius": 0.0018, "origin_radius": 0.005},
}
# The target's copy of the same triads, drawn thinner so an overlapping pred/target pair
# stays readable (the prediction is the thing under test, so it keeps the bolder axes).
_TARGET_TRIAD_SCALE = 0.8
# Pose triads ALONG the two action-chunk lines (GT + prediction), every N steps, off by default.
# Deliberately smaller than the action-step triads above so the selected step stays the loudest
# frame in the scene; the length is a slider because a 5 cm hand stroke and a 60 cm reach want
# different axes. Both series get the same stride so a GT triad and its predicted twin sit at
# the same step and the rotation error is readable as the angle between the two.
_PATH_AXES_DEFAULT_STRIDE = 5
_PATH_AXES_DEFAULT_LENGTH_MM = 30.0
_PATH_AXES_RADIUS_RATIO = 0.045  # axes_radius / axes_length, the action-step triads' proportion

# Force arrows. Each arrow REUSES THE COLOUR OF THE TRAJECTORY IT BELONGS TO — blue for the
# masked GT chunk, orange for the prediction, green for the recorded episode during playback —
# rather than introducing a third "force" hue. That is the point: the scene already teaches
# blue=GT / orange=pred / green=recorded through the three polylines, so an arrow's colour
# answers "whose force is this?" with no extra legend, and a blue arrow beside an orange one
# reads as the same comparison the two lines already make. Kept as plain int tuples because
# viser's add_arrows wants a colour, not a uint8 array.
_FORCE_TARGET_COLOR = tuple(int(c) for c in _TARGET_BLUE)
_FORCE_PREDICTION_COLOR = tuple(int(c) for c in _PREDICTION_ORANGE)
_FORCE_PLAYBACK_COLOR = tuple(int(c) for c in _EPISODE_GREEN)
_FORCE_ARROW_STYLE = {"shaft_radius": 0.0035, "head_radius": 0.009, "head_length": 0.018}
# Radius still differs slightly, reusing the TRIADS' target scale rather than inventing a
# second number: when a well-predicted pair nearly coincides, the thicker arrow occludes the
# thinner one in 3D, and the bolder one should be the prediction — the thing under test.
_FORCE_ARROW_TARGET_SCALE = _TARGET_TRIAD_SCALE


@dataclass(frozen=True)
class _RolloutRequest:
    generation: int
    sample_index: int
    seed: int
    num_steps: int


@dataclass
class _GripperOverlay:
    """The three yam gripper meshes per role, borrowed from UMI Collection's
    ``gripper_only_view`` (the MODE=1pt collection-viz geometry). Each piece already
    carries its closed/open wrist-relative endpoints, so the jaw is LERPed straight
    from the predicted open fraction — the stroke length itself is not needed here."""

    geoms: dict            # role -> gripper_only_view.GripperGeom


def _find_umi_viz_dir() -> str | None:
    """Locate UMI Collection's ``viz/`` directory (holds gripper_only_view.py).

    Walks up from this file to the repository root and checks the vendored
    ``third_party/UMI_collection/viz`` path. Returns ``None`` if absent so the
    caller can degrade to a mesh-free scene instead of raising.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    root = here
    for _ in range(6):
        candidate = os.path.join(root, "third_party", "UMI_collection", "viz")
        if os.path.isfile(os.path.join(candidate, "gripper_only_view.py")):
            return candidate
        parent = os.path.dirname(root)
        if parent == root:
            break
        root = parent
    return None


def _load_gripper_overlay(roles: tuple[str, ...]) -> tuple[_GripperOverlay | None, str]:
    """Best-effort load of the yam gripper meshes for the given layout roles.

    Only "left"/"right" roles have a physical gripper (the head puck does not), so
    other roles are skipped. Returns ``(overlay, message)``; ``overlay`` is ``None``
    when the UMI viz package, its URDF, or ``yourdfpy``/``trimesh`` are unavailable,
    with ``message`` explaining why (printed once, never raised — the viewer still
    works as trajectory lines + frames)."""
    viz_dir = _find_umi_viz_dir()
    if viz_dir is None:
        return None, "UMI_collection/viz not found; drawing frames only (no gripper mesh)"
    gripper_roles = tuple(role for role in roles if role in ("left", "right"))
    if not gripper_roles:
        return None, "layout has no left/right gripper role; drawing frames only"
    # append, not insert(0): gripper_only_view is a unique name, and the UMI viz dir
    # also holds generically-named modules (collection_viz, test_*), which must not
    # shadow same-named top-level imports for the rest of this process.
    if viz_dir not in sys.path:
        sys.path.append(viz_dir)
    try:
        import gripper_only_view as gov  # noqa: E402  (path injected just above)

        shared_urdf = gov._load_urdf(gov._YAM_URDF_DEFAULT)
        geoms = {role: gov.build_gripper_geom(role, urdf=shared_urdf) for role in gripper_roles}
    except Exception as exc:  # noqa: BLE001 - overlay is optional; never kill the viewer
        return None, f"gripper mesh overlay unavailable ({exc!r}); drawing frames only"
    return _GripperOverlay(geoms=geoms), (
        f"gripper mesh overlay active for roles {', '.join(gripper_roles)}"
    )


def _wxyz_from_rotmat(rotmat: np.ndarray) -> tuple[float, float, float, float]:
    """(3, 3) rotation matrix -> wxyz quaternion, degrading to identity if degenerate.

    Shares :func:`fuji.utils.rot6d.rotmat_to_quat_xyzw` (the repo's single rotation
    source) rather than re-deriving Shepperd's method here."""
    matrix = np.asarray(rotmat, dtype=np.float64)
    determinant = float(np.linalg.det(matrix))
    if not np.isfinite(matrix).all() or not np.isclose(determinant, 1.0, atol=1e-5):
        return 1.0, 0.0, 0.0, 0.0
    try:
        xyzw = rotmat_to_quat_xyzw(matrix)
    except ValueError:
        return 1.0, 0.0, 0.0, 0.0
    return float(xyzw[3]), float(xyzw[0]), float(xyzw[1]), float(xyzw[2])


def _segments(points: np.ndarray, valid: np.ndarray | None = None) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    if points.shape[0] < 2:
        return np.empty((0, 2, 3), dtype=np.float32)
    keep = np.ones(points.shape[0] - 1, dtype=bool)
    if valid is not None:
        valid = np.asarray(valid, dtype=bool)
        keep &= valid[:-1] & valid[1:]
    return np.stack((points[:-1][keep], points[1:][keep]), axis=1)


def _path_axes_steps(horizon: int, stride: int, valid: np.ndarray | None = None) -> list[int]:
    """Steps that get a pose triad along a chunk line: ``0, stride, 2*stride, ...`` plus the LAST
    step (so the chunk's end orientation is always drawn, whatever the stride), minus any step
    whose pose is masked invalid (a masked GT step is not a measurement, so it gets no axes —
    the same rule that breaks the blue line there). ``stride < 1`` is read as 1."""
    horizon = int(horizon)
    if horizon <= 0:
        return []
    stride = max(1, int(stride))
    steps = list(range(0, horizon, stride))
    if steps[-1] != horizon - 1:
        steps.append(horizon - 1)
    if valid is not None:
        valid = np.asarray(valid, dtype=bool).reshape(-1)
        steps = [k for k in steps if bool(valid[k])]
    return steps


def _checked_sample_index(sample_index: int, sample_count: int) -> int:
    index = int(sample_index)
    if not 0 <= index < sample_count:
        raise IndexError(f"sample index {index} is outside [0, {sample_count - 1}]")
    return index


def _checked_port(port: int) -> int:
    port = int(port)
    if not 1 <= port <= 65534:
        raise ValueError(f"port must be in [1, 65534], got {port}")
    return port


def _require_port_available(host: str, port: int) -> None:
    try:
        addresses = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    except OSError as exc:
        raise RuntimeError(f"cannot resolve viewer host {host!r}: {exc}") from exc
    checked = set()
    for family, sock_type, protocol, _, address in addresses:
        key = family, sock_type, protocol, address
        if key in checked:
            continue
        checked.add(key)
        try:
            with socket.socket(family, sock_type, protocol) as probe:
                probe.bind(address)
        except OSError as exc:
            raise RuntimeError(f"viser port {port} is unavailable on {host}: {exc}") from exc


def _checked_gui_integer(value, label: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be an integer, got {value!r}")
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value.strip())
        except ValueError:
            pass
    elif isinstance(value, Real) and np.isfinite(value) and float(value).is_integer():
        return int(value)
    raise ValueError(f"{label} must be an integer, got {value!r}")


def _draw_path(server, name: str, points: np.ndarray, color: np.ndarray, valid=None):
    segments = _segments(points, valid)
    if not segments.size:
        return server.scene.add_line_segments(
            name,
            points=np.zeros((1, 2, 3), dtype=np.float32),
            colors=np.zeros((1, 2, 3), dtype=np.uint8),
            line_width=4.0,
            visible=False,
        )
    colors = np.broadcast_to(color, (segments.shape[0], 2, 3)).copy()
    return server.scene.add_line_segments(name, points=segments, colors=colors, line_width=4.0)


def _wxyz(rot6d: np.ndarray) -> tuple[float, float, float, float]:
    return _wxyz_from_rotmat(rot6d_to_rotmat(np.asarray(rot6d, dtype=np.float64)))


def _draw_force_arrow(
    server,
    name: str,
    force_sensor,
    *,
    color,
    scale_m_per_n: float,
    radius_scale: float = 1.0,
    visible: bool = True,
) -> float:
    """Add/replace ONE force arrow, in the coordinates of whatever frame ``name`` sits under.

    ``force_sensor`` is the raw ``(3,)`` bota-sensor triple as MEASURED/predicted; both the
    sensor->TCP relabel and the measured->commanded sign flip happen inside
    :func:`force_arrow_local_segment`, so what is drawn is ``F^cmd``. The node MUST be a child
    of a frame already driven to the trajectory point at the TCP orientation — then the returned
    segment's local coordinates are the TCP's own axes and the arrow can never disagree with the
    drawn gripper. The returned magnitude is the wrench's own ‖F‖, which the flip leaves alone.

    Re-adding by name is a replace in viser (the same idiom ``_draw_path`` and UMI's
    ``WrenchArrows`` use), so there is no handle to retain and no leak. Returns the drawn
    magnitude in N (0.0 when nothing was drawn) for the caller's readout.
    """
    segment = None
    if visible and force_sensor is not None:
        segment = force_arrow_local_segment(force_sensor, scale_m_per_n=scale_m_per_n)
    if segment is None:
        # Hidden rather than absent: a zero-length arrow would still render a head blob.
        server.scene.add_arrows(
            name, points=np.zeros((1, 2, 3), dtype=np.float32), colors=color, visible=False
        )
        return 0.0
    server.scene.add_arrows(
        name,
        points=segment[None].astype(np.float32),
        colors=color,
        shaft_radius=_FORCE_ARROW_STYLE["shaft_radius"] * radius_scale,
        head_radius=_FORCE_ARROW_STYLE["head_radius"] * radius_scale,
        head_length=_FORCE_ARROW_STYLE["head_length"] * radius_scale,
        visible=True,
    )
    return float(np.linalg.norm(np.asarray(force_sensor, dtype=np.float64).reshape(-1)))


def _chart_runs(values: np.ndarray, valid: np.ndarray) -> list[np.ndarray]:
    indices = np.flatnonzero(np.asarray(valid, dtype=bool) & np.isfinite(values))
    if not indices.size:
        return []
    boundaries = np.flatnonzero(np.diff(indices) > 1) + 1
    return [run for run in np.split(indices, boundaries) if run.size]


def _chart_svg(
    title: str,
    labels: tuple[str, ...],
    times: np.ndarray,
    target: np.ndarray,
    prediction: np.ndarray,
    valid: np.ndarray,
    target_colors: tuple[str, ...],
) -> str:
    """Render browser-safe chart paths; unlike Float64Array uPlot data, these preserve gaps."""
    times = np.asarray(times, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    if target.ndim == 1:
        target = target[:, None]
    if prediction.ndim == 1:
        prediction = prediction[:, None]
    valid = np.asarray(valid, dtype=bool)
    if valid.ndim == 1:
        valid = np.broadcast_to(valid[:, None], target.shape)

    left, right, top, bottom = 54.0, 708.0, 30.0, 145.0
    observed = [prediction[np.isfinite(prediction)]]
    if np.any(valid):
        observed.append(target[valid & np.isfinite(target)])
    finite = [part for part in observed if part.size]
    # np.concatenate([]) raises, so keep the empty case on the fallback branch below.
    values = np.concatenate(finite) if finite else np.empty(0)
    y_min, y_max = (float(values.min()), float(values.max())) if values.size else (-1.0, 1.0)
    if math.isclose(y_min, y_max):
        padding = max(abs(y_min) * 0.1, 1e-3)
    else:
        padding = (y_max - y_min) * 0.08
    y_min -= padding
    y_max += padding
    t_min = float(times.min()) if times.size else 0.0
    t_max = float(times.max()) if times.size else 1.0
    if math.isclose(t_min, t_max):
        t_max = t_min + 1.0

    def x_coord(index: int) -> float:
        return left + (float(times[index]) - t_min) / (t_max - t_min) * (right - left)

    def y_coord(value: float) -> float:
        return bottom - (value - y_min) / (y_max - y_min) * (bottom - top)

    # Light card matching the viser GUI panel (no dark fill). Ground truth is drawn
    # solid, the prediction dashed, both in the SAME per-axis colour so an axis's
    # GT/pred pair reads as one hue and the solid-vs-dashed contrast alone carries
    # the comparison — the MODE=1pt collection-viz convention.
    parts = [
        '<div style="width:100%;overflow-x:auto">',
        '<svg viewBox="0 0 720 190" width="100%" role="img" '
        f'aria-label="{html.escape(title)}" '
        'style="display:block;min-width:720px;background:#ffffff;'
        'border:1px solid #d9dfe7;border-radius:6px">',
        f'<text x="12" y="19" fill="#1a2230" font-size="13" font-weight="600">{html.escape(title)}</text>',
    ]
    for tick in range(5):
        fraction = tick / 4.0
        y = bottom - fraction * (bottom - top)
        value = y_min + fraction * (y_max - y_min)
        parts.append(f'<line x1="{left}" y1="{y:.2f}" x2="{right}" y2="{y:.2f}" stroke="#e3e8ef"/>')
        parts.append(
            f'<text x="{left - 6}" y="{y + 4:.2f}" text-anchor="end" fill="#5b6673" font-size="10">{value:.3g}</text>'
        )
    parts.extend(
        [
            f'<text x="{left}" y="{bottom + 17}" fill="#5b6673" font-size="10">{t_min:.3g}s</text>',
            f'<text x="{right}" y="{bottom + 17}" text-anchor="end" fill="#5b6673" font-size="10">{t_max:.3g}s</text>',
        ]
    )

    for axis, label in enumerate(labels):
        axis_color = target_colors[axis]
        for run_index, run in enumerate(_chart_runs(target[:, axis], valid[:, axis])):
            if run.size == 1:
                index = int(run[0])
                parts.append(
                    f'<circle data-kind="target" data-axis="{axis}" data-run="{run_index}" '
                    f'cx="{x_coord(index):.2f}" cy="{y_coord(float(target[index, axis])):.2f}" '
                    f'r="2.5" fill="{axis_color}"/>'
                )
            else:
                points = " ".join(
                    f"{x_coord(int(index)):.2f},{y_coord(float(target[index, axis])):.2f}" for index in run
                )
                parts.append(
                    f'<polyline data-kind="target" data-axis="{axis}" data-run="{run_index}" '
                    f'points="{points}" fill="none" stroke="{axis_color}" stroke-width="2.0" '
                    'stroke-linejoin="round" stroke-linecap="round"/>'
                )
        predicted_valid = np.isfinite(prediction[:, axis])
        for run_index, run in enumerate(_chart_runs(prediction[:, axis], predicted_valid)):
            if run.size == 1:
                # A one-point polyline draws nothing; mark the isolated sample like the
                # target branch does (hollow, so it reads as a dashed-series point).
                index = int(run[0])
                parts.append(
                    f'<circle data-kind="prediction" data-axis="{axis}" data-run="{run_index}" '
                    f'cx="{x_coord(index):.2f}" cy="{y_coord(float(prediction[index, axis])):.2f}" '
                    f'r="2.5" fill="none" stroke="{axis_color}" stroke-width="1.7"/>'
                )
                continue
            points = " ".join(
                f"{x_coord(int(index)):.2f},{y_coord(float(prediction[index, axis])):.2f}" for index in run
            )
            parts.append(
                f'<polyline data-kind="prediction" data-axis="{axis}" data-run="{run_index}" '
                f'points="{points}" fill="none" stroke="{axis_color}" stroke-width="1.7" '
                'stroke-dasharray="6 4" stroke-linejoin="round" stroke-linecap="round"/>'
            )
        # Legend: a solid swatch (GT) and a dashed swatch (pred), both in the axis
        # colour, so the solid-vs-dashed key is explicit rather than colour-coded.
        legend_x = left + axis * ((right - left) / max(1, len(labels)))
        parts.append(
            f'<line x1="{legend_x:.2f}" y1="175" x2="{legend_x + 16:.2f}" y2="175" '
            f'stroke="{axis_color}" stroke-width="2.0"/>'
        )
        parts.append(
            f'<line x1="{legend_x + 58:.2f}" y1="175" x2="{legend_x + 74:.2f}" y2="175" '
            f'stroke="{axis_color}" stroke-width="1.7" stroke-dasharray="6 4"/>'
        )
        parts.append(
            f'<text x="{legend_x + 20:.2f}" y="178" fill="{axis_color}" font-size="10">'
            f"{html.escape(label)} gt</text>"
        )
        parts.append(
            f'<text x="{legend_x + 78:.2f}" y="178" fill="{axis_color}" font-size="10">pred</text>'
        )
    parts.append("</svg></div>")
    return "".join(parts)


def _add_vector_chart(gui, title: str, labels: tuple[str, ...]):
    initial = np.zeros((2, len(labels)), dtype=np.float64)
    chart = gui.add_html(
        _chart_svg(title, labels, np.array([0.0, 1.0]), initial, initial, np.ones_like(initial, bool), _AXIS_COLORS)
    )
    chart._inference_title = title
    chart._inference_labels = labels
    chart._inference_target_colors = _AXIS_COLORS
    return chart


def _set_vector_chart(
    chart,
    times: np.ndarray,
    target: np.ndarray,
    prediction: np.ndarray,
    valid: np.ndarray,
) -> None:
    chart.content = _chart_svg(
        chart._inference_title,
        chart._inference_labels,
        times,
        target,
        prediction,
        valid,
        chart._inference_target_colors,
    )


def _add_scalar_chart(gui, title: str, unit: str):
    initial = np.zeros((2, 1), dtype=np.float64)
    chart = gui.add_html(
        _chart_svg(
            title,
            (unit,),
            np.array([0.0, 1.0]),
            initial,
            initial,
            np.ones_like(initial, bool),
            (_AXIS_COLORS[2],),
        )
    )
    chart._inference_title = title
    chart._inference_labels = (unit,)
    chart._inference_target_colors = (_AXIS_COLORS[2],)
    return chart


def _set_scalar_chart(chart, times, target, prediction, valid) -> None:
    _set_vector_chart(
        chart,
        times,
        np.asarray(target, dtype=np.float64)[:, None],
        np.asarray(prediction, dtype=np.float64)[:, None],
        np.asarray(valid, dtype=bool)[:, None],
    )


def _format_metric(value: float, unit: str) -> str:
    return "n/a" if not np.isfinite(value) else f"{value:.3f} {unit}"


@dataclass(frozen=True)
class _PlaybackData:
    """The recorded demonstration tail, prepared once per Play, in the display frame.

    Built by :func:`_build_playback_data` from :meth:`HLSampleSource.episode_forward_poses`.
    Every role's row 0 is the sample's own anchor (target step 0), rebased against the SAME
    reference :func:`_anchor_reference` gives the static chunk — so the swept cluster rides
    exactly on the green recorded-path line and shows the identical gripper/TCP/tracker/wrist
    relationship the training chunk does. Held immutably; the render loop only advances a
    cursor over it.
    """

    times_s: np.ndarray                 # (N,) camera times, 0 at the play-start frame
    tail_len: int                       # N (>= 1)
    home_anchored: bool                 # the frame this tail was rebased for
    #: role -> {"position": (N,3), "wxyz": [N wxyz tuples], "gripper_frac": (N,) | None,
    #:          "wrench": (N,6) | None, "wrench_valid": (N,) | None, "anchored": bool}.
    #: gripper_frac drives the mesh jaw (recorded width, since there is no prediction during
    #: playback); "wrench" is the per-frame recorded sensor-frame wrench, carried through
    #: UNROTATED because it is frame-invariant (the display rebase applies to poses only, and
    #: the sensor->TCP relabel is applied at draw time against the posed frame); "anchored" is
    #: False for a role left raw.
    per_role: dict


@dataclass(frozen=True)
class _DisplayPoses:
    """One role's drawable poses for the current rollout, in the chosen display frame.

    Built by :func:`_build_display_poses`, which either passes the decoded absolute poses
    straight through (raw frame) or re-bases all of them onto the upright display home
    against ONE shared reference. Holding all four series together is what guarantees they
    are drawn in a single coherent frame — a per-series rebase is exactly how the
    trajectory lines and the gripper cluster would silently drift apart.
    """

    predicted_position: np.ndarray      # (H, 3)
    predicted_wxyz: tuple              # H wxyz tuples
    target_position: np.ndarray         # (H, 3)
    target_wxyz: tuple                 # H wxyz tuples
    episode_position: np.ndarray | None  # (N, 3) recorded tail, or None
    #: Whether the rebase was actually APPLIED to this role. False either because the
    #: viewer is in raw mode or because this role's reference rotation was degenerate and
    #: it fell back — so the GUI can say which, instead of a ticked checkbox implying a
    #: rebase that silently did not happen for one role.
    anchored: bool = False


def _anchor_reference(role_result, home_anchored: bool) -> tuple[np.ndarray, np.ndarray] | None:
    """The ``(ref_pos, ref_rot)`` this role rebases against, or ``None`` to leave it raw.

    The reference is the TARGET's step 0 — the sample's own observation-time anchor, the
    pose both the target chunk and the prediction are decoded against. Returning it here (as
    opposed to inlining the choice) is what lets the static action chunk AND the playback
    tail rebase against the IDENTICAL reference: the recorded demonstration must ride exactly
    on the green line and share the gripper/TCP/tracker/wrist relationship the trained chunk
    shows, so the two rebases cannot be allowed to drift.

    Returns ``None`` when the viewer is in raw mode OR step 0 is not a usable anchor — masked
    invalid (its pose is not a real measurement) or a degenerate rotation — which is exactly
    when the caller must fall back to the raw frame rather than render a silently wrong scene.
    With ``drop_glitch_windows`` on (the trained default) an invalid step 0 is not a reachable
    anchor, but a ``--keep-glitch-windows`` run or a future filter could expose one, so the
    guard is cheap insurance rather than dead code."""
    if not home_anchored:
        return None
    reference_pos = np.asarray(role_result.target_position, dtype=np.float64)[0]
    reference_rot = rot6d_to_rotmat(np.asarray(role_result.target_rot6d, dtype=np.float64)[0])
    step0_valid = bool(np.asarray(role_result.pose_valid)[0])
    reference_ok = (
        step0_valid
        and np.isfinite(reference_rot).all()
        and np.isclose(float(np.linalg.det(reference_rot)), 1.0, atol=1e-5)
    )
    return (reference_pos, reference_rot) if reference_ok else None


def _heading_alignment(
    result, home_anchored: bool, enabled: bool, role_geometry=None,
) -> tuple[np.ndarray | None, float, str | None]:
    """``(W, heading_deg, source)`` — ONE shared yaw-only world rotation for this rollout, or None.

    Derived per ROLLOUT, not once at startup, because the heading is the sample's own: it comes
    from the CHEST role's anchor (target step 0), the same reference every role rebases against
    (``source == "chest"``). A layout with NO chest — ``left_only`` (the plain ``diffusion_umi``
    preset, e.g. wipe_board) or ``both_hands`` — falls back to the GRIPPER roles' anchor grip yaw,
    relative to each hand's HOME orientation (:func:`ready_pose.grip_yaw_alignment`; ``source ==
    "LEFT grip yaw (no chest anchor)"``). That fallback is MEASURED: at episode start the hand's
    TCP x-axis yaw tracks the chest heading to a median of +1.8 deg (rotate_box) / -8.3 deg
    (push_cart), std 6.6-8.0 deg, never more than 45 deg off. Without it a one-hand scene was
    drawn MIRRORED — wipe_board's operator reads -118.8 deg (std 7.0 over 291 starts), so a hand
    moving in its own forward was drawn going backwards next to a body facing +x, while the two
    5pt pages beside it were turned correctly by their chest.
    Returns ``(None, 0.0, None)`` in the raw frame, when the caller has not opted in, or when no
    usable anchor exists (a hand within 20 deg of vertical has no yaw to read).
    """
    if not (enabled and home_anchored):
        return None, 0.0, None
    head = next((r for r in result.roles if r.role == "head"), None)
    reference = None if head is None else _anchor_reference(head, home_anchored)
    if reference is not None:
        world, heading = ready_pose.world_yaw_alignment(reference[1])
        return world, heading, "chest"
    hands: dict[str, np.ndarray] = {}
    homes: dict[str, np.ndarray | None] = {}
    for role_result in result.roles:
        if role_result.role not in GRIPPER_ROLES:
            continue
        ref = _anchor_reference(role_result, home_anchored)
        if ref is None:
            continue
        hands[role_result.role] = ref[1]
        geometry = None if role_geometry is None else role_geometry.get(role_result.role)
        homes[role_result.role] = None if geometry is None else geometry.home_rotmat
    world, heading, used = ready_pose.grip_yaw_alignment(hands, homes)
    if world is None:
        return None, 0.0, None
    return world, heading, "+".join(r.upper() for r in used) + " grip yaw (no chest anchor)"


def _build_display_poses(
    role_result, episode_positions: np.ndarray | None, geometry, home_anchored: bool,
    world_rotmat: np.ndarray | None = None,
) -> _DisplayPoses:
    """Resolve one role's drawable poses for the chosen display frame.

    The reference is the TARGET's step 0 (see :func:`_anchor_reference`) — the sample's own
    observation-time anchor. Using that single reference for the prediction, the target AND
    the recorded tail is what makes the rebase error-preserving and keeps the three lines
    mutually registered (the tail's first point still coincides with the target's step 0).

    In raw mode this returns the decoded absolute aligned-world poses untouched, so the
    viewer stays a strict fidelity check when the toggle is off.
    """
    predicted_rot = rot6d_to_rotmat(np.asarray(role_result.predicted_rot6d, dtype=np.float64))
    target_rot = rot6d_to_rotmat(np.asarray(role_result.target_rot6d, dtype=np.float64))
    predicted_pos = np.asarray(role_result.predicted_position, dtype=np.float64)
    target_pos = np.asarray(role_result.target_position, dtype=np.float64)
    tail = None if episode_positions is None else np.asarray(episode_positions, dtype=np.float64)
    anchored = False

    reference = _anchor_reference(role_result, home_anchored)
    if reference is not None:
        reference_pos, reference_rot = reference
        home = geometry.home_pos
        # The collection-viz / zarr-replay rebase (gripper_only_view.home_anchor): positions
        # are TRANSLATED onto the home (world directions kept, so the grid still means "the
        # floor"), orientations get the right-multiplied world delta R_cur @ R_ref^T. Applied
        # with ONE shared reference (target step 0) to prediction, target AND tail so all
        # three stay registered; reference_rot is passed to positions too for call symmetry
        # but is ignored there (see home_anchor_positions).
        # ``world_rotmat`` is ONE shared yaw-only rotation for the whole scene (all roles, all
        # three series) or None. Sharing it is the load-bearing part: a per-role matrix would
        # rotate each role's displacement differently and destroy the hand-vs-chest geometry.
        predicted_pos = home_anchor_positions(
            predicted_pos, reference_pos, home, reference_rot, world_rotmat=world_rotmat)
        target_pos = home_anchor_positions(
            target_pos, reference_pos, home, reference_rot, world_rotmat=world_rotmat)
        if tail is not None:
            tail = home_anchor_positions(
                tail, reference_pos, home, reference_rot, world_rotmat=world_rotmat)
        # ``home_rotmat`` is None for the display home (=> identity, the collection-viz
        # convention) and the FK'd ready-pose orientation for --display-frame g1-ready-pose.
        predicted_rot = home_anchor_rotmats(predicted_rot, reference_rot, geometry.home_rotmat,
                                            world_rotmat=world_rotmat)
        target_rot = home_anchor_rotmats(target_rot, reference_rot, geometry.home_rotmat,
                                         world_rotmat=world_rotmat)
        anchored = True

    return _DisplayPoses(
        predicted_position=predicted_pos,
        predicted_wxyz=tuple(_wxyz_from_rotmat(matrix) for matrix in predicted_rot),
        target_position=target_pos,
        target_wxyz=tuple(_wxyz_from_rotmat(matrix) for matrix in target_rot),
        episode_position=tail,
        anchored=anchored,
    )


def _build_playback_data(result, forward_poses, role_geometry, home_anchored: bool,
                         world_rotmat: np.ndarray | None = None,
                         world_by_role: dict | None = None) -> _PlaybackData:
    """Prepare the recorded demonstration tail for playback, in the chosen display frame.

    ``forward_poses`` is ``(times_s, per_role)`` from :meth:`HLSampleSource.episode_forward_poses`;
    ``role_geometry`` is the same ``role -> RoleGeometry`` map the static scene uses, so playback
    hangs the cluster off the identical display home. Each role is rebased against the SAME
    reference :func:`_anchor_reference` gives the static action chunk (target step 0), so the
    swept gripper/TCP/tracker/wrist cluster rides exactly on the green recorded-path line — the
    recorded demonstration IS the ground truth, so it must not sit in a different frame than the
    chunk drawn from the same anchor. In raw mode the absolute poses pass through untouched. A
    role whose step 0 is not a usable anchor stays raw (``per_role[role]["anchored"] == False``),
    mirroring :func:`_build_display_poses`.
    """
    times_s, per_role_src = forward_poses
    times_s = np.asarray(times_s, dtype=np.float64)
    tail_len = int(times_s.shape[0])
    role_result_by_name = {role_result.role: role_result for role_result in result.roles}
    per_role: dict = {}
    for role, entry in per_role_src.items():
        role_result = role_result_by_name.get(role)
        positions = np.asarray(entry["position"], dtype=np.float64).reshape(-1, 3)
        rotmats = rot6d_to_rotmat(np.asarray(entry["rot6d"], dtype=np.float64))
        anchored = False
        reference = None if role_result is None else _anchor_reference(role_result, home_anchored)
        if reference is not None and role in role_geometry:
            reference_pos, reference_rot = reference
            geometry = role_geometry[role]
            # Per SOURCE FRAME, exactly as the static scene (see _rebuild_scene): the tail must
            # ride on the lines, so the two rebases cannot use different matrices for one role.
            world_r = world_rotmat if world_by_role is None else world_by_role.get(role)
            positions = home_anchor_positions(
                positions, reference_pos, geometry.home_pos, reference_rot,
                world_rotmat=world_r)
            rotmats = home_anchor_rotmats(rotmats, reference_rot, geometry.home_rotmat,
                                          world_rotmat=world_r)
            anchored = True
        gripper_m = entry.get("gripper_m")
        gripper_frac = None
        if gripper_m is not None:
            gripper_frac = np.clip(
                np.asarray(gripper_m, dtype=np.float64) / GRIPPER_WIDTH_REF_M, 0.0, 1.0
            )
        wrench = entry.get("wrench")
        wrench_valid = entry.get("wrench_valid")
        per_role[role] = {
            "position": positions,
            "wxyz": [_wxyz_from_rotmat(matrix) for matrix in rotmats],
            "gripper_frac": gripper_frac,
            # NOT rebased: the wrench is expressed in the bota SENSOR frame, which rides the
            # tool, so it is invariant to where the tool is drawn. The only rotation it needs
            # is the sensor->TCP relabel, and that is applied at draw time in the coordinates
            # of the posed frame — so it follows the rebase without being touched here.
            "wrench": None if wrench is None else np.asarray(wrench, dtype=np.float64).reshape(-1, 6),
            "wrench_valid": None if wrench_valid is None else np.asarray(wrench_valid, dtype=bool).reshape(-1),
            "anchored": anchored,
        }
    return _PlaybackData(
        times_s=times_s,
        tail_len=max(tail_len, 1),
        home_anchored=bool(home_anchored),
        per_role=per_role,
    )


def run_viewer(
    session,
    *,
    host: str = "127.0.0.1",
    port: int = 8086,
    sample_index: int = 0,
    seed: int = 0,
    motion_completion: dict | None = None,
    keypoint_ik: dict | None = None,
    display_frame: str = "display-home",
    zero_arms: bool = True,
    home_anchor_path: str | None = None,
) -> None:
    """Run a blocking interactive viewer for a Flow or Diffusion inference session.

    ``motion_completion`` — ``None`` (default) means the page is exactly what it has always
    been: no humanoid, no G1 URDF load, no model subprocess, no extra panel. A dict of
    :class:`~fuji.s1_policy.inference.mc_humanoid.MCHumanoid` kwargs opts in; a missing
    prerequisite disables the panel with a reason instead of failing the launch.

    ``keypoint_ik`` — ``None`` (default) means no G1 body and no URDF load. A dict of
    :class:`~fuji.s1_policy.inference.keypoint_humanoid.KeypointHumanoid` kwargs opts in; it is
    the ``fivepoint_geom_*`` counterpart of ``motion_completion`` and the two are independent
    (a layout can satisfy at most one of them, so they never both draw).

    ``display_frame`` — ``"display-home"`` (default) hangs the home-anchored scene on
    :data:`~fuji.s1_policy.inference.display.DISPLAY_HOME_POS` exactly as before;
    ``"g1-ready-pose"`` hangs it on the pose the robot holds after the operator presses **A**
    (see :mod:`fuji.s1_policy.inference.ready_pose`). The second is what a whole-body
    completion needs, and it changes no metric — only where the scene sits.
    """
    try:
        import viser
    except ImportError as exc:
        raise RuntimeError("viser is required for the browser UI; install the project's viz extra") from exc

    if not len(session.source):
        raise ValueError("inference dataset has no usable samples")
    sample_index = _checked_sample_index(sample_index, len(session.source))
    port = _checked_port(port)
    _require_port_available(host, port)

    server = viser.ViserServer(host=host, port=port, label=f"FUJI S1 {session.algorithm} inference")
    if server.get_port() != port:
        actual_port = server.get_port()
        server.stop()
        raise RuntimeError(f"viser port {port} is unavailable; it selected {actual_port} instead")
    try:
        _serve_viewer(session, server, host=host, port=port, sample_index=sample_index, seed=seed,
                      motion_completion=motion_completion, keypoint_ik=keypoint_ik,
                      display_frame=display_frame,
                      zero_arms=zero_arms, home_anchor_path=home_anchor_path)
    finally:
        server.stop()


def _serve_viewer(session, server, *, host: str, port: int, sample_index: int, seed: int,
                  motion_completion: dict | None = None, keypoint_ik: dict | None = None,
                  display_frame: str = "display-home", zero_arms: bool = True,
                  home_anchor_path: str | None = None) -> None:
    server.scene.set_up_direction("+z")
    # xy grid, ON by default (Display -> "Ground grid" hides it for a clean screenshot), kept as
    # a spatial reference. It IS the world floor in the raw frame; in the home-anchored frame the
    # scene is rotated to identity, so the grid is a fixed reference plane rather than the true
    # floor and vertical motion against it is not physical — the "3D frame" readout states which
    # meaning is live so the grid can't be misread (a physical lift can render as a descent while
    # anchored). The handle is kept because the toggle flips its .visible from the render loop.
    grid_handle = server.scene.add_grid(
        "/grid", plane="xy", shadow_opacity=0.0, infinite_grid=True, cell_size=0.1)
    gui = server.gui
    gui.configure_theme(control_layout="collapsible", control_width="large")

    with gui.add_folder("Inference"):
        sample_slider = gui.add_slider(
            "Dataset sample index",
            min=0,
            max=len(session.source) - 1,
            step=1,
            initial_value=sample_index,
        )
        seed_input = gui.add_text("Seed", initial_value=str(seed))
        steps_input = gui.add_number(
            "Sampler steps",
            initial_value=int(session.num_steps),
            min=1,
            max=session.max_num_steps,
            step=1,
        )
        rollout_button = gui.add_button("Run inference")
        status = gui.add_text("Status", initial_value="waiting", disabled=True)

    with gui.add_folder("Sample"):
        sample_text = gui.add_text("Identity", initial_value="-", disabled=True)
        split_text = gui.add_text("Split", initial_value="-", disabled=True)
        validity_text = gui.add_text("Inputs", initial_value="-", disabled=True)
        action_step = gui.add_slider("Action step", min=0, max=ACTION_HORIZON - 1, step=1, initial_value=0)

    with gui.add_folder("Metrics"):
        metric_text = {
            "position_cm": gui.add_text("Aggregate position", initial_value="-", disabled=True),
            "rotation_geodesic_deg": gui.add_text("Aggregate rotation", initial_value="-", disabled=True),
            "gripper_mm": gui.add_text("Aggregate gripper", initial_value="-", disabled=True),
            "force_n": gui.add_text("Aggregate force", initial_value="-", disabled=True),
            "torque_nm": gui.add_text("Aggregate torque", initial_value="-", disabled=True),
        }

    image_handles = []
    with gui.add_folder("Conditioning images"):
        # CAMERA-MAJOR, matching HLDataset.__getitem__'s own stacking order
        # (`for key in self.image_keys for hf in hist_frames`) — the tuple in
        # `RolloutResult.conditioning_images` is that stack decoded, and the zip below is
        # strict, so a wrong nesting order here mislabels every view rather than erroring.
        # One panel per (camera, history frame): a 2-camera / To=1 checkpoint
        # (diffusion_umi_3tracker) gets two, and iterating history alone gave it one and
        # tripped the strict zip.
        history_len = session.source.dataset.obs_history_len
        for camera in session.source.image_keys:
            for index in range(history_len):
                suffix = " (t0)" if index == history_len - 1 else ""
                label = f"{camera} obs {index}{suffix}" if len(session.source.image_keys) > 1 \
                    else f"Observation {index}{suffix}"
                image_handles.append(
                    gui.add_image(
                        np.zeros((16, 16, 3), dtype=np.uint8),
                        label=label,
                        format="png",
                    )
                )

    charts = {}
    with gui.add_folder("Trajectory charts"):
        for role in session.source.layout.roles:
            # "stored" not "world": the charts deliberately plot the RAW absolute
            # aligned-world values regardless of the 3D home-anchor toggle, so they stay a
            # faithful data readout while the scene carries the display frame.
            role_charts = {"position": _add_vector_chart(gui, f"{role} position (stored m)", ("x", "y", "z"))}
            if role in session.source.layout.gripper_roles:
                role_charts["gripper"] = _add_scalar_chart(gui, f"{role} gripper", "m")
            if role in session.source.layout.force_roles:
                # F^cmd, not F_meas: the sign flip is applied to BOTH series (see _cmd_chart),
                # so the target/prediction comparison is untouched and the plotted sign is the
                # one the wire would carry. The title says so — a chart labelled "force" beside
                # an arrow drawn the other way is exactly the ambiguity this removes.
                role_charts["force"] = _add_vector_chart(
                    gui, f"{role} F^cmd = -F_meas (sensor frame N)", ("Fx", "Fy", "Fz")
                )
                role_charts["torque"] = _add_vector_chart(
                    gui, f"{role} T^cmd = -T_meas (sensor frame Nm)", ("Tx", "Ty", "Tz")
                )
            charts[role] = role_charts

    # Optional yam gripper meshes (the MODE=1pt collection-viz geometry). Loaded
    # once here; if unavailable the viewer still draws trajectory lines + triads.
    overlay, overlay_message = _load_gripper_overlay(session.source.layout.roles)
    print(f"[s1-inference] {overlay_message}", flush=True)
    # Where the mesh + triad cluster hangs off each trajectory point, derived from the
    # DATASET's own T_tracker_tcp extrinsic (never a hardcoded lever — the extrinsic
    # differs between datasets, so hardcoding one silently mis-draws the others).
    # --display-frame g1-ready-pose replaces those homes with the pose the robot HOLDS after
    # the operator presses A, so the chunk is placed where deploy would place it. Resolved
    # BEFORE the geometry because it is one of its inputs; unavailable prerequisites degrade to
    # the display homes with a printed reason (the trajectory is still correct either way).
    deploy_anchor = None
    if display_frame == "g1-ready-pose":
        deploy_anchor, deploy_reason = ready_pose.resolve_deploy_anchor(
            session.source.layout.roles, zero_arms=zero_arms, home_anchor_path=home_anchor_path)
        if deploy_anchor is None:
            print(f"[s1-inference] display frame FELL BACK to the display homes: "
                  f"{deploy_reason}", flush=True)
        else:
            print(f"[s1-inference] display frame: {deploy_anchor.describe()}", flush=True)
            for note in deploy_anchor.notes:
                print(f"[s1-inference] {note}", flush=True)
    home_pose = None if deploy_anchor is None else {
        role: (deploy_anchor.pos[role], deploy_anchor.rotmat[role]) for role in deploy_anchor.pos
    }
    role_geometry, geometry_notes = resolve_role_geometry(
        session.source.layout, session.source.file_attrs, home_pose)
    for note in geometry_notes:
        print(f"[s1-inference] {note}", flush=True)
    # WHICH SOURCE FRAME each role's displacement is in. `tracker_roles_required` is the
    # dataset's own statement of which roles a tracker recorded; every other role's pose was
    # DERIVED by forward kinematics and is already in the completion's pelvis frame. For all but
    # `fivepoint_*` the two lists are equal, so the yaw-only scene rotation stays exactly the one
    # shared matrix display.py requires. See display.world_rotmat_by_role.
    roles_in_order = tuple(session.source.layout.roles)
    recorded_roles = tuple(tracker_roles_required(session.source.layout.mode))
    derived_roles = tuple(r for r in roles_in_order if r not in recorded_roles)
    if derived_roles:
        print(f"[s1-inference] role(s) {list(derived_roles)} are FK-DERIVED (pelvis frame, no "
              f"root pose), not recorded by a tracker: the scene's yaw-only turn to the "
              f"operator's forward is applied to {list(recorded_roles)} only, because a "
              f"body-frame displacement has no heading to undo.", flush=True)
        # A MODE=5pt recording (push_gantry onwards, `convert_to_umi.py --feet recorded`) also
        # carries the two foot PUCKS as `hl_{left,right}_foot_*`. This layout does not read them —
        # its foot targets and its state are the completion's FK — so say so, or the line above
        # reads as "the file has no recorded feet" on a file that does.
        if any(dict(a or {}).get("feet_recorded") for a in (session.source.file_attrs or ())):
            print("[s1-inference]   (this dataset ALSO carries the two foot pucks' RECORDED poses, "
                  "hl_{left,right}_foot_*; the fivepoint_* layout does not train on or draw them — "
                  "the drawn feet are the completion's FK, as on every other 5pt file.)", flush=True)
    # Per-role residuals, NOT one arbitrary role's: resolve_role_geometry reports per-file
    # extrinsic disagreement, so collapsing to next(iter(...)) would throw that away. Rolled
    # up as the worst case, naming the role, so nothing is presented as global that isn't.
    worst_role, worst_geometry = max(
        role_geometry.items(), key=lambda item: float(np.linalg.norm(item[1].tcp_triad_offset))
    )
    tcp_residual_mm = float(np.linalg.norm(worst_geometry.tcp_triad_offset)) * 1000.0
    # Same tolerance the provenance note uses (TCP_RESIDUAL_TOL_MM), so the "3D frame" badge
    # and the "Geometry note" line can never disagree on an f32 round-trip. Reported with one
    # decimal: a 0.3 mm residual printed as "0 mm off" reads as self-contradictory.
    datum_is_tcp = tcp_residual_mm < TCP_RESIDUAL_TOL_MM
    datum_text = (
        "TCP (stored pose is the fingertip)"
        if datum_is_tcp
        else f"NOT TCP: {worst_role} triad {tcp_residual_mm:.1f} mm off — re-convert"
    )

    with gui.add_folder("Display"):
        show_grid = gui.add_checkbox("Ground grid", initial_value=True)
        show_gripper_mesh = gui.add_checkbox(
            "Gripper mesh", initial_value=overlay is not None, disabled=overlay is None
        )
        show_frames = gui.add_checkbox("EE frames (TCP/tracker/wrist)", initial_value=True)
        # Force arrows at the TCP, drawn only for roles that actually carry an F/T sensor.
        # Disabled (not merely unticked) when the layout declares none, so an empty checkbox
        # cannot read as "arrows are on but the force is zero".
        has_force_role = bool(session.source.layout.force_roles)
        show_force_arrows = gui.add_checkbox(
            "F^cmd arrows (TCP frame)", initial_value=has_force_role, disabled=not has_force_role
        )
        force_scale_slider = gui.add_slider(
            "Force arrow mm/N", min=2.0, max=40.0, step=1.0, initial_value=FORCE_ARROW_SCALE_M_PER_N * 1000.0
        )
        force_arrow_text = gui.add_text("|F| at step", initial_value="-", disabled=True)
        # Spell the colour key out. The arrows deliberately reuse the trajectory hues rather
        # than inventing a "force" colour, so the mapping is worth stating once next to the
        # toggle instead of relying on the operator connecting arrow to line by eye.
        gui.add_text("Arrow colours", initial_value="blue = GT · orange = pred · green = recorded",
                     disabled=True)
        # BOTH conventions, side by side: the rotation says which axis, the sign says which WAY.
        # Either one alone leaves "is it pushing into the table?" unanswerable from the page.
        gui.add_text("Arrow frame", initial_value=TCP_FROM_SENSOR_NOTE, disabled=True)
        gui.add_text("Arrow sign", initial_value=CMD_FROM_MEAS_NOTE, disabled=True)
        # The three trajectory lines, each with its own toggle so GT and prediction can be read
        # one at a time: the masked GT chunk (blue), the prediction (orange) and the recorded
        # rest-of-episode EE path (green) from the sampled frame to its episode end — the
        # demonstration this chunk was cut from, for overall-shape context.
        show_target_line = gui.add_checkbox("GT chunk line (blue)", initial_value=True)
        show_prediction_line = gui.add_checkbox("Prediction line (orange)", initial_value=True)
        show_episode_path = gui.add_checkbox("Episode path (recorded, green)", initial_value=True)
        # Pose triads along BOTH chunk lines every N steps (plus the last step), sized by the
        # slider. Off by default: at stride 1 this is 2 x H frames per role. The triads follow
        # their line's toggle, so hiding the GT line hides the GT triads too.
        show_path_axes = gui.add_checkbox(
            "Pose axes along GT + prediction", initial_value=False,
            hint="a triad on both chunk lines every N steps; the two sliders below size it")
        # ⚠️ Moving either slider TURNS THE AXES ON (see _axes_slider_request). Without that, a
        # slider under an unticked master checkbox is a control that silently does nothing — which
        # is exactly how it read on the rig: the page had been restarted, the checkbox was back to
        # its default OFF, and the sliders looked broken rather than inactive.
        path_axes_stride = gui.add_slider(
            "Axes every N steps", min=1, max=ACTION_HORIZON, step=1,
            initial_value=min(_PATH_AXES_DEFAULT_STRIDE, ACTION_HORIZON),
            hint="moving this turns the axes on",
        )
        path_axes_length = gui.add_slider(
            "Axis length mm", min=10.0, max=100.0, step=5.0, initial_value=_PATH_AXES_DEFAULT_LENGTH_MM,
            hint="moving this turns the axes on",
        )
        # +Z UP (default ON), matching the collection viz and the zarr replay. The stored
        # pose is the puck held in the hand (+z into the floor), so a raw render hangs the
        # gripper upside down. Uses the SAME rebase as gripper_only_view.home_anchor (positions
        # translated onto ONE shared home, orientations right-multiplied by R_cur @ R_ref^T),
        # so the GT trajectory is drawn identically to the zarr replay and world directions are
        # preserved. OFF = strict raw absolute aligned-world frame.
        home_anchor_view = gui.add_checkbox("Home-anchored +Z up (like collection)", initial_value=True)
        # What the 3D scene is showing RIGHT NOW, reconciled by the render loop. The
        # checkbox alone can lie: a role with a degenerate reference rotation silently stays
        # in the raw frame, and the charts never follow the display rebase regardless.
        frame_text = gui.add_text("3D frame", initial_value="-", disabled=True)
        # Position/gripper are raw stored values; the wrench charts are the ONE exception — they
        # carry the F^cmd sign so they agree with the arrows (see the module docstring). Saying
        # "always raw" here while the chart titles say F^cmd would be the page contradicting
        # itself on the exact question the sign exists to answer.
        gui.add_text("Charts/readouts",
                     initial_value="raw stored values, except the wrench charts (F^cmd = -F_meas)",
                     disabled=True)
        gui.add_text("Trajectory datum", initial_value=datum_text, disabled=True)
        # The geometry provenance also goes to stdout, but an operator who SSH-tunnels the
        # port never sees the terminal — so surface it on the page too.
        for index, note in enumerate(geometry_notes):
            gui.add_text(f"Geometry note {index + 1}", initial_value=note, disabled=True)

    with gui.add_folder("Playback"):
        # Plays the RECORDED demonstration forward from the selected sample's frame to its
        # episode end — the gripper cluster sweeps the green recorded path while the RGB
        # observation below advances in real time (the 15 Hz camera grid). It shows the
        # ground-truth recording, not a per-frame model run, so it is GPU-free and always
        # real-time; the static blue/orange chunk stays drawn for context (same display
        # frame). "Stop" returns to the single-sample action-step view.
        play_button = gui.add_button("▶ Play")
        stop_button = gui.add_button("⏹ Stop")
        # Range is set to the tail length when playback data is first built (0 == the
        # sample's own t0 frame). Dragging it seeks and enters playback view.
        play_slider = gui.add_slider("Playback frame", min=0, max=1, step=1, initial_value=0)
        speed_slider = gui.add_slider("Speed x", min=0.25, max=4.0, step=0.25, initial_value=1.0)
        loop_checkbox = gui.add_checkbox("Loop", initial_value=False)
        play_text = gui.add_text("Playback", initial_value="idle", disabled=True)
        # The RAW recorded camera frame at the current playback cursor (NOT the normalized
        # model input the "Conditioning images" panel shows). Advances with the cursor.
        playback_image = gui.add_image(
            np.zeros((16, 16, 3), dtype=np.uint8), label="Observation (recorded)", format="jpeg"
        )

    # OPT-IN whole-body motion completion, LAST in the panel because it is a second model
    # downstream of the policy rather than a readout of it. Constructed unconditionally so the
    # render loop can call it without branching; inert (and panel-less) without the flag. See
    # fuji/s1_policy/inference/mc_humanoid.py for why this is not an arm IK.
    from fuji.s1_policy.inference.mc_humanoid import MCHumanoid  # noqa: PLC0415

    # In the READY-POSE frame the placement is already FK'd: the floor is z=0 by construction,
    # so dz is PINNED to 0 rather than derived from the chest puck. Deriving it there would be
    # actively wrong — auto_dz_chest targets RIG_CHEST_STAND_Z = 1.30 while the robot's own
    # chest keypoint at this pose sits at 1.151, so it would lift the whole take ~15 cm and
    # push the commanded hands back out of the checkpoint's training band. Passed as a default
    # the caller can still override.
    mc_defaults: dict = {"enabled": False}
    if deploy_anchor is not None:
        mc_defaults.update(cond_mode="absolute", dz_wrist=0.0,
                           placement_note=deploy_anchor.describe())
    mc = MCHumanoid(server, **{**mc_defaults, **(motion_completion or {})})
    mc.build_gui(gui)
    if motion_completion is not None and not mc.available:
        print(f"[s1-inference] motion completion DISABLED: {mc.disabled_reason}", flush=True)

    # OPT-IN FIVE-POINT IK BODY — the `fivepoint_geom_*` sibling of the panel above. Same six
    # methods, driven from the same five call sites; inert (and panel-less) without the flag. It
    # needs the READY-POSE frame as a PREREQUISITE and not a preference: the three tracked roles
    # and the two FK'd feet are in different source frames, so the display homes put the feet
    # metres from the body (measured 543 mm of IK residual). Refused with that reason rather than
    # drawn wrong. See fuji/s1_policy/inference/keypoint_humanoid.py.
    from fuji.s1_policy.inference.keypoint_humanoid import KeypointHumanoid  # noqa: PLC0415

    kp_defaults: dict = {"enabled": False}
    if keypoint_ik is not None:
        # The IK roles this layout does NOT predict (a left_only checkpoint: right, head, both
        # feet) are HELD at the ready pose — the same FK, over all five roles, in the same frame
        # the predicted roles were re-based onto above (their step 0 IS deploy_anchor.pos[role]).
        # Resolved only when something is missing, so a five-point checkpoint runs the one FK it
        # always did.
        from fuji.s1_policy.inference.keypoint_ik import IK_ROLES  # noqa: PLC0415

        kp_held: dict = {}
        kp_missing = [r for r in IK_ROLES if r not in session.source.layout.roles]
        if deploy_anchor is not None and kp_missing:
            ik_anchor, ik_reason = ready_pose.resolve_deploy_anchor(
                IK_ROLES, zero_arms=zero_arms, home_anchor_path=home_anchor_path)
            if ik_anchor is None:
                print(f"[s1-inference] five-point IK body: cannot hold {kp_missing} at the ready "
                      f"pose ({ik_reason}); the body will be disabled", flush=True)
            else:
                kp_held = {r: (ik_anchor.pos[r], ik_anchor.rotmat[r]) for r in kp_missing}
                print(f"[s1-inference] five-point IK body: layout {session.source.layout.mode!r} "
                      f"predicts {list(session.source.layout.roles)}; {kp_missing} are HELD at the "
                      f"ready pose for both bodies (the rig's wire does the same).", flush=True)
        kp_defaults.update(
            frame_ok=deploy_anchor is not None,
            frame_note=(deploy_anchor.describe() if deploy_anchor is not None else
                        "--keypoint-ik needs --display-frame g1-ready-pose and its two "
                        "submodules; the display homes are not one coherent frame for five "
                        "roles"),
            seed_dof=None if deploy_anchor is None else deploy_anchor.dof,
            home_anchor_path=home_anchor_path,
            held_poses=kp_held)
    kp = KeypointHumanoid(server, **{**kp_defaults, **(keypoint_ik or {})})
    kp.build_gui(gui)
    if keypoint_ik is not None and not kp.available:
        print(f"[s1-inference] five-point IK body DISABLED: {kp.disabled_reason}", flush=True)

    request_queue: queue.Queue[_RolloutRequest] = queue.Queue(maxsize=1)
    request_lock = threading.RLock()
    request_generation = {"value": 0}
    step_state = {"requested": 0, "shown": -1}
    current_result: RolloutResult | None = None
    # role -> _DisplayPoses for the currently shown rollout. Render-thread only.
    display_state: dict[str, _DisplayPoses] = {}
    # The shared yaw-only world rotation of the CURRENT rollout, for the readout. Render-thread
    # only; ``world`` is None whenever the scene was not turned.
    heading_state: dict = {"world": None, "deg": 0.0, "source": None}
    # role -> (green line handle, had real segments). Recorded rest-of-episode path.
    # Mutated ONLY on the render thread (apply_result + the render loop); the toggle
    # handler never touches it — see episode_path_state.
    episode_path_handles: dict[str, tuple[object, bool]] = {}
    # Visibility is per-sample (not per-step). "buttons raise flags; the render loop
    # mutates": the toggle callback (a viser callback thread) only writes "requested"
    # under request_lock; the render loop diffs it against "applied" and flips the green
    # lines' .visible there, so episode_path_handles is never read/written across threads.
    episode_path_state = {"requested": bool(show_episode_path.value), "applied": None}
    # Same flag discipline for the home-anchor toggle: the callback only records the
    # request, the render loop rebuilds every line + repositions the cluster.
    anchor_state = {"requested": bool(home_anchor_view.value), "applied": None}
    # Grid toggle, same discipline. Reconciled every tick, with or without a rollout.
    grid_state = {"requested": bool(show_grid.value), "applied": None}
    # The blue/orange chunk lines' toggles + handles (role -> series -> (handle, had_segments)).
    # Handles are retained ONLY for the .visible flip; a frame switch still rebuilds them.
    line_state = {
        "target": {"requested": bool(show_target_line.value), "applied": None},
        "prediction": {"requested": bool(show_prediction_line.value), "applied": None},
    }
    chunk_path_handles: dict[str, dict[str, tuple[object, bool]]] = {}
    # Along-path pose triads: the request is (enabled, stride, length_mm) as one tuple so a
    # stride change while enabled is one reconcile; `built` is what the CURRENT handles were
    # created with (None after every scene rebuild, since the poses moved), so a stride changed
    # while the option was OFF is still honoured when it is switched on.
    path_axes_state = {
        "requested": (bool(show_path_axes.value), int(path_axes_stride.value), float(path_axes_length.value)),
        "applied": None,
    }
    path_axes_handles: dict[str, dict[str, list]] = {}   # role -> series -> [FrameHandle]
    path_axes_built = {"key": None}                      # (stride, length_mm) of path_axes_handles
    gripper_nodes: dict = {}  # role -> {"pieces": {piece_name: mesh handle}}
    triad_nodes: dict = {}    # role -> {"prediction": {...}, "target": {...}}
    # Playback cluster nodes: role -> {"eef","wrist","triads",...} for the swept recorded
    # pose. A SEPARATE subtree (/playback/<role>) from the /cluster action-step nodes, so the
    # honest pred-vs-target action view is never overloaded to mean "recorded frame".
    playback_nodes: dict = {}
    # "buttons raise flags; the render loop mutates." Every field here is written under
    # request_lock by a viser callback; ONLY the render loop reads them and owns all scene
    # mutation + the _PlaybackData. `mode` is "idle" (single-sample action view) or "play".
    # `playing` is the run/pause state; `cursor`/`requested_cursor` split the frame the loop
    # last drew from a seek the slider callback asked for; `epoch` invalidates a stale seek
    # after a new rollout. `next_advance_s` is the wall-clock deadline for the next frame.
    playback_state = {
        "mode": "idle",
        "playing": False,
        "requested_cursor": None,   # int frame a slider-seek asked for, else None
        "cursor": 0,                # frame the render loop last drew
        "applied_mode": "idle",     # mode the render loop last reconciled the scene to
        "speed": float(speed_slider.value),
        "loop": bool(loop_checkbox.value),
        "next_advance_s": 0.0,      # monotonic deadline for the next auto-advance
        "epoch": 0,                 # bumped by apply_result so a stale seek is ignored
        # Guard for viser echoing a render-loop slider write back through on_update. Server-
        # side `play_slider.value = k` fires the callback SYNCHRONOUSLY on the writing
        # thread; the render loop sets this True around its own writes so that echo is
        # ignored, while a real client drag (a different thread, serialized by request_lock)
        # never observes it True and so is honoured. See the slider callback.
        "suppress_echo": False,
    }
    # The prepared recorded tail for the CURRENT rollout, rebuilt on each rollout AND on a
    # home-anchor flip (same frame decision as the static scene). Render-thread only.
    playback_data: dict[str, _PlaybackData | None] = {"value": None}
    # Render-thread-only mirror of what the SCENE currently shows, so helpers that don't
    # take the flags as args (update_markers) can gate on it without re-reading state under
    # the lock. Only the render loop writes it, always inside its atomic block.
    view = {"playback_active": False}

    def displayed_suffix() -> str:
        if current_result is None:
            return "; no rollout displayed"
        return (
            f"; displaying sample {current_result.sample.dataset_index}, "
            f"seed {current_result.seed}, {current_result.num_steps} steps"
        )

    def enqueue_rollout(requested_sample: int, requested_seed: int, requested_steps: int) -> None:
        with request_lock:
            request_generation["value"] += 1
            request = _RolloutRequest(
                generation=request_generation["value"],
                sample_index=requested_sample,
                seed=requested_seed,
                num_steps=requested_steps,
            )
            try:
                request_queue.get_nowait()
            except queue.Empty:
                pass
            status.value = f"queued sample {request.sample_index}{displayed_suffix()}"
            request_queue.put_nowait(request)

    @rollout_button.on_click
    def _(_event) -> None:
        with request_lock:
            try:
                requested = (
                    _checked_gui_integer(sample_slider.value, "sample index"),
                    _checked_gui_integer(seed_input.value, "seed"),
                    _checked_gui_integer(steps_input.value, "sampler steps"),
                )
            except ValueError as exc:
                status.value = str(exc)
                return
            enqueue_rollout(*requested)

    @action_step.on_update
    def _(_event) -> None:
        with request_lock:
            step_state["requested"] = int(action_step.value)

    def _force_repose(_event) -> None:
        # A visibility toggle changes no step, so invalidate the shown marker to make
        # the render loop re-pose frames + meshes at the current step next tick.
        with request_lock:
            step_state["shown"] = -1

    show_frames.on_update(_force_repose)
    if overlay is not None:
        show_gripper_mesh.on_update(_force_repose)

    def _force_arrow_repose(_event) -> None:
        # Same flag discipline as _force_repose, plus invalidating the PLAYBACK view: a paused
        # playback redraws only when applied_mode changes, so without this a scale change would
        # move the action-step arrows and leave the swept recorded one at the old length.
        with request_lock:
            step_state["shown"] = -1
            playback_state["applied_mode"] = None

    show_force_arrows.on_update(_force_arrow_repose)
    force_scale_slider.on_update(_force_arrow_repose)

    @show_episode_path.on_update
    def _(_event) -> None:
        # Flag only (like _force_repose): the render loop owns the .visible flip, so the
        # green-line handles are never touched from this viser callback thread.
        with request_lock:
            episode_path_state["requested"] = bool(show_episode_path.value)

    @home_anchor_view.on_update
    def _(_event) -> None:
        # Flag only. Switching frames rebuilds every trajectory line AND re-poses the
        # cluster, both of which are scene mutations the render loop owns.
        with request_lock:
            anchor_state["requested"] = bool(home_anchor_view.value)

    @show_grid.on_update
    def _(_event) -> None:
        with request_lock:
            grid_state["requested"] = bool(show_grid.value)

    def _line_toggle(_event) -> None:
        # Flag only, like the green line: the render loop owns the blue/orange handles too.
        with request_lock:
            line_state["target"]["requested"] = bool(show_target_line.value)
            line_state["prediction"]["requested"] = bool(show_prediction_line.value)

    show_target_line.on_update(_line_toggle)
    show_prediction_line.on_update(_line_toggle)

    def _path_axes_request(_event=None) -> None:
        # One tuple per request so enabled/stride/length changes cannot interleave.
        with request_lock:
            path_axes_state["requested"] = (
                bool(show_path_axes.value),
                max(1, int(path_axes_stride.value)),
                float(path_axes_length.value),
            )

    def _axes_slider_request(_event=None) -> None:
        """A slider move means "draw the axes LIKE THIS", so it ENABLES them if they are off.

        The checkbox visibly ticks itself, so the coupling is shown rather than hidden. The write
        MUST happen with ``request_lock`` released: a server-side ``.value`` set fires that
        handle's own on_update SYNCHRONOUSLY on this thread, and ``request_lock`` is a plain
        (non-reentrant) Lock — taking it around this would deadlock the GUI callback thread."""
        if not bool(show_path_axes.value):
            show_path_axes.value = True     # may re-enter _path_axes_request; no lock held here
        _path_axes_request()

    show_path_axes.on_update(_path_axes_request)
    path_axes_stride.on_update(_axes_slider_request)
    path_axes_length.on_update(_axes_slider_request)

    @play_button.on_click
    def _(_event) -> None:
        # Flag only: the render loop owns the playback cluster + cursor. Entering "play"
        # from "idle" starts at the current cursor (0 on a fresh rollout, or wherever a
        # seek left it); a second click while already playing is a no-op resume.
        with request_lock:
            data = playback_data["value"]
            if data is None:
                play_text.value = "no rollout to play yet"
                return
            # Pressing ▶ at (or past) the last frame replays from the start rather than
            # sitting on a finished tail.
            if playback_state["cursor"] >= data.tail_len - 1:
                playback_state["cursor"] = 0
                playback_state["applied_mode"] = None  # force a redraw of the reset cursor
            playback_state["mode"] = "play"
            playback_state["playing"] = True
            # Advance immediately on the next tick rather than waiting a stale deadline.
            playback_state["next_advance_s"] = 0.0

    @stop_button.on_click
    def _(_event) -> None:
        # Return to the single-sample action-step view. Flag only; the render loop tears
        # down the playback cluster and restores the static scene.
        with request_lock:
            playback_state["mode"] = "idle"
            playback_state["playing"] = False
            playback_state["requested_cursor"] = None

    @play_slider.on_update
    def _(_event) -> None:
        # A drag seeks and pauses (scrubbing to inspect a frame). Server-side .value sets
        # ALSO fire this callback (viser echoes them), so the render loop distinguishes a
        # real user seek from its own auto-advance echo by comparing against the frame it
        # last wrote — see the render loop. Here we only record the request + which epoch it
        # belongs to, so a seek landing after a new rollout is dropped rather than applied to
        # the wrong tail.
        with request_lock:
            if playback_data["value"] is None or playback_state["suppress_echo"]:
                return
            playback_state["requested_cursor"] = int(play_slider.value)

    @speed_slider.on_update
    def _(_event) -> None:
        with request_lock:
            playback_state["speed"] = float(speed_slider.value)

    @loop_checkbox.on_update
    def _(_event) -> None:
        with request_lock:
            playback_state["loop"] = bool(loop_checkbox.value)

    def _ensure_cluster_nodes(role: str) -> dict:
        """Create (once) one role's display cluster: two eef frames, each carrying a wrist
        child with the three collection-viz triads, plus the gripper meshes on the
        prediction.

        The scene graph mirrors the zarr replay's, which is what makes the two views
        directly comparable::

            /cluster/<role>/<prediction|target>          <- driven to the trajectory point
                            └── wrist                     <- at geometry.wrist_offset
                                 ├── tcp / tracker / wrist_yaw_link   (the three triads)
                                 └── grip_{base,finger1,finger2}      (prediction only)

        Parenting the triads under the wrist at their FIXED wrist-relative offsets (rather
        than composing world poses per step) means the whole cluster rides the trajectory
        rigidly and cannot drift from the mesh — the offsets are set once, here."""
        if role in triad_nodes:
            return triad_nodes[role]
        geometry = role_geometry[role]
        entry: dict = {}
        for kind, scale in (("prediction", 1.0), ("target", _TARGET_TRIAD_SCALE)):
            base = f"/cluster/{role}/{kind}"
            eef = server.scene.add_frame(base, show_axes=False, origin_radius=0.0)
            wrist = server.scene.add_frame(
                f"{base}/wrist",
                show_axes=False,
                origin_radius=0.0,
                position=tuple(geometry.wrist_offset),
            )
            triads = {}
            if geometry.is_gripper_role:
                # The three requested triads, and only these — same set, names and relative
                # placement as the collection viz / zarr replay.
                for name, offset in TRIAD_OFFSETS_WRIST.items():
                    style = _TRIAD_STYLE[name]
                    # The tracker triad is the physical VIVE puck, bolted on rotated relative to
                    # the wrist; as a child of the wrist frame it must carry that rotation
                    # (R_ext^T) or its axes render at the WRIST/TCP orientation. tcp and
                    # wrist_yaw_link legitimately share the wrist orientation, so they stay
                    # identity. Mirrors replay_zarr_viz.puck_triad_local_quat / the zarr replay.
                    orientation = (
                        {"wxyz": _wxyz_from_rotmat(geometry.tracker_local_rotmat)}
                        if name == "tracker"
                        else {}
                    )
                    triads[name] = server.scene.add_frame(
                        f"{base}/wrist/{name}",
                        show_axes=True,
                        axes_length=style["axes_length"] * scale,
                        axes_radius=style["axes_radius"] * scale,
                        origin_radius=style["origin_radius"] * scale,
                        position=tuple(offset),
                        **orientation,
                    )
                    server.scene.add_label(f"{base}/wrist/{name}/label", f"{name} ({kind[:4]})")
            else:
                # A body reference (the head puck) has no gripper, so it gets ONE triad at
                # its own stored point. Inventing a "TCP" for it would be a fabricated frame.
                style = _TRIAD_STYLE["tracker"]
                triads["tracker"] = server.scene.add_frame(
                    base + "/point",
                    show_axes=True,
                    axes_length=style["axes_length"] * scale,
                    axes_radius=style["axes_radius"] * scale,
                    origin_radius=style["origin_radius"] * scale,
                )
            entry[kind] = {"eef": eef, "wrist": wrist, "triads": triads}
        triad_nodes[role] = entry

        # Gripper meshes hang under the PREDICTION's wrist frame (one mesh keeps an
        # overlapping pred/target pair readable; the target still shows all three triads).
        if overlay is not None and role in overlay.geoms:
            parent = f"/cluster/{role}/prediction/wrist"
            pieces = {}
            for piece in overlay.geoms[role].pieces():
                pieces[piece.name] = server.scene.add_mesh_trimesh(
                    f"{parent}/grip_{piece.name}",
                    piece.mesh,
                    position=tuple(piece.pos_closed),
                    wxyz=tuple(piece.quat),
                )
            gripper_nodes[role] = {"pieces": pieces}
        return entry

    def _draw_role_force_arrows(role_result, index: int, *, visible: bool, scale: float) -> str | None:
        """Draw one role's target + predicted force arrows at action step ``index``.

        Render thread only. Returns a ``"role: gt 4.2 / pred 3.8 N"`` readout line, or ``None``
        for a role with no F/T sensor (nothing is drawn and nothing is claimed).

        The TARGET arrow is suppressed when that step's force columns are masked — a masked
        wrench is stored as zeros, and drawing a zero-length arrow beside a live predicted one
        would read as "the model over-predicts force here" when the truth is simply unmeasured.
        The PREDICTION is always drawn: the model emits a value at every step, and hiding it
        would hide exactly the hallucinated-force case worth seeing.
        """
        if role_result.predicted_wrench is None:
            return None
        role = role_result.role
        _ensure_cluster_nodes(role)  # arrows are children of these frames
        target_valid = (
            role_result.wrench_valid is not None and bool(role_result.wrench_valid[index, :3].all())
        )
        predicted_n = _draw_force_arrow(
            server,
            f"/cluster/{role}/prediction/force",
            role_result.predicted_wrench[index, :3],
            color=_FORCE_PREDICTION_COLOR,
            scale_m_per_n=scale,
            visible=visible,
        )
        target_n = _draw_force_arrow(
            server,
            f"/cluster/{role}/target/force",
            role_result.target_wrench[index, :3] if target_valid else None,
            color=_FORCE_TARGET_COLOR,
            scale_m_per_n=scale,
            radius_scale=_FORCE_ARROW_TARGET_SCALE,
            visible=visible,
        )
        gt = f"{target_n:.2f}" if target_valid else "masked"
        return f"{role}: gt {gt} / pred {predicted_n:.2f} N"

    def update_markers(result: RolloutResult, index: int) -> None:
        """Drive every role's action-step cluster to the selected step (render thread only).

        Static-cluster visibility is gated on ``not view["playback_active"]`` so that while
        the recorded demonstration is playing, an anchor-flip or toggle that re-invokes this
        function cannot re-show the frozen action-step triads/meshes on top of the moving
        playback cluster (the eef nodes are invisible transform carriers, so only the
        triads + meshes need gating)."""
        index = int(np.clip(index, 0, ACTION_HORIZON - 1))
        static_shown = not view["playback_active"]
        frames_visible = static_shown and bool(show_frames.value)
        mesh_visible = static_shown and overlay is not None and bool(show_gripper_mesh.value)
        arrows_visible = static_shown and bool(show_force_arrows.value)
        arrow_scale = float(force_scale_slider.value) / 1000.0
        arrow_readout: list[str] = []
        for role_result in result.roles:
            role = role_result.role
            poses = display_state.get(role)
            if poses is None:
                continue
            entry = _ensure_cluster_nodes(role)
            target_valid = bool(role_result.pose_valid[index])
            for kind, position, wxyz, visible in (
                ("prediction", poses.predicted_position[index], poses.predicted_wxyz[index], frames_visible),
                (
                    "target",
                    poses.target_position[index],
                    poses.target_wxyz[index],
                    frames_visible and target_valid,
                ),
            ):
                nodes = entry[kind]
                nodes["eef"].position = tuple(position)
                nodes["eef"].wxyz = tuple(wxyz)
                for handle in nodes["triads"].values():
                    handle.visible = visible
            # Jaw open fraction from the PREDICTED gripper width (m -> [0,1]); a role
            # without a gripper channel holds the meshes closed.
            pieces = gripper_nodes.get(role, {}).get("pieces")
            if pieces:
                frac = 0.0
                if role_result.predicted_gripper_m is not None:
                    frac = float(np.clip(role_result.predicted_gripper_m[index] / GRIPPER_WIDTH_REF_M, 0.0, 1.0))
                for piece in overlay.geoms[role].pieces():
                    node = pieces[piece.name]
                    node.position = tuple(piece.pos_at(frac))
                    node.visible = mesh_visible
            # Force arrows hang under the SAME eef frames posed above, so their segments are
            # TCP-local and follow the display rebase for free. Gated on the arrow toggle alone
            # (not on show_frames): the wrench is a separate quantity from the triads, and an
            # operator who hides the axes to see the paths still wants the contact force.
            readout = _draw_role_force_arrows(
                role_result, index, visible=arrows_visible, scale=arrow_scale
            )
            if readout is not None:
                arrow_readout.append(readout)
        force_arrow_text.value = " · ".join(arrow_readout) if arrow_readout else "no F/T role"
        step_state["shown"] = index

    def _ensure_playback_nodes(role: str) -> dict:
        """Create (once) one role's PLAYBACK cluster under ``/playback/<role>``.

        A dedicated subtree, structurally identical to the action-step ``/cluster`` one (eef
        -> wrist -> the three triads + gripper meshes) but green-tinted labels and its own
        nodes, so playing the recorded demonstration never overloads the pred/target action
        view. Built lazily the first time a role is played; hidden (not destroyed) on Stop."""
        if role in playback_nodes:
            return playback_nodes[role]
        geometry = role_geometry[role]
        base = f"/playback/{role}"
        eef = server.scene.add_frame(base, show_axes=False, origin_radius=0.0)
        wrist = server.scene.add_frame(
            f"{base}/wrist", show_axes=False, origin_radius=0.0, position=tuple(geometry.wrist_offset)
        )
        triads = {}
        if geometry.is_gripper_role:
            for name, offset in TRIAD_OFFSETS_WRIST.items():
                style = _TRIAD_STYLE[name]
                # The SAME puck relabel the static cluster carries (see _ensure_cluster_nodes): the
                # tracker triad is a child of the wrist frame and renders at the wrist/TCP
                # orientation unless it is given R_ext^T. This cluster lacked it, so during
                # ▶ Play the tracker axes sat exactly on the TCP's — on every layout, and first
                # noticed on the one-hand wipe_board page where playback is the natural view.
                orientation = (
                    {"wxyz": _wxyz_from_rotmat(geometry.tracker_local_rotmat)}
                    if name == "tracker"
                    else {}
                )
                triads[name] = server.scene.add_frame(
                    f"{base}/wrist/{name}",
                    show_axes=True,
                    axes_length=style["axes_length"],
                    axes_radius=style["axes_radius"],
                    origin_radius=style["origin_radius"],
                    position=tuple(offset),
                    **orientation,
                )
                server.scene.add_label(f"{base}/wrist/{name}/label", f"{name} (rec)")
        else:
            style = _TRIAD_STYLE["tracker"]
            triads["tracker"] = server.scene.add_frame(
                f"{base}/point",
                show_axes=True,
                axes_length=style["axes_length"],
                axes_radius=style["axes_radius"],
                origin_radius=style["origin_radius"],
            )
        pieces = {}
        if overlay is not None and role in overlay.geoms:
            for piece in overlay.geoms[role].pieces():
                pieces[piece.name] = server.scene.add_mesh_trimesh(
                    f"{base}/wrist/grip_{piece.name}",
                    piece.mesh,
                    position=tuple(piece.pos_closed),
                    wxyz=tuple(piece.quat),
                )
        entry = {"eef": eef, "wrist": wrist, "triads": triads, "pieces": pieces}
        playback_nodes[role] = entry
        return entry

    def _hide_playback_cluster() -> None:
        """Hide every playback-cluster node (render thread only).

        Used on Stop and when a role has no playback data. The static action cluster is
        re-shown by the following ``update_markers`` (gated on ``view['playback_active']``),
        so this only needs to blank the playback subtree."""
        for role, entry in playback_nodes.items():
            entry["eef"].visible = False
            for handle in entry["triads"].values():
                handle.visible = False
            for node in entry["pieces"].values():
                node.visible = False
            # The arrow is re-added (not handle-toggled) on every draw, so hiding it means
            # re-adding it hidden — the same idiom _draw_force_arrow uses for a sub-threshold
            # wrench. Leaving it out here would strand the last drawn arrow in the scene after
            # Stop, floating where the recorded pose happened to be.
            _draw_force_arrow(
                server, f"/playback/{role}/force", None,
                color=_FORCE_PLAYBACK_COLOR, scale_m_per_n=FORCE_ARROW_SCALE_M_PER_N, visible=False,
            )

    def _pose_playback_cluster(data: _PlaybackData, cursor: int) -> None:
        """Drive every role's playback cluster to the recorded pose at ``cursor``.

        Render-thread only. Position + orientation come from the rebased recorded tail; the
        mesh jaw is driven by the RECORDED gripper width (there is no prediction here). A
        role left raw (degenerate anchor) still poses — its ``anchored`` flag only affects
        the readout, not whether it draws."""
        cursor = int(np.clip(cursor, 0, data.tail_len - 1))
        frames_visible = bool(show_frames.value)
        mesh_visible = overlay is not None and bool(show_gripper_mesh.value)
        arrows_visible = bool(show_force_arrows.value)
        arrow_scale = float(force_scale_slider.value) / 1000.0
        arrow_readout: list[str] = []
        for role, entry in data.per_role.items():
            if role not in role_geometry:
                continue
            nodes = _ensure_playback_nodes(role)
            nodes["eef"].position = tuple(entry["position"][cursor])
            nodes["eef"].wxyz = tuple(entry["wxyz"][cursor])
            nodes["eef"].visible = True
            for handle in nodes["triads"].values():
                handle.visible = frames_visible
            pieces = nodes["pieces"]
            if pieces:
                frac = 0.0
                gripper_frac = entry.get("gripper_frac")
                if gripper_frac is not None:
                    frac = float(gripper_frac[cursor])
                for piece in overlay.geoms[role].pieces():
                    node = pieces[piece.name]
                    node.position = tuple(piece.pos_at(frac))
                    node.visible = mesh_visible
            # The RECORDED contact force at this frame, on the swept cluster. Same TCP-local
            # parenting as the action-step arrows, so the two views are directly comparable:
            # play to a frame and its green arrow is the ground truth the red pred/target pair
            # at that step is claiming to reproduce.
            wrench = entry.get("wrench")
            valid = entry.get("wrench_valid")
            if wrench is None:
                continue
            measured = valid is None or bool(valid[cursor])
            drawn = _draw_force_arrow(
                server,
                f"/playback/{role}/force",
                wrench[cursor, :3] if measured else None,
                color=_FORCE_PLAYBACK_COLOR,
                scale_m_per_n=arrow_scale,
                visible=arrows_visible,
            )
            arrow_readout.append(f"{role}: rec {drawn:.2f} N" if measured else f"{role}: rec masked")
        if arrow_readout:
            force_arrow_text.value = " · ".join(arrow_readout)

    def _read_episode_tails(result: RolloutResult) -> dict[str, np.ndarray]:
        """Recorded rest-of-episode EE positions per role, in the RAW stored frame.

        Best-effort: a read failure logs and yields no tails, never killing the rollout —
        the offline model comparison does not depend on this overlay."""
        try:
            return session.source.episode_forward_positions(result.sample)
        except Exception as exc:  # noqa: BLE001 - overlay is optional; never kill the viewer
            print(f"[s1-inference] episode path unavailable: {exc!r}", flush=True)
            return {}

    def _build_playback_for_result(result: RolloutResult, home_anchored: bool) -> _PlaybackData | None:
        """Prepare the recorded tail for playback in the current display frame (best-effort).

        Returns ``None`` if the recorded poses cannot be read; playback then reports that
        rather than the viewer dying. Rebuilt on each rollout AND on a home-anchor flip so
        the swept cluster shares the static scene's frame decision exactly."""
        try:
            forward = session.source.episode_forward_poses(result.sample)
        except Exception as exc:  # noqa: BLE001 - playback is optional; never kill the viewer
            print(f"[s1-inference] playback poses unavailable: {exc!r}", flush=True)
            return None
        # Re-derived rather than read off heading_state: this runs on the same rollout, so the
        # two agree by construction, and a stale cached matrix would put the swept cluster in a
        # different frame than the lines — the exact drift this function's docstring warns of.
        world_rotmat, _deg, _source = _heading_alignment(
            result, home_anchored, deploy_anchor is not None, role_geometry)
        return _build_playback_data(
            result, forward, role_geometry, home_anchored, world_rotmat=world_rotmat,
            world_by_role=world_rotmat_by_role(world_rotmat, roles_in_order, recorded_roles))

    def _playback_rgb(cursor: int) -> np.ndarray:
        """Recorded camera frame at ``cursor`` (best-effort ``(H,W,3)`` uint8).

        Read straight off the zarr on the render thread (JXL decode ~1 ms, well under the
        66.7 ms frame budget), so playback needs no prefetch worker. A read failure yields a
        small black frame and logs, rather than stalling the loop."""
        try:
            return session.source.episode_image(current_result.sample, cursor)
        except Exception as exc:  # noqa: BLE001 - image is optional; never kill the loop
            print(f"[s1-inference] playback image unavailable at {cursor}: {exc!r}", flush=True)
            return np.zeros((16, 16, 3), dtype=np.uint8)

    def _draw_playback_frame(cursor: int, *, update_slider: bool) -> None:
        """Pose the playback cluster + RGB + readout at ``cursor`` in one atomic message.

        Render thread only. ``update_slider`` writes the scrubber to follow an auto-advance
        (guarded so viser's synchronous echo of that write is ignored); a user-driven seek
        leaves it False because the slider already holds the value."""
        data = playback_data["value"]
        if data is None:
            return
        cursor = int(np.clip(cursor, 0, data.tail_len - 1))
        with server.atomic():
            _pose_playback_cluster(data, cursor)
            playback_image.image = _playback_rgb(cursor)
            if update_slider:
                with request_lock:
                    playback_state["suppress_echo"] = True
                    try:
                        play_slider.value = cursor
                    finally:
                        playback_state["suppress_echo"] = False
        t_s = float(data.times_s[cursor]) if data.times_s.size else 0.0
        playback_state["cursor"] = cursor
        play_text.value = f"frame {cursor + 1}/{data.tail_len}  t={t_s:.2f}s  x{playback_state['speed']:.2f}"

    def _enter_playback_view() -> None:
        """Hide the static action cluster and mark the scene as playing (render thread only).

        Does NOT draw the playback frame — the caller's advance/seek logic owns that, so the
        frame is drawn exactly once per tick."""
        view["playback_active"] = True
        with server.atomic():
            if current_result is not None:
                # update_markers now gates the static triads/meshes off (view flag True).
                update_markers(current_result, step_state["shown"] if step_state["shown"] >= 0 else 0)

    def _exit_playback_view() -> None:
        """Restore the single-sample action-step scene (render thread only)."""
        view["playback_active"] = False
        with server.atomic():
            _hide_playback_cluster()
            if current_result is not None:
                # Re-show the static cluster at the last action step the scrubber held.
                update_markers(current_result, step_state["requested"])

    def _arm_next_deadline(cursor: int, now: float, speed: float) -> None:
        """Set the wall-clock deadline for the frame AFTER ``cursor``, honouring the real
        camera grid (``times_s``) divided by the speed multiplier. Falls back to the 15 Hz
        nominal period at the tail end where there is no next timestamp."""
        data = playback_data["value"]
        if data is None:
            return
        times = data.times_s
        if cursor + 1 < times.size:
            dt = float(times[cursor + 1] - times[cursor])
        elif times.size >= 2:
            dt = float(times[-1] - times[-2])
        else:
            dt = 1.0 / 15.0
        dt = max(1e-3, dt) / speed
        with request_lock:
            playback_state["next_advance_s"] = now + dt

    def _reconcile_playback() -> None:
        """Advance / seek / start / stop playback, reconciling flags to the scene.

        Render thread only, called once per loop tick. Same discipline as the rest of the
        loop — callbacks only wrote ``playback_state``; every scene mutation happens here.
        Draws at most one playback frame per tick."""
        with request_lock:
            mode = playback_state["mode"]
            applied_mode = playback_state["applied_mode"]
            playing = playback_state["playing"]
            requested_cursor = playback_state["requested_cursor"]
            playback_state["requested_cursor"] = None
            speed = max(0.01, float(playback_state["speed"]))
            loop = bool(playback_state["loop"])
        data = playback_data["value"]

        # A user seek forces the playback view (scrubbing to inspect a frame, even from idle)
        # and pauses auto-advance so the frame holds still.
        if requested_cursor is not None and data is not None:
            with request_lock:
                playback_state["mode"] = "play"
                playback_state["playing"] = False
            mode, playing = "play", False

        entering = mode == "play" and applied_mode != "play"
        if entering:
            _enter_playback_view()
        elif mode == "idle" and applied_mode != "idle":
            _exit_playback_view()
        with request_lock:
            playback_state["applied_mode"] = mode

        if mode != "play" or data is None:
            return

        # 1) A seek this tick wins over auto-advance (draws the sought frame, stays paused).
        if requested_cursor is not None:
            _draw_playback_frame(requested_cursor, update_slider=False)
            return
        # 2) Just entered play (via ▶ or an anchor-flip rebuild): draw the held cursor so the
        #    playback cluster appears, and arm the first deadline if we are actually running.
        if entering:
            cursor = playback_state["cursor"]
            _draw_playback_frame(cursor, update_slider=True)
            if playing:
                _arm_next_deadline(cursor, time.monotonic(), speed)
            return
        if not playing:
            return
        # 3) Auto-advance, paced to the recorded camera grid / speed.
        now = time.monotonic()
        with request_lock:
            deadline = playback_state["next_advance_s"]
        if now < deadline:
            return
        cursor = playback_state["cursor"]
        if cursor >= data.tail_len - 1:
            if loop:
                _draw_playback_frame(0, update_slider=True)
                _arm_next_deadline(0, now, speed)
            else:
                with request_lock:
                    playback_state["playing"] = False
                play_text.value = f"done — {data.tail_len} frames (press ▶ to replay, or enable Loop)"
            return
        cursor += 1
        _draw_playback_frame(cursor, update_slider=True)
        _arm_next_deadline(cursor, now, speed)

    def _clear_path_axes() -> None:
        """Render-thread only. Remove every along-path triad; the set is re-created on demand."""
        for per_series in path_axes_handles.values():
            for handles in per_series.values():
                for handle in handles:
                    handle.remove()
        path_axes_handles.clear()
        path_axes_built["key"] = None

    def _rebuild_path_axes(result: RolloutResult, stride: int, length_mm: float) -> None:
        """Render-thread only. Re-create the strided pose triads along both chunk lines from the
        SAME display_state the lines were drawn from (so they can never sit in another frame).
        The previous set is removed explicitly rather than re-added by name: the set SHRINKS when
        the stride grows, and a stale frame at an old stride would read as a pose the chunk has.
        Created invisible; the caller reconciles visibility against the toggles."""
        _clear_path_axes()
        length = float(length_mm) / 1000.0
        for role_result in result.roles:
            role = role_result.role
            poses = display_state.get(role)
            if poses is None:
                continue
            per_series: dict[str, list] = {}
            for series, positions, wxyzs, valid in (
                ("target", poses.target_position, poses.target_wxyz,
                 np.asarray(role_result.pose_valid, dtype=bool)),
                ("prediction", poses.predicted_position, poses.predicted_wxyz, None),
            ):
                handles = []
                for k in _path_axes_steps(int(positions.shape[0]), stride, valid):
                    handles.append(server.scene.add_frame(
                        f"/trajectory/{role}/{series}_axes/{k}",
                        position=tuple(float(v) for v in positions[k]),
                        wxyz=tuple(float(v) for v in wxyzs[k]),
                        axes_length=length,
                        axes_radius=length * _PATH_AXES_RADIUS_RATIO,
                        origin_radius=0.0,
                        visible=False,
                    ))
                per_series[series] = handles
            path_axes_handles[role] = per_series
        path_axes_built["key"] = (int(stride), float(length_mm))

    def _apply_trajectory_display(result: RolloutResult, lines: dict[str, bool],
                                  axes: tuple[bool, int, float]) -> None:
        """Render-thread only: reconcile the blue/orange line toggles and the along-path triads.
        One function for both because the triads inherit their line's visibility."""
        for per_series in chunk_path_handles.values():
            for series, (handle, had_segments) in per_series.items():
                handle.visible = bool(lines[series]) and had_segments
        enabled, stride, length_mm = axes
        if enabled and path_axes_built["key"] != (int(stride), float(length_mm)):
            _rebuild_path_axes(result, int(stride), float(length_mm))
        for per_series in path_axes_handles.values():
            for series, handles in per_series.items():
                show = bool(enabled) and bool(lines[series])
                for handle in handles:
                    handle.visible = show
        for series, value in lines.items():
            line_state[series]["applied"] = bool(value)
        path_axes_state["applied"] = (bool(enabled), int(stride), float(length_mm))

    def _apply_episode_path_visibility(visible: bool) -> None:
        """Render-thread-only flip of every green line's visibility (honours had_segments)."""
        for handle, had_segments in episode_path_handles.values():
            handle.visible = visible and had_segments
        episode_path_state["applied"] = visible

    def _rebuild_scene(result: RolloutResult, home_anchored: bool) -> None:
        """Recompute every role's display poses and redraw all three trajectory lines.

        Render-thread only. Called on a new rollout AND when the home-anchor toggle flips,
        so the lines and the cluster are always rebuilt together from ONE frame decision —
        the failure mode this structure exists to prevent is a line drawn in one frame while
        the gripper sits in the other."""
        tails = _read_episode_tails(result)
        # ONE heading for this rollout, computed before the per-role loop so every role — and
        # the playback tail rebuilt below — is turned by the identical matrix.
        world_rotmat, heading_deg, heading_source = _heading_alignment(
            result, home_anchored, deploy_anchor is not None, role_geometry)
        heading_state["world"] = world_rotmat
        heading_state["deg"] = heading_deg
        heading_state["source"] = heading_source
        # ONE MATRIX PER SOURCE FRAME, not per role. For every layout but `fivepoint_*` this is
        # the shared matrix for all of them and changes nothing; there, the two FK'd FEET are
        # already in the completion's pelvis frame and must not be turned by the operator's
        # heading. See display.world_rotmat_by_role for the measurement and the invariant.
        world_by_role = world_rotmat_by_role(world_rotmat, roles_in_order, recorded_roles)
        for role_result in result.roles:
            role = role_result.role
            poses = _build_display_poses(
                role_result, tails.get(role), role_geometry[role], home_anchored,
                world_rotmat=world_by_role.get(role),
            )
            display_state[role] = poses
            # Blue = masked GT chunk, orange = prediction. Re-added by name each rebuild,
            # which viser treats as a replace (not a leak); the handles are retained ONLY for
            # the Display toggles' .visible flip — nothing re-poses a line after creation, a
            # frame switch rebuilds them. had_segments mirrors the green line's rule, so a
            # toggle cannot re-show a placeholder line drawn for a fully masked chunk.
            target_valid = role_result.position_valid.all(axis=-1)
            chunk_path_handles[role] = {
                "target": (
                    _draw_path(server, f"/trajectory/{role}/target", poses.target_position,
                               _TARGET_BLUE, target_valid),
                    bool(_segments(poses.target_position, target_valid).size),
                ),
                "prediction": (
                    _draw_path(server, f"/trajectory/{role}/prediction", poses.predicted_position,
                               _PREDICTION_ORANGE),
                    bool(_segments(poses.predicted_position).size),
                ),
            }
            tail = poses.episode_position
            had_segments = tail is not None and tail.shape[0] >= 2
            episode_path_handles[role] = (
                _draw_path(
                    server,
                    f"/trajectory/{role}/episode",
                    tail if had_segments else np.zeros((1, 3), dtype=np.float32),
                    _EPISODE_GREEN,
                ),
                had_segments,
            )
        # Report the frame the scene is ACTUALLY in, per role. A role whose reference
        # rotation was degenerate stays raw even with the checkbox ticked (see
        # _build_display_poses), and that must be visible rather than implied-away. The grid
        # is always drawn (operator keeps it as a spatial reference); the readout says what it
        # means in each frame, because in the anchored frame the scene is rotated to identity
        # so the grid is a fixed REFERENCE PLANE, not the true world floor — vertical motion
        # against it is not physical (a physical lift can render as a descent). In the raw
        # frame it IS the world floor.
        if not home_anchored:
            frame_text.value = "raw absolute aligned-world (strict fidelity); grid = world floor"
        else:
            fell_back = sorted(role for role, poses in display_state.items() if not poses.anchored)
            if fell_back:
                frame_text.value = (
                    f"home-anchored +Z up, EXCEPT {', '.join(fell_back)}: degenerate "
                    "reference rotation, left in the raw frame; grid = reference plane, not world floor"
                )
            elif deploy_anchor is not None:
                # The ready-pose home DOES stand on the grid (its z=0 is the floor the FK put
                # the soles on), so the grid caveat above does not apply here and saying it
                # would be wrong. The heading is named because it is the term that decides LEFT
                # from RIGHT, and a silently un-turned scene mirrors the motion.
                turned = ("scene NOT turned (no chest or hand anchor with a readable yaw): "
                          "left/right is the RECORDED world's, not the body's"
                          if heading_state["world"] is None else
                          f"turned {-heading_state['deg']:+.1f} deg to the operator's forward "
                          f"({heading_state['source']})")
                frame_text.value = (
                    f"G1 READY POSE home (pelvis z {deploy_anchor.pelvis_z:.3f}); "
                    f"{turned}; grid = the robot's floor"
                )
            else:
                frame_text.value = (
                    "home-anchored +Z up (translated; pos/rot errors preserved); grid = "
                    "reference plane, not world floor (vertical vs grid is not physical)"
                )
        # The whole-body completion is re-conditioned HERE, from the very `display_state` the
        # lines above were drawn from — including on a home-anchor flip, because that flip
        # genuinely changes the poses being completed (it is not a camera move). Feeding it
        # anywhere else is how the humanoid would end up in a different frame than the chunk
        # it is supposed to be executing. No-op without --motion-completion.
        mc.on_rollout(result, display_state, session.source.file_attrs, home_anchored)
        # The five-point IK body, from the SAME display_state and for the same reason: a body
        # solved in a different frame than the lines it hangs on is the failure this page exists
        # to make visible. No-op without --keypoint-ik.
        kp.on_rollout(result, display_state, session.source.file_attrs, home_anchored)
        # Rebuild the playback tail in the SAME frame decision as the static scene, so a
        # home-anchor flip mid-playback keeps the swept cluster registered with the lines.
        # Cursor is preserved (the tail length is frame-independent); the render loop re-poses
        # the playback cluster next tick because applied_mode is invalidated below.
        playback_data["value"] = _build_playback_for_result(result, home_anchored)
        # New line handles default visible=True; force the render loop to reconcile them
        # with the current toggle next tick (also covers a toggle flipped mid-inference).
        episode_path_state["applied"] = None
        for series in line_state:
            line_state[series]["applied"] = None
        # The along-path triads were hung on the OLD display poses: drop them, and let the same
        # reconcile re-create them at the current stride if the option is on.
        _clear_path_axes()
        path_axes_state["applied"] = None
        anchor_state["applied"] = home_anchored
        # The cluster must be re-posed into the new frame before the next paint.
        step_state["shown"] = -1
        # Force the playback view to re-pose too (covers an anchor flip while playing).
        playback_state["applied_mode"] = None

    def apply_result(result: RolloutResult) -> None:
        nonlocal current_result
        ref = result.sample
        sample_text.value = (
            f"index={ref.dataset_index}, file={ref.file_index}, episode={ref.global_episode_id} "
            f"(local {ref.file_episode_id}), frame={ref.episode_frame}"
        )
        split_text.value = ref.split
        # Name WHICH sensor is gated on a multi-Bota layout: the two hands tare independently,
        # so "invalid/gated" alone leaves the operator unable to tell a half-gated sample
        # (one hand's window incomplete) from a fully gated one.
        if result.force_history_valid_per_sensor:
            force_text = ", ".join(
                f"{role}={'valid' if flag else 'gated'}"
                for role, flag in zip(session.source.layout.force_roles,
                                      result.force_history_valid_per_sensor, strict=True))
        else:
            force_text = "valid" if result.force_history_valid else "invalid/gated"
        validity_text.value = (
            f"state={'valid' if result.state_valid.all() else 'partly invalid'}, "
            f"force_history={force_text}"
        )
        units = {
            "position_cm": "cm",
            "rotation_geodesic_deg": "deg",
            "gripper_mm": "mm",
            "force_n": "N",
            "torque_nm": "Nm",
        }
        for name, handle in metric_text.items():
            handle.value = _format_metric(float(result.metrics[name]), units[name])
        for handle, image in zip(image_handles, result.conditioning_images, strict=True):
            handle.image = image

        with request_lock:
            home_anchored = bool(anchor_state["requested"])
        _rebuild_scene(result, home_anchored)

        for role_result in result.roles:
            role = role_result.role
            # Charts stay on the RAW stored values: they are a data readout, so they must
            # not move with a display-frame toggle (same discipline as the zarr replay's
            # readouts). The 3D lines above carry the display frame.
            _set_vector_chart(
                charts[role]["position"],
                result.action_times_s,
                role_result.target_position,
                role_result.predicted_position,
                role_result.position_valid,
            )
            if role_result.predicted_gripper_m is not None:
                _set_scalar_chart(
                    charts[role]["gripper"],
                    result.action_times_s,
                    role_result.target_gripper_m,
                    role_result.predicted_gripper_m,
                    role_result.gripper_valid,
                )
            if role_result.predicted_wrench is not None:
                # Charts carry F^cmd, matching the arrows. ONE flip per wrench, applied to
                # target and prediction alike, so the two series stay comparable and the
                # validity mask (which is about measurement, not sign) is untouched.
                target_cmd = cmd_from_meas_wrench(role_result.target_wrench)
                predicted_cmd = cmd_from_meas_wrench(role_result.predicted_wrench)
                _set_vector_chart(
                    charts[role]["force"],
                    result.action_times_s,
                    target_cmd[:, :3],
                    predicted_cmd[:, :3],
                    role_result.wrench_valid[:, :3],
                )
                _set_vector_chart(
                    charts[role]["torque"],
                    result.action_times_s,
                    target_cmd[:, 3:],
                    predicted_cmd[:, 3:],
                    role_result.wrench_valid[:, 3:],
                )
        action_step.value = 0
        with request_lock:
            step_state["requested"] = 0
            step_state["shown"] = -1
            # A new rollout stops any in-progress playback and resets its cursor to the new
            # sample's t0. The tail was rebuilt in _rebuild_scene; size the scrubber to it.
            data = playback_data["value"]
            tail_len = data.tail_len if data is not None else 1
            playback_state["mode"] = "idle"
            playback_state["playing"] = False
            playback_state["cursor"] = 0
            playback_state["requested_cursor"] = None
            playback_state["applied_mode"] = None
            playback_state["epoch"] += 1
            playback_state["suppress_echo"] = True
            try:
                # max must be >= min for viser; a 1-frame tail (min==max==0) still valid.
                play_slider.max = max(1, tail_len - 1)
                play_slider.value = 0
            finally:
                playback_state["suppress_echo"] = False
            duration_s = float(data.times_s[-1]) if data is not None and data.times_s.size else 0.0
            play_text.value = (
                f"idle — {tail_len} recorded frames (~{duration_s:.1f}s)"
                if data is not None
                else "recorded poses unavailable for this sample"
            )
        update_markers(result, 0)
        current_result = result
        status.value = (
            f"ready: {result.algorithm}, sample {ref.dataset_index}, seed {result.seed}, {result.num_steps} steps"
        )

    enqueue_rollout(sample_index, int(seed), int(session.num_steps))
    print(
        f"[s1-inference] http://{host}:{port} - choose an exact dataset sample and run inference",
        flush=True,
    )
    try:
        while True:
            try:
                request = request_queue.get_nowait()
            except queue.Empty:
                request = None
            if request is not None:
                with request_lock:
                    if request.generation != request_generation["value"]:
                        request = None
                    else:
                        status.value = f"running sample {request.sample_index}{displayed_suffix()}"
            if request is not None:
                try:
                    # GIL PRIORITY over the five-point IK worker for the duration of the sample.
                    # That worker is pure Python (yourdfpy FK + a 6x6 solve, thousands of times per
                    # chunk) while this is a Python loop issuing CUDA launches, so the two fight for
                    # one interpreter and the GPU idles waiting for a thread that cannot get it —
                    # measured on the rig page, which has the identical pair, 386 ms -> 3758 ms.
                    # The gate is a no-op without --keypoint-ik (nothing ever calls wait()).
                    with kp.infer_gate.priority():
                        result = session.infer(request.sample_index, seed=request.seed,
                                               num_steps=request.num_steps)
                    with request_lock:
                        if request.generation == request_generation["value"]:
                            with server.atomic():
                                apply_result(result)
                except Exception as exc:  # noqa: BLE001 - surface errors without killing the UI
                    with request_lock:
                        if request.generation == request_generation["value"]:
                            status.value = (
                                f"inference failed for sample {request.sample_index}: {exc!r}{displayed_suffix()}"
                            )
                            print(f"[s1-inference] inference failed: {exc!r}", flush=True)
            # The grid needs no rollout; reconcile it every tick (render thread owns the handle).
            with request_lock:
                grid_requested = grid_state["requested"]
            if grid_requested != grid_state["applied"]:
                grid_handle.visible = bool(grid_requested)
                grid_state["applied"] = bool(grid_requested)
            if current_result is not None:
                with request_lock:
                    anchor_requested = anchor_state["requested"]
                    anchor_applied = anchor_state["applied"]
                    requested_step = step_state["requested"]
                    shown_step = step_state["shown"]
                    episode_requested = episode_path_state["requested"]
                    episode_applied = episode_path_state["applied"]
                    lines_requested = {series: bool(state["requested"]) for series, state in line_state.items()}
                    lines_applied = {series: state["applied"] for series, state in line_state.items()}
                    axes_requested = path_axes_state["requested"]
                    axes_applied = path_axes_state["applied"]
                # A display-frame toggle rebuilds every line AND re-poses the cluster; both
                # must flush in ONE message or the lines briefly render in the new frame with
                # the cluster still in the old. _rebuild_scene sets step_state["shown"] = -1,
                # so the marker re-pose below runs in this same atomic block. A plain
                # scrub-only step change reuses the same block (its atomic is re-entrant).
                if anchor_requested != anchor_applied:
                    with server.atomic():
                        _rebuild_scene(current_result, bool(anchor_requested))
                        update_markers(current_result, requested_step)
                elif requested_step != shown_step:
                    # Scrub only: pred frame, target frame, and gripper meshes update as one
                    # message (no one-frame tearing while dragging the slider).
                    with server.atomic():
                        update_markers(current_result, requested_step)
                # Reconcile the green-line toggle here (render thread owns the handles):
                # the callback only sets episode_path_state["requested"].
                if episode_requested != episode_applied:
                    with server.atomic():
                        _apply_episode_path_visibility(episode_requested)
                # Same for the blue/orange lines and the along-path triads. Runs AFTER a frame
                # rebuild in the same tick (the rebuild cleared the triads and reset `applied`),
                # so the triads are always hung on the poses the lines were just drawn from.
                if lines_requested != lines_applied or axes_requested != axes_applied:
                    with server.atomic():
                        _apply_trajectory_display(current_result, lines_requested, axes_requested)
                _reconcile_playback()
                # Whole-body completion, driven by the SAME action step the cluster is drawn
                # at: poll() queues any window the step has reached (worker thread runs it)
                # and returns True when one has landed. Re-drawn on a landed window OR a
                # scrub, so the humanoid tracks the slider without re-rendering every tick.
                # Both calls are no-ops without --motion-completion.
                mc_landed = mc.poll(requested_step)
                if mc_landed or requested_step != shown_step:
                    with server.atomic():
                        mc.render_at(requested_step)
                # Same contract for the IK body: poll() reports a landed solve, and a scrub
                # re-poses without re-solving (the solve is per ROLLOUT, not per step).
                kp_landed = kp.poll(requested_step)
                if kp_landed or requested_step != shown_step:
                    with server.atomic():
                        kp.render_at(requested_step)
            time.sleep(0.02)
    except KeyboardInterrupt:
        pass
    finally:
        mc.stop()
        kp.stop()
