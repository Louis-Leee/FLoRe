#!/usr/bin/env python3
"""Merge multiple holosoma-format NPZ motion files into a single NPZ file.

The holosoma training framework (MotionConfig) only accepts a single .npz motion
file. This script concatenates multiple motion clips along the time dimension,
inserting interpolated reset segments between clips to avoid discontinuities.

NPZ format (as consumed by holosoma MotionLoader):
    fps            : scalar
    joint_pos      : [T, 36]   (first 7 are root DOFs: xyz + wxyz)
    joint_vel      : [T, 35]   (first 6 are root DOFs: lin_vel + ang_vel)
    body_pos_w     : [T, B, 3]
    body_quat_w    : [T, B, 4] (wxyz)
    body_lin_vel_w : [T, B, 3] (optional)
    body_ang_vel_w : [T, B, 3] (optional)
    joint_names    : [29]      string array
    body_names     : [B]       string array

Usage examples:
    # Merge all motions in a directory
    python merge_motions.py /path/to/motion_data/lafan1_holosoma/ merged_all.npz

    # Merge only walk motions
    python merge_motions.py /path/to/motion_data/lafan1_holosoma/ merged_walks.npz --category walk

    # Merge specific files
    python merge_motions.py file1.npz file2.npz file3.npz -o merged.npz
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Quaternion utilities (wxyz convention, numpy)
# ---------------------------------------------------------------------------


def _quat_normalize(q: np.ndarray) -> np.ndarray:
    """Normalize quaternion(s) in wxyz format."""
    norm = np.linalg.norm(q, axis=-1, keepdims=True)
    norm = np.where(norm < 1e-10, 1.0, norm)
    return q / norm


def _quat_dot(q0: np.ndarray, q1: np.ndarray) -> np.ndarray:
    """Dot product between quaternion arrays, returned with shape (..., 1)."""
    return np.sum(q0 * q1, axis=-1, keepdims=True)


def _slerp(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
    """Spherical linear interpolation between quaternion arrays (wxyz).

    Args:
        q0: Start quaternions, shape (..., 4).
        q1: End quaternions, shape (..., 4).
        t:  Interpolation factor in [0, 1].

    Returns:
        Interpolated quaternions, same shape as inputs.
    """
    q0 = _quat_normalize(q0)
    q1 = _quat_normalize(q1)

    dot = _quat_dot(q0, q1)

    # Ensure shortest path.
    neg_mask = dot < 0
    q1 = np.where(neg_mask, -q1, q1)
    dot = np.abs(dot)

    # For nearly identical quaternions, fall back to linear interpolation.
    linear_mask = dot > 0.9995
    dot = np.clip(dot, -1.0, 1.0)
    theta = np.arccos(dot)
    sin_theta = np.sin(theta)

    # Avoid division by zero.
    safe_sin = np.where(sin_theta < 1e-10, 1.0, sin_theta)
    s0 = np.sin((1.0 - t) * theta) / safe_sin
    s1 = np.sin(t * theta) / safe_sin

    result = s0 * q0 + s1 * q1

    # Where linear_mask is True, use simple lerp + normalize.
    lerp_result = _quat_normalize((1.0 - t) * q0 + t * q1)
    result = np.where(linear_mask, lerp_result, result)

    return _quat_normalize(result)


# ---------------------------------------------------------------------------
# Data loading and validation
# ---------------------------------------------------------------------------

# Keys that are concatenated along the time axis.
TIME_KEYS = ["joint_pos", "joint_vel", "body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w"]

# Keys that must be identical across all clips.
METADATA_KEYS = ["fps", "joint_names", "body_names"]


def load_clip(path: Path) -> Dict[str, np.ndarray]:
    """Load a single NPZ motion clip and return its arrays as a dict."""
    with np.load(path, allow_pickle=True) as data:
        clip: Dict[str, np.ndarray] = {}
        for key in list(data.keys()):
            clip[key] = data[key]
    return clip


def validate_clips(clips: List[Dict[str, np.ndarray]], paths: List[Path]) -> None:
    """Validate that all clips share the same metadata and compatible shapes.

    Raises ValueError with a descriptive message on any mismatch.
    """
    if len(clips) < 2:
        return

    ref = clips[0]
    ref_path = paths[0]

    for i, (clip, path) in enumerate(zip(clips[1:], paths[1:]), start=1):
        # fps must match exactly.
        if not np.array_equal(ref["fps"], clip["fps"]):
            raise ValueError(f"FPS mismatch: {ref_path} has fps={ref['fps']}, but {path} has fps={clip['fps']}.")

        # joint_names and body_names must match.
        for key in ("joint_names", "body_names"):
            if key in ref and key in clip:
                if not np.array_equal(ref[key], clip[key]):
                    raise ValueError(
                        f"{key} mismatch between {ref_path} and {path}.\n"
                        f"  Reference: {ref[key].tolist()}\n"
                        f"  Got:       {clip[key].tolist()}"
                    )

        # Shapes along non-time dimensions must match for time-series arrays.
        for key in TIME_KEYS:
            if key in ref and key in clip:
                ref_shape = ref[key].shape[1:]
                clip_shape = clip[key].shape[1:]
                if ref_shape != clip_shape:
                    raise ValueError(
                        f"Shape mismatch for '{key}': {ref_path} has shape "
                        f"(T, {ref_shape}), but {path} has shape (T, {clip_shape})."
                    )


# ---------------------------------------------------------------------------
# Reset segment generation
# ---------------------------------------------------------------------------


def build_reset_segment(
    clip_end: Dict[str, np.ndarray], clip_start: Dict[str, np.ndarray], num_frames: int
) -> Dict[str, np.ndarray]:
    """Build an interpolated reset segment between two clips.

    Linearly interpolates positions and velocities; spherically interpolates
    quaternions. The segment has exactly ``num_frames`` frames, starting one
    step after the end of the previous clip and ending one step before the
    start of the next clip (exclusive on both endpoints to avoid duplicating
    boundary frames).

    Args:
        clip_end:   Dict of arrays representing the last frame of the previous clip.
        clip_start: Dict of arrays representing the first frame of the next clip.
        num_frames: Number of interpolation frames to generate.

    Returns:
        Dict of arrays with shape (num_frames, ...) for each time-series key.
    """
    segment: Dict[str, np.ndarray] = {}

    # Alphas from just past 0 to just before 1 (exclusive endpoints).
    alphas = np.linspace(0.0, 1.0, num_frames + 2)[1:-1]

    for key in TIME_KEYS:
        if key not in clip_end or key not in clip_start:
            continue

        start_frame = clip_end[key]  # shape: (non-time dims)
        end_frame = clip_start[key]  # shape: (non-time dims)

        if "quat" in key:
            # Spherical interpolation for quaternions (wxyz).
            frames = []
            for alpha in alphas:
                frames.append(_slerp(start_frame, end_frame, alpha))
            segment[key] = np.stack(frames, axis=0)
        else:
            # Linear interpolation for everything else.
            frames = []
            for alpha in alphas:
                frames.append(start_frame + alpha * (end_frame - start_frame))
            segment[key] = np.stack(frames, axis=0)

    return segment


def extract_frame(clip: Dict[str, np.ndarray], index: int) -> Dict[str, np.ndarray]:
    """Extract a single frame from a clip, returning arrays without the time dim."""
    frame: Dict[str, np.ndarray] = {}
    for key in TIME_KEYS:
        if key in clip:
            frame[key] = clip[key][index]
    return frame


# ---------------------------------------------------------------------------
# Merging logic
# ---------------------------------------------------------------------------


def merge_clips(clips: List[Dict[str, np.ndarray]], reset_frames: int = 50) -> Dict[str, np.ndarray]:
    """Merge multiple clips into a single motion dict.

    Clips are concatenated along the time dimension with interpolated reset
    segments inserted between consecutive clips.

    Args:
        clips:        List of clip dicts as returned by ``load_clip``.
        reset_frames: Number of interpolation frames between clips.

    Returns:
        Merged clip dict ready to be saved as NPZ.
    """
    if len(clips) == 1:
        return clips[0]

    # Accumulate per-key lists of arrays to concatenate.
    parts: Dict[str, List[np.ndarray]] = {key: [] for key in TIME_KEYS}

    for i, clip in enumerate(clips):
        # Append the clip itself.
        for key in TIME_KEYS:
            if key in clip:
                parts[key].append(clip[key])

        # Insert a reset segment between clips (not after the last one).
        if i < len(clips) - 1 and reset_frames > 0:
            end_frame = extract_frame(clip, -1)
            start_frame = extract_frame(clips[i + 1], 0)
            segment = build_reset_segment(end_frame, start_frame, reset_frames)
            for key in TIME_KEYS:
                if key in segment:
                    parts[key].append(segment[key])

    # Build the merged dict.
    merged: Dict[str, np.ndarray] = {}

    # Copy metadata from the first clip.
    ref = clips[0]
    for key in METADATA_KEYS:
        if key in ref:
            merged[key] = ref[key]

    # Concatenate time-series arrays.
    for key in TIME_KEYS:
        if parts[key]:
            merged[key] = np.concatenate(parts[key], axis=0)

    return merged


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------


def discover_files(inputs: List[str], category: Optional[str] = None) -> List[Path]:
    """Resolve input arguments to a sorted list of NPZ file paths.

    If a single directory is given, all .npz files in it are collected
    (optionally filtered by ``category`` prefix). Otherwise, explicit file
    paths are returned as-is.
    """
    paths: List[Path] = []

    if len(inputs) == 1 and Path(inputs[0]).is_dir():
        directory = Path(inputs[0])
        pattern = f"{category}*.npz" if category else "*.npz"
        paths = sorted(directory.glob(pattern))
        if not paths:
            raise FileNotFoundError(f"No .npz files matching pattern '{pattern}' found in {directory}.")
    else:
        for p in inputs:
            path = Path(p)
            if not path.exists():
                raise FileNotFoundError(f"File not found: {path}")
            if not path.suffix == ".npz":
                raise ValueError(f"Expected .npz file, got: {path}")
            paths.append(path)

    if category and not (len(inputs) == 1 and Path(inputs[0]).is_dir()):
        paths = [p for p in paths if p.stem.startswith(category)]
        if not paths:
            raise FileNotFoundError(f"No files matching category prefix '{category}' among the provided files.")

    return paths


# ---------------------------------------------------------------------------
# Summary reporting
# ---------------------------------------------------------------------------


def print_summary(
    paths: List[Path],
    clips: List[Dict[str, np.ndarray]],
    merged: Dict[str, np.ndarray],
    reset_frames: int,
    output_path: Path,
) -> None:
    """Print a human-readable summary of the merge operation."""
    fps = float(merged["fps"])
    total_frames = int(merged["joint_pos"].shape[0])
    total_duration = total_frames / fps

    print("\n" + "=" * 70)
    print("  Motion Merge Summary")
    print("=" * 70)
    print(f"  Output file : {output_path}")
    print(f"  FPS         : {fps:.1f}")
    print(f"  Reset frames: {reset_frames} ({reset_frames / fps:.2f}s between clips)")
    print("-" * 70)
    print(f"  {'Clip':<40s} {'Frames':>8s} {'Duration':>10s}")
    print("-" * 70)

    for path, clip in zip(paths, clips):
        frames = clip["joint_pos"].shape[0]
        duration = frames / fps
        print(f"  {path.name:<40s} {frames:>8d} {duration:>9.2f}s")

    print("-" * 70)

    num_resets = len(clips) - 1
    reset_total = num_resets * reset_frames
    clip_total = sum(c["joint_pos"].shape[0] for c in clips)

    print(f"  {'Clip frames total':<40s} {clip_total:>8d} {clip_total / fps:>9.2f}s")
    if num_resets > 0:
        print(f"  {'Reset segments (' + str(num_resets) + ')':<40s} {reset_total:>8d} {reset_total / fps:>9.2f}s")
    print(f"  {'MERGED TOTAL':<40s} {total_frames:>8d} {total_duration:>9.2f}s")

    # Shape summary.
    print("-" * 70)
    for key in TIME_KEYS:
        if key in merged:
            print(f"  {key:<20s} : {merged[key].shape}")
    print("=" * 70 + "\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge multiple holosoma-format NPZ motion files into one.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  # Merge all motions in a directory\n"
            "  python merge_motions.py /path/to/motion_data/lafan1_holosoma/ merged_all.npz\n\n"
            "  # Merge only walk motions\n"
            "  python merge_motions.py /path/to/motion_data/lafan1_holosoma/ merged_walks.npz --category walk\n\n"
            "  # Merge specific files\n"
            "  python merge_motions.py file1.npz file2.npz file3.npz -o merged.npz\n"
        ),
    )

    parser.add_argument(
        "inputs",
        nargs="+",
        help=(
            "Input NPZ files or a single directory containing NPZ files. "
            "When a directory is given, the second positional argument is "
            "treated as the output path."
        ),
    )
    parser.add_argument("-o", "--output", type=str, default=None, help="Output path for the merged NPZ file.")
    parser.add_argument(
        "--category",
        type=str,
        default=None,
        help=(
            "Filter NPZ files by category prefix (e.g., --category walk "
            "matches walk*.npz). Only applies when a directory is given."
        ),
    )
    parser.add_argument(
        "--reset-frames", type=int, default=50, help="Number of interpolation frames between clips (default: 50)."
    )

    args = parser.parse_args(argv)

    # Resolve output path from positional args if -o is not given.
    # Convention: if the first arg is a directory, the second positional arg
    # is the output path.
    if args.output is None:
        if len(args.inputs) >= 2 and Path(args.inputs[0]).is_dir():
            args.output = args.inputs[1]
            args.inputs = [args.inputs[0]]
        else:
            parser.error(
                "Output path is required. Use -o/--output or provide the output "
                "as the second positional argument after a directory."
            )

    return args


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)

    output_path = Path(args.output)
    reset_frames: int = args.reset_frames
    category: Optional[str] = args.category

    # Discover input files.
    paths = discover_files(args.inputs, category=category)
    print(f"Found {len(paths)} motion file(s) to merge.")

    # Load all clips.
    clips = [load_clip(p) for p in paths]
    print(f"Loaded {len(clips)} clip(s).")

    # Validate compatibility.
    validate_clips(clips, paths)
    print("Validation passed: all clips are compatible.")

    # Merge.
    merged = merge_clips(clips, reset_frames=reset_frames)

    # Save.
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(str(output_path), **merged)
    print(f"Saved merged motion to {output_path}")

    # Summary.
    print_summary(paths, clips, merged, reset_frames, output_path)


if __name__ == "__main__":
    main()
