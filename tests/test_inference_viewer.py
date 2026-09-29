"""Pure-logic pieces of the offline inference viewer's Display controls (no viser needed).

The viewer module imports viser lazily, so its helpers are testable headlessly. The along-path
pose triads are the one new piece with arithmetic in it: WHICH steps get a triad for a given
stride, and how a masked GT step is treated.
"""
from __future__ import annotations

import numpy as np
import pytest

from fuji.s1_policy.inference.viewer import _path_axes_steps


def test_path_axes_steps_stride_and_last_step():
    # 0, 5, ..., 45 plus the last step 49, so the chunk's end orientation is always drawn.
    assert _path_axes_steps(50, 5) == [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 49]
    # A stride that lands exactly on the last step does not duplicate it.
    assert _path_axes_steps(50, 7) == [0, 7, 14, 21, 28, 35, 42, 49]
    # Stride 1 = every step, once.
    assert _path_axes_steps(4, 1) == [0, 1, 2, 3]


def test_path_axes_steps_degenerate_strides_and_horizons():
    # A stride past the horizon still gives both ends of the chunk.
    assert _path_axes_steps(50, 200) == [0, 49]
    # stride < 1 is read as 1 rather than raising or looping forever.
    assert _path_axes_steps(3, 0) == [0, 1, 2]
    assert _path_axes_steps(3, -4) == [0, 1, 2]
    # A one-step chunk has exactly one triad; an empty one has none.
    assert _path_axes_steps(1, 5) == [0]
    assert _path_axes_steps(0, 5) == []


def test_path_axes_steps_skip_masked_gt_steps():
    valid = np.ones(50, dtype=bool)
    valid[40:] = False  # the chunk runs past the episode end: steps 40..49 are not measurements
    steps = _path_axes_steps(50, 10, valid)
    assert steps == [0, 10, 20, 30]  # 40 and the appended last step 49 are both masked out
    # The prediction passes valid=None and keeps every strided step.
    assert _path_axes_steps(50, 10, None) == [0, 10, 20, 30, 40, 49]


@pytest.mark.parametrize("horizon,stride", [(30, 5), (50, 16), (100, 30), (200, 50)])
def test_path_axes_steps_are_unique_sorted_and_in_range(horizon, stride):
    steps = _path_axes_steps(horizon, stride)
    assert steps == sorted(set(steps))
    assert steps[0] == 0 and steps[-1] == horizon - 1
    assert all(0 <= k < horizon for k in steps)
