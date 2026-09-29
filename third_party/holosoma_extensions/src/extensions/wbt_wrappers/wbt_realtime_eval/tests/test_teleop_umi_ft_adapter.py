"""`:8771` (both hands, per-sample, 3-DoF policy force + 6-DoF sidecars) -> UMI's `ft_batch`.

The whole adapter is a re-shaping, and every failure mode is silent — an empty column, a
half-width wrench, or the wrong compensation model, all of which convert cleanly and only
show up as a policy that cannot use force. So the tests are about which curve lands in which
file:

  * `*_raw`  (6-DoF, uncompensated, untared) -> `samples`      -> `ft_wrench.npy`
  * `*_pipe` (6-DoF, gravity-only, tared)    -> `samples_comp` -> `ft_wrench_comp.npy`,
    which is the file `convert_to_umi.py` PREFERS for `robot0_ft_wrench`.
  * the bridge's own `left`/`right` (3-DoF, RNEA) must reach NEITHER: it is 3 of the 6 columns
    UMI needs, and a zero-padded torque would read as "this sensor measured no torque".

Loaded by path: the adapter lives in the force_deploy sandbox next to the launchers it serves,
not in this package.
"""

from __future__ import annotations

import importlib.util
import os
import pathlib
import sys

import pytest


def _adapter():
    """`teleop_umi_ft_adapter` out of force_deploy, or skip by name."""
    if "teleop_umi_ft_adapter" in sys.modules:
        return sys.modules["teleop_umi_ft_adapter"]
    # tests/ -> wbt_realtime_eval -> wbt_wrappers -> extensions -> src -> <repo root>
    root = pathlib.Path(__file__).resolve().parents[5]
    path = root / "force_deploy" / "teleop_umi_ft_adapter.py"
    if not path.is_file():
        pytest.skip(f"adapter not found at {path}")
    pytest.importorskip("websockets")
    spec = importlib.util.spec_from_file_location("teleop_umi_ft_adapter", os.fspath(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["teleop_umi_ft_adapter"] = mod
    spec.loader.exec_module(mod)
    return mod


_W6 = [1.0, 2.0, 3.0, 0.1, 0.2, 0.3]


def test_a_six_dof_wrench_becomes_a_timestamped_row() -> None:
    A = _adapter()
    assert A._row(12.5, _W6) == [12.5, *_W6]  # noqa: SLF001


def test_a_three_dof_force_is_refused_not_padded() -> None:
    """The bridge's `left`/`right` keys are FORCE ONLY. Padding them would fabricate torque."""
    A = _adapter()
    assert A._row(1.0, [1.0, 2.0, 3.0]) is None  # noqa: SLF001


def test_an_absent_sensor_is_none_not_zero() -> None:
    """`null` on the wire means "no sensor configured", which must stay distinguishable from
    "the sensor read zero" — the same rule UMI's raw layout enforces on disk."""
    A = _adapter()
    assert A._row(1.0, None) is None  # noqa: SLF001


def test_extra_columns_are_truncated_to_a_wrench() -> None:
    """Defensive, and cheap: a longer vector is still a wrench in its first six columns."""
    A = _adapter()
    assert A._row(0.0, [*_W6, 99.0]) == [0.0, *_W6]  # noqa: SLF001


def test_the_two_hands_have_independent_rings() -> None:
    """One stream per hand, never merged: the two Botas have independent tares and independent
    `ft_sign_verified`, and UMI's recorder latches that provenance PER ROLE."""
    A = _adapter()
    left, right = A.HandStream("left"), A.HandStream("right")
    left.raw.append([0.0, *_W6])
    assert len(left.take()[0]) == 1
    assert right.take() == ([], [])


def test_take_drains_so_a_batch_is_never_resent() -> None:
    """A duplicated sample would put two readings at one timestamp on the recorded timeline."""
    A = _adapter()
    st = A.HandStream("left")
    st.raw.append([0.0, *_W6])
    st.comp.append([0.0, *_W6])
    raw, comp = st.take()
    assert (len(raw), len(comp)) == (1, 1)
    assert st.take() == ([], [])


def test_the_republished_ports_avoid_the_policy_record_tap() -> None:
    """8779 is `RECORD=1`'s policy tap in 3_run_policy_vive_force.sh — and it is ALSO
    `recording_format.FT_RIGHT_WS_PORT_DEFAULT`. Binding it here would make the two recorders
    fight for one port, with the loser reporting "no frames"."""
    A = _adapter()
    assert A.RIGHT_PORT_DEFAULT != 8779
    assert A.LEFT_PORT_DEFAULT != 8779
    assert A.LEFT_PORT_DEFAULT != A.RIGHT_PORT_DEFAULT


def test_missing_streams_can_be_made_fatal() -> None:
    """`--require` exists because an empty force column is only visible after conversion, once
    the demonstrations have already been performed."""
    A = _adapter()
    with pytest.raises(SystemExit):
        A._missing("raw", "FT_EMIT_RAW=1", "ft_wrench.npy", True)  # noqa: SLF001
    A._missing("raw", "FT_EMIT_RAW=1", "ft_wrench.npy", False)  # warns, returns  # noqa: SLF001


def test_the_batch_frame_is_what_the_recorder_parses() -> None:
    """Exact key names: `recorder.on_ft_batch` reads `samples`, `samples_comp`, `compensated`
    and `ft_comp_algo`, and ignores anything else — so a renamed key records nothing."""
    A = _adapter()
    f = A.batch_frame([[0.0, *_W6]], [[0.0, *_W6]], "bota")
    assert f["type"] == "ft_batch"
    assert f["samples"] == [[0.0, *_W6]]
    assert f["samples_comp"] == [[0.0, *_W6]]
    assert f["compensated"] is True
    assert f["ft_comp_algo"] == "bota"
    assert all(len(s) == 7 for s in f["samples"]), "recorder drops rows shorter than 7"


def test_no_compensated_sidecar_means_no_algo_and_compensated_false() -> None:
    """A `--pipeline`-less bridge must produce a RAW-labelled take, not a compensated one with
    an empty sidecar: the recorder latches `compensated` from the FIRST batch and `attrs.
    ft_compensated` follows it into the zarr."""
    A = _adapter()
    f = A.batch_frame([[0.0, *_W6]], [], "bota")
    assert f["compensated"] is False
    assert "samples_comp" not in f
    assert "ft_comp_algo" not in f
