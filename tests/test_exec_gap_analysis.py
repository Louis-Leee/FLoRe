"""Tests for the offline execution-gap analysis (``scripts/measure_exec_gap.py`` M5-M8 and
``scripts/exec_gap_corrector.py`` C1-C5).

These are analysis scripts, not serving code, so what has to be protected is the ARITHMETIC — a
silently wrong estimator here does not crash a robot, it publishes a number someone then designs a
corrector around. Each test below pins one property that a real reading of this data depended on:

1. ``_chunk_step_path`` measures the commanded arc along the DECODED CHUNK at step resolution.
   Measuring it along the sampled ticks instead short-cuts the arc and inflates the deficit; that
   bug is what made the closure identity appear to fail by a factor of 1.4.
2. The increment gain recovers a known gain, and is INVARIANT to the anchor, because differencing
   inside a chunk cancels the anchor exactly. That invariance is the estimator's whole claim.
3. ``fit_gain_lag`` separates a DC deficit from a lag. If it cannot, "gain vs lag" collapses back
   into whichever one the chosen timescale happens to expose.
4. ``fit_G`` recovers scalar / diagonal / full gains exactly on noiseless data, including the
   cross-axis coupling that turned out to carry 14 of the 19 recoverable percent.
5. The residual split is orthogonal, so along and normal components cannot double-count.
6. Pairs never straddle a chunk boundary, and the eps clamp bounds the carried anchor lead.
7. The wire-force reader undoes ``F_cmd = -F_meas``; comparing the two raw would report twice the
   force as an error.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parents[1]
_SCRIPTS = _ROOT / "scripts"


def _load(name: str):
    """Import a script by path. ``exec_gap_corrector`` imports ``measure_exec_gap``, so
    ``scripts/`` has to be importable before it is loaded."""
    for p in (str(_SCRIPTS), str(_ROOT)):
        if p not in sys.path:
            sys.path.insert(0, p)
    spec = importlib.util.spec_from_file_location(name, _SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


meg = _load("measure_exec_gap")
egc = _load("exec_gap_corrector")


# ------------------------------------------------------------------------------------ the path
def test_chunk_step_path_is_arc_length_along_the_decoded_chunk():
    """A straight chunk of 10 steps of 2 mm has arc length 2*k mm at step k, regardless of which
    ticks happened to sample it. Sampling every third tick must not change the answer."""
    d = np.zeros((10, 3))
    d[:, 0] = np.arange(10) * 0.002
    decoded = {7: d}
    steps = np.array([0, 3, 6, 9])
    path, _ = meg._chunk_step_path(decoded, np.full(4, 7), steps)
    assert np.allclose(path, steps * 0.002)


def test_chunk_step_path_restarts_at_every_chunk():
    """Path is measured from each chunk's OWN step 0; a running total across the boundary would
    make the deficit look like it survives the re-anchor when it does not."""
    d = np.zeros((5, 3))
    d[:, 0] = np.arange(5) * 0.01
    decoded = {0: d, 1: d}
    ch = np.array([0, 0, 0, 1, 1, 1])
    st = np.array([0, 1, 2, 0, 1, 2])
    path, _ = meg._chunk_step_path(decoded, ch, st)
    assert np.allclose(path, [0.0, 0.01, 0.02, 0.0, 0.01, 0.02])


def test_step_disp_is_per_step_index_not_per_tick():
    """The lever arm of the closure identity. Ten 2 mm steps must average 2 mm even when only
    every third tick was recorded — averaging over tick pairs would report ~6 mm."""
    d = np.zeros((10, 3))
    d[:, 0] = np.arange(10) * 0.002
    _, disp = meg._chunk_step_path({0: d}, np.zeros(4, dtype=int), np.array([0, 3, 6, 9]))
    assert disp.size == 9
    assert np.allclose(disp, 0.002)


def test_step_disp_ignores_the_unexecuted_tail():
    """An interrupted chunk's tail was never asked for, so it must not enter the average."""
    d = np.zeros((10, 3))
    d[:, 0] = np.arange(10) * 0.002
    d[5:, 0] = 0.008 + np.arange(5) * 0.05  # a wild tail nobody executed
    _, disp = meg._chunk_step_path({0: d}, np.zeros(3, dtype=int), np.array([0, 1, 2]))
    assert disp.size == 2
    assert np.allclose(disp, 0.002)


# ------------------------------------------------------------------------------- the gain fits
def _synthetic_take(g: np.ndarray, n: int = 60, n_chunks: int = 3, seed: int = 0,
                    anchor_jump: float = 0.0) -> dict:
    """A take whose measured increments are exactly ``dcmd @ g.T``, in the packed group format."""
    rng = np.random.default_rng(seed)
    cmd = np.zeros((n, 3))
    pos = np.zeros((n, 3))
    chunk = np.repeat(np.arange(n_chunks), n // n_chunks)[:n]
    step = np.concatenate([np.arange((chunk == c).sum()) for c in range(n_chunks)])
    for c in range(n_chunks):
        m = chunk == c
        steps = rng.normal(0, 0.004, size=(m.sum(), 3))
        steps[0] = 0.0
        off = np.cumsum(steps, axis=0)
        a = np.full(3, anchor_jump * c)
        cmd[m] = a + off
        pos[m] = a + off @ np.asarray(g).T
    ok = np.ones(n, dtype=bool)
    u, cnt = np.unique(chunk, return_counts=True)
    nrows = np.array([dict(zip(u.tolist(), cnt.tolist()))[int(x)] for x in chunk])
    return dict(raw=(cmd, pos, step, chunk, ok, np.zeros(n)), chunk_nrows=nrows,
                dt=1 / 30, step=step, chunk=chunk)


@pytest.mark.parametrize("g", [0.5, 0.61, 1.0])
def test_increment_gain_recovers_a_known_scalar_gain(g):
    groups = [_synthetic_take(np.eye(3) * g, seed=s) for s in range(4)]
    r = meg.gain_increment(groups, 1)
    assert r is not None
    assert r[0] == pytest.approx(g, abs=1e-6)


def test_increment_gain_is_invariant_to_the_anchor():
    """The estimator's central claim: within a chunk every row shares one anchor, so the anchor
    cancels in the difference and its measurement noise cannot bias the slope. A 20 cm per-chunk
    anchor jump must not move the answer at all."""
    plain = [_synthetic_take(np.eye(3) * 0.61, seed=s) for s in range(4)]
    shifted = [_synthetic_take(np.eye(3) * 0.61, seed=s, anchor_jump=0.2) for s in range(4)]
    assert meg.gain_increment(plain, 1)[0] == pytest.approx(meg.gain_increment(shifted, 1)[0],
                                                            abs=1e-9)


def test_within_chunk_pairs_never_straddle_a_boundary():
    g = _synthetic_take(np.eye(3) * 0.6, n=60, n_chunks=3)
    cmd, pos, step, chunk, ok, _ = g["raw"]
    for w in (1, 5, 11):
        n_before = 0
        lo = np.arange(0, cmd.shape[0] - w)
        for a in lo:
            if chunk[a] == chunk[a + w]:
                n_before += 1
        dc, _, _, _ = meg.within_chunk_pairs(g, w)
        assert dc.shape[0] == n_before


def test_min_rows_filter_holds_the_population_fixed():
    """Pooling short and long chunks confounds the timescale sweep; the filter is what removes it."""
    g = _synthetic_take(np.eye(3) * 0.6, n=60, n_chunks=3)  # 20 rows per chunk
    assert meg.within_chunk_pairs(g, 1, 0)[0].shape[0] > 0
    assert meg.within_chunk_pairs(g, 1, 21)[0].shape[0] == 0


# ------------------------------------------------------------------------------- gain vs lag
def _fit_on_curve(curve: np.ndarray, dt: float):
    """Run ``fit_gain_lag`` against a SYNTHESISED gain-vs-window curve.

    The fit reads its curve through ``gain_vs_window``, so substituting that is how a known
    ``(g_DC, tau)`` can be put in and checked on the way out. Restored in a ``finally`` because
    these tests share a module instance.
    """
    orig = meg.gain_vs_window
    try:
        meg.gain_vs_window = lambda groups, min_rows=0: curve
        return meg.fit_gain_lag([object()], dt)
    finally:
        meg.gain_vs_window = orig


def test_lag_shape_limits():
    """Over a vanishing window a lagged plant realizes nothing; over an infinite one, everything."""
    assert meg._lag_shape(np.array([1e-6]), 1.0)[0] == pytest.approx(0.0, abs=1e-5)
    assert meg._lag_shape(np.array([1e6]), 1.0)[0] == pytest.approx(1.0, abs=1e-5)
    assert 0.0 < meg._lag_shape(np.array([0.05]), 0.05)[0] < 1.0


def test_lag_shape_is_monotone_in_window():
    t = np.geomspace(1e-3, 10, 50)
    v = meg._lag_shape(t, 0.02)
    assert np.all(np.diff(v) > 0)


def test_fit_gain_lag_separates_a_dc_deficit_from_a_lag():
    """Synthesise the curve a plant with g_DC=0.61 and tau=17 ms would produce and check both
    parameters come back. Without this the two are not identifiable and the whole M5 reading —
    that BOTH are present — would be unsupported."""
    dt, g_dc, tau = 0.0536, 0.61, 0.017
    t = np.asarray(meg.PAIR_WINDOWS, dtype=float) * dt
    out = _fit_on_curve(g_dc * meg._lag_shape(t, tau), dt)
    assert out is not None
    assert out[0] == pytest.approx(g_dc, rel=0.05)
    assert out[1] == pytest.approx(tau, rel=0.3)


def test_fit_gain_lag_prefers_a_flat_gain_when_there_is_no_lag():
    """A memoryless plant must not be handed a spurious time constant: with a flat curve the fitted
    DC gain IS the curve and the flat-model reference residual is zero."""
    out = _fit_on_curve(np.full(len(meg.PAIR_WINDOWS), 0.61), 0.0536)
    assert out[0] == pytest.approx(0.61, rel=0.05)
    assert out[3] == pytest.approx(0.0, abs=1e-9)  # the flat-gain residual it is compared against


# ---------------------------------------------------------------------------- the matrix gains
def test_fit_G_recovers_scalar_diag_and_full_exactly():
    rng = np.random.default_rng(3)
    a = rng.normal(0, 0.01, size=(500, 3))
    for mode, truth in (("scalar", np.eye(3) * 0.61),
                        ("diag", np.diag([0.46, 0.66, 0.48])),
                        ("full", np.array([[0.50, -0.04, 0.24],
                                           [-0.09, 0.67, 0.07],
                                           [0.31, -0.23, 0.52]]))):
        b = a @ truth.T
        assert np.allclose(egc.fit_G(a, b, mode), truth, atol=1e-8)


def test_scalar_fit_cannot_represent_cross_axis_coupling():
    """The C1 result in miniature: a coupled plant leaves a large residual under a scalar gain and
    none under a full matrix. This is why the diagonal step bought 1% and the full one 14%."""
    rng = np.random.default_rng(4)
    a = rng.normal(0, 0.01, size=(800, 3))
    truth = np.array([[0.50, 0.0, 0.24], [0.0, 0.67, 0.0], [0.31, 0.0, 0.52]])
    b = a @ truth.T
    r_scalar = b - a @ egc.fit_G(a, b, "scalar").T
    r_full = b - a @ egc.fit_G(a, b, "full").T
    assert np.linalg.norm(r_full) < 1e-9 < np.linalg.norm(r_scalar)


def test_residual_split_is_orthogonal():
    """along^2 + normal^2 == |r|^2, so the two components can never double-count the error."""
    rng = np.random.default_rng(5)
    a = rng.normal(0, 0.01, size=(200, 3))
    r = rng.normal(0, 0.002, size=(200, 3))
    al, no = egc._split(r, a)
    assert np.allclose(al ** 2 + no ** 2, np.sum(r ** 2, axis=1), atol=1e-18)


def test_loto_gain_does_not_reward_extra_parameters_on_noise():
    """Leave-one-take-out is the guard that makes C1's 9-vs-1 comparison meaningful. On pure noise
    the 9-parameter model must NOT beat the 1-parameter one out of sample."""
    rng = np.random.default_rng(6)
    bundle = [dict(dcmd=rng.normal(0, 0.01, size=(200, 3)),
                   dmeas=rng.normal(0, 0.01, size=(200, 3)),
                   f=np.zeros(200), step=np.arange(200) % 30,
                   chunk=np.arange(200) // 20) for _ in range(8)]
    s = egc.loto_gain(bundle, "scalar")
    f = egc.loto_gain(bundle, "full")
    assert f[0] >= s[0] * 0.999


# ------------------------------------------------------------------------------- the eps clamp
def test_divergence_replay_clamp_bounds_the_carried_anchor_lead():
    """eps bounds the anchor lead AT THE RE-ANCHOR, the only instant ``resolve`` runs. It does NOT
    bound the same difference measured mid-chunk (the arm keeps moving while the anchor is held),
    and it does not bound the commanded-vs-measured gap, which adds the chunk's own excursion on
    top. Conflating those three is what the archived boundary_and_divergence.txt did."""
    groups = [_synthetic_take(np.eye(3) * 0.3, n=90, n_chunks=6, seed=s) for s in range(6)]
    for g in groups:
        cmd, _, _, chunk, ok, _ = g["raw"]
        g["anchors"] = {int(c): cmd[np.where(ok & (chunk == c))[0][0]].copy()
                        for c in np.unique(chunk)}
    eps = 0.01
    r = meg.divergence_replay(groups, eps)
    assert r is not None
    assert r[3] <= eps + 1e-9, "peak anchor lead must never exceed eps"
    unclamped = meg.divergence_replay(groups, None)
    assert unclamped[1] > r[1], "removing the clamp must let the ask diverge further"


# --------------------------------------------------------------------------- the force sign flip
def test_wire_force_reader_undoes_the_command_sign_flip():
    """``F_cmd = -F_meas``. If the reader does not undo it, a perfectly tracked force reads as
    twice the force in error."""
    from fuji.utils.force_frame import CMD_FROM_MEAS_SIGN

    class _FakeTake:
        def commanded_force_sensor(self):
            t = np.array([0.0, 1.0, 2.0])
            f = np.tile(np.array([3.0, -1.0, 2.0]) * CMD_FROM_MEAS_SIGN, (3, 1))
            return t, f

    got = meg._wire_force_at_ticks(_FakeTake(), np.array([0.5, 1.5]))
    assert np.allclose(got, np.array([3.0, -1.0, 2.0]))


def test_cum_path_helpers_handle_empty_input():
    """Every one of these runs over 87 takes of varying quality; an empty chunk must not raise."""
    path, disp = meg._chunk_step_path({}, np.zeros(0, dtype=int), np.zeros(0, dtype=int))
    assert path.size == 0 and disp.size == 0
    assert meg.gain_increment([], 1) is None
