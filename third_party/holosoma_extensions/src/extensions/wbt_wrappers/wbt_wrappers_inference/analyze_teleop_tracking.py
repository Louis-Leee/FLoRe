#!/usr/bin/env python3
"""Quantify + visualize VIVE-teleop hand-tracking error (commanded vs executed).

Reads a ``HOLOSOMA_TELEOP_TRACK_LOG`` NPZ (written by ``HolosomaWBTPolicy`` when
that env var is set during a live or sim teleop run) and answers the question
"how well does the robot hand follow my tracker?". Both poses in the NPZ are
already in the **robot's own (pelvis) frame** — the commanded ``vr_2point`` the
policy tracked, and the executed pose obtained by the SAME root-invariant FK on
the robot's measured joints — so every number and plot here is a same-frame,
apples-to-apples comparison.

Quantitative (printed + JSON): per-axis RMSE / MAE / max position error (cm),
the error-norm distribution, an estimated end-to-end latency (ms, from the
cmd→exec cross-correlation lag) with the lag-compensated residual error, a
per-axis tracking gain (how much of the commanded amplitude the hand actually
realizes), and the orientation geodesic error (deg).

Qualitative (figure): for the chosen hand, the cmd-vs-exec trajectory in all
three robot planes (XY/XZ/YZ — a circle drawn about an axis shows as a circle in
the plane normal to it), the x/y/z time-series overlays, the position-error and
orientation-error time-series, and a metrics panel.

Usage (env hssim — needs matplotlib):

    python -m wbt_wrappers_inference.analyze_teleop_tracking \
        --log g1_deploy/teleop_track.npz \
        --hand right \
        --out g1_deploy/teleop_track

  (writes ``<out>_right.png`` + ``<out>_right_metrics.json``; ``--hand both``
  emits one of each per hand). Also exposed as the console script
  ``wbt-analyze-teleop-tracking``.
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np

from wbt_wrappers_inference.utils.teleop_tracking_log import HAND_LABELS, geodesic_angle_deg

M_TO_CM = 100.0


# --------------------------------------------------------------------------- #
# Metric helpers
# --------------------------------------------------------------------------- #
def _best_lag_samples(cmd: np.ndarray, exec_: np.ndarray, max_lag: int) -> tuple[int, float]:
    """Lag (samples ≥0) at which ``exec_`` best follows ``cmd``, + that score.

    Tests ``exec_[t] ≈ cmd[t - lag]`` for ``lag`` in ``[0, max_lag]`` via a
    global-mean-centered, variance-weighted cross-correlation score (an
    approximate per-window Pearson r — the series is de-meaned once over the full
    span, not per slice, so the score is exact at lag 0 and drifts slightly as the
    window shrinks; this is fine for RANKING the best lag, which is all we use it
    for). Axes are weighted by their command variance, so a still axis
    contributes ~nothing. A positive lag means the executed pose trails the
    command (the expected controller delay). Returns ``(0, 0.0)`` when there is
    not enough motion to estimate a lag.
    """
    n = cmd.shape[0]
    if n < 4:
        return 0, 0.0
    cmd_c = cmd - cmd.mean(axis=0, keepdims=True)
    exe_c = exec_ - exec_.mean(axis=0, keepdims=True)
    var = cmd_c.var(axis=0)
    if float(var.sum()) < 1e-12:
        return 0, 0.0
    w = var / var.sum()  # per-axis weight
    best_lag, best_corr = 0, -np.inf
    max_lag = int(min(max_lag, n - 2))
    for lag in range(0, max_lag + 1):
        a = cmd_c[: n - lag]  # command at t-lag
        b = exe_c[lag:]  # executed at t
        denom = np.sqrt((a * a).sum(axis=0) * (b * b).sum(axis=0))
        with np.errstate(divide="ignore", invalid="ignore"):
            corr_ax = np.where(denom > 1e-12, (a * b).sum(axis=0) / denom, 0.0)
        corr = float((corr_ax * w).sum())
        if corr > best_corr:
            best_corr, best_lag = corr, lag
    return best_lag, best_corr


def _axis_metrics(cmd: np.ndarray, exec_: np.ndarray) -> dict:
    """Per-axis + norm position error metrics (cm) and tracking gain."""
    err = (exec_ - cmd) * M_TO_CM  # (N, 3) cm
    norm = np.linalg.norm(err, axis=1)
    cmd_std = cmd.std(axis=0)
    exec_std = exec_.std(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        gain = np.where(cmd_std > 1e-6, exec_std / cmd_std, np.nan)
    return {
        "rmse_cm": {ax: float(np.sqrt(np.mean(err[:, i] ** 2))) for i, ax in enumerate("xyz")},
        "mae_cm": {ax: float(np.mean(np.abs(err[:, i]))) for i, ax in enumerate("xyz")},
        "max_abs_cm": {ax: float(np.max(np.abs(err[:, i]))) for i, ax in enumerate("xyz")},
        "bias_cm": {ax: float(np.mean(err[:, i])) for i, ax in enumerate("xyz")},
        "err_norm_cm": {
            "rmse": float(np.sqrt(np.mean(norm**2))),
            "median": float(np.median(norm)),
            "p95": float(np.percentile(norm, 95)),
            "max": float(np.max(norm)),
        },
        "tracking_gain": {ax: (float(gain[i]) if np.isfinite(gain[i]) else None) for i, ax in enumerate("xyz")},
    }


def _orientation_metrics(cmd_orn: np.ndarray, exec_orn: np.ndarray) -> dict:
    """Per-tick geodesic orientation error (deg) → distribution summary."""
    angs = np.array([geodesic_angle_deg(cmd_orn[t], exec_orn[t]) for t in range(cmd_orn.shape[0])])
    return {
        "geodesic_deg": {
            "mean": float(np.mean(angs)),
            "median": float(np.median(angs)),
            "p95": float(np.percentile(angs, 95)),
            "max": float(np.max(angs)),
        },
        "_per_tick_deg": angs,  # kept for plotting; stripped before JSON
    }


def compute_metrics(
    cmd_pos: np.ndarray,
    exec_pos: np.ndarray,
    cmd_orn: np.ndarray,
    exec_orn: np.ndarray,
    rl_rate: float,
    elapsed_s: float | None = None,
) -> dict:
    """Full metric bundle for one hand's (cmd, exec) position+orientation streams.

    ``elapsed_s`` (the analyzed window's true wall-clock span) is used for
    ``duration_s`` when available, so the reported duration matches the plotted
    time axis even under loop jitter / dropped ticks; it falls back to ``n/rl_rate``.
    """
    n = cmd_pos.shape[0]
    pos = _axis_metrics(cmd_pos, exec_pos)
    # Latency: cap the lag search at ~0.5 s of motion.
    max_lag = max(1, int(round(0.5 * rl_rate)))
    lag, corr = _best_lag_samples(cmd_pos, exec_pos, max_lag)
    lag_ms = 1000.0 * lag / rl_rate if rl_rate > 0 else 0.0
    # Lag-compensated residual: align exec to cmd by the estimated lag, then the
    # error norm RMSE isolates shape/amplitude error from pure delay.
    if lag > 0 and n - lag >= 2:
        comp = (exec_pos[lag:] - cmd_pos[: n - lag]) * M_TO_CM
        comp_rmse = float(np.sqrt(np.mean(np.linalg.norm(comp, axis=1) ** 2)))
    else:
        comp_rmse = pos["err_norm_cm"]["rmse"]
    orn = _orientation_metrics(cmd_orn, exec_orn)
    if elapsed_s is not None:
        duration = float(elapsed_s)
    elif rl_rate > 0:
        duration = float(n / rl_rate)
    else:
        duration = None
    return {
        "n_ticks": int(n),
        "duration_s": duration,
        "position": pos,
        "latency": {
            "estimated_ms": float(lag_ms),
            "lag_samples": int(lag),
            "xcorr_at_lag": float(corr),
            "lag_compensated_err_rmse_cm": comp_rmse,
        },
        "orientation": orn,
    }


def _strip_private(d: dict) -> dict:
    """Drop keys starting with '_' (the per-tick arrays kept only for plotting)."""
    return {k: (_strip_private(v) if isinstance(v, dict) else v) for k, v in d.items() if not k.startswith("_")}


# --------------------------------------------------------------------------- #
# Plotting
# --------------------------------------------------------------------------- #
def _plot_hand(hand: str, cmd_pos, exec_pos, t_rel, metrics, out_png: str, live_pos=None) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    orn_per_tick = metrics["orientation"]["_per_tick_deg"]
    fig, axes = plt.subplots(3, 3, figsize=(16, 13))
    fig.suptitle(
        f"VIVE teleop tracking — {hand.upper()} hand (commanded tracker target vs executed, pelvis frame)\n"
        f"{metrics['n_ticks']} ticks, {metrics['duration_s']:.1f}s   |   "
        f"err RMSE={metrics['position']['err_norm_cm']['rmse']:.2f} cm, "
        f"latency≈{metrics['latency']['estimated_ms']:.0f} ms, "
        f"orn med={metrics['orientation']['geodesic_deg']['median']:.1f}°",
        fontsize=12,
    )

    planes = [
        (0, 1, "X", "Y", "XY (circle about Z)"),
        (0, 2, "X", "Z", "XZ (circle about Y)"),
        (1, 2, "Y", "Z", "YZ (circle about X)"),
    ]
    for col, (i, j, li, lj, title) in enumerate(planes):
        ax = axes[0][col]
        ax.plot(
            cmd_pos[:, i] * M_TO_CM, cmd_pos[:, j] * M_TO_CM, "-", color="tab:blue", lw=1.6, label="commanded (tracker)"
        )
        ax.plot(
            exec_pos[:, i] * M_TO_CM,
            exec_pos[:, j] * M_TO_CM,
            "-",
            color="tab:red",
            lw=1.6,
            alpha=0.85,
            label="executed (robot)",
        )
        ax.plot(cmd_pos[0, i] * M_TO_CM, cmd_pos[0, j] * M_TO_CM, "o", color="tab:blue", ms=7)
        ax.plot(exec_pos[0, i] * M_TO_CM, exec_pos[0, j] * M_TO_CM, "o", color="tab:red", ms=7)
        ax.set_xlabel(f"{li} (cm)")
        ax.set_ylabel(f"{lj} (cm)")
        ax.set_title(title)
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(True, alpha=0.3)
        if col == 0:
            ax.legend(fontsize=8, loc="best")

    for col, ax_lbl in enumerate("xyz"):
        ax = axes[1][col]
        ax.plot(t_rel, cmd_pos[:, col] * M_TO_CM, "-", color="tab:blue", lw=1.4, label="commanded")
        ax.plot(t_rel, exec_pos[:, col] * M_TO_CM, "-", color="tab:red", lw=1.4, alpha=0.85, label="executed")
        gain = metrics["position"]["tracking_gain"][ax_lbl]
        rmse = metrics["position"]["rmse_cm"][ax_lbl]
        ax.set_xlabel("t (s)")
        ax.set_ylabel(f"{ax_lbl} (cm)")
        ax.set_title(
            f"{ax_lbl}(t)  RMSE={rmse:.2f} cm  gain={gain:.2f}"
            if gain is not None
            else f"{ax_lbl}(t)  RMSE={rmse:.2f} cm"
        )
        ax.grid(True, alpha=0.3)
        if col == 0:
            ax.legend(fontsize=8, loc="best")

    # Position error per axis + norm over time.
    ax = axes[2][0]
    err = (exec_pos - cmd_pos) * M_TO_CM
    for col, ax_lbl in enumerate("xyz"):
        ax.plot(t_rel, err[:, col], lw=1.0, label=f"{ax_lbl} err")
    ax.plot(t_rel, np.linalg.norm(err, axis=1), "k-", lw=1.6, label="‖err‖")
    ax.set_xlabel("t (s)")
    ax.set_ylabel("position error (cm)")
    ax.set_title("Tracking error over time")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="best")

    # Orientation geodesic error over time.
    ax = axes[2][1]
    ax.plot(t_rel, orn_per_tick, "-", color="tab:purple", lw=1.2)
    ax.set_xlabel("t (s)")
    ax.set_ylabel("orientation error (deg)")
    ax.set_title(
        f"Orientation geodesic error  med={metrics['orientation']['geodesic_deg']['median']:.1f}°  "
        f"max={metrics['orientation']['geodesic_deg']['max']:.1f}°"
    )
    ax.grid(True, alpha=0.3)

    # Metrics text panel.
    ax = axes[2][2]
    ax.axis("off")
    p = metrics["position"]
    lat = metrics["latency"]
    lines = [
        f"ticks: {metrics['n_ticks']}   dur: {metrics['duration_s']:.1f}s",
        "",
        "POSITION ERROR (executed − commanded), cm",
        f"  RMSE   x={p['rmse_cm']['x']:.2f}  y={p['rmse_cm']['y']:.2f}  z={p['rmse_cm']['z']:.2f}",
        f"  MAE    x={p['mae_cm']['x']:.2f}  y={p['mae_cm']['y']:.2f}  z={p['mae_cm']['z']:.2f}",
        f"  max|·| x={p['max_abs_cm']['x']:.2f}  y={p['max_abs_cm']['y']:.2f}  z={p['max_abs_cm']['z']:.2f}",
        f"  bias   x={p['bias_cm']['x']:+.2f}  y={p['bias_cm']['y']:+.2f}  z={p['bias_cm']['z']:+.2f}",
        f"  ‖err‖  RMSE={p['err_norm_cm']['rmse']:.2f}  med={p['err_norm_cm']['median']:.2f}  "
        f"p95={p['err_norm_cm']['p95']:.2f}  max={p['err_norm_cm']['max']:.2f}",
        "",
        "TRACKING GAIN (executed amp / commanded amp)",
        "  x={}  y={}  z={}".format(
            *[f"{p['tracking_gain'][a]:.2f}" if p["tracking_gain"][a] is not None else "n/a" for a in "xyz"]
        ),
        "",
        "LATENCY (cmd → exec cross-correlation)",
        f"  ≈ {lat['estimated_ms']:.0f} ms ({lat['lag_samples']} ticks), xcorr={lat['xcorr_at_lag']:.3f}",
        f"  lag-compensated ‖err‖ RMSE = {lat['lag_compensated_err_rmse_cm']:.2f} cm",
        "",
        "ORIENTATION ERROR (geodesic), deg",
        f"  mean={metrics['orientation']['geodesic_deg']['mean']:.1f}  "
        f"med={metrics['orientation']['geodesic_deg']['median']:.1f}  "
        f"p95={metrics['orientation']['geodesic_deg']['p95']:.1f}  "
        f"max={metrics['orientation']['geodesic_deg']['max']:.1f}",
    ]
    ax.text(0.0, 1.0, "\n".join(lines), va="top", ha="left", family="monospace", fontsize=9, transform=ax.transAxes)

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_png, dpi=110)
    plt.close(fig)
    print(f"  figure → {out_png}")


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def _slice_hand(arr_pos: np.ndarray, arr_orn: np.ndarray, hand_idx: int):
    """Extract one hand's (pos[N,3], orn[N,6]) slices from the packed arrays."""
    ps = slice(hand_idx * 3, hand_idx * 3 + 3)
    os_ = slice(hand_idx * 6, hand_idx * 6 + 6)
    return arr_pos[:, ps], arr_orn[:, os_]


def _contiguous_runs(selected_rows: np.ndarray) -> list[tuple[int, int]]:
    """Split ascending original-row indices into maximal consecutive runs.

    Returns a list of (start_pos, end_pos) half-open slices INTO ``selected_rows``
    (not original rows), one per run where the original row index increments by 1.
    A STOP→START gap, an arm/disarm toggle, or a dropped tick breaks a run.
    """
    if selected_rows.size == 0:
        return []
    breaks = np.flatnonzero(np.diff(selected_rows) != 1) + 1
    bounds = np.concatenate([[0], breaks, [selected_rows.size]])
    return [(int(bounds[i]), int(bounds[i + 1])) for i in range(len(bounds) - 1)]


def analyze(
    log_path: str, hands: list[str], out_stem: str, armed_only: bool, no_plot: bool, largest_run: bool = False
) -> dict:
    data = np.load(log_path, allow_pickle=True)
    rl_rate = float(data["rl_rate"]) if "rl_rate" in data else 50.0
    cmd_pos_all = data["cmd_pos"].astype(np.float64)
    cmd_orn_all = data["cmd_orn"].astype(np.float64)
    exec_pos_all = data["exec_pos"].astype(np.float64)
    exec_orn_all = data["exec_orn"].astype(np.float64)
    armed = data["armed"].astype(bool) if "armed" in data else np.ones(cmd_pos_all.shape[0], dtype=bool)
    t = data["t"].astype(np.float64) if "t" in data else np.arange(cmd_pos_all.shape[0]) / rl_rate

    mask = np.ones(cmd_pos_all.shape[0], dtype=bool)
    note = "all ticks"
    if armed_only and armed.any():
        mask = armed.copy()
        note = f"armed ticks only ({int(armed.sum())}/{armed.size})"
    elif armed_only and not armed.any():
        note = "all ticks (no armed ticks in log)"
    # Drop any rows with NaN cmd/exec (cmd/exec are always valid on the 2pt path,
    # but guard against partial frames).
    valid = ~(np.isnan(cmd_pos_all).any(axis=1) | np.isnan(exec_pos_all).any(axis=1))
    mask &= valid

    # The selection may be NON-CONTIGUOUS: arm/disarm/recenter and multiple
    # START→STOP cycles all leave gaps within one NPZ. Treating the concatenation
    # as one uniform time series corrupts the latency lag, lag-compensated error,
    # duration, and the time-series/trajectory plots across each seam. Detect the
    # contiguous runs (consecutive original-row indices) and warn; with
    # --largest-run, analyze ONLY the single longest run for clean numbers.
    selected_rows = np.flatnonzero(mask)
    runs = _contiguous_runs(selected_rows)
    if len(runs) > 1:
        if largest_run:
            best = max(runs, key=lambda r: r[1] - r[0])
            keep_rows = selected_rows[best[0] : best[1]]
            mask = np.zeros_like(mask)
            mask[keep_rows] = True
            note += f" → largest of {len(runs)} contiguous runs ({keep_rows.size} ticks)"
            print(f"  ⚠️ selection had {len(runs)} disjoint runs; --largest-run → analyzing the longest only.")
        else:
            print(
                f"  ⚠️ WARNING: selection spans {len(runs)} DISJOINT runs (arm/disarm, recenter, or "
                f"multiple START→STOP cycles in one log). Latency/lag-compensated error and the time "
                f"plots are computed across the seams and may be misleading. Use --largest-run to "
                f"analyze only the longest contiguous span, or --all-ticks to see the whole timeline."
            )
            note += f" [{len(runs)} disjoint runs — see warning]"

    if int(mask.sum()) < 4:
        raise SystemExit(
            f"Only {int(mask.sum())} usable ticks ({note}) — not enough to analyze. "
            "Did you ARM the VIVE clutch and move the trackers? (try --all-ticks)"
        )

    cmd_pos_all, cmd_orn_all = cmd_pos_all[mask], cmd_orn_all[mask]
    exec_pos_all, exec_orn_all = exec_pos_all[mask], exec_orn_all[mask]
    t_sel = t[mask]
    t_rel = t_sel - t_sel[0]
    # True wall-clock span of the analyzed window (matches the plotted t axis).
    elapsed_s = float(t_sel[-1] - t_sel[0]) if t_sel.size >= 2 and np.all(np.isfinite(t_sel)) else None

    print(f"Loaded {log_path}: {mask.size} ticks, analyzing {int(mask.sum())} ({note}), rl_rate={rl_rate:.0f} Hz")
    results: dict = {
        "log": log_path,
        "rl_rate": rl_rate,
        "analyzed_ticks": int(mask.sum()),
        "selection": note,
        "n_contiguous_runs": len(runs),
        "hands": {},
    }

    for hand in hands:
        hi = HAND_LABELS.index(hand)
        cmd_pos, cmd_orn = _slice_hand(cmd_pos_all, cmd_orn_all, hi)
        exec_pos, exec_orn = _slice_hand(exec_pos_all, exec_orn_all, hi)
        metrics = compute_metrics(cmd_pos, exec_pos, cmd_orn, exec_orn, rl_rate, elapsed_s=elapsed_s)
        clean = _strip_private(metrics)
        results["hands"][hand] = clean

        p = clean["position"]
        print(f"\n=== {hand.upper()} hand ===")
        print(
            f"  ‖err‖ RMSE = {p['err_norm_cm']['rmse']:.2f} cm  (median {p['err_norm_cm']['median']:.2f}, max {p['err_norm_cm']['max']:.2f})"
        )
        print(f"  per-axis RMSE  x={p['rmse_cm']['x']:.2f}  y={p['rmse_cm']['y']:.2f}  z={p['rmse_cm']['z']:.2f}  cm")
        print(
            "  tracking gain  x={}  y={}  z={}".format(
                *[f"{p['tracking_gain'][a]:.2f}" if p["tracking_gain"][a] is not None else "n/a" for a in "xyz"]
            )
        )
        print(
            f"  latency ≈ {clean['latency']['estimated_ms']:.0f} ms  (lag-compensated ‖err‖ RMSE {clean['latency']['lag_compensated_err_rmse_cm']:.2f} cm)"
        )
        print(
            f"  orientation geodesic  median {clean['orientation']['geodesic_deg']['median']:.1f}°  max {clean['orientation']['geodesic_deg']['max']:.1f}°"
        )

        if not no_plot:
            _plot_hand(hand, cmd_pos, exec_pos, t_rel, metrics, f"{out_stem}_{hand}.png")
        out_json = f"{out_stem}_{hand}_metrics.json"
        with open(out_json, "w") as f:
            json.dump({"hand": hand, **clean, "meta": {"log": log_path, "selection": note}}, f, indent=2)
        print(f"  metrics → {out_json}")

    return results


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Quantify + plot VIVE-teleop hand tracking error (cmd vs executed).")
    ap.add_argument("--log", required=True, help="teleop-tracking NPZ (from HOLOSOMA_TELEOP_TRACK_LOG)")
    ap.add_argument("--hand", default="right", choices=["left", "right", "both"], help="which hand to analyze")
    ap.add_argument("--out", default="", help="output stem (default: <log without .npz>)")
    ap.add_argument("--all-ticks", action="store_true", help="analyze all ticks, not just the armed (teleop) span")
    ap.add_argument(
        "--largest-run",
        action="store_true",
        help="when the selection is non-contiguous (arm/disarm, recenter, multiple START→STOP), "
        "analyze only the single longest contiguous run (clean latency/plots)",
    )
    ap.add_argument("--no-plot", action="store_true", help="metrics + JSON only, skip the figure")
    args = ap.parse_args(argv)

    out_stem = args.out or (args.log[:-4] if args.log.endswith(".npz") else args.log)
    hands = ["left", "right"] if args.hand == "both" else [args.hand]
    analyze(
        args.log, hands, out_stem, armed_only=not args.all_ticks, no_plot=args.no_plot, largest_run=args.largest_run
    )


if __name__ == "__main__":
    main(sys.argv[1:])
