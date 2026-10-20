#!/usr/bin/env python3
"""Fit quantile (q01/q99) normalization stats for the HL-v2 S1 policy.

Iterates the SAME :class:`HLDataset` the trainer builds, with the same ``--mode``, so
the stats it writes match exactly what the trainer will feed the model. Three keys are
fitted, matching the three tensors ``Trainer._apply_norm`` normalizes:

    actions        (N*16,   action_dim)   the flow-matching target
    state          (N*To,   state_dim)    per-keypoint wrt_start_rot6d (+ gripper)
    force_history   (N*L,    6*n_ft)       the causal wrench input (L = FORCE_HIST_LEN)

Two details that matter, and that a naive fit gets wrong:

* **Masked entries are EXCLUDED.** An invalid action step (past the episode end, or an
  unmeasured force bin) holds a stale value, and a reserved-but-sensorless column is a
  hard zero. Fitting over them drags the quantiles toward those non-measurements — and
  because a q99-q01 span below 0.02 is floored (``norm_stats._QUANTILE_MIN_RANGE``), a
  mostly-masked column collapses to a constant -1.0. Rows are gathered per column
  using ``action_mask``.
* **Widths are asserted at load time** by the trainer (``_check_norm_stats``), because
  ``normalize_quantile`` silently passes through trailing dims beyond the stats width.
  Re-fit whenever ``--mode`` changes.

Usage
-----
    uv run python scripts/fit_hl_norm_stats.py \
        --zarr-paths data/session.zarr.zip \
        --mode left_only \
        --val-last-episodes 4 \
        --out checkpoints/norm_stats_hl_left.json
"""

from __future__ import annotations

import argparse
import logging

import numpy as np

from flore.s1_policy.data.hl_dataset import HLDataset
from flore.s1_policy.data.layout import (
    ACTION_SPAN_S,
    FORCE_DIM,
    FORCE_HIST_SPAN_S,
    MODES,
    OBS_HISTORY_DT_S,
    OBS_HISTORY_LEN,
    WB_JOINT_TARGETS,
    get_layout,
)
from flore.s1_policy.data.norm_stats import NormStats, save_norm_stats

logger = logging.getLogger("fit_hl_norm_stats")


#: How far past a quantile the observed extreme may lie, in units of the [q01, q99] span,
#: before that side is extended to the extreme.
#:
#: The signature of a CLIPPED column is asymmetry: one side's quantile sits on a hard wall
#: (so the span collapses to the narrow tail on that side) while the other side runs many
#: spans further out. Measured on real data, the two rot6d diagonal cosines have
#: ``(q01 - min) / span`` of 3.7 and 4.8, versus 0.07-0.9 for every position, force, and
#: off-diagonal rot6d column. A threshold of 1.5 separates them with a wide margin and
#: leaves ordinary outlier clipping — the thing quantile normalization is FOR — intact.
_EXTREME_OVERRUN_SPANS = 1.5

#: How close (in spans) the OPPOSITE side's extreme must sit to its quantile for that side
#: to count as a hard wall. Together with :data:`_EXTREME_OVERRUN_SPANS` this is the
#: clipped-vs-outlier discriminator: a clipped column overruns on one side while the other
#: side hugs its quantile, whereas an outlier-laden column has a normal tail on the
#: opposite side. Measured: rot6d col 3 has hi_over 0.01 (a wall at +1.0); a unit normal
#: with 5 samples at 50 has lo_over 0.45 (a normal tail) — 0.1 separates them.
_WALL_FRAC = 0.1


def _quantiles_masked(cols: list[np.ndarray], width: int, label: str,
                      robust_one_sided: bool = True) -> NormStats:
    """Per-column q01/q99 (+mean/std) over only the UNMASKED entries of each column.

    ``cols[d]`` holds every valid observation of column ``d``. A column with no valid
    observation at all gets a degenerate (0, 0) range, which the normalizer's 0.02 floor
    turns into a constant — correct, since nothing ever supervises it, and loud below.

    CLIPPED COLUMNS (``robust_one_sided``). Two rot6d columns are the diagonal cosines
    ``R_rel[0,0]`` and ``R_rel[1,1]`` of a SMALL relative rotation, so they pile up
    against a hard ceiling at exactly +1.0. q99 then lands *on* the ceiling and the
    [q01, q99] span measures only the narrow upper tail, while the physically real lower
    tail runs far past q01 — measured on real data: spans of 0.137 and 0.070 mapping the
    true minimum to **-9.1 and -15.1** in normalized space, against a target of ~[-1, 1].
    That is exactly the out-of-range AdaLN conditioning this normalization exists to
    prevent, and it is invisible unless you look per column.

    The detector is ASYMMETRY (see :data:`_EXTREME_OVERRUN_SPANS`): a side whose observed
    extreme overruns its quantile by more than ~1.5 spans is a side whose span is
    describing a narrow tail rather than the distribution, and only that side is extended.
    Ordinary outlier clipping — what quantile normalization is FOR — is left intact,
    because a few wild samples do not stretch the extreme by multiple spans relative to a
    healthy span.
    """
    q01 = np.zeros(width)
    q99 = np.zeros(width)
    mean = np.zeros(width)
    std = np.zeros(width)
    empty = []
    widened = []
    for d in range(width):
        v = cols[d]
        if v.size == 0:
            empty.append(d)
            continue
        lo, hi = np.percentile(v, [1, 99])
        if robust_one_sided and (hi - lo) > 1e-9:
            span = float(hi - lo)
            vmin, vmax = float(v.min()), float(v.max())
            lo_over = (lo - vmin) / span      # how far the low extreme overruns q01
            hi_over = (vmax - hi) / span      # ...and the high extreme, q99
            # A CLIPPED column overruns on exactly ONE side while the other side HUGS its
            # quantile (the wall). An outlier-laden column overruns too, but its opposite
            # side is a normal tail sitting well inside. Requiring the pair — a big
            # overrun AND a wall on the other side — separates them:
            #   measured rot6d col 3: lo_over 12.2, hi_over 0.01  -> clipped
            #   5 samples at 50 in a unit normal: hi_over 10.0, lo_over 0.45 -> outliers
            if lo_over > _EXTREME_OVERRUN_SPANS and hi_over <= _WALL_FRAC:
                lo = vmin
                widened.append(d)
            elif hi_over > _EXTREME_OVERRUN_SPANS and lo_over <= _WALL_FRAC:
                hi = vmax
                widened.append(d)
        q01[d], q99[d] = lo, hi
        mean[d], std[d] = v.mean(), v.std()
    if empty:
        logger.warning("%s: columns %s had NO valid samples (reserved-but-unmeasured?) "
                       "-> degenerate range; they must stay masked in training.",
                       label, empty)
    if widened:
        logger.info("%s: columns %s are CLIPPED (the observed extreme overruns a quantile "
                    "by >%.1f spans, so the span described only a tail) -> that side "
                    "extended, keeping the real range in ~[-1, 1].",
                    label, widened, _EXTREME_OVERRUN_SPANS)
    return NormStats(q01=q01, q99=q99, mean=mean, std=std)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--zarr-paths", nargs="+", required=True, help="One or more .zarr.zip paths.")
    ap.add_argument("--out", required=True, help="Output norm_stats.json path.")
    ap.add_argument("--mode", default="left_only", choices=[m for m in MODES if m != "legacy32"],
                    help="MUST match training (--hl-data.mode).")
    ap.add_argument("--wb-joint-target", default="absolute", choices=list(WB_JOINT_TARGETS),
                    help="--mode wholebody ONLY: what the 29 joint target columns mean. MUST "
                         "match training (--hl-data.wb-joint-target). 'delta' subtracts the "
                         "chunk's own row 0, which is a completely different per-column scale "
                         "at the same width, so a crossed stats file would silently mis-scale "
                         "every joint (and clip_sample would crush it). Recorded in the "
                         "exact-equality dataset_knobs, so the trainer refuses the swap.")
    ap.add_argument("--max-samples", type=int, default=0,
                    help="Subsample this many anchors for the fit (0 = use all).")
    ap.add_argument("--val-last-episodes", type=int, default=4,
                    help="Reserve the final N global flattened episodes for validation (default: 4).")
    ap.add_argument("--drop-invalid-force-history", action="store_true",
                    help="Drop anchors with invalid force history; MUST match training.")
    ap.add_argument("--keep-glitch-windows", action="store_true",
                    help="Fit over ALL anchors, including those whose window straddles a "
                         "tracker glitch. Default OFF (drop them) — MUST match training's "
                         "--hl-data.drop-glitch-windows so the stats cover the same anchors.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--image-key", default="camera0_rgb")
    # MUST match training's --hl-data.image-keys. The stats themselves do not depend on the
    # pixels, but HLDataset records the camera list in `dataset_knobs`, and
    # trainer.validate_split_provenance compares that dict for EXACT equality — so a
    # 2-camera run fitted with 1 camera is refused at startup rather than mid-run.
    ap.add_argument("--image-keys", nargs="+", default=None,
                    help="Every camera the training run reads, in encoder order (e.g. "
                         "camera0_rgb camera1_rgb). Default: just --image-key.")
    ap.add_argument("--image-mean", type=float, default=0.5)
    ap.add_argument("--image-std", type=float, default=0.5)
    ap.add_argument("--obs-history-len", type=int, default=OBS_HISTORY_LEN)
    ap.add_argument("--obs-history-dt-s", type=float, default=OBS_HISTORY_DT_S)
    args = ap.parse_args()

    layout = get_layout(args.mode)
    ds = HLDataset(
        list(args.zarr_paths),
        mode=args.mode,
        image_key=args.image_key,
        image_keys=tuple(args.image_keys) if args.image_keys else None,
        image_mean=args.image_mean,
        image_std=args.image_std,
        drop_invalid_force_history=args.drop_invalid_force_history,
        drop_glitch_windows=not args.keep_glitch_windows,
        glitch_pre_s=FORCE_HIST_SPAN_S,
        glitch_post_s=ACTION_SPAN_S,
        obs_history_len=args.obs_history_len,
        obs_history_dt_s=args.obs_history_dt_s,
        require_episode_ends=args.val_last_episodes > 0,
        wb_joint_target=args.wb_joint_target,
    )
    train_idxs, _ = ds.split_indices(args.val_last_episodes)
    n_total = len(train_idxs)
    if args.max_samples and args.max_samples < n_total:
        rng = np.random.default_rng(args.seed)
        idxs = rng.choice(np.asarray(train_idxs), size=args.max_samples, replace=False)
    else:
        idxs = np.asarray(train_idxs)
    logger.info(
        "Fitting HL norm stats over %d / %d TRAIN anchors; %d validation episodes excluded "
        "(mode=%s, action_dim=%d)",
        len(idxs), n_total, args.val_last_episodes, args.mode, layout.action_dim)

    act_cols: list[list[np.ndarray]] = [[] for _ in range(layout.action_dim)]
    state_cols: list[list[np.ndarray]] = [[] for _ in range(layout.state_dim)]
    # PER CHANNEL, not per anchor: with two F/T sensors an anchor can have a measured left
    # window and an unmeasured right one, and gating the whole 12-channel row on the AND
    # would drop every such left measurement from the fit. Channel d belongs to sensor
    # d // FORCE_DIM (the block order HLDataset writes).
    force_cols: list[list[np.ndarray]] = [[] for _ in range(layout.force_hist_channels)]
    n_hist_invalid = 0
    n_ft = max(1, len(layout.force_roles))

    for i, idx in enumerate(idxs):
        s = ds[int(idx)]
        a = s["action_chunk"].numpy()          # (16, AD)
        m = s["action_mask"].numpy()           # (16, AD)
        for d in range(layout.action_dim):
            valid = a[:, d][m[:, d]]
            if valid.size:
                act_cols[d].append(valid)
        # State is (To, SD) (or (SD,) at To==1); every obs frame contributes to the
        # per-column fit, EXCLUDING fabricated entries (state_valid False — e.g. a
        # dead-encoder gripper zero) so the quantiles describe real measurements only.
        st = np.atleast_2d(s["state"].numpy())          # (To, SD)
        sv = np.atleast_2d(s["state_valid"].numpy())     # (To, SD) bool
        for d in range(layout.state_dim):
            v = st[:, d][sv[:, d]]
            if v.size:
                state_cols[d].append(v)
        # Only a MEASURED history contributes: an invalid window is a stale hold, and
        # including it would bias the input scaling toward non-measurements. `reshape(-1)`
        # handles both shapes the dataset emits — a scalar for a one-sensor layout, (n_ft,)
        # for more — so `.item()` (which raises on a 2-element tensor) is never used.
        fhv = np.asarray(s["force_history_valid"].numpy()).reshape(-1)
        fh = s["force_history"].numpy()                  # (L, C)
        if fhv.size == 1:
            fhv = np.repeat(fhv, n_ft)
        for d in range(layout.force_hist_channels):
            if bool(fhv[d // FORCE_DIM]):
                force_cols[d].append(fh[:, d])
        if not fhv.any():
            n_hist_invalid += 1
        if (i + 1) % 1000 == 0:
            logger.info("  %d/%d", i + 1, len(idxs))
    ds.close()

    stats: dict[str, NormStats] = {}
    stats["actions"] = _quantiles_masked(
        [np.concatenate(c) if c else np.zeros(0) for c in act_cols],
        layout.action_dim, "actions")

    stats["state"] = _quantiles_masked(
        [np.concatenate(c) if c else np.zeros(0) for c in state_cols],
        layout.state_dim, "state")

    # The layout check is not redundant with the observation count: a force-free layout
    # (``FLORE_DROP_FORCE``) still yields "valid" windows, they are just 0 CHANNELS wide, so
    # a naive emptiness test passes while there is nothing to fit. Writing the 0-width entry
    # that produced instead is worse than omitting it — the summary log below reduces over
    # ``q99 - q01`` and a zero-size array has no minimum, so the tool wrote the file and
    # THEN died. Omitting is also what every consumer already handles, via ``key in stats``.
    if layout.force_hist_channels and any(force_cols):
        stats["force_history"] = _quantiles_masked(
            [np.concatenate(c) if c else np.zeros(0) for c in force_cols],
            layout.force_hist_channels, "force_history")
        logger.info("force_history: %d/%d anchors had at least one measured sensor window "
                    "(%d skipped); per-sensor bin counts %s",
                    len(idxs) - n_hist_invalid, len(idxs), n_hist_invalid,
                    [sum(a.size for a in force_cols[i * FORCE_DIM]) for i in range(n_ft)])
    elif not layout.force_hist_channels:
        # Intentional (pure-position ablation), so INFO rather than a warning — and said
        # explicitly, because the other branch's message would send someone hunting a
        # converter bug that isn't there.
        logger.info("Layout declares no F/T sensor (force_hist_channels=0) — force_history "
                    "stats OMITTED by design. The force ACTION columns are reserved but "
                    "unsupervised in this bundle.")
    else:
        logger.warning("No anchor had a valid force history — force_history stats OMITTED. "
                       "Check the converter's tare/coverage masks.")

    split_meta = ds.split_provenance(args.val_last_episodes)
    split_meta.update({
        # Keep these legacy top-level tags for readable errors and old tooling.
        "mode": args.mode,
        "action_dim": layout.action_dim,
        "zarr_paths": list(args.zarr_paths),
        "drop_glitch_windows": not args.keep_glitch_windows,
        "fit_max_samples": args.max_samples,
        "fit_seed": args.seed,
        "fit_train_anchor_count": len(idxs),
        "available_train_anchor_count": n_total,
    })
    save_norm_stats(stats, args.out, meta=split_meta)
    logger.info("Saved HL norm stats -> %s (mode=%s, drop_glitch_windows=%s)",
                args.out, args.mode, not args.keep_glitch_windows)
    for key, st in stats.items():
        span = np.maximum(st.q99 - st.q01, 0.0)
        logger.info("  %-14s width=%d  q01..q99 span: min=%.4f max=%.4f mean=%.4f",
                    key, st.q01.shape[0], float(span.min()), float(span.max()),
                    float(span.mean()))


if __name__ == "__main__":
    main()
