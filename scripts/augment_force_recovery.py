#!/usr/bin/env python3
"""Build a bank of funnel-shaped force-RECOVERY trajectories from an HL-v2 ``.zarr.zip``.

Reads the stored ``hl_<role>_action_force`` chunks — the exact wrench targets the model is
trained to emit — picks the windows where a recovery is well-posed (in contact, force actually
moving, every step supervised), and for each one synthesizes
:attr:`~fuji.utils.force_funnel.FunnelConfig.n_aug` trajectories that start out-of-distribution
and funnel back onto the demonstration. See :mod:`fuji.utils.force_funnel` for the method and
its mapping onto 1001-Demos (``docs/papers/1001demos.pdf``).

**This script only WRITES a new file.** It opens the dataset read-only, touches no converter,
no ``HLDataset``, no training code, and nothing under ``fuji/s1_policy/``. The output is a
self-describing ``.npz`` sidecar; no training run reads it yet.

    uv run scripts/augment_force_recovery.py <dataset.zarr.zip> -o bank.npz
    uv run scripts/augment_force_recovery.py <dataset.zarr.zip> --episodes 0-9 --n-aug 16
    uv run scripts/augment_force_recovery.py <dataset.zarr.zip> --dry-run    # stats only

Arrays in the output, ``N`` = total generated trajectories:

    episode        (N,)        int32   source episode index
    frame          (N,)        int32   source ANCHOR frame, GLOBAL (dataset-wide) index
    force_action   (N, H, 6)   float32 the generated chunk — the augmented target
    force_history  (N, L, 6)   float32 the observation to pair with it (== chunk[0], held)
    force_demo     (N, H, 6)   float32 the source demo chunk (CONTEXT; see force_target)
    force_target   (N, H, 6)   float32 what the funnel actually converged onto -- constant at
                                       the t0 wrench under --target-mode anchor (the default)
    merge_step     (N,)        int32   step at which it closed 90% of its gap to force_target
    window_frame   (W,)        int32   the W selected anchors (frames), for provenance
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fuji.utils.force_funnel import (  # noqa: E402
    FORCE_SLICE,
    TARGET_MODES,
    FunnelConfig,
    WindowGate,
    generate_recovery_chunks,
    observation_for,
    select_windows,
    wrench_consistency,
)


def _git_sha(repo_root: str) -> str:
    try:
        out = subprocess.run(["git", "-C", repo_root, "rev-parse", "HEAD"],
                             capture_output=True, text=True, timeout=10, check=False)
        return out.stdout.strip() or "unknown"
    except Exception:  # noqa: BLE001 - provenance is best-effort, never fatal
        return "unknown"


def _parse_episodes(spec: str | None, n_ep: int) -> np.ndarray:
    """``None`` -> all; ``"0-9"`` -> 0..9; ``"3,7,11"`` -> those three."""
    if not spec:
        return np.arange(n_ep, dtype=np.int64)
    out: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part.lstrip("-"):
            lo, hi = part.split("-", 1)
            out.extend(range(int(lo), int(hi) + 1))
        else:
            out.append(int(part))
    idx = np.asarray(sorted(set(out)), dtype=np.int64)
    if idx.size == 0:
        raise SystemExit(f"--episodes {spec!r} selected nothing")
    if idx.min() < 0 or idx.max() >= n_ep:
        raise SystemExit(f"--episodes {spec!r} out of range for a {n_ep}-episode dataset")
    return idx


def open_hl_dataset(path: str):
    """Open an HL-v2 ``.zarr.zip`` read-only, refusing one this augmentation cannot use.

    The refusals are loud on purpose: a dataset without ``hl_windows`` has no stored action
    chunks at all, and one whose force is not in the bota sensor frame would make every
    generated wrench a frame error rather than an augmentation.
    """
    import zarr
    from imagecodecs.numcodecs import register_codecs

    register_codecs()  # camera0_rgb is JpegXl; array metadata will not load without it
    g = zarr.open(path, "r")
    attrs = dict(g.attrs)
    if not attrs.get("hl_windows"):
        raise SystemExit(
            f"{path}: attrs.hl_windows is not set — this is not an HL-v2 dataset, so it has no "
            f"stored hl_*_action_force chunks to augment. Re-convert with the HL window writer."
        )
    frame = attrs.get("hl_force_frame")
    if frame not in (None, "bota_sensor"):
        raise SystemExit(
            f"{path}: attrs.hl_force_frame={frame!r}. This tool generates wrenches in the "
            f"SENSOR frame (the frame HLDataset passes through un-relativized); a different "
            f"frame would silently make every generated chunk a frame error."
        )
    return g, attrs


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("zarr_path", help="an HL-v2 .zarr.zip (attrs.hl_windows must be set)")
    p.add_argument("-o", "--out", default=None, help="output .npz (default: <dataset>.force_recovery.npz)")
    p.add_argument("--role", default=None, help="hand role (default: attrs.robot0_primary_role, else 'left')")
    p.add_argument("--episodes", default=None, help='"0-9" or "3,7,11" (default: all)')
    p.add_argument("--dry-run", action="store_true", help="report yields and stats, write nothing")
    p.add_argument("--seed", type=int, default=0, help="base RNG seed; each window uses seed+frame")

    g_gate = p.add_argument_group("window gate")
    g_gate.add_argument("--contact-min-n", type=float, default=WindowGate.contact_min_n,
                        help="peak |F| the window must reach somewhere")
    g_gate.add_argument("--anchor-min-force-n", type=float, default=WindowGate.anchor_min_force_n,
                        help="|F| AT t0 — 'contact starts above this'")
    g_gate.add_argument("--target-min-force-n", type=float, default=WindowGate.target_min_force_n,
                        help="|F| at the last step — the recovery target must be a real force")
    g_gate.add_argument("--ramp-min-n", type=float, default=WindowGate.ramp_min_n)
    g_gate.add_argument("--stride", type=int, default=WindowGate.stride)

    g_fun = p.add_argument_group("funnel")
    g_fun.add_argument("--n-aug", type=int, default=FunnelConfig.n_aug)
    g_fun.add_argument("--target-mode", choices=TARGET_MODES, default=FunnelConfig.target_mode,
                       help="what the funnel converges onto: 'anchor' = the single wrench at t0 "
                            "(one fixed setpoint, the paper's shape); 'demo' = the demo chunk "
                            "step by step. They differ by a median 86 deg of force direction.")
    g_fun.add_argument("--force-radius-scale", type=float, default=FunnelConfig.force_radius_scale)
    g_fun.add_argument("--torque-cone-deg", type=float, default=FunnelConfig.torque_cone_deg)
    g_fun.add_argument("--funnel-power", type=float, default=FunnelConfig.funnel_power)
    g_fun.add_argument("--lambda-acc", type=float, default=FunnelConfig.lambda_acc)
    g_fun.add_argument("--w-min", type=float, default=FunnelConfig.w_min)
    g_fun.add_argument("--max-force-n", type=float, default=FunnelConfig.max_force_n)
    g_fun.add_argument("--project-torque", action="store_true",
                       help="snap T onto r x F (breaks the terminal T constraint; experimental)")
    args = p.parse_args(argv if argv is not None else sys.argv[1:])

    if not os.path.exists(args.zarr_path):
        raise SystemExit(f"not found: {args.zarr_path}")
    g, attrs = open_hl_dataset(args.zarr_path)

    role = args.role or attrs.get("robot0_primary_role") or "left"
    horizon = int(attrs["hl_action_horizon"])
    hist_len = int(attrs.get("hl_force_hist_len", 1))
    dt = 1.0 / float(attrs.get("hl_action_hz", 30.0))
    key_force = f"hl_{role}_action_force"
    key_valid = f"hl_{role}_action_force_valid"
    key_hist_valid = f"hl_{role}_force_history_valid"
    for k in (key_force, key_valid, key_hist_valid):
        if k not in g["data"]:
            raise SystemExit(f"{args.zarr_path}: missing data/{k} (role={role!r}?)")

    gate = WindowGate(contact_min_n=args.contact_min_n, ramp_min_n=args.ramp_min_n,
                      anchor_min_force_n=args.anchor_min_force_n,
                      target_min_force_n=args.target_min_force_n,
                      stride=max(1, args.stride))
    cfg = FunnelConfig(n_aug=args.n_aug, target_mode=args.target_mode,
                       force_radius_scale=args.force_radius_scale,
                       torque_cone_deg=args.torque_cone_deg, funnel_power=args.funnel_power,
                       lambda_acc=args.lambda_acc, w_min=args.w_min,
                       max_force_n=args.max_force_n, project_torque=args.project_torque)

    ends = np.asarray(g["meta/episode_ends"][:], dtype=np.int64)
    starts = np.concatenate([[0], ends[:-1]]).astype(np.int64)
    eps = _parse_episodes(args.episodes, len(ends))
    print(f"[aug] {os.path.basename(args.zarr_path)}: role={role} H={horizon} L={hist_len} "
          f"dt={dt:.5f}s  {len(ends)} episodes ({len(eps)} selected)")
    print(f"[aug] gate: |F|peak>={gate.contact_min_n} N, |F(t0)|>={gate.anchor_min_force_n} N, "
          f"|F(t0+{horizon - 1})|>={gate.target_min_force_n} N, ramp>={gate.ramp_min_n} N, "
          f"stride={gate.stride}, all-{horizon}-valid")
    print(f"[aug] funnel: target_mode={cfg.target_mode} n_aug={cfg.n_aug} "
          f"power={cfg.funnel_power} lambda_acc={cfg.lambda_acc} "
          f"r_scale={cfg.force_radius_scale} cone={cfg.torque_cone_deg}deg "
          f"project_torque={cfg.project_torque}")

    af, av, ahv = g[f"data/{key_force}"], g[f"data/{key_valid}"], g[f"data/{key_hist_valid}"]
    out_ep: list[np.ndarray] = []
    out_fr: list[np.ndarray] = []
    out_act: list[np.ndarray] = []
    out_hist: list[np.ndarray] = []
    out_demo: list[np.ndarray] = []
    out_target: list[np.ndarray] = []
    out_merge: list[np.ndarray] = []
    all_windows: list[np.ndarray] = []
    n_anchor = 0
    n_window = 0
    rejects: dict[str, int] = {}
    n_rejected = 0
    demo_resid: list[float] = []
    t_start = time.time()

    for ep in eps:
        a, b = int(starts[ep]), int(ends[ep])
        demo_all = np.asarray(af[a:b], dtype=np.float64)          # (T, H, 6)
        valid = np.asarray(av[a:b], dtype=np.float32) > 0.5       # (T, H)
        hvalid = np.asarray(ahv[a:b], dtype=np.float32)           # (T, 1)
        n_anchor += demo_all.shape[0]
        force = demo_all[:, :, FORCE_SLICE]
        mag = np.linalg.norm(force, axis=2)                        # (T, H)
        peak = mag.max(axis=1)
        ramp = np.linalg.norm(force[:, -1] - force[:, 0], axis=1)
        sel = select_windows(valid, hvalid, peak, ramp, anchor_force_n=mag[:, 0],
                             target_force_n=mag[:, -1], gate=gate)
        n_window += sel.size
        all_windows.append(sel + a)
        for local in sel:
            demo = demo_all[int(local)]
            batch = generate_recovery_chunks(demo, dt=dt, cfg=cfg,
                                             rng=int(args.seed) + a + int(local))
            n_rejected += batch.n_rejected
            for k, v in batch.reject_reasons.items():
                rejects[k] = rejects.get(k, 0) + v
            demo_resid.append(wrench_consistency(demo)[0])
            if batch.n == 0:
                continue
            out_ep.append(np.full(batch.n, ep, dtype=np.int32))
            out_fr.append(np.full(batch.n, a + int(local), dtype=np.int32))
            out_act.append(batch.chunks.astype(np.float32))
            out_hist.append(np.stack([observation_for(c, hist_len) for c in batch.chunks]
                                     ).astype(np.float32))
            out_demo.append(np.broadcast_to(demo, batch.chunks.shape).astype(np.float32))
            out_target.append(np.broadcast_to(batch.reference, batch.chunks.shape).astype(np.float32))
            out_merge.append(batch.merge_steps.astype(np.int32))

    elapsed = time.time() - t_start
    if not out_act:
        raise SystemExit("[aug] no window passed the gate — loosen --contact-min-n / --ramp-min-n")

    act = np.concatenate(out_act)
    merge = np.concatenate(out_merge)
    demo_arr = np.concatenate(out_demo)
    target_arr = np.concatenate(out_target)
    slew = np.linalg.norm(np.diff(act[:, :, FORCE_SLICE], axis=1), axis=2) / dt
    # Gap to the TARGET, which is what the funnel had to close -- not to the demo, which under
    # anchor mode is a different trajectory the funnel was never aiming at.
    gap0 = np.linalg.norm(act[:, 0] - target_arr[:, 0], axis=1)
    print(f"\n[aug] anchors scanned {n_anchor}  windows passed {n_window} "
          f"({100.0 * n_window / max(n_anchor, 1):.1f}% of anchors, stride {gate.stride} applied)")
    print(f"[aug] generated {act.shape[0]} trajectories in {elapsed:.1f}s "
          f"({act.shape[0] / max(elapsed, 1e-9):.0f}/s)")
    delivered = act.shape[0]
    asked = n_window * cfg.n_aug
    print(f"[aug] delivery {delivered}/{asked} ({100.0 * delivered / max(asked, 1):.1f}%); "
          f"threw away {n_rejected} draws ({100.0 * n_rejected / max(n_rejected + delivered, 1):.1f}%) "
          f"-> {rejects or '{}'}")
    print(f"[aug] initial gap |w_aug(0) - target(0)|: p50={np.percentile(gap0, 50):.2f} "
          f"p90={np.percentile(gap0, 90):.2f} max={gap0.max():.2f}")
    print(f"[aug] merge step (90% of gap closed): p50={np.percentile(merge, 50):.0f} "
          f"p90={np.percentile(merge, 90):.0f} of {horizon - 1}")
    print(f"[aug] peak |dF/dt|: p50={np.percentile(slew.max(1), 50):.1f} "
          f"p90={np.percentile(slew.max(1), 90):.1f} N/s")
    print(f"[aug] source-window T=rxF residual: p50={np.percentile(demo_resid, 50):.3f} "
          f"p90={np.percentile(demo_resid, 90):.3f}")

    if args.dry_run:
        print("[aug] --dry-run: nothing written")
        return 0

    out_path = args.out or (args.zarr_path.replace(".zarr.zip", "") + ".force_recovery.npz")
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    meta = {
        "tool": "scripts/augment_force_recovery.py",
        "method": "fuji.utils.force_funnel (1001-Demos funnel augmentation in wrench space)",
        "fuji_git_sha": _git_sha(repo_root),
        "dataset": os.path.abspath(args.zarr_path),
        "dataset_attrs": {k: attrs.get(k) for k in
                          ("hl_action_horizon", "hl_action_hz", "hl_force_hist_len",
                           "hl_force_frame", "hl_action_first_step", "robot0_primary_role")},
        "role": role,
        "episodes": eps.tolist(),
        "seed": int(args.seed),
        "gate": {k: getattr(gate, k) for k in ("contact_min_n", "anchor_min_force_n",
                                               "target_min_force_n", "ramp_min_n",
                                               "require_all_valid", "stride")},
        "config": {f.name: getattr(cfg, f.name) for f in cfg.__dataclass_fields__.values()},
        # lambda_acc is None by default (it resolves per target_mode), so record what was
        # actually solved with — a consumer reading "None" out of the bank learns nothing.
        "lambda_acc_resolved": cfg.resolved_lambda_acc,
        "n_anchors_scanned": int(n_anchor),
        "n_windows": int(n_window),
        "n_trajectories": int(delivered),
        "n_draws_rejected": int(n_rejected),
        "reject_reasons": rejects,
        # Stated in the file itself so a consumer cannot mistake this for a full-observation
        # augmentation: image, state and pose-action columns are the demo's, untouched.
        "scope": "force channel only; image/state/pose-action columns are unmodified",
        "consumed_by_training": False,
    }
    tmp = out_path + ".tmp.npz"
    np.savez_compressed(
        tmp,
        episode=np.concatenate(out_ep),
        frame=np.concatenate(out_fr),
        force_action=act,
        force_history=np.concatenate(out_hist),
        force_demo=demo_arr,
        force_target=target_arr,
        merge_step=merge,
        window_frame=np.concatenate(all_windows).astype(np.int32),
        meta_json=np.array(json.dumps(meta, indent=2)),
    )
    os.replace(tmp, out_path)
    print(f"[aug] wrote {out_path} ({os.path.getsize(out_path) / 1e6:.1f} MB)")
    print("[aug] NOTE: no training code reads this file — wiring it into HLDataset is a "
          "separate, deliberate step.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
