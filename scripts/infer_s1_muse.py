#!/usr/bin/env python3
"""Run deterministic offline inference and viser visualization for a MuSe-on-UVA checkpoint.

The fourth sibling of ``infer_s1_{flow,diffusion,act}.py``, and the same viewer: the selected
index is an exact HLDataset sample, and the page compares the native-layout prediction with
that sample's masked target on the fixed t0-inclusive 30 Hz action grid, force included.

Three things differ, all forced by the model living in another environment
(``fuji/s1_policy/inference/muse.py`` has the full reasoning):

  * ``--zarr-path`` is REQUIRED. The siblings read their zarr list out of the checkpoint's
    ``hl_data`` block; a UVA checkpoint records a co-training MIX (five sessions for
    ``uva_hl_force``), so which one to compare against has to be said.
  * ``--backend-python`` must point at the uvaforce interpreter (or ``$BACKEND_PY``). Use
    ``./scripts/run_infer_s1_muse.sh``, which resolves it and — load-bearing — derives
    ``FUJI_ACTION_HORIZON`` from the checkpoint before this process starts.
  * there is no ``--num-steps``. The UVA action head decodes in one pass; its own diffusion-head
    step counts are fixed by the checkpoint. Same call as the ACT CLI: a flag that silently did
    nothing would be worse than its absence.

``--seed`` DOES change the output here (unlike ACT): ``sample_tokens`` draws diffusion noise for
both the action and force heads, and the backend seeds torch per request, so a given
``(sample, seed)`` is reproducible while different seeds show the sampler's spread.
"""

from __future__ import annotations

import argparse
import json
import sys

from fuji.s1_policy.inference.common import result_summary
from fuji.s1_policy.inference.muse import MuseInferenceSession
from fuji.s1_policy.serving.hl_muse import resolve_backend_python


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "checkpoint", nargs="?", default="",
        help="UVA workspace .ckpt or exported .slim.pt. Omit for a RANDOM-INIT wiring check "
             "(every prediction is meaningless — the page says so).")
    parser.add_argument(
        "--zarr-path", required=True,
        help="the HL-v2 *_v3.zarr.zip to compare against (REQUIRED: a UVA checkpoint records a "
             "co-training mix, not one file). Both loaders read this same file.")
    parser.add_argument("--backend-python", default=None,
                        help="uvaforce interpreter; defaults to $BACKEND_PY then "
                             "~/miniconda3/envs/uvaforce/bin/python")
    parser.add_argument("--dataset-root", default=None,
                        help="dir the UVA loader resolves zarr names against (default: the "
                             "--zarr-path directory)")
    parser.add_argument("--device", default="cuda", help="torch device for the BACKEND process")
    parser.add_argument("--sample-index", type=int, default=None,
                        help="initial exact HLDataset sample index (default: the first index "
                             "whose UVA window is fully in-episode)")
    parser.add_argument("--seed", type=int, default=0, help="sampling noise seed")
    parser.add_argument("--val-last-episodes", type=int, default=0,
                        help="label the last N episodes 'validation' in the page. FUJI's split "
                             "convention, not UVA's — that run holds out by ratio, so this is "
                             "advisory unless you know they line up.")
    parser.add_argument("--force-fusion", default=None, choices=["early", "late", "both"],
                        help="override the checkpoint's fusion mode (ablation only)")
    parser.add_argument("--config-name", default="uva_hl_force",
                        help="hydra config used only for a random-init run")
    parser.add_argument("--host", default="127.0.0.1", help="viser bind host")
    parser.add_argument("--port", type=int, default=8090, help="viser port")
    parser.add_argument(
        "--self-test", action="store_true",
        help="run one headless rollout, print JSON, and exit without importing viser")
    args = parser.parse_args(argv)

    session = MuseInferenceSession(
        args.zarr_path,
        checkpoint=args.checkpoint or None,
        backend_python=resolve_backend_python(args.backend_python),
        device=args.device,
        dataset_root=args.dataset_root,
        val_last_episodes=args.val_last_episodes,
        force_fusion=args.force_fusion,
        config_name=args.config_name,
    )
    try:
        print(f"[s1-inference] {session.describe()}", file=sys.stderr, flush=True)
        if not session.legal_indices:
            raise SystemExit(
                "no dataset anchor has a fully in-episode UVA window. Either the episodes are "
                f"shorter than {session.meta.num_actions * 2} frames or --zarr-path is not the "
                "session this checkpoint was trained on.")
        # Default to a usable anchor rather than index 0: FUJI's anchor set includes frames
        # near an episode start/end that UVA's longer window cannot cover, and landing the
        # viewer on one would greet the operator with an error in the status bar.
        index = args.sample_index if args.sample_index is not None else session.legal_indices[0]
        if args.self_test:
            result = session.infer(index, seed=args.seed)
            print(json.dumps(result_summary(result), indent=2))
            return 0
        from fuji.s1_policy.inference.viewer import run_viewer

        run_viewer(
            session,
            host=args.host,
            port=args.port,
            sample_index=index,
            seed=args.seed,
        )
        return 0
    finally:
        session.close()


if __name__ == "__main__":
    raise SystemExit(main())
