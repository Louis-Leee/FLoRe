"""Live VIVE-teleop build entry for the 3pt hybrid UNIFIED anneal policy.

The browser twin of the rig's ``MODE=3pt-unified-anneal``
(``force_deploy/3_run_policy_vive_force.sh``): the SAME
``wbt-vive-teleop-bridge`` process, the SAME pelvis-frame wire, the SAME
checkpoint — but the whole-body controller runs in browser MuJoCo/WASM instead of
on the G1. Use it to rehearse a teleop session, to sanity-check tracker
axes/handedness before touching hardware, and to read the commanded-vs-achieved
correspondence at the three tracked points.

Preset: ``g1-29dof-holosoma-3pt-hybrid-unified-anneal-wbt-live`` (170D/frame x 4
= 680D). Only ``vr_3point_pos_tcp`` / ``vr_3point_orn`` are live; everything else
is what any hybrid bundle uses — ``command_lower_body`` / ``motion_ref_ori_b``
from the injected ``idle_default`` standing clip, ``force_cmd_local`` from the
sliders, ``force_meas_local`` / ``integrator_state`` / ``x_tgt_shift`` from the
mjswan-fork runtime's causal spring wall + Jᵀ servo, and ``contact_phase``
recomputed from that F^meas by the same hysteresis+LPF the deploy runtime uses.

Additive: it reuses ``main.setup_builder`` wholesale and adds nothing to the 2pt
live path (``main_vive``) or to any replay preset.

Two shells. **(1)** the tracker bridge — 3pt mode, TCP hand point::

    MODE=3 bash force_deploy/1_vive_bridge.sh
    #   == scripts/_vive_teleop_bridge.sh --orientation on --mode 3pt   (HAND_POINT=tcp default)

**(2)** this build (or ``scripts/_vive_teleop3_build.sh``, which pins PYTHONPATH
to this checkout and fills in the defaults)::

    python -m wbt_realtime_eval.main_vive3 \\
        --model-path logs/WBT-Holosoma/wda3h4ll/model_19000.onnx \\
        --motion-dir ~/data/sonic_manipulation_test \\
        --vive-ws-url ws://localhost:8770

Then click **Recalibrate** in the browser (holds the calibration pose for 5 s,
resets the sim, re-anchors the bridge) — same button, same bridge control
messages, as the 2pt live path.

Two things worth knowing before reading the result:

* **The bridge must run ``--hand-point tcp``** (its default). This preset's
  position term is ``vr_3point_pos_tcp`` — the 0.242 m fingertip TCP — and the
  0.18 m-lever variant has the IDENTICAL 9D width, so a mismatch is invisible to
  every dimension check. We refuse the build unless the checkpoint's own actor
  obs names the ``_tcp`` term, but nothing here can see which point the bridge is
  streaming, so that half is on the launcher.
* **The Jᵀ servo runs in ``servo_enabled`` mode, not the deploy's observe-only
  mode.** These checkpoints deploy with the servo demoted to a reference
  generator (Δq never enters the PD path) while the integrator keeps winding; the
  browser runtime only has all-on / all-off. With ``F^cmd = 0`` — the slider
  default, and the whole pure-position teleop regime — the wall never seeds,
  F^meas stays 0, the integrator stays 0 and Δq is identically 0, so the two
  modes are bit-identical. They diverge only once you push an F^cmd slider; the
  Hybrid panel's live servo toggle is the A/B for that.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from loguru import logger

import mjswan  # noqa: F401  (ensures mjswan import side effects match main.py)

from wbt_realtime_eval.main import _read_onnx_metadata, setup_builder
from wbt_realtime_eval.main_vive import validate_home_anchor_matches_meta
from wbt_realtime_eval.presets import get_preset
from wbt_realtime_eval.vive import load_home_anchor

_LIVE_3PT_PRESET = "g1-29dof-holosoma-3pt-hybrid-unified-anneal-wbt-live"

# The two live term keys, and the home_anchor.json field each falls back to before
# the first bridge frame (NEVER zeros: a zero 3pt pos commands all three points to
# the pelvis origin, which collapses the arms and yanks the torso).
_LIVE_TERM_HOME_FIELD = {
    # 9D = [L_xyz, R_xyz, Head_xyz], pelvis frame, hands at the 0.242 m TCP.
    "vr_3point_pos_tcp_live": "home_pos_pelvis_3pt_tcp",
    # 12D = 3x xyzw quaternion (the vr_3point_orn rep, NOT rot6d).
    "vr_3point_orn_live": "home_orn_quat_xyzw_3pt",
}


# Training and inference spell two of the obs terms differently (`dof_pos` /
# `dof_vel` on the training branch, `joint_pos` / `joint_vel` in the inference
# ObservationConfig this package's presets mirror). Same term, same width, same
# slot — a naming difference must never trip a layout check. Mirrored from
# ``HolosomaWBTPolicy._OBS_TERM_CANON`` (holosoma_wbt.py), the source of truth for
# the equivalent deploy-side guard ``_validate_obs_term_order``; duplicated rather
# than imported because that module pulls in the whole deploy inference stack and
# this runs at BUILD time.
_OBS_TERM_CANON = {
    "dof_pos": "joint_pos",
    "dof_vel": "joint_vel",
    "joint_pos": "joint_pos",
    "joint_vel": "joint_vel",
    "motion_command": "command",
    "tml_command": "command",
    "command": "command",
}


def _canon_obs_term(name: str) -> str:
    """Strip a leading zero-padded ``obsNN_`` prefix, then canonicalize the label."""
    if name[:3] == "obs" and "_" in name:
        head, _, rest = name.partition("_")
        if head[3:].isdigit():
            name = rest
    return _OBS_TERM_CANON.get(name, name)


def _checkpoint_actor_obs_terms(meta: dict[str, str]) -> tuple[list[str], bool] | None:
    """The checkpoint's ACTOR obs terms, canonicalized, plus whether their order is
    trustworthy. None if the metadata has no actor group.

    Holosoma's obs manager concatenates in ``sorted(keys())`` (LEXICAL) order, so
    the on-the-wire sequence is the sorted key order — not the JSON's insertion
    order. Training keys are zero-padded (``obs00_..obs13_``) precisely so lexical
    == numeric; when the pad is wide enough for the term count, the sorted order IS
    the layout and can be asserted positionally. A bare or too-narrow scheme
    (``obs10_`` sorts before ``obs2_``) makes order ambiguous, so we report
    ``False`` and the caller drops to a set check. Same rule as the deploy-side
    ``_validate_obs_term_order``.
    """
    try:
        exp = json.loads(meta.get("experiment_config", "{}"))
    except (json.JSONDecodeError, TypeError):
        return None
    terms = exp.get("observation", {}).get("groups", {}).get("actor_obs", {}).get("terms", {})
    if not isinstance(terms, dict) or not terms:
        return None
    keys = [str(k) for k in terms]

    def _pad_width(k: str) -> int:
        if k[:3] == "obs" and "_" in k:
            head = k.partition("_")[0]
            if head[3:].isdigit():
                return len(head[3:])
        return -1  # not an obsNN_ key -> disqualifies order

    widths = {_pad_width(k) for k in keys}
    order_trusted = -1 not in widths and len(keys) < 10 ** min(widths)
    return [_canon_obs_term(k) for k in sorted(keys)], order_trusted


def _validate_actor_obs_matches_preset_named(meta: dict[str, str], preset_name: str) -> None:
    """Refuse the build unless the checkpoint's actor obs IS this preset's layout.

    The normalizer-length check in ``setup_builder`` only proves the checkpoint is
    680D. That is not enough here, because the two things most likely to be wrong
    are both width-preserving:

    * ``vr_3point_pos`` (the 0.18 m VR lever) vs ``vr_3point_pos_tcp`` (the 0.242 m
      fingertip TCP) — same 9D, same three bodies, targets 6.7 cm apart per hand.
      This is the same trap ``_validate_obs_term_order`` guards on the deploy side,
      and the reason the pre-2026-08-03 unified checkpoints (iss0qs71 / m3s0pw9i)
      cannot be served against this preset.
    * a term REORDER, e.g. ``contact_phase`` landing before ``force_meas_local``,
      which shifts every later slot by one.

    It is a free check: the checkpoint states its own layout.
    """
    found = _checkpoint_actor_obs_terms(meta)
    if found is None:
        logger.warning(
            "ONNX experiment_config has no actor_obs terms; cannot verify the obs layout. "
            "The normalizer WIDTH check still applies, but the width-preserving traps "
            "would NOT be caught: a lever-vs-TCP position term (identical 9D/15D), or the "
            "dense-vs-global-vel collision at 732-D. Verify the checkpoint by hand."
        )
        return
    ckpt, order_trusted = found
    want = [_canon_obs_term(k.removesuffix("_live")) for k in get_preset(preset_name).obs_keys]

    def _fail(what: str) -> None:
        extra = set(ckpt) - set(want)
        missing = set(want) - set(ckpt)
        hint = ""
        for lever, tcp in (("vr_3point_pos", "vr_3point_pos_tcp"),
                           ("vr_5point_pos", "vr_5point_pos_tcp")):
            if lever in extra and tcp in missing:
                hint = (
                    "\n  This checkpoint tracks the 0.18 m VR LEVER, not the 0.242 m fingertip "
                    "TCP. IDENTICAL width, target 6.7 cm off per hand — use a TCP-unified "
                    "retrain (e.g. run wda3h4ll for 3pt, dlcxc59d for 5pt), not iss0qs71 / "
                    "m3s0pw9i."
                )
        if "command" in missing and any(k.startswith("vr_") for k in extra):
            hint = (
                "\n  This looks like a KEYPOINT checkpoint being built as DENSE. 732-D is also "
                "g1-29dof-holosoma-3pt-hybrid-global-vel-wbt's width, so the normalizer check "
                "cannot separate them — that is exactly what this guard is for."
            )
        raise ValueError(
            f"Checkpoint actor obs {what} preset {preset_name!r}.\n"
            f"  checkpoint: {tuple(ckpt)}\n"
            f"  preset    : {tuple(want)}" + hint
        )

    if sorted(ckpt) != sorted(want):
        _fail("does not contain the same terms as")
    if order_trusted and ckpt != want:
        # Same terms, different sequence: the width gate passes and the policy gets
        # a scrambled vector. Name the offending slots.
        slots = "\n".join(
            f"    slot {i}: preset={w!r} but checkpoint={c!r}"
            for i, (w, c) in enumerate(zip(want, ckpt, strict=True))
            if w != c
        )
        raise ValueError(
            f"Checkpoint actor obs ORDER does not match preset {preset_name!r} "
            f"(the width matched but the per-term sequence does NOT — the policy "
            f"would receive a scrambled observation):\n{slots}"
        )
    if not order_trusted:
        logger.warning(
            "Checkpoint obs keys are not safely zero-padded, so their lexical order is "
            "ambiguous; verified the term SET only (order not asserted)."
        )
    logger.info(f"Actor obs layout matches preset {preset_name!r} ({len(want)} terms).")


def _validate_actor_obs_matches_preset(meta: dict[str, str]) -> None:
    """The 3pt-live entry's own guard — the named check pinned to this module's preset."""
    _validate_actor_obs_matches_preset_named(meta, _LIVE_3PT_PRESET)


def _live_3pt_per_key_params(ws_url: str) -> dict[str, dict]:
    """Per-term config for the live 3pt obs classes: the WebSocket URL the browser
    connects to, plus the default-pose HOME fallback each returns until the first
    live frame arrives.

    HOME is the flattened 3-point default-pose target from ``vive/home_anchor.json``
    — the SAME arrays the bridge calibrates onto (``home_pos_pelvis_3pt_tcp`` /
    ``home_orn_quat_xyzw_3pt``), so browser fallback and bridge anchor cannot drift.
    """
    anchor = load_home_anchor()
    out: dict[str, dict] = {}
    for key, field in _LIVE_TERM_HOME_FIELD.items():
        rows = anchor.get(field)
        if not rows:
            raise ValueError(
                f"vive/home_anchor.json is missing {field!r}, the HOME fallback for the live "
                f"term {key!r}. Recompute the anchor (it must carry the 3pt TCP arrays)."
            )
        out[key] = {"vive_ws_url": ws_url, "home": [float(v) for row in rows for v in row]}
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="wbt-vive-teleop-3pt",
        description=(
            "Live VIVE 3-tracker teleop of the 3pt hybrid UNIFIED anneal policy in "
            "browser MuJoCo — the sim twin of MODE=3pt-unified-anneal. Run the tracker "
            "bridge in 3pt mode separately (it holds the pysurvive singleton)."
        ),
    )
    parser.add_argument("--model-path", required=True, type=Path, help="A 680-D vr_3point_pos_tcp checkpoint.")
    parser.add_argument(
        "--motion-dir",
        required=True,
        type=Path,
        help="Reference motions (still required — supplies the non-VR obs terms "
        "via the injected idle_default clip; any clips here populate the picker).",
    )
    parser.add_argument("--output-dir", default=Path("dist/wbt_realtime_eval_vive3"), type=Path)
    parser.add_argument(
        "--vive-ws-url",
        default="ws://localhost:8770",
        help="WebSocket URL of the VIVE bridge the browser connects to (must match "
        "wbt-vive-teleop-bridge --host/--port). The bridge MUST be in --mode 3pt "
        "--hand-point tcp.",
    )
    parser.add_argument(
        "--scene-xml",
        default=None,
        type=Path,
        help="MuJoCo render scene (absolute or repo-root-relative). If omitted, "
        "auto-derived from the ONNX robot_urdf_path (yam checkpoint -> yam scene).",
    )
    parser.add_argument("--no-launch", action="store_true", help="Build only; don't start the server.")
    args = parser.parse_args()

    model_path = args.model_path.resolve()
    logger.info(f"Live VIVE 3pt teleop build: preset={_LIVE_3PT_PRESET} ws={args.vive_ws_url}")
    # One ORT load for both metadata checks (the checkpoint is ~780 MB unstripped).
    meta = _read_onnx_metadata(model_path)
    # Refuse a checkpoint whose default pose the committed HOME anchor was not
    # computed for (the live targets would be mis-anchored).
    validate_home_anchor_matches_meta(meta, source=str(model_path))
    # Refuse a 680-D checkpoint whose actor obs is a DIFFERENT 680-D layout — most
    # importantly the 0.18 m-lever position term instead of the 0.242 m TCP one.
    _validate_actor_obs_matches_preset(meta)

    builder = setup_builder(
        preset=_LIVE_3PT_PRESET,
        model_path=model_path,
        motion_dir=args.motion_dir.resolve(),
        output_dir=args.output_dir.resolve(),
        scene_xml=args.scene_xml,
        per_key_params=_live_3pt_per_key_params(args.vive_ws_url),
    )
    app = builder.build(output_dir=str(args.output_dir.resolve()))
    logger.info(
        "Built. Start the tracker bridge in another shell: "
        "`MODE=3 bash force_deploy/1_vive_bridge.sh` (3pt + TCP hand "
        "point), hold the calibration pose ~3 s, then click Recalibrate in the browser."
    )
    if args.no_launch or os.getenv("MJSWAN_NO_LAUNCH") == "1":
        return
    app.launch()


if __name__ == "__main__":
    main()
