"""Apply the mjswan runtime patch needed for holosoma_2pt_wbt eval.

Run once after installing wbt_realtime_eval.  Idempotent — re-running on an
already-patched mjswan is a no-op.

Usage::

    python -m wbt_realtime_eval.patches.apply

Why this exists: mjswan's default ``PolicyRunner.runOnnxInference`` feeds raw
observation values to the ONNX model. The ``holosoma_2pt_wbt`` policy was
trained with an ``EmpiricalNormalization`` that expects pre-clipped (±3σ)
input. Without the pre-clip, the very first inference produces runaway actions.
This patch adds an ``applyOnnxNormalizerPreClip`` step driven by
``policyConfig.extras.normalizer_mean/std/clip`` (a no-op when those keys are
absent, so other policies are unaffected).

The patch is regenerated whenever a maintainer updates the in-tree mjswan
checkout under ``third_party/mjswan/``::

    diff -u /path/to/pristine-mjswan/template/src/core/engine/runtime.ts \\
            third_party/mjswan/src/mjswan/template/src/core/engine/runtime.ts \\
        > src/extensions/wbt_wrappers/wbt_realtime_eval/wbt_realtime_eval/patches/mjswan-runtime-normalizer.patch
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import mjswan

_HERE = Path(__file__).resolve().parent
_TARGET_REL = "template/src/core/engine/runtime.ts"

# Each patch lists one-or-more (relative_target, sentinel) markers that prove
# it fully landed. A patch is "already applied" only when ALL its sentinels are
# present; if SOME are present and some missing (a partial / half-applied
# state, which GNU `patch` does NOT roll back on a later-file hunk failure) we
# re-run `patch` (which detects the already-applied files and skips them) and
# then hard-fail if any sentinel is still missing. Order matters: the force-ext
# patch is generated ON TOP of the normalizer patch, so it applies SECOND.
# Sentinels are matched against the TARGET FILE TEXT, so they must be strings
# that appear ONLY in the real patched code, never in a comment (a comment
# mentioning the method name would falsely report the patch as already applied
# on a half-applied file). Hence the `private <method>(` / `export function
# <name>` forms with the opening paren.
_PATCHES: list[tuple[Path, list[tuple[str, str]]]] = [
    (
        _HERE / "mjswan-runtime-normalizer.patch",
        [("template/src/core/engine/runtime.ts", "private applyOnnxNormalizerPreClip(")],
    ),
    (
        _HERE / "mjswan-runtime-forceext.patch",
        [
            # The force-ext patch spans TWO files: the xfrc injection in
            # runtime.ts AND resetSlidersToDefault in SliderPanel.ts. Both
            # sentinels must be present or the patch is incomplete.
            ("template/src/core/engine/runtime.ts", "private applyExternalForces("),
            ("template/src/core/policy/SliderPanel.ts", "export function resetSlidersToDefault("),
        ],
    ),
    (
        # Hybrid controller: the causal spring wall + F/T sensor + Jᵀ servo in
        # runtime.ts (exp:g1-wbt-2pt-hybrid). Generated ON TOP of the forceext
        # patch (the fork commit range d34e5fd..HEAD), so it applies THIRD.
        # A no-op for non-hybrid policies (all its code is gated on
        # policyConfig.extras.hybrid).
        _HERE / "mjswan-runtime-hybrid.patch",
        [
            ("template/src/core/engine/runtime.ts", "private applySpringWallForces("),
            ("template/src/core/engine/runtime.ts", "private hybridServoStep("),
            # Live-normal wall (2026-08-06). Sentinels must track the patch's LATEST
            # content, not just its original hunks: a tree patched by an EARLIER
            # revision of this same patch file satisfies the two above and would be
            # reported "already applied", silently missing the newer code. Any time
            # this patch grows a feature, add a sentinel for it.
            ("template/src/core/engine/runtime.ts", "wbt_hybrid_wall_live"),
            # F^meas inertial-term opt-out (2026-09-15): the panel's "F meas inertial
            # term" checkbox writes this key and computeHybridFMeas gates m·a_wrist on it.
            ("template/src/core/engine/runtime.ts", "wbt_hybrid_fmeas_inertial"),
        ],
    ),
    (
        # vr_3point tracking-point markers (exp:g1-wbt-3pt* presets): the ONLY
        # runtime change is widening three TrackingCommand members from `private`
        # to `protected` so the WbtTrackingCommand subclass (materialized into the
        # generated custom_commands.ts from wbt_realtime_eval/template/) can read
        # the reference-motion ghost poses. The marker DRAWING code lives in that
        # generated command term, NOT in a patch. Independent of the other three
        # patches (different file), so it applies FOURTH. Sentinels are the two
        # widened declarations (the `protected` keyword only appears here). Uses the
        # trailing type-colon form since these are fields, not methods (no paren).
        _HERE / "mjswan-runtime-3pt-markers.patch",
        [
            ("template/src/core/command/TrackingCommand.ts", "protected readonly ghostData:"),
            ("template/src/core/command/TrackingCommand.ts", "protected referenceVisible:"),
        ],
    ),
]

# Files whose ABSENCE means the installed mjswan predates the in-tree fork's
# live-slider / policy-extras infrastructure (the published 0.6.0 PyPI wheel
# lacks these). The patches cannot apply without them.
_REQUIRED_INFRA = (
    "template/src/core/policy/SliderPanel.ts",
    "template/src/core/policy/extras.ts",
)


def _mjswan_root() -> Path:
    return Path(mjswan.__file__).resolve().parent


def _sentinels_present(root: Path, checks: list[tuple[str, str]]) -> list[bool]:
    """For each (relative_target, sentinel), whether the sentinel is in the file."""
    out: list[bool] = []
    for rel, sentinel in checks:
        path = root / rel
        out.append(path.is_file() and sentinel in path.read_text())
    return out


def _apply_one(root: Path, patch: Path, checks: list[tuple[str, str]]) -> int:
    """Apply one (possibly multi-file) patch idempotently. Returns 0 on
    success / already-applied."""
    if not patch.is_file():
        print(f"Patch file missing: {patch}", file=sys.stderr)
        return 1
    present = _sentinels_present(root, checks)
    if all(present):
        names = ", ".join(s for _, s in checks)
        print(f"  {patch.name}: already applied (sentinels present: {names})")
        return 0
    if any(present):
        missing = [s for (_, s), ok in zip(checks, present) if not ok]
        print(
            f"  {patch.name}: PARTIALLY applied (missing sentinel(s): "
            f"{', '.join(missing)}) — re-running patch to complete.",
            file=sys.stderr,
        )
    # `--forward` skips already-applied hunks (no reverse y/n prompt); `--batch`
    # + stdin=DEVNULL make patch NON-INTERACTIVE so a missing/renamed target
    # auto-skips instead of blocking on a `File to patch:` prompt in a TTY.
    cmd = ["patch", "-p1", "--forward", "--batch", "--input", str(patch)]
    print(f"  Applying {patch.name} to {root}")
    result = subprocess.run(
        cmd, cwd=root, capture_output=True, text=True, stdin=subprocess.DEVNULL
    )
    # A nonzero rc with all sentinels already present is treated as success
    # below (the real gate is the post-apply sentinel check, not patch's rc).
    after = _sentinels_present(root, checks)
    if not all(after):
        missing = [s for (_, s), ok in zip(checks, after) if not ok]
        print(result.stdout)
        print(result.stderr, file=sys.stderr)
        print(
            f"  {patch.name} applied but sentinel(s) still missing: "
            f"{', '.join(missing)} — manual inspection required.",
            file=sys.stderr,
        )
        return result.returncode or 2
    return 0


def main() -> int:
    root = _mjswan_root()
    target = root / _TARGET_REL
    if not target.is_file():
        print(f"mjswan target missing: {target}", file=sys.stderr)
        print(
            "Install the in-tree mjswan checkout: `pip install -e third_party/mjswan`.",
            file=sys.stderr,
        )
        return 1

    # The patches require the in-tree fork's live-slider / policy-extras
    # infrastructure. The published PyPI wheel (even same version string) does
    # NOT ship these files, so a plain `pip install mjswan` cannot be patched.
    missing_infra = [rel for rel in _REQUIRED_INFRA if not (root / rel).is_file()]
    if missing_infra:
        print(
            "This mjswan install lacks the live-slider / policy-extras files the "
            f"patches require: {', '.join(missing_infra)}.",
            file=sys.stderr,
        )
        print(
            "These ship only in the in-tree fork. Install it with "
            "`pip install -e third_party/mjswan` (the published PyPI wheel will "
            "NOT work), then re-run this command.",
            file=sys.stderr,
        )
        return 1

    if shutil.which("patch") is None:
        print("`patch` command not found — install GNU patch or apply manually.", file=sys.stderr)
        return 1

    print(f"Patching mjswan runtime at {root}")
    for patch, checks in _PATCHES:
        rc = _apply_one(root, patch, checks)
        if rc != 0:
            return rc

    print("All mjswan runtime patches applied successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
