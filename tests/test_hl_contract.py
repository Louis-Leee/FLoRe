"""Cross-repo contract pins between the CONVERTER and the TRAINER.

The HL-v2 window geometry is declared twice, in two separate git repositories:

    UMI_collection/processing/umi_convert_core.py   (writes the zarr arrays)
    flore/s1_policy/data/layout.py                   (sizes the model, slices the columns)

They MUST agree. A silent divergence mis-times every label — an 8-bin history over 1 s
read as an 8-bin history over 2 s is numerically indistinguishable in the arrays, and the
policy simply learns the wrong temporal relationship. ``HLDataset._check_file`` asserts a
PRODUCED file's attrs against the layout, which catches a stale dataset; nothing catches
the two source constants drifting apart before a file is ever produced. These tests do.

Skipped (not failed) when the submodule is absent OR predates the HL-v2 converter, so a
checkout whose gitlink has not been bumped yet still runs the rest of the suite. The skip
guard tests for a HL-v2 SYMBOL, not merely for the file: a stale submodule does contain
``umi_convert_core.py``, so an existence check would sail past the guard and then error out
on a missing attribute. The skip is deliberately LOUD about which side is stale, because a
silent skip is how both repos would drift with a green suite — the exact failure this file
exists to prevent.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from flore.s1_policy.data import layout as L

_CONVERTER_DIR = Path(__file__).resolve().parents[1] / "third_party" / "UMI_collection" / "processing"


_STALE_HINT = (
    "third_party/UMI_collection is absent or predates the HL-v2 converter "
    "(no {sym!r} in {mod}). This cross-repo pin cannot run. Update the submodule: "
    "git -C third_party/UMI_collection checkout feat/hl-v2-converter"
)


def _import_from_submodule(module: str, required_symbol: str, extra_paths=()):
    """Import a converter module, SKIPPING if the submodule predates HL-v2.

    The guard is on a SYMBOL, not on file existence: a stale gitlink still ships
    ``umi_convert_core.py``, so an existence check would pass and every test would then
    die on ``AttributeError`` instead of skipping cleanly.
    """
    if not (_CONVERTER_DIR / f"{module}.py").exists():
        pytest.skip(_STALE_HINT.format(sym=required_symbol, mod=module))
    added = [str(_CONVERTER_DIR)] + [str(p) for p in extra_paths]
    for p in added:
        sys.path.insert(0, p)
    try:
        mod = __import__(module)
    except Exception as exc:  # noqa: BLE001 — a stale submodule may fail to import
        pytest.skip(f"{_STALE_HINT.format(sym=required_symbol, mod=module)} ({exc})")
    finally:
        for p in added:
            if p in sys.path:
                sys.path.remove(p)
    if not hasattr(mod, required_symbol):
        pytest.skip(_STALE_HINT.format(sym=required_symbol, mod=module))
    return mod


@pytest.fixture(scope="module")
def core():
    """The converter's pure-resampling core, imported from the submodule."""
    return _import_from_submodule("umi_convert_core", "FORCE_HIST_LEN")


@pytest.fixture(scope="module")
def conv():
    """The converter CLI module (for HANDS_MODES / FT_ROLES / GRIPPER_ROLES)."""
    return _import_from_submodule(
        "convert_to_umi", "HANDS_MODES",
        extra_paths=(_CONVERTER_DIR.parent / "recording",))


def test_window_geometry_constants_agree(core):
    """The numbers that define the contract, pinned across both repos.

    Contract v3 (2026-08-07): NO history (1 bin at t0), 50 steps at 30 Hz. This is THE
    cross-repo guard — the converter writes these into the zarr attrs and
    hl_dataset._check_file gates the loader on the FLoRe copies, so a silent divergence
    mis-times every training label while every shape still matches."""
    # The force-bin count and the action-step count are compared on the DEFAULTS of both
    # sides, because both sides are deliberately re-pinnable per run (converter:
    # --hl-force-hist-len / --hl-action-horizon; FLoRe: FLORE_FORCE_HIST_LEN /
    # FLORE_ACTION_HORIZON) for an L=1 vs L=6 or an H=50 vs H=30 A/B. What must never drift
    # is the default contract they agree on -- and per run, the zarr attr vs the resolved
    # value, which hl_dataset._check_file gates.
    assert core.HLGrid().force_hist_len == L.FORCE_HIST_LEN_DEFAULT == 1
    assert core.HLGrid().hist_span_s == pytest.approx(1.0 / 30.0)
    assert core.HLGrid().action_horizon == L.ACTION_HORIZON_DEFAULT == 50
    assert core.ACTION_HZ == L.ACTION_HZ == 30.0
    assert core.ACTION_DT == L.ACTION_DT == pytest.approx(1.0 / 30.0)
    assert core.HLGrid().action_span_s == pytest.approx(50.0 / 30.0)
    # The RATE is what both sides pin exactly; the horizon pin may only shorten the COUNT,
    # so L.ACTION_SPAN_S is asserted below as a derivation, not against a literal.
    # NOT cross-checked here: the two RESOLVED counts. The converter's GRID is re-pinned
    # only by its own CLI and the FLoRe copy only by the env, so a FLoRe-side pin cannot
    # reach `core` and comparing them would merely forbid running pytest under a pin. The
    # per-run guard that matters is zarr-attr vs resolved value, in
    # test_hl_dataset.py::_check_file's tests -- a file's own recorded grid, not a global.
    # Each side must be internally self-consistent: spans are DERIVED from (count, rate),
    # so a hand-edited span literal on either side shows up here.
    assert L.ACTION_DT == pytest.approx(1.0 / L.ACTION_HZ)
    assert L.ACTION_SPAN_S == pytest.approx(L.ACTION_HORIZON * L.ACTION_DT)
    assert L.FORCE_HIST_SPAN_S == pytest.approx(L.FORCE_HIST_LEN * L.ACTION_DT)
    assert core.ACTION_DT == pytest.approx(core.ACTION_SPAN_S / core.ACTION_HORIZON)
    assert core.FORCE_HIST_DT == pytest.approx(core.FORCE_HIST_SPAN_S / core.FORCE_HIST_LEN)
    # The one input bin is exactly one action step wide (v3): the force the model SEES at
    # t0 is the same average the first predicted step is scored against.
    assert core.FORCE_HIST_DT == pytest.approx(core.ACTION_DT)


def test_force_dim_agrees(core):
    """A wrench is 6-DoF on both sides; the converter's arrays are [.., 6]."""
    assert L.FORCE_DIM == 6
    # The converter slices ft[:, 1:7] (t + 6 wrench components), so its wrench width is
    # fixed by the recording column layout.
    sys.path.insert(0, str(_CONVERTER_DIR.parent / "recording"))
    try:
        import recording_format as fmt

        assert len(fmt.FT_COLS) == 1 + L.FORCE_DIM, fmt.FT_COLS
    finally:
        sys.path.remove(str(_CONVERTER_DIR.parent / "recording"))


def test_hands_modes_match_layout_modes(conv):
    """Every layout mode must be producible by exactly one --hands value, and every
    --hands value must correspond to a layout mode. Otherwise a mode is either
    unbuildable or a flag produces arrays no layout can read."""
    layout_modes = {m for m in L.MODES if m != "legacy32"}
    assert set(L.HANDS_FLAG_BY_MODE) == layout_modes
    assert set(L.HANDS_FLAG_BY_MODE.values()) == set(conv.HANDS_MODES)
    # ...and the mapping is injective over the modes that come STRAIGHT out of a conversion (no
    # two share a flag), which is what makes `attrs.hands_mode` a sufficient gate for them.
    #
    # Two families deliberately REUSE their source conversion's flag, because each is a
    # POST-PROCESS of one rather than a conversion of its own — demanding a distinct flag would
    # mean a second copy of every camera frame of the same session:
    #
    #   * `wholebody*`   = a `--hands both_head_dual` zarr + the 29-DoF `wb_*` arrays from
    #                      UMI_collection's mc_to_hl_wholebody.py.  Discriminator on disk:
    #                      `attrs.wb_windows` (pinned by test_diffusion_umi_wholebody.py).
    #   * `fivepoint_*`  = one of THOSE + the eight foot arrays from
    #                      scripts/add_foot_targets.py.  Discriminator: `attrs.wb_foot_targets`
    #                      (pinned by test_diffusion_umi_5pt.py).
    #
    # So a `fivepoint_*` file is a strict superset of a `wholebody` one, which is a strict
    # superset of a `both_head_dual` one, and ONE recording session feeds all three.
    derived = set(L._WB_MODES) | set(L._5PT_MODES)
    kp_flags = [f for m, f in L.HANDS_FLAG_BY_MODE.items() if m not in derived]
    assert len(set(kp_flags)) == len(kp_flags)
    for mode in derived:
        assert L.HANDS_FLAG_BY_MODE[mode] in set(kp_flags), mode


def test_hands_flag_helper_covers_every_mode():
    for mode in L.MODES:
        if mode == "legacy32":
            with pytest.raises(ValueError, match="no --hands flag"):
                L.hands_flag(mode)
        else:
            assert L.hands_flag(mode) in ("left", "right", "both", "both_head",
                                          "both_head_dual")
    with pytest.raises(ValueError):
        L.hands_flag("nonsense")


def test_tracker_roles_match_the_converters_hands_modes(conv):
    """The tracker roles a layout mode needs must be exactly the roles the converter's
    matching --hands value requires — else a mode passes the FLoRe-side check and then
    fails (or silently under-reads) in the converter."""
    for mode, flag in L.HANDS_FLAG_BY_MODE.items():
        assert L.tracker_roles_required(mode) == conv.HANDS_MODES[flag], mode


def test_sensor_roles_match_the_converters(conv):
    """DECLARED hardware must agree: the layout sizes the model's force/state inputs
    from force_roles/gripper_roles, while the converter decides which roles actually get
    force/gripper arrays from HL_FT_ROLES/HL_GRIPPER_ROLES. A disagreement means the model
    has an input the data never fills (or vice versa).

    Read PER MODE, through the converter's own resolver: `both_head` and `both_head_dual`
    need the identical three pucks and differ ONLY here, so a module-level constant could
    not express both and this check would silently pass for whichever one it matched."""
    for mode, flag in L.HANDS_FLAG_BY_MODE.items():
        lay = L.get_layout(mode)
        roles = set(conv.HANDS_MODES[flag])
        want_force = roles & set(conv.hl_ft_roles(flag))
        want_grip = roles & set(conv.hl_gripper_roles(flag))
        if set(lay.force_roles) != want_force:
            # right_only is a KNOWN gap: the layout declares a right-hand sensor on the
            # assumption the hardware moved, but the converter is hardcoded left-only.
            # Assert it is the ONLY exception, so a new divergence still fails.
            assert mode == "right_only", (
                f"{mode}: layout force_roles={lay.force_roles} but converter would emit "
                f"force for {sorted(want_force)}")
        if set(lay.gripper_roles) != want_grip:
            assert mode == "right_only", (
                f"{mode}: layout gripper_roles={lay.gripper_roles} but converter would "
                f"emit gripper for {sorted(want_grip)}")


def test_zarr_key_names_match_on_both_sides(conv):
    """The dataset's key construction must be exactly the converter's emission, name for
    name, for every quantity. Enumerated programmatically rather than trusted."""
    from flore.s1_policy.data.hl_dataset import HLDataset

    for mode, flag in L.HANDS_FLAG_BY_MODE.items():
        if mode == "right_only":
            continue                     # known gap, covered above
        lay = L.get_layout(mode)
        roles = conv.HANDS_MODES[flag]

        # What the converter emits (mirrors _build_hl_windows' key construction).
        emitted = set()
        for role in roles:
            p = conv._hl_prefix(role)
            emitted |= {f"{p}action_pose", f"{p}action_valid"}
            if role in conv.hl_gripper_roles(flag):
                emitted |= {f"{p}action_gripper", f"{p}action_gripper_valid"}
            if role in conv.hl_ft_roles(flag):
                emitted |= {f"{p}force_history", f"{p}force_history_valid",
                            f"{p}action_force", f"{p}action_force_valid"}

        # What the dataset REQUIRES — mirroring _check_file's own list exactly, including the
        # gripper VALIDITY companion (it is required, not opportunistic: a dead-encoder take is
        # zero-filled and 0.0 m is a legal "closed"). The pose pair is keyed on
        # `tracker_roles_required`, which is what _check_file uses, and which differs from
        # `lay.roles` for the whole-body mode: its 29 joint columns come from a completion
        # conditioned on the CHEST too, so `hl_head_action_valid` is required there even though
        # no chest block exists to iterate.
        required = set()
        for role in set(L.tracker_roles_required(mode)) | set(lay.roles):
            if role in L.FOOT_ACTION_KEYS:
                # The `fivepoint_*` family's two FOOT roles are NOT converter output — nothing
                # on this rig sees a foot. Their arrays are `wb_*`, FK'd from the completion's
                # joints by scripts/add_foot_targets.py, and `_check_file` skips them here for
                # exactly this reason (`_check_foot_targets` checks them instead). Mirroring
                # that skip is what keeps this test a mirror rather than a second opinion.
                continue
            required |= {f"hl_{role}_action_pose", f"hl_{role}_action_valid"}
            if role in lay.gripper_roles:
                required |= {f"hl_{role}_action_gripper", f"hl_{role}_action_gripper_valid"}
            if role in lay.force_roles:
                required |= {f"hl_{role}_force_history", f"hl_{role}_force_history_valid",
                             f"hl_{role}_action_force", f"hl_{role}_action_force_valid"}

        missing = required - emitted
        assert not missing, f"{mode}: dataset requires keys the converter never emits: {sorted(missing)}"
        # With `required` now a faithful mirror there is nothing left over on real modes; keep
        # the check so a converter that starts emitting a NEW key has to be noticed here rather
        # than absorbed by a wildcard.
        extra = emitted - required
        assert not extra, f"{mode}: converter emits keys no layout reads: {sorted(extra)}"
        assert HLDataset is not None      # imported for the side-effect of a clean import


def test_prefix_helper_matches_the_dataset_naming(conv):
    """One function on each side builds 'hl_<role>_'; they must be identical."""
    for role in ("left", "right", "head"):
        assert conv._hl_prefix(role) == f"hl_{role}_"


def test_gripper_normalization_reference_matches_the_rig_calibration(conv):
    """The trainer divides gripper width by GRIPPER_WIDTH_REF_M to reach [0,1]. That must
    match the physical full-open width the converter calibrates against and now records
    in attrs (gripper_width_open_m) — otherwise every gripper label is scaled wrong, and
    UMI's 0.08 m grasp threshold lands at the wrong normalized value."""
    from flore.s1_policy.data.transforms import GRIPPER_WIDTH_REF_M

    calib_dir = _CONVERTER_DIR.parent / "robot_assets" / "calibration"
    if not (calib_dir / "gripper_range.json").exists():
        pytest.skip("rig gripper calibration not present")
    import json

    rig = json.loads((calib_dir / "gripper_range.json").read_text())
    assert GRIPPER_WIDTH_REF_M == pytest.approx(rig["width_open_m"]), (
        f"training normalizes by {GRIPPER_WIDTH_REF_M} m but the rig calibration says "
        f"{rig['width_open_m']} m")


def test_tare_settle_is_recorded_and_positive(core):
    """The tare span width reaches the zarr attrs, so a loader can reason about it."""
    assert core.TARE_SETTLE_S > 0.0
    spans = core.tare_transient_spans([100.0])
    # BACKWARD from the tare — the contaminated samples precede it.
    assert spans[0, 1] == pytest.approx(100.0)
    assert spans[0, 0] == pytest.approx(100.0 - core.TARE_SETTLE_S)


# ---------------------------------------------------------------------------
# Normalization robustness for one-sided columns
# ---------------------------------------------------------------------------
def test_one_sided_columns_are_widened_to_min_max():
    """REGRESSION. Two rot6d columns are the diagonal cosines R_rel[0,0] / R_rel[1,1] of
    a SMALL relative rotation, so they pile against a hard ceiling at exactly +1.0. q99
    then lands ON the ceiling and [q01, q99] measures only the narrow upper tail, while
    the real lower tail runs far past q01.

    Measured on real data before the fix: spans of 0.137 and 0.070 mapped the true
    minimum to -9.1 and -15.1 in normalized space, against a ~[-1, 1] target — exactly
    the out-of-range AdaLN conditioning normalization exists to prevent, and invisible
    unless inspected per column. After: worst |normalized| 15.08 -> 1.89."""
    import numpy as np

    from flore.s1_policy.data.norm_stats import normalize_quantile
    from scripts.fit_hl_norm_stats import _quantiles_masked

    rng = np.random.default_rng(0)
    # A cosine-like column: 99% of the mass jammed just under the +1.0 ceiling, with a
    # real lower tail to ~0.3. This reproduces the measured shape of rot6d cols 3 and 7,
    # whose (q01 - min)/span was 3.7 and 4.8.
    cos_col = 1.0 - np.abs(rng.normal(scale=0.02, size=20000))
    cos_col[:40] = np.linspace(0.30, 0.90, 40)
    # A symmetric column with a few wild outliers. Quantile clipping is exactly right
    # here, so it must NOT be widened — that is what q01/q99 is for.
    sym_col = rng.normal(size=20000)
    sym_col[:5] = 50.0

    stats = _quantiles_masked([cos_col, sym_col], 2, "test")

    # Only the CLIPPED side of the clipped column moved: q01 -> the observed min...
    assert stats.q01[0] == pytest.approx(cos_col.min())
    # ...while its q99 stays the (already wall-hugging) quantile, not stretched further.
    assert stats.q99[0] <= 1.0
    # The real tail now lands far closer to [-1, 1] than the raw quantile pair allows.
    one = type(stats)(q01=stats.q01[:1], q99=stats.q99[:1])
    n_fixed = abs(normalize_quantile(cos_col.reshape(-1, 1), one)).max()
    plain = _quantiles_masked([cos_col], 1, "test", robust_one_sided=False)
    n_plain = abs(normalize_quantile(cos_col.reshape(-1, 1), plain)).max()
    assert n_plain > 5.0, n_plain            # the defect, reproduced
    assert n_fixed <= 1.01, n_fixed          # ...and removed
    assert n_fixed < n_plain / 4

    # The outlier column keeps its robust quantiles: NOT widened to 50.
    assert stats.q99[1] < 5.0, stats.q99[1]
    assert stats.q01[1] > -5.0, stats.q01[1]
