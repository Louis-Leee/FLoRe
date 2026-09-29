#!/usr/bin/env bash
# One-shot installer for the yam_gripper browser-viz stack into the hssim env.
# setup_isaacsim.sh installs holosoma core + common + quadruped only; the
# wbt-realtime-eval CLI additionally needs these four editable packages plus
# their import-time deps (mujoco, nodeenv for the mjswan client build).
set -euo pipefail

# ROOT = THIS checkout, derived from the script's own location (it was hardcoded to a second
# clone under $HOME, which exists on no other machine). HOLOSOMA_EXT_ROOT overrides.
ROOT_DIR="${HOLOSOMA_EXT_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
export PATH="$HOME/.holosoma_deps/miniconda3/envs/${ENV_NAME:-hssim}/bin:$PATH"

echo "=== python: $(which python) ==="
python --version

# Order matters: holosoma_inference (core inference lib) -> wbt_wrappers_inference
# (depends on holosoma_inference) -> mjswan (in-tree, pulls mujoco) ->
# wbt_realtime_eval (depends on mjswan>=0.5). nodeenv is needed by mjswan's
# client build step; install it explicitly so build() doesn't pip-install mid-run.
echo "=== [1/5] holosoma_inference (editable) ==="
pip install -e "$ROOT_DIR/thirdparty/holosoma/src/holosoma_inference"

echo "=== [2/5] wbt_wrappers_inference (editable, from wbt_wrappers root) ==="
pip install -e "$ROOT_DIR/src/extensions/wbt_wrappers"

echo "=== [3/5] mjswan (editable, IN-TREE submodule — required for TS patches) ==="
# The release vendors mjswan next to this package (third_party/mjswan); a standalone
# holosoma_extensions checkout has it as the thirdparty/mjswan submodule. MJSWAN_DIR overrides.
MJSWAN_DIR="${MJSWAN_DIR:-$ROOT_DIR/../mjswan}"
[ -f "$MJSWAN_DIR/pyproject.toml" ] || MJSWAN_DIR="$ROOT_DIR/thirdparty/mjswan"
pip install -e "$MJSWAN_DIR"

echo "=== [4/5] wbt_realtime_eval (editable) ==="
pip install -e "$ROOT_DIR/src/extensions/wbt_wrappers/wbt_realtime_eval"

echo "=== [5/5] nodeenv (mjswan client build toolchain) ==="
pip install "nodeenv>=1.9.0"

echo "=== import smoke test ==="
python - <<'PY'
import importlib
for m in ["mjswan", "mujoco", "holosoma_inference", "wbt_wrappers_inference",
          "wbt_realtime_eval", "pinocchio", "nodeenv"]:
    importlib.import_module(m)
    print("  OK  ", m)
from holosoma_inference.config.config_values.robot import g1_29dof
print("  g1_29dof dof_names:", len(g1_29dof.dof_names))
PY

echo "=== console scripts ==="
for c in wbt-realtime-eval wbt-realtime-eval-patch-mjswan wbt-realtime-eval-serve; do
  printf "  %-32s " "$c"; which "$c" || echo MISSING
done

echo "=== VIZ STACK INSTALL COMPLETE ==="
