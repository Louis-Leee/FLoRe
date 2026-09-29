#!/usr/bin/env bash

# Example Usage:
# ENV_NAME=ext-hssim HOLOSOMA_LOCAL_DIR=/path/to/holosoma bash scripts/setup_isaacsim.sh
# If ENV_NAME is not set, it is defaulted to hssim
# HOLOSOMA_LOCAL_DIR is the path to the local holosoma directory
# If HOLOSOMA_LOCAL_DIR is not set, use the holosoma submodule in the thirdparty directory

set -euo pipefail

# Resolve the directory of this script, following symlinks (works in bash and zsh).
_resolve_script_dir() {
  # shellcheck disable=SC2296
  local source="${BASH_SOURCE[0]:-${(%):-%x}}"
  while [ -h "$source" ]; do
    local dir
    dir="$( cd -P "$( dirname "$source" )" >/dev/null && pwd )"
    source="$(readlink "$source")"
    [[ $source != /* ]] && source="$dir/$source"
  done
  cd -P "$( dirname "$source" )" >/dev/null && pwd
}
SCRIPT_DIR="$(_resolve_script_dir)"
unset -f _resolve_script_dir
ROOT_DIR=$(dirname "$SCRIPT_DIR")
THIRDPARTY_DIR="$ROOT_DIR/thirdparty"
HOLOSOMA_DIR="$THIRDPARTY_DIR/holosoma"

# Check for custom holosoma
HOLOSOMA_LOCAL_DIR="${HOLOSOMA_LOCAL_DIR:-}"
if [[ -n "$HOLOSOMA_LOCAL_DIR" ]]; then
    echo "[holosoma_ext] Using local holosoma from $HOLOSOMA_LOCAL_DIR"
    HOLOSOMA_DIR="$HOLOSOMA_LOCAL_DIR"
else
    echo "[holosoma_ext] Using holosoma from $HOLOSOMA_DIR"
fi

CONDA_ENV_NAME=${ENV_NAME:-hssim}

# Debugging: print conda_env_name and holosoma_dir for sanity check
echo "[holosoma_ext] CONDA_ENV_NAME: $CONDA_ENV_NAME"
echo "[holosoma_ext] HOLOSOMA_DIR: $HOLOSOMA_DIR"

INSTALL_QUADRUPED=${INSTALL_QUADRUPED:-0}
INSTALL_X1=${INSTALL_X1:-0}

# Always run bootstrap to ensure correct branch/commit is checked out
echo "[holosoma_ext] Running bootstrap to ensure correct holosoma version..."
bash "$SCRIPT_DIR/bootstrap-holosoma.sh"

echo "[holosoma_ext] Running holosoma setup_isaacsim.sh from $HOLOSOMA_DIR"
(cd "$HOLOSOMA_DIR" && CONDA_ENV_NAME="$CONDA_ENV_NAME" bash scripts/setup_isaacsim.sh)

echo "[holosoma_ext] Activating conda env: $CONDA_ENV_NAME"
source "$HOLOSOMA_DIR/scripts/source_common.sh"
source "$CONDA_ROOT/bin/activate" "$CONDA_ENV_NAME"

echo "[holosoma_ext] Installing holosoma (editable)"
pip install -e "$HOLOSOMA_DIR/src/holosoma"

echo "[holosoma_ext] Installing extension packages (editable)"
pip install -e "$ROOT_DIR/src/extensions/common"

if [[ "$INSTALL_QUADRUPED" == "1" ]]; then
    echo "[holosoma_ext] Installing quadruped (editable)"
    pip install -e "$ROOT_DIR/src/extensions/quadruped"
else
    echo "[holosoma_ext] Skipping quadruped..."
fi
if [[ "$INSTALL_X1" == "1" ]]; then
    echo "[holosoma_ext] Installing humanoid_retargeting (editable)"
    pip install -e "$ROOT_DIR/src/extensions/humanoid_retargeting"
else
    echo "[holosoma_ext] Skipping humanoid_retargeting..."
fi

echo "[holosoma_ext] Done. Source scripts/source_isaacsim_setup.sh before running training."
