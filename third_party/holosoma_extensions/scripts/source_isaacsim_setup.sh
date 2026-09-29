# Convenience wrapper to activate holosoma + extension IsaacSim environment.
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
# Choose which holosoma directory to use: local or submodule
THIRDPARTY_DIR="$ROOT_DIR/thirdparty"
HOLOSOMA_SUBMODULE_DIR="$THIRDPARTY_DIR/holosoma"
HOLOSOMA_LOCAL_DIR="${HOLOSOMA_LOCAL_DIR:-}"
if [[ -n "$HOLOSOMA_LOCAL_DIR" ]]; then
    echo "[holosoma_ext] Using local holosoma from $HOLOSOMA_LOCAL_DIR"
    HOLOSOMA_DIR="$HOLOSOMA_LOCAL_DIR"
else
    echo "[holosoma_ext] Using holosoma from $HOLOSOMA_SUBMODULE_DIR"
    HOLOSOMA_DIR="$HOLOSOMA_SUBMODULE_DIR"
fi

if [ ! -d "$HOLOSOMA_DIR" ]; then
    echo "ERROR: holosoma not found at $HOLOSOMA_DIR. Run: bash holosoma_extensions/scripts/bootstrap-holosoma.sh" >&2
    return 1
fi

# Choose which conda environment to use
CONDA_ENV_NAME=${ENV_NAME:-hssim}
echo "[holosoma_ext] Using conda environment: $CONDA_ENV_NAME"
CONDA_ENV_NAME="$CONDA_ENV_NAME" source "$HOLOSOMA_DIR/scripts/source_isaacsim_setup.sh"
