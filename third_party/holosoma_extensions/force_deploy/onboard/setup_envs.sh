#!/usr/bin/env bash
# Build the rig's two environments INSIDE the onboard container, at the versions they were run with.
# Run via the wrapper (it mounts the persistent HOME):   ~/flore_rig/rig.sh bash onboard/setup_envs.sh
# Idempotent: re-running only installs what is missing / differs.
#
#   force_deploy/.venv-bota                   py3.12.3  (noble /usr/bin/python3.12)  2_ft_bridge.sh
#   ~/.holosoma_deps/miniconda3/envs/hssim    py3.11.16 (uv CPython)                 3_run_policy_vive_force.sh
set -euo pipefail
ONB="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REQ="$ONB/requirements"
FD="$(dirname "$ONB")"                               # force_deploy
HX="$(dirname "$FD")"                                # holosoma_extensions
HSSIM="$HOME/.holosoma_deps/miniconda3/envs/hssim"
export UV_LINK_MODE=copy

echo "== [1/2] Bota env (py3.12) -> $FD/.venv-bota"
uv venv --allow-existing -p /usr/bin/python3.12 "$FD/.venv-bota"
uv pip install --python "$FD/.venv-bota/bin/python" -r "$REQ/bota.txt"

echo "== [2/2] hssim (py3.11.16) -> $HSSIM"
uv python install 3.11.16
uv venv --allow-existing -p 3.11.16 "$HSSIM"
uv pip install --python "$HSSIM/bin/python" -r "$REQ/hssim-runtime.txt" -c "$REQ/hssim-constraints.txt"
# The same editable source trees the workstation's hssim links (--no-deps: their declared deps
# are the training stack; the runtime set above is what the low level imports).
uv pip install --python "$HSSIM/bin/python" --no-deps \
    -e "$HX/thirdparty/holosoma/src/holosoma" \
    -e "$HX/thirdparty/holosoma/src/holosoma_inference" \
    -e "$HX/src/extensions/common" \
    -e "$HX/src/extensions/wbt_wrappers"

echo "== versions"
for py in "$FD/.venv-bota/bin/python" "$HSSIM/bin/python"; do
  printf '%-70s %s\n' "$py" "$("$py" -c 'import sys; print(sys.version.split()[0])')"
done
uv pip list --python "$FD/.venv-bota/bin/python" 2>/dev/null | grep -iE '^(bota|numpy|websockets) '
uv pip list --python "$HSSIM/bin/python" 2>/dev/null | grep -iE '^(onnxruntime|numpy|unitree-sdk2|torch|mujoco|pin|tyro|websockets|sshkeyboard) '
