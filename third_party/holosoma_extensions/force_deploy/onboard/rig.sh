#!/usr/bin/env bash
# Run ONE rig command inside the onboard container on the G1 Jetson.
#
#   ~/flore_rig/rig.sh [-C DIR] [VAR=VAL ...] CMD [ARGS ...]
#
# CMD runs from force_deploy/ (as on the workstation), or from DIR (relative to the FLoRe
# checkout, or absolute). Leading VAR=VAL words become the command's environment, so a line
#   MODE=5pt-unified-anneal RECORD=1 bash 3_run_policy_vive_force.sh
# becomes
#   ~/flore_rig/rig.sh MODE=5pt-unified-anneal RECORD=1 bash 3_run_policy_vive_force.sh
#
# Same lifetime as a terminal: Ctrl-C reaches the command as SIGINT (force_bridge.py only
# cleans up on SIGINT), and if this terminal or its SSH session goes away the container is
# sent SIGINT and then removed — a rig process never outlives the terminal it was started
# from. (Inside tmux the terminal survives an SSH drop, and so does the process.)
# The container's HOME is ~/flore_rig/home on the robot: onboard/sync_to_g1.sh puts the code
# there, onboard/setup_envs.sh the environments.
set -euo pipefail
RIG_HOME="${RIG_HOME:-$HOME/flore_rig/home}"
IMAGE="${RIG_IMAGE:-flore-rig:latest}"
CHOME=/home/rig                                   # RIG_HOME, as the container sees it
REPO=$CHOME/FLoRe
WD=$REPO/third_party/holosoma_extensions/force_deploy
if [ "${1:-}" = "-C" ]; then
  case "$2" in /*) WD="$2" ;; *) WD="$REPO/$2" ;; esac
  shift 2
fi
[ $# -gt 0 ] || { sed -n '2,17p' "$0"; exit 2; }
[ -d "$RIG_HOME/FLoRe/third_party/holosoma_extensions/force_deploy" ] \
  || { echo "rig: $RIG_HOME/FLoRe missing — run onboard/sync_to_g1.sh on the workstation first" >&2; exit 2; }

TTY=(); [ -t 0 ] && [ -t 1 ] && TTY=(-it)
PROXY=(); for v in http_proxy https_proxy no_proxy; do [ -n "${!v:-}" ] && PROXY+=(-e "$v=${!v}"); done
NAME="flore-rig-$$-$(date +%s)"

stop() {
  docker kill --signal=INT "$NAME" >/dev/null 2>&1 || return 0
  for _ in $(seq 20); do docker inspect "$NAME" >/dev/null 2>&1 || return 0; sleep 0.25; done
  docker rm -f "$NAME" >/dev/null 2>&1 || true
}
trap stop HUP TERM

# docker runs in the BACKGROUND and we `wait`: bash defers a trap while a FOREGROUND child runs,
# so with a plain foreground `docker run` an SSH drop left the container running (measured). A
# trapped signal interrupts `wait` at once. `<&0` keeps the terminal as docker's stdin (a
# non-interactive shell would otherwise give a background job /dev/null).
set +e
docker run --rm "${TTY[@]}" --init --name "$NAME" \
  --privileged --net=host --ipc=host \
  -v /dev:/dev -v /run/udev:/run/udev:ro \
  -v "$RIG_HOME":"$CHOME" \
  -e TERM="${TERM:-xterm}" "${PROXY[@]}" \
  -w "$WD" "$IMAGE" env "$@" <&0 &
DPID=$!
wait "$DPID"; rc=$?
if [ "$rc" -gt 128 ] && kill -0 "$DPID" 2>/dev/null; then   # woken by HUP/TERM: stop() ran in the trap
  stop
  wait "$DPID" 2>/dev/null
fi
exit "$rc"
