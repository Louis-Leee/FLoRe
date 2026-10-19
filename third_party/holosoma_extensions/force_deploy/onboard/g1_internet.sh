#!/usr/bin/env bash
# Temporary internet for the G1 Jetson through the workstation — needed ONLY to build the image or
# run setup_envs.sh (the robot has no route out). Foreground; Ctrl-C ends it.
#   workstation:  bash onboard/g1_internet.sh
#   robot      :  http_proxy=http://127.0.0.1:3128 https_proxy=http://127.0.0.1:3128 <command>
#                 (rig.sh forwards http_proxy/https_proxy into the container)
set -euo pipefail
G1="${G1:-unitree@192.168.123.164}"
PORT="${PORT:-3128}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python3 "$HERE/proxy.py" "$PORT" > /dev/null & PX=$!
trap 'kill $PX 2>/dev/null' EXIT INT TERM
sleep 0.5
echo "[g1-internet] proxy on this workstation 127.0.0.1:$PORT -> robot 127.0.0.1:$PORT (Ctrl-C ends)"
ssh -tt -o ExitOnForwardFailure=yes -o ServerAliveInterval=5 -o ControlPath=none \
    -R "127.0.0.1:$PORT:127.0.0.1:$PORT" "$G1" 'exec sleep infinity' </dev/null >/dev/null
