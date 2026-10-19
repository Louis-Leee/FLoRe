#!/usr/bin/env bash
# Workstation side of the ONBOARD rig. Every rig socket stays on 127.0.0.1 on BOTH machines, so the
# planner on the workstation and the bridges + low level on the robot use exactly the URLs they use
# when everything runs on the workstation:
#   robot -> workstation (-L, the page subscribes):  8771 F/T · 8773/8777 cameras · 8784/8785 jaws · 8779 RecordTap
#   workstation -> robot (-R, the low level subscribes): 8770 the S0 wire (hl_serving_5pt.py is its server)
# Two SSH connections: the camera JPEGs (~4 MB/s) get their own TCP stream so a frame never queues
# ahead of a force sample or a wire frame. Each reconnects by itself after a drop. Ctrl-C closes both.
# -tt + a remote `sleep`, NOT -N: only an interactive (pty) session makes OpenSSH set TCP_NODELAY.
# Measured 2026-10-12 on the 88 Hz :8771 stream, same 20 s window: robot-local 15 ms max gap,
# -tt tunnel 15 ms, -N tunnel 53 ms (Nagle holding each small frame for the delayed ACK).
set -uo pipefail
G1="${G1:-unitree@192.168.123.164}"
CTL_L=(8771 8784 8785 8779)
CAM_L=(8773 8777)

# The low level runs ON THE ROBOT, and the workstation can still reach the robot's DDS over the
# cable: a low level here would be a SECOND lowcmd publisher fed by the same :8770 wire. Refuse
# while one runs.
local_ll() { pgrep -af 'wbt_wrappers_inference\.run_policy' | grep -v pgrep; }
if local_ll >/dev/null; then
  echo "[g1-tunnel] REFUSING: a holosoma low level (run_policy) is running ON this workstation — with the onboard"
  echo "[g1-tunnel] rig it must run only on the robot. Stop it first:"; local_ll
  exit 1
fi

busy=()
for p in "${CTL_L[@]}" "${CAM_L[@]}"; do
  ss -Hltn "sport = :$p" | grep -q . && busy+=("$p")
done
if [ ${#busy[@]} -gt 0 ]; then
  echo "[g1-tunnel] REFUSING: this workstation already listens on ${busy[*]} — a process here would shadow the robot's:"
  ss -ltnp 2>/dev/null | grep -E ":($(IFS='|'; echo "${busy[*]}"))\s"
  exit 1
fi

OPTS=(-tt -o BatchMode=yes -o ExitOnForwardFailure=yes -o ServerAliveInterval=1 -o ServerAliveCountMax=3
      -o ConnectTimeout=5 -o ControlMaster=no -o ControlPath=none -o Compression=no)
CTL=(); for p in "${CTL_L[@]}"; do CTL+=(-L "127.0.0.1:$p:127.0.0.1:$p"); done
CTL+=(-R "127.0.0.1:8770:127.0.0.1:8770")
CAM=(); for p in "${CAM_L[@]}"; do CAM+=(-L "127.0.0.1:$p:127.0.0.1:$p"); done

loop() {  # name, pre-hook, forwards...
  local name=$1 pre=$2; shift 2
  while :; do
    [ -n "$pre" ] && ssh -o BatchMode=yes -o ConnectTimeout=5 -o ControlPath=none "$G1" "$pre" </dev/null >/dev/null 2>&1
    echo "[g1-tunnel] $name: connecting ($G1)"
    ssh "${OPTS[@]}" "$@" "$G1" 'exec sleep infinity' </dev/null >/dev/null
    echo "[g1-tunnel] $name: dropped (exit $?) — reconnecting in 1 s"
    sleep 1
  done
}
# A dead previous session can keep holding the robot's 127.0.0.1:8770 listener; free it first.
# set -m: each loop (+ its ssh) is its own process group, so cleanup kills both without orphans.
set -m
loop control "fuser -k -n tcp 8770 2>/dev/null; true" "${CTL[@]}" &
P1=$!
loop cameras "" "${CAM[@]}" &
P2=$!
set +m
trap 'kill -- -$P1 -$P2 2>/dev/null; echo "[g1-tunnel] closed"; exit 0' INT TERM HUP
sleep 3
echo "[g1-tunnel] this workstation now listens on: $(ss -Hltn | awk '{print $4}' | grep -oE ':(8771|8773|8777|8779|8784|8785)$' | tr -d ':' | sort -n | tr '\n' ' ')"
echo "[g1-tunnel] robot listens on 8770 -> this workstation: $(ssh -o BatchMode=yes -o ControlPath=none "$G1" 'ss -Hltn "sport = :8770" | wc -l' </dev/null)"
echo "[g1-tunnel] keep this terminal open; Ctrl-C closes both tunnels."
echo "[g1-tunnel] \"channel N: open failed: connect failed: Connection refused\" = a client here (usually the"
echo "[g1-tunnel]   planner) tried a port whose robot-side bridge is not running yet — normal until they are all up."
# Which robot-side bridges are up, printed when it changes (one multiplexed ssh every 5 s).
NAMES=([8771]="8771 F/T" [8773]="8773 cam L" [8777]="8777 cam R" [8784]="8784 jaw L" [8785]="8785 jaw R"
       [8779]="8779 RecordTap(low level)")
robot_ports() {
  ssh -o BatchMode=yes -o ConnectTimeout=3 -o ControlMaster=auto -o ControlPersist=60 \
      -o "ControlPath=/tmp/g1-tunnel-status-%C" "$G1" 'ss -Hltn' </dev/null 2>/dev/null \
    | awk '{print $4}' | grep -oE ':(8771|8773|8777|8779|8784|8785)$' | tr -d ':' | sort -un | tr '\n' ' '
}
last=""
while :; do
  now=$(robot_ports)
  if [ "$now" != "$last" ]; then
    up=(); down=()
    for p in 8771 8773 8777 8784 8785 8779; do
      if [[ " $now " == *" $p "* ]]; then up+=("${NAMES[$p]}"); else down+=("${NAMES[$p]}"); fi
    done
    echo "[g1-tunnel] robot bridges up: ${up[*]:-none} | not running: ${down[*]:-none}"
    last="$now"
  fi
  if local_ll >/dev/null; then
    echo "[g1-tunnel] ⚠️⚠️  A LOW LEVEL IS RUNNING ON THIS WORKSTATION — TWO lowcmd publishers if the robot runs one too."
    echo "[g1-tunnel]       Stop this one now:"; local_ll
  fi
  sleep 5
done
