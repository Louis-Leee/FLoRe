#!/usr/bin/env bash
# TERMINAL 2 — Bota F/T bridge (bimanual), WS :8771.
#
# Publishes per-hand SENSOR-frame Newtons (payload-compensated + tared + anti-alias
# filtered). Runs force_bridge.py BY PATH with the Bota env (py3.12 + the Bota gen-A
# toolkit, onboard/requirements/bota.txt; onboard/setup_envs.sh builds it at
# force_deploy/.venv-bota, BOTA_PY=/path/to/python points at another one) and reads the
# two sensors' config JSONs from BOTA_CONFIG (default force_deploy/bota_config/):
#
#   bota_driver/bota_binary.json                               driver config, one for both
#   bota_payload_utils/payload_compensator_{left,right}.json   each tool's payload (FT_ALGO=bota)
#   rnea_dynamic_compensator/rnea_compensator_{left,right}.json  the same for FT_ALGO=rnea
#
# They describe YOUR sensors and the tools bolted to them, so the release ships none.
#
# COMPENSATION ALGORITHM (FT_ALGO, default 'rnea'):
#   rnea = independent single-body Newton-Euler DYNAMIC compensation — removes the
#          tool's inertial reaction m·a that Bota gravity-only leaves as an
#          accel-proportional phantom in force_meas (the escalating startup shake).
#          Needs a per-side rnea_compensator_*.json.
#   bota = the official gravity-only compensator (the old baseline). FT_ALGO=bota.
# The compensated force is low-passed at ANTIALIAS_HZ (default 12 Hz) at the read
# rate BEFORE the 50 Hz policy samples it, so high-frequency residual can't alias
# into the control band. 12 Hz < the 25 Hz policy Nyquist, delay ~13 ms < one 20 ms
# tick. Set ANTIALIAS_HZ=0 to disable.
#
# ⚠️ Validate each hand's RNEA wrench_sign with a static tilt before trusting it: |F|
# should stay ~0 as you reorient the tool in free space; if it GROWS, flip wrench_sign in
# that hand's rnea_compensator_*.json.
#
# Serial→hand: LEFT_SERIAL / RIGHT_SERIAL, a substring of each sensor's name under
# /dev/serial/by-id (the defaults are the paper's rig; swap the two if L/R read reversed).
#
# ⚠️ Do NOT pass --read-hz 0 (two unthrottled read loops hog the GIL → the WS
# never binds). The default 250 Hz is the validated 2-sensor rate; the recurring
# "reading too slow" bota_WARN is BENIGN.
#
# HIGH-LEVEL PLANNER (the force-reactive policy): add --pipeline (forwarded via "$@", or set
# FT_PIPELINE=1) to ALSO publish a 6-DoF gravity-only, tared, anti-aliased wrench
# (left_pipe/right_pipe) — the exact curve the force policy trained on — REGARDLESS of
# FT_ALGO. The WS already fans out to all clients, so the low-level controller (T3) and the
# HL planner both connect to :8771; the low-level's left/right stream is byte-identical
# (it ignores *_pipe). Example: bash 2_ft_bridge.sh --pipeline   (or FT_PIPELINE=1 bash 2_ft_bridge.sh)
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."   # holosoma_extensions

_BOTA="${BOTA_CONFIG:-$(pwd)/force_deploy/bota_config}"
_RNEA=$_BOTA/rnea_dynamic_compensator
_PY="${BOTA_PY:-$(pwd)/force_deploy/.venv-bota/bin/python}"
LEFT_SERIAL="${LEFT_SERIAL:-3870355D3532}"
RIGHT_SERIAL="${RIGHT_SERIAL:-386B355D3532}"

FT_ALGO="${FT_ALGO:-rnea}"        # rnea (dynamic, default) | bota (gravity-only)
ANTIALIAS_HZ="${ANTIALIAS_HZ:-12.0}"
# FT_EMIT_RAW=1 → ALSO publish the truly-RAW wrench (left_raw/right_raw) + gravity-only
# force (left_grav/right_grav) as EXTRA WS keys, for the teleop recorder's raw↔gravity↔RNEA
# toggle. DEFAULT OFF ⇒ the published left/right (what the policy reads) + the read-loop
# timing are byte-identical; the policy ignores the extra keys either way.
EMIT_RAW_ARGS=()
case "${FT_EMIT_RAW:-}" in
  1 | true | TRUE | yes | on) EMIT_RAW_ARGS+=(--emit-raw) ;;
esac

# Fail here, by name, rather than inside the bridge after the serial ports are opened.
[ -x "$_PY" ] || { echo "[2_ft_bridge] no Bota env at $_PY — build it (onboard/setup_envs.sh) or set BOTA_PY=" >&2; exit 2; }
_CFGS=(bota_driver/bota_binary.json
       bota_payload_utils/payload_compensator_left.json bota_payload_utils/payload_compensator_right.json)
if [ "$FT_ALGO" = rnea ]; then
  _CFGS+=(rnea_dynamic_compensator/rnea_compensator_left.json rnea_dynamic_compensator/rnea_compensator_right.json)
fi
for _f in "${_CFGS[@]}"; do
  [ -f "$_BOTA/$_f" ] || { echo "[2_ft_bridge] missing $_BOTA/$_f — your sensors' configs (see the header), or set BOTA_CONFIG=" >&2; exit 2; }
done

exec "$_PY" \
  src/extensions/wbt_wrappers/wbt_wrappers_inference/force_bridge.py \
  --driver-config  "$_BOTA/bota_driver/bota_binary.json" \
  --ft-algo "$FT_ALGO" --antialias-hz "$ANTIALIAS_HZ" "${EMIT_RAW_ARGS[@]}" \
  --left-serial  "$LEFT_SERIAL" --left-compensator  "$_BOTA/bota_payload_utils/payload_compensator_left.json" \
  --left-rnea-config  "$_RNEA/rnea_compensator_left.json" \
  --right-serial "$RIGHT_SERIAL" --right-compensator "$_BOTA/bota_payload_utils/payload_compensator_right.json" \
  --right-rnea-config "$_RNEA/rnea_compensator_right.json" \
  --port 8771 "$@"
