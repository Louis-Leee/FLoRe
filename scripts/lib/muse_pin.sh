# shellcheck shell=bash
# Derive FLORE_ACTION_HORIZON from a MuSe-on-UVA checkpoint. Sourced by
# scripts/run_hl_serving.sh and scripts/run_infer_s1_muse.sh so the two cannot drift.
#
# WHY A LAUNCHER HAS TO DO THIS: flore/s1_policy/data/layout.py resolves ACTION_HORIZON from
# FLORE_ACTION_HORIZON at IMPORT time, and everything downstream (the execute-N-of-S choices, the
# recorder's grid, the offline target chunk) is sized from that module constant — so by the time
# any FLoRe code could read the checkpoint, the constant is already fixed. The flow/diffusion/act
# path solves it by reading `config.json`; a UVA checkpoint keeps its config as a pickled
# OmegaConf inside the `.pt`, which only the uvaforce interpreter can unpickle, hence a probe
# subprocess. The probe builds NO model and touches no GPU.
#
# Sets and exports FLORE_ACTION_HORIZON. Returns 2 if an inherited value disagrees with the
# checkpoint — the same "refuse rather than guess which is meant" contract run_hl_serving.sh
# already applies to a config.json horizon. The probe ALSO refuses a checkpoint trained with
# down_sample_steps != 1, whose model step is several recorded frames and would therefore land
# every predicted step at the wrong instant on FLoRe's 30 Hz grid.

muse_pin_action_horizon() {
  local backend_py="$1" flore_root="$2" ckpt="$3"
  local server="$flore_root/scripts/muse_hl_infer_server.py"
  local probe_args=(--print-meta-key num_actions)
  if [ -n "$ckpt" ]; then
    if [ ! -f "$ckpt" ]; then
      echo "[run] FATAL: checkpoint '$ckpt' is not a file" >&2
      return 2
    fi
    probe_args+=(--checkpoint "$ckpt")
  fi

  local horizon
  if ! horizon="$("$backend_py" "$server" "${probe_args[@]}" 2>/dev/null | tail -n 1)"; then
    echo "[run] FATAL: could not probe the checkpoint's chunk length. Re-run for the traceback:" >&2
    echo "[run]   $backend_py $server ${probe_args[*]}" >&2
    return 2
  fi
  case "$horizon" in
    ''|*[!0-9]*)
      echo "[run] FATAL: the checkpoint probe printed '$horizon', not a chunk length. Re-run for" >&2
      echo "[run]        the traceback: $backend_py $server ${probe_args[*]}" >&2
      return 2 ;;
  esac

  if [ -n "${FLORE_ACTION_HORIZON:-}" ] && [ "$FLORE_ACTION_HORIZON" != "$horizon" ]; then
    echo "[run] FATAL: FLORE_ACTION_HORIZON=$FLORE_ACTION_HORIZON in the environment, but the" >&2
    echo "[run]        checkpoint declares num_actions=$horizon. Refusing to guess which is meant." >&2
    echo "[run]        Unset FLORE_ACTION_HORIZON and let the checkpoint decide." >&2
    return 2
  fi
  export FLORE_ACTION_HORIZON="$horizon"
}
