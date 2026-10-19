#!/usr/bin/env bash
# DRY RUN of the low level: pass as PY= to 3_run_policy_vive_force.sh. Every launcher check that runs Python
# ("$PY" - <<heredoc>: number check, hand-geometry probe, ONNX family probe) runs for real; the
# final `$PY -m wbt_wrappers_inference.run_policy ...` is printed with its HOLOSOMA_* environment
# and NOT executed, so the robot is never touched.
#
#   ~/flore_rig/rig.sh PY=/home/rig/FLoRe/third_party/holosoma_extensions/force_deploy/onboard/py_dryrun.sh \
#       MODE=5pt-unified-anneal HOLOSOMA_VIVE_RELATIVE=0 RECORD=1 FORCE_CMD_WIRE=1 bash 3_run_policy_vive_force.sh
#
# (Do not use `--help` for this: run_policy.main() catches tyro's SystemExit and falls back to its
# manual parser, which then STARTS the policy.)
REAL="${REAL_PY:-$HOME/.holosoma_deps/miniconda3/envs/hssim/bin/python}"
if [ "${1:-}" = "-m" ] && [ "${2:-}" = "wbt_wrappers_inference.run_policy" ]; then
  echo "[py-dryrun] WOULD RUN (not executed): $REAL $*"
  env | grep -E '^(HOLOSOMA_|OMP_NUM|OPENBLAS_NUM|MKL_NUM|NUMEXPR_NUM)' | sort | sed 's/^/[py-dryrun]   /'
  exit 0
fi
exec "$REAL" "$@"
