#!/usr/bin/env zsh
set -euo pipefail

PROJECT_DIR="${0:A:h}"

# Warm-up convergence was established by the full preset experiment. This
# wrapper collects only the paired six-event baseline.
env \
  EXPERIMENT_MODE=basic \
  RUN_WARMUP_CALIBRATION=0 \
  MAIN_TRIALS="${MAIN_TRIALS:-5}" \
  CHAINS="${CHAINS:-conv_only,conv_relu_pool}" \
  REGIMES="${REGIMES:-low,high}" \
  SCOPES="${SCOPES:-user,kernel}" \
  "$PROJECT_DIR/run_memory_metrics_experiment.zsh"
