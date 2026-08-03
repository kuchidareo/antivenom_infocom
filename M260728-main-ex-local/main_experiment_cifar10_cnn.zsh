#!/usr/bin/env zsh
set -euo pipefail

SCRIPT_DIR="${0:A:h}"

# The expanded CIFAR-10 partition makes five trials roughly a six-hour run.
# Three trials gives 18 runs per device and targets a 3.6-4 hour wall time.
export TRIALS="${TRIALS:-3}"
export LOCAL_EPOCHS="${LOCAL_EPOCHS:-10}"
export REFERENCE_TRIALS=0

case "${1:-run}" in
  run)
    exec "$SCRIPT_DIR/main_experiment_local.zsh" cifar10-cnn
    ;;
  collect)
    exec "$SCRIPT_DIR/main_experiment_local.zsh" cifar10-cnn-collect
    ;;
  *)
    print -u2 "Usage: ./main_experiment_cifar10_cnn.zsh [run|collect]"
    exit 2
    ;;
esac
