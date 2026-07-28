#!/usr/bin/env zsh
set -euo pipefail

SCRIPT_DIR="${0:A:h}"
exec "$SCRIPT_DIR/main_experiment_local.zsh" "$@"
