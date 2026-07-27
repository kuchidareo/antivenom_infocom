#!/usr/bin/env zsh
set -euo pipefail

PROJECT_DIR="${0:A:h}"
REPO_DIR="${PROJECT_DIR:h}"
PYTHON="${PYTHON:-${REPO_DIR}/venv/bin/python}"

if [[ ! -x "$PYTHON" ]]; then
  print -u2 "Python is not executable: $PYTHON"
  print -u2 "Set PYTHON=/path/to/python and run again."
  exit 1
fi

OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_DIR}/controlled_vit_softmax_results}"
REGIMES="${REGIMES:-low,mid,high}"
SEEDS="${SEEDS:-42}"
TRIAL_ID="${TRIAL_ID:-trial_0}"
DEVICE_ID="${DEVICE_ID:-$(hostname -s)}"
BATCH_SIZE="${BATCH_SIZE:-16}"
GRID_SIZE="${GRID_SIZE:-8}"
HEADS="${HEADS:-4}"
MID_PROTOTYPES="${MID_PROTOTYPES:-8}"
REPEATS="${REPEATS:-10000}"
WARMUP="${WARMUP:-500}"
THREADS="${THREADS:-1}"
START_DELAY="${START_DELAY:-0}"
PERF_PROFILE="${PERF_PROFILE:-auto}"
PERF_EVENTS="${PERF_EVENTS:-}"
PERF_ENABLED="${PERF_ENABLED:-1}"

mkdir -p "$OUTPUT_DIR"

typeset -a common_args
common_args=(
  --trial-id "$TRIAL_ID"
  --device-id "$DEVICE_ID"
  --batch-size "$BATCH_SIZE"
  --grid-size "$GRID_SIZE"
  --heads "$HEADS"
  --mid-prototypes "$MID_PROTOTYPES"
  --repeats "$REPEATS"
  --warmup "$WARMUP"
  --threads "$THREADS"
  --start-delay "$START_DELAY"
  --output-dir "$OUTPUT_DIR"
  --perf-profile "$PERF_PROFILE"
)
if [[ -n "$PERF_EVENTS" ]]; then
  common_args+=(--perf-events "$PERF_EVENTS")
fi
if [[ "$PERF_ENABLED" != "1" ]]; then
  common_args+=(--disable-perf)
fi

# Arguments select one condition directly. With no arguments, run every
# score-row entropy regime and seed sequentially in separate processes.
if (( $# > 0 )); then
  exec "$PYTHON" "$PROJECT_DIR/controlled_ex_ViT.py" "${common_args[@]}" "$@"
fi

split_csv() {
  local value="$1"
  print -l -- ${(s:,:)value}
}

typeset regime seed
for regime in "${(@f)$(split_csv "$REGIMES")}"; do
  for seed in "${(@f)$(split_csv "$SEEDS")}"; do
    print "Running operator=softmax regime=${regime} seed=${seed}"
    "$PYTHON" "$PROJECT_DIR/controlled_ex_ViT.py" \
      "${common_args[@]}" \
      --regime "$regime" \
      --seed "$seed"
    done
done

print "Controlled ViT Softmax experiment complete: $OUTPUT_DIR"
