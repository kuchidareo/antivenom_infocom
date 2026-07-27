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

OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_DIR}/controlled_entropy_results}"
OPERATORS="${OPERATORS:-relu,maxpool}"
REGIMES="${REGIMES:-low,mid,high}"
TEMPORALS="${TEMPORALS:-stable,changing}"
SEEDS="${SEEDS:-42}"
TRIAL_ID="${TRIAL_ID:-trial_0}"
DEVICE_ID="${DEVICE_ID:-$(hostname -s)}"
REPEATS="${REPEATS:-50000}"
WARMUP="${WARMUP:-1000}"
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

# Arguments select one condition directly. With no arguments, run the full
# operator x entropy regime x temporal-behavior matrix sequentially.
if (( $# > 0 )); then
  exec "$PYTHON" "$PROJECT_DIR/controlled_ex.py" "${common_args[@]}" "$@"
fi

split_csv() {
  local value="$1"
  print -l -- ${(s:,:)value}
}

typeset operator regime temporal seed
for operator in "${(@f)$(split_csv "$OPERATORS")}"; do
  for regime in "${(@f)$(split_csv "$REGIMES")}"; do
    for temporal in "${(@f)$(split_csv "$TEMPORALS")}"; do
      for seed in "${(@f)$(split_csv "$SEEDS")}"; do
        print "Running operator=${operator} regime=${regime} temporal=${temporal} seed=${seed}"
        "$PYTHON" "$PROJECT_DIR/controlled_ex.py" \
          "${common_args[@]}" \
          --operator "$operator" \
          --regime "$regime" \
          --temporal "$temporal" \
          --seed "$seed"
      done
    done
  done
done

print "Controlled entropy experiment complete: $OUTPUT_DIR"
