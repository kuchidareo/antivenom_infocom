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
OPERATORS="${OPERATORS:-relu,maxpool,conv}"
REGIMES="${REGIMES:-low,mid,high}"
TEMPORALS="${TEMPORALS:-stable,changing}"
SEEDS="${SEEDS:-42}"
TRIAL_ID="${TRIAL_ID:-trial_0}"
DEVICE_ID="${DEVICE_ID:-$(hostname -s)}"
REPEATS_OVERRIDE="${REPEATS:-}"
WARMUP_OVERRIDE="${WARMUP:-}"
RELU_MAXPOOL_REPEATS="${RELU_MAXPOOL_REPEATS:-${REPEATS_OVERRIDE:-50000}}"
RELU_MAXPOOL_WARMUP="${RELU_MAXPOOL_WARMUP:-${WARMUP_OVERRIDE:-1000}}"
CONV_REPEATS="${CONV_REPEATS:-${REPEATS_OVERRIDE:-1000}}"
CONV_WARMUP="${CONV_WARMUP:-${WARMUP_OVERRIDE:-100}}"
THREADS="${THREADS:-1}"
START_DELAY="${START_DELAY:-0}"
BATCH_SIZE="${BATCH_SIZE:-16}"
CHANNELS="${CHANNELS:-64}"
HEIGHT="${HEIGHT:-32}"
WIDTH="${WIDTH:-32}"
ACTIVATION_RATE="${ACTIVATION_RATE:-0.5}"
POOL_SIZE="${POOL_SIZE:-2}"
POOL_STRIDE="${POOL_STRIDE:-2}"
CONV_OUT_CHANNELS="${CONV_OUT_CHANNELS:-64}"
CONV_KERNEL_SIZE="${CONV_KERNEL_SIZE:-3}"
CONV_STRIDE="${CONV_STRIDE:-1}"
CONV_PADDING="${CONV_PADDING:-1}"
BANK_SIZE="${BANK_SIZE:-16}"
PERF_PROFILE="${PERF_PROFILE:-auto}"
PERF_EVENTS="${PERF_EVENTS:-}"
PERF_ENABLED="${PERF_ENABLED:-1}"

mkdir -p "$OUTPUT_DIR"

typeset -a common_args
common_args=(
  --trial-id "$TRIAL_ID"
  --device-id "$DEVICE_ID"
  --threads "$THREADS"
  --start-delay "$START_DELAY"
  --output-dir "$OUTPUT_DIR"
  --perf-profile "$PERF_PROFILE"
  --batch-size "$BATCH_SIZE"
  --channels "$CHANNELS"
  --height "$HEIGHT"
  --width "$WIDTH"
  --activation-rate "$ACTIVATION_RATE"
  --pool-size "$POOL_SIZE"
  --pool-stride "$POOL_STRIDE"
  --conv-out-channels "$CONV_OUT_CHANNELS"
  --conv-kernel-size "$CONV_KERNEL_SIZE"
  --conv-stride "$CONV_STRIDE"
  --conv-padding "$CONV_PADDING"
  --bank-size "$BANK_SIZE"
)
if [[ -n "$REPEATS_OVERRIDE" ]]; then
  common_args+=(--repeats "$REPEATS_OVERRIDE")
fi
if [[ -n "$WARMUP_OVERRIDE" ]]; then
  common_args+=(--warmup "$WARMUP_OVERRIDE")
fi
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
  typeset operator_repeats operator_warmup
  if [[ "$operator" == "conv" ]]; then
    operator_repeats="$CONV_REPEATS"
    operator_warmup="$CONV_WARMUP"
  else
    operator_repeats="$RELU_MAXPOOL_REPEATS"
    operator_warmup="$RELU_MAXPOOL_WARMUP"
  fi
  for regime in "${(@f)$(split_csv "$REGIMES")}"; do
    for temporal in "${(@f)$(split_csv "$TEMPORALS")}"; do
      for seed in "${(@f)$(split_csv "$SEEDS")}"; do
        print "Running operator=${operator} regime=${regime} temporal=${temporal} seed=${seed}"
        "$PYTHON" "$PROJECT_DIR/controlled_ex.py" \
          "${common_args[@]}" \
          --operator "$operator" \
          --regime "$regime" \
          --temporal "$temporal" \
          --seed "$seed" \
          --repeats "$operator_repeats" \
          --warmup "$operator_warmup"
      done
    done
  done
done

print "Controlled entropy experiment complete: $OUTPUT_DIR"
