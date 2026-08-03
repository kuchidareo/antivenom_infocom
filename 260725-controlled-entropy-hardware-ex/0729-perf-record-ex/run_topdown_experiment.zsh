#!/usr/bin/env zsh
set -euo pipefail

PROJECT_DIR="${0:A:h}"
PARENT_DIR="${PROJECT_DIR:h}"
REPO_DIR="${PARENT_DIR:h}"
PYTHON="${PYTHON:-${REPO_DIR}/venv/bin/python}"
PERF="${PERF:-perf}"
CONTROLLED_EX="${CONTROLLED_EX:-${PROJECT_DIR}/controlled_ex.py}"

[[ -x "$PYTHON" ]] || { print -u2 "Python is not executable: $PYTHON"; exit 1; }
command -v "$PERF" >/dev/null || { print -u2 "perf is unavailable: $PERF"; exit 1; }
[[ -f "$CONTROLLED_EX" ]] || { print -u2 "Missing controlled experiment: $CONTROLLED_EX"; exit 1; }
if ! "$PERF" list TopdownL1 2>&1 | grep -q 'tma_backend_bound'; then
  print -u2 "The current CPU/perf installation does not provide TopdownL1."
  print -u2 "TopdownL1 is generally an Intel-specific PMU metric group and is not portable to Raspberry Pi/Jetson."
  exit 2
fi
if ! "$PERF" list tma_bad_speculation 2>&1 | grep -q 'tma_bad_speculation'; then
  print -u2 "The current CPU/perf installation does not provide tma_bad_speculation."
  exit 2
fi

RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d%H%M%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_DIR}/topdown_results/${RUN_STAMP}}"
OPERATORS="${OPERATORS:-relu,maxpool,conv}"
REGIMES="${REGIMES:-low,high}"
SCOPES="${SCOPES:-user,kernel}"
TRIALS="${TRIALS:-1}"
BASE_SEED="${BASE_SEED:-42}"
DEVICE_ID="${DEVICE_ID:-$(hostname -s)}"

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
THREADS="${THREADS:-1}"

MAXPOOL_WARMUP="${MAXPOOL_WARMUP:-1000}"
MAXPOOL_REPEATS="${MAXPOOL_REPEATS:-50000}"
RELU_WARMUP="${RELU_WARMUP:-1000}"
RELU_REPEATS="${RELU_REPEATS:-50000}"
CONV_WARMUP="${CONV_WARMUP:-100}"
CONV_REPEATS="${CONV_REPEATS:-1000}"
BAD_SPEC_OPERATORS="${BAD_SPEC_OPERATORS:-relu,maxpool,conv}"
BAD_SPEC_METRICS="${BAD_SPEC_METRICS:-tma_bad_speculation,tma_branch_mispredicts,tma_machine_clears,tma_mispredicts_resteers,tma_clears_resteers}"

mkdir -p "$OUTPUT_ROOT"
MANIFEST="${OUTPUT_ROOT}/manifest.csv"
print 'operator,regime,temporal,trial_id,seed,device_id,scope,metric_pass,metric_spec,run_id,json_path,topdown_jsonl,runner_log' > "$MANIFEST"

split_csv() {
  local value="$1"
  print -l -- ${(s:,:)value}
}

csv_contains() {
  local csv="$1"
  local needle="$2"
  local item
  for item in "${(@f)$(split_csv "$csv")}"; do
    [[ "$item" == "$needle" ]] && return 0
  done
  return 1
}

scope_option() {
  case "$1" in
    user) print -- --all-user ;;
    kernel) print -- --all-kernel ;;
    *) print -u2 "Unknown scope: $1"; return 2 ;;
  esac
}

run_condition() {
  local operator="$1"
  local regime="$2"
  local trial_index="$3"
  local scope="$4"
  local metric_pass="$5"
  local metric_spec="$6"
  local scope_flag
  scope_flag="$(scope_option "$scope")"
  local seed=$(( BASE_SEED + trial_index ))
  local trial_id="trial_${trial_index}"
  local warmup repeats run_id
  if [[ "$operator" == "conv" ]]; then
    warmup="$CONV_WARMUP"
    repeats="$CONV_REPEATS"
  elif [[ "$operator" == "relu" ]]; then
    warmup="$RELU_WARMUP"
    repeats="$RELU_REPEATS"
  else
    warmup="$MAXPOOL_WARMUP"
    repeats="$MAXPOOL_REPEATS"
  fi

  run_id="${operator}_${regime}_stable_seed${seed}_${trial_id}_b${BATCH_SIZE}_c${CHANNELS}_h${HEIGHT}_w${WIDTH}"
  if [[ "$operator" == "conv" ]]; then
    run_id+="_oc${CONV_OUT_CHANNELS}_k${CONV_KERNEL_SIZE}_s${CONV_STRIDE}_p${CONV_PADDING}"
  fi

  local condition_dir="${OUTPUT_ROOT}/${operator}/${regime}/${trial_id}/${scope}/${metric_pass}"
  local profile_id="${run_id}_${scope}_${metric_pass}"
  local json_path="${condition_dir}/${run_id}.json"
  local topdown_jsonl="${condition_dir}/${profile_id}.jsonl"
  local runner_log="${condition_dir}/${profile_id}.log"
  local control_fifo="${condition_dir}/.${profile_id}.control.fifo"
  local ack_fifo="${condition_dir}/.${profile_id}.ack.fifo"
  mkdir -p "$condition_dir"
  mkfifo "$control_fifo" "$ack_fifo"

  print "==> ${metric_pass} operator=${operator} regime=${regime} scope=${scope} trial=${trial_id} seed=${seed} metrics=${metric_spec}"
  local perf_exit=0
  "$PERF" stat \
    --delay=-1 \
    --control "fifo:${control_fifo},${ack_fifo}" \
    "$scope_flag" \
    --metrics "$metric_spec" \
    --metric-no-threshold \
    --json-output \
    --output "$topdown_jsonl" \
    -- env PYTHONPATH="${PARENT_DIR}${PYTHONPATH:+:${PYTHONPATH}}" \
      "$PYTHON" "$CONTROLLED_EX" \
      --operator "$operator" \
      --regime "$regime" \
      --temporal stable \
      --seed "$seed" \
      --trial-id "$trial_id" \
      --device-id "$DEVICE_ID" \
      --batch-size "$BATCH_SIZE" \
      --channels "$CHANNELS" \
      --height "$HEIGHT" \
      --width "$WIDTH" \
      --activation-rate "$ACTIVATION_RATE" \
      --pool-size "$POOL_SIZE" \
      --pool-stride "$POOL_STRIDE" \
      --conv-out-channels "$CONV_OUT_CHANNELS" \
      --conv-kernel-size "$CONV_KERNEL_SIZE" \
      --conv-stride "$CONV_STRIDE" \
      --conv-padding "$CONV_PADDING" \
      --warmup "$warmup" \
      --repeats "$repeats" \
      --threads "$THREADS" \
      --start-delay 0 \
      --output-dir "$condition_dir" \
      --disable-perf \
      --perf-control-fifo "$control_fifo" \
      --perf-control-ack-fifo "$ack_fifo" \
      >"$runner_log" 2>&1 || perf_exit=$?
  rm -f "$control_fifo" "$ack_fifo"
  cat "$runner_log"
  if (( perf_exit != 0 )); then
    print -u2 "${metric_pass} perf stat failed with exit ${perf_exit}: $runner_log"
    return "$perf_exit"
  fi
  [[ -s "$topdown_jsonl" ]] || { print -u2 "Empty ${metric_pass} output: $topdown_jsonl"; return 1; }
  [[ -s "$json_path" ]] || { print -u2 "Missing controlled JSON: $json_path"; return 1; }
  local manifest_metric_spec="${metric_spec//,/;}"
  print -r -- "${operator},${regime},stable,${trial_id},${seed},${DEVICE_ID},${scope},${metric_pass},${manifest_metric_spec},${run_id},${json_path},${topdown_jsonl},${runner_log}" >> "$MANIFEST"
}

typeset operator regime scope trial_index
for operator in "${(@f)$(split_csv "$OPERATORS")}"; do
  [[ "$operator" == (relu|maxpool|conv) ]] || { print -u2 "Unsupported operator: $operator"; exit 2; }
  for regime in "${(@f)$(split_csv "$REGIMES")}"; do
    [[ "$regime" == (low|high) ]] || { print -u2 "Unsupported regime: $regime"; exit 2; }
    for (( trial_index = 0; trial_index < TRIALS; trial_index++ )); do
      for scope in "${(@f)$(split_csv "$SCOPES")}"; do
        scope_option "$scope" >/dev/null
        run_condition "$operator" "$regime" "$trial_index" "$scope" \
          topdown_l1 TopdownL1
        if csv_contains "$BAD_SPEC_OPERATORS" "$operator"; then
          run_condition "$operator" "$regime" "$trial_index" "$scope" \
            bad_speculation "$BAD_SPEC_METRICS"
        fi
      done
    done
  done
done

print "Topdown experiment complete: $OUTPUT_ROOT"
print "Analyze with:"
print "  $PYTHON $PROJECT_DIR/topdown_analysis.py --input-dir $OUTPUT_ROOT"
