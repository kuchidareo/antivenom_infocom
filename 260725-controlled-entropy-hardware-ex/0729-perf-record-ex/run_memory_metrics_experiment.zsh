#!/usr/bin/env zsh
set -euo pipefail

PROJECT_DIR="${0:A:h}"
PARENT_DIR="${PROJECT_DIR:h}"
REPO_DIR="${PARENT_DIR:h}"
PYTHON="${PYTHON:-${REPO_DIR}/venv/bin/python}"
PERF="${PERF:-perf}"
TASKSET="${TASKSET:-taskset}"
CONTROLLED_EX="${CONTROLLED_EX:-${PROJECT_DIR}/controlled_ex.py}"

[[ -x "$PYTHON" ]] || { print -u2 "Python is not executable: $PYTHON"; exit 1; }
command -v "$PERF" >/dev/null || { print -u2 "perf is unavailable: $PERF"; exit 1; }
command -v "$TASKSET" >/dev/null || { print -u2 "taskset is unavailable: $TASKSET"; exit 1; }
[[ -f "$CONTROLLED_EX" ]] || { print -u2 "Missing controlled experiment: $CONTROLLED_EX"; exit 1; }

RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d%H%M%S)}"
EXPERIMENT_MODE="${EXPERIMENT_MODE:-presets}"
if [[ "$EXPERIMENT_MODE" == "basic" ]]; then
  DEFAULT_OUTPUT_BASE="basic_event_results"
  SOURCE_GROUP_LABEL="BasicEvents"
else
  DEFAULT_OUTPUT_BASE="memory_metric_results"
  SOURCE_GROUP_LABEL="AllPresets"
fi
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_DIR}/${DEFAULT_OUTPUT_BASE}/${RUN_STAMP}}"
DEVICE_ID="${DEVICE_ID:-$(hostname -s)}"
BASE_SEED="${BASE_SEED:-42}"
SYSTEM_CPU="${SYSTEM_CPU:-2}"

CHAINS="${CHAINS:-conv_only,conv_relu_pool}"
REGIMES="${REGIMES:-low,high}"
SCOPES="${SCOPES:-user,kernel}"
MAIN_TRIALS="${MAIN_TRIALS:-5}"
COOLDOWN_SECONDS="${COOLDOWN_SECONDS:-1}"
REQUIRE_NMI_WATCHDOG_OFF="${REQUIRE_NMI_WATCHDOG_OFF:-1}"

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
INPUT_BANK_SIZE="${INPUT_BANK_SIZE:-250}"
WARMUP_BANK_SIZE="${WARMUP_BANK_SIZE:-16}"
EVENTS_PER_PASS="${EVENTS_PER_PASS:-1}"
SYSTEM_EVENTS_PER_PASS="${SYSTEM_EVENTS_PER_PASS:-2}"

[[ "$INPUT_BANK_SIZE" == <-> ]] && (( INPUT_BANK_SIZE > 0 )) || {
  print -u2 "INPUT_BANK_SIZE must be a positive integer"
  exit 2
}
[[ "$WARMUP_BANK_SIZE" == <-> ]] && (( WARMUP_BANK_SIZE > 0 )) || {
  print -u2 "WARMUP_BANK_SIZE must be a positive integer"
  exit 2
}
CONV_WARMUP="${CONV_WARMUP:-200}"
CONV_REPEATS="${CONV_REPEATS:-250}"
RELU_WARMUP="${RELU_WARMUP:-1000}"
RELU_REPEATS="${RELU_REPEATS:-250}"
MAXPOOL_WARMUP="${MAXPOOL_WARMUP:-1000}"
MAXPOOL_REPEATS="${MAXPOOL_REPEATS:-250}"

# Warm-up calibration is deliberately lightweight and does not repeat every
# memory metric pass. It checks steady-state runtime and IPC before main runs.
RUN_WARMUP_CALIBRATION="${RUN_WARMUP_CALIBRATION:-1}"
CALIBRATION_TRIALS="${CALIBRATION_TRIALS:-2}"
CONV_WARMUP_LEVELS="${CONV_WARMUP_LEVELS:-0,1,5,10,25,50,100,200}"
CONV_CALIBRATION_REPEATS="${CONV_CALIBRATION_REPEATS:-100}"
RELU_WARMUP_LEVELS="${RELU_WARMUP_LEVELS:-0,10,100,500,1000,2000}"
RELU_CALIBRATION_REPEATS="${RELU_CALIBRATION_REPEATS:-100}"
MAXPOOL_WARMUP_LEVELS="${MAXPOOL_WARMUP_LEVELS:-0,10,100,500,1000,2000}"
MAXPOOL_CALIBRATION_REPEATS="${MAXPOOL_CALIBRATION_REPEATS:-100}"

split_csv() {
  local value="$1"
  print -l -- ${(s:,:)value}
}

operator_for_chain() {
  case "$1" in
    relu_only) print -- relu ;;
    maxpool_only) print -- maxpool ;;
    conv_only|conv_relu_pool|conv_relu_pool_autograd) print -- conv ;;
    *) print -u2 "Unknown chain: $1"; return 2 ;;
  esac
}

chain_settings() {
  case "$1" in
    relu_only)
      print -- "$RELU_WARMUP|$RELU_REPEATS|$RELU_WARMUP_LEVELS|$RELU_CALIBRATION_REPEATS"
      ;;
    maxpool_only)
      print -- "$MAXPOOL_WARMUP|$MAXPOOL_REPEATS|$MAXPOOL_WARMUP_LEVELS|$MAXPOOL_CALIBRATION_REPEATS"
      ;;
    conv_only|conv_relu_pool|conv_relu_pool_autograd)
      print -- "$CONV_WARMUP|$CONV_REPEATS|$CONV_WARMUP_LEVELS|$CONV_CALIBRATION_REPEATS"
      ;;
    *) print -u2 "Unknown chain: $1"; return 2 ;;
  esac
}

scope_option() {
  case "$1" in
    user) print -- --all-user ;;
    kernel) print -- --all-kernel ;;
    system) print -- --all-cpus ;;
    *) print -u2 "Unknown scope: $1"; return 2 ;;
  esac
}

event_for_perf() {
  local event="$1"
  local body
  case "$event" in
    cpu@*@)
      body="${event#cpu@}"
      body="${body%@}"
      print -r -- "cpu/${body}/"
      ;;
    arb@*@)
      body="${event#arb@}"
      body="${body%@}"
      print -r -- "arb/${body}/"
      ;;
    *@*@)
      local base="${event%%@*}"
      body="${event#*@}"
      body="${body%@}"
      print -r -- "${base}/${body}/"
      ;;
    *)
      print -r -- "$event"
      ;;
  esac
}

if (( REQUIRE_NMI_WATCHDOG_OFF )) && [[ -r /proc/sys/kernel/nmi_watchdog ]]; then
  if [[ "$(</proc/sys/kernel/nmi_watchdog)" != "0" ]]; then
    print -u2 "kernel.nmi_watchdog is enabled and consumes a PMU counter."
    print -u2 "Disable it before this experiment:"
    print -u2 "  sudo sysctl kernel.nmi_watchdog=0"
    print -u2 "Restore it afterward with:"
    print -u2 "  sudo sysctl kernel.nmi_watchdog=1"
    exit 2
  fi
fi

mkdir -p "$OUTPUT_ROOT"
"$PYTHON" "$PROJECT_DIR/build_memory_metric_plan.py" \
  --perf "$PERF" \
  --output-dir "$OUTPUT_ROOT" \
  --system-cpu "$SYSTEM_CPU" \
  --events-per-pass "$EVENTS_PER_PASS" \
  --system-events-per-pass "$SYSTEM_EVENTS_PER_PASS" \
  --mode "$EXPERIMENT_MODE"

typeset -a PASS_DEFINITIONS
PASS_DEFINITIONS=()
while IFS=$'\t' read -r scope_class pass_name event_count events; do
  [[ -n "$scope_class" ]] || continue
  PASS_DEFINITIONS+=("${scope_class}|${pass_name}|${event_count}|${events}")
done < "$OUTPUT_ROOT/pass_plan.tsv"
(( ${#PASS_DEFINITIONS[@]} > 0 )) || {
  print -u2 "Generated perf pass plan is empty"
  exit 2
}

MANIFEST="${OUTPUT_ROOT}/manifest.csv"
print 'run_kind,pair_id,sequence_position,operator,chain,regime,temporal,conv_autograd_enabled,relu_autograd_enabled,maxpool_autograd_enabled,conv_output_shape,relu_output_shape,pool_output_shape,conv_output_requires_grad,relu_output_requires_grad,pool_output_requires_grad,trial_id,seed,device_id,scope,source_group,pass_name,pass_mode,warmup,repeats,bank_size,warmup_bank_size,bank_working_set_bytes,spec,run_id,json_path,perf_jsonl,runner_log' > "$MANIFEST"

run_condition() {
  local run_kind="$1"
  local pair_id="$2"
  local sequence_position="$3"
  local chain="$4"
  local operator
  operator="$(operator_for_chain "$chain")"
  local regime="$5"
  local trial_index="$6"
  local scope="$7"
  local source_group="$8"
  local pass_name="$9"
  local pass_mode="${10}"
  local spec="${11}"
  local warmup="${12}"
  local repeats="${13}"
  local seed=$(( BASE_SEED + trial_index ))
  local trial_id="trial_${trial_index}"
  local bank_tensor_bytes=$(( BATCH_SIZE * CHANNELS * HEIGHT * WIDTH * 4 ))
  local bank_working_set_bytes=$(( INPUT_BANK_SIZE * bank_tensor_bytes ))
  local conv_height=$(( (HEIGHT + 2 * CONV_PADDING - CONV_KERNEL_SIZE) / CONV_STRIDE + 1 ))
  local conv_width=$(( (WIDTH + 2 * CONV_PADDING - CONV_KERNEL_SIZE) / CONV_STRIDE + 1 ))
  local conv_pool_height=$(( (conv_height - POOL_SIZE) / POOL_STRIDE + 1 ))
  local conv_pool_width=$(( (conv_width - POOL_SIZE) / POOL_STRIDE + 1 ))
  local direct_pool_height=$(( (HEIGHT - POOL_SIZE) / POOL_STRIDE + 1 ))
  local direct_pool_width=$(( (WIDTH - POOL_SIZE) / POOL_STRIDE + 1 ))
  local conv_shape=""
  local relu_shape=""
  local pool_shape=""
  local conv_requires_grad=0
  local relu_requires_grad=0
  local pool_requires_grad=0
  if [[ "$operator" == "conv" ]]; then
    conv_shape="${BATCH_SIZE}x${CONV_OUT_CHANNELS}x${conv_height}x${conv_width}"
    relu_shape="$conv_shape"
    pool_shape="${BATCH_SIZE}x${CONV_OUT_CHANNELS}x${conv_pool_height}x${conv_pool_width}"
    conv_requires_grad=1
  elif [[ "$operator" == "relu" ]]; then
    relu_shape="${BATCH_SIZE}x${CHANNELS}x${HEIGHT}x${WIDTH}"
    relu_requires_grad=1
  else
    pool_shape="${BATCH_SIZE}x${CHANNELS}x${direct_pool_height}x${direct_pool_width}"
    pool_requires_grad=1
  fi
  local relu_autograd_enabled=0
  local maxpool_autograd_enabled=0
  if [[ "$chain" == "conv_relu_pool_autograd" ]]; then
    relu_autograd_enabled=1
    maxpool_autograd_enabled=1
    relu_requires_grad=1
    pool_requires_grad=1
  elif [[ "$chain" == "relu_only" ]]; then
    relu_autograd_enabled=1
  elif [[ "$chain" == "maxpool_only" ]]; then
    maxpool_autograd_enabled=1
  fi
  if (( INPUT_BANK_SIZE < repeats )); then
    print -u2 "Conv INPUT_BANK_SIZE (${INPUT_BANK_SIZE}) must be >= repeats (${repeats})"
    return 2
  fi
  local run_id="${operator}_${chain}_${regime}_dataset_seed${seed}_${trial_id}_b${BATCH_SIZE}_c${CHANNELS}_h${HEIGHT}_w${WIDTH}"
  if [[ "$operator" == "conv" ]]; then
    run_id+="_oc${CONV_OUT_CHANNELS}_k${CONV_KERNEL_SIZE}_s${CONV_STRIDE}_p${CONV_PADDING}"
  fi

  local condition_dir="${OUTPUT_ROOT}/${run_kind}/${operator}/${chain}/${pair_id}/${scope}/${regime}/${pass_name}"
  local profile_id="${run_id}_${scope}_${pass_name}_w${warmup}_r${repeats}"
  local json_path="${condition_dir}/${run_id}.json"
  local perf_jsonl="${condition_dir}/${profile_id}_perf_stat.jsonl"
  local runner_log="${condition_dir}/${profile_id}.log"
  local control_fifo="${condition_dir}/.${profile_id}.control.fifo"
  local ack_fifo="${condition_dir}/.${profile_id}.ack.fifo"
  mkdir -p "$condition_dir"
  mkfifo "$control_fifo" "$ack_fifo"

  typeset -a perf_args command_prefix requested_events
  perf_args=(stat --delay=-1 --control "fifo:${control_fifo},${ack_fifo}")
  command_prefix=()
  case "$pass_mode" in
    raw_thread)
      perf_args+=("$(scope_option "$scope")" --no-scale)
      requested_events=("${(@s:|:)spec}")
      for requested_event in "${requested_events[@]}"; do
        perf_args+=(--event "$(event_for_perf "$requested_event")")
      done
      ;;
    raw_system)
      [[ "$scope" == "system" ]] || { print -u2 "System metric requires system scope"; return 2; }
      perf_args+=(--all-cpus --cpu "$SYSTEM_CPU" --no-scale)
      requested_events=("${(@s:|:)spec}")
      for requested_event in "${requested_events[@]}"; do
        perf_args+=(--event "$(event_for_perf "$requested_event")")
      done
      command_prefix=("$TASKSET" -c "$SYSTEM_CPU")
      ;;
    *)
      print -u2 "Unknown pass mode: $pass_mode"
      return 2
      ;;
  esac
  perf_args+=(--json-output --output "$perf_jsonl")

  print "==> ${operator} ${chain} ${regime} ${scope} ${source_group}/${pass_name} trial=${trial_id} warmup=${warmup} repeats=${repeats}"
  local perf_exit=0
  "$PERF" "${perf_args[@]}" -- "${command_prefix[@]}" \
    env PYTHONPATH="${PARENT_DIR}${PYTHONPATH:+:${PYTHONPATH}}" \
    "$PYTHON" "$CONTROLLED_EX" \
      --operator "$operator" \
      --chain "$chain" \
      --regime "$regime" \
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
      --bank-size "$INPUT_BANK_SIZE" \
      --warmup-bank-size "$WARMUP_BANK_SIZE" \
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
    print -u2 "perf stat failed with exit ${perf_exit}: $runner_log"
    return "$perf_exit"
  fi
  [[ -s "$perf_jsonl" ]] || { print -u2 "Empty perf output: $perf_jsonl"; return 1; }
  [[ -s "$json_path" ]] || { print -u2 "Missing controlled JSON: $json_path"; return 1; }

  print -r -- "${run_kind},${pair_id},${sequence_position},${operator},${chain},${regime},dataset_bank,${conv_requires_grad},${relu_autograd_enabled},${maxpool_autograd_enabled},${conv_shape},${relu_shape},${pool_shape},${conv_requires_grad},${relu_requires_grad},${pool_requires_grad},${trial_id},${seed},${DEVICE_ID},${scope},${source_group},${pass_name},${pass_mode},${warmup},${repeats},${INPUT_BANK_SIZE},${WARMUP_BANK_SIZE},${bank_working_set_bytes},${pass_name},${run_id},${json_path},${perf_jsonl},${runner_log}" >> "$MANIFEST"
  if [[ "$COOLDOWN_SECONDS" != "0" ]]; then
    sleep "$COOLDOWN_SECONDS"
  fi
}

typeset -a chains regimes scopes ordered_chains ordered_regimes ordered_scopes ordered_passes
chains=("${(@f)$(split_csv "$CHAINS")}")
regimes=("${(@f)$(split_csv "$REGIMES")}")
scopes=("${(@f)$(split_csv "$SCOPES")}")
for chain in "${chains[@]}"; do
  [[ "$chain" == (relu_only|maxpool_only|conv_only|conv_relu_pool|conv_relu_pool_autograd) ]] || {
    print -u2 "Unsupported chain: $chain"
    exit 2
  }
done
for regime in "${regimes[@]}"; do
  [[ "$regime" == (low|high) ]] || { print -u2 "Unsupported regime: $regime"; exit 2; }
done
for scope in "${scopes[@]}"; do scope_option "$scope" >/dev/null; done

if (( RUN_WARMUP_CALIBRATION )); then
  print "Running warm-up calibration for Conv chains: ${CHAINS}"
  for chain in "${chains[@]}"; do
    IFS='|' read -r main_warmup main_repeats warmup_csv calibration_repeats \
      <<< "$(chain_settings "$chain")"
    typeset -a warmup_levels ordered_levels
    warmup_levels=("${(@f)$(split_csv "$warmup_csv")}")
    for (( calibration_trial = 0; calibration_trial < CALIBRATION_TRIALS; calibration_trial++ )); do
      if (( calibration_trial % 2 == 0 )); then
        ordered_levels=("${warmup_levels[@]}")
      else
        ordered_levels=("${(@Oa)warmup_levels}")
      fi
      position=0
      for warmup in "${ordered_levels[@]}"; do
        if (( (calibration_trial + position) % 2 == 0 )); then
          ordered_regimes=(low high)
        else
          ordered_regimes=(high low)
        fi
        for regime in "${ordered_regimes[@]}"; do
          pair_id="warmup_${warmup}_rep_${calibration_trial}"
          run_condition warmup_calibration "$pair_id" "$position" "$chain" \
            "$regime" "$calibration_trial" user Warmup calibration raw_thread \
            'cycles|instructions' "$warmup" "$calibration_repeats"
        done
        (( position += 1 ))
      done
    done
  done
fi

print "Running grouped memory analysis for chains: ${CHAINS}"
for (( trial_index = 0; trial_index < MAIN_TRIALS; trial_index++ )); do
  if (( trial_index % 2 == 0 )); then
    ordered_chains=("${chains[@]}")
    ordered_regimes=(low high)
    ordered_scopes=(user kernel)
    ordered_passes=("${PASS_DEFINITIONS[@]}")
  else
    ordered_chains=("${(@Oa)chains}")
    ordered_regimes=(high low)
    ordered_scopes=(kernel user)
    ordered_passes=("${(@Oa)PASS_DEFINITIONS}")
  fi
  for chain in "${ordered_chains[@]}"; do
    IFS='|' read -r main_warmup main_repeats _ _ <<< "$(chain_settings "$chain")"
    for regime in "${ordered_regimes[@]}"; do
      position=0
      for definition in "${ordered_passes[@]}"; do
        IFS='|' read -r scope_class pass_name event_count spec <<< "$definition"
        source_group="$SOURCE_GROUP_LABEL"
        if [[ "$scope_class" == "system" ]]; then
          run_condition main "pair_${trial_index}" "$position" "$chain" \
            "$regime" "$trial_index" system "$source_group" "$pass_name" \
            raw_system "$spec" "$main_warmup" "$main_repeats"
        else
          for scope in "${ordered_scopes[@]}"; do
            run_condition main "pair_${trial_index}" "$position" "$chain" \
              "$regime" "$trial_index" "$scope" "$source_group" "$pass_name" \
              raw_thread "$spec" "$main_warmup" "$main_repeats"
          done
        fi
        (( position += 1 ))
      done
    done
  done
done

print "Memory metric experiment complete: $OUTPUT_ROOT"
print "Analyze with:"
print "  $PYTHON $PROJECT_DIR/memory_metrics_analysis.py --input-dir $OUTPUT_ROOT"
