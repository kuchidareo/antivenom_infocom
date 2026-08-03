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

RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d%H%M%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_DIR}/perf_record_results/${RUN_STAMP}}"
OPERATORS="${OPERATORS:-maxpool,conv}"
REGIMES="${REGIMES:-low,high}"
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
CONV_WARMUP="${CONV_WARMUP:-100}"
CONV_REPEATS="${CONV_REPEATS:-1000}"

SCOPES="${SCOPES:-user,kernel}"
EVENT_PASSES="${EVENT_PASSES:-}"
PERF_FREQ="${PERF_FREQ:-199}"
PERF_CLOCKID="${PERF_CLOCKID:-1}"
CALL_GRAPH="${CALL_GRAPH:-dwarf,8192}"
REPORT_PERCENT_LIMIT="${REPORT_PERCENT_LIMIT:-0.1}"
ENABLE_JIT_SYMBOLS="${ENABLE_JIT_SYMBOLS:-1}"
ONEDNN_JIT_PROFILE_MODE="${ONEDNN_JIT_PROFILE_MODE:-6}"

if [[ -z "$EVENT_PASSES" ]]; then
  if [[ -f /etc/nv_tegra_release || -d /sys/devices/platform/tegra-soc ]]; then
    EVENT_PASSES='cycles,instructions,l1d_cache,l1d_cache_refill,l2d_cache,l2d_cache_refill'
  elif [[ "$(uname -m)" == (aarch64|arm64|armv7l) ]]; then
    EVENT_PASSES='cycles,instructions,l1d_cache_rd,l1d_cache_refill_rd,l2d_cache_rd,l2d_cache_refill_rd'
  else
    # Generic cache-references/misses normally represent LLC on x86, not L2.
    EVENT_PASSES='cycles,instructions,L1-dcache-loads,L1-dcache-load-misses,cache-references,cache-misses'
  fi
fi

mkdir -p "$OUTPUT_ROOT"
MANIFEST="${OUTPUT_ROOT}/manifest.csv"
print 'operator,regime,temporal,trial_id,seed,device_id,scope,pass_id,event,run_id,json_path,perf_data,flat_report,detailed_report' > "$MANIFEST"

split_csv() {
  local value="$1"
  print -l -- ${(s:,:)value}
}

split_passes() {
  local value="$1"
  print -l -- ${(s:;:)value}
}

scope_modifier() {
  case "$1" in
    user) print u ;;
    kernel) print k ;;
    *) print -u2 "Unknown scope: $1"; return 2 ;;
  esac
}

normalize_event_passes() {
  local pass event event_count
  local -a pass_events chunk normalized
  normalized=()
  for pass in "${(@f)$(split_passes "$EVENT_PASSES")}"; do
    pass_events=("${(@s:,:)pass}")
    event_count=${#pass_events}
    if (( event_count == 0 )); then
      print -u2 "An event pass is empty."
      return 2
    fi
    chunk=()
    for event in "${pass_events[@]}"; do
      event="${event//[[:space:]]/}"
      [[ -n "$event" ]] || { print -u2 "An event name is empty."; return 2; }
      chunk+=("$event")
      if (( ${#chunk} == 6 )); then
        normalized+=("${(j:,:)chunk}")
        chunk=()
      fi
    done
    if (( ${#chunk} > 0 )); then
      normalized+=("${(j:,:)chunk}")
    fi
  done
  EVENT_PASSES="${(j:;:)normalized}"
}

run_profile_pass() {
  local operator="$1"
  local regime="$2"
  local trial_index="$3"
  local scope="$4"
  local pass_index="$5"
  local pass_spec="$6"
  local modifier
  modifier="$(scope_modifier "$scope")"
  local -a base_events perf_event_args
  base_events=("${(@s:,:)pass_spec}")
  perf_event_args=()
  local event
  for event in "${base_events[@]}"; do
    perf_event_args+=(--event "${event}:${modifier}")
  done

  local seed=$(( BASE_SEED + trial_index ))
  local trial_id="trial_${trial_index}"
  local warmup repeats run_id

  if [[ "$operator" == "conv" ]]; then
    warmup="$CONV_WARMUP"
    repeats="$CONV_REPEATS"
  else
    warmup="$MAXPOOL_WARMUP"
    repeats="$MAXPOOL_REPEATS"
  fi

  run_id="${operator}_${regime}_stable_seed${seed}_${trial_id}_b${BATCH_SIZE}_c${CHANNELS}_h${HEIGHT}_w${WIDTH}"
  if [[ "$operator" == "conv" ]]; then
    run_id+="_oc${CONV_OUT_CHANNELS}_k${CONV_KERNEL_SIZE}_s${CONV_STRIDE}_p${CONV_PADDING}"
  fi

  local condition_dir="${OUTPUT_ROOT}/${operator}/${regime}/${trial_id}/${scope}/pass_${pass_index}"
  local profile_id="${run_id}_${scope}_pass${pass_index}"
  local worker_log="${condition_dir}/${profile_id}.stdout.log"
  local raw_perf_data="${condition_dir}/${profile_id}.raw.perf.data"
  local perf_data="${condition_dir}/${profile_id}.perf.data"
  local flat_report="${condition_dir}/${profile_id}_perf_report.csv"
  local detailed_report="${condition_dir}/${profile_id}_perf_report.txt"
  local header_report="${condition_dir}/${profile_id}_perf_header.txt"
  local json_path="${condition_dir}/${run_id}.json"
  local control_fifo="${condition_dir}/.${profile_id}.control.fifo"
  local ack_fifo="${condition_dir}/.${profile_id}.ack.fifo"
  local jitdump_dir="${condition_dir}/jitdump"
  mkdir -p "$condition_dir"
  mkdir -p "$jitdump_dir"
  mkfifo "$control_fifo" "$ack_fifo"

  print "==> operator=${operator} regime=${regime} temporal=stable trial=${trial_id} seed=${seed} scope=${scope} pass=${pass_index} events=${pass_spec}"
  local perf_exit=0
  "$PERF" record \
    --delay=-1 \
    --control "fifo:${control_fifo},${ack_fifo}" \
    "${perf_event_args[@]}" \
    --clockid "$PERF_CLOCKID" \
    --freq "$PERF_FREQ" \
    --call-graph "$CALL_GRAPH" \
    --output "$raw_perf_data" \
    -- env \
      JITDUMPDIR="$jitdump_dir" \
      ONEDNN_JIT_PROFILE="$ONEDNN_JIT_PROFILE_MODE" \
      DNNL_JIT_PROFILE="$ONEDNN_JIT_PROFILE_MODE" \
      PYTHONPATH="${PARENT_DIR}${PYTHONPATH:+:${PYTHONPATH}}" \
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
      >"$worker_log" 2>&1 || perf_exit=$?
  rm -f "$control_fifo" "$ack_fifo"
  cat "$worker_log"
  if (( perf_exit != 0 )); then
    print -u2 "perf record failed with exit ${perf_exit}: $raw_perf_data"
    return "$perf_exit"
  fi
  [[ -s "$raw_perf_data" ]] || { print -u2 "Empty perf.data: $raw_perf_data"; return 1; }

  if (( ENABLE_JIT_SYMBOLS )); then
    "$PERF" inject --jit \
      --input "$raw_perf_data" \
      --output "$perf_data"
    [[ -s "$perf_data" ]] || {
      print -u2 "perf inject produced an empty file: $perf_data"
      return 1
    }
    rm -f "$raw_perf_data"
  else
    mv -f "$raw_perf_data" "$perf_data"
  fi

  "$PERF" report \
    --input "$perf_data" \
    --stdio --stdio-color never \
    --children --show-nr-samples \
    --percent-limit "$REPORT_PERCENT_LIMIT" \
    --sort comm,dso,symbol \
    >"$detailed_report"

  "$PERF" report \
    --input "$perf_data" \
    --stdio --stdio-color never \
    --no-children --call-graph none \
    --show-nr-samples --percent-limit 0 \
    --fields overhead,sample,period,comm,dso,symbol \
    --field-separator ';' \
    >"$flat_report"

  "$PERF" report --input "$perf_data" --header-only >"$header_report"

  for event in "${base_events[@]}"; do
    print -r -- "${operator},${regime},stable,${trial_id},${seed},${DEVICE_ID},${scope},${pass_index},${event},${run_id},${json_path},${perf_data},${flat_report},${detailed_report}" >> "$MANIFEST"
  done
}

normalize_event_passes
print "Perf event passes (maximum 6 events each): $EVENT_PASSES"
print "oneDNN JIT symbol resolution: ${ENABLE_JIT_SYMBOLS} (profile mode ${ONEDNN_JIT_PROFILE_MODE})"
print "perf event clock ID: ${PERF_CLOCKID}"

typeset operator regime trial_index scope pass_spec
typeset pass_index
for operator in "${(@f)$(split_csv "$OPERATORS")}"; do
  if [[ "$operator" != "maxpool" && "$operator" != "conv" ]]; then
    print -u2 "Unsupported operator for this experiment: $operator"
    exit 2
  fi
  for regime in "${(@f)$(split_csv "$REGIMES")}"; do
    if [[ "$regime" != "low" && "$regime" != "high" ]]; then
      print -u2 "This experiment accepts only low/high regimes: $regime"
      exit 2
    fi
    for (( trial_index = 0; trial_index < TRIALS; trial_index++ )); do
      for scope in "${(@f)$(split_csv "$SCOPES")}"; do
        scope_modifier "$scope" >/dev/null
        pass_index=0
        for pass_spec in "${(@f)$(split_passes "$EVENT_PASSES")}"; do
          run_profile_pass "$operator" "$regime" "$trial_index" "$scope" "$pass_index" "$pass_spec"
          (( pass_index += 1 ))
        done
      done
    done
  done
done

print "perf record experiment complete: $OUTPUT_ROOT"
print "Visualize with:"
print "  $PYTHON $PROJECT_DIR/visualization.py --input-dir $OUTPUT_ROOT"
