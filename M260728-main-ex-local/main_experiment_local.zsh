#!/usr/bin/env zsh
set -euo pipefail

SCRIPT_DIR="${0:A:h}"
LOCAL_REPO_DIR="${SCRIPT_DIR:h}"

SSH_USER="${SSH_USER:-rasheed}"
SSH_PASSWORD="${SSH_PASSWORD:-modenaottun}"
SSH_PORT="${SSH_PORT:-22}"
REMOTE_REPO_DIR="${REMOTE_REPO_DIR:-/home/rasheed/kuchida/antivenom_infocom}"
REMOTE_PROJECT_DIR="${REMOTE_PROJECT_DIR:-${REMOTE_REPO_DIR}/M260728-main-ex-local}"
REMOTE_PYTHON="${REMOTE_PYTHON:-${REMOTE_REPO_DIR}/venv/bin/python}"
REMOTE_IID_DATA_DIR="${REMOTE_IID_DATA_DIR:-${REMOTE_REPO_DIR}/iid-data}"
REMOTE_NONIID_DATA_DIR="${REMOTE_NONIID_DATA_DIR:-${REMOTE_REPO_DIR}/non-iid-data}"
REMOTE_BG_SCRIPT="${REMOTE_BG_SCRIPT:-${REMOTE_REPO_DIR}/M260718-robustness/run_bg_workloads.sh}"

REMOTE_LOG_BASE="${REMOTE_LOG_BASE:-logs/main_0728}"
REMOTE_PILOT_LOG_BASE="${REMOTE_PILOT_LOG_BASE:-logs/main_0728_pilot}"
LOCAL_LOG_BASE="${LOCAL_LOG_BASE:-${SCRIPT_DIR}/collected_logs}"

TRIALS="${TRIALS:-5}"
LOCAL_EPOCHS="${LOCAL_EPOCHS:-10}"
PILOT_TRIALS=1
PILOT_EPOCHS=1
REFERENCE_TRIALS=0
NUM_ROUNDS=10
BASE_SEED="${BASE_SEED:-260728}"
LEARNING_RATE="${LEARNING_RATE:-0.001}"
HARDWARE_FPS="${HARDWARE_FPS:-50}"
PERF_FPS="${PERF_FPS:-50}"
MONITOR_PHASES="${MONITOR_PHASES:-forward,backward}"
MIN_FORWARD_SAMPLES="${MIN_FORWARD_SAMPLES:-2.0}"
BADSAMPLER_KAPPA="${BADSAMPLER_KAPPA:-2.0}"

CIFAR_CNN_BATCH_SIZE="${CIFAR_CNN_BATCH_SIZE:-128}"
MODEL_BATCH_SIZE="${MODEL_BATCH_SIZE:-16}"
HIGH_RES_BATCH_SIZE="${HIGH_RES_BATCH_SIZE:-16}"
CIFAR_EXPECTED_BATCHES="${CIFAR_EXPECTED_BATCHES:-16}"
CIFAR_MODEL_MAX_TRAIN_SAMPLES="${CIFAR_MODEL_MAX_TRAIN_SAMPLES:-256}"
NONIID_ALPHA="${NONIID_ALPHA:-0.3}"

CIFAR_DATASET="uoft-cs/cifar10"
CIFAR_SLUG="cifar10"
TRASHNET_DATASET="kuchidareo/small_trashnet"
TRASHNET_SLUG="small_trashnet"
CHINESE_DATASET="kuchidareo/chinese_trafficsign_dataset"
CHINESE_SLUG="chinese_trafficsign_dataset"

BG_PROFILE="${BG_PROFILE:-medium}"
BG_PID_FILE="${BG_PID_FILE:-/tmp/antivenom_main_0728_bg.pid}"
BG_OUTPUT="${BG_OUTPUT:-${REMOTE_PROJECT_DIR}/logs/bg_workloads/run_bg_workloads.out}"

RPI_PERF_EVENTS="${RPI_PERF_EVENTS:-cycles,instructions,task-clock,context-switches,cpu-migrations,page-faults,branch-misses,l1d_cache_rd,l1d_cache_refill_rd,l1d_cache_wr,l1d_cache_refill_wr,l2d_cache_rd,l2d_cache_refill_rd,l2d_cache_wr,l2d_cache_refill_wr,bus_access_rd,bus_access_wr,mem_access,ase_spec,vfp_spec,inst_spec}"

DEVICE_SPECS=(
  "client_0|192.168.0.112"
  "client_1|192.168.0.113"
  "client_2|192.168.0.114"
  "client_3|192.168.0.115"
  "client_4|192.168.0.116"
  "client_5|192.168.0.117"
  "client_6|192.168.0.118"
  "client_7|192.168.0.119"
  "client_8|192.168.0.120"
  "client_9|192.168.0.121"
)

# label|poisoning_method|augmentation_profile|data_variant|partition_method
CONDITION_SPECS=(
  "clean|clean|baseline|iid|iid"
  "moderate_augmentation|clean|moderate|iid|iid"
  "strong_augmentation|clean|strong|iid|iid"
  "availability_shortcuts|availability_shortcuts|baseline|iid|iid"
  "badsampling|badsampling|baseline|iid|iid"
  "non_iid|clean|baseline|non_iid|dirichlet_noniid"
)

ACTION="${1:-both}"
typeset -a ACTIVE_DEVICE_SPECS
BG_ACTIVE=0

usage() {
  cat <<'EOF'
Usage:
  ./main_experiment.zsh pull
  ./main_experiment.zsh sync
  ./main_experiment.zsh check
  ./main_experiment.zsh pilot
  ./main_experiment.zsh run
  ./main_experiment.zsh collect
  ./main_experiment.zsh cifar10-cnn
  ./main_experiment.zsh both

`both` performs: pull -> rsync prepared datasets -> check -> pilot gate -> full run -> collect.
`run` performs: check -> pilot gate -> full run.
`cifar10-cnn` pulls the repository and runs only Phase 1 without check/pilot.

Dataset preparation is not run here. `sync` copies the server's existing
`iid-data/` and `non-iid-data/` trees to every reachable Raspberry Pi.

Phase 1, concurrently on reachable Raspberry Pis 192.168.0.112-121:
  CIFAR-10, SimpleCNN, batch 128, 6 independent conditions.

Phase 2, after the Phase 1 barrier:
  .112 CIFAR-10 ResNet18, batch 16
  .113 CIFAR-10 MobileNetV3-Small, batch 16
  .114 CIFAR-10 TinyViT depth 4, batch 16
  .115 Small TrashNet SimpleCNN, 224x224, batch 16
  .116 Chinese traffic signs SimpleCNN, 224x224, batch 16
  .117 CIFAR-10 SimpleCNN, batch 128, under BG I, II, and I+II

The sixth condition is clean Dirichlet non-IID (alpha 0.3).
CIFAR batch-16 model stages use at most 256 train samples (about 16 batches).
Every full stage uses 5 trials and 10 epochs. Reference trials are disabled.
EOF
}

ssh_base_cmd() {
  if [[ -n "$SSH_PASSWORD" ]]; then
    command -v sshpass >/dev/null 2>&1 || {
      print -u2 "SSH_PASSWORD is set, but sshpass is not installed."
      return 1
    }
    print -- "sshpass -p ${(q)SSH_PASSWORD} ssh -p ${(q)SSH_PORT} -o StrictHostKeyChecking=accept-new -o ConnectTimeout=12 -o ServerAliveInterval=10 -o ServerAliveCountMax=3"
  else
    print -- "ssh -p ${(q)SSH_PORT} -o StrictHostKeyChecking=accept-new -o ConnectTimeout=12 -o ServerAliveInterval=10 -o ServerAliveCountMax=3"
  fi
}

ssh_run() {
  local host="$1"
  local remote_command="$2"
  local ssh_cmd="$(ssh_base_cmd)"
  eval "$ssh_cmd ${(q)SSH_USER}@${(q)host} ${(q)remote_command}"
}

host_is_reachable() {
  ping -c 1 -W 1 "$1" >/dev/null 2>&1
}

refresh_active_devices() {
  ACTIVE_DEVICE_SPECS=()
  local spec host
  for spec in "${DEVICE_SPECS[@]}"; do
    host="${spec#*|}"
    if host_is_reachable "$host"; then
      ACTIVE_DEVICE_SPECS+=("$spec")
    else
      print -u2 "Skipping unreachable device: $host"
    fi
  done
  (( ${#ACTIVE_DEVICE_SPECS[@]} > 0 )) || {
    print -u2 "No configured Raspberry Pi is reachable."
    return 1
  }
}

active_spec_for_host() {
  local wanted="$1"
  local spec
  for spec in "${ACTIVE_DEVICE_SPECS[@]}"; do
    if [[ "${spec#*|}" == "$wanted" ]]; then
      print -r -- "$spec"
      return 0
    fi
  done
  return 1
}

wait_for_jobs() {
  local -a job_pids=("$@")
  local job_pid
  local failed=0
  for job_pid in "${job_pids[@]}"; do
    if ! wait "$job_pid"; then
      failed=1
    fi
  done
  (( failed == 0 ))
}

pull_repositories() {
  refresh_active_devices
  print "Pulling the latest repository on ${#ACTIVE_DEVICE_SPECS} reachable devices..."
  local -a job_pids
  local spec host
  for spec in "${ACTIVE_DEVICE_SPECS[@]}"; do
    host="${spec#*|}"
    (
      print "==> git pull --rebase $host"
      ssh_run "$host" "cd '$REMOTE_REPO_DIR' && git pull --rebase"
    ) &
    job_pids+=("$!")
  done
  wait_for_jobs "${job_pids[@]}"
}

sync_datasets() {
  refresh_active_devices
  local sync_script="${LOCAL_REPO_DIR}/sync_shared_datasets_to_devices.zsh"
  test -x "$sync_script" || {
    print -u2 "Missing executable dataset sync script: $sync_script"
    return 2
  }

  local -a hosts
  local spec host
  for spec in "${ACTIVE_DEVICE_SPECS[@]}"; do
    host="${spec#*|}"
    hosts+=("$host")
  done

  print "Rsyncing the prepared iid-data and non-iid-data trees to ${#hosts} reachable Raspberry Pis..."
  SSH_USER="$SSH_USER" \
    SSH_PASSWORD="$SSH_PASSWORD" \
    SSH_PORT="$SSH_PORT" \
    REMOTE_REPO_DIR="$REMOTE_REPO_DIR" \
    "$sync_script" "${hosts[@]}"
}

dataset_check_command() {
  local data_root="$1"
  local slug="$2"
  local client_id="$3"
  local expected_batches="$4"
  local batch_size="$5"
  print -r -- "
    root='$data_root/$slug'
    test -f \"\$root/partition_metadata.csv\"
    test -f \"\$root/augmented/moderate/PREPARED\"
    test -f \"\$root/augmented/strong/PREPARED\"
    test -f \"\$root/poisoned/badsampling/$client_id/sampling_plan.json\"
    '$REMOTE_PYTHON' -c \"import csv, json, math, pathlib; p=pathlib.Path(r'\$root/partition_metadata.csv'); rows=list(csv.DictReader(p.open())); clean=[r for r in rows if r.get('dataset_split') == 'train' and r.get('client_id') == '$client_id' and r.get('poisoning_method') == 'clean']; shortcut=[r for r in rows if r.get('dataset_split') == 'train' and r.get('client_id') == '$client_id' and r.get('poisoning_method') == 'availability_shortcuts']; plan=json.loads(pathlib.Path(r'\$root/poisoned/badsampling/$client_id/sampling_plan.json').read_text()); candidates=plan.get('candidates', []); expected_batches=$expected_batches; batch_size=$batch_size; actual_batches=math.ceil(len(clean) / batch_size); assert clean, 'no clean training rows'; assert {r.get('partition_method') for r in clean} == {'iid'}, 'IID metadata mismatch'; assert expected_batches <= 0 or actual_batches == expected_batches, f'expected {expected_batches} batches at batch size {batch_size}, got {actual_batches} from {len(clean)} rows'; assert len(shortcut) == len(clean), f'shortcut rows {len(shortcut)} != clean rows {len(clean)}'; assert plan.get('source_poisoning_method') == 'clean', 'BadSampler plan must use clean records'; assert len(candidates) == len(clean), f'BadSampler candidates {len(candidates)} != clean rows {len(clean)}'; print('$slug $client_id train_count', len(clean), 'batches', actual_batches)\""
}

noniid_dataset_check_command() {
  local slug="$1"
  local client_id="$2"
  local batch_size="$3"
  local max_samples="$4"
  local expected_batches="${5:-0}"
  print -r -- "
    root='$REMOTE_NONIID_DATA_DIR/$slug'
    test -f \"\$root/partition_metadata.csv\"
    '$REMOTE_PYTHON' -c \"import csv, math, pathlib; p=pathlib.Path(r'\$root/partition_metadata.csv'); rows=list(csv.DictReader(p.open())); clean=[r for r in rows if r.get('dataset_split') == 'train' and r.get('client_id') == '$client_id' and r.get('poisoning_method') == 'clean']; methods={r.get('partition_method') for r in clean}; limit=$max_samples; expected=$expected_batches; used=min(len(clean), limit) if limit > 0 else len(clean); batches=math.ceil(used / $batch_size); assert clean, 'no non-IID clean training rows'; assert methods == {'dirichlet_noniid'}, f'unexpected partition methods: {methods}'; assert expected <= 0 or batches == expected, f'expected {expected} non-IID batches, got {batches}'; print('$slug $client_id noniid_train_count', len(clean), 'used', used, 'batches_at_bs_$batch_size', batches)\""
}

check_one_device() {
  local spec="$1"
  local client_id="${spec%%|*}"
  local host="${spec#*|}"
  local checks="$(dataset_check_command "$REMOTE_IID_DATA_DIR" "$CIFAR_SLUG" "$client_id" "$CIFAR_EXPECTED_BATCHES" "$CIFAR_CNN_BATCH_SIZE")"
  checks+="$(noniid_dataset_check_command "$CIFAR_SLUG" "$client_id" "$CIFAR_CNN_BATCH_SIZE" 0 "$CIFAR_EXPECTED_BATCHES")"

  if [[ "$host" == "192.168.0.112" || "$host" == "192.168.0.113" || "$host" == "192.168.0.114" ]]; then
    checks+="$(noniid_dataset_check_command "$CIFAR_SLUG" "$client_id" "$MODEL_BATCH_SIZE" "$CIFAR_MODEL_MAX_TRAIN_SAMPLES" "$CIFAR_EXPECTED_BATCHES")"
  fi

  if [[ "$host" == "192.168.0.115" ]]; then
    checks+="$(dataset_check_command "$REMOTE_IID_DATA_DIR" "$TRASHNET_SLUG" "$client_id" 0 "$HIGH_RES_BATCH_SIZE")"
    checks+="$(noniid_dataset_check_command "$TRASHNET_SLUG" "$client_id" "$HIGH_RES_BATCH_SIZE" 0 0)"
  fi
  if [[ "$host" == "192.168.0.116" ]]; then
    checks+="$(dataset_check_command "$REMOTE_IID_DATA_DIR" "$CHINESE_SLUG" "$client_id" 0 "$HIGH_RES_BATCH_SIZE")"
    checks+="$(noniid_dataset_check_command "$CHINESE_SLUG" "$client_id" "$HIGH_RES_BATCH_SIZE" 0 0)"
  fi

  print "==> environment check $host $client_id"
  ssh_run "$host" "
    set -e
    test -d '$REMOTE_PROJECT_DIR'
    test -x '$REMOTE_PYTHON'
    test -f '$REMOTE_PROJECT_DIR/running_ml.py'
    test -f '$REMOTE_PROJECT_DIR/forward_timing_summary.py'
    cd '$REMOTE_PROJECT_DIR'
    '$REMOTE_PYTHON' --version
    '$REMOTE_PYTHON' -c \"import torch, torchvision, psutil, PIL, numpy; from models import get_model; assert not torch.cuda.is_available(); names=['simple_cnn','resnet18','mobilenet_v3_small','tiny_vit']; [get_model(name,10,(32,32),model_depth=4) for name in names]; print('models=' + ','.join(names)); print('torch=' + torch.__version__); print('torchvision=' + torchvision.__version__)\"
    perf --version
    printf '%s\n' '$SSH_PASSWORD' | sudo -S sysctl kernel.perf_event_paranoid=-1 >/dev/null
    perf stat -e '$RPI_PERF_EVENTS' -- '$REMOTE_PYTHON' -c 'sum(i*i for i in range(1000000))' >/dev/null 2>&1
    $checks
  "
}

check_bg_environment() {
  active_spec_for_host "192.168.0.117" >/dev/null || return 0
  print "==> background-workload check 192.168.0.117"
  ssh_run "192.168.0.117" "
    set -e
    test -x '$REMOTE_BG_SCRIPT'
    command -v iperf3 >/dev/null
    '$REMOTE_PYTHON' -c 'import cv2, numpy'
    printf '%s\n' background_dependencies=ok
    PYTHON_BIN='$REMOTE_PYTHON' '$REMOTE_BG_SCRIPT' --group group1 --profile '$BG_PROFILE' --dry-run
    PYTHON_BIN='$REMOTE_PYTHON' '$REMOTE_BG_SCRIPT' --group group2 --profile '$BG_PROFILE' --dry-run
    PYTHON_BIN='$REMOTE_PYTHON' '$REMOTE_BG_SCRIPT' --group both --profile '$BG_PROFILE' --dry-run
  "
}

check_environments() {
  refresh_active_devices
  local -a job_pids
  local spec
  for spec in "${ACTIVE_DEVICE_SPECS[@]}"; do
    check_one_device "$spec" &
    job_pids+=("$!")
  done
  wait_for_jobs "${job_pids[@]}"
  check_bg_environment
  print "Environment checks passed on ${#ACTIVE_DEVICE_SPECS} devices."
}

augment_json() {
  local profile="$1"
  local resize="$2"
  print -r -- "{\"enabled\":true,\"_profile\":\"${profile}\",\"resize\":[${resize},${resize}],\"horizontal_flip\":false,\"normalize\":true}"
}

run_stage() {
  local host="$1"
  local client_id="$2"
  local log_base="$3"
  local stage_label="$4"
  local dataset_name="$5"
  local model_name="$6"
  local model_depth="$7"
  local batch_size="$8"
  local resize="$9"
  local condition_spec="${10}"
  local trials="${11}"
  local epochs="${12}"
  local max_train_samples="${13:-0}"
  local bg_group="${14:-none}"
  local -a fields
  fields=("${(@ps:|:)condition_spec}")
  local condition_label="${fields[1]}"
  local poisoning_method="${fields[2]}"
  local augmentation_profile="${fields[3]}"
  local data_variant="${fields[4]}"
  local partition_method="${fields[5]}"
  local data_dir="$REMOTE_IID_DATA_DIR"
  if [[ "$data_variant" == "non_iid" ]]; then
    data_dir="$REMOTE_NONIID_DATA_DIR"
  fi
  local stage_dir="${log_base}/${host}/${stage_label}/${condition_label}"
  local augment="$(augment_json "$augmentation_profile" "$resize")"
  local run_role="analysis"
  local bg_profile="none"
  local bg_args
  if [[ "$bg_group" != "none" ]]; then
    bg_profile="$BG_PROFILE"
  fi
  if [[ "$log_base" == "$REMOTE_PILOT_LOG_BASE" ]]; then
    run_role="pilot"
  fi
  bg_args="--background-workload-group '$bg_group' --background-workload-profile '$bg_profile'"
  if [[ "$bg_group" != "none" ]]; then
    bg_args="--background-workload-enabled $bg_args"
  fi

  print "==> $host $stage_label/$condition_label"
  print "    dataset=$dataset_name partition=$partition_method model=$model_name batch=$batch_size max_samples=$max_train_samples trials=$trials epochs=$epochs bg=$bg_group"
  ssh_run "$host" "
    set -e
    cd '$REMOTE_PROJECT_DIR'
    mkdir -p '$stage_dir'
    run_marker=\$(mktemp)
    CUDA_VISIBLE_DEVICES='' '$REMOTE_PYTHON' running_ml.py \\
      --experiment-id 'main_0728_${stage_label}_${condition_label}' \\
      --run-role '$run_role' \\
      --dataset '$dataset_name' \\
      --dataset-split train \\
      --data-dir '$data_dir' \\
      --log-dir '$stage_dir' \\
      --client-id '$client_id' \\
      --device-id '$host' \\
      --host '$host' \\
      --poisoning-method '$poisoning_method' \\
      --partition-method '$partition_method' \\
      --noniid-alpha '$NONIID_ALPHA' \\
      --reference-trials '$REFERENCE_TRIALS' \\
      --trials '$trials' \\
      --local-epochs '$epochs' \\
      --batch-size '$batch_size' \\
      --max-train-samples '$max_train_samples' \\
      --num-rounds '$NUM_ROUNDS' \\
      --learning-rate '$LEARNING_RATE' \\
      --seed '$BASE_SEED' \\
      --badsampler-kappa '$BADSAMPLER_KAPPA' \\
      --perf-events '$RPI_PERF_EVENTS' \\
      --perf-fps '$PERF_FPS' \\
      --hardware-fps '$HARDWARE_FPS' \\
      --monitor-phases '$MONITOR_PHASES' \\
      --augment '$augment' \\
      --model '$model_name' \\
      --model-depth '$model_depth' \\
      --model-width-multiplier 1.0 \\
      --model-target-pam-mb 0 \\
      $bg_args

    hardware_count=\$(find '$stage_dir' -maxdepth 1 -type f -name '*.csv' ! -name '*_metrics.csv' ! -name '*_perf.csv' -newer \"\$run_marker\" | wc -l)
    metrics_count=\$(find '$stage_dir' -maxdepth 1 -type f -name '*_metrics.csv' -newer \"\$run_marker\" | wc -l)
    perf_count=\$(find '$stage_dir' -maxdepth 1 -type f -name '*_perf.csv' -newer \"\$run_marker\" | wc -l)
    test \"\$hardware_count\" -eq '$trials'
    test \"\$metrics_count\" -eq '$trials'
    test \"\$perf_count\" -eq '$trials'
    for metrics_file in \$(find '$stage_dir' -maxdepth 1 -type f -name '*_metrics.csv' -newer \"\$run_marker\"); do
      grep -q 'forward_elapsed_ms' \"\$metrics_file\"
      grep -q 'clean_test_epoch' \"\$metrics_file\"
    done
    for perf_file in \$(find '$stage_dir' -maxdepth 1 -type f -name '*_perf.csv' -newer \"\$run_marker\"); do
      grep -q ',ok,' \"\$perf_file\"
    done
    rm -f \"\$run_marker\"
  "
}

run_condition_set() {
  local host="$1" client_id="$2" log_base="$3" stage_label="$4"
  local dataset_name="$5" model_name="$6" model_depth="$7"
  local batch_size="$8" resize="$9" trials="${10}" epochs="${11}"
  local max_train_samples="${12:-0}"
  local bg_group="${13:-none}"
  local condition_spec
  for condition_spec in "${CONDITION_SPECS[@]}"; do
    run_stage "$host" "$client_id" "$log_base" "$stage_label" "$dataset_name" \
      "$model_name" "$model_depth" "$batch_size" "$resize" "$condition_spec" \
      "$trials" "$epochs" "$max_train_samples" "$bg_group"
  done
}

start_bg_workload() {
  local group="$1"
  local host="192.168.0.117"
  print "==> start BG $group on $host"
  BG_ACTIVE=1
  ssh_run "$host" "
    set -e
    mkdir -p '${BG_OUTPUT:h}'
    if test -f '$BG_PID_FILE' && kill -0 \$(cat '$BG_PID_FILE') 2>/dev/null; then
      echo 'background workload already running' >&2
      exit 3
    fi
    nohup env PYTHON_BIN='$REMOTE_PYTHON' '$REMOTE_BG_SCRIPT' --group '$group' --profile '$BG_PROFILE' >'$BG_OUTPUT' 2>&1 < /dev/null &
    echo \$! > '$BG_PID_FILE'
    sleep 3
    kill -0 \$(cat '$BG_PID_FILE')
  "
}

stop_bg_workload() {
  local host="192.168.0.117"
  if (( BG_ACTIVE == 0 )); then
    return 0
  fi
  print "==> stop BG on $host"
  ssh_run "$host" "
    if test -f '$BG_PID_FILE'; then
      pid=\$(cat '$BG_PID_FILE')
      kill \"\$pid\" 2>/dev/null || true
      for ignored in 1 2 3 4 5; do
        kill -0 \"\$pid\" 2>/dev/null || break
        sleep 1
      done
      kill -9 \"\$pid\" 2>/dev/null || true
      rm -f '$BG_PID_FILE'
    fi
  " || true
  BG_ACTIVE=0
}

run_bg_condition_sets() {
  local log_base="$1" trials="$2" epochs="$3"
  local group rc_code=0
  for group in group1 group2 both; do
    start_bg_workload "$group"
    run_condition_set "192.168.0.117" "client_5" "$log_base" \
      "phase2_bg_${group}_cifar10_cnn" "$CIFAR_DATASET" "simple_cnn" 3 \
      "$CIFAR_CNN_BATCH_SIZE" 32 "$trials" "$epochs" 0 "$group" || rc_code=$?
    stop_bg_workload
    (( rc_code == 0 )) || return "$rc_code"
  done
}

summarize_pilot_stage() {
  local host="$1" stage_dir="$2" batch_size="$3"
  local expected_batches="${4:-0}"
  ssh_run "$host" "
    cd '$REMOTE_PROJECT_DIR'
    '$REMOTE_PYTHON' forward_timing_summary.py \\
      --input-dir '$stage_dir/clean' \\
      --configured-batch-size '$batch_size' \\
      --hardware-fps '$HARDWARE_FPS' \\
      --minimum-expected-samples '$MIN_FORWARD_SAMPLES' \\
      --minimum-actual-samples '$MIN_FORWARD_SAMPLES' \\
      --expected-batches '$expected_batches' \\
      --latest-run \\
      --fail-below
  "
}

run_common_pilot_device() {
  local spec="$1"
  local client_id="${spec%%|*}" host="${spec#*|}"
  local stage="phase1_cifar10_cnn"
  run_stage "$host" "$client_id" "$REMOTE_PILOT_LOG_BASE" "$stage" \
    "$CIFAR_DATASET" simple_cnn 3 "$CIFAR_CNN_BATCH_SIZE" 32 \
    "clean|clean|baseline|iid|iid" "$PILOT_TRIALS" "$PILOT_EPOCHS"
  summarize_pilot_stage "$host" "$REMOTE_PILOT_LOG_BASE/$host/$stage" "$CIFAR_CNN_BATCH_SIZE" "$CIFAR_EXPECTED_BATCHES"
}

run_special_pilot() {
  local host="$1" client_id="$2" stage="$3" dataset="$4" model="$5"
  local depth="$6" batch="$7" resize="$8" max_train_samples="${9:-0}"
  run_stage "$host" "$client_id" "$REMOTE_PILOT_LOG_BASE" "$stage" "$dataset" \
    "$model" "$depth" "$batch" "$resize" "clean|clean|baseline|iid|iid" \
    "$PILOT_TRIALS" "$PILOT_EPOCHS" "$max_train_samples"
  local expected_batches=0
  if (( max_train_samples > 0 )); then
    expected_batches=$(( max_train_samples / batch ))
  fi
  summarize_pilot_stage "$host" "$REMOTE_PILOT_LOG_BASE/$host/$stage" "$batch" "$expected_batches"
}

run_pilot() {
  refresh_active_devices
  print "Running Phase 1 forward-duration pilots on ${#ACTIVE_DEVICE_SPECS} devices..."
  local -a job_pids
  local spec
  for spec in "${ACTIVE_DEVICE_SPECS[@]}"; do
    run_common_pilot_device "$spec" &
    job_pids+=("$!")
  done
  wait_for_jobs "${job_pids[@]}"

  print "Running specialized configuration pilots..."
  job_pids=()
  active_spec_for_host 192.168.0.112 >/dev/null && {
    run_special_pilot 192.168.0.112 client_0 phase2_cifar10_resnet18 "$CIFAR_DATASET" resnet18 18 "$MODEL_BATCH_SIZE" 32 "$CIFAR_MODEL_MAX_TRAIN_SAMPLES" &
    job_pids+=("$!")
  }
  active_spec_for_host 192.168.0.113 >/dev/null && {
    run_special_pilot 192.168.0.113 client_1 phase2_cifar10_mobilenet_v3_small "$CIFAR_DATASET" mobilenet_v3_small 3 "$MODEL_BATCH_SIZE" 32 "$CIFAR_MODEL_MAX_TRAIN_SAMPLES" &
    job_pids+=("$!")
  }
  active_spec_for_host 192.168.0.114 >/dev/null && {
    run_special_pilot 192.168.0.114 client_2 phase2_cifar10_tiny_vit "$CIFAR_DATASET" tiny_vit 4 "$MODEL_BATCH_SIZE" 32 "$CIFAR_MODEL_MAX_TRAIN_SAMPLES" &
    job_pids+=("$!")
  }
  active_spec_for_host 192.168.0.115 >/dev/null && {
    run_special_pilot 192.168.0.115 client_3 phase2_trashnet_cnn "$TRASHNET_DATASET" simple_cnn 3 "$HIGH_RES_BATCH_SIZE" 224 &
    job_pids+=("$!")
  }
  active_spec_for_host 192.168.0.116 >/dev/null && {
    run_special_pilot 192.168.0.116 client_4 phase2_chinese_cnn "$CHINESE_DATASET" simple_cnn 3 "$HIGH_RES_BATCH_SIZE" 224 &
    job_pids+=("$!")
  }
  wait_for_jobs "${job_pids[@]}"

  if active_spec_for_host 192.168.0.117 >/dev/null; then
    local group stage
    for group in group1 group2 both; do
      stage="phase2_bg_${group}_cifar10_cnn"
      start_bg_workload "$group"
      run_stage 192.168.0.117 client_5 "$REMOTE_PILOT_LOG_BASE" "$stage" \
        "$CIFAR_DATASET" simple_cnn 3 "$CIFAR_CNN_BATCH_SIZE" 32 \
        "clean|clean|baseline|iid|iid" "$PILOT_TRIALS" "$PILOT_EPOCHS" 0 "$group"
      stop_bg_workload
      summarize_pilot_stage 192.168.0.117 "$REMOTE_PILOT_LOG_BASE/192.168.0.117/$stage" "$CIFAR_CNN_BATCH_SIZE" "$CIFAR_EXPECTED_BATCHES"
    done
  fi
  print "All forward-duration pilots passed."
}

run_phase1_device() {
  local spec="$1"
  run_condition_set "${spec#*|}" "${spec%%|*}" "$REMOTE_LOG_BASE" \
    phase1_cifar10_cnn "$CIFAR_DATASET" simple_cnn 3 \
    "$CIFAR_CNN_BATCH_SIZE" 32 "$TRIALS" "$LOCAL_EPOCHS"
}

run_phase1() {
  print "Running Phase 1 concurrently on ${#ACTIVE_DEVICE_SPECS} devices..."
  local -a job_pids
  local spec
  for spec in "${ACTIVE_DEVICE_SPECS[@]}"; do
    run_phase1_device "$spec" &
    job_pids+=("$!")
  done
  wait_for_jobs "${job_pids[@]}"
  print "Phase 1 completed on every reachable device."
}

run_phase2() {
  print "Running assigned Phase 2 workloads concurrently..."
  local -a job_pids
  active_spec_for_host 192.168.0.112 >/dev/null && {
    run_condition_set 192.168.0.112 client_0 "$REMOTE_LOG_BASE" phase2_cifar10_resnet18 \
      "$CIFAR_DATASET" resnet18 18 "$MODEL_BATCH_SIZE" 32 "$TRIALS" "$LOCAL_EPOCHS" "$CIFAR_MODEL_MAX_TRAIN_SAMPLES" &
    job_pids+=("$!")
  }
  active_spec_for_host 192.168.0.113 >/dev/null && {
    run_condition_set 192.168.0.113 client_1 "$REMOTE_LOG_BASE" phase2_cifar10_mobilenet_v3_small \
      "$CIFAR_DATASET" mobilenet_v3_small 3 "$MODEL_BATCH_SIZE" 32 "$TRIALS" "$LOCAL_EPOCHS" "$CIFAR_MODEL_MAX_TRAIN_SAMPLES" &
    job_pids+=("$!")
  }
  active_spec_for_host 192.168.0.114 >/dev/null && {
    run_condition_set 192.168.0.114 client_2 "$REMOTE_LOG_BASE" phase2_cifar10_tiny_vit \
      "$CIFAR_DATASET" tiny_vit 4 "$MODEL_BATCH_SIZE" 32 "$TRIALS" "$LOCAL_EPOCHS" "$CIFAR_MODEL_MAX_TRAIN_SAMPLES" &
    job_pids+=("$!")
  }
  active_spec_for_host 192.168.0.115 >/dev/null && {
    run_condition_set 192.168.0.115 client_3 "$REMOTE_LOG_BASE" phase2_trashnet_cnn \
      "$TRASHNET_DATASET" simple_cnn 3 "$HIGH_RES_BATCH_SIZE" 224 "$TRIALS" "$LOCAL_EPOCHS" &
    job_pids+=("$!")
  }
  active_spec_for_host 192.168.0.116 >/dev/null && {
    run_condition_set 192.168.0.116 client_4 "$REMOTE_LOG_BASE" phase2_chinese_cnn \
      "$CHINESE_DATASET" simple_cnn 3 "$HIGH_RES_BATCH_SIZE" 224 "$TRIALS" "$LOCAL_EPOCHS" &
    job_pids+=("$!")
  }
  active_spec_for_host 192.168.0.117 >/dev/null && {
    run_bg_condition_sets "$REMOTE_LOG_BASE" "$TRIALS" "$LOCAL_EPOCHS" &
    job_pids+=("$!")
  }
  wait_for_jobs "${job_pids[@]}"
  print "Phase 2 completed."
}

run_full_experiment() {
  refresh_active_devices
  print "Main experiment: ${#ACTIVE_DEVICE_SPECS} reachable devices, $TRIALS trials, $LOCAL_EPOCHS epochs"
  print "  CIFAR SimpleCNN batch=$CIFAR_CNN_BATCH_SIZE; other models/high-resolution batch=$MODEL_BATCH_SIZE"
  print "  hardware/perf sampling=${HARDWARE_FPS}/${PERF_FPS} FPS"
  print "  hardware/perf retained phases=$MONITOR_PHASES"
  run_phase1
  run_phase2
}

collect_logs() {
  refresh_active_devices
  mkdir -p "$LOCAL_LOG_BASE"
  local -a job_pids
  local spec host destination
  for spec in "${ACTIVE_DEVICE_SPECS[@]}"; do
    host="${spec#*|}"
    destination="$LOCAL_LOG_BASE/$host"
    (
      mkdir -p "$destination/full" "$destination/pilot"
      local -a full_command=(
        rsync -az --partial --stats --human-readable
        -e "ssh -p ${SSH_PORT} -o StrictHostKeyChecking=accept-new -o ConnectTimeout=12"
        "${SSH_USER}@${host}:${REMOTE_PROJECT_DIR}/${REMOTE_LOG_BASE}/${host}/"
        "$destination/full/"
      )
      local -a pilot_command=(
        rsync -az --partial --stats --human-readable
        -e "ssh -p ${SSH_PORT} -o StrictHostKeyChecking=accept-new -o ConnectTimeout=12"
        "${SSH_USER}@${host}:${REMOTE_PROJECT_DIR}/${REMOTE_PILOT_LOG_BASE}/${host}/"
        "$destination/pilot/"
      )
      if [[ -n "$SSH_PASSWORD" ]]; then
        sshpass -p "$SSH_PASSWORD" "${full_command[@]}"
        sshpass -p "$SSH_PASSWORD" "${pilot_command[@]}"
      else
        "${full_command[@]}"
        "${pilot_command[@]}"
      fi
    ) &
    job_pids+=("$!")
  done
  wait_for_jobs "${job_pids[@]}"
  print "Logs collected under $LOCAL_LOG_BASE"
}

cleanup() {
  stop_bg_workload
}
trap cleanup EXIT INT TERM

case "$ACTION" in
  pull)
    pull_repositories
    ;;
  sync)
    sync_datasets
    ;;
  check)
    check_environments
    ;;
  pilot)
    check_environments
    run_pilot
    ;;
  run)
    check_environments
    run_pilot
    run_full_experiment
    ;;
  collect)
    collect_logs
    ;;
  cifar10-cnn)
    pull_repositories
    refresh_active_devices
    print "Focused experiment: CIFAR-10 + SimpleCNN, 6 conditions, $TRIALS trials, $LOCAL_EPOCHS epochs"
    run_phase1
    ;;
  both)
    pull_repositories
    sync_datasets
    check_environments
    run_pilot
    run_full_experiment
    collect_logs
    ;;
  -h|--help|help)
    usage
    ;;
  *)
    usage >&2
    exit 1
    ;;
esac
