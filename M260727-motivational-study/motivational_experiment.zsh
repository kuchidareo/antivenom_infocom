#!/usr/bin/env zsh
set -euo pipefail

SCRIPT_DIR="${0:A:h}"
LOCAL_REPO_DIR="${SCRIPT_DIR:h}"

SSH_USER="${SSH_USER:-rasheed}"
SSH_PASSWORD="${SSH_PASSWORD:-modenaottun}"
SSH_PORT="${SSH_PORT:-22}"
DEVICE_HOST="${DEVICE_HOST:-192.168.0.112}"
REMOTE_REPO_DIR="${REMOTE_REPO_DIR:-/home/rasheed/kuchida/antivenom_infocom}"
REMOTE_PROJECT_DIR="${REMOTE_PROJECT_DIR:-${REMOTE_REPO_DIR}/M260727-motivational-study}"
REMOTE_PYTHON="${REMOTE_PYTHON:-${REMOTE_REPO_DIR}/venv/bin/python}"
REMOTE_IID_DATA_DIR="${REMOTE_IID_DATA_DIR:-${REMOTE_REPO_DIR}/iid-data}"
LOCAL_IID_DATA_DIR="${LOCAL_IID_DATA_DIR:-${LOCAL_REPO_DIR}/iid-data}"
REMOTE_LOG_BASE="${REMOTE_LOG_BASE:-logs/motivational_0727_30_trials}"
LOCAL_LOG_BASE="${LOCAL_LOG_BASE:-${SCRIPT_DIR}/collected_logs_30_trials}"

MONITORING_FPS="${MONITORING_FPS:-10}"
PERF_FPS="${PERF_FPS:-$MONITORING_FPS}"
HARDWARE_FPS="${HARDWARE_FPS:-$MONITORING_FPS}"
TRIALS="${TRIALS:-30}"
LOCAL_EPOCHS="${LOCAL_EPOCHS:-10}"
BATCH_SIZE="${BATCH_SIZE:-16}"
BASE_SEED="${BASE_SEED:-260727}"
LEARNING_RATE="${LEARNING_RATE:-0.001}"

DATASET_NAME="uoft-cs/cifar10"
DATASET_SLUG="cifar10"
CLIENT_ID="${CLIENT_ID:-client_0}"
REFERENCE_TRIALS=0
NUM_ROUNDS=10

RPI_PERF_EVENTS="${RPI_PERF_EVENTS:-cycles,instructions,task-clock,context-switches,cpu-migrations,page-faults,branches,branch-misses,l1d_cache_rd,l1d_cache_refill_rd,l1d_cache_wr,l1d_cache_refill_wr,l2d_cache_rd,l2d_cache_refill_rd,l2d_cache_wr,l2d_cache_refill_wr,bus_access_rd,bus_access_wr,mem_access,ase_spec,vfp_spec,inst_spec}"
JETSON_PERF_EVENTS="${JETSON_PERF_EVENTS:-cycles,instructions,task-clock,context-switches,cpu-migrations,page-faults,br_retired,br_mis_pred_retired,l1d_cache,l1d_cache_refill,l1d_cache_wb,l2d_cache,l2d_cache_refill,l2d_cache_wb,bus_access,mem_access,inst_spec}"

# label|poisoning_method|augmentation_profile|model|model_depth
STAGE_SPECS=(
  "cifar10_cnn_clean|clean|baseline|simple_cnn|3"
  "cifar10_cnn_moderate_augmentation|clean|moderate|simple_cnn|3"
  "cifar10_cnn_strong_augmentation|clean|strong|simple_cnn|3"
  "cifar10_cnn_availability_shortcut|availability_shortcuts|baseline|simple_cnn|3"
  "cifar10_vit_clean|clean|baseline|tiny_vit|4"
  "cifar10_vit_moderate_augmentation|clean|moderate|tiny_vit|4"
  "cifar10_vit_strong_augmentation|clean|strong|tiny_vit|4"
  "cifar10_vit_availability_shortcut|availability_shortcuts|baseline|tiny_vit|4"
)

ACTION="${1:-both}"

usage() {
  cat <<'EOF'
Usage:
  ./motivational_experiment.zsh [pull|sync|check|run|collect|both]

Default `both` flow:
  git pull -> sync IID CIFAR-10 -> check -> run -> collect

The eight stages run on the Raspberry Pi 4 at 192.168.0.112:
  SimpleCNN: clean, moderate augmentation, strong augmentation,
             availability shortcut
  TinyViT:   clean, moderate augmentation, strong augmentation,
             availability shortcut

Defaults:
  client partition: client_0
  trials:           30 per condition
  epochs:           10
  batch size:       16
  learning rate:    0.001
  perf/psutil:      10 Hz
  compute:          CPU (CUDA_VISIBLE_DEVICES is empty)
EOF
}

remote_log_root() {
  print -r -- "${REMOTE_LOG_BASE}/${DEVICE_HOST}"
}

local_log_dir() {
  print -r -- "${LOCAL_LOG_BASE}/${DEVICE_HOST}"
}

perf_events_for_device() {
  if [[ "$DEVICE_HOST" == "192.168.0.141" ]]; then
    print -r -- "$JETSON_PERF_EVENTS"
  else
    print -r -- "$RPI_PERF_EVENTS"
  fi
}

device_label() {
  if [[ "$DEVICE_HOST" == "192.168.0.141" ]]; then
    print -r -- "Jetson"
  else
    print -r -- "Raspberry Pi 4"
  fi
}

ssh_base_cmd() {
  if [[ -n "$SSH_PASSWORD" ]]; then
    if ! command -v sshpass >/dev/null 2>&1; then
      print "SSH_PASSWORD is set, but sshpass is not installed." >&2
      exit 1
    fi
    print -- "sshpass -p ${(q)SSH_PASSWORD} ssh -p ${(q)SSH_PORT} -o StrictHostKeyChecking=accept-new -o ConnectTimeout=15 -o ServerAliveInterval=10 -o ServerAliveCountMax=3"
  else
    print -- "ssh -p ${(q)SSH_PORT} -o StrictHostKeyChecking=accept-new -o ConnectTimeout=15 -o ServerAliveInterval=10 -o ServerAliveCountMax=3"
  fi
}

ssh_run() {
  local remote_command="$1"
  local ssh_cmd
  ssh_cmd="$(ssh_base_cmd)"
  eval "$ssh_cmd ${(q)SSH_USER}@${(q)DEVICE_HOST} ${(q)remote_command}"
}

pull_repository() {
  print "==> git pull --rebase ${SSH_USER}@${DEVICE_HOST}:${REMOTE_REPO_DIR}"
  ssh_run "cd '$REMOTE_REPO_DIR' && git pull --rebase"
}

sync_dataset() {
  local source_dir="${LOCAL_IID_DATA_DIR}/${DATASET_SLUG}"
  local destination_dir="${REMOTE_IID_DATA_DIR}/${DATASET_SLUG}"
  local -a rsync_command

  test -f "${source_dir}/partition_metadata.csv" || {
    print "Missing local CIFAR-10 metadata: ${source_dir}/partition_metadata.csv" >&2
    return 2
  }
  ssh_run "mkdir -p '$destination_dir'"
  print "==> rsync ${source_dir} -> ${SSH_USER}@${DEVICE_HOST}:${destination_dir}"
  rsync_command=(
    rsync -az --partial --stats --human-readable
    -e "ssh -p ${SSH_PORT} -o StrictHostKeyChecking=accept-new -o ConnectTimeout=15"
    "${source_dir}/"
    "${SSH_USER}@${DEVICE_HOST}:${destination_dir}/"
  )
  if [[ -n "$SSH_PASSWORD" ]]; then
    sshpass -p "$SSH_PASSWORD" "${rsync_command[@]}"
  else
    "${rsync_command[@]}"
  fi
}

enable_and_check_perf() {
  local perf_events="$(perf_events_for_device)"
  print "==> checking $(device_label) perf events on ${DEVICE_HOST}"
  ssh_run "
    set -e
    printf '%s\n' '$SSH_PASSWORD' | sudo -S sysctl kernel.perf_event_paranoid=-1 >/dev/null
    perf stat -e '$perf_events' -- '$REMOTE_PYTHON' -c 'sum(i*i for i in range(20000000))' >/dev/null
  "
}

check_environment() {
  print "==> checking ${SSH_USER}@${DEVICE_HOST}"
  ssh_run "
    set -e
    test -d '$REMOTE_PROJECT_DIR'
    test -x '$REMOTE_PYTHON'
    test -f '$REMOTE_PROJECT_DIR/running_ml.py'
    test -f '$REMOTE_IID_DATA_DIR/cifar10/partition_metadata.csv'
    test -f '$REMOTE_IID_DATA_DIR/cifar10/augmented/moderate/PREPARED'
    test -f '$REMOTE_IID_DATA_DIR/cifar10/augmented/strong/PREPARED'
    test -f '$REMOTE_IID_DATA_DIR/cifar10/poisoned/availability_shortcuts/shortcut_bank.json'
    cd '$REMOTE_PROJECT_DIR'
    '$REMOTE_PYTHON' --version
    perf --version
    help_text=\$(CUDA_VISIBLE_DEVICES='' '$REMOTE_PYTHON' running_ml.py --help)
    printf '%s\n' \"\$help_text\" | grep -q -- '--perf-fps'
    printf '%s\n' \"\$help_text\" | grep -q -- '--hardware-fps'
    CUDA_VISIBLE_DEVICES='' '$REMOTE_PYTHON' -c \"import torch; from models import get_model; assert not torch.cuda.is_available(); cnn=get_model('simple_cnn',10,(32,32)); vit=get_model('tiny_vit',10,(32,32),model_depth=4); print('device=cpu'); print('simple_cnn_parameters=',sum(p.numel() for p in cnn.parameters())); print('tiny_vit_parameters=',sum(p.numel() for p in vit.parameters()))\"
  "
  enable_and_check_perf
}

run_stage() {
  local spec="$1"
  local -a fields
  fields=("${(@ps:|:)spec}")
  local label="${fields[1]}"
  local poisoning_method="${fields[2]}"
  local augmentation_profile="${fields[3]}"
  local model_name="${fields[4]}"
  local model_depth="${fields[5]}"
  local log_dir="$(remote_log_root)/${label}"
  local perf_events="$(perf_events_for_device)"
  local augment_json="{\"enabled\":true,\"_profile\":\"${augmentation_profile}\",\"resize\":[32,32],\"horizontal_flip\":false,\"normalize\":true}"

  print "==> stage ${label}"
  print "    method=${poisoning_method} profile=${augmentation_profile}"
  print "    model=${model_name} epochs=${LOCAL_EPOCHS} batch_size=${BATCH_SIZE}"
  ssh_run "
    set -e
    cd '$REMOTE_PROJECT_DIR'
    run_marker=\$(mktemp)
    CUDA_VISIBLE_DEVICES='' '$REMOTE_PYTHON' running_ml.py \\
      --experiment-id 'motivational_0727_${label}' \\
      --dataset '$DATASET_NAME' \\
      --dataset-split train \\
      --data-dir '$REMOTE_IID_DATA_DIR' \\
      --log-dir '$log_dir' \\
      --client-id '$CLIENT_ID' \\
      --device-id '$DEVICE_HOST' \\
      --host '$DEVICE_HOST' \\
      --poisoning-method '$poisoning_method' \\
      --partition-method iid \\
      --reference-trials '$REFERENCE_TRIALS' \\
      --trials '$TRIALS' \\
      --local-epochs '$LOCAL_EPOCHS' \\
      --batch-size '$BATCH_SIZE' \\
      --num-rounds '$NUM_ROUNDS' \\
      --learning-rate '$LEARNING_RATE' \\
      --seed '$BASE_SEED' \\
      --perf-events '$perf_events' \\
      --perf-fps '$PERF_FPS' \\
      --hardware-fps '$HARDWARE_FPS' \\
      --augment '$augment_json' \\
      --model '$model_name' \\
      --model-depth '$model_depth' \\
      --model-width-multiplier 1.0 \\
      --model-target-pam-mb 0

    perf_file_count=0
    for perf_file in \$(find '$log_dir' -maxdepth 1 -type f -name '*_perf.csv' -newer \"\$run_marker\" -print); do
      perf_file_count=\$((perf_file_count + 1))
      grep -q ',ok,' \"\$perf_file\" || {
        echo \"perf produced no successful rows: \$perf_file\" >&2
        exit 5
      }
    done
    test \"\$perf_file_count\" -eq '$TRIALS' || {
      echo \"expected $TRIALS perf files for ${label}, found \$perf_file_count\" >&2
      exit 6
    }
    rm -f \"\$run_marker\"
  "
}

run_experiments() {
  print "Running ${#STAGE_SPECS} motivational-study stages sequentially on ${DEVICE_HOST} $(device_label) CPU"
  print "  client=${CLIENT_ID} trials=${TRIALS} epochs=${LOCAL_EPOCHS} batch_size=${BATCH_SIZE}"
  print "  perf_fps=${PERF_FPS} hardware_fps=${HARDWARE_FPS}"
  local spec
  for spec in "${STAGE_SPECS[@]}"; do
    run_stage "$spec"
  done
  print "Motivational experiment completed."
}

collect_logs() {
  local remote_log_root="$(remote_log_root)"
  local local_log_dir="$(local_log_dir)"
  mkdir -p "$local_log_dir"
  local -a rsync_command
  rsync_command=(
    rsync -az --partial --stats --human-readable
    -e "ssh -p ${SSH_PORT} -o StrictHostKeyChecking=accept-new -o ConnectTimeout=15"
    "${SSH_USER}@${DEVICE_HOST}:${REMOTE_PROJECT_DIR}/${remote_log_root}/"
    "${local_log_dir}/"
  )
  print "==> collecting ${DEVICE_HOST} logs into ${local_log_dir}"
  if [[ -n "$SSH_PASSWORD" ]]; then
    sshpass -p "$SSH_PASSWORD" "${rsync_command[@]}"
  else
    "${rsync_command[@]}"
  fi
}

case "$ACTION" in
  pull)
    pull_repository
    ;;
  sync)
    sync_dataset
    ;;
  check)
    check_environment
    ;;
  run)
    check_environment
    run_experiments
    ;;
  collect)
    collect_logs
    ;;
  both)
    pull_repository
    sync_dataset
    check_environment
    run_experiments
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
