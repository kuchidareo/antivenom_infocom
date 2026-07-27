#!/usr/bin/env zsh
set -euo pipefail

SCRIPT_DIR="${0:A:h}"
LOCAL_REPO_DIR="${SCRIPT_DIR:h}"

SSH_USER="${SSH_USER:-rasheed}"
SSH_PASSWORD="${SSH_PASSWORD:-modenaottun}"
SSH_PORT="${SSH_PORT:-22}"
JETSON_HOST="${JETSON_HOST:-192.168.0.141}"
REMOTE_REPO_DIR="${REMOTE_REPO_DIR:-/home/rasheed/kuchida/antivenom_infocom}"
REMOTE_PROJECT_DIR="${REMOTE_PROJECT_DIR:-${REMOTE_REPO_DIR}/260708-cache-analysis}"
REMOTE_PYTHON="${REMOTE_PYTHON:-${REMOTE_REPO_DIR}/venv/bin/python}"
REMOTE_IID_DATA_DIR="${REMOTE_IID_DATA_DIR:-${REMOTE_REPO_DIR}/iid-data}"
REMOTE_NONIID_DATA_DIR="${REMOTE_NONIID_DATA_DIR:-${REMOTE_REPO_DIR}/non-iid-data}"
LOCAL_IID_DATA_DIR="${LOCAL_IID_DATA_DIR:-${LOCAL_REPO_DIR}/iid-data}"
LOCAL_NONIID_DATA_DIR="${LOCAL_NONIID_DATA_DIR:-${LOCAL_REPO_DIR}/non-iid-data}"
REMOTE_LOG_ROOT="${REMOTE_LOG_ROOT:-logs/cache_0727_jetson_cpu/${JETSON_HOST}}"
LOCAL_LOG_DIR="${LOCAL_LOG_DIR:-${SCRIPT_DIR}/collected_logs/logs_0727_jetson_cpu}"

MONITORING_FPS="${MONITORING_FPS:-50}"
PERF_FPS="${PERF_FPS:-$MONITORING_FPS}"
HARDWARE_FPS="${HARDWARE_FPS:-$MONITORING_FPS}"
TRIALS="${TRIALS:-1}"
LOCAL_EPOCHS="${LOCAL_EPOCHS:-10}"
BATCH_SIZE="${BATCH_SIZE:-16}"
BASE_SEED="${BASE_SEED:-260727}"
LEARNING_RATE="${LEARNING_RATE:-0.001}"

DATASET_NAME="uoft-cs/cifar10"
CLIENT_ID="client_1"
REFERENCE_TRIALS=0
NUM_ROUNDS=10
NONIID_ALPHA=0.3

JETSON_PERF_EVENTS="${JETSON_PERF_EVENTS:-cycles,instructions,task-clock,context-switches,cpu-migrations,page-faults,br_retired,br_mis_pred_retired,l1d_cache,l1d_cache_refill,l1d_cache_wb,l2d_cache,l2d_cache_refill,l2d_cache_wb,bus_access,mem_access,inst_spec}"

# label|data_dir|poisoning_method|augmentation_profile|model|partition_method|model_depth
STAGE_SPECS=(
  "cifar10_iid|${REMOTE_IID_DATA_DIR}|clean|baseline|simple_cnn|iid|3"
  "cifar10_availability_shortcut|${REMOTE_IID_DATA_DIR}|availability_shortcuts|baseline|simple_cnn|iid|3"
  "cifar10_moderate_augmentation|${REMOTE_IID_DATA_DIR}|clean|moderate|simple_cnn|iid|3"
  "cifar10_strong_augmentation|${REMOTE_IID_DATA_DIR}|clean|strong|simple_cnn|iid|3"
  "cifar10_non_iid|${REMOTE_NONIID_DATA_DIR}|clean|baseline|simple_cnn|dirichlet_noniid|3"
  "cifar10_vit|${REMOTE_IID_DATA_DIR}|clean|baseline|tiny_vit|iid|4"
)

ACTION="${1:-both}"

usage() {
  cat <<'EOF'
Usage:
  ./0727_experiment.zsh [pull|sync|check|run|collect|both]

Default `both` flow:
  git pull -> sync IID/non-IID CIFAR-10 -> check -> run -> collect

The six sequential CPU-only stages on 192.168.0.141 are:
  1. CIFAR-10 IID, clean, SimpleCNN
  2. CIFAR-10 IID, availability shortcut, SimpleCNN
  3. CIFAR-10 IID, saved moderate augmentation, SimpleCNN
  4. CIFAR-10 IID, saved strong augmentation, SimpleCNN
  5. CIFAR-10 Dirichlet non-IID (alpha 0.3), clean, SimpleCNN
  6. CIFAR-10 IID, clean, TinyViT

Defaults:
  client partition: client_1
  trials:           1
  epochs:           10
  batch size:       16
  learning rate:    0.001
  perf/psutil:      50 Hz
  compute:          CPU (CUDA_VISIBLE_DEVICES is empty)
EOF
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
  eval "$ssh_cmd ${(q)SSH_USER}@${(q)JETSON_HOST} ${(q)remote_command}"
}

pull_repository() {
  print "==> git pull --rebase ${SSH_USER}@${JETSON_HOST}:${REMOTE_REPO_DIR}"
  ssh_run "cd '$REMOTE_REPO_DIR' && git pull --rebase"
}

sync_dataset_root() {
  local local_data_dir="$1"
  local remote_data_dir="$2"
  local source_dir="${local_data_dir}/cifar10"
  local destination_dir="${remote_data_dir}/cifar10"
  local -a rsync_command

  test -f "${source_dir}/partition_metadata.csv" || {
    print "Missing local CIFAR-10 metadata: ${source_dir}/partition_metadata.csv" >&2
    return 2
  }
  ssh_run "mkdir -p '$destination_dir'"
  print "==> rsync ${source_dir} -> ${SSH_USER}@${JETSON_HOST}:${destination_dir}"
  rsync_command=(
    rsync -az --partial --stats --human-readable
    -e "ssh -p ${SSH_PORT} -o StrictHostKeyChecking=accept-new -o ConnectTimeout=15"
    "${source_dir}/"
    "${SSH_USER}@${JETSON_HOST}:${destination_dir}/"
  )
  if [[ -n "$SSH_PASSWORD" ]]; then
    sshpass -p "$SSH_PASSWORD" "${rsync_command[@]}"
  else
    "${rsync_command[@]}"
  fi
}

sync_datasets() {
  sync_dataset_root "$LOCAL_IID_DATA_DIR" "$REMOTE_IID_DATA_DIR"
  sync_dataset_root "$LOCAL_NONIID_DATA_DIR" "$REMOTE_NONIID_DATA_DIR"
}

enable_and_check_perf() {
  print "==> checking Jetson perf events"
  ssh_run "
    set -e
    printf '%s\n' '$SSH_PASSWORD' | sudo -S sysctl kernel.perf_event_paranoid=-1 >/dev/null
    perf stat -e '$JETSON_PERF_EVENTS' -- '$REMOTE_PYTHON' -c 'sum(i*i for i in range(20000000))' >/dev/null
  "
}

check_environment() {
  print "==> checking ${SSH_USER}@${JETSON_HOST}"
  ssh_run "
    set -e
    test -d '$REMOTE_PROJECT_DIR'
    test -x '$REMOTE_PYTHON'
    test -f '$REMOTE_IID_DATA_DIR/cifar10/partition_metadata.csv'
    test -f '$REMOTE_NONIID_DATA_DIR/cifar10/partition_metadata.csv'
    test -f '$REMOTE_IID_DATA_DIR/cifar10/augmented/moderate/PREPARED'
    test -f '$REMOTE_IID_DATA_DIR/cifar10/augmented/strong/PREPARED'
    test -f '$REMOTE_IID_DATA_DIR/cifar10/poisoned/availability_shortcuts/shortcut_bank.json'
    cd '$REMOTE_PROJECT_DIR'
    '$REMOTE_PYTHON' --version
    perf --version
    help_text=\$(CUDA_VISIBLE_DEVICES='' '$REMOTE_PYTHON' running_ml.py --help)
    printf '%s\n' \"\$help_text\" | grep -q -- '--partition-method'
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
  local data_dir="${fields[2]}"
  local poisoning_method="${fields[3]}"
  local augmentation_profile="${fields[4]}"
  local model_name="${fields[5]}"
  local partition_method="${fields[6]}"
  local model_depth="${fields[7]}"
  local log_dir="${REMOTE_LOG_ROOT}/${label}"
  local augment_json="{\"enabled\":true,\"_profile\":\"${augmentation_profile}\",\"resize\":[32,32],\"horizontal_flip\":false,\"normalize\":true}"

  print "==> stage ${label}"
  print "    data=${data_dir} method=${poisoning_method} profile=${augmentation_profile}"
  print "    model=${model_name} partition=${partition_method} epochs=${LOCAL_EPOCHS}"
  ssh_run "
    set -e
    cd '$REMOTE_PROJECT_DIR'
    run_marker=\$(mktemp)
    CUDA_VISIBLE_DEVICES='' '$REMOTE_PYTHON' running_ml.py \\
      --experiment-id 'cache_0727_${label}' \\
      --dataset '$DATASET_NAME' \\
      --dataset-split train \\
      --data-dir '$data_dir' \\
      --log-dir '$log_dir' \\
      --client-id '$CLIENT_ID' \\
      --device-id '$JETSON_HOST' \\
      --host '$JETSON_HOST' \\
      --poisoning-method '$poisoning_method' \\
      --partition-method '$partition_method' \\
      --noniid-alpha '$NONIID_ALPHA' \\
      --reference-trials '$REFERENCE_TRIALS' \\
      --trials '$TRIALS' \\
      --local-epochs '$LOCAL_EPOCHS' \\
      --batch-size '$BATCH_SIZE' \\
      --num-rounds '$NUM_ROUNDS' \\
      --learning-rate '$LEARNING_RATE' \\
      --seed '$BASE_SEED' \\
      --perf-events '$JETSON_PERF_EVENTS' \\
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
  print "Running six CIFAR-10 stages sequentially on ${JETSON_HOST} CPU"
  print "  client=${CLIENT_ID} trials=${TRIALS} epochs=${LOCAL_EPOCHS} batch_size=${BATCH_SIZE}"
  print "  perf_fps=${PERF_FPS} hardware_fps=${HARDWARE_FPS}"
  local spec
  for spec in "${STAGE_SPECS[@]}"; do
    run_stage "$spec"
  done
  print "0727 experiment completed."
}

collect_logs() {
  mkdir -p "$LOCAL_LOG_DIR"
  local -a rsync_command
  rsync_command=(
    rsync -az --partial --stats --human-readable
    -e "ssh -p ${SSH_PORT} -o StrictHostKeyChecking=accept-new -o ConnectTimeout=15"
    "${SSH_USER}@${JETSON_HOST}:${REMOTE_PROJECT_DIR}/${REMOTE_LOG_ROOT}/"
    "${LOCAL_LOG_DIR}/"
  )
  print "==> collecting logs into ${LOCAL_LOG_DIR}"
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
    sync_datasets
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
    sync_datasets
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
