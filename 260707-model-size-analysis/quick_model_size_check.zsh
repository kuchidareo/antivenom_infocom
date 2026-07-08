#!/usr/bin/env zsh
set -euo pipefail

SCRIPT_DIR="${0:A:h}"
SERVER_REPO_DIR="${SCRIPT_DIR:h}"

SSH_PASSWORD="${SSH_PASSWORD:-modenaottun}"
SSH_PORT="${SSH_PORT:-22}"
DATASET_NAME="${DATASET_NAME:-kuchidareo/small_trashnet}"
CONDITIONS="${CONDITIONS:-clean,availability_shortcuts}"
TRIALS="${TRIALS:-1}"
REFERENCE_TRIALS="${REFERENCE_TRIALS:-0}"
LOCAL_EPOCHS="${LOCAL_EPOCHS:-1}"
BATCH_SIZE="${BATCH_SIZE:-16}"
MODEL_DEPTH="${MODEL_DEPTH:-5}"
RPI4_MODEL_TARGET_PAM_MB="${RPI4_MODEL_TARGET_PAM_MB:-400}"
JETSON_CPU_MODEL_TARGET_PAM_MB="${JETSON_CPU_MODEL_TARGET_PAM_MB:-500}"
PAM_CALIBRATION_STEPS="${PAM_CALIBRATION_STEPS:-8}"
REMOTE_PYTHON_REL="${REMOTE_PYTHON_REL:-venv/bin/python}"
REMOTE_PROJECT_NAME="${REMOTE_PROJECT_NAME:-260707-model-size-analysis}"
REMOTE_DATA_DIR_NAME="${REMOTE_DATA_DIR_NAME:-iid-data}"
REMOTE_LOG_ROOT="${REMOTE_LOG_ROOT:-logs/model_size_quick}"
SYNC_DATA="${SYNC_DATA:-yes}"

ACTION="${1:-both}"

DEVICE_SPECS=(
  "rasheed|192.168.0.112|client_0|/home/rasheed/kuchida/antivenom_infocom"
  "rasheed|192.168.0.113|client_1|/home/rasheed/kuchida/antivenom_infocom"
  "rasheed|192.168.0.114|client_2|/home/rasheed/kuchida/antivenom_infocom"
  "rasheed|192.168.0.115|client_3|/home/rasheed/kuchida/antivenom_infocom"
  "rasheed|192.168.0.116|client_4|/home/rasheed/kuchida/antivenom_infocom"
  "rasheed|192.168.0.117|client_5|/home/rasheed/kuchida/antivenom_infocom"
  "rasheed|192.168.0.118|client_6|/home/rasheed/kuchida/antivenom_infocom"
  "rasheed|192.168.0.119|client_7|/home/rasheed/kuchida/antivenom_infocom"
  "rasheed|192.168.0.120|client_8|/home/rasheed/kuchida/antivenom_infocom"
  "rasheed|192.168.0.121|client_9|/home/rasheed/kuchida/antivenom_infocom"
  "rasheed|192.168.0.141|client_1|/home/rasheed/kuchida/antivenom_infocom"
  "rasheed|192.168.0.142|client_2|/home/rasheed/kuchida/antivenom_infocom"
  "reo|192.168.0.131|client_0|/home/reo/kuchida/antivenom_infocom"
)

RUN_DEVICE_SPECS=(
  "rasheed|192.168.0.112|client_0|/home/rasheed/kuchida/antivenom_infocom|rpi4|${RPI4_MODEL_TARGET_PAM_MB}"
  "rasheed|192.168.0.141|client_1|/home/rasheed/kuchida/antivenom_infocom|jetson_cpu|${JETSON_CPU_MODEL_TARGET_PAM_MB}"
)

usage() {
  cat <<'EOF'
Usage:
  ./quick_model_size_check.zsh [pull|run|both|check]

Default:
  ./quick_model_size_check.zsh both

Environment overrides:
  CONDITIONS="clean,availability_shortcuts"
  TRIALS=1
  LOCAL_EPOCHS=1
  BATCH_SIZE=16
  MODEL_DEPTH=5
  RPI4_MODEL_TARGET_PAM_MB=400
  JETSON_CPU_MODEL_TARGET_PAM_MB=500
  PAM_CALIBRATION_STEPS=8
  SSH_PASSWORD=modenaottun
  SYNC_DATA=yes

This script:
  1. git pull --rebase on 192.168.0.112-121, 141, 142, 131
  2. rsyncs <repo>/iid-data to the run devices when SYNC_DATA=yes
  3. runs a quick local-ML check only on:
       192.168.0.112 as client_0, RPI4
       192.168.0.141 as client_1, Jetson-CPU
  4. uses data from:
       <repo>/iid-data
EOF
}

ssh_base_cmd() {
  if [[ -n "$SSH_PASSWORD" ]]; then
    if ! command -v sshpass >/dev/null 2>&1; then
      print "SSH_PASSWORD is set, but sshpass is not installed." >&2
      print "Install sshpass or configure SSH keys." >&2
      exit 1
    fi
    print -- "sshpass -p ${(q)SSH_PASSWORD} ssh -p ${(q)SSH_PORT} -o StrictHostKeyChecking=accept-new -o ConnectTimeout=15 -o ServerAliveInterval=10 -o ServerAliveCountMax=3"
  else
    print -- "ssh -p ${(q)SSH_PORT} -o StrictHostKeyChecking=accept-new -o ConnectTimeout=15 -o ServerAliveInterval=10 -o ServerAliveCountMax=3"
  fi
}

rsync_rsh() {
  if [[ -n "$SSH_PASSWORD" ]]; then
    if ! command -v sshpass >/dev/null 2>&1; then
      print "SSH_PASSWORD is set, but sshpass is not installed." >&2
      print "Install sshpass or configure SSH keys." >&2
      exit 1
    fi
    print -- "sshpass -p ${SSH_PASSWORD} ssh -p ${SSH_PORT} -o StrictHostKeyChecking=accept-new -o ConnectTimeout=15"
  else
    print -- "ssh -p ${SSH_PORT} -o StrictHostKeyChecking=accept-new -o ConnectTimeout=15"
  fi
}

ssh_run() {
  local user="$1"
  local host="$2"
  local remote_command="$3"
  local ssh_cmd
  ssh_cmd="$(ssh_base_cmd)"
  eval "$ssh_cmd ${(q)user}@${(q)host} ${(q)remote_command}"
}

pull_all_devices() {
  print "Pulling latest repo on all devices..."
  local spec
  for spec in "${DEVICE_SPECS[@]}"; do
    local -a spec_fields
    spec_fields=("${(@ps:|:)spec}")
    local user="${spec_fields[1]}"
    local host="${spec_fields[2]}"
    local repo_dir="${spec_fields[4]}"
    print "==> git pull --rebase ${user}@${host}:${repo_dir}"
    ssh_run "$user" "$host" "cd '$repo_dir' && git pull --rebase"
  done
}

sync_data_to_run_devices() {
  if [[ "$SYNC_DATA" != "yes" ]]; then
    print "Skipping data sync because SYNC_DATA=${SYNC_DATA}."
    return
  fi
  if ! command -v rsync >/dev/null 2>&1; then
    print "rsync is required but was not found." >&2
    exit 1
  fi

  local local_data_dir="${SERVER_REPO_DIR}/${REMOTE_DATA_DIR_NAME}"
  if [[ ! -d "$local_data_dir/small_trashnet" ]]; then
    print "Local shared data directory is missing: ${local_data_dir}/small_trashnet" >&2
    exit 1
  fi

  print "Syncing shared data to run devices..."
  local spec
  for spec in "${RUN_DEVICE_SPECS[@]}"; do
    local -a spec_fields
    spec_fields=("${(@ps:|:)spec}")
    local user="${spec_fields[1]}"
    local host="${spec_fields[2]}"
    local repo_dir="${spec_fields[4]}"
    local remote_data_dir="${repo_dir}/${REMOTE_DATA_DIR_NAME}"
    local remote="${user}@${host}"
    print "==> rsync ${local_data_dir}/ -> ${remote}:${remote_data_dir}/"
    ssh_run "$user" "$host" "mkdir -p '$remote_data_dir'"
    rsync -az --delete -e "$(rsync_rsh)" "${local_data_dir}/" "${remote}:${remote_data_dir}/"
  done
}

check_run_devices() {
  print "Checking Python, project dir, and shared iid-data on run devices..."
  local spec
  for spec in "${RUN_DEVICE_SPECS[@]}"; do
    local -a spec_fields
    spec_fields=("${(@ps:|:)spec}")
    local user="${spec_fields[1]}"
    local host="${spec_fields[2]}"
    local repo_dir="${spec_fields[4]}"
    local project_dir="${repo_dir}/${REMOTE_PROJECT_NAME}"
    local data_dir="${repo_dir}/${REMOTE_DATA_DIR_NAME}"
    local remote_python="${repo_dir}/${REMOTE_PYTHON_REL}"
    print "==> check ${user}@${host}"
    ssh_run "$user" "$host" "
      set -e
      if [ ! -d '$project_dir' ]; then
        echo 'missing project_dir: $project_dir' >&2
        exit 2
      fi
      if [ ! -d '$data_dir/small_trashnet' ]; then
        echo 'missing shared data: $data_dir/small_trashnet' >&2
        echo 'run: SYNC_DATA=yes ./quick_model_size_check.zsh run' >&2
        exit 3
      fi
      cd '$project_dir'
      '$remote_python' --version
      '$remote_python' -c \"from dataset_preparation import get_num_classes; print('num_classes', get_num_classes('$data_dir', dataset_name='$DATASET_NAME'))\"
    "
  done
}

run_one_device_model_size_check() {
  local spec="$1"
  local -a spec_fields
  spec_fields=("${(@ps:|:)spec}")
  local user="${spec_fields[1]}"
  local host="${spec_fields[2]}"
  local client_id="${spec_fields[3]}"
  local repo_dir="${spec_fields[4]}"
  local device_type="${spec_fields[5]}"
  local device_target_pam_mb="${spec_fields[6]}"
  local project_dir="${repo_dir}/${REMOTE_PROJECT_NAME}"
  local data_dir="${repo_dir}/${REMOTE_DATA_DIR_NAME}"
  local remote_python="${repo_dir}/${REMOTE_PYTHON_REL}"
  local -a model_specs=(
    "simple|simple_cnn|0"
    "pam${device_target_pam_mb}mb|pam_cnn|${device_target_pam_mb}"
  )
  local model_spec

  print "==> device start ${user}@${host} ${device_type} pam_target=${device_target_pam_mb}MB"
  for model_spec in "${model_specs[@]}"; do
    local -a model_fields
    model_fields=("${(@ps:|:)model_spec}")
    local model_label="${model_fields[1]}"
    local model_name="${model_fields[2]}"
    local target_pam_mb="${model_fields[3]}"
    local log_dir="${REMOTE_LOG_ROOT}/${host}/${model_label}"

    print "==> run ${user}@${host} ${device_type} model=${model_name} target_pam=${target_pam_mb}"
    ssh_run "$user" "$host" "
      set -e
      cd '$project_dir'
      CUDA_VISIBLE_DEVICES='' '$remote_python' running_ml.py \
        --dataset '$DATASET_NAME' \
        --data-dir '$data_dir' \
        --log-dir '$log_dir' \
        --client-id '$client_id' \
        --device-id '$host' \
        --host '$host' \
        --poisoning-method '$CONDITIONS' \
        --reference-trials '$REFERENCE_TRIALS' \
        --trials '$TRIALS' \
        --local-epochs '$LOCAL_EPOCHS' \
        --batch-size '$BATCH_SIZE' \
        --model '$model_name' \
        --model-depth '$MODEL_DEPTH' \
        --model-target-pam-mb '$target_pam_mb' \
        --model-pam-calibration-steps '$PAM_CALIBRATION_STEPS'
    "
  done
  print "==> device finished ${user}@${host} ${device_type}"
}

run_quick_model_size_check() {
  print "Running quick model-size local ML check..."
  print "  dataset: ${DATASET_NAME}"
  print "  conditions: ${CONDITIONS}"
  print "  trials: ${TRIALS}, reference_trials: ${REFERENCE_TRIALS}, local_epochs: ${LOCAL_EPOCHS}"
  print "  models:"
  print "    192.168.0.112 rpi4: simple_cnn and pam_cnn target ${RPI4_MODEL_TARGET_PAM_MB} MB"
  print "    192.168.0.141 jetson_cpu: simple_cnn and pam_cnn target ${JETSON_CPU_MODEL_TARGET_PAM_MB} MB"
  print "  parallelism: one worker per run device; each device runs its models/conditions sequentially"

  local spec
  local -a pids=()
  for spec in "${RUN_DEVICE_SPECS[@]}"; do
    run_one_device_model_size_check "$spec" &
    pids+=("$!")
  done

  local exit_code=0
  local pid
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
      exit_code=1
    fi
  done
  if (( exit_code != 0 )); then
    print "One or more model-size device runs failed." >&2
    return "$exit_code"
  fi
  print "Quick model-size check finished."
}

case "$ACTION" in
  pull)
    pull_all_devices
    ;;
  check)
    check_run_devices
    ;;
  run)
    sync_data_to_run_devices
    check_run_devices
    run_quick_model_size_check
    ;;
  both)
    pull_all_devices
    sync_data_to_run_devices
    check_run_devices
    run_quick_model_size_check
    ;;
  -h|--help|help)
    usage
    ;;
  *)
    usage >&2
    exit 1
    ;;
esac
