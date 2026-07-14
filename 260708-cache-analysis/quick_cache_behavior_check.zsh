#!/usr/bin/env zsh
set -euo pipefail

SCRIPT_DIR="${0:A:h}"
SERVER_REPO_DIR="${SCRIPT_DIR:h}"

SSH_PASSWORD="${SSH_PASSWORD:-modenaottun}"
SSH_PORT="${SSH_PORT:-22}"
DATASET_NAME="${DATASET_NAME:-kuchidareo/small_trashnet}"
DATASET_SLUG="${DATASET_NAME##*/}"
CONDITIONS="${CONDITIONS:-clean,unlearnable_examples,availability_shortcuts}"
TRIALS="${TRIALS:-1}"
REFERENCE_TRIALS="${REFERENCE_TRIALS:-0}"
LOCAL_EPOCHS="${LOCAL_EPOCHS:-10}"
BATCH_SIZE="${BATCH_SIZE:-16}"
MODEL_NAME="${MODEL_NAME:-simple_cnn}"
MODEL_DEPTH="${MODEL_DEPTH:-5}"
LARGE_MODEL_TARGET_PAM_MB="${LARGE_MODEL_TARGET_PAM_MB:-500}"
PAM_CALIBRATION_STEPS="${PAM_CALIBRATION_STEPS:-8}"
REMOTE_PYTHON_REL="${REMOTE_PYTHON_REL:-venv/bin/python}"
REMOTE_PROJECT_NAME="${REMOTE_PROJECT_NAME:-260708-cache-analysis}"
REMOTE_DATA_DIR_NAME="${REMOTE_DATA_DIR_NAME:-iid-data}"
REMOTE_LOG_ROOT="${REMOTE_LOG_ROOT:-logs/cache_quick}"
AUGMENT_JSON="${AUGMENT_JSON:-{\"enabled\":true,\"resize\":[224,224],\"horizontal_flip\":true,\"normalize\":true}}"

ACTION="${1:-both}"

DEVICE_SPECS=(
  "rasheed|192.168.0.112|client_0|/home/rasheed/kuchida/antivenom_infocom"
)

RUN_DEVICE_SPECS=(
  "rasheed|192.168.0.112|client_0|/home/rasheed/kuchida/antivenom_infocom|rpi4"
)

MODEL_SPECS=(
  "${MODEL_NAME}|${MODEL_NAME}|0"
)

usage() {
  cat <<'EOF'
Usage:
  ./quick_cache_behavior_check.zsh [pull|run|both|check]

Default:
  ./quick_cache_behavior_check.zsh both

Environment overrides:
  CONDITIONS="clean,unlearnable_examples,availability_shortcuts"
  DATASET_NAME="uoft-cs/cifar10"
  MODEL_NAME="resnet18"
  AUGMENT_JSON='{"enabled":true,"resize":[224,224],"horizontal_flip":true,"normalize":true}'
  TRIALS=1
  LOCAL_EPOCHS=10
  BATCH_SIZE=16
  MODEL_DEPTH=5
  LARGE_MODEL_TARGET_PAM_MB=500
  PAM_CALIBRATION_STEPS=8
  SSH_PASSWORD=modenaottun

This script:
  1. git pull --rebase on 192.168.0.112
  2. sets kernel.perf_event_paranoid=-1 before training
  3. runs a quick local-ML cache check only on:
       192.168.0.112 as client_0, RPI4
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

check_run_devices() {
  print "Checking Python, perf, project dir, and shared iid-data on run devices..."
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
      if [ ! -d '$data_dir/$DATASET_SLUG' ]; then
        echo 'missing data_dir: $data_dir/$DATASET_SLUG' >&2
        exit 3
      fi
      cd '$project_dir'
      '$remote_python' --version
      perf --version
      '$remote_python' -c \"from dataset_preparation import get_num_classes; print('num_classes', get_num_classes('$data_dir', dataset_name='$DATASET_NAME'))\"
    "
  done
}

run_quick_cache_behavior_check() {
  print "Running quick cache-behavior local ML check..."
  print "  dataset: ${DATASET_NAME}"
  print "  model: ${MODEL_NAME}"
  print "  augment: ${AUGMENT_JSON}"
  print "  conditions: ${CONDITIONS}"
  print "  trials: ${TRIALS}, reference_trials: ${REFERENCE_TRIALS}, local_epochs: ${LOCAL_EPOCHS}"
  print "  perf: sudo sysctl kernel.perf_event_paranoid=-1 before each run"

  local spec model_spec
  for spec in "${RUN_DEVICE_SPECS[@]}"; do
    local -a spec_fields
    spec_fields=("${(@ps:|:)spec}")
    local user="${spec_fields[1]}"
    local host="${spec_fields[2]}"
    local client_id="${spec_fields[3]}"
    local repo_dir="${spec_fields[4]}"
    local device_type="${spec_fields[5]}"
    local project_dir="${repo_dir}/${REMOTE_PROJECT_NAME}"
    local data_dir="${repo_dir}/${REMOTE_DATA_DIR_NAME}"
    local remote_python="${repo_dir}/${REMOTE_PYTHON_REL}"

    for model_spec in "${MODEL_SPECS[@]}"; do
      local -a model_fields
      model_fields=("${(@ps:|:)model_spec}")
      local model_label="${model_fields[1]}"
      local model_name="${model_fields[2]}"
      local target_pam_mb="${model_fields[3]}"
      local log_dir="${REMOTE_LOG_ROOT}/${host}/${model_label}"

      print "==> run ${user}@${host} ${device_type} model=${model_name}"
      ssh_run "$user" "$host" "
        set -e
        printf '%s\n' '$SSH_PASSWORD' | sudo -S sysctl kernel.perf_event_paranoid=-1
        cd '$project_dir'
        '$remote_python' running_ml.py \
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
          --augment '$AUGMENT_JSON' \
          --model '$model_name' \
          --model-depth '$MODEL_DEPTH' \
          --model-target-pam-mb '$target_pam_mb' \
          --model-pam-calibration-steps '$PAM_CALIBRATION_STEPS'
      "
    done
  done
  print "Quick cache-behavior check finished."
}

case "$ACTION" in
  pull)
    pull_all_devices
    ;;
  check)
    check_run_devices
    ;;
  run)
    check_run_devices
    run_quick_cache_behavior_check
    ;;
  both)
    pull_all_devices
    check_run_devices
    run_quick_cache_behavior_check
    ;;
  -h|--help|help)
    usage
    ;;
  *)
    usage >&2
    exit 1
    ;;
esac
