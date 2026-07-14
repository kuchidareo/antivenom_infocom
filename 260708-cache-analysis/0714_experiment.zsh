#!/usr/bin/env zsh
set -euo pipefail

SCRIPT_DIR="${0:A:h}"

SSH_PASSWORD="${SSH_PASSWORD:-modenaottun}"
SSH_PORT="${SSH_PORT:-22}"
CONDITIONS="${CONDITIONS:-clean,unlearnable_examples,availability_shortcuts,random_label_flipping}"
TRIALS="${TRIALS:-5}"
REFERENCE_TRIALS="${REFERENCE_TRIALS:-0}"
LOCAL_EPOCHS="${LOCAL_EPOCHS:-15}"
BATCH_SIZE="${BATCH_SIZE:-16}"
BASE_SEED="${BASE_SEED:-260714}"
MODEL_DEPTH="${MODEL_DEPTH:-5}"
PAM_CALIBRATION_STEPS="${PAM_CALIBRATION_STEPS:-8}"
REMOTE_PYTHON_REL="${REMOTE_PYTHON_REL:-venv/bin/python}"
REMOTE_PROJECT_NAME="${REMOTE_PROJECT_NAME:-260708-cache-analysis}"
REMOTE_DATA_DIR_NAME="${REMOTE_DATA_DIR_NAME:-iid-data}"
REMOTE_LOG_ROOT="${REMOTE_LOG_ROOT:-logs/cache_0714}"
LOCAL_DATA_DIR="${LOCAL_DATA_DIR:-${SCRIPT_DIR:h}/iid-data}"
AUGMENT_JSON="${AUGMENT_JSON:-{\"enabled\":true,\"resize\":[224,224],\"horizontal_flip\":true,\"normalize\":true}}"

ACTION="${1:-both}"

# user|host|client_id|repo_dir|device_type
DEVICE_SPECS=(
  "rasheed|192.168.0.112|client_0|/home/rasheed/kuchida/antivenom_infocom|rpi4"
  "rasheed|192.168.0.114|client_2|/home/rasheed/kuchida/antivenom_infocom|rpi4"
  "rasheed|192.168.0.115|client_3|/home/rasheed/kuchida/antivenom_infocom|rpi3"
  "rasheed|192.168.0.141|client_1|/home/rasheed/kuchida/antivenom_infocom|jetson_cpu"
)

# label|Hugging Face repository
DATASET_SPECS=(
  "small_trashnet|kuchidareo/small_trashnet"
  "cifar10|uoft-cs/cifar10"
)

# label|model factory name
MODEL_SPECS=(
  "simple_cnn|simple_cnn"
  "resnet18|resnet18"
)

usage() {
  cat <<'EOF'
Usage:
  ./0714_experiment.zsh [pull|sync|check|run|both]

Default:
  ./0714_experiment.zsh both

Experiment grid on each device:
  datasets: small_trashnet, cifar10
  models:   simple_cnn, resnet18
  total:    4 dataset/model combinations per device

Devices run in parallel. The four dataset/model combinations run sequentially
inside each device.

The default "both" action performs:
  git pull -> rsync both datasets -> environment check -> experiment

Default experiment settings:
  CONDITIONS="clean,unlearnable_examples,availability_shortcuts,random_label_flipping"
  TRIALS=5
  REFERENCE_TRIALS=0
  LOCAL_EPOCHS=15
  BATCH_SIZE=16
  AUGMENT_JSON='{"enabled":true,"resize":[224,224],"horizontal_flip":true,"normalize":true}'

Environment variables above can be overridden when launching the script.
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

rsync_one_dataset() {
  local user="$1"
  local host="$2"
  local repo_dir="$3"
  local dataset_slug="$4"
  local source_dir="${LOCAL_DATA_DIR}/${dataset_slug}"
  local destination_dir="${repo_dir}/${REMOTE_DATA_DIR_NAME}/${dataset_slug}"
  local -a rsync_command

  if [[ ! -d "$source_dir" ]]; then
    print "Missing local dataset: ${source_dir}" >&2
    return 2
  fi

  ssh_run "$user" "$host" "mkdir -p '$destination_dir'"
  print "==> rsync ${dataset_slug} -> ${user}@${host}:${destination_dir}"

  rsync_command=(
    rsync -az --partial --stats --human-readable
    -e "ssh -p ${SSH_PORT} -o StrictHostKeyChecking=accept-new -o ConnectTimeout=15 -o ServerAliveInterval=10 -o ServerAliveCountMax=3"
    "${source_dir}/"
    "${user}@${host}:${destination_dir}/"
  )
  if [[ -n "$SSH_PASSWORD" ]]; then
    sshpass -p "$SSH_PASSWORD" "${rsync_command[@]}"
  else
    "${rsync_command[@]}"
  fi
}

sync_all_datasets() {
  print "Synchronizing small_trashnet and cifar10 to experiment devices..."
  local spec dataset_spec
  for spec in "${DEVICE_SPECS[@]}"; do
    local -a fields
    fields=("${(@ps:|:)spec}")
    local user="${fields[1]}"
    local host="${fields[2]}"
    local repo_dir="${fields[4]}"

    for dataset_spec in "${DATASET_SPECS[@]}"; do
      local -a dataset_fields
      dataset_fields=("${(@ps:|:)dataset_spec}")
      rsync_one_dataset "$user" "$host" "$repo_dir" "${dataset_fields[1]}"
    done
  done
}

pull_all_devices() {
  print "Pulling latest repo on experiment devices..."
  local spec
  for spec in "${DEVICE_SPECS[@]}"; do
    local -a fields
    fields=("${(@ps:|:)spec}")
    local user="${fields[1]}"
    local host="${fields[2]}"
    local repo_dir="${fields[4]}"
    print "==> git pull --rebase ${user}@${host}:${repo_dir}"
    ssh_run "$user" "$host" "cd '$repo_dir' && git pull --rebase"
  done
}

check_one_device() {
  local spec="$1"
  local -a fields
  fields=("${(@ps:|:)spec}")
  local user="${fields[1]}"
  local host="${fields[2]}"
  local repo_dir="${fields[4]}"
  local project_dir="${repo_dir}/${REMOTE_PROJECT_NAME}"
  local data_dir="${repo_dir}/${REMOTE_DATA_DIR_NAME}"
  local remote_python="${repo_dir}/${REMOTE_PYTHON_REL}"

  print "==> check ${user}@${host}"
  ssh_run "$user" "$host" "
    set -e
    test -d '$project_dir' || { echo 'missing project_dir: $project_dir' >&2; exit 2; }
    test -d '$data_dir/small_trashnet' || { echo 'missing dataset: $data_dir/small_trashnet' >&2; exit 3; }
    test -d '$data_dir/cifar10' || { echo 'missing dataset: $data_dir/cifar10' >&2; exit 3; }
    cd '$project_dir'
    '$remote_python' --version
    perf --version
    '$remote_python' -c \"import torch, torchvision; from dataset_preparation import get_num_classes; from models import get_model; print('torch', torch.__version__, 'torchvision', torchvision.__version__); print('small_trashnet classes', get_num_classes('$data_dir', 'kuchidareo/small_trashnet')); print('cifar10 classes', get_num_classes('$data_dir', 'uoft-cs/cifar10')); print('resnet18 parameters', sum(p.numel() for p in get_model('resnet18', 10, (224, 224)).parameters()))\"
  "
}

check_all_devices() {
  print "Checking Python, torchvision, perf, project directory, and both datasets..."
  local spec
  for spec in "${DEVICE_SPECS[@]}"; do
    check_one_device "$spec"
  done
}

run_device_grid() {
  local spec="$1"
  local -a fields
  fields=("${(@ps:|:)spec}")
  local user="${fields[1]}"
  local host="${fields[2]}"
  local client_id="${fields[3]}"
  local repo_dir="${fields[4]}"
  local device_type="${fields[5]}"
  local project_dir="${repo_dir}/${REMOTE_PROJECT_NAME}"
  local data_dir="${repo_dir}/${REMOTE_DATA_DIR_NAME}"
  local remote_python="${repo_dir}/${REMOTE_PYTHON_REL}"

  print "==> device grid start ${user}@${host} (${device_type}, ${client_id})"
  ssh_run "$user" "$host" "printf '%s\n' '$SSH_PASSWORD' | sudo -S sysctl kernel.perf_event_paranoid=-1"

  local dataset_spec model_spec
  for dataset_spec in "${DATASET_SPECS[@]}"; do
    local -a dataset_fields
    dataset_fields=("${(@ps:|:)dataset_spec}")
    local dataset_label="${dataset_fields[1]}"
    local dataset_name="${dataset_fields[2]}"

    for model_spec in "${MODEL_SPECS[@]}"; do
      local -a model_fields
      model_fields=("${(@ps:|:)model_spec}")
      local model_label="${model_fields[1]}"
      local model_name="${model_fields[2]}"
      local log_dir="${REMOTE_LOG_ROOT}/${host}/${dataset_label}/${model_label}"

      print "==> run ${host} dataset=${dataset_name} model=${model_name} epochs=${LOCAL_EPOCHS} trials=${TRIALS}"
      ssh_run "$user" "$host" "
        set -e
        cd '$project_dir'
        '$remote_python' running_ml.py \\
          --dataset '$dataset_name' \\
          --data-dir '$data_dir' \\
          --log-dir '$log_dir' \\
          --client-id '$client_id' \\
          --device-id '$host' \\
          --host '$host' \\
          --poisoning-method '$CONDITIONS' \\
          --reference-trials '$REFERENCE_TRIALS' \\
          --trials '$TRIALS' \\
          --local-epochs '$LOCAL_EPOCHS' \\
          --batch-size '$BATCH_SIZE' \\
          --seed '$BASE_SEED' \\
          --augment '$AUGMENT_JSON' \\
          --model '$model_name' \\
          --model-depth '$MODEL_DEPTH' \\
          --model-target-pam-mb 0 \\
          --model-pam-calibration-steps '$PAM_CALIBRATION_STEPS'
      "
    done
  done

  print "==> device grid finished ${host}"
}

run_all_devices_parallel() {
  print "Running the 2-dataset x 2-model grid on four devices..."
  print "  conditions: ${CONDITIONS}"
  print "  epochs: ${LOCAL_EPOCHS}, trials: ${TRIALS}, reference_trials: ${REFERENCE_TRIALS}"
  print "  logs: ${REMOTE_LOG_ROOT}/<host>/<dataset>/<model>"

  local -a job_pids job_labels
  local spec
  for spec in "${DEVICE_SPECS[@]}"; do
    local -a fields
    fields=("${(@ps:|:)spec}")
    run_device_grid "$spec" &
    job_pids+=("$!")
    job_labels+=("${fields[2]}")
  done

  local failed=0
  local idx
  for (( idx = 1; idx <= ${#job_pids}; idx++ )); do
    if wait "${job_pids[$idx]}"; then
      print "==> completed ${job_labels[$idx]}"
    else
      local exit_code=$?
      print "==> FAILED ${job_labels[$idx]} (exit ${exit_code})" >&2
      failed=1
    fi
  done

  if (( failed )); then
    print "One or more device grids failed." >&2
    return 1
  fi
  print "0714 experiment grid finished on all devices."
}

case "$ACTION" in
  pull)
    pull_all_devices
    ;;
  sync)
    sync_all_datasets
    ;;
  check)
    check_all_devices
    ;;
  run)
    check_all_devices
    run_all_devices_parallel
    ;;
  both)
    pull_all_devices
    sync_all_datasets
    check_all_devices
    run_all_devices_parallel
    ;;
  -h|--help|help)
    usage
    ;;
  *)
    usage >&2
    exit 1
    ;;
esac
