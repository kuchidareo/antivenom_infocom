#!/usr/bin/env zsh
set -euo pipefail

PROJECT_DIR="${0:A:h}"
REPO_DIR="${PROJECT_DIR:h}"
PYTHON="${PYTHON:-${REPO_DIR}/venv/bin/python}"
SSH_USER="${SSH_USER:-rasheed}"
SSH_PASSWORD="${SSH_PASSWORD:-}"
PING_TIMEOUT_SEC="${PING_TIMEOUT_SEC:-1}"

REMOTE_REPO_DIR="${REMOTE_REPO_DIR:-/home/rasheed/kuchida/antivenom_infocom}"
REMOTE_PROJECT_DIR="${REMOTE_PROJECT_DIR:-${REMOTE_REPO_DIR}/260724-modrl-or-input-analysis}"
REMOTE_PYTHON="${REMOTE_PYTHON:-${REMOTE_REPO_DIR}/venv/bin/python}"
REMOTE_DATA_DIR="${REMOTE_DATA_DIR:-${REMOTE_REPO_DIR}/iid-data}"

DATASET="${DATASET:-kuchidareo/small_trashnet}"
TRAIN_CONDITIONS="${TRAIN_CONDITIONS:-clean,availability_shortcuts}"
DATA_CONDITIONS="${DATA_CONDITIONS:-clean,availability_shortcuts}"
MODEL_CONDITIONS="${MODEL_CONDITIONS:-clean,availability_shortcuts}"
MODELS="${MODELS:-simple_cnn,vit_b_16}"
LOCAL_EPOCHS="${LOCAL_EPOCHS:-10}"
BATCH_SIZE="${BATCH_SIZE:-16}"
TRIAL_ID="${TRIAL_ID:-trial_0}"
MAX_BATCHES="${MAX_BATCHES:-0}"
WARMUP_BATCHES="${WARMUP_BATCHES:-1}"
PERF_EVENTS="${PERF_EVENTS:-}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d%H%M%S)}"

device_lines() {
  print -- "client_1:192.168.0.141"
  print -- "client_2:192.168.0.142"
}

host_is_reachable() {
  ping -c 1 -W "$PING_TIMEOUT_SEC" "$1" >/dev/null 2>&1
}

reachable_device_lines() {
  local device host
  for device in "${(@f)$(device_lines)}"; do
    host="${device#*:}"
    if host_is_reachable "$host"; then
      print -- "$device"
    else
      print "Skipping unreachable device: ${host}" >&2
    fi
  done
}

ssh_run() {
  local host="$1"
  local remote_command="$2"
  local target="${SSH_USER}@${host}"
  if [[ -n "$SSH_PASSWORD" ]]; then
    command -v sshpass >/dev/null 2>&1 || {
      print "SSH_PASSWORD is set, but sshpass is not installed." >&2
      return 1
    }
    sshpass -p "$SSH_PASSWORD" ssh -o StrictHostKeyChecking=accept-new \
      "$target" "$remote_command"
  else
    ssh -o StrictHostKeyChecking=accept-new "$target" "$remote_command"
  fi
}

require_reachable_devices() {
  local -a active_devices
  active_devices=("${(@f)$(reachable_device_lines)}")
  (( ${#active_devices[@]} > 0 )) || {
    print "No reachable devices." >&2
    return 1
  }
}

pull_remote_repositories() {
  print "Pulling the latest repository on reachable devices..."
  local device host
  for device in "${(@f)$(reachable_device_lines)}"; do
    host="${device#*:}"
    print "==> git pull --rebase ${host}"
    ssh_run "$host" "cd '$REMOTE_REPO_DIR' && git pull --rebase"
  done
}

check_remote_environment() {
  print "Checking Python, perf, project files, models, and Small TrashNet data..."
  local device host
  require_reachable_devices
  for device in "${(@f)$(reachable_device_lines)}"; do
    host="${device#*:}"
    print "==> check ${host}"
    ssh_run "$host" "
      set -e
      test -d '$REMOTE_PROJECT_DIR'
      test -x '$REMOTE_PYTHON'
      test -f '$REMOTE_PROJECT_DIR/running_ml.py'
      test -f '$REMOTE_PROJECT_DIR/controlled_forward_monitor.py'
      test -f '$REMOTE_DATA_DIR/small_trashnet/partition_metadata.csv'
      command -v perf
      cd '$REMOTE_PROJECT_DIR'
      '$REMOTE_PYTHON' --version
      '$REMOTE_PYTHON' -c 'from models import get_model, get_monitoring_layers; names = [\"simple_cnn\", \"vit_b_16\"]; [get_monitoring_layers(get_model(name, 6), name) for name in names]; print(\"models\", \",\".join(names))'
      '$REMOTE_PYTHON' running_ml.py --help >/dev/null
      '$REMOTE_PYTHON' controlled_forward_monitor.py --help >/dev/null
    "
  done
}

configure_remote_perf() {
  print "Configuring perf_event_paranoid=-1..."
  local device host
  for device in "${(@f)$(reachable_device_lines)}"; do
    host="${device#*:}"
    print "==> perf setup ${host}"
    if [[ -n "$SSH_PASSWORD" ]]; then
      ssh_run "$host" \
        "printf '%s\n' '$SSH_PASSWORD' | sudo -S sysctl kernel.perf_event_paranoid=-1"
    else
      ssh_run "$host" "sudo -n sysctl kernel.perf_event_paranoid=-1"
    fi
  done
}

validate_remote_perf_events() {
  print "Validating host-specific perf event profiles..."
  local device host events
  for device in "${(@f)$(reachable_device_lines)}"; do
    host="${device#*:}"
    if [[ -n "$PERF_EVENTS" ]]; then
      events="$PERF_EVENTS"
    else
      events="$(
        cd "$PROJECT_DIR"
        "$PYTHON" -c "from perf_logger import default_perf_events_for_host; print(','.join(default_perf_events_for_host('$host')))"
      )"
    fi
    print "==> perf events ${host}: ${events}"
    ssh_run "$host" "perf stat -e '$events' -- true >/dev/null"
  done
}

prepare_remote_experiment() {
  pull_remote_repositories
  check_remote_environment
  configure_remote_perf
  validate_remote_perf_events
}

wait_for_jobs() {
  local description="$1"
  shift
  local -a job_specs=("$@")
  local spec pid label failed=0
  for spec in "${job_specs[@]}"; do
    pid="${spec%%:*}"
    label="${spec#*:}"
    if wait "$pid"; then
      print "==> finished ${description}: ${label}"
    else
      print "==> failed ${description}: ${label}" >&2
      failed=1
    fi
  done
  (( failed == 0 ))
}

train_models() {
  local model device client_id host
  local -a jobs
  print "Training and saving final model states:"
  print "  models: ${MODELS}"
  print "  conditions: ${TRAIN_CONDITIONS}"
  print "  dataset: ${DATASET}"
  print "  epochs: ${LOCAL_EPOCHS}"
  print "  batch size: ${BATCH_SIZE}"
  print "  compute device: CPU"

  for model in ${(s:,:)MODELS}; do
    jobs=()
    for device in "${(@f)$(reachable_device_lines)}"; do
      client_id="${device%%:*}"
      host="${device#*:}"
      print "==> train ${host} ${model}"
      ssh_run "$host" "
        set -e
        cd '$REMOTE_PROJECT_DIR'
        CUDA_VISIBLE_DEVICES='' '$REMOTE_PYTHON' running_ml.py \
          --dataset '$DATASET' \
          --data-dir '$REMOTE_DATA_DIR' \
          --log-dir logs/training \
          --model-dir models \
          --client-id '$client_id' \
          --device-id '$host' \
          --host '$host' \
          --model '$model' \
          --local-epochs '$LOCAL_EPOCHS' \
          --batch-size '$BATCH_SIZE' \
          --reference-trials 0 \
          --trials 1 \
          --poisoning-method '$TRAIN_CONDITIONS' \
          --disable-perf \
          --save-trained-models
      " &
      jobs+=("$!:${host}/${model}")
    done
    wait_for_jobs "training" "${jobs[@]}"
  done
}

monitor_models() {
  local model device client_id host perf_option
  local -a jobs data_conditions model_conditions
  perf_option=""
  if [[ -n "$PERF_EVENTS" ]]; then
    perf_option="--perf-events '$PERF_EVENTS'"
  fi
  data_conditions=(${(s:,:)DATA_CONDITIONS})
  model_conditions=(${(s:,:)MODEL_CONDITIONS})

  print "Running controlled forward-only monitoring:"
  print "  models: ${MODELS}"
  print "  data conditions: ${DATA_CONDITIONS}"
  print "  model-state conditions: ${MODEL_CONDITIONS}"
  print "  matrix cells per model: $((${#data_conditions[@]} * ${#model_conditions[@]}))"
  print "  batch size: ${BATCH_SIZE}"
  print "  whole-forward and logical-layer perf: enabled"

  for model in ${(s:,:)MODELS}; do
    jobs=()
    for device in "${(@f)$(reachable_device_lines)}"; do
      client_id="${device%%:*}"
      host="${device#*:}"
      print "==> monitor ${host} ${model}"
      ssh_run "$host" "
        set -e
        cd '$REMOTE_PROJECT_DIR'
        CUDA_VISIBLE_DEVICES='' '$REMOTE_PYTHON' controlled_forward_monitor.py \
          --dataset '$DATASET' \
          --data-dir '$REMOTE_DATA_DIR' \
          --model-dir models \
          --output-dir logs/controlled_forward \
          --client-id '$client_id' \
          --device-id '$host' \
          --host '$host' \
          --model '$model' \
          --trial-id '$TRIAL_ID' \
          --batch-size '$BATCH_SIZE' \
          --data-conditions '$DATA_CONDITIONS' \
          --model-conditions '$MODEL_CONDITIONS' \
          --max-batches '$MAX_BATCHES' \
          --warmup-batches '$WARMUP_BATCHES' \
          --run-id '$RUN_ID' \
          $perf_option
      " &
      jobs+=("$!:${host}/${model}")
    done
    wait_for_jobs "monitoring" "${jobs[@]}"
  done
}

usage() {
  cat <<'EOF'
Usage:
  ./run_experiments.zsh check
  ./run_experiments.zsh train
  ./run_experiments.zsh monitor
  ./run_experiments.zsh run

With no argument, `run` is used. It pulls and validates the remote project,
trains clean and availability-shortcut models, saves final checkpoints, and
then monitors the four input/model-state combinations using forward passes only.

Defaults:
  models: simple_cnn,vit_b_16
  dataset: IID small_trashnet
  batch size: 16
  epochs: 10
  clients: 192.168.0.141 and 192.168.0.142

Useful overrides:
  MODELS=simple_cnn
  LOCAL_EPOCHS=1
  TRAIN_CONDITIONS=clean,availability_shortcuts
  DATA_CONDITIONS=clean,availability_shortcuts
  MODEL_CONDITIONS=clean,availability_shortcuts
  MAX_BATCHES=1
  PERF_EVENTS=cycles,instructions,task-clock
EOF
}

main() {
  local mode="${1:-run}"
  case "$mode" in
    check)
      prepare_remote_experiment
      ;;
    train)
      prepare_remote_experiment
      train_models
      ;;
    monitor)
      prepare_remote_experiment
      monitor_models
      ;;
    run|all)
      prepare_remote_experiment
      train_models
      monitor_models
      ;;
    -h|--help|help)
      usage
      ;;
    *)
      print "Unknown mode: ${mode}" >&2
      usage >&2
      return 1
      ;;
  esac
}

main "$@"
