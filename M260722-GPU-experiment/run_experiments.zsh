#!/usr/bin/env zsh
set -euo pipefail

PROJECT_DIR="${0:A:h}"
REPO_DIR="${PROJECT_DIR:h}"
PYTHON="${PYTHON:-${REPO_DIR}/venv/bin/python}"
SSH_USER="${SSH_USER:-rasheed}"
SSH_PASSWORD="${SSH_PASSWORD:-}"
PING_TIMEOUT_SEC="${PING_TIMEOUT_SEC:-1}"

REMOTE_REPO_DIR="${REMOTE_REPO_DIR:-/home/rasheed/kuchida/antivenom_infocom}"
REMOTE_PROJECT_DIR="${REMOTE_PROJECT_DIR:-${REMOTE_REPO_DIR}/M260722-GPU-experiment}"
REMOTE_PYTHON="${REMOTE_PYTHON:-${REMOTE_REPO_DIR}/venv/bin/python}"
REMOTE_DATA_DIR="${REMOTE_DATA_DIR:-${REMOTE_REPO_DIR}/iid-data}"

DATASET="kuchidareo/small_trashnet"
METHODS="clean,availability_shortcuts"
LOCAL_EPOCHS=10
TRIALS=1
REFERENCE_TRIALS=0
BATCH_SIZE=16

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
      print "Skipping unreachable GPU device: ${host}" >&2
    fi
  done
}

ssh_run() {
  local host="$1"
  local command="$2"
  local target="${SSH_USER}@${host}"
  if [[ -n "$SSH_PASSWORD" ]]; then
    command -v sshpass >/dev/null 2>&1 || {
      print "SSH_PASSWORD is set, but sshpass is not installed." >&2
      return 1
    }
    sshpass -p "$SSH_PASSWORD" ssh -o StrictHostKeyChecking=accept-new "$target" "$command"
  else
    ssh -o StrictHostKeyChecking=accept-new "$target" "$command"
  fi
}

pull_remote_repositories() {
  print "Pulling the latest repository on reachable GPU devices..."
  local device host
  for device in "${(@f)$(reachable_device_lines)}"; do
    host="${device#*:}"
    print "==> git pull --rebase ${host}"
    ssh_run "$host" "cd '$REMOTE_REPO_DIR' && git pull --rebase"
  done
}

check_remote_environment() {
  print "Checking CUDA, Python, project files, and Small TrashNet data..."
  local -a active_devices
  local device host
  active_devices=("${(@f)$(reachable_device_lines)}")
  (( ${#active_devices[@]} > 0 )) || {
    print "No reachable GPU devices." >&2
    return 1
  }

  for device in "${active_devices[@]}"; do
    host="${device#*:}"
    print "==> check ${host}"
    ssh_run "$host" "
      set -e
      test -d '$REMOTE_PROJECT_DIR'
      test -x '$REMOTE_PYTHON'
      test -f '$REMOTE_PROJECT_DIR/running_ml.py'
      test -f '$REMOTE_DATA_DIR/small_trashnet/partition_metadata.csv'
      '$REMOTE_PYTHON' --version
      '$REMOTE_PYTHON' -c 'import torch; assert torch.cuda.is_available(), "CUDA is unavailable"; print("torch", torch.__version__); print("cuda", torch.version.cuda); print("gpu", torch.cuda.get_device_name(0))'
      cd '$REMOTE_PROJECT_DIR'
      '$REMOTE_PYTHON' running_ml.py --help >/dev/null
    "
  done
}

run_experiment() {
  local -a active_devices pids labels
  local device client_id host pid index rc failed=0
  active_devices=("${(@f)$(reachable_device_lines)}")
  (( ${#active_devices[@]} > 0 )) || {
    print "No reachable GPU devices." >&2
    return 1
  }

  print "Running local GPU experiment on ${#active_devices[@]} device(s):"
  print "  dataset: ${DATASET}"
  print "  conditions: ${METHODS}"
  print "  reference trials: ${REFERENCE_TRIALS}"
  print "  analysis trials: ${TRIALS}"
  print "  epochs per run: ${LOCAL_EPOCHS}"
  print "  batch size: ${BATCH_SIZE}"
  print "  compute device: cuda (required)"
  print "  GPU logger: not enabled yet"
  print "  perf logger: disabled"

  for device in "${active_devices[@]}"; do
    client_id="${device%%:*}"
    host="${device#*:}"
    print "==> start ${host} (${client_id})"
    ssh_run "$host" "
      set -e
      cd '$REMOTE_PROJECT_DIR'
      '$REMOTE_PYTHON' running_ml.py \\
        --dataset '$DATASET' \\
        --data-dir '$REMOTE_DATA_DIR' \\
        --log-dir 'logs/local_ml' \\
        --client-id '$client_id' \\
        --device-id '$host' \\
        --host '$host' \\
        --poisoning-method '$METHODS' \\
        --reference-trials '$REFERENCE_TRIALS' \\
        --trials '$TRIALS' \\
        --local-epochs '$LOCAL_EPOCHS' \\
        --batch-size '$BATCH_SIZE' \\
        --torch-device cuda \\
        --disable-perf
    " &
    pid=$!
    pids+=("$pid")
    labels+=("$host")
  done

  for index in {1..${#pids[@]}}; do
    if wait "${pids[$index]}"; then
      print "==> finished ${labels[$index]}"
    else
      rc=$?
      print "==> failed ${labels[$index]} (exit ${rc})" >&2
      failed=1
    fi
  done
  (( failed == 0 )) || return 1
  print "GPU experiment finished."
}

usage() {
  cat <<'EOF'
Usage:
  ./run_experiments.zsh check
  ./run_experiments.zsh run

With no argument, the script runs `check` and then `run`.

The run executes on reachable Jetson devices 192.168.0.141 and .142:
  - IID kuchidareo/small_trashnet
  - SimpleCNN, batch size 16
  - CUDA required
  - no reference runs
  - trial_0 clean: 10 epochs
  - trial_0 availability_shortcuts: 10 epochs
  - perf disabled; GPU logger will be added separately
EOF
}

main() {
  local mode="${1:-all}"
  case "$mode" in
    check)
      pull_remote_repositories
      check_remote_environment
      ;;
    run)
      run_experiment
      ;;
    all)
      pull_remote_repositories
      check_remote_environment
      run_experiment
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
