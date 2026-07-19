#!/usr/bin/env zsh
set -euo pipefail

SERVER_PROJECT_DIR="${0:A:h}"
SSH_USER="${SSH_USER:-rasheed}"
SSH_PASSWORD="${SSH_PASSWORD:-}"
SSH_PORT="${SSH_PORT:-22}"
PING_TIMEOUT_SEC="${PING_TIMEOUT_SEC:-1}"

REMOTE_PROJECT_DIR="${REMOTE_PROJECT_DIR:-/home/rasheed/kuchida/antivenom_infocom/M260719-robustness-library}"
REMOTE_DATA_DIR="${REMOTE_DATA_DIR:-${REMOTE_PROJECT_DIR:h}/iid-data}"
REMOTE_BOOTSTRAP_PYTHON="${REMOTE_BOOTSTRAP_PYTHON:-/home/rasheed/kuchida/antivenom_infocom/venv/bin/python}"
REMOTE_PYTHON="${REMOTE_PYTHON:-${REMOTE_PROJECT_DIR}/.venv/bin/python}"

DATASET="kuchidareo/small_trashnet"
METHODS="clean,unlearnable_examples,random_label_flipping,target_label_flipping,availability_shortcuts"
LOCAL_EPOCHS="${LOCAL_EPOCHS:-10}"
BATCH_SIZE="${BATCH_SIZE:-16}"
TRIALS="${TRIALS:-1}"
SEED="${SEED:-260626}"
PERF_FPS="${PERF_FPS:-10}"
PERF_EVENTS="${PERF_EVENTS:-}"
JETSON_PERF_EVENTS="cycles,instructions,task-clock,context-switches,cpu-migrations,page-faults,br_retired,br_mis_pred_retired,l1d_cache,l1d_cache_refill,l1d_cache_wb,l2d_cache,l2d_cache_refill,l2d_cache_wb,bus_access,mem_access,inst_spec"

device_lines() {
  print -- "client_1:192.168.0.141"
  print -- "client_2:192.168.0.142"
}

ssh_target() {
  print -- "${SSH_USER}@$1"
}

host_is_reachable() {
  ping -c 1 -W "$PING_TIMEOUT_SEC" "$1" >/dev/null 2>&1
}

ssh_run() {
  local host="$1"
  local command="$2"
  local target="$(ssh_target "$host")"
  if [[ -n "$SSH_PASSWORD" ]]; then
    command -v sshpass >/dev/null || {
      print -u2 "SSH_PASSWORD is set, but sshpass is not installed."
      return 1
    }
    sshpass -p "$SSH_PASSWORD" ssh -p "$SSH_PORT" \
      -o StrictHostKeyChecking=accept-new "$target" "$command"
  else
    ssh -p "$SSH_PORT" -o StrictHostKeyChecking=accept-new "$target" "$command"
  fi
}

rsync_project() {
  local host="$1"
  local target="$(ssh_target "$host")"
  local -a transport
  if [[ -n "$SSH_PASSWORD" ]]; then
    transport=(sshpass -p "$SSH_PASSWORD" ssh -p "$SSH_PORT" -o StrictHostKeyChecking=accept-new)
  else
    transport=(ssh -p "$SSH_PORT" -o StrictHostKeyChecking=accept-new)
  fi
  ssh_run "$host" "mkdir -p '$REMOTE_PROJECT_DIR'"
  rsync -az \
    --include='*.py' \
    --include='*.zsh' \
    --include='*.txt' \
    --exclude='*' \
    -e "${(j: :)transport}" \
    "${SERVER_PROJECT_DIR}/" \
    "${target}:${REMOTE_PROJECT_DIR}/"
}

reachable_devices() {
  local device host
  for device in "${(@f)$(device_lines)}"; do
    host="${device#*:}"
    if host_is_reachable "$host"; then
      print -- "$device"
    else
      print -u2 "Unreachable device: $host"
    fi
  done
}

sync_code() {
  local device host
  command -v rsync >/dev/null || {
    print -u2 "rsync is required."
    return 1
  }
  for device in "${(@f)$(reachable_devices)}"; do
    host="${device#*:}"
    print "Synchronizing experiment code to $host"
    rsync_project "$host"
  done
}

setup_devices() {
  local device host
  sync_code
  for device in "${(@f)$(reachable_devices)}"; do
    host="${device#*:}"
    print "Installing TensorFlow environment on $host"
    ssh_run "$host" "
      set -e
      test -x '$REMOTE_BOOTSTRAP_PYTHON'
      if [ ! -x '$REMOTE_PYTHON' ]; then
        '$REMOTE_BOOTSTRAP_PYTHON' -m venv '${REMOTE_PROJECT_DIR}/.venv'
      fi
      '$REMOTE_PYTHON' -m pip install --upgrade pip
      '$REMOTE_PYTHON' -m pip install -r '${REMOTE_PROJECT_DIR}/requirements-tensorflow.txt'
    "
  done
  configure_perf
  check_devices
}

configure_perf() {
  local device host
  for device in "${(@f)$(reachable_devices)}"; do
    host="${device#*:}"
    print "Configuring perf access on $host"
    if [[ -n "$SSH_PASSWORD" ]]; then
      ssh_run "$host" "printf '%s\n' '$SSH_PASSWORD' | sudo -S sysctl kernel.perf_event_paranoid=-1"
    else
      ssh_run "$host" "sudo -n sysctl kernel.perf_event_paranoid=-1"
    fi
  done
}

check_devices() {
  local device client_id host
  local found=0
  for device in "${(@f)$(reachable_devices)}"; do
    found=1
    client_id="${device%%:*}"
    host="${device#*:}"
    print "Checking $host ($client_id)"
    ssh_run "$host" "
      set -eu
      echo '[check] project and TensorFlow environment'
      test -d '$REMOTE_PROJECT_DIR' || {
        echo 'Missing remote project: $REMOTE_PROJECT_DIR' >&2
        exit 1
      }
      test -x '$REMOTE_PYTHON' || {
        echo 'Missing TensorFlow environment: $REMOTE_PYTHON' >&2
        echo 'Run ./run_experiments.zsh setup first.' >&2
        exit 1
      }
      test -f '${REMOTE_DATA_DIR}/small_trashnet/partition_metadata.csv' || {
        echo 'Missing prepared dataset under ${REMOTE_DATA_DIR}/small_trashnet.' >&2
        exit 1
      }
      command -v perf >/dev/null || {
        echo 'perf is not installed on $host.' >&2
        exit 1
      }

      echo '[check] Jetson perf events'
      if ! perf stat -e '$JETSON_PERF_EVENTS' -- true; then
        echo 'Perf validation failed on $host.' >&2
        echo 'Run ./run_experiments.zsh setup to configure perf access.' >&2
        exit 1
      fi

      cd '$REMOTE_PROJECT_DIR'
      echo '[check] TensorFlow and prepared dataset'
      CUDA_VISIBLE_DEVICES=-1 '$REMOTE_PYTHON' running_ml.py \
        --data-dir '$REMOTE_DATA_DIR' \
        --dataset '$DATASET' \
        --client-id '$client_id' \
        --host '$host' \
        --device-id '$host' \
        --poisoning-method '$METHODS' \
        --validate-only

      echo '[check] TensorFlow experiment tests'
      CUDA_VISIBLE_DEVICES=-1 '$REMOTE_PYTHON' -m unittest -v test_tensorflow_experiment.py
      echo '[ok] $host ($client_id)'
    "
  done
  (( found )) || {
    print -u2 "No configured Jetson is reachable."
    return 1
  }
}

run_experiment() {
  local device client_id host perf_option
  local -a pids
  local failed=0
  sync_code
  check_devices

  if [[ -n "$PERF_EVENTS" ]]; then
    perf_option="--perf-events '$PERF_EVENTS'"
  else
    perf_option=""
  fi

  for device in "${(@f)$(reachable_devices)}"; do
    client_id="${device%%:*}"
    host="${device#*:}"
    print "Starting TensorFlow experiment on $host ($client_id)"
    ssh_run "$host" "
      set -e
      cd '$REMOTE_PROJECT_DIR'
      CUDA_VISIBLE_DEVICES=-1 '$REMOTE_PYTHON' running_ml.py \
        --data-dir '$REMOTE_DATA_DIR' \
        --dataset '$DATASET' \
        --partition-method iid \
        --client-id '$client_id' \
        --host '$host' \
        --device-id '$host' \
        --model simple_cnn \
        --batch-size '$BATCH_SIZE' \
        --local-epochs '$LOCAL_EPOCHS' \
        --trials '$TRIALS' \
        --seed '$SEED' \
        --poisoning-method '$METHODS' \
        --perf-fps '$PERF_FPS' \
        $perf_option
    " &
    pids+=("$!")
  done

  for pid in "${pids[@]}"; do
    wait "$pid" || failed=1
  done
  (( failed == 0 )) || return 1
  print "TensorFlow experiment completed on all reachable devices."
}

usage() {
  cat <<'EOF'
Usage:
  ./run_experiments.zsh setup  # sync code, create remote venvs, install TensorFlow
  ./run_experiments.zsh check  # check TensorFlow, CPU-only mode, data, and perf
  ./run_experiments.zsh run    # run all five conditions on .141 and .142

Environment overrides:
  SSH_PASSWORD=...
  LOCAL_EPOCHS=10
  BATCH_SIZE=16
  TRIALS=1
  SEED=260626
  PERF_FPS=10
  PERF_EVENTS=                 # empty selects the Jetson event profile
EOF
}

case "${1:-run}" in
  setup)
    setup_devices
    ;;
  check)
    check_devices
    ;;
  run|all)
    run_experiment
    ;;
  -h|--help|help)
    usage
    ;;
  *)
    print -u2 "Unknown mode: $1"
    usage >&2
    exit 1
    ;;
esac
