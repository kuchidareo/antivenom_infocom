#!/usr/bin/env zsh
set -euo pipefail

SERVER_PROJECT_DIR="${0:A:h}"
SERVER_REPO_DIR="${SERVER_PROJECT_DIR:h}"
SERVER_PYTHON="${PYTHON:-${SERVER_REPO_DIR}/venv/bin/python}"
SSH_PASSWORD="${SSH_PASSWORD:-}"

SMALL_TRASHNET_DATASET="kuchidareo/small_trashnet"
CIFAR10_DATASET="uoft-cs/cifar10"
DEFAULT_METHODS="clean,unlearnable_examples,availability_shortcuts"

REFERENCE_TRIALS="${REFERENCE_TRIALS:-5}"
ANALYSIS_TRIALS="${ANALYSIS_TRIALS:-5}"
BG_WORKLOAD_ENABLED="${BG_WORKLOAD_ENABLED:-1}"
BG_WORKLOAD_GROUP="${BG_WORKLOAD_GROUP:-group1}"
BG_WORKLOAD_PROFILE="${BG_WORKLOAD_PROFILE:-medium}"
BG_WORKLOAD_TEST_DURATION="${BG_WORKLOAD_TEST_DURATION:-10}"
BG_WORKLOAD_PID_FILE="${BG_WORKLOAD_PID_FILE:-/tmp/antivenom_bg_workload.pid}"
BG_WORKLOAD_PYTHON="${BG_WORKLOAD_PYTHON:-/home/rasheed/kuchida/antivenom_infocom/venv/bin/python}"
BG_INSTALL_DEPENDENCIES="${BG_INSTALL_DEPENDENCIES:-1}"
BG_INSTALL_OPENCV="${BG_INSTALL_OPENCV:-1}"
BG_OPENCV_PIP_PACKAGE="${BG_OPENCV_PIP_PACKAGE:-opencv-python-headless}"
BG_WORKLOAD_CHECKED=0
PING_TIMEOUT_SEC="${PING_TIMEOUT_SEC:-1}"

config_value() {
  local name="$1"
  cd "$SERVER_PROJECT_DIR"
  "$SERVER_PYTHON" -c "import experiment_config as c; print(getattr(c, '$name'))"
}

device_lines() {
  cd "$SERVER_PROJECT_DIR"
  "$SERVER_PYTHON" -c "import experiment_config as c; [print(f\"{d['client_id']}:{d['host']}\") for d in c.DEVICES]"
}

host_is_reachable() {
  local host="$1"
  ping -c 1 -W "$PING_TIMEOUT_SEC" "$host" >/dev/null 2>&1
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

REMOTE_PROJECT_DIR="$(config_value DEFAULT_REMOTE_PROJECT_DIR)"
REMOTE_REPO_DIR="${REMOTE_PROJECT_DIR:h}"
REMOTE_PYTHON="$(config_value DEFAULT_REMOTE_PYTHON)"
SSH_USER="$(config_value DEFAULT_SSH_USER)"

ssh_target() {
  local host="$1"
  if [[ -n "$SSH_USER" ]]; then
    print -- "${SSH_USER}@${host}"
  else
    print -- "$host"
  fi
}

ssh_run() {
  local host="$1"
  local remote_command="$2"
  local target
  target="$(ssh_target "$host")"
  if [[ -n "$SSH_PASSWORD" ]]; then
    if ! command -v sshpass >/dev/null 2>&1; then
      print "SSH_PASSWORD is set, but sshpass is not installed." >&2
      print "Install sshpass or configure SSH keys." >&2
      exit 1
    fi
    sshpass -p "$SSH_PASSWORD" ssh -o StrictHostKeyChecking=accept-new "$target" "$remote_command"
  else
    ssh "$target" "$remote_command"
  fi
}

pull_remote_repos() {
  print "Updating remote repositories with git pull --rebase..."
  for device in "${(@f)$(reachable_device_lines)}"; do
    local host="${device#*:}"
    print "==> git pull --rebase ${host}"
    ssh_run "$host" "cd '$REMOTE_REPO_DIR' && git pull --rebase"
  done
}

check_remote_environment() {
  print "Checking remote Python, project directory, and shared iid-data..."
  for device in "${(@f)$(reachable_device_lines)}"; do
    local host="${device#*:}"
    print "==> check ${host}"
    ssh_run "$host" "
      set -e
      test -d '$REMOTE_PROJECT_DIR'
      test -x '$REMOTE_PYTHON'
      '$REMOTE_PYTHON' --version
      test -d '${REMOTE_PROJECT_DIR:h}/iid-data/small_trashnet'
      test -d '${REMOTE_PROJECT_DIR:h}/iid-data/cifar10'
    "
  done
}

bg_args_for_python() {
  print -- "--background-workload-enabled --background-workload-group '$BG_WORKLOAD_GROUP' --background-workload-profile '$BG_WORKLOAD_PROFILE'"
}

bg_requires_perception() {
  case "$BG_WORKLOAD_GROUP" in
    group1|perception|both|all)
      return 0
      ;;
    *)
      return 1
      ;;
  esac
}

install_bg_dependencies() {
  if [[ "$BG_WORKLOAD_ENABLED" != "1" || "$BG_INSTALL_DEPENDENCIES" != "1" ]]; then
    return
  fi
  if ! bg_requires_perception || [[ "$BG_INSTALL_OPENCV" != "1" ]]; then
    return
  fi

  print "Installing/checking background workload dependencies on all devices..."
  for device in "${(@f)$(reachable_device_lines)}"; do
    local host="${device#*:}"
    print "==> bg deps ${host}"
    ssh_run "$host" "
      set -e
      if '$BG_WORKLOAD_PYTHON' -c 'import cv2, numpy' >/dev/null 2>&1; then
        echo '[bg][deps] cv2/numpy already available'
      else
        echo '[bg][deps] installing $BG_OPENCV_PIP_PACKAGE'
        '$BG_WORKLOAD_PYTHON' -m pip install --upgrade '$BG_OPENCV_PIP_PACKAGE'
        '$BG_WORKLOAD_PYTHON' -c 'import cv2, numpy; print(\"[bg][deps] cv2/numpy import ok\")'
      fi
    "
  done
}

check_bg_workloads() {
  if [[ "$BG_WORKLOAD_ENABLED" != "1" || "$BG_WORKLOAD_CHECKED" == "1" ]]; then
    return
  fi
  install_bg_dependencies
  print "Checking background workload dry-run/test on all devices..."
  local perception_check="true"
  if bg_requires_perception; then
    perception_check="test -x '$BG_WORKLOAD_PYTHON' && '$BG_WORKLOAD_PYTHON' -c 'import cv2, numpy'"
  fi
  for device in "${(@f)$(reachable_device_lines)}"; do
    local host="${device#*:}"
    print "==> bg check ${host}"
    ssh_run "$host" "
      set -e
      cd '$REMOTE_PROJECT_DIR'
      test -x ./run_bg_workloads.sh
      $perception_check
      PYTHON_BIN='$BG_WORKLOAD_PYTHON' ./run_bg_workloads.sh --group '$BG_WORKLOAD_GROUP' --profile '$BG_WORKLOAD_PROFILE' --dry-run
      PYTHON_BIN='$BG_WORKLOAD_PYTHON' ./run_bg_workloads.sh --group '$BG_WORKLOAD_GROUP' --profile '$BG_WORKLOAD_PROFILE' --test --duration-sec '$BG_WORKLOAD_TEST_DURATION'
    "
  done
  BG_WORKLOAD_CHECKED=1
}

start_bg_workloads() {
  if [[ "$BG_WORKLOAD_ENABLED" != "1" ]]; then
    return
  fi
  print "Starting background workloads: group=${BG_WORKLOAD_GROUP}, profile=${BG_WORKLOAD_PROFILE}"
  for device in "${(@f)$(reachable_device_lines)}"; do
    local host="${device#*:}"
    ssh_run "$host" "
      set -e
      cd '$REMOTE_PROJECT_DIR'
      mkdir -p logs/bg_workloads
      if [ -f '$BG_WORKLOAD_PID_FILE' ] && kill -0 \$(cat '$BG_WORKLOAD_PID_FILE') 2>/dev/null; then
        echo 'bg workload already running on ${host}: pid='\"\$(cat '$BG_WORKLOAD_PID_FILE')\";
      else
        nohup env PYTHON_BIN='$BG_WORKLOAD_PYTHON' ./run_bg_workloads.sh --group '$BG_WORKLOAD_GROUP' --profile '$BG_WORKLOAD_PROFILE' > logs/bg_workloads/run_bg_workloads.out 2>&1 < /dev/null &
        echo \$! > '$BG_WORKLOAD_PID_FILE'
        sleep 2
        if ! kill -0 \$(cat '$BG_WORKLOAD_PID_FILE') 2>/dev/null; then
          echo 'bg workload failed to stay running on ${host}' >&2
          tail -n 80 logs/bg_workloads/run_bg_workloads.out >&2 || true
          exit 1
        fi
        echo 'bg workload started on ${host}: pid='\"\$(cat '$BG_WORKLOAD_PID_FILE')\";
      fi
    "
  done
}

stop_bg_workloads() {
  if [[ "$BG_WORKLOAD_ENABLED" != "1" ]]; then
    return
  fi
  print "Stopping background workloads..."
  for device in "${(@f)$(reachable_device_lines)}"; do
    local host="${device#*:}"
    ssh_run "$host" "
      if [ -f '$BG_WORKLOAD_PID_FILE' ]; then
        pid=\$(cat '$BG_WORKLOAD_PID_FILE')
        kill \"\$pid\" 2>/dev/null || true
        sleep 1
        kill -9 \"\$pid\" 2>/dev/null || true
        rm -f '$BG_WORKLOAD_PID_FILE'
        echo 'bg workload stopped on ${host}'
      else
        echo 'no bg workload pid file on ${host}'
      fi
    " || true
  done
}

run_with_bg_workloads() {
  local exit_code=0
  check_bg_workloads
  start_bg_workloads || {
    exit_code=$?
    stop_bg_workloads
    return "$exit_code"
  }
  "$@" || exit_code=$?
  stop_bg_workloads
  return "$exit_code"
}

run_local_ml_stage() {
  local stage_name="$1"
  local dataset_name="$2"
  local methods="$3"
  local reference_trials="$4"
  local analysis_trials="$5"
  local use_bg="$6"

  local reference_option="--reference-trials '$reference_trials'"
  local trials_option="--trials '$analysis_trials'"
  local method_option="--poisoning-method '$methods'"
  local bg_option=""

  if [[ "$use_bg" == "1" ]]; then
    bg_option="$(bg_args_for_python)"
  fi

  print
  print "Running stage: ${stage_name}"
  print "  dataset: ${dataset_name}"
  print "  methods: ${methods}"
  print "  reference_trials: ${reference_trials}, analysis_trials: ${analysis_trials}"
  print "  bg_noise: ${use_bg}"

  for device in "${(@f)$(reachable_device_lines)}"; do
    local client_id="${device%%:*}"
    local host="${device#*:}"
    ssh_run "$host" "
      set -e
      cd '$REMOTE_PROJECT_DIR'
      '$REMOTE_PYTHON' running_ml.py \
        --dataset '$dataset_name' \
        --client-id '$client_id' \
        --device-id '$host' \
        --host '$host' \
        $reference_option \
        $trials_option \
        $method_option \
        $bg_option
    " &
  done
  wait
  print "Finished stage: ${stage_name}"
}

run_all_local_ml() {
  local methods="${1:-$DEFAULT_METHODS}"

  pull_remote_repos
  check_remote_environment

  run_local_ml_stage \
    "small_trashnet_no_bg_reference_and_analysis" \
    "$SMALL_TRASHNET_DATASET" \
    "$methods" \
    "$REFERENCE_TRIALS" \
    "$ANALYSIS_TRIALS" \
    "0"

  run_local_ml_stage \
    "cifar10_no_bg_analysis_only" \
    "$CIFAR10_DATASET" \
    "$methods" \
    "0" \
    "$ANALYSIS_TRIALS" \
    "0"

  run_with_bg_workloads run_local_ml_stage \
    "small_trashnet_bg_type_i_analysis_only" \
    "$SMALL_TRASHNET_DATASET" \
    "$methods" \
    "0" \
    "$ANALYSIS_TRIALS" \
    "1"
}

usage() {
  cat <<'EOF'
Usage:
  ./run_experiments.zsh check
  ./run_experiments.zsh bg-check
  ./run_experiments.zsh all [conditions]

Default experiment:
  1. small_trashnet without bg:
     5 clean global reference runs, then 5 clean + 5 unlearnable_examples + 5 availability_shortcuts analysis runs.
  2. cifar10 without bg:
     5 clean + 5 unlearnable_examples + 5 availability_shortcuts analysis runs.
  3. small_trashnet with bg-noise type I:
     5 clean + 5 unlearnable_examples + 5 availability_shortcuts analysis runs.

Examples:
  ./run_experiments.zsh all
  ./run_experiments.zsh all clean,unlearnable_examples,availability_shortcuts

Environment:
  SSH_PASSWORD=...                 optional if SSH keys are configured
  REFERENCE_TRIALS=5
  ANALYSIS_TRIALS=5
  BG_WORKLOAD_GROUP=group1         type I/perception workload
  BG_WORKLOAD_PROFILE=medium
  BG_WORKLOAD_TEST_DURATION=10
EOF
}

main() {
  local mode="${1:-all}"
  shift || true

  case "$mode" in
    check)
      pull_remote_repos
      check_remote_environment
      ;;
    bg-check)
      pull_remote_repos
      check_remote_environment
      check_bg_workloads
      ;;
    all)
      run_all_local_ml "${1:-$DEFAULT_METHODS}"
      ;;
    -h|--help|help)
      usage
      ;;
    *)
      print "Unknown mode: $mode" >&2
      usage >&2
      exit 1
      ;;
  esac
}

main "$@"
