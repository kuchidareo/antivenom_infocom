#!/usr/bin/env zsh
set -euo pipefail

SERVER_PROJECT_DIR="${0:A:h}"
SERVER_REPO_DIR="${SERVER_PROJECT_DIR:h}"
SERVER_PYTHON="${PYTHON:-${SERVER_REPO_DIR}/venv/bin/python}"
SSH_PASSWORD="${SSH_PASSWORD:-}"

SMALL_TRASHNET_DATASET="kuchidareo/small_trashnet"
CIFAR10_DATASET="uoft-cs/cifar10"
DEFAULT_METHODS="clean,unlearnable_examples,availability_shortcuts"
DEFAULT_FL_METHODS="unlearnable_examples,availability_shortcuts,random_label_flipping,clean"
FL_DATASET="${FL_DATASET:-$SMALL_TRASHNET_DATASET}"
FL_POISONED_CLIENT_COUNTS="${FL_POISONED_CLIENT_COUNTS:-10,7,4,1}"
FL_TRIALS="${FL_TRIALS:-1}"

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
PERF_ENABLED="${PERF_ENABLED:-1}"
PERF_FPS="${PERF_FPS:-10}"
PERF_EVENTS="${PERF_EVENTS:-}"

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
FL_NUM_ROUNDS="${FL_NUM_ROUNDS:-$(config_value DEFAULT_FL_NUM_ROUNDS)}"
FL_LOCAL_EPOCHS="${FL_LOCAL_EPOCHS:-$(config_value DEFAULT_FL_LOCAL_EPOCHS)}"
FL_BATCH_SIZE="${FL_BATCH_SIZE:-$(config_value DEFAULT_BATCH_SIZE)}"

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
      if [ '$PERF_ENABLED' = '1' ]; then
        command -v perf
      fi
    "
  done
}

check_fl_environment() {
  print "Checking Flower on the server and reachable clients..."
  (
    cd "$SERVER_PROJECT_DIR"
    test -f fl_client.py
    test -f fl_server.py
    test -f running_fl.py
    test -f perf_logger.py
    "$SERVER_PYTHON" -c 'import datasets, flwr, numpy, PIL, psutil, torch, torchvision; print("server Flower", flwr.__version__)'
    "$SERVER_PYTHON" -c 'import subprocess, sys; help_text = subprocess.check_output([sys.executable, "fl_client.py", "--help"], text=True); assert "{clean," in help_text, "local fl_client.py does not support clean FL"'
    "$SERVER_PYTHON" -c 'import subprocess, sys; help_text = subprocess.check_output([sys.executable, "fl_server.py", "--help"], text=True); assert "{clean," in help_text, "local fl_server.py does not support clean FL"'
  )
  for device in "${(@f)$(reachable_device_lines)}"; do
    local host="${device#*:}"
    print "==> Flower check ${host}"
    ssh_run "$host" "
      set -e
      cd '$REMOTE_PROJECT_DIR'
      test -f fl_client.py
      test -f fl_server.py
      test -f perf_logger.py
      '$REMOTE_PYTHON' -c 'import datasets, flwr, numpy, PIL, psutil, torch, torchvision; print(\"client Flower\", flwr.__version__)'
      '$REMOTE_PYTHON' -c 'import subprocess, sys; help_text = subprocess.check_output([sys.executable, \"fl_client.py\", \"--help\"], text=True); assert \"{clean,\" in help_text, \"remote fl_client.py is stale and does not support clean FL\"'
      '$REMOTE_PYTHON' -c 'import subprocess, sys; help_text = subprocess.check_output([sys.executable, \"fl_server.py\", \"--help\"], text=True); assert \"{clean,\" in help_text, \"remote fl_server.py is stale and does not support clean FL\"'
    "
  done
}

configure_remote_perf() {
  if [[ "$PERF_ENABLED" != "1" ]]; then
    return
  fi

  print "Configuring perf_event_paranoid=-1 on reachable devices..."
  for device in "${(@f)$(reachable_device_lines)}"; do
    local host="${device#*:}"
    print "==> perf setup ${host}"
    if [[ -n "$SSH_PASSWORD" ]]; then
      ssh_run "$host" "printf '%s\n' '$SSH_PASSWORD' | sudo -S sysctl kernel.perf_event_paranoid=-1"
    else
      ssh_run "$host" "sudo -n sysctl kernel.perf_event_paranoid=-1"
    fi
  done
}

validate_remote_perf_events() {
  if [[ "$PERF_ENABLED" != "1" ]]; then
    print "Perf monitoring is disabled; skipping PMU event validation."
    return
  fi

  print "Validating host-specific perf event profiles..."
  for device in "${(@f)$(reachable_device_lines)}"; do
    local host="${device#*:}"
    local events
    events="$(
      cd "$SERVER_PROJECT_DIR"
      "$SERVER_PYTHON" -c "from perf_logger import default_perf_events_for_host; print(','.join(default_perf_events_for_host('$host')))"
    )"
    print "==> perf events ${host}"
    ssh_run "$host" "
      set -e
      output='/tmp/antivenom_perf_event_check.out'
      if perf stat -e '$events' -- true >\"\$output\" 2>&1; then
        rm -f \"\$output\"
        echo 'perf event profile: ok'
      else
        cat \"\$output\" >&2
        rm -f \"\$output\"
        exit 1
      fi
    "
  done
}

print_reachable_fl_clients() {
  local -a configured_lines active_lines active_ids
  local device
  configured_lines=("${(@f)$(device_lines)}")
  active_lines=("${(@f)$(reachable_device_lines)}")
  for device in "${active_lines[@]}"; do
    active_ids+=("${device%%:*}")
  done
  print "Reachable FL clients (${#active_ids[@]}/${#configured_lines[@]}): ${(j:,:)active_ids}"
}

perf_args_for_python() {
  if [[ "$PERF_ENABLED" == "1" ]]; then
    print -- "--enable-perf --perf-fps '$PERF_FPS' --perf-events '$PERF_EVENTS'"
  else
    print -- "--disable-perf"
  fi
}

normalize_fl_methods() {
  local raw="$1"
  local -a selected
  local token
  raw="${raw//,/ }"

  for token in ${(z)raw}; do
    case "$token" in
      clean|unlearnable_examples|random_label_flipping|target_label_flipping|availability_shortcuts)
        if (( ${selected[(Ie)$token]} == 0 )); then
          selected+=("$token")
        fi
        ;;
      *)
        print "Unknown FL condition: ${token}" >&2
        return 1
        ;;
    esac
  done

  if (( ${#selected[@]} == 0 )); then
    print "No FL attack conditions remain after validation." >&2
    return 1
  fi
  print -- "${(j:,:)selected}"
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
  local perf_option
  perf_option="$(perf_args_for_python)"

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
        $perf_option \
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
  configure_remote_perf

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

run_fl_experiments() {
  local methods="${1:-$DEFAULT_FL_METHODS}"
  local -a active_lines active_ids perf_args
  local device

  methods="$(normalize_fl_methods "$methods")"

  active_lines=("${(@f)$(reachable_device_lines)}")
  if (( ${#active_lines[@]} == 0 )); then
    print "No reachable FL clients." >&2
    return 1
  fi
  for device in "${active_lines[@]}"; do
    active_ids+=("${device%%:*}")
  done

  local count
  for count in ${(s:,:)FL_POISONED_CLIENT_COUNTS}; do
    if [[ "$count" != <-> ]]; then
      print "Invalid poisoned-client count: ${count}" >&2
      return 1
    fi
    if (( count > ${#active_ids[@]} )); then
      print "Cannot run poisoned_client_count=${count} with only ${#active_ids[@]} reachable FL clients." >&2
      print "All 10 clients must be reachable for the poisoned_client_count=10 condition." >&2
      return 1
    fi
  done

  if [[ "$PERF_ENABLED" == "1" ]]; then
    perf_args=(--enable-perf --perf-fps "$PERF_FPS" --perf-events "$PERF_EVENTS")
  else
    perf_args=(--disable-perf)
  fi

  print
  print "Running Flower FL experiments:"
  print "  dataset: ${FL_DATASET}"
  print "  methods: ${methods}"
  print "  clean baseline poisoned_client_count: 0"
  print "  attack poisoned_client_counts: ${FL_POISONED_CLIENT_COUNTS}"
  print "  trials: ${FL_TRIALS}"
  print "  rounds per trial: ${FL_NUM_ROUNDS}"
  print "  local epochs per client per round: ${FL_LOCAL_EPOCHS}"
  print "  batch size: ${FL_BATCH_SIZE}"
  print "  active_clients: ${(j:,:)active_ids}"
  print "  client_perf: ${PERF_ENABLED}, fps=${PERF_FPS}"

  cd "$SERVER_PROJECT_DIR"
  "$SERVER_PYTHON" running_fl.py \
    --dry-run \
    --dataset "$FL_DATASET" \
    --poisoning-methods "$methods" \
    --poisoned-client-counts "$FL_POISONED_CLIENT_COUNTS" \
    --trials "$FL_TRIALS" \
    --num-rounds "$FL_NUM_ROUNDS" \
    --local-epochs "$FL_LOCAL_EPOCHS" \
    --batch-size "$FL_BATCH_SIZE" \
    --active-client-ids "${(j:,:)active_ids}" \
    --ssh-password "$SSH_PASSWORD" \
    "${perf_args[@]}"

  "$SERVER_PYTHON" running_fl.py \
    --dataset "$FL_DATASET" \
    --poisoning-methods "$methods" \
    --poisoned-client-counts "$FL_POISONED_CLIENT_COUNTS" \
    --trials "$FL_TRIALS" \
    --num-rounds "$FL_NUM_ROUNDS" \
    --local-epochs "$FL_LOCAL_EPOCHS" \
    --batch-size "$FL_BATCH_SIZE" \
    --active-client-ids "${(j:,:)active_ids}" \
    --ssh-password "$SSH_PASSWORD" \
    --server-log-hardware \
    "${perf_args[@]}"
  print "Flower FL experiments finished."
}

usage() {
  cat <<'EOF'
Usage:
  ./run_experiments.zsh check
  ./run_experiments.zsh fl-check
  ./run_experiments.zsh bg-check
  ./run_experiments.zsh all [conditions]
  ./run_experiments.zsh fl [attack_conditions]

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
  ./run_experiments_0718.zsh fl unlearnable_examples,availability_shortcuts,random_label_flipping,clean

Environment:
  SSH_PASSWORD=...                 optional if SSH keys are configured
  REFERENCE_TRIALS=5
  ANALYSIS_TRIALS=5
  BG_WORKLOAD_GROUP=group1         type I/perception workload
  BG_WORKLOAD_PROFILE=medium
  BG_WORKLOAD_TEST_DURATION=10
  PERF_ENABLED=1                   collect perf CSV files; use 0 to disable
  PERF_FPS=10
  PERF_EVENTS=                     empty uses perf_logger.py defaults
  FL_DATASET=kuchidareo/small_trashnet
  FL_POISONED_CLIENT_COUNTS=10,7,4,1
  FL_TRIALS=1
  FL_NUM_ROUNDS=15
  FL_LOCAL_EPOCHS=1
  FL_BATCH_SIZE=16
EOF
}

main() {
  local mode="${1:-all}"
  shift || true

  case "$mode" in
    check)
      pull_remote_repos
      check_remote_environment
      configure_remote_perf
      ;;
    fl-check)
      pull_remote_repos
      check_remote_environment
      check_fl_environment
      configure_remote_perf
      validate_remote_perf_events
      print_reachable_fl_clients
      print "FL environment check finished; no experiment was started."
      ;;
    bg-check)
      pull_remote_repos
      check_remote_environment
      configure_remote_perf
      check_bg_workloads
      ;;
    all)
      run_all_local_ml "${1:-$DEFAULT_METHODS}"
      ;;
    fl)
      pull_remote_repos
      check_remote_environment
      check_fl_environment
      configure_remote_perf
      validate_remote_perf_events
      run_fl_experiments "${1:-$DEFAULT_FL_METHODS}"
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
