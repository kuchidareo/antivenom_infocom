#!/usr/bin/env zsh
set -euo pipefail

SERVER_PROJECT_DIR="${0:A:h}"
SERVER_PYTHON="${PYTHON:-${SERVER_PROJECT_DIR:h}/venv/bin/python}"
SSH_PASSWORD="${SSH_PASSWORD:-}"

config_value() {
  local name="$1"
  cd "$SERVER_PROJECT_DIR"
  "$SERVER_PYTHON" -c "import experiment_config as c; print(getattr(c, '$name'))"
}

device_lines() {
  cd "$SERVER_PROJECT_DIR"
  "$SERVER_PYTHON" -c "import experiment_config as c; [print(f\"{d['client_id']}:{d['host']}\") for d in c.DEVICES]"
}

REMOTE_PROJECT_DIR="$(config_value DEFAULT_REMOTE_PROJECT_DIR)"
REMOTE_REPO_DIR="${REMOTE_PROJECT_DIR:h}"
REMOTE_PYTHON="$(config_value DEFAULT_REMOTE_PYTHON)"
SSH_USER="$(config_value DEFAULT_SSH_USER)"
BG_WORKLOAD_ENABLED="${BG_WORKLOAD_ENABLED:-1}"
BG_WORKLOAD_GROUP="${BG_WORKLOAD_GROUP:-$(config_value DEFAULT_BACKGROUND_WORKLOAD_GROUP)}"
BG_WORKLOAD_PROFILE="${BG_WORKLOAD_PROFILE:-$(config_value DEFAULT_BACKGROUND_WORKLOAD_PROFILE)}"
BG_WORKLOAD_TEST_DURATION="${BG_WORKLOAD_TEST_DURATION:-10}"
BG_WORKLOAD_PID_FILE="${BG_WORKLOAD_PID_FILE:-/tmp/antivenom_bg_workload.pid}"
BG_WORKLOAD_PYTHON="${BG_WORKLOAD_PYTHON:-/home/rasheed/kuchida/antivenom_infocom/venv/bin/python}"
BG_WORKLOAD_CHECKED=0

normalize_method_token() {
  local token="${1:l}"
  token="${token//-/_}"
  token="${token// /}"
  case "$token" in
    clean|c)
      print -- "clean"
      ;;
    unlearnable_examples)
      print -- "unlearnable_examples"
      ;;
    random_label_flipping)
      print -- "random_label_flipping"
      ;;
    target_label_flipping)
      print -- "target_label_flipping"
      ;;
    availability_shortcuts)
      print -- "availability_shortcuts"
      ;;
    all|both)
      print -- "all"
      ;;
    "")
      print -- ""
      ;;
    *)
      print "Unknown condition '$1'." >&2
      print "Allowed: clean, unlearnable_examples, random_label_flipping, target_label_flipping, availability_shortcuts, all" >&2
      exit 1
      ;;
  esac
}

normalize_conditions() {
  local allow_clean="${1:-yes}"
  shift || true
  local raw="$*"
  raw="${raw//,/ }"
  raw="${raw//;/ }"
  if [[ -z "${raw// /}" ]]; then
    print -- ""
    return
  fi

  local -a methods
  local -A seen
  local token method
  for token in ${(z)raw}; do
    method="$(normalize_method_token "$token")"
    [[ -z "$method" ]] && continue
    if [[ "$method" == "all" ]]; then
      if [[ "$allow_clean" == "yes" ]]; then
        print -- "all"
      else
        print -- "unlearnable_examples,random_label_flipping,target_label_flipping,availability_shortcuts"
      fi
      return
    fi
    if [[ "$allow_clean" != "yes" && "$method" == "clean" ]]; then
      print "Skipping clean for FL because FL conditions are attack methods only." >&2
      continue
    fi
    if [[ -z "${seen[$method]:-}" ]]; then
      methods+=("$method")
      seen[$method]=1
    fi
  done
  print -- "${(j:,:)methods}"
}

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

check_remote_python() {
  print "Checking remote Python on all devices..."
  for device in "${(@f)$(device_lines)}"; do
    local host="${device#*:}"
    ssh_run "$host" "cd '$REMOTE_PROJECT_DIR' && '$REMOTE_PYTHON' --version"
  done
}

pull_remote_repos() {
  print "Updating remote repositories with git pull --rebase..."
  for device in "${(@f)$(device_lines)}"; do
    local host="${device#*:}"
    ssh_run "$host" "cd '$REMOTE_REPO_DIR' && git pull --rebase"
  done
}

bg_args_for_python() {
  if [[ "$BG_WORKLOAD_ENABLED" != "1" ]]; then
    print -- ""
    return
  fi
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

check_bg_workloads() {
  if [[ "$BG_WORKLOAD_ENABLED" != "1" || "$BG_WORKLOAD_CHECKED" == "1" ]]; then
    return
  fi
  print "Checking background workload dependencies on all devices..."
  local perception_check=""
  if bg_requires_perception; then
    perception_check="test -x '$BG_WORKLOAD_PYTHON' && '$BG_WORKLOAD_PYTHON' -c 'import cv2, numpy'"
  else
    perception_check="true"
  fi
  for device in "${(@f)$(device_lines)}"; do
    local host="${device#*:}"
    print "Checking bg workload on ${host}..."
    ssh_run "$host" "
      cd '$REMOTE_PROJECT_DIR' &&
      test -x ./run_bg_workloads.sh &&
      $perception_check &&
      if ! command -v iperf3 >/dev/null 2>&1; then
        echo '[bg][warn] iperf3 not found; communication workload will be skipped.'
      fi &&
      cd '$REMOTE_PROJECT_DIR' &&
      PYTHON_BIN='$BG_WORKLOAD_PYTHON' ./run_bg_workloads.sh --group '$BG_WORKLOAD_GROUP' --profile '$BG_WORKLOAD_PROFILE' --dry-run &&
      PYTHON_BIN='$BG_WORKLOAD_PYTHON' ./run_bg_workloads.sh --group '$BG_WORKLOAD_GROUP' --profile '$BG_WORKLOAD_PROFILE' --test --duration-sec '$BG_WORKLOAD_TEST_DURATION'
    "
  done
  BG_WORKLOAD_CHECKED=1
}

start_bg_workloads() {
  if [[ "$BG_WORKLOAD_ENABLED" != "1" ]]; then
    return
  fi
  print "Starting background workloads on all devices: group=${BG_WORKLOAD_GROUP}, profile=${BG_WORKLOAD_PROFILE}"
  for device in "${(@f)$(device_lines)}"; do
    local host="${device#*:}"
    ssh_run "$host" "
      cd '$REMOTE_PROJECT_DIR' &&
      mkdir -p logs/bg_workloads &&
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
  print "Stopping background workloads on all devices..."
  for device in "${(@f)$(device_lines)}"; do
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
  local status=0
  check_bg_workloads
  start_bg_workloads || {
    status=$?
    stop_bg_workloads
    return "$status"
  }
  "$@" || status=$?
  stop_bg_workloads
  return "$status"
}

run_local_ml() {
  local methods="${1:-}"
  local method_option=""
  local reference_option=""
  local extra_options=""
  if [[ -n "$methods" ]]; then
    method_option="--poisoning-method '$methods'"
    print "Starting local ML on all devices for conditions: $methods"
    if [[ "$methods" != "all" && ",$methods," != *",clean,"* ]]; then
      reference_option="--reference-trials 0"
      print "Skipping local ML global clean reference runs for this subset."
    fi
  else
    print "Starting local ML on all devices using experiment_config.py defaults..."
  fi
  extra_options="$reference_option $method_option $(bg_args_for_python)"
  for device in "${(@f)$(device_lines)}"; do
    local client_id="${device%%:*}"
    local host="${device#*:}"
    ssh_run "$host" "
      cd '$REMOTE_PROJECT_DIR' &&
      '$REMOTE_PYTHON' running_ml.py \
        --client-id '$client_id' \
        --device-id '$host' \
        --host '$host' $extra_options
    " &
  done
  wait
  print "Local ML finished."
}

dry_run_fl() {
  local methods="${1:-}"
  local -a method_args bg_args
  bg_args=()
  if [[ "$BG_WORKLOAD_ENABLED" == "1" ]]; then
    bg_args=(--background-workload-enabled --background-workload-group "$BG_WORKLOAD_GROUP" --background-workload-profile "$BG_WORKLOAD_PROFILE")
  fi
  if [[ -n "$methods" ]]; then
    method_args=(--poisoning-methods "$methods")
    print "Previewing FL SSH commands for attack conditions: $methods"
  else
    method_args=()
    print "Previewing FL SSH commands using experiment_config.py defaults..."
  fi
  cd "$SERVER_PROJECT_DIR"
  "$SERVER_PYTHON" running_fl.py \
    --dry-run \
    --ssh-password "$SSH_PASSWORD" \
    "${bg_args[@]}" \
    "${method_args[@]}"
}

run_fl() {
  local methods="${1:-}"
  local -a method_args bg_args
  bg_args=()
  if [[ "$BG_WORKLOAD_ENABLED" == "1" ]]; then
    bg_args=(--background-workload-enabled --background-workload-group "$BG_WORKLOAD_GROUP" --background-workload-profile "$BG_WORKLOAD_PROFILE")
  fi
  if [[ -n "$methods" ]]; then
    method_args=(--poisoning-methods "$methods")
    print "Starting FL experiments for attack conditions: $methods"
  else
    method_args=()
    print "Starting FL experiments using experiment_config.py defaults..."
  fi
  cd "$SERVER_PROJECT_DIR"
  "$SERVER_PYTHON" running_fl.py \
    --ssh-password "$SSH_PASSWORD" \
    --server-log-hardware \
    "${bg_args[@]}" \
    "${method_args[@]}"
  print "FL finished."
}

usage() {
  cat <<'EOF'
Usage:
  ./run_experiments.zsh check
  ./run_experiments.zsh ml [conditions]
  ./run_experiments.zsh fl-dry-run [attack_conditions]
  ./run_experiments.zsh fl [attack_conditions]
  ./run_experiments.zsh both [conditions]
  ./run_experiments.zsh [conditions]

Examples:
  ./run_experiments.zsh ml random_label_flipping,target_label_flipping
  ./run_experiments.zsh ml clean unlearnable_examples
  ./run_experiments.zsh ml availability_shortcuts
  ./run_experiments.zsh fl target_label_flipping
  ./run_experiments.zsh fl availability_shortcuts
  ./run_experiments.zsh random_label_flipping target_label_flipping

Experiment settings are read from experiment_config.py.
For password-based SSH:
  export SSH_PASSWORD='your_password'
Background workload defaults for this bg-noise directory:
  BG_WORKLOAD_ENABLED=1
  BG_WORKLOAD_GROUP=both
  BG_WORKLOAD_PROFILE=medium
  BG_WORKLOAD_TEST_DURATION=10
  BG_WORKLOAD_PYTHON=/home/rasheed/kuchida/antivenom_infocom/venv/bin/python
Do not run local ML and FL at the same time.
EOF
}

main() {
  local mode="${1:-}"
  local -a condition_args
  case "$mode" in
    check|ml|fl-dry-run|fl|both|"")
      shift || true
      condition_args=("$@")
      ;;
    *)
      condition_args=("$@")
      mode="ml"
      ;;
  esac

  local ml_methods=""
  local fl_methods=""

  case "$mode" in
    check)
      check_remote_python
      check_bg_workloads
      ;;
    ml)
      if (( ${#condition_args[@]} > 0 )); then
        ml_methods="$(normalize_conditions yes "${condition_args[@]}")"
      fi
      pull_remote_repos
      check_remote_python
      run_with_bg_workloads run_local_ml "$ml_methods"
      ;;
    fl-dry-run)
      if (( ${#condition_args[@]} > 0 )); then
        fl_methods="$(normalize_conditions no "${condition_args[@]}")"
        if [[ -z "$fl_methods" ]]; then
          print "No FL attack conditions selected." >&2
          exit 1
        fi
      fi
      dry_run_fl "$fl_methods"
      ;;
    fl)
      if (( ${#condition_args[@]} > 0 )); then
        fl_methods="$(normalize_conditions no "${condition_args[@]}")"
        if [[ -z "$fl_methods" ]]; then
          print "No FL attack conditions selected." >&2
          exit 1
        fi
      fi
      pull_remote_repos
      dry_run_fl "$fl_methods"
      run_with_bg_workloads run_fl "$fl_methods"
      ;;
    both)
      if (( ${#condition_args[@]} > 0 )); then
        ml_methods="$(normalize_conditions yes "${condition_args[@]}")"
        fl_methods="$(normalize_conditions no "${condition_args[@]}")"
        if [[ -z "$fl_methods" ]]; then
          print "No FL attack conditions selected." >&2
          exit 1
        fi
      fi
      pull_remote_repos
      check_remote_python
      run_with_bg_workloads run_local_ml "$ml_methods"
      pull_remote_repos
      dry_run_fl "$fl_methods"
      run_with_bg_workloads run_fl "$fl_methods"
      ;;
    *)
      usage
      exit 1
      ;;
  esac
}

main "$@"
