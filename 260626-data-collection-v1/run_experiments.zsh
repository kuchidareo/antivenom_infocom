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

run_local_ml() {
  print "Starting local ML on all devices..."
  for device in "${(@f)$(device_lines)}"; do
    local client_id="${device%%:*}"
    local host="${device#*:}"
    ssh_run "$host" "
      cd '$REMOTE_PROJECT_DIR' &&
      '$REMOTE_PYTHON' running_ml.py \
        --client-id '$client_id' \
        --device-id '$host' \
        --host '$host'
    " &
  done
  wait
  print "Local ML finished."
}

dry_run_fl() {
  print "Previewing FL SSH commands..."
  cd "$SERVER_PROJECT_DIR"
  "$SERVER_PYTHON" running_fl.py \
    --dry-run \
    --ssh-password "$SSH_PASSWORD"
}

run_fl() {
  print "Starting FL experiments..."
  cd "$SERVER_PROJECT_DIR"
  "$SERVER_PYTHON" running_fl.py \
    --ssh-password "$SSH_PASSWORD" \
    --server-log-hardware
  print "FL finished."
}

usage() {
  cat <<'EOF'
Usage:
  ./run_experiments.zsh check
  ./run_experiments.zsh ml
  ./run_experiments.zsh fl-dry-run
  ./run_experiments.zsh fl
  ./run_experiments.zsh both

Experiment settings are read from experiment_config.py.
For password-based SSH:
  export SSH_PASSWORD='your_password'
Do not run local ML and FL at the same time.
EOF
}

main() {
  local mode="${1:-}"
  case "$mode" in
    check)
      check_remote_python
      ;;
    ml)
      check_remote_python
      pull_remote_repos
      run_local_ml
      ;;
    fl-dry-run)
      dry_run_fl
      ;;
    fl)
      dry_run_fl
      pull_remote_repos
      run_fl
      ;;
    both)
      check_remote_python
      pull_remote_repos
      run_local_ml
      dry_run_fl
      pull_remote_repos
      run_fl
      ;;
    *)
      usage
      exit 1
      ;;
  esac
}

main "$@"
