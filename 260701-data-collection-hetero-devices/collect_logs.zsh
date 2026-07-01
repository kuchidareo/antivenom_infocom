#!/usr/bin/env zsh
set -euo pipefail

# Server-side project path.
SERVER_PROJECT_DIR="/home/user/kuchida/antivenom_infocom/260701-data-collection-hetero-devices"
SERVER_PYTHON="${PYTHON:-${SERVER_PROJECT_DIR:h}/venv/bin/python}"

# SSH credentials. Password is read from SSH_PASSWORD env var.
SSH_PASSWORD="${SSH_PASSWORD:-}"

DELETE_REMOTE=0
DEST_DIR="${SERVER_PROJECT_DIR}/collected_logs"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --delete-remote)
      DELETE_REMOTE=1
      shift
      ;;
    --help|-h)
      cat <<'EOF'
Usage:
  ./collect_logs.zsh [DEST_DIR]
  ./collect_logs.zsh --delete-remote [DEST_DIR]

By default, logs are only copied. With --delete-remote, each Raspberry Pi's
remote logs/ directory is removed and recreated after that device's rsync
finishes successfully.
EOF
      exit 0
      ;;
    *)
      DEST_DIR="$1"
      shift
      ;;
  esac
done

device_lines() {
  cd "$SERVER_PROJECT_DIR"
  "$SERVER_PYTHON" -c "import experiment_config as c; [print('|'.join([d['host'], c.device_ssh_user(d), c.device_remote_project_dir(d)])) for d in c.DEVICES]"
}

rsync_remote() {
  local host="$1"
  local ssh_user="$2"
  local remote_project_dir="$3"
  local src="${ssh_user}@${host}:${remote_project_dir}/logs/"
  local dst="${DEST_DIR}/${host}/"
  mkdir -p "$dst"
  if [[ -n "$SSH_PASSWORD" ]]; then
    if ! command -v sshpass >/dev/null 2>&1; then
      print "SSH_PASSWORD is set, but sshpass is not installed." >&2
      print "Install sshpass or configure SSH keys." >&2
      exit 1
    fi
    sshpass -p "$SSH_PASSWORD" rsync -av \
      -e "ssh -o StrictHostKeyChecking=accept-new" \
      "$src" "$dst"
  else
    rsync -av "$src" "$dst"
  fi
}

ssh_remote() {
  local host="$1"
  local ssh_user="$2"
  local remote_command="$3"
  local target="${ssh_user}@${host}"
  if [[ -n "$SSH_PASSWORD" ]]; then
    sshpass -p "$SSH_PASSWORD" ssh -o StrictHostKeyChecking=accept-new "$target" "$remote_command"
  else
    ssh "$target" "$remote_command"
  fi
}

delete_remote_logs() {
  local host="$1"
  local ssh_user="$2"
  local remote_project_dir="$3"
  ssh_remote "$host" "$ssh_user" "rm -rf '${remote_project_dir}/logs' && mkdir -p '${remote_project_dir}/logs'"
}

collect_server_logs() {
  local dst="${DEST_DIR}/server/"
  mkdir -p "$dst"
  rsync -av "${SERVER_PROJECT_DIR}/logs/" "$dst"
}

main() {
  mkdir -p "$DEST_DIR"
  print "Collecting server logs into ${DEST_DIR}/server/"
  collect_server_logs

  for device in "${(@f)$(device_lines)}"; do
    local -a fields
    fields=("${(@ps:|:)device}")
    local host="${fields[1]}"
    local ssh_user="${fields[2]}"
    local remote_project_dir="${fields[3]}"
    print "Collecting logs from ${ssh_user}@${host} into ${DEST_DIR}/${host}/"
    rsync_remote "$host" "$ssh_user" "$remote_project_dir"
    if [[ "$DELETE_REMOTE" -eq 1 ]]; then
      print "Deleting remote logs on ${host}"
      delete_remote_logs "$host" "$ssh_user" "$remote_project_dir"
    fi
  done

  print "Log collection finished: ${DEST_DIR}"
  if [[ "$DELETE_REMOTE" -eq 1 ]]; then
    print "Remote Raspberry Pi logs were deleted after successful rsync."
  fi
}

main "$@"
