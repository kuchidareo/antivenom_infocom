#!/usr/bin/env zsh
set -euo pipefail

# Server-side project path.
SERVER_PROJECT_DIR="${0:A:h}"

# Remote experiment project path.
REMOTE_PROJECT_DIR="/home/rasheed/kuchida/antivenom_infocom/260724-modrl-or-input-analysis"

# SSH credentials. Password is read from SSH_PASSWORD env var.
SSH_USER="rasheed"
SSH_PASSWORD="${SSH_PASSWORD:-}"
PING_TIMEOUT_SEC="${PING_TIMEOUT_SEC:-1}"

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

By default, logs and trained models are copied. With --delete-remote, each device's
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

DEVICES=(
  "192.168.0.141"
  "192.168.0.142"
)

host_is_reachable() {
  local host="$1"
  ping -c 1 -W "$PING_TIMEOUT_SEC" "$host" >/dev/null 2>&1
}

rsync_remote_tree() {
  local host="$1"
  local tree="$2"
  local src="${SSH_USER}@${host}:${REMOTE_PROJECT_DIR}/${tree}/"
  local dst="${DEST_DIR}/${host}/${tree}/"
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
  local remote_command="$2"
  local target="${SSH_USER}@${host}"
  if [[ -n "$SSH_PASSWORD" ]]; then
    sshpass -p "$SSH_PASSWORD" ssh -o StrictHostKeyChecking=accept-new "$target" "$remote_command"
  else
    ssh "$target" "$remote_command"
  fi
}

delete_remote_logs() {
  local host="$1"
  ssh_remote "$host" "rm -rf '${REMOTE_PROJECT_DIR}/logs' && mkdir -p '${REMOTE_PROJECT_DIR}/logs'"
}

collect_server_logs() {
  local tree
  for tree in logs models; do
    if [[ -d "${SERVER_PROJECT_DIR}/${tree}" ]]; then
      mkdir -p "${DEST_DIR}/server/${tree}"
      rsync -av "${SERVER_PROJECT_DIR}/${tree}/" "${DEST_DIR}/server/${tree}/"
    fi
  done
}

main() {
  mkdir -p "$DEST_DIR"
  print "Collecting server logs into ${DEST_DIR}/server/"
  collect_server_logs

  for host in "${DEVICES[@]}"; do
    if ! host_is_reachable "$host"; then
      print "Skipping unreachable device: ${host}" >&2
      continue
    fi
    print "Collecting logs and models from ${host} into ${DEST_DIR}/${host}/"
    rsync_remote_tree "$host" logs
    rsync_remote_tree "$host" models
    if [[ "$DELETE_REMOTE" -eq 1 ]]; then
      print "Deleting remote logs on ${host}"
      delete_remote_logs "$host"
    fi
  done

  print "Log collection finished: ${DEST_DIR}"
  if [[ "$DELETE_REMOTE" -eq 1 ]]; then
    print "Remote device logs were deleted after successful rsync."
  fi
}

main "$@"
