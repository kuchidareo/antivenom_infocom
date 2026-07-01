#!/usr/bin/env zsh
set -euo pipefail

# Server-side project path.
SERVER_PROJECT_DIR="/home/user/kuchida/antivenom_infocom/260701-data-collection-availability-attack-non-iid"

# Remote Raspberry Pi project path.
REMOTE_PROJECT_DIR="/home/rasheed/kuchida/antivenom_infocom/260701-data-collection-availability-attack-non-iid"

# SSH credentials. Password is read from SSH_PASSWORD env var.
SSH_USER="rasheed"
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

DEVICES=(
  "192.168.0.112"
  "192.168.0.113"
  "192.168.0.114"
  "192.168.0.115"
  "192.168.0.116"
  "192.168.0.117"
  "192.168.0.118"
  "192.168.0.119"
  "192.168.0.120"
  "192.168.0.121"
)

rsync_remote() {
  local host="$1"
  local src="${SSH_USER}@${host}:${REMOTE_PROJECT_DIR}/logs/"
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
  local dst="${DEST_DIR}/server/"
  mkdir -p "$dst"
  rsync -av "${SERVER_PROJECT_DIR}/logs/" "$dst"
}

main() {
  mkdir -p "$DEST_DIR"
  print "Collecting server logs into ${DEST_DIR}/server/"
  collect_server_logs

  for host in "${DEVICES[@]}"; do
    print "Collecting logs from ${host} into ${DEST_DIR}/${host}/"
    rsync_remote "$host"
    if [[ "$DELETE_REMOTE" -eq 1 ]]; then
      print "Deleting remote logs on ${host}"
      delete_remote_logs "$host"
    fi
  done

  print "Log collection finished: ${DEST_DIR}"
  if [[ "$DELETE_REMOTE" -eq 1 ]]; then
    print "Remote Raspberry Pi logs were deleted after successful rsync."
  fi
}

main "$@"
