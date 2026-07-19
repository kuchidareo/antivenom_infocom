#!/usr/bin/env zsh
set -euo pipefail

PROJECT_DIR="${0:A:h}"
REMOTE_PROJECT_DIR="${REMOTE_PROJECT_DIR:-/home/rasheed/kuchida/antivenom_infocom/M260719-robustness-library}"
SSH_USER="${SSH_USER:-rasheed}"
SSH_PASSWORD="${SSH_PASSWORD:-}"
SSH_PORT="${SSH_PORT:-22}"
PING_TIMEOUT_SEC="${PING_TIMEOUT_SEC:-1}"
DEST_DIR="${PROJECT_DIR}/collected_logs"
DELETE_REMOTE=0
DEVICES=(192.168.0.141 192.168.0.142)

while (( $# > 0 )); do
  case "$1" in
    --delete-remote)
      DELETE_REMOTE=1
      ;;
    -h|--help)
      print "Usage: ./collect_logs.zsh [--delete-remote] [DEST_DIR]"
      exit 0
      ;;
    *)
      DEST_DIR="$1"
      ;;
  esac
  shift
done

ssh_target() {
  print -- "${SSH_USER}@$1"
}

ssh_run() {
  local host="$1"
  local command="$2"
  if [[ -n "$SSH_PASSWORD" ]]; then
    sshpass -p "$SSH_PASSWORD" ssh -p "$SSH_PORT" \
      -o StrictHostKeyChecking=accept-new "$(ssh_target "$host")" "$command"
  else
    ssh -p "$SSH_PORT" -o StrictHostKeyChecking=accept-new \
      "$(ssh_target "$host")" "$command"
  fi
}

for host in "${DEVICES[@]}"; do
  if ! ping -c 1 -W "$PING_TIMEOUT_SEC" "$host" >/dev/null 2>&1; then
    print -u2 "Skipping unreachable device: $host"
    continue
  fi
  destination="${DEST_DIR}/${host}/"
  mkdir -p "$destination"
  print "Collecting $host into $destination"
  if [[ -n "$SSH_PASSWORD" ]]; then
    sshpass -p "$SSH_PASSWORD" rsync -av \
      -e "ssh -p $SSH_PORT -o StrictHostKeyChecking=accept-new" \
      "$(ssh_target "$host"):${REMOTE_PROJECT_DIR}/logs/" "$destination"
  else
    rsync -av -e "ssh -p $SSH_PORT -o StrictHostKeyChecking=accept-new" \
      "$(ssh_target "$host"):${REMOTE_PROJECT_DIR}/logs/" "$destination"
  fi
  if (( DELETE_REMOTE )); then
    ssh_run "$host" "rm -rf '${REMOTE_PROJECT_DIR}/logs' && mkdir -p '${REMOTE_PROJECT_DIR}/logs'"
  fi
done

print "Log collection finished: $DEST_DIR"
