#!/usr/bin/env zsh
set -euo pipefail

SCRIPT_DIR="${0:A:h}"
LOCAL_DATASET_DIR="${LOCAL_DATASET_DIR:-${SCRIPT_DIR:h}/iid-data/small_trashnet}"
REMOTE_PROJECT_DIR="${REMOTE_PROJECT_DIR:-/home/rasheed/kuchida/antivenom_infocom/M260719-robustness-library}"
REMOTE_DATASET_DIR="${REMOTE_DATASET_DIR:-${REMOTE_PROJECT_DIR:h}/iid-data/small_trashnet}"
SSH_USER="${SSH_USER:-rasheed}"
SSH_PASSWORD="${SSH_PASSWORD:-}"
SSH_PORT="${SSH_PORT:-22}"
PING_TIMEOUT_SEC="${PING_TIMEOUT_SEC:-1}"
DEVICES=(192.168.0.141 192.168.0.142)

[[ -d "$LOCAL_DATASET_DIR" ]] || {
  print -u2 "Local dataset does not exist: $LOCAL_DATASET_DIR"
  exit 1
}
command -v rsync >/dev/null || {
  print -u2 "rsync is required."
  exit 1
}

for host in "${DEVICES[@]}"; do
  if ! ping -c 1 -W "$PING_TIMEOUT_SEC" "$host" >/dev/null 2>&1; then
    print -u2 "Skipping unreachable device: $host"
    continue
  fi
  target="${SSH_USER}@${host}"
  print "Synchronizing small_trashnet to $host"
  if [[ -n "$SSH_PASSWORD" ]]; then
    sshpass -p "$SSH_PASSWORD" ssh -p "$SSH_PORT" \
      -o StrictHostKeyChecking=accept-new "$target" \
      "mkdir -p '$REMOTE_DATASET_DIR'"
    sshpass -p "$SSH_PASSWORD" rsync -az --delete \
      -e "ssh -p $SSH_PORT -o StrictHostKeyChecking=accept-new" \
      "${LOCAL_DATASET_DIR}/" "${target}:${REMOTE_DATASET_DIR}/"
  else
    ssh -p "$SSH_PORT" -o StrictHostKeyChecking=accept-new "$target" \
      "mkdir -p '$REMOTE_DATASET_DIR'"
    rsync -az --delete \
      -e "ssh -p $SSH_PORT -o StrictHostKeyChecking=accept-new" \
      "${LOCAL_DATASET_DIR}/" "${target}:${REMOTE_DATASET_DIR}/"
  fi
done
