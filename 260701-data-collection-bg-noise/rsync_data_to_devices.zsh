#!/usr/bin/env zsh
set -euo pipefail

SCRIPT_DIR="${0:A:h}"
LOCAL_DATA_DIR="${LOCAL_DATA_DIR:-${SCRIPT_DIR}/data}"
REMOTE_PROJECT_DIR="${REMOTE_PROJECT_DIR:-/home/rasheed/kuchida/antivenom_infocom/260701-data-collection-bg-noise}"
REMOTE_DATA_DIR="${REMOTE_PROJECT_DIR}/data"
SSH_USER="${SSH_USER:-rasheed}"
SSH_PASSWORD="${SSH_PASSWORD:-modenaottun}"
SSH_PORT="${SSH_PORT:-22}"

DEVICES=(
  192.168.0.112
  192.168.0.113
  192.168.0.114
  192.168.0.115
  192.168.0.116
  192.168.0.117
  192.168.0.118
  192.168.0.119
  192.168.0.120
  192.168.0.121
)

if [[ ! -d "${LOCAL_DATA_DIR}" ]]; then
  echo "Local data directory does not exist: ${LOCAL_DATA_DIR}" >&2
  exit 1
fi

if ! command -v rsync >/dev/null 2>&1; then
  echo "rsync is required but was not found." >&2
  exit 1
fi

if command -v sshpass >/dev/null 2>&1; then
  SSH_CMD=(sshpass -p "${SSH_PASSWORD}" ssh -p "${SSH_PORT}" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null)
  RSYNC_RSH="sshpass -p ${SSH_PASSWORD} ssh -p ${SSH_PORT} -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null"
else
  echo "sshpass was not found. You may be prompted for the SSH password for each device." >&2
  SSH_CMD=(ssh -p "${SSH_PORT}" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null)
  RSYNC_RSH="ssh -p ${SSH_PORT} -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null"
fi

echo "Local data:  ${LOCAL_DATA_DIR}/"
echo "Remote data: ${REMOTE_DATA_DIR}/"
echo

for HOST in "${DEVICES[@]}"; do
  REMOTE="${SSH_USER}@${HOST}"
  echo "==> ${REMOTE}"

  "${SSH_CMD[@]}" "${REMOTE}" \
    "set -e; cd '${REMOTE_PROJECT_DIR}'; rm -rf data; mkdir -p data"

  rsync -az --delete \
    -e "${RSYNC_RSH}" \
    "${LOCAL_DATA_DIR}/" \
    "${REMOTE}:${REMOTE_DATA_DIR}/"

  echo "done: ${HOST}"
done

echo
echo "All device data directories were replaced and synced."
