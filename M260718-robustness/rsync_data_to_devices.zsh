#!/usr/bin/env zsh
set -euo pipefail

SCRIPT_DIR="${0:A:h}"
LOCAL_DATA_DIR="${LOCAL_DATA_DIR:-${SCRIPT_DIR:h}/iid-data}"
REMOTE_PROJECT_DIR="${REMOTE_PROJECT_DIR:-/home/rasheed/kuchida/antivenom_infocom/M260718-robustness}"
REMOTE_DATA_DIR="${REMOTE_DATA_DIR:-${REMOTE_PROJECT_DIR:h}/iid-data}"
SSH_USER="${SSH_USER:-rasheed}"
SSH_PASSWORD="${SSH_PASSWORD:-modenaottun}"
SSH_PORT="${SSH_PORT:-22}"
PING_TIMEOUT_SEC="${PING_TIMEOUT_SEC:-1}"

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

host_is_reachable() {
  local host="$1"
  ping -c 1 -W "${PING_TIMEOUT_SEC}" "${host}" >/dev/null 2>&1
}

echo "Local data:  ${LOCAL_DATA_DIR}/"
echo "Remote data: ${REMOTE_DATA_DIR}/"
echo

for HOST in "${DEVICES[@]}"; do
  if ! host_is_reachable "${HOST}"; then
    echo "==> ${SSH_USER}@${HOST}"
    echo "skip: host is unreachable"
    continue
  fi

  REMOTE="${SSH_USER}@${HOST}"
  echo "==> ${REMOTE}"

  "${SSH_CMD[@]}" "${REMOTE}" \
    "set -e; rm -rf '${REMOTE_DATA_DIR}'; mkdir -p '${REMOTE_DATA_DIR}'"

  rsync -az --delete \
    -e "${RSYNC_RSH}" \
    "${LOCAL_DATA_DIR}/" \
    "${REMOTE}:${REMOTE_DATA_DIR}/"

  echo "done: ${HOST}"
done

echo
echo "All device data directories were replaced and synced."
