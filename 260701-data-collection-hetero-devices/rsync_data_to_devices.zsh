#!/usr/bin/env zsh
set -euo pipefail

SCRIPT_DIR="${0:A:h}"
SERVER_PYTHON="${PYTHON:-${SCRIPT_DIR:h}/venv/bin/python}"
LOCAL_DATA_DIR="${LOCAL_DATA_DIR:-${SCRIPT_DIR}/data}"
SSH_PASSWORD="${SSH_PASSWORD:-modenaottun}"
SSH_PORT="${SSH_PORT:-22}"

device_lines() {
  cd "$SCRIPT_DIR"
  "$SERVER_PYTHON" -c "import experiment_config as c; [print('|'.join([d['host'], c.device_ssh_user(d), c.device_remote_project_dir(d)])) for d in c.DEVICES]"
}

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
echo "Remote data: per-device project data/ from experiment_config.py"
echo

for device in "${(@f)$(device_lines)}"; do
  fields=("${(@ps:|:)device}")
  HOST="${fields[1]}"
  SSH_USER="${fields[2]}"
  REMOTE_PROJECT_DIR="${fields[3]}"
  REMOTE_DATA_DIR="${REMOTE_PROJECT_DIR}/data"
  REMOTE="${SSH_USER}@${HOST}"
  echo "==> ${REMOTE}"
  echo "    ${REMOTE_DATA_DIR}/"

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
