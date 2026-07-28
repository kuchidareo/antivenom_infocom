#!/usr/bin/env zsh
set -u
set -o pipefail

SCRIPT_DIR="${0:A:h}"
REMOTE_REPO_DIR="${REMOTE_REPO_DIR:-/home/rasheed/kuchida/antivenom_infocom}"
SSH_USER="${SSH_USER:-rasheed}"
SSH_PASSWORD="${SSH_PASSWORD:-modenaottun}"
SSH_PORT="${SSH_PORT:-22}"
CONNECT_TIMEOUT_SEC="${CONNECT_TIMEOUT_SEC:-5}"

DATASET_DIRS=(iid-data non-iid-data)
DEFAULT_DEVICES=(
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
  192.168.0.141
  192.168.0.142
)

if (( $# > 0 )); then
  DEVICES=("$@")
else
  DEVICES=("${DEFAULT_DEVICES[@]}")
fi

for dataset_dir in "${DATASET_DIRS[@]}"; do
  if [[ ! -d "${SCRIPT_DIR}/${dataset_dir}" ]]; then
    echo "Missing local dataset directory: ${SCRIPT_DIR}/${dataset_dir}" >&2
    exit 1
  fi
done

if ! command -v rsync >/dev/null 2>&1; then
  echo "rsync is required." >&2
  exit 1
fi

if command -v sshpass >/dev/null 2>&1; then
  SSH_CMD=(
    sshpass -p "${SSH_PASSWORD}" ssh
    -p "${SSH_PORT}"
    -o ConnectTimeout="${CONNECT_TIMEOUT_SEC}"
    -o StrictHostKeyChecking=no
    -o UserKnownHostsFile=/dev/null
  )
  RSYNC_RSH="sshpass -p ${SSH_PASSWORD} ssh -p ${SSH_PORT} -o ConnectTimeout=${CONNECT_TIMEOUT_SEC} -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null"
else
  echo "sshpass is unavailable; SSH may prompt for a password." >&2
  SSH_CMD=(
    ssh
    -p "${SSH_PORT}"
    -o ConnectTimeout="${CONNECT_TIMEOUT_SEC}"
    -o StrictHostKeyChecking=no
    -o UserKnownHostsFile=/dev/null
  )
  RSYNC_RSH="ssh -p ${SSH_PORT} -o ConnectTimeout=${CONNECT_TIMEOUT_SEC} -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null"
fi

failed_hosts=()
skipped_hosts=()

for host in "${DEVICES[@]}"; do
  remote="${SSH_USER}@${host}"
  echo "==> ${remote}"

  if ! "${SSH_CMD[@]}" "${remote}" true >/dev/null 2>&1; then
    echo "    skip: SSH connection unavailable"
    skipped_hosts+=("${host}")
    continue
  fi

  host_failed=0
  for dataset_dir in "${DATASET_DIRS[@]}"; do
    local_dir="${SCRIPT_DIR}/${dataset_dir}"
    remote_dir="${REMOTE_REPO_DIR}/${dataset_dir}"
    echo "    syncing ${remote_dir}"

    if ! "${SSH_CMD[@]}" "${remote}" \
      "mkdir -p '${remote_dir}'"; then
      echo "    error: could not create ${remote_dir}" >&2
      host_failed=1
      break
    fi

    if ! rsync -az --partial --delay-updates --delete-delay \
      -e "${RSYNC_RSH}" \
      "${local_dir}/" \
      "${remote}:${remote_dir}/"; then
      echo "    error: rsync failed for ${dataset_dir}" >&2
      host_failed=1
      break
    fi

    remote_count="$(
      "${SSH_CMD[@]}" "${remote}" \
        "find '${remote_dir}' -type f -name '*.jpeg' | wc -l" \
        | tr -d '[:space:]'
    )"
    expected_count="$(
      find "${local_dir}" -type f -name '*.jpeg' | wc -l | tr -d '[:space:]'
    )"
    if [[ "${remote_count}" != "${expected_count}" ]]; then
      echo "    error: ${dataset_dir} has ${remote_count} JPEGs; expected ${expected_count}" >&2
      host_failed=1
      break
    fi
    echo "    ${dataset_dir}: ${remote_count} JPEGs verified"
  done

  if (( host_failed )); then
    failed_hosts+=("${host}")
  else
    echo "    done"
  fi
done

echo
echo "Sync summary"
echo "  completed: $(( ${#DEVICES[@]} - ${#failed_hosts[@]} - ${#skipped_hosts[@]} ))"
echo "  skipped:   ${#skipped_hosts[@]} ${skipped_hosts[*]:-}"
echo "  failed:    ${#failed_hosts[@]} ${failed_hosts[*]:-}"

if (( ${#failed_hosts[@]} > 0 )); then
  exit 1
fi
