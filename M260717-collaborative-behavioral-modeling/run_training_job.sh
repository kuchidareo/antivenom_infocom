#!/usr/bin/env bash
set -uo pipefail

umask 022

readonly ROOT="${ANTIVENOM_ROOT:-/local/antivenom}"
readonly TRAINING_STATE_DIR="${ANTIVENOM_TRAINING_STATE_DIR:-${ROOT}/state/training}"
readonly RUNNING_FILE="${TRAINING_STATE_DIR}/running"
readonly DONE_FILE="${TRAINING_STATE_DIR}/done"
readonly FAILED_FILE="${TRAINING_STATE_DIR}/failed"
readonly PID_FILE="${TRAINING_STATE_DIR}/pid"
readonly LOCK_FILE="${TRAINING_STATE_DIR}/lock"

readonly LOCAL_EXPERIMENT_RUNNER="${LOCAL_EXPERIMENT_RUNNER:?LOCAL_EXPERIMENT_RUNNER is required}"
readonly TRAINING_METHODS="${TRAINING_METHODS:-clean,availability_shortcuts}"

mkdir -p "${TRAINING_STATE_DIR}"

# Only one training process may own this state directory at a time.
exec 9>"${LOCK_FILE}"
if ! flock -n 9; then
    echo "Training is already running; lock is held: ${LOCK_FILE}"
    exit 75
fi

write_state() {
    local destination="$1"
    local status="$2"
    local exit_code="${3:-}"
    local temporary="${destination}.tmp.$$"

    {
        printf 'status=%s\n' "${status}"
        printf 'pid=%s\n' "$$"
        printf 'hostname=%s\n' "$(hostname 2>/dev/null || true)"
        printf 'training_methods=%s\n' "${TRAINING_METHODS}"
        printf 'timestamp=%s\n' "$(date --iso-8601=seconds)"
        if [[ -n "${exit_code}" ]]; then
            printf 'exit_code=%s\n' "${exit_code}"
        fi
    } > "${temporary}"

    mv -f "${temporary}" "${destination}"
}

finish() {
    local exit_code=$?

    trap - EXIT
    rm -f "${RUNNING_FILE}" "${PID_FILE}"

    if [[ "${exit_code}" -eq 0 ]]; then
        rm -f "${FAILED_FILE}"
        write_state "${DONE_FILE}" done "${exit_code}"
        echo "Training completed: $(date --iso-8601=seconds)"
    else
        rm -f "${DONE_FILE}"
        write_state "${FAILED_FILE}" failed "${exit_code}"
        echo "Training failed with exit code ${exit_code}: $(date --iso-8601=seconds)" >&2
    fi

    exit "${exit_code}"
}

# nohup already ignores SIGHUP. Preserve that behavior for the full job.
trap '' HUP
trap 'exit 130' INT
trap 'exit 143' TERM
trap finish EXIT

rm -f "${DONE_FILE}" "${FAILED_FILE}"
printf '%s\n' "$$" > "${PID_FILE}.tmp.$$"
mv -f "${PID_FILE}.tmp.$$" "${PID_FILE}"
write_state "${RUNNING_FILE}" running

echo "Training started: $(date --iso-8601=seconds)"
echo "Training PID: $$"
echo "Training methods: ${TRAINING_METHODS}"
echo "Experiment runner: ${LOCAL_EXPERIMENT_RUNNER}"

zsh "${LOCAL_EXPERIMENT_RUNNER}" run "${TRAINING_METHODS}"
