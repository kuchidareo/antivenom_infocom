#!/usr/bin/env zsh
set -euo pipefail

SCRIPT_DIR="${0:A:h}"
REPO_DIR="${SCRIPT_DIR:h}"
LOCAL_PYTHON="${PYTHON:-${REPO_DIR}/venv/bin/python}"

HOST="192.168.0.112"
CLIENT_ID="client_0"
DATASET="kuchidareo/small_trashnet"
LOCAL_EPOCHS=10
TRIALS=1
REFERENCE_TRIALS=0
CPU_FREQ_SAMPLE_MS="${CPU_FREQ_SAMPLE_MS:-1}"
PING_TIMEOUT_SEC="${PING_TIMEOUT_SEC:-1}"
SSH_PASSWORD="${SSH_PASSWORD:-}"

usage() {
  cat <<'EOF'
Usage:
  ./run_small_experiment.zsh [conditions]

Runs local ML only on 192.168.0.112 with small_trashnet, 10 epochs,
no reference runs, and one analysis trial per condition.

Default conditions:
  clean,unlearnable_examples,availability_shortcuts

Examples:
  ./run_small_experiment.zsh
  ./run_small_experiment.zsh clean
  ./run_small_experiment.zsh clean,availability_shortcuts

Environment:
  SSH_PASSWORD=...          optional when SSH keys are configured
  CPU_FREQ_SAMPLE_MS=1     internal frequency sampling interval
  PING_TIMEOUT_SEC=1
EOF
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  usage
  exit 0
fi

METHODS="${1:-clean,unlearnable_examples,availability_shortcuts}"

config_value() {
  local name="$1"
  cd "$SCRIPT_DIR"
  "$LOCAL_PYTHON" -c "import experiment_config as c; print(getattr(c, '$name'))"
}

REMOTE_PROJECT_DIR="$(config_value DEFAULT_REMOTE_PROJECT_DIR)"
REMOTE_REPO_DIR="${REMOTE_PROJECT_DIR:h}"
REMOTE_PYTHON="$(config_value DEFAULT_REMOTE_PYTHON)"
SSH_USER="$(config_value DEFAULT_SSH_USER)"
SSH_TARGET="${SSH_USER}@${HOST}"
REMOTE_DATASET_DIR="${REMOTE_REPO_DIR}/iid-data/small_trashnet"

ssh_run() {
  local remote_command="$1"
  if [[ -n "$SSH_PASSWORD" ]]; then
    if ! command -v sshpass >/dev/null 2>&1; then
      print "SSH_PASSWORD is set, but sshpass is not installed." >&2
      exit 1
    fi
    sshpass -p "$SSH_PASSWORD" ssh \
      -o StrictHostKeyChecking=accept-new \
      "$SSH_TARGET" \
      "$remote_command"
  else
    ssh "$SSH_TARGET" "$remote_command"
  fi
}

if ! ping -c 1 -W "$PING_TIMEOUT_SEC" "$HOST" >/dev/null 2>&1; then
  print "Device is unreachable; no experiment was run: ${HOST}" >&2
  exit 1
fi

print "Updating the remote repository..."
ssh_run "cd '$REMOTE_REPO_DIR' && git pull --rebase"

print "Checking the remote environment..."
ssh_run "
  set -e
  test -d '$REMOTE_PROJECT_DIR'
  test -x '$REMOTE_PYTHON'
  test -d '$REMOTE_DATASET_DIR'
  '$REMOTE_PYTHON' --version
  '$REMOTE_PYTHON' -c 'import torch, torchvision, psutil, numpy, PIL; print("dependencies: ok")'
"

print "Running small local-ML experiment:"
print "  device: ${SSH_TARGET} (${CLIENT_ID})"
print "  dataset: ${DATASET}"
print "  methods: ${METHODS}"
print "  epochs: ${LOCAL_EPOCHS}"
print "  analysis trials per method: ${TRIALS}"
print "  reference trials: ${REFERENCE_TRIALS}"
print "  CPU-frequency internal sampling: ${CPU_FREQ_SAMPLE_MS} ms"
print "  saved hardware frequency: 10 FPS window average"

ssh_run "
  set -e
  cd '$REMOTE_PROJECT_DIR'
  '$REMOTE_PYTHON' running_ml.py \
    --dataset '$DATASET' \
    --client-id '$CLIENT_ID' \
    --device-id '$HOST' \
    --host '$HOST' \
    --local-epochs '$LOCAL_EPOCHS' \
    --reference-trials '$REFERENCE_TRIALS' \
    --trials '$TRIALS' \
    --poisoning-method '$METHODS' \
    --cpu-freq-sample-ms '$CPU_FREQ_SAMPLE_MS'
"

print "Small experiment finished on ${HOST}."
print "Remote logs: ${REMOTE_PROJECT_DIR}/logs/local_ml"
