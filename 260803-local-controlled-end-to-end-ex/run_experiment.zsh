#!/usr/bin/env zsh
set -euo pipefail

SCRIPT_DIR="${0:A:h}"
REPO_DIR="${SCRIPT_DIR:h}"
RUN_TIMESTAMP="${RUN_TIMESTAMP:-$(date +%Y%m%d%H%M%S)}"

LOCAL_PYTHON="${LOCAL_PYTHON:-${REPO_DIR}/venv/bin/python}"
LOCAL_DATA_DIR="${LOCAL_DATA_DIR:-${REPO_DIR}/iid-data}"
REMOTE_REPO_DIR="${REMOTE_REPO_DIR:-/home/rasheed/kuchida/antivenom_infocom}"
REMOTE_PROJECT_DIR="${REMOTE_PROJECT_DIR:-${REMOTE_REPO_DIR}/260803-local-controlled-end-to-end-ex}"
REMOTE_PYTHON="${REMOTE_PYTHON:-${REMOTE_REPO_DIR}/venv/bin/python}"
REMOTE_DATA_DIR="${REMOTE_DATA_DIR:-${REMOTE_REPO_DIR}/iid-data}"

SSH_USER="${SSH_USER:-rasheed}"
SSH_PASSWORD="${SSH_PASSWORD:-modenaottun}"
SSH_PORT="${SSH_PORT:-22}"
REMOTE_SUDO_PASSWORD="${REMOTE_SUDO_PASSWORD:-${SSH_PASSWORD}}"
LOCAL_SUDO_PASSWORD="${LOCAL_SUDO_PASSWORD:-}"

DATASET="${DATASET:-uoft-cs/cifar10}"
CLIENT_ID="${CLIENT_ID:-client_0}"
BATCH_SIZE="${BATCH_SIZE:-128}"
EXPECTED_BATCHES="${EXPECTED_BATCHES:-16}"
LOCAL_EPOCHS="${LOCAL_EPOCHS:-15}"
REPLAY_EPOCHS="${REPLAY_EPOCHS:-15}"
TRIALS="${TRIALS:-1}"
BASE_SEED="${BASE_SEED:-260803}"
LEARNING_RATE="${LEARNING_RATE:-0.001}"
PERF_EVENTS="${PERF_EVENTS:-cycles,instructions,branches,branch-misses,L1-dcache-loads,L1-dcache-load-misses}"

COLLECT_ROOT="${COLLECT_ROOT:-${SCRIPT_DIR}/collected_logs/${RUN_TIMESTAMP}}"

typeset -A REMOTE_HOSTS
REMOTE_HOSTS=(
  rpi112 192.168.0.112
  jetson141 192.168.0.141
)

usage() {
  cat <<'EOF'
Usage:
  ./run_experiment.zsh local
  ./run_experiment.zsh rpi112
  ./run_experiment.zsh jetson141
  ./run_experiment.zsh all

`all` runs local, Raspberry Pi 4 (.112), and Jetson CPU (.141) concurrently.
For each baseline/strong-augmentation condition, the script first trains
SimpleCNN and saves its final checkpoint. It then reloads that checkpoint and
runs real-data forward/backward replay with gradients enabled but no optimizer
updates. Training and frozen replay default to 15 epochs and one trial.
Only leaf-layer forward/backward PMU rows and metrics are written. Remote logs
and checkpoints are rsynced into collected_logs/<timestamp>/<target>/.
EOF
}

ssh_command() {
  local host="$1"
  shift
  if [[ -n "$SSH_PASSWORD" ]]; then
    command sshpass -p "$SSH_PASSWORD" ssh \
      -p "$SSH_PORT" \
      -o StrictHostKeyChecking=accept-new \
      -o ConnectTimeout=12 \
      -o ServerAliveInterval=10 \
      -o ServerAliveCountMax=3 \
      "${SSH_USER}@${host}" "$@"
  else
    command ssh \
      -p "$SSH_PORT" \
      -o StrictHostKeyChecking=accept-new \
      -o ConnectTimeout=12 \
      -o ServerAliveInterval=10 \
      -o ServerAliveCountMax=3 \
      "${SSH_USER}@${host}" "$@"
  fi
}

host_is_reachable() {
  ping -c 1 -W 1 "$1" >/dev/null 2>&1
}

configure_local_perf() {
  if [[ "$(sysctl -n kernel.perf_event_paranoid 2>/dev/null || print 999)" == "-1" ]]; then
    return
  fi
  if sudo -n sysctl kernel.perf_event_paranoid=-1 >/dev/null 2>&1; then
    return
  fi
  if [[ -n "$LOCAL_SUDO_PASSWORD" ]]; then
    print -r -- "$LOCAL_SUDO_PASSWORD" | sudo -S -p '' sysctl kernel.perf_event_paranoid=-1 >/dev/null
    return
  fi
  print -u2 "Local perf access is restricted. Run: sudo sysctl kernel.perf_event_paranoid=-1"
  return 1
}

configure_remote_perf() {
  local host="$1"
  local quoted_password="${(q)REMOTE_SUDO_PASSWORD}"
  ssh_command "$host" \
    "if [ \"\$(sysctl -n kernel.perf_event_paranoid 2>/dev/null)\" = -1 ]; then exit 0; fi; if sudo -n sysctl kernel.perf_event_paranoid=-1 >/dev/null 2>&1; then exit 0; fi; printf '%s\\n' ${quoted_password} | sudo -S -p '' sysctl kernel.perf_event_paranoid=-1 >/dev/null"
}

preflight_local() {
  test -x "$LOCAL_PYTHON"
  test -f "${SCRIPT_DIR}/running_ml.py"
  configure_local_perf
  CUDA_VISIBLE_DEVICES='' "$LOCAL_PYTHON" "${SCRIPT_DIR}/preflight.py" \
    --data-dir "$LOCAL_DATA_DIR" \
    --dataset "$DATASET" \
    --client-id "$CLIENT_ID" \
    --batch-size "$BATCH_SIZE" \
    --expected-batches "$EXPECTED_BATCHES" \
    --perf-events "$PERF_EVENTS"
}

preflight_remote() {
  local target="$1"
  local host="${REMOTE_HOSTS[$target]}"
  configure_remote_perf "$host"
  local -a command=(
    env CUDA_VISIBLE_DEVICES=''
    "$REMOTE_PYTHON" "${REMOTE_PROJECT_DIR}/preflight.py"
    --data-dir "$REMOTE_DATA_DIR"
    --dataset "$DATASET"
    --client-id "$CLIENT_ID"
    --batch-size "$BATCH_SIZE"
    --expected-batches "$EXPECTED_BATCHES"
    --perf-events "$PERF_EVENTS"
  )
  ssh_command "$host" "test -x ${(q)REMOTE_PYTHON} && test -f ${(q)REMOTE_PROJECT_DIR}/running_ml.py && ${(q)command[@]}"
}

run_command() {
  local python="$1"
  local project_dir="$2"
  local data_dir="$3"
  local log_dir="$4"
  local target="$5"
  local host="$6"
  local scenario="$7"
  local profile="$8"
  local experiment_mode="$9"
  local epochs="${10}"
  local checkpoint_path="${11}"
  local augment="{\"enabled\":true,\"_profile\":\"${profile}\",\"resize\":[32,32],\"horizontal_flip\":false,\"normalize\":true}"
  local -a command=(
    env CUDA_VISIBLE_DEVICES='' PYTHONHASHSEED="$BASE_SEED"
    "$python" "${project_dir}/running_ml.py"
    --dataset "$DATASET"
    --data-dir "$data_dir"
    --log-dir "${log_dir}/${scenario}/${experiment_mode}"
    --model simple_cnn
    --batch-size "$BATCH_SIZE"
    --local-epochs "$epochs"
    --learning-rate "$LEARNING_RATE"
    --num-clients 10
    --client-id "$CLIENT_ID"
    --device-id "$target"
    --host "$host"
    --partition-method iid
    --dataset-split train
    --poisoning-method clean
    --reference-trials 0
    --trials "$TRIALS"
    --seed "$BASE_SEED"
    --scenario "$scenario"
    --experiment-mode "$experiment_mode"
    --checkpoint-path "$checkpoint_path"
    --experiment-id "layer_e2e_${RUN_TIMESTAMP}_${target}_${scenario}_${experiment_mode}"
    --augment "$augment"
    --perf-events "$PERF_EVENTS"
  )
  print -r -- "${(q)command[@]}"
}

run_local_target() {
  local target=local
  local host="$(hostname)"
  local log_root="${SCRIPT_DIR}/logs/layer_end_to_end/${RUN_TIMESTAMP}/${target}"
  print "==> preflight ${target}"
  preflight_local
  mkdir -p "$log_root"
  local scenario profile command checkpoint
  for scenario profile in baseline baseline strong_augmentation strong; do
    checkpoint="${log_root}/${scenario}/trained_model.pt"
    print "==> train ${target} scenario=${scenario}"
    command="$(run_command "$LOCAL_PYTHON" "$SCRIPT_DIR" "$LOCAL_DATA_DIR" "$log_root" "$target" "$host" "$scenario" "$profile" train "$LOCAL_EPOCHS" "$checkpoint")"
    eval "$command"
    print "==> frozen replay ${target} scenario=${scenario} checkpoint=${checkpoint}"
    command="$(run_command "$LOCAL_PYTHON" "$SCRIPT_DIR" "$LOCAL_DATA_DIR" "$log_root" "$target" "$host" "$scenario" "$profile" frozen_replay "$REPLAY_EPOCHS" "$checkpoint")"
    eval "$command"
  done
  mkdir -p "${COLLECT_ROOT}/${target}"
  rsync -a "${log_root}/" "${COLLECT_ROOT}/${target}/"
}

collect_remote_logs() {
  local target="$1"
  local host="${REMOTE_HOSTS[$target]}"
  local source="${SSH_USER}@${host}:${REMOTE_PROJECT_DIR}/logs/layer_end_to_end/${RUN_TIMESTAMP}/${target}/"
  local destination="${COLLECT_ROOT}/${target}/"
  mkdir -p "$destination"
  if [[ -n "$SSH_PASSWORD" ]]; then
    sshpass -p "$SSH_PASSWORD" rsync -az \
      -e "ssh -p ${SSH_PORT} -o StrictHostKeyChecking=accept-new" \
      "$source" "$destination"
  else
    rsync -az -e "ssh -p ${SSH_PORT} -o StrictHostKeyChecking=accept-new" \
      "$source" "$destination"
  fi
}

run_remote_target() {
  local target="$1"
  local host="${REMOTE_HOSTS[$target]}"
  if ! host_is_reachable "$host"; then
    print -u2 "Skipping unreachable target ${target} (${host})."
    return 0
  fi
  print "==> preflight ${target} (${host})"
  preflight_remote "$target"
  local log_root="${REMOTE_PROJECT_DIR}/logs/layer_end_to_end/${RUN_TIMESTAMP}/${target}"
  local scenario profile command checkpoint
  for scenario profile in baseline baseline strong_augmentation strong; do
    checkpoint="${log_root}/${scenario}/trained_model.pt"
    print "==> train ${target} scenario=${scenario}"
    command="$(run_command "$REMOTE_PYTHON" "$REMOTE_PROJECT_DIR" "$REMOTE_DATA_DIR" "$log_root" "$target" "$host" "$scenario" "$profile" train "$LOCAL_EPOCHS" "$checkpoint")"
    ssh_command "$host" "mkdir -p ${(q)log_root} && cd ${(q)REMOTE_PROJECT_DIR} && ${command}"
    print "==> frozen replay ${target} scenario=${scenario} checkpoint=${checkpoint}"
    command="$(run_command "$REMOTE_PYTHON" "$REMOTE_PROJECT_DIR" "$REMOTE_DATA_DIR" "$log_root" "$target" "$host" "$scenario" "$profile" frozen_replay "$REPLAY_EPOCHS" "$checkpoint")"
    ssh_command "$host" "mkdir -p ${(q)log_root} && cd ${(q)REMOTE_PROJECT_DIR} && ${command}"
  done
  print "==> collect ${target}"
  collect_remote_logs "$target"
}

run_all() {
  mkdir -p "$COLLECT_ROOT"
  typeset -a pids labels
  (run_local_target) &
  pids+=("$!")
  labels+=(local)
  (run_remote_target rpi112) &
  pids+=("$!")
  labels+=(rpi112)
  (run_remote_target jetson141) &
  pids+=("$!")
  labels+=(jetson141)

  local failed=0 index
  for index in {1..${#pids}}; do
    if ! wait "${pids[$index]}"; then
      print -u2 "Target failed: ${labels[$index]}"
      failed=1
    fi
  done
  (( failed == 0 ))
}

ACTION="${1:-all}"
if (( TRIALS != 1 )); then
  print -u2 "This checkpoint/replay experiment requires TRIALS=1; got ${TRIALS}."
  exit 2
fi
case "$ACTION" in
  local)
    run_local_target
    ;;
  rpi112|jetson141)
    run_remote_target "$ACTION"
    ;;
  all)
    run_all
    ;;
  -h|--help|help)
    usage
    exit 0
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac

print "Experiment complete. Collected logs: ${COLLECT_ROOT}"
