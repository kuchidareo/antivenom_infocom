#!/usr/bin/env zsh
set -euo pipefail

SCRIPT_DIR="${0:A:h}"
REPO_DIR="${SCRIPT_DIR:h}"
PROJECT_DIR="${ROBUSTNESS_PROJECT_DIR:-${REPO_DIR}/M260718-robustness}"
PYTHON_BIN="${PYTHON:-${REPO_DIR}/venv/bin/python}"
IID_DATA_ROOT="${IID_DATA_DIR:-${DATA_DIR:-${REPO_DIR}/iid-data}}"
NONIID_DATA_ROOT="${NONIID_DATA_DIR:-${REPO_DIR}/noniid-data}"
LOCAL_LOG_DIR="${LOG_DIR:-${SCRIPT_DIR}/logs/local_ml}"

SMALL_TRASHNET_DATASET="kuchidareo/small_trashnet"
CIFAR10_DATASET="uoft-cs/cifar10"
CHINESE_TRAFFIC_SIGN_DATASET="kuchidareo/chinese_trafficsign_dataset"
DEFAULT_METHODS="clean"

REFERENCE_TRIALS="${REFERENCE_TRIALS:-1}"
ANALYSIS_TRIALS="${ANALYSIS_TRIALS:-1}"
LOCAL_EPOCHS="${LOCAL_EPOCHS:-10}"
BASELINE_BATCH_SIZE="${BASELINE_BATCH_SIZE:-16}"
BATCH_SIZE_VARIANTS="${BATCH_SIZE_VARIANTS:-4,8,32}"
BASELINE_MODEL="${BASELINE_MODEL:-simple_cnn}"
MODEL_VARIANTS="${MODEL_VARIANTS:-resnet18,mobilenet_v3_large,swin_t}"
NONIID_ALPHA="${NONIID_ALPHA:-0.3}"
CLIENT_SELECTION_SEED="${CLIENT_SELECTION_SEED:-260626}"

PERF_ENABLED="${PERF_ENABLED:-1}"
PERF_FPS="${PERF_FPS:-50}"
PERF_EVENTS="${PERF_EVENTS:-}"
PERF_PROFILE="${PERF_PROFILE:-auto}"

BG_WORKLOAD_PROFILE="${BG_WORKLOAD_PROFILE:-medium}"
BG_WORKLOAD_TEST_DURATION="${BG_WORKLOAD_TEST_DURATION:-10}"
BG_WORKLOAD_PID_FILE="${BG_WORKLOAD_PID_FILE:-/tmp/antivenom_robustness_bg.pid}"
BG_WORKLOAD_OUTPUT="${SCRIPT_DIR}/logs/bg_workloads/run_bg_workloads.out"
BG_WORKLOAD_CHECKED_GROUPS=""

RUN_HOST="${HOST_LABEL:-$(hostname)}"
RUN_DEVICE_ID="${DEVICE_ID:-$RUN_HOST}"
FIXED_CLIENT_ID="${CLIENT_ID:-}"

require_identity() {
  if [[ -n "$FIXED_CLIENT_ID" && "$FIXED_CLIENT_ID" != client_<0-9> ]]; then
    print -u2 "Invalid CLIENT_ID '${FIXED_CLIENT_ID}'; expected client_0 through client_9."
    return 1
  fi
}

normalize_methods() {
  local raw="$1"
  local -a selected
  local token
  raw="${raw//,/ }"

  for token in ${(z)raw}; do
    case "$token" in
      clean|unlearnable_examples|random_label_flipping|target_label_flipping|availability_shortcuts)
        if (( ${selected[(Ie)$token]} == 0 )); then
          selected+=("$token")
        fi
        ;;
      *)
        print -u2 "Unknown poisoning method: ${token}"
        return 1
        ;;
    esac
  done

  (( ${#selected[@]} > 0 )) || {
    print -u2 "No poisoning methods were selected."
    return 1
  }
  print -- "${(j:,:)selected}"
}

select_client_id() {
  local stage_name="$1"
  if [[ -n "$FIXED_CLIENT_ID" ]]; then
    print -- "$FIXED_CLIENT_ID"
    return
  fi

  "$PYTHON_BIN" -c '
import hashlib
import random
import sys

seed, device_id, stage_name = sys.argv[1:]
stages = [
    "iid_small_trashnet",
    "iid_chinese_traffic_sign",
    "iid_cifar10",
    "noniid_small_trashnet",
    "noniid_chinese_traffic_sign",
    "noniid_cifar10",
]
if stage_name in stages:
    clients = [f"client_{index}" for index in range(10)]
    random.Random(f"{seed}:{device_id}").shuffle(clients)
    print(clients[stages.index(stage_name)])
    raise SystemExit

digest = hashlib.sha256(f"{seed}:{device_id}:{stage_name}".encode()).digest()
value = int.from_bytes(digest[:8], byteorder="big")
print("client_%d" % (value % 10))
' "$CLIENT_SELECTION_SEED" "$RUN_DEVICE_ID" "$stage_name"
}

dataset_dir_name() {
  case "$1" in
    "$SMALL_TRASHNET_DATASET") print -- "small_trashnet" ;;
    "$CIFAR10_DATASET") print -- "cifar10" ;;
    "$CHINESE_TRAFFIC_SIGN_DATASET") print -- "chinese_trafficsign_dataset" ;;
    *) return 1 ;;
  esac
}

check_local_environment() {
  require_identity
  cd "$PROJECT_DIR"

  test -x "$PYTHON_BIN" || {
    print -u2 "Python is not executable: ${PYTHON_BIN}"
    return 1
  }
  "$PYTHON_BIN" --version
  "$PYTHON_BIN" -c 'import datasets, numpy, PIL, psutil, torch, torchvision; print("Python dependencies: ok")'

  local data_root dataset_name dataset_path
  for data_root in "$IID_DATA_ROOT" "$NONIID_DATA_ROOT"; do
    for dataset_name in "$SMALL_TRASHNET_DATASET" "$CHINESE_TRAFFIC_SIGN_DATASET" "$CIFAR10_DATASET"; do
      dataset_path="${data_root}/$(dataset_dir_name "$dataset_name")"
      test -d "$dataset_path" || {
        print -u2 "Missing prepared dataset: ${dataset_path}"
        return 1
      }
    done
  done

  "$PYTHON_BIN" -c 'from models import get_model; names = "simple_cnn,resnet18,mobilenet_v3_large,swin_t".split(","); assert all(get_model(name, num_classes=6) is not None for name in names); print("models", ",".join(names))'

  if [[ "$PERF_ENABLED" == "1" ]]; then
    command -v perf >/dev/null || {
      print -u2 "perf is not installed."
      return 1
    }
    perf --version
  fi

  print "host_label: ${RUN_HOST}"
  print "device_id: ${RUN_DEVICE_ID}"
  if [[ -n "$FIXED_CLIENT_ID" ]]; then
    print "client_selection: fixed (${FIXED_CLIENT_ID})"
  else
    print "client_selection: reproducibly random per stage (seed=${CLIENT_SELECTION_SEED})"
  fi
  print "project_dir: ${PROJECT_DIR}"
  print "iid_data_root: ${IID_DATA_ROOT}"
  print "noniid_data_root: ${NONIID_DATA_ROOT}"
  print "log_dir: ${LOCAL_LOG_DIR}"
  print "compute_device: cpu (CUDA disabled)"
}

configure_local_perf() {
  if [[ "$PERF_ENABLED" != "1" ]]; then
    print "Perf monitoring is disabled."
    return
  fi

  print "Configuring kernel.perf_event_paranoid=-1 locally..."
  if (( EUID == 0 )); then
    sysctl kernel.perf_event_paranoid=-1
  else
    sudo sysctl kernel.perf_event_paranoid=-1
  fi

  local events selected_profile machine event_constant
  events="$PERF_EVENTS"
  if [[ -z "$events" ]]; then
    selected_profile="$PERF_PROFILE"
    if [[ "$selected_profile" == "auto" ]]; then
      machine="$(uname -m)"
      if [[ -f /etc/nv_tegra_release ]] || grep -aEqi 'tegra|nvidia' /proc/device-tree/compatible 2>/dev/null; then
        selected_profile="jetson"
      elif [[ "$machine" == aarch64 || "$machine" == arm* ]]; then
        selected_profile="rpi"
      else
        selected_profile="x86"
      fi
    fi
    case "$selected_profile" in
      jetson) event_constant="JETSON_PERF_EVENTS" ;;
      rpi) event_constant="RPI_PERF_EVENTS" ;;
      x86) event_constant="X86_PERF_EVENTS" ;;
      common) event_constant="COMMON_PERF_EVENTS" ;;
      *)
        print -u2 "Invalid PERF_PROFILE '${PERF_PROFILE}'; expected auto, common, x86, rpi, or jetson."
        return 1
        ;;
    esac
    events="$(
      cd "$PROJECT_DIR"
      "$PYTHON_BIN" -c "import perf_logger; print(','.join(perf_logger.${event_constant}))"
    )"
    print "Detected perf profile: ${selected_profile}"
  fi
  PERF_EVENTS="$events"

  print "Validating perf events: ${events}"
  perf stat -e "$events" -- true >/dev/null
}

bg_group_needs_opencv() {
  [[ "$1" == "group1" || "$1" == "both" ]]
}

check_bg_group() {
  local group="$1"
  if [[ ",${BG_WORKLOAD_CHECKED_GROUPS}," == *",${group},"* ]]; then
    return
  fi

  cd "$PROJECT_DIR"
  test -x ./run_bg_workloads.sh
  if bg_group_needs_opencv "$group"; then
    "$PYTHON_BIN" -c 'import cv2, numpy; print("background OpenCV dependencies: ok")'
  fi
  if [[ "$group" == "group2" || "$group" == "both" ]]; then
    if ! command -v iperf3 >/dev/null; then
      print -u2 "Warning: iperf3 is unavailable; the communication workload will be skipped."
    fi
  fi

  PYTHON_BIN="$PYTHON_BIN" ./run_bg_workloads.sh \
    --group "$group" \
    --profile "$BG_WORKLOAD_PROFILE" \
    --dry-run
  PYTHON_BIN="$PYTHON_BIN" ./run_bg_workloads.sh \
    --group "$group" \
    --profile "$BG_WORKLOAD_PROFILE" \
    --test \
    --duration-sec "$BG_WORKLOAD_TEST_DURATION"

  if [[ -n "$BG_WORKLOAD_CHECKED_GROUPS" ]]; then
    BG_WORKLOAD_CHECKED_GROUPS+=",${group}"
  else
    BG_WORKLOAD_CHECKED_GROUPS="$group"
  fi
}

stop_bg_workloads() {
  if [[ ! -f "$BG_WORKLOAD_PID_FILE" ]]; then
    return
  fi

  local bg_pid
  bg_pid="$(<"$BG_WORKLOAD_PID_FILE")"
  kill "$bg_pid" 2>/dev/null || true
  wait "$bg_pid" 2>/dev/null || true
  rm -f "$BG_WORKLOAD_PID_FILE"
}

start_bg_workloads() {
  local group="$1"
  check_bg_group "$group"
  stop_bg_workloads
  mkdir -p "${BG_WORKLOAD_OUTPUT:h}"

  print "Starting local background workload: group=${group}, profile=${BG_WORKLOAD_PROFILE}"
  cd "$PROJECT_DIR"
  PYTHON_BIN="$PYTHON_BIN" ./run_bg_workloads.sh \
    --group "$group" \
    --profile "$BG_WORKLOAD_PROFILE" \
    >"$BG_WORKLOAD_OUTPUT" 2>&1 &
  local bg_pid="$!"
  print -- "$bg_pid" >"$BG_WORKLOAD_PID_FILE"
  sleep 2

  if ! kill -0 "$bg_pid" 2>/dev/null; then
    print -u2 "Background workload failed to remain running."
    tail -n 80 "$BG_WORKLOAD_OUTPUT" >&2 || true
    stop_bg_workloads
    return 1
  fi
}

run_local_stage() {
  local stage_name="$1"
  local dataset_name="$2"
  local methods="$3"
  local reference_trials="$4"
  local analysis_trials="$5"
  local batch_size="$6"
  local model_name="$7"
  local bg_group="${8:-}"
  local data_root="${9:-$IID_DATA_ROOT}"
  local partition_method="${10:-iid}"
  local selected_client_id
  local -a command perf_args bg_args

  selected_client_id="$(select_client_id "$stage_name")"

  if [[ "$PERF_ENABLED" == "1" ]]; then
    perf_args=(--enable-perf --perf-fps "$PERF_FPS" --perf-events "$PERF_EVENTS")
  else
    perf_args=(--disable-perf)
  fi
  if [[ -n "$bg_group" ]]; then
    bg_args=(
      --background-workload-enabled
      --background-workload-group "$bg_group"
      --background-workload-profile "$BG_WORKLOAD_PROFILE"
    )
  else
    bg_args=()
  fi

  print
  print "Running local stage: ${stage_name}"
  print "  host/client: ${RUN_HOST}/${selected_client_id}"
  print "  dataset: ${dataset_name}"
  print "  data_root: ${data_root}"
  print "  partition_method: ${partition_method}"
  print "  poisoning_methods: ${methods}"
  print "  model: ${model_name}"
  print "  local_epochs: ${LOCAL_EPOCHS}, batch_size: ${batch_size}"
  print "  reference_trials: ${reference_trials}, analysis_trials: ${analysis_trials}"
  print "  compute_device: cpu (CUDA disabled)"
  print "  bg_noise: ${bg_group:-none}"

  command=(
    "$PYTHON_BIN" running_ml.py
    --dataset "$dataset_name"
    --data-dir "$data_root"
    --log-dir "$LOCAL_LOG_DIR"
    --client-id "$selected_client_id"
    --device-id "$RUN_DEVICE_ID"
    --host "$RUN_HOST"
    --model "$model_name"
    --local-epochs "$LOCAL_EPOCHS"
    --batch-size "$batch_size"
    --partition-method "$partition_method"
    --noniid-alpha "$NONIID_ALPHA"
    --reference-trials "$reference_trials"
    --trials "$analysis_trials"
    --poisoning-method "$methods"
    "${perf_args[@]}"
    "${bg_args[@]}"
  )

  cd "$PROJECT_DIR"
  CUDA_VISIBLE_DEVICES='' "${command[@]}"
  print "Finished local stage: ${stage_name}"
}

run_bg_stage() {
  local group="$1"
  shift
  local run_exit_code=0

  start_bg_workloads "$group"
  run_local_stage "$@" "$group" || run_exit_code=$?
  stop_bg_workloads
  return "$run_exit_code"
}

run_model_stages() {
  local methods="$1"
  local model_name
  for model_name in ${(s:,:)MODEL_VARIANTS}; do
    run_local_stage \
      "model_${model_name}" \
      "$SMALL_TRASHNET_DATASET" \
      "$methods" \
      "0" \
      "$ANALYSIS_TRIALS" \
      "$BASELINE_BATCH_SIZE" \
      "$model_name"
  done
}

prepare_local_experiment() {
  check_local_environment
  configure_local_perf
}

run_all_stages() {
  local methods="${1:-$DEFAULT_METHODS}"
  methods="$(normalize_methods "$methods")"
  prepare_local_experiment

  run_local_stage "iid_small_trashnet" "$SMALL_TRASHNET_DATASET" \
    "$methods" "0" "$ANALYSIS_TRIALS" "$BASELINE_BATCH_SIZE" "$BASELINE_MODEL" \
    "" "$IID_DATA_ROOT" "iid"
  run_local_stage "iid_chinese_traffic_sign" "$CHINESE_TRAFFIC_SIGN_DATASET" \
    "$methods" "0" "$ANALYSIS_TRIALS" "$BASELINE_BATCH_SIZE" "$BASELINE_MODEL" \
    "" "$IID_DATA_ROOT" "iid"
  run_local_stage "iid_cifar10" "$CIFAR10_DATASET" \
    "$methods" "0" "$ANALYSIS_TRIALS" "$BASELINE_BATCH_SIZE" "$BASELINE_MODEL" \
    "" "$IID_DATA_ROOT" "iid"

  run_local_stage "noniid_small_trashnet" "$SMALL_TRASHNET_DATASET" \
    "$methods" "0" "$ANALYSIS_TRIALS" "$BASELINE_BATCH_SIZE" "$BASELINE_MODEL" \
    "" "$NONIID_DATA_ROOT" "dirichlet_noniid"
  run_local_stage "noniid_chinese_traffic_sign" "$CHINESE_TRAFFIC_SIGN_DATASET" \
    "$methods" "0" "$ANALYSIS_TRIALS" "$BASELINE_BATCH_SIZE" "$BASELINE_MODEL" \
    "" "$NONIID_DATA_ROOT" "dirichlet_noniid"
  run_local_stage "noniid_cifar10" "$CIFAR10_DATASET" \
    "$methods" "0" "$ANALYSIS_TRIALS" "$BASELINE_BATCH_SIZE" "$BASELINE_MODEL" \
    "" "$NONIID_DATA_ROOT" "dirichlet_noniid"
}

print_plan_line() {
  local stage_name="$1"
  local dataset_name="$2"
  local partition_method="$3"
  print "  ${stage_name}: dataset=${dataset_name} partition=${partition_method} client=$(select_client_id "$stage_name")"
}

print_experiment_plan() {
  print "Local experiment plan (device_id=${RUN_DEVICE_ID}, client_seed=${CLIENT_SELECTION_SEED}):"
  print_plan_line "iid_small_trashnet" "$SMALL_TRASHNET_DATASET" "iid"
  print_plan_line "iid_chinese_traffic_sign" "$CHINESE_TRAFFIC_SIGN_DATASET" "iid"
  print_plan_line "iid_cifar10" "$CIFAR10_DATASET" "iid"
  print_plan_line "noniid_small_trashnet" "$SMALL_TRASHNET_DATASET" "dirichlet_noniid"
  print_plan_line "noniid_chinese_traffic_sign" "$CHINESE_TRAFFIC_SIGN_DATASET" "dirichlet_noniid"
  print_plan_line "noniid_cifar10" "$CIFAR10_DATASET" "dirichlet_noniid"
}

usage() {
  cat <<'EOF'
Usage:
  ./run_experiments_local.zsh check
  ./run_experiments_local.zsh plan
  ./run_experiments_local.zsh bg-check
  ./run_experiments_local.zsh models [poisoning_methods]
  ./run_experiments_local.zsh run [poisoning_methods]

This script runs training directly on the current machine. It does not use SSH.

The network address is not used. Each stage reproducibly selects a random
client_0 through client_9. To force one partition for every stage:
  CLIENT_ID=client_1 ./run_experiments_local.zsh run
  ./run_experiments_local.zsh run clean,availability_shortcuts

Default run order:
  IID:     small_trashnet, Chinese traffic signs, CIFAR-10
  non-IID: small_trashnet, Chinese traffic signs, CIFAR-10

Common overrides:
  PYTHON=/path/to/venv/bin/python
  IID_DATA_DIR=/path/to/iid-data
  NONIID_DATA_DIR=/path/to/noniid-data
  LOG_DIR=/path/to/logs/local_ml
  HOST_LABEL=my-host
  DEVICE_ID=my-device
  CLIENT_SELECTION_SEED=260626
  PERF_PROFILE=auto              auto, common, x86, rpi, or jetson
  PERF_ENABLED=0
  LOCAL_EPOCHS=10
  REFERENCE_TRIALS=1
  ANALYSIS_TRIALS=1
EOF
}

cleanup() {
  stop_bg_workloads
}
trap cleanup EXIT INT TERM

main() {
  local mode="${1:-run}"
  case "$mode" in
    check)
      prepare_local_experiment
      ;;
    plan)
      require_identity
      print_experiment_plan
      ;;
    bg-check)
      prepare_local_experiment
      check_bg_group group1
      check_bg_group group2
      check_bg_group both
      ;;
    models)
      prepare_local_experiment
      run_model_stages "$(normalize_methods "${2:-$DEFAULT_METHODS}")"
      ;;
    run|all)
      run_all_stages "${2:-$DEFAULT_METHODS}"
      ;;
    -h|--help|help)
      usage
      ;;
    *)
      print -u2 "Unknown mode: ${mode}"
      usage >&2
      return 1
      ;;
  esac
}

main "$@"
