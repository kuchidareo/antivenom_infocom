#!/usr/bin/env bash
# Background workloads for low-resource Raspberry Pi experiments.
#
# The script starts optional perception, communication, and logmap-style
# workloads with explicit profiles. Without --use-systemd-scope, throttling is
# best-effort through nice values and duty-cycle sleeps, not a hard CPU quota.

set -euo pipefail

VENV_DIR="${VENV_DIR:-/home/rasheed/kuchida/antivenom_infocom/venv}"
PYTHON_BIN="${PYTHON_BIN:-${VENV_DIR}/bin/python}"

GROUP="both"
PROFILE="medium"
TEST_MODE=0
DRY_RUN=0
DURATION_SEC=30

FEATURE=""
CAM_INDEX="0"
TARGET_FPS=""
PERCEPTION_UTIL=""

IPERF_SERVER="${IPERF_SERVER:-192.0.2.10}"
UPLOAD_MBIT=""
BURST_SEC=""
IDLE_SEC=""

LOGMAP_IO=""
LOGMAP_NET=""
TILE_URL="${TILE_URL:-https://tile.openstreetmap.org/10/550/340.png}"
TILE_PERIOD_SEC=""
CLEAN_PERIOD_SEC="${CLEAN_PERIOD_SEC:-3600}"

USE_SYSTEMD_SCOPE=0
CPU_QUOTA="20%"

TMP_ROOT=""
LOG_DIR=""
IO_DIR=""
PERCEPTION_PY=""
LOGMAP_PY=""

RUN_PERCEPTION=0
RUN_COMMS=0
RUN_LOGMAP=0

pids=()
workload_names=()
workload_logs=()

usage() {
  cat <<'EOF'
Usage:
  ./run_bg_workloads.sh [options]

Groups:
  --group group1 | perception   Start perception only.
  --group group2 | aux          Start communication + logmap.
  --group both   | all          Start all workloads. Default: both.

Profiles:
  --profile light   Pi 3-safe low load.
  --profile medium  Pi 4 standard load. Default.
  --profile heavy   Stress profile; warns because it may be too heavy.

Test mode:
  --test                  Run for --duration-sec seconds, then cleanup.
  --duration-sec FLOAT    Test duration. Default: 30.
  --dry-run               Print final config and planned workloads, then exit.

Main options:
  --perception-util FLOAT
  --target-fps FLOAT
  --feature fast|orb
  --upload-mbit FLOAT
  --burst-sec FLOAT
  --idle-sec FLOAT
  --tile-period-sec FLOAT
  --logmap-io 0|1
  --logmap-net 0|1
  --iperf-server HOST
  --tile-url URL
  --use-systemd-scope
  --cpu-quota 20%
  --help

Examples:
  ./run_bg_workloads.sh --group group1 --profile light
  ./run_bg_workloads.sh --group group2 --profile medium --test --duration-sec 60
  ./run_bg_workloads.sh --group both --profile light --use-systemd-scope --cpu-quota 20%

Raspberry Pi 3 safe checks:
  ./run_bg_workloads.sh --test --duration-sec 30 --group group1 --profile light
  ./run_bg_workloads.sh --test --duration-sec 30 --group group2 --profile light
  ./run_bg_workloads.sh --test --duration-sec 30 --group both --profile light

Raspberry Pi 4 standard:
  ./run_bg_workloads.sh --group both --profile medium
EOF
}

die() {
  echo "[bg][error] $*" >&2
  exit 1
}

warn() {
  echo "[bg][warn] $*" >&2
}

parse_args() {
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --group)
        GROUP="${2:-}"
        shift 2
        ;;
      --profile)
        PROFILE="${2:-}"
        shift 2
        ;;
      --perception-util)
        PERCEPTION_UTIL="${2:-}"
        shift 2
        ;;
      --target-fps)
        TARGET_FPS="${2:-}"
        shift 2
        ;;
      --feature)
        FEATURE="${2:-}"
        shift 2
        ;;
      --upload-mbit)
        UPLOAD_MBIT="${2:-}"
        shift 2
        ;;
      --burst-sec)
        BURST_SEC="${2:-}"
        shift 2
        ;;
      --idle-sec)
        IDLE_SEC="${2:-}"
        shift 2
        ;;
      --tile-period-sec)
        TILE_PERIOD_SEC="${2:-}"
        shift 2
        ;;
      --logmap-io)
        LOGMAP_IO="${2:-}"
        shift 2
        ;;
      --logmap-net)
        LOGMAP_NET="${2:-}"
        shift 2
        ;;
      --iperf-server)
        IPERF_SERVER="${2:-}"
        shift 2
        ;;
      --tile-url)
        TILE_URL="${2:-}"
        shift 2
        ;;
      --test)
        TEST_MODE=1
        shift
        ;;
      --duration-sec)
        DURATION_SEC="${2:-}"
        shift 2
        ;;
      --dry-run)
        DRY_RUN=1
        shift
        ;;
      --use-systemd-scope)
        USE_SYSTEMD_SCOPE=1
        shift
        ;;
      --cpu-quota)
        CPU_QUOTA="${2:-}"
        shift 2
        ;;
      --help|-h)
        usage
        exit 0
        ;;
      *)
        die "Unknown option: $1"
        ;;
    esac
  done
}

apply_profile_defaults() {
  case "${PROFILE}" in
    light)
      FEATURE="${FEATURE:-fast}"
      TARGET_FPS="${TARGET_FPS:-10}"
      PERCEPTION_UTIL="${PERCEPTION_UTIL:-0.08}"
      UPLOAD_MBIT="${UPLOAD_MBIT:-0.5}"
      BURST_SEC="${BURST_SEC:-0.3}"
      IDLE_SEC="${IDLE_SEC:-2.0}"
      TILE_PERIOD_SEC="${TILE_PERIOD_SEC:-15}"
      LOGMAP_IO="${LOGMAP_IO:-0}"
      LOGMAP_NET="${LOGMAP_NET:-1}"
      ;;
    medium)
      FEATURE="${FEATURE:-fast}"
      TARGET_FPS="${TARGET_FPS:-20}"
      PERCEPTION_UTIL="${PERCEPTION_UTIL:-0.15}"
      UPLOAD_MBIT="${UPLOAD_MBIT:-2}"
      BURST_SEC="${BURST_SEC:-0.5}"
      IDLE_SEC="${IDLE_SEC:-1.0}"
      TILE_PERIOD_SEC="${TILE_PERIOD_SEC:-7}"
      LOGMAP_IO="${LOGMAP_IO:-0}"
      LOGMAP_NET="${LOGMAP_NET:-1}"
      ;;
    heavy)
      FEATURE="${FEATURE:-orb}"
      TARGET_FPS="${TARGET_FPS:-30}"
      PERCEPTION_UTIL="${PERCEPTION_UTIL:-0.30}"
      UPLOAD_MBIT="${UPLOAD_MBIT:-5}"
      BURST_SEC="${BURST_SEC:-1.0}"
      IDLE_SEC="${IDLE_SEC:-0.5}"
      TILE_PERIOD_SEC="${TILE_PERIOD_SEC:-3}"
      LOGMAP_IO="${LOGMAP_IO:-0}"
      LOGMAP_NET="${LOGMAP_NET:-1}"
      ;;
    *)
      die "--profile must be one of: light, medium, heavy"
      ;;
  esac
}

resolve_group() {
  case "${GROUP}" in
    group1|perception)
      GROUP="group1"
      RUN_PERCEPTION=1
      RUN_COMMS=0
      RUN_LOGMAP=0
      ;;
    group2|aux)
      GROUP="group2"
      RUN_PERCEPTION=0
      RUN_COMMS=1
      RUN_LOGMAP=1
      ;;
    both|all)
      GROUP="both"
      RUN_PERCEPTION=1
      RUN_COMMS=1
      RUN_LOGMAP=1
      ;;
    *)
      die "--group must be one of: group1, perception, group2, aux, both, all"
      ;;
  esac
}

validate_float() {
  local name="$1"
  local value="$2"
  [[ "${value}" =~ ^[0-9]+([.][0-9]+)?$ ]] || die "${name} must be numeric; got '${value}'"
}

validate_binary() {
  local name="$1"
  local value="$2"
  [[ "${value}" == "0" || "${value}" == "1" ]] || die "${name} must be 0 or 1; got '${value}'"
}

validate_config() {
  if [[ "${DRY_RUN}" != "1" ]]; then
    [[ -x "${PYTHON_BIN}" ]] || die "Python executable not found or not executable: ${PYTHON_BIN}"
  fi
  [[ "${FEATURE}" == "fast" || "${FEATURE}" == "orb" ]] || die "--feature must be fast or orb"

  validate_float "--target-fps" "${TARGET_FPS}"
  validate_float "--perception-util" "${PERCEPTION_UTIL}"
  validate_float "--upload-mbit" "${UPLOAD_MBIT}"
  validate_float "--burst-sec" "${BURST_SEC}"
  validate_float "--idle-sec" "${IDLE_SEC}"
  validate_float "--tile-period-sec" "${TILE_PERIOD_SEC}"
  validate_float "--duration-sec" "${DURATION_SEC}"
  validate_binary "--logmap-io" "${LOGMAP_IO}"
  validate_binary "--logmap-net" "${LOGMAP_NET}"

  if [[ "${PROFILE}" == "heavy" ]]; then
    warn "--profile heavy may be too heavy for Raspberry Pi 3."
  fi
  awk -v x="${PERCEPTION_UTIL}" 'BEGIN { exit !(x > 0.5) }' && \
    warn "--perception-util ${PERCEPTION_UTIL} is high for Pi 3/4." || true
  awk -v x="${TARGET_FPS}" 'BEGIN { exit !(x > 30.0) }' && \
    warn "--target-fps ${TARGET_FPS} is above 30 and may be too heavy." || true
  if [[ "${LOGMAP_IO}" == "1" ]]; then
    warn "--logmap-io 1 increases SD/eMMC writes."
  fi
}

maybe_reexec_with_systemd_scope() {
  if [[ "${USE_SYSTEMD_SCOPE}" != "1" ]]; then
    return
  fi
  if [[ "${BG_WORKLOAD_SYSTEMD_SCOPE:-0}" == "1" ]]; then
    return
  fi
  if ! command -v systemd-run >/dev/null 2>&1; then
    warn "--use-systemd-scope requested but systemd-run is not available; continuing without CPUQuota."
    return
  fi
  echo "[bg] re-executing under systemd-run scope with CPUQuota=${CPU_QUOTA}"
  exec systemd-run --scope -p "CPUQuota=${CPU_QUOTA}" env BG_WORKLOAD_SYSTEMD_SCOPE=1 "$0" "$@"
}

setup_tmpdir() {
  if [[ -d "/dev/shm" && -w "/dev/shm" ]]; then
    TMP_ROOT="/dev/shm"
  else
    TMP_ROOT="/tmp"
  fi
  LOG_DIR="${TMP_ROOT}/bg_workloads_logs"
  IO_DIR="${TMP_ROOT}/bg_workloads_io"
  mkdir -p "${LOG_DIR}"
}

print_config() {
  local perception_enabled="no"
  local comms_enabled="no"
  local logmap_enabled="no"
  [[ "${RUN_PERCEPTION}" == "1" ]] && perception_enabled="yes"
  [[ "${RUN_COMMS}" == "1" ]] && comms_enabled="yes"
  [[ "${RUN_LOGMAP}" == "1" ]] && logmap_enabled="yes"

  echo "[bg] config:"
  echo "  group: ${GROUP}"
  echo "  profile: ${PROFILE}"
  echo "  throttling: best-effort nice + duty-cycle sleeps"
  if [[ "${USE_SYSTEMD_SCOPE}" == "1" && "${BG_WORKLOAD_SYSTEMD_SCOPE:-0}" == "1" ]]; then
    echo "  systemd CPUQuota: ${CPU_QUOTA}"
  fi
  echo "  perception:"
  echo "    enabled: ${perception_enabled}"
  echo "    feature: ${FEATURE}"
  echo "    target_fps: ${TARGET_FPS}"
  echo "    perception_util: ${PERCEPTION_UTIL}"
  echo "  comms:"
  echo "    enabled: ${comms_enabled}"
  echo "    iperf_server: ${IPERF_SERVER}"
  echo "    upload_mbit: ${UPLOAD_MBIT}"
  echo "    burst_sec: ${BURST_SEC}"
  echo "    idle_sec: ${IDLE_SEC}"
  echo "  logmap:"
  echo "    enabled: ${logmap_enabled}"
  echo "    logmap_io: ${LOGMAP_IO}"
  echo "    logmap_net: ${LOGMAP_NET}"
  echo "    tile_period_sec: ${TILE_PERIOD_SEC}"
  echo "  python: ${PYTHON_BIN}"
  echo "  logs: ${LOG_DIR}"
}

generate_perception_py() {
  PERCEPTION_PY="$(mktemp -p "${TMP_ROOT}" bg_perception.XXXXXX.py)"
  cat > "${PERCEPTION_PY}" <<'PY'
import os
import time

import cv2 as cv
import numpy as np

FEATURE = os.getenv("FEATURE", "fast").lower()
CAM_INDEX = int(os.getenv("CAM_INDEX", "0"))
TARGET_FPS = float(os.getenv("TARGET_FPS", "10"))
PERCEPTION_UTIL = float(os.getenv("PERCEPTION_UTIL", "0.08"))
period = 1.0 / max(TARGET_FPS, 0.1)

cap = cv.VideoCapture(CAM_INDEX)
use_synthetic = not cap.isOpened()
if use_synthetic:
    width, height = 640, 480
    frame = (np.random.rand(height, width, 3) * 255).astype(np.uint8)
    print("[perception] camera unavailable; using synthetic image", flush=True)

if FEATURE == "orb":
    detector = cv.ORB_create(nfeatures=500)

    def run_feature(img):
        detector.detectAndCompute(img, None)
else:
    detector = cv.FastFeatureDetector_create(threshold=20, nonmaxSuppression=True)

    def run_feature(img):
        detector.detect(img, None)

while True:
    start = time.time()

    if use_synthetic:
        img = frame
        _, buf = cv.imencode(".jpg", img, [int(cv.IMWRITE_JPEG_QUALITY), 85])
        img = cv.imdecode(buf, cv.IMREAD_COLOR)
    else:
        ok, img = cap.read()
        if not ok:
            use_synthetic = True
            continue

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    run_feature(gray)

    elapsed = time.time() - start
    util = max(min(PERCEPTION_UTIL, 0.95), 0.01)
    desired_cycle = max(period, elapsed / util)
    sleep_sec = desired_cycle - elapsed
    if sleep_sec > 0:
        time.sleep(sleep_sec)
PY
}

generate_logmap_py() {
  LOGMAP_PY="$(mktemp -p "${TMP_ROOT}" bg_logmap.XXXXXX.py)"
  cat > "${LOGMAP_PY}" <<'PY'
import os
import random
import time
import urllib.request
from pathlib import Path

ENABLE_IO = os.getenv("LOGMAP_IO", "0") != "0"
ENABLE_NET = os.getenv("LOGMAP_NET", "1") != "0"

TILE_URL = os.getenv("TILE_URL", "https://tile.openstreetmap.org/10/550/340.png")
TILE_PERIOD = float(os.getenv("TILE_PERIOD_SEC", "15"))
CLEAN_PERIOD = float(os.getenv("CLEAN_PERIOD_SEC", "3600"))

IO_DIR = None
seq_path = None
seq_file = None

if ENABLE_IO:
    IO_DIR = Path(os.getenv("IO_DIR", "/tmp/bg_workloads_io"))
    IO_DIR.mkdir(parents=True, exist_ok=True)
    seq_path = IO_DIR / "seq.bin"
    seq_file = seq_path.open("ab", buffering=0)

SIZES = [4 * 1024, 8 * 1024, 16 * 1024, 64 * 1024, 128 * 1024]
SYNC_EVERY = 200
wcount = 0
last_tile = 0.0
last_cleanup = time.time()


def random_write():
    global wcount
    if IO_DIR is None:
        return
    fname = IO_DIR / f"rnd_{random.randint(0, 999999):06d}.bin"
    with fname.open("ab", buffering=0) as handle:
        handle.write(os.urandom(random.choice(SIZES)))
    wcount += 1
    if seq_file is not None and wcount % SYNC_EVERY == 0:
        try:
            os.fsync(seq_file.fileno())
        except Exception:
            pass


def sequential_append():
    if seq_file is not None:
        seq_file.write(os.urandom(random.choice(SIZES)))


def fetch_tile():
    if not ENABLE_NET:
        return
    try:
        with urllib.request.urlopen(TILE_URL, timeout=3) as response:
            data = response.read()
        if ENABLE_IO and IO_DIR is not None:
            (IO_DIR / "tile.cache").write_bytes(data)
    except Exception:
        pass


def periodic_cleanup():
    global seq_file
    if IO_DIR is None:
        return
    for path in IO_DIR.glob("rnd_*.bin"):
        try:
            path.unlink()
        except Exception:
            pass
    try:
        if seq_file is not None:
            seq_file.close()
    except Exception:
        pass
    try:
        if seq_path is not None and seq_path.exists():
            seq_path.unlink()
    except Exception:
        pass
    if seq_path is not None:
        try:
            seq_file = seq_path.open("ab", buffering=0)
        except Exception:
            seq_file = None


while True:
    now = time.time()

    if ENABLE_IO and IO_DIR is not None:
        random_write()
        sequential_append()
        time.sleep(0.02)
    else:
        time.sleep(0.05)

    if now - last_tile >= TILE_PERIOD:
        fetch_tile()
        last_tile = now

    if ENABLE_IO and now - last_cleanup >= CLEAN_PERIOD:
        periodic_cleanup()
        last_cleanup = time.time()
PY
}

start_perception() {
  generate_perception_py
  local log_path="${LOG_DIR}/bg_perception.out"
  echo "[bg] perception start"
  FEATURE="${FEATURE}" CAM_INDEX="${CAM_INDEX}" TARGET_FPS="${TARGET_FPS}" PERCEPTION_UTIL="${PERCEPTION_UTIL}" \
    nice -n 10 "${PYTHON_BIN}" "${PERCEPTION_PY}" >"${log_path}" 2>&1 &
  pids+=("$!")
  workload_names+=("perception")
  workload_logs+=("${log_path}")
}

start_comms() {
  if ! command -v iperf3 >/dev/null 2>&1; then
    warn "iperf3 not found; skipping communication workload."
    return
  fi
  local log_path="${LOG_DIR}/bg_iperf3.out"
  echo "[bg] comms start"
  IPERF_SERVER="${IPERF_SERVER}" UPLOAD_MBIT="${UPLOAD_MBIT}" BURST_SEC="${BURST_SEC}" \
    IDLE_SEC="${IDLE_SEC}" LOG_PATH="${log_path}" \
    nice -n 10 bash -c '
    set +e
    while true; do
      iperf3 -u -c "$IPERF_SERVER" -b "${UPLOAD_MBIT}M" -t "$BURST_SEC" -l 1200 >"$LOG_PATH" 2>&1
      sleep "$IDLE_SEC"
    done
  ' &
  pids+=("$!")
  workload_names+=("communication")
  workload_logs+=("${log_path}")
}

start_logmap() {
  generate_logmap_py
  local log_path="${LOG_DIR}/bg_logmap.out"
  echo "[bg] logmap start (LOGMAP_IO=${LOGMAP_IO}, LOGMAP_NET=${LOGMAP_NET})"
  LOGMAP_IO="${LOGMAP_IO}" LOGMAP_NET="${LOGMAP_NET}" TILE_URL="${TILE_URL}" \
    TILE_PERIOD_SEC="${TILE_PERIOD_SEC}" CLEAN_PERIOD_SEC="${CLEAN_PERIOD_SEC}" IO_DIR="${IO_DIR}" \
    nice -n 10 "${PYTHON_BIN}" "${LOGMAP_PY}" >"${log_path}" 2>&1 &
  pids+=("$!")
  workload_names+=("logmap")
  workload_logs+=("${log_path}")
}

start_group1() {
  if [[ "${RUN_PERCEPTION}" == "1" ]]; then
    start_perception
  fi
}

start_group2() {
  if [[ "${RUN_COMMS}" == "1" ]]; then
    start_comms
  fi
  if [[ "${RUN_LOGMAP}" == "1" ]]; then
    start_logmap
  fi
}

print_started_summary() {
  echo "[bg] started workloads:"
  if [[ "${#pids[@]}" -eq 0 ]]; then
    echo "  none"
    return
  fi
  local i
  for i in "${!pids[@]}"; do
    echo "  ${workload_names[$i]} pid=${pids[$i]} log=${workload_logs[$i]}"
  done
}

monitor_test_run() {
  local duration_int
  duration_int="$(printf "%.0f" "${DURATION_SEC}")"
  local end_time=$((SECONDS + duration_int))
  echo "[bg] test mode: running for ${DURATION_SEC}s"
  while [[ "${SECONDS}" -lt "${end_time}" ]]; do
    if [[ "${#pids[@]}" -gt 0 ]]; then
      echo "[bg] ps snapshot:"
      ps -o pid,pcpu,pmem,comm -p "$(IFS=,; echo "${pids[*]}")" || true
    fi
    sleep 5
  done
}

cleanup() {
  local exit_code=$?
  if [[ "${#pids[@]}" -gt 0 ]]; then
    echo "[bg] stopping..."
    local pid
    for pid in "${pids[@]}"; do
      kill "${pid}" 2>/dev/null || true
    done
    sleep 1
    for pid in "${pids[@]}"; do
      kill -9 "${pid}" 2>/dev/null || true
    done
  fi
  rm -f "${PERCEPTION_PY:-}" "${LOGMAP_PY:-}" 2>/dev/null || true
  exit "${exit_code}"
}

main() {
  local original_args=("$@")
  parse_args "$@"
  apply_profile_defaults
  resolve_group
  maybe_reexec_with_systemd_scope "${original_args[@]}"
  setup_tmpdir
  validate_config
  print_config

  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "[bg] dry-run: no workloads started."
    echo "[bg] planned workloads:"
    [[ "${RUN_PERCEPTION}" == "1" ]] && echo "  perception"
    [[ "${RUN_COMMS}" == "1" ]] && echo "  communication"
    [[ "${RUN_LOGMAP}" == "1" ]] && echo "  logmap"
    exit 0
  fi

  trap cleanup EXIT INT TERM

  start_group1
  start_group2

  print_started_summary
  echo "[bg] logs: ${LOG_DIR}"
  echo "[bg] CPU/memory check: ps -o pid,pcpu,pmem,comm -p $(IFS=,; echo "${pids[*]:-}")"

  if [[ "${TEST_MODE}" == "1" ]]; then
    monitor_test_run
    echo "[bg] test summary:"
    echo "  group: ${GROUP}"
    echo "  profile: ${PROFILE}"
    echo "  duration: ${DURATION_SEC}s"
    print_started_summary
    echo "  logs: ${LOG_DIR}"
    echo "  check command: ps -o pid,pcpu,pmem,comm -p <pid>"
    return 0
  fi

  wait
}

main "$@"
