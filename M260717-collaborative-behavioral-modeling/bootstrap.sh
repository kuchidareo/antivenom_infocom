#!/usr/bin/env zsh
set -Eeuo pipefail

ROOT=/local/antivenom
STATE_DIR="${ROOT}/state"
LOG_DIR="${ROOT}/logs"
METADATA_DIR="${ROOT}/metadata"
DATASET_DIR="${ROOT}/datasets"
RESULT_DIR="${ROOT}/results"
VENV="${ROOT}/venv"

mkdir -p \
  "${STATE_DIR}" \
  "${LOG_DIR}" \
  "${METADATA_DIR}" \
  "${DATASET_DIR}" \
  "${RESULT_DIR}"

chmod 0777 \
  "${LOG_DIR}" \
  "${METADATA_DIR}" \
  "${DATASET_DIR}" \
  "${RESULT_DIR}"

exec >> "${LOG_DIR}/bootstrap.log" 2>&1

echo "Bootstrap started: $(date --iso-8601=seconds)"

export DEBIAN_FRONTEND=noninteractive

# perfと基本ツール
if ! command -v perf >/dev/null 2>&1; then
    apt-get update

    apt-get install -y \
      git \
      python3 \
      python3-pip \
      python3-venv \
      numactl \
      jq \
      linux-tools-common

    if ! apt-get install -y "linux-tools-$(uname -r)"; then
        apt-get install -y linux-tools-generic
    fi
fi

# Experiment内でperfを一般ユーザーから利用可能にする
echo "kernel.perf_event_paranoid=-1" \
  > /etc/sysctl.d/99-antivenom-perf.conf

sysctl -p /etc/sysctl.d/99-antivenom-perf.conf

# Python環境
if [[ ! -x "${VENV}/bin/python" ]]; then
    python3 -m venv "${VENV}"
    "${VENV}/bin/pip" install --upgrade pip
fi

"${VENV}/bin/pip" install \
  -r /local/repository/requirements-cloudlab.txt

# Metadata
hostname -f > "${METADATA_DIR}/hostname.txt"
uname -a > "${METADATA_DIR}/uname.txt"
lscpu -J > "${METADATA_DIR}/lscpu.json"
lscpu -e=CPU,CORE,SOCKET,NODE,CACHE,ONLINE \
  > "${METADATA_DIR}/cpu_topology.csv"
free -b > "${METADATA_DIR}/memory.txt"
numactl --hardware > "${METADATA_DIR}/numa.txt"
perf --version > "${METADATA_DIR}/perf_version.txt"

git -C /local/repository rev-parse HEAD \
  > "${METADATA_DIR}/git_commit.txt"

"${VENV}/bin/pip" freeze \
  > "${METADATA_DIR}/pip_freeze.txt"

"${VENV}/bin/python" - <<'PY' \
  > "${METADATA_DIR}/torch_config.txt"
import torch

print("torch_version:", torch.__version__)
print("num_threads:", torch.get_num_threads())
print("interop_threads:", torch.get_num_interop_threads())
print(torch.__config__.show())
PY

date --iso-8601=seconds > "${ROOT}/READY"

echo "Bootstrap finished: $(date --iso-8601=seconds)"
