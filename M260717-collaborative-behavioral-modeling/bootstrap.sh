#!/usr/bin/env bash
set -Eeuo pipefail

umask 022


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

readonly ROOT="/local/antivenom"
readonly STATE_DIR="${ROOT}/state"
readonly LOG_DIR="${ROOT}/logs"
readonly METADATA_DIR="${ROOT}/metadata"
readonly DATASET_DIR="${ROOT}/datasets"
readonly RESULT_DIR="${ROOT}/results"
readonly VENV="${ROOT}/venv"

# bootstrap.sh が置かれているディレクトリを自動的に取得する。
readonly REPO_DIR="$(
    cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
    pwd -P
)"

readonly REQUIREMENTS="${REPO_DIR}/requirements.txt"
readonly TORCH_REQUIREMENTS="${REPO_DIR}/requirements-torch-cpu.txt"

readonly READY_FILE="${ROOT}/READY"
readonly READY_TMP="${ROOT}/READY.tmp"
readonly FAILED_FILE="${ROOT}/FAILED"
readonly FAILED_TMP="${ROOT}/FAILED.tmp"

# 1の場合、perfを利用できなければbootstrapを失敗させる。
# 必要に応じてprofile.py側から0を設定できる。
readonly REQUIRE_PERF="${ANTIVENOM_REQUIRE_PERF:-1}"


# ---------------------------------------------------------------------------
# Initial setup
# ---------------------------------------------------------------------------

mkdir -p \
    "${STATE_DIR}" \
    "${LOG_DIR}" \
    "${METADATA_DIR}" \
    "${DATASET_DIR}" \
    "${RESULT_DIR}"

chmod 0755 \
    "${STATE_DIR}" \
    "${METADATA_DIR}"

chmod 1777 \
    "${LOG_DIR}" \
    "${DATASET_DIR}" \
    "${RESULT_DIR}"

exec >> "${LOG_DIR}/bootstrap.log" 2>&1

# 前回実行時の状態ファイルを消す。
rm -f \
    "${READY_FILE}" \
    "${READY_TMP}" \
    "${FAILED_FILE}" \
    "${FAILED_TMP}"


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

on_error() {
    local exit_code=$?
    local line_number="${1:-unknown}"
    local failed_command="${2:-unknown}"

    # エラーハンドラ内で再びERR trapが走らないようにする。
    trap - ERR
    set +e

    {
        printf 'status=failed\n'
        printf 'exit_code=%s\n' "${exit_code}"
        printf 'line=%s\n' "${line_number}"
        printf 'command=%q\n' "${failed_command}"
        printf 'failed_at=%s\n' "$(date --iso-8601=seconds)"
        printf 'hostname=%s\n' "$(hostname 2>/dev/null || true)"
        printf 'cluster=%s\n' "${ANTIVENOM_CLUSTER:-unknown}"
        printf 'hardware_type=%s\n' \
            "${ANTIVENOM_HARDWARE_TYPE:-unknown}"
    } > "${FAILED_TMP}"

    mv -f "${FAILED_TMP}" "${FAILED_FILE}"
    rm -f "${READY_FILE}" "${READY_TMP}"

    echo
    echo "Bootstrap failed"
    echo "  exit code: ${exit_code}"
    echo "  line:      ${line_number}"
    echo "  command:   ${failed_command}"
    echo "  details:   ${FAILED_FILE}"
    echo "  log:       ${LOG_DIR}/bootstrap.log"

    exit "${exit_code}"
}

trap 'on_error "${LINENO}" "${BASH_COMMAND}"' ERR

# 実行したコマンド、ファイル、行番号、時刻をログに残す。
export PS4='+ $(date --iso-8601=seconds) ${BASH_SOURCE}:${LINENO}: '
set -x


# ---------------------------------------------------------------------------
# Preconditions
# ---------------------------------------------------------------------------

echo "Bootstrap started: $(date --iso-8601=seconds)"
echo "Repository directory: ${REPO_DIR}"
echo "Cluster: ${ANTIVENOM_CLUSTER:-unknown}"
echo "Hardware type: ${ANTIVENOM_HARDWARE_TYPE:-unknown}"

if [[ "${EUID}" -ne 0 ]]; then
    echo "bootstrap.sh must run as root" >&2
    exit 1
fi

if [[ ! -f "${REQUIREMENTS}" ]]; then
    echo "Requirements file not found: ${REQUIREMENTS}" >&2
    exit 1
fi

if [[ ! -f "${TORCH_REQUIREMENTS}" ]]; then
    echo "Torch requirements file not found: ${TORCH_REQUIREMENTS}" >&2
    exit 1
fi

if ! command -v apt-get >/dev/null 2>&1; then
    echo "This bootstrap currently requires an apt-based OS image" >&2
    exit 1
fi

export DEBIAN_FRONTEND=noninteractive


# ---------------------------------------------------------------------------
# System packages
# ---------------------------------------------------------------------------

readonly REQUIRED_PACKAGES=(
    git
    jq
    numactl
    procps
    python3
    python3-pip
    python3-venv
    util-linux
    linux-tools-common
)

missing_packages=()

for package in "${REQUIRED_PACKAGES[@]}"; do
    if ! dpkg-query -W \
        -f='${db:Status-Abbrev}' \
        "${package}" 2>/dev/null |
        grep -q '^ii '; then
        missing_packages+=("${package}")
    fi
done

if (( ${#missing_packages[@]} > 0 )); then
    apt-get update
    apt-get install -y --no-install-recommends \
        "${missing_packages[@]}"
fi


# ---------------------------------------------------------------------------
# perf
# ---------------------------------------------------------------------------

perf_works() {
    command -v perf >/dev/null 2>&1 &&
        perf --version >/dev/null 2>&1
}

if ! perf_works; then
    # 実行中のカーネルと完全に一致するパッケージを最初に試す。
    if ! apt-get install -y --no-install-recommends \
        "linux-tools-$(uname -r)"; then
        echo "Exact linux-tools package is unavailable for $(uname -r)"
    fi
fi

if ! perf_works; then
    # exact packageがない場合のfallback。
    if ! apt-get install -y --no-install-recommends \
        linux-tools-generic; then
        echo "linux-tools-generic could not be installed"
    fi
fi

if perf_works; then
    cat > /etc/sysctl.d/99-antivenom-perf.conf <<'EOF'
kernel.perf_event_paranoid=-1
EOF

    sysctl -p /etc/sysctl.d/99-antivenom-perf.conf

    perf --version > "${METADATA_DIR}/perf_version.txt"
else
    printf '%s\n' \
        "perf is unavailable for kernel $(uname -r)" \
        > "${METADATA_DIR}/perf_unavailable.txt"

    if [[ "${REQUIRE_PERF}" == "1" ]]; then
        echo "perf is required but is not operational" >&2
        exit 1
    fi
fi


# ---------------------------------------------------------------------------
# Python environment
# ---------------------------------------------------------------------------

venv_created=0

if [[ ! -x "${VENV}/bin/python" ]]; then
    rm -rf "${VENV}"
    python3 -m venv "${VENV}"
    venv_created=1
fi

readonly REQUIREMENTS_HASH="$(
    sha256sum "${REQUIREMENTS}" |
        awk '{print $1}'
)"

readonly REQUIREMENTS_STATE="${STATE_DIR}/requirements.sha256"

installed_hash=""

if [[ -f "${REQUIREMENTS_STATE}" ]]; then
    installed_hash="$(cat "${REQUIREMENTS_STATE}")"
fi

if [[ "${venv_created}" == "1" ||
      "${REQUIREMENTS_HASH}" != "${installed_hash}" ]]; then
    "${VENV}/bin/python" -m pip install \
        --disable-pip-version-check \
        --upgrade pip

    # 通常の依存ライブラリ
    "${VENV}/bin/python" -m pip install \
        --disable-pip-version-check \
        -r "${REQUIREMENTS}"

    # PyTorch CPU版
    "${VENV}/bin/python" -m pip install \
        --disable-pip-version-check \
        -r "${TORCH_REQUIREMENTS}"

    "${VENV}/bin/python" -m pip check


    printf '%s\n' "${REQUIREMENTS_HASH}" \
        > "${REQUIREMENTS_STATE}.tmp"

    mv -f \
        "${REQUIREMENTS_STATE}.tmp" \
        "${REQUIREMENTS_STATE}"
fi


# ---------------------------------------------------------------------------
# Optional metadata collection
#
# Metadata commands should not cause the entire bootstrap to fail.
# Any error is recorded in a corresponding .error file.
# ---------------------------------------------------------------------------

capture_optional() {
    local output_file="$1"
    shift

    local error_file="${output_file}.error"
    local status

    rm -f "${error_file}"

    if "$@" > "${output_file}" 2>&1; then
        return 0
    else
        status=$?

        {
            printf 'exit_code=%s\n' "${status}"
            printf 'command='
            printf '%q ' "$@"
            printf '\n'
        } > "${error_file}"

        return 0
    fi
}


capture_optional \
    "${METADATA_DIR}/hostname.txt" \
    hostname -f

capture_optional \
    "${METADATA_DIR}/uname.txt" \
    uname -a

capture_optional \
    "${METADATA_DIR}/architecture.txt" \
    uname -m

capture_optional \
    "${METADATA_DIR}/kernel_release.txt" \
    uname -r

capture_optional \
    "${METADATA_DIR}/os-release.txt" \
    cat /etc/os-release

capture_optional \
    "${METADATA_DIR}/kernel_cmdline.txt" \
    cat /proc/cmdline

capture_optional \
    "${METADATA_DIR}/lscpu.txt" \
    lscpu

# 古いlscpuでは-Jが利用できないことがある。
capture_optional \
    "${METADATA_DIR}/lscpu.json" \
    lscpu -J

# 列名の対応状況がデバイスやutil-linuxのバージョンで異なるため、
# 特定の列を固定せず、利用可能な全列を出力する。
capture_optional \
    "${METADATA_DIR}/cpu_topology.txt" \
    lscpu -e

capture_optional \
    "${METADATA_DIR}/memory.txt" \
    free -b

capture_optional \
    "${METADATA_DIR}/numa.txt" \
    numactl --hardware

capture_optional \
    "${METADATA_DIR}/git_commit.txt" \
    git -C "${REPO_DIR}" rev-parse HEAD

capture_optional \
    "${METADATA_DIR}/git_status.txt" \
    git -C "${REPO_DIR}" status --short

capture_optional \
    "${METADATA_DIR}/dpkg_packages.txt" \
    dpkg-query -W

capture_optional \
    "${METADATA_DIR}/pip_freeze.txt" \
    "${VENV}/bin/python" -m pip freeze

{
    printf 'cluster=%s\n' "${ANTIVENOM_CLUSTER:-unknown}"
    printf 'hardware_type=%s\n' \
        "${ANTIVENOM_HARDWARE_TYPE:-unknown}"
} > "${METADATA_DIR}/cloudlab_selection.txt"


# ---------------------------------------------------------------------------
# PyTorch validation
#
# requirements-cloudlab.txt にtorchが含まれている前提。
# importできなければ実験環境が未完成なのでbootstrapを失敗させる。
# ---------------------------------------------------------------------------

"${VENV}/bin/python" - \
    > "${METADATA_DIR}/torch_config.txt" <<'PY'
import platform

import torch

print("platform:", platform.platform())
print("machine:", platform.machine())
print("python_version:", platform.python_version())
print("torch_version:", torch.__version__)
print("num_threads:", torch.get_num_threads())
print("interop_threads:", torch.get_num_interop_threads())
print("cuda_available:", torch.cuda.is_available())
print("torch_config:")
print(torch.__config__.show())
PY


# ---------------------------------------------------------------------------
# Completion
# ---------------------------------------------------------------------------

set +x

{
    printf 'status=ready\n'
    printf 'completed_at=%s\n' "$(date --iso-8601=seconds)"
    printf 'hostname=%s\n' "$(hostname 2>/dev/null || true)"
    printf 'cluster=%s\n' "${ANTIVENOM_CLUSTER:-unknown}"
    printf 'hardware_type=%s\n' \
        "${ANTIVENOM_HARDWARE_TYPE:-unknown}"
    printf 'repository=%s\n' "${REPO_DIR}"
    printf 'requirements_hash=%s\n' "${REQUIREMENTS_HASH}"
} > "${READY_TMP}"

mv -f "${READY_TMP}" "${READY_FILE}"

trap - ERR
rm -f "${FAILED_FILE}" "${FAILED_TMP}"

echo "Bootstrap finished: $(date --iso-8601=seconds)"
echo "Ready file: ${READY_FILE}"
