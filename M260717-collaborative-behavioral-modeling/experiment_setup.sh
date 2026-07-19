#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd -P)"
PYTHON_BIN="${PYTHON:-${REPO_DIR}/venv/bin/python}"
DATASET_PREPARATION="${DATASET_PREPARATION:-${REPO_DIR}/dataset_preparation.py}"
ROBUSTNESS_PROJECT_DIR="${ROBUSTNESS_PROJECT_DIR:-${REPO_DIR}/M260718-robustness}"
IID_DATA_DIR="${IID_DATA_DIR:-${REPO_DIR}/iid-data}"
NONIID_DATA_DIR="${NONIID_DATA_DIR:-${REPO_DIR}/noniid-data}"
UNLEARNABLE_REPO="${UNLEARNABLE_REPO:-${REPO_DIR}/Unlearnable-Examples}"

NUM_CLIENTS="${NUM_CLIENTS:-10}"
NONIID_ALPHA="${NONIID_ALPHA:-0.3}"
SEED="${SEED:-260626}"
TEST_FRACTION="${TEST_FRACTION:-0.2}"
TEST_SEED="${TEST_SEED:-260626}"
PREPARE_SCENARIOS="${PREPARE_SCENARIOS:-all}"
AUGMENT='{"enabled":true,"resize":[224,224],"horizontal_flip":true,"normalize":true}'

DATASETS=(
  "kuchidareo/small_trashnet"
  "kuchidareo/chinese_trafficsign_dataset"
  "uoft-cs/cifar10"
)

check_environment() {
  [[ -x "$PYTHON_BIN" ]] || {
    printf 'Python is not executable: %s\n' "$PYTHON_BIN" >&2
    return 1
  }
  [[ -f "$DATASET_PREPARATION" ]] || {
    printf 'Dataset preparation script is missing: %s\n' "$DATASET_PREPARATION" >&2
    return 1
  }
  [[ -d "$UNLEARNABLE_REPO" ]] || {
    printf 'Unlearnable-Examples repository is missing: %s\n' "$UNLEARNABLE_REPO" >&2
    return 1
  }
  [[ -d "$ROBUSTNESS_PROJECT_DIR" ]] || {
    printf 'Robustness project is missing: %s\n' "$ROBUSTNESS_PROJECT_DIR" >&2
    return 1
  }

  "$PYTHON_BIN" --version
  (
    cd "$ROBUSTNESS_PROJECT_DIR"
    "$PYTHON_BIN" -c 'import datasets, numpy, PIL, sklearn, torch, torchvision; print("dataset dependencies: ok")'
  )
  run_dataset_preparation --help >/dev/null

  printf 'repository: %s\n' "$REPO_DIR"
  printf 'IID output: %s\n' "$IID_DATA_DIR"
  printf 'non-IID output: %s\n' "$NONIID_DATA_DIR"
  printf 'datasets: %s\n' "${DATASETS[*]}"
  printf 'scenarios: %s\n' "$PREPARE_SCENARIOS"
}

run_dataset_preparation() {
  (
    cd "$ROBUSTNESS_PROJECT_DIR"
    "$PYTHON_BIN" -c '
import runpy
import sys

script = sys.argv.pop(1)
sys.argv[0] = script
runpy.run_path(script, run_name="__main__")
' "$DATASET_PREPARATION" "$@"
  )
}

prepare_all() {
  local dataset="$1"
  local data_dir="$2"
  local partition_method="$3"

  printf '\nPreparing dataset\n'
  printf '  dataset: %s\n' "$dataset"
  printf '  data_dir: %s\n' "$data_dir"
  printf '  partition_method: %s\n' "$partition_method"

  run_dataset_preparation \
      --dataset "$dataset" \
      --data-dir "$data_dir" \
      --num-clients "$NUM_CLIENTS" \
      --partition-method "$partition_method" \
      --noniid-alpha "$NONIID_ALPHA" \
      --prepare-scenarios "$PREPARE_SCENARIOS" \
      --seed "$SEED" \
      --test-fraction "$TEST_FRACTION" \
      --test-seed "$TEST_SEED" \
      --augment "$AUGMENT" \
      --unlearnable-repo "$UNLEARNABLE_REPO"
}

prepare_iid() {
  local dataset
  mkdir -p "$IID_DATA_DIR"
  for dataset in "${DATASETS[@]}"; do
    prepare_all "$dataset" "$IID_DATA_DIR" iid
  done
}

prepare_noniid() {
  local dataset
  mkdir -p "$NONIID_DATA_DIR"
  for dataset in "${DATASETS[@]}"; do
    prepare_all "$dataset" "$NONIID_DATA_DIR" dirichlet_noniid
  done
}

usage() {
  cat <<'EOF'
Usage:
  ./experiment_setup.sh check
  ./experiment_setup.sh iid
  ./experiment_setup.sh noniid
  ./experiment_setup.sh all

The default mode is all. Existing complete datasets are reused; this script
does not pass --force and therefore does not intentionally overwrite them.

Environment overrides:
  PYTHON=/path/to/venv/bin/python
  IID_DATA_DIR=/path/to/iid-data
  NONIID_DATA_DIR=/path/to/noniid-data
  PREPARE_SCENARIOS=all
  NUM_CLIENTS=10
  NONIID_ALPHA=0.3
EOF
}

main() {
  local mode="${1:-all}"
  case "$mode" in
    -h|--help|help)
      usage
      return
      ;;
  esac
  check_environment

  case "$mode" in
    check)
      ;;
    iid)
      prepare_iid
      ;;
    noniid|non-iid)
      prepare_noniid
      ;;
    all)
      prepare_iid
      prepare_noniid
      ;;
    *)
      printf 'Unknown mode: %s\n' "$mode" >&2
      usage >&2
      return 1
      ;;
  esac
}

main "$@"
