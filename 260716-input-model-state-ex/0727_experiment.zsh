#!/usr/bin/env zsh
set -euo pipefail

SCRIPT_DIR="${0:A:h}"

SSH_USER="${SSH_USER:-rasheed}"
SSH_PASSWORD="${SSH_PASSWORD:-modenaottun}"
SSH_PORT="${SSH_PORT:-22}"
JETSON_HOST="${JETSON_HOST:-192.168.0.141}"
REMOTE_REPO_DIR="${REMOTE_REPO_DIR:-/home/rasheed/kuchida/antivenom_infocom}"
REMOTE_PROJECT_DIR="${REMOTE_PROJECT_DIR:-${REMOTE_REPO_DIR}/260716-input-model-state-ex}"
REMOTE_PYTHON="${REMOTE_PYTHON:-${REMOTE_REPO_DIR}/venv/bin/python}"
REMOTE_DATA_DIR="${REMOTE_DATA_DIR:-${REMOTE_REPO_DIR}/iid-data}"
REMOTE_LOG_ROOT="${REMOTE_LOG_ROOT:-logs/input_and_augmentation_state_0727/${JETSON_HOST}/local_ml}"
REMOTE_AUGMENTATION_LOG_DIR="${REMOTE_AUGMENTATION_LOG_DIR:-${REMOTE_LOG_ROOT}/clean_to_strong}"
REMOTE_SHORTCUT_LOG_DIR="${REMOTE_SHORTCUT_LOG_DIR:-${REMOTE_LOG_ROOT}/clean_to_availability_shortcut}"
LOCAL_LOG_DIR="${LOCAL_LOG_DIR:-${SCRIPT_DIR}/collected_logs/input_and_augmentation_state_0727/${JETSON_HOST}/local_ml}"
MONITORING_FPS="${MONITORING_FPS:-50}"
PERF_FPS="${PERF_FPS:-$MONITORING_FPS}"
HARDWARE_FPS="${HARDWARE_FPS:-$MONITORING_FPS}"

DATASET_NAME="uoft-cs/cifar10"
CLIENT_ID="client_1"
AUGMENTATION_SEQUENCES="baseline:strong"
INPUT_SEQUENCES="clean:availability_shortcuts"
TRIALS=1
REFERENCE_TRIALS=0
STAGE_EPOCHS=10
TOTAL_EPOCHS=$((STAGE_EPOCHS * 2))
BATCH_SIZE=16
BASE_SEED=260727
TEST_FRACTION=0.2
TEST_SEED=260626
NUM_ROUNDS=10
LEARNING_RATE=0.001
MODEL_NAME="simple_cnn"
MODEL_DEPTH=3
AUGMENT_JSON='{"enabled":true,"_profile":"baseline","resize":[32,32],"horizontal_flip":false,"normalize":true}'

JETSON_PERF_EVENTS="cycles,instructions,task-clock,context-switches,cpu-migrations,page-faults,br_retired,br_mis_pred_retired,l1d_cache,l1d_cache_refill,l1d_cache_wb,l2d_cache,l2d_cache_refill,l2d_cache_wb,bus_access,mem_access,inst_spec"

ACTION="${1:-both}"

usage() {
  cat <<'EOF'
Usage:
  ./0727_experiment.zsh [check|run|collect|both]

Actions:
  check    Pull the repository and validate CPU, perf, and CIFAR-10 inputs.
  run      Check, then run both continuous 20-epoch sequences.
  collect  Copy the remote logs into this experiment directory.
  both     Check, run, and collect (default).

Fixed experiment:
  device:      192.168.0.141, CPU only
  dataset:     CIFAR-10 IID, client_1, 32x32
  model:       SimpleCNN depth 3
  sequences:   baseline/clean epochs 0-9 -> strong augmentation epochs 10-19
               clean IID epochs 0-9 -> availability shortcut epochs 10-19
  continuity:  one model and one Adam optimizer across each stage boundary
               the two sequences are independent runs initialized with the same seed
  evaluation:  global clean CIFAR-10 test split after every epoch
  monitoring:  perf and psutil at 50 Hz by default
  logs:        one hardware/perf/metrics CSV set per sequence
EOF
}

ssh_base_cmd() {
  if [[ -n "$SSH_PASSWORD" ]]; then
    if ! command -v sshpass >/dev/null 2>&1; then
      print "SSH_PASSWORD is set, but sshpass is not installed." >&2
      exit 1
    fi
    print -- "sshpass -p ${(q)SSH_PASSWORD} ssh -p ${(q)SSH_PORT} -o StrictHostKeyChecking=accept-new -o ConnectTimeout=15 -o ServerAliveInterval=10 -o ServerAliveCountMax=3"
  else
    print -- "ssh -p ${(q)SSH_PORT} -o StrictHostKeyChecking=accept-new -o ConnectTimeout=15 -o ServerAliveInterval=10 -o ServerAliveCountMax=3"
  fi
}

ssh_run() {
  local remote_command="$1"
  local ssh_cmd
  ssh_cmd="$(ssh_base_cmd)"
  eval "$ssh_cmd ${(q)SSH_USER}@${(q)JETSON_HOST} ${(q)remote_command}"
}

enable_and_check_perf() {
  print "==> checking Jetson perf events"
  ssh_run "
    set -e
    printf '%s\n' '$SSH_PASSWORD' | sudo -S sysctl kernel.perf_event_paranoid=-1 >/dev/null
    perf stat -e '$JETSON_PERF_EVENTS' -- '$REMOTE_PYTHON' -c 'sum(i*i for i in range(20000000))' >/dev/null
  "
}

check_environment() {
  print "==> updating remote repository on ${SSH_USER}@${JETSON_HOST}"
  ssh_run "cd '$REMOTE_REPO_DIR' && git pull --rebase"

  print "==> checking ${SSH_USER}@${JETSON_HOST}"
  ssh_run "
    set -e
    test -d '$REMOTE_PROJECT_DIR' || { echo 'missing project: $REMOTE_PROJECT_DIR' >&2; exit 2; }
    test -f '$REMOTE_DATA_DIR/cifar10/partition_metadata.csv' || {
      echo 'missing CIFAR-10 metadata: $REMOTE_DATA_DIR/cifar10/partition_metadata.csv' >&2
      exit 3
    }
    test -f '$REMOTE_DATA_DIR/cifar10/augmented/strong/PREPARED' || {
      echo 'missing strong augmentation: $REMOTE_DATA_DIR/cifar10/augmented/strong/PREPARED' >&2
      exit 3
    }
    test -f '$REMOTE_DATA_DIR/cifar10/poisoned/availability_shortcuts/shortcut_bank.json' || {
      echo 'missing availability shortcut: $REMOTE_DATA_DIR/cifar10/poisoned/availability_shortcuts/shortcut_bank.json' >&2
      exit 3
    }
    cd '$REMOTE_PROJECT_DIR'
    '$REMOTE_PYTHON' --version
    perf --version
    help_text=\$(CUDA_VISIBLE_DEVICES='' '$REMOTE_PYTHON' running_ml.py --help)
    printf '%s\n' "\$help_text" | grep -q -- '--augmentation-sequences' || {
      echo 'remote running_ml.py does not support --augmentation-sequences' >&2
      exit 4
    }
    printf '%s\n' "\$help_text" | grep -q -- '--input-sequences' || {
      echo 'remote running_ml.py does not support --input-sequences' >&2
      exit 4
    }
    printf '%s\n' "\$help_text" | grep -q -- '--stage-epochs' || {
      echo 'remote running_ml.py does not support --stage-epochs' >&2
      exit 4
    }
    CUDA_VISIBLE_DEVICES='' '$REMOTE_PYTHON' -c "
from pathlib import Path
import torch
from dataset_preparation import get_num_classes, load_metadata_records
assert not torch.cuda.is_available()
clean_train = load_metadata_records(data_dir='$REMOTE_DATA_DIR', dataset_name='$DATASET_NAME', client_id='$CLIENT_ID', poisoning_method='clean', split='train')
clean_test = load_metadata_records(data_dir='$REMOTE_DATA_DIR', dataset_name='$DATASET_NAME', client_id='all', poisoning_method='clean', split='test')
shortcut_train = load_metadata_records(data_dir='$REMOTE_DATA_DIR', dataset_name='$DATASET_NAME', client_id='$CLIENT_ID', poisoning_method='availability_shortcuts', split='train')
assert clean_train and clean_test and shortcut_train
assert Path('$REMOTE_DATA_DIR/cifar10/augmented/strong/PREPARED').is_file()
print('device=cpu')
print('num_classes=', get_num_classes('$REMOTE_DATA_DIR', '$DATASET_NAME'))
print('clean_train_client=', len(clean_train))
print('shortcut_train_client=', len(shortcut_train))
print('global_clean_test=', len(clean_test))
"
  "
  enable_and_check_perf
}

run_experiment() {
  print "==> running two continuous CIFAR-10 state-transition experiments"
  print "    sequence 1=baseline(10) -> strong(10)"
  print "    sequence 2=clean(10) -> availability_shortcuts(10)"
  print "    model=${MODEL_NAME} client=${CLIENT_ID} batch_size=${BATCH_SIZE}"
  print "    perf_fps=${PERF_FPS} hardware_fps=${HARDWARE_FPS}"
  print "    remote logs=${REMOTE_LOG_ROOT}"

  ssh_run "
    set -e
    cd '$REMOTE_PROJECT_DIR'
    run_marker=\$(mktemp)

    echo 'running baseline -> strong augmentation'
    CUDA_VISIBLE_DEVICES='' '$REMOTE_PYTHON' running_ml.py \
      --experiment-id 'cifar10_augmentation_baseline_to_strong' \
      --dataset '$DATASET_NAME' \
      --dataset-split train \
      --data-dir '$REMOTE_DATA_DIR' \
      --log-dir '$REMOTE_AUGMENTATION_LOG_DIR' \
      --client-id '$CLIENT_ID' \
      --device-id '$JETSON_HOST' \
      --host '$JETSON_HOST' \
      --augmentation-sequences '$AUGMENTATION_SEQUENCES' \
      --stage-epochs '$STAGE_EPOCHS' \
      --test-fraction '$TEST_FRACTION' \
      --test-seed '$TEST_SEED' \
      --reference-trials '$REFERENCE_TRIALS' \
      --trials '$TRIALS' \
      --local-epochs '$STAGE_EPOCHS' \
      --batch-size '$BATCH_SIZE' \
      --num-rounds '$NUM_ROUNDS' \
      --learning-rate '$LEARNING_RATE' \
      --seed '$BASE_SEED' \
      --perf-events '$JETSON_PERF_EVENTS' \
      --perf-fps '$PERF_FPS' \
      --hardware-fps '$HARDWARE_FPS' \
      --augment '$AUGMENT_JSON' \
      --model '$MODEL_NAME' \
      --model-depth '$MODEL_DEPTH' \
      --model-width-multiplier 1.0 \
      --model-target-pam-mb 0

    echo 'running clean -> availability shortcut'
    CUDA_VISIBLE_DEVICES='' '$REMOTE_PYTHON' running_ml.py \
      --experiment-id 'cifar10_clean_to_availability_shortcut' \
      --dataset '$DATASET_NAME' \
      --dataset-split train \
      --data-dir '$REMOTE_DATA_DIR' \
      --log-dir '$REMOTE_SHORTCUT_LOG_DIR' \
      --client-id '$CLIENT_ID' \
      --device-id '$JETSON_HOST' \
      --host '$JETSON_HOST' \
      --input-sequences '$INPUT_SEQUENCES' \
      --stage-epochs '$STAGE_EPOCHS' \
      --test-fraction '$TEST_FRACTION' \
      --test-seed '$TEST_SEED' \
      --reference-trials '$REFERENCE_TRIALS' \
      --trials '$TRIALS' \
      --local-epochs '$STAGE_EPOCHS' \
      --batch-size '$BATCH_SIZE' \
      --num-rounds '$NUM_ROUNDS' \
      --learning-rate '$LEARNING_RATE' \
      --seed '$BASE_SEED' \
      --perf-events '$JETSON_PERF_EVENTS' \
      --perf-fps '$PERF_FPS' \
      --hardware-fps '$HARDWARE_FPS' \
      --augment '$AUGMENT_JSON' \
      --model '$MODEL_NAME' \
      --model-depth '$MODEL_DEPTH' \
      --model-width-multiplier 1.0 \
      --model-target-pam-mb 0

    validate_sequence() {
      log_dir=\"\$1\"
      sequence_label=\"\$2\"
      first_annotation=\"\$3\"
      second_annotation=\"\$4\"

      perf_files=\$(find \"\$log_dir\" -maxdepth 1 -type f -name '*_perf.csv' -newer \"\$run_marker\" -print)
      perf_file_count=\$(printf '%s\n' \"\$perf_files\" | sed '/^[[:space:]]*\$/d' | wc -l)
      test \"\$perf_file_count\" -eq 1 || {
        echo \"expected 1 continuous perf file for \$sequence_label, found \$perf_file_count\" >&2
        exit 6
      }
      perf_file=\$(printf '%s\n' \"\$perf_files\" | head -n 1)
      grep -q ',ok,' \"\$perf_file\" || {
        echo \"perf produced no successful rows: \$perf_file\" >&2
        exit 5
      }
      grep -q \"\$first_annotation\" \"\$perf_file\" || {
        echo \"perf log lacks first-stage annotation \$first_annotation: \$perf_file\" >&2
        exit 5
      }
      grep -q \"\$second_annotation\" \"\$perf_file\" || {
        echo \"perf log lacks second-stage annotation \$second_annotation: \$perf_file\" >&2
        exit 5
      }

      metrics_files=\$(find \"\$log_dir\" -maxdepth 1 -type f -name '*_metrics.csv' -newer \"\$run_marker\" -print)
      metrics_file_count=\$(printf '%s\n' \"\$metrics_files\" | sed '/^[[:space:]]*\$/d' | wc -l)
      test \"\$metrics_file_count\" -eq 1 || {
        echo \"expected 1 continuous metrics file for \$sequence_label, found \$metrics_file_count\" >&2
        exit 8
      }
      metrics_file=\$(printf '%s\n' \"\$metrics_files\" | head -n 1)
      eval_count=\$(grep -c 'clean_test_epoch' \"\$metrics_file\" || true)
      test \"\$eval_count\" -eq '$TOTAL_EPOCHS' || {
        echo \"expected $TOTAL_EPOCHS clean evaluations, found \$eval_count: \$metrics_file\" >&2
        exit 7
      }
      grep -q \"\$sequence_label\" \"\$metrics_file\" || {
        echo \"metrics log lacks sequence annotation \$sequence_label: \$metrics_file\" >&2
        exit 7
      }
    }

    validate_sequence \
      '$REMOTE_AUGMENTATION_LOG_DIR' \
      'augmentation_baseline_to_strong' \
      ',baseline,' \
      ',strong,'
    validate_sequence \
      '$REMOTE_SHORTCUT_LOG_DIR' \
      'clean_to_availability_shortcuts' \
      ',clean,' \
      ',availability_shortcuts,'
    rm -f "\$run_marker"
    echo 'validated both continuous 20-epoch runs'
  "
}

collect_logs() {
  mkdir -p "$LOCAL_LOG_DIR"
  local -a rsync_command
  rsync_command=(
    rsync -az --partial --stats --human-readable
    -e "ssh -p ${SSH_PORT} -o StrictHostKeyChecking=accept-new -o ConnectTimeout=15"
    "${SSH_USER}@${JETSON_HOST}:${REMOTE_PROJECT_DIR}/${REMOTE_LOG_ROOT}/"
    "${LOCAL_LOG_DIR}/"
  )
  print "==> collecting logs into ${LOCAL_LOG_DIR}"
  if [[ -n "$SSH_PASSWORD" ]]; then
    sshpass -p "$SSH_PASSWORD" "${rsync_command[@]}"
  else
    "${rsync_command[@]}"
  fi
}

case "$ACTION" in
  check)
    check_environment
    ;;
  run)
    check_environment
    run_experiment
    ;;
  collect)
    collect_logs
    ;;
  both)
    check_environment
    run_experiment
    collect_logs
    ;;
  -h|--help|help)
    usage
    ;;
  *)
    usage >&2
    exit 1
    ;;
esac
