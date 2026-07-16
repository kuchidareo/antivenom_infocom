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
REMOTE_LOG_DIR="${REMOTE_LOG_DIR:-logs/input_model_state_0716/${JETSON_HOST}/local_ml}"
LOCAL_LOG_DIR="${LOCAL_LOG_DIR:-${SCRIPT_DIR}/collected_logs/${JETSON_HOST}/local_ml}"
MONITORING_FPS="${MONITORING_FPS:-50}"
PERF_FPS="${PERF_FPS:-$MONITORING_FPS}"
HARDWARE_FPS="${HARDWARE_FPS:-$MONITORING_FPS}"

# client_0 keeps the dataset partition and poisoned samples fixed so that the
# input order and prior model-training condition are the intended variables.
DATASET_NAME="kuchidareo/small_trashnet"
CLIENT_ID="client_0"
INPUT_SEQUENCES="clean:availability_shortcuts,availability_shortcuts:clean"
TRIALS=1
REFERENCE_TRIALS=0
STAGE_EPOCHS=10
TOTAL_EPOCHS=$((STAGE_EPOCHS * 2))
BATCH_SIZE=16
BASE_SEED=260626
TEST_FRACTION=0.2
TEST_SEED=260626
NUM_ROUNDS=10
LEARNING_RATE=0.001
MODEL_NAME="simple_cnn"
MODEL_DEPTH=5
AUGMENT_JSON='{"enabled":true,"resize":[224,224],"horizontal_flip":true,"normalize":true}'

JETSON_PERF_EVENTS="cycles,instructions,task-clock,context-switches,cpu-migrations,page-faults,br_retired,br_mis_pred_retired,l1d_cache,l1d_cache_refill,l1d_cache_wb,l2d_cache,l2d_cache_refill,l2d_cache_wb,bus_access,mem_access,inst_spec"

ACTION="${1:-both}"

usage() {
  cat <<'EOF'
Usage:
  ./0716_experiment.zsh [check|run|collect|both]

Actions:
  check    Check the Jetson CPU environment, dataset, and perf event list.
  run      Check and run both continuous input/model-state sequences.
  collect  Copy the remote logs into this experiment's collected_logs directory.
  both     Check, run, and collect (default).

Fixed comparison condition:
  device:      192.168.0.141, CPU only
  partition:   client_0
  dataset:     kuchidareo/small_trashnet
  model:       simple_cnn
  sequences:   clean(10) -> availability_shortcuts(10)
               availability_shortcuts(10) -> clean(10)
  trials:      1 analysis trial, 0 reference trials
  training:    20 continuous epochs per sequence, batch size 16, seed 260626
               one model and one Adam optimizer are retained across both stages
  evaluation:  same global clean test split after every epoch
               test fraction 20%, seed 260626; poisoned test rows are forbidden
  monitoring:  perf and psutil both 50 FPS by default (20 ms intervals)
               override together with MONITORING_FPS, or separately with
               PERF_FPS and HARDWARE_FPS
  augmentation: resize 224x224, horizontal flip, normalize
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
  ssh_run "
    set -e
    cd '$REMOTE_REPO_DIR'
    git pull --rebase
  "

  print "==> checking ${SSH_USER}@${JETSON_HOST}"
  ssh_run "
    set -e
    test -d '$REMOTE_PROJECT_DIR' || { echo 'missing project: $REMOTE_PROJECT_DIR' >&2; exit 2; }
    test -d '$REMOTE_DATA_DIR/small_trashnet' || { echo 'missing dataset: $REMOTE_DATA_DIR/small_trashnet' >&2; exit 3; }
    cd '$REMOTE_PROJECT_DIR'
    CUDA_VISIBLE_DEVICES='' '$REMOTE_PYTHON' dataset_preparation.py \
      --dataset '$DATASET_NAME' \
      --data-dir '$REMOTE_DATA_DIR' \
      --prepare-scenarios all \
      --test-fraction '$TEST_FRACTION' \
      --test-seed '$TEST_SEED'
    '$REMOTE_PYTHON' --version
    perf --version
    help_text=\$(CUDA_VISIBLE_DEVICES='' '$REMOTE_PYTHON' running_ml.py --help)
    printf '%s\n' \"\$help_text\" | grep -q -- '--perf-fps' || {
      echo 'remote running_ml.py does not support --perf-fps' >&2
      exit 4
    }
    printf '%s\n' \"\$help_text\" | grep -q -- '--hardware-fps' || {
      echo 'remote running_ml.py does not support --hardware-fps' >&2
      exit 4
    }
    printf '%s\n' \"\$help_text\" | grep -q -- '--input-sequences' || {
      echo 'remote running_ml.py does not support --input-sequences' >&2
      exit 4
    }
    printf '%s\n' \"\$help_text\" | grep -q -- '--stage-epochs' || {
      echo 'remote running_ml.py does not support --stage-epochs' >&2
      exit 4
    }
    CUDA_VISIBLE_DEVICES='' '$REMOTE_PYTHON' -c \"import torch; from dataset_preparation import get_num_classes, load_metadata_records; assert not torch.cuda.is_available(); clean_train=load_metadata_records(data_dir='$REMOTE_DATA_DIR',dataset_name='$DATASET_NAME',client_id='$CLIENT_ID',poisoning_method='clean',split='train'); shortcut_train=load_metadata_records(data_dir='$REMOTE_DATA_DIR',dataset_name='$DATASET_NAME',client_id='$CLIENT_ID',poisoning_method='availability_shortcuts',split='train'); clean_test=load_metadata_records(data_dir='$REMOTE_DATA_DIR',dataset_name='$DATASET_NAME',client_id='all',poisoning_method='clean',split='test'); poisoned_test=load_metadata_records(data_dir='$REMOTE_DATA_DIR',dataset_name='$DATASET_NAME',client_id='all',poisoning_method='availability_shortcuts',split='test'); assert clean_train and shortcut_train and clean_test; assert not poisoned_test; print('device=cpu'); print('torch_threads=',torch.get_num_threads()); print('num_classes=',get_num_classes('$REMOTE_DATA_DIR','$DATASET_NAME')); print('clean_train_client=',len(clean_train)); print('shortcut_train_client=',len(shortcut_train)); print('global_clean_test=',len(clean_test)); print('poisoned_test=',len(poisoned_test))\"
  "
  enable_and_check_perf
}

run_experiment() {
  print "==> running continuous input/model-state comparison on Jetson CPU"
  print "    dataset=${DATASET_NAME} model=${MODEL_NAME} client=${CLIENT_ID}"
  print "    sequences=${INPUT_SEQUENCES} stage_epochs=${STAGE_EPOCHS} total_epochs=${TOTAL_EPOCHS} trials=${TRIALS}"
  print "    evaluation=global clean test split after every epoch"
  print "    perf_fps=${PERF_FPS} hardware_fps=${HARDWARE_FPS}"
  print "    remote logs=${REMOTE_LOG_DIR}"

  ssh_run "
    set -e
    cd '$REMOTE_PROJECT_DIR'
    run_marker=\$(mktemp)
    CUDA_VISIBLE_DEVICES='' '$REMOTE_PYTHON' running_ml.py \
      --dataset '$DATASET_NAME' \
      --dataset-split train \
      --data-dir '$REMOTE_DATA_DIR' \
      --log-dir '$REMOTE_LOG_DIR' \
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

    perf_file_count=0
    for perf_file in \$(find '$REMOTE_LOG_DIR' -maxdepth 1 -type f -name '*_perf.csv' -newer \"\$run_marker\" -print); do
      perf_file_count=\$((perf_file_count + 1))
      grep -q ',ok,' \"\$perf_file\" || {
        echo \"perf produced no successful rows: \$perf_file\" >&2
        exit 5
      }
    done
    test \"\$perf_file_count\" -eq 2 || {
      echo \"expected 2 perf files, found \$perf_file_count in $REMOTE_LOG_DIR\" >&2
      exit 6
    }
    metrics_file_count=0
    for metrics_file in \$(find '$REMOTE_LOG_DIR' -maxdepth 1 -type f -name '*_metrics.csv' -newer \"\$run_marker\" -print); do
      metrics_file_count=\$((metrics_file_count + 1))
      eval_count=\$(grep -c 'clean_test_epoch' \"\$metrics_file\" || true)
      test \"\$eval_count\" -eq '$TOTAL_EPOCHS' || {
        echo \"expected $TOTAL_EPOCHS clean evaluations, found \$eval_count: \$metrics_file\" >&2
        exit 7
      }
    done
    test \"\$metrics_file_count\" -eq 2 || {
      echo \"expected 2 metrics files, found \$metrics_file_count in $REMOTE_LOG_DIR\" >&2
      exit 8
    }
    rm -f \"\$run_marker\"
    echo \"validated \$perf_file_count perf files and \$metrics_file_count metrics files\"
  "
}

collect_logs() {
  mkdir -p "$LOCAL_LOG_DIR"
  local -a rsync_command
  rsync_command=(
    rsync -az --partial --stats --human-readable
    -e "ssh -p ${SSH_PORT} -o StrictHostKeyChecking=accept-new -o ConnectTimeout=15"
    "${SSH_USER}@${JETSON_HOST}:${REMOTE_PROJECT_DIR}/${REMOTE_LOG_DIR}/"
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
