#!/usr/bin/env bash
# Launch distributed AllTracker inference on Beaker Gantry (preemptible-friendly).
#
# Default: 10 nodes × 1 GPU (multi-replica). Each node runs one shard; preemption only
# loses progress on the affected node. Re-submit the same command to resume.
#
#   ./scripts/beaker/launch_inference_gantry.sh \
#     --user-name yejink \
#     --dataset-root /weka/oe-training/yejink/data/robotwin2.0/robotwin2.0
#
# Optional: one node with many GPUs (all shards on one machine):
#   ./scripts/beaker/launch_inference_gantry.sh ... --mode multi-gpu --replicas 1 --gpus 10
#
# Requires: pip install beaker-gantry && gantry config

set -euo pipefail

USER_NAME=""
DATASET_ROOT=""
NUM_GPUS=1
NUM_REPLICAS=10
INFERENCE_MODE="multi-replica"
WORKSPACE="ai2/vida"
BUDGET="ai2/robotics"
PRIORITY="normal"
WEKA_BUCKET="oe-training-default"
WEKA_MOUNT="oe-training"
IMAGE_SIZE=640
BATCH_SIZE=4
MAX_BATCH_FRAMES=1200
MIN_MOTION_PX=6
EXTRA_ARGS=()
GANTRY_EXTRA=(--allow-dirty)

usage() {
  cat <<'EOF'
Usage:
  launch_inference_gantry.sh --user-name NAME --dataset-root PATH [options]

Required:
  --user-name NAME       Weka user (for default dataset path hints)
  --dataset-root PATH    Dataset root on Weka (must contain observation.images.*)

Options:
  --nodes N              Alias for --replicas (default: 10 nodes)
  --replicas N           Gantry replicas / nodes (default: 10)
  --gpus N               GPUs per node (default: 1). Use 10 with --mode multi-gpu for single-node.
  --mode MODE            multi-replica | multi-gpu (default: multi-replica)
  --image-size N         Max side (640 keeps RoboTwin native resolution)
  --batch-size N         Videos per forward pass (default: 4)
  --max-batch-frames N   batch_size * padded_length cap (default: 1200)
  --min-motion-px N      Point motion threshold (default: 6)
  --workspace, --budget, --priority
  --weka-bucket, --weka-mount
  Extra tokens -> ALLTRACKER_EXTRA_ARGS (e.g. --sort-by-length)
EOF
  exit 1
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --user-name) USER_NAME="$2"; shift 2 ;;
    --dataset-root) DATASET_ROOT="$2"; shift 2 ;;
    --gpus) NUM_GPUS="$2"; shift 2 ;;
    --replicas) NUM_REPLICAS="$2"; shift 2 ;;
    --nodes) NUM_REPLICAS="$2"; shift 2 ;;
    --mode) INFERENCE_MODE="$2"; shift 2 ;;
    --image-size) IMAGE_SIZE="$2"; shift 2 ;;
    --batch-size) BATCH_SIZE="$2"; shift 2 ;;
    --max-batch-frames) MAX_BATCH_FRAMES="$2"; shift 2 ;;
    --min-motion-px) MIN_MOTION_PX="$2"; shift 2 ;;
    --workspace) WORKSPACE="$2"; shift 2 ;;
    --budget) BUDGET="$2"; shift 2 ;;
    --priority) PRIORITY="$2"; shift 2 ;;
    --weka-bucket) WEKA_BUCKET="$2"; shift 2 ;;
    --weka-mount) WEKA_MOUNT="$2"; shift 2 ;;
    --allow-dirty) shift ;;
    -h|--help) usage ;;
    *) EXTRA_ARGS+=("$1"); shift ;;
  esac
done

[[ -n "${USER_NAME}" ]] || usage
if [[ -z "${DATASET_ROOT}" ]]; then
  DATASET_ROOT="/weka/${WEKA_MOUNT}/${USER_NAME}/data/robotwin2.0/robotwin2.0"
fi

if [[ "${INFERENCE_MODE}" == "multi-replica" ]]; then
  if [[ "${NUM_REPLICAS}" -le 1 ]]; then
    echo "Error: multi-replica mode needs --nodes (or --replicas) > 1" >&2
    exit 1
  fi
  if [[ "${NUM_GPUS}" -ne 1 ]]; then
    echo "Warning: multi-replica mode expects 1 GPU per node; forcing --gpus 1 (was ${NUM_GPUS})" >&2
    NUM_GPUS=1
  fi
fi

resolve_gantry() {
  if [[ -n "${GANTRY:-}" && -x "${GANTRY}" ]]; then
    echo "${GANTRY}"
    return 0
  fi
  if command -v gantry >/dev/null 2>&1; then
    command -v gantry
    return 0
  fi
  local repo_root="$1"
  if [[ -x "${repo_root}/.venv/bin/gantry" ]]; then
    echo "${repo_root}/.venv/bin/gantry"
    return 0
  fi
  if python -m gantry --help >/dev/null 2>&1; then
    echo "python -m gantry"
    return 0
  fi
  return 1
}

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ALLTRACKER_EXTRA="${EXTRA_ARGS[*]:-}"

JOB_NAME="alltracker-infer-$(basename "${DATASET_ROOT}")-${NUM_REPLICAS}x${NUM_GPUS}gpu"

GANTRY_ARGS=(
  run
  --yes
  "${GANTRY_EXTRA[@]}"
  --workspace "${WORKSPACE}"
  --budget "${BUDGET}"
  --priority "${PRIORITY}"
  --gpus "${NUM_GPUS}"
  --replicas "${NUM_REPLICAS}"
  --shared-memory 32GiB
  --memory 128GiB
  --weka "${WEKA_BUCKET}:/weka/${WEKA_MOUNT}"
  --cluster "ai2/saturn"
  --cluster "ai2/jupiter"
  --env "USER_NAME=${USER_NAME}"
  --env "DATASET_ROOT=${DATASET_ROOT}"
  --env "NUM_GPUS=${NUM_GPUS}"
  --env "INFERENCE_MODE=${INFERENCE_MODE}"
  --env "IMAGE_SIZE=${IMAGE_SIZE}"
  --env "BATCH_SIZE=${BATCH_SIZE}"
  --env "MAX_BATCH_FRAMES=${MAX_BATCH_FRAMES}"
  --env "MIN_MOTION_PX=${MIN_MOTION_PX}"
  --env "ALLTRACKER_EXTRA_ARGS=${ALLTRACKER_EXTRA}"
  --env "ALLTRACKER_LOG_DIR=/weka/${WEKA_MOUNT}/${USER_NAME}/logs/alltracker"
  --python-manager uv
  --uv-torch-backend cu128
  --default-python-version 3.12
  --install " (command -v apt-get >/dev/null && apt-get update -qq && apt-get install -y -qq ffmpeg python3.12-dev build-essential) || true; unset CUDA_HOME; uv pip install -r requirements.txt --torch-backend cu128"
  --name "${JOB_NAME}"
  --description "AllTracker inference ${DATASET_ROOT} (${USER_NAME}) ${NUM_REPLICAS}x${NUM_GPUS}gpu mode=${INFERENCE_MODE}"
)

GANTRY_ARGS+=(--propagate-preemption)
if [[ "${NUM_REPLICAS}" -gt 1 ]]; then
  GANTRY_ARGS+=(--leader-selection --host-networking --propagate-failure)
fi

cd "${REPO_ROOT}"
chmod +x scripts/beaker/run_inference.sh scripts/beaker/setup_job_env.sh

if ! GANTRY_CMD="$(resolve_gantry "${REPO_ROOT}")"; then
  echo "Error: gantry not found. Install: pip install beaker-gantry && gantry config" >&2
  exit 127
fi

echo "[launch] dataset=${DATASET_ROOT}"
echo "[launch] replicas=${NUM_REPLICAS} gpus=${NUM_GPUS} mode=${INFERENCE_MODE}"
echo "[launch] image_size=${IMAGE_SIZE} batch=${BATCH_SIZE} max_batch_frames=${MAX_BATCH_FRAMES}"
echo ">>> ${GANTRY_CMD} ${GANTRY_ARGS[*]} -- bash scripts/beaker/run_inference.sh"
# shellcheck disable=SC2086
exec ${GANTRY_CMD} "${GANTRY_ARGS[@]}" -- bash scripts/beaker/run_inference.sh
