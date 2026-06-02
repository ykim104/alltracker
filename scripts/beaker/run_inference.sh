#!/usr/bin/env bash
# Run AllTracker point-track inference inside a Beaker/Gantry container.
#
# Modes (set by launch_inference_gantry.sh):
#   multi-gpu (default): one node, NUM_GPUS parallel workers (shard 0..N-1, cuda:0..N-1)
#   multi-replica:       one process per replica; shard from BEAKER_REPLICA_RANK/COUNT
#
# Resume after preemption: re-submit the same Gantry command. Workers skip outputs that
# pass ffprobe validation; partial .tmp files are ignored.
set -euo pipefail

beaker_on_err() {
  echo "[alltracker-beaker] ERROR: exit $? at ${BASH_SOURCE[1]}:${BASH_LINENO[0]}: ${BASH_COMMAND}" >&2
}
trap beaker_on_err ERR

DATASET_ROOT="${DATASET_ROOT:?Set DATASET_ROOT (Weka path to robotwin2.0 root)}"
NUM_GPUS="${NUM_GPUS:-1}"
INFERENCE_MODE="${INFERENCE_MODE:-multi-gpu}"

if [[ -n "${CODE_DIR:-}" ]]; then
  cd "${CODE_DIR}"
else
  echo "[alltracker-beaker] Using gantry/git checkout: $(pwd)"
fi

_BEAKER_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=setup_job_env.sh
source "${_BEAKER_SCRIPT_DIR}/setup_job_env.sh"

if [[ ! -d "${DATASET_ROOT}" ]]; then
  echo "[alltracker-beaker] ERROR: DATASET_ROOT not found: ${DATASET_ROOT}" >&2
  exit 1
fi

# Defaults tuned for RoboTwin 640x480 on ~48GB GPUs.
IMAGE_SIZE="${IMAGE_SIZE:-640}"
BATCH_SIZE="${BATCH_SIZE:-4}"
MAX_BATCH_FRAMES="${MAX_BATCH_FRAMES:-1200}"
MIN_MOTION_PX="${MIN_MOTION_PX:-6}"
DECODE_WORKERS="${DECODE_WORKERS:-3}"
ENCODE_WORKERS="${ENCODE_WORKERS:-2}"
INFERENCE_ITERS="${INFERENCE_ITERS:-4}"
WINDOW_LEN="${WINDOW_LEN:-16}"
RATE="${RATE:-8}"
VIDEO_BACKEND="${VIDEO_BACKEND:-ffmpeg}"
EXTRA_ARGS=()
if [[ -n "${ALLTRACKER_EXTRA_ARGS:-}" ]]; then
  # shellcheck disable=SC2206
  EXTRA_ARGS=(${ALLTRACKER_EXTRA_ARGS})
fi

common_args=(
  --dataset-root "${DATASET_ROOT}"
  --image-size "${IMAGE_SIZE}"
  --batch-size "${BATCH_SIZE}"
  --max-batch-frames "${MAX_BATCH_FRAMES}"
  --min-motion-px "${MIN_MOTION_PX}"
  --decode-workers "${DECODE_WORKERS}"
  --encode-workers "${ENCODE_WORKERS}"
  --inference-iters "${INFERENCE_ITERS}"
  --window-len "${WINDOW_LEN}"
  --rate "${RATE}"
  --video-backend "${VIDEO_BACKEND}"
)

if [[ "${ALLTRACKER_TINY:-0}" == "1" ]]; then
  common_args+=(--tiny)
fi

echo "[alltracker-beaker] dataset=${DATASET_ROOT} mode=${INFERENCE_MODE} gpus=${NUM_GPUS}"
echo "[alltracker-beaker] image_size=${IMAGE_SIZE} batch=${BATCH_SIZE} max_batch_frames=${MAX_BATCH_FRAMES}"

run_worker() {
  local shard_id="$1"
  local num_shards="$2"
  local device="$3"
  local log_file="${ALLTRACKER_LOG_DIR:-/tmp/alltracker_logs}/shard_${shard_id}.log"
  mkdir -p "$(dirname "${log_file}")"
  echo "[alltracker-beaker] shard ${shard_id}/${num_shards} -> ${device} (log: ${log_file})"
  "${PYTHON}" inference_dataset_gantry.py \
    --shard-id "${shard_id}" \
    --num-shards "${num_shards}" \
    --device "${device}" \
    "${common_args[@]}" \
    "${EXTRA_ARGS[@]}" \
    2>&1 | tee "${log_file}"
}

mkdir -p "${ALLTRACKER_LOG_DIR:-/tmp/alltracker_logs}"

if [[ "${INFERENCE_MODE}" == "multi-replica" ]]; then
  replica_rank="${BEAKER_REPLICA_RANK:-${ALLTRACKER_SHARD_ID:-0}}"
  replica_count="${BEAKER_REPLICA_COUNT:-${ALLTRACKER_NUM_SHARDS:-1}}"
  run_worker "${replica_rank}" "${replica_count}" "cuda:0"
  exit 0
fi

if [[ "${NUM_GPUS}" -le 1 ]]; then
  run_worker 0 1 "cuda:0"
  exit 0
fi

pids=()
for gpu_id in $(seq 0 $((NUM_GPUS - 1))); do
  (
    run_worker "${gpu_id}" "${NUM_GPUS}" "cuda:${gpu_id}"
  ) &
  pids+=("$!")
done

fail=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    fail=1
  fi
done

if [[ "${fail}" -ne 0 ]]; then
  echo "[alltracker-beaker] One or more shards failed." >&2
  exit 1
fi

echo "[alltracker-beaker] All ${NUM_GPUS} shards finished."
