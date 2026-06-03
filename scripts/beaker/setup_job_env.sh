# shellcheck shell=bash
# Shared Beaker/Gantry environment for AllTracker inference jobs.

if [[ -n "${VIRTUAL_ENV:-}" && -x "${VIRTUAL_ENV}/bin/python" ]]; then
  export PATH="${VIRTUAL_ENV}/bin:${PATH}"
fi

export PYTHON="${PYTHON:-$(command -v python)}"
echo "[alltracker-beaker] PYTHON=${PYTHON} ($("${PYTHON}" --version 2>&1))"

export OPENCV_FFMPEG_CAPTURE_OPTIONS="${OPENCV_FFMPEG_CAPTURE_OPTIONS:-hwaccel;none}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

if [[ -n "${CUDA_HOME:-}" && ! -x "${CUDA_HOME}/bin/nvcc" ]]; then
  unset CUDA_HOME
fi

_BEAKER_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
bash "${_BEAKER_SCRIPT_DIR}/install_ffmpeg.sh"
if ! command -v ffprobe >/dev/null 2>&1 || ! command -v ffmpeg >/dev/null 2>&1; then
  echo "[alltracker-beaker] ERROR: ffmpeg/ffprobe not on PATH after install_ffmpeg.sh" >&2
  exit 1
fi
echo "[alltracker-beaker] ffmpeg=$(command -v ffmpeg) ffprobe=$(command -v ffprobe)"
