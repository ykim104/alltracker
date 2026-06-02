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
