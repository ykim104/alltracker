#!/usr/bin/env bash
# Ensure ffmpeg and ffprobe are on PATH (Gantry/Beaker images may lack them).
set -euo pipefail

if command -v ffprobe >/dev/null 2>&1 && command -v ffmpeg >/dev/null 2>&1; then
  echo "[install_ffmpeg] Using $(command -v ffprobe) and $(command -v ffmpeg)"
  exit 0
fi

install_via_apt() {
  command -v apt-get >/dev/null 2>&1 || return 1
  export DEBIAN_FRONTEND=noninteractive
  apt-get update -qq
  apt-get install -y -qq --no-install-recommends ffmpeg
  command -v ffprobe >/dev/null 2>&1
}

install_via_yum() {
  command -v dnf >/dev/null 2>&1 || command -v yum >/dev/null 2>&1 || return 1
  local pkg_mgr
  pkg_mgr="$(command -v dnf || command -v yum)"
  "${pkg_mgr}" install -y ffmpeg ffmpeg-devel || "${pkg_mgr}" install -y ffmpeg
  command -v ffprobe >/dev/null 2>&1
}

install_static() {
  local prefix="${HOME}/.local/ffmpeg-static"
  local bindir="${prefix}/bin"
  if [[ -x "${bindir}/ffprobe" && -x "${bindir}/ffmpeg" ]]; then
    export PATH="${bindir}:${PATH}"
    return 0
  fi
  local url="https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-linux64-gpl.tar.xz"
  local work
  work="$(mktemp -d)"
  trap 'rm -rf "${work}"' RETURN
  echo "[install_ffmpeg] Downloading static ffmpeg from BtbN/FFmpeg-Builds..."
  curl -fsSL "${url}" -o "${work}/ffmpeg.tar.xz"
  tar -xJf "${work}/ffmpeg.tar.xz" -C "${work}"
  local extracted
  extracted="$(find "${work}" -maxdepth 1 -type d -name 'ffmpeg-*' | head -n 1)"
  [[ -n "${extracted}" ]] || return 1
  mkdir -p "${bindir}"
  install -m 0755 "${extracted}/bin/ffmpeg" "${extracted}/bin/ffprobe" "${bindir}/"
  export PATH="${bindir}:${PATH}"
  command -v ffprobe >/dev/null 2>&1
}

if install_via_apt || install_via_yum || install_static; then
  echo "[install_ffmpeg] Installed ffprobe=$(command -v ffprobe) ffmpeg=$(command -v ffmpeg)"
  exit 0
fi

echo "[install_ffmpeg] ERROR: could not install ffmpeg/ffprobe (apt, yum, or static fallback failed)" >&2
exit 1
