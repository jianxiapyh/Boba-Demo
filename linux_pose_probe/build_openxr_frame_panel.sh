#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

if [[ -n "${CXX:-}" ]]; then
  compiler="${CXX}"
elif command -v g++ >/dev/null 2>&1; then
  compiler="g++"
elif command -v c++ >/dev/null 2>&1; then
  compiler="c++"
else
  echo "No C++ compiler found; tried CXX, g++, and c++." >&2
  exit 127
fi

read -r -a openxr_flags <<<"$(pkg-config --cflags --libs glfw3 gl x11 openxr)"
extra_flags=()
if pkg-config --exists jsoncpp; then
  read -r -a jsoncpp_flags <<<"$(pkg-config --cflags --libs jsoncpp)"
  extra_flags+=("${jsoncpp_flags[@]}")
fi

"${compiler}" -std=c++17 -O2 -Wall -Wextra -pedantic \
  openxr_frame_panel.cpp \
  -o openxr_frame_panel \
  "${openxr_flags[@]}" \
  "${extra_flags[@]}"
