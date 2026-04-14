#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
output_path="${script_dir}/boba_immersive_bridge"
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

"${compiler}" \
  -std=c++17 \
  -O2 \
  -Wall \
  -Wextra \
  -pedantic \
  -DBOBA_IMMERSIVE_BRIDGE \
  "${script_dir}/openxr_frame_panel.cpp" \
  -o "${output_path}" \
  "${openxr_flags[@]}" \
  "${extra_flags[@]}"

printf 'Built %s\n' "${output_path}"
