#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
output_path="${script_dir}/openxr_hand_controller_probe"
compiler="${CXX:-g++}"

read -r -a openxr_flags <<<"$(pkg-config --cflags --libs openxr)"

"${compiler}" \
  -std=c++17 \
  -O2 \
  -Wall \
  -Wextra \
  -pedantic \
  "${script_dir}/openxr_hand_controller_probe.cpp" \
  -o "${output_path}" \
  "${openxr_flags[@]}"

printf 'Built %s\n' "${output_path}"
