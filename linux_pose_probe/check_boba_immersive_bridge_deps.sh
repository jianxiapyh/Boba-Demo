#!/usr/bin/env bash
set -euo pipefail

missing_tools=()
missing_modules=()

if ! command -v pkg-config >/dev/null 2>&1; then
  missing_tools+=("pkg-config")
fi

if ! command -v g++ >/dev/null 2>&1; then
  missing_tools+=("g++")
fi

if command -v pkg-config >/dev/null 2>&1; then
  for module in glfw3 gl x11 openxr; do
    if ! pkg-config --exists "${module}"; then
      missing_modules+=("${module}")
    fi
  done
fi

if [[ ${#missing_tools[@]} -eq 0 && ${#missing_modules[@]} -eq 0 ]]; then
  exit 0
fi

echo "Missing system packages required to build the Boba immersive OpenXR bridge on Ubuntu 22.04." >&2
if [[ ${#missing_tools[@]} -gt 0 ]]; then
  echo "Missing tools: ${missing_tools[*]}" >&2
fi
if [[ ${#missing_modules[@]} -gt 0 ]]; then
  echo "Missing pkg-config modules: ${missing_modules[*]}" >&2
fi
echo "Install the required packages with:" >&2
echo "  sudo apt install pkg-config libglfw3-dev libgl1-mesa-dev libx11-dev libopenxr-dev" >&2
echo "Optional only: sudo apt install libjsoncpp-dev" >&2
exit 1
