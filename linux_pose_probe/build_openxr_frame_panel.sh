#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

g++ -std=c++17 -O2 -Wall -Wextra -pedantic \
  openxr_frame_panel.cpp \
  -o openxr_frame_panel \
  $(pkg-config --cflags --libs glfw3 gl x11 openxr)
