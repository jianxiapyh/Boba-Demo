#!/usr/bin/env bash

set -euo pipefail

STEAMVR_ROOT="${HOME}/.local/share/Steam/steamapps/common/SteamVR"
export XR_RUNTIME_JSON="${STEAMVR_ROOT}/steamxr_linux64.json"
export LD_LIBRARY_PATH="${STEAMVR_ROOT}/bin/linux64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

MODE="${1:-live-primary}"

case "${MODE}" in
  live-primary)
    python interactive_playground_batched_view_orin.py \
      --case_name double_stretch_sloth -exp --n_dup 0 \
      --input_source live_openxr_controller \
      --quest_display_mode primary \
      --interactive_window_mode hidden
    ;;
  live-primary-eval)
    python interactive_playground_batched_view_orin.py \
      --case_name double_stretch_sloth -exp -eval --n_dup 0 \
      --input_source live_openxr_controller \
      --quest_display_mode primary \
      --interactive_window_mode hidden
    ;;
  panel)
    python interactive_playground_batched_view_orin.py \
      --case_name double_stretch_sloth -exp --n_dup 0 \
      --input_source live_openxr_controller \
      --quest_display_mode panel \
      --interactive_window_mode visible
    ;;
  replay)
    python interactive_playground_batched_view_orin.py \
      --case_name double_stretch_sloth -exp --n_dup 0
    ;;
  *)
    echo "Usage: $0 [live-primary|live-primary-eval|panel|replay]" >&2
    exit 1
    ;;
esac
