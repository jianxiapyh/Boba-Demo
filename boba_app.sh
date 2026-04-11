#!/usr/bin/env bash

# Example launcher snippets for the trimmed Boba demo export.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"
export BOBA_GSPLAT_SOURCE_ROOT="${BOBA_GSPLAT_SOURCE_ROOT:-${SCRIPT_DIR}/../Boba_OpenSource/gaussian_splatting/submodules/gsplat}"

# Quest primary display with live Quest controllers.
python interactive_playground_batched_view_orin.py \
  --case_name double_stretch_sloth -exp -eval --n_dup 0 \
  --input_source live_openxr_controller \
  --quest_display_mode primary \
  --interactive_window_mode hidden

# Quest immersive stereo scene with the sloth centered on a lab table.
python interactive_playground_batched_view_orin.py \
  --case_name double_stretch_sloth -exp --n_dup 0 \
  --input_source live_openxr_controller \
  --quest_display_mode immersive \
  --scene_preset simple_lab \
  --interactive_window_mode hidden

# Quest immersive stereo scene with detailed render profiling enabled.
python interactive_playground_batched_view_orin.py \
  --case_name double_stretch_sloth -exp --n_dup 0 \
  --input_source live_openxr_controller \
  --quest_display_mode immersive \
  --scene_preset simple_lab \
  --interactive_window_mode hidden \
  --immersive_render_preset balanced \
  --render_profile \
  --render_profile_every 30

# Quest immersive experimental fast mode using center-view stereo reprojection.
python interactive_playground_batched_view_orin.py \
  --case_name double_stretch_sloth -exp --n_dup 0 \
  --input_source live_openxr_controller \
  --quest_display_mode immersive \
  --scene_preset simple_lab \
  --interactive_window_mode hidden \
  --immersive_render_preset performance

# Quest panel mirror while keeping the local window visible.
python interactive_playground_batched_view_orin.py \
  --case_name double_stretch_sloth -exp -eval --n_dup 0 \
  --input_source live_openxr_controller \
  --quest_display_mode panel \
  --interactive_window_mode visible

# Desktop-only replay baseline.
python interactive_playground_batched_view_orin.py \
  --case_name double_stretch_sloth -exp -eval --n_dup 0

python interactive_playground_batched_view_orin.py \
  --case_name double_stretch_sloth -exp --n_dup 0 \
  --input_source live_openxr_controller \
  --quest_display_mode primary \
  --interactive_window_mode hidden
