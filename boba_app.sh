#!/usr/bin/env bash

# Example launcher snippets for the trimmed Boba demo export.

# Quest primary display with live Quest controllers.
python interactive_playground_batched_view_orin.py \
  --case_name double_stretch_sloth -exp -eval --n_dup 0 \
  --input_source live_openxr_controller \
  --quest_display_mode primary \
  --interactive_window_mode hidden

# Quest panel mirror while keeping the local window visible.
python interactive_playground_batched_view_orin.py \
  --case_name double_stretch_sloth -exp -eval --n_dup 0 \
  --input_source live_openxr_controller \
  --quest_display_mode panel \
  --interactive_window_mode visible

# Desktop-only replay baseline.
python interactive_playground_batched_view_orin.py \
  --case_name double_stretch_sloth -exp -eval --nzzzz_dup 0

python interactive_playground_batched_view_orin.py \
  --case_name double_stretch_sloth -exp --n_dup 0 \
  --input_source live_openxr_controller \
  --quest_display_mode primary \
  --interactive_window_mode hidden