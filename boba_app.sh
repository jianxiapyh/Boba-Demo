#!/usr/bin/env bash

# Canonical launcher snippets for the shipped Quest immersive demo.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"
export BOBA_GSPLAT_SOURCE_ROOT="${BOBA_GSPLAT_SOURCE_ROOT:-${SCRIPT_DIR}/../Boba_OpenSource/gaussian_splatting/submodules/gsplat}"

# Default Quest immersive run: live OpenXR controllers, immersive display,
# simple_lab scene, balanced preset.
python boba_quest_immersive.py \
  --case_name double_stretch_sloth \
  --n_dup 0 \
  --interactive_window_mode hidden

  python boba_quest_immersive.py \
  --case_name double_stretch_sloth \
  --n_dup 0 \
  --interactive_window_mode visible

# Same run with render profiling enabled.
python boba_quest_immersive.py \
  --case_name double_stretch_sloth \
  --n_dup 0 \
  --interactive_window_mode hidden \
  --render_profile \
  --render_profile_every 30
