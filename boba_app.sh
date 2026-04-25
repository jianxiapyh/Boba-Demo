#!/usr/bin/env bash

# Canonical launcher snippets for the shipped Quest immersive demo.
# Direct `python boba_quest_immersive.py ...` launches now self-heal the
# required conda/CUDA runtime library path, but this wrapper keeps the
# same exports explicit for reproducibility on RTX6000 Blackwell.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PATH="${CUDA_HOME}/bin:${PATH}"
if [[ -n "${CONDA_PREFIX:-}" ]]; then
  export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${CUDA_HOME}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
else
  export LD_LIBRARY_PATH="${CUDA_HOME}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi
export BOBA_GSPLAT_SOURCE_ROOT="${BOBA_GSPLAT_SOURCE_ROOT:-${SCRIPT_DIR}/../Boba/gaussian_splatting/submodules/gsplat}"

# Default Quest immersive run: live OpenXR controllers, immersive display,
# ILLIXR_lab scene, balanced preset. Public demo cases are sloth, rope,
# hq_rope, and rope_game.
python boba_quest_immersive.py \
  --case_name sloth \
  --n_dup 0 \
  --interactive_window_mode hidden

python boba_quest_immersive.py \
  --case_name rope \
  --n_dup 0 \
  --interactive_window_mode hidden

python boba_quest_immersive.py \
  --case_name hq_rope \
  --n_dup 0 \
  --interactive_window_mode hidden

python boba_quest_immersive.py \
  --case_name rope_game \
  --n_dup 0 \
  --interactive_window_mode hidden

python boba_quest_immersive.py \
  --case_name sloth \
  --n_dup 0 \
  --interactive_window_mode visible

# Same run with render profiling enabled.
python boba_quest_immersive.py \
  --case_name sloth \
  --n_dup 0 \
  --interactive_window_mode hidden \
  --profile \
  --profile_freq 30


python boba_quest_immersive.py \
  --case_name rope_game \
  --n_dup 0 \
  --interactive_window_mode visible \
  --immersive_static_scene_backend native_gl \
  --immersive_static_scene_overlap on \
  --immersive_present_pipeline off \
  --immersive_static_scene_reuse adaptive \
  --immersive_gaussian_render stereo_batched \
  --immersive_native_gl_texture_mode stable_mipmap \
  --immersive_native_gl_anisotropy 8 \
  --immersive_native_gl_msaa_samples 4 \
  --immersive_native_gl_depth_format depth32f \
  --immersive_eye_resolution 1408 \
  --immersive_controller_translation_scale 0.5 

  --profile \
  --profile_freq 30


python boba_quest_immersive.py \
  --case_name hq_rope_game \
  --n_dup 0 \
  --interactive_window_mode hidden \
  --immersive_static_scene_backend native_gl \
  --immersive_static_scene_overlap on \
  --immersive_present_pipeline off \
  --immersive_static_scene_reuse adaptive \
  --immersive_gaussian_render stereo_batched \
  --immersive_native_gl_texture_mode stable_mipmap \
  --immersive_native_gl_anisotropy 8 \
  --immersive_native_gl_msaa_samples 4 \
  --immersive_native_gl_depth_format depth32f \
  --immersive_eye_resolution 1344 \
  --immersive_controller_translation_scale 0.75 \
  --immersive_viewer_upload_mode pbo \
  --immersive_viewer_upload_thread auto \
  --immersive_viewer_upload_late_wait_us 0
