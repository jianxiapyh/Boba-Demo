#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPO_ROOT}"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda was not found on PATH." >&2
  echo "Initialize Conda, then rerun this launcher. The required environment is 'phystwin'." >&2
  exit 127
fi

if ! conda env list | awk '$1 == "phystwin" { found = 1 } END { exit !found }'; then
  echo "The required Conda environment 'phystwin' was not found." >&2
  echo "Install and successfully run Boba-Batched first, then rerun this launcher." >&2
  exit 1
fi

exec conda run --no-capture-output -n phystwin env PYTHONNOUSERSITE=1 \
  python boba_quest_immersive.py \
  --case_name rope_game \
  --n_dup 0 \
  --interactive_window_mode hidden \
  --immersive_static_scene_backend native_gl \
  --immersive_static_scene_overlap on \
  --immersive_present_pipeline off \
  --immersive_static_scene_reuse adaptive \
  --immersive_gaussian_render stereo_batched \
  --immersive_native_gl_texture_mode stable_mipmap \
  --immersive_native_gl_anisotropy 8 \
  --immersive_native_gl_mipmap_lod_bias 0.50 \
  --immersive_native_gl_msaa_samples 4 \
  --immersive_native_gl_depth_format depth32f \
  --immersive_eye_resolution 1344 \
  --immersive_controller_translation_scale 0.25 \
  --immersive_viewer_upload_mode pbo \
  --immersive_viewer_upload_thread auto
