#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPO_ROOT}"

RUNTIME_ENV="phystwin-cu132"
CONDA_BIN="${CONDA_EXE:-}"
if [[ -z "${CONDA_BIN}" ]]; then
  CONDA_BIN="$(command -v conda || true)"
fi

if [[ -z "${CONDA_BIN}" || ! -x "${CONDA_BIN}" ]]; then
  echo "conda was not found on PATH." >&2
  echo "Initialize Conda, then rerun this launcher. The required environment is '${RUNTIME_ENV}'." >&2
  exit 127
fi

if ! "${CONDA_BIN}" env list | awk -v wanted="${RUNTIME_ENV}" '$1 == wanted { found = 1 } END { exit !found }'; then
  echo "The required Conda environment '${RUNTIME_ENV}' was not found." >&2
  echo "Create or restore that CUDA/rendering environment, then rerun this launcher." >&2
  echo "The Boba runtime and event assets are already included in this checkout." >&2
  exit 1
fi

# Always launch a clean child in the supported demo environment.  In particular,
# inherited CONDA_* values can make a nested `conda run` execute the right
# Python while incorrectly advertising the parent environment to subprocesses.
exec env \
  -u CONDA_PREFIX \
  -u CONDA_DEFAULT_ENV \
  -u CONDA_SHLVL \
  -u CUDA_HOME \
  -u LD_LIBRARY_PATH \
  "${CONDA_BIN}" run --no-capture-output -n "${RUNTIME_ENV}" \
  env PYTHONNOUSERSITE=1 \
  python boba_quest_immersive.py \
  --case_name rope_game \
  --n_dup 0 \
  --interactive_window_mode visible \
  --immersive_start_posture standing \
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
  --immersive_controller_max_motion_interval_m 0.05 \
  --immersive_viewer_upload_mode pbo \
  --immersive_viewer_upload_thread auto \
  "$@"
