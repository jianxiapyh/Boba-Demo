# Boba Quest Rope Game

This repository runs the `rope_game` case: a Quest/OpenXR rope pick-and-place demo with live controller input in the `ILLIXR_lab` scene.

## Requirements

- Ubuntu 22.04 with an X11 desktop/OpenGL session. Hidden-window mode still requires a working display session.
- An NVIDIA CUDA workstation. The provided setup is tuned for RTX 6000 Blackwell with CUDA under `/usr/local/cuda`.
- A working Quest with ALVR/SteamVR configured as the OpenXR runtime.
- The `boba` Conda environment from the main Boba checkout.
- The vendored `gsplat` source from the main Boba checkout. Its default location is:

```text
../Boba/gaussian_splatting/submodules/gsplat/
```

If that checkout is elsewhere, set `BOBA_GSPLAT_SOURCE_ROOT` to its `gsplat` directory before setup or launch.

## Setup

Install the native OpenXR bridge dependencies on Ubuntu 22.04:

```bash
sudo apt install g++ pkg-config libglfw3-dev libgl1-mesa-dev libx11-dev libopenxr-dev
```

Activate the runtime environment and run the repository setup once:

```bash
conda activate boba
bash env_install/RTX6000_env_install.sh
```

Validate the assets used by `rope_game`:

```bash
python tools/fetch_demo_case_assets.py --case rope_game --check-only
```

The case uses:

- `assets/rope_game/manifest.json` for the course and tutorial.
- `assets/rope/` for the model, calibration, simulation data, metadata, parameters, and Gaussian PLY.
- `configs/real.yaml` for the rope simulation configuration.
- `assets/scenes/ILLIXR_lab/` for the immersive room.

## Run

Run from an activated `boba` environment:

```bash
python boba_quest_immersive.py \
  --case_name rope_game \
  --n_dup 0 \
  --interactive_window_mode hidden \
  --immersive_static_scene_backend native_gl \
  --immersive_static_scene_overlap on \
  --immersive_present_pipeline off \
  --immersive_static_scene_reuse off \
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
```

Native-GL overlap requires the present pipeline to remain off and currently forces static-scene reuse off. The launcher configures its CUDA and Conda runtime library paths automatically when started from the activated environment.

## Troubleshooting

If asset validation reports missing files or Git LFS pointers, hydrate only this case:

```bash
python tools/fetch_demo_case_assets.py --case rope_game
```

If the native bridge preflight fails, rerun the Ubuntu dependency command from the setup section.

If OpenXR cannot find the headset, confirm that SteamVR and ALVR are running and the Quest is connected. For a non-default SteamVR installation, set `XR_RUNTIME_JSON` to its OpenXR runtime JSON.
