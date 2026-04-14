# Boba Demo

This repo contains the code for the shipped Quest-enabled Boba demo path: live OpenXR controller input on Linux, immersive desktop compositing, and Quest immersive display.

This export is intentionally trimmed. It does not include:
- `data/`
- `experiments/`
- `experiments_optimization/`
- `gaussian_splatting/`
- generated outputs, logs, or probe binaries

Those large assets are expected to already exist alongside this repo on the target machine.

## Expected layout

The following directories should already be present in the repo root before running:

```text
data/
experiments/
experiments_optimization/
gaussian_splatting/
```

This repo provides the live demo code around them:
- `boba_quest_immersive.py`
- `qqtt/`
- `configs/`
- `linux_pose_probe/`

## Environment

The intended environment is the existing `phystwin` Conda environment used for the Boba demo. ALVR/SteamVR/OpenXR runtime setup is assumed to already be installed on the machine.

This demo resolves `gsplat` from the sibling `Boba_OpenSource` checkout instead of a stock pip wheel. The default expected source tree is:

```text
../Boba_OpenSource/gaussian_splatting/submodules/gsplat/
```

If your `Boba_OpenSource` checkout lives elsewhere, set `BOBA_GSPLAT_SOURCE_ROOT` to that vendored `gsplat` source root before launching.

If you need the RTX 5090 environment helper used during development, see:

```bash
env_install/5090_env_install.sh
```

## Main run commands

Default Quest immersive run:

```bash
conda run -n phystwin env PYTHONNOUSERSITE=1 python boba_quest_immersive.py \
  --case_name double_stretch_sloth \
  --n_dup 0 \
  --interactive_window_mode hidden
```

Quest immersive run with render profiling:

```bash
conda run -n phystwin env PYTHONNOUSERSITE=1 python boba_quest_immersive.py \
  --case_name double_stretch_sloth \
  --n_dup 0 \
  --interactive_window_mode hidden \
  --render_profile \
  --render_profile_every 30
```

The launcher is intentionally fixed to:
- `input_source=live_openxr_controller`
- `quest_display_mode=immersive`
- `scene_preset=simple_lab`
- `immersive_render_preset=balanced`

## OpenXR helper programs

`linux_pose_probe/` contains the standalone OpenXR/Linux helper programs used for headset, hand, controller, and Quest frame-panel bring-up. Build scripts are included; binaries are intentionally not committed.
