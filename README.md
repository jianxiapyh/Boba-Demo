# Boba Demo

This repo contains the code for the Quest-enabled Boba demo path: live OpenXR controller input on Linux, desktop compositing, and Quest panel display.

This export is intentionally trimmed. It does not include:
- `data/`
- `experiments/`
- `experiments_optimization/`
- `gaussian_output/`
- `gaussian_splatting/`
- generated outputs, logs, or probe binaries

Those large assets are expected to already exist alongside this repo on the target machine.

## Expected layout

The following directories should already be present in the repo root before running:

```text
data/
experiments/
experiments_optimization/
gaussian_output/
gaussian_splatting/
```

This repo provides the live demo code around them:
- `interactive_playground_batched_view_orin.py`
- `qqtt/`
- `configs/`
- `linux_pose_probe/`

`gaussian_output_dynamic/` is an output folder created by the app at runtime. It is not a required input dependency.

## Environment

The intended environment is the existing `phystwin` Conda environment used for the Boba demo. ALVR/SteamVR/OpenXR runtime setup is assumed to already be installed on the machine.

If you need the RTX 5090 environment helper used during development, see:

```bash
env_install/5090_env_install.sh
```

## Main run commands

By default, runs write a compact `gaussian_output_dynamic/<case_name>/performance_summary.txt`.
Adding `-eval` enables capture artifacts under `gaussian_output_dynamic/<case_name>/` and keeps the verbose frame-compositing breakdown in the saved summary.

`--quest_display_mode primary` is the heaviest presentation path because it renders at a higher Quest-target compositing resolution than the desktop/panel modes.

Desktop + Quest primary display:

```bash
python interactive_playground_batched_view_orin.py \
  --case_name double_stretch_sloth -exp --n_dup 0 \
  --input_source live_openxr_controller \
  --quest_display_mode primary \
  --interactive_window_mode hidden
```

Desktop-only replay baseline:

```bash
python interactive_playground_batched_view_orin.py \
  --case_name double_stretch_sloth -exp --n_dup 0
```

Quest panel mirror:

```bash
python interactive_playground_batched_view_orin.py \
  --case_name double_stretch_sloth -exp --n_dup 0 \
  --input_source live_openxr_controller \
  --quest_display_mode panel
```

Quest primary capture / profiling:

```bash
python interactive_playground_batched_view_orin.py \
  --case_name double_stretch_sloth -exp -eval --n_dup 0 \
  --input_source live_openxr_controller \
  --quest_display_mode primary \
  --interactive_window_mode hidden
```

## OpenXR helper programs

`linux_pose_probe/` contains the standalone OpenXR/Linux helper programs used for headset, hand, controller, and Quest frame-panel bring-up. Build scripts are included; binaries are intentionally not committed.
