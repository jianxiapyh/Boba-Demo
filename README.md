# Boba Demo

This repo contains the shipped Quest immersive Boba demo path: live OpenXR controller input on Linux, immersive desktop compositing, and Quest immersive display.

The only public packaged demo cases in this branch are:
- `sloth`
- `rope`

Each case resolves entirely from `assets/<case>/`, and the shipped room assets live under `assets/scenes/ILLIXR_lab/`.

## Runtime assets

The immersive demo no longer depends on runtime assets from:
- `data/`
- `experiments/`
- `experiments_optimization/`
- `gaussian_output/`

The packaged runtime bundles live under:
- `assets/sloth/`
- `assets/rope/`

For the shipped runtime Gaussian PLYs:
- `assets/sloth/sloth.ply` is copied from `Boba/gaussian_output/double_stretch_sloth/.../iteration_10000/point_cloud.ply`
- `assets/rope/rope.ply` is copied from `Boba/gaussian_output/single_lift_rope/.../iteration_10000/point_cloud.ply`

No alignment, annotation, filler-training, or candidate-generation workflow is kept in this branch anymore. This repo is now a runtime-only demo package.

## Environment

The intended environment is the existing `phystwin` Conda environment used for the Boba demo. ALVR/SteamVR/OpenXR runtime setup is assumed to already be installed on the machine.

This demo still resolves `gsplat` from the sibling `Boba_OpenSource` checkout instead of a stock pip wheel. The default expected source tree is:

```text
../Boba_OpenSource/gaussian_splatting/submodules/gsplat/
```

If your `Boba_OpenSource` checkout lives elsewhere, set `BOBA_GSPLAT_SOURCE_ROOT` to that vendored `gsplat` source root before launching.

## Main run commands

Default Quest immersive run:

```bash
conda run -n phystwin env PYTHONNOUSERSITE=1 python boba_quest_immersive.py \
  --case_name sloth \
  --n_dup 0 \
  --interactive_window_mode hidden
```

Alternate packaged case:

```bash
conda run -n phystwin env PYTHONNOUSERSITE=1 python boba_quest_immersive.py \
  --case_name rope \
  --n_dup 0 \
  --interactive_window_mode hidden
```

Quest immersive run with render profiling:

```bash
conda run -n phystwin env PYTHONNOUSERSITE=1 python boba_quest_immersive.py \
  --case_name sloth \
  --n_dup 0 \
  --interactive_window_mode hidden \
  --render_profile \
  --render_profile_every 30
```

The launcher is intentionally fixed to:
- `input_source=live_openxr_controller`
- `quest_display_mode=immersive`
- `scene_preset=ILLIXR_lab`
- `immersive_render_preset=balanced`
