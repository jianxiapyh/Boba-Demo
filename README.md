# Boba Demo

This repo contains the shipped Quest immersive Boba demo path: live OpenXR controller input on Linux, immersive desktop compositing, and Quest immersive display.

The public packaged demo cases in this branch are:
- `sloth`
- `rope`
- `hq_rope`
- `rope_game`
- `hq_rope_game`
- `hybrid_rope_game`
- `hybrid_rope_game_1`

Compatibility alias:
- `hq_rope_0 -> hq_rope`

Each case has a manifest under `assets/<case>/`, and the shipped room assets live under `assets/scenes/ILLIXR_lab/`.

## Runtime assets

The immersive demo no longer depends on runtime assets from:
- `data/`
- `experiments/`
- `experiments_optimization/`
- `gaussian_output/`

The packaged runtime bundles live under:
- `assets/sloth/`
- `assets/rope/`
- `assets/hq_rope/`
- `assets/rope_game/`
- `assets/hq_rope_game/`
- `assets/hybrid_rope_game/`
- `assets/hybrid_rope_game_1/`

For the shipped runtime Gaussian PLYs:
- `assets/sloth/sloth.ply` is copied from `Boba/gaussian_output/double_stretch_sloth/.../iteration_10000/point_cloud.ply`
- `assets/rope/rope.ply` is copied from `Boba/gaussian_output/single_lift_rope/.../iteration_10000/point_cloud.ply`
- `assets/hq_rope/hq_rope.ply` is rebaked from `feng_rope/data/different_types/feng_rope_v8_0000/shape/object.ply`
- `assets/hq_rope_game/phystwin_rope.ply` is extracted from `shashuo0104/gs-scans` at `rope/rope.ply`.

`hq_rope_game` uses the same game logic, tutorial, and course as `rope_game`, but its object assets come from the original PhysTwin rope release (`shashuo0104/phystwin-rope`, `1495` object spring-mass nodes), not the retrained `assets/hq_rope` package. Target zone sizing is resolved from the original rope span at startup.

`hybrid_rope_game` uses the stable `assets/rope` spring-mass model and rope-game behavior, but renders the higher-quality PhysTwin Gaussian from `assets/hq_rope_game/phystwin_rope.ply` after a startup principal-axis visual retarget onto the rope simulation rest shape.

`hybrid_rope_game_1` uses the same hybrid rope assets and runtime behavior with a table-first, front-sofa-finish course.

The packaged runtime no longer requires `multi_ctrls.pkl`; controller traces come from `final_data.pkl` for every shipped case.

No alignment, annotation, filler-training, or candidate-generation workflow is kept in this branch anymore. This repo is now a runtime-only demo package.

## Environment

The intended environment on RTX6000 Blackwell is the existing `boba` Conda environment from the main `Boba` checkout. ALVR/SteamVR/OpenXR runtime setup is assumed to already be installed on the machine.

Run the RTX6000 environment preflight/install from an activated `boba` shell:

```bash
conda activate boba
bash env_install/RTX6000_env_install.sh
```

The RTX6000 installer installs/uses `git-lfs`, hydrates manifest-referenced demo assets that are still checked out as LFS pointers, rebuilds the local CUDA extensions with `TORCH_CUDA_ARCH_LIST=12.0`, and rebuilds `pycuda` with CUDA/OpenGL interop enabled.

If you only need to recheck or hydrate packaged assets:

```bash
python tools/fetch_demo_case_assets.py --all
python tools/fetch_demo_case_assets.py --case sloth
```

This demo resolves `gsplat` from the sibling `Boba` checkout instead of a stock pip wheel. The default expected source tree is:

```text
../Boba/gaussian_splatting/submodules/gsplat/
```

If your `Boba` checkout lives elsewhere, set `BOBA_GSPLAT_SOURCE_ROOT` to that vendored `gsplat` source root before launching.

The launcher expects:
- `PYTHONNOUSERSITE=1`
- `CUDA_HOME=/usr/local/cuda`
- `LD_LIBRARY_PATH` beginning with `$CONDA_PREFIX/lib:$CUDA_HOME/lib64`

Direct `python boba_quest_immersive.py ...` launches self-reexec once with that runtime contract when started from an activated `boba` environment.

## Main run commands

Canonical RTX6000 Quest immersive run:

```bash
conda activate boba
python boba_quest_immersive.py \
  --case_name rope_game \
  --n_dup 0 \
  --interactive_window_mode hidden
```

Alternate packaged cases:

```bash
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
  --case_name hq_rope_game \
  --n_dup 0 \
  --interactive_window_mode hidden

python boba_quest_immersive.py \
  --case_name hybrid_rope_game \
  --n_dup 0 \
  --interactive_window_mode hidden

python boba_quest_immersive.py \
  --case_name hybrid_rope_game_1 \
  --n_dup 0 \
  --interactive_window_mode hidden
```

Quest immersive run with render profiling:

```bash
python boba_quest_immersive.py \
  --case_name sloth \
  --n_dup 0 \
  --interactive_window_mode hidden \
  --profile \
  --profile_freq 30
```

If the native bridge dependency preflight reports missing packages on Ubuntu 22.04:

```bash
sudo apt install pkg-config libglfw3-dev libgl1-mesa-dev libx11-dev libopenxr-dev
```

The launcher is intentionally fixed to:
- `input_source=live_openxr_controller`
- `quest_display_mode=immersive`
- `scene_preset=ILLIXR_lab`
- `immersive_render_preset=balanced`
