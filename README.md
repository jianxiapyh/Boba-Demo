# Boba Quest Immersive Demo

This repository contains one Quest/OpenXR experience with two runtime-selectable Gaussian objects and two launch-time scenes:

| Scene | Rope | Sloth |
| --- | --- | --- |
| Lab (default) | Game: course, targets, timer, HUD, and finish screen | Free play |
| Mip-NeRF 360 Garden | Free play | Free play |

Switching Rope/Sloth keeps the OpenXR session, selected scene, and head alignment alive. The Garden and active object are rendered together in one stereo-batched Gaussian pass and published through a Garden-only direct-output path; the Lab scene retains its mesh/Gaussian depth compositor. Garden's static Gaussians are stored in deterministic spatial chunks. Each source frame submits the conservative union of chunks intersecting either eye frustum, then leaves per-Gaussian frustum/radius culling to gsplat; the active object is always included. Fast invisible primitives supply contact: an analytic closed tabletop cylinder (top, rim, and underside), a finite patio plane, and inexpensive boxes for the table understructure. The detailed closed proxy mesh is retained only for developer visualization; vegetation, lawn, planters, and distant scenery remain non-colliding.

## Prerequisite: a working Boba-Batched machine

Before setting up this demo, install and successfully run the current `Boba_Batched` branch in its recommended `phystwin-cu132` Conda environment. That proves the machine already has the compatible NVIDIA driver, CUDA toolkit, PyTorch/CUDA stack, Conda installation, compiler toolchain, and desktop OpenGL/X11 support.

This demo reuses the installed packages in `phystwin-cu132`; it does **not** import source or assets from the Boba-Batched checkout. The custom `gsplat` source, Rope and Sloth data, and room assets required at runtime are committed here, so the Boba-Batched checkout does not need to be beside this repository. The Sloth Gaussian is tracked with Git LFS; clone with Git LFS enabled so the PLY payload is checked out rather than left as a pointer.

Launch from the same working X11 session used for Boba-Batched. A valid `DISPLAY` is required even though the interactive window is hidden.

## Install the demo additions

Clone the demo and install only its pinned add-on package:

```bash
git clone --branch Boba-Immersive-Demo-Quest \
  https://github.com/jianxiapyh/Boba-Demo.git Boba-Demo
cd Boba-Demo

conda run -n phystwin-cu132 env PYTHONNOUSERSITE=1 \
  python -m pip install -r requirements-demo.txt
```

`requirements-demo.txt` intentionally omits Torch, CUDA, NumPy, Warp, Open3D, PyCUDA, and the rest of the Boba-Batched core stack. Pip's normal only-if-needed behavior leaves compatible installed dependencies in place; do not use `--upgrade` with this command.

The command also does not install or replace `gsplat` in `phystwin-cu132`. The demo selects its committed fork only inside the demo process, so a later Boba-Batched process continues to use the source and backend from its own checkout.

Validate both selectable objects and the default Lab assets:

```bash
conda run -n phystwin-cu132 env PYTHONNOUSERSITE=1 \
  python tools/fetch_demo_case_assets.py
```

### One-time Garden setup

Garden is intentionally not stored in Git. Install the official pretrained Garden model and deterministically build the cleaned full/balanced/performance runtime tiers with:

```bash
conda run -n phystwin-cu132 env PYTHONNOUSERSITE=1 \
  python tools/fetch_demo_case_assets.py --scene garden --fetch
```

The setup command downloads the official Graphdeco pretrained-model archive once, verifies its pinned SHA-256 checksum, extracts Garden below the ignored `data/garden/source/` directory, removes the vase/flowers/tray, reconstructs the exposed tabletop from neighboring wood Gaussians, reinforces low-opacity tabletop splats for solid grazing-angle views, and writes the ignored LOD cache below `data/garden/runtime/`. Balanced and performance use deterministic top-k opacity pruning on only the exterior background, retaining 30% and 10% of those Gaussians respectively. The interaction region is protected at full density using each Gaussian's calibrated spatial support rather than center position alone. Each tier is also reordered into contiguous spatial chunks with conservative four-sigma bounds; this makes runtime stereo-frustum selection a cache operation rather than a per-frame full-scene gather. A valid cache is checksum-checked and reused, so later setup and launch operations work offline.

The upstream archive is large (approximately 13.6 GB), and the source plus generated tiers need additional local disk space. Only the source/license metadata, checksum manifest, deterministic calibration/removal settings, and compact collision-proxy description are committed. See `assets/scenes/garden/ASSET_LICENSE.md` before redistributing or using the asset outside its research terms.

To validate an already-installed Garden without downloading anything:

```bash
conda run -n phystwin-cu132 env PYTHONNOUSERSITE=1 \
  python tools/fetch_demo_case_assets.py --scene garden
```

The native OpenXR bridge requires OpenXR, GLFW, OpenGL, and X11 development files. Check them with:

```bash
bash linux_pose_probe/check_boba_immersive_bridge_deps.sh
```

Only if that check reports missing packages, install the command it prints. This is the only additional Ubuntu development setup normally needed beyond a working Boba-Batched machine. The bridge builds automatically on first launch if its ignored local binary is absent.

## Set up SteamVR and ALVR

The desktop and headset each need a matching part of ALVR:

1. Install the native Linux Steam package and SteamVR, then launch SteamVR once. Valve notes that the Steam Snap and Flatpak packages are unsupported for SteamVR on Linux; follow [SteamVR for Linux Support](https://help.steampowered.com/en/faqs/view/18A4-1E10-8A94-3DDA).
2. Download the Linux ALVR Launcher and install/launch the ALVR **streamer on the desktop**.
3. Connect the Quest over USB for installation and use the launcher's **Install APK** action to install the matching ALVR **client on the Quest**. Follow the [official ALVR installation guide](https://github.com/alvr-org/ALVR/wiki/Installation-guide), including Quest developer-mode/USB authorization when requested.
4. Start the desktop ALVR streamer and SteamVR. Open ALVR on the Quest, keep the headset awake, and select **Trust** for it in the streamer's Devices tab.
5. Confirm that SteamVR sees the headset and both controllers before starting the demo. Keep the desktop and Quest on a network suitable for ALVR streaming.

The desktop streamer and Quest client must use matching ALVR versions. Installing both through the same launcher version is the simplest way to ensure this.

## Run

From the cloned repository, run the canonical launcher:

```bash
./boba_app.sh
```

Launch the Gaussian Garden alternative with:

```bash
./boba_app.sh --scene garden
```

Garden never downloads during launch. If its local cache is missing or stale, startup exits before XR with the exact one-time setup command. The default is `balanced`; choose another tier explicitly with `--garden-quality full` or `performance`. The optional `--garden-quality auto` mode uses the highest cached tier measured at or above 72 source FPS for the current GPU, NVIDIA driver, renderer revision, model hash, and 1344-per-eye configuration. It never silently lowers eye resolution. An unprofiled `auto` run starts at `balanced`, records 120 source frames, and stores the result in the ignored local profile cache.

The launcher can be called from `base`, `phystwin`, `phystwin-cu130`, `phystwin-cu132`, or a shell with no active Conda environment. It discards inherited Conda/CUDA display variables and always starts a clean child in `phystwin-cu132`, so nested `conda run` metadata cannot make the correct Python report the wrong environment. It also works by absolute path from another directory, validates both objects plus only the selected scene before starting XR, and launches with Rope. Lab keeps the existing native-GL static room and compositor. Garden selects the union of conservatively visible static chunks, appends the complete active object, and publishes the combined Gaussian result directly before adding the shared controller/UI overlays. Both retain the fixed 1344-pixel-per-eye launcher default and controller translation scale `0.25`.

## Headset controls

- Trigger/Select: grab and move the active object. In the selector, point at a row and press Trigger to choose it.
- X/A: cycle interaction anchors during normal play. In the selector, move the highlighted row; press Trigger to confirm.
- Either joystick up/down: move the highlighted selector row once per deflection; recenter before moving again.
- Y/B short tap: in Lab, restart the Rope course or reset Sloth; in Garden, reset either selected object to its settled center-table pose.
- Y/B hold for 0.75 seconds: open the object selector. Y/B cancels it.
- Grip hold: exit the demo as before.

In Lab, selecting **Rope — Game** always creates a fresh course at target one with a reset timer; selecting **Sloth — Free Play** removes all Rope targets and HUD elements. In Garden, the selector labels both objects **Free Play**, and no goals, timer, targets, course HUD, or completion screen are created. Either object starts from its original settled position on the center table. A dark, headset-locked progress overlay retains the last valid view while the new object loads and settles, avoiding white frames. If loading fails, the demo shows an error and restores the previous object.

For collision/placement calibration, add `--garden-debug-collision`. Once head alignment is known, the demo exports the ignored `data/garden/debug/collision_proxy_world.obj`, including the detailed world-space reference proxy and placement-frame axes. Runtime contact uses the calibrated closed tabletop cylinder, patio surface, and understructure boxes described above.

The first run on a machine can pause while the repository-local custom `gsplat` CUDA extension and native OpenXR bridge compile. Those machine-specific products and JIT caches remain untracked. Later launches reuse the warmed caches.

Stop the game with `Ctrl+C` in the launching terminal.

## Troubleshooting

### `phystwin-cu132` is missing or core imports fail

Return to Boba-Batched and confirm it runs successfully in `phystwin-cu132`. Repair that baseline there rather than installing a second Torch/CUDA stack in this repository.

### Custom `gsplat` fails to compile

Confirm that `nvcc` is on `PATH`, its CUDA toolkit is compatible with the Torch build in `phystwin-cu132`, the NVIDIA driver is available, and the Torch extension cache is writable. The demo must report its `gsplat` source inside this checkout; it must not resolve from Boba-Batched or a global installation.

### Garden reports missing or stale data

Rerun the one-time setup command shown above. Setup verifies the official source and every generated runtime payload; launch performs fast manifest/calibration checks and never starts a network transfer. If an explicit tier is missing, either regenerate all tiers or choose one already present.

### PyCUDA/OpenGL interop fails

Boba-Batched installs PyCUDA, but this demo additionally requires `pycuda.gl`. Verify it without changing the environment:

```bash
conda run -n phystwin-cu132 env PYTHONNOUSERSITE=1 \
  python -c 'import pycuda.gl; print("pycuda.gl is available")'
```

If that import fails, rebuild PyCUDA with CUDA OpenGL interoperability enabled for the existing `phystwin-cu132` environment. Do not replace Torch, CUDA, or NumPy while doing so.

### SteamVR or OpenXR cannot find the headset

- Confirm SteamVR and the desktop ALVR streamer are both running, the Quest ALVR client is open, and the headset is trusted in ALVR.
- Confirm the desktop streamer and Quest client versions match, and check local firewall/network rules if the headset is not discovered.
- Make SteamVR the active OpenXR runtime. If the OpenXR loader still selects the wrong runtime, set `XR_RUNTIME_JSON` to the `steamxr_linux64.json` file in the local SteamVR installation before running `boba_app.sh`.
- If the bridge dependency check fails, install the reported development packages and rerun it.

### No display or OpenGL context

Run from the working X11 desktop session and confirm `DISPLAY` is set. Hidden-window mode still creates an OpenGL context and cannot run from a display-less shell.
