# Boba Quest Immersive Demo

> **Operating the demo?** Start with the
> [copy-friendly HTML operator guide](IMMERSIVE_DEMO_OPERATOR_GUIDE.html), or
> clone this branch and run `./open_operator_guide.sh` to open it locally. A
> [plain Markdown version](IMMERSIVE_DEMO_OPERATOR_GUIDE.md) is also available.

This repository contains one Quest/OpenXR experience with two runtime-selectable Gaussian objects and three launch-time scenes:

| Scene | Rope | Sloth |
| --- | --- | --- |
| Lab (default) | Game: course, targets, timer, HUD, and finish screen | Free play |
| Mip-NeRF 360 Garden | Free play | Free play |
| Insta360 Ambulance | Free play | Free play |

Switching Rope/Sloth keeps the OpenXR session, selected scene, and head alignment alive. Garden and Ambulance each combine their static scene with the active object in one stereo-batched Gaussian pass; Lab retains its mesh/Gaussian depth compositor. Garden's static Gaussians are stored in deterministic spatial chunks and use an analytic patio-table collision proxy. The launcher now uses an explicit standing layout by default, with no automatic posture detection. Lab starts at a 1.55 m eye height. Ambulance decodes the bundled SOG v2 asset directly, places the active object on top of the stretcher, anchors the launch-time headset in the clear aisle at a 1.45 m eye height with a 30-degree downward view toward the mattress, and uses the captured full-resolution padded-mattress shell plus aggressively simplified source-mesh geometry for the side hardware and undercarriage. Pass `--immersive_start_posture seated` to use the preserved seated layouts.

## Prerequisite: a working Boba-Batched machine

Before setting up this demo, install and successfully run the current `Boba_Batched` branch in its recommended `phystwin-cu132` Conda environment. That proves the machine already has the compatible NVIDIA driver, CUDA toolkit, PyTorch/CUDA stack, Conda installation, compiler toolchain, and desktop OpenGL/X11 support.

This demo reuses the installed packages in `phystwin-cu132`; it does **not** import source or assets from the Boba-Batched checkout. The custom `gsplat` source, Rope and Sloth data, and room assets required at runtime are committed here, so the Boba-Batched checkout does not need to be beside this repository. The Sloth Gaussian is tracked with Git LFS; clone with Git LFS enabled so the PLY payload is checked out rather than left as a pointer.

Launch from the same working X11 session used for Boba-Batched. A valid `DISPLAY` is required. The interactive window is visible by default and uses an independent spectator camera: the selected scene, deformable object, controller rays, and tracked 3D headset mesh are rendered together in scene space. The headset mesh is the asset used by ILLIXR's `plugins/debugview`; the Quest continues to receive its normal stereo eye views. Pass `--interactive_window_mode hidden` to disable the desktop spectator view.

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

For a concise on-site handoff covering the Lab and Ambulance scenes, open the
offline guide with copy buttons:

```bash
./open_operator_guide.sh
```

The same instructions are also available as a
[local webpage](IMMERSIVE_DEMO_OPERATOR_GUIDE.html) and a
[plain Markdown guide](IMMERSIVE_DEMO_OPERATOR_GUIDE.md).

From the cloned repository, run the canonical launcher:

```bash
./boba_app.sh
```

Launch the Gaussian Garden alternative with:

```bash
./boba_app.sh --scene garden
```

Garden never downloads during launch. If its local cache is missing or stale, startup exits before XR with the exact one-time setup command. The default is `balanced`; choose another tier explicitly with `--garden-quality full` or `performance`. The optional `--garden-quality auto` mode uses the highest cached tier measured at or above 72 source FPS for the current GPU, NVIDIA driver, renderer revision, model hash, and 1344-per-eye configuration. It never silently lowers eye resolution. An unprofiled `auto` run starts at `balanced`, records 120 source frames, and stores the result in the ignored local profile cache.

Launch the bundled Insta360 Ambulance scene with:

```bash
./boba_app.sh --scene ambulance
```

The Ambulance SOG is validated by version, Gaussian count, and SHA-256 checksum before XR starts. It is decoded directly in memory, so no PLY conversion or one-time preparation command is required. Stretcher contact uses the captured mattress shell at its full selected resolution (46,829 triangles), preserving its real curvature and longer asymmetric end. The handles, side frame, legs, and undercarriage remain aggressively simplified (10,513 triangles), keeping the complete position/index-only collision proxy near 1.06 MB. It is used as a two-sided collision surface; rendering colors, normals, UVs, textures, and materials are deliberately omitted. The authored object anchor remains at the calibrated mattress reference, and the hidden startup settle resolves individual nodes onto the irregular captured surface. To avoid paying for an exact BVH query on all 167 spring substeps, Ambulance checks the source mesh every 16 substeps, always checks the final substep, and sweeps continuously from the previous check so fast motion cannot tunnel through the skipped interval.

The launcher can be called from `base`, `phystwin`, `phystwin-cu130`, `phystwin-cu132`, or a shell with no active Conda environment. It discards inherited Conda/CUDA display variables and always starts a clean child in `phystwin-cu132`, so nested `conda run` metadata cannot make the correct Python report the wrong environment. It also works by absolute path from another directory, validates both objects, the selected scene, and the shared spectator-headset assets before starting XR, and launches with Rope. Lab keeps the existing native-GL static room and compositor. Garden and Ambulance append the complete active object to their static Gaussians and publish the combined result directly before adding the shared controller/UI overlays. All scenes retain the fixed 1344-pixel-per-eye launcher default and controller translation-scale multiplier `0.25`. Rope and Sloth both have a case-default gain of `4.0`, producing an effective gain of `1.0`: 5 cm of real controller motion maps to 5 cm in scene space for either object. While an object is grabbed, its simulated controller target advances by at most 0.05 m in each rendered simulation period. Any excess is carried into later displayed frames and always chases the newest tracked pose, so every frame runs one physics graph followed by one LBS/render update instead of blocking presentation on a burst of catch-up graphs. Releasing the grab discards unfinished catch-up motion. The largest consecutive driven-point displacement measured across all 22 recorded 30 FPS test trajectories is `0.047083356 m`; the 5 cm production limit leaves 2.92 mm (6.2%) headroom. [The calibration record](assets/controller_motion_calibration.json) contains the per-case measurements. Override the bound with `--immersive_controller_max_motion_interval_m` when stress testing.

## Headset controls

- Trigger/Select: point near an interaction marker and grab; its enlarged invisible ray target takes priority even when another part of the object is in front. In the selector, point at a row and press Trigger to choose it.
- X/A: optionally cycle interaction anchors when direct pointing is ambiguous. In the selector, move the highlighted row; press Trigger to confirm.
- Either joystick up/down: move the highlighted selector row once per deflection; recenter before moving again.
- Y/B short tap: in Lab, restart the Rope course or reset Sloth; in Garden or Ambulance, reset either selected object to its settled interaction-surface pose.
- Y/B hold for 0.75 seconds: open the object selector. Y/B cancels it.
- Grip hold: exit the demo as before.

In Lab, selecting **Rope — Game** always creates a fresh course at target one with a reset timer; selecting **Sloth — Free Play** removes all Rope targets and HUD elements. In Garden and Ambulance, the selector labels both objects **Free Play**, and no goals, timer, targets, course HUD, or completion screen are created. Either object starts from its original settled position on the calibrated interaction surface. A dark, headset-locked progress overlay retains the last valid view while the new object loads and settles, avoiding white frames. If loading fails, the demo shows an error and restores the previous object.

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

## License

Except where otherwise noted, Boba-Demo is licensed under the
[Apache License 2.0](LICENSE). See [NOTICE](NOTICE) for attribution information.
Vendored software and externally sourced assets retain their respective
licenses; consult the license files distributed with those components and the
asset-specific licensing documents under `assets/`.
