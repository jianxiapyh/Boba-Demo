# Boba Quest Rope Game

This repository contains one Quest/OpenXR demo: `rope_game`. It is a high-quality, three-sofa rope experience using the Boba-Batched custom `gsplat` fork with the existing two-eye rasterization path.

## Prerequisite: a working Boba-Batched machine

Before setting up this demo, install and successfully run the `Boba_Batched` branch in its `phystwin` Conda environment. That proves the machine already has the compatible NVIDIA driver, CUDA toolkit, PyTorch/CUDA stack, Conda installation, compiler toolchain, and desktop OpenGL/X11 support.

This demo reuses the installed packages in `phystwin`; it does **not** import source or assets from the Boba-Batched checkout. The custom `gsplat` source, rope data, and room assets required at runtime are committed here, so the Boba-Batched checkout does not need to be beside this repository.

Launch from the same working X11 session used for Boba-Batched. A valid `DISPLAY` is required even though the interactive window is hidden.

## Install the demo additions

Clone the demo and install only its pinned add-on packages:

```bash
git clone --branch Boba-Immersive-Demo-Quest \
  https://github.com/jianxiapyh/Boba-Demo.git Boba-Demo
cd Boba-Demo

conda run -n phystwin env PYTHONNOUSERSITE=1 \
  python -m pip install -r requirements-demo.txt
```

`requirements-demo.txt` intentionally omits Torch, CUDA, NumPy, Warp, Open3D, PyCUDA, and the rest of the Boba-Batched core stack. Pip's normal only-if-needed behavior leaves compatible installed dependencies in place; do not use `--upgrade` with this command.

The command also does not install or replace `gsplat` in `phystwin`. The demo selects its committed fork only inside the demo process, so a later Boba-Batched process continues to use the source and backend from its own checkout.

Validate the committed rope and room assets:

```bash
conda run -n phystwin env PYTHONNOUSERSITE=1 \
  python tools/fetch_demo_case_assets.py
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
bash boba_app.sh
```

The launcher also works when called by absolute path from another directory. It finds the repository root, verifies that `phystwin` exists, and starts exactly one `rope_game` session through `conda run`. It uses the native-GL static scene, one batched two-eye `gsplat.rasterization()` call, 1344-pixel eye resolution, and controller translation scale `0.25`.

The first run on a machine can pause while the repository-local custom `gsplat` CUDA extension and native OpenXR bridge compile. Those machine-specific products and JIT caches remain untracked. Later launches reuse the warmed caches.

Stop the game with `Ctrl+C` in the launching terminal.

## Troubleshooting

### `phystwin` is missing or core imports fail

Return to Boba-Batched and confirm it runs successfully in `phystwin`. Repair that baseline there rather than installing a second Torch/CUDA stack in this repository.

### Custom `gsplat` fails to compile

Confirm that `nvcc` is on `PATH`, its CUDA toolkit is compatible with the Torch build in `phystwin`, the NVIDIA driver is available, and the Torch extension cache is writable. The demo must report its `gsplat` source inside this checkout; it must not resolve from Boba-Batched or a global installation.

### PyCUDA/OpenGL interop fails

Boba-Batched installs PyCUDA, but this demo additionally requires `pycuda.gl`. Verify it without changing the environment:

```bash
conda run -n phystwin env PYTHONNOUSERSITE=1 \
  python -c 'import pycuda.gl; print("pycuda.gl is available")'
```

If that import fails, rebuild PyCUDA with CUDA OpenGL interoperability enabled for the existing `phystwin` environment. Do not replace Torch, CUDA, or NumPy while doing so.

### SteamVR or OpenXR cannot find the headset

- Confirm SteamVR and the desktop ALVR streamer are both running, the Quest ALVR client is open, and the headset is trusted in ALVR.
- Confirm the desktop streamer and Quest client versions match, and check local firewall/network rules if the headset is not discovered.
- Make SteamVR the active OpenXR runtime. If the OpenXR loader still selects the wrong runtime, set `XR_RUNTIME_JSON` to the `steamxr_linux64.json` file in the local SteamVR installation before running `boba_app.sh`.
- If the bridge dependency check fails, install the reported development packages and rerun it.

### No display or OpenGL context

Run from the working X11 desktop session and confirm `DISPLAY` is set. Hidden-window mode still creates an OpenGL context and cannot run from a display-less shell.
