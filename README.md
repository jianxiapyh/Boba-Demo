# Boba Phone Demo

This branch packages Boba Demo 2: a 100-session batched replay on the workstation display with QR-based phone claiming, phone controls, and a low-latency per-session phone stream.

## Prerequisite: working Boba-Batched

This branch deliberately reuses the environment prepared for Boba-Batched. Before setting up the phone demo, the following must already work:

- `Boba_Batched` is checked out at `/home/yihan/Research/Boba_Latest`.
- Its `phystwin` Conda environment can run a CUDA/rendering Boba-Batched workload.
- NVIDIA CUDA, X11/OpenGL, PyCUDA OpenGL interoperability, and the custom Boba-Batched `gsplat` runtime are operational.

The runtime snapshot in this branch was imported from `Boba_Batched@99e50055a60a4bc7e5022abba1a938bf386b273d`. The preflight also verifies the expected interfaces in the active `Boba_Batched` checkout so a later compatible revision can be reused.

If the Boba-Batched checkout is elsewhere, export its location before setup and launch:

```bash
export BOBA_BATCHED_ROOT=/path/to/Boba_Latest
```

The phone-demo setup does not install or upgrade PyTorch, CUDA, Warp, PyCUDA, NumPy, Open3D, or `gsplat`.

## One-time phone-demo setup

```bash
conda activate phystwin
mkdir -p /home/yihan/Research
git clone --single-branch --branch Boba-Phone-Demo \
  https://github.com/jianxiapyh/Boba-Demo.git \
  /home/yihan/Research/Boba-Phone-Demo
cd /home/yihan/Research/Boba-Phone-Demo

bash env_install/install_demo2_extras.sh
bash scripts/demo2_preflight.sh
python tools/validate_demo2_assets.py --case single_push_rope_4
python tools/validate_demo2_assets.py --case double_stretch_sloth
python -m unittest discover -s demos/demo2/test -p 'test_*.py' -v
```

`install_demo2_extras.sh` installs only missing phone-web dependencies and verifies that core Boba package versions do not change.

Normal users do not need to filter trajectories. Each packaged case already
contains its validated 100-trajectory controller bank.

## Run

For a concise on-site handoff with exact commands for both network modes, see
[Phone Demo: On-Site Launch Guide](PHONE_DEMO_OPERATOR_GUIDE.md). A
[local browser version with copy buttons](PHONE_DEMO_OPERATOR_GUIDE.html) is
also available and works offline.

Open the local page directly:

~~~bash
xdg-open /home/yihan/Research/Boba-Phone-Demo/PHONE_DEMO_OPERATOR_GUIDE.html
~~~

Run a bounded one-session smoke test first:

```bash
bash scripts/run_demo2.sh \
  --case_name single_push_rope_4 \
  --batch_size 1 \
  --batch_grid_cols 1 \
  --max_frames 3
```

Start the full demo:

```bash
bash scripts/run_demo2.sh \
  --case_name single_push_rope_4 \
  --batch_size 100 \
  --batch_grid_cols 10 \
  --batch_image_resolution 640x480 \
  --host 0.0.0.0 \
  --port 7860
```

To run the two-interaction-point sloth demo, use the same command with
`--case_name double_stretch_sloth`. Its left and right phone controls drive the
viewer-left and viewer-right attachment regions independently. Phone arrow
directions are calibrated from the selected case's camera, which accounts for
the sloth view's opposite X/Y orientation without changing Z behavior.

Batch 100 is the target configuration and has been exercised on RTX PRO 6000 Blackwell. Use a smaller explicit `--batch_size` on GPUs that cannot fit the full workload.

## Connect a phone

The workstation display shows a large QR code with rolling aggregate simulation
throughput beneath it. Open the QR from iPhone Safari or Android Chrome while
the phone and workstation are on the same LAN. The phone lists available
sessions; claiming one resets that instance and switches the browser to its
controller and video stream.

Phone video is activity-gated and adaptive by default. A claim receives one
initial image; an idle phone keeps that image without receiving duplicate
JPEGs. While a button is held, each display acknowledgement requests the next
simulation render. After release, updates continue until measured object motion
is tiny for five frames. Use `--phone_stream_max_fps N` only when an explicit
event bandwidth ceiling is needed.

Only TCP port `7860` is used. If UFW is active, permit that port only from the trusted LAN. Replace the interface and subnet in this example:

```bash
sudo ufw allow in on <lan-interface> from <lan-subnet> to any port 7860 proto tcp
```

If a VPN or multiple network adapters cause the QR code to advertise the wrong address, pass the workstation LAN address explicitly:

```bash
bash scripts/run_demo2.sh \
  --case_name single_push_rope_4 \
  --batch_size 100 \
  --public_url http://192.168.1.50:7860
```

`--public_url` changes the advertised URL; it does not create an HTTPS tunnel. Some iPhones warn about local HTTP pages. For a public deployment, provide an access-controlled HTTPS tunnel separately and pass its URL through `--public_url`. Anyone who can reach this server can claim an available session, so do not expose it without access controls.

For the dedicated travel router, keep using the fixed demo address:

```bash
bash scripts/run_demo2.sh \
  --case_name double_stretch_sloth \
  --host 0.0.0.0 \
  --port 7860 \
  --travel_router
```

`--travel_router` shows two numbered QR codes: the far-left code joins the
hardcoded `Emacs` Wi-Fi network and the far-right code opens the controller at
the workstation's current LAN address. This avoids hardcoding a DHCP address.
The note below the Wi-Fi QR reads `IF ALREADY ON ANOTHER WI-FI / DISCONNECT,
THEN SCAN`.
Omit `--travel_router` for public tunnel URLs and other LANs; they continue to
show only the controller QR. If multiple network adapters make automatic address
detection choose incorrectly, add `--public_url` with the workstation's current
travel-router address.

Before presenting the demo, verify from the physical phone that it can scan the QR code, claim and control a session, release it, and reclaim a session after the heartbeat timeout. Automated tests cover the same API state transitions, but they cannot validate a particular phone, Wi-Fi network, or iOS local-HTTP policy.

See [`demos/demo2/VALIDATION.md`](demos/demo2/VALIDATION.md) for the recorded environment versions and completed validation matrix.

## Packaged assets

Runtime payloads are under `assets/single_push_rope_4/` and
`assets/double_stretch_sloth/`, with each case resolved through its manifest.
They contain only the checkpoint, calibration, metadata, optimized parameters,
final data, Gaussian PLY, background, and filtered controller bank required at
runtime.

The raw trajectory bank and original training dataset are intentionally excluded. `demos/filter_demo_trajectories.py` remains a developer tool for users who separately possess those inputs.

## Troubleshooting

- **Wrong environment:** activate `phystwin`; the scripts reject other environments.
- **Missing Boba-Batched checkout:** set `BOBA_BATCHED_ROOT` to the working checkout.
- **OpenGL/display failure:** confirm `DISPLAY` is set and that a Boba-Batched rendering command works in the same shell.
- **`libstdc++` import errors:** launch through `scripts/run_demo2.sh`, which places `$CONDA_PREFIX/lib` before system libraries.
- **Phone cannot connect:** confirm both devices are on the same non-isolated LAN, allow TCP `7860`, and use an explicit `--public_url`.
- **CUDA out of memory:** retry with a smaller explicit `--batch_size` and matching `--batch_grid_cols`.
