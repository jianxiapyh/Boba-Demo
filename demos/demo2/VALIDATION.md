# Demo 2 validation record

Validation date: 2026-07-10

Boba-Batched source: `Boba_Batched@99e50055a60a4bc7e5022abba1a938bf386b273d`

GPU: NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition, 95 GiB

Environment: `phystwin`, Python 3.10.20, CUDA 12.8 PyTorch build

## Shared-environment audit

`env_install/install_demo2_extras.sh` found Flask, qrcode/Pillow, and Ninja already installed, so it performed no package installation. Its before/after snapshots were identical:

| Distribution | Before | After |
| --- | --- | --- |
| torch | 2.10.0+cu128 | 2.10.0+cu128 |
| torchvision | 0.25.0+cu128 | 0.25.0+cu128 |
| torchaudio | 2.10.0+cu128 | 2.10.0+cu128 |
| numpy | 1.26.4 | 1.26.4 |
| scipy | 1.15.3 | 1.15.3 |
| warp-lang | 1.12.1 | 1.12.1 |
| pycuda | 2026.1 | 2026.1 |
| gsplat | 1.5.3 | 1.5.3 |
| pytorch3d | 0.7.9 | 0.7.9 |
| open3d | 0.19.0 | 0.19.0 |
| PyOpenGL | 3.1.0 | 3.1.0 |
| glfw | 2.10.0 | 2.10.0 |
| kornia | 0.8.2 | 0.8.2 |
| Pillow | 12.2.0 | 12.2.0 |

Phone additions present during validation: Flask 3.1.3, qrcode 8.2, Pillow 12.2.0, and Ninja 1.13.0.

## Passed checks

- Full preflight: NVIDIA/CUDA, PyTorch CUDA allocation, `pycuda.gl`, custom Boba-Batched `gsplat.rasterization_shared_template`, hidden GLFW/OpenGL context, additions, and packaged assets.
- Post-extras Boba-Batched smoke: standard headless `single_push_rope_4`, batch 1, all 81 measured frames.
- Phone/API/session suite: 27 tests, including claim conflict, authenticated controls, heartbeat, release, timeout, invalid IDs, stream authorization, and port-bind failure.
- Asset-validator suite: 4 tests.
- Packaged assets: 9 manifest entries, 100 controller trajectories, 34,983 Gaussian vertices, and 8 matching provenance hashes.
- Final bounded Demo 2 runtime: batch 1 and batch 100 with packaged assets and the shared `phystwin` environment.
- Live localhost integration: claim, conflict, control, heartbeat, release, reclaim, timeout, and an authenticated MJPEG frame.

## Manual physical-phone check

A particular phone and LAN cannot be validated automatically. Before presenting the demo, use the QR code from the target iPhone and verify claim, visible stream, controls, release, and timeout/reclaim behavior on that network. Do not treat the automated API checks as proof that iOS local-HTTP policy or Wi-Fi isolation is configured correctly.
