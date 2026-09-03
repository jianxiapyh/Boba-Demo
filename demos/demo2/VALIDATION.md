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

- Full preflight: bundled runtime files, NVIDIA/CUDA, PyTorch CUDA allocation, `pycuda.gl`, bundled `gsplat.rasterization_shared_template`, hidden GLFW/OpenGL context, additions, and packaged assets.
- Post-extras bundled-runtime smoke: standard headless `single_push_rope_4`, batch 1, all 81 measured frames.
- Phone/API/session suite, including six-direction phone calibration, claim conflict, authenticated controls, heartbeat, release, timeout, invalid IDs, stream authorization, port-bind failure, and bundled-runtime provenance.
- Asset-validator suite: 4 tests.
- Packaged assets: 9 manifest entries, 100 controller trajectories, 34,983 Gaussian vertices, and 8 matching provenance hashes.
- Final bounded Demo 2 runtime: batch 1 and batch 100 with packaged assets and the shared `phystwin` environment.
- Live localhost integration: claim, conflict, control, heartbeat, release, reclaim, timeout, and an authenticated MJPEG frame.

## Manual physical-phone check

A particular phone and LAN cannot be validated automatically. Before presenting the demo, use the QR code from the target iPhone and verify claim, visible stream, controls, release, and timeout/reclaim behavior on that network. Do not treat the automated API checks as proof that iOS local-HTTP policy or Wi-Fi isolation is configured correctly.

## 2026-09-01 incremental validation

- Full preflight passed against the current CUDA/OpenGL environment and now
  validates both packaged cases.
- `double_stretch_sloth` validated with 9 manifest entries, 100 controller
  trajectories, 233,293 Gaussian vertices, and 8 matching provenance hashes.
- The 43-test Demo 2 suite passed, including real sloth-data checks for two
  disjoint, independently moving controller regions and rolling aggregate
  throughput calculations.
- The travel-router monitor overlay was rendered and visually checked at
  848x480 with a labeled 320-pixel Wi-Fi QR at far left, a labeled 320-pixel
  controller QR at far right, and `AGG THROUGHPUT` below the controller without
  overlap. The Wi-Fi QR includes a two-line disconnect reminder underneath.
  Automated decoding recovered both the `Emacs` Wi-Fi payload and the current
  `192.168.0.x` controller URL.
- A new live, full-screen batch-100 sloth endurance run was not performed during
  this incremental check; physical-phone and sustained-load checks remain part
  of the pre-presentation checklist.
- A follow-up camera-projection regression verifies that all six phone arrow
  directions agree with the visible motion for both rope and sloth. In
  particular, sloth left/right uses the opposite world-Y sign from rope while
  retaining the same table-Z sign.

## 2026-09-02 no-quality-loss phone-stream validation

- The controlled transport comparison held publishing at 10 FPS while retaining
  640x480 and JPEG quality 80. Both transports delivered the exact
  already-encoded JPEG bytes; the WebSocket path did not resize or re-encode
  them.
- The Demo 2 suite passed, including real loopback WebSocket delivery,
  byte equality, per-frame display acknowledgement, MJPEG fallback hooks,
  latest-only asynchronous encoding, and change-driven phone input.
- A synthetic 640x480 test pattern was used for both transport benchmarks. It
  averaged about 37 KiB per quality-80 JPEG and every received frame was decoded
  and checked for the expected dimensions and producer identity.
- On local loopback with a normal consumer, MJPEG and WebSocket both displayed
  10.2 FPS with median frame ages of 2.8 ms and 1.9 ms respectively. With a
  deliberately slow 200 ms/frame consumer, MJPEG queued stale data (1791 ms
  median, 3013 ms maximum); acknowledged WebSocket stayed current (28 ms
  median, 100 ms maximum).
- Through a temporary Cloudflare Quick Tunnel, normal-consumer median frame age
  was 215 ms for MJPEG versus 13 ms for WebSocket. With the same slow consumer,
  it was 2054 ms versus 68 ms (3275 ms versus 113 ms maximum). The freshness
  path displayed 4.4 FPS rather than 5.0 FPS in that artificial slow case
  because it waits for display acknowledgement instead of accumulating frames.
- With ten 640x480 sloth-demo frames, synchronous quality-80 encoding blocked
  the caller for 40.30 ms median per publish group. Two encoding workers reduced
  caller blocking to 8.35 ms (79.3%) while encoding all 300 submitted frames;
  the resulting JPEG bytes matched the synchronous encoder.
- Flask-Sock 0.7.0 and simple-websocket 1.1.0 were installed in `phystwin`.
  The guarded additions installer confirmed that the audited CUDA/PyTorch and
  scientific package versions were unchanged.
- Change-driven controls eliminate the old 33 ms idle input timer. Over a
  30-second idle claim this removes about 900 redundant input POSTs; including
  the unchanged heartbeat and session-list polling, expected periodic requests
  fall from about 925 to 26 (about 97%). Button transitions are serialized and
  retried so press/release ordering is retained.
- The complete GPU/OpenGL preflight passed, followed by a bounded three-frame
  `double_stretch_sloth` runtime using two interaction points and two JPEG
  encoding workers. It started the WebSocket-with-MJPEG-fallback server and
  exited cleanly.

## 2026-09-02 adaptive-stream follow-up

- The fixed 10 FPS phone sampling timer was replaced with per-session demand.
  After displaying a WebSocket frame, the phone acknowledges it; the next
  simulation render then captures and encodes only the sessions waiting for a
  frame. The default has no timer below the simulation cadence. The optional
  `--phone_stream_max_fps` flag supplies a safety ceiling, and the old
  `--phone_stream_fps` spelling remains as a compatibility alias.
- The 51-test suite passed. New coverage verifies multi-consumer demand,
  consumption/cancellation, refusal of requests after shutdown, ignoring a
  pre-connection stale frame, and waiting for both display acknowledgement and
  the next demanded simulation frame.
- On local loopback, the adaptive WebSocket followed the 30 FPS synthetic
  simulation at 30.2 FPS. With a deliberate 200 ms/frame client delay, it
  naturally settled at 4.2 FPS while delivered frames remained about 2 ms old.
- Through a temporary Cloudflare Quick Tunnel, the adaptive WebSocket delivered
  29.8 FPS with 12 ms median post-encode frame age. The delayed client again
  settled at 4.2 FPS with 13 ms median frame age and no stale-frame queue.
- A bounded real-GPU `double_stretch_sloth` run served 30 authenticated adaptive
  frames at 30.41 FPS. Every JPEG decoded at 640x480, averaged 28.8 KiB, and the
  runtime exited cleanly after its bound.

## 2026-09-02 activity-gated settling follow-up

- The adaptive transport is now gated by authoritative server-side button
  state. A claimed session publishes one current initial image, publishes no
  duplicate images while idle, and resumes on the first simulation frame after
  an input becomes active. Existing WebSocket demand remains pending while
  idle, so waking the stream does not require another client round trip.
- On release, RMS object-position displacement is calculated for all settling
  sessions in one GPU batch. Five consecutive frames at or below `1e-4` world
  units/frame trigger one final image and then idle. If motion remains after one
  second, image delivery falls to 2 FPS while motion detection continues, so a
  long oscillation cannot consume full-rate bandwidth indefinitely.
- The complete 58-test suite passed, including activity-state transitions,
  stable-frame counting, recovery pacing, action interruption, reclaim reset,
  current-frame MJPEG/WebSocket startup, batched motion calculation, and all
  earlier Demo 2 regressions.
- A real-GPU `double_stretch_sloth` interaction produced exactly one initial
  frame and then no idle frames. A 0.6-second button action produced 19 frames;
  after release, 19 settling frames were delivered before the stream became
  idle again. Every checked image decoded at 640x480.
- Isolated RTX PRO 6000 measurements put the position snapshot at 0.009 ms and
  the complete detector at 0.052 ms for one settling session, 0.053 ms for ten,
  and 0.066 ms for all 100 sessions (median; 0.069 ms p95 for 100). This is
  negligible beside simulation, rendering, and JPEG encoding.
