# Boba Demo 2

Demo 2 runs a batched replay on the workstation monitor and lets phone users claim
individual sessions through a QR-code mobile web client.

Demo 2 reuses the working `phystwin` environment from the Boba-Batched checkout at
`/home/yihan/Research/Boba_Latest`. From the phone-demo repository, install the
small web dependency layer and validate the runtime first:

```bash
conda activate phystwin
bash env_install/install_demo2_extras.sh
bash scripts/demo2_preflight.sh
python tools/validate_demo2_assets.py --case single_push_rope_4
python tools/validate_demo2_assets.py --case double_stretch_sloth
```

Each case includes its filtered 100-trajectory controller bank. Normal users
should not run the filtering tool.

Start the demo through the runtime launcher so CUDA and Conda library paths are
configured consistently:

```bash
bash scripts/run_demo2.sh \
  --case_name single_push_rope_4 \
  --batch_size 100 \
  --batch_grid_cols 10 \
  --batch_image_resolution 640x480 \
  --host 0.0.0.0 \
  --port 7860
```

Run the double-stretch sloth with the same launcher:

```bash
bash scripts/run_demo2.sh \
  --case_name double_stretch_sloth \
  --batch_size 100 \
  --batch_grid_cols 10 \
  --batch_image_resolution 640x480 \
  --host 0.0.0.0 \
  --port 7860
```

Phones should be on the same LAN as the workstation. The QR code on the public
display opens the phone UI in iPhone Safari or Android Chrome. Phone input is
sampled at the start of each batched simulation frame; late input waits for the
next batch loop. Claiming a session switches the phone into a landscape-first
controller view in the normal mobile browser. `Release` stays in the upper-right
corner. The controller view sizes itself to the phone's visible browser viewport
so the stream and controls are not hidden behind the address or browser bars.
iPhone Safari/Chrome may keep browser chrome in normal QR mode; for a true
no-address-bar iPhone demo, open the page in Safari and use Add to Home Screen.
The QR overlay defaults to 320 pixels. The panel directly beneath it reports
rolling aggregate throughput in instances per second: actual public-display
frame cadence multiplied by the active batch size. Override the QR dimensions
with `--qr_size` when using a lower-resolution monitor.
Pass `--travel_router` on the dedicated router to add a numbered Wi-Fi QR for
the hardcoded `Emacs` demo network at the far-left edge while the controller QR
stays at the far-right edge. The controller QR automatically uses the
workstation's current LAN address unless `--public_url` overrides it. Omit the
flag for Cloudflare Tunnel and other LANs; they retain the single controller QR.
The Wi-Fi panel includes a reminder to disconnect the current Wi-Fi before
scanning when a phone will not switch networks automatically.

If an iPhone refuses the local HTTP URL with an HTTPS-only or not-secure
navigation warning, use this temporary local-demo workaround on that phone:
open the iOS Settings app, tap `Apps`, tap `Safari`, find `Privacy & Security`,
and toggle off `Not Secure Connection Warning`. Re-enable it after the demo.
For a stricter public demo setup, use an HTTPS tunnel and pass its URL with
`--public_url`.

When a phone claims a session, that instance resets to `--replay_start` and
stops following the default replay. It only moves from held phone buttons. When
the phone releases the session, or its heartbeat expires, that instance resets
again and restarts the default replay from frame 0.

The phone client shows original-playground-style Left Hand and Right Hand arrow
controls. In portrait, the render stays above the controls and the controls sit
on a light high-contrast strip so the original dark empty-arrow asset remains
visible. In landscape, the stream fills the phone viewport with the arrows over
the rendered image. Held arrows turn red for the left hand and blue for the
right hand. The public grid and phone stream also use the original PhysTwin hand
icons, anchored at the closest object/structure point to the controlled
controller group. `--demo2_control_parts auto` uses one shared controller for
normal cases and two controller groups when the case name contains `double`
anywhere, or when the case name is listed in `--demo2_double_control_cases` such
as `weird_package`. `double_stretch_sloth` therefore exposes two independent
interaction regions automatically: the phone's left controls drive the
viewer-left attachment and its right controls drive the viewer-right attachment.
Arrow-to-world signs are derived from each case's camera calibration, so the
sloth's rotated table view still makes left move left and forward move forward;
this does not alter physical table-Z collision handling.

`--phone_control_max_offset 0.0` is the default and disables the radial manual
travel clamp, matching the original playground behavior where a held button
keeps moving the controller. Table safety is always enforced separately. The
calibrated reset pose is already resting on the physical `z=0` table, so Z-down
cannot move farther into the table. After Z-up lifts the controller, Z-down can
return it to the reset pose. X/Y travel remains unbounded unless
`--phone_control_max_offset` is positive.

The phone stream defaults to `--phone_stream_size 640x480` so landscape control
mode stays readable. Use `--phone_stream_size 320x240` only when you need to save
LAN bandwidth. When multiple sessions are claimed, Demo 2 resizes all claimed
phone tiles together in one batched GPU operation before publishing per-session
JPEG frames. The phone normally receives those same JPEG bytes over an
acknowledged WebSocket: it requests the newest frame only after displaying the
previous one, so a slow phone or tunnel does not build a stale-frame queue. It
automatically falls back to MJPEG if WebSocket setup fails. JPEG encoding runs
outside the simulation/render thread, and phone controls are sent only when the
held-button state changes. Streaming is demand-driven by default: a ready phone
receives the next simulation frame instead of waiting for a separate 10 FPS
timer. Each claim gets one initial frame, then an idle phone retains that image
without duplicate JPEG traffic. A button press activates full adaptive
streaming. After release, a batched GPU check compares object positions between
simulation frames; five consecutive frames below the motion threshold produce
one final image and return the session to idle. Motion lasting beyond one second
switches to 2 FPS recovery updates until it is stable. A fast active phone can
therefore follow the packaged case's 30 FPS simulation, while a slow phone
receives fewer but still-current frames. Add
`--phone_stream_max_fps N` only when an event needs an explicit bandwidth or
encoding ceiling (`--phone_stream_fps` remains a compatibility alias). Frame
resolution stays 640x480 and JPEG quality stays 80.

Advanced settling defaults are `--phone_stream_settle_motion 1e-4`,
`--phone_stream_settle_frames 5`, `--phone_stream_full_rate_settle_s 1.0`, and
`--phone_stream_recovery_fps 2.0`. Normal demo launches should leave them at
their tested values.

To repeat the transport benchmark locally, or through a temporary Cloudflare
Quick Tunnel, run:

```bash
python tools/benchmark_demo2_phone_transport.py
python tools/benchmark_demo2_phone_transport.py --cloudflare
```

The benchmark serves a generated test pattern, not a repository or camera
image, and closes its temporary tunnel on exit.

For debugging the batch-100 lower-right tile, add `--demo2_debug_motion`. It
writes first-cycle target-motion diagnostics to `results/demo2/<case>/motion_debug.json`
and warns if the last session target moves while its rendered tile stays static.

## Developer-only trajectory regeneration

`demos/filter_demo_trajectories.py` is retained for developers who separately
possess the raw `multi_ctrls.pkl` bank and original case layout. It is not part
of the packaged runtime setup and is not needed to launch Demo 2.
