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
```

The filtered 100-trajectory controller bank is packaged with the case. Normal
users should not run the filtering tool.

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

Phones should be on the same LAN as the workstation. The QR code on the public
display opens the phone UI in iPhone Safari or Android Chrome. Phone input is
sampled at the start of each batched simulation frame; late input waits for the
next batch loop. Claiming a session switches the phone into a landscape-first
controller view in the normal mobile browser. `Release` stays in the upper-right
corner. The controller view sizes itself to the phone's visible browser viewport
so the stream and controls are not hidden behind the address or browser bars.
iPhone Safari/Chrome may keep browser chrome in normal QR mode; for a true
no-address-bar iPhone demo, open the page in Safari and use Add to Home Screen.

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
as `weird_package`.

`--phone_control_max_offset 0.0` is the default and disables manual travel
clamping, matching the original playground behavior where a held button keeps
moving the controller. Set a positive value only when a bounded demo workspace is
desired.

The phone stream defaults to `--phone_stream_size 640x480` so landscape control
mode stays readable. Use `--phone_stream_size 320x240` only when you need to save
LAN bandwidth. When multiple sessions are claimed, Demo 2 resizes all claimed
phone tiles together in one batched GPU operation before publishing per-session
MJPEG frames.

For debugging the batch-100 lower-right tile, add `--demo2_debug_motion`. It
writes first-cycle target-motion diagnostics to `results/demo2/<case>/motion_debug.json`
and warns if the last session target moves while its rendered tile stays static.

## Developer-only trajectory regeneration

`demos/filter_demo_trajectories.py` is retained for developers who separately
possess the raw `multi_ctrls.pkl` bank and original case layout. It is not part
of the packaged runtime setup and is not needed to launch Demo 2.
