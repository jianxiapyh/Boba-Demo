# Boba Immersive Demo: On-Site Launch Guide

This guide is for the person operating the demo computer. It covers the two
event scenes: the default mesh-based **Lab** and the Gaussian **Ambulance**.

For a copy-friendly offline version, open
[the local operator webpage](IMMERSIVE_DEMO_OPERATOR_GUIDE.html), or run:

~~~bash
cd /home/yihan/Research/Boba-Demo
./open_operator_guide.sh
~~~

## Start ALVR first

Open Terminal 1 and run:

~~~bash
cd /home/yihan/Downloads/alvr_streamer_linux/bin
./alvr_dashboard
~~~

Keep Terminal 1 and the ALVR dashboard open. After the dashboard appears:

1. Put the computer and Quest on **Emacs**, the default demo network.
2. Start SteamVR.
3. Open ALVR on the Quest.
4. Trust/connect the headset in the desktop ALVR dashboard.
5. Confirm that SteamVR sees the headset and both controllers.

### If the network is not Emacs

The currently saved ALVR connection is:

| Field | Current value |
| --- | --- |
| **Hostname** | `7921.client` |
| **IP Addresses** | `10.200.15.239` |

These values are not guaranteed to remain valid after changing networks. To
update them:

1. Connect the computer and Quest to the same low-latency network. Use Ethernet
   for the computer when available.
2. On the Quest, open ALVR and copy the **Hostname** and **IP address** shown on
   its welcome/trust screen.
3. In the desktop ALVR dashboard, open **Devices**, find the saved Quest, and
   choose **Edit connection**.
4. Replace **Hostname** and the existing **IP Addresses** entry with the exact
   values shown in the Quest. Use **Add new** only when intentionally keeping an
   additional known address.
5. Select **Save**, choose **Trust** if requested, and confirm that it connects.

Use the values displayed by ALVR inside the Quest. Do not enter the computer's
hostname or IP address.

## Prepare the headset and computer

- Connect the demo computer to power, a monitor, keyboard, and mouse.
- Charge the Quest and check both controller batteries.
- Have the wearer stand, put on the Quest, keep it upright, and look naturally
  forward. Do not launch while the headset is lying on a desk; startup captures
  its position and direction.

The launcher currently uses an explicit **standing** layout; it does not try to
detect posture automatically. Use `--immersive_start_posture seated` only for a
seated test.

## Choose a scene

| Scene | Use it for | Available experiences |
| --- | --- | --- |
| **Lab (default)** | The standard mesh-based demo | Rope game or Sloth free play |
| **Ambulance** | The immersive ambulance environment | Rope or Sloth free play on the stretcher |

The scene is chosen when the demo starts. To change scenes, stop the demo and
launch it again. Rope and Sloth can be switched without restarting.

## Start the Lab scene

Open a terminal and run:

~~~bash
source /home/yihan/miniconda3/etc/profile.d/conda.sh
cd /home/yihan/Research/Boba-Demo
./boba_app.sh
~~~

The demo starts with Rope. In Lab, Rope includes the course, targets, timer,
and HUD. Sloth is free play. The standing layout places the headset 1.55 m above
the Lab floor, with the table 0.82 m below and 0.78 m ahead of the headset.

## Start the Ambulance scene

Open a terminal and run:

~~~bash
source /home/yihan/miniconda3/etc/profile.d/conda.sh
cd /home/yihan/Research/Boba-Demo
./boba_app.sh --scene ambulance
~~~

The initial view should be from the clear aisle beside the stretcher, with the
Rope resting on top of it. The headset is 1.45 m above the ambulance floor and
starts with a 30-degree downward view toward the mattress. Both Rope and Sloth
are free play in this scene.

The `source` command makes Conda available. The launcher then selects the
required `phystwin-cu132` environment automatically; the operator does not need
to activate it.

## Finish startup and check the demo

1. Press either trigger once to advance each tutorial page.
2. On the final page, wait for **Ready**, then press a trigger.
3. Confirm that the chosen scene and Rope appear in the headset.
4. Confirm that both controller rays move and that Trigger can grab and release
   the object.
5. Check the desktop spectator window. It should show the scene, object,
   controller rays, and tracked headset.

The first launch on a computer can pause while CUDA or the OpenXR bridge is
compiled. Leave the terminal open and wait while it continues printing output.

## Headset controls

| Control | Action |
| --- | --- |
| **Trigger** | Advance tutorial pages; during play, hold to grab and release to drop |
| **Point at a marker** | Select that interaction point automatically |
| **X or A** | Optionally cycle interaction points when pointing is ambiguous |
| **Y or B, short press** | Restart the Lab Rope course or reset the current free-play object |
| **Y or B, hold 0.75 seconds** | Open the Rope/Sloth selector |
| **Either joystick up/down** | Move through the selector; recenter it before moving again |
| **Trigger in selector** | Confirm the highlighted object |
| **Y or B in selector** | Cancel the selector |
| **Grip, hold** | Exit the demo |

When switching Rope or Sloth, wait for the loading overlay to finish. Do not
press the selection buttons repeatedly.

## Reset between attendees

- Short-press **Y or B** to return the current experience to its starting state.
- In Ambulance, this returns the object to the stretcher.
- If a new wearer starts with a badly tilted or displaced view, stop and relaunch
  while that person is wearing the headset and looking forward.

## Stop the demo

Press **Ctrl+C** in the launch terminal. A wearer can also hold either Grip to
exit. Then close SteamVR and ALVR when the event is finished.

## Quick troubleshooting

### The headset or controllers are not detected

Stop the demo. Confirm that the ALVR client is open on the Quest, the headset is
trusted in the desktop streamer, and SteamVR shows the headset and both
controllers. Then launch the demo again.

### The initial view is tilted or in the wrong place

Stop the demo, recenter the SteamVR/ALVR view if needed, and relaunch with the
headset worn upright while the wearer looks naturally forward.

### The final tutorial page does not continue

Wait until it says **Ready**, release the trigger fully, and press it once. The
demo may still be preparing the scene in the background.

### The desktop window is missing or OpenGL fails

Launch from the computer's local X11 desktop terminal. The default spectator
window requires a valid `DISPLAY`, even though the Quest receives its own view.

### Object switching appears frozen

Wait for the progress overlay. If the terminal reports an error or stops making
progress, press **Ctrl+C**, verify SteamVR is still connected, and relaunch the
same scene.
