# Phone Demo: On-Site Launch Guide

This guide is for the person carrying the demo computer. The demo case is
**double_stretch_sloth**.

For a browser version with buttons that copy each command, open
[PHONE_DEMO_OPERATOR_GUIDE.html](PHONE_DEMO_OPERATOR_GUIDE.html). It is a
self-contained local page and does not require Internet access.

To open that page directly from a terminal:

~~~bash
xdg-open /home/yihan/Research/Boba-Phone-Demo/PHONE_DEMO_OPERATOR_GUIDE.html
~~~

## Choose one mode

| Mode | Use it when | What attendees do |
| --- | --- | --- |
| **Travel router** | The venue Internet is unavailable or unreliable | Join the **Emacs** Wi-Fi, then scan the controller QR |
| **Cloudflare** | The computer has working Internet access | Stay on conference Wi-Fi or cellular and scan the controller QR |

## Before opening the venue

- Connect the computer to its monitor, keyboard, and mouse.
- Keep the computer connected to power.
- Have one phone available for a complete test.
- For travel-router mode, bring the router, its power supply, and an Ethernet cable.
- Phone video adapts automatically to each device; do not add a phone-FPS flag
  to the launch commands.

## Mode 1: Travel router

Internet access is not required.

1. Power on the travel router and wait for it to finish starting.
2. Connect the router to the demo computer with Ethernet.
3. Open a terminal and paste this complete command:

~~~bash
source /home/yihan/miniconda3/etc/profile.d/conda.sh
conda activate phystwin
cd /home/yihan/Research/Boba-Phone-Demo

bash scripts/run_demo2.sh \
  --case_name double_stretch_sloth \
  --batch_size 100 \
  --batch_grid_cols 10 \
  --batch_image_resolution 640x480 \
  --host 0.0.0.0 \
  --port 7860 \
  --qr_size 320 \
  --travel_router
~~~

4. Wait until the demo appears on the monitor.
5. Confirm that the monitor shows:

   - a Wi-Fi QR code on the far left;
   - a controller QR code on the far right; and
   - aggregate throughput below the controller QR.

The travel-router Wi-Fi is:

- Network: **Emacs**
- Password: **315810612**

Tell each attendee:

1. If already connected to another Wi-Fi network, disconnect from it.
2. Scan the left QR code and join **Emacs**.
3. Scan the right QR code to open the controller.

If the phone warns that **Emacs** has no Internet access, choose to remain
connected. The travel-router demo does not need Internet access.

Before the public demo, use the test phone to join **Emacs**, open the
controller, move both interaction points, and release them.

## Mode 2: Cloudflare

The demo computer must have working Internet access. Attendee phones do not
need to join **Emacs**; they can remain on conference Wi-Fi or cellular data.

### Terminal 1: Create the public address

Open a terminal and run:

~~~bash
cloudflared tunnel --url http://127.0.0.1:7860
~~~

Cloudflare prints a box like this in the terminal:

~~~text
2026-09-02T13:53:52Z INF +--------------------------------------------------------------------------------------------+
2026-09-02T13:53:52Z INF |  Your quick Tunnel has been created! Visit it at (it may take some time to be reachable):  |
2026-09-02T13:53:52Z INF |  https://attending-graduated-streams-distributors.trycloudflare.com                         |
2026-09-02T13:53:52Z INF +--------------------------------------------------------------------------------------------+
~~~

Copy only the complete address beginning with **https://** and ending with
**.trycloudflare.com**. The words in the address change every time. Keep
Terminal 1 open. Precheck messages followed by no new output are normal; the
tunnel is waiting for visitors.

### Terminal 2: Start the phone demo

Open a second terminal. Paste the block below after replacing the example
address with the exact address copied from Terminal 1:

~~~bash
source /home/yihan/miniconda3/etc/profile.d/conda.sh
conda activate phystwin
cd /home/yihan/Research/Boba-Phone-Demo

DEMO_PUBLIC_URL='https://PASTE-THE-GENERATED-URL.trycloudflare.com'

bash scripts/run_demo2.sh \
  --case_name double_stretch_sloth \
  --batch_size 100 \
  --batch_grid_cols 10 \
  --batch_image_resolution 640x480 \
  --host 127.0.0.1 \
  --port 7860 \
  --qr_size 320 \
  --public_url "$DEMO_PUBLIC_URL"
~~~

Wait until the demo appears. Confirm that the controller QR and aggregate
throughput are visible. Scan the QR with the test phone, open the controller,
move both interaction points, and release them. For the strongest check, put
the test phone on cellular data rather than the computer's network.

Only the controller QR is shown in Cloudflare mode. The missing left-side
Wi-Fi QR is intentional.

The generated Cloudflare address is public while the tunnel is running. Do not
post it publicly, and stop the tunnel after the demo.

## Stop the demo

- Travel-router mode: press **Ctrl+C** in the demo terminal.
- Cloudflare mode: press **Ctrl+C** in Terminal 2, then press **Ctrl+C** in
  Terminal 1.

## Quick troubleshooting

### The command says the phystwin environment is not active

Run these commands again, then retry the launch command:

~~~bash
source /home/yihan/miniconda3/etc/profile.d/conda.sh
conda activate phystwin
~~~

### Emacs does not appear immediately

Wait several seconds for the router to finish broadcasting. Confirm that the
router is powered and its Ethernet cable is connected. If needed, join it
manually with network **Emacs** and password **315810612**.

### A phone says it cannot join Emacs

Disconnect the phone from its current Wi-Fi network, wait until **Emacs**
appears in the Wi-Fi list, and scan the Wi-Fi QR again.

### The travel-router controller page does not open

Confirm that the phone is connected to **Emacs**. In the demo terminal, find
the line beginning with **[Demo2] Phone URL:**. The address should normally
start with **http://192.168.0.**.

To see the computer's current router-side address, open another terminal and
run:

~~~bash
ip -4 -brief address show enp0s31f6
~~~

### The Cloudflare address is hard to find

Scroll upward in Terminal 1 and look for **trycloudflare.com**. If no address
appears after about one minute, press **Ctrl+C** and run the Cloudflare command
again.

### Cloudflare shows Error 502

The tunnel is running, but the phone demo is not reachable locally. Confirm
that Terminal 2 is still running and that both commands use port **7860**, then
reload the phone page.

### The terminal says port 7860 is already in use

An older demo is probably still running. Return to its terminal, press
**Ctrl+C**, and run the selected mode again.
