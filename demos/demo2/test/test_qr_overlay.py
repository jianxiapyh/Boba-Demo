import unittest

import numpy as np

from demos.demo2.qr_overlay import (
    TRAVEL_ROUTER_WIFI_PASSWORD,
    TRAVEL_ROUTER_WIFI_SSID,
    make_public_display_qr_overlays,
    make_travel_router_wifi_payload,
)


class Demo2QrOverlayTest(unittest.TestCase):
    def test_wifi_qr_requires_explicit_travel_router_mode(self):
        _, default_wifi = make_public_display_qr_overlays(
            "http://192.168.0.218:7860",
            size=96,
        )
        _, travel_router_wifi = make_public_display_qr_overlays(
            "http://192.168.0.218:7860",
            size=96,
            include_travel_router_wifi=True,
        )
        self.assertIsNone(default_wifi)
        self.assertIsNotNone(travel_router_wifi)

    def test_wifi_payload_contains_hardcoded_demo_credentials(self):
        self.assertEqual(TRAVEL_ROUTER_WIFI_SSID, "Emacs")
        self.assertEqual(TRAVEL_ROUTER_WIFI_PASSWORD, "315810612")
        self.assertEqual(
            make_travel_router_wifi_payload(),
            "WIFI:T:WPA;S:Emacs;P:315810612;;",
        )

    def test_cloudflare_url_keeps_single_qr(self):
        size = 96
        controller_overlay, wifi_overlay = make_public_display_qr_overlays(
            "https://boba-example.trycloudflare.com",
            size=size,
        )
        self.assertEqual(controller_overlay.shape, (size, size, 3))
        self.assertEqual(controller_overlay.dtype, np.uint8)
        self.assertIsNone(wifi_overlay)

    def test_travel_router_url_builds_two_separate_numbered_qrs(self):
        size = 96
        controller_overlay, wifi_overlay = make_public_display_qr_overlays(
            "http://192.168.0.218:7860",
            size=size,
            include_travel_router_wifi=True,
        )
        self.assertGreater(controller_overlay.shape[0], size)
        self.assertGreater(wifi_overlay.shape[0], controller_overlay.shape[0])
        for overlay in (controller_overlay, wifi_overlay):
            self.assertEqual(overlay.shape[1:], (size, 3))
            self.assertEqual(overlay.dtype, np.uint8)


if __name__ == "__main__":
    unittest.main()
