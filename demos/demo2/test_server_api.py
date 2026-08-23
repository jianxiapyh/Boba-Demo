import json
import pickle
import socket
import tempfile
import unittest
from html.parser import HTMLParser
from pathlib import Path

import numpy as np

from demos.demo2.case_assets import (
    REQUIRED_MANIFEST_KEYS,
    load_demo2_case_config,
    resolve_demo2_case_assets,
    select_controller_bank,
)
from demos.demo2.session_manager import SessionManager
from demos.demo2.replay_state import (
    ReplayStateStore,
    build_replay_action_table,
    controls_from_world_delta,
)
from demos.demo2.streaming import MjpegFrameStore
from demos.demo2_server import (
    REPO_ROOT,
    build_parser,
    create_app,
    load_phone_client_html,
    start_flask_server,
)


class ButtonParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.buttons = {}

    def handle_starttag(self, tag, attrs):
        if tag != "button":
            return
        attrs = dict(attrs)
        if attrs.get("class") != "control-button":
            return
        key = (attrs.get("data-hand"), attrs.get("data-key"))
        self.buttons[key] = attrs


class Demo2ServerApiTest(unittest.TestCase):
    def test_replay_state_route_reports_rendered_actions(self):
        manager = SessionManager(num_sessions=2)
        store = ReplayStateStore(num_sessions=2, control_parts=2)
        store.set_runtime_fps(30)
        app = create_app(manager, MjpegFrameStore(), store)
        client = app.test_client()

        self.assertEqual(client.get("/api/replay/0").status_code, 503)
        store.publish(
            17,
            [4, 8],
            [(('left', 'zneg'), ('right', 'xpos')), ()],
        )
        response = client.get("/api/replay/0")
        self.assertEqual(response.status_code, 200)
        state = response.get_json()
        self.assertEqual(state["sequence"], 17)
        self.assertEqual(state["frame_idx"], 4)
        self.assertEqual(state["control_parts"], 2)
        self.assertEqual(state["runtime_fps"], 30.0)
        self.assertEqual(
            state["controls"],
            [
                {"hand": "left", "control": "zneg"},
                {"hand": "right", "control": "xpos"},
            ],
        )

    def test_replay_world_motion_maps_to_phone_buttons(self):
        self.assertEqual(
            controls_from_world_delta("left", (-1, 2, -3)),
            (("left", "xneg"), ("left", "ypos"), ("left", "zneg")),
        )
        self.assertEqual(
            controls_from_world_delta("right", (1, -2, 3)),
            (("right", "xpos"), ("right", "yneg"), ("right", "zpos")),
        )

    def test_replay_action_table_is_neutral_on_reset_and_tracks_both_hands(self):
        points = np.zeros((1, 3, 4, 3), dtype=np.float32)
        points[0, 1, :2, 0] = -0.1
        points[0, 1, 2:, 2] = 0.2
        points[0, 2] = points[0, 1]
        actions = build_replay_action_table(points, ([0, 1], [2, 3]))
        self.assertEqual(actions[0][0], ())
        self.assertEqual(
            actions[0][1],
            (("left", "xneg"), ("right", "zpos")),
        )
        self.assertEqual(actions[0][2], ())

    def test_single_interaction_point_uses_one_aggregate_control_pad(self):
        points = np.zeros((1, 2, 4, 3), dtype=np.float32)
        points[0, 1, :2, 0] = -0.2
        points[0, 1, 2:, 2] = 0.4
        actions = build_replay_action_table(points, ([0, 1, 2, 3],))
        self.assertEqual(
            actions[0][1],
            (("left", "xneg"), ("left", "zpos")),
        )
        filtered = build_replay_action_table(
            points,
            ([0, 1, 2, 3],),
            motion_epsilon=0.3,
        )
        self.assertEqual(filtered[0][1], ())

    def test_claim_conflict_and_input_auth(self):
        manager = SessionManager(num_sessions=2)
        app = create_app(manager, MjpegFrameStore())
        client = app.test_client()

        claim = client.post("/api/sessions/0/claim")
        self.assertEqual(claim.status_code, 200)
        claim_json = claim.get_json()
        token = claim_json["token"]
        self.assertEqual(claim_json["claim_id"], 1)

        conflict = client.post("/api/sessions/0/claim")
        self.assertEqual(conflict.status_code, 409)

        bad_input = client.post(
            "/api/sessions/0/input",
            json={"token": "wrong", "dx": 1, "dy": 0},
        )
        self.assertEqual(bad_input.status_code, 403)

        good_input = client.post(
            "/api/sessions/0/input",
            json={
                "token": token,
                "left": {"x": 1, "y": -1, "z": 1},
                "right": {"x": -1, "y": 1, "z": 0},
            },
        )
        self.assertEqual(good_input.status_code, 200)
        self.assertEqual(
            manager.snapshot_sessions()[0]["left"],
            (1.0, -1.0, 1.0),
        )
        self.assertEqual(
            manager.snapshot_sessions()[0]["right"],
            (-1.0, 1.0, 0.0),
        )

    def test_legacy_joystick_input_still_works(self):
        manager = SessionManager(num_sessions=1)
        app = create_app(manager, MjpegFrameStore())
        client = app.test_client()
        token = client.post("/api/sessions/0/claim").get_json()["token"]

        response = client.post(
            "/api/sessions/0/input",
            json={"token": token, "dx": 0.5, "dy": -0.25},
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(manager.snapshot_inputs(), {0: (-0.25, 0.5, 0.0)})

    def test_stream_requires_token(self):
        app = create_app(SessionManager(num_sessions=1), MjpegFrameStore())
        client = app.test_client()
        self.assertEqual(client.get("/stream/0.mjpg").status_code, 403)

    def test_heartbeat_release_and_timeout_routes(self):
        now = [100.0]
        manager = SessionManager(num_sessions=1, heartbeat_timeout_s=0.5)
        manager._now = lambda: now[0]
        app = create_app(manager, MjpegFrameStore())
        client = app.test_client()

        token = client.post("/api/sessions/0/claim").get_json()["token"]
        self.assertEqual(
            client.post("/api/sessions/0/heartbeat", json={"token": token}).status_code,
            200,
        )
        self.assertEqual(
            client.post("/api/sessions/0/release", json={"token": token}).status_code,
            200,
        )
        self.assertTrue(client.get("/api/sessions").get_json()["sessions"][0]["available"])

        client.post("/api/sessions/0/claim")
        now[0] += 0.6
        self.assertTrue(client.get("/api/sessions").get_json()["sessions"][0]["available"])

    def test_out_of_range_session_routes_return_not_found(self):
        app = create_app(SessionManager(num_sessions=1), MjpegFrameStore())
        client = app.test_client()
        for method, path in (
            (client.post, "/api/sessions/99/claim"),
            (client.post, "/api/sessions/99/heartbeat"),
            (client.post, "/api/sessions/99/release"),
            (client.post, "/api/sessions/99/input"),
            (client.get, "/stream/99.mjpg"),
        ):
            self.assertEqual(method(path).status_code, 404)

    def test_server_bind_failure_is_reported_before_runtime_starts(self):
        listener = socket.socket()
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        try:
            occupied_port = listener.getsockname()[1]
            app = create_app(SessionManager(num_sessions=1), MjpegFrameStore())
            with self.assertRaisesRegex(RuntimeError, "could not bind"):
                start_flask_server(app, "127.0.0.1", occupied_port)
        finally:
            listener.close()

    def test_demo2_asset_route_serves_original_control_assets(self):
        app = create_app(SessionManager(num_sessions=1), MjpegFrameStore())
        client = app.test_client()
        for filename in (
            "arrow_empty.png",
            "arrow_1.png",
            "arrow_2.png",
            "Picture1.png",
            "Picture2.png",
        ):
            response = client.get(f"/assets/{filename}")
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.mimetype, "image/png")
            response.close()
        self.assertEqual(client.get("/assets/not_allowed.png").status_code, 404)

    def test_manifest_route_supports_phone_standalone_mode(self):
        app = create_app(SessionManager(num_sessions=1), MjpegFrameStore())
        client = app.test_client()

        response = client.get("/manifest.webmanifest")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.mimetype, "application/manifest+json")
        manifest = json.loads(response.data)
        self.assertEqual(manifest["display"], "standalone")
        self.assertEqual(manifest["orientation"], "landscape")
        self.assertEqual(manifest["icons"][0]["src"], "/assets/Picture1.png")

    def test_phone_overlay_button_mapping_is_demo_calibrated(self):
        parser = ButtonParser()
        parser.feed(load_phone_client_html())

        expected = {
            ("left", "w"): ("x", "1"),
            ("left", "s"): ("x", "-1"),
            ("left", "a"): ("y", "-1"),
            ("left", "d"): ("y", "1"),
            ("left", "q"): ("z", "-1"),
            ("left", "e"): ("z", "1"),
            ("right", "i"): ("x", "1"),
            ("right", "k"): ("x", "-1"),
            ("right", "j"): ("y", "-1"),
            ("right", "l"): ("y", "1"),
            ("right", "u"): ("z", "-1"),
            ("right", "o"): ("z", "1"),
        }
        for key, (axis, direction) in expected.items():
            self.assertIn(key, parser.buttons)
            self.assertEqual(parser.buttons[key]["data-axis"], axis)
            self.assertEqual(parser.buttons[key]["data-dir"], direction)

    def test_phone_client_has_landscape_no_panel_control_hooks(self):
        html = load_phone_client_html()
        hand_panel_css = html.split(".hand-panel {", 1)[1].split("}", 1)[0]

        self.assertIn("controller-mode", html)
        self.assertNotIn('id="fullscreen"', html)
        self.assertNotIn("Full Screen", html)
        self.assertIn('id="release"', html)
        self.assertNotIn("requestControllerFullscreen", html)
        self.assertNotIn("requestFullscreen", html)
        self.assertNotIn("screen.orientation", html)
        self.assertNotIn("fullscreen-hint", html)
        self.assertIn("apple-mobile-web-app-capable", html)
        self.assertIn("/manifest.webmanifest", html)
        self.assertIn("--demo2-view-height", html)
        self.assertIn("--demo2-view-width", html)
        self.assertIn("window.visualViewport", html)
        self.assertIn("function updateViewportVars()", html)
        self.assertIn('window.visualViewport.addEventListener("resize", updateViewportVars)', html)
        self.assertIn("height: var(--demo2-view-height);", html)
        self.assertNotIn("Rotate phone sideways", html)
        self.assertNotIn("rotate-prompt", html)
        self.assertIn("wide-arrow", html)
        self.assertIn('/assets/arrow_empty.png', html)
        self.assertIn('/assets/arrow_1.png', html)
        self.assertIn('/assets/arrow_2.png', html)
        self.assertIn("@media (orientation: landscape)", html)
        self.assertIn("grid-template-rows: minmax(0, 1fr) auto;", html)
        self.assertIn("body.controller-mode #stream", html)
        self.assertIn("object-fit: cover;", html)
        self.assertIn("position: static;", html)
        self.assertIn("background: rgba(244, 246, 248, 0.96);", html)
        self.assertNotIn("--control-rail", html)
        self.assertIn(".hand-panel.left .control-button.active .wide-arrow", html)
        self.assertIn(".hand-panel.right .control-button.active .wide-arrow", html)
        self.assertEqual(html.count("gap: 5px;"), 1)
        self.assertNotIn("background", hand_panel_css)
        self.assertNotIn("border", hand_panel_css)
        self.assertNotIn("box-shadow", hand_panel_css)

    def test_default_phone_stream_and_control_offset(self):
        args = build_parser().parse_args([])
        self.assertEqual(args.case_name, "single_push_rope_4")
        self.assertEqual(args.phone_stream_size, (640, 480))
        self.assertEqual(args.phone_control_max_offset, 0.0)
        self.assertIsNone(args.demo2_runtime_fps)
        self.assertIsNone(args.demo2_replay_marker_session)
        self.assertEqual(args.demo2_replay_action_threshold, 0.0)
        self.assertIsNone(args.controller_pkl)
        self.assertFalse(hasattr(args, "base_path"))
        self.assertFalse(hasattr(args, "gaussian_path"))
        self.assertFalse(hasattr(args, "bg_img_path"))

    def test_demo2_phone_stream_resize_is_batched(self):
        trainer_source = Path(REPO_ROOT / "qqtt" / "engine" / "trainer_warp.py").read_text()
        publish_block = trainer_source.split(
            "if (\n                    publish_frame_fn is not None",
            1,
        )[1].split("next_stream_publish = now + stream_interval", 1)[0]
        self.assertIn("occupied_ids = torch.tensor", publish_block)
        self.assertIn("stream_tiles = F.interpolate", publish_block)
        self.assertIn("batch_frames[occupied_ids].permute(0, 3, 1, 2)", publish_block)
        self.assertIn("for idx, session_id in enumerate(occupied_sessions):", publish_block)
        self.assertNotIn("tile = batch_frames[session_id]", publish_block)
        self.assertNotIn("unsqueeze(0)", publish_block)


class Demo2CaseManifestTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.repo_root = Path(self.temp_dir.name)
        self.case_dir = self.repo_root / "assets" / "single_push_rope_4"
        self.case_dir.mkdir(parents=True)
        config_path = self.repo_root / "configs" / "real.yaml"
        config_path.parent.mkdir()
        config_path.write_text("FPS: 30\n", encoding="utf-8")

        self.manifest = {
            "best_model": "best_model.pth",
            "calibrate": "calibrate.pkl",
            "config": "configs/real.yaml",
            "final_data": "final_data.pkl",
            "gaussian_ply": "gaussian.ply",
            "metadata": "metadata.json",
            "optimal_params": "optimal_params.pkl",
            "controller_bank": "controller_bank.pkl",
            "background_image": "background.png",
        }
        for key, relative_path in self.manifest.items():
            if key == "config":
                continue
            (self.case_dir / relative_path).write_bytes(key.encode("utf-8"))
        self._write_manifest()

    def tearDown(self):
        self.temp_dir.cleanup()

    def _write_manifest(self):
        (self.case_dir / "manifest.json").write_text(
            json.dumps(self.manifest),
            encoding="utf-8",
        )

    def test_manifest_resolves_every_required_runtime_asset(self):
        assets = resolve_demo2_case_assets(
            self.repo_root,
            "single_push_rope_4",
        )

        self.assertEqual(set(REQUIRED_MANIFEST_KEYS), set(self.manifest))
        self.assertEqual(assets.config, self.repo_root / "configs" / "real.yaml")
        self.assertEqual(assets.gaussian_ply, self.case_dir / "gaussian.ply")
        self.assertEqual(assets.controller_bank, self.case_dir / "controller_bank.pkl")
        self.assertEqual(select_controller_bank(assets, None), assets.controller_bank)

    def test_manifest_rejects_missing_required_key(self):
        del self.manifest["controller_bank"]
        self._write_manifest()

        with self.assertRaisesRegex(KeyError, "controller_bank"):
            resolve_demo2_case_assets(self.repo_root, "single_push_rope_4")

    def test_manifest_reports_missing_packaged_file(self):
        (self.case_dir / "gaussian.ply").unlink()

        with self.assertRaisesRegex(FileNotFoundError, "gaussian_ply"):
            resolve_demo2_case_assets(self.repo_root, "single_push_rope_4")

    def test_controller_override_is_explicit_and_must_exist(self):
        assets = resolve_demo2_case_assets(self.repo_root, "single_push_rope_4")
        override_path = self.repo_root / "developer_controller.pkl"
        override_path.write_bytes(b"override")

        self.assertEqual(
            select_controller_bank(assets, override_path),
            override_path,
        )
        with self.assertRaisesRegex(FileNotFoundError, "--controller_pkl override"):
            select_controller_bank(assets, self.repo_root / "missing.pkl")

    def test_case_config_uses_only_manifest_paths(self):
        calibrate_path = self.case_dir / self.manifest["calibrate"]
        optimal_path = self.case_dir / self.manifest["optimal_params"]
        metadata_path = self.case_dir / self.manifest["metadata"]
        with calibrate_path.open("wb") as handle:
            pickle.dump([np.eye(4)], handle)
        with optimal_path.open("wb") as handle:
            pickle.dump({"global_spring_Y": 123.0}, handle)
        metadata_path.write_text(
            json.dumps({"intrinsics": [[1.0, 0.0], [0.0, 1.0]], "WH": [640, 480]}),
            encoding="utf-8",
        )

        class FakeConfig:
            def load_from_yaml(self, path):
                self.loaded_yaml = path

            def set_optimal_params(self, params):
                self.optimal_params = params

        class FakeLogger:
            def info(self, _message):
                pass

        assets = resolve_demo2_case_assets(self.repo_root, "single_push_rope_4")
        cfg = FakeConfig()
        metadata = load_demo2_case_config(assets, cfg, FakeLogger())

        self.assertEqual(cfg.loaded_yaml, str(assets.config))
        self.assertEqual(cfg.optimal_params, {"global_spring_Y": 123.0})
        np.testing.assert_array_equal(cfg.c2ws, np.asarray([np.eye(4)]))
        np.testing.assert_array_equal(cfg.w2cs, np.asarray([np.eye(4)]))
        np.testing.assert_array_equal(cfg.intrinsics, np.eye(2))
        self.assertEqual(cfg.WH, [640, 480])
        self.assertEqual(cfg.bg_img_path, str(assets.background_image))
        self.assertEqual(metadata["WH"], [640, 480])

    def test_case_name_cannot_escape_assets_directory(self):
        with self.assertRaisesRegex(ValueError, "Invalid Demo 2 case name"):
            resolve_demo2_case_assets(self.repo_root, "../single_push_rope_4")


if __name__ == "__main__":
    unittest.main()
