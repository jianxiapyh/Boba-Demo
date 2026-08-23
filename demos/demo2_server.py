import argparse
import json
import os
import pickle
import sys
import threading
from pathlib import Path

import torch
from flask import Flask, Response, abort, jsonify, request, send_from_directory
from werkzeug.serving import make_server

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
ENV_BIN = Path(sys.executable).resolve().parent
os.environ["PATH"] = f"{ENV_BIN}:{os.environ.get('PATH', '')}"

from demos.demo2.control import resolve_demo2_control_parts
from demos.demo2.case_assets import (
    load_demo2_case_config,
    resolve_demo2_case_assets,
    select_controller_bank,
)
from demos.demo2.qr_overlay import make_public_url, make_qr_rgb
from demos.demo2.replay_state import ReplayStateStore
from demos.demo2.session_manager import SessionManager
from demos.demo2.streaming import MjpegFrameStore


def parse_size(value):
    if isinstance(value, tuple):
        return value
    parts = str(value).lower().split("x")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(f"Expected WIDTHxHEIGHT, got {value!r}")
    width, height = int(parts[0]), int(parts[1])
    if width < 1 or height < 1:
        raise argparse.ArgumentTypeError("Width and height must be positive")
    return width, height


def build_parser():
    parser = argparse.ArgumentParser(description="Boba Demo 2 server")
    parser.add_argument("--case_name", type=str, default="single_push_rope_4")
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--batch_grid_cols", type=int, default=10)
    parser.add_argument("--batch_image_resolution", choices=("native", "640x480"), default="640x480")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--public_url", type=str, default=None)
    parser.add_argument(
        "--controller_pkl",
        type=str,
        default=None,
        help="Development override for the controller bank in the case manifest.",
    )
    parser.add_argument("--replay_start", type=int, default=0)
    parser.add_argument("--replay_end", type=int, default=None)
    parser.add_argument(
        "--demo2_runtime_fps",
        type=float,
        default=None,
        help=(
            "Simulation/replay frame rate. Defaults to the packaged case metadata "
            "FPS instead of running as fast as the renderer allows."
        ),
    )
    parser.add_argument("--phone_stream_size", type=parse_size, default=parse_size("640x480"))
    parser.add_argument("--phone_stream_fps", type=float, default=10.0)
    parser.add_argument("--heartbeat_timeout_s", type=float, default=10.0)
    parser.add_argument("--phone_control_step", type=float, default=0.005)
    parser.add_argument(
        "--phone_control_max_offset",
        type=float,
        default=0.0,
        help="Maximum accumulated manual offset; 0 disables clamping.",
    )
    parser.add_argument("--demo2_control_parts", choices=("auto", "1", "2"), default="auto")
    parser.add_argument(
        "--demo2_replay_marker_session",
        type=int,
        default=None,
        help="Show the real replay interaction-point marker on one unclaimed tile.",
    )
    parser.add_argument(
        "--demo2_replay_action_threshold",
        type=float,
        default=0.0,
        help=(
            "Minimum aggregate controller motion per runtime step before poster "
            "arrows activate; use a small positive value to suppress subpixel easing."
        ),
    )
    parser.add_argument("--demo2_double_control_cases", type=str, default="weird_package")
    parser.add_argument(
        "--demo2_debug_motion",
        action="store_true",
        help="Print and save first-cycle per-session target-motion diagnostics.",
    )
    parser.add_argument(
        "--demo2_debug_motion_path",
        type=str,
        default=None,
        help="Optional JSON path for --demo2_debug_motion output.",
    )
    parser.add_argument("--qr_size", type=int, default=220)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--max_frames", type=int, default=None, help="Optional smoke-test frame limit.")
    return parser


def load_phone_client_html():
    html_path = Path(__file__).resolve().parent / "demo2" / "phone_client.html"
    return html_path.read_text(encoding="utf-8")


def request_token():
    data = request.get_json(silent=True) or {}
    return data.get("token") or request.args.get("token") or ""


def create_app(session_manager, stream_store, replay_state_store=None):
    app = Flask(__name__)
    phone_html = load_phone_client_html()
    asset_dir = REPO_ROOT / "assets"
    allowed_assets = {
        "arrow_empty.png",
        "arrow_1.png",
        "arrow_2.png",
        "Picture1.png",
        "Picture2.png",
    }

    def require_valid_session(session_id):
        if session_id < 0 or session_id >= session_manager.num_sessions:
            abort(404)

    @app.get("/")
    def index():
        return Response(phone_html, mimetype="text/html")

    @app.get("/manifest.webmanifest")
    def manifest():
        return Response(
            json.dumps(
                {
                    "name": "Boba Demo 2",
                    "short_name": "Boba Demo",
                    "start_url": "/",
                    "scope": "/",
                    "display": "standalone",
                    "orientation": "landscape",
                    "background_color": "#050607",
                    "theme_color": "#050607",
                    "icons": [
                        {
                            "src": "/assets/Picture1.png",
                            "sizes": "any",
                            "type": "image/png",
                        }
                    ],
                }
            ),
            mimetype="application/manifest+json",
        )

    @app.get("/assets/<path:filename>")
    def demo2_asset(filename):
        if filename not in allowed_assets:
            abort(404)
        return send_from_directory(asset_dir, filename)

    @app.get("/api/sessions")
    def list_sessions():
        return jsonify({"sessions": session_manager.list_sessions(token=request.args.get("token"))})

    @app.get("/api/replay/<int:session_id>")
    def replay_state(session_id):
        require_valid_session(session_id)
        if replay_state_store is None:
            return jsonify({"error": "Replay state is unavailable"}), 503
        state = replay_state_store.get(session_id)
        if state is None:
            return jsonify({"error": "Replay state is not ready"}), 503
        return jsonify(state)

    @app.post("/api/sessions/<int:session_id>/claim")
    def claim_session(session_id):
        require_valid_session(session_id)
        claim = session_manager.claim(session_id)
        if claim is None:
            return jsonify({"error": "Session already occupied"}), 409
        return jsonify({"session_id": session_id, **claim})

    @app.post("/api/sessions/<int:session_id>/heartbeat")
    def heartbeat(session_id):
        require_valid_session(session_id)
        if not session_manager.heartbeat(session_id, request_token()):
            return jsonify({"error": "Invalid or expired session token"}), 403
        return jsonify({"ok": True})

    @app.post("/api/sessions/<int:session_id>/release")
    def release(session_id):
        require_valid_session(session_id)
        if not session_manager.release(session_id, request_token()):
            return jsonify({"error": "Invalid or expired session token"}), 403
        return jsonify({"ok": True})

    @app.post("/api/sessions/<int:session_id>/input")
    def update_input(session_id):
        require_valid_session(session_id)
        data = request.get_json(silent=True) or {}
        input_kwargs = {
            "x": data.get("x", 0.0),
            "y": data.get("y", 0.0),
            "z": data.get("z", 0.0),
        }
        if "left" in data or "right" in data:
            input_kwargs = {
                "left": data.get("left", {}),
                "right": data.get("right", {}),
            }
        if "dx" in data or "dy" in data:
            input_kwargs = {
                "dx": data.get("dx", 0.0),
                "dy": data.get("dy", 0.0),
            }
        if not session_manager.update_input(
            session_id,
            data.get("token") or request_token(),
            **input_kwargs,
        ):
            return jsonify({"error": "Invalid or expired session token"}), 403
        return jsonify({"ok": True})

    @app.get("/stream/<int:session_id>.mjpg")
    def stream(session_id):
        require_valid_session(session_id)
        token = request.args.get("token", "")
        if not session_manager.validate(session_id, token):
            abort(403)
        generator = stream_store.mjpeg_generator(
            session_id,
            stop_fn=lambda: not session_manager.validate(session_id, token),
        )
        return Response(
            generator,
            mimetype="multipart/x-mixed-replace; boundary=frame",
            headers={"Cache-Control": "no-cache"},
        )

    return app


def start_flask_server(app, host, port):
    try:
        server = make_server(host, int(port), app, threaded=True)
    except SystemExit as exc:
        raise RuntimeError(
            f"Demo 2 could not bind its phone server to {host}:{int(port)}"
        ) from exc
    thread = threading.Thread(
        target=server.serve_forever,
        name="demo2-flask",
        daemon=True,
    )
    thread.start()
    return server, thread


def validate_controller_bank_metadata(pkl_path, case_name, replay_start, replay_end):
    with open(pkl_path, "rb") as handle:
        root = pickle.load(handle)
    if not isinstance(root, dict):
        return root
    meta_case = root.get("case_name")
    if meta_case is not None and meta_case != case_name:
        raise ValueError(
            f"Controller bank was made for case {meta_case!r}, "
            f"but --case_name is {case_name!r}."
        )
    meta_start = root.get("replay_start")
    meta_end = root.get("replay_end")
    if meta_start is not None and int(meta_start) != int(replay_start):
        raise ValueError(
            f"Controller bank replay_start={meta_start}, "
            f"but --replay_start={replay_start}."
        )
    if replay_end is not None and meta_end is not None and int(meta_end) != int(replay_end):
        raise ValueError(
            f"Controller bank replay_end={meta_end}, "
            f"but --replay_end={replay_end}."
        )
    return root


def main():
    args = build_parser().parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch_size must be positive")
    if args.batch_grid_cols < 1:
        raise ValueError("--batch_grid_cols must be positive")
    if args.phone_control_step < 0:
        raise ValueError("--phone_control_step must be non-negative")
    if args.phone_control_max_offset < 0:
        raise ValueError("--phone_control_max_offset must be non-negative")
    if args.demo2_runtime_fps is not None and args.demo2_runtime_fps <= 0:
        raise ValueError("--demo2_runtime_fps must be positive")
    if args.demo2_replay_marker_session is not None and not (
        0 <= args.demo2_replay_marker_session < args.batch_size
    ):
        raise ValueError("--demo2_replay_marker_session must fit inside the batch")
    if args.demo2_replay_action_threshold < 0:
        raise ValueError("--demo2_replay_action_threshold must be non-negative")
    double_control_cases = [
        value.strip()
        for value in args.demo2_double_control_cases.split(",")
        if value.strip()
    ]
    resolved_control_parts = resolve_demo2_control_parts(
        args.case_name,
        requested=args.demo2_control_parts,
        double_control_cases=double_control_cases,
    )

    case_assets = resolve_demo2_case_assets(REPO_ROOT, args.case_name)
    controller_pkl = select_controller_bank(case_assets, args.controller_pkl)
    controller_root = validate_controller_bank_metadata(
        controller_pkl,
        args.case_name,
        args.replay_start,
        args.replay_end,
    )
    if (
        args.replay_end is None
        and isinstance(controller_root, dict)
        and controller_root.get("replay_end") is not None
    ):
        args.replay_end = int(controller_root["replay_end"])
        print(f"[Demo2] Using controller-bank replay_end={args.replay_end}")

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    try:
        torch.set_float32_matmul_precision("high")
    except AttributeError:
        pass

    from gaussian_splatting import dynamic_utils as dynamic_utils_backend
    from interactive_playground import create_gl_window, set_all_seeds

    print(
        "[Boba] dynamic_utils backend: "
        f"{dynamic_utils_backend.SELECTED_DYNAMIC_UTIL_VARIANT} "
        f"(device={dynamic_utils_backend.DETECTED_DEVICE_NAME}, "
        f"BOBA_DEVICE={dynamic_utils_backend.BOBA_DEVICE})"
    )

    public_url = make_public_url(args.host, args.port, public_url=args.public_url)
    qr_overlay = make_qr_rgb(public_url, size=args.qr_size)
    print(f"[Demo2] Phone URL: {public_url}")
    print(f"[Demo2] Control parts: {resolved_control_parts}")

    session_manager = SessionManager(
        num_sessions=args.batch_size,
        heartbeat_timeout_s=args.heartbeat_timeout_s,
    )
    stream_store = MjpegFrameStore(jpeg_quality=80)
    replay_state_store = ReplayStateStore(
        args.batch_size,
        control_parts=resolved_control_parts,
    )
    app = create_app(session_manager, stream_store, replay_state_store)
    start_flask_server(app, args.host, args.port)

    output_dir = args.output_dir or os.path.join("results", "demo2", args.case_name)
    os.makedirs(output_dir, exist_ok=True)
    motion_debug_path = args.demo2_debug_motion_path
    if args.demo2_debug_motion and motion_debug_path is None:
        motion_debug_path = os.path.join(output_dir, "motion_debug.json")

    window = create_gl_window(640, 480, use_screen_resolution=True)
    _ = torch.empty(1, device="cuda")
    set_all_seeds(42)

    import pycuda.driver as cuda_driver

    cuda_driver.init()
    cuda_ctx = cuda_driver.Context.attach()

    try:
        from qqtt import InvPhyTrainerWarp
        from qqtt.utils import cfg, logger

        case_metadata = load_demo2_case_config(case_assets, cfg, logger)
        demo2_runtime_fps = (
            float(args.demo2_runtime_fps)
            if args.demo2_runtime_fps is not None
            else float(case_metadata.get("fps", 30.0))
        )
        if demo2_runtime_fps <= 0:
            raise ValueError("Demo 2 runtime FPS resolved to a non-positive value")
        replay_state_store.set_runtime_fps(demo2_runtime_fps)
        print(f"[Demo2] Runtime FPS: {demo2_runtime_fps:g}")

        logger.set_log_file(path=output_dir, name="demo2_server")
        trainer = InvPhyTrainerWarp(
            data_path=str(case_assets.final_data),
            base_dir=output_dir,
        )
        controller_points_group = trainer.load_controller_points_group_pkl(
            controller_pkl,
            device=cfg.device,
        )
        if controller_points_group.shape[0] < args.batch_size:
            raise ValueError(
                f"Controller bank contains {controller_points_group.shape[0]} trajectories, "
                f"but Demo 2 needs {args.batch_size}."
            )

        trainer.run_batched_demo2_runtime(
            model_path=str(case_assets.best_model),
            gs_path=str(case_assets.gaussian_ply),
            window=window,
            cuda_ctx=cuda_ctx,
            controller_points_group=controller_points_group,
            batch_size=args.batch_size,
            batch_grid_cols=args.batch_grid_cols,
            batch_image_resolution=args.batch_image_resolution,
            replay_start=args.replay_start,
            replay_end=args.replay_end,
            session_snapshot_fn=session_manager.snapshot_sessions,
            publish_frame_fn=stream_store.publish_rgb,
            publish_replay_state_fn=replay_state_store.publish,
            qr_overlay_rgb=qr_overlay,
            phone_stream_size=args.phone_stream_size,
            phone_stream_fps=args.phone_stream_fps,
            demo2_runtime_fps=demo2_runtime_fps,
            phone_control_step=args.phone_control_step,
            phone_control_max_offset=args.phone_control_max_offset,
            demo2_control_parts=resolved_control_parts,
            demo2_replay_marker_session=args.demo2_replay_marker_session,
            demo2_replay_action_threshold=args.demo2_replay_action_threshold,
            demo2_debug_motion=args.demo2_debug_motion,
            demo2_debug_motion_path=motion_debug_path,
            max_frames=args.max_frames,
        )
    finally:
        import glfw

        try:
            cuda_ctx.pop()
        except Exception:
            pass
        try:
            glfw.make_context_current(window)
        except Exception:
            pass
        try:
            glfw.destroy_window(window)
        finally:
            glfw.terminate()


if __name__ == "__main__":
    main()
