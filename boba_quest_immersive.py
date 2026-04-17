#!/usr/bin/env python3

from __future__ import annotations

import glob
import json
import os
import pickle
import random
import shutil
import sys
from argparse import ArgumentParser
from pathlib import Path

np = None
torch = None
REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_SCENE_ASSETS_ROOT = REPO_ROOT / "assets" / "scenes"
PUBLIC_DEMO_CASES = ("sloth", "rope", "hq_rope")
COMPAT_DEMO_CASE_ALIASES = {
    "hq_rope_0": "hq_rope",
}
SUPPORTED_DEMO_CASE_ARGUMENTS = PUBLIC_DEMO_CASES + tuple(COMPAT_DEMO_CASE_ALIASES.keys())
DEMO_CASE_WORLD_SCALE = {
    "hq_rope": 0.3932700391790796,
}
DEMO_CASE_LENGTH_LIKE_CFG_KEYS = (
    "object_radius",
    "controller_radius",
    "collision_dist",
)


def canonical_demo_case_name(case_name: str) -> str:
    case_key = str(case_name).strip().lower()
    return str(COMPAT_DEMO_CASE_ALIASES.get(case_key, case_key))


def resolve_demo_case_manifest(case_name: str) -> tuple[str, Path, dict]:
    canonical_case = canonical_demo_case_name(case_name)
    manifest_path = REPO_ROOT / "assets" / canonical_case / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Immersive demo assets for case '{case_name}' were not found at {manifest_path}. "
            "This branch expects self-contained demo runtime assets under ./assets/<case>/. "
            f"Public packaged cases in this branch are: {', '.join(PUBLIC_DEMO_CASES)}."
        )
    with open(manifest_path, "r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    return canonical_case, manifest_path.parent, manifest


def manifest_file_path(manifest_dir: Path, manifest: dict, key: str) -> str:
    relative_path = manifest.get(key)
    if relative_path is None:
        raise KeyError(f"Manifest is missing required key: {key}")
    return str((manifest_dir / relative_path).resolve())


def demo_case_world_scale(case_name: str) -> float:
    case_key = canonical_demo_case_name(case_name)
    return float(DEMO_CASE_WORLD_SCALE.get(case_key, 1.0))


def apply_demo_case_world_scale_to_cfg(cfg, case_name: str) -> float:
    scale = demo_case_world_scale(case_name)
    cfg.demo_case_world_scale = scale
    if abs(scale - 1.0) <= 1e-8:
        return scale
    for attr_name in DEMO_CASE_LENGTH_LIKE_CFG_KEYS:
        if not hasattr(cfg, attr_name):
            continue
        setattr(cfg, attr_name, float(getattr(cfg, attr_name)) * scale)
    return scale


def set_all_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_gl_window(width: int, height: int, visible: bool = True):
    import glfw
    from OpenGL import GL as gl

    assert glfw.init(), "GLFW init failed"
    glfw.window_hint(glfw.CLIENT_API, glfw.OPENGL_API)
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 4)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 6)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
    glfw.window_hint(glfw.VISIBLE, glfw.TRUE if visible else glfw.FALSE)

    window = glfw.create_window(width, height, "Boba Quest Immersive", None, None)
    assert window, "create_window failed (need X11 desktop GL)"

    glfw.make_context_current(window)
    _ = gl.glGetString(gl.GL_VERSION)
    glfw.swap_interval(0)
    return window


def configure_local_python_paths():
    repo_root = os.path.dirname(os.path.abspath(__file__))
    submodule_roots = (
        (
            os.path.join(repo_root, "gaussian_splatting", "submodules", "simple-knn"),
            ("simple_knn/_C*.so",),
        ),
        (
            os.path.join(
                repo_root,
                "gaussian_splatting",
                "submodules",
                "diff-gaussian-rasterization",
            ),
            ("diff_gaussian_rasterization/_C*.so",),
        ),
        (
            os.path.join(repo_root, "gaussian_splatting", "submodules", "fused-ssim"),
            ("fused_ssim_cuda*.so",),
        ),
    )
    for path, required_globs in reversed(submodule_roots):
        if not os.path.isdir(path):
            continue

        has_native_build = any(glob.glob(os.path.join(path, pattern)) for pattern in required_globs)
        if not has_native_build:
            print(
                f"[python_path] skipping local submodule without built extension: {path}",
                flush=True,
            )
            continue

        if path not in sys.path:
            sys.path.insert(0, path)


def prefer_system_ninja_binary():
    current_ninja = shutil.which("ninja")
    system_ninja = Path("/usr/bin/ninja")
    if current_ninja is None or not system_ninja.exists():
        return
    current_path = Path(current_ninja)
    try:
        if current_path.resolve() == system_ninja.resolve():
            return
    except OSError:
        return
    try:
        launcher_text = current_path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return
    if "from ninja import ninja" not in launcher_text:
        return
    existing_path = os.environ.get("PATH", "")
    os.environ["PATH"] = f"/usr/bin:/bin:{existing_path}"
    print(
        f"[startup] preferring system ninja binary over Python wrapper: {current_path} -> {system_ninja}",
        flush=True,
    )


def attach_pycuda_context_for_current_torch_device():
    import pycuda.driver as cuda_driver

    cuda_driver.init()
    cuda_device_index = int(torch.cuda.current_device())
    cuda_device = cuda_driver.Device(cuda_device_index)
    attach_error = None

    try:
        ctx = cuda_device.retain_primary_context()
        ctx.push()
        print(
            f"[quest_display] cuda_interop_context=primary device=cuda:{cuda_device_index}",
            flush=True,
        )
        return ctx
    except Exception as exc:
        attach_error = exc

    try:
        ctx = cuda_driver.Context.attach()
        print(
            "[quest_display] cuda_interop_context=attached "
            f"device=cuda:{cuda_device_index} "
            f"primary_error={type(attach_error).__name__}: {attach_error}",
            flush=True,
        )
        return ctx
    except Exception as exc:
        raise RuntimeError(
            "Unable to create a PyCUDA context for Quest immersive rendering: "
            f"primary_context_error={type(attach_error).__name__}: {attach_error}; "
            f"attach_error={type(exc).__name__}: {exc}"
        ) from exc


def prioritize_conda_bin():
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if not conda_prefix:
        return

    conda_bin = os.path.join(conda_prefix, "bin")
    if not os.path.isdir(conda_bin):
        return

    path_parts = [part for part in os.environ.get("PATH", "").split(os.pathsep) if part]
    path_parts = [part for part in path_parts if part != conda_bin]
    os.environ["PATH"] = os.pathsep.join([conda_bin] + path_parts)


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(
        description=(
            "Run the shipped Boba Quest immersive demo. "
            "This launcher is fixed to live OpenXR controllers, immersive Quest display, "
            "the ILLIXR_lab scene, and the balanced immersive preset. "
            "Runtime demo assets are resolved from ./assets/."
        )
    )
    parser.add_argument(
        "--case_name",
        type=str,
        choices=SUPPORTED_DEMO_CASE_ARGUMENTS,
        default="sloth",
        help="public packaged demo case",
    )
    parser.add_argument("--n_dup", type=int, default=0, help="must remain 0 for the shipped Quest demo")
    parser.add_argument(
        "--scene_assets_root",
        type=str,
        default=str(DEFAULT_SCENE_ASSETS_ROOT),
        help="root directory for immersive scene assets",
    )
    parser.add_argument(
        "--interactive_window_mode",
        choices=("visible", "hidden"),
        default="hidden",
        help="show or hide the local OpenGL window while Quest output stays active",
    )
    parser.add_argument(
        "--render_profile",
        action="store_true",
        help="write render_profile_summary.txt and render_profile_frames.csv",
    )
    parser.add_argument(
        "--render_profile_every",
        type=int,
        default=30,
        help="print one detailed render profile line every N profiled frames",
    )
    parser.add_argument(
        "--immersive_timewarp",
        choices=("off", "scene_depth_reproject"),
        default="off",
        help=(
            "late-warp mode for immersive Quest output: "
            "'off' keeps the current shipped path, "
            "'scene_depth_reproject' late-warps the fully composed scene via depth reprojection"
        ),
    )
    parser.add_argument(
        "--immersive_static_scene_overlap",
        choices=("off", "on"),
        default="off",
        help=(
            "static scene overlap mode for immersive Quest output: "
            "'off' keeps the serial reference path, "
            "'on' overlaps balanced room/table rendering with simulation+LBS without enabling time warp"
        ),
    )
    parser.add_argument(
        "--immersive_framegen",
        choices=("off", "static", "adaptive"),
        default="off",
        help=(
            "static-scene frame generation mode for immersive Quest output: "
            "'off' renders the static room/table every frame, "
            "'static' reuses the last static scene every other frame, "
            "'adaptive' reuses only when head motion stays within conservative guardrails"
        ),
    )
    parser.add_argument(
        "--immersive_gaussian_render",
        choices=("serial", "stereo_parallel"),
        default="serial",
        help=(
            "Gaussian render scheduling mode for immersive Quest output: "
            "'serial' renders left/right Gaussian eyes sequentially, "
            "'stereo_parallel' experimentally renders both eyes concurrently on separate CUDA streams "
            "(currently only supported with overlap=off and timewarp=off)"
        ),
    )
    return parser


def main(argv: list[str] | None = None):
    parser = build_parser()
    args = parser.parse_args(argv)

    global np, torch
    import numpy as np  # type: ignore[assignment]
    import torch  # type: ignore[assignment]

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    try:
        torch.set_float32_matmul_precision("high")
    except AttributeError:
        pass

    if args.n_dup != 0:
        raise ValueError("The shipped Quest immersive launcher supports only --n_dup 0.")

    print("[quest_display] input_source=live_openxr_controller", flush=True)
    print("[quest_display] controller_mode=multi_points", flush=True)
    print("[quest_display] mode=immersive", flush=True)
    print("[quest_display] scene_preset=ILLIXR_lab", flush=True)
    print("[quest_display] immersive_render_preset=balanced", flush=True)
    print(f"[quest_display] immersive_timewarp={args.immersive_timewarp}", flush=True)
    print(
        "[quest_display] immersive_static_scene_overlap="
        f"{args.immersive_static_scene_overlap}",
        flush=True,
    )
    print(
        f"[quest_display] immersive_framegen={args.immersive_framegen}",
        flush=True,
    )
    print(
        f"[quest_display] immersive_gaussian_render={args.immersive_gaussian_render}",
        flush=True,
    )
    print(
        f"[quest_display] interactive_window_mode={args.interactive_window_mode}",
        flush=True,
    )

    window = create_gl_window(
        848,
        400,
        visible=(args.interactive_window_mode == "visible"),
    )
    if args.interactive_window_mode == "hidden":
        print(
            "[interactive_window] running with hidden local window; "
            "rendering stays active for Quest/offscreen output",
            flush=True,
        )

    _ = torch.empty(1, device="cuda")
    set_all_seeds(42)

    ctx = attach_pycuda_context_for_current_torch_device()

    prioritize_conda_bin()
    prefer_system_ninja_binary()
    configure_local_python_paths()
    from qqtt import InvPhyTrainerWarp
    from qqtt.utils import logger, cfg

    case_name = args.case_name
    canonical_case_name, manifest_dir, case_manifest = resolve_demo_case_manifest(case_name)
    if canonical_case_name != case_name:
        print(
            "[quest_display] demo case alias resolved: "
            f"requested={case_name} canonical={canonical_case_name}",
            flush=True,
        )

    cfg.load_from_yaml(case_manifest.get("config", "configs/real.yaml"))
    cfg.demo_case_name = canonical_case_name

    base_dir = f"./temp_experiments/{canonical_case_name}"

    optimal_path = manifest_file_path(manifest_dir, case_manifest, "optimal_params")
    with open(optimal_path, "rb") as f:
        optimal_params = pickle.load(f)
    cfg.set_optimal_params(optimal_params)
    demo_case_scale = apply_demo_case_world_scale_to_cfg(cfg, canonical_case_name)
    if abs(demo_case_scale - 1.0) > 1e-8:
        print(
            "[quest_display] demo case world scale: "
            f"case={canonical_case_name} scale={demo_case_scale:.8f} "
            f"object_radius={float(cfg.object_radius):.8f} "
            f"controller_radius={float(cfg.controller_radius):.8f} "
            f"collision_dist={float(cfg.collision_dist):.8f}",
            flush=True,
        )

    with open(manifest_file_path(manifest_dir, case_manifest, "calibrate"), "rb") as f:
        c2ws = pickle.load(f)
    w2cs = [np.linalg.inv(c2w) for c2w in c2ws]
    cfg.c2ws = np.array(c2ws)
    cfg.w2cs = np.array(w2cs)

    with open(
        manifest_file_path(manifest_dir, case_manifest, "metadata"),
        "r",
        encoding="utf-8",
    ) as f:
        data = json.load(f)
    cfg.intrinsics = np.array(data["intrinsics"])
    cfg.WH = data["WH"]
    gaussians_path = manifest_file_path(manifest_dir, case_manifest, "gaussian_ply")

    logger.set_log_file(path=base_dir, name="inference_log")

    trainer = InvPhyTrainerWarp(
        data_path=manifest_file_path(manifest_dir, case_manifest, "final_data"),
        base_dir=base_dir,
    )

    best_model_path = manifest_file_path(manifest_dir, case_manifest, "best_model")
    output_dir = os.path.join("./gaussian_output_dynamic", canonical_case_name)

    try:
        trainer.interactive_playground_quest_immersive_balanced(
            best_model_path,
            gaussians_path,
            output_dir=output_dir,
            n_dup=args.n_dup,
            window=window,
            cuda_ctx=ctx,
            interactive_window_mode=args.interactive_window_mode,
            scene_assets_root=args.scene_assets_root,
            render_profile=args.render_profile,
            render_profile_every=args.render_profile_every,
            immersive_timewarp=args.immersive_timewarp,
            immersive_static_scene_overlap=args.immersive_static_scene_overlap,
            immersive_framegen=args.immersive_framegen,
            immersive_gaussian_render=args.immersive_gaussian_render,
        )
    finally:
        import glfw

        try:
            glfw.make_context_current(window)
        except Exception:
            pass
        try:
            glfw.destroy_window(window)
            glfw.terminate()
        except Exception:
            pass

        try:
            ctx.detach()
        except Exception:
            pass


if __name__ == "__main__":
    main()
