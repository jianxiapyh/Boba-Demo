#!/usr/bin/env python3

from __future__ import annotations

import glob
import json
import os
import pickle
import random
import shutil
import subprocess
import sys
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from pathlib import Path

from tools.fetch_demo_case_assets import DemoAssetValidationError, validate_demo_case_assets

np = None
torch = None
REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_SCENE_ASSETS_ROOT = REPO_ROOT / "assets" / "scenes"
PUBLIC_DEMO_CASES = ("sloth", "rope", "hq_rope", "rope_game", "hq_rope_game")
COMPAT_DEMO_CASE_ALIASES = {
    "hq_rope_0": "hq_rope",
}
SUPPORTED_DEMO_CASE_ARGUMENTS = PUBLIC_DEMO_CASES + tuple(COMPAT_DEMO_CASE_ALIASES.keys())
SHARED_TUTORIAL_SLIDES = (
    "controls_overview.png",
    "interaction_tips.png",
)
DEMO_CASE_WORLD_SCALE = {
    "hq_rope": 0.3932700391790796,
}
DEMO_CASE_LENGTH_LIKE_CFG_KEYS = (
    "object_radius",
    "controller_radius",
    "collision_dist",
)
DEMO_CASE_PHYSICS_PROFILES = {
    "hq_rope_game": {
        "dt": 5e-5,
        "num_substeps": 667,
        "self_collision": True,
    },
}
RUNTIME_ENV_READY_SENTINEL = "BOBA_IMMERSIVE_RUNTIME_READY"
DEFAULT_CUDA_HOME = "/usr/local/cuda"
DEFAULT_GSPLAT_SOURCE_ROOT = (
    REPO_ROOT.parent / "Boba" / "gaussian_splatting" / "submodules" / "gsplat"
)
BRIDGE_DEPS_CHECK_SCRIPT = (
    REPO_ROOT / "linux_pose_probe" / "check_boba_immersive_bridge_deps.sh"
)
CUDA_STARTUP_DEBUG_ENV = "BOBA_CUDA_STARTUP_DEBUG"


class StartupConfigurationError(RuntimeError):
    pass


def detected_conda_prefix() -> str | None:
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix and Path(conda_prefix).is_dir():
        return conda_prefix

    inferred_prefix = Path(sys.prefix)
    if (inferred_prefix / "conda-meta").is_dir():
        return str(inferred_prefix)
    return None


def prepend_env_entries(current_value: str | None, leading_entries: list[str]) -> str:
    cleaned_leading_entries = [entry for entry in leading_entries if entry]
    existing_entries = [entry for entry in (current_value or "").split(os.pathsep) if entry]
    filtered_entries = [entry for entry in existing_entries if entry not in cleaned_leading_entries]
    return os.pathsep.join(cleaned_leading_entries + filtered_entries)


def ensure_direct_launch_runtime_env(argv: list[str] | None = None) -> None:
    current_env = os.environ.copy()
    launch_env = current_env.copy()
    launch_env["PYTHONNOUSERSITE"] = "1"

    cuda_home = str(Path(current_env.get("CUDA_HOME") or DEFAULT_CUDA_HOME))
    launch_env["CUDA_HOME"] = cuda_home
    launch_env["PATH"] = prepend_env_entries(
        current_env.get("PATH"),
        [str(Path(cuda_home) / "bin")],
    )

    conda_prefix = detected_conda_prefix()
    ld_library_leading_entries = []
    if conda_prefix:
        ld_library_leading_entries.append(str(Path(conda_prefix) / "lib"))
    ld_library_leading_entries.append(str(Path(cuda_home) / "lib64"))
    launch_env["LD_LIBRARY_PATH"] = prepend_env_entries(
        current_env.get("LD_LIBRARY_PATH"),
        ld_library_leading_entries,
    )

    launch_env.setdefault("BOBA_GSPLAT_SOURCE_ROOT", str(DEFAULT_GSPLAT_SOURCE_ROOT))

    reexec_keys = ("PYTHONNOUSERSITE", "LD_LIBRARY_PATH")
    needs_reexec = any(current_env.get(key) != launch_env.get(key) for key in reexec_keys)
    if not needs_reexec:
        for key in ("CUDA_HOME", "PATH", "BOBA_GSPLAT_SOURCE_ROOT"):
            os.environ[key] = launch_env[key]
        return

    if current_env.get(RUNTIME_ENV_READY_SENTINEL) == "1":
        return

    launch_env[RUNTIME_ENV_READY_SENTINEL] = "1"
    print(
        "[startup] re-executing with conda/CUDA runtime libraries for RTX6000 compatibility",
        flush=True,
    )
    exec_argv = [sys.executable, str(Path(__file__).resolve())]
    exec_argv.extend(list(argv) if argv is not None else sys.argv[1:])
    os.execvpe(sys.executable, exec_argv, launch_env)


def ensure_immersive_bridge_system_deps() -> None:
    clean_env = os.environ.copy()
    clean_env.pop("LD_LIBRARY_PATH", None)
    result = subprocess.run(
        ["bash", str(BRIDGE_DEPS_CHECK_SCRIPT)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
        env=clean_env,
    )
    if result.returncode == 0:
        return

    detail = result.stderr.strip() or result.stdout.strip() or (
        "Immersive bridge dependency preflight failed."
    )
    raise StartupConfigurationError(detail)


def env_flag_enabled(name: str) -> bool:
    value = str(os.environ.get(name, "")).strip().lower()
    return value not in {"", "0", "false", "off", "no"}


def print_cuda_startup_debug_banner(torch_module) -> None:
    if not env_flag_enabled(CUDA_STARTUP_DEBUG_ENV):
        return

    if torch_module.cuda.is_available():
        try:
            current_device = int(torch_module.cuda.current_device())
            current_device_name = torch_module.cuda.get_device_name(current_device)
            capability = torch_module.cuda.get_device_capability(current_device)
        except Exception as exc:
            current_device = None
            current_device_name = f"unavailable ({type(exc).__name__}: {exc})"
            capability = "unavailable"
    else:
        current_device = None
        current_device_name = "cuda_unavailable"
        capability = "unavailable"

    print(
        "[quest_display] startup cuda debug: "
        f"python={sys.executable} "
        f"conda_prefix={os.environ.get('CONDA_PREFIX', '<unset>')} "
        f"ld_library_path={os.environ.get('LD_LIBRARY_PATH', '<unset>')} "
        f"torch={getattr(torch_module, '__version__', '<unknown>')} "
        f"torch_cuda={getattr(torch_module.version, 'cuda', '<unknown>')} "
        f"current_device={current_device} "
        f"current_device_name={current_device_name} "
        f"capability={capability}",
        flush=True,
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


def resolve_demo_case_tutorial_slides(manifest_dir: Path, manifest: dict, case_name: str) -> list[str]:
    tutorial_dir = REPO_ROOT / "assets" / "tutorial"
    slide_paths = [tutorial_dir / slide_name for slide_name in SHARED_TUTORIAL_SLIDES]
    extra_slides = manifest.get("tutorial_extra_slides", [])
    if extra_slides is None:
        extra_slides = []
    if not isinstance(extra_slides, list):
        raise TypeError(
            f"Manifest key 'tutorial_extra_slides' for case '{case_name}' must be a list."
        )
    for relative_path in extra_slides:
        slide_paths.append((manifest_dir / str(relative_path)).resolve())

    missing_paths = [str(path) for path in slide_paths if not Path(path).exists()]
    if missing_paths:
        raise FileNotFoundError(
            "Immersive tutorial slide assets are missing for "
            f"case '{case_name}': " + ", ".join(missing_paths)
        )
    return [str(Path(path).resolve()) for path in slide_paths]


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


def apply_demo_case_physics_profile_to_cfg(cfg, case_name: str) -> dict | None:
    case_key = canonical_demo_case_name(case_name)
    profile = DEMO_CASE_PHYSICS_PROFILES.get(case_key)
    if profile is None:
        return None
    previous = {}
    if "dt" in profile:
        previous["dt"] = float(getattr(cfg, "dt"))
        cfg.dt = float(profile["dt"])
    if "num_substeps" in profile:
        previous["num_substeps"] = int(getattr(cfg, "num_substeps"))
        cfg.num_substeps = int(profile["num_substeps"])
    if "self_collision" in profile:
        previous["self_collision"] = bool(getattr(cfg, "self_collision", False))
        cfg.self_collision = bool(profile["self_collision"])
    return previous


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


def prioritize_conda_runtime_libs():
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if not conda_prefix:
        return

    conda_lib = os.path.join(conda_prefix, "lib")
    if not os.path.isdir(conda_lib):
        return

    lib_parts = [part for part in os.environ.get("LD_LIBRARY_PATH", "").split(os.pathsep) if part]
    lib_parts = [part for part in lib_parts if part != conda_lib]
    os.environ["LD_LIBRARY_PATH"] = os.pathsep.join([conda_lib] + lib_parts)


def configure_immersive_viewer_upload_runtime(args) -> None:
    upload_mode = str(args.immersive_viewer_upload_mode).strip().lower()
    upload_thread = str(args.immersive_viewer_upload_thread).strip().lower()
    late_wait_us = int(args.immersive_viewer_upload_late_wait_us)
    ring_slots = int(args.immersive_viewer_upload_ring_slots)
    busy_backoff_us = int(args.immersive_viewer_upload_busy_backoff_us)
    if late_wait_us < 0:
        raise ValueError("--immersive_viewer_upload_late_wait_us must be >= 0.")
    if ring_slots < 3 or ring_slots > 8:
        raise ValueError("--immersive_viewer_upload_ring_slots must be between 3 and 8.")
    if busy_backoff_us < 0:
        raise ValueError("--immersive_viewer_upload_busy_backoff_us must be >= 0.")

    # The C++ OpenXR bridge consumes these as process environment settings.
    # Keep the user-facing control as launcher flags so runs are reproducible
    # and not affected by stale shell exports.
    os.environ["BOBA_IMMERSIVE_VIEWER_UPLOAD_MODE"] = upload_mode
    os.environ["BOBA_IMMERSIVE_VIEWER_UPLOAD_THREAD"] = upload_thread
    os.environ["BOBA_IMMERSIVE_VIEWER_UPLOAD_LATE_WAIT_US"] = str(late_wait_us)
    os.environ["BOBA_IMMERSIVE_VIEWER_UPLOAD_RING_SLOTS"] = str(ring_slots)
    os.environ["BOBA_IMMERSIVE_VIEWER_UPLOAD_BUSY_BACKOFF_US"] = str(busy_backoff_us)


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(
        formatter_class=ArgumentDefaultsHelpFormatter,
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
        "--profile",
        action="store_true",
        help=(
            "enable detailed immersive diagnostics and write "
            "render_profile_summary.txt and render_profile_frames.csv"
        ),
    )
    parser.add_argument(
        "--profile_freq",
        type=int,
        default=30,
        help="print one detailed profile line every N profiled frames",
    )
    parser.add_argument(
        "--immersive_timewarp",
        choices=("off", "scene_depth_reproject"),
        default="off",
        help=(
            "late-warp mode for immersive Quest output: "
            "'off' keeps the current shipped path, "
            "'scene_depth_reproject' late-warps the fully composed scene via depth reprojection "
            "(required for --immersive_framegen static|adaptive)"
        ),
    )
    parser.add_argument(
        "--immersive_static_scene_overlap",
        choices=("off", "on"),
        default="on",
        help=(
            "static scene overlap mode for immersive Quest output: "
            "'off' keeps the serial reference path, "
            "'on' overlaps static room rendering with simulation+LBS. "
            "For native_gl this uses the full_scene_per_eye worker path; "
            "legacy pyrender/gpu use the balanced room/table worker path"
        ),
    )
    parser.add_argument(
        "--immersive_static_scene_reuse",
        choices=("off", "static", "adaptive"),
        default="off",
        help=(
            "real-frame static-scene reuse mode for immersive Quest output: "
            "'off' always renders a fresh static scene, "
            "'static' reuses cached static-scene outputs up to a fixed age limit, "
            "'adaptive' adds motion/head-reset guardrails before reuse"
        ),
    )
    parser.add_argument(
        "--immersive_static_scene_backend",
        choices=("pyrender", "gpu", "native_gl"),
        default="pyrender",
        help=(
            "backend for balanced static-scene rendering: "
            "'pyrender' keeps the shipped OpenGL/PyRender path, "
            "'gpu' enables the GPU-first hybrid balanced backend, using "
            "CUDA-native layers where they are parity-proven and faster while "
            "falling back to pyrender internally when needed "
            "(v1 supports balanced_support_focus only, with --immersive_timewarp off "
            "and --immersive_framegen off), "
            "'native_gl' renders the full room with the native OpenGL path and "
            "CUDA readback (v1 requires true-stereo serial mode with overlap/timewarp/framegen off)"
        ),
    )
    parser.add_argument(
        "--immersive_native_gl_texture_mode",
        choices=("stable", "stable_mipmap", "legacy"),
        default="stable_mipmap",
        help=(
            "native GL texture sampling mode: "
            "'stable' uses clamp-to-edge, non-mipmapped linear sampling; "
            "'stable_mipmap' uses clamp-to-edge with mipmapped linear sampling "
            "and optional anisotropic filtering; "
            "'legacy' keeps repeat wrapping with mipmapped linear sampling"
        ),
    )
    parser.add_argument(
        "--immersive_native_gl_anisotropy",
        type=int,
        choices=(1, 2, 4, 8, 16),
        default=8,
        help=(
            "requested native GL anisotropic texture filtering level for "
            "--immersive_native_gl_texture_mode stable_mipmap. "
            "Ignored by stable and legacy modes"
        ),
    )
    parser.add_argument(
        "--immersive_native_gl_msaa_samples",
        type=int,
        choices=(1, 2, 4),
        default=4,
        help=(
            "native GL MSAA sample count for the static scene FBO. "
            "Use 1 to disable MSAA, or 2/4 for quality-first edge stability"
        ),
    )
    parser.add_argument(
        "--immersive_native_gl_depth_format",
        choices=("depth24", "depth32f"),
        default="depth32f",
        help=(
            "native GL depth renderbuffer format. "
            "'depth32f' improves depth precision near occlusion boundaries; "
            "'depth24' keeps the legacy format"
        ),
    )
    parser.add_argument(
        "--immersive_gaussian_source_validation",
        choices=("off", "on"),
        default="off",
        help=(
            "expensive Gaussian source corruption validator for rope-family "
            "immersive runs. Keep off for normal native-GL gameplay; use on for "
            "debugging source-coverage rollback behavior"
        ),
    )
    parser.add_argument(
        "--immersive_eye_resolution",
        type=int,
        default=1408,
        help=(
            "square per-eye immersive output resolution. "
            "1408 is the default native GL quality/performance preset; "
            "use 1024 for speed or 1536/2048 for quality experiments"
        ),
    )
    parser.add_argument(
        "--immersive_static_scene_mode",
        choices=("balanced_focus", "balanced_support_focus"),
        default=None,
        help=(
            "legacy pyrender/gpu static-scene mode for immersive Quest output. "
            "Omit this flag for native_gl, which always uses full_scene_per_eye. "
            "'balanced_focus' uses the fast table-special ROI path, "
            "'balanced_support_focus' keeps the fast balanced path but lets one sharp ROI follow the active support surface"
        ),
    )
    parser.add_argument(
        "--immersive_support_entry_overlay",
        action="store_true",
        help=(
            "draw scene-anchored component_id labels on visible support objects and "
            "highlight the active support for debugging"
        ),
    )
    parser.add_argument(
        "--immersive_framegen",
        choices=("off", "static", "adaptive"),
        default="off",
        help=(
            "in-between display-frame generation mode for immersive Quest output: "
            "'off' sends only real source frames, "
            "'static' may synthesize one extra head-reprojected stereo frame after a real source frame, "
            "'adaptive' does the same with stricter motion/freshness guardrails "
            "(static/adaptive require --immersive_timewarp scene_depth_reproject)"
        ),
    )
    parser.add_argument(
        "--immersive_gaussian_render",
        choices=("serial", "stereo_parallel", "stereo_batched"),
        default=None,
        help=(
            "Gaussian render scheduling mode for immersive Quest output: "
            "default is 'stereo_batched' for native_gl and 'serial' otherwise; "
            "'serial' renders left/right Gaussian eyes sequentially, "
            "'stereo_parallel' experimentally renders both eyes concurrently on separate CUDA streams "
            "(requires --immersive_timewarp off on the reference path; overlap=on is also supported "
            "on the narrow stable path with --immersive_present_pipeline, "
            "--immersive_framegen off, and --immersive_timewarp off, and on the "
            "framegen path with --immersive_framegen static|adaptive and "
            "--immersive_timewarp scene_depth_reproject; non-overlap framegen v1 supports "
            "--immersive_gaussian_render serial only), "
            "'stereo_batched' experimentally renders both eyes in one batched gsplat call "
            "(requires --immersive_timewarp off; non-native backends also require "
            "--immersive_static_scene_overlap off; not supported for framegen v1)"
        ),
    )
    parser.add_argument(
        "--immersive_present_pipeline",
        choices=("off", "on"),
        default="on",
        help=(
            "experimentally pipelines compose + overlay + publish onto a presentation "
            "worker while keeping the current same-frame overlap path "
            "(requires --immersive_static_scene_overlap on; the narrow stable "
            "stereo_parallel path also requires --immersive_timewarp off and "
            "--immersive_framegen off; framegen may auto-use the same backend even "
            "without this flag)"
        ),
    )
    parser.add_argument(
        "--immersive_controller_translation_scale",
        type=float,
        default=1.2,
        help=(
            "multiplier on the case-default live controller translation gain used "
            "to map real controller motion into immersive world-space motion"
        ),
    )
    parser.add_argument(
        "--immersive_viewer_upload_mode",
        choices=("pbo", "direct", "legacy"),
        default="pbo",
        help=(
            "viewer texture upload implementation: "
            "'pbo' uses the asynchronous pixel-buffer path, "
            "'direct' uploads directly from shared mmap pointers on the render thread, "
            "'legacy' keeps the rollback vector-copy path"
        ),
    )
    parser.add_argument(
        "--immersive_viewer_upload_thread",
        choices=("auto", "render", "async"),
        default="auto",
        help=(
            "viewer upload scheduling: "
            "'auto' tries the async uploader and falls back to render-thread upload, "
            "'render' keeps uploads on the OpenXR render thread, "
            "'async' requires the shared-context async uploader"
        ),
    )
    parser.add_argument(
        "--immersive_viewer_upload_late_wait_us",
        type=int,
        default=0,
        help=(
            "bounded render-thread late-poll wait in microseconds. "
            "0 disables waiting; async upload should normally keep this at 0"
        ),
    )
    parser.add_argument(
        "--immersive_viewer_upload_ring_slots",
        type=int,
        default=5,
        help=(
            "number of PBO/texture slots in the immersive viewer upload ring. "
            "Allowed range is 3..8; async upload defaults to 5 to avoid slot pressure"
        ),
    )
    parser.add_argument(
        "--immersive_viewer_upload_busy_backoff_us",
        type=int,
        default=100,
        help=(
            "async uploader backoff in microseconds when no upload slot is reusable. "
            "0 keeps yield-only debug behavior"
        ),
    )
    return parser


def main(argv: list[str] | None = None):
    ensure_direct_launch_runtime_env(argv)
    parser = build_parser()
    args = parser.parse_args(argv)
    immersive_present_pipeline_enabled = (
        str(args.immersive_present_pipeline).strip().lower() == "on"
    )

    if args.n_dup != 0:
        raise ValueError("The shipped Quest immersive launcher supports only --n_dup 0.")
    if int(args.immersive_eye_resolution) <= 0:
        raise ValueError("--immersive_eye_resolution must be a positive integer.")
    configure_immersive_viewer_upload_runtime(args)
    immersive_static_scene_backend = str(
        args.immersive_static_scene_backend
    ).strip().lower()
    if (
        immersive_static_scene_backend == "native_gl"
        and args.immersive_static_scene_mode is not None
    ):
        raise ValueError(
            "--immersive_static_scene_mode is not used with "
            "--immersive_static_scene_backend native_gl; remove it. "
            "Native GL always uses static_scene_path=full_scene_per_eye."
        )
    if (
        immersive_static_scene_backend != "native_gl"
        and args.immersive_static_scene_mode is None
    ):
        args.immersive_static_scene_mode = "balanced_support_focus"
    if args.immersive_gaussian_render is None:
        args.immersive_gaussian_render = (
            "stereo_batched"
            if immersive_static_scene_backend == "native_gl"
            else "serial"
        )

    case_name = args.case_name
    canonical_case_name, manifest_dir, case_manifest = resolve_demo_case_manifest(case_name)
    validate_demo_case_assets(REPO_ROOT, canonical_case_name, manifest_dir, case_manifest)
    if canonical_case_name != case_name:
        print(
            "[quest_display] demo case alias resolved: "
            f"requested={case_name} canonical={canonical_case_name}",
            flush=True,
        )
    ensure_immersive_bridge_system_deps()

    global np, torch
    import numpy as np  # type: ignore[assignment]
    import torch  # type: ignore[assignment]

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    try:
        torch.set_float32_matmul_precision("high")
    except AttributeError:
        pass

    print("[quest_display] input_source=live_openxr_controller", flush=True)
    print("[quest_display] controller_mode=multi_points", flush=True)
    print("[quest_display] mode=immersive", flush=True)
    print("[quest_display] scene_preset=ILLIXR_lab", flush=True)
    print("[quest_display] immersive_render_preset=balanced", flush=True)
    print(
        "[quest_display] immersive_eye_resolution="
        f"{int(args.immersive_eye_resolution)}",
        flush=True,
    )
    print(f"[quest_display] immersive_timewarp={args.immersive_timewarp}", flush=True)
    print(
        "[quest_display] immersive_static_scene_overlap="
        f"{args.immersive_static_scene_overlap}",
        flush=True,
    )
    print(
        "[quest_display] immersive_static_scene_reuse="
        f"{args.immersive_static_scene_reuse}",
        flush=True,
    )
    print(
        "[quest_display] immersive_static_scene_backend="
        f"{args.immersive_static_scene_backend}",
        flush=True,
    )
    print(
        "[quest_display] immersive_gaussian_source_validation="
        f"{args.immersive_gaussian_source_validation}",
        flush=True,
    )
    if immersive_static_scene_backend == "native_gl":
        print(
            "[quest_display] immersive_static_scene_path=full_scene_per_eye",
            flush=True,
        )
        print(
            "[quest_display] immersive_native_gl_options="
            f"texture_mode={args.immersive_native_gl_texture_mode} "
            f"anisotropy={int(args.immersive_native_gl_anisotropy)} "
            f"msaa_samples={int(args.immersive_native_gl_msaa_samples)} "
            f"depth_format={args.immersive_native_gl_depth_format}",
            flush=True,
        )
    else:
        print(
            "[quest_display] immersive_static_scene_mode="
            f"{args.immersive_static_scene_mode}",
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
        "[quest_display] immersive_present_pipeline="
        f"{str(args.immersive_present_pipeline).strip().lower()}",
        flush=True,
    )
    print(
        "[quest_display] immersive_controller_translation_scale="
        f"{float(args.immersive_controller_translation_scale):.3f}",
        flush=True,
    )
    print(
        "[quest_display] immersive_viewer_upload="
        f"mode={args.immersive_viewer_upload_mode} "
        f"thread={args.immersive_viewer_upload_thread} "
        f"late_wait_us={int(args.immersive_viewer_upload_late_wait_us)} "
        f"ring_slots={int(args.immersive_viewer_upload_ring_slots)} "
        f"busy_backoff_us={int(args.immersive_viewer_upload_busy_backoff_us)}",
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
    print_cuda_startup_debug_banner(torch)
    set_all_seeds(42)

    ctx = attach_pycuda_context_for_current_torch_device()

    prioritize_conda_bin()
    prioritize_conda_runtime_libs()
    prefer_system_ninja_binary()
    configure_local_python_paths()
    from qqtt import InvPhyTrainerWarp
    from qqtt.utils import logger, cfg

    cfg.load_from_yaml(case_manifest.get("config", "configs/real.yaml"))
    cfg.demo_case_name = canonical_case_name
    cfg.demo_game_mode = str(case_manifest.get("game_mode", "")).strip().lower()
    cfg.demo_game_course_path = (
        None
        if case_manifest.get("game_course") is None
        else manifest_file_path(manifest_dir, case_manifest, "game_course")
    )
    cfg.demo_tutorial_slide_paths = resolve_demo_case_tutorial_slides(
        manifest_dir,
        case_manifest,
        canonical_case_name,
    )

    base_dir = f"./temp_experiments/{canonical_case_name}"

    optimal_path = manifest_file_path(manifest_dir, case_manifest, "optimal_params")
    with open(optimal_path, "rb") as f:
        optimal_params = pickle.load(f)
    cfg.set_optimal_params(optimal_params)
    demo_case_scale = apply_demo_case_world_scale_to_cfg(cfg, canonical_case_name)
    previous_physics_profile = apply_demo_case_physics_profile_to_cfg(
        cfg,
        canonical_case_name,
    )
    if abs(demo_case_scale - 1.0) > 1e-8:
        print(
            "[quest_display] demo case world scale: "
            f"case={canonical_case_name} scale={demo_case_scale:.8f} "
            f"object_radius={float(cfg.object_radius):.8f} "
            f"controller_radius={float(cfg.controller_radius):.8f} "
            f"collision_dist={float(cfg.collision_dist):.8f}",
            flush=True,
        )
    if previous_physics_profile is not None:
        print(
            "[quest_display] demo case physics profile: "
            f"case={canonical_case_name} "
            f"dt={float(cfg.dt):.8g} "
            f"num_substeps={int(cfg.num_substeps)} "
            f"self_collision={bool(getattr(cfg, 'self_collision', False))} "
            f"previous_dt={previous_physics_profile.get('dt', float(cfg.dt)):.8g} "
            "previous_num_substeps="
            f"{previous_physics_profile.get('num_substeps', int(cfg.num_substeps))} "
            "previous_self_collision="
            f"{bool(previous_physics_profile.get('self_collision', getattr(cfg, 'self_collision', False)))} "
            "reason=original_phystwin_rope_stability_self_collision "
            "size_scale=1.0",
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
    cfg.live_openxr_verbose_console_diagnostics = bool(args.profile)

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
            profile=args.profile,
            profile_freq=args.profile_freq,
            immersive_timewarp=args.immersive_timewarp,
            immersive_static_scene_overlap=args.immersive_static_scene_overlap,
            immersive_static_scene_reuse=args.immersive_static_scene_reuse,
            immersive_static_scene_backend=args.immersive_static_scene_backend,
            immersive_eye_resolution=args.immersive_eye_resolution,
            immersive_static_scene_mode=args.immersive_static_scene_mode,
            immersive_native_gl_texture_mode=args.immersive_native_gl_texture_mode,
            immersive_native_gl_anisotropy=args.immersive_native_gl_anisotropy,
            immersive_native_gl_msaa_samples=args.immersive_native_gl_msaa_samples,
            immersive_native_gl_depth_format=args.immersive_native_gl_depth_format,
            immersive_gaussian_source_validation=(
                args.immersive_gaussian_source_validation
            ),
            immersive_support_entry_overlay=args.immersive_support_entry_overlay,
            immersive_framegen=args.immersive_framegen,
            immersive_gaussian_render=args.immersive_gaussian_render,
            immersive_present_pipeline=immersive_present_pipeline_enabled,
            immersive_controller_translation_scale=(
                args.immersive_controller_translation_scale
            ),
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
    try:
        main()
    except (DemoAssetValidationError, StartupConfigurationError) as exc:
        raise SystemExit(str(exc)) from exc
