#!/usr/bin/env python3

from __future__ import annotations

import json
import gc
import math
import os
import pickle
import random
import shutil
import subprocess
import sys
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from pathlib import Path

from tools.fetch_demo_case_assets import (
    DemoAssetValidationError,
    PUBLIC_SCENES,
    validate_all_demo_assets,
)

np = None
torch = None
REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_SCENE_ASSETS_ROOT = REPO_ROOT / "assets" / "scenes"
PUBLIC_DEMO_CASES = ("rope_game", "sloth")
DIRECT_GAUSSIAN_SCENES = frozenset({"garden", "ambulance"})
IMMERSIVE_START_POSTURES = ("standing", "seated")
DEFAULT_IMMERSIVE_CONTROLLER_MAX_MOTION_INTERVAL_M = 0.05
INTERACTIVE_WINDOW_ASPECT_WIDTH = 16
INTERACTIVE_WINDOW_ASPECT_HEIGHT = 9
INTERACTIVE_WINDOW_TARGET_MONITOR_AREA = 0.55
INTERACTIVE_WINDOW_MAX_WORKAREA_FRACTION = 0.94
INTERACTIVE_WINDOW_FALLBACK_WORKAREA = (1920, 1080)
SHARED_TUTORIAL_SLIDES = (
    "controls_overview.png",
    "interaction_tips.png",
)
RUNTIME_ENV_READY_SENTINEL = "BOBA_IMMERSIVE_RUNTIME_READY"
REQUIRED_RUNTIME_ENV = "phystwin-cu132"
DEFAULT_CUDA_HOME = "/usr/local/cuda"
BRIDGE_DEPS_CHECK_SCRIPT = (
    REPO_ROOT / "linux_pose_probe" / "check_boba_immersive_bridge_deps.sh"
)
CUDA_STARTUP_DEBUG_ENV = "BOBA_CUDA_STARTUP_DEBUG"
ILLIXR_BRIDGE_ENV_VARS = (
    "BOBA_ILLIXR_INPUT_SOCKET",
    "BOBA_ILLIXR_FRAME_PATH",
    "BOBA_ILLIXR_OVERLAY_PATH",
    "BOBA_ILLIXR_MODAL_PATH",
)


class StartupConfigurationError(RuntimeError):
    pass


def detected_conda_prefix() -> str | None:
    # Prefer the interpreter's real prefix.  CONDA_PREFIX can describe the
    # caller when this process was started through nested `conda run`.
    inferred_prefix = Path(sys.prefix)
    if (inferred_prefix / "conda-meta").is_dir():
        return str(inferred_prefix)

    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix and Path(conda_prefix).is_dir():
        return conda_prefix
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

    conda_prefix = detected_conda_prefix()
    active_env = Path(conda_prefix).resolve().name if conda_prefix else "none"
    if active_env != REQUIRED_RUNTIME_ENV:
        raise StartupConfigurationError(
            "Boba Demo must execute in the "
            f"{REQUIRED_RUNTIME_ENV!r} Conda environment; actual interpreter "
            f"environment is {active_env!r} (sys.prefix={Path(sys.prefix).resolve()}).\n"
            "Run ./boba_app.sh from any shell to enter the correct environment "
            "automatically."
        )

    conda_cuda_home = Path(conda_prefix) if conda_prefix else None
    if conda_cuda_home is not None and (conda_cuda_home / "bin" / "nvcc").is_file():
        cuda_home = str(conda_cuda_home)
    else:
        cuda_home = str(Path(current_env.get("CUDA_HOME") or DEFAULT_CUDA_HOME))
    launch_env["CUDA_HOME"] = cuda_home
    launch_env["PATH"] = prepend_env_entries(
        current_env.get("PATH"),
        [str(Path(cuda_home) / "bin")],
    )

    ld_library_leading_entries = []
    if conda_prefix:
        ld_library_leading_entries.append(str(Path(conda_prefix) / "lib"))
        ld_library_leading_entries.append(
            str(Path(conda_prefix) / "targets" / "x86_64-linux" / "lib")
        )
    ld_library_leading_entries.append(str(Path(cuda_home) / "lib64"))
    ld_library_leading_entries.append(
        str(Path(cuda_home) / "targets" / "x86_64-linux" / "lib")
    )
    launch_env["LD_LIBRARY_PATH"] = prepend_env_entries(
        current_env.get("LD_LIBRARY_PATH"),
        ld_library_leading_entries,
    )

    reexec_keys = ("PYTHONNOUSERSITE", "LD_LIBRARY_PATH")
    needs_reexec = any(current_env.get(key) != launch_env.get(key) for key in reexec_keys)
    if not needs_reexec:
        for key in ("CUDA_HOME", "PATH"):
            os.environ[key] = launch_env[key]
        return

    if current_env.get(RUNTIME_ENV_READY_SENTINEL) == "1":
        return

    launch_env[RUNTIME_ENV_READY_SENTINEL] = "1"
    print(
        "[startup] re-executing with phystwin-cu132 CUDA runtime libraries: "
        f"CUDA_HOME={cuda_home}",
        flush=True,
    )
    exec_argv = [sys.executable, str(Path(__file__).resolve())]
    exec_argv.extend(list(argv) if argv is not None else sys.argv[1:])
    os.execvpe(sys.executable, exec_argv, launch_env)


def ensure_immersive_bridge_system_deps() -> None:
    if os.environ.get("BOBA_ILLIXR_INPUT_SOCKET"):
        return
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


def configure_illixr_launch(enabled: bool) -> None:
    """Make the ILLIXR bridge an explicit launch-time opt-in."""
    if not enabled:
        for name in ILLIXR_BRIDGE_ENV_VARS:
            os.environ.pop(name, None)
        return

    missing = [name for name in ILLIXR_BRIDGE_ENV_VARS if not os.environ.get(name)]
    if missing:
        raise StartupConfigurationError(
            "--illixr requires the ILLIXR launcher environment; missing "
            + ", ".join(missing)
        )


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
    return str(case_name).strip().lower()


def resolve_demo_case_manifest(case_name: str) -> tuple[str, Path, dict]:
    canonical_case = canonical_demo_case_name(case_name)
    if canonical_case not in PUBLIC_DEMO_CASES:
        raise ValueError(
            f"Unsupported demo case '{case_name}'. Packaged cases: "
            f"{', '.join(PUBLIC_DEMO_CASES)}."
        )
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


def manifest_config_path(manifest: dict) -> str:
    relative_path = manifest.get("config")
    if relative_path is None:
        raise KeyError("Manifest is missing required key: config")
    config_path = Path(str(relative_path))
    if not config_path.is_absolute():
        config_path = REPO_ROOT / config_path
    return str(config_path.resolve())


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


def set_all_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def interactive_window_size_for_workarea(
    workarea_width: int,
    workarea_height: int,
) -> tuple[int, int]:
    """Return a 16:9 window covering slightly over half the monitor area."""

    workarea_width = int(workarea_width)
    workarea_height = int(workarea_height)
    if workarea_width <= 0 or workarea_height <= 0:
        raise ValueError("Monitor work-area dimensions must be positive.")

    aspect = float(INTERACTIVE_WINDOW_ASPECT_WIDTH) / float(
        INTERACTIVE_WINDOW_ASPECT_HEIGHT
    )
    target_area = (
        float(workarea_width)
        * float(workarea_height)
        * float(INTERACTIVE_WINDOW_TARGET_MONITOR_AREA)
    )
    target_height = math.sqrt(target_area / aspect)
    target_width = target_height * aspect
    max_width = float(workarea_width) * INTERACTIVE_WINDOW_MAX_WORKAREA_FRACTION
    max_height = float(workarea_height) * INTERACTIVE_WINDOW_MAX_WORKAREA_FRACTION
    fit_scale = min(1.0, max_width / target_width, max_height / target_height)
    target_width *= fit_scale
    target_height *= fit_scale

    # Derive one integer dimension from the other so rounding cannot produce a
    # visibly different aspect ratio.
    width = max(1, int(round(target_width)))
    height = max(
        1,
        int(
            round(
                width
                * float(INTERACTIVE_WINDOW_ASPECT_HEIGHT)
                / float(INTERACTIVE_WINDOW_ASPECT_WIDTH)
            )
        ),
    )
    if height > int(max_height):
        height = max(1, int(max_height))
        width = max(
            1,
            int(
                round(
                    height
                    * float(INTERACTIVE_WINDOW_ASPECT_WIDTH)
                    / float(INTERACTIVE_WINDOW_ASPECT_HEIGHT)
                )
            ),
        )
    return width, height


def _primary_monitor_workarea(glfw_module):
    monitor = glfw_module.get_primary_monitor()
    fallback_width, fallback_height = INTERACTIVE_WINDOW_FALLBACK_WORKAREA
    if monitor is None:
        return None, (0, 0, fallback_width, fallback_height)

    get_workarea = getattr(glfw_module, "get_monitor_workarea", None)
    if callable(get_workarea):
        try:
            workarea = tuple(int(value) for value in get_workarea(monitor))
            if len(workarea) == 4 and workarea[2] > 0 and workarea[3] > 0:
                return monitor, workarea
        except Exception:
            pass

    try:
        mode = glfw_module.get_video_mode(monitor)
        monitor_width = int(mode.size.width)
        monitor_height = int(mode.size.height)
        monitor_x, monitor_y = glfw_module.get_monitor_pos(monitor)
        if monitor_width > 0 and monitor_height > 0:
            return monitor, (
                int(monitor_x),
                int(monitor_y),
                monitor_width,
                monitor_height,
            )
    except Exception:
        pass
    return monitor, (0, 0, fallback_width, fallback_height)


def create_gl_window(
    width: int | None = None,
    height: int | None = None,
    visible: bool = True,
):
    import glfw
    from OpenGL import GL as gl

    assert glfw.init(), "GLFW init failed"
    monitor, workarea = _primary_monitor_workarea(glfw)
    adaptive_size = width is None and height is None
    if adaptive_size:
        width, height = interactive_window_size_for_workarea(
            workarea[2],
            workarea[3],
        )
    elif width is None or height is None:
        raise ValueError("Interactive window width and height must be supplied together.")
    width = int(width)
    height = int(height)
    glfw.window_hint(glfw.CLIENT_API, glfw.OPENGL_API)
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 4)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 6)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
    glfw.window_hint(glfw.VISIBLE, glfw.TRUE if visible else glfw.FALSE)

    window = glfw.create_window(width, height, "Boba Quest Immersive", None, None)
    assert window, "create_window failed (need X11 desktop GL)"

    if adaptive_size:
        set_aspect_ratio = getattr(glfw, "set_window_aspect_ratio", None)
        if callable(set_aspect_ratio):
            set_aspect_ratio(
                window,
                INTERACTIVE_WINDOW_ASPECT_WIDTH,
                INTERACTIVE_WINDOW_ASPECT_HEIGHT,
            )
        if visible:
            try:
                glfw.set_window_pos(
                    window,
                    int(workarea[0]) + max(0, (int(workarea[2]) - width) // 2),
                    int(workarea[1]) + max(0, (int(workarea[3]) - height) // 2),
                )
            except Exception:
                pass
        coverage = float(width * height) / float(workarea[2] * workarea[3])
        print(
            "[interactive_window] monitor-aware window: "
            f"size={width}x{height} aspect=16:9 "
            f"monitor_workarea={workarea[2]}x{workarea[3]} "
            f"area_coverage={coverage * 100.0:.1f}%",
            flush=True,
        )

    glfw.make_context_current(window)
    _ = gl.glGetString(gl.GL_VERSION)
    glfw.swap_interval(0)
    return window


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
    ring_slots = int(args.immersive_viewer_upload_ring_slots)
    busy_backoff_us = int(args.immersive_viewer_upload_busy_backoff_us)
    if ring_slots < 3 or ring_slots > 8:
        raise ValueError("--immersive_viewer_upload_ring_slots must be between 3 and 8.")
    if busy_backoff_us < 0:
        raise ValueError("--immersive_viewer_upload_busy_backoff_us must be >= 0.")

    # The C++ OpenXR bridge consumes these as process environment settings.
    # Keep the user-facing control as launcher flags so runs are reproducible
    # and not affected by stale shell exports.
    os.environ["BOBA_IMMERSIVE_VIEWER_UPLOAD_MODE"] = upload_mode
    os.environ["BOBA_IMMERSIVE_VIEWER_UPLOAD_THREAD"] = upload_thread
    os.environ["BOBA_IMMERSIVE_VIEWER_UPLOAD_RING_SLOTS"] = str(ring_slots)
    os.environ["BOBA_IMMERSIVE_VIEWER_UPLOAD_BUSY_BACKOFF_US"] = str(busy_backoff_us)


def configure_demo_case_runtime(
    case_name: str,
    cfg,
    logger,
    scene_name: str = "lab",
) -> dict:
    """Load one packaged object's isolated configuration and runtime paths."""

    canonical_case_name, manifest_dir, case_manifest = resolve_demo_case_manifest(
        case_name
    )
    cfg.reset()
    config_path = manifest_config_path(case_manifest)
    cfg.load_from_yaml(config_path)
    cfg.demo_case_name = canonical_case_name
    cfg.demo_game_mode = str(case_manifest.get("game_mode", "")).strip().lower()
    cfg.demo_game_course_path = (
        None
        if case_manifest.get("game_course") is None
        else manifest_file_path(manifest_dir, case_manifest, "game_course")
    )
    cfg.visual_gaussian_retarget = (
        str(case_manifest.get("visual_gaussian_retarget") or "").strip().lower()
    )
    cfg.visual_gaussian_driver_case = (
        str(case_manifest.get("visual_gaussian_driver_case") or "").strip().lower()
    )
    cfg.visual_gaussian_source_case = (
        str(case_manifest.get("visual_gaussian_source_case") or "").strip().lower()
    )
    cfg.demo_tutorial_slide_paths = resolve_demo_case_tutorial_slides(
        manifest_dir,
        case_manifest,
        canonical_case_name,
    )
    scene_name = str(scene_name).strip().lower()
    cfg.immersive_scene_name = scene_name
    if scene_name in DIRECT_GAUSSIAN_SCENES:
        cfg.demo_game_mode = "free_play"
        cfg.demo_game_course_path = None
        cfg.demo_tutorial_slide_paths = [
            str(REPO_ROOT / "assets" / "tutorial" / slide_name)
            for slide_name in SHARED_TUTORIAL_SLIDES
        ]

    optimal_path = manifest_file_path(manifest_dir, case_manifest, "optimal_params")
    with open(optimal_path, "rb") as handle:
        cfg.set_optimal_params(pickle.load(handle))
    cfg.demo_case_world_scale = 1.0

    with open(
        manifest_file_path(manifest_dir, case_manifest, "calibrate"),
        "rb",
    ) as handle:
        c2ws = pickle.load(handle)
    w2cs = [np.linalg.inv(c2w) for c2w in c2ws]
    cfg.c2ws = np.array(c2ws)
    cfg.w2cs = np.array(w2cs)

    with open(
        manifest_file_path(manifest_dir, case_manifest, "metadata"),
        "r",
        encoding="utf-8",
    ) as handle:
        metadata = json.load(handle)
    cfg.intrinsics = np.array(metadata["intrinsics"])
    cfg.WH = metadata["WH"]

    base_dir = f"./temp_experiments/{canonical_case_name}"
    logger.set_log_file(path=base_dir, name="inference_log")
    print(
        "[quest_display] demo case config: "
        f"case={canonical_case_name} "
        f"mode={cfg.demo_game_mode} "
        f"config={config_path} "
        f"dt={float(cfg.dt):.8g} "
        f"num_substeps={int(cfg.num_substeps)} "
        f"self_collision={bool(getattr(cfg, 'self_collision', False))} "
        f"object_radius={float(cfg.object_radius):.8f} "
        f"controller_radius={float(cfg.controller_radius):.8f} "
        f"collision_dist={float(cfg.collision_dist):.8f}",
        flush=True,
    )
    return {
        "case_name": canonical_case_name,
        "manifest_dir": manifest_dir,
        "manifest": case_manifest,
        "base_dir": base_dir,
        "data_path": manifest_file_path(
            manifest_dir,
            case_manifest,
            "final_data",
        ),
        "best_model_path": manifest_file_path(
            manifest_dir,
            case_manifest,
            "best_model",
        ),
        "gaussians_path": manifest_file_path(
            manifest_dir,
            case_manifest,
            "gaussian_ply",
        ),
        "output_dir": os.path.join(
            "./gaussian_output_dynamic",
            canonical_case_name,
        ),
    }


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(
        formatter_class=ArgumentDefaultsHelpFormatter,
        description=(
            "Run the shipped Boba Quest immersive demo. "
            "Standalone OpenXR is the default; --illixr enables the ILLIXR-managed bridge. "
            "The launcher uses immersive Quest display, "
            "a launch-time Lab, Garden, or Ambulance scene, and the balanced "
            "immersive preset. "
            "Runtime demo assets are resolved from ./assets/."
        )
    )
    parser.add_argument(
        "--illixr",
        action="store_true",
        help=(
            "use the ILLIXR-managed input/output bridge; intended for the ILLIXR "
            "plugin, which supplies the required socket and shared-memory paths"
        ),
    )
    parser.add_argument(
        "--scene",
        choices=PUBLIC_SCENES,
        default="lab",
        help=(
            "launch the mesh Lab, Gaussian Mip-NeRF 360 Garden, or bundled "
            "Gaussian Ambulance scene"
        ),
    )
    parser.add_argument(
        "--garden-quality",
        choices=("auto", "full", "balanced", "performance"),
        default="balanced",
        help=(
            "Garden Gaussian tier; balanced is the default, while auto uses "
            "the cached 72-FPS hardware profile"
        ),
    )
    parser.add_argument(
        "--garden-debug-collision",
        action="store_true",
        help="export the aligned Garden collision mesh for developer inspection",
    )
    parser.add_argument(
        "--case_name",
        type=str,
        choices=PUBLIC_DEMO_CASES,
        default="rope_game",
        help=(
            "initial packaged object; Rope is the normal demo default and both "
            "objects remain selectable at runtime"
        ),
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
        default="visible",
        help="show or hide the local OpenGL window while Quest output stays active",
    )
    parser.add_argument(
        "--immersive_start_posture",
        choices=IMMERSIVE_START_POSTURES,
        default="standing",
        help=(
            "explicit standing or seated startup layout; automatic posture "
            "detection is intentionally not enabled"
        ),
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
            "'adaptive' adds motion/head-reset guardrails before reuse. "
            "Native GL full-scene overlap currently forces this setting to 'off'"
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
            "CUDA readback (supports serial or stereo-batched Gaussian rendering "
            "with overlap on or off; requires timewarp, framegen, and the present "
            "pipeline off)"
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
        "--immersive_native_gl_mipmap_lod_bias",
        type=float,
        default=0.0,
        help=(
            "native GL mipmap LOD bias for --immersive_native_gl_texture_mode "
            "stable_mipmap. Positive values sample blurrier mip levels sooner; "
            "use 0.50 as a small far-distance anti-shimmer diagnostic"
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
            "expensive Gaussian source corruption validator for rope_game "
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
        "--immersive_controller_max_motion_interval_m",
        type=float,
        default=DEFAULT_IMMERSIVE_CONTROLLER_MAX_MOTION_INTERVAL_M,
        help=(
            "maximum active controller-target translation in virtual scene meters "
            "advanced by one full physics period; the default is the measured "
            "maximum across all 22 recorded test trajectories, rounded upward to "
            "a practical 5 cm bound. Larger tracking jumps catch up over later "
            "rendered frames toward the newest tracked pose, with one physics "
            "period and one LBS/render update per frame"
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
    configure_illixr_launch(args.illixr)
    if args.scene in DIRECT_GAUSSIAN_SCENES:
        # These scenes are themselves part of the batched Gaussian render. The
        # static adapter supplies a depthless retained frame, so static workers
        # and cross-frame scene reuse are intentionally disabled.
        args.immersive_timewarp = "off"
        args.immersive_static_scene_overlap = "off"
        args.immersive_static_scene_reuse = "off"
        args.immersive_static_scene_backend = "gpu"
        args.immersive_static_scene_mode = "balanced_support_focus"
        args.immersive_framegen = "off"
        args.immersive_gaussian_render = "stereo_batched"
        args.immersive_present_pipeline = "off"
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

    validate_all_demo_assets(
        REPO_ROOT,
        scene_name=args.scene,
        garden_quality=args.garden_quality,
    )
    print(
        "[quest_display] validated selectable objects: Rope, Sloth",
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

    input_source = (
        "illixr_switchboard"
        if os.environ.get("BOBA_ILLIXR_INPUT_SOCKET")
        else "live_openxr_controller"
    )
    print(f"[quest_display] input_source={input_source}", flush=True)
    print("[quest_display] controller_mode=multi_points", flush=True)
    print("[quest_display] mode=immersive", flush=True)
    scene_preset = {
        "lab": "ILLIXR_lab",
        "garden": "Mip-NeRF_360_garden",
        "ambulance": "Insta360_ambulance",
    }[args.scene]
    print(f"[quest_display] scene_preset={scene_preset}", flush=True)
    if args.scene == "garden":
        print(
            f"[quest_display] garden_quality={args.garden_quality} mode=free_play",
            flush=True,
        )
    elif args.scene == "ambulance":
        print("[quest_display] ambulance_sog=v2 mode=free_play", flush=True)
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
            f"mipmap_lod_bias={float(args.immersive_native_gl_mipmap_lod_bias):.3f} "
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
        "[quest_display] immersive_controller_max_motion_interval_m="
        f"{float(args.immersive_controller_max_motion_interval_m):.3f}",
        flush=True,
    )
    print(
        "[quest_display] immersive_viewer_upload="
        f"mode={args.immersive_viewer_upload_mode} "
        f"thread={args.immersive_viewer_upload_thread} "
        f"ring_slots={int(args.immersive_viewer_upload_ring_slots)} "
        f"busy_backoff_us={int(args.immersive_viewer_upload_busy_backoff_us)}",
        flush=True,
    )
    print(
        f"[quest_display] interactive_window_mode={args.interactive_window_mode}",
        flush=True,
    )
    print(
        f"[quest_display] immersive_start_posture={args.immersive_start_posture}",
        flush=True,
    )

    # Finish all Python/native-module imports before attaching PyCUDA.  If a
    # dependency is missing, this prevents PyCUDA's context-stack abort from
    # obscuring the actionable import error during interpreter shutdown.
    prioritize_conda_bin()
    prioritize_conda_runtime_libs()
    prefer_system_ninja_binary()
    from qqtt import InvPhyTrainerWarp
    from qqtt.utils import logger, cfg

    window = create_gl_window(
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

    active_case_name = canonical_demo_case_name(args.case_name)
    immersive_bridge = None
    immersive_session_state = {}
    show_startup_tutorial = True
    rollback_case_name = None
    last_switch_result = None
    trainer = None
    try:
        while True:
            try:
                set_all_seeds(42)
                runtime_case = configure_demo_case_runtime(
                    active_case_name,
                    cfg,
                    logger,
                    scene_name=args.scene,
                )
                cfg.live_openxr_verbose_console_diagnostics = bool(args.profile)
                trainer = InvPhyTrainerWarp(
                    data_path=runtime_case["data_path"],
                    base_dir=runtime_case["base_dir"],
                )
                episode_result = (
                    trainer.interactive_playground_quest_immersive_balanced(
                        runtime_case["best_model_path"],
                        runtime_case["gaussians_path"],
                        output_dir=runtime_case["output_dir"],
                        n_dup=args.n_dup,
                        window=window,
                        cuda_ctx=ctx,
                        interactive_window_mode=args.interactive_window_mode,
                        scene_assets_root=args.scene_assets_root,
                        immersive_start_posture=args.immersive_start_posture,
                        profile=args.profile,
                        profile_freq=args.profile_freq,
                        immersive_timewarp=args.immersive_timewarp,
                        immersive_static_scene_overlap=(
                            args.immersive_static_scene_overlap
                        ),
                        immersive_static_scene_reuse=(
                            args.immersive_static_scene_reuse
                        ),
                        immersive_static_scene_backend=(
                            args.immersive_static_scene_backend
                        ),
                        immersive_eye_resolution=args.immersive_eye_resolution,
                        immersive_static_scene_mode=args.immersive_static_scene_mode,
                        immersive_native_gl_texture_mode=(
                            args.immersive_native_gl_texture_mode
                        ),
                        immersive_native_gl_anisotropy=(
                            args.immersive_native_gl_anisotropy
                        ),
                        immersive_native_gl_mipmap_lod_bias=(
                            args.immersive_native_gl_mipmap_lod_bias
                        ),
                        immersive_native_gl_msaa_samples=(
                            args.immersive_native_gl_msaa_samples
                        ),
                        immersive_native_gl_depth_format=(
                            args.immersive_native_gl_depth_format
                        ),
                        immersive_gaussian_source_validation=(
                            args.immersive_gaussian_source_validation
                        ),
                        immersive_support_entry_overlay=(
                            args.immersive_support_entry_overlay
                        ),
                        immersive_framegen=args.immersive_framegen,
                        immersive_gaussian_render=args.immersive_gaussian_render,
                        immersive_present_pipeline=(
                            immersive_present_pipeline_enabled
                        ),
                        immersive_controller_translation_scale=(
                            args.immersive_controller_translation_scale
                        ),
                        immersive_controller_max_motion_interval_m=(
                            args.immersive_controller_max_motion_interval_m
                        ),
                        existing_immersive_bridge=immersive_bridge,
                        immersive_session_state=immersive_session_state,
                        show_startup_tutorial=show_startup_tutorial,
                        manage_cuda_context=False,
                        scene_name=args.scene,
                        garden_quality=args.garden_quality,
                        garden_debug_collision=args.garden_debug_collision,
                    )
                )
            except Exception as exc:
                if rollback_case_name is None or immersive_bridge is None:
                    raise
                failed_case_name = active_case_name
                print(
                    "[quest_display] object load failed; restoring previous object: "
                    f"failed_case={failed_case_name} "
                    f"restore_case={rollback_case_name} "
                    f"error={type(exc).__name__}: {exc}",
                    flush=True,
                )
                if last_switch_result is not None:
                    left_frame = last_switch_result.get("last_left_frame")
                    right_frame = last_switch_result.get("last_right_frame")
                    if left_frame is not None and right_frame is not None:
                        try:
                            immersive_bridge.publish_stereo_frames(
                                left_frame,
                                right_frame,
                                overlay_bitmap_quad=last_switch_result.get(
                                    "error_overlay_bitmap_quad"
                                ),
                            )
                        except Exception as overlay_exc:
                            print(
                                "[quest_display] unable to publish switch error overlay: "
                                f"{type(overlay_exc).__name__}: {overlay_exc}",
                                flush=True,
                            )
                active_case_name = rollback_case_name
                rollback_case_name = None
                show_startup_tutorial = False
                trainer = None
                gc.collect()
                torch.cuda.empty_cache()
                continue

            immersive_bridge = episode_result.get("bridge", immersive_bridge)
            action = str(episode_result.get("action", "exit")).strip().lower()
            if action != "switch":
                break

            next_case_name = canonical_demo_case_name(
                episode_result.get("next_case")
            )
            if next_case_name not in PUBLIC_DEMO_CASES:
                raise RuntimeError(
                    f"Runtime selector requested unsupported case: {next_case_name}"
                )
            rollback_case_name = active_case_name
            active_case_name = next_case_name
            immersive_session_state = dict(
                episode_result.get("session_state") or {}
            )
            last_switch_result = episode_result
            show_startup_tutorial = False
            trainer = None
            gc.collect()
            torch.cuda.empty_cache()
    finally:
        import glfw

        if immersive_bridge is not None:
            try:
                immersive_bridge.stop()
            except Exception:
                pass

        interactive_preview_renderer = immersive_session_state.get(
            "interactive_preview_renderer"
        )
        if interactive_preview_renderer is not None:
            try:
                interactive_preview_renderer.delete()
            except Exception:
                pass

        interactive_spectator_renderer = immersive_session_state.get(
            "interactive_spectator_renderer"
        )
        if interactive_spectator_renderer is not None:
            try:
                interactive_spectator_renderer.delete()
            except Exception:
                pass

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
            ctx.pop()
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
