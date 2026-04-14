#!/usr/bin/env python3

from __future__ import annotations

import glob
import json
import os
import pickle
import random
import sys
from argparse import ArgumentParser

np = None
torch = None


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
            "the simple_lab scene, and the balanced immersive preset."
        )
    )
    parser.add_argument("--base_path", type=str, default="./data/different_types")
    parser.add_argument("--gaussian_path", type=str, default="./gaussian_output")
    parser.add_argument("--bg_img_path", type=str, default="./data/bg.png")
    parser.add_argument("--case_name", type=str, default="double_stretch_sloth")
    parser.add_argument("--n_dup", type=int, default=0, help="must remain 0 for the shipped Quest demo")
    parser.add_argument(
        "--scene_assets_root",
        type=str,
        default="./data/open_scene_assets",
        help="root directory for the immersive scene assets",
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
    print("[quest_display] scene_preset=simple_lab", flush=True)
    print("[quest_display] immersive_render_preset=balanced", flush=True)
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
    configure_local_python_paths()
    from qqtt import InvPhyTrainerWarp
    from qqtt.utils import logger, cfg

    base_path = args.base_path
    case_name = args.case_name

    if ("cloth" in case_name) or ("package" in case_name):
        cfg.load_from_yaml("configs/cloth.yaml")
    else:
        cfg.load_from_yaml("configs/real.yaml")

    base_dir = f"./temp_experiments/{case_name}"

    optimal_path = f"./experiments_optimization/{case_name}/optimal_params.pkl"
    assert os.path.exists(optimal_path), f"{case_name}: Optimal parameters not found: {optimal_path}"
    with open(optimal_path, "rb") as f:
        optimal_params = pickle.load(f)
    cfg.set_optimal_params(optimal_params)

    with open(f"{base_path}/{case_name}/calibrate.pkl", "rb") as f:
        c2ws = pickle.load(f)
    w2cs = [np.linalg.inv(c2w) for c2w in c2ws]
    cfg.c2ws = np.array(c2ws)
    cfg.w2cs = np.array(w2cs)

    with open(f"{base_path}/{case_name}/metadata.json", "r") as f:
        data = json.load(f)
    cfg.intrinsics = np.array(data["intrinsics"])
    cfg.WH = data["WH"]
    cfg.bg_img_path = args.bg_img_path

    exp_name = "init=hybrid_iso=True_ldepth=0.001_lnormal=0.0_laniso_0.0_lseg=1.0"
    gaussians_path = (
        f"{args.gaussian_path}/{case_name}/{exp_name}/point_cloud/iteration_10000/point_cloud.ply"
    )

    logger.set_log_file(path=base_dir, name="inference_log")

    trainer = InvPhyTrainerWarp(
        data_path=f"{base_path}/{case_name}/final_data.pkl",
        base_dir=base_dir,
    )

    best_model_path = glob.glob(f"experiments/{case_name}/train/best_*.pth")[0]
    output_dir = os.path.join("./gaussian_output_dynamic", case_name)

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
