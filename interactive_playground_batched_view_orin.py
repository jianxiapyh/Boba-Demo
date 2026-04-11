#!/usr/bin/env python3
# run_batched_vis_orin.py
#
# Accepts -exp/--replay_experiment_trace for CLI compatibility, but ignores it.

from __future__ import annotations

import os
import glob
import json
import pickle
import random
import sys
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import torch

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
try:
    torch.set_float32_matmul_precision("high")
except AttributeError:
    pass


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

    window = glfw.create_window(width, height, "Interactive Playground (Zero-copy)", None, None)
    assert window, "create_window failed (need X11 desktop GL)"

    glfw.make_context_current(window)
    _ = gl.glGetString(gl.GL_VERSION)  # force GL init
    glfw.swap_interval(0)
    return window


def configure_local_python_paths():
    repo_root = os.path.dirname(os.path.abspath(__file__))
    submodule_roots = (
        os.path.join(repo_root, "gaussian_splatting", "submodules", "simple-knn"),
        os.path.join(repo_root, "gaussian_splatting", "submodules", "diff-gaussian-rasterization"),
        os.path.join(repo_root, "gaussian_splatting", "submodules", "fused-ssim"),
    )
    for path in reversed(submodule_roots):
        if os.path.isdir(path) and path not in sys.path:
            sys.path.insert(0, path)


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


def main():
    parser = ArgumentParser()
    parser.add_argument("--base_path", type=str, default="./data/different_types")
    parser.add_argument("--gaussian_path", type=str, default="./gaussian_output")
    parser.add_argument("--bg_img_path", type=str, default="./data/bg.png")
    parser.add_argument("--case_name", type=str, default="double_lift_cloth_3")
    parser.add_argument(
        "-eval",
        "--eval_image_quality",
        action="store_true",
        help=(
            "write capture artifacts and verbose frame-compositing timing under "
            "./gaussian_output_dynamic/<case_name>"
        ),
    )
    parser.add_argument("--n_dup", type=int, default=0, help="number of object duplicates")
    parser.add_argument(
        "--input_source",
        choices=("recorded", "live_openxr", "live_openxr_controller"),
        default="recorded",
        help="choose between replayed trajectories, live Quest hand joints, and live Quest controllers",
    )
    parser.add_argument(
        "--controller_mode",
        choices=("multi_points",),
        default="multi_points",
        help="controller attachment mode for live_openxr_controller; multi_points is the default",
    )
    parser.add_argument(
        "--quest_display_mode",
        choices=("off", "panel", "primary"),
        default="off",
        help="Quest display mode for the final composited frame; 'primary' treats Quest as the intended display target",
    )
    parser.add_argument(
        "--interactive_window_mode",
        choices=("visible", "hidden"),
        default="visible",
        help="whether to show the local Interactive Playground window or keep it hidden as an offscreen GL context",
    )

    # Compatibility flag (accepted but not used)
    parser.add_argument(
        "-exp", "--replay_experiment_trace",
        action="store_true",
        help="(compat) accepted but ignored in this script"
    )

    args = parser.parse_args()

    if args.quest_display_mode != "off" and args.input_source != "live_openxr_controller":
        if args.input_source == "recorded":
            print(
                "[quest_display] overriding input_source=recorded -> "
                "live_openxr_controller for Quest mode",
                flush=True,
            )
            args.input_source = "live_openxr_controller"
        else:
            raise ValueError(
                "Quest display mode currently supports only "
                "--input_source live_openxr_controller"
            )
    if args.quest_display_mode == "primary":
        print("[quest_display] quest_primary_display enabled", flush=True)
        print(
            f"[quest_display] input_source={args.input_source}",
            flush=True,
        )
        print(
            f"[quest_display] interactive_window_mode={args.interactive_window_mode}",
            flush=True,
        )

    # -------------------------
    # (0) GL context FIRST
    # -------------------------
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

    # -------------------------
    # (1) Create Torch primary CUDA context (AFTER GL exists)
    # -------------------------
    _ = torch.empty(1, device="cuda")
    set_all_seeds(42)

    # -------------------------
    # (2) Attach PyCUDA to Torch primary context (AFTER GL + CUDA exist)
    # -------------------------
    import pycuda.driver as cuda_driver

    cuda_driver.init()
    ctx = cuda_driver.Context.attach()

    # -------------------------
    # (3) Import your stack AFTER contexts exist
    # -------------------------
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

    summary_output_path = os.path.join("./gaussian_output_dynamic", case_name)

    try:
        trainer.interactive_playground_batched_visualization(
            best_model_path,
            gaussians_path,
            summary_output_path=summary_output_path,
            save_eval_artifacts=args.eval_image_quality,
            n_dup=args.n_dup,
            window=window,
            cuda_ctx=ctx,
            input_source=args.input_source,
            controller_mode=args.controller_mode,
            quest_display_mode=args.quest_display_mode,
            interactive_window_mode=args.interactive_window_mode,
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
