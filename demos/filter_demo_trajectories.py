import argparse
import glob
import os
import pickle
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
ENV_BIN = Path(sys.executable).resolve().parent
os.environ["PATH"] = f"{ENV_BIN}:{os.environ.get('PATH', '')}"

from qqtt.engine.trainer_warp import BatchedReplayCheckError


DEFAULT_EXP_NAME = "init=hybrid_iso=True_ldepth=0.001_lnormal=0.0_laniso_0.0_lseg=1.0"


def build_parser():
    parser = argparse.ArgumentParser(description="Filter Demo 2 controller trajectories")
    parser.add_argument("--base_path", type=str, default="./data/different_types")
    parser.add_argument("--gaussian_path", type=str, default="./gaussian_output")
    parser.add_argument("--bg_img_path", type=str, default="./data/bg.png")
    parser.add_argument("--case_name", type=str, default="single_push_rope_4")
    parser.add_argument("--input_pkl", type=str, default=None)
    parser.add_argument("--output_pkl", type=str, default=None)
    parser.add_argument("--batch_size_needed", type=int, default=100)
    parser.add_argument("--filter_batch_size", type=int, default=128)
    parser.add_argument("--max_candidates", type=int, default=None)
    parser.add_argument("--replay_start", type=int, default=0)
    parser.add_argument("--replay_end", type=int, default=None)
    parser.add_argument("--sim_force_mode", type=str, default="gather")
    parser.add_argument("--gaussian_render_mode", type=str, default="shared_template")
    return parser


def default_input_pkl(base_path, case_name):
    return os.path.join(base_path, case_name, "multi_ctrls.pkl")


def default_output_pkl(base_path, case_name):
    return os.path.join(base_path, case_name, "multi_ctrls_demo2_filtered.pkl")


def load_controller_group_cpu(pkl_path):
    with open(pkl_path, "rb") as handle:
        root = pickle.load(handle)
    if not isinstance(root, dict) or "controller_points_group" not in root:
        raise ValueError(f"{pkl_path} must contain a controller_points_group dict entry")
    group = root["controller_points_group"]
    if not isinstance(group, list) or not group:
        raise ValueError("controller_points_group must be a non-empty list")

    arrays = []
    expected_shape = None
    for idx, item in enumerate(group):
        arr = np.asarray(item, dtype=np.float32)
        if arr.ndim != 3 or arr.shape[-1] != 3:
            raise ValueError(
                f"controller_points_group[{idx}] must have shape (T,C,3), got {arr.shape}"
            )
        if expected_shape is None:
            expected_shape = arr.shape
        elif arr.shape != expected_shape:
            raise ValueError(
                f"controller_points_group[{idx}] shape {arr.shape} does not match "
                f"{expected_shape}"
            )
        arrays.append(arr)
    tensor = torch.from_numpy(np.stack(arrays, axis=0))
    return root, arrays, tensor


def failure_to_dict(indices, failure):
    original = getattr(failure, "original_error", None)
    return {
        "candidate_indices": list(indices),
        "frame_idx": getattr(failure, "frame_idx", None),
        "batch_element": getattr(failure, "batch_element", None),
        "hinted_local_instance": getattr(failure, "hinted_instance", None),
        "message": str(failure),
        "original_error": str(original) if original is not None else None,
    }


def run_candidate_batch(
    trainer,
    model_path,
    gs_path,
    group_tensor_cpu,
    indices,
    args,
):
    chunk = group_tensor_cpu[list(indices)].to(device="cuda", non_blocking=False)
    try:
        trainer.check_batched_replay_lbs(
            model_path=model_path,
            gs_path=gs_path,
            controller_points_group=chunk,
            replay_start=args.replay_start,
            replay_end=args.replay_end,
            gaussian_render_mode=args.gaussian_render_mode,
            sim_force_mode=args.sim_force_mode,
        )
        return True, None
    except BatchedReplayCheckError as exc:
        return False, exc
    finally:
        del chunk
        torch.cuda.empty_cache()


def isolate_good_indices(
    trainer,
    model_path,
    gs_path,
    group_tensor_cpu,
    indices,
    args,
    bad_failures,
):
    ok, failure = run_candidate_batch(
        trainer,
        model_path,
        gs_path,
        group_tensor_cpu,
        indices,
        args,
    )
    if ok:
        print(f"[filter] accepted batch of {len(indices)} trajectories")
        return list(indices)

    if len(indices) == 1:
        print(f"[filter] rejected trajectory {indices[0]}: {failure}")
        bad_failures.append(failure_to_dict(indices, failure))
        return []

    hinted = getattr(failure, "hinted_instance", None)
    if hinted is not None and 0 <= hinted < len(indices):
        hinted_index = indices[hinted]
        print(
            "[filter] batch failed; testing hinted candidate "
            f"{hinted_index} before recursive split"
        )
        hinted_good = isolate_good_indices(
            trainer,
            model_path,
            gs_path,
            group_tensor_cpu,
            [hinted_index],
            args,
            bad_failures,
        )
        remaining = [idx for idx in indices if idx != hinted_index]
        if len(remaining) < len(indices):
            return hinted_good + isolate_good_indices(
                trainer,
                model_path,
                gs_path,
                group_tensor_cpu,
                remaining,
                args,
                bad_failures,
            )

    mid = len(indices) // 2
    print(f"[filter] batch failed; splitting {len(indices)} -> {mid} + {len(indices) - mid}")
    left = isolate_good_indices(
        trainer,
        model_path,
        gs_path,
        group_tensor_cpu,
        indices[:mid],
        args,
        bad_failures,
    )
    right = isolate_good_indices(
        trainer,
        model_path,
        gs_path,
        group_tensor_cpu,
        indices[mid:],
        args,
        bad_failures,
    )
    return left + right


def main():
    args = build_parser().parse_args()
    if args.batch_size_needed < 1:
        raise ValueError("--batch_size_needed must be positive")
    if args.filter_batch_size < 1:
        raise ValueError("--filter_batch_size must be positive")

    args.input_pkl = args.input_pkl or default_input_pkl(args.base_path, args.case_name)
    args.output_pkl = args.output_pkl or default_output_pkl(args.base_path, args.case_name)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    try:
        torch.set_float32_matmul_precision("high")
    except AttributeError:
        pass

    from gaussian_splatting import dynamic_utils as dynamic_utils_backend
    from interactive_playground import load_case_config, set_all_seeds
    from qqtt import InvPhyTrainerWarp
    from qqtt.utils import cfg, logger

    print(
        "[Boba] dynamic_utils backend: "
        f"{dynamic_utils_backend.SELECTED_DYNAMIC_UTIL_VARIANT} "
        f"(device={dynamic_utils_backend.DETECTED_DEVICE_NAME}, "
        f"BOBA_DEVICE={dynamic_utils_backend.BOBA_DEVICE})"
    )
    set_all_seeds(42)
    _ = torch.empty(1, device="cuda")

    root, group_arrays, group_tensor_cpu = load_controller_group_cpu(args.input_pkl)
    total_candidates = int(group_tensor_cpu.shape[0])
    candidate_count = total_candidates
    if args.max_candidates is not None:
        candidate_count = min(candidate_count, int(args.max_candidates))
    if candidate_count < args.batch_size_needed:
        raise ValueError(
            f"Only {candidate_count} candidates available, but "
            f"--batch_size_needed={args.batch_size_needed}."
        )

    load_case_config(args, cfg, logger)
    output_dir = os.path.join("results", "demo2_filter", args.case_name)
    os.makedirs(output_dir, exist_ok=True)
    logger.set_log_file(path=output_dir, name="filter_demo_trajectories")

    gaussians_path = (
        f"{args.gaussian_path}/{args.case_name}/{DEFAULT_EXP_NAME}/point_cloud/"
        "iteration_10000/point_cloud.ply"
    )
    best_model_matches = glob.glob(f"experiments/{args.case_name}/train/best_*.pth")
    if not best_model_matches:
        raise FileNotFoundError(f"Unable to find best checkpoint for case: {args.case_name}")
    best_model_path = best_model_matches[0]

    trainer = InvPhyTrainerWarp(
        data_path=f"{args.base_path}/{args.case_name}/final_data.pkl",
        base_dir=output_dir,
    )

    good_indices = []
    bad_failures = []
    cursor = 0
    final_verified = False
    while cursor < candidate_count or len(good_indices) >= args.batch_size_needed:
        if len(good_indices) >= args.batch_size_needed:
            selected = good_indices[: args.batch_size_needed]
            print(
                f"[filter] verifying selected {len(selected)} trajectories together"
            )
            ok, failure = run_candidate_batch(
                trainer,
                best_model_path,
                gaussians_path,
                group_tensor_cpu,
                selected,
                args,
            )
            if ok:
                final_verified = True
                good_indices = selected
                break

            print(f"[filter] selected batch failed final verification: {failure}")
            selected_good = isolate_good_indices(
                trainer,
                best_model_path,
                gaussians_path,
                group_tensor_cpu,
                selected,
                args,
                bad_failures,
            )
            if len(selected_good) == len(selected):
                raise RuntimeError(
                    "Selected trajectories pass in smaller recursive groups but fail "
                    "when run together. Try a smaller --batch_size_needed or inspect "
                    "the runtime batch-level failure."
                )
            remaining_good = good_indices[args.batch_size_needed :]
            good_indices = selected_good + remaining_good
            continue

        if cursor >= candidate_count:
            break

        end = min(candidate_count, cursor + args.filter_batch_size)
        batch_indices = list(range(cursor, end))
        print(
            f"[filter] testing candidates {cursor}:{end} "
            f"({len(good_indices)}/{args.batch_size_needed} good so far)"
        )
        good_indices.extend(
            isolate_good_indices(
                trainer,
                best_model_path,
                gaussians_path,
                group_tensor_cpu,
                batch_indices,
                args,
                bad_failures,
            )
        )
        cursor = end

    if len(good_indices) < args.batch_size_needed or not final_verified:
        raise RuntimeError(
            f"Only found {len(good_indices)} good trajectories, "
            f"but {args.batch_size_needed} are required. "
            "Increase --max_candidates or choose another case/replay range."
        )

    filtered_group = [group_arrays[idx] for idx in good_indices]
    output = dict(root)
    output["controller_points_group"] = filtered_group
    output["source_indices"] = good_indices
    output["bad_indices"] = sorted(
        {
            idx
            for failure in bad_failures
            for idx in failure.get("candidate_indices", [])
            if len(failure.get("candidate_indices", [])) == 1
        }
    )
    output["failures"] = bad_failures
    output["case_name"] = args.case_name
    output["replay_start"] = int(args.replay_start)
    output["replay_end"] = int(args.replay_end) if args.replay_end is not None else None
    output["created_at"] = datetime.now(timezone.utc).isoformat()
    output["input_pkl"] = args.input_pkl

    os.makedirs(os.path.dirname(args.output_pkl), exist_ok=True)
    with open(args.output_pkl, "wb") as handle:
        pickle.dump(output, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print(
        f"[filter] wrote {len(good_indices)} filtered trajectories to {args.output_pkl}"
    )


if __name__ == "__main__":
    main()
