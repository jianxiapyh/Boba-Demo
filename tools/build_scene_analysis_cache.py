#!/usr/bin/env python3

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys
import types

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _load_immersive_scene_module():
    package_name = "qqtt"
    package_root = REPO_ROOT / package_name
    if package_name not in sys.modules:
        package_module = types.ModuleType(package_name)
        package_module.__path__ = [str(package_root)]
        sys.modules[package_name] = package_module
    module_name = f"{package_name}.immersive_scene"
    if module_name in sys.modules:
        return sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(
        module_name,
        package_root / "immersive_scene.py",
    )
    if spec is None or spec.loader is None:
        raise ImportError("Could not load qqtt.immersive_scene")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _default_scene_assets_root() -> Path:
    return REPO_ROOT / "assets" / "scenes" / "ILLIXR_lab"


def main() -> None:
    immersive_scene = _load_immersive_scene_module()
    parser = argparse.ArgumentParser(
        description="Build the committed ILLIXR_lab scene-analysis cache artifact.",
    )
    parser.add_argument(
        "--scene-assets-root",
        type=Path,
        default=_default_scene_assets_root(),
        help="Path to the vendored ILLIXR_lab scene bundle or a parent directory containing it.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional explicit output path for the cache artifact.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=64,
        help="Dummy renderer width used while rebuilding the cache.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=64,
        help="Dummy renderer height used while rebuilding the cache.",
    )
    args = parser.parse_args()

    cache_path, debug = immersive_scene.build_illixr_scene_analysis_cache(
        scene_assets_root=args.scene_assets_root,
        output_path=args.output,
        width=int(args.width),
        height=int(args.height),
    )
    print(
        json.dumps(
            {
                "cache_path": str(cache_path),
                "debug": debug,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
