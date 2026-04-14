#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import sys
import types
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scene-assets-root",
        default="./assets/scenes",
        help="Root directory that contains the tracked ILLIXR_lab manifest.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    package = types.ModuleType("qqtt")
    package.__path__ = [str(repo_root / "qqtt")]
    sys.modules.setdefault("qqtt", package)

    module_path = repo_root / "qqtt" / "immersive_scene.py"
    spec = importlib.util.spec_from_file_location("qqtt.immersive_scene", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load immersive scene module from {module_path}")
    immersive_scene = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = immersive_scene
    spec.loader.exec_module(immersive_scene)
    ensure_illixr_lab_assets = getattr(
        immersive_scene,
        "ensure_illixr_lab_assets",
        immersive_scene.ensure_simple_lab_assets,
    )

    resolved = ensure_illixr_lab_assets(repo_root / args.scene_assets_root)
    print(f"Validated immersive ILLIXR_lab assets in {resolved}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
