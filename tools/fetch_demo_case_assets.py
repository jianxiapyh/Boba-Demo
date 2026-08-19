#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
PUBLIC_DEMO_CASES = ("rope_game", "sloth")
PUBLIC_SCENES = ("lab", "garden")
GARDEN_QUALITIES = ("auto", "full", "balanced", "performance")
LFS_POINTER_HEADER = "version https://git-lfs.github.com/spec/v1"
REQUIRED_MANIFEST_PAYLOAD_KEYS = (
    "best_model",
    "calibrate",
    "final_data",
    "gaussian_ply",
    "metadata",
    "optimal_params",
)
OPTIONAL_MANIFEST_PAYLOAD_KEYS = (
    "game_course",
    "asset_provenance",
    "asset_license",
)
LIST_MANIFEST_PAYLOAD_KEYS = ("tutorial_extra_slides",)
SHARED_TUTORIAL_SLIDES = (
    "controls_overview.png",
    "interaction_tips.png",
)
KNOWN_PAYLOAD_FINGERPRINTS = {
    "assets/rope_game/best_model.pth": {
        "sha256": "e48fc6e5170e370579bc7511d39f5d56a355fe9b2a9c2bca63891a3182241b72",
        "size": 120818,
    },
    "assets/rope_game/calibrate.pkl": {
        "sha256": "160962e646f60bdfd70a71793e11c8152a0d05543762999213d489ef50849f1d",
        "size": 601,
    },
    "assets/rope_game/final_data.pkl": {
        "sha256": "35167df19341465d567de0b94cdab95317fe547fd21dc45640c05b0e4e9a3794",
        "size": 1084852,
    },
    "assets/rope_game/metadata.json": {
        "sha256": "f95c83b25929e401ae98438bcba648e72b037d9cc4892f057b6e92a1b3bd8989",
        "size": 499,
    },
    "assets/rope_game/optimal_params.pkl": {
        "sha256": "38d2d93801d1daae1a48aba83b39638a9658528d9a0a69a37a6e02b04739e505",
        "size": 551,
    },
    "assets/rope_game/phystwin_rope.ply": {
        "sha256": "c552a93bd4a4fe14a940a4ab0be0602f81af2513da3d2cf78265dd28dc79527c",
        "size": 8650282,
    },
    "assets/rope_game/course_v1.json": {
        "sha256": "eac7a03b8392231edfcba99c2eebfa9eefd455f0a3a1736c7c741bd64516da01",
        "size": 787,
    },
    "assets/rope_game/tutorial_rope_game_goal.png": {
        "sha256": "8447810d581b9401e97514ebee7db44fb41aa086d3c53c0f0e8702d062f7fb42",
        "size": 71960,
    },
    "assets/sloth/best_model.pth": {
        "sha256": "b027af7baae371fd4e0ddbf33b7d7d9f919517081b26907bbe10cd2d2161d549",
        "size": 274034,
    },
    "assets/sloth/calibrate.pkl": {
        "sha256": "160962e646f60bdfd70a71793e11c8152a0d05543762999213d489ef50849f1d",
        "size": 601,
    },
    "assets/sloth/final_data.pkl": {
        "sha256": "fca68a1dec035f6252cf2cbe35717425f210124fb92d3b8f26d7b4cdef2d4635",
        "size": 16275526,
    },
    "assets/sloth/metadata.json": {
        "sha256": "dbf86424791736ce7c476ab9e2885f05d29a2228b32352e799d6836fd3970b47",
        "size": 500,
    },
    "assets/sloth/optimal_params.pkl": {
        "sha256": "bde43cb1d2bdeede2e0407f32cc1fe46e272cb1aaf19c72d8e73c5769472beec",
        "size": 551,
    },
    "assets/sloth/sloth.ply": {
        "sha256": "fc0301db3e5fd077d153e3bb2d68cf609db1ebc6968932101f34d731b6aec5d2",
        "size": 55991805,
    },
}


class DemoAssetValidationError(RuntimeError):
    pass


@dataclass(frozen=True)
class ManifestAsset:
    case_name: str
    key: str
    path: Path
    repo_relative_path: str
    must_start_with_ply: bool = False


def _repo_relative_path(repo_root: Path, path: Path) -> str:
    try:
        return str(path.resolve().relative_to(repo_root.resolve()))
    except ValueError as exc:
        raise DemoAssetValidationError(
            f"Manifest payload resolves outside the repository: {path}"
        ) from exc


def resolve_demo_case_manifest(repo_root: Path, case_name: str) -> tuple[str, Path, dict]:
    normalized_case = str(case_name).strip().lower()
    if normalized_case not in PUBLIC_DEMO_CASES:
        raise DemoAssetValidationError(
            f"Unsupported demo case '{case_name}'. Packaged cases: "
            f"{', '.join(PUBLIC_DEMO_CASES)}."
        )
    manifest_path = repo_root / "assets" / normalized_case / "manifest.json"
    if not manifest_path.is_file():
        raise DemoAssetValidationError(
            f"Missing manifest for packaged case '{normalized_case}': {manifest_path}"
        )
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    return normalized_case, manifest_path.parent, manifest


def resolve_manifest_path(
    repo_root: Path,
    manifest_dir: Path,
    manifest: dict,
    key: str,
) -> Path:
    relative_path = manifest.get(key)
    if relative_path is None:
        raise DemoAssetValidationError(f"Manifest is missing required key: {key}")
    base_dir = repo_root if key == "config" else manifest_dir
    path = (base_dir / str(relative_path)).resolve()
    _repo_relative_path(repo_root, path)
    return path


def resolve_case_manifest_assets(
    repo_root: Path,
    case_name: str,
    manifest_dir: Path,
    manifest: dict,
) -> list[ManifestAsset]:
    config_path = resolve_manifest_path(repo_root, manifest_dir, manifest, "config")
    assets = [
        ManifestAsset(
            case_name=case_name,
            key="config",
            path=config_path,
            repo_relative_path=_repo_relative_path(repo_root, config_path),
        )
    ]

    for key in REQUIRED_MANIFEST_PAYLOAD_KEYS:
        path = resolve_manifest_path(repo_root, manifest_dir, manifest, key)
        assets.append(
            ManifestAsset(
                case_name=case_name,
                key=key,
                path=path,
                repo_relative_path=_repo_relative_path(repo_root, path),
                must_start_with_ply=(key == "gaussian_ply"),
            )
        )

    for key in OPTIONAL_MANIFEST_PAYLOAD_KEYS:
        if manifest.get(key) is None:
            continue
        path = resolve_manifest_path(repo_root, manifest_dir, manifest, key)
        assets.append(
            ManifestAsset(
                case_name=case_name,
                key=key,
                path=path,
                repo_relative_path=_repo_relative_path(repo_root, path),
            )
        )

    for key in LIST_MANIFEST_PAYLOAD_KEYS:
        values = manifest.get(key, [])
        if not isinstance(values, list):
            raise DemoAssetValidationError(
                f"Manifest key '{key}' must be a list, got {type(values).__name__}."
            )
        for index, relative_path in enumerate(values):
            path = (manifest_dir / str(relative_path)).resolve()
            assets.append(
                ManifestAsset(
                    case_name=case_name,
                    key=f"{key}[{index}]",
                    path=path,
                    repo_relative_path=_repo_relative_path(repo_root, path),
                )
            )

    return assets


def resolve_shared_runtime_assets(
    repo_root: Path,
    scene_name: str = "lab",
    garden_quality: str = "balanced",
) -> list[ManifestAsset]:
    scene_name = str(scene_name).strip().lower()
    if scene_name not in PUBLIC_SCENES:
        raise DemoAssetValidationError(
            f"Unsupported scene {scene_name!r}; expected one of {PUBLIC_SCENES}."
        )
    shared_assets = []
    tutorial_dir = repo_root / "assets" / "tutorial"
    for slide_name in SHARED_TUTORIAL_SLIDES:
        path = (tutorial_dir / slide_name).resolve()
        shared_assets.append(
            ManifestAsset(
                case_name="shared",
                key=f"tutorial:{slide_name}",
                path=path,
                repo_relative_path=_repo_relative_path(repo_root, path),
            )
        )

    if scene_name == "garden":
        try:
            from qqtt.garden_assets import validate_garden_runtime_selection

            garden_paths = validate_garden_runtime_selection(
                repo_root,
                garden_quality,
            )
        except Exception as exc:
            raise DemoAssetValidationError(
                f"Garden scene assets are not ready: {exc}\n"
                "Install them once with:\n"
                "  conda run -n phystwin-cu132 env PYTHONNOUSERSITE=1 "
                "python tools/fetch_demo_case_assets.py --scene garden --fetch"
            ) from exc
        for index, path in enumerate(garden_paths):
            shared_assets.append(
                ManifestAsset(
                    case_name="shared",
                    key=f"garden[{index}]",
                    path=Path(path).resolve(),
                    repo_relative_path=_repo_relative_path(repo_root, Path(path)),
                    must_start_with_ply=(Path(path).suffix.lower() == ".ply"),
                )
            )
        return shared_assets

    scene_dir = repo_root / "assets" / "scenes" / "ILLIXR_lab"
    scene_manifest_path = scene_dir / "manifest.json"
    shared_assets.append(
        ManifestAsset(
            case_name="shared",
            key="scene_manifest",
            path=scene_manifest_path,
            repo_relative_path=_repo_relative_path(repo_root, scene_manifest_path),
        )
    )
    if not scene_manifest_path.is_file():
        return shared_assets
    with scene_manifest_path.open("r", encoding="utf-8") as handle:
        scene_manifest = json.load(handle)
    scene_paths = {
        "scene_model": scene_manifest.get("scene_model"),
        "scene_material": scene_manifest.get("scene_material"),
        "scene_analysis_cache": scene_manifest.get("scene_analysis_cache"),
    }
    textures = scene_manifest.get("textures", [])
    if not isinstance(textures, list):
        raise DemoAssetValidationError("ILLIXR_lab manifest key 'textures' must be a list.")
    scene_paths.update({f"texture[{index}]": value for index, value in enumerate(textures)})
    for key, relative_path in scene_paths.items():
        if not relative_path:
            raise DemoAssetValidationError(
                f"ILLIXR_lab manifest is missing required asset entry: {key}"
            )
        path = (scene_dir / str(relative_path)).resolve()
        shared_assets.append(
            ManifestAsset(
                case_name="shared",
                key=key,
                path=path,
                repo_relative_path=_repo_relative_path(repo_root, path),
            )
        )
    return shared_assets


def _read_lfs_pointer(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        with path.open("r", encoding="utf-8", errors="strict") as handle:
            return handle.readline().strip() == LFS_POINTER_HEADER
    except UnicodeDecodeError:
        return False


def _sha256sum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_asset(asset: ManifestAsset) -> None:
    if not asset.path.is_file():
        raise DemoAssetValidationError(
            f"[{asset.case_name}] {asset.key}: missing tracked asset {asset.path}"
        )
    if _read_lfs_pointer(asset.path):
        raise DemoAssetValidationError(
            f"[{asset.case_name}] {asset.key}: Git LFS pointer found instead of payload at "
            f"{asset.path}. Install Git LFS and run 'git lfs pull'."
        )
    if asset.must_start_with_ply:
        with asset.path.open("rb") as handle:
            if handle.read(4) != b"ply\n":
                raise DemoAssetValidationError(
                    f"[{asset.case_name}] {asset.key}: invalid Gaussian PLY header at {asset.path}"
                )
    fingerprint = KNOWN_PAYLOAD_FINGERPRINTS.get(asset.repo_relative_path)
    if fingerprint is None:
        return
    actual_size = asset.path.stat().st_size
    if actual_size != int(fingerprint["size"]):
        raise DemoAssetValidationError(
            f"[{asset.case_name}] {asset.key}: {asset.path} has size {actual_size}; "
            f"expected {fingerprint['size']}"
        )
    actual_sha256 = _sha256sum(asset.path)
    if actual_sha256 != fingerprint["sha256"]:
        raise DemoAssetValidationError(
            f"[{asset.case_name}] {asset.key}: {asset.path} has sha256 {actual_sha256}; "
            f"expected {fingerprint['sha256']}"
        )


def validate_demo_case_assets(
    repo_root: Path,
    case_name: str,
    manifest_dir: Path,
    manifest: dict,
    scene_name: str = "lab",
    garden_quality: str = "balanced",
) -> list[ManifestAsset]:
    normalized_case = str(case_name).strip().lower()
    if normalized_case not in PUBLIC_DEMO_CASES:
        raise DemoAssetValidationError(
            f"Unsupported demo case '{case_name}'. Packaged cases: "
            f"{', '.join(PUBLIC_DEMO_CASES)}."
        )
    expected_manifest_dir = (repo_root / "assets" / normalized_case).resolve()
    if manifest_dir.resolve() != expected_manifest_dir:
        raise DemoAssetValidationError(
            f"{normalized_case} manifest must live at {expected_manifest_dir}, "
            f"got {manifest_dir.resolve()}"
        )
    assets = resolve_case_manifest_assets(
        repo_root,
        normalized_case,
        manifest_dir,
        manifest,
    )
    assets.extend(
        resolve_shared_runtime_assets(
            repo_root,
            scene_name=scene_name,
            garden_quality=garden_quality,
        )
    )
    for asset in assets:
        _validate_asset(asset)
    return assets


def validate_all_demo_assets(
    repo_root: Path,
    scene_name: str = "lab",
    garden_quality: str = "balanced",
) -> list[ManifestAsset]:
    assets = []
    shared_paths = set()
    for case_name in PUBLIC_DEMO_CASES:
        canonical_case, manifest_dir, manifest = resolve_demo_case_manifest(
            repo_root,
            case_name,
        )
        case_assets = validate_demo_case_assets(
            repo_root,
            canonical_case,
            manifest_dir,
            manifest,
            scene_name=scene_name,
            garden_quality=garden_quality,
        )
        for asset in case_assets:
            if asset.case_name == "shared":
                if asset.repo_relative_path in shared_paths:
                    continue
                shared_paths.add(asset.repo_relative_path)
            assets.append(asset)
    return assets


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate packaged Rope/Sloth and shared Quest scene assets."
    )
    parser.add_argument(
        "--case",
        choices=PUBLIC_DEMO_CASES + ("all",),
        default="all",
        help="packaged demo case, or all selectable cases",
    )
    parser.add_argument(
        "--scene",
        choices=PUBLIC_SCENES,
        default="lab",
        help="validate assets for only the selected launch-time scene",
    )
    parser.add_argument(
        "--garden-quality",
        choices=GARDEN_QUALITIES,
        default="balanced",
        help="Garden runtime quality tier to prepare or validate",
    )
    parser.add_argument(
        "--fetch",
        action="store_true",
        help="download and deterministically prepare Garden assets when needed",
    )
    parser.add_argument(
        "--garden-archive",
        type=Path,
        default=None,
        help="optional already-downloaded official models.zip archive",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.fetch:
        if args.scene != "garden":
            raise DemoAssetValidationError("--fetch is currently supported only with --scene garden.")
        try:
            from qqtt.garden_assets import fetch_and_prepare_garden

            last_percent = [-1]

            def report_progress(downloaded: int, total: int | None) -> None:
                if total is None or total <= 0:
                    return
                percent = int(100 * downloaded / total)
                if percent >= last_percent[0] + 5:
                    print(f"Garden archive download: {percent}%", flush=True)
                    last_percent[0] = percent

            outputs = fetch_and_prepare_garden(
                REPO_ROOT,
                archive_override=args.garden_archive,
                progress_callback=report_progress,
            )
            print(
                "Prepared Garden tiers: "
                + ", ".join(
                    f"{quality} ({metadata['gaussian_count']:,} Gaussians, "
                    f"{metadata['spatial_chunk_count']:,} spatial chunks)"
                    for quality, metadata in outputs.items()
                ),
                flush=True,
            )
        except Exception as exc:
            raise DemoAssetValidationError(f"Garden setup failed: {exc}") from exc
    if args.case == "all":
        assets = validate_all_demo_assets(
            REPO_ROOT,
            scene_name=args.scene,
            garden_quality=args.garden_quality,
        )
    else:
        case_name, manifest_dir, manifest = resolve_demo_case_manifest(
            REPO_ROOT,
            args.case,
        )
        assets = validate_demo_case_assets(
            REPO_ROOT,
            case_name,
            manifest_dir,
            manifest,
            scene_name=args.scene,
            garden_quality=args.garden_quality,
        )
    print(
        f"Validated {len(assets)} packaged demo/shared runtime asset entries.",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except DemoAssetValidationError as exc:
        raise SystemExit(str(exc)) from exc
