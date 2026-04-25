#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
LFS_POINTER_HEADER = "version https://git-lfs.github.com/spec/v1"
LFS_POINTER_OID_PREFIX = "oid sha256:"
REQUIRED_MANIFEST_PAYLOAD_KEYS = (
    "best_model",
    "calibrate",
    "final_data",
    "gaussian_ply",
    "metadata",
    "optimal_params",
)
OPTIONAL_MANIFEST_PAYLOAD_KEYS = ("game_course",)
LIST_MANIFEST_PAYLOAD_KEYS = ("tutorial_extra_slides",)
COMPAT_DEMO_CASE_ALIASES = {
    "hq_rope_0": "hq_rope",
}
KNOWN_PAYLOAD_FINGERPRINTS = {
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


@dataclass(frozen=True)
class AssetProblem:
    asset: ManifestAsset
    reason: str
    detail: str


def canonical_demo_case_name(case_name: str) -> str:
    case_key = str(case_name).strip().lower()
    return str(COMPAT_DEMO_CASE_ALIASES.get(case_key, case_key))


def list_demo_case_names(repo_root: Path) -> list[str]:
    assets_root = repo_root / "assets"
    return sorted(
        manifest_path.parent.name for manifest_path in assets_root.glob("*/manifest.json")
    )


def resolve_demo_case_manifest(repo_root: Path, case_name: str) -> tuple[str, Path, dict]:
    canonical_case = canonical_demo_case_name(case_name)
    manifest_path = repo_root / "assets" / canonical_case / "manifest.json"
    if not manifest_path.exists():
        available_cases = ", ".join(list_demo_case_names(repo_root))
        raise FileNotFoundError(
            f"Immersive demo manifest for case '{case_name}' was not found at {manifest_path}. "
            f"Available packaged cases: {available_cases}."
        )
    with open(manifest_path, "r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    return canonical_case, manifest_path.parent, manifest


def resolve_manifest_path(
    repo_root: Path,
    manifest_dir: Path,
    manifest: dict,
    key: str,
) -> Path:
    relative_path = manifest.get(key)
    if relative_path is None:
        raise KeyError(f"Manifest is missing required key: {key}")
    base_dir = repo_root if key == "config" else manifest_dir
    return (base_dir / str(relative_path)).resolve()


def resolve_case_manifest_assets(
    repo_root: Path,
    case_name: str,
    manifest_dir: Path,
    manifest: dict,
) -> list[ManifestAsset]:
    config_path = resolve_manifest_path(repo_root, manifest_dir, manifest, "config")
    if not config_path.exists():
        raise FileNotFoundError(
            f"Config file referenced by case '{case_name}' is missing: {config_path}"
        )

    assets: list[ManifestAsset] = []
    for key in REQUIRED_MANIFEST_PAYLOAD_KEYS:
        path = resolve_manifest_path(repo_root, manifest_dir, manifest, key)
        assets.append(
            ManifestAsset(
                case_name=case_name,
                key=key,
                path=path,
                repo_relative_path=str(path.relative_to(repo_root)),
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
                repo_relative_path=str(path.relative_to(repo_root)),
            )
        )

    for key in LIST_MANIFEST_PAYLOAD_KEYS:
        values = manifest.get(key, [])
        if values is None:
            values = []
        if not isinstance(values, list):
            raise TypeError(
                f"Manifest key '{key}' for case '{case_name}' must be a list, got {type(values).__name__}."
            )
        for index, relative_path in enumerate(values):
            path = (manifest_dir / str(relative_path)).resolve()
            assets.append(
                ManifestAsset(
                    case_name=case_name,
                    key=f"{key}[{index}]",
                    path=path,
                    repo_relative_path=str(path.relative_to(repo_root)),
                )
            )

    return assets


def read_lfs_pointer_metadata(path: Path) -> dict[str, str] | None:
    if not path.exists() or not path.is_file():
        return None
    try:
        with open(path, "r", encoding="utf-8", errors="strict") as handle:
            lines = [handle.readline().strip() for _ in range(3)]
    except UnicodeDecodeError:
        return None
    if not lines or lines[0] != LFS_POINTER_HEADER:
        return None

    metadata: dict[str, str] = {}
    if len(lines) > 1 and lines[1].startswith(LFS_POINTER_OID_PREFIX):
        metadata["sha256"] = lines[1][len(LFS_POINTER_OID_PREFIX) :]
    if len(lines) > 2 and lines[2].startswith("size "):
        metadata["size"] = lines[2].split(" ", 1)[1]
    return metadata


def read_file_prefix(path: Path, size: int) -> bytes:
    with open(path, "rb") as handle:
        return handle.read(size)


def describe_known_payload_hint(asset: ManifestAsset) -> str:
    fingerprint = KNOWN_PAYLOAD_FINGERPRINTS.get(asset.repo_relative_path)
    if fingerprint is None:
        return ""
    return (
        " Expected payload fingerprint: "
        f"sha256 {fingerprint['sha256']}, size {fingerprint['size']} bytes."
    )


def find_asset_problems(assets: list[ManifestAsset]) -> list[AssetProblem]:
    problems: list[AssetProblem] = []
    for asset in assets:
        if not asset.path.exists():
            problems.append(
                AssetProblem(
                    asset=asset,
                    reason="missing",
                    detail=f"Missing manifest payload: {asset.path}",
                )
            )
            continue

        pointer_metadata = read_lfs_pointer_metadata(asset.path)
        if pointer_metadata is not None:
            oid = pointer_metadata.get("sha256", "<unknown>")
            size = pointer_metadata.get("size", "<unknown>")
            problems.append(
                AssetProblem(
                    asset=asset,
                    reason="lfs_pointer",
                    detail=(
                        f"Git LFS pointer detected for {asset.path} "
                        f"(sha256 {oid}, size {size} bytes)."
                    ),
                )
            )
            continue

        if asset.must_start_with_ply:
            if read_file_prefix(asset.path, 4) != b"ply\n":
                problems.append(
                    AssetProblem(
                        asset=asset,
                        reason="invalid_ply",
                        detail=f"Gaussian PLY does not start with 'ply': {asset.path}",
                    )
                )

    return problems


def format_problem(problem: AssetProblem) -> str:
    suffix = describe_known_payload_hint(problem.asset)
    return (
        f"[{problem.asset.case_name}] {problem.asset.key}: "
        f"{problem.detail}{suffix}"
    )


def ensure_git_lfs_available(repo_root: Path) -> None:
    clean_env = os.environ.copy()
    clean_env.pop("LD_LIBRARY_PATH", None)
    probe = subprocess.run(
        ["git", "-C", str(repo_root), "lfs", "version"],
        capture_output=True,
        text=True,
        check=False,
        env=clean_env,
    )
    if probe.returncode != 0:
        stderr = probe.stderr.strip()
        raise DemoAssetValidationError(
            "git-lfs is required to hydrate demo payloads but is not available. "
            "Install git-lfs, run `git -C "
            f"{repo_root} lfs install --local`, and rerun the fetch tool. "
            f"git reported: {stderr or probe.stdout.strip() or 'git lfs unavailable'}"
        )


def lfs_pull_paths(repo_root: Path, repo_relative_paths: list[str]) -> None:
    if not repo_relative_paths:
        return
    ensure_git_lfs_available(repo_root)
    clean_env = os.environ.copy()
    clean_env.pop("LD_LIBRARY_PATH", None)
    command = [
        "git",
        "-C",
        str(repo_root),
        "lfs",
        "pull",
        f"--include={','.join(repo_relative_paths)}",
        "--exclude=",
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False, env=clean_env)
    if result.returncode != 0:
        stderr = result.stderr.strip()
        stdout = result.stdout.strip()
        raise DemoAssetValidationError(
            "git lfs pull failed while hydrating demo payloads. "
            f"Command: {' '.join(command)}. "
            f"{stderr or stdout or 'No output from git lfs pull.'}"
        )


def sha256sum(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_known_payload_fingerprint(asset: ManifestAsset) -> None:
    fingerprint = KNOWN_PAYLOAD_FINGERPRINTS.get(asset.repo_relative_path)
    if fingerprint is None:
        return
    actual_size = asset.path.stat().st_size
    expected_size = int(fingerprint["size"])
    if actual_size != expected_size:
        raise DemoAssetValidationError(
            f"{asset.path} has size {actual_size} bytes, expected {expected_size} bytes."
        )
    actual_sha = sha256sum(asset.path)
    if actual_sha != fingerprint["sha256"]:
        raise DemoAssetValidationError(
            f"{asset.path} has sha256 {actual_sha}, expected {fingerprint['sha256']}."
        )


def validate_demo_case_assets(
    repo_root: Path,
    case_name: str,
    manifest_dir: Path,
    manifest: dict,
) -> list[ManifestAsset]:
    assets = resolve_case_manifest_assets(repo_root, case_name, manifest_dir, manifest)
    problems = find_asset_problems(assets)
    if not problems:
        return assets

    fix_command = f"python tools/fetch_demo_case_assets.py --case {case_name}"
    formatted = "\n".join(f"- {format_problem(problem)}" for problem in problems)
    raise DemoAssetValidationError(
        "Demo assets are not ready for runtime.\n"
        f"{formatted}\n"
        f"Run `{fix_command}` from {repo_root} or rerun `bash env_install/RTX6000_env_install.sh`."
    )


def resolve_selected_cases(repo_root: Path, requested_cases: list[str] | None, select_all: bool) -> list[str]:
    if select_all or not requested_cases:
        return list_demo_case_names(repo_root)
    return [canonical_demo_case_name(case_name) for case_name in requested_cases]


def hydrate_demo_case_assets(
    repo_root: Path,
    case_names: list[str],
    allow_fetch: bool,
) -> list[ManifestAsset]:
    all_assets: list[ManifestAsset] = []
    fetch_candidates: list[str] = []
    unresolved_messages: list[str] = []

    for case_name in case_names:
        canonical_case, manifest_dir, manifest = resolve_demo_case_manifest(repo_root, case_name)
        assets = resolve_case_manifest_assets(repo_root, canonical_case, manifest_dir, manifest)
        all_assets.extend(assets)
        for problem in find_asset_problems(assets):
            unresolved_messages.append(format_problem(problem))
            if problem.reason in {"missing", "lfs_pointer"}:
                fetch_candidates.append(problem.asset.repo_relative_path)

    if unresolved_messages and not allow_fetch:
        raise DemoAssetValidationError(
            "Detected unresolved demo payloads:\n"
            + "\n".join(f"- {message}" for message in unresolved_messages)
        )

    unique_fetch_candidates = sorted(set(fetch_candidates))
    if unique_fetch_candidates:
        print(
            "Hydrating demo payloads with git-lfs:",
            ", ".join(unique_fetch_candidates),
            flush=True,
        )
        lfs_pull_paths(repo_root, unique_fetch_candidates)

    for case_name in case_names:
        canonical_case, manifest_dir, manifest = resolve_demo_case_manifest(repo_root, case_name)
        assets = validate_demo_case_assets(repo_root, canonical_case, manifest_dir, manifest)
        for asset in assets:
            verify_known_payload_fingerprint(asset)

    return all_assets


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Hydrate and validate packaged Boba-Demo case assets from Git LFS."
    )
    parser.add_argument(
        "--case",
        action="append",
        help="specific packaged case to validate or hydrate; repeat to pass multiple cases",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="scan every assets/*/manifest.json case in the repo",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="validate manifests without running `git lfs pull`",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    case_names = resolve_selected_cases(REPO_ROOT, args.case, args.all)
    if not case_names:
        raise DemoAssetValidationError("No demo case manifests were found under ./assets/.")

    assets = hydrate_demo_case_assets(
        repo_root=REPO_ROOT,
        case_names=case_names,
        allow_fetch=not args.check_only,
    )
    print(
        f"Validated {len(assets)} manifest-referenced payload entries across cases: "
        + ", ".join(case_names),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except DemoAssetValidationError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1)
