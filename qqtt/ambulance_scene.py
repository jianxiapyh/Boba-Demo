"""Insta360 ambulance Gaussian scene integration."""

from __future__ import annotations

import gc
import json
import math
import struct
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from .garden_assets import GardenAssetError, sha256_file, sh_rotation_matrix
from .garden_scene import (
    GardenSceneRenderer,
    _matrix_to_quaternion_wxyz_torch,
    _normalize,
)
from .immersive_scene import (
    SimpleLabLayout,
    normalize_immersive_start_posture,
)
from .sog_loader import load_sog_gaussian_model, read_sog_metadata


AMBULANCE_SCENE_NAME = "ambulance"
AMBULANCE_MAX_PROJECTED_RADIUS_PX = 1024.0
AMBULANCE_MANIFEST_RELATIVE_PATH = Path(
    "assets/scenes/ambulance_insta360/manifest.json"
)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise GardenAssetError(f"Unable to read ambulance JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise GardenAssetError(f"Ambulance JSON must contain an object: {path}")
    return value


def _resolve_repo_path(repo_root: Path, relative_path: str | Path) -> Path:
    path = (repo_root / Path(relative_path)).resolve()
    try:
        path.relative_to(repo_root)
    except ValueError as exc:
        raise GardenAssetError(
            f"Ambulance asset path escapes the repository: {path}"
        ) from exc
    return path


def load_ambulance_manifest(
    repo_root: str | Path,
) -> tuple[Path, dict[str, Any], Path, dict[str, Any]]:
    repo_root = Path(repo_root).resolve()
    manifest_path = repo_root / AMBULANCE_MANIFEST_RELATIVE_PATH
    if not manifest_path.is_file():
        raise GardenAssetError(
            f"Ambulance scene manifest is missing: {manifest_path}"
        )
    manifest = _read_json(manifest_path)
    if int(manifest.get("schema_version", -1)) != 1:
        raise GardenAssetError("Unsupported ambulance manifest schema.")
    if str(manifest.get("scene_name", "")).strip().lower() != AMBULANCE_SCENE_NAME:
        raise GardenAssetError("Ambulance manifest has the wrong scene_name.")
    calibration_path = _resolve_repo_path(repo_root, manifest["calibration"])
    calibration = _read_json(calibration_path)
    if int(calibration.get("schema_version", -1)) != 1:
        raise GardenAssetError("Unsupported ambulance calibration schema.")
    return manifest_path, manifest, calibration_path, calibration


def ambulance_startup_gaze_pitch_down_degrees(
    repo_root: str | Path,
    start_posture: str = "seated",
) -> float:
    """Return the authored initial headset pitch for the selected view."""

    _, _, _, calibration = load_ambulance_manifest(repo_root)
    start_posture = normalize_immersive_start_posture(start_posture)
    view = calibration.get(f"{start_posture}_view")
    pitch_down_degrees = float(
        view.get("startup_gaze_pitch_down_degrees", float("nan"))
        if isinstance(view, dict)
        else float("nan")
    )
    if (
        not np.isfinite(pitch_down_degrees)
        or pitch_down_degrees < 0.0
        or pitch_down_degrees > 60.0
    ):
        raise GardenAssetError(
            f"Ambulance {start_posture} startup gaze pitch must be between 0 "
            "and 60 degrees down."
        )
    return pitch_down_degrees


def _interaction_surface_frame_canonical(
    calibration: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    interaction_surface = calibration.get("interaction_surface")
    if not isinstance(interaction_surface, dict):
        raise GardenAssetError("Ambulance calibration has no interaction surface.")
    plane = np.asarray(
        interaction_surface.get("canonical_plane_z_down_from_xy"),
        dtype=np.float32,
    )
    if plane.shape != (3,) or not np.all(np.isfinite(plane)):
        raise GardenAssetError(
            "Ambulance interaction surface must provide a finite canonical "
            "plane z=a*x+b*y+c."
        )
    center_xy = np.asarray(
        interaction_surface.get("canonical_center_xy_m", [0.0, 0.0]),
        dtype=np.float32,
    )
    axis_u_xy = np.asarray(
        interaction_surface.get("canonical_axis_u_xy", [1.0, 0.0]),
        dtype=np.float32,
    )
    if center_xy.shape != (2,) or not np.all(np.isfinite(center_xy)):
        raise GardenAssetError(
            "Ambulance interaction surface center must contain two finite "
            "canonical-plane coordinates."
        )
    if (
        axis_u_xy.shape != (2,)
        or not np.all(np.isfinite(axis_u_xy))
        or float(np.linalg.norm(axis_u_xy)) <= 1.0e-6
    ):
        raise GardenAssetError(
            "Ambulance interaction surface long axis must contain two finite "
            "nonzero canonical-plane coordinates."
        )
    axis_u_xy = axis_u_xy / float(np.linalg.norm(axis_u_xy))
    slope_x, slope_y, plane_intercept = [float(value) for value in plane]
    center_z = (
        slope_x * float(center_xy[0])
        + slope_y * float(center_xy[1])
        + plane_intercept
    )
    center = np.array(
        [float(center_xy[0]), float(center_xy[1]), center_z],
        dtype=np.float32,
    )
    normal = _normalize(
        np.array([slope_x, slope_y, -1.0], dtype=np.float32),
        np.array([0.0, 0.0, -1.0], dtype=np.float32),
    )
    axis_u = _normalize(
        np.array(
            [
                float(axis_u_xy[0]),
                float(axis_u_xy[1]),
                slope_x * float(axis_u_xy[0])
                + slope_y * float(axis_u_xy[1]),
            ],
            dtype=np.float32,
        ),
        np.array([1.0, 0.0, 0.0], dtype=np.float32),
    )
    axis_v = _normalize(
        np.cross(axis_u, normal),
        np.array([0.0, 1.0, 0.0], dtype=np.float32),
    )
    return center, normal, axis_u, axis_v


def _interaction_surface_collision_heightfield(
    calibration: dict[str, Any],
) -> tuple[int, int, np.ndarray, str, float]:
    interaction_surface = calibration.get("interaction_surface")
    heightfield = (
        interaction_surface.get("collision_heightfield")
        if isinstance(interaction_surface, dict)
        else None
    )
    if not isinstance(heightfield, dict):
        raise GardenAssetError(
            "Ambulance interaction surface has no collision heightfield."
        )
    if int(heightfield.get("schema_version", -1)) != 2:
        raise GardenAssetError(
            "Ambulance collision heightfield must use schema version 2."
        )
    cells_u = int(heightfield.get("cells_u", 0))
    cells_v = int(heightfield.get("cells_v", 0))
    if (
        not 4 <= cells_u <= 128
        or not 4 <= cells_v <= 128
        or cells_u % 2 != 0
        or cells_v % 2 != 0
    ):
        raise GardenAssetError(
            "Ambulance collision heightfield dimensions must be even and between "
            "4 and 128."
        )
    footprint = heightfield.get("footprint")
    if (
        not isinstance(footprint, dict)
        or str(footprint.get("kind", "")).strip().lower() != "capsule"
    ):
        raise GardenAssetError(
            "Ambulance collision heightfield must use a capsule footprint."
        )
    surface_size = np.asarray(
        interaction_surface.get("size_m"),
        dtype=np.float32,
    )
    if (
        surface_size.shape != (3,)
        or not np.all(np.isfinite(surface_size))
        or float(np.min(surface_size)) <= 0.0
        or float(surface_size[0]) <= float(surface_size[1])
    ):
        raise GardenAssetError(
            "Ambulance capsule surface size must contain positive length, "
            "width, and depth values, with length greater than width."
        )
    profile = heightfield.get("profile")
    if (
        not isinstance(profile, dict)
        or str(profile.get("kind", "")).strip().lower() != "convex_crown"
    ):
        raise GardenAssetError(
            "Ambulance collision heightfield must use a convex_crown profile."
        )
    center_raise = float(
        profile.get("center_above_fitted_plane_m", float("nan"))
    )
    length_edge_drop = float(
        profile.get("length_edge_drop_m", float("nan"))
    )
    width_edge_drop = float(
        profile.get("width_edge_drop_m", float("nan"))
    )
    length_exponent = float(profile.get("length_exponent", float("nan")))
    width_exponent = float(profile.get("width_exponent", float("nan")))
    edge_round_radius = float(
        heightfield.get("edge_round_radius_m", float("nan"))
    )
    profile_values = np.asarray(
        [
            center_raise,
            length_edge_drop,
            width_edge_drop,
            length_exponent,
            width_exponent,
            edge_round_radius,
        ],
        dtype=np.float64,
    )
    if (
        not np.all(np.isfinite(profile_values))
        or center_raise <= 0.0
        or length_edge_drop <= 0.0
        or width_edge_drop <= 0.0
        or length_exponent < 2.0
        or width_exponent < 2.0
        or edge_round_radius <= 0.0
        or edge_round_radius >= 0.25 * float(surface_size[1])
    ):
        raise GardenAssetError(
            "Ambulance collision crown parameters must be finite and positive, "
            "with profile exponents of at least 2 and a valid rounded edge."
        )
    normalized_u = np.linspace(-1.0, 1.0, cells_u + 1, dtype=np.float32)
    normalized_v = np.linspace(-1.0, 1.0, cells_v + 1, dtype=np.float32)
    grid_u, grid_v = np.meshgrid(normalized_u, normalized_v)
    offsets = (
        center_raise
        - length_edge_drop * np.abs(grid_u) ** length_exponent
        - width_edge_drop * np.abs(grid_v) ** width_exponent
    ).astype(np.float32)
    maximum_offset = float(
        heightfield.get(
            "max_normal_offset_from_fitted_plane_m",
            float("nan"),
        )
    )
    if (
        not np.isfinite(maximum_offset)
        or maximum_offset <= 0.0
        or float(np.max(np.abs(offsets))) > maximum_offset
    ):
        raise GardenAssetError(
            "Ambulance collision crown exceeds its declared normal-offset bound."
        )
    half_u = 0.5 * float(surface_size[0])
    half_v = 0.5 * float(surface_size[1])
    capsule_spine_half_length = half_u - half_v
    capsule_end_u = np.maximum(
        np.abs(grid_u * half_u) - capsule_spine_half_length,
        0.0,
    )
    capsule_v = grid_v * half_v
    inside_footprint = (
        capsule_end_u * capsule_end_u + capsule_v * capsule_v
        <= half_v * half_v + 1.0e-6
    )
    if float(np.min(offsets[inside_footprint], initial=0.0)) < -1.0e-6:
        raise GardenAssetError(
            "Ambulance collision crown must remain at or above the fitted plane."
        )
    return (
        cells_u,
        cells_v,
        np.ascontiguousarray(offsets),
        "capsule",
        edge_round_radius,
    )


def _stretcher_detail_collision_spec(
    calibration: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Validate the source-mesh-derived stretcher proxy description."""

    proxy = calibration.get("stretcher_detail_collision_proxy")
    if not isinstance(proxy, dict) or int(proxy.get("schema_version", -1)) != 2:
        raise GardenAssetError(
            "Ambulance calibration has no supported stretcher detail collision proxy."
        )
    if (
        str(proxy.get("coordinate_frame", "")).strip()
        != "mattress_collision_center_local_uvw_m"
    ):
        raise GardenAssetError(
            "Ambulance stretcher detail proxy has the wrong coordinate frame."
        )
    source_mesh_sha256 = str(proxy.get("source_mesh_sha256", "")).strip().lower()
    if (
        len(source_mesh_sha256) != 64
        or any(character not in "0123456789abcdef" for character in source_mesh_sha256)
    ):
        raise GardenAssetError(
            "Ambulance stretcher detail proxy has an invalid source SHA-256."
        )
    raw_components = proxy.get("components")
    if (
        not isinstance(raw_components, list)
        or not 1 <= len(raw_components) <= 16
    ):
        raise GardenAssetError(
            "Ambulance stretcher detail proxy must contain 1 to 16 components."
        )

    components: list[dict[str, Any]] = []
    component_names: set[str] = set()
    for index, raw_component in enumerate(raw_components):
        if not isinstance(raw_component, dict):
            raise GardenAssetError(
                f"Ambulance stretcher detail component {index} must be an object."
            )
        name = str(raw_component.get("name", "")).strip()
        vertex_count = int(raw_component.get("vertex_count", 0))
        face_count = int(raw_component.get("face_count", 0))
        if not name or name in component_names:
            raise GardenAssetError(
                "Ambulance stretcher detail component names must be unique and "
                "non-empty."
            )
        if (
            vertex_count < 3
            or face_count < 1
            or vertex_count > 100_000
            or face_count > 200_000
        ):
            raise GardenAssetError(
                f"Ambulance stretcher detail component {name!r} has invalid "
                "vertex or face counts."
            )
        component_names.add(name)
        components.append(
            {
                "name": name,
                "vertex_count": vertex_count,
                "face_count": face_count,
            }
        )

    raw_contact = proxy.get("contact")
    if not isinstance(raw_contact, dict):
        raise GardenAssetError(
            "Ambulance stretcher detail proxy must contain contact parameters."
        )
    raw_substep_interval = raw_contact.get("substep_interval", 1)
    if isinstance(raw_substep_interval, bool):
        raise GardenAssetError(
            "Ambulance stretcher detail proxy substep interval is invalid."
        )
    try:
        substep_interval = int(raw_substep_interval)
    except (TypeError, ValueError, OverflowError) as exc:
        raise GardenAssetError(
            "Ambulance stretcher detail proxy substep interval is invalid."
        ) from exc
    if substep_interval < 1 or substep_interval != raw_substep_interval:
        raise GardenAssetError(
            "Ambulance stretcher detail proxy substep interval is invalid."
        )
    contact = {
        "two_sided": bool(raw_contact.get("two_sided", False)),
        "substep_interval": substep_interval,
        "query_distance_m": float(
            raw_contact.get("query_distance_m", float("nan"))
        ),
        "winding_accuracy": float(
            raw_contact.get("winding_accuracy", float("nan"))
        ),
        "winding_threshold": float(
            raw_contact.get("winding_threshold", float("nan"))
        ),
        "margin_m": float(raw_contact.get("margin_m", float("nan"))),
        "friction": float(raw_contact.get("friction", float("nan"))),
        "restitution": float(
            raw_contact.get("restitution", float("nan"))
        ),
    }
    if (
        not contact["two_sided"]
        or not all(
            np.isfinite(value)
            for key, value in contact.items()
            if key not in {"two_sided", "substep_interval"}
        )
        or contact["query_distance_m"] <= 0.0
        or contact["winding_accuracy"] <= 0.0
        or not 0.0 <= contact["winding_threshold"] <= 1.0
        or contact["margin_m"] < 0.0
        or not 0.0 <= contact["friction"] <= 2.0
        or not 0.0 <= contact["restitution"] <= 1.0
    ):
        raise GardenAssetError(
            "Ambulance stretcher detail proxy contact parameters are invalid."
        )
    return components, contact


def _glb_accessor_array(
    document: dict[str, Any],
    binary_chunk: bytes,
    accessor_index: int,
    *,
    component_type: int,
    accessor_type: str,
    columns: int,
) -> np.ndarray:
    try:
        accessor = document["accessors"][accessor_index]
        buffer_view = document["bufferViews"][accessor["bufferView"]]
    except (KeyError, IndexError, TypeError) as exc:
        raise GardenAssetError("Ambulance collision GLB has an invalid accessor.") from exc
    if (
        int(accessor.get("componentType", -1)) != component_type
        or str(accessor.get("type", "")) != accessor_type
        or bool(accessor.get("normalized", False))
        or "sparse" in accessor
        or int(buffer_view.get("buffer", -1)) != 0
        or "byteStride" in buffer_view
    ):
        raise GardenAssetError(
            "Ambulance collision GLB must use tightly packed float positions "
            "and unsigned-int triangle indices."
        )
    count = int(accessor.get("count", 0))
    if count <= 0:
        raise GardenAssetError("Ambulance collision GLB has an empty accessor.")
    dtype = np.dtype("<f4" if component_type == 5126 else "<u4")
    byte_offset = int(buffer_view.get("byteOffset", 0)) + int(
        accessor.get("byteOffset", 0)
    )
    value_count = count * columns
    byte_length = value_count * dtype.itemsize
    view_end = int(buffer_view.get("byteOffset", 0)) + int(
        buffer_view.get("byteLength", 0)
    )
    if (
        byte_offset < 0
        or byte_offset + byte_length > view_end
        or byte_offset + byte_length > len(binary_chunk)
    ):
        raise GardenAssetError(
            "Ambulance collision GLB accessor exceeds its binary buffer."
        )
    array = np.frombuffer(
        binary_chunk,
        dtype=dtype,
        count=value_count,
        offset=byte_offset,
    )
    return np.ascontiguousarray(array.reshape(count, columns))


@lru_cache(maxsize=4)
def _read_stretcher_collision_glb_cached(
    path_text: str,
    file_size: int,
    modified_time_ns: int,
) -> tuple[dict[str, Any], ...]:
    del file_size, modified_time_ns
    path = Path(path_text)
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise GardenAssetError(
            f"Unable to read Ambulance collision proxy {path}: {exc}"
        ) from exc
    if len(payload) < 20:
        raise GardenAssetError("Ambulance collision proxy is not a valid GLB.")
    magic, version, declared_length = struct.unpack_from("<4sII", payload, 0)
    if magic != b"glTF" or version != 2 or declared_length != len(payload):
        raise GardenAssetError("Ambulance collision proxy has an invalid GLB header.")
    json_chunk = None
    binary_chunk = None
    offset = 12
    while offset < len(payload):
        if offset + 8 > len(payload):
            raise GardenAssetError("Ambulance collision GLB has a truncated chunk.")
        chunk_length, chunk_type = struct.unpack_from("<II", payload, offset)
        offset += 8
        chunk_end = offset + chunk_length
        if chunk_end > len(payload):
            raise GardenAssetError("Ambulance collision GLB chunk exceeds the file.")
        chunk = payload[offset:chunk_end]
        if chunk_type == 0x4E4F534A:
            json_chunk = chunk
        elif chunk_type == 0x004E4942:
            binary_chunk = chunk
        offset = chunk_end
    if json_chunk is None or binary_chunk is None:
        raise GardenAssetError(
            "Ambulance collision GLB must contain JSON and binary chunks."
        )
    try:
        document = json.loads(json_chunk.decode("utf-8").rstrip(" \t\r\n\x00"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise GardenAssetError("Ambulance collision GLB JSON is invalid.") from exc
    if not isinstance(document, dict):
        raise GardenAssetError("Ambulance collision GLB JSON must be an object.")
    if any(
        key in document
        for key in ("materials", "textures", "images", "samplers")
    ):
        raise GardenAssetError(
            "Ambulance collision GLB must not contain rendering resources."
        )
    asset = document.get("asset")
    extras = asset.get("extras") if isinstance(asset, dict) else None
    if (
        not isinstance(asset, dict)
        or str(asset.get("version", "")) != "2.0"
        or not isinstance(extras, dict)
        or extras.get("coordinate_frame")
        != "mattress_collision_center_local_uvw_m"
        or extras.get("vertex_attributes") != ["POSITION"]
    ):
        raise GardenAssetError(
            "Ambulance collision GLB has unsupported asset metadata."
        )
    embedded_source_sha256 = str(extras.get("source_sha256", "")).strip().lower()
    if (
        len(embedded_source_sha256) != 64
        or any(
            character not in "0123456789abcdef"
            for character in embedded_source_sha256
        )
    ):
        raise GardenAssetError(
            "Ambulance collision GLB has an invalid source checksum."
        )
    buffers = document.get("buffers")
    if (
        not isinstance(buffers, list)
        or len(buffers) != 1
        or "uri" in buffers[0]
        or int(buffers[0].get("byteLength", -1)) > len(binary_chunk)
    ):
        raise GardenAssetError(
            "Ambulance collision GLB must have one embedded binary buffer."
        )
    scenes = document.get("scenes")
    nodes = document.get("nodes")
    meshes = document.get("meshes")
    scene_index = int(document.get("scene", -1))
    if (
        not isinstance(scenes, list)
        or not isinstance(nodes, list)
        or not isinstance(meshes, list)
        or not 0 <= scene_index < len(scenes)
    ):
        raise GardenAssetError("Ambulance collision GLB scene graph is invalid.")
    scene_nodes = scenes[scene_index].get("nodes")
    if not isinstance(scene_nodes, list) or not scene_nodes:
        raise GardenAssetError("Ambulance collision GLB scene is empty.")

    components: list[dict[str, Any]] = []
    component_names: set[str] = set()
    for node_index in scene_nodes:
        try:
            node = nodes[int(node_index)]
            mesh = meshes[int(node["mesh"])]
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise GardenAssetError(
                "Ambulance collision GLB contains an invalid mesh node."
            ) from exc
        if any(key in node for key in ("matrix", "translation", "rotation", "scale")):
            raise GardenAssetError(
                "Ambulance collision GLB mesh nodes must use identity transforms."
            )
        name = str(node.get("name", "")).strip()
        primitives = mesh.get("primitives") if isinstance(mesh, dict) else None
        if (
            not name
            or name in component_names
            or not isinstance(primitives, list)
            or len(primitives) != 1
        ):
            raise GardenAssetError(
                "Ambulance collision GLB components must be uniquely named "
                "single-primitive meshes."
            )
        primitive = primitives[0]
        attributes = primitive.get("attributes")
        if (
            attributes is None
            or set(attributes) != {"POSITION"}
            or "material" in primitive
            or "targets" in primitive
            or int(primitive.get("mode", 4)) != 4
            or "indices" not in primitive
        ):
            raise GardenAssetError(
                "Ambulance collision GLB primitives may contain only POSITION "
                "and triangle indices."
            )
        vertices = _glb_accessor_array(
            document,
            binary_chunk,
            int(attributes["POSITION"]),
            component_type=5126,
            accessor_type="VEC3",
            columns=3,
        ).astype(np.float32, copy=False)
        indices = _glb_accessor_array(
            document,
            binary_chunk,
            int(primitive["indices"]),
            component_type=5125,
            accessor_type="SCALAR",
            columns=1,
        ).reshape(-1)
        if (
            len(indices) % 3 != 0
            or int(indices.min(initial=0)) < 0
            or int(indices.max(initial=0)) >= len(vertices)
            or not np.isfinite(vertices).all()
        ):
            raise GardenAssetError(
                f"Ambulance collision component {name!r} has invalid geometry."
            )
        faces = np.ascontiguousarray(indices.reshape(-1, 3), dtype=np.int32)
        triangle_vertices = vertices[faces]
        twice_area = np.linalg.norm(
            np.cross(
                triangle_vertices[:, 1] - triangle_vertices[:, 0],
                triangle_vertices[:, 2] - triangle_vertices[:, 0],
            ),
            axis=1,
        )
        if float(twice_area.min()) <= 1.0e-10:
            raise GardenAssetError(
                f"Ambulance collision component {name!r} has degenerate triangles."
            )
        component_names.add(name)
        components.append(
            {
                "name": name,
                "vertices": vertices,
                "faces": faces,
                "source_sha256": embedded_source_sha256,
            }
        )
    return tuple(components)


def _read_stretcher_collision_glb(path: Path) -> tuple[dict[str, Any], ...]:
    try:
        stat = path.stat()
    except OSError as exc:
        raise GardenAssetError(
            f"Ambulance collision proxy is missing: {path}"
        ) from exc
    return _read_stretcher_collision_glb_cached(
        str(path.resolve()),
        int(stat.st_size),
        int(stat.st_mtime_ns),
    )


def _stretcher_detail_collision_geometry(
    repo_root: Path,
    manifest: dict[str, Any],
    calibration: dict[str, Any],
) -> tuple[Path, tuple[dict[str, Any], ...], dict[str, Any]]:
    component_specs, contact = _stretcher_detail_collision_spec(calibration)
    try:
        proxy_path = _resolve_repo_path(repo_root, manifest["collision_proxy"])
    except KeyError as exc:
        raise GardenAssetError(
            "Ambulance manifest does not declare collision_proxy."
        ) from exc
    components = _read_stretcher_collision_glb(proxy_path)
    actual_signature = [
        (component["name"], len(component["vertices"]), len(component["faces"]))
        for component in components
    ]
    expected_signature = [
        (spec["name"], spec["vertex_count"], spec["face_count"])
        for spec in component_specs
    ]
    if actual_signature != expected_signature:
        raise GardenAssetError(
            "Ambulance collision proxy components do not match calibration: "
            f"{actual_signature!r} != {expected_signature!r}."
        )
    total_vertices = sum(signature[1] for signature in actual_signature)
    total_faces = sum(signature[2] for signature in actual_signature)
    if (
        total_vertices != int(manifest.get("collision_proxy_vertex_count", -1))
        or total_faces != int(manifest.get("collision_proxy_face_count", -1))
    ):
        raise GardenAssetError(
            "Ambulance collision proxy totals do not match its manifest."
        )
    manifest_source_sha256 = str(
        manifest.get("collision_proxy_source_sha256", "")
    ).strip().lower()
    calibration_source_sha256 = str(
        calibration["stretcher_detail_collision_proxy"]["source_mesh_sha256"]
    ).strip().lower()
    embedded_source_sha256 = str(components[0]["source_sha256"])
    if not (
        manifest_source_sha256
        == calibration_source_sha256
        == embedded_source_sha256
    ):
        raise GardenAssetError(
            "Ambulance collision proxy source checksum differs between its "
            "manifest and calibration."
        )
    return proxy_path, components, contact


def _sample_collision_heightfield(
    offsets: np.ndarray,
    *,
    extent_u: float,
    extent_v: float,
    local_u: float,
    local_v: float,
) -> float:
    """Sample the same checkerboard-triangulated field used by Warp."""

    offsets = np.asarray(offsets, dtype=np.float32)
    cells_v = int(offsets.shape[0] - 1)
    cells_u = int(offsets.shape[1] - 1)
    normalized_u = np.clip(
        (float(local_u) + float(extent_u)) / (2.0 * float(extent_u)),
        0.0,
        1.0,
    )
    normalized_v = np.clip(
        (float(local_v) + float(extent_v)) / (2.0 * float(extent_v)),
        0.0,
        1.0,
    )
    scaled_u = normalized_u * cells_u
    scaled_v = normalized_v * cells_v
    cell_u = min(int(scaled_u), cells_u - 1)
    cell_v = min(int(scaled_v), cells_v - 1)
    fraction_u = scaled_u - cell_u
    fraction_v = scaled_v - cell_v
    height_00 = float(offsets[cell_v, cell_u])
    height_10 = float(offsets[cell_v, cell_u + 1])
    height_01 = float(offsets[cell_v + 1, cell_u])
    height_11 = float(offsets[cell_v + 1, cell_u + 1])
    if (cell_u + cell_v) % 2 == 0:
        if fraction_v <= fraction_u:
            return (
                height_00
                + fraction_u * (height_10 - height_00)
                + fraction_v * (height_11 - height_10)
            )
        return (
            height_00
            + fraction_u * (height_11 - height_01)
            + fraction_v * (height_01 - height_00)
        )
    if fraction_u + fraction_v <= 1.0:
        return (
            height_00
            + fraction_u * (height_10 - height_00)
            + fraction_v * (height_01 - height_00)
        )
    return (
        height_11
        + (1.0 - fraction_u) * (height_01 - height_11)
        + (1.0 - fraction_v) * (height_10 - height_11)
    )


def _sample_projected_mesh_upper_surface(
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    local_u: float,
    local_v: float,
) -> tuple[float, float, bool]:
    """Sample the uppermost triangle in a mesh's local u/v projection.

    The mattress proxy is an open captured shell rather than a heightfield.
    A vertical line can therefore intersect more than one fold near its sides;
    choosing the largest local w value selects the surface an object
    approaching from above meets first. Outside points use their closest
    projected triangle edge for height and footprint-overrun diagnostics.
    """

    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) < 3:
        raise ValueError("Projected surface vertices must have shape (V, 3).")
    if (
        faces.ndim != 2
        or faces.shape[1] != 3
        or len(faces) < 1
        or int(faces.min()) < 0
        or int(faces.max()) >= len(vertices)
    ):
        raise ValueError("Projected surface faces must have valid shape (F, 3).")

    triangles = vertices[faces]
    query = np.array([float(local_u), float(local_v)], dtype=np.float64)
    origin = triangles[:, 0, :2]
    edge_u = triangles[:, 1, :2] - origin
    edge_v = triangles[:, 2, :2] - origin
    query_delta = query[None, :] - origin
    determinant = (
        edge_u[:, 0] * edge_v[:, 1]
        - edge_v[:, 0] * edge_u[:, 1]
    )
    projected_non_degenerate = np.abs(determinant) > 1.0e-12
    barycentric_u = np.zeros(len(triangles), dtype=np.float64)
    barycentric_v = np.zeros(len(triangles), dtype=np.float64)
    barycentric_u[projected_non_degenerate] = (
        query_delta[projected_non_degenerate, 0]
        * edge_v[projected_non_degenerate, 1]
        - edge_v[projected_non_degenerate, 0]
        * query_delta[projected_non_degenerate, 1]
    ) / determinant[projected_non_degenerate]
    barycentric_v[projected_non_degenerate] = (
        edge_u[projected_non_degenerate, 0]
        * query_delta[projected_non_degenerate, 1]
        - query_delta[projected_non_degenerate, 0]
        * edge_u[projected_non_degenerate, 1]
    ) / determinant[projected_non_degenerate]
    tolerance = 1.0e-7
    containing = (
        projected_non_degenerate
        & (barycentric_u >= -tolerance)
        & (barycentric_v >= -tolerance)
        & (barycentric_u + barycentric_v <= 1.0 + tolerance)
    )
    interpolated_w = (
        triangles[:, 0, 2]
        + barycentric_u * (triangles[:, 1, 2] - triangles[:, 0, 2])
        + barycentric_v * (triangles[:, 2, 2] - triangles[:, 0, 2])
    )
    if np.any(containing):
        return float(np.max(interpolated_w[containing])), 0.0, True

    closest_distance_squared = float("inf")
    closest_height = float("-inf")
    for first_index, second_index in ((0, 1), (1, 2), (2, 0)):
        first = triangles[:, first_index, :2]
        segment = triangles[:, second_index, :2] - first
        segment_length_squared = np.sum(segment * segment, axis=1)
        fraction = np.zeros(len(triangles), dtype=np.float64)
        valid_segment = segment_length_squared > 1.0e-16
        fraction[valid_segment] = np.clip(
            np.sum(
                (query[None, :] - first[valid_segment])
                * segment[valid_segment],
                axis=1,
            )
            / segment_length_squared[valid_segment],
            0.0,
            1.0,
        )
        closest = first + fraction[:, None] * segment
        distance_squared = np.sum((closest - query[None, :]) ** 2, axis=1)
        edge_index = int(np.argmin(distance_squared))
        edge_distance_squared = float(distance_squared[edge_index])
        edge_height = float(
            triangles[edge_index, first_index, 2]
            + fraction[edge_index]
            * (
                triangles[edge_index, second_index, 2]
                - triangles[edge_index, first_index, 2]
            )
        )
        if (
            edge_distance_squared < closest_distance_squared - 1.0e-15
            or (
                abs(edge_distance_squared - closest_distance_squared) <= 1.0e-15
                and edge_height > closest_height
            )
        ):
            closest_distance_squared = edge_distance_squared
            closest_height = edge_height
    if not np.isfinite(closest_distance_squared) or not np.isfinite(closest_height):
        raise ValueError("Projected surface has no finite triangle edges.")
    return closest_height, math.sqrt(max(closest_distance_squared, 0.0)), False


def ambulance_mattress_alignment_metrics(
    layout: SimpleLabLayout,
    support_center_world: np.ndarray,
    *,
    table_surface_center_world: np.ndarray | None = None,
) -> dict[str, float | bool | str]:
    """Measure a settled support point against the captured mattress shell."""

    if (
        str(getattr(layout, "scene_name", "")).strip().lower()
        != AMBULANCE_SCENE_NAME
    ):
        raise ValueError("Ambulance mattress metrics require an ambulance layout.")

    support_center = np.asarray(support_center_world, dtype=np.float32).reshape(3)
    del table_surface_center_world
    axis_u = _normalize(
        np.asarray(layout.ambulance_mattress_axis_u_world, dtype=np.float32),
        np.array([1.0, 0.0, 0.0], dtype=np.float32),
    )
    axis_v = _normalize(
        np.asarray(layout.ambulance_mattress_axis_v_world, dtype=np.float32),
        np.array([0.0, 1.0, 0.0], dtype=np.float32),
    )
    normal = _normalize(
        np.asarray(layout.ambulance_mattress_normal_world, dtype=np.float32),
        np.array([0.0, 0.0, -1.0], dtype=np.float32),
    )
    collision_frame_center = np.asarray(
        layout.ambulance_mattress_collision_frame_center_world,
        dtype=np.float32,
    ).reshape(3)
    delta = support_center - collision_frame_center
    offset_u = float(np.dot(delta, axis_u))
    offset_v = float(np.dot(delta, axis_v))
    local_vertices = np.asarray(
        layout.ambulance_mattress_collision_mesh_local_vertices,
        dtype=np.float32,
    )
    local_faces = np.asarray(
        layout.ambulance_mattress_collision_mesh_faces,
        dtype=np.int32,
    )
    surface_height, edge_overrun, inside_surface = (
        _sample_projected_mesh_upper_surface(
            local_vertices,
            local_faces,
            local_u=offset_u,
            local_v=offset_v,
        )
    )
    local_bounds_min = local_vertices.min(axis=0)
    local_bounds_max = local_vertices.max(axis=0)
    footprint_center_u = 0.5 * float(local_bounds_min[0] + local_bounds_max[0])
    footprint_center_v = 0.5 * float(local_bounds_min[1] + local_bounds_max[1])
    extent_u = 0.5 * float(local_bounds_max[0] - local_bounds_min[0])
    extent_v = 0.5 * float(local_bounds_max[1] - local_bounds_min[1])
    bounds_value = max(
        abs(offset_u - footprint_center_u) / max(extent_u, 1.0e-8),
        abs(offset_v - footprint_center_v) / max(extent_v, 1.0e-8),
    )
    plane_error = abs(float(np.dot(delta, normal)) - surface_height)
    footprint_value = (
        min(bounds_value, 1.0)
        if inside_surface
        else max(
            1.0 + edge_overrun / max(min(extent_u, extent_v), 1.0e-8),
            bounds_value,
        )
    )
    return {
        "offset_u_m": offset_u,
        "offset_v_m": offset_v,
        "plane_error_m": plane_error,
        "surface_height_offset_m": surface_height,
        "extent_u_m": extent_u,
        "extent_v_m": extent_v,
        "footprint_kind": "source_mesh_projection",
        "edge_round_radius_m": 0.0,
        "edge_round_drop_m": 0.0,
        "footprint_value": float(footprint_value),
        "edge_overrun_m": float(edge_overrun),
        "inside_surface": bool(inside_surface),
    }


def validate_ambulance_scene(
    repo_root: str | Path,
    *,
    verify_payload_hash: bool = True,
) -> tuple[Path, Path, Path, Path]:
    repo_root = Path(repo_root).resolve()
    manifest_path, manifest, calibration_path, calibration = (
        load_ambulance_manifest(repo_root)
    )
    sog_path = _resolve_repo_path(repo_root, manifest["sog"])
    metadata = read_sog_metadata(sog_path)
    expected_count = int(manifest.get("gaussian_count", 0))
    if int(metadata.get("count", 0)) != expected_count:
        raise GardenAssetError(
            "Ambulance SOG Gaussian count does not match its manifest: "
            f"{metadata.get('count')} != {expected_count}."
        )
    if int(metadata.get("version", -1)) != int(manifest.get("sog_version", -1)):
        raise GardenAssetError("Ambulance SOG version does not match its manifest.")
    sh_record = metadata.get("shN")
    sh_degree = int(sh_record.get("bands", 0)) if isinstance(sh_record, dict) else 0
    if sh_degree != int(manifest.get("sh_degree", -1)) or sh_degree != 3:
        raise GardenAssetError(
            "Ambulance SOG must provide degree-3 spherical harmonics."
        )
    if verify_payload_hash:
        expected_hash = str(manifest.get("sog_sha256", "")).strip().lower()
        actual_hash = sha256_file(sog_path)
        if not expected_hash or actual_hash != expected_hash:
            raise GardenAssetError(
                f"Ambulance SOG checksum mismatch at {sog_path}: {actual_hash}; "
                f"expected {expected_hash}."
            )

    rotation = np.asarray(
        calibration.get("source_to_canonical_rotation"),
        dtype=np.float32,
    )
    if (
        rotation.shape != (3, 3)
        or not np.all(np.isfinite(rotation))
        or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1.0e-5)
        or float(np.linalg.det(rotation)) < 0.999
    ):
        raise GardenAssetError(
            "Ambulance source_to_canonical_rotation must be a proper rotation."
        )
    scale = float(calibration.get("meters_per_source_unit", 0.0))
    if not np.isfinite(scale) or scale <= 0.0:
        raise GardenAssetError(
            "Ambulance meters_per_source_unit must be finite and positive."
        )
    interaction_center = np.asarray(
        calibration.get("source_interaction_surface_center"),
        dtype=np.float32,
    )
    seated_view = calibration.get("seated_view")
    if interaction_center.shape != (3,) or not np.all(np.isfinite(interaction_center)):
        raise GardenAssetError(
            "Ambulance source_interaction_surface_center must contain three "
            "finite values."
        )
    if not isinstance(seated_view, dict):
        raise GardenAssetError("Ambulance calibration has no seated_view record.")
    source_head = np.asarray(
        seated_view.get("source_head_position"),
        dtype=np.float32,
    )
    source_forward = np.asarray(
        seated_view.get("source_forward_direction"),
        dtype=np.float32,
    )
    source_gaze_target = np.asarray(
        seated_view.get("source_gaze_target"),
        dtype=np.float32,
    )
    startup_gaze_pitch_down_degrees = float(
        seated_view.get("startup_gaze_pitch_down_degrees", float("nan"))
    )
    if source_head.shape != (3,) or not np.all(np.isfinite(source_head)):
        raise GardenAssetError(
            "Ambulance seated_view source_head_position must contain three "
            "finite values."
        )
    if source_forward.shape != (3,) or not np.all(np.isfinite(source_forward)):
        raise GardenAssetError(
            "Ambulance seated_view source_forward_direction must contain three "
            "finite values."
        )
    if (
        source_gaze_target.shape != (3,)
        or not np.all(np.isfinite(source_gaze_target))
    ):
        raise GardenAssetError(
            "Ambulance seated_view source_gaze_target must contain three "
            "finite values."
        )
    if (
        not np.isfinite(startup_gaze_pitch_down_degrees)
        or startup_gaze_pitch_down_degrees < 0.0
        or startup_gaze_pitch_down_degrees > 60.0
    ):
        raise GardenAssetError(
            "Ambulance startup gaze pitch must be between 0 and 60 degrees down."
        )
    source_forward_norm = float(np.linalg.norm(source_forward))
    if source_forward_norm <= 1.0e-6:
        raise GardenAssetError(
            "Ambulance seated_view source_forward_direction must be nonzero."
        )
    source_forward = source_forward / source_forward_norm
    canonical_forward = rotation @ source_forward
    if not np.allclose(
        canonical_forward,
        np.array([0.0, 1.0, 0.0], dtype=np.float32),
        atol=2.0e-4,
    ):
        raise GardenAssetError(
            "Ambulance seated_view must face canonical forward toward the stretcher."
        )
    declared_height = float(
        seated_view.get("headset_height_above_floor_m", 0.0)
    )
    diagnostics = calibration.get("fit_diagnostics")
    floor_up = np.asarray(
        diagnostics.get("source_floor_up_normal")
        if isinstance(diagnostics, dict)
        else None,
        dtype=np.float32,
    )
    floor_offset = float(
        diagnostics.get("source_floor_plane_offset", float("nan"))
        if isinstance(diagnostics, dict)
        else float("nan")
    )
    if floor_up.shape != (3,) or not np.all(np.isfinite(floor_up)):
        raise GardenAssetError("Ambulance calibration has no valid fitted floor normal.")
    floor_up_norm = float(np.linalg.norm(floor_up))
    if floor_up_norm <= 1.0e-6 or not np.isfinite(floor_offset):
        raise GardenAssetError("Ambulance calibration has no valid fitted floor plane.")
    floor_up = floor_up / floor_up_norm
    source_gaze_direction = source_gaze_target - source_head
    source_gaze_direction -= (
        float(np.dot(source_gaze_direction, floor_up)) * floor_up
    )
    source_gaze_direction_norm = float(np.linalg.norm(source_gaze_direction))
    if source_gaze_direction_norm <= 1.0e-6:
        raise GardenAssetError(
            "Ambulance seated headset gaze target has no horizontal separation."
        )
    source_gaze_direction /= source_gaze_direction_norm
    if float(np.dot(source_forward, source_gaze_direction)) < 0.995:
        raise GardenAssetError(
            "Ambulance seated headset must face the authored stretcher gaze target."
        )
    measured_height = float(
        (np.dot(floor_up, source_head) + floor_offset) * scale
    )
    if (
        not np.isfinite(declared_height)
        or declared_height <= 0.0
        or abs(measured_height - declared_height) > 0.02
    ):
        raise GardenAssetError(
            "Ambulance seated headset height does not match the fitted floor: "
            f"measured={measured_height:.4f}m "
            f"declared={declared_height:.4f}m."
        )
    interaction_surface = calibration.get("interaction_surface")
    if (
        not isinstance(interaction_surface, dict)
        or interaction_surface.get("support_name") != "stretcher_mattress"
    ):
        raise GardenAssetError(
            "Ambulance interaction surface must identify the stretcher mattress."
        )
    (
        canonical_surface_center,
        canonical_surface_normal,
        canonical_surface_axis_u,
        canonical_surface_axis_v,
    ) = _interaction_surface_frame_canonical(calibration)
    (
        collision_cells_u,
        collision_cells_v,
        collision_offsets,
        _,
        _,
    ) = _interaction_surface_collision_heightfield(calibration)
    proxy_path, _, _ = _stretcher_detail_collision_geometry(
        repo_root,
        manifest,
        calibration,
    )
    if verify_payload_hash:
        expected_proxy_hash = str(
            manifest.get("collision_proxy_sha256", "")
        ).strip().lower()
        actual_proxy_hash = sha256_file(proxy_path)
        if not expected_proxy_hash or actual_proxy_hash != expected_proxy_hash:
            raise GardenAssetError(
                "Ambulance collision proxy checksum mismatch at "
                f"{proxy_path}: {actual_proxy_hash}; expected "
                f"{expected_proxy_hash}."
            )
    collision_center_raise = float(
        collision_offsets[collision_cells_v // 2, collision_cells_u // 2]
    )
    declared_collision_center_raise = float(
        diagnostics.get(
            "collision_crown_center_above_fitted_plane_m",
            float("nan"),
        )
    )
    if (
        not np.isfinite(declared_collision_center_raise)
        or abs(collision_center_raise - declared_collision_center_raise) > 1.0e-6
    ):
        raise GardenAssetError(
            "Ambulance collision crown center does not match fit diagnostics."
        )
    if (
        float(np.dot(canonical_surface_normal, np.array([0.0, 0.0, -1.0])))
        < 0.98
        or abs(float(np.dot(canonical_surface_normal, canonical_surface_axis_u)))
        > 1.0e-5
        or abs(float(np.dot(canonical_surface_normal, canonical_surface_axis_v)))
        > 1.0e-5
        or abs(float(np.dot(canonical_surface_axis_u, canonical_surface_axis_v)))
        > 1.0e-5
    ):
        raise GardenAssetError(
            "Ambulance fitted mattress plane must be a near-horizontal "
            "orthogonal surface frame."
        )
    floor_spec = calibration.get("floor")
    declared_surface_height = float(
        floor_spec.get("down_from_interaction_surface_m", 0.0)
        if isinstance(floor_spec, dict)
        else 0.0
    )
    source_surface_center = interaction_center + (
        rotation.T @ canonical_surface_center
    ) / scale
    measured_surface_height = float(
        (np.dot(floor_up, source_surface_center) + floor_offset) * scale
    )
    if (
        not np.isfinite(declared_surface_height)
        or declared_surface_height <= 0.0
        or abs(measured_surface_height - declared_surface_height) > 0.02
    ):
        raise GardenAssetError(
            "Ambulance stretcher height does not match the fitted floor: "
            f"measured={measured_surface_height:.4f}m "
            f"declared={declared_surface_height:.4f}m."
        )
    canonical_head = (
        (source_head - interaction_center) @ rotation.T * scale
    )
    canonical_gaze_target = (
        (source_gaze_target - interaction_center) @ rotation.T * scale
    )
    canonical_gaze_delta = canonical_gaze_target - canonical_head
    if (
        canonical_head[1] >= -0.5
        or canonical_head[2] >= -0.1
        or canonical_gaze_delta[1] <= 0.5
    ):
        raise GardenAssetError(
            "Ambulance seated headset must be over the side bench, behind and "
            "above the stretcher, with its horizontal gaze directed across the "
            "aisle toward the stretcher."
        )

    standing_view = calibration.get("standing_view")
    if not isinstance(standing_view, dict):
        raise GardenAssetError("Ambulance calibration has no standing_view record.")
    standing_head = np.asarray(
        standing_view.get("canonical_head_position_m"),
        dtype=np.float32,
    )
    standing_forward = np.asarray(
        standing_view.get("canonical_forward_direction"),
        dtype=np.float32,
    )
    standing_height = float(
        standing_view.get("headset_height_above_floor_m", float("nan"))
    )
    standing_pitch = float(
        standing_view.get("startup_gaze_pitch_down_degrees", float("nan"))
    )
    standing_clearance = float(
        standing_view.get("mattress_side_clearance_m", float("nan"))
    )
    if (
        standing_head.shape != (3,)
        or not np.all(np.isfinite(standing_head))
        or standing_forward.shape != (3,)
        or not np.all(np.isfinite(standing_forward))
        or float(np.linalg.norm(standing_forward)) <= 1.0e-6
        or not np.isfinite(standing_height)
        or not np.isfinite(standing_pitch)
        or not np.isfinite(standing_clearance)
    ):
        raise GardenAssetError(
            "Ambulance standing_view must contain finite head, forward, height, "
            "pitch, and mattress-clearance values."
        )
    standing_forward /= float(np.linalg.norm(standing_forward))
    floor_canonical_z = (
        float(calibration["interaction_surface"]["canonical_plane_z_down_from_xy"][2])
        + declared_surface_height
    )
    measured_standing_height = floor_canonical_z - float(standing_head[2])
    surface_size = np.asarray(
        calibration["interaction_surface"]["size_m"],
        dtype=np.float32,
    )
    signed_side_offset = float(
        np.dot(standing_head - canonical_surface_center, canonical_surface_axis_v)
    )
    expected_side_offset = -(
        0.5 * float(surface_size[1]) + standing_clearance
    )
    standing_to_mattress = canonical_surface_center - standing_head
    standing_to_mattress[2] = 0.0
    standing_to_mattress /= max(
        float(np.linalg.norm(standing_to_mattress)),
        1.0e-6,
    )
    if (
        standing_view.get("support_name") != "center_aisle"
        or standing_view.get("faces") != "stretcher"
        or abs(measured_standing_height - standing_height) > 0.02
        or not 1.45 <= standing_height <= 1.65
        or not 15.0 <= standing_pitch <= 40.0
        or standing_clearance < 0.10
        or abs(signed_side_offset - expected_side_offset) > 0.02
        or float(np.dot(standing_forward, standing_to_mattress)) < 0.95
    ):
        raise GardenAssetError(
            "Ambulance standing headset must be floor-calibrated in the clear "
            "aisle beside the mattress and face the stretcher."
        )
    return sog_path, manifest_path, calibration_path, proxy_path


def _box_geometry(
    bounds_min: np.ndarray,
    bounds_max: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    mins = np.asarray(bounds_min, dtype=np.float32).reshape(3)
    maxs = np.asarray(bounds_max, dtype=np.float32).reshape(3)
    vertices = np.array(
        [
            [mins[0], mins[1], mins[2]],
            [maxs[0], mins[1], mins[2]],
            [maxs[0], maxs[1], mins[2]],
            [mins[0], maxs[1], mins[2]],
            [mins[0], mins[1], maxs[2]],
            [maxs[0], mins[1], maxs[2]],
            [maxs[0], maxs[1], maxs[2]],
            [mins[0], maxs[1], maxs[2]],
        ],
        dtype=np.float32,
    )
    faces = np.array(
        [
            [0, 2, 1],
            [0, 3, 2],
            [4, 5, 6],
            [4, 6, 7],
            [0, 1, 5],
            [0, 5, 4],
            [1, 2, 6],
            [1, 6, 5],
            [2, 3, 7],
            [2, 7, 6],
            [3, 0, 4],
            [3, 4, 7],
        ],
        dtype=np.int32,
    )
    return vertices, faces


def _oriented_support_box_geometry(
    center: np.ndarray,
    axis_u: np.ndarray,
    axis_v: np.ndarray,
    surface_normal: np.ndarray,
    size: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a support slab whose first four vertices lie on its top plane."""

    center = np.asarray(center, dtype=np.float32).reshape(3)
    axis_u = _normalize(axis_u, np.array([1.0, 0.0, 0.0], dtype=np.float32))
    axis_v = _normalize(axis_v, np.array([0.0, 1.0, 0.0], dtype=np.float32))
    surface_normal = _normalize(
        surface_normal,
        np.array([0.0, 0.0, -1.0], dtype=np.float32),
    )
    size = np.asarray(size, dtype=np.float32).reshape(3)
    half_u = 0.5 * float(size[0])
    half_v = 0.5 * float(size[1])
    thickness = float(size[2])
    top = np.stack(
        [
            center - axis_u * half_u - axis_v * half_v,
            center + axis_u * half_u - axis_v * half_v,
            center + axis_u * half_u + axis_v * half_v,
            center - axis_u * half_u + axis_v * half_v,
        ],
        axis=0,
    ).astype(np.float32)
    vertices = np.concatenate(
        [top, top - surface_normal[None, :] * thickness],
        axis=0,
    ).astype(np.float32)
    faces = np.array(
        [
            [0, 2, 1],
            [0, 3, 2],
            [4, 5, 6],
            [4, 6, 7],
            [0, 1, 5],
            [0, 5, 4],
            [1, 2, 6],
            [1, 6, 5],
            [2, 3, 7],
            [2, 7, 6],
            [3, 0, 4],
            [3, 4, 7],
        ],
        dtype=np.int32,
    )
    return vertices, faces


def _oriented_heightfield_capsule_slab_geometry(
    center: np.ndarray,
    axis_u: np.ndarray,
    axis_v: np.ndarray,
    surface_normal: np.ndarray,
    size: np.ndarray,
    normal_offsets: np.ndarray,
    *,
    edge_round_radius: float,
    radial_ring_count: int = 32,
    angular_segment_count: int = 128,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a crowned capsule top and a deliberately coarse closed underside."""

    center = np.asarray(center, dtype=np.float32).reshape(3)
    axis_u = _normalize(axis_u, np.array([1.0, 0.0, 0.0], dtype=np.float32))
    axis_v = _normalize(axis_v, np.array([0.0, 1.0, 0.0], dtype=np.float32))
    surface_normal = _normalize(
        surface_normal,
        np.array([0.0, 0.0, -1.0], dtype=np.float32),
    )
    size = np.asarray(size, dtype=np.float32).reshape(3)
    offsets = np.asarray(normal_offsets, dtype=np.float32)
    if offsets.ndim != 2 or min(offsets.shape) < 3:
        raise ValueError("Mattress heightfield must contain at least 3x3 vertices.")
    radial_ring_count = int(radial_ring_count)
    angular_segment_count = int(angular_segment_count)
    edge_round_radius = float(edge_round_radius)
    if (
        not np.isfinite(edge_round_radius)
        or edge_round_radius <= 0.0
        or radial_ring_count < 2
        or angular_segment_count < 8
        or angular_segment_count % 4 != 0
    ):
        raise ValueError("Mattress capsule tessellation is invalid.")
    half_u = 0.5 * float(size[0])
    half_v = 0.5 * float(size[1])
    slab_depth = float(size[2])
    if (
        min(half_u, half_v, slab_depth) <= 0.0
        or half_u <= half_v
        or edge_round_radius >= half_v
    ):
        raise ValueError(
            "Mattress collision slab dimensions and edge radius are invalid."
        )
    capsule_spine_half_length = half_u - half_v

    def boundary_coordinates(angle: float) -> tuple[float, float]:
        cosine = math.cos(angle)
        sine = math.sin(angle)
        ray_distance = float("inf")
        if abs(sine) > 1.0e-8:
            side_distance = half_v / abs(sine)
            if (
                abs(side_distance * cosine)
                <= capsule_spine_half_length + 1.0e-8
            ):
                ray_distance = side_distance
        if not np.isfinite(ray_distance):
            discriminant = (
                half_v * half_v
                - capsule_spine_half_length
                * capsule_spine_half_length
                * sine
                * sine
            )
            ray_distance = (
                capsule_spine_half_length * abs(cosine)
                + math.sqrt(max(discriminant, 0.0))
            )
        return ray_distance * cosine, ray_distance * sine

    top_vertices = [
        center
        + surface_normal
        * _sample_collision_heightfield(
            offsets,
            extent_u=half_u,
            extent_v=half_v,
            local_u=0.0,
            local_v=0.0,
        )
    ]
    boundary_uv = []
    for ring_index in range(1, radial_ring_count + 1):
        uniform_ring_fraction = float(ring_index) / float(radial_ring_count)
        # Concentrate rings near the padded roll-off, where its tangent turns
        # rapidly toward the side, without spending runtime collision work on
        # these diagnostic/export triangles.
        ring_fraction = 1.0 - (1.0 - uniform_ring_fraction) ** 1.35
        for segment_index in range(angular_segment_count):
            angle = (
                2.0
                * math.pi
                * float(segment_index)
                / float(angular_segment_count)
            )
            boundary_u, boundary_v = boundary_coordinates(angle)
            local_u = ring_fraction * boundary_u
            local_v = ring_fraction * boundary_v
            height = _sample_collision_heightfield(
                offsets,
                extent_u=half_u,
                extent_v=half_v,
                local_u=local_u,
                local_v=local_v,
            )
            closest_spine_u = float(
                np.clip(
                    local_u,
                    -capsule_spine_half_length,
                    capsule_spine_half_length,
                )
            )
            footprint_signed_distance = (
                math.hypot(local_u - closest_spine_u, local_v) - half_v
            )
            if footprint_signed_distance > -edge_round_radius:
                corner_coordinate = min(
                    max(
                        footprint_signed_distance + edge_round_radius,
                        0.0,
                    ),
                    edge_round_radius,
                )
                height -= edge_round_radius - math.sqrt(
                    max(
                        edge_round_radius * edge_round_radius
                        - corner_coordinate * corner_coordinate,
                        0.0,
                    )
                )
            top_vertices.append(
                center
                + axis_u * local_u
                + axis_v * local_v
                + surface_normal * height
            )
            if ring_index == radial_ring_count:
                boundary_uv.append((local_u, local_v))

    faces: list[list[int]] = []
    first_ring_start = 1
    for segment_index in range(angular_segment_count):
        next_segment = (segment_index + 1) % angular_segment_count
        faces.append(
            [
                0,
                first_ring_start + next_segment,
                first_ring_start + segment_index,
            ]
        )
    for ring_index in range(1, radial_ring_count):
        inner_start = 1 + (ring_index - 1) * angular_segment_count
        outer_start = inner_start + angular_segment_count
        for segment_index in range(angular_segment_count):
            next_segment = (segment_index + 1) % angular_segment_count
            inner_current = inner_start + segment_index
            inner_next = inner_start + next_segment
            outer_current = outer_start + segment_index
            outer_next = outer_start + next_segment
            faces.extend(
                (
                    [inner_current, outer_next, outer_current],
                    [inner_current, inner_next, outer_next],
                )
            )

    top_vertex_count = len(top_vertices)
    bottom_center_index = top_vertex_count
    bottom_boundary_start = bottom_center_index + 1
    bottom_center = center - surface_normal * slab_depth
    bottom_vertices = [bottom_center]
    for local_u, local_v in boundary_uv:
        bottom_vertices.append(
            bottom_center + axis_u * local_u + axis_v * local_v
        )
    top_boundary_start = (
        1 + (radial_ring_count - 1) * angular_segment_count
    )
    for segment_index in range(angular_segment_count):
        next_segment = (segment_index + 1) % angular_segment_count
        top_current = top_boundary_start + segment_index
        top_next = top_boundary_start + next_segment
        bottom_current = bottom_boundary_start + segment_index
        bottom_next = bottom_boundary_start + next_segment
        faces.extend(
            (
                [top_current, bottom_next, bottom_current],
                [top_current, top_next, bottom_next],
            )
        )
        faces.append(
            [bottom_center_index, bottom_current, bottom_next]
        )
    vertices = np.asarray(top_vertices + bottom_vertices, dtype=np.float32)
    return vertices, np.asarray(faces, dtype=np.int32)


def _polyline_tube_geometry(
    points: np.ndarray,
    radius: float,
    *,
    radial_segments: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a small closed tube around a measured 3D centerline."""

    points = np.asarray(points, dtype=np.float32)
    radius = float(radius)
    radial_segments = int(radial_segments)
    if (
        points.ndim != 2
        or points.shape[1] != 3
        or points.shape[0] < 2
        or not np.all(np.isfinite(points))
        or not np.isfinite(radius)
        or radius <= 0.0
        or radial_segments < 6
    ):
        raise ValueError("Polyline tube geometry inputs are invalid.")
    segment_directions = np.diff(points, axis=0)
    segment_lengths = np.linalg.norm(segment_directions, axis=1)
    if float(np.min(segment_lengths, initial=float("inf"))) <= 1.0e-6:
        raise ValueError("Polyline tube centerline contains a zero-length segment.")
    segment_directions /= segment_lengths[:, None]

    tangents = np.empty_like(points)
    tangents[0] = segment_directions[0]
    tangents[-1] = segment_directions[-1]
    for point_index in range(1, len(points) - 1):
        tangent = segment_directions[point_index - 1] + segment_directions[point_index]
        tangent_length = float(np.linalg.norm(tangent))
        tangents[point_index] = (
            tangent / tangent_length
            if tangent_length > 1.0e-6
            else segment_directions[point_index]
        )

    reference_candidates = np.eye(3, dtype=np.float32)
    reference = reference_candidates[
        int(np.argmin(np.abs(reference_candidates @ tangents[0])))
    ]
    ring_u = _normalize(
        np.cross(tangents[0], reference),
        np.array([1.0, 0.0, 0.0], dtype=np.float32),
    )
    ring_axes_u = [ring_u]
    ring_axes_v = [
        _normalize(
            np.cross(tangents[0], ring_u),
            np.array([0.0, 1.0, 0.0], dtype=np.float32),
        )
    ]
    for point_index in range(1, len(points)):
        tangent = tangents[point_index]
        transported_u = ring_axes_u[-1] - tangent * float(
            np.dot(ring_axes_u[-1], tangent)
        )
        if float(np.linalg.norm(transported_u)) <= 1.0e-6:
            reference = reference_candidates[
                int(np.argmin(np.abs(reference_candidates @ tangent)))
            ]
            transported_u = np.cross(tangent, reference)
        transported_u = _normalize(
            transported_u,
            np.array([1.0, 0.0, 0.0], dtype=np.float32),
        )
        transported_v = _normalize(
            np.cross(tangent, transported_u),
            np.array([0.0, 1.0, 0.0], dtype=np.float32),
        )
        ring_axes_u.append(transported_u)
        ring_axes_v.append(transported_v)

    vertices = []
    for point, axis_u, axis_v in zip(points, ring_axes_u, ring_axes_v):
        for segment_index in range(radial_segments):
            angle = 2.0 * math.pi * float(segment_index) / float(radial_segments)
            vertices.append(
                point
                + radius
                * (math.cos(angle) * axis_u + math.sin(angle) * axis_v)
            )

    faces: list[list[int]] = []
    for point_index in range(len(points) - 1):
        first_ring = point_index * radial_segments
        second_ring = first_ring + radial_segments
        for segment_index in range(radial_segments):
            next_segment = (segment_index + 1) % radial_segments
            first_current = first_ring + segment_index
            first_next = first_ring + next_segment
            second_current = second_ring + segment_index
            second_next = second_ring + next_segment
            faces.extend(
                (
                    [first_current, first_next, second_next],
                    [first_current, second_next, second_current],
                )
            )

    start_center_index = len(vertices)
    vertices.append(points[0])
    end_center_index = len(vertices)
    vertices.append(points[-1])
    end_ring = (len(points) - 1) * radial_segments
    for segment_index in range(radial_segments):
        next_segment = (segment_index + 1) % radial_segments
        faces.append(
            [start_center_index, next_segment, segment_index]
        )
        faces.append(
            [
                end_center_index,
                end_ring + segment_index,
                end_ring + next_segment,
            ]
        )
    return (
        np.ascontiguousarray(vertices, dtype=np.float32),
        np.ascontiguousarray(faces, dtype=np.int32),
    )


def make_ambulance_layout(
    head_position: np.ndarray,
    forward_direction: np.ndarray,
    *,
    repo_root: str | Path,
    scene_up: np.ndarray | None = None,
    start_posture: str = "seated",
) -> SimpleLabLayout:
    repo_root = Path(repo_root).resolve()
    validate_ambulance_scene(repo_root, verify_payload_hash=False)
    _, manifest, _, calibration = load_ambulance_manifest(repo_root)
    head_position = np.asarray(head_position, dtype=np.float32).reshape(3)
    start_posture = normalize_immersive_start_posture(start_posture)
    scene_up = _normalize(
        np.array([0.0, 0.0, -1.0], dtype=np.float32)
        if scene_up is None
        else scene_up,
        np.array([0.0, 0.0, -1.0], dtype=np.float32),
    )
    forward = np.asarray(forward_direction, dtype=np.float32).reshape(3)
    forward = forward - float(np.dot(forward, scene_up)) * scene_up
    forward = _normalize(forward, np.array([0.0, 1.0, 0.0], dtype=np.float32))
    right = _normalize(
        np.cross(scene_up, forward),
        np.array([1.0, 0.0, 0.0], dtype=np.float32),
    )
    down = -scene_up
    canonical_to_world_rotation = np.stack(
        [right, forward, down],
        axis=1,
    ).astype(np.float32)
    if float(np.linalg.det(canonical_to_world_rotation)) < 0.999:
        raise GardenAssetError("Ambulance layout basis is not right-handed.")

    source_to_canonical_rotation = np.asarray(
        calibration["source_to_canonical_rotation"],
        dtype=np.float32,
    ).reshape(3, 3)
    source_interaction_center = np.asarray(
        calibration["source_interaction_surface_center"],
        dtype=np.float32,
    ).reshape(3)
    source_seated_head = np.asarray(
        calibration["seated_view"]["source_head_position"],
        dtype=np.float32,
    ).reshape(3)
    source_gaze_target = np.asarray(
        calibration["seated_view"]["source_gaze_target"],
        dtype=np.float32,
    ).reshape(3)
    canonical_seated_head = (
        (source_seated_head - source_interaction_center)
        @ source_to_canonical_rotation.T
        * float(calibration["meters_per_source_unit"])
    ).astype(np.float32)
    canonical_gaze_target = (
        (source_gaze_target - source_interaction_center)
        @ source_to_canonical_rotation.T
        * float(calibration["meters_per_source_unit"])
    ).astype(np.float32)
    standing_view = calibration.get("standing_view")
    if not isinstance(standing_view, dict):
        raise GardenAssetError("Ambulance calibration has no standing_view record.")
    canonical_standing_head = np.asarray(
        standing_view.get("canonical_head_position_m"),
        dtype=np.float32,
    ).reshape(3)
    canonical_start_head = (
        canonical_standing_head
        if start_posture == "standing"
        else canonical_seated_head
    )
    # Keep the decoded SOG anchored by its legacy canonical reference so that
    # changing the fitted collision surface cannot move the ambulance around
    # the seated viewer.
    scene_origin_world = (
        head_position - canonical_start_head @ canonical_to_world_rotation.T
    ).astype(np.float32)
    (
        canonical_surface_center,
        canonical_surface_normal,
        canonical_surface_axis_u,
        canonical_surface_axis_v,
    ) = _interaction_surface_frame_canonical(calibration)
    fitted_surface_center = (
        canonical_surface_center @ canonical_to_world_rotation.T
        + scene_origin_world
    ).astype(np.float32)
    table_normal = _normalize(
        canonical_surface_normal @ canonical_to_world_rotation.T,
        scene_up,
    )
    table_axis_u = _normalize(
        canonical_surface_axis_u @ canonical_to_world_rotation.T,
        right,
    )
    table_axis_v = _normalize(
        canonical_surface_axis_v @ canonical_to_world_rotation.T,
        forward,
    )
    table_size = np.asarray(
        calibration["interaction_surface"]["size_m"],
        dtype=np.float32,
    ).reshape(3)
    floor_spec = calibration["floor"]
    floor_size = np.asarray(floor_spec["size_m"], dtype=np.float32).reshape(3)
    floor_distance = float(floor_spec["down_from_interaction_surface_m"])

    # The source proxy's local frame was authored around the former analytic
    # crown center. Keep that frame solely as the transform origin; the actual
    # spawn/contact height now comes from the captured mattress triangles.
    (
        mattress_cells_u,
        mattress_cells_v,
        mattress_fitted_plane_offsets,
        _,
        _,
    ) = _interaction_surface_collision_heightfield(calibration)
    mattress_center_raise = float(
        mattress_fitted_plane_offsets[
            mattress_cells_v // 2,
            mattress_cells_u // 2,
        ]
    )
    mattress_collision_frame_center = (
        fitted_surface_center + table_normal * mattress_center_raise
    ).astype(np.float32)
    canonical_collision_frame_center = (
        canonical_surface_center
        + canonical_surface_normal * mattress_center_raise
    ).astype(np.float32)
    (
        detail_proxy_path,
        detail_components,
        detail_contact,
    ) = _stretcher_detail_collision_geometry(
        repo_root,
        manifest,
        calibration,
    )
    mattress_components = [
        component
        for component in detail_components
        if component["name"] == "mattress_surface_full_resolution"
    ]
    if len(mattress_components) != 1:
        raise GardenAssetError(
            "Ambulance collision proxy must contain exactly one full-resolution "
            "mattress surface."
        )
    mattress_local_vertices = np.ascontiguousarray(
        mattress_components[0]["vertices"],
        dtype=np.float32,
    )
    mattress_faces = np.ascontiguousarray(
        mattress_components[0]["faces"],
        dtype=np.int32,
    )
    (
        mattress_surface_center_local_height,
        mattress_surface_center_edge_distance,
        mattress_surface_center_inside,
    ) = _sample_projected_mesh_upper_surface(
        mattress_local_vertices,
        mattress_faces,
        local_u=0.0,
        local_v=0.0,
    )
    if (
        not mattress_surface_center_inside
        or mattress_surface_center_edge_distance > 1.0e-6
    ):
        raise GardenAssetError(
            "Full-resolution ambulance mattress does not cover its spawn center."
        )
    captured_surface_center = (
        mattress_collision_frame_center
        + table_normal * mattress_surface_center_local_height
    ).astype(np.float32)
    canonical_captured_surface_center = (
        canonical_collision_frame_center
        + canonical_surface_normal * mattress_surface_center_local_height
    ).astype(np.float32)
    # Preserve the already-calibrated object spawn and headset gaze reference.
    # The object was originally authored around this collision-frame center;
    # changing the startup anchor to the globally highest captured wrinkle
    # visibly lifts it and makes the otherwise-correct seated view look wrong.
    # Contact recovery and the hidden startup settle place individual nodes on
    # the irregular source-mesh surface without moving the authored anchor.
    table_center = mattress_collision_frame_center.copy()
    canonical_collision_center = canonical_collision_frame_center.copy()
    canonical_gaze_target = canonical_collision_center.copy()
    mattress_spawn_local_height = 0.0
    mattress_spawn_center = table_center.copy()

    detail_geometry_parts = []
    detail_vertices_parts = []
    detail_faces_parts = []
    detail_vertex_offset = 0
    mattress_vertices = None
    for detail_component in detail_components:
        local_vertices = np.asarray(
            detail_component["vertices"],
            dtype=np.float32,
        )
        detail_vertices_part = (
            mattress_collision_frame_center[None, :]
            + local_vertices[:, 0:1] * table_axis_u[None, :]
            + local_vertices[:, 1:2] * table_axis_v[None, :]
            + local_vertices[:, 2:3] * table_normal[None, :]
        ).astype(np.float32)
        detail_faces_part = np.ascontiguousarray(
            detail_component["faces"],
            dtype=np.int32,
        )
        detail_bounds_min = detail_vertices_part.min(axis=0).astype(np.float32)
        detail_bounds_max = detail_vertices_part.max(axis=0).astype(np.float32)
        is_mattress_surface = (
            detail_component["name"] == "mattress_surface_full_resolution"
        )
        detail_kind = (
            "source_mesh_full_resolution_surface"
            if is_mattress_surface
            else "source_mesh_decimated_surface"
        )
        detail_geometry_parts.append(
            (
                detail_component["name"],
                detail_kind,
                detail_vertices_part,
                detail_faces_part,
                detail_bounds_min,
                detail_bounds_max,
            )
        )
        if is_mattress_surface:
            mattress_vertices = detail_vertices_part
        detail_vertices_parts.append(detail_vertices_part)
        detail_faces_parts.append(detail_faces_part + detail_vertex_offset)
        detail_vertex_offset += int(detail_vertices_part.shape[0])
    detail_vertices = np.concatenate(detail_vertices_parts, axis=0).astype(
        np.float32
    )
    detail_faces = np.concatenate(detail_faces_parts, axis=0).astype(np.int32)
    detail_component_bounds = np.asarray(
        [
            [bounds_min, bounds_max]
            for _, _, _, _, bounds_min, bounds_max in detail_geometry_parts
        ],
        dtype=np.float32,
    )
    if mattress_vertices is None:
        raise GardenAssetError(
            "Full-resolution ambulance mattress geometry was not transformed."
        )
    table_bounds_min = mattress_vertices.min(axis=0).astype(np.float32)
    table_bounds_max = mattress_vertices.max(axis=0).astype(np.float32)
    table_bounds = np.stack(
        [table_bounds_min, table_bounds_max],
        axis=0,
    ).astype(np.float32)

    # The floor calibration is anchored at the legacy plane reference, not at
    # the offset mattress-outline center. Moving the capsule to the measured
    # padded top must not move the ambulance floor with it.
    fitted_plane = np.asarray(
        calibration["interaction_surface"]["canonical_plane_z_down_from_xy"],
        dtype=np.float32,
    )
    floor_canonical_center = np.array(
        [0.0, 0.0, float(fitted_plane[2]) + floor_distance],
        dtype=np.float32,
    )
    floor_top_center = (
        floor_canonical_center @ canonical_to_world_rotation.T
        + scene_origin_world
    )
    floor_bounds_min = np.array(
        [
            floor_canonical_center[0] - 0.5 * floor_size[0],
            floor_canonical_center[1] - 0.5 * floor_size[1],
            floor_canonical_center[2],
        ],
        dtype=np.float32,
    )
    floor_bounds_max = np.array(
        [
            floor_canonical_center[0] + 0.5 * floor_size[0],
            floor_canonical_center[1] + 0.5 * floor_size[1],
            floor_canonical_center[2] + floor_size[2],
        ],
        dtype=np.float32,
    )
    floor_vertices_canonical, floor_faces = _box_geometry(
        floor_bounds_min,
        floor_bounds_max,
    )
    floor_vertices = (
        floor_vertices_canonical @ canonical_to_world_rotation.T
        + scene_origin_world[None, :]
    ).astype(np.float32)
    floor_world_bounds_min = floor_vertices.min(axis=0).astype(np.float32)
    floor_world_bounds_max = floor_vertices.max(axis=0).astype(np.float32)

    world_vertices_parts = []
    world_faces_parts = []
    mesh_metadata = []
    vertex_offset = 0
    face_offset = 0
    collision_geometry_parts = [
        (
            name,
            kind,
            kind == "source_mesh_full_resolution_surface",
            vertices,
            faces,
            bounds_min,
            bounds_max,
        )
        for name, kind, vertices, faces, bounds_min, bounds_max in detail_geometry_parts
    ]
    collision_geometry_parts.append(
        (
            "ambulance_floor",
            "box",
            True,
            floor_vertices,
            floor_faces,
            floor_world_bounds_min,
            floor_world_bounds_max,
        )
    )
    for (
        name,
        kind,
        support,
        vertices,
        faces,
        bounds_min,
        bounds_max,
    ) in collision_geometry_parts:
        mesh_metadata.append(
            {
                "name": name,
                "kind": kind,
                "support": support,
                "vertex_start": int(vertex_offset),
                "vertex_count": int(vertices.shape[0]),
                "face_start": int(face_offset),
                "face_count": int(faces.shape[0]),
                "world_bounds_min": bounds_min,
                "world_bounds_max": bounds_max,
            }
        )
        world_vertices_parts.append(vertices)
        world_faces_parts.append(faces + vertex_offset)
        vertex_offset += int(vertices.shape[0])
        face_offset += int(faces.shape[0])

    # Small AABB patches follow samples of the same captured triangles used by
    # physics. They are only for supported/resting-state diagnostics; the mesh
    # remains authoritative for actual contact.
    support_boxes = []
    mattress_local_bounds_min = mattress_local_vertices.min(axis=0)
    mattress_local_bounds_max = mattress_local_vertices.max(axis=0)
    mattress_span_u = float(
        mattress_local_bounds_max[0] - mattress_local_bounds_min[0]
    )
    mattress_span_v = float(
        mattress_local_bounds_max[1] - mattress_local_bounds_min[1]
    )
    support_patch_cells_u = max(8, int(math.ceil(mattress_span_u / 0.12)))
    support_patch_cells_v = max(6, int(math.ceil(mattress_span_v / 0.075)))
    support_patch_size_u = mattress_span_u / float(support_patch_cells_u)
    support_patch_size_v = mattress_span_v / float(support_patch_cells_v)
    mattress_support_patch_count = 0
    for patch_u_index in range(support_patch_cells_u):
        patch_u = (
            float(mattress_local_bounds_min[0])
            + (float(patch_u_index) + 0.5) * support_patch_size_u
        )
        for patch_v_index in range(support_patch_cells_v):
            patch_v = (
                float(mattress_local_bounds_min[1])
                + (float(patch_v_index) + 0.5) * support_patch_size_v
            )
            (
                patch_top_offset,
                _patch_edge_distance,
                patch_inside_surface,
            ) = _sample_projected_mesh_upper_surface(
                mattress_local_vertices,
                mattress_faces,
                local_u=patch_u,
                local_v=patch_v,
            )
            if not patch_inside_surface:
                continue
            patch_thickness = max(
                float(table_size[2]),
                patch_top_offset - float(mattress_local_bounds_min[2]) + 0.01,
            )
            patch_center = (
                mattress_collision_frame_center
                + table_axis_u * patch_u
                + table_axis_v * patch_v
                + table_normal * patch_top_offset
            )
            patch_vertices, _ = _oriented_support_box_geometry(
                patch_center,
                table_axis_u,
                table_axis_v,
                table_normal,
                np.array(
                    [
                        support_patch_size_u,
                        support_patch_size_v,
                        patch_thickness,
                    ],
                    dtype=np.float32,
                ),
            )
            support_boxes.append(
                [
                    patch_vertices.min(axis=0).astype(np.float32),
                    patch_vertices.max(axis=0).astype(np.float32),
                ]
            )
            mattress_support_patch_count += 1
    support_boxes.append([floor_world_bounds_min, floor_world_bounds_max])

    runtime_surfaces = [
        {
            "name": "ambulance_floor",
            "kind": "rectangle",
            "center": floor_top_center.astype(np.float32),
            "normal": scene_up.copy(),
            "axis_u": right.copy(),
            "axis_v": forward.copy(),
            "extent_u": 0.5 * float(floor_size[0]),
            "extent_v": 0.5 * float(floor_size[1]),
        },
    ]
    layout = SimpleLabLayout(
        table_top_center=table_center,
        table_size=table_size,
        floor_z=float(floor_top_center[2]),
        room_half_extent=np.array(
            [0.5 * floor_size[0], 0.5 * floor_size[1]],
            dtype=np.float32,
        ),
        wall_height=1.8,
        scene_up=scene_up,
        room_center_xy=np.asarray(floor_top_center[:2], dtype=np.float32),
        static_collider_boxes=None,
        static_collider_box_metadata=None,
        support_surface_boxes=np.asarray(support_boxes, dtype=np.float32),
        active_table_bounds=table_bounds,
        active_table_surface_center=mattress_spawn_center.copy(),
        smooth_tabletop_bounds=table_bounds.copy(),
        smooth_tabletop_patch_count=mattress_support_patch_count,
        start_posture=start_posture,
        startup_head_height_above_floor_m=float(
            calibration[f"{start_posture}_view"]["headset_height_above_floor_m"]
        ),
    )
    layout.scene_name = AMBULANCE_SCENE_NAME
    layout.scene_forward = forward
    layout.scene_right = right
    layout.canonical_to_world_rotation = canonical_to_world_rotation
    layout.canonical_to_world_translation = scene_origin_world
    layout.ambulance_scene_reference_world = scene_origin_world.copy()
    layout.ambulance_mattress_center_canonical = canonical_collision_center.copy()
    layout.ambulance_mattress_collision_frame_center_canonical = (
        canonical_collision_frame_center.copy()
    )
    layout.ambulance_mattress_collision_frame_center_world = (
        mattress_collision_frame_center.copy()
    )
    layout.ambulance_mattress_captured_surface_center_canonical = (
        canonical_captured_surface_center.copy()
    )
    layout.ambulance_mattress_captured_surface_center_world = (
        captured_surface_center.copy()
    )
    layout.ambulance_mattress_fitted_plane_center_canonical = (
        canonical_surface_center.copy()
    )
    layout.ambulance_mattress_fitted_plane_center_world = (
        fitted_surface_center.copy()
    )
    layout.ambulance_mattress_center_raise_m = mattress_center_raise
    layout.ambulance_mattress_collision_frame_raise_m = mattress_center_raise
    layout.ambulance_mattress_surface_center_local_height_m = (
        mattress_surface_center_local_height
    )
    layout.ambulance_mattress_spawn_local_height_m = (
        mattress_spawn_local_height
    )
    layout.ambulance_mattress_spawn_center_world = mattress_spawn_center.copy()
    layout.ambulance_mattress_normal_world = table_normal.copy()
    layout.ambulance_mattress_axis_u_world = table_axis_u.copy()
    layout.ambulance_mattress_axis_v_world = table_axis_v.copy()
    layout.ambulance_mattress_footprint_kind = "source_mesh_projection"
    layout.ambulance_mattress_edge_round_radius_m = 0.0
    layout.ambulance_seated_head_canonical = canonical_seated_head
    layout.ambulance_seated_head_world = (
        canonical_seated_head @ canonical_to_world_rotation.T
        + scene_origin_world
    ).astype(np.float32)
    layout.ambulance_standing_head_canonical = canonical_standing_head.copy()
    layout.ambulance_standing_head_world = (
        canonical_standing_head @ canonical_to_world_rotation.T
        + scene_origin_world
    ).astype(np.float32)
    layout.ambulance_start_head_canonical = canonical_start_head.copy()
    layout.ambulance_start_head_world = head_position.copy()
    layout.ambulance_start_posture = start_posture
    layout.ambulance_gaze_target_canonical = canonical_gaze_target
    layout.ambulance_gaze_target_world = (
        canonical_gaze_target @ canonical_to_world_rotation.T
        + scene_origin_world
    ).astype(np.float32)
    layout.ambulance_seated_forward_world = forward.copy()
    layout.ambulance_seated_view = dict(calibration["seated_view"])
    layout.ambulance_standing_view = dict(calibration["standing_view"])
    layout.static_collision_mesh_vertices = np.concatenate(
        world_vertices_parts,
        axis=0,
    ).astype(np.float32)
    layout.static_collision_mesh_faces = np.concatenate(
        world_faces_parts,
        axis=0,
    ).astype(np.int32)
    layout.static_collision_mesh_metadata = mesh_metadata
    layout.static_collision_mesh_contact = dict(calibration["contact"])
    layout.ambulance_mattress_collision_mesh_vertices = mattress_vertices.copy()
    layout.ambulance_mattress_collision_mesh_local_vertices = (
        mattress_local_vertices.copy()
    )
    layout.ambulance_mattress_collision_mesh_faces = mattress_faces.copy()
    layout.static_collision_detail_mesh_vertices = detail_vertices.copy()
    layout.static_collision_detail_mesh_faces = detail_faces.copy()
    layout.static_collision_detail_mesh_component_bounds = (
        detail_component_bounds.copy()
    )
    layout.static_collision_detail_mesh_contact = dict(detail_contact)
    layout.static_collision_detail_mesh_two_sided = bool(
        detail_contact["two_sided"]
    )
    layout.static_collision_detail_mesh_source_asset = detail_proxy_path
    layout.static_collision_detail_mesh_metadata = [
        metadata
        for metadata in mesh_metadata
        if metadata["kind"].startswith("source_mesh_")
    ]
    layout.static_collision_surfaces = runtime_surfaces
    layout.static_collision_boxes = np.zeros((0, 2, 3), dtype=np.float32)
    layout.ambulance_manifest = manifest
    return layout


class AmbulanceSceneRenderer(GardenSceneRenderer):
    """Combined Gaussian renderer backed directly by the ambulance SOG."""

    scene_runtime_name = AMBULANCE_SCENE_NAME
    scene_display_name = "Ambulance"

    def __init__(
        self,
        scene_assets_root: str | Path,
        width: int,
        height: int,
        *,
        repo_root: str | Path | None = None,
        eye_resolution: int | None = None,
        **_kwargs,
    ) -> None:
        self.repo_root = (
            Path(repo_root).resolve()
            if repo_root is not None
            else Path(__file__).resolve().parents[1]
        )
        self.scene_assets_root = Path(scene_assets_root).resolve()
        self.width = int(width)
        self.height = int(height)
        self.requested_quality = "full"
        self.quality = "full"
        self.eye_resolution = int(eye_resolution or width)
        self.layout: SimpleLabLayout | None = None
        self._blank_cache = {}
        self._static_gaussians = None
        self._render_storage = None
        self._render_storages = []
        self._combined_gaussians = None
        self._combined_models = []
        self._active_render_slot = 0
        self._chunk_rebuild_stream = None
        self._pending_chunk_rebuild = None
        self._static_count = 0
        self._active_static_count = 0
        self._dynamic_capacity = 0
        self._dynamic_count = 0
        self._dynamic_gaussians = None
        self._chunk_starts = np.zeros((0,), dtype=np.int64)
        self._chunk_counts = np.zeros((0,), dtype=np.int64)
        self._chunk_centers_world = np.zeros((0, 3), dtype=np.float32)
        self._chunk_radii = np.zeros((0,), dtype=np.float32)
        self._active_chunk_ids: tuple[int, ...] = ()
        self._chunk_selection_initialized = False
        self._chunk_rebuild_count = 0
        self._last_chunk_selection_debug: dict[str, Any] = {}
        self._profile_source_frame_seconds: list[float] = []
        self._profile_written = True

        (
            self.manifest_path,
            self.manifest,
            self.calibration_path,
            self.calibration,
        ) = load_ambulance_manifest(self.repo_root)
        self.runtime_ply_path = _resolve_repo_path(
            self.repo_root,
            self.manifest["sog"],
        )
        count = int(self.manifest["gaussian_count"])
        chunk_config = dict(self.manifest.get("spatial_chunks", {}))
        self.runtime_metadata = {
            "source_sha256": str(self.manifest["sog_sha256"]),
            "gaussian_count": count,
            "spatial_chunk_config": {
                "near_plane_m": float(chunk_config.get("near_plane_m", 0.01)),
                "frustum_padding_m": float(
                    chunk_config.get("frustum_padding_m", 0.1)
                ),
                "prefetch_margin_ratio": float(
                    chunk_config.get("prefetch_margin_ratio", 0.2)
                ),
            },
            # SOG is already Morton ordered. A single conservative chunk keeps
            # launch preprocessing at zero and lets gsplat perform its native
            # per-Gaussian frustum/radius culling.
            "spatial_chunks": [
                {
                    "start": 0,
                    "count": count,
                    "sphere_center": [0.0, 0.0, 0.0],
                    "sphere_radius": 100.0,
                }
            ],
        }

    def scene_analysis_cache_debug(self) -> dict[str, Any]:
        return {
            "status": "prepared",
            "reason": "ambulance_sog_v2",
            "schema": 2,
            "input_hash": self.manifest["sog_sha256"],
            "path": str(self.runtime_ply_path),
        }

    def last_balanced_render_debug(self) -> dict[str, Any]:
        return {
            "request_kind": "ambulance_combined_gaussian",
            "effective_backend": "gpu",
            "quality": "full",
        }

    def _load_static_gaussian_model(self, device):
        import torch

        from gaussian_splatting.rotation_utils import quaternion_multiply

        print(
            "[quest_display] decoding ambulance SOG v2: "
            f"{self.runtime_ply_path}",
            flush=True,
        )
        static = load_sog_gaussian_model(
            self.runtime_ply_path,
            device=device,
        )
        source_center = torch.as_tensor(
            self.calibration["source_interaction_surface_center"],
            dtype=static._xyz.dtype,
            device=static._xyz.device,
        )
        rotation_np = np.asarray(
            self.calibration["source_to_canonical_rotation"],
            dtype=np.float32,
        ).reshape(3, 3)
        rotation = torch.as_tensor(
            rotation_np,
            dtype=static._xyz.dtype,
            device=static._xyz.device,
        )
        scale = float(self.calibration["meters_per_source_unit"])
        with torch.no_grad():
            static._xyz = (
                (static._xyz - source_center.unsqueeze(0)) @ rotation.T
            ) * scale
            static._scaling = static._scaling + math.log(scale)
            frame_quaternion = _matrix_to_quaternion_wxyz_torch(
                rotation_np,
                torch_module=torch,
                device=static._rotation.device,
                dtype=static._rotation.dtype,
            )
            static._rotation = quaternion_multiply(
                frame_quaternion.unsqueeze(0),
                static.get_rotation,
            )
            sh_transform = torch.as_tensor(
                sh_rotation_matrix(rotation_np),
                dtype=static._features_dc.dtype,
                device=static._features_dc.device,
            )
            features = torch.cat(
                [static._features_dc, static._features_rest],
                dim=1,
            )
            features = torch.einsum("ij,njc->nic", sh_transform, features)
            static._features_dc = features[:, :1].contiguous()
            static._features_rest = features[:, 1:].contiguous()
        print(
            "[quest_display] ambulance Gaussian scene ready: "
            f"gaussians={int(static._xyz.shape[0]):,} "
            f"meters_per_source_unit={scale:.4f}",
            flush=True,
        )
        gc.collect()
        return static
