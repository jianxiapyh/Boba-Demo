#!/usr/bin/env python3
"""Build the lightweight Ambulance stretcher collision proxy.

The source reconstruction is intentionally treated as build-time input.  The
runtime asset written by this script contains only local-space POSITION data
and triangle indices; colors, normals, UVs, textures, and materials are not
copied from the capture mesh.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np


EXPECTED_SOURCE_SHA256 = (
    "41b6e4537f8ad6b55ca9a71ca5e75e5c58435da66b208e18f75932be47d072f4"
)
GLB_JSON_CHUNK_TYPE = 0x4E4F534A
GLB_BINARY_CHUNK_TYPE = 0x004E4942


@dataclass(frozen=True)
class RegionSpec:
    name: str
    u_bounds: tuple[float, float]
    v_bounds: tuple[float, float]
    w_bounds: tuple[float, float]
    target_face_count: int
    minimum_component_faces: int
    largest_component_only: bool = False
    preserve_full_resolution: bool = False


REGIONS = (
    RegionSpec(
        name="mattress_surface_full_resolution",
        # Preserve the padded shell itself exactly as reconstructed.  The
        # slightly asymmetric longitudinal limits are measured from the
        # capture (the +u end extends farther than the former 1.8 m capsule).
        u_bounds=(-1.02, 1.08),
        v_bounds=(-0.285, 0.285),
        w_bounds=(-0.10, 0.10),
        target_face_count=1,
        minimum_component_faces=1_000,
        largest_component_only=True,
        preserve_full_resolution=True,
    ),
    RegionSpec(
        name="side_hardware_positive_v",
        u_bounds=(-1.05, 1.05),
        # The mattress component owns its captured padded sides.  This region
        # starts just outside it and remains aggressively simplified because
        # only the rail/handle silhouette matters for contact.
        v_bounds=(0.275, 0.42),
        w_bounds=(-0.14, 0.15),
        target_face_count=2500,
        minimum_component_faces=200,
        largest_component_only=True,
    ),
    RegionSpec(
        name="side_hardware_negative_v",
        u_bounds=(-1.05, 1.05),
        v_bounds=(-0.42, -0.275),
        w_bounds=(-0.14, 0.15),
        target_face_count=2500,
        minimum_component_faces=200,
        largest_component_only=True,
    ),
    RegionSpec(
        name="lower_structure",
        u_bounds=(-1.05, 1.05),
        v_bounds=(-0.42, 0.42),
        # Stop above the reconstructed floor plane. Runtime already has an
        # analytic floor, so retaining that sheet would add triangles without
        # improving stretcher contact.
        w_bounds=(-0.56, -0.125),
        target_face_count=6000,
        minimum_component_faces=300,
    ),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _normalize(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=np.float64)
    length = float(np.linalg.norm(value))
    if not np.isfinite(length) or length <= 1.0e-12:
        raise ValueError("Cannot normalize an invalid zero-length vector.")
    return value / length


def _mattress_local_vertices(
    source_vertices: np.ndarray,
    calibration: dict,
) -> np.ndarray:
    source_center = np.asarray(
        calibration["source_interaction_surface_center"], dtype=np.float64
    )
    source_to_canonical = np.asarray(
        calibration["source_to_canonical_rotation"], dtype=np.float64
    )
    scale = float(calibration["meters_per_source_unit"])
    canonical_vertices = (
        (np.asarray(source_vertices, dtype=np.float64) - source_center)
        @ source_to_canonical.T
        * scale
    )

    surface = calibration["interaction_surface"]
    center_xy = np.asarray(surface["canonical_center_xy_m"], dtype=np.float64)
    slope_x, slope_y, intercept = [
        float(value) for value in surface["canonical_plane_z_down_from_xy"]
    ]
    fitted_center = np.array(
        [
            center_xy[0],
            center_xy[1],
            slope_x * center_xy[0] + slope_y * center_xy[1] + intercept,
        ],
        dtype=np.float64,
    )
    axis_u_xy = _normalize(
        np.asarray(surface["canonical_axis_u_xy"], dtype=np.float64)
    )
    normal = _normalize(np.array([slope_x, slope_y, -1.0], dtype=np.float64))
    axis_u = _normalize(
        np.array(
            [
                axis_u_xy[0],
                axis_u_xy[1],
                slope_x * axis_u_xy[0] + slope_y * axis_u_xy[1],
            ],
            dtype=np.float64,
        )
    )
    axis_v = _normalize(np.cross(axis_u, normal))
    profile = surface["collision_heightfield"]["profile"]
    collision_center = fitted_center + normal * float(
        profile["center_above_fitted_plane_m"]
    )
    delta = canonical_vertices - collision_center
    return np.ascontiguousarray(
        np.stack(
            [delta @ axis_u, delta @ axis_v, delta @ normal],
            axis=1,
        ),
        dtype=np.float64,
    )


def _load_source_mesh(path: Path) -> tuple[np.ndarray, np.ndarray]:
    import trimesh

    loaded = trimesh.load(path, force="scene", process=False)
    if isinstance(loaded, trimesh.Trimesh):
        meshes = [loaded]
    elif isinstance(loaded, trimesh.Scene):
        meshes = []
        for node_name in loaded.graph.nodes_geometry:
            transform, geometry_name = loaded.graph[node_name]
            geometry = loaded.geometry[geometry_name].copy()
            geometry.apply_transform(transform)
            meshes.append(geometry)
    else:
        raise TypeError(f"Unsupported source mesh object: {type(loaded)!r}")
    if not meshes:
        raise ValueError("The source GLB does not contain triangle geometry.")
    combined = trimesh.util.concatenate(meshes)
    vertices = np.asarray(combined.vertices, dtype=np.float64)
    faces = np.asarray(combined.faces, dtype=np.int64)
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError("Source mesh vertices do not have shape (V, 3).")
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError("Source mesh faces do not have shape (F, 3).")
    return vertices, faces


def _compact_mesh(
    vertices: np.ndarray,
    faces: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    faces = np.asarray(faces, dtype=np.int64)
    if len(faces) == 0:
        raise ValueError("Cannot compact an empty triangle set.")
    triangle_vertices = np.asarray(vertices, dtype=np.float64)[faces]
    twice_area = np.linalg.norm(
        np.cross(
            triangle_vertices[:, 1] - triangle_vertices[:, 0],
            triangle_vertices[:, 2] - triangle_vertices[:, 0],
        ),
        axis=1,
    )
    faces = faces[twice_area > 1.0e-10]
    if len(faces) == 0:
        raise ValueError("Triangle set contains only degenerate faces.")
    used = np.unique(faces.reshape(-1))
    remap = np.full(len(vertices), -1, dtype=np.int64)
    remap[used] = np.arange(len(used), dtype=np.int64)
    return (
        np.ascontiguousarray(np.asarray(vertices)[used], dtype=np.float64),
        np.ascontiguousarray(remap[faces], dtype=np.int64),
    )


def _connected_face_components(faces: np.ndarray) -> list[np.ndarray]:
    import trimesh

    mesh = trimesh.Trimesh(
        vertices=np.zeros((int(faces.max()) + 1, 3), dtype=np.float64),
        faces=faces,
        process=False,
    )
    components = trimesh.graph.connected_components(
        mesh.face_adjacency,
        nodes=np.arange(len(faces), dtype=np.int64),
        min_len=1,
    )
    return sorted(
        (np.asarray(component, dtype=np.int64) for component in components),
        key=lambda component: (-len(component), int(component.min())),
    )


def _simplify(
    vertices: np.ndarray,
    faces: np.ndarray,
    target_faces: int,
) -> tuple[np.ndarray, np.ndarray]:
    import open3d as o3d

    if len(faces) <= target_faces:
        return _compact_mesh(vertices, faces)
    mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(np.asarray(vertices, dtype=np.float64)),
        o3d.utility.Vector3iVector(np.asarray(faces, dtype=np.int32)),
    )
    simplified = mesh.simplify_quadric_decimation(
        target_number_of_triangles=int(target_faces),
        maximum_error=float("inf"),
        boundary_weight=10.0,
    )
    simplified.remove_degenerate_triangles()
    simplified.remove_duplicated_triangles()
    simplified.remove_duplicated_vertices()
    simplified.remove_unreferenced_vertices()
    return _compact_mesh(
        np.asarray(simplified.vertices, dtype=np.float64),
        np.asarray(simplified.triangles, dtype=np.int64),
    )


def _extract_components(
    local_vertices: np.ndarray,
    source_faces: np.ndarray,
) -> list[dict]:
    face_centers = local_vertices[source_faces].mean(axis=1)
    output: list[dict] = []
    for region in REGIONS:
        mask = (
            (face_centers[:, 0] >= region.u_bounds[0])
            & (face_centers[:, 0] <= region.u_bounds[1])
            & (face_centers[:, 1] >= region.v_bounds[0])
            & (face_centers[:, 1] <= region.v_bounds[1])
            & (face_centers[:, 2] >= region.w_bounds[0])
            & (face_centers[:, 2] <= region.w_bounds[1])
        )
        region_vertices, region_faces = _compact_mesh(
            local_vertices,
            source_faces[mask],
        )
        face_components = _connected_face_components(region_faces)
        if region.largest_component_only:
            face_components = face_components[:1]
        else:
            face_components = [
                component
                for component in face_components
                if len(component) >= region.minimum_component_faces
            ]
        if not face_components:
            raise ValueError(f"Region {region.name!r} has no retained components.")

        retained_face_count = sum(len(component) for component in face_components)
        for component_index, face_indices in enumerate(face_components):
            component_vertices, component_faces = _compact_mesh(
                region_vertices,
                region_faces[face_indices],
            )
            if region.preserve_full_resolution:
                simplified_vertices, simplified_faces = _compact_mesh(
                    component_vertices,
                    component_faces,
                )
            else:
                target_faces = max(
                    96,
                    int(
                        round(
                            region.target_face_count
                            * len(component_faces)
                            / retained_face_count
                        )
                    ),
                )
                target_faces = min(target_faces, len(component_faces))
                simplified_vertices, simplified_faces = _simplify(
                    component_vertices,
                    component_faces,
                    target_faces,
                )
            suffix = (
                ""
                if len(face_components) == 1
                else f"_{component_index:02d}"
            )
            output.append(
                {
                    "name": f"{region.name}{suffix}",
                    "vertices": np.ascontiguousarray(
                        simplified_vertices, dtype=np.float32
                    ),
                    "faces": np.ascontiguousarray(
                        simplified_faces, dtype=np.uint32
                    ),
                    "source_face_count": int(len(component_faces)),
                }
            )
    return output


def _pad_bytes(value: bytes, pad_byte: bytes) -> bytes:
    padding = (-len(value)) % 4
    return value + pad_byte * padding


def _write_minimal_glb(
    path: Path,
    components: list[dict],
    *,
    source_sha256: str,
) -> None:
    binary = bytearray()
    buffer_views = []
    accessors = []
    meshes = []
    nodes = []

    for component in components:
        vertices = np.asarray(component["vertices"], dtype="<f4")
        faces = np.asarray(component["faces"], dtype="<u4")
        while len(binary) % 4:
            binary.append(0)
        position_offset = len(binary)
        binary.extend(vertices.tobytes(order="C"))
        position_view = len(buffer_views)
        buffer_views.append(
            {
                "buffer": 0,
                "byteOffset": position_offset,
                "byteLength": int(vertices.nbytes),
                "target": 34962,
            }
        )
        position_accessor = len(accessors)
        accessors.append(
            {
                "bufferView": position_view,
                "componentType": 5126,
                "count": int(len(vertices)),
                "type": "VEC3",
                "min": vertices.min(axis=0).astype(float).tolist(),
                "max": vertices.max(axis=0).astype(float).tolist(),
            }
        )

        while len(binary) % 4:
            binary.append(0)
        index_offset = len(binary)
        binary.extend(faces.reshape(-1).tobytes(order="C"))
        index_view = len(buffer_views)
        buffer_views.append(
            {
                "buffer": 0,
                "byteOffset": index_offset,
                "byteLength": int(faces.nbytes),
                "target": 34963,
            }
        )
        index_accessor = len(accessors)
        accessors.append(
            {
                "bufferView": index_view,
                "componentType": 5125,
                "count": int(faces.size),
                "type": "SCALAR",
                "min": [int(faces.min())],
                "max": [int(faces.max())],
            }
        )

        mesh_index = len(meshes)
        meshes.append(
            {
                "name": component["name"],
                "primitives": [
                    {
                        "attributes": {"POSITION": position_accessor},
                        "indices": index_accessor,
                        "mode": 4,
                    }
                ],
            }
        )
        nodes.append({"name": component["name"], "mesh": mesh_index})

    document = {
        "asset": {
            "version": "2.0",
            "generator": "Boba-Demo Ambulance collision proxy builder",
            "extras": {
                "source_sha256": source_sha256,
                "coordinate_frame": "mattress_collision_center_local_uvw_m",
                "vertex_attributes": ["POSITION"],
            },
        },
        "scene": 0,
        "scenes": [{"nodes": list(range(len(nodes)))}],
        "nodes": nodes,
        "meshes": meshes,
        "buffers": [{"byteLength": len(binary)}],
        "bufferViews": buffer_views,
        "accessors": accessors,
    }
    json_chunk = _pad_bytes(
        json.dumps(document, separators=(",", ":")).encode("utf-8"),
        b" ",
    )
    binary_chunk = _pad_bytes(bytes(binary), b"\x00")
    total_length = 12 + 8 + len(json_chunk) + 8 + len(binary_chunk)
    payload = bytearray(struct.pack("<4sII", b"glTF", 2, total_length))
    payload.extend(
        struct.pack("<II", len(json_chunk), GLB_JSON_CHUNK_TYPE)
    )
    payload.extend(json_chunk)
    payload.extend(
        struct.pack("<II", len(binary_chunk), GLB_BINARY_CHUNK_TYPE)
    )
    payload.extend(binary_chunk)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument(
        "--calibration",
        type=Path,
        default=Path("assets/scenes/ambulance_insta360/calibration.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "assets/scenes/ambulance_insta360/stretcher_collision_proxy.glb"
        ),
    )
    parser.add_argument(
        "--expected-source-sha256",
        default=EXPECTED_SOURCE_SHA256,
    )
    args = parser.parse_args()

    source = args.source.resolve()
    calibration_path = args.calibration.resolve()
    output = args.output.resolve()
    source_sha256 = _sha256(source)
    if source_sha256 != args.expected_source_sha256.strip().lower():
        raise ValueError(
            "Source mesh SHA-256 mismatch: "
            f"{source_sha256}; expected {args.expected_source_sha256}."
        )
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    source_vertices, source_faces = _load_source_mesh(source)
    local_vertices = _mattress_local_vertices(source_vertices, calibration)
    components = _extract_components(local_vertices, source_faces)
    _write_minimal_glb(
        output,
        components,
        source_sha256=source_sha256,
    )

    report = {
        "output": str(output),
        "output_sha256": _sha256(output),
        "output_bytes": output.stat().st_size,
        "source": str(source),
        "source_sha256": source_sha256,
        "source_vertices": int(len(source_vertices)),
        "source_faces": int(len(source_faces)),
        "proxy_vertices": int(
            sum(len(component["vertices"]) for component in components)
        ),
        "proxy_faces": int(
            sum(len(component["faces"]) for component in components)
        ),
        "components": [
            {
                "name": component["name"],
                "vertices": int(len(component["vertices"])),
                "faces": int(len(component["faces"])),
                "source_faces": int(component["source_face_count"]),
                "bounds_min": np.asarray(component["vertices"])
                .min(axis=0)
                .astype(float)
                .tolist(),
                "bounds_max": np.asarray(component["vertices"])
                .max(axis=0)
                .astype(float)
                .tolist(),
            }
            for component in components
        ],
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
