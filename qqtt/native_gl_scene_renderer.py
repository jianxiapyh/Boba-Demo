from __future__ import annotations

import ctypes
import gzip
import json
import math
import os
import pickle
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from .pyrender_cuda_bridge import (
    _InteropBuffer,
    _get_unpack_function,
    graphics_map_flags,
)


_SCENE_NAME = "ILLIXR_lab"
_DEFAULT_CLEAR_RGBA = (243, 244, 246, 255)
_BACKGROUND_MATERIAL = "Output.001"
_GL_TEXTURE_MAX_ANISOTROPY_EXT = 0x84FE
_GL_MAX_TEXTURE_MAX_ANISOTROPY_EXT = 0x84FF


@dataclass
class _MaterialInfo:
    name: str
    kd: np.ndarray
    ke: np.ndarray
    texture_name: str | None


@dataclass
class _ObjGroupAsset:
    material_name: str
    vertices: np.ndarray
    indices: np.ndarray
    triangle_count: int


@dataclass
class _GlMesh:
    material_name: str
    vao: int
    vbo: int
    ebo: int
    texture_id: int
    index_count: int
    vertex_count: int
    triangle_count: int
    color_factor: np.ndarray


def _resolve_scene_root(scene_assets_root: str | Path) -> Path:
    root = Path(scene_assets_root).resolve()
    if (root / "manifest.json").exists():
        return root
    scene_root = root / _SCENE_NAME
    if not (scene_root / "manifest.json").exists():
        raise FileNotFoundError(
            f"Could not find {_SCENE_NAME}/manifest.json under {root}"
        )
    return scene_root


def _read_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _parse_mtl(path: Path) -> dict[str, _MaterialInfo]:
    materials: dict[str, _MaterialInfo] = {}
    current: _MaterialInfo | None = None
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            key = parts[0]
            if key == "newmtl" and len(parts) >= 2:
                current = _MaterialInfo(
                    name=parts[1],
                    kd=np.ones((3,), dtype=np.float32),
                    ke=np.zeros((3,), dtype=np.float32),
                    texture_name=None,
                )
                materials[current.name] = current
            elif current is not None and key == "Kd" and len(parts) >= 4:
                current.kd = np.asarray(
                    [float(v) for v in parts[1:4]],
                    dtype=np.float32,
                )
            elif current is not None and key == "Ke" and len(parts) >= 4:
                current.ke = np.asarray(
                    [float(v) for v in parts[1:4]],
                    dtype=np.float32,
                )
            elif current is not None and key == "map_Kd" and len(parts) >= 2:
                current.texture_name = parts[-1]
    return materials


def _resolve_obj_index(raw: str, count: int) -> int | None:
    if not raw:
        return None
    value = int(raw)
    if value > 0:
        return value - 1
    return count + value


def _parse_obj_vertex_token(
    token: str,
    vertex_count: int,
    uv_count: int,
    normal_count: int,
) -> tuple[int, int | None, int | None]:
    pieces = token.split("/")
    vertex_index = _resolve_obj_index(pieces[0], vertex_count)
    if vertex_index is None:
        raise ValueError(f"OBJ face token is missing a vertex index: {token!r}")
    uv_index = _resolve_obj_index(pieces[1], uv_count) if len(pieces) >= 2 else None
    normal_index = (
        _resolve_obj_index(pieces[2], normal_count)
        if len(pieces) >= 3
        else None
    )
    return int(vertex_index), uv_index, normal_index


def _load_scene_cache(
    scene_root: Path,
    manifest: dict[str, Any],
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    cache_name = manifest.get("scene_analysis_cache", "scene_analysis_cache_v4.pkl.gz")
    cache_path = scene_root / str(cache_name)
    debug = {
        "status": "miss",
        "reason": "not_attempted",
        "path": str(cache_path),
        "schema": manifest.get("scene_analysis_cache_schema"),
    }
    if not cache_path.exists():
        debug["reason"] = "missing"
        return None, debug
    try:
        with gzip.open(cache_path, "rb") as handle:
            payload = pickle.load(handle)
        if isinstance(payload, dict) and isinstance(payload.get("analysis"), dict):
            payload = payload["analysis"]
        debug["status"] = "hit"
        debug["reason"] = "loaded"
        return payload, debug
    except Exception as exc:
        debug["reason"] = f"{type(exc).__name__}: {exc}"
        return None, debug


def _rotation_x_matrix(degrees: float) -> np.ndarray:
    radians = math.radians(float(degrees))
    c = math.cos(radians)
    s = math.sin(radians)
    return np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, c, -s, 0.0],
            [0.0, s, c, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )


def _transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    ones = np.ones((points.shape[0], 1), dtype=np.float32)
    homogeneous = np.concatenate([points, ones], axis=1)
    transformed = (np.asarray(transform, dtype=np.float32) @ homogeneous.T).T
    return transformed[:, :3].astype(np.float32)


def _normalize_rows(values: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return np.divide(
        values,
        np.maximum(norms, 1.0e-8),
        out=np.zeros_like(values, dtype=np.float32),
    ).astype(np.float32)


def _projection_from_intrinsic(
    intrinsic: np.ndarray,
    width: int,
    height: int,
    znear: float,
    zfar: float,
) -> np.ndarray:
    intrinsic = np.asarray(intrinsic, dtype=np.float32)
    fx = float(intrinsic[0, 0])
    fy = float(intrinsic[1, 1])
    cx = float(intrinsic[0, 2])
    cy = float(intrinsic[1, 2])
    w = float(width)
    h = float(height)
    projection = np.zeros((4, 4), dtype=np.float32)
    projection[0, 0] = 2.0 * fx / w
    projection[1, 1] = 2.0 * fy / h
    projection[0, 2] = 1.0 - (2.0 * cx / w)
    projection[1, 2] = (2.0 * cy / h) - 1.0
    projection[2, 2] = -(zfar + znear) / (zfar - znear)
    projection[2, 3] = -(2.0 * zfar * znear) / (zfar - znear)
    projection[3, 2] = -1.0
    return projection


class NativeGlSceneRenderer:
    """Native OpenGL full-room renderer with CUDA tensor readback."""

    def __init__(
        self,
        scene_assets_root: str | Path,
        width: int,
        height: int,
        *,
        render_background: bool = False,
        znear: float = 0.02,
        zfar: float = 100.0,
        texture_mode: str = "stable_mipmap",
        anisotropy: int = 8,
        msaa_samples: int = 4,
        depth_format: str = "depth32f",
        device: torch.device | str | None = None,
        output_ring_size: int = 4,
    ) -> None:
        self.scene_root = _resolve_scene_root(scene_assets_root)
        self.width = int(width)
        self.height = int(height)
        self.render_background = bool(render_background)
        self.znear = float(znear)
        self.zfar = float(zfar)
        self.texture_mode = str(texture_mode).strip().lower()
        if self.texture_mode not in {"stable", "stable_mipmap", "legacy"}:
            raise ValueError(
                "texture_mode must be one of {'stable', 'stable_mipmap', 'legacy'}"
            )
        self.requested_anisotropy = int(anisotropy)
        if self.requested_anisotropy not in {1, 2, 4, 8, 16}:
            raise ValueError("anisotropy must be one of {1, 2, 4, 8, 16}")
        self.effective_anisotropy = 1.0
        self.anisotropy_reason = "disabled_for_texture_mode"
        self.msaa_samples = int(msaa_samples)
        if self.msaa_samples not in {1, 2, 4}:
            raise ValueError("msaa_samples must be one of {1, 2, 4}")
        self.depth_format = str(depth_format).strip().lower()
        if self.depth_format not in {"depth24", "depth32f"}:
            raise ValueError("depth_format must be one of {'depth24', 'depth32f'}")
        if device is None:
            device = torch.device("cuda", torch.cuda.current_device())
        self.device = torch.device(device)
        self._output_ring_size = max(int(output_ring_size), 2)
        self.manifest = _read_json(self.scene_root / "manifest.json")
        self._cache_payload, self._scene_analysis_cache_debug = _load_scene_cache(
            self.scene_root,
            self.manifest,
        )
        self._background_excluded_faces = self._background_table_face_exclusions()
        self._materials = _parse_mtl(
            self.scene_root / str(self.manifest["scene_material"])
        )
        self._asset_groups = self._parse_obj_groups()

        self._platform = None
        self._gl = None
        self._program = 0
        self._uniform_mvp = -1
        self._uniform_texture = -1
        self._uniform_color_factor = -1
        self._fbo_targets: dict[str, dict[str, int]] = {}
        self._texture_cache: dict[str, int] = {}
        self._white_texture = 0
        self._meshes: list[_GlMesh] = []
        self._world_transform = np.eye(4, dtype=np.float32)
        self._interop_dims: tuple[int, int] | None = None
        self._color_pbo: _InteropBuffer | None = None
        self._depth_pbo: _InteropBuffer | None = None
        self._color_tensors: list[torch.Tensor] = []
        self._depth_tensors: list[torch.Tensor] = []
        self._output_ring_index = 0
        self._last_render_debug: dict[str, Any] | None = None

        self._init_context()
        self._init_gl_state()

    def _background_table_face_exclusions(self) -> dict[str, set[int]]:
        if not self.render_background or self._cache_payload is None:
            return {}
        table_component_ids = {
            int(component_id)
            for component_id in self._cache_payload.get("table_component_ids", [])
        }
        if not table_component_ids:
            return {}
        excluded: set[int] = set()
        for record in self._cache_payload.get("furniture_component_records", []):
            if int(record.get("id", -1)) not in table_component_ids:
                continue
            excluded.update(int(index) for index in record.get("face_indices", []))
        return {_BACKGROUND_MATERIAL: excluded} if excluded else {}

    def _parse_obj_groups(self) -> list[_ObjGroupAsset]:
        obj_path = self.scene_root / str(self.manifest["scene_model"])
        positions: list[tuple[float, float, float]] = []
        uvs: list[tuple[float, float]] = []
        normals: list[tuple[float, float, float]] = []
        builders: dict[str, dict[str, Any]] = {}
        material_face_index: dict[str, int] = {}
        current_material = "default"

        def builder_for(material_name: str) -> dict[str, Any]:
            if material_name not in builders:
                builders[material_name] = {
                    "lookup": {},
                    "vertices": [],
                    "indices": [],
                    "triangle_count": 0,
                }
            return builders[material_name]

        def append_vertex(
            builder: dict[str, Any],
            key: tuple[int, int | None, int | None],
        ) -> int:
            lookup = builder["lookup"]
            cached = lookup.get(key)
            if cached is not None:
                return int(cached)
            vertex_index, uv_index, normal_index = key
            px, py, pz = positions[vertex_index]
            if uv_index is None:
                tu, tv = 0.0, 0.0
            else:
                tu, tv = uvs[uv_index]
            if normal_index is None:
                nx, ny, nz = 0.0, 0.0, 1.0
            else:
                nx, ny, nz = normals[normal_index]
            new_index = len(builder["vertices"])
            builder["vertices"].append((px, py, pz, tu, tv, nx, ny, nz))
            lookup[key] = new_index
            return new_index

        with open(obj_path, "r", encoding="utf-8", errors="replace") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                key = parts[0]
                if key == "v" and len(parts) >= 4:
                    positions.append((float(parts[1]), float(parts[2]), float(parts[3])))
                elif key == "vt" and len(parts) >= 3:
                    uvs.append((float(parts[1]), float(parts[2])))
                elif key == "vn" and len(parts) >= 4:
                    normals.append((float(parts[1]), float(parts[2]), float(parts[3])))
                elif key == "usemtl" and len(parts) >= 2:
                    current_material = parts[1]
                    material_face_index.setdefault(current_material, 0)
                elif key == "f" and len(parts) >= 4:
                    face_index = material_face_index.get(current_material, 0)
                    material_face_index[current_material] = face_index + 1
                    excluded = self._background_excluded_faces.get(current_material)
                    if excluded is not None and face_index in excluded:
                        continue
                    parsed = [
                        _parse_obj_vertex_token(
                            token,
                            len(positions),
                            len(uvs),
                            len(normals),
                        )
                        for token in parts[1:]
                    ]
                    builder = builder_for(current_material)
                    for index in range(1, len(parsed) - 1):
                        for vertex_key in (parsed[0], parsed[index], parsed[index + 1]):
                            builder["indices"].append(
                                append_vertex(builder, vertex_key)
                            )
                        builder["triangle_count"] += 1

        groups: list[_ObjGroupAsset] = []
        for material_name, builder in builders.items():
            if not builder["indices"]:
                continue
            groups.append(
                _ObjGroupAsset(
                    material_name=material_name,
                    vertices=np.asarray(builder["vertices"], dtype=np.float32),
                    indices=np.asarray(builder["indices"], dtype=np.uint32),
                    triangle_count=int(builder["triangle_count"]),
                )
            )
        groups.sort(key=lambda group: group.material_name)
        return groups

    def _init_context(self) -> None:
        platform_name = os.environ.get("PYOPENGL_PLATFORM", "").strip().lower()
        if platform_name in {"", "auto"}:
            platform_name = "pyglet"

        if platform_name == "egl":
            from pyrender.platforms import egl

            device_id = int(os.environ.get("EGL_DEVICE_ID", "0"))
            self._platform = egl.EGLPlatform(
                self.width,
                self.height,
                device=egl.get_device_by_index(device_id),
            )
        elif platform_name == "osmesa":
            from pyrender.platforms import osmesa

            self._platform = osmesa.OSMesaPlatform(self.width, self.height)
        else:
            from pyrender.platforms import pyglet_platform

            self._platform = pyglet_platform.PygletPlatform(self.width, self.height)

        self._platform.init_context()
        self._platform.make_current()

        from OpenGL import GL as gl

        self._gl = gl

    def _init_gl_state(self) -> None:
        gl = self._gl
        assert gl is not None
        version = gl.glGetString(gl.GL_VERSION)
        if version is None:
            raise RuntimeError("OpenGL context was created but GL_VERSION is unavailable.")
        gl.glViewport(0, 0, self.width, self.height)
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glDepthFunc(gl.GL_LEQUAL)
        gl.glDisable(gl.GL_CULL_FACE)
        gl.glDisable(gl.GL_BLEND)
        gl.glPixelStorei(gl.GL_UNPACK_ALIGNMENT, 1)
        gl.glPixelStorei(gl.GL_PACK_ALIGNMENT, 1)
        self._program = self._create_program()
        self._uniform_mvp = gl.glGetUniformLocation(self._program, "u_mvp")
        self._uniform_texture = gl.glGetUniformLocation(self._program, "u_texture")
        self._uniform_color_factor = gl.glGetUniformLocation(
            self._program,
            "u_color_factor",
        )
        self._resolve_texture_anisotropy()
        self._white_texture = self._create_solid_texture((255, 255, 255, 255))
        self._create_fbo_targets()

    def _compile_shader(self, shader_type: int, source: str) -> int:
        gl = self._gl
        assert gl is not None
        shader = int(gl.glCreateShader(shader_type))
        gl.glShaderSource(shader, source)
        gl.glCompileShader(shader)
        ok = gl.glGetShaderiv(shader, gl.GL_COMPILE_STATUS)
        if not ok:
            log = gl.glGetShaderInfoLog(shader).decode("utf-8", errors="replace")
            gl.glDeleteShader(shader)
            raise RuntimeError(f"OpenGL shader compile failed: {log}")
        return shader

    def _create_program(self) -> int:
        gl = self._gl
        assert gl is not None
        vertex_source = """
            #version 330 core
            layout(location = 0) in vec3 in_position;
            layout(location = 1) in vec2 in_uv;
            layout(location = 2) in vec3 in_normal;
            uniform mat4 u_mvp;
            out vec2 v_uv;
            void main() {
                v_uv = in_uv;
                gl_Position = u_mvp * vec4(in_position, 1.0);
            }
        """
        fragment_source = """
            #version 330 core
            in vec2 v_uv;
            uniform sampler2D u_texture;
            uniform vec3 u_color_factor;
            out vec4 out_color;
            void main() {
                vec4 texel = texture(u_texture, v_uv);
                vec3 rgb = clamp(texel.rgb * u_color_factor, 0.0, 1.0);
                rgb = pow(rgb, vec3(1.0 / 2.2));
                out_color = vec4(rgb, texel.a);
            }
        """
        vertex_shader = self._compile_shader(self._gl.GL_VERTEX_SHADER, vertex_source)
        fragment_shader = self._compile_shader(
            self._gl.GL_FRAGMENT_SHADER,
            fragment_source,
        )
        program = int(gl.glCreateProgram())
        gl.glAttachShader(program, vertex_shader)
        gl.glAttachShader(program, fragment_shader)
        gl.glLinkProgram(program)
        gl.glDeleteShader(vertex_shader)
        gl.glDeleteShader(fragment_shader)
        ok = gl.glGetProgramiv(program, gl.GL_LINK_STATUS)
        if not ok:
            log = gl.glGetProgramInfoLog(program).decode("utf-8", errors="replace")
            gl.glDeleteProgram(program)
            raise RuntimeError(f"OpenGL program link failed: {log}")
        return program

    def _depth_internal_format(self) -> int:
        gl = self._gl
        assert gl is not None
        if self.depth_format == "depth32f":
            return int(gl.GL_DEPTH_COMPONENT32F)
        return int(gl.GL_DEPTH_COMPONENT24)

    def _resolve_texture_anisotropy(self) -> None:
        if self.texture_mode != "stable_mipmap":
            self.effective_anisotropy = 1.0
            self.anisotropy_reason = "disabled_for_texture_mode"
            return
        if self.requested_anisotropy <= 1:
            self.effective_anisotropy = 1.0
            self.anisotropy_reason = "requested_1"
            return
        gl = self._gl
        assert gl is not None
        max_param = int(
            getattr(
                gl,
                "GL_MAX_TEXTURE_MAX_ANISOTROPY_EXT",
                _GL_MAX_TEXTURE_MAX_ANISOTROPY_EXT,
            )
        )
        try:
            raw_max = gl.glGetFloatv(max_param)
            max_supported = float(np.asarray(raw_max, dtype=np.float32).reshape(-1)[0])
        except Exception as exc:
            self.effective_anisotropy = 1.0
            self.anisotropy_reason = f"unsupported:{type(exc).__name__}"
            return
        if max_supported <= 1.0:
            self.effective_anisotropy = 1.0
            self.anisotropy_reason = f"unsupported:max={max_supported:.3g}"
            return
        self.effective_anisotropy = float(
            min(float(self.requested_anisotropy), max_supported)
        )
        if self.effective_anisotropy < float(self.requested_anisotropy):
            self.anisotropy_reason = f"clamped_to_{max_supported:.3g}"
        else:
            self.anisotropy_reason = "enabled"

    def _apply_texture_anisotropy(self) -> None:
        if self.texture_mode != "stable_mipmap" or self.effective_anisotropy <= 1.0:
            return
        gl = self._gl
        assert gl is not None
        texture_param = int(
            getattr(gl, "GL_TEXTURE_MAX_ANISOTROPY_EXT", _GL_TEXTURE_MAX_ANISOTROPY_EXT)
        )
        try:
            gl.glTexParameterf(
                gl.GL_TEXTURE_2D,
                texture_param,
                float(self.effective_anisotropy),
            )
        except Exception as exc:
            self.effective_anisotropy = 1.0
            self.anisotropy_reason = f"set_failed:{type(exc).__name__}"

    def _texture_wrap_mode(self) -> int:
        gl = self._gl
        assert gl is not None
        if self.texture_mode in {"stable", "stable_mipmap"}:
            return int(gl.GL_CLAMP_TO_EDGE)
        return int(gl.GL_REPEAT)

    def _create_solid_texture(self, rgba: tuple[int, int, int, int]) -> int:
        gl = self._gl
        assert gl is not None
        texture_id = int(gl.glGenTextures(1))
        pixel = np.asarray([[rgba]], dtype=np.uint8)
        gl.glBindTexture(gl.GL_TEXTURE_2D, texture_id)
        gl.glTexImage2D(
            gl.GL_TEXTURE_2D,
            0,
            gl.GL_RGBA8,
            1,
            1,
            0,
            gl.GL_RGBA,
            gl.GL_UNSIGNED_BYTE,
            pixel,
        )
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
        wrap_mode = self._texture_wrap_mode()
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, wrap_mode)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, wrap_mode)
        gl.glBindTexture(gl.GL_TEXTURE_2D, 0)
        return texture_id

    def _texture_for_material(self, material: _MaterialInfo | None) -> int:
        gl = self._gl
        assert gl is not None
        if material is None or material.texture_name is None:
            return self._white_texture
        texture_name = str(material.texture_name)
        cached = self._texture_cache.get(texture_name)
        if cached is not None:
            return cached

        texture_path = self.scene_root / texture_name
        image = Image.open(texture_path).convert("RGBA")
        image = image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
        pixels = np.ascontiguousarray(np.asarray(image, dtype=np.uint8))
        texture_id = int(gl.glGenTextures(1))
        gl.glBindTexture(gl.GL_TEXTURE_2D, texture_id)
        gl.glTexImage2D(
            gl.GL_TEXTURE_2D,
            0,
            gl.GL_RGBA8,
            int(image.width),
            int(image.height),
            0,
            gl.GL_RGBA,
            gl.GL_UNSIGNED_BYTE,
            pixels,
        )
        if self.texture_mode in {"legacy", "stable_mipmap"}:
            gl.glGenerateMipmap(gl.GL_TEXTURE_2D)
            gl.glTexParameteri(
                gl.GL_TEXTURE_2D,
                gl.GL_TEXTURE_MIN_FILTER,
                gl.GL_LINEAR_MIPMAP_LINEAR,
            )
        else:
            gl.glTexParameteri(
                gl.GL_TEXTURE_2D,
                gl.GL_TEXTURE_MIN_FILTER,
                gl.GL_LINEAR,
            )
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
        wrap_mode = self._texture_wrap_mode()
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, wrap_mode)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, wrap_mode)
        self._apply_texture_anisotropy()
        gl.glBindTexture(gl.GL_TEXTURE_2D, 0)
        self._texture_cache[texture_name] = texture_id
        return texture_id

    def _create_fbo_targets(self) -> None:
        gl = self._gl
        assert gl is not None
        depth_internal_format = self._depth_internal_format()
        for name in ("left", "right", "center"):
            fbo = int(gl.glGenFramebuffers(1))
            color = int(gl.glGenTextures(1))
            depth = int(gl.glGenRenderbuffers(1))
            gl.glBindTexture(gl.GL_TEXTURE_2D, color)
            gl.glTexImage2D(
                gl.GL_TEXTURE_2D,
                0,
                gl.GL_RGBA8,
                self.width,
                self.height,
                0,
                gl.GL_RGBA,
                gl.GL_UNSIGNED_BYTE,
                None,
            )
            gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
            gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
            gl.glTexParameteri(
                gl.GL_TEXTURE_2D,
                gl.GL_TEXTURE_WRAP_S,
                gl.GL_CLAMP_TO_EDGE,
            )
            gl.glTexParameteri(
                gl.GL_TEXTURE_2D,
                gl.GL_TEXTURE_WRAP_T,
                gl.GL_CLAMP_TO_EDGE,
            )
            gl.glBindRenderbuffer(gl.GL_RENDERBUFFER, depth)
            gl.glRenderbufferStorage(
                gl.GL_RENDERBUFFER,
                depth_internal_format,
                self.width,
                self.height,
            )
            gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, fbo)
            gl.glFramebufferTexture2D(
                gl.GL_FRAMEBUFFER,
                gl.GL_COLOR_ATTACHMENT0,
                gl.GL_TEXTURE_2D,
                color,
                0,
            )
            gl.glFramebufferRenderbuffer(
                gl.GL_FRAMEBUFFER,
                gl.GL_DEPTH_ATTACHMENT,
                gl.GL_RENDERBUFFER,
                depth,
            )
            status = gl.glCheckFramebufferStatus(gl.GL_FRAMEBUFFER)
            if status != gl.GL_FRAMEBUFFER_COMPLETE:
                raise RuntimeError(f"OpenGL framebuffer is incomplete: 0x{int(status):x}")
            target = {
                "fbo": fbo,
                "color": color,
                "depth": depth,
                "draw_fbo": fbo,
            }
            if self.msaa_samples > 1:
                msaa_fbo = int(gl.glGenFramebuffers(1))
                msaa_color = int(gl.glGenRenderbuffers(1))
                msaa_depth = int(gl.glGenRenderbuffers(1))
                gl.glBindRenderbuffer(gl.GL_RENDERBUFFER, msaa_color)
                gl.glRenderbufferStorageMultisample(
                    gl.GL_RENDERBUFFER,
                    self.msaa_samples,
                    gl.GL_RGBA8,
                    self.width,
                    self.height,
                )
                gl.glBindRenderbuffer(gl.GL_RENDERBUFFER, msaa_depth)
                gl.glRenderbufferStorageMultisample(
                    gl.GL_RENDERBUFFER,
                    self.msaa_samples,
                    depth_internal_format,
                    self.width,
                    self.height,
                )
                gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, msaa_fbo)
                gl.glFramebufferRenderbuffer(
                    gl.GL_FRAMEBUFFER,
                    gl.GL_COLOR_ATTACHMENT0,
                    gl.GL_RENDERBUFFER,
                    msaa_color,
                )
                gl.glFramebufferRenderbuffer(
                    gl.GL_FRAMEBUFFER,
                    gl.GL_DEPTH_ATTACHMENT,
                    gl.GL_RENDERBUFFER,
                    msaa_depth,
                )
                status = gl.glCheckFramebufferStatus(gl.GL_FRAMEBUFFER)
                if status != gl.GL_FRAMEBUFFER_COMPLETE:
                    raise RuntimeError(
                        f"OpenGL MSAA framebuffer is incomplete: 0x{int(status):x}"
                    )
                target.update(
                    {
                        "draw_fbo": msaa_fbo,
                        "msaa_fbo": msaa_fbo,
                        "msaa_color": msaa_color,
                        "msaa_depth": msaa_depth,
                    }
                )
            self._fbo_targets[name] = target
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0)
        gl.glBindTexture(gl.GL_TEXTURE_2D, 0)
        gl.glBindRenderbuffer(gl.GL_RENDERBUFFER, 0)

    def _compute_world_transform(self, layout: Any) -> np.ndarray:
        if self._cache_payload is None:
            raise RuntimeError(
                "Native GL renderer needs scene_analysis_cache_v4.pkl.gz for scene placement."
            )
        target_xy = np.asarray(
            self.manifest.get("target_table_size_m", [0.95, 0.68]),
            dtype=np.float32,
        )
        bounds_min, bounds_max = self._cache_payload[
            "asset_table_scale_reference_bounds"
        ]
        bounds_min = np.asarray(bounds_min, dtype=np.float32)
        bounds_max = np.asarray(bounds_max, dtype=np.float32)
        extent_xy = np.array(
            [
                float(bounds_max[0] - bounds_min[0]),
                float(bounds_max[2] - bounds_min[2]),
            ],
            dtype=np.float32,
        )
        scene_scale = float(
            max(
                float(target_xy[0]) / max(float(extent_xy[0]), 1.0e-4),
                float(target_xy[1]) / max(float(extent_xy[1]), 1.0e-4),
            )
        )
        rotation = _rotation_x_matrix(
            float(self.manifest.get("x_rotation_degrees", -90.0))
        )
        scale = np.diag([scene_scale, scene_scale, scene_scale, 1.0]).astype(
            np.float32
        )
        pre_translation = rotation @ scale
        startup_center = np.asarray(
            self._cache_payload["asset_startup_table_patch"]["center"],
            dtype=np.float32,
        ).reshape(1, 3)
        startup_world = _transform_points(startup_center, pre_translation)[0]
        translation = np.asarray(layout.table_top_center, dtype=np.float32) - startup_world
        world_transform = np.eye(4, dtype=np.float32)
        world_transform[:3, 3] = translation
        return (world_transform @ pre_translation).astype(np.float32)

    def _resolve_msaa_target(self, target: dict[str, int]) -> None:
        if int(target.get("msaa_fbo", 0)) == 0:
            return
        gl = self._gl
        assert gl is not None
        gl.glBindFramebuffer(gl.GL_READ_FRAMEBUFFER, int(target["msaa_fbo"]))
        gl.glBindFramebuffer(gl.GL_DRAW_FRAMEBUFFER, int(target["fbo"]))
        gl.glReadBuffer(gl.GL_COLOR_ATTACHMENT0)
        gl.glDrawBuffer(gl.GL_COLOR_ATTACHMENT0)
        gl.glBlitFramebuffer(
            0,
            0,
            self.width,
            self.height,
            0,
            0,
            self.width,
            self.height,
            gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT,
            gl.GL_NEAREST,
        )

    def set_layout(self, layout: Any) -> None:
        self._platform.make_current()
        self._world_transform = self._compute_world_transform(layout)
        self._delete_meshes()
        self._upload_meshes()

    def _upload_meshes(self) -> None:
        gl = self._gl
        assert gl is not None
        matrix3 = self._world_transform[:3, :3]
        normal_matrix = np.linalg.inv(matrix3).T.astype(np.float32)
        meshes: list[_GlMesh] = []
        for group in self._asset_groups:
            vertices = np.asarray(group.vertices, dtype=np.float32)
            positions = _transform_points(vertices[:, :3], self._world_transform)
            normals = _normalize_rows(vertices[:, 5:8] @ normal_matrix.T)
            upload = np.ascontiguousarray(
                np.column_stack([positions, vertices[:, 3:5], normals]).astype(
                    np.float32
                )
            )
            indices = np.ascontiguousarray(group.indices.astype(np.uint32))
            material = self._materials.get(group.material_name)
            texture_id = self._texture_for_material(material)
            if material is None:
                color_factor = np.ones((3,), dtype=np.float32)
            elif material.texture_name is None:
                color_factor = np.clip(
                    material.kd + material.ke,
                    0.0,
                    1.0,
                ).astype(np.float32)
            else:
                color_factor = np.clip(material.kd, 0.0, 1.0).astype(np.float32)

            vao = int(gl.glGenVertexArrays(1))
            vbo = int(gl.glGenBuffers(1))
            ebo = int(gl.glGenBuffers(1))
            gl.glBindVertexArray(vao)
            gl.glBindBuffer(gl.GL_ARRAY_BUFFER, vbo)
            gl.glBufferData(
                gl.GL_ARRAY_BUFFER,
                int(upload.nbytes),
                upload,
                gl.GL_STATIC_DRAW,
            )
            gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, ebo)
            gl.glBufferData(
                gl.GL_ELEMENT_ARRAY_BUFFER,
                int(indices.nbytes),
                indices,
                gl.GL_STATIC_DRAW,
            )
            stride = int(upload.shape[1] * upload.dtype.itemsize)
            gl.glEnableVertexAttribArray(0)
            gl.glVertexAttribPointer(0, 3, gl.GL_FLOAT, gl.GL_FALSE, stride, None)
            gl.glEnableVertexAttribArray(1)
            gl.glVertexAttribPointer(
                1,
                2,
                gl.GL_FLOAT,
                gl.GL_FALSE,
                stride,
                ctypes.c_void_p(12),
            )
            gl.glEnableVertexAttribArray(2)
            gl.glVertexAttribPointer(
                2,
                3,
                gl.GL_FLOAT,
                gl.GL_FALSE,
                stride,
                ctypes.c_void_p(20),
            )
            gl.glBindVertexArray(0)
            meshes.append(
                _GlMesh(
                    material_name=group.material_name,
                    vao=vao,
                    vbo=vbo,
                    ebo=ebo,
                    texture_id=texture_id,
                    index_count=int(indices.shape[0]),
                    vertex_count=int(upload.shape[0]),
                    triangle_count=int(group.triangle_count),
                    color_factor=color_factor,
                )
            )
        self._meshes = meshes

    def _ensure_cuda_readback_targets(self) -> None:
        width = int(self.width)
        height = int(self.height)
        dims = (width, height)
        if self._interop_dims == dims:
            return

        self._delete_cuda_readback_targets()
        gl = self._gl
        assert gl is not None
        color_bytes = width * height * 4
        depth_bytes = width * height * np.dtype(np.float32).itemsize
        self._color_pbo = _InteropBuffer.create(
            gl.GL_PIXEL_PACK_BUFFER,
            color_bytes,
            usage=gl.GL_STREAM_READ,
            map_flags=graphics_map_flags.READ_ONLY,
        )
        self._depth_pbo = _InteropBuffer.create(
            gl.GL_PIXEL_PACK_BUFFER,
            depth_bytes,
            usage=gl.GL_STREAM_READ,
            map_flags=graphics_map_flags.READ_ONLY,
        )
        self._color_tensors = [
            torch.empty(
                (height, width, 4),
                dtype=torch.float32,
                device=self.device,
            )
            for _ in range(self._output_ring_size)
        ]
        self._depth_tensors = [
            torch.empty(
                (height, width),
                dtype=torch.float32,
                device=self.device,
            )
            for _ in range(self._output_ring_size)
        ]
        self._output_ring_index = 0
        self._interop_dims = dims

    def _next_output_tensors(self) -> tuple[torch.Tensor, torch.Tensor]:
        color_tensor = self._color_tensors[self._output_ring_index]
        depth_tensor = self._depth_tensors[self._output_ring_index]
        self._output_ring_index = (
            self._output_ring_index + 1
        ) % self._output_ring_size
        return color_tensor, depth_tensor

    def _launch_unpack_kernel(
        self,
        color_ptr: int,
        depth_ptr: int,
        output_color_tensor: torch.Tensor,
        output_depth_tensor: torch.Tensor,
    ) -> None:
        pixel_count = int(self.width) * int(self.height)
        block_size = 256
        grid_size = max((pixel_count + block_size - 1) // block_size, 1)
        unpack_function = _get_unpack_function()
        unpack_function.prepared_call(
            (grid_size, 1, 1),
            (block_size, 1, 1),
            int(color_ptr),
            int(depth_ptr),
            int(output_color_tensor.data_ptr()),
            int(output_depth_tensor.data_ptr()),
            int(self.width),
            int(self.height),
            np.float32(self.znear),
            np.float32(self.zfar),
            int(1),
        )

    def _draw_eye_to_target(
        self,
        camera_pose_world: np.ndarray,
        intrinsic: np.ndarray,
        target_name: str,
    ) -> tuple[float, int, int]:
        if not self._meshes:
            raise RuntimeError("Native GL scene layout has not been configured.")
        gl = self._gl
        assert gl is not None
        self._platform.make_current()
        target = self._fbo_targets[str(target_name)]
        projection = _projection_from_intrinsic(
            intrinsic,
            self.width,
            self.height,
            self.znear,
            self.zfar,
        )
        view = np.linalg.inv(np.asarray(camera_pose_world, dtype=np.float32)).astype(
            np.float32
        )
        mvp = (projection @ view).astype(np.float32)

        start = time.perf_counter()
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, int(target["draw_fbo"]))
        gl.glViewport(0, 0, self.width, self.height)
        gl.glUseProgram(self._program)
        gl.glUniformMatrix4fv(self._uniform_mvp, 1, gl.GL_TRUE, mvp)
        gl.glUniform1i(self._uniform_texture, 0)
        gl.glActiveTexture(gl.GL_TEXTURE0)
        clear = [float(v) / 255.0 for v in _DEFAULT_CLEAR_RGBA]
        gl.glClearColor(clear[0], clear[1], clear[2], clear[3])
        gl.glClearDepth(1.0)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        draw_calls = 0
        triangle_count = 0
        for mesh in self._meshes:
            gl.glBindTexture(gl.GL_TEXTURE_2D, mesh.texture_id)
            gl.glUniform3fv(self._uniform_color_factor, 1, mesh.color_factor)
            gl.glBindVertexArray(mesh.vao)
            gl.glDrawElements(
                gl.GL_TRIANGLES,
                mesh.index_count,
                gl.GL_UNSIGNED_INT,
                None,
            )
            draw_calls += 1
            triangle_count += mesh.triangle_count
        gl.glBindVertexArray(0)
        gl.glBindTexture(gl.GL_TEXTURE_2D, 0)
        gl.glUseProgram(0)
        self._resolve_msaa_target(target)
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0)
        return (time.perf_counter() - start) * 1000.0, draw_calls, triangle_count

    def _read_target_to_cuda(self, target_name: str) -> tuple[torch.Tensor, torch.Tensor, float]:
        gl = self._gl
        assert gl is not None
        self._platform.make_current()
        self._ensure_cuda_readback_targets()
        assert self._color_pbo is not None
        assert self._depth_pbo is not None
        output_color_tensor, output_depth_tensor = self._next_output_tensors()
        target = self._fbo_targets[str(target_name)]

        read_start = time.perf_counter()
        gl.glBindFramebuffer(gl.GL_READ_FRAMEBUFFER, target["fbo"])
        gl.glReadBuffer(gl.GL_COLOR_ATTACHMENT0)
        gl.glBindBuffer(gl.GL_PIXEL_PACK_BUFFER, int(self._depth_pbo.gl_id))
        gl.glReadPixels(
            0,
            0,
            self.width,
            self.height,
            gl.GL_DEPTH_COMPONENT,
            gl.GL_FLOAT,
            ctypes.c_void_p(0),
        )
        gl.glBindBuffer(gl.GL_PIXEL_PACK_BUFFER, int(self._color_pbo.gl_id))
        gl.glReadPixels(
            0,
            0,
            self.width,
            self.height,
            gl.GL_RGBA,
            gl.GL_UNSIGNED_BYTE,
            ctypes.c_void_p(0),
        )
        gl.glBindBuffer(gl.GL_PIXEL_PACK_BUFFER, 0)
        gl.glBindFramebuffer(gl.GL_READ_FRAMEBUFFER, 0)
        gl.glFlush()

        color_mapping = self._color_pbo.registered_buffer.map()
        depth_mapping = self._depth_pbo.registered_buffer.map()
        try:
            color_ptr, _ = color_mapping.device_ptr_and_size()
            depth_ptr, _ = depth_mapping.device_ptr_and_size()
            self._launch_unpack_kernel(
                color_ptr=color_ptr,
                depth_ptr=depth_ptr,
                output_color_tensor=output_color_tensor,
                output_depth_tensor=output_depth_tensor,
            )
        finally:
            try:
                depth_mapping.unmap()
            finally:
                color_mapping.unmap()
        return output_color_tensor, output_depth_tensor, (
            time.perf_counter() - read_start
        ) * 1000.0

    def render_eye(
        self,
        camera_pose_world: np.ndarray,
        intrinsic: np.ndarray,
        width: int | None = None,
        height: int | None = None,
        *,
        target_name: str = "center",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if width is not None and int(width) != int(self.width):
            raise ValueError(
                f"Native GL renderer is fixed at width={self.width}, got {width}."
            )
        if height is not None and int(height) != int(self.height):
            raise ValueError(
                f"Native GL renderer is fixed at height={self.height}, got {height}."
            )
        draw_ms, draw_calls, triangle_count = self._draw_eye_to_target(
            camera_pose_world,
            intrinsic,
            str(target_name),
        )
        color, depth, readback_ms = self._read_target_to_cuda(str(target_name))
        self._last_render_debug = {
            "request_kind": "full",
            "effective_backend": "native_gl",
            "native_gl_draw_wall_ms": float(draw_ms),
            "native_gl_readback_wall_ms": float(readback_ms),
            "draw_calls": int(draw_calls),
            "triangle_count": int(triangle_count),
            "width": int(self.width),
            "height": int(self.height),
            "native_gl_texture_mode": self.texture_mode,
            "native_gl_requested_anisotropy": int(self.requested_anisotropy),
            "native_gl_effective_anisotropy": float(self.effective_anisotropy),
            "native_gl_anisotropy_reason": str(self.anisotropy_reason),
            "native_gl_msaa_samples": int(self.msaa_samples),
            "native_gl_depth_format": self.depth_format,
        }
        return color, depth

    def read_color(self, target_name: str = "center") -> np.ndarray:
        gl = self._gl
        assert gl is not None
        self._platform.make_current()
        target = self._fbo_targets[str(target_name)]
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, target["fbo"])
        pixels = gl.glReadPixels(
            0,
            0,
            self.width,
            self.height,
            gl.GL_RGBA,
            gl.GL_UNSIGNED_BYTE,
        )
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0)
        array = np.frombuffer(pixels, dtype=np.uint8).reshape(
            self.height,
            self.width,
            4,
        )
        return np.ascontiguousarray(np.flipud(array))

    def pyrender_readback_mode(self) -> str:
        return "native_gl_cuda_interop"

    def pyrender_readback_reason(self) -> str | None:
        return None

    def scene_analysis_cache_debug(self) -> dict[str, Any]:
        return dict(self._scene_analysis_cache_debug)

    def last_render_debug(self) -> dict[str, Any] | None:
        if self._last_render_debug is None:
            return None
        return dict(self._last_render_debug)

    def stats(self) -> dict[str, Any]:
        return {
            "backend": "native_gl",
            "layer": "background" if self.render_background else "full",
            "width": self.width,
            "height": self.height,
            "draw_calls_per_eye": int(len(self._meshes)),
            "triangle_count_per_eye": int(
                sum(mesh.triangle_count for mesh in self._meshes)
            ),
            "vertex_count_per_eye": int(sum(mesh.vertex_count for mesh in self._meshes)),
            "material_count": int(len(self._meshes)),
            "texture_count": int(len(self._texture_cache)),
            "readback_mode": self.pyrender_readback_mode(),
            "texture_mode": self.texture_mode,
            "native_gl_requested_anisotropy": int(self.requested_anisotropy),
            "native_gl_effective_anisotropy": float(self.effective_anisotropy),
            "native_gl_anisotropy_reason": str(self.anisotropy_reason),
            "msaa_samples": int(self.msaa_samples),
            "depth_format": self.depth_format,
        }

    def _delete_cuda_readback_targets(self) -> None:
        for buffer in (self._color_pbo, self._depth_pbo):
            if buffer is None:
                continue
            try:
                buffer.delete()
            except Exception:
                pass
        self._color_pbo = None
        self._depth_pbo = None
        self._color_tensors = []
        self._depth_tensors = []
        self._interop_dims = None

    def _delete_meshes(self) -> None:
        gl = self._gl
        if gl is None:
            self._meshes = []
            return
        for mesh in self._meshes:
            try:
                gl.glDeleteBuffers(1, [int(mesh.ebo)])
                gl.glDeleteBuffers(1, [int(mesh.vbo)])
                gl.glDeleteVertexArrays(1, [int(mesh.vao)])
            except Exception:
                pass
        self._meshes = []

    def delete(self) -> None:
        gl = self._gl
        if gl is not None:
            try:
                self._platform.make_current()
            except Exception:
                pass
            self._delete_cuda_readback_targets()
            self._delete_meshes()
            try:
                for target in self._fbo_targets.values():
                    if int(target.get("msaa_depth", 0)):
                        gl.glDeleteRenderbuffers(1, [int(target["msaa_depth"])])
                    if int(target.get("msaa_color", 0)):
                        gl.glDeleteRenderbuffers(1, [int(target["msaa_color"])])
                    if int(target.get("msaa_fbo", 0)):
                        gl.glDeleteFramebuffers(1, [int(target["msaa_fbo"])])
                    gl.glDeleteRenderbuffers(1, [int(target["depth"])])
                    gl.glDeleteTextures(1, [int(target["color"])])
                    gl.glDeleteFramebuffers(1, [int(target["fbo"])])
            except Exception:
                pass
            self._fbo_targets = {}
            try:
                for texture_id in self._texture_cache.values():
                    gl.glDeleteTextures(1, [int(texture_id)])
                if self._white_texture:
                    gl.glDeleteTextures(1, [int(self._white_texture)])
            except Exception:
                pass
            self._texture_cache = {}
            try:
                if self._program:
                    gl.glDeleteProgram(int(self._program))
            except Exception:
                pass
        if self._platform is not None:
            try:
                self._platform.delete_context()
            except Exception:
                pass
            self._platform = None
