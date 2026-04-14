from __future__ import annotations

import ctypes
import os
from dataclasses import dataclass

import numpy as np
import torch
from OpenGL import GL as gl
from pycuda import driver as cuda_driver
from pycuda.compiler import SourceModule
from pycuda.gl import RegisteredBuffer, graphics_map_flags
from pyrender.constants import RenderFlags
from pyrender.renderer import Renderer


_SUPPORT_CACHE: tuple[bool, str] | None = None
_UNPACK_MODULE = None
_UNPACK_FUNCTION = None


def probe_pyrender_cuda_bridge_support() -> tuple[bool, str]:
    global _SUPPORT_CACHE
    if _SUPPORT_CACHE is not None:
        return _SUPPORT_CACHE

    try:
        if not torch.cuda.is_available():
            _SUPPORT_CACHE = (False, "torch.cuda.is_available() is false")
            return _SUPPORT_CACHE
        if not cuda_driver.have_gl_ext():
            _SUPPORT_CACHE = (False, "PyCUDA was built without GL interop support")
            return _SUPPORT_CACHE
        _ = RegisteredBuffer
        _ = graphics_map_flags.READ_ONLY
        _ = SourceModule
    except Exception as exc:  # pragma: no cover - environment-dependent
        _SUPPORT_CACHE = (False, f"{type(exc).__name__}: {exc}")
        return _SUPPORT_CACHE

    _SUPPORT_CACHE = (True, "available")
    return _SUPPORT_CACHE


def _get_unpack_function():
    global _UNPACK_MODULE, _UNPACK_FUNCTION
    if _UNPACK_FUNCTION is not None:
        return _UNPACK_FUNCTION

    _UNPACK_MODULE = SourceModule(
        r"""
        extern "C" __global__ void unpack_rgba_depth(
            const unsigned char* color_in,
            const float* depth_in,
            float* color_out,
            float* depth_out,
            int width,
            int height,
            float z_near,
            float z_far,
            int use_z_far
        ) {
            int pixel_index = blockIdx.x * blockDim.x + threadIdx.x;
            int pixel_count = width * height;
            if (pixel_index >= pixel_count) {
                return;
            }

            int x = pixel_index % width;
            int y = pixel_index / width;
            int src_y = height - 1 - y;
            int src_index = src_y * width + x;
            int src_color_index = src_index * 4;
            int dst_color_index = pixel_index * 4;

            color_out[dst_color_index + 0] = (float) color_in[src_color_index + 0];
            color_out[dst_color_index + 1] = (float) color_in[src_color_index + 1];
            color_out[dst_color_index + 2] = (float) color_in[src_color_index + 2];
            color_out[dst_color_index + 3] = (float) color_in[src_color_index + 3];

            float raw_depth = depth_in[src_index];
            if (!isfinite(raw_depth) || raw_depth >= 1.0f) {
                depth_out[pixel_index] = 0.0f;
                return;
            }

            float ndc_depth = 2.0f * raw_depth - 1.0f;
            float linear_depth = 0.0f;
            if (use_z_far) {
                linear_depth = (2.0f * z_near * z_far) /
                    (z_far + z_near - ndc_depth * (z_far - z_near));
            } else {
                linear_depth = 2.0f * z_near / (1.0f - ndc_depth);
            }
            if (!isfinite(linear_depth) || linear_depth <= 1.0e-6f) {
                depth_out[pixel_index] = 0.0f;
                return;
            }
            depth_out[pixel_index] = linear_depth;
        }
        """,
        no_extern_c=True,
    )
    _UNPACK_FUNCTION = _UNPACK_MODULE.get_function("unpack_rgba_depth")
    _UNPACK_FUNCTION.prepare("PPPPiiffi")
    return _UNPACK_FUNCTION


@dataclass
class _PixelPackBuffer:
    target: int
    gl_id: int
    registered_buffer: RegisteredBuffer
    size_bytes: int

    @classmethod
    def create(cls, target: int, size_bytes: int):
        gl_id = int(gl.glGenBuffers(1))
        gl.glBindBuffer(target, gl_id)
        gl.glBufferData(target, int(size_bytes), None, gl.GL_STREAM_READ)
        gl.glBindBuffer(target, 0)
        registered_buffer = RegisteredBuffer(
            gl_id,
            graphics_map_flags.READ_ONLY,
        )
        return cls(
            target=int(target),
            gl_id=gl_id,
            registered_buffer=registered_buffer,
            size_bytes=int(size_bytes),
        )

    def delete(self) -> None:
        try:
            self.registered_buffer.unregister()
        except Exception:
            pass
        try:
            gl.glDeleteBuffers(1, [int(self.gl_id)])
        except Exception:
            pass


class PyrenderCudaInteropRenderer(Renderer):
    def __init__(
        self,
        viewport_width: int,
        viewport_height: int,
        point_size: float = 1.0,
        device: torch.device | str | None = None,
    ):
        super().__init__(viewport_width, viewport_height, point_size=point_size)
        if device is None:
            device = torch.device("cuda", torch.cuda.current_device())
        self.device = torch.device(device)
        self.readback_mode = "gl_cuda_interop"
        self.fallback_reason: str | None = None
        self._interop_enabled = True
        self._color_pbo: _PixelPackBuffer | None = None
        self._depth_pbo: _PixelPackBuffer | None = None
        self._interop_dims: tuple[int, int] | None = None
        self._output_ring_size = 4
        self._output_ring_index = 0
        self._color_tensors: list[torch.Tensor] = []
        self._depth_tensors: list[torch.Tensor] = []
        self._logged_runtime_fallback = False

    def delete(self):
        self._delete_interop_resources()
        super().delete()

    def _delete_interop_resources(self) -> None:
        if self._color_pbo is not None:
            self._color_pbo.delete()
            self._color_pbo = None
        if self._depth_pbo is not None:
            self._depth_pbo.delete()
            self._depth_pbo = None
        self._interop_dims = None
        self._color_tensors = []
        self._depth_tensors = []
        self._output_ring_index = 0

    def _fallback_to_cpu(self, exc: Exception) -> None:
        self._interop_enabled = False
        self.readback_mode = "cpu_fallback"
        self.fallback_reason = f"{type(exc).__name__}: {exc}"
        self._delete_interop_resources()
        if not self._logged_runtime_fallback:
            print(
                "[pyrender_cuda_bridge] GL-to-CUDA interop failed; "
                f"falling back to pyrender CPU readback ({self.fallback_reason})",
                flush=True,
            )
            self._logged_runtime_fallback = True

    def _ensure_interop_targets(self, width: int, height: int) -> None:
        width = int(width)
        height = int(height)
        dims = (width, height)
        if self._interop_dims == dims:
            return

        self._delete_interop_resources()
        color_bytes = width * height * 4
        depth_bytes = width * height * np.dtype(np.float32).itemsize
        self._color_pbo = _PixelPackBuffer.create(gl.GL_PIXEL_PACK_BUFFER, color_bytes)
        self._depth_pbo = _PixelPackBuffer.create(gl.GL_PIXEL_PACK_BUFFER, depth_bytes)
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
        width: int,
        height: int,
        color_ptr: int,
        depth_ptr: int,
        output_color_tensor: torch.Tensor,
        output_depth_tensor: torch.Tensor,
        z_near: float,
        z_far: float | None,
    ) -> None:
        pixel_count = int(width) * int(height)
        block_size = 256
        grid_size = max((pixel_count + block_size - 1) // block_size, 1)
        unpack_function = _get_unpack_function()
        use_z_far = int(z_far is not None and np.isfinite(z_far) and z_far > 0.0)
        unpack_function.prepared_call(
            (grid_size, 1, 1),
            (block_size, 1, 1),
            int(color_ptr),
            int(depth_ptr),
            int(output_color_tensor.data_ptr()),
            int(output_depth_tensor.data_ptr()),
            int(width),
            int(height),
            np.float32(z_near),
            np.float32(0.0 if z_far is None else z_far),
            int(use_z_far),
        )

    def _read_main_framebuffer(self, scene, flags):
        if (
            not self._interop_enabled
            or not bool(flags & RenderFlags.RGBA)
            or bool(flags & RenderFlags.DEPTH_ONLY)
        ):
            return super()._read_main_framebuffer(scene, flags)

        try:
            return self._read_main_framebuffer_interop(scene)
        except Exception as exc:  # pragma: no cover - environment-dependent
            self._fallback_to_cpu(exc)
            return super()._read_main_framebuffer(scene, flags)

    def _read_main_framebuffer_interop(self, scene):
        width, height = self._main_fb_dims[0], self._main_fb_dims[1]
        self._ensure_interop_targets(width, height)
        assert self._color_pbo is not None
        assert self._depth_pbo is not None
        output_color_tensor, output_depth_tensor = self._next_output_tensors()

        gl.glBindFramebuffer(gl.GL_READ_FRAMEBUFFER, self._main_fb_ms)
        gl.glBindFramebuffer(gl.GL_DRAW_FRAMEBUFFER, self._main_fb)
        gl.glBlitFramebuffer(
            0, 0, width, height, 0, 0, width, height,
            gl.GL_COLOR_BUFFER_BIT, gl.GL_LINEAR,
        )
        gl.glBlitFramebuffer(
            0, 0, width, height, 0, 0, width, height,
            gl.GL_DEPTH_BUFFER_BIT, gl.GL_NEAREST,
        )
        gl.glBindFramebuffer(gl.GL_READ_FRAMEBUFFER, self._main_fb)

        gl.glBindBuffer(gl.GL_PIXEL_PACK_BUFFER, int(self._depth_pbo.gl_id))
        gl.glReadPixels(
            0,
            0,
            width,
            height,
            gl.GL_DEPTH_COMPONENT,
            gl.GL_FLOAT,
            ctypes.c_void_p(0),
        )

        gl.glBindBuffer(gl.GL_PIXEL_PACK_BUFFER, int(self._color_pbo.gl_id))
        gl.glReadPixels(
            0,
            0,
            width,
            height,
            gl.GL_RGBA,
            gl.GL_UNSIGNED_BYTE,
            ctypes.c_void_p(0),
        )
        gl.glBindBuffer(gl.GL_PIXEL_PACK_BUFFER, 0)
        gl.glFlush()

        color_mapping = self._color_pbo.registered_buffer.map()
        depth_mapping = self._depth_pbo.registered_buffer.map()
        try:
            color_ptr, _ = color_mapping.device_ptr_and_size()
            depth_ptr, _ = depth_mapping.device_ptr_and_size()
            self._launch_unpack_kernel(
                width=width,
                height=height,
                color_ptr=color_ptr,
                depth_ptr=depth_ptr,
                output_color_tensor=output_color_tensor,
                output_depth_tensor=output_depth_tensor,
                z_near=float(scene.main_camera_node.camera.znear),
                z_far=scene.main_camera_node.camera.zfar,
            )
        finally:
            try:
                depth_mapping.unmap()
            finally:
                color_mapping.unmap()

        return output_color_tensor, output_depth_tensor


class PyrenderCudaInteropOffscreenRenderer:
    def __init__(
        self,
        viewport_width: int,
        viewport_height: int,
        point_size: float = 1.0,
        device: torch.device | str | None = None,
    ):
        from pyrender.renderer import Renderer as PyrenderRenderer

        self.viewport_width = int(viewport_width)
        self.viewport_height = int(viewport_height)
        self.point_size = float(point_size)
        self._device = device
        self._platform = None
        self._renderer = None
        self.readback_mode = "cpu_fallback"
        self.fallback_reason: str | None = None
        self._fallback_renderer_cls = PyrenderRenderer
        self._create()

    def render(self, scene, flags=RenderFlags.NONE, seg_node_map=None):
        self._platform.make_current()
        if (
            self._platform.viewport_height != self.viewport_height
            or self._platform.viewport_width != self.viewport_width
        ):
            if not self._platform.supports_framebuffers():
                self.delete()
                self._create()

        self._platform.make_current()
        self._renderer.viewport_width = self.viewport_width
        self._renderer.viewport_height = self.viewport_height
        self._renderer.point_size = self.point_size

        if self._platform.supports_framebuffers():
            flags |= RenderFlags.OFFSCREEN
            retval = self._renderer.render(scene, flags, seg_node_map)
        else:  # pragma: no cover - pyrender EGL/pyglet supports framebuffers
            self._renderer.render(scene, flags, seg_node_map)
            depth = self._renderer.read_depth_buf()
            if flags & RenderFlags.DEPTH_ONLY:
                retval = depth
            else:
                color = self._renderer.read_color_buf()
                retval = color, depth

        self.readback_mode = getattr(self._renderer, "readback_mode", "cpu_fallback")
        self.fallback_reason = getattr(self._renderer, "fallback_reason", None)
        self._platform.make_uncurrent()
        return retval

    def delete(self):
        if self._platform is None or self._renderer is None:
            return
        self._platform.make_current()
        self._renderer.delete()
        self._platform.delete_context()
        del self._renderer
        del self._platform
        self._renderer = None
        self._platform = None
        import gc

        gc.collect()

    def _create(self):
        if "PYOPENGL_PLATFORM" not in os.environ:
            from pyrender.platforms.pyglet_platform import PygletPlatform

            self._platform = PygletPlatform(self.viewport_width, self.viewport_height)
        elif os.environ["PYOPENGL_PLATFORM"] == "egl":
            from pyrender.platforms import egl

            device_id = int(os.environ.get("EGL_DEVICE_ID", "0"))
            egl_device = egl.get_device_by_index(device_id)
            self._platform = egl.EGLPlatform(
                self.viewport_width,
                self.viewport_height,
                device=egl_device,
            )
        elif os.environ["PYOPENGL_PLATFORM"] == "osmesa":
            from pyrender.platforms.osmesa import OSMesaPlatform

            self._platform = OSMesaPlatform(self.viewport_width, self.viewport_height)
        else:
            raise ValueError(
                f"Unsupported PyOpenGL platform: {os.environ['PYOPENGL_PLATFORM']}"
            )
        self._platform.init_context()
        self._platform.make_current()

        interop_supported, interop_reason = probe_pyrender_cuda_bridge_support()
        if interop_supported:
            try:
                self._renderer = PyrenderCudaInteropRenderer(
                    self.viewport_width,
                    self.viewport_height,
                    point_size=self.point_size,
                    device=self._device,
                )
                self.readback_mode = "gl_cuda_interop"
                self.fallback_reason = None
                return
            except Exception as exc:  # pragma: no cover - environment-dependent
                self.fallback_reason = f"{type(exc).__name__}: {exc}"
        else:
            self.fallback_reason = interop_reason

        self._renderer = self._fallback_renderer_cls(
            self.viewport_width,
            self.viewport_height,
            point_size=self.point_size,
        )
        self.readback_mode = "cpu_fallback"

    def __del__(self):
        try:
            self.delete()
        except Exception:
            pass
