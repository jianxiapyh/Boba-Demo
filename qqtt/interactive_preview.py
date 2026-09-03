from __future__ import annotations


INTERACTIVE_PREVIEW_ASPECT_WIDTH = 16
INTERACTIVE_PREVIEW_ASPECT_HEIGHT = 9
INTERACTIVE_PREVIEW_MAX_SCALE_UNITS = 60


def resolve_interactive_preview_render_size(
    framebuffer_width: int,
    framebuffer_height: int,
) -> tuple[int, int]:
    """Choose an exact 16:9 render size capped at 960x540.

    The desktop window may be much larger than the render texture. Keeping the
    spectator render bounded avoids turning a larger operator window into a
    proportional increase in Gaussian-rendering cost.
    """

    framebuffer_width = int(framebuffer_width)
    framebuffer_height = int(framebuffer_height)
    if framebuffer_width <= 0 or framebuffer_height <= 0:
        raise ValueError("Interactive preview framebuffer dimensions must be positive.")
    scale_units = min(
        framebuffer_width // INTERACTIVE_PREVIEW_ASPECT_WIDTH,
        framebuffer_height // INTERACTIVE_PREVIEW_ASPECT_HEIGHT,
        INTERACTIVE_PREVIEW_MAX_SCALE_UNITS,
    )
    scale_units = max(1, int(scale_units))
    return (
        INTERACTIVE_PREVIEW_ASPECT_WIDTH * scale_units,
        INTERACTIVE_PREVIEW_ASPECT_HEIGHT * scale_units,
    )


def fit_interactive_preview_viewport(
    framebuffer_width: int,
    framebuffer_height: int,
    content_width: int,
    content_height: int,
) -> tuple[int, int, int, int]:
    """Fit content inside a framebuffer without changing its aspect ratio."""

    framebuffer_width = max(1, int(framebuffer_width))
    framebuffer_height = max(1, int(framebuffer_height))
    content_width = max(1, int(content_width))
    content_height = max(1, int(content_height))
    if framebuffer_width * content_height > framebuffer_height * content_width:
        viewport_height = framebuffer_height
        viewport_width = max(
            1,
            int(round(viewport_height * float(content_width) / content_height)),
        )
    else:
        viewport_width = framebuffer_width
        viewport_height = max(
            1,
            int(round(viewport_width * float(content_height) / content_width)),
        )
    return (
        (framebuffer_width - viewport_width) // 2,
        (framebuffer_height - viewport_height) // 2,
        viewport_width,
        viewport_height,
    )


_PREVIEW_VERTEX_SHADER = """
#version 330 core
out vec2 uv;
const vec2 V[4]=vec2[4](vec2(-1,-1),vec2(1,-1),vec2(-1,1),vec2(1,1));
const vec2 T[4]=vec2[4](vec2(0,0),vec2(1,0),vec2(0,1),vec2(1,1));
void main(){ gl_Position=vec4(V[gl_VertexID],0,1); uv=T[gl_VertexID]; }
"""

_PREVIEW_FRAGMENT_SHADER = """
#version 330 core
in vec2 uv; out vec4 frag; uniform sampler2D uTex;
void main(){ frag = texture(uTex, vec2(uv.x, 1.0 - uv.y)); }
"""


class InteractivePreviewRenderer:
    """Own the local preview resources for the lifetime of one GLFW window.

    Object changes start a new trainer episode while retaining the same GLFW and
    CUDA contexts. Keeping this renderer alive across those episodes avoids
    unregistering and immediately re-registering CUDA/GL interop buffers on the
    live window.
    """

    def __init__(
        self,
        window,
        width: int,
        height: int,
        *,
        device,
        glfw_module=None,
        gl_module=None,
        uploader_factory=None,
    ):
        self.window = window
        self.width = int(width)
        self.height = int(height)
        if self.width <= 0 or self.height <= 0:
            raise ValueError("Interactive preview dimensions must be positive.")
        self.device = device
        self._glfw = glfw_module
        self._gl = gl_module
        self._uploader_factory = uploader_factory
        self._texture = None
        self._uploader = None
        self._program = None
        self._vao = None
        self._framebuffer_size = None
        self._host_frame_staging = None
        self._initialized = False
        self._deleted = False

    @property
    def initialized(self) -> bool:
        return bool(self._initialized and not self._deleted)

    @property
    def deleted(self) -> bool:
        return bool(self._deleted)

    def is_compatible(self, window, width: int, height: int, *, device) -> bool:
        return bool(
            not self._deleted
            and self.window == window
            and self.width == int(width)
            and self.height == int(height)
            and str(self.device) == str(device)
        )

    def _load_dependencies(self) -> None:
        if self._glfw is None:
            import glfw

            self._glfw = glfw
        if self._gl is None:
            from OpenGL import GL as gl

            self._gl = gl
        if self._uploader_factory is None:
            from qqtt.pyrender_cuda_bridge import PreviewTextureCudaUploader

            self._uploader_factory = PreviewTextureCudaUploader

    def _make_context_current(self) -> None:
        if self._glfw.get_current_context() != self.window:
            self._glfw.make_context_current(self.window)

    def _framebuffer_size_callback(self, _window, width, height) -> None:
        self._framebuffer_size = (max(1, int(width)), max(1, int(height)))

    @staticmethod
    def _shader_error_text(error) -> str:
        if isinstance(error, bytes):
            return error.decode(errors="replace")
        return str(error)

    def _compile_shader(self, kind, source: str) -> int:
        shader_id = int(self._gl.glCreateShader(kind))
        self._gl.glShaderSource(shader_id, source)
        self._gl.glCompileShader(shader_id)
        if not self._gl.glGetShaderiv(shader_id, self._gl.GL_COMPILE_STATUS):
            error = self._shader_error_text(self._gl.glGetShaderInfoLog(shader_id))
            self._gl.glDeleteShader(shader_id)
            raise RuntimeError(error)
        return shader_id

    def initialize(self) -> None:
        if self._deleted:
            raise RuntimeError("Interactive preview renderer has already been deleted.")
        self._load_dependencies()
        self._make_context_current()
        if self._initialized:
            return
        self._glfw.set_framebuffer_size_callback(
            self.window,
            self._framebuffer_size_callback,
        )
        self._framebuffer_size_callback(
            self.window,
            *self._glfw.get_framebuffer_size(self.window),
        )

        vertex_shader = None
        fragment_shader = None
        try:
            self._texture = int(self._gl.glGenTextures(1))
            self._gl.glBindTexture(self._gl.GL_TEXTURE_2D, self._texture)
            self._gl.glTexParameteri(
                self._gl.GL_TEXTURE_2D,
                self._gl.GL_TEXTURE_MIN_FILTER,
                self._gl.GL_LINEAR,
            )
            self._gl.glTexParameteri(
                self._gl.GL_TEXTURE_2D,
                self._gl.GL_TEXTURE_MAG_FILTER,
                self._gl.GL_LINEAR,
            )
            self._gl.glTexImage2D(
                self._gl.GL_TEXTURE_2D,
                0,
                self._gl.GL_RGBA8,
                self.width,
                self.height,
                0,
                self._gl.GL_RGBA,
                self._gl.GL_UNSIGNED_BYTE,
                None,
            )
            self._gl.glBindTexture(self._gl.GL_TEXTURE_2D, 0)
            self._uploader = self._uploader_factory(
                self._texture,
                self.width,
                self.height,
                device=self.device,
            )

            vertex_shader = self._compile_shader(
                self._gl.GL_VERTEX_SHADER,
                _PREVIEW_VERTEX_SHADER,
            )
            fragment_shader = self._compile_shader(
                self._gl.GL_FRAGMENT_SHADER,
                _PREVIEW_FRAGMENT_SHADER,
            )
            self._program = int(self._gl.glCreateProgram())
            self._gl.glAttachShader(self._program, vertex_shader)
            self._gl.glAttachShader(self._program, fragment_shader)
            self._gl.glLinkProgram(self._program)
            if not self._gl.glGetProgramiv(
                self._program,
                self._gl.GL_LINK_STATUS,
            ):
                raise RuntimeError(
                    self._shader_error_text(
                        self._gl.glGetProgramInfoLog(self._program)
                    )
                )
            self._gl.glUseProgram(self._program)
            self._gl.glUniform1i(
                self._gl.glGetUniformLocation(self._program, "uTex"),
                0,
            )
            self._gl.glUseProgram(0)
            self._vao = int(self._gl.glGenVertexArrays(1))
            self._initialized = True
        except Exception:
            self._delete_resources()
            raise
        finally:
            if vertex_shader is not None:
                self._gl.glDeleteShader(vertex_shader)
            if fragment_shader is not None:
                self._gl.glDeleteShader(fragment_shader)

    def _prepare_upload_frame(self, frame_rgba_u8):
        import torch
        import torch.nn.functional as F

        if not torch.is_tensor(frame_rgba_u8):
            return frame_rgba_u8
        if frame_rgba_u8.ndim != 3 or int(frame_rgba_u8.shape[-1]) != 4:
            raise ValueError(
                "Interactive preview frames must have shape (H,W,4), got "
                f"{tuple(frame_rgba_u8.shape)}."
            )
        if tuple(frame_rgba_u8.shape[:2]) != (self.height, self.width):
            source_dtype = frame_rgba_u8.dtype
            source_height = int(frame_rgba_u8.shape[0])
            source_width = int(frame_rgba_u8.shape[1])
            fit_x, fit_y, fit_width, fit_height = (
                fit_interactive_preview_viewport(
                    self.width,
                    self.height,
                    source_width,
                    source_height,
                )
            )
            resized = F.interpolate(
                frame_rgba_u8.permute(2, 0, 1)
                .unsqueeze(0)
                .to(dtype=torch.float32),
                size=(fit_height, fit_width),
                mode="bilinear",
                align_corners=False,
            )
            if source_dtype == torch.uint8:
                resized = resized.round().clamp(0.0, 255.0).to(torch.uint8)
            else:
                resized = resized.to(dtype=source_dtype)
            if (fit_width, fit_height) == (self.width, self.height):
                frame_rgba_u8 = (
                    resized.squeeze(0).permute(1, 2, 0).contiguous()
                )
            else:
                frame_rgba_u8 = torch.zeros(
                    (self.height, self.width, 4),
                    dtype=source_dtype,
                    device=frame_rgba_u8.device,
                )
                frame_rgba_u8[..., 3].fill_(255)
                frame_rgba_u8[
                    fit_y : fit_y + fit_height,
                    fit_x : fit_x + fit_width,
                ].copy_(resized.squeeze(0).permute(1, 2, 0))
                frame_rgba_u8 = frame_rgba_u8.contiguous()
        if frame_rgba_u8.device.type == "cuda":
            return frame_rgba_u8

        target_device = getattr(self._uploader, "device", self.device)
        staging_matches = bool(
            self._host_frame_staging is not None
            and tuple(self._host_frame_staging.shape) == tuple(frame_rgba_u8.shape)
            and self._host_frame_staging.dtype == frame_rgba_u8.dtype
            and str(self._host_frame_staging.device) == str(target_device)
        )
        if not staging_matches:
            self._host_frame_staging = torch.empty(
                tuple(frame_rgba_u8.shape),
                dtype=frame_rgba_u8.dtype,
                device=target_device,
            )
        self._host_frame_staging.copy_(
            frame_rgba_u8,
            non_blocking=bool(frame_rgba_u8.is_pinned()),
        )
        return self._host_frame_staging

    def present(self, frame_rgba_u8) -> None:
        self.initialize()
        self._uploader.upload(self._prepare_upload_frame(frame_rgba_u8))
        fb_width, fb_height = self._framebuffer_size
        viewport = fit_interactive_preview_viewport(
            fb_width,
            fb_height,
            self.width,
            self.height,
        )
        self._gl.glDisable(self._gl.GL_DEPTH_TEST)
        self._gl.glClear(self._gl.GL_COLOR_BUFFER_BIT)
        self._gl.glViewport(*viewport)
        self._gl.glUseProgram(self._program)
        self._gl.glBindVertexArray(self._vao)
        self._gl.glActiveTexture(self._gl.GL_TEXTURE0)
        self._gl.glBindTexture(self._gl.GL_TEXTURE_2D, self._texture)
        self._gl.glDrawArrays(self._gl.GL_TRIANGLE_STRIP, 0, 4)
        self._gl.glBindTexture(self._gl.GL_TEXTURE_2D, 0)
        self._gl.glBindVertexArray(0)
        self._gl.glUseProgram(0)
        self._glfw.swap_buffers(self.window)

    def _delete_resources(self) -> None:
        if self._uploader is not None:
            try:
                self._uploader.delete()
            except Exception:
                pass
            self._uploader = None
        if self._program is not None:
            try:
                self._gl.glDeleteProgram(self._program)
            except Exception:
                pass
            self._program = None
        if self._texture is not None:
            try:
                self._gl.glDeleteTextures([self._texture])
            except Exception:
                pass
            self._texture = None
        if self._vao is not None:
            try:
                self._gl.glDeleteVertexArrays(1, [self._vao])
            except Exception:
                pass
            self._vao = None
        self._host_frame_staging = None
        self._initialized = False

    def delete(self) -> None:
        if self._deleted:
            return
        self._load_dependencies()
        try:
            self._make_context_current()
        except Exception:
            pass
        try:
            self._glfw.set_framebuffer_size_callback(self.window, None)
        except Exception:
            pass
        self._delete_resources()
        self._deleted = True


def reuse_or_create_interactive_preview_renderer(
    existing_renderer,
    window,
    width: int,
    height: int,
    *,
    device,
):
    """Reuse preview interop resources when a new object episode starts."""

    if existing_renderer is not None and existing_renderer.is_compatible(
        window,
        width,
        height,
        device=device,
    ):
        return existing_renderer
    if existing_renderer is not None:
        existing_renderer.delete()
    return InteractivePreviewRenderer(
        window,
        width,
        height,
        device=device,
    )
