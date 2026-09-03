from __future__ import annotations

from qqtt.interactive_preview import (
    InteractivePreviewRenderer,
    fit_interactive_preview_viewport,
    resolve_interactive_preview_render_size,
    reuse_or_create_interactive_preview_renderer,
)


class _FakeGlfw:
    def __init__(self):
        self.current_context = None
        self.callback = None
        self.callback_set_count = 0
        self.swap_count = 0

    def get_current_context(self):
        return self.current_context

    def make_context_current(self, window):
        self.current_context = window

    def set_framebuffer_size_callback(self, _window, callback):
        self.callback = callback
        self.callback_set_count += 1

    @staticmethod
    def get_framebuffer_size(_window):
        return (848, 400)

    def swap_buffers(self, _window):
        self.swap_count += 1


class _FakeGl:
    GL_TEXTURE_2D = 1
    GL_TEXTURE_MIN_FILTER = 2
    GL_TEXTURE_MAG_FILTER = 3
    GL_LINEAR = 4
    GL_RGBA8 = 5
    GL_RGBA = 6
    GL_UNSIGNED_BYTE = 7
    GL_VERTEX_SHADER = 8
    GL_FRAGMENT_SHADER = 9
    GL_COMPILE_STATUS = 10
    GL_LINK_STATUS = 11
    GL_DEPTH_TEST = 12
    GL_COLOR_BUFFER_BIT = 13
    GL_TEXTURE0 = 14
    GL_TRIANGLE_STRIP = 15

    def __init__(self):
        self.texture_create_count = 0
        self.deleted_textures = []
        self.deleted_programs = []
        self.deleted_vaos = []
        self.deleted_shaders = []
        self.draw_count = 0
        self.viewports = []
        self._next_shader = 100

    def glGenTextures(self, _count):
        self.texture_create_count += 1
        return 20

    @staticmethod
    def glBindTexture(*_args):
        return None

    @staticmethod
    def glTexParameteri(*_args):
        return None

    @staticmethod
    def glTexImage2D(*_args):
        return None

    def glCreateShader(self, _kind):
        self._next_shader += 1
        return self._next_shader

    @staticmethod
    def glShaderSource(*_args):
        return None

    @staticmethod
    def glCompileShader(*_args):
        return None

    @staticmethod
    def glGetShaderiv(*_args):
        return True

    @staticmethod
    def glGetShaderInfoLog(*_args):
        return b""

    def glDeleteShader(self, shader):
        self.deleted_shaders.append(shader)

    @staticmethod
    def glCreateProgram():
        return 30

    @staticmethod
    def glAttachShader(*_args):
        return None

    @staticmethod
    def glLinkProgram(*_args):
        return None

    @staticmethod
    def glGetProgramiv(*_args):
        return True

    @staticmethod
    def glGetProgramInfoLog(*_args):
        return b""

    @staticmethod
    def glUseProgram(*_args):
        return None

    @staticmethod
    def glUniform1i(*_args):
        return None

    @staticmethod
    def glGetUniformLocation(*_args):
        return 0

    @staticmethod
    def glGenVertexArrays(_count):
        return 40

    def glViewport(self, *args):
        self.viewports.append(tuple(int(value) for value in args))

    @staticmethod
    def glDisable(*_args):
        return None

    @staticmethod
    def glClear(*_args):
        return None

    @staticmethod
    def glBindVertexArray(*_args):
        return None

    @staticmethod
    def glActiveTexture(*_args):
        return None

    def glDrawArrays(self, *_args):
        self.draw_count += 1

    def glDeleteProgram(self, program):
        self.deleted_programs.append(program)

    def glDeleteTextures(self, textures):
        self.deleted_textures.extend(textures)

    def glDeleteVertexArrays(self, _count, vaos):
        self.deleted_vaos.extend(vaos)


class _FakeUploader:
    def __init__(self, texture, width, height, *, device):
        self.texture = texture
        self.width = width
        self.height = height
        self.device = device
        self.upload_count = 0
        self.uploaded_frames = []
        self.delete_count = 0

    def upload(self, frame):
        self.upload_count += 1
        self.uploaded_frames.append(frame)

    def delete(self):
        self.delete_count += 1


def test_large_window_uses_bounded_standard_preview_resolution():
    assert resolve_interactive_preview_render_size(1920, 1080) == (960, 540)
    assert resolve_interactive_preview_render_size(3840, 2160) == (960, 540)
    assert resolve_interactive_preview_render_size(848, 400) == (704, 396)


def test_preview_viewport_preserves_aspect_ratio_with_letterboxing():
    assert fit_interactive_preview_viewport(1200, 800, 960, 540) == (
        0,
        62,
        1200,
        675,
    )
    assert fit_interactive_preview_viewport(1600, 700, 960, 540) == (
        178,
        0,
        1244,
        700,
    )


def test_preview_interop_resources_are_reused_across_object_episodes():
    fake_glfw = _FakeGlfw()
    fake_gl = _FakeGl()
    uploaders = []

    def uploader_factory(*args, **kwargs):
        uploader = _FakeUploader(*args, **kwargs)
        uploaders.append(uploader)
        return uploader

    window = object()
    first_episode_renderer = InteractivePreviewRenderer(
        window,
        1344,
        1344,
        device="cuda:0",
        glfw_module=fake_glfw,
        gl_module=fake_gl,
        uploader_factory=uploader_factory,
    )
    first_episode_renderer.initialize()

    second_episode_renderer = reuse_or_create_interactive_preview_renderer(
        first_episode_renderer,
        window,
        1344,
        1344,
        device="cuda:0",
    )
    second_episode_renderer.present(object())

    assert second_episode_renderer is first_episode_renderer
    assert fake_gl.texture_create_count == 1
    assert len(uploaders) == 1
    assert uploaders[0].upload_count == 1
    assert uploaders[0].delete_count == 0


def test_cpu_switch_progress_frame_is_staged_for_the_uploader():
    import torch

    fake_glfw = _FakeGlfw()
    fake_gl = _FakeGl()
    uploader = None

    def uploader_factory(*args, **kwargs):
        nonlocal uploader
        uploader = _FakeUploader(*args, **kwargs)
        return uploader

    renderer = InteractivePreviewRenderer(
        object(),
        4,
        3,
        device="cpu",
        glfw_module=fake_glfw,
        gl_module=fake_gl,
        uploader_factory=uploader_factory,
    )
    cpu_frame = torch.arange(3 * 4 * 4, dtype=torch.uint8).reshape(3, 4, 4)
    renderer.present(cpu_frame)
    first_staging_frame = uploader.uploaded_frames[-1]
    renderer.present(cpu_frame + 1)

    assert first_staging_frame is not cpu_frame
    assert uploader.uploaded_frames[-1] is first_staging_frame
    assert torch.equal(uploader.uploaded_frames[-1], cpu_frame + 1)


def test_eye_sized_fallback_frame_is_letterboxed_without_stretching():
    import torch

    fake_glfw = _FakeGlfw()
    fake_gl = _FakeGl()
    uploader = None

    def uploader_factory(*args, **kwargs):
        nonlocal uploader
        uploader = _FakeUploader(*args, **kwargs)
        return uploader

    renderer = InteractivePreviewRenderer(
        object(),
        8,
        4,
        device="cpu",
        glfw_module=fake_glfw,
        gl_module=fake_gl,
        uploader_factory=uploader_factory,
    )
    renderer.present(torch.full((12, 12, 4), 127, dtype=torch.uint8))

    uploaded = uploader.uploaded_frames[-1]
    assert uploaded.shape == (4, 8, 4)
    assert uploaded.dtype == torch.uint8
    assert torch.all(uploaded[:, 2:6, :] == 127)
    assert torch.all(uploaded[:, :2, :3] == 0)
    assert torch.all(uploaded[:, 6:, :3] == 0)
    assert torch.all(uploaded[:, :2, 3] == 255)
    assert torch.all(uploaded[:, 6:, 3] == 255)


def test_preview_renderer_final_cleanup_is_idempotent():
    fake_glfw = _FakeGlfw()
    fake_gl = _FakeGl()
    uploader = None

    def uploader_factory(*args, **kwargs):
        nonlocal uploader
        uploader = _FakeUploader(*args, **kwargs)
        return uploader

    renderer = InteractivePreviewRenderer(
        object(),
        1344,
        1344,
        device="cuda:0",
        glfw_module=fake_glfw,
        gl_module=fake_gl,
        uploader_factory=uploader_factory,
    )
    renderer.initialize()
    renderer.delete()
    renderer.delete()

    assert renderer.deleted
    assert uploader.delete_count == 1
    assert fake_gl.deleted_textures == [20]
    assert fake_gl.deleted_programs == [30]
    assert fake_gl.deleted_vaos == [40]
    assert fake_glfw.callback is None
