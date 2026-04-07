#define GL_GLEXT_PROTOTYPES
#define XR_USE_PLATFORM_XLIB
#define XR_USE_GRAPHICS_API_OPENGL
#define GLFW_EXPOSE_NATIVE_X11
#define GLFW_EXPOSE_NATIVE_GLX

#include <X11/Xlib.h>
#include <GL/gl.h>
#include <GL/glext.h>
#include <GL/glx.h>
#include <GLFW/glfw3.h>
#include <GLFW/glfw3native.h>
#include <openxr/openxr.h>
#include <openxr/openxr_platform.h>

#include <algorithm>
#include <chrono>
#include <csignal>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fcntl.h>
#include <iomanip>
#include <iostream>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <thread>
#include <unistd.h>
#include <vector>

namespace {

constexpr int kPanelWindowWidth = 64;
constexpr int kPanelWindowHeight = 64;
constexpr uint32_t kExpectedHeaderVersion = 1;
constexpr float kPanelDistanceMeters = 1.1f;
constexpr float kPanelWidthMeters = 1.2f;
constexpr float kPanelYOffsetMeters = 0.0f;
constexpr float kNearZ = 0.02f;
constexpr float kFarZ = 100.0f;
constexpr float kSelectPressedThreshold = 0.75f;
volatile std::sig_atomic_t g_stop_requested = 0;

struct SharedFrameHeader {
    char magic[8];
    uint32_t version;
    uint32_t width;
    uint32_t height;
    uint32_t channels;
    uint32_t frame_bytes;
    uint32_t slot_count;
    uint64_t latest_frame_id;
    uint64_t latest_slot;
    uint8_t padding[16];
};

static_assert(sizeof(SharedFrameHeader) == 64, "SharedFrameHeader size mismatch");

struct SharedFrameFile {
    int fd = -1;
    void* mapped = MAP_FAILED;
    size_t mapped_size = 0;
    const SharedFrameHeader* header = nullptr;
    const uint8_t* payload = nullptr;
};

struct SwapchainView {
    XrSwapchain handle = XR_NULL_HANDLE;
    uint32_t width = 0;
    uint32_t height = 0;
    std::vector<XrSwapchainImageOpenGLKHR> images;
};

struct Mat4 {
    float m[16];
};

struct ControllerPoseSample {
    const char* source = "none";
    bool action_active = false;
    bool position_valid = false;
    bool orientation_valid = false;
    bool position_tracked = false;
    bool orientation_tracked = false;
    XrPosef pose{};
};

struct SelectStateSample {
    const char* source = "none";
    bool available = false;
    bool pressed = false;
    float value = 0.0f;
};

template <typename T>
T MakeXrStruct(XrStructureType type) {
    T value{};
    value.type = type;
    return value;
}

std::string XrResultString(XrInstance instance, XrResult result) {
    if (instance != XR_NULL_HANDLE) {
        char buffer[XR_MAX_RESULT_STRING_SIZE];
        if (XR_SUCCEEDED(xrResultToString(instance, result, buffer))) {
            return buffer;
        }
    }
    return std::to_string(result);
}

bool CheckXr(XrInstance instance, XrResult result, const char* what) {
    if (XR_SUCCEEDED(result)) {
        return true;
    }
    std::cerr << what << " failed: " << XrResultString(instance, result) << "\n";
    return false;
}

bool StringToPath(XrInstance instance, const char* path_string, XrPath* path) {
    return CheckXr(instance, xrStringToPath(instance, path_string, path), path_string);
}

bool HasExtension(const std::vector<XrExtensionProperties>& extensions, const char* name) {
    for (const auto& extension : extensions) {
        if (std::strcmp(extension.extensionName, name) == 0) {
            return true;
        }
    }
    return false;
}

void HandleSignal(int) {
    g_stop_requested = 1;
}

bool ParseArgs(int argc, char** argv, std::string* frame_path) {
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--frame-path" && i + 1 < argc) {
            *frame_path = argv[++i];
            continue;
        }
        std::cerr << "Usage: openxr_frame_panel --frame-path /tmp/boba_quest_frame.bin\n";
        return false;
    }

    if (frame_path->empty()) {
        std::cerr << "Missing required --frame-path argument.\n";
        return false;
    }
    return true;
}

bool SuggestBindingsForProfile(
    XrInstance instance,
    const char* profile_string,
    const std::vector<XrActionSuggestedBinding>& bindings) {
    XrPath profile_path = XR_NULL_PATH;
    if (!StringToPath(instance, profile_string, &profile_path)) {
        return false;
    }

    XrInteractionProfileSuggestedBinding suggested =
        MakeXrStruct<XrInteractionProfileSuggestedBinding>(
            XR_TYPE_INTERACTION_PROFILE_SUGGESTED_BINDING);
    suggested.interactionProfile = profile_path;
    suggested.suggestedBindings = bindings.data();
    suggested.countSuggestedBindings = static_cast<uint32_t>(bindings.size());

    const XrResult result = xrSuggestInteractionProfileBindings(instance, &suggested);
    if (result == XR_ERROR_PATH_UNSUPPORTED || result == XR_ERROR_PATH_INVALID) {
        return true;
    }
    return CheckXr(instance, result, "xrSuggestInteractionProfileBindings");
}

bool AppendPoseBindings(
    XrInstance instance,
    XrAction grip_pose_action,
    XrAction aim_pose_action,
    std::vector<XrActionSuggestedBinding>* bindings) {
    struct BindingSpec {
        XrAction action;
        const char* path;
    };

    const BindingSpec specs[] = {
        {grip_pose_action, "/user/hand/left/input/grip/pose"},
        {grip_pose_action, "/user/hand/right/input/grip/pose"},
        {aim_pose_action, "/user/hand/left/input/aim/pose"},
        {aim_pose_action, "/user/hand/right/input/aim/pose"},
    };

    bindings->reserve(bindings->size() + 4);
    for (const auto& spec : specs) {
        XrPath path = XR_NULL_PATH;
        if (!StringToPath(instance, spec.path, &path)) {
            return false;
        }
        bindings->push_back({spec.action, path});
    }
    return true;
}

bool AppendSelectBindings(
    XrInstance instance,
    XrAction select_click_action,
    XrAction select_value_action,
    const char* profile_string,
    std::vector<XrActionSuggestedBinding>* bindings) {
    struct BindingSpec {
        XrAction action;
        const char* path;
    };

    std::vector<BindingSpec> specs;
    const std::string profile(profile_string);
    if (profile == "/interaction_profiles/khr/simple_controller") {
        specs = {
            {select_click_action, "/user/hand/left/input/select/click"},
            {select_click_action, "/user/hand/right/input/select/click"},
        };
    } else {
        specs = {
            {select_click_action, "/user/hand/left/input/trigger/value"},
            {select_click_action, "/user/hand/right/input/trigger/value"},
            {select_value_action, "/user/hand/left/input/trigger/value"},
            {select_value_action, "/user/hand/right/input/trigger/value"},
        };
    }

    bindings->reserve(bindings->size() + specs.size());
    for (const auto& spec : specs) {
        XrPath path = XR_NULL_PATH;
        if (!StringToPath(instance, spec.path, &path)) {
            return false;
        }
        bindings->push_back({spec.action, path});
    }
    return true;
}

bool AppendAnchorCycleBindings(
    XrInstance instance,
    XrAction anchor_cycle_click_action,
    const char* profile_string,
    std::vector<XrActionSuggestedBinding>* bindings) {
    if (std::strcmp(profile_string, "/interaction_profiles/oculus/touch_controller") != 0) {
        return true;
    }

    const char* paths[] = {
        "/user/hand/left/input/x/click",
        "/user/hand/right/input/a/click",
    };
    bindings->reserve(bindings->size() + 2);
    for (const char* path_string : paths) {
        XrPath path = XR_NULL_PATH;
        if (!StringToPath(instance, path_string, &path)) {
            return false;
        }
        bindings->push_back({anchor_cycle_click_action, path});
    }
    return true;
}

bool AppendSnapAssistBindings(
    XrInstance instance,
    XrAction snap_assist_click_action,
    const char* profile_string,
    std::vector<XrActionSuggestedBinding>* bindings) {
    if (std::strcmp(profile_string, "/interaction_profiles/oculus/touch_controller") != 0) {
        return true;
    }

    const char* paths[] = {
        "/user/hand/left/input/y/click",
        "/user/hand/right/input/b/click",
    };
    bindings->reserve(bindings->size() + 2);
    for (const char* path_string : paths) {
        XrPath path = XR_NULL_PATH;
        if (!StringToPath(instance, path_string, &path)) {
            return false;
        }
        bindings->push_back({snap_assist_click_action, path});
    }
    return true;
}

bool QueryControllerPose(
    XrInstance instance,
    XrSession session,
    XrAction pose_action,
    XrPath subaction_path,
    XrSpace action_space,
    XrSpace base_space,
    XrTime sample_time,
    ControllerPoseSample* sample) {
    XrActionStateGetInfo get_info =
        MakeXrStruct<XrActionStateGetInfo>(XR_TYPE_ACTION_STATE_GET_INFO);
    get_info.action = pose_action;
    get_info.subactionPath = subaction_path;

    XrActionStatePose state = MakeXrStruct<XrActionStatePose>(XR_TYPE_ACTION_STATE_POSE);
    if (!CheckXr(instance, xrGetActionStatePose(session, &get_info, &state),
                 "xrGetActionStatePose")) {
        return false;
    }

    sample->action_active = state.isActive == XR_TRUE;

    XrSpaceLocation location = MakeXrStruct<XrSpaceLocation>(XR_TYPE_SPACE_LOCATION);
    if (!CheckXr(instance, xrLocateSpace(action_space, base_space, sample_time, &location),
                 "xrLocateSpace(controller)")) {
        return false;
    }

    sample->position_valid = (location.locationFlags & XR_SPACE_LOCATION_POSITION_VALID_BIT) != 0;
    sample->orientation_valid =
        (location.locationFlags & XR_SPACE_LOCATION_ORIENTATION_VALID_BIT) != 0;
    sample->position_tracked =
        (location.locationFlags & XR_SPACE_LOCATION_POSITION_TRACKED_BIT) != 0;
    sample->orientation_tracked =
        (location.locationFlags & XR_SPACE_LOCATION_ORIENTATION_TRACKED_BIT) != 0;
    sample->pose = location.pose;
    return true;
}

void SelectPreferredControllerPose(
    const ControllerPoseSample& grip,
    const ControllerPoseSample& aim,
    ControllerPoseSample* selected) {
    *selected = grip;
    selected->source = "grip";

    const bool aim_preferred =
        (aim.action_active && (aim.position_valid || aim.orientation_valid)) ||
        (!grip.action_active && aim.action_active) ||
        ((!grip.position_valid && !grip.orientation_valid) &&
         (aim.position_valid || aim.orientation_valid));
    if (aim_preferred) {
        *selected = aim;
        selected->source = "aim";
    }
}

bool QueryBooleanActionState(
    XrInstance instance,
    XrSession session,
    XrAction click_action,
    XrPath subaction_path,
    SelectStateSample* sample) {
    XrActionStateGetInfo get_info =
        MakeXrStruct<XrActionStateGetInfo>(XR_TYPE_ACTION_STATE_GET_INFO);
    get_info.action = click_action;
    get_info.subactionPath = subaction_path;

    XrActionStateBoolean state =
        MakeXrStruct<XrActionStateBoolean>(XR_TYPE_ACTION_STATE_BOOLEAN);
    if (!CheckXr(instance, xrGetActionStateBoolean(session, &get_info, &state),
                 "xrGetActionStateBoolean")) {
        return false;
    }

    if (state.isActive == XR_TRUE) {
        sample->available = true;
        sample->pressed = state.currentState == XR_TRUE;
        sample->value = state.currentState == XR_TRUE ? 1.0f : 0.0f;
        sample->source = "click";
    }
    return true;
}

bool QuerySelectValueState(
    XrInstance instance,
    XrSession session,
    XrAction value_action,
    XrPath subaction_path,
    SelectStateSample* sample) {
    XrActionStateGetInfo get_info =
        MakeXrStruct<XrActionStateGetInfo>(XR_TYPE_ACTION_STATE_GET_INFO);
    get_info.action = value_action;
    get_info.subactionPath = subaction_path;

    XrActionStateFloat state = MakeXrStruct<XrActionStateFloat>(XR_TYPE_ACTION_STATE_FLOAT);
    if (!CheckXr(instance, xrGetActionStateFloat(session, &get_info, &state),
                 "xrGetActionStateFloat")) {
        return false;
    }

    if (state.isActive == XR_TRUE) {
        sample->available = true;
        sample->pressed = sample->pressed || state.currentState >= kSelectPressedThreshold;
        if (std::strcmp(sample->source, "none") == 0 || state.currentState > sample->value) {
            sample->value = state.currentState;
            sample->source = "value";
        }
    }
    return true;
}

void PrintControllerJson(const char* prefix, const ControllerPoseSample& pose,
                         const SelectStateSample& select,
                         const SelectStateSample& anchor_cycle,
                         const SelectStateSample& snap_assist) {
    std::cout << "\"" << prefix << "\":{";
    std::cout << "\"source\":\"" << pose.source << "\",";
    std::cout << "\"active\":" << (pose.action_active ? 1 : 0) << ",";
    std::cout << "\"position_valid\":" << (pose.position_valid ? 1 : 0) << ",";
    std::cout << "\"orientation_valid\":" << (pose.orientation_valid ? 1 : 0) << ",";
    std::cout << "\"position_tracked\":" << (pose.position_tracked ? 1 : 0) << ",";
    std::cout << "\"orientation_tracked\":" << (pose.orientation_tracked ? 1 : 0) << ",";
    std::cout << "\"position\":["
              << pose.pose.position.x << "," << pose.pose.position.y << ","
              << pose.pose.position.z << "],";
    std::cout << "\"orientation\":["
              << pose.pose.orientation.x << "," << pose.pose.orientation.y << ","
              << pose.pose.orientation.z << "," << pose.pose.orientation.w << "],";
    std::cout << "\"select_available\":" << (select.available ? 1 : 0) << ",";
    std::cout << "\"select_pressed\":" << (select.pressed ? 1 : 0) << ",";
    std::cout << "\"select_value\":" << select.value << ",";
    std::cout << "\"select_source\":\"" << select.source << "\",";
    std::cout << "\"anchor_cycle_available\":" << (anchor_cycle.available ? 1 : 0) << ",";
    std::cout << "\"anchor_cycle_pressed\":" << (anchor_cycle.pressed ? 1 : 0) << ",";
    std::cout << "\"anchor_cycle_source\":\"" << anchor_cycle.source << "\",";
    std::cout << "\"snap_assist_available\":" << (snap_assist.available ? 1 : 0) << ",";
    std::cout << "\"snap_assist_pressed\":" << (snap_assist.pressed ? 1 : 0) << ",";
    std::cout << "\"snap_assist_source\":\"" << snap_assist.source << "\"";
    std::cout << "}";
}

bool OpenSharedFrameFile(const std::string& frame_path, SharedFrameFile* file) {
    file->fd = open(frame_path.c_str(), O_RDONLY);
    if (file->fd < 0) {
        perror("open(frame_path)");
        return false;
    }

    struct stat st {};
    if (fstat(file->fd, &st) != 0) {
        perror("fstat(frame_path)");
        close(file->fd);
        file->fd = -1;
        return false;
    }

    file->mapped_size = static_cast<size_t>(st.st_size);
    file->mapped = mmap(nullptr, file->mapped_size, PROT_READ, MAP_SHARED, file->fd, 0);
    if (file->mapped == MAP_FAILED) {
        perror("mmap(frame_path)");
        close(file->fd);
        file->fd = -1;
        return false;
    }

    file->header = static_cast<const SharedFrameHeader*>(file->mapped);
    if (std::memcmp(file->header->magic, "BOBAQST1", 8) != 0) {
        std::cerr << "Shared frame header magic mismatch.\n";
        munmap(file->mapped, file->mapped_size);
        close(file->fd);
        file->mapped = MAP_FAILED;
        file->fd = -1;
        return false;
    }
    if (file->header->version != kExpectedHeaderVersion) {
        std::cerr << "Shared frame header version mismatch: " << file->header->version << "\n";
        munmap(file->mapped, file->mapped_size);
        close(file->fd);
        file->mapped = MAP_FAILED;
        file->fd = -1;
        return false;
    }

    const size_t expected_size =
        sizeof(SharedFrameHeader) +
        static_cast<size_t>(file->header->slot_count) * file->header->frame_bytes;
    if (file->mapped_size < expected_size) {
        std::cerr << "Shared frame file is smaller than expected.\n";
        munmap(file->mapped, file->mapped_size);
        close(file->fd);
        file->mapped = MAP_FAILED;
        file->fd = -1;
        return false;
    }

    file->payload = static_cast<const uint8_t*>(file->mapped) + sizeof(SharedFrameHeader);
    std::cerr << "Opened shared frame file " << frame_path << " ("
              << file->header->width << "x" << file->header->height
              << " channels=" << file->header->channels
              << " slots=" << file->header->slot_count << ")\n";
    return true;
}

void CloseSharedFrameFile(SharedFrameFile* file) {
    if (file->mapped != MAP_FAILED) {
        munmap(file->mapped, file->mapped_size);
        file->mapped = MAP_FAILED;
    }
    if (file->fd >= 0) {
        close(file->fd);
        file->fd = -1;
    }
    file->mapped_size = 0;
    file->header = nullptr;
    file->payload = nullptr;
}

bool UpdateDisplayFrameIfNeeded(const SharedFrameFile& file, uint64_t* latest_frame_id,
                                std::vector<uint8_t>* display_rgba) {
    const SharedFrameHeader header = *file.header;
    if (header.latest_frame_id == *latest_frame_id) {
        return false;
    }
    if (header.latest_slot >= header.slot_count) {
        std::cerr << "Invalid latest_slot in shared frame header: " << header.latest_slot << "\n";
        return false;
    }

    const size_t slot_offset = static_cast<size_t>(header.latest_slot) * header.frame_bytes;
    const uint8_t* source = file.payload + slot_offset;
    display_rgba->assign(source, source + header.frame_bytes);
    *latest_frame_id = header.latest_frame_id;
    return true;
}

bool ResolveGlxBinding(GLFWwindow* window, Display** xdisplay, uint32_t* visual_id,
                       GLXFBConfig* fb_config, GLXDrawable* drawable, GLXContext* context) {
    *xdisplay = glfwGetX11Display();
    *drawable = glfwGetGLXWindow(window);
    *context = glfwGetGLXContext(window);
    if (*xdisplay == nullptr || *drawable == 0 || *context == nullptr) {
        std::cerr << "Failed to resolve GLFW native GLX handles.\n";
        return false;
    }

    unsigned int fbconfig_id = 0;
    glXQueryDrawable(*xdisplay, *drawable, GLX_FBCONFIG_ID, &fbconfig_id);
    if (fbconfig_id == 0) {
        std::cerr << "GLX_FBCONFIG_ID query returned 0.\n";
        return false;
    }

    const int attribs[] = {GLX_FBCONFIG_ID, static_cast<int>(fbconfig_id), None};
    int config_count = 0;
    GLXFBConfig* configs =
        glXChooseFBConfig(*xdisplay, DefaultScreen(*xdisplay), attribs, &config_count);
    if (configs == nullptr || config_count < 1) {
        std::cerr << "glXChooseFBConfig failed for GLX_FBCONFIG_ID=" << fbconfig_id << "\n";
        return false;
    }

    *fb_config = configs[0];
    XVisualInfo* visual_info = glXGetVisualFromFBConfig(*xdisplay, *fb_config);
    if (visual_info == nullptr) {
        XFree(configs);
        std::cerr << "glXGetVisualFromFBConfig failed.\n";
        return false;
    }
    *visual_id = static_cast<uint32_t>(visual_info->visualid);
    XFree(visual_info);
    XFree(configs);
    return true;
}

bool PumpEvents(XrInstance instance, XrSession session,
                XrViewConfigurationType view_configuration_type, XrSessionState* session_state,
                bool* session_running, bool* exit_requested) {
    XrEventDataBuffer event = MakeXrStruct<XrEventDataBuffer>(XR_TYPE_EVENT_DATA_BUFFER);

    for (;;) {
        const XrResult poll_result = xrPollEvent(instance, &event);
        if (poll_result == XR_EVENT_UNAVAILABLE) {
            return true;
        }
        if (!CheckXr(instance, poll_result, "xrPollEvent")) {
            return false;
        }

        if (event.type == XR_TYPE_EVENT_DATA_SESSION_STATE_CHANGED) {
            const auto* changed =
                reinterpret_cast<const XrEventDataSessionStateChanged*>(&event);
            *session_state = changed->state;
            std::cerr << "Session state -> " << changed->state << "\n";

            if (changed->state == XR_SESSION_STATE_READY && !*session_running) {
                XrSessionBeginInfo begin_info =
                    MakeXrStruct<XrSessionBeginInfo>(XR_TYPE_SESSION_BEGIN_INFO);
                begin_info.primaryViewConfigurationType = view_configuration_type;
                if (!CheckXr(instance, xrBeginSession(session, &begin_info), "xrBeginSession")) {
                    return false;
                }
                *session_running = true;
            } else if (changed->state == XR_SESSION_STATE_STOPPING && *session_running) {
                if (!CheckXr(instance, xrEndSession(session), "xrEndSession")) {
                    return false;
                }
                *session_running = false;
            } else if (changed->state == XR_SESSION_STATE_EXITING ||
                       changed->state == XR_SESSION_STATE_LOSS_PENDING) {
                *exit_requested = true;
            }
        }

        event = MakeXrStruct<XrEventDataBuffer>(XR_TYPE_EVENT_DATA_BUFFER);
    }
}

XrEnvironmentBlendMode ChooseBlendMode(XrInstance instance, XrSystemId system_id,
                                       XrViewConfigurationType view_configuration_type) {
    uint32_t mode_count = 0;
    if (!CheckXr(instance,
                 xrEnumerateEnvironmentBlendModes(instance, system_id, view_configuration_type, 0,
                                                  &mode_count, nullptr),
                 "xrEnumerateEnvironmentBlendModes(count)")) {
        return XR_ENVIRONMENT_BLEND_MODE_OPAQUE;
    }

    std::vector<XrEnvironmentBlendMode> modes(mode_count);
    if (!CheckXr(instance,
                 xrEnumerateEnvironmentBlendModes(instance, system_id, view_configuration_type,
                                                  mode_count, &mode_count, modes.data()),
                 "xrEnumerateEnvironmentBlendModes(list)")) {
        return XR_ENVIRONMENT_BLEND_MODE_OPAQUE;
    }

    for (const auto mode : modes) {
        if (mode == XR_ENVIRONMENT_BLEND_MODE_OPAQUE) {
            return mode;
        }
    }
    return modes.empty() ? XR_ENVIRONMENT_BLEND_MODE_OPAQUE : modes.front();
}

int64_t ChooseSwapchainFormat(XrInstance instance, XrSession session) {
    uint32_t format_count = 0;
    if (!CheckXr(instance, xrEnumerateSwapchainFormats(session, 0, &format_count, nullptr),
                 "xrEnumerateSwapchainFormats(count)")) {
        return 0;
    }

    std::vector<int64_t> formats(format_count, 0);
    if (!CheckXr(instance,
                 xrEnumerateSwapchainFormats(session, format_count, &format_count, formats.data()),
                 "xrEnumerateSwapchainFormats(list)")) {
        return 0;
    }

    const int64_t preferred[] = {
        static_cast<int64_t>(GL_SRGB8_ALPHA8),
        static_cast<int64_t>(GL_RGBA8),
    };
    for (const int64_t candidate : preferred) {
        if (std::find(formats.begin(), formats.end(), candidate) != formats.end()) {
            return candidate;
        }
    }
    return formats.empty() ? 0 : formats.front();
}

GLuint CompileShader(GLenum type, const char* source) {
    const GLuint shader = glCreateShader(type);
    glShaderSource(shader, 1, &source, nullptr);
    glCompileShader(shader);

    GLint status = GL_FALSE;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &status);
    if (status != GL_TRUE) {
        GLint log_length = 0;
        glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &log_length);
        std::vector<GLchar> log(std::max(1, log_length), '\0');
        glGetShaderInfoLog(shader, log_length, nullptr, log.data());
        std::cerr << "Shader compile failed: " << log.data() << "\n";
        glDeleteShader(shader);
        return 0;
    }
    return shader;
}

Mat4 IdentityMatrix() {
    Mat4 matrix{};
    matrix.m[0] = 1.0f;
    matrix.m[5] = 1.0f;
    matrix.m[10] = 1.0f;
    matrix.m[15] = 1.0f;
    return matrix;
}

Mat4 Multiply(const Mat4& a, const Mat4& b) {
    Mat4 out{};
    for (int col = 0; col < 4; ++col) {
        for (int row = 0; row < 4; ++row) {
            float sum = 0.0f;
            for (int k = 0; k < 4; ++k) {
                sum += a.m[k * 4 + row] * b.m[col * 4 + k];
            }
            out.m[col * 4 + row] = sum;
        }
    }
    return out;
}

Mat4 TranslationMatrix(float x, float y, float z) {
    Mat4 matrix = IdentityMatrix();
    matrix.m[12] = x;
    matrix.m[13] = y;
    matrix.m[14] = z;
    return matrix;
}

Mat4 ScaleMatrix(float x, float y, float z) {
    Mat4 matrix{};
    matrix.m[0] = x;
    matrix.m[5] = y;
    matrix.m[10] = z;
    matrix.m[15] = 1.0f;
    return matrix;
}

Mat4 PoseMatrix(const XrPosef& pose) {
    const float x = pose.orientation.x;
    const float y = pose.orientation.y;
    const float z = pose.orientation.z;
    const float w = pose.orientation.w;

    const float xx = x * x;
    const float yy = y * y;
    const float zz = z * z;
    const float xy = x * y;
    const float xz = x * z;
    const float yz = y * z;
    const float wx = w * x;
    const float wy = w * y;
    const float wz = w * z;

    Mat4 matrix = IdentityMatrix();
    matrix.m[0] = 1.0f - 2.0f * (yy + zz);
    matrix.m[1] = 2.0f * (xy + wz);
    matrix.m[2] = 2.0f * (xz - wy);

    matrix.m[4] = 2.0f * (xy - wz);
    matrix.m[5] = 1.0f - 2.0f * (xx + zz);
    matrix.m[6] = 2.0f * (yz + wx);

    matrix.m[8] = 2.0f * (xz + wy);
    matrix.m[9] = 2.0f * (yz - wx);
    matrix.m[10] = 1.0f - 2.0f * (xx + yy);

    matrix.m[12] = pose.position.x;
    matrix.m[13] = pose.position.y;
    matrix.m[14] = pose.position.z;
    return matrix;
}

Mat4 InverseRigidTransform(const Mat4& matrix) {
    Mat4 inverse = IdentityMatrix();
    inverse.m[0] = matrix.m[0];
    inverse.m[1] = matrix.m[4];
    inverse.m[2] = matrix.m[8];
    inverse.m[4] = matrix.m[1];
    inverse.m[5] = matrix.m[5];
    inverse.m[6] = matrix.m[9];
    inverse.m[8] = matrix.m[2];
    inverse.m[9] = matrix.m[6];
    inverse.m[10] = matrix.m[10];

    const float tx = matrix.m[12];
    const float ty = matrix.m[13];
    const float tz = matrix.m[14];
    inverse.m[12] = -(inverse.m[0] * tx + inverse.m[4] * ty + inverse.m[8] * tz);
    inverse.m[13] = -(inverse.m[1] * tx + inverse.m[5] * ty + inverse.m[9] * tz);
    inverse.m[14] = -(inverse.m[2] * tx + inverse.m[6] * ty + inverse.m[10] * tz);
    return inverse;
}

Mat4 ProjectionMatrix(const XrFovf& fov, float near_z, float far_z) {
    const float tan_left = std::tan(fov.angleLeft);
    const float tan_right = std::tan(fov.angleRight);
    const float tan_down = std::tan(fov.angleDown);
    const float tan_up = std::tan(fov.angleUp);
    const float tan_width = tan_right - tan_left;
    const float tan_height = tan_up - tan_down;

    Mat4 matrix{};
    matrix.m[0] = 2.0f / tan_width;
    matrix.m[5] = 2.0f / tan_height;
    matrix.m[8] = (tan_right + tan_left) / tan_width;
    matrix.m[9] = (tan_up + tan_down) / tan_height;
    matrix.m[10] = -(far_z + near_z) / (far_z - near_z);
    matrix.m[11] = -1.0f;
    matrix.m[14] = -(2.0f * far_z * near_z) / (far_z - near_z);
    return matrix;
}

GLuint CreatePanelProgram() {
    const char* vertex_source = R"GLSL(
        #version 330 core
        out vec2 uv;
        uniform mat4 uMvp;
        const vec3 positions[6] = vec3[6](
            vec3(-0.5, -0.5, 0.0),
            vec3( 0.5, -0.5, 0.0),
            vec3( 0.5,  0.5, 0.0),
            vec3(-0.5, -0.5, 0.0),
            vec3( 0.5,  0.5, 0.0),
            vec3(-0.5,  0.5, 0.0)
        );
        const vec2 texcoords[6] = vec2[6](
            vec2(0.0, 0.0),
            vec2(1.0, 0.0),
            vec2(1.0, 1.0),
            vec2(0.0, 0.0),
            vec2(1.0, 1.0),
            vec2(0.0, 1.0)
        );
        void main() {
            gl_Position = uMvp * vec4(positions[gl_VertexID], 1.0);
            uv = texcoords[gl_VertexID];
        }
    )GLSL";

    const char* fragment_source = R"GLSL(
        #version 330 core
        in vec2 uv;
        out vec4 frag;
        uniform sampler2D uSource;
        void main() {
            frag = texture(uSource, vec2(uv.x, 1.0 - uv.y));
        }
    )GLSL";

    const GLuint vertex_shader = CompileShader(GL_VERTEX_SHADER, vertex_source);
    const GLuint fragment_shader = CompileShader(GL_FRAGMENT_SHADER, fragment_source);
    if (vertex_shader == 0 || fragment_shader == 0) {
        glDeleteShader(vertex_shader);
        glDeleteShader(fragment_shader);
        return 0;
    }

    const GLuint program = glCreateProgram();
    glAttachShader(program, vertex_shader);
    glAttachShader(program, fragment_shader);
    glLinkProgram(program);
    glDeleteShader(vertex_shader);
    glDeleteShader(fragment_shader);

    GLint status = GL_FALSE;
    glGetProgramiv(program, GL_LINK_STATUS, &status);
    if (status != GL_TRUE) {
        GLint log_length = 0;
        glGetProgramiv(program, GL_INFO_LOG_LENGTH, &log_length);
        std::vector<GLchar> log(std::max(1, log_length), '\0');
        glGetProgramInfoLog(program, log_length, nullptr, log.data());
        std::cerr << "Program link failed: " << log.data() << "\n";
        glDeleteProgram(program);
        return 0;
    }
    return program;
}

bool CreateViewSwapchains(XrInstance instance, XrSession session, int64_t swapchain_format,
                          const std::vector<XrViewConfigurationView>& config_views,
                          std::vector<SwapchainView>* swapchain_views) {
    swapchain_views->clear();
    swapchain_views->resize(config_views.size());

    for (size_t view_index = 0; view_index < config_views.size(); ++view_index) {
        const auto& config = config_views[view_index];
        auto& swapchain_view = (*swapchain_views)[view_index];
        swapchain_view.width = config.recommendedImageRectWidth;
        swapchain_view.height = config.recommendedImageRectHeight;

        XrSwapchainCreateInfo swapchain_info =
            MakeXrStruct<XrSwapchainCreateInfo>(XR_TYPE_SWAPCHAIN_CREATE_INFO);
        swapchain_info.createFlags = 0;
        swapchain_info.usageFlags =
            XR_SWAPCHAIN_USAGE_COLOR_ATTACHMENT_BIT | XR_SWAPCHAIN_USAGE_SAMPLED_BIT;
        swapchain_info.format = swapchain_format;
        swapchain_info.sampleCount = 1;
        swapchain_info.width = swapchain_view.width;
        swapchain_info.height = swapchain_view.height;
        swapchain_info.faceCount = 1;
        swapchain_info.arraySize = 1;
        swapchain_info.mipCount = 1;

        if (!CheckXr(instance, xrCreateSwapchain(session, &swapchain_info, &swapchain_view.handle),
                     "xrCreateSwapchain(view)")) {
            return false;
        }

        uint32_t image_count = 0;
        if (!CheckXr(instance,
                     xrEnumerateSwapchainImages(swapchain_view.handle, 0, &image_count, nullptr),
                     "xrEnumerateSwapchainImages(count)")) {
            return false;
        }

        swapchain_view.images.resize(image_count);
        for (auto& image : swapchain_view.images) {
            image = MakeXrStruct<XrSwapchainImageOpenGLKHR>(
                XR_TYPE_SWAPCHAIN_IMAGE_OPENGL_KHR);
        }
        if (!CheckXr(instance,
                     xrEnumerateSwapchainImages(
                         swapchain_view.handle, image_count, &image_count,
                         reinterpret_cast<XrSwapchainImageBaseHeader*>(
                             swapchain_view.images.data())),
                     "xrEnumerateSwapchainImages(list)")) {
            return false;
        }

        for (const auto& image : swapchain_view.images) {
            glBindTexture(GL_TEXTURE_2D, image.image);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        }
        glBindTexture(GL_TEXTURE_2D, 0);
    }

    return true;
}

void DestroyViewSwapchains(std::vector<SwapchainView>* swapchain_views) {
    for (auto& swapchain_view : *swapchain_views) {
        if (swapchain_view.handle != XR_NULL_HANDLE) {
            xrDestroySwapchain(swapchain_view.handle);
            swapchain_view.handle = XR_NULL_HANDLE;
        }
        swapchain_view.images.clear();
    }
}

}  // namespace

int main(int argc, char** argv) {
    std::signal(SIGINT, HandleSignal);
    std::signal(SIGTERM, HandleSignal);
    std::cout << std::fixed << std::setprecision(6) << std::unitbuf;
    std::cerr << std::fixed << std::setprecision(6) << std::unitbuf;

    std::string frame_path;
    if (!ParseArgs(argc, argv, &frame_path)) {
        return 2;
    }

    SharedFrameFile shared_frame;
    if (!OpenSharedFrameFile(frame_path, &shared_frame)) {
        return 3;
    }
    if (shared_frame.header->channels != 4) {
        std::cerr << "Expected RGBA frame data, got channels="
                  << shared_frame.header->channels << "\n";
        CloseSharedFrameFile(&shared_frame);
        return 4;
    }

    if (!glfwInit()) {
        std::cerr << "glfwInit failed.\n";
        CloseSharedFrameFile(&shared_frame);
        return 5;
    }

    glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
    glfwWindowHint(GLFW_CLIENT_API, GLFW_OPENGL_API);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    GLFWwindow* window =
        glfwCreateWindow(kPanelWindowWidth, kPanelWindowHeight, "Boba Quest Frame Panel", nullptr,
                         nullptr);
    if (window == nullptr) {
        std::cerr << "glfwCreateWindow failed.\n";
        glfwTerminate();
        CloseSharedFrameFile(&shared_frame);
        return 6;
    }
    glfwMakeContextCurrent(window);
    glfwSwapInterval(0);

    const GLubyte* gl_version = glGetString(GL_VERSION);
    std::cerr << "OpenGL version: "
              << (gl_version ? reinterpret_cast<const char*>(gl_version) : "(null)") << "\n";

    Display* xdisplay = nullptr;
    GLXFBConfig fb_config = nullptr;
    GLXDrawable drawable = 0;
    GLXContext glx_context = nullptr;
    uint32_t visual_id = 0;
    if (!ResolveGlxBinding(window, &xdisplay, &visual_id, &fb_config, &drawable, &glx_context)) {
        glfwDestroyWindow(window);
        glfwTerminate();
        CloseSharedFrameFile(&shared_frame);
        return 7;
    }

    uint32_t extension_count = 0;
    if (!CheckXr(XR_NULL_HANDLE,
                 xrEnumerateInstanceExtensionProperties(nullptr, 0, &extension_count, nullptr),
                 "xrEnumerateInstanceExtensionProperties(count)")) {
        glfwDestroyWindow(window);
        glfwTerminate();
        CloseSharedFrameFile(&shared_frame);
        return 8;
    }
    std::vector<XrExtensionProperties> extensions(extension_count);
    for (auto& extension : extensions) {
        extension = MakeXrStruct<XrExtensionProperties>(XR_TYPE_EXTENSION_PROPERTIES);
    }
    if (!CheckXr(XR_NULL_HANDLE,
                 xrEnumerateInstanceExtensionProperties(nullptr, extension_count, &extension_count,
                                                       extensions.data()),
                 "xrEnumerateInstanceExtensionProperties(list)")) {
        glfwDestroyWindow(window);
        glfwTerminate();
        CloseSharedFrameFile(&shared_frame);
        return 8;
    }

    if (!HasExtension(extensions, XR_KHR_OPENGL_ENABLE_EXTENSION_NAME)) {
        std::cerr << "Runtime does not expose XR_KHR_opengl_enable.\n";
        glfwDestroyWindow(window);
        glfwTerminate();
        CloseSharedFrameFile(&shared_frame);
        return 9;
    }

    const std::vector<const char*> enabled_extensions = {
        XR_KHR_OPENGL_ENABLE_EXTENSION_NAME,
    };

    XrInstanceCreateInfo instance_info =
        MakeXrStruct<XrInstanceCreateInfo>(XR_TYPE_INSTANCE_CREATE_INFO);
    std::strncpy(instance_info.applicationInfo.applicationName, "openxr_frame_panel",
                 XR_MAX_APPLICATION_NAME_SIZE - 1);
    std::strncpy(instance_info.applicationInfo.engineName, "none", XR_MAX_ENGINE_NAME_SIZE - 1);
    instance_info.applicationInfo.applicationVersion = 1;
    instance_info.applicationInfo.engineVersion = 1;
    instance_info.applicationInfo.apiVersion = XR_CURRENT_API_VERSION;
    instance_info.enabledExtensionCount = static_cast<uint32_t>(enabled_extensions.size());
    instance_info.enabledExtensionNames = enabled_extensions.data();

    XrInstance instance = XR_NULL_HANDLE;
    if (!CheckXr(XR_NULL_HANDLE, xrCreateInstance(&instance_info, &instance), "xrCreateInstance")) {
        glfwDestroyWindow(window);
        glfwTerminate();
        CloseSharedFrameFile(&shared_frame);
        return 10;
    }

    PFN_xrGetOpenGLGraphicsRequirementsKHR get_graphics_requirements = nullptr;
    if (!CheckXr(instance,
                 xrGetInstanceProcAddr(
                     instance, "xrGetOpenGLGraphicsRequirementsKHR",
                     reinterpret_cast<PFN_xrVoidFunction*>(&get_graphics_requirements)),
                 "xrGetInstanceProcAddr(xrGetOpenGLGraphicsRequirementsKHR)")) {
        xrDestroyInstance(instance);
        glfwDestroyWindow(window);
        glfwTerminate();
        CloseSharedFrameFile(&shared_frame);
        return 10;
    }

    XrSystemGetInfo system_info = MakeXrStruct<XrSystemGetInfo>(XR_TYPE_SYSTEM_GET_INFO);
    system_info.formFactor = XR_FORM_FACTOR_HEAD_MOUNTED_DISPLAY;
    XrSystemId system_id = XR_NULL_SYSTEM_ID;
    if (!CheckXr(instance, xrGetSystem(instance, &system_info, &system_id), "xrGetSystem")) {
        xrDestroyInstance(instance);
        glfwDestroyWindow(window);
        glfwTerminate();
        CloseSharedFrameFile(&shared_frame);
        return 11;
    }

    XrGraphicsRequirementsOpenGLKHR graphics_requirements =
        MakeXrStruct<XrGraphicsRequirementsOpenGLKHR>(
            XR_TYPE_GRAPHICS_REQUIREMENTS_OPENGL_KHR);
    if (!CheckXr(instance, get_graphics_requirements(instance, system_id, &graphics_requirements),
                 "xrGetOpenGLGraphicsRequirementsKHR")) {
        xrDestroyInstance(instance);
        glfwDestroyWindow(window);
        glfwTerminate();
        CloseSharedFrameFile(&shared_frame);
        return 12;
    }

    const XrViewConfigurationType view_configuration_type =
        XR_VIEW_CONFIGURATION_TYPE_PRIMARY_STEREO;
    const XrEnvironmentBlendMode blend_mode =
        ChooseBlendMode(instance, system_id, view_configuration_type);

    uint32_t config_view_count = 0;
    if (!CheckXr(instance,
                 xrEnumerateViewConfigurationViews(instance, system_id, view_configuration_type, 0,
                                                   &config_view_count, nullptr),
                 "xrEnumerateViewConfigurationViews(count)")) {
        xrDestroyInstance(instance);
        glfwDestroyWindow(window);
        glfwTerminate();
        CloseSharedFrameFile(&shared_frame);
        return 13;
    }
    std::vector<XrViewConfigurationView> config_views(config_view_count);
    for (auto& view : config_views) {
        view = MakeXrStruct<XrViewConfigurationView>(XR_TYPE_VIEW_CONFIGURATION_VIEW);
    }
    if (!CheckXr(instance,
                 xrEnumerateViewConfigurationViews(instance, system_id, view_configuration_type,
                                                   config_view_count, &config_view_count,
                                                   config_views.data()),
                 "xrEnumerateViewConfigurationViews(list)")) {
        xrDestroyInstance(instance);
        glfwDestroyWindow(window);
        glfwTerminate();
        CloseSharedFrameFile(&shared_frame);
        return 13;
    }

    XrGraphicsBindingOpenGLXlibKHR graphics_binding =
        MakeXrStruct<XrGraphicsBindingOpenGLXlibKHR>(
            XR_TYPE_GRAPHICS_BINDING_OPENGL_XLIB_KHR);
    graphics_binding.xDisplay = xdisplay;
    graphics_binding.visualid = visual_id;
    graphics_binding.glxFBConfig = fb_config;
    graphics_binding.glxDrawable = drawable;
    graphics_binding.glxContext = glx_context;

    XrSessionCreateInfo session_info = MakeXrStruct<XrSessionCreateInfo>(XR_TYPE_SESSION_CREATE_INFO);
    session_info.next = &graphics_binding;
    session_info.systemId = system_id;

    XrSession session = XR_NULL_HANDLE;
    if (!CheckXr(instance, xrCreateSession(instance, &session_info, &session), "xrCreateSession")) {
        xrDestroyInstance(instance);
        glfwDestroyWindow(window);
        glfwTerminate();
        CloseSharedFrameFile(&shared_frame);
        return 14;
    }

    XrActionSetCreateInfo action_set_info =
        MakeXrStruct<XrActionSetCreateInfo>(XR_TYPE_ACTION_SET_CREATE_INFO);
    std::strncpy(action_set_info.actionSetName, "questpanelctrl",
                 XR_MAX_ACTION_SET_NAME_SIZE - 1);
    std::strncpy(action_set_info.localizedActionSetName, "Quest Panel Controller Actions",
                 XR_MAX_LOCALIZED_ACTION_SET_NAME_SIZE - 1);
    action_set_info.priority = 0;

    XrActionSet action_set = XR_NULL_HANDLE;
    if (!CheckXr(instance, xrCreateActionSet(instance, &action_set_info, &action_set),
                 "xrCreateActionSet")) {
        xrDestroySession(session);
        xrDestroyInstance(instance);
        glfwDestroyWindow(window);
        glfwTerminate();
        CloseSharedFrameFile(&shared_frame);
        return 15;
    }

    XrPath left_hand_path = XR_NULL_PATH;
    XrPath right_hand_path = XR_NULL_PATH;
    if (!StringToPath(instance, "/user/hand/left", &left_hand_path) ||
        !StringToPath(instance, "/user/hand/right", &right_hand_path)) {
        xrDestroyActionSet(action_set);
        xrDestroySession(session);
        xrDestroyInstance(instance);
        glfwDestroyWindow(window);
        glfwTerminate();
        CloseSharedFrameFile(&shared_frame);
        return 15;
    }
    const XrPath subaction_paths[] = {left_hand_path, right_hand_path};

    auto create_action = [&](const char* action_name, const char* localized_name,
                             XrActionType action_type, XrAction* action) -> bool {
        XrActionCreateInfo action_info =
            MakeXrStruct<XrActionCreateInfo>(XR_TYPE_ACTION_CREATE_INFO);
        std::strncpy(action_info.actionName, action_name, XR_MAX_ACTION_NAME_SIZE - 1);
        std::strncpy(action_info.localizedActionName, localized_name,
                     XR_MAX_LOCALIZED_ACTION_NAME_SIZE - 1);
        action_info.actionType = action_type;
        action_info.countSubactionPaths = 2;
        action_info.subactionPaths = subaction_paths;
        return CheckXr(instance, xrCreateAction(action_set, &action_info, action),
                       "xrCreateAction");
    };

    XrAction grip_pose_action = XR_NULL_HANDLE;
    XrAction aim_pose_action = XR_NULL_HANDLE;
    XrAction select_click_action = XR_NULL_HANDLE;
    XrAction select_value_action = XR_NULL_HANDLE;
    XrAction anchor_cycle_click_action = XR_NULL_HANDLE;
    XrAction snap_assist_click_action = XR_NULL_HANDLE;
    if (!create_action("grip_pose", "Grip Pose", XR_ACTION_TYPE_POSE_INPUT,
                       &grip_pose_action) ||
        !create_action("aim_pose", "Aim Pose", XR_ACTION_TYPE_POSE_INPUT, &aim_pose_action) ||
        !create_action("select_click", "Select Click", XR_ACTION_TYPE_BOOLEAN_INPUT,
                       &select_click_action) ||
        !create_action("select_value", "Select Value", XR_ACTION_TYPE_FLOAT_INPUT,
                       &select_value_action) ||
        !create_action("anchor_cycle_click", "Anchor Cycle Click",
                       XR_ACTION_TYPE_BOOLEAN_INPUT, &anchor_cycle_click_action) ||
        !create_action("snap_assist_click", "Snap Assist Click",
                       XR_ACTION_TYPE_BOOLEAN_INPUT, &snap_assist_click_action)) {
        xrDestroyActionSet(action_set);
        xrDestroySession(session);
        xrDestroyInstance(instance);
        glfwDestroyWindow(window);
        glfwTerminate();
        CloseSharedFrameFile(&shared_frame);
        return 15;
    }

    const char* profiles[] = {
        "/interaction_profiles/khr/simple_controller",
        "/interaction_profiles/oculus/touch_controller",
        "/interaction_profiles/htc/vive_controller",
        "/interaction_profiles/valve/index_controller",
        "/interaction_profiles/microsoft/motion_controller",
    };
    for (const char* profile : profiles) {
        std::vector<XrActionSuggestedBinding> bindings;
        if (!AppendPoseBindings(instance, grip_pose_action, aim_pose_action, &bindings) ||
            !AppendSelectBindings(instance, select_click_action, select_value_action, profile,
                                  &bindings) ||
            !AppendAnchorCycleBindings(instance, anchor_cycle_click_action, profile, &bindings) ||
            !AppendSnapAssistBindings(instance, snap_assist_click_action, profile, &bindings) ||
            !SuggestBindingsForProfile(instance, profile, bindings)) {
            xrDestroyActionSet(action_set);
            xrDestroySession(session);
            xrDestroyInstance(instance);
            glfwDestroyWindow(window);
            glfwTerminate();
            CloseSharedFrameFile(&shared_frame);
            return 15;
        }
    }

    XrSessionActionSetsAttachInfo attach_info =
        MakeXrStruct<XrSessionActionSetsAttachInfo>(
            XR_TYPE_SESSION_ACTION_SETS_ATTACH_INFO);
    attach_info.countActionSets = 1;
    attach_info.actionSets = &action_set;
    if (!CheckXr(instance, xrAttachSessionActionSets(session, &attach_info),
                 "xrAttachSessionActionSets")) {
        xrDestroyActionSet(action_set);
        xrDestroySession(session);
        xrDestroyInstance(instance);
        glfwDestroyWindow(window);
        glfwTerminate();
        CloseSharedFrameFile(&shared_frame);
        return 15;
    }

    auto create_action_space = [&](XrAction action, XrPath subaction_path,
                                   XrSpace* space, const char* label) -> bool {
        XrActionSpaceCreateInfo space_info =
            MakeXrStruct<XrActionSpaceCreateInfo>(XR_TYPE_ACTION_SPACE_CREATE_INFO);
        space_info.action = action;
        space_info.subactionPath = subaction_path;
        space_info.poseInActionSpace.orientation.w = 1.0f;
        return CheckXr(instance, xrCreateActionSpace(session, &space_info, space), label);
    };

    XrSpace grip_left_space = XR_NULL_HANDLE;
    XrSpace grip_right_space = XR_NULL_HANDLE;
    XrSpace aim_left_space = XR_NULL_HANDLE;
    XrSpace aim_right_space = XR_NULL_HANDLE;
    if (!create_action_space(grip_pose_action, left_hand_path, &grip_left_space,
                             "xrCreateActionSpace(grip_left)") ||
        !create_action_space(grip_pose_action, right_hand_path, &grip_right_space,
                             "xrCreateActionSpace(grip_right)") ||
        !create_action_space(aim_pose_action, left_hand_path, &aim_left_space,
                             "xrCreateActionSpace(aim_left)") ||
        !create_action_space(aim_pose_action, right_hand_path, &aim_right_space,
                             "xrCreateActionSpace(aim_right)")) {
        if (aim_right_space != XR_NULL_HANDLE) xrDestroySpace(aim_right_space);
        if (aim_left_space != XR_NULL_HANDLE) xrDestroySpace(aim_left_space);
        if (grip_right_space != XR_NULL_HANDLE) xrDestroySpace(grip_right_space);
        if (grip_left_space != XR_NULL_HANDLE) xrDestroySpace(grip_left_space);
        xrDestroyActionSet(action_set);
        xrDestroySession(session);
        xrDestroyInstance(instance);
        glfwDestroyWindow(window);
        glfwTerminate();
        CloseSharedFrameFile(&shared_frame);
        return 15;
    }

    const int64_t color_format = ChooseSwapchainFormat(instance, session);
    if (color_format == 0) {
        std::cerr << "Failed to choose a swapchain format.\n";
        xrDestroySession(session);
        xrDestroyInstance(instance);
        glfwDestroyWindow(window);
        glfwTerminate();
        CloseSharedFrameFile(&shared_frame);
        return 15;
    }

    std::vector<SwapchainView> swapchain_views;
    if (!CreateViewSwapchains(instance, session, color_format, config_views, &swapchain_views)) {
        DestroyViewSwapchains(&swapchain_views);
        xrDestroySession(session);
        xrDestroyInstance(instance);
        glfwDestroyWindow(window);
        glfwTerminate();
        CloseSharedFrameFile(&shared_frame);
        return 16;
    }

    GLuint source_texture = 0;
    glGenTextures(1, &source_texture);
    glBindTexture(GL_TEXTURE_2D, source_texture);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, shared_frame.header->width,
                 shared_frame.header->height, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
    glBindTexture(GL_TEXTURE_2D, 0);

    const GLuint program = CreatePanelProgram();
    if (program == 0) {
        glDeleteTextures(1, &source_texture);
        DestroyViewSwapchains(&swapchain_views);
        xrDestroySession(session);
        xrDestroyInstance(instance);
        glfwDestroyWindow(window);
        glfwTerminate();
        CloseSharedFrameFile(&shared_frame);
        return 17;
    }

    GLuint vao = 0;
    glGenVertexArrays(1, &vao);
    GLuint framebuffer = 0;
    glGenFramebuffers(1, &framebuffer);
    const GLint source_location = glGetUniformLocation(program, "uSource");
    const GLint mvp_location = glGetUniformLocation(program, "uMvp");

    XrReferenceSpaceCreateInfo local_space_info =
        MakeXrStruct<XrReferenceSpaceCreateInfo>(XR_TYPE_REFERENCE_SPACE_CREATE_INFO);
    local_space_info.referenceSpaceType = XR_REFERENCE_SPACE_TYPE_LOCAL;
    local_space_info.poseInReferenceSpace.orientation.w = 1.0f;
    XrSpace local_space = XR_NULL_HANDLE;
    if (!CheckXr(instance, xrCreateReferenceSpace(session, &local_space_info, &local_space),
                 "xrCreateReferenceSpace(LOCAL)")) {
        glDeleteFramebuffers(1, &framebuffer);
        glDeleteVertexArrays(1, &vao);
        glDeleteProgram(program);
        glDeleteTextures(1, &source_texture);
        DestroyViewSwapchains(&swapchain_views);
        xrDestroySession(session);
        xrDestroyInstance(instance);
        glfwDestroyWindow(window);
        glfwTerminate();
        CloseSharedFrameFile(&shared_frame);
        return 18;
    }
    std::vector<XrView> views(config_view_count);
    for (auto& view : views) {
        view = MakeXrStruct<XrView>(XR_TYPE_VIEW);
    }
    std::vector<XrCompositionLayerProjectionView> projection_views(config_view_count);
    for (auto& view : projection_views) {
        view = MakeXrStruct<XrCompositionLayerProjectionView>(
            XR_TYPE_COMPOSITION_LAYER_PROJECTION_VIEW);
    }

    XrSessionState session_state = XR_SESSION_STATE_UNKNOWN;
    bool session_running = false;
    bool exit_requested = false;
    uint64_t latest_frame_id = 0;
    uint64_t logged_source_frame_id = 0;
    uint64_t controller_sample_count = 0;
    std::vector<uint8_t> display_rgba(shared_frame.header->frame_bytes, 0);
    const float panel_height_meters =
        kPanelWidthMeters *
        (static_cast<float>(shared_frame.header->height) /
         static_cast<float>(shared_frame.header->width));
    Mat4 panel_model = IdentityMatrix();
    bool panel_anchor_initialized = false;
    XrActiveActionSet active_action_set{action_set, XR_NULL_PATH};
    XrActionsSyncInfo sync_info = MakeXrStruct<XrActionsSyncInfo>(XR_TYPE_ACTIONS_SYNC_INFO);
    sync_info.countActiveActionSets = 1;
    sync_info.activeActionSets = &active_action_set;

    while (!g_stop_requested && !exit_requested) {
        glfwPollEvents();
        if (glfwWindowShouldClose(window)) {
            break;
        }

        if (!PumpEvents(instance, session, view_configuration_type, &session_state, &session_running,
                        &exit_requested)) {
            break;
        }
        if (!session_running) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            continue;
        }

        XrFrameWaitInfo wait_info = MakeXrStruct<XrFrameWaitInfo>(XR_TYPE_FRAME_WAIT_INFO);
        XrFrameState frame_state = MakeXrStruct<XrFrameState>(XR_TYPE_FRAME_STATE);
        if (!CheckXr(instance, xrWaitFrame(session, &wait_info, &frame_state), "xrWaitFrame")) {
            break;
        }

        XrFrameBeginInfo begin_info = MakeXrStruct<XrFrameBeginInfo>(XR_TYPE_FRAME_BEGIN_INFO);
        if (!CheckXr(instance, xrBeginFrame(session, &begin_info), "xrBeginFrame")) {
            break;
        }

        if (!CheckXr(instance, xrSyncActions(session, &sync_info), "xrSyncActions")) {
            break;
        }

        ControllerPoseSample grip_left;
        ControllerPoseSample grip_right;
        ControllerPoseSample aim_left;
        ControllerPoseSample aim_right;
        if (!QueryControllerPose(instance, session, grip_pose_action, left_hand_path,
                                 grip_left_space, local_space, frame_state.predictedDisplayTime,
                                 &grip_left) ||
            !QueryControllerPose(instance, session, grip_pose_action, right_hand_path,
                                 grip_right_space, local_space, frame_state.predictedDisplayTime,
                                 &grip_right) ||
            !QueryControllerPose(instance, session, aim_pose_action, left_hand_path,
                                 aim_left_space, local_space, frame_state.predictedDisplayTime,
                                 &aim_left) ||
            !QueryControllerPose(instance, session, aim_pose_action, right_hand_path,
                                 aim_right_space, local_space, frame_state.predictedDisplayTime,
                                 &aim_right)) {
            break;
        }

        ControllerPoseSample selected_left;
        ControllerPoseSample selected_right;
        SelectPreferredControllerPose(grip_left, aim_left, &selected_left);
        SelectPreferredControllerPose(grip_right, aim_right, &selected_right);

        SelectStateSample select_left;
        SelectStateSample select_right;
        SelectStateSample anchor_cycle_left;
        SelectStateSample anchor_cycle_right;
        SelectStateSample snap_assist_left;
        SelectStateSample snap_assist_right;
        if (!QueryBooleanActionState(instance, session, select_click_action, left_hand_path,
                                     &select_left) ||
            !QueryBooleanActionState(instance, session, select_click_action, right_hand_path,
                                     &select_right) ||
            !QuerySelectValueState(instance, session, select_value_action, left_hand_path,
                                   &select_left) ||
            !QuerySelectValueState(instance, session, select_value_action, right_hand_path,
                                   &select_right) ||
            !QueryBooleanActionState(instance, session, anchor_cycle_click_action,
                                     left_hand_path, &anchor_cycle_left) ||
            !QueryBooleanActionState(instance, session, anchor_cycle_click_action,
                                     right_hand_path, &anchor_cycle_right) ||
            !QueryBooleanActionState(instance, session, snap_assist_click_action,
                                     left_hand_path, &snap_assist_left) ||
            !QueryBooleanActionState(instance, session, snap_assist_click_action,
                                     right_hand_path, &snap_assist_right)) {
            break;
        }

        std::cout << "{";
        std::cout << "\"sample\":" << controller_sample_count << ",";
        PrintControllerJson(
            "left", selected_left, select_left, anchor_cycle_left, snap_assist_left
        );
        std::cout << ",";
        PrintControllerJson(
            "right", selected_right, select_right, anchor_cycle_right, snap_assist_right
        );
        std::cout << "}\n";
        ++controller_sample_count;

        std::vector<const XrCompositionLayerBaseHeader*> layers;
        XrCompositionLayerProjection projection_layer =
            MakeXrStruct<XrCompositionLayerProjection>(XR_TYPE_COMPOSITION_LAYER_PROJECTION);

        if (frame_state.shouldRender == XR_TRUE) {
            if (UpdateDisplayFrameIfNeeded(shared_frame, &latest_frame_id, &display_rgba)) {
                if (latest_frame_id == 1 || latest_frame_id >= logged_source_frame_id + 120) {
                    std::cerr << "Panel received source frame " << latest_frame_id << "\n";
                    logged_source_frame_id = latest_frame_id;
                }
            }

            glBindTexture(GL_TEXTURE_2D, source_texture);
            glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
            glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, shared_frame.header->width,
                            shared_frame.header->height, GL_RGBA, GL_UNSIGNED_BYTE,
                            display_rgba.data());
            glBindTexture(GL_TEXTURE_2D, 0);

            XrViewLocateInfo locate_info =
                MakeXrStruct<XrViewLocateInfo>(XR_TYPE_VIEW_LOCATE_INFO);
            locate_info.viewConfigurationType = view_configuration_type;
            locate_info.displayTime = frame_state.predictedDisplayTime;
            locate_info.space = local_space;
            XrViewState view_state = MakeXrStruct<XrViewState>(XR_TYPE_VIEW_STATE);
            uint32_t view_count_output = 0;
            if (!CheckXr(instance,
                         xrLocateViews(session, &locate_info, &view_state,
                                       static_cast<uint32_t>(views.size()), &view_count_output,
                                       views.data()),
                         "xrLocateViews")) {
                break;
            }

            if (!panel_anchor_initialized && view_count_output > 0) {
                const Mat4 initial_head_pose = PoseMatrix(views[0].pose);
                panel_model = Multiply(
                    initial_head_pose,
                    Multiply(
                        TranslationMatrix(0.0f, kPanelYOffsetMeters, -kPanelDistanceMeters),
                        ScaleMatrix(kPanelWidthMeters, panel_height_meters, 1.0f)));
                panel_anchor_initialized = true;
                std::cerr << "Presentation path: LOCAL-space world-locked textured panel anchored "
                          << "at initial head pose, z-offset=" << -kPanelDistanceMeters
                          << " width=" << kPanelWidthMeters
                          << " height=" << panel_height_meters << "\n";
            }

            projection_layer.space = local_space;
            projection_layer.viewCount = view_count_output;
            projection_layer.views = projection_views.data();

            for (uint32_t view_index = 0; view_index < view_count_output; ++view_index) {
                auto& swapchain_view = swapchain_views[view_index];
                auto& projection_view = projection_views[view_index];
                projection_view = MakeXrStruct<XrCompositionLayerProjectionView>(
                    XR_TYPE_COMPOSITION_LAYER_PROJECTION_VIEW);
                projection_view.pose = views[view_index].pose;
                projection_view.fov = views[view_index].fov;

                XrSwapchainImageAcquireInfo acquire_info =
                    MakeXrStruct<XrSwapchainImageAcquireInfo>(
                        XR_TYPE_SWAPCHAIN_IMAGE_ACQUIRE_INFO);
                uint32_t image_index = 0;
                if (!CheckXr(instance,
                             xrAcquireSwapchainImage(swapchain_view.handle, &acquire_info,
                                                     &image_index),
                             "xrAcquireSwapchainImage")) {
                    exit_requested = true;
                    break;
                }

                XrSwapchainImageWaitInfo wait_swapchain_info =
                    MakeXrStruct<XrSwapchainImageWaitInfo>(
                        XR_TYPE_SWAPCHAIN_IMAGE_WAIT_INFO);
                wait_swapchain_info.timeout = XR_INFINITE_DURATION;
                if (!CheckXr(instance,
                             xrWaitSwapchainImage(swapchain_view.handle, &wait_swapchain_info),
                             "xrWaitSwapchainImage")) {
                    exit_requested = true;
                    break;
                }

                glBindFramebuffer(GL_FRAMEBUFFER, framebuffer);
                glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D,
                                       swapchain_view.images[image_index].image, 0);
                if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
                    std::cerr << "Framebuffer incomplete for view " << view_index << "\n";
                    exit_requested = true;
                } else {
                    const Mat4 view_matrix =
                        InverseRigidTransform(PoseMatrix(views[view_index].pose));
                    const Mat4 projection_matrix =
                        ProjectionMatrix(views[view_index].fov, kNearZ, kFarZ);
                    const Mat4 mvp_matrix =
                        Multiply(projection_matrix, Multiply(view_matrix, panel_model));
                    glViewport(0, 0, static_cast<GLsizei>(swapchain_view.width),
                               static_cast<GLsizei>(swapchain_view.height));
                    glDisable(GL_DEPTH_TEST);
                    glDisable(GL_CULL_FACE);
                    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
                    glClear(GL_COLOR_BUFFER_BIT);
                    glUseProgram(program);
                    glBindVertexArray(vao);
                    glActiveTexture(GL_TEXTURE0);
                    glBindTexture(GL_TEXTURE_2D, source_texture);
                    glUniform1i(source_location, 0);
                    glUniformMatrix4fv(mvp_location, 1, GL_FALSE, mvp_matrix.m);
                    glDrawArrays(GL_TRIANGLES, 0, 6);
                    glBindTexture(GL_TEXTURE_2D, 0);
                    glBindVertexArray(0);
                    glUseProgram(0);
                    glBindFramebuffer(GL_FRAMEBUFFER, 0);
                    glFinish();
                }

                XrSwapchainImageReleaseInfo release_info =
                    MakeXrStruct<XrSwapchainImageReleaseInfo>(
                        XR_TYPE_SWAPCHAIN_IMAGE_RELEASE_INFO);
                if (!CheckXr(instance, xrReleaseSwapchainImage(swapchain_view.handle, &release_info),
                             "xrReleaseSwapchainImage")) {
                    exit_requested = true;
                    break;
                }

                projection_view.subImage.swapchain = swapchain_view.handle;
                projection_view.subImage.imageRect.offset = {0, 0};
                projection_view.subImage.imageRect.extent = {
                    static_cast<int32_t>(swapchain_view.width),
                    static_cast<int32_t>(swapchain_view.height),
                };
                projection_view.subImage.imageArrayIndex = 0;
            }

            if (!exit_requested) {
                layers.push_back(
                    reinterpret_cast<const XrCompositionLayerBaseHeader*>(&projection_layer));
            }
        }

        XrFrameEndInfo end_info = MakeXrStruct<XrFrameEndInfo>(XR_TYPE_FRAME_END_INFO);
        end_info.displayTime = frame_state.predictedDisplayTime;
        end_info.environmentBlendMode = blend_mode;
        end_info.layerCount = static_cast<uint32_t>(layers.size());
        end_info.layers = layers.empty() ? nullptr : layers.data();
        if (!CheckXr(instance, xrEndFrame(session, &end_info), "xrEndFrame")) {
            break;
        }
    }

    if (session_running) {
        xrEndSession(session);
    }
    xrDestroySpace(local_space);
    xrDestroySpace(aim_right_space);
    xrDestroySpace(aim_left_space);
    xrDestroySpace(grip_right_space);
    xrDestroySpace(grip_left_space);
    glDeleteFramebuffers(1, &framebuffer);
    glDeleteVertexArrays(1, &vao);
    glDeleteProgram(program);
    glDeleteTextures(1, &source_texture);
    DestroyViewSwapchains(&swapchain_views);
    xrDestroyActionSet(action_set);
    xrDestroySession(session);
    xrDestroyInstance(instance);
    glfwDestroyWindow(window);
    glfwTerminate();
    CloseSharedFrameFile(&shared_frame);
    return 0;
}
