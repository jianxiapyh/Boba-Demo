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
#include <atomic>
#include <chrono>
#include <cctype>
#include <csignal>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <thread>
#include <unistd.h>
#include <vector>

namespace {

constexpr int kPanelWindowWidth = 64;
constexpr int kPanelWindowHeight = 64;
#ifdef BOBA_IMMERSIVE_BRIDGE
constexpr uint32_t kMinExpectedHeaderVersion = 2;
constexpr uint32_t kMaxExpectedHeaderVersion = 3;
constexpr const char* kExpectedSharedFrameMagic = "BOBAQIM1";
constexpr const char* kBinaryUsageName = "boba_immersive_bridge";
constexpr const char* kApplicationName = "Demo: Boba Immersive XR Application";
#else
constexpr uint32_t kExpectedHeaderVersion = 2;
constexpr const char* kExpectedSharedFrameMagic = "BOBAQST1";
constexpr const char* kBinaryUsageName = "boba_immersive_demo";
constexpr const char* kApplicationName = "Demo: Boba Immersive XR Application";
#endif
constexpr float kPanelDistanceMeters = 1.1f;
constexpr float kPanelWidthMeters = 1.2f;
constexpr float kPanelYOffsetMeters = 0.0f;
constexpr float kModalHeadLockedDistanceMeters = 1.35f;
constexpr float kNearZ = 0.02f;
constexpr float kFarZ = 100.0f;
constexpr float kSelectPressedThreshold = 0.75f;
constexpr float kExitPressedThreshold = 0.85f;
constexpr uint32_t kPresentationModeStereoFullscreen = 0u;
constexpr uint32_t kPresentationModeMonoPanel = 1u;
constexpr uint32_t kPresentationModeHeadLockedPanel = 2u;
constexpr const char* kExpectedOverlayMagic = "BOBAOVL1";
constexpr uint32_t kOverlayHeaderVersion = 2u;
constexpr uint32_t kOverlayCommandStrideFloats = 14u;
constexpr const char* kExpectedModalMagic = "BOBAMOD1";
constexpr uint32_t kModalHeaderVersion = 1u;
constexpr uint32_t kModalValidFlagVisible = 1u << 0u;
constexpr uint32_t kModalValidFlagLeft = 1u << 1u;
constexpr uint32_t kModalValidFlagRight = 1u << 2u;
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
    uint32_t presentation_mode;
    uint32_t reserved0;
    uint8_t padding[8];
};

static_assert(sizeof(SharedFrameHeader) == 64, "SharedFrameHeader size mismatch");

struct SharedFrameFile {
    int fd = -1;
    void* mapped = MAP_FAILED;
    size_t mapped_size = 0;
    const SharedFrameHeader* header = nullptr;
#ifdef BOBA_IMMERSIVE_BRIDGE
    const struct SharedFramePoseMetadataSlot* pose_metadata = nullptr;
    uint32_t pose_metadata_slot_count = 0;
#endif
    const uint8_t* payload = nullptr;
};

#ifdef BOBA_IMMERSIVE_BRIDGE
constexpr uint32_t kSharedFramePoseMetadataValidLeft = 1u << 0;
constexpr uint32_t kSharedFramePoseMetadataValidRight = 1u << 1;

#pragma pack(push, 1)
struct SharedFramePoseMetadataSlot {
    uint64_t frame_id;
    uint32_t valid_flags;
    uint32_t reserved0;
    float left_position[3];
    float left_orientation[4];
    float left_fov[4];
    float right_position[3];
    float right_orientation[4];
    float right_fov[4];
    uint8_t padding[24];
};
#pragma pack(pop)

static_assert(
    sizeof(SharedFramePoseMetadataSlot) == 128,
    "SharedFramePoseMetadataSlot size mismatch");

struct ImmersiveFramePoseMetadata {
    uint64_t frame_id = 0;
    bool valid[2] = {false, false};
    XrPosef pose[2] = {};
    XrFovf fov[2] = {};
};

bool FloatArrayIsFinite(const float* values, size_t count) {
    for (size_t index = 0; index < count; ++index) {
        if (!std::isfinite(values[index])) {
            return false;
        }
    }
    return true;
}

bool ReadPoseMetadataEye(const float* position,
                         const float* orientation,
                         const float* fov,
                         XrPosef* pose_out,
                         XrFovf* fov_out) {
    if (
        !FloatArrayIsFinite(position, 3) ||
        !FloatArrayIsFinite(orientation, 4) ||
        !FloatArrayIsFinite(fov, 4)
    ) {
        return false;
    }
    pose_out->position = {position[0], position[1], position[2]};
    pose_out->orientation = {
        orientation[0],
        orientation[1],
        orientation[2],
        orientation[3],
    };
    fov_out->angleLeft = fov[0];
    fov_out->angleRight = fov[1];
    fov_out->angleUp = fov[2];
    fov_out->angleDown = fov[3];
    return true;
}

ImmersiveFramePoseMetadata ReadImmersiveFramePoseMetadata(
    const SharedFrameFile& file,
    const SharedFrameHeader& header) {
    ImmersiveFramePoseMetadata metadata;
    metadata.frame_id = header.latest_frame_id;
    if (
        file.pose_metadata == nullptr ||
        header.latest_slot >= file.pose_metadata_slot_count
    ) {
        return metadata;
    }
    const SharedFramePoseMetadataSlot& slot_metadata =
        file.pose_metadata[header.latest_slot];
    if (slot_metadata.frame_id != header.latest_frame_id) {
        return metadata;
    }
    metadata.frame_id = slot_metadata.frame_id;
    if (
        (slot_metadata.valid_flags & kSharedFramePoseMetadataValidLeft) != 0 &&
        ReadPoseMetadataEye(slot_metadata.left_position,
                            slot_metadata.left_orientation,
                            slot_metadata.left_fov,
                            &metadata.pose[0],
                            &metadata.fov[0])
    ) {
        metadata.valid[0] = true;
    }
    if (
        (slot_metadata.valid_flags & kSharedFramePoseMetadataValidRight) != 0 &&
        ReadPoseMetadataEye(slot_metadata.right_position,
                            slot_metadata.right_orientation,
                            slot_metadata.right_fov,
                            &metadata.pose[1],
                            &metadata.fov[1])
    ) {
        metadata.valid[1] = true;
    }
    return metadata;
}

bool ImmersiveFramePoseMetadataStereoValid(
    const ImmersiveFramePoseMetadata& metadata) {
    return metadata.valid[0] && metadata.valid[1];
}
#endif

#pragma pack(push, 1)
struct SharedOverlayHeader {
    char magic[8];
    uint32_t version;
    uint32_t command_stride_floats;
    uint32_t max_commands_per_eye;
    uint64_t latest_overlay_id;
    uint32_t left_count;
    uint32_t right_count;
    uint32_t reserved0;
    uint8_t padding[24];
};
#pragma pack(pop)

static_assert(sizeof(SharedOverlayHeader) == 64, "SharedOverlayHeader size mismatch");

#pragma pack(push, 1)
struct SharedOverlaySlotMetadata {
    uint64_t frame_id;
    uint32_t left_count;
    uint32_t right_count;
    uint32_t reserved0;
    uint32_t reserved1;
    uint8_t padding[8];
};
#pragma pack(pop)

static_assert(
    sizeof(SharedOverlaySlotMetadata) == 32,
    "SharedOverlaySlotMetadata size mismatch");

struct SharedOverlayFile {
    int fd = -1;
    void* mapped = MAP_FAILED;
    size_t mapped_size = 0;
    const SharedOverlayHeader* header = nullptr;
    const SharedOverlaySlotMetadata* slot_metadata = nullptr;
    uint32_t slot_count = 0;
    const float* payload = nullptr;
};

#pragma pack(push, 1)
struct SharedModalHeader {
    char magic[8];
    uint32_t version;
    uint32_t max_width;
    uint32_t max_height;
    uint32_t slot_count;
    uint64_t latest_modal_id;
    uint32_t reserved0;
    uint32_t reserved1;
    uint8_t padding[24];
};
#pragma pack(pop)

static_assert(sizeof(SharedModalHeader) == 64, "SharedModalHeader size mismatch");

#pragma pack(push, 1)
struct SharedModalSlotMetadata {
    uint64_t frame_id;
    uint32_t valid_flags;
    uint32_t width;
    uint32_t height;
    uint32_t reserved0;
    float left_quad[8];
    float right_quad[8];
    float width_m;
    float height_m;
    uint8_t padding[32];
};
#pragma pack(pop)

static_assert(
    sizeof(SharedModalSlotMetadata) == 128,
    "SharedModalSlotMetadata size mismatch");

struct SharedModalFile {
    int fd = -1;
    void* mapped = MAP_FAILED;
    size_t mapped_size = 0;
    const SharedModalHeader* header = nullptr;
    const SharedModalSlotMetadata* slot_metadata = nullptr;
    uint32_t slot_count = 0;
    const uint8_t* payload = nullptr;
};

struct ModalOverlayData {
    bool visible = false;
    uint32_t width = 0;
    uint32_t height = 0;
    float width_m = 0.0f;
    float height_m = 0.0f;
    bool eye_valid[2] = {false, false};
    float quads[2][8] = {};
};

struct ModalReadPayload {
    ModalOverlayData data;
    std::vector<uint8_t> rgba;
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

struct ThumbstickStateSample {
    bool available = false;
    float x = 0.0f;
    float y = 0.0f;
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

#ifdef BOBA_IMMERSIVE_BRIDGE
const char* PresentationModeLabel(uint32_t presentation_mode) {
    switch (presentation_mode) {
        case kPresentationModeStereoFullscreen:
            return "stereo_fullscreen";
        case kPresentationModeMonoPanel:
            return "mono_panel";
        case kPresentationModeHeadLockedPanel:
            return "head_locked_panel";
        default:
            return "unknown";
    }
}

bool IsValidPresentationMode(uint32_t presentation_mode) {
    return presentation_mode == kPresentationModeStereoFullscreen ||
           presentation_mode == kPresentationModeMonoPanel ||
           presentation_mode == kPresentationModeHeadLockedPanel;
}
#endif

void HandleSignal(int) {
    g_stop_requested = 1;
}

bool ParseArgs(int argc,
               char** argv,
               std::string* frame_path,
               std::string* overlay_path,
               std::string* overlay_modal_path) {
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--frame-path" && i + 1 < argc) {
            *frame_path = argv[++i];
            continue;
        }
        if (arg == "--overlay-path" && i + 1 < argc) {
            *overlay_path = argv[++i];
            continue;
        }
        if (arg == "--overlay-modal-path" && i + 1 < argc) {
            *overlay_modal_path = argv[++i];
            continue;
        }
        std::cerr << "Usage: " << kBinaryUsageName
                  << " --frame-path /tmp/boba_quest_frame.bin"
                  << " [--overlay-path /tmp/boba_quest_overlay.bin]"
                  << " [--overlay-modal-path /tmp/boba_quest_overlay_modal.bin]\n";
        return false;
    }

    if (frame_path->empty()) {
        std::cerr << "Missing required --frame-path argument.\n";
        return false;
    }
    return true;
}

#ifdef BOBA_IMMERSIVE_BRIDGE
enum class ImmersiveViewerUploadMode {
    Pbo,
    DirectMmap,
    LegacyCopy,
};

enum class ImmersiveViewerUploadThreadRequest {
    Auto,
    Render,
    Async,
};

enum class ImmersiveViewerUploadThreadMode {
    Render,
    Async,
};

const char* ImmersiveViewerUploadModeLabel(ImmersiveViewerUploadMode mode) {
    switch (mode) {
        case ImmersiveViewerUploadMode::Pbo:
            return "pbo";
        case ImmersiveViewerUploadMode::DirectMmap:
            return "direct";
        case ImmersiveViewerUploadMode::LegacyCopy:
            return "legacy";
        default:
            return "unknown";
    }
}

const char* ImmersiveViewerUploadThreadModeLabel(ImmersiveViewerUploadThreadMode mode) {
    switch (mode) {
        case ImmersiveViewerUploadThreadMode::Render:
            return "render";
        case ImmersiveViewerUploadThreadMode::Async:
            return "async";
        default:
            return "unknown";
    }
}

ImmersiveViewerUploadMode ReadImmersiveViewerUploadMode() {
    const char* raw_mode = std::getenv("BOBA_IMMERSIVE_VIEWER_UPLOAD_MODE");
    if (raw_mode == nullptr || raw_mode[0] == '\0') {
        return ImmersiveViewerUploadMode::Pbo;
    }

    std::string mode(raw_mode);
    std::transform(mode.begin(), mode.end(), mode.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });

    if (mode == "pbo" || mode == "auto" || mode == "default") {
        return ImmersiveViewerUploadMode::Pbo;
    }
    if (mode == "direct" || mode == "mmap" || mode == "zero_copy" ||
        mode == "zerocopy") {
        return ImmersiveViewerUploadMode::DirectMmap;
    }
    if (mode == "legacy" || mode == "copy" || mode == "vector") {
        return ImmersiveViewerUploadMode::LegacyCopy;
    }
    std::cerr << "Unknown BOBA_IMMERSIVE_VIEWER_UPLOAD_MODE='" << raw_mode
              << "', using pbo upload.\n";
    return ImmersiveViewerUploadMode::Pbo;
}

ImmersiveViewerUploadThreadRequest ReadImmersiveViewerUploadThreadRequest() {
    const char* raw_mode = std::getenv("BOBA_IMMERSIVE_VIEWER_UPLOAD_THREAD");
    if (raw_mode == nullptr || raw_mode[0] == '\0') {
        return ImmersiveViewerUploadThreadRequest::Auto;
    }

    std::string mode(raw_mode);
    std::transform(mode.begin(), mode.end(), mode.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    if (mode == "auto" || mode == "default") {
        return ImmersiveViewerUploadThreadRequest::Auto;
    }
    if (mode == "render" || mode == "render_thread" || mode == "off") {
        return ImmersiveViewerUploadThreadRequest::Render;
    }
    if (mode == "async" || mode == "thread" || mode == "upload_thread") {
        return ImmersiveViewerUploadThreadRequest::Async;
    }
    std::cerr << "Unknown BOBA_IMMERSIVE_VIEWER_UPLOAD_THREAD='" << raw_mode
              << "', using auto.\n";
    return ImmersiveViewerUploadThreadRequest::Auto;
}

constexpr uint64_t kDefaultImmersiveUploadSlotCount = 5;
constexpr uint64_t kMinImmersiveUploadSlotCount = 3;
constexpr uint64_t kMaxImmersiveUploadSlotCount = 8;
constexpr uint64_t kDefaultImmersiveViewerUploadBusyBackoffUs = 100;

struct ImmersiveUploadSlot {
    GLuint textures[2] = {0, 0};
    GLuint pbos[2] = {0, 0};
    GLuint modal_texture = 0;
    GLsync fence = nullptr;
    bool has_frame = false;
    uint64_t frame_id = 0;
    std::vector<float> overlay_commands[2];
    ModalOverlayData modal_overlay;
#ifdef BOBA_IMMERSIVE_BRIDGE
    ImmersiveFramePoseMetadata pose_metadata;
#endif
};

void ConfigureSourceTexture(GLuint texture, uint32_t width, uint32_t height) {
    glBindTexture(GL_TEXTURE_2D, texture);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0,
                 GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
}

void UploadModalTexture(GLuint texture, const ModalReadPayload& modal_payload) {
    if (
        texture == 0 ||
        !modal_payload.data.visible ||
        modal_payload.data.width == 0 ||
        modal_payload.data.height == 0 ||
        modal_payload.rgba.empty()
    ) {
        return;
    }
    glBindTexture(GL_TEXTURE_2D, texture);
    glTexImage2D(GL_TEXTURE_2D,
                 0,
                 GL_RGBA8,
                 modal_payload.data.width,
                 modal_payload.data.height,
                 0,
                 GL_RGBA,
                 GL_UNSIGNED_BYTE,
                 modal_payload.rgba.data());
    glBindTexture(GL_TEXTURE_2D, 0);
}

void DestroyImmersiveUploadSlots(std::vector<ImmersiveUploadSlot>* slots) {
    if (slots == nullptr) {
        return;
    }
    for (auto& slot : *slots) {
        if (slot.fence != nullptr) {
            glDeleteSync(slot.fence);
            slot.fence = nullptr;
        }
        glDeleteBuffers(2, slot.pbos);
        slot.pbos[0] = 0;
        slot.pbos[1] = 0;
        glDeleteTextures(2, slot.textures);
        slot.textures[0] = 0;
        slot.textures[1] = 0;
        if (slot.modal_texture != 0) {
            glDeleteTextures(1, &slot.modal_texture);
            slot.modal_texture = 0;
        }
        slot.has_frame = false;
        slot.modal_overlay = ModalOverlayData{};
    }
    slots->clear();
}

bool InitializeImmersivePboUploadSlots(uint32_t width,
                                       uint32_t height,
                                       size_t eye_frame_bytes,
                                       uint32_t modal_max_width,
                                       uint32_t modal_max_height,
                                       uint64_t requested_slot_count,
                                       std::vector<ImmersiveUploadSlot>* slots,
                                       std::string* error_message) {
    if (slots == nullptr) {
        return false;
    }
    DestroyImmersiveUploadSlots(slots);
    slots->resize(static_cast<size_t>(requested_slot_count));
    for (auto& slot : *slots) {
        glGenTextures(2, slot.textures);
        glGenBuffers(2, slot.pbos);
        glGenTextures(1, &slot.modal_texture);
        for (int eye = 0; eye < 2; ++eye) {
            if (slot.textures[eye] == 0 || slot.pbos[eye] == 0) {
                if (error_message != nullptr) {
                    *error_message = "failed to allocate texture or PBO";
                }
                DestroyImmersiveUploadSlots(slots);
                return false;
            }
            ConfigureSourceTexture(slot.textures[eye], width, height);
            glBindBuffer(GL_PIXEL_UNPACK_BUFFER, slot.pbos[eye]);
            glBufferData(GL_PIXEL_UNPACK_BUFFER,
                         static_cast<GLsizeiptr>(eye_frame_bytes),
                         nullptr,
                         GL_STREAM_DRAW);
        }
        if (slot.modal_texture == 0) {
            if (error_message != nullptr) {
                *error_message = "failed to allocate modal texture";
            }
            DestroyImmersiveUploadSlots(slots);
            return false;
        }
        ConfigureSourceTexture(slot.modal_texture,
                               std::max<uint32_t>(modal_max_width, 1u),
                               std::max<uint32_t>(modal_max_height, 1u));
    }
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
    glBindTexture(GL_TEXTURE_2D, 0);
    const GLenum error = glGetError();
    if (error != GL_NO_ERROR) {
        if (error_message != nullptr) {
            *error_message = "OpenGL error during PBO initialization: " +
                             std::to_string(static_cast<unsigned int>(error));
        }
        DestroyImmersiveUploadSlots(slots);
        return false;
    }
    return true;
}

bool RetireUploadSlotFence(ImmersiveUploadSlot* slot) {
    if (slot == nullptr || slot->fence == nullptr) {
        return true;
    }
    const GLenum wait_result = glClientWaitSync(slot->fence, 0, 0);
    if (wait_result == GL_ALREADY_SIGNALED || wait_result == GL_CONDITION_SATISFIED) {
        glDeleteSync(slot->fence);
        slot->fence = nullptr;
        return true;
    }
    return false;
}

int FindReusableUploadSlot(std::vector<ImmersiveUploadSlot>* slots,
                           int preferred_slot,
                           uint64_t* busy_slot_count) {
    if (slots == nullptr || slots->empty()) {
        return -1;
    }
    const int slot_count = static_cast<int>(slots->size());
    preferred_slot = ((preferred_slot % slot_count) + slot_count) % slot_count;
    for (int offset = 0; offset < slot_count; ++offset) {
        const int slot_index = (preferred_slot + offset) % slot_count;
        ImmersiveUploadSlot& slot = (*slots)[slot_index];
        if (RetireUploadSlotFence(&slot)) {
            return slot_index;
        }
        if (busy_slot_count != nullptr) {
            ++(*busy_slot_count);
        }
    }
    return -1;
}

int FindReusableUploadSlotExcluding(std::vector<ImmersiveUploadSlot>* slots,
                                    int preferred_slot,
                                    int excluded_slot_a,
                                    int excluded_slot_b,
                                    int excluded_slot_c,
                                    uint64_t* busy_slot_count) {
    if (slots == nullptr || slots->empty()) {
        return -1;
    }
    const int slot_count = static_cast<int>(slots->size());
    preferred_slot = ((preferred_slot % slot_count) + slot_count) % slot_count;
    for (int offset = 0; offset < slot_count; ++offset) {
        const int slot_index = (preferred_slot + offset) % slot_count;
        if (slot_index == excluded_slot_a ||
            slot_index == excluded_slot_b ||
            slot_index == excluded_slot_c) {
            continue;
        }
        ImmersiveUploadSlot& slot = (*slots)[slot_index];
        if (RetireUploadSlotFence(&slot)) {
            return slot_index;
        }
        if (busy_slot_count != nullptr) {
            ++(*busy_slot_count);
        }
    }
    return -1;
}

uint64_t ReadUnsignedEnvOrDefault(const char* name, uint64_t default_value) {
    const char* raw_value = std::getenv(name);
    if (raw_value == nullptr || raw_value[0] == '\0') {
        return default_value;
    }
    char* end_ptr = nullptr;
    const unsigned long long parsed = std::strtoull(raw_value, &end_ptr, 10);
    if (end_ptr == raw_value || (end_ptr != nullptr && *end_ptr != '\0')) {
        std::cerr << "Invalid " << name << "='" << raw_value
                  << "', using " << default_value << ".\n";
        return default_value;
    }
    return static_cast<uint64_t>(parsed);
}

uint64_t ReadUnsignedEnvClamped(const char* name,
                                uint64_t default_value,
                                uint64_t min_value,
                                uint64_t max_value) {
    uint64_t value = ReadUnsignedEnvOrDefault(name, default_value);
    if (value < min_value || value > max_value) {
        std::cerr << "Invalid " << name << "=" << value
                  << ", using " << default_value << ".\n";
        return default_value;
    }
    return value;
}
#endif

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

bool AppendAnchorResetBindings(
    XrInstance instance,
    XrAction anchor_reset_click_action,
    const char* profile_string,
    std::vector<XrActionSuggestedBinding>* bindings) {
    if (std::strcmp(profile_string, "/interaction_profiles/oculus/touch_controller") != 0) {
        return true;
    }

    const char* paths[] = {
        "/user/hand/left/input/thumbstick/click",
        "/user/hand/right/input/thumbstick/click",
    };
    bindings->reserve(bindings->size() + 2);
    for (const char* path_string : paths) {
        XrPath path = XR_NULL_PATH;
        if (!StringToPath(instance, path_string, &path)) {
            return false;
        }
        bindings->push_back({anchor_reset_click_action, path});
    }
    return true;
}

bool AppendThumbstickBindings(
    XrInstance instance,
    XrAction thumbstick_axis_action,
    const char* profile_string,
    std::vector<XrActionSuggestedBinding>* bindings) {
    if (std::strcmp(profile_string, "/interaction_profiles/oculus/touch_controller") != 0) {
        return true;
    }

    const char* paths[] = {
        "/user/hand/left/input/thumbstick",
        "/user/hand/right/input/thumbstick",
    };
    bindings->reserve(bindings->size() + 2);
    for (const char* path_string : paths) {
        XrPath path = XR_NULL_PATH;
        if (!StringToPath(instance, path_string, &path)) {
            return false;
        }
        bindings->push_back({thumbstick_axis_action, path});
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

bool AppendExitBindings(
    XrInstance instance,
    XrAction exit_value_action,
    const char* profile_string,
    std::vector<XrActionSuggestedBinding>* bindings) {
    if (std::strcmp(profile_string, "/interaction_profiles/khr/simple_controller") == 0) {
        return true;
    }

    const char* paths[] = {
        "/user/hand/left/input/squeeze/value",
        "/user/hand/right/input/squeeze/value",
    };
    bindings->reserve(bindings->size() + 2);
    for (const char* path_string : paths) {
        XrPath path = XR_NULL_PATH;
        if (!StringToPath(instance, path_string, &path)) {
            return false;
        }
        bindings->push_back({exit_value_action, path});
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

bool QueryThumbstickState(
    XrInstance instance,
    XrSession session,
    XrAction thumbstick_axis_action,
    XrPath subaction_path,
    ThumbstickStateSample* sample) {
    XrActionStateGetInfo get_info =
        MakeXrStruct<XrActionStateGetInfo>(XR_TYPE_ACTION_STATE_GET_INFO);
    get_info.action = thumbstick_axis_action;
    get_info.subactionPath = subaction_path;

    XrActionStateVector2f state =
        MakeXrStruct<XrActionStateVector2f>(XR_TYPE_ACTION_STATE_VECTOR2F);
    if (!CheckXr(instance, xrGetActionStateVector2f(session, &get_info, &state),
                 "xrGetActionStateVector2f(thumbstick)")) {
        return false;
    }

    if (state.isActive == XR_TRUE) {
        sample->available = true;
        sample->x = state.currentState.x;
        sample->y = state.currentState.y;
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

bool QueryExitValueState(
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
                 "xrGetActionStateFloat(exit)")) {
        return false;
    }

    if (state.isActive == XR_TRUE) {
        sample->available = true;
        sample->value = state.currentState;
        sample->pressed = state.currentState >= kExitPressedThreshold;
        sample->source = "value";
    }
    return true;
}

void PrintControllerJson(const char* prefix, const ControllerPoseSample& pose,
                         const ControllerPoseSample& grip,
                         const ControllerPoseSample& aim,
                         const SelectStateSample& select,
                         const SelectStateSample& anchor_cycle,
                         const SelectStateSample& anchor_reset,
                         const ThumbstickStateSample& thumbstick,
                         const SelectStateSample& snap_assist,
                         const SelectStateSample& exit_value) {
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
    std::cout << "\"grip_active\":" << (grip.action_active ? 1 : 0) << ",";
    std::cout << "\"grip_position_valid\":" << (grip.position_valid ? 1 : 0) << ",";
    std::cout << "\"grip_orientation_valid\":" << (grip.orientation_valid ? 1 : 0) << ",";
    std::cout << "\"grip_position_tracked\":" << (grip.position_tracked ? 1 : 0) << ",";
    std::cout << "\"grip_orientation_tracked\":" << (grip.orientation_tracked ? 1 : 0)
              << ",";
    std::cout << "\"grip_position\":["
              << grip.pose.position.x << "," << grip.pose.position.y << ","
              << grip.pose.position.z << "],";
    std::cout << "\"grip_orientation\":["
              << grip.pose.orientation.x << "," << grip.pose.orientation.y << ","
              << grip.pose.orientation.z << "," << grip.pose.orientation.w << "],";
    std::cout << "\"aim_active\":" << (aim.action_active ? 1 : 0) << ",";
    std::cout << "\"aim_position_valid\":" << (aim.position_valid ? 1 : 0) << ",";
    std::cout << "\"aim_orientation_valid\":" << (aim.orientation_valid ? 1 : 0) << ",";
    std::cout << "\"aim_position_tracked\":" << (aim.position_tracked ? 1 : 0) << ",";
    std::cout << "\"aim_orientation_tracked\":" << (aim.orientation_tracked ? 1 : 0)
              << ",";
    std::cout << "\"aim_position\":["
              << aim.pose.position.x << "," << aim.pose.position.y << ","
              << aim.pose.position.z << "],";
    std::cout << "\"aim_orientation\":["
              << aim.pose.orientation.x << "," << aim.pose.orientation.y << ","
              << aim.pose.orientation.z << "," << aim.pose.orientation.w << "],";
    std::cout << "\"select_available\":" << (select.available ? 1 : 0) << ",";
    std::cout << "\"select_pressed\":" << (select.pressed ? 1 : 0) << ",";
    std::cout << "\"select_value\":" << select.value << ",";
    std::cout << "\"select_source\":\"" << select.source << "\",";
    std::cout << "\"anchor_cycle_available\":" << (anchor_cycle.available ? 1 : 0) << ",";
    std::cout << "\"anchor_cycle_pressed\":" << (anchor_cycle.pressed ? 1 : 0) << ",";
    std::cout << "\"anchor_cycle_source\":\"" << anchor_cycle.source << "\",";
    std::cout << "\"anchor_reset_available\":" << (anchor_reset.available ? 1 : 0) << ",";
    std::cout << "\"anchor_reset_pressed\":" << (anchor_reset.pressed ? 1 : 0) << ",";
    std::cout << "\"anchor_reset_source\":\"" << anchor_reset.source << "\",";
    std::cout << "\"thumbstick_available\":" << (thumbstick.available ? 1 : 0) << ",";
    std::cout << "\"thumbstick_x\":" << thumbstick.x << ",";
    std::cout << "\"thumbstick_y\":" << thumbstick.y << ",";
    std::cout << "\"snap_assist_available\":" << (snap_assist.available ? 1 : 0) << ",";
    std::cout << "\"snap_assist_pressed\":" << (snap_assist.pressed ? 1 : 0) << ",";
    std::cout << "\"snap_assist_source\":\"" << snap_assist.source << "\",";
    std::cout << "\"exit_available\":" << (exit_value.available ? 1 : 0) << ",";
    std::cout << "\"exit_pressed\":" << (exit_value.pressed ? 1 : 0) << ",";
    std::cout << "\"exit_value\":" << exit_value.value << ",";
    std::cout << "\"exit_source\":\"" << exit_value.source << "\"";
    std::cout << "}";
}

#ifdef BOBA_IMMERSIVE_BRIDGE
void PrintEyeJson(const char* prefix, const XrView& view, const SwapchainView& swapchain_view,
                  bool pose_valid, bool pose_tracked) {
    std::cout << "\"" << prefix << "\":{";
    std::cout << "\"pose_valid\":" << (pose_valid ? 1 : 0) << ",";
    std::cout << "\"pose_tracked\":" << (pose_tracked ? 1 : 0) << ",";
    std::cout << "\"position\":["
              << view.pose.position.x << "," << view.pose.position.y << ","
              << view.pose.position.z << "],";
    std::cout << "\"orientation\":["
              << view.pose.orientation.x << "," << view.pose.orientation.y << ","
              << view.pose.orientation.z << "," << view.pose.orientation.w << "],";
    std::cout << "\"fov\":{"
              << "\"angle_left\":" << view.fov.angleLeft << ","
              << "\"angle_right\":" << view.fov.angleRight << ","
              << "\"angle_up\":" << view.fov.angleUp << ","
              << "\"angle_down\":" << view.fov.angleDown << "},";
    std::cout << "\"recommended_width\":" << swapchain_view.width << ",";
    std::cout << "\"recommended_height\":" << swapchain_view.height;
    std::cout << "}";
}
#endif

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
    if (std::memcmp(file->header->magic, kExpectedSharedFrameMagic, 8) != 0) {
        std::cerr << "Shared frame header magic mismatch.\n";
        munmap(file->mapped, file->mapped_size);
        close(file->fd);
        file->mapped = MAP_FAILED;
        file->fd = -1;
        return false;
    }
#ifdef BOBA_IMMERSIVE_BRIDGE
    if (
        file->header->version < kMinExpectedHeaderVersion ||
        file->header->version > kMaxExpectedHeaderVersion
    ) {
        std::cerr << "Shared frame header version mismatch: " << file->header->version
                  << " (expected " << kMinExpectedHeaderVersion << ".."
                  << kMaxExpectedHeaderVersion << ")\n";
        munmap(file->mapped, file->mapped_size);
        close(file->fd);
        file->mapped = MAP_FAILED;
        file->fd = -1;
        return false;
    }
    size_t metadata_bytes = 0;
    if (file->header->version >= 3) {
        metadata_bytes = static_cast<size_t>(file->header->reserved0);
        const size_t required_metadata_bytes =
            static_cast<size_t>(file->header->slot_count) *
            sizeof(SharedFramePoseMetadataSlot);
        if (metadata_bytes < required_metadata_bytes) {
            std::cerr << "Immersive shared frame metadata is smaller than expected: got "
                      << metadata_bytes << " expected at least "
                      << required_metadata_bytes << "\n";
            munmap(file->mapped, file->mapped_size);
            close(file->fd);
            file->mapped = MAP_FAILED;
            file->fd = -1;
            return false;
        }
        file->pose_metadata =
            reinterpret_cast<const SharedFramePoseMetadataSlot*>(
                static_cast<const uint8_t*>(file->mapped) + sizeof(SharedFrameHeader));
        file->pose_metadata_slot_count =
            static_cast<uint32_t>(metadata_bytes / sizeof(SharedFramePoseMetadataSlot));
    }
    const size_t payload_offset = sizeof(SharedFrameHeader) + metadata_bytes;
#else
    if (file->header->version != kExpectedHeaderVersion) {
        std::cerr << "Shared frame header version mismatch: " << file->header->version << "\n";
        munmap(file->mapped, file->mapped_size);
        close(file->fd);
        file->mapped = MAP_FAILED;
        file->fd = -1;
        return false;
    }
    const size_t payload_offset = sizeof(SharedFrameHeader);
#endif

    const size_t expected_size =
        payload_offset +
        static_cast<size_t>(file->header->slot_count) * file->header->frame_bytes;
    if (file->mapped_size < expected_size) {
        std::cerr << "Shared frame file is smaller than expected.\n";
        munmap(file->mapped, file->mapped_size);
        close(file->fd);
        file->mapped = MAP_FAILED;
        file->fd = -1;
        return false;
    }
#ifdef BOBA_IMMERSIVE_BRIDGE
    const uint32_t eye_frame_bytes =
        file->header->width * file->header->height * file->header->channels;
    if (file->header->frame_bytes != eye_frame_bytes * 2u) {
        std::cerr << "Immersive shared frame_bytes mismatch: got "
                  << file->header->frame_bytes << " expected " << (eye_frame_bytes * 2u)
                  << "\n";
        munmap(file->mapped, file->mapped_size);
        close(file->fd);
        file->mapped = MAP_FAILED;
        file->fd = -1;
        return false;
    }
#endif

    file->payload = static_cast<const uint8_t*>(file->mapped) + payload_offset;
    std::cerr << "Opened shared frame file " << frame_path << " ("
              << file->header->width << "x" << file->header->height
              << " channels=" << file->header->channels
              << " slots=" << file->header->slot_count
              << " version=" << file->header->version
#ifdef BOBA_IMMERSIVE_BRIDGE
              << " metadata_bytes=" << metadata_bytes
#endif
              << ")\n";
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
#ifdef BOBA_IMMERSIVE_BRIDGE
    file->pose_metadata = nullptr;
    file->pose_metadata_slot_count = 0;
#endif
    file->payload = nullptr;
}

bool OpenSharedOverlayFile(const std::string& overlay_path, SharedOverlayFile* file) {
    if (overlay_path.empty()) {
        return false;
    }
    file->fd = open(overlay_path.c_str(), O_RDONLY);
    if (file->fd < 0) {
        perror("open(overlay_path)");
        return false;
    }

    struct stat st {};
    if (fstat(file->fd, &st) != 0) {
        perror("fstat(overlay_path)");
        close(file->fd);
        file->fd = -1;
        return false;
    }

    file->mapped_size = static_cast<size_t>(st.st_size);
    file->mapped = mmap(nullptr, file->mapped_size, PROT_READ, MAP_SHARED, file->fd, 0);
    if (file->mapped == MAP_FAILED) {
        perror("mmap(overlay_path)");
        close(file->fd);
        file->fd = -1;
        return false;
    }

    file->header = static_cast<const SharedOverlayHeader*>(file->mapped);
    if (std::memcmp(file->header->magic, kExpectedOverlayMagic, 8) != 0) {
        std::cerr << "Shared overlay header magic mismatch.\n";
        munmap(file->mapped, file->mapped_size);
        close(file->fd);
        file->mapped = MAP_FAILED;
        file->fd = -1;
        return false;
    }
    if (file->header->version != kOverlayHeaderVersion ||
        file->header->command_stride_floats != kOverlayCommandStrideFloats) {
        std::cerr << "Shared overlay header version/stride mismatch: version="
                  << file->header->version << " stride="
                  << file->header->command_stride_floats << "\n";
        munmap(file->mapped, file->mapped_size);
        close(file->fd);
        file->mapped = MAP_FAILED;
        file->fd = -1;
        return false;
    }
    file->slot_count = file->header->reserved0;
    if (file->slot_count == 0) {
        std::cerr << "Shared overlay header has zero slot_count.\n";
        munmap(file->mapped, file->mapped_size);
        close(file->fd);
        file->mapped = MAP_FAILED;
        file->fd = -1;
        return false;
    }
    const size_t metadata_bytes =
        static_cast<size_t>(file->slot_count) * sizeof(SharedOverlaySlotMetadata);
    const size_t payload_bytes =
        static_cast<size_t>(file->slot_count) * 2u *
        static_cast<size_t>(file->header->max_commands_per_eye) *
        static_cast<size_t>(file->header->command_stride_floats) * sizeof(float);
    const size_t expected_size =
        sizeof(SharedOverlayHeader) + metadata_bytes + payload_bytes;
    if (file->mapped_size < expected_size) {
        std::cerr << "Shared overlay file is smaller than expected.\n";
        munmap(file->mapped, file->mapped_size);
        close(file->fd);
        file->mapped = MAP_FAILED;
        file->fd = -1;
        return false;
    }
    file->slot_metadata = reinterpret_cast<const SharedOverlaySlotMetadata*>(
        static_cast<const uint8_t*>(file->mapped) + sizeof(SharedOverlayHeader));
    file->payload = reinterpret_cast<const float*>(
        static_cast<const uint8_t*>(file->mapped) + sizeof(SharedOverlayHeader) +
            metadata_bytes);
    std::cerr << "Opened shared overlay file " << overlay_path
              << " max_commands_per_eye=" << file->header->max_commands_per_eye
              << " slot_count=" << file->slot_count
              << "\n";
    return true;
}

void CloseSharedOverlayFile(SharedOverlayFile* file) {
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
    file->slot_metadata = nullptr;
    file->slot_count = 0;
    file->payload = nullptr;
}

bool OpenSharedModalFile(const std::string& modal_path, SharedModalFile* file) {
    if (modal_path.empty()) {
        return false;
    }
    file->fd = open(modal_path.c_str(), O_RDONLY);
    if (file->fd < 0) {
        perror("open(overlay_modal_path)");
        return false;
    }

    struct stat st {};
    if (fstat(file->fd, &st) != 0) {
        perror("fstat(overlay_modal_path)");
        close(file->fd);
        file->fd = -1;
        return false;
    }

    file->mapped_size = static_cast<size_t>(st.st_size);
    file->mapped = mmap(nullptr, file->mapped_size, PROT_READ, MAP_SHARED, file->fd, 0);
    if (file->mapped == MAP_FAILED) {
        perror("mmap(overlay_modal_path)");
        close(file->fd);
        file->fd = -1;
        return false;
    }

    file->header = static_cast<const SharedModalHeader*>(file->mapped);
    if (std::memcmp(file->header->magic, kExpectedModalMagic, 8) != 0) {
        std::cerr << "Shared modal header magic mismatch.\n";
        munmap(file->mapped, file->mapped_size);
        close(file->fd);
        file->mapped = MAP_FAILED;
        file->fd = -1;
        return false;
    }
    if (file->header->version != kModalHeaderVersion ||
        file->header->max_width == 0 ||
        file->header->max_height == 0 ||
        file->header->slot_count == 0) {
        std::cerr << "Shared modal header version/size mismatch: version="
                  << file->header->version << " max_width="
                  << file->header->max_width << " max_height="
                  << file->header->max_height << " slot_count="
                  << file->header->slot_count << "\n";
        munmap(file->mapped, file->mapped_size);
        close(file->fd);
        file->mapped = MAP_FAILED;
        file->fd = -1;
        return false;
    }
    file->slot_count = file->header->slot_count;
    const size_t metadata_bytes =
        static_cast<size_t>(file->slot_count) * sizeof(SharedModalSlotMetadata);
    const size_t payload_bytes =
        static_cast<size_t>(file->slot_count) *
        static_cast<size_t>(file->header->max_width) *
        static_cast<size_t>(file->header->max_height) * 4u;
    const size_t expected_size =
        sizeof(SharedModalHeader) + metadata_bytes + payload_bytes;
    if (file->mapped_size < expected_size) {
        std::cerr << "Shared modal file is smaller than expected.\n";
        munmap(file->mapped, file->mapped_size);
        close(file->fd);
        file->mapped = MAP_FAILED;
        file->fd = -1;
        return false;
    }
    file->slot_metadata = reinterpret_cast<const SharedModalSlotMetadata*>(
        static_cast<const uint8_t*>(file->mapped) + sizeof(SharedModalHeader));
    file->payload =
        static_cast<const uint8_t*>(file->mapped) + sizeof(SharedModalHeader) +
        metadata_bytes;
    std::cerr << "Opened shared modal overlay file " << modal_path
              << " max_width=" << file->header->max_width
              << " max_height=" << file->header->max_height
              << " slot_count=" << file->slot_count
              << "\n";
    return true;
}

void CloseSharedModalFile(SharedModalFile* file) {
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
    file->slot_metadata = nullptr;
    file->slot_count = 0;
    file->payload = nullptr;
}

enum class OverlayLatchReadStatus {
    Unavailable,
    Match,
    Empty,
    Mismatch,
};

OverlayLatchReadStatus ReadOverlayCommandsForFrameSlot(
    const SharedOverlayFile& file,
    uint64_t frame_id,
    uint64_t frame_slot,
    std::vector<float>* left_commands,
    std::vector<float>* right_commands) {
    if (left_commands != nullptr) {
        left_commands->clear();
    }
    if (right_commands != nullptr) {
        right_commands->clear();
    }
    if (
        file.header == nullptr ||
        file.slot_metadata == nullptr ||
        file.payload == nullptr
    ) {
        return OverlayLatchReadStatus::Unavailable;
    }
    if (frame_slot >= file.slot_count) {
        return OverlayLatchReadStatus::Mismatch;
    }
    const SharedOverlayHeader header = *file.header;
    const SharedOverlaySlotMetadata metadata =
        file.slot_metadata[static_cast<size_t>(frame_slot)];
    if (metadata.frame_id != frame_id) {
        return OverlayLatchReadStatus::Mismatch;
    }
    const uint32_t max_commands = header.max_commands_per_eye;
    const uint32_t stride = header.command_stride_floats;
    const uint32_t left_count = std::min(metadata.left_count, max_commands);
    const uint32_t right_count = std::min(metadata.right_count, max_commands);
    const size_t slot_stride_floats =
        2u * static_cast<size_t>(max_commands) * static_cast<size_t>(stride);
    const float* slot_base =
        file.payload + static_cast<size_t>(frame_slot) * slot_stride_floats;
    const float* left_base = slot_base;
    const float* right_base =
        slot_base + static_cast<size_t>(max_commands) * static_cast<size_t>(stride);
    if (left_commands != nullptr) {
        left_commands->assign(
            left_base,
            left_base + static_cast<size_t>(left_count) * stride);
    }
    if (right_commands != nullptr) {
        right_commands->assign(
            right_base,
            right_base + static_cast<size_t>(right_count) * stride);
    }
    return (left_count == 0 && right_count == 0)
        ? OverlayLatchReadStatus::Empty
        : OverlayLatchReadStatus::Match;
}

OverlayLatchReadStatus ReadModalForFrameSlot(
    const SharedModalFile& file,
    uint64_t frame_id,
    uint64_t frame_slot,
    ModalReadPayload* payload) {
    if (payload != nullptr) {
        payload->data = ModalOverlayData{};
        payload->rgba.clear();
    }
    if (
        file.header == nullptr ||
        file.slot_metadata == nullptr ||
        file.payload == nullptr
    ) {
        return OverlayLatchReadStatus::Unavailable;
    }
    if (frame_slot >= file.slot_count) {
        return OverlayLatchReadStatus::Mismatch;
    }
    const SharedModalHeader header = *file.header;
    const SharedModalSlotMetadata metadata =
        file.slot_metadata[static_cast<size_t>(frame_slot)];
    if (metadata.frame_id != frame_id) {
        return OverlayLatchReadStatus::Mismatch;
    }
    if (
        (metadata.valid_flags & kModalValidFlagVisible) == 0 ||
        metadata.width == 0 ||
        metadata.height == 0
    ) {
        return OverlayLatchReadStatus::Empty;
    }
    const uint32_t width = std::min(metadata.width, header.max_width);
    const uint32_t height = std::min(metadata.height, header.max_height);
    if (payload != nullptr) {
        payload->data.visible = true;
        payload->data.width = width;
        payload->data.height = height;
        payload->data.width_m =
            (std::isfinite(metadata.width_m) && metadata.width_m > 0.0f)
                ? metadata.width_m
                : 0.0f;
        payload->data.height_m =
            (std::isfinite(metadata.height_m) && metadata.height_m > 0.0f)
                ? metadata.height_m
                : 0.0f;
        payload->data.eye_valid[0] =
            (metadata.valid_flags & kModalValidFlagLeft) != 0;
        payload->data.eye_valid[1] =
            (metadata.valid_flags & kModalValidFlagRight) != 0;
        std::memcpy(payload->data.quads[0], metadata.left_quad, sizeof(metadata.left_quad));
        std::memcpy(payload->data.quads[1], metadata.right_quad, sizeof(metadata.right_quad));
        payload->rgba.resize(static_cast<size_t>(width) * height * 4u);
        const size_t slot_stride =
            static_cast<size_t>(header.max_width) *
            static_cast<size_t>(header.max_height) * 4u;
        const uint8_t* slot_base =
            file.payload + static_cast<size_t>(frame_slot) * slot_stride;
        for (uint32_t row = 0; row < height; ++row) {
            const uint8_t* row_src =
                slot_base + static_cast<size_t>(row) * header.max_width * 4u;
            uint8_t* row_dst =
                payload->rgba.data() + static_cast<size_t>(row) * width * 4u;
            std::memcpy(row_dst, row_src, static_cast<size_t>(width) * 4u);
        }
    }
    return OverlayLatchReadStatus::Match;
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

#ifdef BOBA_IMMERSIVE_BRIDGE
bool UpdateStereoFramePointersIfNeeded(const SharedFrameFile& file,
                                       uint64_t* latest_frame_id,
                                       uint64_t* frame_id_delta,
                                       uint64_t* latest_slot,
                                       uint32_t* presentation_mode,
                                       const uint8_t** left_eye_rgba,
                                       const uint8_t** right_eye_rgba,
                                       ImmersiveFramePoseMetadata* pose_metadata) {
    const SharedFrameHeader header = *file.header;
    if (header.latest_frame_id == *latest_frame_id) {
        if (frame_id_delta != nullptr) {
            *frame_id_delta = 0;
        }
        return false;
    }
    if (header.latest_slot >= header.slot_count) {
        std::cerr << "Invalid latest_slot in shared frame header: " << header.latest_slot
                  << "\n";
        if (frame_id_delta != nullptr) {
            *frame_id_delta = 0;
        }
        return false;
    }

    const uint32_t eye_frame_bytes = header.width * header.height * header.channels;
    const size_t slot_offset = static_cast<size_t>(header.latest_slot) * header.frame_bytes;
    const uint8_t* source = file.payload + slot_offset;
    const uint64_t previous_frame_id = *latest_frame_id;
    if (left_eye_rgba != nullptr) {
        *left_eye_rgba = source;
    }
    if (right_eye_rgba != nullptr) {
        *right_eye_rgba = source + eye_frame_bytes;
    }
    if (pose_metadata != nullptr) {
        *pose_metadata = ReadImmersiveFramePoseMetadata(file, header);
    }
    *latest_frame_id = header.latest_frame_id;
    if (latest_slot != nullptr) {
        *latest_slot = header.latest_slot;
    }
    if (presentation_mode != nullptr) {
        const uint32_t header_presentation_mode = header.presentation_mode;
        if (IsValidPresentationMode(header_presentation_mode)) {
            *presentation_mode = header_presentation_mode;
        } else {
            std::cerr << "Invalid immersive presentation_mode in shared frame header: "
                      << header_presentation_mode << ", defaulting to stereo_fullscreen\n";
            *presentation_mode = kPresentationModeStereoFullscreen;
        }
    }
    if (frame_id_delta != nullptr) {
        *frame_id_delta =
            (header.latest_frame_id > previous_frame_id)
                ? (header.latest_frame_id - previous_frame_id)
                : 1;
    }
    return true;
}

bool UpdateStereoFramesIfNeeded(const SharedFrameFile& file, uint64_t* latest_frame_id,
                                uint64_t* frame_id_delta,
                                uint64_t* latest_slot,
                                uint32_t* presentation_mode,
                                std::vector<uint8_t>* left_eye_rgba,
                                std::vector<uint8_t>* right_eye_rgba,
                                ImmersiveFramePoseMetadata* pose_metadata = nullptr) {
    const uint8_t* left_source = nullptr;
    const uint8_t* right_source = nullptr;
    const bool updated = UpdateStereoFramePointersIfNeeded(file,
                                                           latest_frame_id,
                                                           frame_id_delta,
                                                           latest_slot,
                                                           presentation_mode,
                                                           &left_source,
                                                           &right_source,
                                                           pose_metadata);
    if (!updated) {
        return false;
    }

    const SharedFrameHeader header = *file.header;
    const uint32_t eye_frame_bytes = header.width * header.height * header.channels;
    left_eye_rgba->assign(left_source, left_source + eye_frame_bytes);
    right_eye_rgba->assign(right_source, right_source + eye_frame_bytes);
    return true;
}
#endif

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

XrVector3f RotateVectorByQuaternion(const XrQuaternionf& q, const XrVector3f& v) {
    const float tx = 2.0f * (q.y * v.z - q.z * v.y);
    const float ty = 2.0f * (q.z * v.x - q.x * v.z);
    const float tz = 2.0f * (q.x * v.y - q.y * v.x);
    return {
        v.x + q.w * tx + (q.y * tz - q.z * ty),
        v.y + q.w * ty + (q.z * tx - q.x * tz),
        v.z + q.w * tz + (q.x * ty - q.y * tx),
    };
}

XrPosef MakeHeadLockedModalPose(const std::vector<XrView>& views,
                                uint32_t view_count_output) {
    XrPosef pose{};
    pose.orientation.w = 1.0f;
    if (views.empty() || view_count_output == 0) {
        pose.position.z = -kModalHeadLockedDistanceMeters;
        return pose;
    }
    const uint32_t count =
        std::min<uint32_t>(view_count_output, static_cast<uint32_t>(views.size()));
    pose.orientation = views[0].pose.orientation;
    XrVector3f center{0.0f, 0.0f, 0.0f};
    for (uint32_t index = 0; index < count; ++index) {
        center.x += views[index].pose.position.x;
        center.y += views[index].pose.position.y;
        center.z += views[index].pose.position.z;
    }
    const float inv_count = 1.0f / static_cast<float>(count);
    center.x *= inv_count;
    center.y *= inv_count;
    center.z *= inv_count;

    const XrVector3f forward_offset =
        RotateVectorByQuaternion(pose.orientation,
                                 {0.0f, 0.0f, -kModalHeadLockedDistanceMeters});
    pose.position = {
        center.x + forward_offset.x,
        center.y + forward_offset.y,
        center.z + forward_offset.z,
    };
    return pose;
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

GLuint CreateOverlayProgram() {
    const char* vertex_source = R"GLSL(
        #version 330 core
        layout(location = 0) in vec2 aPos;
        layout(location = 1) in vec4 aColor;
        uniform vec2 uSourceSize;
        out vec4 vColor;
        void main() {
            vec2 ndc = vec2(
                (aPos.x / max(uSourceSize.x, 1.0)) * 2.0 - 1.0,
                1.0 - (aPos.y / max(uSourceSize.y, 1.0)) * 2.0
            );
            gl_Position = vec4(ndc, 0.0, 1.0);
            vColor = aColor;
        }
    )GLSL";

    const char* fragment_source = R"GLSL(
        #version 330 core
        in vec4 vColor;
        out vec4 frag;
        void main() {
            frag = vColor;
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
        std::cerr << "Overlay program link failed: " << log.data() << "\n";
        glDeleteProgram(program);
        return 0;
    }
    return program;
}

void AppendOverlayVertex(std::vector<float>* vertices,
                         float x,
                         float y,
                         float r,
                         float g,
                         float b,
                         float a) {
    vertices->push_back(x);
    vertices->push_back(y);
    vertices->push_back(std::clamp(r / 255.0f, 0.0f, 1.0f));
    vertices->push_back(std::clamp(g / 255.0f, 0.0f, 1.0f));
    vertices->push_back(std::clamp(b / 255.0f, 0.0f, 1.0f));
    vertices->push_back(std::clamp(a, 0.0f, 1.0f));
}

void AppendOverlayTriangle(std::vector<float>* vertices,
                           float x0, float y0,
                           float x1, float y1,
                           float x2, float y2,
                           float r, float g, float b, float a) {
    AppendOverlayVertex(vertices, x0, y0, r, g, b, a);
    AppendOverlayVertex(vertices, x1, y1, r, g, b, a);
    AppendOverlayVertex(vertices, x2, y2, r, g, b, a);
}

void AppendOverlayLine(std::vector<float>* vertices, const float* cmd) {
    const float sx = cmd[1];
    const float sy = cmd[2];
    const float ex = cmd[3];
    const float ey = cmd[4];
    const float radius = std::max(0.5f, cmd[5]);
    const float alpha = cmd[6];
    const float r = cmd[7];
    const float g = cmd[8];
    const float b = cmd[9];
    const float dx = ex - sx;
    const float dy = ey - sy;
    const float length = std::sqrt(dx * dx + dy * dy);
    if (length <= 1.0e-4f) {
        return;
    }
    const float nx = -dy / length * radius;
    const float ny = dx / length * radius;
    AppendOverlayTriangle(vertices, sx + nx, sy + ny, ex + nx, ey + ny,
                          ex - nx, ey - ny, r, g, b, alpha);
    AppendOverlayTriangle(vertices, sx + nx, sy + ny, ex - nx, ey - ny,
                          sx - nx, sy - ny, r, g, b, alpha);
}

void AppendOverlayMarker(std::vector<float>* vertices, const float* cmd) {
    const float cx = cmd[1];
    const float cy = cmd[2];
    const float radius = std::max(1.0f, cmd[5]);
    const float alpha = cmd[6];
    const float r = cmd[7];
    const float g = cmd[8];
    const float b = cmd[9];
    const float x0 = cx - radius;
    const float y0 = cy - radius;
    const float x1 = cx + radius;
    const float y1 = cy + radius;
    AppendOverlayTriangle(vertices, x0, y0, x1, y0, x1, y1, r, g, b, alpha);
    AppendOverlayTriangle(vertices, x0, y0, x1, y1, x0, y1, r, g, b, alpha);
}

void DrawOverlayCommands(const std::vector<float>& commands,
                         GLuint overlay_program,
                         GLuint overlay_vao,
                         GLuint overlay_vbo,
                         GLint overlay_source_size_location,
                         uint32_t source_width,
                         uint32_t source_height) {
    if (commands.empty() || overlay_program == 0 || overlay_vao == 0 || overlay_vbo == 0) {
        return;
    }
    std::vector<float> vertices;
    vertices.reserve(commands.size() * 18u);
    const size_t command_count = commands.size() / kOverlayCommandStrideFloats;
    for (size_t command_index = 0; command_index < command_count; ++command_index) {
        const float* cmd = commands.data() + command_index * kOverlayCommandStrideFloats;
        const int command_type = static_cast<int>(std::round(cmd[0]));
        if (command_type == 0) {
            AppendOverlayLine(&vertices, cmd);
        } else if (command_type == 1) {
            AppendOverlayMarker(&vertices, cmd);
        }
    }
    if (vertices.empty()) {
        return;
    }
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glUseProgram(overlay_program);
    glUniform2f(overlay_source_size_location,
                static_cast<float>(source_width),
                static_cast<float>(source_height));
    glBindVertexArray(overlay_vao);
    glBindBuffer(GL_ARRAY_BUFFER, overlay_vbo);
    glBufferData(GL_ARRAY_BUFFER,
                 static_cast<GLsizeiptr>(vertices.size() * sizeof(float)),
                 vertices.data(),
                 GL_STREAM_DRAW);
    glDrawArrays(GL_TRIANGLES, 0, static_cast<GLsizei>(vertices.size() / 6u));
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);
    glUseProgram(0);
    glDisable(GL_BLEND);
}

GLuint CreateModalProgram() {
    const char* vertex_source = R"GLSL(
        #version 330 core
        layout(location = 0) in vec2 aPos;
        layout(location = 1) in vec2 aUv;
        uniform vec2 uSourceSize;
        out vec2 vUv;
        void main() {
            vec2 ndc = vec2(
                (aPos.x / max(uSourceSize.x, 1.0)) * 2.0 - 1.0,
                1.0 - (aPos.y / max(uSourceSize.y, 1.0)) * 2.0
            );
            gl_Position = vec4(ndc, 0.0, 1.0);
            vUv = aUv;
        }
    )GLSL";

    const char* fragment_source = R"GLSL(
        #version 330 core
        in vec2 vUv;
        out vec4 frag;
        uniform sampler2D uModal;
        void main() {
            frag = texture(uModal, vUv);
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
        std::cerr << "Modal program link failed: " << log.data() << "\n";
        glDeleteProgram(program);
        return 0;
    }
    return program;
}

void DrawModalOverlay(const ModalOverlayData& modal,
                      GLuint modal_texture,
                      GLuint modal_program,
                      GLuint modal_vao,
                      GLuint modal_vbo,
                      GLint modal_source_size_location,
                      GLint modal_texture_location,
                      uint32_t eye_index,
                      uint32_t source_width,
                      uint32_t source_height) {
    if (
        !modal.visible ||
        modal_texture == 0 ||
        modal_program == 0 ||
        modal_vao == 0 ||
        modal_vbo == 0 ||
        eye_index > 1u ||
        !modal.eye_valid[eye_index] ||
        modal.width == 0 ||
        modal.height == 0
    ) {
        return;
    }
    const float* q = modal.quads[eye_index];
    const float vertices[] = {
        q[0], q[1], 0.0f, 0.0f,
        q[2], q[3], 1.0f, 0.0f,
        q[4], q[5], 1.0f, 1.0f,
        q[0], q[1], 0.0f, 0.0f,
        q[4], q[5], 1.0f, 1.0f,
        q[6], q[7], 0.0f, 1.0f,
    };
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glUseProgram(modal_program);
    glUniform2f(modal_source_size_location,
                static_cast<float>(source_width),
                static_cast<float>(source_height));
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, modal_texture);
    glUniform1i(modal_texture_location, 0);
    glBindVertexArray(modal_vao);
    glBindBuffer(GL_ARRAY_BUFFER, modal_vbo);
    glBufferData(GL_ARRAY_BUFFER,
                 static_cast<GLsizeiptr>(sizeof(vertices)),
                 vertices,
                 GL_STREAM_DRAW);
    glDrawArrays(GL_TRIANGLES, 0, 6);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);
    glBindTexture(GL_TEXTURE_2D, 0);
    glUseProgram(0);
    glDisable(GL_BLEND);
}

void DrawModalTextureRect(GLuint modal_texture,
                          GLuint modal_program,
                          GLuint modal_vao,
                          GLuint modal_vbo,
                          GLint modal_source_size_location,
                          GLint modal_texture_location,
                          uint32_t width,
                          uint32_t height) {
    if (
        modal_texture == 0 ||
        modal_program == 0 ||
        modal_vao == 0 ||
        modal_vbo == 0 ||
        width == 0 ||
        height == 0
    ) {
        return;
    }
    const float w = static_cast<float>(width);
    const float h = static_cast<float>(height);
    const float vertices[] = {
        0.0f, 0.0f, 0.0f, 0.0f,
        w,    0.0f, 1.0f, 0.0f,
        w,    h,    1.0f, 1.0f,
        0.0f, 0.0f, 0.0f, 0.0f,
        w,    h,    1.0f, 1.0f,
        0.0f, h,    0.0f, 1.0f,
    };
    glDisable(GL_BLEND);
    glUseProgram(modal_program);
    glUniform2f(modal_source_size_location, w, h);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, modal_texture);
    glUniform1i(modal_texture_location, 0);
    glBindVertexArray(modal_vao);
    glBindBuffer(GL_ARRAY_BUFFER, modal_vbo);
    glBufferData(GL_ARRAY_BUFFER,
                 static_cast<GLsizeiptr>(sizeof(vertices)),
                 vertices,
                 GL_STREAM_DRAW);
    glDrawArrays(GL_TRIANGLES, 0, 6);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);
    glBindTexture(GL_TEXTURE_2D, 0);
    glUseProgram(0);
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

void DestroySwapchainView(SwapchainView* swapchain_view) {
    if (swapchain_view == nullptr) {
        return;
    }
    if (swapchain_view->handle != XR_NULL_HANDLE) {
        xrDestroySwapchain(swapchain_view->handle);
        swapchain_view->handle = XR_NULL_HANDLE;
    }
    swapchain_view->images.clear();
    swapchain_view->width = 0;
    swapchain_view->height = 0;
}

bool CreateSingleSwapchainView(XrInstance instance,
                               XrSession session,
                               int64_t swapchain_format,
                               uint32_t width,
                               uint32_t height,
                               const char* label,
                               SwapchainView* swapchain_view) {
    if (swapchain_view == nullptr || width == 0 || height == 0) {
        return false;
    }
    DestroySwapchainView(swapchain_view);
    swapchain_view->width = width;
    swapchain_view->height = height;

    XrSwapchainCreateInfo swapchain_info =
        MakeXrStruct<XrSwapchainCreateInfo>(XR_TYPE_SWAPCHAIN_CREATE_INFO);
    swapchain_info.createFlags = 0;
    swapchain_info.usageFlags =
        XR_SWAPCHAIN_USAGE_COLOR_ATTACHMENT_BIT | XR_SWAPCHAIN_USAGE_SAMPLED_BIT;
    swapchain_info.format = swapchain_format;
    swapchain_info.sampleCount = 1;
    swapchain_info.width = width;
    swapchain_info.height = height;
    swapchain_info.faceCount = 1;
    swapchain_info.arraySize = 1;
    swapchain_info.mipCount = 1;

    if (!CheckXr(instance,
                 xrCreateSwapchain(session, &swapchain_info, &swapchain_view->handle),
                 label)) {
        DestroySwapchainView(swapchain_view);
        return false;
    }

    uint32_t image_count = 0;
    if (!CheckXr(instance,
                 xrEnumerateSwapchainImages(swapchain_view->handle, 0, &image_count, nullptr),
                 "xrEnumerateSwapchainImages(modal count)")) {
        DestroySwapchainView(swapchain_view);
        return false;
    }

    swapchain_view->images.resize(image_count);
    for (auto& image : swapchain_view->images) {
        image = MakeXrStruct<XrSwapchainImageOpenGLKHR>(
            XR_TYPE_SWAPCHAIN_IMAGE_OPENGL_KHR);
    }
    if (!CheckXr(instance,
                 xrEnumerateSwapchainImages(
                     swapchain_view->handle, image_count, &image_count,
                     reinterpret_cast<XrSwapchainImageBaseHeader*>(
                         swapchain_view->images.data())),
                 "xrEnumerateSwapchainImages(modal list)")) {
        DestroySwapchainView(swapchain_view);
        return false;
    }

    for (const auto& image : swapchain_view->images) {
        glBindTexture(GL_TEXTURE_2D, image.image);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    }
    glBindTexture(GL_TEXTURE_2D, 0);
    return true;
}

bool RenderModalTextureToQuadSwapchain(XrInstance instance,
                                       const ModalOverlayData& modal,
                                       GLuint modal_texture,
                                       SwapchainView* modal_swapchain,
                                       GLuint framebuffer,
                                       GLuint modal_program,
                                       GLuint modal_vao,
                                       GLuint modal_vbo,
                                       GLint modal_source_size_location,
                                       GLint modal_texture_location) {
    if (
        modal_swapchain == nullptr ||
        modal_swapchain->handle == XR_NULL_HANDLE ||
        modal_swapchain->images.empty() ||
        !modal.visible ||
        modal_texture == 0 ||
        modal.width == 0 ||
        modal.height == 0
    ) {
        return false;
    }
    const uint32_t draw_width = std::min(modal.width, modal_swapchain->width);
    const uint32_t draw_height = std::min(modal.height, modal_swapchain->height);
    if (draw_width == 0 || draw_height == 0) {
        return false;
    }

    XrSwapchainImageAcquireInfo acquire_info =
        MakeXrStruct<XrSwapchainImageAcquireInfo>(
            XR_TYPE_SWAPCHAIN_IMAGE_ACQUIRE_INFO);
    uint32_t image_index = 0;
    if (!CheckXr(instance,
                 xrAcquireSwapchainImage(modal_swapchain->handle,
                                         &acquire_info,
                                         &image_index),
                 "xrAcquireSwapchainImage(modal)")) {
        return false;
    }

    bool render_ok = true;
    XrSwapchainImageWaitInfo wait_info =
        MakeXrStruct<XrSwapchainImageWaitInfo>(
            XR_TYPE_SWAPCHAIN_IMAGE_WAIT_INFO);
    wait_info.timeout = XR_INFINITE_DURATION;
    if (!CheckXr(instance,
                 xrWaitSwapchainImage(modal_swapchain->handle, &wait_info),
                 "xrWaitSwapchainImage(modal)")) {
        render_ok = false;
    } else {
        glBindFramebuffer(GL_FRAMEBUFFER, framebuffer);
        glFramebufferTexture2D(GL_FRAMEBUFFER,
                               GL_COLOR_ATTACHMENT0,
                               GL_TEXTURE_2D,
                               modal_swapchain->images[image_index].image,
                               0);
        if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
            std::cerr << "Framebuffer incomplete for modal quad layer.\n";
            render_ok = false;
        } else {
            glViewport(0, 0,
                       static_cast<GLsizei>(modal_swapchain->width),
                       static_cast<GLsizei>(modal_swapchain->height));
            glDisable(GL_DEPTH_TEST);
            glDisable(GL_CULL_FACE);
            glDisable(GL_BLEND);
            glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
            glClear(GL_COLOR_BUFFER_BIT);

            glViewport(0, 0,
                       static_cast<GLsizei>(draw_width),
                       static_cast<GLsizei>(draw_height));
            DrawModalTextureRect(modal_texture,
                                 modal_program,
                                 modal_vao,
                                 modal_vbo,
                                 modal_source_size_location,
                                 modal_texture_location,
                                 draw_width,
                                 draw_height);
            if (modal_swapchain->height > draw_height) {
                glViewport(0,
                           static_cast<GLint>(modal_swapchain->height - draw_height),
                           static_cast<GLsizei>(draw_width),
                           static_cast<GLsizei>(draw_height));
                DrawModalTextureRect(modal_texture,
                                     modal_program,
                                     modal_vao,
                                     modal_vbo,
                                     modal_source_size_location,
                                     modal_texture_location,
                                     draw_width,
                                     draw_height);
            }
            glBindFramebuffer(GL_FRAMEBUFFER, 0);
        }
    }

    XrSwapchainImageReleaseInfo release_info =
        MakeXrStruct<XrSwapchainImageReleaseInfo>(
            XR_TYPE_SWAPCHAIN_IMAGE_RELEASE_INFO);
    if (!CheckXr(instance,
                 xrReleaseSwapchainImage(modal_swapchain->handle, &release_info),
                 "xrReleaseSwapchainImage(modal)")) {
        return false;
    }
    return render_ok;
}

}  // namespace

int main(int argc, char** argv) {
    std::signal(SIGINT, HandleSignal);
    std::signal(SIGTERM, HandleSignal);
    std::cout << std::fixed << std::setprecision(6) << std::unitbuf;
    std::cerr << std::fixed << std::setprecision(6) << std::unitbuf;

    std::string frame_path;
    std::string overlay_path;
    std::string overlay_modal_path;
    if (!ParseArgs(argc, argv, &frame_path, &overlay_path, &overlay_modal_path)) {
        return 2;
    }

    SharedFrameFile shared_frame;
    if (!OpenSharedFrameFile(frame_path, &shared_frame)) {
        return 3;
    }
    SharedOverlayFile shared_overlay;
    const bool overlay_enabled = OpenSharedOverlayFile(overlay_path, &shared_overlay);
    SharedModalFile shared_modal;
    const bool modal_enabled = OpenSharedModalFile(overlay_modal_path, &shared_modal);
    if (shared_frame.header->channels != 4) {
        std::cerr << "Expected RGBA frame data, got channels="
                  << shared_frame.header->channels << "\n";
        CloseSharedModalFile(&shared_modal);
        CloseSharedOverlayFile(&shared_overlay);
        CloseSharedFrameFile(&shared_frame);
        return 4;
    }

    if (!glfwInit()) {
        std::cerr << "glfwInit failed.\n";
        CloseSharedOverlayFile(&shared_overlay);
        CloseSharedFrameFile(&shared_frame);
        return 5;
    }

    glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
    glfwWindowHint(GLFW_CLIENT_API, GLFW_OPENGL_API);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    GLFWwindow* window = glfwCreateWindow(kPanelWindowWidth, kPanelWindowHeight,
                                          kApplicationName, nullptr, nullptr);
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
    std::strncpy(instance_info.applicationInfo.applicationName, kApplicationName,
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
    XrAction anchor_reset_click_action = XR_NULL_HANDLE;
    XrAction thumbstick_axis_action = XR_NULL_HANDLE;
    XrAction snap_assist_click_action = XR_NULL_HANDLE;
    XrAction exit_value_action = XR_NULL_HANDLE;
    if (!create_action("grip_pose", "Grip Pose", XR_ACTION_TYPE_POSE_INPUT,
                       &grip_pose_action) ||
        !create_action("aim_pose", "Aim Pose", XR_ACTION_TYPE_POSE_INPUT, &aim_pose_action) ||
        !create_action("select_click", "Select Click", XR_ACTION_TYPE_BOOLEAN_INPUT,
                       &select_click_action) ||
        !create_action("select_value", "Select Value", XR_ACTION_TYPE_FLOAT_INPUT,
                       &select_value_action) ||
        !create_action("anchor_cycle_click", "Anchor Cycle Click",
                       XR_ACTION_TYPE_BOOLEAN_INPUT, &anchor_cycle_click_action) ||
        !create_action("anchor_reset_click", "Anchor Reset Click",
                       XR_ACTION_TYPE_BOOLEAN_INPUT, &anchor_reset_click_action) ||
        !create_action("thumbstick_axis", "Thumbstick Axis",
                       XR_ACTION_TYPE_VECTOR2F_INPUT, &thumbstick_axis_action) ||
        !create_action("snap_assist_click", "Snap Assist Click",
                       XR_ACTION_TYPE_BOOLEAN_INPUT, &snap_assist_click_action) ||
        !create_action("exit_value", "Exit Value", XR_ACTION_TYPE_FLOAT_INPUT,
                       &exit_value_action)) {
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
            !AppendAnchorResetBindings(instance, anchor_reset_click_action, profile, &bindings) ||
            !AppendThumbstickBindings(instance, thumbstick_axis_action, profile, &bindings) ||
            !AppendSnapAssistBindings(instance, snap_assist_click_action, profile, &bindings) ||
            !AppendExitBindings(instance, exit_value_action, profile, &bindings) ||
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

    const uint32_t eye_frame_bytes =
        shared_frame.header->width * shared_frame.header->height * shared_frame.header->channels;
#ifdef BOBA_IMMERSIVE_BRIDGE
    const ImmersiveViewerUploadMode requested_viewer_upload_mode =
        ReadImmersiveViewerUploadMode();
    ImmersiveViewerUploadMode viewer_upload_mode = requested_viewer_upload_mode;
    const ImmersiveViewerUploadThreadRequest requested_viewer_upload_thread =
        ReadImmersiveViewerUploadThreadRequest();
    ImmersiveViewerUploadThreadMode viewer_upload_thread_mode =
        ImmersiveViewerUploadThreadMode::Render;
    std::string viewer_upload_thread_fallback_reason = "none";
    const uint64_t viewer_upload_ring_slots = ReadUnsignedEnvClamped(
        "BOBA_IMMERSIVE_VIEWER_UPLOAD_RING_SLOTS",
        kDefaultImmersiveUploadSlotCount,
        kMinImmersiveUploadSlotCount,
        kMaxImmersiveUploadSlotCount);
    const uint64_t viewer_upload_busy_backoff_us = ReadUnsignedEnvOrDefault(
        "BOBA_IMMERSIVE_VIEWER_UPLOAD_BUSY_BACKOFF_US",
        kDefaultImmersiveViewerUploadBusyBackoffUs);
    std::vector<ImmersiveUploadSlot> immersive_upload_slots;
    int active_immersive_upload_slot = -1;
    int next_immersive_upload_slot = 0;
    ImmersiveFramePoseMetadata active_source_pose_metadata;
#endif
    GLuint source_textures[2] = {0, 0};
#ifdef BOBA_IMMERSIVE_BRIDGE
    glGenTextures(2, source_textures);
    const int source_texture_count = 2;
#else
    glGenTextures(1, source_textures);
    const int source_texture_count = 1;
#endif
    for (int texture_index = 0; texture_index < source_texture_count; ++texture_index) {
        glBindTexture(GL_TEXTURE_2D, source_textures[texture_index]);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, shared_frame.header->width,
                     shared_frame.header->height, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
    }
    glBindTexture(GL_TEXTURE_2D, 0);
#ifdef BOBA_IMMERSIVE_BRIDGE
    const uint32_t modal_max_width =
        modal_enabled ? shared_modal.header->max_width : 1u;
    const uint32_t modal_max_height =
        modal_enabled ? shared_modal.header->max_height : 1u;
    GLuint active_modal_texture = 0;
    ModalOverlayData active_modal_overlay;
    SwapchainView modal_quad_swapchain;
    bool modal_quad_layer_available = false;
    std::string viewer_modal_layer_mode = "disabled";
    uint64_t viewer_modal_layer_present_count = 0;
    if (modal_enabled) {
        glGenTextures(1, &active_modal_texture);
        if (active_modal_texture != 0) {
            ConfigureSourceTexture(active_modal_texture,
                                   std::max<uint32_t>(modal_max_width, 1u),
                                   std::max<uint32_t>(modal_max_height, 1u));
        }
        modal_quad_layer_available =
            CreateSingleSwapchainView(instance,
                                      session,
                                      color_format,
                                      std::max<uint32_t>(modal_max_width, 1u),
                                      std::max<uint32_t>(modal_max_height, 1u),
                                      "xrCreateSwapchain(modal quad)",
                                      &modal_quad_swapchain);
        if (modal_quad_layer_available) {
            viewer_modal_layer_mode = "head_locked_quad";
            std::cerr << "Immersive bridge modal layer mode: "
                      << viewer_modal_layer_mode
                      << " max_width=" << modal_quad_swapchain.width
                      << " max_height=" << modal_quad_swapchain.height
                      << " z_offset=" << -kModalHeadLockedDistanceMeters << "\n";
        } else {
            std::cerr << "Immersive bridge modal layer unavailable; "
                      << "falling back to Python/source-frame modal path if enabled.\n";
        }
    }
    if (requested_viewer_upload_mode == ImmersiveViewerUploadMode::Pbo) {
        std::string pbo_error;
        if (!InitializeImmersivePboUploadSlots(shared_frame.header->width,
                                               shared_frame.header->height,
                                               static_cast<size_t>(eye_frame_bytes),
                                               modal_max_width,
                                               modal_max_height,
                                               viewer_upload_ring_slots,
                                               &immersive_upload_slots,
                                               &pbo_error)) {
            std::cerr << "Immersive bridge viewer PBO upload unavailable; "
                      << "falling back to legacy upload";
            if (!pbo_error.empty()) {
                std::cerr << ": " << pbo_error;
            }
            std::cerr << "\n";
            viewer_upload_mode = ImmersiveViewerUploadMode::LegacyCopy;
        }
    }
    std::cerr << "Immersive bridge viewer upload mode: "
              << ImmersiveViewerUploadModeLabel(viewer_upload_mode)
              << " ring_slots=" << viewer_upload_ring_slots
              << " busy_backoff_us=" << viewer_upload_busy_backoff_us << "\n";
#endif

    const GLuint program = CreatePanelProgram();
    if (program == 0) {
#ifdef BOBA_IMMERSIVE_BRIDGE
        DestroyImmersiveUploadSlots(&immersive_upload_slots);
        DestroySwapchainView(&modal_quad_swapchain);
        if (active_modal_texture != 0) {
            glDeleteTextures(1, &active_modal_texture);
        }
#endif
        glDeleteTextures(source_texture_count, source_textures);
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
    GLuint overlay_program = 0;
    GLuint overlay_vao = 0;
    GLuint overlay_vbo = 0;
    GLint overlay_source_size_location = -1;
    GLuint modal_program = 0;
    GLuint modal_vao = 0;
    GLuint modal_vbo = 0;
    GLint modal_source_size_location = -1;
    GLint modal_texture_location = -1;
    if (overlay_enabled) {
        overlay_program = CreateOverlayProgram();
        if (overlay_program != 0) {
            overlay_source_size_location = glGetUniformLocation(overlay_program, "uSourceSize");
            glGenVertexArrays(1, &overlay_vao);
            glGenBuffers(1, &overlay_vbo);
            glBindVertexArray(overlay_vao);
            glBindBuffer(GL_ARRAY_BUFFER, overlay_vbo);
            glEnableVertexAttribArray(0);
            glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE,
                                  static_cast<GLsizei>(6 * sizeof(float)),
                                  reinterpret_cast<void*>(0));
            glEnableVertexAttribArray(1);
            glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE,
                                  static_cast<GLsizei>(6 * sizeof(float)),
                                  reinterpret_cast<void*>(2 * sizeof(float)));
            glBindBuffer(GL_ARRAY_BUFFER, 0);
            glBindVertexArray(0);
        }
    }
    if (modal_enabled) {
        modal_program = CreateModalProgram();
        if (modal_program != 0) {
            modal_source_size_location =
                glGetUniformLocation(modal_program, "uSourceSize");
            modal_texture_location =
                glGetUniformLocation(modal_program, "uModal");
            glGenVertexArrays(1, &modal_vao);
            glGenBuffers(1, &modal_vbo);
            glBindVertexArray(modal_vao);
            glBindBuffer(GL_ARRAY_BUFFER, modal_vbo);
            glEnableVertexAttribArray(0);
            glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE,
                                  static_cast<GLsizei>(4 * sizeof(float)),
                                  reinterpret_cast<void*>(0));
            glEnableVertexAttribArray(1);
            glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE,
                                  static_cast<GLsizei>(4 * sizeof(float)),
                                  reinterpret_cast<void*>(2 * sizeof(float)));
            glBindBuffer(GL_ARRAY_BUFFER, 0);
            glBindVertexArray(0);
        }
    }
#ifdef BOBA_IMMERSIVE_BRIDGE
    if (modal_enabled && modal_quad_layer_available && modal_program == 0) {
        modal_quad_layer_available = false;
        viewer_modal_layer_mode = "disabled";
        std::cerr << "Immersive bridge modal layer disabled: modal shader unavailable.\n";
    }
#endif

    XrReferenceSpaceCreateInfo local_space_info =
        MakeXrStruct<XrReferenceSpaceCreateInfo>(XR_TYPE_REFERENCE_SPACE_CREATE_INFO);
    local_space_info.referenceSpaceType = XR_REFERENCE_SPACE_TYPE_LOCAL;
    local_space_info.poseInReferenceSpace.orientation.w = 1.0f;
    XrSpace local_space = XR_NULL_HANDLE;
    if (!CheckXr(instance, xrCreateReferenceSpace(session, &local_space_info, &local_space),
                 "xrCreateReferenceSpace(LOCAL)")) {
        glDeleteFramebuffers(1, &framebuffer);
#ifdef BOBA_IMMERSIVE_BRIDGE
        DestroyImmersiveUploadSlots(&immersive_upload_slots);
        DestroySwapchainView(&modal_quad_swapchain);
        if (active_modal_texture != 0) {
            glDeleteTextures(1, &active_modal_texture);
        }
#endif
        if (overlay_vbo != 0) {
            glDeleteBuffers(1, &overlay_vbo);
        }
        if (overlay_vao != 0) {
            glDeleteVertexArrays(1, &overlay_vao);
        }
        if (overlay_program != 0) {
            glDeleteProgram(overlay_program);
        }
        if (modal_vbo != 0) {
            glDeleteBuffers(1, &modal_vbo);
        }
        if (modal_vao != 0) {
            glDeleteVertexArrays(1, &modal_vao);
        }
        if (modal_program != 0) {
            glDeleteProgram(modal_program);
        }
        glDeleteVertexArrays(1, &vao);
        glDeleteProgram(program);
        glDeleteTextures(source_texture_count, source_textures);
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
    uint64_t applied_source_update_count = 0;
    uint64_t logged_applied_source_update_count = 0;
    uint64_t source_frame_delta_count = 0;
    uint64_t logged_source_frame_delta_count = 0;
    uint64_t coalesced_source_frame_count = 0;
    uint64_t rendered_frame_count = 0;
    uint64_t logged_rendered_frame_count = 0;
    uint64_t texture_upload_count = 0;
    uint64_t logged_texture_upload_count = 0;
    double texture_upload_ms_sum = 0.0;
    double logged_texture_upload_ms_sum = 0.0;
    double texture_upload_mmap_copy_ms_sum = 0.0;
    double logged_texture_upload_mmap_copy_ms_sum = 0.0;
    double texture_upload_gl_ms_sum = 0.0;
    double logged_texture_upload_gl_ms_sum = 0.0;
    double texture_upload_gl_left_ms_sum = 0.0;
    double logged_texture_upload_gl_left_ms_sum = 0.0;
    double texture_upload_gl_right_ms_sum = 0.0;
    double logged_texture_upload_gl_right_ms_sum = 0.0;
    uint64_t texture_upload_slot_miss_count = 0;
    uint64_t logged_texture_upload_slot_miss_count = 0;
    uint64_t texture_upload_slot_drop_count = 0;
    uint64_t logged_texture_upload_slot_drop_count = 0;
    uint64_t texture_upload_slot_busy_count = 0;
    uint64_t logged_texture_upload_slot_busy_count = 0;
    uint64_t texture_upload_busy_backoff_count = 0;
    uint64_t logged_texture_upload_busy_backoff_count = 0;
    double texture_upload_busy_backoff_ms_sum = 0.0;
    double logged_texture_upload_busy_backoff_ms_sum = 0.0;
    uint64_t render_without_upload_count = 0;
    uint64_t logged_render_without_upload_count = 0;
    uint64_t texture_upload_no_new_frame_count = 0;
    uint64_t logged_texture_upload_no_new_frame_count = 0;
#ifdef BOBA_IMMERSIVE_BRIDGE
    std::string viewer_projection_pose_mode = "current_view_fallback";
    uint64_t viewer_source_pose_metadata_valid_count = 0;
    uint64_t viewer_source_pose_metadata_invalid_count = 0;
    uint64_t viewer_source_pose_metadata_fallback_count = 0;
#endif
    uint64_t controller_sample_count = 0;
    auto first_source_update_time = std::chrono::steady_clock::time_point{};
    auto last_source_update_log_time = std::chrono::steady_clock::time_point{};
    auto first_render_frame_time = std::chrono::steady_clock::time_point{};
    auto last_render_log_time = std::chrono::steady_clock::time_point{};
    const float panel_height_meters =
        kPanelWidthMeters *
        (static_cast<float>(shared_frame.header->height) /
         static_cast<float>(shared_frame.header->width));
    Mat4 panel_model = IdentityMatrix();
    bool panel_anchor_initialized = false;
#ifdef BOBA_IMMERSIVE_BRIDGE
    std::vector<uint8_t> display_rgba_left;
    std::vector<uint8_t> display_rgba_right;
    if (viewer_upload_mode == ImmersiveViewerUploadMode::LegacyCopy) {
        display_rgba_left.assign(eye_frame_bytes, 0);
        display_rgba_right.assign(eye_frame_bytes, 0);
    }
    std::vector<float> overlay_commands_left;
    std::vector<float> overlay_commands_right;
    uint32_t current_presentation_mode = kPresentationModeStereoFullscreen;
    uint32_t logged_presentation_mode = current_presentation_mode;
    uint32_t previous_presentation_mode = current_presentation_mode;
    GLFWwindow* async_upload_window = nullptr;
    std::atomic<bool> async_upload_stop_requested(false);
    std::thread async_upload_thread;
    std::mutex viewer_upload_stats_mutex;
    std::mutex async_upload_state_mutex;
    int async_ready_upload_slot = -1;
    int async_recently_rendered_slot = -1;
    uint32_t async_ready_presentation_mode = kPresentationModeStereoFullscreen;
    uint64_t viewer_async_upload_count = 0;
    uint64_t viewer_async_ready_slot_count = 0;
    uint64_t viewer_async_poll_no_new_count = 0;
    uint64_t viewer_overlay_latched_match_count = 0;
    uint64_t viewer_overlay_latched_mismatch_count = 0;
    uint64_t viewer_overlay_latched_empty_count = 0;
    uint64_t viewer_modal_latched_match_count = 0;
    uint64_t viewer_modal_latched_mismatch_count = 0;
    uint64_t viewer_modal_latched_empty_count = 0;
    if (requested_viewer_upload_thread != ImmersiveViewerUploadThreadRequest::Render &&
        viewer_upload_mode == ImmersiveViewerUploadMode::Pbo) {
        async_upload_window =
            glfwCreateWindow(1, 1, "Boba Immersive Upload", nullptr, window);
        if (async_upload_window != nullptr) {
            viewer_upload_thread_mode = ImmersiveViewerUploadThreadMode::Async;
            viewer_upload_thread_fallback_reason = "none";
            glfwMakeContextCurrent(window);
        } else {
            viewer_upload_thread_fallback_reason = "glfwCreateWindow_failed";
            std::cerr << "Immersive bridge async viewer upload unavailable: "
                      << viewer_upload_thread_fallback_reason
                      << "; using render-thread upload.\n";
        }
    } else if (
        requested_viewer_upload_thread == ImmersiveViewerUploadThreadRequest::Async &&
        viewer_upload_mode != ImmersiveViewerUploadMode::Pbo
    ) {
        viewer_upload_thread_fallback_reason = "async_requires_pbo";
        std::cerr << "Immersive bridge async viewer upload unavailable: "
                  << viewer_upload_thread_fallback_reason
                  << "; using render-thread upload.\n";
    } else if (
        requested_viewer_upload_thread == ImmersiveViewerUploadThreadRequest::Auto &&
        viewer_upload_mode != ImmersiveViewerUploadMode::Pbo
    ) {
        viewer_upload_thread_fallback_reason = "async_requires_pbo";
    }
    std::cerr << "Immersive bridge viewer upload thread: mode="
              << ImmersiveViewerUploadThreadModeLabel(viewer_upload_thread_mode)
              << " fallback_reason=" << viewer_upload_thread_fallback_reason
              << "\n";
#else
    std::vector<uint8_t> display_rgba(shared_frame.header->frame_bytes, 0);
#endif
    XrActiveActionSet active_action_set{action_set, XR_NULL_PATH};
    XrActionsSyncInfo sync_info = MakeXrStruct<XrActionsSyncInfo>(XR_TYPE_ACTIONS_SYNC_INFO);
    sync_info.countActiveActionSets = 1;
    sync_info.activeActionSets = &active_action_set;

#ifdef BOBA_IMMERSIVE_BRIDGE
    auto try_upload_latest_stereo_frame = [&]() -> bool {
        uint64_t source_frame_delta = 0;
        uint64_t source_frame_slot = 0;
        const uint8_t* upload_left_rgba = nullptr;
        const uint8_t* upload_right_rgba = nullptr;
        ImmersiveFramePoseMetadata upload_pose_metadata;
        std::vector<float> upload_overlay_commands_left;
        std::vector<float> upload_overlay_commands_right;
        double frame_mmap_copy_ms = 0.0;
        double frame_gl_upload_left_ms = 0.0;
        double frame_gl_upload_right_ms = 0.0;

        auto poll_stereo_source = [&]() -> bool {
            if (viewer_upload_mode == ImmersiveViewerUploadMode::LegacyCopy) {
                const auto copy_start = std::chrono::steady_clock::now();
                const bool updated =
                    UpdateStereoFramesIfNeeded(shared_frame,
                                               &latest_frame_id,
                                               &source_frame_delta,
                                               &source_frame_slot,
                                               &current_presentation_mode,
                                               &display_rgba_left,
                                               &display_rgba_right,
                                               &upload_pose_metadata);
                const auto copy_end = std::chrono::steady_clock::now();
                if (updated) {
                    frame_mmap_copy_ms +=
                        std::chrono::duration<double, std::milli>(
                            copy_end - copy_start).count();
                    upload_left_rgba = display_rgba_left.data();
                    upload_right_rgba = display_rgba_right.data();
                }
                return updated;
            }
            return UpdateStereoFramePointersIfNeeded(shared_frame,
                                                     &latest_frame_id,
                                                     &source_frame_delta,
                                                     &source_frame_slot,
                                                     &current_presentation_mode,
                                                     &upload_left_rgba,
                                                     &upload_right_rgba,
                                                     &upload_pose_metadata);
        };

        bool stereo_source_updated = poll_stereo_source();
        if (!stereo_source_updated) {
            ++texture_upload_no_new_frame_count;
            return false;
        }

        const auto source_update_time = std::chrono::steady_clock::now();
        ++applied_source_update_count;
        source_frame_delta_count += source_frame_delta;
        if (source_frame_delta > 0) {
            coalesced_source_frame_count += source_frame_delta - 1;
        }
        if (applied_source_update_count == 1) {
            first_source_update_time = source_update_time;
            last_source_update_log_time = source_update_time;
            logged_applied_source_update_count = 0;
            logged_source_frame_delta_count = 0;
        }
        if (current_presentation_mode != logged_presentation_mode) {
            std::cerr << "Immersive bridge presentation mode: "
                      << PresentationModeLabel(current_presentation_mode) << "\n";
            logged_presentation_mode = current_presentation_mode;
        }
        if (current_presentation_mode == kPresentationModeMonoPanel &&
            previous_presentation_mode != kPresentationModeMonoPanel) {
            panel_anchor_initialized = false;
        }
        if (latest_frame_id == 1 || latest_frame_id >= logged_source_frame_id + 120) {
            std::cerr << "Immersive bridge received source frame " << latest_frame_id
                      << "\n";
            logged_source_frame_id = latest_frame_id;
        }
        const OverlayLatchReadStatus overlay_latch_status =
            ReadOverlayCommandsForFrameSlot(shared_overlay,
                                            latest_frame_id,
                                            source_frame_slot,
                                            &upload_overlay_commands_left,
                                            &upload_overlay_commands_right);
        ModalReadPayload upload_modal_payload;
        const OverlayLatchReadStatus modal_latch_status =
            ReadModalForFrameSlot(shared_modal,
                                  latest_frame_id,
                                  source_frame_slot,
                                  &upload_modal_payload);
        const double elapsed_s =
            std::chrono::duration<double>(
                source_update_time - first_source_update_time).count();
        const double since_last_log_s =
            std::chrono::duration<double>(
                source_update_time - last_source_update_log_time).count();
        if (applied_source_update_count == 1 || since_last_log_s >= 1.0) {
            const uint64_t applied_updates_since_last_log =
                applied_source_update_count - logged_applied_source_update_count;
            const uint64_t source_frame_delta_since_last_log =
                source_frame_delta_count - logged_source_frame_delta_count;
            const double update_recent_fps =
                (since_last_log_s > 0.0)
                    ? (static_cast<double>(applied_updates_since_last_log) /
                       since_last_log_s)
                    : 0.0;
            const double source_delta_recent_fps =
                (since_last_log_s > 0.0)
                    ? (static_cast<double>(source_frame_delta_since_last_log) /
                       since_last_log_s)
                    : 0.0;
            std::cerr << std::fixed << std::setprecision(2)
                      << "Immersive bridge viewer_source_stats "
                      << "latest_frame_id=" << latest_frame_id << " "
                      << "update_count=" << applied_source_update_count << " "
                      << "source_frame_delta_count=" << source_frame_delta_count << " "
                      << "coalesced_frame_count=" << coalesced_source_frame_count << " "
                      << "elapsed_s=" << elapsed_s << " "
                      << "update_recent_fps=" << update_recent_fps << " "
                      << "source_delta_recent_fps=" << source_delta_recent_fps << "\n";
            last_source_update_log_time = source_update_time;
            logged_applied_source_update_count = applied_source_update_count;
            logged_source_frame_delta_count = source_frame_delta_count;
        }

        const auto upload_start = std::chrono::steady_clock::now();
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
        bool upload_completed = true;
        if (viewer_upload_mode == ImmersiveViewerUploadMode::Pbo) {
            uint64_t busy_slot_count = 0;
            const int upload_slot_index =
                FindReusableUploadSlot(&immersive_upload_slots,
                                       next_immersive_upload_slot,
                                       &busy_slot_count);
            texture_upload_slot_busy_count += busy_slot_count;
            if (upload_slot_index < 0) {
                ++texture_upload_slot_miss_count;
                upload_completed = false;
            } else {
                ImmersiveUploadSlot& upload_slot =
                    immersive_upload_slots[static_cast<size_t>(upload_slot_index)];
                const uint8_t* eye_sources[2] = {
                    upload_left_rgba,
                    upload_right_rgba,
                };
                bool pbo_upload_ok = true;
                for (int eye = 0; eye < 2; ++eye) {
                    const auto copy_start = std::chrono::steady_clock::now();
                    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, upload_slot.pbos[eye]);
                    glBufferData(GL_PIXEL_UNPACK_BUFFER,
                                 static_cast<GLsizeiptr>(eye_frame_bytes),
                                 nullptr,
                                 GL_STREAM_DRAW);
                    void* mapped = glMapBufferRange(
                        GL_PIXEL_UNPACK_BUFFER,
                        0,
                        static_cast<GLsizeiptr>(eye_frame_bytes),
                        GL_MAP_WRITE_BIT | GL_MAP_INVALIDATE_BUFFER_BIT);
                    if (mapped == nullptr) {
                        pbo_upload_ok = false;
                        break;
                    }
                    std::memcpy(mapped, eye_sources[eye], eye_frame_bytes);
                    if (glUnmapBuffer(GL_PIXEL_UNPACK_BUFFER) != GL_TRUE) {
                        pbo_upload_ok = false;
                        break;
                    }
                    const auto copy_end = std::chrono::steady_clock::now();
                    frame_mmap_copy_ms +=
                        std::chrono::duration<double, std::milli>(
                            copy_end - copy_start).count();

                    const auto eye_upload_start = std::chrono::steady_clock::now();
                    glBindTexture(GL_TEXTURE_2D, upload_slot.textures[eye]);
                    glTexSubImage2D(GL_TEXTURE_2D,
                                    0,
                                    0,
                                    0,
                                    shared_frame.header->width,
                                    shared_frame.header->height,
                                    GL_RGBA,
                                    GL_UNSIGNED_BYTE,
                                    reinterpret_cast<const void*>(0));
                    const auto eye_upload_end = std::chrono::steady_clock::now();
                    const double eye_upload_ms =
                        std::chrono::duration<double, std::milli>(
                            eye_upload_end - eye_upload_start).count();
                    if (eye == 0) {
                        frame_gl_upload_left_ms += eye_upload_ms;
                    } else {
                        frame_gl_upload_right_ms += eye_upload_ms;
                    }
                }
                glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
                glBindTexture(GL_TEXTURE_2D, 0);
                if (!pbo_upload_ok) {
                    std::cerr << "Immersive bridge PBO upload failed; "
                              << "falling back to legacy upload.\n";
                    viewer_upload_mode = ImmersiveViewerUploadMode::LegacyCopy;
                    upload_completed = false;
                } else {
                    UploadModalTexture(upload_slot.modal_texture, upload_modal_payload);
                    if (upload_slot.fence != nullptr) {
                        glDeleteSync(upload_slot.fence);
                        upload_slot.fence = nullptr;
                    }
                    upload_slot.fence = glFenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE, 0);
                    upload_slot.has_frame = true;
                    upload_slot.frame_id = latest_frame_id;
                    upload_slot.pose_metadata = upload_pose_metadata;
                    upload_slot.overlay_commands[0] = upload_overlay_commands_left;
                    upload_slot.overlay_commands[1] = upload_overlay_commands_right;
                    upload_slot.modal_overlay = upload_modal_payload.data;
                    active_immersive_upload_slot = upload_slot_index;
                    next_immersive_upload_slot =
                        (upload_slot_index + 1) %
                        static_cast<int>(immersive_upload_slots.size());
                }
            }
        }
        if (viewer_upload_mode != ImmersiveViewerUploadMode::Pbo) {
            const auto left_upload_start = std::chrono::steady_clock::now();
            glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
            glBindTexture(GL_TEXTURE_2D, source_textures[0]);
            glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, shared_frame.header->width,
                            shared_frame.header->height, GL_RGBA, GL_UNSIGNED_BYTE,
                            upload_left_rgba);
            const auto left_upload_end = std::chrono::steady_clock::now();
            frame_gl_upload_left_ms =
                std::chrono::duration<double, std::milli>(
                    left_upload_end - left_upload_start).count();
            const auto right_upload_start = std::chrono::steady_clock::now();
            glBindTexture(GL_TEXTURE_2D, source_textures[1]);
            glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, shared_frame.header->width,
                            shared_frame.header->height, GL_RGBA, GL_UNSIGNED_BYTE,
                            upload_right_rgba);
            const auto right_upload_end = std::chrono::steady_clock::now();
            frame_gl_upload_right_ms =
                std::chrono::duration<double, std::milli>(
                    right_upload_end - right_upload_start).count();
            glBindTexture(GL_TEXTURE_2D, 0);
            upload_completed = true;
            active_source_pose_metadata = upload_pose_metadata;
            overlay_commands_left = upload_overlay_commands_left;
            overlay_commands_right = upload_overlay_commands_right;
            UploadModalTexture(active_modal_texture, upload_modal_payload);
            active_modal_overlay = upload_modal_payload.data;
        }
        const auto upload_end = std::chrono::steady_clock::now();
        if (upload_completed) {
            texture_upload_ms_sum +=
                std::chrono::duration<double, std::milli>(
                    upload_end - upload_start).count();
            texture_upload_mmap_copy_ms_sum += frame_mmap_copy_ms;
            texture_upload_gl_left_ms_sum += frame_gl_upload_left_ms;
            texture_upload_gl_right_ms_sum += frame_gl_upload_right_ms;
            texture_upload_gl_ms_sum += (
                frame_gl_upload_left_ms + frame_gl_upload_right_ms
            );
            if (ImmersiveFramePoseMetadataStereoValid(upload_pose_metadata)) {
                ++viewer_source_pose_metadata_valid_count;
            } else {
                ++viewer_source_pose_metadata_invalid_count;
            }
            if (overlay_latch_status == OverlayLatchReadStatus::Match) {
                ++viewer_overlay_latched_match_count;
            } else if (overlay_latch_status == OverlayLatchReadStatus::Empty) {
                ++viewer_overlay_latched_empty_count;
            } else if (overlay_latch_status == OverlayLatchReadStatus::Mismatch) {
                ++viewer_overlay_latched_mismatch_count;
            }
            if (modal_latch_status == OverlayLatchReadStatus::Match) {
                ++viewer_modal_latched_match_count;
            } else if (modal_latch_status == OverlayLatchReadStatus::Empty) {
                ++viewer_modal_latched_empty_count;
            } else if (modal_latch_status == OverlayLatchReadStatus::Mismatch) {
                ++viewer_modal_latched_mismatch_count;
            }
            ++texture_upload_count;
        }
        return upload_completed;
    };
    if (
        viewer_upload_thread_mode == ImmersiveViewerUploadThreadMode::Async &&
        async_upload_window != nullptr
    ) {
        async_upload_thread = std::thread([&]() {
            glfwMakeContextCurrent(async_upload_window);
            glfwSwapInterval(0);

            uint64_t async_latest_frame_id = 0;
            uint64_t async_applied_source_update_count = 0;
            uint64_t async_logged_applied_source_update_count = 0;
            uint64_t async_source_frame_delta_count = 0;
            uint64_t async_logged_source_frame_delta_count = 0;
            uint64_t async_coalesced_source_frame_count = 0;
            auto async_first_source_update_time =
                std::chrono::steady_clock::time_point{};
            auto async_last_source_update_log_time =
                std::chrono::steady_clock::time_point{};
            int async_next_upload_slot = 0;
            uint64_t async_slot_pressure_frame_id = 0;

            while (
                !async_upload_stop_requested.load(std::memory_order_relaxed) &&
                !g_stop_requested
            ) {
                const SharedFrameHeader header = *shared_frame.header;
                if (
                    async_slot_pressure_frame_id != 0 &&
                    header.latest_frame_id != async_slot_pressure_frame_id
                ) {
                    {
                        std::lock_guard<std::mutex> stats_lock(
                            viewer_upload_stats_mutex);
                        ++texture_upload_slot_drop_count;
                    }
                    async_slot_pressure_frame_id = 0;
                }
                if (header.latest_frame_id == async_latest_frame_id) {
                    {
                        std::lock_guard<std::mutex> stats_lock(
                            viewer_upload_stats_mutex);
                        ++viewer_async_poll_no_new_count;
                    }
                    std::this_thread::sleep_for(std::chrono::microseconds(100));
                    continue;
                }
                if (header.latest_slot >= header.slot_count) {
                    std::cerr << "Invalid latest_slot in shared frame header: "
                              << header.latest_slot << "\n";
                    std::this_thread::sleep_for(std::chrono::microseconds(250));
                    continue;
                }

                uint64_t busy_slot_count = 0;
                int excluded_active_slot = -1;
                int excluded_ready_slot = -1;
                int excluded_recently_rendered_slot = -1;
                {
                    std::lock_guard<std::mutex> state_lock(
                        async_upload_state_mutex);
                    excluded_active_slot = active_immersive_upload_slot;
                    excluded_ready_slot = async_ready_upload_slot;
                    excluded_recently_rendered_slot = async_recently_rendered_slot;
                }
                const int upload_slot_index =
                    FindReusableUploadSlotExcluding(&immersive_upload_slots,
                                                    async_next_upload_slot,
                                                    excluded_active_slot,
                                                    excluded_ready_slot,
                                                    excluded_recently_rendered_slot,
                                                    &busy_slot_count);
                {
                    std::lock_guard<std::mutex> stats_lock(viewer_upload_stats_mutex);
                    texture_upload_slot_busy_count += busy_slot_count;
                }
                if (upload_slot_index < 0) {
                    if (async_slot_pressure_frame_id != header.latest_frame_id) {
                        std::lock_guard<std::mutex> stats_lock(
                            viewer_upload_stats_mutex);
                        ++texture_upload_slot_miss_count;
                        async_slot_pressure_frame_id = header.latest_frame_id;
                    }
                    const auto backoff_start = std::chrono::steady_clock::now();
                    std::this_thread::yield();
                    if (viewer_upload_busy_backoff_us > 0) {
                        std::this_thread::sleep_for(
                            std::chrono::microseconds(
                                viewer_upload_busy_backoff_us));
                    }
                    const auto backoff_end = std::chrono::steady_clock::now();
                    {
                        std::lock_guard<std::mutex> stats_lock(
                            viewer_upload_stats_mutex);
                        ++texture_upload_busy_backoff_count;
                        texture_upload_busy_backoff_ms_sum +=
                            std::chrono::duration<double, std::milli>(
                                backoff_end - backoff_start).count();
                    }
                    continue;
                }

                const uint32_t header_presentation_mode =
                    IsValidPresentationMode(header.presentation_mode)
                        ? header.presentation_mode
                        : kPresentationModeStereoFullscreen;
                const ImmersiveFramePoseMetadata upload_pose_metadata =
                    ReadImmersiveFramePoseMetadata(shared_frame, header);
                std::vector<float> upload_overlay_commands_left;
                std::vector<float> upload_overlay_commands_right;
                const OverlayLatchReadStatus overlay_latch_status =
                    ReadOverlayCommandsForFrameSlot(shared_overlay,
                                                    header.latest_frame_id,
                                                    header.latest_slot,
                                                    &upload_overlay_commands_left,
                                                    &upload_overlay_commands_right);
                ModalReadPayload upload_modal_payload;
                const OverlayLatchReadStatus modal_latch_status =
                    ReadModalForFrameSlot(shared_modal,
                                          header.latest_frame_id,
                                          header.latest_slot,
                                          &upload_modal_payload);
                const uint32_t local_eye_frame_bytes =
                    header.width * header.height * header.channels;
                const size_t slot_offset =
                    static_cast<size_t>(header.latest_slot) * header.frame_bytes;
                const uint8_t* source = shared_frame.payload + slot_offset;
                const uint8_t* eye_sources[2] = {
                    source,
                    source + local_eye_frame_bytes,
                };

                ImmersiveUploadSlot& upload_slot =
                    immersive_upload_slots[static_cast<size_t>(upload_slot_index)];
                double frame_mmap_copy_ms = 0.0;
                double frame_gl_upload_left_ms = 0.0;
                double frame_gl_upload_right_ms = 0.0;
                const auto upload_start = std::chrono::steady_clock::now();
                glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
                bool pbo_upload_ok = true;
                for (int eye = 0; eye < 2; ++eye) {
                    const auto copy_start = std::chrono::steady_clock::now();
                    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, upload_slot.pbos[eye]);
                    glBufferData(GL_PIXEL_UNPACK_BUFFER,
                                 static_cast<GLsizeiptr>(eye_frame_bytes),
                                 nullptr,
                                 GL_STREAM_DRAW);
                    void* mapped = glMapBufferRange(
                        GL_PIXEL_UNPACK_BUFFER,
                        0,
                        static_cast<GLsizeiptr>(eye_frame_bytes),
                        GL_MAP_WRITE_BIT | GL_MAP_INVALIDATE_BUFFER_BIT);
                    if (mapped == nullptr) {
                        pbo_upload_ok = false;
                        break;
                    }
                    std::memcpy(mapped, eye_sources[eye], eye_frame_bytes);
                    if (glUnmapBuffer(GL_PIXEL_UNPACK_BUFFER) != GL_TRUE) {
                        pbo_upload_ok = false;
                        break;
                    }
                    const auto copy_end = std::chrono::steady_clock::now();
                    frame_mmap_copy_ms +=
                        std::chrono::duration<double, std::milli>(
                            copy_end - copy_start).count();

                    const auto eye_upload_start = std::chrono::steady_clock::now();
                    glBindTexture(GL_TEXTURE_2D, upload_slot.textures[eye]);
                    glTexSubImage2D(GL_TEXTURE_2D,
                                    0,
                                    0,
                                    0,
                                    shared_frame.header->width,
                                    shared_frame.header->height,
                                    GL_RGBA,
                                    GL_UNSIGNED_BYTE,
                                    reinterpret_cast<const void*>(0));
                    const auto eye_upload_end = std::chrono::steady_clock::now();
                    const double eye_upload_ms =
                        std::chrono::duration<double, std::milli>(
                            eye_upload_end - eye_upload_start).count();
                    if (eye == 0) {
                        frame_gl_upload_left_ms += eye_upload_ms;
                    } else {
                        frame_gl_upload_right_ms += eye_upload_ms;
                    }
                }
                glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
                glBindTexture(GL_TEXTURE_2D, 0);
                if (!pbo_upload_ok) {
                    std::cerr << "Immersive bridge async PBO upload failed; "
                              << "dropping upload.\n";
                    std::this_thread::sleep_for(std::chrono::microseconds(250));
                    continue;
                }
                UploadModalTexture(upload_slot.modal_texture, upload_modal_payload);

                GLsync upload_fence = glFenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE, 0);
                glFlush();
                bool upload_fence_complete = upload_fence == nullptr;
                while (
                    upload_fence != nullptr &&
                    !async_upload_stop_requested.load(std::memory_order_relaxed) &&
                    !g_stop_requested
                ) {
                    const GLenum wait_result =
                        glClientWaitSync(upload_fence,
                                         GL_SYNC_FLUSH_COMMANDS_BIT,
                                         1000000);
                    if (
                        wait_result == GL_ALREADY_SIGNALED ||
                        wait_result == GL_CONDITION_SATISFIED
                    ) {
                        upload_fence_complete = true;
                        break;
                    }
                    if (wait_result == GL_WAIT_FAILED) {
                        std::cerr << "Immersive bridge async upload fence wait failed.\n";
                        break;
                    }
                }
                if (upload_fence != nullptr) {
                    glDeleteSync(upload_fence);
                }
                if (!upload_fence_complete) {
                    continue;
                }

                const auto upload_end = std::chrono::steady_clock::now();
                upload_slot.has_frame = true;
                upload_slot.frame_id = header.latest_frame_id;
                upload_slot.pose_metadata = upload_pose_metadata;
                upload_slot.overlay_commands[0] = upload_overlay_commands_left;
                upload_slot.overlay_commands[1] = upload_overlay_commands_right;
                upload_slot.modal_overlay = upload_modal_payload.data;
                {
                    std::lock_guard<std::mutex> state_lock(async_upload_state_mutex);
                    async_ready_upload_slot = upload_slot_index;
                    async_ready_presentation_mode = header_presentation_mode;
                }
                async_next_upload_slot =
                    (upload_slot_index + 1) %
                    static_cast<int>(immersive_upload_slots.size());

                const uint64_t previous_frame_id = async_latest_frame_id;
                async_latest_frame_id = header.latest_frame_id;
                if (async_slot_pressure_frame_id == async_latest_frame_id) {
                    async_slot_pressure_frame_id = 0;
                }
                const uint64_t source_frame_delta =
                    (header.latest_frame_id > previous_frame_id)
                        ? (header.latest_frame_id - previous_frame_id)
                        : 1;
                const auto source_update_time = std::chrono::steady_clock::now();
                ++async_applied_source_update_count;
                async_source_frame_delta_count += source_frame_delta;
                if (source_frame_delta > 0) {
                    async_coalesced_source_frame_count += source_frame_delta - 1;
                }
                if (async_applied_source_update_count == 1) {
                    async_first_source_update_time = source_update_time;
                    async_last_source_update_log_time = source_update_time;
                    async_logged_applied_source_update_count = 0;
                    async_logged_source_frame_delta_count = 0;
                }
                const double elapsed_s =
                    std::chrono::duration<double>(
                        source_update_time - async_first_source_update_time).count();
                const double since_last_log_s =
                    std::chrono::duration<double>(
                        source_update_time - async_last_source_update_log_time).count();
                if (async_applied_source_update_count == 1 || since_last_log_s >= 1.0) {
                    const uint64_t applied_updates_since_last_log =
                        async_applied_source_update_count -
                        async_logged_applied_source_update_count;
                    const uint64_t source_frame_delta_since_last_log =
                        async_source_frame_delta_count -
                        async_logged_source_frame_delta_count;
                    const double update_recent_fps =
                        (since_last_log_s > 0.0)
                            ? (static_cast<double>(applied_updates_since_last_log) /
                               since_last_log_s)
                            : 0.0;
                    const double source_delta_recent_fps =
                        (since_last_log_s > 0.0)
                            ? (static_cast<double>(source_frame_delta_since_last_log) /
                               since_last_log_s)
                            : 0.0;
                    std::cerr << std::fixed << std::setprecision(2)
                              << "Immersive bridge viewer_source_stats "
                              << "latest_frame_id=" << async_latest_frame_id << " "
                              << "update_count=" << async_applied_source_update_count << " "
                              << "source_frame_delta_count="
                              << async_source_frame_delta_count << " "
                              << "coalesced_frame_count="
                              << async_coalesced_source_frame_count << " "
                              << "elapsed_s=" << elapsed_s << " "
                              << "update_recent_fps=" << update_recent_fps << " "
                              << "source_delta_recent_fps="
                              << source_delta_recent_fps << "\n";
                    async_last_source_update_log_time = source_update_time;
                    async_logged_applied_source_update_count =
                        async_applied_source_update_count;
                    async_logged_source_frame_delta_count =
                        async_source_frame_delta_count;
                }

                {
                    std::lock_guard<std::mutex> stats_lock(viewer_upload_stats_mutex);
                    texture_upload_ms_sum +=
                        std::chrono::duration<double, std::milli>(
                            upload_end - upload_start).count();
                    texture_upload_mmap_copy_ms_sum += frame_mmap_copy_ms;
                    texture_upload_gl_left_ms_sum += frame_gl_upload_left_ms;
                    texture_upload_gl_right_ms_sum += frame_gl_upload_right_ms;
                    texture_upload_gl_ms_sum += (
                        frame_gl_upload_left_ms + frame_gl_upload_right_ms
                    );
                    if (ImmersiveFramePoseMetadataStereoValid(upload_pose_metadata)) {
                        ++viewer_source_pose_metadata_valid_count;
                    } else {
                        ++viewer_source_pose_metadata_invalid_count;
                    }
                    if (overlay_latch_status == OverlayLatchReadStatus::Match) {
                        ++viewer_overlay_latched_match_count;
                    } else if (overlay_latch_status == OverlayLatchReadStatus::Empty) {
                        ++viewer_overlay_latched_empty_count;
                    } else if (overlay_latch_status == OverlayLatchReadStatus::Mismatch) {
                        ++viewer_overlay_latched_mismatch_count;
                    }
                    if (modal_latch_status == OverlayLatchReadStatus::Match) {
                        ++viewer_modal_latched_match_count;
                    } else if (modal_latch_status == OverlayLatchReadStatus::Empty) {
                        ++viewer_modal_latched_empty_count;
                    } else if (modal_latch_status == OverlayLatchReadStatus::Mismatch) {
                        ++viewer_modal_latched_mismatch_count;
                    }
                    ++texture_upload_count;
                    ++viewer_async_upload_count;
                    ++viewer_async_ready_slot_count;
                }
            }

            glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
            glBindTexture(GL_TEXTURE_2D, 0);
            glfwMakeContextCurrent(nullptr);
        });
    }
#endif

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
        SelectStateSample anchor_reset_left;
        SelectStateSample anchor_reset_right;
        ThumbstickStateSample thumbstick_left;
        ThumbstickStateSample thumbstick_right;
        SelectStateSample snap_assist_left;
        SelectStateSample snap_assist_right;
        SelectStateSample exit_left;
        SelectStateSample exit_right;
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
            !QueryBooleanActionState(instance, session, anchor_reset_click_action,
                                     left_hand_path, &anchor_reset_left) ||
            !QueryBooleanActionState(instance, session, anchor_reset_click_action,
                                     right_hand_path, &anchor_reset_right) ||
            !QueryThumbstickState(instance, session, thumbstick_axis_action,
                                  left_hand_path, &thumbstick_left) ||
            !QueryThumbstickState(instance, session, thumbstick_axis_action,
                                  right_hand_path, &thumbstick_right) ||
            !QueryBooleanActionState(instance, session, snap_assist_click_action,
                                     left_hand_path, &snap_assist_left) ||
            !QueryBooleanActionState(instance, session, snap_assist_click_action,
                                     right_hand_path, &snap_assist_right) ||
            !QueryExitValueState(instance, session, exit_value_action, left_hand_path,
                                 &exit_left) ||
            !QueryExitValueState(instance, session, exit_value_action, right_hand_path,
                                 &exit_right)) {
            break;
        }

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
        const bool eye_pose_valid =
            (view_state.viewStateFlags & XR_VIEW_STATE_POSITION_VALID_BIT) != 0 &&
            (view_state.viewStateFlags & XR_VIEW_STATE_ORIENTATION_VALID_BIT) != 0;
        const bool eye_pose_tracked =
            (view_state.viewStateFlags & XR_VIEW_STATE_POSITION_TRACKED_BIT) != 0 &&
            (view_state.viewStateFlags & XR_VIEW_STATE_ORIENTATION_TRACKED_BIT) != 0;

        std::cout << "{";
        std::cout << "\"sample\":" << controller_sample_count << ",";
        PrintControllerJson(
            "left",
            selected_left,
            grip_left,
            aim_left,
            select_left,
            anchor_cycle_left,
            anchor_reset_left,
            thumbstick_left,
            snap_assist_left,
            exit_left
        );
        std::cout << ",";
        PrintControllerJson(
            "right",
            selected_right,
            grip_right,
            aim_right,
            select_right,
            anchor_cycle_right,
            anchor_reset_right,
            thumbstick_right,
            snap_assist_right,
            exit_right
        );
#ifdef BOBA_IMMERSIVE_BRIDGE
        std::cout << ",";
        PrintEyeJson(
            "left_eye",
            views[0],
            swapchain_views[0],
            eye_pose_valid && view_count_output > 0,
            eye_pose_tracked && view_count_output > 0
        );
        std::cout << ",";
        PrintEyeJson(
            "right_eye",
            views[std::min<uint32_t>(1u, static_cast<uint32_t>(views.size() - 1))],
            swapchain_views[std::min<size_t>(1u, swapchain_views.size() - 1)],
            eye_pose_valid && view_count_output > 1,
            eye_pose_tracked && view_count_output > 1
        );
#endif
        std::cout << "}\n";
        ++controller_sample_count;

        std::vector<const XrCompositionLayerBaseHeader*> layers;
        XrCompositionLayerProjection projection_layer =
            MakeXrStruct<XrCompositionLayerProjection>(XR_TYPE_COMPOSITION_LAYER_PROJECTION);
#ifdef BOBA_IMMERSIVE_BRIDGE
        XrCompositionLayerQuad modal_quad_layer =
            MakeXrStruct<XrCompositionLayerQuad>(XR_TYPE_COMPOSITION_LAYER_QUAD);
#endif

        if (frame_state.shouldRender == XR_TRUE) {
            const auto now = std::chrono::steady_clock::now();
            ++rendered_frame_count;
            if (rendered_frame_count == 1) {
                first_render_frame_time = now;
                last_render_log_time = now;
                logged_rendered_frame_count = 0;
            }
            const double render_elapsed_s =
                std::chrono::duration<double>(now - first_render_frame_time).count();
            const double render_since_last_log_s =
                std::chrono::duration<double>(now - last_render_log_time).count();
            if (rendered_frame_count == 1 || render_since_last_log_s >= 1.0) {
#ifdef BOBA_IMMERSIVE_BRIDGE
                std::lock_guard<std::mutex> upload_stats_lock(viewer_upload_stats_mutex);
#endif
                const uint64_t rendered_since_last_log =
                    rendered_frame_count - logged_rendered_frame_count;
                const uint64_t texture_uploads_since_last_log =
                    texture_upload_count - logged_texture_upload_count;
                const uint64_t texture_upload_slot_misses_since_last_log =
                    texture_upload_slot_miss_count - logged_texture_upload_slot_miss_count;
                const uint64_t texture_upload_slot_drops_since_last_log =
                    texture_upload_slot_drop_count - logged_texture_upload_slot_drop_count;
                const uint64_t texture_upload_busy_slots_since_last_log =
                    texture_upload_slot_busy_count - logged_texture_upload_slot_busy_count;
                const uint64_t texture_upload_busy_backoffs_since_last_log =
                    texture_upload_busy_backoff_count -
                    logged_texture_upload_busy_backoff_count;
                const uint64_t render_without_upload_since_last_log =
                    render_without_upload_count - logged_render_without_upload_count;
                const uint64_t texture_upload_no_new_since_last_log =
                    texture_upload_no_new_frame_count -
                    logged_texture_upload_no_new_frame_count;
                const double texture_upload_ms_since_last_log =
                    texture_upload_ms_sum - logged_texture_upload_ms_sum;
                const double texture_upload_mmap_copy_ms_since_last_log =
                    texture_upload_mmap_copy_ms_sum -
                    logged_texture_upload_mmap_copy_ms_sum;
                const double texture_upload_gl_ms_since_last_log =
                    texture_upload_gl_ms_sum - logged_texture_upload_gl_ms_sum;
                const double texture_upload_gl_left_ms_since_last_log =
                    texture_upload_gl_left_ms_sum -
                    logged_texture_upload_gl_left_ms_sum;
                const double texture_upload_gl_right_ms_since_last_log =
                    texture_upload_gl_right_ms_sum -
                    logged_texture_upload_gl_right_ms_sum;
                const double texture_upload_busy_backoff_ms_since_last_log =
                    texture_upload_busy_backoff_ms_sum -
                    logged_texture_upload_busy_backoff_ms_sum;
                const double render_recent_fps =
                    (render_since_last_log_s > 0.0)
                        ? (static_cast<double>(rendered_since_last_log) / render_since_last_log_s)
                        : 0.0;
                const double texture_upload_recent_fps =
                    (render_since_last_log_s > 0.0)
                        ? (static_cast<double>(texture_uploads_since_last_log) /
                           render_since_last_log_s)
                        : 0.0;
                const double texture_upload_avg_ms =
                    (texture_uploads_since_last_log > 0)
                        ? (texture_upload_ms_since_last_log /
                           static_cast<double>(texture_uploads_since_last_log))
                        : 0.0;
                const double texture_upload_mmap_copy_avg_ms =
                    (texture_uploads_since_last_log > 0)
                        ? (texture_upload_mmap_copy_ms_since_last_log /
                           static_cast<double>(texture_uploads_since_last_log))
                        : 0.0;
                const double texture_upload_gl_avg_ms =
                    (texture_uploads_since_last_log > 0)
                        ? (texture_upload_gl_ms_since_last_log /
                           static_cast<double>(texture_uploads_since_last_log))
                        : 0.0;
                const double texture_upload_gl_left_avg_ms =
                    (texture_uploads_since_last_log > 0)
                        ? (texture_upload_gl_left_ms_since_last_log /
                           static_cast<double>(texture_uploads_since_last_log))
                        : 0.0;
                const double texture_upload_gl_right_avg_ms =
                    (texture_uploads_since_last_log > 0)
                        ? (texture_upload_gl_right_ms_since_last_log /
                           static_cast<double>(texture_uploads_since_last_log))
                        : 0.0;
                const double texture_upload_busy_backoff_avg_ms =
                    (texture_upload_busy_backoffs_since_last_log > 0)
                        ? (texture_upload_busy_backoff_ms_since_last_log /
                           static_cast<double>(
                               texture_upload_busy_backoffs_since_last_log))
                        : 0.0;
                std::cerr << std::fixed << std::setprecision(2)
                          << "Immersive bridge viewer_render_stats "
                          << "rendered_count=" << rendered_frame_count << " "
                          << "elapsed_s=" << render_elapsed_s << " "
                          << "recent_fps=" << render_recent_fps << " "
                          << "texture_upload_count=" << texture_upload_count << " "
                          << "texture_upload_recent_fps=" << texture_upload_recent_fps << " "
                          << "texture_upload_avg_ms=" << texture_upload_avg_ms << " "
#ifdef BOBA_IMMERSIVE_BRIDGE
                          << "texture_upload_mode="
                          << ImmersiveViewerUploadModeLabel(viewer_upload_mode) << " "
                          << "viewer_upload_thread_mode="
                          << ImmersiveViewerUploadThreadModeLabel(viewer_upload_thread_mode)
                          << " "
                          << "viewer_upload_thread_fallback_reason="
                          << viewer_upload_thread_fallback_reason << " "
                          << "viewer_upload_ring_slots="
                          << viewer_upload_ring_slots << " "
                          << "viewer_upload_busy_backoff_us="
                          << viewer_upload_busy_backoff_us << " "
#else
                          << "texture_upload_mode=legacy "
                          << "viewer_upload_thread_mode=render "
                          << "viewer_upload_thread_fallback_reason=none "
                          << "viewer_upload_ring_slots=0 "
                          << "viewer_upload_busy_backoff_us=0 "
#endif
                          << "texture_upload_mmap_copy_avg_ms="
                          << texture_upload_mmap_copy_avg_ms << " "
                          << "texture_upload_gl_avg_ms=" << texture_upload_gl_avg_ms << " "
                          << "texture_upload_gl_left_avg_ms="
                          << texture_upload_gl_left_avg_ms << " "
                          << "texture_upload_gl_right_avg_ms="
                          << texture_upload_gl_right_avg_ms << " "
                          << "texture_upload_slot_miss_count="
                          << texture_upload_slot_miss_count << " "
                          << "texture_upload_slot_miss_recent_count="
                          << texture_upload_slot_misses_since_last_log << " "
                          << "texture_upload_slot_drop_count="
                          << texture_upload_slot_drop_count << " "
                          << "texture_upload_slot_drop_recent_count="
                          << texture_upload_slot_drops_since_last_log << " "
                          << "texture_upload_slot_busy_count="
                          << texture_upload_slot_busy_count << " "
                          << "texture_upload_slot_busy_recent_count="
                          << texture_upload_busy_slots_since_last_log << " "
                          << "texture_upload_busy_backoff_count="
                          << texture_upload_busy_backoff_count << " "
                          << "texture_upload_busy_backoff_recent_count="
                          << texture_upload_busy_backoffs_since_last_log << " "
                          << "texture_upload_busy_backoff_avg_ms="
                          << texture_upload_busy_backoff_avg_ms << " "
                          << "render_without_upload_count="
                          << render_without_upload_count << " "
                          << "render_without_upload_recent_count="
                          << render_without_upload_since_last_log << " "
                          << "texture_upload_no_new_frame_count="
                          << texture_upload_no_new_frame_count << " "
                          << "texture_upload_no_new_frame_recent_count="
                          << texture_upload_no_new_since_last_log << " "
#ifdef BOBA_IMMERSIVE_BRIDGE
                          << "viewer_async_upload_count="
                          << viewer_async_upload_count << " "
                          << "viewer_async_ready_slot_count="
                          << viewer_async_ready_slot_count << " "
                          << "viewer_async_poll_no_new_count="
                          << viewer_async_poll_no_new_count << " "
                          << "viewer_projection_pose_mode="
                          << viewer_projection_pose_mode << " "
                          << "viewer_source_pose_metadata_valid_count="
                          << viewer_source_pose_metadata_valid_count << " "
                          << "viewer_source_pose_metadata_invalid_count="
                          << viewer_source_pose_metadata_invalid_count << " "
                          << "viewer_source_pose_metadata_fallback_count="
                          << viewer_source_pose_metadata_fallback_count << " "
                          << "viewer_overlay_latched_match_count="
                          << viewer_overlay_latched_match_count << " "
                          << "viewer_overlay_latched_mismatch_count="
                          << viewer_overlay_latched_mismatch_count << " "
                          << "viewer_overlay_latched_empty_count="
                          << viewer_overlay_latched_empty_count << " "
                          << "viewer_modal_latched_match_count="
                          << viewer_modal_latched_match_count << " "
                          << "viewer_modal_latched_mismatch_count="
                          << viewer_modal_latched_mismatch_count << " "
                          << "viewer_modal_latched_empty_count="
                          << viewer_modal_latched_empty_count << " "
                          << "viewer_modal_layer_present_count="
                          << viewer_modal_layer_present_count << " "
                          << "viewer_modal_layer_mode="
                          << viewer_modal_layer_mode
#else
                          << "viewer_async_upload_count=0 "
                          << "viewer_async_ready_slot_count=0 "
                          << "viewer_async_poll_no_new_count=0 "
                          << "viewer_projection_pose_mode=current_view_fallback "
                          << "viewer_source_pose_metadata_valid_count=0 "
                          << "viewer_source_pose_metadata_invalid_count=0 "
                          << "viewer_source_pose_metadata_fallback_count=0 "
                          << "viewer_overlay_latched_match_count=0 "
                          << "viewer_overlay_latched_mismatch_count=0 "
                          << "viewer_overlay_latched_empty_count=0 "
                          << "viewer_modal_latched_match_count=0 "
                          << "viewer_modal_latched_mismatch_count=0 "
                          << "viewer_modal_latched_empty_count=0 "
                          << "viewer_modal_layer_present_count=0 "
                          << "viewer_modal_layer_mode=disabled"
#endif
                          << "\n";
                last_render_log_time = now;
                logged_rendered_frame_count = rendered_frame_count;
                logged_texture_upload_count = texture_upload_count;
                logged_texture_upload_ms_sum = texture_upload_ms_sum;
                logged_texture_upload_mmap_copy_ms_sum = texture_upload_mmap_copy_ms_sum;
                logged_texture_upload_gl_ms_sum = texture_upload_gl_ms_sum;
                logged_texture_upload_gl_left_ms_sum = texture_upload_gl_left_ms_sum;
                logged_texture_upload_gl_right_ms_sum = texture_upload_gl_right_ms_sum;
                logged_texture_upload_slot_miss_count = texture_upload_slot_miss_count;
                logged_texture_upload_slot_drop_count = texture_upload_slot_drop_count;
                logged_texture_upload_slot_busy_count = texture_upload_slot_busy_count;
                logged_texture_upload_busy_backoff_count =
                    texture_upload_busy_backoff_count;
                logged_texture_upload_busy_backoff_ms_sum =
                    texture_upload_busy_backoff_ms_sum;
                logged_render_without_upload_count = render_without_upload_count;
                logged_texture_upload_no_new_frame_count =
                    texture_upload_no_new_frame_count;
            }
#ifdef BOBA_IMMERSIVE_BRIDGE
            previous_presentation_mode = current_presentation_mode;
            bool upload_attempted_this_render_frame = false;
            bool uploaded_this_render_frame = false;
            bool source_projection_pose_used_this_render_frame = false;
#else
            if (UpdateDisplayFrameIfNeeded(shared_frame, &latest_frame_id, &display_rgba)) {
                if (latest_frame_id == 1 || latest_frame_id >= logged_source_frame_id + 120) {
                    std::cerr << "Panel received source frame " << latest_frame_id << "\n";
                    logged_source_frame_id = latest_frame_id;
                }

                const auto upload_start = std::chrono::steady_clock::now();
                glBindTexture(GL_TEXTURE_2D, source_textures[0]);
                glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
                glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, shared_frame.header->width,
                                shared_frame.header->height, GL_RGBA, GL_UNSIGNED_BYTE,
                                display_rgba.data());
                glBindTexture(GL_TEXTURE_2D, 0);
                const auto upload_end = std::chrono::steady_clock::now();
                texture_upload_ms_sum +=
                    std::chrono::duration<double, std::milli>(upload_end - upload_start).count();
                ++texture_upload_count;
            }
#endif

#ifdef BOBA_IMMERSIVE_BRIDGE
            bool render_as_world_locked_panel =
                current_presentation_mode == kPresentationModeMonoPanel;
            bool render_as_head_locked_panel =
                current_presentation_mode == kPresentationModeHeadLockedPanel;
            bool render_as_panel =
                render_as_world_locked_panel || render_as_head_locked_panel;
#else
            const bool render_as_world_locked_panel = true;
            const bool render_as_head_locked_panel = false;
            const bool render_as_panel = true;
            if (render_as_world_locked_panel && !panel_anchor_initialized && view_count_output > 0) {
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
#endif

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

#ifdef BOBA_IMMERSIVE_BRIDGE
                if (view_index == 0 && !upload_attempted_this_render_frame) {
                    upload_attempted_this_render_frame = true;
                    if (viewer_upload_thread_mode == ImmersiveViewerUploadThreadMode::Async) {
                        int adopted_upload_slot = -1;
                        uint32_t adopted_presentation_mode =
                            current_presentation_mode;
                        {
                            std::lock_guard<std::mutex> state_lock(
                                async_upload_state_mutex);
                            if (async_ready_upload_slot >= 0) {
                                adopted_upload_slot = async_ready_upload_slot;
                                async_recently_rendered_slot =
                                    active_immersive_upload_slot;
                                adopted_presentation_mode =
                                    async_ready_presentation_mode;
                                async_ready_upload_slot = -1;
                            }
                        }
                        if (adopted_upload_slot >= 0) {
                            active_immersive_upload_slot = adopted_upload_slot;
                            current_presentation_mode = adopted_presentation_mode;
                            uploaded_this_render_frame = true;
                        } else {
                            uploaded_this_render_frame = false;
                        }
                    } else {
                        uploaded_this_render_frame = try_upload_latest_stereo_frame();
                    }
                    if (!uploaded_this_render_frame) {
                        ++render_without_upload_count;
                    }
                    render_as_world_locked_panel =
                        current_presentation_mode == kPresentationModeMonoPanel;
                    render_as_head_locked_panel =
                        current_presentation_mode == kPresentationModeHeadLockedPanel;
                    render_as_panel =
                        render_as_world_locked_panel || render_as_head_locked_panel;
                    if (
                        render_as_world_locked_panel &&
                        previous_presentation_mode != kPresentationModeMonoPanel
                    ) {
                        panel_anchor_initialized = false;
                    }
                    if (
                        render_as_world_locked_panel &&
                        !panel_anchor_initialized &&
                        view_count_output > 0
                    ) {
                        const Mat4 initial_head_pose = PoseMatrix(views[0].pose);
                        panel_model = Multiply(
                            initial_head_pose,
                            Multiply(
                                TranslationMatrix(0.0f,
                                                  kPanelYOffsetMeters,
                                                  -kPanelDistanceMeters),
                                ScaleMatrix(kPanelWidthMeters,
                                            panel_height_meters,
                                            1.0f)));
                        panel_anchor_initialized = true;
                        std::cerr
                            << "Presentation path: LOCAL-space world-locked "
                            << "textured panel anchored at initial head pose, "
                            << "z-offset=" << -kPanelDistanceMeters
                            << " width=" << kPanelWidthMeters
                            << " height=" << panel_height_meters << "\n";
                    }
                }
                if (!render_as_panel && view_index < 2) {
                    const ImmersiveFramePoseMetadata* active_pose_metadata =
                        &active_source_pose_metadata;
                    if (
                        viewer_upload_mode == ImmersiveViewerUploadMode::Pbo &&
                        active_immersive_upload_slot >= 0 &&
                        static_cast<size_t>(active_immersive_upload_slot) <
                            immersive_upload_slots.size()
                    ) {
                        const ImmersiveUploadSlot& active_upload_slot =
                            immersive_upload_slots[
                                static_cast<size_t>(active_immersive_upload_slot)];
                        active_pose_metadata = &active_upload_slot.pose_metadata;
                    }
                    if (
                        active_pose_metadata != nullptr &&
                        ImmersiveFramePoseMetadataStereoValid(*active_pose_metadata)
                    ) {
                        const uint32_t eye_index = std::min<uint32_t>(view_index, 1u);
                        projection_view.pose = active_pose_metadata->pose[eye_index];
                        projection_view.fov = active_pose_metadata->fov[eye_index];
                        source_projection_pose_used_this_render_frame = true;
                    }
                }
#endif

                glBindFramebuffer(GL_FRAMEBUFFER, framebuffer);
                glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D,
                                       swapchain_view.images[image_index].image, 0);
                if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
                    std::cerr << "Framebuffer incomplete for view " << view_index << "\n";
                    exit_requested = true;
                } else {
                    Mat4 mvp_matrix = IdentityMatrix();
                    if (render_as_panel) {
                        Mat4 panel_model_for_view = panel_model;
                        if (render_as_head_locked_panel) {
                            panel_model_for_view = Multiply(
                                PoseMatrix(views[view_index].pose),
                                Multiply(
                                    TranslationMatrix(0.0f,
                                                      kPanelYOffsetMeters,
                                                      -kPanelDistanceMeters),
                                    ScaleMatrix(kPanelWidthMeters,
                                                panel_height_meters,
                                                1.0f)));
                        }
                        const Mat4 view_matrix =
                            InverseRigidTransform(PoseMatrix(views[view_index].pose));
                        const Mat4 projection_matrix =
                            ProjectionMatrix(views[view_index].fov, kNearZ, kFarZ);
                        mvp_matrix =
                            Multiply(projection_matrix,
                                     Multiply(view_matrix, panel_model_for_view));
                    } else {
                        mvp_matrix = ScaleMatrix(2.0f, 2.0f, 1.0f);
                    }
                    glViewport(0, 0, static_cast<GLsizei>(swapchain_view.width),
                               static_cast<GLsizei>(swapchain_view.height));
                    glDisable(GL_DEPTH_TEST);
                    glDisable(GL_CULL_FACE);
                    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
                    glClear(GL_COLOR_BUFFER_BIT);
                    glUseProgram(program);
                    glBindVertexArray(vao);
                    glActiveTexture(GL_TEXTURE0);
                    GLuint source_texture_for_view =
                        render_as_panel
                            ? source_textures[0]
#ifdef BOBA_IMMERSIVE_BRIDGE
                            : source_textures[std::min<uint32_t>(view_index, 1u)]
#else
                            : source_textures[0]
#endif
                        ;
#ifdef BOBA_IMMERSIVE_BRIDGE
                    if (
                        viewer_upload_mode == ImmersiveViewerUploadMode::Pbo &&
                        active_immersive_upload_slot >= 0 &&
                        static_cast<size_t>(active_immersive_upload_slot) <
                            immersive_upload_slots.size()
                    ) {
                        const ImmersiveUploadSlot& active_upload_slot =
                            immersive_upload_slots[
                                static_cast<size_t>(active_immersive_upload_slot)];
                        const uint32_t eye_texture_index =
                            render_as_panel
                                ? 0u
                                : std::min<uint32_t>(view_index, 1u);
                        source_texture_for_view =
                            active_upload_slot.textures[eye_texture_index];
                    }
#endif
                    glBindTexture(GL_TEXTURE_2D, source_texture_for_view);
                    glUniform1i(source_location, 0);
                    glUniformMatrix4fv(mvp_location, 1, GL_FALSE, mvp_matrix.m);
                    glDrawArrays(GL_TRIANGLES, 0, 6);
                    if (!render_as_panel && overlay_program != 0) {
#ifdef BOBA_IMMERSIVE_BRIDGE
                        const uint32_t overlay_eye_index =
                            std::min<uint32_t>(view_index, 1u);
                        const std::vector<float>* overlay_commands =
                            (overlay_eye_index == 0u)
                                ? &overlay_commands_left
                                : &overlay_commands_right;
                        if (
                            viewer_upload_mode == ImmersiveViewerUploadMode::Pbo &&
                            active_immersive_upload_slot >= 0 &&
                            static_cast<size_t>(active_immersive_upload_slot) <
                                immersive_upload_slots.size()
                        ) {
                            const ImmersiveUploadSlot& active_upload_slot =
                                immersive_upload_slots[
                                    static_cast<size_t>(active_immersive_upload_slot)];
                            overlay_commands =
                                &active_upload_slot.overlay_commands[overlay_eye_index];
                        }
                        DrawOverlayCommands(*overlay_commands,
                                            overlay_program,
                                            overlay_vao,
                                            overlay_vbo,
                                            overlay_source_size_location,
                                            shared_frame.header->width,
                                            shared_frame.header->height);
#endif
                    }
                    if (
                        !render_as_panel &&
                        modal_program != 0
#ifdef BOBA_IMMERSIVE_BRIDGE
                        &&
                        !modal_quad_layer_available
#endif
                    ) {
#ifdef BOBA_IMMERSIVE_BRIDGE
                        const uint32_t modal_eye_index =
                            std::min<uint32_t>(view_index, 1u);
                        const ModalOverlayData* modal_overlay = &active_modal_overlay;
                        GLuint modal_texture_for_view = active_modal_texture;
                        if (
                            viewer_upload_mode == ImmersiveViewerUploadMode::Pbo &&
                            active_immersive_upload_slot >= 0 &&
                            static_cast<size_t>(active_immersive_upload_slot) <
                                immersive_upload_slots.size()
                        ) {
                            const ImmersiveUploadSlot& active_upload_slot =
                                immersive_upload_slots[
                                    static_cast<size_t>(active_immersive_upload_slot)];
                            modal_overlay = &active_upload_slot.modal_overlay;
                            modal_texture_for_view = active_upload_slot.modal_texture;
                        }
                        DrawModalOverlay(*modal_overlay,
                                         modal_texture_for_view,
                                         modal_program,
                                         modal_vao,
                                         modal_vbo,
                                         modal_source_size_location,
                                         modal_texture_location,
                                         modal_eye_index,
                                         shared_frame.header->width,
                                         shared_frame.header->height);
#endif
                    }
                    glBindTexture(GL_TEXTURE_2D, 0);
                    glBindVertexArray(0);
                    glUseProgram(0);
                    glBindFramebuffer(GL_FRAMEBUFFER, 0);
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

#ifdef BOBA_IMMERSIVE_BRIDGE
            if (!upload_attempted_this_render_frame) {
                ++render_without_upload_count;
            }
            if (render_as_head_locked_panel) {
                viewer_projection_pose_mode = "head_locked_panel";
            } else if (!render_as_world_locked_panel) {
                if (source_projection_pose_used_this_render_frame) {
                    viewer_projection_pose_mode = "source_frame_pose";
                } else {
                    viewer_projection_pose_mode = "current_view_fallback";
                    ++viewer_source_pose_metadata_fallback_count;
                }
            } else {
                viewer_projection_pose_mode = "current_view_fallback";
            }
#endif

            if (!exit_requested) {
                layers.push_back(
                    reinterpret_cast<const XrCompositionLayerBaseHeader*>(&projection_layer));
            }

#ifdef BOBA_IMMERSIVE_BRIDGE
            if (
                !exit_requested &&
                modal_quad_layer_available &&
                modal_program != 0 &&
                view_count_output > 0
            ) {
                const ModalOverlayData* modal_overlay = &active_modal_overlay;
                GLuint modal_texture_for_layer = active_modal_texture;
                if (
                    viewer_upload_mode == ImmersiveViewerUploadMode::Pbo &&
                    active_immersive_upload_slot >= 0 &&
                    static_cast<size_t>(active_immersive_upload_slot) <
                        immersive_upload_slots.size()
                ) {
                    const ImmersiveUploadSlot& active_upload_slot =
                        immersive_upload_slots[
                            static_cast<size_t>(active_immersive_upload_slot)];
                    modal_overlay = &active_upload_slot.modal_overlay;
                    modal_texture_for_layer = active_upload_slot.modal_texture;
                }
                if (
                    modal_overlay != nullptr &&
                    modal_overlay->visible &&
                    modal_overlay->width > 0 &&
                    modal_overlay->height > 0 &&
                    modal_texture_for_layer != 0
                ) {
                    const uint32_t modal_width =
                        std::min(modal_overlay->width, modal_quad_swapchain.width);
                    const uint32_t modal_height =
                        std::min(modal_overlay->height, modal_quad_swapchain.height);
                    if (
                        modal_width > 0 &&
                        modal_height > 0 &&
                        RenderModalTextureToQuadSwapchain(
                            instance,
                            *modal_overlay,
                            modal_texture_for_layer,
                            &modal_quad_swapchain,
                            framebuffer,
                            modal_program,
                            modal_vao,
                            modal_vbo,
                            modal_source_size_location,
                            modal_texture_location)
                    ) {
                        const float fallback_width_m = 0.72f;
                        const float layer_width_m =
                            (std::isfinite(modal_overlay->width_m) &&
                             modal_overlay->width_m > 0.0f)
                                ? modal_overlay->width_m
                                : fallback_width_m;
                        const float fallback_height_m =
                            layer_width_m *
                            (static_cast<float>(modal_height) /
                             std::max(1.0f, static_cast<float>(modal_width)));
                        const float layer_height_m =
                            (std::isfinite(modal_overlay->height_m) &&
                             modal_overlay->height_m > 0.0f)
                                ? modal_overlay->height_m
                                : fallback_height_m;
                        modal_quad_layer.layerFlags =
                            XR_COMPOSITION_LAYER_BLEND_TEXTURE_SOURCE_ALPHA_BIT |
                            XR_COMPOSITION_LAYER_UNPREMULTIPLIED_ALPHA_BIT;
                        modal_quad_layer.space = local_space;
                        modal_quad_layer.eyeVisibility = XR_EYE_VISIBILITY_BOTH;
                        modal_quad_layer.subImage.swapchain =
                            modal_quad_swapchain.handle;
                        modal_quad_layer.subImage.imageRect.offset = {0, 0};
                        modal_quad_layer.subImage.imageRect.extent = {
                            static_cast<int32_t>(modal_width),
                            static_cast<int32_t>(modal_height),
                        };
                        modal_quad_layer.subImage.imageArrayIndex = 0;
                        modal_quad_layer.pose =
                            MakeHeadLockedModalPose(views, view_count_output);
                        modal_quad_layer.size = {layer_width_m, layer_height_m};
                        layers.push_back(
                            reinterpret_cast<const XrCompositionLayerBaseHeader*>(
                                &modal_quad_layer));
                        ++viewer_modal_layer_present_count;
                    }
                }
            }
#endif
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

#ifdef BOBA_IMMERSIVE_BRIDGE
    async_upload_stop_requested.store(true, std::memory_order_relaxed);
    if (async_upload_thread.joinable()) {
        async_upload_thread.join();
    }
    glfwMakeContextCurrent(window);
    if (async_upload_window != nullptr) {
        glfwDestroyWindow(async_upload_window);
        async_upload_window = nullptr;
    }
#endif

    if (session_running) {
        xrEndSession(session);
    }
    xrDestroySpace(local_space);
    xrDestroySpace(aim_right_space);
    xrDestroySpace(aim_left_space);
    xrDestroySpace(grip_right_space);
    xrDestroySpace(grip_left_space);
    glDeleteFramebuffers(1, &framebuffer);
    if (overlay_vbo != 0) {
        glDeleteBuffers(1, &overlay_vbo);
    }
    if (overlay_vao != 0) {
        glDeleteVertexArrays(1, &overlay_vao);
    }
    if (overlay_program != 0) {
        glDeleteProgram(overlay_program);
    }
    if (modal_vbo != 0) {
        glDeleteBuffers(1, &modal_vbo);
    }
    if (modal_vao != 0) {
        glDeleteVertexArrays(1, &modal_vao);
    }
    if (modal_program != 0) {
        glDeleteProgram(modal_program);
    }
    glDeleteVertexArrays(1, &vao);
    glDeleteProgram(program);
#ifdef BOBA_IMMERSIVE_BRIDGE
    DestroyImmersiveUploadSlots(&immersive_upload_slots);
    DestroySwapchainView(&modal_quad_swapchain);
    if (active_modal_texture != 0) {
        glDeleteTextures(1, &active_modal_texture);
    }
#endif
    glDeleteTextures(source_texture_count, source_textures);
    DestroyViewSwapchains(&swapchain_views);
    xrDestroyActionSet(action_set);
    xrDestroySession(session);
    xrDestroyInstance(instance);
    glfwDestroyWindow(window);
    glfwTerminate();
    CloseSharedModalFile(&shared_modal);
    CloseSharedOverlayFile(&shared_overlay);
    CloseSharedFrameFile(&shared_frame);
    return 0;
}
