#define XR_USE_TIMESPEC
#include <openxr/openxr.h>
#include <openxr/openxr_platform.h>

#include <chrono>
#include <cmath>
#include <ctime>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <thread>
#include <vector>

namespace {

constexpr int kDefaultSampleCount = 120;
constexpr int kReadyTimeoutSeconds = 10;

template <typename T>
T MakeXrStruct(XrStructureType type) {
    T value{};
    value.type = type;
    return value;
}

XrPosef IdentityPose() {
    XrPosef pose{};
    pose.orientation.w = 1.0f;
    return pose;
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

bool HasExtension(const std::vector<XrExtensionProperties>& extensions, const char* name) {
    for (const auto& extension : extensions) {
        if (std::strcmp(extension.extensionName, name) == 0) {
            return true;
        }
    }

    return false;
}

float QuaternionDot(const XrQuaternionf& a, const XrQuaternionf& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}

XrQuaternionf NormalizeQuaternion(XrQuaternionf quaternion) {
    const float norm = std::sqrt(QuaternionDot(quaternion, quaternion));
    if (norm <= 0.0f) {
        XrQuaternionf identity{};
        identity.w = 1.0f;
        return identity;
    }

    quaternion.x /= norm;
    quaternion.y /= norm;
    quaternion.z /= norm;
    quaternion.w /= norm;
    return quaternion;
}

XrPosef ApproximateHeadPoseFromViews(const std::vector<XrView>& views, uint32_t view_count) {
    XrPosef pose = views.front().pose;
    if (view_count < 2) {
        return pose;
    }

    pose.position.x = 0.5f * (views[0].pose.position.x + views[1].pose.position.x);
    pose.position.y = 0.5f * (views[0].pose.position.y + views[1].pose.position.y);
    pose.position.z = 0.5f * (views[0].pose.position.z + views[1].pose.position.z);

    XrQuaternionf first = views[0].pose.orientation;
    XrQuaternionf second = views[1].pose.orientation;
    if (QuaternionDot(first, second) < 0.0f) {
        second.x = -second.x;
        second.y = -second.y;
        second.z = -second.z;
        second.w = -second.w;
    }

    XrQuaternionf blended{};
    blended.x = first.x + second.x;
    blended.y = first.y + second.y;
    blended.z = first.z + second.z;
    blended.w = first.w + second.w;
    pose.orientation = NormalizeQuaternion(blended);
    return pose;
}

bool PumpEvents(
    XrInstance instance,
    XrSession session,
    XrViewConfigurationType view_configuration_type,
    XrSessionState* session_state,
    bool* session_running,
    bool* exit_requested) {
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

int ParseSampleCount(int argc, char** argv) {
    if (argc < 2) {
        return kDefaultSampleCount;
    }

    try {
        return std::max(1, std::stoi(argv[1]));
    } catch (const std::exception&) {
        std::cerr << "Invalid sample count '" << argv[1]
                  << "'. Usage: openxr_headset_pose_probe [sample_count]\n";
        return -1;
    }
}

}  // namespace

int main(int argc, char** argv) {
    const int sample_count = ParseSampleCount(argc, argv);
    if (sample_count < 0) {
        return 2;
    }

    uint32_t extension_count = 0;
    if (!CheckXr(
            XR_NULL_HANDLE,
            xrEnumerateInstanceExtensionProperties(nullptr, 0, &extension_count, nullptr),
            "xrEnumerateInstanceExtensionProperties(count)")) {
        return 1;
    }

    std::vector<XrExtensionProperties> extensions(extension_count);
    for (auto& extension : extensions) {
        extension = MakeXrStruct<XrExtensionProperties>(XR_TYPE_EXTENSION_PROPERTIES);
    }
    if (!CheckXr(
            XR_NULL_HANDLE,
            xrEnumerateInstanceExtensionProperties(
                nullptr, extension_count, &extension_count, extensions.data()),
            "xrEnumerateInstanceExtensionProperties(list)")) {
        return 1;
    }

    const bool has_headless = HasExtension(extensions, XR_MND_HEADLESS_EXTENSION_NAME);
    const bool has_convert_timespec =
        HasExtension(extensions, XR_KHR_CONVERT_TIMESPEC_TIME_EXTENSION_NAME);
    std::cout << "Runtime extension count: " << extension_count << "\n";
    std::cout << "XR_MND_headless available: " << (has_headless ? "yes" : "no") << "\n";
    std::cout << "XR_KHR_convert_timespec_time available: "
              << (has_convert_timespec ? "yes" : "no") << "\n";
    if (!has_headless) {
        std::cerr
            << "SteamVR does not advertise XR_MND_headless here, so this minimal probe "
               "cannot create a session without graphics.\n";
        return 3;
    }

    std::vector<const char*> enabled_extensions = {XR_MND_HEADLESS_EXTENSION_NAME};
    if (has_convert_timespec) {
        enabled_extensions.push_back(XR_KHR_CONVERT_TIMESPEC_TIME_EXTENSION_NAME);
    }

    XrInstanceCreateInfo instance_info =
        MakeXrStruct<XrInstanceCreateInfo>(XR_TYPE_INSTANCE_CREATE_INFO);
    std::strncpy(instance_info.applicationInfo.applicationName, "openxr_hmd_probe",
                 XR_MAX_APPLICATION_NAME_SIZE - 1);
    instance_info.applicationInfo.applicationVersion = 1;
    std::strncpy(instance_info.applicationInfo.engineName, "none", XR_MAX_ENGINE_NAME_SIZE - 1);
    instance_info.applicationInfo.engineVersion = 1;
    instance_info.applicationInfo.apiVersion = XR_CURRENT_API_VERSION;
    instance_info.enabledExtensionCount = static_cast<uint32_t>(enabled_extensions.size());
    instance_info.enabledExtensionNames = enabled_extensions.data();

    XrInstance instance = XR_NULL_HANDLE;
    if (!CheckXr(
            XR_NULL_HANDLE, xrCreateInstance(&instance_info, &instance), "xrCreateInstance")) {
        return 1;
    }

    XrInstanceProperties instance_properties =
        MakeXrStruct<XrInstanceProperties>(XR_TYPE_INSTANCE_PROPERTIES);
    if (!CheckXr(instance, xrGetInstanceProperties(instance, &instance_properties),
                 "xrGetInstanceProperties")) {
        xrDestroyInstance(instance);
        return 1;
    }

    std::cout << "Runtime name: " << instance_properties.runtimeName << "\n";
    std::cout << "Runtime version: " << XR_VERSION_MAJOR(instance_properties.runtimeVersion) << "."
              << XR_VERSION_MINOR(instance_properties.runtimeVersion) << "."
              << XR_VERSION_PATCH(instance_properties.runtimeVersion) << "\n";

    PFN_xrConvertTimespecTimeToTimeKHR convert_timespec_time = nullptr;
    if (has_convert_timespec) {
        const XrResult get_proc_result = xrGetInstanceProcAddr(
            instance, "xrConvertTimespecTimeToTimeKHR",
            reinterpret_cast<PFN_xrVoidFunction*>(&convert_timespec_time));
        if (!CheckXr(instance, get_proc_result, "xrGetInstanceProcAddr(xrConvertTimespecTimeToTimeKHR)")) {
            xrDestroyInstance(instance);
            return 1;
        }
    }

    XrSystemGetInfo system_info = MakeXrStruct<XrSystemGetInfo>(XR_TYPE_SYSTEM_GET_INFO);
    system_info.formFactor = XR_FORM_FACTOR_HEAD_MOUNTED_DISPLAY;

    XrSystemId system_id = XR_NULL_SYSTEM_ID;
    if (!CheckXr(instance, xrGetSystem(instance, &system_info, &system_id), "xrGetSystem")) {
        xrDestroyInstance(instance);
        return 1;
    }

    XrSystemProperties system_properties =
        MakeXrStruct<XrSystemProperties>(XR_TYPE_SYSTEM_PROPERTIES);
    if (!CheckXr(instance, xrGetSystemProperties(instance, system_id, &system_properties),
                 "xrGetSystemProperties")) {
        xrDestroyInstance(instance);
        return 1;
    }

    std::cout << "System name: " << system_properties.systemName << "\n";
    std::cout << "Orientation tracking: "
              << (system_properties.trackingProperties.orientationTracking ? "yes" : "no")
              << "\n";
    std::cout << "Position tracking: "
              << (system_properties.trackingProperties.positionTracking ? "yes" : "no") << "\n";

    const XrViewConfigurationType view_configuration_type =
        XR_VIEW_CONFIGURATION_TYPE_PRIMARY_STEREO;
    uint32_t blend_mode_count = 0;
    if (!CheckXr(instance,
                 xrEnumerateEnvironmentBlendModes(
                     instance, system_id, view_configuration_type, 0, &blend_mode_count, nullptr),
                 "xrEnumerateEnvironmentBlendModes(count)")) {
        xrDestroyInstance(instance);
        return 1;
    }

    std::vector<XrEnvironmentBlendMode> blend_modes(blend_mode_count);
    if (blend_mode_count > 0 &&
        !CheckXr(instance,
                 xrEnumerateEnvironmentBlendModes(instance, system_id, view_configuration_type,
                                                  blend_mode_count, &blend_mode_count,
                                                  blend_modes.data()),
                 "xrEnumerateEnvironmentBlendModes(list)")) {
        xrDestroyInstance(instance);
        return 1;
    }

    const XrEnvironmentBlendMode blend_mode =
        blend_modes.empty() ? XR_ENVIRONMENT_BLEND_MODE_OPAQUE : blend_modes.front();

    XrSessionCreateInfo session_info =
        MakeXrStruct<XrSessionCreateInfo>(XR_TYPE_SESSION_CREATE_INFO);
    session_info.systemId = system_id;

    XrSession session = XR_NULL_HANDLE;
    if (!CheckXr(instance, xrCreateSession(instance, &session_info, &session), "xrCreateSession")) {
        xrDestroyInstance(instance);
        return 1;
    }

    XrSessionState session_state = XR_SESSION_STATE_UNKNOWN;
    bool session_running = false;
    bool exit_requested = false;

    const auto ready_deadline =
        std::chrono::steady_clock::now() + std::chrono::seconds(kReadyTimeoutSeconds);
    while (!session_running && !exit_requested &&
           std::chrono::steady_clock::now() < ready_deadline) {
        if (!PumpEvents(instance, session, view_configuration_type, &session_state,
                        &session_running, &exit_requested)) {
            xrDestroySession(session);
            xrDestroyInstance(instance);
            return 1;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }

    if (!session_running) {
        std::cerr << "Session never reached READY/RUNNING state. Last state=" << session_state
                  << "\n";
        xrDestroySession(session);
        xrDestroyInstance(instance);
        return 4;
    }

    XrReferenceSpaceCreateInfo local_space_info =
        MakeXrStruct<XrReferenceSpaceCreateInfo>(XR_TYPE_REFERENCE_SPACE_CREATE_INFO);
    local_space_info.referenceSpaceType = XR_REFERENCE_SPACE_TYPE_LOCAL;
    local_space_info.poseInReferenceSpace = IdentityPose();

    XrSpace local_space = XR_NULL_HANDLE;
    if (!CheckXr(instance, xrCreateReferenceSpace(session, &local_space_info, &local_space),
                 "xrCreateReferenceSpace(local)")) {
        xrDestroySession(session);
        xrDestroyInstance(instance);
        return 1;
    }

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "sample,view_count,pos_valid,ori_valid,pos_tracked,ori_tracked,px,py,pz,qx,qy,qz,qw\n";

    uint32_t view_capacity = 0;
    if (!CheckXr(instance,
                 xrEnumerateViewConfigurationViews(instance, system_id, view_configuration_type, 0,
                                                   &view_capacity, nullptr),
                 "xrEnumerateViewConfigurationViews(count)")) {
        xrDestroySpace(local_space);
        xrDestroySession(session);
        xrDestroyInstance(instance);
        return 1;
    }

    std::vector<XrViewConfigurationView> view_configs(view_capacity);
    for (auto& view_config : view_configs) {
        view_config = MakeXrStruct<XrViewConfigurationView>(XR_TYPE_VIEW_CONFIGURATION_VIEW);
    }
    if (!CheckXr(instance,
                 xrEnumerateViewConfigurationViews(instance, system_id, view_configuration_type,
                                                   view_capacity, &view_capacity,
                                                   view_configs.data()),
                 "xrEnumerateViewConfigurationViews(list)")) {
        xrDestroySpace(local_space);
        xrDestroySession(session);
        xrDestroyInstance(instance);
        return 1;
    }

    std::vector<XrView> views(view_capacity);
    for (auto& view : views) {
        view = MakeXrStruct<XrView>(XR_TYPE_VIEW);
    }

    for (int sample = 0; sample < sample_count && !exit_requested; ++sample) {
        if (!PumpEvents(instance, session, view_configuration_type, &session_state,
                        &session_running, &exit_requested)) {
            xrDestroySpace(local_space);
            xrDestroySession(session);
            xrDestroyInstance(instance);
            return 1;
        }

        XrFrameWaitInfo wait_info = MakeXrStruct<XrFrameWaitInfo>(XR_TYPE_FRAME_WAIT_INFO);
        XrFrameState frame_state = MakeXrStruct<XrFrameState>(XR_TYPE_FRAME_STATE);
        if (!CheckXr(instance, xrWaitFrame(session, &wait_info, &frame_state), "xrWaitFrame")) {
            xrDestroySpace(local_space);
            xrDestroySession(session);
            xrDestroyInstance(instance);
            return 1;
        }

        XrFrameBeginInfo begin_info =
            MakeXrStruct<XrFrameBeginInfo>(XR_TYPE_FRAME_BEGIN_INFO);
        if (!CheckXr(instance, xrBeginFrame(session, &begin_info), "xrBeginFrame")) {
            xrDestroySpace(local_space);
            xrDestroySession(session);
            xrDestroyInstance(instance);
            return 1;
        }

        XrViewLocateInfo locate_info = MakeXrStruct<XrViewLocateInfo>(XR_TYPE_VIEW_LOCATE_INFO);
        locate_info.viewConfigurationType = view_configuration_type;
        locate_info.displayTime = frame_state.predictedDisplayTime;
        locate_info.space = local_space;

        XrViewState view_state = MakeXrStruct<XrViewState>(XR_TYPE_VIEW_STATE);
        uint32_t view_count = 0;
        XrResult locate_result =
            xrLocateViews(session, &locate_info, &view_state, view_capacity, &view_count,
                          views.data());
        if (locate_result == XR_ERROR_TIME_INVALID && convert_timespec_time != nullptr) {
            timespec now{};
            clock_gettime(CLOCK_MONOTONIC, &now);

            XrTime current_time = 0;
            if (!CheckXr(instance, convert_timespec_time(instance, &now, &current_time),
                         "xrConvertTimespecTimeToTimeKHR")) {
                xrDestroySpace(local_space);
                xrDestroySession(session);
                xrDestroyInstance(instance);
                return 1;
            }

            locate_info.displayTime = current_time;
            locate_result = xrLocateViews(session, &locate_info, &view_state, view_capacity,
                                          &view_count, views.data());
        }
        if (!CheckXr(instance, locate_result, "xrLocateViews")) {
            xrDestroySpace(local_space);
            xrDestroySession(session);
            xrDestroyInstance(instance);
            return 1;
        }

        const bool position_valid =
            (view_state.viewStateFlags & XR_VIEW_STATE_POSITION_VALID_BIT) != 0;
        const bool orientation_valid =
            (view_state.viewStateFlags & XR_VIEW_STATE_ORIENTATION_VALID_BIT) != 0;
        const bool position_tracked =
            (view_state.viewStateFlags & XR_VIEW_STATE_POSITION_TRACKED_BIT) != 0;
        const bool orientation_tracked =
            (view_state.viewStateFlags & XR_VIEW_STATE_ORIENTATION_TRACKED_BIT) != 0;

        const XrPosef head_pose =
            view_count > 0 ? ApproximateHeadPoseFromViews(views, view_count) : IdentityPose();

        const float px = position_valid ? head_pose.position.x
                                        : std::numeric_limits<float>::quiet_NaN();
        const float py = position_valid ? head_pose.position.y
                                        : std::numeric_limits<float>::quiet_NaN();
        const float pz = position_valid ? head_pose.position.z
                                        : std::numeric_limits<float>::quiet_NaN();
        const float qx = orientation_valid ? head_pose.orientation.x
                                           : std::numeric_limits<float>::quiet_NaN();
        const float qy = orientation_valid ? head_pose.orientation.y
                                           : std::numeric_limits<float>::quiet_NaN();
        const float qz = orientation_valid ? head_pose.orientation.z
                                           : std::numeric_limits<float>::quiet_NaN();
        const float qw = orientation_valid ? head_pose.orientation.w
                                           : std::numeric_limits<float>::quiet_NaN();

        std::cout << sample << "," << view_count << "," << (position_valid ? 1 : 0) << ","
                  << (orientation_valid ? 1 : 0) << "," << (position_tracked ? 1 : 0) << ","
                  << (orientation_tracked ? 1 : 0) << "," << px << "," << py << "," << pz << ","
                  << qx << "," << qy << "," << qz << "," << qw << "\n";

        XrFrameEndInfo end_info = MakeXrStruct<XrFrameEndInfo>(XR_TYPE_FRAME_END_INFO);
        end_info.displayTime = frame_state.predictedDisplayTime;
        end_info.environmentBlendMode = blend_mode;
        end_info.layerCount = 0;
        end_info.layers = nullptr;
        if (!CheckXr(instance, xrEndFrame(session, &end_info), "xrEndFrame")) {
            xrDestroySpace(local_space);
            xrDestroySession(session);
            xrDestroyInstance(instance);
            return 1;
        }
    }

    xrDestroySpace(local_space);
    xrDestroySession(session);
    xrDestroyInstance(instance);
    return exit_requested ? 5 : 0;
}
