#define XR_USE_TIMESPEC
#include <openxr/openxr.h>
#include <openxr/openxr_platform.h>

#include <chrono>
#include <csignal>
#include <cstring>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

namespace {

constexpr int kReadyTimeoutSeconds = 10;
volatile std::sig_atomic_t g_stop_requested = 0;

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

int ParseSampleCount(int argc, char** argv) {
    if (argc < 2) {
        return 0;
    }

    try {
        return std::max(0, std::stoi(argv[1]));
    } catch (const std::exception&) {
        std::cerr << "Invalid sample count '" << argv[1]
                  << "'. Usage: openxr_hand_joint_stream [sample_count]\n";
        return -1;
    }
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

struct HandFns {
    PFN_xrConvertTimespecTimeToTimeKHR convert_timespec_time = nullptr;
    PFN_xrCreateHandTrackerEXT create_hand_tracker = nullptr;
    PFN_xrDestroyHandTrackerEXT destroy_hand_tracker = nullptr;
    PFN_xrLocateHandJointsEXT locate_hand_joints = nullptr;
};

bool LoadHandFns(XrInstance instance, bool has_convert_timespec, HandFns* fns) {
    if (has_convert_timespec &&
        !CheckXr(instance,
                 xrGetInstanceProcAddr(
                     instance, "xrConvertTimespecTimeToTimeKHR",
                     reinterpret_cast<PFN_xrVoidFunction*>(&fns->convert_timespec_time)),
                 "xrGetInstanceProcAddr(xrConvertTimespecTimeToTimeKHR)")) {
        return false;
    }

    if (!CheckXr(instance,
                 xrGetInstanceProcAddr(
                     instance, "xrCreateHandTrackerEXT",
                     reinterpret_cast<PFN_xrVoidFunction*>(&fns->create_hand_tracker)),
                 "xrGetInstanceProcAddr(xrCreateHandTrackerEXT)")) {
        return false;
    }
    if (!CheckXr(instance,
                 xrGetInstanceProcAddr(
                     instance, "xrDestroyHandTrackerEXT",
                     reinterpret_cast<PFN_xrVoidFunction*>(&fns->destroy_hand_tracker)),
                 "xrGetInstanceProcAddr(xrDestroyHandTrackerEXT)")) {
        return false;
    }
    if (!CheckXr(instance,
                 xrGetInstanceProcAddr(
                     instance, "xrLocateHandJointsEXT",
                     reinterpret_cast<PFN_xrVoidFunction*>(&fns->locate_hand_joints)),
                 "xrGetInstanceProcAddr(xrLocateHandJointsEXT)")) {
        return false;
    }
    return true;
}

bool CreateHandTracker(
    XrInstance instance,
    XrSession session,
    const HandFns& fns,
    XrHandEXT hand,
    XrHandTrackerEXT* tracker) {
    XrHandTrackerCreateInfoEXT create_info =
        MakeXrStruct<XrHandTrackerCreateInfoEXT>(XR_TYPE_HAND_TRACKER_CREATE_INFO_EXT);
    create_info.hand = hand;
    create_info.handJointSet = XR_HAND_JOINT_SET_DEFAULT_EXT;
    return CheckXr(instance, fns.create_hand_tracker(session, &create_info, tracker),
                   "xrCreateHandTrackerEXT");
}

XrTime GetSampleTime(
    XrInstance instance,
    const HandFns& fns,
    XrTime fallback_time) {
    if (fns.convert_timespec_time == nullptr) {
        return fallback_time;
    }

    timespec now{};
    if (clock_gettime(CLOCK_MONOTONIC, &now) != 0) {
        return fallback_time;
    }

    XrTime current_time = fallback_time;
    if (XR_SUCCEEDED(fns.convert_timespec_time(instance, &now, &current_time))) {
        return current_time;
    }
    return fallback_time;
}

bool SampleHand(
    XrInstance instance,
    const HandFns& fns,
    XrHandTrackerEXT tracker,
    XrSpace base_space,
    XrTime sample_time,
    XrBool32* is_active,
    XrHandJointLocationEXT* joints) {
    XrHandJointLocationsEXT locations =
        MakeXrStruct<XrHandJointLocationsEXT>(XR_TYPE_HAND_JOINT_LOCATIONS_EXT);
    locations.jointCount = XR_HAND_JOINT_COUNT_EXT;
    locations.jointLocations = joints;

    XrHandJointsLocateInfoEXT locate_info =
        MakeXrStruct<XrHandJointsLocateInfoEXT>(XR_TYPE_HAND_JOINTS_LOCATE_INFO_EXT);
    locate_info.baseSpace = base_space;
    locate_info.time = sample_time;

    if (!CheckXr(instance, fns.locate_hand_joints(tracker, &locate_info, &locations),
                 "xrLocateHandJointsEXT")) {
        return false;
    }

    *is_active = locations.isActive;
    return true;
}

void PrintHandJson(const char* prefix, XrBool32 is_active, const XrHandJointLocationEXT* joints) {
    std::cout << "\"" << prefix << "_active\":" << (is_active == XR_TRUE ? 1 : 0) << ",";

    std::cout << "\"" << prefix << "_valid\":[";
    for (int i = 0; i < XR_HAND_JOINT_COUNT_EXT; ++i) {
        if (i > 0) {
            std::cout << ",";
        }
        const bool valid = (joints[i].locationFlags & XR_SPACE_LOCATION_POSITION_VALID_BIT) != 0;
        std::cout << (valid ? 1 : 0);
    }
    std::cout << "],";

    std::cout << "\"" << prefix << "_positions\":[";
    for (int i = 0; i < XR_HAND_JOINT_COUNT_EXT; ++i) {
        if (i > 0) {
            std::cout << ",";
        }
        const auto& position = joints[i].pose.position;
        std::cout << "[" << position.x << "," << position.y << "," << position.z << "]";
    }
    std::cout << "]";
}

void HandleSignal(int) {
    g_stop_requested = 1;
}

}  // namespace

int main(int argc, char** argv) {
    std::signal(SIGINT, HandleSignal);
    std::signal(SIGTERM, HandleSignal);
    std::cout << std::fixed << std::setprecision(6) << std::unitbuf;

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
    const bool has_hand_tracking = HasExtension(extensions, XR_EXT_HAND_TRACKING_EXTENSION_NAME);
    if (!has_headless) {
        std::cerr << "Runtime does not expose XR_MND_headless.\n";
        return 3;
    }
    if (!has_hand_tracking) {
        std::cerr << "Runtime does not expose XR_EXT_hand_tracking.\n";
        return 4;
    }

    std::vector<const char*> enabled_extensions = {
        XR_MND_HEADLESS_EXTENSION_NAME,
        XR_EXT_HAND_TRACKING_EXTENSION_NAME,
    };
    if (has_convert_timespec) {
        enabled_extensions.push_back(XR_KHR_CONVERT_TIMESPEC_TIME_EXTENSION_NAME);
    }

    XrInstanceCreateInfo instance_info =
        MakeXrStruct<XrInstanceCreateInfo>(XR_TYPE_INSTANCE_CREATE_INFO);
    std::strncpy(instance_info.applicationInfo.applicationName, "openxr_hand_joint_stream",
                 XR_MAX_APPLICATION_NAME_SIZE - 1);
    std::strncpy(instance_info.applicationInfo.engineName, "none", XR_MAX_ENGINE_NAME_SIZE - 1);
    instance_info.applicationInfo.applicationVersion = 1;
    instance_info.applicationInfo.engineVersion = 1;
    instance_info.applicationInfo.apiVersion = XR_CURRENT_API_VERSION;
    instance_info.enabledExtensionCount = static_cast<uint32_t>(enabled_extensions.size());
    instance_info.enabledExtensionNames = enabled_extensions.data();

    XrInstance instance = XR_NULL_HANDLE;
    if (!CheckXr(XR_NULL_HANDLE, xrCreateInstance(&instance_info, &instance), "xrCreateInstance")) {
        return 1;
    }

    XrSystemGetInfo system_info = MakeXrStruct<XrSystemGetInfo>(XR_TYPE_SYSTEM_GET_INFO);
    system_info.formFactor = XR_FORM_FACTOR_HEAD_MOUNTED_DISPLAY;

    XrSystemId system_id = XR_NULL_SYSTEM_ID;
    if (!CheckXr(instance, xrGetSystem(instance, &system_info, &system_id), "xrGetSystem")) {
        xrDestroyInstance(instance);
        return 1;
    }

    XrSessionCreateInfo session_info =
        MakeXrStruct<XrSessionCreateInfo>(XR_TYPE_SESSION_CREATE_INFO);
    session_info.systemId = system_id;

    XrSession session = XR_NULL_HANDLE;
    if (!CheckXr(instance, xrCreateSession(instance, &session_info, &session), "xrCreateSession")) {
        xrDestroyInstance(instance);
        return 1;
    }

    HandFns fns;
    if (!LoadHandFns(instance, has_convert_timespec, &fns)) {
        xrDestroySession(session);
        xrDestroyInstance(instance);
        return 1;
    }

    XrViewConfigurationType view_configuration_type =
        XR_VIEW_CONFIGURATION_TYPE_PRIMARY_STEREO;
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
        std::cerr << "Session never reached READY/RUNNING state.\n";
        xrDestroySession(session);
        xrDestroyInstance(instance);
        return 5;
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

    XrHandTrackerEXT left_tracker = XR_NULL_HANDLE;
    XrHandTrackerEXT right_tracker = XR_NULL_HANDLE;
    if (!CreateHandTracker(instance, session, fns, XR_HAND_LEFT_EXT, &left_tracker) ||
        !CreateHandTracker(instance, session, fns, XR_HAND_RIGHT_EXT, &right_tracker)) {
        if (left_tracker != XR_NULL_HANDLE) {
            fns.destroy_hand_tracker(left_tracker);
        }
        xrDestroySpace(local_space);
        xrDestroySession(session);
        xrDestroyInstance(instance);
        return 1;
    }

    XrHandJointLocationEXT left_joints[XR_HAND_JOINT_COUNT_EXT];
    XrHandJointLocationEXT right_joints[XR_HAND_JOINT_COUNT_EXT];
    int samples_emitted = 0;

    while (!exit_requested && !g_stop_requested &&
           (sample_count == 0 || samples_emitted < sample_count)) {
        if (!PumpEvents(instance, session, view_configuration_type, &session_state,
                        &session_running, &exit_requested)) {
            break;
        }

        XrFrameWaitInfo wait_info = MakeXrStruct<XrFrameWaitInfo>(XR_TYPE_FRAME_WAIT_INFO);
        XrFrameState frame_state = MakeXrStruct<XrFrameState>(XR_TYPE_FRAME_STATE);
        if (!CheckXr(instance, xrWaitFrame(session, &wait_info, &frame_state), "xrWaitFrame")) {
            break;
        }

        XrFrameBeginInfo begin_info =
            MakeXrStruct<XrFrameBeginInfo>(XR_TYPE_FRAME_BEGIN_INFO);
        if (!CheckXr(instance, xrBeginFrame(session, &begin_info), "xrBeginFrame")) {
            break;
        }

        const XrTime sample_time = GetSampleTime(instance, fns, frame_state.predictedDisplayTime);
        XrBool32 left_active = XR_FALSE;
        XrBool32 right_active = XR_FALSE;
        if (!SampleHand(instance, fns, left_tracker, local_space, sample_time, &left_active,
                        left_joints) ||
            !SampleHand(instance, fns, right_tracker, local_space, sample_time, &right_active,
                        right_joints)) {
            break;
        }

        std::cout << "{";
        std::cout << "\"sample\":" << samples_emitted << ",";
        PrintHandJson("left", left_active, left_joints);
        std::cout << ",";
        PrintHandJson("right", right_active, right_joints);
        std::cout << "}\n";

        XrFrameEndInfo end_info = MakeXrStruct<XrFrameEndInfo>(XR_TYPE_FRAME_END_INFO);
        end_info.displayTime = frame_state.predictedDisplayTime;
        end_info.environmentBlendMode = XR_ENVIRONMENT_BLEND_MODE_OPAQUE;
        end_info.layerCount = 0;
        end_info.layers = nullptr;
        if (!CheckXr(instance, xrEndFrame(session, &end_info), "xrEndFrame")) {
            break;
        }

        ++samples_emitted;
    }

    fns.destroy_hand_tracker(left_tracker);
    fns.destroy_hand_tracker(right_tracker);
    xrDestroySpace(local_space);
    xrDestroySession(session);
    xrDestroyInstance(instance);
    return 0;
}
