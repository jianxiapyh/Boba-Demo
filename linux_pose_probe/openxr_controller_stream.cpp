#define XR_USE_TIMESPEC
#include <openxr/openxr.h>
#include <openxr/openxr_platform.h>

#include <algorithm>
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
constexpr float kSelectPressedThreshold = 0.75f;
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
                  << "'. Usage: openxr_controller_stream [sample_count]\n";
        return -1;
    }
}

bool StringToPath(XrInstance instance, const char* path_string, XrPath* path) {
    return CheckXr(instance, xrStringToPath(instance, path_string, path), path_string);
}

std::string PathToString(XrInstance instance, XrPath path) {
    if (path == XR_NULL_PATH) {
        return "(null)";
    }

    uint32_t size = 0;
    if (!CheckXr(instance, xrPathToString(instance, path, 0, &size, nullptr),
                 "xrPathToString(count)")) {
        return "(error)";
    }

    std::string buffer(size, '\0');
    if (!CheckXr(instance, xrPathToString(instance, path, size, &size, buffer.data()),
                 "xrPathToString(value)")) {
        return "(error)";
    }
    if (!buffer.empty() && buffer.back() == '\0') {
        buffer.pop_back();
    }
    return buffer;
}

std::string GetLocalizedSourceName(XrInstance instance, XrSession session, XrPath source_path) {
    XrInputSourceLocalizedNameGetInfo get_info =
        MakeXrStruct<XrInputSourceLocalizedNameGetInfo>(
            XR_TYPE_INPUT_SOURCE_LOCALIZED_NAME_GET_INFO);
    get_info.sourcePath = source_path;
    get_info.whichComponents = XR_INPUT_SOURCE_LOCALIZED_NAME_USER_PATH_BIT |
                               XR_INPUT_SOURCE_LOCALIZED_NAME_INTERACTION_PROFILE_BIT |
                               XR_INPUT_SOURCE_LOCALIZED_NAME_COMPONENT_BIT;

    uint32_t size = 0;
    const XrResult count_result =
        xrGetInputSourceLocalizedName(session, &get_info, 0, &size, nullptr);
    if (count_result == XR_ERROR_PATH_UNSUPPORTED) {
        return "(localized name unsupported)";
    }
    if (!CheckXr(instance, count_result, "xrGetInputSourceLocalizedName(count)")) {
        return "(localized name error)";
    }

    std::string buffer(size, '\0');
    if (!CheckXr(instance,
                 xrGetInputSourceLocalizedName(session, &get_info, size, &size, buffer.data()),
                 "xrGetInputSourceLocalizedName(value)")) {
        return "(localized name error)";
    }
    if (!buffer.empty() && buffer.back() == '\0') {
        buffer.pop_back();
    }
    return buffer;
}

void PrintBoundSourcesForAction(XrInstance instance, XrSession session, XrAction action,
                                const char* label) {
    XrBoundSourcesForActionEnumerateInfo enumerate_info =
        MakeXrStruct<XrBoundSourcesForActionEnumerateInfo>(
            XR_TYPE_BOUND_SOURCES_FOR_ACTION_ENUMERATE_INFO);
    enumerate_info.action = action;

    uint32_t source_count = 0;
    if (!CheckXr(instance,
                 xrEnumerateBoundSourcesForAction(session, &enumerate_info, 0, &source_count,
                                                  nullptr),
                 "xrEnumerateBoundSourcesForAction(count)")) {
        return;
    }

    std::cerr << label << " bound sources:\n";
    if (source_count == 0) {
        std::cerr << "  (none)\n";
        return;
    }

    std::vector<XrPath> sources(source_count, XR_NULL_PATH);
    if (!CheckXr(instance,
                 xrEnumerateBoundSourcesForAction(session, &enumerate_info, source_count,
                                                  &source_count, sources.data()),
                 "xrEnumerateBoundSourcesForAction(list)")) {
        return;
    }

    for (XrPath source : sources) {
        std::cerr << "  " << PathToString(instance, source) << " -> "
                  << GetLocalizedSourceName(instance, session, source) << "\n";
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

XrTime GetSampleTime(
    XrInstance instance,
    PFN_xrConvertTimespecTimeToTimeKHR convert_timespec_time,
    XrTime fallback_time) {
    if (convert_timespec_time == nullptr) {
        return fallback_time;
    }

    timespec now{};
    if (clock_gettime(CLOCK_MONOTONIC, &now) != 0) {
        return fallback_time;
    }

    XrTime current_time = fallback_time;
    if (XR_SUCCEEDED(convert_timespec_time(instance, &now, &current_time))) {
        return current_time;
    }
    return fallback_time;
}

struct ControllerPoseSample {
    const char* source = "none";
    bool action_active = false;
    bool position_valid = false;
    bool orientation_valid = false;
    bool position_tracked = false;
    bool orientation_tracked = false;
    XrPosef pose = IdentityPose();
};

struct SelectStateSample {
    const char* source = "none";
    bool available = false;
    bool pressed = false;
    float value = 0.0f;
};

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
    if (!CheckXr(instance, xrGetActionStatePose(session, &get_info, &state), "xrGetActionStatePose")) {
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
    if (!has_headless) {
        std::cerr << "Runtime does not expose XR_MND_headless.\n";
        return 3;
    }

    std::vector<const char*> enabled_extensions = {XR_MND_HEADLESS_EXTENSION_NAME};
    if (has_convert_timespec) {
        enabled_extensions.push_back(XR_KHR_CONVERT_TIMESPEC_TIME_EXTENSION_NAME);
    }

    XrInstanceCreateInfo instance_info =
        MakeXrStruct<XrInstanceCreateInfo>(XR_TYPE_INSTANCE_CREATE_INFO);
    std::strncpy(instance_info.applicationInfo.applicationName, "openxr_controller_stream",
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

    PFN_xrConvertTimespecTimeToTimeKHR convert_timespec_time = nullptr;
    if (has_convert_timespec) {
        if (!CheckXr(instance,
                     xrGetInstanceProcAddr(
                         instance, "xrConvertTimespecTimeToTimeKHR",
                         reinterpret_cast<PFN_xrVoidFunction*>(&convert_timespec_time)),
                     "xrGetInstanceProcAddr(xrConvertTimespecTimeToTimeKHR)")) {
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

    XrActionSetCreateInfo action_set_info =
        MakeXrStruct<XrActionSetCreateInfo>(XR_TYPE_ACTION_SET_CREATE_INFO);
    std::strncpy(action_set_info.actionSetName, "controllerplay", XR_MAX_ACTION_SET_NAME_SIZE - 1);
    std::strncpy(action_set_info.localizedActionSetName, "Controller Gameplay",
                 XR_MAX_LOCALIZED_ACTION_SET_NAME_SIZE - 1);
    action_set_info.priority = 0;

    XrActionSet action_set = XR_NULL_HANDLE;
    if (!CheckXr(instance, xrCreateActionSet(instance, &action_set_info, &action_set),
                 "xrCreateActionSet")) {
        xrDestroyInstance(instance);
        return 1;
    }

    XrPath left_hand_path = XR_NULL_PATH;
    XrPath right_hand_path = XR_NULL_PATH;
    if (!StringToPath(instance, "/user/hand/left", &left_hand_path) ||
        !StringToPath(instance, "/user/hand/right", &right_hand_path)) {
        xrDestroyActionSet(action_set);
        xrDestroyInstance(instance);
        return 1;
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
        return CheckXr(instance, xrCreateAction(action_set, &action_info, action), "xrCreateAction");
    };

    XrAction grip_pose_action = XR_NULL_HANDLE;
    XrAction aim_pose_action = XR_NULL_HANDLE;
    XrAction select_click_action = XR_NULL_HANDLE;
    XrAction select_value_action = XR_NULL_HANDLE;
    XrAction anchor_cycle_click_action = XR_NULL_HANDLE;
    XrAction snap_assist_click_action = XR_NULL_HANDLE;
    if (!create_action("grip_pose", "Grip Pose", XR_ACTION_TYPE_POSE_INPUT, &grip_pose_action) ||
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
        xrDestroyInstance(instance);
        return 1;
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
            !AppendSelectBindings(instance, select_click_action, select_value_action, profile, &bindings) ||
            !AppendAnchorCycleBindings(instance, anchor_cycle_click_action, profile, &bindings) ||
            !AppendSnapAssistBindings(instance, snap_assist_click_action, profile, &bindings) ||
            !SuggestBindingsForProfile(instance, profile, bindings)) {
            xrDestroyActionSet(action_set);
            xrDestroyInstance(instance);
            return 1;
        }
    }

    XrSessionCreateInfo session_info =
        MakeXrStruct<XrSessionCreateInfo>(XR_TYPE_SESSION_CREATE_INFO);
    session_info.systemId = system_id;

    XrSession session = XR_NULL_HANDLE;
    if (!CheckXr(instance, xrCreateSession(instance, &session_info, &session), "xrCreateSession")) {
        xrDestroyActionSet(action_set);
        xrDestroyInstance(instance);
        return 1;
    }

    XrSessionActionSetsAttachInfo attach_info =
        MakeXrStruct<XrSessionActionSetsAttachInfo>(XR_TYPE_SESSION_ACTION_SETS_ATTACH_INFO);
    attach_info.countActionSets = 1;
    attach_info.actionSets = &action_set;
    if (!CheckXr(instance, xrAttachSessionActionSets(session, &attach_info),
                 "xrAttachSessionActionSets")) {
        xrDestroySession(session);
        xrDestroyActionSet(action_set);
        xrDestroyInstance(instance);
        return 1;
    }

    auto create_action_space = [&](XrAction action, XrPath subaction_path,
                                   XrSpace* space, const char* label) -> bool {
        XrActionSpaceCreateInfo space_info =
            MakeXrStruct<XrActionSpaceCreateInfo>(XR_TYPE_ACTION_SPACE_CREATE_INFO);
        space_info.action = action;
        space_info.subactionPath = subaction_path;
        space_info.poseInActionSpace = IdentityPose();
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
        xrDestroySession(session);
        xrDestroyActionSet(action_set);
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
            xrDestroySpace(aim_right_space);
            xrDestroySpace(aim_left_space);
            xrDestroySpace(grip_right_space);
            xrDestroySpace(grip_left_space);
            xrDestroySession(session);
            xrDestroyActionSet(action_set);
            xrDestroyInstance(instance);
            return 1;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
    if (!session_running) {
        std::cerr << "Session never reached READY/RUNNING state.\n";
        xrDestroySpace(aim_right_space);
        xrDestroySpace(aim_left_space);
        xrDestroySpace(grip_right_space);
        xrDestroySpace(grip_left_space);
        xrDestroySession(session);
        xrDestroyActionSet(action_set);
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
        xrDestroySpace(aim_right_space);
        xrDestroySpace(aim_left_space);
        xrDestroySpace(grip_right_space);
        xrDestroySpace(grip_left_space);
        xrDestroySession(session);
        xrDestroyActionSet(action_set);
        xrDestroyInstance(instance);
        return 1;
    }

    XrActiveActionSet active_action_set{action_set, XR_NULL_PATH};
    XrActionsSyncInfo sync_info = MakeXrStruct<XrActionsSyncInfo>(XR_TYPE_ACTIONS_SYNC_INFO);
    sync_info.countActiveActionSets = 1;
    sync_info.activeActionSets = &active_action_set;

    int samples_emitted = 0;
    bool printed_binding_summary = false;
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

        if (!CheckXr(instance, xrSyncActions(session, &sync_info), "xrSyncActions")) {
            break;
        }

        if (!printed_binding_summary) {
            PrintBoundSourcesForAction(instance, session, grip_pose_action, "grip_pose");
            PrintBoundSourcesForAction(instance, session, aim_pose_action, "aim_pose");
            PrintBoundSourcesForAction(instance, session, select_click_action, "select_click");
            PrintBoundSourcesForAction(instance, session, select_value_action, "select_value");
            PrintBoundSourcesForAction(
                instance, session, anchor_cycle_click_action, "anchor_cycle_click");
            PrintBoundSourcesForAction(
                instance, session, snap_assist_click_action, "snap_assist_click");
            printed_binding_summary = true;
        }

        const XrTime sample_time =
            GetSampleTime(instance, convert_timespec_time, frame_state.predictedDisplayTime);

        ControllerPoseSample grip_left;
        ControllerPoseSample grip_right;
        ControllerPoseSample aim_left;
        ControllerPoseSample aim_right;
        if (!QueryControllerPose(instance, session, grip_pose_action, left_hand_path,
                                 grip_left_space, local_space, sample_time, &grip_left) ||
            !QueryControllerPose(instance, session, grip_pose_action, right_hand_path,
                                 grip_right_space, local_space, sample_time, &grip_right) ||
            !QueryControllerPose(instance, session, aim_pose_action, left_hand_path,
                                 aim_left_space, local_space, sample_time, &aim_left) ||
            !QueryControllerPose(instance, session, aim_pose_action, right_hand_path,
                                 aim_right_space, local_space, sample_time, &aim_right)) {
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
        std::cout << "\"sample\":" << samples_emitted << ",";
        PrintControllerJson("left", selected_left, select_left, anchor_cycle_left, snap_assist_left);
        std::cout << ",";
        PrintControllerJson("right", selected_right, select_right, anchor_cycle_right,
                            snap_assist_right);
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

    xrDestroySpace(local_space);
    xrDestroySpace(aim_right_space);
    xrDestroySpace(aim_left_space);
    xrDestroySpace(grip_right_space);
    xrDestroySpace(grip_left_space);
    xrDestroySession(session);
    xrDestroyActionSet(action_set);
    xrDestroyInstance(instance);
    return 0;
}
