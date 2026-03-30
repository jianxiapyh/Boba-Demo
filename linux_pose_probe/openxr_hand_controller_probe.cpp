#define XR_USE_TIMESPEC
#include <openxr/openxr.h>
#include <openxr/openxr_platform.h>

#include <chrono>
#include <cstring>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <thread>
#include <vector>

namespace {

constexpr int kDefaultSampleCount = 10;
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

bool IsHandRelatedExtension(const char* name) {
    return std::strstr(name, "hand") != nullptr || std::strstr(name, "palm") != nullptr;
}

int ParseSampleCount(int argc, char** argv) {
    if (argc < 2) {
        return kDefaultSampleCount;
    }

    try {
        return std::max(1, std::stoi(argv[1]));
    } catch (const std::exception&) {
        std::cerr << "Invalid sample count '" << argv[1]
                  << "'. Usage: openxr_hand_controller_probe [sample_count]\n";
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

struct HandProbeFns {
    PFN_xrCreateHandTrackerEXT create = nullptr;
    PFN_xrDestroyHandTrackerEXT destroy = nullptr;
    PFN_xrLocateHandJointsEXT locate = nullptr;
};

struct HandSample {
    bool is_active = false;
    bool palm_valid = false;
    bool index_tip_valid = false;
    XrVector3f palm{};
    XrVector3f index_tip{};
};

struct ControllerSample {
    const char* source = "none";
    bool action_active = false;
    bool position_valid = false;
    bool orientation_valid = false;
    bool position_tracked = false;
    bool orientation_tracked = false;
    XrPosef pose = IdentityPose();
};

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

    std::cout << label << " bound sources:\n";
    if (source_count == 0) {
        std::cout << "  (none)\n";
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
        std::cout << "  " << PathToString(instance, source) << " -> "
                  << GetLocalizedSourceName(instance, session, source) << "\n";
    }
}

void PrintRelevantExtensions(const std::vector<XrExtensionProperties>& extensions) {
    std::cout << "Enumerated hand-related runtime extensions:\n";
    bool found = false;
    for (const auto& extension : extensions) {
        if (!IsHandRelatedExtension(extension.extensionName)) {
            continue;
        }
        found = true;
        std::cout << "  " << extension.extensionName << " (spec " << extension.extensionVersion
                  << ")\n";
    }
    if (!found) {
        std::cout << "  (none)\n";
    }
}

bool LoadHandFns(XrInstance instance, HandProbeFns* fns) {
    if (!CheckXr(instance,
                 xrGetInstanceProcAddr(instance, "xrCreateHandTrackerEXT",
                                       reinterpret_cast<PFN_xrVoidFunction*>(&fns->create)),
                 "xrGetInstanceProcAddr(xrCreateHandTrackerEXT)")) {
        return false;
    }
    if (!CheckXr(instance,
                 xrGetInstanceProcAddr(instance, "xrDestroyHandTrackerEXT",
                                       reinterpret_cast<PFN_xrVoidFunction*>(&fns->destroy)),
                 "xrGetInstanceProcAddr(xrDestroyHandTrackerEXT)")) {
        return false;
    }
    if (!CheckXr(instance,
                 xrGetInstanceProcAddr(instance, "xrLocateHandJointsEXT",
                                       reinterpret_cast<PFN_xrVoidFunction*>(&fns->locate)),
                 "xrGetInstanceProcAddr(xrLocateHandJointsEXT)")) {
        return false;
    }
    return true;
}

bool CreateHandTracker(
    XrInstance instance,
    XrSession session,
    const HandProbeFns& fns,
    XrHandEXT hand,
    XrHandTrackerEXT* tracker) {
    XrHandTrackerCreateInfoEXT create_info =
        MakeXrStruct<XrHandTrackerCreateInfoEXT>(XR_TYPE_HAND_TRACKER_CREATE_INFO_EXT);
    create_info.hand = hand;
    create_info.handJointSet = XR_HAND_JOINT_SET_DEFAULT_EXT;
    return CheckXr(instance, fns.create(session, &create_info, tracker), "xrCreateHandTrackerEXT");
}

bool SampleHand(
    XrInstance instance,
    const HandProbeFns& fns,
    XrHandTrackerEXT tracker,
    XrSpace base_space,
    XrTime sample_time,
    HandSample* sample) {
    XrHandJointLocationEXT joint_locations[XR_HAND_JOINT_COUNT_EXT];
    XrHandJointLocationsEXT locations =
        MakeXrStruct<XrHandJointLocationsEXT>(XR_TYPE_HAND_JOINT_LOCATIONS_EXT);
    locations.jointCount = XR_HAND_JOINT_COUNT_EXT;
    locations.jointLocations = joint_locations;

    XrHandJointsLocateInfoEXT locate_info =
        MakeXrStruct<XrHandJointsLocateInfoEXT>(XR_TYPE_HAND_JOINTS_LOCATE_INFO_EXT);
    locate_info.baseSpace = base_space;
    locate_info.time = sample_time;

    if (!CheckXr(instance, fns.locate(tracker, &locate_info, &locations), "xrLocateHandJointsEXT")) {
        return false;
    }

    sample->is_active = locations.isActive == XR_TRUE;

    const auto& palm = joint_locations[XR_HAND_JOINT_PALM_EXT];
    sample->palm_valid =
        (palm.locationFlags & XR_SPACE_LOCATION_POSITION_VALID_BIT) != 0;
    sample->palm = palm.pose.position;

    const auto& index_tip = joint_locations[XR_HAND_JOINT_INDEX_TIP_EXT];
    sample->index_tip_valid =
        (index_tip.locationFlags & XR_SPACE_LOCATION_POSITION_VALID_BIT) != 0;
    sample->index_tip = index_tip.pose.position;
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
        std::cout << "Binding suggestion skipped for " << profile_string << ": "
                  << XrResultString(instance, result) << "\n";
        return true;
    }
    return CheckXr(instance, result, "xrSuggestInteractionProfileBindings");
}

bool BuildPoseBindings(
    XrInstance instance,
    XrAction grip_pose_action,
    XrAction aim_pose_action,
    const char* profile_string,
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

    bindings->clear();
    bindings->reserve(4);
    for (const auto& spec : specs) {
        XrPath path = XR_NULL_PATH;
        if (!StringToPath(instance, spec.path, &path)) {
            return false;
        }
        bindings->push_back({spec.action, path});
    }

    return SuggestBindingsForProfile(instance, profile_string, *bindings);
}

bool QueryControllerPose(
    XrInstance instance,
    XrSession session,
    XrAction pose_action,
    XrPath subaction_path,
    XrSpace action_space,
    XrSpace base_space,
    XrTime sample_time,
    ControllerSample* sample) {
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

void SelectPreferredControllerSample(
    const ControllerSample& grip,
    const ControllerSample& aim,
    ControllerSample* selected) {
    *selected = grip;
    selected->source = "grip";

    const bool aim_preferred = (aim.action_active && (aim.position_valid || aim.orientation_valid)) ||
                               (!grip.action_active && aim.action_active) ||
                               ((!grip.position_valid && !grip.orientation_valid) &&
                                (aim.position_valid || aim.orientation_valid));
    if (aim_preferred) {
        *selected = aim;
        selected->source = "aim";
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
    const bool has_hand_tracking = HasExtension(extensions, XR_EXT_HAND_TRACKING_EXTENSION_NAME);
    const bool has_msft_hand_interaction =
        HasExtension(extensions, XR_MSFT_HAND_INTERACTION_EXTENSION_NAME);
    const bool has_hand_tracking_mesh =
        HasExtension(extensions, XR_MSFT_HAND_TRACKING_MESH_EXTENSION_NAME);

    std::cout << "Runtime extension count: " << extension_count << "\n";
    PrintRelevantExtensions(extensions);
    std::cout << "XR_MND_headless: " << (has_headless ? "yes" : "no") << "\n";
    std::cout << "XR_KHR_convert_timespec_time: " << (has_convert_timespec ? "yes" : "no")
              << "\n";
    std::cout << "XR_EXT_hand_tracking: " << (has_hand_tracking ? "yes" : "no") << "\n";
    std::cout << "XR_MSFT_hand_interaction: " << (has_msft_hand_interaction ? "yes" : "no")
              << "\n";
    std::cout << "XR_MSFT_hand_tracking_mesh: " << (has_hand_tracking_mesh ? "yes" : "no")
              << "\n";

    if (!has_headless) {
        std::cerr << "This runtime does not expose XR_MND_headless, so the minimal headless probe "
                     "cannot open a session.\n";
        return 3;
    }

    std::vector<const char*> enabled_extensions = {XR_MND_HEADLESS_EXTENSION_NAME};
    if (has_convert_timespec) {
        enabled_extensions.push_back(XR_KHR_CONVERT_TIMESPEC_TIME_EXTENSION_NAME);
    }
    if (has_hand_tracking) {
        enabled_extensions.push_back(XR_EXT_HAND_TRACKING_EXTENSION_NAME);
    }

    XrInstanceCreateInfo instance_info =
        MakeXrStruct<XrInstanceCreateInfo>(XR_TYPE_INSTANCE_CREATE_INFO);
    std::strncpy(instance_info.applicationInfo.applicationName, "openxr_input_probe",
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

    XrActionSetCreateInfo action_set_info =
        MakeXrStruct<XrActionSetCreateInfo>(XR_TYPE_ACTION_SET_CREATE_INFO);
    std::strncpy(action_set_info.actionSetName, "gameplay", XR_MAX_ACTION_SET_NAME_SIZE - 1);
    std::strncpy(action_set_info.localizedActionSetName, "Gameplay",
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

    auto create_pose_action = [&](const char* action_name, const char* localized_name,
                                  XrAction* action) -> bool {
        XrActionCreateInfo action_info =
            MakeXrStruct<XrActionCreateInfo>(XR_TYPE_ACTION_CREATE_INFO);
        std::strncpy(action_info.actionName, action_name, XR_MAX_ACTION_NAME_SIZE - 1);
        std::strncpy(action_info.localizedActionName, localized_name,
                     XR_MAX_LOCALIZED_ACTION_NAME_SIZE - 1);
        action_info.actionType = XR_ACTION_TYPE_POSE_INPUT;
        action_info.countSubactionPaths = 2;
        action_info.subactionPaths = subaction_paths;
        return CheckXr(instance, xrCreateAction(action_set, &action_info, action), "xrCreateAction");
    };

    XrAction grip_pose_action = XR_NULL_HANDLE;
    XrAction aim_pose_action = XR_NULL_HANDLE;
    if (!create_pose_action("grip_pose", "Grip Pose", &grip_pose_action) ||
        !create_pose_action("aim_pose", "Aim Pose", &aim_pose_action)) {
        xrDestroyActionSet(action_set);
        xrDestroyInstance(instance);
        return 1;
    }

    std::vector<XrActionSuggestedBinding> bindings;
    const char* profiles[] = {
        "/interaction_profiles/khr/simple_controller",
        "/interaction_profiles/oculus/touch_controller",
        "/interaction_profiles/htc/vive_controller",
        "/interaction_profiles/valve/index_controller",
        "/interaction_profiles/microsoft/motion_controller",
    };
    for (const char* profile : profiles) {
        if (!BuildPoseBindings(instance, grip_pose_action, aim_pose_action, profile, &bindings)) {
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

    XrActionSpaceCreateInfo grip_left_space_info =
        MakeXrStruct<XrActionSpaceCreateInfo>(XR_TYPE_ACTION_SPACE_CREATE_INFO);
    grip_left_space_info.action = grip_pose_action;
    grip_left_space_info.subactionPath = left_hand_path;
    grip_left_space_info.poseInActionSpace = IdentityPose();

    XrActionSpaceCreateInfo grip_right_space_info =
        MakeXrStruct<XrActionSpaceCreateInfo>(XR_TYPE_ACTION_SPACE_CREATE_INFO);
    grip_right_space_info.action = grip_pose_action;
    grip_right_space_info.subactionPath = right_hand_path;
    grip_right_space_info.poseInActionSpace = IdentityPose();

    XrActionSpaceCreateInfo aim_left_space_info =
        MakeXrStruct<XrActionSpaceCreateInfo>(XR_TYPE_ACTION_SPACE_CREATE_INFO);
    aim_left_space_info.action = aim_pose_action;
    aim_left_space_info.subactionPath = left_hand_path;
    aim_left_space_info.poseInActionSpace = IdentityPose();

    XrActionSpaceCreateInfo aim_right_space_info =
        MakeXrStruct<XrActionSpaceCreateInfo>(XR_TYPE_ACTION_SPACE_CREATE_INFO);
    aim_right_space_info.action = aim_pose_action;
    aim_right_space_info.subactionPath = right_hand_path;
    aim_right_space_info.poseInActionSpace = IdentityPose();

    XrSpace grip_left_space = XR_NULL_HANDLE;
    XrSpace grip_right_space = XR_NULL_HANDLE;
    XrSpace aim_left_space = XR_NULL_HANDLE;
    XrSpace aim_right_space = XR_NULL_HANDLE;
    if (!CheckXr(instance, xrCreateActionSpace(session, &grip_left_space_info, &grip_left_space),
                 "xrCreateActionSpace(grip_left)") ||
        !CheckXr(instance, xrCreateActionSpace(session, &grip_right_space_info, &grip_right_space),
                 "xrCreateActionSpace(grip_right)") ||
        !CheckXr(instance, xrCreateActionSpace(session, &aim_left_space_info, &aim_left_space),
                 "xrCreateActionSpace(aim_left)") ||
        !CheckXr(instance, xrCreateActionSpace(session, &aim_right_space_info, &aim_right_space),
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
        std::cerr << "Session never reached READY/RUNNING state. Last state=" << session_state
                  << "\n";
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

    HandProbeFns hand_fns;
    XrHandTrackerEXT left_hand_tracker = XR_NULL_HANDLE;
    XrHandTrackerEXT right_hand_tracker = XR_NULL_HANDLE;
    bool hand_probe_ready = false;
    if (has_hand_tracking) {
        hand_probe_ready = LoadHandFns(instance, &hand_fns) &&
                           CreateHandTracker(instance, session, hand_fns, XR_HAND_LEFT_EXT,
                                             &left_hand_tracker) &&
                           CreateHandTracker(instance, session, hand_fns, XR_HAND_RIGHT_EXT,
                                             &right_hand_tracker);
        if (!hand_probe_ready) {
            std::cout << "Hand tracking extension is enumerated, but hand tracker creation failed.\n";
        }
    } else {
        std::cout << "Hand tracking extension is not enumerated, so this OpenXR runtime path "
                     "does not expose OpenXR hand joints.\n";
    }

    XrInteractionProfileState left_profile_state =
        MakeXrStruct<XrInteractionProfileState>(XR_TYPE_INTERACTION_PROFILE_STATE);
    XrInteractionProfileState right_profile_state =
        MakeXrStruct<XrInteractionProfileState>(XR_TYPE_INTERACTION_PROFILE_STATE);
    if (CheckXr(instance, xrGetCurrentInteractionProfile(session, left_hand_path, &left_profile_state),
                "xrGetCurrentInteractionProfile(left)")) {
        std::cout << "Left interaction profile: "
                  << PathToString(instance, left_profile_state.interactionProfile) << "\n";
    }
    if (CheckXr(instance,
                xrGetCurrentInteractionProfile(session, right_hand_path, &right_profile_state),
                "xrGetCurrentInteractionProfile(right)")) {
        std::cout << "Right interaction profile: "
                  << PathToString(instance, right_profile_state.interactionProfile) << "\n";
    }

    XrActiveActionSet active_action_set{action_set, XR_NULL_PATH};
    XrActionsSyncInfo sync_info = MakeXrStruct<XrActionsSyncInfo>(XR_TYPE_ACTIONS_SYNC_INFO);
    sync_info.countActiveActionSets = 1;
    sync_info.activeActionSets = &active_action_set;

    std::cout << std::fixed << std::setprecision(6);
    std::cout
        << "sample,hand_ext,left_hand_active,left_palm_valid,left_palm_x,left_palm_y,left_palm_z,"
        << "left_index_valid,left_index_x,left_index_y,left_index_z,right_hand_active,"
        << "right_palm_valid,right_palm_x,right_palm_y,right_palm_z,right_index_valid,right_index_x,"
        << "right_index_y,right_index_z,left_controller_source,left_controller_active,"
        << "left_controller_pos_valid,left_controller_ori_valid,left_controller_px,left_controller_py,"
        << "left_controller_pz,right_controller_source,right_controller_active,"
        << "right_controller_pos_valid,right_controller_ori_valid,right_controller_px,"
        << "right_controller_py,right_controller_pz\n";

    bool observed_left_hand = false;
    bool observed_right_hand = false;
    bool observed_left_controller = false;
    bool observed_right_controller = false;
    bool printed_binding_summary = false;

    for (int sample_index = 0; sample_index < sample_count && !exit_requested; ++sample_index) {
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

        const XrTime sample_time =
            GetSampleTime(instance, convert_timespec_time, frame_state.predictedDisplayTime);

        HandSample left_hand_sample;
        HandSample right_hand_sample;
        if (hand_probe_ready) {
            if (!SampleHand(instance, hand_fns, left_hand_tracker, local_space, sample_time,
                            &left_hand_sample) ||
                !SampleHand(instance, hand_fns, right_hand_tracker, local_space, sample_time,
                            &right_hand_sample)) {
                hand_probe_ready = false;
            }
        }

        if (!CheckXr(instance, xrSyncActions(session, &sync_info), "xrSyncActions")) {
            break;
        }

        if (!printed_binding_summary) {
            XrInteractionProfileState synced_left_profile =
                MakeXrStruct<XrInteractionProfileState>(XR_TYPE_INTERACTION_PROFILE_STATE);
            XrInteractionProfileState synced_right_profile =
                MakeXrStruct<XrInteractionProfileState>(XR_TYPE_INTERACTION_PROFILE_STATE);
            if (CheckXr(instance,
                        xrGetCurrentInteractionProfile(session, left_hand_path, &synced_left_profile),
                        "xrGetCurrentInteractionProfile(left, synced)")) {
                std::cout << "Left interaction profile after sync: "
                          << PathToString(instance, synced_left_profile.interactionProfile) << "\n";
            }
            if (CheckXr(instance, xrGetCurrentInteractionProfile(session, right_hand_path,
                                                                 &synced_right_profile),
                        "xrGetCurrentInteractionProfile(right, synced)")) {
                std::cout << "Right interaction profile after sync: "
                          << PathToString(instance, synced_right_profile.interactionProfile) << "\n";
            }
            PrintBoundSourcesForAction(instance, session, grip_pose_action, "grip_pose");
            PrintBoundSourcesForAction(instance, session, aim_pose_action, "aim_pose");
            printed_binding_summary = true;
        }

        ControllerSample grip_left;
        ControllerSample grip_right;
        ControllerSample aim_left;
        ControllerSample aim_right;
        if (!QueryControllerPose(instance, session, grip_pose_action, left_hand_path, grip_left_space,
                                 local_space, sample_time, &grip_left) ||
            !QueryControllerPose(instance, session, grip_pose_action, right_hand_path,
                                 grip_right_space, local_space, sample_time, &grip_right) ||
            !QueryControllerPose(instance, session, aim_pose_action, left_hand_path, aim_left_space,
                                 local_space, sample_time, &aim_left) ||
            !QueryControllerPose(instance, session, aim_pose_action, right_hand_path, aim_right_space,
                                 local_space, sample_time, &aim_right)) {
            break;
        }

        ControllerSample selected_left;
        ControllerSample selected_right;
        SelectPreferredControllerSample(grip_left, aim_left, &selected_left);
        SelectPreferredControllerSample(grip_right, aim_right, &selected_right);

        observed_left_hand = observed_left_hand || (left_hand_sample.is_active && left_hand_sample.palm_valid);
        observed_right_hand =
            observed_right_hand || (right_hand_sample.is_active && right_hand_sample.palm_valid);
        observed_left_controller =
            observed_left_controller || (selected_left.action_active && selected_left.position_valid);
        observed_right_controller =
            observed_right_controller || (selected_right.action_active && selected_right.position_valid);

        auto nan = std::numeric_limits<float>::quiet_NaN();
        std::cout << sample_index << "," << (has_hand_tracking ? 1 : 0) << ","
                  << (left_hand_sample.is_active ? 1 : 0) << ","
                  << (left_hand_sample.palm_valid ? 1 : 0) << ","
                  << (left_hand_sample.palm_valid ? left_hand_sample.palm.x : nan) << ","
                  << (left_hand_sample.palm_valid ? left_hand_sample.palm.y : nan) << ","
                  << (left_hand_sample.palm_valid ? left_hand_sample.palm.z : nan) << ","
                  << (left_hand_sample.index_tip_valid ? 1 : 0) << ","
                  << (left_hand_sample.index_tip_valid ? left_hand_sample.index_tip.x : nan) << ","
                  << (left_hand_sample.index_tip_valid ? left_hand_sample.index_tip.y : nan) << ","
                  << (left_hand_sample.index_tip_valid ? left_hand_sample.index_tip.z : nan) << ","
                  << (right_hand_sample.is_active ? 1 : 0) << ","
                  << (right_hand_sample.palm_valid ? 1 : 0) << ","
                  << (right_hand_sample.palm_valid ? right_hand_sample.palm.x : nan) << ","
                  << (right_hand_sample.palm_valid ? right_hand_sample.palm.y : nan) << ","
                  << (right_hand_sample.palm_valid ? right_hand_sample.palm.z : nan) << ","
                  << (right_hand_sample.index_tip_valid ? 1 : 0) << ","
                  << (right_hand_sample.index_tip_valid ? right_hand_sample.index_tip.x : nan) << ","
                  << (right_hand_sample.index_tip_valid ? right_hand_sample.index_tip.y : nan) << ","
                  << (right_hand_sample.index_tip_valid ? right_hand_sample.index_tip.z : nan) << ","
                  << selected_left.source << "," << (selected_left.action_active ? 1 : 0) << ","
                  << (selected_left.position_valid ? 1 : 0) << ","
                  << (selected_left.orientation_valid ? 1 : 0) << ","
                  << (selected_left.position_valid ? selected_left.pose.position.x : nan) << ","
                  << (selected_left.position_valid ? selected_left.pose.position.y : nan) << ","
                  << (selected_left.position_valid ? selected_left.pose.position.z : nan) << ","
                  << selected_right.source << "," << (selected_right.action_active ? 1 : 0) << ","
                  << (selected_right.position_valid ? 1 : 0) << ","
                  << (selected_right.orientation_valid ? 1 : 0) << ","
                  << (selected_right.position_valid ? selected_right.pose.position.x : nan) << ","
                  << (selected_right.position_valid ? selected_right.pose.position.y : nan) << ","
                  << (selected_right.position_valid ? selected_right.pose.position.z : nan) << "\n";

        XrFrameEndInfo end_info = MakeXrStruct<XrFrameEndInfo>(XR_TYPE_FRAME_END_INFO);
        end_info.displayTime = frame_state.predictedDisplayTime;
        end_info.environmentBlendMode = XR_ENVIRONMENT_BLEND_MODE_OPAQUE;
        end_info.layerCount = 0;
        end_info.layers = nullptr;
        if (!CheckXr(instance, xrEndFrame(session, &end_info), "xrEndFrame")) {
            break;
        }
    }

    std::cout << "Observed left hand joints: " << (observed_left_hand ? "yes" : "no") << "\n";
    std::cout << "Observed right hand joints: " << (observed_right_hand ? "yes" : "no") << "\n";
    std::cout << "Observed left controller pose: " << (observed_left_controller ? "yes" : "no")
              << "\n";
    std::cout << "Observed right controller pose: " << (observed_right_controller ? "yes" : "no")
              << "\n";

    if (left_hand_tracker != XR_NULL_HANDLE && hand_fns.destroy != nullptr) {
        hand_fns.destroy(left_hand_tracker);
    }
    if (right_hand_tracker != XR_NULL_HANDLE && hand_fns.destroy != nullptr) {
        hand_fns.destroy(right_hand_tracker);
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
