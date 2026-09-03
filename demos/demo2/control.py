def clamp_unit(value):
    return max(-1.0, min(1.0, float(value)))


def clamp_vector3(vector):
    if isinstance(vector, dict):
        return (
            clamp_unit(vector.get("x", 0.0)),
            clamp_unit(vector.get("y", 0.0)),
            clamp_unit(vector.get("z", 0.0)),
        )
    values = list(vector or ())
    values = (values + [0.0, 0.0, 0.0])[:3]
    return tuple(clamp_unit(value) for value in values)


# Backward-compatible calibration for the packaged rope scene. Runtime code
# derives these signs from each case's camera because the sloth camera is
# rotated in the table plane relative to the rope camera.
PHONE_TO_WORLD_AXIS_SIGNS = (-1.0, -1.0, 1.0)


def normalize_axis_signs(axis_signs):
    values = tuple(float(value) for value in axis_signs)
    if len(values) != 3 or any(value == 0.0 for value in values):
        raise ValueError("axis_signs must contain three non-zero values")
    return tuple(1.0 if value > 0.0 else -1.0 for value in values)


def control_vector_to_step(
    x,
    y,
    z,
    step_size,
    axis_signs=PHONE_TO_WORLD_AXIS_SIGNS,
):
    step_size = float(step_size)
    axis_signs = normalize_axis_signs(axis_signs)
    return (
        clamp_unit(x) * step_size * axis_signs[0],
        clamp_unit(y) * step_size * axis_signs[1],
        clamp_unit(z) * step_size * axis_signs[2],
    )


def resolve_phone_to_world_axis_signs(
    controller_points,
    *,
    w2c,
    intrinsic,
    projection_step=0.005,
    fallback_axis_signs=PHONE_TO_WORLD_AXIS_SIGNS,
):
    """Derive world-axis signs that make phone arrows move in screen space.

    Phone +X is the forward/up arrow, +Y is the right arrow, and +Z is the
    down arrow. Only signs are calibrated; the physical world axes and table-Z
    collision behavior remain unchanged.
    """

    import numpy as np

    points = np.asarray(controller_points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or points.shape[0] < 1:
        raise ValueError("controller_points must have shape (N, 3) with N >= 1")
    w2c = np.asarray(w2c, dtype=np.float64)
    intrinsic = np.asarray(intrinsic, dtype=np.float64)
    if w2c.shape != (4, 4) or intrinsic.shape != (3, 3):
        raise ValueError("w2c and intrinsic must have shapes (4, 4) and (3, 3)")
    projection_step = abs(float(projection_step))
    if projection_step == 0.0:
        raise ValueError("projection_step must be positive")
    fallback_axis_signs = normalize_axis_signs(fallback_axis_signs)

    projection = intrinsic @ w2c[:3, :]

    def project(point):
        projected = projection @ np.concatenate([point, [1.0]])
        if not np.isfinite(projected).all() or abs(float(projected[2])) < 1e-9:
            return None
        return projected[:2] / projected[2]

    center = points.mean(axis=0)
    base_pixel = project(center)
    if base_pixel is None:
        return fallback_axis_signs

    # Relevant screen coordinate and desired direction for positive phone X/Y/Z.
    # Image coordinates increase rightward and downward.
    screen_components = (1, 0, 1)
    desired_screen_signs = (-1.0, 1.0, 1.0)
    resolved = []
    for axis, (component, desired_sign) in enumerate(
        zip(screen_components, desired_screen_signs)
    ):
        displaced = center.copy()
        displaced[axis] += projection_step
        displaced_pixel = project(displaced)
        if displaced_pixel is None:
            resolved.append(fallback_axis_signs[axis])
            continue
        screen_delta = float(displaced_pixel[component] - base_pixel[component])
        if not np.isfinite(screen_delta) or abs(screen_delta) < 1e-6:
            resolved.append(fallback_axis_signs[axis])
            continue
        resolved.append(1.0 if screen_delta * desired_sign > 0.0 else -1.0)
    return tuple(resolved)


def add_vectors_clamped(*vectors):
    x = y = z = 0.0
    for vector in vectors:
        vx, vy, vz = clamp_vector3(vector)
        x += vx
        y += vy
        z += vz
    return (clamp_unit(x), clamp_unit(y), clamp_unit(z))


def joystick_to_interactive_2d_step(dx, dy, step_size):
    step_size = float(step_size)
    return (
        clamp_unit(dy) * step_size,
        clamp_unit(dx) * step_size,
        0.0,
    )


def legacy_joystick_to_control_vector(dx, dy):
    return (clamp_unit(dy), clamp_unit(dx), 0.0)


def resolve_demo2_control_parts(case_name, requested="auto", double_control_cases=None):
    requested = str(requested or "auto").lower()
    if requested in ("1", "one"):
        return 1
    if requested in ("2", "two"):
        return 2
    if requested != "auto":
        raise ValueError("demo2_control_parts must be 'auto', '1', or '2'")

    case_name = str(case_name or "")
    normalized = case_name.lower()
    special_cases = {
        str(value).strip().lower()
        for value in (double_control_cases or ())
        if str(value).strip()
    }
    if "double" in normalized or normalized in special_cases:
        return 2
    return 1


def resolve_controller_part_indices(
    controller_points,
    control_parts,
    *,
    w2c,
    intrinsic,
):
    """Split controller nodes into viewer-left and viewer-right regions."""

    import numpy as np

    points = np.asarray(controller_points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("controller_points must have shape (N, 3)")
    all_indices = np.arange(points.shape[0], dtype=np.int64)
    if int(control_parts) == 1:
        return (all_indices,)
    if int(control_parts) != 2:
        raise ValueError("control_parts must be 1 or 2")
    if points.shape[0] < 2:
        return (all_indices, all_indices.copy())

    try:
        from sklearn.cluster import KMeans
    except Exception as exc:
        raise RuntimeError(
            "Demo 2 two-hand control requires scikit-learn for controller "
            "point splitting."
        ) from exc

    labels = KMeans(n_clusters=2, random_state=0, n_init=10).fit_predict(points)
    masks = [labels == 0, labels == 1]
    if not masks[0].any() or not masks[1].any():
        return (all_indices, all_indices.copy())

    projection = np.asarray(intrinsic, dtype=np.float32) @ np.asarray(
        w2c, dtype=np.float32
    )[:3, :]
    projected_x = []
    for mask in masks:
        center_h = np.concatenate([points[mask].mean(axis=0), [1.0]])
        projected = projection @ center_h
        projected_x.append(float(projected[0] / projected[-1]))
    if projected_x[0] > projected_x[1]:
        masks.reverse()

    return tuple(np.flatnonzero(mask).astype(np.int64) for mask in masks)
