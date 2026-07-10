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


def control_vector_to_step(x, y, z, step_size):
    step_size = float(step_size)
    return (
        clamp_unit(x) * step_size,
        clamp_unit(y) * step_size,
        clamp_unit(z) * step_size,
    )


def add_vectors_clamped(*vectors):
    x = y = z = 0.0
    for vector in vectors:
        vx, vy, vz = clamp_vector3(vector)
        x += vx
        y += vy
        z += vz
    return (clamp_unit(x), clamp_unit(y), clamp_unit(z))


def joystick_to_interactive_2d_step(dx, dy, step_size):
    return control_vector_to_step(dy, dx, 0.0, step_size)


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
