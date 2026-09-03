"""Table-plane safety helpers for Demo 2 phone controller targets."""

import torch


def _normalized_reverse_factor(reverse_factor):
    reverse_factor = float(reverse_factor)
    if reverse_factor not in (-1.0, 1.0):
        raise ValueError("reverse_factor must be -1.0 or 1.0")
    return reverse_factor


def build_ground_offset_limits(
    base_controller_targets,
    control_masks,
    reverse_factor,
    *,
    ground_z=0.0,
    clearance=0.0,
):
    """Return the minimum safe signed-Z offset for each session/control part.

    The simulator considers ``(z - ground_z) * reverse_factor >= 0`` to be the
    valid side of the table. Each control part moves by one shared offset, so
    its closest controller point determines the absolute plane limit. The
    calibrated base pose is also a limit: the rope can already be touching the
    table even when its kinematic controller points still have clearance.
    """

    if not torch.is_tensor(base_controller_targets):
        raise TypeError("base_controller_targets must be a torch.Tensor")
    if base_controller_targets.ndim != 3 or base_controller_targets.shape[-1] != 3:
        raise ValueError(
            "base_controller_targets must have shape (sessions, controllers, 3)"
        )
    if not control_masks:
        raise ValueError("control_masks must contain at least one control part")

    reverse_factor = _normalized_reverse_factor(reverse_factor)
    clearance = float(clearance)
    if clearance < 0.0:
        raise ValueError("clearance must be non-negative")

    signed_base_z = (
        base_controller_targets[..., 2] - float(ground_z)
    ) * reverse_factor
    limits = []
    for part_idx, mask in enumerate(control_masks):
        indices = torch.as_tensor(
            mask,
            device=base_controller_targets.device,
            dtype=torch.long,
        ).reshape(-1)
        if indices.numel() == 0:
            raise ValueError(f"control_masks[{part_idx}] must not be empty")
        closest_signed_z = signed_base_z.index_select(1, indices).amin(dim=1)
        absolute_plane_limit = clearance - closest_signed_z
        # Never drive farther toward the table than the calibrated rest pose.
        # A positive signed offset may decrease toward zero while returning
        # from a prior away-from-table movement. It must never become negative,
        # which would hide compression in the controller-object springs.
        limits.append(torch.clamp(absolute_plane_limit, min=0.0))

    return torch.stack(limits, dim=1)


def clamp_control_offsets_to_ground(
    control_offsets,
    minimum_signed_z_offsets,
    reverse_factor,
):
    """Clamp accumulated offsets in place so controller targets stay valid."""

    if not torch.is_tensor(control_offsets):
        raise TypeError("control_offsets must be a torch.Tensor")
    if control_offsets.ndim != 3 or control_offsets.shape[-1] != 3:
        raise ValueError("control_offsets must have shape (sessions, parts, 3)")
    if tuple(minimum_signed_z_offsets.shape) != tuple(control_offsets.shape[:2]):
        raise ValueError(
            "minimum_signed_z_offsets must match control_offsets[:2], got "
            f"{tuple(minimum_signed_z_offsets.shape)} and "
            f"{tuple(control_offsets.shape[:2])}"
        )

    reverse_factor = _normalized_reverse_factor(reverse_factor)
    limits = minimum_signed_z_offsets.to(
        device=control_offsets.device,
        dtype=control_offsets.dtype,
    )
    signed_z_offsets = control_offsets[..., 2] * reverse_factor
    clamped_signed_z_offsets = torch.maximum(signed_z_offsets, limits)
    control_offsets[..., 2].copy_(clamped_signed_z_offsets * reverse_factor)
    return control_offsets
