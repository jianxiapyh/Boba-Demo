import numpy as np
import torch
from qqtt.utils import logger, cfg
import warp as wp

wp.init()
wp.set_device("cuda:0")
if not cfg.use_graph:
    wp.config.mode = "debug"
    wp.config.verbose = True
    wp.config.verify_autograd_array_access = True


class State:
    def __init__(self, wp_init_vertices, num_control_points):
        self.wp_x = wp.zeros_like(wp_init_vertices, requires_grad=True)
        self.wp_v_before_collision = wp.zeros_like(wp_init_vertices, requires_grad=True)
        self.wp_v_before_ground = wp.zeros_like(wp_init_vertices, requires_grad=True)
        self.wp_v = wp.zeros_like(self.wp_x, requires_grad=True)
        self.wp_vertice_forces = wp.zeros_like(self.wp_x, requires_grad=True)
        self.wp_dummy_vertices_forces = wp.zeros_like(self.wp_x, requires_grad=False)
        # No need to compute the gradient for the control points
        self.wp_control_x = wp.zeros(
            (num_control_points), dtype=wp.vec3, requires_grad=False
        )
        self.wp_control_v = wp.zeros_like(self.wp_control_x, requires_grad=False)

    def clear_forces(self):
        self.wp_vertice_forces.zero_()
        #additional 
        self.wp_dummy_vertices_forces.zero_()


    @property
    def requires_grad(self):
        """Indicates whether the state arrays have gradient computation enabled."""
        return self.wp_x.requires_grad


@wp.kernel(enable_backward=False)
def copy_vec3(data: wp.array(dtype=wp.vec3), origin: wp.array(dtype=wp.vec3)):
    tid = wp.tid()
    origin[tid] = data[tid]


@wp.kernel(enable_backward=False)
def set_control_points(
    num_substeps: int,
    original_control_point: wp.array(dtype=wp.vec3),
    target_control_point: wp.array(dtype=wp.vec3),
    step: int,
    control_x: wp.array(dtype=wp.vec3),
):
    # Set the control points in each substep
    tid = wp.tid()

    t = float(step + 1) / float(num_substeps)
    control_x[tid] = (
        original_control_point[tid]
        + (target_control_point[tid] - original_control_point[tid]) * t
    )


#pyh potential additional optimization
#inv_rest_lengths is precomputed
#no log and exp for spring_Y, and already clamped for inference
@wp.kernel
def eval_springs_batched_opt(
    #batched [all obj][all controller]
    x: wp.array(dtype=wp.vec3), 
    v: wp.array(dtype=wp.vec3), 
    control_x: wp.array(dtype=wp.vec3), 
    control_v: wp.array(dtype=wp.vec3), 
    #shared 
    springs: wp.array(dtype=wp.vec2i),
    inv_rest_lengths: wp.array(dtype=float),
    spring_Y_clamped: wp.array(dtype=float),
    dashpot_damping: float,
    #pyh added variable to help indexing Spring_Y
    object_spring_single: int,
    object_spring_total: int,
    controller_spring_single: int,
    object_massnode_single: int,
    controller_massnode_single: int,
    f: wp.array(dtype=wp.vec3), #these needs to be all instance
):
    tid = wp.tid()
    local_idx = 0
    inst = 0
    #pyh if tid is a object-object spring
    if tid < object_spring_total:
        inst = tid // object_spring_single
        local_idx = tid - inst * object_spring_single
    else:
        #pyh now it is a controller-object spring
        t = tid - object_spring_total
        inst = t // controller_spring_single
        local_idx = object_spring_single + (t - inst * controller_spring_single)

    y = spring_Y_clamped[local_idx]

    #we need to remap to global
    global_idx1 = 0
    global_idx2 = 0

    local_idx1 = springs[local_idx][0] #lets say its (3,5) but in 2 instance it might be (13,15)
    local_idx2 = springs[local_idx][1]

    idx1_control = False
    idx2_control = False

    #it is a controller mass node
    if local_idx1 >= object_massnode_single:
        idx1_control = True
        ctrl1 = inst * controller_massnode_single + (local_idx1 - object_massnode_single)
        x1 = control_x[ctrl1]
        v1 = control_v[ctrl1]
    else:
        #it is an object mass node
        global_idx1 = inst * object_massnode_single + local_idx1
        x1 = x[global_idx1]
        v1 = v[global_idx1]
    
    if local_idx2 >= object_massnode_single:
        idx2_control = True
        ctrl2 = inst * controller_massnode_single + (local_idx2 - object_massnode_single)
        x2 = control_x[ctrl2]
        v2 = control_v[ctrl2]
    else:
        #it is an object mass node
        global_idx2 = inst * object_massnode_single + local_idx2
        x2 = x[global_idx2]
        v2 = v[global_idx2]

    inv_rest = inv_rest_lengths[local_idx]

    dis = x2 - x1
    dis_len = wp.length(dis)

    d = dis / wp.max(dis_len, 1e-6)

    spring_force = (
        y
        * (dis_len * inv_rest - 1.0)
        * d
    )

    v_rel = wp.dot(v2 - v1, d)
    dashpot_forces = dashpot_damping * v_rel * d

    overall_force = spring_force + dashpot_forces

    if not idx1_control:
        wp.atomic_add(f, global_idx1, overall_force)
    if not idx2_control:
        wp.atomic_sub(f, global_idx2, overall_force)

#updated since masses are shared
@wp.kernel
def update_vel_from_force(
    v: wp.array(dtype=wp.vec3),
    f: wp.array(dtype=wp.vec3),
    masses: wp.array(dtype=wp.float32),
    dt: float,
    drag_damping: float,
    reverse_factor: float,
    object_massnode_single: int,
    v_new: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()
    local_idx = 0

    inst = tid // object_massnode_single
    local_idx = tid - inst * object_massnode_single

    v0 = v[tid]
    f0 = f[tid]
    m0 = masses[local_idx]

    drag_damping_factor = wp.exp(-dt * drag_damping)
    all_force = f0 + m0 * wp.vec3(0.0, 0.0, -9.8) * reverse_factor
    a = all_force / m0
    v1 = v0 + a * dt
    v2 = v1 * drag_damping_factor

    v_new[tid] = v2


@wp.func
def loop(
    i: int,
    collision_indices: wp.array2d(dtype=wp.int32),
    collision_number: wp.array(dtype=wp.int32),
    x: wp.array(dtype=wp.vec3),
    v: wp.array(dtype=wp.vec3),
    masses: wp.array(dtype=wp.float32),
    masks: wp.array(dtype=wp.int32),
    collision_dist: float,
    clamp_collide_object_elas: float,
    clamp_collide_object_fric: float,
    mass_i: float, 
    offset: int, 
):
    x1 = x[i]
    v1 = v[i]

    #pyh added 
    m1 = mass_i
    mask1 = masks[i-offset]

    valid_count = float(0.0)
    J_sum = wp.vec3(0.0, 0.0, 0.0)
    for k in range(collision_number[i]):
        index = collision_indices[i][k]
        x2 = x[index]
        v2 = v[index]

        #pyh added
        local_idx2 = index - offset
        m2 = masses[local_idx2]
        mask2 = masks[local_idx2]

        dis = x2 - x1
        dis_len = wp.length(dis)
        relative_v = v2 - v1
        # If the distance is less than the collision distance and the two points are moving towards each other
        if (
            mask1 != mask2
            and dis_len < collision_dist
            and wp.dot(dis, relative_v) < -1e-4
        ):
            valid_count += 1.0

            collision_normal = dis / wp.max(dis_len, 1e-6)
            v_rel_n = wp.dot(relative_v, collision_normal) * collision_normal
            impulse_n = (-(1.0 + clamp_collide_object_elas) * v_rel_n) / (
                1.0 / m1 + 1.0 / m2
            )
            v_rel_n_length = wp.length(v_rel_n)

            v_rel_t = relative_v - v_rel_n
            v_rel_t_length = wp.max(wp.length(v_rel_t), 1e-6)
            a = wp.max(
                0.0,
                1.0
                - clamp_collide_object_fric
                * (1.0 + clamp_collide_object_elas)
                * v_rel_n_length
                / v_rel_t_length,
            )
            impulse_t = (a - 1.0) * v_rel_t / (1.0 / m1 + 1.0 / m2)

            J = impulse_n + impulse_t

            J_sum += J

    return valid_count, J_sum


#pyh this is getting updated since resting_collision_pair is single instance now
@wp.kernel(enable_backward=False)
def update_potential_collision_restmap(
    x: wp.array(dtype=wp.vec3),
    masks: wp.array(dtype=wp.int32),
    collision_dist: float,
    grid: wp.uint64,
    resting_collision_pairs: wp.array2d(dtype=wp.bool),
    object_massnode_single: int,
    collision_indices: wp.array2d(dtype=wp.int32),
    collision_number: wp.array(dtype=wp.int32),
):
    tid = wp.tid()
    
    # order threads by cell
    i = wp.hash_grid_point_id(grid, tid)

    #pyh added
    inst_i = i//object_massnode_single
    local_i = i - inst_i * object_massnode_single

    x1 = x[i]
    #pyh changed
    mask1 = masks[local_i]

    neighbors = wp.hash_grid_query(grid, x1, collision_dist * 5.0)

    for neighbor_index in neighbors:
        if neighbor_index != i:
            inst_neighbor = neighbor_index // object_massnode_single
            #if they are not from same instance we also skip 
            if inst_neighbor != inst_i:
                continue
            local_neighbor = neighbor_index - inst_neighbor * object_massnode_single

            if resting_collision_pairs[local_i][local_neighbor] == True or resting_collision_pairs[local_neighbor][local_i] == True:
                continue    
            x2 = x[neighbor_index]        
            mask2 = masks[local_neighbor]

            dis = x2 - x1
            dis_len = wp.length(dis)
            # If the distance is less than the collision distance and the two points are moving towards each other
            if mask1 != mask2 and dis_len < collision_dist:
                collision_indices[i][collision_number[i]] = neighbor_index
                collision_number[i] += 1

@wp.kernel
def object_collision(
    x: wp.array(dtype=wp.vec3),
    v: wp.array(dtype=wp.vec3),
    masses: wp.array(dtype=wp.float32),
    masks: wp.array(dtype=wp.int32),
    collide_object_elas: wp.array(dtype=float),
    collide_object_fric: wp.array(dtype=float),
    collision_dist: float,
    collision_indices: wp.array2d(dtype=wp.int32),
    collision_number: wp.array(dtype=wp.int32),
    object_massnode_single: int,
    v_new: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()

    local_idx = 0
    inst = tid // object_massnode_single
    offset = inst * object_massnode_single
    local_idx = tid - offset

    v1 = v[tid]
    m1 = masses[local_idx]

    clamp_collide_object_elas = wp.clamp(collide_object_elas[0], low=0.0, high=1.0)
    clamp_collide_object_fric = wp.clamp(collide_object_fric[0], low=0.0, high=2.0)

    valid_count, J_sum = loop(
        tid,
        collision_indices,
        collision_number,
        x,
        v,
        masses,
        masks,
        collision_dist,
        clamp_collide_object_elas,
        clamp_collide_object_fric,
        m1,
        offset,
    )

    if valid_count > 0:
        J_average = J_sum / valid_count
        v_new[tid] = v1 - J_average / m1
    else:
        v_new[tid] = v1

#pyh this is changing to batched version
@wp.kernel(enable_backward=False)
def build_resting_collision_pairs(
    x: wp.array(dtype=wp.vec3),
    rest_exclusion_dist: float,
    grid: wp.uint64,
    resting_collision_pairs: wp.array2d(dtype=wp.bool),
):

    tid = wp.tid()

    # order threads by cell
    i = wp.hash_grid_point_id(grid, tid)

    x1 = x[i]

    neighbors = wp.hash_grid_query(grid, x1, rest_exclusion_dist)
    for index in neighbors:
        if index < i:
            # Preserve PhysTwin's trained behavior: the rest map contains the
            # hash-query candidates, not a second exact-distance-filtered set.
            resting_collision_pairs[i][index] = wp.bool(1)
            resting_collision_pairs[index][i] = wp.bool(1)

@wp.func
def apply_surface_collision_response(
    velocity: wp.vec3,
    normal: wp.vec3,
    clamp_collide_elas: float,
    clamp_collide_fric: float,
):
    v_normal = wp.dot(velocity, normal) * normal
    v_tangent = velocity - v_normal
    v_normal_length = wp.length(v_normal)
    v_tangent_length = wp.max(wp.length(v_tangent), 1e-6)
    v_normal_new = -clamp_collide_elas * v_normal
    tangent_scale = wp.max(
        0.0,
        1.0
        - clamp_collide_fric
        * (1.0 + clamp_collide_elas)
        * v_normal_length
        / v_tangent_length,
    )
    v_tangent_new = tangent_scale * v_tangent
    return v_normal_new + v_tangent_new


@wp.func
def sample_static_surface_heightfield(
    offsets: wp.array(dtype=float),
    offset_start: int,
    cells_u: int,
    cells_v: int,
    extent_u: float,
    extent_v: float,
    local_u: float,
    local_v: float,
):
    """Return height and local derivatives for a triangulated heightfield."""

    normalized_u = wp.clamp(
        (local_u + extent_u) / (2.0 * extent_u),
        0.0,
        1.0,
    )
    normalized_v = wp.clamp(
        (local_v + extent_v) / (2.0 * extent_v),
        0.0,
        1.0,
    )
    scaled_u = normalized_u * float(cells_u)
    scaled_v = normalized_v * float(cells_v)
    cell_u = wp.min(int(scaled_u), cells_u - 1)
    cell_v = wp.min(int(scaled_v), cells_v - 1)
    fraction_u = scaled_u - float(cell_u)
    fraction_v = scaled_v - float(cell_v)
    row_stride = cells_u + 1
    index_00 = offset_start + cell_v * row_stride + cell_u
    height_00 = offsets[index_00]
    height_10 = offsets[index_00 + 1]
    height_01 = offsets[index_00 + row_stride]
    height_11 = offsets[index_00 + row_stride + 1]

    height = float(0.0)
    derivative_fraction_u = float(0.0)
    derivative_fraction_v = float(0.0)
    if (cell_u + cell_v) % 2 == 0:
        if fraction_v <= fraction_u:
            height = (
                height_00
                + fraction_u * (height_10 - height_00)
                + fraction_v * (height_11 - height_10)
            )
            derivative_fraction_u = height_10 - height_00
            derivative_fraction_v = height_11 - height_10
        else:
            height = (
                height_00
                + fraction_u * (height_11 - height_01)
                + fraction_v * (height_01 - height_00)
            )
            derivative_fraction_u = height_11 - height_01
            derivative_fraction_v = height_01 - height_00
    else:
        if fraction_u + fraction_v <= 1.0:
            height = (
                height_00
                + fraction_u * (height_10 - height_00)
                + fraction_v * (height_01 - height_00)
            )
            derivative_fraction_u = height_10 - height_00
            derivative_fraction_v = height_01 - height_00
        else:
            height = (
                height_11
                + (1.0 - fraction_u) * (height_01 - height_11)
                + (1.0 - fraction_v) * (height_10 - height_11)
            )
            derivative_fraction_u = height_11 - height_01
            derivative_fraction_v = height_11 - height_10

    derivative_u = derivative_fraction_u * float(cells_u) / (2.0 * extent_u)
    derivative_v = derivative_fraction_v * float(cells_v) / (2.0 * extent_v)
    return wp.vec3(height, derivative_u, derivative_v)


@wp.func
def rounded_capsule_heightfield_contact(
    offsets: wp.array(dtype=float),
    offset_start: int,
    cells_u: int,
    cells_v: int,
    surface_normal: wp.vec3,
    surface_axis_u: wp.vec3,
    surface_axis_v: wp.vec3,
    surface_extent_u: float,
    surface_extent_v: float,
    edge_radius: float,
    relative_position: wp.vec3,
):
    """Return signed distance and normal for a crowned rounded capsule top."""

    local_u = wp.dot(relative_position, surface_axis_u)
    local_v = wp.dot(relative_position, surface_axis_v)
    axial_distance = wp.dot(relative_position, surface_normal)
    height_sample = sample_static_surface_heightfield(
        offsets,
        offset_start,
        cells_u,
        cells_v,
        surface_extent_u,
        surface_extent_v,
        local_u,
        local_v,
    )
    height_distance = axial_distance - height_sample[0]
    top_normal = wp.normalize(
        surface_normal
        - surface_axis_u * height_sample[1]
        - surface_axis_v * height_sample[2]
    )

    capsule_radius = surface_extent_v
    capsule_spine_half_length = wp.max(
        surface_extent_u - capsule_radius,
        0.0,
    )
    closest_spine_u = wp.clamp(
        local_u,
        -capsule_spine_half_length,
        capsule_spine_half_length,
    )
    radial_u = local_u - closest_spine_u
    radial_v = local_v
    radial_length = wp.sqrt(radial_u * radial_u + radial_v * radial_v)
    outward_normal = surface_axis_v
    if radial_length > 1.0e-8:
        outward_normal = (
            surface_axis_u * (radial_u / radial_length)
            + surface_axis_v * (radial_v / radial_length)
        )

    footprint_distance = radial_length - capsule_radius
    rounded_radius = wp.clamp(
        edge_radius,
        1.0e-5,
        0.5 * capsule_radius,
    )
    corner_u = footprint_distance + rounded_radius
    corner_h = height_distance + rounded_radius
    positive_u = wp.max(corner_u, 0.0)
    positive_h = wp.max(corner_h, 0.0)
    outside_length = wp.sqrt(
        positive_u * positive_u + positive_h * positive_h
    )
    signed_distance = (
        wp.min(wp.max(corner_u, corner_h), 0.0)
        + outside_length
        - rounded_radius
    )

    contact_normal = top_normal
    if corner_u > 0.0 and corner_h > 0.0:
        contact_normal = wp.normalize(
            outward_normal * corner_u + top_normal * corner_h
        )
    elif corner_u > corner_h:
        contact_normal = outward_normal
    return wp.vec4(
        signed_distance,
        contact_normal[0],
        contact_normal[1],
        contact_normal[2],
    )


@wp.kernel
def integrate_ground_collision(
    x: wp.array(dtype=wp.vec3),
    v: wp.array(dtype=wp.vec3),
    collide_elas: wp.array(dtype=float),
    collide_fric: wp.array(dtype=float),
    dt: float,
    reverse_factor: float,
    use_ground_plane: int,
    static_box_mins: wp.array(dtype=wp.vec3),
    static_box_maxs: wp.array(dtype=wp.vec3),
    static_box_count: int,
    static_surface_centers: wp.array(dtype=wp.vec3),
    static_surface_normals: wp.array(dtype=wp.vec3),
    static_surface_axes_u: wp.array(dtype=wp.vec3),
    static_surface_axes_v: wp.array(dtype=wp.vec3),
    static_surface_extents_u: wp.array(dtype=float),
    static_surface_extents_v: wp.array(dtype=float),
    static_surface_kinds: wp.array(dtype=wp.int32),
    static_surface_edge_radii: wp.array(dtype=float),
    static_surface_heightfield_offsets: wp.array(dtype=float),
    static_surface_heightfield_starts: wp.array(dtype=wp.int32),
    static_surface_heightfield_cells_u: wp.array(dtype=wp.int32),
    static_surface_heightfield_cells_v: wp.array(dtype=wp.int32),
    static_surface_count: int,
    static_surface_query_distance: float,
    static_surface_margin: float,
    static_surface_restitution: float,
    static_surface_friction: float,
    static_mesh_enabled: int,
    static_mesh_sweep_start: wp.array(dtype=wp.vec3),
    static_mesh: wp.uint64,
    static_mesh_two_sided: int,
    static_mesh_component_mins: wp.array(dtype=wp.vec3),
    static_mesh_component_maxs: wp.array(dtype=wp.vec3),
    static_mesh_component_count: int,
    static_mesh_query_distance: float,
    static_mesh_winding_accuracy: float,
    static_mesh_winding_threshold: float,
    static_mesh_margin: float,
    static_mesh_restitution: float,
    static_mesh_friction: float,
    x_new: wp.array(dtype=wp.vec3),
    v_new: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()

    x0 = x[tid]
    v0 = v[tid]

    clamp_collide_elas = wp.clamp(collide_elas[0], low=0.0, high=1.0)
    clamp_collide_fric = wp.clamp(collide_fric[0], low=0.0, high=2.0)
    normal = wp.vec3(0.0, 0.0, 1.0) * reverse_factor
    collision_eps = float(1.0e-4)

    x_result = x0 + v0 * dt
    v_result = v0

    if use_ground_plane != 0:
        x_z = x0[2]
        v_z = v0[2]
        next_x_z = (x_z + v_z * dt) * reverse_factor
        if next_x_z < 0.0 and v_z * reverse_factor < -1e-4:
            v_after = apply_surface_collision_response(
                v0,
                normal,
                clamp_collide_elas,
                clamp_collide_fric,
            )
            toi = -x_z / v_z
            x_result = x0 + v0 * toi + v_after * (dt - toi)
            v_result = v_after

    best_hit_t = float(2.0)
    best_hit_normal = wp.vec3(0.0, 0.0, 0.0)
    has_sweep_hit = int(0)
    best_inside_depth = float(1.0e8)
    best_inside_normal = wp.vec3(0.0, 0.0, 0.0)
    best_inside_projected = x0
    has_inside_hit = int(0)
    displacement = v0 * dt

    for box_index in range(static_box_count):
        box_min = static_box_mins[box_index]
        box_max = static_box_maxs[box_index]
        inside_box = (
            x0[0] >= box_min[0]
            and x0[0] <= box_max[0]
            and x0[1] >= box_min[1]
            and x0[1] <= box_max[1]
            and x0[2] >= box_min[2]
            and x0[2] <= box_max[2]
        )
        if inside_box:
            dist_x_min = x0[0] - box_min[0]
            dist_x_max = box_max[0] - x0[0]
            dist_y_min = x0[1] - box_min[1]
            dist_y_max = box_max[1] - x0[1]
            dist_z_min = x0[2] - box_min[2]
            dist_z_max = box_max[2] - x0[2]

            nearest_dist = dist_x_min
            inside_normal = wp.vec3(-1.0, 0.0, 0.0)
            inside_projected = wp.vec3(box_min[0] - collision_eps, x0[1], x0[2])

            if dist_x_max < nearest_dist:
                nearest_dist = dist_x_max
                inside_normal = wp.vec3(1.0, 0.0, 0.0)
                inside_projected = wp.vec3(box_max[0] + collision_eps, x0[1], x0[2])
            if dist_y_min < nearest_dist:
                nearest_dist = dist_y_min
                inside_normal = wp.vec3(0.0, -1.0, 0.0)
                inside_projected = wp.vec3(x0[0], box_min[1] - collision_eps, x0[2])
            if dist_y_max < nearest_dist:
                nearest_dist = dist_y_max
                inside_normal = wp.vec3(0.0, 1.0, 0.0)
                inside_projected = wp.vec3(x0[0], box_max[1] + collision_eps, x0[2])
            if dist_z_min < nearest_dist:
                nearest_dist = dist_z_min
                inside_normal = wp.vec3(0.0, 0.0, -1.0)
                inside_projected = wp.vec3(x0[0], x0[1], box_min[2] - collision_eps)
            if dist_z_max < nearest_dist:
                nearest_dist = dist_z_max
                inside_normal = wp.vec3(0.0, 0.0, 1.0)
                inside_projected = wp.vec3(x0[0], x0[1], box_max[2] + collision_eps)

            if has_inside_hit == 0 or nearest_dist < best_inside_depth:
                best_inside_depth = nearest_dist
                best_inside_normal = inside_normal
                best_inside_projected = inside_projected
                has_inside_hit = int(1)
        else:
            t_enter = float(0.0)
            t_exit = float(1.0)
            candidate_normal = wp.vec3(0.0, 0.0, 0.0)
            valid_hit = int(1)

            d0 = displacement[0]
            if wp.abs(d0) < collision_eps:
                if x0[0] < box_min[0] or x0[0] > box_max[0]:
                    valid_hit = int(0)
            else:
                inv_d0 = 1.0 / d0
                t0a = (box_min[0] - x0[0]) * inv_d0
                t0b = (box_max[0] - x0[0]) * inv_d0
                near_t0 = wp.min(t0a, t0b)
                far_t0 = wp.max(t0a, t0b)
                if near_t0 > t_enter:
                    t_enter = near_t0
                    if t0a < t0b:
                        candidate_normal = wp.vec3(-1.0, 0.0, 0.0)
                    else:
                        candidate_normal = wp.vec3(1.0, 0.0, 0.0)
                t_exit = wp.min(t_exit, far_t0)
                if t_enter > t_exit:
                    valid_hit = int(0)

            d1 = displacement[1]
            if valid_hit != 0:
                if wp.abs(d1) < collision_eps:
                    if x0[1] < box_min[1] or x0[1] > box_max[1]:
                        valid_hit = int(0)
                else:
                    inv_d1 = 1.0 / d1
                    t1a = (box_min[1] - x0[1]) * inv_d1
                    t1b = (box_max[1] - x0[1]) * inv_d1
                    near_t1 = wp.min(t1a, t1b)
                    far_t1 = wp.max(t1a, t1b)
                    if near_t1 > t_enter:
                        t_enter = near_t1
                        if t1a < t1b:
                            candidate_normal = wp.vec3(0.0, -1.0, 0.0)
                        else:
                            candidate_normal = wp.vec3(0.0, 1.0, 0.0)
                    t_exit = wp.min(t_exit, far_t1)
                    if t_enter > t_exit:
                        valid_hit = int(0)

            d2 = displacement[2]
            if valid_hit != 0:
                if wp.abs(d2) < collision_eps:
                    if x0[2] < box_min[2] or x0[2] > box_max[2]:
                        valid_hit = int(0)
                else:
                    inv_d2 = 1.0 / d2
                    t2a = (box_min[2] - x0[2]) * inv_d2
                    t2b = (box_max[2] - x0[2]) * inv_d2
                    near_t2 = wp.min(t2a, t2b)
                    far_t2 = wp.max(t2a, t2b)
                    if near_t2 > t_enter:
                        t_enter = near_t2
                        if t2a < t2b:
                            candidate_normal = wp.vec3(0.0, 0.0, -1.0)
                        else:
                            candidate_normal = wp.vec3(0.0, 0.0, 1.0)
                    t_exit = wp.min(t_exit, far_t2)
                    if t_enter > t_exit:
                        valid_hit = int(0)

            if (
                valid_hit != 0
                and t_enter >= 0.0
                and t_enter <= 1.0
                and wp.dot(v0, candidate_normal) < -1e-4
                and t_enter < best_hit_t
            ):
                best_hit_t = t_enter
                best_hit_normal = candidate_normal
                has_sweep_hit = int(1)

    # Fast finite primitives are used by Garden and Ambulance. Sweeps prevent
    # tunnelling, while the bounded recovery path resolves resting penetration.
    # kind=0 is a local rectangle, kind=1 is a top-only disk, kind=2 is a finite
    # cylinder, kind=3 is a rectangular heightfield, kind=4 is the same
    # heightfield clipped to a fourth-power superellipse, kind=5 clips the
    # heightfield to a long-axis capsule, and kind=6 gives that capsule a
    # continuous padded roll-off instead of a hard top-only edge.
    best_surface_t = float(2.0)
    best_surface_normal = wp.vec3(0.0, 0.0, 0.0)
    has_surface_sweep = int(0)
    best_surface_penetration = float(1.0e8)
    best_surface_recovery_normal = wp.vec3(0.0, 0.0, 0.0)
    has_surface_recovery = int(0)
    surface_motion = x_result - x0
    surface_margin = wp.max(static_surface_margin, 0.0)
    surface_query_distance = wp.max(static_surface_query_distance, surface_margin)

    for surface_index in range(static_surface_count):
        surface_center = static_surface_centers[surface_index]
        surface_normal = static_surface_normals[surface_index]
        surface_axis_u = static_surface_axes_u[surface_index]
        surface_axis_v = static_surface_axes_v[surface_index]
        surface_extent_u = static_surface_extents_u[surface_index]
        surface_extent_v = static_surface_extents_v[surface_index]
        surface_kind = static_surface_kinds[surface_index]
        surface_edge_radius = static_surface_edge_radii[surface_index]

        previous_relative = x0 - surface_center
        candidate_relative = x_result - surface_center
        previous_u = wp.dot(previous_relative, surface_axis_u)
        previous_v = wp.dot(previous_relative, surface_axis_v)
        candidate_u = wp.dot(candidate_relative, surface_axis_u)
        candidate_v = wp.dot(candidate_relative, surface_axis_v)
        previous_distance = wp.dot(previous_relative, surface_normal)
        candidate_distance = wp.dot(candidate_relative, surface_normal)
        candidate_surface_normal = surface_normal
        if surface_kind == 6:
            previous_contact = rounded_capsule_heightfield_contact(
                static_surface_heightfield_offsets,
                static_surface_heightfield_starts[surface_index],
                static_surface_heightfield_cells_u[surface_index],
                static_surface_heightfield_cells_v[surface_index],
                surface_normal,
                surface_axis_u,
                surface_axis_v,
                surface_extent_u,
                surface_extent_v,
                surface_edge_radius,
                previous_relative,
            )
            candidate_contact = rounded_capsule_heightfield_contact(
                static_surface_heightfield_offsets,
                static_surface_heightfield_starts[surface_index],
                static_surface_heightfield_cells_u[surface_index],
                static_surface_heightfield_cells_v[surface_index],
                surface_normal,
                surface_axis_u,
                surface_axis_v,
                surface_extent_u,
                surface_extent_v,
                surface_edge_radius,
                candidate_relative,
            )
            previous_distance = previous_contact[0]
            candidate_distance = candidate_contact[0]
            candidate_surface_normal = wp.vec3(
                candidate_contact[1],
                candidate_contact[2],
                candidate_contact[3],
            )
        elif surface_kind == 3 or surface_kind == 4 or surface_kind == 5:
            heightfield_start = static_surface_heightfield_starts[surface_index]
            heightfield_cells_u = static_surface_heightfield_cells_u[surface_index]
            heightfield_cells_v = static_surface_heightfield_cells_v[surface_index]
            previous_sample = sample_static_surface_heightfield(
                static_surface_heightfield_offsets,
                heightfield_start,
                heightfield_cells_u,
                heightfield_cells_v,
                surface_extent_u,
                surface_extent_v,
                previous_u,
                previous_v,
            )
            candidate_sample = sample_static_surface_heightfield(
                static_surface_heightfield_offsets,
                heightfield_start,
                heightfield_cells_u,
                heightfield_cells_v,
                surface_extent_u,
                surface_extent_v,
                candidate_u,
                candidate_v,
            )
            previous_distance = previous_distance - previous_sample[0]
            candidate_distance = candidate_distance - candidate_sample[0]
            candidate_surface_normal = wp.normalize(
                surface_normal
                - surface_axis_u * candidate_sample[1]
                - surface_axis_v * candidate_sample[2]
            )
        distance_delta = previous_distance - candidate_distance
        surface_sweep_detected = int(0)
        rounded_bracketed_sweep = int(0)
        sweep_lower_hit_t = float(0.0)
        sweep_upper_hit_t = float(1.0)
        if (
            previous_distance >= surface_margin
            and candidate_distance < surface_margin
            and distance_delta > collision_eps
        ):
            surface_sweep_detected = int(1)

        if (
            surface_kind == 6
            and surface_sweep_detected == 0
            and previous_distance >= surface_margin
            and wp.length(surface_motion) > 0.5 * surface_edge_radius
            and wp.min(previous_u, candidate_u)
            <= surface_extent_u + surface_edge_radius
            and wp.max(previous_u, candidate_u)
            >= -surface_extent_u - surface_edge_radius
            and wp.min(previous_v, candidate_v)
            <= surface_extent_v + surface_edge_radius
            and wp.max(previous_v, candidate_v)
            >= -surface_extent_v - surface_edge_radius
        ):
            previous_probe_distance = previous_distance
            previous_probe_t = float(0.0)
            for edge_probe_index in range(1, 9):
                edge_probe_t = float(edge_probe_index) / 8.0
                edge_probe_relative = (
                    x0 + surface_motion * edge_probe_t - surface_center
                )
                edge_probe_contact = rounded_capsule_heightfield_contact(
                    static_surface_heightfield_offsets,
                    static_surface_heightfield_starts[surface_index],
                    static_surface_heightfield_cells_u[surface_index],
                    static_surface_heightfield_cells_v[surface_index],
                    surface_normal,
                    surface_axis_u,
                    surface_axis_v,
                    surface_extent_u,
                    surface_extent_v,
                    surface_edge_radius,
                    edge_probe_relative,
                )
                if (
                    surface_sweep_detected == 0
                    and previous_probe_distance >= surface_margin
                    and edge_probe_contact[0] < surface_margin
                ):
                    surface_sweep_detected = int(1)
                    rounded_bracketed_sweep = int(1)
                    sweep_lower_hit_t = previous_probe_t
                    sweep_upper_hit_t = edge_probe_t
                previous_probe_distance = edge_probe_contact[0]
                previous_probe_t = edge_probe_t

        if surface_sweep_detected != 0:
            hit_t = 0.5 * (sweep_lower_hit_t + sweep_upper_hit_t)
            if rounded_bracketed_sweep == 0:
                hit_t = (previous_distance - surface_margin) / distance_delta
            if surface_kind == 6:
                # The rounded edge distance is nonlinear along a fast sweep.
                # Refine its time of impact so a vertical drop cannot be left
                # embedded in the curved roll-off by linear interpolation.
                capsule_spine_half_length = wp.max(
                    surface_extent_u - surface_extent_v,
                    0.0,
                )
                previous_spine_u = wp.clamp(
                    previous_u,
                    -capsule_spine_half_length,
                    capsule_spine_half_length,
                )
                candidate_spine_u = wp.clamp(
                    candidate_u,
                    -capsule_spine_half_length,
                    capsule_spine_half_length,
                )
                previous_radial_u = previous_u - previous_spine_u
                candidate_radial_u = candidate_u - candidate_spine_u
                previous_footprint_distance = (
                    wp.sqrt(
                        previous_radial_u * previous_radial_u
                        + previous_v * previous_v
                    )
                    - surface_extent_v
                )
                candidate_footprint_distance = (
                    wp.sqrt(
                        candidate_radial_u * candidate_radial_u
                        + candidate_v * candidate_v
                    )
                    - surface_extent_v
                )
                if (
                    rounded_bracketed_sweep != 0
                    or previous_footprint_distance > -surface_edge_radius
                    or candidate_footprint_distance > -surface_edge_radius
                ):
                    lower_hit_t = sweep_lower_hit_t
                    upper_hit_t = sweep_upper_hit_t
                    for _edge_sweep_iteration in range(8):
                        probe_hit_t = 0.5 * (lower_hit_t + upper_hit_t)
                        probe_relative = (
                            x0 + surface_motion * probe_hit_t - surface_center
                        )
                        probe_contact = rounded_capsule_heightfield_contact(
                            static_surface_heightfield_offsets,
                            static_surface_heightfield_starts[surface_index],
                            static_surface_heightfield_cells_u[surface_index],
                            static_surface_heightfield_cells_v[surface_index],
                            surface_normal,
                            surface_axis_u,
                            surface_axis_v,
                            surface_extent_u,
                            surface_extent_v,
                            surface_edge_radius,
                            probe_relative,
                        )
                        if probe_contact[0] >= surface_margin:
                            lower_hit_t = probe_hit_t
                        else:
                            upper_hit_t = probe_hit_t
                    hit_t = 0.5 * (lower_hit_t + upper_hit_t)
            if hit_t >= 0.0 and hit_t <= 1.0:
                hit_point = x0 + surface_motion * hit_t
                hit_relative = hit_point - surface_center
                hit_u = wp.dot(hit_relative, surface_axis_u)
                hit_v = wp.dot(hit_relative, surface_axis_v)
                inside_footprint = int(0)
                hit_normal = surface_normal
                if surface_kind == 6:
                    hit_contact = rounded_capsule_heightfield_contact(
                        static_surface_heightfield_offsets,
                        static_surface_heightfield_starts[surface_index],
                        static_surface_heightfield_cells_u[surface_index],
                        static_surface_heightfield_cells_v[surface_index],
                        surface_normal,
                        surface_axis_u,
                        surface_axis_v,
                        surface_extent_u,
                        surface_extent_v,
                        surface_edge_radius,
                        hit_relative,
                    )
                    hit_normal = wp.vec3(
                        hit_contact[1],
                        hit_contact[2],
                        hit_contact[3],
                    )
                    inside_footprint = int(1)
                elif surface_kind == 3 or surface_kind == 4 or surface_kind == 5:
                    hit_sample = sample_static_surface_heightfield(
                        static_surface_heightfield_offsets,
                        static_surface_heightfield_starts[surface_index],
                        static_surface_heightfield_cells_u[surface_index],
                        static_surface_heightfield_cells_v[surface_index],
                        surface_extent_u,
                        surface_extent_v,
                        hit_u,
                        hit_v,
                    )
                    hit_normal = wp.normalize(
                        surface_normal
                        - surface_axis_u * hit_sample[1]
                        - surface_axis_v * hit_sample[2]
                    )
                if surface_kind == 4:
                    normalized_hit_u = hit_u / surface_extent_u
                    normalized_hit_v = hit_v / surface_extent_v
                    normalized_hit_u_sq = normalized_hit_u * normalized_hit_u
                    normalized_hit_v_sq = normalized_hit_v * normalized_hit_v
                    if (
                        normalized_hit_u_sq * normalized_hit_u_sq
                        + normalized_hit_v_sq * normalized_hit_v_sq
                        <= 1.0
                    ):
                        inside_footprint = int(1)
                elif surface_kind == 5:
                    capsule_radius = surface_extent_v
                    capsule_spine_half_length = wp.max(
                        surface_extent_u - capsule_radius,
                        0.0,
                    )
                    capsule_end_u = wp.max(
                        wp.abs(hit_u) - capsule_spine_half_length,
                        0.0,
                    )
                    if (
                        capsule_end_u * capsule_end_u + hit_v * hit_v
                        <= capsule_radius * capsule_radius
                    ):
                        inside_footprint = int(1)
                elif surface_kind == 1 or surface_kind == 2:
                    footprint_radius = surface_extent_u
                    if surface_kind == 2:
                        footprint_radius = footprint_radius + surface_margin
                    if (
                        hit_u * hit_u + hit_v * hit_v
                        <= footprint_radius * footprint_radius
                    ):
                        inside_footprint = int(1)
                elif (
                    wp.abs(hit_u) <= surface_extent_u
                    and wp.abs(hit_v) <= surface_extent_v
                ):
                    inside_footprint = int(1)
                if inside_footprint != 0 and hit_t < best_surface_t:
                    best_surface_t = hit_t
                    best_surface_normal = hit_normal
                    has_surface_sweep = int(1)

        if surface_kind == 2:
            # The tabletop is a closed finite cylinder. The side sweep catches
            # fast lateral motion at the rim; the bottom sweep prevents a rope
            # hanging over the edge from swinging through the underside.
            cylinder_radius = surface_extent_u
            cylinder_depth = surface_extent_v
            expanded_radius = cylinder_radius + surface_margin
            bottom_distance = -cylinder_depth - surface_margin
            candidate_radius_sq = (
                candidate_u * candidate_u + candidate_v * candidate_v
            )

            distance_up = candidate_distance - previous_distance
            if (
                previous_distance <= bottom_distance
                and candidate_distance > bottom_distance
                and distance_up > collision_eps
            ):
                bottom_hit_t = (
                    bottom_distance - previous_distance
                ) / distance_up
                if bottom_hit_t >= 0.0 and bottom_hit_t <= 1.0:
                    bottom_hit = x0 + surface_motion * bottom_hit_t
                    bottom_relative = bottom_hit - surface_center
                    bottom_u = wp.dot(bottom_relative, surface_axis_u)
                    bottom_v = wp.dot(bottom_relative, surface_axis_v)
                    if (
                        bottom_u * bottom_u + bottom_v * bottom_v
                        <= expanded_radius * expanded_radius
                        and bottom_hit_t < best_surface_t
                    ):
                        best_surface_t = bottom_hit_t
                        best_surface_normal = -surface_normal
                        has_surface_sweep = int(1)

            previous_relative = x0 - surface_center
            previous_u = wp.dot(previous_relative, surface_axis_u)
            previous_v = wp.dot(previous_relative, surface_axis_v)
            motion_u = candidate_u - previous_u
            motion_v = candidate_v - previous_v
            radial_a = motion_u * motion_u + motion_v * motion_v
            radial_b = 2.0 * (
                previous_u * motion_u + previous_v * motion_v
            )
            radial_c = (
                previous_u * previous_u
                + previous_v * previous_v
                - expanded_radius * expanded_radius
            )
            if radial_a > collision_eps and radial_c >= 0.0:
                discriminant = radial_b * radial_b - 4.0 * radial_a * radial_c
                if discriminant >= 0.0:
                    side_hit_t = (
                        -radial_b - wp.sqrt(discriminant)
                    ) / (2.0 * radial_a)
                    if side_hit_t >= 0.0 and side_hit_t <= 1.0:
                        side_hit_distance = previous_distance + (
                            candidate_distance - previous_distance
                        ) * side_hit_t
                        if (
                            side_hit_distance <= surface_margin
                            and side_hit_distance >= bottom_distance
                        ):
                            side_u = previous_u + motion_u * side_hit_t
                            side_v = previous_v + motion_v * side_hit_t
                            side_radius = wp.sqrt(
                                side_u * side_u + side_v * side_v
                            )
                            if side_radius > collision_eps:
                                side_normal = (
                                    surface_axis_u * (side_u / side_radius)
                                    + surface_axis_v * (side_v / side_radius)
                                )
                                if (
                                    wp.dot(surface_motion, side_normal)
                                    < -collision_eps
                                    and side_hit_t < best_surface_t
                                ):
                                    best_surface_t = side_hit_t
                                    best_surface_normal = side_normal
                                    has_surface_sweep = int(1)

            # Signed distance and gradient for a capped cylinder give a smooth
            # rounded contact normal just outside the sharp top/bottom rims.
            # Inside the solid, they select the nearest exit face, avoiding the
            # large topward corrections that destabilized a hanging rope.
            candidate_radius = wp.sqrt(candidate_radius_sq)
            radial_direction = surface_axis_u
            if candidate_radius > collision_eps:
                radial_direction = (
                    surface_axis_u * (candidate_u / candidate_radius)
                    + surface_axis_v * (candidate_v / candidate_radius)
                )
            half_depth = 0.5 * cylinder_depth
            axial_from_middle = candidate_distance + half_depth
            axial_sign = float(1.0)
            if axial_from_middle < 0.0:
                axial_sign = -1.0
            radial_q = candidate_radius - cylinder_radius
            axial_q = wp.abs(axial_from_middle) - half_depth
            cylinder_signed_distance = float(0.0)
            cylinder_normal = surface_normal * axial_sign
            if radial_q > 0.0 and axial_q > 0.0:
                corner_distance = wp.sqrt(
                    radial_q * radial_q + axial_q * axial_q
                )
                cylinder_signed_distance = corner_distance
                if corner_distance > collision_eps:
                    cylinder_normal = (
                        radial_direction * (radial_q / corner_distance)
                        + surface_normal
                        * (axial_sign * axial_q / corner_distance)
                    )
            elif radial_q > axial_q:
                cylinder_signed_distance = radial_q
                cylinder_normal = radial_direction
            else:
                cylinder_signed_distance = axial_q
                cylinder_normal = surface_normal * axial_sign

            cylinder_penetration = surface_margin - cylinder_signed_distance
            if (
                cylinder_penetration > 0.0
                and cylinder_penetration <= surface_query_distance
                and cylinder_penetration < best_surface_penetration
            ):
                best_surface_penetration = cylinder_penetration
                best_surface_recovery_normal = cylinder_normal
                has_surface_recovery = int(1)
        else:
            penetration = surface_margin - candidate_distance
            if penetration > 0.0 and penetration <= surface_query_distance:
                inside_recovery_footprint = int(0)
                if surface_kind == 6:
                    inside_recovery_footprint = int(1)
                elif surface_kind == 4:
                    normalized_candidate_u = candidate_u / surface_extent_u
                    normalized_candidate_v = candidate_v / surface_extent_v
                    normalized_candidate_u_sq = (
                        normalized_candidate_u * normalized_candidate_u
                    )
                    normalized_candidate_v_sq = (
                        normalized_candidate_v * normalized_candidate_v
                    )
                    if (
                        normalized_candidate_u_sq * normalized_candidate_u_sq
                        + normalized_candidate_v_sq * normalized_candidate_v_sq
                        <= 1.0
                    ):
                        inside_recovery_footprint = int(1)
                elif surface_kind == 5:
                    capsule_radius = surface_extent_v
                    capsule_spine_half_length = wp.max(
                        surface_extent_u - capsule_radius,
                        0.0,
                    )
                    capsule_end_u = wp.max(
                        wp.abs(candidate_u) - capsule_spine_half_length,
                        0.0,
                    )
                    if (
                        capsule_end_u * capsule_end_u
                        + candidate_v * candidate_v
                        <= capsule_radius * capsule_radius
                    ):
                        inside_recovery_footprint = int(1)
                elif surface_kind == 1:
                    if (
                        candidate_u * candidate_u + candidate_v * candidate_v
                        <= surface_extent_u * surface_extent_u
                    ):
                        inside_recovery_footprint = int(1)
                elif (
                    wp.abs(candidate_u) <= surface_extent_u
                    and wp.abs(candidate_v) <= surface_extent_v
                ):
                    inside_recovery_footprint = int(1)
                if (
                    inside_recovery_footprint != 0
                    and penetration < best_surface_penetration
                ):
                    best_surface_penetration = penetration
                    best_surface_recovery_normal = candidate_surface_normal
                    has_surface_recovery = int(1)

    surface_restitution = wp.clamp(static_surface_restitution, 0.0, 1.0)
    surface_friction = wp.clamp(static_surface_friction, 0.0, 2.0)
    if has_inside_hit != 0:
        v_after = v0
        if wp.dot(v0, best_inside_normal) < -1e-4:
            v_after = apply_surface_collision_response(
                v0,
                best_inside_normal,
                clamp_collide_elas,
                clamp_collide_fric,
            )
        x_result = best_inside_projected + v_after * dt
        v_result = v_after
    elif has_surface_sweep != 0 or has_sweep_hit != 0:
        if (
            has_surface_sweep != 0
            and (has_sweep_hit == 0 or best_surface_t <= best_hit_t)
        ):
            v_after = v_result
            if wp.dot(v_result, best_surface_normal) < -1.0e-4:
                v_after = apply_surface_collision_response(
                    v_result,
                    best_surface_normal,
                    surface_restitution,
                    surface_friction,
                )
            x_result = x0 + surface_motion * best_surface_t
            v_result = v_after
        else:
            toi = best_hit_t * dt
            hit_point = x0 + v0 * toi + best_hit_normal * collision_eps
            v_after = apply_surface_collision_response(
                v0,
                best_hit_normal,
                clamp_collide_elas,
                clamp_collide_fric,
            )
            x_result = hit_point + v_after * (dt - toi)
            v_result = v_after
    elif has_surface_recovery != 0:
        if wp.dot(v_result, best_surface_recovery_normal) < -1.0e-4:
            v_result = apply_surface_collision_response(
                v_result,
                best_surface_recovery_normal,
                surface_restitution,
                surface_friction,
            )
        x_result = (
            x_result
            + best_surface_recovery_normal * best_surface_penetration
        )

    # Triangle meshes handle detail that is not well represented by the fast
    # analytic surfaces. Closed meshes use signed winding; source-derived open
    # meshes can instead act as a two-sided shell with a small contact margin.
    mesh_broadphase_hit = int(0)
    if static_mesh_enabled != 0:
        # A costly source-mesh query may be scheduled less often than the
        # spring integrator.  Sweep from the state immediately following the
        # previous query so skipped substeps cannot tunnel through a thin
        # triangle shell.
        mesh_x0 = static_mesh_sweep_start[tid]
        motion_min = wp.vec3(
            wp.min(mesh_x0[0], x_result[0]),
            wp.min(mesh_x0[1], x_result[1]),
            wp.min(mesh_x0[2], x_result[2]),
        )
        motion_max = wp.vec3(
            wp.max(mesh_x0[0], x_result[0]),
            wp.max(mesh_x0[1], x_result[1]),
            wp.max(mesh_x0[2], x_result[2]),
        )
        broadphase_padding = wp.vec3(
            static_mesh_query_distance,
            static_mesh_query_distance,
            static_mesh_query_distance,
        )
        for component_index in range(static_mesh_component_count):
            component_min = (
                static_mesh_component_mins[component_index]
                - broadphase_padding
            )
            component_max = (
                static_mesh_component_maxs[component_index]
                + broadphase_padding
            )
            if (
                motion_max[0] >= component_min[0]
                and motion_min[0] <= component_max[0]
                and motion_max[1] >= component_min[1]
                and motion_min[1] <= component_max[1]
                and motion_max[2] >= component_min[2]
                and motion_min[2] <= component_max[2]
            ):
                mesh_broadphase_hit = int(1)

    if static_mesh_enabled != 0 and mesh_broadphase_hit != 0:
        mesh_restitution = wp.clamp(static_mesh_restitution, 0.0, 1.0)
        mesh_friction = wp.clamp(static_mesh_friction, 0.0, 2.0)
        resolved_mesh = int(0)
        mesh_x0 = static_mesh_sweep_start[tid]
        motion = x_result - mesh_x0
        motion_length = wp.length(motion)
        if motion_length > 1.0e-8:
            direction = motion / motion_length
            ray = wp.mesh_query_ray(
                static_mesh,
                mesh_x0,
                direction,
                motion_length,
            )
            if ray.result:
                surface_point = wp.mesh_eval_position(
                    static_mesh,
                    ray.face,
                    ray.u,
                    ray.v,
                )
                mesh_normal = wp.normalize(ray.normal)
                ray_accepted = int(0)
                if static_mesh_two_sided != 0:
                    ray_accepted = int(1)
                    if wp.dot(direction, mesh_normal) > 0.0:
                        mesh_normal = -mesh_normal
                elif wp.dot(direction, mesh_normal) < 0.0:
                    ray_accepted = int(1)
                if ray_accepted != 0:
                    if wp.dot(v_result, mesh_normal) < -1.0e-4:
                        v_result = apply_surface_collision_response(
                            v_result,
                            mesh_normal,
                            mesh_restitution,
                            mesh_friction,
                        )
                    x_result = surface_point + mesh_normal * static_mesh_margin
                    resolved_mesh = int(1)

        if resolved_mesh == 0:
            if static_mesh_two_sided != 0:
                shell_query = wp.mesh_query_point_no_sign(
                    static_mesh,
                    x_result,
                    max_dist=static_mesh_query_distance,
                )
                if shell_query.result:
                    surface_point = wp.mesh_eval_position(
                        static_mesh,
                        shell_query.face,
                        shell_query.u,
                        shell_query.v,
                    )
                    delta = x_result - surface_point
                    delta_length = wp.length(delta)
                    contact_error = delta_length - static_mesh_margin
                    if contact_error < 0.0:
                        mesh_normal = wp.normalize(
                            wp.mesh_eval_face_normal(
                                static_mesh,
                                shell_query.face,
                            )
                        )
                        if delta_length > 1.0e-8:
                            mesh_normal = delta / delta_length
                        elif wp.dot(v_result, mesh_normal) > 0.0:
                            mesh_normal = -mesh_normal
                        if wp.dot(v_result, mesh_normal) < -1.0e-4:
                            v_result = apply_surface_collision_response(
                                v_result,
                                mesh_normal,
                                mesh_restitution,
                                mesh_friction,
                            )
                        x_result = x_result + mesh_normal * (-contact_error)
            else:
                query = wp.mesh_query_point_sign_winding_number(
                    static_mesh,
                    x_result,
                    max_dist=static_mesh_query_distance,
                    accuracy=static_mesh_winding_accuracy,
                    threshold=static_mesh_winding_threshold,
                )
                if query.result:
                    surface_point = wp.mesh_eval_position(
                        static_mesh,
                        query.face,
                        query.u,
                        query.v,
                    )
                    delta = x_result - surface_point
                    delta_length = wp.length(delta)
                    signed_distance = delta_length * query.sign
                    contact_error = signed_distance - static_mesh_margin
                    if contact_error < 0.0 and delta_length > 1.0e-8:
                        # Negative sign means the candidate is inside the closed
                        # mesh, so this reverses delta into the outward normal.
                        mesh_normal = delta / delta_length * query.sign
                        if wp.dot(v_result, mesh_normal) < -1.0e-4:
                            v_result = apply_surface_collision_response(
                                v_result,
                                mesh_normal,
                                mesh_restitution,
                                mesh_friction,
                            )
                        x_result = x_result + mesh_normal * (-contact_error)

    x_new[tid] = x_result
    v_new[tid] = v_result

class SpringMassSystemWarp:
    def __init__(
        self,
        init_springs, #already in batch format
        init_rest_lengths, #shared 
        init_masses, #shared
        init_masks, #shared if provided
        # per instance var
        init_vertices, #already passed in batch format, require num_object_points to be set correctly to all object mass nodes
        init_velocities,
        #no changed needed
        dt, 
        num_substeps, 
        dashpot_damping,  
        drag_damping, 
        collision_dist, 
        reverse_z, 
        spring_Y_min, 
        spring_Y_max, 
        self_collision, 
        #updated variable
        collide_elas, 
        collide_fric, 
        collide_object_elas, 
        collide_object_fric, 
        spring_Y,
        #added
        object_massnodes_total, #original num_object_points
        object_massnodes_single,
        object_springs_single, 
        object_springs_total, 
        controller_massnodes_single, 
        controller_springs_single, 
        controller_rest_location, #replaces controller_points 
        number_of_instance,
        use_ground_plane=True,
        self_collision_rest_exclusion_multiplier=5.0,
    ):
        logger.info(f"[SIMULATION]: Initialize the Spring-Mass System")
        self.device = cfg.device

        #assigning single copies
        self.n_springs = object_springs_total + controller_springs_single * number_of_instance

        self.torch_springs = init_springs.contiguous()
        self.torch_rest_lengths = init_rest_lengths.contiguous()
        self.torch_inv_rest_lengths = (
            1.0 / self.torch_rest_lengths.clamp_min(1e-6)
        ).contiguous()

        self.wp_springs = wp.from_torch(
            self.torch_springs, dtype=wp.vec2i, requires_grad=False
        )

        self.wp_rest_lengths = wp.from_torch(
            self.torch_rest_lengths, dtype=wp.float32, requires_grad=False
        )
        self.wp_inv_rest_length = wp.from_torch(
            self.torch_inv_rest_lengths, dtype=wp.float32, requires_grad=False
        )
        
        #for testing eval_spring timing
        self.wp_spring_tmp = wp.zeros((self.n_springs,), dtype=wp.vec3, requires_grad=False)


        self.wp_masses = wp.from_torch(
            init_masses, dtype=wp.float32, requires_grad=False
        )
        self.wp_masks = None #only useful when self-collision is on
        if self_collision:
            if init_masks is None:
                print(f"Using default masks for self-collision")
                default_masks = torch.arange(
                    object_massnodes_single, dtype=torch.int32, device=self.device
                )            
                self.wp_masks = wp.from_torch(default_masks, dtype=wp.int32, requires_grad=False)
            else:
                print(f"Using provided masks for self-collision")
                assert init_masks.shape[0] == object_massnodes_single, "init_masks shape mismatch"
                self.wp_masks = wp.from_torch(
                    init_masks, dtype=wp.int32, requires_grad=False
                )

        #a large vector of all object mass nodes from all instances
        self.wp_init_vertices = wp.from_torch(
            init_vertices[:object_massnodes_total].contiguous(),
            dtype=wp.vec3,
            requires_grad=False,
        )

        #ones doesnt change 
        self.dt = dt
        self.num_substeps = num_substeps
        self.dashpot_damping = dashpot_damping
        self.drag_damping = drag_damping
        self.collision_dist = collision_dist
        self.self_collision_rest_exclusion_multiplier = float(
            self_collision_rest_exclusion_multiplier
        )
        if (
            not np.isfinite(self.self_collision_rest_exclusion_multiplier)
            or self.self_collision_rest_exclusion_multiplier < 1.0
        ):
            raise ValueError(
                "self_collision_rest_exclusion_multiplier must be finite and >= 1.0"
            )
        self.self_collision_rest_exclusion_distance = float(
            self.collision_dist
        ) * self.self_collision_rest_exclusion_multiplier
        self.reverse_factor = 1.0 if not reverse_z else -1.0
        self.spring_Y_min = spring_Y_min
        self.spring_Y_max = spring_Y_max
        self.use_ground_plane = int(bool(use_ground_plane))
        # variable for collision detection
        self.object_collision_flag = 0
        self.resting_collision_pairs = None
        self.wp_single_x = None
        self.collision_grid = None
        self.wp_collision_indices = None
        self.wp_collision_number = None
        self.static_box_count = 0
        zero_boxes = torch.zeros((1, 3), dtype=torch.float32, device=self.device)
        self.wp_static_box_mins = wp.from_torch(
            zero_boxes.clone(), dtype=wp.vec3, requires_grad=False
        )
        self.wp_static_box_maxs = wp.from_torch(
            zero_boxes.clone(), dtype=wp.vec3, requires_grad=False
        )
        self.static_surface_count = 0
        zero_surface_vec3 = torch.zeros((1, 3), dtype=torch.float32, device=self.device)
        zero_surface_float = torch.zeros((1,), dtype=torch.float32, device=self.device)
        zero_surface_kind = torch.zeros((1,), dtype=torch.int32, device=self.device)
        self.torch_static_surface_centers = zero_surface_vec3.clone()
        self.torch_static_surface_normals = zero_surface_vec3.clone()
        self.torch_static_surface_axes_u = zero_surface_vec3.clone()
        self.torch_static_surface_axes_v = zero_surface_vec3.clone()
        self.torch_static_surface_extents_u = zero_surface_float.clone()
        self.torch_static_surface_extents_v = zero_surface_float.clone()
        self.torch_static_surface_kinds = zero_surface_kind.clone()
        self.torch_static_surface_edge_radii = zero_surface_float.clone()
        self.torch_static_surface_heightfield_offsets = zero_surface_float.clone()
        self.torch_static_surface_heightfield_starts = zero_surface_kind.clone()
        self.torch_static_surface_heightfield_cells_u = zero_surface_kind.clone()
        self.torch_static_surface_heightfield_cells_v = zero_surface_kind.clone()
        self.wp_static_surface_centers = wp.from_torch(
            self.torch_static_surface_centers, dtype=wp.vec3, requires_grad=False
        )
        self.wp_static_surface_normals = wp.from_torch(
            self.torch_static_surface_normals, dtype=wp.vec3, requires_grad=False
        )
        self.wp_static_surface_axes_u = wp.from_torch(
            self.torch_static_surface_axes_u, dtype=wp.vec3, requires_grad=False
        )
        self.wp_static_surface_axes_v = wp.from_torch(
            self.torch_static_surface_axes_v, dtype=wp.vec3, requires_grad=False
        )
        self.wp_static_surface_extents_u = wp.from_torch(
            self.torch_static_surface_extents_u, dtype=wp.float32, requires_grad=False
        )
        self.wp_static_surface_extents_v = wp.from_torch(
            self.torch_static_surface_extents_v, dtype=wp.float32, requires_grad=False
        )
        self.wp_static_surface_kinds = wp.from_torch(
            self.torch_static_surface_kinds, dtype=wp.int32, requires_grad=False
        )
        self.wp_static_surface_edge_radii = wp.from_torch(
            self.torch_static_surface_edge_radii,
            dtype=wp.float32,
            requires_grad=False,
        )
        self.wp_static_surface_heightfield_offsets = wp.from_torch(
            self.torch_static_surface_heightfield_offsets,
            dtype=wp.float32,
            requires_grad=False,
        )
        self.wp_static_surface_heightfield_starts = wp.from_torch(
            self.torch_static_surface_heightfield_starts,
            dtype=wp.int32,
            requires_grad=False,
        )
        self.wp_static_surface_heightfield_cells_u = wp.from_torch(
            self.torch_static_surface_heightfield_cells_u,
            dtype=wp.int32,
            requires_grad=False,
        )
        self.wp_static_surface_heightfield_cells_v = wp.from_torch(
            self.torch_static_surface_heightfield_cells_v,
            dtype=wp.int32,
            requires_grad=False,
        )
        self.static_surface_query_distance = 0.1
        self.static_surface_margin = 0.0
        self.static_surface_restitution = 0.0
        self.static_surface_friction = 0.5
        self.static_mesh_enabled = 0
        self.static_mesh_substep_interval = 1
        self.static_mesh_id = 0
        self.static_mesh_two_sided = 0
        self.static_mesh_component_count = 0
        self.static_mesh_query_distance = 0.1
        self.static_mesh_winding_accuracy = 2.0
        self.static_mesh_winding_threshold = 0.5
        self.static_mesh_margin = 0.0
        self.static_mesh_restitution = 0.0
        self.static_mesh_friction = 0.5
        self.torch_static_mesh_component_mins = zero_boxes.clone()
        self.torch_static_mesh_component_maxs = zero_boxes.clone()
        self.wp_static_mesh_component_mins = wp.from_torch(
            self.torch_static_mesh_component_mins,
            dtype=wp.vec3,
            requires_grad=False,
        )
        self.wp_static_mesh_component_maxs = wp.from_torch(
            self.torch_static_mesh_component_maxs,
            dtype=wp.vec3,
            requires_grad=False,
        )
        self.wp_static_mesh_points = None
        self.wp_static_mesh_indices = None
        self.torch_static_mesh_points = None
        self.torch_static_mesh_indices = None
        self.static_collision_mesh = None
        
        if self_collision:
            self.object_collision_flag = 1
            self.resting_collision_pairs = wp.zeros((object_massnodes_single, object_massnodes_single),dtype=wp.bool, requires_grad=False)
        
            self.wp_single_x = wp.empty(
                shape=(object_massnodes_single, ),
                dtype=wp.vec3,
                device=cfg.device,
                requires_grad=False,
            )
            
            self.collision_grid = wp.HashGrid(128, 128, 128)

            self.wp_collision_indices = wp.zeros(
                (self.wp_init_vertices.shape[0], 500),
                dtype=wp.int32,
                requires_grad=False,
            )
            self.wp_collision_number = wp.zeros(
                (self.wp_init_vertices.shape[0]), dtype=wp.int32, requires_grad=False
            )

        self.wp_collide_elas = wp.from_torch(
            collide_elas.to(device=self.device, dtype=torch.float32).detach().reshape(1).contiguous(),
            requires_grad=cfg.collision_learn,
        )
        self.wp_collide_fric = wp.from_torch(
            collide_fric.to(device=self.device, dtype=torch.float32).detach().reshape(1).contiguous(),
            requires_grad=cfg.collision_learn,
        )
        self.wp_collide_object_elas = wp.from_torch(
            collide_object_elas.to(device=self.device, dtype=torch.float32).detach().reshape(1).contiguous(),
            requires_grad=cfg.collision_learn,
        )
        self.wp_collide_object_fric = wp.from_torch(
            collide_object_fric.to(device=self.device, dtype=torch.float32).detach().reshape(1).contiguous(),
            requires_grad=cfg.collision_learn,
        )
        spring_Y_temp = spring_Y.to(device=self.device, dtype=torch.float32).contiguous()
        self.torch_log_spring_Y = torch.log(spring_Y_temp.clamp_min(1e-12)).contiguous()
        self.wp_spring_Y = wp.from_torch(
            self.torch_log_spring_Y,
            dtype=wp.float32,
            requires_grad=True,
        )

        self.torch_spring_Y_clamped = spring_Y_temp.clamp(
            min=self.spring_Y_min, max=self.spring_Y_max
        ).contiguous()
        self.wp_spring_Y_clamped = wp.from_torch(
            self.torch_spring_Y_clamped, dtype=wp.float32, requires_grad=False
        )


        assert self.torch_spring_Y_clamped.numel() == self.wp_springs.shape[0]
        
        self.object_massnode_single = object_massnodes_single
        self.object_massnode_total = object_massnodes_total
        self.object_spring_single = object_springs_single
        self.object_spring_total = object_springs_total
        self.controller_massnode_single = controller_massnodes_single
        self.controller_spring_single = controller_springs_single
        self.number_of_instance = number_of_instance
        self.springs_per_instance = object_springs_single + controller_springs_single

        self.wp_init_velocities = None
        if init_velocities is None:
            self.wp_init_velocities = wp.zeros(
                shape=(self.object_massnode_total, ),
                dtype=wp.vec3,
                device=cfg.device,
                requires_grad=False)
        else:
            assert init_velocities.shape[0] == object_massnodes_total, "init_velocities shape mismatch"
            self.wp_init_velocities = wp.from_torch(
                init_velocities.contiguous(),
                dtype=wp.vec3,
                requires_grad=False,
            )

        self.wp_original_control_point = wp.from_torch(
            controller_rest_location.clone(), dtype=wp.vec3, requires_grad=False
        )
        self.wp_target_control_point = wp.from_torch(
            controller_rest_location.clone(), dtype=wp.vec3, requires_grad=False
        )
        self.num_controller_points = controller_rest_location.shape[0]
  
        # Preallocating the warp parameters
        self.wp_states = []
        for i in range(self.num_substeps + 1):
            state = State(self.wp_init_vertices, self.num_controller_points)
            self.wp_states.append(state)
        
        #add early return counters
        self.wp_early_return_count = wp.zeros((1,), dtype=wp.int32, device = cfg.device, requires_grad=False)

    def set_static_collision_boxes(self, boxes):
        if boxes is None:
            self.static_box_count = 0
            return

        boxes = boxes.to(device=self.device, dtype=torch.float32)
        if boxes.numel() == 0:
            self.static_box_count = 0
            return
        if boxes.ndim != 3 or boxes.shape[1:] != (2, 3):
            raise ValueError(f"static boxes shape {tuple(boxes.shape)} != (N,2,3)")

        mins = boxes[:, 0].contiguous()
        maxs = boxes[:, 1].contiguous()
        self.wp_static_box_mins = wp.from_torch(mins, dtype=wp.vec3, requires_grad=False)
        self.wp_static_box_maxs = wp.from_torch(maxs, dtype=wp.vec3, requires_grad=False)
        self.static_box_count = int(boxes.shape[0])

    def set_static_collision_surfaces(
        self,
        surfaces,
        *,
        query_distance=0.1,
        margin=0.0,
        restitution=0.0,
        friction=0.5,
    ):
        """Install immutable finite surface colliders before CUDA graph capture."""

        if not surfaces:
            self.static_surface_count = 0
            return

        centers = []
        normals = []
        axes_u = []
        axes_v = []
        extents_u = []
        extents_v = []
        kinds = []
        edge_radii = []
        heightfield_offsets = []
        heightfield_starts = []
        heightfield_cells_u = []
        heightfield_cells_v = []
        for index, surface in enumerate(surfaces):
            kind_name = str(surface.get("kind", "")).strip().lower()
            if kind_name not in {
                "rectangle",
                "disk",
                "cylinder",
                "heightfield",
                "heightfield_superellipse",
                "heightfield_capsule",
                "heightfield_rounded_capsule",
            }:
                raise ValueError(
                    f"static collision surface {index} has unsupported kind {kind_name!r}"
                )
            center = np.asarray(surface["center"], dtype=np.float32).reshape(3)
            normal = np.asarray(surface["normal"], dtype=np.float32).reshape(3)
            axis_u = np.asarray(surface["axis_u"], dtype=np.float32).reshape(3)
            axis_v = np.asarray(surface["axis_v"], dtype=np.float32).reshape(3)
            extent_u = float(surface["extent_u"])
            extent_v = float(surface["extent_v"])
            edge_radius = float(surface.get("edge_radius", 0.0))
            values = np.concatenate([center, normal, axis_u, axis_v])
            if not np.isfinite(values).all() or not np.isfinite(
                [extent_u, extent_v, edge_radius]
            ).all():
                raise ValueError(
                    f"static collision surface {index} contains non-finite values"
                )
            if extent_u <= 0.0 or extent_v <= 0.0:
                raise ValueError(
                    f"static collision surface {index} extents must be positive"
                )
            if kind_name == "heightfield_rounded_capsule":
                if edge_radius <= 0.0 or edge_radius >= 0.5 * extent_v:
                    raise ValueError(
                        f"static rounded capsule {index} edge radius must be "
                        "positive and less than half its capsule radius"
                    )
            elif edge_radius != 0.0:
                raise ValueError(
                    f"static collision surface {index} only supports an edge "
                    "radius for heightfield_rounded_capsule"
                )
            normal_length = float(np.linalg.norm(normal))
            axis_u_length = float(np.linalg.norm(axis_u))
            axis_v_length = float(np.linalg.norm(axis_v))
            if min(normal_length, axis_u_length, axis_v_length) <= 1.0e-6:
                raise ValueError(
                    f"static collision surface {index} basis contains a zero vector"
                )
            normal = normal / normal_length
            axis_u = axis_u / axis_u_length
            axis_v = axis_v / axis_v_length
            if (
                abs(float(np.dot(normal, axis_u))) > 1.0e-4
                or abs(float(np.dot(normal, axis_v))) > 1.0e-4
                or abs(float(np.dot(axis_u, axis_v))) > 1.0e-4
            ):
                raise ValueError(
                    f"static collision surface {index} basis must be orthogonal"
                )
            centers.append(center)
            normals.append(normal)
            axes_u.append(axis_u)
            axes_v.append(axis_v)
            extents_u.append(extent_u)
            extents_v.append(extent_v)
            edge_radii.append(edge_radius)
            kinds.append(
                {
                    "rectangle": 0,
                    "disk": 1,
                    "cylinder": 2,
                    "heightfield": 3,
                    "heightfield_superellipse": 4,
                    "heightfield_capsule": 5,
                    "heightfield_rounded_capsule": 6,
                }[kind_name]
            )
            if kind_name in {
                "heightfield",
                "heightfield_superellipse",
                "heightfield_capsule",
                "heightfield_rounded_capsule",
            }:
                offsets = np.asarray(
                    surface.get("normal_offsets_m"),
                    dtype=np.float32,
                )
                if (
                    offsets.ndim != 2
                    or min(offsets.shape) < 3
                    or not np.isfinite(offsets).all()
                ):
                    raise ValueError(
                        f"static collision heightfield {index} must contain a "
                        "finite grid with at least 3x3 vertices"
                    )
                heightfield_starts.append(len(heightfield_offsets))
                heightfield_cells_u.append(int(offsets.shape[1] - 1))
                heightfield_cells_v.append(int(offsets.shape[0] - 1))
                heightfield_offsets.extend(offsets.reshape(-1).tolist())
            else:
                heightfield_starts.append(0)
                heightfield_cells_u.append(0)
                heightfield_cells_v.append(0)

        scalar_values = {
            "query_distance": float(query_distance),
            "margin": float(margin),
            "restitution": float(restitution),
            "friction": float(friction),
        }
        if not all(np.isfinite(value) for value in scalar_values.values()):
            raise ValueError("static collision surface contact parameters must be finite")
        if scalar_values["query_distance"] <= 0.0:
            raise ValueError("static surface query_distance must be positive")
        if scalar_values["margin"] < 0.0:
            raise ValueError("static surface margin must be non-negative")
        if not 0.0 <= scalar_values["restitution"] <= 1.0:
            raise ValueError("static surface restitution must be in [0, 1]")
        if not 0.0 <= scalar_values["friction"] <= 2.0:
            raise ValueError("static surface friction must be in [0, 2]")

        def _float_tensor(values):
            return torch.as_tensor(
                np.ascontiguousarray(values, dtype=np.float32),
                dtype=torch.float32,
                device=self.device,
            ).contiguous()

        self.torch_static_surface_centers = _float_tensor(centers)
        self.torch_static_surface_normals = _float_tensor(normals)
        self.torch_static_surface_axes_u = _float_tensor(axes_u)
        self.torch_static_surface_axes_v = _float_tensor(axes_v)
        self.torch_static_surface_extents_u = _float_tensor(extents_u)
        self.torch_static_surface_extents_v = _float_tensor(extents_v)
        self.torch_static_surface_kinds = torch.as_tensor(
            np.ascontiguousarray(kinds, dtype=np.int32),
            dtype=torch.int32,
            device=self.device,
        ).contiguous()
        self.torch_static_surface_edge_radii = _float_tensor(edge_radii)
        if not heightfield_offsets:
            heightfield_offsets = [0.0]
        self.torch_static_surface_heightfield_offsets = _float_tensor(
            heightfield_offsets
        )
        self.torch_static_surface_heightfield_starts = torch.as_tensor(
            np.ascontiguousarray(heightfield_starts, dtype=np.int32),
            dtype=torch.int32,
            device=self.device,
        ).contiguous()
        self.torch_static_surface_heightfield_cells_u = torch.as_tensor(
            np.ascontiguousarray(heightfield_cells_u, dtype=np.int32),
            dtype=torch.int32,
            device=self.device,
        ).contiguous()
        self.torch_static_surface_heightfield_cells_v = torch.as_tensor(
            np.ascontiguousarray(heightfield_cells_v, dtype=np.int32),
            dtype=torch.int32,
            device=self.device,
        ).contiguous()
        self.wp_static_surface_centers = wp.from_torch(
            self.torch_static_surface_centers, dtype=wp.vec3, requires_grad=False
        )
        self.wp_static_surface_normals = wp.from_torch(
            self.torch_static_surface_normals, dtype=wp.vec3, requires_grad=False
        )
        self.wp_static_surface_axes_u = wp.from_torch(
            self.torch_static_surface_axes_u, dtype=wp.vec3, requires_grad=False
        )
        self.wp_static_surface_axes_v = wp.from_torch(
            self.torch_static_surface_axes_v, dtype=wp.vec3, requires_grad=False
        )
        self.wp_static_surface_extents_u = wp.from_torch(
            self.torch_static_surface_extents_u, dtype=wp.float32, requires_grad=False
        )
        self.wp_static_surface_extents_v = wp.from_torch(
            self.torch_static_surface_extents_v, dtype=wp.float32, requires_grad=False
        )
        self.wp_static_surface_kinds = wp.from_torch(
            self.torch_static_surface_kinds, dtype=wp.int32, requires_grad=False
        )
        self.wp_static_surface_edge_radii = wp.from_torch(
            self.torch_static_surface_edge_radii,
            dtype=wp.float32,
            requires_grad=False,
        )
        self.wp_static_surface_heightfield_offsets = wp.from_torch(
            self.torch_static_surface_heightfield_offsets,
            dtype=wp.float32,
            requires_grad=False,
        )
        self.wp_static_surface_heightfield_starts = wp.from_torch(
            self.torch_static_surface_heightfield_starts,
            dtype=wp.int32,
            requires_grad=False,
        )
        self.wp_static_surface_heightfield_cells_u = wp.from_torch(
            self.torch_static_surface_heightfield_cells_u,
            dtype=wp.int32,
            requires_grad=False,
        )
        self.wp_static_surface_heightfield_cells_v = wp.from_torch(
            self.torch_static_surface_heightfield_cells_v,
            dtype=wp.int32,
            requires_grad=False,
        )
        self.static_surface_count = len(surfaces)
        self.static_surface_query_distance = scalar_values["query_distance"]
        self.static_surface_margin = scalar_values["margin"]
        self.static_surface_restitution = scalar_values["restitution"]
        self.static_surface_friction = scalar_values["friction"]

    def set_static_collision_mesh(
        self,
        vertices,
        faces,
        *,
        two_sided=False,
        component_bounds=None,
        substep_interval=1,
        query_distance=0.1,
        winding_accuracy=2.0,
        winding_threshold=0.5,
        margin=0.0,
        restitution=0.0,
        friction=0.5,
    ):
        """Install one immutable collision mesh before graph capture.

        Closed meshes use signed-winding recovery. ``two_sided`` permits an
        open reconstruction surface and treats its triangles as a thin shell.
        ``substep_interval`` reduces expensive BVH queries while the integrator
        retains a continuous sweep from the preceding mesh query.
        """

        vertices_np = np.asarray(vertices, dtype=np.float32)
        faces_np = np.asarray(faces, dtype=np.int32)
        if vertices_np.ndim != 2 or vertices_np.shape[1] != 3:
            raise ValueError("static mesh vertices must have shape (V, 3)")
        if faces_np.ndim != 2 or faces_np.shape[1] != 3:
            raise ValueError("static mesh faces must have shape (F, 3)")
        if len(vertices_np) == 0 or len(faces_np) == 0:
            raise ValueError("static collision mesh must be non-empty")
        if not np.isfinite(vertices_np).all():
            raise ValueError("static mesh vertices contain non-finite values")
        if int(faces_np.min()) < 0 or int(faces_np.max()) >= len(vertices_np):
            raise ValueError("static mesh contains an invalid face index")

        if component_bounds is None:
            component_bounds_np = np.asarray(
                [[vertices_np.min(axis=0), vertices_np.max(axis=0)]],
                dtype=np.float32,
            )
        else:
            component_bounds_np = np.asarray(component_bounds, dtype=np.float32)
        if (
            component_bounds_np.ndim != 3
            or component_bounds_np.shape[1:] != (2, 3)
            or component_bounds_np.shape[0] == 0
            or not np.isfinite(component_bounds_np).all()
            or np.any(
                component_bounds_np[:, 1, :]
                <= component_bounds_np[:, 0, :]
            )
        ):
            raise ValueError(
                "static mesh component_bounds must have finite shape (N, 2, 3) "
                "with positive extents"
            )

        two_sided = bool(two_sided)
        if not two_sided:
            # Signed winding requires a closed surface. Every undirected edge
            # of each disconnected proxy component must occur twice.
            edge_counts = {}
            for face in faces_np:
                for first, second in (
                    (int(face[0]), int(face[1])),
                    (int(face[1]), int(face[2])),
                    (int(face[2]), int(face[0])),
                ):
                    edge = (min(first, second), max(first, second))
                    edge_counts[edge] = edge_counts.get(edge, 0) + 1
            open_edges = [
                edge for edge, count in edge_counts.items() if count != 2
            ]
            if open_edges:
                raise ValueError(
                    "static collision mesh is not closed; "
                    f"{len(open_edges)} edges do not have exactly two incident faces"
                )

        scalar_values = {
            "query_distance": float(query_distance),
            "winding_accuracy": float(winding_accuracy),
            "winding_threshold": float(winding_threshold),
            "margin": float(margin),
            "restitution": float(restitution),
            "friction": float(friction),
        }
        if not all(np.isfinite(value) for value in scalar_values.values()):
            raise ValueError("static mesh contact parameters must be finite")
        if scalar_values["query_distance"] <= 0.0:
            raise ValueError("static mesh query_distance must be positive")
        if scalar_values["winding_accuracy"] <= 0.0:
            raise ValueError("static mesh winding_accuracy must be positive")
        if not 0.0 <= scalar_values["winding_threshold"] <= 1.0:
            raise ValueError("static mesh winding_threshold must be in [0, 1]")
        if scalar_values["margin"] < 0.0:
            raise ValueError("static mesh margin must be non-negative")
        if not 0.0 <= scalar_values["restitution"] <= 1.0:
            raise ValueError("static mesh restitution must be in [0, 1]")
        if not 0.0 <= scalar_values["friction"] <= 2.0:
            raise ValueError("static mesh friction must be in [0, 2]")
        if isinstance(substep_interval, bool):
            raise ValueError("static mesh substep_interval must be a positive integer")
        try:
            substep_interval_value = int(substep_interval)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                "static mesh substep_interval must be a positive integer"
            ) from exc
        if substep_interval_value < 1 or substep_interval_value != substep_interval:
            raise ValueError("static mesh substep_interval must be a positive integer")

        vertices_t = torch.as_tensor(
            np.ascontiguousarray(vertices_np),
            dtype=torch.float32,
            device=self.device,
        ).contiguous()
        faces_t = torch.as_tensor(
            np.ascontiguousarray(faces_np.reshape(-1)),
            dtype=torch.int32,
            device=self.device,
        ).contiguous()
        self.torch_static_mesh_points = vertices_t
        self.torch_static_mesh_indices = faces_t
        self.wp_static_mesh_points = wp.from_torch(
            vertices_t,
            dtype=wp.vec3,
            requires_grad=False,
        )
        self.wp_static_mesh_indices = wp.from_torch(
            faces_t,
            dtype=wp.int32,
            requires_grad=False,
        )
        self.torch_static_mesh_component_mins = torch.as_tensor(
            np.ascontiguousarray(component_bounds_np[:, 0, :]),
            dtype=torch.float32,
            device=self.device,
        ).contiguous()
        self.torch_static_mesh_component_maxs = torch.as_tensor(
            np.ascontiguousarray(component_bounds_np[:, 1, :]),
            dtype=torch.float32,
            device=self.device,
        ).contiguous()
        self.wp_static_mesh_component_mins = wp.from_torch(
            self.torch_static_mesh_component_mins,
            dtype=wp.vec3,
            requires_grad=False,
        )
        self.wp_static_mesh_component_maxs = wp.from_torch(
            self.torch_static_mesh_component_maxs,
            dtype=wp.vec3,
            requires_grad=False,
        )
        self.static_collision_mesh = wp.Mesh(
            points=self.wp_static_mesh_points,
            indices=self.wp_static_mesh_indices,
            support_winding_number=not two_sided,
        )
        self.static_mesh_enabled = 1
        self.static_mesh_substep_interval = substep_interval_value
        self.static_mesh_id = self.static_collision_mesh.id
        self.static_mesh_two_sided = int(two_sided)
        self.static_mesh_component_count = int(component_bounds_np.shape[0])
        self.static_mesh_query_distance = scalar_values["query_distance"]
        self.static_mesh_winding_accuracy = scalar_values["winding_accuracy"]
        self.static_mesh_winding_threshold = scalar_values["winding_threshold"]
        self.static_mesh_margin = scalar_values["margin"]
        self.static_mesh_restitution = scalar_values["restitution"]
        self.static_mesh_friction = scalar_values["friction"]

    def update_local_spring_subset(self, spring_indices, springs, rest_lengths):
        spring_indices = spring_indices.to(
            device=self.torch_springs.device, dtype=torch.long
        )
        springs = springs.to(
            device=self.torch_springs.device, dtype=self.torch_springs.dtype
        )
        rest_lengths = rest_lengths.to(
            device=self.torch_rest_lengths.device,
            dtype=self.torch_rest_lengths.dtype,
        )
        inv_rest_lengths = 1.0 / rest_lengths.clamp_min(1e-6)
        self.torch_springs.index_copy_(0, spring_indices, springs)
        self.torch_rest_lengths.index_copy_(0, spring_indices, rest_lengths)
        self.torch_inv_rest_lengths.index_copy_(0, spring_indices, inv_rest_lengths)

    def update_local_spring_stiffness_subset(self, spring_indices, spring_y):
        spring_indices = spring_indices.to(
            device=self.torch_spring_Y_clamped.device, dtype=torch.long
        )
        clamped = spring_y.to(
            device=self.torch_spring_Y_clamped.device,
            dtype=self.torch_spring_Y_clamped.dtype,
        ).clamp(min=0.0, max=self.spring_Y_max)
        log_clamped = torch.log(clamped.clamp_min(1e-12))
        self.torch_spring_Y_clamped.index_copy_(0, spring_indices, clamped)
        self.torch_log_spring_Y.index_copy_(0, spring_indices, log_clamped)



    #pyh creating cuda graph requires all scalar variable to be set, that's not possible in batched 
    #so we are moving it out
    def create_cuda_graph(self):
        # Create the CUDA graph to acclerate
        with wp.ScopedCapture() as forward_capture:
            self.step()   
        self.forward_graph = forward_capture.graph

    # Create the rest map for self-collision in frame 0
    #pyh updated for batched version
    def create_resting_case(self):
        #we only need build restmap for one instance to be shared but warp does not allow slicing 
        wp.launch(
            copy_vec3,
            dim=self.object_massnode_single,
            inputs=[self.wp_states[0].wp_x],
            outputs=[self.wp_single_x],
        )

        self.resting_collision_pairs.zero_()
        self.collision_grid.build(
            self.wp_single_x,
            self.self_collision_rest_exclusion_distance,
        )
        print(
            "create resting case implementation "
            "(doesn't mean its in use check update_collision_graph) "
            "rest_exclusion_multiplier="
            f"{self.self_collision_rest_exclusion_multiplier:.3f} "
            "rest_exclusion_distance_m="
            f"{self.self_collision_rest_exclusion_distance:.6f}"
        )
        wp.launch(
            build_resting_collision_pairs,
            dim=self.object_massnode_single,
            inputs=[
                self.wp_single_x,
                self.self_collision_rest_exclusion_distance,
                self.collision_grid.id,
                ],
            outputs=[self.resting_collision_pairs],            
        )  
    
    def set_controller_interactive(
        self, last_controller_interactive, controller_interactive
    ):
        # Set the controller points
        wp.launch(
            copy_vec3,
            dim=self.num_controller_points,
            inputs=[last_controller_interactive],
            outputs=[self.wp_original_control_point],
        )
        wp.launch(
            copy_vec3,
            dim=self.num_controller_points,
            inputs=[controller_interactive],
            outputs=[self.wp_target_control_point],
        )

    def set_init_state(self, wp_x, wp_v):
        assert (
            self.object_massnode_total == wp_x.shape[0]
            and self.object_massnode_total == self.wp_states[0].wp_x.shape[0]
        )

        wp.launch(
            copy_vec3,
            dim=self.object_massnode_total,
            inputs=[wp_x],
            outputs=[self.wp_states[0].wp_x],
        )
        wp.launch(
            copy_vec3,
            dim=self.object_massnode_total,
            inputs=[wp_v],
            outputs=[self.wp_states[0].wp_v],
        )

    def update_collision_graph(self):
        #pyh build a big hash grid over all instances are ok as long as the offset is big
        self.collision_grid.build(self.wp_states[0].wp_x, self.collision_dist * 5.0)
        self.wp_collision_number.zero_()

        wp.launch(
          update_potential_collision_restmap,
          dim=self.object_massnode_total,
          inputs=[
              self.wp_states[0].wp_x,
              self.wp_masks,
              self.collision_dist,
              self.collision_grid.id,
              self.resting_collision_pairs,
              self.object_massnode_single,
          ],
          outputs=[self.wp_collision_indices, self.wp_collision_number],
        )

    def step(self):
        # The full-resolution Ambulance shell is much more expensive than the
        # analytic floor/surface primitives.  Query it at its configured
        # cadence, always include the final substep, and sweep over every
        # skipped state so the optimization does not introduce tunnelling.
        last_static_mesh_query_state = 0
        for i in range(self.num_substeps):
            self.wp_states[i].clear_forces()

            # Set the control point
            wp.launch(
                set_control_points,
                dim=self.num_controller_points,
                inputs=[
                    self.num_substeps,
                    self.wp_original_control_point,
                    self.wp_target_control_point,
                    i,
                ],
                outputs=[self.wp_states[i].wp_control_x],
            )

            wp.launch(
                kernel=eval_springs_batched_opt,
                 dim=self.n_springs,
                inputs=[
                    self.wp_states[i].wp_x,
                    self.wp_states[i].wp_v,
                    self.wp_states[i].wp_control_x,
                    self.wp_states[i].wp_control_v,
                    self.wp_springs,
                    self.wp_inv_rest_length,
                    self.wp_spring_Y_clamped,
                    self.dashpot_damping,
                    self.object_spring_single,
                    self.object_spring_total,
                    self.controller_spring_single,
                    self.object_massnode_single,
                    self.controller_massnode_single,
                ],
                outputs=[self.wp_states[i].wp_vertice_forces],
            )    

            if self.object_collision_flag:
                output_v = self.wp_states[i].wp_v_before_collision
            else:
                output_v = self.wp_states[i].wp_v_before_ground

            # Update the output_v using the vertive_forces
            wp.launch(
                kernel=update_vel_from_force,
                dim=self.object_massnode_total,
                inputs=[
                    self.wp_states[i].wp_v,
                    self.wp_states[i].wp_vertice_forces,
                    #shared
                    self.wp_masses,
                    self.dt,
                    self.drag_damping,
                    self.reverse_factor,
                    #pyh added
                    self.object_massnode_single,
                ],
                outputs=[output_v],
            )

            if self.object_collision_flag:
                # Update the wp_v_before_ground based on the collision handling
                wp.launch(
                    kernel=object_collision,
                    dim=self.object_massnode_total,
                    inputs=[
                        self.wp_states[i].wp_x,
                        self.wp_states[i].wp_v_before_collision,
                        #shared
                        self.wp_masses,
                        self.wp_masks,
                        self.wp_collide_object_elas,
                        self.wp_collide_object_fric,
                        self.collision_dist,
                        self.wp_collision_indices,
                        self.wp_collision_number,
                        #added
                        self.object_massnode_single,
                    ],
                    outputs=[self.wp_states[i].wp_v_before_ground],
                )

            static_mesh_enabled_this_substep = self.static_mesh_enabled
            static_mesh_sweep_start = self.wp_states[i].wp_x
            if self.static_mesh_enabled:
                query_interval = max(int(self.static_mesh_substep_interval), 1)
                query_this_substep = (
                    (i + 1) % query_interval == 0
                    or i == self.num_substeps - 1
                )
                if query_this_substep:
                    static_mesh_sweep_start = self.wp_states[
                        last_static_mesh_query_state
                    ].wp_x
                    last_static_mesh_query_state = i + 1
                else:
                    static_mesh_enabled_this_substep = 0

            # Update the x and v
            wp.launch(
                kernel=integrate_ground_collision,
                dim=self.object_massnode_total,
                inputs=[
                    self.wp_states[i].wp_x,
                    self.wp_states[i].wp_v_before_ground,
                    self.wp_collide_elas,
                    self.wp_collide_fric,
                    self.dt,
                    self.reverse_factor,
                    self.use_ground_plane,
                    self.wp_static_box_mins,
                    self.wp_static_box_maxs,
                    self.static_box_count,
                    self.wp_static_surface_centers,
                    self.wp_static_surface_normals,
                    self.wp_static_surface_axes_u,
                    self.wp_static_surface_axes_v,
                    self.wp_static_surface_extents_u,
                    self.wp_static_surface_extents_v,
                    self.wp_static_surface_kinds,
                    self.wp_static_surface_edge_radii,
                    self.wp_static_surface_heightfield_offsets,
                    self.wp_static_surface_heightfield_starts,
                    self.wp_static_surface_heightfield_cells_u,
                    self.wp_static_surface_heightfield_cells_v,
                    self.static_surface_count,
                    self.static_surface_query_distance,
                    self.static_surface_margin,
                    self.static_surface_restitution,
                    self.static_surface_friction,
                    static_mesh_enabled_this_substep,
                    static_mesh_sweep_start,
                    self.static_mesh_id,
                    self.static_mesh_two_sided,
                    self.wp_static_mesh_component_mins,
                    self.wp_static_mesh_component_maxs,
                    self.static_mesh_component_count,
                    self.static_mesh_query_distance,
                    self.static_mesh_winding_accuracy,
                    self.static_mesh_winding_threshold,
                    self.static_mesh_margin,
                    self.static_mesh_restitution,
                    self.static_mesh_friction,
                ],
                outputs=[self.wp_states[i + 1].wp_x, self.wp_states[i + 1].wp_v],
            )         
