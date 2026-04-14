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


@wp.func
def _collision_response(
    velocity: wp.vec3,
    normal: wp.vec3,
    collide_elas: float,
    collide_fric: float,
) -> wp.vec3:
    v_normal = wp.dot(velocity, normal) * normal
    v_tao = velocity - v_normal
    v_normal_length = wp.length(v_normal)
    v_tao_length = wp.max(wp.length(v_tao), 1e-6)
    v_normal_new = -collide_elas * v_normal
    a = wp.max(
        0.0,
        1.0
        - collide_fric
        * (1.0 + collide_elas)
        * v_normal_length
        / v_tao_length,
    )
    v_tao_new = a * v_tao
    return v_normal_new + v_tao_new


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
    collision_dist: float,
    grid: wp.uint64,
    resting_collision_pairs: wp.array2d(dtype=wp.bool),
):

    tid = wp.tid()

    # order threads by cell
    i = wp.hash_grid_point_id(grid, tid)

    x1 = x[i]

    neighbors = wp.hash_grid_query(grid, x1, collision_dist * 5.0)
    for index in neighbors:
        if index < i:
            resting_collision_pairs[i][index] = wp.bool(1)
            resting_collision_pairs[index][i] = wp.bool(1)

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
    x_new: wp.array(dtype=wp.vec3),
    v_new: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()

    x0 = x[tid]
    v0 = v[tid]

    clamp_collide_elas = wp.clamp(collide_elas[0], low=0.0, high=1.0)
    clamp_collide_fric = wp.clamp(collide_fric[0], low=0.0, high=2.0)
    normal = wp.vec3(0.0, 0.0, 1.0) * reverse_factor

    x_result = x0 + v0 * dt
    v_result = v0

    if use_ground_plane != 0:
        x_z = x0[2]
        v_z = v0[2]
        next_x_z = (x_z + v_z * dt) * reverse_factor
        if next_x_z < 0.0 and v_z * reverse_factor < -1e-4:
            v_after = _collision_response(v0, normal, clamp_collide_elas, clamp_collide_fric)
            toi = -x_z / v_z
            x_result = x0 + v0 * toi + v_after * (dt - toi)
            v_result = v_after

    collision_start = x0
    collision_velocity = v_result
    collision_delta = collision_velocity * dt
    inside_found = int(0)
    inside_best_distance = float(1.0e12)
    inside_best_normal = wp.vec3(0.0, 0.0, 0.0)
    inside_best_projection = collision_start
    hit_found = int(0)
    best_hit_t = float(2.0)
    best_hit_normal = wp.vec3(0.0, 0.0, 0.0)
    best_hit_point = x_result

    for box_index in range(static_box_count):
        box_min = static_box_mins[box_index]
        box_max = static_box_maxs[box_index]
        inside_box = (
            collision_start[0] >= box_min[0]
            and collision_start[0] <= box_max[0]
            and collision_start[1] >= box_min[1]
            and collision_start[1] <= box_max[1]
            and collision_start[2] >= box_min[2]
            and collision_start[2] <= box_max[2]
        )
        if inside_box:
            dist_x_min = collision_start[0] - box_min[0]
            dist_x_max = box_max[0] - collision_start[0]
            dist_y_min = collision_start[1] - box_min[1]
            dist_y_max = box_max[1] - collision_start[1]
            dist_z_min = collision_start[2] - box_min[2]
            dist_z_max = box_max[2] - collision_start[2]

            nearest_distance = float(dist_x_min)
            nearest_normal = wp.vec3(-1.0, 0.0, 0.0)
            projected = wp.vec3(
                box_min[0] - 1.0e-4,
                collision_start[1],
                collision_start[2],
            )

            if dist_x_max < nearest_distance:
                nearest_distance = dist_x_max
                nearest_normal = wp.vec3(1.0, 0.0, 0.0)
                projected = wp.vec3(
                    box_max[0] + 1.0e-4,
                    collision_start[1],
                    collision_start[2],
                )
            if dist_y_min < nearest_distance:
                nearest_distance = dist_y_min
                nearest_normal = wp.vec3(0.0, -1.0, 0.0)
                projected = wp.vec3(
                    collision_start[0],
                    box_min[1] - 1.0e-4,
                    collision_start[2],
                )
            if dist_y_max < nearest_distance:
                nearest_distance = dist_y_max
                nearest_normal = wp.vec3(0.0, 1.0, 0.0)
                projected = wp.vec3(
                    collision_start[0],
                    box_max[1] + 1.0e-4,
                    collision_start[2],
                )
            if dist_z_min < nearest_distance:
                nearest_distance = dist_z_min
                nearest_normal = wp.vec3(0.0, 0.0, -1.0)
                projected = wp.vec3(
                    collision_start[0],
                    collision_start[1],
                    box_min[2] - 1.0e-4,
                )
            if dist_z_max < nearest_distance:
                nearest_distance = dist_z_max
                nearest_normal = wp.vec3(0.0, 0.0, 1.0)
                projected = wp.vec3(
                    collision_start[0],
                    collision_start[1],
                    box_max[2] + 1.0e-4,
                )

            if nearest_distance < inside_best_distance:
                inside_found = 1
                inside_best_distance = nearest_distance
                inside_best_normal = nearest_normal
                inside_best_projection = projected
        else:
            t_enter = float(0.0)
            t_exit = float(1.0)
            hit_normal = wp.vec3(0.0, 0.0, 0.0)
            valid_hit = int(1)

            for axis in range(3):
                p = collision_start[axis]
                delta = collision_delta[axis]
                min_v = box_min[axis]
                max_v = box_max[axis]

                if wp.abs(delta) < 1.0e-8:
                    if p < min_v or p > max_v:
                        valid_hit = int(0)
                else:
                    inv_delta = float(1.0) / delta
                    t1 = (min_v - p) * inv_delta
                    t2 = (max_v - p) * inv_delta
                    entry_t = wp.min(t1, t2)
                    exit_t = wp.max(t1, t2)
                    axis_normal = wp.vec3(0.0, 0.0, 0.0)

                    if t1 < t2:
                        if axis == 0:
                            axis_normal = wp.vec3(-1.0, 0.0, 0.0)
                        elif axis == 1:
                            axis_normal = wp.vec3(0.0, -1.0, 0.0)
                        else:
                            axis_normal = wp.vec3(0.0, 0.0, -1.0)
                    else:
                        if axis == 0:
                            axis_normal = wp.vec3(1.0, 0.0, 0.0)
                        elif axis == 1:
                            axis_normal = wp.vec3(0.0, 1.0, 0.0)
                        else:
                            axis_normal = wp.vec3(0.0, 0.0, 1.0)

                    if entry_t > t_enter:
                        t_enter = float(entry_t)
                        hit_normal = axis_normal
                    t_exit = float(wp.min(t_exit, exit_t))
                    if t_enter > t_exit:
                        valid_hit = int(0)

            if valid_hit != 0 and t_enter >= 0.0 and t_enter <= 1.0 and t_enter < best_hit_t:
                hit_found = int(1)
                best_hit_t = float(t_enter)
                best_hit_normal = hit_normal
                best_hit_point = collision_start + collision_delta * t_enter

    if inside_found != 0:
        v_after = collision_velocity
        if wp.dot(collision_velocity, inside_best_normal) < 0.0:
            v_after = _collision_response(
                collision_velocity,
                inside_best_normal,
                clamp_collide_elas,
                clamp_collide_fric,
            )
        x_result = inside_best_projection + v_after * dt
        v_result = v_after
    elif hit_found != 0:
        v_after = _collision_response(
            collision_velocity,
            best_hit_normal,
            clamp_collide_elas,
            clamp_collide_fric,
        )
        remaining_t = 1.0 - best_hit_t
        x_result = best_hit_point + best_hit_normal * 1.0e-4 + v_after * (dt * remaining_t)
        v_result = v_after

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

        self.collision_grid.build(self.wp_single_x, self.collision_dist * 5.0)
        print("create resting case implementation (doesn't mean its in use check update_collision_graph)")
        wp.launch(
            build_resting_collision_pairs,
            dim=self.object_massnode_single,
            inputs=[
                self.wp_single_x,
                self.collision_dist,
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
                ],
                outputs=[self.wp_states[i + 1].wp_x, self.wp_states[i + 1].wp_v],
            )         
