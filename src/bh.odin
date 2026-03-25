package nbody

import "core:thread"
import "core:sync"
import "core:mem"
import "core:container/small_array"
import "core:fmt"
import "core:math/linalg"

INVALID_NODE :: -1
INVALID_PARTICLE :: -1

Node :: struct {
	center_of_mass:  [3]f32,
	total_mass:      f32,
	first_child_idx: i32,
	particle_count:  u32,
	particle_idx:    i32,
	child_mask:      u8,
	depth:           u8,
	_reserved:       [2]u8,
}

Octree :: struct {
	nodes:      [dynamic]Node,
	particles:  ^#soa[]Particle,
	center:     [3]f32,
	half_width: f32,
}

get_octant_index :: #force_inline proc(center: [3]f32, pos: [3]f32) -> int {
	index := 0
	if pos.x >= center.x do index |= 1 // Bit 0: X-axis
	if pos.y >= center.y do index |= 2 // Bit 1: Y-axis
	if pos.z >= center.z do index |= 4 // Bit 2: Z-axis
	return index
}

get_child_center :: #force_inline proc(
	parent_center: [3]f32,
	parent_half_width: f32,
	octant: int,
) -> [3]f32 {
	half_width := parent_half_width * 0.5
	return {
		parent_center.x + (octant & 1 != 0 ? half_width : -half_width),
		parent_center.y + (octant & 2 != 0 ? half_width : -half_width),
		parent_center.z + (octant & 4 != 0 ? half_width : -half_width),
	}
}

octree_init :: proc(
	t: ^Octree,
	particles: ^#soa[]Particle,
	center: [3]f32,
	half_width: f32,
	allocator := context.allocator,
	loc := #caller_location,
) {
	t.center = center
	t.half_width = half_width
	t.particles = particles
	t.nodes = make([dynamic]Node, 1, allocator, loc)
	t.nodes[0].particle_idx = INVALID_PARTICLE
	t.nodes[0].first_child_idx = INVALID_NODE
}

octree_reset :: proc(t: ^Octree, center: [3]f32, half_width: f32) {
	clear(&t.nodes)

	t.center = center
	t.half_width = half_width

	append(&t.nodes, Node{particle_idx = INVALID_PARTICLE, first_child_idx = INVALID_NODE})
}

octree_destroy :: proc(t: ^Octree) {
	delete(t.nodes)
	t^ = {}
}

octree_insert :: proc(t: ^Octree, particle: i32) {
	octree_insert_recursive(t, 0, particle, t.center, t.half_width)
}

MAX_DEPTH :: 32

@(private)
octree_insert_recursive :: proc(
	t: ^Octree,
	node_index, particle_index: i32,
	center: [3]f32,
	half_width: f32,
) -> bool {
	// Shared updates
	t.nodes[node_index].particle_count += 1

	new_particle_mass := t.particles.mass[particle_index]
	new_particle_pos := [3]f32 {
		t.particles.px[particle_index],
		t.particles.py[particle_index],
		t.particles.pz[particle_index],
	}

	old_mass := t.nodes[node_index].total_mass
	t.nodes[node_index].total_mass += new_particle_mass
	t.nodes[node_index].center_of_mass =
		(t.nodes[node_index].center_of_mass * old_mass + new_particle_pos * new_particle_mass) /
		t.nodes[node_index].total_mass

	if t.nodes[node_index].depth >= MAX_DEPTH {
		fmt.eprintf("ERROR: Octree max depth reached for particle %d", particle_index)
		return false
	}

	// Empty node -- insert in place
	if t.nodes[node_index].particle_idx == INVALID_PARTICLE &&
	   t.nodes[node_index].first_child_idx == INVALID_NODE {
		t.nodes[node_index].particle_idx = particle_index
		return true
	}

	// Leaf node -- subdivide, insert old, insert new
	if t.nodes[node_index].particle_idx != INVALID_PARTICLE {
		// initialize children
		first_child := i32(len(t.nodes))
		resize(&t.nodes, len(t.nodes) + 8)
		for i in 0 ..< 8 {
			t.nodes[int(first_child) + i] = {
				particle_idx    = INVALID_PARTICLE,
				first_child_idx = INVALID_NODE,
				depth           = t.nodes[node_index].depth + 1,
			}
		}

		// swap current node to internal
		replace_index := t.nodes[node_index].particle_idx
		t.nodes[node_index].particle_idx = INVALID_PARTICLE
		t.nodes[node_index].first_child_idx = first_child

		// insert particle previously in this node at correct child
		octree_insert_at_child(t, node_index, replace_index, center, half_width)
	}

	// Internal node -- recurse at correct child
	octree_insert_at_child(t, node_index, particle_index, center, half_width)
	return true
}

@(private)
octree_insert_at_child :: proc(
	t: ^Octree,
	parent_index: i32,
	particle_index: i32,
	center: [3]f32,
	half_width: f32,
) {
	child_octant := get_octant_index(
		center,
		[3]f32 {
			t.particles.px[particle_index],
			t.particles.py[particle_index],
			t.particles.pz[particle_index],
		},
	)
	t.nodes[parent_index].child_mask |= 1 << uint(child_octant)

	center := get_child_center(center, half_width, child_octant)
	octree_insert_recursive(
		t,
		t.nodes[parent_index].first_child_idx + i32(child_octant),
		particle_index,
		center,
		half_width / 2,
	)
}

@(private)
bh_accumulte_force :: proc(
	t: ^Octree,
	particle_index: i32,
	root_width: f32,
	theta: f32,
	g: f32 = G,
	eps: f32 = EPS,
	allocator := context.allocator,
) {
	Stack_Entry :: struct {
		node_index:   i32,
		region_width: f32,
	}
	nodes := small_array.Small_Array(256, Stack_Entry){}
	small_array.append(&nodes, Stack_Entry{0, root_width})

	particle_pos := [3]f32 {
		t.particles.px[particle_index],
		t.particles.py[particle_index],
		t.particles.pz[particle_index],
	}

	for small_array.len(nodes) > 0 {
		entry := small_array.pop_back(&nodes)
		node := &t.nodes[entry.node_index]
		dist := linalg.distance(particle_pos, node.center_of_mass)

		// leaf node or sufficiently far
		if node.first_child_idx == INVALID_NODE || entry.region_width < theta * dist {
			if node.particle_idx == particle_index do continue
			fx, fy, fz := calculate_force(
				t.particles.mass[particle_index],
				node.total_mass,
				particle_pos.x,
				particle_pos.y,
				particle_pos.z,
				node.center_of_mass.x,
				node.center_of_mass.y,
				node.center_of_mass.z,
				g,
				eps,
			)
			t.particles.ax[:][particle_index] += fx
			t.particles.ay[:][particle_index] += fy
			t.particles.az[:][particle_index] += fz

			continue
		}

		// internal node -- recurse on existing children
		child_width := entry.region_width * 0.5
		for i in 0 ..< 8 {
			// skip non-existent children
			if node.child_mask & (0b1 << uint(i)) == 0 do continue
			small_array.append(&nodes, Stack_Entry{node.first_child_idx + i32(i), child_width})
		}
	}
}

bh_reorder :: proc(t: ^Octree, allocator := context.allocator) {
	indices := make([]i32, len(t.particles), allocator)
	defer delete(indices)
	index := 0

	nodes := small_array.Small_Array(256, i32){}
	small_array.append(&nodes, 0)

	for small_array.len(nodes) > 0 {
		node := t.nodes[small_array.pop_back(&nodes)]
		if node.first_child_idx == INVALID_NODE && node.particle_idx == INVALID_PARTICLE do continue
		if node.particle_idx != INVALID_PARTICLE {
			indices[index] = node.particle_idx
			index += 1
		}

		for i in 0..<8 {
			if node.child_mask & (0b1 << uint(i)) == 0 do continue
			small_array.append(&nodes, node.first_child_idx + i32(i))
		}
	}

	assert(index == len(indices), "these should be equal")

	reordered_particles := make(#soa[]Particle, len(t.particles), allocator)
	defer delete(reordered_particles)

	for index, i in indices {
		reordered_particles[i] = t.particles[index]
	}

	mem.copy(rawptr(&t.particles.px[0]), rawptr(&reordered_particles.px[0]), size_of(Particle)*len(t.particles))
}

bh_simulate :: proc(
	t: ^Octree,
	center: [3]f32,
	half_width: f32,
	theta: f32,
	g: f32 = G,
	eps: f32 = EPS,
	reorder := true,
) {
	assert(len(t.particles) != 0, "Barnes-Hut tree must have particles to simulate")
	octree_reset(t, center, half_width)
	for i in 0 ..< len(t.particles) do octree_insert(t, i32(i))
	if reorder do bh_reorder(t)
	for i in 0 ..< len(t.particles) do bh_accumulte_force(t, i32(i), half_width * 2, theta, g, eps)
}

bh_simulate_threaded :: proc(
	t: ^Octree,
	center: [3]f32,
	half_width: f32,
	theta: f32,
	g: f32 = G,
	eps: f32 = EPS,
	reorder := true,
	allocator := context.allocator,
) {
	assert(len(t.particles) != 0, "Barnes-Hut tree must have particles to simulate")
	octree_reset(t, center, half_width)
	for i in 0 ..< len(t.particles) do octree_insert(t, i32(i))
	if reorder do bh_reorder(t)

	TaskData :: struct {
		start, stop: int,
		t: ^Octree,
		half_width, theta, g, eps: f32,
		wg:        ^sync.Wait_Group,
	}

	n_tasks := len(global_pool.threads) * THREAD_MULT * 4
	tasks := make([]TaskData, n_tasks, allocator)
	defer delete(tasks)

	task_handler :: proc(task: thread.Task) {
		data := cast(^TaskData)task.data
		defer sync.wait_group_done(data.wg)

		for i in data.start ..< data.stop do bh_accumulte_force(data.t, i32(i), data.half_width * 2, data.theta, data.g, data.eps) 
	}

	n_per_task := len(t.particles) / n_tasks + 1

	wg: sync.Wait_Group
	sync.wait_group_add(&wg, n_tasks)

	for index in 0 ..< n_tasks {
		task_data := &tasks[index]
		task_data.start = index * n_per_task
		task_data.stop = min(len(t.particles), (index + 1) * n_per_task)
		task_data.t = t
		task_data.half_width = half_width
		task_data.theta = theta
		task_data.eps = eps
		task_data.g = g
		task_data.wg = &wg

		thread.pool_add_task(&global_pool, mem.nil_allocator(), task_handler, task_data, index)
	}

	sync.wait_group_wait(&wg)
	global_thread_pool_clear_done()
}
