package main

import "core:log"
import "core:math"
import "core:math/linalg"
import "core:math/rand"
import "core:mem"
import dial "shared:dial"
import nbody "src"

N_BODIES :: 8000
G :: 100
EPS :: 10

// INVALID_NODE :: -1
// INVALID_PARTICLE :: -1
//
// Node :: struct {
// 	center_of_mass:  [3]f32,
// 	total_mass:      f32,
// 	first_child_idx: i32,
// 	particle_count:  u32,
// 	particle_idx:    i32,
// 	child_mask:      u8,
// 	depth:           u8,
// 	_reserved:       [2]u8,
// }
//
// Octree :: struct {
// 	nodes:      [dynamic]Node,
// 	particles:  ^#soa[]Particle,
// 	center:     [3]f32,
// 	half_width: f32,
// }
//
// get_octant_index :: #force_inline proc(center: [3]f32, pos: [3]f32) -> int {
// 	index := 0
// 	if pos.x >= center.x do index |= 1 // Bit 0: X-axis
// 	if pos.y >= center.y do index |= 2 // Bit 1: Y-axis
// 	if pos.z >= center.z do index |= 4 // Bit 2: Z-axis
// 	return index
// }
//
// get_child_center :: #force_inline proc(
// 	parent_center: [3]f32,
// 	parent_half_width: f32,
// 	octant: int,
// ) -> [3]f32 {
// 	half_width := parent_half_width * 0.5
// 	return {
// 		parent_center.x + (octant & 1 != 0 ? half_width : -half_width),
// 		parent_center.y + (octant & 2 != 0 ? half_width : -half_width),
// 		parent_center.z + (octant & 4 != 0 ? half_width : -half_width),
// 	}
// }
//
// octree_init :: proc(
// 	t: ^Octree,
// 	particles: ^#soa[]Particle,
// 	center: [3]f32,
// 	half_width: f32,
// 	allocator := context.allocator,
// 	loc := #caller_location,
// ) {
// 	t.center = center
// 	t.half_width = half_width
// 	t.particles = particles
// 	t.nodes = make([dynamic]Node, 1, allocator, loc)
// 	t.nodes[0].particle_idx = INVALID_PARTICLE
// 	t.nodes[0].first_child_idx = INVALID_NODE
// }
//
// octree_reset :: proc(t: ^Octree, center: [3]f32, half_width: f32) {
// 	clear(&t.nodes)
//
// 	t.center = center
// 	t.half_width = half_width
//
// 	append(&t.nodes, Node{particle_idx = INVALID_PARTICLE, first_child_idx = INVALID_NODE})
// }
//
// octree_destroy :: proc(t: ^Octree) {
// 	delete(t.nodes)
// 	t^ = {}
// }
//
// octree_insert :: proc(t: ^Octree, particle: i32) {
// 	octree_insert_recursive(t, 0, particle, t.center, t.half_width)
// }
//
// @(private)
// octree_insert_recursive :: proc(
// 	t: ^Octree,
// 	node_index, particle_index: i32,
// 	center: [3]f32,
// 	half_width: f32,
// ) {
// 	// Shared updates
// 	node := &t.nodes[node_index]
// 	node.particle_count += 1
//
// 	new_particle := &t.particles[particle_index]
// 	old_mass := node.total_mass
// 	node.total_mass += new_particle.mass
// 	for i in 0 ..< 3 {
// 		node.center_of_mass[i] =
// 			(node.center_of_mass[i] * old_mass + new_particle.pos[i] * new_particle.mass) /
// 			node.total_mass
// 	}
//
// 	// Empty node -- insert in place
// 	if node.particle_idx == INVALID_PARTICLE && node.first_child_idx == INVALID_NODE {
// 		node.particle_idx = particle_index
// 		return
// 	}
//
// 	// Leaf node -- subdivide, insert old, insert new
// 	if node.particle_idx != INVALID_PARTICLE {
// 		first_child := i32(len(t.nodes))
// 		reserve(&t.nodes, len(t.nodes) + 8)
// 		for i in 0 ..< 8 {
// 			t.nodes[int(first_child) + i] = {
// 				particle_idx    = INVALID_PARTICLE,
// 				first_child_idx = INVALID_NODE,
// 				depth           = node.depth + 1,
// 			}
// 		}
//
// 		node = &t.nodes[node_index]
// 		replace_octant := get_octant_index(center, t.particles[node.particle_idx].pos)
// 	}
// }

VERTEX_SHADER :: "assets/shaders/particle.vert.spv"
FRAGMENT_SHADER :: "assets/shaders/particle.frag.spv"

// Constants for stability
CENTRAL_MASS :: 50_000.0
PARTICLE_MASS :: 0.1
INNER_RADIUS :: 50.0
OUTER_RADIUS :: 1000.0

init_stable_disk :: proc(particles: #soa[]nbody.Particle) {
	// 1. Anchor the center
	particles.px[:][0], particles.py[:][0], particles.pz[:][0] = 0, 0, 0
	particles.vx[:][0], particles.vy[:][0], particles.vz[:][0] = 0, 0, 0
	particles.mass[:][0] = CENTRAL_MASS

	for i in 1 ..< len(particles) {
		// Distribute radius using square root for uniform disk density
		// Otherwise, particles clump too much at the center
		r := INNER_RADIUS + (math.sqrt(rand.float32()) * (OUTER_RADIUS - INNER_RADIUS))
		angle := rand.float32() * math.TAU

		px := math.cos(angle) * r
		py := math.sin(angle) * r

		particles.px[:][i] = px
		particles.py[:][i] = py
		particles.pz[:][i] = rand.float32_range(-2, 2) // Tiny bit of vertical jitter

		// 2. Calculate stable orbital velocity considering softening (EPS)
		// Force with softening: F = G*M*r / (r^2 + EPS)^1.5
		// Set Centripetal Force (mv^2/r) equal to Gravitational Force
		r_sq := r * r
		denom := math.pow(r_sq + EPS, 1.5)
		speed := math.sqrt((G * CENTRAL_MASS * r_sq) / denom)

		// 3. Tangential velocity (perpendicular to position vector)
		dir_x := -py / r
		dir_y := px / r

		// Add a tiny bit of random dispersion (5%) so it's not a "perfect" crystal
		dispersion := 1.0 + rand.float32_range(-0.05, 0.05)

		particles.vx[:][i] = dir_x * speed * dispersion
		particles.vy[:][i] = dir_y * speed * dispersion
		particles.vz[:][i] = rand.float32_range(-0.1, 0.1)

		particles.mass[:][i] = PARTICLE_MASS
	}
}

update_particles :: proc(particles: #soa[]nbody.Particle, dt: f32) {
	for &p in particles {
		p.vx += p.ax * dt / p.mass
		p.vy += p.ay * dt / p.mass
		p.vz += p.az * dt / p.mass
		p.px += p.vx * dt
		p.py += p.vy * dt
		p.pz += p.vz * dt
		p.ax = 0
		p.ay = 0
		p.az = 0
	}
}

main :: proc() {
	log_level: log.Level = .Info
	log_opts: log.Options = {.Level, .Line, .Terminal_Color}
	log_ident := "Dial"

	when ODIN_DEBUG {
		log_level = .Debug
		log_opts |= {.Date, .Time, .Short_File_Path}

		track: mem.Tracking_Allocator
		mem.tracking_allocator_init(&track, context.allocator)
		context.allocator = mem.tracking_allocator(&track)

		defer {
			if len(track.allocation_map) > 0 {
				log.errorf("=== %v allocations not freed: ===\n", len(track.allocation_map))
				for _, entry in track.allocation_map {
					log.errorf("- %v bytes @ %v\n", entry.size, entry.location)
				}
			}
			if len(track.bad_free_array) > 0 {
				log.errorf("=== %v incorrect frees: ===\n", len(track.bad_free_array))
				for entry in track.bad_free_array {
					log.errorf("- %p @ %v\n", entry.memory, entry.location)
				}
			}
			mem.tracking_allocator_destroy(&track)
		}
	}

	context.logger = log.create_console_logger(log_level, log_opts, log_ident)

	if !dial.init_engine({.VIDEO, .EVENTS}, {name = "N-Body", size = {600, 600}, flags = {.VULKAN, .HIGH_PIXEL_DENSITY, .RESIZABLE}}, ODIN_DEBUG) do return
	defer dial.destroy_engine()

	dial.set_exit_key(.ESCAPE)

	vertex_shader := dial.create_shader(#load(VERTEX_SHADER, []u32), .Vertex)
	fragment_shader := dial.create_shader(#load(FRAGMENT_SHADER, []u32), .Fragment)
	defer {
		dial.destroy_shader(vertex_shader)
		dial.destroy_shader(fragment_shader)
	}

	CameraData :: struct {
		view, proj: matrix[4, 4]f32,
	}
	gpu_px := dial.create_shared_memory(f32, N_BODIES)
	gpu_py := dial.create_shared_memory(f32, N_BODIES)
	gpu_pz := dial.create_shared_memory(f32, N_BODIES)
	camera_data := dial.create_shared_memory(CameraData)

	InstanceData :: struct {
		x, y, z: rawptr,
		camera:  rawptr,
	}

	particles := make(#soa[]nbody.Particle, N_BODIES)
	defer delete(particles)

	init_stable_disk(particles)

	for !dial.should_quit() {
		dt := dial.delta()
		log.infof("frame completed in %.0fms", dial.delta() * 1000)

		rtb: dial.RenderTargetBuilder
		defer dial.rtb_clear(&rtb)

		nbody.naive_force_soa_simd_threaded(particles[:], 8, G, EPS)
		update_particles(particles[:], dt)

		half_size: f32 = OUTER_RADIUS * 1.05

		camera: CameraData = {
			view = linalg.matrix4_look_at_f32({0, 0, -10}, {0, 0, 0}, {0, 1, 0}),
			proj = linalg.matrix_ortho3d_f32(
				-half_size,
				half_size,
				half_size,
				-half_size,
				-1000,
				1000,
			),
		}

		mem.copy(rawptr(&gpu_px.cpu[0]), rawptr(&particles.px[0]), size_of(f32) * N_BODIES)
		mem.copy(rawptr(&gpu_py.cpu[0]), rawptr(&particles.py[0]), size_of(f32) * N_BODIES)
		mem.copy(rawptr(&gpu_pz.cpu[0]), rawptr(&particles.pz[0]), size_of(f32) * N_BODIES)
		camera_data.cpu^ = camera

		if swapchain, frame_arena, buf, ok := dial.frame_prepare(); ok {
			instance_data := dial.create_shared_frame_memory(InstanceData, frame_arena)
			instance_data.cpu^ = {
				x      = gpu_px.gpu.ptr,
				y      = gpu_py.gpu.ptr,
				z      = gpu_pz.gpu.ptr,
				camera = camera_data.gpu.ptr,
			}

			dial.rtb_set_color_target(&rtb, swapchain, {})
			if dial.begin_rendering(buf, dial.rtb_build_render_pass_desc(&rtb)) {
				dial.set_blend_state(
					buf,
					{
						enable = true,
						alpha_op = .Max,
						color_op = .Add,
						src_color_factor = .One,
						dst_color_factor = .One_Minus_Src_Alpha,
						src_alpha_factor = .One,
						dst_alpha_factor = .One,
						color_write_mask = {.R, .G, .B, .A},
					},
				)
				dial.set_shaders(buf, vertex_shader, fragment_shader)
				dial.draw_instanced(buf, instance_data, {}, 6, N_BODIES)
			}
		}
	}
}
