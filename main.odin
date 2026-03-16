package nbody

import "core:log"
import "core:math"
import glm "core:math/linalg"
import "core:math/rand"
import "core:mem"
import dial "shared:dial"

N_BODIES :: 100

Particle :: struct {
	pos:   [3]f32,
	vel:   [3]f32,
	accel: [3]f32,
	mass:  f32,
}

particles: #soa[N_BODIES]Particle

VERTEX_SHADER :: "assets/shaders/particle.vert.spv"
FRAGMENT_SHADER :: "assets/shaders/particle.frag.spv"

G :: 10
EPS :: 10

naive_force :: proc(particles: #soa[]Particle) {
	for i in 0 ..< len(particles) {
		pos_i := particles.pos[i]
		mass_i := particles.mass[i]
		for j in i + 1 ..< len(particles) {
			pos_j := particles.pos[j]
			mass_j := particles.mass[j]

			diff := pos_j - pos_i
			dist_sq := glm.dot(diff, diff) + EPS

			dist_inv := 1 / math.sqrt(dist_sq)
			dist_inv_cube := dist_inv * dist_inv * dist_inv

			force := diff * G * mass_i * mass_j * dist_inv_cube

			particles.accel[:][i] += force
			particles.accel[:][j] -= force
		}
	}
}

update_particles :: proc(particles: #soa[]Particle, dt: f32) {
	for &p in particles {
		p.vel += p.accel * dt / p.mass
		p.pos += p.vel * dt
		p.accel = {}
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
	gpu_pos := dial.create_shared_memory([3]f32, N_BODIES)
	camera_data := dial.create_shared_memory(CameraData)

	InstanceData :: struct {
		instances: rawptr,
		camera:    rawptr,
	}

	particles[0] = {
		mass = 10000,
	}
	for i in 1 ..< N_BODIES {
		angle := rand.float32_range(0, glm.PI * 2)
		d := rand.float32_range(20, 450)

		px := math.cos(angle) * d
		py := math.sin(angle) * d
		particles.pos[i] = {px, py, 0}

		dir_x := -py
		dir_y := px

		dir := glm.normalize([2]f32{dir_x, dir_y})

		speed := math.sqrt((G * 10000) / d)
		particles.vel[i] = {dir.x * speed, dir.y * speed, 0}
		particles.mass[i] = 1
	}

	for !dial.should_quit() {
		rtb: dial.RenderTargetBuilder
		defer dial.rtb_clear(&rtb)

		half_size: f32 = 500.
		camera: CameraData = {
			view = glm.matrix4_look_at_f32({0, 0, -10}, {0, 0, 0}, {0, 1, 0}),
			proj = glm.matrix_ortho3d_f32(
				-half_size,
				half_size,
				half_size,
				-half_size,
				-1000,
				1000,
			),
		}

		naive_force(particles[:])
		update_particles(particles[:], dial.delta())
		mem.copy(rawptr(&gpu_pos.cpu[0]), rawptr(&particles.pos), size_of([3]f32) * N_BODIES)
		camera_data.cpu^ = camera

		if swapchain, frame_arena, buf, ok := dial.frame_prepare(); ok {
			instance_data := dial.create_shared_frame_memory(InstanceData, frame_arena)
			instance_data.cpu^ = {
				instances = gpu_pos.gpu.ptr,
				camera    = camera_data.gpu.ptr,
			}

			dial.rtb_set_color_target(&rtb, swapchain, {})
			if dial.begin_rendering(buf, dial.rtb_build_render_pass_desc(&rtb)) {
				dial.set_shaders(buf, vertex_shader, fragment_shader)
				dial.draw_instanced(buf, instance_data, {}, 6, N_BODIES)
			}
		}
	}
}
