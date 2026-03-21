package main

import "core:log"
import "core:math"
import "core:math/linalg"
import "core:math/rand"
import "core:mem"
import "core:os"
import dial "shared:dial"
import nbody "src"

N_BODIES :: 8000
G :: 100
EPS :: 10

VERTEX_SHADER :: "assets/shaders/particle.vert.spv"
FRAGMENT_SHADER :: "assets/shaders/particle.frag.spv"

CENTRAL_MASS :: 50_000.0
PARTICLE_MASS :: 0.1
INNER_RADIUS :: 50.0
OUTER_RADIUS :: 1000.0

init_stable_disk :: proc(particles: #soa[]nbody.Particle) {
	particles.px[:][0], particles.py[:][0], particles.pz[:][0] = 0, 0, 0
	particles.vx[:][0], particles.vy[:][0], particles.vz[:][0] = 0, 0, 0
	particles.mass[:][0] = CENTRAL_MASS

	for i in 1 ..< len(particles) {
		r := INNER_RADIUS + (math.sqrt(rand.float32()) * (OUTER_RADIUS - INNER_RADIUS))
		angle := rand.float32() * math.TAU

		px := math.cos(angle) * r
		py := math.sin(angle) * r

		particles.px[:][i] = px
		particles.py[:][i] = py
		particles.pz[:][i] = rand.float32_range(-2, 2)

		r_sq := r * r
		denom := math.pow(r_sq + EPS, 1.5)
		speed := math.sqrt((G * CENTRAL_MASS * r_sq) / denom)

		dir_x := -py / r
		dir_y := px / r

		dispersion := 1.0 + rand.float32_range(-0.05, 0.05)

		particles.vx[:][i] = dir_x * speed * dispersion
		particles.vy[:][i] = dir_y * speed * dispersion
		particles.vz[:][i] = rand.float32_range(-0.1, 0.1)

		particles.mass[:][i] = PARTICLE_MASS
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

	nbody.init_global_thread_pool(os.get_processor_core_count())
	defer nbody.shutdown_global_thread_pool()

	for !dial.should_quit() {
		dt := dial.delta()

		rtb: dial.RenderTargetBuilder
		defer dial.rtb_clear(&rtb)

		nbody.naive_force_soa_simd_threaded(particles[:], G, EPS)
		nbody.update_particles_soa_simd(particles[:], dt)

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
