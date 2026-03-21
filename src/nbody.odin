package nbody

import "core:fmt"
import "core:math"
import "core:math/linalg"
import "core:math/rand"
import "core:mem"
import "core:simd"
import "core:thread"
import "core:time"


G :: 1000
EPS :: 10

Particle :: struct {
	px, py, pz: f32,
	vx, vy, vz: f32,
	ax, ay, az: f32,
	mass:       f32,
}

calculate_force :: #force_inline proc(
	m1, m2: f32,
	px1, py1, pz1, px2, py2, pz2: f32,
	g: f32 = G,
	eps: f32 = EPS,
) -> (
	x, y, z: f32,
) {
	dx := px2 - px1
	dy := py2 - py1
	dz := pz2 - pz1
	dist_sq := dx * dx + dy * dy + dz * dz + eps

	dist_inv := 1 / math.sqrt(dist_sq)
	dist_inv_cube := dist_inv * dist_inv * dist_inv

	mag := g * m1 * m2 * dist_inv_cube

	return dx * mag, dy * mag, dz * mag
}

naive_force :: proc(particles: []Particle, g: f32 = G, eps: f32 = EPS) {
	n := len(particles)
	for &a, i in particles {
		for &b, j in particles {
			if i == j do continue
			fx, fy, fz := calculate_force(
				a.mass,
				b.mass,
				a.px,
				a.py,
				a.pz,
				b.px,
				b.py,
				b.pz,
				g,
				eps,
			)
			a.ax += fx
			a.ay += fy
			a.az += fz
		}
	}
}

naive_force_soa :: proc(particles: #soa[]Particle, g: f32 = G, eps: f32 = EPS) {
	for i in 0 ..< len(particles) {
		for j in 0 ..< len(particles) {
			if i == j do continue
			fx, fy, fz := calculate_force(
				particles.mass[i],
				particles.mass[j],
				particles.px[i],
				particles.py[i],
				particles.pz[i],
				particles.px[j],
				particles.py[j],
				particles.pz[j],
				g,
				eps,
			)
			particles.ax[:][i] += fx
			particles.ay[:][i] += fy
			particles.az[:][i] += fz
		}
	}
}

naive_force_soa_threaded :: proc(
	particles: #soa[]Particle,
	threads := 8,
	g: f32 = G,
	eps: f32 = EPS,
	allocator := context.allocator,
) {
	pool: thread.Pool
	thread.pool_init(&pool, allocator, threads)
	thread.pool_start(&pool)

	TaskData :: struct {
		particles:   #soa[]Particle,
		start, stop: int,
		g, eps:      f32,
	}

	tasks := make([]TaskData, threads, allocator)

	task_handler :: proc(task: thread.Task) {
		data := cast(^TaskData)task.data
		particles := data.particles
		stop := min(data.stop, len(particles))
		for i in data.start ..< stop {
			for j in 0 ..< len(particles) {
				if i == j do continue
				fx, fy, fz := calculate_force(
					particles.mass[i],
					particles.mass[j],
					particles.px[i],
					particles.py[i],
					particles.pz[i],
					particles.px[j],
					particles.py[j],
					particles.pz[j],
					data.g,
					data.eps,
				)
				particles.ax[:][i] += fx
				particles.ay[:][i] += fy
				particles.az[:][i] += fz
			}
		}
	}

	n_per_task := len(particles) / threads + 1

	for index in 0 ..< threads {
		task_data := &tasks[index]
		task_data.particles = particles
		task_data.start = index * n_per_task
		task_data.stop = (index + 1) * n_per_task
		task_data.g = g
		task_data.eps = eps

		thread.pool_add_task(&pool, mem.nil_allocator(), task_handler, task_data, index)
	}

	thread.pool_finish(&pool)
	delete(tasks)
	thread.pool_destroy(&pool)
}

@(private)
WIDTH :: #config(WIDTH, 16)

@(private)
horizontal_sum :: proc(v: #simd[WIDTH]f32) -> (s: f32) {
	for i in 0 ..< WIDTH {
		s += simd.extract(v, i)
	}

	return
}

// set [WIDTH]E to 0..WIDTH-1
@(private)
iota :: proc($V: typeid/#simd[$N]$E) -> (result: V) {
	for i in 0 ..< N {
		result = simd.replace(result, i, E(i))
	}
	return
}

@(private)
process_chunk :: proc(
	particles: #soa[]Particle,
	x, y, z, m: f32,
	ax, ay, az: ^#simd[WIDTH]f32,
	g: f32 = G,
	eps: f32 = EPS,
	mask: #simd[WIDTH]u32,
) {
	px_ptr := cast(^#simd[WIDTH]f32)particles.px
	py_ptr := cast(^#simd[WIDTH]f32)particles.py
	pz_ptr := cast(^#simd[WIDTH]f32)particles.pz
	m_ptr := cast(^#simd[WIDTH]f32)particles.mass

	px := simd.masked_load(px_ptr, cast(#simd[WIDTH]f32)0, mask)
	py := simd.masked_load(py_ptr, cast(#simd[WIDTH]f32)0, mask)
	pz := simd.masked_load(pz_ptr, cast(#simd[WIDTH]f32)0, mask)
	m_j := simd.masked_load(m_ptr, cast(#simd[WIDTH]f32)0, mask)

	dx := px - x
	dy := py - y
	dz := pz - z

	dist_sq := dx * dx + dy * dy + dz * dz + eps
	dist_inv := simd.recip(simd.sqrt(dist_sq))
	dist_inv_cube := dist_inv * dist_inv * dist_inv

	mag := g * m * m_j * dist_inv_cube

	ax^ += dx * mag
	ay^ += dy * mag
	az^ += dz * mag
}

naive_force_soa_simd :: proc(particles: #soa[]Particle, g: f32 = G, eps: f32 = EPS) {
	for i in 0 ..< len(particles) {
		p_i := particles[i]
		temp_particles := particles

		ax, ay, az: #simd[WIDTH]f32
		for len(temp_particles) >= WIDTH {
			process_chunk(
				temp_particles,
				p_i.px,
				p_i.py,
				p_i.pz,
				p_i.mass,
				&ax,
				&ay,
				&az,
				g,
				eps,
				max(u32),
			)
			temp_particles = temp_particles[WIDTH:]
		}

		if len(particles) > 0 {
			index := iota(#simd[WIDTH]i32)
			mask := simd.lanes_le(index, cast(#simd[WIDTH]i32)len(temp_particles))
			process_chunk(
				temp_particles,
				p_i.px,
				p_i.py,
				p_i.pz,
				p_i.mass,
				&ax,
				&ay,
				&az,
				g,
				eps,
				mask,
			)
		}

		particles.ax[:][i] = horizontal_sum(ax)
		particles.ay[:][i] = horizontal_sum(ay)
		particles.az[:][i] = horizontal_sum(az)
	}
}

naive_force_soa_simd_threaded :: proc(
	particles: #soa[]Particle,
	threads := 8,
	g: f32 = G,
	eps: f32 = EPS,
	allocator := context.allocator,
) {
	pool: thread.Pool
	thread.pool_init(&pool, allocator, threads)
	thread.pool_start(&pool)

	TaskData :: struct {
		particles:   #soa[]Particle,
		start, stop: int,
		g, eps:      f32,
	}

	tasks := make([]TaskData, threads, allocator)

	task_handler :: proc(task: thread.Task) {
		data := cast(^TaskData)task.data
		particles := data.particles
		stop := min(data.stop, len(particles))
		for i in data.start ..< stop {
			p_i := particles[i]
			temp_particles := particles

			ax, ay, az: #simd[WIDTH]f32
			for len(temp_particles) >= WIDTH {
				process_chunk(
					temp_particles,
					p_i.px,
					p_i.py,
					p_i.pz,
					p_i.mass,
					&ax,
					&ay,
					&az,
					data.g,
					data.eps,
					max(u32),
				)
				temp_particles = temp_particles[WIDTH:]
			}

			if len(particles) > 0 {
				index := iota(#simd[WIDTH]i32)
				mask := simd.lanes_le(index, cast(#simd[WIDTH]i32)len(temp_particles))
				process_chunk(
					temp_particles,
					p_i.px,
					p_i.py,
					p_i.pz,
					p_i.mass,
					&ax,
					&ay,
					&az,
					data.g,
					data.eps,
					mask,
				)
			}

			particles.ax[:][i] = horizontal_sum(ax)
			particles.ay[:][i] = horizontal_sum(ay)
			particles.az[:][i] = horizontal_sum(az)
		}
	}

	n_per_task := len(particles) / threads + 1

	for index in 0 ..< threads {
		task_data := &tasks[index]
		task_data.particles = particles
		task_data.start = index * n_per_task
		task_data.stop = (index + 1) * n_per_task
		task_data.g = g
		task_data.eps = eps

		thread.pool_add_task(&pool, mem.nil_allocator(), task_handler, task_data, index)
	}

	thread.pool_finish(&pool)
	delete(tasks)
	thread.pool_destroy(&pool)
	for i in 0 ..< len(particles) {
	}
}

@(private)
timeit :: proc(
	to_time: proc(),
	iterations := 100,
	warmup_iterations := 5,
) -> (
	mean_t, min_t, max_t: f64,
) {
	for _ in 0 ..< warmup_iterations {
		to_time()
	}

	min_t = max(f64)

	for i in 0 ..< iterations {
		now := time.now()
		to_time()
		t := time.duration_milliseconds(time.since(now))

		min_t = min(min_t, t)
		max_t = max(max_t, t)
		mean_t += t
	}

	return mean_t / f64(iterations), min_t, max_t
}

@(private)
compare_results :: proc(aos: []Particle, soa: #soa[]Particle, label: string) {
	EPSILON :: 1e-1

	for i in 0 ..< len(aos) {
		dx := math.abs(aos[i].ax - soa.ax[i])
		dy := math.abs(aos[i].ay - soa.ay[i])
		dz := math.abs(aos[i].az - soa.az[i])

		if dx > EPSILON || dy > EPSILON || dz > EPSILON {
			fmt.panicf(
				"Validation failed for %s at index %d!\n  AOS: [%f, %f, %f]\n  SOA: [%f, %f, %f]",
				label,
				i,
				aos[i].ax,
				aos[i].ay,
				aos[i].az,
				soa.ax[i],
				soa.ay[i],
				soa.az[i],
			)
		}
	}
	fmt.printfln(" - %s results match Naive AOS with tolerance %f.", label, EPSILON)
}

@(private)
N :: 5000
@(private)
ITERATIONS :: 100

@(private)
main :: proc() {
	particles_base := make([]Particle, N)
	particles_soa_base := make(#soa[]Particle, N)
	defer delete(particles_base)
	defer delete(particles_soa_base)

	center_mass: f32 = 1000
	particles_base[0].mass = center_mass
	particles_soa_base.mass[:][0] = center_mass

	for i in 1 ..< N {
		angle := rand.float32_range(0, linalg.PI * 2)
		d := rand.float32_range(20, 450)
		px := math.cos(angle) * d
		py := math.sin(angle) * d

		particles_base[i].px, particles_base[i].py = px, py
		particles_soa_base.px[:][i], particles_soa_base.py[:][i] = px, py

		dir := linalg.normalize([2]f32{-py, px})
		speed := math.sqrt((G * center_mass) / d)

		particles_base[i].vx, particles_base[i].vy = dir.x * speed, dir.y * speed
		particles_base[i].mass = 1
		particles_soa_base.vx[:][i], particles_soa_base.vy[:][i] = dir.x * speed, dir.y * speed
		particles_soa_base.mass[:][i] = 1
	}

	fmt.println("--- Validating Implementations ---")

	gold_aos := make([]Particle, N)
	mem.copy(raw_data(gold_aos), raw_data(particles_base), size_of(Particle) * N)
	naive_force(gold_aos)

	test_soa := make(#soa[]Particle, N)
	mem.copy(
		rawptr(&test_soa.px[:][0]),
		rawptr(&particles_soa_base.px[:][0]),
		size_of(f32) * N * 10,
	)
	naive_force_soa(test_soa)
	compare_results(gold_aos, test_soa, "Naive SOA")

	mem.zero_explicit(rawptr(&test_soa.ax[:][0]), size_of(f32) * N * 3)
	naive_force_soa_threaded(test_soa, 8)
	compare_results(gold_aos, test_soa, "Threaded SOA")

	mem.zero_explicit(rawptr(&test_soa.ax[:][0]), size_of(f32) * N * 3)
	naive_force_soa_simd(test_soa)
	compare_results(gold_aos, test_soa, "SIMD SOA")

	mem.zero_explicit(rawptr(&test_soa.ax[:][0]), size_of(f32) * N * 3)
	naive_force_soa_simd_threaded(test_soa, 8)
	compare_results(gold_aos, test_soa, "Threaded SIMD SOA")

	delete(gold_aos)
	delete(test_soa)
	fmt.println("All validations passed!\n")

	fmt.println("--- Benchmarking Implementations ---")
	fmt.printfln("N-Body computation comparison with %d bodies", N)
	time_naive_force :: proc() {
		particles := make([]Particle, N)
		defer delete(particles)
		naive_force(particles[:])
	}
	fmt.printfln("Naive Approach -- O(n):")
	mean_t, min_t, max_t := timeit(time_naive_force, ITERATIONS)
	fmt.printfln("Avg: %0.2fms -- Min: %0.2fms -- Max: %0.2fms", mean_t, min_t, max_t)

	time_naive_force_soa :: proc() {
		particles := make(#soa[]Particle, N)
		defer delete(particles)
		naive_force_soa(particles[:])
	}
	fmt.printfln("\nNaive Approach (SOA) -- O(n):")
	mean_t, min_t, max_t = timeit(time_naive_force_soa, ITERATIONS)
	fmt.printfln("Avg: %0.2fms -- Min: %0.2fms -- Max: %0.2fms", mean_t, min_t, max_t)

	time_naive_force_soa_threaded :: proc() {
		particles := make(#soa[]Particle, N)
		defer delete(particles)
		naive_force_soa_threaded(particles[:], 8)
	}
	fmt.printfln("\nNaive Approach (SOA, threaded) -- O(n):")
	mean_t, min_t, max_t = timeit(time_naive_force_soa_threaded, ITERATIONS)
	fmt.printfln("Avg: %0.2fms -- Min: %0.2fms -- Max: %0.2fms", mean_t, min_t, max_t)

	time_naive_force_soa_simd :: proc() {
		particles := make(#soa[]Particle, N)
		defer delete(particles)
		naive_force_soa_simd(particles[:])
	}
	fmt.printfln("\nNaive Approach (SOA, SIMD) -- O(n):")
	mean_t, min_t, max_t = timeit(time_naive_force_soa_simd, ITERATIONS)
	fmt.printfln("Avg: %0.2fms -- Min: %0.2fms -- Max: %0.2fms", mean_t, min_t, max_t)

	time_naive_force_soa_simd_threaded :: proc() {
		particles := make(#soa[]Particle, N)
		defer delete(particles)
		naive_force_soa_simd_threaded(particles[:], 8)
	}
	fmt.printfln("\nNaive Approach (SOA, SIMD, threaded) -- O(n):")
	mean_t, min_t, max_t = timeit(time_naive_force_soa_simd_threaded, ITERATIONS)
	fmt.printfln("Avg: %0.2fms -- Min: %0.2fms -- Max: %0.2fms", mean_t, min_t, max_t)
}
