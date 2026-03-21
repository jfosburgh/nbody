package nbody

import "base:runtime"
import "core:fmt"
import "core:math"
import "core:math/linalg"
import "core:math/rand"
import "core:mem"
import "core:simd"
import "core:sync"
import "core:sys/info"
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
	g: f32 = G,
	eps: f32 = EPS,
	allocator := context.allocator,
) {
	TaskData :: struct {
		particles:   #soa[]Particle,
		start, stop: int,
		g, eps:      f32,
		wg:          ^sync.Wait_Group,
	}

	n_tasks := min(len(global_pool.threads) * THREAD_MULT, len(particles) / WIDTH)
	tasks := make([]TaskData, n_tasks, allocator)
	defer delete(tasks)

	task_handler :: proc(task: thread.Task) {
		data := cast(^TaskData)task.data
		defer sync.wait_group_done(data.wg)

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

	n_per_task := len(particles) / n_tasks + 1

	wg: sync.Wait_Group
	sync.wait_group_add(&wg, n_tasks)

	for index in 0 ..< n_tasks {
		task_data := &tasks[index]
		task_data.particles = particles
		task_data.start = index * n_per_task
		task_data.stop = (index + 1) * n_per_task
		task_data.g = g
		task_data.eps = eps
		task_data.wg = &wg

		thread.pool_add_task(&global_pool, mem.nil_allocator(), task_handler, task_data, index)
	}

	sync.wait_group_wait(&wg)
	global_thread_pool_clear_done()
}

@(private)
WIDTH :: #config(WIDTH, 16)
@(private)
THREADS :: #config(THREADS, -1)
@(private)
THREAD_MULT :: 2

@(private)
horizontal_sum :: proc(v: #simd[WIDTH]f32) -> (s: f32) {
	for i in 0 ..< WIDTH {
		s += simd.extract(v, i)
	}

	return
}

@(private)
iota :: proc($V: typeid/#simd[$N]$E) -> (result: V) {
	for i in 0 ..< N {
		result = simd.replace(result, i, E(i))
	}
	return
}

@(private)
process_force_chunk :: proc(
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

@(private)
process_update_chunk :: proc(particles: #soa[]Particle, dt: f32, mask: #simd[WIDTH]u32) {
	px_ptr := cast(^#simd[WIDTH]f32)particles.px
	py_ptr := cast(^#simd[WIDTH]f32)particles.py
	pz_ptr := cast(^#simd[WIDTH]f32)particles.pz
	vx_ptr := cast(^#simd[WIDTH]f32)particles.vx
	vy_ptr := cast(^#simd[WIDTH]f32)particles.vy
	vz_ptr := cast(^#simd[WIDTH]f32)particles.vz
	ax_ptr := cast(^#simd[WIDTH]f32)particles.ax
	ay_ptr := cast(^#simd[WIDTH]f32)particles.ay
	az_ptr := cast(^#simd[WIDTH]f32)particles.az
	m_ptr := cast(^#simd[WIDTH]f32)particles.mass

	px := simd.masked_load(px_ptr, cast(#simd[WIDTH]f32)0, mask)
	py := simd.masked_load(py_ptr, cast(#simd[WIDTH]f32)0, mask)
	pz := simd.masked_load(pz_ptr, cast(#simd[WIDTH]f32)0, mask)
	vx := simd.masked_load(vx_ptr, cast(#simd[WIDTH]f32)0, mask)
	vy := simd.masked_load(vy_ptr, cast(#simd[WIDTH]f32)0, mask)
	vz := simd.masked_load(vz_ptr, cast(#simd[WIDTH]f32)0, mask)
	ax := simd.masked_load(ax_ptr, cast(#simd[WIDTH]f32)0, mask)
	ay := simd.masked_load(ay_ptr, cast(#simd[WIDTH]f32)0, mask)
	az := simd.masked_load(az_ptr, cast(#simd[WIDTH]f32)0, mask)
	m := simd.masked_load(m_ptr, cast(#simd[WIDTH]f32)0, mask)

	vx += ax * dt / m
	vy += ay * dt / m
	vz += az * dt / m

	px += vx * dt
	py += vy * dt
	pz += vz * dt

	simd.masked_store(px_ptr, px, mask)
	simd.masked_store(py_ptr, py, mask)
	simd.masked_store(pz_ptr, pz, mask)
	simd.masked_store(vx_ptr, vx, mask)
	simd.masked_store(vy_ptr, vy, mask)
	simd.masked_store(vz_ptr, vz, mask)
	simd.masked_store(ax_ptr, cast(#simd[WIDTH]f32)0, mask)
	simd.masked_store(ay_ptr, cast(#simd[WIDTH]f32)0, mask)
	simd.masked_store(az_ptr, cast(#simd[WIDTH]f32)0, mask)
}

naive_force_soa_simd :: proc(particles: #soa[]Particle, g: f32 = G, eps: f32 = EPS) {
	for &p_i in particles {
		temp_particles := particles

		ax, ay, az: #simd[WIDTH]f32
		for len(temp_particles) >= WIDTH {
			process_force_chunk(
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

		if len(temp_particles) > 0 {
			index := iota(#simd[WIDTH]i32)
			mask := simd.lanes_lt(index, cast(#simd[WIDTH]i32)len(temp_particles))
			process_force_chunk(
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

		p_i.ax = horizontal_sum(ax)
		p_i.ay = horizontal_sum(ay)
		p_i.az = horizontal_sum(az)
	}
}

naive_force_soa_simd_threaded :: proc(
	particles: #soa[]Particle,
	g: f32 = G,
	eps: f32 = EPS,
	allocator := context.allocator,
) {
	TaskData :: struct {
		proc_particles: #soa[]Particle,
		all_particles:  #soa[]Particle,
		g, eps:         f32,
		wg:             ^sync.Wait_Group,
	}

	n_tasks := min(len(global_pool.threads) * THREAD_MULT, len(particles) / WIDTH)
	tasks := make([]TaskData, n_tasks, allocator)
	defer delete(tasks)

	task_handler :: proc(task: thread.Task) {
		index := iota(#simd[WIDTH]i32)
		data := cast(^TaskData)task.data
		defer sync.wait_group_done(data.wg)
		for &p_i in data.proc_particles {
			temp_particles := data.all_particles

			ax, ay, az: #simd[WIDTH]f32
			for len(temp_particles) >= WIDTH {
				process_force_chunk(
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

			if len(temp_particles) > 0 {
				mask := simd.lanes_lt(index, cast(#simd[WIDTH]i32)len(temp_particles))
				process_force_chunk(
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

			p_i.ax = horizontal_sum(ax)
			p_i.ay = horizontal_sum(ay)
			p_i.az = horizontal_sum(az)
		}
	}

	n_per_task := len(particles) / n_tasks + 1

	wg: sync.Wait_Group
	sync.wait_group_add(&wg, n_tasks)

	for index in 0 ..< n_tasks {
		task_data := &tasks[index]
		start := index * n_per_task
		stop := min(len(particles), (index + 1) * n_per_task)
		task_data.proc_particles = particles[start:stop]
		task_data.all_particles = particles
		task_data.g = g
		task_data.eps = eps
		task_data.wg = &wg

		thread.pool_add_task(&global_pool, mem.nil_allocator(), task_handler, task_data, index)
	}

	sync.wait_group_wait(&wg)
	global_thread_pool_clear_done()
}

update_particles :: proc(particles: []Particle, dt: f32) {
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

update_particles_soa :: proc(particles: []Particle, dt: f32) {
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

update_particles_soa_threaded :: proc(
	particles: #soa[]Particle,
	dt: f32,
	allocator := context.allocator,
) {
	TaskData :: struct {
		particles: #soa[]Particle,
		dt:        f32,
		wg:        ^sync.Wait_Group,
	}

	n_tasks := min(len(global_pool.threads) * THREAD_MULT, len(particles) / WIDTH)
	tasks := make([]TaskData, n_tasks, allocator)
	defer delete(tasks)

	task_handler :: proc(task: thread.Task) {
		data := cast(^TaskData)task.data
		defer sync.wait_group_done(data.wg)
		particles := data.particles
		dt := data.dt
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

	n_per_task := len(particles) / n_tasks + 1

	wg: sync.Wait_Group
	sync.wait_group_add(&wg, n_tasks)

	for index in 0 ..< n_tasks {
		task_data := &tasks[index]
		start := index * n_per_task
		stop := min(len(particles), (index + 1) * n_per_task)
		task_data.particles = particles[start:stop]
		task_data.dt = dt
		task_data.wg = &wg

		thread.pool_add_task(&global_pool, mem.nil_allocator(), task_handler, task_data, index)
	}

	sync.wait_group_wait(&wg)
	global_thread_pool_clear_done()
}

update_particles_soa_simd :: proc(
	particles: #soa[]Particle,
	dt: f32,
	allocator := context.allocator,
) {
	particles := particles
	for len(particles) >= WIDTH {
		process_update_chunk(particles, dt, max(u32))
		particles = particles[WIDTH:]
	}

	if len(particles) > 0 {
		index := iota(#simd[WIDTH]i32)
		mask := simd.lanes_lt(index, cast(#simd[WIDTH]i32)len(particles))
		process_update_chunk(particles, dt, mask)
	}
}

update_particles_soa_simd_threaded :: proc(
	particles: #soa[]Particle,
	dt: f32,
	allocator := context.allocator,
) {
	TaskData :: struct {
		particles: #soa[]Particle,
		dt:        f32,
		wg:        ^sync.Wait_Group,
	}

	n_tasks := min(len(global_pool.threads) * THREAD_MULT, len(particles) / WIDTH)
	tasks := make([]TaskData, n_tasks, allocator)
	defer delete(tasks)

	task_handler :: proc(task: thread.Task) {
		data := cast(^TaskData)task.data
		defer sync.wait_group_done(data.wg)
		particles := data.particles
		dt := data.dt

		for len(particles) >= WIDTH {
			process_update_chunk(particles, dt, max(u32))
			particles = particles[WIDTH:]
		}

		if len(particles) > 0 {
			index := iota(#simd[WIDTH]i32)
			mask := simd.lanes_lt(index, cast(#simd[WIDTH]i32)len(particles))
			process_update_chunk(particles, dt, mask)
		}
	}

	n_per_task := len(particles) / n_tasks + 1

	wg: sync.Wait_Group
	sync.wait_group_add(&wg, n_tasks)

	for index in 0 ..< n_tasks {
		task_data := &tasks[index]
		start := index * n_per_task
		stop := min(len(particles), (index + 1) * n_per_task)
		task_data.particles = particles[start:stop]
		task_data.dt = dt
		task_data.wg = &wg

		thread.pool_add_task(&global_pool, mem.nil_allocator(), task_handler, task_data, index)
	}

	sync.wait_group_wait(&wg)
	global_thread_pool_clear_done()
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
compare_force_results :: proc(aos: []Particle, soa: #soa[]Particle, label: string) {
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
compare_update_results :: proc(aos: []Particle, soa: #soa[]Particle, label: string) {
	EPSILON :: 1e-8

	for i in 0 ..< len(aos) {
		dpx := math.abs(aos[i].px - soa.px[i])
		dpy := math.abs(aos[i].py - soa.py[i])
		dpz := math.abs(aos[i].pz - soa.pz[i])
		dvx := math.abs(aos[i].vx - soa.vx[i])
		dvy := math.abs(aos[i].vy - soa.vy[i])
		dvz := math.abs(aos[i].vz - soa.vz[i])
		dax := math.abs(aos[i].ax - soa.ax[i])
		day := math.abs(aos[i].ay - soa.ay[i])
		daz := math.abs(aos[i].az - soa.az[i])

		if dpx > EPSILON ||
		   dpy > EPSILON ||
		   dpz > EPSILON ||
		   dvx > EPSILON ||
		   dvy > EPSILON ||
		   dvz > EPSILON ||
		   dax > EPSILON ||
		   day > EPSILON ||
		   daz > EPSILON {
			fmt.panicf(
				"Update validation failed for %s at index %d!\n  AOS Pos: [%f, %f, %f]\n  SOA Pos: [%f, %f, %f]",
				label,
				i,
				aos[i].px,
				aos[i].py,
				aos[i].pz,
				soa.px[i],
				soa.py[i],
				soa.pz[i],
			)
		}
	}
	fmt.printfln(" - %s results match Naive AOS with tolerance %f.", label, EPSILON)
}

@(private)
N :: 5000
@(private)
ITERATIONS :: 100

global_pool: thread.Pool

init_global_thread_pool :: proc(num_threads: int) {
	thread.pool_init(&global_pool, context.allocator, num_threads)
	thread.pool_start(&global_pool)
}

global_thread_pool_clear_done :: proc() {
	for {
		if _, ok := thread.pool_pop_done(&global_pool); !ok do return
	}
}

shutdown_global_thread_pool :: proc() {
	global_thread_pool_clear_done()
	thread.pool_finish(&global_pool)
	thread.pool_destroy(&global_pool)
}


@(private)
main :: proc() {
	threads := THREADS
	if threads == -1 do _, threads, _ = info.cpu_core_count()
	init_global_thread_pool(threads)
	defer shutdown_global_thread_pool()

	validate()
	benchmark()
	find_crossover()
}

validate :: proc() {
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
	compare_force_results(gold_aos, test_soa, "Naive SOA")

	mem.zero_explicit(rawptr(&test_soa.ax[:][0]), size_of(f32) * N * 3)
	naive_force_soa_threaded(test_soa)
	compare_force_results(gold_aos, test_soa, "Threaded SOA")

	mem.zero_explicit(rawptr(&test_soa.ax[:][0]), size_of(f32) * N * 3)
	naive_force_soa_simd(test_soa)
	compare_force_results(gold_aos, test_soa, "SIMD SOA")

	mem.zero_explicit(rawptr(&test_soa.ax[:][0]), size_of(f32) * N * 3)
	naive_force_soa_simd_threaded(test_soa)
	compare_force_results(gold_aos, test_soa, "Threaded SIMD SOA")

	delete(gold_aos)
	delete(test_soa)

	fmt.println("--- Validating Update Implementations ---")
	DT :: 0.01

	gold_aos_up := make([]Particle, N)
	mem.copy(raw_data(gold_aos_up), raw_data(particles_base), size_of(Particle) * N)
	for i in 0 ..< N {
		gold_aos_up[i].ax, gold_aos_up[i].ay, gold_aos_up[i].az = 10, 10, 10
		gold_aos_up[i].mass = 1
	}
	update_particles(gold_aos_up, DT)

	test_soa_up := make(#soa[]Particle, N)
	reset_update_test_soa :: proc(target: #soa[]Particle, base: #soa[]Particle) {
		mem.copy(rawptr(&target.px[:][0]), rawptr(&base.px[:][0]), size_of(f32) * N * 10)
		for i in 0 ..< N {
			target.ax[:][i], target.ay[:][i], target.az[:][i] = 10, 10, 10
			target.mass[:][i] = 1
		}
	}

	reset_update_test_soa(test_soa_up, particles_soa_base)
	update_particles_soa_threaded(test_soa_up, DT)
	compare_update_results(gold_aos_up, test_soa_up, "Threaded SOA Update")

	reset_update_test_soa(test_soa_up, particles_soa_base)
	update_particles_soa_simd(test_soa_up, DT)
	compare_update_results(gold_aos_up, test_soa_up, "SIMD SOA Update")

	reset_update_test_soa(test_soa_up, particles_soa_base)
	update_particles_soa_simd_threaded(test_soa_up, DT)
	compare_update_results(gold_aos_up, test_soa_up, "Threaded SIMD SOA Update")

	delete(gold_aos_up)
	delete(test_soa_up)

	fmt.println("All validations passed!\n")
}

benchmark :: proc() {
	fmt.println("--- Benchmarking Force Implementations ---")
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

	fmt.println("\n--- Benchmarking Update Implementations ---")

	time_update_particles :: proc() {
		particles := make([]Particle, N)
		defer delete(particles)
		update_particles(particles[:], 0.01)
	}
	fmt.printfln("Naive Update -- O(n):")
	mean_t, min_t, max_t = timeit(time_update_particles, ITERATIONS)
	fmt.printfln("Avg: %0.2fms -- Min: %0.2fms -- Max: %0.2fms", mean_t, min_t, max_t)

	time_update_particles_soa_threaded :: proc() {
		particles := make(#soa[]Particle, N)
		defer delete(particles)
		update_particles_soa_threaded(particles[:], 0.01)
	}
	fmt.printfln("\nThreaded SOA Update -- O(n):")
	mean_t, min_t, max_t = timeit(time_update_particles_soa_threaded, ITERATIONS)
	fmt.printfln("Avg: %0.2fms -- Min: %0.2fms -- Max: %0.2fms", mean_t, min_t, max_t)

	time_update_particles_soa_simd :: proc() {
		particles := make(#soa[]Particle, N)
		defer delete(particles)
		update_particles_soa_simd(particles[:], 0.01)
	}
	fmt.printfln("\nSIMD SOA Update -- O(n):")
	mean_t, min_t, max_t = timeit(time_update_particles_soa_simd, ITERATIONS)
	fmt.printfln("Avg: %0.2fms -- Min: %0.2fms -- Max: %0.2fms", mean_t, min_t, max_t)

	time_update_particles_soa_simd_threaded :: proc() {
		particles := make(#soa[]Particle, N)
		defer delete(particles)
		update_particles_soa_simd_threaded(particles[:], 0.01)
	}
	fmt.printfln("\nThreaded SIMD SOA Update -- O(n):")
	mean_t, min_t, max_t = timeit(time_update_particles_soa_simd_threaded, ITERATIONS)
	fmt.printfln("Avg: %0.2fms -- Min: %0.2fms -- Max: %0.2fms", mean_t, min_t, max_t)
}

find_crossover :: proc() {
	fmt.println("\n--- Finding Crossover Points ---")

	// Force Crossover Search
	fmt.println("Searching for Force Crossover (SIMD vs SIMD Threaded)...")
	for n := 100; n <= 5000; n += 100 {
		particles := make(#soa[]Particle, n)
		for i in 0 ..< n do particles.mass[:][i] = 1.0
		defer delete(particles)

		t_simd := 0.0
		for _ in 0 ..< 10 {
			now := time.now()
			naive_force_soa_simd(particles)
			t_simd += time.duration_milliseconds(time.since(now))
		}
		t_simd /= 10.0

		t_threaded := 0.0
		for _ in 0 ..< 10 {
			now := time.now()
			naive_force_soa_simd_threaded(particles, 8)
			t_threaded += time.duration_milliseconds(time.since(now))
		}
		t_threaded /= 10.0

		if t_threaded < t_simd {
			fmt.printfln(
				" > Force Crossover found at N = %d (SIMD: %0.4fms, Threaded: %0.4fms)",
				n,
				t_simd,
				t_threaded,
			)
			break
		}
	}

	// Update Crossover Search
	fmt.println("\nSearching for Update Crossover (SIMD vs SIMD Threaded)...")
	// Update is O(n), so we need much larger N to overcome thread overhead
	for n := 10000; n <= 2000000; n += 50000 {
		particles := make(#soa[]Particle, n)
		for i in 0 ..< n do particles.mass[:][i] = 1.0
		defer delete(particles)

		t_simd := 0.0
		for _ in 0 ..< 50 {
			now := time.now()
			update_particles_soa_simd(particles, 0.01)
			t_simd += time.duration_milliseconds(time.since(now))
		}
		t_simd /= 50.0

		t_threaded := 0.0
		for _ in 0 ..< 50 {
			now := time.now()
			update_particles_soa_simd_threaded(particles, 0.01)
			t_threaded += time.duration_milliseconds(time.since(now))
		}
		t_threaded /= 50.0

		if t_threaded < t_simd {
			fmt.printfln(
				" > Update Crossover found at N = %d (SIMD: %0.4fms, Threaded: %0.4fms)",
				n,
				t_simd,
				t_threaded,
			)
			break
		}

		if n % 500000 == 0 {
			fmt.printfln("   ...Checked up to N = %d, Threaded is still slower.", n)
		}
	}
}
