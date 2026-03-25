package nbody

import "base:runtime"
import "core:math"
import "core:mem"
import "core:simd"
import "core:sync"
import "core:thread"


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
	for &a, i in particles[:n - 1] {
		for &b in particles[i + 1:] {
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
			b.ax -= fx
			b.ay -= fy
			b.az -= fz
		}
	}
}

naive_force_soa :: proc(particles: #soa[]Particle, g: f32 = G, eps: f32 = EPS) {
	n := len(particles)
	for &a, i in particles[:n - 1] {
		for &b in particles[i + 1:] {
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
			b.ax -= fx
			b.ay -= fy
			b.az -= fz
		}
	}
}

naive_force_threaded :: proc(
	particles: []Particle,
	g: f32 = G,
	eps: f32 = EPS,
	allocator := context.allocator,
) {
	TaskData :: struct {
		particles:   []Particle,
		start, stop: int,
		g, eps:      f32,
		wg:          ^sync.Wait_Group,
	}

	n_tasks := len(global_pool.threads) * THREAD_MULT
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
					particles[i].mass,
					particles[j].mass,
					particles[i].px,
					particles[i].py,
					particles[i].pz,
					particles[j].px,
					particles[j].py,
					particles[j].pz,
					data.g,
					data.eps,
				)
				particles[i].ax += fx
				particles[i].ay += fy
				particles[i].az += fz
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

update_particles_soa :: proc(particles: #soa[]Particle, dt: f32) {
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
