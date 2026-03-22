#+private
package nbody

import "core:fmt"
import "core:math"
import "core:math/linalg"
import "core:math/rand"
import "core:mem"
import "core:sys/info"
import "core:time"

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

EPSILON :: 1e-1

compare_force_results_aos :: proc(
	aos: []Particle,
	aos_test: []Particle,
	label: string,
	epsilon: f32 = EPSILON,
) {

	for i in 0 ..< len(aos) {
		dx := math.abs(aos[i].ax - aos_test[i].ax)
		dy := math.abs(aos[i].ay - aos_test[i].ay)
		dz := math.abs(aos[i].az - aos_test[i].az)

		if dx > epsilon || dy > epsilon || dz > epsilon {
			fmt.panicf(
				"Validation failed for %s at index %d!\n  GT: [%f, %f, %f]\n  TEST: [%f, %f, %f]",
				label,
				i,
				aos[i].ax,
				aos[i].ay,
				aos[i].az,
				aos_test[i].ax,
				aos_test[i].ay,
				aos_test[i].az,
			)
		}
	}
	fmt.printfln(" - %s results match Naive AOS with tolerance %f.", label, epsilon)
}

compare_force_results :: proc(
	aos: []Particle,
	soa: #soa[]Particle,
	label: string,
	epsilon: f32 = EPSILON,
) {

	for i in 0 ..< len(aos) {
		dx := math.abs(aos[i].ax - soa.ax[i])
		dy := math.abs(aos[i].ay - soa.ay[i])
		dz := math.abs(aos[i].az - soa.az[i])

		if dx > epsilon || dy > epsilon || dz > epsilon {
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
	fmt.printfln(" - %s results match Naive AOS with tolerance %f.", label, epsilon)
}

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

N :: 5000
ITERATIONS :: 100

main :: proc() {
	threads := THREADS
	if threads == -1 do _, threads, _ = info.cpu_core_count()
	init_global_thread_pool(threads)
	defer shutdown_global_thread_pool()

	validate()
	benchmark()
	// find_crossover()
	find_limits()
}

init_aos :: proc(particles: []Particle) {
	center_mass: f32 = 1000
	particles[0].mass = center_mass

	for i in 1 ..< N {
		px := rand.float32_range(-450, 450)
		py := rand.float32_range(-450, 450)
		pz := rand.float32_range(-450, 450)

		particles[i].px, particles[i].py, particles[i].pz = px, py, pz

		dir := linalg.normalize([3]f32{-py, px, pz})
		speed := math.sqrt(
			(G * center_mass) / linalg.distance([3]f32{0, 0, 0}, [3]f32{px, py, pz}),
		)

		particles[i].vx, particles[i].vy, particles[i].vz =
			dir.x * speed, dir.y * speed, dir.z * speed
		particles[i].mass = 1
	}
}

init_soa :: proc(particles: #soa[]Particle) {
	center_mass: f32 = 1000
	particles.mass[:][0] = center_mass

	for i in 1 ..< N {
		px := rand.float32_range(-450, 450)
		py := rand.float32_range(-450, 450)
		pz := rand.float32_range(-450, 450)

		particles.px[:][i], particles.py[:][i], particles.pz[:][i] = px, py, pz

		dir := linalg.normalize([3]f32{-py, px, pz})
		speed := math.sqrt(
			(G * center_mass) / linalg.distance([3]f32{0, 0, 0}, [3]f32{px, py, pz}),
		)

		particles.vx[:][i], particles.vy[:][i], particles.vz[:][i] =
			dir.x * speed, dir.y * speed, dir.z * speed
		particles.mass[:][i] = 1
	}
}

init_both :: proc(particles: []Particle, particles_soa: #soa[]Particle) {
	center_mass: f32 = 1000
	particles[0].mass = center_mass
	particles_soa.mass[:][0] = center_mass

	for i in 1 ..< N {
		px := rand.float32_range(-450, 450)
		py := rand.float32_range(-450, 450)
		pz := rand.float32_range(-450, 450)

		particles[i].px, particles[i].py, particles[i].pz = px, py, pz
		particles_soa.px[:][i], particles_soa.py[:][i], particles_soa.pz[:][i] = px, py, pz

		dir := linalg.normalize([3]f32{-py, px, pz})
		speed := math.sqrt(
			(G * center_mass) / linalg.distance([3]f32{0, 0, 0}, [3]f32{px, py, pz}),
		)

		particles[i].vx, particles[i].vy, particles[i].vz =
			dir.x * speed, dir.y * speed, dir.z * speed
		particles[i].mass = 1
		particles_soa.vx[:][i], particles_soa.vy[:][i], particles_soa.vz[:][i] =
			dir.x * speed, dir.y * speed, dir.z * speed
		particles_soa.mass[:][i] = 1
	}
}

validate :: proc() {
	particles_base := make([]Particle, N)
	particles_soa_base := make(#soa[]Particle, N)
	init_both(particles_base, particles_soa_base)

	fmt.println("--- Validating Implementations ---")

	gold_aos := make([]Particle, N)
	test_aos := make([]Particle, N)
	mem.copy(raw_data(gold_aos), raw_data(particles_base), size_of(Particle) * N)
	mem.copy(raw_data(test_aos), raw_data(particles_base), size_of(Particle) * N)
	naive_force(gold_aos)

	test_soa := make(#soa[]Particle, N)
	mem.copy(
		rawptr(&test_soa.px[:][0]),
		rawptr(&particles_soa_base.px[:][0]),
		size_of(f32) * N * 10,
	)
	naive_force_soa(test_soa)
	compare_force_results(gold_aos, test_soa, "Naive SOA")

	naive_force_threaded(test_aos)
	compare_force_results_aos(gold_aos, test_aos, "Threaded AOS")

	mem.zero_explicit(rawptr(&test_soa.ax[:][0]), size_of(f32) * N * 3)
	naive_force_soa_threaded(test_soa)
	compare_force_results(gold_aos, test_soa, "Threaded SOA")

	mem.zero_explicit(rawptr(&test_soa.ax[:][0]), size_of(f32) * N * 3)
	naive_force_soa_simd(test_soa)
	compare_force_results(gold_aos, test_soa, "SIMD SOA")

	mem.zero_explicit(rawptr(&test_soa.ax[:][0]), size_of(f32) * N * 3)
	naive_force_soa_simd_threaded(test_soa)
	compare_force_results(gold_aos, test_soa, "Threaded SIMD SOA")

	mem.zero_explicit(rawptr(&test_soa.ax[:][0]), size_of(f32) * N * 3)
	t: Octree
	octree_init(&t, &test_soa, {0, 0, 0}, 500)
	defer octree_destroy(&t)
	bh_simulate(&t, {0, 0, 0}, 500, 0.0)
	compare_force_results(gold_aos, test_soa, "Barnes Hut", 1)

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

g_particles_aos: []Particle
g_particles_soa: #soa[]Particle

benchmark :: proc() {
	g_particles_aos = make([]Particle, N)
	g_particles_soa = make(#soa[]Particle, N)
	init_both(g_particles_aos, g_particles_soa)
	defer delete(g_particles_aos)
	defer delete(g_particles_soa)

	fmt.println("--- Benchmarking Force Implementations ---")
	fmt.printfln("N-Body computation comparison with %d bodies", N)

	time_naive_force :: proc() {
		naive_force(g_particles_aos)
	}
	fmt.printfln("Naive Approach -- O(n):")
	mean_t, min_t, max_t := timeit(time_naive_force, ITERATIONS)
	fmt.printfln("Avg: %0.2fms -- Min: %0.2fms -- Max: %0.2fms", mean_t, min_t, max_t)

	time_naive_force_soa :: proc() {
		naive_force_soa(g_particles_soa)
	}
	fmt.printfln("\nNaive Approach (SOA) -- O(n):")
	mean_t, min_t, max_t = timeit(time_naive_force_soa, ITERATIONS)
	fmt.printfln("Avg: %0.2fms -- Min: %0.2fms -- Max: %0.2fms", mean_t, min_t, max_t)

	time_naive_force_threaded :: proc() {
		naive_force_threaded(g_particles_aos)
	}
	fmt.printfln("\nNaive Approach (threaded) -- O(n):")
	mean_t, min_t, max_t = timeit(time_naive_force_threaded, ITERATIONS)
	fmt.printfln("Avg: %0.2fms -- Min: %0.2fms -- Max: %0.2fms", mean_t, min_t, max_t)

	time_naive_force_soa_threaded :: proc() {
		naive_force_soa_threaded(g_particles_soa)
	}
	fmt.printfln("\nNaive Approach (SOA, threaded) -- O(n):")
	mean_t, min_t, max_t = timeit(time_naive_force_soa_threaded, ITERATIONS)
	fmt.printfln("Avg: %0.2fms -- Min: %0.2fms -- Max: %0.2fms", mean_t, min_t, max_t)

	time_naive_force_soa_simd :: proc() {
		naive_force_soa_simd(g_particles_soa)
	}
	fmt.printfln("\nNaive Approach (SOA, SIMD) -- O(n):")
	mean_t, min_t, max_t = timeit(time_naive_force_soa_simd, ITERATIONS)
	fmt.printfln("Avg: %0.2fms -- Min: %0.2fms -- Max: %0.2fms", mean_t, min_t, max_t)

	time_naive_force_soa_simd_threaded :: proc() {
		naive_force_soa_simd_threaded(g_particles_soa)
	}
	fmt.printfln("\nNaive Approach (SOA, SIMD, threaded) -- O(n):")
	mean_t, min_t, max_t = timeit(time_naive_force_soa_simd_threaded, ITERATIONS)
	fmt.printfln("Avg: %0.2fms -- Min: %0.2fms -- Max: %0.2fms", mean_t, min_t, max_t)

	time_barnes_hut_force :: proc() {
		t: Octree
		octree_init(&t, &g_particles_soa, {0, 0, 0}, 500)
		defer octree_destroy(&t)
		bh_simulate(&t, {0, 0, 0}, 500, 0.2)
	}
	fmt.printfln("\nBarnes Hut:")
	mean_t, min_t, max_t = timeit(time_barnes_hut_force, ITERATIONS)
	fmt.printfln("Avg: %0.2fms -- Min: %0.2fms -- Max: %0.2fms", mean_t, min_t, max_t)

	fmt.println("\n--- Benchmarking Update Implementations ---")

	time_update_particles :: proc() {
		update_particles(g_particles_aos, 0.01)
	}
	fmt.printfln("Naive Update -- O(n):")
	mean_t, min_t, max_t = timeit(time_update_particles, ITERATIONS)
	fmt.printfln("Avg: %0.2fms -- Min: %0.2fms -- Max: %0.2fms", mean_t, min_t, max_t)

	time_update_particles_soa :: proc() {
		update_particles_soa(g_particles_soa, 0.01)
	}
	fmt.printfln("\nNaive SOA Update -- O(n):")
	mean_t, min_t, max_t = timeit(time_update_particles_soa, ITERATIONS)
	fmt.printfln("Avg: %0.2fms -- Min: %0.2fms -- Max: %0.2fms", mean_t, min_t, max_t)

	time_update_particles_soa_threaded :: proc() {
		update_particles_soa_threaded(g_particles_soa, 0.01)
	}
	fmt.printfln("\nThreaded SOA Update -- O(n):")
	mean_t, min_t, max_t = timeit(time_update_particles_soa_threaded, ITERATIONS)
	fmt.printfln("Avg: %0.2fms -- Min: %0.2fms -- Max: %0.2fms", mean_t, min_t, max_t)

	time_update_particles_soa_simd :: proc() {
		update_particles_soa_simd(g_particles_soa, 0.01)
	}
	fmt.printfln("\nSIMD SOA Update -- O(n):")
	mean_t, min_t, max_t = timeit(time_update_particles_soa_simd, ITERATIONS)
	fmt.printfln("Avg: %0.2fms -- Min: %0.2fms -- Max: %0.2fms", mean_t, min_t, max_t)

	time_update_particles_soa_simd_threaded :: proc() {
		update_particles_soa_simd_threaded(g_particles_soa, 0.01)
	}
	fmt.printfln("\nThreaded SIMD SOA Update -- O(n):")
	mean_t, min_t, max_t = timeit(time_update_particles_soa_simd_threaded, ITERATIONS)
	fmt.printfln("Avg: %0.2fms -- Min: %0.2fms -- Max: %0.2fms", mean_t, min_t, max_t)
}

find_limits :: proc() {
	TARGET_MS :: 16.666
	LIMIT_MAX_N :: 100_000
	g_particles_aos = make([]Particle, LIMIT_MAX_N)
	g_particles_soa = make(#soa[]Particle, LIMIT_MAX_N)
	init_both(g_particles_aos, g_particles_soa)
	defer delete(g_particles_aos)
	defer delete(g_particles_soa)

	fmt.println("\n--- Finding Real-time Limits (60Hz / 16.6ms budget) ---")

	step :: 100

	{
		n := 100
		for n <= LIMIT_MAX_N {
			avg: f64 = 0
			for i in 0 ..< 100 {
				start := time.now()
				naive_force(g_particles_aos[:n])
				avg += time.duration_milliseconds(time.since(start))
			}
			if avg / 100 > TARGET_MS do break
			n += step
		}
		fmt.printfln("Naive: ~%d bodies", n)
	}

	{
		n := 100
		for n <= LIMIT_MAX_N {
			avg: f64 = 0
			for i in 0 ..< 100 {
				start := time.now()
				naive_force_soa(g_particles_soa[:n])
				avg += time.duration_milliseconds(time.since(start))
			}
			if avg / 100 > TARGET_MS do break
			n += step
		}
		fmt.printfln("Naive SOA Limit: ~%d bodies", n)
	}

	{
		n := 100
		for n <= LIMIT_MAX_N {
			avg: f64 = 0
			for i in 0 ..< 100 {
				start := time.now()
				naive_force_threaded(g_particles_aos[:n])
				avg += time.duration_milliseconds(time.since(start))
			}
			if avg / 100 > TARGET_MS do break
			n += step
		}
		fmt.printfln("Naive Threaded Limit: ~%d bodies", n)
	}

	{
		n := 100
		for n <= LIMIT_MAX_N {
			avg: f64 = 0
			for i in 0 ..< 100 {
				start := time.now()
				naive_force_soa_threaded(g_particles_soa[:n])
				avg += time.duration_milliseconds(time.since(start))
			}
			if avg / 100 > TARGET_MS do break
			n += step
		}
		fmt.printfln("Naive SOA Threaded Limit: ~%d bodies", n)
	}

	{
		n := 100
		for n <= LIMIT_MAX_N {
			avg: f64 = 0
			for i in 0 ..< 100 {
				start := time.now()
				naive_force_soa_simd(g_particles_soa[:n])
				avg += time.duration_milliseconds(time.since(start))
			}
			if avg / 100 > TARGET_MS do break
			n += step
		}
		fmt.printfln("Naive SIMD Limit: ~%d bodies", n)
	}

	{
		n := 100
		for n <= LIMIT_MAX_N {
			avg: f64 = 0
			for i in 0 ..< 100 {
				start := time.now()
				naive_force_soa_simd_threaded(g_particles_soa[:n])
				avg += time.duration_milliseconds(time.since(start))
			}
			if avg / 100 > TARGET_MS do break
			n += step
		}
		fmt.printfln("Naive Threaded SIMD Limit: ~%d bodies", n)
	}

	{
		n := 100
		for n <= LIMIT_MAX_N {
			subset := g_particles_soa[:n]
			t: Octree
			octree_init(&t, &subset, {0, 0, 0}, 500)
			bh_simulate(&t, {0, 0, 0}, 500, 0.2)
			octree_reset(&t, {0, 0, 0}, 500)
			avg: f64 = 0
			for i in 0 ..< 100 {
				start := time.now()
				bh_simulate(&t, {0, 0, 0}, 500, 0.2)
				avg += time.duration_milliseconds(time.since(start))
				octree_reset(&t, {0, 0, 0}, 500)
			}
			octree_destroy(&t)
			if avg / 100 > TARGET_MS do break
			n += step
		}
		fmt.printfln("Barnes-Hut Limit: ~%d bodies", n)
	}
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
