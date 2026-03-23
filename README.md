# Odin N-Body Simulation
This repo contains my experiments with implementing N-Body simulations in [Odin](https://odin-lang.org/). The goal is to investigate the algorithms used and the approaches to optimization that are effective for them.

## Usage
To run the demo:
```
odin run .
```
I recommend using the optimization flag `speed`: `odin run . -o:speed`

Also included in the `src/utils.odin` file are some benchmarking functions of varying usefulness. This should definitely be run with the `speed` optimization flag:
```
odin run ./src -o:speed
```

## Implemented Algorithms
### Naive/Brute Force
- [x] Baseline
- [x] SOA (?)
- [x] Multi-threading
- [x] SIMD
- [ ] Compute shader
### Barnes-Hut
- [x] Baseline
- [ ] Particle Reordering
- [ ] Threaded force accumulation
- [ ] Threaded tree building (?)
- [ ] Compute Shader

## Dependencies
The core functionality currently has no third party dependencies, but the basic visualization present in the `main` function relies on having the excellent [Odin implementation by Leonardo Temperanza](https://github.com/LeonardoTemperanza/no_gfx_api/tree/main?tab=readme-ov-file) of a low-level graphics API inspired by the [No Graphics API](https://www.sebastianaaltonen.com/blog/no-graphics-api) blog post by Sebastian Aaltonen available in the `shared` collection. When I get around to implementing GPU compute shader versions of these algorithms, this will likely become a strict dependency.
