module WVTICs

# ---------------------------------------------------------------------------
# WVTICs — Weighted Voronoi Tessellation initial conditions.
# Julia port of the C code in /e/ocean2/users/lboess/WVTICs/src.
# See CLAUDE.md (alongside the C source) for the authoritative plan.
#
# Phase 0 scaffolding: types, parameter-file parser, SoA particle container,
# and a no-op `main(parfile)` driver reproducing the `main.c` call sequence
# with stubs for the not-yet-ported stages.
# ---------------------------------------------------------------------------

using StaticArrays
using NearestNeighbors
using SPHKernels
using GadgetIO
using GadgetUnits
using FFTW
using LinearAlgebra
using Random
using Distributed

# --- Types -----------------------------------------------------------------
include("types/parameters.jl")
include("types/particles.jl")

# --- IO --------------------------------------------------------------------
include("io/parameter_file.jl")
include("io/snapshot.jl")

# --- Problems --------------------------------------------------------------
# Included before setup/positions because `setup.jl` annotates the problem
# struct (`::Problem`) in `mpart_from_integral`'s signature, which must be
# defined at that file's include time.
include("problems/problems.jl")

# --- Parallel backend ------------------------------------------------------
# The swappable threaded driver (`_run_chunks` / `THREAD_BACKEND`) is included
# BEFORE the hot-loop files (setup/sph/wvt) that call it, so the primitive and
# its `Polyester`/`OhMyThreads` imports are in scope at their include time.
# (`distributed.jl` is still a Phase-D stub and stays at the end.)
include("parallel/threads.jl")

# --- Setup / positions -----------------------------------------------------
include("setup/setup.jl")
include("setup/positions.jl")

# --- SPH -------------------------------------------------------------------
include("sph/kernels.jl")
include("sph/neighbors.jl")
include("sph/density.jl")

# --- WVT loop --------------------------------------------------------------
# diagnostics + redistribution before relax: relax.jl uses `Quadruplet`,
# `calculate_stats_on`, the diagnostics writers, and the redistribution
# entry points (Phase 3 wiring).
include("wvt/redistribution.jl")
include("wvt/diagnostics.jl")
include("wvt/relax.jl")

# --- Fields ----------------------------------------------------------------
include("fields/turbulent_B.jl")

# --- Parallel backends -----------------------------------------------------
# (threads.jl is included earlier, before the hot-loop files that use it.)
include("parallel/distributed.jl")

# --- Driver ----------------------------------------------------------------

"""
    main(parfile::AbstractString; verbose::Bool = true) -> Particles

Top-level driver. Reproduces the `main.c` call sequence:

```
read_param_file -> setup -> make_positions! -> make_ids!
  -> regularise_sph_particles! -> make_velocities! -> make_temperatures!
  -> make_magnetic_fields! -> make_post_processing! -> write_output
```

Phase 0: every stage past `read_param_file` is a stub (see the respective
files). Running it parses the parameter file and walks the full pipeline
shape without error, returning the (currently empty) [`Particles`](@ref)
container.
"""
function main(parfile::AbstractString; verbose::Bool = true)
    verbose && println("--- This is WVT ICs (Julia port), Version $(VERSION_STR) ---")

    param = read_param_file(parfile)

    particles, problem = setup(param)

    make_positions!(particles, param, problem)
    make_ids!(particles, param)

    regularise_sph_particles!(particles, param, problem)

    make_velocities!(particles, param, problem)
    make_temperatures!(particles, param, problem)
    make_magnetic_fields!(particles, param, problem)
    make_post_processing!(particles, param, problem)

    # `calculate_density_function_correction` (C `Calculate_Bias()` /
    # `bias_correction.c`, the artificial density-model correction diagnostic)
    # is a diagnostic print only — Phase 1+ (not yet ported / stubbed).

    write_output(particles, param, problem; verbose = verbose)

    verbose && println("done")
    return particles
end

const VERSION_STR = "0.1.0"

export Parameters, ProblemParameters, KernelConfig, KernelType,
       CubicSpline, WendlandC2, WendlandC4, WendlandC6, WendlandC8,
       WendlandC10, WendlandC12,
       default_kernel_config,
       Particles,
       read_param_file,
       read_param_toml,
       regularise_sph_particles!,
       regularise_sph_particles_distributed!,
       init_workers, plan_workers, detect_scheduler, scheduler_pool_size,
       decompose_domain, Decomposition, peano_keys, select_ghosts,
       distributed_iteration_reductions, write_output_distributed,
       setup, setup_problem, Problem, PROBLEM_REGISTRY,
       make_positions!, make_ids!,
       make_velocities!, make_temperatures!,
       make_magnetic_fields!, make_post_processing!,
       make_turbulent_Bfield, make_turbulent_postprocess,
       write_output,
       main

end # module WVTICs
