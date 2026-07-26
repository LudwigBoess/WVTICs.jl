module WVTICs

# ---------------------------------------------------------------------------
# WVTICs — Weighted Voronoi Tessellation initial conditions.
# Julia port of the C code in /e/ocean2/users/lboess/WVTICs/src.
# See CLAUDE.md (alongside the C source) for the authoritative plan.
#
# Phase 0 scaffolding: types, parameter-file parser, SoA particle container,
# and a `make_sph_wvtics(parfile)` driver reproducing the `main.c` call
# sequence with stubs for the not-yet-ported stages.
# ---------------------------------------------------------------------------

using StaticArrays
using NearestNeighbors
using SPHKernels
using GadgetIO
using FFTW
using LinearAlgebra
using Printf
using Random
using Distributed
using Logging: with_logger, NullLogger
using PrecompileTools: @setup_workload, @compile_workload

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

# Package version string, read once at module load from `Project.toml`
# (single source of truth — no second copy to keep in sync with the manifest).
# `Project.toml` sits one directory above this file (`src/WVTICs.jl`).
const VERSION_STR = TOML.parsefile(joinpath(@__DIR__, "..", "Project.toml"))["version"]

"""
    make_sph_wvtics(parfile::AbstractString;
                    kernel::KernelConfig = default_kernel_config(),
                    verbose::Bool = true) -> Particles

Top-level driver and the **only** function this package exports. Reproduces
the `main.c` call sequence:

```
read_param_file -> setup -> make_positions! -> make_ids!
  -> regularise_sph_particles! -> make_velocities! -> make_temperatures!
  -> make_magnetic_fields! -> make_post_processing! -> write_output
```

The SPH `kernel` (and its dimension / `DESNNGB` / `NNGBDEV` / `NGBMAX`) is
chosen at the call site via a [`KernelConfig`](@ref) and threaded through the
relaxation; it defaults to
[`default_kernel_config`](@ref) (Wendland C4, 3D — the active C Makefile
default). Build a different one from a built-in kernel with
`KernelConfig(WendlandC6; dim = 3)`, or from **any** `SPHKernels.jl` kernel
instance with `KernelConfig(SPHKernels.WendlandC6(Float64, 3); desnngb = 295)`
— so kernels beyond the built-in set work without changing WVTICs.

If the parameter file sets `DesNumNgb > 0` it overrides the kernel's target
neighbour count (`DESNNGB`) for this run.

Returns the [`Particles`](@ref) container.
"""
function make_sph_wvtics(parfile::AbstractString;
                         kernel::KernelConfig = default_kernel_config(),
                         verbose::Bool = true)
    verbose && println("--- This is WVT ICs (Julia port), Version $(VERSION_STR) ---")

    param = read_param_file(parfile)

    # Parameter-file `DesNumNgb` (target neighbour count, `DESNNGB`) overrides
    # the selected kernel's default when > 0 — the runtime tuning knob for the
    # neighbour count of whatever `AbstractSPHKernel` was passed.
    if param.DesNumNgb > 0
        kernel = with_desnngb(kernel, param.DesNumNgb)
    end

    particles, problem = setup(param)

    make_positions!(particles, param, problem)
    make_ids!(particles, param)

    regularise_sph_particles!(particles, param, problem, setup_problem(param),
                              kernel; verbose = verbose)

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

# Public API: only the top-level driver is exported. Every other type and
# function (Parameters, KernelConfig, the kernel types, setup, the make_*
# stages, the distributed helpers, …) remains internal and reachable only
# via explicit qualification (`WVTICs.KernelConfig`) or
# `using WVTICs: name` — it is not brought into scope by `using WVTICs`.
export make_sph_wvtics

# --- Precompilation --------------------------------------------------------
# Compile the hot path at precompile time so the first real call is fast. The
# workload uses the constant-density problem (total mass exactly 1, so Mpart =
# 1/Npart directly — the 512³ setup grid is skipped). The relaxation call tree
# specialises per kernel type (KernelConfig{K} carries a concrete kernel), so
# both the driver default (Wendland C4) and the cheap CubicSpline are compiled
# through a full relaxation; the remaining kernels get their leaf value/deriv/
# bias evaluations forced in 2D and 3D. The TOML parser and snapshot writer are
# compiled into a temp dir so precompilation has no side effects.
@setup_workload begin
    @compile_workload begin
      # Silence logging: the tiny-N relaxation legitimately `@warn`s and
      # GadgetIO `@info`s per block; neither should surface during precompile.
      with_logger(NullLogger()) do
        # One full relaxation for a given kernel at particle count `np`.
        function _pc_relax(np::Int, kc::KernelConfig)
            param = Parameters()
            param.Npart = np
            param.Maxiter = 1
            param.MpsFraction = 5.0
            param.StepReduction = 0.95
            param.LimitMps = (-1.0, -1.0, -1.0, -1.0)
            param.MoveFractionMin = 0.01
            param.MoveFractionMax = 0.01
            param.ProbesFraction = 0.1
            param.RedistributionFrequency = 5
            param.LastMoveStep = 256
            prob = setup_problem(param)
            problem = ProblemParameters(; Name = prob.name, Mpart = 1.0 / np,
                                          Boxsize = prob.boxsize,
                                          Rho_Max = prob.rho_max,
                                          Periodic = prob.periodic)
            particles = Particles(np)
            make_positions!(particles, param, problem)
            make_ids!(particles, param)
            regularise_sph_particles!(particles, param, problem, prob, kc;
                                      verbose = false, output_diagnostics = false)
            return particles, param, problem
        end

        # Driver default (WC4, DESNNGB 200 → needs N > 200) and cheap Cubic.
        particles, param, problem = _pc_relax(343, KernelConfig(WendlandC4; dim = 3))
        _pc_relax(64, KernelConfig(CubicSpline; dim = 3))

        # Leaf kernel evaluations for every supported kernel × dimension.
        for kt in (CubicSpline, WendlandC2, WendlandC4, WendlandC6, WendlandC8)
            for d in (2, 3)
                k = KernelConfig(kt; dim = d)
                sph_kernel(k, 0.3, 1.0)
                sph_kernel_deriv(k, 0.3, 1.0)
                sph_bias_correction(k, 1.0, 1.0 / 343, 1.0)
            end
        end

        # Parameter parser + snapshot writer (temp dir → no side effects).
        mktempdir() do dir
            tomlfile = joinpath(@__DIR__, "..", "parameters.toml")
            isfile(tomlfile) && read_param_file(tomlfile)
            write_output(particles, param, problem;
                         verbose = false, output_diagnostics = false,
                         filename = joinpath(dir, "precompile_ic"))
        end
      end
    end
end

end # module WVTICs
