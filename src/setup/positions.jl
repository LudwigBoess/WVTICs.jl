# Positions + IDs + per-problem appliers — Julia port of `positions.c` and
# `ids.c`.
#
# `make_positions!`  -> `positions.c::Make_Positions`
#     uniform random sampling and von-Neumann rejection sampling.
#     Rejection sampling is the DEFAULT (active C Makefile uses
#     -DREJECTION_SAMPLING; CLAUDE.md §1.3).
# `make_ids!`        -> `ids.c::Make_IDs`           (spaced IDs)
# `make_velocities!` / `make_temperatures!` / `make_magnetic_fields!` /
# `make_post_processing!` -> `positions.c::Make_{Velocities,Temperatures,
#     Magnetic_Fields,PostProcessing}` — implemented here in Phase 1 because
#     they are trivial given the problem callbacks (they are needed for a
#     meaningful round-trip / KH problem) and are pure functions of position.
#
# RNG note (CLAUDE.md §"Mapping C constructs", §5): the C code uses
# `erand48(Omp.Seed)` with the per-thread seed `14041981*(threadid+1)`
# (`main.c`). Julia uses per-task `Random.Xoshiro` RNGs seeded
# `RNG_BASE_SEED * (chunk_index)`. Output is therefore **statistically
# equivalent but NOT bit-identical** to the C stream — by design and
# documented.

"""
    RNG_BASE_SEED

Base seed for the per-task position-sampling RNGs. Mirrors the C constant
`14041981` (`main.c` `Omp.Seed = 14041981*(threadid+1)`). The Julia stream is
*not* bit-equivalent to C `erand48` (different generator); see file header.
"""
const RNG_BASE_SEED = 14041981

"""
    PositionSampling

Sampling mode for [`make_positions!`](@ref) (the C compile flags
`REJECTION_SAMPLING` / `PEANO_SAMPLING` / none):

- `RejectionSampling` — von Neumann rejection sampling. **Default**, matches
  the active C Makefile.
- `UniformSampling`   — plain uniform random positions (C default build, no
  flag).
"""
@enum PositionSampling RejectionSampling UniformSampling

"""
    make_positions!(particles, param, problem, prob;
                    sampling = RejectionSampling, seed = RNG_BASE_SEED)

Port of `positions.c::Make_Positions`. Fills `particles.pos[i]` for
`i = 1:Npart` and sets `particles.type[i] = 0` (all gas, as in C).

- `RejectionSampling` (default): von Neumann — draw a uniform position in the
  box; accept iff `Rho_Max * rand() < ρ(pos)` (C: loop while `rho >= rho_r`
  with `rho = Rho_Max*erand48`,
  `rho_r = Density_Func_Ptr(ipart, density_function_correction)`).
  Uses `param.density_function_correction` (C `Param.BiasCorrection`) as the
  density-callback correction argument exactly like C.
- `UniformSampling`: `pos = rand() .* Boxsize` per axis.

Parallel sampling uses one `Xoshiro` RNG per chunk seeded reproducibly
(`seed * chunk`), so a run is deterministic for a fixed thread count but
**not** bit-identical to the C `erand48` stream (documented; CLAUDE.md §5).
2D is not handled in Phase 1 (the two ported problems are 3D); a `dim`-aware
path is a later-phase concern.
"""
function make_positions!(particles::Particles, param::Parameters,
                         problem::ProblemParameters, prob::Problem;
                         sampling::PositionSampling = RejectionSampling,
                         seed::Integer = RNG_BASE_SEED)
    n = param.Npart
    n == 0 && return nothing

    bx = problem.Boxsize[1]
    by = problem.Boxsize[2]
    bz = problem.Boxsize[3]
    rho_max = problem.Rho_Max
    density_function_correction = param.density_function_correction

    # Split the particle range into chunks (one RNG / scratch probe per
    # chunk) so the density callback's pos read is race-free and the run is
    # reproducible for a fixed chunk count.
    nchunks = max(1, Threads.nthreads())
    chunks = _chunk_ranges(n, nchunks)
    nc = length(chunks)

    # Swappable parallel driver (see src/parallel/threads.jl). The `do c`
    # body is a pure function-barrier forward to `_positions_chunk!` — no
    # boxed capture. Per-chunk `Xoshiro(seed*c)` keeps the run deterministic
    # for a fixed chunk count (unchanged by the backend; identical results).
    dfun = prob.density
    _run_chunks(nc) do c
        _positions_chunk!(c, chunks, particles, dfun, sampling, seed,
                          bx, by, bz, rho_max, density_function_correction)
    end

    return nothing
end

# Per-chunk function barrier for `make_positions!`. Concrete, type-annotated
# args only (no boxed capture); `dfun::F` is the `prob.density` ::Function
# field specialised behind this barrier. One `Xoshiro(seed*c)` per chunk
# (NEVER threadid-indexed) so a fixed chunk count is reproducible — and
# identical across all threading backends (each backend runs `c in 1:nc`
# exactly once over disjoint particle ranges).
@noinline function _positions_chunk!(c::Int, chunks::Vector{UnitRange{Int}},
                                     particles::Particles, dfun::F,
                                     sampling::PositionSampling,
                                     seed::Integer, bx::Float64, by::Float64,
                                     bz::Float64, rho_max::Float64,
                                     density_function_correction) where {F}
    rng = Random.Xoshiro(seed * c)
    rng_lo, rng_hi = first(chunks[c]), last(chunks[c])
    @inbounds for ipart in rng_lo:rng_hi
        if sampling === RejectionSampling
            # C: rho = 0; rho_r = 0; while (rho >= rho_r) { draw; ... }
            # First iteration always runs (0 >= 0). Accept when
            # rho < rho_r  <=>  Rho_Max*rand < ρ(pos).
            while true
                px = rand(rng) * bx
                py = rand(rng) * by
                pz = rand(rng) * bz
                particles.pos[ipart] = SVector{3,Float64}(px, py, pz)
                rho = rho_max * rand(rng)
                rho_r = dfun(particles, ipart, density_function_correction)
                if rho < rho_r
                    break
                end
            end
        else # UniformSampling
            px = rand(rng) * bx
            py = rand(rng) * by
            pz = rand(rng) * bz
            particles.pos[ipart] = SVector{3,Float64}(px, py, pz)
        end
        particles.type[ipart] = Int32(0)
    end
    return nothing
end

# Backwards/looser-arity entry point: the `make_sph_wvtics` driver calls
# `make_positions!(particles, param, problem)`. Resolve the problem callbacks
# from the registry so the driver does not need to thread `prob` through.
function make_positions!(particles::Particles, param::Parameters,
                         problem::ProblemParameters)
    prob = setup_problem(param)
    return make_positions!(particles, param, problem, prob)
end

"""
    _chunk_ranges(n, k) -> Vector{UnitRange{Int}}

Split `1:n` into at most `k` contiguous, near-equal ranges (empty ranges
dropped). Helper for deterministic per-chunk RNG seeding.
"""
function _chunk_ranges(n::Integer, k::Integer)
    n <= 0 && return UnitRange{Int}[]
    k = max(1, min(k, n))
    base = div(n, k)
    rem = mod(n, k)
    ranges = UnitRange{Int}[]
    start = 1
    for c in 1:k
        len = base + (c <= rem ? 1 : 0)
        len == 0 && continue
        push!(ranges, start:(start + len - 1))
        start += len
    end
    return ranges
end

"""
    make_ids!(particles, param)

Port of `ids.c::Make_IDs`. Assigns spaced particle IDs so an ID-based domain
decomposition is balanced. Algorithm (verbatim from `ids.c`):

1. `delta` = smallest integer `> 127` that divides `Npart` (search
   `delta = 128, 129, …`); if none `≤ Npart`, `delta = 1`.
2. If `delta > 1`: `id = 1 - delta`, `start = 1`; for each particle
   `id += delta`; if `id > Npart` then `start += 1; id = start`; assign `id`.
3. If `delta == 1`: IDs are left as-is (the C dead loop
   `for(ipart=Npart; ipart<Npart; …)` never executes, so with `delta==1`
   the IDs stay at their zero-initialised value — preserved here).

IDs are `UInt32` (`particles.id`).
"""
function make_ids!(particles::Particles, param::Parameters)
    npart = param.Npart
    npart == 0 && return nothing

    # find delta: smallest divisor of Npart strictly greater than 127
    delta = 127
    while true
        delta += 1
        if (npart % delta) == 0 || delta > npart
            break
        end
    end

    if delta > npart
        delta = 1
    end

    if delta > 1
        id = 1 - delta
        start = 1
        for ipart in 1:npart
            id += delta
            if id > npart
                start += 1
                id = start
            end
            particles.id[ipart] = UInt32(id)
        end
    end
    # delta == 1: C leaves IDs untouched (the only assignment loop is the C
    # dead `for(ipart=Npart; ipart<Npart; …)`), so IDs remain zero here too.

    return nothing
end

# --- per-problem appliers (positions.c::Make_{Velocities,...}) -------------
#
# Implemented in Phase 1 (not deferred): they are pure functions of the
# already-sampled positions and are needed for a meaningful Kelvin-Helmholtz
# round-trip. The C versions just loop and call the problem function pointers.

"""    make_velocities!(particles, param, problem)

Port of `positions.c::Make_Velocities`. Applies the problem velocity callback
to every particle, writing `particles.vel` (stored `Float32`, matching the C
`P.Vel` write through a `float` buffer).
"""
function make_velocities!(particles::Particles, param::Parameters,
                          problem::ProblemParameters)
    prob = setup_problem(param)
    _apply_velocities!(particles, prob.velocity, param.Npart)
    return nothing
end

@noinline function _apply_velocities!(particles::Particles, vfun::F,
                                      n::Int) where {F}
    @inbounds for ipart in 1:n
        v = vfun(particles, ipart)
        particles.vel[ipart] = SVector{3,Float32}(v[1], v[2], v[3])
    end
    return nothing
end

"""    make_temperatures!(particles, param, problem)

Port of `positions.c::Make_Temperatures`. Applies the problem internal-energy
callback, writing `particles.u` (`Float32`, as C `SphP.U`).
"""
function make_temperatures!(particles::Particles, param::Parameters,
                            problem::ProblemParameters)
    prob = setup_problem(param)
    _apply_internal_energy!(particles, prob.internal_energy, param.Npart)
    return nothing
end

@noinline function _apply_internal_energy!(particles::Particles, ufun::F,
                                           n::Int) where {F}
    @inbounds for ipart in 1:n
        particles.u[ipart] = Float32(ufun(particles, ipart))
    end
    return nothing
end

"""    make_magnetic_fields!(particles, param, problem)

Port of `positions.c::Make_Magnetic_Fields`. Applies the problem B-field
callback, writing `particles.bfld` (`Float32`, as C `SphP.Bfld`).
"""
function make_magnetic_fields!(particles::Particles, param::Parameters,
                               problem::ProblemParameters)
    prob = setup_problem(param)
    _apply_magnetic_fields!(particles, prob.bfield, param.Npart)
    return nothing
end

@noinline function _apply_magnetic_fields!(particles::Particles, bfun::F,
                                           n::Int) where {F}
    @inbounds for ipart in 1:n
        b = bfun(particles, ipart)
        particles.bfld[ipart] = SVector{3,Float32}(b[1], b[2], b[3])
    end
    return nothing
end

"""    make_post_processing!(particles, param, problem)

Port of `positions.c::Make_PostProcessing`. Runs the optional per-problem
post-processing hook (no-op for the Phase-1 problems).
"""
function make_post_processing!(particles::Particles, param::Parameters,
                               problem::ProblemParameters)
    prob = setup_problem(param)
    prob.postprocess!(particles, param, problem)
    return nothing
end
