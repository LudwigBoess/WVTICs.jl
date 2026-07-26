# Positions, IDs, and per-problem appliers (ports positions.c and ids.c).
#
# `make_positions!`  -> uniform random sampling and von-Neumann rejection
#     sampling; rejection sampling is the DEFAULT.
# `make_ids!`        -> spaced particle IDs.
# `make_velocities!` / `make_temperatures!` / `make_magnetic_fields!` /
# `make_post_processing!` -> apply the problem velocity/internal-energy/B-field/
#     post-processing callbacks (pure functions of the sampled positions).
#
# RNG note: positions use per-task `Random.Xoshiro` RNGs seeded
# `RNG_BASE_SEED * chunk_index`, statistically equivalent to but NOT
# bit-identical to the C `erand48` stream.

"""
    RNG_BASE_SEED

Base seed for the per-task position-sampling RNGs. The Julia stream is *not*
bit-equivalent to the C `erand48` stream (different generator); see file header.
"""
const RNG_BASE_SEED = 14041981

"""
    PositionSampling

Sampling mode for [`make_positions!`](@ref):

- `RejectionSampling` — von Neumann rejection sampling. **Default**.
- `UniformSampling`   — plain uniform random positions.
"""
@enum PositionSampling RejectionSampling UniformSampling

"""
    make_positions!(particles, param, problem, prob;
                    sampling = RejectionSampling, seed = RNG_BASE_SEED)

Fill `particles.pos[i]` for `i = 1:Npart` and set `particles.type[i] = 0` (all
gas).

- `RejectionSampling` (default): von Neumann — draw a uniform position in the
  box; accept iff `Rho_Max * rand() < ρ(pos)`. Uses
  `param.density_function_correction` as the density-callback correction
  argument.
- `UniformSampling`: `pos = rand() .* Boxsize` per axis.

Parallel sampling uses one `Xoshiro` RNG per chunk seeded reproducibly
(`seed * chunk`), so a run is deterministic for a fixed chunk count but **not**
bit-identical to the C `erand48` stream. Only 3D is handled.
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

    # Parallel driver (see src/parallel/threads.jl). The `do c` body is a pure
    # function-barrier forward to `_positions_chunk!` — no boxed capture.
    # Per-chunk `Xoshiro(seed*c)` keeps the run deterministic for a fixed chunk
    # count.
    dfun = prob.density
    _run_chunks(nc) do c
        _positions_chunk!(c, chunks, particles, dfun, sampling, seed,
                          bx, by, bz, rho_max, density_function_correction)
    end

    return nothing
end

# Per-chunk function barrier for `make_positions!`. Concrete, type-annotated
# args only (no boxed capture); `dfun::F` is the `prob.density` ::Function
# field specialised behind this barrier. One `Xoshiro(seed*c)` per chunk (NEVER
# threadid-indexed) so a fixed chunk count is reproducible.
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
            # First iteration always runs. Accept when
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

Assign spaced particle IDs so an ID-based domain decomposition is balanced:

1. `delta` = smallest integer `> 127` that divides `Npart` (search
   `delta = 128, 129, …`); if none `≤ Npart`, `delta = 1`.
2. If `delta > 1`: `id = 1 - delta`, `start = 1`; for each particle
   `id += delta`; if `id > Npart` then `start += 1; id = start`; assign `id`.
3. If `delta == 1`: IDs are left at their zero-initialised value.

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
    # delta == 1: IDs are left untouched, so they remain zero.

    return nothing
end

# --- per-problem appliers -------------------------------------------------
#
# Each loops over particles and applies a problem callback to the already-
# sampled positions.

"""    make_velocities!(particles, param, problem)

Apply the problem velocity callback to every particle, writing `particles.vel`
(stored `Float32`).
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

Apply the problem internal-energy callback, writing `particles.u` (`Float32`).
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

Apply the problem B-field callback, writing `particles.bfld` (`Float32`).
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

Run the optional per-problem post-processing hook.
"""
function make_post_processing!(particles::Particles, param::Parameters,
                               problem::ProblemParameters)
    prob = setup_problem(param)
    prob.postprocess!(particles, param, problem)
    return nothing
end
