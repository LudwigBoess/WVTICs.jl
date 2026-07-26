# Allocates particles, selects the problem, and computes the particle mass
# (ports setup.c::Setup, setup_problem, mpart_from_integral).
#
# `setup`:
#   1. allocate particles (Npart)             -> `Particles(Npart)`
#   2. select the problem from the registry   -> `setup_problem(param)` (problems.jl)
#   3. compute Mpart                          -> 512^3 midpoint mass integral

"""
    setup(param::Parameters) -> (Particles, ProblemParameters)

Allocate the SoA [`Particles`](@ref) container, select the problem from
`(Problem_Flag, Problem_Subflag)` via the problem registry (`problems.jl`), fill
[`ProblemParameters`](@ref) (name, boxsize, periodicity, `Rho_Max`), and compute
`Mpart` by 512³ midpoint integration of the density model
(`mpart_from_integral`).

Asserts that `Boxsize[1]` is the largest axis (required for neighbour finding).
"""
function setup(param::Parameters)
    particles = Particles(param.Npart)

    prob = setup_problem(param)

    problem = ProblemParameters(
        Name = prob.name,
        Mpart = 1.0,                 # renormalised below
        Boxsize = prob.boxsize,
        Rho_Max = prob.rho_max,
        Periodic = prob.periodic,
    )

    # require the first axis to be the largest (for neighbour finding)
    (problem.Boxsize[1] >= problem.Boxsize[2] &&
     problem.Boxsize[1] >= problem.Boxsize[3]) ||
        error("Boxsize[0] has to be largest for ngb finding to work.")

    problem.Mpart = mpart_from_integral(particles, param, problem, prob)

    return particles, problem
end

# Function-barrier inner kernel for one outer-i slice of the midpoint
# integral. Parameterised on the density function type `F` so the hot triple
# loop is fully specialised (type-stable, allocation-free, no dynamic
# dispatch). Returns the (un-scaled) density sum over the i-th x-slab.
function _mpart_slice(dfun::F, s::Particles, i::Int, N::Int,
                      dx::Float64, dy::Float64, dz::Float64)::Float64 where {F}
    x = (i + 0.5) * dx
    acc = 0.0
    @inbounds for j in 0:(N - 1)
        y = (j + 0.5) * dy
        for k in 0:(N - 1)
            z = (k + 0.5) * dz
            s.pos[1] = SVector{3,Float64}(x, y, z)
            acc += dfun(s, 1, 0.0)
        end
    end
    return acc
end

# Per-chunk function barrier for `mpart_from_integral`'s outer parallel loop.
# Concrete, type-annotated args only (no boxed capture): each chunk owns
# `scratch[c]` (race-free probe) and writes only `partial[c]`.
@noinline function _mpart_chunk!(c::Int, chunks::Vector{UnitRange{Int}},
                                 scratch::Vector{Particles},
                                 partial::Vector{Float64}, dfun::F, N::Int,
                                 dx::Float64, dy::Float64,
                                 dz::Float64) where {F}
    s = scratch[c]
    acc = 0.0
    # chunks index 1:N; the grid index i is 0-based: i = idx - 1.
    @inbounds for idx in chunks[c]
        acc += _mpart_slice(dfun, s, idx - 1, N, dx, dy, dz)
    end
    @inbounds partial[c] = acc
    return nothing
end

"""
    mpart_from_integral(particles, param, problem, prob) -> Float64

Integrate the problem density model on an `N=1<<9` (512) cells-per-side midpoint
grid (3D) and return `M_tot / Npart`. The density callback is evaluated with
`density_function_correction = 0.0` so the mass integral is conserved.

The density function reads a particle's position, so a scratch `Particles(1)`
probe is moved over the grid and `prob.density(probe, 1, 0.0)` evaluated.

Parallelised by partitioning the outer (`i`) loop into contiguous index ranges.
One scratch probe and one partial-sum accumulator are preallocated *per chunk*
and indexed by the loop's chunk variable `c` (never `Threads.threadid()`, which
on Julia ≥1.12 with the `:dynamic` scheduler is not bounded by `nthreads()`, so
a `threadid()`-indexed buffer can go out of bounds). Each task owns chunk `c`
and writes only `scratch[c]` / `partial[c]`, so the probe write is race-free.
The midpoint sum is reproducible: per-`i` slices accumulate in ascending `i`
order within each chunk and the chunk partials are summed in chunk order.
"""
function mpart_from_integral(particles::Particles, param::Parameters,
                             problem::ProblemParameters, prob::Problem)
    N = 1 << 9                       # 512 cells per side
    bx = problem.Boxsize[1]
    by = problem.Boxsize[2]
    bz = problem.Boxsize[3]
    dx = bx / N
    dy = by / N
    dz = bz / N
    cellvol = dx * dy * dz

    # Partition the outer (1-based) index range 1:N into contiguous chunks.
    # One scratch probe + one partial accumulator PER CHUNK, indexed by the
    # chunk variable `c` only. Using Threads.threadid() here is an unsafe
    # anti-pattern: on Julia ≥1.12 with the :dynamic scheduler threadid() is
    # NOT bounded by nthreads(), so scratch[threadid()] is out of bounds.
    nchunks = max(1, Threads.nthreads())
    chunks = _chunk_ranges(N, nchunks)
    nc = length(chunks)
    scratch = [Particles(1) for _ in 1:nc]   # per-chunk race-free probe
    partial = zeros(Float64, nc)             # per-chunk summed contribution

    # `prob.density` is a `::Function` field (dynamic dispatch). Pass it
    # through a function barrier so the ~1.3e8-call triple loop specialises
    # on its concrete type (no per-call dynamic dispatch / allocation).
    dfun = prob.density

    # Parallel driver (see src/parallel/threads.jl). The `do c` body is a pure
    # function-barrier forward to `_mpart_chunk!` — no boxed capture; results are
    # race-free per chunk.
    _run_chunks(nc) do c
        _mpart_chunk!(c, chunks, scratch, partial, dfun, N, dx, dy, dz)
    end

    tot_mass = 0.0
    for c in 1:nc
        tot_mass += partial[c]
    end
    tot_mass *= cellvol

    mpart = tot_mass / param.Npart

    mpart > 0 ||
        error("Particle mass has to be finite, have $mpart")

    return mpart
end
