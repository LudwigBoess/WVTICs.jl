# Setup — Julia port of `setup.c::Setup`, `setup_problem` (dispatch) and
# `mpart_from_integral`.
#
# C `Setup()`:
#   1. allocate P / SphP (Npart)              -> `Particles(Npart)`
#   2. setup_problem(Flag, Subflag)           -> `setup_problem(param)` (problems.jl)
#   3. mpart_from_integral()                  -> 512^3 midpoint mass integral
#
# Standard Gadget units (setup.c lines 4-6): ULength = 3.08568025e21 cm,
# UMass = 1.989e43 g, UVel = 1e5 cm/s. Wired via `GadgetUnits.GadgetPhysical`.

"""
    GADGET_UNITS :: GadgetPhysical

Standard Gadget unit system used by WVTICs, exactly the constants hard-coded
in `setup.c`:

- `ULength = 3.08568025e21` cm  (kpc)
- `UMass   = 1.989e43` g        (1e10 Msol)
- `UVel    = 1e5` cm/s          (km/s)

Built with `GadgetUnits.GadgetPhysical(l_unit, m_unit, v_unit)`. Phase 1 only
needs the unit system available (it does not yet convert any quantity); later
phases use it for physical-unit diagnostics.
"""
const GADGET_UNITS = GadgetPhysical(3.08568025e21, 1.989e43, 1.0e5)

"""
    setup(param::Parameters) -> (Particles, ProblemParameters)

Port of `setup.c::Setup`. Allocates the SoA [`Particles`](@ref) container,
selects the problem from `(Problem_Flag, Problem_Subflag)` via the problem
registry (`problems.jl`), fills [`ProblemParameters`](@ref) (name, boxsize,
periodicity, `Rho_Max`), and computes `Mpart` by 512³ midpoint integration of
the density model (`mpart_from_integral`).

The C `Boxsize[0]` (axis 0) = largest-axis invariant is asserted with the
same message ("Boxsize[0] has to be largest for ngb finding to work."); in
the Julia 1-based layout this is `Boxsize[1]`.
"""
function setup(param::Parameters)
    particles = Particles(param.Npart)

    prob = setup_problem(param)

    problem = ProblemParameters(
        Name = prob.name,
        Mpart = 1.0,                 # renormalised below (C sets Problem.Mpart=1)
        Boxsize = prob.boxsize,
        Rho_Max = prob.rho_max,
        Periodic = prob.periodic,
    )

    # C: Assert(Boxsize[0] >= Boxsize[1] && Boxsize[0] >= Boxsize[2], ...)
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
# `scratch[c]` (race-free probe) and writes only `partial[c]`. Identical for
# every threading backend.
@noinline function _mpart_chunk!(c::Int, chunks::Vector{UnitRange{Int}},
                                 scratch::Vector{Particles},
                                 partial::Vector{Float64}, dfun::F, N::Int,
                                 dx::Float64, dy::Float64,
                                 dz::Float64) where {F}
    s = scratch[c]
    acc = 0.0
    # chunks are over 1:N; the C grid index i is 0-based: i = idx - 1.
    @inbounds for idx in chunks[c]
        acc += _mpart_slice(dfun, s, idx - 1, N, dx, dy, dz)
    end
    @inbounds partial[c] = acc
    return nothing
end

"""
    mpart_from_integral(particles, param, problem, prob) -> Float64

Port of `setup.c::mpart_from_integral`. Integrates the problem density model
on a `N=1<<9` (512) cells-per-side midpoint grid (3D) and returns
`M_tot / Npart`. The density callback is called with
`density_function_correction = 0.0` to conserve the mass integral
(CLAUDE.md §1.2; C passes `0.0` to `Density_Func_Ptr`).

The C code abuses `P[0].Pos` as the integration probe (the density function
reads `P[ipart].Pos`). We replicate that exactly: a single scratch particle
(`particles.pos[1]`) is moved over the grid and `prob.density(particles, 1,
0.0)` evaluated. **Caveat:** this writes `particles.pos[1]`; positions are
(re)assigned afterwards by `make_positions!`, exactly as in C where
`Make_Positions` overwrites `P[0].Pos`. The `particles.pos[1]` slot is
restored to its prior value on return for safety.

Parallelised by partitioning the outer (`i`) loop into `nchunks` contiguous
index ranges (CLAUDE.md §4b: the 512³ integration is serial in C and trivially
parallel). One scratch `Particles` probe **and** one partial-sum accumulator
are preallocated *per chunk* and indexed by the loop's chunk variable `c`
(never `Threads.threadid()`, which on Julia ≥1.12 with the `:dynamic`
scheduler is not bounded by `nthreads()`). Each task owns chunk `c` and writes
only `scratch[c]` / `partial[c]`, so the probe write is race-free. The
midpoint sum is reproducible (per-`i` slices are accumulated in ascending `i`
order within each chunk and the chunk partials are summed in chunk order).
"""
function mpart_from_integral(particles::Particles, param::Parameters,
                             problem::ProblemParameters, prob::Problem)
    N = 1 << 9                       # 512, as C: const int N = 1ULL << 9;
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

    # Swappable parallel driver (see src/parallel/threads.jl). The `do c`
    # body is a pure function-barrier forward to `_mpart_chunk!` — no boxed
    # capture; results are race-free per chunk. Backend-independent semantics.
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
