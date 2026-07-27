# ===========================================================================
# Distributed-memory backend: run the WVT relaxation across `nworkers()` Julia
# worker processes. Additive to the serial/threaded path, which stays the
# reference: single process (`nworkers() == 1`) the driver delegates straight
# to the serial `regularise_sph_particles!`.
#
# The coordinator holds the authoritative `Particles` and runs the same
# relaxation loop as the serial path, driven through a `DistributedEngine`
# (the `WvtEngine` interface in wvt/relax.jl). Each iteration:
#
#  * Domain decomposition. Peano–Hilbert keys (GadgetIO's
#    `peano_hilbert_key`/`get_int_pos`) sort the particles; the sorted array
#    is split into per-worker contiguous, particle-count-balanced ranges by a
#    recursive load-balanced bisection (`decompose_domain`/`_find_split!`).
#    The split keys off the worker count only, so it is node-count agnostic.
#
#  * Ghost halos. Each worker's owned particles define an AABB; it imports
#    every particle within a halo width of that box (`select_ghosts`,
#    min-image / periodic-face aware). The width grows until it covers
#    2·(max solved hsml), so each owned particle's neighbour set is complete.
#
#  * Density solve + displacement on the workers via `remotecall`, reusing the
#    serial kernels (`find_sph_quantities!`, `_wvt_displacement!`) over each
#    `[owned; ghost]` slice; only owned results are authoritative. The engine
#    mirrors the serial Verlet gate — between rebuilds each worker builds its
#    tree over the positions frozen at the last rebuild but solves on the
#    current ones (the serial "stale tree index, current distances" behaviour),
#    so with complete ghosts the owned results match serial up to float
#    reassociation.
#
#  * Reductions on the coordinator: the relative-density-error reduction is
#    gathered from per-worker partials; the model-hsml metric and the move
#    tally are pure functions of the gathered state and run with the serial
#    kernels. Redistribution (Metropolis, global) runs the serial
#    `redistribute_particles!` on the coordinator, then re-solves.
#
# IO (`write_output_distributed`) writes one Gadget file per non-empty slice
# or, for `single_file`, gathers to a single file. Worker launch
# (`init_workers`) is pluggable: local `addprocs`, or `SlurmManager`/
# `PBSManager` via ClusterManagers, with `:auto` detecting the scheduler from
# the environment. The ClusterManagers import lives only in this file.
# ===========================================================================

using Distributed
using StaticArrays
import GadgetIO
import ClusterManagers

# ---------------------------------------------------------------------------
# ClusterManagers / batch-scheduler-aware worker launch
# ---------------------------------------------------------------------------

"""
    detect_scheduler() -> Symbol

Inspect the environment and return `:slurm`, `:pbs` or `:local`. Pure /
side-effect-free — used by [`init_workers`](@ref)'s `:auto` mode.

* `:slurm` if `SLURM_JOB_ID` is set,
* else `:pbs` if `PBS_JOBID` is set,
* else `:local`.
"""
function detect_scheduler()
    if haskey(ENV, "SLURM_JOB_ID")
        return :slurm
    elseif haskey(ENV, "PBS_JOBID")
        return :pbs
    else
        return :local
    end
end

# Parse a positive integer from an env var, or `nothing` if absent/invalid.
function _env_int(key::AbstractString)
    haskey(ENV, key) || return nothing
    v = tryparse(Int, strip(ENV[key]))
    (v === nothing || v <= 0) ? nothing : v
end

"""
    scheduler_pool_size(sched::Symbol) -> Int

Number of worker processes the *scheduler allocation* implies (sized from the
allocation, not a hardcoded count). Side-effect-free.

* SLURM: `SLURM_NTASKS`, else `SLURM_NNODES`, else `1`.
* PBS:   the number of lines in `PBS_NODEFILE` (if readable), else
  `PBS_NP`/`NCPUS`, else `1`.
* local: `1` (caller passes an explicit `n`).
"""
function scheduler_pool_size(sched::Symbol)
    if sched === :slurm
        n = _env_int("SLURM_NTASKS")
        n === nothing && (n = _env_int("SLURM_NNODES"))
        return n === nothing ? 1 : n
    elseif sched === :pbs
        nf = get(ENV, "PBS_NODEFILE", "")
        if !isempty(nf) && isfile(nf)
            cnt = 0
            for ln in eachline(nf)
                isempty(strip(ln)) || (cnt += 1)
            end
            cnt > 0 && return cnt
        end
        n = _env_int("PBS_NP")
        n === nothing && (n = _env_int("NCPUS"))
        return n === nothing ? 1 : n
    else
        return 1
    end
end

"""
    plan_workers(; manager=:auto, n=nothing)
        -> (chosen_manager::Symbol, count::Int)

Pure planning step behind [`init_workers`](@ref): resolve the *effective*
manager and worker count without launching anything.

* `manager=:auto` → `detect_scheduler()`; the count is the explicit `n` if
  given, else `scheduler_pool_size(chosen)`.
* `manager=:local` → `(:local, n === nothing ? max(1, Sys.CPU_THREADS) : n)`.
* `manager ∈ (:slurm,:pbs)` → that manager; count = `n` if given else
  `scheduler_pool_size(manager)`.
"""
function plan_workers(; manager::Symbol = :auto, n::Union{Nothing,Integer} = nothing)
    chosen = manager === :auto ? detect_scheduler() : manager
    chosen in (:local, :slurm, :pbs) ||
        throw(ArgumentError("unknown manager $(manager); use :auto/:local/:slurm/:pbs"))
    if chosen === :local
        cnt = n === nothing ? max(1, Sys.CPU_THREADS) : Int(n)
    else
        cnt = n === nothing ? scheduler_pool_size(chosen) : Int(n)
    end
    return chosen, max(1, cnt)
end

"""
    init_workers(; manager=:auto, n=nothing, project=Base.active_project(),
                 depot=first(DEPOT_PATH), timeout=300, exeflags=String[],
                 dry_run=false, kwargs...) -> Vector{Int}

Pluggable worker-launch entry point. Selects local `addprocs` vs
`addprocs(SlurmManager(n))` vs `addprocs(PBSManager(n))`; `:auto` detects the
scheduler from the environment and sizes the pool from the allocation.

`--project` (`project`) and `JULIA_DEPOT_PATH` (`depot`) are propagated to
the workers so they can `using WVTICs` on remote nodes (no shared address
space assumed — message-passing only).  `timeout` (seconds) sets
`JULIA_WORKER_TIMEOUT` (the Distributed worker-handshake timeout) for the
launch — a generous default for staggered scheduler startup; the SLURM
manager additionally applies its own `ExponentialBackOff`.  Extra `kwargs`
are forwarded to `addprocs`.  Returns the new worker ids (`workers()` minus
the existing pool).

`dry_run=true` performs only the planning step ([`plan_workers`](@ref)) and
returns an empty vector — used by the env-detection unit test so CI never
launches a real scheduler.
"""
function init_workers(; manager::Symbol = :auto,
                        n::Union{Nothing,Integer} = nothing,
                        project::AbstractString = (Base.active_project() === nothing ?
                                                   "@." : Base.active_project()),
                        depot::AbstractString = (isempty(DEPOT_PATH) ? "" :
                                                 first(DEPOT_PATH)),
                        timeout::Real = 300,
                        exeflags = String[],
                        dry_run::Bool = false,
                        kwargs...)
    chosen, cnt = plan_workers(; manager = manager, n = n)
    dry_run && return Int[]

    flags = String["--project=$(project)"]
    append!(flags, collect(String.(exeflags)))
    env = Pair{String,String}[]
    isempty(depot) || push!(env, "JULIA_DEPOT_PATH" => depot)

    # Generous worker-handshake timeout for staggered scheduler startup
    # (restored after launch so the global state is untouched).
    prev_to = get(ENV, "JULIA_WORKER_TIMEOUT", nothing)
    ENV["JULIA_WORKER_TIMEOUT"] = string(Int(round(timeout)))

    existing = Set(workers())
    try
        if chosen === :local
            addprocs(cnt; exeflags = flags, env = env,
                     topology = :master_worker, kwargs...)
        elseif chosen === :slurm
            # SlurmManager(np) has an Integer constructor; addprocs over it is
            # the ClusterManagers v2.x launch path.
            addprocs(ClusterManagers.SlurmManager(cnt);
                     exeflags = flags, env = env, kwargs...)
        elseif chosen === :pbs
            # PBSManager's struct is (np, queue, wd) with no 1-arg
            # constructor; `addprocs_pbs(np; …)` is the v2.x convenience that
            # builds it with sane defaults (queue = ``, wd = ENV["HOME"]) and
            # the retry/backoff.
            ClusterManagers.addprocs_pbs(cnt; exeflags = flags, env = env,
                                         kwargs...)
        end
    finally
        if prev_to === nothing
            haskey(ENV, "JULIA_WORKER_TIMEOUT") &&
                delete!(ENV, "JULIA_WORKER_TIMEOUT")
        else
            ENV["JULIA_WORKER_TIMEOUT"] = prev_to
        end
    end
    return [w for w in workers() if !(w in existing)]
end

# ---------------------------------------------------------------------------
# 1. Domain decomposition by Peano–Hilbert space-filling curve
# ---------------------------------------------------------------------------
#
# GadgetIO's `peano_hilbert_key(bits,x,y,z)` + `get_int_pos(pos,corner,fac,
# bits)` are reused directly. `bits=21` is GadgetIO's 3-D Peano resolution
# (`KeyHeader`/`peano_hilbert_key` default): 2^21 cells per axis, ample for
# any WVTICs N.

const PEANO_BITS = 21

"""
    peano_keys(positions, box) -> Vector{Int}

Per-particle 3-D Peano–Hilbert key (GadgetIO `peano_hilbert_key` /
`get_int_pos`, `PEANO_BITS` resolution).  `box::NTuple{3,Float64}`
is the problem box; the curve is laid over `[0,box]` with the cube side =
`max(box)` (so the integer grid is isotropic, matching GadgetIO's
single-`domain_fac` convention).  Returns `Int` keys (fits in `Int64` for
21 bits → ≤ 2^63).
"""
function peano_keys(positions::AbstractVector{SVector{3,Float64}},
                    box::NTuple{3,Float64})
    side = max(box[1], box[2], box[3])
    side <= 0.0 && (side = 1.0)
    fac = (1 << PEANO_BITS) / side
    n = length(positions)
    keys = Vector{Int}(undef, n)
    @inbounds for i in 1:n
        p = positions[i]
        ix = GadgetIO.get_int_pos(p[1], 0.0, fac, PEANO_BITS)
        iy = GadgetIO.get_int_pos(p[2], 0.0, fac, PEANO_BITS)
        iz = GadgetIO.get_int_pos(p[3], 0.0, fac, PEANO_BITS)
        keys[i] = Int(GadgetIO.peano_hilbert_key(PEANO_BITS, ix, iy, iz))
    end
    return keys
end

# Recursive load-balanced bisection over the sorted key array (Gadget-style
# domain split). Each `count[i]` is a per-position cost (==1 here, but kept as
# an array so a per-particle work model drops in unchanged). `bounds[cpu]`
# receives the (first,last) sorted-index range for processor `cpu` (1-based,
# cpustart..cpustart+ncpu-1).
function _find_split!(bounds::Vector{Tuple{Int,Int}},
                      count::Vector{Int}, cpustart::Int, ncpu::Int,
                      First::Int, Last::Int)
    if ncpu == 1
        bounds[cpustart] = (First, Last)
        return
    end
    ncpu_left  = ncpu ÷ 2
    ncpu_right = ncpu - ncpu_left

    # Both sides must hold at least their own cpu count so the recursion can
    # never produce an empty range (split ∈ [lo, hi], inclusive):
    #   left  = First:(split-1)  needs ≥ ncpu_left  elements ⇒ split ≥ lo
    #   right = split:Last       needs ≥ ncpu_right elements ⇒ split ≤ hi
    lo = First + ncpu_left
    hi = Last - ncpu_right + 1

    load = 0
    @inbounds for i in First:Last
        load += count[i]
    end

    # Seed the split load-proportionally to the left cpu fraction
    # (`split ≈ First + n·ncpu_left/ncpu`, optimal to ±1 for unit cost), then
    # slide it either way while that lowers the larger per-cpu average load.
    # The split is clamped to [lo, hi] so each side is non-empty and holds ≥
    # its own cpu count (a non-empty interval whenever ncpu ≤ n).
    n     = Last - First + 1
    split = First + round(Int, n * ncpu_left / ncpu)
    split = clamp(split, lo, hi)
    load_left = 0
    @inbounds for i in First:(split - 1)
        load_left += count[i]
    end

    @inbounds while true
        cur = max(load_left / ncpu_left,
                  (load - load_left) / ncpu_right)
        moved = false
        # slide right: move element `split` from the right side to the left
        if split < hi
            nw = max((load_left + count[split]) / ncpu_left,
                     (load - load_left - count[split]) / ncpu_right)
            if nw < cur
                load_left += count[split]
                split += 1
                moved = true
            end
        end
        # else slide left: move element `split-1` from left to right side
        if !moved && split > lo
            nw = max((load_left - count[split - 1]) / ncpu_left,
                     (load - load_left + count[split - 1]) / ncpu_right)
            if nw < cur
                load_left -= count[split - 1]
                split -= 1
                moved = true
            end
        end
        moved || break
    end
    split = clamp(split, lo, hi)

    _find_split!(bounds, count, cpustart, ncpu_left, First, split - 1)
    _find_split!(bounds, count, cpustart + ncpu_left, ncpu_right,
                 split, Last)
    return
end

"""
    Decomposition

Result of [`decompose_domain`](@ref).

* `order`  : the permutation sorting `1:N` by Peano key (`order[p]` is the
  original particle index at sorted position `p`).
* `bounds` : `Vector{Tuple{Int,Int}}`, length `nparts`; `bounds[w]` is the
  inclusive `(first,last)` *sorted-position* range owned by worker slot `w`.
* `owner`  : `Vector{Int}` of length `N`; `owner[i]` is the worker slot
  (1-based) owning original particle `i`.
"""
struct Decomposition
    order::Vector{Int}
    bounds::Vector{Tuple{Int,Int}}
    owner::Vector{Int}
end

"""
    decompose_domain(positions, box, nparts) -> Decomposition

Peano–Hilbert space-filling-curve domain decomposition. Computes PH keys
([`peano_keys`](@ref)), sorts `1:N` by key, and splits the sorted array into
`nparts` contiguous, particle-count-balanced ranges via the recursive
load-balanced bisection [`_find_split!`](@ref). Node-count agnostic: the split
keys off `nparts` only.

Guarantees: keys sorted within each partition, partitions contiguous &
disjoint, union == `1:N`, and particle-count balance to within one per
recursive bisection level.
"""
function decompose_domain(positions::AbstractVector{SVector{3,Float64}},
                          box::NTuple{3,Float64}, nparts::Integer)
    n = length(positions)
    nparts = max(1, Int(nparts))
    keys = peano_keys(positions, box)
    order = sortperm(keys)                     # stable enough; ties broken by idx
    bounds = Vector{Tuple{Int,Int}}(undef, nparts)
    owner = Vector{Int}(undef, n)
    if n == 0
        for w in 1:nparts
            bounds[w] = (1, 0)                 # empty range
        end
        return Decomposition(order, bounds, owner)
    end
    if nparts == 1
        bounds[1] = (1, n)
        @inbounds for p in 1:n
            owner[order[p]] = 1
        end
        return Decomposition(order, bounds, owner)
    end
    cost = ones(Int, n)                        # unit per-particle cost
    nparts = min(nparts, n)                    # never more parts than particles
    bnd = Vector{Tuple{Int,Int}}(undef, nparts)
    _find_split!(bnd, cost, 1, nparts, 1, n)
    # pad if nparts was clamped to n (extra workers get empty ranges)
    for w in 1:length(bounds)
        bounds[w] = w <= nparts ? bnd[w] : (1, 0)
    end
    @inbounds for w in 1:nparts
        f, l = bounds[w]
        for p in f:l
            owner[order[p]] = w
        end
    end
    return Decomposition(order, bounds, owner)
end

# ---------------------------------------------------------------------------
# 2. Ghost / halo exchange
# ---------------------------------------------------------------------------

# AABB (min,max corner) of a set of positions.
function _aabb(positions::Vector{SVector{3,Float64}}, idx::AbstractVector{Int})
    if isempty(idx)
        z = SVector{3,Float64}(0.0, 0.0, 0.0)
        return z, z
    end
    @inbounds begin
        lo = positions[idx[1]]
        hi = positions[idx[1]]
        for t in 2:length(idx)
            p = positions[idx[t]]
            lo = SVector{3,Float64}(min(lo[1], p[1]), min(lo[2], p[2]),
                                    min(lo[3], p[3]))
            hi = SVector{3,Float64}(max(hi[1], p[1]), max(hi[2], p[2]),
                                    max(hi[3], p[3]))
        end
    end
    return lo, hi
end

# Per-boundary halo width = 2·max(hsml) over `idx`, recomputed from the current
# hsml each time.
function _halo_width(hsml::Vector{Float32}, idx::AbstractVector{Int})
    m = 0.0
    @inbounds for t in eachindex(idx)
        h = Float64(hsml[idx[t]])
        h > m && (m = h)
    end
    return 2.0 * m
end

# Minimum-image distance from point `p` to the AABB [lo,hi], per-axis,
# periodicity-aware.  0 inside the box.  Used to decide whether a remote
# particle is within `width` of this worker's domain (ghost candidate),
# including across periodic faces.
@inline function _aabb_dist2(p::SVector{3,Float64}, lo::SVector{3,Float64},
                             hi::SVector{3,Float64}, box::NTuple{3,Float64},
                             periodic::NTuple{3,Bool})
    d2 = 0.0
    @inbounds for a in 1:3
        x = p[a]
        l = lo[a]
        h = hi[a]
        # gap from x (and, on a periodic axis, its ±L images) to [l,h]
        dd = _gap_to_interval(x, l, h)
        if periodic[a] && box[a] > 0.0
            L = box[a]
            dd = min(dd, _gap_to_interval(x - L, l, h),
                         _gap_to_interval(x + L, l, h))
        end
        d2 += dd * dd
    end
    return d2
end

# Non-negative gap from scalar `x` to the closed interval `[l,h]` (0 inside).
@inline function _gap_to_interval(x::Float64, l::Float64, h::Float64)
    return x < l ? (l - x) : (x > h ? (x - h) : 0.0)
end

"""
    select_ghosts(remote_pos, lo, hi, width, box, periodic) -> Vector{Int}

Indices into `remote_pos` of particles within `width` of the AABB `[lo,hi]`
(per-axis minimum-image, periodic-face aware). Pure helper used by the halo
exchange.
"""
function select_ghosts(remote_pos::Vector{SVector{3,Float64}},
                        lo::SVector{3,Float64}, hi::SVector{3,Float64},
                        width::Float64, box::NTuple{3,Float64},
                        periodic::NTuple{3,Bool})
    out = Int[]
    w2 = width * width
    @inbounds for j in eachindex(remote_pos)
        if _aabb_dist2(remote_pos[j], lo, hi, box, periodic) <= w2
            push!(out, j)
        end
    end
    return out
end

# ---------------------------------------------------------------------------
# Worker kernels (run on each worker via remotecall)
# ---------------------------------------------------------------------------
#
# A worker receives a slice = [owned; ghost] with the owned particles at
# indices 1:nown. It builds a local KDTree over the slice and runs the
# existing SPH kernels over it; only the owned results (1:nown) are
# authoritative and returned to the coordinator. Ghosts are read-only
# neighbour sources whose own authoritative results are computed by their
# owning worker. If the slice contains every neighbour of each owned particle
# within its interaction radius, the owned results equal the serial
# computation for the same positions (up to floating-point reassociation).

# Density solve over one slice. Returns the owned SPH quantities and this
# slice's partial relative-density-error reduction (combined on the coordinator
# → a genuine distributed reduction over the solved densities). `maxh_solved`
# is the largest solved hsml over owned particles — the coordinator uses it to
# check the ghost halo was wide enough (see [`engine_density_solve!`](@ref)).
#
# The KDTree is built over `tree_pos` (the positions frozen at the last Verlet
# rebuild) while the solve reads `slice.pos` (the current positions) for the
# distances — mirroring the serial solve, which reuses a Verlet-cached tree
# (stale index) but recomputes distances on the current positions. At a rebuild
# `tree_pos == slice.pos`.
function _worker_density_solve(slice::Particles,
                               tree_pos::Vector{SVector{3,Float64}}, nown::Int,
                               param::Parameters, problem::ProblemParameters,
                               prob::Problem, kc::KernelConfig)
    tree = build_tree(tree_pos)
    find_sph_quantities!(slice, param, problem, prob, kc; tree = tree)
    bias = param.density_function_correction
    rho = Vector{Float32}(undef, nown)
    hsml = Vector{Float32}(undef, nown)
    vhf = Vector{Float32}(undef, nown)
    emin = floatmax(Float64)
    emax = 0.0
    esum = 0.0
    esq = 0.0
    maxh_solved = 0.0
    @inbounds for i in 1:nown
        rm = Float64(prob.density(slice, i, bias))
        err = abs((Float64(slice.rho[i]) - rm) / rm)
        emin = min(emin, err)
        emax = max(emax, err)
        esum += err
        esq += err * err
        rho[i] = slice.rho[i]
        hsml[i] = slice.hsml[i]
        vhf[i] = slice.varhsmlfac[i]
        maxh_solved = max(maxh_solved, Float64(slice.hsml[i]))
    end
    return (rho = rho, hsml = hsml, varhsmlfac = vhf, emin = emin,
            emax = emax, esum = esum, esq = esq, maxh_solved = maxh_solved)
end

# WVT repulsive displacement over one slice. `pos`/`mhsml` cover [owned; ghost]
# (indices 1:nown owned); returns the owned displacement components. The
# candidate lists are built at `query_r` over the slice's own KDTree, so —
# given complete ghosts — each owned particle's neighbour set matches serial.
function _worker_displacement(pos::Vector{SVector{3,Float64}},
                              mhsml::Vector{Float64}, nown::Int,
                              kc::KernelConfig, step::Float64,
                              box::NTuple{3,Float64},
                              periodic::NTuple{3,Bool}, dim::Int,
                              query_r::Float64)
    m = length(pos)
    tree = build_tree(pos)
    nchunks = max(1, Threads.nthreads())
    chunks = _chunk_ranges(nown, nchunks)          # owned particles only
    nc = length(chunks)
    wscratch = [WvtScratch() for _ in 1:nc]
    cand_lists = [Int[] for _ in 1:m]
    _rebuild_candidate_lists!(cand_lists, pos, tree, query_r, box, periodic,
                              wscratch, chunks)
    dx = zeros(Float32, m)
    dy = zeros(Float32, m)
    dz = zeros(Float32, m)
    _run_chunks(nc) do c
        _wvt_displacement_chunk!(c, chunks, pos, mhsml, dx, dy, dz,
                                 cand_lists, kc, step, box, periodic, dim)
    end
    return (dx[1:nown], dy[1:nown], dz[1:nown])
end

# ---------------------------------------------------------------------------
# Distributed execution engine (coordinator side)
# ---------------------------------------------------------------------------
#
# The coordinator holds the authoritative `Particles`. Each iteration it
# re-decomposes the domain by Peano key across the workers, exchanges ghost
# halos, runs the density solve and displacement on the workers via
# `remotecall`, scatters the owned results back, and combines the gathered
# per-worker partial reductions. The relaxation control flow (step reduction,
# convergence, redistribution) stays on the coordinator, driven through the
# same `WvtEngine` interface the `LocalEngine` implements.

mutable struct DistributedEngine <: WvtEngine
    workers::Vector{Int}
    chunks::Vector{UnitRange{Int}}    # coordinator-side chunking (move tally)
    box::NTuple{3,Float64}
    periodic::NTuple{3,Bool}
    decomp::Union{Nothing,Decomposition}
    owned::Vector{Vector{Int}}        # per-worker owned global indices
    slice_gids::Vector{Vector{Int}}   # per-worker [owned; ghost] (fixed between rebuilds)
    widths::Vector{Float64}           # per-worker ghost halo width
    ref_pos::Vector{SVector{3,Float64}}  # positions frozen at the last rebuild
    have_state::Bool                  # a decomposition + halo exist to reuse
    red_error::NTuple{4,Float64}      # combined error reduction (min,max,sum,sq)
    prev_query_r::Float64             # displacement query_r (halo carry-over)
    r_skin::Float64
end

function DistributedEngine(particles::Particles, problem::ProblemParameters)
    n = length(particles)
    nchunks = max(1, Threads.nthreads())
    return DistributedEngine(workers(), _chunk_ranges(n, nchunks),
                             problem.Boxsize, problem.Periodic, nothing,
                             Vector{Vector{Int}}(), Vector{Vector{Int}}(),
                             Float64[], copy(particles.pos), false,
                             (floatmax(Float64), 0.0, 0.0, 0.0), 0.0, 0.0)
end

# Owned + ghost global indices for worker slot `w` at halo width `width`.
function _worker_slice_gids(e::DistributedEngine, pos::Vector{SVector{3,Float64}},
                            w::Int, width::Float64)
    owned = e.owned[w]
    isempty(owned) && return owned, Int[]
    lo, hi = _aabb(pos, owned)
    sel = select_ghosts(pos, lo, hi, width, e.box, e.periodic)
    ghost = Int[j for j in sel if e.decomp.owner[j] != w]
    return owned, ghost
end

# Iteration-1 halo width floor: no solved hsml yet, so size the halo from the
# model-hsml scale (a large bound; extra ghosts are harmless, missing ones are
# not) and let the grow-retry loop widen it if the solve needs more.
function _initial_width(problem::ProblemParameters, kc::KernelConfig,
                        prev_query_r::Float64)
    voln = _vol_norm(kc.dim)
    h_est = kc.dim == 2 ?
        sqrt(Float64(kc.desnngb) * problem.Mpart / pi) :
        cbrt(Float64(kc.desnngb) * problem.Mpart / voln)
    return max(4.0 * h_est, prev_query_r)
end

# Density solve with a Verlet gate mirroring the serial path. A rebuild
# (first call, forced re-solve, or accumulated displacement past 0.5·r_skin)
# re-decomposes, grows each worker's halo until it covers 2·(max solved hsml)
# so every owned particle's neighbour set is complete, and freezes the
# decomposition + positions. Between rebuilds the decomposition + halo are
# reused and the workers build their tree over the frozen positions but solve
# on the current ones — exactly the serial "stale tree, current distances"
# behaviour, so the owned results match serial up to float reassociation.
function engine_density_solve!(e::DistributedEngine, particles::Particles,
                               param::Parameters, problem::ProblemParameters,
                               prob::Problem, kc::KernelConfig, n::Int,
                               boxv::NTuple{3,Float64},
                               periodic::NTuple{3,Bool}; force::Bool = false)
    rebuild = force || !e.have_state
    if !rebuild
        md2 = _max_disp2(particles.pos, e.ref_pos, n, boxv, periodic, e.chunks)
        rebuild = md2 > (0.5 * e.r_skin)^2
    end

    if rebuild
        results = _rebuild_and_solve!(e, particles, param, problem, prob, kc)
        copyto!(e.ref_pos, particles.pos)
        e.have_state = true
    else
        results = _reuse_solve!(e, particles, param, problem, prob, kc)
    end
    _scatter_and_combine!(e, particles, results)
    return nothing
end

# Rebuild: decompose, then grow each worker's halo until it covers
# 2·(max solved hsml). At a rebuild the frozen tree positions equal the current
# positions. Returns the per-worker solve results and stores the fixed slices.
function _rebuild_and_solve!(e::DistributedEngine, particles::Particles,
                             param::Parameters, problem::ProblemParameters,
                             prob::Problem, kc::KernelConfig)
    nparts = length(e.workers)
    e.decomp = decompose_domain(particles.pos, problem.Boxsize, nparts)
    e.owned = [Int[e.decomp.order[p]
                    for p in e.decomp.bounds[w][1]:e.decomp.bounds[w][2]]
               for w in 1:nparts]
    e.slice_gids = [Int[] for _ in 1:nparts]
    w0 = _initial_width(problem, kc, e.prev_query_r)
    e.widths = [max(w0, _halo_width(particles.hsml, e.owned[w]))
                for w in 1:nparts]

    results = Vector{Any}(undef, nparts)
    pending = [w for w in 1:nparts if !isempty(e.owned[w])]
    max_rounds = 8
    for round in 1:max_rounds
        isempty(pending) && break
        futs = Dict{Int,Any}()
        for w in pending
            _, ghost = _worker_slice_gids(e, particles.pos, w, e.widths[w])
            e.slice_gids[w] = vcat(e.owned[w], ghost)
            slice = _subset_particles(particles, e.slice_gids[w])
            futs[w] = remotecall(_worker_density_solve, e.workers[w], slice,
                                 copy(slice.pos), length(e.owned[w]), param,
                                 problem, prob, kc)
        end
        newpending = Int[]
        for w in pending
            res = fetch(futs[w])
            results[w] = res
            # halo must cover 2·(max solved hsml) for a complete neighbour set
            if e.widths[w] + 1e-12 < 2.0 * res.maxh_solved
                e.widths[w] = 2.0 * res.maxh_solved
                push!(newpending, w)
            end
        end
        pending = newpending
        if round == max_rounds && !isempty(pending)
            @warn "distributed density halo did not converge in $max_rounds rounds; results may deviate from serial for slots $pending"
        end
    end
    return results
end

# Reuse: solve the fixed slices, building each worker's tree over the frozen
# (last-rebuild) positions while the solve reads the current positions.
function _reuse_solve!(e::DistributedEngine, particles::Particles,
                       param::Parameters, problem::ProblemParameters,
                       prob::Problem, kc::KernelConfig)
    nparts = length(e.workers)
    results = Vector{Any}(undef, nparts)
    futs = Dict{Int,Any}()
    for w in 1:nparts
        isempty(e.owned[w]) && continue
        gids = e.slice_gids[w]
        slice = _subset_particles(particles, gids)            # current pos/hsml
        tree_pos = SVector{3,Float64}[e.ref_pos[g] for g in gids]  # frozen
        futs[w] = remotecall(_worker_density_solve, e.workers[w], slice,
                             tree_pos, length(e.owned[w]), param, problem,
                             prob, kc)
    end
    for w in 1:nparts
        isempty(e.owned[w]) && continue
        results[w] = fetch(futs[w])
    end
    return results
end

# Scatter owned SPH results into the coordinator arrays and combine the
# gathered per-worker error partials (associative ops → serial-equivalent).
function _scatter_and_combine!(e::DistributedEngine, particles::Particles,
                               results::Vector{Any})
    emin = floatmax(Float64)
    emax = 0.0
    esum = 0.0
    esq = 0.0
    @inbounds for w in 1:length(e.owned)
        isempty(e.owned[w]) && continue
        res = results[w]
        emin = min(emin, res.emin)
        emax = max(emax, res.emax)
        esum += res.esum
        esq += res.esq
        owned = e.owned[w]
        for k in eachindex(owned)
            g = owned[k]
            particles.rho[g] = res.rho[k]
            particles.hsml[g] = res.hsml[k]
            particles.varhsmlfac[g] = res.varhsmlfac[k]
        end
    end
    e.red_error = (emin, emax, esum, esq)
    return nothing
end

# Combined relative-density-error reduction (gathered worker partials over the
# solved densities → a genuine distributed reduction).
engine_error_stats(e::DistributedEngine, particles::Particles, prob::Problem,
                   n::Int, bias) = e.red_error

# Model-hsml metric. This is a pure function of the model density (analytic,
# per-position) — it needs no neighbour data, so it is computed on the
# coordinator with the exact serial kernel. Bit-identical to serial, which
# keeps `r_skin` (and hence the Verlet-gate decision) in lockstep with the
# serial path.
engine_model_hsml!(e::DistributedEngine, particles::Particles,
                   mhsml::Vector{Float64}, prob::Problem, n::Int, bias,
                   wvtnngb::Float64, mpart::Float64, voln::Float64, dim::Int,
                   median_boxsize::Float64) =
    _model_hsml_metric!(particles, mhsml, prob.density, n, bias, wvtnngb,
                        mpart, voln, dim, median_boxsize)

# Run the displacement on the workers over [owned; ghost] slices (ghosts wide
# enough to cover query_r) and scatter the owned deltas back.
function engine_displacement!(e::DistributedEngine, particles::Particles,
                              mhsml::Vector{Float64},
                              deltas::NTuple{3,Vector{Float32}},
                              kc::KernelConfig, step::Float64,
                              boxv::NTuple{3,Float64},
                              periodic::NTuple{3,Bool}, dim::Int,
                              voln::Float64, max_hsml_norm::Float64,
                              mean_hsml::Float64, n::Int)
    e.r_skin = _skin_radius(mean_hsml)
    query_r = max_hsml_norm * 1.05 + e.r_skin
    e.prev_query_r = query_r
    nparts = length(e.workers)
    dx, dy, dz = deltas
    futs = Dict{Int,Any}()
    for w in 1:nparts
        isempty(e.owned[w]) && continue
        width = max(e.widths[w], query_r)
        _, ghost = _worker_slice_gids(e, particles.pos, w, width)
        gids = vcat(e.owned[w], ghost)
        slicepos = SVector{3,Float64}[particles.pos[g] for g in gids]
        slicemhsml = Float64[mhsml[g] for g in gids]
        futs[w] = remotecall(_worker_displacement, e.workers[w], slicepos,
                             slicemhsml, length(e.owned[w]), kc, step, boxv,
                             periodic, dim, query_r)
    end
    @inbounds for w in 1:nparts
        isempty(e.owned[w]) && continue
        odx, ody, odz = fetch(futs[w])
        owned = e.owned[w]
        for k in eachindex(owned)
            g = owned[k]
            dx[g] = odx[k]
            dy[g] = ody[k]
            dz[g] = odz[k]
        end
    end
    return nothing
end

# Move + per-axis box wrap + moveMps tallies on the coordinator (needs only
# the gathered deltas + solved hsml).
engine_move!(e::DistributedEngine, particles::Particles,
             deltas::NTuple{3,Vector{Float32}}, desnngb::Float64,
             voln::Float64, dim::Int, boxv::NTuple{3,Float64},
             periodic::NTuple{3,Bool}) =
    _move_particles!(particles, deltas, desnngb, voln, dim, boxv, periodic,
                     e.chunks)

# ---------------------------------------------------------------------------
# Global per-iteration reduction primitive
# ---------------------------------------------------------------------------

"""
    distributed_iteration_reductions(particles, decomp, prob, kc, mpart,
                                     density_function_correction)
        -> NamedTuple

Compute the global per-iteration reductions from per-worker-slot partials:
each Peano slot's owned slice produces partial error/`vSphSum`/`max_hsml`
sums, and the coordinator combines them with the same associative/commutative
ops as the serial `_error_stats` / `_fill_model_hsml!`, so the global
`errMean`/`norm_hsml` match the serial values up to floating-point
reassociation.

Iterating slot-by-slot over the Peano-sorted order (not the raw `1:N`) makes
the partition explicit — the partials are the per-worker contributions a
gather-combine collects, and they sum to the global serial reduction.
"""
function distributed_iteration_reductions(particles::Particles,
                                          decomp::Decomposition,
                                          prob::Problem, kc::KernelConfig,
                                          mpart::Real,
                                          density_function_correction)
    n = length(particles)
    dim = kc.dim
    voln = dim == 2 ? Float64(pi) : (4.0 * pi / 3.0)
    wvtnngb = Float64(kc.desnngb)
    mpart = Float64(mpart)
    emin = floatmax(Float64)
    emax = 0.0
    esum = 0.0
    esq = 0.0
    vsph = 0.0
    maxh = 0.0
    nparts = length(decomp.bounds)
    @inbounds for w in 1:nparts
        f, l = decomp.bounds[w]
        # per-worker-slot partial — the `prob.density` ::Function-field call is
        # isolated behind `_reduce_slot` (a `where {F}` function barrier) so the
        # inner loop is type-stable.
        pmin, pmax, psum, psq, pvs, pmh =
            _reduce_slot(particles, decomp.order, prob.density,
                         density_function_correction, f, l,
                         wvtnngb, mpart, voln, dim)
        emin = min(emin, pmin)
        emax = max(emax, pmax)
        esum += psum
        esq += psq
        vsph += pvs
        maxh = max(maxh, pmh)
    end
    errMean = esum / max(1, n)
    errSigma = sqrt(max(0.0, esq / max(1, n) - errMean * errMean))
    return (errMin = emin, errMax = emax, errMean = errMean,
            errSigma = errSigma, vSphSum = vsph, max_hsml = maxh)
end

# Per-worker-slot reduction over one decomposition slice. Function barrier on
# the `prob.density` ::Function field (`dfun::F` concrete inside) so the inner
# loop is type-stable. Returns the slot partials.
@noinline function _reduce_slot(particles::Particles, order::Vector{Int},
                                dfun::F, density_function_correction,
                                f::Int, l::Int, wvtnngb::Float64,
                                mpart::Float64, voln::Float64,
                                dim::Int) where {F}
    pmin = floatmax(Float64)
    pmax = 0.0
    psum = 0.0
    psq = 0.0
    pvs = 0.0
    pmh = 0.0
    @inbounds for p in f:l
        i = order[p]
        rm = Float64(dfun(particles, i, density_function_correction))
        err = abs((Float64(particles.rho[i]) - rm) / rm)
        pmin = min(pmin, err)
        pmax = max(pmax, err)
        psum += err
        psq += err * err
        if dim == 2
            h = sqrt(wvtnngb * mpart / rm / pi)
            pvs += h * h
        else
            h = cbrt(wvtnngb * mpart / rm / voln)
            pvs += h * h * h
        end
        pmh = max(pmh, h)
    end
    return pmin, pmax, psum, psq, pvs, pmh
end

"""
    regularise_sph_particles_distributed!(particles, param, problem, prob, kc;
        kwargs...) -> particles

Top-level distributed driver. Runs the WVT relaxation across `nworkers()`
processes: each iteration re-decomposes the domain by Peano key, exchanges
ghost halos, and runs the density solve and displacement on the workers via
`remotecall`, gathering the global reductions to the coordinator. The
relaxation control flow (step reduction, convergence, redistribution) and the
authoritative `Particles` stay on the coordinator, driven through the same
loop as the serial path ([`_regularise_loop!`](@ref)) with a
[`DistributedEngine`](@ref).

When `nworkers() == 1` this delegates to the serial/threaded
[`regularise_sph_particles!`](@ref), giving the identical single-process result.

With complete ghost halos the per-particle results match serial up to
floating-point reassociation (neighbour sums in a different order, reductions
gathered per worker). Workers must have `using WVTICs` in scope; the driver
loads it defensively before the first `remotecall`.

`kwargs` are forwarded to the relaxation loop (`output_diagnostics`,
`verbose`, `seed`, …).
"""
function regularise_sph_particles_distributed!(particles::Particles,
                                               param::Parameters,
                                               problem::ProblemParameters,
                                               prob::Problem,
                                               kc::KernelConfig;
                                               kwargs...)
    param.Npart == 0 && return particles

    # No extra workers ⇒ the serial path (identical result single-process).
    if nworkers() <= 1
        return regularise_sph_particles!(particles, param, problem, prob, kc;
                                         kwargs...)
    end

    # Ensure the package is loaded on every worker (needed to resolve the
    # remotecall'd kernels + deserialise the SoA types). Idempotent.
    try
        Distributed.remotecall_eval(Main, workers(), :(using WVTICs))
    catch err
        @warn "could not load WVTICs on workers; ensure `@everywhere using WVTICs`" exception = err
    end

    engine = DistributedEngine(particles, problem)
    return _regularise_loop!(engine, particles, param, problem, prob, kc;
                             kwargs...)
end

# 4-arg convenience (mirrors the serial signature) — derive prob + kernel.
function regularise_sph_particles_distributed!(particles::Particles,
                                               param::Parameters,
                                               problem::ProblemParameters;
                                               kwargs...)
    prob = setup_problem(param)
    kc = default_kernel_config()
    return regularise_sph_particles_distributed!(particles, param, problem,
                                                 prob, kc; kwargs...)
end

# ---------------------------------------------------------------------------
# Distributed IO — one Gadget file per worker (GadgetIO multi-file)
# ---------------------------------------------------------------------------

"""
    write_output_distributed(particles, param, problem, decomp;
        filename=problem.Name, single_file=false, kwargs...) -> Vector{String}

Per-worker multi-file Gadget snapshot. Writes one SnapFormat-2 file per
non-empty decomposition slice (`filename.0 … filename.{k-1}`) holding that
slice's owned particles, so the set reads back via
`GadgetIO.read_header`/`read_block` and the per-file `npart[1]` sum to the
correct *global* particle count. `single_file=true` (or one slice) gathers to
a single file via the serial [`write_output`](@ref). Paths are taken as given
(absolute / shared-FS — no localhost assumption). `kwargs` forward to
`write_output`.

Each per-worker file's header records *that file's* particle count in both
`npart[1]` and `nall[1]` with `num_files = 1` (the WVTICs snapshot writer is
single-file by design). This is a valid standalone snapshot per file and the
slice union is loss-free; it is not the Gadget convention of a global `nall`
+ `num_files = k` across the set. For a single canonical file use
`single_file = true`.
"""
function write_output_distributed(particles::Particles, param::Parameters,
                                  problem::ProblemParameters,
                                  decomp::Decomposition;
                                  filename::AbstractString = problem.Name,
                                  single_file::Bool = false,
                                  verbose::Bool = false,
                                  kwargs...)
    nparts = length(decomp.bounds)
    nonempty = [w for w in 1:nparts if decomp.bounds[w][2] >= decomp.bounds[w][1]]
    if single_file || length(nonempty) <= 1
        write_output(particles, param, problem;
                     filename = filename, verbose = verbose, kwargs...)
        return String[filename]
    end
    written = String[]
    for (k, w) in enumerate(nonempty)
        f, l = decomp.bounds[w]
        gids = Int[decomp.order[p] for p in f:l]
        sub = _subset_particles(particles, gids)
        subparam = deepcopy(param)
        subparam.Npart = length(gids)        # this file's count
        subprob = deepcopy(problem)
        path = string(filename, ".", k - 1)
        write_output(sub, subparam, subprob;
                     filename = path, verbose = verbose, kwargs...)
        push!(written, path)
    end
    return written
end

# Materialise a Particles holding only the given global indices (owned slice
# for a per-worker file).  Order preserved.
function _subset_particles(p::Particles, gids::Vector{Int})
    m = length(gids)
    s = Particles(m)
    @inbounds for k in 1:m
        g = gids[k]
        s.pos[k] = p.pos[g]
        s.vel[k] = p.vel[g]
        s.id[k] = p.id[g]
        s.type[k] = p.type[g]
        s.key[k] = p.key[g]
        s.redistributed[k] = p.redistributed[g]
        s.u[k] = p.u[g]
        s.rho[k] = p.rho[g]
        s.hsml[k] = p.hsml[g]
        s.varhsmlfac[k] = p.varhsmlfac[g]
        s.rho_model[k] = p.rho_model[g]
        s.bfld[k] = p.bfld[g]
    end
    return s
end
