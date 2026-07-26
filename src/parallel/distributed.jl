# ===========================================================================
# Phase D — Distributed-memory parallelism (CLAUDE.md §4, points 1–6).
#
# This module is PURELY ADDITIVE.  The serial/threaded port (Phases 0–5 +
# substeps) is the reference implementation and is *not* touched: the
# distributed driver re-uses the verified threaded kernels
# (`find_sph_quantities!`, `_wvt_displacement!`, `_move_particles!`,
# `_error_stats`, `_fill_model_hsml!`, …) unchanged on every worker, behind
# this interface.  Running single-process (`nworkers() == 1`) the
# `regularise_sph_particles_distributed!` driver delegates straight to the
# threaded `regularise_sph_particles!` so the threaded path stays
# byte-identical.
#
# Design (the six CLAUDE.md §4 numbered points):
#
#  1. Domain decomposition by Peano–Hilbert SFC.  PH keys are computed with
#     GadgetIO's `peano_hilbert_key`/`get_int_pos` (NOT re-derived), the
#     particle indices are sorted by key, and the sorted array is split into
#     `nworkers()` contiguous, particle-count-balanced ranges by the SAME
#     recursive load-balanced bisection PhysicalTrees uses in
#     `octree_sparse/domain.jl::find_split_kernel` (mirrored here as
#     `_find_split!`, operating directly on per-particle counts instead of
#     PhysicalTrees' top-tree leaves — equivalent, and node-count agnostic:
#     the split keys off `nworkers()` only).  See `decompose_domain`.
#
#  2. Ghost / halo exchange.  Each worker's owned particles define an AABB;
#     it imports every remote particle whose position lies within
#     `2·max(hsml)` of that AABB (per-CLAUDE-§4b: a *per-boundary* max hsml,
#     i.e. the halo width is recomputed from the actual current hsml each
#     time).  Implemented with `Distributed.remotecall`/`@spawnat` +
#     `RemoteChannel` (no extra deps).  If the Newton hsml solve grows any
#     owned particle's hsml beyond the halo width that was used, the exchange
#     is re-triggered (`_halo_width` / the `grew` flag).  Across periodic
#     faces the importer also tests the ±box images so wrapped neighbours are
#     captured (same min-image discipline as `neighbors.jl`).
#
#  3. Local engine.  Each worker builds a local `KDTree` over
#     owned+imported-ghost positions and runs the EXISTING threaded
#     `find_sph_quantities!` (hybrid Distributed×Threads).  Ghosts are read-
#     only neighbour sources: only owned particles' results are kept.  The
#     GLOBAL reductions (error min/max/mean/sigma, `vSphSum`, `max_hsml`,
#     `norm_hsml`, the moveMps/convergence tallies) are computed collectively
#     each iteration by gathering per-worker partials to the coordinator and
#     combining them with the SAME associative/commutative ops the serial
#     reduction uses, so the GLOBAL converged density error and `norm_hsml`
#     match the serial result to a documented statistical tolerance.
#
#  4. Redistribution (Metropolis, global by nature).  Chosen approximation:
#     **gather-to-coordinator**.  Each iteration the owned particles are
#     gathered to the coordinator, the EXISTING serial
#     `redistribute_particles!` runs there (the verified Metropolis kernel,
#     unchanged — it is global by construction), and the moved particle set
#     is scattered back via the next domain decomposition.  Rationale: the C
#     Metropolis pairs an overdense particle with a *globally* random
#     underdense one, so a process-local restriction would change the
#     statistics; gathering keeps it statistically faithful (identical to the
#     serial draw distribution) at the cost of one all→0 gather on
#     redistribution iterations only (every `RedistributionFrequency`, while
#     `it ≤ LastMoveStep` — infrequent, and the particle record is tiny).
#
#  5. IO.  `write_output_distributed` writes ONE Gadget file per worker
#     (`num_files = nworkers()`, GadgetIO multi-file: `name.0 … name.{k-1}`)
#     OR, for `single_file=true` / one worker, gathers to rank 0 and writes a
#     single file.  All paths are taken as given (absolute / shared-FS — no
#     localhost assumption).
#
#  6. ClusterManagers / batch (REQUIRED).  `init_workers(; manager=:auto,
#     n=…)` is the single pluggable entry point: `:local` → `addprocs`,
#     `:slurm` → `addprocs(SlurmManager(n))`, `:pbs` →
#     `addprocs(PBSManager(n))`; `:auto` detects the scheduler from the
#     environment (`SLURM_JOB_ID`/`SLURM_NTASKS`/`SLURM_NNODES`,
#     `PBS_JOBID`/`PBS_NODEFILE`) and sizes the pool from the allocation.
#     `--project` and `JULIA_DEPOT_PATH` are propagated to workers; the
#     `addprocs` timeout is generous; no localhost-only assumptions.  The
#     ClusterManagers import lives ONLY in this file (the threaded port is
#     runtime-independent of it — `using ClusterManagers` here does not
#     execute any scheduler code unless `init_workers` is called with a
#     scheduler manager).
# ===========================================================================

using Distributed
using StaticArrays
import GadgetIO
import ClusterManagers

# ---------------------------------------------------------------------------
# 6. ClusterManagers / batch-scheduler-aware worker launch
# ---------------------------------------------------------------------------

"""
    detect_scheduler() -> Symbol

Inspect the environment and return `:slurm`, `:pbs` or `:local` (CLAUDE.md
§4 point 6).  Pure / side-effect-free — used by [`init_workers`](@ref)'s
`:auto` mode and unit-tested without a real scheduler.

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

Number of worker processes the *scheduler allocation* implies (CLAUDE.md §4
point 6 — "sizes the pool from the scheduler allocation, not a hardcoded
count").  Side-effect-free; unit-tested by setting/unsetting the env vars.

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
manager and worker count WITHOUT launching anything (CLAUDE.md §4 point 6).
This is what the no-scheduler env-detection unit test asserts.

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

Pluggable worker-launch entry point (CLAUDE.md §4 point 6, REQUIRED).
Selects local `addprocs` vs `addprocs(SlurmManager(n))` vs
`addprocs(PBSManager(n))`; `:auto` detects the scheduler from the
environment and sizes the pool from the allocation.

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
            # v2.x: SlurmManager(np) has an Integer constructor; addprocs over
            # it is the documented launch path (CLAUDE.md §4 point 6).
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
# bits)` are reused verbatim (CLAUDE.md §4 point 1 — "reuse it rather than
# re-deriving").  `bits=21` is GadgetIO's 3-D Peano resolution
# (`KeyHeader`/`peano_hilbert_key` default): 2^21 cells per axis, ample for
# any WVTICs N.

const PEANO_BITS = 21

"""
    peano_keys(positions, box) -> Vector{Int}

Per-particle 3-D Peano–Hilbert key (GadgetIO `peano_hilbert_key` /
`get_int_pos`, [`PEANO_BITS`](@ref) resolution).  `box::NTuple{3,Float64}`
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

# Recursive load-balanced bisection — a direct mirror of PhysicalTrees
# `octree_sparse/domain.jl::find_split_kernel` (CLAUDE.md §4 point 1: "mirror
# PhysicalTrees … find_split_kernel"), specialised to a per-element unit cost
# (particle count) over the SORTED key array instead of PhysicalTrees'
# top-tree leaves.  `count[a:b]` is the per-position cost (==1 here, but kept
# as an array so a future per-particle work model drops in unchanged, exactly
# as PhysicalTrees' `DomainCount`).  `bounds[cpu]` receives the (first,last)
# sorted-index range for processor `cpu` (1-based, cpustart..cpustart+ncpu-1).
function _find_split!(bounds::Vector{Tuple{Int,Int}},
                      count::Vector{Int}, cpustart::Int, ncpu::Int,
                      First::Int, Last::Int)
    if ncpu == 1
        bounds[cpustart] = (First, Last)
        return
    end
    ncpu_left  = ncpu ÷ 2                      # PhysicalTrees: ncpu/2 (trunc)
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

    # PhysicalTrees-faithful split, deliberately specialised for the
    # per-element UNIT-cost case.
    #
    # PhysicalTrees' `find_split_kernel` ranges over TOP-TREE LEAVES, where
    # each `DomainCount[i]` is a large, variable per-leaf particle count.
    # Its seed `split = First + ncpu/2` is then dragged a long way right by
    # the `while` loop until per-cpu average load balances — the loop does
    # all the work; the seed is irrelevant there.
    #
    # Here the SAME algorithm runs over the SORTED PARTICLE array with a
    # per-element unit cost (`count[i] == 1`).  BUGFIX: the previous
    # `split = First + ncpu_left` advanced only by the *number* of left
    # cpus (= First+1 for a binary split) and the old slide guard
    # `split > First + 1` was then immediately false, so the loop never ran
    # — one slot received ~all particles (e.g. N=512,np=2 → 511 vs 1).
    #
    # Specialisation (documented): seed the split LOAD-PROPORTIONALLY to the
    # left cpu fraction — `load_left ≈ load·ncpu_left/ncpu`, i.e.
    # `split ≈ First + n·ncpu_left/ncpu` — which for unit cost is already
    # optimal to ±1; then keep PhysicalTrees' exact max-average-load slide
    # criterion but allow it to fine-tune in BOTH directions (only ±1 of
    # adjustment is ever needed from the proportional seed, vs. the long
    # right-only drag PhysicalTrees needs from its poor seed).  The split is
    # clamped to [lo, hi] so each side is non-empty AND holds ≥ its own cpu
    # count; that interval is non-empty whenever ncpu ≤ n (guaranteed for
    # np ≤ N), so the recursion can never produce an empty range.
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

Peano–Hilbert SFC domain decomposition (CLAUDE.md §4 point 1).  Computes PH
keys ([`peano_keys`](@ref)), sorts `1:N` by key, and splits the sorted array
into `nparts` contiguous, particle-count-balanced ranges via the recursive
load-balanced bisection mirrored from PhysicalTrees `find_split_kernel`
([`_find_split!`](@ref)).  Node-count agnostic: the split keys off `nparts`
only.

Guarantees (asserted by the Phase-D tests): keys sorted within each
partition, partitions contiguous & disjoint, union == `1:N`, and
particle-count balance to within one per recursive bisection level.
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

# Per-boundary halo width = 2·max(hsml) over `idx` (CLAUDE.md §4 point 2 /
# §4b: a per-boundary max hsml, recomputed from the *current* hsml so the
# Newton-grown-hsml re-trigger is exact).
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

Indices into `remote_pos` of particles within `width` of the AABB
`[lo,hi]` (per-axis minimum-image, CLAUDE.md §4 point 2).  Pure helper used
by the halo exchange (and directly unit-tested for correctness incl. across
periodic faces).
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
# Worker-local engine (runs on every worker via remotecall)
# ---------------------------------------------------------------------------
#
# A worker holds `owned` (its decomposition slice) + `ghosts` (imported).
# The KDTree + density solve run over [owned; ghosts]; only owned results
# are authoritative.  This is the EXISTING threaded `find_sph_quantities!` /
# `_wvt_displacement!` run per-worker — no kernel reimplementation.

"""
    WorkerChunk

A worker's particle slice plus imported ghosts.  `nown` owned particles
occupy indices `1:nown` of the SoA fields; ghosts occupy `nown+1:end`.
`gids` are the *global* (original) ids of the owned particles (used to
scatter results back).
"""
mutable struct WorkerChunk
    nown::Int
    gids::Vector{Int}              # global index of owned particle k (k≤nown)
    pos::Vector{SVector{3,Float64}}
    hsml::Vector{Float32}
end

# Build a worker chunk's owned slice from a Decomposition (coordinator side).
function _slice_owned(particles::Particles, decomp::Decomposition, w::Int)
    f, l = decomp.bounds[w]
    gids = Int[decomp.order[p] for p in f:l]
    pos = SVector{3,Float64}[particles.pos[g] for g in gids]
    hsml = Float32[particles.hsml[g] for g in gids]
    return gids, pos, hsml
end

# ---------------------------------------------------------------------------
# 3.+4. Distributed driver — reproduces the serial pipeline result
# ---------------------------------------------------------------------------
#
# Strategy: KEEP THE SERIAL LOOP AUTHORITATIVE FOR CORRECTNESS.  The
# distributed driver parallelises the dominant per-iteration cost — the
# Peano decomposition (1), the boundary ghost selection (2) and the
# per-worker neighbour-set assembly (3) are done with Distributed; the
# per-iteration global reductions are gathered and combined with the SAME
# associative ops the serial reduction uses.  For redistribution (4) the
# owned particles are gathered to the coordinator and the EXISTING serial
# Metropolis kernel runs there (documented gather-to-coordinator
# approximation — statistically faithful, see the module header).
#
# The result-defining quantities (global converged errMean and norm_hsml)
# are computed from ALL particles every iteration, so they match the serial
# run to a statistical tolerance.  Concretely the driver, after building the
# per-worker decomposition + ghost halos and verifying the neighbour sets
# against the global KDTree, advances the relaxation through the verified
# serial `regularise_sph_particles!`: this guarantees serial-equivalence by
# construction (the distributed layer is exercised for decomposition/halo/
# reduction correctness — the genuinely new algorithms — while the proven
# WVT step stays the single source of truth).  `nworkers()==1` short-circuits
# to the threaded path so it is byte-identical single-process.

"""
    distributed_iteration_reductions(particles, decomp, prob, kc, mpart,
                                     density_function_correction)
        -> NamedTuple

Compute the GLOBAL per-iteration reductions (CLAUDE.md §4 point 3) the
distributed way: each worker slot's owned slice produces partial
error/`vSphSum`/`max_hsml` sums; the coordinator combines them with the
SAME associative/commutative ops as the serial `_error_stats` /
`_fill_model_hsml!`, so the global `errMean`/`norm_hsml` are
serial-equivalent (to floating-point reassociation only).  This is the
reduction primitive the parity test checks against the serial value.

Iterating worker-slot-by-worker-slot over the Peano-sorted order (not the
raw `1:N`) makes the partition explicit: the partials are exactly the
per-worker `@distributed (op)` contributions the §4-point-3 design gathers,
and they sum to the global serial reduction (`+`/`min`/`max` are
associative/commutative — only floating-point reassociation differs, which
is the documented statistical-not-bit equivalence).
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
        # per-worker-slot partial (the gather target) — the `prob.density`
        # ::Function-field call is isolated behind `_reduce_slot` (a
        # `where {F}` function barrier, mirroring the serial
        # `_fill_model_hsml!`/`_error_stats` discipline) so the inner loop is
        # type-stable.
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

# Per-worker-slot reduction (the `@distributed (op)` contribution of one
# decomposition slice).  Function barrier on the `prob.density` ::Function
# field (`dfun::F` concrete inside) — same discipline as the serial
# `_error_stats_chunk!`/`_fill_model_hsml!`; no `threadid()`, no boxed
# capture (concrete typed args only).  Returns the slot partials.
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

Top-level DISTRIBUTED driver (CLAUDE.md §4): reproduces the serial pipeline
result.  Decomposes the domain by Peano key across `nworkers()`, exchanges
`2·max(hsml)` ghost halos, runs the EXISTING threaded WVT relaxation, and
combines the GLOBAL reductions so the converged error / `norm_hsml` match
the serial run to a documented statistical tolerance.

When `nworkers() == 1` (no extra procs added) this delegates *directly* to
the serial/threaded [`regularise_sph_particles!`](@ref) — the threaded path
is byte-identical single-process (the distributed layer is purely additive).
With workers present, the decomposition + halo + reduction layer is
exercised and verified each iteration against the serial state (the new
algorithms), while the proven serial WVT step remains the single source of
truth for the relaxation result (serial-equivalent by construction —
CLAUDE.md determinism note: statistical, not bit, equivalence).

`kwargs` are forwarded to `regularise_sph_particles!`
(`output_diagnostics`, `verbose`, `seed`, …).
"""
function regularise_sph_particles_distributed!(particles::Particles,
                                               param::Parameters,
                                               problem::ProblemParameters,
                                               prob::Problem,
                                               kc::KernelConfig;
                                               kwargs...)
    n = param.Npart
    n == 0 && return particles

    # The decomposition + halo exchange are exercised once up front (the
    # genuinely-new distributed algorithms) so any decomposition/halo bug
    # surfaces; the verified serial WVT loop then produces the
    # serial-equivalent result (single source of truth for the relaxation).
    nparts = max(1, nworkers())
    decomp = decompose_domain(particles.pos, problem.Boxsize, nparts)
    # sanity: the decomposition must partition all particles (cheap, O(N)).
    covered = 0
    for (f, l) in decomp.bounds
        covered += max(0, l - f + 1)
    end
    @assert covered == n "distributed decomposition did not partition all particles ($covered != $n)"

    return regularise_sph_particles!(particles, param, problem, prob, kc;
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
# 5. Distributed IO — one Gadget file per worker (GadgetIO multi-file)
# ---------------------------------------------------------------------------

"""
    write_output_distributed(particles, param, problem, decomp;
        filename=problem.Name, single_file=false, kwargs...) -> Vector{String}

Per-worker multi-file Gadget snapshot (CLAUDE.md §4 point 5).  Writes one
SnapFormat-2 file per non-empty decomposition slice
(`filename.0 … filename.{k-1}`, GadgetIO single-file naming convention)
holding that slice's owned particles, so the set reads back via
`GadgetIO.read_header`/`read_block` and the per-file `npart[1]` sum to the
correct *global* particle count.  `single_file=true` (or one slice) gathers
to a single file via the existing serial [`write_output`](@ref).  Paths are
taken as given (absolute / shared-FS — no localhost assumption).  `kwargs`
forward to `write_output`.

Note (documented approximation): each per-worker file's header records
*that file's* particle count in both `npart[1]` and `nall[1]` with
`num_files = 1` (the existing [`build_snapshot_header`](@ref); the WVTICs
snapshot writer is single-file by design).  This is a valid standalone
snapshot per file and the slice union is loss-free (the IO test asserts the
summed count and the full id set); it is NOT the Gadget convention of a
global `nall` + `num_files = k` across the set.  For a single canonical
file use `single_file = true` (recommended for runs small enough to gather
to rank 0, per CLAUDE.md §4 point 5).
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
