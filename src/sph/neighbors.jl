# Phase 2 — neighbour search (ports `tree.c`'s ngb role; replaces the linked
# Peano octree with `NearestNeighbors.jl` per CLAUDE.md §1.5 / §4).
#
# A plain `KDTree` is built over the positions (`Vector{SVector{3,Float64}}`,
# `reorder=true`).  Periodicity is handled by **manual minimum-image
# correction** after a plain-tree query — NOT `PeriodicTree`, whose dedup
# buffer is mutable and not thread-safe (CLAUDE.md §4).  We query with a guard
# radius (the candidate's true min-image distance can be smaller than its
# Euclidean distance, but never larger when we add a per-axis box-image probe),
# then re-test every candidate with the per-axis branchless min-image distance
# gated by `Problem.Periodic`, exactly as `tree.c` corrects distances.
#
# §4b optimisations implemented here:
#   * branchless min-image  d -= L·round(d/L)  (no C `while` loop)
#   * `inrangecount`        for allocation-free count-only Newton iterations
#   * `inrange!`            into a caller-preallocated buffer for the final list
#   * `query_candidates!`   "query once, iterate on a cached candidate list":
#                           one ball query at an upper-bound radius, then the
#                           Newton solve filters the cached candidates by
#                           distance without re-querying the tree.

using NearestNeighbors
using StaticArrays

# ---------------------------------------------------------------------------
# Branchless periodic minimum image
# ---------------------------------------------------------------------------

"""
    minimum_image(d, L, periodic) -> Float64

Branchless minimum-image displacement along one axis (CLAUDE.md §4b):
`d -= L·round(d/L)` when `periodic`, identity otherwise.  Replaces the C
`while (d > boxhalf) d -= boxsize` loop in `sph.c` / `tree.c`.
"""
@inline function minimum_image(d::Float64, L::Float64, periodic::Bool)
    return periodic ? d - L * round(d / L) : d
end

"""
    periodic_dist2(a, b, box, periodic) -> Float64

Squared distance between points `a` and `b` (`SVector{3,Float64}`) under
per-axis minimum-image, `box::NTuple{3,Float64}`,
`periodic::NTuple{3,Bool}`.  Matches the `r2` accumulation in
`sph.c::Find_hsml` / `tree.c::Find_ngb_tree`.
"""
@inline function periodic_dist2(a::SVector{3,Float64}, b::SVector{3,Float64},
                                box::NTuple{3,Float64},
                                periodic::NTuple{3,Bool})
    dx = minimum_image(a[1] - b[1], box[1], periodic[1])
    dy = minimum_image(a[2] - b[2], box[2], periodic[2])
    dz = minimum_image(a[3] - b[3], box[3], periodic[3])
    return dx * dx + dy * dy + dz * dz
end

# ---------------------------------------------------------------------------
# KDTree build
# ---------------------------------------------------------------------------

"""
    build_tree(positions; leafsize = 32) -> KDTree

Build a `KDTree` over `positions::Vector{SVector{3,Float64}}` with
`reorder=true` (cache-locality; the tree subsumes the C Peano sort, CLAUDE.md
§1.5).  Immutable and safe for concurrent read queries from multiple threads.
"""
function build_tree(positions::AbstractVector{SVector{3,Float64}};
                    leafsize::Int = 32)
    return KDTree(positions, Euclidean(); leafsize = leafsize, reorder = true)
end

# Largest box extent over the periodic axes — used to size the query guard
# radius so a wrapped image cannot be missed.
@inline function _max_periodic_box(box::NTuple{3,Float64},
                                   periodic::NTuple{3,Bool})
    m = 0.0
    periodic[1] && (m = max(m, box[1]))
    periodic[2] && (m = max(m, box[2]))
    periodic[3] && (m = max(m, box[3]))
    return m
end

"""
    guess_hsml(tree, positions, ipart, desnngb, box, periodic; dim=3) -> Float64

Initial smoothing-length estimate replacing C `tree.c::Guess_hsml` (CLAUDE.md
§1.5).  Uses a `knn` query for the `min(desnngb, N-1)+1`-th nearest neighbour
distance and inflates it to the radius that, at the local number density,
encloses ≈`DESNNGB` particles:

    n_local ≈ k / (4π/3 · r_k³)     (3D)        n_local ≈ k / (π · r_k²)  (2D)
    hsml    = ( (4π/3·DESNNGB) / n_local·… )^(1/dim)  ⇒  r_k·(DESNNGB/k)^(1/dim)

i.e. `r_k · (DESNNGB / k)^(1/dim)`.  Robust on glasses and clustered data;
matches the spirit of the C parent-node number-density guess.  Periodic edge
particles get a slightly small `r_k` from the plain tree — acceptable because
`find_hsml`'s outer widen loop corrects any under/over-estimate.
"""
function guess_hsml(tree::KDTree, positions::AbstractVector{SVector{3,Float64}},
                    ipart::Int, desnngb::Int,
                    box::NTuple{3,Float64}, periodic::NTuple{3,Bool};
                    dim::Int = 3)
    n = length(positions)
    n <= 1 && return _max_periodic_box(box, periodic) > 0 ?
                     0.1 * _max_periodic_box(box, periodic) : 1.0
    k = min(desnngb, n - 1) + 1            # include self
    idxs, dists = knn(tree, positions[ipart], k, true)
    r_k = isempty(dists) ? 0.0 : Float64(last(dists))
    if r_k <= 0.0
        # Degenerate (coincident points): fall back to a box-based guess.
        L = _max_periodic_box(box, periodic)
        return L > 0 ? L * (desnngb / max(1, n))^(1.0 / dim) : 1.0
    end
    keff = max(1, k - 1)                   # neighbours excluding self
    return r_k * (desnngb / keff)^(1.0 / dim)
end

# ---------------------------------------------------------------------------
# Periodic ball queries (manual min-image on a plain tree)
# ---------------------------------------------------------------------------

"""
    query_candidates!(buf, tmp, tree, positions, point, radius, box, periodic) -> buf

Fill `buf` (preallocated, will be `empty!`-ed) with **candidate** indices for a
periodic fixed-radius query: all tree points within `radius` of `point` OR of
any of `point`'s box images along the periodic axes.  The candidate set is a
superset of the true minimum-image neighbour set within `radius`; the caller
re-filters by `periodic_dist2`.  This is the "query once" half of the §4b
cached-candidate pattern — call it ONCE at an upper-bound radius, then iterate
the Newton solve over the returned list.

Non-periodic axes use a single query; periodic axes additionally query the
`±box` shifted points so a neighbour wrapped across a face is captured.  For
the WVT regime (`radius ≪ box`) only the home cell + immediate face images
matter, so we probe the 3^(#periodic-axes) nearest images but prune images
whose shift exceeds the box (cannot contribute).
"""
function query_candidates!(buf::Vector{Int}, tmp::Vector{Int}, tree::KDTree,
                           positions::AbstractVector{SVector{3,Float64}},
                           point::SVector{3,Float64}, radius::Float64,
                           box::NTuple{3,Float64}, periodic::NTuple{3,Bool})
    empty!(buf)
    np = periodic[1] || periodic[2] || periodic[3]
    if !np
        inrange!(buf, tree, point, radius)
        return buf
    end
    # Per-axis offsets to probe.  Always 0.  On a periodic axis, the wrapped
    # neighbour across the *upper* face is reachable only if the query ball
    # crosses x=L  (point[a] + radius > L)  ⇒ probe a +L-shifted query; the
    # *lower* face only if  point[a] - radius < 0  ⇒ probe a -L-shifted query.
    # This pruning is EXACT for radius < box (the WVT/density regime): a
    # neighbour whose minimum image lies within `radius` is found by exactly
    # one of these shifted queries.  Interior particles ⇒ a single query.
    sx = _axis_offsets(point[1], radius, box[1], periodic[1])
    sy = _axis_offsets(point[2], radius, box[2], periodic[2])
    sz = _axis_offsets(point[3], radius, box[3], periodic[3])
    nq = 0
    for ox in sx
        ox === nothing && continue
        for oy in sy
            oy === nothing && continue
            for oz in sz
                oz === nothing && continue
                q = SVector{3,Float64}(point[1] + ox, point[2] + oy,
                                       point[3] + oz)
                empty!(tmp)
                inrange!(tmp, tree, q, radius)
                append!(buf, tmp)
                nq += 1
            end
        end
    end
    # Deduplicate only if more than one query ran (a particle can match
    # through several images when radius ≳ box; rare in the WVT regime).
    if nq > 1 && length(buf) > 1
        sort!(buf)
        unique!(buf)
    end
    return buf
end

# Returns a length-3 tuple of {Float64, nothing}: the offsets to probe on one
# axis.  Index 1 is always 0.0; indices 2/3 are ±L or `nothing` (skip).
@inline function _axis_offsets(x::Float64, radius::Float64, L::Float64,
                               periodic::Bool)
    if !periodic
        return (0.0, nothing, nothing)
    end
    lo = (x - radius < 0.0)   ? L  : nothing      # neighbour wrapped from top
    hi = (x + radius > L)     ? -L : nothing      # neighbour wrapped from bottom
    return (0.0, lo, hi)
end

"""
    periodic_inrangecount(tree, point, radius, periodic) -> Int

Allocation-free neighbour count.  Non-periodic ⇒ a direct
`NearestNeighbors.inrangecount` (the §4b allocation-free count path).
Periodic ⇒ returns `-1`: a periodic count must re-test the per-axis
minimum-image distance, which the density solver does allocation-free via
[`count_within!`](@ref) over the cached candidate list.
"""
@inline function periodic_inrangecount(tree::KDTree,
                                        point::SVector{3,Float64},
                                        radius::Float64,
                                        periodic::NTuple{3,Bool})
    if !(periodic[1] || periodic[2] || periodic[3])
        return inrangecount(tree, point, radius)
    end
    # Periodic counts must go through the candidate path (min-image re-test);
    # callers in density.jl use `count_within!` over a cached candidate list.
    return -1
end

"""
    count_within!(cands, positions, point, r2max, box, periodic) -> Int

Count cached candidate indices whose minimum-image squared distance to `point`
is `< r2max`.  Allocation-free; used for the count-only Newton iterations
(§4b) over the cached candidate list from [`query_candidates!`](@ref).
"""
@inline function count_within!(cands::Vector{Int},
                               positions::AbstractVector{SVector{3,Float64}},
                               point::SVector{3,Float64}, r2max::Float64,
                               box::NTuple{3,Float64},
                               periodic::NTuple{3,Bool})
    c = 0
    @inbounds for t in eachindex(cands)
        j = cands[t]
        if periodic_dist2(point, positions[j], box, periodic) < r2max
            c += 1
        end
    end
    return c
end
