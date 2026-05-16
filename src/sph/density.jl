# Phase 2 — SPH density: ports `sph.c::Find_sph_quantities` + `Find_hsml`.
#
# Per particle, solve for `Hsml` such that the kernel-weighted neighbour count
# `wkNgb ≈ DESNNGB` within `NNGBDEV`, producing the kernel-self-bias-corrected `Rho`,
# `Hsml`, and `VarHsmlFac = 1 / (1 + hsml/(3ρ)·dρ/dhsml)`.
#
# Control flow follows `sph.c` exactly:
#   * outer widen/narrow loop by HSML_FACTOR (1.24 3D / 1.15 2D) until
#     DESNNGB ≤ ngbcnt < NGBMAX,
#   * inner Newton–Raphson + bisection on wkNgb to within NNGBDEV,
#   * `it > 128` hard exit, `|upper-lower| < 1e-4` escape (×1.26 volume).
#
# §4b optimisations:
#   * reuse the previous `particles.hsml[i]` as the Newton seed; only fall
#     back to `2·guess_hsml` when it is 0 (matches `sph.c`'s
#     `if (Hsml==0) hsml = 2*Guess_hsml`),
#   * "query once": one periodic ball query at an upper-bound radius caches a
#     candidate list; the whole Newton iteration filters that list by distance
#     (no per-iteration tree query) — `count_within!` for the count-only
#     widen/narrow loop, a single filtered pass for the final ρ sum,
#   * branchless minimum-image (in neighbors.jl),
#   * chunk-indexed `Threads.@threads` (per-chunk preallocated buffers; NEVER
#     `threadid()`-indexed — the Phase-1 bug pattern).

using StaticArrays

const _HSML_FACTOR_3D = 1.24
const _HSML_FACTOR_2D = 1.15
const _SQRT3 = 1.7320508075688772   # sqrt(3); C `sqrt3` 1.73205080756887719

@inline _hsml_factor(dim::Int) = dim == 2 ? _HSML_FACTOR_2D : _HSML_FACTOR_3D
# NOTE: ALWAYS returns `Float64` (never `Irrational{:π}`). `voln` is threaded
# through `::Float64`-annotated function barriers (`_fill_model_hsml!`,
# `_wvt_displacement!`, `_count_moves`, `_move_particles!`,
# `_autocalibrate_step`); a bare `pi` for `dim==2` is `Irrational{:π}` and
# would fail dispatch on those signatures (the genuine pre-existing 2D-path
# MethodError). `Float64(pi)` keeps the helper type-stable on both paths.
@inline _vol_norm(dim::Int) = dim == 2 ? Float64(pi) : (4.0 * pi / 3.0)   # 4π/3

"""
    NgbScratch

Per-chunk preallocated scratch for the per-particle solve (one instance per
parallel chunk — indexed by the chunk loop variable, never by `threadid()`).

* `cands`  : cached candidate index list from one periodic ball query
             (the Newton solve filters THIS list by distance — §4b "query
             once, iterate on a cached candidate list"; no per-iteration tree
             query and no per-iteration allocation)
* `qtmp`   : per-image query scratch for `query_candidates!` (periodic path)
"""
struct NgbScratch
    cands::Vector{Int}
    qtmp::Vector{Int}
end
NgbScratch() = NgbScratch(Int[], Int[])

# ---------------------------------------------------------------------------
# Inner Newton/bisection refinement — ports `sph.c::Find_hsml`
# ---------------------------------------------------------------------------
#
# Operates on the cached candidate list `cands` (indices). Returns
# `(part_done::Bool, hsml, rho, dRhodHsml)`.  `rho` already has the SPHKernels
# kernel self-bias correction applied when `part_done` (see below).
#
# Function barrier: `kc`, `mpart`, `box`, `periodic` are concrete; the
# candidate list is a plain `Vector{Int}`; no captured Function fields.
function _find_hsml(positions::Vector{SVector{3,Float64}}, ipart::Int,
                    cands::Vector{Int}, hsml0::Float64,
                    kc::KernelConfig, mpart::Float64,
                    box::NTuple{3,Float64}, periodic::NTuple{3,Bool})

    desnngb = Float64(kc.desnngb)
    nngbdev = kc.nngbdev
    dim = kc.dim
    voln = _vol_norm(dim)

    upper = hsml0 * _SQRT3
    lower = 0.0
    hsml = hsml0
    rho = 0.0
    dRhodHsml = 0.0
    it = 0
    part_done = false

    pos_i = @inbounds positions[ipart]

    @inbounds while true
        wkNgb = 0.0
        rho = 0.0
        dRhodHsml = 0.0
        it += 1

        h_inv = 1.0 / hsml
        h2 = hsml * hsml

        for t in eachindex(cands)
            jpart = cands[t]
            r2 = periodic_dist2(pos_i, positions[jpart], box, periodic)
            r2 > h2 && continue                       # C: if (r2 > p2(hsml)) continue
            r = sqrt(r2)
            wk = sph_kernel(kc, r, h_inv)
            dwk = sph_kernel_deriv(kc, r, h_inv)
            if dim == 2
                wkNgb += pi * wk * h2
            else
                wkNgb += voln * wk * (h2 * hsml)       # 4π/3 · wk · h³
            end
            rho += mpart * wk
            dRhodHsml += -mpart * (3.0 / hsml * wk + r / hsml * dwk)
        end

        if it > 128                                    # hard exit, not enough ngb
            break
        end

        ngbDev = abs(wkNgb - desnngb)
        if ngbDev < nngbdev
            part_done = true
            break
        end

        if abs(upper - lower) < 1e-4                   # find more neighbours!
            hsml *= 1.26                               # double the volume
            break
        end

        if ngbDev < 0.5 * desnngb                      # Newton–Raphson
            omega = 1.0 + dRhodHsml * hsml / (3.0 * rho)
            fac = 1.0 - (wkNgb - desnngb) / (3.0 * wkNgb * omega)
            fac = min(_hsml_factor(dim), fac)          # handle overshoot
            fac = max(1.0 / _hsml_factor(dim), fac)
            hsml *= fac
        else                                           # bisection
            if wkNgb > desnngb
                upper = hsml
            end
            if wkNgb < desnngb
                lower = hsml
            end
            hsml = cbrt(0.5 * (lower * lower * lower + upper * upper * upper))
        end
    end

    if part_done
        # Kernel self-bias correction (Dehnen&Aly / Cullen&Dehnen). Routed
        # through `SPHKernels.bias_correction` (user-directed divergence from
        # the C `kernel.c::bias_correction_WC*` formula — see PORT_STATUS.md
        # "Phase-2 follow-up: density-path kernel self-bias → SPHKernels").
        # NOTE: `SPHKernels.bias_correction` returns the *already corrected
        # density* `ρ − δρ` (NOT a Δ), so we assign rather than add. Applied
        # only when the hsml solve converged (`part_done`), matching where the
        # C `Find_hsml` applied its correction. `1/hsml` is the same `h_inv`;
        # `kc.desnngb` is the desired neighbour count (DESNNGB).
        rho = sph_bias_correction(kc, rho, mpart, 1.0 / hsml)
    end

    return part_done, hsml, rho, dRhodHsml
end

# ---------------------------------------------------------------------------
# Per-particle driver — ports the `sph.c::Find_sph_quantities` body
# ---------------------------------------------------------------------------
#
# Mirrors the C outer loop: widen/narrow `hsml` by HSML_FACTOR using the
# (cached-candidate) neighbour count until DESNNGB ≤ ngbcnt < NGBMAX, then run
# `_find_hsml`.  The §4b "query once" pattern: a single periodic ball query at
# an upper-bound radius caches the candidate list; the widen/narrow loop and
# the Newton solve both filter that cached list by distance (re-querying only
# when the working radius would exceed the cached query radius).
function _solve_particle!(particles::Particles,
                          positions::Vector{SVector{3,Float64}},
                          tree::KDTree, ipart::Int, scratch::NgbScratch,
                          kc::KernelConfig, mpart::Float64,
                          box::NTuple{3,Float64}, periodic::NTuple{3,Bool})

    hsml = Float64(particles.hsml[ipart])
    if hsml == 0.0
        # §4b: only pay for the knn-based guess when there is no usable seed
        # (matches `sph.c`'s `if (Hsml==0) hsml = 2*Guess_hsml`).
        hg = guess_hsml(tree, positions, ipart, kc.desnngb,
                        box, periodic; dim = kc.dim)
        hsml = 2.0 * hg                                # C: 2*Guess_hsml (always too large)
    end
    isfinite(hsml) || error("hsml not finite ipart=$ipart")

    ngbmax = kc.ngbmax
    hf = _hsml_factor(kc.dim)
    pos_i = @inbounds positions[ipart]

    # One ball query at an upper bound; re-query only if the working radius
    # grows past it (rare — the widen loop multiplies by ≤1.26 per step).
    query_r = hsml * 1.05
    query_candidates!(scratch.cands, scratch.qtmp, tree, positions, pos_i,
                      query_r, box, periodic)

    # Safeguard: the C outer loop is unbounded; in practice it converges in a
    # handful of widen/narrow passes. Cap passes to avoid a pathological hang
    # in degenerate inputs and still write finite, best-effort quantities
    # (this only triggers when the C code would also fail to find DESNNGB).
    outer_it = 0
    while true
        outer_it += 1
        if outer_it > 4096
            part_done, hsml, rho, dRhodHsml =
                _find_hsml(positions, ipart, scratch.cands, hsml,
                           kc, mpart, box, periodic)
            vf = 1.0 / (1.0 + hsml / (3.0 * max(rho, eps())) * dRhodHsml)
            @inbounds particles.hsml[ipart] = Float32(hsml)
            @inbounds particles.rho[ipart] = Float32(rho)
            @inbounds particles.varhsmlfac[ipart] = Float32(vf)
            return
        end
        if hsml > query_r
            query_r = hsml * 1.05
            query_candidates!(scratch.cands, scratch.qtmp, tree, positions,
                              pos_i, query_r, box, periodic)
        end

        h2 = hsml * hsml
        ngbcnt = count_within!(scratch.cands, positions, pos_i, h2,
                               box, periodic)

        if ngbcnt >= ngbmax                            # prevent ngblist overflow
            hsml /= hf
            continue
        end
        if ngbcnt < kc.desnngb
            hsml *= hf
            continue
        end

        part_done, hsml, rho, dRhodHsml =
            _find_hsml(positions, ipart, scratch.cands, hsml,
                       kc, mpart, box, periodic)

        if part_done
            varHsmlFac = 1.0 / (1.0 + hsml / (3.0 * rho) * dRhodHsml)
            @inbounds particles.hsml[ipart] = Float32(hsml)
            @inbounds particles.rho[ipart] = Float32(rho)
            @inbounds particles.varhsmlfac[ipart] = Float32(varHsmlFac)
            return
        end

        # not done: C re-widens (the `ngbcnt < DESNNGB && !part_done` branch
        # only triggers on too-few-ngb; `_find_hsml` returning not-done after a
        # `<1e-4` escape already scaled hsml ×1.26 — loop and re-query/refine).
        if hsml > query_r
            query_r = hsml * 1.05
            query_candidates!(scratch.cands, scratch.qtmp, tree, positions,
                              pos_i, query_r, box, periodic)
        end
    end
end

# ---------------------------------------------------------------------------
# Public entry — ports `sph.c::Find_sph_quantities`
# ---------------------------------------------------------------------------

"""
    find_sph_quantities!(particles, param, problem, prob, kc;
                          tree = nothing, leafsize = 32) -> KDTree

Compute per-particle `hsml`, kernel-self-bias-corrected `rho`, and `varhsmlfac` by solving
the SPH continuity equation for the kernel-weighted neighbour count
(`sph.c::Find_sph_quantities`).

* `particles` — `Particles` SoA; `hsml`/`rho`/`varhsmlfac` are written.
* `param::Parameters` — used for `density_function_correction` (the artificial
  density-model correction; the *kernel self-bias* correction is internal and
  from `kc`, via `SPHKernels.bias_correction`).
* `problem::ProblemParameters` — `Mpart`, `Boxsize`, `Periodic`.
* `prob::Problem` — problem callbacks;
  `prob.density(particles,i,density_function_correction)` is used to set
  `rho_model` (`density_function_correction = param.density_function_correction`,
  per the Phase-1 handoff).
* `kc::KernelConfig` — kernel + DESNNGB/NNGBDEV/NGBMAX/dim.
* `tree` — optional prebuilt `KDTree` (Verlet/skin reuse, §4b).  If `nothing`
  a fresh tree is built (matches the C per-iteration rebuild).

Returns the `KDTree` used (so a caller can cache/reuse it, §4b skin-list).

Reuses each `particles.hsml[i]` as the Newton seed when non-zero (§4b);
parallelised with the verified chunk-indexed `Threads.@threads` pattern
(per-chunk `NgbScratch`; never indexed by `threadid()`).
"""
function find_sph_quantities!(particles::Particles, param::Parameters,
                              problem::ProblemParameters, prob::Problem,
                              kc::KernelConfig;
                              tree::Union{Nothing,KDTree} = nothing,
                              leafsize::Int = 32)

    n = length(particles)
    n == 0 && return build_tree(particles.pos; leafsize = leafsize)

    positions = particles.pos
    kdtree = tree === nothing ? build_tree(positions; leafsize = leafsize) :
                                tree

    mpart = Float64(problem.Mpart)
    box = problem.Boxsize
    periodic = problem.Periodic
    density_function_correction = param.density_function_correction

    # Model density per particle (diagnostic / WVT target; uses the artificial
    # density-model correction param).
    _fill_rho_model!(particles, prob, n, density_function_correction)

    nchunks = max(1, Threads.nthreads())
    chunks = _chunk_ranges(n, nchunks)
    nc = length(chunks)
    scratch = [NgbScratch() for _ in 1:nc]

    # Swappable parallel driver (see src/parallel/threads.jl). The `do c`
    # body is a pure function-barrier forward to `_density_chunk!` — no boxed
    # capture; each chunk owns `scratch[c]` and disjoint particle indices, so
    # results are identical across all threading backends.
    #
    # Per-site override to :base (dynamic scheduling). The per-particle
    # Newton/bisection hsml solve has highly variable cost (iterations differ
    # per particle), so Base.Threads' dynamic schedule load-balances it
    # markedly better than Polyester's static `@batch` (benchmarked ~1.3-1.6x
    # faster here at N=2e4 and 1e5). Every other hot loop uses the global
    # :polyester default; this is the one work-imbalanced kernel where :base
    # wins and it dominates per-iteration cost at scale.
    _run_chunks(nc, Val(:base)) do c
        _density_chunk!(c, chunks, particles, positions, kdtree, scratch,
                        kc, mpart, box, periodic)
    end

    return kdtree
end

# Per-chunk function barrier for `find_sph_quantities!`. Concrete,
# type-annotated args only (no boxed capture). `scratch[c]` (the per-chunk
# `NgbScratch`, NEVER threadid-indexed) is owned by chunk `c`; the body is the
# verified `_solve_particle!` inner loop, unchanged.
@noinline function _density_chunk!(c::Int, chunks::Vector{UnitRange{Int}},
                                   particles::Particles,
                                   positions::Vector{SVector{3,Float64}},
                                   kdtree::KDTree,
                                   scratch::Vector{NgbScratch},
                                   kc::KernelConfig, mpart::Float64,
                                   box::NTuple{3,Float64},
                                   periodic::NTuple{3,Bool})
    sc = scratch[c]
    @inbounds for ipart in chunks[c]
        _solve_particle!(particles, positions, kdtree, ipart, sc,
                         kc, mpart, box, periodic)
    end
    return nothing
end

# Function barrier: isolate the `prob.density` ::Function-field call so the
# loop stays type-stable (mirrors the Phase-1 `_mpart_slice` pattern).
function _fill_rho_model!(particles::Particles, prob::Problem, n::Int,
                          density_function_correction)
    dfun = prob.density
    @inbounds for i in 1:n
        particles.rho_model[i] =
            Float32(dfun(particles, i, density_function_correction))
    end
    return nothing
end

# Driver-convenience method: re-derive `prob` and use the default kernel
# config so the Phase-0/1 `main` pipeline can call this without error once
# Phase 3 wires it in.  The relax stub stays a no-op (Phase 3 owns the loop).
function find_sph_quantities!(particles::Particles, param::Parameters,
                              problem::ProblemParameters;
                              kc::KernelConfig = default_kernel_config())
    prob = setup_problem(param)
    find_sph_quantities!(particles, param, problem, prob, kc)
    return nothing
end
