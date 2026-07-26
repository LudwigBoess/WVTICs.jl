# Phase 3 — Regularise_sph_particles, the core WVT relaxation loop.
# Faithful port of `wvt_relax.c::Regularise_sph_particles` (CLAUDE.md §1.4).
#
# Per iteration (exact C order, see wvt_relax.c lines 71–317):
#   1. find_sph_quantities!  (reuse the Phase-2 solve + returned KDTree;
#      hsml is preserved between iterations as the Newton seed, §4b)
#   2. it++ > Maxiter → break
#   3. (SAVE_WVT_STEPS off by default) writeStepFile
#   4. reset_redistribution_flags!; if it ≤ LastMoveStep && it % RedistFreq == 0:
#        moveFraction = MoveFractionMax · exp(-decay·(it/RedistFreq - 1)),
#        decay = log(MoveFractionMax/MoveFractionMin)/(LastMoveStep/RedistFreq - 1),
#        movePart   = Npart·moveFraction,
#        maxProbes  = Npart·ProbesFraction·moveFraction/MoveFractionMax,
#        redistribute_particles!; find_sph_quantities! again
#   5. relative-density-error stats (min/max/mean/sigma, errDiff)
#   6. model-density metric  hsml[i] = (WVTNNGB·Mpart/ρ / (4π/3))^(1/3)
#      (2D (·/π)^(1/2)); vSphSum += hsml³ (2D hsml²); max_hsml;
#      norm_hsml = (WVTNNGB/vSphSum/(4π/3))^(1/3)·median_boxsize; hsml *= norm
#   7. WVT repulsive displacement:
#        delta += step·h·wk·d̂,  h = 0.5(hsml_i+hsml_j),
#        wk = sph_kernel(r,h)·h³ (2D ·h²), skip self and r²>h²,
#        Assert r²>0 (two coincident particles → C error message)
#   8. move + per-axis box wrap; moveMps[k] = % with |δ| > {1,.1,.01,.001}·d_mps,
#      d_mps = (4π/3·Hsml³/DESNNGB)^(1/3) (2D (π·Hsml²/DESNNGB)^(1/3))
#   9. it==1 low/high-movement warnings
#  10. (OUTPUT_DIAGNOSTICS on by default) diagnostics.log row
#  11. converge: break if any moveMps[k] < LimitMps[k];
#      step *= StepReduction if cnt1 > last_cnt and NOT a redistribution step
#
# §4b optimisations applied (all kept correct, see PORT_STATUS.md):
#   * Verlet / skin neighbour list: the per-particle candidate index list is
#     cached at radius `query_r = max_hsml·1.05 + r_skin` and REUSED across
#     iterations for BOTH the density solve (via find_sph_quantities!'s tree
#     reuse) and the displacement loop; the KDTree + candidate lists are
#     rebuilt only when the max accumulated displacement since the last build
#     exceeds 0.5·r_skin (the Verlet criterion).  `r_skin` = a fraction of the
#     mean-particle-spacing-scaled hsml (see `_skin_radius`).
#   * Hsml-seed reuse: hsml is NOT zeroed between iterations (find_sph!
#     reuses it as the Newton seed → ~1–2 inner iterations).
#   * Branchless minimum-image (periodic_dist2 / minimum_image, Phase 2).
#   * Chunk-indexed Threads.@threads (per-chunk scratch indexed by the loop
#     variable c — NEVER threadid(); the Phase-1 anti-pattern).

using StaticArrays

# ALWAYS returns `Float64` (never `Irrational{:π}`) — see the `_vol_norm`
# note in src/sph/density.jl. `voln` is passed as a concrete `::Float64`
# argument into `_fill_model_hsml!` / `_wvt_displacement!` / `_count_moves`
# / `_move_particles!` / `_autocalibrate_step`; a bare `pi` (dim==2) is
# `Irrational{:π}` ⇒ MethodError on those `::Float64` signatures (the 2D
# auto-path bug). `Float64(pi)` makes the 2D model-hsml/displacement path
# type-stable end to end.
@inline _wvt_vol_norm(dim::Int) = dim == 2 ? Float64(pi) : (4.0 * pi / 3.0)

# C `Find_sph_quantities` rebuilds the octree every iteration.  We instead use
# a Verlet skin: the cached candidate lists (and the KDTree) stay valid while
# every particle has moved < 0.5·r_skin since the last build.  r_skin is set
# to a fraction of the *model* hsml scale (≈ mean particle spacing), which is
# the radius the displacement loop and the density solve actually probe.
const _WVT_SKIN_FRACTION = 0.5    # r_skin = 0.5 · mean(model hsml)

@inline _skin_radius(mean_hsml::Float64) = _WVT_SKIN_FRACTION * mean_hsml

# --- MpsFraction auto-calibration constants --------------------------------
# (MPSFRACTION_ANALYSIS.md §3 / §4.3 — algorithm constants, NOT user
# parameters.) `MpsFraction <= 0.0` (omitted / 0.0 / the parser's "auto"
# sentinel) ⇒ the hybrid analytic-seed + bisection safety net runs; any
# finite positive `MpsFraction` ⇒ exact legacy behaviour, byte-identical.
#
# Analytic seed (Option A): the constant-density README geometric-centre
# `MpsFraction` value, dimension-aware (≈0.8 in 3D, ≈0.1 in 2D — the value
# parameters.toml already hinted). `step` is then formed from it exactly as
# the legacy formula does (`1/(npart_1D·MpsFraction)`).
const MPS_AUTO_SEED_3D = 0.8
const MPS_AUTO_SEED_2D = 0.1
# Target / accept band for moveMps[0] (% of particles with |δ| > d_mps after
# the first displacement pass): the band the C `it==1` warnings encode
# (wvt_relax.c:281–289 — <10% "decrease MpsFraction", >80% "increase"). We
# bisect `step` toward the band centre so StepReduction has the most room.
const MPS_AUTO_BAND_LO = 10.0
const MPS_AUTO_BAND_HI = 80.0
const MPS_AUTO_TARGET = 40.0
# Total trial budget (1 seed + bracket + bisection); analysis §3.3 worst case.
const MPS_AUTO_MAX_TRIALS = 12

@inline _is_auto_mps(mpsfraction::Real) =
    !(isfinite(mpsfraction) && mpsfraction > 0.0)

# Analytic-seed `step` for the auto path (legacy formula with the seed
# MpsFraction, dimension-aware npart_1D — identical algebra to the C
# `step = 1/(npart_1D·MpsFraction)`).
@inline function _analytic_seed_step(n::Int, dim::Int)
    npart_1d = dim == 2 ? n^(1.0 / 2.0) : n^(1.0 / 3.0)
    seed_mps = dim == 2 ? MPS_AUTO_SEED_2D : MPS_AUTO_SEED_3D
    return 1.0 / (npart_1d * seed_mps)
end

# Per-chunk scratch for the displacement loop: a cached candidate list per
# particle is held in the shared `cand_lists`; `qtmp` is the per-image query
# scratch for `query_candidates!`.  Indexed by the chunk loop variable.
struct WvtScratch
    qtmp::Vector{Int}
end
WvtScratch() = WvtScratch(Int[])

"""
    regularise_sph_particles!(particles, param, problem, prob, kc;
        save_wvt_steps = false, output_diagnostics = true,
        diagnostics_path = "diagnostics.log", seed = RNG_BASE_SEED) -> particles

Faithful port of `wvt_relax.c::Regularise_sph_particles` (CLAUDE.md §1.4).
Relaxes `particles.pos` toward the model density field via the WVT repulsive
scheme, with periodic Metropolis redistribution.  Writes
`particles.hsml/rho/varhsmlfac/rho_model/pos`.

* `prob::Problem` — problem callbacks (`prob.density` is the model density).
* `kc::KernelConfig` — kernel + DESNNGB/NNGBDEV/NGBMAX/dim (WVTNNGB = DESNNGB).
* `save_wvt_steps` — C `SAVE_WVT_STEPS`, default **off** (writes a snapshot
  per iteration via `write_step_file`).
* `output_diagnostics` — C `OUTPUT_DIAGNOSTICS`, default **on** (writes
  `diagnostics.log`).

Returns `particles` (mutated in place).
"""
function regularise_sph_particles!(particles::Particles, param::Parameters,
                                   problem::ProblemParameters, prob::Problem,
                                   kc::KernelConfig;
                                   save_wvt_steps::Bool = false,
                                   output_diagnostics::Bool = true,
                                   diagnostics_path::AbstractString = "diagnostics.log",
                                   seed::Integer = RNG_BASE_SEED,
                                   verbose::Bool = true)

    n = param.Npart
    n == 0 && return particles

    box = problem.Boxsize
    boxv = (box[1], box[2], box[3])
    periodic = problem.Periodic
    dim = kc.dim
    voln = _wvt_vol_norm(dim)
    desnngb = Float64(kc.desnngb)
    mpart = Float64(problem.Mpart)
    wvtnngb = desnngb                       # C: #define WVTNNGB DESNNGB

    # C: median_boxsize = fmax(Boxsize[1], Boxsize[2]) (0-based) → here the
    # 1-based equivalent over the two non-largest axes (Boxsize[1] is largest).
    median_boxsize = max(box[2], box[3])

    # `MpsFraction <= 0.0` (omitted / 0.0 / the parser's "auto" sentinel) ⇒
    # the hybrid analytic-seed + bisection auto-calibration runs (below, once
    # the scratch arrays exist). A finite POSITIVE value ⇒ exact legacy
    # behaviour: step = 1/(npart_1D·MpsFraction), npart_1D = N^(1/3) (2D
    # N^(1/2)) — byte-identical to the pre-substep formula / the C code.
    auto_mps = _is_auto_mps(param.MpsFraction)
    npart_1d = dim == 2 ? n^(1.0 / 2.0) : n^(1.0 / 3.0)
    step = auto_mps ? 0.0 : 1.0 / (npart_1d * param.MpsFraction)

    errLast = floatmax(Float64)
    errDiff = floatmax(Float64)
    last_cnt = floatmax(Float64)

    if verbose
        println("Starting iterative SPH regularisation")
        println("   Maxiter=", param.Maxiter, ", MpsFraction=",
                (auto_mps ? "auto" : string(param.MpsFraction)),
                " StepReduction=", param.StepReduction,
                " LimitMps=(", param.LimitMps[1], ",", param.LimitMps[2], ",",
                param.LimitMps[3], ",", param.LimitMps[4], ")")
    end

    # model-density metric hsml (C `hsml[]`, distinct from SphP.Hsml) and the
    # displacement accumulator (C `delta[3][]`, kept Float32 to match
    # diagnostics.c::calculateStatsOn precision).
    mhsml = zeros(Float64, n)
    dx = zeros(Float32, n)
    dy = zeros(Float32, n)
    dz = zeros(Float32, n)
    deltas = (dx, dy, dz)

    # Verlet-skin state: cached per-particle candidate index lists, the radius
    # they were queried at, and the reference positions at the last (re)build.
    cand_lists = [Int[] for _ in 1:n]
    ref_pos = copy(particles.pos)
    r_skin = 0.0
    tree::Union{Nothing,KDTree} = nothing
    have_lists = false

    nchunks = max(1, Threads.nthreads())
    chunks = _chunk_ranges(n, nchunks)
    nc = length(chunks)
    wscratch = [WvtScratch() for _ in 1:nc]

    # --- MpsFraction startup auto-calibration (MPSFRACTION_ANALYSIS.md §3) -
    # Only when MpsFraction was omitted/0.0/"auto". Reuses the step-
    # independent tree/density solve/model-hsml/candidate lists ONCE and
    # bisects `step` so the first-iteration moveMps[0] lands in the C
    # author's band. Does NOT mutate particle positions and does NOT advance
    # the relaxation; the main loop below then proceeds completely unchanged
    # (it rebuilds its own it==1 tree/solve from the same untouched
    # positions). The transient mhsml/cand_lists scratch written here is
    # fully overwritten by the loop's first iteration before use.
    if auto_mps
        step = _autocalibrate_step(particles, param, problem, prob, kc,
                                   mhsml, deltas, cand_lists, wscratch,
                                   chunks, n, boxv, periodic, dim, voln,
                                   desnngb, wvtnngb, mpart, median_boxsize;
                                   verbose = verbose)
        # zero the scratch the calibration touched so the main loop starts
        # from the same clean state as the legacy path (cand_lists/mhsml are
        # recomputed at it==1; deltas is overwritten each displacement pass —
        # zero it defensively so no stale calibration displacement leaks).
        fill!(dx, 0.0f0)
        fill!(dy, 0.0f0)
        fill!(dz, 0.0f0)
        for cl in cand_lists
            empty!(cl)
        end
        fill!(mhsml, 0.0)
        copyto!(ref_pos, particles.pos)
        if !(isfinite(step) && step > 0.0)      # final safety net
            step = _analytic_seed_step(n, dim)
        end
    end

    if output_diagnostics
        init_iteration_diagnostics(diagnostics_path)
    end

    it = 0
    while true

        # --- 0. Verlet rebuild gate (BEFORE the SPH solve) ----------------
        # The C rebuilds its spatial index at the start of *every* iteration
        # (inside Find_sph_quantities).  We keep a Verlet skin instead: the
        # KDTree is rebuilt only when the max accumulated min-image
        # displacement since the last build exceeds 0.5·r_skin (or on the
        # first iteration, or right after a redistribution — flagged via
        # have_lists=false).  Doing this BEFORE find_sph! guarantees the
        # density solve AND the displacement loop consume the same
        # fresh-enough tree.  r_skin is carried from the previous iteration
        # (the model-density length scale is essentially static — total box
        # volume is conserved — so this is the standard Verlet construction);
        # the first iteration has r_skin==0 ⇒ have_lists==false ⇒ rebuild.
        tree_rebuilt = false
        if !have_lists
            tree = build_tree(particles.pos)
            copyto!(ref_pos, particles.pos)
            tree_rebuilt = true
        else
            md2 = _max_disp2(particles.pos, ref_pos, n, boxv, periodic, chunks)
            if md2 > (0.5 * r_skin)^2
                tree = build_tree(particles.pos)
                copyto!(ref_pos, particles.pos)
                tree_rebuilt = true
            end
        end

        # --- 1. SPH quantities (hsml/rho/varhsmlfac/rho_model) -------------
        # find_sph! also reuses particles.hsml as the Newton seed (§4b — do
        # NOT zero it).  `tree` is fresh-enough by the Verlet gate above.
        tree = find_sph_quantities!(particles, param, problem, prob, kc;
                                    tree = tree)

        # --- 2. iteration / Maxiter ---------------------------------------
        it += 1
        if it - 1 > param.Maxiter      # C: `if (it++ > Maxiter)` (post-incr)
            verbose && println("Max iterations reached, result might not be ",
                               "converged properly.")
            break
        end

        # --- 3. SAVE_WVT_STEPS (default off) -------------------------------
        if save_wvt_steps
            write_step_file(particles, param, problem, it;
                            output_diagnostics = output_diagnostics)
        end

        # --- 4. redistribution --------------------------------------------
        reset_redistribution_flags!(particles)
        if it <= param.LastMoveStep && it % param.RedistributionFrequency == 0
            firstIt = 1
            amplitude = param.MoveFractionMax
            # C: LastMoveStep / RedistributionFrequency is INTEGER division
            # (both int); likewise it / RedistributionFrequency.
            decay_denom = div(param.LastMoveStep,
                              param.RedistributionFrequency) - firstIt
            decay = log(param.MoveFractionMax / param.MoveFractionMin) /
                    decay_denom
            moveFraction = amplitude *
                exp(-decay *
                    (div(it, param.RedistributionFrequency) - firstIt))
            # Documented, intentional deviation from wvt_relax.c
            # (CLAUDE.md §1.4 / PORT_STATUS.md).  The C computes `decay`
            # with a `LastMoveStep/RedistributionFrequency - 1` denominator
            # and then casts `Npart*moveFraction` to `int`.  When that
            # denominator is <= 0 (e.g. LastMoveStep == RedistributionFrequency
            # ⇒ div == 1 ⇒ denom == 0) the C divides by zero, and with
            # MoveFractionMax == MoveFractionMin the numerator log(1) == 0
            # too, so `decay` becomes 0/0 = NaN, `moveFraction` NaN, and the
            # C silently casts NaN to `int` (undefined garbage, no crash).
            # Julia's `Int`/`floor(Int, …)` correctly refuses NaN/Inf
            # (InexactError).  We therefore guard the degenerate /
            # non-finite case the C leaves undefined: skip the
            # redistribution step for this iteration (treat as
            # movePart == 0 / no redistribution) instead of computing
            # Int(NaN).  This is strictly safer and produces NO behavioural
            # change for well-posed parameters (decay_denom > 0 and finite
            # decay/moveFraction), where the exact C arithmetic is kept.
            if decay_denom > 0 && isfinite(decay) && isfinite(moveFraction)
                movePart = floor(Int, param.Npart * moveFraction)
                maxProbes = floor(Int, param.Npart * param.ProbesFraction *
                                  moveFraction / param.MoveFractionMax)
                nm, np = redistribute_particles!(particles, param, problem,
                                                 prob, movePart, maxProbes;
                                                 seed = seed + it)
                if verbose
                    println("Attempting to redistribute ", movePart,
                            " particles (=", movePart * 100.0 / param.Npart,
                            "%) by probing ", maxProbes)
                    println("Redistributed ", nm,
                            " particles after probing ", np, " particles")
                end
                # C re-runs Find_sph_quantities() after the moves.  Force a
                # tree + candidate-list rebuild (positions jumped arbitrarily).
                tree = find_sph_quantities!(particles, param, problem, prob,
                                            kc; tree = nothing)
                copyto!(ref_pos, particles.pos)
                have_lists = false
            elseif verbose
                println("Skipping redistribution at iteration ", it,
                        ": degenerate decay (denominator ", decay_denom,
                        ", decay ", decay, ", moveFraction ", moveFraction,
                        ") — non-finite, deviating from wvt_relax.c's ",
                        "undefined int(NaN) cast.")
            end
        end

        # --- 5. relative-density-error stats ------------------------------
        errMin, errMax, errMean, errSigma =
            _error_stats(particles, prob.density, n,
                         param.density_function_correction, chunks)
        errMean /= n
        errSigma = sqrt(max(0.0, errSigma / n - errMean * errMean))
        errDiff = (errLast - errMean) / errMean

        if verbose
            println("   #", it, ": Err min=", errMin, " max=", errMax,
                    " mean=", errMean, " sigma=", errSigma,
                    " diff=", errDiff, " step=", step)
        end
        errLast = errMean

        # --- 6. model-density metric hsml + norm --------------------------
        # Function barrier isolates the `prob.density` ::Function-field call
        # (mirrors Phase-2 `_fill_rho_model!`); keeps the N-loop type-stable.
        vSphSum, max_hsml =
            _fill_model_hsml!(particles, mhsml, prob.density, n,
                              param.density_function_correction, wvtnngb,
                              mpart, voln, dim)
        norm_hsml = dim == 2 ?
            sqrt(wvtnngb / vSphSum / pi) * median_boxsize :
            cbrt(wvtnngb / vSphSum / voln) * median_boxsize
        @inbounds for i in 1:n
            mhsml[i] *= norm_hsml
        end
        mean_hsml = (vSphSum / n)
        mean_hsml = dim == 2 ? sqrt(mean_hsml) : cbrt(mean_hsml)
        mean_hsml *= norm_hsml

        # --- Verlet skin: refresh the cached candidate lists --------------
        # r_skin scales with the (normalised) model hsml — the radius the
        # displacement loop probes (carried to the next iteration's top-of-
        # loop tree-rebuild gate).  query_r ≥ the largest possible pair
        # h = 0.5(hsml_i+hsml_j) PLUS the mutual-drift margin r_skin, so a
        # neighbour can drift up to 2·(0.5·r_skin) before the gate forces a
        # rebuild and the cached list is still a faithful superset (test 7).
        # The candidate lists are rebuilt exactly when the tree was rebuilt
        # this iteration (Verlet gate at step 0) — never on stale positions.
        r_skin = _skin_radius(mean_hsml)
        max_hsml_norm = max_hsml * norm_hsml
        query_r = max_hsml_norm * 1.05 + r_skin

        if tree_rebuilt || !have_lists
            _rebuild_candidate_lists!(cand_lists, particles.pos, tree,
                                      query_r, boxv, periodic, wscratch,
                                      chunks)
            have_lists = true
        end

        # --- 7. WVT repulsive displacement --------------------------------
        _wvt_displacement!(particles, mhsml, deltas, cand_lists, kc, step,
                           boxv, periodic, dim, voln, chunks)

        # --- 8. move + box wrap + moveMps ---------------------------------
        cnt, cnt1, cnt2, cnt3 =
            _move_particles!(particles, deltas, desnngb, voln, dim,
                             boxv, periodic, chunks)
        moveMps = (cnt * 100.0 / param.Npart, cnt1 * 100.0 / param.Npart,
                   cnt2 * 100.0 / param.Npart, cnt3 * 100.0 / param.Npart)

        if verbose
            println("        Del ", moveMps[1], "% > Dmps; ", moveMps[2],
                    "% > Dmps/10; ", moveMps[3], "% > Dmps/100; ",
                    moveMps[4], "% > Dmps/1000")
        end

        # --- 9. it==1 movement warnings -----------------------------------
        if it == 1
            if moveMps[1] < 10.0
                @warn "Hardly any initial movement detected. Consider decreasing MpsFraction in the parameter file!"
            elseif moveMps[1] > 80.0
                @warn "A lot of initial movement detected. Consider increasing MpsFraction in the parameter file!"
            end
        end

        # --- 10. diagnostics.log row --------------------------------------
        if output_diagnostics
            errQ = Quadruplet(errMin, errMax, errMean, errSigma)
            deltaQ = calculate_stats_on(deltas, n)
            write_iteration_diagnostics(it, errQ, errDiff, moveMps, deltaQ;
                                        path = diagnostics_path)
        end

        # --- 11. convergence ----------------------------------------------
        if (moveMps[1] < param.LimitMps[1]) ||
           (moveMps[2] < param.LimitMps[2]) ||
           (moveMps[3] < param.LimitMps[3]) ||
           (moveMps[4] < param.LimitMps[4])
            break
        end

        # force convergence if the distribution doesn't tighten and we are
        # NOT on a redistribution step (C `wvt_relax.c:311` condition, EXACTLY
        # — byte-identical for 3D and for the normal 2D tightening case).
        reduced = false
        if cnt1 > last_cnt &&
           (it > param.LastMoveStep ||
            it % param.RedistributionFrequency != 0)
            step *= param.StepReduction
            reduced = true
        end

        # 2D-only supplementary anneal (PORT_STATUS.md "Phase 4 follow-up:
        # 2D relaxation plateau"). The C `cnt1 > last_cnt` proxy for "the
        # distribution does not tighten" is STRUCTURALLY DEAD whenever
        # `cnt1 == 0`: `last_cnt` is the previous `cnt1` (≥ 0, or `floatmax`
        # on it=1), so `0 > last_cnt` is provably FALSE for every reachable
        # state — the C never reduces `step` once the Dmps/10 band empties.
        # In 3D `d_mps` is the true mean particle spacing, so the Dmps/10
        # band stays populated (and noisily fluctuates ⇒ `cnt1 > last_cnt`
        # fires) right up to the converged frozen glass; `cnt1` only reaches
        # exactly 0 at/after that glass, where the C `LimitMps` stop has
        # already broken — so 3D never relies on this branch and is left
        # byte-identical (further guarded by `dim == 2`).
        #
        # In 2D the C `d_mps = (π·h²/DESNNGB)^(1/3)` (wvt_relax.c:225, the
        # documented `1.0/3.0`-in-2D quirk, faithfully ported) is the cube
        # root of an AREA — dimensionally `spacing^(2/3)`, numerically ≈ 5×
        # the true mean spacing in box units. The Dmps/10 band therefore
        # empties (`cnt1 → 0`) within ~100 iterations while the glass is
        # still loose (moveMps[3] ≈ 95–100%, errMean ≈ 0.11 and DRIFTING UP),
        # the C trigger goes permanently dead, `step` freezes at the
        # auto-calibrated (intentionally band-centred, hence large) value,
        # and the relaxation jitters at a noisy non-equilibrium forever — the
        # exact reported symptom. We restore the C author's stated INTENT
        # ("force convergence if distribution doesnt tighten",
        # wvt_relax.c:310) using the dimension-robust signal the C itself
        # already computes every iteration: the mean density error stopped
        # improving (`errDiff = (errLast - errMean)/errMean ≤ 0`). Only when
        # the C trigger is structurally dead (`cnt1 == 0`), the glass is
        # NOT yet at the C stop (`moveMps[4] ≥ LimitMps[4]`, i.e. the run
        # would otherwise continue forever), AND the error is not improving,
        # AND we are not on a C-protected redistribution step — apply the
        # SAME `step *= StepReduction`. This cannot ever co-fire with the C
        # trigger (mutually exclusive on `cnt1 == 0`) and changes nothing in
        # 3D (dim guard + the regime is unreachable before the 3D glass has
        # converged), so the legacy/auto/3D behaviour is preserved.
        if dim == 2 && !reduced && cnt1 == 0 &&
           errDiff <= 0.0 &&
           moveMps[4] >= param.LimitMps[4] &&
           (it > param.LastMoveStep ||
            it % param.RedistributionFrequency != 0)
            step *= param.StepReduction
        end

        last_cnt = Float64(cnt1)
    end

    verbose && println("done")
    return particles
end

# 4-arg convenience (driver / main): derive prob + default kernel config.
function regularise_sph_particles!(particles::Particles, param::Parameters,
                                   problem::ProblemParameters; kwargs...)
    prob = setup_problem(param)
    kc = default_kernel_config()
    return regularise_sph_particles!(particles, param, problem, prob, kc;
                                     kwargs...)
end

# model-density metric hsml (ports wvt_relax.c lines 122–138).  Function
# barrier on the `prob.density` ::Function field (`dfun::F` is concrete inside).
function _fill_model_hsml!(particles::Particles, mhsml::Vector{Float64},
                           dfun::F, n::Int, density_function_correction,
                           wvtnngb::Float64, mpart::Float64, voln::Float64,
                           dim::Int) where {F}
    vSphSum = 0.0
    max_hsml = 0.0
    @inbounds for i in 1:n
        rho = Float64(dfun(particles, i, density_function_correction))
        particles.rho_model[i] = Float32(rho)
        if dim == 2
            h = sqrt(wvtnngb * mpart / rho / pi)
            mhsml[i] = h
            vSphSum += h * h
        else
            h = cbrt(wvtnngb * mpart / rho / voln)
            mhsml[i] = h
            vSphSum += h * h * h
        end
        max_hsml = max(max_hsml, h)
    end
    return vSphSum, max_hsml
end

# --- MpsFraction startup auto-calibration ----------------------------------
# MPSFRACTION_ANALYSIS.md §3/§4: analytic seed + log-space bisection of
# `step` so the first-iteration moveMps[0] lands in the C author's
# (MPS_AUTO_BAND_LO, MPS_AUTO_BAND_HI) band, targeting the centre.
#
# Step-INDEPENDENT work (the KDTree, the SPH density solve / d_mps inputs,
# the model-hsml metric, and the cached candidate lists) is computed ONCE;
# only the displacement pass + the count-only moveMps[0] tally are recomputed
# per trial, on scratch `deltas` (positions are NEVER mutated — calibration
# does not advance the relaxation; the main loop then runs unchanged).
#
# moveMps[0] is provably monotone increasing in `step` for fixed positions
# (|δ| ∝ step, d_mps fixed by the density solve — analysis §3.2), so a clean
# bracket + geometric bisection on log(step) is safe. Falls back to the
# analytic-Courant "never worse than legacy" seed if the band is unreachable
# within MPS_AUTO_MAX_TRIALS (analysis §4.4). Returns a finite positive step.
function _autocalibrate_step(particles::Particles, param::Parameters,
                             problem::ProblemParameters, prob::Problem,
                             kc::KernelConfig, mhsml::Vector{Float64},
                             deltas::NTuple{3,Vector{Float32}},
                             cand_lists::Vector{Vector{Int}},
                             wscratch::Vector{WvtScratch},
                             chunks::Vector{UnitRange{Int}}, n::Int,
                             box::NTuple{3,Float64},
                             periodic::NTuple{3,Bool}, dim::Int,
                             voln::Float64, desnngb::Float64,
                             wvtnngb::Float64, mpart::Float64,
                             median_boxsize::Float64; verbose::Bool = true)

    seed_step = _analytic_seed_step(n, dim)

    # --- step-independent setup, computed ONCE (reused for every trial) ----
    # Tree + density solve on the (un-mutated) initial positions.
    tree = build_tree(particles.pos)
    tree = find_sph_quantities!(particles, param, problem, prob, kc;
                                tree = tree)
    # model-density metric hsml + norm (positions-only; step-independent).
    vSphSum, max_hsml =
        _fill_model_hsml!(particles, mhsml, prob.density, n,
                          param.density_function_correction, wvtnngb,
                          mpart, voln, dim)
    norm_hsml = dim == 2 ?
        sqrt(wvtnngb / vSphSum / pi) * median_boxsize :
        cbrt(wvtnngb / vSphSum / voln) * median_boxsize
    @inbounds for i in 1:n
        mhsml[i] *= norm_hsml
    end
    mean_hsml = vSphSum / n
    mean_hsml = dim == 2 ? sqrt(mean_hsml) : cbrt(mean_hsml)
    mean_hsml *= norm_hsml
    r_skin = _skin_radius(mean_hsml)
    query_r = max_hsml * norm_hsml * 1.05 + r_skin
    _rebuild_candidate_lists!(cand_lists, particles.pos, tree, query_r,
                              box, periodic, wscratch, chunks)

    # one trial = displacement into scratch `deltas` + count-only moveMps[0].
    # Positions are NOT mutated; `_move_particles!` is NOT called.
    function trial_movemps0(s::Float64)
        _wvt_displacement!(particles, mhsml, deltas, cand_lists, kc, s,
                           box, periodic, dim, voln, chunks)
        cnt, _, _, _ = _count_moves(particles, deltas, desnngb, voln, dim,
                                    chunks)
        return cnt * 100.0 / param.Npart
    end

    s = seed_step
    m = trial_movemps0(s)
    trials = 1
    if MPS_AUTO_BAND_LO <= m <= MPS_AUTO_BAND_HI
        verbose && println("   MpsFraction auto-calibration: analytic seed ",
                           "accepted (step=", s, ", moveMps[0]=", m,
                           "%, trials=", trials, ")")
        return s
    end

    factor = 4.0
    s_lo = s
    m_lo = m
    s_hi = s
    m_hi = m
    bracketed = false
    if m < MPS_AUTO_TARGET
        # step too small ⇒ enlarge until moveMps[0] reaches the target.
        while trials < MPS_AUTO_MAX_TRIALS && m < MPS_AUTO_TARGET
            s_lo = s
            m_lo = m
            s *= factor
            m = trial_movemps0(s)
            trials += 1
        end
        s_hi = s
        m_hi = m
        bracketed = m >= MPS_AUTO_TARGET
    else
        # step too large ⇒ shrink until moveMps[0] drops to the target.
        while trials < MPS_AUTO_MAX_TRIALS && m > MPS_AUTO_TARGET
            s_hi = s
            m_hi = m
            s /= factor
            m = trial_movemps0(s)
            trials += 1
        end
        s_lo = s
        m_lo = m
        bracketed = m <= MPS_AUTO_TARGET
    end

    # If the seed itself already landed in band on either bracket endpoint,
    # accept it (cheap common case after one bracket step).
    if MPS_AUTO_BAND_LO <= m <= MPS_AUTO_BAND_HI
        verbose && println("   MpsFraction auto-calibration: bracket step ",
                           "in band (step=", s, ", moveMps[0]=", m,
                           "%, trials=", trials, ")")
        return s
    end

    if !bracketed
        # Could not bracket the target within the budget (e.g. a
        # near-discontinuous Sod-like field). Fall back to the analytic
        # "never worse than legacy" seed and continue (analysis §4.4).
        verbose && println("   WARNING: MpsFraction auto-calibration could ",
                           "not bracket the target in ", trials,
                           " trials; falling back to the analytic seed ",
                           "step=", seed_step, " (never worse than legacy).")
        return seed_step
    end

    # Geometric (log-space) bisection between the bracket endpoints until
    # moveMps[0] lands in band or the trial budget is exhausted.
    best = s
    while trials < MPS_AUTO_MAX_TRIALS
        sm = sqrt(s_lo * s_hi)
        mm = trial_movemps0(sm)
        trials += 1
        best = sm
        if MPS_AUTO_BAND_LO <= mm <= MPS_AUTO_BAND_HI
            verbose && println("   MpsFraction auto-calibration: converged ",
                               "(step=", sm, ", moveMps[0]=", mm,
                               "%, trials=", trials, ")")
            return sm
        end
        if mm < MPS_AUTO_TARGET
            s_lo = sm
        else
            s_hi = sm
        end
    end

    # Budget exhausted without landing exactly in band: return the geometric
    # midpoint of the final (tight) bracket — guaranteed finite & positive
    # and, since the bracket straddles the band centre, in/adjacent to band;
    # StepReduction then adapts downstream.
    final = sqrt(s_lo * s_hi)
    if !(isfinite(final) && final > 0.0)
        final = seed_step                       # never let a bad step escape
    end
    verbose && println("   MpsFraction auto-calibration: trial budget (",
                       MPS_AUTO_MAX_TRIALS,
                       ") reached, using bracket midpoint step=", final,
                       " (last best=", best, ").")
    return final
end

# --- per-chunk function barriers for the WVT parallel loops ----------------
# Every relax hot loop runs through `_run_chunks` (see src/parallel/threads.jl)
# with a thin `do c -> _xxx_chunk!(c, ...)` forwarder so the closure captures
# only concrete, already-typed arguments (NO boxed fat closure). The bodies
# below are byte-for-byte the previously-inline `Threads.@threads` bodies;
# each chunk writes only its own per-chunk slot (NEVER threadid-indexed), so
# results and determinism are identical across all three backends.

@noinline function _error_stats_chunk!(c::Int, chunks::Vector{UnitRange{Int}},
                                       particles::Particles, dfun::F,
                                       density_function_correction,
                                       pmin::Vector{Float64},
                                       pmax::Vector{Float64},
                                       psum::Vector{Float64},
                                       psq::Vector{Float64}) where {F}
    lmin = floatmax(Float64)
    lmax = 0.0
    lsum = 0.0
    lsq = 0.0
    @inbounds for ipart in chunks[c]
        rho_model = Float64(dfun(particles, ipart, density_function_correction))
        err = abs((Float64(particles.rho[ipart]) - rho_model) / rho_model)
        lmin = min(err, lmin)
        lmax = max(err, lmax)
        lsum += err
        lsq += err * err
    end
    @inbounds begin
        pmin[c] = lmin
        pmax[c] = lmax
        psum[c] = lsum
        psq[c] = lsq
    end
    return nothing
end

@noinline function _max_disp2_chunk!(c::Int, chunks::Vector{UnitRange{Int}},
                                     pos::Vector{SVector{3,Float64}},
                                     ref::Vector{SVector{3,Float64}},
                                     box::NTuple{3,Float64},
                                     periodic::NTuple{3,Bool},
                                     pmax::Vector{Float64})
    m = 0.0
    @inbounds for i in chunks[c]
        d2 = periodic_dist2(pos[i], ref[i], box, periodic)
        m = max(m, d2)
    end
    @inbounds pmax[c] = m
    return nothing
end

@noinline function _rebuild_candidate_lists_chunk!(c::Int,
                                                   chunks::Vector{UnitRange{Int}},
                                                   cand_lists::Vector{Vector{Int}},
                                                   pos::Vector{SVector{3,Float64}},
                                                   tree::KDTree,
                                                   query_r::Float64,
                                                   box::NTuple{3,Float64},
                                                   periodic::NTuple{3,Bool},
                                                   wscratch::Vector{WvtScratch})
    ws = wscratch[c]
    @inbounds for ipart in chunks[c]
        query_candidates!(cand_lists[ipart], ws.qtmp, tree, pos,
                          pos[ipart], query_r, box, periodic)
    end
    return nothing
end

@noinline function _wvt_displacement_chunk!(c::Int,
                                            chunks::Vector{UnitRange{Int}},
                                            pos::Vector{SVector{3,Float64}},
                                            mhsml::Vector{Float64},
                                            dx::Vector{Float32},
                                            dy::Vector{Float32},
                                            dz::Vector{Float32},
                                            cand_lists::Vector{Vector{Int}},
                                            kc::KernelConfig, step::Float64,
                                            box::NTuple{3,Float64},
                                            periodic::NTuple{3,Bool},
                                            dim::Int)
    @inbounds for ipart in chunks[c]
        dxi = 0.0
        dyi = 0.0
        dzi = 0.0
        hi = mhsml[ipart]
        pi3 = pos[ipart]
        cl = cand_lists[ipart]
        for t in eachindex(cl)
            jpart = cl[t]
            jpart == ipart && continue
            pj = pos[jpart]
            ddx = minimum_image(pi3[1] - pj[1], box[1], periodic[1])
            ddy = minimum_image(pi3[2] - pj[2], box[2], periodic[2])
            ddz = minimum_image(pi3[3] - pj[3], box[3], periodic[3])
            r2 = ddx * ddx + ddy * ddy + ddz * ddz
            # C: Assert(r2 > 0, "Found two particles ... same location")
            r2 > 0.0 || error(
                "Found two particles $ipart & $jpart at the same " *
                "location. Consider increasing the space between your " *
                "density field and the box boundaries.")
            h = 0.5 * (hi + mhsml[jpart])
            r2 > h * h && continue
            r = sqrt(r2)
            h_inv = 1.0 / h
            kernel_fac = dim == 2 ? h * h : h * h * h
            wk = sph_kernel(kc, r, h_inv) * kernel_fac
            fac = step * h * wk / r
            dxi += fac * ddx
            dyi += fac * ddy
            if dim != 2
                dzi += fac * ddz
            end
        end
        dx[ipart] = Float32(dxi)
        dy[ipart] = Float32(dyi)
        dz[ipart] = Float32(dzi)
    end
    return nothing
end

# moveMps count math (ports wvt_relax.c lines 217–246), factored out of the
# move loop so it can be reused by the startup auto-calibration *without*
# mutating particle positions or any other state. Pure function of the
# displacement magnitudes + `hsmlv` (the SPH-solve `Hsml`, the C `d_mps`
# input) — byte-identical to the count the C / the move loop computes.
@inline function _count_move_thresholds(ipart::Int,
                                        dx::Vector{Float32},
                                        dy::Vector{Float32},
                                        dz::Vector{Float32},
                                        hsmlv, desnngb::Float64,
                                        voln::Float64, dim::Int)
    @inbounds begin
        ex = Float64(dx[ipart])
        ey = Float64(dy[ipart])
        ez = Float64(dz[ipart])
        d = sqrt(ex * ex + ey * ey + ez * ez)
        h = Float64(hsmlv[ipart])
    end
    d_mps = dim == 2 ?
        cbrt(pi * h * h / desnngb) :
        cbrt(voln * h * h * h / desnngb)
    b0 = d > 1.0 * d_mps
    b1 = d > 0.1 * d_mps
    b2 = d > 0.01 * d_mps
    b3 = d > 0.001 * d_mps
    return b0, b1, b2, b3
end

# Count-only chunk: the moveMps[0..3] tallies WITHOUT touching `pos` or any
# other state (the calibration trial path). Same arithmetic as the move loop.
@noinline function _count_moves_chunk!(c::Int,
                                       chunks::Vector{UnitRange{Int}},
                                       hsmlv,
                                       dx::Vector{Float32},
                                       dy::Vector{Float32},
                                       dz::Vector{Float32},
                                       desnngb::Float64, voln::Float64,
                                       dim::Int,
                                       pc::Vector{Int}, pc1::Vector{Int},
                                       pc2::Vector{Int}, pc3::Vector{Int})
    lc = 0
    lc1 = 0
    lc2 = 0
    lc3 = 0
    @inbounds for ipart in chunks[c]
        b0, b1, b2, b3 = _count_move_thresholds(ipart, dx, dy, dz, hsmlv,
                                                desnngb, voln, dim)
        b0 && (lc += 1)
        b1 && (lc1 += 1)
        b2 && (lc2 += 1)
        b3 && (lc3 += 1)
    end
    @inbounds begin
        pc[c] = lc
        pc1[c] = lc1
        pc2[c] = lc2
        pc3[c] = lc3
    end
    return nothing
end

@noinline function _move_particles_chunk!(c::Int,
                                          chunks::Vector{UnitRange{Int}},
                                          pos::Vector{SVector{3,Float64}},
                                          hsmlv,
                                          dx::Vector{Float32},
                                          dy::Vector{Float32},
                                          dz::Vector{Float32},
                                          desnngb::Float64, voln::Float64,
                                          dim::Int, box::NTuple{3,Float64},
                                          pc::Vector{Int}, pc1::Vector{Int},
                                          pc2::Vector{Int}, pc3::Vector{Int})
    lc = 0
    lc1 = 0
    lc2 = 0
    lc3 = 0
    @inbounds for ipart in chunks[c]
        # exact same count math as the count-only path (factored helper) so
        # the legacy / normal-loop behaviour is byte-identical to before.
        b0, b1, b2, b3 = _count_move_thresholds(ipart, dx, dy, dz, hsmlv,
                                                desnngb, voln, dim)
        b0 && (lc += 1)
        b1 && (lc1 += 1)
        b2 && (lc2 += 1)
        b3 && (lc3 += 1)
        ex = Float64(dx[ipart])
        ey = Float64(dy[ipart])
        ez = Float64(dz[ipart])
        p = pos[ipart]
        nx = _box_wrap(p[1] + ex, box[1])
        ny = _box_wrap(p[2] + ey, box[2])
        nz = _box_wrap(p[3] + ez, box[3])
        pos[ipart] = SVector{3,Float64}(nx, ny, nz)
    end
    @inbounds begin
        pc[c] = lc
        pc1[c] = lc1
        pc2[c] = lc2
        pc3[c] = lc3
    end
    return nothing
end

# --- relative-density-error reduction (chunked, no threadid indexing) ------
# C: `#pragma omp parallel for reduction(...)` over relativeDensityError.
# `dfun::F` (the `prob.density` ::Function field) is passed concretely so the
# threaded reduction stays type-stable (function barrier).
function _error_stats(particles::Particles, dfun::F, n::Int,
                      density_function_correction,
                      chunks::Vector{UnitRange{Int}}) where {F}
    nc = length(chunks)
    pmin = fill(floatmax(Float64), nc)
    pmax = zeros(Float64, nc)
    psum = zeros(Float64, nc)
    psq = zeros(Float64, nc)
    _run_chunks(nc) do c
        _error_stats_chunk!(c, chunks, particles, dfun,
                            density_function_correction,
                            pmin, pmax, psum, psq)
    end
    emin = floatmax(Float64)
    emax = 0.0
    esum = 0.0
    esq = 0.0
    for c in 1:nc
        emin = min(emin, pmin[c])
        emax = max(emax, pmax[c])
        esum += psum[c]
        esq += psq[c]
    end
    return emin, emax, esum, esq
end

# max squared min-image displacement since `ref_pos` (Verlet rebuild test).
function _max_disp2(pos::Vector{SVector{3,Float64}},
                    ref::Vector{SVector{3,Float64}}, n::Int,
                    box::NTuple{3,Float64}, periodic::NTuple{3,Bool},
                    chunks::Vector{UnitRange{Int}})
    nc = length(chunks)
    pmax = zeros(Float64, nc)
    _run_chunks(nc) do c
        _max_disp2_chunk!(c, chunks, pos, ref, box, periodic, pmax)
    end
    return maximum(pmax)
end

# Rebuild every particle's cached candidate index list (Verlet skin).
function _rebuild_candidate_lists!(cand_lists::Vector{Vector{Int}},
                                   pos::Vector{SVector{3,Float64}},
                                   tree::KDTree, query_r::Float64,
                                   box::NTuple{3,Float64},
                                   periodic::NTuple{3,Bool},
                                   wscratch::Vector{WvtScratch},
                                   chunks::Vector{UnitRange{Int}})
    nc = length(chunks)
    _run_chunks(nc) do c
        _rebuild_candidate_lists_chunk!(c, chunks, cand_lists, pos, tree,
                                        query_r, box, periodic, wscratch)
    end
    return nothing
end

# --- WVT repulsive displacement (ports wvt_relax.c lines 151–213) ----------
function _wvt_displacement!(particles::Particles, mhsml::Vector{Float64},
                            deltas::NTuple{3,Vector{Float32}},
                            cand_lists::Vector{Vector{Int}},
                            kc::KernelConfig, step::Float64,
                            box::NTuple{3,Float64}, periodic::NTuple{3,Bool},
                            dim::Int, voln::Float64,
                            chunks::Vector{UnitRange{Int}})
    pos = particles.pos
    dx, dy, dz = deltas
    nc = length(chunks)
    _run_chunks(nc) do c
        _wvt_displacement_chunk!(c, chunks, pos, mhsml, dx, dy, dz,
                                 cand_lists, kc, step, box, periodic, dim)
    end
    return nothing
end

# --- move + per-axis box wrap + moveMps thresholds ------------------------
# Ports wvt_relax.c lines 217–270.  The C wrap is a per-axis `while` loop:
# while Pos<0 Pos+=L; while Pos>L Pos-=L  (note: strict `> L`, not `>= L`).
# Reproduced exactly (NOT branchless min-image — the spec requires the C
# wrap semantics here, incl. the non-periodic axes which the C wraps too).
@inline function _box_wrap(x::Float64, L::Float64)
    L <= 0.0 && return x
    while x < 0.0
        x += L
    end
    while x > L
        x -= L
    end
    return x
end

function _move_particles!(particles::Particles,
                          deltas::NTuple{3,Vector{Float32}},
                          desnngb::Float64, voln::Float64, dim::Int,
                          box::NTuple{3,Float64}, periodic::NTuple{3,Bool},
                          chunks::Vector{UnitRange{Int}})
    pos = particles.pos
    hsmlv = particles.hsml
    dx, dy, dz = deltas
    nc = length(chunks)
    pc = zeros(Int, nc)
    pc1 = zeros(Int, nc)
    pc2 = zeros(Int, nc)
    pc3 = zeros(Int, nc)
    _run_chunks(nc) do c
        _move_particles_chunk!(c, chunks, pos, hsmlv, dx, dy, dz,
                               desnngb, voln, dim, box, pc, pc1, pc2, pc3)
    end
    return sum(pc), sum(pc1), sum(pc2), sum(pc3)
end

# Count-only sibling of `_move_particles!`: returns the same
# (cnt,cnt1,cnt2,cnt3) tallies the move loop would produce for the given
# scratch displacements, WITHOUT moving any particle (used by the startup
# auto-calibration trials — positions/state are never mutated). Same chunk /
# threading discipline (per-chunk slot indexed by the loop var, no threadid).
function _count_moves(particles::Particles,
                      deltas::NTuple{3,Vector{Float32}},
                      desnngb::Float64, voln::Float64, dim::Int,
                      chunks::Vector{UnitRange{Int}})
    hsmlv = particles.hsml
    dx, dy, dz = deltas
    nc = length(chunks)
    pc = zeros(Int, nc)
    pc1 = zeros(Int, nc)
    pc2 = zeros(Int, nc)
    pc3 = zeros(Int, nc)
    _run_chunks(nc) do c
        _count_moves_chunk!(c, chunks, hsmlv, dx, dy, dz,
                            desnngb, voln, dim, pc, pc1, pc2, pc3)
    end
    return sum(pc), sum(pc1), sum(pc2), sum(pc3)
end
