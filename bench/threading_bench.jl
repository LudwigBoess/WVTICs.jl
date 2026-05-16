# ---------------------------------------------------------------------------
# Threading-backend benchmark harness.
#
# Run with:
#     julia --project=. -t auto bench/threading_bench.jl
#     julia --project=. -t auto bench/threading_bench.jl 100000 20000
#
# Measures every hot threaded kernel under the three swappable backends
#   :base        — Base.Threads.@threads      (current default)
#   :polyester   — Polyester.@batch
#   :ohmythreads — OhMyThreads.tforeach
# in ONE process, WITHOUT editing source between runs and WITHOUT calling the
# slow `setup()` 512³ integral. The workload (`Particles` + `Parameters` +
# `ProblemParameters` + `Problem`) is built directly, exactly as the Phase-3
# tests do (constant density ρ≡1 on the unit periodic cube, Mpart analytic).
#
# Each kernel's *outer* chunk loop is reproduced here driving the SAME
# real per-chunk function-barrier helpers (`WVTICs._density_chunk!`,
# `_wvt_displacement_chunk!`, …) through `WVTICs._run_chunks(kernel, nc,
# Val(backend))`, so the measured work is the production code path; only the
# backend `Val` differs. Nothing in the package or repo is mutated (no
# snapshots; `mktempdir` is used for the unavoidable diagnostics path).
#
# Methodology: warm up once (untimed, forces compilation) then report the
# MINIMUM of `NRUNS` `@elapsed` timings (no BenchmarkTools dependency). The
# table prints time in ms per (kernel × backend) and the speedup vs :base.
# ---------------------------------------------------------------------------

using WVTICs
using WVTICs: Particles, Parameters, ProblemParameters, KernelConfig,
              WendlandC4, setup_problem, _chunk_ranges, _run_chunks,
              find_sph_quantities!, build_tree, _wvt_vol_norm, _skin_radius,
              NgbScratch, WvtScratch,
              _density_chunk!, _positions_chunk!, _mpart_chunk!,
              _error_stats_chunk!, _max_disp2_chunk!,
              _rebuild_candidate_lists_chunk!, _wvt_displacement_chunk!,
              _move_particles_chunk!, _fill_model_hsml!, _mpart_slice
using StaticArrays
using Random

const BACKENDS = (:base, :polyester, :ohmythreads)
const NRUNS = 5

_ms(t) = string(round(t * 1e3; digits = 3))
_spd(tb, tx) = tx > 0 ? string(round(tb / tx; digits = 2)) : "n/a"

# Fixed-width table row (defined before `main`: macros expand at parse time).
macro printf_row(a, b, c, d, e, f)
    quote
        println(rpad($(esc(a)), 34), " | ",
                lpad($(esc(b)), 9), " | ", lpad($(esc(c)), 9), " | ",
                lpad($(esc(d)), 9), " | ", lpad($(esc(e)), 6), " | ",
                lpad($(esc(f)), 6))
    end
end

# --- timing helper ---------------------------------------------------------
# Warm up once (untimed) then return the minimum of `NRUNS` `@elapsed` runs.
function bestof(f!)
    f!()                              # warm-up / compile (untimed)
    t = Inf
    for _ in 1:NRUNS
        t = min(t, @elapsed f!())
    end
    return t
end

# --- direct workload construction (no setup(), no 512³ integral) -----------
# Constant density ρ≡1 on the unit periodic cube; Mpart set analytically
# (M_tot = L³·ρ = 1 ⇒ Mpart = 1/N). Positions: deterministic jittered
# near-lattice so the tree/neighbour structure is representative.
function make_workload(N::Int)
    L = 1.0
    ps = Particles(N)
    rng = Random.Xoshiro(20240516)
    @inbounds for i in 1:N
        ps.pos[i] = SVector{3,Float64}(rand(rng) * L, rand(rng) * L,
                                       rand(rng) * L)
        ps.type[i] = Int32(0)
    end
    param = Parameters()
    param.Npart = N
    param.Problem_Flag = 0
    param.Problem_Subflag = 0
    param.BiasCorrection = 0.0
    param.MpsFraction = 2.0          # only used for `step` scale; benign
    problem = ProblemParameters(; Name = "IC_Constant_Density",
                                  Mpart = 1.0 / N,
                                  Boxsize = (L, L, L),
                                  Rho_Max = 1.0,
                                  Periodic = (true, true, true))
    prob = setup_problem(param)
    kc = KernelConfig(WendlandC4; dim = 3)
    return ps, param, problem, prob, kc, L
end

# --- per-kernel runners parameterised on the backend Val -------------------
# Each reproduces the production kernel's OUTER driver, dispatching the SAME
# real per-chunk helpers through `_run_chunks(…, Val(backend))`.

# find_sph_quantities! body (fresh tree each call, like the Verlet first iter).
function run_density(ps, param, problem, prob, kc, nc, chunks, scratch,
                     ::Val{B}) where {B}
    positions = ps.pos
    tree = build_tree(positions)
    mpart = Float64(problem.Mpart)
    box = problem.Boxsize
    periodic = problem.Periodic
    _run_chunks(nc, Val(B)) do c
        _density_chunk!(c, chunks, ps, positions, tree, scratch,
                        kc, mpart, box, periodic)
    end
    return nothing
end

# make_positions! rejection-sampling body. Writes into a private scratch
# `Particles` so the shared workload state is never disturbed.
function run_positions(psout, prob, nc, chunks, ::Val{B}) where {B}
    dfun = prob.density
    _run_chunks(nc, Val(B)) do c
        _positions_chunk!(c, chunks, psout, dfun, WVTICs.RejectionSampling,
                          14041981, 1.0, 1.0, 1.0, 1.0, 0.0)
    end
    return nothing
end

# mpart_from_integral body — REDUCED grid (Ngrid per side, not 1<<9).
function run_mpart(prob, Ngrid::Int, nc, chunks, scratch, partial,
                   ::Val{B}) where {B}
    dfun = prob.density
    dx = 1.0 / Ngrid
    _run_chunks(nc, Val(B)) do c
        _mpart_chunk!(c, chunks, scratch, partial, dfun, Ngrid, dx, dx, dx)
    end
    return nothing
end

# _error_stats body.
function run_error_stats(ps, prob, nc, chunks, pmin, pmax, psum, psq,
                         ::Val{B}) where {B}
    dfun = prob.density
    _run_chunks(nc, Val(B)) do c
        _error_stats_chunk!(c, chunks, ps, dfun, 0.0, pmin, pmax, psum, psq)
    end
    return nothing
end

# _wvt_displacement! body (operates on cached candidate lists + model hsml).
function run_displacement(ps, mhsml, dx, dy, dz, cand_lists, kc, box,
                          periodic, nc, chunks, ::Val{B}) where {B}
    pos = ps.pos
    _run_chunks(nc, Val(B)) do c
        _wvt_displacement_chunk!(c, chunks, pos, mhsml, dx, dy, dz,
                                 cand_lists, kc, 0.1, box, periodic, 3)
    end
    return nothing
end

# --- main ------------------------------------------------------------------
function main(Ns::Vector{Int})
    nthr = Threads.nthreads()
    println("="^74)
    println("WVTICs threading-backend benchmark")
    println("  Julia $(VERSION)   Threads.nthreads() = $nthr")
    println("  backends      : ", join(BACKENDS, ", "))
    println("  timing        : min of $NRUNS @elapsed runs (1 warm-up)")
    println("  workload      : constant ρ≡1, unit periodic cube, WC4 kernel")
    println("="^74)

    for N in Ns
        ps, param, problem, prob, kc, L = make_workload(N)
        nc = max(1, nthr)
        chunks = _chunk_ranges(N, nc)
        nc = length(chunks)

        # --- shared per-kernel state (built once per N, reused per backend)
        ngb_scratch = [NgbScratch() for _ in 1:nc]
        pmin = fill(floatmax(Float64), nc); pmax = zeros(Float64, nc)
        psum = zeros(Float64, nc);          psq = zeros(Float64, nc)

        # density solve once to get a realistic hsml/rho state for the WVT
        # displacement + error-stats kernels (uses production driver).
        find_sph_quantities!(ps, param, problem, prob, kc)

        # model hsml + cached candidate lists for the displacement kernel.
        mhsml = zeros(Float64, N)
        _fill_model_hsml!(ps, mhsml, prob.density, N, 0.0,
                          Float64(kc.desnngb), Float64(problem.Mpart),
                          _wvt_vol_norm(3), 3)
        @inbounds for i in 1:N
            mhsml[i] = mhsml[i] <= 0 ? Float64(ps.hsml[i]) : mhsml[i]
        end
        dxv = zeros(Float32, N); dyv = zeros(Float32, N); dzv = zeros(Float32, N)
        tree = build_tree(ps.pos)
        cand_lists = [Int[] for _ in 1:N]
        wscratch = [WvtScratch() for _ in 1:nc]
        mean_h = sum(mhsml) / N
        query_r = maximum(mhsml) * 1.05 + _skin_radius(mean_h)
        _run_chunks(nc, Val(:base)) do c
            _rebuild_candidate_lists_chunk!(c, chunks, cand_lists, ps.pos,
                                            tree, query_r, problem.Boxsize,
                                            problem.Periodic, wscratch)
        end

        # private scratch container for the positions kernel (rejection
        # sampling overwrites pos/type — keep the shared workload intact).
        ps_pos = Particles(N)

        # reduced mpart grid: 1<<6 per side (vs production 1<<9).
        Ngrid = 1 << 6
        mp_nc = max(1, nthr)
        mp_chunks = _chunk_ranges(Ngrid, mp_nc); mp_nc = length(mp_chunks)
        mp_scratch = [Particles(1) for _ in 1:mp_nc]
        mp_partial = zeros(Float64, mp_nc)

        kernels = (
            ("find_sph_quantities!",
             b -> run_density(ps, param, problem, prob, kc, nc, chunks,
                              ngb_scratch, b)),
            ("make_positions! (rejection)",
             b -> run_positions(ps_pos, prob, nc, chunks, b)),
            ("_wvt_displacement!",
             b -> run_displacement(ps, mhsml, dxv, dyv, dzv, cand_lists,
                                    kc, problem.Boxsize, problem.Periodic,
                                    nc, chunks, b)),
            ("_error_stats",
             b -> run_error_stats(ps, prob, nc, chunks, pmin, pmax, psum,
                                  psq, b)),
            ("mpart_from_integral (grid=$(Ngrid)³)",
             b -> run_mpart(prob, Ngrid, mp_nc, mp_chunks, mp_scratch,
                            mp_partial, b)),
        )

        println()
        println("N = $N  (chunks = $nc)")
        println("-"^74)
        @printf_row("kernel", "base [ms]", "poly [ms]", "omt [ms]",
                    "poly×", "omt×")
        println("-"^74)
        for (name, runner) in kernels
            tb = bestof(() -> runner(Val(:base)))
            tp = bestof(() -> runner(Val(:polyester)))
            to = bestof(() -> runner(Val(:ohmythreads)))
            @printf_row(name,
                        _ms(tb), _ms(tp), _ms(to),
                        _spd(tb, tp), _spd(tb, to))
        end
        println("-"^74)
    end
    println()
    println("Speedup = base_time / backend_time  (>1 ⇒ backend faster).")
    println("Flip `const THREAD_BACKEND` in src/parallel/threads.jl to the ",
            "winner.")
    return nothing
end

# --- entry -----------------------------------------------------------------
let
    Ns = isempty(ARGS) ? [100_000, 20_000] : parse.(Int, ARGS)
    main(Ns)
end
