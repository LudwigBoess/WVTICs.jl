# Phase 3 — diagnostics.  Ports `diagnostics.c` (+ `diagnostics.h`) and the
# `wvt_relax.c::writeStepFile` / `SAVE_WVT_STEPS` snapshot hook.
#
# C reference behaviour:
#   * `OUTPUT_DIAGNOSTICS` is ON by default → a `diagnostics.log` file with the
#     exact tab-separated header from `diagnostics.c::initIterationDiagnostics`
#     and one `%03d` + 13× `%+7.5e` row per iteration
#     (`writeIterationDiagnostics`).
#   * `calculateStatsOn(values[3], n)` → (min,max,mean,sigma) of the per-particle
#     3-vector magnitude `v = sqrt(vx²+vy²+vz²)`, with
#     `sigma = sqrt(Σv²/n − mean²)` (NB: the C accumulates `v2`, not `v`, into
#     `sigma`, i.e. variance of the *magnitude*).
#   * `SAVE_WVT_STEPS` is OFF by default → `writeStepFile(it)` writes
#     `Problem.Name_NNN` via `Write_output(0)` then restores `Problem.Name`.
#     Exposed here as a keyword flag, default off, matching the C default.

"""
    Quadruplet

Port of C `struct Quadruplet { double min, max, mean, sigma; }` — the
min/max/mean/sigma summary used for the error and the displacement.
"""
struct Quadruplet
    min::Float64
    max::Float64
    mean::Float64
    sigma::Float64
end

"""
    calculate_stats_on(delta::NTuple{3,Vector{Float32}}, n) -> Quadruplet

Port of `diagnostics.c::calculateStatsOn`.  Over the 3-component displacement
`delta` (`delta[1]/[2]/[3]` parallel `Vector{Float32}`), compute the
min/max/mean and `sigma = sqrt(Σ‖δ‖² / n − mean²)` of the magnitude
`‖δ‖ = sqrt(δx²+δy²+δz²)`.  Faithful to the C: `mean` accumulates `v`,
`sigma` accumulates `v2 = v²` (variance of the magnitude, **not** of the
components).
"""
function calculate_stats_on(delta::NTuple{3,Vector{Float32}}, n::Int)
    if n <= 0
        return Quadruplet(0.0, 0.0, 0.0, 0.0)
    end
    dx, dy, dz = delta
    vmin = typemax(Float64)
    vmax = 0.0
    mean = 0.0
    sigma = 0.0
    @inbounds for i in 1:n
        v2 = Float64(dx[i])^2 + Float64(dy[i])^2 + Float64(dz[i])^2
        v = sqrt(v2)
        vmin = min(v, vmin)
        vmax = max(v, vmax)
        mean += v
        sigma += v2
    end
    mean /= n
    sigma = sqrt(max(0.0, sigma / n - mean * mean))
    return Quadruplet(vmin, vmax, mean, sigma)
end

# C `%03d` iteration index and `%+7.5e` values via Printf (which reproduces
# the C printf formats exactly). NaN/Inf keep the lowercase C spelling.
_fmt_i03(n::Integer) = @sprintf("%03d", n)

function _fmt_e(x::Real)
    v = Float64(x)
    isfinite(v) && return @sprintf("%+7.5e", v)
    isnan(v) && return "+nan"
    return v > 0 ? "+inf" : "-inf"
end

"""
    DIAGNOSTICS_LOG_HEADER

The exact `diagnostics.c::initIterationDiagnostics` header line.  The C source
uses a backslash line-continuation that injects the source indentation
(12 spaces) before `dmps/100`; reproduced verbatim so a parser written against
the C output behaves identically.
"""
const DIAGNOSTICS_LOG_HEADER =
    "Iter\tError min\tError max\tError mean\tError sigma\tError diff\t" *
    "Move dmps\tdmps/10\t            dmps/100\tdmps/1000\t" *
    "Delta min\tDelta max\tDelta mean\tDelta sigma\n"

"""
    init_iteration_diagnostics(path = "diagnostics.log")

Port of `diagnostics.c::initIterationDiagnostics`: (re)create the log file and
write the column header.  Called once before the WVT loop (C `OUTPUT_DIAGNOSTICS`
default on).
"""
function init_iteration_diagnostics(path::AbstractString = "diagnostics.log")
    open(path, "w") do fp
        write(fp, DIAGNOSTICS_LOG_HEADER)
    end
    return nothing
end

"""
    write_iteration_diagnostics(iteration, error::Quadruplet, diff_error,
                                move_mps::NTuple{4}, delta::Quadruplet;
                                path = "diagnostics.log")

Port of `diagnostics.c::writeIterationDiagnostics`: append one row
`%03d` + 13× `%+7.5e` (tab-separated, newline-terminated) to the log.
Column order matches the C exactly: iter, error{min,max,mean,sigma},
errDiff, moveMps[0..3], delta{min,max,mean,sigma}.
"""
function write_iteration_diagnostics(iteration::Int, error::Quadruplet,
                                     diff_error::Real,
                                     move_mps::NTuple{4,Float64},
                                     delta::Quadruplet;
                                     path::AbstractString = "diagnostics.log")
    open(path, "a") do fp
        row = string(
            _fmt_i03(iteration), '\t',
            _fmt_e(error.min), '\t',
            _fmt_e(error.max), '\t',
            _fmt_e(error.mean), '\t',
            _fmt_e(error.sigma), '\t',
            _fmt_e(diff_error), '\t',
            _fmt_e(move_mps[1]), '\t',
            _fmt_e(move_mps[2]), '\t',
            _fmt_e(move_mps[3]), '\t',
            _fmt_e(move_mps[4]), '\t',
            _fmt_e(delta.min), '\t',
            _fmt_e(delta.max), '\t',
            _fmt_e(delta.mean), '\t',
            _fmt_e(delta.sigma), '\n',
        )
        write(fp, row)
    end
    return nothing
end

"""
    write_step_file(particles, param, problem, it; output_diagnostics)

Port of `wvt_relax.c::writeStepFile` (`SAVE_WVT_STEPS`, default **off**).
Writes a snapshot named `"<problem.Name>_NNN"` via the Phase-1 `write_output`
then restores `problem.Name`, exactly as the C temporarily rewrites
`Problem.Name`.  Not verbose (C calls `Write_output(0)`).
"""
function write_step_file(particles::Particles, param::Parameters,
                         problem::ProblemParameters, it::Int;
                         output_diagnostics::Bool = true)
    base = problem.Name
    stepname = string(base, '_', _fmt_i03(it))
    write_output(particles, param, problem;
                 verbose = false,
                 output_diagnostics = output_diagnostics,
                 filename = stepname)
    return stepname
end
