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
#
# `Printf` is intentionally NOT used (it is a stdlib but is not declared in
# Project.toml; the Phase-3 constraints forbid adding deps).  The C
# `%03d` / `%+7.5e` formats are reproduced by hand below.

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

# --- hand-rolled C printf equivalents (no Printf dep) -----------------------

# C `"%03d"` — minimum 3 digits, zero-padded, with sign for negatives.
function _fmt_i03(n::Integer)
    neg = n < 0
    s = string(abs(Int(n)))
    if length(s) < 3
        s = "0"^(3 - length(s)) * s
    end
    return neg ? "-" * s : s
end

# C `"%+7.5e"` — forced sign, 5 fractional digits, lowercase `e`, exponent
# with explicit sign and ≥2 digits (the width-7 flag never pads here because
# the mantissa+exponent already exceed 7 chars; C does not truncate either).
function _fmt_e(x::Real)
    v = Float64(x)
    if isnan(v)
        return "+nan"
    elseif isinf(v)
        return v > 0 ? "+inf" : "-inf"
    end
    sign = v < 0 ? "-" : "+"
    a = abs(v)
    exp10 = 0
    if a != 0.0
        exp10 = floor(Int, log10(a))
        mant = a / exp10_pow(exp10)
        # guard against log10 rounding pushing mantissa out of [1,10)
        if mant >= 10.0
            mant /= 10.0
            exp10 += 1
        elseif mant < 1.0
            mant *= 10.0
            exp10 -= 1
        end
    else
        mant = 0.0
    end
    # round mantissa to 5 fractional digits; carry can bump it to 10.0
    scaled = round(mant * 1e5)
    if scaled >= 1.0e6
        scaled /= 10.0
        scaled = round(scaled)
        exp10 += 1
    end
    iscaled = Int(scaled)
    intpart = div(iscaled, 100000)
    fracpart = iscaled - intpart * 100000
    fracs = string(fracpart)
    fracs = "0"^(5 - length(fracs)) * fracs
    esign = exp10 < 0 ? "-" : "+"
    eabs = string(abs(exp10))
    if length(eabs) < 2
        eabs = "0"^(2 - length(eabs)) * eabs
    end
    return string(sign, intpart, '.', fracs, 'e', esign, eabs)
end

# 10^e for integer e, exactly for the range that occurs here.
@inline function exp10_pow(e::Int)
    return 10.0^e
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
    write_step_file(particles, param, problem, it; kernel, output_diagnostics)

Port of `wvt_relax.c::writeStepFile` (`SAVE_WVT_STEPS`, default **off**).
Writes a snapshot named `"<problem.Name>_NNN"` via the Phase-1 `write_output`
then restores `problem.Name`, exactly as the C temporarily rewrites
`Problem.Name`.  Not verbose (C calls `Write_output(0)`).
"""
function write_step_file(particles::Particles, param::Parameters,
                         problem::ProblemParameters, it::Int;
                         kernel::KernelConfig = default_kernel_config(),
                         output_diagnostics::Bool = true)
    base = problem.Name
    stepname = string(base, '_', _fmt_i03(it))
    write_output(particles, param, problem;
                 verbose = false, kernel = kernel,
                 output_diagnostics = output_diagnostics,
                 filename = stepname)
    return stepname
end
