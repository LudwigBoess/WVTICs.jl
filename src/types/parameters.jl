# Parameter, problem and kernel-configuration types.
#
# Ports the C compile-time / global structs from `globals.h`:
#   - `struct Parameters`         -> `Parameters`         (parameter-file values)
#   - `struct ProblemParameters`  -> `ProblemParameters`  (per-problem geometry)
#   - the `DESNNGB`/`NNGBDEV`/`NGBMAX` `#define` cascade -> `KernelConfig`
#
# In the C code DESNNGB/NNGBDEV/NGBMAX are chosen by compile-time `#ifdef`s on
# the selected kernel and `TWO_DIM`. Here they become runtime values held by a
# `KernelConfig`, looked up from the tables below (CLAUDE.md §1.6, globals.h).

"""
    Parameters

Mutable port of the C `struct Parameters` (`globals.h`), populated by
[`read_param_file`](@ref) from the ASCII parameter file.

Field meanings (from `globals.h` comments):

- `Npart`                   : number of particles
- `Maxiter`                 : max WVT iterations
- `MpsFraction`             : move this fraction of the mean particle separation
- `StepReduction`           : force convergence at this rate
- `density_function_correction` : artificial density-model correction factor
                              compensating imperfect WVT convergence (C
                              `Param.BiasCorrection`; legacy parameter-file tag
                              `BiasCorrection` is still accepted)
- `LimitMps`                : 4-element convergence criterion for particle movement
                              (thresholds for fractions 1, 0.1, 0.01, 0.001 of d_mps)
- `MoveFractionMin`         : min fraction of particles moved during redistribution
- `MoveFractionMax`         : max fraction of particles moved during redistribution
- `ProbesFraction`          : redistribution probe fraction
- `RedistributionFrequency` : run redistribution every N iterations
- `LastMoveStep`            : last iteration at which redistribution runs
- `Problem_Flag`            : problem family selector
- `Problem_Subflag`         : problem sub-selector
"""
Base.@kwdef mutable struct Parameters
    Npart::Int = 0
    Maxiter::Int = 0
    MpsFraction::Float64 = 0.0
    StepReduction::Float64 = 0.0
    density_function_correction::Float64 = 0.0
    LimitMps::NTuple{4,Float64} = (0.0, 0.0, 0.0, 0.0)
    MoveFractionMin::Float64 = 0.0
    MoveFractionMax::Float64 = 0.0
    ProbesFraction::Float64 = 0.0
    RedistributionFrequency::Int = 0
    LastMoveStep::Int = 0
    Problem_Flag::Int = 0
    Problem_Subflag::Int = 0
end

"""
    ProblemParameters

Mutable port of the C `struct ProblemParameters` (`globals.h`), filled by the
problem setup (Phase 1+).

- `Name`     : output snapshot file name / problem name
- `Mpart`    : particle mass (from `mpart_from_integral`)
- `Boxsize`  : box dimensions; **`Boxsize[1]` is ALWAYS the largest axis**
               (asserted in the C code; neighbour finding relies on it)
- `Rho_Max`  : maximum model density (for rejection sampling)
- `Periodic` : per-axis periodicity flags
"""
Base.@kwdef mutable struct ProblemParameters
    Name::String = ""
    Mpart::Float64 = 0.0
    Boxsize::NTuple{3,Float64} = (0.0, 0.0, 0.0)
    Rho_Max::Float64 = 0.0
    Periodic::NTuple{3,Bool} = (false, false, false)
end

"""
    KernelType

Enumeration of the SPH kernels selectable in the C build (via `#ifdef`).
`WC10`/`WC12` are listed for completeness but are **not** available in
`SPHKernels.jl` (CLAUDE.md §1.6 / §5) and are unsupported in this port.
"""
@enum KernelType begin
    CubicSpline
    WendlandC2
    WendlandC4
    WendlandC6
    WendlandC8
    WendlandC10   # unsupported (missing from SPHKernels.jl)
    WendlandC12   # unsupported (missing from SPHKernels.jl)
end

# DESNNGB / NNGBDEV table, keyed by (kernel, dimension).
# Values transcribed directly from the `#ifdef` cascade in `globals.h`.
# NGBMAX = DESNNGB * 8, except cubic spline where NGBMAX = DESNNGB * 16.
const _KERNEL_TABLE_3D = Dict{KernelType,Tuple{Int,Float64}}(
    CubicSpline => (50, 0.05),
    WendlandC2  => (64, 0.05),
    WendlandC4  => (200, 0.05),
    WendlandC6  => (295, 0.02),   # C "default" (no flag) branch
    WendlandC8  => (400, 0.05),
    WendlandC10 => (600, 0.05),
    WendlandC12 => (800, 0.05),
)

# 2D values from globals.h: cubic 14/0.01, WC2 16/0.01, otherwise 44/0.01.
# Only Cubic / WC2 have explicit 2D branches; every other kernel falls into
# the shared `#else` (DESNNGB 44, NNGBDEV 0.01) in the C `TWO_DIM` block.
const _KERNEL_TABLE_2D = Dict{KernelType,Tuple{Int,Float64}}(
    CubicSpline => (14, 0.01),
    WendlandC2  => (16, 0.01),
    WendlandC4  => (44, 0.01),
    WendlandC6  => (44, 0.01),
    WendlandC8  => (44, 0.01),
    WendlandC10 => (44, 0.01),
    WendlandC12 => (44, 0.01),
)

"""
    KernelConfig

Runtime replacement for the C compile-time `DESNNGB`/`NNGBDEV`/`NGBMAX`
macros (`globals.h`, CLAUDE.md §1.6).

Fields:
- `kernel`  : selected [`KernelType`]
- `dim`     : 2 or 3 (replaces the `TWO_DIM` compile flag)
- `desnngb` : SPH kernel-weighted desired neighbour count (`DESNNGB`)
- `nngbdev` : tolerance on the kernel-weighted neighbour count (`NNGBDEV`)
- `ngbmax`  : neighbour list cap (`NGBMAX`)

Construct via [`KernelConfig(kernel; dim=3)`](@ref).
"""
struct KernelConfig
    kernel::KernelType
    dim::Int
    desnngb::Int
    nngbdev::Float64
    ngbmax::Int
end

"""
    KernelConfig(kernel::KernelType; dim::Integer = 3)

Build a [`KernelConfig`](@ref), looking up `DESNNGB`/`NNGBDEV` from the
kernel+dimension tables transcribed from `globals.h`. `NGBMAX` is
`DESNNGB * 16` for the cubic spline and `DESNNGB * 8` otherwise.

Throws for `WendlandC10`/`WendlandC12` (not available in `SPHKernels.jl`)
and for `dim` ∉ (2, 3).
"""
function KernelConfig(kernel::KernelType; dim::Integer = 3)
    dim in (2, 3) || throw(ArgumentError("dim must be 2 or 3, got $dim"))
    if kernel === WendlandC10 || kernel === WendlandC12
        throw(ArgumentError("kernel $kernel is not available in SPHKernels.jl " *
                            "and is unsupported in this port (CLAUDE.md §1.6/§5)"))
    end
    table = dim == 3 ? _KERNEL_TABLE_3D : _KERNEL_TABLE_2D
    desnngb, nngbdev = table[kernel]
    ngbmax = kernel === CubicSpline ? desnngb * 16 : desnngb * 8
    return KernelConfig(kernel, Int(dim), desnngb, nngbdev, ngbmax)
end

"""
    default_kernel_config(; dim = 3)

The kernel used by the active C Makefile (`SPH_WC4` → Wendland C4,
DESNNGB 200; CLAUDE.md §1.6). Used as the Phase-0 default.
"""
default_kernel_config(; dim::Integer = 3) = KernelConfig(WendlandC4; dim = dim)
