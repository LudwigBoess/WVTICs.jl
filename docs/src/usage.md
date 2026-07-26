```@meta
CurrentModule = WVTICs
```

# Usage

## The driver

[`make_sph_wvtics`](@ref) is the top-level entry point. It reads a parameter
file and runs the whole pipeline:

```
read_param_file → setup → make_positions! → make_ids!
  → regularise_sph_particles! → make_velocities! → make_temperatures!
  → make_magnetic_fields! → make_post_processing! → write_output
```

```julia
using WVTICs
particles = make_sph_wvtics("ics.par"; kernel = default_kernel_config(),
                            verbose = true)
```

It returns the relaxed [`Particles`](@ref) container and writes the snapshot to
the problem's name in the current working directory.

The individual stages are also callable directly (all internal) if you want to
build a pipeline by hand — see the [API](@ref) reference.

## Parameter files

Two formats are accepted; [`read_param_file`](@ref) dispatches on the
extension. A `.toml` file is parsed as a flat TOML table; any other extension
uses the ASCII `tag value` parser (comment character `%`, one value per line).
Both produce the same [`Parameters`](@ref).

### Tags

| Tag | Type | Meaning |
|---|---|---|
| `Npart` | Int | number of particles (**required**) |
| `Maxiter` | Int | maximum relaxation iterations (**required**) |
| `Problem_Flag` | Int | problem family (**required**, see [Problems](@ref)) |
| `Problem_Subflag` | Int | problem sub-selector (**required**) |
| `MpsFraction` | Float / `"auto"` | initial WVT step as a fraction of the mean particle spacing; see below |
| `StepReduction` | Float | step shrink factor applied on convergence stalls (e.g. `0.95`) |
| `LimitMps`, `LimitMps10`, `LimitMps100`, `LimitMps1000` | Float | convergence thresholds for the fraction of particles moving more than `1, 0.1, 0.01, 0.001 ×` the mean spacing (map onto `LimitMps[1..4]`) |
| `MoveFractionMin`, `MoveFractionMax` | Float | min/max fraction of particles moved per redistribution |
| `ProbesFraction` | Float | redistribution probe budget fraction |
| `RedistributionFrequency` | Int | run redistribution every N iterations |
| `LastMoveStep` | Int | last iteration at which redistribution runs |
| `density_function_correction` | Float | artificial density-model correction (legacy tag `BiasCorrection` still accepted) |
| `DesNumNgb` | Int | override the target SPH neighbour count `DESNNGB`; `0` = use the kernel default |

Required tags (`Npart`, `Maxiter`, `Problem_Flag`, `Problem_Subflag`) error if
absent. Every other tag defaults to `0`/`0.0`. Unknown tags (e.g. the C tool's
`PNG_Filename`) are ignored.

### `MpsFraction` auto-calibration

`MpsFraction` sets the initial step size. Omit it, set it to `0.0`, or set it to
the string `"auto"` to let the relaxation calibrate the step at startup: it
seeds from an analytic value (≈`0.8` in 3D, ≈`0.1` in 2D) and runs a short
log-space bisection into a stable band, so no per-problem or per-`N` tuning is
needed. A finite positive value pins the exact legacy formula
`step = 1 / (N^{1/dim} · MpsFraction)` (the C reference used `5.0` in 3D,
`0.1` in 2D).

### Example (TOML)

```toml
Npart   = 2000
Maxiter = 10
MpsFraction   = 5.0
StepReduction = 0.95
LimitMps     = -1.0
LimitMps10   = -1.0
LimitMps100  = -1.0
LimitMps1000 = 1.0
MoveFractionMin         = 0.01
MoveFractionMax         = 0.01
ProbesFraction          = 0.1
RedistributionFrequency = 5
LastMoveStep            = 10
density_function_correction = 0.0
DesNumNgb = 0
Problem_Flag    = 0
Problem_Subflag = 0
```

The same tags in ASCII form (`Npart 2000`, one `tag value` per line, `%`
comments) is what `ics.par` uses.

## Choosing a kernel

The SPH kernel, its dimension, and its `DESNNGB`/`NNGBDEV`/`NGBMAX` are held in
a [`KernelConfig`](@ref) passed to the driver. Build one from a built-in kernel
type (the neighbour-count tables are looked up automatically):

```julia
using WVTICs: KernelConfig, CubicSpline, WendlandC2, WendlandC4, WendlandC6, WendlandC8
kc = KernelConfig(WendlandC4; dim = 3)      # DESNNGB = 200
kc2 = KernelConfig(CubicSpline; dim = 2)    # DESNNGB = 14
```

or from **any** `SPHKernels.jl` kernel instance (so kernels beyond the built-in
set work with no code change — `desnngb` is then required):

```julia
import SPHKernels
kc = KernelConfig(SPHKernels.WendlandC6(Float64, 3); desnngb = 295)
```

[`default_kernel_config`](@ref) returns Wendland C4 in 3D (the C Makefile
default). A parameter-file `DesNumNgb > 0` overrides the neighbour count via
[`with_desnngb`](@ref). `KernelConfig` is parametric on the concrete kernel
type, so kernel evaluation on the hot path is statically dispatched.

## Position sampling

[`make_positions!`](@ref) supports two modes via the `sampling` keyword:

- `RejectionSampling` (default) — von-Neumann rejection against the target
  density, so the initial draw already tracks ``\rho``;
- `UniformSampling` — plain uniform positions in the box.

Sampling uses one seeded `Xoshiro` RNG per chunk; a run is reproducible for a
fixed thread count but is not bit-identical to the original C `erand48` stream.

## Output

[`write_output`](@ref) writes a Gadget `SnapFormat 2` snapshot: a `HEAD` block,
a self-describing `INFO` block, then the data blocks

```
POS, VEL, ID, HSML, RHO, U, BFLD, RHOM [, REDI]
```

`RHOM` (model density) and `REDI` (redistribution flag) are custom blocks; `REDI`
is written only when diagnostics are on. Read the snapshot back with GadgetIO:

```julia
using GadgetIO
h   = read_header("IC_Constant_Density")
pos = read_block("IC_Constant_Density", "POS"; parttype = 0)
rho = read_block("IC_Constant_Density", "RHO"; parttype = 0)
```

## Diagnostics

When `output_diagnostics` is on (the default), the relaxation appends one row
per iteration to `diagnostics.log`: the min/max/mean/σ of the relative density
error, the error change, the four `moveMps` fractions, and the min/max/mean/σ of
the per-particle displacement. Watch the mean error fall and the `moveMps`
fractions shrink to gauge convergence.
```
