# Phase 2 — thin SPH-kernel adapter (ports `kernel.c` / `kernel.h`).
#
# Maps the project's `KernelConfig` (Cubic / WC2 / WC4 / WC6 / WC8) to kernel
# value / derivative / self-bias-correction evaluations (all via SPHKernels.jl).
#
# DESIGN NOTE (kernel value/derivative vs. bias correction — see PORT_STATUS.md):
#
#   * Kernel VALUE and DERIVATIVE are now evaluated through the user's
#     `SPHKernels.jl` package.  Historically WVTICs reimplemented the
#     `kernel.c` polynomials directly because `SPHKernels.WendlandC4`'s
#     `kernel_deriv` carried a `-288/3` coefficient where the analytic
#     Wendland-C4 derivative (and `kernel.c::sph_kernel_derivative_WC4`)
#     require `-280/3`.  That defect has been **fixed upstream** in
#     `SPHKernels/src/wendland/C4.jl` (now `-280/3`, its own oracles
#     corrected, suite green).  With SPHKernels analytically C-correct, the
#     C-polynomial reimplementation is redundant and was removed; value and
#     derivative dispatch to `SPHKernels.kernel_value` / `kernel_deriv`.
#
#     Normalisation/sign match `kernel.c`/`sph.c` exactly:
#       - `SPHKernels.kernel_value(k,u,h⁻¹)  = norm·h⁻¹^dim · P(u)`
#       - `SPHKernels.kernel_deriv(k,u,h⁻¹)  = h⁻¹·norm·h⁻¹^dim · P'(u)`
#     i.e. the same `norm·h⁻¹^dim·poly` / `…·h⁻¹·deriv_poly` structure the
#     C code uses.  Every SPHKernels `kernel_deriv` polynomial is negative on
#     (0,1) — identical sign to the C `sph_kernel_derivative_*` — so
#     `density.jl`'s consumer `dRhodHsml += -m·(3/h·wk + r/h·dwk)`
#     (mirroring `sph.c`) is unchanged: it expects the C-signed (negative)
#     derivative, which is exactly what SPHKernels now returns.
#
#   * The kernel SELF-BIAS correction (Dehnen&Aly / Cullen&Dehnen) on the
#     density path is now routed through `SPHKernels.bias_correction`.
#     This is an **intentional, user-directed divergence** from the C
#     `kernel.c::bias_correction_WC*` formula (which the project previously
#     reproduced bit-faithfully).  The two differ definitionally:
#       1. `SPHKernels.bias_correction(k, ρ, m, h⁻¹, n)` returns the *already
#          corrected density* `ρ - δρ` (NOT a separate Δ added to ρ as the C
#          `sph.c` line 200 does), using the kernel's OWN normalisation, for
#          WC2/WC4/WC6 only (Cubic/WC8 pass ρ through unchanged), and
#          additionally clamps the WC6 term to `< 0.2·ρ`.
#       2. C `bias_correction_WC4` evaluates the term with the **WC2** kernel
#          norm at u=0 (the C WC4→WC2 quirk, CLAUDE.md §1.6/§5); SPHKernels
#          uses the proper **WC4** norm `495/32π`.
#       3. Sign: C *adds* (`+0.01342…` for WC4); SPHKernels *subtracts* (so
#          its net WC4 effect is the opposite sign of the C term).
#     The user decided to adopt the SPHKernels genuine self-bias on the
#     density path (its WC4-derivative defect having been fixed upstream
#     earlier in this project).  The C-faithful term is retired from the
#     density path.  See PORT_STATUS.md "Phase-2 follow-up: density-path
#     kernel self-bias → SPHKernels" for the full rationale and the test
#     re-derivation.
#
#   `SPHKernels.bias_correction` in 3D (n = DESNNGB):
#
#     Cubic/WC8 : ρ unchanged (SPHKernels defines no correction — fallback)
#     WC2  : ρ − 0.0294 ·(n·0.01)^-0.977 · m · (21/2π   · h⁻³)
#     WC4  : ρ − 0.01342·(n·0.01)^-1.579 · m · (495/32π  · h⁻³)
#     WC6  : ρ − 0.0116 ·(n·0.01)^-2.236 · m · (1365/64π · h⁻³)
#            (only if that δρ < 0.2·ρ — SPHKernels' WC6 clamp)
#
#   For dim≠3 SPHKernels uses the matching DESNNGB and the 2D/1D `kernel_norm`
#   power (`h⁻²`/`h⁻¹`); SPHKernels applies no 2D-vs-3D special-casing, so
#   neither does this adapter (it forwards `kc.dim`).  This differs from the
#   retired C-faithful path, which forced 2D bias = 0.
#
# §4b: `h_inv` is computed once per particle by the caller and threaded
# through; the per-neighbour kernel calls take `(u, h_inv)` and pass `u` and
# the precomputed `h_inv` straight to SPHKernels — `1/h` is never recomputed
# per call, and raw `(r,h)` is never passed (which would force SPHKernels to
# recompute `h_inv` internally).

"""
    sph_kernel_type(kc::KernelConfig)

Map a [`KernelConfig`](@ref) to the corresponding `SPHKernels.jl` kernel
*instance* with the correct dimension.  This is the single source of the
project-`KernelType` → SPHKernels mapping, reused on the hot path by
[`sph_kernel`](@ref) / [`sph_kernel_deriv`](@ref) and available to external
callers wanting the library object.

Restricted to Cubic / WC2 / WC4 / WC6 / WC8 (C10/C12 are out of scope and
already rejected by `KernelConfig`).
"""
function sph_kernel_type(kc::KernelConfig)
    d = kc.dim
    k = kc.kernel
    k === CubicSpline && return SPHKernels.Cubic(Float64, d)
    k === WendlandC2  && return SPHKernels.WendlandC2(Float64, d)
    k === WendlandC4  && return SPHKernels.WendlandC4(Float64, d)
    k === WendlandC6  && return SPHKernels.WendlandC6(Float64, d)
    k === WendlandC8  && return SPHKernels.WendlandC8(Float64, d)
    throw(ArgumentError("kernel $(kc.kernel) is unsupported in this port " *
                        "(CLAUDE.md §1.6/§5)"))
end

"""
    sph_kernel(kc::KernelConfig, r, h_inv) -> Float64

SPH kernel value `W(r,h)` for the configured kernel, evaluated via
`SPHKernels.kernel_value`.  `h_inv = 1/h` is supplied by the caller (computed
once per particle, §4b); `r` is the (already minimum-image–corrected) pair
distance.  `u = r·h_inv` and `h_inv` are passed straight to SPHKernels (no
per-call `1/h`).  Returns 0 for `r ≥ h`.

Normalisation/sign match `kernel.c::sph_kernel_*` exactly:
`SPHKernels.kernel_value(k,u,h⁻¹) = norm·h⁻¹^dim·P(u)`.
"""
@inline function sph_kernel(kc::KernelConfig, r::Real, h_inv::Real)
    hi = Float64(h_inv)
    u = Float64(r) * hi
    k = kc.kernel
    d = kc.dim
    # Branch on KernelType so each arm is type-stable (concrete kernel struct,
    # `Float64` return); union-splitting is contained here, off the call path.
    if k === CubicSpline
        return SPHKernels.kernel_value(SPHKernels.Cubic(Float64, d), u, hi)
    elseif k === WendlandC2
        return SPHKernels.kernel_value(SPHKernels.WendlandC2(Float64, d), u, hi)
    elseif k === WendlandC4
        return SPHKernels.kernel_value(SPHKernels.WendlandC4(Float64, d), u, hi)
    elseif k === WendlandC6
        return SPHKernels.kernel_value(SPHKernels.WendlandC6(Float64, d), u, hi)
    elseif k === WendlandC8
        return SPHKernels.kernel_value(SPHKernels.WendlandC8(Float64, d), u, hi)
    end
    throw(ArgumentError("unsupported kernel $(kc.kernel)"))
end

"""
    sph_kernel_deriv(kc::KernelConfig, r, h_inv) -> Float64

SPH kernel derivative `dW/dr (r,h)` for the configured kernel
(`sph_kernel_derivative` in `kernel.c`), evaluated via
`SPHKernels.kernel_deriv`.  `h_inv = 1/h` supplied by the caller (§4b);
`u = r·h_inv` and `h_inv` are passed straight through (no per-call `1/h`).

`SPHKernels.kernel_deriv(k,u,h⁻¹) = h⁻¹·norm·h⁻¹^dim·P'(u)`, with `P'` negative
on (0,1) — same sign as `kernel.c::sph_kernel_derivative_*` — so this matches
`density.jl`'s `dRhodHsml += -m·(3/h·wk + r/h·dwk)` consumer (which mirrors
`sph.c` and expects the C-signed, negative derivative).  Returns 0 for `r ≥ h`.
"""
@inline function sph_kernel_deriv(kc::KernelConfig, r::Real, h_inv::Real)
    hi = Float64(h_inv)
    u = Float64(r) * hi
    k = kc.kernel
    d = kc.dim
    if k === CubicSpline
        return SPHKernels.kernel_deriv(SPHKernels.Cubic(Float64, d), u, hi)
    elseif k === WendlandC2
        return SPHKernels.kernel_deriv(SPHKernels.WendlandC2(Float64, d), u, hi)
    elseif k === WendlandC4
        return SPHKernels.kernel_deriv(SPHKernels.WendlandC4(Float64, d), u, hi)
    elseif k === WendlandC6
        return SPHKernels.kernel_deriv(SPHKernels.WendlandC6(Float64, d), u, hi)
    elseif k === WendlandC8
        return SPHKernels.kernel_deriv(SPHKernels.WendlandC8(Float64, d), u, hi)
    end
    throw(ArgumentError("unsupported kernel $(kc.kernel)"))
end

"""
    sph_bias_correction(kc::KernelConfig, density, mpart, h_inv) -> Float64

Apply the **kernel self-bias correction** (Dehnen&Aly 2012 eq. 18+19 /
Cullen&Dehnen) to an SPH density estimate, returning the **corrected
density** (NOT a Δ).  This is the genuine kernel self-bias, routed through
`SPHKernels.bias_correction` — an **intentional, user-directed divergence**
from the C `kernel.c::bias_correction_WC*` formula (which is C-faithful but
carries an opposite WC4 sign, a WC4→WC2-norm quirk, adds a separate Δ to ρ,
and lacks SPHKernels' WC6 `<0.2ρ` clamp).  See PORT_STATUS.md
"Phase-2 follow-up: density-path kernel self-bias → SPHKernels".

`SPHKernels.bias_correction(k, ρ, m, h⁻¹, n)` returns `ρ − δρ`:

  * Cubic / WC8                ⇒ ρ unchanged (SPHKernels defines no
    correction for B-splines / WC8 — documented fallback, matches
    SPHKernels' own behaviour)
  * WC2  ⇒ ρ − 0.0294 ·(n·0.01)^-0.977 · m · 𝒩_WC2(h⁻¹)
  * WC4  ⇒ ρ − 0.01342·(n·0.01)^-1.579 · m · 𝒩_WC4(h⁻¹)
  * WC6  ⇒ ρ − 0.0116 ·(n·0.01)^-2.236 · m · 𝒩_WC6(h⁻¹)  (only if
    that term `< 0.2·ρ`; SPHKernels' WC6 clamp)

`n = kc.desnngb` (DESNNGB, the desired neighbour count); `𝒩_X(h⁻¹) =
norm_X · h⁻¹^dim` is `SPHKernels.kernel_norm` (so the `dim`-dependent power
is handled by SPHKernels — for `dim≠3` the matching DESNNGB and 2D/1D norm
are used; SPHKernels itself applies no 2D/3D guard).  `Float64` return.

The `if k === …` chain is union-split on the `KernelType` enum so each arm
binds a concrete SPHKernels kernel struct (no boxed captures / dynamic
dispatch on the hot path), mirroring [`sph_kernel`](@ref).
"""
@inline function sph_bias_correction(kc::KernelConfig, density::Real,
                                     mpart::Real, h_inv::Real)
    rho = Float64(density)
    hi = Float64(h_inv)
    m = Float64(mpart)
    n = kc.desnngb                                            # DESNNGB
    d = kc.dim
    k = kc.kernel
    if k === CubicSpline
        return SPHKernels.bias_correction(SPHKernels.Cubic(Float64, d),
                                          rho, m, hi, n)
    elseif k === WendlandC2
        return SPHKernels.bias_correction(SPHKernels.WendlandC2(Float64, d),
                                          rho, m, hi, n)
    elseif k === WendlandC4
        return SPHKernels.bias_correction(SPHKernels.WendlandC4(Float64, d),
                                          rho, m, hi, n)
    elseif k === WendlandC6
        return SPHKernels.bias_correction(SPHKernels.WendlandC6(Float64, d),
                                          rho, m, hi, n)
    elseif k === WendlandC8
        return SPHKernels.bias_correction(SPHKernels.WendlandC8(Float64, d),
                                          rho, m, hi, n)
    end
    throw(ArgumentError("unsupported kernel $(kc.kernel)"))
end
