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
#   SPHKernels itself applies no 2D-vs-3D special-casing on the bias term.
#   This adapter, however, **gates the correction to `dim == 3`** (returns
#   ρ unchanged for `dim != 3`) — the C-faithful behaviour: each C
#   `kernel.c::bias_correction_WC{2,4,6}` opens with
#   `#ifdef TWO_DIM return 0.0;`, so the self-bias term is a 3D-only
#   correction (the Dehnen&Aly coefficients are fitted to the 3D relation
#   and have no valid 2D analogue).  Forwarding the 3D-fitted term into a
#   2D run (3D coefficients × the 2D `kernel_norm` `norm·h⁻²` × the 2D
#   DESNNGB) injects a spurious, h-dependent density bias that the WVT
#   relaxation — whose entire 2D tuning assumes the *unbiased* raw 2D
#   estimator (C `bias = 0`) — cannot anneal away: the 2D-specific
#   density-normalisation defect behind the "2D constant-density
#   convergence QUALITY" open issue.  Restoring the C 2D=0 gate makes the
#   2D density path C-byte-faithful while leaving 3D byte-identical (the
#   `dim == 3` arm and its SPHKernels call are unchanged).  This supersedes
#   the earlier "intentional divergence: 2D applies the correction" note —
#   that divergence was the unfixed-quality root cause; the WC4-sign /
#   WC4→WC2-norm / Δ-vs-corrected-ρ / WC6-clamp 3D rationale is unchanged.
#
# §4b: `h_inv` is computed once per particle by the caller and threaded
# through; the per-neighbour kernel calls take `(u, h_inv)` and pass `u` and
# the precomputed `h_inv` straight to SPHKernels — `1/h` is never recomputed
# per call, and raw `(r,h)` is never passed (which would force SPHKernels to
# recompute `h_inv` internally).

"""
    sph_kernel(kc::KernelConfig, r, h_inv) -> Float64

SPH kernel value `W(r,h)` for the configured kernel, evaluated **directly via
`SPHKernels.kernel_value`** on the stored `kc.kernel` instance (the project no
longer re-implements the kernel polynomials or their dispatch, and no longer
re-allocates a kernel object per call — the instance is built once at
[`KernelConfig`](@ref) construction and reused here).  `h_inv = 1/h` is supplied
by the caller (computed once per particle, §4b); `r` is the (already
minimum-image–corrected) pair distance.  `u = r·h_inv` and `h_inv` are passed
straight to SPHKernels (no per-call `1/h`).  Returns 0 for `r ≥ h`.

Normalisation/sign match `kernel.c::sph_kernel_*` exactly:
`SPHKernels.kernel_value(k,u,h⁻¹) = norm·h⁻¹^dim·P(u)`.
"""
@inline function sph_kernel(kc::KernelConfig, r::Real, h_inv::Real)
    hi = Float64(h_inv)
    u = Float64(r) * hi
    return SPHKernels.kernel_value(kc.kernel, u, hi)
end

"""
    sph_kernel_deriv(kc::KernelConfig, r, h_inv) -> Float64

SPH kernel derivative `dW/dr (r,h)` for the configured kernel
(`sph_kernel_derivative` in `kernel.c`), evaluated **directly via
`SPHKernels.kernel_deriv`** on the stored `kc.kernel` instance — no
re-defined polynomial / dispatch and no per-call re-allocation.  `h_inv = 1/h`
supplied by the caller (§4b); `u = r·h_inv` and `h_inv` are passed straight
through (no per-call `1/h`).

`SPHKernels.kernel_deriv(k,u,h⁻¹) = h⁻¹·norm·h⁻¹^dim·P'(u)`, with `P'` negative
on (0,1) — same sign as `kernel.c::sph_kernel_derivative_*` — so this matches
`density.jl`'s `dRhodHsml += -m·(3/h·wk + r/h·dwk)` consumer (which mirrors
`sph.c` and expects the C-signed, negative derivative).  Returns 0 for `r ≥ h`.
"""
@inline function sph_kernel_deriv(kc::KernelConfig, r::Real, h_inv::Real)
    hi = Float64(h_inv)
    u = Float64(r) * hi
    return SPHKernels.kernel_deriv(kc.kernel, u, hi)
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
norm_X · h⁻¹^dim` is `SPHKernels.kernel_norm`.  `Float64` return.

**Dimension gate (C-faithful, the 2D-quality fix):** the correction is
applied **only for `dim == 3`**.  For `dim != 3` `ρ` is returned
**unchanged** — matching the C `kernel.c::bias_correction_WC{2,4,6}`,
each of which begins `#ifdef TWO_DIM return 0.0;` (the Dehnen&Aly
coefficients are fitted to the 3D self-bias relation and have no valid
2D analogue; the C author hard-codes the 2D term to zero).  This closes
the only C-vs-Julia divergence in the 2D density path — see the inline
comment on the `if d != 3` guard and PORT_STATUS.md's
"2D constant-density convergence QUALITY" issue.  3D is byte-identical
(the `dim == 3` SPHKernels call is unchanged).

The kernel object is the stored `kc.kernel` instance and the correction itself
is `SPHKernels.bias_correction` — the project does not re-implement the
Dehnen&Aly term (mirroring [`sph_kernel`](@ref) / [`sph_kernel_deriv`](@ref)).
"""
@inline function sph_bias_correction(kc::KernelConfig, density::Real,
                                     mpart::Real, h_inv::Real)
    rho = Float64(density)
    hi = Float64(h_inv)
    m = Float64(mpart)
    n = kc.desnngb                                            # DESNNGB
    d = kc.dim
    # ── 2D/1D: NO kernel self-bias correction (C-faithful) ───────────────
    # The C `kernel.c::bias_correction_WC{2,4,6}` each open with
    # `#ifdef TWO_DIM return 0.0;` — i.e. the self-bias term is applied
    # **only in 3D**; in a `TWO_DIM` build the correction is identically
    # zero (and `bias_correction` for Cubic/WC8 is `0.0` in every
    # dimension).  The Dehnen&Aly / Cullen&Dehnen coefficients
    # (`0.01342`, exponent `-1.579`, …) are fitted for the *3D* SPH
    # self-bias relation and have NO valid 2D analogue — which is exactly
    # why the C author hard-codes the 2D term to zero.  `SPHKernels`
    # itself applies no 2D/3D guard, so it would (incorrectly for this
    # port's purpose) evaluate that 3D-fitted term with the 2D
    # `kernel_norm` (`norm·h⁻²`) and the 2D DESNNGB, injecting a spurious
    # density bias the WVT relaxation (whose entire 2D tuning — `d_mps`,
    # model-hsml, step annealing — assumes the *unbiased* raw 2D
    # estimator, C `bias=0`) cannot anneal away: a 2D-specific
    # density-normalisation defect, the documented "2D constant-density
    # convergence QUALITY" open issue.  Gate the correction to `dim == 3`
    # so the 2D density path is C-byte-faithful (raw estimator, no
    # correction).  This is the SINGLE C-vs-Julia divergence remaining in
    # the entire 2D density/kernel path (every other 2D constant/exponent
    # — kernel norms `1/h²`, `wkNgb = π·wk·h²`, `ρ = m·wk`, model-hsml
    # `√(WVTNNGB·m/ρ/π)`, `norm_hsml`, `d_mps`, `npart_1D = N^(1/2)` — was
    # audited byte-identical to the C `#ifdef TWO_DIM` branches).
    #
    # 3D path is **byte-identical**: `d == 3` falls straight through to
    # the `SPHKernels.bias_correction` call below on the stored `kc.kernel`
    # instance (same SPHKernels call, same args, same result — 3D
    # constant-density still converges to errMean ≈ 5.4e-4 / ×345).  The
    # dimension guard is type-stable (`d::Int` compared to a literal;
    # returns the already-`Float64` `rho`).
    if d != 3
        return rho
    end
    return SPHKernels.bias_correction(kc.kernel, rho, m, hi, n)
end
