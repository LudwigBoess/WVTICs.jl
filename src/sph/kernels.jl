# Thin SPH-kernel adapter over SPHKernels.jl.
#
# Maps the project's `KernelConfig` (Cubic / WC2 / WC4 / WC6 / WC8) to kernel
# value, derivative, and self-bias-correction evaluations on the stored
# `kc.kernel` instance:
#   * value      -> `SPHKernels.kernel_value(k, u, h⁻¹)  = norm·h⁻¹^dim·P(u)`
#   * derivative -> `SPHKernels.kernel_deriv(k, u, h⁻¹)  = h⁻¹·norm·h⁻¹^dim·P'(u)`,
#     with `P'` negative on (0,1), matching `density.jl`'s
#     `dRhodHsml += -m·(3/h·wk + r/h·dwk)` consumer.
#   * self-bias  -> `SPHKernels.bias_correction`, applied on the density path
#     in 3D only (see below).
#
# `h_inv = 1/h` is computed once per particle by the caller and threaded
# through; the per-neighbour calls pass `u = r·h_inv` and `h_inv` straight to
# SPHKernels (no per-call `1/h`).

"""
    sph_kernel(kc::KernelConfig, r, h_inv) -> Float64

SPH kernel value `W(r,h)` for the configured kernel, via
`SPHKernels.kernel_value` on the stored `kc.kernel` instance (built once at
[`KernelConfig`](@ref) construction).  `h_inv = 1/h` is supplied by the caller
(computed once per particle); `r` is the (already minimum-image–corrected) pair
distance.  `u = r·h_inv` and `h_inv` are passed straight to SPHKernels.  Returns
0 for `r ≥ h`.
"""
@inline function sph_kernel(kc::KernelConfig, r::Real, h_inv::Real)
    hi = Float64(h_inv)
    u = Float64(r) * hi
    return SPHKernels.kernel_value(kc.kernel, u, hi)
end

"""
    sph_kernel_deriv(kc::KernelConfig, r, h_inv) -> Float64

SPH kernel derivative `dW/dr (r,h)` for the configured kernel, via
`SPHKernels.kernel_deriv` on the stored `kc.kernel` instance.  `h_inv = 1/h`
supplied by the caller; `u = r·h_inv` and `h_inv` are passed straight through.

`SPHKernels.kernel_deriv(k,u,h⁻¹) = h⁻¹·norm·h⁻¹^dim·P'(u)`, with `P'` negative
on (0,1), matching `density.jl`'s `dRhodHsml += -m·(3/h·wk + r/h·dwk)` consumer.
Returns 0 for `r ≥ h`.
"""
@inline function sph_kernel_deriv(kc::KernelConfig, r::Real, h_inv::Real)
    hi = Float64(h_inv)
    u = Float64(r) * hi
    return SPHKernels.kernel_deriv(kc.kernel, u, hi)
end

"""
    sph_bias_correction(kc::KernelConfig, density, mpart, h_inv) -> Float64

Apply the kernel self-bias correction (Dehnen&Aly 2012 eq. 18+19 /
Cullen&Dehnen) to an SPH density estimate, returning the corrected density
(NOT a Δ), via `SPHKernels.bias_correction` on the stored `kc.kernel` instance.

`SPHKernels.bias_correction(k, ρ, m, h⁻¹, n)` returns `ρ − δρ`:

  * Cubic / WC8 ⇒ ρ unchanged (no correction defined)
  * WC2  ⇒ ρ − 0.0294 ·(n·0.01)^-0.977 · m · 𝒩_WC2(h⁻¹)
  * WC4  ⇒ ρ − 0.01342·(n·0.01)^-1.579 · m · 𝒩_WC4(h⁻¹)
  * WC6  ⇒ ρ − 0.0116 ·(n·0.01)^-2.236 · m · 𝒩_WC6(h⁻¹)  (only if
    that term `< 0.2·ρ`)

`n = kc.desnngb` (DESNNGB, the desired neighbour count); `𝒩_X(h⁻¹) =
norm_X · h⁻¹^dim` is `SPHKernels.kernel_norm`.  `Float64` return.

The correction is applied only for `dim == 3`; for `dim != 3` `ρ` is returned
unchanged, since the Dehnen&Aly coefficients are fitted to the 3D self-bias
relation and have no valid 2D analogue.
"""
@inline function sph_bias_correction(kc::KernelConfig, density::Real,
                                     mpart::Real, h_inv::Real)
    rho = Float64(density)
    hi = Float64(h_inv)
    m = Float64(mpart)
    n = kc.desnngb                                            # DESNNGB
    d = kc.dim
    # 2D/1D: no kernel self-bias correction. The Dehnen&Aly / Cullen&Dehnen
    # coefficients (0.01342, exponent -1.579, …) are fitted for the 3D SPH
    # self-bias relation and have no valid 2D analogue; applying them in 2D
    # (3D coefficients × 2D `kernel_norm` × 2D DESNNGB) would inject a
    # spurious density bias the WVT relaxation cannot anneal away. Gate the
    # correction to `dim == 3`; for other dims return the density unchanged.
    if d != 3
        return rho
    end
    return SPHKernels.bias_correction(kc.kernel, rho, m, hi, n)
end
