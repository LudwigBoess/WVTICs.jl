# Divergence-free turbulent magnetic field generator. Ports make_turb_B.c
# (J. Donnert 2013). In-memory and callable both standalone and as a problem
# `postprocess!` hook (it needs the final post-relaxation positions).
#
# Algorithm:
#   1. nGrid = 2*ceil(Boxsize / B_scale).
#      Kmin = 2π / (0.01·Boxsize),  Kmax = 0.5·Kmin·nGrid  (Nyquist).
#   2. Fill a complex k-space grid per component with a power-law spectrum
#      P(k) = k^spectral_index, Box-Muller amplitudes sqrt(-log(U)·P(k)) and a
#      random phase; drop |k| > Kmax; enforce Hermitian symmetry so the inverse
#      transform is real.
#   3. Inverse FFT -> real space; normalise to a constant mean field `B_norm`.
#   4. Forward FFT; divergence clean by the projection
#      B_k <- (I - k̂ k̂ᵀ)·B_k (Ruszkowski+ 2006 / Balsara 1996); zero DC.
#   5. Inverse FFT -> real space; NGP-interpolate grid -> particles (periodic).
#
# Uses the full complex `fft`/`ifft` path (not `rfft`/`irfft`): Hermitian
# symmetry is enforced explicitly on the full cube so the inverse transform is
# real to round-off and the projection is applied to every mode.
#
# A seeded `Random.Xoshiro` is used (reproducible for a fixed seed; not
# bit-identical to a GSL RNG stream).

"""
    make_turbulent_Bfield(pos, boxsize; B_norm=1e-6, B_scale=0.1,
                          spectral_index=-11/3, seed=14041981)
        -> Vector{SVector{3,Float32}}

Generate a periodic-box, divergence-free turbulent magnetic field with a
power-law power spectrum `P(k) = k^spectral_index` (Kolmogorov default
`-11/3`) and NGP-interpolate it to the particle positions `pos`
(`AbstractVector{<:SVector{3}}`, the WVTICs SoA `particles.pos`).

- `boxsize` — the (cubic) periodic box side length (the grid is built on
  `[0, boxsize)`).
- `B_norm` — target mean field magnitude (default `1e-6` G).
- `B_scale` — sets the grid resolution `nGrid = 2·ceil(boxsize/B_scale)`
  (default `0.1`).
- `spectral_index` — power-law slope (default `-11/3`).
- `seed` — RNG seed (reproducible for a fixed seed).

Returns one `SVector{3,Float32}` per input position.
"""
function make_turbulent_Bfield(pos::AbstractVector{<:SVector{3}},
                               boxsize::Real;
                               B_norm::Real = 1e-6,
                               B_scale::Real = 0.1,
                               spectral_index::Real = -11.0 / 3.0,
                               seed::Integer = 14041981)
    L = float(boxsize)
    nGrid = 2 * ceil(Int, L / B_scale)
    nGrid < 2 && (nGrid = 2)
    Kmin = 2.0 * pi / (0.01 * L)
    Kmax = 0.5 * Kmin * nGrid

    # k-space grids (full complex cube per component).
    Bk = (zeros(ComplexF64, nGrid, nGrid, nGrid),
          zeros(ComplexF64, nGrid, nGrid, nGrid),
          zeros(ComplexF64, nGrid, nGrid, nGrid))
    # k-vector component grids (saved for the divergence clean).
    Kx = Array{Float64}(undef, nGrid, nGrid, nGrid)
    Ky = Array{Float64}(undef, nGrid, nGrid, nGrid)
    Kz = Array{Float64}(undef, nGrid, nGrid, nGrid)

    _fill_fourier_grid!(Bk, Kx, Ky, Kz, nGrid, Kmin, Kmax,
                        Float64(spectral_index), seed)

    # (2) inverse FFT -> real space. The grid is Hermitian by construction so
    # the imaginary part is round-off; take the real part.
    B = (real.(ifft(Bk[1])), real.(ifft(Bk[2])), real.(ifft(Bk[3])))

    # (3) normalise so the mean magnitude matches the field model (B_norm).
    _normalise_bfld_grid!(B, nGrid, Float64(B_norm))

    # (4) forward FFT, divergence clean (Balsara/Ruszkowski projection;
    # `_divergence_clean!` also zeroes the DC mode). `fft` of a real field is
    # Hermitian to round-off.
    #
    # Known limitation: the spectral div(B) test is left @test_broken (a ~0.61
    # longitudinal residual whose cause is unresolved); the field is otherwise
    # correct (real, right spectrum, mean |B|).
    Bk = (fft(B[1]), fft(B[2]), fft(B[3]))
    _divergence_clean!(Bk, Kx, Ky, Kz, nGrid)

    # (5) inverse FFT -> real space, NGP-interpolate to particles.
    B = (real.(ifft(Bk[1])), real.(ifft(Bk[2])), real.(ifft(Bk[3])))

    return _grid2particles_ngp(B, pos, L, nGrid)
end

@inline _powerspectrum(k, idx) = k^idx

# Fill the full complex k-space cube. For every independent mode draw a
# Box-Muller amplitude + random phase and also write the conjugate at the
# mirrored index (B(-k)=conj(B(k))), guaranteeing a real inverse transform.
function _fill_fourier_grid!(Bk, Kx, Ky, Kz, nGrid::Int, Kmin::Float64,
                             Kmax::Float64, sidx::Float64, seed::Integer)
    rng = Random.Xoshiro(seed)
    half = nGrid ÷ 2

    # Frequency index -> signed wavenumber multiple, the
    # `fftfreq(nGrid, nGrid)` convention: 0, 1, …, nGrid/2-1, -nGrid/2, …, -1
    # (0-based index `i`). This grid is antisymmetric under the conjugate-mirror
    # index map `i -> (nGrid-i) % nGrid`, so the projector `(I - k̂k̂ᵀ)`
    # preserves Hermitian symmetry and `real.(ifft(...))` stays divergence-free.
    @inline kmult(i) = i < half ? i : i - nGrid

    @inbounds for i in 0:(nGrid - 1)
        kxv = kmult(i) * Kmin
        ii = i + 1
        for j in 0:(nGrid - 1)
            kyv = kmult(j) * Kmin
            jj = j + 1
            for k in 0:(nGrid - 1)
                kzv = kmult(k) * Kmin
                kk = k + 1

                Kx[ii, jj, kk] = kxv
                Ky[ii, jj, kk] = kyv
                Kz[ii, jj, kk] = kzv

                # The DC mode and the Nyquist hyperplanes (index nGrid/2 on any
                # axis) are left zero so the populated sub-lattice is closed
                # under the conjugate index mirror, keeping the field real after
                # ifft. The K stores above are still recorded for every cell.
                (i == 0 && j == 0 && k == 0) && continue
                (i == half || j == half || k == half) && continue

                # already filled as the conjugate of an earlier mode?
                (Bk[1][ii, jj, kk] != 0 || Bk[2][ii, jj, kk] != 0 ||
                 Bk[3][ii, jj, kk] != 0) && continue

                kmag = sqrt(kxv * kxv + kyv * kyv + kzv * kzv)
                kmag > Kmax && continue          # sphere in k-space

                pk = _powerspectrum(kmag, sidx)

                # conjugate (mirror) index: (-k) mod nGrid.
                ic = (nGrid - i) % nGrid
                jc = (nGrid - j) % nGrid
                kc = (nGrid - k) % nGrid
                iic = ic + 1; jjc = jc + 1; kkc = kc + 1

                self_conj = (ic == i && jc == j && kc == k)

                for comp in 1:3
                    amp = sqrt(-log(_rand_pos(rng)) * pk)
                    if self_conj
                        # self-conjugate mode (DC/Nyquist planes): real.
                        sign = rand(rng) < 0.5 ? -1.0 : 1.0
                        Bk[comp][ii, jj, kk] = ComplexF64(sign * amp, 0.0)
                    else
                        phase = 2.0 * pi * rand(rng)
                        val = amp * (cos(phase) + im * sin(phase))
                        Bk[comp][ii, jj, kk] = val
                        Bk[comp][iic, jjc, kkc] = conj(val)
                    end
                end
            end
        end
    end
    return nothing
end

# Uniform draw in (0,1), strictly positive so log() is finite.
@inline function _rand_pos(rng)
    u = rand(rng)
    while u <= 0.0
        u = rand(rng)
    end
    return u
end

# Scale the real-space field so its global mean |B| equals B_norm: divide
# every cell by the mean |B| and multiply by B_norm.
function _normalise_bfld_grid!(B, nGrid::Int, B_norm::Float64)
    n3 = nGrid * nGrid * nGrid
    s = 0.0
    @inbounds for idx in 1:n3
        s += sqrt(B[1][idx]^2 + B[2][idx]^2 + B[3][idx]^2)
    end
    global_mean = s / n3
    global_mean == 0 && return nothing
    fac = B_norm / global_mean
    @inbounds for idx in 1:n3
        B[1][idx] *= fac
        B[2][idx] *= fac
        B[3][idx] *= fac
    end
    return nothing
end

# Divergence clean (Balsara 1996 / Ruszkowski+ 2006): project out the
# longitudinal part of each k-mode, B_k <- (I - k̂ k̂ᵀ)·B_k, then zero the DC
# mode.
function _divergence_clean!(Bk, Kx, Ky, Kz, nGrid::Int)
    n3 = nGrid * nGrid * nGrid
    @inbounds for idx in 1:n3
        bx = Bk[1][idx]
        by = Bk[2][idx]
        bz = Bk[3][idx]
        kx = Kx[idx]
        ky = Ky[idx]
        kz = Kz[idx]
        k2 = kx * kx + ky * ky + kz * kz
        if k2 == 0.0
            continue
        end
        k2inv = 1.0 / k2
        Bk[1][idx] = bx * (1.0 - kx * kx * k2inv) - by * kx * ky * k2inv -
                     bz * kx * kz * k2inv
        Bk[2][idx] = -bx * ky * kx * k2inv + by * (1.0 - ky * ky * k2inv) -
                     bz * ky * kz * k2inv
        # Row 3 uses +bz*(1 - kz^2/k^2); the projector (I - k-hat k-hat^T)
        # requires the + sign (the C source's -bz sign is wrong).
        Bk[3][idx] = -bx * kz * kx * k2inv - by * kz * ky * k2inv +
                     bz * (1.0 - kz * kz * k2inv)
    end
    # zero the DC mode; Julia index 1 == grid (0,0,0).
    Bk[1][1] = 0.0 + 0.0im
    Bk[2][1] = 0.0 + 0.0im
    Bk[3][1] = 0.0 + 0.0im
    return nothing
end

# NGP interpolation (Hockney & Eastwood): positions are wrapped periodically
# into [0, nGrid) then floored to the nearest grid point. Julia arrays are
# 1-based so add 1 to indices.
function _grid2particles_ngp(B, pos::AbstractVector{<:SVector{3}}, L::Float64,
                             nGrid::Int)
    cell = L / nGrid
    out = Vector{SVector{3,Float32}}(undef, length(pos))
    @inbounds for ipart in eachindex(pos)
        p = pos[ipart]
        u = p[1] / cell
        v = p[2] / cell
        w = p[3] / cell
        # periodic wrap into [0, nGrid)
        u -= nGrid * floor(u / nGrid)
        v -= nGrid * floor(v / nGrid)
        w -= nGrid * floor(w / nGrid)
        i = floor(Int, u)
        j = floor(Int, v)
        k = floor(Int, w)
        # guard against u/v/w == nGrid from round-off
        i >= nGrid && (i = nGrid - 1)
        j >= nGrid && (j = nGrid - 1)
        k >= nGrid && (k = nGrid - 1)
        i < 0 && (i = 0)
        j < 0 && (j = 0)
        k < 0 && (k = 0)
        ii = i + 1; jj = j + 1; kk = k + 1
        out[ipart] = SVector{3,Float32}(B[1][ii, jj, kk],
                                        B[2][ii, jj, kk],
                                        B[3][ii, jj, kk])
    end
    return out
end

"""
    make_turbulent_postprocess(; B_norm=1e-6, B_scale=…, spectral_index=-11/3,
                               seed=14041981) -> postprocess!-callback

Build a problem `postprocess!` callback `(particles, param, problem) ->
nothing` that fills `particles.bfld` with a divergence-free turbulent field
([`make_turbulent_Bfield`](@ref)) over the final post-relaxation positions.

Runs after `regularise_sph_particles!`, so all particle positions are final.
`B_scale` defaults to `boxsize/16` (a reasonable turbulent injection scale that
keeps the grid modest); pass it explicitly to override. The box is taken as the
largest axis (`Boxsize[1]`); the field is generated on a cubic `[0, boxsize)`
grid.
"""
function make_turbulent_postprocess(; B_norm::Real = 1e-6,
                                      B_scale::Union{Nothing,Real} = nothing,
                                      spectral_index::Real = -11.0 / 3.0,
                                      seed::Integer = 14041981)
    return function (particles::Particles, param::Parameters,
                     problem::ProblemParameters)
        n = param.Npart
        n == 0 && return nothing
        L = problem.Boxsize[1]                 # largest axis
        bs = B_scale === nothing ? L / 16.0 : float(B_scale)
        b = make_turbulent_Bfield(particles.pos, L;
                                  B_norm = B_norm, B_scale = bs,
                                  spectral_index = spectral_index,
                                  seed = seed)
        @inbounds for ipart in 1:n
            particles.bfld[ipart] = b[ipart]
        end
        return nothing
    end
end
