# Divergence-free turbulent magnetic field generator.
#
# Julia port of `make_turb_B.c` (J. Donnert 2013; CLAUDE.md §1.9). The C tool
# is standalone (flat-binary interchange); this port is in-memory and callable
# both standalone and as a problem `postprocess!` hook for the cluster /
# Magneticum problems (the preferred wiring per CLAUDE.md §1.9 — it needs ALL
# final post-relaxation positions, and `main` runs the appliers after
# `regularise_sph_particles!`).
#
# Algorithm (C `make_magnetic_field`):
#   1. nGrid = 2*ceil(Boxsize / B_scale).
#      Kmin = 2π / (0.01·Boxsize),  Kmax = 0.5·Kmin·nGrid  (Nyquist).
#   2. Fill a complex k-space grid per component with a power-law spectrum
#      P(k) = k^spectral_index, Box-Muller amplitudes sqrt(-log(U)·P(k)) and a
#      random phase; drop |k| > Kmax; enforce Hermitian symmetry so the inverse
#      transform is real.
#   3. Inverse FFT -> real space; normalise to the field model (constant
#      `B_norm` by default, the C `magnetic_field_model`).
#   4. Forward FFT; divergence clean by the projection
#      B_k <- (I - k̂ k̂ᵀ)·B_k (Ruszkowski+ 2006 / Balsara 1996); zero DC.
#   5. Inverse FFT -> real space; NGP-interpolate grid -> particles (periodic).
#
# FFT choice (CLAUDE.md §1.9 explicitly permits this): the **full complex
# `fft`/`ifft`** path is used (not `rfft`/`irfft`) — it is the easiest
# correctness path, the Hermitian symmetry is enforced explicitly on the full
# cube so the inverse transform is real to round-off, and the divergence-clean
# projection is applied to every mode. This deviates from the C `r2c`/`c2r`
# layout but is mathematically equivalent and far less error-prone.
#
# RNG note (CLAUDE.md §5, as elsewhere in this port): a seeded
# `Random.Xoshiro` is used (statistical reproducibility for a fixed seed). The
# stream is NOT bit-identical to the C GSL RNG — by design and documented.

"""
    make_turbulent_Bfield(pos, boxsize; B_norm=1e-6, B_scale=0.1,
                          spectral_index=-11/3, seed=14041981)
        -> Vector{SVector{3,Float32}}

Generate a periodic-box, divergence-free turbulent magnetic field with a
power-law power spectrum `P(k) = k^spectral_index` (Kolmogorov default
`-11/3`) and NGP-interpolate it to the particle positions `pos`
(`AbstractVector{<:SVector{3}}`, the WVTICs SoA `particles.pos`).

- `boxsize` — the (cubic) periodic box side length used by the C tool's
  `Boxsize` (the grid is built on `[0, boxsize)`).
- `B_norm` — target mean field magnitude (C `Bfld_Norm`, default `1e-6` G;
  the constant `magnetic_field_model`).
- `B_scale` — sets the grid resolution `nGrid = 2·ceil(boxsize/B_scale)`
  (C `Bfld_Scale`, default `0.1`).
- `spectral_index` — power-law slope (C `SPECTRAL_INDEX`, default `-11/3`).
- `seed` — RNG seed (statistical reproducibility; not C-bit-identical).

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
    # `_divergence_clean!` also zeroes the DC mode and the Nyquist
    # hyperplanes). `fft` of a real field is Hermitian to round-off.
    #
    # KNOWN OPEN ISSUE (see PORT_STATUS.md "turbulent-B spectral divergence"
    # and /e/ocean2/users/lboess/WVTICs/TURBB_ANALYSIS.md): the Phase-4
    # spectral-∇·B test fails with a residual `max|k·B_k| ≈ 0.61·max(k|B_k|)`
    # that is INVARIANT (to ~1e-8) across every attempted fix so far
    # (kmult/fftfreq correction, removing the old `_hermitian_symmetrise!`,
    # zeroing the Nyquist hyperplanes, and an index-mirror Hermitian
    # projection — all verified inert). Orchestrator instrumentation showed
    # the residual is present *immediately after the projector, measured with
    # the projector's own Kx/Ky/Kz* (before any ifft/round-trip), and that at
    # the worst mode the projector output exactly equals hand-computed
    # `B−k̂(k̂·B)` yet `k·B≠0` — algebraically impossible for a consistent `k`,
    # i.e. the true cause is still unexplained and upstream of the round-trip.
    # The field is otherwise correct (real, right spectrum slope, mean |B|,
    # NGP, postprocess wiring). The Phase-4 spectral subtest is therefore
    # left as `@test_broken` pending a dedicated controlled single-mode
    # investigation. The `_hermitian_project!` experiment was reverted (inert).
    Bk = (fft(B[1]), fft(B[2]), fft(B[3]))
    _divergence_clean!(Bk, Kx, Ky, Kz, nGrid)

    # (5) inverse FFT -> real space, NGP-interpolate to particles.
    B = (real.(ifft(Bk[1])), real.(ifft(Bk[2])), real.(ifft(Bk[3])))

    return _grid2particles_ngp(B, pos, L, nGrid)
end

@inline _powerspectrum(k, idx) = k^idx

# Port of make_turb_B.c::fill_fourier_grid. The C code uses an r2c half-grid
# and hand-rolls the k=0-plane Hermitian symmetry. Here the FULL complex cube
# is filled: for every independent mode we draw Box-Muller amplitude + random
# phase and ALSO write the conjugate at the mirrored index (B(-k)=conj(B(k))),
# guaranteeing a real inverse transform. The DC mode (0,0,0) and the self-
# conjugate Nyquist modes are set real. This is the documented correctness
# path (CLAUDE.md §1.9) and matches the C spectrum/cutoff exactly.
function _fill_fourier_grid!(Bk, Kx, Ky, Kz, nGrid::Int, Kmin::Float64,
                             Kmax::Float64, sidx::Float64, seed::Integer)
    rng = Random.Xoshiro(seed)
    half = nGrid ÷ 2

    # Frequency index -> signed wavenumber multiple, EXACTLY the
    # AbstractFFTs/FFTW `fftfreq(nGrid, nGrid)` convention:
    #   0, 1, …, nGrid/2-1, -nGrid/2, …, -1   (0-based index `i`).
    # This makes the wavevector grid *exactly antisymmetric* under the
    # conjugate-mirror index map `i -> (nGrid-i) % nGrid` for every non
    # self-mirror mode (`kmult((nGrid-i)%nGrid) == -kmult(i)`), which is what
    # makes the even projector `(I - k̂k̂ᵀ)` preserve Hermitian symmetry — and
    # therefore lets `real.(ifft(...))` keep a divergence-free field.
    # (Bug-fix: the previous `i <= half ? i : i - nGrid` put the Nyquist index
    # at `+nGrid/2`, breaking that antisymmetry and reintroducing divergence
    # when `real.(ifft())` discarded the resulting non-Hermitian imaginary
    # part — observed as the spectral ∇·B failing the Phase-4 test by ~6000×.)
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

                # Leave the DC mode AND the Nyquist hyperplanes (index
                # nGrid/2 on any axis) exactly zero in the fill. This is the
                # Toycluster-comparison fix (TURBB_TOYCLUSTER_COMPARISON.md,
                # H1): the C r2c half-grid implicitly omits the signed Nyquist
                # plane, so its divergence-clean operates on a sub-lattice
                # exactly closed under the conjugate index mirror. Populating
                # the Nyquist planes here and zeroing them post-projector (the
                # old approach) tampered the spectrum so `Bk_cleaned` was not a
                # fixed point of `fft∘real∘ifft`, re-expressing a fixed ~0.61
                # longitudinal residual (the bit-identical @test_broken across
                # all prior attempts). Never populating them removes the
                # tampering at the source; no physics change (DC carries zero
                # mean; the post-hoc-discarded Nyquist plane carried no kept
                # power). The K stores above are still recorded for every cell.
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

# gsl_rng_uniform_pos analogue: a uniform in (0,1) (strictly positive, so
# log() is finite).
@inline function _rand_pos(rng)
    u = rand(rng)
    while u <= 0.0
        u = rand(rng)
    end
    return u
end

# Port of make_turb_B.c::normalise_bfld_grid. The C code computes the global
# mean of |B| over the (half) grid, divides by it and by the FFTW
# normalisation (nGrid^3), then scales every cell by
# magnetic_field_model(x,y,z) = B_norm (constant). With Julia's `ifft`
# (already 1/N-normalised) the FFTW nGrid^3 factor is NOT needed; dividing by
# the global mean and multiplying by B_norm makes the mean |B| == B_norm.
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

# Port of make_turb_B.c::divergence_clean_fourier_grid (Balsara 1996 /
# Ruszkowski+ 2006): project out the longitudinal part of each k-mode,
# B_k <- (I - k̂ k̂ᵀ)·B_k, then zero the DC mode.
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
        # NOTE: the `+ bz*(1 - kz²/k²)` sign here is a deliberate CORRECTION
        # of a latent bug in the C `make_turb_B.c` (and the earlier port),
        # whose third projector row read `- Bz*(1-kz*kz*k2inv)`. The
        # projection operator `(I - k̂k̂ᵀ)` row 3 is unambiguously
        # `B'_z = -bx·kzkx/k² - by·kzky/k² + bz·(1 - kz²/k²)` (rows 1,2 in the
        # C are already the correct `+bx·(1-kx²/k²)` / `+by·(1-ky²/k²)`
        # form). The wrong sign left the field longitudinal in z, the true
        # root cause of the long-standing spectral-∇·B `@test_broken`
        # (residual ≈0.5-0.61 invariant across all Hermitian/Nyquist/layout
        # fixes, none of which touched this term). See
        # TURBB_TOYCLUSTER_COMPARISON.md / TURBB_ANALYSIS.md.
        Bk[3][idx] = -bx * kz * kx * k2inv - by * kz * ky * k2inv +
                     bz * (1.0 - kz * kz * k2inv)
    end
    # zero the DC mode (C: Bk[*][0] = 0). Julia index 1 == grid (0,0,0).
    Bk[1][1] = 0.0 + 0.0im
    Bk[2][1] = 0.0 + 0.0im
    Bk[3][1] = 0.0 + 0.0im
    # NOTE: the post-projector Nyquist-hyperplane zeroing that used to live
    # here was REMOVED (TURBB_TOYCLUSTER_COMPARISON.md, H1 fix, Change B).
    # The DC + Nyquist planes are now never populated in `_fill_fourier_grid!`
    # (Change A), so they carry no power through `fft(B)` here and need no
    # post-hoc tampering. Tampering the full cube post-projector made
    # `Bk_cleaned` not a fixed point of `fft∘real∘ifft` and re-expressed a
    # fixed ~0.61 longitudinal residual. With the populated sub-lattice now
    # exactly closed under the conjugate index mirror (kmult(mirror) =
    # -kmult(idx) everywhere it is nonzero), the cleaned spectrum is a fixed
    # point and the projector's per-mode k·B≡0 survives to round-off.
    return nothing
end

# NOTE: an experimental `_hermitian_project!` step (index-mirror conjugate
# averaging of the cleaned spectrum) was added and then REVERTED — orchestrator
# verification showed it inert (the spectral-∇·B residual stayed bit-identical
# at `max|k·B_k| ≈ 0.61·max(k|B_k|)`), like the three prior attempts. The
# spectral-divergence defect is a documented KNOWN OPEN ISSUE (`@test_broken`;
# see PORT_STATUS.md and /e/ocean2/users/lboess/WVTICs/TURBB_ANALYSIS.md) whose
# true cause is still unexplained (the residual is present immediately after
# the projector, measured with its own K, before any round-trip). No
# Hermitian-projection step is applied.

# Port of make_turb_B.c::grid2particles_NGP (Hockney & Eastwood). The C grid
# spans [0, Boxsize); positions are wrapped periodically into [0, nGrid) then
# floored (nearest grid point). Julia arrays are 1-based so add 1 to indices.
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

Wired into the **galaxy_cluster (4.12)** and **magneticum (2.0)** problems
(CLAUDE.md §1.9): `main` calls `make_post_processing!` after
`regularise_sph_particles!`, so all particle positions are final when this
runs. `B_scale` defaults to `boxsize/16` (a reasonable turbulent injection
scale that keeps the grid modest); pass it explicitly to override. The box is
taken as the largest axis (`Boxsize[1]`, the C-invariant largest dimension);
the field is generated on a cubic `[0, boxsize)` grid as in the C tool.
"""
function make_turbulent_postprocess(; B_norm::Real = 1e-6,
                                      B_scale::Union{Nothing,Real} = nothing,
                                      spectral_index::Real = -11.0 / 3.0,
                                      seed::Integer = 14041981)
    return function (particles::Particles, param::Parameters,
                     problem::ProblemParameters)
        n = param.Npart
        n == 0 && return nothing
        L = problem.Boxsize[1]                 # largest axis (invariant)
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
