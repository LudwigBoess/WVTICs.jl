# Problem registry: each test setup is a `Problem` struct holding the
# density / internal-energy / velocity / B-field / post-processing callbacks
# plus the geometry it imposes on the `ProblemParameters`. A
# `Dict{Tuple{Int,Int}, Function}` keyed by `(Problem_Flag, Problem_Subflag)`
# maps to the per-problem constructor, matching the `ics.par` header table.
#
# Conventions:
# - `boxsize[1]` (axis 1) is the largest invariant, asserted in `setup`.
# - `density_function_correction` is an artificial density-model correction
#   compensating imperfect WVT convergence
#   (`ret += (ret-RHO_MEAN)*density_function_correction`), applied by the
#   tophat/sawtooth/sine/linear-alfven problems. `setup`/`mpart_from_integral`
#   pass 0.0; the WVT / relaxation path passes `param.density_function_correction`.
# - U callbacks read `particles.rho[ipart]` (post-relaxation density); the
#   appliers run after `regularise_sph_particles!`.
# - The PNG path (problem 2.1) is out of scope and registered to an error.

"""
    Problem

Per-problem definition: the physics callbacks plus the geometry the setup
imposes.

Callback signatures:

- `density(particles, ipart, density_function_correction) -> Float64`
- `internal_energy(particles, ipart) -> Float64`
- `velocity(particles, ipart) -> SVector{3,Float64}`
- `bfield(particles, ipart) -> SVector{3,Float64}`
- `postprocess!(particles, param, problem) -> nothing`

`density` is passed the `ipart` index and reads the *current* particle
position from `particles.pos[ipart]` (`mpart_from_integral` uses `pos[1]` as a
scratch probe; see `mpart_from_integral` in `setup.jl`).
"""
struct Problem
    name::String
    boxsize::NTuple{3,Float64}
    periodic::NTuple{3,Bool}
    rho_max::Float64
    density::Function
    internal_energy::Function
    velocity::Function
    bfield::Function
    postprocess!::Function
end

# --- zero callbacks --------------------------------------------------------

zero_density(particles, ipart, density_function_correction)::Float64 = 0.0
zero_U(particles, ipart)::Float64 = 0.0
zero_vec(particles, ipart) = zero(SVector{3,Float64})
zero_postprocess!(particles, param, problem) = nothing

# helpers: squares/cubes and common constants
@inline _p2(a) = a * a
@inline _p3(a) = a * a * a
const _SQRT2 = sqrt(2.0)
const _SQRT4PI = sqrt(4.0 * pi)     # MHD field normalisation (Gaussian units)
const GAMMA = 5.0 / 3.0             # adiabatic index for the ideal-gas problems

# --- Problem 0.0 : Constant Density ---------------------------------------

constant_density(particles, ipart, density_function_correction)::Float64 = 1.0

"""
    setup_constant_density(param) -> Problem

Unit box, periodic, uniform density 1.0, `Rho_Max = 1.0`.
"""
function setup_constant_density(param::Parameters)
    return Problem(
        "IC_Constant_Density",
        (1.0, 1.0, 1.0),
        (true, true, true),
        1.0,
        constant_density,
        zero_U,
        zero_vec,
        zero_vec,
        zero_postprocess!,
    )
end

# --- Problem 0.1 : Top-Hat density ----------------------------------------

# density step 0.5 about mean 1.0
@inline function _tophat_density(particles::Particles, ipart::Int, density_function_correction)::Float64
    x = particles.pos[ipart][1]
    halfstep = 0.5
    rho_max = 1.0 + halfstep
    rho_min = 1.0 - halfstep
    if x <= 0.1 || x > 0.9
        ret = rho_min
    elseif x > 0.4 && x <= 0.6
        ret = rho_max
    elseif x > 0.6
        ret = rho_max - (rho_max - rho_min) * (x - 0.6) / 0.3
    else
        ret = rho_min + (rho_max - rho_min) * (x - 0.1) / 0.3
    end
    ret += (ret - 1.0) * density_function_correction
    return ret
end

"""
    setup_tophat(param) -> Problem

Box 1 x 0.5 x 0.1, periodic, `Rho_Max = 1.5`. Applies the
`density_function_correction` term.
"""
function setup_tophat(param::Parameters)
    return Problem(
        "IC_TopHat",
        (1.0, 0.5, 0.1),
        (true, true, true),
        1.5,
        _tophat_density,
        zero_U,
        zero_vec,
        zero_vec,
        zero_postprocess!,
    )
end

# --- Problem 0.2 : Sawtooth density ---------------------------------------

@inline function _sawtooth_density(particles::Particles, ipart::Int, density_function_correction)::Float64
    x = particles.pos[ipart][1]
    if x > 0.5
        x -= 0.5
    end
    halfstep = 0.5
    rho_max = 1.0 + halfstep
    rho_min = 1.0 - halfstep
    ret = rho_min + (rho_max - rho_min) * x / 0.5
    ret += (ret - 1.0) * density_function_correction
    return ret
end

"""
    setup_sawtooth(param) -> Problem

Box 1 x 0.1 x 0.1, periodic, `Rho_Max = 1.5`. Applies the
`density_function_correction` term.
"""
function setup_sawtooth(param::Parameters)
    return Problem(
        "IC_Sawtooth",
        (1.0, 0.1, 0.1),
        (true, true, true),
        1.5,
        _sawtooth_density,
        zero_U,
        zero_vec,
        zero_vec,
        zero_postprocess!,
    )
end

# --- Problem 0.3 : Sine wave density --------------------------------------

# one full wave about mean 1.0
@inline function _sinewave_density(particles::Particles, ipart::Int, density_function_correction)::Float64
    x = particles.pos[ipart][1]
    ret = 1.0 + 0.5 * sin(2.0 * pi * x)
    ret += (ret - 1.0) * density_function_correction
    return ret
end

"""
    setup_sinewave(param) -> Problem

Box 1 x 0.75 x 0.75, periodic, `Rho_Max = 1.5`. Applies the
`density_function_correction` term.
"""
function setup_sinewave(param::Parameters)
    return Problem(
        "IC_SineWave",
        (1.0, 0.75, 0.75),
        (true, true, true),
        1.5,
        _sinewave_density,
        zero_U,
        zero_vec,
        zero_vec,
        zero_postprocess!,
    )
end

# --- Problem 1.0 : Gradient density ---------------------------------------

# defaults: 1x1x1 box, periodic, Rho_Max 1.0; no density_function_correction
@inline function _gradient_density(particles::Particles, ipart::Int, density_function_correction)::Float64
    x = particles.pos[ipart][1]
    halfstep = 0.5
    rho_max = 1.0 + halfstep
    rho_min = 1.0 - halfstep
    if x <= 0.25
        return rho_min
    elseif x >= 0.75
        return rho_max
    end
    return rho_min + (rho_max - rho_min) * (x - 0.25) / 0.5
end

"""
    setup_gradient(param) -> Problem

Box 1 x 1 x 1, periodic, `Rho_Max = 1.0`. No `density_function_correction`
term.
"""
function setup_gradient(param::Parameters)
    return Problem(
        "IC_GradientDensity",
        (1.0, 1.0, 1.0),
        (true, true, true),
        1.0,
        _gradient_density,
        zero_U,
        zero_vec,
        zero_vec,
        zero_postprocess!,
    )
end

# --- Problem 2.0 : Magneticum logo ----------------------------------------

# "MAGNETICUM" logo density mask, contrast 16. Box 1 x 1 x 0.5, Rho_Max 1.0.
@inline function _magneticum_density(particles::Particles, ipart::Int, density_function_correction)::Float64
    p = particles.pos[ipart]
    bx = 1.0; by = 1.0; bz = 0.5
    x = p[1] / bx
    y = p[2] / by
    z = p[3] / bz
    rho = 1.0
    if z < 0.1 || z > 0.9
        return rho / 16.0
    end
    # M
    if (0.00 < x < 0.02) && (0.45 < y < 0.56); return rho; end
    if (0.04 < x < 0.06) && (0.45 < y < 0.55); return rho; end
    if (0.08 < x < 0.10) && (0.45 < y < 0.55); return rho; end
    if (0.00 < x < 0.10) && (0.53 < y < 0.55); return rho; end
    # A
    if (0.12 < x < 0.14) && (0.45 < y < 0.56); return rho; end
    if (0.17 < x < 0.19) && (0.45 < y < 0.55); return rho; end
    if (0.12 < x < 0.19) && (0.53 < y < 0.55); return rho; end
    if (0.12 < x < 0.19) && (0.49 < y < 0.51); return rho; end
    # G
    if (0.21 < x < 0.23) && (0.45 < y < 0.55); return rho; end
    if (0.21 < x < 0.28) && (0.53 < y < 0.55); return rho; end
    if (0.21 < x < 0.28) && (0.45 < y < 0.47); return rho; end
    if (0.26 < x < 0.28) && (0.45 < y < 0.49); return rho; end
    # N
    if (0.30 < x < 0.32) && (0.45 < y < 0.56); return rho; end
    if (0.36 < x < 0.38) && (0.45 < y < 0.55); return rho; end
    if (0.30 < x < 0.38) && (0.53 < y < 0.55); return rho; end
    # E
    if (0.40 < x < 0.42) && (0.45 < y < 0.55); return rho; end
    if (0.40 < x < 0.48) && (0.53 < y < 0.55); return rho; end
    if (0.40 < x < 0.48) && (0.49 < y < 0.51); return rho; end
    if (0.40 < x < 0.48) && (0.45 < y < 0.47); return rho; end
    # T
    if (0.53 < x < 0.55) && (0.45 < y < 0.55); return rho; end
    if (0.50 < x < 0.58) && (0.53 < y < 0.55); return rho; end
    # I
    if (0.63 < x < 0.65) && (0.45 < y < 0.55); return rho; end
    if (0.60 < x < 0.68) && (0.53 < y < 0.55); return rho; end
    if (0.60 < x < 0.68) && (0.45 < y < 0.47); return rho; end
    # C
    if (0.70 < x < 0.72) && (0.45 < y < 0.55); return rho; end
    if (0.70 < x < 0.78) && (0.53 < y < 0.55); return rho; end
    if (0.70 < x < 0.78) && (0.45 < y < 0.47); return rho; end
    # U
    if (0.80 < x < 0.82) && (0.45 < y < 0.55); return rho; end
    if (0.86 < x < 0.88) && (0.45 < y < 0.55); return rho; end
    if (0.80 < x < 0.88) && (0.45 < y < 0.47); return rho; end
    # M
    if (0.90 < x < 0.92) && (0.45 < y < 0.56); return rho; end
    if (0.94 < x < 0.96) && (0.45 < y < 0.55); return rho; end
    if (0.98 < x < 1.00) && (0.45 < y < 0.55); return rho; end
    if (0.90 < x < 1.00) && (0.53 < y < 0.55); return rho; end
    # Underline
    if (0.00 < x < 1.00) && (0.41 < y < 0.43); return rho; end
    return rho / 16.0
end

"""
    setup_magneticum(param) -> Problem

Box 1 x 1 x 0.5, periodic, `Rho_Max = 1.0`. The "MAGNETICUM" logo density
mask; turbulent B set via `postprocess!` (see `make_turbulent_postprocess`).
"""
function setup_magneticum(param::Parameters)
    return Problem(
        "IC_Magneticum",
        (1.0, 1.0, 0.5),
        (true, true, true),
        1.0,
        _magneticum_density,
        zero_U,
        zero_vec,
        zero_vec,
        make_turbulent_postprocess(),
    )
end

# --- Problem 2.1 : PNG logo — OUT OF SCOPE --------------------------------

function setup_png(param::Parameters)
    error("problem 2.1 (PNG logo) is out of scope for this port; not ported.")
end

# --- Problem 3.x : Double Shock --------------------------------------------

# per-subflag parameters: {upstream cs [km/s], Mach, downstream Velx}
const _DS_PARAMS = ((850.0, 2.0, 1.6 * 2000.0),
                    (850.0, 3.0, 2.4 * 2000.0),
                    (850.0, 4.0, 2.9 * 2000.0))

# Mo+ 2010 eq. 8.49 / 8.50
@inline _ds_compression(M, gamma) =
    1.0 / (1.0 / _p2(M) + (gamma - 1.0) / (gamma + 1.0) * (1.0 - 1.0 / _p2(M)))
@inline _ds_pressure(M, gamma) =
    2.0 * gamma / (gamma + 1.0) * M * M - (gamma - 1.0) / (gamma + 1.0)

# bisection (Press+ 1992) for the second-shock Mach number
function _ds_find_M1(cs_up, v_dw, gamma)
    left = 1.0
    right = 100.0
    M = 0.0
    while true
        M = left + 0.5 * (right - left)
        sigma = _ds_compression(M, gamma)
        res = sigma * M - v_dw / cs_up
        if abs(res) < 1e-4
            break
        end
        if res < 0
            left = M
        else
            right = M
        end
    end
    return M
end

# Returns (Rho::NTuple{3}, U::NTuple{3}, Velx::NTuple{3}) — the converged
# three-region shock-tube state.
function _ds_state(subflag::Int)
    gamma = GAMMA
    cs0, mach0, velx2 = _DS_PARAMS[subflag + 1]
    # Rho[0] = 1e-28 g/cm^3 in standard Gadget units
    ULength = 3.08568025e21
    UMass = 1.989e43
    rho0 = 1e-28 * (_p3(ULength) / UMass)
    u0 = cs0 * cs0 / gamma / (gamma - 1.0)

    cs = [cs0, 0.0, 0.0]
    Rho = [rho0, 0.0, 0.0]
    U = [u0, 0.0, 0.0]
    Velx = [0.0, 0.0, velx2]   # Velx[0]=0 (frame of upstream gas), Velx[2] set
    Mach = [mach0, 0.0]

    # First shock
    sigma_v = _ds_compression(Mach[1], gamma)
    sigma_P = _ds_pressure(Mach[1], gamma)
    sigma_T = sigma_P / sigma_v
    Rho[2] = Rho[1] * sigma_v
    Velx[2] = cs[1] * Mach[1] * (1.0 - 1.0 / sigma_v)
    U[2] = U[1] * sigma_T
    cs[2] = sqrt(U[2] * gamma * (gamma - 1.0))

    # Second shock
    Mach[2] = _ds_find_M1(cs[2], Velx[3], gamma)
    sigma_v = _ds_compression(Mach[2], gamma)
    sigma_P = _ds_pressure(Mach[2], gamma)
    sigma_T = sigma_P / sigma_v
    Rho[3] = Rho[2] * sigma_v
    U[3] = U[2] * sigma_T
    cs[3] = sqrt(U[3] * gamma * (gamma - 1.0))

    return ((Rho[1], Rho[2], Rho[3]), (U[1], U[2], U[3]),
            (Velx[1], Velx[2], Velx[3]))
end

"""
    setup_double_shock(subflag) -> (param -> Problem)

Box 2000 x 200 x 100, periodic, three regions split at `XBoxhalf` and
`1.5*XBoxhalf`. `subflag` selects the Mach number (0: Mach 2, 1: Mach 3,
2: Mach 4). `Rho_Max = Rho[3]*1.1`.
"""
function setup_double_shock(subflag::Int)
    return function (param::Parameters)
        Rho, U, Velx = _ds_state(subflag)
        bx = 2000.0
        xhalf = bx / 2.0
        rho_max = Rho[3] * 1.1

        function _ds_density(particles::Particles, ipart::Int, density_function_correction)::Float64
            x = particles.pos[ipart][1]
            if x < xhalf
                return Rho[1]
            elseif x < 1.5 * xhalf
                return Rho[2]
            else
                return Rho[3]
            end
        end
        density = _ds_density
        function _ds_internal_energy(particles::Particles, ipart::Int)::Float64
            x = particles.pos[ipart][1]
            if x < xhalf
                return U[1]
            elseif x < 1.5 * xhalf
                return U[2]
            else
                return U[3]
            end
        end
        internal_energy = _ds_internal_energy
        function _ds_velocity(particles::Particles, ipart::Int)
            x = particles.pos[ipart][1]
            vx = x < xhalf ? Velx[1] : (x < 1.5 * xhalf ? Velx[2] : Velx[3])
            return SVector{3,Float64}(vx, 0.0, 0.0)
        end
        velocity = _ds_velocity

        return Problem(
            "IC_DoubleShock",
            (bx, 200.0, 100.0),
            (true, true, true),
            rho_max,
            density,
            internal_energy,
            velocity,
            zero_vec,
            zero_postprocess!,
        )
    end
end

# --- Problem 4.0 : Sod Shock ----------------------------------------------

function setup_sod_shock(param::Parameters)
    bx = 140.0
    function _sod_density(particles::Particles, ipart::Int, density_function_correction)::Float64
        return particles.pos[ipart][1] <= 0.5 * bx ? 1.0 : 0.125
    end
    density = _sod_density
    function _sod_internal_energy(particles::Particles, ipart::Int)::Float64
        gamma = GAMMA
        if particles.pos[ipart][1] <= 0.5 * bx
            return 1.0 / (gamma - 1.0) / 1.0       # pLeft/(γ-1)/rhoLeft
        else
            return 0.1 / (gamma - 1.0) / 0.125     # pRight/(γ-1)/rhoRight
        end
    end
    internal_energy = _sod_internal_energy
    return Problem(
        "IC_SodShock",
        (bx, 1.0, 1.0),
        (true, true, true),
        1.0,
        density,
        internal_energy,
        zero_vec,
        zero_vec,
        zero_postprocess!,
    )
end

# --- Problem 4.1 : Sedov Blast --------------------------------------------

const _SEDOV_RHO = 1.24e7
const _SEDOV_NNPART = 296
const _SEDOV_U_SN = 0.00502765   # 1e51 erg per unit mass, Gadget units

_sedov_density(particles, ipart, density_function_correction)::Float64 = _SEDOV_RHO
_sedov_U(particles, ipart)::Float64 = 0.0

# Centred Euclidean distance of each particle to the box centre; returns the
# distance to the `NNpart`-th nearest particle (the radius enclosing the
# SN-energy core). Uses a centred squared distance on every axis (the C
# y-term sign typo is fixed here, so the sum under `sqrt` is always ≥ 0).
function _sedov_abs_maxdist(particles::Particles, bx, by, bz, npart::Int)
    rs = Vector{Float64}(undef, npart)
    @inbounds for i in 1:npart
        p = particles.pos[i]
        rx = (p[1] - 0.5 * bx) * (p[1] - 0.5 * bx)
        ry = (p[2] - 0.5 * by) * (p[2] - 0.5 * by)
        rz = (p[3] - 0.5 * bz) * (p[3] - 0.5 * bz)
        rs[i] = sqrt(rx + ry + rz)
    end
    sort!(rs)
    idx = min(_SEDOV_NNPART, npart)
    return rs[idx]
end

function _sedov_postprocess!(particles::Particles, param::Parameters,
                             problem::ProblemParameters)
    n = param.Npart
    n == 0 && return nothing
    bx, by, bz = problem.Boxsize
    maxd = _sedov_abs_maxdist(particles, bx, by, bz, n)
    @inbounds for ipart in 1:n
        p = particles.pos[ipart]
        x = p[1] - bx * 0.5
        y = p[2] - by * 0.5
        z = p[3] - bz * 0.5
        radius = sqrt(x * x + y * y + z * z)
        if radius <= maxd
            particles.u[ipart] = Float32(_SEDOV_U_SN)
        end
    end
    return nothing
end

"""
    setup_sedov_blast(param) -> Problem

Box 3 x 3 x 3, periodic, constant density `1.24e7`, `Rho_Max = 1.24e7`.
`postprocess!` injects the supernova energy into the innermost `NNpart`(=296)
particles.
"""
function setup_sedov_blast(param::Parameters)
    return Problem(
        "IC_SedovBlast",
        (3.0, 3.0, 3.0),
        (true, true, true),
        _SEDOV_RHO,
        _sedov_density,
        _sedov_U,
        zero_vec,
        zero_vec,
        _sedov_postprocess!,
    )
end

# --- Problem 4.2 : Kelvin-Helmholtz ---------------------------------------

# outer layer if y/boxsize_y <= 1/3 or > 2/3
@inline function _kh_is_outer_layer(particles::Particles, ipart::Int,
                                    boxsize_y::Float64)
    y = particles.pos[ipart][2]
    frac = y / boxsize_y
    return frac <= 1.0 / 3.0 || frac > 2.0 / 3.0
end

"""
    setup_kelvin_helmholtz(param) -> Problem

Box 256 x 256 x 16, periodic, `Rho_Max = 6.26e-8`. The
`density_function_correction` argument is unused here (kept for signature
parity).
"""
function setup_kelvin_helmholtz(param::Parameters)
    boxsize = (256.0, 256.0, 16.0)
    by = boxsize[2]

    density = function (particles::Particles, ipart::Int, density_function_correction)
        return _kh_is_outer_layer(particles, ipart, by) ? 3.13e-8 : 6.26e-8
    end

    internal_energy = (particles, ipart) -> 101527.0

    velocity = function (particles::Particles, ipart::Int)
        vx = _kh_is_outer_layer(particles, ipart, by) ? 40.0 : -40.0
        deltaVy = 4.0
        lambda = 128.0
        x = particles.pos[ipart][1]
        y = particles.pos[ipart][2]
        vy = deltaVy * (
            sin(2.0 * pi * (x + lambda / 2.0) / lambda) *
                exp(-(10.0 * (y - 64.0))^2) -
            sin(2.0 * pi * x / lambda) *
                exp(-(10.0 * (y + 64.0))^2)
        )
        return SVector{3,Float64}(vx, vy, 0.0)
    end

    return Problem(
        "IC_KelvinHelmholtz",
        boxsize,
        (true, true, true),
        6.26e-8,
        density,
        internal_energy,
        velocity,
        zero_vec,
        zero_postprocess!,
    )
end

# --- Problem 4.3 : Keplerian Ring — flagged "Error in result" -------------

function setup_keplerian_ring(param::Parameters)
    error("problem 4.3 (Keplerian Ring) is flagged \"Error in result. " *
          "Needs to be checked.\" in the C reference (ics.par); not ported " *
          "(avoids shipping a wrong IC).")
end

# --- Problem 4.4 : Cold Blob ----------------------------------------------

function setup_blob(param::Parameters)
    bx = 8000.0; by = 2000.0; bz = 2000.0
    function _blob_density(particles::Particles, ipart::Int, density_function_correction)::Float64
        p = particles.pos[ipart]
        x = p[1] - 3000.0
        y = p[2] - by * 0.5
        z = p[3] - bz * 0.5
        r = sqrt(x * x + y * y + z * z)
        return r < 197.0 ? 3.13e-7 : 3.13e-8
    end
    density = _blob_density
    internal_energy = (particles, ipart) -> 0.05
    velocity = (particles, ipart) -> SVector{3,Float64}(1000.0, 0.0, 0.0)
    return Problem(
        "IC_Blob",
        (bx, by, bz),
        (true, true, true),
        3.13e-7,
        density,
        internal_energy,
        velocity,
        zero_vec,
        zero_postprocess!,
    )
end

# --- Problem 4.5 : Hydrostatic Sphere — flagged "not implemented yet" ------

function setup_hydrostatic_sphere(param::Parameters)
    error("problem 4.5 (Hydrostatic Sphere) is flagged \"not implemented " *
          "yet\" in the C reference (ics.par; no C source exists); not ported.")
end

# --- Problem 4.6 : Evrard Collapse ----------------------------------------

function setup_evrard_collapse(param::Parameters)
    bx = 10.0; by = 10.0; bz = 10.0
    function _evrard_density(particles::Particles, ipart::Int, density_function_correction)::Float64
        p = particles.pos[ipart]
        x = p[1] - bx * 0.5
        y = p[2] - by * 0.5
        z = p[3] - bz * 0.5
        r = sqrt(x * x + y * y + z * z)
        eps = 0.01
        return r < 1.0 ? 1.0 / (2.0 * pi * (r + eps)) : 0.001
    end
    density = _evrard_density
    internal_energy = (particles, ipart) -> 0.05
    return Problem(
        "IC_Evrard_Collapse",
        (bx, by, bz),
        (true, true, true),
        10.0,
        density,
        internal_energy,
        zero_vec,
        zero_vec,
        zero_postprocess!,
    )
end

# --- Problem 4.7 : Zeldovich Pancake --------------------------------------

const _ZP_HUBBLE = 67.74
const _ZP_G = 6.67259e-8
@inline _zp_rho() = 3.0 * _ZP_HUBBLE * _ZP_HUBBLE / 8.0 / pi / _ZP_G
@inline _zp_q_of_x(px) = 0.0120544 + 0.999977 * px

function setup_zeldovich_pancake(param::Parameters)
    bx = 64.0
    rho = _zp_rho()
    k = 2.0 * pi / bx
    z_start = 100.0
    z_crit = 1.0
    function _zp_density(particles::Particles, ipart::Int, density_function_correction)::Float64
        q = _zp_q_of_x(particles.pos[ipart][1])
        return rho / (1.0 - (1.0 + z_crit) / (1.0 + z_start) * cos(k * q))
    end
    density = _zp_density
    velocity = function (particles::Particles, ipart::Int)
        q = _zp_q_of_x(particles.pos[ipart][1])
        vx = -_ZP_HUBBLE * (1.0 + z_crit) / sqrt(1.0 + z_start) *
             sin(k * q) / k
        return SVector{3,Float64}(vx, 0.0, 0.0)
    end
    function _zp_internal_energy(particles::Particles, ipart::Int)::Float64
        gamma = GAMMA
        temp_zero = 1.0
        kb = 1.380658e-16
        yhe = (1.0 - 0.76) / (4.0 * 0.76)
        mu = (1.0 + 4.0 * yhe) / (1.0 + 3.0 * yhe + 1.0)
        vu = 1e5
        prtn = 1.672623e-24
        u_fac = kb / ((gamma - 1.0) * vu * vu * prtn * mu)
        return u_fac * temp_zero * (z_start / z_crit)^2 *
               (particles.rho[ipart] / rho)^(2.0 / 3.0)
    end
    internal_energy = _zp_internal_energy
    return Problem(
        "IC_Zeldovich_Pancake",
        (bx, bx, bx),
        (true, true, true),
        rho,
        density,
        internal_energy,
        velocity,
        zero_vec,
        zero_postprocess!,
    )
end

# --- Problem 4.8 : Box ----------------------------------------------------

# inner box test. The y-bound uses bx*0.5 (not by*0.5); the arguments to
# `abs` are positive constants, so it just reproduces the centred box.
@inline function _box_is_inner(particles::Particles, ipart::Int,
                                bx, by, bz)::Bool
    p = particles.pos[ipart]
    x = p[1] - bx * 0.5
    y = p[2] - by * 0.5
    z = p[3] - bz * 0.5
    return x <= abs(bx * 0.5) && y <= abs(bx * 0.5) && z <= abs(bz * 0.5)
end

function setup_box(param::Parameters)
    bx = 1.0; by = 1.0; bz = 0.1
    function _box_density(particles::Particles, ipart::Int, density_function_correction)::Float64
        return _box_is_inner(particles, ipart, bx, by, bz) ? 4.0 : 1.0
    end
    density = _box_density
    velocity = (particles, ipart) -> SVector{3,Float64}(142.3, -31.3, 0.0)
    function _box_internal_energy(particles::Particles, ipart::Int)::Float64
        gamma = GAMMA
        pressure = 2.5
        return pressure / (gamma - 1.0) / particles.rho[ipart]
    end
    internal_energy = _box_internal_energy
    return Problem(
        "IC_Box",
        (bx, by, bz),
        (true, true, true),
        4.0,
        density,
        internal_energy,
        velocity,
        zero_vec,
        zero_postprocess!,
    )
end

# --- Problem 4.9 : Gresho Vortex ------------------------------------------

function setup_gresho_vortex(param::Parameters)
    bx = 1.0; by = 1.0; bz = 0.1
    density = (particles, ipart, density_function_correction) -> 1.0
    velocity = function (particles::Particles, ipart::Int)
        p = particles.pos[ipart]
        x = p[1] - bx * 0.5
        y = p[2] - by * 0.5
        r = sqrt(x * x + y * y)
        phi = atan(y, x)
        if r < 0.2
            return SVector{3,Float64}(-5.0 * r * sin(phi),
                                      5.0 * r * cos(phi), 0.0)
        elseif r < 0.4
            return SVector{3,Float64}(-(2.0 - 5.0 * r) * sin(phi),
                                      (2.0 - 5.0 * r) * cos(phi), 0.0)
        else
            return SVector{3,Float64}(0.0, 0.0, 0.0)
        end
    end
    function _gresho_internal_energy(particles::Particles, ipart::Int)::Float64
        gamma = GAMMA
        rho = 1.0
        p = particles.pos[ipart]
        x = p[1] - bx * 0.5
        y = p[2] - by * 0.5
        r = sqrt(x * x + y * y)
        if r < 0.2
            return (5.0 + 12.0 * r * r) / (gamma - 1.0) / rho
        elseif r < 0.4
            return (9.0 + 12.0 * r * r) / (gamma - 1.0) / rho
        else
            return (3.0 + 4.0 * log(2.0)) / (gamma - 1.0) / rho
        end
    end
    internal_energy = _gresho_internal_energy
    return Problem(
        "IC_Gresho",
        (bx, by, bz),
        (true, true, true),
        1.0,
        density,
        internal_energy,
        velocity,
        zero_vec,
        zero_postprocess!,
    )
end

# --- Problem 4.10 : Exponential Disk — flagged "does not work" -------------

function setup_exponential_disk(param::Parameters)
    error("problem 4.10 (Exponential Disk) is flagged \"does not work\" in " *
          "the C reference (ics.par); not ported (avoids shipping a wrong IC).")
end

# --- Problem 4.11 : Boss-Bodenheimer --------------------------------------

function setup_boss(param::Parameters)
    b = 0.032
    rho = 56458.857
    function _boss_density(particles::Particles, ipart::Int, density_function_correction)::Float64
        p = particles.pos[ipart]
        x = p[1] - b * 0.5
        y = p[2] - b * 0.5
        z = p[3] - b * 0.5
        r = sqrt(x * x + y * y + z * z)
        if r < 0.016
            return rho * (1.0 + 0.1 * cos(2.0 * atan(y, x)))
        else
            return rho * 0.05
        end
    end
    density = _boss_density
    return Problem(
        "IC_Boss",
        (b, b, b),
        (false, false, false),
        rho * 1.1,
        density,
        zero_U,
        zero_vec,
        zero_vec,
        zero_postprocess!,
    )
end

# --- Problem 4.12 : Isolated Galaxy Cluster --------------------------------

# halo β-model: Rho0=1e-26 g/cm^3, Beta=2/3, Rcore=20 kpc. Box 1000^3 kpc.
const _GC_RHO0 = 1e-26
const _GC_BETA = 2.0 / 3.0
const _GC_RCORE = 20.0

@inline _gc_betamodel(r) = _GC_RHO0 * (1.0 + _p2(r / _GC_RCORE))^(-1.5 * _GC_BETA)

function setup_galaxy_cluster(param::Parameters)
    b = 1000.0
    function _gc_density(particles::Particles, ipart::Int, density_function_correction)::Float64
        p = particles.pos[ipart]
        x = p[1] - 0.5 * b
        y = p[2] - 0.5 * b
        z = p[3] - 0.5 * b
        r = sqrt(x * x + y * y + z * z)
        return _gc_betamodel(r)
    end
    density = _gc_density
    return Problem(
        "IC_GalaxyCluster",
        (b, b, b),
        (true, true, true),
        _GC_RHO0,
        density,
        zero_U,            # no internal energy
        zero_vec,          # no velocity
        zero_vec,
        make_turbulent_postprocess(),   # turbulent B set here
    )
end

# --- Problem 5.0 : Fast Rotor ---------------------------------------------

function setup_rotor(param::Parameters)
    bx = 1.0; by = 1.0; bz = 0.1
    function _rotor_density(particles::Particles, ipart::Int, density_function_correction)::Float64
        p = particles.pos[ipart]
        x = p[1] - bx * 0.5
        y = p[2] - by * 0.5
        r = sqrt(_p2(x) + _p2(y))
        if 0.0 < r <= 0.1
            return 10.0
        elseif 0.1 < r <= 0.115
            return 1.0 + 9.0 * ((0.115 - r) / (0.115 - 0.1))
        else
            return 1.0
        end
    end
    density = _rotor_density
    velocity = function (particles::Particles, ipart::Int)
        p = particles.pos[ipart]
        x = p[1] - bx * 0.5
        y = p[2] - by * 0.5
        r = sqrt(_p2(x) + _p2(y))
        if 0.0 < r <= 0.1
            return SVector{3,Float64}(-2.0 * y / 0.1, 2.0 * x / 0.1, 0.0)
        elseif 0.1 < r <= 0.115
            f = (0.115 - r) / (0.115 - 0.1)
            return SVector{3,Float64}(-2.0 * y / 0.1 * f / r,
                                      2.0 * x / 0.1 * f / r, 0.0)
        else
            return SVector{3,Float64}(0.0, 0.0, 0.0)
        end
    end
    bfield = (particles, ipart) ->
        SVector{3,Float64}(5.0 / _SQRT4PI, 0.0, 0.0)
    function _rotor_internal_energy(particles::Particles, ipart::Int)::Float64
        gamma = GAMMA
        pressure = 1.0
        p = particles.pos[ipart]
        x = p[1] - bx * 0.5
        y = p[2] - by * 0.5
        r = sqrt(_p2(x) + _p2(y))
        if 0.0 < r <= 0.1
            return pressure / (gamma - 1.0) / 10.0
        elseif 0.1 < r <= 0.115
            return pressure / (gamma - 1.0) /
                   (1.0 + 9.0 * ((0.115 - r) / (0.115 - 0.1)))
        else
            return pressure / (gamma - 1.0) / 1.0
        end
    end
    internal_energy = _rotor_internal_energy
    return Problem(
        "IC_Rotor",
        (bx, by, bz),
        (true, true, true),
        10.0,
        density,
        internal_energy,
        velocity,
        bfield,
        zero_postprocess!,
    )
end

# --- Problem 5.1 : Strong Blast -------------------------------------------

function setup_strong_blast(param::Parameters)
    bx = 1.0; by = 1.0; bz = 0.1
    density = (particles, ipart, density_function_correction) -> 1.0
    velocity = (particles, ipart) -> SVector{3,Float64}(0.0, 0.0, 0.0)
    bfield = (particles, ipart) ->
        SVector{3,Float64}(1.0 / sqrt(2.0), 1.0 / sqrt(2.0), 0.0)
    function _sb_internal_energy(particles::Particles, ipart::Int)::Float64
        p = particles.pos[ipart]
        x = p[1] - bx * 0.5
        y = p[2] - by * 0.5
        radius = sqrt(_p2(x) + _p2(y))
        gamma = GAMMA
        pressure = 1.0
        if 0.0 < radius <= 0.1
            pressure = 10.0
        elseif radius > 0.1
            pressure = 0.1
        end
        return pressure / (gamma - 1.0) / particles.rho[ipart]
    end
    internal_energy = _sb_internal_energy
    return Problem(
        "IC_StrongBlast",
        (bx, by, bz),
        (true, true, true),
        1.0,
        density,
        internal_energy,
        velocity,
        bfield,
        zero_postprocess!,
    )
end

# --- Problem 5.2 : Orszag-Tang Vortex -------------------------------------

function setup_orszag_tang_vortex(param::Parameters)
    bx = 1.0; by = 1.0; bz = 0.1
    rho = 25.0 / (36.0 * pi)
    density = (particles, ipart, density_function_correction) -> rho
    velocity = function (particles::Particles, ipart::Int)
        p = particles.pos[ipart]
        x = p[1] / bx
        y = p[2] / by
        return SVector{3,Float64}(-sin(2.0 * pi * y), sin(2.0 * pi * x), 0.0)
    end
    bfield = function (particles::Particles, ipart::Int)
        p = particles.pos[ipart]
        x = p[1] / bx
        y = p[2] / by
        s = _SQRT4PI
        return SVector{3,Float64}(-sin(2.0 * pi * y) / s,
                                  sin(2.0 * pi * x) / s, 0.0)
    end
    function _ot_internal_energy(particles::Particles, ipart::Int)::Float64
        gamma = GAMMA
        pressure = 5.0 / 12.0 * pi
        return pressure / (gamma - 1.0) / rho
    end
    internal_energy = _ot_internal_energy
    return Problem(
        "IC_Orszag_Tang",
        (bx, by, bz),
        (true, true, true),
        rho,
        density,
        internal_energy,
        velocity,
        bfield,
        zero_postprocess!,
    )
end

# --- Problem 5.3 : Linear Alfven Wave -------------------------------------

function setup_linear_alfven_wave(param::Parameters)
    bx = 1.0; by = 0.1; bz = 0.1
    function _alfven_density(particles::Particles, ipart::Int, density_function_correction)::Float64
        x = particles.pos[ipart][1] / bx
        ret = 1.0 + 1e-6 * sin(2.0 * pi * x)
        ret += (ret - 1.0) * density_function_correction
        return ret
    end
    density = _alfven_density
    velocity = (particles, ipart) -> SVector{3,Float64}(0.0, 0.0, 0.0)
    bfield = function (particles::Particles, ipart::Int)
        s = _SQRT4PI
        return SVector{3,Float64}(s * 1.0, s * _SQRT2, s * 0.5)
    end
    function _alfven_internal_energy(particles::Particles, ipart::Int)::Float64
        gamma = GAMMA
        pressure = 1.0 / gamma
        return pressure / (gamma - 1.0) / particles.rho[ipart]
    end
    internal_energy = _alfven_internal_energy
    return Problem(
        "IC_LinearAlfvenWave",
        (bx, by, bz),
        (true, true, true),
        1.0 + 1e-6,
        density,
        internal_energy,
        velocity,
        bfield,
        zero_postprocess!,
    )
end

# --- Problem 5.4 : Rayleigh-Taylor Instability ----------------------------

function setup_rayleigh_taylor(param::Parameters)
    bx = 1.0; by = 0.5; bz = 0.1
    function _rt_density(particles::Particles, ipart::Int, density_function_correction)::Float64
        x = particles.pos[ipart][1] / bx
        rho1 = 1.0; rho2 = 2.0; delta = 0.025
        return rho1 + (rho2 - rho1) / (1.0 + exp(-(x - 0.5) / delta))
    end
    density = _rt_density
    velocity = function (particles::Particles, ipart::Int)
        p = particles.pos[ipart]
        x = p[1] / bx
        y = p[2] / by
        if x > 0.3 && x < 0.7
            vx = 0.025 * (1.0 + cos(8.0 * pi * (y + 0.25))) *
                 (1.0 + cos(5.0 * pi * (x - 0.5)))
            return SVector{3,Float64}(vx, 0.0, 0.0)
        else
            return SVector{3,Float64}(0.0, 0.0, 0.0)
        end
    end
    bfield = (particles, ipart) ->
        SVector{3,Float64}(0.0, _SQRT4PI * 0.07, 0.0)
    function _rt_internal_energy(particles::Particles, ipart::Int)::Float64
        gamma = 1.4
        rho2 = 2.0
        grav_acc = -0.5
        x = particles.pos[ipart][1] / bx
        rho = particles.rho[ipart]
        pressure = rho2 / gamma + grav_acc * rho * (x - 0.5)
        return pressure / (gamma - 1.0) / rho
    end
    internal_energy = _rt_internal_energy
    return Problem(
        "IC_RayleighTaylorInstability",
        (bx, by, bz),
        (true, true, true),
        2.0,
        density,
        internal_energy,
        velocity,
        bfield,
        zero_postprocess!,
    )
end

# --- Problem 5.5..5.16 : Ryu-Jones Shocktubes — flagged "not working" ------

const _RYU_JONES_NAMES = Dict(
     5 => "1A",  6 => "1B",  7 => "2A",  8 => "2B",
     9 => "3A", 10 => "3B", 11 => "4A", 12 => "4B",
    13 => "4C", 14 => "4D", 15 => "5A", 16 => "5B")

function _make_ryu_jones_error(subflag::Int)
    name = get(_RYU_JONES_NAMES, subflag, string(subflag))
    return function (param::Parameters)
        error("problem 5.$(subflag) (Ryu-Jones Shocktube $(name)) is " *
              "flagged \"not working\" in the C reference (ics.par); not " *
              "ported (avoids shipping a wrong IC).")
    end
end

# --- Problem 6.x : User-defined ics ---------------------------------------

function setup_user(param::Parameters)
    error("problem 6.$(param.Problem_Subflag) (user-defined IC, user.c) is " *
          "not ported — user.c is a per-user template with no fixed analytic " *
          "formula; define it in Julia directly if needed.")
end

# --- registry --------------------------------------------------------------

"""
    PROBLEM_REGISTRY :: Dict{Tuple{Int,Int}, Function}

Maps `(Problem_Flag, Problem_Subflag)` to a constructor
`param::Parameters -> Problem`, matching the `ics.par` header table. Problems
flagged "not working" / "not implemented" / "Error in result" are registered
to an explicit error, and the PNG path (2.1) to an out-of-scope error, so the
dispatch is complete but a wrong IC is never silently produced.
"""
const PROBLEM_REGISTRY = merge(Dict{Tuple{Int,Int},Function}(
    # 0.x — simple periodic tests
    (0, 0)  => setup_constant_density,
    (0, 1)  => setup_tophat,
    (0, 2)  => setup_sawtooth,
    (0, 3)  => setup_sinewave,
    # 1.x — simple non-periodic tests
    (1, 0)  => setup_gradient,
    # 2.x — logos
    (2, 0)  => setup_magneticum,
    (2, 1)  => setup_png,                       # OUT OF SCOPE (errors)
    # 3.x — double shock test
    (3, 0)  => setup_double_shock(0),
    (3, 1)  => setup_double_shock(1),
    (3, 2)  => setup_double_shock(2),
    # 4.x — classical hydrodynamics tests
    (4, 0)  => setup_sod_shock,
    (4, 1)  => setup_sedov_blast,
    (4, 2)  => setup_kelvin_helmholtz,
    (4, 3)  => setup_keplerian_ring,            # flagged: Error in result
    (4, 4)  => setup_blob,
    (4, 5)  => setup_hydrostatic_sphere,        # flagged: not implemented
    (4, 6)  => setup_evrard_collapse,
    (4, 7)  => setup_zeldovich_pancake,
    (4, 8)  => setup_box,
    (4, 9)  => setup_gresho_vortex,
    (4, 10) => setup_exponential_disk,          # flagged: does not work
    (4, 11) => setup_boss,
    (4, 12) => setup_galaxy_cluster,
    # 5.x — classical MHD tests
    (5, 0)  => setup_rotor,
    (5, 1)  => setup_strong_blast,
    (5, 2)  => setup_orszag_tang_vortex,
    (5, 3)  => setup_linear_alfven_wave,
    (5, 4)  => setup_rayleigh_taylor,
    ),
    # Ryu-Jones shocktubes 5.5..5.16: all flagged "not working", registered
    # to an explicit error so the dispatch stays complete.
    Dict{Tuple{Int,Int},Function}(
        (5, sub) => _make_ryu_jones_error(sub) for sub in 5:16),
)

"""
    setup_problem(param::Parameters) -> Problem

Dispatch on `(param.Problem_Flag, param.Problem_Subflag)` via
[`PROBLEM_REGISTRY`](@ref). Flag 6 (user-defined) dispatches to
`setup_user` for any subflag. Errors for an unregistered problem.
"""
function setup_problem(param::Parameters)
    key = (param.Problem_Flag, param.Problem_Subflag)
    if param.Problem_Flag == 6
        return setup_user(param)
    end
    haskey(PROBLEM_REGISTRY, key) || error(
        "Effect $(param.Problem_Flag).$(param.Problem_Subflag) not implemented")
    return PROBLEM_REGISTRY[key](param)
end
