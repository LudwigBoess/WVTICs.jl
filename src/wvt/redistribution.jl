# Metropolis-style particle redistribution.
#
# Algorithm:
#   * pick a random untouched (`!redistributed`) particle whose density is
#     above the model (high-energy state); accept it for movement with
#     probability `erf(relErrSigned)`;
#   * pick a random untouched underdense particle as a target, accepted with
#     probability `-relErrSigned`;
#   * move the chosen particle to within `0.3·Hsml` of the target, per-axis,
#     re-drawing until the coordinate is in `[0, Boxsize)` (periodic axes wrap;
#     non-periodic axes reject-and-redraw);
#   * mark the moved particle `redistributed = true`;
#   * stop once `move_part` particles are moved OR `max_probes` probes are spent.
#   `relErrSigned(i) = (Rho - rhoModel) / rhoModel`,
#   `rhoModel = prob.density(i, density_function_correction)`.
#
# Selection is serial with a single per-call `Xoshiro` RNG (base seed
# `14041981`): race-free and statistically equivalent to a parallel selection.
# The Metropolis structure, the `erf`/`-relErr` accept probabilities, the
# `0.3·Hsml` proximity move, the probe/move bounds, and the `redistributed`
# exclusivity are all preserved.

using Random

# --- erf without SpecialFunctions (Abramowitz & Stegun 7.1.26) -------------
# Max abs error ≈ 1.5e-7 on the whole real line — far inside the statistical
# tolerance of the redistribution accept test (a single uniform-vs-prob draw).
# `erf` is odd: erf(-x) = -erf(x).
@inline function _erf(x::Float64)
    s = x < 0 ? -1.0 : 1.0
    ax = abs(x)
    t = 1.0 / (1.0 + 0.3275911 * ax)
    y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t -
                0.284496736) * t + 0.254829592) * t * exp(-ax * ax)
    return s * y
end

"""
    relative_density_error_signed(particles, prob, ipart,
                                  density_function_correction) -> Float64

Signed relative density error `(Rho - rhoModel) / rhoModel` with
`rhoModel = prob.density(particles, ipart, density_function_correction)`.
Ranges from −1 (ρ=0) to +∞ (ρ→∞).
"""
# Inner form taking the density callback as `dfun::F`, so hot callers (the
# redistribution probe loops) specialise on its concrete type instead of
# dispatching on the abstract `prob.density` field each call.
@inline function _rel_density_error_signed(particles::Particles, dfun::F,
                                           ipart::Int,
                                           density_function_correction) where {F}
    rho_model = Float64(dfun(particles, ipart, density_function_correction))
    return (Float64(particles.rho[ipart]) - rho_model) / rho_model
end

@inline relative_density_error_signed(particles::Particles, prob::Problem,
                                       ipart::Int, density_function_correction) =
    _rel_density_error_signed(particles, prob.density, ipart,
                              density_function_correction)

"""
    relative_density_error(particles, prob, ipart,
                           density_function_correction) -> Float64

Absolute value of [`relative_density_error_signed`](@ref).  Used by the WVT
loop's error stats.
"""
@inline function relative_density_error(particles::Particles, prob::Problem,
                                         ipart::Int,
                                         density_function_correction)
    return abs(relative_density_error_signed(particles, prob, ipart,
                                             density_function_correction))
end

"""
    reset_redistribution_flags!(particles)

Clear every `particles.redistributed[i]` to `false`.
"""
function reset_redistribution_flags!(particles::Particles)
    fill!(particles.redistributed, false)
    return nothing
end

# Random 1-based particle index in [1, N]: `floor(rand()*N) + 1`, clamped
# (rand can be 1.0-ε but never 1).
@inline _random_particle(rng, n::Int) = min(n, floor(Int, rand(rng) * n) + 1)

# Draw a coordinate within ±0.3·Hsml of particle `jpart` along `axis`,
# re-drawing until it lands in [0, Boxsize); periodic axes wrap a single
# out-of-range step (if ret >= L, ret -= L; else if ret < 0, ret += L).
@inline function _position_in_proximity(rng, particles::Particles,
                                         jpart::Int, axis::Int,
                                         boxsize::NTuple{3,Float64},
                                         periodic::NTuple{3,Bool})
    L = boxsize[axis]
    per = periodic[axis]
    base = particles.pos[jpart][axis]
    h = 0.3 * Float64(particles.hsml[jpart])
    ret = -1.0
    # bounded loop: degenerate Hsml=0 would loop forever; cap and fall back to
    # the target's own (in-box) coordinate.
    for _ in 1:1024
        ret = base + (2.0 * rand(rng) - 1.0) * h
        if per
            if ret >= L
                ret -= L
            elseif ret < 0.0
                ret += L
            end
        end
        if 0.0 <= ret < L
            return ret
        end
    end
    return clamp(base, 0.0, prevfloat(L))
end

"""
    redistribute_particles!(particles, param, problem, prob, move_part,
                            max_probes; seed = RNG_BASE_SEED) -> (moved, probes)

Performs the Metropolis redistribution: up to `move_part` overdense untouched
particles are accepted (prob `erf(relErr)`) and relocated to within `0.3·Hsml`
of a randomly chosen underdense untouched particle, bounded by `max_probes`
probes.  Mutates `particles.pos` and `particles.redistributed`.  Returns
`(n_redistributed, n_probes)`.

Serial selection with one `Xoshiro(seed)` RNG — see the note at the top of
this file.  Density/`rho` are not recomputed during the process (the moves are
a small fraction); the caller re-runs `find_sph_quantities!` afterwards.
"""
function redistribute_particles!(particles::Particles, param::Parameters,
                                 problem::ProblemParameters, prob::Problem,
                                 move_part::Int, max_probes::Int;
                                 seed::Integer = RNG_BASE_SEED)
    n = param.Npart
    (n <= 0 || move_part <= 0 || max_probes <= 0) && return (0, 0)
    return _redistribute!(particles, prob.density, n,
                          param.density_function_correction,
                          problem.Boxsize, problem.Periodic,
                          move_part, max_probes, seed)
end

# Function barrier holding the whole probe loop, specialised on the density
# callback type `F` so the accept-test density calls dispatch statically.
@noinline function _redistribute!(particles::Particles, dfun::F, n::Int,
                                  density_function_correction,
                                  box::NTuple{3,Float64}, per::NTuple{3,Bool},
                                  move_part::Int, max_probes::Int,
                                  seed::Integer) where {F}
    rng = Random.Xoshiro(seed)

    redist = 0
    probes = 0

    # attempt up to move_part moves, sharing the probe budget.
    for _ in 1:move_part
        probes >= max_probes && break

        # pick a particle to redistribute: random untouched particle, accepted
        # with prob erf(relErrSigned); each accept test consumes a probe.
        ipart = -1
        while true
            cand = _random_particle(rng, n)
            while particles.redistributed[cand]
                cand = _random_particle(rng, n)
            end
            if probes >= max_probes
                ipart = -1
                break
            end
            probes += 1
            if rand(rng) < _erf(_rel_density_error_signed(particles, dfun,
                                              cand, density_function_correction))
                # accepted for movement: claim it.
                particles.redistributed[cand] = true
                ipart = cand
                break
            end
            # rejected: redraw
        end
        ipart < 0 && break

        # pick a target location: random untouched particle accepted with
        # prob (-relErrSigned) (i.e. underdense).
        jpart = _random_particle(rng, n)
        guard = 0
        while particles.redistributed[jpart] ||
              !(rand(rng) < -_rel_density_error_signed(particles, dfun,
                                            jpart, density_function_correction))
            jpart = _random_particle(rng, n)
            guard += 1
            guard > 64 * n && break   # safety: no acceptable target exists
        end

        # move the particle into the target's neighbourhood
        px = _position_in_proximity(rng, particles, jpart, 1, box, per)
        py = _position_in_proximity(rng, particles, jpart, 2, box, per)
        pz = _position_in_proximity(rng, particles, jpart, 3, box, per)
        particles.pos[ipart] = SVector{3,Float64}(px, py, pz)
        redist += 1
    end

    return (redist, probes)
end
