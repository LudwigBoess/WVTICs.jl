# Struct-of-arrays particle container.
#
# SoA layout chosen because it is `NearestNeighbors.jl`'s fast input path, it
# vectorises, and it reorders cheaply (Peano / KDTree reorder). Positions are
# `SVector{3,Float64}`; the rest are `Float32`.

using StaticArrays

"""
    Particles

Struct-of-arrays container for all SPH particles. All vectors are kept the
same length (`Npart`); index `i` addresses particle `i` consistently across
every field.

Position / vector fields:
- `pos::Vector{SVector{3,Float64}}`  — positions (double precision)
- `vel::Vector{SVector{3,Float32}}`  — velocities
- `id::Vector{UInt32}`               — particle IDs (authoritative, UInt32)
- `type::Vector{Int32}`              — particle type (all 0 = gas here)
- `key::Vector{UInt128}`             — 128-bit Peano key (domain decomposition)
- `redistributed::Vector{Bool}`      — redistribution flag

Gas fields, all `Float32`:
- `u::Vector{Float32}`               — internal energy
- `rho::Vector{Float32}`             — density
- `hsml::Vector{Float32}`            — smoothing length
- `varhsmlfac::Vector{Float32}`      — variable-hsml factor
- `rho_model::Vector{Float32}`       — model (target) density
- `bfld::Vector{SVector{3,Float32}}` — magnetic field

Construct an all-zero container of length `n` with `Particles(n)`.
"""
struct Particles
    pos::Vector{SVector{3,Float64}}
    vel::Vector{SVector{3,Float32}}
    id::Vector{UInt32}
    type::Vector{Int32}
    key::Vector{UInt128}
    redistributed::Vector{Bool}

    u::Vector{Float32}
    rho::Vector{Float32}
    hsml::Vector{Float32}
    varhsmlfac::Vector{Float32}
    rho_model::Vector{Float32}
    bfld::Vector{SVector{3,Float32}}
end

"""
    Particles(n::Integer)

Allocate a zero-initialised [`Particles`](@ref) container holding `n`
particles (all type 0 / gas).
"""
function Particles(n::Integer)
    n >= 0 || throw(ArgumentError("number of particles must be ≥ 0, got $n"))
    nn = Int(n)
    return Particles(
        fill(zero(SVector{3,Float64}), nn),
        fill(zero(SVector{3,Float32}), nn),
        zeros(UInt32, nn),
        zeros(Int32, nn),
        zeros(UInt128, nn),
        zeros(Bool, nn),
        zeros(Float32, nn),
        zeros(Float32, nn),
        zeros(Float32, nn),
        zeros(Float32, nn),
        zeros(Float32, nn),
        fill(zero(SVector{3,Float32}), nn),
    )
end

"""
    Base.length(p::Particles)

Number of particles in the container (length of any parallel field).
"""
Base.length(p::Particles) = length(p.pos)

Base.eachindex(p::Particles) = Base.OneTo(length(p))
