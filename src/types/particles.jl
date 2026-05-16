# Struct-of-arrays particle container (CLAUDE.md §2, Option A — recommended).
#
# Replaces the buggy `Gas` PhysicalParticles stub. Mirrors the C `P` /`SphP`
# layout (`globals.h`):
#
#   struct ParticleData    { double Pos[3]; double Vel[3]; int32_t ID;
#                            int Type; peanoKey Key; int Tree_Parent;
#                            bool Redistributed; } *P;
#   struct GasParticleData { float U; float Rho; float Hsml; float VarHsmlFac;
#                            float ID; float Rho_Model; float Bfld[3]; } *SphP;
#
# SoA layout chosen because it is `NearestNeighbors.jl`'s fast input path, it
# vectorises, and it reorders cheaply (Peano / KDTree reorder). Positions are
# `SVector{3,Float64}`; the rest are `Float32` matching the C `SphP` types.
#
# `SphP.ID` in C is a (redundant) `float` copy of `P.ID`; it is preserved here
# as a parallel field for output/round-trip fidelity but `id` is the
# authoritative UInt32 ID.

using StaticArrays

"""
    Particles

Struct-of-arrays container for all SPH particles. All vectors are kept the
same length (`Npart`); index `i` addresses particle `i` consistently across
every field.

Position / vector fields (C `P`):
- `pos::Vector{SVector{3,Float64}}`  — `P.Pos` (double precision)
- `vel::Vector{SVector{3,Float32}}`  — `P.Vel`
- `id::Vector{UInt32}`               — `P.ID` (authoritative, UInt32)
- `type::Vector{Int32}`              — `P.Type` (all 0 = gas here)
- `key::Vector{UInt128}`             — `P.Key` (128-bit Peano key; Phase 2+)
- `tree_parent::Vector{Int32}`       — `P.Tree_Parent` (Phase 2+)
- `redistributed::Vector{Bool}`      — `P.Redistributed`

Gas fields (C `SphP`), all `Float32` to match the C struct:
- `u::Vector{Float32}`               — `SphP.U`
- `rho::Vector{Float32}`             — `SphP.Rho`
- `hsml::Vector{Float32}`            — `SphP.Hsml`
- `varhsmlfac::Vector{Float32}`      — `SphP.VarHsmlFac`
- `rho_model::Vector{Float32}`       — `SphP.Rho_Model`
- `bfld::Vector{SVector{3,Float32}}` — `SphP.Bfld`

Construct an all-zero container of length `n` with `Particles(n)`.
"""
struct Particles
    pos::Vector{SVector{3,Float64}}
    vel::Vector{SVector{3,Float32}}
    id::Vector{UInt32}
    type::Vector{Int32}
    key::Vector{UInt128}
    tree_parent::Vector{Int32}
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
particles (all type 0 / gas, matching the C code where every particle is gas).
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
        zeros(Int32, nn),
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
