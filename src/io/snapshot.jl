# Write_output — Julia port of `io.c::Write_output` / `write_header` /
# `add_block` / `set_block_info` / `fill_write_buffer`, using GadgetIO's
# Gadget SnapFormat-2 writer.
#
# Block order (all particles type 0, gas):
#   POS, VEL, ID, HSML, RHO, U, BFLD, RHOM [, REDI]
# A single order is used for every kernel. SnapFormat-2 tags each data block
# with a 4-char label and readers seek by label, so the block sequence is not
# semantically significant; the C `io.h` `#ifdef SPH_CUBIC_SPLINE` positional
# reordering is unnecessary for a label-addressed reader.
# `RHOM` ("Model Density") and `REDI` ("Redistributed") are custom blocks not
# in GadgetIO's default INFO tables -> explicit `InfoLine`s are written.
# `REDI` is appended iff OUTPUT_DIAGNOSTICS (default on).
#
# Element types (io.c::set_block_info / fill_write_buffer):
#   POS/VEL/BFLD : Float32, 3 components   (C casts double Pos/Vel to float)
#   ID           : UInt32
#   RHO/RHOM/HSML/U : Float32, 1 component
#   REDI         : Int32   (C writes P.Redistributed as int)

"""
    snapshot_block_order(; output_diagnostics=true) -> Vector{Symbol}

The Gadget-2 data-block order `POS, VEL, ID, HSML, RHO, U, BFLD, RHOM`, with
`REDI` appended iff `output_diagnostics` (C `OUTPUT_DIAGNOSTICS`, default on).
One order is used for all kernels: SnapFormat-2 blocks are label-addressed, so
the C `#ifdef SPH_CUBIC_SPLINE` positional reordering is not needed.
"""
function snapshot_block_order(; output_diagnostics::Bool = true)
    order = Symbol[b[1] for b in _SNAPSHOT_BLOCKS if b[1] !== :REDI]
    output_diagnostics && push!(order, :REDI)
    return order
end

# One spec per data block: (symbol, 4-char label, element type, #components,
# Particles source field). The data-block order is this table's order; the
# diagnostics-only REDI block is appended by `snapshot_block_order`. POS/VEL/
# BFLD are 3-component Float32 (POS casts the Float64 positions); ID and REDI
# are UInt32 (C writes Redistributed as an int); RHO/RHOM/HSML/U are Float32.
const _SNAPSHOT_BLOCKS = (
    (:POS,  "POS",  Float32, 3, :pos),
    (:VEL,  "VEL",  Float32, 3, :vel),
    (:ID,   "ID",   UInt32,  1, :id),
    (:HSML, "HSML", Float32, 1, :hsml),
    (:RHO,  "RHO",  Float32, 1, :rho),
    (:U,    "U",    Float32, 1, :u),
    (:BFLD, "BFLD", Float32, 3, :bfld),
    (:RHOM, "RHOM", Float32, 1, :rho_model),
    (:REDI, "REDI", UInt32,  1, :redistributed),
)

const _BLOCK_SPEC = Dict{Symbol,Tuple{String,DataType,Int,Symbol}}(
    b[1] => (b[2], b[3], b[4], b[5]) for b in _SNAPSHOT_BLOCKS)

_block_label(b::Symbol) = _BLOCK_SPEC[b][1]

"""
    snapshot_info_lines(order::Vector{Symbol}) -> Vector{InfoLine}

Build the `INFO` block entries (one per data block, gas-only `is_present`),
including the custom `RHOM` / `REDI` lines. Element types and component counts
come from `_SNAPSHOT_BLOCKS`.
"""
function snapshot_info_lines(order::Vector{Symbol})
    gas = Int32[1, 0, 0, 0, 0, 0]
    return [let (label, T, dim, _) = _BLOCK_SPEC[b]
                InfoLine(label, T, Int32(dim), copy(gas))
            end for b in order]
end

# Float32 3xN matrix from a vector of 3-component SVectors (POS casts Float64).
function _vec3_matrix(src::AbstractVector, n::Int)
    d = Matrix{Float32}(undef, 3, n)
    @inbounds for i in 1:n
        v = src[i]
        d[1, i] = Float32(v[1]); d[2, i] = Float32(v[2]); d[3, i] = Float32(v[3])
    end
    return d
end

# Write buffer for one block (io.c::fill_write_buffer): a 3xN Float32 matrix
# for the 3-component blocks, else a length-N vector of the block's element
# type (ID/REDI → UInt32; RHO/RHOM/HSML/U → Float32).
function _block_data(particles::Particles, b::Symbol, n::Int)
    spec = get(_BLOCK_SPEC, b, nothing)
    spec === nothing && error("Block not found $b")   # io.c Assert(0,...)
    _, T, dim, src = spec
    field = getfield(particles, src)
    return dim == 3 ? _vec3_matrix(field, n) : T.(field)
end

"""
    build_snapshot_header(param, problem) -> SnapshotHeader

Port of `io.c::write_header`. All particles are type 0 (gas):
`npart[1] = nall[1] = Npart`, `massarr[1] = Mpart`, `boxsize = Boxsize[1]`
(C `Problem.Boxsize[0]`, the largest axis), cosmology fields 0,
`num_files = 1`. (`SnapshotHeader` vectors are 1-based; index 1 == C type 0.)
"""
function build_snapshot_header(param::Parameters, problem::ProblemParameters)
    npart = zeros(Int32, 6)
    nall = zeros(UInt32, 6)
    massarr = zeros(Float64, 6)
    npart[1] = Int32(param.Npart)
    nall[1] = UInt32(param.Npart)
    massarr[1] = problem.Mpart

    return SnapshotHeader(
        npart,                       # npart
        massarr,                     # massarr
        0.0,                         # time
        0.0,                         # z (redshift)
        Int32(0),                    # flag_sfr
        Int32(0),                    # flag_feedback
        nall,                        # nall
        Int32(0),                    # flag_cooling
        Int32(1),                    # num_files
        problem.Boxsize[1],          # boxsize (C Header.BoxSize = Boxsize[0])
        0.0,                         # omega_0
        0.0,                         # omega_l
        0.0,                         # h0
        Int32(0),                    # flag_stellarage
        Int32(0),                    # flag_metals
        zeros(UInt32, 6),            # npartTotalHighWord
        Int32(0),                    # flag_entropy_instead_u
        Int32(0),                    # flag_doubleprecision
        Int32(0),                    # flag_ic_info
        Float32(0.0),                # lpt_scalingfactor
        zeros(Int32, 12),            # fill
    )
end

"""
    write_output(particles, param, problem;
                 verbose = true, output_diagnostics = true,
                 filename = problem.Name)

Port of `io.c::Write_output`. Writes a Gadget **SnapFormat-2** snapshot:
`HEAD` block, then a custom `INFO` block (so the custom `RHOM`/`REDI` blocks
are self-describing and round-trippable), then the data blocks in the fixed
`io.h` order (`REDI` appended iff `output_diagnostics`, matching C
`OUTPUT_DIAGNOSTICS`).

`filename` defaults to `problem.Name` (the C code writes to `Problem.Name` in
the current working directory). Returns the path written.
"""
function write_output(particles::Particles, param::Parameters,
                       problem::ProblemParameters;
                       verbose::Bool = true,
                       output_diagnostics::Bool = true,
                       filename::AbstractString = problem.Name)
    n = param.Npart
    order = snapshot_block_order(; output_diagnostics = output_diagnostics)
    header = build_snapshot_header(param, problem)
    info = snapshot_info_lines(order)

    if verbose
        println("Output :")
        println("   File Name = ", filename)
        println("   Npart     = ", n)
        println("   Blocks    = ", join(string.(order), ", "))
    end

    f = open(filename, "w")
    try
        write_header(f, header; snap_format = 2)
        write_info_block(f, info; snap_format = 2)
        for b in order
            data = _block_data(particles, b, n)
            write_block(f, data, _block_label(b); snap_format = 2)
        end
    finally
        close(f)
    end

    verbose && println("done")
    return filename
end
