# Write_output — Julia port of `io.c::Write_output` / `write_header` /
# `add_block` / `set_block_info` / `fill_write_buffer`, using GadgetIO's
# Gadget SnapFormat-2 writer.
#
# Block order (io.h `enum iofields`), all particles type 0 (gas):
#   default            : POS, VEL, ID, RHO, RHOM, HSML, U, BFLD [, REDI]
#   SPH_CUBIC_SPLINE   : POS, VEL, ID, U,  RHO,  HSML, BFLD, RHOM [, REDI]
# `RHOM` ("Model Density") and `REDI` ("Redistributed") are custom blocks not
# in GadgetIO's default INFO tables -> explicit `InfoLine`s are written.
# The active C Makefile uses SPH_WC4 (NOT cubic spline) and has
# OUTPUT_DIAGNOSTICS on by default, so the default order incl. REDI is the
# Phase-1 default here; the cubic-spline reordering is selected from the
# kernel config.
#
# Element types (io.c::set_block_info / fill_write_buffer):
#   POS/VEL/BFLD : Float32, 3 components   (C casts double Pos/Vel to float)
#   ID           : UInt32
#   RHO/RHOM/HSML/U : Float32, 1 component
#   REDI         : Int32   (C writes P.Redistributed as int)

"""
    snapshot_block_order(kcfg::KernelConfig; output_diagnostics=true)
        -> Vector{Symbol}

The Gadget-2 block order from `io.h`'s `enum iofields`. Cubic spline uses the
reordered enum (`POS,VEL,ID,U,RHO,HSML,BFLD,RHOM`), every other kernel uses
the default (`POS,VEL,ID,RHO,RHOM,HSML,U,BFLD`). `REDI` is appended iff
`output_diagnostics` (C `OUTPUT_DIAGNOSTICS`, default on).
"""
function snapshot_block_order(kcfg::KernelConfig; output_diagnostics::Bool = true)
    order = if kcfg.kernel isa SPHKernels.Cubic
        Symbol[:POS, :VEL, :ID, :U, :RHO, :HSML, :BFLD, :RHOM]
    else
        Symbol[:POS, :VEL, :ID, :RHO, :RHOM, :HSML, :U, :BFLD]
    end
    output_diagnostics && push!(order, :REDI)
    return order
end

# 4-char Gadget block label for each enum field (io.c::set_block_info).
const _BLOCK_LABEL = Dict{Symbol,String}(
    :POS  => "POS",
    :VEL  => "VEL",
    :ID   => "ID",
    :RHO  => "RHO",
    :RHOM => "RHOM",
    :HSML => "HSML",
    :U    => "U",
    :BFLD => "BFLD",
    :REDI => "REDI",
)

"""
    snapshot_info_lines(order::Vector{Symbol}) -> Vector{InfoLine}

Build the `INFO` block entries (one per data block, gas-only `is_present`),
including the explicit custom `RHOM` / `REDI` lines (CLAUDE.md §1.8). Data
types and dimensionality mirror `io.c::set_block_info`/`fill_write_buffer`.
"""
function snapshot_info_lines(order::Vector{Symbol})
    gas = Int32[1, 0, 0, 0, 0, 0]
    lines = InfoLine[]
    for b in order
        if b === :POS || b === :VEL || b === :BFLD
            push!(lines, InfoLine(_BLOCK_LABEL[b], Float32, Int32(3), copy(gas)))
        elseif b === :ID
            push!(lines, InfoLine("ID", UInt32, Int32(1), copy(gas)))
        elseif b === :REDI
            # C writes P.Redistributed via ((int*)wbuf); GadgetIO's INFO
            # writer only encodes Float32/Float64/UInt32/UInt64 datatype
            # names, so REDI is written as UInt32 (4-byte, like C int) for a
            # round-trippable diagnostics-only block.
            push!(lines, InfoLine("REDI", UInt32, Int32(1), copy(gas)))
        else # RHO, RHOM, HSML, U  -> Float32 scalar
            push!(lines, InfoLine(_BLOCK_LABEL[b], Float32, Int32(1), copy(gas)))
        end
    end
    return lines
end

# Build the write buffer for one block (io.c::fill_write_buffer). POS/VEL/BFLD
# become a 3xN Float32 matrix (GadgetIO writes Matrix column-major as N
# vectors of length size(data,1)); scalars become a length-N vector.
function _block_data(particles::Particles, b::Symbol, n::Int)
    if b === :POS
        d = Matrix{Float32}(undef, 3, n)
        @inbounds for i in 1:n
            p = particles.pos[i]
            d[1, i] = Float32(p[1]); d[2, i] = Float32(p[2]); d[3, i] = Float32(p[3])
        end
        return d
    elseif b === :VEL
        d = Matrix{Float32}(undef, 3, n)
        @inbounds for i in 1:n
            v = particles.vel[i]
            d[1, i] = v[1]; d[2, i] = v[2]; d[3, i] = v[3]
        end
        return d
    elseif b === :BFLD
        d = Matrix{Float32}(undef, 3, n)
        @inbounds for i in 1:n
            v = particles.bfld[i]
            d[1, i] = v[1]; d[2, i] = v[2]; d[3, i] = v[3]
        end
        return d
    elseif b === :ID
        return copy(particles.id)::Vector{UInt32}
    elseif b === :RHO
        return copy(particles.rho)::Vector{Float32}
    elseif b === :RHOM
        return copy(particles.rho_model)::Vector{Float32}
    elseif b === :HSML
        return copy(particles.hsml)::Vector{Float32}
    elseif b === :U
        return copy(particles.u)::Vector{Float32}
    elseif b === :REDI
        return UInt32.(particles.redistributed)::Vector{UInt32}
    else
        error("Block not found $b")   # mirrors io.c Assert(0,"Block not found")
    end
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
                 verbose = true, kernel = default_kernel_config(),
                 output_diagnostics = true, filename = problem.Name)

Port of `io.c::Write_output`. Writes a Gadget **SnapFormat-2** snapshot:
`HEAD` block, then a custom `INFO` block (so the custom `RHOM`/`REDI` blocks
are self-describing and round-trippable), then the data blocks in the exact
`io.h` order for the selected kernel (default vs. cubic-spline reordering;
`REDI` appended iff `output_diagnostics`, matching C `OUTPUT_DIAGNOSTICS`).

`filename` defaults to `problem.Name` (the C code writes to `Problem.Name` in
the current working directory). Returns the path written.
"""
function write_output(particles::Particles, param::Parameters,
                       problem::ProblemParameters;
                       verbose::Bool = true,
                       kernel::KernelConfig = default_kernel_config(),
                       output_diagnostics::Bool = true,
                       filename::AbstractString = problem.Name)
    n = param.Npart
    order = snapshot_block_order(kernel; output_diagnostics = output_diagnostics)
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
            write_block(f, data, _BLOCK_LABEL[b]; snap_format = 2)
        end
    finally
        close(f)
    end

    verbose && println("done")
    return filename
end
