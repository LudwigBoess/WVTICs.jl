# One-time environment bootstrap for the WVTICs package.
#
# Phase 0 could not run this itself (the background agent's sandbox denied
# all `julia` execution — see PORT_STATUS.md "Blocker"). Run it once:
#
#   julia --project=/home/moon/lboess/.julia/dev/WVTICs \
#         /home/moon/lboess/.julia/dev/WVTICs/bootstrap.jl
#
# It `dev`s the three local packages by path and instantiates the rest
# (all required registered deps are already present in the depot:
# NearestNeighbors, StaticArrays, FFTW; Distributed/Random/LinearAlgebra are
# stdlibs). Then it precompiles and runs the test suite.

using Pkg

Pkg.activate("/home/moon/lboess/.julia/dev/WVTICs")

Pkg.develop([
    Pkg.PackageSpec(path = "/home/moon/lboess/.julia/dev/GadgetIO"),
    Pkg.PackageSpec(path = "/home/moon/lboess/.julia/dev/GadgetUnits"),
    Pkg.PackageSpec(path = "/home/moon/lboess/.julia/dev/SPHKernels"),
])

Pkg.instantiate()
Pkg.precompile()

@info "Bootstrap complete. Running test suite…"
Pkg.test()
