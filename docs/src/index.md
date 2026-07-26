```@meta
CurrentModule = WVTICs
```

# WVTICs.jl

*Weighted Voronoi Tessellation initial conditions for SPH.*

WVTICs generates low-noise, glass-like initial conditions for smoothed-particle
hydrodynamics. Given a target density field ``\rho(\mathbf{x})`` it relaxes a
set of particle positions until their SPH density estimate matches
``\rho`` — a *relaxed glass* with far less sampling noise than a random or
lattice draw — then applies the problem's velocity, internal-energy and
magnetic-field fields and writes a Gadget-2 (`SnapFormat 2`) snapshot.

It is a Julia port of the WVT ICs generator by J. Donnert; see the
[Arth et al. (2019)](https://arxiv.org/abs/1907.11250) method paper.

## The method in brief

Each iteration:

1. solve the SPH smoothing length and density for every particle
   (Newton–Raphson on the kernel-weighted neighbour count);
2. push every particle away from its neighbours with a repulsive,
   kernel-weighted force scaled so the local particle spacing tracks the target
   density;
3. periodically apply a Monte-Carlo *redistribution* step (Metropolis) that
   moves over-dense particles next to under-dense ones, to escape local minima;
4. shrink the step size as the configuration converges.

The result is a particle distribution whose number density ``\propto \rho``,
suitable as an SPH IC.

## Installation

```julia
using Pkg
Pkg.add(url = "https://github.com/LudwigBoess/WVTICs.jl")
```

## Quick start

Write a parameter file (see [Usage](@ref) for the full schema) — or use the
bundled `parameters.toml` (a 2000-particle constant-density smoke test) — and
run the driver:

```julia
using WVTICs

# Reads the parameter file, runs the full pipeline, writes the snapshot,
# and returns the relaxed `Particles`.
particles = make_sph_wvtics("parameters.toml")
```

`make_sph_wvtics` is the only exported symbol; everything else is reachable via
qualification (`WVTICs.KernelConfig`) or `using WVTICs: KernelConfig`.

Choose a different SPH kernel at the call site:

```julia
using WVTICs: KernelConfig, WendlandC6
make_sph_wvtics("ics.par"; kernel = KernelConfig(WendlandC6; dim = 3))
```

Run multi-threaded by starting Julia with threads — the per-particle loops
scale across them automatically:

```sh
julia -t 8 --project -e 'using WVTICs; make_sph_wvtics("ics.par")'
```

## Features

- Target-density relaxation with periodic Monte-Carlo redistribution.
- Any [`SPHKernels.jl`](https://github.com/LudwigBoess/SPHKernels.jl) kernel
  (Cubic spline, Wendland C2/C4/C6/C8) with a runtime-selectable neighbour
  count; genuine Dehnen–Aly kernel self-bias correction.
- A registry of ready-made test problems (constant density, Kelvin–Helmholtz,
  Sod, Sedov, Gresho, Orszag–Tang, …) selected by two integer flags.
- Rejection or uniform position sampling.
- Divergence-free turbulent magnetic-field generation.
- ASCII (`tag value`) and TOML parameter files.
- Gadget `SnapFormat 2` output via
  [`GadgetIO.jl`](https://github.com/LudwigBoess/GadgetIO.jl), including custom
  model-density (`RHOM`) and redistribution (`REDI`) blocks.
- Shared-memory threading across every hot loop.

## Contents

```@contents
Pages = ["usage.md", "problems.md", "parallel.md", "api.md"]
Depth = 2
```
