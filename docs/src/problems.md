```@meta
CurrentModule = WVTICs
```

# Problems

A *problem* defines the physics of an IC: the target density field and the
velocity, internal-energy, magnetic-field and post-processing callbacks, plus
the box geometry. Problems are selected by the two integer parameter-file tags
`Problem_Flag` and `Problem_Subflag`, looked up in `PROBLEM_REGISTRY`
(`(flag, subflag) → constructor`) by [`setup_problem`](@ref).

## Built-in problems

| Flag.Sub | Problem | Notes |
|---|---|---|
| 0.0 | Constant density | unit periodic box (the default / smoke test) |
| 0.1 | Top-hat density | periodic |
| 0.2 | Sawtooth density | periodic |
| 0.3 | Sine-wave density | periodic |
| 1.0 | Gradient density | non-periodic tests |
| 2.0 | Magneticum logo | turbulent B via post-processing |
| 3.0 / 3.1 / 3.2 | Double shock | Mach 2 / 3 / 4 |
| 4.0 | Sod shock tube | |
| 4.1 | Sedov blast | |
| 4.2 | Kelvin–Helmholtz | |
| 4.4 | Blob | |
| 4.6 | Evrard collapse | |
| 4.7 | Zel'dovich pancake | |
| 4.8 | Box | |
| 4.9 | Gresho vortex | |
| 4.11 | Boss | |
| 4.12 | Galaxy cluster | turbulent B via post-processing |
| 5.0 | MHD rotor | |
| 5.1 | Strong blast | |
| 5.2 | Orszag–Tang vortex | |
| 5.3 | Linear Alfvén wave | |
| 5.4 | Rayleigh–Taylor | |

### Unsupported / not ported

These `(flag, subflag)` values are registered but raise an informative error
rather than silently produce a wrong IC:

- **2.1** PNG logo — out of scope for this port.
- **4.3** Keplerian ring, **4.5** hydrostatic sphere, **4.10** exponential
  disk — flagged unusable in the reference.
- **5.5 … 5.16** Ryu–Jones shocktubes — flagged "not working" in the reference.
- **6.x** user-defined ICs — define directly in Julia (see below) rather than
  through the C template.

## Selecting a problem

Set the two flags in the parameter file:

```toml
Problem_Flag    = 4
Problem_Subflag = 2      # Kelvin–Helmholtz
```

## Adding a custom problem

A [`Problem`](@ref) holds five callbacks plus geometry:

```julia
using WVTICs
using WVTICs: Problem, zero_U, zero_vec, zero_postprocess!
using StaticArrays

# density(particles, ipart, density_function_correction) -> Float64
my_density(p, i, dfc) = 1.0 + 0.1 * sin(2π * p.pos[i][1])

prob = Problem(
    "IC_MyProblem",            # output name
    (1.0, 1.0, 1.0),           # boxsize (axis 1 must be the largest)
    (true, true, true),        # periodicity per axis
    1.2,                       # Rho_Max (upper bound for rejection sampling)
    my_density,
    zero_U,                    # internal_energy(particles, ipart)
    zero_vec,                  # velocity(particles, ipart) -> SVector{3}
    zero_vec,                  # bfield(particles, ipart)   -> SVector{3}
    zero_postprocess!,         # postprocess!(particles, param, problem)
)
```

Register it under an unused `(flag, subflag)` so the parameter-file path can
reach it:

```julia
WVTICs.PROBLEM_REGISTRY[(9, 0)] = _ -> prob
```

The `density` callback receives the particle index and reads the current
position from `particles.pos[ipart]`; `density_function_correction` is the
artificial density-model correction (pass-through for most problems). The box
invariant is that `boxsize[1]` is the largest axis — [`setup`](@ref) asserts it,
because the neighbour search relies on it.
```
