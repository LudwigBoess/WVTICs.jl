```@meta
CurrentModule = WVTICs
```

# Parallelism

## Shared memory (threads)

Every hot per-particle loop — the density solve, the WVT displacement, the
error/statistics reductions, and the mass integral — is threaded. Just start
Julia with threads; nothing else is required:

```sh
julia -t 8 --project -e 'using WVTICs; make_sph_wvtics("ics.par")'
```

Work is split into `nthreads()` contiguous chunks, each writing only its own
scratch/accumulator slot (indexed by the chunk number, never `threadid()`), so
results are deterministic for a fixed thread count and independent of the
threading backend. The backend is selectable in `src/parallel/threads.jl`
(`Base.Threads`, `Polyester`, or `OhMyThreads`); the default is `Polyester`,
with the load-imbalanced density solve overridden to `Base.Threads` dynamic
scheduling.

This is the recommended way to scale WVTICs on a single node.

## Distributed memory (experimental)

!!! warning "In development"
    The distributed **launch, domain-decomposition, ghost-selection, reduction
    and multi-file-IO helpers are implemented and unit-tested**, and
    [`regularise_sph_particles_distributed!`](@ref) produces correct output.
    The genuine cross-process execution of the relaxation is still being
    built: today the driver performs the Peano decomposition and then advances
    the relaxation with the verified serial/threaded loop on the coordinator.
    See `distributed_backend.md` in the repository root for the design and
    remaining work. For production runs, use shared-memory threading above.

The intended model is one `Distributed.jl` worker per rank/node (with threads
within each), a Peano–Hilbert space-filling-curve domain decomposition, ghost
halos, and collective global reductions each iteration. Running single-process
(`nworkers() == 1`) it is byte-identical to the threaded path.

### Worker launch

[`init_workers`](@ref) is the single pluggable entry point:

- `:local` → `addprocs(n)` (defaults to the CPU count),
- `:slurm` → `addprocs(SlurmManager(n))`,
- `:pbs`   → `addprocs(PBSManager(n))`,
- `:auto`  → detects the scheduler from the environment
  (`SLURM_JOB_ID`/`SLURM_NTASKS`/`SLURM_NNODES`, `PBS_JOBID`/`PBS_NODEFILE`)
  and sizes the pool from the *allocation*.

`--project` and `JULIA_DEPOT_PATH` are propagated so every (possibly remote)
node can `using WVTICs`. [`plan_workers`](@ref), [`detect_scheduler`](@ref) and
[`scheduler_pool_size`](@ref) are the side-effect-free planning helpers (usable
without a real scheduler). `ClusterManagers` is imported only by the distributed
backend; the threaded path is runtime-independent of it.

```julia
using WVTICs, Distributed
init_workers(; manager = :auto)          # or :local / :slurm / :pbs, n = …
@everywhere using WVTICs
```

### Batch scripts

`examples/run_distributed.jl` is a scheduler-agnostic driver; the
`examples/sbatch_wvtics.sh` (SLURM) and `examples/qsub_wvtics.sh` (PBS/Torque)
scripts submit it (edit the absolute paths and parameter file first). Locally:

```sh
WVTICS_NWORKERS=4 julia --project=/abs/WVTICs -t 4 \
    examples/run_distributed.jl ics.par
```

### Distributed IO

[`write_output_distributed`](@ref) writes one Gadget file per non-empty domain
slice (`name.0 … name.{k-1}`), or gathers to a single file with
`single_file = true`. Paths are taken as given — use absolute / shared-parallel
filesystem paths on HPC; no localhost co-location is assumed.

### Notes on determinism

The parallel result is *statistically* equivalent to the serial run, not
bit-identical: RNG streams and floating-point reduction order differ by design.
The verified serial WVT step is the single source of truth for the relaxation;
the distributed layer adds the decomposition, halo exchange and collective
reductions around it.
```
