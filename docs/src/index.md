```@meta
CurrentModule = WVTICs
```

# WVTICs

Documentation for [WVTICs](https://github.com/LudwigBoess/WVTICs.jl).

## Running on a cluster (distributed-memory, Phase D)

WVTICs has two parallel backends behind one interface (the serial/threaded
port is the reference implementation; the distributed backend is purely
additive):

* **Shared memory** — `Threads.@threads` / Polyester across the per-particle
  loops. Just run with `julia -t <nthreads>`; nothing else is needed.
* **Distributed memory** — `Distributed.jl` with a Peano–Hilbert
  space-filling-curve domain decomposition, `2·max(hsml)` ghost halos and
  collective global reductions. Hybrid: one Distributed worker per
  rank/node, threads within each. Single-process it is byte-identical to the
  threaded path.

### Worker launch

`init_workers(; manager=:auto, n=nothing)` is the single pluggable
entry point:

* `:local` → `addprocs(n)` (defaults to the CPU count),
* `:slurm` → `addprocs(SlurmManager(n))`,
* `:pbs`   → `addprocs(PBSManager(n))`,
* `:auto`  → detects the scheduler from the environment
  (`SLURM_JOB_ID`/`SLURM_NTASKS`/`SLURM_NNODES`,
  `PBS_JOBID`/`PBS_NODEFILE`) and sizes the pool from the *allocation*
  (not a hardcoded count). `--project` and `JULIA_DEPOT_PATH` are
  propagated so every (possibly remote) node can `using WVTICs`.

`plan_workers`, `detect_scheduler` and `scheduler_pool_size` are the
side-effect-free planning helpers (unit-tested without a real scheduler).
ClusterManagers is imported only by the distributed backend; the threaded
port is runtime-independent of it.

### Batch scripts

`examples/run_distributed.jl` is the scheduler-agnostic driver. Submit it
with the example batch scripts (edit the absolute paths / parameter file):

```sh
sbatch examples/sbatch_wvtics.sh        # SLURM
qsub   examples/qsub_wvtics.sh          # PBS / Torque
```

or run locally for testing:

```sh
WVTICS_NWORKERS=4 julia --project=/abs/WVTICs -t 4 \
    examples/run_distributed.jl ics.par
```

### IO on a shared filesystem

`write_output_distributed` writes one Gadget file per worker
(`name.0 … name.{k-1}`, `num_files = nworkers()`), or gathers to a single
file with `single_file=true`. All paths are taken as given — use
absolute / shared-parallel-filesystem paths on HPC; no localhost
co-location is assumed (message-passing only).

### Correctness

The GLOBAL converged density error and `norm_hsml` match the serial run to
a documented statistical (not bit) tolerance — like the rest of the port,
RNG/ordering differences make the equivalence statistical. The verified
serial WVT step remains the single source of truth for the relaxation
result; the distributed layer adds only the Peano decomposition, the ghost
halo exchange and the collective reductions.

```@index
```

```@autodocs
Modules = [WVTICs]
```
