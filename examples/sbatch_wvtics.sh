#!/bin/bash
# ===========================================================================
# SLURM batch script for a distributed WVTICs run (Phase D, CLAUDE.md §4
# point 6).  Allocates N tasks (possibly across multiple nodes) and launches
# the WVTICs distributed driver.  WVTICs' init_workers(; manager=:auto)
# detects SLURM from the environment (SLURM_JOB_ID / SLURM_NTASKS /
# SLURM_NNODES) and sizes the Distributed.jl worker pool from THIS
# allocation — node-count agnostic by construction (the Peano-key domain
# split keys off nworkers() only).
#
# Submit:   sbatch examples/sbatch_wvtics.sh
# Edit the #SBATCH lines, the absolute paths and the parameter file below.
# ===========================================================================
#SBATCH --job-name=wvtics
#SBATCH --nodes=2
#SBATCH --ntasks=8                 # -> WVTICs worker pool size (SLURM_NTASKS)
#SBATCH --cpus-per-task=4          # threads per worker (hybrid Distributed x Threads)
#SBATCH --time=02:00:00
#SBATCH --output=wvtics-%j.out
#SBATCH --error=wvtics-%j.err

set -euo pipefail

# --- ABSOLUTE / shared-filesystem paths (HPC: shared parallel FS assumed) ---
WVTICS_PROJECT="/home/moon/lboess/.julia/dev/WVTICs"
JULIA_BIN="${JULIA_BIN:-julia}"
PARFILE="${1:-${WVTICS_PROJECT}/ics.par}"
WORKDIR="${WVTICS_WORKDIR:-${SLURM_SUBMIT_DIR:-$PWD}}"   # shared-FS output dir

# Propagate the depot so every (remote) node can resolve WVTICs + deps.
export JULIA_DEPOT_PATH="${JULIA_DEPOT_PATH:-$HOME/.julia}"
# Threads per worker = cpus-per-task (the per-node shared-memory loops).
export JULIA_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

cd "$WORKDIR"

# init_workers(; manager=:auto) reads SLURM_NTASKS for the pool size.
"$JULIA_BIN" --project="$WVTICS_PROJECT" -t "$JULIA_NUM_THREADS" \
    "${WVTICS_PROJECT}/examples/run_distributed.jl" "$PARFILE"
