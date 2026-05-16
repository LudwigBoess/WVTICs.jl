#!/bin/bash
# ===========================================================================
# PBS/Torque batch script for a distributed WVTICs run (Phase D, CLAUDE.md
# §4 point 6).  Allocates nodes x ppn and launches the WVTICs distributed
# driver.  WVTICs' init_workers(; manager=:auto) detects PBS from the
# environment (PBS_JOBID / PBS_NODEFILE) and sizes the Distributed.jl worker
# pool from the PBS_NODEFILE line count — node-count agnostic (the Peano-key
# domain split keys off nworkers() only).
#
# Submit:   qsub examples/qsub_wvtics.sh
# Edit the #PBS lines, the absolute paths and the parameter file below.
# ===========================================================================
#PBS -N wvtics
#PBS -l nodes=2:ppn=4              # 2 nodes x 4 -> 8 entries in PBS_NODEFILE
#PBS -l walltime=02:00:00
#PBS -o wvtics.out
#PBS -e wvtics.err
#PBS -j oe

set -euo pipefail

# --- ABSOLUTE / shared-filesystem paths (HPC: shared parallel FS assumed) ---
WVTICS_PROJECT="/home/moon/lboess/.julia/dev/WVTICs"
JULIA_BIN="${JULIA_BIN:-julia}"
PARFILE="${PARFILE:-${WVTICS_PROJECT}/ics.par}"
WORKDIR="${WVTICS_WORKDIR:-${PBS_O_WORKDIR:-$PWD}}"      # shared-FS output dir

# Propagate the depot so every (remote) node can resolve WVTICs + deps.
export JULIA_DEPOT_PATH="${JULIA_DEPOT_PATH:-$HOME/.julia}"
# Threads per worker: derive from ppn if exported, else 1.
export JULIA_NUM_THREADS="${WVTICS_THREADS:-1}"

cd "$WORKDIR"

# init_workers(; manager=:auto) reads PBS_NODEFILE for the pool size.
"$JULIA_BIN" --project="$WVTICS_PROJECT" -t "$JULIA_NUM_THREADS" \
    "${WVTICS_PROJECT}/examples/run_distributed.jl" "$PARFILE"
