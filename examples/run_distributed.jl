# ===========================================================================
# WVTICs distributed driver — launched by the SLURM/PBS batch scripts in
# this directory (or locally for testing).  Phase D (CLAUDE.md §4).
#
# This script is scheduler-agnostic: `init_workers(; manager=:auto)` detects
# SLURM / PBS / local from the environment and sizes the worker pool from the
# scheduler allocation (CLAUDE.md §4 point 6).  The threaded port is NOT
# imported through any ClusterManagers path — only this distributed
# worker-launch entry point uses it.
#
# Usage (local):
#   julia --project=/abs/path/to/WVTICs examples/run_distributed.jl ics.par
#   WVTICS_NWORKERS=4 julia --project=... examples/run_distributed.jl ics.par
#
# Usage (cluster): submit examples/sbatch_wvtics.sh or examples/qsub_wvtics.sh
# (they `cd` to a shared-FS workdir and invoke this script).
# ===========================================================================

using WVTICs
# `WVTICs` exports only `make_sph_wvtics`; the distributed pipeline drives the
# stages individually, so pull the internal entry points in explicitly
# (`using M: name` resolves unexported names).
using WVTICs: init_workers, read_param_file, setup, setup_problem,
       make_positions!, make_ids!, make_velocities!, make_temperatures!,
       make_magnetic_fields!, make_post_processing!,
       regularise_sph_particles_distributed!, decompose_domain,
       write_output_distributed
using Distributed

const PARFILE = length(ARGS) >= 1 ? ARGS[1] : "ics.par"

# Optional explicit override (else sized from the scheduler allocation).
const NREQ = haskey(ENV, "WVTICS_NWORKERS") ?
             parse(Int, ENV["WVTICS_NWORKERS"]) : nothing

# 1. Launch the worker pool (scheduler-aware; :auto picks local/SLURM/PBS).
#    --project and JULIA_DEPOT_PATH are propagated so every (possibly remote)
#    node can `using WVTICs`.
added = WVTICs.init_workers(; manager = :auto, n = NREQ)
@info "WVTICs distributed: launched $(length(added)) worker(s); " *
      "nworkers()=$(nworkers())"

try
    # Make the package available on every worker (message-passing only;
    # no shared address space assumed).
    @everywhere using WVTICs
    @everywhere using WVTICs: read_param_file, setup, setup_problem,
        make_positions!, make_ids!, make_velocities!, make_temperatures!,
        make_magnetic_fields!, make_post_processing!,
        regularise_sph_particles_distributed!, decompose_domain,
        write_output_distributed

    # 2. Standard pipeline, but the relaxation goes through the DISTRIBUTED
    #    driver (Peano decomposition + 2·max(hsml) halo + global reductions;
    #    reproduces the serial result to a documented statistical tolerance;
    #    delegates to the threaded path verbatim when nworkers()==1).
    param = read_param_file(PARFILE)
    particles, problem = setup(param)
    make_positions!(particles, param, problem)
    make_ids!(particles, param)

    regularise_sph_particles_distributed!(particles, param, problem;
                                          verbose = true)

    make_velocities!(particles, param, problem)
    make_temperatures!(particles, param, problem)
    make_magnetic_fields!(particles, param, problem)
    make_post_processing!(particles, param, problem)

    # 3. IO: one Gadget file per worker on the shared filesystem
    #    (num_files = nworkers()); gather to a single file with
    #    single_file=true if preferred for small runs.
    prob = setup_problem(param)
    decomp = decompose_domain(particles.pos, problem.Boxsize,
                              max(1, nworkers()))
    files = write_output_distributed(particles, param, problem, decomp;
                                     filename = problem.Name, verbose = true)
    @info "WVTICs distributed: wrote $(length(files)) snapshot file(s)" files
finally
    # Always release the scheduler allocation's workers cleanly.
    isempty(added) || rmprocs(added)
end
