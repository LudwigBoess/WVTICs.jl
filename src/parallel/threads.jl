# ---------------------------------------------------------------------------
# Shared-memory threaded backend — swappable parallel driver.
#
# A single internal primitive, `_run_chunks`, abstracts the outer chunk-parallel
# loop (`for c in 1:nc`) used by every hot kernel over three swappable backends
# with identical behaviour:
#
#   * :base        — Base.Threads.@threads
#   * :polyester   — Polyester.@batch
#   * :ohmythreads — OhMyThreads.tforeach
#
# Each backend runs `kernel(c)` exactly once per chunk index `c in 1:nc`; each
# chunk owns disjoint indices and writes only its own `scratch[c]`/`partial[c]`
# slot, and cross-chunk reductions run after the parallel region in ascending
# chunk order — so results are identical for a fixed thread count.
#
# CRITICAL — no boxed captures: callers pass `_run_chunks(nc) do c ... end`
# whose body only forwards to a concrete, type-annotated helper (function
# barrier). The closure then captures only the concrete arguments it forwards;
# the heavy, type-stable work lives in the helper.
# ---------------------------------------------------------------------------

using Polyester: @batch
import OhMyThreads

"""
    THREAD_BACKEND :: Val

The default threading backend for [`_run_chunks`](@ref). Edit this single line
to switch between `Val(:base)`, `Val(:polyester)` and `Val(:ohmythreads)`.
"""
const THREAD_BACKEND = Val(:polyester)

"""
    _run_chunks(kernel, nc) -> nothing
    _run_chunks(kernel, nc, ::Val{backend}) -> nothing

Run `kernel(c)` for every chunk index `c in 1:nc` in parallel using the
selected `backend` (defaults to [`THREAD_BACKEND`](@ref)). `kernel` MUST be a
function-barrier closure (`c -> _xxx_chunk!(c, ...)`) so no large array is
captured by reference. Returns `nothing`; all results are written into the
caller's per-chunk arrays.
"""
@inline _run_chunks(kernel::F, nc::Integer) where {F} =
    _run_chunks(kernel, nc, THREAD_BACKEND)

# :base — Base.Threads.@threads. Each task owns chunk `c` and touches only slot
# `c`, so the chunk pattern is correct under any schedule.
@inline function _run_chunks(kernel::F, nc::Integer, ::Val{:base}) where {F}
    Threads.@threads for c in 1:nc
        kernel(c)
    end
    return nothing
end

# :polyester — Polyester.@batch. Same loop, lower task-spawn overhead.
@inline function _run_chunks(kernel::F, nc::Integer, ::Val{:polyester}) where {F}
    @batch for c in 1:nc
        kernel(c)
    end
    return nothing
end

# :ohmythreads — OhMyThreads.tforeach. `chunking=false` because the indices are
# already chunked (`nc == nthreads()`, each `c` one unit of work); re-chunking
# would break the partition the `_chunk_ranges` scratch arrays are sized for.
@inline function _run_chunks(kernel::F, nc::Integer, ::Val{:ohmythreads}) where {F}
    OhMyThreads.tforeach(1:nc; chunking = false) do c
        kernel(c)
    end
    return nothing
end
