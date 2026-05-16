# ---------------------------------------------------------------------------
# Shared-memory threaded backend — swappable parallel driver.
#
# A single internal primitive, `_run_chunks`, abstracts the *outer*
# chunk-parallel loop (`for c in 1:nc`) used by every hot kernel. The chunk
# pattern itself (`_chunk_ranges` + per-chunk scratch/accumulator arrays
# indexed by the loop variable `c`, NEVER `threadid()`) is unchanged — see
# PORT_STATUS.md "Threading backend refactor". Only the loop *construct* is
# parameterised over three backends:
#
#   * :base        — Base.Threads.@threads (the current behaviour, default)
#   * :polyester   — Polyester.@batch
#   * :ohmythreads — OhMyThreads.tforeach
#
# Behaviour/determinism is IDENTICAL across all three: each backend executes
# `kernel(c)` exactly once for every chunk index `c in 1:nc`, each chunk owns
# disjoint particle indices and writes only its own `scratch[c]`/`partial[c]`
# slot, and all cross-chunk reductions happen *after* the parallel region in
# ascending chunk order. The number of chunks is `max(1, nthreads())` at the
# call sites, exactly as before, so per-chunk results are unchanged for a
# fixed thread count regardless of backend.
#
# CRITICAL — no boxed captures: callers pass `_run_chunks(nc) do c ... end`
# whose body does nothing but forward to a concrete, type-annotated helper
# (function barrier). The closure therefore captures only the small set of
# already-concrete arguments it forwards; the heavy, type-stable work lives in
# the helper, identical for all three backends.
# ---------------------------------------------------------------------------

using Polyester: @batch
import OhMyThreads

"""
    THREAD_BACKEND :: Val

The default threading backend for [`_run_chunks`](@ref). One trivially
editable line — the orchestrator flips this to the benchmarked winner
(`Val(:base)`, `Val(:polyester)` or `Val(:ohmythreads)`). Defaults to
`Val(:base)` (current `Base.Threads` behaviour) so verification is
apples-to-apples with the pre-refactor suite.
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

# :base — current behaviour (Base.Threads.@threads). `:static` is NOT used so
# this stays a drop-in for both the plain `@threads` sites and the `:static`
# `mpart_from_integral` site (the chunk pattern is correct under either
# schedule: each task owns chunk `c` and touches only slot `c`).
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

# :ohmythreads — OhMyThreads.tforeach. `tforeach` (not `@tasks`) is chosen
# deliberately: the call site is already a function-barrier closure, so the
# function-call form composes directly with the existing `c -> helper(c, ...)`
# discipline with zero macro-hygiene/capture surprises, and we pass
# `chunking=false` because we have ALREADY chunked — `nc == nthreads()` and
# each `c` is exactly one unit of work; we do NOT want OhMyThreads to
# re-chunk our chunk indices (that would change the per-chunk partition our
# `_chunk_ranges` scratch arrays are sized for).
@inline function _run_chunks(kernel::F, nc::Integer, ::Val{:ohmythreads}) where {F}
    OhMyThreads.tforeach(1:nc; chunking = false) do c
        kernel(c)
    end
    return nothing
end
