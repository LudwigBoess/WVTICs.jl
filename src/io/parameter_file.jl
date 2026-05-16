# Pure-Julia port of `io.c::Read_param_file`.
#
# The C parser:
#   - reads the file line by line;
#   - splits each line on whitespace via `sscanf(buf,"%s%s%s",...)`; a line
#     with fewer than 2 tokens is skipped;
#   - if the first token starts with '%' the line is a comment and is skipped;
#   - otherwise the first token is matched against the known tag set; the first
#     match consumes the tag (each tag is parsed at most once — `tagDone`);
#   - the second token is converted with atoi/atof/strcpy per the tag type;
#   - after the file, every tag must have been seen, else it errors and exits.
#
# Tag set: exactly the tags registered in `io.c::Read_param_file`, MINUS
# `PNG_Filename` (PNG is out of scope per CLAUDE.md). `LimitMps`,
# `LimitMps10`, `LimitMps100`, `LimitMps1000` map onto `Param.LimitMps[0..3]`.
#
# Note: C `atoi`/`atof` are lenient (parse a leading numeric prefix, default 0
# on garbage). We use `parse` after a tolerant cleanup; values in real param
# files are clean. `LimitMps -1` style integers parse fine as Float64.
#
# TOML support (Julia-port extension; no C analogue):
#   `read_param_file` dispatches on the file extension: a `.toml` (case-
#   insensitive) path is parsed via the `TOML` standard library
#   (`read_param_toml`); every other extension uses the ASCII parser above,
#   byte-for-byte unchanged. The TOML schema is a single FLAT top-level table
#   whose keys are the same tags as the ASCII format (1:1):
#
#     Npart, Maxiter, MpsFraction, StepReduction,
#     LimitMps, LimitMps10, LimitMps100, LimitMps1000,
#     MoveFractionMin, MoveFractionMax, ProbesFraction,
#     RedistributionFrequency, LastMoveStep, density_function_correction
#     (legacy alias `BiasCorrection` still accepted),
#     Problem_Flag, Problem_Subflag
#
#   REQUIRED keys (error if absent, mirroring the ASCII "missing tag" error):
#     Npart, Maxiter, Problem_Flag, Problem_Subflag.
#   DEFAULTED keys (use the `Parameters()` kwdef default — all 0.0/0 — if
#   omitted): every other tag above. The four `LimitMps*` keys map onto
#   `LimitMps[1..4]`; any omitted limit defaults to 0.0.
#   The `PNG_Filename` tag is out of scope and unsupported (an unknown key in
#   the TOML table is ignored, exactly as the ASCII parser ignores it).
#
# Type coercion: `TOML.parsefile` yields native Int/Float64/String. Integer
# fields are coerced with `Int(...)` (a TOML float like `2000.0` for an Int
# tag is accepted iff integral, matching the lenient ASCII intent); real
# fields accept Int or Float and are coerced to `Float64`. The `LimitMps`
# 4-tuple is built as `NTuple{4,Float64}`, identical to the ASCII path.
import TOML

# TOML key -> (field symbol, type). Same tag set / semantics as _PARAM_TAGS.
const _PARAM_TOML_REQUIRED =
    ("Npart", "Maxiter", "Problem_Flag", "Problem_Subflag")

# Coerce a TOML scalar to an Int the way the ASCII (atoi) path would: real
# integers pass through; a Float64 must be integral.
function _toml_to_int(key::AbstractString, v)
    if v isa Integer
        return Int(v)
    elseif v isa AbstractFloat
        isinteger(v) || error("TOML parameter '$key' = $v is not an integer.")
        return Int(v)
    else
        error("TOML parameter '$key' must be an integer, got $(typeof(v)).")
    end
end

# Coerce a TOML scalar to Float64 (Int or Float accepted), like atof.
# Special case: the (case-insensitive) string "auto" is accepted ONLY for the
# `MpsFraction` tag and maps to the `0.0` auto-calibration sentinel (see
# MPSFRACTION_ANALYSIS.md §4.3); a numeric value keeps exact legacy behaviour.
function _toml_to_real(key::AbstractString, v)
    if v isa Real
        return Float64(v)
    elseif v isa AbstractString && key == "MpsFraction" &&
           lowercase(strip(v)) == "auto"
        return 0.0                              # auto-calibrate sentinel
    else
        error("TOML parameter '$key' must be a number, got $(typeof(v)).")
    end
end

"""
    read_param_toml(filename::AbstractString) -> Parameters

Parse a TOML parameter file into a [`Parameters`](@ref) (Julia-port
extension; the C code has no TOML reader). The schema is a single flat
top-level table whose keys are the same tags as the ASCII format:

`Npart`, `Maxiter`, `MpsFraction`, `StepReduction`, `LimitMps`,
`LimitMps10`, `LimitMps100`, `LimitMps1000`, `MoveFractionMin`,
`MoveFractionMax`, `ProbesFraction`, `RedistributionFrequency`,
`LastMoveStep`, `density_function_correction` (legacy alias `BiasCorrection`
still accepted), `Problem_Flag`, `Problem_Subflag`.

`Npart`, `Maxiter`, `Problem_Flag` and `Problem_Subflag` are **required**;
every other key is optional and defaults to the `Parameters()` default
(`0`/`0.0`; the `LimitMps*` entries default to `0.0`). Unknown keys (e.g.
`PNG_Filename`) are ignored, exactly as the ASCII parser ignores unknown
tags. Errors (mirroring the ASCII parser's `ErrorException`) if the file is
missing or a required key is absent.

The returned `Parameters` has identical field semantics and types to the
value returned by the ASCII [`read_param_file`](@ref).
"""
function read_param_toml(filename::AbstractString)
    isfile(filename) || error("Parameter file $filename not found.")

    local table
    try
        table = TOML.parsefile(filename)
    catch e
        error("Failed to parse TOML parameter file '$filename': $e")
    end

    missing_required = [k for k in _PARAM_TOML_REQUIRED if !haskey(table, k)]
    if !isempty(missing_required)
        error("Value(s) for tag(s) $(join(sort(missing_required), ", ")) " *
              "missing in parameter file '$filename'.")
    end

    param = Parameters()
    limits = zeros(Float64, 4)

    for (rawkey, val) in table
        # Resolve legacy key aliases (legacy `BiasCorrection` key still
        # accepted, mapping to the renamed `density_function_correction`).
        key = _canonical_param_tag(rawkey)
        idx = findfirst(t -> t[1] == key, _PARAM_TAGS)
        idx === nothing && continue               # unknown key (e.g. PNG_Filename)
        _, field, typ = _PARAM_TAGS[idx]
        if typ === :int
            setfield!(param, field, _toml_to_int(key, val))
        else # :real
            v = _toml_to_real(key, val)
            if field === :limit1
                limits[1] = v
            elseif field === :limit2
                limits[2] = v
            elseif field === :limit3
                limits[3] = v
            elseif field === :limit4
                limits[4] = v
            else
                setfield!(param, field, v)
            end
        end
    end

    param.LimitMps = (limits[1], limits[2], limits[3], limits[4])
    return param
end

# Tag -> (field symbol, type) ; :limit1..:limit4 are special-cased into LimitMps.
const _PARAM_TAGS = Tuple{String,Symbol,Symbol}[
    ("Npart",                   :Npart,                   :int),
    ("Maxiter",                 :Maxiter,                 :int),
    ("MpsFraction",             :MpsFraction,             :real),
    ("StepReduction",           :StepReduction,           :real),
    ("LimitMps",                :limit1,                  :real),
    ("LimitMps10",              :limit2,                  :real),
    ("LimitMps100",             :limit3,                  :real),
    ("LimitMps1000",            :limit4,                  :real),
    ("MoveFractionMin",         :MoveFractionMin,         :real),
    ("MoveFractionMax",         :MoveFractionMax,         :real),
    ("ProbesFraction",          :ProbesFraction,          :real),
    ("RedistributionFrequency", :RedistributionFrequency, :int),
    ("LastMoveStep",            :LastMoveStep,            :int),
    ("density_function_correction", :density_function_correction, :real),
    ("Problem_Flag",            :Problem_Flag,            :int),
    ("Problem_Subflag",         :Problem_Subflag,         :int),
]

# Back-compatible legacy parameter-file tag aliases. The C reference parameter
# file (`/e/ocean2/users/lboess/WVTICs/ics.par`) and all existing `.par`/`.toml`
# files use the old C `BiasCorrection` tag; it is still accepted and maps onto
# the renamed `density_function_correction` field (and the new
# `density_function_correction` tag is accepted as well). Maps legacy tag ->
# canonical tag (the canonical tag must exist in `_PARAM_TAGS`).
const _PARAM_TAG_ALIASES = Dict{String,String}(
    "BiasCorrection" => "density_function_correction",
)

# Resolve a (possibly legacy) tag to its canonical `_PARAM_TAGS` name.
_canonical_param_tag(tag::AbstractString) =
    get(_PARAM_TAG_ALIASES, tag, tag)

# Mimic C atoi: take the longest leading [+-]?digits prefix, else 0.
function _c_atoi(s::AbstractString)
    m = match(r"^[+-]?\d+", strip(s))
    return m === nothing ? 0 : parse(Int, m.match)
end

# Mimic C atof: parse a leading float prefix, else 0.0.
function _c_atof(s::AbstractString)
    m = match(r"^[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?", strip(s))
    return m === nothing ? 0.0 : parse(Float64, m.match)
end

"""
    read_param_file(filename::AbstractString) -> Parameters

Read a WVT parameter file into a [`Parameters`](@ref), dispatching on the
file extension:

- a `*.toml` path (extension match is case-insensitive) is parsed via
  [`read_param_toml`](@ref) (TOML standard library);
- any other extension uses the pure-Julia port of `io.c::Read_param_file`,
  the ASCII `tag value` parser (comment character `%`), **unchanged**.

Both paths return the SAME `Parameters` type with identical field semantics
and types. Errors (mirroring the C `exit(1)`) if the file is missing or if a
required tag is absent. The `PNG_Filename` tag is intentionally not supported
(PNG is out of scope, CLAUDE.md scope decisions); it is treated as an unknown
tag and ignored on both paths.

Correctly parses `/e/ocean2/users/lboess/WVTICs/ics.par` (ASCII) and the
example `parameters.toml` (TOML).
"""
function read_param_file(filename::AbstractString)
    if endswith(lowercase(String(filename)), ".toml")
        return read_param_toml(filename)
    end
    return _read_param_ascii(filename)
end

# Pure-Julia port of `io.c::Read_param_file` (ASCII parser). Behaviourally
# byte-for-byte the original `read_param_file`; only renamed so the public
# `read_param_file` can dispatch ASCII vs TOML by extension.
function _read_param_ascii(filename::AbstractString)
    isfile(filename) || error("Parameter file $filename not found.")

    param = Parameters()
    limits = zeros(Float64, 4)
    done = Dict{String,Bool}(t[1] => false for t in _PARAM_TAGS)

    for raw in eachline(filename)
        tokens = split(raw)                       # whitespace split, drops empties
        length(tokens) < 2 && continue            # sscanf(...) < 2  -> skip
        startswith(tokens[1], '%') && continue    # buf1[0]=='%'     -> comment

        # Resolve legacy tag aliases (e.g. the C `BiasCorrection` tag maps to
        # the renamed `density_function_correction` field) before lookup so
        # existing `.par` files (incl. the C reference `ics.par`) keep working.
        tag = _canonical_param_tag(tokens[1])
        val = tokens[2]

        idx = findfirst(t -> t[1] == tag, _PARAM_TAGS)
        idx === nothing && continue               # unknown tag (e.g. PNG_Filename)
        haskey(done, tag) && done[tag] && continue # tagDone: parse once

        _, field, typ = _PARAM_TAGS[idx]
        if typ === :int
            v = _c_atoi(val)
            setfield!(param, field, Int(v))
        else # :real
            # `MpsFraction auto` (case-insensitive) ⇒ 0.0 auto-calibration
            # sentinel (MPSFRACTION_ANALYSIS.md §4.3). Any numeric value
            # parses exactly as before via the lenient C-atof port, so
            # legacy `.par` files are byte-identical.
            v = (tag == "MpsFraction" && lowercase(strip(val)) == "auto") ?
                0.0 : _c_atof(val)
            if field === :limit1
                limits[1] = v
            elseif field === :limit2
                limits[2] = v
            elseif field === :limit3
                limits[3] = v
            elseif field === :limit4
                limits[4] = v
            else
                setfield!(param, field, v)
            end
        end
        done[tag] = true
    end

    missing_tags = [t for (t, ok) in done if !ok]
    if !isempty(missing_tags)
        error("Value(s) for tag(s) $(join(sort(missing_tags), ", ")) " *
              "missing in parameter file '$filename'.")
    end

    param.LimitMps = (limits[1], limits[2], limits[3], limits[4])
    return param
end
