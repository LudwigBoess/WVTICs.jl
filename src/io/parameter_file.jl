# Reads WVT parameter files into a `Parameters` (ports io.c::Read_param_file).
#
# `read_param_file` dispatches on the extension: a `.toml` path is parsed via
# the TOML standard library, any other extension uses the ASCII `tag value`
# parser (comment character `%`, one value per tag). Both formats share the
# same flat tag set and return an identical `Parameters`.
#
# Tags:
#     Npart, Maxiter, MpsFraction, StepReduction,
#     LimitMps, LimitMps10, LimitMps100, LimitMps1000,
#     MoveFractionMin, MoveFractionMax, ProbesFraction,
#     RedistributionFrequency, LastMoveStep, density_function_correction
#     (legacy alias `BiasCorrection` still accepted),
#     Problem_Flag, Problem_Subflag, DesNumNgb
#
# `LimitMps`, `LimitMps10`, `LimitMps100`, `LimitMps1000` map onto
# `LimitMps[1..4]`. `DesNumNgb` sets the target SPH neighbour count (`DESNNGB`);
# it is optional and defaults to 0 (use the kernel's built-in default).
#
# Required tags (error if absent): Npart, Maxiter, Problem_Flag,
# Problem_Subflag. Every other tag defaults to the `Parameters()` default
# (0/0.0; each omitted `LimitMps*` defaults to 0.0). `PNG_Filename` is not
# supported: it is treated as an unknown tag and ignored on both paths.
#
# ASCII values are parsed leniently (leading numeric prefix, else 0), matching
# C atoi/atof. TOML scalars are coerced: integer tags accept an Int or an
# integral Float; real tags accept an Int or a Float coerced to Float64.
import TOML

# TOML key -> (field symbol, type). Same tag set / semantics as _PARAM_TAGS.
const _PARAM_TOML_REQUIRED =
    ("Npart", "Maxiter", "Problem_Flag", "Problem_Subflag")

# Coerce a TOML scalar to Int: integers pass through, a Float must be integral.
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

# Coerce a TOML scalar to Float64 (Int or Float accepted). The case-insensitive
# string "auto" is accepted only for the `MpsFraction` tag and maps to the 0.0
# auto-calibration sentinel; any numeric value is used as-is.
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

Parse a TOML parameter file into a [`Parameters`](@ref). The schema is a single
flat top-level table whose keys are the same tags as the ASCII format:

`Npart`, `Maxiter`, `MpsFraction`, `StepReduction`, `LimitMps`,
`LimitMps10`, `LimitMps100`, `LimitMps1000`, `MoveFractionMin`,
`MoveFractionMax`, `ProbesFraction`, `RedistributionFrequency`,
`LastMoveStep`, `density_function_correction` (legacy alias `BiasCorrection`
still accepted), `Problem_Flag`, `Problem_Subflag`, `DesNumNgb` (target SPH
neighbour count `DESNNGB`, optional, defaults to 0).

`Npart`, `Maxiter`, `Problem_Flag` and `Problem_Subflag` are **required**;
every other key is optional and defaults to the `Parameters()` default
(`0`/`0.0`; the `LimitMps*` entries default to `0.0`). Unknown keys (e.g.
`PNG_Filename`) are ignored. Errors if the file is missing or a required key is
absent.

The returned `Parameters` has identical field semantics and types to the value
returned by the ASCII [`read_param_file`](@ref).
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
        # Map the legacy `BiasCorrection` key onto `density_function_correction`.
        key = _canonical_param_tag(rawkey)
        entry = get(_PARAM_TAG_MAP, key, nothing)
        entry === nothing && continue             # unknown key (e.g. PNG_Filename)
        field, typ = entry
        v = typ === :int ? _toml_to_int(key, val) : _toml_to_real(key, val)
        _store_param!(param, limits, field, v)
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
    ("DesNumNgb",               :DesNumNgb,               :int),
]

# tag -> (field, type) for O(1) lookup by both the ASCII and TOML parsers.
const _PARAM_TAG_MAP = Dict{String,Tuple{Symbol,Symbol}}(
    t[1] => (t[2], t[3]) for t in _PARAM_TAGS)

# Route a coerced value into the Parameters struct, or into the LimitMps
# accumulator when the field is one of the four :limitN pseudo-fields.
function _store_param!(param::Parameters, limits::Vector{Float64},
                       field::Symbol, value)
    if field === :limit1
        limits[1] = value
    elseif field === :limit2
        limits[2] = value
    elseif field === :limit3
        limits[3] = value
    elseif field === :limit4
        limits[4] = value
    else
        setfield!(param, field, value)
    end
    return nothing
end

# Optional parameter tags: parsed if present but not required. `DesNumNgb` sets
# the target SPH neighbour count (`DESNNGB`) and overrides the kernel's default
# when > 0; omitting it keeps the kernel default.
const _PARAM_OPTIONAL_TAGS = ("DesNumNgb",)

# Legacy parameter-file tag aliases: maps an old tag to its canonical
# `_PARAM_TAGS` name. `BiasCorrection` is still accepted and maps onto the
# `density_function_correction` field.
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

Read a WVT parameter file into a [`Parameters`](@ref), dispatching on the file
extension:

- a `*.toml` path (extension match is case-insensitive) is parsed via
  [`read_param_toml`](@ref) (TOML standard library);
- any other extension uses the ASCII `tag value` parser (comment character
  `%`).

Both paths return the SAME `Parameters` type with identical field semantics and
types. Errors if the file is missing or if a required tag is absent. The
`PNG_Filename` tag is not supported; it is treated as an unknown tag and ignored
on both paths.
"""
function read_param_file(filename::AbstractString)
    if endswith(lowercase(String(filename)), ".toml")
        return read_param_toml(filename)
    end
    return _read_param_ascii(filename)
end

# ASCII `tag value` parser (ports io.c::Read_param_file). Dispatched to by the
# public `read_param_file` for any non-`.toml` extension.
function _read_param_ascii(filename::AbstractString)
    isfile(filename) || error("Parameter file $filename not found.")

    param = Parameters()
    limits = zeros(Float64, 4)
    # Optional tags (`_PARAM_OPTIONAL_TAGS`, e.g. `DesNumNgb`) are excluded from
    # the required set, so files that omit them still parse.
    done = Dict{String,Bool}(t[1] => false for t in _PARAM_TAGS
                             if !(t[1] in _PARAM_OPTIONAL_TAGS))

    for raw in eachline(filename)
        tokens = split(raw)                       # whitespace split, drops empties
        length(tokens) < 2 && continue            # < 2 tokens  -> skip
        startswith(tokens[1], '%') && continue    # leading '%' -> comment

        # Resolve legacy tag aliases (`BiasCorrection` ->
        # `density_function_correction`) before lookup.
        tag = _canonical_param_tag(tokens[1])
        val = tokens[2]

        entry = get(_PARAM_TAG_MAP, tag, nothing)
        entry === nothing && continue             # unknown tag (e.g. PNG_Filename)
        haskey(done, tag) && done[tag] && continue # parse each tag at most once

        field, typ = entry
        if typ === :int
            _store_param!(param, limits, field, _c_atoi(val))
        else # :real — `MpsFraction auto` (case-insensitive) is the 0.0
            # auto-calibration sentinel; any numeric value parses via _c_atof.
            v = (tag == "MpsFraction" && lowercase(strip(val)) == "auto") ?
                0.0 : _c_atof(val)
            _store_param!(param, limits, field, v)
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
