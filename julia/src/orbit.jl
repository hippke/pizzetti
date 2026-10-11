# Keplerian orbits and light curves (batman conventions), generic in the number type
# so that ForwardDiff differentiates through the orbit as well.

"""
    TransitParams(; t0, per, rp, a, inc = 90, ecc = 0, w = 90, law = Quadratic(0.4, 0.25))
    TransitParams(; t0, per, rp, a, b, ecc = 0, w = 90, law)

Parameters of a transiting planet, in batman's conventions: time of inferior
conjunction `t0`, period `per`, radius ratio `rp`, semi-major axis `a` (stellar radii),
inclination `inc` (degrees) or impact parameter `b`, eccentricity `ecc`, argument of
periastron `w` (degrees), limb-darkening `law` (or batman's `u` and `limb_dark`).
"""
struct TransitParams{T<:Real,L<:LimbDarkening}
    t0::T
    per::T
    rp::T
    a::T
    inc::T
    ecc::T
    w::T
    law::L
end

function TransitParams(; t0 = 0.0, per, rp, a, inc = nothing, b = nothing, ecc = 0.0, w = 90.0,
                       law = nothing, u = nothing, limb_dark = "quadratic")
    if law === nothing
        law = u === nothing ? Quadratic(0.4, 0.25) : limb_darkening(limb_dark, u)
    end
    if inc === nothing
        b === nothing && (b = 0.0)
        # b = a cos i (1 - e²)/(1 + e sin ω) (impact parameter at conjunction)
        fac = (1 - ecc^2) / (1 + ecc * sind(w))
        inc = acosd(b / (a * fac))
    end
    T = promote_type(typeof(float(t0)), typeof(float(per)), typeof(float(rp)), typeof(float(a)),
                     typeof(float(inc)), typeof(float(ecc)), typeof(float(w)))
    TransitParams{T,typeof(law)}(T(t0), T(per), T(rp), T(a), T(inc), T(ecc), T(w), law)
end

"Impact parameter at inferior conjunction."
impact_parameter(p::TransitParams) = p.a * cosd(p.inc) * (1 - p.ecc^2) / (1 + p.ecc * sind(p.w))

"""
    eccentric_anomaly(M, e)

Solves Kepler's equation `E - e sin E = M` (Newton's method from a robust start,
to machine precision). Generic: dual numbers carry the implicit derivative.
"""
function eccentric_anomaly(M, e)
    T = promote_type(typeof(float(M)), typeof(float(e)))
    twopi = 2T(π)
    Mr = M - twopi * floor(_val(M) / _val(twopi))
    E = _val(e) < 0.8 ? Mr + e * sin(Mr) : T(π) + zero(Mr)
    tol = 4eps(_float(T))
    for _ in 1:100
        f = E - e * sin(E) - Mr
        dE = f / (1 - e * cos(E))
        E -= dE
        abs(_val(dE)) < tol * max(1, abs(_val(E))) && break
    end
    # one more step so that dual parts converge as well
    E -= (E - e * sin(E) - Mr) / (1 - e * cos(E))
    return E + (M - Mr)
end

"""
    rsky(t, params) -> z

Projected separation of the centres (stellar radii) at time(s) `t`. Samples with the
planet behind the star get `z = 100` (as in batman).
"""
function rsky(t::Real, P::TransitParams)
    T = promote_type(typeof(float(t)), eltype_params(P))
    inc = P.inc * T(π) / 180
    ω = P.w * T(π) / 180
    si = sin(inc)
    e = P.ecc
    f0 = T(π) / 2 - ω                                  # true anomaly at conjunction
    if _val(e) < 1e-5
        # true = mean anomaly; time of periastron as in batman
        tp = P.t0 - P.per * f0 / (2T(π))
        x = (t - tp) / P.per
        s = sinpi(2 * (x + ω / (2T(π))))
        ss = s * si
        ss <= 0 && return T(100)
        return P.a * sqrt(max(1 - ss * ss, zero(T)))
    end
    E0 = 2atan(sqrt((1 - e) / (1 + e)) * tan(f0 / 2))
    M0 = E0 - e * sin(E0)
    tp = P.t0 - P.per * M0 / (2T(π))
    M = 2T(π) / P.per * (t - tp)
    E = eccentric_anomaly(M, e)
    f = 2atan(sqrt((1 + e) / (1 - e)) * tan(E / 2))
    s = sin(f + ω)
    s * si <= 0 && return T(100)
    return P.a * (1 - e^2) / (1 + e * cos(f)) * sqrt(max(1 - s * s * si * si, zero(T)))
end
rsky(t::AbstractArray, P::TransitParams) = rsky.(t, Ref(P))

eltype_params(::TransitParams{T,L}) where {T,L} = promote_type(T, _lawtype(L))
_lawtype(::Type{<:LimbDarkening}) = Float64
_lawtype(::Type{<:Linear{T}}) where {T} = T
_lawtype(::Type{<:Quadratic{T}}) where {T} = T
_lawtype(::Type{<:SquareRoot{T}}) where {T} = T
_lawtype(::Type{<:Nonlinear{T}}) where {T} = T
_lawtype(::Type{<:Power2{T}}) where {T} = T
_lawtype(::Type{<:Polynomial{T}}) where {T} = T
_lawtype(::Type{<:PowerSum{T}}) where {T} = T

"""
    light_curve(t, params; exp_time = 0, supersample = 1, integration = :gauss,
                series_tol = 1e-9, limb = :auto)

Relative flux at times `t`. With `exp_time > 0` the flux is averaged over each
exposure with `supersample` points: `:gauss` (Gauss-Legendre nodes, default),
`:midpoint` (equal sub-intervals) or `:batman` (batman's equally spaced points
including the exposure edges).
"""
function light_curve(t::AbstractVector, P::TransitParams; exp_time = 0.0, supersample::Integer = 1,
                     integration::Symbol = :gauss, series_tol = SERIES_TOL, limb::Symbol = :auto)
    if exp_time > 0 && supersample > 1
        offs, wts = _exposure_nodes(supersample, exp_time, integration)
        ts = vec(t' .+ offs)                       # supersample × nt, column-major
        z = rsky(ts, P)
        f = occult(z, P.rp, P.law; series_tol = series_tol, limb = limb)
        fm = reshape(f, supersample, length(t))
        return vec(wts' * fm)
    end
    z = rsky(t, P)
    return occult(z, P.rp, P.law; series_tol = series_tol, limb = limb)
end

"""
    light_curve!(flux, t, params; kw...)

In-place [`light_curve`](@ref) (no exposure integration).
"""
function light_curve!(out::AbstractVector, t::AbstractVector, P::TransitParams; series_tol = SERIES_TOL,
                      limb::Symbol = :auto)
    z = rsky(t, P)
    occult!(out, z, P.rp, P.law; series_tol = series_tol, limb = limb)
end

function _exposure_nodes(n::Integer, dt, integration::Symbol)
    if integration == :gauss
        u, w = gauss_legendre(n)
        return (u .- 0.5) .* dt, w
    elseif integration == :midpoint
        return ((1:n) .- 0.5) ./ n .* dt .- dt / 2, fill(1.0 / n, n)
    elseif integration == :batman
        return collect(range(-dt / 2, dt / 2; length = n)), fill(1.0 / n, n)
    end
    throw(ArgumentError("integration must be :gauss, :midpoint or :batman"))
end
