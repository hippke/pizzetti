# Occultation flux: interior series where it is proven accurate, exact or
# hypergeometric/quadrature solutions near and across the limb.

# :auto uses the contact expansions when there are at least this many limb samples per
# j-term of their set-up (set-up ~5.5 µs per exponent and j-term, quadrature ~0.65 µs per
# exponent and sample)
const CONTACT_BREAK_EVEN = 8.5

const _LENS_CACHE = Dict{Float64,LensTable{Float64}}()
const _LENS_LOCK = ReentrantLock()

"Number of j-terms the limb tables need when the limb solution starts at z_c (cost only)."
function limb_jmax(p::Float64, zc::Float64)
    zl = max(zc, 0.0) - p
    zl <= 0 && return CONTACT_JC
    xm = 1 - zl * zl
    J = ceil(Int, 1.3 * log(2.0^-60 * (1 - xm)) / log(xm)) + 8
    return min(CONTACT_JC, max(CONTACT_NS, 16 * ((J + 15) ÷ 16)))
end

# `nc` coefficients per series: the fixed-order contact expansions use the first
# CONTACT_NS + 1 (a third of the table, so a third of the set-up when the exponent
# changes from call to call, as for a fitted power-2 law); the proven pieces all of them
function lens_tables(γ::Vector{Float64}, jm::Int; nc::Int = HYP_NC)
    lock(_LENS_LOCK) do
        map(γ) do g
            t = get(_LENS_CACHE, g, nothing)
            if t === nothing || jmax(t) < jm || size(t.T, 4) < nc
                length(_LENS_CACHE) > 256 && empty!(_LENS_CACHE)
                t = lens_table(g, jm; nc = nc)
                _LENS_CACHE[g] = t
            end
            t
        end
    end
end

# Gauss functions of the exact derivatives (Section 5.4) for one exponent. Mutable so
# that it is passed by reference (each HypCache holds 240 coefficients).
mutable struct DHyp
    γ::Float64
    F1::HypCache                 # ₂F₁(-γ, ½; 1; ·)
    F2::HypCache                 # ₂F₁(-γ, 3/2; 2; ·)
    H1::HypCache                 # B(½, γ+1) ₂F₁(½, ½; γ+3/2; ·)   (limb, Eq. dPlimb)
    H2::HypCache                 # 2 B(3/2, γ+1) ₂F₁(½, 3/2; γ+5/2; ·)
end
DHyp(g::Float64) = DHyp(g, HypCache(-g, 0.5, 1.0), HypCache(-g, 1.5, 2.0),
                        HypCache(0.5, 0.5, g + 1.5, _beta(0.5, g + 1)), HypCache(0.5, 1.5, g + 2.5, 2_beta(1.5, g + 1)))

struct DTerm
    γ::Float64
    a::Vector{Float64}
    ad::Vector{Float64}
    dh::DHyp
    rules::QuadRules{Float64}
    ce::Vector{ContactExpansion} # limb: P_γ, and P at γ ± h, γ ± h/2 (empty: quadrature)
    hγ::Float64
end

"""
    Occultor(p, law; series_tol = 1e-9, limb = :auto, nlimb = 0)

Precomputed light-curve solution for radius ratio `p` and limb-darkening `law`
(set-up once per parameter set; evaluate with `o(z)` or broadcast `o.(z)`).

* `z ≤ z_c` (occultor inside the disk): the interior series, flux error proven
  to be at most `series_tol` (Theorem 2). `series_tol = 0` disables it.
* other on-star samples: the exact Mandel & Agol (2002) solution for the quadratic
  family (uniform, linear, quadratic, polynomial of degree ≤ 2); otherwise
  - `limb = :trapezoid`: the fixed-node sinh-trapezoid rule (Section 5.3; ~1e-15, no set-up),
  - `limb = :interpolate`: its Chebyshev interpolants (~1e-15, ~40 µs set-up),
  - `limb = :fast`: contact expansions (Section 5.2; ~1e-15, a posteriori estimate),
  - `limb = :quadrature`: Green's-theorem quadrature (~1e-13),
  - `limb = :auto` (default): for `p < 0.3`, `:interpolate` from `INTERP_BREAK_EVEN` = 150
    limb samples on (`nlimb`, or counted from `samples`), `:trapezoid` otherwise; for larger
    occultors (or when the band inside the disk has no valid contour), `:fast` when there are
    enough limb samples to amortise its set-up (about 8.5 per j-term of the hypergeometric
    tables), `:quadrature` otherwise.
  The trapezoid rules are used for `p < 0.3`, the contact expansions for `p < 0.5`, both for
  Float64 only.

All parts are generic in the number type: dual numbers (ForwardDiff), BigFloat
(then the limb uses quadrature with nodes of full precision).
"""
struct Occultor{T<:Real,L<:LimbDarkening,F<:AbstractFloat}
    p::T
    law::L
    γ::Vector{T}
    c::Vector{T}
    total::T
    zc::Float64
    quad::Bool
    u1::T
    u2::T
    A0::T
    A1::T
    sγ::Vector{T}
    sa::Vector{Vector{T}}
    usecontact::Bool
    contact::ContactExpansion
    usetrap::Bool
    trap::TrapLimb
    useproven::Bool
    proven::ProvenLimb
    rules::QuadRules{F}
    limb::Symbol
    dterms::Base.RefValue{Vector{DTerm}}
    nlimb::Int
end

const _NO_CONTACT = ContactExpansion(0.0, zeros(3, 2), zeros(2, 1), zeros(0, 1), zeros(0, 2, 1), zeros(0, 2, 1), Float64[], Inf, 0.0, 0.0)
const _NO_PROVEN = ProvenLimb(0.0, Float64[], zeros(0, 2), Int[], Float64[], zeros(0, 1, 3, 1), Int[], Float64[])

function Occultor(p::Real, law::LimbDarkening; series_tol = SERIES_TOL, limb::Symbol = :auto, nlimb::Integer = 0,
                  M::Integer = SERIES_M, samples = nothing)
    (isodd(M) && M >= 1) || throw(ArgumentError("the series order M must be odd"))
    limb in (:auto, :fast, :quadrature, :proven, :trapezoid, :interpolate) ||
        throw(ArgumentError("limb must be :auto, :fast, :quadrature, :proven, :trapezoid or :interpolate"))
    α0, c0 = power_terms(law)
    T = promote_type(typeof(float(p)), eltype(α0), eltype(c0))
    p = abs(T(p))
    α = T.(α0)
    c = T.(c0)
    γ = α ./ 2
    N = sum(c[i] / (α[i] + 2) for i in eachindex(c))
    total = 2T(π) * N
    zc = series_tol > 0 ? _series_cut(Float64(_val(p)), Float64.(_val.(α)), Float64.(_val.(c)), Float64(_val(series_tol)), M) : -1.0
    # Theorem 2 needs M + 1 > γ for every exponent
    any(g -> _val(g) >= M + 1, γ) && (zc = -1.0)
    qc = quadratic_coefficients(law)
    quad = qc !== nothing
    u1, u2 = quad ? (T(qc[1]), T(qc[2])) : (zero(T), zero(T))
    # interior series: F = A0 + A1 z² - Σ w^γ P(v)
    scale = p * p / (2N)
    A0 = one(T)
    A1 = zero(T)
    sγ = T[]
    sa = Vector{T}[]
    if zc >= 0
        for i in eachindex(γ)
            g = γ[i]
            if _isconst(g, 0)
                A0 -= c[i] * scale
            elseif _isconst(g, 1)
                A0 -= c[i] * scale * (1 - p * p / 2)
                A1 += c[i] * scale
            else
                a = series_coeffs(series_table(g, M), p, c[i] * scale)
                j = findfirst(x -> _isconst(x, _val(g)) && x == g, sγ)
                if j === nothing
                    push!(sγ, g)
                    push!(sa, -a)
                else
                    sa[j] .-= a
                end
            end
        end
    end
    if samples !== nothing && limb == :auto
        pv = abs(_val(p))
        nlimb = 0
        for zi in samples
            a = abs(_val(zi))
            nlimb += (a > zc) & (a < 1 + pv)
        end
    end
    usecontact = false
    contact = _NO_CONTACT
    useproven = false
    proven = _NO_PROVEN
    mode = limb
    mode == :proven && zc < 0 && (mode = :fast)      # interior series disabled: nothing to match
    # fixed-node rule / interpolants (Section 5.3): :auto for p < 0.3 unless the band inside
    # the disk has no valid contour, in which case the contact expansions are used as before
    usetrap = false
    trap = _NO_TRAP
    if !quad && T === Float64 && 0 < p < TRAP_PMAX && mode in (:auto, :trapezoid, :interpolate)
        pf = Float64(p)
        if mode != :auto || !(zc > pf && band_gamma(pf, zc) < 0)
            γf = Float64.(γ)
            interp = (mode == :interpolate || (mode == :auto && nlimb >= INTERP_BREAK_EVEN)) && interp_supported(γf)
            trap = TrapLimb(pf, γf, Float64.(c), zc, interp)
            usetrap = true
        end
    end
    jl = T === Float64 && p < 0.5 ? limb_jmax(Float64(p), zc) : CONTACT_JC
    if !quad && !usetrap && T === Float64 && p < 0.5 && (mode == :fast || mode == :proven ||
                                            (mode == :auto && nlimb >= CONTACT_BREAK_EVEN * jl))
        if mode == :proven
            tabs = lens_tables(γ, jl)
            τ = series_tol > 0 ? Float64(series_tol) : SERIES_TOL
            proven = proven_build(p, γ, c, tabs, zc, total, τ)
            useproven = true
        else
            # exponents 0 and 1 in closed form, the others by the expansions
            ib = [i for i in eachindex(γ) if !(γ[i] == 0 || γ[i] == 1)]
            tabs = lens_tables(γ[ib], jl; nc = CONTACT_NS + 1)
            ce = contact_build(p, γ[ib], c[ib], tabs, zc)
            if ce.est <= CONTACT_TOL
                c0 = sum((c[i] for i in eachindex(γ) if γ[i] == 0); init = 0.0)
                c1 = sum((c[i] for i in eachindex(γ) if γ[i] == 1); init = 0.0)
                usecontact = true
                contact = ContactExpansion(ce.p, ce.bounds, ce.REG, ce.SA, ce.SB, ce.SC, ce.m, ce.est, c0, c1)
            end
        end
    end
    rules = quad_rules(γ, _float(T))
    dterms = Ref(DTerm[])                               # filled on demand (dual numbers)
    Occultor{T,typeof(law),_float(T)}(p, law, γ, c, total, zc, quad, u1, u2, A0, A1, sγ, sa, usecontact, contact,
                                      usetrap, trap, useproven, proven, rules, limb, dterms, nlimb)
end

@inline function _wpow(w, g)
    _isconst(g, 0.5) && return sqrt(w)
    _isconst(g, 0.25) && return sqrt(sqrt(w))
    _isconst(g, 0.75) && (s = sqrt(w); return s * sqrt(s))
    _isconst(g, 1.5) && return w * sqrt(w)
    return exp(g * log(w))
end

@inline function _horner_v(a::AbstractVector, v)
    s = zero(promote_type(eltype(a), typeof(v)))
    @inbounds for i in length(a):-1:1
        s = muladd(s, v, a[i])
    end
    s
end

"Interior series at z (valid for z ≤ z_c)."
@inline function _series_flux(o::Occultor, z)
    z2 = z * z
    w = 1 - z2
    v = 1 / w
    f = o.A0 + o.A1 * z2
    @inbounds for i in eachindex(o.sγ)
        f += _wpow(w, o.sγ[i]) * _horner_v(o.sa[i], v)
    end
    f
end

"Which method `o(z)` uses at separation z: :out, :full, :series, :quadratic, :contact, :interpolate, :trapezoid or :quadrature."
function method(o::Occultor, z)
    z = abs(z)
    p = o.p
    (p <= 0 || z >= 1 + p) && return :out
    p >= 1 && z <= p - 1 && return :full
    z <= o.zc && return :series
    o.quad && return :quadratic
    o.useproven && !isnan(o.proven(Float64(z))) && return :proven
    if o.usetrap
        zf = Float64(z)
        o.trap.interp && !isnan(_interp_eval(o.trap, zf)) && return :interpolate
        isnan(_direct_eval(o.trap, zf, _trap_scratch(o.trap)...)) || return :trapezoid
    end
    if o.usecontact
        r = o.contact(Float64(z))
        isnan(r) || return :contact
    end
    return :quadrature
end

function (o::Occultor{T})(z::Real) where {T}
    S = promote_type(T, typeof(float(z)))
    z = abs(S(z))
    p = o.p
    (p <= 0 || z >= 1 + p) && return one(S)
    p >= 1 && z <= p - 1 && return zero(S)
    z <= o.zc && return _series_eval(o, z, S)
    return _limb_eval(o, z, S)
end

# Per-term data for the exact derivatives of the ForwardDiff extension, computed once per
# Occultor on first use: the series of P_γ = π p² D_γ and of ∂P_γ/∂γ (complex step) to
# order 31, the Taylor coefficients of ₂F₁(-γ, ½; 1; κ) and ₂F₁(-γ, 3/2; 2; κ), and the
# single-term quadrature rule.
const DTERM_M = 31

# the parts of a DTerm that depend on γ only (cached: in a fit with fixed limb darkening
# every light curve has the same exponents)
struct _DTermγ
    A::Matrix{ComplexF64}        # series table of P_γ at γ + i h (complex step)
    dh::DHyp
end
const _DTERMγ_CACHE = Dict{Float64,_DTermγ}()
const _DTERMγ_LOCK = ReentrantLock()

function _dterm_γ(gv::Float64)
    lock(_DTERMγ_LOCK) do
        get!(_DTERMγ_CACHE, gv) do
            length(_DTERMγ_CACHE) > 512 && empty!(_DTERMγ_CACHE)
            _DTermγ(series_table(complex(gv, 1e-30), DTERM_M), DHyp(gv))
        end
    end
end

function _dterms(o::Occultor)
    if isempty(o.dterms[])
        pv = Float64(_val(o.p))
        jl = pv < 0.5 ? limb_jmax(pv, o.zc) : CONTACT_JC
        usece = pv < 0.5 && o.nlimb >= CONTACT_BREAK_EVEN * jl
        v = DTerm[]
        for g in o.γ
            gv = Float64(_val(g))
            h = 1e-30
            dγ = _dterm_γ(gv)
            ac = series_coeffs(dγ.A, pv, π * pv^2)
            ce = ContactExpansion[]
            hγ = 1e-3 * max(1.0, abs(gv))
            if usece && !(gv == 0 || gv == 1)
                gs = _isconst(g, gv) ? (gv,) : (gv, gv + hγ, gv - hγ, gv + hγ / 2, gv - hγ / 2)
                for gg in gs
                    e = contact_build(pv, [gg], [1.0], lens_tables([gg], jl; nc = CONTACT_NS + 1), o.zc)
                    e.est <= CONTACT_TOL || (empty!(ce); break)
                    push!(ce, e)
                end
            end
            a = real.(ac)
            na = something(findlast(!iszero, a), 1)          # terminating series (integer γ)
            push!(v, DTerm(gv, a[1:na], imag.(ac) ./ h, dγ.dh, quad_rules([gv]), ce, hγ))
        end
        o.dterms[] = v
    end
    o.dterms[]
end

# interior series (the ForwardDiff extension adds exact derivatives here)
_series_eval(o::Occultor, z, ::Type{S}) where {S} = S(_series_flux(o, z))

# flux beyond the interior series (the ForwardDiff extension adds exact derivatives here)
function _limb_eval(o::Occultor, z, ::Type{S}) where {S}
    p = o.p
    if o.quad
        # the elliptic integrals lose accuracy within ~1e-12 of internal tangency (K(k) → ∞)
        abs(z - (1 - p)) > 1e-9 && return S(quadratic_flux(z, p, o.u1, o.u2))
        return S(1 - blocked_quadrature(z, p, o.γ, o.c, o.rules) / o.total)
    end
    if o.useproven && S === Float64
        r = o.proven(Float64(z))
        isnan(r) || return 1 - r / o.total
    end
    if o.usetrap && S === Float64
        r = o.trap(Float64(z))
        isnan(r) || return 1 - r / o.total
    end
    if o.usecontact && S === Float64
        r = o.contact(Float64(z))
        isnan(r) || return 1 - r / o.total
    end
    return S(1 - blocked_quadrature(z, p, o.γ, o.c, o.rules) / o.total)
end

"""
    occult(z, p, law; series_tol = 1e-9, limb = :auto)

Relative flux of a star with limb-darkening `law` occulted by an opaque disk of
radius `p` (stellar radii) at projected separations `z` (scalar or array).
See [`Occultor`](@ref) for the methods and options.
"""
function occult(z::AbstractArray, p::Real, law::LimbDarkening; series_tol = SERIES_TOL, limb::Symbol = :auto,
                M::Integer = SERIES_M)
    o = Occultor(p, law; series_tol = series_tol, limb = limb, M = M, samples = z)
    out = similar(z, promote_type(float(eltype(z)), typeof(o.p)))
    evaluate!(out, o, z)
end
occult(z::Real, p::Real, law::LimbDarkening; kw...) = Occultor(p, law; kw...)(z)

# single Float64 samples of the quadratic family: the closed form directly (no set-up)
function occult(z::Float64, p::Float64, law::Union{Uniform,Linear{Float64},Quadratic{Float64}}; kw...)
    zz = abs(z)
    pp = abs(p)
    (isempty(kw) && abs(zz - (1 - pp)) > 1e-9) || return Occultor(p, law; kw...)(z)
    u1, u2 = quadratic_coefficients(law)
    return quadratic_flux(zz, pp, u1, u2)
end

"""
    occult!(out, z, p, law; series_tol = 1e-9, limb = :auto)

In-place version of [`occult`](@ref).
"""
function occult!(out::AbstractArray, z::AbstractArray, p::Real, law::LimbDarkening; series_tol = SERIES_TOL,
                 limb::Symbol = :auto, M::Integer = SERIES_M)
    o = Occultor(p, law; series_tol = series_tol, limb = limb, M = M, samples = z)
    evaluate!(out, o, z)
end

"""
    evaluate!(out, o::Occultor, z)

`out .= o.(z)`. For Float64 the separations are scanned in blocks of `BLOCK_SIZE`. Runs of
blocks in which no sample lies on the star (all `|z| ≥ 1 + p`, flux exactly 1) are filled with 1.
In the other runs the interior series is evaluated for all samples in a branch-free loop that
the compiler vectorises (SIMD), and the remaining samples (near and across the limb) are
computed in a second pass.
"""
evaluate!(out::AbstractArray, o::Occultor, z::AbstractArray) = (out .= o.(z))

"Block length of the Float64 `evaluate!`: blocks without a sample on the star are skipped."
const BLOCK_SIZE = 64

# Is any sample of z[lo:hi] on the star? `!(|z| ≥ 1 + p)`, not `|z| < 1 + p`, so that a NaN counts as
# on the star and is handled by the scalar path, as without blocks.
function _any_on_star(z, lo::Int, hi::Int, one_p::Float64)
    n = 0
    @inbounds @simd for i in lo:hi
        n += !(abs(z[i]) >= one_p)
    end
    n > 0
end

function evaluate!(out::AbstractArray{Float64}, o::Occultor{Float64}, z::AbstractArray{Float64})
    axes(out) == axes(z) || throw(DimensionMismatch("out and z must have the same axes"))
    if IndexStyle(out, z) isa IndexLinear
        one_p = 1 + o.p
        r = eachindex(IndexLinear(), z)
        i, last_i = first(r), last(r)
        while i <= last_i
            # extend the run [i, j] of blocks that are all on the star or all off the star
            j = min(i + BLOCK_SIZE - 1, last_i)
            on = _any_on_star(z, i, j, one_p)
            while j < last_i
                hi = min(j + BLOCK_SIZE, last_i)
                _any_on_star(z, j + 1, hi, one_p) == on || break
                j = hi
            end
            if on
                _evaluate_block!(view(out, i:j), o, view(z, i:j))
            else
                fill!(view(out, i:j), 1.0)
            end
            i = j + 1
        end
    else
        _evaluate_block!(out, o, z)
    end
    out
end

# series pass for all samples, then the samples beyond z_c
function _evaluate_block!(out, o::Occultor{Float64}, z)
    zc = o.zc
    if zc >= 0
        _series_pass!(out, o, z)
    end
    one_p = 1 + o.p
    if o.usetrap
        # limb samples in one batch (scratch arrays shared by the samples)
        idx = Int[]
        @inbounds for i in eachindex(z)
            zi = abs(z[i])
            (zi > zc) & (zi < one_p) && push!(idx, i)
        end
        if !isempty(idx)
            zs = [abs(z[i]) for i in idx]
            bl = similar(zs)
            trap_blocked!(bl, o.trap, zs)
            @inbounds for q in eachindex(idx)
                out[idx[q]] = isnan(bl[q]) ? o(zs[q]) : 1 - bl[q] / o.total
            end
        end
        @inbounds for i in eachindex(z)
            zi = abs(z[i])
            if !(zi <= zc) && !((zi > zc) & (zi < one_p))
                out[i] = zi >= one_p ? 1.0 : o(zi)
            end
        end
        return out
    end
    @inbounds for i in eachindex(z)
        zi = abs(z[i])
        if !(zi <= zc)
            out[i] = zi >= one_p ? 1.0 : o(zi)
        end
    end
    out
end

# ---- vectorised interior kernels (values for |z| > z_c are overwritten by the caller) ----
function _series_pass!(out, o::Occultor{Float64}, z)
    A0, A1 = o.A0, o.A1
    nt = length(o.sγ)
    M1 = length(o.sa) > 0 ? length(o.sa[1]) : 0
    if nt == 0
        @inbounds @simd for i in eachindex(z)
            zi = z[i]
            out[i] = muladd(A1, zi * zi, A0)
        end
    elseif M1 == 12 && nt == 1 && o.sγ[1] == 0.5
        _kernel_half!(out, z, A0, A1, ntuple(j -> o.sa[1][j], Val(12)))
    elseif M1 == 12 && nt == 1
        _kernel_pow!(out, z, A0, A1, o.sγ[1], ntuple(j -> o.sa[1][j], Val(12)))
    elseif M1 == 12 && all(g -> g in (0.25, 0.5, 0.75), o.sγ)
        tup(g) = (k = findfirst(==(g), o.sγ); k === nothing ? ntuple(_ -> 0.0, Val(12)) : ntuple(j -> o.sa[k][j], Val(12)))
        _kernel_quarter!(out, z, A0, A1, tup(0.25), tup(0.5), tup(0.75))
    else
        @inbounds for i in eachindex(z)
            zi = abs(z[i])
            out[i] = _series_flux(o, min(zi, 1 - 1e-12))
        end
    end
    out
end

@inline _w(zi) = max(1.0 - zi * zi, 1e-12)

function _kernel_half!(out, z, A0, A1, a::NTuple{N,Float64}) where {N}
    @inbounds @simd for i in eachindex(z)
        zi = z[i]
        z2 = zi * zi
        w = ifelse(1.0 - z2 > 1e-12, 1.0 - z2, 1e-12)
        v = 1.0 / w
        out[i] = muladd(A1, z2, A0) + sqrt(w) * evalpoly(v, a)
    end
end

function _kernel_pow!(out, z, A0, A1, g, a::NTuple{N,Float64}) where {N}
    @inbounds @simd for i in eachindex(z)
        zi = z[i]
        z2 = zi * zi
        w = ifelse(1.0 - z2 > 1e-12, 1.0 - z2, 1e-12)
        v = 1.0 / w
        out[i] = muladd(A1, z2, A0) + _vpow(w, g) * evalpoly(v, a)
    end
end

function _kernel_quarter!(out, z, A0, A1, aq::NTuple{N,Float64}, ah::NTuple{N,Float64}, a3::NTuple{N,Float64}) where {N}
    @inbounds @simd for i in eachindex(z)
        zi = z[i]
        z2 = zi * zi
        w = ifelse(1.0 - z2 > 1e-12, 1.0 - z2, 1e-12)
        v = 1.0 / w
        s = sqrt(w)
        q = sqrt(s)
        out[i] = muladd(A1, z2, A0) + q * evalpoly(v, aq) + s * evalpoly(v, ah) + s * q * evalpoly(v, a3)
    end
end

# (flux, certified) at one sample
function _eval_certified(o::Occultor{Float64}, z::Float64)
    m = method(o, z)
    f = o(z)
    return f, m in (:out, :full, :series, :quadratic, :proven)
end
