# Gauss hypergeometric functions of the limb solution (paper, Section 5).
#
# For a term (1 - r²)^γ, Green's theorem reduces the blocked flux near and across
# the limb to Euler integrals. With x = 1 - (z - p)², k = x/(4zp), κ = 1/k:
#   limb crossing:  P_γ = -x^(γ+1) √k / (2(γ+1)) Σ_j c_j B(½, γ+2+j) H_j(k),
#                   H_j(k) = ₂F₁(½, ½; γ+5/2+j; k),
#   interior:       P_γ = -π x^(γ+1) / (2(γ+1)) Σ_j c_j G_j(κ),
#                   G_j(κ) = ₂F₁(-(γ+1+j), ½; 1; κ),
#   c_0 = 1 + p² - z²,  c_j = (p² - z²) x^j  (j ≥ 1).
# Both families have c - a - b = γ + 3/2 + j. Their power series are tabulated
# once per exponent: Taylor series for argument ≤ ½, and the connection formula
# (DLMF 15.8.4; 15.8.10 for integer c - a - b) in y = 1 - argument beyond,
#   F = A(y) + y^m (ln y Bl(y) + Bc(y)).

"Series switch from the Taylor form to the connection form at this argument."
const HYP_XS = 0.5
"Stored coefficients per power series (Float64)."
const HYP_NC = 96

# (log|Γ(x)|, sign Γ(x)); sign 0 at the poles. Generic via reflection.
function _lgs(x::T) where {T<:Real}
    if x <= 0 && isinteger(_val(x))
        return zero(T), 0
    end
    if x > 0
        return loggamma(x), 1
    end
    s = sinpi(x)                    # Γ(x) = π / (sin(πx) Γ(1 - x))
    return log(T(π)) - log(abs(s)) - loggamma(1 - x), (s > 0 ? 1 : -1)
end

# Γ(a1)Γ(a2) / (Γ(b1)Γ(b2)) without overflow
function _gratio(a1, a2, b1, b2)
    l1, s1 = _lgs(a1)
    l2, s2 = _lgs(a2)
    l3, s3 = _lgs(b1)
    l4, s4 = _lgs(b2)
    (s3 == 0 || s4 == 0) && return zero(l1)
    return s1 * s2 * s3 * s4 * exp(l1 + l2 - l3 - l4)
end

function _taylor!(out, a, b, c)
    t = one(eltype(out))
    for i in eachindex(out)
        out[i] = t
        n = i - 1
        t *= (n + a) * (n + b) / ((n + c) * (n + 1))
    end
    out
end

# A, Bl, Bc with F = A(y) + y^m (ln y Bl(y) + Bc(y)), y = 1 - x
function _connection!(A, Bl, Bc, a, b, c)
    T = eltype(A)
    n = length(A)
    m = c - a - b
    mi = round(_val(m))
    fill!(A, zero(T)); fill!(Bl, zero(T)); fill!(Bc, zero(T))
    if abs(_val(m) - mi) > 1e-9
        CA = _gratio(c, m, c - a, c - b)
        CB = _gratio(c, -m, a, b)
        _taylor!(A, a, b, 1 - m)
        _taylor!(Bc, c - a, c - b, 1 + m)
        A .*= CA
        Bc .*= CB
        return
    end
    mm = Int(mi)
    t = _gratio(T(mi), c, a + mi, b + mi)
    for i in 0:min(mm, n)-1
        A[i+1] = t
        i + 1 < mm && (t *= (i + a) * (i + b) / ((i + 1) * (i + 1 - mi)))
    end
    l1, s1 = _lgs(c)
    l3, s3 = _lgs(a)
    l4, s4 = _lgs(b)
    (s3 == 0 || s4 == 0) && return
    pre = -(isodd(mm) ? -1 : 1) * s1 * s3 * s4 * exp(l1 - l3 - l4 - loggamma(T(mi) + 1))
    t = one(T)
    for i in 0:n-1
        ψ = -digamma(T(i + 1)) - digamma(T(i + mi + 1)) + digamma(a + i + mi) + digamma(b + i + mi)
        Bl[i+1] = pre * t
        Bc[i+1] = pre * t * ψ
        t *= (a + mi + i) * (b + mi + i) / ((i + 1) * (i + mi + 1))
    end
end

# number of coefficients needed for |argument| ≤ xmax (relative 2^-60 for Float64)
function _length(co::AbstractVector, xmax)
    big_ = zero(eltype(co))
    w = one(xmax)
    for c in co
        big_ = max(big_, abs(c) * w)
        w *= xmax
    end
    iszero(big_) && return 0
    thr = eps(_float(eltype(co))) / 256 * big_
    L = 0
    w = one(xmax)
    for i in eachindex(co)
        abs(co[i]) * w > thr && (L = i)
        w *= xmax
    end
    L
end

"""
    LensTable

Coefficient tables of the limb solution for one exponent γ: `T[fam, j+1, slot, n+1]`
(fam 1: limb crossing B(½, γ+2+j) H_j; fam 2: interior π G_j; slot 1: Taylor in the
argument; slots 2-4: A, Bl, Bc of the connection form) and the used lengths `L`.
"""
struct LensTable{T}
    γ::T
    T::Array{T,4}
    L::Array{Int,3}
end

jmax(t::LensTable) = size(t.T, 2) - 1

function lens_table(γ::T, jm::Integer; nc::Integer = HYP_NC) where {T<:Real}
    Tb = zeros(T, 2, jm + 1, 4, nc)
    L = zeros(Int, 2, jm + 1, 4)
    A = zeros(T, nc); Bl = zeros(T, nc); Bc = zeros(T, nc); Ty = zeros(T, nc)
    half = T(1) / 2
    for j in 0:jm, fam in 1:2
        if fam == 1
            a, b, c = half, half, γ + T(5) / 2 + j
            s = exp(loggamma(half) + loggamma(γ + 2 + j) - loggamma(γ + T(5) / 2 + j))
        else
            a, b, c = -(γ + 1 + j), half, one(T)
            s = T(π)
        end
        _taylor!(Ty, a, b, c)
        _connection!(A, Bl, Bc, a, b, c)
        for (sl, v) in enumerate((Ty, A, Bl, Bc))
            @views Tb[fam, j+1, sl, :] .= v .* s
            L[fam, j+1, sl] = _length(view(Tb, fam, j + 1, sl, :), T(HYP_XS))
        end
    end
    LensTable{T}(γ, Tb, L)
end

@inline function _horner(co, n, x)
    s = zero(x)
    @inbounds for i in n:-1:1
        s = s * x + co[i]
    end
    s
end

"""
    hyp2f1_unit(a, b, c, x, y = 1 - x)

Gauss hypergeometric function ₂F₁(a, b; c; x) for 0 ≤ x ≤ 1 and c - a - b > 0
(including integer c - a - b, the logarithmic case), by its Taylor series for
x ≤ ½ and the connection formula about x = 1 beyond. Generic in the number type;
the series are summed until the terms are below the working precision. `y = 1 - x` can
be passed when it is known more accurately than by the subtraction.
"""
function hyp2f1_unit(a, b, c, x, y = 1 - x)
    T = float(promote_type(typeof(a), typeof(b), typeof(c), typeof(x)))
    a, b, c, x, y = T(a), T(b), T(c), T(x), T(y)
    epsT = eps(_float(T))
    if x <= 0.5
        s = zero(T)
        t = one(T)
        for n in 0:100000
            s += t
            t *= (n + a) * (n + b) / ((n + c) * (n + 1)) * x
            (abs(_val(t)) <= epsT * abs(_val(s)) / 4 && n > 2) && break
            iszero(t) && break
        end
        return s
    end
    m = c - a - b
    mi = round(_val(m))
    # number of terms: geometric convergence with ratio y ≤ ½ (plus slack for the prefactors)
    N = ceil(Int, -log2(epsT) / max(-log2(_val(y)), 1e-3)) + 30
    N = min(N, 100000)
    A = zeros(T, N); Bl = zeros(T, N); Bc = zeros(T, N)
    _connection!(A, Bl, Bc, a, b, c)
    r = _horner(A, N, y)
    if y > 0
        ym = exp(m * log(y))
        r += ym * (log(y) * _horner(Bl, N, y) + _horner(Bc, N, y))
    end
    return r
end

"""
    HypCache(a, b, c, scale = 1)

`scale` × ₂F₁(a, b; c; x) on 0 ≤ x ≤ 1 (c - a - b > 0) from precomputed coefficients: the Taylor
series for x ≤ ½ and the connection form A(y) + y^m (ln y Bl(y) + Bc(y)), y = 1 - x, beyond;
60 terms give 2^-60 on both halves. Float64.
"""
struct HypCache
    T::Vector{Float64}           # vectors, not tuples: the struct stays small and is passed
    A::Vector{Float64}           # by reference (copies of large tuples cost more than the
    Bl::Vector{Float64}          # polynomial evaluations)
    Bc::Vector{Float64}
    m::Float64
    nT::Int          # Taylor coefficients up to the last non-zero one (terminating series)
end

function HypCache(a::Float64, b::Float64, c::Float64, scale::Float64 = 1.0)
    n = 60
    Ty = zeros(n); A = zeros(n); Bl = zeros(n); Bc = zeros(n)
    _taylor!(Ty, a, b, c)
    _connection!(A, Bl, Bc, a, b, c)
    Ty .*= scale; A .*= scale; Bl .*= scale; Bc .*= scale
    nT = something(findlast(!iszero, Ty), 1)
    HypCache(Ty, A, Bl, Bc, c - a - b, nT)
end

# Terms needed for |u| ≤ x so that u^n < 2^-56 (the coefficients of the Gauss functions
# used here are at most of order one and do not increase): a few comparisons instead of
# a logarithm; 60 terms at x = ½.
@inline function _nterms(x::Float64)
    x <= 0.0078125 && return 9            # 2^-7
    x <= 0.03125 && return 13             # 2^-5
    x <= 0.0625 && return 16              # 2^-4
    x <= 0.125 && return 21               # 2^-3
    x <= 0.25 && return 31                # 2^-2
    x <= 0.354 && return 40               # 2^-1.5
    return 60
end

# Horner over the first n entries of a tuple
@inline function _evalpoly_n(x::Float64, c::Vector{Float64}, n::Int)
    s = @inbounds c[n]
    @inbounds for i in n-1:-1:1
        s = muladd(s, x, c[i])
    end
    s
end

# y = 1 - x, given separately when it is known more accurately than by subtraction
# (near the contacts the argument is 1 - O(δ) and ₂F₁ has a y^m singular part)
@inline function (h::HypCache)(x::Float64, y::Float64 = 1.0 - x)
    x <= 0.5 && return _evalpoly_n(x, h.T, min(h.nT, _nterms(x)))
    n = _nterms(y)
    r = _evalpoly_n(y, h.A, n)
    if y > 0
        ly = log(y)
        r += exp(h.m * ly) * (ly * _evalpoly_n(y, h.Bl, n) + _evalpoly_n(y, h.Bc, n))
    end
    r
end

# two Horner chains interleaved (independent, so they overlap in the pipeline)
@inline function _evalpoly2_n(x::Float64, c::Vector{Float64}, d::Vector{Float64}, n::Int)
    s = @inbounds c[n]
    t = @inbounds d[n]
    @inbounds for i in n-1:-1:1
        s = muladd(s, x, c[i])
        t = muladd(t, x, d[i])
    end
    s, t
end

"""
    hyp_pair(h, g, x, y = 1 - x)

`(h(x), g(x))` for two `HypCache`s with the same `c - a - b` (the pairs of the
derivative formulas): one logarithm and one power, interleaved polynomial evaluation.
"""
@inline function hyp_pair(h::HypCache, g::HypCache, x::Float64, y::Float64 = 1.0 - x)
    if x <= 0.5
        n = _nterms(x)
        return _evalpoly2_n(x, h.T, g.T, min(max(h.nT, g.nT), n))
    end
    n = _nterms(y)
    rh, rg = _evalpoly2_n(y, h.A, g.A, n)
    if y > 0
        ly = log(y)
        ym = exp(h.m * ly)
        lh, lg = _evalpoly2_n(y, h.Bl, g.Bl, n)
        ch, cg = _evalpoly2_n(y, h.Bc, g.Bc, n)
        rh += ym * (ly * lh + ch)
        rg += exp(g.m * ly) * (ly * lg + cg)
    end
    rh, rg
end
