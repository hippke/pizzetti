# Interior series (paper, Section 3; Theorems 1 and 2, Corollary 1).
#
# For a term (1 - r^2)^γ of the intensity, the mean over the occulting disk is
#     D_γ(z, p) = w^γ Σ_m β_m <ε^m>,    w = 1 - z^2,  v = 1/w,
#     β_m = (-1)^m binom(γ, m),  <ε^m> = v^m Σ_{l even ≤ m} G(m, l) z^l p^(2m-l),
#     G(m, l) = binom(m, l) binom(l, l/2) 2/(2m - l + 2) ≥ 0.
# Truncated after m = M it is S_M = w^γ Σ_j a_j(p) v^j with a_j(p) = Σ_q A[j, q] p^(2q).
# All β_m with m > γ share one sign and |β_m| does not increase, which gives the
# bound (M odd, M + 1 > γ)
#     |S_M - D_γ| ≤ B_M = w^γ |β_{M+1}| <ε^(M+1)> / (1 - e),   e = (2zp + p^2)/w < 1,
# increasing in z, so that one cut z_c per light curve guarantees the tolerance.

"Truncation order of the interior series (odd): degree of the polynomial in v = 1/(1 - z²)."
const SERIES_M = 11

"Default bound on the absolute flux error of the interior series."
const SERIES_TOL = 1e-9

function _binom(n, k)
    r = one(n)
    for i in 0:k-1
        r = r * (n - i) / (i + 1)
    end
    r
end

_Gml(::Type{T}, m, l) where {T} = _binom(T(m), l) * _binom(T(l), l ÷ 2) * 2 / T(2m - l + 2)

"β_m = (-1)^m binom(γ, m)."
function series_beta(γ, m::Integer)
    b = one(γ)
    for i in 1:m
        b = b * (i - 1 - γ) / i
    end
    b
end

"""
    series_table(γ, M = SERIES_M) -> A

Matrix `A[j+1, q+1]` with `S_M = w^γ Σ_j v^j Σ_q A[j+1, q+1] p^(2q)` (Theorem 1).
Generic in the number type (Float64, BigFloat, dual numbers, intervals).
"""
function series_table(γ::T, M::Integer = SERIES_M) where {T<:Number}
    A = zeros(T, M + 1, M + 1)
    for m in 0:M
        bm = series_beta(γ, m)
        iszero(bm) && continue
        for l in 0:2:m
            c = bm * _Gml(T, m, l)
            q = m - l ÷ 2
            h = l ÷ 2
            for i in 0:h
                A[m-i+1, q+1] += c * _binom(T(h), i) * (iseven(i) ? 1 : -1)
            end
        end
    end
    A
end

"`scale * a_j(p)` for j = 0..M (Horner in p²)."
function series_coeffs(A::AbstractMatrix, p, scale)
    n = size(A, 1)
    p2 = p * p
    T = promote_type(eltype(A), typeof(p), typeof(scale))
    a = Vector{T}(undef, n)
    @inbounds for j in 1:n
        s = zero(T)
        for q in n:-1:1
            s = s * p2 + A[j, q]
        end
        a[j] = s * scale
    end
    a
end

"""
    series_bound(z, p, γ, M = SERIES_M)

Theorem 2: rigorous bound on |S_M - D_γ| (requires M odd and M + 1 > γ).
Zero for integer γ ≤ M (the series terminates). Written as
`e^(M+1) Σ_l G (λ/2)^l (1-λ)^(M+1-l)`, `λ = 2z/(2z+p)`, so that no power over-
or underflows. Returns `Inf` where the series does not converge (e ≥ 1).
"""
function series_bound(z, p, γ, M::Integer = SERIES_M)
    T = float(promote_type(typeof(z), typeof(p), typeof(γ)))
    m1 = M + 1
    bm = abs(series_beta(T(γ), m1))
    iszero(bm) && return zero(T)
    w = 1 - z * z
    e = (2z * p + p * p) / w
    e < 1 || return T(Inf)
    λ = 2z / (2z + p)
    s = zero(T)
    for l in 0:2:m1
        s += _Gml(T, m1, l) * (λ / 2)^l * (1 - λ)^(m1 - l)
    end
    w^γ * bm * e^m1 * s / (1 - e)
end

"Bound on the flux error of the truncated series: `scale Σ |c_i| B_M(γ_i)`."
function series_flux_bound(z, p, γ::AbstractVector, cabs::AbstractVector, scale, M::Integer = SERIES_M)
    t = zero(float(typeof(z)))
    for i in eachindex(γ)
        iszero(cabs[i]) || (t += cabs[i] * series_bound(z, p, γ[i], M))
    end
    scale * t
end

"""
    series_cut(p, law; series_tol = 1e-9) -> z_c

Corollary 1: the largest separation `z_c` for which the flux error of the
interior series is proven to be at most `series_tol`. The series is used for
`z ≤ z_c`. Returns `-1.0` if the series is not used.
"""
function series_cut(p, law::LimbDarkening; series_tol = SERIES_TOL, M::Integer = SERIES_M)
    α, c = power_terms(law)
    _series_cut(Float64(_val(abs(p))), Float64.(_val.(α)), Float64.(_val.(c)), Float64(_val(series_tol)), M)
end

function _series_cut(p::Float64, α::Vector{Float64}, c::Vector{Float64}, tol::Float64, M::Integer = SERIES_M)
    norm = sum(c[i] / (α[i] + 2) for i in eachindex(c))
    norm == 0 && return -1.0
    _zcut(p, α ./ 2, abs.(c), p * p / (2 * abs(norm)), tol, M)
end

# the scale p^2/(2N) needs the signed coefficients; _zcut takes it explicitly.
# The bound factorises as B_M(z; γ) = |β_{M+1}(γ)| w^γ E(z), with E independent of γ:
#   E(z) = e^(M+1) Σ_l G(M+1, l) (λ/2)^l (1 - λ)^(M+1-l) / (1 - e),
# so one E per trial z serves all exponents.
const _GM_CACHE = Dict{Int,Vector{Float64}}()

function _Gcoef(M::Integer)
    get!(_GM_CACHE, M) do
        m1 = M + 1
        [_Gml(Float64, m1, l) for l in 0:2:m1]
    end
end

function _Ebound(z, p, G::Vector{Float64}, m1::Int)
    w = 1 - z * z
    e = (2z * p + p * p) / w
    e < 1 || return Inf
    λ = 2z / (2z + p)
    h2 = (λ / 2)^2
    q = 1 - λ
    # Σ_l G_l (λ/2)^l q^(m1-l), l = 0, 2, ..., m1 (m1 even): Horner in (λ/2)²/q² from the top
    if q > 0
        r = h2 / (q * q)
        s = 0.0
        for k in length(G):-1:1
            s = muladd(s, r, G[k])
        end
        s *= q^m1
    else
        s = G[end] * h2^(m1 ÷ 2)
    end
    return e^m1 * s / (1 - e)
end

function _zcut(p::Float64, γ::Vector{Float64}, cabs::Vector{Float64}, scale::Float64, tol::Float64,
               M::Integer = SERIES_M)
    (0 < p < 1 && tol > 0 && isfinite(scale)) || return -1.0
    m1 = M + 1
    G = _Gcoef(M)
    bm = [cabs[i] * abs(series_beta(γ[i], m1)) * scale for i in eachindex(γ)]
    all(iszero, bm) && return (1.0 - p) * (1 - 1e-15)       # integer exponents only: the series is exact
    function fb(z)
        w = 1 - z * z
        t = 0.0
        for i in eachindex(γ)
            iszero(bm[i]) || (t += bm[i] * (γ[i] == 0 ? 1.0 : w^γ[i]))
        end
        t * _Ebound(z, p, G, m1)
    end
    fb(0.0) <= tol || return -1.0
    lo, hi = 0.0, 1.0 - p
    hi1 = hi * (1 - 1e-15)
    fb(hi1) <= tol && return hi1
    for _ in 1:52
        mid = (lo + hi) / 2
        if fb(mid) <= tol
            lo = mid
        else
            hi = mid
        end
        hi - lo <= 1e-15 && break
    end
    lo
end
