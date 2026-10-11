module PizzettiIntervalArithmeticExt

# Rigorous enclosures of the flux on the interior series (Theorem 2 plus rounding):
# the coefficient table is computed in exact rational arithmetic, the series and its
# truncation bound in interval arithmetic with outward rounding. The resulting
# interval contains the exact flux of the occultation (the integral over the
# occulted area), not just the value of the truncated series.

using Pizzetti
using Pizzetti: SERIES_M, power_terms, LimbDarkening
using IntervalArithmetic

_I(x) = interval(Float64, x)

function _rational_table(γ::Rational{BigInt}, M::Int)
    A = zeros(Rational{BigInt}, M + 1, M + 1)
    binom(n, k) = (r = one(Rational{BigInt}); for i in 0:k-1; r = r * (n - i) / (i + 1); end; r)
    G(m, l) = binom(Rational{BigInt}(m), l) * binom(Rational{BigInt}(l), l ÷ 2) * 2 // (2m - l + 2)
    for m in 0:M
        bm = one(Rational{BigInt})
        for i in 1:m
            bm = bm * (i - 1 - γ) / i
        end
        iszero(bm) && continue
        for l in 0:2:m
            c = bm * G(m, l)
            q = m - l ÷ 2
            h = l ÷ 2
            for i in 0:h
                A[m-i+1, q+1] += c * binom(Rational{BigInt}(h), i) * (iseven(i) ? 1 : -1)
            end
        end
    end
    A
end

function _beta_rat(γ::Rational{BigInt}, m::Int)
    b = one(Rational{BigInt})
    for i in 1:m
        b = b * (i - 1 - γ) / i
    end
    b
end

# Theorem 2 bound in interval arithmetic (sup is the rigorous bound)
function _bound_interval(zI, pI, γ::Rational{BigInt}, M::Int)
    m1 = M + 1
    bm = abs(_beta_rat(γ, m1))
    iszero(bm) && return _I(0)
    w = 1 - zI * zI
    e = (2 * zI * pI + pI * pI) / w
    sup(e) < 1 || return interval(0.0, Inf)
    λ = 2 * zI / (2 * zI + pI)
    s = _I(0)
    for l in 0:2:m1
        Gml = binomial(big(m1), l) * binomial(big(l), l ÷ 2) * 2 // (2m1 - l + 2)
        s += _I(Gml) * (λ / 2)^l * (1 - λ)^(m1 - l)
    end
    wg = iszero(γ) ? _I(1) : exp(_I(γ) * log(w))
    return wg * _I(bm) * e^m1 * s / (1 - e)
end

"""
    occult_interval(z, p, law; M = SERIES_M) -> Interval

Rigorous enclosure of the exact flux for an occultor inside the stellar disk
(requires the series to converge, `(2zp + p²)/(1 - z²) < 1`): the interior series
evaluated in interval arithmetic with exact rational coefficients, widened by the
truncation bound of Theorem 2. Throws a `DomainError` where the series does not
converge (near and across the limb).
"""
function Pizzetti.occult_interval(z::Real, p::Real, law::LimbDarkening; M::Int = SERIES_M)
    zI = _I(abs(z))
    pI = _I(abs(p))
    α, c = power_terms(law)
    cI = [_I(x) for x in c]
    γs = [Rational{BigInt}(Float64(a)) / 2 for a in α]
    N = sum(cI[i] / (_I(Rational{BigInt}(Float64(α[i]))) + 2) for i in eachindex(cI))
    scale = pI * pI / (2 * N)
    w = 1 - zI * zI
    v = 1 / w
    p2 = pI * pI
    S = _I(0)
    B = _I(0)
    for i in eachindex(γs)
        γ = γs[i]
        A = _rational_table(γ, M)
        P = _I(0)
        for j in M+1:-1:1
            a = _I(0)
            for q in M+1:-1:1
                a = a * p2 + _I(A[j, q])
            end
            P = P * v + a
        end
        wg = iszero(γ) ? _I(1) : exp(_I(γ) * log(w))
        S += cI[i] * wg * P
        b = _bound_interval(zI, pI, γ, M)
        isfinite(sup(b)) || throw(DomainError(z, "the interior series does not converge here"))
        B += abs(cI[i]) * b
    end
    F = 1 - scale * S
    err = abs(scale) * B
    return interval(inf(F - err), sup(F + err))
end

end
