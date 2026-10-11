module PizzettiForwardDiffExt

# ForwardDiff support. The interior series is differentiated automatically (it is a
# polynomial). Beyond it, automatic differentiation of the elliptic integrals or the
# quadrature loses accuracy within ~1e-6 of the contacts (derivatives of arccos and
# K(k) diverge there). Instead, first derivatives are supplied from the exact
# hypergeometric formulas of Section 5.4 (∂P/∂z, ∂P/∂p) and the linearity in the
# coefficients; the derivative with respect to an exponent γ (e.g. the power-2 α) is
# the derivative of the Gauss-Jacobi quadrature.

using Pizzetti
using Pizzetti: Occultor, blocked_quadrature, blocked_derivatives, quad_rules, _val
using ForwardDiff
using ForwardDiff: Dual, value, partials

Pizzetti._val(d::Dual) = Pizzetti._val(value(d))
Pizzetti._float(::Type{<:Dual{T,V}}) where {T,V} = Pizzetti._float(V)
Pizzetti._isconst(d::Dual, v) = Pizzetti._isconst(value(d), v) && all(iszero, partials(d))

# ∂P/∂γ: Richardson-extrapolated central differences of the (fully converged)
# Gauss-Jacobi quadrature; error ~1e-12 relative. (Dual exponents inside the
# Gauss-Jacobi weight would leave an integrable log singularity, ~1e-6.)
function _dP_dγ(z, p, γ)
    P(g) = blocked_quadrature(z, p, [g], [1.0], quad_rules([g]))
    h = 1e-3 * max(1.0, abs(γ))
    D1 = (P(γ + h) - P(γ - h)) / 2h
    D2 = (P(γ + h / 2) - P(γ - h / 2)) / h
    return (4D2 - D1) / 3
end

# ∂P/∂z, ∂P/∂p inside the disk (Eq. dPin) with cached hypergeometric coefficients
function _interior_derivatives(d::Pizzetti.DTerm, z::Float64, p::Float64)
    zmp = z - p
    x = (1 - zmp) * (1 + zmp)
    κ = z > 0 ? min(4z * p / x, 1.0) : 0.0
    F1 = d.F1(κ)
    F2 = d.F2(κ)
    pre = 2π * p * (d.γ == 0 ? 1.0 : exp(d.γ * log(x)))
    return pre * (F2 - F1), pre * F1
end

# Interior: the value is the series (proven accuracy); the partial derivatives are the
# exact ones of Section 5.4 (the derivative of the truncated series is not bounded by
# Theorem 2 and is accurate only to ~1e-7 at the cut for series_tol = 1e-9).
function Pizzetti._series_eval(o::Occultor, z, ::Type{S}) where {S<:Dual{<:Any,Float64}}
    zD = convert(S, z)
    pD = convert(S, o.p)
    zv = value(zD)
    pv = value(pD)
    w = 1 - zv * zv
    v = 1 / w
    lw = log(w)
    num = zero(S)
    den = zero(S)
    dt = Pizzetti._dterms(o)
    for i in eachindex(o.γ)
        γD = convert(S, o.γ[i])
        cD = convert(S, o.c[i])
        γv = value(γD)
        d = dt[i]
        wg = γv == 0 ? 1.0 : exp(γv * lw)
        Ps = Pizzetti._horner_v(d.a, v)
        Pv = wg * Ps
        Pz, Pp = _interior_derivatives(d, zv, pv)
        ∂ = Pz * partials(zD) + Pp * partials(pD)
        if !all(iszero, partials(γD))
            ∂ += wg * (lw * Ps + Pizzetti._horner_v(d.ad, v)) * partials(γD)
        end
        num += cD * S(Pv, ∂)
        den += cD * π / (γD + 1)
    end
    return 1 - num / den
end

# first-order duals over Float64 only (nested duals, e.g. Hessians, use plain AD)
function Pizzetti._limb_eval(o::Occultor, z, ::Type{S}) where {S<:Dual{<:Any,Float64}}
    zD = convert(S, z)
    pD = convert(S, o.p)
    zv = value(zD)
    pv = value(pD)
    num = zero(S)
    den = zero(S)
    dt = Pizzetti._dterms(o)
    P01 = pv < 1 ? Pizzetti._P01(zv, pv) : (NaN, NaN)
    for i in eachindex(o.γ)
        γD = convert(S, o.γ[i])
        cD = convert(S, o.c[i])
        γv = value(γD)
        d = dt[i]
        # value: closed form (γ = 0, 1), contact expansion, or quadrature
        Pv = γv == 0 && pv < 1 ? P01[1] : γv == 1 && pv < 1 ? P01[2] :
             isempty(d.ce) ? NaN : d.ce[1](zv)
        isnan(Pv) && (Pv = blocked_quadrature(zv, pv, (γv,), (1.0,), d.rules))
        Pz, Pp = blocked_derivatives(zv, pv, γv)
        ∂ = Pz * partials(zD) + Pp * partials(pD)
        if !all(iszero, partials(γD))
            Pγ = NaN
            if length(d.ce) == 5
                Pp1, Pm1, Pp2, Pm2 = d.ce[2](zv), d.ce[3](zv), d.ce[4](zv), d.ce[5](zv)
                h = d.hγ
                Pγ = (4 * (Pp2 - Pm2) / h - (Pp1 - Pm1) / 2h) / 3
            end
            isnan(Pγ) && (Pγ = _dP_dγ(zv, pv, γv))
            ∂ += Pγ * partials(γD)
        end
        num += cD * S(Pv, ∂)
        den += cD * π / (γD + 1)
    end
    return 1 - num / den
end

end
