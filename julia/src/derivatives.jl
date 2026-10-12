# Derivatives of the flux (paper, Section 5.4).
#
# By Reynolds' transport theorem the derivatives of the blocked integral
# P_γ(z, p) = ∫ (1 - r²)^γ dA over the occulted region are line integrals of the
# intensity over the part of the occultor's rim that lies on the star, which are
# again Gauss hypergeometric functions. With x = 1 - (z - p)²:
#
# occultor inside the disk (z + p ≤ 1), κ = 4zp/x:
#   ∂P/∂p = 2πp x^γ ₂F₁(-γ, ½; 1; κ)
#   ∂P/∂z = 2πp x^γ [₂F₁(-γ, 3/2; 2; κ) - ₂F₁(-γ, ½; 1; κ)]
# occultor on the limb (|1 - p| < z < 1 + p), k = x/(4zp):
#   ∂P/∂p = 2p x^γ √k B(½, γ+1) ₂F₁(½, ½; γ+3/2; k)
#   ∂P/∂z = 2p x^γ √k [2k B(3/2, γ+1) ₂F₁(½, 3/2; γ+5/2; k) - B(½, γ+1) ₂F₁(½, ½; γ+3/2; k)]
#
# For I = Σ c_i μ^α_i (γ_i = α_i/2) and F = 1 - Σ c_i P_γi / T, T = Σ c_i π/(γ_i + 1):
#   ∂F/∂z = -Σ c_i ∂P_γi/∂z / T,   ∂F/∂p likewise,
#   ∂F/∂c_i = -[P_γi - (1 - F) π/(γ_i + 1)] / T.

_beta(a, b) = exp(loggamma(a) + loggamma(b) - loggamma(a + b))

# ₂F₁(-γ, ½; 1; κ) and ₂F₁(-γ, 3/2; 2; κ): for Float64 from cached coefficients
const _F12_CACHE = Dict{Float64,NTuple{2,HypCache}}()
const _F12_LOCK = ReentrantLock()

function _interior_F12(γ::Float64, κ::Float64, yκ::Float64)
    h1, h2 = lock(_F12_LOCK) do
        length(_F12_CACHE) > 1024 && empty!(_F12_CACHE)
        get!(() -> (HypCache(-γ, 0.5, 1.0), HypCache(-γ, 1.5, 2.0)), _F12_CACHE, γ)
    end
    return h1(κ, yκ), h2(κ, yκ)
end
function _interior_F12(γ, κ, yκ)
    T = typeof(κ)
    half = one(T) / 2
    return hyp2f1_unit(-γ, half, one(T), κ, yκ), hyp2f1_unit(-γ, 3half, 2one(T), κ, yκ)
end

"""
    blocked_flux(z, p, γ; series_tol = 1e-9, limb = :auto)

The blocked integral `P_γ(z, p) = ∫ (1 - r²)^γ dA` over the part of the unit stellar
disk covered by an occultor of radius `p` at separation `z` (not normalised;
`π/(γ+1)` for the whole star).
"""
function blocked_flux(z, p, γ; series_tol = SERIES_TOL, limb::Symbol = :auto)
    o = Occultor(p, PowerSum([2 * float(γ)], [one(float(γ))]); series_tol = series_tol, limb = limb)
    (1 - o(z)) * oftype(float(γ), π) / (γ + 1)
end

"""
    blocked_derivatives(z, p, γ) -> (∂P/∂z, ∂P/∂p)

Exact derivatives of the blocked integral (Section 5.4), from Gauss hypergeometric
functions (no quadrature, no differentiation of truncated series).
"""
function blocked_derivatives(z, p, γ)
    T = float(promote_type(typeof(z), typeof(p), typeof(γ)))
    z = abs(T(z)); p = abs(T(p)); γ = T(γ)
    zero2 = (zero(T), zero(T))
    (p <= 0 || z >= 1 + p) && return zero2
    p >= 1 && z <= p - 1 && return zero2
    δ1 = _one_minus_sum(z, -p)                      # 1 - z + p, accurate at first contact
    δ2 = _one_minus_sum(z, p)                       # 1 - z - p, accurate at second contact
    x = δ1 * _one_minus_sum(p, -z)                  # 1 - (z - p)²
    xg = x^γ
    half = one(T) / 2
    if δ2 >= 0
        κ = z > 0 ? 4z * p / x : zero(T)
        κ = min(κ, one(T))
        yκ = max(δ2 * (1 + z + p) / x, zero(T))     # 1 - κ
        F1, F2 = _interior_F12(γ, κ, yκ)
        pre = 2T(π) * p * xg
        return (pre * (F2 - F1), pre * F1)
    end
    k = min(x / (4z * p), one(T))
    yk = max(-δ2 * (z + p + 1) / (4z * p), zero(T))             # 1 - k
    sk = sqrt(k)
    B1 = _beta(half, γ + 1)
    H1 = hyp2f1_unit(half, half, γ + 3half, k, yk)
    H2 = hyp2f1_unit(half, 3half, γ + 5half, k, yk)
    pre = 2p * xg * sk
    return (pre * (2k * _beta(3half, γ + 1) * H2 - B1 * H1), pre * B1 * H1)
end

# the same from cached Gauss functions (Float64; dual numbers and the quadratic family)
function blocked_derivatives(d::DTerm, z::Float64, p::Float64)
    if d.γ == 0
        r = _bderiv01(z, p)
        return (r[1], r[2])
    elseif d.γ == 1
        r = _bderiv01(z, p)
        return (r[3], r[4])
    end
    return _bderiv(d.dh, z, p)
end

# 1 - (a + b) to full relative precision also when a + b is close to 1 (TwoSum: the
# rounding error of a + b is recovered exactly, and 1 - s is exact for ½ ≤ s ≤ 2)
@inline function _one_minus_sum(a, b)
    s = a + b
    bb = s - a
    e = (a - (s - bb)) + (b - bb)
    return (1 - s) - e
end

function _bderiv(dh::DHyp, z::Float64, p::Float64)
    γ = dh.γ
    z = abs(z); p = abs(p)
    (p <= 0 || z >= 1 + p) && return (0.0, 0.0)
    p >= 1 && z <= p - 1 && return (0.0, 0.0)
    δ1 = _one_minus_sum(z, -p)                      # 1 - z + p (first contact: 0)
    δ2 = _one_minus_sum(z, p)                       # 1 - z - p (second contact: 0)
    x = δ1 * _one_minus_sum(p, -z)                  # 1 - (z - p)²
    xg = γ == 0 ? 1.0 : γ == 1 ? x : γ == 0.5 ? sqrt(x) : exp(γ * log(x))
    if δ2 >= 0
        κ = z > 0 ? min(4z * p / x, 1.0) : 0.0
        yκ = max(δ2 * (1 + z + p) / x, 0.0)
        F1, F2 = hyp_pair(dh.F1, dh.F2, κ, yκ)
        pre = 2π * p * xg
        return (pre * (F2 - F1), pre * F1)
    end
    zp4 = 4z * p
    k = min(x / zp4, 1.0)
    yk = max(-δ2 * (z + p + 1) / zp4, 0.0)
    H1, H2 = hyp_pair(dh.H1, dh.H2, k, yk)
    pre = 2p * xg * sqrt(k)
    return (pre * (k * H2 - H1), pre * H1)
end

# Exponents 0 and 1 in elementary form. The rim of the occultor, r² = z² + p² + 2zp cos φ,
# lies on the star for cos φ ≤ c₀ = (1 - z² - p²)/(2zp), i.e. φ₀ ≤ φ ≤ 2π - φ₀, and with
# 1 - r² = A₀ - B cos φ (A₀ = 1 - z² - p², B = 2zp), κ₀ = π - φ₀:
#   ∂P₀/∂p = 2p κ₀,                 ∂P₀/∂z = -2p sin φ₀,
#   ∂P₁/∂p = 2p (κ₀ A₀ + B sin φ₀),  ∂P₁/∂z = -p [2 A₀ sin φ₀ + B (κ₀ - sin φ₀ cos φ₀)].
function _bderiv01(z::Float64, p::Float64)
    z = abs(z); p = abs(p)
    (p <= 0 || z >= 1 + p) && return (0.0, 0.0, 0.0, 0.0)
    p >= 1 && z <= p - 1 && return (0.0, 0.0, 0.0, 0.0)
    A0 = (1 - z * z) - p * p
    B = 2z * p
    δ2 = _one_minus_sum(z, p)                       # 1 - z - p
    if δ2 >= 0 || z == 0                            # whole rim on the star
        sφ, cφ, κ0 = 0.0, 1.0, Float64(π)
    else
        δ1 = _one_minus_sum(z, -p)                  # 1 - z + p
        om = -δ2 * (1 + z + p) / B                  # 1 - c₀, accurate at second contact
        op = δ1 * _one_minus_sum(p, -z) / B         # 1 + c₀, accurate at both tangencies
        c0 = om < op ? 1 - om : op - 1
        sφ = sqrt(max(om * op, 0.0))
        cφ = c0
        κ0 = _acos_acc(-c0, op, om)                 # π - φ₀
    end
    return (-2p * sφ, 2p * κ0, -p * (2A0 * sφ + B * (κ0 - sφ * cφ)), 2p * (κ0 * A0 + B * sφ))
end

# Gauss functions of the derivatives for γ = 0, ½, 1 (the quadratic family), built once
const _QFAM = Ref{Any}(nothing)
function _qfam()
    q = _QFAM[]
    if q === nothing
        q = (DHyp(0.0), DHyp(0.5), DHyp(1.0))
        _QFAM[] = q
    end
    return q::NTuple{3,DHyp}
end

# P_0, P_½, P_1 in closed form: lens area, Mandel & Agol's λ-term, elementary
function _P_quadratic(z::Float64, p::Float64)
    (p <= 0 || z >= 1 + p) && return (0.0, 0.0, 0.0)
    p >= 1 && z <= p - 1 && return (π, 2π / 3, π / 2)
    P0, P1 = _P01(z, p)
    Ph = (1 - quadratic_flux(z, p, 1.0, 0.0)) * (2π / 3)       # I = μ: c = (0, 1, 0)
    return (P0, Ph, P1)
end

"""
    flux_gradient_quadratic(z, p, u1, u2) -> (F, ∂F/∂z, ∂F/∂p, ∂F/∂u1, ∂F/∂u2)

Quadratic law: flux and its exact derivatives, from closed forms (P_0, P_½, P_1 and
the cached Gauss functions of Section 5.4), about 150 ns per call. The flux is the
exact closed form (not the interior series).
"""
function flux_gradient_quadratic(z::Float64, p::Float64, u1::Float64, u2::Float64)
    s = z < 0 ? -1.0 : 1.0
    z = abs(z); p = abs(p)
    c = (1 - u1 - u2, u1 + 2u2, -u2)
    γ = (0.0, 0.5, 1.0)
    tot = π * (c[1] + 2c[2] / 3 + c[3] / 2)
    P = _P_quadratic(z, p)
    Fx = 1 - (c[1] * P[1] + c[2] * P[2] + c[3] * P[3]) / tot
    q = _qfam()
    Pz0, Pp0, Pz1, Pp1 = _bderiv01(z, p)
    Pzh, Pph = _bderiv(q[2], z, p)
    dz = -(c[1] * Pz0 + c[2] * Pzh + c[3] * Pz1) / tot
    dp = -(c[1] * Pp0 + c[2] * Pph + c[3] * Pp1) / tot
    dc1 = -(P[1] - (1 - Fx) * π) / tot
    dc2 = -(P[2] - (1 - Fx) * 2π / 3) / tot
    dc3 = -(P[3] - (1 - Fx) * π / 2) / tot
    # c = (1 - u1 - u2, u1 + 2u2, -u2)
    return Fx, s * dz, dp, -dc1 + dc2, -dc1 + 2dc2 - dc3
end

"""
    flux_gradient(z, p, law; series_tol = 1e-9) -> (F, ∂F/∂z, ∂F/∂p, ∂F/∂c)

Flux and its exact derivatives with respect to the separation, the radius ratio and
the coefficients `c` of the power terms `I = Σ c_i μ^α_i` of the law
(see [`power_terms`](@ref)). Use [`flux_gradient_law`](@ref) for the derivatives with
respect to the parameters of the law itself.
"""
function flux_gradient(z, p, law::LimbDarkening; series_tol = SERIES_TOL)
    α, c = power_terms(law)
    T = float(promote_type(typeof(z), typeof(p), eltype(c)))
    γ = T.(α) ./ 2
    F = occult(z, p, law; series_tol = series_tol, limb = :fast)
    Fx = occult(z, p, law; series_tol = 0, limb = :quadrature)          # exact to ~1e-16, for ∂F/∂c
    tot = sum(c[i] * T(π) / (γ[i] + 1) for i in eachindex(c))
    dz = zero(T)
    dp = zero(T)
    dc = zeros(T, length(c))
    for i in eachindex(c)
        Pz, Pp = blocked_derivatives(z, p, γ[i])
        dz -= c[i] * Pz / tot
        dp -= c[i] * Pp / tot
        Pi = blocked_flux(z, p, γ[i]; series_tol = 0, limb = :quadrature)     # exact to ~1e-16
        dc[i] = -(Pi - (1 - Fx) * T(π) / (γ[i] + 1)) / tot
    end
    return F, (abs(z) == z ? dz : -dz), dp, dc
end

"""
    coefficient_jacobian(law) -> J

`J[i, j] = ∂c_i/∂u_j`: derivatives of the power-term coefficients with respect to the
law's own parameters (linear laws only: all laws except the exponent of `Power2`
and the exponents of `PowerSum`, whose derivatives need the derivative of P_γ
with respect to γ; use ForwardDiff for those).
"""
coefficient_jacobian(::Uniform) = zeros(1, 0)
coefficient_jacobian(::Linear) = reshape([-1.0, 1.0], 2, 1)
coefficient_jacobian(::Quadratic) = [-1.0 -1.0; 1.0 2.0; 0.0 -1.0]
coefficient_jacobian(::SquareRoot) = [-1.0 -1.0; 1.0 0.0; 0.0 1.0]
coefficient_jacobian(::Nonlinear) = [-1.0 -1.0 -1.0 -1.0; 1.0 0 0 0; 0 1.0 0 0; 0 0 1.0 0; 0 0 0 1.0]
coefficient_jacobian(::Power2) = reshape([-1.0, 1.0], 2, 1)          # with respect to c only
function coefficient_jacobian(L::Polynomial)
    N = length(L.u)
    J = zeros(N + 1, N)
    for n in 1:N
        b = 1.0
        for k in 0:n
            J[k+1, n] = -b
            b = -b * (n - k) / (k + 1)
        end
    end
    J
end

"""
    flux_gradient_law(z, p, law) -> (F, ∂F/∂z, ∂F/∂p, ∂F/∂u)

Like [`flux_gradient`](@ref), with the derivatives taken with respect to the law's
parameters (`u1, u2` of the quadratic law, `c1..c4` of the nonlinear law, `c` of
the power-2 law, the `u_n` of the polynomial law).
"""
function flux_gradient_law(z, p, law::LimbDarkening; kw...)
    F, dz, dp, dc = flux_gradient(z, p, law; kw...)
    return F, dz, dp, coefficient_jacobian(law)' * dc
end
