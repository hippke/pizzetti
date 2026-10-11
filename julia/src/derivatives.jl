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

function _interior_F12(γ::Float64, κ::Float64)
    h1, h2 = lock(_F12_LOCK) do
        length(_F12_CACHE) > 1024 && empty!(_F12_CACHE)
        get!(() -> (HypCache(-γ, 0.5, 1.0), HypCache(-γ, 1.5, 2.0)), _F12_CACHE, γ)
    end
    return h1(κ), h2(κ)
end
function _interior_F12(γ, κ)
    T = typeof(κ)
    half = one(T) / 2
    return hyp2f1_unit(-γ, half, one(T), κ), hyp2f1_unit(-γ, 3half, 2one(T), κ)
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
    zmp = z - p
    x = (1 - zmp) * (1 + zmp)                       # 1 - (z - p)², accurate
    xg = x^γ
    half = one(T) / 2
    if z + p <= 1
        κ = z > 0 ? 4z * p / x : zero(T)
        κ = min(κ, one(T))
        F1, F2 = _interior_F12(γ, κ)
        pre = 2T(π) * p * xg
        return (pre * (F2 - F1), pre * F1)
    end
    k = min(x / (4z * p), one(T))
    sk = sqrt(k)
    B1 = _beta(half, γ + 1)
    H1 = hyp2f1_unit(half, half, γ + 3half, k)
    H2 = hyp2f1_unit(half, 3half, γ + 5half, k)
    pre = 2p * xg * sk
    return (pre * (2k * _beta(3half, γ + 1) * H2 - B1 * H1), pre * B1 * H1)
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
