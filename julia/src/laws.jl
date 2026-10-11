# Limb-darkening laws. Every law is a sum of powers of mu = sqrt(1 - r^2):
#     I(mu) = sum_i c_i mu^alpha_i,
# which is all the occultation code needs (`power_terms`). The quadratic family
# (uniform, linear, quadratic) additionally has the exact elliptic-integral
# solution of Mandel & Agol (2002) for the samples near the limb.

"""
    LimbDarkening

Supertype of the limb-darkening laws. A law `L` provides
[`power_terms`](@ref)`(L) -> (α, c)` with `I(μ) = Σ c[i] μ^α[i]`.

Laws (batman conventions unless noted):

| law | intensity |
|---|---|
| `Uniform()` | `1` |
| `Linear(u1)` | `1 - u1 (1 - μ)` |
| `Quadratic(u1, u2)` | `1 - u1 (1 - μ) - u2 (1 - μ)^2` |
| `SquareRoot(u1, u2)` | `1 - u1 (1 - μ) - u2 (1 - √μ)` |
| `Nonlinear(c1, c2, c3, c4)` | `1 - Σ c_n (1 - μ^(n/2))` (Claret 2000) |
| `Power2(c, α)` | `1 - c (1 - μ^α)` (Hestroffer 1997) |
| `Polynomial(u)` | `1 - Σ u_n (1 - μ)^n` (Agol, Luger & Foreman-Mackey 2020; Limbdark.jl) |
| `PowerSum(α, c)` | `Σ c_i μ^α_i` (any non-negative exponents) |

The intensity is not normalised; the flux is relative to the unocculted star.
"""
abstract type LimbDarkening end

struct Uniform <: LimbDarkening end

struct Linear{T<:Real} <: LimbDarkening
    u1::T
end

struct Quadratic{T<:Real} <: LimbDarkening
    u1::T
    u2::T
end
Quadratic(u1::Real, u2::Real) = Quadratic(promote(u1, u2)...)

struct SquareRoot{T<:Real} <: LimbDarkening
    u1::T
    u2::T
end
SquareRoot(u1::Real, u2::Real) = SquareRoot(promote(u1, u2)...)

struct Nonlinear{T<:Real} <: LimbDarkening
    c1::T
    c2::T
    c3::T
    c4::T
end
Nonlinear(c1::Real, c2::Real, c3::Real, c4::Real) = Nonlinear(promote(c1, c2, c3, c4)...)

struct Power2{T<:Real} <: LimbDarkening
    c::T
    α::T
    function Power2{T}(c, α) where {T<:Real}
        _val(α) > 0 || throw(ArgumentError("power-2 limb darkening needs α > 0"))
        new{T}(c, α)
    end
end
function Power2(c::Real, α::Real)
    cc, aa = promote(float(c), float(α))
    Power2{typeof(cc)}(cc, aa)
end

struct Polynomial{T<:Real} <: LimbDarkening
    u::Vector{T}
end
Polynomial(u::AbstractVector{<:Real}) = Polynomial(collect(float.(u)))

struct PowerSum{T<:Real} <: LimbDarkening
    α::Vector{T}
    c::Vector{T}
    function PowerSum{T}(α, c) where {T<:Real}
        length(α) == length(c) || throw(DimensionMismatch("α and c must have the same length"))
        all(a -> _val(a) >= 0, α) || throw(ArgumentError("exponents must be non-negative"))
        new{T}(α, c)
    end
end
function PowerSum(α::AbstractVector{<:Real}, c::AbstractVector{<:Real})
    T = float(promote_type(eltype(α), eltype(c)))
    PowerSum{T}(collect(T, α), collect(T, c))
end

"""
    power_terms(law) -> (α, c)

Exponents and coefficients with `I(μ) = Σ c[i] μ^α[i]`. Terms with equal
exponents are merged, zero coefficients are kept (their derivatives matter).
"""
power_terms(::Uniform) = ([0.0], [1.0])
function power_terms(L::Linear{T}) where {T}
    F = float(T)
    (F[0, 1], F[1 - L.u1, L.u1])
end
function power_terms(L::Quadratic{T}) where {T}
    F = float(T)
    (F[0, 1, 2], F[1 - L.u1 - L.u2, L.u1 + 2L.u2, -L.u2])
end
function power_terms(L::SquareRoot{T}) where {T}
    F = float(T)
    (F[0, 1, 0.5], F[1 - L.u1 - L.u2, L.u1, L.u2])
end
function power_terms(L::Nonlinear{T}) where {T}
    F = float(T)
    (F[0, 0.5, 1, 1.5, 2], F[1 - L.c1 - L.c2 - L.c3 - L.c4, L.c1, L.c2, L.c3, L.c4])
end
function power_terms(L::Power2{T}) where {T}
    F = float(T)
    (F[0, L.α], F[1 - L.c, L.c])
end
power_terms(L::PowerSum) = (copy(L.α), copy(L.c))
function power_terms(L::Polynomial{T}) where {T}
    # 1 - sum_n u_n (1 - mu)^n = sum_k c_k mu^k
    F = float(T)
    N = length(L.u)
    c = zeros(F, N + 1)
    c[1] = one(F)
    for n in 1:N
        b = one(F)                       # binomial(n, k) (-1)^k
        for k in 0:n
            c[k+1] -= L.u[n] * b
            b = -b * (n - k) / (k + 1)
        end
    end
    (F[k for k in 0:N], c)
end

"""
    quadratic_coefficients(law) -> (u1, u2) or `nothing`

The quadratic-law coefficients of a law of the quadratic family (uniform,
linear, quadratic, and polynomials of degree <= 2); `nothing` otherwise.
"""
quadratic_coefficients(::LimbDarkening) = nothing
quadratic_coefficients(::Uniform) = (0.0, 0.0)
quadratic_coefficients(L::Linear{T}) where {T} = (float(L.u1), zero(float(T)))
quadratic_coefficients(L::Quadratic) = (float(L.u1), float(L.u2))
function quadratic_coefficients(L::Polynomial{T}) where {T}
    N = length(L.u)
    N == 0 && return (zero(float(T)), zero(float(T)))
    N == 1 && return (float(L.u[1]), zero(float(T)))
    N == 2 && return (float(L.u[1]), float(L.u[2]))
    return nothing
end

const _BATMAN_LAWS = Dict("uniform" => 0, "linear" => 1, "quadratic" => 2, "squareroot" => 2,
                          "nonlinear" => 4, "power2" => 2)

"""
    limb_darkening(name, u)

Law from batman's name and coefficient list, e.g. `limb_darkening("power2", [0.6, 0.6])`.
"""
function limb_darkening(name::AbstractString, u::AbstractVector)
    if name in ("logarithmic", "exponential", "custom")
        throw(ArgumentError("\"$name\" limb darkening is not a sum of powers of μ and is not supported"))
    end
    haskey(_BATMAN_LAWS, name) || throw(ArgumentError("unknown limb-darkening law \"$name\""))
    n = _BATMAN_LAWS[name]
    length(u) == n || throw(ArgumentError("$name limb darkening needs $n coefficient(s), got $(length(u))"))
    name == "uniform" && return Uniform()
    name == "linear" && return Linear(u[1])
    name == "quadratic" && return Quadratic(u[1], u[2])
    name == "squareroot" && return SquareRoot(u[1], u[2])
    name == "nonlinear" && return Nonlinear(u[1], u[2], u[3], u[4])
    return Power2(u[1], u[2])
end
limb_darkening(name::Symbol, u) = limb_darkening(String(name), u)

"""
    intensity(law, μ)

Specific intensity `I(μ)` (not normalised).
"""
function intensity(L::LimbDarkening, μ)
    α, c = power_terms(L)
    s = zero(promote_type(eltype(c), typeof(μ)))
    for i in eachindex(α)
        s += c[i] * (α[i] == 0 ? one(μ) : μ^α[i])
    end
    s
end
