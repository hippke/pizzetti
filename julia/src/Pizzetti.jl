"""
    Pizzetti

Fast limb-darkened transit light curves with a proven error bound
(Hippke & Agol, in prep.). Julia implementation of the Python package
[pizzetti](https://github.com/hippke/pizzetti).

While the occultor is inside the stellar disk, the flux is `w^γ` times a
polynomial in `v = 1/(1 - z²)` (Pizzetti's mean-value expansion), with a closed-form
bound on the truncation error. Near and across the limb: the exact Mandel & Agol
solution (quadratic family), contact expansions built from Gauss hypergeometric
functions, or Green's-theorem quadrature.

Main entry points: [`occult`](@ref), [`Occultor`](@ref), [`light_curve`](@ref),
[`transit_poly`](@ref) (Limbdark.jl-compatible), [`flux_gradient`](@ref).
"""
module Pizzetti

using SpecialFunctions: loggamma, digamma
using LinearAlgebra: eigvals, SymTridiagonal

export LimbDarkening, Uniform, Linear, Quadratic, SquareRoot, Nonlinear, Power2, Polynomial, PowerSum
export power_terms, limb_darkening, intensity
export series_cut, series_bound, series_table, SERIES_M, SERIES_TOL
export Occultor, occult, occult!, evaluate!, method
export quadratic_flux, blocked_quadrature, hyp2f1_unit
export blocked_flux, blocked_derivatives, flux_gradient, flux_gradient_law, coefficient_jacobian
export transit_poly, transit_poly!
export ProvenLimb, occult_certified, occult_interval
export TransitParams, impact_parameter, eccentric_anomaly, rsky, light_curve, light_curve!

# ---- number-type helpers (extended for dual numbers and intervals) ----
"Primal value of a number (strips dual-number partials)."
_val(x) = x
"Floating-point type underlying a number type."
_float(::Type{T}) where {T<:AbstractFloat} = T
_float(::Type{T}) where {T<:Real} = Float64
_float(x) = _float(typeof(x))
"True if x equals v and carries no derivative information."
_isconst(x, v) = x == v

include("laws.jl")
include("simdmath.jl")
include("series.jl")
include("quadratic.jl")
include("quadrature.jl")
include("hyp.jl")
include("contact.jl")
include("proven.jl")
include("occult.jl")
include("derivatives.jl")
include("orbit.jl")
include("compat.jl")

"""
    occult_interval(z, p, law) -> Interval

Rigorous interval enclosure of the exact flux on the interior series (needs
`using IntervalArithmetic`; see the extension for details).
"""
function occult_interval end

end
