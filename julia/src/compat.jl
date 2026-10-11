# Interface of Limbdark.jl (Agol, Luger & Foreman-Mackey 2020): polynomial limb
# darkening I(μ) = 1 - Σ_n u_n (1 - μ)^n, occultor radius r and impact parameter b.

"""
    transit_poly(r, b, u_n)

Flux of a star with polynomial limb darkening `I(μ) = 1 - Σ u_n (1 - μ)^n`
occulted by a disk of radius `r` at separation `b` (same call as Limbdark.jl).
"""
transit_poly(r::Real, b::Real, u_n::AbstractVector) = occult(b, r, Polynomial(u_n))

"""
    transit_poly!(r, b, u_n, dfdrbu)

Flux as [`transit_poly`](@ref); fills `dfdrbu` (length `N + 2`) with
`[∂F/∂r, ∂F/∂b, ∂F/∂u_1, …, ∂F/∂u_N]`, the exact derivatives of Section 5.4
(same call as Limbdark.jl).
"""
function transit_poly!(r::Real, b::Real, u_n::AbstractVector, dfdrbu::AbstractVector)
    length(dfdrbu) == length(u_n) + 2 || throw(DimensionMismatch("dfdrbu must have length N + 2"))
    F, db, dr, du = flux_gradient_law(b, r, Polynomial(u_n))
    dfdrbu[1] = dr
    dfdrbu[2] = db
    dfdrbu[3:end] .= du
    return F
end
