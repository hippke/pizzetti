# Interface of Limbdark.jl (Agol, Luger & Foreman-Mackey 2020): polynomial limb
# darkening I(μ) = 1 - Σ_n u_n (1 - μ)^n, occultor radius r and impact parameter b.

"""
    transit_poly(r, b, u_n)

Flux of a star with polynomial limb darkening `I(μ) = 1 - Σ u_n (1 - μ)^n`
occulted by a disk of radius `r` at separation `b` (same call as Limbdark.jl). Degrees
≤ 2 use the exact closed form of Mandel & Agol (2002); higher degrees the methods of
[`occult`](@ref) (interior series within its proven bound).
"""
function transit_poly(r::Real, b::Real, u_n::AbstractVector)
    if length(u_n) <= 2 && r isa Float64 && b isa Float64 && eltype(u_n) == Float64
        u1 = length(u_n) >= 1 ? u_n[1] : 0.0
        u2 = length(u_n) >= 2 ? u_n[2] : 0.0
        return quadratic_flux(b, r, u1, u2)          # exact closed form, as Limbdark.jl
    end
    occult(b, r, Polynomial(u_n))
end

"""
    transit_poly!(r, b, u_n, dfdrbu)

Flux as [`transit_poly`](@ref); fills `dfdrbu` (length `N + 2`) with
`[∂F/∂r, ∂F/∂b, ∂F/∂u_1, …, ∂F/∂u_N]`, the exact derivatives of Section 5.4
(same call as Limbdark.jl).
"""
function transit_poly!(r::Real, b::Real, u_n::AbstractVector, dfdrbu::AbstractVector)
    length(dfdrbu) == length(u_n) + 2 || throw(DimensionMismatch("dfdrbu must have length N + 2"))
    if length(u_n) <= 2 && r isa Float64 && b isa Float64 && eltype(u_n) == Float64 && abs(r) < 1
        u1 = length(u_n) >= 1 ? u_n[1] : 0.0
        u2 = length(u_n) >= 2 ? u_n[2] : 0.0
        F, db, dr, d1, d2 = flux_gradient_quadratic(b, r, u1, u2)
        dfdrbu[1] = dr
        dfdrbu[2] = db
        length(u_n) >= 1 && (dfdrbu[3] = d1)
        length(u_n) >= 2 && (dfdrbu[4] = d2)
        return F
    end
    F, db, dr, du = flux_gradient_law(b, r, Polynomial(u_n))
    dfdrbu[1] = dr
    dfdrbu[2] = db
    dfdrbu[3:end] .= du
    return F
end
