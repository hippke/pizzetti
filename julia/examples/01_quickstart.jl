# # Pizzetti.jl in five minutes
#
# Run with:  julia --project=. 01_quickstart.jl   (from julia/examples; writes figures/)

using Pizzetti
using Plots
ENV["GKSwstype"] = "100"                     # headless plotting
mkpath(joinpath(@__DIR__, "figures"))

# ## 1. Flux of an occulted star
# `occult(z, p, law)`: relative flux for projected separations z (stellar radii),
# radius ratio p and a limb-darkening law.
law = Power2(0.6, 0.6)                          # I(μ) = 1 - c (1 - μ^α)
z = range(0, 1.15, length = 2000)
f = occult(z, 0.1, law)
println("flux at z = 0: ", f[1])

# Every law is a sum of powers of μ:
for L in (Quadratic(0.4, 0.25), SquareRoot(0.3, 0.4), Nonlinear(0.5, 0.1, 0.2, -0.1), Power2(0.6, 0.6),
          Polynomial([0.3, 0.2, 0.1]))
    α, c = power_terms(L)
    println(rpad(string(L), 46), " exponents ", α, "  coefficients ", round.(c, digits = 3))
end

# ## 2. Which method computes which sample
# An `Occultor` holds everything that depends only on p and the law (set up once per
# light curve). While the planet is inside the disk and z ≤ z_c, the flux is the interior
# series, whose error is *proven* to be below `series_tol` (default 1e-9). Beyond z_c:
# power series about the contacts (:contact), or quadrature for short light curves.
o = Occultor(0.1, law; limb = :fast)
println("series cut z_c = ", o.zc, "  (1 - p = 0.9)")
m = method.(Ref(o), z)
cols = Dict(:series => :royalblue, :contact => :darkorange, :quadrature => :gray, :out => :black)
plt = plot(xlabel = "z (stellar radii)", ylabel = "relative flux", legend = :bottomright, size = (700, 420))
for k in (:series, :contact, :out)
    sel = m .== k
    any(sel) && scatter!(plt, z[sel], o.(z[sel]), ms = 1.5, msw = 0, color = cols[k], label = string(k))
end
vline!(plt, [o.zc], ls = :dash, color = :gray, label = "z_c")
savefig(plt, joinpath(@__DIR__, "figures", "01_methods.png"))

# ## 3. A transit light curve
prm = TransitParams(per = 3.5, rp = 0.1, a = 9.0, b = 0.35, law = Quadratic(0.40, 0.25))
t = range(-0.15, 0.15, length = 3000)
flux = light_curve(t, prm)
flux30 = light_curve(t, prm; exp_time = 30 / 1440, supersample = 7)   # 30-min exposures
plt = plot(24 .* t, 1e6 .* (flux .- 1), label = "instantaneous", xlabel = "hours from mid-transit",
           ylabel = "flux - 1 (ppm)", lw = 2, size = (700, 420))
plot!(plt, 24 .* t, 1e6 .* (flux30 .- 1), label = "30-min exposures", lw = 2, ls = :dash)
savefig(plt, joinpath(@__DIR__, "figures", "01_lightcurve.png"))

# ## 4. Derivatives
# With ForwardDiff loaded, dual numbers flow through everything; the flux derivatives
# with respect to z and p are the exact hypergeometric expressions of the paper (Sec. 5.4).
using ForwardDiff
g = ForwardDiff.gradient(v -> occult(v[1], v[2], Power2(v[3], v[4])), [0.95, 0.1, 0.6, 0.6])
println("∂F/∂(z, p, c, α) at z = 0.95: ", g)
F, dz, dp, dc = flux_gradient(0.95, 0.1, law)          # the same, without automatic differentiation
println("flux_gradient:                  ", [dz, dp])

# ## 5. Arbitrary precision
setprecision(BigFloat, 200) do
    fb = occult(big"0.3", big"0.1", Power2(big"0.6", big"0.6"); series_tol = 1e-40, M = 61)
    println("flux at z = 0.3 to 60 digits: ", fb)
    println("Float64 result differs by:    ", Float64(occult(0.3, 0.1, law) - fb))
end
println("figures written to ", joinpath(@__DIR__, "figures"))
