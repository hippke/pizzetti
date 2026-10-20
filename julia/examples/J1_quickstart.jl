# # J1 · Pizzetti.jl in ten minutes
#
# Level: beginner · Prerequisites: none · Runtime: ~1 min (mostly compilation)
#
# Covers: limb-darkening laws as sums of powers of μ, `occult`, the `Occultor` (set up once,
# evaluate many times), which method computes which sample, `evaluate!` for allocation-free
# loops, transit light curves with exposure-time integration, and a first look at derivatives
# and arbitrary precision (both treated in depth in J2 and J3).
#
# Run with:  julia --project=. J1_quickstart.jl   (from julia/examples; writes figures/)
# The scripts are written in the Literate.jl format: `Literate.notebook("J1_quickstart.jl")`
# turns them into notebooks, `Literate.markdown` into documentation pages.

using Pizzetti
using Plots
ENV["GKSwstype"] = "100"                     # headless plotting
mkpath(joinpath(@__DIR__, "figures"))

# ## 1. Flux of an occulted star
# `occult(z, p, law)`: relative flux for projected separations z (in stellar radii),
# radius ratio p and a limb-darkening law.
law = Power2(0.6, 0.6)                          # I(μ) = 1 - c (1 - μ^α)
z = range(0, 1.15, length = 2000)
f = occult(z, 0.1, law)
println("flux at z = 0: ", f[1])

# Every supported law is a sum of powers of μ; `power_terms` shows the decomposition.
# `PowerSum` takes arbitrary exponents, `Polynomial` the coefficients of (1 - μ)^n.
for L in (Quadratic(0.4, 0.25), SquareRoot(0.3, 0.4), Nonlinear(0.5, 0.1, 0.2, -0.1), Power2(0.6, 0.6),
          Polynomial([0.3, 0.2, 0.1]), PowerSum([0.0, 0.7, 1.3], [0.5, 0.3, 0.2]))
    α, c = power_terms(L)
    println(rpad(string(L), 52), " exponents ", round.(α, digits = 3), "  coefficients ", round.(c, digits = 3))
end

# ## 2. The Occultor, and which method computes which sample
# An `Occultor` holds everything that depends only on p and the law, so that it is set up
# once per light curve and then called like a function. While the planet is inside the disk
# and z ≤ z_c, the flux is the interior series, whose error is *proven* to be below
# `series_tol` (default 1e-9). Beyond z_c, the quadratic family uses the exact Mandel & Agol
# solution; other laws (for p < 0.3) the fixed-node trapezoid rule for the limb integral or,
# for long light curves (`nlimb` ≥ 150 limb samples), Chebyshev interpolants built from it.
o = Occultor(0.1, law; nlimb = length(z))
println("series cut z_c = ", round(o.zc, digits = 4), "  (1 - p = 0.9)")
m = method.(Ref(o), z)
println("methods used: ", Dict(k => count(==(k), m) for k in unique(m)))
cols = Dict(:series => :royalblue, :interpolate => :darkorange, :trapezoid => :seagreen,
            :contact => :purple, :quadrature => :gray, :out => :black, :full => :black)
plt = plot(xlabel = "z (stellar radii)", ylabel = "relative flux", legend = :bottomright, size = (700, 420))
for k in unique(m)
    sel = m .== k
    scatter!(plt, z[sel], o.(z[sel]), ms = 1.5, msw = 0, color = cols[k], label = string(k))
end
vline!(plt, [o.zc], ls = :dash, color = :gray, label = "z_c")
savefig(plt, joinpath(@__DIR__, "figures", "J1_methods.png"))

# The limb method can be chosen explicitly (`limb = :auto` is the default). All agree to ~1e-15 near the limb (the interior
# samples differ by at most series_tol from the exact value, here switched off with 0):
zl = collect(range(o.zc, 1.1, length = 500))
ref = occult(zl, 0.1, law; limb = :quadrature, series_tol = 0)
for lm in (:trapezoid, :interpolate, :fast, :quadrature, :proven)
    d = maximum(abs.(occult(zl, 0.1, law; limb = lm, series_tol = 0) .- ref))
    println(rpad(string(lm), 12), " max |ΔF| against quadrature: ", round(d, sigdigits = 2))
end
# (:fast is the contact expansions of the paper's Section 5.2, :proven their certified form.)

# ## 3. Evaluation into a buffer
# In a loop (an MCMC step, an N-body integration) the Occultor and an output buffer are reused.
# For the quadratic family `evaluate!` allocates nothing; for the other laws, only work arrays
# for the limb samples.
zs = collect(range(0, 1.2, length = 100_000))
out = similar(zs)
o2 = Occultor(0.1, law; nlimb = 20_000)
evaluate!(out, o2, zs)                                  # compile
t_eval = @elapsed for _ in 1:20; evaluate!(out, o2, zs); end
t_setup = @elapsed for _ in 1:20; Occultor(0.1, law; nlimb = 20_000); end
oq = Occultor(0.1, Quadratic(0.4, 0.25))
evaluate!(out, oq, zs)
allocs(o) = @allocated evaluate!(out, o, zs)
allocs(o2); allocs(oq)
println("power-2, evaluate! of 10^5 samples: ", round(1e3 * t_eval / 20, digits = 3), " ms; set-up: ",
        round(1e6 * t_setup / 20, digits = 1), " µs; allocated: ", round(allocs(o2) / 2^20, digits = 2), " MiB")
println("quadratic, allocated: ", allocs(oq), " bytes")

# ## 4. A transit light curve
# `TransitParams` + `light_curve` compute the Keplerian orbit and the flux; `b` or `inc`
# set the geometry, `ecc` and `w` (degrees) an eccentric orbit.
prm = TransitParams(per = 3.5, rp = 0.1, a = 9.0, b = 0.35, law = Quadratic(0.40, 0.25))
t = range(-0.15, 0.15, length = 3000)
flux = light_curve(t, prm)
flux30 = light_curve(t, prm; exp_time = 30 / 1440, supersample = 7)   # 30-min exposures
fecc = light_curve(t, TransitParams(per = 3.5, rp = 0.1, a = 9.0, b = 0.35, ecc = 0.3, w = 45.0,
                                    law = Quadratic(0.40, 0.25)))
plt = plot(24 .* t, 1e6 .* (flux .- 1), label = "instantaneous", xlabel = "hours from mid-transit",
           ylabel = "flux - 1 (ppm)", lw = 2, size = (700, 420))
plot!(plt, 24 .* t, 1e6 .* (flux30 .- 1), label = "30-min exposures", lw = 2, ls = :dash)
plot!(plt, 24 .* t, 1e6 .* (fecc .- 1), label = "e = 0.3, ω = 45°", lw = 2, ls = :dot)
savefig(plt, joinpath(@__DIR__, "figures", "J1_lightcurve.png"))

# ## 5. Derivatives (more in J2)
# With ForwardDiff loaded, dual numbers flow through everything; the flux derivatives
# with respect to z and p are the exact hypergeometric expressions of the paper (Sec. 5.4).
using ForwardDiff
g = ForwardDiff.gradient(v -> occult(v[1], v[2], Power2(v[3], v[4])), [0.95, 0.1, 0.6, 0.6])
println("∂F/∂(z, p, c, α) at z = 0.95: ", g)
F, dz, dp, dc = flux_gradient(0.95, 0.1, law)          # the same, without automatic differentiation
println("flux_gradient:                  ", [dz, dp])

# ## 6. Arbitrary precision (more in J3)
setprecision(BigFloat, 200) do
    fb = occult(big"0.3", big"0.1", Power2(big"0.6", big"0.6"); series_tol = 1e-40, M = 61)
    println("flux at z = 0.3 to 60 digits: ", fb)
    println("Float64 result differs by:    ", Float64(occult(0.3, 0.1, law) - fb))
end
println("figures written to ", joinpath(@__DIR__, "figures"))

# ## What you learned
# - every law is a sum of powers of μ, and `occult` / `Occultor` compute its flux;
# - the interior series covers most of a transit with a proven error; the limb uses the exact
#   quadratic solution, the trapezoid rule or its interpolants (or, on request, the contact
#   expansions, their proven form, or quadrature);
# - reuse an `Occultor` and `evaluate!` into a buffer in loops.
#
# Next: J2_gradients.jl (gradient-based fitting), J3_precision_and_rigour.jl (proofs at work).
