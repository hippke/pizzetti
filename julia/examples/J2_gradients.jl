# # J2 · Gradients: a gradient-based fit of a super-Earth with the power-2 law
#
# Level: intermediate · Prerequisites: J1 · Runtime: ~2 min
#
# Kepler-like data: 30 transits at 1-min cadence, p = 0.02, power-2 limb darkening with
# both coefficients free. The likelihood is differentiated with ForwardDiff through the
# orbit and the light curve; Pizzetti supplies exact flux derivatives (paper, Sec. 5.4).
# We compare a gradient-free optimiser with L-BFGS, check the gradient against finite
# differences, and get parameter uncertainties from the Hessian (Laplace approximation).
#
# Run with:  julia --project=. J2_gradients.jl

using Pizzetti, ForwardDiff, Optim, Random, LinearAlgebra, Printf

# ## Data
truth = (t0 = 0.0, per = 10.0, rp = 0.02, a = 20.0, b = 0.4, c = 0.6, α = 0.6)
σ = 150e-6
t = vec([e * truth.per + dt for dt in -0.12:(1 / 1440):0.12, e in 0:29])
model(θ) = light_curve(t, TransitParams(t0 = θ[1], per = θ[2], rp = θ[3], a = θ[4], b = θ[5],
                                         law = Power2(θ[6], θ[7])))
θtrue = collect(values(truth))
# noise-free model: quadrature with the interior series switched off
f0 = occult(rsky(t, TransitParams(t0 = 0.0, per = 10.0, rp = 0.02, a = 20.0, b = 0.4, law = Power2(0.6, 0.6))),
            0.02, Power2(0.6, 0.6); series_tol = 0, limb = :quadrature)
y = f0 .+ σ .* randn(Xoshiro(7), length(t))
println(length(t), " samples")

# Parameters are scaled to order unity for the optimiser; b enters through |b|.
scale = [1e-4, 1e-6, 1e-4, 0.3, 0.1, 0.1, 0.1]
function nll(x)
    θ = θtrue .+ x .* scale
    θ[5] = abs(θ[5])
    (0 < θ[6] < 1 && 0 < θ[7] < 2 && θ[4] > 1) || return oftype(x[1], Inf)
    r = y .- model(θ)
    return sum(abs2, r) / (2σ^2)
end
x0 = [3.0, 2.0, 5.0, -1.3, 1.0, -1.0, 1.5]          # a start away from the truth

# ## Cost of one evaluation and of one gradient
nll(x0); ForwardDiff.gradient(nll, x0)
tv = @elapsed for _ in 1:20; nll(x0); end
tg = @elapsed for _ in 1:20; ForwardDiff.gradient(nll, x0); end
@printf("likelihood %.2f ms, gradient (7 parameters, ForwardDiff) %.2f ms = %.1f likelihoods\n",
        1e3tv / 20, 1e3tg / 20, tg / tv)

# ## Gradient check against central finite differences
g = ForwardDiff.gradient(nll, x0)
h = 1e-4
gfd = [(nll(x0 .+ h .* (1:7 .== i)) - nll(x0 .- h .* (1:7 .== i))) / 2h for i in 1:7]
println("gradient, ForwardDiff:        ", round.(g, sigdigits = 8))
println("gradient, finite differences: ", round.(gfd, sigdigits = 8))

# ## Optimisation: gradient-free vs gradient-based
lower = [-Inf, -Inf, -Inf, (1.5 - 20.0) / 0.3, -Inf, (0.0 - 0.6) / 0.1, (0.0 - 0.6) / 0.1] .+ 1e-6
upper = [Inf, Inf, Inf, Inf, Inf, (1.0 - 0.6) / 0.1, (2.0 - 0.6) / 0.1] .- 1e-6
run_nm() = optimize(nll, x0, NelderMead(), Optim.Options(iterations = 20_000, g_tol = 1e-10))
run_bfgs() = optimize(nll, x -> ForwardDiff.gradient(nll, x), lower, upper, x0, Fminbox(LBFGS()),
                      Optim.Options(g_tol = 1e-6); inplace = false)
run_nm(); run_bfgs()                        # compile first, then time
r_nm = run_nm()
r_bfgs = run_bfgs()
for (name, r) in (("Nelder-Mead", r_nm), ("L-BFGS + ForwardDiff", r_bfgs))
    @printf("%-22s -lnL = %.4f  f-calls %5d  g-calls %4d  time %.2f s\n", name, Optim.minimum(r),
            Optim.f_calls(r), Optim.g_calls(r), Optim.time_run(r))
end
xbest = Optim.minimizer(r_bfgs)

# ## Uncertainties from the Hessian (Laplace approximation)
H = ForwardDiff.hessian(nll, xbest)
Σ = inv(Symmetric(H))
θbest = θtrue .+ xbest .* scale
θbest[5] = abs(θbest[5])
names = ("t0", "P", "Rp/R*", "a/R*", "b", "c", "alpha")
println("\nparameter      best fit      ± (Hessian)     truth")
for i in 1:7
    @printf("%-8s %14.8g  %12.3g  %12.8g\n", names[i], θbest[i], sqrt(Σ[i, i]) * scale[i], θtrue[i])
end

# ## What you learned
# - the whole model (orbit, light curve, χ²) is differentiable with ForwardDiff, and the
#   gradient of a seven-parameter likelihood costs about ten likelihoods;
# - L-BFGS with exact gradients needs far fewer likelihood calls than Nelder-Mead;
# - the Hessian at the optimum gives Laplace-approximation uncertainties.
#
# Next: J3_precision_and_rigour.jl, or J4_photodynamics.jl for gradients through an N-body code.
