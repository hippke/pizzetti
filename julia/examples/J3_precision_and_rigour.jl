# # J3 · Precision and rigour: certified light curves, interval enclosures, BigFloat
#
# Level: advanced · Prerequisites: J1 · Runtime: ~2 min
#
# Three levels of rigour, from cheap to strict:
#  1. `occult` (default): interior series with a proven bound (series_tol, default 1e-9);
#     near the limb the trapezoid rule or its interpolants (~1e-15, verified numerically).
#  2. `occult_certified`: the limb pieces carry a proven bound as well (paper, Theorem 3);
#     returns, per sample, whether its error is proven to be below series_tol.
#  3. `occult_interval` (IntervalArithmetic): an interval that contains the exact flux,
#     including all rounding errors of the evaluation (interior samples).
# Everything is checked here against an independent BigFloat computation.
#
# Run with:  julia --project=. J3_precision_and_rigour.jl

using Pizzetti, IntervalArithmetic, Printf

law = Nonlinear(0.6, -0.2, 0.3, -0.1)
lawb = Nonlinear(big"0.6", big"-0.2", big"0.3", big"-0.1")
z = collect(range(0, 1.25, length = 600))

# ## Level 2: certified light curves
# (The time of the first call includes compilation.)
for p in (0.0092, 0.1, 0.2, 0.25), τ in (1e-9, 1e-12)
    t = @elapsed (f, cert) = occult_certified(z, p, law; series_tol = τ)
    ref = setprecision(BigFloat, 128) do
        ob = Occultor(big(p), lawb; series_tol = 0)          # BigFloat quadrature, no series
        [Float64(ob(big(zi))) for zi in z]
    end
    err = abs.(f .- ref)
    @printf("p = %-6g τ = %.0e: %5.1f %% certified, max error certified %.1e, uncertified %.1e  (%.0f ms)\n",
            p, τ, 100 * count(cert) / length(z), maximum(err[cert]; init = 0.0),
            maximum(err[.!cert]; init = 0.0), 1e3t)
end

# ## Level 3: an enclosure of the exact flux (interior)
println()
for p in (0.02, 0.1), zi in (0.0, 0.5, 0.8)
    I = occult_interval(zi, p, law)
    ref = setprecision(BigFloat, 200) do
        occult(big(zi), big(p), lawb; series_tol = 0)
    end
    @printf("p = %-5g z = %-4g  F ∈ [%.17f, %.17f]  width %.1e  contains the 60-digit value: %s\n",
            p, zi, inf(I), sup(I), diam(I), inf(I) <= ref <= sup(I))
end

# ## Theorem 2 in action: the bound and the true error of the truncated series
println("\nγ = 1/2, p = 0.1: true truncation error vs bound B_M (both in BigFloat)")
setprecision(BigFloat, 200) do
    p = big"0.1"
    γ = big"0.5"
    A = series_table(γ, 11)
    for zf in (0.0, 0.3, 0.6, 0.8, 0.85)
        zz = big(zf)
        w = 1 - zz^2
        S = w^γ * sum(sum(A[j+1, q+1] * p^(2q) for q in 0:11) * (1 / w)^j for j in 0:11)
        D = blocked_flux(zz, p, γ; series_tol = 0) / (big(π) * p^2)
        B = series_bound(zz, p, γ, 11)
        @printf("z = %.2f  error %.3e  bound %.3e  error/bound %.3f\n", zf, Float64(D - S), Float64(B), Float64(abs(D - S) / B))
    end
end

# ## Arbitrary precision: references of your own
# Everything is generic in the number type. With BigFloat, the series order M and the
# tolerance can be raised until the result is converged to any number of digits:
println("\nBigFloat light-curve point, z = 0.5, p = 0.1, power-2 law")
setprecision(BigFloat, 256) do
    lb = Power2(big"0.6", big"0.6")
    prev = big(0)
    for (M, τ) in ((11, 1e-9), (21, 1e-20), (41, 1e-40), (61, 1e-60))
        v = occult(big"0.5", big"0.1", lb; series_tol = τ, M = M)
        @printf("M = %2d, series_tol = %.0e: F = %s   change %.1e\n", M, τ, string(v)[1:40], Float64(abs(v - prev)))
        prev = v
    end
end

# ## What you learned
# - the default `occult` carries a proven bound in the interior and ~1e-15 at the limb;
# - `occult_certified` extends the proof to the limb and tells you which samples carry it;
# - `occult_interval` encloses the exact flux including rounding errors;
# - Theorem 2's bound is tight (error/bound close to one) and BigFloat gives references to
#   any precision.
#
# Next: J4_photodynamics.jl.
