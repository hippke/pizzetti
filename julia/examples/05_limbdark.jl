# # Pizzetti.jl next to Limbdark.jl (Agol, Luger & Foreman-Mackey 2020)
#
# Pizzetti provides Limbdark.jl's interface for polynomial limb darkening,
#     I(μ) = 1 - Σ_n u_n (1 - μ)^n,   transit_poly(r, b, u_n),   transit_poly!(r, b, u_n, dfdrbu),
# so the two can be swapped. This script compares fluxes and gradients on a grid and
# times both. Limbdark.jl is the analytic solution of Agol et al. (2020) for polynomial laws;
# Pizzetti uses the interior series (proven to 1e-9 by default) and, near the limb, the exact
# quadratic solution (degree ≤ 2) or the contact expansions/quadrature (higher degree).
#
# Run with:  julia --project=. 05_limbdark.jl

using Pizzetti, Printf
import Limbdark

for u in ([0.4, 0.25], [0.3, 0.2, 0.1], [0.5, -0.2, 0.3, -0.1])
    worst_f = 0.0
    worst_g = 0.0
    for r in (0.01, 0.05, 0.1, 0.2), b in range(0, 1 + r - 1e-9, length = 301)
        f1 = Pizzetti.transit_poly(r, b, u)
        f2 = Limbdark.transit_poly(r, b, u)
        worst_f = max(worst_f, abs(f1 - f2))
        g1 = zeros(length(u) + 2)
        g2 = zeros(length(u) + 2)
        Pizzetti.transit_poly!(r, b, u, g1)
        Limbdark.transit_poly!(r, b, u, g2)
        worst_g = max(worst_g, maximum(abs.(g1 .- g2)))
    end
    @printf("u = %-22s  max |ΔF| = %.1e   max |Δ gradient| = %.1e\n", string(u), worst_f, worst_g)
end

# Timing per call (single point; the per-light-curve set-up is included in Pizzetti's number,
# so this favours Limbdark.jl; see benchmark/ for light curves).
using BenchmarkTools
u = [0.4, 0.25]
for b in (0.3, 0.95)
    t1 = @belapsed Pizzetti.transit_poly(0.1, $b, $u)
    t2 = @belapsed Limbdark.transit_poly(0.1, $b, $u)
    @printf("b = %.2f: Pizzetti %.0f ns, Limbdark %.0f ns per call\n", b, 1e9t1, 1e9t2)
end
# A light curve with a precomputed Occultor (the way Pizzetti is meant to be used):
z = collect(range(0, 1.1, length = 10_000))
o = Occultor(0.1, Polynomial(u))
out = similar(z)
t1 = @belapsed evaluate!($out, $o, $z)
t2 = @belapsed [Limbdark.transit_poly(0.1, zi, $u) for zi in $z]
# Limbdark.jl with its transit structure re-used across samples (no re-initialisation)
ts = Limbdark.transit_init(0.1, 0.5, u, false)
function limbdark_lc!(out, ts, z)
    for i in eachindex(z)
        ts.b = z[i]
        out[i] = Limbdark.transit_poly!(ts)
    end
    out
end
out2 = similar(z)
t3 = @belapsed limbdark_lc!($out2, $ts, $z)
@printf("10 000 samples: Pizzetti %.3f ms; Limbdark %.3f ms (transit_poly), %.3f ms (re-used structure)\n",
        1e3t1, 1e3t2, 1e3t3)
@printf("largest difference of the two light curves: %.1e\n", maximum(abs.(out .- out2)))
