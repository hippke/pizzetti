# Benchmarks of Pizzetti.jl (single thread). Run from julia/benchmark:
#     julia --project=. -O3 benchmarks.jl
# Writes results.md (Markdown tables) next to this file.

using Pizzetti, BenchmarkTools, ForwardDiff, Printf
import Limbdark

const OUT = IOBuffer()
out(s...) = (print(stdout, s...); print(OUT, s...))

t_ns(b) = minimum(b).time
cpu = try
    strip(split(read(`lscpu`, String), "Model name:")[2] |> x -> split(x, "\n")[1])
catch
    Sys.cpu_info()[1].model
end
out("# Pizzetti.jl benchmarks\n\nJulia ", VERSION, ", ", cpu, ", single thread. Minimum over repeated runs.\n\n")

# ---------------------------------------------------------------- per sample
out("## Cost per sample\n\n| law | p | interior series | limb: contact | limb: quadrature | quadratic: exact |\n|---|---|---|---|---|---|\n")
for law in (Quadratic(0.4, 0.25), Power2(0.6, 0.6), Nonlinear(0.5, 0.1, 0.2, -0.1)), p in (0.0092, 0.1)
    o = Occultor(p, law; limb = :fast)
    zi = collect(range(0, o.zc, length = 10^5))
    zl = collect(range(o.zc + 1e-9, 1 + p - 1e-9, length = 10^4))
    a = similar(zi); b = similar(zl)
    ti = t_ns(@benchmark evaluate!($a, $o, $zi)) / length(zi)
    if law isa Quadratic
        tx = t_ns(@benchmark $b .= $o.($zl)) / length(zl)
        out(@sprintf("| %s | %g | %.2f ns | — | — | %.0f ns |\n", nameof(typeof(law)), p, ti, tx))
    else
        tc = t_ns(@benchmark $b .= $o.($zl)) / length(zl)
        oq = Occultor(p, law; limb = :quadrature)
        tq = t_ns(@benchmark $b .= $oq.($zl)) / length(zl)
        out(@sprintf("| %s | %g | %.2f ns | %.0f ns | %.0f ns | — |\n", nameof(typeof(law)), p, ti, tc, tq))
    end
end

out("\n## Set-up per light curve\n\n| law | p | series cut + coefficients | contact expansions | proven pieces |\n|---|---|---|---|---|\n")
for law in (Power2(0.6, 0.6), Nonlinear(0.5, 0.1, 0.2, -0.1)), p in (0.0092, 0.1)
    t1 = t_ns(@benchmark Occultor($p, $law; limb = :quadrature))
    t2 = t_ns(@benchmark Occultor($p, $law; limb = :fast))
    t3 = t_ns(@benchmark Occultor($p, $law; limb = :proven) samples = 3 evals = 1)
    out(@sprintf("| %s | %g | %.1f µs | %.2f ms | %.1f ms |\n", nameof(typeof(law)), p, t1 / 1e3, t2 / 1e6, t3 / 1e6))
end

# ---------------------------------------------------------------- light curves
out("\n## Light curves (orbit + flux), as in a fit\n\n")
out("Kepler-like super-Earth: 30 transits at 1-min cadence (10 380 samples), p = 0.02, a/R = 20, b = 0.4;\n")
out("TESS-like hot Jupiter: 7 transits at 2-min cadence (1316 samples), p = 0.1, a/R = 9, b = 0.35.\n\n")
out("| case | law | light curve | gradient (ForwardDiff, 7 parameters) |\n|---|---|---|---|\n")
cases = (("Kepler super-Earth", vec([e * 10.0 + dt for dt in -0.12:(1 / 1440):0.12, e in 0:29]), 10.0, 0.02, 20.0, 0.4),
         ("TESS hot Jupiter", vec([e * 3.5 + dt for dt in -0.13:(2 / 1440):0.13, e in 0:6]), 3.5, 0.1, 9.0, 0.35))
for (name, t, per, rp, a, b) in cases, law in (Quadratic(0.4, 0.25), Power2(0.6, 0.6))
    lc(θ) = light_curve(t, TransitParams(t0 = θ[1], per = θ[2], rp = θ[3], a = θ[4], b = θ[5],
                                         law = law isa Quadratic ? Quadratic(θ[6], θ[7]) : Power2(θ[6], θ[7])))
    θ = [0.0, per, rp, a, b, (law isa Quadratic ? (0.4, 0.25) : (0.6, 0.6))...]
    tl = t_ns(@benchmark $lc($θ))
    f(x) = sum(abs2, lc(x))
    tg = t_ns(@benchmark ForwardDiff.gradient($f, $θ) samples = 20 evals = 1)
    out(@sprintf("| %s | %s | %.3f ms | %.2f ms |\n", name, nameof(typeof(law)), tl / 1e6, tg / 1e6))
end

# ---------------------------------------------------------------- Limbdark.jl
out("\n## Against Limbdark.jl (quadratic law, p = 0.1, 10 000 samples)\n\n")
z = collect(range(0, 1.1, length = 10_000))
u = [0.4, 0.25]
o = Occultor(0.1, Polynomial(u))
r = similar(z)
t1 = t_ns(@benchmark evaluate!($r, $o, $z))
ts = Limbdark.transit_init(0.1, 0.5, u, false)
function ld!(out, ts, z)
    for i in eachindex(z)
        ts.b = z[i]
        out[i] = Limbdark.transit_poly!(ts)
    end
    out
end
r2 = similar(z)
t2 = t_ns(@benchmark ld!($r2, $ts, $z))
t3 = t_ns(@benchmark Pizzetti.transit_poly(0.1, 0.5, $u))
t4 = t_ns(@benchmark Limbdark.transit_poly(0.1, 0.5, $u))
# with the gradient [∂F/∂r, ∂F/∂b, ∂F/∂u1, ∂F/∂u2] at every sample
tsg = Limbdark.transit_init(0.1, 0.5, u, true)
r3 = similar(z)
t5 = t_ns(@benchmark ld!($r3, $tsg, $z))
g = zeros(4)
function pz!(out, g, u, z)
    for i in eachindex(z)
        out[i] = Pizzetti.transit_poly!(0.1, z[i], u, g)
    end
    out
end
r4 = similar(z)
t6 = t_ns(@benchmark pz!($r4, $g, $u, $z))
out(@sprintf("| | Pizzetti.jl | Limbdark.jl |\n|---|---|---|\n| light curve | %.3f ms | %.3f ms |\n| light curve with gradient (r, b, u) | %.3f ms | %.3f ms |\n| single call `transit_poly` | %.0f ns | %.0f ns |\n",
             t1 / 1e6, t2 / 1e6, t6 / 1e6, t5 / 1e6, t3, t4))
out("\nLimbdark.jl re-uses its work structure across samples (`transit_poly!(t)`); `transit_poly(r, b, u)`\nallocates it per call. Pizzetti.jl's gradient for the quadratic family uses closed forms\n(`flux_gradient_quadratic`).\n")
out(@sprintf("\nLargest difference of the light curves: %.1e (the interior series, proven below 1e-9).\n",
             maximum(abs.(r .- r2))))

write(joinpath(@__DIR__, "results.md"), String(take!(OUT)))
