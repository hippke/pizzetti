# # J6 · The limb integral in Bartky's variable
#
# Level: advanced · Prerequisites: J1, J3 · Runtime: ~2 min (mostly compilation)
#
# Covers: the paper's Section 5.3 step by step, each claim checked numerically. The limb
# integral as a complete integral in the rim angle ϑ; Bartky's variable β with tan ϑ = sinh β;
# why the trapezoid rule on fixed nodes converges geometrically and uniformly up to both
# contacts (a strip of analyticity of fixed width, plus a pole removed by its residue); the
# shifted contour inside the disk; the connection formula 𝒦 = 𝒦_reg + y^ν 𝒦_sing that makes
# Chebyshev interpolants possible; a generic implementation that gives BigFloat references for
# free; and the cost of the package's optimised rules.
#
# Run with:  julia --project=. J6_bartky.jl   (from julia/examples; writes figures/)

using Pizzetti, QuadGK, Printf, Plots
ENV["GKSwstype"] = "100"
mkpath(joinpath(@__DIR__, "figures"))

# ## 1. From the rim integral to a complete integral
# On the limb (1 - p < z < 1 + p), Green's theorem reduces the blocked integral
# P_α = ∫ μ^α dA over the occulted area to a line integral along the occulter's rim inside the
# star (paper, Eq. P1d, with s = r²). With x = 1 - (z - p)², k = x / (4zp), y = 1 - k and the
# rim angle ϑ defined by 1 - s = x cos²ϑ, it becomes a complete integral of Legendre type,
#
#     P_α = -2√k x^(α/2+1) / (α+2) · 𝒦(y; x),
#     𝒦(y; x) = ∫_0^{π/2} cos^(α+3)ϑ / √(1 - k sin²ϑ) · [1 + (p² - z²) / (1 - x cos²ϑ)] dϑ,
#
# and Bartky's substitution tan ϑ = sinh β maps it onto the half line,
#
#     𝒦(y; x) = ∫_0^∞ cosh^-(α+3)β / √(1 + y sinh²β) · [1 + (p² - z²) cosh²β / (cosh²β - x)] dβ.
#
# We write 𝒦(u; w) as a function of two arguments; along the light curve u = y and w = x.

geom(z, p) = (x = 1 - (z - p)^2; k = x / (4z * p); (x = x, k = k, y = 1 - k, c0 = p^2 - z^2))
prefactor(g, α) = -2sqrt(g.k) * g.x^(α / 2 + 1) / (α + 2)

"𝒦 in the rim angle ϑ, by adaptive quadrature (the textbook form)."
K_theta(k, x, c0, α; rtol = 1e-14) =
    quadgk(ϑ -> cos(ϑ)^(α + 3) / sqrt(1 - k * sin(ϑ)^2) * (1 + c0 / (1 - x * cos(ϑ)^2)), 0, π / 2; rtol = rtol)[1]

# The trapezoid rule in β, written generically: it runs in any number type T. The integrand
# has simple poles where cosh²β = w, at β = ±i·acos√w; their contribution to the trapezoid
# error is a geometric series of residues, known in closed form and subtracted (`pole = true`).
function K_bartky(u, w, c0, α; h = 0.2, tail = 1e-16, pole = true)
    T = float(promote_type(typeof(u), typeof(w), typeof(c0), typeof(h)))
    f(β) = cosh(β)^(-(α + 3)) / sqrt(1 + u * sinh(β)^2) * (1 + c0 * cosh(β)^2 / (cosh(β)^2 - w))
    N = ceil(Int, (log(1 / T(tail)) + (α + 3) * log(T(2))) / ((α + 3) * h)) + 1   # cosh^-(α+3) < tail
    s = f(zero(T)) / 2
    for n in 1:N
        s += f(n * h)
    end
    s *= h
    d = 1 - u * (1 - w)
    if pole && 0 < w < 1 && d > 0
        a = acos(sqrt(w))
        e = exp(-2T(π) * a / h)
        s -= T(π) * c0 * sqrt(w) / sqrt((1 - w) * d) * e / (1 - e) * w^(-(α + 3) / 2)
    end
    return s
end

p, α = 0.1, 0.6
println("P_α at p = $p, α = $α: rim angle ϑ (quadgk) | Bartky trapezoid | Green's-theorem quadrature")
for z in (0.9 + 1e-9, 0.95, 1.0, 1.05, 1.1 - 1e-9)
    g = geom(z, p)
    P_th = prefactor(g, α) * K_theta(g.k, g.x, g.c0, α)
    P_tr = prefactor(g, α) * K_bartky(g.y, g.x, g.c0, α)
    P_q = blocked_flux(z, p, α / 2; limb = :quadrature)
    @printf("z = %-14.10g %.15e  %.15e  %.15e\n", z, P_th, P_tr, P_q)
end

# ## 2. Why a trapezoid rule with fixed nodes works: a strip of fixed width
# The trapezoid rule on β_n = n h converges like exp(-2πd/h) for an integrand that is analytic
# in the strip |Im β| < d. The branch points of (1 + y sinh²β)^(-1/2) lie where
# sinh²β = -1/y; for 0 < y < 1 this is on the line Im β = π/2, for *every* modulus. The factor
# cosh^-(α+3)β is singular on the same line. So d = π/2 all the way to both contacts, and the
# rule converges like exp(-π²/h), uniformly in z. Only the two simple poles cosh²β = x lie
# closer to the axis, and their effect is removed exactly by the residue term.

function integrand_map(z, p, α)
    g = geom(z, p)
    f(β) = cosh(β)^(-(α + 3)) / sqrt(1 + g.y * sinh(β)^2) * (1 + g.c0 * cosh(β)^2 / (cosh(β)^2 - g.x))
    re = range(0, 5, length = 401)
    im_ = range(0, 1.9, length = 191)
    L = [log10(abs(f(complex(r, i))) + 1e-300) for i in im_, r in re]
    heatmap(re, im_, clamp.(L, -4, 4); c = :grays, xlabel = "Re β", ylabel = "Im β", clims = (-4, 4),
            title = @sprintf("z = %.4f  (y = %.0e)", z, g.y), colorbar_title = "log₁₀|integrand|")
    hline!([π / 2], color = :black, ls = :dash, label = "Im β = π/2")
    scatter!([0.0], [acos(sqrt(g.x))], color = :white, msc = :black, label = "pole cosh²β = x")
    scatter!([acosh(1 / sqrt(g.y))], [π / 2], color = :black, msc = :white, marker = :diamond,
             label = "branch point 1 + y sinh²β = 0")
end
# (On Im β = π/2, sinh²β = -cosh²(Re β), so the branch point sits at Re β = acosh(1/√y): it
# moves outwards as y → 0, but never leaves the line.)
plt = plot(integrand_map(1.0, p, α), integrand_map(0.9 + 1e-3, p, α), layout = (1, 2), size = (1150, 430),
           left_margin = 6Plots.mm, bottom_margin = 6Plots.mm, right_margin = 4Plots.mm)
savefig(plt, joinpath(@__DIR__, "figures", "J6_strip.png"))

# Convergence against the step h, with a BigFloat reference from the same generic code
# (h = 0.1, 256-bit arithmetic: the discretisation error exp(-π²/h) is 1e-43):
cases = (("mid-limb", 1.0), ("1e-12 from second contact", 0.9 + 1e-12), ("1e-4 from first contact", 1.1 - 1e-4))
ref = Dict(name => setprecision(BigFloat, 256) do
               g = geom(big(z), big(p))
               prefactor(g, big(α)) * K_bartky(g.y, g.x, g.c0, big(α); h = big"0.1", tail = 1e-45)
           end for (name, z) in cases)
Ph(z; kw...) = (g = geom(z, p); prefactor(g, α) * K_bartky(g.y, g.x, g.c0, α; kw...))
hs = 0.1:0.025:0.7
plt = plot(xlabel = "trapezoid step h", ylabel = "absolute error of P_α", yscale = :log10, ylims = (1e-19, 1e-1),
           legend = :bottomright, size = (700, 420))
for (j, (name, z)) in enumerate(cases)
    r = Float64(ref[name])
    err = [max(abs(Ph(z; h = h) - r), 1e-19) for h in hs]
    err0 = [max(abs(Ph(z; h = h, pole = false) - r), 1e-19) for h in hs]
    plot!(plt, hs, err, lw = 2, color = :black, ls = (:solid, :dash, :dot)[j], label = name)
    j == 1 && plot!(plt, hs, err0, lw = 1, color = :gray, label = "mid-limb, without the pole term")
end
plot!(plt, hs, 0.01 .* exp.(-π^2 ./ hs), color = :black, lw = 1, ls = :dashdot, label = "∝ exp(-π²/h)")
savefig(plt, joinpath(@__DIR__, "figures", "J6_convergence.png"))
for (name, z) in cases
    @printf("h = 0.2, %-26s P_α = %.6e, error %.1e\n", name, Float64(ref[name]), abs(Ph(z) - Float64(ref[name])))
end
# (Errors are in units of the blocked flux, which is what enters the light curve. Towards first
# contact the poles cosh²β = x approach the line Im β = π/2, so the *relative* error of 𝒦 grows,
# but P_α vanishes like k^ν there and the flux error stays at rounding level.)

# For comparison: Gauss–Legendre in ϑ with the same number of nodes converges fast in the
# middle of the limb but slowly near second contact, where 1/√(1 - k sin²ϑ) develops a
# logarithmic singularity at ϑ = π/2 (k → 1). Bartky's variable stretches exactly that region.
println("\nGauss–Legendre in ϑ, n nodes: error of P_α")
for n in (20, 40, 80)
    xg, wg = gauss(n, 0, π / 2)
    row = map(cases) do (name, z)
        g = geom(z, p)
        v = prefactor(g, α) * sum(wg[i] * cos(xg[i])^(α + 3) / sqrt(1 - g.k * sin(xg[i])^2) * (1 + g.c0 / (1 - g.x * cos(xg[i])^2))
                for i in 1:n)
        @sprintf("%-26s %.1e", name, abs(v - Float64(ref[name])))
    end
    println("n = $n:  ", join(row, "   "))
end
g = geom(1.0, p)
@printf("the Bartky rule at h = 0.2 uses %d nodes\n",
        ceil(Int, (log(1e16) + (α + 3) * log(2)) / ((α + 3) * 0.2)) + 1)

# ## 3. Inside the disk: a shifted contour
# For an occulter inside the disk near the limb (z_c < z < 1 - p), k > 1 and y < 0. The
# flux is Re 𝒦(y - i0), and the branch point sinh β = (-y)^(-1/2) now lies on the real
# integration path, so the rule on the real axis fails. Moving the path to β + iγ (γ ≈ 0.5)
# restores a strip of width ≈ γ around it; the package does this for every band sample.
z = 0.88
g = geom(z, p)
@printf("\nz = %.2f: y = %.3f, branch point at β = %.3f on the real axis\n", z, g.y, asinh(1 / sqrt(-g.y)))
fβ(β) = cosh(β)^(-(α + 3)) / sqrt(complex(1 + g.y * sinh(β)^2)) * (1 + g.c0 * cosh(β)^2 / (cosh(β)^2 - g.x))
naive = prefactor(g, α) * real(0.2 * (fβ(0.0) / 2 + sum(fβ(0.2n) for n in 1:60)))     # real axis, h = 0.2
q = blocked_flux(z, p, α / 2; series_tol = 0, limb = :quadrature)
oc = Occultor(p, PowerSum([α], [1.0]); limb = :trapezoid)        # band samples use the contour
tr = (1 - oc(z)) * π / (α / 2 + 1)
@printf("P_α: real-axis rule %.12f | shifted contour %.16f | quadrature %.16f\n", naive, tr, q)

# ## 4. The connection formula: what makes interpolation possible
# Along a light curve at fixed p, z (and so x and p² - z²) is an analytic function of y near
# second contact, and 𝒦 is a function of y alone. It has the structure
#
#     𝒦(y) = 𝒦_reg(y) + y^ν 𝒦_sing(y),    ν = (α + 3)/2,
#
# with 𝒦_reg, 𝒦_sing analytic at y = 0, so 𝒦 itself is only ⌊ν⌋ times differentiable there.
# The singular part is the same integral at the complementary modulus,
#
#     𝒦_sing(y) = -𝒦(k; -x y / k) / (k^ν sin πν),
#
# so both parts are computed with the trapezoid rule, and 𝒦_reg = 𝒦 - y^ν 𝒦_sing. Inside the
# disk (y < 0), Re 𝒦 = 𝒦_reg + cos(πν) |y|^ν 𝒦_sing with the same two functions.
zofk(k, p) = p * (1 - 2k) + sqrt(p^2 * (1 - 2k)^2 + 1 - p^2)        # inverse of k(z)
ν = (α + 3) / 2
function parts(y; p = p, α = α)
    k = 1 - y
    g = geom(zofk(k, p), p)
    K = blocked_flux(zofk(k, p), p, α / 2; series_tol = 0, limb = :quadrature) / prefactor(g, α)
    Ks = -K_bartky(k, -g.x * y / k, g.c0, α) / (k^ν * sin(π * ν))
    Kr = y >= 0 ? K - y^ν * Ks : K - cos(π * ν) * abs(y)^ν * Ks
    return K, Kr, Ks
end
ys = range(-0.2, 0.4, length = 241)
P = parts.(ys)
p1 = plot(ys, getindex.(P, 1), lw = 2, color = :black, label = "K", xlabel = "y = 1 - k  (second contact at y = 0)")
plot!(p1, ys, getindex.(P, 2), lw = 2, color = :black, ls = :dash, label = "K_reg")
plot!(p1, ys, -0.5 .* getindex.(P, 3), lw = 2, color = :gray, label = "-K_sing / 2")
vline!(p1, [0.0], color = :gray, ls = :dot, label = "")

# Chebyshev coefficients on y ∈ [-0.2, 0.4]: 𝒦 converges only algebraically (the y^ν
# singularity), its two parts geometrically: ~1e-13 at degree 18 even on this wide interval.
# (The package splits the limb into three shorter pieces and reaches 1e-15 at degree 14-18.)
function cheb(f, a, b, n)
    θ = [π * (j + 0.5) / n for j in 0:n-1]
    v = [f((a + b) / 2 + (b - a) / 2 * cos(t)) for t in θ]
    [2 / n * sum(v[j] * cos(m * θ[j]) for j in 1:n) for m in 0:n-1]
end
n = 26
C = [cheb(y -> parts(y)[i], -0.2, 0.4, n) for i in 1:3]
p2 = plot(xlabel = "Chebyshev degree", ylabel = "|coefficient|", yscale = :log10, ylims = (1e-18, 1))
for (i, (lab, ls)) in enumerate((("K", :solid), ("K_reg", :dash), ("K_sing", :dot)))
    scatter!(p2, 0:n-1, max.(abs.(C[i]), 1e-18), color = :black, markershape = (:circle, :rect, :utriangle)[i],
             ms = 3, label = lab)
end
savefig(plot(p1, p2, layout = (1, 2), size = (1100, 430), left_margin = 6Plots.mm, bottom_margin = 6Plots.mm), joinpath(@__DIR__, "figures", "J6_connection.png"))
@printf("\ncoefficient of degree 18:  𝒦 %.1e   𝒦_reg %.1e   𝒦_sing %.1e\n", abs(C[1][19]), abs(C[2][19]), abs(C[3][19]))

# ## 5. BigFloat for free
# Because `K_bartky` is generic, a 40-digit limb flux needs only a smaller step and a longer
# tail: the error is exp(-π²/h), so h = 0.1 gives about 43 digits.
setprecision(BigFloat, 200) do
    zb, pb, αb = big"0.97", big"0.1", big"0.6"
    g = geom(zb, pb)
    Pb = prefactor(g, αb) * K_bartky(g.y, g.x, g.c0, αb; h = big"0.1", tail = 1e-45)
    Pb2 = prefactor(g, αb) * K_bartky(g.y, g.x, g.c0, αb; h = big"0.08", tail = 1e-50)
    println("\nP_α(z = 0.97) to 40 digits: ", string(Pb)[1:42])
    @printf("change with h = 0.08: %.1e;  Float64 package value differs by %.1e\n",
            Float64(abs(Pb2 - Pb)), Float64(blocked_flux(0.97, 0.1, 0.3; limb = :trapezoid) - Pb))
end

# ## 6. Cost
# The package folds the exponent into the weights w_n cosh^-(α+3)β_n once per light curve, so
# a sample costs one square root and one division per node, shared by all exponents of a law.
# The interpolants of Section 4 cost ~40 µs to build and then a few dozen ns per sample.
using BenchmarkTools
law = Power2(0.6, 0.6)
zc = series_cut(p, law)
zl = collect(range(zc + 1e-9, 1 + p - 1e-9, length = 20_000))      # limb samples only
out = similar(zl)
println("\nper limb sample (power-2 law, p = $p):")
for lm in (:trapezoid, :interpolate, :quadrature)
    o = Occultor(p, law; limb = lm, nlimb = length(zl))
    t = @belapsed evaluate!($out, $o, $zl)
    @printf("  %-12s %6.0f ns\n", lm, 1e9 * t / length(zl))
end
g = geom(1.0, p)
t = @belapsed K_bartky($(g.y), $(g.x), $(g.c0), $α)
@printf("  the generic K_bartky above: %.0f ns per call (pow and cosh at every node)\n", 1e9t)

# ## What you learned
# - the limb integral is a complete integral; in Bartky's variable its integrand is analytic in
#   a strip of fixed width π/2 for every modulus, so a trapezoid rule on fixed nodes converges
#   like exp(-π²/h) up to both contacts, once the poles' residue term is subtracted;
# - inside the disk the same rule runs on a shifted contour;
# - the singular part at second contact is the same integral at the complementary modulus, so
#   𝒦_reg and 𝒦_sing are smooth, and Chebyshev interpolants of modest degree reach 1e-15;
# - written generically, the rule gives arbitrary-precision references in a few lines.
#
# See the paper, Section 5.3 and Figure 1, for where this rule sits among the methods.
