# # Photodynamics: NbodyGradient + Pizzetti
#
# A two-planet system near the 2:1 resonance shows transit-timing variations (TTVs).
# NbodyGradient.jl (Agol, Hernandez & Langford 2021) integrates the N-body problem and
# returns, for every transit, the mid-transit time t_c, the sky velocity v and the impact
# parameter b, together with their derivatives with respect to the initial orbital elements
# and masses. Around each transit the sky path is a straight line,
#     z(t) = sqrt(b² + v² (t - t_c)²) / R⋆,
# and Pizzetti computes the flux and its exact derivative ∂F/∂z (paper, Sec. 5.4). The chain
# rule then gives the derivative of every light-curve sample with respect to every initial
# condition: the gradient of a photodynamical likelihood.
#
# Run with:  julia --project=. 03_photodynamics.jl   (writes figures/)

using Pizzetti, NbodyGradient, ForwardDiff, Plots, Printf, LinearAlgebra
ENV["GKSwstype"] = "100"
mkpath(joinpath(@__DIR__, "figures"))

const RSUN_AU = 0.00465047
const Rstar = 1.0 * RSUN_AU
law = Power2(0.6, 0.6)
radius_ratio = [0.03, 0.06]                         # planets b and c

# ## Initial conditions: star, a sub-Neptune and a Saturn-mass planet near 2:1
#           m        P       t0    ecosω   esinω   I            Ω
elements = [1.0      0.0     0.0   0.0     0.0     0.0          0.0;
            3.0e-5   10.0    2.0   0.01    0.005   π/2 - 0.010  0.0;
            3.0e-4   20.55   5.0   0.02   -0.01    π/2 - 0.005  0.0]
tmax = 400.0

function transits(elements; grad = true)
    ic = ElementsIC(0.0, 3, elements)
    s = State(ic)
    tp = TransitParameters(tmax, ic)
    Integrator(ahl21!, 10.0 / 40, tmax)(s, tp; grad = grad)
    return tp
end
tp = transits(elements)

# ## The light curve: 2-min cadence within ±0.3 d of every transit
function windows(tp; half = 0.3, cadence = 2 / 1440)
    ts = Float64[]
    for j in 2:3, k in 1:tp.count[j]
        tc = tp.ttbv[1, j, k]
        0 < tc < tmax || continue
        append!(ts, tc-half:cadence:tc+half)
    end
    sort!(unique!(ts))
end
t = windows(tp)
println(length(t), " light-curve samples, ", tp.count[2], " + ", tp.count[3], " transits")

# For sample time ti and planet j: the transit k nearest in time, and z.
function nearest(tp, j, ti)
    best, kb = Inf, 0
    for k in 1:tp.count[j]
        d = abs(ti - tp.ttbv[1, j, k])
        d < best && ((best, kb) = (d, k))
    end
    kb
end
zsky(tc, v, b2, ti) = sqrt(b2 + v^2 * (ti - tc)^2) / Rstar

function lightcurve(tp, t, ps, law)
    F = ones(length(t))
    for (i, ti) in enumerate(t), j in 2:3
        k = nearest(tp, j, ti)
        z = zsky(tp.ttbv[1, j, k], tp.ttbv[2, j, k], tp.ttbv[3, j, k], ti)
        F[i] += occult(z, ps[j-1], law) - 1
    end
    F
end
F = lightcurve(tp, t, radius_ratio, law)

# ## Gradient of every sample with respect to every initial element (chain rule)
# dF/d(element) = ∂F/∂z · [∂z/∂t_c, ∂z/∂v, ∂z/∂b²] · d(t_c, v, b²)/d(element)
function lightcurve_jacobian(tp, t, ps, law)
    J = zeros(length(t), 7, 3)                      # sample × element × body
    for (i, ti) in enumerate(t), j in 2:3
        k = nearest(tp, j, ti)
        tc, v, b2 = tp.ttbv[1, j, k], tp.ttbv[2, j, k], tp.ttbv[3, j, k]
        z = zsky(tc, v, b2, ti)
        z >= 1 + ps[j-1] && continue
        _, dFdz, _, _ = flux_gradient(z, ps[j-1], law)
        dz = ForwardDiff.gradient(q -> zsky(q[1], q[2], q[3], ti), [tc, v, b2])
        for l in 1:7, body in 1:3
            J[i, l, body] += dFdz * (dz[1] * tp.dtbvdelements[1, j, k, l, body] +
                                     dz[2] * tp.dtbvdelements[2, j, k, l, body] +
                                     dz[3] * tp.dtbvdelements[3, j, k, l, body])
        end
    end
    J
end
tJ = @elapsed J = lightcurve_jacobian(tp, t, radius_ratio, law)
@printf("Jacobian of %d samples x 21 initial conditions: %.2f s\n", length(t), tJ)

# ## Check against finite differences of the whole pipeline (N-body + light curve)
# The columns of the elements matrix are (m, P, t0, ecosω, esinω, I, Ω); NbodyGradient's
# derivative slots are (P, t0, ecosω, esinω, I, Ω, m).
slot(col) = col == 1 ? 7 : col - 1
function fd_column(col, body; h)
    e1 = copy(elements); e1[body, col] += h
    e2 = copy(elements); e2[body, col] -= h
    F1 = lightcurve(transits(e1; grad = false), t, radius_ratio, law)
    F2 = lightcurve(transits(e2; grad = false), t, radius_ratio, law)
    (F1 .- F2) ./ 2h
end
println("\nelement  body   max|dF/dx| (chain rule)   max|chain rule - finite diff.| / max|dF/dx|")
for (name, col, body, h) in (("m", 1, 3, 1e-8), ("P", 2, 3, 1e-6), ("t0", 3, 2, 1e-6), ("ecosω", 4, 3, 1e-6),
                             ("I", 6, 2, 1e-7))
    fd = fd_column(col, body; h = h)
    an = J[:, slot(col), body]
    @printf("%-8s %4d   %14.4e              %10.1e\n", name, body, maximum(abs, an), maximum(abs, an .- fd) / maximum(abs, an))
end

# The finite differences straddle the contact points, where the flux is not smooth
# (it has a δ^(γ+3/2) term at second contact), which limits their agreement to ~1e-5.

# ## Figures: TTVs and one transit with its sensitivity to the mass of planet c
plt1 = plot(xlabel = "transit epoch", ylabel = "TTV (minutes)", size = (700, 350), legend = :topleft)
for (j, name) in ((2, "planet b"), (3, "planet c"))
    n = tp.count[j]
    tc = [tp.ttbv[1, j, k] for k in 1:n if 0 < tp.ttbv[1, j, k] < tmax]
    e = 0:length(tc)-1
    lin = hcat(ones(length(e)), e) \ tc
    scatter!(plt1, e, 1440 .* (tc .- (lin[1] .+ lin[2] .* e)), label = name, ms = 4)
end
savefig(plt1, joinpath(@__DIR__, "figures", "03_ttv.png"))

k = 8
tcb = tp.ttbv[1, 2, k]
sel = abs.(t .- tcb) .< 0.3
plt2 = plot(layout = (2, 1), size = (700, 550), link = :x)
plot!(plt2[1], 24 .* (t[sel] .- tcb), 1e6 .* (F[sel] .- 1), lw = 2, label = "planet b, transit $k",
      ylabel = "flux - 1 (ppm)")
plot!(plt2[2], 24 .* (t[sel] .- tcb), 1e6 .* J[sel, slot(1), 3] .* 1e-5, lw = 2, color = :darkred,
      label = "ΔF for Δm_c = 1e-5 M⊙ (chain rule)", xlabel = "hours from mid-transit", ylabel = "ppm")
savefig(plt2, joinpath(@__DIR__, "figures", "03_transit_sensitivity.png"))
println("figures written to ", joinpath(@__DIR__, "figures"))
