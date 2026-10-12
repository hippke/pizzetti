# The two likelihoods of the Python inference example (examples/inference, paper Section 6.4)
# in Julia: cost of one evaluation (orbit, light curve, χ²) at posterior samples, and of
# its gradient with respect to all seven parameters (ForwardDiff). Every call uses a new
# posterior sample, as in a fit.  Run from julia/benchmark:  julia --project=. -O3 inference_cases.jl

using Pizzetti, ForwardDiff, JSON, Printf, Statistics

const D = JSON.parsefile(joinpath(@__DIR__, "..", "..", "examples", "inference", "results", "posterior_samples.json"))

function make_time(c)
    t1 = collect(-c["halfwindow"]:c["cadence"]:(c["halfwindow"] - 1e-12))     # numpy arange
    vec([c["t0"] + e * c["per"] + dt for dt in t1, e in 0:c["ntransits"]-1])
end

# sampling parameters -> law (Kipping 2013 q1, q2 for the quadratic law; c, α for power-2)
function law_of(kind, l1, l2)
    if kind == "quadratic"
        s = sqrt(l1)
        return Quadratic(2s * l2, s * (1 - 2l2))
    end
    return Power2(l1, l2)
end

function model(t, kind, θ)
    t0, per, rp, a, b, l1, l2 = θ
    light_curve(t, TransitParams(t0 = t0, per = per, rp = rp, a = a, b = b, law = law_of(kind, l1, l2)))
end

function main()
    println("Julia ", VERSION, ", single thread\n")
    println("| case | samples | likelihood | gradient (7 parameters) | gradient / likelihood |")
    println("|---|---|---|---|---|")
    for case in ("tess", "kepler")
        c = D[case]
        kind = c["law"]
        t = make_time(c)
        @assert length(t) == c["n_points"]
        y = model(t, kind, Float64.(c["theta_true"]))
        ivar = 1 / c["sigma"]^2
        loglike(θ) = -0.5 * ivar * sum(abs2, y .- model(t, kind, θ))
        S = [Float64.(s) for s in c["samples"]]
        g = similar(S[1])
        cfg = ForwardDiff.GradientConfig(loglike, S[1], ForwardDiff.Chunk{7}())
        loglike(S[1]); ForwardDiff.gradient!(g, loglike, S[1], cfg)          # compile
        tl = Float64[]; tg = Float64[]
        for rep in 1:5
            push!(tl, @elapsed(for s in S; loglike(s); end) / length(S))
            push!(tg, @elapsed(for s in S; ForwardDiff.gradient!(g, loglike, s, cfg); end) / length(S))
        end
        @printf("| %s | %d | %.3f ms | %.2f ms | %.1f |\n", case, length(t), 1e3 * median(tl), 1e3 * median(tg),
                median(tg) / median(tl))
    end
end

main()
