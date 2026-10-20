@testset "limb: fixed-node rule and interpolants (Section 5.3)" begin
    laws = (Power2(0.6, 0.6), Nonlinear(0.5, 0.1, 0.2, -0.1), SquareRoot(0.3, 0.4), Power2(0.7, 1.7), Power2(0.6, 1.0),
            Power2(0.6, 0.05), Power2(0.6, 3.7), Power2(0.6, 8.0), PowerSum([0.0, 0.3, 1.1], [0.4, 0.4, 0.2]))
    for law in laws, p in (0.001, 0.0092, 0.05, 0.1, 0.15, 0.2, 0.29), tol in (1e-9, 0.0)
        d = (10.0 .^ range(-14, -1, length = 30)) .* p
        z = vcat(collect(range(0, 1 + 1.2p, length = 1201)), 1 - p .+ d, 1 - p .- d, 1 + p .- d)
        fq = occult(z, p, law; series_tol = tol, limb = :quadrature)
        for m in (:trapezoid, :interpolate, :auto)
            @test maximum(abs.(occult(z, p, law; series_tol = tol, limb = m) .- fq)) < 3e-15
        end
        # the scalar path agrees with the batched evaluate!
        o = Occultor(p, law; series_tol = tol, limb = :interpolate)
        @test maximum(abs.(o.(z) .- occult(z, p, law; series_tol = tol, limb = :interpolate))) < 1e-15
    end
    o = Occultor(0.1, Power2(0.6, 0.6); limb = :interpolate)
    @test o.usetrap && o.trap.interp
    @test method(o, 0.95) == :interpolate && method(o, 0.5) == :series
    @test method(Occultor(0.1, Power2(0.6, 0.6); limb = :trapezoid), 0.95) == :trapezoid
    @test !Occultor(0.1, Power2(0.6, 0.999); limb = :interpolate).trap.interp   # near odd α: direct rule
    # reference vectors (independent tanh-sinh quadrature)
    cases = JSON.parsefile(joinpath(TESTDATA, "blocked_reference.json"))["double"]
    for cs in cases
        p, g = cs["p"], cs["gamma"]
        (g == 0 || p >= Pizzetti.TRAP_PMAX) && continue
        unit = π / (g + 1)
        for (z, P) in zip(cs["z"], cs["P"])
            for m in (:trapezoid, :interpolate)
                f = occult([z], p, Power2(1.0, 2g); series_tol = 0, limb = m)[1]
                @test abs((1 - f) * unit - P) <= 3e-15 * unit
            end
        end
    end
end
