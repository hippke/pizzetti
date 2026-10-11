# Independent references (pizzetti/testdata): tanh-sinh quadrature of the area integral
@testset "independent reference (area integral)" begin
    d = JSON.parsefile(joinpath(TESTDATA, "blocked_reference.json"))
    worst = Dict(:series => 0.0, :quadrature => 0.0, :fast => 0.0, :proven => 0.0)
    for cs in d["double"], (z, P) in zip(cs["z"], cs["P"])
        p = cs["p"]; g = cs["gamma"]
        unit = π / (g + 1)                         # flux of the whole star for this term
        for limb in (:quadrature, :fast, :proven)
            e = abs(blocked_flux(z, p, g; series_tol = 0, limb = limb) - P) / unit
            worst[limb] = max(worst[limb], e)
        end
        # with the interior series: within its proven tolerance
        e = abs(blocked_flux(z, p, g; series_tol = 1e-9) - P) / unit
        worst[:series] = max(worst[:series], e)
    end
    @test worst[:quadrature] < 3e-15
    @test worst[:fast] < 3e-15
    @test worst[:proven] < 3e-15
    @test worst[:series] < 1e-9 * 1.0001
    # 30-digit reference: BigFloat quadrature
    setprecision(BigFloat, 160) do
        for cs in d["mp30"], (z, P) in zip(cs["z"], cs["P"])
            Pj = blocked_flux(big(z), big(cs["p"]), big(cs["gamma"]); series_tol = 0)
            @test abs(Pj - parse(BigFloat, P)) < 1e-28
        end
    end
end
