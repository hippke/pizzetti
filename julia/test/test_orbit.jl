@testset "orbits and light curves (cross-check with the Python package)" begin
    d = JSON.parsefile(joinpath(TESTDATA, "python_lightcurves.json"))
    for cs in d["cases"]
        P = cs["params"]
        prm = TransitParams(t0 = P["t0"], per = P["per"], rp = P["rp"], a = P["a"], inc = P["inc"], ecc = P["ecc"],
                            w = P["w"], u = Float64.(P["u"]), limb_dark = P["limb_dark"])
        t = Float64.(cs["t"])
        z = rsky(t, prm)
        zp = Float64.(cs["z"])
        @test maximum(abs.(z .- zp)) < 1e-12 * P["a"]
        fp = Float64.(cs["flux"])
        # same method (series off, quadrature limb); the Python quadrature is accurate to
        # ~1e-10 for rp = 0.3 (Julia's Gauss-Jacobi version to ~1e-16)
        f0 = light_curve(t, prm; series_tol = 0, limb = :quadrature)
        @test maximum(abs.(f0 .- fp)) < (P["rp"] > 0.2 ? 1e-10 : P["rp"] > 0.05 ? 1e-12 : 1e-14)
        # default settings: within the series tolerance
        @test maximum(abs.(light_curve(t, prm) .- fp)) < 1e-9
    end
    # Kepler's equation
    for e in (0.0, 0.1, 0.5, 0.9, 0.99), M in range(-7, 7, length = 29)
        E = eccentric_anomaly(M, e)
        @test E - e * sin(E) ≈ M atol = 1e-13
    end
    # impact parameter round trip
    prm = TransitParams(per = 3.0, rp = 0.1, a = 10.0, b = 0.4, ecc = 0.2, w = 60.0)
    @test impact_parameter(prm) ≈ 0.4
    # exposure-time integration converges to the exact average
    prm = TransitParams(per = 3.5, rp = 0.1, a = 9.0, b = 0.3, law = Power2(0.6, 0.6))
    t = collect(range(-0.08, 0.08, length = 81))
    dt = 0.02
    ref = [quadgk(s -> light_curve([s], prm)[1], ti - dt / 2, ti + dt / 2; rtol = 1e-13)[1] / dt for ti in t]
    f_g = light_curve(t, prm; exp_time = dt, supersample = 31, integration = :gauss)
    f_m = light_curve(t, prm; exp_time = dt, supersample = 31, integration = :midpoint)
    @test maximum(abs.(f_g .- ref)) < 1e-6
    @test maximum(abs.(f_m .- ref)) < 1e-6
    # ForwardDiff through the orbit
    g = ForwardDiff.gradient(v -> sum(light_curve(t, TransitParams(per = 3.5, rp = v[1], a = v[2], b = v[3],
                                                                 law = Power2(0.6, 0.6)))), [0.1, 9.0, 0.3])
    h = 1e-6
    s(v) = sum(light_curve(t, TransitParams(per = 3.5, rp = v[1], a = v[2], b = v[3], law = Power2(0.6, 0.6))))
    fd = [(s([0.1 + h, 9.0, 0.3]) - s([0.1 - h, 9.0, 0.3])) / 2h, (s([0.1, 9.0 + h, 0.3]) - s([0.1, 9.0 - h, 0.3])) / 2h,
          (s([0.1, 9.0, 0.3 + h]) - s([0.1, 9.0, 0.3 - h])) / 2h]
    @test g ≈ fd rtol = 1e-5
end
