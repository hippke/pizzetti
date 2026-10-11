@testset "fast kernels" begin
    # branch-free log, exp, pow (SIMD interior kernel of the power-2 law)
    x = vcat(10 .^ range(-12, 0, length = 20001), rand(Xoshiro(1), 1000))
    @test maximum(abs(Pizzetti._vlog(xi) - log(big(xi))) for xi in x) < 4e-15
    @test maximum(abs(Pizzetti._vexp(y) / exp(big(y)) - 1) for y in range(-700, 1, length = 20001)) < 4e-16
    @test maximum(abs(Pizzetti._vpow(xi, 0.37) / big(xi)^big"0.37" - 1) for xi in x) < 4e-15
    # sin(2π u) in turns
    @test maximum(abs(Pizzetti._sin_turns(u) - sinpi(2 * big(u))) for u in range(-30, 30, length = 20001)) < 4e-16
    # vectorised evaluate! equals the scalar evaluation for every law and sample
    for law in (Quadratic(0.4, 0.25), Power2(0.6, 0.6), Nonlinear(0.5, 0.1, 0.2, -0.1), SquareRoot(0.3, 0.4),
                PowerSum([0.0, 0.3, 1.1], [0.4, 0.4, 0.2]), Polynomial([0.3, 0.2, 0.1, 0.05])), p in (0.02, 0.1, 0.4)
        z = collect(range(0, 1 + p + 0.05, length = 3001))
        for limb in (:auto, :fast, :quadrature)
            o = Occultor(p, law; limb = limb)
            out = similar(z)
            evaluate!(out, o, z)
            @test maximum(abs.(out .- o.(z))) < 4e-16
        end
    end
    # circular-orbit rsky (SIMD) against the scalar version
    P = TransitParams(per = 3.7, rp = 0.1, a = 11.0, b = 0.3, w = 37.0, law = Quadratic(0.4, 0.25))
    t = collect(range(-10, 10, length = 20001))
    zv = rsky(t, P)
    zs = [rsky(ti, P) for ti in t]
    both = (zv .< 50) .& (zs .< 50)
    @test maximum(abs.(zv[both] .- zs[both])) < 1e-12
    # the in-front/behind decision may differ only at quadrature (z = a), far from transit
    @test all(abs.(min.(zv, zs)[.!both] .- P.a) .< 1e-6 .|| (zv[.!both] .== zs[.!both]))
    # cached hypergeometric functions
    for (a, b, c) in ((-0.3, 0.5, 1.0), (-0.3, 1.5, 2.0), (-0.5, 0.5, 1.0), (-1.0, 1.5, 2.0)), x in (0.0, 0.3, 0.5, 0.51, 0.9, 1.0)
        @test Pizzetti.HypCache(a, b, c)(x) ≈ hyp2f1_unit(a, b, c, x) rtol = 1e-14
    end
    # closed forms of the integer exponents used with the contact expansions
    for p in (0.02, 0.1, 0.3), z in range(0, 1 + p, length = 50)[2:end-1]
        P0, P1 = Pizzetti._P01(z, p)
        @test P0 ≈ blocked_quadrature(z, p, [0.0], [1.0]) atol = 1e-15
        @test P1 ≈ blocked_quadrature(z, p, [1.0], [1.0]) atol = 1e-15
    end
    # :auto: contact expansions when enough limb samples per j-term
    law = Power2(0.6, 0.6)
    @test !Occultor(0.02, law; nlimb = 100).usecontact
    @test Occultor(0.02, law; nlimb = 400).usecontact
    @test !Occultor(0.1, law; nlimb = 400).usecontact
    @test Occultor(0.1, law; nlimb = 1000).usecontact
    # scalar fast path of the quadratic family
    for p in (0.05, 0.5, 1.5), z in (0.0, 0.3, 0.9, 1.2)
        @test occult(z, p, Quadratic(0.4, 0.25)) == Occultor(p, Quadratic(0.4, 0.25); series_tol = 0)(z) ||
              abs(occult(z, p, Quadratic(0.4, 0.25)) - quadratic_flux(z, p, 0.4, 0.25)) < 1e-15
    end
end
