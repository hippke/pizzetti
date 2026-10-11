@testset "quadratic family (exact solution)" begin
    # against the Green's-theorem quadrature and BigFloat
    for p in (0.0092, 0.1, 0.5, 1.3), z in vcat(range(0, 1 + p, length = 41)[1:end-1], [abs(1 - p), p, 1.0])
        z < 1 + p || continue
        for (u1, u2) in ((0.4, 0.25), (0.0, 0.0), (0.6, 0.0), (1.0, -0.2))
            fq = quadratic_flux(z, p, u1, u2)
            α, c = power_terms(Quadratic(u1, u2))
            fg = 1 - blocked_quadrature(z, p, α ./ 2, c) / (2π * sum(c ./ (α .+ 2)))
            @test fq ≈ fg atol = 2e-14
        end
    end
    setprecision(BigFloat, 200) do
        for (z, p) in ((0.98, 0.02), (0.3, 0.1), (1.05, 0.1), (0.1, 0.1), (0.5, 0.9))
            ref = quadratic_flux(big(z), big(p), big"0.4", big"0.25")
            α, c = power_terms(Quadratic(big"0.4", big"0.25"))
            refq = 1 - blocked_quadrature(big(z), big(p), α ./ 2, c) / (2big(π) * sum(c ./ (α .+ 2)))
            @test abs(ref - refq) < 1e-40
            # the closed form loses ~1e-13 exactly at internal tangency; occult() switches there
            z == 1 - p || @test abs(quadratic_flux(z, p, 0.4, 0.25) - ref) < 2e-15
            @test abs(occult(z, p, Quadratic(0.4, 0.25)) - ref) < 2e-15
        end
    end
    # occult with the quadratic law uses series + exact solution
    z = range(0, 1.2, length = 2001)
    f = occult(z, 0.1, Quadratic(0.4, 0.25))
    fr = [quadratic_flux(zi, 0.1, 0.4, 0.25) for zi in z]
    @test maximum(abs.(f .- fr)) < 1e-9
    @test occult(2.0, 0.1, Quadratic(0.4, 0.25)) == 1
    @test occult(0.1, 1.5, Quadratic(0.4, 0.25)) == 0
    @test occult(0.3, 0.0, Quadratic(0.4, 0.25)) == 1
end
