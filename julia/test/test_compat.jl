@testset "Limbdark.jl interface" begin
    u = [0.4, 0.25]
    for r in (0.01, 0.1, 0.5), b in (0.0, 0.3, 0.95, 1.05)
        @test transit_poly(r, b, u) ≈ quadratic_flux(b, r, u[1], u[2]) atol = 1e-9
    end
    u3 = [0.3, 0.2, 0.1, 0.05]
    for r in (0.01, 0.1, 0.3), b in (0.0, 0.5, 0.9, 1.0, 1.1)
        b < 1 + r || continue
        f = transit_poly(r, b, u3)
        α, c = power_terms(Polynomial(u3))
        fq = 1 - blocked_quadrature(b, r, α ./ 2, c) / (2π * sum(c ./ (α .+ 2)))
        @test f ≈ fq atol = 1e-9
        g = zeros(length(u3) + 2)
        @test transit_poly!(r, b, u3, g) ≈ f
        gf = ForwardDiff.gradient(v -> transit_poly(v[1], v[2], v[3:end]), [r, b, u3...])
        @test g ≈ gf atol = 1e-9
    end
end

@testset "quadratic law: closed-form gradient (transit_poly!)" begin
    u = [0.4, 0.25]
    J = coefficient_jacobian(Quadratic(0.4, 0.25))
    setprecision(BigFloat, 128) do
        for r in (0.0092, 0.1, 0.5), b in (0.0, 0.3, 0.6, 0.95, 1.0, 1.04)
            b < 1 + r || continue
            for bb in (b, r, 1 - r + 1e-9, 1 - r - 1e-9, 1 + r - 1e-9)    # incl. near the contacts
                g = zeros(4)
                F = transit_poly!(r, bb, u, g)
                Fb, dz, dp, dc = flux_gradient(big(bb), big(r), Quadratic(big(0.4), big(0.25)))
                ref = Float64.([dp; dz; J' * dc])
                @test F ≈ Float64(quadratic_flux(big(bb), big(r), big(0.4), big(0.25))) atol = 1e-13   # exact closed form
                @test g ≈ ref atol = 1e-13
            end
        end
    end
    # agrees with ForwardDiff through the general code
    g = zeros(4)
    transit_poly!(0.1, 0.5, u, g)
    gf = ForwardDiff.gradient(v -> transit_poly(v[1], v[2], v[3:end]), [0.1, 0.5, u...])
    @test g ≈ gf atol = 1e-12
end
