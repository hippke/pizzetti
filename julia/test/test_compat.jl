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
