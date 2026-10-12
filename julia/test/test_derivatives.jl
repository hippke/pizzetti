@testset "derivatives (Section 5.4), ForwardDiff" begin
    laws = (Quadratic(0.4, 0.25), Power2(0.6, 0.6), Nonlinear(0.5, 0.1, 0.2, -0.1), Polynomial([0.3, 0.2, 0.1]))
    for law in laws, p in (0.02, 0.1, 0.3), zf in (0.0, 0.3, 0.7, 0.95, 1.0, 1.03, 1.1)
        z = zf < 1 ? zf * (1 - p) : 1 - p + (zf - 1) / 0.1 * 2p     # interior points, then across the limb
        z < 1 + p || continue
        F, dz, dp, dc = flux_gradient(z, p, law)
        # central finite differences of the quadrature (no series), step 1e-6
        α, c = power_terms(law)
        h = 1e-6
        Fq(zz, pp, cc) = 1 - blocked_quadrature(zz, pp, α ./ 2, cc) / (2π * sum(cc ./ (α .+ 2)))
        if z > 2h && abs(z - (1 - p)) > 2h && abs(z - p) > 2h
            @test dz ≈ (Fq(z + h, p, c) - Fq(z - h, p, c)) / 2h atol = 2e-8
        end
        # (finite differences are not accurate across the kink at internal tangency)
        abs(z - (1 - p)) > 2h && @test dp ≈ (Fq(z, p + h, c) - Fq(z, p - h, c)) / 2h atol = 2e-8
        for i in eachindex(c)
            e = zeros(length(c)); e[i] = h
            @test dc[i] ≈ (Fq(z, p, c .+ e) - Fq(z, p, c .- e)) / 2h atol = 2e-8
        end
        # ForwardDiff through the whole code (series, exact solution, quadrature)
        gz = ForwardDiff.derivative(x -> occult(x, p, law), z)
        gp = ForwardDiff.derivative(x -> occult(z, x, law), p)
        z > 0 && @test gz ≈ dz atol = 1e-12
        @test gp ≈ dp atol = 1e-12
    end
    # law parameters, including the power-2 exponent (ForwardDiff only)
    z, p = 0.5, 0.1
    F, dz, dp, du = flux_gradient_law(z, p, Quadratic(0.4, 0.25))
    g = ForwardDiff.gradient(v -> occult(z, p, Quadratic(v[1], v[2])), [0.4, 0.25])
    @test du ≈ g rtol = 1e-10
    gα = ForwardDiff.derivative(a -> occult(z, p, Power2(0.6, a)), 0.6)
    fd = (occult(z, p, Power2(0.6, 0.6 + 1e-6)) - occult(z, p, Power2(0.6, 0.6 - 1e-6))) / 2e-6
    @test gα ≈ fd rtol = 1e-6
    setprecision(BigFloat, 256) do
        for zz in (0.9, 0.95, 1.05)       # limb: against BigFloat finite differences
            gα = ForwardDiff.derivative(a -> occult(zz, p, Power2(0.6, a)), 0.6)
            hb = big"1e-15"
            Fb(a) = occult(big(zz), big(p), Power2(big"0.6", a); series_tol = 0)
            @test gα ≈ (Fb(big"0.6" + hb) - Fb(big"0.6" - hb)) / 2hb rtol = 1e-10
        end
    end
    # exactly at the contacts the derivatives stay exact (analytic, Section 5.4)
    for zz in (1 - p, p, 1 + p - 1e-12)
        F, dz, dp, dc = flux_gradient(zz, p, Quadratic(0.4, 0.25))
        g = ForwardDiff.gradient(v -> occult(v[1], v[2], Quadratic(0.4, 0.25)), [zz, p])
        @test g ≈ [dz, dp] atol = 1e-13
    end
end

@testset "derivatives: elementary forms (γ = 0, 1) and cached Gauss functions" begin
    setprecision(BigFloat, 160) do
        for p in (0.0092, 0.1, 0.3, 0.7, 1.3)
            zs = vcat(range(0, 1 + p, length = 12)[1:end-1],
                      [abs(1 - p) + 1e-9, 1 - p - 1e-9, 1 + p - 1e-9, p + 1e-9, abs(p - 1) + 1e-12])
            for z in zs
                (p >= 1 && z <= p - 1) && continue
                r = Pizzetti._bderiv01(z, p)
                a = blocked_derivatives(big(z), big(p), big(0))
                b = blocked_derivatives(big(z), big(p), big(1))
                @test collect(r) ≈ Float64.([a[1], a[2], b[1], b[2]]) atol = 1e-15 * 2π * p
                p < 1 || continue
                for (i, g) in enumerate((0.0, 0.5, 1.0))
                    c = blocked_derivatives(big(z), big(p), big(g))
                    @test collect(Pizzetti._bderiv(Pizzetti._qfam()[i], z, p)) ≈ Float64.(collect(c)) atol = 1e-14 * 2π * p
                end
            end
        end
    end
end
