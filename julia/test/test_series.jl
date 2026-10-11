@testset "interior series (Theorems 1, 2, Corollary 1)" begin
    setprecision(BigFloat, 256) do
        # table equals the direct double sum of Theorem 1 (exact arithmetic in BigFloat)
        for γ in (big"0.25", big"0.5", big"0.3", big"1.37")
            A = series_table(γ, 11)
            for (z, p) in ((big"0.3", big"0.05"), (big"0.7", big"0.1"))
                w = 1 - z^2
                v = 1 / w
                S = w^γ * sum(sum(A[j+1, q+1] * p^(2q) for q in 0:11) * v^j for j in 0:11)
                Sd = w^γ * sum(Pizzetti.series_beta(γ, m) * v^m *
                               sum(Pizzetti._Gml(BigFloat, m, l) * z^l * p^(2m - l) for l in 0:2:m) for m in 0:11)
                @test abs(S - Sd) < 1e-60
            end
        end
    end
    # Theorem 2: the bound holds and is close to the true error (BigFloat quadrature as truth)
    setprecision(BigFloat, 160) do
        for γ in (0.25, 0.5, 0.3, 0.75, 1.37), p in (0.02, 0.1, 0.2), zf in (0.0, 0.3, 0.6, 0.85)
            z = zf * (1 - p)
            e = (2z * p + p^2) / (1 - z^2)
            e < 0.9 || continue
            D = blocked_flux(big(z), big(p), big(γ); series_tol = 0) / (big(π) * big(p)^2)
            A = series_table(big(γ), 11)
            w = 1 - big(z)^2
            S = w^big(γ) * sum(sum(A[j+1, q+1] * big(p)^(2q) for q in 0:11) * (1 / w)^j for j in 0:11)
            B = series_bound(big(z), big(p), big(γ), 11)
            @test abs(S - D) <= B + 1e-40          # reference accurate to ~1e-45
            # one-sided: all omitted terms share the sign of β_{M+1}
            @test sign(D - S) == sign(Pizzetti.series_beta(big(γ), 12)) || abs(S - D) < 1e-40
        end
    end
    # bound increases with z; the cut meets the tolerance
    for law in (Power2(0.6, 0.6), Nonlinear(0.5, 0.1, 0.2, -0.1), Quadratic(0.4, 0.25)), p in (0.0092, 0.1, 0.3)
        zc = series_cut(p, law; series_tol = 1e-9)
        @test zc > 0
        α, c = power_terms(law)
        N = sum(c ./ (α .+ 2))
        fb(z) = Pizzetti.series_flux_bound(z, p, α ./ 2, abs.(c), p^2 / (2abs(N)))
        zs = range(0, zc, length = 50)
        @test issorted(fb.(zs))
        @test fb(zc) <= 1e-9
        @test fb(min(zc + 1e-6, 1 - p - 1e-9)) > 1e-9 || zc > 1 - p - 1e-6
    end
    @test series_cut(0.1, Power2(0.6, 0.6); series_tol = 0) == -1
    @test series_bound(0.3, 0.1, 1.0) == 0          # integer exponent: series terminates
end
