@testset "limb: contact expansions and proven pieces" begin
    laws = (Power2(0.6, 0.6), Nonlinear(0.5, 0.1, 0.2, -0.1), SquareRoot(0.3, 0.4), Power2(0.7, 1.7),
            PowerSum([0.0, 0.3, 1.1], [0.4, 0.4, 0.2]))
    for law in laws, p in (0.0092, 0.05, 0.1, 0.2)
        zc = series_cut(p, law)
        z = vcat(range(zc + 1e-9, 1 + p - 1e-12, length = 400), [1 - p, 1 - p + 1e-13, 1 - p - 1e-13, 1 + p - 1e-13])
        oq = Occultor(p, law; limb = :quadrature)
        of = Occultor(p, law; limb = :fast)
        @test of.usecontact
        fq = oq.(z)
        ff = of.(z)
        @test maximum(abs.(ff .- fq)) < 2e-15
        @test all(m -> m in (:contact, :quadrature, :series), method.(Ref(of), z))
        # proven: every certified sample within the tolerance
        for τ in (1e-9, 1e-12)
            f, cert = occult_certified(z, p, law; series_tol = τ)
            @test maximum(abs.(f .- fq)) <= τ
            p <= 0.15 && @test all(cert)
        end
    end
    # :auto switches to the contact expansions for long light curves only
    law = Power2(0.6, 0.6)
    @test !Occultor(0.1, law; nlimb = 10).usecontact
    @test Occultor(0.1, law; nlimb = 1000).usecontact
    @test !Occultor(0.6, law; limb = :fast).usecontact          # large occulters: quadrature
    # certification report
    f, cert = occult_certified(collect(range(0, 1.3, length = 1001)), 0.25, law; series_tol = 1e-12)
    @test 0 < count(cert) < 1001
    @test all(cert[1:100])                                       # interior series
    @test_throws ArgumentError Occultor(0.1, law; limb = :magic)
end

@testset "interval enclosures (IntervalArithmetic extension)" begin
    setprecision(BigFloat, 200) do
        for law in (Power2(0.6, 0.6), Nonlinear(0.5, 0.1, 0.2, -0.1)), p in (0.02, 0.1), z in (0.0, 0.3, 0.8)
            I = occult_interval(z, p, law)
            lawb = law isa Power2 ? Power2(big"0.6", big"0.6") : Nonlinear(big"0.5", big"0.1", big"0.2", big"-0.1")
            ref = occult(big(z), big(p), lawb; series_tol = 0)
            @test inf(I) <= ref <= sup(I)
            @test diam(I) < 1e-9
        end
    end
    @test_throws DomainError occult_interval(0.95, 0.1, Power2(0.6, 0.6))
end
