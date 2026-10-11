@testset "generic number types" begin
    setprecision(BigFloat, 128) do
        o = Occultor(big"0.1", Power2(big"0.6", big"0.6"); series_tol = 1e-30, M = 41)
        @test Occultor(big"0.1", Power2(big"0.6", big"0.6"); series_tol = 1e-30).zc == -1    # M = 11 cannot reach 1e-30
        @test_throws ArgumentError Occultor(0.1, Power2(0.6, 0.6); M = 10)
        @test o(big"0.3") isa BigFloat
        @test o(big"1.05") isa BigFloat
        # the series cut adapts to the tolerance
        @test o.zc > 0
        @test abs(o(big"0.3") - occult(big"0.3", big"0.1", Power2(big"0.6", big"0.6"); series_tol = 0)) < 1e-30
    end
    @test occult(0.3f0, 0.1, Power2(0.6, 0.6)) isa Float64
    @test Occultor(0.1, Quadratic(0.4, 0.25)) isa Occultor{Float64}
    # broadcasting and in-place
    z = collect(range(0, 1.2, length = 100))
    out = similar(z)
    occult!(out, z, 0.1, Power2(0.6, 0.6))
    @test out == occult(z, 0.1, Power2(0.6, 0.6))
    @test size(occult(reshape(z, 10, 10), 0.1, Power2(0.6, 0.6))) == (10, 10)
end
