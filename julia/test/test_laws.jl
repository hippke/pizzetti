@testset "laws" begin
    μs = range(0, 1, length = 11)
    laws = [Uniform(), Linear(0.5), Quadratic(0.4, 0.25), SquareRoot(0.3, 0.4), Nonlinear(0.5, 0.1, 0.2, -0.1),
            Power2(0.6, 0.6), Polynomial([0.3, 0.2, 0.1]), PowerSum([0.0, 0.7, 1.3], [0.5, 0.3, 0.2])]
    direct = [μ -> 1.0, μ -> 1 - 0.5(1 - μ), μ -> 1 - 0.4(1 - μ) - 0.25(1 - μ)^2,
              μ -> 1 - 0.3(1 - μ) - 0.4(1 - sqrt(μ)),
              μ -> 1 - 0.5(1 - μ^0.5) - 0.1(1 - μ) - 0.2(1 - μ^1.5) + 0.1(1 - μ^2),
              μ -> 1 - 0.6(1 - μ^0.6), μ -> 1 - 0.3(1 - μ) - 0.2(1 - μ)^2 - 0.1(1 - μ)^3,
              μ -> 0.5 + 0.3μ^0.7 + 0.2μ^1.3]
    for (L, f) in zip(laws, direct)
        @test all(intensity(L, μ) ≈ f(μ) for μ in μs)
    end
    @test limb_darkening("power2", [0.6, 0.6]) == Power2(0.6, 0.6)
    @test limb_darkening("nonlinear", [1, 2, 3, 4]) isa Nonlinear
    @test limb_darkening(:uniform, Float64[]) == Uniform()
    @test_throws ArgumentError limb_darkening("logarithmic", [0.1, 0.2])
    @test_throws ArgumentError limb_darkening("quadratic", [0.1])
    @test_throws ArgumentError Power2(0.5, -1.0)
    @test_throws DimensionMismatch PowerSum([0.0, 1.0], [1.0])
    @test Pizzetti.quadratic_coefficients(Polynomial([0.3, 0.2])) == (0.3, 0.2)
    @test Pizzetti.quadratic_coefficients(Polynomial([0.3, 0.2, 0.1])) === nothing
end
