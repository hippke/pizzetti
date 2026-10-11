@testset "hypergeometric functions" begin
    # hyp2f1_unit against HypergeometricFunctions.jl, incl. the logarithmic case
    for (a, b, c) in ((0.5, 0.5, 2.3), (-0.3, 0.5, 1.0), (-1.37, 1.5, 2.0), (0.5, 1.5, 2.5 + 0.3),
                      (0.5, 0.5, 2.0), (-0.5, 0.5, 1.0), (0.5, 1.5, 3.0)), x in (0.0, 0.1, 0.5, 0.7, 0.95, 0.999)
        @test hyp2f1_unit(a, b, c, x) ≈ _₂F₁(a, b, c, x) rtol = 1e-13
    end
    # at x = 1: Gauss's sum
    for (a, b, c) in ((0.5, 0.5, 2.3), (-0.3, 0.5, 1.0))
        g = exp(Pizzetti.loggamma(c) + Pizzetti.loggamma(c - a - b) - Pizzetti.loggamma(c - a) - Pizzetti.loggamma(c - b))
        @test hyp2f1_unit(a, b, c, 1.0) ≈ g rtol = 1e-13
    end
    # the tabulated families of the limb solution
    γ = 0.3
    tab = Pizzetti.lens_table(γ, 5)
    for j in 0:5, x in (0.2, 0.5, 0.8, 0.99)
        H = _₂F₁(0.5, 0.5, γ + 2.5 + j, x) * Pizzetti._beta(0.5, γ + 2 + j)
        G = _₂F₁(-(γ + 1 + j), 0.5, 1.0, x) * π
        for (fam, ref) in ((1, H), (2, G))
            if x <= 0.5
                v = Pizzetti._horner(tab.T[fam, j+1, 1, :], tab.L[fam, j+1, 1], x)
            else
                y = 1 - x
                m = γ + 1.5 + j
                v = Pizzetti._horner(tab.T[fam, j+1, 2, :], tab.L[fam, j+1, 2], y) +
                    y^m * (log(y) * Pizzetti._horner(tab.T[fam, j+1, 3, :], tab.L[fam, j+1, 3], y) +
                           Pizzetti._horner(tab.T[fam, j+1, 4, :], tab.L[fam, j+1, 4], y))
            end
            @test v ≈ ref rtol = 1e-13
        end
    end
end
