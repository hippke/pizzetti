using Test
using Pizzetti
using ForwardDiff
using IntervalArithmetic
using JSON
using QuadGK
using HypergeometricFunctions: _₂F₁
using Aqua
using Random: Xoshiro

const TESTDATA = joinpath(@__DIR__, "..", "..", "testdata")

@testset "Pizzetti.jl" begin
    include("test_laws.jl")
    include("test_series.jl")
    include("test_reference.jl")
    include("test_quadratic.jl")
    include("test_hypergeometric.jl")
    include("test_limb.jl")
    include("test_derivatives.jl")
    include("test_generic.jl")
    include("test_orbit.jl")
    include("test_compat.jl")
    include("test_kernels.jl")
    include("test_aqua.jl")
end
