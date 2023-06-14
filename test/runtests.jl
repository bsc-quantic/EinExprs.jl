using Test
using EinExprs

@testset "Unit tests" verbose = true begin
    include("EinExpr_test.jl")
    include("Counters_test.jl")
    @testset "Optimizers" verbose = true begin
        include("Exhaustive_test.jl")
    end
end

@testset "Integration tests" verbose = true begin
    include("ext/ChainRulesCore_test.jl")
    include("ext/Makie_test.jl")
end