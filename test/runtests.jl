using Test
using Tenet
using EinExprs
using Compat

@testset "Unit tests" verbose = true begin
    include("EinExpr_test.jl")
    include("SizedEinExpr_test.jl")
    include("Counters_test.jl")
    @testset "Optimizers" begin
        include("Naive_test.jl")
        include("Exhaustive_test.jl")
        include("Greedy_test.jl")
        include("LineGraph_test.jl")
        include("HyPar_test.jl")
    end
    include("Slicing_test.jl")
end

@testset "Integration tests" verbose = true begin
    include("ext/Makie_test.jl")
end
