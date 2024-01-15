@testset "Counters" begin
    using EinExprs: removedrank

    sizedict = Dict(:i => 2, :j => 3, :k => 4, :l => 5)

    @testset "identity" begin
        tensor = EinExpr([:i, :j])
        expr = EinExpr([:i, :j], [tensor])

        @test flops(expr, sizedict) == 0
        @test removedsize(expr, sizedict) == 0
        @test removedrank(expr, sizedict) == 0
    end

    @testset "transpose" begin
        tensor = EinExpr([:i, :j])
        expr = EinExpr([:j, :i], [tensor])

        @test flops(expr, sizedict) == 0
        @test removedsize(expr, sizedict) == 0
        @test removedrank(expr, sizedict) == 0
    end

    @testset "axis sum" begin
        tensor = EinExpr([:i, :j])
        expr = EinExpr([:i], [tensor])

        @test flops(expr, sizedict) == 6
        @test removedsize(expr, sizedict) == 4
        @test removedrank(expr, sizedict) == 1
    end

    @testset "diagonal" begin
        tensor = EinExpr([:i, :i])
        expr = EinExpr([:i], [tensor])

        @test flops(expr, sizedict) == 0
        @test removedsize(expr, sizedict) == 2
        @test removedrank(expr, sizedict) == 1
    end

    @testset "trace" begin
        tensor = EinExpr([:i, :i])
        expr = EinExpr(Symbol[], [tensor])

        @test flops(expr, sizedict) == 2
        @test removedsize(expr, sizedict) == 3
        @test removedrank(expr, sizedict) == 2
    end

    @testset "outer product" begin
        tensors = [EinExpr([:i, :j]), EinExpr([:k, :l])]
        expr = EinExpr([:i, :j, :k, :l], tensors)

        @test flops(expr, sizedict) == prod(2:5)
        @test removedsize(expr, sizedict) == -94
        @test removedrank(expr, sizedict) == -2
    end

    @testset "inner product" begin
        tensors = [EinExpr([:i]), EinExpr([:i])]
        expr = EinExpr(Symbol[], tensors)

        @test flops(expr, sizedict) == 2
        @test removedsize(expr, sizedict) == 3
        @test removedrank(expr, sizedict) == 1
    end

    @testset "matrix multiplication" begin
        tensors = [EinExpr([:i, :j]), EinExpr([:j, :k])]
        expr = EinExpr([:i, :k], tensors)

        @test flops(expr, sizedict) == 2 * 3 * 4
        @test removedsize(expr, sizedict) == 10
        @test removedrank(expr, sizedict) == 0
    end
end
