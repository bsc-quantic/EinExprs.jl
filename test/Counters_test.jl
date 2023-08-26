@testset "Counters" begin
    @testset "identity" begin
        tensor = EinExpr((:i, :j), Dict(:i => 2, :j => 3))
        expr = EinExpr((:i, :j), [tensor])

        @test flops(expr) == 0
        @test removedsize(expr) == 0
    end

    @testset "transpose" begin
        tensor = EinExpr((:i, :j), Dict(:i => 2, :j => 3))
        expr = EinExpr([:j, :i], [tensor])

        @test flops(expr) == 0
        @test removedsize(expr) == 0
    end

    @testset "axis sum" begin
        tensor = EinExpr((:i, :j), Dict(:i => 2, :j => 3))
        expr = EinExpr((:i,), [tensor])

        @test flops(expr) == 6
        @test removedsize(expr) == 4
    end

    @testset "diagonal" begin
        tensor = EinExpr((:i, :i), Dict(:i => 2))
        expr = EinExpr((:i,), [tensor])

        @test flops(expr) == 0
        @test removedsize(expr) == 2
    end

    @testset "trace" begin
        tensor = EinExpr((:i, :i), Dict(:i => 2))
        expr = EinExpr(Symbol[], [tensor])

        @test flops(expr) == 2
        @test removedsize(expr) == 3
    end

    @testset "outer product" begin
        tensors = [EinExpr((:i, :j), Dict(:i => 2, :j => 3)), EinExpr((:k, :l), Dict(:k => 4, :l => 5))]
        expr = EinExpr((:i, :j, :k, :l), tensors)

        @test flops(expr) == prod(2:5)
        @test removedsize(expr) == -94
    end

    @testset "inner product" begin
        tensors = [EinExpr((:i,), Dict(:i => 2)), EinExpr((:i,), Dict(:i => 2))]
        expr = EinExpr(Symbol[], tensors)

        @test flops(expr) == 2
        @test removedsize(expr) == 3
    end

    @testset "matrix multiplication" begin
        tensors = [EinExpr((:i, :k), Dict(:i => 2, :k => 3)), EinExpr((:k, :j), Dict(:k => 3, :j => 4))]
        expr = EinExpr((:i, :j), tensors)

        @test flops(expr) == 2 * 3 * 4
        @test removedsize(expr) == 10
    end
end
