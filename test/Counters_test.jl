@testset "Counters" begin
    @testset "identity" begin
        tensor = Tensor(rand(2, 3), (:i, :j))
        expr = EinExpr((:i, :j), [tensor])

        @test flops(expr) == 0
        @test removedsize(expr) == 0
    end

    @testset "transpose" begin
        tensor = Tensor(rand(2, 3), (:i, :j))
        expr = EinExpr([:j, :i], [tensor])

        @test flops(expr) == 0
        @test removedsize(expr) == 0
    end

    @testset "axis sum" begin
        tensor = Tensor(rand(2, 3), (:i, :j))
        expr = EinExpr((:i,), [tensor])

        @test flops(expr) == 6
        @test removedsize(expr) == 4
    end

    @testset "diagonal" begin
        tensor = Tensor(rand(2, 2), (:i, :i))
        expr = EinExpr((:i,), [tensor])

        @test flops(expr) == 0
        @test removedsize(expr) == 2
    end

    @testset "trace" begin
        tensor = Tensor(rand(2, 2), (:i, :i))
        expr = EinExpr((), [tensor])

        @test flops(expr) == 2
        @test removedsize(expr) == 3
    end

    @testset "outer product" begin
        tensors = [Tensor(rand(2, 3), (:i, :j)), Tensor(rand(4, 5), (:k, :l))]
        expr = EinExpr((:i, :j, :k, :l), tensors)

        @test flops(expr) == prod(2:5)
        @test removedsize(expr) == -94
    end

    @testset "inner product" begin
        tensors = [Tensor(rand(2), (:i,)), Tensor(rand(2), (:i,))]
        expr = EinExpr((), tensors)

        @test flops(expr) == 2
        @test removedsize(expr) == 3
    end

    @testset "matrix multiplication" begin
        tensors = [Tensor(rand(2, 3), (:i, :k)), Tensor(rand(3, 4), (:k, :j))]
        expr = EinExpr((:i, :j), tensors)

        @test flops(expr) == 2 * 3 * 4
        @test removedsize(expr) == 10
    end
end
