@testset "EinExpr" begin
    using Tensors

    @testset "identity" begin
        tensor = Tensor(rand(2, 3), (:i, :j))
        expr = EinExpr([tensor])

        @test expr.head == collect(labels(tensor))
        @test expr.args == [tensor]

        @test labels(expr) == collect(labels(tensor))
        @test ndims(expr) == 2

        @test size(expr, :i) == 2
        @test size(expr, :j) == 3
        @test size(expr) == (2, 3)
    end

    @testset "transpose" begin
        tensor = Tensor(rand(2, 3), (:i, :j))
        expr = EinExpr([tensor], [:j, :i])

        @test expr.head == collect(reverse(labels(tensor)))
        @test expr.args == [tensor]

        @test labels(expr) == collect(reverse(labels(tensor)))
        @test ndims(expr) == 2

        @test size(expr, :i) == 2
        @test size(expr, :j) == 3
        @test size(expr) == (3, 2)
    end

    @testset "axis sum" begin
        tensor = Tensor(rand(2, 3), (:i, :j))
        expr = EinExpr([tensor], (:i,))

        @test expr.head == [:i]
        @test expr.args == [tensor]

        @test labels(expr) == [:i]
        @test labels(expr, all=true) == [:i, :j]

        @test size(expr, :i) == 2
        @test size(expr, :j) == 3
        @test size(expr) == (2,)
    end

    @testset "diagonal" begin
        tensor = Tensor(rand(2, 2), (:i, :i))
        expr = EinExpr([tensor], (:i,))

        @test expr.head == [:i]
        @test expr.args == [tensor]

        @test labels(expr) == [:i]
        @test labels(expr, all=true) == labels(expr)

        @test size(expr, :i) == 2
        @test size(expr) == (2,)
    end

    @testset "trace" begin
        tensor = Tensor(rand(2, 2), (:i, :i))
        expr = EinExpr([tensor], ())

        @test isempty(expr.head)
        @test expr.args == [tensor]

        @test isempty(labels(expr))
        @test labels(expr, all=true) == [:i]

        @test size(expr, :i) == 2
        @test size(expr) == ()
    end

    @testset "outer product" begin
        tensors = [
            Tensor(rand(2, 3), (:i, :j)),
            Tensor(rand(4, 5), (:k, :l)),
        ]
        expr = EinExpr(tensors)

        @test expr.head == [:i, :j, :k, :l]
        @test expr.args == tensors

        @test labels(expr) == mapreduce(collect ∘ labels, vcat, tensors)
        @test labels(expr, all=true) == labels(expr)
        @test ndims(expr) == 4

        for (i, d) in zip([:i, :j, :k, :l], [2, 3, 4, 5])
            @test size(expr, i) == d
        end
        @test size(expr) == (2, 3, 4, 5)
    end

    @testset "inner product" begin
        tensors = [
            Tensor(rand(2), (:i,)),
            Tensor(rand(2), (:i,)),
        ]
        expr = EinExpr(tensors)

        @test isempty(expr.head)
        @test expr.args == tensors

        @test isempty(labels(expr))
        @test labels(expr, all=true) == [:i]
        @test ndims(expr) == 0

        @test size(expr, :i) == 2
        @test size(expr) == ()
    end

    @testset "matrix multiplication" begin
        tensors = [
            Tensor(rand(2, 3), (:i, :k)),
            Tensor(rand(3, 4), (:k, :j)),
        ]
        expr = EinExpr(tensors)

        @test expr.head == [:i, :j]
        @test expr.args == tensors

        @test labels(expr) == [:i, :j]
        @test labels(expr, all=true) == [:i, :k, :j]
        @test ndims(expr) == 2

        @test size(expr, :i) == 2
        @test size(expr, :j) == 4
        @test size(expr, :k) == 3
        @test size(expr) == (2, 4)
    end
end