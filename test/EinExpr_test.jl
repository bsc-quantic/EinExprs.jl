@testset "EinExpr" begin
    using Tensors
    using EinExprs: suminds

    @testset "identity" begin
        tensor = Tensor(rand(2, 3), (:i, :j))
        expr = EinExpr([tensor])

        @test expr.head == labels(tensor)
        @test expr.args == [tensor]

        @test labels(expr) == labels(tensor)
        @test ndims(expr) == 2

        @test size(expr, :i) == 2
        @test size(expr, :j) == 3
        @test size(expr) == (2, 3)

        @test isempty(suminds(expr))
        @test isempty(suminds(expr, parallel=true))
    end

    @testset "transpose" begin
        tensor = Tensor(rand(2, 3), (:i, :j))
        expr = EinExpr([tensor], (:j, :i))

        @test expr.head == reverse(labels(tensor))
        @test expr.args == [tensor]

        @test labels(expr) == reverse(labels(tensor))
        @test ndims(expr) == 2

        @test size(expr, :i) == 2
        @test size(expr, :j) == 3
        @test size(expr) == (3, 2)

        @test isempty(suminds(expr))
        @test isempty(suminds(expr, parallel=true))
    end

    @testset "axis sum" begin
        tensor = Tensor(rand(2, 3), (:i, :j))
        expr = EinExpr([tensor], (:i,))

        @test expr.head == (:i,)
        @test expr.args == [tensor]

        @test labels(expr) == (:i,)
        @test labels(expr, all=true) == (:i, :j)

        @test size(expr, :i) == 2
        @test size(expr, :j) == 3
        @test size(expr) == (2,)

        @test suminds(expr) == [:j]
        @test isempty(suminds(expr, parallel=true))
    end

    @testset "diagonal" begin
        tensor = Tensor(rand(2, 2), (:i, :i))
        expr = EinExpr([tensor], (:i,))

        @test expr.head == (:i,)
        @test expr.args == [tensor]

        @test labels(expr) == (:i,)
        @test labels(expr, all=true) == labels(expr)

        @test size(expr, :i) == 2
        @test size(expr) == (2,)

        @test isempty(suminds(expr))
        @test isempty(suminds(expr, parallel=true))
    end

    @testset "trace" begin
        tensor = Tensor(rand(2, 2), (:i, :i))
        expr = EinExpr([tensor], ())

        @test isempty(expr.head)
        @test expr.args == [tensor]

        @test isempty(labels(expr))
        @test labels(expr, all=true) == (:i,)

        @test size(expr, :i) == 2
        @test size(expr) == ()

        @test suminds(expr) == [:i]
        @test isempty(suminds(expr, parallel=true))
    end

    @testset "outer product" begin
        tensors = [
            Tensor(rand(2, 3), (:i, :j)),
            Tensor(rand(4, 5), (:k, :l)),
        ]
        expr = EinExpr(tensors)

        @test expr.head == (:i, :j, :k, :l,)
        @test expr.args == tensors

        @test labels(expr) == Tuple(mapreduce(collect ∘ labels, vcat, tensors))
        @test labels(expr, all=true) == labels(expr)
        @test ndims(expr) == 4

        for (i, d) in zip([:i, :j, :k, :l], [2, 3, 4, 5])
            @test size(expr, i) == d
        end
        @test size(expr) == (2, 3, 4, 5)

        @test isempty(suminds(expr))
        @test isempty(suminds(expr, parallel=true))
    end

    @testset "inner product" begin
        @testset "Vector" begin
            tensors = [
                Tensor(rand(2), (:i,)),
                Tensor(rand(2), (:i,)),
            ]
            expr = EinExpr(tensors)

            @test isempty(expr.head)
            @test expr.args == tensors

            @test isempty(labels(expr))
            @test labels(expr, all=true) == (:i,)
            @test ndims(expr) == 0

            @test size(expr, :i) == 2
            @test size(expr) == ()

            @test suminds(expr) == [:i]
            @test suminds(expr, parallel=true) == [[:i]]
        end
        @testset "Matrix" begin
            tensors = [
                Tensor(rand(2, 3), (:i, :j)),
                Tensor(rand(2, 3), (:i, :j)),
            ]
            expr = EinExpr(tensors)

            @test isempty(expr.head)
            @test expr.args == tensors

            @test isempty(labels(expr))
            @test labels(expr, all=true) == (:i, :j)
            @test ndims(expr) == 0

            @test size(expr, :i) == 2
            @test size(expr, :j) == 3
            @test size(expr) == ()

            @test suminds(expr) == [:i, :j]
            @test Set(Set.(suminds(expr, parallel=true))) == Set([Set([:i, :j])])
        end
    end

    @testset "matrix multiplication" begin
        tensors = [
            Tensor(rand(2, 3), (:i, :k)),
            Tensor(rand(3, 4), (:k, :j)),
        ]
        expr = EinExpr(tensors)

        @test expr.head == (:i, :j)
        @test expr.args == tensors

        @test labels(expr) == (:i, :j)
        @test labels(expr, all=true) == (:i, :k, :j)
        @test ndims(expr) == 2

        @test size(expr, :i) == 2
        @test size(expr, :j) == 4
        @test size(expr, :k) == 3
        @test size(expr) == (2, 4)

        @test suminds(expr) == [:k]
        @test suminds(expr, parallel=true) == [[:k]]
    end
end