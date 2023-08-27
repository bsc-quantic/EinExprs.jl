@testset "EinExpr" begin
    using LinearAlgebra

    @testset "identity" begin
        tensor = EinExpr([:i, :j], Dict(:i => 2, :j => 3))
        expr = EinExpr((:i, :j), [tensor])

        @test expr.head == head(tensor)
        @test expr.args == [tensor]

        @test head(expr) == head(tensor)
        @test ndims(expr) == 2

        @test size(expr, :i) == 2
        @test size(expr, :j) == 3
        @test size(expr) == (2, 3)

        @test isempty(suminds(expr))
        @test isempty(parsuminds(expr))
    end

    @testset "transpose" begin
        tensor = EinExpr([:i, :j], Dict(:i => 2, :j => 3))
        expr = EinExpr((:j, :i), [tensor])

        @test expr.head == reverse(inds(tensor))
        @test expr.args == [tensor]

        @test head(expr) == reverse(inds(tensor))
        @test ndims(expr) == 2

        @test size(expr, :i) == 2
        @test size(expr, :j) == 3
        @test size(expr) == (3, 2)

        @test isempty(suminds(expr))
        @test isempty(parsuminds(expr))
    end

    @testset "axis sum" begin
        tensor = EinExpr([:i, :j], Dict(:i => 2, :j => 3))
        expr = EinExpr((:i,), [tensor])

        @test all(splat(==), zip(expr.head, [:i]))
        @test expr.args == [tensor]

        @test all(splat(==), zip(head(expr), (:i,)))
        @test all(splat(==), zip(inds(expr), (:i, :j)))

        @test size(expr, :i) == 2
        @test size(expr, :j) == 3
        @test size(expr) == (2,)

        @test suminds(expr) == [:j]
        @test isempty(parsuminds(expr))
    end

    @testset "diagonal" begin
        tensor = EinExpr([:i, :i], Dict(:i => 2))
        expr = EinExpr((:i,), [tensor])

        @test all(splat(==), zip(expr.head, (:i,)))
        @test expr.args == [tensor]

        @test all(splat(==), zip(head(expr), (:i,)))
        @test all(splat(==), zip(inds(expr), head(expr)))

        @test size(expr, :i) == 2
        @test size(expr) == (2,)

        @test isempty(suminds(expr))
        @test isempty(parsuminds(expr))
    end

    @testset "trace" begin
        tensor = EinExpr([:i, :i], Dict(:i => 2))
        expr = EinExpr(Symbol[], [tensor])

        @test isempty(expr.head)
        @test expr.args == [tensor]

        @test isempty(head(expr))
        @test all(splat(==), zip(inds(expr), (:i,)))

        @test size(expr, :i) == 2
        @test size(expr) == ()

        @test suminds(expr) == [:i]
        @test isempty(parsuminds(expr))
    end

    @testset "outer product" begin
        tensors = [EinExpr((:i, :j), Dict(:i => 2, :j => 3)), EinExpr((:k, :l), Dict(:k => 4, :l => 5))]
        expr = EinExpr((:i, :j, :k, :l), tensors)

        @test all(splat(==), zip(expr.head, (:i, :j, :k, :l)))
        @test expr.args == tensors

        @test all(splat(==), zip(head(expr), mapreduce(collect ∘ inds, vcat, tensors)))
        @test all(splat(==), zip(inds(expr), head(expr)))
        @test ndims(expr) == 4

        for (i, d) in zip([:i, :j, :k, :l], [2, 3, 4, 5])
            @test size(expr, i) == d
        end
        @test size(expr) == (2, 3, 4, 5)

        @test isempty(suminds(expr))
        @test isempty(parsuminds(expr))
    end

    @testset "inner product" begin
        @testset "Vector" begin
            tensors = [EinExpr((:i,), Dict(:i => 2)), EinExpr((:i,), Dict(:i => 2))]
            expr = EinExpr(Symbol[], tensors)

            @test isempty(expr.head)
            @test expr.args == tensors

            @test isempty(head(expr))
            @test all(splat(==), zip(inds(expr), (:i,)))
            @test ndims(expr) == 0

            @test size(expr, :i) == 2
            @test size(expr) == ()

            @test suminds(expr) == [:i]
            @test parsuminds(expr) == [[:i]]
        end
        @testset "Matrix" begin
            tensors = [EinExpr((:i, :j), Dict(:i => 2, :j => 3)), EinExpr((:i, :j), Dict(:i => 2, :j => 3))]
            expr = EinExpr(Symbol[], tensors)

            @test isempty(expr.head)
            @test expr.args == tensors

            @test isempty(head(expr))
            @test all(splat(==), zip(inds(expr), (:i, :j)))
            @test ndims(expr) == 0

            @test size(expr, :i) == 2
            @test size(expr, :j) == 3
            @test size(expr) == ()

            @test issetequal(suminds(expr), [:i, :j])
            @test Set(Set.(parsuminds(expr))) == Set([Set([:i, :j])])
        end
    end

    @testset "matrix multiplication" begin
        tensors = [EinExpr((:i, :k), Dict(:i => 2, :k => 3)), EinExpr((:k, :j), Dict(:k => 3, :j => 4))]
        expr = EinExpr((:i, :j), tensors)

        @test all(splat(==), zip(expr.head, (:i, :j)))
        @test expr.args == tensors

        @test all(splat(==), zip(head(expr), (:i, :j)))
        @test all(splat(==), zip(inds(expr), (:i, :k, :j)))
        @test ndims(expr) == 2

        @test size(expr, :i) == 2
        @test size(expr, :j) == 4
        @test size(expr, :k) == 3
        @test size(expr) == (2, 4)

        @test suminds(expr) == [:k]
        @test parsuminds(expr) == [[:k]]
    end

    @testset "manual path" begin
        tensors = [
            EinExpr([:j, :b, :i, :h], Dict(i => 2 for i in [:j, :b, :i, :h])),
            EinExpr([:a, :c, :e, :f], Dict(i => 2 for i in [:a, :c, :e, :f])),
            EinExpr([:j], Dict(i => 2 for i in [:j])),
            EinExpr([:e, :a, :g], Dict(i => 2 for i in [:e, :a, :g])),
            EinExpr([:f, :b], Dict(i => 2 for i in [:f, :b])),
            EinExpr([:i, :h, :d], Dict(i => 2 for i in [:i, :h, :d])),
            EinExpr([:d, :g, :c], Dict(i => 2 for i in [:d, :g, :c])),
        ]

        path = EinExpr(Symbol[], tensors)
        @test issetequal(suminds(path), [:a, :b, :c, :d, :e, :f, :g, :h, :i, :j])
    end
end
