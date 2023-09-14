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

        @test isempty(hyperinds(expr))
        @test isempty(suminds(expr))
        @test isempty(parsuminds(expr))

        @test select(expr, :i) == [tensor]
        @test select(expr, :j) == [tensor]

        @test neighbours(expr, :i) == [:j]
        @test neighbours(expr, :j) == [:i]
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

        @test isempty(hyperinds(expr))
        @test isempty(suminds(expr))
        @test isempty(parsuminds(expr))

        @test select(expr, :i) == [tensor]
        @test select(expr, :j) == [tensor]

        @test neighbours(expr, :i) == [:j]
        @test neighbours(expr, :j) == [:i]
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

        @test isempty(hyperinds(expr))
        @test suminds(expr) == [:j]
        @test isempty(parsuminds(expr))

        @test select(expr, :i) == [tensor]
        @test select(expr, :j) == [tensor]

        @test neighbours(expr, :i) == [:j]
        @test neighbours(expr, :j) == [:i]
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

        @test isempty(hyperinds(expr))
        @test isempty(suminds(expr))
        @test isempty(parsuminds(expr))

        @test select(expr, :i) == [tensor]

        @test isempty(neighbours(expr, :i))
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

        @test isempty(hyperinds(expr))
        @test suminds(expr) == [:i]
        @test isempty(parsuminds(expr))

        @test select(expr, :i) == [tensor]

        @test isempty(neighbours(expr, :i))
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

        @test isempty(hyperinds(expr))
        @test isempty(suminds(expr))
        @test isempty(parsuminds(expr))

        @test select(expr, :i) == select(expr, :j) == select(expr, [:i, :j]) == [tensors[1]]
        @test select(expr, :k) == select(expr, :l) == select(expr, [:k, :l]) == [tensors[2]]

        @test neighbours(expr, :i) == [:j]
        @test neighbours(expr, :j) == [:i]
        @test neighbours(expr, :k) == [:l]
        @test neighbours(expr, :l) == [:k]
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

            @test isempty(hyperinds(expr))
            @test suminds(expr) == [:i]
            @test parsuminds(expr) == [[:i]]

            @test select(expr, :i) == tensors

            @test isempty(neighbours(expr, :i))
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

            @test isempty(hyperinds(expr))
            @test issetequal(suminds(expr), [:i, :j])
            @test Set(Set.(parsuminds(expr))) == Set([Set([:i, :j])])

            @test select(expr, :i) == select(expr, :j) == select(expr, [:i, :j]) == tensors

            @test neighbours(expr, :i) == [:j]
            @test neighbours(expr, :j) == [:i]
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

        @test isempty(hyperinds(expr))
        @test suminds(expr) == [:k]
        @test parsuminds(expr) == [[:k]]

        @test select(expr, :i) == [tensors[1]]
        @test select(expr, :j) == [tensors[2]]
        @test select(expr, :k) == tensors

        @test neighbours(expr, :i) == neighbours(expr, :j) == [:k]
        @test neighbours(expr, :k) == [:i, :j]
    end

    @testset "hyperindex contraction" begin
        @testset "hyperindex is not summed" begin
            tensors = [
                EinExpr([:i, :β, :j], Dict(i => 2 for i in [:i, :β, :j])),
                EinExpr([:k, :β], Dict(i => 2 for i in [:k, :β])),
                EinExpr([:β, :l, :m], Dict(i => 2 for i in [:β, :l, :m])),
            ]

            expr = sum(tensors, skip = [:β])

            @test all(splat(==), zip(expr.head, (:i, :j, :k, :l, :m, :β)))
            @test expr.args == tensors

            @test issetequal(head(expr), (:i, :j, :k, :l, :m, :β))
            @test issetequal(inds(expr), (:i, :j, :k, :l, :m, :β))
            @test ndims(expr) == 6

            @test all(i -> size(expr, i) == 2, inds(expr))
            @test size(expr) == tuple(fill(2, 6)...)

            @test issetequal(hyperinds(expr), [:β])
            @test isempty(suminds(expr))
            @test_broken isempty(parsuminds(expr))

            @test select(expr, :i) == [tensors[1]]
            @test select(expr, :j) == [tensors[1]]
            @test select(expr, :k) == [tensors[2]]
            @test select(expr, :β) == tensors

            @test issetequal(neighbours(expr, :i), [:j, :β])
            @test issetequal(neighbours(expr, :β), [:i, :j, :k, :l, :m])
        end

        @testset "hyperindex is summed" begin
            tensors = [
                EinExpr([:i, :β, :j], Dict(i => 2 for i in [:i, :β, :j])),
                EinExpr([:k, :β], Dict(i => 2 for i in [:k, :β])),
                EinExpr([:β, :l, :m], Dict(i => 2 for i in [:β, :l, :m])),
            ]

            expr = sum(tensors)

            @test all(splat(==), zip(expr.head, (:i, :j, :k, :l, :m)))
            @test expr.args == tensors

            @test issetequal(head(expr), (:i, :j, :k, :l, :m))
            @test issetequal(inds(expr), (:i, :j, :k, :l, :m, :β))
            @test ndims(expr) == 5

            @test all(i -> size(expr, i) == 2, inds(expr))
            @test size(expr) == tuple(fill(2, 5)...)

            @test issetequal(hyperinds(expr), [:β])
            @test issetequal(suminds(expr), [:β])
            @test issetequal(parsuminds(expr), [[:β]])

            @test select(expr, :i) == [tensors[1]]
            @test select(expr, :j) == [tensors[1]]
            @test select(expr, :k) == [tensors[2]]
            @test select(expr, :β) == tensors

            @test issetequal(neighbours(expr, :i), [:j, :β])
            @test issetequal(neighbours(expr, :β), [:i, :j, :k, :l, :m])
        end
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
        @test isempty(hyperinds(path))
        @test issetequal(suminds(path), [:a, :b, :c, :d, :e, :f, :g, :h, :i, :j])
    end
end
