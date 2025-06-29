@testset "EinExpr" begin
    using LinearAlgebra

    @testset "identity" begin
        tensor = EinExpr([:i, :j])
        expr = EinExpr([:i, :j], [tensor])

        @test expr.head == head(tensor)
        @test expr.args == [tensor]

        @test head(expr) == head(tensor)
        @test ndims(expr) == 2

        @test isempty(hyperinds(expr))
        @test isempty(suminds(expr))
        @test isempty(parsuminds(expr))

        @test select(expr, :i) == [expr, tensor]
        @test select(expr, :j) == [expr, tensor]

        @test neighbours(expr, :i) == [:j]
        @test neighbours(expr, :j) == [:i]
    end

    @testset "transpose" begin
        tensor = EinExpr([:i, :j])
        expr = EinExpr([:j, :i], [tensor])

        @test expr.head == reverse(inds(tensor))
        @test expr.args == [tensor]

        @test head(expr) == reverse(inds(tensor))
        @test ndims(expr) == 2

        @test isempty(hyperinds(expr))
        @test isempty(suminds(expr))
        @test isempty(parsuminds(expr))

        @test select(expr, :i) == [expr, tensor]
        @test select(expr, :j) == [expr, tensor]

        @test neighbours(expr, :i) == [:j]
        @test neighbours(expr, :j) == [:i]
    end

    @testset "axis sum" begin
        tensor = EinExpr([:i, :j])
        expr = EinExpr([:i], [tensor])

        @test all(splat(==)), zip(expr.head, [:i])
        @test expr.args == [tensor]

        @test all(splat(==)), zip(head(expr), [:i])
        @test all(splat(==)), zip(inds(expr), [:i, :j])

        @test isempty(hyperinds(expr))
        @test suminds(expr) == [:j]
        @test isempty(parsuminds(expr))

        @test select(expr, :i) == [expr, tensor]
        @test select(expr, :j) == [tensor]

        @test neighbours(expr, :i) == [:j]
        @test neighbours(expr, :j) == [:i]
    end

    @testset "diagonal" begin
        tensor = EinExpr([:i, :i])
        expr = EinExpr([:i], [tensor])

        @test all(splat(==)), zip(expr.head, [:i])
        @test expr.args == [tensor]

        @test all(splat(==)), zip(head(expr), [:i])
        @test all(splat(==)), zip(inds(expr), head(expr))

        @test isempty(hyperinds(expr))
        @test isempty(suminds(expr))
        @test isempty(parsuminds(expr))

        @test select(expr, :i) == [expr, tensor]

        @test isempty(neighbours(expr, :i))
    end

    @testset "trace" begin
        tensor = EinExpr([:i, :i])
        expr = EinExpr(Symbol[], [tensor])

        @test isempty(expr.head)
        @test expr.args == [tensor]

        @test isempty(head(expr))
        @test all(splat(==)), zip(inds(expr), [:i])

        @test isempty(hyperinds(expr))
        @test suminds(expr) == [:i]
        @test isempty(parsuminds(expr))

        @test select(expr, :i) == [tensor]

        @test isempty(neighbours(expr, :i))
    end

    @testset "outer product" begin
        tensors = [EinExpr([:i, :j]), EinExpr([:k, :l])]
        expr = EinExpr([:i, :j, :k, :l], tensors)

        @test all(splat(==)), zip(expr.head, [:i, :j, :k, :l])
        @test expr.args == tensors

        @test all(splat(==)), zip(head(expr), mapreduce(collect ∘ inds, vcat, tensors))
        @test all(splat(==)), zip(inds(expr), head(expr))
        @test ndims(expr) == 4

        @test isempty(hyperinds(expr))
        @test isempty(suminds(expr))
        @test isempty(parsuminds(expr))

        @test select(expr, :i) == select(expr, :j) == select(expr, [:i, :j]) == [expr, tensors[1]]
        @test select(expr, :k) == select(expr, :l) == select(expr, [:k, :l]) == [expr, tensors[2]]

        @test neighbours(expr, :i) == [:j]
        @test neighbours(expr, :j) == [:i]
        @test neighbours(expr, :k) == [:l]
        @test neighbours(expr, :l) == [:k]
    end

    @testset "inner product" begin
        @testset "Vector" begin
            tensors = [EinExpr([:i]), EinExpr([:i])]
            expr = EinExpr(Symbol[], tensors)

            @test isempty(expr.head)
            @test expr.args == tensors

            @test isempty(head(expr))
            @test all(splat(==)), zip(inds(expr), [:i])
            @test ndims(expr) == 0

            @test isempty(hyperinds(expr))
            @test suminds(expr) == [:i]
            @test parsuminds(expr) == [[:i]]

            @test select(expr, :i) == tensors

            @test isempty(neighbours(expr, :i))
        end
        @testset "Matrix" begin
            tensors = [EinExpr([:i, :j]), EinExpr([:i, :j])]
            expr = EinExpr(Symbol[], tensors)

            @test isempty(expr.head)
            @test expr.args == tensors

            @test isempty(head(expr))
            @test all(splat(==)), zip(inds(expr), [:i, :j])
            @test ndims(expr) == 0

            @test isempty(hyperinds(expr))
            @test issetequal(suminds(expr), [:i, :j])
            @test Set(Set.(parsuminds(expr))) == Set([Set([:i, :j])])

            @test select(expr, :i) == select(expr, :j) == select(expr, [:i, :j]) == tensors

            @test neighbours(expr, :i) == [:j]
            @test neighbours(expr, :j) == [:i]
        end
    end

    @testset "matrix multiplication" begin
        tensors = [EinExpr([:i, :k]), EinExpr([:k, :j])]
        expr = EinExpr([:i, :j], tensors)

        @test all(splat(==)), zip(expr.head, [:i, :j])
        @test expr.args == tensors

        @test all(splat(==)), zip(head(expr), [:i, :j])
        @test all(splat(==)), zip(inds(expr), [:i, :k, :j])
        @test ndims(expr) == 2

        @test isempty(hyperinds(expr))
        @test suminds(expr) == [:k]
        @test parsuminds(expr) == [[:k]]

        @test select(expr, :i) == [expr, tensors[1]]
        @test select(expr, :j) == [expr, tensors[2]]
        @test select(expr, :k) == tensors

        @test neighbours(expr, :i) == neighbours(expr, :j) == [:k]
        @test neighbours(expr, :k) == [:i, :j]
    end

    @testset "hyperindex contraction" begin
        @testset "hyperindex is not summed" begin
            tensors = [EinExpr([:i, :β, :j]), EinExpr([:k, :β]), EinExpr([:β, :l, :m])]
            expr = sum(tensors, skip = [:β])

            @test issetequal(head(expr), (:i, :j, :k, :l, :m, :β))
            @test issetequal(inds(expr), (:i, :j, :k, :l, :m, :β))
            @test ndims(expr) == 6

            @test issetequal(hyperinds(expr), [:β])
            @test isempty(suminds(expr))
            @test_broken isempty(parsuminds(expr))

            @test select(expr, :i) == [expr, tensors[1]]
            @test select(expr, :j) == [expr, tensors[1]]
            @test select(expr, :k) == [expr, tensors[2]]
            @test select(expr, :β) == [expr, tensors...]

            @test issetequal(neighbours(expr, :i), [:j, :β])
            @test issetequal(neighbours(expr, :β), [:i, :j, :k, :l, :m])
        end

        @testset "hyperindex is summed" begin
            tensors = [EinExpr([:i, :β, :j]), EinExpr([:k, :β]), EinExpr([:β, :l, :m])]
            expr = sum(tensors)

            @test all(splat(==)), zip(expr.head, [:i, :j, :k, :l, :m])
            @test expr.args == tensors

            @test issetequal(head(expr), [:i, :j, :k, :l, :m])
            @test issetequal(inds(expr), (:i, :j, :k, :l, :m, :β))
            @test ndims(expr) == 5

            @test issetequal(hyperinds(expr), [:β])
            @test issetequal(suminds(expr), [:β])
            @test issetequal(parsuminds(expr), [[:β]])

            @test select(expr, :i) == [expr, tensors[1]]
            @test select(expr, :j) == [expr, tensors[1]]
            @test select(expr, :k) == [expr, tensors[2]]
            @test select(expr, :β) == tensors

            @test issetequal(neighbours(expr, :i), [:j, :β])
            @test issetequal(neighbours(expr, :β), [:i, :j, :k, :l, :m])
        end
    end

    @testset "manual path" begin
        tensors = [
            EinExpr([:j, :b, :i, :h]),
            EinExpr([:a, :c, :e, :f]),
            EinExpr([:j]),
            EinExpr([:e, :a, :g]),
            EinExpr([:f, :b]),
            EinExpr([:i, :h, :d]),
            EinExpr([:d, :g, :c]),
        ]

        path = EinExpr(Symbol[], tensors)
        @test isempty(hyperinds(path))
        @test issetequal(suminds(path), [:a, :b, :c, :d, :e, :f, :g, :h, :i, :j])
    end
end
