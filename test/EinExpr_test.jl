@testset "EinExpr" begin
    using Tensors
    using EinExprs: suminds
    using LinearAlgebra

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

        # TODO contract test?
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

        # TODO contract test?
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

        A = parent(contract(expr))
        B = PermutedDimsArray(
            reshape(
                kron(parent.(reverse(tensors))...),
                collect(Iterators.flatten(zip(size.(tensors)...)))...
            ), (1, 3, 2, 4))

        @test A ≈ B
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

            @test only(contract(expr)) ≈ dot(parent.(tensors)...)
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

            @test only(contract(expr)) ≈ dot(parent.(tensors)...)
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

        @test contract(expr) ≈ *(parent.(tensors)...)
    end

    @testset "manual path" begin
        sizes = Dict(
            :o => 2,
            :b => 2,
            :p => 2,
            :n => 2,
            :j => 2,
            :k => 2,
            :d => 2,
            :e => 2,
            :c => 2,
            :h => 2,
            :i => 2,
            :l => 2,
            :m => 2,
            :q => 2,
            :a => 2,
            :f => 2,
            :g => 2,
        )

        tensors = [
            Tensor(ones((sizes[i] for i in [:f, :l, :i])...), [:f, :l, :i]),
            Tensor(ones((sizes[i] for i in [:b, :e])...), [:b, :e]),
            Tensor(ones((sizes[i] for i in [:g, :n, :l, :a])...), [:g, :n, :l, :a]),
            Tensor(ones((sizes[i] for i in [:o, :i, :m, :c])...), [:o, :i, :m, :c]),
            Tensor(ones((sizes[i] for i in [:k, :d, :h, :a, :n, :j])...), [:k, :d, :h, :a, :n, :j]),
            Tensor(ones((sizes[i] for i in [:m, :f, :q])...), [:m, :f, :q]),
            Tensor(ones((sizes[i] for i in [:p, :k])...), [:p, :k]),
            Tensor(ones((sizes[i] for i in [:c, :e, :h])...), [:c, :e, :h]),
            Tensor(ones((sizes[i] for i in [:g, :q])...), [:g, :q]),
            Tensor(ones((sizes[i] for i in [:d, :b, :o])...), [:d, :b, :o]),
        ]

        expr = EinExpr(tensors, [:p, :j])

        @test sum(tensors..., inds=[:p, :j]) == expr

        for inds in [[:q], [:m], [:f, :i], [:g, :l], [:b], [:o], [:c, :e], [:n, :a, :d, :h], [:k]]
            foo = sum(expr, inds)
            @test foo == sum!(expr, inds)
        end
    end
end