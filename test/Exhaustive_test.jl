@testset "Exhaustive" begin
    sizes = Dict(
        :o => 3,
        :b => 7,
        :p => 6,
        :n => 7,
        :j => 9,
        :k => 8,
        :d => 4,
        :e => 2,
        :c => 2,
        :h => 5,
        :i => 5,
        :l => 10,
        :m => 7,
        :q => 5,
        :a => 3,
        :f => 7,
        :g => 3,
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

    expr = einexpr(Exhaustive, EinExpr(tensors, [:p, :j]))
    @test expr isa EinExpr
    # TODO traverse through the tree and check everything is ok
    @test mapreduce(flops, +, expr) == 48753
    # FIXME non-determinist behaviour on order
    @test issetequal(path(expr), [[:q], [:m], [:f, :i], [:g, :l], [:b], [:o], [:c, :e], [:n, :a, :d, :h], [:k]])
end
