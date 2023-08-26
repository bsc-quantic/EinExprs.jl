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
        EinExpr([:f, :l, :i], filter(p -> p.first ∈ [:f, :l, :i], sizes)),
        EinExpr([:b, :e], filter(p -> p.first ∈ [:b, :e], sizes)),
        EinExpr([:g, :n, :l, :a], filter(p -> p.first ∈ [:g, :n, :l, :a], sizes)),
        EinExpr([:o, :i, :m, :c], filter(p -> p.first ∈ [:o, :i, :m, :c], sizes)),
        EinExpr([:k, :d, :h, :a, :n, :j], filter(p -> p.first ∈ [:k, :d, :h, :a, :n, :j], sizes)),
        EinExpr([:m, :f, :q], filter(p -> p.first ∈ [:m, :f, :q], sizes)),
        EinExpr([:p, :k], filter(p -> p.first ∈ [:p, :k], sizes)),
        EinExpr([:c, :e, :h], filter(p -> p.first ∈ [:c, :e, :h], sizes)),
        EinExpr([:g, :q], filter(p -> p.first ∈ [:g, :q], sizes)),
        EinExpr([:d, :b, :o], filter(p -> p.first ∈ [:d, :b, :o], sizes)),
    ]

    expr = einexpr(Exhaustive, EinExpr([:p, :j], tensors))
    @test expr isa EinExpr
    # TODO traverse through the tree and check everything is ok
    @test mapreduce(flops, +, EinExprs.Branches(expr)) == 48753
    # FIXME non-determinist behaviour on order
    @test issetequal(
        contractorder(expr),
        [[:q], [:m], [:f, :i], [:g, :l], [:b], [:o], [:c, :e], [:n, :a, :d, :h], [:k]],
    )
end
