@testset "Exhaustive" begin
    sizedict = Dict(i => 2 for i in [:a, :b, :c, :d, :e, :f, :g, :h, :i, :j])
    tensors = [
        EinExpr([:j, :b, :i, :h]),
        EinExpr([:a, :c, :e, :f]),
        EinExpr([:j]),
        EinExpr([:e, :a, :g]),
        EinExpr([:f, :b]),
        EinExpr([:i, :h, :d]),
        EinExpr([:d, :g, :c]),
    ]
    expr = EinExpr(Symbol[], tensors)
    sexpr = SizedEinExpr(expr, sizedict)

    path = einexpr(Exhaustive, sexpr)

    @test path isa SizedEinExpr

    @test mapreduce(flops, +, Branches(path)) == 92

    @test all(splat(issetequal), zip(contractorder(path), [[:a, :e], [:c, :g], [:f], [:j], [:i, :h], [:d, :b]]))

    @testset "hyperedges" begin
        sizedict = Dict(i => 2 for i in [:i, :j, :k, :l, :m, :β])
        a = EinExpr([:i, :β, :j])
        b = EinExpr([:k, :β])
        c = EinExpr([:β, :l, :m])

        path = einexpr(EinExprs.Exhaustive(), SizedEinExpr(sum([a, b, c], skip = [:β]), sizedict))
        @test all(∋(:β) ∘ head, branches(path))

        path = einexpr(EinExprs.Exhaustive(), SizedEinExpr(sum([a, b, c], skip = Symbol[]), sizedict))
        @test all(∋(:β) ∘ head, branches(path)[1:end-1])
        @test all(!∋(:β) ∘ head, branches(path)[end:end])
    end
end
