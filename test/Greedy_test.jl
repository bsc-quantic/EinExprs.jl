@testset "Greedy" begin
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
    expr = sum(tensors)

    path = einexpr(Greedy(), SizedEinExpr(expr, sizedict))

    @test path isa SizedEinExpr

    @test mapreduce(flops, +, Branches(path)) == 100

    @test all(
        @compat(splat(issetequal)),
        zip(contractorder(path), [[:i, :h], [:j], [:a, :e], [:g, :c], [:f], [:b, :d]]),
    )

    @testset "example: let unchanged" begin
        sizedict = Dict(i => 2 for i in [:i, :j, :k, :l, :m])
        tensors = [EinExpr([:i, :j, :k]), EinExpr([:k, :l, :m])]
        expr = sum(tensors, skip = [:i, :j, :l, :m])
        sexpr = SizedEinExpr(expr, sizedict)

        path = einexpr(Greedy(), sexpr)

        @test suminds(path) == [:k]
    end

    @testset "hyperedges" begin
        sizedict = Dict(i => 2 for i in [:i, :j, :k, :l, :m, :β])
        a = EinExpr([:i, :β, :j])
        b = EinExpr([:k, :β])
        c = EinExpr([:β, :l, :m])

        path = einexpr(EinExprs.Greedy(), SizedEinExpr(sum([a, b, c], skip = [:β]), sizedict))
        @test all(∋(:β) ∘ head, branches(path))

        path = einexpr(EinExprs.Greedy(), SizedEinExpr(sum([a, b, c], skip = Symbol[]), sizedict))
        @test all(∋(:β) ∘ head, branches(path)[1:end-1])
        @test all(!∋(:β) ∘ head, branches(path)[end:end])
    end
end
